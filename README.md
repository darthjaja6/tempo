<h1 align="center">Tempo</h1>

<h3 align="center">
Fast serving engine for System 1 + System 2 robot policies
</h3>

<p align="center">
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-green.svg" alt="License"></a>
</p>

<p align="center">
| <a href="#getting-started"><b>Getting Started</b></a> | <a href="#supported-models"><b>Models</b></a> | <a href="./docs/architecture.md"><b>Architecture</b></a> | <a href="#benchmarks"><b>Benchmarks</b></a> | <a href="./examples"><b>Examples</b></a> |
</p>

---

**Tempo** is a fast serving engine for dual-system robot policies, supporting mainstream VLAs (`pi05`, `GR00T`, `SmolVLA`) and custom System 1 + System 2 models.

On [`LIBERO-Dynamic`](https://github.com/renshaojie233/LIBERO-Dynamic) (`1x RTX 5090`), Tempo **beats latency-retrained `VLASH` without retraining**:

| 1 Robot ([`pi05_libero`](https://github.com/Physical-Intelligence/openpi), Zero Retraining) | Retrained | Success Rate | Mean Latency |
| :--- | :---: | :---: | :---: |
| `LeRobot PyTorch` (Sync / Async) | No | 43.1% / 48.4% | 147 ms |
| `VLASH` | **Yes** | 55.4% | 104 ms |
| `OpenPI JAX Async` | No | 57.8% | 60 ms |
| **Tempo** | **No** | **60.9%** | **39 ms** |

On the SOTA [`pi05/mixed_ft3000`](https://huggingface.co/rsj2003/libero-dynamic-paper-models/tree/main/pi05/mixed_ft3000) checkpoint, Tempo matches native `OpenPI JAX` serving on a single robot (as both easily keep up with the benchmark's `20 Hz` control rate)—yet once scaled to **8 concurrent robots** on one `RTX 5090`, `OpenPI JAX` falls behind and collapses while Tempo holds firm:

| 8 Concurrent Robots ([`pi05/mixed_ft3000`](https://huggingface.co/rsj2003/libero-dynamic-paper-models/tree/main/pi05/mixed_ft3000)) | Success Rate | Mean Latency |
| :--- | :---: | :---: |
| `OpenPI JAX Async` | 16.1% | 280 ms |
| **Tempo** | **52.3% (3.25x)** | **27 ms** |

Tempo is fast with:
- [Lock-free paged KV and latent cache](./docs/architecture.md#1-lock-free-paged-state-cache) across Slow and Fast streams
- Paged Split-K cross-attention and 3D relative-time RoPE
- C++ deadline batching and latest-frame sensor coalescing
- [Piecewise CUDA Graph slicing (`<= 2 ms`)](./docs/architecture.md#2-sub-2ms-micro-slicing--deadline-batching), `torch.compile`, and `FP8` quantization
- [Real-Time Chunking (RTC)](./docs/architecture.md#3-real-time-chunking-rtc--horizon-ring) with frozen-prefix splicing and overrun blending

Tempo is easy to use with:
- Native Hugging Face checkpoint loading (`torch` + `safetensors` + `tokenizers` only)
- Standard PyTorch `nn.Module` interfaces (`SlowBackbone` and `FastHead`) for custom policies
- [Colocated, disaggregated Slow/Fast, and pipeline-parallel](./docs/architecture.md#4-slowfast-disaggregation--edge-cloud-split-serving) multi-GPU placement
- Edge-cloud split serving over TCP (`RemoteSlowServer` / `RemoteSlowClient`)

For system design and architecture details, see [docs/architecture.md](./docs/architecture.md).

## Getting Started

Install on Linux (`Python >= 3.10`, `PyTorch >= 2.5`, `CUDA >= 12.0`, NVIDIA `SM80+`):

```bash
git clone https://github.com/darthjaja6/tempo.git && cd tempo
pip install -e . --no-build-isolation
```

Serve a pretrained policy in a real-time control loop:

```python
import time
import torch
from tempo import Engine

engine = Engine("lerobot/smolvla_base")
engine.set_prompt("pick up the cup")

cams = torch.randint(0, 256, engine.slow_obs_shape, dtype=torch.uint8)  # simulated RGB cameras
engine.push_slow(cams)
engine.wait_ready()

for step in range(500):               # 1 s real-time loop at 500 Hz (2 ms/step)
    if step % 250 == 0:               # 2 Hz camera frame -> System 2 (Slow)
        engine.push_slow(cams)
    if step % 25 == 0:                # 20 Hz chunk refresh -> System 1 (Fast)
        engine.push_fast()
    actions, *_ = engine.sample_actions(cpu=True)
    time.sleep(0.002)
print(actions.shape)                  # torch.Size([1, 6])
```

- Custom `SlowBackbone` + `FastHead` policy: [examples/custom_model.py](./examples/custom_model.py)
- Modular `(Slow × Fast)` pairing & `@engine.stage`: [examples/quickstart.py](./examples/quickstart.py)
- Multi-GPU disaggregation (`slow_devices=[0], fast_devices=[1, 2, 3]`) & edge-cloud serving: [examples/multi_gpu_serving.py](./examples/multi_gpu_serving.py)

## Supported Models

| Model | System 2 (`SlowBackbone`) | System 1 (`FastHead`) |
| :--- | :--- | :--- |
| `lerobot/pi05_base` (+ fine-tuned variants) | PaliGemma-2B | Gemma Action Expert (10-step Flow) |
| `nvidia/GR00T-N1.5-3B` | Eagle2-VLM (SigLIP + Qwen3-1.7B) | Cross-Attention DiT (4-step Flow) |
| `lerobot/smolvla_base` (+ fine-tuned variants) | SmolVLM2 (SigLIP + Llama-16L) | SmolLM2 Action Expert (10-step Flow) |

To serve a custom System 1 + System 2 policy, subclass `SlowBackbone` and `FastHead` connected by a `TemporalChannel` (`mode="hidden"` or `mode="kv"`), register a Hugging Face loader via `@ModelRegistry.register`, or wire custom stages with `@engine.stage`—see [examples/custom_model.py](./examples/custom_model.py).

## Benchmarks

Across mainstream pure-PyTorch VLAs, Tempo cuts multi-stream tail latency by orders of magnitude over multi-stream PyTorch on the same hardware, while Slow/Fast multi-GPU disaggregation and pipeline parallelism cut 3B+ VLA cluster VRAM by **45–59%**:

| Model | GPUs | Concurrent Robots | Tail Latency (`P99`: Tempo vs. PyTorch) |
| :--- | :--- | :---: | :---: |
| **`pi05` (3.3B)** | `4x H100 SXM` (`Colocated`) | 4 | **26.8 ms** vs. 2.94 s (**109.7x lower**) |
| **`GR00T-N1.5` (3B)** | `4x H100 SXM` (`1S : 3F`) | 48 | **13.3 ms** vs. 447.9 ms (**33.6x lower**) |
| **`SmolVLA` (450M)** | `2x RTX PRO 6000` (`1S : 1F`) | 16 | **24.4 ms** vs. 294.5 ms (**12.1x lower**) |

Full [`LIBERO-Dynamic`](https://github.com/renshaojie233/LIBERO-Dynamic) and `LIBERO` results, Slow/Fast disaggregation VRAM matrices, and reproduction commands → [benchmarks/README.md](./benchmarks/README.md).

## License

Apache-2.0 — see [LICENSE](./LICENSE).
