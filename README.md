<h1 align="center">Tempo</h1>

<h3 align="center">
Fast serving engine for System 1 + System 2 robot policies
</h3>

<p align="center">
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-green.svg" alt="License"></a>
</p>

<p align="center">
| <a href="#getting-started"><b>Getting Started</b></a> | <a href="#supported-models"><b>Models</b></a> | <a href="#benchmarks"><b>Benchmarks</b></a> | <a href="./examples"><b>Examples</b></a> |
</p>

---

**Tempo** is a fast serving engine for dual-system robot policies, supporting mainstream VLAs (`pi05`, `GR00T`, `SmolVLA`) and custom System 1 + System 2 models.

On a **single `RTX 4090`**, serving the same pretrained checkpoint with Tempo finishes tasks faster and directly improves success rates on dynamic manipulation benchmarks:

**[LIBERO-Dynamic](https://github.com/renshaojie233/LIBERO-Dynamic) (`pi05` on `1x RTX 4090`, `20 Hz`, `360` episodes)**

| Deployment | Success Rate | Completion Time | Robot Stall |
| :--- | :---: | :---: | :---: |
| Synchronous | 31.7% | 17.82 s | 40.0% |
| `torch.compile` Sync | 30.6% | 18.66 s | 35.8% |
| Python Async | 7.2% | 8.37 s | 1.6% |
| **Tempo** | **39.2% (+7.5–8.6 pts)** | **10.35 s (1.7–1.8x faster)** | **0.3% (>100x lower)** |

When scaling across **multiple GPUs** to serve concurrent robots or large-scale parallel evaluation, Tempo cuts tail latency by orders of magnitude over multi-stream PyTorch on the same hardware:

**Multi-GPU Serving (Tempo vs. Multi-Stream PyTorch)**

| Model | GPUs | Concurrent Robots | Tail Latency (`P99`) |
| :--- | :--- | :---: | :---: |
| **`pi05` (3.3B)** | `4x H100 SXM` | 4 | **26.8 ms** vs. 2.94 s (**109.7x lower**) |
| **`GR00T-N1.5` (3B)** | `4x H100 SXM` | 48 | **13.3 ms** vs. 447.9 ms (**33.6x lower**) |
| **`SmolVLA` (450M)** | `2x RTX PRO 6000` | 16 | **24.4 ms** vs. 294.5 ms (**12.1x lower**) |

Tempo is fast with:
- Lock-free paged KV and latent cache across Slow and Fast streams
- Paged Split-K cross-attention and 3D relative-time RoPE
- C++ deadline batching and latest-frame sensor coalescing
- Piecewise CUDA Graph slicing (`<= 2 ms`), `torch.compile`, and `FP8` quantization
- Real-Time Chunking (RTC) with frozen-prefix splicing and overrun blending

Tempo is easy to use with:
- Native Hugging Face checkpoint loading (`torch` + `safetensors` + `tokenizers` only)
- Standard PyTorch `nn.Module` interfaces (`SlowBackbone` and `FastHead`) for custom policies
- Colocated, disaggregated Slow/Fast, and pipeline-parallel multi-GPU placement
- Edge-cloud split serving over TCP (`RemoteSlowServer` / `RemoteSlowClient`)

<p align="center">
  <img src="./architecture.svg" width="100%" alt="Tempo Architecture">
</p>

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

- Modular `(Slow × Fast)` pairing & `@engine.stage`: [examples/quickstart.py](./examples/quickstart.py)
- Multi-GPU disaggregation (`slow_devices=[0], fast_devices=[1, 2, 3]`) & edge-cloud serving: [examples/multi_gpu_serving.py](./examples/multi_gpu_serving.py)

## Supported Models

| Model | System 2 (`SlowBackbone`) | System 1 (`FastHead`) |
| :--- | :--- | :--- |
| `lerobot/pi05_base` (+ fine-tuned variants) | PaliGemma-2B | Gemma Action Expert (10-step Flow) |
| `nvidia/GR00T-N1.5-3B` | Eagle2-VLM (SigLIP + Qwen3-1.7B) | Cross-Attention DiT (4-step Flow) |
| `lerobot/smolvla_base` (+ fine-tuned variants) | SmolVLM2 (SigLIP + Llama-16L) | SmolLM2 Action Expert (10-step Flow) |

### Adding Custom System 1 + System 2 Models

Supporting a new dual-system architecture only requires subclassing `SlowBackbone` and `FastHead` connected by a `TemporalChannel` (`mode="hidden"` for latent features or `mode="kv"` for layer-wise KV cross-attention)—Tempo automatically handles piecewise CUDA Graph slicing, lock-free paging, `torch.compile`, and Real-Time Chunking (RTC):

```python
import torch
import torch.nn as nn
from tempo import Engine, FastHead, SlowBackbone, TemporalChannel, mark_sliceable

class MySlowVLM(SlowBackbone):
    def __init__(self):
        super().__init__()
        self.channel = TemporalChannel(hidden_dim=256, seq_len=64, mode="hidden")
        self.obs_shape = (64, 256)
        self.encoder = nn.Sequential(mark_sliceable(nn.Linear(256, 256)), nn.SiLU())

    def forward(self, obs, ctx):
        ctx.write_hidden_state(self.encoder(obs))  # or ctx.write_layer_kv(...) in mode="kv"

class MyFastPolicy(FastHead):
    action_dim, chunk_size, obs_shape = 7, 50, (14,)

    def __init__(self):
        super().__init__()
        self.head = nn.Linear(14 + 256, 50 * 7)

    def forward(self, fast_obs, state, frozen_prefix):
        hidden, _ = state.read_hidden()            # or state.cross_attend(q, layer_idx)
        pred = self.head(torch.cat([fast_obs, hidden.mean(dim=1)], dim=-1)).view(-1, 50, 7)
        p = frozen_prefix.size(1)
        return torch.cat([state.splice_prefix(pred[:, :p], frozen_prefix), pred[:, p:]], dim=1)

engine = Engine(slow=MySlowVLM(), fast=MyFastPolicy())
engine.push_slow(torch.randn(64, 256))
engine.wait_ready(torch.randn(14))
print(engine.sample_actions(cpu=True)[0].shape)  # torch.Size([1, 7])
```

You can also register Hugging Face checkpoint loaders via `@ModelRegistry.register("my_vla")` or define custom execution graphs with `@engine.stage`. Walk through runnable end-to-end scripts in [`examples/`](./examples).

## Benchmarks

### Slow/Fast Multi-GPU Disaggregation

Decoupling `SlowBackbone` and `FastHead` onto dedicated GPUs isolates Fast control loops from heavy VLM prefill—cutting `GR00T-N1.5` `Chunk P99` on `H100 SXM` from `22.1 ms` (`Colocated`) to **`9.6 ms`** (`1S : 3F`, **2.3x lower**) while raising capacity from `32` to **`48` concurrent robots**. On large 3B+ VLAs (`lerobot/pi05_base`), sharing or pipeline-parallelizing a single Slow VLM across Fast GPUs cuts total cluster VRAM by **45–59%**:

| `pi05` Placement (`4` GPUs, `4` Robots) | `4x H100 SXM` Chunk `P50 / P99` | `4x H100` Peak / Total VRAM | `4x RTX PRO 6000` Chunk `P50 / P99` | `4x RTX PRO 6000` Peak / Total VRAM |
| :--- | :---: | :---: | :---: | :---: |
| `4x Colocated` (`slow=[0..3], fast=[0..3]`) | **20.55 / 26.82 ms** | 5.89 / 23.55 GB | **22.12 / 28.76 ms** | 5.84 / 23.35 GB |
| `1S : 3F` (`slow=[0], fast=[1,2,3]`) | 20.57 / 36.83 ms | 6.42 / **12.68 GB (-46%)** | 22.18 / 39.74 ms | 5.72 / **9.76 GB (-58%)** |
| `2-Stage PP + 2F` (`slow=[[0,1]], fast=[2,3]`) | 20.40 / 36.40 ms | **4.33** / 12.95 GB (-45%) | 22.18 / 39.68 ms | **3.53** / **9.50 GB (-59%)** |

Full `1x`–`4x` GPU serving matrices, [`inspect-robots`](https://github.com/robocurve/inspect-robots) × `LIBERO` (`400` episodes) / [`LIBERO-Dynamic`](https://github.com/renshaojie233/LIBERO-Dynamic) (`360` episodes) evaluations, and reproduction commands → [benchmarks/README.md](./benchmarks/README.md).

## License

Apache-2.0 — see [LICENSE](./LICENSE).
