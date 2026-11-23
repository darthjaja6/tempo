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

**Tempo** is a fast inference and serving engine for `System 1 + System 2` robot policies.

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

Serve a pretrained policy:

```python
import torch
from tempo import Engine

engine = Engine("lerobot/smolvla_base", prompt="pick up the cup")
engine.push_slow(torch.zeros(engine.slow_obs_shape, dtype=torch.uint8))
engine.wait_ready()
actions, *_ = engine.sample_actions(cpu=True)
```

- Custom `SlowBackbone` / `FastHead` policies & `@engine.stage`: [examples/quickstart.py](./examples/quickstart.py)
- Multi-GPU disaggregation (`slow_devices=[0], fast_devices=[1, 2, 3]`) & edge-cloud serving: [examples/multi_gpu_serving.py](./examples/multi_gpu_serving.py)

## Supported Models

| Model | System 2 (`SlowBackbone`) | System 1 (`FastHead`) |
| :--- | :--- | :--- |
| `lerobot/pi05_base` (+ fine-tuned variants) | PaliGemma-2B | Gemma Action Expert (10-step Flow) |
| `nvidia/GR00T-N1.5-3B` | Eagle2-VLM (SigLIP + Qwen3-1.7B) | Cross-Attention DiT (4-step Flow) |
| `lerobot/smolvla_base` (+ fine-tuned variants) | SmolVLM2 (SigLIP + Llama-16L) | SmolLM2 Action Expert (10-step Flow) |

## Benchmarks

Evaluated against a multi-stream PyTorch `SDPA` baseline ([full tables & reproduction →](./benchmarks/README.md)):

| Model | Hardware & Placement | Max Streams at SLO | Chunk `P99` (Tempo vs. PyTorch) | Zero-Underrun Capacity |
| :--- | :--- | :---: | :---: | :---: |
| `nvidia/GR00T-N1.5-3B` | `4x H100 SXM` (`1S : 3F`) | **12** | **9.56 ms** vs. 393.8 ms (**41.2x**) | 48 streams |
| `lerobot/pi05_base` | `4x H100 SXM` (`Colocated`) | **4** | **26.82 ms** vs. 2,943.4 ms (**109.7x**) | 4 streams |
| `lerobot/pi05_base` | `1x RTX 4090` (`Colocated`) | **1** | **23.90 ms** vs. 271.6 ms (**11.4x**) | 2 streams |
| `lerobot/smolvla_base` | `1x RTX 4090` (`Colocated`) | **8** | **24.36 ms** vs. 136.1 ms (**5.6x**) | 12 streams |

On real-time moving-object manipulation ([robocurve/inspect-robots](https://github.com/robocurve/inspect-robots) × [LIBERO-Dynamic](https://github.com/renshaojie233/LIBERO-Dynamic), `0.03–0.09 m/s`), Tempo reaches **50.0%** task success with `0` underruns vs. **33.3%** for synchronous chunking and **11.1%** for async PyTorch at `N = 2` concurrent streams ([details](./benchmarks/README.md#4-real-time-moving-object-manipulation-inspect-robots--libero-dynamic)).

## License

Apache-2.0 — see [LICENSE](./LICENSE).
