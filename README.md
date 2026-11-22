<h1 align="center">Tempo</h1>

<h3 align="center">
Fast serving engine for System 1 + System 2 robot policies
</h3>

<p align="center">
| <a href="#install"><b>Install</b></a> | <a href="#quick-start"><b>Quick Start</b></a> | <a href="#models-and-hardware"><b>Models &amp; Hardware</b></a> | <a href="#benchmarks"><b>Benchmarks</b></a> | <a href="./examples"><b>Examples</b></a> | <a href="./LICENSE"><b>License</b></a> |
</p>

---

**Tempo** is a fast serving engine for `System 1 + System 2` robot policies such as $\pi_{0.5}$, GR00T-N1.5, SmolVLA, and Helix, supporting single-GPU, multi-GPU, and edge-cloud deployments.

<p align="center">
  <img src="./architecture.svg" width="100%" alt="Tempo Architecture">
</p>

Evaluated against a multi-stream PyTorch `SDPA` baseline ([details →](#benchmarks)):

| Model | Hardware &amp; Placement | Max Streams at SLO<br>(Tempo vs. PyTorch) | Chunk Latency `P99`<br>(Tempo vs. PyTorch) | Zero-Underrun Capacity /<br>Closed-Loop Success |
| :--- | :--- | :---: | :---: | :---: |
| **`nvidia/GR00T-N1.5-3B`** | **`4x H100 SXM`** (`1S : 3F`) | **`12`** vs. `0` | **`9.56 ms`** vs. `393.8 ms` (**`41.2x`**) | **`48` streams** (`13.3 ms` P99) / **`100.0%`** vs. `0.0%` |
| **`nvidia/GR00T-N1.5-3B`** | **`1x H100 SXM`** (`Colocated`) | **`8`** vs. `0` | **`22.12 ms`** vs. `150.6 ms` (**`6.8x`**) | **`32` streams** (`24.6 ms` P99) / **`100.0%`** vs. `0.0%` |
| **`lerobot/pi05_base`** | **`4x H100 SXM`** (`Colocated`) | **`4`** vs. `0` | **`26.82 ms`** vs. `2,943.4 ms` (**`109.7x`**) | `4` streams / **`100.0%`** vs. `0.0%` |
| **`lerobot/pi05_base`** | **`2x H100 SXM`** (`1S : 1F`) | **`1`** vs. `0` | **`20.76 ms`** vs. `423.6 ms` (**`20.4x`**) | `2` streams / **`100.0%`** vs. `0.0%` |
| **`lerobot/pi05_base`** | **`1x RTX 4090`** (`Colocated`) | **`1`** vs. `0` | **`23.90 ms`** vs. `271.6 ms` (**`11.4x`**) | `2` streams (`5.55 GB` VRAM, `0` vs. `2,952` underruns) |
| **`lerobot/smolvla_base`** | **`1x RTX 4090`** (`Colocated`) | **`8`** vs. `0` | **`24.36 ms`** vs. `136.1 ms` (**`5.6x`**) | **`12` streams** (`0.98 GB` VRAM, `0` vs. `18,974` underruns) |
| **`lerobot/smolvla_base`** | **`2x RTX PRO 6000`** (`1S : 1F`) | **`12`** vs. `0` | **`22.98 ms`** vs. `292.1 ms` (**`12.7x`**) | **`16` streams** (`1.07 GB` VRAM, `0` vs. `15,152` underruns) |

### Key Features

- **Lock-free paged KV and latent cache** across asynchronous Slow (System 2) and Fast (System 1) streams
- **3D relative-time RoPE** and Split-K cross-attention for asynchronous sensor timestamps
- **C++ scheduler** with deadline batching and latest-frame sensor coalescing
- **Piecewise CUDA Graph slicing** (`<= 2 ms`) so Slow VLM prefill does not stall Fast action generation on a single GPU
- **Automatic `torch.compile`, bucketed CUDA Graphs, and `FP8`** for standard PyTorch `nn.Module` policies
- **Real-Time Chunking (RTC)** ring buffer with frozen-prefix splicing and smooth overrun blending
- **Flexible multi-GPU and edge-cloud placement**: colocated, disaggregated Slow/Fast GPUs, Slow pipeline parallelism, and TCP edge-cloud split serving

## Models and Hardware

### Built-In Models

| Checkpoint | System 2 (`SlowBackbone`) | System 1 (`FastHead`) | Params (Active / Total) |
| :--- | :--- | :--- | :---: |
| **`lerobot/pi05_base`** | PaliGemma-2B (`18L`) | Gemma Action Expert (`18L`, 10-step Flow) | `2.83B` / `3.3B` |
| **`nvidia/GR00T-N1.5-3B`** | Eagle2-VLM (`SigLIP + Qwen3-1.7B`) | Cross-Attention DiT (`16L`, 4-step Flow) | `2.09B` / `3.0B` |
| **`lerobot/smolvla_base`** | SmolVLM2 (`SigLIP + Llama-16L`) | SmolLM2 Action Expert (`16L`, 10-step Flow) | `0.36B` / `450M` |
| **`DirectActionHead`** | Any `SlowBackbone` | 1-step MLP regression head | Configurable |

Fine-tuned variants (e.g. `lerobot/pi05_libero_finetuned_v044`, `lerobot/smolvla_libero`) work out of the box, and custom architectures can be registered via `@ModelRegistry.register(...)`.

### Supported GPUs

| Architecture | GPUs | Supported Fast Precision |
| :--- | :--- | :--- |
| **Blackwell (`SM100–SM120`)** | B200, RTX PRO 6000 Blackwell, RTX 5090 | `fp8`, `bf16` |
| **Hopper (`SM90`)** | H100, H200, H800 | `fp8`, `bf16` |
| **Ada (`SM89`)** | RTX 4090, RTX 6000 Ada, L40S | `fp8`, `bf16` |
| **Ampere (`SM80–SM86`)** | A100, A40, RTX 3090 | `bf16` |

## Install

Requires Linux (`x86_64`), Python `>= 3.10`, PyTorch `>= 2.5` with CUDA, and CUDA Toolkit `>= 12.0`.

```bash
git clone https://github.com/darthjaja6/tempo.git
cd tempo
pip install -e . --no-build-isolation
```

<details>
<summary><b>Container &amp; Build Notes</b></summary>

- **Ubuntu 24.04 / PEP 668 containers**: `python3 -m venv --system-site-packages .venv && source .venv/bin/activate` before installing.
- **Headless Docker builds (no GPU attached)**: Export `TORCH_CUDA_ARCH_LIST="8.9;9.0;12.0"` before `pip install`.
- **Incremental C++/CUDA rebuilds**: Run `python build_ext.py`.

</details>

## Quick Start

### 1. Serve a Pretrained Checkpoint

```python
import torch
from tempo import Engine

with Engine(
    "lerobot/pi05_base",          # or "nvidia/GR00T-N1.5-3B", "lerobot/smolvla_base", or local path
    num_streams=4,
    slow_devices=[0],             # [0] colocated, [0] + fast_devices=[1,2,3] disaggregated, or [[0,1]] PP
    fast_devices=[0],
    fast_precision="fp8",         # "fp8" on SM89+, "bf16" on SM80+
    prompt="pick up the cup",
) as engine:
    engine.push_slow_batch(torch.randint(0, 256, (4, *engine.slow_obs_shape), dtype=torch.uint8))
    engine.push_fast_batch(torch.randn(4, *engine.fast_obs_shape, dtype=torch.bfloat16))
    actions, fast_aoi_ms, slow_aoi_ms, underruns = engine.sample_actions(cpu=True)
```

### 2. Custom or Modular `SlowBackbone` × `FastHead`

Pair any `SlowBackbone` with any compatible `FastHead`:

```python
from tempo import DirectActionHead, Engine, Pi05Config, Pi05SlowBackbone

slow = Pi05SlowBackbone(Pi05Config(), device="cuda:0")
fast = DirectActionHead(slow.channel, action_dim=14, chunk_size=50, state_dim=14, hidden_dim=512)

with Engine(slow=slow, fast=fast, num_streams=4, slow_devices=[0], fast_devices=[0]) as engine:
    ...
```

To integrate a new model, subclass `SlowBackbone` and `FastHead`:

```python
from tempo import DualSystemDeployment, FastHead, ModelRegistry, SlowBackbone, deploy_modules

class MySlowBackbone(SlowBackbone):
    def forward(self, obs, ctx):
        ctx.write_hidden_state(self.encoder(obs))  # or ctx.write_layer_kv(layer_idx, k, v)

class MyFastHead(FastHead):
    def forward(self, fast_obs, state, frozen_prefix):
        hidden, delta_t_sec = state.read_hidden()  # or state.cross_attend(q, layer_idx=i)
        chunk = self.policy_net(fast_obs, hidden, delta_t_sec)
        P = frozen_prefix.size(1)
        chunk[:, :P] = state.splice_prefix(chunk[:, :P], frozen_prefix)
        return chunk  # [B, chunk_size, action_dim]

@ModelRegistry.register("my_policy", "MyPolicy")
def build_my_policy(config, checkpoint_dir, *, slow_devices, fast_devices, fast_precision="bf16", prompt=None, **kwargs) -> DualSystemDeployment:
    return deploy_modules(MySlowBackbone(config), MyFastHead(config, fast_precision=fast_precision), slow_devices=slow_devices, fast_devices=fast_devices, prompt=prompt)
```

### 3. Multi-GPU and Edge-Cloud Examples

```bash
python examples/quickstart.py
python examples/quickstart.py --model lerobot/smolvla_base
python examples/multi_gpu_serving.py --model nvidia/GR00T-N1.5-3B --topology auto
```

See [examples/quickstart.py](./examples/quickstart.py) and [examples/multi_gpu_serving.py](./examples/multi_gpu_serving.py) for `@engine.stage`, `engine.pipeline`, and `RemoteSlowServer` / `RemoteSlowClient` usage.

## Benchmarks

Benchmarked against a multi-stream PyTorch `SDPA` baseline (`benchmarks/baselines.py`) under `500 Hz` closed-loop polling. **Max Streams at SLO** is the highest stream count `N` satisfying **`Chunk P99 <= deadline`**, **`Underruns == 0`**, and **`Slow AoI P99 <= 950 ms`**.

### 1. Cross-Model &amp; Cross-GPU Scaling

| Model (Deadline) | Hardware &amp; Placement | Max Streams at SLO<br>(Tempo vs. Base) | Chunk `P99`<br>(Tempo vs. Base) | Step-Response<br>(Tempo vs. Base) | Peak / Cluster VRAM &amp; FP8 Cosine |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`GR00T-N1.5-3B`** (`25 ms`) | `1x H100 SXM` (`Colocated`) | **`8`** vs. `0` | **`22.12 ms`** vs. `150.59 ms` (`6.8x`) | **`54.8 ms`** vs. `147.8 ms` | `3.98 / 3.98 GB` (`0.99994`) |
| **`GR00T-N1.5-3B`** (`25 ms`) | `2x H100 SXM` (`1S : 1F`) | **`8`** vs. `0` | **`10.38 ms`** vs. `125.59 ms` (`12.1x`) | **`55.1 ms`** vs. `128.0 ms` | `3.23 / 4.73 GB` (`0.99997`) |
| **`GR00T-N1.5-3B`** (`25 ms`) | **`4x H100 SXM` (`1S : 3F`)** | **`12`** vs. `0` | **`9.56 ms`** vs. `393.81 ms` (**`41.2x`**) | **`56.2 ms`** vs. `555.9 ms` | **`2.67 / 6.95 GB`** (`0.99993`) |
| **`pi0.5`** (`35 ms`) | `1x RTX 4090` (`Colocated`) | **`1`** vs. `0` | **`23.90 ms`** vs. `271.60 ms` (`11.4x`) | **`169.8 ms`** vs. `1,500 ms` | `5.49 / 5.49 GB` (`0.99926`) |
| **`pi0.5`** (`30 ms`) | `1x RTX PRO 6000` (`Colocated`) | **`1`** vs. `0` | **`28.82 ms`** vs. `394.92 ms` (`13.7x`) | **`66.9 ms`** vs. `1,500 ms` | `6.10 / 6.10 GB` (`0.99992`) |
| **`pi0.5`** (`30 ms`) | `2x RTX PRO 6000` (`1S : 1F`) | **`1`** vs. `0` | **`23.71 ms`** vs. `381.87 ms` (`16.1x`) | **`65.9 ms`** vs. `1,500 ms` | `5.35 / 6.63 GB` (`0.99992`) |
| **`pi0.5`** (`30 ms`) | `1x H100 SXM` (`Colocated`) | **`1`** vs. `0` | **`28.31 ms`** vs. `457.39 ms` (`16.2x`) | **`64.9 ms`** vs. `1,500 ms` | `6.15 / 6.15 GB` (`0.99992`) |
| **`pi0.5`** (`30 ms`) | `2x H100 SXM` (`1S : 1F`) | **`1`** vs. `0` | **`20.76 ms`** vs. `423.56 ms` (`20.4x`) | **`65.8 ms`** vs. `1,500 ms` | `5.85 / 7.18 GB` (`0.99993`) |
| **`SmolVLA`** (`25 ms`) | `1x RTX 4090` (`Colocated`) | **`8`** vs. `0` | **`24.36 ms`** vs. `136.06 ms` (`5.6x`) | `9.46 ms` @ `N=1` | **`0.89 / 0.89 GB`** (`0.99339`) |
| **`SmolVLA`** (`25 ms`) | `1x RTX PRO 6000` (`Colocated`) | **`12`** vs. `0` | **`24.11 ms`** vs. `360.54 ms` (`15.0x`) | `16.27 ms` @ `N=1` | `1.19 / 1.19 GB` (`0.99455`) |
| **`SmolVLA`** (`25 ms`) | `2x RTX PRO 6000` (`1S : 1F`) | **`12`** vs. `0` | **`22.98 ms`** vs. `292.08 ms` (`12.7x`) | `12.35 ms` @ `N=1` | `1.05 / 1.53 GB` (`0.99446`) |

### 2. 4-GPU Placement Comparison (`lerobot/pi05_base`, `N = 4` Streams)

Disaggregating Slow/Fast GPUs (`1S : 3F`) or pipelining the Slow VLM (`1xPP(2)S : 2F`) cuts cluster VRAM by `45%–59%` while keeping `0` underruns across both NVLink and PCIe systems:

#### `4x NVIDIA H100 80GB SXM` (NVLink)

| Placement (`N=4`) | Chunk `P50 / P99`<br>(Tempo vs. Base `P99`) | Underruns<br>(Tempo vs. Base) | Step-Response<br>(Tempo vs. Base) | Closed-Loop Success<br>(Tempo vs. Base) | Cluster / Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`4x Colocated`** (`slow=[0..3], fast=[0..3]`) | **`20.55 / 26.82 ms`** vs. `2,943.4 ms` | **`0`** vs. `20` | **`65.4 ms`** vs. `1,500 ms` | **`100.0%`** vs. `0.0%` | `23.55 / 5.89 GB` |
| `2S : 2F` (`slow=[0,1], fast=[2,3]`) | `20.38 / 36.88 ms` vs. `1,431.1 ms` | **`0`** vs. `748` | **`65.4 ms`** vs. `1,500 ms` | **`100.0%`** vs. `0.0%` | `15.69 / 5.85 GB` |
| **`1S : 3F`** (`slow=[0], fast=[1,2,3]`) | `20.57 / 36.83 ms` vs. `2,320.8 ms` | **`0`** vs. `109` | **`66.6 ms`** vs. `1,500 ms` | **`100.0%`** vs. `0.0%` | **`12.68 / 6.42 GB` (`-46%`)** |
| **`1xPP(2)S : 2F`** (`slow=[[0,1]], fast=[2,3]`) | `20.40 / 36.40 ms` vs. `804.3 ms` | **`0`** vs. `1,833` | **`64.9 ms`** vs. `1,500 ms` | **`100.0%`** vs. `0.0%` | **`12.95 / 4.33 GB` (`-45%`)** |

#### `4x NVIDIA RTX PRO 6000 Blackwell` (PCIe)

| Placement (`N=4`) | Chunk `P50 / P99`<br>(Tempo vs. Base `P99`) | Underruns<br>(Tempo vs. Base) | Step-Response<br>(Tempo vs. Base) | Closed-Loop Success<br>(Tempo vs. Base) | Cluster / Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`4x Colocated`** (`slow=[0..3], fast=[0..3]`) | **`22.12 / 28.76 ms`** vs. `2,676.6 ms` | **`0`** vs. `35` | **`67.4 ms`** vs. `1,500 ms` | **`100.0%`** vs. `0.0%` | `23.35 / 5.84 GB` |
| `2S : 2F` (`slow=[0,1], fast=[2,3]`) | `23.48 / 42.45 ms` vs. `1,198.5 ms` | **`0`** vs. `939` | **`67.3 ms`** vs. `1,500 ms` | **`100.0%`** vs. `0.0%` | `13.68 / 5.35 GB` |
| **`1S : 3F`** (`slow=[0], fast=[1,2,3]`) | `22.18 / 39.74 ms` vs. `1,681.5 ms` | **`0`** vs. `461` | **`66.3 ms`** vs. `1,500 ms` | **`100.0%`** vs. `0.0%` | **`9.76 / 5.72 GB` (`-58%`)** |
| **`1xPP(2)S : 2F`** (`slow=[[0,1]], fast=[2,3]`) | `22.18 / 39.68 ms` vs. `1,445.6 ms` | **`0`** vs. `755` | **`66.3 ms`** vs. `1,500 ms` | **`100.0%`** vs. `0.0%` | **`9.50 / 3.53 GB` (`-59%`)** |

### Reproduce

```bash
# Single-GPU colocated
python -m benchmarks.run_suite --model lerobot/smolvla_base --mode colocated --deadline-ms 25 --streams 1,4,8,12,16
python -m benchmarks.run_suite --model lerobot/pi05_base --mode colocated --deadline-ms 35 --streams 1,2,4,8

# Disaggregated 1S:1F & 1S:3F scaling
python -m benchmarks.run_suite --model nvidia/GR00T-N1.5-3B --mode both --deadline-ms 25 --streams 1,4,8,16,24,32
python -m benchmarks.run_suite --model nvidia/GR00T-N1.5-3B --mode custom --slow-devices 0 --fast-devices 1,2,3 --deadline-ms 25 --streams 4,12,24,32,42,48

# 4-GPU placement matrix (Colocated, 2S:2F, 1S:3F, 1xPP(2)S:2F)
python -m benchmarks.run_suite --model lerobot/pi05_base --mode four_gpu_matrix --deadline-ms 30 --streams 4,8,12,16,24,32
```

## License

Released under the [Apache-2.0 License](./LICENSE).
