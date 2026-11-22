<h1 align="center">Tempo</h1>

<h3 align="center">
Fast serving engine for System 1 + System 2 robot policies
</h3>

<p align="center">
| <a href="#install"><b>Install</b></a> | <a href="#quick-start"><b>Quick Start</b></a> | <a href="#supported-models-and-hardware"><b>Models &amp; Hardware</b></a> | <a href="#benchmark-and-performance"><b>Benchmarks</b></a> | <a href="./examples"><b>Examples</b></a> | <a href="./LICENSE"><b>License</b></a> |
</p>

---

**Tempo** is a fast inference and serving engine for `System 1 + System 2` robot policies—such as $\pi_{0.5}$, GR00T-N1.5, SmolVLA, and Helix—where a `1–10 Hz` Vision-Language Model (`SlowBackbone`) conditions a `10–30 Hz` reactive action head (`FastHead`) driving `100–1000 Hz` continuous robot control. It pairs a zero-GIL runtime with a clean PyTorch `nn.Module` interface for low-latency serving across single-GPU, multi-GPU, and edge-cloud deployments.

<p align="center">
  <img src="./architecture.svg" width="100%" alt="Tempo Architecture">
</p>

Evaluated on **three public Hugging Face checkpoints** (`nvidia/GR00T-N1.5-3B`, `lerobot/pi05_base`, `lerobot/smolvla_base`) against a multi-GPU **PyTorch Dual-Stream (`SDPA` + per-stream `RWLock`)** baseline on identical GPU placements ([full benchmark matrix →](#benchmark-and-performance)):

| Pretrained Checkpoint | Hardware &amp; Topology | `SLO Goodput`<br>(Tempo vs. PyTorch) | Fast Chunk `P99` (`TTAC`)<br>at `SLO` Load | `0%`-Underrun Capacity /<br>Closed-Loop Success |
| :--- | :--- | :---: | :---: | :---: |
| **`nvidia/GR00T-N1.5-3B`** (`3.0B`, 4-step DiT) | **`4x H100 SXM`** (`1S : 3F`) | **`12` streams** vs. `0` | **`9.56 ms`** vs. `393.8 ms` (**`41.2x`**) | **`48` streams** (`13.3 ms` P99) / **`100.0%`** vs. `0.0%` |
| **`nvidia/GR00T-N1.5-3B`** (`3.0B`, 4-step DiT) | **`1x H100 SXM`** (`Colocated`) | **`8` streams** vs. `0` | **`22.12 ms`** vs. `150.6 ms` (**`6.8x`**) | **`32` streams** (`24.6 ms` P99) / **`100.0%`** vs. `0.0%` |
| **`lerobot/pi05_base`** (`3.3B`, 10-step Flow) | **`4x H100 SXM`** (`Colocated`) | **`4` streams** vs. `0` | **`26.82 ms`** vs. `2,943.4 ms` (**`109.7x`**) | `4` streams / **`100.0%`** vs. `0.0%` |
| **`lerobot/pi05_base`** (`3.3B`, 10-step Flow) | **`2x H100 SXM`** (`1S : 1F`) | **`1` stream** vs. `0` | **`20.76 ms`** vs. `423.6 ms` (**`20.4x`**) | `2` streams / **`100.0%`** vs. `0.0%` |
| **`lerobot/pi05_base`** (`3.3B`, 10-step Flow) | **`1x RTX 4090`** (`Colocated`) | **`1` stream** vs. `0` | **`23.90 ms`** vs. `271.6 ms` (**`11.4x`**) | `2` streams (`5.55 GB` VRAM, `0` vs. `2,952` underruns) |
| **`lerobot/smolvla_base`** (`450M`, 10-step Flow) | **`1x RTX 4090`** (`Colocated`) | **`8` streams** vs. `0` | **`24.36 ms`** vs. `136.1 ms` (**`5.6x`**) | **`12` streams** (`0.98 GB` VRAM, `0` vs. `18,974` underruns) |
| **`lerobot/smolvla_base`** (`450M`, 10-step Flow) | **`2x RTX PRO 6000`** (`1S : 1F`) | **`12` streams** vs. `0` | **`22.98 ms`** vs. `292.1 ms` (**`12.7x`**) | **`16` streams** (`1.07 GB` VRAM, `0` vs. `15,152` underruns) |

Tempo is fast with:

- Lock-free **`(M+2)`-slot Temporal State Cache (`TSC`)** (`M` reader stages) for zero-blocking, zero-torn-read cross-rate KV (`mode="kv"`) and latent (`mode="hidden"`) sharing
- Bounded **3D Relative-Time RoPE** (`(h, w)` spatial RoPE on write, on-chip SRAM `t_now - t_obs` RoPE + adaptive Split-K WMMA cross-attention on read)
- Zero-GIL C++ **Age-of-Information (`AoI`) + Slack-Time scheduler** with `O(1)` latest-only sensor coalescing
- Automatic **$\le 2.0\text{ ms}$ piecewise CUDA Graph micro-slicing** for single-GPU Slow/Fast colocation
- Engine-managed **regional `torch.compile` + bucketed Fast `CUDA Graphs` + `FP8` (`E4M3`)** quantization—accelerating pure PyTorch models with zero model-specific CUDA patches
- Hardware-clocked **Real-Time Chunking (`RTC`)** ring with frozen-prefix splicing and cubic Hermite overrun blending

Tempo is clean and extensible for open-source developers:

- **5-line Hugging Face serving** via a single unified `Engine("lerobot/pi05_base", ...)` constructor
- **Algorithm-neutral `SlowBackbone` × `FastHead` (`nn.Module`) standard** supporting $n \times m$ modular composition across Flow-Matching, DiT Diffusion, Autoregressive, or 1-step Parallel Regression heads (`DirectActionHead`)
- Declarative **`x` Slow : `y` Fast (`xSyF`)** GPU disaggregation and overlapped **Slow Pipeline Parallelism (`PP`)** with automatic NVLink P2P or pinned-host DMA pre-push
- Built-in `RemoteSlowServer` / `RemoteSlowClient` Echo-Timestamp TCP bridge for Edge-Cloud split serving

## Supported Models and Hardware

### Built-In Pretrained Models

| Hugging Face Repo ID | Architecture (`System 2` → `TemporalChannel` → `System 1`) | Active / Total Params | Action Head |
| :--- | :--- | :---: | :--- |
| **`lerobot/pi05_base`** | `PaliGemma-2B` (`18L`) → `mode="kv"` (`head_dim=256`) → `Gemma Expert` (`18L`, AdaRMS) | `2.83B` / `3.3B` | 10-step Flow-Matching (`32` DoF) |
| **`nvidia/GR00T-N1.5-3B`** | `Eagle2-VLM` (`SigLIP-400M + Qwen3-1.7B`) → `mode="hidden"` (`2048d`) → `Cross-Attn DiT` (`16L`) | `2.09B` / `3.0B` | 4-step Flow/DiT (`32` DoF) |
| **`lerobot/smolvla_base`** | `SmolVLM2` (`SigLIP + Llama-16L`) → `mode="kv"` (`head_dim=64`) → `Interleaved Expert` (`16L`) | `0.36B` / `450M` | 10-step Flow-Matching (`6` DoF) |
| **`DirectActionHead`** *(modular)* | Pairs with **any** `kv` or `hidden` `SlowBackbone` (`Engine(slow=..., fast=...)`) | Configurable | 1-step Parallel Regression |

Register any custom Hub or local checkpoint architecture (`config.json` `"type"`, `"model_type"`, or `"architectures"`) via `@ModelRegistry.register("my_policy", "MyPolicy")`.

### Supported GPU Architectures

| Platform | Representative GPUs | Precision &amp; Interconnect Support |
| :--- | :--- | :--- |
| **NVIDIA Blackwell (`SM100–SM120`)** | B200, RTX PRO 6000 Blackwell, RTX 5090 | `fast_precision="fp8"` (`E4M3`) + `"bf16"`; NVLink P2P &amp; PCIe DMA |
| **NVIDIA Hopper (`SM90`)** | H100, H200, H800 | `fast_precision="fp8"` (`E4M3`) + `"bf16"`; NVLink P2P &amp; PCIe DMA |
| **NVIDIA Ada (`SM89`)** | RTX 4090, RTX 6000 Ada, L40S | `fast_precision="fp8"` (`E4M3`) + `"bf16"`; PCIe P2P &amp; pinned-host DMA |
| **NVIDIA Ampere (`SM80–SM86`)** | A100, A40, RTX 3090 | `fast_precision="bf16"` Slow &amp; Fast CUDA Graphs; NVLink &amp; PCIe DMA |

## Install

Requires Linux (`x86_64`), Python `>= 3.10`, PyTorch `>= 2.5` with CUDA, and CUDA Toolkit `>= 12.0` (`nvcc`, `g++` with C++17).

```bash
git clone https://github.com/darthjaja6/tempo.git
cd tempo
pip install -e . --no-build-isolation
```

<details>
<summary><b>Container &amp; Build Notes</b></summary>

- **PEP 668 containers (e.g., Ubuntu 24.04 with preinstalled PyTorch)**: Create a venv inheriting system PyTorch first: `python3 -m venv --system-site-packages .venv && source .venv/bin/activate`.
- **Headless Docker builds (no GPU attached)**: Export `TORCH_CUDA_ARCH_LIST="8.9;9.0;12.0"` before running `pip install`.
- **Incremental C++/CUDA rebuilds**: Run `python build_ext.py` from the repository root.

</details>

## Quick Start

### 1. Serve a Pretrained Hugging Face Checkpoint in 5 Lines

```python
import torch
from tempo import Engine

with Engine(
    "lerobot/pi05_base",          # or "nvidia/GR00T-N1.5-3B", "lerobot/smolvla_base", or local path
    num_streams=4,
    slow_devices=[0],             # e.g. [0] colocated, [0] + fast_devices=[1,2,3] (1S:3F), or [[0,1]] (PP)
    fast_devices=[0],
    fast_precision="fp8",         # "fp8" on Ada/Hopper/Blackwell (SM89+), "bf16" on Ampere+
    prompt="pick up the cup",
) as engine:
    engine.push_slow_batch(torch.randint(0, 256, (4, *engine.slow_obs_shape), dtype=torch.uint8))
    engine.push_fast_batch(torch.randn(4, *engine.fast_obs_shape, dtype=torch.bfloat16))
    actions, fast_aoi_ms, slow_aoi_ms, underruns = engine.sample_actions(cpu=True)  # poll at 100-1000 Hz
```

### 2. Modular `SlowBackbone` × `FastHead` ($n \times m$ Composition)

Write standard PyTorch `nn.Module` subclasses—no custom CUDA kernels or diffusion-specific method names required. Tempo automatically compiles pure-PyTorch `FastHead` blocks via `torch.compile` (`torch.library.custom_op` over `TemporalStateCache`) and captures them into bucketed `CUDA Graphs`:

```python
import torch.nn as nn
from tempo import DirectActionHead, Engine, FastHead, Pi05Config, Pi05SlowBackbone, SlowBackbone

# Pair any SlowBackbone with any compatible FastHead (Flow-Matching, DiT, or 1-step Regression):
slow = Pi05SlowBackbone(Pi05Config(), device="cuda:0")
fast = DirectActionHead(slow.channel, action_dim=14, chunk_size=50, state_dim=14, hidden_dim=512)

with Engine(slow=slow, fast=fast, num_streams=4, slow_devices=[0], fast_devices=[0]) as engine:
    ...
```

To implement your own `SlowBackbone` or `FastHead`, override `forward` (`uint8` camera frames pushed via `push_slow` are scaled to `[0, 1]` `bfloat16` while floating-point tensors pass through untouched, and `nn.ModuleList` / `nn.Sequential` blocks are sliced and `torch.compile`-fused automatically):

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

### 3. Low-Level `@engine.stage`, `engine.pipeline`, and Edge-Cloud Serving

For custom multi-rate execution graphs, multi-stage Slow Pipeline Parallelism (`[[0, 1]]`), or split Edge-Cloud deployments (`RemoteSlowServer` / `RemoteSlowClient`), use `@engine.stage` and `engine.pipeline(...)` directly:

```bash
python examples/quickstart.py
python examples/quickstart.py --model lerobot/smolvla_base
python examples/multi_gpu_serving.py --model nvidia/GR00T-N1.5-3B --topology auto
```

See [examples/quickstart.py](./examples/quickstart.py) and [examples/multi_gpu_serving.py](./examples/multi_gpu_serving.py) for complete runnable scripts.

## Benchmark and Performance

All benchmarks load **real public Hugging Face weights** (`safetensors` + tokenizer) and evaluate Tempo against `PyTorchDualStreamBaseline` (`benchmarks/baselines.py`: dedicated Slow/Fast CUDA streams + per-stream `RWLock` + PyTorch `SDPA`) on **identical GPU topologies**.

**`SLO Goodput`**: Maximum concurrent robot streams `N` simultaneously satisfying **`Fast TTAC P99 <= deadline_ms`**, **`Hard Underruns == 0`**, and **`Slow AoI P99 <= 950 ms`** under continuous `500 Hz` clocked actuation polling.

### 1. Cross-Model &amp; Cross-GPU Scaling (`1x`, `2x`, `4x` GPUs)

| Model &amp; Deadline | Hardware &amp; Topology | `SLO Goodput`<br>(Tempo vs. Base) | `Fast TTAC P99`<br>(Tempo vs. Base) | Step-Response<br>(Tempo vs. Base) | Peak / Cluster VRAM &amp; FP8 Parity |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`GR00T-N1.5-3B`** (`25 ms`) | `1x H100 SXM` (`Colocated`) | **`8`** vs. `0` | **`22.12 ms`** vs. `150.59 ms` (`6.8x`) | **`54.8 ms`** vs. `147.8 ms` | `3.98 / 3.98 GB` (`cos=0.99994`) |
| **`GR00T-N1.5-3B`** (`25 ms`) | `2x H100 SXM` (`1S : 1F`) | **`8`** vs. `0` | **`10.38 ms`** vs. `125.59 ms` (`12.1x`) | **`55.1 ms`** vs. `128.0 ms` | `3.23 / 4.73 GB` (`cos=0.99997`) |
| **`GR00T-N1.5-3B`** (`25 ms`) | **`4x H100 SXM` (`1S : 3F`)** | **`12`** vs. `0` | **`9.56 ms`** vs. `393.81 ms` (**`41.2x`**) | **`56.2 ms`** vs. `555.9 ms` | **`2.67 / 6.95 GB`** (`cos=0.99993`) |
| **`pi0.5`** (`35 ms`) | `1x RTX 4090` (`Colocated`) | **`1`** vs. `0` | **`23.90 ms`** vs. `271.60 ms` (`11.4x`) | **`169.8 ms`** vs. `1,500 ms` | `5.49 / 5.49 GB` (`cos=0.99926`) |
| **`pi0.5`** (`30 ms`) | `1x RTX PRO 6000` (`Colocated`) | **`1`** vs. `0` | **`28.82 ms`** vs. `394.92 ms` (`13.7x`) | **`66.9 ms`** vs. `1,500 ms` | `6.10 / 6.10 GB` (`cos=0.99992`) |
| **`pi0.5`** (`30 ms`) | `2x RTX PRO 6000` (`1S : 1F`) | **`1`** vs. `0` | **`23.71 ms`** vs. `381.87 ms` (`16.1x`) | **`65.9 ms`** vs. `1,500 ms` | `5.35 / 6.63 GB` (`cos=0.99992`) |
| **`pi0.5`** (`30 ms`) | `1x H100 SXM` (`Colocated`) | **`1`** vs. `0` | **`28.31 ms`** vs. `457.39 ms` (`16.2x`) | **`64.9 ms`** vs. `1,500 ms` | `6.15 / 6.15 GB` (`cos=0.99992`) |
| **`pi0.5`** (`30 ms`) | `2x H100 SXM` (`1S : 1F`) | **`1`** vs. `0` | **`20.76 ms`** vs. `423.56 ms` (`20.4x`) | **`65.8 ms`** vs. `1,500 ms` | `5.85 / 7.18 GB` (`cos=0.99993`) |
| **`SmolVLA`** (`25 ms`) | `1x RTX 4090` (`Colocated`) | **`8`** vs. `0` | **`24.36 ms`** vs. `136.06 ms` (`5.6x`) | `9.46 ms` TTAC @ `N=1` | **`0.89 / 0.89 GB`** (`cos=0.99339`) |
| **`SmolVLA`** (`25 ms`) | `1x RTX PRO 6000` (`Colocated`) | **`12`** vs. `0` | **`24.11 ms`** vs. `360.54 ms` (`15.0x`) | `16.27 ms` TTAC @ `N=1` | `1.19 / 1.19 GB` (`cos=0.99455`) |
| **`SmolVLA`** (`25 ms`) | `2x RTX PRO 6000` (`1S : 1F`) | **`12`** vs. `0` | **`22.98 ms`** vs. `292.08 ms` (`12.7x`) | `12.35 ms` TTAC @ `N=1` | `1.05 / 1.53 GB` (`cos=0.99446`) |

### 2. Equal-Hardware 4-GPU Topology Matrix (`lerobot/pi05_base`, `N = 4` Concurrent Robots)

Because `lerobot/pi05_base` executes **10 sequential transformer flow-matching steps** (`head_dim=256`, `~20.5 ms` at `B=1` on H100), placing Fast replicas on all 4 GPUs (`4x Colocated`) minimizes `B=1` Fast chunk latency (`26.82 ms` P99), whereas disaggregating with **`1S : 3F`** or **`1xPP(2)S : 2F`** cuts cluster VRAM by **`45%–59%`** and per-GPU peak VRAM down to **`3.53–4.33 GB`** while maintaining **`0` hard underruns** and **`100.0%` closed-loop tracking success** across every topology (vs. `0.0%` on the PyTorch baseline):

#### `4x NVIDIA H100 80GB SXM` (`NV18` NVLink Direct P2P)

| 4-GPU Topology (`lerobot/pi05_base`, `N=4`) | `Fast TTAC` `P50 / P99`<br>(Tempo vs. Base `P99`) | Hard Underruns<br>(Tempo vs. Base) | Step-Response<br>(Tempo vs. Base) | Closed-Loop Success<br>(Tempo vs. Base) | Cluster VRAM /<br>Peak per GPU |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`4x Colocated`** (`slow=[0..3], fast=[0..3]`) | **`20.55 / 26.82 ms`** vs. `2,943.4 ms` | **`0`** vs. `20` | **`65.4 ms`** vs. `1,500 ms` | **`100.0%`** (`RMSE 0.151`) vs. `0.0%` | `23.55 GB` / `5.89 GB` |
| `2S : 2F` (`slow=[0,1], fast=[2,3]`) | `20.38 / 36.88 ms` vs. `1,431.1 ms` | **`0`** vs. `748` | **`65.4 ms`** vs. `1,500 ms` | **`100.0%`** (`RMSE 0.170`) vs. `0.0%` | `15.69 GB` / `5.85 GB` |
| **`1S : 3F`** (`slow=[0], fast=[1,2,3]`) | `20.57 / 36.83 ms` vs. `2,320.8 ms` | **`0`** vs. `109` | **`66.6 ms`** vs. `1,500 ms` | **`100.0%`** (`RMSE 0.163`) vs. `0.0%` | **`12.68 GB` (`-46%`)** / `6.42 GB` |
| **`1xPP(2)S : 2F`** (`slow=[[0,1]], fast=[2,3]`) | `20.40 / 36.40 ms` vs. `804.3 ms` | **`0`** vs. `1,833` | **`64.9 ms`** vs. `1,500 ms` | **`100.0%`** (`RMSE 0.158`) vs. `0.0%` | **`12.95 GB` (`-45%`)** / **`4.33 GB`** |

#### `4x NVIDIA RTX PRO 6000 Blackwell Server Edition` (`96 GB`, Pure PCIe Without NVLink)

| 4-GPU Topology (`lerobot/pi05_base`, `N=4`) | `Fast TTAC` `P50 / P99`<br>(Tempo vs. Base `P99`) | Hard Underruns<br>(Tempo vs. Base) | Step-Response<br>(Tempo vs. Base) | Closed-Loop Success<br>(Tempo vs. Base) | Cluster VRAM /<br>Peak per GPU |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`4x Colocated`** (`slow=[0..3], fast=[0..3]`) | **`22.12 / 28.76 ms`** vs. `2,676.6 ms` | **`0`** vs. `35` | **`67.4 ms`** vs. `1,500 ms` | **`100.0%`** (`RMSE 0.213`) vs. `0.0%` | `23.35 GB` / `5.84 GB` |
| `2S : 2F` (`slow=[0,1], fast=[2,3]`) | `23.48 / 42.45 ms` vs. `1,198.5 ms` | **`0`** vs. `939` | **`67.3 ms`** vs. `1,500 ms` | **`100.0%`** (`RMSE 0.182`) vs. `0.0%` | `13.68 GB` / `5.35 GB` |
| **`1S : 3F`** (`slow=[0], fast=[1,2,3]`) | `22.18 / 39.74 ms` vs. `1,681.5 ms` | **`0`** vs. `461` | **`66.3 ms`** vs. `1,500 ms` | **`100.0%`** (`RMSE 0.174`) vs. `0.0%` | **`9.76 GB` (`-58%`)** / `5.72 GB` |
| **`1xPP(2)S : 2F`** (`slow=[[0,1]], fast=[2,3]`) | `22.18 / 39.68 ms` vs. `1,445.6 ms` | **`0`** vs. `755` | **`66.3 ms`** vs. `1,500 ms` | **`100.0%`** (`RMSE 0.163`) vs. `0.0%` | **`9.50 GB` (`-59%`)** / **`3.53 GB`** |

*(Conversely, for 4-step DiT heads such as `nvidia/GR00T-N1.5-3B` where a `B=16` Fast batch takes `< 14 ms`, asymmetric **`1S : 3F`** achieves **`9.56 ms` TTAC P99 at `N = 12`** and **`13.34 ms` TTAC P99 with `0` underruns up to `N = 48` streams**.)*

### Reproduce Benchmarks

```bash
# 1-GPU Colocated (e.g. SmolVLA or pi0.5 on RTX 4090)
python -m benchmarks.run_suite --model lerobot/smolvla_base --mode colocated --deadline-ms 25 --streams 1,4,8,12,16
python -m benchmarks.run_suite --model lerobot/pi05_base --mode colocated --deadline-ms 35 --streams 1,2,4,8

# 1x vs 2x (1S:1F) Comparison & Asymmetric 1S:3F Scaling
python -m benchmarks.run_suite --model nvidia/GR00T-N1.5-3B --mode both --deadline-ms 25 --streams 1,4,8,16,24,32
python -m benchmarks.run_suite --model nvidia/GR00T-N1.5-3B --mode custom --slow-devices 0 --fast-devices 1,2,3 --deadline-ms 25 --streams 4,12,24,32,42,48

# Full 4-GPU Topology Matrix (Colocated, 2S:2F, 1S:3F, 1xPP(2)S:2F)
python -m benchmarks.run_suite --model lerobot/pi05_base --mode four_gpu_matrix --deadline-ms 30 --streams 4,8,12,16,24,32
```

## Learn More

- **[Runnable Examples (`./examples`)](./examples)**: Standalone scripts for 5-line Hugging Face serving, $n \times m$ `(SlowBackbone, FastHead)` pairing, multi-GPU (`xSyF` &amp; Slow `PP`), and Edge-Cloud TCP serving.
- **[License (`./LICENSE`)](./LICENSE)**: Released under the Apache-2.0 License.
