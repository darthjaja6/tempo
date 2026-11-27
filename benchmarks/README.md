# Tempo Benchmarks

## 1. Real-Time Dynamic Manipulation

Evaluations run at `20 Hz` (`50 ms` physics step) on [`LIBERO-Dynamic`](https://github.com/renshaojie233/LIBERO-Dynamic) (`40` tasks × `3` motion families × `3` target speeds × `5` trajectories = `1,800` official episodes) and standard `LIBERO` (`400` episodes) without pausing simulation during policy inference (`K` denotes `20 Hz` control steps executed per chunk replan).

### 1.1 Zero-Retraining Single Robot (`1x RTX 5090`, `1,800` Episodes)

Serving Physical Intelligence's unmodified static OpenPI `pi05_libero` checkpoint (`gs://openpi-assets/checkpoints/pi05_libero`, zero dynamic or latency retraining) with Tempo outperforms both latency-retrained `VLASH` (`+5.5` pts) and official `OpenPI JAX Async` (`+3.1` pts):

| System | Dynamic / Latency Retraining | Success (`1,800` Eps) | `0.03 / 0.06 / 0.09 m/s` | Chunk Latency |
| :--- | :---: | :---: | :---: | :---: |
| `LeRobot PyTorch Sync` (`K = 10`) | No | 43.1% (775/1,800) | 48.7% / 42.5% / 38.0% | 147.2 ms (blocking) |
| `LeRobot PyTorch Async` (`K = 6`) | No | 48.4% (871/1,800) | 55.7% / 45.2% / 44.3% | 147.2 ms |
| `VLASH` | **Yes (latency-aware)** | 55.4% (998/1,800) | 65.7% / 54.7% / 46.0% | 104.0 ms |
| `OpenPI JAX Async` (`K = 4`) | No | 57.8% (1,041/1,800) | 64.8% / 57.5% / 51.2% | 60.0 ms |
| **Tempo** (`K = 5`) | **No** | **60.9% (1,097/1,800)** | **69.7% / 60.3% / 52.8%** | **38.5 ms mean (`38.2` `P50` / `46.1` `P99`)** |

### 1.2 Dynamically Fine-Tuned SOTA: From `1` Robot to `8` Concurrent Robots (`1x RTX 5090`)

**`1` Robot (Single-Stream Ceiling)** — On the dynamically fine-tuned SOTA checkpoint (`rsj2003/libero-dynamic-paper-models`, `pi05/mixed_ft3000`), pure-PyTorch Tempo (`37.0 ms P50`, `0` underruns) lifts success by **`+34.1` pts** over synchronous PyTorch and matches `OpenPI JAX Async` (`71.5 ms` mean) `1:1` at the `20 Hz` single-robot simulation ceiling across all target speed tiers:

| Target Speed Tier (`N = 1`) | Synchronous PyTorch (`K = 4`) | `OpenPI JAX Async` (Current SOTA, `10 Hz`) | **Tempo** |
| :--- | :---: | :---: | :---: |
| Standard (`0.03 / 0.06 / 0.09 m/s`) | 39.2% (47/120; `41.7%` on `360`) | 74.2% (89/120) | **73.3% (88/120, `K = 2`, `p = 1.00`)** |
| Fast (`0.10 / 0.12 / 0.15 m/s`) | 24.2% (29/120) | 48.3% (58/120) | **50.0% (180/360, `K = 3`)** |
| High-Speed (`0.15 / 0.18 / 0.20 m/s`) | — | 33.3% (40/120) | **32.5% (39/120, `K = 2`, `p = 1.00`)** |

**`8` Concurrent Robots (`1,800` Official Episodes)** — Once that single `RTX 5090` serves **`8` concurrent robots** on the same SOTA checkpoint (`K = 4`, `20 Hz`), monolithic `OpenPI JAX Async` serializes chunk requests (`280.5 ms` mean / `346.8 ms` `P90` latency, `63.2%` action queue underflow) and collapses to `16.1%`. Tempo micro-slices VLM prefill (`<= 2 ms`), batches `CUDA Graph` action heads, and sustains **`52.3%` (`+36.2` pts / `3.25x`, two-sided McNemar `p = 1.19e-132`)** with `0.8%` robot stall and **`12.7 tasks/min`** single-GPU goodput (`19.8 tasks/min` at `0.03 m/s`):

| System (`8` Robots / GPU) | Overall (`1,800` Eps) | `0.03 / 0.06 / 0.09 m/s` | `spatial / object / goal / 10` | Chunk Latency | Queue Underflow |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `OpenPI JAX Async` (Current SOTA, `K = 4`) | 16.1% (290/1,800) | 23.3% / 14.7% / 10.3% | 27.8% / 10.2% / 24.4% / 2.0% | 280.5 ms mean / 346.8 ms `P90` | 63.2% |
| **Tempo** (`K = 4`) | **52.3% (942/1,800)** | **67.5% / 52.5% / 37.0%** | **45.1% / 58.0% / 73.1% / 33.1%** | **26.7 ms mean (`23.7` `P50` / `43.4` `P99`)** | **0.15%** |

Against static `B = 8` batched synchronous inference (`Synchronous K = 5, N = 8` on `lerobot/pi05_libero_finetuned_v044`, `360` episodes), Tempo also improves both success rate (**`33.6%` vs. `28.1%`**) and fleet goodput (**`6.75` vs. `2.24 tasks/min`, `3.01x`**).

### 1.3 Synchronous vs. Asynchronous Ablations (`1x RTX 4090`)

`lerobot/pi05_libero_finetuned_v044` (`BF16`, `20 Hz` real-time control via [`inspect-robots`](https://github.com/robocurve/inspect-robots), identical checkpoint across all deployments; raw episode logs in [`libero_dynamic_eval.json`](./libero_dynamic_eval.json)):

**Moving-Object `LIBERO-Dynamic` (`360` Episodes: `40` Tasks × `3` Motion Families × `0.03 / 0.06 / 0.09 m/s`)**

| Deployment | Success | `0.03 / 0.06 / 0.09 m/s` | Time-to-Success (Paired vs. Tempo) | Robot Stall | Throughput (`tasks/min`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Synchronous (`K = 10`) | 31.7% (114/360) | 35.8% / 30.8% / 28.3% | 17.82 s (17.07 vs. 9.32 s, **1.83x**) | 40.0% | 0.48 |
| `torch.compile` Sync (`K = 4`) | 30.6% (110/360) | 38.3% / 25.8% / 27.5% | 18.66 s (15.76 vs. 9.25 s, **1.70x**) | 35.8% | 0.48 |
| Python Async | 7.2% (26/360) | 6.7% / 6.7% / 8.3% | 8.37 s (6.77 vs. 6.56 s, 1.03x) | 1.6% | 0.15 |
| **Tempo** | **39.2% (141/360)** | **46.7% / 35.0% / 35.8%** | **10.35 s** | **0.3%** | **1.05 (2.19x)** |

**Standard `LIBERO` (`400` Episodes: `40` Tasks × `10` Trials Across `spatial`, `object`, `goal`, `10`)**

| Deployment | Success (`spatial / object / goal / 10`) | Time-to-Success (Paired vs. Tempo) | Robot Stall | Throughput (`tasks/min`) | Chunk Latency (`P50 / P99`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Synchronous (`K = 10`) | 97.2% (`98 / 100 / 95 / 96%`) | 12.20 s (12.04 vs. 7.55 s, **1.60x**) | 41.2% | 4.61 | 376.4 / 488.6 ms |
| `torch.compile` Sync (`K = 4`) | 96.2% (`99 / 100 / 90 / 96%`) | 11.20 s (11.05 vs. 7.49 s, **1.48x**) | 35.7% | 4.89 | 115.5 / 164.5 ms |
| Python Async | 49.0% (`44 / 46 / 58 / 48%`) | 7.66 s (7.49 vs. 7.32 s, 1.02x) | 3.6% | 2.37 | 578.5 / 828.8 ms |
| **Tempo** | **96.0%** (`97 / 98 / 96 / 93%`) | **7.56 s** | **0.8%** | **7.18 (1.47–1.56x)** | **24.2 / 25.1 ms** |

## 2. Multi-Model Serving Throughput & Tail Latency

Each stream drives `2 Hz` Slow camera updates, `20 Hz` Fast chunk requests (`FP8` Fast heads on `SM89+`), and `500 Hz` action polling against a `BF16` multi-stream PyTorch `SDPA` + per-stream lock baseline ([`baselines.py`](./baselines.py)).

- **Max Streams at SLO**: Highest stream count meeting `Chunk P99 <= deadline`, `0` underruns, and `Slow staleness P99 <= 950 ms` (the PyTorch baseline misses the latency/underrun SLO at `1` stream).
- **FP8 Accuracy**: `FP8` Fast heads retain `>0.999` cosine similarity against `BF16` on `GR00T-N1.5` and `pi05`, and `>0.993` on `SmolVLA`.
- **Placement**: `Colocated` runs Slow and Fast on each GPU; `1S : 1F` and `1S : 3F` dedicate 1 GPU to Slow VLM and 1 or 3 GPUs to Fast action heads (`Chunk P99` and `VRAM` measured at `Max Streams at SLO`).

| Model (Deadline) | Hardware & Placement | Max Streams at SLO | Chunk `P99` (Tempo vs. PyTorch) | Zero-Underrun Capacity | Peak GPU / Total VRAM |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `nvidia/GR00T-N1.5-3B` (25 ms) | `4x H100 SXM` (`1S : 3F`) | **12** | **9.56 ms** vs. 393.8 ms (41.2x) | 48 streams (**13.3 ms** vs. 447.9 ms `P99`) | 2.67 / 6.95 GB |
| `nvidia/GR00T-N1.5-3B` (25 ms) | `2x H100 SXM` (`1S : 1F`) | **8** | **10.38 ms** vs. 125.6 ms (12.1x) | 32 streams | 3.23 / 4.73 GB |
| `nvidia/GR00T-N1.5-3B` (25 ms) | `1x H100 SXM` (`Colocated`) | **8** | **22.12 ms** vs. 150.6 ms (6.8x) | 32 streams | 3.98 / 3.98 GB |
| `lerobot/pi05_base` (30 ms) | `4x H100 SXM` (`Colocated`) | **4** | **26.82 ms** vs. 2,943.4 ms (109.7x) | 4 streams | 5.89 / 23.55 GB |
| `lerobot/pi05_base` (30 ms) | `2x H100 SXM` (`1S : 1F`) | **1** | **20.76 ms** vs. 423.6 ms (20.4x) | 2 streams | 5.85 / 7.18 GB |
| `lerobot/pi05_base` (30 ms) | `1x H100 SXM` (`Colocated`) | **1** | **28.31 ms** vs. 457.4 ms (16.2x) | 2 streams | 6.15 / 6.15 GB |
| `lerobot/pi05_base` (30 ms) | `2x RTX PRO 6000` (`1S : 1F`) | **1** | **23.71 ms** vs. 381.9 ms (16.1x) | 2 streams | 5.35 / 6.63 GB |
| `lerobot/pi05_base` (30 ms) | `1x RTX PRO 6000` (`Colocated`) | **1** | **28.82 ms** vs. 394.9 ms (13.7x) | 2 streams | 6.10 / 6.10 GB |
| `lerobot/pi05_base` (35 ms) | `1x RTX 4090` (`Colocated`) | **1** | **23.90 ms** vs. 271.6 ms (11.4x) | 2 streams | 5.49 / 5.49 GB |
| `lerobot/smolvla_base` (25 ms) | `2x RTX PRO 6000` (`1S : 1F`) | **12** | **22.98 ms** vs. 292.1 ms (12.7x) | 16 streams (**24.4 ms** vs. 294.5 ms `P99`) | 1.05 / 1.53 GB |
| `lerobot/smolvla_base` (25 ms) | `1x RTX PRO 6000` (`Colocated`) | **12** | **24.11 ms** vs. 360.5 ms (15.0x) | 16 streams | 1.19 / 1.19 GB |
| `lerobot/smolvla_base` (25 ms) | `1x RTX 4090` (`Colocated`) | **8** | **24.36 ms** vs. 136.1 ms (5.6x) | 12 streams | 0.89 / 0.89 GB |

## 3. Multi-GPU Disaggregation & Pipeline Parallelism (`lerobot/pi05_base`, 4 Streams)

All placements below run with `0` underruns:

| Placement (`N = 4`) | `4x H100 SXM` Chunk `P50 / P99` | `4x H100` Peak / Total VRAM | `4x RTX PRO 6000` Chunk `P50 / P99` | `4x RTX PRO 6000` Peak / Total VRAM |
| :--- | :---: | :---: | :---: | :---: |
| `4x Colocated` (`slow=[0..3], fast=[0..3]`) | **20.55 / 26.82 ms** | 5.89 / 23.55 GB | **22.12 / 28.76 ms** | 5.84 / 23.35 GB |
| `2S : 2F` (`slow=[0,1], fast=[2,3]`) | 20.38 / 36.88 ms | 5.85 / 15.69 GB | 23.48 / 42.45 ms | 5.35 / 13.68 GB |
| `1S : 3F` (`slow=[0], fast=[1,2,3]`) | 20.57 / 36.83 ms | 6.42 / **12.68 GB (-46%)** | 22.18 / 39.74 ms | 5.72 / **9.76 GB (-58%)** |
| `2-Stage PP + 2F` (`slow=[[0,1]], fast=[2,3]`) | 20.40 / 36.40 ms | **4.33** / 12.95 GB (-45%) | 22.18 / 39.68 ms | **3.53** / **9.50 GB (-59%)** |

## 4. Reproduce

```bash
# Single-robot 1,800-episode LIBERO-Dynamic on static OpenPI pi05_libero (converted to safetensors)
python -m benchmarks.eval_libero_dynamic --dyn-repo /path/to/LIBERO-Dynamic \
  --checkpoint /path/to/pi05_libero_static --suites dynamic --policies tempo \
  --streams 1 --slow-every-steps 2 --fast-every-steps 5 --traj-indices 0,1,2,3,4 \
  --out-json benchmarks/libero_dynamic_n1_static_1800.json

# 8-robot single-GPU fleet on 1,800-episode LIBERO-Dynamic (rsj2003/libero-dynamic-paper-models: pi05/mixed_ft3000)
python -m benchmarks.eval_libero_dynamic --dyn-repo /path/to/LIBERO-Dynamic \
  --checkpoint /path/to/pi05_mixed_ft3000 --suites dynamic --policies tempo \
  --streams 8 --slow-every-steps 2 --fast-every-steps 4 --traj-indices 0,1,2,3,4 \
  --out-json benchmarks/libero_dynamic_n8_sota_1800.json

# Synchronous vs. asynchronous suite on static LIBERO (400 eps) + LIBERO-Dynamic (360 eps)
python -m benchmarks.eval_libero_dynamic --dyn-repo /path/to/LIBERO-Dynamic \
  --out-json benchmarks/libero_dynamic_eval.json

# Multi-model & multi-GPU streaming latency/goodput suites
python -m benchmarks.run_suite --model lerobot/smolvla_base --mode colocated --deadline-ms 25 --streams 1,4,8,12,16
python -m benchmarks.run_suite --model nvidia/GR00T-N1.5-3B --mode custom --slow-devices 0 --fast-devices 1,2,3 --deadline-ms 25 --streams 4,12,24,32,42,48
python -m benchmarks.run_suite --model lerobot/pi05_base --mode four_gpu_matrix --deadline-ms 30 --streams 4,8,12,16,24,32
```
