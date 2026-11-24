# Tempo Benchmarks

Each stream drives `2 Hz` Slow camera updates, `20 Hz` Fast chunk requests (`FP8` Fast heads on `SM89+`), and `500 Hz` action polling against a `BF16` multi-stream PyTorch `SDPA` + per-stream `RWLock` baseline ([`baselines.py`](./baselines.py)).

- **Max Streams at SLO**: Highest stream count meeting `Chunk P99 <= deadline`, `0` underruns, and `Slow staleness P99 <= 950 ms` (the PyTorch baseline misses the latency/underrun SLO at `1` stream).
- **FP8 Accuracy**: `FP8` Fast heads retain `>0.999` cosine similarity against `BF16` on GR00T-N1.5 and $\pi_{0.5}$, and `>0.993` on SmolVLA.

## 1. Cross-Model & Cross-GPU Scaling

`Colocated` runs Slow and Fast on each GPU; `1S : 1F` and `1S : 3F` dedicate 1 GPU to Slow VLM and 1 or 3 GPUs to Fast action heads. `Chunk P99` and `VRAM` are measured at `Max Streams at SLO`.

| Model (Deadline) | Hardware & Placement | Max Streams at SLO | Chunk `P99` (Tempo vs. PyTorch) | Zero-Underrun Capacity | Peak GPU / Total VRAM |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `nvidia/GR00T-N1.5-3B` (25 ms) | `4x H100 SXM` (`1S : 3F`) | **12** | **9.56 ms** vs. 393.8 ms (41.2x) | 48 streams | 2.67 / 6.95 GB |
| `nvidia/GR00T-N1.5-3B` (25 ms) | `2x H100 SXM` (`1S : 1F`) | **8** | **10.38 ms** vs. 125.6 ms (12.1x) | 32 streams | 3.23 / 4.73 GB |
| `nvidia/GR00T-N1.5-3B` (25 ms) | `1x H100 SXM` (`Colocated`) | **8** | **22.12 ms** vs. 150.6 ms (6.8x) | 32 streams | 3.98 / 3.98 GB |
| `lerobot/pi05_base` (30 ms) | `2x H100 SXM` (`1S : 1F`) | **1** | **20.76 ms** vs. 423.6 ms (20.4x) | 2 streams | 5.85 / 7.18 GB |
| `lerobot/pi05_base` (30 ms) | `1x H100 SXM` (`Colocated`) | **1** | **28.31 ms** vs. 457.4 ms (16.2x) | 2 streams | 6.15 / 6.15 GB |
| `lerobot/pi05_base` (30 ms) | `2x RTX PRO 6000` (`1S : 1F`) | **1** | **23.71 ms** vs. 381.9 ms (16.1x) | 2 streams | 5.35 / 6.63 GB |
| `lerobot/pi05_base` (30 ms) | `1x RTX PRO 6000` (`Colocated`) | **1** | **28.82 ms** vs. 394.9 ms (13.7x) | 2 streams | 6.10 / 6.10 GB |
| `lerobot/pi05_base` (35 ms) | `1x RTX 4090` (`Colocated`) | **1** | **23.90 ms** vs. 271.6 ms (11.4x) | 2 streams | 5.49 / 5.49 GB |
| `lerobot/smolvla_base` (25 ms) | `2x RTX PRO 6000` (`1S : 1F`) | **12** | **22.98 ms** vs. 292.1 ms (12.7x) | 16 streams | 1.05 / 1.53 GB |
| `lerobot/smolvla_base` (25 ms) | `1x RTX PRO 6000` (`Colocated`) | **12** | **24.11 ms** vs. 360.5 ms (15.0x) | 16 streams | 1.19 / 1.19 GB |
| `lerobot/smolvla_base` (25 ms) | `1x RTX 4090` (`Colocated`) | **8** | **24.36 ms** vs. 136.1 ms (5.6x) | 12 streams | 0.89 / 0.89 GB |

## 2. 4-GPU Placement Trade-Offs (`lerobot/pi05_base`, 4 Streams)

All Tempo placements below run with `0` underruns:

| Placement (`N = 4`) | `4x H100 SXM` Chunk `P50 / P99` | `4x H100` Peak / Total VRAM | `4x RTX PRO 6000` Chunk `P50 / P99` | `4x RTX PRO 6000` Peak / Total VRAM |
| :--- | :---: | :---: | :---: | :---: |
| `4x Colocated` (`slow=[0..3], fast=[0..3]`) | **20.55 / 26.82 ms** | 5.89 / 23.55 GB | **22.12 / 28.76 ms** | 5.84 / 23.35 GB |
| `2S : 2F` (`slow=[0,1], fast=[2,3]`) | 20.38 / 36.88 ms | 5.85 / 15.69 GB | 23.48 / 42.45 ms | 5.35 / 13.68 GB |
| `1S : 3F` (`slow=[0], fast=[1,2,3]`) | 20.57 / 36.83 ms | 6.42 / **12.68 GB (-46%)** | 22.18 / 39.74 ms | 5.72 / **9.76 GB (-58%)** |
| `2-Stage PP + 2F` (`slow=[[0,1]], fast=[2,3]`) | 20.40 / 36.40 ms | **4.33** / 12.95 GB (-45%) | 22.18 / 39.68 ms | **3.53** / **9.50 GB (-59%)** |

## 3. Closed-Loop LIBERO Sanity Check (`1x RTX 4090`)

End-to-end MuJoCo rollouts via `tempo.Engine` ([`libero_e2e_eval.json`](./libero_e2e_eval.json), 30 Hz control loop, `0` underruns, `~200 µs` `sample_actions` P50):

| Suite | Task | `pi05_libero_finetuned_v044` (3.3B) | `smolvla_libero` (450M) |
| :--- | :--- | :---: | :---: |
| `libero_spatial` | Pick up the black bowl between the plate and ramekin and place it on the plate | Pass (83 steps) | Fail (220 steps) |
| `libero_object` | Pick up the alphabet soup and place it in the basket | Pass (137 steps) | Pass (143 steps) |
| `libero_goal` | Open the middle drawer of the cabinet | Pass (129 steps) | Pass (200 steps) |

## 4. Real-Time Task Completion & Dynamic Reaction (`inspect-robots` × `LIBERO`)

Evaluated on `1x RTX 4090` (`lerobot/pi05_libero_finetuned_v044`, 3.3B `BF16`) through [`robocurve/inspect-robots`](https://github.com/robocurve/inspect-robots) (`SELF_PACED` 20 Hz wall-clock physics, [`libero_dynamic_eval.json`](./libero_dynamic_eval.json)) against three baselines: standard open-loop chunking (`SyncChunk`, `K = 10`), `torch.compile`-accelerated reactive sync (`CompiledSync`, `K = 4`), and async dual-stream PyTorch (`PyTorchAsync`, `SDPA` + per-stream lock).

### 4.1 Standard LIBERO (`12` Scenes across `libero_spatial`, `libero_object`, `libero_goal`)

| Load | Policy | Success | Time-to-Success (Paired vs. Tempo) | Robot Stall | Goodput (`tasks/min`) | Latency (`P50 / P99`) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| `N = 1` | `SyncChunk` (`K = 10`) | 91.7% (11/12) | 8.95 s (8.95 vs. 5.69 s, **1.57x**) | 41.7% | 5.47 | 341.3 / 380.7 ms |
| `N = 1` | `CompiledSync` (`K = 4`) | 100.0% (12/12) | 8.73 s (8.73 vs. 5.78 s, **1.51x**) | 35.0% | 6.87 | 104.1 / 123.1 ms |
| `N = 1` | `PyTorchAsync` (`SDPA`) | 16.7% (2/12) | 7.53 s (7.53 vs. 5.64 s, **1.33x**) | 3.4% | 0.80 | 598.8 / 781.5 ms |
| `N = 1` | **Tempo (`Engine`)** | **100.0% (12/12)** | **5.78 s** | **1.1%** | **10.38** | **23.6 / 24.3 ms** |
| `N = 2` | `SyncChunk` (`K = 10`) | 91.7% (11/12) | 9.10 s (9.10 vs. 5.72 s, **1.59x**) | 42.2% | 10.78 | 349.7 / 388.2 ms |
| `N = 2` | `CompiledSync` (`K = 4`) | 100.0% (12/12) | 8.47 s (8.47 vs. 5.84 s, **1.45x**) | 35.9% | 14.16 | 109.8 / 123.0 ms |
| `N = 2` | `PyTorchAsync` (`SDPA`) | 16.7% (2/12) | 8.74 s (8.74 vs. 5.63 s, **1.55x**) | 3.5% | 1.57 | 586.8 / 841.8 ms |
| `N = 2` | **Tempo (`Engine`)** | **100.0% (12/12)** | **5.84 s** | **1.7%** | **20.55** | **39.3 / 46.0 ms** |

### 4.2 Moving-Object [`LIBERO-Dynamic`](https://github.com/renshaojie233/LIBERO-Dynamic) (`24` Scenes at `0.03 / 0.06 / 0.09 m/s`)

| Load | Policy | Overall Success | `0.03 / 0.06 / 0.09 m/s` | Paired Time-to-Success | Goodput (`tasks/min`) |
| :---: | :--- | :---: | :---: | :---: | :---: |
| `N = 1` | `SyncChunk` (`K = 10`) | 54.2% (13/24) | 37.5% / 75.0% / 50.0% | 17.37 vs. **7.28 s** (**2.39x**) | 0.62 |
| `N = 1` | `CompiledSync` (`K = 4`) | 41.7% (10/24) | 62.5% / 37.5% / 25.0% | 13.56 vs. **7.26 s** (**1.87x**) | 0.43 |
| `N = 1` | `PyTorchAsync` (`SDPA`) | 12.5% (3/24) | 25.0% / 0.0% / 12.5% | 7.76 vs. **6.88 s** (**1.13x**) | 0.17 |
| `N = 1` | **Tempo (`Engine`)** | 50.0% (12/24) | **62.5%** / 25.0% / **62.5%** | **11.57 s** mean (`0.2%` stall) | **0.84** |
| `N = 2` | `SyncChunk` (`K = 10`) | 33.3% (8/24) | 37.5% / 37.5% / 25.0% | 19.79 vs. **11.22 s** (**1.76x**) | 0.65 |
| `N = 2` | `CompiledSync` (`K = 4`) | 54.2% (13/24) | 75.0% / 50.0% / 37.5% | 13.99 vs. **9.80 s** (**1.43x**) | 1.25 |
| `N = 2` | `PyTorchAsync` (`SDPA`) | 8.3% (2/24) | 12.5% / 0.0% / 12.5% | 9.60 vs. 11.61 s (1 scene) | 0.22 |
| `N = 2` | **Tempo (`Engine`)** | 45.8% (11/24) | 50.0% / 25.0% / **62.5%** | **11.29 s** mean (`0.3%` stall) | **1.50** |

Note: On `l2` (`0.06 m/s`), `LIBERO-Dynamic`'s multi-obstacle CPU MuJoCo contact substeps ran at `~360 ms/step` (`0.14x` real-time), causing wall-clock horizon clocks (`50 ms/step`) to advance faster than simulated physics; on real-time tiers (`l1 + l3`, `16` scenes), Tempo achieves **62.5% (10/16)** at `N = 1` (vs. 43.8% for both sync baselines) and **62.5% (5/8)** on the fastest `0.09 m/s` tier at both `N = 1` and `N = 2` (**2.35–4.04 tasks/min** goodput vs. **0.82–2.42** for sync).

## Reproduce

```bash
python -m benchmarks.run_suite --model lerobot/smolvla_base --mode colocated --deadline-ms 25 --streams 1,4,8,12,16
python -m benchmarks.run_suite --model nvidia/GR00T-N1.5-3B --mode both --deadline-ms 25 --streams 1,4,8,16,24,32
python -m benchmarks.run_suite --model nvidia/GR00T-N1.5-3B --mode custom --slow-devices 0 --fast-devices 1,2,3 --deadline-ms 25 --streams 4,12,24,32,42,48
python -m benchmarks.run_suite --model lerobot/pi05_base --mode four_gpu_matrix --deadline-ms 30 --streams 4,8,12,16,24,32
python -m benchmarks.eval_libero_dynamic --dyn-repo /path/to/LIBERO-Dynamic --out-json benchmarks/libero_dynamic_eval.json
```
