# Tempo Benchmarks

## 1. Serving Throughput & Tail Latency

Each stream drives `2 Hz` Slow camera updates, `20 Hz` Fast chunk requests (`FP8` Fast heads on `SM89+`), and `500 Hz` action polling against a `BF16` multi-stream PyTorch `SDPA` + per-stream lock baseline ([`baselines.py`](./baselines.py)).

- **Max Streams at SLO**: Highest stream count meeting `Chunk P99 <= deadline`, `0` underruns, and `Slow staleness P99 <= 950 ms` (the PyTorch baseline misses the latency/underrun SLO at `1` stream).
- **FP8 Accuracy**: `FP8` Fast heads retain `>0.999` cosine similarity against `BF16` on GR00T-N1.5 and $\pi_{0.5}$, and `>0.993` on SmolVLA.
- **Placement**: `Colocated` runs Slow and Fast on each GPU; `1S : 1F` and `1S : 3F` dedicate 1 GPU to Slow VLM and 1 or 3 GPUs to Fast action heads (`Chunk P99` and `VRAM` measured at `Max Streams at SLO`).

| Model (Deadline) | Hardware & Placement | Max Streams at SLO | Chunk `P99` (Tempo vs. PyTorch) | Zero-Underrun Capacity | Peak GPU / Total VRAM |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `nvidia/GR00T-N1.5-3B` (25 ms) | `4x H100 SXM` (`1S : 3F`) | **12** | **9.56 ms** vs. 393.8 ms (41.2x) | 48 streams | 2.67 / 6.95 GB |
| `nvidia/GR00T-N1.5-3B` (25 ms) | `2x H100 SXM` (`1S : 1F`) | **8** | **10.38 ms** vs. 125.6 ms (12.1x) | 32 streams | 3.23 / 4.73 GB |
| `nvidia/GR00T-N1.5-3B` (25 ms) | `1x H100 SXM` (`Colocated`) | **8** | **22.12 ms** vs. 150.6 ms (6.8x) | 32 streams | 3.98 / 3.98 GB |
| `lerobot/pi05_base` (30 ms) | `4x H100 SXM` (`Colocated`) | **4** | **26.82 ms** vs. 2,943.4 ms (109.7x) | 4 streams | 5.89 / 23.55 GB |
| `lerobot/pi05_base` (30 ms) | `2x H100 SXM` (`1S : 1F`) | **1** | **20.76 ms** vs. 423.6 ms (20.4x) | 2 streams | 5.85 / 7.18 GB |
| `lerobot/pi05_base` (30 ms) | `1x H100 SXM` (`Colocated`) | **1** | **28.31 ms** vs. 457.4 ms (16.2x) | 2 streams | 6.15 / 6.15 GB |
| `lerobot/pi05_base` (30 ms) | `2x RTX PRO 6000` (`1S : 1F`) | **1** | **23.71 ms** vs. 381.9 ms (16.1x) | 2 streams | 5.35 / 6.63 GB |
| `lerobot/pi05_base` (30 ms) | `1x RTX PRO 6000` (`Colocated`) | **1** | **28.82 ms** vs. 394.9 ms (13.7x) | 2 streams | 6.10 / 6.10 GB |
| `lerobot/pi05_base` (35 ms) | `1x RTX 4090` (`Colocated`) | **1** | **23.90 ms** vs. 271.6 ms (11.4x) | 2 streams | 5.49 / 5.49 GB |
| `lerobot/smolvla_base` (25 ms) | `2x RTX PRO 6000` (`1S : 1F`) | **12** | **22.98 ms** vs. 292.1 ms (12.7x) | 16 streams | 1.05 / 1.53 GB |
| `lerobot/smolvla_base` (25 ms) | `1x RTX PRO 6000` (`Colocated`) | **12** | **24.11 ms** vs. 360.5 ms (15.0x) | 16 streams | 1.19 / 1.19 GB |
| `lerobot/smolvla_base` (25 ms) | `1x RTX 4090` (`Colocated`) | **8** | **24.36 ms** vs. 136.1 ms (5.6x) | 12 streams | 0.89 / 0.89 GB |

## 2. Multi-GPU Placement (`lerobot/pi05_base`, 4 Streams)

All placements below run with `0` underruns:

| Placement (`N = 4`) | `4x H100 SXM` Chunk `P50 / P99` | `4x H100` Peak / Total VRAM | `4x RTX PRO 6000` Chunk `P50 / P99` | `4x RTX PRO 6000` Peak / Total VRAM |
| :--- | :---: | :---: | :---: | :---: |
| `4x Colocated` (`slow=[0..3], fast=[0..3]`) | **20.55 / 26.82 ms** | 5.89 / 23.55 GB | **22.12 / 28.76 ms** | 5.84 / 23.35 GB |
| `2S : 2F` (`slow=[0,1], fast=[2,3]`) | 20.38 / 36.88 ms | 5.85 / 15.69 GB | 23.48 / 42.45 ms | 5.35 / 13.68 GB |
| `1S : 3F` (`slow=[0], fast=[1,2,3]`) | 20.57 / 36.83 ms | 6.42 / **12.68 GB (-46%)** | 22.18 / 39.74 ms | 5.72 / **9.76 GB (-58%)** |
| `2-Stage PP + 2F` (`slow=[[0,1]], fast=[2,3]`) | 20.40 / 36.40 ms | **4.33** / 12.95 GB (-45%) | 22.18 / 39.68 ms | **3.53** / **9.50 GB (-59%)** |

## 3. Real-Time Manipulation

`lerobot/pi05_libero_finetuned_v044` (`1x RTX 4090`, `BF16`, `20 Hz` real-time control via [`inspect-robots`](https://github.com/robocurve/inspect-robots), same checkpoint with zero retraining; raw episode logs in [`libero_dynamic_eval.json`](./libero_dynamic_eval.json)):

- **Synchronous** (`K = 10`): Blocking open-loop chunk inference every 10 steps (`500 ms`).
- **`torch.compile` Sync** (`K = 4`): Compiled blocking chunk inference every 4 steps (`200 ms`).
- **Python Async**: Dual-stream async PyTorch (`10 Hz` Slow, `5 Hz` Fast, `SDPA` + per-stream lock).
- **Tempo**: Lock-free C++/CUDA dual-stream serving (`10 Hz` Slow, `5 Hz` Fast, `CUDA Graph` + `RTC`). `Robot Stall` measures the fraction of episode time the robot stands frozen waiting for blocking inference.

### Moving-Object [`LIBERO-Dynamic`](https://github.com/renshaojie233/LIBERO-Dynamic) (`360` Episodes: `40` Tasks × `3` Motion Families × `0.03 / 0.06 / 0.09 m/s`)

Tempo improves overall moving-object success rate with statistical significance under paired McNemar testing (`p = 0.011` vs. `Synchronous`, `p = 0.0052` vs. `torch.compile Sync`, `p < 1e-10` vs. `Python Async`) while leading across every speed tier and motion family:

| Deployment | Success | `0.03 / 0.06 / 0.09 m/s` | Time-to-Success (Paired vs. Tempo) | Robot Stall | Throughput (`tasks/min`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Synchronous (`K = 10`) | 31.7% (114/360) | 35.8% / 30.8% / 28.3% | 17.82 s (17.07 vs. 9.32 s, **1.83x**) | 40.0% | 0.48 |
| `torch.compile` Sync (`K = 4`) | 30.6% (110/360) | 38.3% / 25.8% / 27.5% | 18.66 s (15.76 vs. 9.25 s, **1.70x**) | 35.8% | 0.48 |
| Python Async | 7.2% (26/360) | 6.7% / 6.7% / 8.3% | 8.37 s (6.77 vs. 6.56 s, 1.03x) | 1.6% | 0.15 |
| **Tempo** | **39.2% (141/360)** | **46.7% / 35.0% / 35.8%** | **10.35 s** | **0.3%** | **1.05 (2.19x)** |

### Standard `LIBERO` (`400` Episodes: `40` Tasks × `10` Trials Across `spatial`, `object`, `goal`, `10`)

| Deployment | Success (`spatial / object / goal / 10`) | Time-to-Success (Paired vs. Tempo) | Robot Stall | Throughput (`tasks/min`) | Chunk Latency (`P50 / P99`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Synchronous (`K = 10`) | 97.2% (`98 / 100 / 95 / 96%`) | 12.20 s (12.04 vs. 7.55 s, **1.60x**) | 41.2% | 4.61 | 376.4 / 488.6 ms |
| `torch.compile` Sync (`K = 4`) | 96.2% (`99 / 100 / 90 / 96%`) | 11.20 s (11.05 vs. 7.49 s, **1.48x**) | 35.7% | 4.89 | 115.5 / 164.5 ms |
| Python Async | 49.0% (`44 / 46 / 58 / 48%`) | 7.66 s (7.49 vs. 7.32 s, 1.02x) | 3.6% | 2.37 | 578.5 / 828.8 ms |
| **Tempo** | **96.0%** (`97 / 98 / 96 / 93%`) | **7.56 s** | **0.8%** | **7.18 (1.47–1.56x)** | **24.2 / 25.1 ms** |

## Reproduce

```bash
python -m benchmarks.run_suite --model lerobot/smolvla_base --mode colocated --deadline-ms 25 --streams 1,4,8,12,16
python -m benchmarks.run_suite --model nvidia/GR00T-N1.5-3B --mode both --deadline-ms 25 --streams 1,4,8,16,24,32
python -m benchmarks.run_suite --model nvidia/GR00T-N1.5-3B --mode custom --slow-devices 0 --fast-devices 1,2,3 --deadline-ms 25 --streams 4,12,24,32,42,48
python -m benchmarks.run_suite --model lerobot/pi05_base --mode four_gpu_matrix --deadline-ms 30 --streams 4,8,12,16,24,32
python -m benchmarks.eval_libero_dynamic --dyn-repo /path/to/LIBERO-Dynamic --out-json benchmarks/libero_dynamic_eval.json
```
