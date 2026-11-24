# Tempo Benchmarks

## 1. Serving Throughput & Tail Latency

Each stream drives `2 Hz` Slow camera updates, `20 Hz` Fast chunk requests (`FP8` Fast heads on `SM89+`), and `500 Hz` action polling against a `BF16` multi-stream PyTorch `SDPA` + per-stream `RWLock` baseline ([`baselines.py`](./baselines.py)).

- **Max Streams at SLO**: Highest stream count meeting `Chunk P99 <= deadline`, `0` underruns, and `Slow staleness P99 <= 950 ms` (the PyTorch baseline misses the latency/underrun SLO at `1` stream).
- **FP8 Accuracy**: `FP8` Fast heads retain `>0.999` cosine similarity against `BF16` on GR00T-N1.5 and $\pi_{0.5}$, and `>0.993` on SmolVLA.
- **Placement**: `Colocated` runs Slow and Fast on each GPU; `1S : 1F` and `1S : 3F` dedicate 1 GPU to Slow VLM and 1 or 3 GPUs to Fast action heads (`Chunk P99` and `VRAM` measured at `Max Streams at SLO`).

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

## 2. Multi-GPU Placement (`lerobot/pi05_base`, 4 Streams)

All placements below run with `0` underruns:

| Placement (`N = 4`) | `4x H100 SXM` Chunk `P50 / P99` | `4x H100` Peak / Total VRAM | `4x RTX PRO 6000` Chunk `P50 / P99` | `4x RTX PRO 6000` Peak / Total VRAM |
| :--- | :---: | :---: | :---: | :---: |
| `4x Colocated` (`slow=[0..3], fast=[0..3]`) | **20.55 / 26.82 ms** | 5.89 / 23.55 GB | **22.12 / 28.76 ms** | 5.84 / 23.35 GB |
| `2S : 2F` (`slow=[0,1], fast=[2,3]`) | 20.38 / 36.88 ms | 5.85 / 15.69 GB | 23.48 / 42.45 ms | 5.35 / 13.68 GB |
| `1S : 3F` (`slow=[0], fast=[1,2,3]`) | 20.57 / 36.83 ms | 6.42 / **12.68 GB (-46%)** | 22.18 / 39.74 ms | 5.72 / **9.76 GB (-58%)** |
| `2-Stage PP + 2F` (`slow=[[0,1]], fast=[2,3]`) | 20.40 / 36.40 ms | **4.33** / 12.95 GB (-45%) | 22.18 / 39.68 ms | **3.53** / **9.50 GB (-59%)** |

## 3. Real-Time Manipulation

`lerobot/pi05_libero_finetuned_v044` (`1x RTX 4090`, `BF16`, `20 Hz` real-time control via [`inspect-robots`](https://github.com/robocurve/inspect-robots), [`libero_dynamic_eval.json`](./libero_dynamic_eval.json)):

- **`SyncChunk` (`K = 10`)**: Open-loop synchronous inference every 10 steps (`500 ms`).
- **`CompiledSync` (`K = 4`)**: `torch.compile` synchronous inference every 4 steps (`200 ms`).
- **`PyTorchAsync` (`SDPA`)**: Dual-stream async PyTorch (`10 Hz` Slow, `5 Hz` Fast).
- **Tempo**: `Engine` (`10 Hz` Slow, `5 Hz` Fast, `CUDA Graph` + `RTC`).

### Standard `LIBERO` (`12` Scenes: `libero_spatial`, `libero_object`, `libero_goal`)

| Policy | Success | Time-to-Success (Paired vs. Tempo) | Robot Stall | Throughput (`tasks/min`) | Chunk Latency (`P50 / P99`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `SyncChunk` (`K = 10`) | 91.7% (11/12) | 8.95 s (8.95 vs. 5.82 s, **1.54x**) | 40.4% | 5.51 | 328.7 / 375.5 ms |
| `CompiledSync` (`K = 4`) | 91.7% (11/12) | 8.18 s (8.18 vs. 5.79 s, **1.41x**) | 34.4% | 6.02 | 104.2 / 129.2 ms |
| `PyTorchAsync` (`SDPA`) | 66.7% (8/12) | 6.80 s (6.80 vs. 6.18 s, **1.10x**) | 4.2% | 4.44 | 524.9 / 674.5 ms |
| **Tempo** | **100.0% (12/12)** | **5.89 s** | **1.0%** | **10.19** | **23.0 / 24.3 ms** |

### Moving-Object [`LIBERO-Dynamic`](https://github.com/renshaojie233/LIBERO-Dynamic) (`24` Scenes: `0.03 / 0.06 / 0.09 m/s`)

| Policy | Success | `0.03 / 0.06 / 0.09 m/s` | Time-to-Success (Paired vs. Tempo) | Robot Stall | Throughput (`tasks/min`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `SyncChunk` (`K = 10`) | 41.7% (10/24) | 37.5% / 50.0% / 37.5% | 13.04 s (15.23 vs. 6.36 s, **2.39x**) | 39.7% | 1.39 |
| `CompiledSync` (`K = 4`) | 41.7% (10/24) | 37.5% / 37.5% / 50.0% | 12.20 s (11.07 vs. 6.09 s, **1.82x**) | 35.0% | 1.49 |
| `PyTorchAsync` (`SDPA`) | 12.5% (3/24) | 0.0% / 12.5% / 25.0% | 7.56 s (6.09 vs. 6.17 s, 0.99x) | 3.0% | 0.59 |
| **Tempo** | **45.8% (11/24)** | **37.5% / 50.0% / 50.0%** | **6.49 s** | **0.7%** | **2.73** |

## Reproduce

```bash
python -m benchmarks.run_suite --model lerobot/smolvla_base --mode colocated --deadline-ms 25 --streams 1,4,8,12,16
python -m benchmarks.run_suite --model nvidia/GR00T-N1.5-3B --mode both --deadline-ms 25 --streams 1,4,8,16,24,32
python -m benchmarks.run_suite --model nvidia/GR00T-N1.5-3B --mode custom --slow-devices 0 --fast-devices 1,2,3 --deadline-ms 25 --streams 4,12,24,32,42,48
python -m benchmarks.run_suite --model lerobot/pi05_base --mode four_gpu_matrix --deadline-ms 30 --streams 4,8,12,16,24,32
PYTHONPATH=/path/to/LIBERO-Dynamic python -m benchmarks.eval_libero_dynamic --dyn-repo /path/to/LIBERO-Dynamic --out-json benchmarks/libero_dynamic_eval.json
```
