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

## 4. Real-Time Moving-Object Manipulation (`inspect-robots` × `LIBERO-Dynamic`)

Evaluated through [`robocurve/inspect-robots`](https://github.com/robocurve/inspect-robots) (`SELF_PACED` 20 Hz wall-clock MuJoCo physics) on [`LIBERO-Dynamic`](https://github.com/renshaojie233/LIBERO-Dynamic) (`1x H100`, [`libero_dynamic_eval.json`](./libero_dynamic_eval.json)), where target objects move continuously at `0.03`, `0.06`, and `0.09 m/s` (`l1 / l2 / l3`) across 18 dynamic scenes (`libero_spatial`, `libero_object`, `libero_goal`) while GPU inference runs in real wall-clock time:

| Model & Load | Policy | Overall Success | `0.03 / 0.06 / 0.09 m/s` | Latency (`P50 / P99`) | Underruns |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `pi05_libero_finetuned_v044` (`N = 1`) | `SyncChunk` (`K = 10`) | 38.9% (7/18) | 33.3% / 50.0% / 33.3% | 183.1 / 223.0 ms (blocking) | — |
| `pi05_libero_finetuned_v044` (`N = 1`) | `PyTorchAsync` (`SDPA`) | 27.8% (5/18) | 33.3% / 16.7% / 33.3% | 576.2 / 4,394.0 ms (`TTAC`) | 763 |
| `pi05_libero_finetuned_v044` (`N = 1`) | **Tempo (`Engine`)** | **50.0% (9/18)** | **50.0% / 50.0% / 50.0%** | **21.7 / 31.3 ms** (`TTAC`) | **0** |
| `pi05_libero_finetuned_v044` (`N = 2`) | `SyncChunk` (`K = 10`) | 33.3% (6/18) | 50.0% / 33.3% / 16.7% | 188.2 / 224.0 ms (blocking) | — |
| `pi05_libero_finetuned_v044` (`N = 2`) | `PyTorchAsync` (`SDPA`) | 11.1% (2/18) | 0.0% / 16.7% / 16.7% | 463.5 / 4,353.8 ms (`TTAC`) | 1,462 |
| `pi05_libero_finetuned_v044` (`N = 2`) | **Tempo (`Engine`)** | **50.0% (9/18)** | **50.0% / 50.0% / 50.0%** | **31.2 / 43.9 ms** (`TTAC`) | **0** |
| `smolvla_libero` (`N = 4`) | `SyncChunk` / `PyTorchAsync` / **Tempo** | 0.0% (0/18 all) | 0.0% / 0.0% / 0.0% | 127.5 ms / 296.1 ms / **25.6 ms** (`P50`) | — / 1,847 / **0** |

## Reproduce

```bash
python -m benchmarks.run_suite --model lerobot/smolvla_base --mode colocated --deadline-ms 25 --streams 1,4,8,12,16
python -m benchmarks.run_suite --model nvidia/GR00T-N1.5-3B --mode both --deadline-ms 25 --streams 1,4,8,16,24,32
python -m benchmarks.run_suite --model nvidia/GR00T-N1.5-3B --mode custom --slow-devices 0 --fast-devices 1,2,3 --deadline-ms 25 --streams 4,12,24,32,42,48
python -m benchmarks.run_suite --model lerobot/pi05_base --mode four_gpu_matrix --deadline-ms 30 --streams 4,8,12,16,24,32
python -m benchmarks.eval_libero_dynamic --dyn-repo /path/to/LIBERO-Dynamic --out-json benchmarks/libero_dynamic_eval.json
```
