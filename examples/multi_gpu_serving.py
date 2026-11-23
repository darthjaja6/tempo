"""Multi-GPU serving (colocated, disaggregated, pipeline-parallel) and edge-cloud split serving."""

import argparse
import pathlib
import sys
import time
import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tempo import Engine, RemoteSlowClient, RemoteSlowServer, TemporalChannel


def run_multi_gpu_topology(model_id: str, topology: str) -> None:
    n_gpus = torch.cuda.device_count()
    if topology == "1s3f":
        if n_gpus < 4:
            raise RuntimeError(f"topology='1s3f' requires >= 4 CUDA GPUs, found {n_gpus}")
        slow_devs, fast_devs = [0], [1, 2, 3]
    elif topology == "pp":
        if n_gpus < 4:
            raise RuntimeError(f"topology='pp' requires >= 4 CUDA GPUs, found {n_gpus}")
        slow_devs, fast_devs = [[0, 1]], [2, 3]
    elif topology == "disagg2":
        if n_gpus < 2:
            raise RuntimeError(f"topology='disagg2' requires >= 2 CUDA GPUs, found {n_gpus}")
        slow_devs, fast_devs = [0], [1]
    elif topology == "colocated":
        if n_gpus < 1:
            raise RuntimeError("topology='colocated' requires >= 1 CUDA GPU")
        devs = list(range(min(n_gpus, 4)))
        slow_devs, fast_devs = devs, devs
    else:
        raise ValueError(f"Unsupported topology '{topology}'")

    with Engine(
        model_id,
        num_streams=8,
        slow_devices=slow_devs,
        fast_devices=fast_devs,
        fast_precision="fp8" if torch.cuda.get_device_capability(0) >= (8, 9) else "bf16",
        prompt="fold the towel neatly",
    ) as engine:
        cams = torch.randint(0, 256, (8, *engine.slow_obs_shape), dtype=torch.uint8)
        fast_in = torch.randn((8, *engine.fast_obs_shape), dtype=torch.bfloat16)

        engine.push_slow_batch(cams)
        while True:
            engine.push_fast_batch(fast_in)
            time.sleep(0.010)
            _, fast_age_ms, _, _ = engine.sample_actions(cpu=True)
            if bool((fast_age_ms >= 0).all()):
                engine.reset_stats()
                break

        t0 = time.monotonic()
        next_fast = t0 + 0.020
        total_underruns = 0
        while time.monotonic() - t0 < 0.18:
            now = time.monotonic()
            if now >= next_fast:
                engine.push_fast_batch(fast_in)
                next_fast += 0.040
            actions, fast_age_ms, slow_age_ms, tick_underruns = engine.sample_actions(cpu=True)
            total_underruns += int(tick_underruns)
            time.sleep(0.002)

        calib = engine.calibration_info or {}

    ready = int(((fast_age_ms >= 0) & (slow_age_ms >= 0)).sum())
    print(
        f"[topology={calib.get('mode')}] model={model_id} slow={calib.get('slow_devices')} "
        f"fast={calib.get('fast_devices')} | actions={tuple(actions.shape)} | ready={ready}/8 | underruns={total_underruns}"
    )


def run_edge_cloud_demo() -> None:
    channel = TemporalChannel(hidden_dim=128, seq_len=32, mode="hidden")
    robot_engine = Engine(num_streams=1, action_dim=7, fast_obs_shape=(16,), step_dt_ms=2.0)
    head = torch.nn.Linear(16 + 128, 7, dtype=torch.bfloat16, device="cuda:0").eval()

    @robot_engine.stage(rate_hz=20.0, reads=channel, devices=[0], deadline_ms=20.0, horizon_ms=100.0)
    def on_robot_fast(fast_obs, state, frozen_prefix):
        hidden, _dt = state.read_hidden()
        q_target = head(torch.cat([fast_obs, hidden.mean(dim=1)], dim=-1))
        B, P, A = frozen_prefix.shape
        chunk = q_target.unsqueeze(1).expand(B, robot_engine.horizon_steps, A).clone()
        chunk[:, :P, :] = state.splice_prefix(chunk[:, :P, :], frozen_prefix)
        return chunk

    def cloud_vlm_fn(batch_obs: torch.Tensor) -> torch.Tensor:
        return batch_obs.mean(dim=-1, keepdim=True).expand(batch_obs.size(0), 32, 128).contiguous()

    with RemoteSlowServer(cloud_vlm_fn, host="127.0.0.1", port=0) as srv, robot_engine:
        with RemoteSlowClient(robot_engine, host="127.0.0.1", port=srv.port) as client:
            client.push_slow(0, torch.randn(32, 64))
            while True:
                robot_engine.push_fast(0, torch.randn(16))
                time.sleep(0.005)
                actions, fast_age_ms, slow_age_ms, underruns = robot_engine.sample_actions(cpu=True)
                if float(fast_age_ms[0]) >= 0.0:
                    break

    print(
        f"[edge-cloud] actions={tuple(actions.shape)} | slow_age={float(slow_age_ms[0]):.1f} ms | "
        f"fast_age={float(fast_age_ms[0]):.1f} ms | underruns={underruns}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="lerobot/pi05_base")
    parser.add_argument("--topology", choices=["auto", "colocated", "disagg2", "1s3f", "pp"], default="auto")
    parser.add_argument("--edge-only", action="store_true")
    args = parser.parse_args()

    if not args.edge_only:
        n = torch.cuda.device_count()
        topo = ("1s3f" if n >= 4 else ("disagg2" if n >= 2 else "colocated")) if args.topology == "auto" else args.topology
        run_multi_gpu_topology(args.model, topo)
    run_edge_cloud_demo()
