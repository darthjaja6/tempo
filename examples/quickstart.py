"""Single-GPU quickstart: pretrained checkpoints, modular (Slow x Fast) pairing, and custom @stage."""

import argparse
import pathlib
import sys
import time
import torch
import torch.nn as nn

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tempo import (
    DirectActionHead,
    Engine,
    Pi05Config,
    Pi05SlowBackbone,
    TemporalChannel,
    mark_sliceable,
)


def run_pretrained_demo(model_id: str) -> None:
    """Serve a pretrained Hugging Face checkpoint."""
    with Engine(model_id, num_streams=4, prompt="pick up the cup") as engine:
        cams = torch.randint(0, 256, (4, *engine.slow_obs_shape), dtype=torch.uint8)
        fast_in = torch.randn((4, *engine.fast_obs_shape), dtype=torch.bfloat16)

        engine.push_slow_batch(cams)
        while len(engine.telemetry().slow_latency_ms) < engine.num_streams:
            time.sleep(0.005)
        engine.push_fast_batch(fast_in)
        while engine.telemetry().fast_requests_served < engine.num_streams:
            time.sleep(0.002)
        engine.reset_stats()

        t0 = time.monotonic()
        next_fast = t0
        while time.monotonic() - t0 < 0.18:
            now = time.monotonic()
            if now >= next_fast:
                engine.push_fast_batch(fast_in)
                next_fast += 0.050
            actions, fast_aoi_ms, slow_aoi_ms, underruns = engine.sample_actions(cpu=True)
            time.sleep(0.002)

    ready = int(((fast_aoi_ms >= 0) & (slow_aoi_ms >= 0)).sum())
    print(f"[pretrained:{model_id}] actions={tuple(actions.shape)} | ready={ready}/4 | underruns={underruns}")


def run_modular_pairing_demo() -> None:
    """Pair a SlowBackbone with a custom FastHead."""
    cfg = Pi05Config(
        num_cameras=2,
        image_size=112,
        vision_width=256,
        vision_mlp_dim=512,
        vision_layers=2,
        vision_heads=4,
        vlm_width=512,
        vlm_mlp_dim=1024,
        num_layers=4,
        num_heads=4,
        num_kv_heads=1,
        head_dim=256,
        prompt_len=8,
    )
    slow = Pi05SlowBackbone(cfg, device="cuda:0")
    fast = DirectActionHead(slow.channel, action_dim=14, chunk_size=50, state_dim=14, hidden_dim=512)
    prompt_embeds = torch.randn((cfg.prompt_len, cfg.vlm_width), dtype=torch.bfloat16) * 0.02

    with Engine(slow=slow, fast=fast, num_streams=4, prompt=prompt_embeds) as engine:
        engine.push_slow_batch(torch.randint(0, 256, (4, *engine.slow_obs_shape), dtype=torch.uint8))
        while len(engine.telemetry().slow_latency_ms) < engine.num_streams:
            time.sleep(0.002)
        engine.push_fast_batch(torch.randn(4, *engine.fast_obs_shape))
        while engine.telemetry().fast_requests_served < engine.num_streams:
            time.sleep(0.002)
        engine.reset_stats()

        t0 = time.monotonic()
        last_fast = -1.0
        while time.monotonic() - t0 < 0.12:
            now = time.monotonic() - t0
            if now - last_fast >= 0.050:
                engine.push_fast_batch(torch.randn(4, *engine.fast_obs_shape))
                last_fast = now
            actions, fast_aoi_ms, slow_aoi_ms, underruns = engine.sample_actions(cpu=True)
            time.sleep(0.002)

    ready = int(((fast_aoi_ms >= 0) & (slow_aoi_ms >= 0)).sum())
    print(f"[modular] actions={tuple(actions.shape)} | ready={ready}/4 | underruns={underruns}")


def run_custom_stage_demo() -> None:
    """Low-level @engine.stage API for custom execution graphs."""
    channel = TemporalChannel(hidden_dim=256, seq_len=50, mode="hidden")
    slow_net = nn.Sequential(
        mark_sliceable(nn.Linear(256, 256, dtype=torch.bfloat16, device="cuda:0")),
        nn.SiLU(),
        mark_sliceable(nn.Linear(256, 256, dtype=torch.bfloat16, device="cuda:0")),
    ).eval()
    fast_proj = nn.Linear(32 + 256, 7, dtype=torch.bfloat16, device="cuda:0").eval()

    engine = Engine(num_streams=4, action_dim=7, slow_obs_shape=(50, 256), fast_obs_shape=(32,))

    @engine.stage(rate_hz=2.0, publishes=channel, devices=[0], max_slice_ms=2.0)
    def slow_stage(obs, ctx):
        ctx.write_hidden_state(slow_net(obs))

    @engine.stage(rate_hz=20.0, reads=channel, devices=[0], deadline_ms=20.0, horizon_ms=100.0)
    def fast_stage(fast_obs, state, frozen_prefix):
        hidden, _delta_t_sec = state.read_hidden()
        q_target = fast_proj(torch.cat([fast_obs, hidden.mean(dim=1)], dim=-1))
        B, P, A = frozen_prefix.shape
        chunk = q_target.unsqueeze(1).expand(B, engine.horizon_steps, A).clone()
        chunk[:, :P, :] = state.splice_prefix(chunk[:, :P, :], frozen_prefix)
        return chunk

    with engine:
        engine.push_slow_batch(torch.randn(4, 50, 256))
        t0 = time.monotonic()
        last_fast = -1.0
        while time.monotonic() - t0 < 0.12:
            now = time.monotonic() - t0
            if now - last_fast >= 0.050:
                engine.push_fast_batch(torch.randn(4, 32))
                last_fast = now
            actions, fast_aoi_ms, slow_aoi_ms, underruns = engine.sample_actions(cpu=True)
            time.sleep(0.002)

    ready = int(((fast_aoi_ms >= 0) & (slow_aoi_ms >= 0)).sum())
    print(f"[custom @stage] actions={tuple(actions.shape)} | ready={ready}/4 | underruns={underruns}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="", help="Optional Hugging Face model ID (e.g. lerobot/pi05_base)")
    args = parser.parse_args()
    if args.model:
        run_pretrained_demo(args.model)
    else:
        print("Tip: pass --model lerobot/smolvla_base (or lerobot/pi05_base) to run pretrained weights.")
    run_modular_pairing_demo()
    run_custom_stage_demo()
