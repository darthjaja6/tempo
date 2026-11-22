from __future__ import annotations

import argparse
import gc
import json
import math
import pathlib
import random
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from benchmarks.baselines import PyTorchDualStreamBaseline, eager_reference_forward
from tempo import Engine, ModelRegistry
from tempo.models.base import DualSystemDeployment


def percentile(vals: List[float], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = min(len(s) - 1, max(0, int(math.ceil((q / 100.0) * len(s))) - 1))
    return float(s[idx])


def parse_slow_devices(spec: str) -> List[List[int]]:
    replicas: List[List[int]] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if "->" in item:
            replicas.append([int(x.strip()) for x in item.split("->") if x.strip()])
        else:
            replicas.append([int(item)])
    if not replicas:
        raise ValueError(f"Invalid --slow-devices '{spec}'")
    return replicas


def parse_fast_devices(spec: str) -> List[int]:
    devs = [int(x.strip()) for x in spec.split(",") if x.strip()]
    if not devs:
        raise ValueError(f"Invalid --fast-devices '{spec}'")
    return devs


def describe_topology(slow_devices: Sequence[Sequence[int]], fast_devices: Sequence[int]) -> str:
    P = len(slow_devices[0])
    R = len(slow_devices)
    F = len(fast_devices)
    if P == 1 and R == F and all(slow_devices[i][0] == fast_devices[i] for i in range(F)):
        return f"{F}-GPU Colocated Micro-Sliced"
    if P == 1:
        return f"Disaggregated {R}S:{F}F Pre-Push (slow={[p[0] for p in slow_devices]}, fast={list(fast_devices)})"
    return (
        f"Disaggregated {R}xPP({P})S:{F}F Pipeline Pre-Push "
        f"(slow={[list(p) for p in slow_devices]}, fast={list(fast_devices)})"
    )


def count_unique_params(dep: DualSystemDeployment) -> int:
    seen: set[int] = set()
    total = 0
    for stg in dep.slow_replicas[0]:
        for t in list(stg.parameters()) + list(stg.buffers()):
            if t.data_ptr() not in seen and t.ndim >= 1:
                seen.add(t.data_ptr())
                total += t.numel()
    for t in list(dep.fast_shards[0].parameters()) + list(dep.fast_shards[0].buffers()):
        if t.data_ptr() not in seen and t.ndim >= 1:
            seen.add(t.data_ptr())
            total += t.numel()
    return total


def _make_engine_from_deployment(
    dep: DualSystemDeployment,
    *,
    num_streams: int,
    slow_devices: Sequence[Sequence[int]],
    fast_devices: Sequence[int],
    deadline_ms: float,
    slow_rate_hz: float = 2.0,
    fast_rate_hz: float = 20.0,
    max_slice_ms: float = 2.0,
) -> Engine:
    act_slow = [list(p) for p in slow_devices[: min(num_streams, len(slow_devices))]]
    act_fast = list(fast_devices[: min(num_streams, len(fast_devices))])
    eng = Engine(
        num_streams=num_streams,
        action_dim=dep.action_dim,
        slow_obs_shape=dep.slow_obs_shape,
        fast_obs_shape=dep.fast_obs_shape,
        step_dt_ms=dep.step_dt_ms,
        max_q_len=dep.max_q_len,
        max_q_heads=dep.max_q_heads,
    )
    eng.deployment = dep
    P = len(act_slow[0])
    slow_fns = []
    for p_idx in range(P):
        m_map = {act_slow[r][p_idx]: dep.slow_replicas[r][p_idx] for r in range(len(act_slow))}

        def _make_s(mp: Dict[int, nn.Module]):
            def _s(inp, ctx):
                return mp[ctx.device.index](inp, ctx)

            return _s

        slow_fns.append(_make_s(m_map))

    if P == 1:
        eng.stage(
            rate_hz=slow_rate_hz,
            devices=[p[0] for p in act_slow],
            publishes=dep.channel,
            max_slice_ms=max_slice_ms,
            batch_size=4,
        )(slow_fns[0])
    else:
        eng.pipeline(
            slow_fns,
            rate_hz=slow_rate_hz,
            publishes=dep.channel,
            devices=act_slow,
            max_slice_ms=max_slice_ms,
            batch_size=4,
        )

    f_map = {act_fast[f_idx]: dep.fast_shards[f_idx] for f_idx in range(len(act_fast))}

    def _f(fast_obs, state, frozen_prefix):
        return f_map[fast_obs.device.index](fast_obs, state, frozen_prefix)

    eng.stage(
        rate_hz=fast_rate_hz,
        devices=act_fast,
        reads=dep.channel,
        deadline_ms=deadline_ms,
        horizon_ms=dep.chunk_size * dep.step_dt_ms,
    )(_f)
    eng.calibration_info = eng.compile_and_calibrate()
    eng.reset()
    return eng


def verify_bf16_parity(
    dep_bf16: DualSystemDeployment,
    *,
    s_obs: torch.Tensor,
    f_obs: torch.Tensor,
    pref_zero: torch.Tensor,
    slow_devices: Sequence[Sequence[int]],
    fast_devices: Sequence[int],
    deadline_ms: float,
) -> Tuple[Dict[str, Any], torch.Tensor]:
    """Verifies numerical parity between pure Eager reference and Tempo BF16/FP8 Paged TSC + CUDA Graphs."""
    B = s_obs.size(0)

    # Compute pure uncompiled Eager reference BEFORE Tempo compiles dep_bf16.fast_shards
    ref_chunks = eager_reference_forward(
        dep_bf16.slow_replicas[0],
        dep_bf16.fast_shards[0],
        dep_bf16.channel,
        s_obs[:1],
        f_obs[:1],
        pref_zero[:1],
        prefix_valid=torch.zeros((1,), dtype=torch.bool),
    ).float().cpu()

    eng_bf16 = _make_engine_from_deployment(
        dep_bf16,
        num_streams=B,
        slow_devices=slow_devices,
        fast_devices=fast_devices,
        deadline_ms=deadline_ms,
    )

    with eng_bf16:
        eng_bf16.push_slow_batch(s_obs)
        while len(eng_bf16.telemetry().slow_latency_ms) < B:
            time.sleep(0.01)
        t_push = eng_bf16.push_fast_batch(f_obs)
        while eng_bf16.telemetry().fast_requests_served < B:
            time.sleep(0.005)
        act_bf16, _, _, u_bf16 = eng_bf16.sample_actions(t_query_ns=t_push, cpu=True)
    ref_at_eval = ref_chunks[0, 0]
    cos_bf16 = F.cosine_similarity(act_bf16[0].float(), ref_at_eval, dim=0).item()
    calib = dict(eng_bf16.calibration_info or {})
    del eng_bf16

    if cos_bf16 < 0.995:
        raise RuntimeError(f"Tempo BF16 parity check failed: cos_sim={cos_bf16:.6f} < 0.995")

    info = {
        "bf16_splice_cos_sim": round(cos_bf16, 6),
        "fp8_chunk_cos_sim": round(cos_bf16, 6),
        "underruns": int(u_bf16),
        "ref_step0_norm": round(float(ref_chunks[0, 0].norm().item()), 4),
        "calibration": calib,
    }
    return info, ref_chunks


def _run_traffic_point_engine(
    eng: Engine,
    dep: DualSystemDeployment,
    *,
    num_streams: int,
    duration_sec: float,
    deadline_ms: float,
) -> Dict[str, Any]:
    eng.reset()
    s_obs = [
        torch.rand(dep.slow_obs_shape, dtype=torch.bfloat16, device=f"cuda:{eng.active_slow_devices[s % len(eng.active_slow_devices)][0]}")
        for s in range(num_streams)
    ]
    f_obs = [
        torch.randn(dep.fast_obs_shape, dtype=torch.bfloat16, device=f"cuda:{eng.active_fast_devices[s % len(eng.active_fast_devices)]}") * 0.2
        for s in range(num_streams)
    ]

    rng = random.Random(42 + num_streams)
    fast_aois: List[float] = []
    slow_aois: List[float] = []
    sample_underruns = 0

    for d in eng.all_devices:
        torch.cuda.reset_peak_memory_stats(d)

    with eng:
        t_init = eng.now_ns()
        for s in range(num_streams):
            eng.push_slow(s, s_obs[s], t_init)
        time.sleep(min(0.8, 0.12 * math.ceil(num_streams / 4)))

        t_pre = time.monotonic()
        next_fast = [t_pre + rng.uniform(0.0, 0.050) for _ in range(num_streams)]
        while time.monotonic() - t_pre < 0.16:
            now = time.monotonic()
            now_ns = eng.now_ns()
            for s in range(num_streams):
                if now >= next_fast[s]:
                    eng.push_fast(s, f_obs[s], now_ns)
                    next_fast[s] = now + max(0.035, rng.gauss(0.050, 0.003))
            time.sleep(0.001)
        eng.reset_stats()

        t0 = time.monotonic()
        next_slow = [t0 + rng.uniform(0.0, 0.50) for _ in range(num_streams)]
        next_sample = t0

        while True:
            now = time.monotonic()
            if now - t0 >= duration_sec:
                break
            now_ns = eng.now_ns()
            for s in range(num_streams):
                if now >= next_slow[s]:
                    eng.push_slow(s, s_obs[s], now_ns)
                    next_slow[s] = now + max(0.25, rng.gauss(0.50, 0.03))
                if now >= next_fast[s]:
                    eng.push_fast(s, f_obs[s], now_ns)
                    next_fast[s] = now + max(0.035, rng.gauss(0.050, 0.003))
            if now >= next_sample:
                _, fa, sa, u = eng.sample_actions(now_ns, cpu=True)
                fast_aois.extend(float(v) for v in fa[:num_streams].tolist() if v >= 0.0)
                slow_aois.extend(float(v) for v in sa[:num_streams].tolist() if v >= 0.0)
                sample_underruns += int(u)
                next_sample = now + 0.002
            time.sleep(0.0005)

        tel = eng.telemetry()
        rtc = eng.rtc_stats()
        dev_vram = [torch.cuda.max_memory_allocated(d) / 1e9 for d in eng.all_devices]

    ttac = list(tel.fast_ttac_ms)
    commit_underruns = int(rtc.hard_underruns)
    underruns = int(sample_underruns)
    p99_ttac = percentile(ttac, 99.0) if ttac else 9999.0
    p99_s_aoi = percentile(slow_aois, 99.0) if slow_aois else 9999.0
    slo_met = (
        bool(ttac)
        and bool(slow_aois)
        and (p99_ttac <= deadline_ms)
        and (underruns == 0)
        and (commit_underruns == 0)
        and (p99_s_aoi <= 950.0)
    )
    res = {
        "streams": num_streams,
        "fast_ttac_p50_ms": round(percentile(ttac, 50.0), 2),
        "fast_ttac_p99_ms": round(p99_ttac, 2),
        "fast_aoi_p99_ms": round(percentile(fast_aois, 99.0), 2),
        "slow_aoi_p50_ms": round(percentile(slow_aois, 50.0), 2),
        "slow_aoi_p99_ms": round(p99_s_aoi, 2),
        "fast_preemptions": int(tel.fast_preemptions),
        "peak_vram_gb": round(max(dev_vram, default=0.0), 2),
        "cluster_vram_gb": round(sum(dev_vram), 2),
        "hard_underruns": underruns,
        "commit_underruns": commit_underruns,
        "slo_met": slo_met,
    }
    del s_obs, f_obs
    return res


def _run_traffic_point_baseline(
    dep: DualSystemDeployment,
    *,
    num_streams: int,
    slow_devices: Sequence[Sequence[int]],
    fast_devices: Sequence[int],
    duration_sec: float,
    deadline_ms: float,
) -> Dict[str, Any]:
    act_fast = list(fast_devices[: min(num_streams, len(fast_devices))])
    fast_map = {act_fast[i]: dep.fast_shards[i] for i in range(len(act_fast))}
    prefix_steps = min(dep.chunk_size - 1, max(1, int(math.ceil(deadline_ms / dep.step_dt_ms)) + 1))

    base = PyTorchDualStreamBaseline(
        dep.channel,
        action_dim=dep.action_dim,
        horizon_steps=dep.chunk_size,
        prefix_steps=prefix_steps,
        slow_devices=slow_devices,
        fast_devices=fast_devices,
        slow_replicas=dep.slow_replicas,
        fast_experts=fast_map,
        num_streams=num_streams,
        slow_batch_size=4,
        max_fast_batch_size=16,
        deadline_ms=deadline_ms,
        step_dt_ms=dep.step_dt_ms,
    )
    s_obs = [
        torch.rand(dep.slow_obs_shape, dtype=torch.bfloat16, device=f"cuda:{base.active_slow_devs[s % base.R][0]}")
        for s in range(num_streams)
    ]
    f_obs = [
        torch.randn(dep.fast_obs_shape, dtype=torch.bfloat16, device=f"cuda:{base.active_fast_devs[s % base.F]}") * 0.2
        for s in range(num_streams)
    ]

    rng = random.Random(42 + num_streams)
    fast_aois: List[float] = []
    slow_aois: List[float] = []
    sample_underruns = 0

    base.start()
    try:
        t_init = time.monotonic_ns()
        for s in range(num_streams):
            base.push_slow(s, s_obs[s], t_init)
        time.sleep(min(0.8, 0.12 * math.ceil(num_streams / 4)))

        t_pre = time.monotonic()
        next_fast = [t_pre + rng.uniform(0.0, 0.050) for _ in range(num_streams)]
        while time.monotonic() - t_pre < 0.16:
            now = time.monotonic()
            now_ns = time.monotonic_ns()
            for s in range(num_streams):
                if now >= next_fast[s]:
                    base.push_fast(s, f_obs[s], now_ns)
                    next_fast[s] = now + max(0.035, rng.gauss(0.050, 0.003))
            time.sleep(0.001)
        with base.stats_lock:
            base.fast_ttac_ms.clear()
            base.slow_latency_ms.clear()
            base.lock_wait_ms.clear()
            base.commit_hard_underruns = 0

        t0 = time.monotonic()
        next_slow = [t0 + rng.uniform(0.0, 0.50) for _ in range(num_streams)]
        next_sample = t0

        while True:
            now = time.monotonic()
            if now - t0 >= duration_sec:
                break
            now_ns = time.monotonic_ns()
            for s in range(num_streams):
                if now >= next_slow[s]:
                    base.push_slow(s, s_obs[s], now_ns)
                    next_slow[s] = now + max(0.25, rng.gauss(0.50, 0.03))
                if now >= next_fast[s]:
                    base.push_fast(s, f_obs[s], now_ns)
                    next_fast[s] = now + max(0.035, rng.gauss(0.050, 0.003))
            if now >= next_sample:
                _, fa, sa, u = base.sample_actions(now_ns)
                fast_aois.extend(v for v in fa if v >= 0.0)
                slow_aois.extend(v for v in sa if v >= 0.0)
                sample_underruns += int(u)
                next_sample = now + 0.002
            time.sleep(0.0005)
    finally:
        base.stop()

    ttac = list(base.fast_ttac_ms)
    commit_underruns = int(base.commit_hard_underruns)
    underruns = int(sample_underruns)
    p99_ttac = percentile(ttac, 99.0) if ttac else 9999.0
    p99_s_aoi = percentile(slow_aois, 99.0) if slow_aois else 9999.0
    slo_met = (
        bool(ttac)
        and bool(slow_aois)
        and (p99_ttac <= deadline_ms)
        and (underruns == 0)
        and (commit_underruns == 0)
        and (p99_s_aoi <= 950.0)
    )
    res = {
        "streams": num_streams,
        "fast_ttac_p50_ms": round(percentile(ttac, 50.0), 2),
        "fast_ttac_p99_ms": round(p99_ttac, 2),
        "fast_aoi_p99_ms": round(percentile(fast_aois, 99.0), 2),
        "slow_aoi_p50_ms": round(percentile(slow_aois, 50.0), 2),
        "slow_aoi_p99_ms": round(p99_s_aoi, 2),
        "lock_wait_p99_ms": round(percentile(base.lock_wait_ms, 99.0), 2),
        "hard_underruns": underruns,
        "commit_underruns": commit_underruns,
        "slo_met": slo_met,
    }
    del base, s_obs, f_obs
    return res


def _measure_reaction_engine(
    eng: Engine,
    obs_a: torch.Tensor,
    obs_b: torch.Tensor,
    f_zero: torch.Tensor,
    act_zero: torch.Tensor,
    delta_dir: torch.Tensor,
    *,
    num_streams: int,
) -> float:
    eng.reset()
    s_dev = f"cuda:{eng.active_slow_devices[0][0]}"
    bg_s = _render_closed_loop_slow_obs(0.0, obs_a[0].to(s_dev), obs_b[0].to(s_dev))
    step_s = obs_b[0].to(s_dev)
    bg_f = f_zero[0].to(f"cuda:{eng.active_fast_devices[0]}")

    eng_react_ms = 1500.0
    with eng:
        for s in range(num_streams):
            eng.push_slow(s, bg_s)
        t_wait = time.monotonic()
        while len(eng.telemetry().slow_latency_ms) < num_streams and time.monotonic() - t_wait < 2.0:
            time.sleep(0.01)
        t_pre = time.monotonic()
        next_fast = [t_pre + (s / max(1, num_streams)) * 0.050 for s in range(num_streams)]
        while time.monotonic() - t_pre < 0.16:
            now = time.monotonic()
            for s in range(num_streams):
                if now >= next_fast[s]:
                    eng.push_fast(s, bg_f)
                    next_fast[s] = now + 0.050
            time.sleep(0.001)

        t0 = time.monotonic()
        eng.push_slow(0, step_s)
        consec = 0
        t_first_hit = 0.0
        while time.monotonic() - t0 < 1.50:
            now = time.monotonic()
            for s in range(num_streams):
                if now >= next_fast[s]:
                    eng.push_fast(s, bg_f)
                    next_fast[s] = now + 0.050
            acts, _, _, _ = eng.sample_actions(cpu=True)
            u_val = float(((acts[0].float() - act_zero) @ delta_dir).item())
            if u_val >= 0.50:
                if consec == 0:
                    t_first_hit = (now - t0) * 1000.0
                consec += 1
                if consec >= 3:
                    eng_react_ms = t_first_hit
                    break
            else:
                consec = 0
            time.sleep(0.001)
    return eng_react_ms


def _measure_reaction_baseline(
    dep_base: DualSystemDeployment,
    obs_a: torch.Tensor,
    obs_b: torch.Tensor,
    f_zero: torch.Tensor,
    act_zero: torch.Tensor,
    delta_dir: torch.Tensor,
    *,
    num_streams: int,
    slow_devices: Sequence[Sequence[int]],
    fast_devices: Sequence[int],
    deadline_ms: float,
) -> float:
    act_fast = list(fast_devices[: min(num_streams, len(fast_devices))])
    fast_map = {act_fast[i]: dep_base.fast_shards[i] for i in range(len(act_fast))}
    prefix_steps = min(dep_base.chunk_size - 1, max(1, int(math.ceil(deadline_ms / dep_base.step_dt_ms)) + 1))
    base = PyTorchDualStreamBaseline(
        dep_base.channel,
        action_dim=dep_base.action_dim,
        horizon_steps=dep_base.chunk_size,
        prefix_steps=prefix_steps,
        slow_devices=slow_devices,
        fast_devices=fast_devices,
        slow_replicas=dep_base.slow_replicas,
        fast_experts=fast_map,
        num_streams=num_streams,
        deadline_ms=deadline_ms,
        step_dt_ms=dep_base.step_dt_ms,
    )
    s_dev = f"cuda:{base.active_slow_devs[0][0]}"
    bg_s = _render_closed_loop_slow_obs(0.0, obs_a[0].to(s_dev), obs_b[0].to(s_dev))
    step_s = obs_b[0].to(s_dev)
    bg_f = f_zero[0].to(f"cuda:{base.active_fast_devs[0]}")
    base_react_ms = 1500.0
    base.start()
    try:
        t_init = time.monotonic_ns()
        for s in range(num_streams):
            base.push_slow(s, bg_s, t_init)
        time.sleep(min(1.5, 0.25 * math.ceil(num_streams / 4) + 0.25))
        t_pre = time.monotonic()
        next_fast = [t_pre + (s / max(1, num_streams)) * 0.050 for s in range(num_streams)]
        while time.monotonic() - t_pre < 0.16:
            now = time.monotonic()
            tns = time.monotonic_ns()
            for s in range(num_streams):
                if now >= next_fast[s]:
                    base.push_fast(s, bg_f, tns)
                    next_fast[s] = now + 0.050
            time.sleep(0.001)

        t0 = time.monotonic()
        base.push_slow(0, step_s, time.monotonic_ns())
        consec = 0
        t_first_hit = 0.0
        while time.monotonic() - t0 < 1.50:
            now = time.monotonic()
            tns = time.monotonic_ns()
            for s in range(num_streams):
                if now >= next_fast[s]:
                    base.push_fast(s, bg_f, tns)
                    next_fast[s] = now + 0.050
            acts, _, _, _ = base.sample_actions(time.monotonic_ns())
            u_val = float(((acts[0].float().cpu() - act_zero) @ delta_dir).item())
            if u_val >= 0.50:
                if consec == 0:
                    t_first_hit = (now - t0) * 1000.0
                consec += 1
                if consec >= 3:
                    base_react_ms = t_first_hit
                    break
            else:
                consec = 0
            time.sleep(0.001)
    finally:
        base.stop()
    del base
    return base_react_ms


def _make_orthogonal_scenes(
    slow_obs_shape: Sequence[int], gen: torch.Generator
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Constructs two orthogonal macroscopic spatial scenes in [-1, 1] avoiding ViT patch CLT cancellation."""
    if len(slow_obs_shape) >= 3:
        leading = int(math.prod(slow_obs_shape[:-2]))
        h, w = int(slow_obs_shape[-2]), int(slow_obs_shape[-1])
        low_a = torch.randn((leading, 1, 4, 4), generator=gen, dtype=torch.float32)
        low_b = torch.randn((leading, 1, 4, 4), generator=gen, dtype=torch.float32)
        raw_a = F.interpolate(low_a, size=(h, w), mode="bilinear", align_corners=False).view(1, *slow_obs_shape)
        raw_b = F.interpolate(low_b, size=(h, w), mode="bilinear", align_corners=False).view(1, *slow_obs_shape)
    else:
        raw_a = torch.randn((1, *slow_obs_shape), generator=gen, dtype=torch.float32)
        raw_b = torch.randn((1, *slow_obs_shape), generator=gen, dtype=torch.float32)
    flat_a = raw_a.flatten()
    flat_b = raw_b.flatten()
    flat_b = flat_b - (torch.dot(flat_a, flat_b) / max(float(torch.dot(flat_a, flat_a).item()), 1e-8)) * flat_a
    rms = 0.55
    scene_a = (raw_a * (rms / max(float(flat_a.pow(2).mean().sqrt().item()), 1e-6))).clamp(-1.0, 1.0).to(torch.bfloat16)
    scene_b = (flat_b.view_as(raw_b) * (rms / max(float(flat_b.pow(2).mean().sqrt().item()), 1e-6))).clamp(-1.0, 1.0).to(torch.bfloat16)
    return scene_a, scene_b


def _render_closed_loop_slow_obs(
    err_scalar: float,
    scene_neg_dev: torch.Tensor,
    scene_pos_dev: torch.Tensor,
) -> torch.Tensor:
    """Spherical constant-contrast visual scene interpolation in [0, 1] on GPU for alpha in [-1, 1]."""
    alpha = max(-1.0, min(1.0, err_scalar))
    phi = (alpha + 1.0) * (math.pi * 0.25)  # 0 at alpha=-1, pi/4 at alpha=0, pi/2 at alpha=+1
    signed = scene_neg_dev * math.cos(phi) + scene_pos_dev * math.sin(phi)
    return (signed * 0.5 + 0.5).clamp(0.0, 1.0)


def _build_closed_loop_fast_obs(
    base_fast_dev: torch.Tensor,
    q_pos: float,
    q_vel: float,
    err_scalar: float,
) -> torch.Tensor:
    """Encodes live 20 Hz proprioceptive feedback (for state-conditioned heads) or flow seed."""
    if base_fast_dev.ndim == 1:
        out = base_fast_dev.clone()
        n = out.numel()
        if n >= 1:
            out[0] = 0.25 * err_scalar
        if n >= 2:
            out[1] = -0.10 * q_vel
        if n >= 3:
            out[2] = 0.15 * q_pos
        return out
    return base_fast_dev


def _calibrate_closed_loop_axis(
    dep: DualSystemDeployment,
    obs_neg: torch.Tensor,
    obs_pos: torch.Tensor,
    f_ref: torch.Tensor,
    *,
    prefix_steps: int,
    use_rtc_equilibrium: bool,
    target_amp: float = 0.60,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """7-point intra-chunk-regularized linear readout calibration over alpha in [-1, 1]."""
    obs_mid = _render_closed_loop_slow_obs(0.0, obs_neg, obs_pos)
    f_mid = _build_closed_loop_fast_obs(f_ref[0], 0.0, 0.0, 0.0).unsqueeze(0)
    p_zero = torch.zeros((1, prefix_steps, dep.action_dim), dtype=torch.bfloat16)

    if use_rtc_equilibrium:
        a0_init = eager_reference_forward(
            dep.slow_replicas[0], dep.fast_shards[0], dep.channel, obs_mid, f_mid, p_zero
        )[0, prefix_steps:].mean(dim=0)
        p_cond = a0_init.view(1, 1, -1).expand(1, prefix_steps, -1).to(torch.bfloat16).contiguous()
        sl = slice(prefix_steps, min(dep.chunk_size, prefix_steps + 25))
    else:
        p_cond = p_zero
        sl = slice(0, min(dep.chunk_size, 25))

    alphas = [-1.0, -0.66, -0.33, 0.0, 0.33, 0.66, 1.0]
    means: List[torch.Tensor] = []
    intra_res: List[torch.Tensor] = []
    for al in alphas:
        obs_al = _render_closed_loop_slow_obs(al, obs_neg, obs_pos)
        f_al = _build_closed_loop_fast_obs(f_ref[0], 0.0, 0.0, al * target_amp).unsqueeze(0)
        win = eager_reference_forward(
            dep.slow_replicas[0], dep.fast_shards[0], dep.channel, obs_al, f_al, p_cond
        )[0, sl].float().cpu()
        m_al = win.mean(dim=0)
        means.append(m_al)
        intra_res.append(win - m_al.unsqueeze(0))

    act_zero = means[3]
    X = torch.stack([m - act_zero for m in means], dim=0)  # [7, D_act]
    y = torch.tensor(alphas, dtype=torch.float32)  # [7]
    R = torch.cat(intra_res, dim=0)  # [7 * W, D_act]
    D_act = dep.action_dim
    cov_intra = (R.t() @ R) / max(1, R.size(0))
    gram = (X.t() @ X) + 2.0 * cov_intra + 1e-5 * torch.eye(D_act, dtype=torch.float32)
    w_readout = torch.linalg.solve(gram, X.t() @ y)
    # Rescale w_readout so mean(|u(-1)|, |u(+1)|) == 1.0 when non-degenerate
    end_gain = 0.5 * (float((X[6] @ w_readout).item()) - float((X[0] @ w_readout).item()))
    if end_gain > 1e-5:
        w_readout = w_readout / end_gain
    return act_zero, w_readout


def _project_control_cmd(
    act_f: torch.Tensor,
    act_zero: torch.Tensor,
    u_dir: torch.Tensor,
) -> List[float]:
    raw = (act_f - act_zero.unsqueeze(0)) @ u_dir
    return raw.clamp(-2.0, 2.0).tolist()


def _measure_closed_loop_engine(
    eng: Engine,
    obs_neg: torch.Tensor,
    obs_pos: torch.Tensor,
    f_ref: torch.Tensor,
    act_zero: torch.Tensor,
    u_dir: torch.Tensor,
    *,
    num_streams: int,
    episode_sec: float = 4.0,
) -> Dict[str, Any]:
    """500 Hz Clocked Closed-Loop Simulation (never pauses physics for inference)."""
    dt_phys = 0.002
    tau_plant = 0.35
    kp_servo = 3.0 * 0.60
    slow_period_s = 0.25
    target_freq_hz = 0.25
    target_amp = 0.60
    rmse_threshold = 0.32

    eng.reset()
    n_slow_rep = len(eng.active_slow_devices)
    n_fast_shd = len(eng.active_fast_devices)
    scene_neg_by_rep = [obs_neg[0].to(f"cuda:{eng.active_slow_devices[r][0]}") for r in range(n_slow_rep)]
    scene_pos_by_rep = [obs_pos[0].to(f"cuda:{eng.active_slow_devices[r][0]}") for r in range(n_slow_rep)]
    f_base_by_shd = [f_ref[0].to(f"cuda:{eng.active_fast_devices[f]}") for f in range(n_fast_shd)]

    phases = [2.0 * math.pi * (s / max(1, num_streams)) for s in range(num_streams)]
    q_pos = [target_amp * math.sin(phases[s]) for s in range(num_streams)]
    q_vel = [0.0 for _ in range(num_streams)]
    sq_err_sum = [0.0 for _ in range(num_streams)]
    err_steps = 0
    eng_jumps: List[float] = []
    eng_sample_u = 0

    for d in eng.all_devices:
        torch.cuda.reset_peak_memory_stats(d)

    with eng:
        t_init = eng.now_ns()
        for s in range(num_streams):
            r = s % n_slow_rep
            eng.push_slow(s, _render_closed_loop_slow_obs(0.0, scene_neg_by_rep[r], scene_pos_by_rep[r]), t_init)
        time.sleep(min(0.6, 0.12 * math.ceil(num_streams / 4)))
        for _ in range(2):
            t_warm = eng.now_ns()
            for s in range(num_streams):
                f = s % n_fast_shd
                eng.push_fast(s, _build_closed_loop_fast_obs(f_base_by_shd[f], q_pos[s], 0.0, 0.0), t_warm)
            time.sleep(0.035)
        eng.reset_stats()

        prev_act: Optional[torch.Tensor] = None
        t0 = time.monotonic()
        step = 0
        next_slow = [t0 + (s / max(1, num_streams)) * slow_period_s for s in range(num_streams)]
        next_fast = [t0 + (s / max(1, num_streams)) * 0.025 for s in range(num_streams)]

        while True:
            now = time.monotonic()
            t_rel = now - t0
            if t_rel >= episode_sec:
                break
            now_ns = eng.now_ns()
            targets = [target_amp * math.sin(2.0 * math.pi * target_freq_hz * t_rel + phases[s]) for s in range(num_streams)]

            for s in range(num_streams):
                err_s = targets[s] - q_pos[s]
                if now >= next_slow[s]:
                    r = s % n_slow_rep
                    obs_s = _render_closed_loop_slow_obs(err_s / target_amp, scene_neg_by_rep[r], scene_pos_by_rep[r])
                    eng.push_slow(s, obs_s, now_ns)
                    next_slow[s] = now + slow_period_s
                if now >= next_fast[s]:
                    f = s % n_fast_shd
                    f_s = _build_closed_loop_fast_obs(f_base_by_shd[f], q_pos[s], q_vel[s], err_s)
                    eng.push_fast(s, f_s, now_ns)
                    next_fast[s] = now + 0.050

            act, _, _, u = eng.sample_actions(now_ns, cpu=True)
            act_f = act[:num_streams].float()
            eng_sample_u += int(u)
            u_vec = _project_control_cmd(act_f, act_zero, u_dir)

            for s in range(num_streams):
                dq = (kp_servo * float(u_vec[s]) - q_pos[s]) / tau_plant
                q_vel[s] = dq
                q_pos[s] += dq * dt_phys
                if step >= 50:
                    e = targets[s] - q_pos[s]
                    sq_err_sum[s] += e * e

            if step >= 50:
                err_steps += 1
                if prev_act is not None:
                    eng_jumps.append(float((act_f - prev_act).abs().max().item()))
            prev_act = act_f.clone()
            step += 1
            sleep_s = (t0 + step * dt_phys) - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)

        eng_commit_u = int(eng.rtc_stats().hard_underruns)
        eng_underruns = int(eng_sample_u)

    rmses = [math.sqrt(sq / max(1, err_steps)) for sq in sq_err_sum]
    succ = sum(1 for r in rmses if r <= rmse_threshold and eng_underruns == 0 and eng_commit_u == 0)
    return {
        "tracking_rmse": round(sum(rmses) / max(1, len(rmses)), 4),
        "success_rate_pct": round(100.0 * succ / max(1, num_streams), 1),
        "p99_step_jump": round(percentile(eng_jumps, 99.0), 4),
        "max_step_jump": round(max(eng_jumps, default=0.0), 4),
        "underruns": eng_underruns,
        "commit_underruns": eng_commit_u,
    }


def _measure_closed_loop_baseline(
    dep_base: DualSystemDeployment,
    obs_neg: torch.Tensor,
    obs_pos: torch.Tensor,
    f_ref: torch.Tensor,
    act_zero: torch.Tensor,
    u_dir: torch.Tensor,
    *,
    num_streams: int,
    slow_devices: Sequence[Sequence[int]],
    fast_devices: Sequence[int],
    episode_sec: float = 4.0,
    deadline_ms: float = 25.0,
) -> Dict[str, Any]:
    """Identical 500 Hz Clocked Closed-Loop Simulation evaluated on PyTorchDualStreamBaseline."""
    dt_phys = 0.002
    tau_plant = 0.35
    kp_servo = 3.0 * 0.60
    slow_period_s = 0.25
    target_freq_hz = 0.25
    target_amp = 0.60
    rmse_threshold = 0.32

    act_fast = list(fast_devices[: min(num_streams, len(fast_devices))])
    fast_map = {act_fast[i]: dep_base.fast_shards[i] for i in range(len(act_fast))}
    prefix_steps = min(dep_base.chunk_size - 1, max(1, int(math.ceil(deadline_ms / dep_base.step_dt_ms)) + 1))
    base = PyTorchDualStreamBaseline(
        dep_base.channel,
        action_dim=dep_base.action_dim,
        horizon_steps=dep_base.chunk_size,
        prefix_steps=prefix_steps,
        slow_devices=slow_devices,
        fast_devices=fast_devices,
        slow_replicas=dep_base.slow_replicas,
        fast_experts=fast_map,
        num_streams=num_streams,
        deadline_ms=deadline_ms,
        step_dt_ms=dep_base.step_dt_ms,
    )
    scene_neg_by_rep = [obs_neg[0].to(f"cuda:{base.active_slow_devs[r][0]}") for r in range(base.R)]
    scene_pos_by_rep = [obs_pos[0].to(f"cuda:{base.active_slow_devs[r][0]}") for r in range(base.R)]
    f_base_by_shd = [f_ref[0].to(f"cuda:{base.active_fast_devs[f]}") for f in range(base.F)]

    phases = [2.0 * math.pi * (s / max(1, num_streams)) for s in range(num_streams)]
    q_pos = [target_amp * math.sin(phases[s]) for s in range(num_streams)]
    q_vel = [0.0 for _ in range(num_streams)]
    sq_err_sum = [0.0 for _ in range(num_streams)]
    err_steps = 0
    base_jumps: List[float] = []
    base_sample_u = 0

    base.start()
    try:
        t_init = time.monotonic_ns()
        for s in range(num_streams):
            r = s % base.R
            base.push_slow(s, _render_closed_loop_slow_obs(0.0, scene_neg_by_rep[r], scene_pos_by_rep[r]), t_init)
        time.sleep(min(0.6, 0.12 * math.ceil(num_streams / 4)))
        for _ in range(2):
            t_warm = time.monotonic_ns()
            for s in range(num_streams):
                f = s % base.F
                base.push_fast(s, _build_closed_loop_fast_obs(f_base_by_shd[f], q_pos[s], 0.0, 0.0), t_warm)
            time.sleep(0.035)
        with base.stats_lock:
            base.commit_hard_underruns = 0

        prev_act: Optional[torch.Tensor] = None
        t0 = time.monotonic()
        step = 0
        next_slow = [t0 + (s / max(1, num_streams)) * slow_period_s for s in range(num_streams)]
        next_fast = [t0 + (s / max(1, num_streams)) * 0.025 for s in range(num_streams)]

        while True:
            now = time.monotonic()
            t_rel = now - t0
            if t_rel >= episode_sec:
                break
            now_ns = time.monotonic_ns()
            targets = [target_amp * math.sin(2.0 * math.pi * target_freq_hz * t_rel + phases[s]) for s in range(num_streams)]

            for s in range(num_streams):
                err_s = targets[s] - q_pos[s]
                if now >= next_slow[s]:
                    r = s % base.R
                    obs_s = _render_closed_loop_slow_obs(err_s / target_amp, scene_neg_by_rep[r], scene_pos_by_rep[r])
                    base.push_slow(s, obs_s, now_ns)
                    next_slow[s] = now + slow_period_s
                if now >= next_fast[s]:
                    f = s % base.F
                    f_s = _build_closed_loop_fast_obs(f_base_by_shd[f], q_pos[s], q_vel[s], err_s)
                    base.push_fast(s, f_s, now_ns)
                    next_fast[s] = now + 0.050

            act_gpu, _, _, u = base.sample_actions(now_ns)
            act_f = act_gpu.float().cpu()
            base_sample_u += int(u)
            u_vec = _project_control_cmd(act_f, act_zero, u_dir)

            for s in range(num_streams):
                dq = (kp_servo * float(u_vec[s]) - q_pos[s]) / tau_plant
                q_vel[s] = dq
                q_pos[s] += dq * dt_phys
                if step >= 50:
                    e = targets[s] - q_pos[s]
                    sq_err_sum[s] += e * e

            if step >= 50:
                err_steps += 1
                if prev_act is not None:
                    base_jumps.append(float((act_f - prev_act).abs().max().item()))
            prev_act = act_f.clone()
            step += 1
            sleep_s = (t0 + step * dt_phys) - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)

        base_commit_u = int(base.commit_hard_underruns)
        base_underruns = int(base_sample_u)
    finally:
        base.stop()
    del base

    rmses = [math.sqrt(sq / max(1, err_steps)) for sq in sq_err_sum]
    succ = sum(1 for r in rmses if r <= rmse_threshold and base_underruns == 0 and base_commit_u == 0)
    return {
        "tracking_rmse": round(sum(rmses) / max(1, len(rmses)), 4),
        "success_rate_pct": round(100.0 * succ / max(1, num_streams), 1),
        "p99_step_jump": round(percentile(base_jumps, 99.0), 4),
        "max_step_jump": round(max(base_jumps, default=0.0), 4),
        "underruns": base_underruns,
        "commit_underruns": base_commit_u,
    }


def run_full_topology_suite(
    model_id: str,
    slow_devices: List[List[int]],
    fast_devices: List[int],
    stream_counts: List[int],
    duration_sec: float = 4.0,
    deadline_ms: float = 25.0,
    model_kwargs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    topo_desc = describe_topology(slow_devices, fast_devices)
    all_devs = sorted({d for pipe in slow_devices for d in pipe} | set(fast_devices))
    num_gpus = len(all_devs)
    has_fp8 = torch.cuda.get_device_capability(all_devs[0]) >= (8, 9)
    print(f"\n=== Model: {model_id} | Topology: {topo_desc} ({num_gpus} GPU) ===", flush=True)

    # Phase A: Tempo (FP8 Fast on Ada/Hopper, BF16 otherwise) — isolated in GPU VRAM
    dep_tempo = ModelRegistry.build_for_engine(
        model_id,
        slow_devices=slow_devices,
        fast_devices=fast_devices,
        fast_precision="fp8" if has_fp8 else "bf16",
        **(model_kwargs or {}),
    )
    params_b = count_unique_params(dep_tempo) / 1e9
    print(f"  Loaded {model_id} ({params_b:.2f}B active params, mode={dep_tempo.channel.mode})", flush=True)

    prefix_steps = min(
        dep_tempo.chunk_size - 1, max(1, int(math.ceil(deadline_ms / dep_tempo.step_dt_ms)) + 1)
    )
    B_par = max(len(slow_devices), len(fast_devices), 2)
    gen_p = torch.Generator().manual_seed(1234)
    p_s_obs = torch.rand((B_par, *dep_tempo.slow_obs_shape), generator=gen_p, dtype=torch.bfloat16)
    p_f_obs = torch.randn((B_par, *dep_tempo.fast_obs_shape), generator=gen_p, dtype=torch.bfloat16) * 0.2
    p_pref = torch.zeros((B_par, prefix_steps, dep_tempo.action_dim), dtype=torch.bfloat16)
    fp8_chunks = eager_reference_forward(
        dep_tempo.slow_replicas[0],
        dep_tempo.fast_shards[0],
        dep_tempo.channel,
        p_s_obs[:1],
        p_f_obs[:1],
        p_pref[:1],
        prefix_valid=torch.zeros((1,), dtype=torch.bool),
    ).float().cpu()

    gen = torch.Generator().manual_seed(777)
    obs_a, obs_b = _make_orthogonal_scenes(dep_tempo.slow_obs_shape, gen)
    if len(dep_tempo.fast_obs_shape) == 2:
        f_ref = torch.randn((1, *dep_tempo.fast_obs_shape), generator=gen, dtype=torch.bfloat16)
    else:
        f_ref = torch.zeros((1, *dep_tempo.fast_obs_shape), dtype=torch.bfloat16)

    min_topo_n = max(len(slow_devices), len(fast_devices), 4)
    valid_streams = [n for n in stream_counts if n >= max(len(slow_devices), len(fast_devices))]
    max_eng_streams = max([*valid_streams, min_topo_n])

    act_zero_e, u_dir_e = _calibrate_closed_loop_axis(
        dep_tempo, obs_a, obs_b, f_ref, prefix_steps=prefix_steps, use_rtc_equilibrium=True
    )

    eng_shared = _make_engine_from_deployment(
        dep_tempo,
        num_streams=max_eng_streams,
        slow_devices=slow_devices,
        fast_devices=fast_devices,
        deadline_ms=deadline_ms,
    )

    tempo_sweep: List[Dict[str, Any]] = []
    tempo_goodput = 0
    for n in valid_streams:
        r_e = _run_traffic_point_engine(
            eng_shared,
            dep_tempo,
            num_streams=n,
            duration_sec=duration_sec,
            deadline_ms=deadline_ms,
        )
        tempo_sweep.append(r_e)
        if r_e["slo_met"]:
            tempo_goodput = n
        print(
            f"  [Tempo    N={n:2d}] TTAC P99={r_e['fast_ttac_p99_ms']:6.2f}ms | "
            f"Slow AoI P99={r_e['slow_aoi_p99_ms']:6.1f}ms | Underruns={r_e['hard_underruns']:3d} | "
            f"VRAM={r_e['peak_vram_gb']:.2f}GB | SLO={'PASS' if r_e['slo_met'] else 'FAIL'}",
            flush=True,
        )

    probe_n = tempo_goodput if tempo_goodput > 0 else min_topo_n

    eng_react_ms = _measure_reaction_engine(
        eng_shared, obs_a, obs_b, f_ref, act_zero_e, u_dir_e, num_streams=probe_n
    )
    cl_eng = _measure_closed_loop_engine(
        eng_shared, obs_a, obs_b, f_ref, act_zero_e, u_dir_e,
        num_streams=probe_n, episode_sec=4.0,
    )

    del eng_shared, dep_tempo
    gc.collect()
    torch.cuda.empty_cache()

    # Phase B: Pure Eager PyTorch Dual-Stream Baseline FIRST, then BF16/FP8 Parity check
    dep_bf16 = ModelRegistry.build_for_engine(
        model_id,
        slow_devices=slow_devices,
        fast_devices=fast_devices,
        fast_precision="bf16",
        **(model_kwargs or {}),
    )

    baseline_sweep: List[Dict[str, Any]] = []
    base_goodput = 0
    for n in stream_counts:
        if n < max(len(slow_devices), len(fast_devices)):
            continue
        r_b = _run_traffic_point_baseline(
            dep_bf16,
            num_streams=n,
            slow_devices=slow_devices,
            fast_devices=fast_devices,
            duration_sec=duration_sec,
            deadline_ms=deadline_ms,
        )
        baseline_sweep.append(r_b)
        if r_b["slo_met"]:
            base_goodput = n
        print(
            f"  [Baseline N={n:2d}] TTAC P99={r_b['fast_ttac_p99_ms']:6.2f}ms | "
            f"Slow AoI P99={r_b['slow_aoi_p99_ms']:6.1f}ms | Underruns={r_b['hard_underruns']:3d} | "
            f"SLO={'PASS' if r_b['slo_met'] else 'FAIL'}",
            flush=True,
        )

    act_zero_b, u_dir_b = _calibrate_closed_loop_axis(
        dep_bf16, obs_a, obs_b, f_ref, prefix_steps=prefix_steps, use_rtc_equilibrium=False
    )
    base_react_ms = _measure_reaction_baseline(
        dep_bf16, obs_a, obs_b, f_ref, act_zero_b, u_dir_b,
        num_streams=probe_n, slow_devices=slow_devices, fast_devices=fast_devices, deadline_ms=deadline_ms,
    )
    cl_base = _measure_closed_loop_baseline(
        dep_bf16, obs_a, obs_b, f_ref, act_zero_b, u_dir_b,
        num_streams=probe_n, slow_devices=slow_devices, fast_devices=fast_devices,
        episode_sec=4.0, deadline_ms=deadline_ms,
    )

    parity, ref_chunks = verify_bf16_parity(
        dep_bf16,
        s_obs=p_s_obs,
        f_obs=p_f_obs,
        pref_zero=p_pref,
        slow_devices=slow_devices,
        fast_devices=fast_devices,
        deadline_ms=deadline_ms,
    )
    if has_fp8:
        cos_fp8 = F.cosine_similarity(fp8_chunks[0].flatten(), ref_chunks[0].flatten(), dim=0).item()
        if not (cos_fp8 >= 0.985):
            raise RuntimeError(f"Tempo FP8 parity check failed: cos_sim={cos_fp8} < 0.985")
        parity["fp8_chunk_cos_sim"] = round(cos_fp8, 6)

    print(
        f"  [Parity] BF16 splice cos={parity['bf16_splice_cos_sim']:.6f} | "
        f"FP8 chunk cos={parity['fp8_chunk_cos_sim']:.6f}",
        flush=True,
    )

    w2 = {
        "streams": probe_n,
        "engine_reaction_ms": round(eng_react_ms, 2),
        "baseline_reaction_ms": round(base_react_ms, 2),
    }
    w3 = {
        "streams": probe_n,
        "engine_tracking_rmse": cl_eng["tracking_rmse"],
        "engine_success_rate_pct": cl_eng["success_rate_pct"],
        "engine_p99_step_jump": cl_eng["p99_step_jump"],
        "engine_max_step_jump": cl_eng["max_step_jump"],
        "engine_underruns": cl_eng["underruns"],
        "baseline_tracking_rmse": cl_base["tracking_rmse"],
        "baseline_success_rate_pct": cl_base["success_rate_pct"],
        "baseline_p99_step_jump": cl_base["p99_step_jump"],
        "baseline_max_step_jump": cl_base["max_step_jump"],
        "baseline_underruns": cl_base["underruns"],
    }
    print(
        f"  [Reaction N={probe_n}] Tempo={w2['engine_reaction_ms']}ms vs Baseline={w2['baseline_reaction_ms']}ms | "
        f"[Closed-Loop N={probe_n}] Success={w3['engine_success_rate_pct']}% (RMSE={w3['engine_tracking_rmse']}) "
        f"vs Baseline={w3['baseline_success_rate_pct']}% (RMSE={w3['baseline_tracking_rmse']})",
        flush=True,
    )

    del dep_bf16
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "model": model_id,
        "model_params_billions": round(params_b, 3),
        "gpu": torch.cuda.get_device_name(all_devs[0]),
        "num_gpus_used": num_gpus,
        "topology": topo_desc,
        "slow_devices": slow_devices,
        "fast_devices": fast_devices,
        "deadline_ms": deadline_ms,
        "stage0_parity": parity,
        "workload1_traffic": {
            "tempo_slo_goodput": tempo_goodput,
            "tempo_goodput_per_gpu": round(tempo_goodput / num_gpus, 2),
            "baseline_slo_goodput": base_goodput,
            "baseline_goodput_per_gpu": round(base_goodput / num_gpus, 2),
            "tempo_sweep": tempo_sweep,
            "baseline_sweep": baseline_sweep,
        },
        "workload2_step_reaction": w2,
        "workload3_closed_loop": w3,
    }


def main() -> None:
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description="Tempo Real-Model Dual-System Serving Benchmark Suite")
    parser.add_argument("--model", type=str, default="lerobot/pi05_base")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["colocated", "disaggregated", "both", "four_gpu_matrix", "custom"],
        default="colocated",
    )
    parser.add_argument("--slow-devices", type=str, default="0")
    parser.add_argument("--fast-devices", type=str, default="0")
    parser.add_argument("--streams", type=str, default="1,4,8,12,16,20,24,32,40,48")
    parser.add_argument("--duration-sec", type=float, default=4.0)
    parser.add_argument("--deadline-ms", type=float, default=25.0)
    parser.add_argument("--num-cameras", type=int, default=0, help="Override camera count (0 = checkpoint default)")
    parser.add_argument(
        "--out",
        type=str,
        default=str(pathlib.Path(__file__).resolve().parent / "results.json"),
    )
    args = parser.parse_args()
    stream_counts = [int(x.strip()) for x in args.streams.split(",") if x.strip()]
    mkw: Dict[str, Any] = {}
    if args.num_cameras > 0:
        mkw["num_cameras"] = args.num_cameras

    if args.mode == "colocated":
        summary = run_full_topology_suite(
            args.model, [[0]], [0], stream_counts, args.duration_sec, args.deadline_ms, mkw
        )
    elif args.mode == "disaggregated":
        if torch.cuda.device_count() < 2:
            raise RuntimeError(f"--mode disaggregated requires >= 2 GPUs, found {torch.cuda.device_count()}")
        summary = run_full_topology_suite(
            args.model, [[0]], [1], stream_counts, args.duration_sec, args.deadline_ms, mkw
        )
    elif args.mode == "both":
        if torch.cuda.device_count() < 2:
            raise RuntimeError(f"--mode both requires >= 2 GPUs, found {torch.cuda.device_count()}")
        summary = {
            "model": args.model,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_count_visible": torch.cuda.device_count(),
            "colocated_1gpu": run_full_topology_suite(
                args.model, [[0]], [0], stream_counts, args.duration_sec, args.deadline_ms, mkw
            ),
            "disaggregated_2gpu": run_full_topology_suite(
                args.model, [[0]], [1], stream_counts, args.duration_sec, args.deadline_ms, mkw
            ),
        }
    elif args.mode == "four_gpu_matrix":
        if torch.cuda.device_count() < 4:
            raise RuntimeError(f"--mode four_gpu_matrix requires >= 4 GPUs, found {torch.cuda.device_count()}")
        topologies = [
            ("colocated_4gpu", [[0], [1], [2], [3]], [0, 1, 2, 3]),
            ("disagg_2s_2f", [[0], [1]], [2, 3]),
            ("disagg_1s_3f", [[0]], [1, 2, 3]),
            ("disagg_pp2s_2f", [[0, 1]], [2, 3]),
        ]
        summary = {
            "model": args.model,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_count_visible": torch.cuda.device_count(),
        }
        for key, s_devs, f_devs in topologies:
            summary[key] = run_full_topology_suite(
                args.model, s_devs, f_devs, stream_counts, args.duration_sec, args.deadline_ms, mkw
            )
    else:
        s_devs = parse_slow_devices(args.slow_devices)
        f_devs = parse_fast_devices(args.fast_devices)
        summary = run_full_topology_suite(
            args.model, s_devs, f_devs, stream_counts, args.duration_sec, args.deadline_ms, mkw
        )

    out_path = pathlib.Path(args.out)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved benchmark results to {out_path}", flush=True)


if __name__ == "__main__":
    main()
