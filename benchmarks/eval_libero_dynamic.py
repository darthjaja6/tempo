#!/usr/bin/env python3
"""
Real-time manipulation benchmark via robocurve/inspect-robots on standard LIBERO & LIBERO-Dynamic.

Measures how inference latency impacts both Task Completion Time (Time-to-Success, stall ratio,
and Throughput in tasks/min/GPU) and Dynamic-Target Success Rate on identical pretrained VLA
weights (`lerobot/pi05_libero_finetuned_v044`, 3.3B BF16) across 4 deployment strategies:
  1. `SyncChunk (K=10)`   : Standard synchronous open-loop chunking (10 steps = 500 ms per chunk)
  2. `CompiledSync (K=4)` : `torch.compile`-accelerated synchronous reactive chunking (4 steps = 200 ms)
  3. `PyTorchAsync (SDPA)`: Asynchronous Python dual-stream (`SDPA` + per-stream lock, 10 Hz Slow / 5 Hz Fast)
  4. `Tempo (Engine)`     : `tempo.Engine` (C++ scheduler + <2 ms VLM slicing + paged KV + CUDA Graph + RTC)
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import os
import pathlib
import shutil
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np
import torch
import yaml

torch.set_num_threads(2)
import torch.nn.functional as F
from safetensors.torch import load_file

from inspect_robots.embodiment import (
    PRIVILEGED_SUCCESS,
    RESETTABLE,
    SELF_PACED,
    EmbodimentBase,
    EmbodimentInfo,
)
from inspect_robots.eval import eval as inspect_eval
from inspect_robots.policy import PolicyBase, PolicyConfig, PolicyInfo
from inspect_robots.scene import Scene
from inspect_robots.spaces import ActionSemantics, Box, CameraSpec, ObservationSpace
from inspect_robots.task import Task
from inspect_robots.types import Action, ActionChunk, Observation, StepResult

from benchmarks.baselines import (
    EagerCaptureContext,
    EagerDenseStateView,
    PyTorchDualStreamBaseline,
)
from tempo import Engine
from tempo.models.base import ModelRegistry
from tempo.models.hub import load_hf_tokenizer, resolve_checkpoint
from tempo.models.pi05 import discretize_pi05_state

legacy: Any = None
get_libero_path: Any = None
OffScreenRenderEnv: Any = None
build_dyn_parser: Any = None
RealtimeEpisode: Any = None


def _ensure_dyn_imports(dyn_repo: pathlib.Path) -> None:
    global legacy, get_libero_path, OffScreenRenderEnv, build_dyn_parser, RealtimeEpisode
    if legacy is not None:
        return
    root = dyn_repo.expanduser().resolve()
    for sub in (root / "scripts", root):
        s = str(sub)
        if sub.exists() and s not in sys.path:
            sys.path.insert(0, s)
    import eval_dynamic_libero_style as _legacy
    from libero.libero import get_libero_path as _get_libero_path
    from libero.libero.envs import OffScreenRenderEnv as _OffScreenRenderEnv
    from realtime_eval.runner import build_parser as _build_dyn_parser
    from realtime_eval.runtime import RealtimeEpisode as _RealtimeEpisode

    legacy = _legacy
    get_libero_path = _get_libero_path
    OffScreenRenderEnv = _OffScreenRenderEnv
    build_dyn_parser = _build_dyn_parser
    RealtimeEpisode = _RealtimeEpisode


ACTION_SPACE = Box(
    shape=(7,),
    low=-np.ones((7,), dtype=np.float32),
    high=np.ones((7,), dtype=np.float32),
    semantics=ActionSemantics(
        control_mode="eef_delta_pose",
        rotation_repr="axis_angle",
        gripper="continuous",
        frame="base",
    ),
)

OBS_SPACE = ObservationSpace(
    cameras=(
        CameraSpec(name="agentview", height=256, width=256, channels=3),
        CameraSpec(name="wrist", height=256, width=256, channels=3),
    ),
    state_keys=("eef_state",),
)

STATIC_SUITE_MAX_STEPS: Dict[str, int] = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
}

CANONICAL_SUITES: Tuple[str, ...] = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_10",
)

CANONICAL_SPEEDS: Tuple[Tuple[str, float], ...] = (
    ("003", 0.03),
    ("006", 0.06),
    ("009", 0.09),
)


def load_mean_std_stats(
    ckpt_dir: pathlib.Path,
    preproc_fname: str,
    postproc_fname: str,
) -> Dict[str, torch.Tensor]:
    pre = load_file(str(ckpt_dir / preproc_fname), device="cpu")
    post = load_file(str(ckpt_dir / postproc_fname), device="cpu")
    return {
        "state_mean": pre["observation.state.mean"].float(),
        "state_std": pre["observation.state.std"].float(),
        "action_mean": post["action.mean"].float(),
        "action_std": post["action.std"].float(),
    }


def resize_square_rgb(img_hwc_uint8: np.ndarray, target_hw: int) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(img_hwc_uint8)).permute(2, 0, 1).unsqueeze(0).float()
    if t.shape[-2] != target_hw or t.shape[-1] != target_hw:
        t = F.interpolate(t, size=(target_hw, target_hw), mode="bilinear", align_corners=False)
    return t.squeeze(0).clamp(0.0, 255.0).to(torch.uint8).contiguous()


def quat2axisangle_np(quat: np.ndarray) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).copy()
    q[3] = np.clip(q[3], -1.0, 1.0)
    den = np.sqrt(max(0.0, 1.0 - q[3] * q[3]))
    if math.isclose(den, 0.0):
        return np.zeros(3, dtype=np.float32)
    return ((q[:3] * 2.0 * math.acos(q[3])) / den).astype(np.float32)


def extract_libero_obs(raw_obs: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    img_main = np.ascontiguousarray(raw_obs["agentview_image"][::-1, ::-1])
    img_wrist = np.ascontiguousarray(raw_obs["robot0_eye_in_hand_image"][::-1, ::-1])
    eef_pos = np.asarray(raw_obs["robot0_eef_pos"], dtype=np.float32)
    eef_aa = quat2axisangle_np(np.asarray(raw_obs["robot0_eef_quat"], dtype=np.float32))
    grip = np.asarray(raw_obs["robot0_gripper_qpos"], dtype=np.float32)
    state_8d = np.concatenate([eef_pos, eef_aa, grip], axis=0).astype(np.float32)
    return img_main, img_wrist, state_8d


def _summarize_ms_list(vals: Sequence[float]) -> Dict[str, Optional[float]]:
    if not vals:
        return {"mean_ms": None, "p50_ms": None, "p99_ms": None}
    arr = np.asarray(vals, dtype=np.float64)
    return {
        "mean_ms": round(float(np.mean(arr)), 2),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
    }


class LiberoRealtimeEmbodiment(EmbodimentBase):
    """Real-time 20 Hz wall-clock LIBERO & LIBERO-Dynamic MuJoCo embodiment for inspect-robots.

    Records exact per-episode 1.0x physical robot completion time, active execution time,
    blocking inference stall time, host MuJoCo physics time, per-episode policy latency stats,
    and moving-target physical displacement.
    """

    def __init__(
        self,
        items_by_id: Dict[str, Dict[str, Any]],
        dyn_args_by_speed: Dict[str, argparse.Namespace],
    ) -> None:
        self._items_by_id = items_by_id
        self._dyn_args_by_speed = dyn_args_by_speed
        any_args = next(iter(dyn_args_by_speed.values()))
        self._default_dyn_args = any_args
        self._control_hz = float(any_args.control_freq)
        self._dt_s = 1.0 / self._control_hz
        self.info = EmbodimentInfo(
            name="libero-realtime-franka-20hz",
            action_space=ACTION_SPACE,
            observation_space=OBS_SPACE,
            control_hz=self._control_hz,
            is_simulated=True,
            capabilities=frozenset({RESETTABLE, PRIVILEGED_SUCCESS, SELF_PACED}),
            morphology="franka-panda-7dof",
        )
        self._ep: Optional[RealtimeEpisode] = None
        self._env: Optional[OffScreenRenderEnv] = None
        self._active_policy: Optional[PolicyBase] = None
        self._on_episode_record: Optional[Callable[[Dict[str, Any]], None]] = None
        self._is_dynamic: bool = False
        self._current_scene_meta: Dict[str, Any] = {}
        self._instruction: str = ""
        self._ep_start_t: float = 0.0
        self._last_step_end_t: float = 0.0
        self._obs_ready_t: float = 0.0
        self._ep_paused_s: float = 0.0
        self._step_count: int = 0
        self._ep_stall_s: float = 0.0
        self._ep_max_disp_m: float = 0.0
        self._ep_success: bool = False
        self._ep_recorded: bool = True
        self._target_init_pos: Dict[str, np.ndarray] = {}
        self.episode_records: List[Dict[str, Any]] = []

    def bind_policy(
        self,
        policy: Optional[PolicyBase],
        on_episode_record: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self._active_policy = policy
        self._on_episode_record = on_episode_record
        if policy is not None and hasattr(policy, "set_reset_hook"):
            policy.set_reset_hook(self.notify_policy_reset_done)

    def notify_policy_reset_done(self) -> None:
        if self._step_count == 0 and not self._ep_recorded:
            now = time.perf_counter()
            self._ep_start_t = now
            self._last_step_end_t = now
            self._obs_ready_t = now
            self._ep_paused_s = 0.0

    @contextlib.contextmanager
    def _paused_sim_clock(self):
        """Pauses episode & policy clocks during CPU MuJoCo physics/rendering so robot time stays 1.0x real-time."""
        t_p0 = time.perf_counter()
        pol = self._active_policy
        if pol is not None and hasattr(pol, "pause_clock"):
            pol.pause_clock()
        try:
            yield
        finally:
            if pol is not None and hasattr(pol, "resume_clock"):
                pol.resume_clock()
            self._ep_paused_s += max(0.0, time.perf_counter() - t_p0)

    def clear_records(self) -> None:
        self._finalize_episode_if_needed()
        self.episode_records.clear()

    def _finalize_episode_if_needed(self) -> None:
        if self._ep_recorded or self._step_count == 0:
            return
        raw_wall_s = max(0.0, self._last_step_end_t - self._ep_start_t)
        wall_s = max(0.0, raw_wall_s - self._ep_paused_s)
        stall_s = min(wall_s, max(0.0, self._ep_stall_s))
        active_exec_s = max(0.0, wall_s - stall_s)

        pol_tel: Dict[str, Any] = {}
        if self._active_policy is not None and hasattr(self._active_policy, "pop_episode_telemetry"):
            pol_tel = self._active_policy.pop_episode_telemetry()

        rec: Dict[str, Any] = {
            "scene_id": str(self._current_scene_meta.get("scene_id", "")),
            "group": str(self._current_scene_meta.get("group", "")),
            "suite": str(self._current_scene_meta.get("suite", "")),
            "task_id": int(self._current_scene_meta.get("task_id", 0)),
            "level": str(self._current_scene_meta.get("level", "static")),
            "speed_mps": float(self._current_scene_meta.get("speed_mps", 0.0)),
            "traj_index": int(self._current_scene_meta.get("traj_index", -1)),
            "init_idx": int(self._current_scene_meta.get("init_idx", 0)),
            "success": bool(self._ep_success),
            "steps": int(self._step_count),
            "max_steps": int(self._current_scene_meta.get("max_steps", 0)),
            "wall_time_s": round(wall_s, 4),
            "active_exec_time_s": round(active_exec_s, 4),
            "stall_time_s": round(stall_s, 4),
            "sim_host_wall_s": round(raw_wall_s, 4),
            "sim_paused_s": round(self._ep_paused_s, 4),
            "max_target_disp_m": round(self._ep_max_disp_m, 4),
            **pol_tel,
        }
        self.episode_records.append(rec)
        self._ep_recorded = True
        if self._on_episode_record is not None:
            self._on_episode_record(rec)

    def _close_env(self) -> None:
        self._finalize_episode_if_needed()
        if self._env is not None:
            self._env.close()
            self._env = None
        self._ep = None

    def close(self) -> None:
        self._close_env()

    def _measure_target_disp(self) -> None:
        if self._env is None:
            return
        for b_id in range(int(self._env.sim.model.nbody)):
            name = str(self._env.sim.model.body_id2name(b_id) or "")
            if not name or name.startswith(("world", "table", "robot0", "gripper0", "mount0", "floor")):
                continue
            pos = np.asarray(self._env.sim.data.body_xpos[b_id], dtype=np.float64).copy()
            if name not in self._target_init_pos:
                self._target_init_pos[name] = pos
            else:
                disp = float(np.linalg.norm(pos[:3] - self._target_init_pos[name][:3]))
                if disp < 1.5 and disp > self._ep_max_disp_m:
                    self._ep_max_disp_m = disp

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        self._close_env()
        gc.collect()
        meta = dict(scene.metadata)
        meta["scene_id"] = scene.id
        self._current_scene_meta = meta
        self._is_dynamic = bool(meta.get("dynamic", True))

        if self._is_dynamic:
            scene_key = str(meta["scene_key"])
            speed_label = str(meta["speed_label"])
            item = self._items_by_id[scene_key]
            dyn_args = self._dyn_args_by_speed[speed_label]
            ep = RealtimeEpisode(item, dyn_args, worker_slot=0)
            ep._configure_env_selection()
            suite, task, bddl = legacy.task_bddl(item["suite"], item["task_id"])
            init_states = suite.get_task_init_states(item["task_id"])

            env = OffScreenRenderEnv(
                bddl_file_name=str(bddl),
                camera_heights=256,
                camera_widths=256,
                control_freq=int(dyn_args.control_freq),
            )
            eff_seed = int(dyn_args.seed if seed is None else seed)
            env.seed(eff_seed)
            init_idx = int(dyn_args.init_state_id) % len(init_states)
            raw_obs, *_ = ep._setup_world(env, init_states[init_idx])
            self._ep = ep
        else:
            for env_k in (
                "LIBERO_DYNAMIC_CONFIG",
                "LIBERO_DYNAMIC_TRAJ_INDEX",
                "LIBERO_DYNAMIC_VARIANT_ID",
                "LIBERO_DYNAMIC_VARIANT",
            ):
                os.environ.pop(env_k, None)
            os.environ["LIBERO_DYNAMIC_AUTO_FINALIZE"] = "false"
            suite_name = str(meta["suite"])
            task_id = int(meta["task_id"])
            init_idx_raw = int(meta["init_idx"])
            suite, task, _ = legacy.task_bddl(suite_name, task_id)
            init_states = suite.get_task_init_states(task_id)
            bddl_path = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
            env = OffScreenRenderEnv(
                bddl_file_name=bddl_path,
                camera_heights=256,
                camera_widths=256,
                control_freq=int(self._default_dyn_args.control_freq),
            )
            eff_seed = int(0 if seed is None else seed)
            env.seed(eff_seed)
            env.reset()
            raw_obs = env.set_init_state(init_states[init_idx_raw % len(init_states)])
            open_grip = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]
            for _ in range(10):
                raw_obs, _, _, _ = env.step(open_grip)
            self._ep = None

        self._env = env
        self._instruction = task.language
        self._step_count = 0
        self._ep_stall_s = 0.0
        self._ep_max_disp_m = 0.0
        self._ep_success = False
        self._ep_recorded = False
        self._target_init_pos.clear()
        if self._is_dynamic:
            self._measure_target_disp()

        img_main, img_wrist, state_8d = extract_libero_obs(raw_obs)
        if self._active_policy is not None and hasattr(self._active_policy, "resume_clock"):
            self._active_policy.resume_clock()
        now = time.perf_counter()
        self._ep_start_t = now
        self._last_step_end_t = now
        self._obs_ready_t = now
        self._ep_paused_s = 0.0
        return Observation(
            images={"agentview": img_main, "wrist": img_wrist},
            state={"eef_state": state_8d},
            instruction=self._instruction,
            state_time=0.0,
        )

    def step(self, action: Action) -> StepResult:
        assert self._env is not None
        t_enter = time.perf_counter()
        policy_wait_s = t_enter - self._obs_ready_t

        # Symmetric 1.0x real-time robot clock rule:
        # 1. If policy.act() blocked (>15 ms), the robot joints stood frozen for `policy_wait_s`
        #    before `action` could begin; on dynamic scenes, advance MuJoCo world physics for
        #    that exact stall duration while pausing the simulator-overhead clock.
        # 2. Allow the full 50 ms (20 Hz) real-time control window to elapse on the unpaused
        #    hardware clock so asynchronous GPU workers run for exact 1.0x physical step time,
        #    then pause clocks during CPU MuJoCo physics & offscreen rendering.
        is_blocking_stall = policy_wait_s > 0.015
        if is_blocking_stall:
            self._ep_stall_s += policy_wait_s
            if self._is_dynamic and self._ep is not None:
                with self._paused_sim_clock():
                    self._ep._apply_sync_latency(self._env, policy_wait_s)

        t_after_stall = time.perf_counter()
        anchor_t = t_after_stall if is_blocking_stall else self._obs_ready_t
        rem = self._dt_s - (t_after_stall - anchor_t)
        if rem > 0.0:
            time.sleep(rem)

        act_np = np.clip(np.asarray(action.data, dtype=np.float32), -1.0, 1.0)
        with self._paused_sim_clock():
            if self._is_dynamic and self._ep is not None:
                raw_obs, reward, done, _info = self._ep._step_env(self._env, act_np)
            else:
                raw_obs, reward, done, _info = self._env.step(act_np.tolist())

            self._step_count += 1
            if self._is_dynamic and self._step_count % 5 == 0:
                self._measure_target_disp()

            is_success = bool(self._env.check_success())
            img_main, img_wrist, state_8d = extract_libero_obs(raw_obs)

        terminated = bool(is_success or done)
        term_reason = "success" if is_success else ("done" if done else None)
        if is_success:
            self._ep_success = True
            if self._is_dynamic:
                self._measure_target_disp()

        now_end = time.perf_counter()
        self._last_step_end_t = now_end
        max_s = int(self._current_scene_meta.get("max_steps", 0))
        if terminated or (max_s > 0 and self._step_count >= max_s):
            self._finalize_episode_if_needed()

        self._obs_ready_t = time.perf_counter()
        return StepResult(
            observation=Observation(
                images={"agentview": img_main, "wrist": img_wrist},
                state={"eef_state": state_8d},
                instruction=self._instruction,
                state_time=self._step_count * self._dt_s,
            ),
            reward=float(reward) if reward is not None else 0.0,
            terminated=terminated,
            termination_reason=term_reason,
            info={"success": is_success},
        )


class ModelPreprocHelper:
    """Shared camera/state normalization and action unnormalization for Pi0.5."""

    def __init__(self, ckpt_dir: pathlib.Path) -> None:
        self.ckpt_dir = ckpt_dir
        self._aux_stream = torch.cuda.Stream(device="cuda:0")
        self.img_size = 224
        self.stats = load_mean_std_stats(
            ckpt_dir,
            "policy_preprocessor_step_2_normalizer_processor.safetensors",
            "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
        )
        self.tok = load_hf_tokenizer("mlx-community/paligemma-3b-mix-224-8bit", checkpoint_dir=ckpt_dir)

    def prompt_len_for(self, instruction: str) -> int:
        # Use 3-digit bins ('255') across all 32 state slots so runtime token length never exceeds prompt_len
        worst_norm_st = torch.full((32,), 2.0, dtype=torch.float32)
        cleaned = instruction.strip().replace("_", " ").replace("\n", " ")
        p = f"Task: {cleaned}, State: {discretize_pi05_state(worst_norm_st, 32)};\nAction: "
        return len(self.tok.encode(p).ids) + 8

    def prepare_inputs(self, obs: Observation) -> Tuple[torch.Tensor, torch.Tensor, str]:
        c1 = resize_square_rgb(obs.images["agentview"], self.img_size)
        c2 = resize_square_rgb(obs.images["wrist"], self.img_size)
        cams = torch.stack([c1, c2], dim=0)
        st = torch.from_numpy(np.asarray(obs.state["eef_state"], dtype=np.float32))
        norm_st = (st - self.stats["state_mean"]) / (self.stats["state_std"] + 1e-8)
        return cams, norm_st, str(obs.instruction or "")

    def update_slow_stage(self, slow_stage: Any, instruction: str, norm_st: torch.Tensor) -> None:
        with torch.cuda.device("cuda:0"), torch.cuda.stream(self._aux_stream):
            slow_stage.set_prompt(instruction, state=norm_st)
        self._aux_stream.synchronize()

    def unnormalize_action(self, norm_act: torch.Tensor) -> np.ndarray:
        act = norm_act.detach().cpu().float() * (self.stats["action_std"] + 1e-8) + self.stats["action_mean"]
        return np.clip(act.numpy(), -1.0, 1.0).astype(np.float32)


class SyncChunkPolicy(PolicyBase):
    """Synchronous ActionChunk policy (optional torch.compile acceleration on FastHead)."""

    def __init__(
        self,
        helper: ModelPreprocHelper,
        *,
        prompt_len: int,
        chunk_horizon: int = 10,
        num_streams: int = 1,
        compile_fast: bool = False,
    ) -> None:
        self.helper = helper
        self.chunk_horizon = chunk_horizon
        self.num_streams = num_streams
        self.compile_fast = compile_fast
        self.dep = ModelRegistry.build_for_engine(
            str(helper.ckpt_dir),
            slow_devices=[[0]],
            fast_devices=[0],
            fast_precision="bf16",
            prompt_len=prompt_len,
        )
        self.slow_stage = self.dep.slow_replicas[0][0]
        raw_fast = self.dep.fast_shards[0]
        self.fast_head = torch.compile(raw_fast, dynamic=False) if compile_fast else raw_fast
        self.dev = torch.device("cuda:0")
        tag = "compiled-sync" if compile_fast else "sync-chunk"
        self.info = PolicyInfo(
            name=f"{tag}-k{chunk_horizon}-n{num_streams}",
            action_space=ACTION_SPACE,
            observation_space=OBS_SPACE,
            control_hz=20.0,
            checkpoint=str(helper.ckpt_dir.name),
        )
        self.config = PolicyConfig(action_horizon=chunk_horizon)
        self.infer_latencies_ms: List[float] = []
        self._ep_infer_ms: List[float] = []
        self._reset_hook: Optional[Callable[[], None]] = None

        with torch.no_grad():
            B = self.num_streams
            w_obs = torch.zeros((B, *self.dep.slow_obs_shape), dtype=torch.bfloat16, device=self.dev)
            ctx = EagerCaptureContext(self.dep.channel, self.dev)
            self.slow_stage(w_obs, ctx)
            ks = [ctx.kv_by_layer[i][0] for i in range(self.dep.channel.layers)]
            vs = [ctx.kv_by_layer[i][1] for i in range(self.dep.channel.layers)]
            view = EagerDenseStateView(
                self.dep.channel,
                batch_ks=ks,
                batch_vs=vs,
                prefix_valid=torch.zeros((B,), dtype=torch.bool, device=self.dev),
            )
            noise = torch.randn((B, *self.dep.fast_obs_shape), dtype=torch.bfloat16, device=self.dev)
            pref = torch.zeros((B, 2, self.dep.action_dim), dtype=torch.bfloat16, device=self.dev)
            for _ in range(2):
                self.fast_head(noise, view, pref)
            torch.cuda.synchronize(self.dev)

    def set_reset_hook(self, hook: Optional[Callable[[], None]]) -> None:
        self._reset_hook = hook

    def pop_episode_telemetry(self) -> Dict[str, Any]:
        ep_ms = list(self._ep_infer_ms)
        self._ep_infer_ms.clear()
        s = _summarize_ms_list(ep_ms)
        return {
            "policy_calls": len(ep_ms),
            "infer_mean_ms": s["mean_ms"],
            "infer_p50_ms": s["p50_ms"],
            "infer_p99_ms": s["p99_ms"],
            "_raw_chunk_ms": [round(float(x), 3) for x in ep_ms],
        }

    def close(self) -> None:
        del self.slow_stage, self.fast_head, self.dep
        if self.compile_fast:
            torch._dynamo.reset()
        gc.collect()
        torch.cuda.empty_cache()

    def reset(self, scene: Scene) -> None:
        self._ep_infer_ms.clear()
        if self._reset_hook is not None:
            self._reset_hook()

    @torch.no_grad()
    def act(self, observation: Observation) -> ActionChunk:
        t0 = time.perf_counter()
        cams, norm_st, instr = self.helper.prepare_inputs(observation)
        self.helper.update_slow_stage(self.slow_stage, instr, norm_st)

        B = self.num_streams
        obs_b = (cams.to(self.dev, dtype=torch.bfloat16) / 255.0).unsqueeze(0).expand(B, -1, -1, -1, -1).contiguous()
        ctx = EagerCaptureContext(self.dep.channel, self.dev)
        self.slow_stage(obs_b, ctx)

        ks = [ctx.kv_by_layer[i][0] for i in range(self.dep.channel.layers)]
        vs = [ctx.kv_by_layer[i][1] for i in range(self.dep.channel.layers)]
        view = EagerDenseStateView(
            self.dep.channel,
            batch_ks=ks,
            batch_vs=vs,
            prefix_valid=torch.zeros((B,), dtype=torch.bool, device=self.dev),
        )
        noise = torch.randn((B, *self.dep.fast_obs_shape), dtype=torch.bfloat16, device=self.dev)
        frozen_pref = torch.zeros((B, 2, self.dep.action_dim), dtype=torch.bfloat16, device=self.dev)
        chunk_out = self.fast_head(noise, view, frozen_pref)
        torch.cuda.synchronize(self.dev)
        lat_s = time.perf_counter() - t0
        lat_ms = lat_s * 1000.0
        self.infer_latencies_ms.append(lat_ms)
        self._ep_infer_ms.append(lat_ms)

        actions = [
            Action(data=self.helper.unnormalize_action(chunk_out[0, k]))
            for k in range(min(self.chunk_horizon, chunk_out.size(1)))
        ]
        return ActionChunk(actions=actions, control_hz=20.0, inference_latency_s=lat_s)


class PyTorchAsyncPolicy(PolicyBase):
    """Asynchronous PyTorch Dual-Stream (SDPA + per-stream lock) policy (10 Hz Slow VLM, 5 Hz Fast chunk)."""

    def __init__(
        self,
        helper: ModelPreprocHelper,
        *,
        prompt_len: int,
        num_streams: int = 1,
        slow_every_steps: int = 2,
        fast_every_steps: int = 4,
    ) -> None:
        self.helper = helper
        self.num_streams = num_streams
        self.slow_every_steps = slow_every_steps
        self.fast_every_steps = fast_every_steps
        self.dep = ModelRegistry.build_for_engine(
            str(helper.ckpt_dir),
            slow_devices=[[0]],
            fast_devices=[0],
            fast_precision="bf16",
            prompt_len=prompt_len,
        )
        self.slow_stage = self.dep.slow_replicas[0][0]
        self.dev = torch.device("cuda:0")
        self._fg_stream = torch.cuda.Stream(device=self.dev)
        self.base = PyTorchDualStreamBaseline(
            self.dep.channel,
            action_dim=self.dep.action_dim,
            horizon_steps=self.dep.chunk_size,
            prefix_steps=2,
            slow_devices=[[0]],
            fast_devices=[0],
            slow_replicas=self.dep.slow_replicas,
            fast_experts={0: self.dep.fast_shards[0]},
            num_streams=self.num_streams,
            slow_batch_size=min(4, self.num_streams),
            max_fast_batch_size=max(4, self.num_streams),
            deadline_ms=45.0,
            step_dt_ms=50.0,
        )
        self.base.start()
        with torch.cuda.device(self.dev), torch.cuda.stream(self._fg_stream):
            self._bg_slow_obs = [
                torch.rand(self.dep.slow_obs_shape, dtype=torch.bfloat16, device=self.dev)
                for _ in range(max(0, self.num_streams - 1))
            ]
            self._bg_fast_obs = [
                torch.randn(self.dep.fast_obs_shape, dtype=torch.bfloat16, device=self.dev)
                for _ in range(max(0, self.num_streams - 1))
            ]
        self._fg_stream.synchronize()
        self._reset_hook: Optional[Callable[[], None]] = None
        self._step_idx = 0
        self.info = PolicyInfo(
            name=f"pytorch-async-n{num_streams}",
            action_space=ACTION_SPACE,
            observation_space=OBS_SPACE,
            control_hz=20.0,
            checkpoint=str(helper.ckpt_dir.name),
        )
        self.config = PolicyConfig(action_horizon=1)
        self.fast_ttac_ms: List[float] = []
        self.act_latencies_us: List[float] = []
        self.underruns: int = 0
        self._ep_act_us: List[float] = []
        self._ep_underruns: int = 0
        self._ep_cold_start_s: float = 0.0

    def set_reset_hook(self, hook: Optional[Callable[[], None]]) -> None:
        self._reset_hook = hook

    def pause_clock(self) -> None:
        self.base.pause_clock()

    def resume_clock(self) -> None:
        self.base.resume_clock()

    def pop_episode_telemetry(self) -> Dict[str, Any]:
        with self.base.stats_lock:
            ep_ttac = list(self.base.fast_ttac_ms)
            self.base.fast_ttac_ms.clear()
        self.fast_ttac_ms.extend(ep_ttac)
        ep_act = list(self._ep_act_us)
        self._ep_act_us.clear()
        ep_u = int(self._ep_underruns)
        self._ep_underruns = 0
        cs_s = float(self._ep_cold_start_s)
        self._ep_cold_start_s = 0.0
        s = _summarize_ms_list(ep_ttac)
        act_p50 = round(float(np.percentile(np.asarray(ep_act, dtype=np.float64), 50)), 2) if ep_act else None
        return {
            "policy_calls": len(ep_ttac),
            "cold_start_s": round(cs_s, 4),
            "infer_mean_ms": s["mean_ms"],
            "infer_p50_ms": s["p50_ms"],
            "infer_p99_ms": s["p99_ms"],
            "act_p50_us": act_p50,
            "ep_underruns": ep_u,
            "_raw_ttac_ms": [round(float(x), 3) for x in ep_ttac],
            "_raw_act_us": [round(float(x), 2) for x in ep_act],
        }

    def close(self) -> None:
        with self.base.stats_lock:
            self.fast_ttac_ms.extend(self.base.fast_ttac_ms)
            self.base.fast_ttac_ms.clear()
        self.base.stop()
        del self.base, self.slow_stage, self.dep
        gc.collect()
        torch.cuda.empty_cache()

    def reset(self, scene: Scene) -> None:
        self.base.stop()
        with self.base.stats_lock:
            if self.base.fast_ttac_ms:
                self.fast_ttac_ms.extend(self.base.fast_ttac_ms)
                self.base.fast_ttac_ms.clear()
        self.base.reset()
        self.base.start()
        self._step_idx = 0
        self._ep_act_us.clear()
        self._ep_underruns = 0
        self._ep_cold_start_s = 0.0
        if self._reset_hook is not None:
            self._reset_hook()

    def act(self, observation: Observation) -> ActionChunk:
        t0 = time.perf_counter()
        cams, norm_st, instr = self.helper.prepare_inputs(observation)

        if self._step_idx == 0:
            self.helper.update_slow_stage(self.slow_stage, instr, norm_st)
            with torch.cuda.device(self.dev), torch.cuda.stream(self._fg_stream):
                obs_bf16 = cams.to(self.dev, dtype=torch.bfloat16) / 255.0
                noise = torch.randn(self.dep.fast_obs_shape, dtype=torch.bfloat16, device=self.dev)
            self._fg_stream.synchronize()

            t_ns = self.base.now_ns()
            self.base.push_slow(obs_bf16, stream_id=0, t_obs_ns=t_ns)
            for s in range(1, self.num_streams):
                self.base.push_slow(self._bg_slow_obs[s - 1], stream_id=s, t_obs_ns=t_ns)
            deadline = time.monotonic() + 15.0
            while not all(self.base.state_t_obs[s] >= t_ns for s in range(self.num_streams)):
                if time.monotonic() > deadline:
                    raise TimeoutError("PyTorchAsyncPolicy timed out waiting for initial Slow state")
                time.sleep(0.001)

            t_fast_ns = self.base.now_ns()
            self.base.push_fast(noise, stream_id=0, t_obs_ns=t_fast_ns)
            for s in range(1, self.num_streams):
                self.base.push_fast(self._bg_fast_obs[s - 1], stream_id=s, t_obs_ns=t_fast_ns)
            deadline = time.monotonic() + 15.0
            while True:
                with self.base.stats_lock:
                    ready_fast = all(self.base.active_t_obs[s] >= t_fast_ns for s in range(self.num_streams))
                if ready_fast:
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError("PyTorchAsyncPolicy timed out waiting for initial Fast chunk")
                time.sleep(0.001)

            act_t, *_ = self.base.sample_actions(self.base.now_ns())
            with self.base.stats_lock:
                self.base.fast_ttac_ms.clear()
                self.base.commit_hard_underruns = 0
        else:
            now_ns = self.base.now_ns()
            if self._step_idx % self.slow_every_steps == 1:
                self.helper.update_slow_stage(self.slow_stage, instr, norm_st)
                with torch.cuda.device(self.dev), torch.cuda.stream(self._fg_stream):
                    obs_bf16 = cams.to(self.dev, dtype=torch.bfloat16) / 255.0
                self._fg_stream.synchronize()
                self.base.push_slow(obs_bf16, stream_id=0, t_obs_ns=now_ns)
                for s in range(1, self.num_streams):
                    self.base.push_slow(self._bg_slow_obs[s - 1], stream_id=s, t_obs_ns=now_ns)
            if self._step_idx % self.fast_every_steps == 0:
                with torch.cuda.device(self.dev), torch.cuda.stream(self._fg_stream):
                    noise = torch.randn(self.dep.fast_obs_shape, dtype=torch.bfloat16, device=self.dev)
                self._fg_stream.synchronize()
                self.base.push_fast(noise, stream_id=0, t_obs_ns=now_ns)
                for s in range(1, self.num_streams):
                    self.base.push_fast(self._bg_fast_obs[s - 1], stream_id=s, t_obs_ns=now_ns)
            act_t, _fa, _sa, u = self.base.sample_actions(self.base.now_ns())
            self.underruns += int(u)
            self._ep_underruns += int(u)

        lat_s = time.perf_counter() - t0
        if self._step_idx == 0:
            self._ep_cold_start_s = lat_s
        else:
            us = lat_s * 1e6
            self.act_latencies_us.append(us)
            self._ep_act_us.append(us)
        self._step_idx += 1
        act = self.helper.unnormalize_action(act_t[0])
        return ActionChunk(actions=[Action(data=act)], control_hz=20.0, inference_latency_s=lat_s)


class TempoPolicy(PolicyBase):
    """Tempo Engine reactive policy: sliced VLM (10 Hz) + lock-free cache + CUDA Graph Fast (5 Hz) + RTC."""

    def __init__(
        self,
        helper: ModelPreprocHelper,
        *,
        prompt_len: int,
        num_streams: int = 1,
        slow_every_steps: int = 2,
        fast_every_steps: int = 4,
    ) -> None:
        self.helper = helper
        self.num_streams = num_streams
        self.slow_every_steps = slow_every_steps
        self.fast_every_steps = fast_every_steps
        self.engine = Engine(
            str(helper.ckpt_dir),
            num_streams=num_streams,
            slow_devices=[0],
            fast_devices=[0],
            slow_rate_hz=20.0 / float(slow_every_steps),
            fast_rate_hz=20.0 / float(fast_every_steps),
            deadline_ms=45.0,
            step_dt_ms=50.0,
            fast_precision="bf16",
            model_kwargs={"prompt_len": prompt_len},
        )
        assert self.engine.deployment is not None
        self.slow_stage = self.engine.deployment.slow_replicas[0][0]
        sib_stream = torch.cuda.Stream(device="cuda:0")
        with torch.cuda.device("cuda:0"), torch.cuda.stream(sib_stream):
            self._bg_slow_obs = [
                torch.randint(0, 256, self.engine.slow_obs_shape, dtype=torch.uint8, device="cuda:0")
                for _ in range(max(0, self.num_streams - 1))
            ]
        sib_stream.synchronize()
        self._reset_hook: Optional[Callable[[], None]] = None
        self._step_idx = 0
        self.info = PolicyInfo(
            name=f"tempo-n{num_streams}",
            action_space=ACTION_SPACE,
            observation_space=OBS_SPACE,
            control_hz=20.0,
            checkpoint=str(helper.ckpt_dir.name),
        )
        self.config = PolicyConfig(action_horizon=1)
        self.fast_ttac_ms: List[float] = []
        self.act_latencies_us: List[float] = []
        self.underruns: int = 0
        self._ep_act_us: List[float] = []
        self._ep_underruns: int = 0
        self._ep_cold_start_s: float = 0.0
        self._ep_telemetry_drained: bool = True

    def set_reset_hook(self, hook: Optional[Callable[[], None]]) -> None:
        self._reset_hook = hook

    def pause_clock(self) -> None:
        self.engine.pause_clock()

    def resume_clock(self) -> None:
        self.engine.resume_clock()

    def pop_episode_telemetry(self) -> Dict[str, Any]:
        ep_ttac = list(self.engine.telemetry().fast_ttac_ms)
        self.engine.reset_stats()
        self.fast_ttac_ms.extend(ep_ttac)
        self._ep_telemetry_drained = True
        ep_act = list(self._ep_act_us)
        self._ep_act_us.clear()
        ep_u = int(self._ep_underruns)
        self._ep_underruns = 0
        cs_s = float(self._ep_cold_start_s)
        self._ep_cold_start_s = 0.0
        s = _summarize_ms_list(ep_ttac)
        act_p50 = round(float(np.percentile(np.asarray(ep_act, dtype=np.float64), 50)), 2) if ep_act else None
        return {
            "policy_calls": len(ep_ttac),
            "cold_start_s": round(cs_s, 4),
            "infer_mean_ms": s["mean_ms"],
            "infer_p50_ms": s["p50_ms"],
            "infer_p99_ms": s["p99_ms"],
            "act_p50_us": act_p50,
            "ep_underruns": ep_u,
            "_raw_ttac_ms": [round(float(x), 3) for x in ep_ttac],
            "_raw_act_us": [round(float(x), 2) for x in ep_act],
        }

    def close(self) -> None:
        if not self._ep_telemetry_drained:
            self.fast_ttac_ms.extend(self.engine.telemetry().fast_ttac_ms)
            self._ep_telemetry_drained = True
        self.engine.stop()
        del self.slow_stage, self.engine
        gc.collect()
        torch.cuda.empty_cache()

    def reset(self, scene: Scene) -> None:
        if not self._ep_telemetry_drained:
            self.fast_ttac_ms.extend(self.engine.telemetry().fast_ttac_ms)
        self.engine.reset()
        self._ep_telemetry_drained = False
        self._step_idx = 0
        self._ep_act_us.clear()
        self._ep_underruns = 0
        self._ep_cold_start_s = 0.0
        if self._reset_hook is not None:
            self._reset_hook()

    def act(self, observation: Observation) -> ActionChunk:
        t0 = time.perf_counter()
        cams, norm_st, instr = self.helper.prepare_inputs(observation)

        if self._step_idx == 0:
            self.helper.update_slow_stage(self.slow_stage, instr, norm_st)
            t_ns = self.engine.now_ns()
            self.engine.push_slow(cams, stream_id=0, t_obs_ns=t_ns)
            for s in range(1, self.num_streams):
                self.engine.push_slow(self._bg_slow_obs[s - 1], stream_id=s, t_obs_ns=t_ns)
            self.engine.wait_ready()
            self.engine.reset_stats()
            actions, *_ = self.engine.sample_actions(cpu=True)
        else:
            now_ns = self.engine.now_ns()
            if self._step_idx % self.slow_every_steps == 1:
                self.helper.update_slow_stage(self.slow_stage, instr, norm_st)
                self.engine.push_slow(cams, stream_id=0, t_obs_ns=now_ns)
                for s in range(1, self.num_streams):
                    self.engine.push_slow(self._bg_slow_obs[s - 1], stream_id=s, t_obs_ns=now_ns)
            if self._step_idx % self.fast_every_steps == 0:
                for s in range(self.num_streams):
                    self.engine.push_fast(stream_id=s, t_obs_ns=now_ns)
            actions, _fa, _sa, u = self.engine.sample_actions(now_ns, cpu=True)
            self.underruns += int(u)
            self._ep_underruns += int(u)

        lat_s = time.perf_counter() - t0
        if self._step_idx == 0:
            self._ep_cold_start_s = lat_s
        else:
            us = lat_s * 1e6
            self.act_latencies_us.append(us)
            self._ep_act_us.append(us)
        self._step_idx += 1
        act = self.helper.unnormalize_action(actions[0])
        return ActionChunk(actions=[Action(data=act)], control_hz=20.0, inference_latency_s=lat_s)


def _locate_base_dynamic_cfg_dir(dyn_repo: pathlib.Path) -> pathlib.Path:
    cand_dirs = [
        dyn_repo / "libero" / "libero" / "dynamic_configs",
        dyn_repo / "libero" / "libero" / "configs",
        dyn_repo / "configs",
        dyn_repo / "templates",
    ]
    for cd in cand_dirs:
        if cd.exists() and list(cd.glob("libero_dynamic_l*.yaml")):
            return cd.resolve()
    found = list(dyn_repo.rglob("libero_dynamic_l1.yaml")) + list(dyn_repo.rglob("libero_dynamic_manual_l1.yaml"))
    if found:
        return found[0].parent.resolve()
    raise FileNotFoundError(f"Could not locate libero_dynamic_l*.yaml under {dyn_repo}")


def _patch_speed_yaml_dir(src_cfg_dir: pathlib.Path, dst_cfg_dir: pathlib.Path, speed_mps: float) -> int:
    """Patches `alpha_mps` across copied LIBERO-Dynamic YAML configs per official speed-sweep protocol."""
    dst_cfg_dir.mkdir(parents=True, exist_ok=True)
    patched = 0

    def _walk(node: Any) -> None:
        nonlocal patched
        if isinstance(node, dict):
            for k, v in list(node.items()):
                if k == "alpha_mps":
                    node[k] = float(speed_mps)
                    patched += 1
                elif k in ("save_video_frames", "save_video_frames_to_disk"):
                    node[k] = False
                elif k in ("model_input_video", "finalize", "logging") and isinstance(v, dict):
                    v["enabled"] = False
                    _walk(v)
                else:
                    _walk(v)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    for src_yaml in sorted(src_cfg_dir.glob("libero_dynamic_l*.yaml")):
        dst_yaml = dst_cfg_dir / src_yaml.name
        shutil.copy2(src_yaml, dst_yaml)
        data = yaml.safe_load(dst_yaml.read_text(encoding="utf-8"))
        _walk(data)
        defs = data.get("defaults")
        if isinstance(defs, dict):
            for sec in ("model_input_video", "finalize", "logging"):
                if isinstance(defs.get(sec), dict):
                    defs[sec]["enabled"] = False
            if isinstance(defs.get("latency"), dict):
                defs["latency"]["save_video_frames"] = False
                defs["latency"]["save_video_frames_to_disk"] = False
        data["description"] = f"{data.get('description', '')} | speed sweep alpha_mps={speed_mps:.3f}"
        dst_yaml.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")

    if patched == 0:
        raise RuntimeError(f"Zero `alpha_mps` fields found when patching {src_cfg_dir}")
    return patched


def build_static_and_dynamic_scenes(
    dyn_repo: pathlib.Path,
    output_dir: pathlib.Path,
    levels: Sequence[str],
    speeds: Sequence[Tuple[str, float]],
    static_tasks: Sequence[Tuple[str, int]],
    static_inits: Sequence[int],
    dyn_tasks: Sequence[Tuple[str, int]],
    dyn_trajs: Sequence[int],
    dyn_max_steps: int,
) -> Tuple[
    Dict[str, argparse.Namespace],
    Dict[str, Dict[str, Any]],
    List[Scene],
    List[Scene],
]:
    base_cfg_dir = _locate_base_dynamic_cfg_dir(dyn_repo)
    output_dir.mkdir(parents=True, exist_ok=True)

    static_scenes: List[Scene] = []
    for suite_name, task_id in static_tasks:
        _suite_obj, task_obj, _ = legacy.task_bddl(suite_name, task_id)
        s_max = STATIC_SUITE_MAX_STEPS.get(suite_name, 300)
        for init_idx in static_inits:
            sid = f"static__{suite_name}__t{task_id:02d}__i{init_idx:02d}"
            static_scenes.append(
                Scene(
                    id=sid,
                    instruction=task_obj.language,
                    init_seed=init_idx,
                    metadata={
                        "dynamic": False,
                        "group": suite_name,
                        "level": "static",
                        "speed_mps": 0.0,
                        "traj_index": -1,
                        "suite": suite_name,
                        "task_id": task_id,
                        "init_idx": init_idx,
                        "max_steps": s_max,
                    },
                )
            )

    wanted_dyn = set(dyn_tasks)
    dyn_args_by_speed: Dict[str, argparse.Namespace] = {}
    items_by_id: Dict[str, Dict[str, Any]] = {}
    dyn_scenes: List[Scene] = []

    for speed_label, speed_mps in speeds:
        speed_root = output_dir / f"speed_{speed_label}"
        patched_cfg_dir = speed_root / "config"
        eff_out_dir = speed_root / "effective"
        _patch_speed_yaml_dir(base_cfg_dir, patched_cfg_dir, speed_mps)

        parser = build_dyn_parser()
        argv = [
            "--config-dir",
            str(patched_cfg_dir),
            "--output-dir",
            str(eff_out_dir),
            "--levels",
            *levels,
            "--traj-indices",
            *[str(i) for i in dyn_trajs],
            "--max-steps",
            str(dyn_max_steps),
            "--reviewed-old-physics",
            "--keep-static-support-contact",
            "--static-support-clearance-m",
            "0.001",
            "--no-save-videos",
            "--no-save-model-input-videos",
            "--no-save-step-trace",
            "--stop-on-success",
        ]
        dyn_args = parser.parse_args(argv)
        legacy.apply_reviewed_old_physics_profile(dyn_args)
        dyn_args.output_dir.mkdir(parents=True, exist_ok=True)
        cfg_dir = dyn_args.config_dir.expanduser().resolve()
        config_path_by_level = legacy.write_effective_config_files(
            cfg_dir, dyn_args.output_dir, dyn_args.levels, dyn_args
        )
        all_items = legacy.build_items(
            cfg_dir,
            dyn_args.levels,
            dyn_args.traj_indices,
            keep_static_support_contact=bool(dyn_args.keep_static_support_contact),
            static_support_clearance_m=float(dyn_args.static_support_clearance_m),
            config_path_by_level=config_path_by_level,
        )
        dyn_args_by_speed[speed_label] = dyn_args

        seen: set[Tuple[str, str, int, int]] = set()
        speed_scenes: List[Scene] = []
        for item in all_items:
            key = (str(item["suite"]), int(item["task_id"]))
            if key not in wanted_dyn:
                continue
            lvl = str(item["level"])
            t_idx = int(item["traj_index"])
            dedup_k = (lvl, key[0], key[1], t_idx)
            if dedup_k in seen:
                continue
            seen.add(dedup_k)

            _suite_obj, task_obj, _ = legacy.task_bddl(key[0], key[1])
            scene_key = f"{item['item_id']}__s{speed_label}"
            items_by_id[scene_key] = item
            speed_scenes.append(
                Scene(
                    id=scene_key,
                    instruction=task_obj.language,
                    init_seed=int(dyn_args.seed),
                    metadata={
                        "dynamic": True,
                        "scene_key": scene_key,
                        "item_id": str(item["item_id"]),
                        "group": f"speed_{speed_label}",
                        "speed_label": speed_label,
                        "speed_mps": float(speed_mps),
                        "level": lvl,
                        "suite": key[0],
                        "task_id": key[1],
                        "traj_index": t_idx,
                        "init_idx": int(dyn_args.init_state_id),
                        "max_steps": int(dyn_max_steps),
                    },
                )
            )
        speed_scenes.sort(
            key=lambda s: (
                str(s.metadata["level"]),
                str(s.metadata["suite"]),
                int(s.metadata["task_id"]),
                int(s.metadata["traj_index"]),
            )
        )
        dyn_scenes.extend(speed_scenes)

    static_scenes.sort(
        key=lambda s: (
            CANONICAL_SUITES.index(str(s.metadata["suite"]))
            if str(s.metadata["suite"]) in CANONICAL_SUITES
            else 99,
            int(s.metadata["task_id"]),
            int(s.metadata["init_idx"]),
        )
    )
    return dyn_args_by_speed, items_by_id, static_scenes, dyn_scenes


def summarize_records(
    records: List[Dict[str, Any]],
    num_streams: int,
) -> Dict[str, Any]:
    total_ep = len(records)
    succ_recs = [r for r in records if r["success"]]
    total_succ = len(succ_recs)
    total_wall_s = sum(float(r["wall_time_s"]) for r in records)
    total_stall_s = sum(float(r["stall_time_s"]) for r in records)
    total_host_s = sum(float(r["sim_host_wall_s"]) for r in records)
    mean_wall_all_s = total_wall_s / max(1, total_ep)
    mean_succ_time_s = (
        sum(float(r["wall_time_s"]) for r in succ_recs) / len(succ_recs) if succ_recs else None
    )
    mean_succ_steps = (
        sum(int(r["steps"]) for r in succ_recs) / len(succ_recs) if succ_recs else None
    )
    stall_pct = 100.0 * total_stall_s / max(1e-6, total_wall_s)
    fleet_goodput_tasks_per_min = num_streams * (total_succ / max(1e-6, total_wall_s / 60.0))

    return {
        "episodes": total_ep,
        "successes": total_succ,
        "success_rate_pct": round(100.0 * total_succ / max(1, total_ep), 1),
        "mean_time_to_success_s": round(mean_succ_time_s, 2) if mean_succ_time_s is not None else None,
        "mean_steps_to_success": round(mean_succ_steps, 1) if mean_succ_steps is not None else None,
        "mean_episode_wall_s": round(mean_wall_all_s, 2),
        "total_wall_s": round(total_wall_s, 2),
        "total_stall_s": round(total_stall_s, 2),
        "total_sim_host_wall_s": round(total_host_s, 2),
        "robot_stall_pct": round(stall_pct, 1),
        "fleet_goodput_tasks_per_min": round(fleet_goodput_tasks_per_min, 2),
        "max_target_disp_m": round(max((float(r["max_target_disp_m"]) for r in records), default=0.0), 4),
    }


def _load_resumed_records(
    live_jsonl: Optional[pathlib.Path],
    suite_tag: str,
    policy_label: str,
    num_streams: int,
) -> Tuple[Dict[str, Dict[str, Any]], List[float], List[float], List[float]]:
    if live_jsonl is None or not live_jsonl.exists():
        return {}, [], [], []
    resumed_raw_rows: Dict[str, Dict[str, Any]] = {}
    for line in live_jsonl.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if (
            row.get("_suite_tag") == suite_tag
            and row.get("_policy_label") == policy_label
            and int(row.get("_num_streams", -1)) == num_streams
            and row.get("scene_id")
        ):
            resumed_raw_rows[str(row["scene_id"])] = row

    resumed: Dict[str, Dict[str, Any]] = {}
    raw_chunk_ms: List[float] = []
    raw_ttac_ms: List[float] = []
    raw_act_us: List[float] = []
    for sid, row in resumed_raw_rows.items():
        raw_chunk_ms.extend(float(x) for x in row.get("_raw_chunk_ms", []))
        raw_ttac_ms.extend(float(x) for x in row.get("_raw_ttac_ms", []))
        raw_act_us.extend(float(x) for x in row.get("_raw_act_us", []))
        resumed[sid] = {k: v for k, v in row.items() if not k.startswith("_")}
    return resumed, raw_chunk_ms, raw_ttac_ms, raw_act_us


def evaluate_policy_suite(
    policy_factory: Callable[[], PolicyBase],
    policy_label: str,
    num_streams: int,
    embodiment: LiberoRealtimeEmbodiment,
    suite_tag: str,
    groups: Dict[str, Tuple[List[Scene], int]],
    log_dir: pathlib.Path,
    live_jsonl: Optional[pathlib.Path] = None,
    keep_raw_samples: bool = False,
) -> Dict[str, Any]:
    resumed_by_id, raw_chunk_ms, raw_ttac_ms, raw_act_us = _load_resumed_records(
        live_jsonl, suite_tag, policy_label, num_streams
    )
    pending_groups: Dict[str, Tuple[List[Scene], int]] = {}
    for grp_name, (scenes, grp_max_steps) in groups.items():
        rem = [sc for sc in scenes if sc.id not in resumed_by_id]
        if rem:
            pending_groups[grp_name] = (rem, grp_max_steps)

    all_recs: List[Dict[str, Any]] = list(resumed_by_id.values())
    if resumed_by_id:
        print(
            f"  [{policy_label}] Resumed {len(resumed_by_id)} completed episodes on {suite_tag} from {live_jsonl}",
            flush=True,
        )

    total_underruns: int = sum(int(r.get("ep_underruns", 0)) for r in all_recs)

    if pending_groups:
        policy = policy_factory()
        embodiment.clear_records()

        def _on_ep(rec: Dict[str, Any]) -> None:
            if live_jsonl is not None:
                live_jsonl.parent.mkdir(parents=True, exist_ok=True)
                payload = {
                    "_suite_tag": suite_tag,
                    "_policy_label": policy_label,
                    "_num_streams": num_streams,
                    "_ts": round(time.time(), 2),
                    **rec,
                }
                with live_jsonl.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    fh.flush()

        embodiment.bind_policy(policy, on_episode_record=_on_ep)
        try:
            for grp_name, (scenes, grp_max_steps) in pending_groups.items():
                task = Task(
                    name=f"{suite_tag}-{grp_name}",
                    scenes=scenes,
                    scorer="success_at_end",
                    max_steps=grp_max_steps,
                )
                ev_log = inspect_eval(
                    task,
                    policy=policy,
                    embodiment=embodiment,
                    log_dir=str(log_dir),
                    fail_on_error=True,
                    store_frames=False,
                    store_records=False,
                )
                embodiment._finalize_episode_if_needed()
                if getattr(ev_log, "status", "success") != "success":
                    raise RuntimeError(
                        f"inspect_eval failed on {suite_tag}/{grp_name}: "
                        f"status={getattr(ev_log, 'status', None)} error={getattr(ev_log, 'error', None)}"
                    )
            for rec in embodiment.episode_records:
                all_recs.append({k: v for k, v in rec.items() if not k.startswith("_")})

            if hasattr(policy, "infer_latencies_ms") and policy.infer_latencies_ms:
                raw_chunk_ms.extend(float(x) for x in policy.infer_latencies_ms)
            if hasattr(policy, "fast_ttac_ms") and policy.fast_ttac_ms:
                raw_ttac_ms.extend(float(x) for x in policy.fast_ttac_ms)
            if hasattr(policy, "act_latencies_us") and policy.act_latencies_us:
                raw_act_us.extend(float(x) for x in policy.act_latencies_us)
            if hasattr(policy, "underruns"):
                total_underruns = sum(int(r.get("ep_underruns", 0)) for r in all_recs)
        finally:
            embodiment.bind_policy(None)
            policy.close()

    expected_ids = {sc.id for scenes, _ in groups.values() for sc in scenes}
    got_ids = {str(r["scene_id"]) for r in all_recs}
    if got_ids != expected_ids:
        missing = sorted(expected_ids - got_ids)
        raise RuntimeError(
            f"[{policy_label}] {suite_tag} missing {len(missing)}/{len(expected_ids)} scenes (first: {missing[:3]})"
        )

    # Deterministic ordering matching input groups
    scene_order = {
        sc.id: idx
        for idx, sc in enumerate(sc for scenes, _ in groups.values() for sc in scenes)
    }
    all_recs.sort(key=lambda r: scene_order.get(str(r["scene_id"]), 10**9))

    return build_suite_result(
        suite_tag=suite_tag,
        policy_label=policy_label,
        num_streams=num_streams,
        all_recs=all_recs,
        group_names=list(groups.keys()),
        raw_chunk_ms=raw_chunk_ms,
        raw_ttac_ms=raw_ttac_ms,
        raw_act_us=raw_act_us,
        has_underruns=("Async" in policy_label or "Tempo" in policy_label),
        total_underruns=total_underruns,
        keep_raw_samples=keep_raw_samples,
    )


def build_suite_result(
    *,
    suite_tag: str,
    policy_label: str,
    num_streams: int,
    all_recs: List[Dict[str, Any]],
    group_names: Sequence[str],
    raw_chunk_ms: Sequence[float],
    raw_ttac_ms: Sequence[float],
    raw_act_us: Sequence[float],
    has_underruns: bool,
    total_underruns: int,
    keep_raw_samples: bool = False,
) -> Dict[str, Any]:
    per_group: Dict[str, Any] = {}
    for grp_name in group_names:
        g_recs = [r for r in all_recs if r.get("group") == grp_name]
        if not g_recs:
            continue
        g_sum = summarize_records(g_recs, num_streams)
        g_sum["speed_mps"] = float(g_recs[0].get("speed_mps", 0.0))
        per_group[grp_name] = g_sum
        print(
            f"  [{policy_label}] {suite_tag}/{grp_name}: "
            f"{g_sum['successes']}/{g_sum['episodes']} ({g_sum['success_rate_pct']:.1f}%) | "
            f"time_to_succ={g_sum['mean_time_to_success_s']}s | "
            f"stall={g_sum['robot_stall_pct']:.1f}% | "
            f"goodput={g_sum['fleet_goodput_tasks_per_min']:.2f} tasks/min",
            flush=True,
        )

    per_suite: Dict[str, Any] = {}
    for s_name in CANONICAL_SUITES:
        s_recs = [r for r in all_recs if r.get("suite") == s_name]
        if s_recs:
            per_suite[s_name] = summarize_records(s_recs, num_streams)

    per_motion_family: Dict[str, Any] = {}
    if suite_tag == "LIBERO-Dynamic":
        for fam in ("l1", "l2", "l3"):
            f_recs = [r for r in all_recs if r.get("level") == fam]
            if f_recs:
                per_motion_family[fam] = summarize_records(f_recs, num_streams)

    overall = summarize_records(all_recs, num_streams)
    result: Dict[str, Any] = {
        "suite": suite_tag,
        "policy": policy_label,
        "model": "lerobot/pi05_libero_finetuned_v044",
        "num_streams": num_streams,
        **overall,
        "per_group": per_group,
        "per_suite": per_suite,
    }
    if per_motion_family:
        result["per_motion_family"] = per_motion_family
    result["episodes_detail"] = all_recs

    if raw_chunk_ms:
        arr = np.asarray(raw_chunk_ms, dtype=np.float64)
        result["chunk_infer_p50_ms"] = round(float(np.percentile(arr, 50)), 2)
        result["chunk_infer_p99_ms"] = round(float(np.percentile(arr, 99)), 2)

    if raw_ttac_ms:
        arr = np.asarray(raw_ttac_ms, dtype=np.float64)
        result["fast_ttac_p50_ms"] = round(float(np.percentile(arr, 50)), 2)
        result["fast_ttac_p99_ms"] = round(float(np.percentile(arr, 99)), 2)

    if raw_act_us:
        arr = np.asarray(raw_act_us, dtype=np.float64)
        result["act_p50_us"] = round(float(np.percentile(arr, 50)), 2)

    if has_underruns:
        result["underruns"] = int(total_underruns)

    if keep_raw_samples:
        result["_raw_chunk_ms"] = [round(float(x), 3) for x in raw_chunk_ms]
        result["_raw_ttac_ms"] = [round(float(x), 3) for x in raw_ttac_ms]
        result["_raw_act_us"] = [round(float(x), 2) for x in raw_act_us]

    return result


def merge_shard_files(shard_paths: Sequence[pathlib.Path], out_path: pathlib.Path) -> None:
    shards = [json.loads(p.read_text(encoding="utf-8")) for p in shard_paths]
    if not shards:
        raise ValueError("No shard files provided")
    expected_num_shards = int(shards[0]["num_shards"])
    got_shard_indices = sorted(int(sh["shard_index"]) for sh in shards)
    if any(int(sh["num_shards"]) != expected_num_shards for sh in shards) or got_shard_indices != list(range(expected_num_shards)):
        raise ValueError(
            f"Incomplete or mismatched shard set: expected indices 0..{expected_num_shards - 1}, got {got_shard_indices}"
        )

    gpu_name = shards[0].get("gpu", "NVIDIA GeForce RTX 4090")
    by_key: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = {}
    for sh in shards:
        for r in sh["runs"]:
            by_key.setdefault((r["suite"], r["policy"], int(r["num_streams"])), []).append(r)

    suite_order = {"LIBERO-Static": 0, "LIBERO-Dynamic": 1}
    pol_order = ["SyncChunk", "CompiledSync", "PyTorchAsync", "Tempo"]
    sorted_keys = sorted(
        by_key.keys(),
        key=lambda k: (
            k[2],
            suite_order.get(k[0], 99),
            next((i for i, p in enumerate(pol_order) if k[1].startswith(p)), 99),
            k[1],
        ),
    )

    suite_rank = {s: idx for idx, s in enumerate(CANONICAL_SUITES)}
    speed_rank = {f"speed_{lbl}": idx for idx, (lbl, _) in enumerate(CANONICAL_SPEEDS)}

    merged_runs: List[Dict[str, Any]] = []
    for suite_tag, pol_label, n_streams in sorted_keys:
        parts = by_key[(suite_tag, pol_label, n_streams)]
        if len(parts) != expected_num_shards:
            raise RuntimeError(
                f"Run ({suite_tag}, {pol_label}, N={n_streams}) present in {len(parts)}/{expected_num_shards} shards"
            )
        all_recs: List[Dict[str, Any]] = []
        seen_ids = set()
        raw_chunk: List[float] = []
        raw_ttac: List[float] = []
        raw_act: List[float] = []
        for pr in parts:
            for ep in pr["episodes_detail"]:
                sid = str(ep["scene_id"])
                if sid in seen_ids:
                    raise RuntimeError(f"Duplicate scene_id {sid} in {suite_tag}/{pol_label}")
                seen_ids.add(sid)
                all_recs.append(ep)
            raw_chunk.extend(float(x) for x in pr.get("_raw_chunk_ms", []))
            raw_ttac.extend(float(x) for x in pr.get("_raw_ttac_ms", []))
            raw_act.extend(float(x) for x in pr.get("_raw_act_us", []))

        if suite_tag == "LIBERO-Static":
            all_recs.sort(key=lambda r: (suite_rank.get(r["suite"], 99), int(r["task_id"]), int(r["init_idx"])))
            grp_names = list(CANONICAL_SUITES)
        else:
            all_recs.sort(
                key=lambda r: (
                    speed_rank.get(r["group"], 99),
                    str(r["level"]),
                    suite_rank.get(r["suite"], 99),
                    int(r["task_id"]),
                    int(r["traj_index"]),
                )
            )
            grp_names = [f"speed_{lbl}" for lbl, _ in CANONICAL_SPEEDS]

        merged_runs.append(
            build_suite_result(
                suite_tag=suite_tag,
                policy_label=pol_label,
                num_streams=n_streams,
                all_recs=all_recs,
                group_names=grp_names,
                raw_chunk_ms=raw_chunk,
                raw_ttac_ms=raw_ttac,
                raw_act_us=raw_act,
                has_underruns=("Async" in pol_label or "Tempo" in pol_label),
                total_underruns=sum(int(r.get("ep_underruns", 0)) for r in all_recs),
                keep_raw_samples=False,
            )
        )

    attach_paired_speedups(merged_runs)
    static_eps = next((r["episodes"] for r in merged_runs if r["suite"] == "LIBERO-Static"), 0)
    dyn_runs = [r for r in merged_runs if r["suite"] == "LIBERO-Dynamic"]
    dyn_eps = dyn_runs[0]["episodes"] if dyn_runs else 0
    traj_indices = sorted({int(ep["traj_index"]) for ep in dyn_runs[0]["episodes_detail"]}) if dyn_runs else []

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "gpu": gpu_name,
                "harness": "robocurve/inspect-robots",
                "protocol": {
                    "static_episodes": static_eps,
                    "dynamic_episodes": dyn_eps,
                    "traj_indices": traj_indices,
                    "num_shards": expected_num_shards,
                },
                "benchmarks": [
                    "LIBERO-Static (40 tasks across spatial, object, goal, 10 @ 20 Hz real-time)",
                    "LIBERO-Dynamic (40 tasks x 3 motion families x 0.03 / 0.06 / 0.09 m/s @ 20 Hz real-time)",
                ],
                "runs": merged_runs,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[RealtimeBench] Merged {expected_num_shards} shards -> {out_path}", flush=True)


def attach_paired_speedups(runs: List[Dict[str, Any]]) -> None:
    """Computes paired Time-to-Success speedup of Tempo over baselines on commonly-succeeded scenes."""
    by_key = {(r["suite"], r["num_streams"], r["policy"]): r for r in runs}
    for (suite_tag, n_s, _pol), r in by_key.items():
        tempo_r = by_key.get((suite_tag, n_s, f"Tempo (Engine, N={n_s})"))
        if tempo_r is None or r is tempo_r:
            continue
        tempo_succ = {
            ep["scene_id"]: float(ep["wall_time_s"])
            for ep in tempo_r["episodes_detail"]
            if ep["success"]
        }
        base_succ = {
            ep["scene_id"]: float(ep["wall_time_s"])
            for ep in r["episodes_detail"]
            if ep["success"]
        }
        common = sorted(set(tempo_succ) & set(base_succ))
        if common:
            t_mean = sum(tempo_succ[k] for k in common) / len(common)
            b_mean = sum(base_succ[k] for k in common) / len(common)
            r["paired_common_successes"] = len(common)
            r["paired_wall_s_vs_tempo"] = {
                "baseline_s": round(b_mean, 2),
                "tempo_s": round(t_mean, 2),
                "tempo_speedup_x": round(b_mean / max(1e-6, t_mean), 2),
            }


def _group_static_scenes(scenes: Sequence[Scene]) -> Dict[str, Tuple[List[Scene], int]]:
    out: Dict[str, Tuple[List[Scene], int]] = {}
    for suite_name in CANONICAL_SUITES:
        sub = [sc for sc in scenes if sc.metadata.get("suite") == suite_name]
        if sub:
            out[suite_name] = (sub, STATIC_SUITE_MAX_STEPS[suite_name])
    return out


def _group_dynamic_scenes(scenes: Sequence[Scene], dyn_max_steps: int) -> Dict[str, Tuple[List[Scene], int]]:
    out: Dict[str, Tuple[List[Scene], int]] = {}
    for speed_label, _ in CANONICAL_SPEEDS:
        g = f"speed_{speed_label}"
        sub = [sc for sc in scenes if sc.metadata.get("group") == g]
        if sub:
            out[g] = (sub, dyn_max_steps)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--dyn-repo", type=pathlib.Path, default=None, help="Path to cloned LIBERO-Dynamic repository")
    parser.add_argument("--merge-shards", type=pathlib.Path, nargs="+", default=None, help="Merge completed shard JSON files into --out-json")
    parser.add_argument("--streams", type=str, default="1", help="Comma-separated concurrent stream counts")
    parser.add_argument("--static-inits", type=str, default="0,1,2,3,4,5,6,7,8,9", help="Static LIBERO init indices")
    parser.add_argument("--traj-indices", type=str, default="0", help="LIBERO-Dynamic trajectory indices (e.g. 0 or 0,1,2,3,4)")
    parser.add_argument("--dyn-max-steps", type=int, default=600, help="Max control steps per LIBERO-Dynamic episode")
    parser.add_argument("--num-shards", type=int, default=1, help="Total shards for multi-GPU parallel evaluation")
    parser.add_argument("--shard-index", type=int, default=0, help="0-based index of this shard")
    parser.add_argument("--reverse-suite-order", action="store_true", help="Run LIBERO-Dynamic before LIBERO-Static")
    parser.add_argument("--live-jsonl", type=pathlib.Path, default=None, help="Append/resume per-episode JSONL path")
    parser.add_argument("--log-dir", type=pathlib.Path, default=None)
    parser.add_argument("--out-json", type=pathlib.Path, default=pathlib.Path("benchmarks/libero_dynamic_eval.json"))
    args = parser.parse_args()

    if args.merge_shards:
        merge_shard_files(args.merge_shards, args.out_json)
        return

    if args.dyn_repo is None:
        parser.error("--dyn-repo is required unless --merge-shards is specified")
    _ensure_dyn_imports(args.dyn_repo)

    if not (0 <= args.shard_index < args.num_shards):
        raise ValueError(f"--shard-index {args.shard_index} out of range for --num-shards {args.num_shards}")

    log_dir = args.log_dir if args.log_dir is not None else pathlib.Path(f"/tmp/inspect_logs_s{args.shard_index}")
    log_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir = log_dir
    gpu_name = torch.cuda.get_device_name(0)
    print(
        f"[RealtimeBench] GPU={gpu_name} | mode={args.mode} | shard={args.shard_index}/{args.num_shards} | log_dir={args.log_dir}",
        flush=True,
    )

    _, pi05_dir = resolve_checkpoint("lerobot/pi05_libero_finetuned_v044")
    pi05_helper = ModelPreprocHelper(pi05_dir)

    stream_counts = tuple(int(x.strip()) for x in args.streams.split(",") if x.strip())
    levels = ["l1", "l2", "l3"]
    speeds = list(CANONICAL_SPEEDS)

    if args.mode == "smoke":
        candidate_tasks = [("libero_spatial", 0), ("libero_10", 0)]
        static_inits = [0]
        dyn_trajs = [0]
        dyn_max_steps = 120
    else:
        candidate_tasks = [(suite, tid) for suite in CANONICAL_SUITES for tid in range(10)]
        static_inits = [int(x.strip()) for x in args.static_inits.split(",") if x.strip()]
        dyn_trajs = [int(x.strip()) for x in args.traj_indices.split(",") if x.strip()]
        dyn_max_steps = int(args.dyn_max_steps)

    work_dir = pathlib.Path(f"/tmp/libero_dyn_work_s{args.shard_index}")
    os.environ["LIBERO_DYNAMIC_LOG_DIR"] = str(work_dir / "libero_dynamic_logs")
    os.environ["LIBERO_DYNAMIC_AUTO_FINALIZE"] = "false"
    dyn_args_by_speed, items_by_id, all_static_scenes, all_dyn_scenes = build_static_and_dynamic_scenes(
        args.dyn_repo,
        work_dir,
        levels,
        speeds,
        candidate_tasks,
        static_inits,
        candidate_tasks,
        dyn_trajs,
        dyn_max_steps,
    )
    if not all_dyn_scenes:
        raise RuntimeError("No LIBERO-Dynamic scenes were built; check config discovery!")

    # Compute uniform prompt_len across the ENTIRE unsharded benchmark so all shards compile identical shapes
    pi05_plen = max(pi05_helper.prompt_len_for(sc.instruction) for sc in (all_static_scenes + all_dyn_scenes))

    static_scenes = all_static_scenes[args.shard_index :: args.num_shards]
    dyn_scenes = all_dyn_scenes[args.shard_index :: args.num_shards]
    print(
        f"[RealtimeBench] Shard {args.shard_index}/{args.num_shards}: "
        f"{len(static_scenes)}/{len(all_static_scenes)} static LIBERO scenes, "
        f"{len(dyn_scenes)}/{len(all_dyn_scenes)} LIBERO-Dynamic scenes (prompt_len={pi05_plen})",
        flush=True,
    )

    static_groups = _group_static_scenes(static_scenes)
    dyn_groups = _group_dynamic_scenes(dyn_scenes, dyn_max_steps)
    keep_raw = args.num_shards > 1

    embodiment = LiberoRealtimeEmbodiment(items_by_id, dyn_args_by_speed)
    all_runs: List[Dict[str, Any]] = []
    try:
        for n_streams in stream_counts:
            factories: List[Tuple[str, Callable[[], PolicyBase]]] = [
                (
                    f"SyncChunk (K=10, N={n_streams})",
                    lambda n=n_streams: SyncChunkPolicy(
                        pi05_helper, prompt_len=pi05_plen, chunk_horizon=10, num_streams=n, compile_fast=False
                    ),
                ),
                (
                    f"CompiledSync (K=4, N={n_streams})",
                    lambda n=n_streams: SyncChunkPolicy(
                        pi05_helper, prompt_len=pi05_plen, chunk_horizon=4, num_streams=n, compile_fast=True
                    ),
                ),
                (
                    f"PyTorchAsync (SDPA, N={n_streams})",
                    lambda n=n_streams: PyTorchAsyncPolicy(
                        pi05_helper, prompt_len=pi05_plen, num_streams=n, slow_every_steps=2, fast_every_steps=4
                    ),
                ),
                (
                    f"Tempo (Engine, N={n_streams})",
                    lambda n=n_streams: TempoPolicy(
                        pi05_helper, prompt_len=pi05_plen, num_streams=n, slow_every_steps=2, fast_every_steps=4
                    ),
                ),
            ]

            def _flush_out_json() -> None:
                attach_paired_speedups(all_runs)
                args.out_json.parent.mkdir(parents=True, exist_ok=True)
                args.out_json.write_text(
                    json.dumps(
                        {
                            "gpu": gpu_name,
                            "harness": "robocurve/inspect-robots",
                            "shard_index": args.shard_index,
                            "num_shards": args.num_shards,
                            "benchmarks": [
                                "LIBERO-Static (40 tasks across spatial, object, goal, 10 @ 20 Hz real-time)",
                                "LIBERO-Dynamic (40 tasks x 3 motion families x 0.03 / 0.06 / 0.09 m/s @ 20 Hz real-time)",
                            ],
                            "runs": all_runs,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )

            suite_plan = [
                ("LIBERO-Static", static_groups),
                ("LIBERO-Dynamic", dyn_groups),
            ]
            if args.reverse_suite_order:
                suite_plan.reverse()

            for suite_tag, suite_groups in suite_plan:
                if not suite_groups:
                    continue
                print(f"\n=== [{suite_tag}] (20 Hz Real-Time) | N={n_streams} | shard={args.shard_index}/{args.num_shards} ===", flush=True)
                for label, factory in factories:
                    res = evaluate_policy_suite(
                        factory,
                        label,
                        n_streams,
                        embodiment,
                        suite_tag,
                        suite_groups,
                        args.log_dir,
                        live_jsonl=args.live_jsonl,
                        keep_raw_samples=keep_raw,
                    )
                    all_runs.append(res)
                    _flush_out_json()
    finally:
        embodiment.close()

    print(f"\n[RealtimeBench] Saved results to {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
