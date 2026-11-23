#!/usr/bin/env python3
"""
Real-time moving-object manipulation benchmark combining robocurve/inspect-robots with LIBERO-Dynamic.

Evaluates 3 `inspect_robots` Policy implementations head-to-head on identical 20 Hz wall-clock
`LIBERO-Dynamic` moving-object episodes (0.03 / 0.06 / 0.09 m/s):
  - `SyncChunkPolicy`: synchronous open-loop action chunking (`K=10`)
  - `PyTorchAsyncPolicy`: eager PyTorch dual-stream `SDPA` + per-stream `RWLock` baseline
  - `TempoPolicy`: `tempo.Engine` (sliced Slow VLM + lock-free paged cache + CUDA Graph Fast + C++ RTC)
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import pathlib
import threading
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

import numpy as np
import torch
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

import eval_dynamic_libero_style as legacy
from libero.libero.envs import OffScreenRenderEnv
from realtime_eval.runner import build_parser as build_dyn_parser
from realtime_eval.runtime import RealtimeEpisode


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


class LiberoDynamicRealtimeEmbodiment(EmbodimentBase):
    """Real-time 20 Hz wall-clock LIBERO-Dynamic MuJoCo embodiment for inspect-robots."""

    def __init__(self, items_by_id: Dict[str, Dict[str, Any]], dyn_args: argparse.Namespace) -> None:
        self._items_by_id = items_by_id
        self._dyn_args = dyn_args
        self._control_hz = float(dyn_args.control_freq)
        self._dt_s = 1.0 / self._control_hz
        self.info = EmbodimentInfo(
            name="libero-dynamic-franka-20hz",
            action_space=ACTION_SPACE,
            observation_space=OBS_SPACE,
            control_hz=self._control_hz,
            is_simulated=True,
            capabilities=frozenset({RESETTABLE, PRIVILEGED_SUCCESS, SELF_PACED}),
            morphology="franka-panda-7dof",
        )
        self._ep: Optional[RealtimeEpisode] = None
        self._env: Optional[OffScreenRenderEnv] = None
        self._instruction: str = ""
        self._obs_ready_t: float = 0.0
        self._diag: Optional[Dict[str, Any]] = None
        self._step_count: int = 0
        self.last_sync_stall_total_s: float = 0.0
        self.last_max_target_disp_m: float = 0.0
        self._target_init_pos: Dict[str, np.ndarray] = {}

    def _close_env(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
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
                disp = float(np.linalg.norm(pos[:2] - self._target_init_pos[name][:2]))
                if disp < 1.5 and disp > self.last_max_target_disp_m:
                    self.last_max_target_disp_m = disp

    def reset(self, scene: Scene, *, seed: int | None = None) -> Observation:
        self._close_env()
        gc.collect()
        item_id = str(scene.metadata["item_id"])
        item = self._items_by_id[item_id]
        ep = RealtimeEpisode(item, self._dyn_args, worker_slot=0)
        ep._configure_env_selection()
        suite, task, bddl = legacy.task_bddl(item["suite"], item["task_id"])
        init_states = suite.get_task_init_states(item["task_id"])

        env = OffScreenRenderEnv(
            bddl_file_name=str(bddl),
            camera_heights=256,
            camera_widths=256,
            control_freq=int(self._dyn_args.control_freq),
        )
        eff_seed = int(self._dyn_args.seed if seed is None else seed)
        env.seed(eff_seed)
        init_idx = int(self._dyn_args.init_state_id) % len(init_states)
        raw_obs, *_ = ep._setup_world(env, init_states[init_idx])

        self._ep = ep
        self._env = env
        self._instruction = task.language
        self._step_count = 0
        self.last_sync_stall_total_s = 0.0
        self.last_max_target_disp_m = 0.0
        self._target_init_pos.clear()
        self._measure_target_disp()

        img_main, img_wrist, state_8d = extract_libero_obs(raw_obs)
        self._obs_ready_t = time.perf_counter()
        return Observation(
            images={"agentview": img_main, "wrist": img_wrist},
            state={"eef_state": state_8d},
            instruction=self._instruction,
            state_time=0.0,
        )

    def step(self, action: Action) -> StepResult:
        assert self._ep is not None and self._env is not None
        t_enter = time.perf_counter()
        policy_wait_s = t_enter - self._obs_ready_t

        # Symmetric wall-clock physics rule:
        # 1. If policy inference exceeded one 50ms control period (blocking policy stall),
        #    advance continuous MuJoCo physics with robot joints held for the stall duration.
        # 2. Otherwise pace the control tick to 1.0x real-time 20 Hz so background GPU threads
        #    compute against real advancing wall-clock time.
        if policy_wait_s > self._dt_s + 0.004:
            stall_s = policy_wait_s - self._dt_s
            self._ep._apply_sync_latency(self._env, stall_s)
            self.last_sync_stall_total_s += stall_s

        act_np = np.clip(np.asarray(action.data, dtype=np.float32), -1.0, 1.0)
        raw_obs, reward, done, _info = self._ep._step_env(self._env, act_np)
        self._step_count += 1
        if self._step_count % 10 == 0:
            self._measure_target_disp()

        is_success = bool(self._env.check_success())
        terminated = bool(is_success or done)
        term_reason = "success" if is_success else ("done" if done else None)

        img_main, img_wrist, state_8d = extract_libero_obs(raw_obs)
        if policy_wait_s <= self._dt_s + 0.004:
            rem = self._dt_s - (time.perf_counter() - self._obs_ready_t)
            if rem > 0.0:
                time.sleep(rem)

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
    """Shared camera/state normalization and action unnormalization for SmolVLA or Pi0.5."""

    def __init__(self, arch: str, ckpt_dir: pathlib.Path) -> None:
        self.arch = arch
        self.ckpt_dir = ckpt_dir
        self._aux_stream = torch.cuda.Stream(device="cuda:0")
        if arch == "smolvla":
            self.img_size = 512
            self.stats = load_mean_std_stats(
                ckpt_dir,
                "policy_preprocessor_step_5_normalizer_processor.safetensors",
                "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
            )
            self.tok = load_hf_tokenizer("HuggingFaceTB/SmolVLM2-500M-Video-Instruct", checkpoint_dir=ckpt_dir)
        elif arch == "pi05":
            self.img_size = 224
            self.stats = load_mean_std_stats(
                ckpt_dir,
                "policy_preprocessor_step_2_normalizer_processor.safetensors",
                "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
            )
            self.tok = load_hf_tokenizer("mlx-community/paligemma-3b-mix-224-8bit", checkpoint_dir=ckpt_dir)
        else:
            raise ValueError(f"Unsupported arch {arch}")

    def prompt_len_for(self, instruction: str, raw_state_8d: np.ndarray) -> int:
        norm_st = (torch.from_numpy(raw_state_8d) - self.stats["state_mean"]) / (self.stats["state_std"] + 1e-8)
        if self.arch == "smolvla":
            p = instruction if instruction.endswith("\n") else f"{instruction}\n"
            return len(self.tok.encode(p).ids)
        cleaned = instruction.strip().replace("_", " ").replace("\n", " ")
        p = f"Task: {cleaned}, State: {discretize_pi05_state(norm_st, 32)};\nAction: "
        return len(self.tok.encode(p).ids) + 4

    def prepare_inputs(self, obs: Observation) -> Tuple[torch.Tensor, torch.Tensor, str]:
        c1 = resize_square_rgb(obs.images["agentview"], self.img_size)
        c2 = resize_square_rgb(obs.images["wrist"], self.img_size)
        cams = torch.stack([c1, c2], dim=0)
        st = torch.from_numpy(np.asarray(obs.state["eef_state"], dtype=np.float32))
        norm_st = (st - self.stats["state_mean"]) / (self.stats["state_std"] + 1e-8)
        return cams, norm_st, str(obs.instruction or "")

    def update_slow_stage(self, slow_stage: Any, instruction: str, norm_st: torch.Tensor) -> None:
        with torch.cuda.device("cuda:0"), torch.cuda.stream(self._aux_stream):
            if self.arch == "smolvla":
                slow_stage.set_state(norm_st)
            else:
                slow_stage.set_prompt(instruction, state=norm_st)
        self._aux_stream.synchronize()

    def unnormalize_action(self, norm_act: torch.Tensor) -> np.ndarray:
        act = norm_act.detach().cpu().float() * (self.stats["action_std"] + 1e-8) + self.stats["action_mean"]
        return np.clip(act.numpy(), -1.0, 1.0).astype(np.float32)


class SyncChunkPolicy(PolicyBase):
    """Standard synchronous open-loop ActionChunk policy (K steps per inference)."""

    def __init__(
        self,
        helper: ModelPreprocHelper,
        *,
        prompt_len: int,
        chunk_horizon: int = 10,
        num_streams: int = 1,
    ) -> None:
        self.helper = helper
        self.chunk_horizon = chunk_horizon
        self.num_streams = num_streams
        self.dep = ModelRegistry.build_for_engine(
            str(helper.ckpt_dir),
            slow_devices=[[0]],
            fast_devices=[0],
            fast_precision="bf16",
            prompt="pick up the object\n" if helper.arch == "smolvla" else None,
            prompt_len=prompt_len,
        )
        self.slow_stage = self.dep.slow_replicas[0][0]
        self.fast_head = self.dep.fast_shards[0]
        self.dev = torch.device("cuda:0")
        self.info = PolicyInfo(
            name=f"sync-chunk-k{chunk_horizon}-n{num_streams}",
            action_space=ACTION_SPACE,
            observation_space=OBS_SPACE,
            control_hz=20.0,
            checkpoint=str(helper.ckpt_dir.name),
        )
        self.config = PolicyConfig(action_horizon=chunk_horizon)
        self.infer_latencies_ms: List[float] = []

    def close(self) -> None:
        del self.slow_stage, self.fast_head, self.dep
        gc.collect()
        torch.cuda.empty_cache()

    def reset(self, scene: Scene) -> None:
        if self.helper.arch == "smolvla":
            p = scene.instruction if scene.instruction.endswith("\n") else f"{scene.instruction}\n"
            with torch.cuda.device("cuda:0"), torch.cuda.stream(self.helper._aux_stream):
                self.slow_stage.set_prompt(p)
            self.helper._aux_stream.synchronize()

    @torch.no_grad()
    def act(self, observation: Observation) -> ActionChunk:
        t0 = time.perf_counter()
        cams, norm_st, instr = self.helper.prepare_inputs(observation)
        self.helper.update_slow_stage(self.slow_stage, instr, norm_st)

        # Reflect N concurrent robot streams served per batch on 1 GPU
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
        self.infer_latencies_ms.append(lat_s * 1000.0)

        actions = [
            Action(data=self.helper.unnormalize_action(chunk_out[0, k]))
            for k in range(min(self.chunk_horizon, chunk_out.size(1)))
        ]
        return ActionChunk(actions=actions, control_hz=20.0, inference_latency_s=lat_s)


class PyTorchAsyncPolicy(PolicyBase):
    """Asynchronous PyTorch Dual-Stream (SDPA + RWLock) reactive policy (H=1)."""

    def __init__(
        self,
        helper: ModelPreprocHelper,
        *,
        prompt_len: int,
        num_streams: int = 1,
        replan_every_steps: int = 2,
    ) -> None:
        self.helper = helper
        self.num_streams = num_streams
        self.replan_every_steps = replan_every_steps
        self.dep = ModelRegistry.build_for_engine(
            str(helper.ckpt_dir),
            slow_devices=[[0]],
            fast_devices=[0],
            fast_precision="bf16",
            prompt="pick up the object\n" if helper.arch == "smolvla" else None,
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
        self._bg_running = False
        self._bg_thread: Optional[threading.Thread] = None
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

    def _stop_bg(self) -> None:
        self._bg_running = False
        if self._bg_thread is not None:
            self._bg_thread.join(timeout=2.0)
            self._bg_thread = None

    def close(self) -> None:
        self._stop_bg()
        with self.base.stats_lock:
            self.fast_ttac_ms.extend(self.base.fast_ttac_ms)
        self.base.stop()
        del self.base, self.slow_stage, self.dep
        gc.collect()
        torch.cuda.empty_cache()

    def _sibling_traffic_loop(self) -> None:
        sib_stream = torch.cuda.Stream(device=self.dev)
        with torch.cuda.device(self.dev), torch.cuda.stream(sib_stream):
            bg_s = torch.rand(self.dep.slow_obs_shape, dtype=torch.bfloat16, device=self.dev)
            bg_f = torch.randn(self.dep.fast_obs_shape, dtype=torch.bfloat16, device=self.dev)
        sib_stream.synchronize()
        tick = 0
        while self._bg_running:
            now_ns = time.monotonic_ns()
            for s in range(1, self.num_streams):
                if tick % self.replan_every_steps == 0:
                    self.base.push_slow(bg_s, stream_id=s, t_obs_ns=now_ns)
                    self.base.push_fast(bg_f, stream_id=s, t_obs_ns=now_ns)
            tick += 1
            time.sleep(0.050)

    def reset(self, scene: Scene) -> None:
        self._stop_bg()
        with self.base.stats_lock:
            self.fast_ttac_ms.extend(self.base.fast_ttac_ms)
            self.base.fast_ttac_ms.clear()
        for r in range(self.base.R):
            with self.base.slow_q_locks[r]:
                self.base.pending_slow[r].clear()
        for f in range(self.base.F):
            with self.base.fast_q_locks[f]:
                self.base.pending_fast[f].clear()
            with self.base.act_locks[f], torch.cuda.device(self.dev), torch.cuda.stream(self.base.sample_streams[f]):
                self.base.active_valid_gpu[f].zero_()
            self.base.sample_streams[f].synchronize()
        with self.base.stats_lock:
            self.base.active_valid = [False for _ in range(self.num_streams)]
        if self.helper.arch == "smolvla":
            p = scene.instruction if scene.instruction.endswith("\n") else f"{scene.instruction}\n"
            with torch.cuda.device("cuda:0"), torch.cuda.stream(self.helper._aux_stream):
                self.slow_stage.set_prompt(p)
            self.helper._aux_stream.synchronize()
        self._step_idx = 0

    def act(self, observation: Observation) -> ActionChunk:
        t0 = time.perf_counter()
        cams, norm_st, instr = self.helper.prepare_inputs(observation)

        if self._step_idx == 0:
            self.helper.update_slow_stage(self.slow_stage, instr, norm_st)
            with torch.cuda.device(self.dev), torch.cuda.stream(self._fg_stream):
                obs_bf16 = cams.to(self.dev, dtype=torch.bfloat16) / 255.0
                noise = torch.randn(self.dep.fast_obs_shape, dtype=torch.bfloat16, device=self.dev)
            self._fg_stream.synchronize()

            t_ns = time.monotonic_ns()
            for s in range(self.num_streams):
                self.base.push_slow(obs_bf16, stream_id=s, t_obs_ns=t_ns)
            deadline = time.monotonic() + 15.0
            while not all(self.base.state_t_obs[s] >= t_ns for s in range(self.num_streams)):
                if time.monotonic() > deadline:
                    raise TimeoutError("PyTorchAsyncPolicy timed out waiting for initial Slow state")
                time.sleep(0.001)

            t_fast_ns = time.monotonic_ns()
            for s in range(self.num_streams):
                self.base.push_fast(noise, stream_id=s, t_obs_ns=t_fast_ns)
            deadline = time.monotonic() + 15.0
            while True:
                with self.base.stats_lock:
                    ready_fast = all(self.base.active_t_obs[s] >= t_fast_ns for s in range(self.num_streams))
                if ready_fast:
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError("PyTorchAsyncPolicy timed out waiting for initial Fast chunk")
                time.sleep(0.001)

            act_t, *_ = self.base.sample_actions(time.monotonic_ns())
            with self.base.stats_lock:
                self.base.fast_ttac_ms.clear()
                self.base.commit_hard_underruns = 0
            if self.num_streams > 1:
                self._bg_running = True
                self._bg_thread = threading.Thread(target=self._sibling_traffic_loop, daemon=True)
                self._bg_thread.start()
        else:
            if self._step_idx % self.replan_every_steps == 0:
                self.helper.update_slow_stage(self.slow_stage, instr, norm_st)
                with torch.cuda.device(self.dev), torch.cuda.stream(self._fg_stream):
                    obs_bf16 = cams.to(self.dev, dtype=torch.bfloat16) / 255.0
                    noise = torch.randn(self.dep.fast_obs_shape, dtype=torch.bfloat16, device=self.dev)
                self._fg_stream.synchronize()
                now_ns = time.monotonic_ns()
                self.base.push_slow(obs_bf16, stream_id=0, t_obs_ns=now_ns)
                self.base.push_fast(noise, stream_id=0, t_obs_ns=now_ns)
            act_t, _fa, _sa, u = self.base.sample_actions(time.monotonic_ns())
            self.underruns += int(u)

        lat_s = time.perf_counter() - t0
        if self._step_idx > 0:
            self.act_latencies_us.append(lat_s * 1e6)
        self._step_idx += 1
        act = self.helper.unnormalize_action(act_t[0])
        return ActionChunk(actions=[Action(data=act)], control_hz=20.0, inference_latency_s=lat_s)


class TempoPolicy(PolicyBase):
    """Tempo Engine reactive policy (H=1): sliced VLM + lock-free cache + CUDA Graph Fast + C++ RTC."""

    def __init__(
        self,
        helper: ModelPreprocHelper,
        *,
        prompt_len: int,
        num_streams: int = 1,
        replan_every_steps: int = 2,
    ) -> None:
        self.helper = helper
        self.num_streams = num_streams
        self.replan_every_steps = replan_every_steps
        self.engine = Engine(
            str(helper.ckpt_dir),
            num_streams=num_streams,
            slow_devices=[0],
            fast_devices=[0],
            slow_rate_hz=10.0,
            fast_rate_hz=10.0,
            deadline_ms=45.0,
            step_dt_ms=50.0,
            fast_precision="bf16",
            prompt="pick up the object\n" if helper.arch == "smolvla" else None,
            model_kwargs={"prompt_len": prompt_len},
        )
        assert self.engine.deployment is not None
        self.slow_stage = self.engine.deployment.slow_replicas[0][0]
        self._bg_running = False
        self._bg_thread: Optional[threading.Thread] = None
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

    def _stop_bg(self) -> None:
        self._bg_running = False
        if self._bg_thread is not None:
            self._bg_thread.join(timeout=2.0)
            self._bg_thread = None

    def close(self) -> None:
        self._stop_bg()
        self.engine.stop()
        del self.slow_stage, self.engine
        gc.collect()
        torch.cuda.empty_cache()

    def _sibling_traffic_loop(self) -> None:
        sib_stream = torch.cuda.Stream(device="cuda:0")
        with torch.cuda.device("cuda:0"), torch.cuda.stream(sib_stream):
            bg_s = torch.randint(0, 256, self.engine.slow_obs_shape, dtype=torch.uint8, device="cuda:0")
        sib_stream.synchronize()
        tick = 0
        while self._bg_running:
            for s in range(1, self.num_streams):
                if tick % self.replan_every_steps == 0:
                    self.engine.push_slow(bg_s, stream_id=s)
                    self.engine.push_fast(stream_id=s)
            tick += 1
            time.sleep(0.050)

    def reset(self, scene: Scene) -> None:
        self._stop_bg()
        self.fast_ttac_ms.extend(self.engine.telemetry().fast_ttac_ms)
        self.engine.reset()
        if self.helper.arch == "smolvla":
            p = scene.instruction if scene.instruction.endswith("\n") else f"{scene.instruction}\n"
            with torch.cuda.device("cuda:0"), torch.cuda.stream(self.helper._aux_stream):
                self.slow_stage.set_prompt(p)
            self.helper._aux_stream.synchronize()
        self._step_idx = 0

    def act(self, observation: Observation) -> ActionChunk:
        t0 = time.perf_counter()
        cams, norm_st, instr = self.helper.prepare_inputs(observation)

        if self._step_idx == 0:
            self.helper.update_slow_stage(self.slow_stage, instr, norm_st)
            for s in range(self.num_streams):
                self.engine.push_slow(cams, stream_id=s)
            self.engine.wait_ready()
            self.engine.reset_stats()
            actions, *_ = self.engine.sample_actions(cpu=True)
            if self.num_streams > 1:
                self._bg_running = True
                self._bg_thread = threading.Thread(target=self._sibling_traffic_loop, daemon=True)
                self._bg_thread.start()
        else:
            if self._step_idx % self.replan_every_steps == 0:
                self.helper.update_slow_stage(self.slow_stage, instr, norm_st)
                self.engine.push_slow(cams)
                self.engine.push_fast()
            actions, _fa, _sa, u = self.engine.sample_actions(cpu=True)
            self.underruns += int(u)

        lat_s = time.perf_counter() - t0
        if self._step_idx > 0:
            self.act_latencies_us.append(lat_s * 1e6)
        self._step_idx += 1
        act = self.helper.unnormalize_action(actions[0])
        return ActionChunk(actions=[Action(data=act)], control_hz=20.0, inference_latency_s=lat_s)


def build_dynamic_scenes(
    dyn_repo: pathlib.Path,
    output_dir: pathlib.Path,
    levels: Sequence[str],
    traj_indices: Sequence[int],
    selected_tasks: Sequence[Tuple[str, int]],
) -> Tuple[argparse.Namespace, Dict[str, Dict[str, Any]], Dict[str, List[Scene]]]:
    parser = build_dyn_parser()
    argv = [
        "--config-dir",
        str(dyn_repo / "libero" / "libero" / "configs"),
        "--output-dir",
        str(output_dir),
        "--levels",
        *levels,
        "--traj-indices",
        *[str(i) for i in traj_indices],
        "--reviewed-old-physics",
        "--keep-static-support-contact",
        "--static-support-clearance-m",
        "0.001",
        "--no-save-videos",
        "--no-save-model-input-videos",
        "--no-save-step-trace",
        "--stop-on-success",
    ]
    # Check config dir location inside LIBERO-Dynamic
    cand_dirs = [
        dyn_repo / "libero" / "libero" / "dynamic_configs",
        dyn_repo / "libero" / "libero" / "configs",
        dyn_repo / "configs",
        dyn_repo / "templates",
    ]
    for cd in cand_dirs:
        if cd.exists() and list(cd.glob("*dynamic*.yaml")):
            argv[1] = str(cd)
            break
    else:
        found = list(dyn_repo.rglob("libero_dynamic_l1.yaml")) + list(dyn_repo.rglob("libero_dynamic_manual_l1.yaml"))
        if found:
            argv[1] = str(found[0].parent)

    dyn_args = parser.parse_args(argv)
    legacy.apply_reviewed_old_physics_profile(dyn_args)
    dyn_args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg_dir = dyn_args.config_dir.expanduser().resolve()
    config_path_by_level = legacy.write_effective_config_files(cfg_dir, dyn_args.output_dir, dyn_args.levels, dyn_args)
    all_items = legacy.build_items(
        cfg_dir,
        dyn_args.levels,
        dyn_args.traj_indices,
        keep_static_support_contact=bool(dyn_args.keep_static_support_contact),
        static_support_clearance_m=float(dyn_args.static_support_clearance_m),
        config_path_by_level=config_path_by_level,
    )

    wanted = set(selected_tasks)
    items_by_id: Dict[str, Dict[str, Any]] = {}
    scenes_by_level: Dict[str, List[Scene]] = {lvl: [] for lvl in levels}
    seen_per_level_task_traj: set[Tuple[str, str, int, int]] = set()

    for item in all_items:
        key = (str(item["suite"]), int(item["task_id"]))
        if key not in wanted:
            continue
        lvl = str(item["level"])
        t_idx = int(item["traj_index"])
        dedup_k = (lvl, key[0], key[1], t_idx)
        if dedup_k in seen_per_level_task_traj:
            continue
        seen_per_level_task_traj.add(dedup_k)

        suite_obj, task_obj, _ = legacy.task_bddl(key[0], key[1])
        del suite_obj
        iid = str(item["item_id"])
        items_by_id[iid] = item
        scenes_by_level[lvl].append(
            Scene(
                id=iid,
                instruction=task_obj.language,
                init_seed=0,
                metadata={
                    "item_id": iid,
                    "level": lvl,
                    "suite": key[0],
                    "task_id": key[1],
                    "traj_index": t_idx,
                },
            )
        )
    return dyn_args, items_by_id, scenes_by_level


def evaluate_policy_across_levels(
    policy_factory,
    policy_label: str,
    embodiment: LiberoDynamicRealtimeEmbodiment,
    scenes_by_level: Dict[str, List[Scene]],
    max_steps: int,
) -> Dict[str, Any]:
    policy = policy_factory()
    level_speed_mps = {"l1": 0.03, "l2": 0.06, "l3": 0.09}
    per_level: Dict[str, Any] = {}
    total_succ = 0
    total_ep = 0
    max_disp = 0.0
    total_stall_s = 0.0

    try:
        for lvl, scenes in scenes_by_level.items():
            if not scenes:
                continue
            task = Task(
                name=f"libero-dynamic-{lvl}",
                scenes=scenes,
                scorer="success_at_end",
                max_steps=max_steps,
            )
            logs = inspect_eval(
                task,
                policy=policy,
                embodiment=embodiment,
                log_dir="/tmp/inspect_logs",
                fail_on_error=True,
                store_frames=False,
                store_records=False,
            )
            log = logs[0]
            lvl_succ = sum(
                1
                for sr in log.samples
                if (sr.reduced.get("success_at_end") or 0.0) >= 0.5
            )
            lvl_n = len(scenes)
            total_succ += lvl_succ
            total_ep += lvl_n
            max_disp = max(max_disp, embodiment.last_max_target_disp_m)
            total_stall_s += embodiment.last_sync_stall_total_s
            sr_pct = round(100.0 * lvl_succ / max(1, lvl_n), 1)
            per_level[lvl] = {
                "speed_mps": level_speed_mps.get(lvl, 0.0),
                "episodes": lvl_n,
                "successes": lvl_succ,
                "success_rate_pct": sr_pct,
            }
            print(
                f"  [{policy_label}] level={lvl} ({level_speed_mps.get(lvl)} m/s): "
                f"{lvl_succ}/{lvl_n} ({sr_pct:.1f}%) | max_target_disp={embodiment.last_max_target_disp_m:.3f}m",
                flush=True,
            )

        if hasattr(policy, "engine"):
            policy.fast_ttac_ms.extend(policy.engine.telemetry().fast_ttac_ms)
        elif hasattr(policy, "base") and policy.base is not None:
            with policy.base.stats_lock:
                policy.fast_ttac_ms.extend(policy.base.fast_ttac_ms)
                policy.base.fast_ttac_ms.clear()

        summary: Dict[str, Any] = {
            "policy": policy_label,
            "total_episodes": total_ep,
            "total_successes": total_succ,
            "overall_success_rate_pct": round(100.0 * total_succ / max(1, total_ep), 1),
            "per_level": per_level,
            "max_target_disp_m": round(max_disp, 4),
        }
        if hasattr(policy, "infer_latencies_ms") and policy.infer_latencies_ms:
            arr = np.asarray(policy.infer_latencies_ms, dtype=np.float64)
            summary["chunk_infer_p50_ms"] = round(float(np.percentile(arr, 50)), 2)
            summary["chunk_infer_p99_ms"] = round(float(np.percentile(arr, 99)), 2)
        if hasattr(policy, "fast_ttac_ms") and policy.fast_ttac_ms:
            arr = np.asarray(policy.fast_ttac_ms, dtype=np.float64)
            summary["fast_ttac_p50_ms"] = round(float(np.percentile(arr, 50)), 2)
            summary["fast_ttac_p99_ms"] = round(float(np.percentile(arr, 99)), 2)
        if hasattr(policy, "act_latencies_us") and policy.act_latencies_us:
            arr = np.asarray(policy.act_latencies_us, dtype=np.float64)
            summary["act_p50_us"] = round(float(np.percentile(arr, 50)), 2)
        if hasattr(policy, "underruns"):
            summary["underruns"] = int(policy.underruns)
        return summary
    finally:
        policy.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--dyn-repo", type=pathlib.Path, required=True, help="Path to cloned LIBERO-Dynamic repository")
    parser.add_argument("--out-json", type=pathlib.Path, default=pathlib.Path("benchmarks/libero_dynamic_eval.json"))
    args = parser.parse_args()

    gpu_name = torch.cuda.get_device_name(0)
    print(f"[DynamicEval] GPU={gpu_name} | mode={args.mode}", flush=True)

    _, smolvla_dir = resolve_checkpoint("lerobot/smolvla_libero")

    _, pi05_dir = resolve_checkpoint("lerobot/pi05_libero_finetuned_v044")
    pi05_helper = ModelPreprocHelper("pi05", pi05_dir)
    smolvla_helper = ModelPreprocHelper("smolvla", smolvla_dir)

    if args.mode == "smoke":
        selected_tasks = [("libero_spatial", 0)]
        traj_indices = [0]
        max_steps = 200
    else:
        selected_tasks = [
            ("libero_spatial", 0),
            ("libero_object", 0),
            ("libero_goal", 0),
        ]
        traj_indices = [0, 1]
        max_steps = 200

    levels = ["l1", "l2", "l3"]
    dyn_args, items_by_id, scenes_by_level = build_dynamic_scenes(
        args.dyn_repo,
        pathlib.Path("/tmp/libero_dyn_work"),
        levels,
        traj_indices,
        selected_tasks,
    )
    total_scenes = sum(len(v) for v in scenes_by_level.values())
    print(f"[DynamicEval] Built {total_scenes} dynamic scenes across levels={levels}", flush=True)
    if total_scenes == 0:
        raise RuntimeError("No LIBERO-Dynamic scenes were built; check config discovery!")

    # Compute max prompt_len across selected tasks
    dummy_st = np.zeros(8, dtype=np.float32)
    pi05_plen = max(
        pi05_helper.prompt_len_for(sc.instruction, dummy_st)
        for scs in scenes_by_level.values()
        for sc in scs
    )
    smol_plen = max(
        smolvla_helper.prompt_len_for(sc.instruction, dummy_st)
        for scs in scenes_by_level.values()
        for sc in scs
    )

    embodiment = LiberoDynamicRealtimeEmbodiment(items_by_id, dyn_args)
    all_runs: List[Dict[str, Any]] = []
    try:
        # Suite 1: Flagship 3.3B Pi0.5 on LIBERO-Dynamic (N=1 in smoke; N=1 and N=2 in full)
        pi05_streams = (1,) if args.mode == "smoke" else (1, 2)
        for n_streams in pi05_streams:
            print(f"\n=== Pi0.5 (3.3B) | Concurrent Robot Streams N={n_streams} ===", flush=True)
            for label, factory in (
                (
                    f"SyncChunk (K=10, N={n_streams})",
                    lambda n=n_streams: SyncChunkPolicy(pi05_helper, prompt_len=pi05_plen, chunk_horizon=10, num_streams=n),
                ),
                (
                    f"PyTorchAsync (SDPA, N={n_streams})",
                    lambda n=n_streams: PyTorchAsyncPolicy(pi05_helper, prompt_len=pi05_plen, num_streams=n, replan_every_steps=2),
                ),
                (
                    f"Tempo (Engine, N={n_streams})",
                    lambda n=n_streams: TempoPolicy(pi05_helper, prompt_len=pi05_plen, num_streams=n, replan_every_steps=2),
                ),
            ):
                res = evaluate_policy_across_levels(factory, label, embodiment, scenes_by_level, max_steps)
                res["model"] = "lerobot/pi05_libero_finetuned_v044"
                res["num_streams"] = n_streams
                all_runs.append(res)

        # Suite 2: 450M SmolVLA on LIBERO-Dynamic under 4-Robot Fleet Load (N=4 streams on 1 GPU)
        if args.mode == "full":
            n_streams = 4
            print(f"\n=== SmolVLA (450M) | Concurrent Robot Streams N={n_streams} ===", flush=True)
            for label, factory in (
                (
                    f"SyncChunk (K=10, N={n_streams})",
                    lambda n=n_streams: SyncChunkPolicy(smolvla_helper, prompt_len=smol_plen, chunk_horizon=10, num_streams=n),
                ),
                (
                    f"PyTorchAsync (SDPA, N={n_streams})",
                    lambda n=n_streams: PyTorchAsyncPolicy(smolvla_helper, prompt_len=smol_plen, num_streams=n, replan_every_steps=2),
                ),
                (
                    f"Tempo (Engine, N={n_streams})",
                    lambda n=n_streams: TempoPolicy(smolvla_helper, prompt_len=smol_plen, num_streams=n, replan_every_steps=2),
                ),
            ):
                res = evaluate_policy_across_levels(factory, label, embodiment, scenes_by_level, max_steps)
                res["model"] = "lerobot/smolvla_libero"
                res["num_streams"] = n_streams
                all_runs.append(res)
    finally:
        embodiment.close()

    payload = {
        "gpu": gpu_name,
        "harness": "robocurve/inspect-robots",
        "benchmark": "LIBERO-Dynamic (20 Hz real-time advancing MuJoCo physics)",
        "runs": all_runs,
    }
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[DynamicEval] Saved results to {args.out_json}", flush=True)
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__":
    main()
