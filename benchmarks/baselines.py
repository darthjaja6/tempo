from __future__ import annotations

import itertools
import math
import threading
import time
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from tempo.channel import TemporalChannel
from tempo.models.base import SlowBackbone


class EagerCaptureContext:
    """In-memory capture context executing SlowBackbone.forward() eagerly without TSC pools."""

    def __init__(
        self,
        channel: TemporalChannel,
        device: torch.device,
        *,
        prompt_slot_ids: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None,
    ) -> None:
        self.channel = channel
        self.mode = channel.mode
        self.frame_seq_len = channel.frame_seq_len
        self.num_kv_heads = channel.num_kv_heads
        self.head_dim = channel.head_dim
        self.device = device
        self.prompt_slot_ids = prompt_slot_ids
        self.seq_lens = seq_lens
        self._probe_mode: bool = False
        self.kv_by_layer: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.hidden: Optional[torch.Tensor] = None

    def slice_point(self) -> None:
        return None

    def write_layer_kv(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
        self.kv_by_layer[int(layer_idx)] = (k.contiguous(), v.contiguous())

    def write_hidden_state(self, hidden: torch.Tensor) -> None:
        self.hidden = hidden.contiguous()


class EagerDenseStateView:
    """Dense FlashAttention-2 / tensor view executing FastHead.forward() eagerly without Paged TSC."""

    def __init__(
        self,
        channel: TemporalChannel,
        *,
        batch_ks: Optional[Sequence[torch.Tensor]] = None,
        batch_vs: Optional[Sequence[torch.Tensor]] = None,
        batch_hidden: Optional[torch.Tensor] = None,
        delta_sec: Optional[torch.Tensor] = None,
        prefix_valid: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None,
        cpu_seq_lens: Optional[Sequence[int]] = None,
    ) -> None:
        self.mode = channel.mode
        self.max_seq_len = channel.max_seq_len
        self.prefix_valid = prefix_valid
        self._ks = batch_ks
        self._vs = batch_vs
        self._hidden = batch_hidden
        if batch_ks is not None:
            B = batch_ks[0].size(0)
            full_s = batch_ks[0].size(1)
            dev = batch_ks[0].device
        else:
            assert batch_hidden is not None
            B = batch_hidden.size(0)
            full_s = batch_hidden.size(1)
            dev = batch_hidden.device

        if seq_lens is not None or cpu_seq_lens is not None:
            if cpu_seq_lens is not None:
                s_min = int(min(cpu_seq_lens))
                s_max = int(max(cpu_seq_lens))
            else:
                assert seq_lens is not None
                s_min = int(seq_lens.min().item())
                s_max = int(seq_lens.max().item())
            if s_min < 1:
                raise ValueError(f"EagerDenseStateView requires seq_lens >= 1, got min={s_min}")
            self._has_seq_mask = s_min != s_max
            self._slice_seq_len: Optional[int] = (
                min(full_s, ((s_max + 7) // 8) * 8) if self._has_seq_mask else min(s_max, full_s)
            )
            self._need_hidden_mask = s_min < full_s
            raw_seq = (
                seq_lens.to(device=dev, dtype=torch.int32)
                if seq_lens is not None
                else torch.tensor(list(cpu_seq_lens), dtype=torch.int32, device=dev)  # type: ignore[arg-type]
            )
            self.seq_lens = raw_seq.clamp(1, full_s).contiguous()
        else:
            self._slice_seq_len = None
            self._has_seq_mask = False
            self._need_hidden_mask = False
            self.seq_lens = torch.full((B,), full_s, dtype=torch.int32, device=dev)

        self._delta_sec = (
            delta_sec
            if delta_sec is not None
            else torch.zeros((B, channel.max_seq_len), dtype=torch.float32, device=dev)
        )

    def splice_prefix(self, current: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.prefix_valid is None:
            return target
        return torch.where(self.prefix_valid.view(-1, *([1] * (current.ndim - 1))), target, current)

    def cross_attend(
        self,
        q: torch.Tensor,
        layer_idx: int,
        out: Optional[torch.Tensor] = None,
        *,
        return_lse: bool = False,
        out_lse: Optional[torch.Tensor] = None,
    ):
        assert self._ks is not None and self._vs is not None
        k = self._ks[layer_idx]  # [B, S, H_kv, D]
        v = self._vs[layer_idx]
        if self._slice_seq_len is not None:
            k = k[:, : self._slice_seq_len]
            v = v[:, : self._slice_seq_len]
        B, Q_len, H_q, D = q.shape
        H_kv = k.size(2)
        S = k.size(1)
        scale = 1.0 / math.sqrt(D)
        groups = H_q // H_kv

        qt = q.transpose(1, 2).contiguous()
        kt = k.transpose(1, 2).repeat_interleave(groups, dim=1).contiguous()
        vt = v.transpose(1, 2).repeat_interleave(groups, dim=1).contiguous()

        if self._has_seq_mask:
            if return_lse:
                s_pad = ((S + 7) // 8) * 8
                kt_p = F.pad(kt, (0, 0, 0, s_pad - S)) if s_pad > S else kt
                vt_p = F.pad(vt, (0, 0, 0, s_pad - S)) if s_pad > S else vt
                key_mask_p = torch.arange(s_pad, device=q.device).view(1, 1, 1, s_pad) < self.seq_lens.clamp(
                    1, S
                ).view(B, 1, 1, 1)
                attn_bias = (
                    torch.where(key_mask_p, 0.0, float("-inf"))
                    .to(dtype=qt.dtype)
                    .expand(B, H_q, Q_len, s_pad)
                    .contiguous()
                )
                attn_out, lse, *_ = torch.ops.aten._scaled_dot_product_efficient_attention(
                    qt, kt_p, vt_p, attn_bias, True, 0.0, False, scale=scale
                )
                res = attn_out.transpose(1, 2).contiguous()
                lse_t = lse[:, :, :Q_len].transpose(1, 2).contiguous()
                if out is not None:
                    out.copy_(res)
                    res = out
                if out_lse is not None:
                    out_lse.copy_(lse_t)
                    lse_t = out_lse
                return res, lse_t
            key_mask = torch.arange(S, device=q.device).view(1, 1, 1, S) < self.seq_lens.clamp(1, S).view(
                B, 1, 1, 1
            )
            res = F.scaled_dot_product_attention(qt, kt, vt, attn_mask=key_mask).transpose(1, 2).contiguous()
            if out is not None:
                out.copy_(res)
                return out
            return res

        if return_lse:
            attn_out, lse, *_ = torch.ops.aten._scaled_dot_product_flash_attention(
                qt, kt, vt, 0.0, False, False, scale=scale
            )
            res = attn_out.transpose(1, 2).contiguous()
            lse_t = lse[:, :, :Q_len].transpose(1, 2).contiguous()
            if out is not None:
                out.copy_(res)
                res = out
            if out_lse is not None:
                out_lse.copy_(lse_t)
                lse_t = out_lse
            return res, lse_t

        res = F.scaled_dot_product_attention(qt, kt, vt).transpose(1, 2).contiguous()
        if out is not None:
            out.copy_(res)
            return out
        return res

    def read_hidden(
        self,
        out: Optional[torch.Tensor] = None,
        out_delta_sec: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self._hidden is not None
        h = self._hidden
        d = self._delta_sec
        S = h.size(1)
        if self._need_hidden_mask:
            valid = torch.arange(S, device=h.device).view(1, S) < self.seq_lens.view(-1, 1)
            h = torch.where(valid.unsqueeze(-1), h, torch.zeros_like(h))
            d = torch.where(valid, d, torch.full_like(d, -1.0))
        if out is not None:
            out.copy_(h)
            h = out
        if out_delta_sec is not None:
            out_delta_sec.copy_(d)
            d = out_delta_sec
        return h, d


@torch.no_grad()
def eager_reference_forward(
    slow_stages: Sequence[nn.Module],
    fast_head: nn.Module,
    channel: TemporalChannel,
    slow_obs: torch.Tensor,
    fast_obs: torch.Tensor,
    frozen_prefix: torch.Tensor,
    *,
    prefix_valid: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Single-stream eager reference pass using identical weights and FlashAttention-2."""
    cur: object = slow_obs
    all_kv: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    hidden: Optional[torch.Tensor] = None
    seq_lens: Optional[torch.Tensor] = None
    B_ref = int(slow_obs.size(0))
    for stg in slow_stages:
        if isinstance(stg, SlowBackbone):
            stg.ensure_stream_capacity(B_ref)
        dev = next(itertools.chain(stg.parameters(), stg.buffers())).device
        stg_seq = seq_lens.to(device=dev) if seq_lens is not None else None
        ctx = EagerCaptureContext(channel, dev, seq_lens=stg_seq)
        if isinstance(cur, torch.Tensor):
            cur = cur.to(device=dev, dtype=torch.bfloat16)
        cur = stg(cur, ctx)
        if ctx.seq_lens is not None:
            seq_lens = ctx.seq_lens
        all_kv.update(ctx.kv_by_layer)
        if ctx.hidden is not None:
            hidden = ctx.hidden

    f_dev = next(itertools.chain(fast_head.parameters(), fast_head.buffers())).device
    pv = prefix_valid.to(device=f_dev, dtype=torch.bool) if prefix_valid is not None else None
    f_seq = seq_lens.to(device=f_dev, dtype=torch.int32) if seq_lens is not None else None
    if channel.mode == "kv":
        ks = [all_kv[l][0].to(f_dev) for l in range(channel.layers)]
        vs = [all_kv[l][1].to(f_dev) for l in range(channel.layers)]
        view = EagerDenseStateView(channel, batch_ks=ks, batch_vs=vs, prefix_valid=pv, seq_lens=f_seq)
    else:
        assert hidden is not None
        view = EagerDenseStateView(channel, batch_hidden=hidden.to(f_dev), prefix_valid=pv, seq_lens=f_seq)

    return fast_head(
        fast_obs.to(device=f_dev, dtype=torch.bfloat16),
        view,
        frozen_prefix.to(device=f_dev, dtype=torch.bfloat16),
    )


class PyTorchDualStreamBaseline:
    """Strong multi-GPU PyTorch Dual-Stream Serving Baseline.

    Runs the exact same SlowBackbone and FastHead weights across identical (R x PP(P)) : F
    GPU topologies using High-Priority Fast CUDA Streams (priority=-1), Background Slow CUDA
    Streams (priority=0), FlashAttention-2 SDPA, per-stream locks preventing torn state reads,
    and vectorized GPU linear horizon interpolation.
    """

    def __init__(
        self,
        channel: TemporalChannel,
        *,
        action_dim: int,
        horizon_steps: int,
        prefix_steps: int,
        slow_devices: Sequence[Sequence[int]],
        fast_devices: Sequence[int],
        slow_replicas: Sequence[Sequence[nn.Module]],
        fast_experts: Mapping[int, nn.Module],
        num_streams: int,
        slow_batch_size: int = 4,
        max_fast_batch_size: int = 16,
        deadline_ms: float = 25.0,
        step_dt_ms: float = 2.0,
        interpolate: bool = True,
        anchor_to_slow_obs: bool = False,
    ) -> None:
        self.channel = channel
        self.action_dim = action_dim
        self.horizon_steps = horizon_steps
        self.prefix_steps = prefix_steps
        self.num_streams = num_streams
        self.deadline_ns = int(round(deadline_ms * 1e6))
        self.step_dt_ns = int(round(step_dt_ms * 1e6))
        self.horizon_ns = (horizon_steps - 1) * self.step_dt_ns
        self.interpolate = bool(interpolate)
        self.anchor_to_slow_obs = bool(anchor_to_slow_obs)

        P = len(slow_devices[0])
        r_act = min(num_streams, len(slow_devices))
        f_act = min(num_streams, len(fast_devices))
        self.active_slow_devs = [list(slow_devices[i]) for i in range(r_act)]
        self.active_fast_devs = list(fast_devices[:f_act])
        self.slow_replicas = [slow_replicas[i] for i in range(r_act)]
        for rep in self.slow_replicas:
            for stg in rep:
                if isinstance(stg, SlowBackbone):
                    stg.ensure_stream_capacity(num_streams)
        self.fast_experts = fast_experts
        self.R = len(self.active_slow_devs)
        self.P = P
        self.F = len(self.active_fast_devs)

        self.slow_batch_size = max(1, min(slow_batch_size, (num_streams + self.R - 1) // self.R))
        self.max_fast_batch_size = max(1, min(max_fast_batch_size, (num_streams + self.F - 1) // self.F))
        self.primary_dev = torch.device(f"cuda:{self.active_fast_devs[0]}")

        self.all_devices: List[int] = []
        for pipe in self.active_slow_devs:
            for d in pipe:
                if d not in self.all_devices:
                    self.all_devices.append(d)
        for d in self.active_fast_devs:
            if d not in self.all_devices:
                self.all_devices.append(d)

        self.p2p_ok: Dict[Tuple[int, int], bool] = {}
        ref_probe = torch.tensor([1.25, -2.5, 3.75, -4.5], dtype=torch.float32)
        for d_dst in self.all_devices:
            for d_src in self.all_devices:
                if d_dst == d_src:
                    self.p2p_ok[(d_dst, d_src)] = True
                elif torch.cuda.can_device_access_peer(d_dst, d_src):
                    src_p = ref_probe.to(f"cuda:{d_src}")
                    dst_p = src_p.to(f"cuda:{d_dst}")
                    torch.cuda.synchronize(d_dst)
                    self.p2p_ok[(d_dst, d_src)] = bool(torch.equal(dst_p.cpu(), ref_probe))
                else:
                    self.p2p_ok[(d_dst, d_src)] = False

        self.fast_shard_of = [s % self.F for s in range(num_streams)]
        self.slow_rep_of = [s % self.R for s in range(num_streams)]
        self.shard_global_ids: List[List[int]] = [[] for _ in range(self.F)]
        self.local_fast_id = [0 for _ in range(num_streams)]
        for s in range(num_streams):
            f = self.fast_shard_of[s]
            self.local_fast_id[s] = len(self.shard_global_ids[f])
            self.shard_global_ids[f].append(s)

        self.fast_ingest_streams = [
            torch.cuda.Stream(device=f"cuda:{d}", priority=0) for d in self.active_fast_devs
        ]
        self.fast_exec_streams = [
            torch.cuda.Stream(device=f"cuda:{d}", priority=-1) for d in self.active_fast_devs
        ]
        self.state_copy_streams = [
            [torch.cuda.Stream(device=f"cuda:{d}", priority=0) for d in self.active_fast_devs]
            for _ in range(self.R)
        ]
        self.sample_streams = [
            torch.cuda.Stream(device=f"cuda:{d}", priority=-1) for d in self.active_fast_devs
        ]
        self.primary_out_stream = torch.cuda.Stream(device=self.primary_dev, priority=-1)
        self.slow_ingest_streams = [
            torch.cuda.Stream(device=f"cuda:{pipe[0]}", priority=0) for pipe in self.active_slow_devs
        ]
        self.slow_stage_streams = [
            [torch.cuda.Stream(device=f"cuda:{d}", priority=0) for d in pipe]
            for pipe in self.active_slow_devs
        ]

        self._clock_cv = threading.Condition(threading.Lock())
        self._clock_paused = False
        self._clock_pause_start_ns = 0
        self._clock_paused_accum_ns = 0

        # Per-stream dense Slow state protected by per-stream mutexes
        self.stream_locks = [threading.Lock() for _ in range(num_streams)]
        self.kv_packed: List[torch.Tensor] = []
        self.hidden_packed: List[torch.Tensor] = []
        self.state_t_obs: List[int] = []
        self.state_seq_len: List[int] = []
        S = channel.frame_seq_len
        for s in range(num_streams):
            f_dev = torch.device(f"cuda:{self.active_fast_devs[self.fast_shard_of[s]]}")
            if channel.mode == "kv":
                self.kv_packed.append(
                    torch.zeros(
                        (channel.layers, 2, 1, S, channel.num_kv_heads, channel.head_dim),
                        dtype=torch.bfloat16,
                        device=f_dev,
                    )
                )
            else:
                self.hidden_packed.append(
                    torch.zeros((1, S, channel.feature_dim), dtype=torch.bfloat16, device=f_dev)
                )
            self.state_t_obs.append(0)
            self.state_seq_len.append(S)

        self.slow_q_locks = [threading.Lock() for _ in range(self.R)]
        self.pending_slow: List[Dict[int, Tuple[int, torch.Tensor, torch.cuda.Event, int, int]]] = [
            {} for _ in range(self.R)
        ]
        self.inflight_slow_row: List[int] = [-1 for _ in range(num_streams)]
        self.fast_q_locks = [threading.Lock() for _ in range(self.F)]
        self.pending_fast: List[Dict[int, Tuple[int, torch.Tensor, torch.cuda.Event]]] = [
            {} for _ in range(self.F)
        ]

        self.act_locks = [threading.Lock() for _ in range(self.F)]
        self.stats_lock = threading.Lock()
        self.active_chunks: List[torch.Tensor] = []
        self.active_t_obs_gpu: List[torch.Tensor] = []
        self.active_slow_t_obs_gpu: List[torch.Tensor] = []
        self.active_valid_gpu: List[torch.Tensor] = []
        self.shard_arange: List[torch.Tensor] = []
        self.zero_prefixes: List[torch.Tensor] = []

        self.h_commit_lids: List[torch.Tensor] = []
        self.h_commit_tobs: List[torch.Tensor] = []
        self.h_commit_stobs: List[torch.Tensor] = []
        self.d_commit_lids: List[torch.Tensor] = []
        self.d_commit_tobs: List[torch.Tensor] = []
        self.d_commit_stobs: List[torch.Tensor] = []

        self.h_sample_out: List[torch.Tensor] = []
        self.h_sample_fa: List[torch.Tensor] = []
        self.h_sample_sa: List[torch.Tensor] = []
        self.h_sample_u: List[torch.Tensor] = []

        for f in range(self.F):
            n_f = len(self.shard_global_ids[f])
            f_dev = torch.device(f"cuda:{self.active_fast_devs[f]}")
            self.active_chunks.append(
                torch.zeros((n_f, horizon_steps, action_dim), dtype=torch.float32, device=f_dev)
            )
            self.active_t_obs_gpu.append(torch.zeros((n_f,), dtype=torch.int64, device=f_dev))
            self.active_slow_t_obs_gpu.append(torch.zeros((n_f,), dtype=torch.int64, device=f_dev))
            self.active_valid_gpu.append(torch.zeros((n_f,), dtype=torch.bool, device=f_dev))
            self.shard_arange.append(torch.arange(n_f, device=f_dev))
            cap_f = max(1, self.max_fast_batch_size)
            self.zero_prefixes.append(
                torch.zeros((cap_f, prefix_steps, action_dim), dtype=torch.bfloat16, device=f_dev)
            )

            self.h_commit_lids.append(torch.empty((cap_f,), dtype=torch.long, pin_memory=True))
            self.h_commit_tobs.append(torch.empty((cap_f,), dtype=torch.int64, pin_memory=True))
            self.h_commit_stobs.append(torch.empty((cap_f,), dtype=torch.int64, pin_memory=True))
            self.d_commit_lids.append(torch.empty((cap_f,), dtype=torch.long, device=f_dev))
            self.d_commit_tobs.append(torch.empty((cap_f,), dtype=torch.int64, device=f_dev))
            self.d_commit_stobs.append(torch.empty((cap_f,), dtype=torch.int64, device=f_dev))

            self.h_sample_out.append(torch.empty((n_f, action_dim), dtype=torch.float32, pin_memory=True))
            self.h_sample_fa.append(torch.empty((n_f,), dtype=torch.float32, pin_memory=True))
            self.h_sample_sa.append(torch.empty((n_f,), dtype=torch.float32, pin_memory=True))
            self.h_sample_u.append(torch.empty((1,), dtype=torch.int64, pin_memory=True))

        self.h_full_out = torch.empty((num_streams, action_dim), dtype=torch.float32, pin_memory=True)
        self.d_full_out = torch.empty((num_streams, action_dim), dtype=torch.float32, device=self.primary_dev)

        self.active_t_obs = [0 for _ in range(num_streams)]
        self.active_slow_t_obs = [0 for _ in range(num_streams)]
        self.active_valid = [False for _ in range(num_streams)]

        self.running = False
        self._threads: List[threading.Thread] = []
        self.fast_ttac_ms: List[float] = []
        self.slow_latency_ms: List[float] = []
        self.commit_hard_underruns = 0
        self.lock_wait_ms: List[float] = []

    def pause_clock(self) -> None:
        with self._clock_cv:
            if not self._clock_paused:
                self._clock_pause_start_ns = time.monotonic_ns()
                self._clock_paused = True

    def resume_clock(self) -> None:
        with self._clock_cv:
            if self._clock_paused:
                self._clock_paused_accum_ns += max(0, time.monotonic_ns() - self._clock_pause_start_ns)
                self._clock_paused = False
            self._clock_cv.notify_all()

    def now_ns(self) -> int:
        with self._clock_cv:
            raw = self._clock_pause_start_ns if self._clock_paused else time.monotonic_ns()
            return raw - self._clock_paused_accum_ns

    def _wait_if_clock_paused(self) -> None:
        with self._clock_cv:
            self._clock_cv.wait_for(lambda: not self._clock_paused or not self.running)

    def _wait_until_logical_ns(self, target_ns: int) -> None:
        while self.running:
            with self._clock_cv:
                if not self._clock_paused:
                    rem_ns = target_ns - (time.monotonic_ns() - self._clock_paused_accum_ns)
                    if rem_ns <= 0:
                        return
                else:
                    self._clock_cv.wait_for(lambda: not self._clock_paused or not self.running)
                    continue
            time.sleep(min(rem_ns * 1e-9, 0.0005))

    def start(self) -> None:
        self.running = True
        self._threads.clear()
        for r in range(self.R):
            t = threading.Thread(target=self._slow_replica_loop, args=(r,), daemon=True)
            self._threads.append(t)
            t.start()
        for f in range(self.F):
            t = threading.Thread(target=self._fast_shard_loop, args=(f,), daemon=True)
            self._threads.append(t)
            t.start()

    def stop(self) -> None:
        self.running = False
        with self._clock_cv:
            self._clock_cv.notify_all()
        for t in self._threads:
            t.join(timeout=5.0)
        self._threads.clear()

    def reset(self) -> None:
        was_running = self.running
        if was_running:
            self.stop()
        self.resume_clock()
        for r in range(self.R):
            with self.slow_q_locks[r]:
                self.pending_slow[r].clear()
        for s in range(self.num_streams):
            self.inflight_slow_row[s] = -1
        for f in range(self.F):
            with self.fast_q_locks[f]:
                self.pending_fast[f].clear()
            f_dev = torch.device(f"cuda:{self.active_fast_devs[f]}")
            with self.act_locks[f], torch.cuda.device(f_dev), torch.cuda.stream(self.sample_streams[f]):
                self.active_valid_gpu[f].zero_()
                self.active_t_obs_gpu[f].zero_()
                self.active_slow_t_obs_gpu[f].zero_()
            self.sample_streams[f].synchronize()
        for s in range(self.num_streams):
            with self.stream_locks[s]:
                self.state_t_obs[s] = 0
                self.state_seq_len[s] = self.channel.frame_seq_len
        with self.stats_lock:
            self.active_t_obs = [0 for _ in range(self.num_streams)]
            self.active_slow_t_obs = [0 for _ in range(self.num_streams)]
            self.active_valid = [False for _ in range(self.num_streams)]
            self.commit_hard_underruns = 0
        if was_running:
            self.start()

    def _copy_tensor_to_dev(
        self, src: torch.Tensor, dst_dev_id: int, dst_stream: torch.cuda.Stream
    ) -> torch.Tensor:
        dst_dev = torch.device(f"cuda:{dst_dev_id}")
        if src.is_cuda and src.device != dst_dev and not self.p2p_ok.get((dst_dev_id, src.device.index), False):
            torch.cuda.current_stream(src.device).synchronize()
            cpu_buf = src.cpu()
            with torch.cuda.device(dst_dev), torch.cuda.stream(dst_stream):
                return cpu_buf.to(dst_dev, dtype=torch.bfloat16, non_blocking=False)
        with torch.cuda.device(dst_dev), torch.cuda.stream(dst_stream):
            if src.is_cuda:
                if src.device == dst_dev:
                    dst_stream.wait_stream(torch.cuda.current_stream(src.device))
                else:
                    torch.cuda.current_stream(src.device).synchronize()
            return src.to(dst_dev, dtype=torch.bfloat16, non_blocking=True).clone()

    def set_prompt(self, prompt: object, *, stream_id: Optional[int] = None, **kwargs: object) -> None:
        if not all(isinstance(self.slow_replicas[r][0], SlowBackbone) for r in range(self.R)):
            raise RuntimeError("set_prompt() requires a SlowBackbone Slow stage")
        target_streams = [stream_id] if stream_id is not None else list(range(self.num_streams))
        for s in target_streams:
            if s < 0 or s >= self.num_streams:
                raise IndexError(f"stream_id {s} out of range [0, {self.num_streams})")
            r = self.slow_rep_of[s]
            stg0 = self.slow_replicas[r][0]
            assert isinstance(stg0, SlowBackbone)
            with stg0.stream_prompt_lock(s), self.slow_q_locks[r]:
                busy_pending = self.pending_slow[r][s][3] if s in self.pending_slow[r] else -1
                busy_inflight = self.inflight_slow_row[s]
                stg0.set_prompt(
                    prompt,  # type: ignore[arg-type]
                    stream_id=s,
                    busy_rows=(busy_pending, busy_inflight),
                    **kwargs,
                )

    def push_slow(
        self,
        obs: torch.Tensor,
        *,
        stream_id: int = 0,
        t_obs_ns: Optional[int] = None,
    ) -> int:
        if stream_id < 0 or stream_id >= self.num_streams:
            raise IndexError(f"stream_id {stream_id} out of range [0, {self.num_streams})")
        if t_obs_ns is None:
            t_obs_ns = self.now_ns()
        r = self.slow_rep_of[stream_id]
        dst_dev_id = self.active_slow_devs[r][0]
        ing_st = self.slow_ingest_streams[r]
        stg0 = self.slow_replicas[r][0]
        if isinstance(stg0, SlowBackbone):
            with stg0.stream_prompt_lock(stream_id), self.slow_q_locks[r]:
                obs_copy = self._copy_tensor_to_dev(obs, dst_dev_id, ing_st)
                ev = torch.cuda.Event()
                ev.record(ing_st)
                seq_l, prompt_row = stg0.get_stream_prompt_meta(stream_id)
                self.pending_slow[r][stream_id] = (t_obs_ns, obs_copy, ev, prompt_row, seq_l)
        else:
            with self.slow_q_locks[r]:
                obs_copy = self._copy_tensor_to_dev(obs, dst_dev_id, ing_st)
                ev = torch.cuda.Event()
                ev.record(ing_st)
                self.pending_slow[r][stream_id] = (t_obs_ns, obs_copy, ev, stream_id, self.channel.frame_seq_len)
        return t_obs_ns

    def push_fast(
        self,
        fast_obs: torch.Tensor,
        *,
        stream_id: int = 0,
        t_obs_ns: Optional[int] = None,
    ) -> int:
        if stream_id < 0 or stream_id >= self.num_streams:
            raise IndexError(f"stream_id {stream_id} out of range [0, {self.num_streams})")
        if t_obs_ns is None:
            t_obs_ns = self.now_ns()
        f = self.fast_shard_of[stream_id]
        dst_dev_id = self.active_fast_devs[f]
        ing_st = self.fast_ingest_streams[f]
        with self.fast_q_locks[f]:
            fobs_copy = self._copy_tensor_to_dev(fast_obs, dst_dev_id, ing_st)
            ev = torch.cuda.Event()
            ev.record(ing_st)
            self.pending_fast[f][stream_id] = (t_obs_ns, fobs_copy, ev)
        return t_obs_ns

    def push_fast_batch(
        self,
        fast_obs_batch: torch.Tensor,
        *,
        t_obs_ns: Optional[int] = None,
    ) -> int:
        if fast_obs_batch.size(0) != self.num_streams:
            raise ValueError(
                f"fast_obs_batch leading dimension ({fast_obs_batch.size(0)}) must equal num_streams ({self.num_streams})"
            )
        if t_obs_ns is None:
            t_obs_ns = self.now_ns()
        for f in range(self.F):
            g_ids = self.shard_global_ids[f]
            if not g_ids:
                continue
            dst_dev_id = self.active_fast_devs[f]
            ing_st = self.fast_ingest_streams[f]
            shard_src = fast_obs_batch if self.F == 1 else fast_obs_batch[g_ids]
            with self.fast_q_locks[f]:
                shard_copy = self._copy_tensor_to_dev(shard_src, dst_dev_id, ing_st)
                ev = torch.cuda.Event()
                ev.record(ing_st)
                for local_idx, s in enumerate(g_ids):
                    self.pending_fast[f][s] = (t_obs_ns, shard_copy[local_idx], ev)
        return t_obs_ns

    def sample_actions(self, t_query_ns: int) -> Tuple[torch.Tensor, List[float], List[float], int]:
        H = self.horizon_steps
        for f in range(self.F):
            f_dev = torch.device(f"cuda:{self.active_fast_devs[f]}")
            s_st = self.sample_streams[f]
            n_f = len(self.shard_global_ids[f])
            with self.act_locks[f], torch.no_grad(), torch.cuda.device(f_dev), torch.cuda.stream(s_st):
                age_ns = t_query_ns - self.active_t_obs_gpu[f]
                has_slow = self.active_slow_t_obs_gpu[f] > 0
                slow_age_ns = t_query_ns - self.active_slow_t_obs_gpu[f]
                origin_age_ns = (
                    torch.where(has_slow, slow_age_ns, age_ns) if self.anchor_to_slow_obs else age_ns
                )

                f_step = (origin_age_ns.float() / float(self.step_dt_ns)).clamp(0.0, float(H - 1))
                if self.interpolate:
                    idx0 = f_step.floor().long()
                    idx1 = (idx0 + 1).clamp(max=H - 1)
                    frac = (f_step - idx0.float()).unsqueeze(-1)
                    v0 = self.active_chunks[f][self.shard_arange[f], idx0]
                    v1 = self.active_chunks[f][self.shard_arange[f], idx1]
                    sampled = v0 + frac * (v1 - v0)
                else:
                    idx_near = (f_step + 0.5).floor().long().clamp(0, H - 1)
                    sampled = self.active_chunks[f][self.shard_arange[f], idx_near]
                out_f = torch.where(self.active_valid_gpu[f].unsqueeze(-1), sampled, torch.zeros_like(sampled))

                neg_one = torch.full((n_f,), -1.0, dtype=torch.float32, device=f_dev)
                fa_f = torch.where(
                    self.active_valid_gpu[f], age_ns.float() * 1e-6, neg_one
                )
                sa_f = torch.where(
                    self.active_valid_gpu[f] & has_slow, slow_age_ns.float() * 1e-6, neg_one
                )
                u_f = (self.active_valid_gpu[f] & (origin_age_ns > self.horizon_ns)).sum().view(1)

                self.h_sample_out[f].copy_(out_f, non_blocking=True)
                self.h_sample_fa[f].copy_(fa_f, non_blocking=True)
                self.h_sample_sa[f].copy_(sa_f, non_blocking=True)
                self.h_sample_u[f].copy_(u_f, non_blocking=True)

        fast_aoi = [0.0] * self.num_streams
        slow_aoi = [0.0] * self.num_streams
        total_underruns = 0
        for f in range(self.F):
            self.sample_streams[f].synchronize()
            g_ids = self.shard_global_ids[f]
            fa_l = self.h_sample_fa[f].tolist()
            sa_l = self.h_sample_sa[f].tolist()
            total_underruns += int(self.h_sample_u[f][0].item())
            for i, g_s in enumerate(g_ids):
                self.h_full_out[g_s].copy_(self.h_sample_out[f][i])
                fast_aoi[g_s] = fa_l[i]
                slow_aoi[g_s] = sa_l[i]

        caller_stream = torch.cuda.current_stream(self.primary_dev)
        with torch.no_grad(), torch.cuda.device(self.primary_dev), torch.cuda.stream(self.primary_out_stream):
            self.d_full_out.copy_(self.h_full_out, non_blocking=True)
            out = self.d_full_out.clone()
            self.primary_out_stream.synchronize()
            out.record_stream(caller_stream)

        return out, fast_aoi, slow_aoi, total_underruns

    def _slow_replica_loop(self, r: int) -> None:
        pipe_devs = self.active_slow_devs[r]
        pipe_mods = self.slow_replicas[r]
        pipe_streams = self.slow_stage_streams[r]

        while self.running:
            self._wait_if_clock_paused()
            if not self.running:
                break
            batch_items = []
            with self.slow_q_locks[r]:
                if self.pending_slow[r]:
                    keys = list(self.pending_slow[r].keys())[: self.slow_batch_size]
                    for k in keys:
                        tup = self.pending_slow[r].pop(k)
                        self.inflight_slow_row[k] = tup[3]
                        batch_items.append((k, *tup))
            if not batch_items:
                time.sleep(0.0005)
                continue

            s_ids = [x[0] for x in batch_items]
            t_obs_list = [x[1] for x in batch_items]
            slot_rows = [x[4] for x in batch_items]
            batch_seq_lens_cpu = [x[5] for x in batch_items]
            stage_ctxs: List[Tuple[int, EagerCaptureContext]] = []
            t_launch_logical = self.now_ns()
            gpu_exec_ns = 0

            try:
                with torch.no_grad():
                    dev0 = torch.device(f"cuda:{pipe_devs[0]}")
                    with torch.cuda.device(dev0), torch.cuda.stream(pipe_streams[0]):
                        for item in batch_items:
                            pipe_streams[0].wait_event(item[3])
                        h = torch.stack([x[2] for x in batch_items], dim=0)

                    for p in range(self.P):
                        p_dev_id = pipe_devs[p]
                        p_dev = torch.device(f"cuda:{p_dev_id}")
                        p_st = pipe_streams[p]
                        if p > 0:
                            prev_dev_id = pipe_devs[p - 1]
                            prev_st = pipe_streams[p - 1]
                            prev_st.synchronize()
                            if self.p2p_ok[(p_dev_id, prev_dev_id)]:
                                with torch.cuda.device(p_dev), torch.cuda.stream(p_st):
                                    h = h.to(p_dev, non_blocking=True)
                            else:
                                h_cpu = h.cpu()
                                with torch.cuda.device(p_dev), torch.cuda.stream(p_st):
                                    h = h_cpu.to(p_dev, non_blocking=False)

                        with torch.cuda.device(p_dev), torch.cuda.stream(p_st):
                            ev0 = torch.cuda.Event(enable_timing=True)
                            ev1 = torch.cuda.Event(enable_timing=True)
                            ev0.record(p_st)
                            slots_t = torch.tensor(slot_rows, dtype=torch.int32, device=p_dev)
                            seq_t = torch.tensor(batch_seq_lens_cpu, dtype=torch.int32, device=p_dev)
                            ctx = EagerCaptureContext(
                                self.channel, p_dev, prompt_slot_ids=slots_t, seq_lens=seq_t
                            )
                            h = pipe_mods[p](h, ctx)
                            ev1.record(p_st)
                            p_st.synchronize()
                            gpu_exec_ns += int(ev0.elapsed_time(ev1) * 1e6)
                            stage_ctxs.append((p_dev_id, ctx))
            finally:
                with self.slow_q_locks[r]:
                    for s in s_ids:
                        self.inflight_slow_row[s] = -1

            self._wait_until_logical_ns(t_launch_logical + gpu_exec_ns)
            if not self.running:
                break

            for idx, s in enumerate(s_ids):
                f = self.fast_shard_of[s]
                f_dev_id = self.active_fast_devs[f]
                f_dev = torch.device(f"cuda:{f_dev_id}")
                cp_st = self.state_copy_streams[r][f]

                if self.channel.mode == "kv":
                    staged_kv: List[Tuple[int, torch.Tensor, torch.Tensor]] = []
                    for p_dev_id, ctx in stage_ctxs:
                        for l_idx, (k_t, v_t) in ctx.kv_by_layer.items():
                            k_one = k_t[idx : idx + 1]
                            v_one = v_t[idx : idx + 1]
                            if not self.p2p_ok[(f_dev_id, p_dev_id)]:
                                k_one = k_one.cpu()
                                v_one = v_one.cpu()
                            staged_kv.append((l_idx, k_one, v_one))
                    with self.stream_locks[s], torch.cuda.device(f_dev), torch.cuda.stream(cp_st):
                        dst_kv = self.kv_packed[s]
                        for l_idx, k_one, v_one in staged_kv:
                            dst_kv[l_idx, 0].copy_(k_one, non_blocking=True)
                            dst_kv[l_idx, 1].copy_(v_one, non_blocking=True)
                        cp_st.synchronize()
                        self.state_t_obs[s] = t_obs_list[idx]
                        self.state_seq_len[s] = batch_seq_lens_cpu[idx]
                else:
                    last_dev_id, last_ctx = stage_ctxs[-1]
                    assert last_ctx.hidden is not None
                    h_one = last_ctx.hidden[idx : idx + 1]
                    if not self.p2p_ok[(f_dev_id, last_dev_id)]:
                        h_one = h_one.cpu()
                    with self.stream_locks[s], torch.cuda.device(f_dev), torch.cuda.stream(cp_st):
                        self.hidden_packed[s].copy_(h_one, non_blocking=True)
                        cp_st.synchronize()
                        self.state_t_obs[s] = t_obs_list[idx]
                        self.state_seq_len[s] = batch_seq_lens_cpu[idx]

                t_done = self.now_ns()
                with self.stats_lock:
                    self.slow_latency_ms.append((t_done - t_obs_list[idx]) * 1e-6)

    def _fast_shard_loop(self, f: int) -> None:
        f_dev_id = self.active_fast_devs[f]
        f_dev = torch.device(f"cuda:{f_dev_id}")
        f_st = self.fast_exec_streams[f]
        s_st = self.sample_streams[f]
        expert = self.fast_experts[f_dev_id]

        while self.running:
            self._wait_if_clock_paused()
            if not self.running:
                break
            batch_items = []
            with self.fast_q_locks[f]:
                if self.pending_fast[f]:
                    keys = list(self.pending_fast[f].keys())[: self.max_fast_batch_size]
                    for k in keys:
                        batch_items.append((k, *self.pending_fast[f].pop(k)))
            if not batch_items:
                time.sleep(0.0002)
                continue

            B = len(batch_items)
            s_ids = [x[0] for x in batch_items]
            t_obs_list = [x[1] for x in batch_items]

            t_lk0 = self.now_ns()
            ordered_unique = sorted(set(s_ids))
            for s in ordered_unique:
                self.stream_locks[s].acquire()
            t_lk1 = self.now_ns()
            with self.stats_lock:
                self.lock_wait_ms.append((t_lk1 - t_lk0) * 1e-6)

            t_gpu_launch = self.now_ns()
            gpu_exec_ns = 0
            try:
                with torch.no_grad(), torch.cuda.device(f_dev), torch.cuda.stream(f_st):
                    slow_ts = [self.state_t_obs[s] for s in s_ids]
                    cpu_seqs = [int(self.state_seq_len[s]) for s in s_ids]
                    ev0 = torch.cuda.Event(enable_timing=True)
                    ev1 = torch.cuda.Event(enable_timing=True)
                    for item in batch_items:
                        f_st.wait_event(item[3])
                    ev0.record(f_st)
                    fast_obs = torch.stack([x[2] for x in batch_items], dim=0)
                    p_valid = torch.zeros((B,), dtype=torch.bool, device=f_dev)
                    if self.channel.mode == "kv":
                        batch_ks = [
                            torch.cat([self.kv_packed[s][l, 0] for s in s_ids], dim=0)
                            for l in range(self.channel.layers)
                        ]
                        batch_vs = [
                            torch.cat([self.kv_packed[s][l, 1] for s in s_ids], dim=0)
                            for l in range(self.channel.layers)
                        ]
                        view = EagerDenseStateView(
                            self.channel,
                            batch_ks=batch_ks,
                            batch_vs=batch_vs,
                            prefix_valid=p_valid,
                            cpu_seq_lens=cpu_seqs,
                        )
                    else:
                        batch_h = torch.cat([self.hidden_packed[s] for s in s_ids], dim=0)
                        view = EagerDenseStateView(
                            self.channel, batch_hidden=batch_h, prefix_valid=p_valid, cpu_seq_lens=cpu_seqs
                        )

                    chunks = expert(fast_obs, view, self.zero_prefixes[f][:B])
                    chunks_f32 = chunks.float()
                    ev1.record(f_st)
                    f_st.synchronize()
                    gpu_exec_ns = int(ev0.elapsed_time(ev1) * 1e6)
            finally:
                for s in reversed(ordered_unique):
                    self.stream_locks[s].release()

            self._wait_until_logical_ns(t_gpu_launch + gpu_exec_ns)
            if not self.running:
                break

            t_done = self.now_ns()
            with self.act_locks[f], torch.no_grad(), torch.cuda.device(f_dev), torch.cuda.stream(s_st):
                for i, s in enumerate(s_ids):
                    self.h_commit_lids[f][i] = self.local_fast_id[s]
                    self.h_commit_tobs[f][i] = t_obs_list[i]
                    self.h_commit_stobs[f][i] = slow_ts[i]

                idx_gpu = self.d_commit_lids[f][:B]
                tobs_gpu = self.d_commit_tobs[f][:B]
                stobs_gpu = self.d_commit_stobs[f][:B]
                idx_gpu.copy_(self.h_commit_lids[f][:B], non_blocking=True)
                tobs_gpu.copy_(self.h_commit_tobs[f][:B], non_blocking=True)
                stobs_gpu.copy_(self.h_commit_stobs[f][:B], non_blocking=True)

                self.active_chunks[f].index_copy_(0, idx_gpu, chunks_f32)
                self.active_t_obs_gpu[f].index_copy_(0, idx_gpu, tobs_gpu)
                self.active_slow_t_obs_gpu[f].index_copy_(0, idx_gpu, stobs_gpu)
                self.active_valid_gpu[f].index_fill_(0, idx_gpu, True)
                s_st.synchronize()

            with self.stats_lock:
                for idx, s in enumerate(s_ids):
                    prev_origin = (
                        self.active_slow_t_obs[s]
                        if (self.anchor_to_slow_obs and self.active_slow_t_obs[s] > 0)
                        else self.active_t_obs[s]
                    )
                    if self.active_valid[s] and (t_done - prev_origin > self.horizon_ns):
                        self.commit_hard_underruns += 1
                    self.active_t_obs[s] = t_obs_list[idx]
                    self.active_slow_t_obs[s] = slow_ts[idx]
                    self.active_valid[s] = True
                    self.fast_ttac_ms.append((t_done - t_obs_list[idx]) * 1e-6)
