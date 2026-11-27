from __future__ import annotations

import contextlib
import copy
from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Any, Callable, ContextManager, Dict, List, Optional, Sequence, Tuple, Union
import weakref

import torch
import torch.nn as nn

from ..channel import SemanticStateView, SlowCaptureContext, TemporalChannel

VALID_FAST_PRECISIONS: Tuple[str, ...] = ("bf16", "fp8")
PROMPT_SLOTS_PER_STREAM: int = 4
_PROMPT_LOCK_GUARD = threading.Lock()
_PROMPT_STREAMS: Dict[torch.device, Tuple[torch.cuda.Stream, threading.Lock]] = {}
_STAGE_STREAM_LOCKS: "weakref.WeakKeyDictionary[nn.Module, Dict[int, threading.RLock]]" = weakref.WeakKeyDictionary()


def _get_prompt_cuda_stream(dev: torch.device) -> Optional[Tuple[torch.cuda.Stream, threading.Lock]]:
    if dev.type != "cuda":
        return None
    with _PROMPT_LOCK_GUARD:
        entry = _PROMPT_STREAMS.get(dev)
        if entry is None:
            entry = (torch.cuda.Stream(device=dev), threading.Lock())
            _PROMPT_STREAMS[dev] = entry
        return entry


def _get_stage_stream_lock(stage: nn.Module, stream_id: int) -> threading.RLock:
    with _PROMPT_LOCK_GUARD:
        per_stream = _STAGE_STREAM_LOCKS.get(stage)
        if per_stream is None:
            per_stream = {}
            _STAGE_STREAM_LOCKS[stage] = per_stream
        lk = per_stream.get(stream_id)
        if lk is None:
            lk = threading.RLock()
            per_stream[stream_id] = lk
        return lk


def validate_fast_precision(fast_precision: str) -> str:
    if fast_precision not in VALID_FAST_PRECISIONS:
        raise ValueError(
            f"fast_precision must be one of {VALID_FAST_PRECISIONS}, got {fast_precision!r}"
        )
    return fast_precision


class SlowBackbone(nn.Module):
    """System-2 backbone base class.

    Subclasses define ``channel`` and ``obs_shape`` and publish KV or hidden states in ``forward(obs, ctx)``
    via ``ctx.write_layer_kv(...)`` or ``ctx.write_hidden_state(...)``.
    """

    channel: TemporalChannel
    obs_shape: Tuple[int, ...]

    def __init__(self) -> None:
        super().__init__()
        self._stream_prompt_meta: List[Tuple[int, int]] = []

    def _init_prompt_ring(
        self,
        prompt_len: int,
        embed_dim: int,
        default_seq_len: int,
        device: torch.device,
    ) -> None:
        # 4-slot Pigeonhole ring per stream:
        # (1 active + 1 pending in mailbox + 1 in-flight on GPU + 1 guaranteed free write slot)
        self.register_buffer(
            "lang_embeds",
            torch.zeros(
                (PROMPT_SLOTS_PER_STREAM, prompt_len, embed_dim),
                dtype=torch.bfloat16,
                device=device,
            ),
        )
        self.register_buffer(
            "stream_seq_lens_gpu",
            torch.full((PROMPT_SLOTS_PER_STREAM,), int(default_seq_len), dtype=torch.int32, device=device),
        )
        self._stream_prompt_meta = [(int(default_seq_len), 0)]

    @torch.no_grad()
    def ensure_stream_capacity(self, num_streams: int) -> None:
        if num_streams < 1:
            raise ValueError(f"num_streams must be >= 1, got {num_streams}")
        cur_cap = len(self._stream_prompt_meta)
        if cur_cap == 0 or num_streams <= cur_cap:
            return
        extra = num_streams - cur_cap
        s0_seq_len, s0_row = self._stream_prompt_meta[0]
        base_emb = self.lang_embeds[s0_row : s0_row + 1].repeat(extra * PROMPT_SLOTS_PER_STREAM, 1, 1)  # type: ignore[index]
        base_seq = self.stream_seq_lens_gpu[s0_row : s0_row + 1].repeat(extra * PROMPT_SLOTS_PER_STREAM)  # type: ignore[index]
        self.register_buffer("lang_embeds", torch.cat([self.lang_embeds, base_emb], dim=0).contiguous())  # type: ignore[list-item]
        self.register_buffer("stream_seq_lens_gpu", torch.cat([self.stream_seq_lens_gpu, base_seq], dim=0).contiguous())  # type: ignore[list-item]
        for s in range(cur_cap, num_streams):
            self._stream_prompt_meta.append((s0_seq_len, PROMPT_SLOTS_PER_STREAM * s))

    def get_stream_prompt_meta(self, stream_id: int = 0) -> Tuple[int, int]:
        cap = len(self._stream_prompt_meta)
        if cap == 0:
            return self.channel.frame_seq_len, stream_id
        if stream_id < 0 or stream_id >= cap:
            raise IndexError(f"stream_id {stream_id} out of range [0, {cap})")
        return self._stream_prompt_meta[stream_id]

    def stream_prompt_lock(self, stream_id: int = 0) -> ContextManager[None]:
        if not self._stream_prompt_meta:
            return contextlib.nullcontext()
        return _get_stage_stream_lock(self, stream_id)

    def _copy_prompt_to_device(
        self,
        emb: torch.Tensor,
        valid_seq_len: int,
        row: Optional[int],
    ) -> None:
        dev = self.lang_embeds.device  # type: ignore[union-attr]
        st_entry = _get_prompt_cuda_stream(dev)
        if st_entry is not None:
            upd_st, dev_lk = st_entry
            caller_st = torch.cuda.current_stream(emb.device) if emb.is_cuda else None
            with dev_lk:
                if caller_st is not None:
                    upd_st.wait_stream(caller_st)
                with torch.cuda.device(dev), torch.cuda.stream(upd_st):
                    emb_dev = emb.to(device=dev, dtype=torch.bfloat16, non_blocking=True)
                    if row is None:
                        self.lang_embeds.copy_(emb_dev.unsqueeze(0).expand_as(self.lang_embeds))  # type: ignore[union-attr]
                        self.stream_seq_lens_gpu.fill_(int(valid_seq_len))  # type: ignore[union-attr]
                    else:
                        self.lang_embeds[row].copy_(emb_dev, non_blocking=True)  # type: ignore[index]
                        self.stream_seq_lens_gpu[row] = int(valid_seq_len)  # type: ignore[index]
                upd_st.synchronize()
        else:
            emb_dev = emb.to(device=dev, dtype=torch.bfloat16)
            if row is None:
                self.lang_embeds.copy_(emb_dev.unsqueeze(0).expand_as(self.lang_embeds))  # type: ignore[union-attr]
                self.stream_seq_lens_gpu.fill_(int(valid_seq_len))  # type: ignore[union-attr]
            else:
                self.lang_embeds[row].copy_(emb_dev)  # type: ignore[index]
                self.stream_seq_lens_gpu[row] = int(valid_seq_len)  # type: ignore[index]

    def _write_prompt_slot(
        self,
        emb: torch.Tensor,
        valid_seq_len: int,
        stream_id: Optional[int],
        busy_rows: Tuple[int, int],
    ) -> None:
        cap = len(self._stream_prompt_meta)
        if stream_id is not None:
            if stream_id < 0 or stream_id >= cap:
                raise IndexError(
                    f"stream_id {stream_id} out of range [0, {cap}); call ensure_stream_capacity() before compilation"
                )
            with _get_stage_stream_lock(self, stream_id):
                _, active_row = self._stream_prompt_meta[stream_id]
                base = PROMPT_SLOTS_PER_STREAM * stream_id
                forbidden = {active_row, int(busy_rows[0]), int(busy_rows[1])}
                target_row = -1
                for k in range(PROMPT_SLOTS_PER_STREAM):
                    cand = base + k
                    if cand not in forbidden:
                        target_row = cand
                        break
                if target_row < 0:
                    raise RuntimeError(
                        f"No free prompt slot available for stream {stream_id} (forbidden={sorted(forbidden)})"
                    )
                self._copy_prompt_to_device(emb, valid_seq_len, target_row)
                self._stream_prompt_meta[stream_id] = (int(valid_seq_len), target_row)
            return

        locks = [_get_stage_stream_lock(self, s) for s in range(cap)]
        for lk in locks:
            lk.acquire()
        try:
            self._copy_prompt_to_device(emb, valid_seq_len, None)
            self._stream_prompt_meta = [
                (int(valid_seq_len), PROMPT_SLOTS_PER_STREAM * s) for s in range(cap)
            ]
        finally:
            for lk in reversed(locks):
                lk.release()

    def _resolve_prompt_slot_ids(
        self,
        B: int,
        device: torch.device,
        ctx: Optional[SlowCaptureContext] = None,
    ) -> torch.Tensor:
        if ctx is not None and bool(ctx._probe_mode):
            return torch.zeros((B,), dtype=torch.int64, device=device)
        cap = len(self._stream_prompt_meta)
        if cap > 0 and ctx is not None and ctx.prompt_slot_ids is not None:
            max_row = cap * PROMPT_SLOTS_PER_STREAM - 1
            return ctx.prompt_slot_ids.to(device=device, dtype=torch.int64).clamp(0, max_row)
        if cap <= 1:
            row0 = self._stream_prompt_meta[0][1] if cap == 1 else 0
            return torch.full((B,), int(row0), dtype=torch.int64, device=device)
        if B > cap:
            raise ValueError(
                f"Batch size {B} exceeds configured stream capacity {cap}; call ensure_stream_capacity({B})"
            )
        rows = [self._stream_prompt_meta[s][1] for s in range(B)]
        return torch.tensor(rows, dtype=torch.int64, device=device)

    def _copy_prompt_ring_from(self, src: "SlowBackbone", device: torch.device) -> None:
        if not src._stream_prompt_meta:
            return
        self.ensure_stream_capacity(len(src._stream_prompt_meta))
        self.lang_embeds.copy_(src.lang_embeds.to(device))  # type: ignore[union-attr]
        self.stream_seq_lens_gpu.copy_(src.stream_seq_lens_gpu.to(device))  # type: ignore[union-attr]
        self._stream_prompt_meta = list(src._stream_prompt_meta)

    def forward(self, obs: torch.Tensor, ctx: SlowCaptureContext) -> object:
        raise NotImplementedError

    def set_prompt(self, prompt: Union[str, Sequence[int], torch.Tensor], **kwargs: Any) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not support dynamic task prompts")

    def set_state(self, state: torch.Tensor) -> None:
        raise NotImplementedError(f"{type(self).__name__} does not support slow-prefix robot state updates")

    def split_pipeline(self, stage_devices: Sequence[int]) -> List[nn.Module]:
        if len(stage_devices) == 1:
            return [self.to(device=f"cuda:{int(stage_devices[0])}", dtype=torch.bfloat16)]
        raise NotImplementedError(
            f"{type(self).__name__} does not implement split_pipeline() for "
            f"{len(stage_devices)}-stage pipeline parallelism"
        )


class FastHead(nn.Module):
    """System-1 action head base class.

    Subclasses define ``action_dim``, ``chunk_size``, and ``obs_shape``, and return action chunks of shape
    ``(B, chunk_size, action_dim)`` from ``forward(fast_obs, state, frozen_prefix)``.
    """

    action_dim: int
    chunk_size: int
    obs_shape: Tuple[int, ...]
    fast_obs_is_noise: bool = False
    rtc_conditioning_supported: bool = True
    default_horizon_ms: float = 100.0
    default_step_dt_ms: Optional[float] = None
    max_q_len: int = 64
    max_q_heads: int = 32

    def resolve_step_dt_ms(self) -> float:
        if self.default_step_dt_ms is not None:
            return float(self.default_step_dt_ms)
        return float(self.default_horizon_ms) / float(max(1, self.chunk_size))

    def validate_channel(self, channel: TemporalChannel) -> None:
        """Raises ValueError if ``channel`` published by a SlowBackbone is incompatible."""
        return None

    def calibrate_fp8(self, channel: TemporalChannel, device: torch.device) -> None:
        """Optional hook invoked before CUDA Graph capture when fast_precision='fp8'."""
        return None

    def forward(
        self,
        fast_obs: torch.Tensor,
        state: SemanticStateView,
        frozen_prefix: torch.Tensor,
    ) -> torch.Tensor:
        raise NotImplementedError


@dataclass
class DualSystemDeployment:
    """Device-placed Slow stages and Fast shards ready for Engine compilation."""

    channel: TemporalChannel
    slow_replicas: List[List[nn.Module]]
    fast_shards: List[FastHead]
    action_dim: int
    chunk_size: int
    slow_obs_shape: Tuple[int, ...]
    fast_obs_shape: Tuple[int, ...]
    fast_obs_is_noise: bool = False
    rtc_conditioning_supported: bool = True
    step_dt_ms: float = 2.0
    max_q_len: int = 64
    max_q_heads: int = 32


BuilderFn = Callable[..., DualSystemDeployment]


def deploy_modules(
    slow: SlowBackbone,
    fast: FastHead,
    *,
    slow_devices: Sequence[Sequence[int]],
    fast_devices: Sequence[int],
    prompt: Optional[Union[str, Sequence[int], torch.Tensor]] = None,
) -> DualSystemDeployment:
    """Validates and places an arbitrary (SlowBackbone, FastHead) pair across target GPUs."""
    if not isinstance(slow, SlowBackbone):
        raise TypeError(f"slow must be a SlowBackbone instance, got {type(slow).__name__}")
    if not isinstance(fast, FastHead):
        raise TypeError(f"fast must be a FastHead instance, got {type(fast).__name__}")

    ch = slow.channel
    fast.validate_channel(ch)

    if prompt is not None:
        slow.set_prompt(prompt)

    slow.eval()
    fast.eval()

    slow_replicas: List[List[nn.Module]] = []
    for r, pipe_devs in enumerate(slow_devices):
        src_slow = slow if r == 0 else copy.deepcopy(slow)
        stages = src_slow.split_pipeline(pipe_devs)
        for stg in stages:
            stg.eval()
        slow_replicas.append(stages)

    fast_shards: List[FastHead] = []
    for f, dev_id in enumerate(fast_devices):
        f_dev = torch.device(f"cuda:{int(dev_id)}")
        shard = fast if f == 0 else copy.deepcopy(fast)
        shard = shard.to(device=f_dev)
        with torch.no_grad():
            for p in shard.parameters():
                if p.is_floating_point() and p.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
                    p.data = p.data.to(dtype=torch.bfloat16)
        shard.eval()
        shard.calibrate_fp8(ch, f_dev)
        fast_shards.append(shard)

    return DualSystemDeployment(
        channel=ch,
        slow_replicas=slow_replicas,
        fast_shards=fast_shards,
        action_dim=int(fast.action_dim),
        chunk_size=int(fast.chunk_size),
        slow_obs_shape=tuple(slow.obs_shape),
        fast_obs_shape=tuple(fast.obs_shape),
        fast_obs_is_noise=bool(fast.fast_obs_is_noise),
        rtc_conditioning_supported=bool(fast.rtc_conditioning_supported),
        step_dt_ms=fast.resolve_step_dt_ms(),
        max_q_len=max(int(fast.chunk_size), int(fast.max_q_len)),
        max_q_heads=int(fast.max_q_heads),
    )


class ModelRegistry:
    """Architecture registry mapping checkpoint model_type / architectures to Tempo builders."""

    _builders: Dict[str, BuilderFn] = {}

    @classmethod
    def register(cls, *names: str) -> Callable[[BuilderFn], BuilderFn]:
        """Registers a builder ``fn(config, checkpoint_dir, *, slow_devices, fast_devices, fast_precision, prompt, **kwargs) -> DualSystemDeployment``."""
        if not names:
            raise ValueError("ModelRegistry.register requires at least one architecture name")

        def decorator(fn: BuilderFn) -> BuilderFn:
            for name in names:
                cls._builders[name] = fn
                cls._builders[name.lower()] = fn
            return fn

        return decorator

    @classmethod
    def list_models(cls) -> List[str]:
        seen: Dict[BuilderFn, str] = {}
        for k, fn in cls._builders.items():
            if fn not in seen:
                seen[fn] = k
        return sorted(seen.values())

    @classmethod
    def resolve(cls, config: Dict[str, Any]) -> BuilderFn:
        candidates: List[str] = []
        for key in ("type", "model_type"):
            val = config.get(key)
            if isinstance(val, str) and val:
                candidates.append(val)
        archs = config.get("architectures")
        if isinstance(archs, list):
            for a in archs:
                if isinstance(a, str) and a:
                    candidates.append(a)

        for cand in candidates:
            fn = cls._builders.get(cand) or cls._builders.get(cand.lower())
            if fn is not None:
                return fn

        raise ValueError(
            f"Unsupported dual-system architecture in config.json (candidates={candidates}). "
            f"Registered architectures: {cls.list_models()}"
        )

    @classmethod
    def build_for_engine(
        cls,
        model_id_or_path: Union[str, Path],
        *,
        slow_devices: Sequence[Sequence[int]],
        fast_devices: Sequence[int],
        fast_precision: str = "bf16",
        prompt: Optional[Union[str, Sequence[int], torch.Tensor]] = None,
        **model_kwargs: Any,
    ) -> DualSystemDeployment:
        validate_fast_precision(fast_precision)
        from .hub import resolve_checkpoint

        # Ensure built-in model implementations are imported and registered
        from . import gr00t as _gr00t  # noqa: F401
        from . import pi05 as _pi05  # noqa: F401
        from . import smolvla as _smolvla  # noqa: F401

        config, ckpt_dir = resolve_checkpoint(model_id_or_path)
        builder = cls.resolve(config)
        return builder(
            config,
            ckpt_dir,
            slow_devices=slow_devices,
            fast_devices=fast_devices,
            fast_precision=fast_precision,
            prompt=prompt,
            **model_kwargs,
        )
