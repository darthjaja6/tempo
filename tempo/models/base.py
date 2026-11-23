from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from ..channel import SemanticStateView, SlowCaptureContext, TemporalChannel

VALID_FAST_PRECISIONS: Tuple[str, ...] = ("bf16", "fp8")


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
        step_dt_ms=fast.resolve_step_dt_ms(),
        max_q_len=max(int(fast.chunk_size), int(fast.max_q_len)),
        max_q_heads=int(fast.max_q_heads),
    )


class ModelRegistry:
    """Architecture registry mapping checkpoint model_type / architectures to Tempo builders."""

    _builders: Dict[str, BuilderFn] = {}

    @classmethod
    def register(cls, *names: str) -> Callable[[BuilderFn], BuilderFn]:
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
