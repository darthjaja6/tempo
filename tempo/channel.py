import atexit
from dataclasses import dataclass, field
import math
import time
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple, Union
import weakref
import torch
import torch.nn as nn


_ACTIVE_ENGINES: "weakref.WeakSet[Engine]" = weakref.WeakSet()


def _shutdown_active_engines() -> None:
    first_exc: Optional[BaseException] = None
    for eng in list(_ACTIVE_ENGINES):
        try:
            eng.stop()
        except BaseException as exc:
            if first_exc is None:
                first_exc = exc
    if first_exc is not None:
        raise first_exc


atexit.register(_shutdown_active_engines)


def _load_ext():
    try:
        from . import _tempo_C

        return _tempo_C
    except ImportError as e:
        raise ImportError(
            "tempo C++/CUDA extension (_tempo_C) is not built. "
            "Run 'pip install -e . --no-build-isolation' or 'python build_ext.py'."
        ) from e


_CUSTOM_OPS_REGISTERED = False


def _ensure_custom_ops() -> None:
    """Registers Tempo C++/CUDA primitives with torch.library so torch.compile fuses surrounding PyTorch ops."""
    global _CUSTOM_OPS_REGISTERED
    if _CUSTOM_OPS_REGISTERED:
        return
    _load_ext()

    @torch.library.custom_op(
        "tempo::paged_cross_attn",
        mutates_args=("splitk_partial_out", "splitk_partial_lse"),
    )
    def paged_cross_attn(
        q: torch.Tensor,
        k_pool: torch.Tensor,
        v_pool: torch.Tensor,
        page_timestamps: torch.Tensor,
        splitk_partial_out: torch.Tensor,
        splitk_partial_lse: torch.Tensor,
        page_table: torch.Tensor,
        seq_lens: torch.Tensor,
        t_now_ns: torch.Tensor,
        layer_idx: int,
        relative_time_rope: bool,
    ) -> torch.Tensor:
        out = torch.empty_like(q)
        _load_ext().paged_rel_rope_cross_attn(
            q,
            k_pool,
            v_pool,
            page_timestamps,
            splitk_partial_out,
            splitk_partial_lse,
            page_table,
            seq_lens,
            t_now_ns,
            out,
            layer_idx,
            relative_time_rope,
            None,
        )
        return out

    @paged_cross_attn.register_fake
    def _paged_cross_attn_fake(
        q: torch.Tensor,
        k_pool: torch.Tensor,
        v_pool: torch.Tensor,
        page_timestamps: torch.Tensor,
        splitk_partial_out: torch.Tensor,
        splitk_partial_lse: torch.Tensor,
        page_table: torch.Tensor,
        seq_lens: torch.Tensor,
        t_now_ns: torch.Tensor,
        layer_idx: int,
        relative_time_rope: bool,
    ) -> torch.Tensor:
        return torch.empty_like(q)

    @torch.library.custom_op(
        "tempo::paged_cross_attn_with_lse",
        mutates_args=("splitk_partial_out", "splitk_partial_lse"),
    )
    def paged_cross_attn_with_lse(
        q: torch.Tensor,
        k_pool: torch.Tensor,
        v_pool: torch.Tensor,
        page_timestamps: torch.Tensor,
        splitk_partial_out: torch.Tensor,
        splitk_partial_lse: torch.Tensor,
        page_table: torch.Tensor,
        seq_lens: torch.Tensor,
        t_now_ns: torch.Tensor,
        layer_idx: int,
        relative_time_rope: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out = torch.empty_like(q)
        out_lse = torch.empty((q.size(0), q.size(1), q.size(2)), dtype=torch.float32, device=q.device)
        _load_ext().paged_rel_rope_cross_attn(
            q,
            k_pool,
            v_pool,
            page_timestamps,
            splitk_partial_out,
            splitk_partial_lse,
            page_table,
            seq_lens,
            t_now_ns,
            out,
            layer_idx,
            relative_time_rope,
            out_lse,
        )
        return out, out_lse

    @paged_cross_attn_with_lse.register_fake
    def _paged_cross_attn_with_lse_fake(
        q: torch.Tensor,
        k_pool: torch.Tensor,
        v_pool: torch.Tensor,
        page_timestamps: torch.Tensor,
        splitk_partial_out: torch.Tensor,
        splitk_partial_lse: torch.Tensor,
        page_table: torch.Tensor,
        seq_lens: torch.Tensor,
        t_now_ns: torch.Tensor,
        layer_idx: int,
        relative_time_rope: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.empty_like(q),
            torch.empty((q.size(0), q.size(1), q.size(2)), dtype=torch.float32, device=q.device),
        )

    @torch.library.custom_op("tempo::gather_paged_hidden", mutates_args=())
    def gather_paged_hidden(
        v_pool: torch.Tensor,
        page_timestamps: torch.Tensor,
        page_table: torch.Tensor,
        seq_lens: torch.Tensor,
        t_now_ns: torch.Tensor,
        max_seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = page_table.size(0)
        feat_dim = v_pool.size(3) * v_pool.size(4)
        out = torch.empty((B, max_seq_len, feat_dim), dtype=torch.bfloat16, device=v_pool.device)
        out_delta = torch.empty((B, max_seq_len), dtype=torch.float32, device=v_pool.device)
        _load_ext().gather_paged_hidden(
            v_pool, page_timestamps, page_table, seq_lens, t_now_ns, out, out_delta
        )
        return out, out_delta

    @gather_paged_hidden.register_fake
    def _gather_paged_hidden_fake(
        v_pool: torch.Tensor,
        page_timestamps: torch.Tensor,
        page_table: torch.Tensor,
        seq_lens: torch.Tensor,
        t_now_ns: torch.Tensor,
        max_seq_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = page_table.size(0)
        feat_dim = v_pool.size(3) * v_pool.size(4)
        return (
            torch.empty((B, max_seq_len, feat_dim), dtype=torch.bfloat16, device=v_pool.device),
            torch.empty((B, max_seq_len), dtype=torch.float32, device=v_pool.device),
        )

    _CUSTOM_OPS_REGISTERED = True


_CLS_COMPILE_CACHE: Dict[Tuple[type, int], Callable] = {}


def _module_device_index(mod: nn.Module) -> int:
    for t in mod.parameters():
        if t.is_cuda:
            return int(t.device.index or 0)
    for t in mod.buffers():
        if t.is_cuda:
            return int(t.device.index or 0)
    return 0


def _compile_fast_module_inplace(mod: nn.Module) -> None:
    """Applies per-device class-level regional torch.compile to repeated ModuleList blocks (or whole forward if flat)."""
    if getattr(mod, "_tempo_compiled", False):
        return
    _ensure_custom_ops()
    torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 1024)
    if hasattr(torch._dynamo.config, "accumulated_recompile_limit"):
        torch._dynamo.config.accumulated_recompile_limit = max(
            torch._dynamo.config.accumulated_recompile_limit, 4096
        )
    dev_idx = _module_device_index(mod)
    compiled_any_block = False
    for child in mod.modules():
        if isinstance(child, nn.ModuleList) and len(child) > 0:
            for sub in child:
                cls = type(sub)
                if cls is nn.ModuleDict or cls.forward is nn.Module.forward:
                    continue
                key = (cls, dev_idx)
                if key not in _CLS_COMPILE_CACHE:
                    _CLS_COMPILE_CACHE[key] = torch.compile(cls.forward, dynamic=True)
                sub.forward = _CLS_COMPILE_CACHE[key].__get__(sub, cls)  # type: ignore[method-assign]
                compiled_any_block = True
    if not compiled_any_block:
        root_cls = type(mod)
        key = (root_cls, dev_idx)
        if key not in _CLS_COMPILE_CACHE:
            _CLS_COMPILE_CACHE[key] = torch.compile(root_cls.forward, dynamic=True)
        mod.forward = _CLS_COMPILE_CACHE[key].__get__(mod, root_cls)  # type: ignore[method-assign]
    mod._tempo_compiled = True  # type: ignore[attr-defined]


def mark_sliceable(module: nn.Module) -> nn.Module:
    """Marks an nn.Module as an automatic piecewise CUDA Graph slice boundary."""
    module._ts_sliceable = True  # type: ignore[attr-defined]
    return module


def _to_bf16_obs(
    obs: torch.Tensor, name: str, device: Optional[Union[str, torch.device]] = None
) -> torch.Tensor:
    if not (obs.is_floating_point() or obs.dtype == torch.uint8):
        raise TypeError(
            f"{name} expects a floating-point (or uint8 image) tensor, got {obs.dtype}; "
            "embed integer token IDs before pushing"
        )
    target_dev = obs.device if device is None else torch.device(device)
    if obs.dtype == torch.uint8:
        return obs.to(device=target_dev, dtype=torch.bfloat16).mul_(1.0 / 255.0).contiguous()
    if obs.device != target_dev or obs.dtype != torch.bfloat16 or not obs.is_contiguous():
        return obs.to(device=target_dev, dtype=torch.bfloat16).contiguous()
    return obs


DeviceSpec = Union[int, Sequence[int], Sequence[Sequence[int]]]


def _parse_slow_topology(devices: DeviceSpec) -> Tuple[Tuple[int, ...], ...]:
    if isinstance(devices, int):
        replicas: Tuple[Tuple[int, ...], ...] = ((int(devices),),)
    else:
        seq = list(devices)
        if not seq:
            raise ValueError("slow_devices must be non-empty")
        if isinstance(seq[0], int):
            replicas = tuple((int(d),) for d in seq)
        else:
            replicas = tuple(tuple(int(d) for d in pipe) for pipe in seq)  # type: ignore[arg-type]
            p_len = len(replicas[0])
            if p_len < 1 or any(len(pipe) != p_len for pipe in replicas):
                raise ValueError(f"All slow pipeline replicas must have equal non-zero stage count: {replicas}")
    seen: Set[int] = set()
    for pipe in replicas:
        for d in pipe:
            if d in seen:
                raise ValueError(f"Duplicate GPU ID {d} in slow_devices {replicas}")
            seen.add(d)
    return replicas


def _normalize_fast_devices(devices: Union[int, Sequence[int]]) -> Tuple[int, ...]:
    if isinstance(devices, int):
        devs = (int(devices),)
    else:
        devs = tuple(int(d) for d in devices)
    if not devs:
        raise ValueError("fast stage devices must be non-empty")
    if len(set(devs)) != len(devs):
        raise ValueError(f"fast stage devices must not contain duplicates: {devs}")
    return devs


def _normalize_slow_devices(devices: DeviceSpec, num_stages: int) -> Tuple[Tuple[int, ...], ...]:
    if num_stages < 1:
        raise ValueError("num_stages must be >= 1")
    if isinstance(devices, int):
        if num_stages != 1:
            raise ValueError(f"Pipeline with {num_stages} stages requires {num_stages} device IDs, got single int")
        replicas = ((int(devices),),)
    else:
        seq = list(devices)
        if not seq:
            raise ValueError("slow stage devices must be non-empty")
        if isinstance(seq[0], int):
            flat = tuple(int(d) for d in seq)
            if num_stages == 1:
                replicas = tuple((d,) for d in flat)
            else:
                if len(flat) != num_stages:
                    raise ValueError(
                        f"1D slow devices {flat} must match pipeline stage count ({num_stages}), "
                        f"or pass 2D replica list [[d0, ...], ...]"
                    )
                replicas = (flat,)
        else:
            replicas = tuple(tuple(int(d) for d in pipe) for pipe in seq)  # type: ignore[arg-type]
            for pipe in replicas:
                if len(pipe) != num_stages:
                    raise ValueError(
                        f"Each slow replica device list {pipe} must have length == num_stages ({num_stages})"
                    )

    seen: Set[int] = set()
    for pipe in replicas:
        for d in pipe:
            if d in seen:
                raise ValueError(f"Duplicate GPU ID {d} in slow stage devices {replicas}")
            seen.add(d)
    return replicas


@dataclass
class TemporalChannel:
    """Cross-rate KV or hidden-state channel shared between Slow and Fast stages."""

    layers: int = 1
    num_kv_heads: int = 2
    head_dim: int = 128
    hidden_dim: Optional[int] = None
    seq_len: Optional[int] = None
    page_size: int = 16
    window_blocks: int = 32
    window_frames: int = 1
    spatial_rope: bool = True
    relative_time_rope: bool = True
    mode: str = "kv"  # "kv" (layer-wise co-pooled KV cache) or "hidden" (dense top-layer hidden state)

    def __post_init__(self) -> None:
        if self.mode not in ("kv", "hidden"):
            raise ValueError(f"Unsupported TemporalChannel mode '{self.mode}'; expected 'kv' or 'hidden'")
        if self.seq_len is not None:
            if self.seq_len < 1:
                raise ValueError(f"seq_len must be >= 1, got {self.seq_len}")
            if self.window_frames > 1 and self.seq_len % max(1, self.page_size) != 0:
                raise ValueError("seq_len must be a multiple of page_size when window_frames > 1")
            self.window_blocks = math.ceil(self.seq_len / max(1, self.page_size))
        if self.layers < 1 or self.page_size < 1 or self.window_blocks < 1 or self.window_frames < 1:
            raise ValueError("layers, page_size, window_blocks, and window_frames must all be >= 1")
        if self.mode == "kv":
            if self.hidden_dim is not None:
                raise ValueError("hidden_dim is only supported with TemporalChannel(mode='hidden')")
            if self.page_size != 16 or self.head_dim not in (64, 128, 256):
                raise ValueError(
                    f"TemporalChannel(mode='kv') requires page_size=16 and head_dim in (64, 128, 256), "
                    f"got page_size={self.page_size}, head_dim={self.head_dim}"
                )
            if self.num_kv_heads < 1:
                raise ValueError("num_kv_heads must be >= 1")
        else:
            if self.hidden_dim is not None:
                if self.hidden_dim < 4 or self.hidden_dim % 4 != 0:
                    raise ValueError(f"hidden_dim must be a positive multiple of 4, got {self.hidden_dim}")
                self.num_kv_heads = 1
                self.head_dim = self.hidden_dim
            else:
                feat_dim = self.num_kv_heads * self.head_dim
                if self.num_kv_heads < 1 or self.head_dim < 4 or self.head_dim % 4 != 0 or feat_dim % 4 != 0:
                    raise ValueError("hidden feature dimension (and head_dim) must be a positive multiple of 4")

    @property
    def feature_dim(self) -> int:
        return self.num_kv_heads * self.head_dim

    @property
    def frame_seq_len(self) -> int:
        return self.seq_len if self.seq_len is not None else self.page_size * self.window_blocks

    @property
    def max_seq_len(self) -> int:
        return self.frame_seq_len * self.window_frames


@dataclass
class SlowCaptureContext:
    """Context passed to Slow stages / SlowBackbone.forward() to publish KV or hidden state."""

    k_pool: Optional[torch.Tensor]
    v_pool: Optional[torch.Tensor]
    page_timestamps: Optional[torch.Tensor]
    page_table: Optional[torch.Tensor]
    start_page_idx: Optional[torch.Tensor]
    t_obs_ns: Optional[torch.Tensor]
    spatial_rope: bool
    relative_time_rope: bool
    spatial_grid_w: int
    page_size: int = 16
    window_frames: int = 1
    frame_seq_len: int = 512
    mode: str = "kv"
    slice_enabled: bool = True
    layer_offset: int = 0
    layer_count: int = 0
    num_kv_heads: int = 2
    head_dim: int = 128
    device: torch.device = field(default_factory=lambda: torch.device("cuda:0"))

    _probe_mode: bool = False
    _recorded_layers: Set[int] = field(default_factory=set)
    _recorded_hidden: bool = False
    _trace_mode: str = "none"  # "none" | "profile" | "capture"
    _exec_stream: Optional[torch.cuda.Stream] = None
    _last_ev: Optional[torch.cuda.Event] = None
    _op_durations_ms: List[float] = field(default_factory=list)
    _step_counter: int = 0
    _split_indices: Set[int] = field(default_factory=set)
    _on_split_cb: Optional[Callable[[], None]] = None

    def slice_point(self) -> None:
        """Marks a candidate micro-slice preemption boundary during compile_and_calibrate()."""
        if self._trace_mode == "none" or self._probe_mode:
            return
        idx = self._step_counter
        self._step_counter += 1
        if self._trace_mode == "profile":
            assert self._exec_stream is not None and self._last_ev is not None
            ev = torch.cuda.Event(enable_timing=True)
            ev.record(self._exec_stream)
            ev.synchronize()
            self._op_durations_ms.append(self._last_ev.elapsed_time(ev))
            self._last_ev = ev
        elif self._trace_mode == "capture":
            if idx in self._split_indices:
                assert self._on_split_cb is not None
                self._on_split_cb()

    @property
    def history_seq_len(self) -> int:
        return (self.window_frames - 1) * self.frame_seq_len

    def gather_history_kv(
        self, layer_idx: int, batch_size: int, num_kv_heads: int, head_dim: int
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Gathers sliding-window KV history pages (with relative-time RoPE) from the shared pool."""
        hist_len = self.history_seq_len
        if self.mode != "kv" or hist_len <= 0:
            return None, None
        if self._probe_mode:
            z = torch.zeros((batch_size, hist_len, num_kv_heads, head_dim), dtype=torch.bfloat16, device=self.device)
            return z, z.clone()

        _tempo_C = _load_ext()
        assert self.k_pool is not None and self.v_pool is not None
        assert self.page_timestamps is not None and self.page_table is not None
        assert self.start_page_idx is not None and self.t_obs_ns is not None
        local_layer = layer_idx - self.layer_offset
        if local_layer < 0 or local_layer >= self.layer_count:
            raise ValueError(
                f"gather_history_kv layer_idx={layer_idx} outside stage range "
                f"[{self.layer_offset}, {self.layer_offset + self.layer_count})"
            )
        dev = self.k_pool.device
        k_hist = torch.empty((batch_size, hist_len, num_kv_heads, head_dim), dtype=torch.bfloat16, device=dev)
        v_hist = torch.empty((batch_size, hist_len, num_kv_heads, head_dim), dtype=torch.bfloat16, device=dev)
        _tempo_C.gather_paged_kv_history(
            self.k_pool,
            self.v_pool,
            self.page_timestamps,
            self.page_table,
            self.start_page_idx,
            self.t_obs_ns,
            k_hist,
            v_hist,
            local_layer,
            self.relative_time_rope,
        )
        return k_hist, v_hist

    def write_layer_kv(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """Publishes bfloat16 KV tensors of shape ``[B, frame_seq_len, num_kv_heads, head_dim]`` for ``layer_idx``."""
        if self.mode != "kv":
            raise RuntimeError("write_layer_kv requires TemporalChannel(mode='kv')")
        if k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16 or k.shape != v.shape or k.ndim != 4:
            raise ValueError(
                f"write_layer_kv expects matching 4D bfloat16 tensors [B, {self.frame_seq_len}, "
                f"{self.num_kv_heads}, {self.head_dim}], got k={k.dtype} {tuple(k.shape)}, v={v.dtype} {tuple(v.shape)}"
            )
        if k.shape[1:] != (self.frame_seq_len, self.num_kv_heads, self.head_dim):
            raise ValueError(
                f"write_layer_kv shape mismatch: expected [B, {self.frame_seq_len}, {self.num_kv_heads}, "
                f"{self.head_dim}], got {tuple(k.shape)}"
            )
        if self._probe_mode:
            self._recorded_layers.add(int(layer_idx))
            return

        _tempo_C = _load_ext()
        assert self.k_pool is not None and self.v_pool is not None
        assert self.page_timestamps is not None and self.page_table is not None
        assert self.start_page_idx is not None and self.t_obs_ns is not None
        local_layer = layer_idx - self.layer_offset
        if local_layer < 0 or local_layer >= self.layer_count:
            raise ValueError(
                f"write_layer_kv layer_idx={layer_idx} outside stage range "
                f"[{self.layer_offset}, {self.layer_offset + self.layer_count})"
            )
        _tempo_C.write_paged_kv(
            k.contiguous(),
            v.contiguous(),
            self.k_pool,
            self.v_pool,
            self.page_timestamps,
            self.page_table,
            self.start_page_idx,
            self.t_obs_ns,
            local_layer,
            self.spatial_rope,
            self.spatial_grid_w,
        )

    def write_hidden_state(self, hidden: torch.Tensor) -> None:
        """Publishes a bfloat16 hidden-state tensor of shape ``[B, frame_seq_len, hidden_dim]``."""
        if self.mode != "hidden":
            raise RuntimeError("write_hidden_state requires TemporalChannel(mode='hidden')")
        feat_dim = self.num_kv_heads * self.head_dim
        if hidden.dtype != torch.bfloat16 or hidden.ndim != 3 or hidden.shape[1:] != (self.frame_seq_len, feat_dim):
            raise ValueError(
                f"write_hidden_state expects bfloat16 tensor [B, {self.frame_seq_len}, {feat_dim}], "
                f"got {hidden.dtype} {tuple(hidden.shape)}"
            )
        if self._probe_mode:
            self._recorded_hidden = True
            return

        _tempo_C = _load_ext()
        assert self.v_pool is not None and self.page_timestamps is not None
        assert self.page_table is not None and self.start_page_idx is not None and self.t_obs_ns is not None
        _tempo_C.write_paged_hidden(
            hidden.contiguous(),
            self.v_pool,
            self.page_timestamps,
            self.page_table,
            self.start_page_idx,
            self.t_obs_ns,
        )


@dataclass
class SemanticStateView:
    """Read-only view of shared Slow KV or hidden state passed to FastHead.forward() / fast stages."""

    k_pool: torch.Tensor
    v_pool: torch.Tensor
    page_timestamps: torch.Tensor
    splitk_partial_out: torch.Tensor
    splitk_partial_lse: torch.Tensor
    page_table: torch.Tensor
    seq_lens: torch.Tensor
    t_now_ns: torch.Tensor
    relative_time_rope: bool
    max_seq_len: int
    mode: str = "kv"
    prefix_valid: Optional[torch.Tensor] = None

    def splice_prefix(self, current: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Selects ``target`` on streams with an active RTC prefix (``prefix_valid``), else keeps ``current``."""
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
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Cross-attends sequence-first bfloat16 query ``[B, Q_len, num_q_heads, head_dim]`` over Slow KV ``layer_idx``."""
        _tempo_C = _load_ext()
        if self.mode != "kv":
            raise RuntimeError("cross_attend requires TemporalChannel(mode='kv')")
        B = self.page_table.size(0)
        num_layers, _p, num_kv_heads, head_dim = self.k_pool.shape[1:]
        max_q_len = self.splitk_partial_out.size(1)
        max_q_heads = self.splitk_partial_out.size(2)
        if q.dtype != torch.bfloat16 or q.ndim != 4 or q.size(0) != B or q.size(3) != head_dim:
            raise ValueError(
                f"cross_attend expects bfloat16 query [B={B}, Q_len, num_q_heads, {head_dim}], "
                f"got {q.dtype} {tuple(q.shape)}"
            )
        if q.size(1) > max_q_len or q.size(2) > max_q_heads or q.size(2) % num_kv_heads != 0:
            raise ValueError(
                f"cross_attend query (Q_len={q.size(1)}, num_q_heads={q.size(2)}) must satisfy "
                f"Q_len <= max_q_len ({max_q_len}), num_q_heads <= max_q_heads ({max_q_heads}), "
                f"and num_q_heads % num_kv_heads ({num_kv_heads}) == 0"
            )
        if layer_idx < 0 or layer_idx >= num_layers:
            raise ValueError(f"cross_attend layer_idx={layer_idx} out of range [0, {num_layers})")
        if not q.is_contiguous():
            q = q.contiguous()
        _ensure_custom_ops()
        if return_lse:
            res_out, res_lse = torch.ops.tempo.paged_cross_attn_with_lse(
                q,
                self.k_pool,
                self.v_pool,
                self.page_timestamps,
                self.splitk_partial_out,
                self.splitk_partial_lse,
                self.page_table,
                self.seq_lens,
                self.t_now_ns,
                layer_idx,
                self.relative_time_rope,
            )
            if out is not None:
                out.copy_(res_out)
                res_out = out
            if out_lse is not None:
                out_lse.copy_(res_lse)
                res_lse = out_lse
            return res_out, res_lse

        res_out = torch.ops.tempo.paged_cross_attn(
            q,
            self.k_pool,
            self.v_pool,
            self.page_timestamps,
            self.splitk_partial_out,
            self.splitk_partial_lse,
            self.page_table,
            self.seq_lens,
            self.t_now_ns,
            layer_idx,
            self.relative_time_rope,
        )
        if out is not None:
            out.copy_(res_out)
            return out
        return res_out

    def read_hidden(
        self,
        out: Optional[torch.Tensor] = None,
        out_delta_sec: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(hidden, delta_t_sec)`` with ``hidden`` ``[B, max_seq_len, hidden_dim]`` bf16 and age ``[B, max_seq_len]`` fp32."""
        _ensure_custom_ops()
        if self.mode != "hidden":
            raise RuntimeError("read_hidden requires TemporalChannel(mode='hidden')")
        res_out, res_delta = torch.ops.tempo.gather_paged_hidden(
            self.v_pool,
            self.page_timestamps,
            self.page_table,
            self.seq_lens,
            self.t_now_ns,
            self.max_seq_len,
        )
        if out is not None:
            out.copy_(res_out)
            res_out = out
        if out_delta_sec is not None:
            out_delta_sec.copy_(res_delta)
            res_delta = out_delta_sec
        return res_out, res_delta


@dataclass
class StageSpec:
    stages: Tuple[Callable, ...]
    rate_hz: float
    slow_devices: Tuple[Tuple[int, ...], ...] = ((0,),)
    fast_devices: Tuple[int, ...] = (0,)
    publishes: Optional[TemporalChannel] = None
    reads: Optional[TemporalChannel] = None
    max_slice_ms: float = 2.0
    deadline_ms: float = 25.0
    horizon_ms: float = 96.0
    batch_size: int = 4
    buckets: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)
    fast_obs_is_noise: bool = False


class Engine:
    """Fast serving engine for System 1 + System 2 robot policies."""

    TemporalChannel = TemporalChannel
    scheduler = None
    _running: bool = False

    def __init__(
        self,
        model: Optional[str] = None,
        *,
        slow: Optional[nn.Module] = None,
        fast: Optional[nn.Module] = None,
        num_streams: int = 1,
        slow_devices: DeviceSpec = (0,),
        fast_devices: Optional[Union[int, Sequence[int]]] = None,
        slow_rate_hz: float = 2.0,
        fast_rate_hz: float = 20.0,
        deadline_ms: Optional[float] = None,
        horizon_ms: Optional[float] = None,
        max_slice_ms: float = 2.0,
        slow_batch_size: int = 4,
        buckets: Sequence[int] = (1, 2, 4, 8, 16, 32, 64),
        fast_precision: Optional[str] = None,
        prompt: Optional[Union[str, Sequence[int], torch.Tensor]] = None,
        compile_fast: bool = True,
        auto_start: bool = True,
        model_kwargs: Optional[Dict[str, object]] = None,
        action_dim: Optional[int] = None,
        slow_obs_shape: Sequence[int] = (),
        fast_obs_shape: Sequence[int] = (),
        step_dt_ms: Optional[float] = None,
        max_slow_aoi_ms: Optional[float] = None,
        max_q_len: int = 64,
        max_q_heads: int = 32,
    ) -> None:
        """Constructs a Tempo serving engine from ``model`` (HF repo ID or path), ``(slow, fast)`` modules, or ``@engine.stage``.

        - ``slow_devices``: ``0`` or ``[0, 1]`` for Slow data-parallel replicas, or ``[[0, 1]]`` for Slow pipeline stages.
        - ``fast_devices``: ``0`` or ``[1, 2, 3]`` for Fast shards (defaults to ``slow_devices`` when single-stage).
        - Timing contract: ``deadline_ms <= 1000 / fast_rate_hz`` and ``(horizon_steps - 1) * step_dt_ms > 1000 / fast_rate_hz + deadline_ms``.
        - ``max_slow_aoi_ms``: Maximum Slow observation staleness (ms) before Slow takes scheduling priority.
        """
        if num_streams < 1:
            raise ValueError(f"num_streams must be >= 1, got {num_streams}")
        if model is not None and (slow is not None or fast is not None):
            raise ValueError("Pass either 'model=...' or '(slow=..., fast=...)', not both")
        if (slow is None) != (fast is None):
            raise ValueError("Modular composition requires passing both slow=... and fast=...")
        if model is None and fast_precision is not None:
            raise ValueError(
                "fast_precision can only be passed when loading a model via model=...; "
                "configure precision directly on your FastHead instance when using (slow=..., fast=...) or @engine.stage"
            )

        self.num_streams = int(num_streams)
        self.max_slow_aoi_ms = max_slow_aoi_ms
        self.compile_fast = bool(compile_fast)
        self._slow_spec: Optional[StageSpec] = None
        self._fast_spec: Optional[StageSpec] = None
        self._running: bool = False
        self.deployment = None
        self.calibration_info: Optional[Dict[str, object]] = None

        self.fast_tscs: List[object] = []
        self.fast_rtcs: List[object] = []
        self.slow_tscs: List[List[Optional[object]]] = []
        self.scheduler = None
        self.disaggregated: bool = False
        self.active_slow_devices: List[List[int]] = []
        self.active_fast_devices: List[int] = []
        self.all_devices: List[int] = []
        self.slice_durations_ms: List[float] = []
        self.slow_bucket_slices_ms: Dict[int, List[float]] = {}
        self.slow_stage_bucket_ms: List[Dict[int, float]] = []
        self.bucket_durations_ms: Dict[int, float] = {}
        self._retained_tensors: List[object] = []

        if model is not None or slow is not None:
            from .models.base import FastHead, ModelRegistry, SlowBackbone, deploy_modules

            slow_topo = _parse_slow_topology(slow_devices)
            if fast_devices is None:
                if len(slow_topo[0]) > 1:
                    raise ValueError(
                        "fast_devices must be specified explicitly when slow_devices uses multi-stage pipeline parallelism"
                    )
                f_topo = tuple(pipe[0] for pipe in slow_topo)
            else:
                f_topo = _normalize_fast_devices(fast_devices)

            act_slow = list(slow_topo[: min(self.num_streams, len(slow_topo))])
            act_fast = list(f_topo[: min(self.num_streams, len(f_topo))])

            if model is not None:
                eff_precision = "bf16" if fast_precision is None else fast_precision
                dep = ModelRegistry.build_for_engine(
                    model,
                    slow_devices=act_slow,
                    fast_devices=act_fast,
                    fast_precision=eff_precision,
                    prompt=prompt,
                    **(model_kwargs or {}),
                )
            else:
                assert isinstance(slow, SlowBackbone) and isinstance(fast, FastHead)
                dep = deploy_modules(
                    slow,
                    fast,
                    slow_devices=act_slow,
                    fast_devices=act_fast,
                    prompt=prompt,
                )

            self.deployment = dep
            self.action_dim = int(action_dim) if action_dim is not None else dep.action_dim
            self.slow_obs_shape = tuple(slow_obs_shape) if slow_obs_shape else dep.slow_obs_shape
            self.fast_obs_shape = tuple(fast_obs_shape) if fast_obs_shape else dep.fast_obs_shape
            eff_dt = float(step_dt_ms) if step_dt_ms is not None else dep.step_dt_ms
            self.step_dt_ms = eff_dt
            self.step_dt_ns = int(round(eff_dt * 1e6))
            self.max_q_len = max(max_q_len, dep.max_q_len)
            self.max_q_heads = max(max_q_heads, dep.max_q_heads)

            P = len(act_slow[0])
            slow_fns = []
            for p_idx in range(P):
                stage_map = {act_slow[r][p_idx]: dep.slow_replicas[r][p_idx] for r in range(len(act_slow))}

                def _make_stage_fn(m_map: Dict[int, nn.Module]) -> Callable:
                    def _fn(inp: torch.Tensor, ctx: SlowCaptureContext) -> object:
                        return m_map[ctx.device.index](inp, ctx)

                    return _fn

                slow_fns.append(_make_stage_fn(stage_map))

            if P == 1:
                self.stage(
                    rate_hz=slow_rate_hz,
                    devices=[pipe[0] for pipe in act_slow],
                    publishes=dep.channel,
                    max_slice_ms=max_slice_ms,
                    batch_size=slow_batch_size,
                )(slow_fns[0])
            else:
                self.pipeline(
                    slow_fns,
                    rate_hz=slow_rate_hz,
                    publishes=dep.channel,
                    devices=act_slow,
                    max_slice_ms=max_slice_ms,
                    batch_size=slow_batch_size,
                )

            fast_map = {act_fast[f_idx]: dep.fast_shards[f_idx] for f_idx in range(len(act_fast))}

            def _fast_fn(
                fast_obs: torch.Tensor, state: SemanticStateView, frozen_prefix: torch.Tensor
            ) -> torch.Tensor:
                return fast_map[fast_obs.device.index](fast_obs, state, frozen_prefix)

            eff_horizon = horizon_ms if horizon_ms is not None else dep.chunk_size * self.step_dt_ms
            self.stage(
                rate_hz=fast_rate_hz,
                devices=act_fast,
                reads=dep.channel,
                deadline_ms=deadline_ms,
                horizon_ms=eff_horizon,
                buckets=buckets,
                fast_obs_is_noise=dep.fast_obs_is_noise,
            )(_fast_fn)

            self.calibration_info = self.compile_and_calibrate()
            self.reset()
            if auto_start:
                self.start()
        else:
            if action_dim is None or action_dim < 1:
                raise ValueError("action_dim must be specified when constructing a custom @engine.stage Engine")
            eff_dt = float(step_dt_ms) if step_dt_ms is not None else 2.0
            self.action_dim = int(action_dim)
            self.slow_obs_shape = tuple(slow_obs_shape)
            self.fast_obs_shape = tuple(fast_obs_shape)
            self.step_dt_ms = eff_dt
            self.step_dt_ns = int(round(eff_dt * 1e6))
            self.max_q_len = max_q_len
            self.max_q_heads = max_q_heads

    @property
    def horizon_steps(self) -> int:
        if self._fast_spec is None:
            raise RuntimeError("Register the Fast stage before querying engine.horizon_steps")
        return max(2, int(round(self._fast_spec.horizon_ms / self.step_dt_ms)))

    @property
    def prefix_steps(self) -> int:
        if self._fast_spec is None:
            raise RuntimeError("Register the Fast stage before querying engine.prefix_steps")
        return min(
            self.horizon_steps - 1, max(1, int(math.ceil(self._fast_spec.deadline_ms / self.step_dt_ms)) + 1)
        )

    def stage(
        self,
        *,
        rate_hz: float,
        devices: DeviceSpec = (0,),
        publishes: Optional[TemporalChannel] = None,
        reads: Optional[TemporalChannel] = None,
        max_slice_ms: float = 2.0,
        deadline_ms: Optional[float] = None,
        horizon_ms: float = 96.0,
        batch_size: int = 4,
        buckets: Sequence[int] = (1, 2, 4, 8, 16, 32, 64),
        fast_obs_is_noise: bool = False,
    ) -> Callable:
        """Decorator registering a custom Slow stage (``publishes=channel``) or Fast stage (``reads=channel``)."""
        if rate_hz <= 0.0:
            raise ValueError("rate_hz must be positive")
        if (publishes is None) == (reads is None):
            raise ValueError("Stage must either publish to (Slow) or read from (Fast) a TemporalChannel")
        if publishes is not None and fast_obs_is_noise:
            raise ValueError("fast_obs_is_noise only applies to Fast stages (reads=...)")

        def decorator(fn: Callable) -> Callable:
            if publishes is not None:
                s_devs = _normalize_slow_devices(devices, num_stages=1)
                self._slow_spec = StageSpec(
                    stages=(fn,),
                    rate_hz=rate_hz,
                    slow_devices=s_devs,
                    publishes=publishes,
                    max_slice_ms=max_slice_ms,
                    batch_size=batch_size,
                )
            else:
                if isinstance(devices, Sequence) and devices and not isinstance(devices[0], int):
                    raise ValueError("Fast stage devices must be an int or 1D sequence of GPU IDs")
                f_devs = _normalize_fast_devices(devices)  # type: ignore[arg-type]
                eff_deadline = deadline_ms if deadline_ms is not None else min(25.0, 1000.0 / rate_hz)
                self._fast_spec = StageSpec(
                    stages=(fn,),
                    rate_hz=rate_hz,
                    fast_devices=f_devs,
                    reads=reads,
                    deadline_ms=eff_deadline,
                    horizon_ms=horizon_ms,
                    buckets=tuple(sorted(set(buckets))),
                    fast_obs_is_noise=bool(fast_obs_is_noise),
                )
            return fn

        return decorator

    def pipeline(
        self,
        stages: Sequence[Callable],
        *,
        rate_hz: float,
        publishes: TemporalChannel,
        devices: DeviceSpec,
        max_slice_ms: float = 2.0,
        batch_size: int = 4,
    ) -> Sequence[Callable]:
        """Registers a P-stage Pipeline-Parallel Slow model across P GPUs (optionally replicated)."""
        if rate_hz <= 0.0:
            raise ValueError("rate_hz must be positive")
        stage_tuple = tuple(stages)
        if not stage_tuple:
            raise ValueError("pipeline stages must be non-empty")
        s_devs = _normalize_slow_devices(devices, num_stages=len(stage_tuple))
        self._slow_spec = StageSpec(
            stages=stage_tuple,
            rate_hz=rate_hz,
            slow_devices=s_devs,
            publishes=publishes,
            max_slice_ms=max_slice_ms,
            batch_size=batch_size,
        )
        return stages

    def compile_and_calibrate(self) -> Dict[str, object]:
        self.stop()
        _tempo_C = _load_ext()

        if self._fast_spec is None:
            raise RuntimeError("Fast stage (reads=...) must be registered before compile_and_calibrate()")
        if not self.fast_obs_shape:
            raise ValueError("fast_obs_shape must be non-empty")

        remote_slow = self._slow_spec is None
        if not remote_slow:
            assert self._slow_spec is not None
            if not self.slow_obs_shape:
                raise ValueError("slow_obs_shape must be non-empty when a local Slow stage is registered")
            ch = self._slow_spec.publishes
            if ch is None or ch != self._fast_spec.reads:
                raise ValueError("Slow and Fast stages must share a matching TemporalChannel")
        else:
            ch = self._fast_spec.reads
            assert ch is not None
            if ch.mode != "hidden":
                raise ValueError(
                    "Fast-only Engine without a local Slow stage requires TemporalChannel(mode='hidden')"
                )

        fast_period_ms = 1000.0 / self._fast_spec.rate_hz
        if self._fast_spec.deadline_ms > fast_period_ms:
            raise ValueError(
                f"Single-slot staged RTC contract requires deadline_ms ({self._fast_spec.deadline_ms} ms) "
                f"<= T_fast ({fast_period_ms:.1f} ms)"
            )
        horizon_steps = max(2, int(round(self._fast_spec.horizon_ms / self.step_dt_ms)))
        effective_horizon_ms = (horizon_steps - 1) * self.step_dt_ms
        if effective_horizon_ms <= fast_period_ms + self._fast_spec.deadline_ms:
            min_dt = (fast_period_ms + self._fast_spec.deadline_ms + 1.0) / max(1, horizon_steps - 1)
            raise ValueError(
                f"RTC lookahead contract violated: effective horizon ({effective_horizon_ms:.1f} ms across "
                f"{horizon_steps} steps at step_dt_ms={self.step_dt_ms:.2f}) must exceed "
                f"T_fast ({fast_period_ms:.1f} ms) + deadline_ms ({self._fast_spec.deadline_ms} ms). "
                f"Pass step_dt_ms>={min_dt:.2f} (or adjust fast_rate_hz / deadline_ms)."
            )

        raw_fast_devs = list(self._fast_spec.fast_devices)
        if remote_slow:
            P = 0
            active_slow_devs: List[List[int]] = []
            active_fast_devs = list(raw_fast_devs[: min(self.num_streams, len(raw_fast_devs))])
            effective_max_slow_aoi_ms = self.max_slow_aoi_ms if self.max_slow_aoi_ms is not None else 1000.0
            max_slice_ms = 2.0
        else:
            assert self._slow_spec is not None
            slow_period_ms = 1000.0 / self._slow_spec.rate_hz
            effective_max_slow_aoi_ms = (
                self.max_slow_aoi_ms if self.max_slow_aoi_ms is not None else 1.35 * slow_period_ms
            )
            max_slice_ms = self._slow_spec.max_slice_ms
            P = len(self._slow_spec.stages)
            raw_slow_devs = list(self._slow_spec.slow_devices)

            exact_colocated = (
                P == 1
                and len(raw_slow_devs) == len(raw_fast_devs)
                and all(raw_slow_devs[i][0] == raw_fast_devs[i] for i in range(len(raw_fast_devs)))
            )
            if exact_colocated:
                k_active = min(self.num_streams, len(raw_fast_devs))
                active_slow_devs = [list(raw_slow_devs[i]) for i in range(k_active)]
                active_fast_devs = list(raw_fast_devs[:k_active])
            else:
                r_active = min(self.num_streams, len(raw_slow_devs))
                f_active = min(self.num_streams, len(raw_fast_devs))
                active_slow_devs = [list(raw_slow_devs[i]) for i in range(r_active)]
                active_fast_devs = list(raw_fast_devs[:f_active])

        self.active_slow_devices = active_slow_devs
        self.active_fast_devices = active_fast_devs
        all_dev_set: List[int] = []
        for pipe in active_slow_devs:
            for d in pipe:
                if d not in all_dev_set:
                    all_dev_set.append(d)
        for d in active_fast_devs:
            if d not in all_dev_set:
                all_dev_set.append(d)
        self.all_devices = all_dev_set

        R = len(active_slow_devs)
        F = len(active_fast_devs)
        max_streams_per_fast = math.ceil(self.num_streams / F)

        if remote_slow:
            b_slow = 1
            slow_buckets: Tuple[int, ...] = ()
        else:
            assert self._slow_spec is not None
            max_streams_per_slow = math.ceil(self.num_streams / R)
            b_slow = min(self._slow_spec.batch_size, max_streams_per_slow)
            slow_buckets = tuple([b for b in (1, 2, 4, 8, 16) if b < b_slow] + [b_slow])

        smaller_buckets = [b for b in self._fast_spec.buckets if b < max_streams_per_fast]
        ceil_buckets = [b for b in self._fast_spec.buckets if b >= max_streams_per_fast]
        valid_global_buckets = tuple(smaller_buckets + (ceil_buckets[:1] if ceil_buckets else []))
        if not valid_global_buckets:
            raise ValueError("fast_stage buckets must contain at least one positive bucket size")
        max_b_fast = max(valid_global_buckets)

        prefix_steps = self.prefix_steps
        max_pages_per_slot = math.ceil(ch.max_seq_len / ch.page_size)
        frame_pages = math.ceil(ch.frame_seq_len / ch.page_size)
        grid_w = max(1, int(math.isqrt(ch.frame_seq_len)))

        self.scheduler = _tempo_C.TimeAwareScheduler(
            num_streams=self.num_streams,
            slow_devices=active_slow_devs,
            fast_devices=active_fast_devs,
            slow_batch_size=b_slow,
            max_fast_batch_size=max_b_fast,
            slow_frame_seq_len=ch.frame_seq_len,
            fast_deadline_ms=self._fast_spec.deadline_ms,
            max_slice_ms=max_slice_ms,
            max_slow_aoi_ms=effective_max_slow_aoi_ms,
        )
        self.disaggregated = bool(self.scheduler.is_disaggregated())
        self._retained_tensors.clear()
        self.fast_tscs.clear()
        self.fast_rtcs.clear()
        self.slow_tscs = [[None for _ in range(P)] for _ in range(R)]

        # 1. Probe Slow stage(s) to verify TSC writes, layer ownership, and inter-stage activation specs
        stage_layer_offsets: List[int] = [0] * P
        stage_layer_counts: List[int] = [0] * P
        stage_writes_tsc: List[bool] = [False] * P
        edge_act_specs: List[List[Tuple[Tuple[int, ...], torch.dtype]]] = []
        edge_unpack_single: List[bool] = []

        if P >= 1:
            probe_obs = torch.zeros(
                (b_slow, *self.slow_obs_shape), dtype=torch.bfloat16, device=f"cuda:{active_slow_devs[0][0]}"
            )
            cur_inp: object = probe_obs
            covered_layers: Set[int] = set()
            hidden_writers = 0

            with torch.no_grad():
                for p in range(P):
                    p_dev = torch.device(f"cuda:{active_slow_devs[0][p]}")
                    with torch.cuda.device(p_dev):
                        probe_ctx = SlowCaptureContext(
                            k_pool=None,
                            v_pool=None,
                            page_timestamps=None,
                            page_table=None,
                            start_page_idx=None,
                            t_obs_ns=None,
                            spatial_rope=ch.spatial_rope,
                            relative_time_rope=ch.relative_time_rope,
                            spatial_grid_w=grid_w,
                            page_size=ch.page_size,
                            window_frames=ch.window_frames,
                            frame_seq_len=ch.frame_seq_len,
                            mode=ch.mode,
                            slice_enabled=False,
                            num_kv_heads=ch.num_kv_heads,
                            head_dim=ch.head_dim,
                            device=p_dev,
                            _probe_mode=True,
                        )
                        out_val = self._slow_spec.stages[p](cur_inp, probe_ctx)

                        if ch.mode == "kv":
                            if probe_ctx._recorded_layers:
                                lo = min(probe_ctx._recorded_layers)
                                hi = max(probe_ctx._recorded_layers)
                                expected = set(range(lo, hi + 1))
                                if probe_ctx._recorded_layers != expected:
                                    raise ValueError(
                                        f"Slow stage {p} must write a contiguous layer range, "
                                        f"got {sorted(probe_ctx._recorded_layers)}"
                                    )
                                if covered_layers & probe_ctx._recorded_layers:
                                    raise ValueError(f"Duplicate KV layer writes across Slow stages at stage {p}")
                                covered_layers |= probe_ctx._recorded_layers
                                stage_writes_tsc[p] = True
                                stage_layer_offsets[p] = lo
                                stage_layer_counts[p] = hi - lo + 1
                        else:
                            if probe_ctx._recorded_hidden:
                                hidden_writers += 1
                                stage_writes_tsc[p] = True
                                stage_layer_offsets[p] = 0
                                stage_layer_counts[p] = 1

                        if p + 1 < P:
                            next_dev = torch.device(f"cuda:{active_slow_devs[0][p + 1]}")
                            if isinstance(out_val, torch.Tensor):
                                tensors = [out_val]
                                edge_unpack_single.append(True)
                            elif isinstance(out_val, (tuple, list)) and len(out_val) > 0:
                                tensors = list(out_val)
                                edge_unpack_single.append(False)
                            else:
                                raise ValueError(
                                    f"Pipeline stage {p} must return a Tensor or non-empty tuple of Tensors "
                                    f"for stage {p + 1}"
                                )
                            specs: List[Tuple[Tuple[int, ...], torch.dtype]] = []
                            next_tensors: List[torch.Tensor] = []
                            for t in tensors:
                                if not isinstance(t, torch.Tensor) or not t.is_cuda or t.size(0) != b_slow:
                                    raise ValueError(
                                        f"Pipeline stage {p} output tensor must be a CUDA tensor with leading "
                                        f"batch dim {b_slow}"
                                    )
                                specs.append((tuple(t.shape[1:]), t.dtype))
                                next_tensors.append(t.detach().contiguous().to(next_dev))
                            edge_act_specs.append(specs)
                            cur_inp = next_tensors[0] if edge_unpack_single[-1] else tuple(next_tensors)

            if ch.mode == "kv" and covered_layers != set(range(ch.layers)):
                raise ValueError(
                    f"Slow stage(s) wrote KV layers {sorted(covered_layers)}, expected 0..{ch.layers - 1} "
                    f"(call ctx.write_layer_kv for every layer)"
                )
            if ch.mode == "hidden" and hidden_writers != 1:
                raise ValueError(
                    f"Hidden-mode Slow stage(s) must call ctx.write_hidden_state(...) exactly once, "
                    f"got {hidden_writers} calls"
                )

        # 2. Allocate Fast shards (TemporalStateCache + RTCHorizonRing + Static Buffers)
        total_tsc_pages = 0
        shard_valid_buckets: List[Tuple[int, ...]] = []
        fast_shard_statics: List[Tuple[torch.Tensor, ...]] = []
        for f in range(F):
            f_dev_id = active_fast_devs[f]
            f_dev = torch.device(f"cuda:{f_dev_id}")
            n_f = self.scheduler.fast_shard_num_streams(f)

            s_smaller = [b for b in self._fast_spec.buckets if b < n_f]
            s_ceil = [b for b in self._fast_spec.buckets if b >= n_f]
            f_buckets = tuple(s_smaller + (s_ceil[:1] if s_ceil else []))
            shard_valid_buckets.append(f_buckets)
            max_b_f = max(f_buckets)

            if remote_slow:
                inflight_writer_frames = frame_pages
            elif not self.disaggregated:
                inflight_writer_frames = b_slow * frame_pages
            else:
                inflight_writer_frames = min(n_f, R * P * b_slow) * frame_pages
            f_max_pages = (
                n_f * max_pages_per_slot
                + inflight_writer_frames
                + min(n_f, max_b_f) * max_pages_per_slot
            )
            total_tsc_pages += f_max_pages

            tsc_f = _tempo_C.TemporalStateCache(
                num_streams=n_f,
                num_readers=1,
                max_pages=f_max_pages,
                num_layers=ch.layers,
                page_size=ch.page_size,
                num_kv_heads=ch.num_kv_heads,
                head_dim=ch.head_dim,
                max_seq_len=ch.max_seq_len,
                kv_mode=(ch.mode == "kv"),
                max_q_len=max(horizon_steps, self.max_q_len),
                max_q_heads=self.max_q_heads,
                device_id=f_dev_id,
            )
            rtc_f = _tempo_C.RTCHorizonRing(
                num_streams=n_f,
                horizon_steps=horizon_steps,
                prefix_steps=prefix_steps,
                action_dim=self.action_dim,
                step_dt_ns=self.step_dt_ns,
                device_id=f_dev_id,
            )
            self.fast_tscs.append(tsc_f)
            self.fast_rtcs.append(rtc_f)

            f_obs_st = torch.zeros((max_b_f, *self.fast_obs_shape), dtype=torch.bfloat16, device=f_dev)
            f_pref_st = torch.zeros((max_b_f, prefix_steps, self.action_dim), dtype=torch.bfloat16, device=f_dev)
            f_pref_valid_st = torch.zeros((max_b_f,), dtype=torch.bool, device=f_dev)
            f_pt_st = torch.zeros((max_b_f, max_pages_per_slot), dtype=torch.int32, device=f_dev)
            f_seq_st = torch.full((max_b_f,), ch.max_seq_len, dtype=torch.int32, device=f_dev)
            f_tnow_st = torch.zeros((max_b_f,), dtype=torch.int64, device=f_dev)
            f_out_st = torch.zeros((max_b_f, horizon_steps, self.action_dim), dtype=torch.bfloat16, device=f_dev)

            statics = (f_obs_st, f_pref_st, f_pref_valid_st, f_pt_st, f_seq_st, f_tnow_st, f_out_st)
            fast_shard_statics.append(statics)
            self._retained_tensors.extend(statics)
            self.scheduler.init_fast_shard(
                f,
                tsc_f,
                rtc_f,
                f_obs_st,
                f_pref_st,
                f_pref_valid_st,
                f_pt_st,
                f_seq_st,
                f_tnow_st,
                f_out_st,
            )

        for d in self.all_devices:
            torch.cuda.synchronize(d)

        # 3. Allocate Slow replicas & pipeline stages and capture Slow CUDA Graphs
        active_ctx_holder: List[Optional[SlowCaptureContext]] = [None]
        auto_slice_ids: Set[int] = set()

        def _global_slice_hook(mod: nn.Module, _inp: Tuple[object, ...], _out: object) -> None:
            if (getattr(mod, "_ts_sliceable", False) or id(mod) in auto_slice_ids) and active_ctx_holder[0] is not None:
                active_ctx_holder[0].slice_point()

        global_hook = nn.modules.module.register_module_forward_hook(_global_slice_hook)

        try:
            self.slow_bucket_slices_ms.clear()
            self.slow_stage_bucket_ms = [{} for _ in range(P)]
            max_colocated_slice_ms = 0.0

            for r in range(R):
                n_r = self.scheduler.slow_replica_num_streams(r)
                r_b_slow = min(b_slow, n_r)
                r_slow_buckets = tuple([b for b in slow_buckets if b < r_b_slow] + [r_b_slow])

                for p in range(P):
                    s_dev_id = active_slow_devs[r][p]
                    s_dev = torch.device(f"cuda:{s_dev_id}")
                    writes_tsc = stage_writes_tsc[p]
                    l_off = stage_layer_offsets[p]
                    l_cnt = stage_layer_counts[p]

                    if not self.disaggregated:
                        s_tsc = self.fast_tscs[r]
                    elif writes_tsc:
                        s_max_pages = n_r * max_pages_per_slot + min(n_r, P * r_b_slow) * frame_pages
                        s_tsc = _tempo_C.TemporalStateCache(
                            num_streams=n_r,
                            num_readers=1,
                            max_pages=s_max_pages,
                            num_layers=l_cnt,
                            page_size=ch.page_size,
                            num_kv_heads=ch.num_kv_heads,
                            head_dim=ch.head_dim,
                            max_seq_len=ch.max_seq_len,
                            kv_mode=(ch.mode == "kv"),
                            max_q_len=16,
                            max_q_heads=self.max_q_heads,
                            device_id=s_dev_id,
                        )
                    else:
                        s_tsc = None
                    self.slow_tscs[r][p] = s_tsc

                    s_obs_st = (
                        torch.zeros((r_b_slow, *self.slow_obs_shape), dtype=torch.bfloat16, device=s_dev)
                        if p == 0
                        else torch.empty((0,), dtype=torch.bfloat16, device=s_dev)
                    )
                    act_in_st: List[torch.Tensor] = []
                    if p > 0:
                        for feat_shape, dt in edge_act_specs[p - 1]:
                            act_in_st.append(torch.zeros((r_b_slow, *feat_shape), dtype=dt, device=s_dev))

                    act_out_st: List[torch.Tensor] = []
                    if p + 1 < P:
                        for feat_shape, dt in edge_act_specs[p]:
                            act_out_st.append(torch.zeros((r_b_slow, *feat_shape), dtype=dt, device=s_dev))

                    if writes_tsc:
                        s_pt_st = torch.arange(
                            r_b_slow * max_pages_per_slot, dtype=torch.int32, device=s_dev
                        ).view(r_b_slow, max_pages_per_slot)
                        s_start_st = torch.full(
                            (r_b_slow,), (ch.window_frames - 1) * frame_pages, dtype=torch.int32, device=s_dev
                        )
                        s_tobs_st = torch.zeros((r_b_slow,), dtype=torch.int64, device=s_dev)
                    else:
                        s_pt_st = torch.empty((0,), dtype=torch.int32, device=s_dev)
                        s_start_st = torch.empty((0,), dtype=torch.int32, device=s_dev)
                        s_tobs_st = torch.empty((0,), dtype=torch.int64, device=s_dev)

                    self._retained_tensors.extend(
                        [s_obs_st, *act_in_st, *act_out_st, s_pt_st, s_start_st, s_tobs_st]
                    )
                    self.scheduler.init_slow_stage(
                        r,
                        p,
                        s_tsc,
                        l_off,
                        l_cnt,
                        s_obs_st,
                        act_in_st,
                        act_out_st,
                        s_pt_st,
                        s_start_st,
                        s_tobs_st,
                    )

                    slow_exec_stream = torch.cuda.ExternalStream(
                        self.scheduler.slow_exec_cuda_stream_raw(r, p), device=s_dev
                    )
                    stage_fn = self._slow_spec.stages[p]
                    torch.cuda.synchronize(s_dev)

                    with torch.no_grad(), torch.cuda.device(s_dev), torch.cuda.stream(slow_exec_stream):
                        for b_s in r_slow_buckets:
                            if writes_tsc:
                                s_pt_st[:b_s].copy_(
                                    torch.arange(b_s * max_pages_per_slot, dtype=torch.int32, device=s_dev).view(
                                        b_s, max_pages_per_slot
                                    )
                                )
                                s_start_st[:b_s].fill_((ch.window_frames - 1) * frame_pages)

                            slow_ctx = SlowCaptureContext(
                                k_pool=s_tsc.k_pool if writes_tsc else None,
                                v_pool=s_tsc.v_pool if writes_tsc else None,
                                page_timestamps=s_tsc.page_timestamps if writes_tsc else None,
                                page_table=s_pt_st[:b_s] if writes_tsc else None,
                                start_page_idx=s_start_st[:b_s] if writes_tsc else None,
                                t_obs_ns=s_tobs_st[:b_s] if writes_tsc else None,
                                spatial_rope=ch.spatial_rope,
                                relative_time_rope=ch.relative_time_rope,
                                spatial_grid_w=grid_w,
                                page_size=ch.page_size,
                                window_frames=ch.window_frames,
                                frame_seq_len=ch.frame_seq_len,
                                mode=ch.mode,
                                slice_enabled=not self.disaggregated,
                                layer_offset=l_off,
                                layer_count=l_cnt,
                                num_kv_heads=ch.num_kv_heads,
                                head_dim=ch.head_dim,
                                device=s_dev,
                            )
                            active_ctx_holder[0] = slow_ctx

                            if p == 0:
                                stg_inp: object = s_obs_st[:b_s]
                            else:
                                sliced_in = [t[:b_s] for t in act_in_st]
                                stg_inp = sliced_in[0] if edge_unpack_single[p - 1] else tuple(sliced_in)

                            def _invoke_stage() -> None:
                                res_out = stage_fn(stg_inp, slow_ctx)
                                if p + 1 < P:
                                    outs = [res_out] if edge_unpack_single[p] else list(res_out)  # type: ignore[arg-type]
                                    for dst_buf, val in zip(act_out_st, outs):
                                        dst_buf[:b_s].copy_(val)

                            slow_ctx._trace_mode = "none"
                            invoked_mods: List[nn.Module] = []
                            if not self.disaggregated:
                                trace_hook = nn.modules.module.register_module_forward_hook(
                                    lambda m, _i, _o: invoked_mods.append(m)
                                )
                                try:
                                    _invoke_stage()
                                finally:
                                    trace_hook.remove()
                                if invoked_mods and not any(getattr(m, "_ts_sliceable", False) for m in invoked_mods):
                                    for m in invoked_mods:
                                        for c in m.modules():
                                            if isinstance(c, (nn.ModuleList, nn.Sequential)):
                                                for sub in c:
                                                    auto_slice_ids.add(id(sub))
                                _invoke_stage()
                            else:
                                for _ in range(2):
                                    _invoke_stage()
                            slow_exec_stream.synchronize()

                            split_indices: Set[int] = set()
                            if not self.disaggregated:
                                best_op_times: Optional[List[float]] = None
                                for pass_idx in range(5):
                                    if pass_idx > 0:
                                        time.sleep(0.004)
                                    slow_ctx._trace_mode = "profile"
                                    slow_ctx._exec_stream = slow_exec_stream
                                    slow_ctx._op_durations_ms = []
                                    slow_ctx._step_counter = 0
                                    ev_init = torch.cuda.Event(enable_timing=True)
                                    ev_init.record(slow_exec_stream)
                                    ev_init.synchronize()
                                    slow_ctx._last_ev = ev_init

                                    _invoke_stage()
                                    ev_tail = torch.cuda.Event(enable_timing=True)
                                    ev_tail.record(slow_exec_stream)
                                    ev_tail.synchronize()
                                    tail_ms = slow_ctx._last_ev.elapsed_time(ev_tail)
                                    cur_times = list(slow_ctx._op_durations_ms) + [tail_ms]
                                    if best_op_times is None:
                                        best_op_times = cur_times
                                    else:
                                        best_op_times = [min(a, b) for a, b in zip(best_op_times, cur_times)]

                                assert best_op_times is not None
                                hard_op_limit_ms = self._slow_spec.max_slice_ms * 1.20
                                for idx_op, t_op in enumerate(best_op_times):
                                    if t_op > hard_op_limit_ms:
                                        raise RuntimeError(
                                            f"Colocated Slow replica {r} (cuda:{s_dev_id}) bucket B={b_s} segment "
                                            f"#{idx_op} took {t_op:.2f} ms > {hard_op_limit_ms:.2f} ms; mark finer "
                                            f"submodules with mark_sliceable() or reduce chunk size for "
                                            f"max_slice_ms={self._slow_spec.max_slice_ms}"
                                        )

                                budget_ms = self._slow_spec.max_slice_ms * 0.85
                                acc_ms = best_op_times[0]
                                for step_idx in range(len(best_op_times) - 1):
                                    nxt_ms = best_op_times[step_idx + 1]
                                    rem_ms = sum(best_op_times[step_idx + 1 :])
                                    if rem_ms >= 0.05 and acc_ms + nxt_ms > budget_ms:
                                        split_indices.add(step_idx)
                                        acc_ms = nxt_ms
                                    else:
                                        acc_ms += nxt_ms

                            slow_ctx._trace_mode = "capture"
                            slow_ctx._step_counter = 0
                            slow_ctx._split_indices = split_indices

                            def _make_split_cb(rep_i: int, stg_i: int, bucket_val: int) -> Callable[[], None]:
                                def _cb() -> None:
                                    self.scheduler.end_slow_slice_capture(rep_i, stg_i, bucket_val)
                                    self.scheduler.begin_slow_slice_capture(rep_i, stg_i, bucket_val)

                                return _cb

                            slow_ctx._on_split_cb = _make_split_cb(r, p, b_s)
                            slow_exec_stream.synchronize()

                            self.scheduler.begin_slow_slice_capture(r, p, b_s)
                            _invoke_stage()
                            self.scheduler.end_slow_slice_capture(r, p, b_s)
                            slow_ctx._trace_mode = "none"

                            prof_slices = list(self.scheduler.profile_slow_slices(r, p, b_s))
                            if not self.disaggregated and prof_slices:
                                max_colocated_slice_ms = max(max_colocated_slice_ms, max(prof_slices))
                            if r == 0:
                                self.slow_stage_bucket_ms[p][b_s] = round(sum(prof_slices), 2)
                                prev_list = self.slow_bucket_slices_ms.get(b_s, [])
                                self.slow_bucket_slices_ms[b_s] = prev_list + prof_slices

                        active_ctx_holder[0] = None
                        if writes_tsc and s_tsc is not None:
                            if ch.mode == "kv":
                                s_tsc.k_pool.zero_()
                            s_tsc.v_pool.zero_()
                            s_tsc.page_timestamps.zero_()
                            s_pt_st.fill_(-1)
                            s_start_st.zero_()
                        if p == 0:
                            s_obs_st.zero_()
                        for t in act_in_st:
                            t.zero_()
                        for t in act_out_st:
                            t.zero_()

            self.slice_durations_ms = list(self.slow_bucket_slices_ms.get(b_slow, []))

            # 4. Capture Fast bucket CUDA Graphs across all Fast shards
            _ensure_custom_ops()
            fast_stage_fn = self._fast_spec.stages[0]
            if self.compile_fast:
                if self.deployment is not None:
                    for shard in self.deployment.fast_shards:
                        _compile_fast_module_inplace(shard)
                else:
                    torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 256)
                    fast_stage_fn = torch.compile(fast_stage_fn, dynamic=True)

            self.bucket_durations_ms.clear()
            for f in range(F):
                f_dev_id = active_fast_devs[f]
                f_dev = torch.device(f"cuda:{f_dev_id}")
                tsc_f = self.fast_tscs[f]
                rtc_f = self.fast_rtcs[f]
                exec_stream = torch.cuda.ExternalStream(
                    self.scheduler.fast_exec_cuda_stream_raw(f), device=f_dev
                )
                (
                    f_obs_st,
                    f_pref_st,
                    f_pref_valid_st,
                    f_pt_st,
                    f_seq_st,
                    f_tnow_st,
                    f_out_st,
                ) = fast_shard_statics[f]

                with torch.no_grad(), torch.cuda.device(f_dev), torch.cuda.stream(exec_stream):
                    for b in shard_valid_buckets[f]:
                        sem_view = SemanticStateView(
                            k_pool=tsc_f.k_pool,
                            v_pool=tsc_f.v_pool,
                            page_timestamps=tsc_f.page_timestamps,
                            splitk_partial_out=tsc_f.splitk_partial_out(0),
                            splitk_partial_lse=tsc_f.splitk_partial_lse(0),
                            page_table=f_pt_st[:b],
                            seq_lens=f_seq_st[:b],
                            t_now_ns=f_tnow_st[:b],
                            relative_time_rope=ch.relative_time_rope,
                            max_seq_len=ch.max_seq_len,
                            mode=ch.mode,
                            prefix_valid=f_pref_valid_st[:b],
                        )
                        f_obs = f_obs_st[:b]
                        f_pref = f_pref_st[:b]
                        f_out = f_out_st[:b]

                        for _ in range(3):
                            res = fast_stage_fn(f_obs, sem_view, f_pref)
                            if (
                                not isinstance(res, torch.Tensor)
                                or res.shape != f_out.shape
                                or res.dtype != torch.bfloat16
                            ):
                                got_desc = (
                                    f"{res.dtype} {tuple(res.shape)}"
                                    if isinstance(res, torch.Tensor)
                                    else type(res).__name__
                                )
                                raise ValueError(
                                    f"Fast stage must return bfloat16 Tensor of shape {tuple(f_out.shape)} "
                                    f"(batch, horizon_steps, action_dim), got {got_desc}"
                                )
                            f_out.copy_(res)
                        exec_stream.synchronize()

                        self.scheduler.begin_fast_bucket_capture(f, b)
                        res = fast_stage_fn(f_obs, sem_view, f_pref)
                        f_out.copy_(res)
                        b_ms = self.scheduler.end_fast_bucket_capture(f, b)
                        if f == 0 or b not in self.bucket_durations_ms:
                            self.bucket_durations_ms[b] = b_ms
                        else:
                            self.bucket_durations_ms[b] = max(self.bucket_durations_ms[b], b_ms)

                    if ch.mode == "kv":
                        tsc_f.k_pool.zero_()
                    tsc_f.v_pool.zero_()
                    tsc_f.page_timestamps.zero_()
                    f_obs_st.zero_()
                    f_pref_st.zero_()
                    f_pref_valid_st.zero_()
                    f_out_st.zero_()
                    f_pt_st.fill_(-1)
                    f_seq_st.zero_()
                    rtc_f.reset_all()
        finally:
            global_hook.remove()
            self.scheduler.abort_active_captures()

        for d in self.all_devices:
            torch.cuda.synchronize(d)
        self.scheduler.reset_telemetry()
        self.scheduler.reset_rtc_stats()

        mode_name = "remote_slow" if remote_slow else ("disaggregated" if self.disaggregated else "colocated")
        return {
            "mode": mode_name,
            "slow_devices": active_slow_devs,
            "fast_devices": active_fast_devs,
            "p2p_direct": bool(self.scheduler.p2p_direct_enabled()),
            "num_slow_slices": len(self.slice_durations_ms),
            "max_slow_slice_ms": 0.0 if self.disaggregated else max_colocated_slice_ms,
            "slow_bucket_total_ms": {b: round(sum(v), 2) for b, v in self.slow_bucket_slices_ms.items()},
            "slow_stage_bucket_ms": self.slow_stage_bucket_ms,
            "mean_slow_slice_ms": sum(self.slice_durations_ms) / max(len(self.slice_durations_ms), 1),
            "fast_bucket_ms": dict(self.bucket_durations_ms),
            "tsc_pool_pages": total_tsc_pages,
        }

    def _require_compiled(self):
        if self.scheduler is None:
            raise RuntimeError("Engine.compile_and_calibrate() must be called before starting or running the engine")
        return self.scheduler

    def start(self) -> "Engine":
        if self.scheduler is None:
            self.calibration_info = self.compile_and_calibrate()
            self.scheduler.reset_all_streams(None)
        if not self._running:
            self.scheduler.start()
            self._running = True
            _ACTIVE_ENGINES.add(self)
        return self

    def stop(self) -> None:
        if self.scheduler is not None:
            self.scheduler.stop()
        self._running = False
        _ACTIVE_ENGINES.discard(self)

    def __del__(self) -> None:
        if self._running and self.scheduler is not None:
            self.scheduler.stop()
            self._running = False

    def _note_remote_slow_push(self, stream_id: int, t_obs_ns: int) -> None:
        self._require_compiled().note_expected_slow_obs(stream_id, t_obs_ns)

    def wait_ready(self, fast_obs: Optional[torch.Tensor] = None, *, timeout_s: float = 10.0) -> None:
        """Blocks until all pending Slow updates settle and a fresh Fast action chunk becomes active."""
        sched = self._require_compiled()
        assert self._fast_spec is not None
        if not self._running:
            raise RuntimeError("Engine must be started before calling wait_ready()")
        if sched.is_clock_paused():
            raise RuntimeError("Cannot call wait_ready() while engine clock is paused")
        if fast_obs is None and not self._fast_spec.fast_obs_is_noise:
            raise ValueError(
                "fast_obs is required in wait_ready(fast_obs) because this model's Fast stage "
                "consumes state rather than flow noise"
            )
        if fast_obs is not None:
            batch_shape = (self.num_streams, *self.fast_obs_shape)
            single_ok = self.num_streams == 1 and fast_obs.shape == self.fast_obs_shape
            if not single_ok and fast_obs.shape != batch_shape:
                expected = f"{self.fast_obs_shape} (or {batch_shape})" if self.num_streams == 1 else str(batch_shape)
                raise ValueError(f"wait_ready expected fast_obs of shape {expected}, got {tuple(fast_obs.shape)}")
            if not (fast_obs.is_floating_point() or fast_obs.dtype == torch.uint8):
                raise TypeError(
                    f"wait_ready expects a floating-point (or uint8 image) tensor, got {fast_obs.dtype}; "
                    "embed integer token IDs before pushing"
                )

        deadline = time.monotonic() + float(timeout_s)
        while not sched.all_streams_slow_settled():
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out after {timeout_s:.1f}s waiting for Slow stage to publish across all streams "
                    "(ensure push_slow / push_slow_batch was called for all streams)"
                )
            time.sleep(0.001)

        if fast_obs is not None and self.num_streams == 1 and fast_obs.shape == self.fast_obs_shape:
            t_fast_ns = self.push_fast(fast_obs)
        else:
            t_fast_ns = self.push_fast_batch(fast_obs)

        while not sched.all_streams_fast_committed_since(t_fast_ns):
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Timed out after {timeout_s:.1f}s waiting for Fast stage to commit across all streams"
                )
            time.sleep(0.001)

        splice_ns = int(sched.max_staged_splice_ns())
        if splice_ns > 0:
            rem_ns = splice_ns - sched.now_ns()
            if rem_ns > 0:
                if time.monotonic() + rem_ns * 1e-9 >= deadline:
                    raise TimeoutError(
                        f"Timed out after {timeout_s:.1f}s waiting for staged Fast chunk splice"
                    )
                time.sleep(rem_ns * 1e-9)
            self.sample_actions(max(splice_ns, sched.now_ns()), cpu=False)

    def __enter__(self) -> "Engine":
        return self.start()

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    def set_prompt(self, prompt: Union[str, Sequence[int], torch.Tensor], **kwargs: object) -> None:
        """Pauses the scheduler, updates the shared prompt buffer on all Slow replicas, and resumes."""
        if self.deployment is None:
            raise RuntimeError("set_prompt() requires an Engine constructed with model=... or (slow=..., fast=...)")
        was_running = self._running
        if was_running and self.scheduler is not None:
            self.scheduler.stop()
            self._running = False
        try:
            for rep in self.deployment.slow_replicas:
                rep[0].set_prompt(prompt, **kwargs)  # type: ignore[attr-defined]
            for pipe in self.active_slow_devices:
                torch.cuda.synchronize(pipe[0])
        finally:
            if was_running and self.scheduler is not None:
                self.scheduler.start()
                self._running = True

    def set_state(self, state: torch.Tensor) -> None:
        """Pauses the scheduler, updates the shared robot state buffer on all Slow replicas, and resumes."""
        if self.deployment is None:
            raise RuntimeError("set_state() requires an Engine constructed with model=... or (slow=..., fast=...)")
        was_running = self._running
        if was_running and self.scheduler is not None:
            self.scheduler.stop()
            self._running = False
        try:
            for rep in self.deployment.slow_replicas:
                rep[0].set_state(state)  # type: ignore[attr-defined]
            for pipe in self.active_slow_devices:
                torch.cuda.synchronize(pipe[0])
        finally:
            if was_running and self.scheduler is not None:
                self.scheduler.start()
                self._running = True

    def now_ns(self) -> int:
        return self._require_compiled().now_ns()

    def pause_clock(self) -> None:
        """Pauses the engine's logical clock during slower-than-real-time simulator stepping."""
        self._require_compiled().pause_clock()

    def resume_clock(self) -> None:
        """Resumes the engine's logical clock after a paused simulator step."""
        self._require_compiled().resume_clock()

    def reset(self, q_init: Optional[torch.Tensor] = None) -> None:
        """Resets all stream trajectories, shared cache slots, and stats."""
        sched = self._require_compiled()
        was_running = self._running
        if was_running:
            sched.stop()
            self._running = False
        sched.reset_all_streams(q_init)
        if was_running:
            sched.start()
            self._running = True

    def reset_stats(self) -> None:
        sched = self._require_compiled()
        sched.reset_rtc_stats()
        sched.reset_telemetry()

    def rtc_stats(self):
        """Returns cumulative RTC ring counters (``hard_underruns``, ``overruns``, ``commits``)."""
        return self._require_compiled().rtc_stats()

    def telemetry(self):
        """Returns C++ scheduler telemetry (``fast_ttac_ms`` chunk latencies, ``slow_latency_ms``, ``fast_requests_served``)."""
        return self._require_compiled().telemetry()

    def push_slow(
        self,
        obs: torch.Tensor,
        *,
        stream_id: int = 0,
        t_obs_ns: Optional[int] = None,
    ) -> int:
        """Submits a Slow observation for ``stream_id`` (uint8 [0, 255] scales to [0, 1] bf16; floats keep range)."""
        sched = self._require_compiled()
        if stream_id < 0 or stream_id >= self.num_streams:
            raise IndexError(f"stream_id {stream_id} out of range [0, {self.num_streams})")
        obs = _to_bf16_obs(obs, "push_slow")
        if t_obs_ns is None:
            t_obs_ns = sched.now_ns()
        sched.submit_slow_obs(stream_id, t_obs_ns, obs)
        return t_obs_ns

    def push_slow_batch(self, obs_batch: torch.Tensor, *, t_obs_ns: Optional[int] = None) -> int:
        sched = self._require_compiled()
        if not self.active_slow_devices:
            raise RuntimeError("push_slow_batch called on Fast-only Engine without a local Slow stage")
        expected = (self.num_streams, *self.slow_obs_shape)
        if obs_batch.shape != expected:
            raise ValueError(f"push_slow_batch expected shape {expected}, got {tuple(obs_batch.shape)}")
        if not (obs_batch.is_floating_point() or obs_batch.dtype == torch.uint8):
            raise TypeError(
                f"push_slow_batch expects a floating-point (or uint8 image) tensor, got {obs_batch.dtype}; "
                "embed integer token IDs before pushing"
            )
        if t_obs_ns is None:
            t_obs_ns = sched.now_ns()
        R = len(self.active_slow_devices)
        for r, pipe in enumerate(self.active_slow_devices):
            shard_obs = _to_bf16_obs(obs_batch[r::R], "push_slow_batch", device=f"cuda:{pipe[0]}")
            for idx, s in enumerate(range(r, self.num_streams, R)):
                sched.submit_slow_obs(s, t_obs_ns, shard_obs[idx])
        return t_obs_ns

    def publish_remote_hidden(self, stream_id: int, hidden: torch.Tensor, t_obs_ns: int) -> bool:
        return bool(self._require_compiled().submit_remote_hidden(stream_id, t_obs_ns, hidden))

    def push_fast(
        self,
        fast_obs: Optional[torch.Tensor] = None,
        *,
        stream_id: int = 0,
        t_obs_ns: Optional[int] = None,
    ) -> int:
        """Submits a Fast observation for ``stream_id`` (or samples standard normal GPU flow noise when ``fast_obs=None``)."""
        sched = self._require_compiled()
        assert self._fast_spec is not None
        if stream_id < 0 or stream_id >= self.num_streams:
            raise IndexError(f"stream_id {stream_id} out of range [0, {self.num_streams})")
        if fast_obs is None:
            if not self._fast_spec.fast_obs_is_noise:
                raise ValueError(
                    "fast_obs is required for this model because its Fast stage consumes state rather than flow noise"
                )
            dst_dev = self.active_fast_devices[stream_id % len(self.active_fast_devices)]
            fast_obs = torch.randn(self.fast_obs_shape, dtype=torch.bfloat16, device=f"cuda:{dst_dev}")
        else:
            fast_obs = _to_bf16_obs(fast_obs, "push_fast")
        if t_obs_ns is None:
            t_obs_ns = sched.now_ns()
        sched.submit_fast_obs(stream_id, t_obs_ns, fast_obs)
        return t_obs_ns

    def push_fast_batch(
        self,
        fast_obs_batch: Optional[torch.Tensor] = None,
        *,
        t_obs_ns: Optional[int] = None,
    ) -> int:
        sched = self._require_compiled()
        assert self._fast_spec is not None
        expected = (self.num_streams, *self.fast_obs_shape)
        if fast_obs_batch is not None:
            if fast_obs_batch.shape != expected:
                raise ValueError(f"push_fast_batch expected shape {expected}, got {tuple(fast_obs_batch.shape)}")
            if not (fast_obs_batch.is_floating_point() or fast_obs_batch.dtype == torch.uint8):
                raise TypeError(
                    f"push_fast_batch expects a floating-point (or uint8 image) tensor, got {fast_obs_batch.dtype}; "
                    "embed integer token IDs before pushing"
                )
        elif not self._fast_spec.fast_obs_is_noise:
            raise ValueError(
                "fast_obs_batch is required for this model because its Fast stage consumes state rather than flow noise"
            )
        if t_obs_ns is None:
            t_obs_ns = sched.now_ns()
        F = len(self.active_fast_devices)
        for f, f_dev in enumerate(self.active_fast_devices):
            s_ids = range(f, self.num_streams, F)
            if fast_obs_batch is None:
                shard_obs = torch.randn(
                    (len(s_ids), *self.fast_obs_shape), dtype=torch.bfloat16, device=f"cuda:{f_dev}"
                )
            else:
                shard_obs = _to_bf16_obs(fast_obs_batch[f::F], "push_fast_batch", device=f"cuda:{f_dev}")
            for idx, s in enumerate(s_ids):
                sched.submit_fast_obs(s, t_obs_ns, shard_obs[idx])
        return t_obs_ns

    def sample_actions(self, t_query_ns: Optional[int] = None, cpu: bool = False):
        """Non-blocking RTC ring read at ``t_query_ns`` (defaults to ``now_ns()``).

        Returns ``(actions [N, action_dim], fast_age_ms [N], slow_age_ms [N], underruns_this_tick: int)``.
        Before a stream's first chunk commits, its ``fast_age_ms`` and ``slow_age_ms`` are ``-1.0``.
        """
        sched = self._require_compiled()
        if t_query_ns is None:
            t_query_ns = sched.now_ns()
        return sched.sample_actions_at(t_query_ns, cpu)
