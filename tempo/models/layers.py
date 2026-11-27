from __future__ import annotations

import itertools
import math
from typing import List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..channel import SemanticStateView, TemporalChannel, mark_sliceable
from .base import FastHead, validate_fast_precision
from .hub import SafetensorLoader


def require_fp8_device(device: Union[str, torch.device]) -> None:
    dev = torch.device(device)
    if dev.type != "cuda":
        raise RuntimeError(f"fast_precision='fp8' requires a CUDA device, got {dev}")
    major, minor = torch.cuda.get_device_capability(dev)
    if (major, minor) < (8, 9):
        raise RuntimeError(
            f"fast_precision='fp8' requires Ada Lovelace (sm_89) or Hopper (sm_90+) Tensor Cores, "
            f"got sm_{major}{minor} on {dev}"
        )


class FP8Linear(nn.Module):
    """BF16 or calibrated static-scale FP8 (float8_e4m3fn) Linear layer for CUDA Graph Fast heads."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = False,
        use_fp8: bool = False,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_fp8 = use_fp8 and (in_features % 16 == 0) and (out_features % 16 == 0)
        if use_fp8 and not self.use_fp8:
            raise ValueError(
                f"FP8Linear requires in_features ({in_features}) and out_features ({out_features}) "
                f"to be multiples of 16"
            )
        dev = torch.device(device) if device is not None else torch.device("cpu")
        if self.use_fp8 and dev.type == "cuda":
            require_fp8_device(dev)

        if self.use_fp8:
            self.register_parameter("weight", None)
            init_w = torch.empty((out_features, in_features), dtype=torch.float32, device=dev)
            nn.init.normal_(init_w, mean=0.0, std=0.02)
            ws = (init_w.abs().amax() / 448.0).clamp_min(1e-12).reshape(1)
            qf = (init_w / ws).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).contiguous()
            self.register_buffer("w_fp8", qf)
            self.register_buffer("w_scale", ws)
            self.register_buffer("act_scale", torch.full((1,), 8.0 / 448.0, dtype=torch.float32, device=dev))
            self.register_buffer("inv_act_scale", torch.full((1,), 448.0 / 8.0, dtype=torch.float32, device=dev))
        else:
            self.weight = nn.Parameter(
                torch.empty((out_features, in_features), dtype=torch.bfloat16, device=dev),
                requires_grad=False,
            )
            nn.init.normal_(self.weight, mean=0.0, std=0.02)

        if bias:
            self.bias = nn.Parameter(
                torch.zeros((out_features,), dtype=torch.bfloat16, device=dev),
                requires_grad=False,
            )
        else:
            self.register_parameter("bias", None)

        self._calibrating: bool = False
        self._obs_amax: float = 0.0

    @torch.no_grad()
    def set_weight(self, weight_bf16: torch.Tensor, bias_bf16: Optional[torch.Tensor] = None) -> None:
        dev = self.w_fp8.device if self.use_fp8 else self.weight.device
        w = weight_bf16.to(device=dev, dtype=torch.bfloat16)
        if w.shape != (self.out_features, self.in_features):
            raise ValueError(
                f"Expected weight shape {(self.out_features, self.in_features)}, got {tuple(w.shape)}"
            )
        if self.use_fp8:
            if dev.type == "cuda":
                require_fp8_device(dev)
            ws = (w.abs().amax().float() / 448.0).clamp_min(1e-12).reshape(1)
            qf = (w.float() / ws).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).contiguous()
            self.w_fp8.copy_(qf)
            self.w_scale.copy_(ws)
        else:
            self.weight.copy_(w)

        if bias_bf16 is not None:
            if self.bias is None:
                raise ValueError("Bias tensor provided to FP8Linear initialized with bias=False")
            self.bias.copy_(bias_bf16.to(device=dev, dtype=torch.bfloat16))

    def begin_calibration(self) -> None:
        if self.use_fp8:
            require_fp8_device(self.w_fp8.device)
            self._calibrating = True
            self._obs_amax = 0.0

    @torch.no_grad()
    def finish_calibration(self) -> None:
        if self.use_fp8 and self._calibrating:
            self._calibrating = False
            s = max(self._obs_amax * 1.25 / 448.0, 1e-4)
            self.act_scale.fill_(s)
            self.inv_act_scale.fill_(1.0 / s)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_fp8:
            return F.linear(x, self.weight, self.bias)
        if self._calibrating:
            cur = float(x.detach().abs().amax().item())
            if cur > self._obs_amax:
                self._obs_amax = cur
        orig_shape = x.shape[:-1]
        x_2d = x.reshape(-1, self.in_features)
        x_fp8 = (x_2d.float() * self.inv_act_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        out_2d = torch._scaled_mm(
            x_fp8,
            self.w_fp8.t(),
            scale_a=self.act_scale,
            scale_b=self.w_scale,
            bias=self.bias,
            out_dtype=torch.bfloat16,
        )
        return out_2d.view(*orig_shape, self.out_features)


def calibrate_fp8_submodules(root: nn.Module, run_fn) -> None:
    fp8_mods = [m for m in root.modules() if isinstance(m, FP8Linear) and m.use_fp8]
    if not fp8_mods:
        return
    for m in fp8_mods:
        m.begin_calibration()
    try:
        with torch.no_grad():
            run_fn()
    finally:
        for m in fp8_mods:
            m.finish_calibration()


class _SlowCalibrationCapture:
    """Collects 1-frame Slow KV or Hidden outputs across pipeline stages for realistic FP8 calibration."""

    def __init__(self, layers: int) -> None:
        self.k_layers: List[Optional[torch.Tensor]] = [None] * layers
        self.v_layers: List[Optional[torch.Tensor]] = [None] * layers
        self.hidden: Optional[torch.Tensor] = None
        self.prompt_slot_ids: Optional[torch.Tensor] = None
        self.seq_lens: Optional[torch.Tensor] = None
        self._probe_mode: bool = False

    def write_layer_kv(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
        self.k_layers[layer_idx] = k.detach()
        self.v_layers[layer_idx] = v.detach()

    def write_hidden_state(self, hidden: torch.Tensor) -> None:
        self.hidden = hidden.detach()


@torch.no_grad()
def capture_slow_calibration_sample(
    slow_stages: Sequence[nn.Module],
    channel: TemporalChannel,
    obs_shape: Tuple[int, ...],
    batch_size: int = 2,
) -> Tuple[Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]:
    """Runs one synthetic observation batch through Slow stages to capture realistic KV/Hidden magnitudes."""
    first_t = next(itertools.chain(slow_stages[0].parameters(), slow_stages[0].buffers()))
    x = torch.rand((batch_size, *obs_shape), dtype=torch.bfloat16, device=first_t.device).contiguous()
    s0 = slow_stages[0]
    if hasattr(s0, "get_stream_prompt_meta"):
        cal_seq_len, active_row = s0.get_stream_prompt_meta(0)  # type: ignore[operator]
    else:
        cal_seq_len, active_row = channel.max_seq_len, 0
    cap = _SlowCalibrationCapture(channel.layers)
    cap.prompt_slot_ids = torch.full((batch_size,), int(active_row), dtype=torch.int64, device=first_t.device)
    cap.seq_lens = torch.full((batch_size,), int(cal_seq_len), dtype=torch.int32, device=first_t.device)
    for stg in slow_stages:
        stg_t = next(itertools.chain(stg.parameters(), stg.buffers()))
        if cap.prompt_slot_ids is not None and cap.prompt_slot_ids.device != stg_t.device:
            cap.prompt_slot_ids = cap.prompt_slot_ids.to(device=stg_t.device)
        if cap.seq_lens is not None and cap.seq_lens.device != stg_t.device:
            cap.seq_lens = cap.seq_lens.to(device=stg_t.device)
        x = stg(x.to(device=stg_t.device, dtype=torch.bfloat16), cap)
    if channel.mode == "kv":
        if any(t is None for t in cap.k_layers) or any(t is None for t in cap.v_layers):
            return None, None
        k_all = torch.stack([t.cpu() for t in cap.k_layers if t is not None], dim=0)
        v_all = torch.stack([t.cpu() for t in cap.v_layers if t is not None], dim=0)
        return (k_all, v_all), None
    if cap.hidden is not None:
        return None, cap.hidden.cpu()
    return None, None


class RMSNorm(nn.Module):
    """Standard float32-variance RMSNorm (Qwen3, SmolLM2, Llama)."""

    def __init__(self, dim: int, eps: float = 1e-6, device: Optional[Union[str, torch.device]] = None) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones((dim,), dtype=torch.bfloat16, device=device), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        normed = x_f * torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (normed * self.weight.float()).to(dtype=x.dtype)


class GemmaRMSNorm(nn.Module):
    """Gemma / PaliGemma unit-offset RMSNorm: normed * (1.0 + weight)."""

    def __init__(self, dim: int, eps: float = 1e-6, device: Optional[Union[str, torch.device]] = None) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros((dim,), dtype=torch.bfloat16, device=device), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_f = x.float()
        normed = x_f * torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return (normed * (1.0 + self.weight.float())).to(dtype=x.dtype)


class AdaRMSNorm(nn.Module):
    """Adaptive RMSNorm with gated residual modulation used in pi0.5 Gemma Action Expert."""

    def __init__(
        self,
        dim: int,
        cond_dim: int,
        eps: float = 1e-6,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.cond_dim = cond_dim
        self.eps = eps
        self.dense = nn.Linear(cond_dim, dim * 3, bias=True, device=device, dtype=torch.bfloat16)
        self.dense.requires_grad_(False)
        nn.init.zeros_(self.dense.weight)
        nn.init.zeros_(self.dense.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mod = self.dense(cond)
        if mod.ndim == 2:
            mod = mod.unsqueeze(1)
        scale, shift, gate = mod.chunk(3, dim=-1)
        x_f = x.float()
        normed = x_f * torch.rsqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        out = normed * (1.0 + scale.float()) + shift.float()
        return out.to(dtype=x.dtype), gate


def apply_rope_1d(
    x: torch.Tensor,
    positions: torch.Tensor,
    *,
    rope_theta: float = 10000.0,
) -> torch.Tensor:
    """Applies half-split 1D Rotary Position Embedding to [B, S, H, D]."""
    d_half = x.size(-1) // 2
    freq_seq = torch.arange(d_half, device=x.device, dtype=torch.float32) / float(d_half)
    inv_freq = rope_theta ** (-freq_seq)
    if positions.ndim == 1:
        angles = positions.float().unsqueeze(0).unsqueeze(-1) * inv_freq.view(1, 1, -1)
    else:
        angles = positions.float().unsqueeze(-1) * inv_freq.view(1, 1, -1)
    cos = torch.cos(angles).unsqueeze(2)
    sin = torch.sin(angles).unsqueeze(2)
    x_f = x.float()
    x1 = x_f[..., :d_half]
    x2 = x_f[..., d_half:]
    out = torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)
    return out.to(dtype=x.dtype)


def sinusoidal_time_embedding(
    timestep: torch.Tensor,
    dim: int,
    *,
    min_period: float = 0.004,
    max_period: float = 4.0,
) -> torch.Tensor:
    """Continuous sinusoidal flow-matching timestep embedding [B, dim] in float32."""
    half = dim // 2
    fraction = torch.linspace(0.0, 1.0, half, dtype=torch.float32, device=timestep.device)
    period = min_period * ((max_period / min_period) ** fraction)
    angles = timestep.float().unsqueeze(-1) * (2.0 * math.pi / period).unsqueeze(0)
    return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


def suffix_self_attention_with_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Computes GQA self-attention over suffix tokens [B, Q, H, D] plus exact float32 LSE via FlashAttention-2."""
    B, Q_len, H_q, D = q.shape
    H_kv = k.size(2)
    groups = H_q // H_kv
    qt = q.transpose(1, 2).contiguous()
    if groups > 1:
        kt = k.unsqueeze(3).expand(B, Q_len, H_kv, groups, D).reshape(B, Q_len, H_q, D).transpose(1, 2).contiguous()
        vt = v.unsqueeze(3).expand(B, Q_len, H_kv, groups, D).reshape(B, Q_len, H_q, D).transpose(1, 2).contiguous()
    else:
        kt = k.transpose(1, 2).contiguous()
        vt = v.transpose(1, 2).contiguous()
    attn_out, lse, *_ = torch.ops.aten._scaled_dot_product_flash_attention(
        qt, kt, vt, 0.0, causal, False, scale=1.0 / math.sqrt(D)
    )
    return attn_out.transpose(1, 2).contiguous(), lse.transpose(1, 2).contiguous()


def merge_attention_lse(
    out_a: torch.Tensor,
    lse_a: torch.Tensor,
    out_b: torch.Tensor,
    lse_b: torch.Tensor,
) -> torch.Tensor:
    """Merges two partial attention outputs (e.g. Paged TSC prefix + local suffix) via exact log-sum-exp."""
    m = torch.maximum(lse_a, lse_b)
    wa = torch.exp(lse_a - m).unsqueeze(-1)
    wb = torch.exp(lse_b - m).unsqueeze(-1)
    merged = (wa * out_a.float() + wb * out_b.float()) / (wa + wb)
    return merged.to(torch.bfloat16)


class SigLIPEncoderLayer(nn.Module):
    """Standard SigLIP / ViT bidirectional encoder block with micro-sliceable boundary."""

    def __init__(
        self,
        hidden_size: int = 1152,
        num_heads: int = 16,
        intermediate_size: int = 4304,
        eps: float = 1e-6,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.layer_norm1 = nn.LayerNorm(hidden_size, eps=eps, device=device, dtype=torch.bfloat16)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=True, device=device, dtype=torch.bfloat16)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=True, device=device, dtype=torch.bfloat16)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=True, device=device, dtype=torch.bfloat16)
        self.out_proj = mark_sliceable(
            nn.Linear(hidden_size, hidden_size, bias=True, device=device, dtype=torch.bfloat16)
        )
        self.layer_norm2 = nn.LayerNorm(hidden_size, eps=eps, device=device, dtype=torch.bfloat16)
        self.fc1 = mark_sliceable(
            nn.Linear(hidden_size, intermediate_size, bias=True, device=device, dtype=torch.bfloat16)
        )
        self.fc2 = mark_sliceable(
            nn.Linear(intermediate_size, hidden_size, bias=True, device=device, dtype=torch.bfloat16)
        )
        self.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        h = self.layer_norm1(x)
        q = self.q_proj(h).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, S, self.hidden_size)
        x = x + self.out_proj(attn)
        h2 = F.gelu(self.fc1(self.layer_norm2(x)), approximate="tanh")
        return x + self.fc2(h2)


class SigLIPVisionTower(nn.Module):
    """Multi-camera SigLIP Vision Tower (used by PaliGemma, GR00T Eagle2, and SmolVLM2)."""

    def __init__(
        self,
        *,
        hidden_size: int = 1152,
        intermediate_size: int = 4304,
        num_layers: int = 27,
        num_heads: int = 16,
        image_size: int = 224,
        patch_size: int = 14,
        eps: float = 1e-6,
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        in_patch_dim = 3 * patch_size * patch_size
        self.patch_embedding = mark_sliceable(
            nn.Linear(in_patch_dim, hidden_size, bias=True, device=device, dtype=torch.bfloat16)
        )
        self.position_embedding = nn.Embedding(
            self.num_patches, hidden_size, device=device, dtype=torch.bfloat16
        )
        self.layers = nn.ModuleList(
            [
                mark_sliceable(
                    SigLIPEncoderLayer(
                        hidden_size=hidden_size,
                        num_heads=num_heads,
                        intermediate_size=intermediate_size,
                        eps=eps,
                        device=device,
                    )
                )
                for _ in range(num_layers)
            ]
        )
        self.post_layernorm = nn.LayerNorm(hidden_size, eps=eps, device=device, dtype=torch.bfloat16)
        self.requires_grad_(False)

    def load_weights(self, loader: SafetensorLoader, prefix: str, device: Union[str, torch.device]) -> None:
        with torch.no_grad():
            conv_w = loader.tensor(f"{prefix}.embeddings.patch_embedding.weight", device=device)
            self.patch_embedding.weight.copy_(conv_w.reshape(self.hidden_size, -1))
            self.patch_embedding.bias.copy_(
                loader.tensor(f"{prefix}.embeddings.patch_embedding.bias", device=device)
            )
            self.position_embedding.weight.copy_(
                loader.tensor(f"{prefix}.embeddings.position_embedding.weight", device=device)
            )
            for idx, blk in enumerate(self.layers):
                lp = f"{prefix}.encoder.layers.{idx}"
                blk.layer_norm1.weight.copy_(loader.tensor(f"{lp}.layer_norm1.weight", device=device))
                blk.layer_norm1.bias.copy_(loader.tensor(f"{lp}.layer_norm1.bias", device=device))
                blk.layer_norm2.weight.copy_(loader.tensor(f"{lp}.layer_norm2.weight", device=device))
                blk.layer_norm2.bias.copy_(loader.tensor(f"{lp}.layer_norm2.bias", device=device))
                loader.load_linear(blk.q_proj, f"{lp}.self_attn.q_proj", device=device)
                loader.load_linear(blk.k_proj, f"{lp}.self_attn.k_proj", device=device)
                loader.load_linear(blk.v_proj, f"{lp}.self_attn.v_proj", device=device)
                loader.load_linear(blk.out_proj, f"{lp}.self_attn.out_proj", device=device)
                loader.load_linear(blk.fc1, f"{lp}.mlp.fc1", device=device)
                loader.load_linear(blk.fc2, f"{lp}.mlp.fc2", device=device)
            self.post_layernorm.weight.copy_(loader.tensor(f"{prefix}.post_layernorm.weight", device=device))
            self.post_layernorm.bias.copy_(loader.tensor(f"{prefix}.post_layernorm.bias", device=device))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encodes [B, num_cams, 3, H, W] in [0, 1] -> [B, num_cams, num_patches, hidden_size]."""
        images = images * 2.0 - 1.0
        B, Cams, C, H, W = images.shape
        p = self.patch_size
        gh, gw = H // p, W // p
        flat_patches = (
            images.reshape(B * Cams, C, gh, p, gw, p)
            .permute(0, 2, 4, 1, 3, 5)
            .reshape(B * Cams, gh * gw, C * p * p)
        )
        x = self.patch_embedding(flat_patches) + self.position_embedding.weight.unsqueeze(0)
        for blk in self.layers:
            x = blk(x)
        x = self.post_layernorm(x)
        return x.view(B, Cams, self.num_patches, self.hidden_size)


class DirectActionHead(FastHead):
    """Single-step cross-attention action head compatible with ``mode='kv'`` or ``mode='hidden'`` channels."""

    def __init__(
        self,
        channel: TemporalChannel,
        *,
        action_dim: int = 32,
        chunk_size: int = 50,
        state_dim: int = 32,
        hidden_dim: int = 512,
        num_layers: int = 4,
        fast_precision: str = "bf16",
        device: Optional[Union[str, torch.device]] = None,
    ) -> None:
        super().__init__()
        validate_fast_precision(fast_precision)
        self.fast_precision = fast_precision
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.obs_shape = (state_dim,)
        self.mode = channel.mode
        self._expected_layers = channel.layers
        self._expected_kv_heads = channel.num_kv_heads
        self._expected_head_dim = channel.head_dim

        use_fp8 = fast_precision == "fp8"
        if self.mode == "kv":
            self.num_q_heads = max(channel.num_kv_heads, (hidden_dim // channel.head_dim // channel.num_kv_heads) * channel.num_kv_heads)
            self.num_q_heads = max(channel.num_kv_heads, self.num_q_heads)
            self.head_dim = channel.head_dim
            attn_dim = self.num_q_heads * self.head_dim
            self.max_q_heads = self.num_q_heads
        else:
            self.num_q_heads = 8
            self.head_dim = hidden_dim // 8
            attn_dim = hidden_dim
            self.max_q_heads = 8
        self.max_q_len = max(64, chunk_size)

        dev = torch.device(device) if device is not None else torch.device("cpu")
        self.query_embed = nn.Parameter(
            torch.randn((chunk_size, hidden_dim), dtype=torch.bfloat16, device=dev) * 0.02,
            requires_grad=False,
        )
        self.state_proj = nn.Linear(state_dim, hidden_dim, bias=True, device=dev, dtype=torch.bfloat16)
        self.prefix_proj = nn.Linear(action_dim, hidden_dim, bias=True, device=dev, dtype=torch.bfloat16)

        self.blocks = nn.ModuleList()
        for idx in range(num_layers):
            blk = nn.ModuleDict(
                {
                    "norm1": RMSNorm(hidden_dim, device=dev),
                    "q_proj": FP8Linear(hidden_dim, attn_dim, use_fp8=use_fp8, device=dev),
                    "o_proj": FP8Linear(attn_dim, hidden_dim, use_fp8=use_fp8, device=dev),
                    "norm2": RMSNorm(hidden_dim, device=dev),
                    "fc1": FP8Linear(hidden_dim, hidden_dim * 4, use_fp8=use_fp8, device=dev),
                    "fc2": FP8Linear(hidden_dim * 4, hidden_dim, use_fp8=use_fp8, device=dev),
                }
            )
            if self.mode == "hidden":
                blk["kv_proj"] = nn.Linear(
                    channel.feature_dim, attn_dim * 2, bias=False, device=dev, dtype=torch.bfloat16
                )
            self.blocks.append(blk)

        self.final_norm = RMSNorm(hidden_dim, device=dev)
        self.out_proj = nn.Linear(hidden_dim, action_dim, bias=True, device=dev, dtype=torch.bfloat16)
        self.requires_grad_(False)

    def validate_channel(self, channel: TemporalChannel) -> None:
        if channel.mode != self.mode:
            raise ValueError(f"DirectActionHead expects mode='{self.mode}', got '{channel.mode}'")
        if self.mode == "kv":
            if channel.num_kv_heads != self._expected_kv_heads or channel.head_dim != self._expected_head_dim:
                raise ValueError(
                    f"KV shape mismatch: expected ({self._expected_kv_heads}, {self._expected_head_dim}), "
                    f"got ({channel.num_kv_heads}, {channel.head_dim})"
                )
        else:
            expected_feat = self._expected_kv_heads * self._expected_head_dim
            if channel.feature_dim != expected_feat:
                raise ValueError(f"Hidden dim mismatch: expected {expected_feat}, got {channel.feature_dim}")

    def calibrate_fp8(self, channel: TemporalChannel, device: torch.device) -> None:
        if self.fast_precision != "fp8":
            return
        require_fp8_device(device)
        from ..channel import _load_ext

        _tempo_C = _load_ext()
        B = 4
        max_pages = B * math.ceil(channel.max_seq_len / channel.page_size) + 8
        kv_mode = channel.mode == "kv"
        tsc = _tempo_C.TemporalStateCache(
            B,
            1,
            max_pages,
            channel.layers,
            channel.page_size,
            channel.num_kv_heads,
            channel.head_dim,
            channel.max_seq_len,
            kv_mode,
            self.max_q_len,
            self.max_q_heads,
            device.index or 0,
        )
        num_p = math.ceil(channel.max_seq_len / channel.page_size)
        pt_rows = []
        for b in range(B):
            if kv_mode:
                k = (
                    torch.randn(
                        (channel.layers, 1, channel.max_seq_len, channel.num_kv_heads, channel.head_dim),
                        dtype=torch.bfloat16,
                        device=device,
                    )
                    * 0.5
                )
                tsc.write_single_stream_kv(b, k, torch.randn_like(k), 100_000_000, False)
            else:
                hid = (
                    torch.randn(
                        (1, channel.max_seq_len, channel.feature_dim),
                        dtype=torch.bfloat16,
                        device=device,
                    )
                    * 0.5
                )
                tsc.write_single_stream_hidden(b, hid, 100_000_000)
            pt_rows.append(list(range(b * num_p, (b + 1) * num_p)))
        view = SemanticStateView(
            k_pool=tsc.k_pool if kv_mode else None,
            v_pool=tsc.v_pool,
            page_timestamps=tsc.page_timestamps,
            splitk_partial_out=tsc.splitk_partial_out(0) if kv_mode else None,
            splitk_partial_lse=tsc.splitk_partial_lse(0) if kv_mode else None,
            page_table=torch.tensor(pt_rows, dtype=torch.int32, device=device),
            seq_lens=torch.full((B,), channel.max_seq_len, dtype=torch.int32, device=device),
            t_now_ns=torch.full((B,), 100_000_000, dtype=torch.int64, device=device),
            relative_time_rope=channel.relative_time_rope,
            max_seq_len=channel.max_seq_len,
            mode=channel.mode,
        )
        dummy_obs = torch.randn((B, self.obs_shape[0]), dtype=torch.bfloat16, device=device) * 0.2
        dummy_pref = torch.zeros(
            (B, min(8, self.chunk_size - 1), self.action_dim), dtype=torch.bfloat16, device=device
        )
        calibrate_fp8_submodules(self, lambda: self.forward(dummy_obs, view, dummy_pref))

    def forward(
        self,
        fast_obs: torch.Tensor,
        state: SemanticStateView,
        frozen_prefix: torch.Tensor,
    ) -> torch.Tensor:
        B = fast_obs.size(0)
        P_len = frozen_prefix.size(1)
        x = self.query_embed.unsqueeze(0) + self.state_proj(fast_obs).unsqueeze(1)
        if P_len > 0:
            # Condition prefix tokens directly on in-flight frozen_prefix when active
            pref_cond = state.splice_prefix(torch.zeros_like(x[:, :P_len]), self.prefix_proj(frozen_prefix))
            x = torch.cat([x[:, :P_len] + pref_cond, x[:, P_len:]], dim=1)

        sem_hidden = state.read_hidden()[0] if self.mode == "hidden" else None
        for idx, blk in enumerate(self.blocks):
            h = blk["norm1"](x)
            q = blk["q_proj"](h).view(B, self.chunk_size, self.num_q_heads, self.head_dim)
            if self.mode == "kv":
                layer_id = (idx * self._expected_layers) // len(self.blocks)
                attn_out = state.cross_attend(q, layer_id)
            else:
                assert sem_hidden is not None
                S_hid = sem_hidden.size(1)
                kv = blk["kv_proj"](sem_hidden).view(B, S_hid, 2, self.num_q_heads, self.head_dim)
                hid_mask = (
                    torch.arange(S_hid, device=sem_hidden.device).view(1, 1, 1, S_hid)
                    < state.seq_lens.to(torch.int64).clamp(1, S_hid).view(B, 1, 1, 1)
                )
                attn_out = F.scaled_dot_product_attention(
                    q.transpose(1, 2),
                    kv[:, :, 0].transpose(1, 2),
                    kv[:, :, 1].transpose(1, 2),
                    attn_mask=hid_mask,
                ).transpose(1, 2)
            x = x + blk["o_proj"](attn_out.reshape(B, self.chunk_size, -1))
            h2 = F.silu(blk["fc1"](blk["norm2"](x)))
            x = x + blk["fc2"](h2)

        pred = self.out_proj(self.final_norm(x))
        if P_len > 0:
            return torch.cat([state.splice_prefix(pred[:, :P_len], frozen_prefix), pred[:, P_len:]], dim=1)
        return pred
