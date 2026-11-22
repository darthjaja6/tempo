from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..channel import SemanticStateView, SlowCaptureContext, TemporalChannel, mark_sliceable
from .base import DualSystemDeployment, FastHead, ModelRegistry, SlowBackbone, validate_fast_precision
from .hub import SafetensorLoader, load_hf_tokenizer
from .layers import (
    FP8Linear,
    RMSNorm,
    SigLIPVisionTower,
    apply_rope_1d,
    calibrate_fp8_submodules,
    capture_slow_calibration_sample,
)

_QWEN3_TOKENIZER_REPO = "Qwen/Qwen3-1.7B"


@dataclass
class GR00TConfig:
    """Configuration for NVIDIA Isaac GR00T-N1.5-3B (nvidia/GR00T-N1.5-3B, 3.0B)."""

    num_cameras: int = 2
    image_size: int = 224
    patch_size: int = 14
    vision_width: int = 1152
    vision_mlp_dim: int = 4304
    vision_layers: int = 27
    vision_heads: int = 16

    llm_width: int = 2048
    llm_mlp_dim: int = 6144
    llm_layers: int = 12
    llm_heads: int = 16
    llm_kv_heads: int = 8
    llm_head_dim: int = 128
    llm_rope_theta: float = 1000000.0
    vocab_size: int = 151680

    vl_sa_layers: int = 4
    vl_sa_heads: int = 32
    vl_sa_head_dim: int = 64
    vl_sa_mlp_dim: int = 8192

    dit_layers: int = 16
    dit_heads: int = 32
    dit_head_dim: int = 48
    dit_output_dim: int = 1024
    num_future_tokens: int = 32
    num_embodiments: int = 32
    embodiment_id: int = 0
    max_pos_embeds: int = 1024

    chunk_size: int = 16
    action_dim: int = 32
    state_dim: int = 64
    num_inference_steps: int = 4
    num_timestep_buckets: int = 1000
    prompt_len: int = 32
    fast_precision: str = "bf16"
    tokenizer_repo: str = _QWEN3_TOKENIZER_REPO

    def __post_init__(self) -> None:
        validate_fast_precision(self.fast_precision)

    @property
    def dit_dim(self) -> int:
        return self.dit_heads * self.dit_head_dim

    @property
    def patches_per_camera(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    @property
    def prefix_seq_len(self) -> int:
        return self.num_cameras * self.patches_per_camera + self.prompt_len

    @classmethod
    def from_dict(cls, d: Dict[str, Any], **overrides: Any) -> "GR00TConfig":
        ah = d.get("action_head_cfg", {})
        dm = ah.get("diffusion_model_cfg", {})
        vlsa = ah.get("vl_self_attention_cfg", {})
        bb = d.get("backbone_cfg", {})

        chunk = int(overrides.get("chunk_size", d.get("action_horizon", ah.get("action_horizon", 16))))

        kwargs: Dict[str, Any] = {
            "llm_layers": int(bb.get("select_layer", 12)),
            "vl_sa_layers": int(vlsa.get("num_layers", 4)),
            "vl_sa_heads": int(vlsa.get("num_attention_heads", 32)),
            "vl_sa_head_dim": int(vlsa.get("attention_head_dim", 64)),
            "dit_layers": int(dm.get("num_layers", 16)),
            "dit_heads": int(dm.get("num_attention_heads", 32)),
            "dit_head_dim": int(dm.get("attention_head_dim", 48)),
            "dit_output_dim": int(dm.get("output_dim", 1024)),
            "num_future_tokens": int(ah.get("num_target_vision_tokens", 32)),
            "chunk_size": chunk,
            "action_dim": int(d.get("action_dim", ah.get("action_dim", 32))),
            "state_dim": int(ah.get("max_state_dim", 64)),
            "num_inference_steps": int(ah.get("num_inference_timesteps", 4)),
            "num_timestep_buckets": int(ah.get("num_timestep_buckets", 1000)),
        }
        kwargs.update(overrides)
        return cls(**kwargs)


class _Qwen3SlowAttn(nn.Module):
    def __init__(self, cfg: GR00TConfig, device: torch.device) -> None:
        super().__init__()
        self.num_heads = cfg.llm_heads
        self.num_kv_heads = cfg.llm_kv_heads
        self.head_dim = cfg.llm_head_dim
        self.rope_theta = cfg.llm_rope_theta
        q_dim = self.num_heads * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim

        self.input_layernorm = RMSNorm(cfg.llm_width, eps=1e-6, device=device)
        self.q_proj = nn.Linear(cfg.llm_width, q_dim, bias=False, device=device, dtype=torch.bfloat16)
        self.k_proj = nn.Linear(cfg.llm_width, kv_dim, bias=False, device=device, dtype=torch.bfloat16)
        self.v_proj = nn.Linear(cfg.llm_width, kv_dim, bias=False, device=device, dtype=torch.bfloat16)
        self.o_proj = nn.Linear(q_dim, cfg.llm_width, bias=False, device=device, dtype=torch.bfloat16)
        self.q_norm = RMSNorm(self.head_dim, eps=1e-6, device=device)
        self.k_norm = RMSNorm(self.head_dim, eps=1e-6, device=device)
        self.requires_grad_(False)

    def forward(self, x: torch.Tensor, pos_ids: torch.Tensor) -> torch.Tensor:
        B, S, _ = x.shape
        h = self.input_layernorm(x)
        q = self.q_norm(self.q_proj(h).view(B, S, self.num_heads, self.head_dim))
        k = self.k_norm(self.k_proj(h).view(B, S, self.num_kv_heads, self.head_dim))
        v = self.v_proj(h).view(B, S, self.num_kv_heads, self.head_dim)
        q = apply_rope_1d(q, pos_ids, rope_theta=self.rope_theta)
        k = apply_rope_1d(k, pos_ids, rope_theta=self.rope_theta)
        groups = self.num_heads // self.num_kv_heads
        k_exp = k.transpose(1, 2).repeat_interleave(groups, dim=1)
        v_exp = v.transpose(1, 2).repeat_interleave(groups, dim=1)
        attn = F.scaled_dot_product_attention(q.transpose(1, 2), k_exp, v_exp, is_causal=True).transpose(1, 2)
        return x + self.o_proj(attn.reshape(B, S, self.num_heads * self.head_dim))


class _Qwen3SlowMLP(nn.Module):
    def __init__(self, cfg: GR00TConfig, device: torch.device) -> None:
        super().__init__()
        self.post_attention_layernorm = RMSNorm(cfg.llm_width, eps=1e-6, device=device)
        self.gate_proj = mark_sliceable(
            nn.Linear(cfg.llm_width, cfg.llm_mlp_dim, bias=False, device=device, dtype=torch.bfloat16)
        )
        self.up_proj = mark_sliceable(
            nn.Linear(cfg.llm_width, cfg.llm_mlp_dim, bias=False, device=device, dtype=torch.bfloat16)
        )
        self.down_proj = mark_sliceable(
            nn.Linear(cfg.llm_mlp_dim, cfg.llm_width, bias=False, device=device, dtype=torch.bfloat16)
        )
        self.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.post_attention_layernorm(x)
        return x + self.down_proj(F.silu(self.gate_proj(h)) * self.up_proj(h))


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, cfg: GR00TConfig, device: torch.device) -> None:
        super().__init__()
        self.attn = mark_sliceable(_Qwen3SlowAttn(cfg, device))
        self.mlp = mark_sliceable(_Qwen3SlowMLP(cfg, device))
        self.requires_grad_(False)

    def load_weights(self, loader: SafetensorLoader, prefix: str, device: torch.device) -> None:
        with torch.no_grad():
            self.attn.input_layernorm.weight.copy_(
                loader.tensor(f"{prefix}.input_layernorm.weight", device=device)
            )
            self.attn.q_norm.weight.copy_(loader.tensor(f"{prefix}.self_attn.q_norm.weight", device=device))
            self.attn.k_norm.weight.copy_(loader.tensor(f"{prefix}.self_attn.k_norm.weight", device=device))
            loader.load_linear(self.attn.q_proj, f"{prefix}.self_attn.q_proj", device=device)
            loader.load_linear(self.attn.k_proj, f"{prefix}.self_attn.k_proj", device=device)
            loader.load_linear(self.attn.v_proj, f"{prefix}.self_attn.v_proj", device=device)
            loader.load_linear(self.attn.o_proj, f"{prefix}.self_attn.o_proj", device=device)
            self.mlp.post_attention_layernorm.weight.copy_(
                loader.tensor(f"{prefix}.post_attention_layernorm.weight", device=device)
            )
            loader.load_linear(self.mlp.gate_proj, f"{prefix}.mlp.gate_proj", device=device)
            loader.load_linear(self.mlp.up_proj, f"{prefix}.mlp.up_proj", device=device)
            loader.load_linear(self.mlp.down_proj, f"{prefix}.mlp.down_proj", device=device)

    def forward(self, x: torch.Tensor, pos_ids: torch.Tensor) -> torch.Tensor:
        return self.mlp(self.attn(x, pos_ids))


class VLSelfAttnBlock(nn.Module):
    """4-layer VL refinement Transformer at the top of GR00T-N1.5's Slow encoder."""

    def __init__(self, cfg: GR00TConfig, device: torch.device) -> None:
        super().__init__()
        self.heads = cfg.vl_sa_heads
        self.head_dim = cfg.vl_sa_head_dim
        dim = self.heads * self.head_dim
        self.norm1 = nn.LayerNorm(dim, eps=1e-5, device=device, dtype=torch.bfloat16)
        self.to_q = nn.Linear(dim, dim, bias=True, device=device, dtype=torch.bfloat16)
        self.to_k = nn.Linear(dim, dim, bias=True, device=device, dtype=torch.bfloat16)
        self.to_v = nn.Linear(dim, dim, bias=True, device=device, dtype=torch.bfloat16)
        self.to_out = mark_sliceable(nn.Linear(dim, dim, bias=True, device=device, dtype=torch.bfloat16))
        self.norm3 = nn.LayerNorm(dim, eps=1e-5, device=device, dtype=torch.bfloat16)
        self.ff_in = mark_sliceable(
            nn.Linear(dim, cfg.vl_sa_mlp_dim, bias=True, device=device, dtype=torch.bfloat16)
        )
        self.ff_out = mark_sliceable(
            nn.Linear(cfg.vl_sa_mlp_dim, dim, bias=True, device=device, dtype=torch.bfloat16)
        )
        self.requires_grad_(False)

    def load_weights(self, loader: SafetensorLoader, prefix: str, device: torch.device) -> None:
        with torch.no_grad():
            self.norm1.weight.copy_(loader.tensor(f"{prefix}.norm1.weight", device=device))
            self.norm1.bias.copy_(loader.tensor(f"{prefix}.norm1.bias", device=device))
            self.norm3.weight.copy_(loader.tensor(f"{prefix}.norm3.weight", device=device))
            self.norm3.bias.copy_(loader.tensor(f"{prefix}.norm3.bias", device=device))
            loader.load_linear(self.to_q, f"{prefix}.attn1.to_q", device=device)
            loader.load_linear(self.to_k, f"{prefix}.attn1.to_k", device=device)
            loader.load_linear(self.to_v, f"{prefix}.attn1.to_v", device=device)
            loader.load_linear(self.to_out, f"{prefix}.attn1.to_out.0", device=device)
            loader.load_linear(self.ff_in, f"{prefix}.ff.net.0.proj", device=device)
            loader.load_linear(self.ff_out, f"{prefix}.ff.net.2", device=device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        h = self.norm1(x)
        q = self.to_q(h).view(B, S, self.heads, self.head_dim).transpose(1, 2)
        k = self.to_k(h).view(B, S, self.heads, self.head_dim).transpose(1, 2)
        v = self.to_v(h).view(B, S, self.heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, S, D)
        x = x + self.to_out(attn)
        h2 = F.gelu(self.ff_in(self.norm3(x)), approximate="tanh")
        return x + self.ff_out(h2)


class GR00TSlowStage(SlowBackbone):
    """Pipeline-capable System-2 Eagle2 (SigLIP2 + Qwen3 + VL-SelfAttn) stage for GR00T-N1.5."""

    def __init__(
        self,
        cfg: GR00TConfig,
        *,
        layer_start: int = 0,
        layer_end: Optional[int] = None,
        is_first_stage: bool = True,
        is_last_stage: bool = True,
        device: Union[str, torch.device] = "cuda:0",
    ) -> None:
        super().__init__()
        dev = torch.device(device)
        self.cfg = cfg
        self.layer_start = layer_start
        self.layer_end = cfg.llm_layers if layer_end is None else layer_end
        self.is_first_stage = is_first_stage
        self.is_last_stage = is_last_stage

        self.channel = TemporalChannel(
            hidden_dim=cfg.llm_width,
            seq_len=cfg.prefix_seq_len,
            mode="hidden",
        )
        self.obs_shape = (cfg.num_cameras, 3, cfg.image_size, cfg.image_size)

        if is_first_stage:
            self.vision_tower = SigLIPVisionTower(
                hidden_size=cfg.vision_width,
                intermediate_size=cfg.vision_mlp_dim,
                num_layers=cfg.vision_layers,
                num_heads=cfg.vision_heads,
                image_size=cfg.image_size,
                patch_size=cfg.patch_size,
                device=dev,
            )
            self.mlp1 = mark_sliceable(
                nn.Linear(cfg.vision_width, cfg.llm_width, bias=True, device=dev, dtype=torch.bfloat16)
            )
            self.register_buffer(
                "lang_embeds",
                torch.zeros((1, cfg.prompt_len, cfg.llm_width), dtype=torch.bfloat16, device=dev),
            )
        else:
            self.vision_tower = None
            self.mlp1 = None

        self.register_buffer("pos_ids", torch.arange(cfg.prefix_seq_len, dtype=torch.int64, device=dev))
        self.llm_layers = nn.ModuleList(
            [Qwen3DecoderLayer(cfg, dev) for _ in range(self.layer_start, self.layer_end)]
        )

        if is_last_stage:
            self.llm_norm = RMSNorm(cfg.llm_width, eps=1e-6, device=dev)
            self.vlln = nn.LayerNorm(cfg.llm_width, eps=1e-5, device=dev, dtype=torch.bfloat16)
            self.vl_blocks = nn.ModuleList(
                [mark_sliceable(VLSelfAttnBlock(cfg, dev)) for _ in range(cfg.vl_sa_layers)]
            )
        else:
            self.llm_norm = None
            self.vlln = None
            self.vl_blocks = None

        self._embed_weight_cpu: Optional[torch.Tensor] = None
        self._checkpoint_dir: Optional[Path] = None
        self._tokenizer: Any = None
        self.requires_grad_(False)

    def load_weights(
        self,
        loader: SafetensorLoader,
        device: Union[str, torch.device],
        *,
        embed_weight_cpu: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        dev = torch.device(device)
        self._checkpoint_dir = loader.ckpt_dir
        e_root = "backbone.eagle_model"
        if embed_weight_cpu is None:
            embed_weight_cpu = loader.tensor(
                f"{e_root}.language_model.model.embed_tokens.weight",
                device="cpu",
                dtype=torch.bfloat16,
            )

        if self.is_first_stage:
            assert self.vision_tower is not None and self.mlp1 is not None
            self.vision_tower.load_weights(loader, f"{e_root}.vision_model.vision_model", dev)
            loader.load_linear(self.mlp1, f"{e_root}.mlp1.0", device=dev)
            self._embed_weight_cpu = embed_weight_cpu

        for local_i, layer in enumerate(self.llm_layers):
            g_idx = self.layer_start + local_i
            layer.load_weights(loader, f"{e_root}.language_model.model.layers.{g_idx}", dev)

        if self.is_last_stage:
            assert self.llm_norm is not None and self.vlln is not None and self.vl_blocks is not None
            with torch.no_grad():
                self.llm_norm.weight.copy_(
                    loader.tensor(f"{e_root}.language_model.model.norm.weight", device=dev)
                )
                self.vlln.weight.copy_(loader.tensor("action_head.vlln.weight", device=dev))
                self.vlln.bias.copy_(loader.tensor("action_head.vlln.bias", device=dev))
            for idx, blk in enumerate(self.vl_blocks):
                blk.load_weights(loader, f"action_head.vl_self_attention.transformer_blocks.{idx}", dev)

        return embed_weight_cpu

    def _get_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = load_hf_tokenizer(self.cfg.tokenizer_repo, checkpoint_dir=self._checkpoint_dir)
        return self._tokenizer

    @torch.no_grad()
    def set_prompt(self, prompt: Union[str, Sequence[int], torch.Tensor], **_kwargs: Any) -> None:
        if not self.is_first_stage:
            return
        if isinstance(prompt, torch.Tensor) and prompt.is_floating_point():
            emb_in = prompt.squeeze(0) if prompt.ndim == 3 and prompt.size(0) == 1 else prompt
            if emb_in.ndim != 2 or emb_in.size(1) != self.cfg.llm_width:
                raise ValueError(
                    f"Floating-point prompt embedding must have shape [L, {self.cfg.llm_width}], "
                    f"got {tuple(prompt.shape)}"
                )
            emb = emb_in.to(device=self.lang_embeds.device, dtype=torch.bfloat16)
            if emb.size(0) >= self.cfg.prompt_len:
                emb = emb[: self.cfg.prompt_len]
            else:
                emb = F.pad(emb, (0, 0, 0, self.cfg.prompt_len - emb.size(0)), value=0.0)
            self.lang_embeds.copy_(emb.unsqueeze(0))
            return

        if self._embed_weight_cpu is None:
            raise RuntimeError(
                "GR00TSlowBackbone cannot embed text or token IDs without loaded vocabulary weights; "
                "load a checkpoint or pass 2D floating-point prompt embeddings [L, llm_width]"
            )

        if isinstance(prompt, str):
            tok = self._get_tokenizer()
            ids = tok.encode(prompt).ids
        elif isinstance(prompt, torch.Tensor):
            ids = [int(v) for v in prompt.flatten().tolist()]
        else:
            ids = [int(v) for v in prompt]

        eos_id = 151643
        if len(ids) >= self.cfg.prompt_len:
            ids = ids[: self.cfg.prompt_len]
        else:
            ids = ids + [eos_id] * (self.cfg.prompt_len - len(ids))

        idx_t = torch.tensor(ids, dtype=torch.long)
        emb = F.embedding(idx_t, self._embed_weight_cpu).to(
            device=self.lang_embeds.device, dtype=torch.bfloat16
        )
        self.lang_embeds.copy_(emb.unsqueeze(0))

    def encode_vl_hidden(self, obs_or_hidden: torch.Tensor) -> torch.Tensor:
        if self.is_first_stage:
            assert self.vision_tower is not None and self.mlp1 is not None
            B = obs_or_hidden.size(0)
            vis = self.vision_tower(obs_or_hidden)
            img_embs = self.mlp1(vis.reshape(B, -1, self.cfg.vision_width))
            x = torch.cat([img_embs, self.lang_embeds.expand(B, -1, -1)], dim=1)
        else:
            x = obs_or_hidden

        for layer in self.llm_layers:
            x = layer(x, self.pos_ids)

        if self.is_last_stage:
            assert self.llm_norm is not None and self.vlln is not None and self.vl_blocks is not None
            x = self.vlln(self.llm_norm(x))
            for blk in self.vl_blocks:
                x = blk(x)
        return x

    def forward(self, obs_or_hidden: torch.Tensor, ctx: SlowCaptureContext) -> torch.Tensor:
        x = self.encode_vl_hidden(obs_or_hidden)
        if self.is_last_stage:
            ctx.write_hidden_state(x)
        return x

    def split_pipeline(self, stage_devices: Sequence[int]) -> List[nn.Module]:
        P = len(stage_devices)
        if P == 1:
            return [self.to(device=f"cuda:{int(stage_devices[0])}", dtype=torch.bfloat16)]
        if not (self.is_first_stage and self.is_last_stage):
            raise RuntimeError("split_pipeline requires a full unsplit GR00TSlowBackbone")
        L = len(self.llm_layers)
        if P > L:
            raise ValueError(f"Cannot split {L} Qwen3 layers across {P} pipeline stages")

        stages: List[nn.Module] = []
        for p in range(P):
            lo = (p * L) // P
            hi = ((p + 1) * L) // P
            dev = torch.device(f"cuda:{int(stage_devices[p])}")
            stg = GR00TSlowStage(
                self.cfg,
                layer_start=lo,
                layer_end=hi,
                is_first_stage=(p == 0),
                is_last_stage=(p == P - 1),
                device=dev,
            )
            with torch.no_grad():
                if p == 0:
                    assert stg.vision_tower is not None and self.vision_tower is not None
                    assert stg.mlp1 is not None and self.mlp1 is not None
                    stg.vision_tower.load_state_dict(self.vision_tower.state_dict())
                    stg.mlp1.load_state_dict(self.mlp1.state_dict())
                    stg.lang_embeds.copy_(self.lang_embeds.to(dev))
                    stg._embed_weight_cpu = self._embed_weight_cpu
                    stg._checkpoint_dir = self._checkpoint_dir
                    stg._tokenizer = self._tokenizer
                for idx_local in range(hi - lo):
                    stg.llm_layers[idx_local].load_state_dict(self.llm_layers[lo + idx_local].state_dict())
                if p == P - 1:
                    assert stg.llm_norm is not None and self.llm_norm is not None
                    assert stg.vlln is not None and self.vlln is not None
                    assert stg.vl_blocks is not None and self.vl_blocks is not None
                    stg.llm_norm.load_state_dict(self.llm_norm.state_dict())
                    stg.vlln.load_state_dict(self.vlln.state_dict())
                    stg.vl_blocks.load_state_dict(self.vl_blocks.state_dict())
            stages.append(stg)
        return stages


GR00TSlowBackbone = GR00TSlowStage


class CategoryLinear(nn.Module):
    def __init__(self, num_cats: int, in_dim: int, out_dim: int, device: torch.device) -> None:
        super().__init__()
        self.W = nn.Parameter(
            torch.randn((num_cats, in_dim, out_dim), dtype=torch.bfloat16, device=device) * 0.02,
            requires_grad=False,
        )
        self.b = nn.Parameter(
            torch.zeros((num_cats, out_dim), dtype=torch.bfloat16, device=device),
            requires_grad=False,
        )

    def load_weights(self, loader: SafetensorLoader, prefix: str, device: torch.device) -> None:
        with torch.no_grad():
            self.W.copy_(loader.tensor(f"{prefix}.W", device=device))
            self.b.copy_(loader.tensor(f"{prefix}.b", device=device))

    def forward(self, x: torch.Tensor, cat_id: int) -> torch.Tensor:
        return torch.matmul(x, self.W[cat_id]) + self.b[cat_id]


class GR00TDiTBlock(nn.Module):
    def __init__(
        self,
        cfg: GR00TConfig,
        *,
        is_cross_attn: bool,
        use_fp8: bool,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.heads = cfg.dit_heads
        self.head_dim = cfg.dit_head_dim
        self.dim = cfg.dit_dim
        self.is_cross_attn = is_cross_attn
        kv_in_dim = cfg.llm_width if is_cross_attn else self.dim

        self.ada_linear = nn.Linear(self.dim, self.dim * 2, bias=True, device=device, dtype=torch.bfloat16)
        self.norm1 = nn.LayerNorm(self.dim, elementwise_affine=False, eps=1e-5)
        self.to_q = FP8Linear(self.dim, self.dim, bias=True, use_fp8=use_fp8, device=device)
        self.to_k = FP8Linear(kv_in_dim, self.dim, bias=True, use_fp8=use_fp8, device=device)
        self.to_v = FP8Linear(kv_in_dim, self.dim, bias=True, use_fp8=use_fp8, device=device)
        self.to_out = FP8Linear(self.dim, self.dim, bias=True, use_fp8=use_fp8, device=device)

        self.norm3 = nn.LayerNorm(self.dim, elementwise_affine=False, eps=1e-5)
        self.ff_in = FP8Linear(self.dim, self.dim * 4, bias=True, use_fp8=use_fp8, device=device)
        self.ff_out = FP8Linear(self.dim * 4, self.dim, bias=True, use_fp8=use_fp8, device=device)
        self.requires_grad_(False)

    def load_weights(self, loader: SafetensorLoader, prefix: str, device: torch.device) -> None:
        loader.load_linear(self.ada_linear, f"{prefix}.norm1.linear", device=device)
        self.to_q.set_weight(
            loader.tensor(f"{prefix}.attn1.to_q.weight", device=device),
            loader.tensor(f"{prefix}.attn1.to_q.bias", device=device),
        )
        self.to_k.set_weight(
            loader.tensor(f"{prefix}.attn1.to_k.weight", device=device),
            loader.tensor(f"{prefix}.attn1.to_k.bias", device=device),
        )
        self.to_v.set_weight(
            loader.tensor(f"{prefix}.attn1.to_v.weight", device=device),
            loader.tensor(f"{prefix}.attn1.to_v.bias", device=device),
        )
        self.to_out.set_weight(
            loader.tensor(f"{prefix}.attn1.to_out.0.weight", device=device),
            loader.tensor(f"{prefix}.attn1.to_out.0.bias", device=device),
        )
        self.ff_in.set_weight(
            loader.tensor(f"{prefix}.ff.net.0.proj.weight", device=device),
            loader.tensor(f"{prefix}.ff.net.0.proj.bias", device=device),
        )
        self.ff_out.set_weight(
            loader.tensor(f"{prefix}.ff.net.2.weight", device=device),
            loader.tensor(f"{prefix}.ff.net.2.bias", device=device),
        )

    def project_cross_kv(self, vl_embs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, _ = vl_embs.shape
        k = self.to_k(vl_embs).view(B, S, self.heads, self.head_dim).transpose(1, 2)
        v = self.to_v(vl_embs).view(B, S, self.heads, self.head_dim).transpose(1, 2)
        return k, v

    def forward(
        self,
        x: torch.Tensor,
        temb: torch.Tensor,
        cross_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        B, T, D = x.shape
        scale, shift = self.ada_linear(F.silu(temb)).chunk(2, dim=-1)
        h = (self.norm1(x.float()) * (1.0 + scale.float().unsqueeze(1)) + shift.float().unsqueeze(1)).to(
            torch.bfloat16
        )
        q = self.to_q(h).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        if self.is_cross_attn:
            assert cross_kv is not None
            k, v = cross_kv
        else:
            k = self.to_k(h).view(B, T, self.heads, self.head_dim).transpose(1, 2)
            v = self.to_v(h).view(B, T, self.heads, self.head_dim).transpose(1, 2)
        attn = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, T, D)
        x = x + self.to_out(attn)
        h2 = self.norm3(x.float()).to(torch.bfloat16)
        x = x + self.ff_out(F.gelu(self.ff_in(h2), approximate="tanh"))
        return x


class GR00TFastHead(FastHead):
    """System-1 16-Layer Cross-Attention DiT Action Head for NVIDIA GR00T-N1.5-3B."""

    def __init__(
        self,
        cfg: GR00TConfig,
        *,
        device: Union[str, torch.device] = "cuda:0",
    ) -> None:
        super().__init__()
        dev = torch.device(device)
        self.cfg = cfg
        self.action_dim = cfg.action_dim
        self.chunk_size = cfg.chunk_size
        self.obs_shape = (cfg.state_dim,)
        self.max_q_len = max(64, 1 + cfg.num_future_tokens + cfg.chunk_size)
        self.max_q_heads = cfg.dit_heads
        self.embodiment_id = cfg.embodiment_id

        validate_fast_precision(cfg.fast_precision)
        use_fp8 = cfg.fast_precision == "fp8"
        dim = cfg.dit_dim  # 1536
        hid = cfg.dit_output_dim  # 1024

        self.state_fc1 = CategoryLinear(cfg.num_embodiments, cfg.state_dim, hid, dev)
        self.state_fc2 = CategoryLinear(cfg.num_embodiments, hid, dim, dev)

        self.act_W1 = CategoryLinear(cfg.num_embodiments, cfg.action_dim, dim, dev)
        self.act_W2 = CategoryLinear(cfg.num_embodiments, dim * 2, dim, dev)
        self.act_W3 = CategoryLinear(cfg.num_embodiments, dim, dim, dev)

        self.dec_fc1 = CategoryLinear(cfg.num_embodiments, hid, hid, dev)
        self.dec_fc2 = CategoryLinear(cfg.num_embodiments, hid, cfg.action_dim, dev)

        self.future_tokens = nn.Embedding(cfg.num_future_tokens, dim, device=dev, dtype=torch.bfloat16)
        self.position_embedding = nn.Embedding(cfg.max_pos_embeds, dim, device=dev, dtype=torch.bfloat16)

        self.time_fc1 = nn.Linear(256, dim, bias=True, device=dev, dtype=torch.bfloat16)
        self.time_fc2 = nn.Linear(dim, dim, bias=True, device=dev, dtype=torch.bfloat16)

        self.blocks = nn.ModuleList(
            [
                GR00TDiTBlock(cfg, is_cross_attn=(idx % 2 == 0), use_fp8=use_fp8, device=dev)
                for idx in range(cfg.dit_layers)
            ]
        )
        self.norm_out = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_1 = nn.Linear(dim, dim * 2, bias=True, device=dev, dtype=torch.bfloat16)
        self.proj_out_2 = nn.Linear(dim, hid, bias=True, device=dev, dtype=torch.bfloat16)

        # Precomputed frequency vectors for action_encoder and diffusers Timesteps(256, flip_sin_to_cos=True, downscale_freq_shift=1)
        act_half = dim // 2
        act_exp = -torch.arange(act_half, dtype=torch.float32, device=dev) * (math.log(10000.0) / act_half)
        self.register_buffer("act_pos_freqs", torch.exp(act_exp))

        t_exp = -math.log(10000.0) * torch.arange(128, dtype=torch.float32, device=dev) / 127.0
        self.register_buffer("dit_time_freqs", torch.exp(t_exp))
        self.register_buffer("action_pos_ids", torch.arange(cfg.chunk_size, dtype=torch.long, device=dev))

        gen = torch.Generator().manual_seed(202615)
        init_eps = torch.randn((1, cfg.chunk_size, cfg.action_dim), generator=gen, dtype=torch.bfloat16)
        self.register_buffer("base_noise", init_eps.to(device=dev))
        self.requires_grad_(False)

    def load_weights(self, loader: SafetensorLoader, device: Union[str, torch.device]) -> None:
        dev = torch.device(device)
        self.state_fc1.load_weights(loader, "action_head.state_encoder.layer1", dev)
        self.state_fc2.load_weights(loader, "action_head.state_encoder.layer2", dev)
        self.act_W1.load_weights(loader, "action_head.action_encoder.W1", dev)
        self.act_W2.load_weights(loader, "action_head.action_encoder.W2", dev)
        self.act_W3.load_weights(loader, "action_head.action_encoder.W3", dev)
        self.dec_fc1.load_weights(loader, "action_head.action_decoder.layer1", dev)
        self.dec_fc2.load_weights(loader, "action_head.action_decoder.layer2", dev)

        with torch.no_grad():
            self.future_tokens.weight.copy_(loader.tensor("action_head.future_tokens.weight", device=dev))
            self.position_embedding.weight.copy_(
                loader.tensor("action_head.position_embedding.weight", device=dev)
            )
        loader.load_linear(
            self.time_fc1, "action_head.model.timestep_encoder.timestep_embedder.linear_1", device=dev
        )
        loader.load_linear(
            self.time_fc2, "action_head.model.timestep_encoder.timestep_embedder.linear_2", device=dev
        )
        for idx, blk in enumerate(self.blocks):
            blk.load_weights(loader, f"action_head.model.transformer_blocks.{idx}", dev)
        loader.load_linear(self.proj_out_1, "action_head.model.proj_out_1", device=dev)
        loader.load_linear(self.proj_out_2, "action_head.model.proj_out_2", device=dev)

    def validate_channel(self, channel: TemporalChannel) -> None:
        if channel.mode != "hidden":
            raise ValueError(f"GR00TFastHead requires TemporalChannel(mode='hidden'), got '{channel.mode}'")
        if channel.feature_dim != self.cfg.llm_width:
            raise ValueError(
                f"GR00TFastHead expects hidden_dim={self.cfg.llm_width}, got {channel.feature_dim}"
            )

    def calibrate_fp8(
        self,
        channel: TemporalChannel,
        device: torch.device,
        sample_hidden: Optional[torch.Tensor] = None,
    ) -> None:
        if self.cfg.fast_precision != "fp8":
            return
        B = 4
        dummy_obs = torch.randn((B, self.cfg.state_dim), dtype=torch.bfloat16, device=device) * 0.2
        if sample_hidden is not None:
            reps = (B + sample_hidden.size(0) - 1) // sample_hidden.size(0)
            dummy_vl = sample_hidden.repeat(reps, 1, 1)[:B].to(device=device, dtype=torch.bfloat16).contiguous()
        else:
            dummy_vl = (
                torch.randn((B, channel.max_seq_len, channel.feature_dim), dtype=torch.bfloat16, device=device) * 0.5
            )
        dummy_noise = self.base_noise.expand(B, -1, -1)
        dummy_pref = torch.zeros((B, min(8, self.chunk_size - 1), self.action_dim), dtype=torch.bfloat16, device=device)
        calibrate_fp8_submodules(
            self, lambda: self.forward_with_noise(dummy_obs, dummy_vl, dummy_noise, frozen_prefix=dummy_pref)
        )

    def _encode_dit_timestep(self, t_disc: int, batch_size: int, device: torch.device) -> torch.Tensor:
        angles = float(t_disc) * self.dit_time_freqs
        proj = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1).to(torch.bfloat16)
        temb = self.time_fc2(F.silu(self.time_fc1(proj.unsqueeze(0))))
        return temb.expand(batch_size, -1)

    def _encode_actions(self, actions: torch.Tensor, t_disc: int) -> torch.Tensor:
        B, T, _ = actions.shape
        cat = self.embodiment_id
        a_emb = self.act_W1(actions, cat)
        freqs = float(t_disc) * self.act_pos_freqs
        tau = torch.cat([torch.sin(freqs), torch.cos(freqs)], dim=-1).to(torch.bfloat16)
        tau_exp = tau.view(1, 1, -1).expand(B, T, -1)
        x = F.silu(self.act_W2(torch.cat([a_emb, tau_exp], dim=-1), cat))
        return self.act_W3(x, cat) + self.position_embedding(self.action_pos_ids).unsqueeze(0)

    def forward_with_noise(
        self,
        fast_obs: torch.Tensor,
        vl_embs: torch.Tensor,
        noise: torch.Tensor,
        *,
        frozen_prefix: Optional[torch.Tensor] = None,
        prefix_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = fast_obs.size(0)
        cat = self.embodiment_id
        state_tok = self.state_fc2(F.relu(self.state_fc1(fast_obs.unsqueeze(1), cat)), cat)
        fut_tok = self.future_tokens.weight.unsqueeze(0).expand(B, -1, -1)

        # Pre-project static Slow VL hidden features once across all 4 Euler steps
        cross_kvs: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [
            blk.project_cross_kv(vl_embs) if blk.is_cross_attn else None for blk in self.blocks
        ]

        actions = noise.to(torch.bfloat16).clone()
        num_steps = self.cfg.num_inference_steps
        dt = 1.0 / float(num_steps)
        P_len = frozen_prefix.size(1) if frozen_prefix is not None else 0

        for step in range(num_steps):
            t_disc = int((step / float(num_steps)) * self.cfg.num_timestep_buckets)
            temb = self._encode_dit_timestep(t_disc, B, actions.device)
            act_feats = self._encode_actions(actions, t_disc)
            h = torch.cat([state_tok, fut_tok, act_feats], dim=1)

            for idx, blk in enumerate(self.blocks):
                h = blk(h, temb, cross_kvs[idx])

            shift, scale = self.proj_out_1(F.silu(temb)).chunk(2, dim=-1)
            h_norm = (
                self.norm_out(h.float()) * (1.0 + scale.float().unsqueeze(1)) + shift.float().unsqueeze(1)
            ).to(torch.bfloat16)
            out_hid = self.proj_out_2(h_norm[:, -self.chunk_size :, :])
            vel = self.dec_fc2(F.relu(self.dec_fc1(out_hid, cat)), cat)
            actions = actions + dt * vel

            if frozen_prefix is not None and P_len > 0:
                t_next = (step + 1) / float(num_steps)
                inpaint = ((1.0 - t_next) * noise[:, :P_len].float() + t_next * frozen_prefix.float()).to(
                    torch.bfloat16
                )
                if prefix_valid is not None:
                    inpaint = torch.where(prefix_valid.view(-1, 1, 1), inpaint, actions[:, :P_len])
                actions = torch.cat([inpaint, actions[:, P_len:]], dim=1)

        return actions

    def forward(
        self,
        fast_obs: torch.Tensor,
        state: SemanticStateView,
        frozen_prefix: torch.Tensor,
    ) -> torch.Tensor:
        if fast_obs.ndim != 2 or fast_obs.size(1) != self.cfg.state_dim:
            raise ValueError(
                f"GR00TFastHead expects fast_obs of shape (B, {self.cfg.state_dim}), got {tuple(fast_obs.shape)}"
            )
        vl_embs, _delta_sec = state.read_hidden()
        noise = self.base_noise.expand(fast_obs.size(0), -1, -1)
        return self.forward_with_noise(
            fast_obs, vl_embs, noise, frozen_prefix=frozen_prefix, prefix_valid=state.prefix_valid
        )


@ModelRegistry.register("gr00t_n1_5", "GR00T_N1_5", "gr00t")
def build_gr00t(
    config: Dict[str, Any],
    checkpoint_dir: Optional[Path],
    *,
    slow_devices: Sequence[Sequence[int]],
    fast_devices: Sequence[int],
    fast_precision: str = "bf16",
    prompt: Optional[Union[str, Sequence[int], torch.Tensor]] = None,
    **model_kwargs: Any,
) -> DualSystemDeployment:
    validate_fast_precision(fast_precision)
    tok_repo = str(model_kwargs.get("tokenizer_repo", _QWEN3_TOKENIZER_REPO))
    if prompt is not None and "prompt_len" not in model_kwargs:
        if isinstance(prompt, str):
            tok = load_hf_tokenizer(tok_repo, checkpoint_dir=checkpoint_dir)
            req_len = len(tok.encode(prompt).ids)
        elif isinstance(prompt, torch.Tensor) and prompt.is_floating_point():
            req_len = int(prompt.size(-2))
        else:
            seq = prompt.flatten().tolist() if isinstance(prompt, torch.Tensor) else list(prompt)
            req_len = len(seq)
        model_kwargs["prompt_len"] = max(GR00TConfig.prompt_len, req_len)

    cfg = GR00TConfig.from_dict(config, fast_precision=fast_precision, **model_kwargs)
    P = len(slow_devices[0])
    L = cfg.llm_layers
    if P > L:
        raise ValueError(f"Cannot split {L} GR00T Qwen3 layers across {P} pipeline stages")

    eff_prompt = prompt if prompt is not None else "manipulate the object"
    slow_replicas: List[List[nn.Module]] = []
    fast_shards: List[FastHead] = []

    if checkpoint_dir is not None:
        with SafetensorLoader(checkpoint_dir) as loader:
            embed_cpu: Optional[torch.Tensor] = None
            for pipe_devs in slow_devices:
                stages: List[nn.Module] = []
                for p, dev_id in enumerate(pipe_devs):
                    lo = (p * L) // P
                    hi = ((p + 1) * L) // P
                    stg = GR00TSlowStage(
                        cfg,
                        layer_start=lo,
                        layer_end=hi,
                        is_first_stage=(p == 0),
                        is_last_stage=(p == P - 1),
                        device=f"cuda:{int(dev_id)}",
                    )
                    embed_cpu = stg.load_weights(loader, f"cuda:{int(dev_id)}", embed_weight_cpu=embed_cpu)
                    if p == 0:
                        stg.set_prompt(eff_prompt)
                    stg.eval()
                    stages.append(stg)
                slow_replicas.append(stages)

            ch_cal = slow_replicas[0][0].channel  # type: ignore[attr-defined]
            sample_hidden = (
                capture_slow_calibration_sample(
                    slow_replicas[0], ch_cal, (cfg.num_cameras, 3, cfg.image_size, cfg.image_size)
                )[1]
                if fast_precision == "fp8"
                else None
            )
            for dev_id in fast_devices:
                f_dev = torch.device(f"cuda:{int(dev_id)}")
                head = GR00TFastHead(cfg, device=f_dev)
                head.load_weights(loader, f_dev)
                head.eval()
                head.calibrate_fp8(ch_cal, f_dev, sample_hidden=sample_hidden)
                fast_shards.append(head)

            loader.verify_all_consumed(
                ignore_keys=("backbone.eagle_model.language_model.lm_head.weight",),
                ignore_prefixes=("backbone.eagle_model.vision_model.vision_model.head.",),
            )
    else:
        for pipe_devs in slow_devices:
            stages = []
            for p, dev_id in enumerate(pipe_devs):
                lo = (p * L) // P
                hi = ((p + 1) * L) // P
                stg = GR00TSlowStage(
                    cfg,
                    layer_start=lo,
                    layer_end=hi,
                    is_first_stage=(p == 0),
                    is_last_stage=(p == P - 1),
                    device=f"cuda:{int(dev_id)}",
                )
                if p == 0 and prompt is not None:
                    stg.set_prompt(prompt)
                stg.eval()
                stages.append(stg)
            slow_replicas.append(stages)

        ch_cal = slow_replicas[0][0].channel  # type: ignore[attr-defined]
        sample_hidden = (
            capture_slow_calibration_sample(
                slow_replicas[0], ch_cal, (cfg.num_cameras, 3, cfg.image_size, cfg.image_size)
            )[1]
            if fast_precision == "fp8"
            else None
        )
        for dev_id in fast_devices:
            f_dev = torch.device(f"cuda:{int(dev_id)}")
            head = GR00TFastHead(cfg, device=f_dev)
            head.eval()
            head.calibrate_fp8(ch_cal, f_dev, sample_hidden=sample_hidden)
            fast_shards.append(head)

    ch = slow_replicas[0][0].channel  # type: ignore[attr-defined]
    return DualSystemDeployment(
        channel=ch,
        slow_replicas=slow_replicas,
        fast_shards=fast_shards,
        action_dim=cfg.action_dim,
        chunk_size=cfg.chunk_size,
        slow_obs_shape=(cfg.num_cameras, 3, cfg.image_size, cfg.image_size),
        fast_obs_shape=(cfg.state_dim,),
        step_dt_ms=fast_shards[0].resolve_step_dt_ms(),
        max_q_len=max(64, 1 + cfg.num_future_tokens + cfg.chunk_size),
        max_q_heads=cfg.dit_heads,
    )
