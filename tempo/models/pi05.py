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
    AdaRMSNorm,
    FP8Linear,
    GemmaRMSNorm,
    SigLIPVisionTower,
    apply_rope_1d,
    calibrate_fp8_submodules,
    capture_slow_calibration_sample,
    merge_attention_lse,
    sinusoidal_time_embedding,
    suffix_self_attention_with_lse,
)

# Public ungated HF mirror hosting the exact 257,152-vocab PaliGemma SentencePiece tokenizer.json
_PALIGEMMA_TOKENIZER_REPO = "mlx-community/paligemma-3b-mix-224-8bit"


def discretize_pi05_state(state: Union[Sequence[float], torch.Tensor], max_state_dim: int = 32) -> str:
    """Discretizes normalized [-1, 1] robot state into 256 bins per OpenPI / LeRobot pi0.5."""
    st = torch.as_tensor(state, dtype=torch.float32).flatten().cpu()
    if st.numel() > max_state_dim:
        st = st[:max_state_dim]
    elif st.numel() < max_state_dim:
        st = F.pad(st, (0, max_state_dim - st.numel()), value=0.0)
    edges = torch.linspace(-1.0, 1.0, 257, dtype=torch.float32)[:-1]
    bins = (torch.bucketize(st, edges, right=True) - 1).clamp(0, 255).tolist()
    return " ".join(str(int(b)) for b in bins)


@dataclass
class Pi05Config:
    """Configuration for Physical Intelligence pi0.5 (lerobot/pi05_base, 3.3B)."""

    num_cameras: int = 3
    image_size: int = 224
    patch_size: int = 14
    vision_width: int = 1152
    vision_mlp_dim: int = 4304
    vision_layers: int = 27
    vision_heads: int = 16

    vlm_width: int = 2048
    vlm_mlp_dim: int = 16384
    num_layers: int = 18
    num_heads: int = 8
    num_kv_heads: int = 1
    head_dim: int = 256
    vocab_size: int = 257152

    expert_width: int = 1024
    expert_mlp_dim: int = 4096

    chunk_size: int = 50
    action_dim: int = 32
    max_action_dim: int = 32
    max_state_dim: int = 32
    num_inference_steps: int = 10
    min_period: float = 0.004
    max_period: float = 4.0
    prompt_len: int = 96
    fast_precision: str = "bf16"
    tokenizer_repo: str = _PALIGEMMA_TOKENIZER_REPO

    def __post_init__(self) -> None:
        validate_fast_precision(self.fast_precision)

    @property
    def patches_per_camera(self) -> int:
        return (self.image_size // self.patch_size) ** 2

    @property
    def prefix_seq_len(self) -> int:
        return self.num_cameras * self.patches_per_camera + self.prompt_len

    @classmethod
    def from_dict(cls, d: Dict[str, Any], **overrides: Any) -> "Pi05Config":
        in_feats = d.get("input_features", {})
        vis_keys = [
            k
            for k, v in in_feats.items()
            if isinstance(v, dict) and v.get("type") == "VISUAL" and "empty_camera" not in k
        ]
        num_cams = len(vis_keys) if vis_keys else 3
        img_res = d.get("image_resolution", [224, 224])
        img_size = int(img_res[0]) if isinstance(img_res, (list, tuple)) and img_res else 224

        out_feats = d.get("output_features", {})
        act_feat = out_feats.get("action", {})
        max_act = int(d.get("max_action_dim", 32))
        act_dim = int(act_feat["shape"][0]) if isinstance(act_feat, dict) and "shape" in act_feat else max_act

        max_st = int(d.get("max_state_dim", 32))

        kwargs: Dict[str, Any] = {
            "num_cameras": num_cams,
            "image_size": img_size,
            "chunk_size": int(d.get("chunk_size", 50)),
            "action_dim": act_dim,
            "max_action_dim": max_act,
            "max_state_dim": max_st,
            "num_inference_steps": int(d.get("num_inference_steps", 10)),
            "min_period": float(d.get("min_period", 0.004)),
            "max_period": float(d.get("max_period", 4.0)),
        }
        kwargs.update(overrides)
        return cls(**kwargs)


class _GemmaSlowAttn(nn.Module):
    def __init__(self, cfg: Pi05Config, device: torch.device) -> None:
        super().__init__()
        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.head_dim = cfg.head_dim
        self.width = cfg.vlm_width
        self.norm = GemmaRMSNorm(cfg.vlm_width, device=device)
        self.q_proj = nn.Linear(cfg.vlm_width, cfg.num_heads * cfg.head_dim, bias=False, device=device, dtype=torch.bfloat16)
        self.k_proj = nn.Linear(cfg.vlm_width, cfg.num_kv_heads * cfg.head_dim, bias=False, device=device, dtype=torch.bfloat16)
        self.v_proj = nn.Linear(cfg.vlm_width, cfg.num_kv_heads * cfg.head_dim, bias=False, device=device, dtype=torch.bfloat16)
        self.o_proj = nn.Linear(cfg.num_heads * cfg.head_dim, cfg.vlm_width, bias=False, device=device, dtype=torch.bfloat16)
        self.requires_grad_(False)

    def forward(
        self, x: torch.Tensor, pos_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, S, _ = x.shape
        h = self.norm(x)
        q = self.q_proj(h).view(B, S, self.num_heads, self.head_dim)
        k = self.k_proj(h).view(B, S, self.num_kv_heads, self.head_dim)
        v = self.v_proj(h).view(B, S, self.num_kv_heads, self.head_dim)
        q = apply_rope_1d(q, pos_ids)
        k = apply_rope_1d(k, pos_ids)
        groups = self.num_heads // self.num_kv_heads
        k_exp = k.transpose(1, 2).repeat_interleave(groups, dim=1)
        v_exp = v.transpose(1, 2).repeat_interleave(groups, dim=1)
        attn = F.scaled_dot_product_attention(q.transpose(1, 2), k_exp, v_exp).transpose(1, 2)
        out = x + self.o_proj(attn.reshape(B, S, self.num_heads * self.head_dim))
        return out, k, v


class _GemmaSlowMLPUp(nn.Module):
    def __init__(self, cfg: Pi05Config, device: torch.device) -> None:
        super().__init__()
        self.norm = GemmaRMSNorm(cfg.vlm_width, device=device)
        self.gate_proj = mark_sliceable(
            nn.Linear(cfg.vlm_width, cfg.vlm_mlp_dim, bias=False, device=device, dtype=torch.bfloat16)
        )
        self.up_proj = mark_sliceable(
            nn.Linear(cfg.vlm_width, cfg.vlm_mlp_dim, bias=False, device=device, dtype=torch.bfloat16)
        )
        self.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        return F.gelu(self.gate_proj(h), approximate="tanh") * self.up_proj(h)


class Pi05GemmaLayer(nn.Module):
    def __init__(self, cfg: Pi05Config, device: torch.device) -> None:
        super().__init__()
        self.attn = mark_sliceable(_GemmaSlowAttn(cfg, device))
        self.mlp_up = mark_sliceable(_GemmaSlowMLPUp(cfg, device))
        self.down_proj = mark_sliceable(
            nn.Linear(cfg.vlm_mlp_dim, cfg.vlm_width, bias=False, device=device, dtype=torch.bfloat16)
        )
        self.requires_grad_(False)

    def load_weights(self, loader: SafetensorLoader, prefix: str, device: torch.device) -> None:
        with torch.no_grad():
            self.attn.norm.weight.copy_(loader.tensor(f"{prefix}.input_layernorm.weight", device=device))
            loader.load_linear(self.attn.q_proj, f"{prefix}.self_attn.q_proj", device=device)
            loader.load_linear(self.attn.k_proj, f"{prefix}.self_attn.k_proj", device=device)
            loader.load_linear(self.attn.v_proj, f"{prefix}.self_attn.v_proj", device=device)
            loader.load_linear(self.attn.o_proj, f"{prefix}.self_attn.o_proj", device=device)
            self.mlp_up.norm.weight.copy_(
                loader.tensor(f"{prefix}.post_attention_layernorm.weight", device=device)
            )
            loader.load_linear(self.mlp_up.gate_proj, f"{prefix}.mlp.gate_proj", device=device)
            loader.load_linear(self.mlp_up.up_proj, f"{prefix}.mlp.up_proj", device=device)
            loader.load_linear(self.down_proj, f"{prefix}.mlp.down_proj", device=device)

    def forward(self, x: torch.Tensor, pos_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, k, v = self.attn(x, pos_ids)
        x = x + self.down_proj(self.mlp_up(x))
        return x, k, v


class Pi05SlowStage(SlowBackbone):
    """Pipeline-capable System-2 PaliGemma VLM stage for pi0.5."""

    def __init__(
        self,
        cfg: Pi05Config,
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
        self.layer_end = cfg.num_layers if layer_end is None else layer_end
        self.is_first_stage = is_first_stage
        self.is_last_stage = is_last_stage

        self.channel = TemporalChannel(
            layers=cfg.num_layers,
            num_kv_heads=cfg.num_kv_heads,
            head_dim=cfg.head_dim,
            seq_len=cfg.prefix_seq_len,
            spatial_rope=False,
            relative_time_rope=False,
            mode="kv",
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
            self.mm_projector = mark_sliceable(
                nn.Linear(cfg.vision_width, cfg.vlm_width, bias=True, device=dev, dtype=torch.bfloat16)
            )
            # Pre-embedded scaled language prompt buffer [1, prompt_len, vlm_width]
            self.register_buffer(
                "lang_embeds",
                torch.zeros((1, cfg.prompt_len, cfg.vlm_width), dtype=torch.bfloat16, device=dev),
            )
        else:
            self.vision_tower = None
            self.mm_projector = None

        self.register_buffer(
            "pos_ids",
            torch.arange(cfg.prefix_seq_len, dtype=torch.int64, device=dev),
        )
        self.layers = nn.ModuleList(
            [Pi05GemmaLayer(cfg, dev) for _ in range(self.layer_start, self.layer_end)]
        )
        self.final_norm = GemmaRMSNorm(cfg.vlm_width, device=dev) if is_last_stage else None
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
        m_prefix = "model." if loader.has("model.action_in_proj.weight") else ""
        p_root = f"{m_prefix}paligemma_with_expert.paligemma"
        if embed_weight_cpu is None:
            embed_weight_cpu = loader.tensor(f"{p_root}.lm_head.weight", device="cpu", dtype=torch.bfloat16)

        if self.is_first_stage:
            assert self.vision_tower is not None and self.mm_projector is not None
            self.vision_tower.load_weights(loader, f"{p_root}.model.vision_tower.vision_model", dev)
            loader.load_linear(self.mm_projector, f"{p_root}.model.multi_modal_projector.linear", device=dev)
            self._embed_weight_cpu = embed_weight_cpu

        for local_i, layer in enumerate(self.layers):
            g_idx = self.layer_start + local_i
            layer.load_weights(loader, f"{p_root}.model.language_model.layers.{g_idx}", dev)

        if self.is_last_stage and self.final_norm is not None:
            with torch.no_grad():
                self.final_norm.weight.copy_(
                    loader.tensor(f"{p_root}.model.language_model.norm.weight", device=dev)
                )
        return embed_weight_cpu

    def _get_tokenizer(self):
        if self._tokenizer is None:
            self._tokenizer = load_hf_tokenizer(self.cfg.tokenizer_repo, checkpoint_dir=self._checkpoint_dir)
        return self._tokenizer

    @torch.no_grad()
    def set_prompt(
        self,
        prompt: Union[str, Sequence[int], torch.Tensor],
        *,
        state: Optional[Union[Sequence[float], torch.Tensor]] = None,
        **_kwargs: Any,
    ) -> None:
        if not self.is_first_stage:
            return
        if isinstance(prompt, torch.Tensor) and prompt.is_floating_point():
            emb_in = prompt.squeeze(0) if prompt.ndim == 3 and prompt.size(0) == 1 else prompt
            if emb_in.ndim != 2 or emb_in.size(1) != self.cfg.vlm_width:
                raise ValueError(
                    f"Floating-point prompt embedding must have shape [L, {self.cfg.vlm_width}], "
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
                "Pi05SlowBackbone cannot embed text or token IDs without loaded vocabulary weights; "
                "load a checkpoint or pass 2D floating-point prompt embeddings [L, vlm_width]"
            )

        if isinstance(prompt, str):
            tok = self._get_tokenizer()
            cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
            if state is not None:
                state_str = discretize_pi05_state(state, self.cfg.max_state_dim)
                text = f"Task: {cleaned}, State: {state_str};\nAction: "
            else:
                text = prompt if prompt.endswith("\n") else f"{prompt}\n"
            ids = tok.encode(text).ids
        elif isinstance(prompt, torch.Tensor):
            ids = [int(v) for v in prompt.flatten().tolist()]
        else:
            ids = [int(v) for v in prompt]

        newline_id = 108  # PaliGemma '\n' token
        if len(ids) >= self.cfg.prompt_len:
            ids = ids[: self.cfg.prompt_len]
        else:
            ids = ids + [newline_id] * (self.cfg.prompt_len - len(ids))

        idx_t = torch.tensor(ids, dtype=torch.long)
        emb = F.embedding(idx_t, self._embed_weight_cpu).to(
            device=self.lang_embeds.device, dtype=torch.bfloat16
        )
        emb = emb * math.sqrt(self.cfg.vlm_width)
        self.lang_embeds.copy_(emb.unsqueeze(0))

    def encode_prefix_embeds(self, obs: torch.Tensor) -> torch.Tensor:
        assert self.is_first_stage and self.vision_tower is not None and self.mm_projector is not None
        B = obs.size(0)
        vis = self.vision_tower(obs)  # [B, Cams, P, 1152]
        img_embs = self.mm_projector(vis.reshape(B, -1, self.cfg.vision_width))
        lang = self.lang_embeds.expand(B, -1, -1)
        return torch.cat([img_embs, lang], dim=1)

    def forward(self, obs_or_hidden: torch.Tensor, ctx: SlowCaptureContext) -> torch.Tensor:
        x = self.encode_prefix_embeds(obs_or_hidden) if self.is_first_stage else obs_or_hidden
        for local_i, layer in enumerate(self.layers):
            x, k, v = layer(x, self.pos_ids)
            ctx.write_layer_kv(self.layer_start + local_i, k, v)
        if self.is_last_stage and self.final_norm is not None:
            x = self.final_norm(x)
        return x

    def split_pipeline(self, stage_devices: Sequence[int]) -> List[nn.Module]:
        P = len(stage_devices)
        if P == 1:
            return [self.to(device=f"cuda:{int(stage_devices[0])}", dtype=torch.bfloat16)]
        if not (self.is_first_stage and self.is_last_stage):
            raise RuntimeError("split_pipeline requires a full unsplit Pi05SlowBackbone")
        L = len(self.layers)
        if P > L:
            raise ValueError(f"Cannot split {L} Gemma layers across {P} pipeline stages")

        stages: List[nn.Module] = []
        for p in range(P):
            lo = (p * L) // P
            hi = ((p + 1) * L) // P
            dev = torch.device(f"cuda:{int(stage_devices[p])}")
            stg = Pi05SlowStage(
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
                    assert stg.mm_projector is not None and self.mm_projector is not None
                    stg.vision_tower.load_state_dict(self.vision_tower.state_dict())
                    stg.mm_projector.load_state_dict(self.mm_projector.state_dict())
                    stg.lang_embeds.copy_(self.lang_embeds.to(dev))
                    stg._embed_weight_cpu = self._embed_weight_cpu
                    stg._checkpoint_dir = self._checkpoint_dir
                    stg._tokenizer = self._tokenizer
                for idx_local in range(hi - lo):
                    stg.layers[idx_local].load_state_dict(self.layers[lo + idx_local].state_dict())
                if p == P - 1 and stg.final_norm is not None and self.final_norm is not None:
                    stg.final_norm.load_state_dict(self.final_norm.state_dict())
            stages.append(stg)
        return stages


Pi05SlowBackbone = Pi05SlowStage


class Pi05ExpertLayer(nn.Module):
    def __init__(self, cfg: Pi05Config, *, use_fp8: bool, device: torch.device) -> None:
        super().__init__()
        self.num_heads = cfg.num_heads
        self.num_kv_heads = cfg.num_kv_heads
        self.head_dim = cfg.head_dim
        self.q_dim = cfg.num_heads * cfg.head_dim
        self.kv_dim = cfg.num_kv_heads * cfg.head_dim

        self.input_layernorm = AdaRMSNorm(cfg.expert_width, cfg.expert_width, device=device)
        self.qkv_proj = FP8Linear(
            cfg.expert_width, self.q_dim + 2 * self.kv_dim, use_fp8=use_fp8, device=device
        )
        self.o_proj = FP8Linear(self.q_dim, cfg.expert_width, use_fp8=use_fp8, device=device)

        self.post_attention_layernorm = AdaRMSNorm(cfg.expert_width, cfg.expert_width, device=device)
        self.gate_up_proj = FP8Linear(
            cfg.expert_width, 2 * cfg.expert_mlp_dim, use_fp8=use_fp8, device=device
        )
        self.down_proj = FP8Linear(cfg.expert_mlp_dim, cfg.expert_width, use_fp8=use_fp8, device=device)
        self.requires_grad_(False)

    def load_weights(self, loader: SafetensorLoader, prefix: str, device: torch.device) -> None:
        loader.load_linear(self.input_layernorm.dense, f"{prefix}.input_layernorm.dense", device=device)
        wq = loader.tensor(f"{prefix}.self_attn.q_proj.weight", device=device)
        wk = loader.tensor(f"{prefix}.self_attn.k_proj.weight", device=device)
        wv = loader.tensor(f"{prefix}.self_attn.v_proj.weight", device=device)
        self.qkv_proj.set_weight(torch.cat([wq, wk, wv], dim=0))
        self.o_proj.set_weight(loader.tensor(f"{prefix}.self_attn.o_proj.weight", device=device))
        loader.load_linear(
            self.post_attention_layernorm.dense, f"{prefix}.post_attention_layernorm.dense", device=device
        )
        wg = loader.tensor(f"{prefix}.mlp.gate_proj.weight", device=device)
        wu = loader.tensor(f"{prefix}.mlp.up_proj.weight", device=device)
        self.gate_up_proj.set_weight(torch.cat([wg, wu], dim=0))
        self.down_proj.set_weight(loader.tensor(f"{prefix}.mlp.down_proj.weight", device=device))

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        pos_ids: torch.Tensor,
        state: SemanticStateView,
        layer_idx: int,
    ) -> torch.Tensor:
        B, Q_len, _ = x.shape
        h, gate_attn = self.input_layernorm(x, cond)
        qkv = self.qkv_proj(h)
        q, k_suf, v_suf = qkv.split([self.q_dim, self.kv_dim, self.kv_dim], dim=-1)
        q = q.view(B, Q_len, self.num_heads, self.head_dim)
        k_suf = k_suf.view(B, Q_len, self.num_kv_heads, self.head_dim)
        v_suf = v_suf.view(B, Q_len, self.num_kv_heads, self.head_dim)

        q = apply_rope_1d(q, pos_ids)
        k_suf = apply_rope_1d(k_suf, pos_ids)

        # Zero-copy Split-KV Attention: Paged TSC prefix + 50x50 suffix self-attention via exact LSE
        out_pre, lse_pre = state.cross_attend(q, layer_idx, return_lse=True)  # type: ignore[misc]
        out_suf, lse_suf = suffix_self_attention_with_lse(q, k_suf, v_suf, causal=False)
        attn_out = merge_attention_lse(out_pre, lse_pre, out_suf, lse_suf)

        x = x + self.o_proj(attn_out.reshape(B, Q_len, self.q_dim)) * gate_attn
        h2, gate_mlp = self.post_attention_layernorm(x, cond)
        gate, up = self.gate_up_proj(h2).chunk(2, dim=-1)
        mlp_out = self.down_proj(F.gelu(gate, approximate="tanh") * up)
        return x + mlp_out * gate_mlp


class Pi05FastHead(FastHead):
    """System-1 Gemma-300M flow-matching action head for pi0.5.

    Robot state is discretized into the System-2 prompt via ``set_prompt(..., state=...)``;
    ``fast_obs`` has shape ``(chunk_size, max_action_dim)`` supplying the initial flow noise.
    """

    def __init__(
        self,
        cfg: Pi05Config,
        *,
        device: Union[str, torch.device] = "cuda:0",
    ) -> None:
        super().__init__()
        dev = torch.device(device)
        self.cfg = cfg
        self.action_dim = cfg.action_dim
        self.chunk_size = cfg.chunk_size
        self.obs_shape = (cfg.chunk_size, cfg.max_action_dim)
        self.max_q_len = max(64, cfg.chunk_size)
        self.max_q_heads = cfg.num_heads

        validate_fast_precision(cfg.fast_precision)
        use_fp8 = cfg.fast_precision == "fp8"
        self.action_in_proj = nn.Linear(
            cfg.max_action_dim, cfg.expert_width, bias=True, device=dev, dtype=torch.bfloat16
        )
        self.time_mlp_in = nn.Linear(
            cfg.expert_width, cfg.expert_width, bias=True, device=dev, dtype=torch.bfloat16
        )
        self.time_mlp_out = nn.Linear(
            cfg.expert_width, cfg.expert_width, bias=True, device=dev, dtype=torch.bfloat16
        )
        self.layers = nn.ModuleList(
            [Pi05ExpertLayer(cfg, use_fp8=use_fp8, device=dev) for _ in range(cfg.num_layers)]
        )
        self.final_norm = AdaRMSNorm(cfg.expert_width, cfg.expert_width, device=dev)
        self.action_out_proj = nn.Linear(
            cfg.expert_width, cfg.max_action_dim, bias=True, device=dev, dtype=torch.bfloat16
        )

        self.register_buffer(
            "suffix_offsets",
            torch.arange(cfg.chunk_size, dtype=torch.int64, device=dev),
        )
        steps_t = torch.tensor(
            [1.0 - s / float(cfg.num_inference_steps) for s in range(cfg.num_inference_steps)],
            dtype=torch.float32,
            device=dev,
        )
        sin_embs = sinusoidal_time_embedding(
            steps_t,
            cfg.expert_width,
            min_period=cfg.min_period,
            max_period=cfg.max_period,
        ).to(torch.bfloat16)
        self.register_buffer("step_sin_embs", sin_embs)
        self.requires_grad_(False)

    def load_weights(self, loader: SafetensorLoader, device: Union[str, torch.device]) -> None:
        dev = torch.device(device)
        m_prefix = "model." if loader.has("model.action_in_proj.weight") else ""
        loader.load_linear(self.action_in_proj, f"{m_prefix}action_in_proj", device=dev)
        loader.load_linear(self.time_mlp_in, f"{m_prefix}time_mlp_in", device=dev)
        loader.load_linear(self.time_mlp_out, f"{m_prefix}time_mlp_out", device=dev)
        loader.load_linear(self.action_out_proj, f"{m_prefix}action_out_proj", device=dev)
        e_root = f"{m_prefix}paligemma_with_expert.gemma_expert.model"
        for idx, layer in enumerate(self.layers):
            layer.load_weights(loader, f"{e_root}.layers.{idx}", dev)
        loader.load_linear(self.final_norm.dense, f"{e_root}.norm.dense", device=dev)

    def validate_channel(self, channel: TemporalChannel) -> None:
        if channel.mode != "kv":
            raise ValueError(f"Pi05FastHead requires TemporalChannel(mode='kv'), got '{channel.mode}'")
        if (
            channel.layers != self.cfg.num_layers
            or channel.num_kv_heads != self.cfg.num_kv_heads
            or channel.head_dim != self.cfg.head_dim
        ):
            raise ValueError(
                f"Pi05FastHead expects KV channel (layers={self.cfg.num_layers}, "
                f"num_kv_heads={self.cfg.num_kv_heads}, head_dim={self.cfg.head_dim}), "
                f"got (layers={channel.layers}, num_kv_heads={channel.num_kv_heads}, head_dim={channel.head_dim})"
            )

    def calibrate_fp8(
        self,
        channel: TemporalChannel,
        device: torch.device,
        sample_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> None:
        if self.cfg.fast_precision != "fp8":
            return
        from ..channel import _load_ext

        _tempo_C = _load_ext()

        B = 4
        max_pages = B * math.ceil(channel.max_seq_len / channel.page_size) + 8
        tsc = _tempo_C.TemporalStateCache(
            B, 1, max_pages, channel.layers, channel.page_size,
            channel.num_kv_heads, channel.head_dim, channel.max_seq_len,
            True, self.max_q_len, self.max_q_heads, device.index or 0,
        )
        num_p = math.ceil(channel.max_seq_len / channel.page_size)
        pt_rows = []
        for b in range(B):
            if sample_kv is not None:
                src_b = b % sample_kv[0].size(1)
                k = sample_kv[0][:, src_b : src_b + 1].to(device=device, dtype=torch.bfloat16).contiguous()
                v = sample_kv[1][:, src_b : src_b + 1].to(device=device, dtype=torch.bfloat16).contiguous()
            else:
                k = torch.randn(
                    (channel.layers, 1, channel.max_seq_len, channel.num_kv_heads, channel.head_dim),
                    dtype=torch.bfloat16, device=device,
                ) * 0.5
                v = torch.randn_like(k)
            tsc.write_single_stream_kv(b, k, v, 100_000_000, False)
            pt_rows.append(list(range(b * num_p, (b + 1) * num_p)))
        view = SemanticStateView(
            k_pool=tsc.k_pool, v_pool=tsc.v_pool, page_timestamps=tsc.page_timestamps,
            splitk_partial_out=tsc.splitk_partial_out(0), splitk_partial_lse=tsc.splitk_partial_lse(0),
            page_table=torch.tensor(pt_rows, dtype=torch.int32, device=device),
            seq_lens=torch.full((B,), channel.max_seq_len, dtype=torch.int32, device=device),
            t_now_ns=torch.full((B,), 100_000_000, dtype=torch.int64, device=device),
            relative_time_rope=channel.relative_time_rope, max_seq_len=channel.max_seq_len, mode="kv",
        )
        dummy_obs = torch.randn(
            (B, self.chunk_size, self.cfg.max_action_dim), dtype=torch.bfloat16, device=device
        )
        dummy_pref = torch.zeros((B, 10, self.action_dim), dtype=torch.bfloat16, device=device)
        calibrate_fp8_submodules(self, lambda: self.forward(dummy_obs, view, dummy_pref))

    def _step_velocity(
        self,
        x_t: torch.Tensor,
        step_idx: int,
        pos_ids: torch.Tensor,
        state: SemanticStateView,
    ) -> torch.Tensor:
        B = x_t.size(0)
        sin_emb = self.step_sin_embs[step_idx : step_idx + 1].expand(B, -1)
        cond = F.silu(self.time_mlp_out(F.silu(self.time_mlp_in(sin_emb))))
        h = self.action_in_proj(x_t.to(torch.bfloat16))
        for idx, layer in enumerate(self.layers):
            h = layer(h, cond, pos_ids, state, idx)
        h, _ = self.final_norm(h, cond)
        return self.action_out_proj(h).float()

    def forward_with_noise(
        self,
        state: SemanticStateView,
        noise: torch.Tensor,
        *,
        frozen_prefix: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Runs exact NOISE_AT_ONE Euler integration from explicit initial noise [B, chunk_size, max_action_dim]."""
        x_t = noise.float().clone()
        pos_ids = state.seq_lens.to(torch.int64).unsqueeze(1) + self.suffix_offsets.unsqueeze(0)
        num_steps = self.cfg.num_inference_steps
        dt = -1.0 / float(num_steps)

        P_len = frozen_prefix.size(1) if frozen_prefix is not None else 0
        pref_f32 = frozen_prefix.float() if frozen_prefix is not None else None

        for step in range(num_steps):
            t_cur = 1.0 - step / float(num_steps)
            v_t = self._step_velocity(x_t, step, pos_ids, state)
            x_t = x_t + dt * v_t
            if pref_f32 is not None and P_len > 0:
                t_next = max(0.0, t_cur + dt)
                inpaint = (1.0 - t_next) * pref_f32 + t_next * noise[:, :P_len, : self.action_dim].float()
                inpaint = state.splice_prefix(x_t[:, :P_len, : self.action_dim], inpaint)
                x_t = torch.cat(
                    [
                        torch.cat([inpaint, x_t[:, :P_len, self.action_dim :]], dim=-1),
                        x_t[:, P_len:, :],
                    ],
                    dim=1,
                )

        return x_t[:, :, : self.action_dim].to(torch.bfloat16)

    def forward(
        self,
        fast_obs: torch.Tensor,
        state: SemanticStateView,
        frozen_prefix: torch.Tensor,
    ) -> torch.Tensor:
        if fast_obs.ndim != 3 or fast_obs.shape[1:] != (self.chunk_size, self.cfg.max_action_dim):
            raise ValueError(
                f"Pi05FastHead expects fast_obs initial noise of shape "
                f"(B, {self.chunk_size}, {self.cfg.max_action_dim}), got {tuple(fast_obs.shape)}"
            )
        return self.forward_with_noise(state, fast_obs, frozen_prefix=frozen_prefix)


@ModelRegistry.register("pi05")
def build_pi05(
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
    tok_repo = str(model_kwargs.get("tokenizer_repo", _PALIGEMMA_TOKENIZER_REPO))
    if prompt is not None and "prompt_len" not in model_kwargs:
        if isinstance(prompt, str):
            tok = load_hf_tokenizer(tok_repo, checkpoint_dir=checkpoint_dir)
            text = prompt if prompt.endswith("\n") else f"{prompt}\n"
            req_len = len(tok.encode(text).ids)
        elif isinstance(prompt, torch.Tensor) and prompt.is_floating_point():
            req_len = int(prompt.size(-2))
        else:
            seq = prompt.flatten().tolist() if isinstance(prompt, torch.Tensor) else list(prompt)
            req_len = len(seq)
        model_kwargs["prompt_len"] = max(Pi05Config.prompt_len, req_len)

    cfg = Pi05Config.from_dict(config, fast_precision=fast_precision, **model_kwargs)
    P = len(slow_devices[0])
    L = cfg.num_layers
    if P > L:
        raise ValueError(f"Cannot split {L} pi0.5 layers across {P} pipeline stages")

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
                    stg = Pi05SlowStage(
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
            sample_kv = (
                capture_slow_calibration_sample(
                    slow_replicas[0], ch_cal, (cfg.num_cameras, 3, cfg.image_size, cfg.image_size)
                )[0]
                if fast_precision == "fp8"
                else None
            )
            for dev_id in fast_devices:
                f_dev = torch.device(f"cuda:{int(dev_id)}")
                head = Pi05FastHead(cfg, device=f_dev)
                head.load_weights(loader, f_dev)
                head.eval()
                head.calibrate_fp8(ch_cal, f_dev, sample_kv=sample_kv)
                fast_shards.append(head)

            m_prefix = "model." if loader.has("model.action_in_proj.weight") else ""
            loader.verify_all_consumed(
                ignore_keys=(f"{m_prefix}paligemma_with_expert.gemma_expert.lm_head.weight",)
            )
    else:
        for pipe_devs in slow_devices:
            stages = []
            for p, dev_id in enumerate(pipe_devs):
                lo = (p * L) // P
                hi = ((p + 1) * L) // P
                stg = Pi05SlowStage(
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
        sample_kv = (
            capture_slow_calibration_sample(
                slow_replicas[0], ch_cal, (cfg.num_cameras, 3, cfg.image_size, cfg.image_size)
            )[0]
            if fast_precision == "fp8"
            else None
        )
        for dev_id in fast_devices:
            f_dev = torch.device(f"cuda:{int(dev_id)}")
            head = Pi05FastHead(cfg, device=f_dev)
            head.eval()
            head.calibrate_fp8(ch_cal, f_dev, sample_kv=sample_kv)
            fast_shards.append(head)

    ch = slow_replicas[0][0].channel  # type: ignore[attr-defined]
    return DualSystemDeployment(
        channel=ch,
        slow_replicas=slow_replicas,
        fast_shards=fast_shards,
        action_dim=cfg.action_dim,
        chunk_size=cfg.chunk_size,
        slow_obs_shape=(cfg.num_cameras, 3, cfg.image_size, cfg.image_size),
        fast_obs_shape=(cfg.chunk_size, cfg.max_action_dim),
        step_dt_ms=fast_shards[0].resolve_step_dt_ms(),
        max_q_len=max(64, cfg.chunk_size),
        max_q_heads=cfg.num_heads,
    )
