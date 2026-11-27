"""Custom System 1 + System 2 policy via SlowBackbone and FastHead."""

import pathlib
import sys
import torch
import torch.nn as nn

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tempo import Engine, FastHead, SlowBackbone, TemporalChannel, mark_sliceable


class MySlowVLM(SlowBackbone):
    def __init__(self) -> None:
        super().__init__()
        self.channel = TemporalChannel(hidden_dim=256, seq_len=64, mode="hidden")
        self.obs_shape = (64, 256)
        self.encoder = nn.Sequential(mark_sliceable(nn.Linear(256, 256)), nn.SiLU())

    def forward(self, obs: torch.Tensor, ctx) -> None:
        ctx.write_hidden_state(self.encoder(obs))  # or ctx.write_layer_kv(...) in mode="kv"


class MyFastPolicy(FastHead):
    action_dim, chunk_size, obs_shape = 7, 50, (14,)

    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(14 + 256, 50 * 7)

    def forward(self, fast_obs: torch.Tensor, state, frozen_prefix: torch.Tensor) -> torch.Tensor:
        hidden, _ = state.read_hidden()  # or state.cross_attend(q, layer_idx) in mode="kv"
        pred = self.head(torch.cat([fast_obs, hidden.mean(dim=1)], dim=-1)).view(-1, 50, 7)
        p = frozen_prefix.size(1)
        return torch.cat([state.splice_prefix(pred[:, :p], frozen_prefix), pred[:, p:]], dim=1)


def main() -> None:
    with Engine(slow=MySlowVLM(), fast=MyFastPolicy()) as engine:
        engine.push_slow(torch.randn(64, 256))
        engine.wait_ready(torch.randn(14))
        actions, *_ = engine.sample_actions(cpu=True)
    print(f"[custom-model] actions={tuple(actions.shape)}")


if __name__ == "__main__":
    main()
