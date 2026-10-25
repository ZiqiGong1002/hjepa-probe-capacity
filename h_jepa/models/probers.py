from torch import nn

from models.module import MLP


def build_prober(*, input_dim, output_dim: int, hidden_dim: int = 512) -> nn.Module:
    # hidden_dim <= 0 gives a linear probe; otherwise the original one-hidden-layer MLP.
    if int(hidden_dim) <= 0:
        return nn.Linear(int(input_dim), int(output_dim))
    return MLP(input_dim=int(input_dim), hidden_dim=int(hidden_dim), output_dim=int(output_dim))
