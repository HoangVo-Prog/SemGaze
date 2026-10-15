from torch import nn


def build_projector(d_model, *, enabled=True):
    return nn.Sequential(nn.Linear(d_model, d_model), nn.LayerNorm(d_model)) if enabled else nn.Identity()
