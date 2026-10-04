from torch import nn


def build_projector(d_model):
    return nn.Sequential(nn.Linear(d_model, d_model), nn.LayerNorm(d_model))
