"""Train one vocabulary row without optimizer state/decay for old rows.

The base vocabulary matrices stay frozen. Separate row parameters replace only
END_FIX in the embedding lookup and LM logits. Tied vocabularies share one row.
"""
import torch
from torch import nn
from torch.nn import functional as F


class RowEmbedding(nn.Module):
    def __init__(self, base, token_id, row):
        super().__init__()
        self.base, self.token_id, self.end_fix_row = base, token_id, row
        self.num_embeddings, self.embedding_dim = base.weight.shape

    @property
    def weight(self):
        return self.base.weight

    def forward(self, input_ids):
        ordinary = self.base(input_ids)
        return torch.where((input_ids == self.token_id).unsqueeze(-1), self.end_fix_row.to(ordinary.dtype), ordinary)


class RowHead(nn.Module):
    def __init__(self, base, token_id, row):
        super().__init__()
        self.base, self.token_id, self.end_fix_row = base, token_id, row
        self.out_features, self.in_features = base.weight.shape

    @property
    def weight(self):
        return self.base.weight

    def forward(self, hidden):
        logits = self.base(hidden)
        replacement = F.linear(hidden, self.end_fix_row.to(hidden.dtype).unsqueeze(0))
        if getattr(self.base, 'bias', None) is not None:
            replacement = replacement + self.base.bias[self.token_id]
        # Linear backward needs its input/weight, not its output. CopySlices
        # replaces just this column and zeros its frozen-base gradient contribution.
        logits[..., self.token_id:self.token_id + 1] = replacement
        return logits


def install_trainable_rows(model, token_id):
    embedding, head = model.get_input_embeddings(), model.get_output_embeddings()
    if isinstance(embedding, RowEmbedding):
        raise ValueError('END_FIX rows already installed')
    tied = embedding.weight is head.weight
    embedding.requires_grad_(False)
    head.requires_grad_(False)
    input_row = nn.Parameter(embedding.weight[token_id].detach().clone())
    output_row = input_row if tied else nn.Parameter(head.weight[token_id].detach().clone())
    model.set_input_embeddings(RowEmbedding(embedding, token_id, input_row))
    model.set_output_embeddings(RowHead(head, token_id, output_row))
    return input_row, output_row, tied
