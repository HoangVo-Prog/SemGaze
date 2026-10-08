"""Targeted regression tests for P1: embedding fast path and state extraction."""
import torch
from torch import nn
from semgaze.model.trainable_tokens import RowEmbedding
from semgaze.state.extractor import extract_query_states


def test_semantic_embedding_fast_path_exact_and_where_end_fix_grad():
    base = nn.Embedding(16, 8)
    base.requires_grad_(False)
    row = nn.Parameter(torch.randn(8))
    emb = RowEmbedding(base, 9, row)
    ids = torch.tensor([[2, 3, 4], [4, 5, 6]])
    torch.testing.assert_close(emb.forward_without_end_fix(ids), emb(ids), atol=0, rtol=0)
    embedding = emb(torch.tensor([[2, 9, 4]]))
    embedding.sum().backward()
    assert row.grad is not None and row.grad.abs().sum() > 0


def test_unpadded_state_extraction_and_backward_parity():
    ids = torch.tensor([[1, 9, 2, 9, 3, 0], [1, 2, 9, 4, 0, 0]])
    labels = torch.full_like(ids, -100)
    labels[0, 1] = labels[0, 3] = labels[1, 2] = 9
    x1 = torch.randn(2, 6, 8, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    padded, _, positions = extract_query_states(x1, ids, labels, 9, [2, 1])
    selected, mask, positions_new = extract_query_states(x2, ids, labels, 9, [2, 1], return_unpadded=True)
    assert mask is None and all(torch.equal(a,b) for a,b in zip(positions, positions_new))
    for old, new in zip((padded[0, :2], padded[1, :1]), selected):
        torch.testing.assert_close(old, new, atol=0, rtol=0)
    sum((padded[0, :2].sum(), padded[1, :1].sum())).backward()
    sum(x.sum() for x in selected).backward()
    torch.testing.assert_close(x1.grad, x2.grad, atol=0, rtol=0)
