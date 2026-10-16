"""Exact shifted full-vocabulary CE, reduced within episodes before batch mean."""
from dataclasses import dataclass
import torch
from torch.nn import functional as F


@dataclass
class SelectedCausalNLL:
    token_nll: torch.Tensor
    episode_losses: torch.Tensor
    token_counts: torch.Tensor

    @property
    def loss(self):
        return self.episode_losses.mean()


def compute_selected_causal_nll(final_hidden, labels, head):
    """[B,L,D], [B,L] -> NLL[M], losses[B]; M excludes ignored/padded targets.

    Pass host labels when available to keep variable-size index discovery off CUDA.
    Position t predicts label t+1, exactly as HF ForCausalLMLoss. The final
    predictor never contributes. Empty supervision is an invalid episode.
    """
    if final_hidden.shape[:2] != labels.shape:
        raise ValueError('hidden/label shapes do not align')
    targets = labels[:, 1:]
    valid = targets != -100
    counts = valid.sum(dim=1)
    if bool((counts == 0).any()):
        raise ValueError('each episode must have supervised causal targets')
    episode, position = valid.nonzero(as_tuple=True)
    device = final_hidden.device
    episode_device = episode.to(device, non_blocking=True)
    selected = final_hidden[episode_device, position.to(device, non_blocking=True)]
    # The vocabulary is never restricted; FP32 matches native HF loss precision.
    token_nll = F.cross_entropy(head(selected).float(), targets[episode, position].to(device), reduction='none')
    sums = token_nll.new_zeros(labels.shape[0]).index_add(0, episode_device, token_nll)
    counts = counts.to(device)
    return SelectedCausalNLL(token_nll, sums / counts, counts)


def forward_backbone(model, inputs, *, use_cache=False):
    """InternVL backbone keeps installed PEFT LoRA layers and trainable rows.

    No adapter disabling/merging, language no_grad, or final-state detachment.
    This deliberately supports the audited HF InternVL architecture only.
    """
    base = model.get_base_model()
    if base.__class__.__name__ != 'InternVLForConditionalGeneration':
        raise TypeError('selected training requires HF InternVLForConditionalGeneration')
    return base.model(**{k: v for k, v in inputs.items() if k != 'labels'},
                      output_hidden_states=False, use_cache=use_cache, return_dict=True)
