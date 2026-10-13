"""Semantic-only LoRA training: assistant response token NLL, no WHERE or projector."""
import json
import math
import random
from pathlib import Path

import torch
from .native import make_native, model_inputs, pack_natives
from .prompt import semantic_target
from .scoring import dump_json


def train(model, processor, records, *, identity, output_dir, prompt_style, epochs,
          batch_size, accumulation, lr, warmup_ratio, seed, save_every=0,
          weight_decay=0.01, max_steps=None):
    if not records or epochs <= 0 or batch_size <= 0 or accumulation <= 0 or lr <= 0:
        raise ValueError('training needs nonempty data and positive epochs/batch/accumulation/learning rate')
    if not (0 <= warmup_ratio <= 1) or save_every < 0:
        raise ValueError('warmup ratio or checkpoint interval invalid')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError('SFT output directory must be empty; use a new run directory')
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError('training must have trainable LoRA weights')
    model.train()
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    micro_batches_per_epoch = math.ceil(len(records) / batch_size)
    updates_per_epoch = math.ceil(micro_batches_per_epoch / accumulation)
    total_steps = epochs * updates_per_epoch
    if max_steps is not None:
        if max_steps < 1:
            raise ValueError('max_steps must be positive')
        total_steps = min(total_steps, max_steps)
    warmup_steps = round(warmup_ratio * total_steps)

    def multiplier(step):
        if warmup_steps and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1 + math.cos(math.pi * min(1.0, progress))))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    info = {'method': 'semantic_sft', 'prompt_style': prompt_style,
            'split_manifest_identity': identity, 'train_query_count': len(records),
            'train_query_ids_sha256': __import__('hashlib').sha256(
                '\n'.join(sorted(q.record_id for q in records)).encode()).hexdigest(),
            'epochs': epochs, 'per_device_batch_size': batch_size,
            'gradient_accumulation_steps': accumulation,
            'effective_batch_size': batch_size * accumulation,
            'optimizer': 'AdamW', 'learning_rate': lr, 'warmup_ratio': warmup_ratio,
            'scheduler': 'cosine', 'weight_decay': weight_decay, 'seed': seed,
            'max_steps': max_steps, 'trainable_parameters': sum(p.numel() for p in params),
            'loss': 'assistant-only next-token cross-entropy; model native loss',
            'conditioning': 'image+task+GT XYD+gold WHY memberships; no supports'}
    dump_json(output / 'resolved_config.json', info)
    optimizer.zero_grad(set_to_none=True)
    step = 0
    with (output / 'training_log.jsonl').open('w', encoding='utf-8') as log:
        for epoch in range(1, epochs + 1):
            shuffled = list(records)
            random.Random(seed + epoch).shuffle(shuffled)
            batches = [shuffled[i:i+batch_size] for i in range(0, len(shuffled), batch_size)]
            for window in range(0, len(batches), accumulation):
                group = batches[window:window+accumulation]
                # Last partial gradient-accumulation window receives full-scale gradients.
                mean_loss = 0.0
                for micro in group:
                    native = [make_native(processor, q, prompt_style, training=True) for q in micro]
                    batch = model_inputs(pack_natives(processor, native, training=True), model)
                    prediction = model(**batch, use_cache=False)
                    loss = prediction.loss
                    if loss is None or not bool(torch.isfinite(loss)):
                        raise FloatingPointError('semantic-only SFT loss is non-finite')
                    mean_loss += float(loss.detach()) / len(group)
                    (loss / len(group)).backward()
                grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                item = {'step': step, 'epoch': epoch, 'loss': mean_loss,
                        'gradient_norm': grad_norm, 'learning_rate': scheduler.get_last_lr()[0]}
                log.write(json.dumps(item, allow_nan=False) + '\n')
                log.flush()
                if save_every and step % save_every == 0:
                    save_checkpoint(model, output / f'checkpoint-step-{step}',
                                    info, epoch=epoch, step=step, optimizer=optimizer, scheduler=scheduler)
                if step >= total_steps:
                    break
            if step >= total_steps:
                break
    save_checkpoint(model, output / 'final', info, epoch=epoch, step=step,
                    optimizer=optimizer, scheduler=scheduler)
    return {**info, 'completed_epochs': epoch, 'optimizer_step': step,
            'checkpoint': str(output / 'final')}


def save_checkpoint(model, path, info, *, epoch, step, optimizer=None, scheduler=None):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    dump_json(path / 'semantic_sft_checkpoint.json', {
        **info, 'checkpoint_epoch': epoch, 'checkpoint_optimizer_step': step})
    if optimizer is not None:
        torch.save({'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict() if scheduler is not None else None,
                    'epoch': epoch, 'optimizer_step': step}, path / 'training_state.pt')
