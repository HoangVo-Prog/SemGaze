"""Exact released HF InternVL base, optionally augmented by semantic-only LoRA."""
from pathlib import Path

BASE_MODEL = 'OpenGVLab/InternVL3_5-8B-HF'
TARGET_MODULES = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')


def load_baseline_model(*, method, device='cuda', precision='bf16', adapter_path=None,
                        gradient_checkpointing=False, training=False):
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    if method not in ('semantic_zero_shot', 'semantic_sft'):
        raise ValueError(f'unsupported method {method}')
    if precision not in ('bf16', 'fp32'):
        raise ValueError('precision must be bf16 or fp32')
    if str(device).startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA device requested but unavailable')
        if precision == 'bf16' and not torch.cuda.is_bf16_supported():
            raise RuntimeError('requested bf16 but CUDA GPU does not support it')
    dtype = {'bf16': torch.bfloat16, 'fp32': torch.float32}[precision]
    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    model = AutoModelForImageTextToText.from_pretrained(BASE_MODEL, dtype=dtype)
    if model.__class__.__name__ != 'InternVLForConditionalGeneration':
        raise ValueError('base model is not native HF InternVLForConditionalGeneration')
    # Never add <END_FIX>, never resize input or output embeddings.
    if method == 'semantic_sft':
        from peft import LoraConfig, PeftModel, get_peft_model
        if adapter_path:
            model = PeftModel.from_pretrained(model, str(Path(adapter_path)), is_trainable=training)
        else:
            targets = [name for name, _ in model.named_modules()
                       if '.language_model.' in name and
                       any(name.endswith('.' + suffix) for suffix in TARGET_MODULES)]
            if not targets or any(not any(name.endswith('.' + suffix) for name in targets)
                                  for suffix in TARGET_MODULES):
                raise ValueError('LoRA target projection module missing from InternVL language model')
            model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                                                     target_modules=targets, task_type='CAUSAL_LM'))
        for name, param in model.named_parameters():
            if param.requires_grad and ('lora_' not in name or '.language_model.' not in name):
                raise ValueError(f'unexpected semantic SFT trainable parameter: {name}')
        if gradient_checkpointing and training:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            model.config.use_cache = False
    else:
        if adapter_path is not None or gradient_checkpointing:
            raise ValueError('frozen zero-shot must not load adapter or enable training')
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    model.to(device=device)
    model.train() if training else model.eval()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if method == 'semantic_zero_shot' and trainable != 0:
        raise ValueError('zero-shot model has trainable parameters')
    if method == 'semantic_sft' and trainable == 0 and training:
        raise ValueError('SFT has no trainable LoRA parameters')
    return model, processor, trainable
