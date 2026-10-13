"""Single-image InternVL semantic baseline greedy inference with resumable rows."""
from pathlib import Path
import json

import torch
from semgaze.semantic.flat.parser import parse_flat_output
from .native import make_native, model_inputs, pack_natives, decode_generated
from .scoring import dump_json, load_jsonl, verify_complete


def predict(model, processor, queries, *, method, prompt_style, output_dir,
            max_new_tokens=512, batch_size=1, resume=False, precision='bf16', model_id=None, run_info=None):
    if batch_size < 1 or max_new_tokens < 1:
        raise ValueError('positive generation batch size and budget required')
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'predictions.jsonl'
    meta_path = directory / 'run_metadata.json'
    if path.exists() and not resume:
        raise FileExistsError(f'{path} exists; explicitly pass --resume to continue')
    metadata = {'method': method, 'prompt_style': prompt_style, 'max_new_tokens': max_new_tokens,
                'do_sample': False, 'decode': 'greedy', 'precision': precision, 'model_id': model_id,
                'input': 'single image + task + textual GT XYD + gold WHY-group membership',
                'support_examples': False, 'state_projector': False,
                'checkpoint_metadata': run_info or {}}
    if meta_path.exists():
        if json.loads(meta_path.read_text()) != metadata:
            raise ValueError('resume metadata differs from requested inference configuration')
    else:
        if path.exists():
            raise ValueError('predictions file exists without trustworthy metadata')
        dump_json(meta_path, metadata)
    old = load_jsonl(path) if path.exists() else []
    expected = {q.record_id for q in queries}
    completed = set()
    for row in old:
        rid = row.get('query_id')
        if rid in completed or rid not in expected or row.get('method') != method or row.get('prompt_style') != prompt_style:
            raise ValueError('resumed predictions contain a duplicate, extra, or mismatched query')
        completed.add(rid)
    model.eval()
    todo = [q for q in queries if q.record_id not in completed]
    tokenizer = processor.tokenizer
    with path.open('a', encoding='utf-8') as stream, torch.inference_mode():
        for start in range(0, len(todo), batch_size):
            batch = todo[start:start + batch_size]
            natives = [make_native(processor, q, prompt_style, training=False) for q in batch]
            packed = model_inputs(pack_natives(processor, natives, left_pad=True), model)
            tokens = model.generate(**packed, max_new_tokens=max_new_tokens, do_sample=False,
                                    eos_token_id=tokenizer.eos_token_id,
                                    pad_token_id=tokenizer.pad_token_id, use_cache=True)
            texts = decode_generated(tokenizer, tokens, packed['input_ids'], max_new_tokens=max_new_tokens)
            if len(texts) != len(batch):
                raise RuntimeError('generation batch dropped a query')
            for q, generated in zip(batch, texts):
                parsed = parse_flat_output(generated, len(q.x_px), len(q.semantic.why_groups))
                row = {'query_id': q.record_id, 'subject': q.subject,
                       'stimulus_id': q.stimulus_id, 'dataset': q.dataset,
                       'condition': q.condition, 'task': q.task,
                       'method': method, 'prompt_style': prompt_style,
                       'gt_fixation_count': len(q.x_px),
                       'gt_group_count': len(q.semantic.why_groups), 'text': generated,
                       'semantic_generation': parsed,
                       'generation': {'do_sample': False, 'max_new_tokens': max_new_tokens,
                                      'eos_token_id': tokenizer.eos_token_id,
                                      'pad_token_id': tokenizer.pad_token_id}}
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
    result = verify_complete(load_jsonl(path), queries, method=method)
    return result
