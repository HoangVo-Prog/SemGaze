"""Native InternVL serialization with explicit, verified assistant spans."""
from dataclasses import dataclass
from pathlib import Path
import torch
from .conversation import build_where_conversation
from .serialization import serialize_xyd_record

ROOT = Path(__file__).resolve().parents[2]


class WhereContextOverflowError(ValueError):
    pass


@dataclass
class NativeBatch:
    inputs: dict
    boundaries: tuple[int, ...]
    response_start: int
    response_length: int
    rendered_text: str
    image_paths: tuple[str, ...]
    response_offsets: tuple[tuple[int, int], ...]


def load_images(paths, cache=None):
    from PIL import Image
    images = []
    for path in paths:
        if cache is not None and path in cache:
            images.append(cache[path])
            continue
        with Image.open(ROOT / path) as image:
            images.append(image.convert('RGB'))
        if cache is not None:
            cache[path] = images[-1]
    return images


def collate_native(processor, messages, image_paths, target=None, *, prompt=None, image_cache=None):
    """Boundaries are host character offsets, never placeholder vocabulary IDs.

    A token straddling a state boundary is split by tokenizing the adjacent text
    segments independently. This is necessary for native BPE tokens such as ':\n'.
    Image expansion itself is performed only by the native processor.
    """
    tokenizer = processor.tokenizer
    prefix = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full = prefix if target is None else processor.apply_chat_template(
        [*messages, {'role': 'assistant', 'content': target}], tokenize=False, add_generation_prompt=False)
    if not full.startswith(prefix):
        raise ValueError('native training template does not preserve the generation prefix')
    encoded = processor(text=[full], images=load_images(image_paths, image_cache), return_tensors='pt',
                        crop_to_patches=False, size={'height': 448, 'width': 448},
                        add_special_tokens=False)
    pixels = encoded['pixel_values']
    if tuple(pixels.shape) != (len(image_paths), 3, 448, 448):
        raise ValueError(f'one 448x448 tile per image required; observed {tuple(pixels.shape)}')
    ids = encoded['input_ids'][0].tolist()
    image_id = processor.image_token_id
    if ids.count(image_id) != len(image_paths) * processor.image_seq_length:
        raise ValueError('native image expansion does not match one tile per image')
    expanded = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if tokenizer.encode(expanded, add_special_tokens=False) != ids:
        raise ValueError('native token stream is not losslessly decodable; cannot derive spans')
    suffix = full[len(prefix):]
    if not expanded.endswith(suffix):
        raise ValueError('native preprocessing altered assistant target')
    start_char = len(expanded) - len(suffix) if target is not None else len(expanded)
    end_char = start_char
    if target is not None:
        eos = tokenizer.eos_token
        if not eos or not suffix.startswith(target + eos) or suffix[len(target + eos):].strip():
            raise ValueError('unsupported native assistant terminator layout')
        end_char = start_char + len(target + eos)
    char_boundaries = []
    if prompt is not None:
        user_prefix = expanded[:start_char]
        if user_prefix.count(prompt.text) != 1:
            raise ValueError('cannot locate unique semantic user text in native expansion')
        offset = user_prefix.index(prompt.text)
        char_boundaries = [offset + b for b in prompt.boundaries]
    # Tokenize text segments on either side of each continuous-vector position.
    all_ids, offsets, token_boundaries, previous = [], [], [], 0
    for boundary in [*char_boundaries, len(expanded)]:
        part = tokenizer(expanded[previous:boundary], add_special_tokens=False, return_offsets_mapping=True)
        all_ids.extend(part['input_ids'])
        offsets.extend((a + previous, b + previous) for a, b in part['offset_mapping'])
        if boundary in char_boundaries:
            token_boundaries.append(len(all_ids))
        previous = boundary
    if tokenizer.decode(all_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) != expanded:
        raise ValueError('state-boundary tokenization changed native text')
    input_ids = torch.tensor([all_ids], dtype=torch.long)
    labels = torch.full_like(input_ids, -100)
    supervised = []
    for i, (a, b) in enumerate(offsets):
        if a < start_char < b or a < end_char < b:
            raise ValueError('assistant span crosses a token boundary')
        if target is not None and start_char <= a < b <= end_char:
            labels[0, i] = input_ids[0, i]
            supervised.append(i)
    if target is not None:
        decoded = tokenizer.decode([all_ids[i] for i in supervised], skip_special_tokens=False,
                                   clean_up_tokenization_spaces=False)
        if decoded != target + tokenizer.eos_token:
            raise ValueError('supervised response differs from target + native EOS')
    if all_ids.count(image_id) != ids.count(image_id):
        raise ValueError('state tokenization changed native image positions')
    inputs = {'input_ids': input_ids, 'attention_mask': torch.ones_like(input_ids),
              'labels': labels, 'pixel_values': pixels}
    return NativeBatch(inputs, tuple(token_boundaries), supervised[0] if supervised else len(all_ids),
                       len(supervised), expanded, tuple(image_paths),
                       tuple((offsets[i][0] - start_char, offsets[i][1] - start_char) for i in supervised))


def collate_where(processor, episode, end_fix_id, context_limit=8192, *, teacher_forcing=True, config=None, image_cache=None):
    token = config['where']['end_fix_token'] if config else '<END_FIX>'
    if config:
        ctx = config['where']['context']
        if len(episode.supports) > ctx['max_k'] or len(episode.supports) + 1 > ctx['max_images_per_episode']:
            raise WhereContextOverflowError('episode exceeds configured support/image capacity')
    messages, image_paths = build_where_conversation(episode, token)
    batch = collate_native(processor, messages, image_paths,
                           serialize_xyd_record(episode.query, token) if teacher_forcing else None, image_cache=image_cache)
    if batch.inputs['input_ids'].shape[1] > context_limit:
        raise WhereContextOverflowError(f'{episode.query.record_id}: complete WHERE episode exceeds {context_limit}; reject/resample, never truncate')
    if teacher_forcing:
        count = int((batch.inputs['labels'] == end_fix_id).sum())
        if count != len(episode.query.x_px):
            raise ValueError('supervised query END_FIX count differs from N')
        if torch.any(batch.inputs['labels'][:, :batch.response_start] != -100):
            raise ValueError('support/query context labels must be ignored')
    return batch


def to_model_device(inputs, model):
    embedding = model.get_input_embeddings().weight
    return {k: v.to(device=embedding.device, dtype=embedding.dtype if v.is_floating_point() else v.dtype, non_blocking=True)
            for k, v in inputs.items()}


@dataclass
class WhereBatch:
    inputs: dict  # text [B,Lmax], pixels [sum_b(K_b+1),3,448,448]
    samples: tuple[NativeBatch, ...]
    episodes: tuple
    metadata: tuple[dict, ...]
    image_cache: dict


def pack_where_batch(processor, episodes, samples, image_cache=None):
    """Right-pad text; flatten images in sample, support, query order."""
    from torch.nn.utils.rnn import pad_sequence
    if not episodes or len(episodes) != len(samples):
        raise ValueError('episode/native batch counts differ or are empty')
    inputs = {key: pad_sequence([s.inputs[key][0] for s in samples], batch_first=True,
                               padding_value=value) for key, value in (
        ('input_ids', processor.tokenizer.pad_token_id), ('attention_mask', 0), ('labels', -100))}
    inputs['pixel_values'] = torch.cat([s.inputs['pixel_values'] for s in samples])
    metadata, image_offset = [], 0
    for b, (episode, sample) in enumerate(zip(episodes, samples)):
        paths = tuple(r.image_path for r in (*episode.supports, episode.query))
        count = len(paths)
        if sample.image_paths != paths or len(sample.inputs['pixel_values']) != count:
            raise ValueError(f'sample {b}: support/query image ordering or count differs')
        placeholders = int((sample.inputs['input_ids'] == processor.image_token_id).sum())
        if placeholders != count * processor.image_seq_length:
            raise ValueError(f'sample {b}: image placeholders do not match feature groups')
        metadata.append(dict(record_id=episode.query.record_id, K=len(episode.supports),
            fixation_count=len(episode.query.x_px), where_length=sample.inputs['input_ids'].shape[1],
            response_start=sample.response_start, response_length=sample.response_length,
            supervised_where_tokens=int((sample.inputs['labels'][:, 1:] != -100).sum()),
            image_count=count, image_paths=paths, image_offset=image_offset,
            support_image_indices=tuple(range(image_offset, image_offset+count-1)),
            query_image_index=image_offset+count-1, semantic_target=episode.query.semantic))
        image_offset += count
    if image_offset != len(inputs['pixel_values']):
        raise ValueError('flattened image count mismatch')
    return WhereBatch(inputs, tuple(samples), tuple(episodes), tuple(metadata),
                      image_cache if image_cache is not None else {})


def collate_where_batch(processor, episodes, end_fix_id, context_limit=8192, *, config=None):
    cache = {}
    samples = [collate_where(processor, e, end_fix_id, context_limit, config=config,
                             image_cache=cache) for e in episodes]
    return pack_where_batch(processor, episodes, samples, cache)
