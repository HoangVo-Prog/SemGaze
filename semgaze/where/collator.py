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


def load_images(paths):
    from PIL import Image
    images = []
    for path in paths:
        with Image.open(ROOT / path) as image:
            images.append(image.convert('RGB'))
    return images


def collate_native(processor, messages, image_paths, target=None, *, prompt=None):
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
    encoded = processor(text=[full], images=load_images(image_paths), return_tensors='pt',
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


def collate_where(processor, episode, end_fix_id, context_limit=8192, *, teacher_forcing=True):
    messages, image_paths = build_where_conversation(episode)
    batch = collate_native(processor, messages, image_paths,
                           serialize_xyd_record(episode.query) if teacher_forcing else None)
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
    return {k: v.to(device=embedding.device, dtype=embedding.dtype if v.is_floating_point() else v.dtype)
            for k, v in inputs.items()}
