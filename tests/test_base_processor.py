"""Optional offline tests against the actual downloaded base processor, no 8B weights."""
import json
from pathlib import Path
import pytest

pytest.importorskip('PIL')
pytest.importorskip('torchvision')
from transformers import AutoProcessor
from jinja2 import Environment
from semgaze.data.schema import normalized_episode_from_dict
from semgaze.where.collator import collate_where, collate_native
from semgaze.where.conversation import image_user
from semgaze.semantic.flat.prompt import build_flat_prompt
from semgaze.semantic.flat.target import build_flat_target

ROOT = Path(__file__).resolve().parents[1]
PROCESSOR = ROOT / '.cache/semgaze-base-processor'


@pytest.mark.skipif(not PROCESSOR.exists(), reason='optional base processor has not been cached locally')
def test_actual_base_vocabulary_template_expansion_and_labels():
    processor = AutoProcessor.from_pretrained(PROCESSOR, local_files_only=True)
    adapter = ROOT / 'DeepGaze-VL/model/visual_search_adapter'
    vocab = json.loads((adapter / 'vocab.json').read_text(encoding='utf-8'))
    vocab.update(json.loads((adapter / 'added_tokens.json').read_text(encoding='utf-8')))
    assert processor.tokenizer.get_vocab() == vocab
    messages = [image_user('support'), {'role': 'assistant', 'content': 'support response'}, image_user('query')]
    template = Environment(trim_blocks=True, lstrip_blocks=True).from_string((adapter / 'chat_template.jinja').read_text())
    for generation in (False, True):
        assert processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=generation) == template.render(
            messages=messages, add_generation_prompt=generation)
    processor.tokenizer.add_tokens(['<END_FIX>'], special_tokens=True)
    ids = processor.tokenizer.encode('<END_FIX>', add_special_tokens=False)
    assert len(ids) == 1
    episode = normalized_episode_from_dict(json.loads((ROOT / 'tests/fixtures/flat_episode.json').read_text()))
    where = collate_where(processor, episode, ids[0])
    assert tuple(where.inputs['pixel_values'].shape) == (2, 3, 448, 448)
    assert (where.inputs['input_ids'] == processor.image_token_id).sum() == 512
    assert (where.inputs['labels'] == ids[0]).sum() == 4
    assert (where.inputs['labels'][:, :where.response_start] == -100).all()
    prompt = build_flat_prompt(episode.query)
    semantic = collate_native(processor, [image_user(prompt.text)], [episode.query.image_path],
                              build_flat_target(episode.query.semantic), prompt=prompt)
    assert tuple(semantic.inputs['pixel_values'].shape) == (1, 3, 448, 448)
    assert len(semantic.boundaries) == 4
    assert (semantic.inputs['input_ids'] == processor.image_token_id).sum() == 256
    assert not (semantic.inputs['input_ids'] == ids[0]).any()
