"""Small real HF InternVL + PEFT graph tests, never a released-weight smoke substitute."""
import json
from pathlib import Path
import pytest
import torch

pytest.importorskip('peft')
pytest.importorskip('PIL')
pytest.importorskip('torchvision')

from transformers import (PreTrainedTokenizerFast, GotOcr2ImageProcessor, InternVLProcessor,
    InternVLConfig, InternVLVisionConfig, Qwen3Config, InternVLForConditionalGeneration, InternVLVideoProcessor)
from peft import LoraConfig, get_peft_model
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from semgaze.data.schema import normalized_episode_from_dict
from semgaze.model.build import FlatModelBundle, assert_trainable_set
from semgaze.model.config import load_config, resolve_config
from semgaze.model.trainable_tokens import install_trainable_rows
from semgaze.state.projector import build_projector
from semgaze.where.collator import collate_where, to_model_device, WhereContextOverflowError
from semgaze.where.conversation import build_where_conversation
from semgaze.where.forward import forward_where
from semgaze.semantic.flat.prompt import build_flat_prompt
from semgaze.semantic.flat.target import build_flat_target
from semgaze.semantic.flat.forward import forward_flat, prepare_flat_inputs, SemanticContextOverflowError
from semgaze.training.flat_step import run_flat_training_step, make_optimizer

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


@pytest.fixture
def episode():
    return normalized_episode_from_dict(json.loads((ROOT / 'tests/fixtures/flat_episode.json').read_text()))


def tiny_bundle(tmp_path, episode, *, tied=False, bare=False):
    torch.manual_seed(123)
    backend = Tokenizer(models.BPE())
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    specials = ['<|endoftext|>', '<|im_start|>', '<|im_end|>', '<img>', '</img>', '<IMG_CONTEXT>', '<video>', '<END_FIX>']
    messages, _ = build_where_conversation(episode)
    text = [m['content'] if isinstance(m['content'], str) else m['content'][1]['text'] for m in messages]
    text += [build_flat_prompt(episode.query).text, build_flat_target(episode.query.semantic), 'assistant\nuser\n']
    backend.train_from_iterator(text, trainers.BpeTrainer(show_progress=False, vocab_size=400, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), special_tokens=specials))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, eos_token='<|im_end|>', pad_token='<|endoftext|>',
        additional_special_tokens=specials[1:])
    for key, token in {'start_image_token': '<img>', 'end_image_token': '</img>',
                       'context_image_token': '<IMG_CONTEXT>', 'video_token': '<video>'}.items():
        setattr(tokenizer, key, token)
        setattr(tokenizer, key + '_id', tokenizer.convert_tokens_to_ids(token))
    processor = InternVLProcessor(image_processor=GotOcr2ImageProcessor(size={'height': 448, 'width': 448}),
        video_processor=InternVLVideoProcessor(),
        tokenizer=tokenizer, image_seq_length=4,
        chat_template=(ROOT / 'DeepGaze-VL/model/visual_search_adapter/chat_template.jinja').read_text())
    config = InternVLConfig(vision_config=InternVLVisionConfig(hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, image_size=448, patch_size=112),
        text_config=Qwen3Config(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=8,
        max_position_embeddings=8192, tie_word_embeddings=tied,
        pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id),
        image_token_id=tokenizer.context_image_token_id, image_seq_length=4, tie_word_embeddings=tied)
    base = InternVLForConditionalGeneration(config)
    if bare:
        return processor, base
    # Random adapter exists only in this explicitly small architecture test.
    model = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, task_type='CAUSAL_LM',
        target_modules=['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']))
    end_fix = tokenizer.convert_tokens_to_ids('<END_FIX>')
    input_row, output_row, tied = install_trainable_rows(model, end_fix)
    projector = build_projector(32)
    diagnostics = assert_trainable_set(model, projector, input_row, output_row)
    cfg = resolve_config(load_config(ROOT / 'configs/flat_single.yaml'))
    cfg['runtime']['device'] = 'cpu'
    cfg['training']['precision'] = 'fp32'
    return FlatModelBundle(model, processor, projector, end_fix, input_row, output_row, tied,
                           cfg, diagnostics, tmp_path)


@pytest.mark.parametrize('tied', [False, True])
def test_joint_backward_and_old_rows_frozen(tmp_path, episode, tied):
    bundle = tiny_bundle(tmp_path, episode, tied=tied)
    old_input = bundle.model.get_input_embeddings().weight.detach().clone()
    old_output = bundle.model.get_output_embeddings().weight.detach().clone()
    row_before = bundle.input_row.detach().clone()
    result = run_flat_training_step(bundle, episode, optimizer_step=True)
    assert result['projector_output_shape'] == (4, 32)
    assert result['inserted_state_count'] == 4
    assert result['query_end_fix_state_count'] == result['supervised_query_end_fix_count'] == 4
    assert result['supervised_support_end_fix_count'] == 0
    assert torch.equal(old_input, bundle.model.get_input_embeddings().weight)
    assert torch.equal(old_output, bundle.model.get_output_embeddings().weight)
    assert not torch.equal(row_before, bundle.input_row)
    assert (bundle.input_row is bundle.output_row) == tied


def test_semantic_only_gradients_reach_where(tmp_path, episode):
    bundle = tiny_bundle(tmp_path, episode)
    where = forward_where(bundle, episode)
    where.states.retain_grad()
    states = bundle.projector(where.states)
    states.retain_grad()
    loss, _ = forward_flat(bundle, episode.query, states)
    loss.backward()
    for value in (states, where.states, bundle.input_row):
        assert value.grad is not None and torch.isfinite(value.grad).all() and value.grad.ne(0).any()
    assert any(p.grad is not None and p.grad.ne(0).any() for n, p in bundle.model.named_parameters() if 'lora_' in n)


def test_native_mask_and_exact_state_positions(tmp_path, episode):
    bundle = tiny_bundle(tmp_path, episode)
    batch = collate_where(bundle.processor, episode, bundle.end_fix_id)
    assert batch.image_paths == tuple(r.image_path for r in (*episode.supports, episode.query))
    labels, ids = batch.inputs['labels'], batch.inputs['input_ids']
    assert (labels[:, :batch.response_start] == -100).all()
    assert ((ids == bundle.end_fix_id) & (labels != -100)).sum() == 4
    # Prompt examples also contain END_FIX; all such occurrences must be masked.
    assert ((ids == bundle.end_fix_id) & (labels == -100)).sum() > 3
    fake_states = torch.randn(4, 32, requires_grad=True)
    inputs, positions, native = prepare_flat_inputs(bundle, episode.query, fake_states)
    assert inputs['inputs_embeds'].shape[1] == native.inputs['input_ids'].shape[1] + 4
    assert inputs['labels'].shape == inputs['attention_mask'].shape == inputs['inputs_embeds'].shape[:2]
    for t, p in enumerate(positions):
        assert torch.equal(inputs['inputs_embeds'][0, p], fake_states[t])
        assert inputs['labels'][0, p] == -100 and inputs['attention_mask'][0, p] == 1
    output = bundle.model(**inputs, use_cache=False)
    output.loss.backward()
    assert fake_states.grad is not None and fake_states.grad.ne(0).any()
    with pytest.raises(WhereContextOverflowError):
        collate_where(bundle.processor, episode, bundle.end_fix_id, context_limit=8)
    bundle.context_limit = 8
    with pytest.raises(SemanticContextOverflowError) as error:
        prepare_flat_inputs(bundle, episode.query, fake_states)
    assert error.value.diagnostics['number_of_inserted_states'] == 4


def test_image_conditioning_is_preserved(tmp_path, episode):
    bundle = tiny_bundle(tmp_path, episode)
    bundle.model.eval()
    states = torch.randn(4, 32)
    inputs, _, _ = prepare_flat_inputs(bundle, episode.query, states)
    with torch.no_grad():
        original = bundle.model(**inputs).logits
        inputs['pixel_values'] = torch.zeros_like(inputs['pixel_values'])
        altered = bundle.model(**inputs).logits
    assert not torch.equal(original, altered)


def test_native_generation_paths(tmp_path, episode):
    from semgaze.evaluation.flat import evaluate_flat_episode
    from semgaze.where.generation import generate_where
    bundle = tiny_bundle(tmp_path, episode)
    result = evaluate_flat_episode(bundle, episode, generation_budget=3)
    assert result['state_source'] == 'teacher_forced_GT_XYD'
    assert result['inserted_state_count'] == 4
    assert bundle.input_row.grad is None
    where = generate_where(bundle, episode)
    assert where['requested_fixation_count'] == 4
    assert 'under_generated' in where


@pytest.mark.parametrize('tied', [False, True])
def test_checkpoint_trainables_round_trip(tmp_path, episode, tied):
    from semgaze.model.checkpoint import save_checkpoint, restore_checkpoint_state
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    bundle = tiny_bundle(tmp_path, episode, tied=tied)
    bundle.optimizer = make_optimizer(bundle)
    bundle.scheduler = torch.optim.lr_scheduler.LambdaLR(bundle.optimizer, lambda _: 1.0)
    run_flat_training_step(bundle, episode, optimizer_step=True)
    checkpoint = tmp_path / 'saved'
    save_checkpoint(bundle, checkpoint, split_manifest_identity='fixture-v1', step=1)
    fresh = tiny_bundle(tmp_path / 'fresh', episode, tied=tied)
    fresh.optimizer = make_optimizer(fresh)
    fresh.scheduler = torch.optim.lr_scheduler.LambdaLR(fresh.optimizer, lambda _: 1.0)
    set_peft_model_state_dict(fresh.model, load_file(str(checkpoint / 'adapter/adapter_model.safetensors')))
    assert restore_checkpoint_state(fresh, checkpoint, split_manifest_identity='fixture-v1', resume_optimizer=True) == 1
    assert torch.equal(bundle.input_row, fresh.input_row)
    assert torch.equal(bundle.output_row, fresh.output_row)
    for a, b in zip(bundle.projector.parameters(), fresh.projector.parameters()):
        assert torch.equal(a, b)
    bundle.model.eval()
    fresh.model.eval()
    with torch.no_grad():
        left = forward_where(bundle, episode)
        right = forward_where(fresh, episode)
    assert torch.equal(left.states, right.states)
    assert torch.equal(left.loss_where, right.loss_where)
    with pytest.raises(ValueError, match='manifest identity'):
        restore_checkpoint_state(fresh, checkpoint, split_manifest_identity='different')


def test_checkpoint_loader_rebuilds_peft_and_rows(tmp_path, episode, monkeypatch):
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from semgaze.model.checkpoint import save_checkpoint, load_checkpoint_bundle
    bundle = tiny_bundle(tmp_path, episode)
    run_flat_training_step(bundle, episode, optimizer_step=True)
    checkpoint = tmp_path / 'checkpoint'
    save_checkpoint(bundle, checkpoint, split_manifest_identity='fixture', step=1)
    processor, base = tiny_bundle(tmp_path / 'new', episode, bare=True)
    # Avoid downloading 8B weights in this integration test; everything after the
    # base/processor boundary, including PEFT.from_pretrained, is the real loader.
    monkeypatch.setattr(AutoProcessor, 'from_pretrained', lambda *args, **kwargs: processor)
    monkeypatch.setattr(AutoModelForImageTextToText, 'from_pretrained', lambda *args, **kwargs: base)
    loaded = load_checkpoint_bundle(checkpoint, split_manifest_identity='fixture')
    bundle.model.eval()
    loaded.model.eval()
    with torch.no_grad():
        assert torch.equal(forward_where(bundle, episode).states, forward_where(loaded, episode).states)


def test_batched_extractor_reads_post_boundary_and_ignores_supports():
    from semgaze.state.extractor import extract_query_states
    hidden = torch.arange(2 * 6 * 3, dtype=torch.float32).reshape(2, 6, 3).requires_grad_()
    ids = torch.tensor([[7, 1, 7, 2, 7, 3], [7, 1, 2, 7, 3, 0]])
    labels = torch.tensor([[-100, -100, 7, 2, 7, 3], [-100, -100, 2, 7, 3, -100]])
    states, mask, positions = extract_query_states(hidden, ids, labels, 7, [2, 1])
    assert torch.equal(states[0], hidden[0, [2, 4]])
    assert torch.equal(states[1, 0], hidden[1, 3])
    assert mask.tolist() == [[True, True], [True, False]]
    states.sum().backward()
    assert hidden.grad[0, 0].eq(0).all() and hidden.grad[0, 2].eq(1).all()


def test_loaded_adapter_tensor_verification(tmp_path, episode):
    from semgaze.model.build import verify_loaded_adapter, ModelPreflightError
    bundle = tiny_bundle(tmp_path, episode)
    saved = tmp_path / 'adapter'
    bundle.model.save_pretrained(saved, safe_serialization=True, save_embedding_layers=False)
    verify_loaded_adapter(bundle.model, saved)
    parameter = next(p for n, p in bundle.model.named_parameters() if 'lora_' in n)
    with torch.no_grad():
        parameter.add_(1)
    with pytest.raises(ModelPreflightError, match='not loaded exactly'):
        verify_loaded_adapter(bundle.model, saved)


def test_non_reentrant_gradient_checkpointing(tmp_path, episode):
    bundle = tiny_bundle(tmp_path, episode)
    bundle.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    result = run_flat_training_step(bundle, episode)
    assert result['projector_grad_nonzero'] and result['end_fix_input_grad_finite']
