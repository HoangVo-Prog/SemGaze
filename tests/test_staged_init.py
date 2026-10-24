"""CPU-only graph-level tests; no InternVL weights or dataset are required."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from semgaze.model.staged_init import validate_staged_config, adapt_checkpoint_for_new_run


class FakeBase(nn.Module):
    def __init__(self, input_row, output_row):
        super().__init__()
        self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=4))
        self.model = nn.Module()
        self.model.vision_tower = nn.Linear(1, 1)
        self.model.multi_modal_projector = nn.Linear(1, 1)
        self.model.language_model = nn.Module()
        self.model.language_model.register_parameter('lora_rank_weight', nn.Parameter(torch.tensor([3.0])))
        self.model.register_parameter('end_fix_input', input_row)
        if output_row is not input_row:
            self.model.register_parameter('end_fix_output', output_row)
        for module in (self.model.vision_tower, self.model.multi_modal_projector):
            module.requires_grad_(False)


class FakeModel(nn.Module):
    def __init__(self, input_row, output_row):
        super().__init__()
        self.base = FakeBase(input_row, output_row)

    def get_base_model(self):
        return self.base


class FakeBundle:
    def __init__(self, source, tied=True):
        self.config = source
        self.input_row = nn.Parameter(torch.ones(4), requires_grad=source['where']['use_end_fix_token'])
        self.output_row = self.input_row if tied else nn.Parameter(
            torch.ones(4) * 2, requires_grad=source['where']['use_end_fix_token'])
        self.model = FakeModel(self.input_row, self.output_row)
        self.projector = (nn.Sequential(nn.Linear(4, 4), nn.LayerNorm(4))
                          if source['semantic']['conditioning'] == 'shared_states' else nn.Identity())
        self.diagnostics = {'existing_checkpoint_marker': True}
        self.trainer_history = [{'step': 70}]
        self.context_limit = 8192

    def trainable_parameters(self):
        params = [p for p in self.model.parameters() if p.requires_grad]
        if self.config['training']['lambda_sem'] > 0:
            params.extend(self.projector.parameters())
        return params


def config(end_fix, semantic_weight, conditioning='explicit_scanpath_text'):
    return {
        'experiment': {'protocol_version': 'flat-v1', 'semantic_mode': 'flat_single_output'},
        'model': {'base_model': 'OpenGVLab/InternVL3_5-8B-HF', 'backend': 'huggingface',
                  'lora': {'rank': 8, 'target_modules': ['q_proj']},
                  'freeze': {'vision_tower': True, 'native_multimodal_projector': True}},
        'where': {'mode': 'xyd', 'end_fix_token': '<END_FIX>', 'use_end_fix_token': end_fix,
                  'coordinates': {'grid_size': 100}, 'duration': {'unit': 'ms'},
                  'context': {'max_total_sequence_length': 4096}},
        'state': {'readout_position': 'query_end_fix_token_position' if end_fix else 'last_duration_numeric_token'},
        'semantic': {'conditioning': conditioning},
        'training': {'lambda_where': 1.0, 'lambda_sem': semantic_weight, 'learning_rate': 2e-5},
        'checkpoint': {'save_projector': conditioning == 'shared_states',
                       'save_trainable_end_fix_rows': True, 'save_end_fix_token_id': True,
                       'save_peft_adapter': True, 'save_processor_and_tokenizer': True,
                       'save_resolved_config': True},
    }


class StagedInitializationTests(unittest.TestCase):
    def make_checkpoint(self):
        tmp = tempfile.TemporaryDirectory()
        path = Path(tmp.name)
        (path / 'metadata.json').write_text(json.dumps({'step': 70}))
        self.addCleanup(tmp.cleanup)
        return path

    def test_where_only_to_joint_explicit_preserves_weights_and_is_new_run(self):
        source = config(True, 0)
        target = config(True, 1)
        target['training']['learning_rate'] = 1e-5
        bundle = FakeBundle(source)
        initial_lora = bundle.model.get_base_model().model.language_model.lora_rank_weight.detach().clone()
        initial_row = bundle.input_row.detach().clone()
        adapted = adapt_checkpoint_for_new_run(bundle, target, source_checkpoint=self.make_checkpoint())
        self.assertIs(adapted, bundle)
        self.assertIsInstance(bundle.projector, nn.Identity)
        torch.testing.assert_close(bundle.model.get_base_model().model.language_model.lora_rank_weight, initial_lora)
        torch.testing.assert_close(bundle.input_row, initial_row)
        self.assertEqual(bundle.trainer_history, [])
        self.assertEqual(bundle.config['training']['learning_rate'], 1e-5)
        self.assertEqual(bundle.diagnostics['staged_initialization']['source_step'], 70)
        self.assertTrue(bundle.input_row.requires_grad)
        self.assertFalse(bundle.diagnostics['staged_initialization']['projector_reinitialized'])

    def test_no_end_fix_where_only_to_shared_joint_initializes_projector_and_rows(self):
        source = config(False, 0)
        target = config(True, 1, 'shared_states')
        bundle = FakeBundle(source, tied=False)
        before = bundle.input_row.detach().clone(), bundle.output_row.detach().clone()
        torch.manual_seed(100)
        adapt_checkpoint_for_new_run(bundle, target, source_checkpoint=self.make_checkpoint())
        self.assertIsInstance(bundle.projector, nn.Sequential)
        self.assertTrue(bundle.input_row.requires_grad)
        self.assertTrue(bundle.output_row.requires_grad)
        self.assertEqual(bundle.context_limit, 4096)
        torch.testing.assert_close(bundle.input_row, before[0])
        torch.testing.assert_close(bundle.output_row, before[1])
        self.assertEqual(bundle.diagnostics['projector_trainable_parameter_count'], 28)
        self.assertEqual(bundle.diagnostics['optimizer_trainable_parameter_count'], 1 + 8 + 28)
        self.assertTrue(bundle.diagnostics['staged_initialization']['projector_reinitialized'])
        torch.optim.AdamW(bundle.trainable_parameters(), lr=2e-5)

    def test_no_end_fix_to_no_end_fix_joint_keeps_rows_frozen(self):
        source = config(False, 0)
        target = config(False, 1)
        bundle = FakeBundle(source)
        adapt_checkpoint_for_new_run(bundle, target, source_checkpoint=self.make_checkpoint())
        self.assertFalse(bundle.input_row.requires_grad)
        self.assertIsInstance(bundle.projector, nn.Identity)
        self.assertEqual(bundle.diagnostics['optimizer_trainable_parameter_count'], 1)

    def test_shared_to_explicit_removes_projector(self):
        source = config(True, 1, 'shared_states')
        target = config(True, 1, 'explicit_scanpath_text')
        bundle = FakeBundle(source)
        self.assertIsInstance(bundle.projector, nn.Sequential)
        adapt_checkpoint_for_new_run(bundle, target, source_checkpoint=self.make_checkpoint())
        self.assertIsInstance(bundle.projector, nn.Identity)
        self.assertEqual(bundle.diagnostics['projector_trainable_parameter_count'], 0)
        self.assertTrue(bundle.diagnostics['staged_initialization']['projector_removed'])
        self.assertFalse(bundle.diagnostics['staged_initialization']['projector_reinitialized'])

    def test_end_fix_to_no_end_fix_freezes_rows(self):
        source = config(True, 0)
        target = config(False, 1)
        bundle = FakeBundle(source)
        adapt_checkpoint_for_new_run(bundle, target, source_checkpoint=self.make_checkpoint())
        self.assertFalse(bundle.input_row.requires_grad)
        self.assertFalse(bundle.output_row.requires_grad)
        self.assertEqual(bundle.diagnostics['optimizer_trainable_parameter_count'], 1)

    def test_invalid_lora_and_vocabulary_rejected_before_mutation(self):
        for change in ('lora', 'end_fix_token', 'base_model'):
            with self.subTest(change=change):
                source = config(False, 0)
                target = config(False, 1)
                if change == 'lora':
                    target['model']['lora']['rank'] = 16
                elif change == 'end_fix_token':
                    target['where']['end_fix_token'] = '<OTHER>'
                else:
                    target['model']['base_model'] = 'different'
                bundle = FakeBundle(source)
                with self.assertRaisesRegex(ValueError, 'matching'):
                    adapt_checkpoint_for_new_run(bundle, target, source_checkpoint=self.make_checkpoint())
                self.assertIs(bundle.config, source)

    def test_where_only_shared_source_does_not_reuse_untrained_projector(self):
        source = config(True, 0, 'shared_states')
        target = config(True, 1, 'shared_states')
        bundle = FakeBundle(source)
        original = bundle.projector
        adapt_checkpoint_for_new_run(bundle, target, source_checkpoint=self.make_checkpoint())
        self.assertIsNot(bundle.projector, original)
        self.assertTrue(bundle.diagnostics['staged_initialization']['projector_reinitialized'])

    def test_shared_projector_must_be_saved(self):
        source = config(False, 0)
        target = config(False, 1, 'shared_states')
        target['checkpoint']['save_projector'] = False
        with self.assertRaisesRegex(ValueError, 'save_projector'):
            validate_staged_config(source, target)


if __name__ == '__main__':
    unittest.main()
