"""Isolated AiR contract tests without importing the external image backbone.

Run with: python -B -m unittest discover -s tests -p test_air_contract.py -v
Production definitions are executed unchanged from their ASTs. Builder datasets
and transforms are capture objects; tensor tests use fixed RGB image tensors.
Adapter tests require NumPy/Pillow; tensor tests require NumPy/PyTorch. These
checks do not establish backbone, full-forward, or export runtime correctness.
"""

import ast
import json
import os
import random
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None

try:
    import torch
except ImportError:
    torch = None


ROOT = Path(__file__).resolve().parents[1]


def load_definitions(relative, namespace, names=None):
    path = ROOT / relative
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    definitions = [node for node in tree.body
                   if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                   and (names is None or node.name in names)]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), 'exec'), namespace)


class DatasetCapture:
    def __init__(self, root_dir, fix_labels, *args, **kwargs):
        self.fix_labels = fix_labels


def contract_namespace():
    namespace = dict(os=os, json=json, np=np, Image=Image,
                     random=random, defaultdict=defaultdict)
    load_definitions('common/config.py', namespace)
    load_definitions('common/utils.py', namespace, {'select_fewshot_subject'})
    load_definitions('common/dataset.py', namespace)
    load_definitions('src/builder.py', namespace,
                     {'_air_split_path', '_load_air_split', '_build_air_datasets'})
    namespace['Siamese_Triplet_Gaze'] = DatasetCapture
    namespace['transforms'] = SimpleNamespace(
        Compose=lambda operations: operations, Resize=lambda **kwargs: None,
        ToTensor=lambda: None, Normalize=lambda **kwargs: None)
    return namespace


def air_config(namespace):
    return namespace['JsonConfig'](str(ROOT / 'configs/air_useremb.json'))


def source_record(subject=0, question='q0', image='one.jpg', length=3):
    return dict(subject_idx=subject, question_id=question, image_id=image,
                width=100, height=50, length=length,
                X=[float(index) for index in range(length)],
                Y=[float(index * 2) for index in range(length)],
                T_start=[index * 200 for index in range(length)],
                T_end=[index * 200 + 123 for index in range(length)])


class ConfigAndSplitTests(unittest.TestCase):
    def setUp(self):
        self.namespace = contract_namespace()
        self.config = air_config(self.namespace)

    def test_all_existing_configs_parse(self):
        for path in (ROOT / 'configs').glob('*.json'):
            with self.subTest(path=path.name):
                self.namespace['JsonConfig'](str(path))

    def test_h3_single_json_and_directory_fallbacks_are_rejected(self):
        for split in ('train', 'validation', 'test'):
            del self.config.Data[split + '_fix_path']
        for fallback in ('processed_data/AiR_fixations_train.json', 'processed_data'):
            self.config.Data.fix_path = fallback
            with self.subTest(fallback=fallback), self.assertRaisesRegex(ValueError, 'template'):
                self.namespace['_air_split_path']('.', self.config, 'validation')

    def test_h3_template_resolves_three_distinct_splits(self):
        for split in ('train', 'validation', 'test'):
            del self.config.Data[split + '_fix_path']
        self.config.Data.fix_path = 'processed_data/AiR_fixations_{split}.json'
        paths = [self.namespace['_air_split_path']('.', self.config, split)
                 for split in ('train', 'validation', 'test')]
        self.assertEqual(len(set(paths)), 3)
        self.assertTrue(paths[1].endswith('AiR_fixations_validation.json'))
        self.config.Data.fix_path = '{unknown}/AiR_fixations_{split}.json'
        with self.assertRaisesRegex(ValueError, 'invalid fix_path template'):
            self.namespace['_air_split_path']('.', self.config, 'train')

    def test_h3_duplicate_and_normalized_alias_paths_are_rejected_before_loading(self):
        for alias in (self.config.Data.train_fix_path,
                      'processed_data/../processed_data/AiR_fixations_train.json'):
            self.config.Data.validation_fix_path = alias
            with self.subTest(alias=alias), self.assertRaisesRegex(ValueError, 'distinct'):
                self.namespace['_build_air_datasets'](self.config, '.', 'cpu')

    def test_m1_config_width_is_rejected_before_file_loading(self):
        self.config.Data.task_embedding_dim = 512
        with self.assertRaisesRegex(ValueError, 'UserEmbeddingNet.*768'):
            self.namespace['_build_air_datasets'](self.config, '.', 'cpu')
        with self.assertRaisesRegex(ValueError, 'UserEmbeddingNet.*768'):
            self.namespace['process_air_data']([], {}, {}, '.', self.config)

    def test_m1_validation_width_matches_actual_model_projection(self):
        tree = ast.parse((ROOT / 'src/models.py').read_text(encoding='utf-8'))
        model = next(node for node in tree.body
                     if isinstance(node, ast.ClassDef) and node.name == 'UserEmbeddingNet')
        projection = next(node.value for node in ast.walk(model)
                          if isinstance(node, ast.Assign)
                          and any(isinstance(target, ast.Attribute)
                                  and target.attr == 'task_transform' for target in node.targets))
        expected = ast.literal_eval(projection.args[0])
        self.assertEqual(self.namespace['validate_air_task_embedding_dim'](self.config), expected)


@unittest.skipUnless(Image is not None, 'real adapter checks require Pillow')
class AdapterAndBuilderTests(unittest.TestCase):
    def setUp(self):
        self.namespace = contract_namespace()
        self.config = air_config(self.namespace)
        self.temporary = tempfile.TemporaryDirectory(prefix='air_contract_')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / 'stimuli').mkdir()
        (self.root / 'processed_data').mkdir()
        self.images = ('one.jpg', 'two.jpg', 'three.jpg')
        for image in self.images:
            Image.new('RGB', (100, 50), (25, 80, 90)).save(self.root / 'stimuli' / image)
        self.embeddings = {question: np.full(768, index, dtype=np.float32)
                           for index, question in enumerate(('q0', 'q1', 'q2', 'qv', 'qt'))}
        np.save(self.root / 'embeddings.npy', self.embeddings)
        self.records = {
            'train': [source_record(subject, 'q' + str(index), image)
                      for subject in range(20) for index, image in enumerate(self.images)],
            'validation': [source_record(subject, 'qv', image)
                           for subject in range(20) for image in self.images[:2]],
            'test': [source_record(subject, 'qt') for subject in range(20)],
        }
        self.write_splits()
        self.mapping = {subject: subject for subject in range(20)}
        self.processed_splits = []
        process = self.namespace['process_air_data']

        def capture_process(records, mapping, *args, **kwargs):
            processed = process(records, mapping, *args, **kwargs)
            self.processed_splits.append((records, mapping, processed))
            return processed

        self.namespace['process_air_data'] = capture_process

    def write_splits(self):
        for split, records in self.records.items():
            path = self.root / 'processed_data' / ('AiR_fixations_' + split + '.json')
            path.write_text(json.dumps(records), encoding='utf-8')

    def build(self, is_eval=False):
        return self.namespace['_build_air_datasets'](self.config, str(self.root), 'cpu', is_eval=is_eval)

    def process(self, records, embeddings=None):
        return self.namespace['process_air_data'](
            records, self.mapping, self.embeddings if embeddings is None else embeddings,
            str(self.root), self.config)

    def assert_shared_mapping(self, expected):
        self.assertEqual(len(self.processed_splits), 3)
        shared = self.processed_splits[0][1]
        self.assertEqual(shared, expected)
        for records, mapping, processed in self.processed_splits:
            self.assertIs(mapping, shared)
            for record, sample in zip(records, processed):
                self.assertEqual(sample['subject_id'], expected[record['subject_idx']])

    def test_adapter_identity_coordinates_duration_order_and_full_scanpath(self):
        processed = self.process([source_record(3), source_record(4, 'q1', length=20)])
        self.assertEqual(len(processed), 2)
        self.assertEqual(processed[0]['subject_id'], 3)
        np.testing.assert_allclose(processed[0]['fixations'],
                                   [[0, 0], [5.12, 15.36], [10.24, 30.72]], rtol=1e-6)
        self.assertEqual(processed[0]['duration'], [123., 123., 123.])
        self.assertEqual(processed[1]['scanpath_length'], 16)
        self.assertEqual(len(processed[1]['fixations']), 16)
        self.assertEqual(processed[0]['image_path'], str((self.root / 'stimuli/one.jpg').resolve()))
        different_size = source_record()
        different_size.update(width=200, height=100)
        resized = self.process([different_size])[0]
        np.testing.assert_allclose(resized['fixations'], np.array(processed[0]['fixations']) / 2)

    def test_saved_question_embeddings_are_selected_by_question_id(self):
        embeddings = self.namespace['load_air_question_embeddings'](self.root / 'embeddings.npy')
        samples = self.process([source_record(question='q0'), source_record(question='q1')], embeddings)
        np.testing.assert_array_equal(samples[0]['task_emb'], embeddings['q0'])
        np.testing.assert_array_equal(samples[1]['task_emb'], embeddings['q1'])
        self.assertFalse(np.array_equal(samples[0]['task_emb'], samples[1]['task_emb']))

    def test_validation_rejects_invalid_records(self):
        changes = ({'Y': [1]}, {'T_end': [1]}, {'length': 2}, {'width': 0},
                   {'T_end': [-1, 323, 523]}, {'question_id': 'missing'},
                   {'image_id': 'missing.jpg'}, {'subject_idx': 99},
                   {'question_id': None})
        for change in changes:
            record = source_record()
            record.update(change)
            with self.subTest(change=change), self.assertRaises((ValueError, KeyError, FileNotFoundError)):
                self.process([record])

    def test_explicit_splits_and_default_subject_identity(self):
        dataset = self.build()
        self.assertEqual({sample['question_id'] for sample in dataset['gaze_train'].fix_labels},
                         {'q0', 'q1', 'q2'})
        self.assertEqual({sample['question_id'] for sample in dataset['gaze_valid'].fix_labels}, {'qv'})
        self.assert_shared_mapping(self.mapping)

    def test_template_split_loading(self):
        for split in self.records:
            del self.config.Data[split + '_fix_path']
        self.config.Data.fix_path = 'processed_data/AiR_fixations_{split}.json'
        dataset = self.build()
        self.assertEqual({sample['question_id'] for sample in dataset['gaze_valid'].fix_labels}, {'qv'})

    def test_h2_exclusion_precedes_capacity_check_and_mapping_is_shared(self):
        self.config.Data.subject = [1, 3]
        self.config.Data.num_subjects = 18
        self.build()
        retained = [subject for subject in range(20) if subject not in (1, 3)]
        self.assert_shared_mapping({subject: mapped for mapped, subject in enumerate(retained)})

    def test_h2_retained_subject_capacity_is_still_checked(self):
        self.config.Data.subject = [18, 19]
        self.config.Data.num_subjects = 17
        with self.assertRaisesRegex(ValueError, '18 subjects.*17'):
            self.build()

    def test_h1_fewshot_uses_requested_order_and_shared_image_support(self):
        self.config.Data.fewshot_subject = [19, 18]
        self.config.Data.num_fewshot = 2
        self.config.Data.num_subjects = 2
        with patch.object(random, 'shuffle', side_effect=lambda names: names.sort()) as shuffle:
            dataset = self.build()
        shuffle.assert_called_once()
        samples = dataset['gaze_train'].fix_labels
        self.assertEqual(len(samples), 4)
        for subject in (0, 1):
            self.assertEqual({sample['image_id'] for sample in samples if sample['subject_id'] == subject},
                             {'one.jpg', 'three.jpg'})
        self.assert_shared_mapping({19: 0, 18: 1})
        self.assertEqual({record['subject_idx'] for record in self.records['train']}, set(range(20)))

    def test_h1_one_shot_selects_one_image_and_all_its_trials(self):
        self.config.Data.fewshot_subject = [18, 19]
        self.config.Data.num_fewshot = 1
        self.records['train'].append(source_record(18, 'q1', 'one.jpg'))
        self.write_splits()
        with patch.object(random, 'shuffle', side_effect=lambda names: names.sort()):
            samples = self.build(is_eval=True)['gaze_train'].fix_labels
        self.assertEqual(len(samples), 3)
        self.assertEqual({sample['image_id'] for sample in samples}, {'one.jpg'})
        self.assert_shared_mapping({18: 0, 19: 1})

    def test_h1_single_subject_one_shot_builder(self):
        self.config.Data.fewshot_subject = [19]
        self.config.Data.num_fewshot = 1
        self.config.Data.num_subjects = 1
        self.assertEqual(len(self.build(is_eval=True)['gaze_train'].fix_labels), 1)
        self.assert_shared_mapping({19: 0})

    def test_h1_missing_support_fails_before_dataset_construction(self):
        self.config.Data.fewshot_subject = [18, 19]
        self.config.Data.num_fewshot = 1
        self.records['train'] = [record for record in self.records['train'] if record['subject_idx'] != 19]
        self.write_splits()
        with self.assertRaisesRegex(ValueError, 'no trials.*19'):
            self.build()

    def test_h1_excluded_fewshot_subject_is_not_reintroduced(self):
        self.config.Data.subject = [19]
        self.config.Data.fewshot_subject = [18, 19]
        with self.assertRaisesRegex(ValueError, 'retained subject_idx'):
            self.build()

    def test_m1_matching_bad_config_and_embedding_are_rejected(self):
        self.config.Data.task_embedding_dim = 512
        with self.assertRaisesRegex(ValueError, 'UserEmbeddingNet.*768'):
            self.process([source_record()], {'q0': np.ones(512)})

    def test_m1_actual_embedding_must_match_projection_without_config_override(self):
        del self.config.Data['task_embedding_dim']
        with self.assertRaisesRegex(ValueError, '512 != expected 768'):
            self.process([source_record()], {'q0': np.ones(512)})

    def test_empty_train_and_validation_are_rejected(self):
        for split in ('train', 'validation'):
            original = self.records[split]
            self.records[split] = []
            self.write_splits()
            with self.subTest(split=split), self.assertRaisesRegex(ValueError, split + ' split is empty'):
                self.build()
            self.records[split] = original


class RGBPlaceholder:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def convert(self, mode):
        if mode != 'RGB':
            raise AssertionError(mode)
        return self


@unittest.skipUnless(torch is not None, 'tensor checks require PyTorch')
class TripletAndBatchTests(unittest.TestCase):
    def setUp(self):
        self.namespace = contract_namespace()
        self.config = air_config(self.namespace)
        self.namespace.update(torch=torch, Dataset=torch.utils.data.Dataset,
                              T=SimpleNamespace(ToTensor=lambda: None),
                              Image=SimpleNamespace(open=lambda path: RGBPlaceholder()))
        load_definitions('common/data.py', self.namespace, {'Siamese_Triplet_Gaze'})
        load_definitions('common/utils.py', self.namespace, {'transform_fixations'})
        load_definitions('train.py', self.namespace, {'compute_output'})
        self.records = [dict(subject_id=subject, image_id='one.jpg', img_name='one.jpg',
                             image_path='one.jpg', question_id=question,
                             fixations=[[0., 0.], [5.12, 15.36], [10.24, 30.72]],
                             duration=[123., 123., 123.], scanpath_length=3,
                             task_emb=np.full(768, index, dtype=np.float32))
                        for subject in (0, 1) for index, question in enumerate(('q0', 'q1'))]

    def dataset(self, records=None, support_export=False):
        return self.namespace['Siamese_Triplet_Gaze'](
            '.', self.records if records is None else records, {}, self.config.Data,
            lambda image: torch.zeros(3, 384, 512), {}, 'cpu', air_split='train',
            air_support_export=support_export)

    def test_normal_triplets_preserve_subject_and_distinct_trial_semantics(self):
        dataset = self.dataset()
        for index in range(len(dataset)):
            sample = dataset[index]
            self.assertEqual(set(sample), {'anchor', 'positive', 'negative'})
            self.assertEqual(sample['anchor']['subject_id'], sample['positive']['subject_id'])
            self.assertNotEqual(sample['anchor']['subject_id'], sample['negative']['subject_id'])
            self.assertNotEqual(sample['anchor']['task_name'], sample['positive']['task_name'])

    def test_missing_triplet_pools_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'positive/negative'):
            self.dataset(self.records[:1] + self.records[2:])
        with self.assertRaisesRegex(ValueError, 'at least two subjects'):
            self.dataset(self.records[:2])

    def test_h1_one_shot_uses_legacy_anchor_only_triplet(self):
        self.config.Data.fewshot_subject = [18]
        self.config.Data.num_fewshot = 1
        sample = self.dataset(self.records[:1], support_export=True)[0]
        self.assertIs(sample['anchor'], sample['positive'])
        self.assertIs(sample['anchor'], sample['negative'])
        batch = next(iter(torch.utils.data.DataLoader(
            self.dataset(self.records[:1], support_export=True), batch_size=1)))
        self.assertEqual(batch['anchor']['subject_id'].tolist(), [0])

    def test_one_shot_exception_requires_active_fewshot_selection(self):
        self.config.Data.num_fewshot = 1
        with self.assertRaisesRegex(ValueError, 'at least two subjects'):
            self.dataset(self.records[:1], support_export=True)

    def test_dataloader_fields_padding_and_question_conditioning(self):
        batch = next(iter(torch.utils.data.DataLoader(self.dataset(), batch_size=2, num_workers=0)))
        for sample in batch.values():
            self.assertEqual(sample['true_state'].shape, (2, 3, 384, 512))
            self.assertEqual(sample['normalized_fixations'].shape, (2, 16, 2))
            self.assertEqual(sample['task_emb'].shape, (2, 768))
            self.assertEqual(sample['subject_id'].dtype, torch.int64)
            self.assertEqual(sample['duration'].shape, (2, 16))
            self.assertEqual(sample['is_padding'].shape, (2, 16))
            self.assertTrue(torch.all(sample['is_padding'][:, :3] == 0))
            self.assertTrue(torch.all(sample['is_padding'][:, 3:] == 1))
            self.assertTrue(torch.all(sample['duration'][:, :3] == 123))
            self.assertTrue(torch.all(sample['duration'][:, 3:] == 0))
            torch.testing.assert_close(sample['normalized_fixations'][:, 3:],
                                       sample['normalized_fixations'][:, 2:3].expand(-1, 13, -1))
        self.assertFalse(torch.equal(batch['anchor']['task_emb'][0], batch['anchor']['task_emb'][1]))

    def test_compute_output_tokenization_contract_with_model_spy(self):
        batch = next(iter(torch.utils.data.DataLoader(self.dataset(), batch_size=2)))
        self.namespace.update(device=torch.device('cpu'), hparams=self.config)

        def model_spy(image, coarse, padding, high, duration, task):
            self.assertEqual(coarse.shape, (2, 16))
            self.assertEqual(high.shape, (2, 16))
            self.assertEqual(padding.shape, (2, 16))
            self.assertEqual(coarse.dtype, torch.int64)
            self.assertEqual(high.dtype, torch.int64)
            self.assertTrue(torch.all(coarse[:, :3] > 0))
            self.assertTrue(torch.all(high[:, :3] > 0))
            self.assertTrue(torch.all(coarse[:, 3:] == 0))
            self.assertTrue(torch.all(high[:, 3:] == 0))
            self.assertLessEqual(coarse.max().item(), (512 // 32) * (384 // 32))
            self.assertLessEqual(high.max().item(), (512 // 4) * (384 // 4))
            self.assertEqual(task.shape, (2, 768))
            return {'contract_checked': True}

        output = self.namespace['compute_output'](batch['anchor'], model_spy, self.config.Data)
        self.assertTrue(output['contract_checked'])


if __name__ == '__main__':
    unittest.main()
