"""Exercise actual builder, dataset, DataLoader, and evaluator definitions.

Image decoding/transforms and UserEmbeddingNet are placeholders because the
external backbone runtime is unavailable. JSON, embeddings, support selection,
tokenization, batching, and export aggregation use the production code.
Run together with the original suite using unittest discovery: test_air*.py.
"""

import ast
import json
import os
import pickle
import random
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from test_air_contract import (
    ROOT, RGBPlaceholder, air_config, contract_namespace, load_definitions,
    source_record, torch,
)


@unittest.skipUnless(torch is not None, 'builder/evaluator integration checks require PyTorch')
class AirModeIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.namespace = contract_namespace()
        self.config = air_config(self.namespace)
        self.config.Data.fewshot_subject = [19, 18]
        self.config.Data.num_fewshot = 1
        self.config.Train.n_workers = 0
        self.temporary = tempfile.TemporaryDirectory(prefix='air_modes_')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / 'processed_data').mkdir()
        (self.root / 'stimuli').mkdir()
        (self.root / 'stimuli/one.jpg').touch()
        (self.root / 'export').mkdir()
        self.namespace.update(
            torch=torch, Dataset=torch.utils.data.Dataset,
            DataLoader=torch.utils.data.DataLoader, join=os.path.join,
            T=SimpleNamespace(ToTensor=lambda: None),
            Image=SimpleNamespace(open=lambda path: RGBPlaceholder()),
            time=time, pickle=pickle, tqdm=lambda iterable: iterable)
        self.namespace['transforms'].Compose = lambda operations: lambda image: torch.zeros(3, 384, 512)
        load_definitions('common/data.py', self.namespace, {'Siamese_Triplet_Gaze'})
        load_definitions('common/utils.py', self.namespace, {'transform_fixations'})
        load_definitions('src/builder.py', self.namespace, {'build'})
        load_definitions('src/eval_user.py', self.namespace,
                         {'evaluate_user_siamese', 'compute_average_accuracy'})

        class NonzeroEmbeddingModel(torch.nn.Module):
            def __init__(self, params, hidden_dim, **kwargs):
                super().__init__()
                self.placeholder_weight = torch.nn.Parameter(torch.zeros(1))
                self.embedding_dim = hidden_dim
                self.num_subjects = params.num_subjects
                self.seen_values = []

            def forward(self, image, coarse, padding, high, duration, task_emb):
                self.seen_values.extend(task_emb[:, 0].tolist())
                return {
                    'user_emb': task_emb[:, :self.embedding_dim],
                    'pred_subject_id': torch.zeros(len(image), self.num_subjects),
                }

        self.namespace['UserEmbeddingNet'] = NonzeroEmbeddingModel
        self.write_records()

    def write_records(self, train_count=1, subjects=(18, 19)):
        embeddings = {}
        for split, count in (('train', train_count), ('validation', 2), ('test', 1)):
            records = []
            for subject in subjects:
                for index in range(count):
                    question = '{}-{}-{}'.format(split, subject, index)
                    records.append(source_record(subject, question))
                    embeddings[question] = np.full(768, subject, dtype=np.float32)
            path = self.root / 'processed_data' / ('AiR_fixations_' + split + '.json')
            path.write_text(json.dumps(records), encoding='utf-8')
        np.save(self.root / 'embeddings.npy', embeddings)

    def build(self, is_eval):
        return self.namespace['build'](self.config, str(self.root), 'cpu', is_eval=is_eval)

    def export(self, model, loader):
        self.namespace['evaluate_user_siamese'](
            0, model, 'cpu', loader, self.config, str(self.root / 'export'))
        return torch.load(self.root / 'export/fewshot_user_embedding_1.pt', map_location='cpu')

    def test_support_export_keeps_partial_batch_and_exports_requested_rows(self):
        model, optimizer, loader, validation, weight, step = self.build(is_eval=True)
        self.assertLess(len(loader.dataset), loader.batch_size)
        self.assertFalse(loader.drop_last)
        self.assertEqual(len(loader), 1)
        self.assertEqual(len(loader.dataset), 2)
        for index in range(len(loader.dataset)):
            sample = loader.dataset[index]
            self.assertIs(sample['anchor'], sample['positive'])
            self.assertIs(sample['anchor'], sample['negative'])
        exported = self.export(model, loader)
        self.assertCountEqual(model.seen_values, [19., 18.])
        self.assertEqual(exported.shape, (20, 384))
        torch.testing.assert_close(exported[0], torch.full((384,), 19.))
        torch.testing.assert_close(exported[1], torch.full((384,), 18.))
        self.assertEqual(torch.count_nonzero(exported[2:]).item(), 0)
        for record in validation.dataset.fix_labels:
            raw_subject = int(record['question_id'].split('-')[1])
            self.assertEqual(record['subject_id'], {19: 0, 18: 1}[raw_subject])

    def test_single_subject_one_shot_export_succeeds(self):
        self.config.Data.fewshot_subject = [19]
        self.write_records(subjects=(19,))
        model, optimizer, loader, validation, weight, step = self.build(is_eval=True)
        self.assertEqual(len(loader), 1)
        sample = loader.dataset[0]
        self.assertIs(sample['anchor'], sample['positive'])
        self.assertIs(sample['anchor'], sample['negative'])
        exported = self.export(model, loader)
        self.assertEqual(model.seen_values, [19.])
        torch.testing.assert_close(exported[0], torch.full((384,), 19.))

    def test_training_one_shot_rejects_missing_positive(self):
        with self.assertRaisesRegex(ValueError, 'AiR-D train anchor.*positive/negative.*subject_id'):
            self.build(is_eval=False)

    def test_training_one_shot_uses_distinct_positive_and_different_subject_negative(self):
        self.write_records(train_count=2)
        self.config.Train.batch_size = 2
        model, optimizer, loader, validation, weight, step = self.build(is_eval=False)
        self.assertTrue(loader.drop_last)
        self.assertFalse(loader.dataset.air_one_shot)
        self.assertEqual(len(loader), 2)
        for batch in loader:
            self.assertTrue(torch.equal(batch['anchor']['subject_id'], batch['positive']['subject_id']))
            self.assertTrue(torch.all(batch['anchor']['subject_id'] != batch['negative']['subject_id']))
            for anchor, positive in zip(batch['anchor']['task_name'], batch['positive']['task_name']):
                self.assertNotEqual(anchor, positive)

    def test_training_one_shot_rejects_missing_negative(self):
        self.config.Data.fewshot_subject = [19]
        self.write_records(train_count=2, subjects=(19,))
        with self.assertRaisesRegex(ValueError, 'at least two subjects'):
            self.build(is_eval=False)

    def test_non_support_air_evaluation_keeps_existing_drop_last(self):
        self.config.Data.fewshot_subject = [-1]
        self.write_records(train_count=2)
        model, optimizer, loader, validation, weight, step = self.build(is_eval=True)
        self.assertTrue(loader.drop_last)
        self.assertFalse(loader.dataset.air_one_shot)

    def test_legacy_builder_drop_last_is_unchanged(self):
        (self.root / 'legacy.json').write_text('[]', encoding='utf-8')
        self.config.Data.fix_path = 'legacy.json'
        self.config.Data.TAP = 'FV'
        legacy_data = torch.utils.data.TensorDataset(torch.ones(2))
        legacy_data.fix_labels = [('one.jpg', 'task', 'freeview', [(1, 2)], 0, True, 0, [100], 'OSIE')] * 2
        self.namespace['process_data'] = lambda *args, **kwargs: {
            'gaze_train': legacy_data, 'gaze_valid': legacy_data,
            'bbox_annos': {}, 'human_cdf': None,
        }
        for name in ('OSIE', 'COCO-Search18', 'COCO-Freeview'):
            self.config.Data.name = name
            for is_eval in (False, True):
                with self.subTest(name=name, is_eval=is_eval):
                    with patch.object(np, 'load', return_value=SimpleNamespace(item=lambda: {})):
                        model, optimizer, loader, validation, weight, step = self.build(is_eval)
                    self.assertTrue(loader.drop_last)

    def test_legacy_sampling_matches_head_for_one_and_multiple_shots(self):
        source = subprocess.check_output(
            ['git', 'show', 'HEAD:isp-senet/SE-Net/common/data.py'], cwd=ROOT, text=True)
        definition = next(node for node in ast.parse(source).body
                          if isinstance(node, ast.ClassDef) and node.name == 'Siamese_Triplet_Gaze')
        baseline_namespace = dict(self.namespace)
        exec(compile(ast.Module(body=[definition], type_ignores=[]), 'baseline_data.py', 'exec'),
             baseline_namespace)
        classes = (baseline_namespace['Siamese_Triplet_Gaze'], self.namespace['Siamese_Triplet_Gaze'])
        labels = [('image-{}.jpg'.format(index), 'task', 'freeview', [(1, 2)], 0, True, subject, [100], 'OSIE')
                  for subject in range(3) for index in range(2)]
        random_state = random.getstate()
        self.addCleanup(random.setstate, random_state)
        for name in ('OSIE', 'COCO-Search18', 'COCO-Freeview'):
            for shots in (1, 2):
                with self.subTest(name=name, num_fewshot=shots):
                    params = SimpleNamespace(name=name, max_traj_length=16, TAP='FV', num_fewshot=shots)
                    outputs = []
                    for dataset_class in classes:
                        with patch.object(np, 'load', return_value=SimpleNamespace(item=lambda: {'task': np.ones(768)})):
                            dataset = dataset_class('.', labels, {}, params, None, {}, 'cpu')
                        dataset.process_data = lambda index: {'index': index, 'subject_id': labels[index][-3]}
                        random.seed(17)
                        outputs.append([dataset[index] for index in range(len(dataset))])
                    self.assertEqual(outputs[0], outputs[1])


if __name__ == '__main__':
    unittest.main()
