#!/usr/bin/env python3
"""Apply the AiR + COCO-Search18 joint-training/evaluation patch to no-attn.

Run from anywhere: python apply_joint_patch.py --repo /path/to/SemGaze
Fail-closed string anchors avoid silently modifying an unrelated revision.
Default is no-write validation; pass --apply to change files.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import difflib
import sys


def replace(text, before, after, label):
    count = text.count(before)
    if count != 1:
        raise ValueError(f'{label}: expected exactly one anchor, found {count}')
    return text.replace(before, after, 1)


def patch_config(text):
    text = replace(text, "        'data.dataset': 'COCO-Search18',\n", '', 'config.dataset hardcap')
    anchor = "    if config['data']['unseen_subjects'] != [7, 8, 9]:\n"
    text = replace(text, anchor, """    if config['data']['dataset'] not in ('COCO-Search18', 'all'):
        raise ValueError('data.dataset must be COCO-Search18 or all (AiR + COCO)')
    air_unseen = config['data'].get('air_unseen_subjects')
    if not isinstance(air_unseen, list) or not air_unseen or any(
            not isinstance(s, str) or not s.strip() for s in air_unseen
    ) or len(set(air_unseen)) != len(air_unseen):
        raise ValueError('data.air_unseen_subjects must contain unique nonempty subject names')
""" + anchor, 'config.air_unseen')
    return text


def patch_schema(text):
    text = replace(text, '    subject: int\n', '    subject: int | str\n', 'schema.subject')
    text = replace(text, '    semantic: NormalizedSemantic\n', "    semantic: NormalizedSemantic\n    dataset: str = 'COCO-Search18'\n", 'schema.dataset')
    text = replace(text,
        '        if type(self.subject) is not int or self.condition not in ("present", "absent"):\n',
        """        valid_coco = (self.dataset == 'COCO-Search18' and type(self.subject) is int
                      and self.condition in ('present', 'absent'))
        valid_air = (self.dataset == 'AiR' and isinstance(self.subject, str)
                     and self.subject.startswith('AiR::') and self.condition == 'vqa')
        if not (valid_coco or valid_air):
""", 'schema.condition')
    text = replace(text,
        '        if any(s.subject != self.query.subject for s in self.supports):\n',
        '        if any(s.subject != self.query.subject or s.dataset != self.query.dataset for s in self.supports):\n',
        'schema.support datasets')
    return text


def patch_read_splits(text):
    text = replace(text,
        "    if 'source_manifests' in manifest or 'source_file' in manifest:\n        from .json_splits import read_current_json_splits\n        return read_current_json_splits(root, manifest, data_config)\n",
        """    if 'source_manifests' in manifest or 'source_file' in manifest:
        if data_config and data_config['dataset'] == 'all':
            from .joint import read_joint_splits
            return read_joint_splits(root, manifest, data_config)
        from .json_splits import read_current_json_splits
        return read_current_json_splits(root, manifest, data_config)
""", 'reader joint dispatch')
    return text


def patch_fewshot(text):
    text = replace(text, '        unseen_subjects = data_config[\'unseen_subjects\']\n',
        """        from .joint import unseen_subject_ids
        unseen_subjects = unseen_subject_ids(data_config)
""", 'sampler unseen')
    text = replace(text,
        "        if set(unseen_subjects) != {7, 8, 9}:\n            raise ValueError(\"COCO unseen subjects must remain {7,8,9}\")\n",
        """        if not {7, 8, 9} <= set(unseen_subjects):
            raise ValueError('COCO unseen subjects must remain {7,8,9}')
""", 'sampler COCO contract')
    start = text.index('def frozen_episode(')
    text = text[:start] + '''def frozen_episode(query, train_by_id, manifest, k, *, draw_id=None, unseen_subjects=None):
    """Dataset-local frozen support lookup; no AiR/COCO cross-contamination."""
    if type(k) is not int or k not in (1, 5, 10):
        raise ValueError('K_eval must be 1/5/10')
    if type(draw_id) is not int:
        raise ValueError('frozen evaluation needs integer draw_id')
    dataset = getattr(query, 'dataset', 'COCO-Search18')
    scoped = manifest.get('datasets', {}).get(dataset, manifest)
    valid_subjects = set(manifest.get('unseen_subject_ids', UNSEEN_SUBJECTS)) if unseen_subjects is None else set(unseen_subjects)
    if query.subject not in valid_subjects:
        raise ValueError('test query is not an unseen subject')
    blocks = scoped['support_draws'].get(str(k), ())
    if draw_id < 0 or draw_id >= len(blocks):
        raise ValueError('missing frozen support draw')
    raw_subject = query.subject.removeprefix('AiR::') if dataset == 'AiR' else str(query.subject)
    name = query.stimulus_id.removeprefix('AiR::') if dataset == 'AiR' else query.stimulus_id
    if name not in scoped['test_stimulus_ids']:
        raise ValueError('test query is outside dataset-local held-out images')
    entries = blocks[draw_id]
    ids = [entry['resolved_record_id_by_subject'][str(raw_subject)] for entry in entries]
    if len(ids) != k or any(rid not in train_by_id for rid in ids):
        raise ValueError('unresolvable frozen supports')
    supports = tuple(train_by_id[rid] for rid in ids)
    for entry, record in zip(entries, supports):
        raw_image = record.stimulus_id.removeprefix('AiR::') if dataset == 'AiR' else record.stimulus_id
        if (record.dataset != dataset or record.subject != query.subject
                or raw_image not in scoped['train_stimulus_ids']
                or entry['image_name'] != raw_image):
            raise ValueError('cross-dataset, cross-subject or invalid support image')
        if dataset == 'COCO-Search18':
            task = entry.get('task', entry.get('unit_id'))
            if task != record.task or entry.get('trial_key') != f'{record.task}::{raw_image}':
                raise ValueError('COCO frozen trial identity mismatch')
        else:
            # AiR support unit is qid, not the question text in record.task.
            # read_joint_splits already verifies qid -> record_id for every draw.
            if entry.get('dataset') != 'AiR':
                raise ValueError('AiR frozen trial dataset mismatch')
    return FlatEpisode(supports, query, draw_id=draw_id)
'''
    return text


def patch_training_entry(text):
    text = replace(text,
        'from semgaze.data.cocosearch18 import read_persisted_splits, CocoSearch18Adapter\n',
        'from semgaze.data.cocosearch18 import read_persisted_splits, CocoSearch18Adapter\nfrom semgaze.data.joint import JointAdapter\n', 'train import')
    text = replace(text,
        "    adapter = CocoSearch18Adapter(ROOT / data['images_root'], annotation_frame=data['annotation_frame'],\n                                  duration_field=data['duration']['source_field'])\n",
        """    adapter_cls = JointAdapter if data['dataset'] == 'all' else CocoSearch18Adapter
    adapter = adapter_cls(ROOT / data['images_root'], annotation_frame=data['annotation_frame'],
                          duration_field=data['duration']['source_field'])
""", 'train adapter')
    text = replace(text,
        "    test_records = [adapter(r) for r in raw['test']]\n",
        """    test_records = [adapter(r) for r in raw['test']]
    from collections import Counter
    from semgaze.data.joint import unseen_subject_ids
    counts_train = Counter(r.dataset for r in records)
    counts_test = Counter(r.dataset for r in test_records)
    if data['dataset'] == 'all' and (set(counts_train) != {'AiR', 'COCO-Search18'}
                                    or set(counts_test) != {'AiR', 'COCO-Search18'}):
        raise ValueError('joint dataset loader omitted AiR or COCO-Search18')
    unseen = unseen_subject_ids(data, manifest)
    eligible = Counter(r.dataset for r in records if r.subject not in unseen)
    print(f'[DATA] train={dict(counts_train)} eligible_queries={dict(eligible)} '
          f'test={dict(counts_test)}', flush=True)
""", 'train source dataset startup audit')
    return text


def patch_eval_entry(text):
    text = replace(text,
        'from semgaze.data.cocosearch18 import read_persisted_splits, CocoSearch18Adapter\n',
        'from semgaze.data.cocosearch18 import read_persisted_splits, CocoSearch18Adapter\nfrom semgaze.data.joint import JointAdapter\n', 'standalone import')
    text = replace(text,
        "    adapter = CocoSearch18Adapter(ROOT / data['images_root'], annotation_frame=data['annotation_frame'],\n                                  duration_field=data['duration']['source_field'])\n",
        """    adapter_cls = JointAdapter if data['dataset'] == 'all' else CocoSearch18Adapter
    adapter = adapter_cls(ROOT / data['images_root'], annotation_frame=data['annotation_frame'],
                          duration_field=data['duration']['source_field'])
""", 'standalone adapter')
    text = replace(text,
        "    queries = [adapter(r) for r in raw['test'] if r['subject'] in data['unseen_subjects']]\n",
        """    from semgaze.data.joint import unseen_subject_ids, qualified_subject
    unseen = unseen_subject_ids(data, manifest)
    queries = [adapter(r) for r in raw['test']
               if qualified_subject(r.get('dataset', 'COCO-Search18'), r['subject']) in unseen]
""", 'standalone queries')
    text = text.replace("unseen_subjects=data['unseen_subjects']", "unseen_subjects=unseen")
    return text


def patch_test(text):
    text = replace(text,
        "    queries = tuple(r for r in test_records if r.subject in unseen_subjects)\n",
        "    queries = tuple(r for r in test_records if r.subject in unseen_subjects)\n", 'test queries noop') if False else text
    text = replace(text,
        "    queries = test_queries(train_by_id, test_records, manifest, bundle.config['data']['unseen_subjects'])\n",
        "    queries = test_queries(train_by_id, test_records, manifest)\n", 'eval test queries')
    text = replace(text,
        "                        unseen_subjects=bundle.config['data']['unseen_subjects']) for q in queries]\n",
        "                        unseen_subjects=manifest['unseen_subject_ids']) for q in queries]\n", 'eval frozen queries')
    text = replace(text,
        "                row.update(test_where=float(where_losses[b]), test_flat=float(flat_losses[b]),",
        "                row.update(test_where=float(where_losses[b]), test_flat=float(flat_losses[b]),") if False else text
    text = replace(text,
        "        row.update(test_where=float(where_losses[b]), test_flat=float(flat_losses[b]),\n",
        "        row.update(test_where=float(where_losses[b]), test_flat=float(flat_losses[b]),\n                   dataset=getattr(ep.query, 'dataset', 'COCO-Search18'),\n", 'eval metric row dataset')
    text = replace(text,
        "    overall, by_k, k_times, batch_stats = EpisodeAccumulator(), {}, {}, {}\n",
        "    overall, by_k, k_times, batch_stats = EpisodeAccumulator(), {}, {}, {}\n    dataset_accumulators = {}\n    dataset_k_accumulators = {}\n", 'eval accumulators')
    text = replace(text,
        "                        if probability_enabled:\n                            probability_rows.extend(score_probability_batch(bundle, batch.episodes,\n                                visual_cache=cache))\n",
        """                        if probability_enabled:
                            # DeepGaze probability scoring has only COCO reference
                            # calibration; never label AiR LL/IG as canonical.
                            coco_episodes = [ep for ep in batch.episodes
                                             if getattr(ep.query, 'dataset', 'COCO-Search18') == 'COCO-Search18']
                            if coco_episodes:
                                probability_rows.extend(score_probability_batch(bundle, coco_episodes,
                                    visual_cache=cache))
""", 'eval prob dataset guard')
    text = replace(text,
        "                            overall.add(key, row)\n",
        """                            overall.add(key, row)
                            dataset = getattr(ep.query, 'dataset', 'COCO-Search18')
                            dataset_accumulators.setdefault(dataset, EpisodeAccumulator()).add(key, row)
                            dataset_k_accumulators.setdefault(dataset, {}).setdefault(str(k), EpisodeAccumulator()).add(key, row)
""", 'eval dataset accumulator update')
    text = replace(text,
        "    return {**overall.finish(expected), 'test_by_k': by_k, 'probability_by_k': probability_by_k,\n",
        """    by_dataset = {ds: acc.finish(list(acc.rows)) | {'episodes': len(acc.rows)}
                  for ds, acc in dataset_accumulators.items()}
    by_dataset_k = {ds: {k: acc.finish(list(acc.rows)) | {'episodes': len(acc.rows)}
                         for k, acc in blocks.items()}
                    for ds, blocks in dataset_k_accumulators.items()}
    print('[EVAL][LOSS] by_dataset=' + ', '.join(
        f'{ds}: queries={sum(q.dataset == ds for q in queries)} loss={v["test_total"]:.4f}'
        for ds, v in sorted(by_dataset.items())), flush=True)
    return {**overall.finish(expected), 'test_by_k': by_k,
            'test_by_dataset': by_dataset, 'test_by_dataset_k': by_dataset_k,
            'probability_by_k': probability_by_k,
            'probability_supported_datasets': ['COCO-Search18'] if probability_enabled else [],
""", 'eval output dataset metrics')
    return text


def patch_predictions(text):
    text = replace(text,
        "            'query_id': query.record_id, 'subject': query.subject,\n",
        "            'query_id': query.record_id, 'subject': query.subject,\n            'dataset': getattr(query, 'dataset', 'COCO-Search18'),\n", 'prediction dataset')
    text = replace(text,
        "    if any(ep.query.subject in bundle.config['data']['unseen_subjects'] for ep in train_batch):\n",
        "    if any(ep.query.subject in set(manifest['unseen_subject_ids']) for ep in train_batch):\n", 'prediction unseen')
    text = replace(text,
        "    queries = test_queries(train_by_id, test_records, manifest, bundle.config['data']['unseen_subjects']) if settings['test_scope'] == 'all_unseen' else ()\n",
        "    queries = test_queries(train_by_id, test_records, manifest) if settings['test_scope'] == 'all_unseen' else ()\n", 'prediction queries')
    text = replace(text,
        "                                    unseen_subjects=bundle.config['data']['unseen_subjects']) for query in queries]\n",
        "                                    unseen_subjects=manifest['unseen_subject_ids']) for query in queries]\n", 'prediction frozen episodes')
    text = replace(text,
        "            'test_prediction_episodes': counts['test'], 'k_values': list(k_values),\n",
        """            'test_prediction_episodes': counts['test'], 'k_values': list(k_values),
            'test_queries_by_dataset': {
                ds: sum(getattr(q, 'dataset', 'COCO-Search18') == ds for q in queries)
                for ds in sorted({getattr(q, 'dataset', 'COCO-Search18') for q in queries})},
""", 'prediction dataset count')
    # Two independent consistency checks (WHERE and semantic). Replace both
    # in one fail-closed operation rather than calling replace() twice.
    subject_anchor = "        if int(row.get('subject', query.subject)) != int(query.subject):\n"
    subject_replacement = "        if str(row.get('subject', query.subject)) != str(query.subject):\n"
    count = text.count(subject_anchor)
    if count != 2:
        raise ValueError(f'prediction string subject: expected exactly 2 anchors, found {count}')
    text = text.replace(subject_anchor, subject_replacement)
    text = replace(text,
        "                                  len(gt_x) != len(gt_d) or gt_dims != (1680, 1050))\n",
        "                                  len(gt_x) != len(gt_d) or\n                                  (getattr(query, 'dataset', 'COCO-Search18') == 'COCO-Search18'\n                                   and gt_dims != (1680, 1050)))\n", 'prediction AiR original image dimensions')
    # Dataset-specific scoring calls the existing single-dataset scorer: it
    # performs complete per-draw coverage checks *within* a dataset, and never
    # blends COCO and AiR means into a misleading single metric.
    anchor = "    metrics_config = config.get('evaluation', {}).get('metrics', {})\n"
    joint_prefix = '''    if config.get('data', {}).get('dataset') == 'all':
        import copy
        from collections import Counter
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        with Path(path).open('r', encoding='utf-8') as stream:
            rows = [json.loads(line) for line in stream if line.strip()]
        counts = Counter(getattr(q, 'dataset', 'COCO-Search18') for q in queries)
        artifacts = {}
        for dataset in ('AiR', 'COCO-Search18'):
            selected_queries = [q for q in queries if getattr(q, 'dataset', 'COCO-Search18') == dataset]
            if not selected_queries:
                raise ValueError(f'joint metrics missing test queries: {dataset}')
            subdir = output / dataset.replace('-', '_')
            subdir.mkdir(parents=True, exist_ok=True)
            selected_rows = [r for r in rows if r.get('split') == 'test' and
                             r.get('dataset', 'COCO-Search18') == dataset]
            subset_path = subdir / 'test_predictions.jsonl'
            with subset_path.open('w', encoding='utf-8') as stream:
                for row in selected_rows:
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\\n')
            scoped_config = copy.deepcopy(config)
            scoped_config['data']['dataset'] = dataset
            artifact = score_prediction_artifact(subset_path, queries=selected_queries,
                config=scoped_config, output_dir=subdir, checkpoint=checkpoint,
                split_manifest_identity=split_manifest_identity)
            artifacts[dataset] = artifact
        result = {'metrics_path': str(output / 'metrics_by_dataset.json'),
                  'by_dataset': artifacts, 'test_queries_by_dataset': dict(counts),
                  'aggregation': 'separate_dataset_metrics_no_joint_scalar'}
        (output / 'metrics_by_dataset.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        return result
'''
    # Target the function only; other occurrences of metrics_config are possible.
    idx = text.index('def score_prediction_artifact(')
    head, tail = text[:idx], text[idx:]
    tail = replace(tail, anchor, joint_prefix + anchor, 'prediction joint scoring entry')
    return head + tail


def patch_training_step(text):
    text = replace(text,
        "        require_seen_training_episode(e, bundle.config['data']['unseen_subjects'])\n",
        """        from semgaze.data.joint import unseen_subject_ids
        require_seen_training_episode(e, unseen_subject_ids(bundle.config['data']))
""", 'train step unseen union')
    return text


def patch_config_yaml(text, *, experiment):
    anchor = '  unseen_subjects: [7, 8, 9]\n'
    added = "  air_unseen_subjects: ['JY', 'SC', 'YN']\n"
    text = replace(text, anchor, anchor + added, 'yaml air unseen')
    if experiment:
        text = replace(text, '  dataset: COCO-Search18\n', '  dataset: all # Joint AiR + COCO-Search18 run\n', 'experiment joint opt-in')
    return text

PATCHERS = {
    'semgaze/model/config.py': patch_config,
    'semgaze/data/schema.py': patch_schema,
    'semgaze/data/cocosearch18.py': patch_read_splits,
    'semgaze/data/fewshot.py': patch_fewshot,
    'train_flat.py': patch_training_entry,
    'semgaze/training/flat_step.py': patch_training_step,
    'evaluate_flat.py': patch_eval_entry,
    'semgaze/evaluation/test.py': patch_test,
    'semgaze/evaluation/predictions.py': patch_predictions,
    'configs/defaults.yaml': lambda s: patch_config_yaml(s, experiment=False),
    'configs/flat_single.yaml': lambda s: patch_config_yaml(s, experiment=True),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--apply', action='store_true', help='Write after every file validates')
    args = parser.parse_args()
    repo = args.repo.resolve()
    if not (repo / 'train_flat.py').exists():
        parser.error('repo must point to the SemGaze checkout')
    staged = {}
    for name, fn in PATCHERS.items():
        path = repo / name
        original = path.read_text(encoding='utf-8')
        modified = fn(original)
        if modified == original:
            raise ValueError(f'{name}: empty patch')
        compile(modified, str(path), 'exec') if name.endswith('.py') else None
        staged[path] = modified
    own_root = Path(__file__).resolve().parent
    already_present = []
    for relative in ('semgaze/data/joint.py', 'tests/test_joint_air_coco.py'):
        source = own_root / relative
        target = repo / relative
        content = source.read_text(encoding='utf-8')
        compile(content, str(target), 'exec')
        if target.exists():
            if not target.is_file() or target.read_bytes() != source.read_bytes():
                raise ValueError(
                    f'{relative} already exists but DIFFERS from the patch kit; '
                    'refusing to overwrite. Compare the two files manually.'
                )
            already_present.append(target)
            continue
        staged[target] = content
    if not args.apply:
        print('DRY RUN: all patch anchors and Python syntax validated; pass --apply to write')
        for path in staged:
            print('  PATCH PLANNED', path.relative_to(repo))
        for path in already_present:
            print('  EXISTS IDENTICAL (safe)', path.relative_to(repo))
        return
    for path, content in staged.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')
        print('PATCHED', path.relative_to(repo))
    for path in already_present:
        print('UNCHANGED (identical)', path.relative_to(repo))
    print('Completed: review git diff; run pytest tests/test_joint_air_coco.py')


if __name__ == '__main__':
    main()
