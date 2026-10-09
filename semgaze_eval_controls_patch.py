#!/usr/bin/env python3
"""Apply evaluation interval/branch controls to SemGaze no-attn.

Usage (from repository root):
    python /path/to/semgaze_eval_controls_patch.py --check
    python /path/to/semgaze_eval_controls_patch.py --diff > eval_controls.patch
    python /path/to/semgaze_eval_controls_patch.py --apply

The patcher checks every source anchor before writing anything, and will not
silently reapply or partially apply a change.
"""
import argparse
import difflib
from pathlib import Path


class Patch:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.before = {}
        self.after = {}

    def replace(self, file, old, new):
        if file not in self.before:
            path = self.root / file
            self.before[file] = path.read_text(encoding='utf-8')
            self.after[file] = self.before[file]
        text = self.after[file]
        count = text.count(old)
        if count != 1:
            raise RuntimeError(f'{file}: expected exactly 1 anchor; found {count}. '
                               'Source changed or patch already applied; no files were written.\n'
                               f'Anchor: {old[:160]!r}')
        self.after[file] = text.replace(old, new, 1)

    def verify(self):
        for path, content in self.after.items():
            if path.endswith('.py'):
                compile(content, path, 'exec')
        if not any(self.before[p] != self.after[p] for p in self.before):
            raise RuntimeError('No source differences found')

    def diff(self):
        for filename in self.before:
            yield from difflib.unified_diff(
                self.before[filename].splitlines(keepends=True),
                self.after[filename].splitlines(keepends=True),
                fromfile='a/' + filename, tofile='b/' + filename,
            )

    def apply(self):
        # All anchors and compilation have been checked before writes.
        for filename, content in self.after.items():
            (self.root / filename).write_text(content, encoding='utf-8')


def make_patch(root):
    p = Patch(root)
    for file in ('configs/defaults.yaml', 'configs/flat_single.yaml'):
        p.replace(file,
                  '  eval_steps: 1000 # required when strategy is steps; counts global optimizer steps\n',
                  '  eval_epoch: 1 # with strategy: epoch, evaluate after each N completed epochs\n'
                  '  eval_steps: 1000 # required when strategy is steps; counts global optimizer steps\n')
        p.replace(file,
                  '  predictions:\n    train_batches: 1\n',
                  '  predictions:\n'
                  '    where: true # autoregressive WHERE predictions and their scanpath metrics\n'
                  '    semantic: true # autoregressive semantic predictions and their metrics\n'
                  '    train_batches: 1\n')

    file = 'semgaze/model/config.py'
    p.replace(file,
              "    positive_int(evaluation['draw'], 'evaluation.draw')\n",
              "    positive_int(evaluation['draw'], 'evaluation.draw')\n"
              "    positive_int(evaluation['eval_epoch'], 'evaluation.eval_epoch')\n")
    p.replace(file,
              "    positive_int(predictions['train_batches'], 'evaluation.predictions.train_batches', minimum=0)\n",
              "    for branch in ('where', 'semantic'):\n"
              "        if type(predictions[branch]) is not bool:\n"
              "            raise ValueError(f'evaluation.predictions.{branch} must be boolean')\n"
              "    positive_int(predictions['train_batches'], 'evaluation.predictions.train_batches', minimum=0)\n")

    file = 'semgaze/evaluation/test.py'
    p.replace(file,
              "    probability_enabled = bool(bundle.config.get('evaluation', {}).get('metrics', {}).get('enabled')) and \\\n        bool(probability_config.get('ll') or probability_config.get('ig'))\n",
              "    # LL/IG are WHERE prediction metrics, not required for teacher-forced loss.\n"
              "    probability_enabled = bool(bundle.config['evaluation']['predictions']['where']) and \\\n"
              "        bool(bundle.config.get('evaluation', {}).get('metrics', {}).get('enabled')) and \\\n"
              "        bool(probability_config.get('ll') or probability_config.get('ig'))\n")

    file = 'semgaze/training/loop.py'
    p.replace(file,
              "    prediction_capacity = settings['train_batches'] * t['per_device_train_batch_size']\n",
              "    prediction_active = settings['where'] or settings['semantic']\n"
              "    prediction_capacity = (settings['train_batches'] * t['per_device_train_batch_size']\n"
              "                           if prediction_active else 0)\n")
    p.replace(file,
              "            should_evaluate = ((evaluation['strategy'] == 'epoch' and epoch_end) or\n",
              "            should_evaluate = ((evaluation['strategy'] == 'epoch' and epoch_end and\n"
              "                                sampler.epoch % evaluation['eval_epoch'] == 0) or\n")
    p.replace(file,
              "                has_predictions = settings['train_batches'] or settings['test_scope'] != 'none'\n",
              "                has_predictions = prediction_active and (\n"
              "                    settings['train_batches'] > 0 or settings['test_scope'] != 'none')\n")
    p.replace(file,
              "                        if predictions.get('metrics_artifact'):\n"
              "                            metric_event = {'event': 'epoch_metrics', 'step': step + 1,\n"
              "                                            'epoch': summary['epoch'], 'split': 'test',\n"
              "                                            'metrics_file': predictions['metrics_artifact'].get('metrics_path'),\n"
              "                                            'metric_namespaces': ('[EVAL][METRIC][WHERE]', '[EVAL][METRIC][SEM]')}\n"
              "                            bundle.trainer_history.append(metric_event)\n"
              "                            stream.write(json.dumps(metric_event, allow_nan=False) + '\\n')\n"
              "                            stream.flush()\n"
              "                            print('[EVAL][METRIC][SEM] frozen semantic metrics complete', flush=True)\n",
              "                        if predictions.get('metrics_artifact'):\n"
              "                            metrics_config = evaluation['metrics']\n"
              "                            names = []\n"
              "                            where_cfg = metrics_config.get('where', {})\n"
              "                            semantic_cfg = metrics_config.get('semantic', {})\n"
              "                            if settings['where'] and any(where_cfg.get(n) for n in ('scanmatch', 'multimatch', 'sed')):\n"
              "                                names.append('[EVAL][METRIC][WHERE]')\n"
              "                            if settings['semantic'] and (semantic_cfg.get('bertscore', {}).get('enabled') or\n"
              "                                                         semantic_cfg.get('cider_r', {}).get('enabled')):\n"
              "                                names.append('[EVAL][METRIC][SEM]')\n"
              "                            if names:\n"
              "                                metric_event = {'event': 'epoch_metrics', 'step': step + 1,\n"
              "                                                'epoch': summary['epoch'], 'split': 'test',\n"
              "                                                'metrics_file': predictions['metrics_artifact'].get('metrics_path'),\n"
              "                                                'metric_namespaces': names}\n"
              "                                bundle.trainer_history.append(metric_event)\n"
              "                                stream.write(json.dumps(metric_event, allow_nan=False) + '\\n')\n"
              "                                stream.flush()\n"
              "                                for name in names:\n"
              "                                    print(f'{name} frozen metrics complete', flush=True)\n")

    file = 'semgaze/evaluation/predictions.py'
    p.replace(file,
              "    settings.setdefault('test_scope', 'all_unseen')\n",
              "    settings.setdefault('test_scope', 'all_unseen')\n"
              "    settings.setdefault('where', True)\n"
              "    settings.setdefault('semantic', True)\n"
              "    for branch in ('where', 'semantic'):\n"
              "        if type(settings[branch]) is not bool:\n"
              "            raise ValueError(f'evaluation.predictions.{branch} must be boolean')\n")
    p.replace(file,
              "    if not settings['train_batches'] and settings['test_scope'] == 'none':\n"
              "        return settings\n",
              "    if (not settings['where'] and not settings['semantic']) or (\n"
              "            not settings['train_batches'] and settings['test_scope'] == 'none'):\n"
              "        return settings\n"
              "    if not settings['semantic']:\n"
              "        # WHERE-only generation needs no semantic token budget.\n"
              "        return settings\n")
    p.replace(file,
              "    where = evaluate_where_episode(bundle, episode) if generated is None else generated[0]\n"
              "    # Canonical semantic inference constructs R from GT WHERE, never predicted\n"
              "    # fixations. Only the semantic response is autoregressively generated here.\n"
              "    semantic = evaluate_flat_episode(bundle, episode, generation_budget=budget) if generated is None else generated[1]\n"
              "    query = episode.query\n"
              "    where_diagnostics = dict(where)\n"
              "    where_text = where_diagnostics.pop('text')\n",
              "    where = evaluate_where_episode(bundle, episode) if generated is None else generated[0]\n"
              "    # Canonical semantic inference constructs R from GT WHERE, never predicted\n"
              "    # fixations. Only the semantic response is autoregressively generated here.\n"
              "    semantic = evaluate_flat_episode(bundle, episode, generation_budget=budget) if generated is None else generated[1]\n"
              "    query = episode.query\n")
    p.replace(file,
              "            'gt_duration_ms_raw': list(query.duration_ms),\n"
              "            'WHERE': {'GT': serialize_xyd_record(query, bundle.config['where']['end_fix_token']), 'PRED': where_text},\n"
              "            'where_generation': where_diagnostics}\n",
              "            'gt_duration_ms_raw': list(query.duration_ms)}\n"
              "    if where is not None:\n"
              "        where_diagnostics = dict(where)\n"
              "        where_text = where_diagnostics.pop('text')\n"
              "        record.update({\n"
              "            'WHERE': {'GT': serialize_xyd_record(query, bundle.config['where']['end_fix_token']), 'PRED': where_text},\n"
              "            'where_generation': where_diagnostics,\n"
              "        })\n")
    p.replace(file,
              "    where_draw_counts = resolve_evaluation_draw_counts(bundle.config['evaluation']['draw'], manifest, k_values)\n"
              "    semantic_use_draws = bundle.config.get('evaluation', {}).get('semantic_use_draws', True)\n"
              "    semantic_draw_counts = resolve_semantic_draw_counts(where_draw_counts, semantic_use_draws)\n",
              "    where_enabled, semantic_enabled = settings['where'], settings['semantic']\n"
              "    if not (where_enabled or semantic_enabled):\n"
              "        raise ValueError('predict_epoch called with both prediction branches disabled')\n"
              "    where_draw_counts = resolve_evaluation_draw_counts(bundle.config['evaluation']['draw'], manifest, k_values)\n"
              "    semantic_use_draws = bundle.config.get('evaluation', {}).get('semantic_use_draws', True)\n"
              "    semantic_draw_counts = (resolve_semantic_draw_counts(where_draw_counts, semantic_use_draws)\n"
              "                            if semantic_enabled else {str(k): 0 for k in k_values})\n"
              "    active_draw_counts = (where_draw_counts if where_enabled else semantic_draw_counts)\n"
              "    train_path = 'both' if where_enabled and semantic_enabled else ('where' if where_enabled else 'semantic')\n")
    p.replace(file,
              "        where_total = sum(len(queries) * where_draw_counts[str(k)] for k in k_values)\n"
              "        semantic_total = sum(len(queries) * semantic_draw_counts[str(k)] for k in k_values)\n"
              "        semantic_draw_label = str(sum(semantic_draw_counts.values())) if semantic_use_draws else \\\n"
              "            'disabled / single traversal'\n"
              "        print(f\"[EVAL][PRED] starting test predictions | queries={len(queries)} | \"\n"
              "              f\"K={list(k_values)} | WHERE draws={sum(where_draw_counts.values())} | \"\n"
              "              f\"semantic draws={semantic_draw_label} | total_episodes={where_total} | \"\n"
              "              f\"semantic_episodes={semantic_total}\", flush=True)\n",
              "        total = sum(len(queries) * active_draw_counts[str(k)] for k in k_values)\n"
              "        semantic_total = sum(len(queries) * semantic_draw_counts[str(k)] for k in k_values)\n"
              "        semantic_draw_label = (str(sum(semantic_draw_counts.values())) if semantic_enabled\n"
              "                               else 'OFF')\n"
              "        where_draw_label = str(sum(where_draw_counts.values())) if where_enabled else 'OFF'\n"
              "        print(f\"[EVAL][PRED] starting test predictions | queries={len(queries)} | \"\n"
              "              f\"K={list(k_values)} | WHERE draws={where_draw_label} | \"\n"
              "              f\"semantic draws={semantic_draw_label} | total_episodes={total} | \"\n"
              "              f\"semantic_episodes={semantic_total}\", flush=True)\n")
    p.replace(file,
              "    # The combined prediction file has one record per WHERE episode. Semantic\n"
              "    # progress is reported separately because it may be a strict subset.\n"
              "    prediction_total = sum(len(queries) * where_draw_counts[str(k)] for k in k_values)\n",
              "    # The combined file has one record per active prediction episode.\n"
              "    # Semantic-only runs do not perform WHERE generation.\n"
              "    prediction_total = sum(len(queries) * active_draw_counts[str(k)] for k in k_values)\n")
    p.replace(file,
              "                        for episode, generated in prediction_batches(bundle, train_batch,\n"
              "                                budget=settings['semantic_max_new_tokens'], cache=cache,\n"
              "                                projected_r_cache=projected_r_cache,\n",
              "                        for episode, generated in prediction_batches(bundle, train_batch,\n"
              "                                budget=settings['semantic_max_new_tokens'], cache=cache,\n"
              "                                path=train_path, projected_r_cache=projected_r_cache,\n")
    p.replace(file,
              "                            draw_count = where_draw_counts[str(k)]\n"
              "                            semantic_draw_count = semantic_draw_counts[str(k)]\n"
              "                            draw_times = []\n"
              "                            semantic_label = str(semantic_draw_count) if semantic_use_draws else \\\n"
              "                                'disabled / single traversal'\n"
              "                            print(f\"[EVAL][PRED] K={k} ({k_position}/{len(k_values)}) | \"\n"
              "                                  f\"WHERE draws={draw_count} | semantic draws={semantic_label}\", flush=True)\n",
              "                            draw_count = active_draw_counts[str(k)]\n"
              "                            where_draw_count = where_draw_counts[str(k)] if where_enabled else 0\n"
              "                            semantic_draw_count = semantic_draw_counts[str(k)]\n"
              "                            draw_times = []\n"
              "                            semantic_label = str(semantic_draw_count) if semantic_enabled else 'OFF'\n"
              "                            where_label = str(where_draw_count) if where_enabled else 'OFF'\n"
              "                            print(f\"[EVAL][PRED] K={k} ({k_position}/{len(k_values)}) | \"\n"
              "                                  f\"WHERE draws={where_label} | semantic draws={semantic_label}\", flush=True)\n")
    p.replace(file,
              "                                prediction_path = 'both' if draw < semantic_draw_count else 'where'\n",
              "                                prediction_path = (\n"
              "                                    'both' if where_enabled and draw < semantic_draw_count else\n"
              "                                    'where' if where_enabled else 'semantic')\n")
    p.replace(file,
              "                                'where_draws': draw_count,\n",
              "                                'where_draws': where_draw_count,\n")
    p.replace(file,
              "                            print(f\"[EVAL][PRED] K={k} ({k_position}/{len(k_values)}) | \"\n"
              "                                  f\"WHERE draws={draw_count} | semantic draws={semantic_label} | complete\", flush=True)\n",
              "                            print(f\"[EVAL][PRED] K={k} ({k_position}/{len(k_values)}) | \"\n"
              "                                  f\"WHERE draws={where_label} | semantic draws={semantic_label} | complete\", flush=True)\n")
    p.replace(file,
              "    if paths.get('test') and bundle.config.get('evaluation', {}).get('metrics', {}).get('enabled'):\n",
              "    if queries and paths.get('test') and bundle.config.get('evaluation', {}).get('metrics', {}).get('enabled'):\n")
    p.replace(file,
              "            'semantic_use_draws': semantic_use_draws,\n"
              "            'prediction_files': paths, 'semantic_max_new_tokens': settings['semantic_max_new_tokens'],\n",
              "            'semantic_use_draws': semantic_use_draws if semantic_enabled else False,\n"
              "            'prediction_branches': {'where': where_enabled, 'semantic': semantic_enabled},\n"
              "            'prediction_files': paths, 'semantic_max_new_tokens': settings['semantic_max_new_tokens'],\n")
    p.replace(file,
              "            'where_conditioning': 'oracle_length; no query GT trajectory',\n"
              "            'semantic_conditioning': 'teacher_forced_GT_XYD states; gold WHY groups'}\n",
              "            **({'where_conditioning': 'oracle_length; no query GT trajectory'} if where_enabled else {}),\n"
              "            **({'semantic_conditioning': 'teacher_forced_GT_XYD states; gold WHY groups'}\n"
              "               if semantic_enabled else {})}\n")
    p.replace(file,
              "    return (f\"[EVAL][PRED] epoch {entry['epoch']} | step {entry['step']} | \"\n"
              "            \"autoregressive predictions complete:\\n\"\n"
              "            f\"  train: {batches} {batch_word} ({entry['train_prediction_episodes']} queries) -> {paths['train']}\\n\"\n"
              "            f\"  test: {entry['test_prediction_queries']} unseen queries x K={k_text} \"\n"
              "            f\"({entry['test_prediction_episodes']} episodes) -> {paths['test']}\\n\"\n"
              "            f\"  semantic test predictions: {entry.get('test_semantic_prediction_episodes', 'n/a')} episodes; \"\n"
              "            f\"draws={'enabled' if entry.get('semantic_use_draws', True) else 'disabled / single traversal'}\\n\"\n"
              "            '  Records contain WHERE GT/PRED; semantic fields appear only on semantic traversals. '\n"
              "            'WHERE is free-running; semantic uses GT WHERE states + gold WHY groups.')\n",
              "    branches = entry.get('prediction_branches', {'where': True, 'semantic': True})\n"
              "    enabled = ', '.join(branch.upper() for branch in ('where', 'semantic') if branches[branch])\n"
              "    return (f\"[EVAL][PRED] epoch {entry['epoch']} | step {entry['step']} | \"\n"
              "            f\"autoregressive predictions complete: {enabled}\\n\"\n"
              "            f\"  train: {batches} {batch_word} ({entry['train_prediction_episodes']} queries) -> {paths['train']}\\n\"\n"
              "            f\"  test: {entry['test_prediction_queries']} unseen queries x K={k_text} \"\n"
              "            f\"({entry['test_prediction_episodes']} episodes) -> {paths['test']}\\n\"\n"
              "            f\"  semantic test predictions: {entry.get('test_semantic_prediction_episodes', 0)} episodes\\n\"\n"
              "            '  Records contain only the enabled prediction branches. '\n"
              "            'WHERE is free-running; semantic uses GT WHERE states + gold WHY groups.')\n")
    # In joint AiR+COCO runs, short-circuit before recursive per-dataset
    # artifact creation when neither enabled branch has scoring configured.
    p.replace(file,
              "    if config.get('data', {}).get('dataset') == 'all':\n",
              "    metric_opts = config.get('evaluation', {}).get('metrics', {})\n"
              "    active_opts = config.get('evaluation', {}).get('predictions', {})\n"
              "    where_opts = metric_opts.get('where', {})\n"
              "    sem_opts = metric_opts.get('semantic', {})\n"
              "    score_where = active_opts.get('where', True) and any(\n"
              "        where_opts.get(k) for k in ('scanmatch', 'multimatch', 'sed'))\n"
              "    score_sem = active_opts.get('semantic', True) and (\n"
              "        sem_opts.get('bertscore', {}).get('enabled') or\n"
              "        sem_opts.get('cider_r', {}).get('enabled'))\n"
              "    if not metric_opts.get('enabled') or not (score_where or score_sem):\n"
              "        return None\n"
              "    if config.get('data', {}).get('dataset') == 'all':\n")
    # Scoring follows selected prediction branches, never merely metric settings.
    p.replace(file,
              "    where_config = metrics_config.get('where', {})\n"
              "    scanpath_enabled = any(bool(where_config.get(name)) for name in ('scanmatch', 'multimatch', 'sed'))\n"
              "    semantic_config = metrics_config.get('semantic', {})\n"
              "    semantic_enabled = bool(semantic_config.get('bertscore', {}).get('enabled') or\n"
              "                            semantic_config.get('cider_r', {}).get('enabled'))\n",
              "    pred_config = config.get('evaluation', {}).get('predictions', {})\n"
              "    where_config = metrics_config.get('where', {})\n"
              "    scanpath_enabled = pred_config.get('where', True) and any(\n"
              "        bool(where_config.get(name)) for name in ('scanmatch', 'multimatch', 'sed'))\n"
              "    semantic_config = metrics_config.get('semantic', {})\n"
              "    semantic_enabled = pred_config.get('semantic', True) and bool(\n"
              "        semantic_config.get('bertscore', {}).get('enabled') or\n"
              "        semantic_config.get('cider_r', {}).get('enabled'))\n")
    p.replace(file,
              "    if semantic_config.get('bertscore', {}).get('enabled'):\n",
              "    if semantic_enabled and semantic_config.get('bertscore', {}).get('enabled'):\n")
    p.replace(file,
              "    if semantic_config.get('cider_r', {}).get('enabled'):\n",
              "    if semantic_enabled and semantic_config.get('cider_r', {}).get('enabled'):\n")
    p.replace(file,
              "    semantic_rows = [row for row in test_rows if 'semantic_generation' in row]\n",
              "    semantic_rows = ([row for row in test_rows if 'semantic_generation' in row]\n"
              "                     if semantic_enabled else [])\n")
    p.replace(file,
              "        where_grouped.setdefault((key[1], key[2]), []).append(row)\n",
              "        if scanpath_enabled:\n"
              "            where_grouped.setdefault((key[1], key[2]), []).append(row)\n")
    p.replace(file,
              "    grouped_keys = sorted((set(where_grouped) if scanpath_enabled else set()) |\n"
              "                          set(semantic_grouped))\n",
              "    grouped_keys = sorted((set(where_grouped) if scanpath_enabled else set()) |\n"
              "                          (set(semantic_grouped) if semantic_enabled else set()))\n")
    p.replace(file,
              "        block['semantic_draw_count'] = len(semantic_draws)\n"
              "        block['semantic_metric_aggregation'] = 'draw_mean' if semantic_use_draws else 'single_traversal'\n"
              "        block['draw_mean'] = aggregate_semantic_metrics(semantic_draws, use_draws=semantic_use_draws)\n",
              "        if semantic_enabled:\n"
              "            block['semantic_draw_count'] = len(semantic_draws)\n"
              "            block['semantic_metric_aggregation'] = 'draw_mean' if semantic_use_draws else 'single_traversal'\n"
              "            block['draw_mean'] = aggregate_semantic_metrics(semantic_draws, use_draws=semantic_use_draws)\n"
              "        else:\n"
              "            block['draw_mean'] = {}\n")
    p.replace(file,
              "        provenance={'bertscore': bert_provenance, 'cider_r': cider_provenance,\n"
              "                    'scanpath': {\n",
              "        provenance={**({'bertscore': bert_provenance, 'cider_r': cider_provenance}\n"
              "                       if semantic_enabled else {}),\n"
              "                    'scanpath': {\n")
    return p


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', type=Path, default=Path('.'))
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--diff', action='store_true')
    mode.add_argument('--apply', action='store_true')
    args = ap.parse_args()
    patch = make_patch(args.repo)
    patch.verify()
    if args.diff:
        print(''.join(patch.diff()), end='')
    elif args.apply:
        patch.apply()
        print('Patched ' + ', '.join(patch.after))
    else:
        print('Patch anchors and Python syntax validated; no files modified.\n'
              'Files: ' + ', '.join(patch.after))


if __name__ == '__main__':
    main()
