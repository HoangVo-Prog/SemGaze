#!/usr/bin/env python3
"""Patch SemGaze no-attn: B2 concise WHERE support prompts by default.

Run from the SemGaze repository root. Only production WHERE/config files are
modified, and only known temporary K10/B2 audit scripts/tests are removed.
No GPU, dataset, checkpoint, evaluation artifacts or network is required.

  python /path/to/apply_semgaze_b2_cleanup.py --check
  python /path/to/apply_semgaze_b2_cleanup.py --apply

Then review `git diff` and make one commit yourself. This script does not commit.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import subprocess
import sys

BASE_AT_AUDIT = "8e4fa28204152e1ed44560ceb6963ed9d86bce14"

PRODUCTION = (
    "semgaze/where/conversation.py",
    "semgaze/where/collator.py",
    "semgaze/model/config.py",
    "configs/defaults.yaml",
    "configs/flat_single.yaml",
    "evaluate_flat.py",
)

TEMP_DIAGNOSTICS = (
    "tools/audit_where_prefix_runtime.py",
    "tools/audit_where_k10_root_cause.py",
    "tools/where_b2_paired.py",
    "tools/run_semgaze_b2_stage2_90.sh",
    "tests/test_audit_where_k10_root_cause.py",
    "tests/test_where_b2_paired.py",
)

OLD_CONVERSATION = '''from .prompt import build_where_prompt
from .serialization import serialize_xyd_record


def image_user(text):
    return {'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': text}]}


def build_where_conversation(episode, end_fix_token="<END_FIX>"):
    messages, images = [], []
    for record in (*episode.supports, episode.query):
        messages.append(image_user(build_where_prompt(record.task, len(record.x_px), end_fix_token)))
        images.append(record.image_path)
        if record is not episode.query:
            messages.append({'role': 'assistant', 'content': serialize_xyd_record(record, end_fix_token)})
    return messages, images
'''

NEW_CONVERSATION = '''from .prompt import build_where_prompt
from .serialization import serialize_xyd_record


SUPPORT_PROMPT_STYLES = ('full', 'concise')


def image_user(text):
    return {'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': text}]}


def build_where_support_prompt(task, n):
    """Exact B2 support prompt; the final query always uses the full WHERE instruction."""
    return (f'Support reference. Task: {task}. '
            f'Fixation count: {n}. Follow the query response format.')


def build_where_conversation(episode, end_fix_token="<END_FIX>", *, support_prompt_style="concise"):
    """Use B2 concise support instructions by default, preserving every support GT."""
    if support_prompt_style not in SUPPORT_PROMPT_STYLES:
        raise ValueError(f'invalid WHERE support_prompt_style: {support_prompt_style!r}')
    messages, images = [], []
    for record in episode.supports:
        prompt = (build_where_prompt(record.task, len(record.x_px), end_fix_token)
                  if support_prompt_style == 'full'
                  else build_where_support_prompt(record.task, len(record.x_px)))
        messages.append(image_user(prompt))
        images.append(record.image_path)
        messages.append({'role': 'assistant', 'content': serialize_xyd_record(record, end_fix_token)})
    query = episode.query
    messages.append(image_user(build_where_prompt(query.task, len(query.x_px), end_fix_token)))
    images.append(query.image_path)
    return messages, images
'''


def _replace_once(data: str, old: str, new: str, path: str) -> str:
    observed = data.count(old)
    if observed != 1:
        raise ValueError(f'{path}: expected exactly one source anchor, found {observed}; stopping to avoid unsafe patch')
    return data.replace(old, new, 1)


def patched(path: str, data: str) -> str:
    if path == 'semgaze/where/conversation.py':
        if data != OLD_CONVERSATION:
            raise ValueError(f'{path}: source differs from audited no-attn version; refusing blind overwrite')
        return NEW_CONVERSATION

    if path == 'semgaze/where/collator.py':
        return _replace_once(
            data,
            "    messages, image_paths = build_where_conversation(episode, token)\n",
            "    style = config['where']['support_prompt_style'] if config else 'concise'\n"
            "    messages, image_paths = build_where_conversation(episode, token, support_prompt_style=style)\n",
            path,
        )

    if path == 'semgaze/model/config.py':
        return _replace_once(
            data,
            "    if type(config['where']['supervision']['use_cache']) is not bool:\n",
            "    if config['where']['support_prompt_style'] not in ('full', 'concise'):\n"
            "        raise ValueError(\"where.support_prompt_style must be 'full' or 'concise'\")\n"
            "    if type(config['where']['supervision']['use_cache']) is not bool:\n",
            path,
        )

    if path == 'evaluate_flat.py':
        data = _replace_once(
            data,
            "    parser.add_argument('--semantic-max-new-tokens', type=int,\n"
            "                        help='Override evaluation.predictions.semantic_max_new_tokens')\n",
            "    parser.add_argument('--semantic-max-new-tokens', type=int,\n"
            "                        help='Override evaluation.predictions.semantic_max_new_tokens')\n"
            "    parser.add_argument('--where-support-prompt-style', choices=('full', 'concise'),\n"
            "                        help='Override WHERE support prompt style from checkpoint (default: concise/B2)')\n",
            path,
        )
        data = _replace_once(
            data,
            "        if args.path is not None:\n",
            "        if args.where_support_prompt_style is not None:\n"
            "            config['where']['support_prompt_style'] = args.where_support_prompt_style\n"
            "        if args.path is not None:\n",
            path,
        )
        data = _replace_once(
            data,
            "        'where_conditioning': 'oracle_length; no query GT trajectory' if enabled_where else None,\n",
            "        'where_support_prompt_style': config['where']['support_prompt_style'],\n"
            "        'where_conditioning': 'oracle_length; no query GT trajectory' if enabled_where else None,\n",
            path,
        )
        return data

    if path in ('configs/defaults.yaml', 'configs/flat_single.yaml'):
        return _replace_once(
            data,
            'where:\n  mode: xyd\n  end_fix_token: "<END_FIX>"\n  require_atomic_end_fix: true\n',
            'where:\n  mode: xyd\n  end_fix_token: "<END_FIX>"\n  require_atomic_end_fix: true\n'
            '  # B2: concise support user messages; final query always keeps the full WHERE instruction.\n'
            '  support_prompt_style: concise # concise (default, B2) | full (original)\n',
            path,
        )
    raise ValueError(f'unexpected source path: {path}')


def run(repo: Path, apply: bool):
    def git(*args, check=True):
        return subprocess.run(['git', *args], cwd=repo, capture_output=True, text=True, check=check)

    if not (repo / '.git').exists():
        raise ValueError(f'not a SemGaze git checkout: {repo}')
    branch = git('branch', '--show-current').stdout.strip()
    if branch != 'no-attn':
        raise ValueError(f'expected branch no-attn, currently {branch!r}')
    head = git('rev-parse', 'HEAD').stdout.strip()
    if head != BASE_AT_AUDIT:
        print(f'WARNING: no-attn HEAD {head} differs from audited {BASE_AT_AUDIT}; source anchors will be checked.', file=sys.stderr)
    # Fail closed if any file that would be modified/deleted has local edits.
    # Ignore untracked copied diagnostics here: the user explicitly asked us to remove
    # those known temporary files as well. Tracked local edits are never overwritten.
    dirty = git('status', '--porcelain', '--untracked-files=no', '--', *PRODUCTION, *TEMP_DIAGNOSTICS).stdout
    if dirty:
        raise ValueError('Target files contain local changes; stash/commit first:\n' + dirty)

    changes = {}
    for path in PRODUCTION:
        before = (repo / path).read_text(encoding='utf-8')
        after = patched(path, before)
        if before == after:
            raise ValueError(f'{path}: empty patch, aborting')
        if path.endswith('.py'):
            ast.parse(after, filename=path)
        changes[path] = after
    deletes = []
    tracked_deletes = []
    for path in TEMP_DIAGNOSTICS:
        target = repo / path
        tracked = git('ls-files', '--error-unmatch', '--', path, check=False).returncode == 0
        if target.exists():
            if not target.is_file() or target.is_symlink():
                raise ValueError(f'unexpected diagnostic file type: {path}')
            deletes.append(path)
            if tracked:
                tracked_deletes.append(path)
        elif tracked:
            raise ValueError(f'tracked diagnostic already removed (uncommitted?): {path}')
    print(f'Branch: {branch} | HEAD: {head}')
    print('Production changes:')
    for p in changes:
        print(f'  MODIFY {p}')
    print('Temporary diagnostic cleanup:')
    for p in deletes:
        print(f'  DELETE {p}')
    if not apply:
        print('CHECK PASS (no files changed). Re-run with --apply to write changes.')
        return

    # All static validation occurs before touching disk. Roll back on write failure.
    backups = {p:(repo/p).read_bytes() for p in (*changes, *deletes)}
    try:
        for p, value in changes.items():
            (repo / p).write_text(value, encoding='utf-8')
        for p in deletes:
            (repo / p).unlink()
    except Exception:
        for p, data in backups.items():
            (repo / p).write_bytes(data)
        raise
    print('PATCH APPLIED; not committed. Suggested review:')
    print('  python -m compileall -q semgaze/where semgaze/model/config.py')
    print('  git diff -- semgaze/where/conversation.py semgaze/where/collator.py semgaze/model/config.py configs/defaults.yaml configs/flat_single.yaml evaluate_flat.py')
    print('  git status --short')
    print('  git add -A -- ' + ' '.join([*PRODUCTION, *tracked_deletes]))
    print('  git commit -m "Default WHERE support prompts to B2 concise and remove K10 diagnostics"')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd(), help='SemGaze git checkout (default: current directory)')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true', help='validate source anchors without modification')
    mode.add_argument('--apply', action='store_true', help='write patch and remove tracked temporary diagnostics')
    a = parser.parse_args()
    run(a.repo.resolve(), apply=a.apply)


if __name__ == '__main__':
    main()
