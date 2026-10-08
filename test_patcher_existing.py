"""Regression: pre-existing bundled files must be identical and stay untouched."""
from pathlib import Path
from tempfile import TemporaryDirectory
import contextlib
import importlib.util
import io
import sys

root = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('joint_patcher_test_existing', root / 'apply_joint_patch.py')
patcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patcher)

with TemporaryDirectory() as td:
    repo = Path(td)
    (repo / 'train_flat.py').write_text('a = 1\n')
    initial = root / 'semgaze/data/joint.py'
    p = repo / 'semgaze/data/joint.py'
    p.parent.mkdir(parents=True)
    p.write_bytes(initial.read_bytes())
    # Simulate existing patch kit files, and one original script needing mutation.
    original_patchers = patcher.PATCHERS
    patcher.PATCHERS = {'train_flat.py': lambda s: s.replace('a = 1', 'a = 2')}
    try:
        # Patch kit lives outside the repo: emulate its read root with original script's __file__.
        old_file = patcher.__file__
        patcher.__file__ = str(root / 'apply_joint_patch.py')
        output = io.StringIO()
        sys.argv = ['apply_joint_patch.py', '--repo', str(repo)]
        with contextlib.redirect_stdout(output):
            patcher.main()
        assert 'EXISTS IDENTICAL (safe) semgaze/data/joint.py' in output.getvalue()
        assert (repo / 'train_flat.py').read_text() == 'a = 1\n', 'dry-run must not write'
        assert p.read_bytes() == initial.read_bytes()
        sys.argv = ['apply_joint_patch.py', '--repo', str(repo), '--apply']
        with contextlib.redirect_stdout(output):
            patcher.main()
        assert (repo / 'train_flat.py').read_text() == 'a = 2\n'
        assert p.read_bytes() == initial.read_bytes()
        assert (repo / 'tests/test_joint_air_coco.py').read_bytes() == (root/'tests/test_joint_air_coco.py').read_bytes()
        # Differing existing file must fail before any write, even with --apply.
        (repo / 'train_flat.py').write_text('a = 1\n')
        p.write_text('DIFFERENT DATA\n')
        try:
            patcher.main()
        except ValueError as exc:
            assert 'DIFFERS from the patch kit' in str(exc)
        else:
            raise AssertionError('conflicting existing files must be rejected')
        assert (repo / 'train_flat.py').read_text() == 'a = 1\n', 'conflict must abort prior to write'
    finally:
        patcher.PATCHERS = original_patchers
        patcher.__file__ = old_file
print('PASS: identical pre-existing files tolerated, conflicting files rejected; dry-run read-only, apply writes')
