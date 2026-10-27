"""Exact handoff between frozen relocation audits and generated instance tests.

Retired bindings keep their original Git identity. They are never relabelled as
algorithm headers or configuration-dependent build outputs.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = json.loads((ROOT/'docs/development/STAGE1_INSTANCE_REFACTOR_MANIFEST.json').read_text())
RECORDS = {entry['source']: entry for entry in MANIFEST['files']}
TARGETS = {entry['target']: entry for entry in MANIFEST['files'] if entry['target']}
TOKEN = re.compile(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|/\*.*?\*/|//[^\n]*|\s+|.', re.S)


def tokens(text):
    return ''.join(m[0] for m in TOKEN.finditer(text)
                   if not m[0].isspace() and not m[0].startswith(('//', '/*')))


def key(path):
    path = Path(path)
    return path.relative_to(ROOT).as_posix() if path.is_absolute() else path.as_posix()


def handoff_record(path):
    name = key(path)
    return RECORDS.get(name, TARGETS.get(name))


def is_retired(path):
    entry = RECORDS.get(key(path))
    return bool(entry and entry['target'] is None)


@lru_cache(None)
def previous_bytes(path):
    entry = handoff_record(path)
    assert entry is not None, ('not an approved handoff', str(path))
    data = subprocess.check_output(['git', 'show', MANIFEST['baseline']+':'+entry['source']], cwd=ROOT)
    assert hashlib.sha256(data).hexdigest() == entry['git_sha256'], entry['source']
    return data


def audit_bytes(path):
    """Use the handoff snapshot only for explicitly changed earlier owners."""
    return previous_bytes(path) if handoff_record(path) else (ROOT/key(path)).read_bytes()


@lru_cache(None)
def verify_handoff():
    assert len(RECORDS) == len(MANIFEST['files']) == 44
    retired = 0
    for entry in MANIFEST['files']:
        before = previous_bytes(entry['source']).decode('utf-8-sig')
        if entry['target'] is None:
            assert not (ROOT/entry['source']).exists(), ('retired source remains', entry['source'])
            code = tokens(before)
            assert not re.search(r'__global__|__device__|mpz_\w+', code), entry['source']
            retired += 1
            continue
        after = (ROOT/entry['target']).read_text(encoding='utf-8-sig')
        if entry['kind'] == 'include_rewrite':
            strip = lambda s: re.sub(r'^\s*#\s*include[^\n]*', '', s, flags=re.M)
            assert tokens(strip(before)) == tokens(strip(after)), entry['source']
        elif entry['kind'] == 'declarations_rewrite':
            pattern = r'cgbn_stage1_kernel_fn cgbn_stage1_kernel_(?:(?:suyama|param2)_)?tpi(?:4|8|16|32)\([^;]*;'
            expected, count = re.subn(pattern, '', before)
            assert count == 12 and tokens(expected) == tokens(after), entry['source']
        elif entry['kind'] == 'dispatch_rewrite':
            assert not re.search(r'__global__|__device__|mpz_\w+', tokens(after)), entry['target']
        elif entry['kind'] == 'body_move':
            start = 'template<class P, int MODE>'
            old = before[before.index(start):before.index('\ncgbn_stage1_kernel_fn cgbn_stage1_domain_constants')]
            new = after[after.index(start):after.index('\ncgbn_stage1_kernel_fn ECM_PRAC_FACTORY')]
            assert tokens(old) == tokens(new), 'constant kernel body/attributes changed'
        elif entry['kind'] == 'metadata':
            for pattern in [r'^ECM_CUDA_TIER\([^\n]+', r'^ECM_CUDA_LEGACY_TYPE\([^\n]+']:
                assert re.findall(pattern, before, re.M) == re.findall(pattern, after, re.M)
            start = '#ifndef ECM_CUDA_STAGE2_CAPACITIES'
            assert tokens(before[before.index(start):before.index('// Explicit PRAC policy')]) == \
                   tokens(after[after.index(start):after.index('// Explicit PRAC policy')])
        else:
            raise AssertionError(entry)
    assert retired == 38
    for name, expected in MANIFEST['unchanged_math'].items():
        current = tokens((ROOT/name).read_text(encoding='utf-8-sig'))
        assert hashlib.sha256(current.encode()).hexdigest() == expected, ('math changed', name)
    return dict(retired_bindings_and_interface=retired, unchanged_math_files=9,
                moved_kernel_bodies=1, handoff_entries=44)
