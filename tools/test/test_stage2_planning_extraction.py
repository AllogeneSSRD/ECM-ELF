"""Verify current planning ownership and unchanged layouts against the migration baseline."""
import argparse
import json
from pathlib import Path
import re
import subprocess
from test_result_publication_extraction import body
from test_stage1_host_modules import normalize

from source_layout import frozen_source
ROOT=Path(__file__).resolve().parents[2]


def masked(s):
    return re.sub(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|/\*.*?\*/|//[^\n]*',lambda m:' '*len(m[0]),s,flags=re.S)


def balanced(s,at,left,right):
    m=masked(s);depth=0
    for i in range(at,len(s)):
        depth+=(m[i]==left)-(m[i]==right)
        if depth==0:return i+1
    raise AssertionError('unbalanced source')


def definitions(s):
    found=[];m=masked(s)
    for match in re.finditer(r'(?m)^([\w:<>]+(?:[ \t]+[\w:<>]+)*[ \t&*]+)(\w+)\(',m):
        end=balanced(s,m.index('(',match.start()),'(',')')
        while end<len(s) and s[end].isspace():end+=1
        if end<len(s) and s[end]=='{':found.append(match[2])
    return found



BASELINE = '6dad3d1020ac08e09cd85fa1695406addedc99d7'


def verify_snapshot(paths):
    # This round changes ownership paths, not Stage2 algorithms or protocols.
    # Compare the complete token stream, including private layouts and ordering.
    checked = {}
    for name in paths:
        original = frozen_source(BASELINE, name)
        current = (ROOT/name).read_text(encoding='utf-8-sig')
        strip = lambda text: normalize(re.sub(r'(?m)^\s*#\s*include[^\n]*', '', text))
        assert strip(original) == strip(current), ('Stage2 implementation changed', name)
        checked[name] = len(definitions(current))
    return checked


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    checked = verify_snapshot(['src/ecm/stage2/planning/planning.cpp',
        'src/ecm/stage2/planning/planning.h', 'src/ecm/stage2/types.h'])
    types = (ROOT/'src/ecm/stage2/types.h').read_text(encoding='utf-8-sig')
    planner = (ROOT/'src/ecm/stage2/planning/planning.cpp').read_text(encoding='utf-8-sig')
    assert 'struct TaskDefaults {uint64_t b2,arena,owner_mb;};' in types
    assert not re.search(r'ecm_toml|ecm_queue_config|application/configuration|\bSettings\b', masked(planner))
    report = dict(complete=True, baseline=BASELINE, complete_planning_tu_unchanged=True,
                  private_layouts_unchanged=True, defaults_fields=3, functions=checked, gpu_runs=0)
    (a.output/'result.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
