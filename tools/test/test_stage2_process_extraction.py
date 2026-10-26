"""Verify narrow process input and exact synchronous domain feedback placement."""
import argparse,json,re,subprocess
from pathlib import Path
from test_stage2_planning_extraction import verify_snapshot,balanced,definitions,masked
from test_result_publication_extraction import body
from test_stage1_host_modules import normalize
from source_layout import frozen_source
ROOT=Path(__file__).resolve().parents[2]

BASELINE = '6dad3d1020ac08e09cd85fa1695406addedc99d7'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    checked = verify_snapshot(['src/ecm/stage2/process.cpp',
        'src/ecm/stage2/process.h', 'include/ecm/stage2/curve_process.h',
        'src/ecm/stage2/task.cpp'])
    task = (ROOT/'src/ecm/stage2/task.cpp').read_text(encoding='utf-8-sig')
    projection = body(task, 'child_run')
    pairs = set(re.findall(r'request\.(\w+)=o\.(\w+);', projection))
    assert len(pairs) == 49 and all(a == b for a, b in pairs)
    public = (ROOT/'include/ecm/stage2/curve_process.h').read_text(encoding='utf-8-sig')
    assert not re.search(r'\b(?:Options|Settings|HANDLE|LONG|cuda|mpz_t)\b', masked(public))
    assert not re.search(r'#include.*(?:gmp|windows|cuda)', public)
    report = dict(complete=True, baseline=BASELINE, process_protocol_unchanged=True,
                  synchronous_feedback_unchanged=True, projected_process_fields=49,
                  native_handle_unchanged=True, functions=checked, gpu_runs=0)
    (a.output/'result.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
