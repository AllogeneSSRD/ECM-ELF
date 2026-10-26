"""Verify Montgomery host codecs/runtime and their unchanged current contracts."""
import argparse
import json
from pathlib import Path
import re
from source_layout import frozen_source
from test_ecm_domain_layout import tokens, definition_body

ROOT = Path(__file__).resolve().parents[2]
BASELINE = '6dad3d1020ac08e09cd85fa1695406addedc99d7'
FOLDER = 'src/ecm/montgomery_stage1/cuda/'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    names = ['host_math.cpp', 'host_math.h', 'output.cpp', 'output.h',
             'checkpoint.cpp', 'checkpoint.h', 'runtime.cu', 'prac/runtime.cu']
    for name in names:
        original = tokens(frozen_source(BASELINE, FOLDER+name))
        current = tokens((ROOT/FOLDER/name).read_text(encoding='utf-8-sig'))
        if name in ('checkpoint.cpp', 'checkpoint.h'):
            assert current.startswith(original), ('existing checkpoint codec changed', name)
        else:
            assert original == current, ('host/runtime computation changed', name)
    original = frozen_source(BASELINE, 'src/ecm/service/stage1/legacy_gpu_checkpoint.h')
    current = (ROOT/FOLDER/'checkpoint.cpp').read_text(encoding='utf-8-sig')
    assert definition_body(original, 'ecm_stage1_import_legacy_gpu') == definition_body(current, 'ecm_stage1_import_legacy_gpu')
    for name in ['host_math.h', 'checkpoint.h', 'output.h']:
        text = (ROOT/FOLDER/name).read_text(encoding='utf-8-sig')
        assert not re.search(r'#\s*include[^\n]*(?:cgbn\.h|cuda\.h|cuda_runtime|launch\.h)', text), name
    report = dict(complete=True, baseline=BASELINE, complete_math_codecs_runtime_unchanged=True,
                  legacy_importer_unchanged=True, cuda_free_host_headers=3, checked_files=names, gpu_runs=0)
    (a.output/'result.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
