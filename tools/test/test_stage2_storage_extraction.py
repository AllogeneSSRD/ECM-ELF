"""Verify current Stage2 progress/receipt/SHA layouts against the migration baseline."""
import argparse
import json
from pathlib import Path
import re
from source_layout import frozen_source
from test_ecm_domain_layout import tokens

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    paths = ['src/io/stage2_checkpoint.cpp', 'include/ecm/io/stage2_checkpoint.h',
             'src/ecm/platform/fingerprint.cpp', 'include/ecm/platform/fingerprint.h']
    for name in paths:
        assert tokens(frozen_source('6dad3d1', name)) == tokens((ROOT/name).read_text(encoding='utf-8-sig')), name
    for name in ['include/ecm/io/stage2_checkpoint.h', 'include/ecm/platform/fingerprint.h']:
        text = (ROOT/name).read_text(encoding='utf-8-sig')
        assert not re.search(r'#\s*include[^\n]*(?:windows|bcrypt|fstream|cuda)', text, re.I)
        assert 'CreateFile' not in text and 'MoveFileEx' not in text
    report = dict(complete=True, baseline='6dad3d1', storage_protocol_unchanged=True,
                  sha_unchanged=True, legacy_import_unchanged=True,
                  state_task_layout_unchanged=True, public_os_headers=0, checked_files=paths, gpu_runs=0)
    (a.output/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
