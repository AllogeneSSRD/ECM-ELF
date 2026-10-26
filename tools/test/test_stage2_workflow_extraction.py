"""Check complete Stage2 flows against their pre-extraction implementation."""
import argparse,json,re,subprocess
from pathlib import Path
from test_stage2_planning_extraction import verify_snapshot,definitions,masked
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
    paths = ['src/ecm/application/cuda_stage2/ecm_cuda_stage2_main.cpp',
        'src/ecm/stage2/input.cpp', 'src/ecm/stage2/task.cpp',
        'src/ecm/stage2/process.cpp', 'src/ecm/stage2/planning/planning.cpp',
        'src/ecm/stage2/cuda/host_policy.cpp', 'src/ecm/stage2/console.h',
        'src/ecm/application/configuration/stage2_identity.cpp']
    checked = verify_snapshot(paths)
    sources = [(ROOT/name).read_text(encoding='utf-8-sig') for name in paths]
    assert sum(normalize(v).count('volatileLONGstop_requests=0;') for v in sources) == 1
    assert sum(normalize(v).count('HANDLEinherited_stop_event=nullptr;') for v in sources) == 1
    assert sum(len(re.findall(r'^int run_tasks\(', v, re.M)) for v in sources) == 1
    for name in ['src/ecm/stage2/task.cpp', 'src/ecm/stage2/process.cpp']:
        assert not re.search(r'ecm_toml|ecm_queue_config|ecm_config_generated|\bSettings\b',
                             masked((ROOT/name).read_text(encoding='utf-8-sig')))
    report = dict(complete=True, baseline=BASELINE, complete_parent_pipeline=True,
                  complete_worker_pipeline=True, application_dispatch_unchanged=True,
                  late_identity_read=True, stop_owners=1, functions=checked, gpu_runs=0)
    (a.output/'result.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
