"""Current Stage1 ownership seams plus the complete frozen computation audit."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from source_layout import frozen_source,current_source

ROOT = Path(__file__).resolve().parents[2]
BASELINE = '6dad3d1020ac08e09cd85fa1695406addedc99d7'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    subprocess.run([sys.executable, 'tools/test/test_ecm_domain_layout.py', '--output',
                    str(args.output/'computation')], cwd=ROOT, check=True)
    read = lambda name: current_source(name).read_text(encoding='utf-8-sig')
    for curve in ('edwards', 'montgomery'):
        for backend in ('cpu', 'cuda'):
            workflow = read(f'src/ecm/{curve}_stage1/{backend}/workflow.cpp')
            assert 'LegacyExecutionResult' not in workflow and 'take_computation_result' not in workflow
            assert 'ComputationResult' in workflow
            assert 'publish_save(' not in workflow and 'append_save_lines' not in workflow
    task = read('src/ecm/stage1/task.cpp')
    assert 'ecm_stage1_checkpoint::Context session;' in task
    assert 'deps.execute(request, execution, session)' in task
    assert 'struct Context::State' in read('src/io/stage1_checkpoint.cpp')
    for name in ('src/io/stage1_checkpoint.cpp', 'src/io/exponent_cache.cpp'):
        text = read(name)
        assert 'thread_local' not in text and 'g_cache_dir' not in text
    for name in ('src/ecm/storage', 'include/ecm/storage'):
        assert not [p for p in (ROOT/name).rglob('*') if p.is_file()]
    previous=json.loads(frozen_source(BASELINE, 'config/ecm_options.json'))
    current=json.loads(read('config/ecm_options.json'))
    added=[option for option in current['options'] if option['key']=='progress_average_seconds']
    assert len(added)==1 and added[0]['default']==0 and added[0]['minimum']==0 and added[0]['maximum']==60
    current['options']=[option for option in current['options'] if option['key']!='progress_average_seconds']
    assert current==previous,'unapproved configuration changes'
    subprocess.run([sys.executable, 'tools/gen/generate_ecm_config.py', '--check'], cwd=ROOT, check=True)
    report = dict(complete=True, baseline=BASELINE, complete_computation_inventory=True,
                  owned_cpu_cuda_results=True, explicit_checkpoint_context=True,
                  storage_removed=True, schema_change='progress_average_seconds_only', gpu_runs=0)
    (args.output/'result.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
