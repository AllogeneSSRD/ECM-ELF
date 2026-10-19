"""Check benchmark GPU-context filtering without invoking CUDA or a GPU tool."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/bench'))
from bench_stage2_version import foreign_gpu_processes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    own = str(out / 'bin/ecm_cuda_stage2.exe')
    rows = [dict(gpu_uuid='GPU0', pid='1', process_name='[Insufficient Permissions]'),
            dict(gpu_uuid='GPU1', pid='2', process_name=own),
            dict(gpu_uuid='GPU1', pid='3', process_name=str(out / 'bin/worker.exe'))]
    assert foreign_gpu_processes(rows, 'GPU1', out) == []
    unknown = dict(gpu_uuid='GPU1', pid='4', process_name='[Insufficient Permissions]')
    assert foreign_gpu_processes(rows + [unknown], 'GPU1', out) == [unknown]
    neighbor = dict(gpu_uuid='GPU1', pid='5', process_name=str(Path(str(out) + '_another') / 'worker.exe'))
    assert foreign_gpu_processes(rows + [neighbor], 'GPU1', out) == [neighbor]
    assert foreign_gpu_processes([], 'GPU1', out) == []
    report = dict(complete=True, checks=4, other_GPU_ignored=True, own_workers_allowed=True,
                  foreign_and_unknown_rejected=True, sibling_prefix_rejected=True, CUDA_calls=0)
    (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
