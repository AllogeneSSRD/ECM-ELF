"""Representative upper CGBN tiers and fail-closed Edwards worker publication."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from test_edwards_gpu_pipeline import records, curve, values


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage1', type=Path, required=True)
    p.add_argument('--stage2', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', type=int, default=1)
    a = p.parse_args(); s1 = a.stage1.resolve(); s2 = a.stage2.resolve(); out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()): raise ValueError('use a fresh evidence directory')
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    report = dict(stage1_sha256=sha(s1), stage2_sha256=sha(s2), test_sha256=sha(__file__),
                  oracle_sha256=sha(Path(__file__).with_name('test_edwards_gpu_pipeline.py')),
                  device=a.device, checks=[], complete=False, formal_performance_samples=0)
    (out/'collector.py').write_bytes(Path(__file__).read_bytes())
    env = {k: v for k, v in os.environ.items() if not k.startswith('NTT_')}
    env.update(NTT_S4_DESCENT_CHECK='1', NTT_NO_PROGRESS='1')
    def call(name, command, input=None, good=True):
        r = subprocess.run([str(v) for v in command], input=input, capture_output=True, timeout=600, env=env, cwd=out)
        raw = r.stdout+r.stderr; log = out/(name+'.log'); log.write_bytes(raw)
        assert (r.returncode == 0) == good, (name, r.returncode, raw[-2000:])
        report['checks'].append(dict(name=name, exit=r.returncode, log_sha256=sha(log)))
        (out/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
        print('PASS', name, flush=True); return raw.decode('utf-8', 'replace')
    for bits in (1279, 4423, 8191, 9689):
        save = out/f'm{bits}.save'
        call(f'stage1_m{bits}', [s1, '--curve', 'ed', '--target', 'gpu', '-d', a.device,
            '-sigma', 20260922, '-gpucurves', 1, '--tmp-dir', out/f'tmp_{bits}', '-save', save, 20, 20],
            str((1 << bits)-1).encode())
        n, x, sigma, _, status = records(save)[0]; assert status == 'POINT'
        if bits == 9689:
            ini = out/'s2.ini'; ini.write_text(f'device={a.device}\n')
            result = out/'wide.jsonl'; debug = out/'wide_debug.log'
            call('stage2_m9689', [s2, '--ini', ini, '--save', save, '--b2', 13230, '--d', 210,
                 '--device', a.device, '--results', result, '--log', out/'wide_engine.log',
                 '--debug-log-file', debug, '--short-calibration', 0, '--tune-runtime', 0, '--factor-only'])
            row = json.loads(result.read_text().splitlines()[-1]); a24, _ = curve(n, sigma)
            expected = values(n, x, a24, 13230)
            found = re.findall(r'descent_values:.*?hash=(\d+)', debug.read_text(encoding='utf-8'))
            assert found and found[-1] == expected and not row['factors'] and not row['bad_factors']
            report['checks'][-1]['independent_leaf_hash'] = expected
    save = out/'sigma64.save'
    common = [s1, '--curve', 'ed', '--target', 'gpu', '-d', a.device,
              '-sigma', (1 << 64)-1, '--tmp-dir', out/'sigma64_tmp', '-save', save]
    call('sigma64', [*common, '-gpucurves', 1, 20, 20], str((1 << 127)-1).encode())
    assert records(save)[0][2] == (1 << 64)-1
    text = call('reject_sigma_wrap', [*common, '-gpucurves', 2, 20, 20], str((1 << 127)-1).encode(), good=False)
    assert 'exceeds 2^64-1' in text
    text = call('reject_lcm12', [*common, '-gpucurves', 1, '--lcm', 12, 20, 20], str((1 << 127)-1).encode(), good=False)
    assert '48*lcm' in text
    # An unwritable destination is a failed publication, not a finished task.
    failed = out/'publication_failure'; failed.mkdir()
    (failed/'e_m127_20.save').mkdir()
    todo = failed/'worktodo.txt'; finished = failed/'finished.txt'; ini = failed/'ecm.toml'
    task = 'ECM2=1,2,127,-1,20,0,1,17\n'; todo.write_text(task)
    ini.write_text(f'curve=ed\ntarget=gpu\ndevice={a.device}\nworktodo={todo}\nfinished={finished}\ntmp_dir={failed}\nlog_file=\n')
    text = call('publication_failure', [s1, '-ini', ini], good=False)
    assert todo.read_text() == task and not finished.exists() and 'publication failed' in text
    report['complete'] = True
    (out/'summary.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__': main()
