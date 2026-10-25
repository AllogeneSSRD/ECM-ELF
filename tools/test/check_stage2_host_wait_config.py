"""Real parent/worker TOML, CLI and expert environment precedence checks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', type=Path, required=True)
    p.add_argument('--save', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', type=int, default=1)
    a = p.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    exe = a.exe.resolve()
    identity = hashlib.sha256(exe.read_bytes()).hexdigest()
    cases = [
        ('default', '', {}, [], (0,0,None)),
        ('toml_on', 'stage2_low_cpu_wait=true\n', {}, [], (1,0,4)),
        ('toml_above_env', 'stage2_low_cpu_wait=false\n', {'NTT_HOST_EVENT_WAIT':'1','NTT_HOST_READBACK_WAIT':'1'}, [], (0,0,0)),
        ('expert_env', '', {'NTT_HOST_EVENT_WAIT':'1','NTT_HOST_READBACK_WAIT':'1'}, [], (1,1,None)),
        ('expert_blocking', '', {'NTT_CUDA_WAIT_MODE':'4'}, [], (0,0,4)),
        ('toml_above_blocking', 'stage2_low_cpu_wait=false\n', {'NTT_CUDA_WAIT_MODE':'4'}, [], (0,0,0)),
        ('cli_above_toml', 'stage2_low_cpu_wait=true\n', {}, ['--low-cpu-wait','0'], (0,0,0)),
        ('cli_on', 'stage2_low_cpu_wait=false\n', {}, ['--low-cpu-wait','1'], (1,0,4)),
        ('worker_override', 'stage2_low_cpu_wait=true\n[workers.2.stage2]\nstage2_low_cpu_wait=false\n', {}, ['--worker','2'], (0,0,0)),
        ('invalid_toml', 'stage2_low_cpu_wait=2\n', {}, [], None),
        ('invalid_cli', '', {}, ['--low-cpu-wait','2'], None)]
    rows = []
    for name, value, changes, extra, expected in cases:
        config = out/(name+'.toml')
        config.write_text('[stage2]\nstage2_tune_runtime=false\nstage2_short_calibration=false\n'+value, encoding='utf-8')
        result,log,debug = [out/(name+s) for s in ('.jsonl','.log','.debug.log')]
        cmd = [str(exe), '--config', str(config), '--save', str(a.save.resolve()), '--device', str(a.device),
               '--curves', '1', '--b2', '260000000', '--d', '30030', '--factor-only',
               '--log', str(log), '--debug-log-file', str(debug), '--results', str(result), *extra]
        env = {k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
        run = subprocess.run(cmd, cwd=out, env=env|changes, capture_output=True, timeout=180)
        (out/(name+'.stdout.log')).write_bytes(run.stdout+run.stderr)
        if expected is None:
            if not run.returncode or (result.exists() and result.stat().st_size):
                raise AssertionError(name+': invalid value accepted/published')
        else:
            if run.returncode:
                raise AssertionError(name+': worker failed')
            text = debug.read_text(encoding='utf-8')
            all_text=text+log.read_text(encoding='utf-8')
            stats = dict(re.findall(r'(\w+)=([^\s|]+)', re.findall(r'^stage2_host_wait_stats: (.*)$', text, re.M)[-1]))
            if (int(stats['adaptive']),int(stats['readback'])) != expected[:2] or stats['timing_fifo']!='1':
                raise AssertionError(name+': resolved policy did not reach child')
            if expected[2] is not None:
                flags=re.findall(r'^stage2_cuda_wait: .*?after=(\d+)',all_text,re.M)
                if not flags or int(flags[-1])&7 != expected[2]:
                    raise AssertionError(name+': device scheduling flags differ')
            if not all(s in all_text for s in ('gmp_check_bad=0','gmp_selftest_bad=0','pending=0','clean=1')):
                raise AssertionError(name+': arithmetic verification missing')
        if hashlib.sha256(exe.read_bytes()).hexdigest()!=identity:
            raise ValueError('Binary changed')
        rows.append(dict(name=name, passed=True, expected=expected, exit_code=run.returncode, command=cmd))
        print(name, 'passed', flush=True)
    (out/'summary.json').write_text(json.dumps(dict(passed=True, binary_sha256=identity, cases=rows), indent=2), encoding='utf-8')


if __name__=='__main__':
    main()
