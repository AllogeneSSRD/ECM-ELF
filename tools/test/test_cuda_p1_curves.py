"""Compare all selected Montgomery instance groups against a frozen full build."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--stage1', type=Path, required=True)
    p.add_argument('--tiers', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', type=int, default=1)
    a = p.parse_args(); out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    exes = dict(old=a.baseline.resolve(), new=a.stage1.resolve())
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    identity = {k: sha(v) for k, v in exes.items()}
    widths = sorted({int(w) for w in a.tiers.split(',') if int(w) <= 16384})
    primes = [v for v in range(2, 258) if all(v%d for d in range(2, math.isqrt(v)+1))]
    small = math.prod(primes)
    config = out/'fixture.toml'
    config.write_text('[stage1]\nauto_stage2 = 0\nedwards_stage2_transfer = false\n')
    env = {k: v for k, v in os.environ.items() if not k.startswith(('NTT_', 'ECM_PRAC_', 'ECM_GPU_STAGE1_'))}
    report = dict(complete=False, identity=identity, device=a.device, cases=[], performance_samples=0)
    def persist(): (out/'result.json').write_text(json.dumps(report, indent=2)+'\n')
    def records(path):
        rows = []
        for line in path.read_text(encoding='utf-8-sig').splitlines():
            if 'SIGMA=' not in line: continue
            fields = dict(v.strip().split('=', 1) for v in line.split(';') if '=' in v)
            fields.pop('WHO', None); fields.pop('TIME', None); rows.append(fields)
        assert rows, path
        return rows
    for width in widths:
        n = (1 << (width-6))-1
        while math.gcd(n, small) != 1: n -= 2
        assert n.bit_length() == width-6
        for param in (0, 2, 3):
            name = f'w{width}_param{param}'; pair = []
            for which, exe in exes.items():
                directory = out/(name+'_'+which); directory.mkdir(); save = directory/'point.save'
                command = [str(exe), '--config', str(config), '--curve', 'mont', '--target', 'gpu',
                           '-d', str(a.device), '--gpu-param', str(param), '-sigma', '26', '-gpucurves', '2',
                           '--tmp-dir', str(directory), '--exp-cache', 'off', '--ckpt', '0',
                           '-save', str(save), '20', '0']
                (directory/'command.json').write_text(json.dumps(command, indent=2)+'\n')
                value = '0x'+format(n, 'x')+'\n'; (directory/'input.txt').write_text(value)
                process = subprocess.run(command, input=value, cwd=directory, env=env, capture_output=True,
                                         text=True, errors='replace', timeout=180)
                (directory/'driver.log').write_text(process.stdout+process.stderr, encoding='utf-8')
                assert process.returncode == 0, (name, which, process.stderr[-2000:])
                pair.append(records(save))
            assert pair[0] == pair[1], name
            report['cases'].append(dict(case=name, modulus_bits=n.bit_length(), saved_records=len(pair[0]), matched=True))
            persist(); print('PASS', name, flush=True)
    assert {k: sha(v) for k, v in exes.items()} == identity
    report['complete'] = True; persist(); print(json.dumps(dict(complete=True, cases=len(report['cases']))))


if __name__ == '__main__': main()
