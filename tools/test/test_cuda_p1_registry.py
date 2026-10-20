"""Query actual CUDA factories, including restricted gaps and per-family ceilings."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage1', type=Path, required=True)
    p.add_argument('--tiers', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', type=int, default=1)
    p.add_argument('--dev', action='store_true')
    p.add_argument('--no-param2', action='store_true')
    a = p.parse_args(); out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    exe = a.stage1.resolve(); identity = hashlib.sha256(exe.read_bytes()).hexdigest()
    selected = sorted(set(map(int, a.tiers.split(','))))
    config = out/'fixture.toml'; config.write_text('[stage1]\nauto_stage2 = 0\n')
    results = []
    for model, param in [('ed', None), ('mont', 0), ('mont', 2), ('mont', 3)]:
        expected = [w for w in selected if model == 'ed' or (w <= 16384 and (not a.dev or w <= 1024))]
        if param == 2 and a.no_param2: expected = []
        carry = 1 if model == 'ed' else 6
        probes = [0]+sorted({max(1, w-carry) for w in selected}|{w-carry+1 for w in selected})
        for bits in probes:
            command = [str(exe), '--config', str(config), '--gpu-info', '--curve', model, '-d', str(a.device)]
            if param is not None: command += ['--gpu-param', str(param)]
            if bits: command += ['--bits', str(bits)]
            process = subprocess.run(command, cwd=out, capture_output=True, text=True, errors='replace', timeout=60)
            log = process.stdout+process.stderr
            name = f'{model}_{param}_{bits}'
            (out/(name+'.log')).write_text(log, encoding='utf-8')
            rows = [dict((k, int(v)) for k, v in re.findall(r'(\w+)=(\d+)', line))
                    for line in log.splitlines() if line.startswith('tier ')]
            wanted = expected if not bits else [w for w in expected if w >= bits+carry][:1]
            assert [row['bits'] for row in rows] == wanted, (name, wanted, log)
            assert (process.returncode == 0) == bool(wanted), (name, process.returncode, log)
            if wanted:
                assert f'carry_bits={carry}' in log
                for row in rows:
                    assert row['tpi'] == (4 if row['bits'] <= 512 else 8 if row['bits'] <= 2048 else 16 if row['bits'] <= 8192 else 32)
                    assert row['tpb'] == 128 and row['ipb'] == 128//row['tpi']
            results.append(dict(case=name, passed=True))
    assert hashlib.sha256(exe.read_bytes()).hexdigest() == identity
    result = dict(complete=True, binary_sha256=identity, device=a.device, tiers=selected,
                  dev=a.dev, no_param2=a.no_param2, checks=len(results), cases=results)
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(complete=True, checks=len(results))))


if __name__ == '__main__': main()
