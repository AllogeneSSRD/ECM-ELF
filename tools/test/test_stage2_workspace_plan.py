"""Check native planning uses the allocator's two/three-buffer policy.

Initializes CUDA only to query a plan; executes no curve arithmetic. Retains
commands, inputs, frozen build identity and every native JSON response.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tools/bench'))
from bench_stage2_production import freeze, sha
from calibrate_stage2_d import phi


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', type=Path, required=True)
    p.add_argument('--save', type=Path, required=True)
    p.add_argument('--carrier-exponent', type=int, default=0)
    p.add_argument('--device', type=int, default=1)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    out, exe, save = a.output.resolve(), a.exe.resolve(), a.save.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('use a fresh output directory')
    out.mkdir(parents=True, exist_ok=True)
    identity, save_sha, tool_sha = freeze(exe), sha(save), sha(__file__)
    ini = out/'manual.ini'
    ini.write_text(f'device={a.device}\n', encoding='utf-8')
    rows = []
    data = dict(complete=False, identity=identity, save=str(save), save_sha256=save_sha,
                tool_sha256=tool_sha, curves_executed=0, rows=rows)
    try:
        for d in (810810, 1021020):
            for pool, reuse in ((1,0), (1,1), (0,0), (0,1)):
                name = f'd{d}_pool{pool}_reuse{reuse}'
                env = {k: v for k, v in os.environ.items() if not k.startswith('NTT_')}
                env.update(NTT_D_MODEL='0', NTT_ARENA_WORKSPACE_POOL=str(pool),
                           NTT_WORKSPACE_REUSE_BQ=str(reuse))
                command = [str(exe), '--ini', str(ini), '--save', str(save),
                    '--carrier-exponent', str(a.carrier_exponent), '--device', str(a.device),
                    '--b2', '2600000000000', '--d', str(d), '--curves', '1',
                    '--arena-mb', '6300', '--owner-budget-mb', '1024', '--plan-only']
                proc = subprocess.run(command, env=env, capture_output=True, timeout=60)
                (out/(name+'.log')).write_bytes(proc.stdout+proc.stderr)
                if proc.returncode:
                    raise ValueError(name+' failed; raw output retained')
                candidates = [json.loads(line) for line in proc.stdout.decode('utf-8').splitlines()
                              if line.startswith('{')]
                if len(candidates) != 1:
                    raise ValueError('expected one native JSON plan')
                plan = candidates[0]
                buffers = 2 if pool and reuse else 3
                degree = phi(d)//2
                nf, nt = plan['fold_length'], plan['tree_length']
                expected_arena = 8*(buffers*nf+2*degree+1+
                                    2*(buffers*nt+2*(degree//2+1)-1))
                expected = dict(type='stage2_plan', curves_executed=0, D=d, P=degree,
                                workspace_buffers=buffers, carrier_exponent=a.carrier_exponent,
                                fold_big_bytes=8*buffers*nf, arena_estimate_bytes=expected_arena)
                for key, value in expected.items():
                    if plan[key] != value:
                        raise ValueError(f'{name}: {key}={plan[key]} expected {value}')
                if plan['arena_estimate_fits'] != (expected_arena <= (6300 << 20)):
                    raise ValueError('arena fit predicate differs from reported estimate')
                if sha(exe) != identity['binary_sha256'] or sha(save) != save_sha or sha(__file__) != tool_sha:
                    raise ValueError('build, input or collector changed')
                rows.append(dict(name=name, command=command,
                    environment={k: v for k, v in env.items() if k.startswith('NTT_')},
                    plan=plan, log_sha256=sha(out/(name+'.log'))))
                print(name, 'OK', flush=True)
        data['complete'] = True
    except Exception as exc:
        data['error'] = str(exc)
        raise
    finally:
        (out/'checks.json').write_text(json.dumps(data, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
