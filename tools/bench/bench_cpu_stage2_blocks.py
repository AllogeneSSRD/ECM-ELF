"""Pair exact and whole-block CPU Stage2 runs on the 8k/16k Edwards cofactors (Windows)."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

from stage2_windows_job import WindowsJob


FACTORS = {8011: 80111, 16319: 270811405193137, 65521: 790298445919, 106501: 15975151, 262139: None}
COMPILERS = {b'cl.exe', b'clang-cl.exe', b'link.exe', b'nvcc.exe', b'cicc.exe', b'ptxas.exe'}


def compiling():
    processes = subprocess.check_output(['tasklist', '/fo', 'csv', '/nh']).lower().splitlines()
    return any(row.split(b'"', 2)[1] in COMPILERS for row in processes if row.startswith(b'"'))


def measure(folder):
    case = json.loads((folder / 'input.json').read_bytes())
    affinity = ','.join(map(str, case['cpus']))
    (folder / 'ecm.toml').write_text(f'config_version=4\n[shared]\ncpu_affinity="{affinity}"\n')
    command = [case['exe'], '--config', 'ecm.toml', '--save', case['save'], '--exponent', case['e'],
               '--curves', 1, '--backend', 'gwpoly',
               '--d', case['d'], '--b2', case['requested'], '--threads', len(case['cpus']),
               '--memory-mib', case['memory'], '--results', 'result.jsonl', '--submission', 'public.txt',
               '--tune-runtime', 0, '--tune-profile', 'empty-profile.toml']
    factor = FACTORS[case['e']]
    if factor is not None:
        command += ['--known-factor', factor]
    if case['arm'] != 'baseline':
        command += ['--b2-round-chain', int(case['arm'] == 'rounded')]
    if case['check']:
        command += ['--gw-check']
    command = list(map(str, command))
    (folder / 'command.json').write_text(json.dumps(command))
    job = WindowsJob()
    overlap = False
    try:
        job.cap_current_process_tree(case['memory'])
        while compiling():
            time.sleep(3)
        with (folder / 'worker.log').open('wb') as log, subprocess.Popen(
                command, cwd=folder, stdout=log, stderr=subprocess.STDOUT) as proc:
            while True:
                try:
                    rc = proc.wait(2)
                    break
                except subprocess.TimeoutExpired:
                    overlap |= compiling()
        usage = job.accounting()
    finally:
        job.close()
    meta = dict(rc=rc, compile_overlap=overlap, **usage)
    (folder / 'accounting.json').write_text(json.dumps(meta))
    if rc or overlap or usage['peak_process_commit_bytes'] > case['memory'] * 1024**2:
        raise RuntimeError(f'Invalid benchmark sample: {folder}, {meta}')
    row = json.loads((folder / 'result.jsonl').read_bytes())
    log = (folder / 'worker.log').read_text(encoding='utf-8')
    engine = row['stage2-engine']
    assert f'logical_cpus={affinity}\n' in log
    assert (row['exponent'], row.get('known-factors', []), row['b1'], row['Edwards']['sigma'], row['status']) == (
        case['e'], [str(factor)] if factor is not None else [], 1000, 26, 'NF')
    assert engine['checked-values'] == (3 if case['check'] else 0) and engine['max-roundoff'] <= .4
    blocks, points = map(int, re.search(r'\bblocks=(\d+) giants=(\d+)', log).groups())
    if case['arm'] == 'rounded':
        assert row['b2'] >= case['requested'] and points == blocks * row['poly-size']
        assert (engine['requested-B2'], engine['actual-B2']) == (case['requested'], row['b2'])
    else:
        assert row['b2'] == case['requested']
    phases = dict(field.split('=') for field in re.search(r'^gwpoly-phases: (.+)$', log, re.M)[1].split())
    return dict(actual=row['b2'], d=row['d'], p=row['poly-size'], fft=row['fft-length'],
                blocks=blocks, giants=points, seconds=engine['seconds'], err=engine['max-roundoff'],
                phases={k: float(v) for k, v in phases.items()},
                peak_mib=usage['peak_process_commit_bytes'] / 1024**2, dll=engine['gw-build'],
                isa=engine['polymult-isa'], fft_description=engine['fft-description'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for arg in ('exe', 'saves', 'output'):
        parser.add_argument('--' + arg, type=Path, required=True)
    parser.add_argument('--baseline', type=Path, help='Optional older CPU executable, measured at exact B2')
    parser.add_argument('--exponents', nargs='+', type=int, choices=FACTORS, default=[8011, 16319])
    parser.add_argument('--b2', type=int, nargs='+', default=[9_000_000_000, 44_000_000_000])
    parser.add_argument('--cpus', type=int, nargs='+', required=True, help='One logical processor per physical core')
    parser.add_argument('--memory-mib', type=int, default=4096)
    parser.add_argument('--d', type=int, default=0)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1 or args.memory_mib < 1 or min(args.cpus) < 0 or len(set(args.cpus)) != len(args.cpus):
        parser.error('Expected positive budget/repeats and distinct non-negative CPU indices')
    if min(args.b2) <= 1000 or len(set(args.b2)) != len(args.b2) or len(set(args.exponents)) != len(args.exponents):
        parser.error('Choose distinct exponents and distinct B2 values above B1=1000')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    arms = {'exact': args.exe, 'rounded': args.exe}
    if args.baseline:
        arms['baseline'] = args.baseline
    inputs = {e: (args.saves / f'm{e}.save').resolve(strict=True) for e in args.exponents}
    files = [*inputs.values(), *(exe.resolve(strict=True) for exe in arms.values())]
    files.extend(p.with_name('gwnum.dll') for p in list(files) if p.suffix == '.exe' and p.with_name('gwnum.dll').exists())
    (out / 'sha256.json').write_text(json.dumps({str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}, indent=2))
    for rep in range(args.repeats + 1):
        order = list(arms)
        # Alternate A/B; for three binaries rotate which occupies the middle slot.
        if rep:
            shift = ((rep-1)//2) % len(order)
            order = order[shift:] + order[:shift]
            if rep % 2 == 0:
                order.reverse()
        for e, save in inputs.items():
            for bound in args.b2:
                for arm in (order if rep else ['rounded']):
                    folder = out / f'm{e}_b{bound}_{arm}_{rep}'
                    folder.mkdir()
                    case = dict(e=e, save=str(save), exe=str(arms[arm].resolve()), arm=arm, requested=bound,
                                memory=args.memory_mib, cpus=args.cpus, d=args.d, check=rep == 0)
                    (folder / 'input.json').write_text(json.dumps(case))
                    subprocess.run([sys.executable, __file__, '--sample', str(folder)], check=True)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--sample':
        sample = Path(sys.argv[2])
        result = measure(sample)
        (sample / 'measurement.json').write_text(json.dumps(result, indent=2))
        print(sample.name, result['actual'], result['fft'], round(result['seconds'], 3), flush=True)
    else:
        main()
