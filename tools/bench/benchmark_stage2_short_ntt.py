"""Frozen-input Stage2 short-NTT A/B; each case has save, b2, d, batch_mb, fold_mb."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import statistics as stats
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fields(text, prefix):
    lines = [line for line in text.splitlines() if line.startswith(prefix + ':')]
    if len(lines) != 1:
        raise ValueError(f'expected one {prefix}, got {len(lines)}')
    return dict(re.findall(r'(\w+)=([^\s]+)', lines[0]))


def parse(log, result, *, leaf_fixture=False):
    text = log.read_text(encoding='utf-8', errors='replace')
    full = fields(text, 'stage2_full_wall')
    work = fields(text, 'ntt_work')
    s4 = fields(text, 's4_multiply_stats')
    split = fields(text, 'real_batched_split')
    mem = fields(text, 'ntt_workspace_stats')
    digest = fields(text, 'real_batched_rootfold')
    counts = re.search(r'^ntt_length_counts: log2_length:num_of_ntt (.+)$', text, re.M)
    histogram = dict((int(k), int(v)) for k, v in re.findall(r'(\d+):(\d+)', counts[1]))
    assert histogram and all(0 <= k <= 32 and v > 0 and v % 3 == 0 for k, v in histogram.items())
    results = [json.loads(line) for line in result.read_text(encoding='utf-8').splitlines()]
    if len(results) != 1:
        raise ValueError('expected one result')
    r = results[0]
    assert r['status'] == 'stage2_completed' and not r['bad_factors'], r
    # clean denotes absence of extra fixtures, not arithmetic correctness.
    # NTT_TARGET_LEAF_HASH explicitly marks the dedicated diagnostic run dirty.
    assert full['clean'] == ('0' if leaf_fixture else '1'), full
    assert int(s4['gmp_selftest_cases']) > 0 and int(s4['gmp_checked']) > 0, s4
    assert s4['gmp_selftest_bad'] == s4['gmp_check_bad'] == '0', s4
    assert sum(histogram.values()) == int(work['transforms']) == 3 * int(work['convolutions'])
    assert int(work['convolutions']) == int(s4['poly_muls'])
    return dict(engine_seconds=float(full['total']), result_seconds=r['seconds'],
                init_seconds=float(full['init']), main_seconds=float(full['main']),
                chunks=int(work['s4_chunks']), changed=int(work['short_batch_changed']),
                transforms=int(work['transforms']), convolutions=int(work['convolutions']),
                histogram=histogram, full_peak_bytes=int(mem['full_peak_bytes']),
                reduce_seconds=float(s4['t_reduce']), checked=int(s4['gmp_checked']),
                selftests=int(s4['gmp_selftest_cases']), launches=int(s4['launches']),
                phases={k: float(v) for k, v in split.items()},
                correctness=[r['hits'], r['bad_factors'], r['factors'],
                             digest['digest_sum'], digest['digest_xor'], digest['words']],
                shape=fields(text, 'stage2_real'),
                policy=fields(text, 'ntt_batch_policy') if 'ntt_batch_policy:' in text else {})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', type=Path, required=True)
    p.add_argument('--profile', type=Path, required=True)
    p.add_argument('--cases', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', type=int, required=True)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--warmups', type=int, default=1)
    p.add_argument('--timeout', type=int, default=600)
    a = p.parse_args()
    if a.device < 0 or a.repeats < 1 or a.warmups < 0 or a.timeout < 1:
        p.error('invalid device/repeats/warmups/timeout')
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    inputs = out / 'inputs'
    inputs.mkdir()
    frozen = {}

    def freeze(path):
        path = path.resolve()
        dest = inputs / sha(path) / path.name
        dest.parent.mkdir(exist_ok=True)
        if not dest.exists():
            dest.write_bytes(path.read_bytes())
        frozen[str(path)] = sha(path)
        return dest

    exe = a.exe.resolve()
    manifest = json.loads(freeze(exe.parent / 'build_manifest.json').read_text(encoding='utf-8-sig'))
    if sha(exe) != manifest['sha256'].lower():
        raise ValueError('binary differs from manifest')
    freeze(exe)
    freeze(exe.parent / 'gmp-10.dll')
    for entry in manifest['sources']:
        name, _, digest = entry.rpartition('=')
        path = ROOT / name
        if len(digest) == 64 and path.is_file():
            if sha(path) != digest.lower():
                raise ValueError('source differs from manifest: ' + name)
            freeze(path)
    freeze(Path(__file__))
    profile = freeze(a.profile)
    cases = json.loads(freeze(a.cases).read_text(encoding='utf-8'))
    if not isinstance(cases, list) or not cases:
        raise ValueError('nonempty cases array required')
    names = set()
    for case in cases:
        name = case.get('name', '')
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_]+', name) or name in names:
            raise ValueError('case names must be unique, using letters/digits/underscore')
        names.add(name)
        for key in ('b2', 'd', 'batch_mb'):
            if type(case.get(key)) is not int or case[key] < 1:
                raise ValueError('positive integer case field required: ' + key)
        if type(case.get('fold_mb', 640)) is not int or case.get('fold_mb', 640) < 0:
            raise ValueError('fold_mb must be a nonnegative integer')
        case['save'] = str(freeze(Path(case['save'])))
    ini = out / 'ecm.toml'
    ini.write_text('verbose=true\nstage2_debug_log=false\nstage2_short_calibration=false\nstage2_tune_runtime=false\n')
    freeze(ini)
    (out / 'identity.json').write_text(json.dumps(dict(binary_sha256=sha(exe),
        device=a.device, repeats=a.repeats, warmups=a.warmups, cases=cases,
        inputs=frozen, git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        dirty=True, env_ntt={k:v for k,v in os.environ.items() if k.startswith('NTT_')}), indent=2))
    telemetry_file = (out / 'gpu.csv').open('w')
    query = 'timestamp,index,uuid,name,utilization.gpu,clocks.sm,clocks.mem,temperature.gpu,power.draw,memory.used'
    telemetry = subprocess.Popen(['nvidia-smi', '-i', str(a.device), '--query-gpu=' + query,
        '--format=csv', '-lms', '500'], stdout=telemetry_file, stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    rows = []
    try:
        for case in cases:
            reference = None
            for turn in range(a.warmups + a.repeats):
                # Alternate order to reduce monotonic drift bias.
                for enabled in ([0, 1] if turn % 2 == 0 else [1, 0]):
                    name = f"{case['name']}_{'on' if enabled else 'off'}_{turn}"
                    log, result = out / (name + '.driver.log'), out / (name + '.result.jsonl')
                    cmd = [str(exe), '--ini', str(ini), '--save', case['save'], '--b2', str(case['b2']),
                        '--d', str(case['d']), '--curves', '1', '--device', str(a.device),
                        '--batch-mb', str(case['batch_mb']), '--arena-mb', '0',
                        '--owner-budget-mb', str(case.get('fold_mb', 640)), '--carrier-exponent', '0',
                        '--short-calibration', '0', '--tune-profile', str(profile),
                        '--tune-ignore', 'gpu,driver,cuda,backend,environment,memory',
                        '--short-ntt-batch-tune', str(enabled), '--log-level', 'phases',
                        '--results', str(result), '--log', str(log)]
                    start = time.time()
                    with (out / (name + '.stdout.log')).open('w', encoding='utf-8') as console:
                        completed = subprocess.run(cmd, cwd=ROOT, stdout=console, stderr=subprocess.STDOUT,
                            timeout=a.timeout, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                    end = time.time()
                    (out / (name + '.command.json')).write_text(json.dumps(dict(command=cmd,
                        start_unix=start, end_unix=end, returncode=completed.returncode), indent=2))
                    if completed.returncode:
                        raise RuntimeError(name + ' failed; evidence retained')
                    row = parse(log, result)
                    invariant = [row['correctness'], row['histogram'], row['convolutions']]
                    if reference is None:
                        reference = invariant
                    assert invariant == reference, (name, 'A/B arithmetic/count discrepancy')
                    if enabled:
                        assert int(row['policy'].get('samples', 0)) > 0, 'empty policy is not an A/B'
                    row.update(case=case['name'], enabled=enabled, turn=turn, warmup=turn < a.warmups,
                        process_seconds=end-start, start_unix=start, end_unix=end)
                    rows.append(row)
                    with (out / 'runs.jsonl').open('a', encoding='utf-8') as f:
                        f.write(json.dumps(row) + '\n')
                    print(f"{name}: engine={row['engine_seconds']:.4f}s process={end-start:.4f}s "
                          f"chunks={row['chunks']} changed={row['changed']} checks={row['checked']} PASS", flush=True)
    finally:
        telemetry.terminate()
        telemetry.wait(timeout=10)
        telemetry_file.close()
    summaries = []
    for case in cases:
        item = dict(case=case['name'])
        for enabled in (0, 1):
            runs = [r for r in rows if r['case'] == case['name'] and r['enabled'] == enabled and not r['warmup']]
            part = {'samples': len(runs)}
            for key in ('engine_seconds', 'result_seconds', 'process_seconds', 'init_seconds', 'main_seconds',
                        'chunks', 'changed', 'full_peak_bytes', 'reduce_seconds', 'launches'):
                values = [r[key] for r in runs]
                median = stats.median(values)
                part[key] = median
                part[key + '_mad'] = stats.median(abs(x-median) for x in values)
            part['phases'] = {k: stats.median(r['phases'][k] for r in runs) for k in runs[0]['phases']}
            part['histogram'] = runs[0]['histogram']
            part['actual_bits'] = int(runs[0]['shape']['N_bits'])
            item['on' if enabled else 'off'] = part
        item['engine_gain_percent'] = 100*(1-item['on']['engine_seconds']/item['off']['engine_seconds'])
        item['result_gain_percent'] = 100*(1-item['on']['result_seconds']/item['off']['result_seconds'])
        item['process_gain_percent'] = 100*(1-item['on']['process_seconds']/item['off']['process_seconds'])
        summaries.append(item)
    (out / 'summary.json').write_text(json.dumps(summaries, indent=2))
    with (out / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['case','actual_bits','off_seconds','on_seconds','gain_percent','off_chunks','on_chunks','changed'])
        writer.writeheader()
        for s in summaries:
            writer.writerow(dict(case=s['case'], actual_bits=s['off']['actual_bits'],
                off_seconds=s['off']['engine_seconds'], on_seconds=s['on']['engine_seconds'],
                gain_percent=s['engine_gain_percent'], off_chunks=s['off']['chunks'],
                on_chunks=s['on']['chunks'], changed=s['on']['changed']))
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == '__main__':
    main()
