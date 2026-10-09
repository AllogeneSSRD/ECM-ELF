"""Same-binary, fixed-D comparison of generic and Mersenne-carrier Stage2.

Retains every invocation and frozen build/input identities. The check matrix
compares complete leaf vectors after projection to target N; timing matrices
disable that extra projection and retain the mandatory arithmetic checks.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import threading
import time

from bench_stage2_production import fields, freeze, read, sha


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', type=Path, required=True)
    p.add_argument('--save', type=Path, required=True)
    p.add_argument('--carrier-exponent', type=int, required=True)
    p.add_argument('--b2', type=int, required=True)
    p.add_argument('--d', type=int, required=True)
    p.add_argument('--device', type=int, default=1)
    p.add_argument('--arena-mb', type=int, default=6300)
    p.add_argument('--fold-mb', type=int, default=640)
    p.add_argument('--batch-mb', type=int, default=256)
    p.add_argument('--mode', choices=('check', 'timing'), default='check')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--timeout', type=int, default=1200)
    p.add_argument('--resume-check', action='store_true', help='Re-parse retained complete check invocations after a collector-only failure')
    p.add_argument('--fixtures', type=Path, help='Optional independent small-input manifest from prepare_stage2_carrier_inputs.py')
    p.add_argument('--telemetry', action='store_true', help='Sample GPU power/clocks/utilization with read-only nvidia-smi queries every 2 s')
    p.add_argument('--projection-only', action='store_true', help='Check full projected leaves without extra per-point ladder/seed diagnostics; still not a timing sample')
    a = p.parse_args()
    if a.d <= 0 or a.b2 <= 0 or not 2 <= a.carrier_exponent <= 16384:
        p.error('positive fixed D/B2 and a valid carrier exponent are required')
    exe, save, out = a.exe.resolve(), a.save.resolve(), a.output.resolve()
    if a.resume_check and a.mode != 'check':
        p.error('only check matrices permit collector recovery')
    if a.projection_only and a.mode != 'check':
        p.error('--projection-only requires --mode check')
    if out.exists() and any(out.iterdir()) and not a.resume_check:
        raise ValueError('use a fresh output directory; raw evidence is never overwritten')
    out.mkdir(parents=True, exist_ok=True)
    identity = freeze(exe)
    if read(exe.parent / 'build_manifest.json').get('engine') != 'production':
        raise ValueError('the production engine is required')
    tool_sha, save_sha = sha(__file__), sha(save)
    oracle = None
    if a.fixtures:
        prepared = read(a.fixtures)
        if not prepared['complete'] or sha(prepared['reference']) != prepared['reference_sha256']:
            raise ValueError('independent reference identity changed')
        matches = [c for c in prepared['cases'] if c['exponent'] == a.carrier_exponent and c['save_sha256'] == save_sha]
        if len(matches) != 1 or matches[0]['D'] != a.d or matches[0]['B2'] != a.b2:
            raise ValueError('reference fixture differs from the requested curve/shape')
        oracle = matches[0]
    if not a.resume_check:
        (out / 'collector.py').write_bytes(Path(__file__).read_bytes())
    ini = out / 'manual.ini'
    ini.write_text(f'device={a.device}\n', encoding='utf-8')
    env = {k: v for k, v in os.environ.items() if not k.startswith('NTT_')}
    env.update(NTT_D_MODEL='0', NTT_GIANT_SEED_PAIR='1', NTT_GIANT_BASE_CPU='0',
               NTT_GIANT_CHAIN_BLOCK='64', NTT_GIANT_CHAIN_MIN='32768',
               NTT_NO_PROGRESS='1', CUDA_LAUNCH_BLOCKING='0')
    if a.mode == 'check':
        env['NTT_TARGET_LEAF_HASH'] = '1'
        if not a.projection_only:
            env.update(NTT_BABY_DEVICE_CHECK='1', NTT_GFINV_SEG_CHECK='1',
                       NTT_GIANT_SEED_CHECK='1', NTT_GIANT_CHAIN_CHECK='1')
    sequence = ('generic', 'carrier', 'carrier', 'generic',
                'carrier', 'generic', 'generic', 'carrier')
    matrix = ([('check', k) for k in ('generic', 'carrier')] if a.mode == 'check'
              else [('warmup', k) for k in ('generic', 'carrier')] +
                   [('timing', k) for k in sequence])
    data = dict(complete=False, identity=identity, tool_sha256=tool_sha,
                input=dict(save=str(save), save_sha256=save_sha,
                           B2=a.b2, D=a.d, carrier_exponent=a.carrier_exponent),
                mode=a.mode, matrix=matrix, runs=[], oracle=oracle, projection_only=a.projection_only,
                oracle_sha256=sha(a.fixtures) if a.fixtures else None)
    if a.resume_check:
        previous = read(out/'measurements.json')
        if previous['complete'] or any(previous[k] != data[k] for k in ('identity', 'input', 'mode')):
            raise ValueError('recovery requires the original unfinished check matrix')
        if previous.get('projection_only', False) != a.projection_only:
            raise ValueError('recovery must preserve the original diagnostic policy')
        if sha(out/'collector.py') != previous['tool_sha256']:
            raise ValueError('original collector changed')
        for row in previous['runs']:
            if sha(row['debug_log']) != row['debug_sha256'] or sha(row['log']) != row['log_sha256']:
                raise ValueError('previous verified log changed')
            if row['environment'] != {k: v for k, v in env.items() if k.startswith('NTT_') or k == 'CUDA_LAUNCH_BLOCKING'}:
                raise ValueError('recovery environment differs from verified rows')
        previous.setdefault('collector_recoveries', []).append(dict(sha256=tool_sha, reason='Parse both readable and dedicated debug streams'))
        (out/('collector_'+tool_sha[:12]+'.py')).write_bytes(Path(__file__).read_bytes())
        data = previous
        data.pop('error', None)

    def persist():
        (out / 'measurements.json').write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')

    def verify():
        if sha(save) != save_sha or sha(__file__) != tool_sha:
            raise ValueError('save or collector changed during the matrix')
        if a.fixtures and (sha(a.fixtures) != data['oracle_sha256'] or sha(prepared['reference']) != prepared['reference_sha256']):
            raise ValueError('independent fixture/reference changed')
        if (sha(exe) != identity['binary_sha256'] or sha(exe.parent / 'build_manifest.json') != identity['build_sha256'] or
                sha(exe.parent/'frozen_sources_manifest.json') != identity['snapshot_sha256']):
            raise ValueError('binary or manifest changed during the matrix')
        for name, want in identity['sources'].items():
            if sha(exe.parent / 'sources' / name) != want:
                raise ValueError('frozen source changed: ' + name)

    persist()
    try:
        for index, (category, key) in enumerate(matrix):
            verify()
            name = f'{index:02d}_{category}_{key}'
            if any(r['name'] == name for r in data['runs']):
                continue
            log, result = out / (name + '.log'), out / (name + '.jsonl')
            debug = out / (name + '.debug.log')
            command = [str(exe), '--ini', str(ini), '--save', str(save),
                       '--b2', str(a.b2), '--d', str(a.d), '--device', str(a.device),
                       '--carrier-exponent', str(a.carrier_exponent if key == 'carrier' else 0),
                       '--arena-mb', str(a.arena_mb), '--owner-budget-mb', str(a.fold_mb),
                       '--batch-mb', str(a.batch_mb), '--curves', '1', '--factor-only',
                       '--log-level', 'curve', '--log', str(log), '--results', str(result),
                       '--debug-log-file', str(debug)]
            entry = dict(name=name, category=category, key=key, command=command,
                         environment={k: v for k, v in env.items() if k.startswith('NTT_') or k == 'CUDA_LAUNCH_BLOCKING'})
            retained = a.resume_check and log.exists() and result.exists() and debug.exists()
            pending = data.get('pending', {})
            if retained and (pending.get('command') != command or pending.get('returncode') != 0 or
                             pending.get('environment') != entry['environment']):
                raise ValueError('only a recorded successful raw invocation can be recovered')
            data['pending'] = entry if not retained else pending
            persist()
            if retained:
                entry.update(returncode=0, process_seconds=pending['process_seconds'], recovered_raw=True)
                if 'stage2_complete: curves_this_run=1' not in (out/(name+'_driver.log')).read_text(encoding='utf-8'):
                    raise ValueError('raw invocation did not report successful completion')
            else:
                start = time.perf_counter()
                stop, samples = threading.Event(), []
                def sample():
                    while not stop.is_set():
                        try:
                            s = subprocess.run(['nvidia-smi', '--id='+str(a.device),
                                '--query-gpu=timestamp,uuid,power.draw,clocks.sm,temperature.gpu,utilization.gpu,memory.used',
                                '--format=csv,noheader,nounits'], capture_output=True, timeout=5)
                            samples.append(dict(seconds=time.perf_counter()-start, returncode=s.returncode,
                                                csv=s.stdout.decode('utf-8', 'replace').strip(),
                                                stderr=s.stderr.decode('utf-8', 'replace').strip()))
                        except (OSError, subprocess.TimeoutExpired) as exc:
                            samples.append(dict(seconds=time.perf_counter()-start, error=str(exc)))
                        stop.wait(2)
                worker = threading.Thread(target=sample, daemon=True) if a.telemetry else None
                if worker:
                    worker.start()
                try:
                    proc = subprocess.run(command, env=env, capture_output=True, timeout=a.timeout)
                finally:
                    stop.set()
                    if worker:
                        worker.join(timeout=6)
                    if a.telemetry:
                        (out/(name+'_telemetry.json')).write_text(json.dumps(samples, indent=2)+'\n', encoding='utf-8')
                (out / (name + '_driver.log')).write_bytes(proc.stdout + proc.stderr)
                entry.update(returncode=proc.returncode, process_seconds=time.perf_counter()-start)
                data['pending'] = entry
                persist()
                if proc.returncode:
                    raise ValueError(name + ' failed; raw logs retained')
            verify()
            text = log.read_text(encoding='utf-8') + '\n' + debug.read_text(encoding='utf-8')
            rows = [json.loads(s) for s in result.read_text(encoding='utf-8').splitlines() if s.strip()]
            if len(rows) != 1:
                raise ValueError('expected exactly one result')
            r = rows[0]
            for token in ('mont_selftest: cases=2048 mismatches=0', 's4_div_check: cases=800 bad=0',
                          'gmp_selftest_bad=0', 'gmp_check_bad=0', 'pending=0'):
                if token not in text:
                    raise ValueError('required arithmetic check missing: ' + token)
            if r['B2'] != a.b2 or r['requested_D'] != a.d or r['device'] != a.device or r['bad_factors']:
                raise ValueError('result/input mismatch')
            n = int(r['N_hex'], 16)
            if ((1 << a.carrier_exponent)-1) % n or any(not 1 < int(f) < n or n % int(f) for f in r['factors']):
                raise ValueError('invalid target/carrier or factor')
            if oracle:
                if n != int(oracle['N_hex'], 16) or r['B1'] != oracle['B1'] or r['sigma'] != oracle['sigma']:
                    raise ValueError('independent reference input differs')
                expected = {v['gcd'] for v in oracle['nonunits'] if 1 < v['gcd'] < n}
                if any(not any(int(f) % g == 0 for f in r['factors']) for g in expected):
                    raise ValueError('independently known denominator factor missing')
            entry.update(result=r, wall=fields(text, 'stage2_full_wall'),
                         phases=fields(text, 'real_batched_split'), coverage=fields(text, 's4_multiply_stats'),
                         shape=fields(text, 'real_shape'), modulus=fields(text, 'stage2_modulus'),
                         workspace=fields(text, 'ntt_workspace_stats'),
                         fold=fields(text, 'real_batched_folddevice'),
                         leaf=fields(text, 'target_descent_values') if a.mode == 'check' else None,
                         log=str(log), log_sha256=sha(log), debug_log=str(debug),
                         debug_sha256=sha(debug), result_sha256=sha(result))
            if oracle and oracle['unit'] and a.mode == 'check':
                if entry['leaf'] != {k: str(v) for k, v in oracle['expected_leaf'].items()}:
                    raise ValueError('independent target-ring monic leaf oracle mismatch')
            data['runs'].append(entry)
            del data['pending']
            persist()
            print(name, entry['wall']['total'], 's', flush=True)
        targets = {(r['result']['N_hex'], r['result']['B1'], r['result']['sigma']) for r in data['runs']}
        factors = {tuple(sorted(r['result']['factors'])) for r in data['runs']}
        if len(targets) != 1 or len(factors) != 1:
            raise ValueError('target input or factor output changed between arms')
        if a.mode == 'check' and (not oracle or oracle['unit']) and len({json.dumps(r['leaf'], sort_keys=True) for r in data['runs']}) != 1:
            raise ValueError('complete target-projected leaf fingerprint mismatch')
        if oracle and not oracle['unit']:
            data['leaf_comparison'] = 'Nonunit fallback X values depend on projective scale; verify target factors instead of equating monic leaf vectors.'
        if a.mode == 'timing':
            timed = [r for r in data['runs'] if r['category'] == 'timing']
            means = {k: statistics.mean(float(r['wall']['total']) for r in timed if r['key'] == k) for k in ('generic', 'carrier')}
            groups = []
            for lo in (0, 4):
                m = {k: statistics.mean(float(r['wall']['total']) for r in timed[lo:lo+4] if r['key'] == k) for k in means}
                groups.append(100 * (1 - m['carrier'] / m['generic']))
            data['summary'] = dict(mean_seconds=means, reduction_percent=100*(1-means['carrier']/means['generic']),
                                   groups_reduction_percent=groups)
        verify()
        data['complete'] = True
    except Exception as exc:
        data['error'] = str(exc)
        raise
    finally:
        persist()
    print(json.dumps(data.get('summary', dict(passed=len(data['runs'])))), flush=True)


if __name__ == '__main__':
    main()
