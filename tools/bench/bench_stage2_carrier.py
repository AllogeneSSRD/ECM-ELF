"""Same-binary comparison of Stage2 carriers, workspace layouts or fixed-D plans.

Retains every invocation and frozen build/input identities. The check matrix
compares fingerprints covering all leaves after projection to target N; timing matrices
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
from stage2_memory_ledger import parse as parse_memory_ledger


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', type=Path, required=True)
    p.add_argument('--save', type=Path, required=True)
    p.add_argument('--carrier-exponent', type=int, default=0)
    p.add_argument('--comparison', choices=('carrier', 'workspace-bq', 'plan', 'chunk','phase-output','owner-cache','giant-chunk'), default='carrier')
    p.add_argument('--b2', type=int, required=True)
    p.add_argument('--d', type=int, required=True)
    p.add_argument('--candidate-d', type=int, help='Second fixed D for a two-buffer plan timing comparison; budgets and arithmetic stay fixed')
    p.add_argument('--device', type=int, default=1)
    p.add_argument('--arena-mb', type=int, default=6300)
    p.add_argument('--fold-mb', type=int, default=640)
    p.add_argument('--batch-mb', type=int, default=256)
    p.add_argument('--baby-mb', type=int, default=512)
    p.add_argument('--giant-point-kb',type=int,default=262144,help='Common X/Z point budget; seeds/segments are additional payload')
    p.add_argument('--mode', choices=('check', 'timing'), default='check')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--timeout', type=int, default=1200)
    p.add_argument('--resume-check', action='store_true', help='Re-parse retained complete check invocations after a collector-only failure')
    p.add_argument('--fixtures', type=Path, help='Optional independent small-input manifest from prepare_stage2_carrier_inputs.py')
    p.add_argument('--telemetry', action='store_true', help='Sample GPU power/clocks/utilization with read-only nvidia-smi queries every 2 s')
    p.add_argument('--projection-only', action='store_true', help='Check full projected leaves without extra per-point ladder/seed diagnostics; still not a timing sample')
    p.add_argument('--workspace-fixture', action='store_true', help='Run the independent two/three-buffer allocation, alias, export and deferred-carry fixture in check mode')
    p.add_argument('--single-arm', choices=('three_buffer', 'two_buffer'), help='Exploratory check of one workspace layout, without claiming a cross-layout comparison')
    p.add_argument('--require-resident', action='store_true', help='Require device fold, scaled root and frontier instead of accepting their fallbacks')
    p.add_argument('--trim-phase-raw', action='store_true', help='Reclaim dead raw staging at inverse/fold and fold/descent boundaries in both arms')
    p.add_argument('--memory-ledger', action='store_true', help='Track every successful engine cudaMalloc/free, including transient allocations; diagnostics only')
    p.add_argument('--request-audit', action='store_true', help='Audit ordered multiplication requests; diagnostics only')
    a = p.parse_args()
    valid_exponent = 2 <= a.carrier_exponent <= 16384 or (a.comparison != 'carrier' and a.carrier_exponent == 0)
    if a.d <= 0 or a.b2 <= 0 or not valid_exponent or min(a.arena_mb, a.fold_mb, a.batch_mb,a.baby_mb) <= 0:
        p.error('positive fixed D/B2 and a valid carrier exponent are required')
    if not 0<a.giant_point_kb<=(2**64-1)//1024:
        p.error('invalid point budget')
    if a.workspace_fixture and a.mode != 'check':
        p.error('--workspace-fixture requires --mode check')
    if a.request_audit and a.mode != 'check':
        p.error('--request-audit requires --mode check; instrumentation is outside formal timing')
    if a.single_arm and (a.mode != 'check' or a.comparison != 'workspace-bq'):
        p.error('--single-arm requires a workspace-bq check')
    if a.comparison == 'plan':
        if a.mode != 'timing' or not a.candidate_d or a.candidate_d <= 0 or a.candidate_d == a.d:
            p.error('plan requires timing and a positive --candidate-d different from --d; validate each D separately first')
        if a.fixtures:
            p.error('plan timing uses separately validated D candidates, not a single-D fixture')
    elif a.candidate_d is not None:
        p.error('--candidate-d requires --comparison plan')
    if a.resume_check and a.comparison != 'carrier':
        p.error('collector recovery currently supports carrier comparisons only')
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
               NTT_GIANT_POINT_BUDGET_KB=str(a.giant_point_kb),NTT_GIANT_CHUNK_FLOOR='0',
               NTT_NO_PROGRESS='1', NTT_BABY_DEVICE_MAX_MB=str(a.baby_mb), CUDA_LAUNCH_BLOCKING='0')
    env['NTT_PHASE_TRIM_RAW'] = '1' if a.trim_phase_raw else '0'
    env['NTT_MEMORY_LEDGER'] = '1' if a.memory_ledger else '0'
    env['NTT_REQUEST_AUDIT'] = '1' if a.request_audit else '0'
    if a.workspace_fixture:
        env['NTT_ARENA_WORKSPACE_TEST'] = '1'
    if a.mode == 'check':
        env['NTT_TARGET_LEAF_HASH'] = '1'
        if not a.projection_only:
            env.update(NTT_BABY_DEVICE_CHECK='1', NTT_GFINV_SEG_CHECK='1',
                       NTT_GIANT_SEED_CHECK='1', NTT_GIANT_CHAIN_CHECK='1')
    keys = {'carrier': ('generic', 'carrier'), 'workspace-bq': ('three_buffer', 'two_buffer'),
            'plan': ('baseline_d', 'candidate_d'),
            'chunk': ('legacy_chunk', 'workspace_chunk'),
            'phase-output': ('retained_output','trimmed_output'),
            'owner-cache': ('kept_cache','trimmed_cache'),
            'giant-chunk': ('legacy_points','bounded_points')}[a.comparison]
    sequence = tuple(keys[i] for i in (0, 1, 1, 0, 1, 0, 0, 1))
    matrix = ([('check', k) for k in ((a.single_arm,) if a.single_arm else keys)] if a.mode == 'check'
              else [('warmup', k) for k in keys] +
                   [('timing', k) for k in sequence])
    data = dict(complete=False, identity=identity, tool_sha256=tool_sha,
                input=dict(save=str(save), save_sha256=save_sha,
                           B2=a.b2, D=a.d, carrier_exponent=a.carrier_exponent),
                mode=a.mode, comparison=a.comparison, matrix=matrix, runs=[], oracle=oracle,
                projection_only=a.projection_only, workspace_fixture=a.workspace_fixture, single_arm=a.single_arm,
                require_resident=a.require_resident,
                trim_phase_raw=a.trim_phase_raw,
                giant_point_kb=a.giant_point_kb,
                memory_ledger=a.memory_ledger,memory_parser_sha256=sha(Path(__file__).with_name('stage2_memory_ledger.py')),
                budgets_mib=dict(arena=a.arena_mb,fold=a.fold_mb,batch=a.batch_mb,baby=a.baby_mb),
                oracle_sha256=sha(a.fixtures) if a.fixtures else None)
    if a.comparison == 'plan':
        data['input']['candidate_D'] = a.candidate_d
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
        if sha(Path(__file__).with_name('stage2_memory_ledger.py'))!=data['memory_parser_sha256']:
            raise ValueError('memory ledger parser changed during the matrix')
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
            use = env.copy()
            if a.comparison == 'workspace-bq':
                use['NTT_WORKSPACE_REUSE_BQ'] = '1' if key == 'two_buffer' else '0'
            elif a.comparison in ('plan','chunk','phase-output','owner-cache','giant-chunk'):
                use['NTT_WORKSPACE_REUSE_BQ'] = '1'
            if a.comparison == 'phase-output':
                use['NTT_PHASE_TRIM_OUTPUT'] = '1' if key == 'trimmed_output' else '0'
            if a.comparison == 'owner-cache':
                use['NTT_PHASE_TRIM_OUTPUT'] = '1'
                use['NTT_OWNER_TRIM_FUSE'] = '1' if key == 'trimmed_cache' else '0'
            if a.comparison == 'giant-chunk':
                use.update(NTT_PHASE_TRIM_OUTPUT='1',NTT_OWNER_TRIM_FUSE='1',
                           NTT_GIANT_CHUNK_FLOOR='1' if key=='bounded_points' else '0')
            if a.comparison == 'chunk':
                use['NTT_S4_WORKSPACE_BUDGET'] = '1' if key == 'workspace_chunk' else '0'
            exponent = a.carrier_exponent if a.comparison != 'carrier' or key == 'carrier' else 0
            run_d = a.candidate_d if key == 'candidate_d' else a.d
            command = [str(exe), '--ini', str(ini), '--save', str(save),
                       '--b2', str(a.b2), '--d', str(run_d), '--device', str(a.device),
                       '--carrier-exponent', str(exponent),
                       '--arena-mb', str(a.arena_mb), '--owner-budget-mb', str(a.fold_mb),
                       '--batch-mb', str(a.batch_mb), '--curves', '1', '--factor-only',
                       '--log-level', 'curve', '--log', str(log), '--results', str(result),
                       '--debug-log-file', str(debug)]
            entry = dict(name=name, category=category, key=key, command=command,
                         environment={k: v for k, v in use.items() if k.startswith('NTT_') or k == 'CUDA_LAUNCH_BLOCKING'})
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
                    proc = subprocess.run(command, env=use, capture_output=True, timeout=a.timeout)
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
            if r['B2'] != a.b2 or r['requested_D'] != run_d or r['device'] != a.device or r['bad_factors']:
                raise ValueError('result/input mismatch')
            if r['carrier_exponent'] != exponent:
                raise ValueError('wrong arithmetic carrier')
            n = int(r['N_hex'], 16)
            if (a.carrier_exponent and ((1 << a.carrier_exponent)-1) % n) or any(not 1 < int(f) < n or n % int(f) for f in r['factors']):
                raise ValueError('invalid target/carrier or factor')
            if a.workspace_fixture and fields(text, 'ntt_workspace_check')['bad'] != '0':
                raise ValueError('workspace fixture failed')
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
            if a.memory_ledger:
                entry['memory_ledger'] = parse_memory_ledger(text)
                if a.workspace_fixture and fields(text,'stage2_memory_ledger_check')['bad']!='0':
                    raise ValueError('memory ledger fixture failed')
            if a.comparison in ('workspace-bq', 'plan', 'chunk','phase-output','owner-cache','giant-chunk'):
                entry['layout'] = fields(text, 'ntt_workspace_layout')
                if entry['layout']['reuse_bq_requested'] != use['NTT_WORKSPACE_REUSE_BQ']:
                    raise ValueError('workspace policy differs from requested arm')
                if (int(entry['layout']['alias_calls']) > 0) != (key != 'three_buffer'):
                    raise ValueError('workspace alias execution differs from requested arm')
            if a.comparison in ('owner-cache','giant-chunk'):
                entry['cache_trim'] = [fields(line,'stage2_cache_trim') for line in text.splitlines()
                    if line.startswith('stage2_cache_trim:')]
                entry['cache_stats'] = fields(text,'ntt_phase_cache_stats')
                entry['arena_accounting'] = fields(text,'ntt_arena_accounting')
                entry['phase_memory'] = [fields(line,'s4_phase_memory') for line in text.splitlines()
                    if line.startswith('s4_phase_memory:')]
                if entry['arena_accounting']['mismatch']!='0' or any(v['subset_accounting_version']!='2' for v in entry['phase_memory']):
                    raise ValueError('actual arena/subset accounting differs from validated contract')
                if key=='kept_cache' and int(entry['cache_stats']['evictions']):
                    raise ValueError('control unexpectedly evicted phase caches')
            if a.comparison=='giant-chunk':
                point=entry['point_plan']=fields(text,'giant_chunk_plan')
                done=entry['point_done']=fields(text,'giant_chunk_done')
                entry['giant_seed']=fields(text,'real_giant_seed')
                entry['device_leaf']=fields(text,'device_gleaf')
                P,nw,budget=(int(point[f]) for f in ('P','nw','budget_bytes'))
                k=budget//(16*nw);floor=key=='bounded_points'
                expected=P*max(1,k//P+(not floor and k%P!=0))
                if (point['floor_requested']!=use['NTT_GIANT_CHUNK_FLOOR'] or budget!=a.giant_point_kb*1024 or
                    int(point['points'])!=expected or int(point['coordinate_cap_bytes'])!=expected*16*nw or
                    bool(int(point['minimum_over_budget']))!=(P*16*nw>budget) or
                    point['estimated_chunks']!=done['chunks'] or int(done['points'])!=int(entry['shape']['giant_points']) or
                    int(done['chain_chunks'])+int(done['ladder_chunks'])!=int(done['chunks'])):
                    raise ValueError('point chunk execution differs from the requested budget/shape')
                if a.workspace_fixture and fields(text,'giant_chunk_check')['bad']!='0':
                    raise ValueError('giant point budget fixture failed')
            if a.comparison == 'phase-output':
                entry['phase_trim'] = [fields(line,'stage2_phase_trim') for line in text.splitlines()
                    if line.startswith('stage2_phase_trim:')]
                released = sum(int(v['output_released_bytes']) for v in entry['phase_trim'])
                if (released > 0) != (key == 'trimmed_output'):
                    raise ValueError('phase output release differs from requested arm')
            if a.comparison == 'chunk':
                entry['chunk_plan'] = fields(text,'s4_chunk_plan')
                entry['reduce_hook_calls'] = sum(int(fields(line,'s4_reduce_stats')['launches'])
                    for line in text.splitlines() if line.startswith('s4_reduce_stats:'))
                if entry['chunk_plan']['workspace_budget'] != use['NTT_S4_WORKSPACE_BUDGET']:
                    raise ValueError('chunk policy differs from requested arm')
                if a.workspace_fixture and fields(text,'s4_chunk_budget_check')['bad'] != '0':
                    raise ValueError('chunk formula fixture failed')
            entry['root'] = fields(text, 'scaled_root_device')
            entry['frontier'] = fields(text, 'scaled_frontier_device')
            # Retain a completed arithmetic run even if a later residency gate
            # rejects the matrix. complete=False/error preserve the failure.
            data['runs'].append(entry)
            del data['pending']
            persist()
            if a.require_resident:
                for prefix in ('real_batched_folddevice', 'scaled_root_device', 'scaled_frontier_device'):
                    if fields(text, prefix)['enabled'] != '1':
                        raise ValueError('required residency fell back: ' + prefix)
            if oracle and oracle['unit'] and a.mode == 'check':
                if entry['leaf'] != {k: str(v) for k, v in oracle['expected_leaf'].items()}:
                    raise ValueError('independent target-ring monic leaf oracle mismatch')
            print(name, entry['wall']['total'], 's', flush=True)
        targets = {(r['result']['N_hex'], r['result']['B1'], r['result']['sigma']) for r in data['runs']}
        factors = {tuple(sorted(r['result']['factors'])) for r in data['runs']}
        if len(targets) != 1 or len(factors) != 1:
            raise ValueError('target input or factor output changed between arms')
        if a.comparison == 'workspace-bq':
            for field in ('launches', 'poly_muls', 'coeffs_reduced', 'gmp_selftest_cases', 'gmp_checked', 'full_checks'):
                if len({r['coverage'][field] for r in data['runs']}) != 1:
                    raise ValueError('arithmetic coverage changed between layouts: ' + field)
        if a.comparison in ('plan','chunk','phase-output','owner-cache','giant-chunk'):
            for key in keys:
                rows = [r for r in data['runs'] if r['key'] == key]
                for field in ('launches', 'poly_muls', 'coeffs_reduced', 'gmp_selftest_cases', 'gmp_checked', 'full_checks'):
                    if len({r['coverage'][field] for r in rows}) != 1:
                        raise ValueError('arithmetic coverage changed within fixed D: ' + key + '/' + field)
        if a.comparison == 'chunk':
            for field in ('poly_muls','coeffs_reduced','gmp_selftest_cases'):
                if len({r['coverage'][field] for r in data['runs']}) != 1:
                    raise ValueError('mathematical work changed between chunk policies: '+field)
            for key in keys:
                if len({r['reduce_hook_calls'] for r in data['runs'] if r['key']==key}) != 1:
                    raise ValueError('reduction hook coverage changed within arm: '+key)
        if a.mode == 'check' and (not oracle or oracle['unit']) and len({json.dumps(r['leaf'], sort_keys=True) for r in data['runs']}) != 1:
            raise ValueError('complete target-projected leaf fingerprint mismatch')
        if oracle and not oracle['unit']:
            data['leaf_comparison'] = 'Nonunit fallback X values depend on projective scale; verify target factors instead of equating monic leaf vectors.'
        if a.mode == 'timing':
            timed = [r for r in data['runs'] if r['category'] == 'timing']
            means = {k: statistics.mean(float(r['wall']['total']) for r in timed if r['key'] == k) for k in keys}
            groups = []
            for lo in (0, 4):
                m = {k: statistics.mean(float(r['wall']['total']) for r in timed[lo:lo+4] if r['key'] == k) for k in means}
                groups.append(100 * (1 - m[keys[1]] / m[keys[0]]))
            data['summary'] = dict(mean_seconds=means, reduction_percent=100*(1-means[keys[1]]/means[keys[0]]),
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
