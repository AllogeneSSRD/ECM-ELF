"""Fixed-D, same-binary Stage2 Graph, outer or pack/zero A/B; retain raw evidence."""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time

from benchmark_stage2_short_ntt import fields, parse, sha as file_sha
from bench_stage2_production import freeze

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return file_sha(Path(path))


def environment(enabled, max_k=19, max_words=1 << 22, min_calls=4, diagnostic=False, variant='graph'):
    env = {k: v for k, v in os.environ.items() if not k.startswith('NTT_')}
    env.update(NTT_SHORT_GRAPH=str(enabled if variant == 'graph' else 0), NTT_SHORT_GRAPH_MAX_K=str(max_k),
               NTT_SHORT_GRAPH_MAX_WORDS=str(max_words), NTT_SHORT_GRAPH_MIN_CALLS=str(min_calls),
               NTT_D_MODEL='0', NTT_NO_PROGRESS='1', CUDA_LAUNCH_BLOCKING='0')
    if variant == 'mid-outer':
        env['NTT_FUSE_MID_COOP'] = str(enabled)
    elif variant == 'joint-outer':
        env.update(NTT_FUSE_MID_COOP='1', NTT_FUSE_JOINT=str(enabled))
    elif variant == 'pack-zero':
        env['NTT_S4_PACK_ZERO'] = str(enabled)
    if diagnostic:
        env['NTT_TARGET_LEAF_HASH'] = '1'
    return env


def command(exe, ini, case, device, log, result, debug=None):
    cmd = [str(exe), '--ini', str(ini), '--save', case['save'], '--b2', str(case['b2']),
           '--d', str(case['d']), '--device', str(device), '--curves', '1',
           '--batch-mb', str(case['batch_mb']), '--arena-mb', str(case.get('arena_mb', 6300)),
           '--owner-budget-mb', str(case.get('fold_mb', 640)),
           '--carrier-exponent', str(case.get('carrier', 0)), '--short-calibration', '0',
           '--short-ntt-batch-tune', '0', '--factor-only', '--log-level', 'phases',
           '--log', str(log), '--results', str(result)]
    if debug:
        cmd += ['--debug-log-file', str(debug)]
    return cmd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', type=Path, required=True)
    p.add_argument('--cases', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', type=int, default=1)
    p.add_argument('--repeats', type=int, default=4)
    p.add_argument('--variant', choices=['graph', 'mid-outer', 'joint-outer', 'pack-zero'], default='graph')
    p.add_argument('--max-k', type=int, default=19)
    p.add_argument('--max-words', type=int, default=1 << 22)
    p.add_argument('--min-calls', type=int, default=4)
    p.add_argument('--diagnostic', action='store_true', help='One off/on all-leaf fingerprint pair; excluded from performance')
    p.add_argument('--timeout', type=int, default=600)
    a = p.parse_args()
    if a.repeats < 1 or not 0 <= a.max_k <= 32 or a.max_words < 1 or a.min_calls < 1 or a.device < 0 or a.timeout < 1:
        p.error('invalid repeat/device/Graph limits/timeout')
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    exe = a.exe.resolve()
    identity = freeze(exe)
    if a.variant != 'graph':
        build = json.loads((exe.parent / 'build_manifest.json').read_text(encoding='utf-8-sig'))
        if (build.get('engine'), build.get('architecture'), build.get('gl_fixed_mode'),
            build.get('add_sub_mask'), build.get('outer_unroll_u')) != ('production', 'sm_89', 3, 1, 0):
            raise ValueError('Middle-outer study requires the supported production sm89/PTX3/canonical build')
    cases = json.loads(a.cases.read_text(encoding='utf-8-sig'))
    if not isinstance(cases, list) or not cases:
        raise ValueError('Nonempty case array required')
    names = set()
    for case in cases:
        if not re.fullmatch(r'[A-Za-z0-9_]+', case.get('name', '')) or case['name'] in names:
            raise ValueError('Unique case names using letters, digits and underscores required')
        names.add(case['name'])
        for key in ('b2', 'd', 'batch_mb'):
            if type(case.get(key)) is not int or case[key] < 1:
                raise ValueError('Positive integer field required: ' + key)
        for key in ('fold_mb', 'arena_mb', 'carrier'):
            if key in case and (type(case[key]) is not int or case[key] < 0):
                raise ValueError('Nonnegative integer field required: ' + key)
        source = Path(case['save']).resolve()
        dest = out / 'inputs' / case['name'] / source.name
        dest.parent.mkdir(parents=True)
        dest.write_bytes(source.read_bytes())
        case['source_save'] = str(source)
        case['save'] = str(dest)
        case['save_sha256'] = sha(dest)
    ini = out / 'manual.ini'
    ini.write_text('verbose=true\nstage2_debug_log=false\nstage2_short_calibration=false\nstage2_tune_runtime=false\n')
    baseline_env = environment(0, a.max_k, a.max_words, a.min_calls, a.diagnostic, a.variant)
    receipt = dict(identity=identity, cases=cases, device=a.device, diagnostic=a.diagnostic,
                   repeats=a.repeats, variant=a.variant, ini_sha256=sha(ini), script_sha256=sha(Path(__file__)),
                   graph_controls={k: v for k, v in baseline_env.items() if k.startswith('NTT_')},
                   argv_vars={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                   timing='engine init+main; no discarded warmups; profiling collected separately')
    (out / 'identity.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    trace_file = (out / 'gpu.csv').open('w', encoding='utf-8')
    telemetry = subprocess.Popen(['nvidia-smi', '-i', str(a.device), '--query-gpu=timestamp,index,uuid,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw,memory.used,utilization.gpu',
                                 '--format=csv', '-lms', '500'], stdout=trace_file, stderr=subprocess.STDOUT,
                                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
    rows = []
    try:
        for case in cases:
            invariant = None
            leaf = None
            repeats = 1 if a.diagnostic else a.repeats
            for turn in range(repeats):
                for enabled in ([0, 1] if turn % 2 == 0 else [1, 0]):
                    if sha(exe) != identity['binary_sha256'] or sha(case['save']) != case['save_sha256']:
                        raise ValueError('Binary or input changed')
                    name = f"{case['name']}_{'on' if enabled else 'off'}_{turn}"
                    log, result = out / (name + '.engine.log'), out / (name + '.results.jsonl')
                    debug = out / (name + '.debug.log') if a.diagnostic else None
                    cmd = command(exe, ini, case, a.device, log, result, debug)
                    env = environment(enabled, a.max_k, a.max_words, a.min_calls, a.diagnostic, a.variant)
                    started = time.time()
                    with (out / (name + '.stdout.log')).open('w', encoding='utf-8') as console:
                        run = subprocess.run(cmd, env=env, cwd=ROOT, stdout=console, stderr=subprocess.STDOUT, timeout=a.timeout,
                                             creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                    ended = time.time()
                    (out / (name + '.command.json')).write_text(json.dumps(dict(command=cmd, environment={k: v for k, v in env.items() if k.startswith('NTT_')},
                        start_unix=started, end_unix=ended, returncode=run.returncode), indent=2), encoding='utf-8')
                    if run.returncode:
                        raise RuntimeError(f'{name} failed ({run.returncode}); evidence retained')
                    row = parse(log, result, leaf_fixture=a.diagnostic)
                    if a.variant != 'graph' and 'RTX 4060 Laptop' not in log.read_text(encoding='utf-8'):
                        raise ValueError('Middle-outer table is not supported on this device')
                    graph = fields(log.read_text(encoding='utf-8'), 'ntt_graph_stats')
                    if int(graph['requested']) != (enabled if a.variant == 'graph' else 0) or int(graph['failures']):
                        raise ValueError('Graph request or capture failed')
                    graph_active = bool(enabled and int(graph['builds']) and int(graph['hits']))
                    selected_lengths = {23, 25} if a.variant == 'joint-outer' else {21, 22, 23}
                    variant_active = graph_active if a.variant == 'graph' else bool(enabled and sum(
                        count for k, count in row['histogram'].items() if int(k) in selected_lengths))
                    if a.variant == 'pack-zero':
                        pack = fields(log.read_text(encoding='utf-8'), 's4_pack_zero_stats')
                        if int(pack['requested'])!=enabled or (not enabled and int(pack['words'])):
                            raise ValueError('Pack/zero request differs')
                        variant_active=bool(enabled and int(pack['words']) and int(pack['avoided_memsets']))
                        row['pack_zero']=pack
                    current = [row['correctness'], row['histogram'], row['convolutions'], row['chunks']]
                    if a.variant=='pack-zero':
                        current.extend(row[key] for key in ('selftests','checked','launches'))
                    if invariant is None:
                        invariant = current
                    if current != invariant:
                        raise ValueError('Arithmetic or workload changed: ' + name)
                    if a.diagnostic:
                        text = log.read_text(encoding='utf-8') + debug.read_text(encoding='utf-8')
                        actual_leaf = fields(text, 'target_descent_values')
                        if leaf is None:
                            leaf = actual_leaf
                        if actual_leaf != leaf:
                            raise ValueError('Complete leaf fingerprint changed: ' + name)
                        row['leaf'] = actual_leaf
                    row.update(case=case['name'], enabled=enabled, turn=turn, graph=graph, graph_active=graph_active,
                               variant=a.variant, variant_active=variant_active, process_seconds=ended-started,
                               start_unix=started, end_unix=ended, diagnostic=a.diagnostic)
                    rows.append(row)
                    with (out / 'runs.jsonl').open('a', encoding='utf-8') as stream:
                        stream.write(json.dumps(row) + '\n')
                    detail = (f"graph_hits={graph['hits']} builds={graph['builds']} saved={graph['saved_launches']}"
                              if a.variant == 'graph' else f"variant={a.variant} selected_transforms={sum(count for k, count in row['histogram'].items() if int(k) in selected_lengths)} enabled={enabled}")
                    if a.variant=='pack-zero':
                        detail=f"variant=pack-zero words={row['pack_zero']['words']} enabled={enabled}"
                    print(f"{name}: total={row['engine_seconds']:.6f}s {detail} PASS", flush=True)
    finally:
        telemetry.terminate()
        telemetry.wait(timeout=10)
        trace_file.close()
    if not any(r['variant_active'] for r in rows):
        raise ValueError('No case executed the selected variant; evidence retained but not an optimization A/B')
    summaries = []
    for case in cases:
        item = dict(case=case['name'], input=case)
        for enabled in (0, 1):
            selected = [r for r in rows if r['case'] == case['name'] and r['enabled'] == enabled]
            part = dict(samples=len(selected))
            for key in ('engine_seconds', 'process_seconds', 'init_seconds', 'main_seconds', 'reduce_seconds', 'full_peak_bytes'):
                values = [r[key] for r in selected]
                part[key] = statistics.median(values)
                part[key + '_mad'] = statistics.median(abs(v-part[key]) for v in values)
            part['phases'] = {k: statistics.median(r['phases'][k] for r in selected) for k in selected[0]['phases']}
            part['graph'] = selected[0]['graph']
            part['histogram'] = selected[0]['histogram']
            item['on' if enabled else 'off'] = part
        item['gain_percent'] = 100 * (1-item['on']['engine_seconds']/item['off']['engine_seconds'])
        summaries.append(item)
    (out / 'summary.json').write_text(json.dumps(dict(complete=True, identity=receipt, summaries=summaries), indent=2), encoding='utf-8')
    with (out / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['case', 'off_seconds', 'on_seconds', 'gain_percent', 'off_mad', 'on_mad', 'graph_hits', 'graph_builds'])
        writer.writeheader()
        for s in summaries:
            writer.writerow(dict(case=s['case'], off_seconds=s['off']['engine_seconds'], on_seconds=s['on']['engine_seconds'], gain_percent=s['gain_percent'],
                                 off_mad=s['off']['engine_seconds_mad'], on_mad=s['on']['engine_seconds_mad'], graph_hits=s['on']['graph']['hits'], graph_builds=s['on']['graph']['builds']))
    print(json.dumps(summaries, indent=2), flush=True)


if __name__ == '__main__':
    main()
