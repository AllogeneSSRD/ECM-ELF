"""Recover historical ECM moduli from Prime95 result JSON and compare GPU timings.

No Prime95/GPU jobs are launched. Exact integer equality, not bit length or the
Mersenne exponent, determines whether a CPU/GPU timing pair is admitted.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import statistics

from analyze_prime95_ecm import csv_file, snapshot


def modulus(exponent, factors):
    n = (1 << exponent) - 1
    for factor in factors:
        f = int(factor)
        if f <= 1 or n % f:
            raise ValueError(f'Invalid known factor {f} for M{exponent}')
        n //= f
    return n


def measure(values):
    values = list(values)
    return dict(n=len(values), mean=statistics.mean(values), median=statistics.median(values),
                min=min(values), max=max(values),
                stdev=statistics.stdev(values) if len(values) > 1 else None)


def recover(cpu, records, log_lines):
    """Anchor each task using sigma-matched F results, propagate verified factors.

    Prime95 writes known-factors BEFORE appending the newly found factor. See
    ecm.cpp JSONaddExponentKnownFactors vs addNewKnownFactor in its bingo path.
    Contiguous worker/exponent/B2-tier spans are treated as tasks; a completed
    task forces a boundary even when the following task has the same settings.
    """
    factor_index = defaultdict(list)
    for record in records:
        if record.get('worktype') == 'ECM' and record.get('status') == 'F':
            sigma = record.get('sigma', record.get('Edwards', {}).get('sigma'))
            factor_index[(record.get('exponent'), str(sigma), record.get('b1'))].append(record)
    by_worker = defaultdict(list)
    for run in cpu['runs']:
        by_worker[run['worker']].append(run)
    tasks, enriched, warnings, offsets = [], [], [], []
    for worker, runs in by_worker.items():
        spans = []
        for run in runs:
            key = (run['exponent'], run['b2_target_bucket'])
            if not spans or key != spans[-1]['key'] or spans[-1]['runs'][-1].get('completed_task'):
                spans.append(dict(key=key, runs=[]))
            spans[-1]['runs'].append(run)
        for task_index, span in enumerate(spans):
            members = span['runs']
            anchors = {}
            for run in members:
                if not run['factors']:
                    continue
                candidates = factor_index[(run['exponent'], run['s'], run['b1'])]
                wanted = sorted(f['value'] for f in run['factors'])
                candidates = [r for r in candidates if sorted(r.get('factors', [])) == wanted]
                if len(candidates) != 1:
                    warnings.append(f"{run['run_id']}: factor JSON match count {len(candidates)}")
                    continue
                anchors[run['run_id']] = candidates[0]
                delta = (datetime.fromisoformat(run['factors'][0]['timestamp']) -
                         datetime.fromisoformat(candidates[0]['timestamp'])).total_seconds()
                offsets.append(delta)
            known = None
            if anchors:
                # No earlier factor event can be skipped when propagating backward.
                first = next(r for r in members if r['run_id'] in anchors)
                prefix = members[:members.index(first)]
                if not any(r['factors'] for r in prefix):
                    known = list(anchors[first['run_id']].get('known-factors', []))
            task_id = f'w{worker}:task{task_index + 1}'
            for run in members:
                anchor = anchors.get(run['run_id'])
                evidence = 'propagated_sigma_matched_factor_JSON'
                if anchor is not None:
                    anchored = list(anchor.get('known-factors', []))
                    if known is not None and modulus(run['exponent'], known) != modulus(run['exponent'], anchored):
                        raise ValueError(f"{run['run_id']}: propagated N disagrees with factor JSON")
                    known = anchored
                    evidence = 'direct_sigma_matched_factor_JSON'
                n = modulus(run['exponent'], known) if known is not None else None
                row = dict(run)
                row.update(task_id=task_id, known_factors_before_curve=list(known) if known is not None else None,
                           actual_input_bits=n.bit_length() if n is not None else None,
                           N_hex=format(n, 'x') if n is not None else None,
                           N_sha256=hashlib.sha256(n.to_bytes((n.bit_length()+7)//8, 'big')).hexdigest() if n is not None else None,
                           modulus_evidence=evidence if n is not None else 'unknown',
                           factor_result_json_line=anchor['_line'] if anchor is not None else None)
                enriched.append(row)
                if run['factors']:
                    if anchor is None or known is None:
                        known = None
                    else:
                        additions = [f['value'] for f in run['factors']]
                        modulus(run['exponent'], known + additions)  # Exact divisibility.
                        known += additions
            tasks.append(dict(task_id=task_id, exponent=span['key'][0], B2=span['key'][1],
                              start_line=members[0]['start_line'], end_line=members[-1]['end_line'],
                              attempts=len(members), factor_anchors=len(anchors),
                              final_known_factors=known, final_N_hex=format(modulus(span['key'][0], known), 'x') if known is not None else None,
                              completed_task=members[-1].get('completed_task')))
    # Independently cross-check the NF completion records by checksum AND timestamp.
    # screen.log and result JSON use different clocks here; infer their offset from
    # sigma matches rather than assuming a timezone from the machine clock.
    offset = Counter(offsets).most_common(1)[0][0] if offsets else None
    for task in tasks:
        completion = task['completed_task']
        if not completion:
            continue
        line = log_lines[completion['line'] - 1]
        code = re.search(r'Wi\d+: ([0-9A-F]+)', line)
        ts = re.search(r'\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)\]', line)
        candidates = [r for r in records if r.get('worktype') == 'ECM' and r.get('status') == 'NF'
                      and r.get('exponent') == task['exponent'] and r.get('b1') == completion['b1']
                      and r.get('b2') == completion['reported_average_b2'] and r.get('curves') == completion['curves']
                      and code and r.get('security-code') == code.group(1)]
        if offset is not None and ts:
            candidates = [r for r in candidates if abs((datetime.fromisoformat(ts.group(1)) -
                          datetime.fromisoformat(r['timestamp'])).total_seconds() - offset) <= 2]
        if len(candidates) == 1:
            record = candidates[0]
            final_known = list(record.get('known-factors', []))
            final_n = modulus(task['exponent'], final_known)
            members = [r for r in enriched if r['task_id'] == task['task_id']]
            if task['final_N_hex'] is None and not any(r['factors'] for r in members):
                # With no factor events, the NF input applies to every curve in
                # this task. This also handles a run containing only NF results.
                task['final_known_factors'] = final_known
                task['final_N_hex'] = format(final_n, 'x')
                for run in members:
                    run.update(known_factors_before_curve=final_known,
                               actual_input_bits=final_n.bit_length(), N_hex=format(final_n, 'x'),
                               N_sha256=hashlib.sha256(final_n.to_bytes((final_n.bit_length()+7)//8, 'big')).hexdigest(),
                               modulus_evidence='completed_NF_JSON_no_factor_events',
                               NF_result_json_line=record['_line'])
            if task['final_N_hex'] is None:
                warnings.append(f"{task['task_id']}: NF found, but unmatched factor events prevent N reconstruction")
                continue
            if task['final_N_hex'] != format(final_n, 'x'):
                raise ValueError(f"{task['task_id']}: NF final N mismatch")
            task['NF_json_line'] = record['_line']
            task['NF_modulus_verified'] = True
        else:
            warnings.append(f"{task['task_id']}: NF match count {len(candidates)}")
    return enriched, tasks, warnings, dict(screen_minus_result_seconds=offset, observed_offsets=dict(Counter(offsets)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cpu-analysis', type=Path, required=True)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--gpu-analysis', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output = args.output.resolve()
    for source in (args.cpu_analysis, args.results, args.gpu_analysis):
        source = source.resolve()
        if source.parent == args.output or source.parent in args.output.parents:
            p.error('Use a separate output directory outside every input directory')
    args.output.mkdir(parents=True, exist_ok=True)
    cpu = json.loads(args.cpu_analysis.read_text(encoding='utf-8'))
    gpu = json.loads(args.gpu_analysis.read_text(encoding='utf-8'))
    if not gpu.get('complete'):
        p.error('GPU analysis is a partial snapshot; use the completed study analysis')
    text, raw, result_source = snapshot(args.results.resolve(), 'utf-8-sig')
    records = []
    for line, value in enumerate(text.splitlines(), 1):
        if not value.strip():
            continue
        row = json.loads(value)
        row['_line'] = line
        records.append(row)
    log_lines = args.cpu_analysis.with_name('screen.snapshot.log').read_text(encoding='utf-8-sig').splitlines()
    runs, tasks, warnings, clocks = recover(cpu, records, log_lines)
    group_members = defaultdict(list)
    for run in runs:
        if run['benchmark_eligible'] and run['N_hex'] and run['b2_target_bucket']:
            group_members[(run['exponent'], run['N_hex'], run['b2_target_bucket'])].append(run)
    gpu_lookup = {}
    for row in gpu['summary']:
        info = next(i for i in gpu['inputs'] if i['exponent'] == row['exponent'] and i['variant'] == row['variant'])
        gpu_lookup[(row['exponent'], format(int(info['N_hex'], 16), 'x'), row['B2'])] = row
    groups, pairs = [], []
    for key, members in sorted(group_members.items(), key=lambda item:(item[0][0], item[0][2], -int(item[0][1], 16))):
        total = measure(r['s2_total_time'] for r in members)
        row = dict(exponent=key[0], bits=members[0]['actual_input_bits'], B2=key[2], N_hex=key[1],
                   samples=len(members), cpu=total, start_line=min(r['start_line'] for r in members),
                   end_line=max(r['end_line'] for r in members), run_ids=[r['run_id'] for r in members],
                   actual_B2_min=min(r['b2'] for r in members), actual_B2_max=max(r['b2'] for r in members),
                   CPU_B1=sorted({r['b1'] for r in members}),
                   CPU_D=sorted({r['D'] for r in members}), CPU_degree=sorted({r['poly_degree'] for r in members}),
                   CPU_fft=sorted({f"{r['s2_fft_type']}:{r['s2_fft']}" for r in members}),
                   CPU_memory_MB=sorted({r['using_mem'] for r in members}),
                   known_factors_before_curve=members[0]['known_factors_before_curve'],
                   phase_seconds={name:statistics.mean(r[field] for r in members) for name,field in
                                  [('init','s2_init_time'),('main','s2_time'),('gcd','s2_gcd_time')]},
                   detail_seconds={name:statistics.mean(sum(e['seconds'] for e in r['phase_events'] if e['phase']==name)
                                  for r in members) for name in ('nQx complete','Ftree build level','PolyR built','PolyG built','PolyH built','H(X) scaled','PolyF up','PolyF down','gg = mul H(X)')})
        match = gpu_lookup.get(key)
        row['gpu_exact_match'] = match is not None
        row['gpu_variant'] = match['variant'] if match else None
        groups.append(row)
        if match:
            pair = dict(row, gpu=match, gpu_speedup=total['mean']/match['mean_seconds'],
                        gpu_minus_cpu_seconds=match['mean_seconds']-total['mean'],
                        cpu_B2_overshoot_percent=(statistics.mean(r['b2'] for r in members)/key[2]-1)*100)
            pair['phase_percent']={name:100*seconds/total['mean'] for name,seconds in row['phase_seconds'].items()}
            pairs.append(pair)
    source_meta = {}
    for label,path in [('cpu_analysis',args.cpu_analysis),('gpu_analysis',args.gpu_analysis)]:
        source_meta[label] = dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    result = dict(schema_version=1,generated_utc=datetime.now(timezone.utc).isoformat(),
                  sources=dict(**source_meta,results=result_source,screen=cpu['source_log']), clocks=clocks,
                  cpu_counts=cpu['counts'], GPU_device=gpu['device'], GPU_memory=gpu['memory'],
                  counts=dict(tasks=len(tasks), recovered_moduli=sum(r['N_hex'] is not None for r in runs),
                              eligible_CPU_curves=sum(r['benchmark_eligible'] for r in runs),
                              CPU_groups=len(groups), exact_pairs=len(pairs),
                              exact_cofactor_pairs=sum(r['gpu_variant']=='cofactor' for r in pairs)),
                  semantics=dict(cpu_seconds='S2 init + S2 main + GCD; excludes S1/planning/PRP',
                                 gpu_seconds='stage2_full_wall.total; excludes S1/automatic D scan/process startup',
                                 speedup='CPU seconds / GPU seconds; >1 means GPU faster',
                                 match='exact N integer and nominal B2 tier; B1/sigma/actual B2/hardware differ',
                                 gpu_power='55 W baseline; no extrapolation from repaired 90 W point'),
                  warnings=warnings, tasks=tasks, runs=runs, groups=groups, exact_pairs=pairs,
                  GPU_summary=gpu['summary'])
    (args.output/'results.snapshot.jsonl').write_bytes(raw)
    (args.output/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    csv_file(args.output/'modulus_runs.csv',[{k:v for k,v in r.items() if k not in ('phase_events','fft_candidates')} for r in runs])
    csv_file(args.output/'cpu_groups.csv',groups)
    csv_file(args.output/'exact_pairs.csv',[dict(exponent=r['exponent'],bits=r['bits'],B2=r['B2'],
             CPU_n=r['samples'],CPU_seconds=r['cpu']['mean'],CPU_std=r['cpu']['stdev'],
             GPU_n=r['gpu']['samples'],GPU_seconds=r['gpu']['mean_seconds'],GPU_std=r['gpu']['std_seconds'],
             GPU_speedup=r['gpu_speedup'],GPU_variant=r['gpu_variant'],
             CPU_actual_B2_min=r['actual_B2_min'],CPU_actual_B2_max=r['actual_B2_max'],
             CPU_B2_overshoot_percent=r['cpu_B2_overshoot_percent'],CPU_D=r['CPU_D'],GPU_D=r['gpu']['D'],
             CPU_degree=r['CPU_degree'],GPU_P=r['gpu']['P'],CPU_fft=r['CPU_fft'],CPU_memory_MB=r['CPU_memory_MB'],
             CPU_B1=r['CPU_B1'],GPU_B1=r['gpu']['B1'],CPU_init=r['phase_seconds']['init'],
             CPU_main=r['phase_seconds']['main'],CPU_GCD=r['phase_seconds']['gcd'],
             GPU_init=r['gpu']['init_seconds'],GPU_main=r['gpu']['main_seconds'],
             start_line=r['start_line'],end_line=r['end_line'],N_sha256=next(x['N_sha256'] for x in runs if x['run_id']==r['run_ids'][0])) for r in pairs])
    print(json.dumps(dict(output=str(args.output),counts=result['counts'],clocks=clocks,warnings=warnings),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
