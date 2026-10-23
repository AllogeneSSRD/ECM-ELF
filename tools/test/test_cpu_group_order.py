"""Read-only dataset qualification and reproducible CPU/GMP vs GP batch timings."""
import argparse
import collections
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sqlite3
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def parts(values):
    return sorted((int(p), int(e)) for p, e, *_ in values)


def product(values):
    return math.prod(p**e for p, e in parts(values))


def command(row, mode='known', milliseconds=2000, baby=200000):
    known = str(row['group_order']) if mode in ('known', 'supplied') else ''
    factors = ','.join(f'{p}:{e}' for p, e in parts(row['group_factorization'])) if mode == 'supplied' else ''
    return '\t'.join(map(str, ['ANALYZE', row.get('family', 'mont'), row.get('param', 0), row['sigma'],
                              row['prime'], known, factors, milliseconds, baby, 4000000, 24, 4096, 1000]))


class Probe:
    def __init__(self, exe, folder):
        self.input = (folder/'native_requests.tsv').open('w', encoding='utf-8')
        self.output = (folder/'native_responses.jsonl').open('w', encoding='utf-8')
        self.errors = (folder/'native_stderr.log').open('w', encoding='utf-8')
        self.process = subprocess.Popen([str(exe), '--stream'], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.errors, text=True, encoding='utf-8', bufsize=1)

    def ask(self, line):
        self.input.write(line+'\n'); self.input.flush()
        self.process.stdin.write(line+'\n'); self.process.stdin.flush()
        raw = self.process.stdout.readline()
        require(raw, 'Native probe stopped unexpectedly')
        self.output.write(raw); self.output.flush()
        return json.loads(raw)

    def close(self):
        self.process.stdin.close()
        require(self.process.wait(timeout=10) == 0, 'Native probe exit failure')
        self.input.close(); self.output.close(); self.errors.close()


def verify_analysis(value, row, counted=False):
    require(value['status'] == 'complete', f"Analysis failed: {value.get('reason')}")
    require(value['point_minimality_checked'], 'Point minimality not checked')
    require(value['modulus_primality'] in ('proven', 'probable'), 'Missing modulus primality')
    require(value['group_order_computed'] == counted, 'Known multiple mislabelled as computed group')
    require(int(value['group_order']) == (int(row['group_order']) if counted else 0), 'Group order mismatch')
    require(int(value['point_order']) == int(row['point_order']), 'Point order mismatch')
    require(parts(value['multiple_factorization']['factors']) == parts(row['group_factorization']), 'Group factorization mismatch')
    require(parts(value['point_factorization']) == parts(row['point_factorization']), 'Point factorization mismatch')
    require(product(value['point_factorization']) == int(value['point_order']), 'Point product mismatch')
    require(int(row['group_order']) % int(value['point_order']) == 0, 'Point does not divide supplied group')


def gp_model(row):
    p, s, param = row['prime'], row['sigma'], row.get('param', 0)
    if row.get('family') == 'edwards':
        return f'm=ecm_edwards_init({p},{s});E=m[1];P=m[2];'
    prefix = f'p={p};s={s};'
    if param == 0:
        return prefix+'u=Mod(s^2-5,p);v=Mod(4*s,p);x=u^3;A=(3*u+v)*(v-u)^3/(4*x*v)-2;x=x/v^3;B=x*(x*(x+A)+1);E=ellinit([0,B*A,0,B^2,0]);P=[B*x,B^2];'
    if param in (1, 3):
        formula = '4*s^2,p)/Mod(2^64,p)' if param == 1 else '4*s,p)/Mod(2^32,p)'
        return prefix+f'A=Mod({formula}-2;B=4*A+10;E=ellinit([0,B*A,0,B^2,0]);P=[2*B,B^2];'
    return prefix+'W=ellinit([0,Mod(36,p)]);t=ellmul(W,[-3,3],s);x3=(3*t[1]+t[2]+6)/(2*(t[2]-3));A=-(3*x3^4+6*x3^2-1)/(4*x3^3);B=1/(4*A+10);E=ellinit([0,B*A,0,B^2,0]);P=[2*B,B];'


def gp_run(gp, body, output, name, timeout=180):
    path = output/f'{name}.gp'; path.write_text(body, encoding='utf-8')
    start = time.perf_counter()
    run = subprocess.run([str(gp), '-q', '-f', '-s', '128M'], input=body.encode(), capture_output=True, timeout=timeout)
    wall = (time.perf_counter()-start)*1000
    (output/f'{name}.stdout.log').write_bytes(run.stdout)
    (output/f'{name}.stderr.log').write_bytes(run.stderr)
    require(run.returncode == 0 and b'***' not in run.stderr, 'GP oracle failure: '+run.stderr.decode(errors='replace')[-1000:])
    return run.stdout.decode().replace('\r', ''), wall


def performance(values):
    if not values:
        return None
    ordered = sorted(values)
    return dict(samples=len(values), median_ms=statistics.median(values),
                p95_ms=ordered[math.ceil(.95*len(ordered))-1], sum_ms=sum(values), max_ms=max(values))


def main():
    started = datetime.now().astimezone()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--probe', type=Path, default=ROOT/'build_cpu_group_order/group_order_probe.exe')
    ap.add_argument('--db', type=Path, default=ROOT/'tools/ecm_dataset/ecm_stage2_dataset.sqlite')
    ap.add_argument('--reference', type=Path, default=ROOT/'data/experiments/edwards_group_order_20261016/reference_03')
    ap.add_argument('--gp', default=shutil.which('gp.exe') or shutil.which('gp'))
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--factor-budget-ms', type=int, default=250)
    ap.add_argument('--count-cases', type=int, default=64)
    ap.add_argument('--benchmark-repeats', type=int, default=3)
    args = ap.parse_args()
    require(args.factor_budget_ms > 0 and args.count_cases > 0 and args.benchmark_repeats >= 3, 'Invalid test budgets')
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    frozen = out/'bin'; frozen.mkdir()
    for name in ('group_order_probe.exe', 'gmp-10.dll', 'manifest.json'):
        shutil.copy2(args.probe.resolve().parent/name, frozen/name)
    manifest = json.loads((frozen/'manifest.json').read_text(encoding='utf-8-sig'))
    require(sha(frozen/'group_order_probe.exe').upper() == manifest['sha256'], 'Executable identity mismatch')
    require(sha(frozen/'gmp-10.dll').upper() == manifest['runtime_sha256'], 'GMP identity mismatch')
    for source, digest in manifest['sources'].items():
        require(sha(ROOT/source).upper() == digest, 'Sources differ from built executable: '+source)
    shutil.copy2(__file__, out/Path(__file__).name)
    require(args.gp, 'GP reference executable not found')
    gp = Path(shutil.which(args.gp) or args.gp).resolve()
    require(gp.is_file(), 'GP reference executable not found: '+str(gp))
    gp_source = (ROOT/'tools/ecm_dataset/edwards_order.gp').read_text(encoding='utf-8')
    # A frozen read transaction keeps concurrent corpus maintenance out of this run.
    db = sqlite3.connect(args.db.resolve().as_uri()+'?mode=ro', uri=True); db.execute('BEGIN')
    db.row_factory = sqlite3.Row
    snapshot = [dict(row) for row in db.execute('SELECT * FROM factors WHERE group_order IS NOT NULL ORDER BY exponent,length(value),value')]
    db.close(); dump(out/'database_rows.json', snapshot)
    dataset = []
    for i, row in enumerate(snapshot):
        require(row['primality_verified'] == 1, 'Corpus modulus is not marked proven')
        dataset.append(dict(id=f'mont_{i}', exponent=row['exponent'], prime=row['value'], sigma=row['sigma'],
                            group_order=row['group_order'], point_order=row['point_order'],
                            group_factorization=json.loads(row['group_factorization']),
                            point_factorization=json.loads(row['point_factorization'])))
    reference = json.loads((args.reference/'summary.json').read_text(encoding='utf-8'))
    require(reference['complete'] and len(reference['results']) == 17, 'Incomplete Edwards reference')
    edwards = []
    for i, row in enumerate(reference['results']):
        require(sha(args.reference/f'case_{i:02d}'/'stdout.log') == row['stdout_sha256'], 'Changed raw Edwards oracle')
        require(product(row['group_factorization']) == int(row['group_order']), 'Invalid group fixture product')
        edwards.append(dict(row, id=f'edwards_{i}', family='edwards', param=-1))
    dump(out/'edwards_reference.json', reference)
    cpu = platform.processor()
    if os.name == 'nt':
        import winreg
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
            cpu = winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
    summary = dict(date=started.date().isoformat(), started_at=started.isoformat(), cpu=cpu, os=platform.platform(), power='not measured',
                   clocks='not pinned; shared desktop', threads=1, python=platform.python_version(),
                   probe_sha256=sha(frozen/'group_order_probe.exe'), gmp_sha256=sha(frozen/'gmp-10.dll'),
                   gp_sha256=sha(gp), runner_sha256=sha(__file__), database_snapshot_sha256=sha(out/'database_rows.json'),
                   manifest=manifest, cases=len(dataset), edwards_cases=len(edwards), results=[], comparisons=[])
    exe = frozen/'group_order_probe.exe'
    unit = subprocess.run([str(exe), '--self-test'], capture_output=True, check=True, timeout=30)
    (out/'self_tests.log').write_bytes(unit.stdout)
    probe = Probe(exe, out)
    factor_times, point_times = [], []
    by_status = collections.Counter(); assurance = collections.Counter()
    try:
        for i, row in enumerate(dataset+edwards):
            factor_value = probe.ask('\t'.join(map(str, ['FACTOR', row['group_order'], args.factor_budget_ms, 4000000, 1000])))
            require(product(factor_value['factors'])*int(factor_value['cofactor']) == int(row['group_order']), 'Partial factor product mismatch')
            expected = dict(parts(row['group_factorization']))
            require(all(expected.get(p, 0) >= e for p, e in parts(factor_value['factors'])), 'Incorrect partial factor')
            by_status[factor_value['status']] += 1
            if factor_value['status'] == 'complete':
                require(parts(factor_value['factors']) == parts(row['group_factorization']), 'Native integer factor mismatch')
                factor_times.append(factor_value['statistics']['total_ms'])
            else:
                require(factor_value['status'] == 'partial', 'Unexpected factor status')
            # Isolate point-order correctness even when bounded refactorization times out.
            value = probe.ask(command(row, 'supplied'))
            verify_analysis(value, row)
            point_times.append(value['statistics']['total_ms']); assurance[value['modulus_primality']] += 1
            summary['results'].append(dict(id=row['id'], factor_status=factor_value['status'],
                                          factor_all_proven=factor_value['all_proven'],
                                          factor_ms=factor_value['statistics']['total_ms'],
                                          point_ms=value['statistics']['total_ms'],
                                          point_match=True, all_proven=value['all_proven']))
            if i % 50 == 0:
                print(json.dumps(dict(phase='corpus', completed=i+1, total=len(dataset)+len(edwards), factor_status=dict(by_status))), flush=True)
        # Stratify by modulus bit length, retaining unique (p,sigma) identities.
        bands = collections.defaultdict(list)
        for row in dataset:
            bands[(int(row['prime']).bit_length()-1)//8].append(row)
        count_rows = []
        eligible = [k for k in sorted(bands) if k <= 8]
        for k in eligible:
            values = bands[k]
            quota = max(1, args.count_cases//len(eligible))
            for j in range(min(quota, len(values))):
                count_rows.append(values[int(j*(len(values)-1)/max(1,quota-1))])
        seen = set(); count_rows = [r for r in count_rows if not ((r['prime'],r['sigma']) in seen or seen.add((r['prime'],r['sigma'])))]
        counts = collections.Counter(); counted_ok = []
        for row in count_rows+edwards:
            value = probe.ask(command(row, 'count', milliseconds=2000))
            counts[value['status']] += 1
            if value['status'] == 'complete':
                verify_analysis(value, row, counted=True);counted_ok.append(row)
            else:
                require(value['status'] == 'resource_limit', 'Unexpected count failure: '+str(value))
            summary['comparisons'].append(dict(id=row['id'], status=value['status'], reason=value['reason'], statistics=value['statistics']))
        # Independent GP models exercise all parameterizations, both sqrt branches,
        # uint64 sigma, and degeneracy; GP errors are returned as explicit cases.
        oracle_rows = []
        for p in (101, 103, 10007, 1000003, 1000000007, 59649589127497217):
            for family,param in [('mont',0),('mont',1),('mont',2),('mont',3),('edwards',-1)]:
                oracle_rows.append(dict(id=f'oracle_{len(oracle_rows)}', prime=str(p), sigma='26', family=family, param=param))
        for family,param in [('mont',0),('mont',1),('mont',2),('mont',3),('edwards',-1)]:
            oracle_rows.append(dict(id=f'oracle_{len(oracle_rows)}', prime='1000003', sigma=str(2**64-1), family=family, param=param))
        body = gp_source+'\n'
        for i,row in enumerate(oracle_rows):
            body += f'iferr({gp_model(row)};n=ellcard(E);o=ellorder(E,P,[n,factor(n)]);print("O|{i}|",n,"|",o),err,print("INVALID|{i}"));\n'
        body += 'print("DONE");quit();\n'
        text,_ = gp_run(gp, body, out, 'parameter_oracle')
        found = {}
        for line in text.splitlines():
            if line.startswith('O|'):
                _, i, n, o = line.split('|'); found[int(i)] = (n,o)
            elif line.startswith('INVALID|'):
                found[int(line.split('|')[1])] = None
        require(len(found) == len(oracle_rows), 'Missing GP parameter oracle cases')
        for i,row in enumerate(oracle_rows):
            value = probe.ask(command(row, 'count', milliseconds=3000))
            if found[i] is None:
                require(value['status'] == 'invalid_input', 'Degenerate curve accepted')
            else:
                require(value['status'] == 'complete', 'Parameter count incomplete: '+str(value))
                require((value['group_order'],value['point_order']) == found[i], 'Parameter group or point mismatch')
        summary['parameter_oracle_cases'] = len(oracle_rows)
        # Fixed exact inputs, three paired batches, one process per batch for both.
        # All three operations match: construct, count, factor, derive exact point order.
        benchmark = sorted(counted_ok, key=lambda r:int(r['prime']))
        benchmark = [benchmark[int(j*(len(benchmark)-1)/min(23,len(benchmark)-1))] for j in range(min(24,len(benchmark)))] if len(benchmark)>1 else benchmark
        require(benchmark, 'No complete count samples available for performance comparison')
        dump(out/'benchmark_inputs.json', benchmark)
        native_batch = '\n'.join(command(r, 'count', milliseconds=5000) for r in benchmark)+'\n'
        gp_batch = gp_source+'\n'
        for i,row in enumerate(benchmark):
            gp_batch += f'{gp_model(row)};n=ellcard(E);f=factor(n);o=ellorder(E,P,[n,f]);print("B|{i}|",n,"|",o);\n'
        gp_batch += 'quit();\n'
        batches = []
        for repeat in range(args.benchmark_repeats):
            timings = {}
            order = ['native','gp'] if repeat % 2 == 0 else ['gp','native']
            for engine in order:
                if engine == 'native':
                    start = time.perf_counter()
                    run = subprocess.run([str(exe),'--stream'],input=native_batch.encode(),capture_output=True,check=True,timeout=180)
                    timings['native_process_ms'] = (time.perf_counter()-start)*1000
                    (out/f'benchmark_{repeat}_native.stdout.log').write_bytes(run.stdout)
                    (out/f'benchmark_{repeat}_native.stderr.log').write_bytes(run.stderr)
                    results = [json.loads(line) for line in run.stdout.decode().splitlines()]
                    require(len(results) == len(benchmark), 'Native batch length mismatch')
                    for value,row in zip(results,benchmark):verify_analysis(value,row,counted=True)
                    timings['native_module_sum_ms'] = sum(v['statistics']['total_ms'] for v in results)
                else:
                    text,wall = gp_run(gp,gp_batch,out,f'benchmark_{repeat}_gp')
                    values = [line.split('|') for line in text.splitlines() if line.startswith('B|')]
                    require(len(values) == len(benchmark), 'GP batch length mismatch')
                    require(all((v[2],v[3]) == (str(r['group_order']),str(r['point_order'])) for v,r in zip(values,benchmark)), 'GP batch mismatch')
                    timings['gp_process_ms'] = wall
            batches.append(timings)
        operation_benchmarks = {}
        for operation in ('factor', 'point_supplied'):
            if operation == 'factor':
                native_input = '\n'.join('\t'.join(map(str,['FACTOR',r['group_order'],5000,4000000,1000])) for r in benchmark)+'\n'
            else:
                native_input = '\n'.join(command(r,'supplied',milliseconds=5000) for r in benchmark)+'\n'
            gp_input = gp_source+'\n'
            for i,row in enumerate(benchmark):
                matrix = '['+';'.join(f'{p},{e}' for p,e in parts(row['group_factorization']))+']'
                if operation == 'factor':
                    gp_input += f'f=factor({row["group_order"]});print("F|{i}|",f=={matrix});\n'
                else:
                    gp_input += f'{gp_model(row)};n={row["group_order"]};f={matrix};o=ellorder(E,P,[n,f]);print("B|{i}|",n,"|",o);\n'
            gp_input += 'quit();\n'
            samples = []
            for repeat in range(args.benchmark_repeats):
                timings = {}
                for engine in (['native','gp'] if repeat%2 == 0 else ['gp','native']):
                    if engine == 'native':
                        start = time.perf_counter()
                        run = subprocess.run([str(exe),'--stream'],input=native_input.encode(),capture_output=True,check=True,timeout=180)
                        timings['native_process_ms'] = (time.perf_counter()-start)*1000
                        (out/f'benchmark_{operation}_{repeat}_native.stdout.log').write_bytes(run.stdout)
                        (out/f'benchmark_{operation}_{repeat}_native.stderr.log').write_bytes(run.stderr)
                        values = [json.loads(line) for line in run.stdout.decode().splitlines()]
                        require(len(values) == len(benchmark), 'Operation batch length mismatch')
                        for value,row in zip(values,benchmark):
                            if operation == 'factor':
                                require(value['status']=='complete' and parts(value['factors'])==parts(row['group_factorization']), 'Factor benchmark mismatch')
                            else:
                                verify_analysis(value,row)
                        timings['native_module_sum_ms'] = sum(v['statistics']['total_ms'] for v in values)
                    else:
                        text,wall = gp_run(gp,gp_input,out,f'benchmark_{operation}_{repeat}_gp')
                        timings['gp_process_ms'] = wall
                        values = [line.split('|') for line in text.splitlines() if line.startswith('F|' if operation=='factor' else 'B|')]
                        require(len(values)==len(benchmark), 'GP operation batch length mismatch')
                        if operation == 'factor':
                            require(all(v[2]=='1' for v in values), 'GP factor benchmark mismatch')
                        else:
                            require(all((v[2],v[3])==(str(r['group_order']),str(r['point_order'])) for v,r in zip(values,benchmark)), 'GP point benchmark mismatch')
                samples.append(timings)
            operation_benchmarks[operation] = dict(
                cases=len(benchmark),repeats=args.benchmark_repeats,batches=samples,
                native_process=performance([b['native_process_ms'] for b in samples]),
                gp_process=performance([b['gp_process_ms'] for b in samples]),
                native_module_sum=performance([b['native_module_sum_ms'] for b in samples]),
                boundary='process startup and input/output included; factor only, or construct and point order with supplied factorization; supplied input prime status is external, native reclassifies')
        summary.update(factor_status=dict(by_status), modulus_assurance=dict(assurance), count_status=dict(counts),
                       factor_complete_timing=performance(factor_times), point_supplied_timing=performance(point_times),
                       benchmark=dict(cases=len(benchmark), repeats=args.benchmark_repeats, batches=batches,
                                      boundary='one process per batch: startup, input/output, construction, full count, factor, exact point order; sequential single thread',
                                      native_process=performance([b['native_process_ms'] for b in batches]),
                                      gp_process=performance([b['gp_process_ms'] for b in batches]),
                                      native_module_sum=performance([b['native_module_sum_ms'] for b in batches])),
                       operation_benchmarks=operation_benchmarks)
        summary['passed'] = True
        dump(out/'summary.json', summary)
        print(json.dumps({k:summary[k] for k in ('passed','factor_status','modulus_assurance','count_status','parameter_oracle_cases','benchmark')},indent=2),flush=True)
    finally:
        probe.close()


if __name__ == '__main__':
    main()
