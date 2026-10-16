"""Audit frozen GPU1 Edwards measurements and emit comparable per-scope statistics.

The first collector stopped at the missing M8191 Montgomery tier. Its five
completed scopes are qualified explicitly; the failed scope is replaced only by
the separately rebuilt and completed original-source wide measurement.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import zipfile

UUID = 'GPU-8a67b1f8-ef1c-3177-a822-813a7ac2224d'
MODELS = ['ed-binary', 'mont0', 'mont3']


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    assert len(values) == 3 and all(math.isfinite(x) and x > 0 for x in values)
    middle = statistics.median(values)
    return dict(samples=values, median=middle, min=min(values), max=max(values),
                mad=statistics.median(abs(x-middle) for x in values))


def audit(folder, bits, models, curves, *, partial=False, window=None):
    r = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    assert r['device'] == 1 and r['telemetry_errors'] == []
    assert r['complete'] is (not partial)
    assert sha(folder/'bin/ecm_cuda.exe') == r['binary_sha256']
    for name, digest in r['runtime_sha256'].items(): assert sha(folder/'bin'/name) == digest
    assert sha(folder/'sources.zip') == r['source_archive_sha256']
    assert sha(folder/'tools.zip') == r['tools_archive_sha256']
    with zipfile.ZipFile(folder/'sources.zip') as archive:
        for name, digest in r['source_sha256'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest, name
        tier_path='src/cuda/ecm_cuda_edwards_tiers.inc'
        edwards_tiers=None
        if tier_path in archive.namelist():
            edwards_tiers=[tuple(map(int,pair)) for pair in re.findall(r'ECM_EDWARDS_TIER\((\d+),(\d+)\)',archive.read(tier_path).decode())]
            cache=(folder/'CMakeCache.txt').read_text(encoding='utf-8-sig')
            chosen=re.search(r'^ECM_TIERS:STRING=(.*)$',cache,re.M)
            assert chosen
            if chosen[1].strip():
                allowed={int(x) for x in chosen[1].strip().split(',')}
                edwards_tiers=[tier for tier in edwards_tiers if tier[0] in allowed]
    with zipfile.ZipFile(folder/'tools.zip') as archive:
        assert hashlib.sha256(archive.read('tools/bench/bench_edwards_gpu_stage1.py')).hexdigest() == r['test_sha256']
    expected = {(n,m,i) for n in bits for m in models for i in range(-1,3)}
    actual = {(s['bits'],s['model'],s['repeat']) for s in r['samples']}
    assert len(actual) == len(r['samples']) and actual == expected, (folder, actual ^ expected)
    collector_raw = (folder.parent/(folder.name+'_driver.log')).read_bytes()
    collector_text = collector_raw.decode('utf-16' if collector_raw.startswith(b'\xff\xfe') else 'utf-8-sig')
    collector_text = '\n'.join(collector_text.splitlines())
    recorded_wall = {name: float(seconds) for name,seconds in re.findall(r'^PASS (\S+) ([\d.]+) seconds$', collector_text, re.M)}
    if partial:
        failed = folder/'m8191_mont3_r-1/run.log'
        assert failed.exists() and b'CGBN kernel not found for 9216 bits' in failed.read_bytes()
        assert not (failed.parent/'completed.save').exists()
    groups = {}
    for s in r['samples']:
        assert s['b1'] == 10000 and s['curves'] == curves and s['sigma'] == 20260922
        assert s['warmup'] == (s['repeat'] < 0)
        run = folder/f"m{s['bits']}_{s['model']}_r{s['repeat']}"
        assert abs(recorded_wall[run.name]-s['process_seconds']) <= 0.000001
        assert sha(run/'run.log') == s['log_sha256'] and sha(run/'completed.save') == s['save_sha256']
        raw = (run/'run.log').read_bytes()
        text = raw.decode('utf-8')
        assert b'device=1' in raw and b'execution_target: gpu' in raw
        assert len(s['geometry']) == 1
        geometry = s['geometry'][0]
        assert geometry['tpb'] == 128 and geometry['blocks'] == (curves+128//geometry['tpi']-1)//(128//geometry['tpi'])
        if s['model'].startswith('ed'):
            if edwards_tiers is not None:
                assert edwards_tiers
                expected_tier=next(tier for tier in edwards_tiers if s['bits']<tier[0])
                assert (geometry['container_bits'],geometry['tpi']) == expected_tier
                assert re.findall(r'GPU Edwards geometry: container_bits=(\d+) tpi=(\d+) tpb=(\d+) blocks=(\d+)',text)==[
                    tuple(str(geometry[k]) for k in ('container_bits','tpi','tpb','blocks'))]
            timer = re.findall(r'GPU Edwards arithmetic: algorithm=(\w+) api_seconds=([\d.]+)', text)
            assert [float(x[1]) for x in timer] == ([] if s['arithmetic_api_seconds'] is None else [s['arithmetic_api_seconds']])
            assert bool(s['naf_metrics']) == (s['model'] != 'ed-binary')
            if s['model'] != 'ed-binary':
                assert len(s['actual_edwards_selection']) == 1
                selected = s['actual_edwards_selection'][0]
                assert selected['window'] == window
                assert selected['doubling'] == (5 if s['model'] == 'ed-5' else 4)
                metric = s['naf_metrics'][0]
                printed = re.findall(r'GPU Edwards NAF: columns=(\d+) additions=(\d+) container_bits=(\d+) tpi=(\d+) dictionary_bytes=(\d+) dictionary_gpu_ms=([\d.]+) loop_gpu_ms=([\d.]+) launches=(\d+)', text)
                assert len(printed) == 1
                expected_printed = [metric[k] for k in ['columns','additions','container_bits','tpi','dictionary_bytes','dictionary_gpu_ms','loop_gpu_ms','launches']]
                assert list(map(float, printed[0])) == expected_printed
                assert metric['dictionary_bytes'] == curves*(geometry['container_bits']//8)*4*(1 << (window-2))
                assert metric['launches'] == (metric['columns']+255)//256
                assert (metric['dictionary_gpu_ms']+metric['loop_gpu_ms'])/1000 <= s['arithmetic_api_seconds']+0.001
            if s['arithmetic_api_seconds'] is not None:
                assert 0 < s['arithmetic_api_seconds'] <= s['process_seconds']
        else:
            printed=re.findall(r'CGBN<(\d+),\s*(\d+)> kernel, N is (\d+) bits \((\d+) blocks x (\d+) threads\)',text)
            assert printed==[tuple(str(geometry[k]) for k in ('tpi','container_bits','bits','blocks','tpb'))]
        if not s['warmup']:
            groups.setdefault((s['bits'],s['model']), []).append(s)
    rows = []
    for (n,m), samples in groups.items():
        row = dict(bits=n,model=m,curves=curves,b1=10000,sigma=20260922,
            binary_sha256=r['binary_sha256'],evidence=folder.name,geometry=samples[0]['geometry'][0],
            process_seconds=stats([s['process_seconds'] for s in samples]))
        if samples[0]['arithmetic_api_seconds'] is not None:
            row['arithmetic_api_seconds'] = stats([s['arithmetic_api_seconds'] for s in samples])
        if samples[0]['naf_metrics']:
            row.update(window=samples[0]['actual_edwards_selection'][0]['window'],
                       doubling=samples[0]['actual_edwards_selection'][0]['doubling'])
            row['cuda_event_span_seconds'] = stats([(s['naf_metrics'][0]['dictionary_gpu_ms']+s['naf_metrics'][0]['loop_gpu_ms'])/1000 for s in samples])
        rows.append(row)
    loaded = []
    with (folder/'telemetry.csv').open(encoding='utf-8') as source:
        for line in csv.reader(source):
            assert len(line) == 9 and int(line[1]) == 1 and line[2].strip() == UUID
            if float(line[8]) >= 50: loaded.append(line)
    assert loaded, ('no loaded GPU1 observations', folder)
    telemetry = dict(loaded_samples=len(loaded))
    for name,index in [('power_w',5),('sm_mhz',6),('temperature_c',4)]:
        values = [float(line[index]) for line in loaded if line[index].strip() != '[N/A]']
        telemetry[name] = dict(min=min(values), median=statistics.median(values), max=max(values)) if values else None
    return dict(evidence=folder.name, binary_sha256=r['binary_sha256'], rows=rows,
                collector_complete=r['complete'], qualified_bits=bits, loaded_telemetry=telemetry)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--suite', help='Audit one complete three-family suite instead of the historical study')
    a = p.parse_args(); evidence = a.evidence.absolute(); out = a.output.absolute()
    out.mkdir(parents=True, exist_ok=False)
    if a.suite:
        suite=audit(evidence/a.suite,[347,677,991,2203,4003,8191],['ed','mont0','mont3'],768,window=6)
        report=dict(complete=True,device=1,uuid=UUID,b1=10000,curves=768,formal_samples=54,
                    warmup_samples=18,suites=[suite],comparison=suite['rows'],analyzer_sha256=sha(Path(__file__)),
                    statement='Three curve families in one executable; complete fresh-process wall, not factor-finding efficiency.')
        (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        with (out/'comparison.csv').open('w',newline='',encoding='utf-8') as target:
            writer=csv.writer(target)
            writer.writerow(['bits','model','curves','b1','container_bits','tpi','median_seconds','min_seconds','max_seconds','mad_seconds','exe_sha256','evidence'])
            for row in sorted(suite['rows'],key=lambda r:(r['bits'],r['model'])):
                stat=row['process_seconds'];g=row['geometry']
                writer.writerow([row['bits'],row['model'],row['curves'],row['b1'],g['container_bits'],g['tpi'],
                                 stat['median'],stat['min'],stat['max'],stat['mad'],row['binary_sha256'],row['evidence']])
        print(json.dumps(dict(complete=True,formal_samples=54,binary_sha256=suite['binary_sha256']),indent=2))
        return
    suites = [audit(evidence/'bench_initial_01',[347,677,991,2203,4003],MODELS,768,partial=True),
              audit(evidence/'bench_initial_wide_01',[8191],MODELS,768),
              audit(evidence/'bench_paired_01',[347,677,991,2203,4003,8191],['ed-binary','ed'],768,window=6)]
    suites += [audit(evidence/f'bench_window{w}_01',[347,2203,8191],['ed'],128,window=w) for w in [6,8,10,12]]
    suites += [audit(evidence/'bench_doubling_01',[347,2203,8191],['ed','ed-5'],128,window=8)]
    base = suites[0]['rows']+suites[1]['rows']; pair = suites[2]['rows']
    index = {(row['bits'],row['model']):row for row in pair}
    comparisons = []
    for n in [347,677,991,2203,4003,8191]:
        b, f = index[n,'ed-binary'], index[n,'ed']
        comparisons.append(dict(bits=n,curves=768,
            process_speedup=b['process_seconds']['median']/f['process_seconds']['median'],
            arithmetic_api_speedup=b['arithmetic_api_seconds']['median']/f['arithmetic_api_seconds']['median']))
    report = dict(complete=True,device=1,uuid=UUID,b1=10000,formal_samples=sum(len(s['rows'])*3 for s in suites),
        warmup_samples=sum(len(s['rows']) for s in suites),suites=suites,initial=base,paired_comparisons=comparisons,
        statement='Per-scope fresh-process medians. CUDA event spans are not profiler kernel durations. No production-B1 projection.',
        analyzer_sha256=sha(Path(__file__)))
    (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    with (out/'initial.csv').open('w',newline='',encoding='utf-8') as target:
        writer=csv.writer(target);writer.writerow(['bits','model','curves','median_seconds','min_seconds','max_seconds','container_bits','tpi','exe_sha256','evidence'])
        for row in sorted(base,key=lambda x:(x['bits'],x['model'])):
            s=row['process_seconds'];writer.writerow([row['bits'],row['model'],row['curves'],s['median'],s['min'],s['max'],row['geometry']['container_bits'],row['geometry']['tpi'],row['binary_sha256'],row['evidence']])
    print(json.dumps(dict(complete=True,formal_samples=report['formal_samples'],paired_comparisons=comparisons),indent=2))


if __name__ == '__main__': main()
