"""Recover independent runtime receipts, including slower samples, without CUDA curves."""
import argparse
import importlib.util
import json
import math
import statistics
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"ecm_report"))
from result_format import normalize_result
import subprocess
import sys
import tomllib
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/test'))
sys.path.insert(0,str(ROOT/'tools/bench'))
from stage2_tune_matrix import decode_ids
from stage2_tune_tables import summaries
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('results','exe','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--sample-limit',type=int,default=10)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    spec=importlib.util.spec_from_file_location('importer',ROOT/'tools/bench/import_stage2_benchmark_tune.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    raw=[normalize_result(json.loads(s)) for s in a.results.read_text(encoding='utf-8').splitlines()]
    expected={r['tune_observation']['sample']['runtime_sample_id'] for r in raw if 'tune_observation' in r}
    _,samples,_=module.import_runtime([a.results,a.results],a.sample_limit)
    assert set().union(*(decode_ids(s['measurement_ids']) for s in samples))==expected
    assert sum(s['repeats'] for s in samples)==len(expected) and len(samples)==2
    source=out/'recovered.toml'
    command=[sys.executable,str(ROOT/'tools/bench/import_stage2_benchmark_tune.py'),'--runtime-results',str(a.results.resolve()),
        '--runtime-results',str(a.results.resolve()),'--runtime-samples',str(a.sample_limit),'--output',str(source)]
    r=subprocess.run(command,capture_output=True,text=True,errors='replace',timeout=60)
    (out/'import.log').write_text(r.stdout+r.stderr,encoding='utf-8');assert not r.returncode,r.stderr
    assert sum(s['repeats'] for s in summaries(tomllib.loads(source.read_text(encoding='utf-8'))))==len(expected)
    ini=out/'ecm.toml';ini.write_text('user=\ncomputer=\nverbose=false\n',encoding='utf-8')
    destination=out/'merged.toml'
    command=[str(a.exe.resolve()),'--ini',str(ini),'--device','1','--tune','ecm','--tune-file',str(destination),
        '--tune-merge',str(source),'--tune-merge',str(source),'--tune-runtime-samples',str(a.sample_limit)]
    r=subprocess.run(command,cwd=out,capture_output=True,text=True,errors='replace',timeout=120)
    (out/'merge.log').write_text(r.stdout+r.stderr,encoding='utf-8');assert not r.returncode,r.stdout+r.stderr
    merged=summaries(tomllib.loads(destination.read_text(encoding='utf-8')))
    assert set().union(*(decode_ids(s['measurement_ids']) for s in merged))==expected
    assert sum(s['repeats'] for s in merged)==len(expected) and len(merged)==2
    for s in merged:
        times=[r['tune_observation']['sample']['mean_seconds'] for r in raw if 'tune_observation' in r and
               all(r['tune_observation']['sample'][k]==s[k] for k in ('d','carrier_exponent','b1','b2'))]
        assert len(times)==s['repeats']
        assert math.isclose(s['mean_seconds'],statistics.fmean(times),rel_tol=1e-12,abs_tol=1e-14)
        assert math.isclose(s['stddev_seconds'],statistics.stdev(times),rel_tol=1e-12,abs_tol=1e-14)
    damaged=out/'bad.jsonl';bad=next(r for r in raw if 'tune_observation' in r)
    bad=json.loads(json.dumps(bad));bad['tune_observation']['sample']['bad']=1
    damaged.write_text(json.dumps(bad)+'\n',encoding='utf-8')
    try:module.import_runtime([damaged])
    except ValueError:pass
    else:raise AssertionError('unchecked receipt admitted')
    report=dict(complete=True,independent_samples=len(expected),slower_samples_retained=True,
        duplicate_import_and_merge_idempotent=True,unchecked_receipt_rejected=True,CUDA_curves=0)
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps(report))
if __name__=='__main__':main()
