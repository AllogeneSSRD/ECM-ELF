"""Compare actual calibration, selection, publication and file merge across planning extraction."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tomllib
from stage2_tune_tables import summaries


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['baseline','exe','save','output']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--device',type=int,default=1);a=p.parse_args()
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    binaries={'before':a.baseline.resolve(),'after':a.exe.resolve()}
    point=next(v for v in a.save.read_text(encoding='utf-8-sig').splitlines() if v.startswith('METHOD='))
    text=re.search(r'\bN=([^;]+);',point)[1].strip()
    n=(1<<127)-1 if text in ['(2^127-1)','2^127-1'] else int(text,0)
    assert n==(1<<127)-1 and 'B1=20;' in point
    env={k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
    snapshots={};checks=[]
    for label,exe in binaries.items():
        work=out/label;work.mkdir();saved=work/'points.save';saved.write_text(point+'\n')
        queue=work/'worktodo.txt';queue.write_text('ECMSTAGE2=1,2,127,-1,"points.save",0,0,1\n')
        ini=work/'ecm.toml';ini.write_text('[shared]\nuser=""\ncomputer=""\n[stage2]\n'
            'stage2_b2=40000\nstage2_tune_warmup=false\nstage2_relocation=false\n',encoding='utf8')
        common=[exe,'--config',ini,'--device',a.device,'--batch-mb','16','--arena-mb','128','--owner-budget-mb','32','--relocation','0']
        def run(name,options,success=True):
            command=list(map(str,common+options));(work/(name+'.command.json')).write_text(json.dumps(command,indent=2)+'\n')
            r=subprocess.run(command,cwd=work,env=env,capture_output=True,text=True,errors='replace',timeout=180)
            log=r.stdout+r.stderr;(work/(name+'.log')).write_text(log,encoding='utf8')
            assert (r.returncode==0)==success,(label,name,r.returncode,log[-4000:])
            return log
        profile=work/'tune.toml'
        # Queue B2=0 exercises the projected configured B2, rather than CLI B2.
        run('calibrate',['--tune','ecm','--tune-worktodo',queue,'--tune-d','210','--tune-repeats','2',
            '--tune-arithmetic','generic','--tune-budget-seconds','30','--tune-file',profile])
        data=tomllib.loads(profile.read_text(encoding='utf-8-sig'))
        assert data['profile']['format']==8
        samples=[v for v in summaries(data) if v.get('source')=='measured']
        assert samples and all(v['b2']==40000 and v['d']==210 and v['repeats']>=2 for v in samples)
        checks.append(label+'_calibration_and_configured_b2')
        log=run('select',['--save',saved,'--b2','40000','--d','210','--plan-only','--tune-profile',profile,'--short-calibration','0'])
        rows=[json.loads(v) for v in log.splitlines() if v.startswith('{')]
        selection=next(v for v in rows if v.get('type')=='tune_selection')
        plan=next(v for v in rows if v.get('type')=='stage2_plan')
        assert selection['D']==plan['D']==210
        assert selection['source']=='measured' and selection['model']=='exact_shape_measurement_v1'
        assert not selection['selected'] and selection['reason']=='default_protected_by_prediction_uncertainty'
        checks.append(label+'_measured_selection')
        internal=work/'internal.jsonl'
        run('curve',['--save',saved,'--b2','40000','--d','210','--curves','1','--tune-profile',profile,'--short-calibration','0',
            '--factor-only','--results',internal,'--submission',work/'results.json.txt','--log',work/'engine.log'])
        result=[json.loads(v) for v in internal.read_text().splitlines()]
        assert len(result)==1 and result[0]['status']=='NF'
        engine=result[0]['stage2-engine']
        assert engine['status']=='stage2_completed' and engine['hits']==0 and engine['bad_factors']==0
        checks.append(label+'_curve_publication')
        merged=work/'merged.toml'
        run('merge',['--tune','ecm','--tune-merge',profile,'--tune-file',merged,'--device','9999'])
        original=tomllib.loads(profile.read_text(encoding='utf-8-sig'));copy=tomllib.loads(merged.read_text(encoding='utf-8-sig'))
        assert original['width']==copy['width'] and original.get('ntt',{})==copy.get('ntt',{})
        checks.append(label+'_merge_without_cuda_device')
        snapshots[label]=dict(sample_shapes=sorted((v['target_bits'],v['arithmetic_bits'],v['carrier_exponent'],v['b1'],v['b2'],v['d']) for v in samples),
                              selection={k:selection[k] for k in ['D','selected','reason','source','model']},
                              status=result[0]['status'],engine_status=engine['status'],hits=engine['hits'],bad_factors=engine['bad_factors'])
    assert snapshots['before']==snapshots['after']
    report=dict(complete=True,checks=checks,semantic_snapshots=snapshots,binaries={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in binaries.items()},
                save_sha256=hashlib.sha256(a.save.read_bytes()).hexdigest(),device=a.device,performance_samples=0)
    (out/'result.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
