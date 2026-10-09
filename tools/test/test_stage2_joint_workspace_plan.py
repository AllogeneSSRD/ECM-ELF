"""Validate real plan-only NTT/S4 joint output; does not execute ECM curves."""
import argparse
import json
import os
from pathlib import Path
import subprocess

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True)
    p.add_argument('--save',type=Path,required=True)
    p.add_argument('--ini',type=Path,required=True)
    p.add_argument('--device',type=int,required=True)
    p.add_argument('--b2',type=int,default=2600000000000)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    rows=[]
    for name,d,b2,cap,env in [
        ('small',2310,a.b2,6300,{}),('legacy',690690,a.b2,6300,{}),
        ('large',1141140,a.b2,6300,{}),
        ('trim',1141140,a.b2,6300,{'NTT_PHASE_TRIM_RAW':'1','NTT_PHASE_TRIM_OUTPUT':'1'}),
        ('bq',1141140,a.b2,6300,{'NTT_WORKSPACE_REUSE_BQ':'1'}),
        ('keyed',2310,a.b2,6300,{'NTT_ARENA_WORKSPACE_POOL':'0'}),
        ('refusal',1141140,a.b2,32,{}),('single',2310,100,6300,{})]:
        command=[str(a.exe.resolve()),'--ini',str(a.ini.resolve()),'--save',str(a.save.resolve()),
                 '--device',str(a.device),'--b2',str(b2),'--d',str(d),'--curves','1',
                 '--arena-mb',str(cap),'--batch-mb','256','--owner-budget-mb','640','--plan-only']
        proc=subprocess.run(command,env=dict(os.environ,**env),capture_output=True,text=True,
                            encoding='utf-8',errors='replace',timeout=90)
        (out/(name+'.log')).write_text(proc.stdout+proc.stderr,encoding='utf-8')
        if proc.returncode:raise RuntimeError(proc.stderr+proc.stdout)
        plan=[json.loads(s) for s in proc.stdout.splitlines() if s.startswith('{')][-1]
        joint=plan['workspace_memory']
        assert not joint['admission_model'] and not joint['process_peak_complete']
        assert joint['ntt_at_peak']+joint['s4_at_peak']==joint['peak_bytes']
        assert not plan['curves_executed']
        if name=='single':
            assert not joint['valid'] and not joint['finished']
        elif name=='refusal':
            assert joint['valid'] and not joint['finished']
            assert joint['reason'].endswith('cap_refusal')
        else:
            assert joint['valid'] and joint['finished'] and not joint['released_bytes']
            assert joint['ntt_peak_bytes']==plan['ntt_memory']['peak_bytes']
            assert joint['s4_peak_bytes']==plan['s4_memory']['peak_bytes']
            assert joint['peak_bytes']<=joint['ntt_peak_bytes']+joint['s4_peak_bytes']
        rows.append(dict(name=name,command=command,environment=env,plan=plan))
    (out/'results.json').write_text(json.dumps(dict(complete=True,cases=rows),indent=2)+'\n',encoding='utf-8')
    print('PASS: 8 real native workspace plans, joint peak, policies, refusal, unsupported G1')

if __name__=='__main__':main()
