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
        ('refusal',1141140,a.b2,32,{}),('single',2310,100,6300,{}),
        ('owner_budget',1141140,a.b2,6300,{}),
        ('frontier_budget',1141140,a.b2,6300,{'NTT_SCALED_FRONTIER_MAX_MB':'0'}),
        ('giant_floor',1141140,a.b2,6300,{'NTT_GIANT_CHUNK_FLOOR':'1'}),
        ('giant_ladder',1141140,a.b2,6300,{'NTT_GIANT_LADDER':'1'}),
        ('giant_nonresident',1141140,a.b2,6300,{'NTT_DEVICE_GLEAF_MAX_MB':'0'}),
        ('giant_ladder_tail',1141140,1141140*(207360+32000-2),6300,{}),
        ('baby_budget',1141140,a.b2,6300,{'NTT_BABY_DEVICE_MAX_MB':'0'}),
        ('initial_diagnostic',1141140,a.b2,6300,{'NTT_XADD6_TEST':'1'})]:
        command=[str(a.exe.resolve()),'--ini',str(a.ini.resolve()),'--save',str(a.save.resolve()),
                 '--device',str(a.device),'--b2',str(b2),'--d',str(d),'--curves','1',
                 '--arena-mb',str(cap),'--batch-mb','256','--owner-budget-mb',
                 '0' if name=='owner_budget' else '640','--plan-only']
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
        resident=plan['resident_workspace_memory']
        assert not resident['admission_model'] and not resident['process_peak_complete']
        assert not resident['headroom_modeled'] and not resident['fallback_modeled']
        assert resident['peak_bytes']==resident['ntt_at_peak']+resident['s4_at_peak']+resident['owner_at_peak']
        if name in ('single','refusal'):
            assert resident['valid']==joint['valid'] and resident['finished']==joint['finished']
        elif name in ('owner_budget','frontier_budget'):
            assert resident['valid'] and not resident['finished']
            assert resident['reason']==('fold_budget_refusal' if name=='owner_budget' else 'frontier_budget_refusal')
        else:
            assert resident['valid'] and resident['finished'] and not resident['released_bytes']
            assert resident['fold_bytes']==plan['owner_bytes']
            assert resident['frontier_bytes']==24*plan['P']
            assert resident['owner_peak_bytes']==resident['fold_bytes']+resident['frontier_bytes']
            assert joint['peak_bytes']<=resident['peak_bytes']<=joint['peak_bytes']+resident['owner_peak_bytes']
        curve=plan['curve_workspace_memory']
        assert not curve['admission_model'] and not curve['process_peak_complete']
        assert curve['version']==2
        assert curve['peak_bytes']==sum(curve[k+'_at_peak'] for k in ('ntt','s4','owner','giant','initial'))
        if name in ('single','refusal','owner_budget','frontier_budget'):
            assert curve['valid']==resident['valid'] and curve['finished']==resident['finished']
            assert curve['reason']==resident['reason']
        elif name=='baby_budget':
            assert curve['valid'] and not curve['finished'] and curve['reason']=='baby_budget_refusal'
            assert not curve['points_consumed'] and not curve['point_chunks']
        elif name=='initial_diagnostic':
            assert not curve['valid'] and not curve['finished'] and curve['reason']=='diagnostic_initial_workspace_not_modeled'
        else:
            gm=plan['giant_memory']
            assert curve['valid'] and curve['finished'] and not curve['released_bytes']
            assert curve['giant_peak_bytes']==gm['peak_bytes']
            assert curve['giant_final_bytes']==gm['accumulation_bytes']
            assert curve['point_chunks']==gm['point_chunks']
            assert curve['points_consumed']==plan['I']
            assert resident['peak_bytes']<=curve['peak_bytes']<=resident['peak_bytes']+gm['peak_bytes']
            assert curve['montgomery_bytes']==8*plan['words']*(3*2048+1)
            assert curve['baby_bytes']==plan['baby_payload_bytes']
            assert curve['initial_peak_bytes']==max(curve['montgomery_bytes'],curve['baby_bytes'])
            assert curve['required_free_bytes']==max(curve['peak_bytes']+curve['reserve_bytes'],
                curve['baby_headroom_bytes'],curve['fold_headroom_bytes'],curve['frontier_headroom_bytes'])
            assert curve['initial_free_snapshot_fits']==(curve['required_free_bytes']<=plan['free_bytes'])
        rows.append(dict(name=name,command=command,environment=env,plan=plan))
    (out/'results.json').write_text(json.dumps(dict(complete=True,cases=rows),indent=2)+'\n',encoding='utf-8')
    print('PASS: 16 real native workspace plans, joint initial/giant/owner lifetimes, budgets, diagnostics, unsupported G1')

if __name__=='__main__':main()
