"""Bind real S4 NCU captures to checked application output; retain metric units."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

from bench_stage2_n_scaling import fields
from bench_stage2_runtime_matrix import completed_records
from result_format import normalize_result


METRICS={
    'milliseconds':'gpu__time_duration.sum',
    'registers':'launch__registers_per_thread',
    'shared':'launch__shared_mem_per_block_dynamic',
    'occupancy_percent':'sm__warps_active.avg.pct_of_peak_sustained_active',
    'issue_percent':'smsp__issue_active.avg.pct_of_peak_sustained_active',
    'eligible_warps':'smsp__warps_eligible.avg.per_cycle_active',
    'long_scoreboard_per_issue':'smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio',
    'short_scoreboard_per_issue':'smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio',
    'wait_per_issue':'smsp__average_warps_issue_stalled_wait_per_issue_active.ratio',
    'local_read_sectors':'l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum',
    'local_write_sectors':'l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum',
    'register_spilling_instructions':'sass__inst_executed_register_spilling',
    'l1_hit_percent':'l1tex__t_sector_hit_rate.pct',
    'l2_hit_percent':'lts__t_sector_hit_rate.pct',
}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(folder,warp):
    meta=json.loads((folder/'command.json').read_text())
    if not meta['diagnostic'] or (folder/'exit.txt').read_text(encoding='utf-8-sig').strip()!='0':
        raise ValueError('Need a successful diagnostic capture')
    app=completed_records(folder/'result.jsonl')
    if len(app)!=1 or app[0]['status']!='NF' or app[0]['source-build']!=meta['exe_sha256']:
        raise ValueError('Profiled application identity/result changed')
    result=normalize_result(app[0]);text='\n'.join((folder/name).read_text(encoding='utf-8') for name in ('engine.log','debug.log'))
    stats=fields(text,'s4_multiply_stats');clock=fields(text,'stage2_full_wall')
    if int(clock['clean'])!=1 or min(int(stats[k]) for k in ('gmp_selftest_cases','gmp_checked'))<=0 or \
       any(int(stats[k]) for k in ('gmp_check_bad','gmp_selftest_bad')) or result['factors'] or result['bad_factors']:
        raise ValueError('Arithmetic qualification failed')
    rows=list(csv.DictReader((folder/'metrics_full.csv').read_text(encoding='utf-8-sig').splitlines()))
    units=next(r for r in rows if not r['ID'].isdigit());data=[r for r in rows if r['ID'].isdigit()]
    if len(data)!=3:raise ValueError('Expected three corresponding real calls')
    calls=[]
    for i,r in enumerate(data):
        name='s4_warp_reduce_kernel' if warp else 's4_reduce_kernel'
        if name not in r['Kernel Name'] or '256' not in r['Kernel Name'] or r['Device']!='1' or r['Block Size']!='(128, 1, 1)':
            raise ValueError('Unexpected kernel/device/geometry')
        row=dict(call=i,grid=r['Grid Size'],kernel=r['Kernel Name'])
        row.update({label:float(r[key].replace(',','')) for label,key in METRICS.items()})
        row['milliseconds']*={'s':1000,'ms':1,'us':.001,'ns':.000001}[units[METRICS['milliseconds']]]
        calls.append(row)
    return dict(folder=str(folder.resolve()),identity=meta,normalized=result,calls=calls,
                units={k:units[v] for k,v in METRICS.items()},coverage=stats,
                work={k:stats[k] for k in ('launches','poly_muls','coeffs_reduced','gmp_selftest_cases','gmp_checked','full_checks')},
                ntt=[s for s in text.splitlines() if s.startswith(('ntt_length_counts:','ntt_convolution_counts:'))],
                files={n:sha(folder/n) for n in ('command.json','metrics_full.csv','trace.ncu-rep','result.jsonl')})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('baseline','candidate','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();old=capture(a.baseline,False);new=capture(a.candidate,True)
    for key in ('N_hex','sigma','B1','B2','carrier_exponent'):
        if old['normalized'][key]!=new['normalized'][key]:raise ValueError('Changed profiled input: '+key)
    if old['work']!=new['work'] or not old['ntt'] or old['ntt']!=new['ntt']:
        raise ValueError('Changed work or check coverage')
    a.output.mkdir(parents=True,exist_ok=False)
    report=dict(complete=True,diagnostic=True,baseline=old,candidate=new,
                scope='Corresponding NW256 ordinary calls after skip=200, count=3. Replay times are not full Stage2 performance.')
    (a.output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    rows=[dict(backend=label,**r) for label,d in (('scalar',old),('coop32',new)) for r in d['calls']]
    with (a.output/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    shutil.copy2(__file__,a.output/'analyzer_snapshot.py')
    print(json.dumps(dict(complete=True,calls=rows)))


if __name__=='__main__':main()
