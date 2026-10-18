"""Audit immutable original-path timings, exact leaves, coverage and NTT tables."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
if hasattr(sys,'set_int_max_str_digits'):sys.set_int_max_str_digits(0)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fields(t,n):
    rows=re.findall(r'^'+re.escape(n)+r': (.*)$',t,re.M)
    if not rows:raise ValueError('Missing '+n)
    return dict(re.findall(r'(\w+)=(\S+)',rows[-1]))
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--partial',action='store_true')
    a=parser.parse_args();folder=a.study.resolve();p=read(folder/'plan.json')
    if not p['complete'] and not a.partial:raise ValueError('Incomplete study')
    if a.output.exists():raise ValueError('Use a fresh audit path')
    collector=folder/'collector_snapshot.py'
    if sha(collector)!=p['collector_sha256'] or sha(p['ini'])!=p['ini_sha256']:raise ValueError('Collector/INI changed')
    for key in ('gates','reference'):
        if sha(p[key]['path'])!=p[key]['sha256']:raise ValueError(key+' changed')
    for name,h in p['helpers'].items():
        if sha(folder/'helpers'/name)!=h:raise ValueError('Frozen helper changed')
    binary_versions={}
    for arm,b in p['binaries'].items():
        exe=Path(b['path'])
        if sha(exe)!=b['sha256']:raise ValueError('Binary changed')
        for n,h in b['runtime_files'].items():
            if sha(exe.parent/n)!=h:raise ValueError('Runtime changed')
        for n,h in b['identity']['sources'].items():
            if sha(exe.parent/'sources'/n)!=h:raise ValueError('Frozen source changed')
        header=(exe.parent/'sources/src/core/ecm_program_version.h').read_text()
        binary_versions[arm]=re.search(r'version\[\]\s*=\s*"([^"]+)"',header)[1]
    expected=set();cells={c['id']:c for c in p['cells']}
    for c in p['cells']:
        if sha(c['save'])!=c['save_sha256']:raise ValueError('Input changed')
        d=c['d'];phi=d;q=2;left=d
        while q*q<=left:
            if left%q==0:
                phi=phi//q*(q-1)
                while left%q==0:left//=q
            q+=1
        if left>1:phi=phi//left*(left-1)
        count=c['b2']//d+2;g=(count+c['p']-1)//c['p']
        if (c['p'],c['points'],c['g'],c['rounded_b2'])!=(phi//2,count,g,max(c['b2'],(g*c['p']-2)*d)):raise ValueError('Boundary geometry differs')
        for repeat in range(p['repeats']+1):
            arms=['baseline','candidate']+(['rounded'] if p['rounding'] and c['id'] in ('m10267_low_normal','m10267_high_normal') else [])
            for arm in arms:expected.add(f'{c["id"]}_{"warmup" if repeat==0 else "timing"}{repeat}_{arm}')
    actual=set();groups={};checked=0;transforms=0;versions={};ntt_tables={}
    for r in p['runs']:
        if r['id'] in actual or r['id'] not in expected:raise ValueError('Duplicate/undeclared invocation')
        if r['warmup']!=(r['repeat']==0) or r['id']!=f'{r["cell"]}_{"warmup" if r["warmup"] else "timing"}{r["repeat"]}_{r["arm"]}':raise ValueError('Warmup/repeat declaration differs')
        actual.add(r['id']);out=Path(r['raw']);c=cells[r['cell']]
        if {str(f.relative_to(out)):sha(f) for f in out.rglob('*') if f.is_file()}!=r['files']:raise ValueError('Raw evidence changed: '+r['id'])
        cmd=read(out/'command.json');argv=cmd['command']
        binary=p['binaries']['baseline' if r['arm']=='baseline' else 'candidate']
        if argv[0]!=binary['path'] or argv[argv.index('--save')+1]!=c['save']:raise ValueError('Binary/input command differs')
        for flag,value in [('--b2',r['b2']),('--d',c['d']),('--relocation',0),('--carrier-exponent',0),('--device',p['device']['index']),('--arena-mb',6300),('--owner-budget-mb',640),('--batch-mb',256),('--short-calibration',0),('--tune-runtime',0),('--ntt-counts',1)]:
            if argv[argv.index(flag)+1]!=str(value):raise ValueError('Command differs: '+flag)
        if cmd['environment']!={'CUDA_LAUNCH_BLOCKING':'0'}:raise ValueError('Launch environment differs')
        row=json.loads((out/'internal.jsonl').read_text().splitlines()[-1]);versions[r['arm']]=row['program']['version']
        if row['program']['version']!=binary_versions['baseline' if r['arm']=='baseline' else 'candidate']:raise ValueError('Compiled/result version differs')
        saved_n=re.search(r'\bN=0x([0-9A-Fa-f]+);',Path(c['save']).read_text())[1]
        if int(saved_n,16)!=int(row['N_hex'],16) or int(saved_n,16).bit_length()!=c['bits']:raise ValueError('Target modulus differs')
        text=(out/'engine.log').read_text()+(out/'debug.log').read_text()
        if r['arm']!='baseline':
            layout=fields(text,'groot_layout')
            expected_layout=(c['bits']+63)//64>128 and (r['b2']//c['d']+2)//c['p']>=4
            if bool(int(layout['balanced']))!=expected_layout or layout['policy']!='wide_repeated_v1':raise ValueError('G tree runtime policy differs')
        oracle=fields(text,'s4_multiply_stats');checked+=int(oracle['gmp_checked'])
        if not int(oracle['gmp_checked']) or any(int(oracle[k]) for k in ('gmp_check_bad','gmp_selftest_bad')):raise ValueError('Arithmetic error')
        timer=fields(text,'stage2_full_wall')
        if timer['clean']!='1' or abs(float(timer['total'])-float(timer['init'])-float(timer['main']))>3e-6:raise ValueError('Timer boundary differs')
        if {k:float(timer[k]) for k in ('shape','init','main','total')}!=r['timing']:raise ValueError('Recorded timer differs')
        if row['curve']!='ed' or row['param'] is not None or row['relocation'] or row['relocation_replayed'] or row['bad_factors']:raise ValueError('Result scope differs')
        if (row['B1'],row['sigma'],row['d'],row['poly-size'])!=(c['b1'],c['sigma'],c['d'],c['p']):raise ValueError('Curve/geometry differs')
        b2=c['rounded_b2'] if r['arm']=='rounded' else c['b2']
        if row['B2']!=r['b2'] or r['b2']!=b2 or row['giant_first']!=1 or row['giant_points']!=b2//c['d']+2:raise ValueError('Coverage mismatch')
        for f in row['factors']:
            if not 1<int(f)<int(row['N_hex'],16) or int(row['N_hex'],16)%int(f):raise ValueError('Invalid factor')
        if c['kind']=='NF' and row['factors']:raise ValueError('NF mismatch')
        if c['kind']=='F' and not any(int(f)%int(c['known_factor'])==0 for f in row['factors']):raise ValueError('Known factor absent')
        leaf=fields(text,'descent_values');key=(c['id'],b2)
        identity=(leaf,sorted(row['factors']))
        if key in groups and groups[key]!=identity:raise ValueError('Cross-build complete leaf/factor mismatch')
        groups[key]=identity
        if r['leaf']!=leaf or r['factors']!=sorted(row['factors']):raise ValueError('Recorded leaf/factor differs')
        coverage=fields(text,'real_batched_projective')
        if coverage!=r['coverage'] or coverage['leaves']!=coverage['gamma_points']:raise ValueError('Gamma coverage differs')
        table=[list(map(int,v)) for v in re.findall(r'^ntt_count_row:\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*$',(out/'engine.log').read_text(),re.M)]
        if not table or len({v[0] for v in table})!=len(table):raise ValueError('NTT table absent/duplicate')
        for L,k,conv,fwd,inv,total in table:
            if L!=1<<k or fwd!=2*conv or inv!=conv or total!=3*conv:raise ValueError('NTT totals differ')
            transforms+=total
        key=(c['id'],r['arm'])
        if key in ntt_tables and ntt_tables[key]!=table:raise ValueError('Repeated work differs')
        ntt_tables[key]=table
    for c in p['cells']:
        protected=(c['bits']+63)//64<=128 or c['points']//c['p']<4
        keys=[(c['id'],arm) for arm in ('baseline','candidate')]
        if protected and all(k in ntt_tables for k in keys) and ntt_tables[keys[0]]!=ntt_tables[keys[1]]:raise ValueError('Protected layout changed NTT work')
    if not a.partial and actual!=expected:raise ValueError('Missing declared samples')
    result=dict(complete=not a.partial and p['complete'] and actual==expected,study_sha256=sha(folder/'plan.json'),
        source_sha256=sha(__file__),versions=versions,runs=len(actual),formal=sum(not r['warmup'] for r in p['runs']),
        gmp_checked=checked,ntt_transforms=transforms,errors=0)
    if (folder/'context.json').exists():
        context=read(folder/'context.json')
        if p['device']['uuid'] not in context['output']:raise ValueError('Context GPU differs')
        result['context_sha256']=sha(folder/'context.json')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
