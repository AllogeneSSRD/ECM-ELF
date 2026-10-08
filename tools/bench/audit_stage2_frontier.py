"""Reconstruct resident descent qualification, native timings and actual transfers."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import statistics
import sys

if hasattr(sys,'set_int_max_str_digits'):sys.set_int_max_str_digits(0)
ROOT=Path(__file__).resolve().parents[2]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))


def need(ok,message):
    if not ok:raise ValueError(message)


def field(text,prefix):
    rows=re.findall(r'^'+re.escape(prefix)+r': (.*)$',text,re.M)
    need(bool(rows),'missing '+prefix)
    return dict(re.findall(r'(\w+)=(\S+)',rows[-1]))


def geometry(p,w):
    pad=1
    while pad<p:pad*=2
    deg=[0]*(2*pad)
    for i in range(p):deg[pad+i]=1
    for i in range(pad-1,0,-1):deg[i]=deg[2*i]+deg[2*i+1]
    base=1;A=B=J=C=pairs=groups=copies=zeros=0
    while base<pad:
        g=collections.Counter()
        for j in range(base):
            for side in [0,1]:
                ci=2*base+2*j+side;a,b=deg[ci],deg[ci^1]
                if not a:zeros+=1;continue
                if not b:copies+=1;C+=a;continue
                g[(a,b)]+=1
        A+=sum(n*(a+b) for (a,b),n in g.items())
        local_B=sum(n*(b+1) for (a,b),n in g.items());need(local_B<=2*p,'F scratch bound')
        B+=local_B;J+=sum(n*a for (a,b),n in g.items());pairs+=sum(g.values());groups+=len(g);base*=2
    return dict(P=p,W=w,pad=pad,levels=pad.bit_length()-1,groups=groups,pairs=pairs,copies=copies,zeros=zeros,
        A_coefficients=A,B_coefficients=B,J_coefficients=J,copy_coefficients=C,metadata_bytes=24*p,f_h2d_bytes=8*w*B,
        metadata_h2d_bytes=24*pairs,leaf_d2h_bytes=8*w*p,avoided_parent_h2d_bytes=8*w*A,avoided_state_d2h_bytes=8*w*J,
        predicted_total_h2d_saved_bytes=8*w*(2*A-B)-24*pairs,predicted_root_and_descent_d2h_saved_bytes=8*w*J,
        predicted_extra_d2d_bytes=8*w*C)


def phi(n):
    result=n;f=2
    while f*f<=n:
        if n%f==0:
            result=result//f*(f-1)
            while n%f==0:n//=f
        f+=1
    return result//n*(n-1) if n>1 else result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True)
    p.add_argument('--profiles',action='store_true');a=p.parse_args();root=a.study.resolve();files={};external={}
    def bind(path):files[path.relative_to(root).as_posix()]=sha(path);return path
    def bind_external(path,digest):need(sha(path)==digest,'external file identity');external[str(Path(path).resolve())]=digest
    native=root/'native_r0';build=read(bind(native/'build_manifest.json'));frozen=read(bind(native/'frozen_sources_manifest.json'))
    need((build['engine'],build['gl_fixed_mode'],build['outer_unroll_u'],build['add_sub_mask'])==('development',3,0,1),'compiled engine/arithmetic')
    binary_sha=sha(bind(native/'ecm_cuda_stage2.exe'));need(binary_sha==build['sha256'].lower()==frozen['binary_sha256'],'compiled binary identity')
    sources={n:h.lower() for n,h in build['source_hashes'].items()};need(sources==frozen['sources'] and len(sources)==30,'full source closure')
    for name,digest in sources.items():need(sha(bind(native/'sources'/name))==digest==sha(ROOT/name),'compiled/current raw source '+name)
    need(len(build['objects'])==5,'five compiled objects')
    for name,digest in build['objects'].items():need(sha(bind(native/'_objects'/(name+'.obj')))==digest.lower(),'compiled object identity')
    raw=(ROOT/'tools/bench/stage2_tree_gpu.cu').read_bytes();edits=read(bind(root/'stage2_source_edit.json'))
    for old,new in reversed(edits['edits']):need(raw.count(new.encode())==1,'unique byte reversal');raw=raw.replace(new.encode(),old.encode())
    need(raw==bind(root/'original_stage2_tree_gpu.cu').read_bytes(),'mixed-newline source preserved')
    resources=read(bind(root/'resources.json'));res=bind(native/'resources.txt')
    need(sha(res)==resources['raw_sha256'] and resources['binary_sha256']==binary_sha,'raw resource export identity')
    actual=dict(re.findall(r'^ (Function [^\r\n]+)\r?\n  ([^\r\n]+)',res.read_text(),re.M))
    need(actual==resources['resources'] and len(actual)==224,'all compiled resources')
    for inv in [0,1]:need('REG:40 STACK:0 SHARED:0 LOCAL:0' in actual[f'Function _Z11tile_kernelILb{inv}ELb1EEvPyPKyyiiS2_yyb:'],'default tile resource')
    need('REG:38 STACK:0 SHARED:0 LOCAL:0' in actual['Function _Z29s4_pack_gather_reverse_kernelPKyS0_yyiiiyyPy:'],'reverse gather resources')
    matrices={};identity=None;models={};formal_reference=None
    for folder,mode,count in [('controls_r1','controls',15),('native_gate_r1','gate',26),('native_timing_r0','timing',10),('native_timing_wide_r0','timing-wide',30)]:
        base=root/folder;report=read(bind(base/'measurements.json'))
        need(report['complete'] and report['mode']==mode and len(report['runs'])==count,'completed matrix '+folder)
        need(report['identity']['binary_sha256']==binary_sha and report['identity']['sources']==sources,'same native compiled identity')
        if identity is None:identity=report['identity']
        need(report['identity']==identity,'one binary across qualification/timing')
        need(sha(bind(base/'collector.py'))==report['tool_sha256'],'actual collector snapshot')
        bind_external(ROOT/'tools/bench/bench_stage2_production.py',report['helper_sha256'])
        reference=ROOT/'build_cuda_cmake/_stage2_sub_native_20261008/native_timing_r0/measurements.json'
        fixtures=ROOT/'build_cuda_cmake/_stage2_wide_20261007/fixtures_r2/fixtures.json'
        bind_external(reference,report['reference_sha256']);bind_external(fixtures,report['fixtures_sha256'])
        old=read(reference);need(old['complete'] and read(fixtures)['complete'],'completed original references')
        expected_large=next(r for r in old['runs'] if r['category']=='timing' and r['mask']==1)
        if mode.startswith('timing'):need(report['gate_sha256']==sha(root/'native_gate_r1/measurements.json'),'timing qualification binding')
        if mode=='controls':
            expected_controls={'control_'+n for n in ['fixtures','pageable','full_window','budget','allocation','owner_off','root_off','poison','unchecked_poison','invalid_flag','reuse0','reuse1','reuse2','budget476','budget477']}
            need({r['category'] for r in report['runs']}==expected_controls,'all distinct declared controls')
        by_case={}
        for row in report['runs']:
            name=row['name'];case=next(c for c in report['cases'] if c['name']==row['case'])
            bind_external(case['save'],case['save_sha256'])
            driver=bind(base/(name+'_driver.log'));need(sha(driver)==row['driver_sha256'],'raw driver output')
            log=bind(base/(name+'.log'));text=log.read_text();combined=text+driver.read_text(errors='replace')
            if mode=='controls':
                control=row['category'][8:];env=row['environment']
                specifications={
                    'fixtures':('NTT_SCALED_FRONTIER_TEST','1',1,None,0),
                    'pageable':('NTT_S4_ASYNC','0',1,None,0),
                    'full_window':('NTT_S4_OUTPUT_WINDOW','0',1,None,0),
                    'budget':('NTT_SCALED_FRONTIER_MAX_MB','0',0,'budget',0),
                    'allocation':('NTT_SCALED_FRONTIER_ALLOC_FAIL','1',0,'allocation_fixture',0),
                    'owner_off':('NTT_FOLD_DEVICE_MAX_MB','0',0,'root_unavailable',0),
                    'root_off':('NTT_SCALED_ROOT_DEVICE','0',0,'root_unavailable',0),
                    'poison':('NTT_SCALED_FRONTIER_TEST_BAD','1',None,None,2),
                    'unchecked_poison':('NTT_SCALED_FRONTIER_TEST_BAD','1',None,None,2),
                    'invalid_flag':('NTT_SCALED_FRONTIER_DEVICE','2',None,None,2),
                    'budget476':('NTT_SCALED_FRONTIER_MAX_MB','476',0,'budget',0),
                    'budget477':('NTT_SCALED_FRONTIER_MAX_MB','477',1,None,0)}
                specifications.update({f'reuse{m}':('NTT_FOLD_OWNER_REUSE',str(m),1,None,0) for m in [0,1,2]})
                key,value,enabled,fallback,code=specifications[control]
                need(env.get(key)==value and row['exit']==code,'control requested configuration '+control)
                if code==0:
                    need(int(row['frontier_stats']['enabled'])==enabled,'control admission '+control)
                    if fallback:need(row['frontier_stats']['fallback']==fallback,'control fallback '+control)
                if control in ['fixtures','pageable','full_window','poison','reuse0','reuse1','reuse2']:
                    need(env.get('NTT_SCALED_FRONTIER_CHECK')=='1','control full GMP nodes '+control)
                if control=='unchecked_poison':need(env.get('NTT_SCALED_FRONTIER_CHECK','0')=='0','unchecked poison fixture')
                need((case['name']=='m4423_large')==(control in ['budget476','budget477']),'control input shape '+control)
            if row['exit']:
                need(mode=='controls' and row['exit']==2 and row['token'] in combined,'intentional CLI rejection')
                need(not (base/(name+'.jsonl')).exists() or not (base/(name+'.jsonl')).read_text().strip(),'failed curve did not produce a completed result')
                continue
            need(row['exit']==0 and 'stage2_complete: curves=1' in driver.read_text(),'terminal native curve')
            result=bind(base/(name+'.jsonl'));records=[json.loads(s) for s in result.read_text().splitlines()]
            need(len(records)==1 and records[0]==row['result'] and sha(result)==row['result_sha256'] and sha(log)==row['log_sha256'],'raw full result')
            r=records[0];n=int(r['N_hex'],16);w=(n.bit_length()+63)//64;degree=phi(case['D'])//2
            need((r['B2'],r['requested_D'],r['device'],r['bad_factors'])==(case['B2'],case['D'],1,0),'actual native input')
            if case.get('N_hex'):need(r['N_hex'].lower()==case['N_hex'].lower() and r['B1']==case['B1'] and r['sigma']==26,'independent wide saved input')
            need(all(1<int(f)<n and n%int(f)==0 for f in r['factors']),'proper factors')
            if case['factor']:need(any(int(f)%case['factor']==0 for f in r['factors']),'known nonunit factor')
            shape=field(text,'real_shape');need(int(shape['giant_points'])==case['points']==case['B2']//case['D']+2 and int(shape['S_bits'])==n.bit_length(),'actual points/bits')
            for key,prefix in [('wall','stage2_full_wall'),('leaf','descent_values'),('coverage','s4_multiply_stats'),('ledger','real_batched_wall'),('root','scaled_root_device'),('scaled','scaled_descent'),('frontier_stats','scaled_frontier_device')]:need(field(text,prefix)==row[key],'raw field '+key)
            need(abs(float(row['ledger']['sum'])-float(row['wall']['main']))<=.003,'closed main ledger')
            for token in ['ntt_addsub_arithmetic: mask=1','ntt_outer_offsets: narrow_mask=0','stage1_skipped=1','mont_selftest: cases=2048 mismatches=0','s4_div_check: cases=800 bad=0','gmp_selftest_bad=0','gmp_check_bad=0','pending=0']:need(token in text,'mandatory check '+token)
            if case.get('expected_leaf'):need(row['leaf']['hash']==case['expected_leaf'],'independent monic evaluation')
            if case['name']=='m4423_large':
                need(row['leaf']==expected_large['leaf'],'original complete large evaluation')
                for key in ['N_hex','B1','B2','sigma','requested_D','device','factors','bad_factors']:need(r[key]==expected_large['result'][key],'original saved input/result')
                for key in ['launches','poly_muls','coeffs_reduced','gmp_selftest_cases','gmp_checked','full_checks']:need(row['coverage'][key]==expected_large['coverage'][key],'original required coverage')
            front=row['frontier_stats'];need(int(front['requested'])==row['frontier'],'actual frontier flag')
            if mode!='controls':need(int(front['enabled'])==(row['frontier'] if case.get('resident_expected',True) else 0),'declared admission')
            if front['enabled']=='1' or row['category'] in ['control_budget476','control_budget477']:
                mask=int(row['environment']['NTT_FOLD_OWNER_REUSE'])
                expected_owner=8*w*((7 if mask==3 else 8 if mask in [1,2] else 9)*degree+(7 if mask in [2,3] else 8))+48
                need(int(front['owner_and_metadata_bytes'])==expected_owner+24*degree,'combined owner/metadata budget')
            if front['enabled']=='1':
                g=geometry(degree,w);models[str((degree,w))]=g
                for key in ['metadata_bytes','f_h2d_bytes','metadata_h2d_bytes','leaf_d2h_bytes','avoided_parent_h2d_bytes','avoided_state_d2h_bytes']:need(int(front[key])==g[key],'exact interface geometry '+key)
                for key in ['levels','copies','zeros']:need(int(row['scaled'][key])==g[key],'exact state traversal '+key)
                need(int(row['scaled']['mul_calls'])==g['groups']+1 and int(row['scaled']['mul_pairs'])==g['pairs']+1,'exact multiply traversal')
                if mode.startswith('timing'):need(front['check_d2h_bytes']=='0' and row['root']['d2h_bytes']=='0','no diagnostic/root readback in formal sample')
            if mode=='gate' and case['D']==210 or row['environment'].get('NTT_SCALED_FRONTIER_CHECK')=='1':
                need(row['scaled']['checked_states']==row['scaled']['states'] and row['scaled']['checked_words']==row['scaled']['words'],'all independent GMP nodes')
            if row['category']=='control_fixtures':need('scaled_frontier_fixture: cases=150 bad=0' in text,'150 complete independent frontier fixtures')
            if mode=='gate' and case['name'].endswith('chunk_tail'):
                chain=[dict(re.findall(r'(\w+)=(\S+)',s)) for s in re.findall(r'^giant_chain_check: (.*)$',text,re.M)]
                need(sum(int(x['points']) for x in chain)==66240 and all(x['mismatches']=='0' for x in chain),'full chain/65-point tail')
                seed=field(text,'real_giant_seed');need(int(seed['checked_words'])==2*w*int(seed['points']) and seed['segments']==seed['segment_checks'],'seed/segment coverage')
            by_case.setdefault(case['name'],[]).append(row)
        summaries={}
        for name,rows in by_case.items():
            need(len({r['leaf']['hash'] for r in rows})==1 and len({tuple(r['result']['factors']) for r in rows})==1,'same complete result')
            if mode!='controls':
                for key in ['launches','poly_muls','coeffs_reduced','gmp_selftest_cases','gmp_checked','full_checks']:need(len({r['coverage'][key] for r in rows})==1,'required coverage conserved')
            if mode=='gate':need([r['frontier'] for r in rows]==[0,1],'gate order')
            if mode.startswith('timing'):
                timed=[r for r in rows if r['category']=='timing'];need([r['frontier'] for r in timed]==[0,1,1,0,1,0,0,1],'formal ABBA/BAAB order')
                need([r['frontier'] for r in rows if r['category']=='warmup']==[0,1],'warmup order')
                means={str(m):statistics.mean(float(r['wall']['total']) for r in timed if r['frontier']==m) for m in [0,1]};groups=[]
                for start in [0,4]:
                    v={m:statistics.mean(float(r['wall']['total']) for r in timed[start:start+4] if r['frontier']==m) for m in [0,1]};groups.append(100*(1-v[1]/v[0]))
                summaries[name]=dict(full_mean_seconds=means,reduction_percent=100*(1-means['1']/means['0']),groups_reduction_percent=groups)
                need(summaries[name]==report['summary'][name],'all full timing means')
                summaries[name]['descent_mean_seconds']={str(m):statistics.mean(float(field((base/(r['name']+'.log')).read_text(),'real_batched_split')['descent']) for r in timed if r['frontier']==m) for m in [0,1]}
                summaries[name]['full_samples_seconds']={str(m):[float(r['wall']['total']) for r in timed if r['frontier']==m] for m in [0,1]}
                summaries[name]['full_median_seconds']={m:statistics.median(samples) for m,samples in summaries[name]['full_samples_seconds'].items()}
        matrices[folder]=dict(processes=count,summary=summaries)
        if mode=='timing':formal_reference=report
    profiles={}
    if a.profiles:
        for mode in [0,1]:
            base=root/f'nsys_f{mode}';manifest=read(bind(base/'manifest.json'));summary=read(bind(base/'summary.json'))
            need(manifest['sha256']==binary_sha and manifest['compiled_add_sub_mask']==1 and manifest['env']['NTT_SCALED_FRONTIER_DEVICE']==str(mode),'profile binary/algorithm')
            need({n:h.lower() for n,h in manifest['sources'].items()}==sources,'profile source closure')
            need(manifest['collector_sha256']==sha(ROOT/'tools/bench/profile_stage2_points.py'),'profile collector identity')
            text=bind(base/'engine.log').read_text();records=[json.loads(s) for s in bind(base/'results.jsonl').read_text().splitlines()]
            need(len(records)==1 and 'stage2_complete: curves=1' in bind(base/'app.log').read_text(),'captured native terminal result')
            expected=next(r for r in formal_reference['runs'] if r['category']=='timing' and r['frontier']==mode)
            need(manifest['save_sha256']==formal_reference['cases'][0]['save_sha256']==sha(manifest['save']),'profile saved input')
            for key in ['N_hex','B1','B2','sigma','requested_D','device','factors','bad_factors']:need(records[0][key]==expected['result'][key],'captured native result')
            need(field(text,'descent_values')==expected['leaf'],'captured complete leaf')
            for key in ['launches','poly_muls','coeffs_reduced','gmp_selftest_cases','gmp_checked','full_checks']:need(field(text,'s4_multiply_stats')[key]==expected['coverage'][key],'captured required coverage')
            front=field(text,'scaled_frontier_device');need(int(front['enabled'])==mode and front['check_d2h_bytes']=='0','captured frontier scope')
            for name in ['profile.log','export.log','run.cmd']:bind(base/name)
            need(int(bind(root/f'nsys_f{mode}_exit.txt').read_text(encoding='utf-8-sig').strip())==0,'profile collector terminal exit')
            db=bind(base/'trace.sqlite');c=sqlite3.connect('file:'+db.as_posix()+'?mode=ro',uri=True)
            kernels=c.execute('select s.value,k.gridX,k.gridY,k.gridZ,k.blockX,k.blockY,k.blockZ,k.dynamicSharedMemory,count(*),sum(k.end-k.start)/1e9 from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName where k.deviceId=1 group by s.value,k.gridX,k.gridY,k.gridZ,k.blockX,k.blockY,k.blockZ,k.dynamicSharedMemory order by s.value,k.gridX,k.gridY,k.gridZ,k.blockX,k.blockY,k.blockZ,k.dynamicSharedMemory').fetchall()
            ntt=[r[:-1] for r in kernels if any(t in r[0] for t in ['tile_kernel','outer_coop_kernel','outer_fwd_kernel','outer_inv_kernel'])]
            times={t:sum(r[-1] for r in kernels if t in r[0]) for t in ['tile_kernel','outer_coop_kernel','s4_pack_gather_reverse_kernel']}
            copies=c.execute('select e.label,count(*),sum(m.bytes),sum(m.end-m.start)/1e9 from CUPTI_ACTIVITY_KIND_MEMCPY m join ENUM_CUDA_MEMCPY_OPER e on e.id=m.copyKind where m.deviceId=1 group by e.label order by e.label').fetchall()
            intervals=sorted((s,e) for table in ['KERNEL','MEMCPY','MEMSET'] for s,e in c.execute('select start,end from CUPTI_ACTIVITY_KIND_'+table+' where deviceId=1'))
            left,right=intervals[0];first=left;union=0
            for start,end in intervals[1:]:
                if start>right:union+=right-left;left,right=start,end
                else:right=max(right,end)
            union+=right-left;span=right-first;window=dict(span=span/1e9,event_union=union/1e9,no_own_gpu_event=(span-union)/1e9,percent_no_own_gpu_event=100*(span-union)/span)
            need(window==summary['window'],'actual captured activity window')
            memory={}
            for kind,label in [(2,'device'),(1,'pinned')]:
                live={};peak=total=allocs=frees=0;where='memKind=?'+(' and deviceId=1' if kind==2 else '')
                for pid,context,address,size,op in c.execute('select globalPid,contextId,address,bytes,memoryOperationType from CUDA_GPU_MEMORY_USAGE_EVENTS where '+where+' order by start',(kind,)):
                    key=(pid,context,address)
                    if op==0:need(key not in live,'unique live allocation');live[key]=size;total+=size;allocs+=1
                    else:need(op==1 and live.pop(key,None)==size,'matched allocation release');total-=size;frees+=1
                    peak=max(peak,total)
                memory[label]=dict(peak_bytes=peak,end_live_bytes=total,allocations=allocs,deallocations=frees)
            c.close();profiles[str(mode)]=dict(ntt_shapes=ntt,kernel_seconds=times,copies=copies,window=window,memory=memory)
        need(profiles['0']['ntt_shapes']==profiles['1']['ntt_shapes'],'same actual NTT shapes/calls')
        copies=[{r[0]:r for r in profiles[str(m)]['copies']} for m in [0,1]];g=geometry(126720,70)
        need(copies[0]['Host-to-Device'][2]-copies[1]['Host-to-Device'][2]==g['predicted_total_h2d_saved_bytes'],'predicted versus actual H2D bytes')
        need(copies[0]['Device-to-Host'][2]-copies[1]['Device-to-Host'][2]==g['predicted_root_and_descent_d2h_saved_bytes'],'predicted versus actual D2H bytes')
        need(copies[1]['Device-to-Device'][2]-copies[0]['Device-to-Device'][2]==g['predicted_extra_d2d_bytes'],'predicted unary-child D2D bytes')
        need(profiles['1']['memory']['device']['end_live_bytes']==0,'all candidate tracked device allocations released')
    report=dict(complete=True,audit_tool_sha256=sha(__file__),binary_sha256=binary_sha,sources=sources,native=matrices,geometry=models,profiles=profiles,verified_files=files,external_files=external,
        scope='Same-binary fixed-D qualification and full-curve timing; Systems is diagnostic. Published production unchanged; no new Auto B2 costs, larger 16k capacity or concurrent total-memory lease certification.')
    (root/'quantitative.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(complete=True,verified_files=len(files),profiles=len(profiles))))


if __name__=='__main__':main()
