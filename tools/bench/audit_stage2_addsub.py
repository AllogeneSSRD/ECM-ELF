"""Rebuild native subtraction qualification, timings and Systems evidence from raw outputs."""
import argparse
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
    lines=[s[len(prefix)+2:] for s in text.splitlines() if s.startswith(prefix+': ')]
    need(bool(lines),'missing '+prefix);return dict(re.findall(r'(\w+)=(\S+)',lines[-1]))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True)
    p.add_argument('--profiles',action='store_true');a=p.parse_args();root=a.study.resolve();files={};external={};builds=[]
    def bind(path):
        files[str(path.relative_to(root))]=sha(path);return path
    def bind_external(path,digest):
        need(sha(path)==digest,'external identity '+str(path));external[str(Path(path).resolve())]=digest
    for mask in [0,1]:
        base=root/f'native_m{mask}';b=read(bind(base/'build_manifest.json'));f=read(bind(base/'frozen_sources_manifest.json'))
        exe=bind(base/'ecm_cuda_stage2.exe')
        need((b['engine'],b['gl_fixed_mode'],b['outer_unroll_u'],b['add_sub_mask'])==('development',3,0,mask),'compiled settings')
        need(sha(exe)==b['sha256'].lower()==f['binary_sha256'],'compiled binary')
        sources={n:h.lower() for n,h in b['source_hashes'].items()};need(f['sources']==sources,'frozen closure')
        for name,digest in sources.items():need(sha(bind(base/'sources'/name))==digest==sha(ROOT/name),'compiled/staged source bytes')
        for name,digest in b['objects'].items():need(sha(bind(base/'_objects'/(name+'.obj')))==digest.lower(),'compiled object')
        builds.append(dict(binary_sha256=sha(exe),sources=sources,objects=b['objects']))
    need(builds[0]['sources']==builds[1]['sources'],'same native source closure')
    original=bind(root/'original_stage2_tree_gpu.cu');edits=read(bind(root/'stage2_source_edit.json'))
    raw=(ROOT/'tools/bench/stage2_tree_gpu.cu').read_bytes()
    for old,new in reversed(edits['edits']):need(raw.count(new.encode())==1,'unique source reversal');raw=raw.replace(new.encode(),old.encode())
    need(raw==original.read_bytes(),'raw mixed-newline source reversal')
    qualified=ROOT/'build_cuda_cmake/_stage2_addsub_20261008/primitive_r0/sources/tools/bench/ntt_goldilocks_addsub.cuh'
    need(sha(qualified)==sha(ROOT/'tools/bench/ntt_goldilocks_addsub.cuh'),'qualified arithmetic unchanged')
    resources=read(bind(root/'resources.json'));resource_changes=[]
    for mask in [0,1]:
        path=bind(root/f'native_m{mask}/resources.txt');want=resources['masks'][str(mask)]
        need(sha(path)==want['file_sha256'],'raw resource identity')
        actual=dict(re.findall(r'^ (Function [^\r\n]+)\r?\n  ([^\r\n]+)',path.read_text(),re.M));need(actual==want['resources'] and len(actual)==223,'full native resources')
        for inv in [0,1]:need('REG:40 STACK:0 SHARED:0 LOCAL:0' in actual[f'Function _Z11tile_kernelILb{inv}ELb1EEvPyPKyyiiS2_yyb:'],'default warp resources')
    r0=resources['masks']['0']['resources'];r1=resources['masks']['1']['resources']
    resource_changes=[dict(name=n,baseline=r0[n],candidate=r1[n]) for n in r0 if r0[n]!=r1[n]]
    controls=read(bind(root/'controls_r1/measurements.json'));need(controls['complete'] and len(controls['runs'])==9,'nine controls')
    need(sha(bind(root/'controls_r1/collector.py'))==controls['tool_sha256'],'controls collector')
    for run in controls['runs']:
        log=bind(root/'controls_r1'/(run['name']+'.log'));need(sha(log)==run['log_sha256'] and run['token'] in log.read_text(encoding='utf-8',errors='replace'),'control raw rejection')
        need(run['exit']==(1 if run['name'].startswith(('production','hostonly')) else 2 if run['name'].startswith('auto') else 0),'control exit')
    all_results={};matrix_identities=None;gate_path=root/'native_gate_r0/measurements.json'
    for folder,mode,total in [('native_gate_r0','gate',26),('native_timing_r0','timing',10),('native_timing_wide_r0','timing-wide',30)]:
        base=root/folder;report=read(bind(base/'measurements.json'));need(report['complete'] and report['mode']==mode and len(report['runs'])==total,'complete native matrix')
        need(sha(bind(base/'collector.py'))==report['tool_sha256'],'native collector identity')
        bind_external(ROOT/'tools/bench/bench_stage2_production.py',report['helper_sha256'])
        reference_path=ROOT/'build_cuda_cmake/_stage2_root_prod_20261008/cross_timing_final_r3/measurements.json'
        fixture_path=ROOT/'build_cuda_cmake/_stage2_wide_20261007/fixtures_r2/fixtures.json'
        bind_external(reference_path,report['reference_sha256']);bind_external(fixture_path,report['fixtures_sha256'])
        reference=read(reference_path)
        need(reference['complete'] and read(fixture_path)['complete'],'completed independent references')
        expected_large=next(r for r in reference['runs'] if r['category']=='timing' and r['key']=='candidate')
        if matrix_identities is None:matrix_identities=report['identity']
        need(report['identity']==matrix_identities,'same compiled matrix identities')
        for mask in [0,1]:need(report['identity'][mask]['binary_sha256']==builds[mask]['binary_sha256'] and report['identity'][mask]['sources']==builds[mask]['sources'],'matrix binary/source binding')
        if mode!='gate':need(report['gate_sha256']==sha(gate_path),'native gate binding')
        by_case={}
        for run in report['runs']:
            name=run['name'];mask=run['mask'];log=bind(base/(name+'.log'));result=bind(base/(name+'.jsonl'));driver=bind(base/(name+'_driver.log'))
            need(sha(log)==run['log_sha256'] and sha(result)==run['result_sha256'] and sha(driver)==run['driver_sha256'],'raw native files')
            text=log.read_text();need(run['exit']==0 and 'stage2_complete: curves=1' in driver.read_text(),'actual native terminal result')
            record=[json.loads(s) for s in result.read_text().splitlines()];need(len(record)==1 and record[0]==run['result'],'raw result record')
            record=record[0];case=next(c for c in report['cases'] if c['name']==run['case']);need(sha(case['save'])==case['save_sha256'],'saved input identity')
            bind_external(case['save'],case['save_sha256'])
            need((record['B2'],record['requested_D'],record['device'],record['bad_factors'])==(case['B2'],case['D'],1,0),'actual D/B2/device')
            n=int(record['N_hex'],16);need(all(1<int(f)<n and n%int(f)==0 for f in record['factors']),'proper factors')
            if case['factor']:need(any(int(f)%case['factor']==0 for f in record['factors']),'known nonunit factor')
            for token in [f'ntt_addsub_arithmetic: mask={mask}','ntt_outer_offsets: narrow_mask=0','stage1_skipped=1',
                'mont_selftest: cases=2048 mismatches=0','s4_div_check: cases=800 bad=0','gmp_selftest_bad=0','gmp_check_bad=0','pending=0']:need(token in text,'native check '+token)
            for key,prefix in [('wall','stage2_full_wall'),('leaf','descent_values'),('coverage','s4_multiply_stats'),('ledger','real_batched_wall'),('root','scaled_root_device'),('scaled','scaled_descent')]:need(field(text,prefix)==run[key],'raw field '+key)
            if case['name']=='m4423_large':
                need(run['leaf']==expected_large['leaf'],'complete original large reference')
                for key in ['N_hex','B1','B2','sigma','requested_D','device','factors','bad_factors']:
                    need(record[key]==expected_large['result'][key],'large reference '+key)
                for key in ['launches','poly_muls','coeffs_reduced','gmp_selftest_cases','gmp_checked','full_checks']:
                    need(run['coverage'][key]==expected_large['coverage'][key],'large reference coverage '+key)
            shape=field(text,'real_shape');need(int(shape['giant_points'])==case['points'] and int(shape['S_bits'])==n.bit_length(),'actual point/bit shape')
            need(abs(float(run['ledger']['sum'])-float(run['wall']['main']))<=.003,'precise timing closure')
            need(run['root']['requested']=='1' and run['root']['checked_words']=='0' and run['root']['check_d2h_bytes']=='0','root policy')
            if case.get('expected_leaf'):need(run['leaf']['hash']==case['expected_leaf'],'independent complete monic leaf')
            if mode=='gate' and case['D']==210:
                fixture=field(text,'scaled_fixture');need(fixture['cases']=='150' and fixture['bad']=='0','scaled GMP fixture')
                need(run['scaled']['checked_states']==run['scaled']['states'] and run['scaled']['checked_words']==run['scaled']['words'],'all GMP nodes')
            if run['chain']:
                affine=[dict(re.findall(r'(\w+)=(\S+)',s)) for s in re.findall(r'^giant_chain_check: (.*)$',text,re.M)]
                need(sum(int(r['points']) for r in affine)==66240 and all(r['mismatches']=='0' for r in affine),'all chain affine points')
                need(run['chain']==dict(checked_chain_points=66240,ladder_tail_points=65),'short tail shape')
                seed=field(text,'real_giant_seed');w=(n.bit_length()+63)//64
                need(int(seed['checked_words'])==2*w*int(seed['points']) and seed['segments']==seed['segment_checks'],'seed/segment checks')
            by_case.setdefault(case['name'],[]).append(run)
        summaries={}
        for name,runs in by_case.items():
            need(len({r['leaf']['hash'] for r in runs})==1 and len({tuple(r['result']['factors']) for r in runs})==1,'cross-build complete result')
            for key in ['launches','poly_muls','coeffs_reduced','gmp_selftest_cases','gmp_checked','full_checks']:need(len({r['coverage'][key] for r in runs})==1,'default coverage conservation')
            if mode=='gate':need([r['mask'] for r in runs]==[0,1],'gate order')
            else:
                timed=[r for r in runs if r['category']=='timing'];need([r['mask'] for r in timed]==[0,1,1,0,1,0,0,1],'formal ABBA/BAAB')
                need([r['mask'] for r in runs if r['category']=='warmup']==[0,1],'warmup order')
                means={str(m):statistics.mean(float(r['wall']['total']) for r in timed if r['mask']==m) for m in [0,1]};groups=[]
                for start in [0,4]:
                    group=timed[start:start+4];v={m:statistics.mean(float(r['wall']['total']) for r in group if r['mask']==m) for m in [0,1]};groups.append(100*(1-v[1]/v[0]))
                summaries[name]=dict(full_mean_seconds=means,reduction_percent=100*(1-means['1']/means['0']),groups_reduction_percent=groups)
                need(report['summary'][name]==summaries[name],'native mean reconstruction')
        all_results[folder]=dict(processes=total,summary=summaries)
    profiles={}
    if a.profiles:
        for mask in [0,1]:
            base=root/f'nsys_m{mask}';manifest=read(bind(base/'manifest.json'));summary=read(bind(base/'summary.json'))
            need(manifest['sha256']==builds[mask]['binary_sha256'] and manifest['compiled_add_sub_mask']==mask,'profile build/mode')
            need({n:h.lower() for n,h in manifest['sources'].items()}==builds[mask]['sources'],'profile compiled closure')
            need(manifest['collector_sha256']==sha(ROOT/'tools/bench/profile_stage2_points.py'),'profile collector identity')
            need(manifest['frozen_manifest_sha256']==sha(root/f'native_m{mask}/frozen_sources_manifest.json'),'profile frozen manifest')
            reference=read(root/'native_timing_r0/measurements.json')
            expected=next(r for r in reference['runs'] if r['category']=='timing' and r['mask']==mask)
            need(sha(manifest['save'])==manifest['save_sha256']==reference['cases'][0]['save_sha256'],'profile saved input')
            text=bind(base/'engine.log').read_text();need(f'ntt_addsub_arithmetic: mask={mask}' in text,'actual profile mode')
            records=[json.loads(s) for s in bind(base/'results.jsonl').read_text().splitlines()]
            need(len(records)==1,'single captured curve')
            for key in ['N_hex','B1','B2','sigma','requested_D','device','factors','bad_factors']:
                need(records[0][key]==expected['result'][key],'profile input/result '+key)
            need('stage2_complete: curves=1' in bind(base/'app.log').read_text(),'captured application completed')
            need(field(text,'descent_values')==expected['leaf'],'complete captured leaf')
            coverage=field(text,'s4_multiply_stats')
            for key in ['launches','poly_muls','coeffs_reduced','gmp_selftest_cases','gmp_checked','full_checks']:
                need(coverage[key]==expected['coverage'][key],'captured coverage '+key)
            for prefix in ['scaled_root_device','real_gscale_device']:
                stats=field(text,prefix)
                need(stats['requested']=='1' and stats['checked_words']=='0' and stats['check_d2h_bytes']=='0','no extra captured GMP copies')
            need(field(text,'scaled_root_device')['enabled']=='1','captured resident root')
            for token in ['ntt_outer_offsets: narrow_mask=0','stage1_skipped=1','gmp_check_bad=0','gmp_selftest_bad=0','pending=0']:
                need(token in text,'captured check '+token)
            for name in ['profile.log','export.log','run.cmd'] :bind(base/name)
            need(int((root/f'nsys_m{mask}_exit.txt').read_text(encoding='utf-8-sig').strip())==0,'profile collector terminal exit')
            db=bind(base/'trace.sqlite');con=sqlite3.connect('file:'+db.as_posix()+'?mode=ro',uri=True)
            memory=read(bind(base/'lifetimes.json'))
            need(memory['trace_sha256']==sha(db) and memory['tool_sha256']==sha(ROOT/'tools/bench/analyze_stage2_trace.py'),'lifetime trace/tool binding')
            for kind,key in [(2,'device_allocations'),(1,'pinned_host_allocations')]:
                live={};peak=total=allocations=deallocations=0
                where='memKind=?'+(' and deviceId=1' if kind==2 else '')
                for pid,context,address,size,op in con.execute('select globalPid,contextId,address,bytes,memoryOperationType from CUDA_GPU_MEMORY_USAGE_EVENTS where '+where+' order by start',(kind,)):
                    lease=(pid,context,address)
                    if op==0:
                        need(lease not in live,'unique allocation');live[lease]=size;total+=size;allocations+=1
                    else:
                        need(op==1 and live.pop(lease,None)==size,'matching free');total-=size;deallocations+=1
                    peak=max(peak,total)
                for name,value in [('peak_bytes',peak),('end_live_bytes',total),('allocations',allocations),('deallocations',deallocations)]:
                    need(memory[key][name]==value,'raw allocation reconstruction '+name)
            kernels=con.execute('select s.value,count(*),sum(k.end-k.start)/1e9 from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName where k.deviceId=1 group by k.shortName').fetchall()
            shapes=con.execute('select s.value,k.gridX,k.gridY,k.gridZ,k.blockX,k.blockY,k.blockZ,k.dynamicSharedMemory,count(*) from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName where k.deviceId=1 group by s.value,k.gridX,k.gridY,k.gridZ,k.blockX,k.blockY,k.blockZ,k.dynamicSharedMemory order by s.value,k.gridX,k.gridY,k.gridZ,k.blockX,k.blockY,k.blockZ,k.dynamicSharedMemory').fetchall()
            counts={};seconds={}
            for part,needle in [('tile','tile_kernel'),('outer','outer_coop_kernel')]:
                counts[part]=sum(r[1] for r in kernels if needle in r[0]);seconds[part]=sum(r[2] for r in kernels if needle in r[0])
            copies=con.execute('select e.label,count(*),sum(m.bytes) from CUPTI_ACTIVITY_KIND_MEMCPY m join ENUM_CUDA_MEMCPY_OPER e on e.id=m.copyKind where m.deviceId=1 group by m.copyKind').fetchall()
            intervals=sorted((start,end) for table in ['KERNEL','MEMCPY','MEMSET'] for start,end in con.execute('select start,end from CUPTI_ACTIVITY_KIND_'+table+' where deviceId=1'))
            left,right=intervals[0];first=left;union=0
            for start,end in intervals[1:]:
                if start>right:union+=right-left;left,right=start,end
                else:right=max(right,end)
            union+=right-left;span=right-first;window=dict(span=span/1e9,event_union=union/1e9,no_own_gpu_event=(span-union)/1e9,percent_no_own_gpu_event=100*(span-union)/span)
            need(window==summary['window'],'profile event union reconstruction');con.close()
            profiles[str(mask)]=dict(counts=counts,seconds=seconds,copies=copies,window=window,kernel_shapes=shapes,
                device_payload_peak_bytes=memory['device_allocations']['peak_bytes'],device_end_live_bytes=memory['device_allocations']['end_live_bytes'],
                pinned_peak_bytes=memory['pinned_host_allocations']['peak_bytes'],pinned_end_live_bytes=memory['pinned_host_allocations']['end_live_bytes'])
        need(profiles['0']['counts']==profiles['1']['counts'] and profiles['0']['copies']==profiles['1']['copies'],'same profile workload/transfers')
        need(profiles['0']['kernel_shapes']==profiles['1']['kernel_shapes'],'all actual kernel names/geometries/counts')
    report=dict(complete=True,audit_tool_sha256=sha(__file__),builds=builds,resources_changed=resource_changes,
        native=all_results,profiles=profiles,verified_files=files,external_files=external,
        scope='Frozen native qualification and complete fixed-D A/B; Systems is diagnostic. No candidate NCU, new D/AutoB2 calibration, large 16k capacity or production release certification.')
    (root/'quantitative.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(complete=True,verified_files=len(files),resources_changed=len(resource_changes),profiles=len(profiles))))


if __name__=='__main__':main()
