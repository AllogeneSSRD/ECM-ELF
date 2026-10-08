"""Reconstruct independent production descent qualification and measured capacity."""
import argparse
import collections
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sqlite3
import statistics
import sys

ROOT=Path(__file__).resolve().parents[2]
if hasattr(sys,'set_int_max_str_digits'):sys.set_int_max_str_digits(0)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
def need(ok,message):
    if not ok:raise ValueError(message)
def field(text,prefix):
    rows=re.findall(r'^'+re.escape(prefix)+r': (.*)$',text,re.M);need(bool(rows),'missing '+prefix)
    return dict(re.findall(r'(\w+)=(\S+)',rows[-1]))
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);p.add_argument('--profiles',action='store_true');a=p.parse_args()
    root=a.study.resolve();files={};external={}
    def bind(path):path=Path(path);files[path.relative_to(root).as_posix()]=sha(path);return path
    def ext(path,digest=None):path=Path(path).resolve();actual=sha(path);need(digest is None or actual==digest,'external identity '+str(path));external[str(path)]=actual;return path
    def frozen(exe):
        b=read(exe.parent/'build_manifest.json');f=read(exe.parent/'frozen_sources_manifest.json')
        need(sha(exe)==b['sha256'].lower()==f['binary_sha256'],'binary/build/snapshot identity')
        sources={n:h.lower() for n,h in b['source_hashes'].items()};need(sources==f['sources'],'raw source closure')
        for name,digest in sources.items():need(sha(exe.parent/'sources'/name)==digest,'frozen source '+name)
        return dict(binary_sha256=sha(exe),build_sha256=sha(exe.parent/'build_manifest.json'),snapshot_sha256=sha(exe.parent/'frozen_sources_manifest.json'),sources=sources)
    native=root/'production_r0';exe=bind(native/'ecm_cuda_stage2.exe');identity=frozen(exe);b=read(bind(native/'build_manifest.json'));bind(native/'frozen_sources_manifest.json')
    need((b['engine'],b['gl_fixed_mode'],b['outer_unroll_u'],b['add_sub_mask'])==('production',3,0,1),'selected production arithmetic')
    cmake=ext(ROOT/'CMakeLists.txt').read_text(encoding='utf-8-sig')
    need('target_compile_definitions(ecm_cuda_stage2 PRIVATE NTT_GL_FIXED_MODE=3 NTT_OUTER_UNROLL_U=0 NTT_GL_ADD_SUB_MASK=1)' in cmake,'CMake production host/CUDA arithmetic definitions')
    need(len(identity['sources'])==28 and not any(n.startswith('tools/bench/') for n in identity['sources']),'independent production closure')
    for name,digest in identity['sources'].items():need(sha(bind(native/'sources'/name))==digest==sha(ROOT/name),'compiled/current raw source '+name)
    need(len(b['objects'])==5,'five compiled objects')
    for name,digest in b['objects'].items():need(sha(bind(native/'_objects'/(name+'.obj')))==digest.lower(),'actual compiled object')
    raw=(ROOT/'src/cuda/ecm_cuda_stage2.cu').read_bytes();edits=read(bind(root/'production_source_edit.json'))
    for old,new in reversed(edits['edits']):need(raw.count(new.encode())==1,'unique production source reversal');raw=raw.replace(new.encode(),old.encode())
    need(raw==bind(root/'original_production.cu').read_bytes(),'production source byte reversal')
    oldroot=ROOT/'build_cuda_cmake/_stage2_root_prod_20261008/production_r3';oldexe=oldroot/'ecm_cuda_stage2.exe';oldidentity=frozen(oldexe);ext(oldexe)
    for name in ['build_manifest.json','frozen_sources_manifest.json']:ext(oldroot/name)
    for name,digest in oldidentity['sources'].items():ext(oldroot/'sources'/name,digest)
    for name,digest in read(oldroot/'build_manifest.json')['objects'].items():ext(oldroot/'_objects'/(name+'.obj'),digest.lower())
    need(raw==(oldroot/'sources/src/cuda/ecm_cuda_stage2.cu').read_bytes(),'actual old production source')
    runtime=(ROOT/'src/cuda/stage2/ntt_runtime.cuh').read_bytes().replace(b'#include "ntt_goldilocks_ptx.cuh"\n#include "ntt_goldilocks_sub.cuh"',b'#include "ntt_goldilocks_ptx.cuh"').replace(b'return gl_sub_canonical_ptx(a, b);',b'return gl_sub(a, b);')
    need(runtime==bind(root/'original_ntt_runtime.cuh').read_bytes()==(oldroot/'sources/src/cuda/stage2/ntt_runtime.cuh').read_bytes(),'production NTT delta')
    dev=ext(ROOT/'tools/bench/stage2_scaled_frontier.cuh').read_bytes();prod=(ROOT/'src/cuda/stage2/scaled_frontier.cuh').read_bytes()
    need(prod==dev.replace(b'// Development scaled descent.',b'// Production scaled descent.').replace(b'std::printf("scaled_frontier_device:',b'stage2_log::print(stage2_log::phases, "scaled_frontier_device:'),'validated frontier algorithm identity')
    dev=ext(ROOT/'tools/bench/ntt_goldilocks_addsub.cuh').read_bytes();need((ROOT/'src/cuda/stage2/ntt_goldilocks_sub.cuh').read_bytes()==dev[:dev.index(b'__device__ __forceinline__ unsigned long long gl_add_canonical_ptx')].rstrip()+b'\n','selected subtraction identity')
    resource=read(bind(root/'resources.json'));r=bind(native/'resources.txt');actual=dict(re.findall(r'^ (Function [^\r\n]+)\r?\n  ([^\r\n]+)',r.read_text(encoding='utf-8-sig'),re.M))
    need(resource['binary_sha256']==identity['binary_sha256'] and resource['raw_sha256']==sha(r) and actual==resource['resources'] and len(actual)==173,'actual production resources')
    for inv in [0,1]:need('REG:40 STACK:0 SHARED:0 LOCAL:0' in actual[f'Function _Z11tile_kernelILb{inv}ELb1EEvPyPKyyiiS2_yyb:'],'selected tile resources')
    need('REG:38 STACK:0 SHARED:0 LOCAL:0' in actual['Function _Z29s4_pack_gather_reverse_kernelPKyS0_yyiiiyyPy:'],'production reverse gather resources')
    geometry_module=ext(ROOT/'tools/bench/audit_stage2_frontier.py');spec=importlib.util.spec_from_file_location('geometry',geometry_module);gmodule=importlib.util.module_from_spec(spec);spec.loader.exec_module(gmodule)
    fixtures=read(ext(ROOT/'build_cuda_cmake/_stage2_wide_20261007/fixtures_r2/fixtures.json'));need(fixtures['complete'],'independent saved fixtures')
    gate=read(bind(root/'native_gate_r0/summary.json'));need(gate['complete'] and gate['passed']==62 and len(gate['runs'])==18 and len(gate['protocols'])==2 and len(gate['rejections'])==42,'full production native matrix')
    need(gate['identity']['binary_sha256']==identity['binary_sha256'] and gate['identity']['tool_sha256']==sha(bind(root/'native_gate_r0/collector.py')),'native collector identity')
    need(gate['identity']['fixture_sha256']==sha(ROOT/'build_cuda_cmake/_stage2_wide_20261007/fixtures_r2/fixtures.json'),'native saved fixture identity')
    for row in gate['runs']:
        base=root/'native_gate_r0';name=row['name'];driver=bind(base/(name+'_driver.log'));need(sha(driver)==row['driver_log_sha256'] and row['exit']==0,'native raw driver')
        log=bind(base/(name+'.log'));result=bind(base/(name+'.jsonl'));need(sha(log)==row['log_sha256'] and sha(result)==row['result_sha256'],'native raw result hashes')
        records=[json.loads(s) for s in result.read_text().splitlines()];need(records==[row['result']],'native full result')
        r=row['result'];n=int(r['N_hex'],16);need(r['bad_factors']==0 and all(1<int(f)<n and n%int(f)==0 for f in r['factors']),'native proper factors')
        case=next(c for c in fixtures['fixtures'] if c['N_hex'].lower()==r['N_hex'].lower());need(r['B1']==case['B1'] and r['sigma']==26 and r['B2']==(210*(row['count']-2) if row['count']>2 else case['B1']+1),'native fixture input')
        text=log.read_text()
        if row['level']=='debug':
            need(field(text,'descent_values')==row['leaf'],'native complete leaf')
            if case['unit']:need(row['leaf']['hash']==case['expected_leaf_hash'][str(row['count'])],'independent monic leaf definition')
            if case['expected_factor']:need(any(int(f)%case['expected_factor']==0 for f in r['factors']),'independent nonunit factor')
            for k,prefix in [('root','scaled_root_device'),('scaled','scaled_descent'),('frontier','scaled_frontier_device'),('gscale','real_gscale_device'),('ledger','real_batched_wall')]:need(field(text,prefix)==row[k],'native raw '+k)
            need(row['scaled']['checked_states']==row['scaled']['states'] and row['scaled']['checked_words']==row['scaled']['words'],'all GMP nodes')
            if name=='frontier_fixtures':need('scaled_frontier_fixture: cases=150 bad=0' in text,'150 production fixtures')
        elif row['level']=='quiet':need(not text.strip() and not driver.read_text().strip(),'quiet logging')
        if row['level']!='debug':need(not re.search(r'^(tree_level|descent_progress|mont_selftest|s4_div_check):',text,re.M),'no internal log leakage')
    for row in gate['protocols']:
        base=root/'native_gate_r0';driver=bind(base/(row['name']+'_driver.log'));need(sha(driver)==row['driver_log_sha256'] and row['exit']==0,'native protocol exit/output')
        if row['name']=='queue':
            result=read(bind(base/'queue.jsonl'));need(result['record']==2 and result['B2']==13230 and result['bad_factors']==0,'actual queued record/optional bounds')
            wide=next(c for c in fixtures['fixtures'] if c['name']=='generic16384');need(result['N_hex'].lower()==wide['N_hex'],'actual wide queue modulus')
            need(not bind(base/'queue.log').read_text().strip() and not driver.read_text().strip(),'Worker INI quiet protocol')
            need(bind(base/'finished.txt').read_text().strip()=='ECMSTAGE2=1,2,16384,-15,"three.save",13230,1,1','one successful queue transaction')
        elif row['name']=='plan':
            plans=[json.loads(x) for x in driver.read_text().splitlines() if x.startswith('{')];need(len(plans)==1 and plans[0]['curves_executed']==0 and plans[0]['bits']==16384 and plans[0]['words']==256,'actual zero-curve wide plan')
        else:need(False,'unknown native protocol')
    for row in gate['rejections']:
        base=root/'native_gate_r0';driver=bind(base/(row['name']+'_driver.log'));need(sha(driver)==row['driver_log_sha256'] and row['exit']==2,'native rejected exit/output')
        if 'log_sha256' in row:need(sha(bind(base/(row['name']+'.log')))==row['log_sha256'],'filtered fault log')
        cmd=row['command']
        if '--results' in cmd:need(not Path(cmd[cmd.index('--results')+1]).exists(),'failed native invocation produced result')
    controls=read(bind(root/'arithmetic_controls_r1/measurements.json'));need(controls['complete'] and len(controls['runs'])==9,'arithmetic/build controls')
    need(sha(bind(root/'arithmetic_controls_r1/collector.py'))==controls['tool_sha256'],'control collector snapshot')
    for row in controls['runs']:
        text=bind(root/'arithmetic_controls_r1'/(row['name']+'.log')).read_text(encoding='utf-8-sig',errors='replace');need(sha(root/'arithmetic_controls_r1'/(row['name']+'.log'))==row['log_sha256'] and row['token'] in text,'control raw rejection/result')
    for mask in [0,1]:
        row=next(r for r in controls['runs'] if r['name']==f'auto_m{mask}');control_exe=ext(Path(row['command'][0]));actual=frozen(control_exe);receipt=controls['identity'][mask]
        need(actual['binary_sha256']==receipt['binary_sha256'] and actual['build_sha256']==receipt['manifest_sha256'],'control compiled identity')
        for name,digest in receipt['objects'].items():need(sha(control_exe.parent/'_objects'/(name+'.obj'))==digest,'unchanged control compiled object')
    cross=read(bind(root/'development_cross_r0/summary.json'));reference_path=ext(ROOT/'build_cuda_cmake/_stage2_frontier_20261008/native_gate_r1/measurements.json',cross['reference_sha256']);reference=read(reference_path)
    ext(ROOT/'tools/bench/bench_stage2_production.py',cross['helper_sha256'])
    need(cross['complete'] and cross['identity']==identity and len(cross['runs'])==13,'complete development/production cross')
    need(sha(bind(root/'development_cross_r0/collector.py'))==cross['tool_sha256'],'cross collector snapshot')
    def raw_curve(base,row,name):
        text=bind(base/(name+'.log')).read_text();driver=bind(base/(name+'_driver.log'));result=bind(base/(name+'.jsonl'))
        need(sha(result)==row['result_sha256'] and sha(base/(name+'.log'))==row['log_sha256'],'raw curve hashes')
        need([json.loads(s) for s in result.read_text().splitlines()]==[row['result']] and 'stage2_complete: curves=1' in driver.read_text(),'terminal full native result')
        need(abs(float(field(text,'real_batched_wall')['sum'])-float(field(text,'stage2_full_wall')['main']))<=.003,'closed precise main ledger')
        for token in ['mont_selftest: cases=2048 mismatches=0','s4_div_check: cases=800 bad=0','gmp_selftest_bad=0','gmp_check_bad=0','pending=0']:need(token in text,'mandatory raw check '+token)
        n=int(row['result']['N_hex'],16);need(row['result']['bad_factors']==0 and all(1<int(f)<n and n%int(f)==0 for f in row['result']['factors']),'proper full curve factors')
        return text
    coverage_keys=['launches','poly_muls','coeffs_reduced','gmp_selftest_cases','gmp_checked','full_checks']
    for row in cross['runs']:
        old=next(r for r in reference['runs'] if r['frontier']==1 and r['case']==row['name']);text=raw_curve(root/'development_cross_r0',row,row['name'])
        for k in ['N_hex','B1','B2','sigma','requested_D','device','factors','bad_factors']:need(row['result'][k]==old['result'][k],'cross actual saved input/result')
        need(field(text,'descent_values')==row['leaf']==old['leaf'],'cross full leaf')
        for k in coverage_keys:need(field(text,'s4_multiply_stats')[k]==old['coverage'][k],'cross default coverage')
        if row['case']['D']==210:need(row['scaled']['checked_states']==row['scaled']['states'] and row['scaled']['checked_words']==row['scaled']['words'],'cross independent GMP nodes')
        if row['name'].endswith('chunk_tail'):
            chain=[dict(re.findall(r'(\w+)=(\S+)',s)) for s in re.findall(r'^giant_chain_check: (.*)$',text,re.M)];need(sum(int(x['points']) for x in chain)==66240 and all(x['mismatches']=='0' for x in chain),'cross full chain/65-tail')
    matrices={};formal={};models={}
    for folder,mode,count in [('cross_timing_r0','timing',10),('cross_timing_wide_r0','timing-wide',30)]:
        base=root/folder;j=read(bind(base/'measurements.json'));need(j['complete'] and j['mode']==mode and len(j['runs'])==count and j['resident_frontier'],'complete formal matrix')
        need(j['identity']=={'baseline':oldidentity,'candidate':identity},'actual compared production binaries');need(sha(bind(base/'collector.py'))==j['tool_sha256'],'formal collector snapshot')
        summaries={}
        for case in j['cases']:
            ext(case['save'],case['save_sha256']);rows=[r for r in j['runs'] if r['case']==case['name']]
            need([r['key'] for r in rows if r['category']=='warmup']==['baseline','candidate'],'declared warmup order')
            timed=[r for r in rows if r['category']=='timing'];need([r['key'] for r in timed]==['baseline','candidate','candidate','baseline','candidate','baseline','baseline','candidate'],'declared ABBA/BAAB order')
            for row in rows:
                text=raw_curve(base,row,row['name']);r=row['result'];shape=field(text,'real_shape')
                need((r['B2'],r['requested_D'],r['device'],r['sigma'])==(case['B2'],case['D'],1,26),'actual formal saved input')
                need(int(shape['giant_points'])==case['points']==case['B2']//case['D']+2 and int(shape['P'].split('=')[-1])==gmodule.phi(case['D'])//2,'actual formal integer geometry')
                if case['name']=='m4423_large':need(int(r['N_hex'],16)==2**4423-1 and r['B1']==1000,'actual M4423 anchor')
                else:
                    proof=next(c for c in fixtures['fixtures'] if c['name']==case['name']);need(r['N_hex'].lower()==proof['N_hex'] and r['B1']==proof['B1'],'actual independent wide save')
                for k,prefix in [('leaf','descent_values'),('coverage','s4_multiply_stats'),('wall','stage2_full_wall'),('phases','real_batched_split')]:need(field(text,prefix)==row[k],'formal raw '+k)
                if row['key']=='candidate':
                    need('ntt_addsub_arithmetic: mask=1' in text and row['frontier']['enabled']=='1' and row['frontier']['check_d2h_bytes']=='0' and row['root']['d2h_bytes']=='0','formal candidate policy/check scope')
            need(len({r['leaf']['hash'] for r in rows})==1 and len({tuple(r['result']['factors']) for r in rows})==1,'formal complete outputs')
            for k in coverage_keys:need(len({r['coverage'][k] for r in rows})==1,'formal required coverage')
            means={k:statistics.mean(float(r['wall']['total']) for r in timed if r['key']==k) for k in ['baseline','candidate']};groups=[]
            for start in [0,4]:
                v={k:statistics.mean(float(r['wall']['total']) for r in timed[start:start+4] if r['key']==k) for k in ['baseline','candidate']};groups.append(100*(1-v['candidate']/v['baseline']))
            summary=dict(full_mean_seconds=means,reduction_percent=100*(1-means['candidate']/means['baseline']),groups_reduction_percent=groups)
            need(summary==(j['summary'] if mode=='timing' else j['summary'][case['name']]),'recomputed formal statistics')
            summary['samples_seconds']={k:[float(r['wall']['total']) for r in timed if r['key']==k] for k in ['baseline','candidate']};summary['descent_mean_seconds']={k:statistics.mean(float(r['phases']['descent']) for r in timed if r['key']==k) for k in ['baseline','candidate']};summaries[case['name']]=summary
            formal[case['name']]=dict(case=case,rows=rows)
        matrices[folder]=summaries
    capacity=read(bind(root/'capacity_r0/summary.json'));need(capacity['complete'] and capacity['identity']=={'baseline':oldidentity,'candidate':identity} and len(capacity['plans'])==4 and len(capacity['runs'])==6 and capacity['formal_performance_samples']==0,'larger 16k capacity matrix')
    need(sha(bind(root/'capacity_r0/collector.py'))==capacity['tool_sha256'],'capacity collector snapshot')
    ext(ROOT/'tools/bench/bench_stage2_production.py',capacity['helper_sha256']);ext(ROOT/'build_cuda_cmake/_stage2_wide_20261007/fixtures_r2/fixtures.json',capacity['fixtures_sha256'])
    for plan in capacity['plans']:
        log=bind(root/'capacity_r0'/(plan['name']+'.log'));need(sha(log)==plan['log_sha256'],'raw capacity planning boundary');rows=[json.loads(s) for s in log.read_text().splitlines() if s.startswith('{')];need(rows==[plan['result']] and not plan['executed_curve'] and plan['result']['curves_executed']==0,'planning boundary not executed curve')
        result=plan['result'];degree=gmodule.phi(result['D'])//2;words=(result['bits']+63)//64
        combined=8*words*(7*degree+7)+48+24*degree
        need(result['D'] in [300300,600600] and result['P']==degree and result['words']==words==256 and result['I']==32768 and result['B1']==20 and result['B2']==result['D']*(32768-2) and plan['combined_owner_metadata_bytes']==combined,'independent planning geometry and owner/metadata payload')
        need((combined>640*2**20)==(result['D']==600600),'independent combined budget boundary')
    for case in capacity['cases']:
        ext(case['save'],case['save_sha256']);rows=[r for r in capacity['runs'] if r['case']==case['name']];need([r['key'] for r in rows]==['baseline','candidate'],'large capacity comparison order')
        for row in rows:
            text=raw_curve(root/'capacity_r0',row,row['name']);r=row['result'];need((r['N_hex'].lower(),r['B1'],r['B2'],r['requested_D'],r['device'],r['sigma'])==(case['N_hex'],case['B1'],case['B2'],case['D'],1,26),'actual capacity saved input');need(field(text,'real_shape')==row['shape'] and int(row['shape']['P'].split('=')[-1])==28800 and int(row['shape']['giant_points'])==32768,'actual larger 16k work geometry')
            need(field(text,'ntt_workspace_stats')==row['workspace'],'actual capacity workspace')
            need(int(row['fold']['enabled'])==int(case['owner_mb']!=0),'capacity owner admission')
            if row['key']=='candidate':need(int(row['frontier']['enabled'])==int(case['owner_mb']!=0) and row['frontier']['check_d2h_bytes']=='0','capacity frontier admission')
        need(rows[0]['leaf']==rows[1]['leaf'] and rows[0]['result']['factors']==rows[1]['result']['factors'],'capacity complete leaf/factor')
        for k in coverage_keys:need(rows[0]['coverage'][k]==rows[1]['coverage'][k],'capacity required coverage')
    profiles={}
    if a.profiles:
        current=ext(ROOT/'tools/bench/profile_stage2_points.py').read_bytes();prior=bind(root/'profile_collector_r0.py').read_bytes()
        current=current.replace(b"help='Verify compiled arithmetic and its actual curve log; production requires mask1'",b"help='Verify compiled development arithmetic and its actual curve log'")
        current=current.replace(b"engine=build.get('engine')\n        assert engine in ('development','production') and build.get('add_sub_mask')==a.add_sub_mask, 'Compiled arithmetic differs'\n        assert engine!='production' or a.add_sub_mask==1, 'Production requires canonical subtraction'",b"assert build.get('engine')=='development' and build.get('add_sub_mask')==a.add_sub_mask, 'Compiled arithmetic differs'")
        need(current==prior,'collector repair changed only production identity validation and help')
        for label in ['large','wide']:
            if label=='large':case=formal['m4423_large']['case'];reference_rows=formal['m4423_large']['rows']
            else:
                case=next(c for c in capacity['cases'] if c['name']=='generic16384_p28800_owner640');reference_rows=[r for r in capacity['runs'] if r['case']==case['name']]
            P=gmodule.phi(case['D'])//2;W=(int(reference_rows[0]['result']['N_hex'],16).bit_length()+63)//64;g=gmodule.geometry(P,W);models[label]=g
            pair={}
            for key in ['baseline','candidate']:
                name=f'nsys_{label}_{key}';base=root/name;manifest=read(bind(base/'manifest.json'));summary=read(bind(base/'summary.json'));exit_receipt=read(bind(root/(name+'_exit.json')))
                expected_identity=oldidentity if key=='baseline' else identity
                need(exit_receipt['exit']==0 and manifest['sha256']==expected_identity['binary_sha256'] and {n:h.lower() for n,h in manifest['sources'].items()}==expected_identity['sources'],'actual captured binary/source identity')
                need(manifest['collector_sha256']==sha(bind(base/'collector.py')),'actual profile collector snapshot')
                expected=next(r for r in reference_rows if r['key']==key and r.get('category','timing')=='timing')
                ext(case['save'],case['save_sha256']);need(manifest['save_sha256']==case['save_sha256'],'profile saved point')
                text=bind(base/'engine.log').read_text();result=[json.loads(x) for x in bind(base/'results.jsonl').read_text().splitlines()]
                need(len(result)==1 and 'stage2_complete: curves=1' in bind(base/'app.log').read_text(),'captured successful native curve')
                for k in ['N_hex','B1','B2','sigma','requested_D','device','factors','bad_factors']:need(result[0][k]==expected['result'][k],'captured actual input/full result')
                need(field(text,'descent_values')==expected['leaf'],'captured complete leaf')
                for k in coverage_keys:need(field(text,'s4_multiply_stats')[k]==expected['coverage'][k],'captured required check/work coverage')
                if key=='candidate':
                    front=field(text,'scaled_frontier_device');need(front['enabled']=='1' and front['check_d2h_bytes']=='0' and manifest['compiled_add_sub_mask']==1,'captured candidate policy')
                    for k in ['metadata_bytes','f_h2d_bytes','metadata_h2d_bytes','leaf_d2h_bytes','avoided_parent_h2d_bytes','avoided_state_d2h_bytes']:need(int(front[k])==g[k],'captured independent interface geometry '+k)
                for item in ['run.cmd','profile.log','export.log']:bind(base/item)
                db=bind(base/'trace.sqlite');c=sqlite3.connect('file:'+db.as_posix()+'?mode=ro',uri=True)
                kernels=c.execute('select s.value,k.gridX,k.gridY,k.gridZ,k.blockX,k.blockY,k.blockZ,k.dynamicSharedMemory,count(*),sum(k.end-k.start)/1e9 from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName where k.deviceId=1 group by s.value,k.gridX,k.gridY,k.gridZ,k.blockX,k.blockY,k.blockZ,k.dynamicSharedMemory order by s.value,k.gridX,k.gridY,k.gridZ,k.blockX,k.blockY,k.blockZ,k.dynamicSharedMemory').fetchall()
                shapes=[r[:-1] for r in kernels if any(t in r[0] for t in ['tile_kernel','outer_coop_kernel','outer_fwd_kernel','outer_inv_kernel'])]
                times={t:sum(r[-1] for r in kernels if t in r[0]) for t in ['tile_kernel','outer_coop_kernel','s4_reduce_kernel','s2g_ladder_kernel','s2g_chain_kernel','s2g_seed_pair_kernel','s4_pack_gather_reverse_kernel']}
                copies=c.execute('select e.label,count(*),sum(m.bytes),sum(m.end-m.start)/1e9 from CUPTI_ACTIVITY_KIND_MEMCPY m join ENUM_CUDA_MEMCPY_OPER e on e.id=m.copyKind where m.deviceId=1 group by e.label order by e.label').fetchall()
                intervals=sorted((start,end) for table in ['KERNEL','MEMCPY','MEMSET'] for start,end in c.execute('select start,end from CUPTI_ACTIVITY_KIND_'+table+' where deviceId=1'))
                left,right=intervals[0];first=left;union=0
                for start,end in intervals[1:]:
                    if start>right:union+=right-left;left,right=start,end
                    else:right=max(right,end)
                union+=right-left;span=right-first;window=dict(span=span/1e9,event_union=union/1e9,no_own_gpu_event=(span-union)/1e9,percent_no_own_gpu_event=100*(span-union)/span)
                need(window==summary['window'],'actual captured GPU event window')
                memory={}
                for kind,description in [(2,'device'),(1,'pinned')]:
                    live={};total=peak=allocs=frees=0;where='memKind=?'+(' and deviceId=1' if kind==2 else '')
                    for pid,ctx,address,size,op in c.execute('select globalPid,contextId,address,bytes,memoryOperationType from CUDA_GPU_MEMORY_USAGE_EVENTS where '+where+' order by start',(kind,)):
                        slot=(pid,ctx,address)
                        if op==0:need(slot not in live,'unique live allocation');live[slot]=size;total+=size;allocs+=1
                        else:need(op==1 and live.pop(slot,None)==size,'matched allocation release');total-=size;frees+=1
                        peak=max(peak,total)
                    memory[description]=dict(peak_bytes=peak,end_live_bytes=total,allocations=allocs,deallocations=frees)
                c.close();need(memory['device']['end_live_bytes']==0,'captured device allocations released')
                sensor=bind(root/(name+'_nvml.csv'));samples=[]
                for line in sensor.read_text().splitlines():
                    cols=[v.strip() for v in line.split(',')]
                    if len(cols)==7 and cols[1]=='1' and cols[3].isdigit():samples.append(cols)
                need(len(samples)>=2 and len({x[2] for x in samples})==1,'actual GPU1 NVML sampling')
                pair[key]=dict(ntt_shapes=shapes,kernel_seconds=times,copies=copies,window=window,memory=memory,nvml=dict(samples=len(samples),gpu_uuid=samples[0][2],device_memory_sample_max_mib=max(int(x[3]) for x in samples),scope='GPU1 whole-device 200ms samples including cold context; not process-only or continuous peak guarantee'))
            need(pair['baseline']['ntt_shapes']==pair['candidate']['ntt_shapes'],'same captured tile/outer geometries and counts')
            copies={k:{r[0]:r for r in v['copies']} for k,v in pair.items()}
            need(copies['baseline']['Host-to-Device'][2]-copies['candidate']['Host-to-Device'][2]==g['predicted_total_h2d_saved_bytes'],'actual versus predicted H2D bytes')
            need(copies['baseline']['Device-to-Host'][2]-copies['candidate']['Device-to-Host'][2]==g['predicted_root_and_descent_d2h_saved_bytes'],'actual versus predicted D2H bytes')
            need(copies['candidate']['Device-to-Device'][2]-copies['baseline']['Device-to-Device'][2]==g['predicted_extra_d2d_bytes'],'actual unary-child D2D bytes')
            profiles[label]=pair
    report=dict(complete=True,audit_tool_sha256=sha(__file__),binary_sha256=identity['binary_sha256'],source_count=28,native_passed=62,control_passed=9,cross_passed=13,formal=matrices,capacity=capacity,profiles=profiles,geometry=models,verified_files=files,external_files=external,scope='Independent production source and fixed-D full curves; no universal speedup or arbitrary-B2/16k memory certification. Published binary and old cost profiles remain unchanged until explicit release qualification.')
    (root/'quantitative.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(complete=True,files=len(files),profiles=len(profiles))))


if __name__=='__main__':main()
