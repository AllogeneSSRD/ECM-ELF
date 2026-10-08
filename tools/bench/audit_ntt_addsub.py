"""Independently reconstruct the fixed arithmetic study from frozen sources and raw logs."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import statistics

ROOT=Path(__file__).resolve().parents[2]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))


def require(ok,message):
    if not ok:raise ValueError(message)


def rows(path,prefix):
    return [dict(re.findall(r'(\w+)=(\S+)',line)) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.startswith(prefix)]


def mean(values):return statistics.mean(values)


def sass_functions(path):
    raw=path.read_bytes();text=raw.decode('utf-16' if raw.startswith(b'\xff\xfe') else 'utf-8')
    found={}
    for match in re.finditer(r'^\s*Function : (\S+)\s*$',text,re.M):
        end=text.find('Function :',match.end());body=text[match.end():end if end>=0 else len(text)]
        instructions=re.findall(r'^\s*/\*[0-9a-f]+\*/\s*(?:@\S+\s+)?([A-Z][A-Z0-9._]*)\b[^\n]*;',body,re.M)
        found[match[1]]=dict(instructions=len(instructions),non_nop=sum(op!='NOP' for op in instructions),opcodes=dict(Counter(instructions)))
    return found


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study',type=Path,required=True);a=p.parse_args()
    root=a.study.resolve();builds={};files={}
    def bind(path):
        files[str(path.relative_to(root))]=sha(path);return path
    for folder in ['primitive_r0',*[f'ntt_m{m}' for m in range(4)],'batch_r1_m0','batch_r1_m1']:
        path=root/folder;manifest=read(bind(path/'manifest.json'))
        binary=path/Path(manifest['exe']).name;require(sha(binary)==manifest['sha256'].lower(),'binary '+folder);bind(binary)
        for name,digest in manifest['sources'].items():require(sha(bind(path/'sources'/name))==digest.lower(),'source '+folder+name)
        builds[folder]=dict(binary_sha256=sha(binary),sources=manifest['sources'])
    for mask in range(4):
        project=root/f'project_m{mask}';generation=read(bind(project/'generation.json'))
        require(generation['mask']==mask and generation['generator_sha256']==sha(ROOT/'tools/bench/prepare_ntt_addsub.py'),'generator identity')
        require(generation['generated_sources']==builds[f'ntt_m{mask}']['sources'],'generated/compiled source closure')
        for name,digest in generation['generated_sources'].items():
            path=bind(project/name);require(sha(path)==digest,'generated source')
            data=path.read_bytes()
            for old,new in reversed(generation['edits'].get(name,[])):
                require(data.count(new.encode())==1,'unique reverse edit');data=data.replace(new.encode(),old.encode())
            require(hashlib.sha256(data).hexdigest()==generation['original_sources'][name],'exact reversal')
        for name,digest in generation['support'].items():require(sha(bind(project/name))==digest,'GMP dependency')
    primitive={}
    terminal=read(bind(root/'primitive_r0/terminal_confirmation.json'))
    require(terminal['complete'] and terminal['binary_sha256']==builds['primitive_r0']['binary_sha256'],'primitive terminal binary')
    for run,(name,code) in zip(terminal['runs'],[('gate',0),('fault',3)]):
        log=bind(root/'primitive_r0'/run['log'])
        require(run['exit']==code and sha(log)==run['log_sha256'] and log.read_bytes()==(root/f'primitive_r0/{name}.log').read_bytes(),'primitive terminal confirmation')
    require(len(terminal['runs'])==2,'two primitive terminal results')
    for name,fault in [('gate',0),('fault',1)]:
        path=bind(root/f'primitive_r0/{name}.log');r=rows(path,'addsub_gate: ')
        require(len(r)==4 and [(int(t['op']),int(t['ptx'])) for t in r]==[(0,0),(0,1),(1,0),(1,1)],'primitive coverage')
        require(all(int(t['words'])==1000210 and int(t['bad'])==(fault if (t['op'],t['ptx'])==('1','1') else 0) for t in r),'primitive GMP comparison')
        chains=rows(path,'addsub_chain_gate: ')
        require(len(chains)==4 and all(int(t['mask'])==i and t['bad']=='0' and t['iterations']=='64' and t['words']=='256' for i,t in enumerate(chains)),'dependent chains')
        require(f'addsub_done: fault={fault} bad={fault} payload_peak=24005040 live=0' in path.read_text(),'primitive completion')
        primitive[name]=dict(pairs_per_operation=1000210,operations=4,chain_cases=4,fault_detected=bool(fault))
    parsed=sass_functions(bind(root/'primitive_r0/sass.txt'));original=read(bind(root/'primitive_r0/sass_summary.json'))
    for entry in original['kernels']:
        require(parsed[entry['name']]['instructions']==entry['instructions'] and parsed[entry['name']]['opcodes']==entry['opcodes'],'primitive SASS reconstruction')
    primitive['sass']={name:value for name,value in parsed.items() if 'addsub_primitive' in name}
    gate=read(bind(root/'ntt_gate_r0/measurements.json'));require(gate['complete'] and gate['mode']=='gate' and len(gate['runs'])==4,'four-mask gates')
    for mask in range(4):
        path=root/f'ntt_gate_r0/mask{mask}';g=read(bind(path/'summary.json'))
        require(g['complete'] and g['binary_sha256']==builds[f'ntt_m{mask}']['binary_sha256'] and len(g['runs'])==17,'gate build/coverage')
        require(gate['identity'][str(mask)]['binary_sha256']==g['binary_sha256'] and gate['identity'][str(mask)]['sources']==builds[f'ntt_m{mask}']['sources'],'outer gate identity')
        for run in g['runs']:
            log=bind(path/(run['name']+'.log'));require(sha(log)==run['log_sha256'],'gate log identity');text=log.read_text()
            require(f'ntt_addsub_mask: value={mask}' in text,'compiled mask gate')
            if run['name'].startswith('gmp_'):
                require(run['exit']==0 and all(t in text for t in ['cases=96 words=27131904 bad=0','calls=4 words=3145728 bad=0','ntt_coop_probe: device=1 bad=0']),'GMP frequency/cached gate')
            if run['name']=='dense':require(run['exit']==0 and 'cases=24 words=2951568 bad=0 live=0' in text,'dense GMP gate')
            if run['name'].startswith('fault_'):require(run['exit']==3 and re.search(r'ntt_fuse_coop_check: .*bad=[1-9]',text),'fault gate')
            if run['name']=='dense_fault':require(run['exit']==3 and re.search(r'ntt_v_dense: .*bad=[1-9]',text),'dense fault gate')
            if run['name'].startswith('invalid_'):require(run['exit']==3,'invalid mask gate')
    ntt={};order=[0,1,3,2,1,2,0,3,2,3,1,0,3,0,2,1]
    for revision in ['r0','r1']:
        folder=root/f'ntt_timing_{revision}';report=read(bind(folder/'measurements.json'))
        require(report['complete'] and report['mode']=='timing' and len(report['runs'])==84 and report['identity']==gate['identity'],'timing matrix identity')
        require(report['gate_sha256']==sha(root/'ntt_gate_r0/measurements.json'),'timing gate binding')
        require(sha(bind(folder/'collector.py'))==report['tool_sha256'],'frozen timing collector')
        want=[(27,'warmup',m,m) for m in range(4)]+[(k,'timing',index,m) for k in range(23,28) for index,m in enumerate(order)]
        actual=[];by_k={}
        for run,(k,category,index,mask) in zip(report['runs'],want):
            require(run['name']==f'k{k}_{category}_{index}_m{mask}' and (run['k'],run['category'],run['mask'])==(k,category,mask),'timing order')
            log=bind(folder/(run['name']+'.log'));require(sha(log)==run['log_sha256'],'timing log')
            r=rows(log,'ntt_v_bench: ');require(len(r)==32 and 'ntt_v_done: live=0' in log.read_text(),'timing completion')
            for j,t in enumerate(r):require((int(t['run']),int(t['repeat']),int(t['mask']),int(t['N']),int(t['batch']),int(t['bad']))==(j//4,j%4,0,1<<k,1,0),'full convolution shape/result')
            value=mean(float(t['seconds']) for t in r if int(t['repeat'])>0);require(value==run['seconds'],'raw event mean')
            if category=='timing':by_k.setdefault(k,[]).append((mask,value))
        for k,items in by_k.items():
            means={str(m):mean(v for mask,v in items if mask==m) for m in range(4)}
            reductions={str(m):100*(1-means[str(m)]/means['0']) for m in [1,2,3]}
            require(report['statistics'][str(k)]==dict(mean_seconds=means,reduction_percent=reductions),'NTT summary reconstruction')
            waves=[]
            for start in range(0,16,4):
                w=dict(items[start:start+4]);waves.append({str(m):100*(1-w[m]/w[0]) for m in [1,2,3]})
            actual.append(dict(k=k,mean_seconds=means,reduction_percent=reductions,latin_waves_reduction_percent=waves))
        ntt[revision]=actual
    batch={}
    batch_gate=read(bind(root/'batch_gate_r1/measurements.json'))
    for mask in range(2):
        require(batch_gate['identity'][mask]['binary_sha256']==builds[f'batch_r1_m{mask}']['binary_sha256'] and batch_gate['identity'][mask]['sources']==builds[f'batch_r1_m{mask}']['sources'],'batch compiled identity')
    for revision in ['gate_r1','timing_r0','timing_r1']:
        folder=root/('batch_'+revision);report=read(bind(folder/'measurements.json'));is_gate=revision.startswith('gate')
        require(report['complete'] and len(report['runs'])==(4 if is_gate else 10) and report['identity']==batch_gate['identity'],'batch matrix identity')
        if not is_gate:require(report['gate_sha256']==sha(root/'batch_gate_r1/measurements.json'),'batch timing gate binding')
        require(sha(bind(folder/'collector.py'))==report['tool_sha256'],'batch collector')
        expected=[(f'gate_m{m}',m) for m in range(2)] if is_gate else [(f'warmup_m{m}',m) for m in range(2)]+[(f'timing_{i}_m{m}',m) for i,m in enumerate([0,1,1,0,1,0,0,1])]
        if is_gate:expected=[(name,m) for m in range(2) for name in [f'gate_m{m}',f'fault_m{m}']]
        timed=[];words=0
        for run,(name,mask) in zip(report['runs'],expected):
            require((run['name'],run['mask'])==(name,mask),'batch order');log=bind(folder/(name+'.log'));require(sha(log)==run['log_sha256'],'batch log')
            r=rows(log,'ntt_batch_sample: ');fault=name.startswith('fault');normal=name.startswith('gate')
            combinations=[(nb,pad,phase,rep) for nb,pad in ([(1,0),(1,17),(3,0),(3,17),(990,0),(990,17)] if normal else [(990,17 if fault else 0)]) for phase in range(3) for rep in range(1 if is_gate else 25)]
            require(len(r)==len(combinations),'batch full coverage')
            for t,(nb,pad,phase,rep) in zip(r,combinations):
                require(tuple(int(t[k]) for k in ['mask','phase','repeat','N','batch','stride','words','bad'])==(mask,phase,rep,2048,nb,2048+pad,nb*(2048+pad),1 if fault and phase==2 else 0),'batch full GMP output/padding/fault')
                words+=int(t['words'])
            require(run['exit']==(3 if fault else 0) and f'ntt_batch_done: mask={mask} bad={1 if fault else 0} live=0' in log.read_text(),'batch terminal result')
            resources=rows(log,'ntt_batch_resource: ')
            require(len(resources)==(12 if normal else 2),'batch resources coverage')
            for index,t in enumerate(resources):require(tuple(int(t[k]) for k in ['mask','inverse','regs','local','shared','dynamic','blocks'])==(mask,index%2,40,0,0,16384,3),'batch resources')
            if name.startswith('timing'):
                values={str(phase):mean(float(t['seconds']) for t in r if int(t['phase'])==phase and int(t['repeat'])>0) for phase in range(3)}
                require(values==run['seconds'],'batch event mean');timed.append((mask,values))
        stats={}
        if not is_gate:
            for phase in range(3):
                means={str(m):mean(v[str(phase)] for mask,v in timed if mask==m) for m in range(2)}
                groups=[]
                for start in [0,4]:
                    group=timed[start:start+4];base=mean(v[str(phase)] for mask,v in group if mask==0);candidate=mean(v[str(phase)] for mask,v in group if mask==1);groups.append(100*(1-candidate/base))
                stats[str(phase)]=dict(mean_seconds=means,reduction_percent=100*(1-means['1']/means['0']),groups_reduction_percent=groups)
                require(report['statistics'][str(phase)]=={k:v for k,v in stats[str(phase)].items() if k!='groups_reduction_percent'},'batch summary reconstruction')
        batch[revision]=dict(processes=len(report['runs']),compared_words_including_padding=words,statistics=stats)
    tiles={}
    for mask in range(2):
        path=bind(root/f'batch_r1_m{mask}/sass.txt');s=sass_functions(path)
        tiles[str(mask)]={n:v for n,v in s.items() if 'tile_kernelILb' in n};require(len(tiles[str(mask)])==4,'four tile SASS instances')
    static=read(bind(root/'static_resources.json'));require(static['complete'],'static resources completion')
    for mask in range(4):
        path=bind(root/f'ntt_m{mask}/resources.txt');require(sha(path)==static['masks'][str(mask)]['file_sha256'],'static resource export identity')
        raw=path.read_bytes();text=raw.decode('utf-16' if raw.startswith(b'\xff\xfe') else 'utf-8-sig')
        resources=dict(re.findall(r'^ (Function [^\r\n]+)\r?\n  ([^\r\n]+)',text,re.M))
        for entry in static['masks'][str(mask)]['resources']:require(resources.get(entry['name'])==entry['resource'],'static resource reconstruction')
        for inv in [0,1]:
            name=f'Function _Z11tile_kernelILb{inv}ELb1EEvPyPKyyiiS2_yyb:'
            require(re.search(r'REG:40 STACK:'+str(8 if mask==2 and inv else 0)+r' SHARED:0 LOCAL:0',resources[name]),'default warp resources')
    require('bad=5093256 live=0' in bind(root/'batch_gate_r0/gate_m0.log').read_text(),'preserve first rejected reference')
    require(b'mpz_powm_ui(w,g,(GL_P-1)/n,p)' in bind(root/'batch_m0/sources/tools/test/ntt_addsub_batch_probe.cu').read_bytes(),'preserve failed source')
    result=dict(complete=True,scope='Independent primitive/NTT/hot-batch study only. No native Stage2 A/B, candidate NCU, production promotion or new D/AutoB2 certification.',
                audit_tool_sha256=sha(__file__),builds=builds,primitive=primitive,ntt=ntt,batch=batch,tile_sass=tiles,static_resources=static,verified_files=files)
    (root/'quantitative.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(complete=True,verified_files=len(files),ntt_processes=168,batch_processes=24)))


if __name__=='__main__':main()
