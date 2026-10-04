"""Accept the isolated Goldilocks integer-MMA experiment against GMP, including fault rejection."""
import argparse,hashlib,json,os,re,subprocess
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe',type=Path,required=True);p.add_argument('--device',type=int,default=1)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    if any(a.output.iterdir()):raise ValueError('Use a fresh output directory')
    exe=a.exe.resolve();sha=hashlib.sha256(exe.read_bytes()).hexdigest();checks={};resources=[];tile_resources=[];groups={}
    manifest=json.loads((exe.parent/'manifest.json').read_text(encoding='utf-8-sig'))
    repo=Path(__file__).resolve().parents[2]
    def verify():
        assert hashlib.sha256(exe.read_bytes()).hexdigest()==sha==manifest['sha256'].lower()
        for name,want in manifest['sources'].items():
            assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==want.lower(),name
    verify()
    env={k:v for k,v in os.environ.items() if not k.startswith('NTT_')}
    for mode,code in [('--check',0),('--fault',3),('--tile-check',0),('--tile-fault',3)]:
        verify()
        r=subprocess.run([str(exe),str(a.device),mode],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=180)
        text=r.stdout.decode('utf-8',errors='replace');(a.output/(mode[2:]+'.log')).write_text(text,encoding='utf-8')
        checks[mode+'_exit']=r.returncode==code
        checks[mode+'_no_live_allocation']=bool(re.search(r'tc_gold_memory: requested_peak_bytes=\d+ live_bytes=0',text))
        checks[mode+'_immutable_exe']=hashlib.sha256(exe.read_bytes()).hexdigest()==sha
        verify()
        if mode=='--check':
            for row in re.findall(r'tc_gold_check: (.*)',text):
                item=dict(re.findall(r'(\w+)=([^\s]+)',row));groups[item['group']]=item
            expected={'matrix_tc':(256,204800),'matrix_cuda':(128,102400),'transform_tc':(512,409600),
                      'transform_cuda':(512,409600),'roundtrip_tc':(256,204800),'roundtrip_cuda':(256,204800)}
            for name,(cases,words) in expected.items():
                row=groups.get(name,{})
                checks[name+'_gmp']=row.get('cases')==str(cases) and row.get('words')==str(words) and row.get('bad')=='0'
            checks['132bit_carry_checked']=all(groups.get(k,{}).get('top_words')==str(n) and
                int(groups.get(k,{}).get('nonzero_tops','0'))>0 for k,n in [('matrix_tc',204800),('transform_tc',409600)])
            for row in re.findall(r'tc_gold_resources: (.*)',text):resources.append(dict(re.findall(r'(\w+)=([^\s]+)',row)))
            checks['resources_complete']=len(resources)==16 and len({(r['kernel'],r['threads']) for r in resources})==16
            checks['no_local_spill']=len(resources)==16 and all(r['local']=='0' and int(r['max_blocks'])>0 for r in resources)
            checks['all_math_zero']=bool(re.search(r'tc_gold_gate: fault=0 injected=0 bad=0',text))
        elif mode=='--fault':
            checks['fault_compared_and_rejected']=bool(re.search(r'tc_gold_check: group=matrix_tc .*bad=1\b',text)) and \
                bool(re.search(r'tc_gold_gate: fault=1 injected=1 bad=1',text))
        elif mode=='--tile-check':
            for row in re.findall(r'tc_gold_check: (.*)',text):
                item=dict(re.findall(r'(\w+)=([^\s]+)',row));groups[item['group']]=item
            expected={'tile_forward_tc':(384,2102016),'tile_inverse_tc':(384,2102016),'tile_roundtrip_tc':(384,2102016),
                      'tile_forward_cuda':(128,700672),'tile_inverse_cuda':(128,700672),'tile_roundtrip_cuda':(128,700672),
                      'tile_B_readonly':(512,2802688),'tile_stride_guards':(512,17408)}
            for name,(cases,words) in expected.items():
                row=groups.get(name,{})
                checks[name+'_gmp']=row.get('cases')==str(cases) and row.get('words')==str(words) and row.get('bad')=='0'
            for row in re.findall(r'tc_gold_tile_resources: (.*)',text):tile_resources.append(dict(re.findall(r'(\w+)=([^\s]+)',row)))
            checks['tile_resources_complete']=len(tile_resources)==8 and len({(r['kernel'],r['threads']) for r in tile_resources})==8
            checks['tile_no_local_spill']=len(tile_resources)==8 and all(r['local']=='0' and int(r['max_blocks'])>0 for r in tile_resources)
            checks['all_tile_math_zero']=bool(re.search(r'tc_gold_tile_gate: fault=0 injected=0 bad=0',text))
        else:
            checks['tile_fault_compared_and_rejected']=bool(re.search(r'tc_gold_check: group=tile_forward_tc .*bad=1\b',text)) and \
                bool(re.search(r'tc_gold_tile_gate: fault=1 injected=1 bad=1',text))
    result={'exe':str(exe),'sha256':sha,'manifest':manifest,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'device':a.device,'checks':checks,'groups':groups,'resources':resources,'tile_resources':tile_resources,
            'passed':sum(checks.values()),'failed':sum(not v for v in checks.values())}
    (a.output/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2));return int(result['failed']!=0)
if __name__=='__main__':raise SystemExit(main())
