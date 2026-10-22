"""Exercise both native forms and compare every probability preset to ecm_prob.

Idle DX11 only: no engine launch, GPU query, queue append or real ECM work.
"""
import argparse,ctypes as c,hashlib,json,math,re,subprocess,sys,time
from pathlib import Path
from win32_probe import window,close,capture,user

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'tools/ecm_prob'))
import model

def main():
    p=argparse.ArgumentParser()
    for key in ['exe','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--font-size',type=int,default=15)
    a=p.parse_args();out=a.out.resolve();out.mkdir();exe=a.exe.resolve()
    assert model.REP_FILE.is_file(),'Comparison requires the existing tools/ecm_prob/out/d_eff_representative.json calibration data'
    values=model.load_representative_d_eff();names={0:'suyama_s10',1:'edwards_Z2xZ8',2:'edwards_Z12',3:'param1_s10',4:'param3_s10',5:'param2_s10',6:'edwards_Z2xZ4',7:'edwards_Z4'}
    user.SetThreadDpiAwarenessContext.argtypes=[c.c_void_p];user.SetThreadDpiAwarenessContext.restype=c.c_void_p;user.SetThreadDpiAwarenessContext(c.c_void_p(-4))
    cfg=out/'ecm.toml';cfg.write_text('[shared]\nworktodo="worktodo.txt"\n[stage1]\ntarget="cpu"\n[gui]\nNumWorkers=1\n[workers.1]\nautostart=0\n');original=cfg.read_bytes()
    checks=0
    for language in ['en','zh']:
        state=out/f'{language}.toml';state.write_text('[ui]\npage=2\nlogs=false\ninspector=false\nfont_size='+str(a.font_size)+'\nscale=1\nlanguage="'+language+'"\n')
        log=out/f'{language}.log';si=subprocess.STARTUPINFO();si.dwFlags=subprocess.STARTF_USESHOWWINDOW;si.wShowWindow=0
        proc=subprocess.Popen([str(exe),'--config',str(cfg),'--ui-state',str(state),'--diagnostics',str(log),'--audit-app','--no-nvml','--no-pet','--language',language],cwd=out,startupinfo=si)
        hwnd=window(proc.pid,'ecm_gui_v2');user.SetWindowPos(hwnd,None,10,10,2200,1150,0x14);user.ShowWindow(hwnd,4)
        def records(prefix):
            return [dict(re.findall(r'(\w+)=([^\s]+)',s)) for s in log.read_text(encoding='utf-8').splitlines() if s.startswith(prefix+':')]
        def current():
            rows=records('form_sample');return rows[-1] if rows else {}
        def wait(predicate):
            end=time.monotonic()+12
            while time.monotonic()<end:
                sample=current()
                if sample and predicate(sample):return sample
                assert proc.poll() is None;time.sleep(.08)
            raise AssertionError(log.read_text(encoding='utf-8')[-4500:])
        def click_rect(rect):
            x1,y1,x2,y2=map(float,rect.split(','));x,y=round((x1+x2)/2),round((y1+y2)/2)
            for action in [0,1,2]:user.PostMessageW(hwnd,0x8021,action,(y<<16)|(x&65535));time.sleep(.12)
        def controls(page):return {int(s['id']):s['rect'] for s in records('curve_choice') if s['page']==str(page)}
        def fields(page):return {s['id']:s for s in records('form_field') if s['page']==str(page)}
        def check_geometry(page,columns):
            f=fields(page);assert f
            for entry in f.values():
                label=list(map(float,entry['label'].split(',')));inp=list(map(float,entry['input'].split(',')));action=list(map(float,entry['action'].split(',')))
                assert label[2]<=inp[0] and inp[2]>inp[0] and inp[2]<=float(entry['right'])+1,entry
                if action[2]:assert action[0]>=inp[2]-1 and action[2]<=float(entry['right'])+1,entry
            pairs=[('worker','curves'),('b1','b2')] if page==2 else [('size','curves'),('b1','b2'),('divisor','target')]
            for first,second in pairs:
                one=list(map(float,f[first]['input'].split(',')));two=list(map(float,f[second]['input'].split(',')))
                assert (abs(one[1]-two[1])<1) if columns==2 else (two[1]>one[1]),(one,two)
            first_column=[pairs[i][0] for i in range(len(pairs))]
            starts=[float(f[key]['input'].split(',')[0]) for key in (first_column if columns==2 else f)]
            assert max(starts)-min(starts)<=1,starts
        try:
            wait(lambda s:s['page']=='2' and s['columns']=='2');time.sleep(.3)
            gen=controls(2);assert set(gen)=={0,1,4,5},gen;check_geometry(2,2)
            for profile,curve in [(1,'e'),(0,'m0'),(5,'m2'),(4,'m3')]:
                click_rect(controls(2)[profile]);wait(lambda s:s['curve']==curve)
            user.PostMessageW(hwnd,0x8021,0,0);time.sleep(.25);capture(hwnd,out/f'generator-{language}-wide.png')
            user.PostMessageW(hwnd,0x8022,7,3);wait(lambda s:s['page']=='3' and s['columns']=='2');time.sleep(.3)
            choices=controls(3);assert set(choices)==set(names)
            ed=[list(map(float,choices[i].split(','))) for i in [1,2,6,7]]
            mont=[list(map(float,choices[i].split(','))) for i in [0,3,5,4]]
            assert len({r[1] for r in ed})==1 and len({r[1] for r in mont})==1 and ed[0][1]<mont[0][1]
            assert all(ed[i][0]<ed[i+1][0] for i in range(3));assert all(mont[i][0]<mont[i+1][0] for i in range(3))
            check_geometry(3,2)
            for empirical in [False,True]:
                if empirical:
                    click_rect(fields(3)['divisor']['action']);wait(lambda s:s['empirical']=='1')
                for profile,name in names.items():
                    click_rect(controls(3)[profile]);s=wait(lambda s:int(s['profile'])==profile)
                    expected=values[name] if empirical else model.THEORETICAL_D.get(name,model.THEORETICAL_D['suyama_s10']*values[name]/values['suyama_s10'])
                    assert math.isclose(float(s['divisor']),expected,rel_tol=1e-12),(s,expected);checks+=1
            click_rect(fields(3)['b2']['action']);wait(lambda s:s['ratio']=='0')
            click_rect(fields(3)['b2']['action']);wait(lambda s:s['ratio']=='1')
            user.PostMessageW(hwnd,0x8021,0,0);time.sleep(.25);capture(hwnd,out/f'probability-{language}-wide.png')
            user.SetWindowPos(hwnd,None,20,20,900,1000,0x14);wait(lambda s:s['columns']=='1');time.sleep(.3);check_geometry(3,1)
            capture(hwnd,out/f'probability-{language}-narrow.png')
            user.PostMessageW(hwnd,0x8022,7,2);wait(lambda s:s['page']=='2' and s['columns']=='1');time.sleep(.3);check_geometry(2,1)
            capture(hwnd,out/f'generator-{language}-narrow.png')
        finally:close(proc,hwnd)
        assert 'D3D11 device lost' not in log.read_text(encoding='utf-8')
    assert cfg.read_bytes()==original and not (out/'worktodo.txt').exists()
    result=dict(passed=True,preset_checks=checks,languages=2,font_size=a.font_size,wide_pairs=True,narrow_single=True,labels_left=True,inputs_aligned=True,actions_right=True,generator_supported_only=True,workflow_unchanged=True,calibration_sha256=hashlib.sha256(model.REP_FILE.read_bytes()).hexdigest(),exe_sha256=hashlib.sha256(exe.read_bytes()).hexdigest())
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))

if __name__=='__main__':main()
