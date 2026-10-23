"""Native Group Order Factor UI: form, import, async GP/native and TOML save.

Uses the audit pointer transport, real DX11 frames and isolated workflow files.
"""
import argparse,ctypes as c,hashlib,json,re,subprocess,time,tomllib,os,shutil
from pathlib import Path
from win32_probe import window,close,capture,user
from renderer_probe import renderer_clean

def clipboard_text():
    user.OpenClipboard.argtypes=[c.c_void_p];user.OpenClipboard.restype=c.c_int
    user.GetClipboardData.argtypes=[c.c_uint];user.GetClipboardData.restype=c.c_void_p
    kernel=c.WinDLL('kernel32',use_last_error=True)
    kernel.GlobalLock.argtypes=[c.c_void_p];kernel.GlobalLock.restype=c.c_void_p
    kernel.GlobalUnlock.argtypes=[c.c_void_p]
    assert user.OpenClipboard(None)
    try:
        handle=user.GetClipboardData(13);assert handle
        address=kernel.GlobalLock(handle);assert address
        try:return c.wstring_at(address)
        finally:kernel.GlobalUnlock(handle)
    finally:user.CloseClipboard()


def main():
    p=argparse.ArgumentParser()
    for key in ['exe','gp','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--gp-only',action='store_true',help='Relocate only GUI/XML, with no helper or GMP DLL')
    a=p.parse_args();exe=a.exe.resolve();gp=a.gp.resolve();out=a.out.resolve();out.mkdir();cases=[]
    if a.gp_only:
        portable=out/'bin';portable.mkdir();shutil.copy2(exe,portable/exe.name);shutil.copytree(exe.parent/'localization/v2',portable/'localization/v2');exe=portable/exe.name
        assert not (portable/'ecm_group_order_ui.exe').exists() and not (portable/'gmp-10.dll').exists()
    user.SetThreadDpiAwarenessContext.argtypes=[c.c_void_p];user.SetThreadDpiAwarenessContext.restype=c.c_void_p;user.SetThreadDpiAwarenessContext(c.c_void_p(-4))
    for language,font in [('en',15),('zh',15),('zh',24)]:
        folder=out/f'{language}{font}';folder.mkdir();cfg=folder/'ecm.toml';state=folder/'ui.toml';log=folder/'ui.log'
        cfg.write_text('# retain this comment\n[shared]\nb2_ratio=10\n[gui]\nNumWorkers=1\n[workers.1]\nautostart=0\n',encoding='utf-8')
        files=[folder/'results.json.txt',folder/'results.json_2.txt']
        for index,path in enumerate(files):
            row=dict(status='F',worktype='ECM',factors=['1009'],exponent=5237)
            if index:row['Edwards']=dict(sigma=26)
            else:row.update(sigma=26,param=1)
            path.write_text(json.dumps(row)+'\n',encoding='utf-8')
        originals=[path.read_bytes() for path in files]
        state.write_text(f'[ui]\npage=7\nlogs=false\ninspector=false\nscale=1\nfont_size={font}\nlanguage="{language}"\n')
        si=subprocess.STARTUPINFO();si.dwFlags=subprocess.STARTF_USESHOWWINDOW;si.wShowWindow=0
        env=os.environ.copy();env['PATH']=str(gp.parent)+os.pathsep+env.get('PATH','')
        proc=subprocess.Popen([str(exe),'--config',str(cfg),'--ui-state',str(state),'--diagnostics',str(log),'--audit-app','--no-nvml','--no-pet'],cwd=folder,startupinfo=si,env=env)
        hwnd=window(proc.pid,'ecm_gui_v2');user.SetWindowPos(hwnd,None,20,20,1920,1400,0x14);user.ShowWindow(hwnd,4)
        def samples():
            content=log.read_text(encoding='utf-8');cut=content.rfind('\ninteraction_sample:');content=content[:cut] if cut>=0 else ''
            rows=[dict(re.findall(r'(\w+)=([^\s]+)',line)) for line in content.splitlines() if line.startswith('group_order_ui:')]
            return rows[-1] if rows else {}
        def wait(predicate):
            end=time.monotonic()+40
            while time.monotonic()<end:
                row=samples()
                if row and predicate(row):return row
                assert proc.poll() is None;time.sleep(.05)
            raise AssertionError(samples())
        def pointer(pos,action):
            x,y=map(round,pos);before=log.read_text(encoding='utf-8').count('\ninteraction_sample:')
            user.PostMessageW(hwnd,0x8021,action,(y<<16)|(x&65535));end=time.monotonic()+8
            while log.read_text(encoding='utf-8').count('\ninteraction_sample:')<before+2:
                assert proc.poll() is None and time.monotonic()<end;time.sleep(.025)
        def click(key):
            time.sleep(.15) # Result/stale hints can change the following controls' Y positions.
            pos=tuple(map(float,samples()[key].split(',')))
            for action in [0,1,2]:pointer(pos,action)
            time.sleep(.15)
        def type_text(value):
            for ch in value:
                if ch=='\b':
                    user.PostMessageW(hwnd,0x100,8,0);time.sleep(.06);user.PostMessageW(hwnd,0x101,8,0);time.sleep(.06)
                else:user.PostMessageW(hwnd,0x102,ord(ch),0)
            time.sleep(.2)
        def backend(index):
            click('go_algorithm');wait(lambda r:f'go_backend{index}' in r);time.sleep(.3);click(f'go_backend{index}');wait(lambda r:r['backend']==str(index))
        def row_geometry():
            row=samples();coords=[tuple(map(float,row[key].split(','))) for key in ['go_gp','go_browse','save','go_algorithm']]
            assert all(coords[n][0]<coords[n+1][0] for n in range(3)),coords
            assert max(v[1] for v in coords)-min(v[1] for v in coords)<=1,coords
        try:
            wait(lambda r:r['files']=='2' and 'factor' in r and 'curve1' in r)
            click('factor');type_text('1009');click('sigma');type_text('\b'*20+'26')
            wait(lambda r:r.get('input_factor')=='1009' and r.get('input_sigma')=='26')
            row_geometry();click('save');time.sleep(.2)
            saved=tomllib.loads(cfg.read_text(encoding='utf-8'));assert saved['gui']['group_order_gp']==str(gp) and saved['shared']['b2_ratio']==10 and saved['workers']['1']['autostart']==0
            assert cfg.read_text(encoding='utf-8').startswith('# retain this comment')
            if not a.gp_only:backend(2)
            click('calculate');r=wait(lambda r:r.get('status')=='complete' and r['busy']=='0');assert int(r['group'])%int(r['point'])==0
            assert r['success']=='1' and r['toast_error']=='0' and int(r['toast_count'])>0
            click('go_copy');copied=clipboard_text()
            assert '#E(F_p): ' in copied and 'ord(P): ' in copied and '\n    = ' in copied and '×' in copied
            assert not any(text in copied for text in ['primality','proven','素数','已证明','"protocol"'])
            (folder/'copied.txt').write_text(copied,encoding='utf-8')
            completed=int(r['completions']);click('calculate');r=wait(lambda r:int(r['completions'])==completed+1 and r['busy']=='0')
            assert r['success']=='1' and r['toast_error']=='0'
            # A mathematical failure is asynchronous; invalid text fails before spawning.
            click('factor');type_text('\b'*4+'15');click('calculate');r=wait(lambda r:int(r['completions'])==completed+2 and r['busy']=='0')
            assert r['status']=='invalid_input' and r['success']=='0' and r['toast_error']=='1'
            click('factor');type_text('\b'*2+'1009');click('sigma');type_text('\b'*2+'x');click('calculate')
            r=wait(lambda r:int(r['completions'])==completed+3);assert r['success']=='0' and r['toast_error']=='1'
            click('sigma');type_text('\b'+'26');click('calculate');r=wait(lambda r:int(r['completions'])==completed+4 and r['busy']=='0')
            assert r['success']=='1' and r['status']=='complete'
            capture(hwnd,folder/('auto-gp-result.png' if a.gp_only else 'native-result.png'))
            click('curve2');time.sleep(.1);assert samples()['profile']=='0','Unsupported Edwards choice must stay disabled'
            # The two real result-file shapes restore parameterization, sigma and p.
            imported=[]
            for index in range(2):
                click('go_files');wait(lambda r:f'go_file{index}' in r);time.sleep(.3);click(f'go_file{index}')
                wait(lambda r:r['selected']==str(index) and r['records']=='1' and 'go_record0' in r)
                click('go_record0');r=wait(lambda r:r['profile'] in ['1','3']);imported.append(int(r['profile']))
            assert sorted(imported)==[1,3],imported
            selected=samples()['selected'];click('go_refresh');wait(lambda r:r['selected']==selected and r['records']=='1')
            backend(1);click('calculate');r=wait(lambda r:r['busy']=='0' and r.get('status')=='complete' and r.get('group')=='1024' and r.get('point')=='32');assert r['profile']=='1'
            capture(hwnd,folder/'gp-import-result.png')
            user.SetWindowPos(hwnd,None,20,20,900,1200,0x14);time.sleep(.5);r=samples()
            row_geometry()
            for key in ['factor','sigma','calculate','save','go_gp']:
                x,y=map(float,r[key].split(','));assert 20<x<920 and 20<y<1220,(key,r)
            capture(hwnd,folder/'narrow.png');cases.append(dict(language=language,font=font,imported_profiles=imported,narrow_width=900,retest_toast=True,async_failure_toast=True,validation_failure_toast=True,plain_math_copy=True))
        except BaseException:
            capture(hwnd,folder/'failure.png');raise
        finally:close(proc,hwnd)
        assert [path.read_bytes() for path in files]==originals
        assert renderer_clean(log.read_text(encoding='utf-8'),'dx11')
    result=dict(passed=True,cases=cases,exe_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),no_worker_or_gpu=True,gp_without_helper_or_gmp=a.gp_only)
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
