"""Actual skin-selector clicks, all seven moods, hot cache reload and persistence."""
import argparse,ctypes as c,hashlib,json,os,re,shutil,subprocess,time,tomllib
from pathlib import Path
from win32_probe import window,close,user,capture,process_sample
from renderer_probe import renderer_clean

def main():
    p=argparse.ArgumentParser();p.add_argument('--exe',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    out=a.out.resolve();out.mkdir();exe=a.exe.resolve();state=out/'ui.toml';log=out/'skins.log';cfg=out/'ecm.toml'
    cfg.write_text('[gui]\nNumWorkers=1\n[workers.1]\nautostart=0\n');business=cfg.read_bytes()
    state.write_text('[ui]\npage=5\ninspector=false\nlogs=false\nlanguage="en"\n[pet]\nenabled=true\nanimate=false\n')
    user.SetThreadDpiAwarenessContext.argtypes=[c.c_void_p];user.SetThreadDpiAwarenessContext.restype=c.c_void_p;assert user.SetThreadDpiAwarenessContext(c.c_void_p(-4))
    user.GetGuiResources.argtypes=[c.c_void_p,c.c_ulong];user.GetGuiResources.restype=c.c_ulong
    startup=subprocess.STARTUPINFO();startup.dwFlags=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
    proc=subprocess.Popen([str(exe),'--config',str(cfg),'--ui-state',str(state),'--diagnostics',str(log),'--audit-app','--no-nvml'],cwd=out,startupinfo=startup)
    hwnd=window(proc.pid,'ecm_gui_v2');user.SetWindowPos(hwnd,None,80,220,1500,1100,0x14);user.ShowWindow(hwnd,4)
    pet=window(proc.pid,'ecm_gui_v2_pet');skins=['classic','miss_pink_elf','herrscher_of_human_ego','hi_love_elf']
    def snapshots(prefix):
        return [dict(re.findall(r'(\w+)=([^\s]+)',line)) for line in log.read_text(encoding='utf-8').splitlines() if line.startswith(prefix)]
    def wait(predicate):
        end=time.monotonic()+12
        while time.monotonic()<end:
            text=log.read_text(encoding='utf-8')
            if predicate(text):return
            assert proc.poll() is None;time.sleep(.04)
        raise AssertionError(log.read_text(encoding='utf-8')[-3000:])
    def click(value):
        x,y=map(int,value.split(','));assert x>0 and y>0,value;packed=(y<<16)|(x&65535)
        user.PostMessageW(hwnd,0x8021,0,packed);time.sleep(.14);user.PostMessageW(hwnd,0x8021,1,packed);time.sleep(.14);user.PostMessageW(hwnd,0x8021,2,packed);time.sleep(.18)
    try:
        wait(lambda t:'pet_skin_sample:' in t and 'pet_skin: id=classic ' in t)
        wait(lambda t:int(snapshots('pet_skin_sample:')[-1]['skin0'].split(',')[0])>250 and 0<int(snapshots('pet_skin_sample:')[-1]['skin0'].split(',')[1])<1000)
        assert snapshots('pet_skin_sample:')[-1]['symbols']=='1','Official name symbols are missing from the selected font'
        offset=len(log.read_text(encoding='utf-8'));user.PostMessageW(hwnd,0x8022,8,1)
        wait(lambda t:'pet: mood=1 ' in t[offset:])
        for index,skin in enumerate(skins):
            click(snapshots('pet_skin_sample:')[-1][f'skin{index}'])
            wait(lambda t:snapshots('pet_skin_sample:')[-1]['selected']==skin)
            wait(lambda t:f'pet_skin: id={skin} ' in t)
            assert window(proc.pid,'ecm_gui_v2_pet')==pet,'Switching should retain the native window'
            offset=len(log.read_text(encoding='utf-8'))
            for mood in range(7):
                user.PostMessageW(hwnd,0x8022,8,mood)
                wait(lambda t,m=mood:f'pet: mood={m} ' in t[offset:]);offset=len(log.read_text(encoding='utf-8'))
                if mood in [0,5]:capture(pet,out/f'{skin}-mood{mood}.png')
            capture(hwnd,out/f'settings-{skin}.png')
        # Warm all costumes first, then repeat changes to detect retained native resources.
        cold=process_sample(proc)
        memory_samples=[dict(phase='after-all-moods',**cold,gdi=user.GetGuiResources(int(proc._handle),0))]
        # Match the measured transition history: a direct jump to Wink on each
        # new PNG exercises a different decoder path from sequential rows 0..6.
        # Keep its cold cost in the evidence, then measure the fixed warm pattern.
        for warm in range(2):
            for index,skin in enumerate(skins):
                previous=log.read_text(encoding='utf-8').count('pet_skin:')
                click(snapshots('pet_skin_sample:')[-1][f'skin{index}'])
                wait(lambda t,skin=skin,n=previous:t.count('pet_skin:')>n and f'pet_skin_sample: selected={skin} ' in t[t.rfind('pet_skin:'):])
            memory_samples.append(dict(phase=f'warm-{warm+1}',**process_sample(proc),gdi=user.GetGuiResources(int(proc._handle),0)))
        before=process_sample(proc);gdi_before=user.GetGuiResources(int(proc._handle),0)
        for iteration in range(3):
            for index,skin in enumerate(skins):
                previous=log.read_text(encoding='utf-8').count('pet_skin:')
                click(snapshots('pet_skin_sample:')[-1][f'skin{index}'])
                wait(lambda t,skin=skin,n=previous:t.count('pet_skin:')>n and f'pet_skin_sample: selected={skin} ' in t[t.rfind('pet_skin:'):])
                assert window(proc.pid,'ecm_gui_v2_pet')==pet
            memory_samples.append(dict(phase=f'round-{iteration+1}',**process_sample(proc),gdi=user.GetGuiResources(int(proc._handle),0)))
            (out/'memory-samples.json').write_text(json.dumps(memory_samples,indent=2));print(json.dumps(memory_samples[-1]),flush=True)
        after=process_sample(proc);gdi_after=user.GetGuiResources(int(proc._handle),0)
        assert gdi_after-gdi_before<=2,(gdi_before,gdi_after)
        assert after['private_bytes']-before['private_bytes']<=8*1024*1024,(before,after)
        save=snapshots('usability_sample:')[-1]['save'];click(save)
        wait(lambda t:state.exists() and tomllib.loads(state.read_text(encoding='utf-8')).get('pet',{}).get('skin')=='hi_love_elf')
    finally:close(proc,hwnd)
    assert proc.returncode==0 and renderer_clean(log.read_text(encoding='utf-8'),'dx11') and 'pet_error:' not in log.read_text(encoding='utf-8')
    restart=out/'restart.log'
    proc=subprocess.Popen([str(exe),'--config',str(cfg),'--ui-state',str(state),'--diagnostics',str(restart),'--frames','12','--no-nvml'],cwd=out,startupinfo=startup)
    hwnd=window(proc.pid,'ecm_gui_v2');user.ShowWindow(hwnd,4)
    try:proc.wait(timeout=20)
    finally:
        if proc.poll() is None:close(proc,hwnd)
    assert proc.returncode==0 and 'pet_skin: id=hi_love_elf ' in restart.read_text(encoding='utf-8')
    assert tomllib.loads(state.read_text(encoding='utf-8'))['pet']['skin']=='hi_love_elf' and cfg.read_bytes()==business
    # A missing optional skin reports the exact asset and disables only the pet.
    missing=out/'missing-skin';missing.mkdir();shutil.copy2(exe,missing/exe.name);missing_log=out/'missing-skin.log'
    proc=subprocess.Popen([str(missing/exe.name),'--ui-state',str(out/'missing.toml'),'--diagnostics',str(missing_log),'--pet','--pet-skin','miss_pink_elf','--frames','12','--no-nvml'],cwd=out,startupinfo=startup)
    hwnd=window(proc.pid,'ecm_gui_v2');user.ShowWindow(hwnd,4)
    try:proc.wait(timeout=20)
    finally:
        if proc.poll() is None:close(proc,hwnd)
    text=missing_log.read_text(encoding='utf-8');assert proc.returncode==0 and 'pet_error: Pet atlas missing' in text and 'pet_ely_miss_pink_elf.png' in text and renderer_clean(text,'dx11')
    invalid=subprocess.run([str(exe),'--pet-skin','../../outside'],capture_output=True,timeout=10);assert invalid.returncode==1 and b'Unknown ELY skin' in invalid.stdout
    result=dict(passed=True,renderer='dx11',skins=skins,moods_per_skin=7,actual_selector_clicks=True,retained_native_hwnd=True,symbols_present=True,restart_preserves_skin=True,missing_asset_graceful=True,arbitrary_asset_path_rejected=True,gdi_before=gdi_before,gdi_after=gdi_after,warm_cycles=2,measured_cycles=3,cold_pattern_growth_bytes=before['private_bytes']-cold['private_bytes'],private_bytes_growth=after['private_bytes']-before['private_bytes'],exe_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),business_unchanged=True,scope='One local DX11 native-window integration sequence; decoder/allocator cold transition cost is reported separately, not a complete P5 performance matrix')
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
