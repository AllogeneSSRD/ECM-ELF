"""Discover, select, validate and persist actual external XML filenames."""
import argparse,ctypes as c,hashlib,json,re,shutil,subprocess,time,tomllib
from pathlib import Path
from win32_probe import window,close,capture,user
from language_probe import select_language
from renderer_probe import renderer_clean

def main():
    p=argparse.ArgumentParser()
    for name in ['exe','out']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();source=a.exe.resolve();out=a.out.resolve();out.mkdir()
    runtime=out/'runtime-中文';runtime.mkdir();exe=runtime/source.name;shutil.copy2(source,exe)
    shutil.copytree(source.parent/'localization',runtime/'localization');loc=runtime/'localization/v2'
    cfg=out/'ecm.toml';cfg.write_text('[gui]\nNumWorkers=1\n[workers.1]\nautostart=0\n');before=cfg.read_bytes()
    state=out/'ui.toml';state.write_text('[ui]\nlanguage="en"\nscale=1\npage=0\nlogs=false\ninspector=false\n')
    user.SetThreadDpiAwarenessContext.argtypes=[c.c_void_p];user.SetThreadDpiAwarenessContext.restype=c.c_void_p
    assert user.SetThreadDpiAwarenessContext(c.c_void_p(-4))
    startup=subprocess.STARTUPINFO();startup.dwFlags=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
    log=out/'ui.log';proc=subprocess.Popen([str(exe),'--config',str(cfg),'--ui-state',str(state),'--diagnostics',str(log),'--audit-app','--no-nvml','--no-pet'],cwd=out,startupinfo=startup)
    hwnd=window(proc.pid,'ecm_gui_v2');user.SetWindowPos(hwnd,None,40,40,1500,1100,0x14);user.ShowWindow(hwnd,4)
    def row(prefix):
        rows=[line for line in log.read_text(encoding='utf-8').splitlines() if line.startswith(prefix+':')]
        return dict(re.findall(r'(\w+)=([^\s]+)',rows[-1])) if rows else {}
    def wait(predicate):
        end=time.monotonic()+12
        while time.monotonic()<end:
            if predicate():return
            assert proc.poll() is None;time.sleep(.07)
        raise AssertionError(log.read_text(encoding='utf-8')[-4000:])
    try:
        user.PostMessageW(hwnd,0x8022,7,5);wait(lambda:row('settings_sample').get('open')=='1' and int(row('language_sample').get('combo','0,0').split(',')[0])>0)
        # Add a third language while running. Opening the dropdown must rescan.
        custom_name='自定义语言.xml';chinese=(loc/'chineseSimplified.xml').read_text(encoding='utf-8')
        custom=chinese.replace('id="action.start" name="开始"','id="action.start" name="执行"')
        assert custom!=chinese;(loc/custom_name).write_text(custom,encoding='utf-8')
        select_language(hwnd,log,custom_name);wait(lambda:row('language_sample').get('start')=='执行')
        assert row('language_sample')['label']=='自定义语言'
        text=log.read_text(encoding='utf-8');assert 'item[options.' not in text
        capture(hwnd,out/'custom-language-selected.png')
        # Reject a malformed format string and keep the active custom dictionary.
        bad=custom.replace('%.1f s','%s s',1);assert bad!=custom;(loc/'bad.xml').write_text(bad,encoding='utf-8')
        select_language(hwnd,log,'bad.xml',accepted=False)
        wait(lambda:'Localization format mismatch:' in log.read_text(encoding='utf-8'))
        assert row('language_sample')['file']==custom_name and row('language_sample')['start']=='执行'
        # Remove and add files while running; reopened list reflects disk contents.
        (loc/'bad.xml').unlink();(loc/'second-custom.xml').write_text(custom,encoding='utf-8')
        select_language(hwnd,log,'second-custom.xml');wait(lambda:row('language_sample').get('loaded')=='second-custom.xml')
        select_language(hwnd,log,'english.xml');wait(lambda:row('language_sample').get('start')=='Start')
        assert row('language_sample')['label']=='english'
        aliases=json.loads((Path(__file__).resolve().parents[3]/'config/gui_v2_localization_aliases.json').read_text(encoding='utf-8'))['aliases'];reverse={value:key for key,value in aliases.items()}
        legacy=re.sub(r'id="([^"]+)"',lambda m:'id="'+reverse.get(m.group(1),m.group(1))+'"',custom)
        assert legacy!=custom;(loc/'legacy.xml').write_text(legacy,encoding='utf-8')
        select_language(hwnd,log,'legacy.xml');wait(lambda:row('language_sample').get('start')=='执行')
        assert row('language_sample')['label']=='legacy'
        capture(hwnd,out/'legacy-keys-loaded.png')
        select_language(hwnd,log,custom_name);wait(lambda:row('language_sample').get('start')=='执行')
        capture(hwnd,out/'filename-dropdown.png')
    except BaseException:
        if proc.poll() is None:capture(hwnd,out/'failure.png')
        raise
    finally:close(proc,hwnd)
    saved=tomllib.loads(state.read_text(encoding='utf-8'));assert saved['ui']['language_file']==custom_name and cfg.read_bytes()==before
    restart=out/'restart.log';r=subprocess.run([str(exe),'--config',str(cfg),'--ui-state',str(state),'--diagnostics',str(restart),'--audit-app','--frames','12','--no-nvml','--no-pet'],cwd=out,startupinfo=startup,timeout=25)
    assert r.returncode==0 and f'file={custom_name} loaded={custom_name}' in restart.read_text(encoding='utf-8') and ' start=执行' in restart.read_text(encoding='utf-8')
    assert renderer_clean(log.read_text(encoding='utf-8'),'dx11')
    report=dict(passed=True,xml_filename_dropdown=True,discovered_while_running=True,custom_dictionary_rendered=True,companion_options_excluded=True,bad_format_preserves_selection=True,add_remove_rescan=True,english_fallback_selection=True,unicode_filename_restart=True,business_unchanged=True,exe_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),scope='Isolated copied runtime and idle worker; no computation or Vulkan rendering')
    (out/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))

if __name__=='__main__':main()
