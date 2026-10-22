"""Regress nested vertical table scrolling and mismatched log/body font metrics."""
import argparse,ctypes as c,hashlib,json,re,subprocess,time
from pathlib import Path
from win32_probe import window,close,capture,user
from renderer_probe import renderer_clean

def main():
    p=argparse.ArgumentParser()
    for key in ['exe','out']:p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();out=a.out.resolve();out.mkdir();exe=a.exe.resolve()
    user.SetThreadDpiAwarenessContext.argtypes=[c.c_void_p];user.SetThreadDpiAwarenessContext.restype=c.c_void_p;assert user.SetThreadDpiAwarenessContext(c.c_void_p(-4))
    rows=[]
    for name,font,scale,width,height,language in [('wide',15,1,1920,1200,'zh'),('narrow',20,1.25,1120,850,'zh'),('large',30,2,1920,1400,'en')]:
        folder=out/name;folder.mkdir();cfg=folder/'ecm.toml';queue=folder/'worktodo.txt';queue.write_text('')
        cfg.write_text('[shared]\nworktodo="worktodo.txt"\n[gui]\nNumWorkers=6\n'+''.join(f'[workers.{i}]\nautostart=0\n' for i in range(1,7)))
        original=cfg.read_bytes();state=folder/'ui.toml';state.write_text(f'[ui]\npage=0\nlanguage="{language}"\nfont_size={font}\nscale={scale}\nlogs=true\ninspector=false\ntoasts=false\n')
        log=folder/'ui.log';startup=subprocess.STARTUPINFO();startup.dwFlags=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
        proc=subprocess.Popen([str(exe),'--config',str(cfg),'--ui-state',str(state),'--diagnostics',str(log),'--audit-app','--no-nvml','--no-pet'],cwd=folder,startupinfo=startup)
        hwnd=window(proc.pid,'ecm_gui_v2');user.SetWindowPos(hwnd,None,40,40,width,height,0x14);user.ShowWindow(hwnd,4)
        try:
            end=time.monotonic()+15
            while time.monotonic()<end:
                text=log.read_text(encoding='utf-8');samples=[dict(re.findall(r'(\w+)=([-\d.,]+)',line)) for line in text.splitlines() if line.startswith('presentation_sample:')]
                if len(samples)>=15 and float(samples[-1].get('body_font','0'))>0 and float(samples[-1]['font_height'])>0:break
                assert proc.poll() is None,text[-1500:];time.sleep(.1)
            else:raise AssertionError(text[-2500:])
            s=samples[-1];capture(hwnd,folder/'layout.png')
            metrics={key:float(s[key]) for key in ['table_scroll_y','table_height','table_content','body_font','font_height','body_glyph','log_glyph','log_body']}
            full_height=metrics['table_scroll_y']<=.5
            if 'table_required' in s:
                metrics['table_required']=float(s['table_required']);metrics['table_data_height']=float(s['table_data_height'])
                full_height=full_height and s['table_rows']=='6' and metrics['table_data_height']+.5>=metrics['table_required']
            matched=metrics['log_body']==1 and metrics['body_font']==metrics['font_height'] and metrics['body_glyph']==metrics['log_glyph']
            rows.append(dict(name=name,font=font,scale=scale,width=width,height=height,language=language,metrics=metrics,full_table_height=full_height,matched_typography=matched))
        finally:close(proc,hwnd)
        assert cfg.read_bytes()==original and renderer_clean(log.read_text(encoding='utf-8'),'dx11')
    passed=all(row['full_table_height'] and row['matched_typography'] for row in rows)
    result=dict(passed=passed,cases=rows,exe_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),business_unchanged=True,scope='Six idle workers; no computation or Vulkan rendering')
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result));assert passed,result

if __name__=='__main__':main()
