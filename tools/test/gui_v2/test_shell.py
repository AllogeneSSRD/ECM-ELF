"""Actual Vulkan platform-window / Controller / sidecar / pet integration."""
import argparse
import ctypes as c
from ctypes import wintypes as w
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from win32_probe import window, capture, user, callback_type


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--exe',type=Path,required=True)
    parser.add_argument('--worker',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--layers',type=Path,required=True)
    args=parser.parse_args(); out=args.out.resolve(); out.mkdir()
    exe=args.exe.resolve(); worker=args.worker.resolve()
    user.SetThreadDpiAwarenessContext.argtypes=[c.c_void_p]; user.SetThreadDpiAwarenessContext.restype=c.c_void_p
    assert user.SetThreadDpiAwarenessContext(c.c_void_p(-4))
    user.IsWindowVisible.argtypes=[w.HWND]; user.IsWindowVisible.restype=w.BOOL
    user.GetWindow.argtypes=[w.HWND,w.UINT]; user.GetWindow.restype=w.HWND
    user.GetWindowLongPtrW.argtypes=[w.HWND,c.c_int]; user.GetWindowLongPtrW.restype=c.c_ssize_t
    startup=subprocess.STARTUPINFO(); startup.dwFlags=subprocess.STARTF_USESHOWWINDOW; startup.wShowWindow=0
    env=os.environ.copy(); env['VK_LAYER_PATH']=str(args.layers.resolve())
    queue=out/'worktodo.txt'; queue.write_text('[Worker #1]\nECMSTAGE1=1,2,127,-1,100,1\n[Worker #2]\nECMSTAGE1=1,2,131,-1,100,1\n')
    config=out/'ecm.toml'
    config.write_text(f'[shared]\nworktodo={json.dumps(str(queue))}\n[gui]\nNumWorkers=2\nexe={json.dumps(str(worker))}\nexit_confirm="ask"\ngraceful_stop_ms=1200\n[workers.1]\nautostart=0\nextra_args="--scenario v2-checkpoint-stop"\n[workers.2]\nautostart=0\nextra_args="--scenario v2-checkpoint-stop"\n')
    before=hashlib.sha256(config.read_bytes()).hexdigest()
    log=out/'shell.log'; state=out/'ui.toml'
    process=subprocess.Popen([str(exe),'--config',str(config),'--ui-state',str(state),'--diagnostics',str(log),'--no-nvml','--validation','--audit-app','--detach-probe','--pet','--start-worker','1'],cwd=out,env=env,startupinfo=startup)
    hwnd=window(process.pid,'ecm_gui_v2')
    def wait(pattern, timeout=15):
        end=time.monotonic()+timeout
        while time.monotonic()<end:
            text=log.read_text(encoding='utf-8') if log.exists() else ''
            if pattern in text: return text
            if process.poll() is not None: raise RuntimeError(f'Host exited {process.returncode}\n{text}')
            time.sleep(.04)
        raise RuntimeError('Missing '+pattern+'\n'+text)
    def platforms():
        found=[]
        @callback_type
        def visit(handle,_):
            owner=w.DWORD(); user.GetWindowThreadProcessId(handle,c.byref(owner))
            name=c.create_unicode_buffer(256); user.GetClassNameW(handle,name,len(name))
            if owner.value==process.pid and name.value=='ImGui Platform': found.append(handle)
            return True
        user.EnumWindows(visit,0); return found
    try:
        wait('started pid=')
        user.ShowWindow(hwnd,4)
        pet=window(process.pid,'ecm_gui_v2_pet')
        assert user.GetWindow(pet,4)==hwnd
        exstyle=user.GetWindowLongPtrW(pet,-20)
        assert exstyle & 0x80000 and exstyle & 0x8000000 and not exstyle & 8
        end=time.monotonic()+15
        while len(platforms())<1 and time.monotonic()<end: time.sleep(.05)
        floating=platforms(); assert floating
        monitors=[]
        monitor_callback=c.WINFUNCTYPE(w.BOOL,w.HMONITOR,w.HDC,c.POINTER(w.RECT),w.LPARAM)
        @monitor_callback
        def monitor(handle,dc,rect,data):
            r=rect.contents; monitors.append([r.left,r.top,r.right,r.bottom]); return True
        assert user.EnumDisplayMonitors(None,None,monitor,0)
        user.GetDpiForWindow.argtypes=[w.HWND]; user.GetDpiForWindow.restype=w.UINT
        monitor_dpi=[]
        for left,top,right,bottom in monitors:
            assert user.SetWindowPos(floating[0],None,left+60,top+80,500,260,0x0014)
            time.sleep(.4); monitor_dpi.append(user.GetDpiForWindow(floating[0]))
            assert set(floating).issubset(platforms())
        # The pet's mouse-activation contract is independent of desktop focus.
        user.SendMessageW.restype=c.c_ssize_t
        assert user.SendMessageW(pet,0x21,hwnd,0x2010001)==3  # MA_NOACTIVATE
        # Resizing the main swapchain must retain a live secondary viewport.
        for width,height in [(1280,900),(900,700),(1500,1000)]:
            assert user.SetWindowPos(hwnd,None,80,350,width,height,0x0014); time.sleep(.35)
            assert set(floating).issubset(platforms())
        capture(hwnd,out/'workspace.png'); capture(pet,out/'pet.png')
        user.ShowWindow(hwnd,6); time.sleep(.3); assert not user.IsWindowVisible(pet)
        user.ShowWindow(hwnd,4); time.sleep(.5); assert user.IsWindowVisible(pet)
        for mood in range(7):
            user.PostMessageW(hwnd,0x8022,8,mood); wait(f'pet: mood={mood} ')
        # Child viewport can be closed and recreated independently.
        user.PostMessageW(hwnd,0x8022,6,0); time.sleep(.4); assert not platforms()
        user.PostMessageW(hwnd,0x8022,6,1); time.sleep(.5); assert platforms()
        # Direct consumer (the same entry contract used by an unchanged old GUI) is refused.
        contender=subprocess.run([str(worker),'--config',str(config),'--worker','1','--scenario','ok'],cwd=out,capture_output=True,startupinfo=startup,timeout=5)
        assert contender.returncode==23 and b'already owned' in contender.stderr
        user.PostMessageW(hwnd,0x10,0,0); wait('close=1')
        assert process.poll() is None
        user.PostMessageW(hwnd,0x8022,5,0); time.sleep(.3); assert process.poll() is None
        user.PostMessageW(hwnd,0x8022,3,0)
        process.wait(timeout=10); assert process.returncode==0
    finally:
        if process.poll() is None: user.PostMessageW(hwnd,0x8022,3,0); process.wait(timeout=10)
    text=log.read_text(encoding='utf-8')
    assert 'validation_summary: errors=0 warnings=0' in text,text
    assert 'worker_processes=0' in text and state.exists()
    assert 'graceful stop requested' in text and 'state Restarting (exit code 5' not in text,text
    assert hashlib.sha256(config.read_bytes()).hexdigest()==before
    assert '[pet]' in state.read_text() and 'enabled = true' in state.read_text()
    # Restart restores a separate sidecar while keeping business config byte-identical.
    restart=out/'restart.log'
    run=subprocess.run([str(exe),'--config',str(config),'--ui-state',str(state),'--diagnostics',str(restart),'--frames','10','--no-nvml','--validation'],cwd=out,env=env,startupinfo=startup,timeout=20)
    assert run.returncode==0 and 'pet: mood=' not in restart.read_text(encoding='utf-8')  # hidden host defers pet work
    assert 'UI state ignored' not in restart.read_text(encoding='utf-8') and 'duplicate UI table' not in restart.read_text(encoding='utf-8')
    assert hashlib.sha256(config.read_bytes()).hexdigest()==before
    result=dict(passed=True,exe_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(),business_unchanged=True,viewport_survives_main_resize=True,
                viewport_close_recreate=True,pet_owned_layered_noactivate=True,pet_minimize_restore=True,pet_moods=7,
                duplicate_consumer_refused=True,close_confirm_cancel_and_graceful=True,sidecar_restart=True,
                physical_monitors=monitors,viewport_monitor_dpi=monitor_dpi,pet_mouse_noactivate=True,
                limits=['Physical differing-DPI and real IME composition require separate qualification','Direct consumer exercises shared entry contract; full old-GUI startup races are covered in P5'])
    (out/'result.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result))


if __name__=='__main__': main()
