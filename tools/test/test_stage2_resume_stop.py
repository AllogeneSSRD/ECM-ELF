"""Validate queue resume and stop after one current Stage2 curve on Windows.

Uses the supplied queue/INI as-is, so successful results advance its curve
counter. Sends one Ctrl+C in a private hidden console after baby generation
starts; does not change GPU settings or interrupt another console.
"""
import argparse
import ctypes
from ctypes import wintypes
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"ecm_report"))
from result_format import normalize_result
import re
import subprocess
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stop_current_console(pid):
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.AttachConsole.argtypes=[wintypes.DWORD]
    kernel.AttachConsole.restype=wintypes.BOOL
    kernel.SetConsoleCtrlHandler.argtypes=[ctypes.c_void_p,wintypes.BOOL]
    kernel.SetConsoleCtrlHandler.restype=wintypes.BOOL
    kernel.GenerateConsoleCtrlEvent.argtypes=[wintypes.DWORD,wintypes.DWORD]
    kernel.GenerateConsoleCtrlEvent.restype=wintypes.BOOL
    kernel.FreeConsole()
    if not kernel.AttachConsole(pid):raise ctypes.WinError(ctypes.get_last_error())
    try:
        if not kernel.SetConsoleCtrlHandler(None,True):raise ctypes.WinError(ctypes.get_last_error())
        # Broadcast CTRL_C only in this private console. Curve workers ignore it;
        # the parent commits the current result and stops before the next curve.
        if not kernel.GenerateConsoleCtrlEvent(0,0):raise ctypes.WinError(ctypes.get_last_error())
        time.sleep(0.2)
    finally:
        kernel.FreeConsole()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe',type=Path,required=True)
    parser.add_argument('--ini',type=Path,required=True)
    parser.add_argument('--worktodo',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--record',type=int,required=True)
    parser.add_argument('--done',type=int,required=True)
    parser.add_argument('--total',type=int,required=True)
    parser.add_argument('--carrier',type=int,default=12011)
    parser.add_argument('--analyze-only',action='store_true')
    args=parser.parse_args();out=args.output.resolve()
    if not args.analyze_only:out.mkdir(parents=True,exist_ok=False)
    exe=args.exe.resolve();ini=args.ini.resolve();queue=args.worktodo.resolve()
    command=[str(exe),'--ini',str(ini),'--once','--short-calibration','0']
    before={'exe_sha256':sha(exe),'ini_sha256':sha(ini),'queue_sha256':sha(queue),'command':command}
    if args.analyze_only:before=normalize_result(json.loads((out/'identity.json').read_text(encoding='utf-8')))
    else:(out/'identity.json').write_text(json.dumps(before,indent=2),encoding='utf-8')
    log=out/'console.log'
    startup=subprocess.STARTUPINFO();startup.dwFlags|=subprocess.STARTF_USESHOWWINDOW;startup.wShowWindow=0
    def run():
        with log.open('wb') as stream:
            proc=subprocess.Popen(command,cwd=ini.parent,stdout=stream,stderr=subprocess.STDOUT,
                                  creationflags=subprocess.CREATE_NEW_CONSOLE,startupinfo=startup)
            stopped=False;begin=time.monotonic()
            try:
                while proc.poll() is None:
                    text=log.read_text(encoding='utf-8',errors='replace')
                    if not stopped and 'stage2_phase: Generate and normalize baby points' in text:
                        stop_current_console(proc.pid);stopped=True
                    if time.monotonic()-begin>180:raise TimeoutError('one-curve resume gate timed out')
                    time.sleep(0.1)
                code=proc.wait()
            finally:
                if proc.poll() is None:
                    proc.kill();proc.wait()
        return {'stopped':stopped,'exit_code':code,'elapsed_seconds':time.monotonic()-begin}
    if not args.analyze_only:
        execution=run()
        (out/'execution.json').write_text(json.dumps(execution,indent=2),encoding='utf-8')
    elif (out/'execution.json').exists():execution=normalize_result(json.loads((out/'execution.json').read_text(encoding='utf-8')))
    else:execution=None
    text=log.read_text(encoding='utf-8',errors='replace')
    if execution:assert execution['stopped'] and execution['exit_code']==0,(execution,text[-2000:])
    else:assert 'Stop requested: finishing current curve;' in text
    starts=re.findall(r'^curve_start: .* record=(\d+) ',text,re.M)
    assert starts==[str(args.record)],starts
    assert text.count('stage2_phase: Plan Stage2 bounds and memory')==1
    assert text.count('stage2_phase: Read and validate Stage1 point')==1
    assert 'stage2_stopped: curves_this_run=1' in text
    paths=re.search(r'^stage2_plan: .* log=(.+?) results=(.+?) checkpoint=',text,re.M)
    assert paths,'missing engine/result paths'
    engine=Path(paths[1]).read_text(encoding='utf-8',errors='replace')
    engine=engine[engine.rfind('stage2_phase: Read and validate Stage1 point'):]
    choices=[normalize_result(json.loads(s)) for s in engine.splitlines() if s.startswith('{"type":"tune_selection"')]
    assert len(choices)==1 and choices[0]['selected'] and choices[0]['carrier_exponent']==args.carrier,choices
    receipts=[normalize_result(json.loads(s)) for s in Path(paths[2]).read_text(encoding='utf-8').splitlines()]
    receipt=receipts[-1]
    assert receipt['record']==args.record and receipt['status']=='stage2_completed'
    assert receipt['bad_factors']==0 and receipt['carrier_exponent']==args.carrier
    assert re.search(r'^stage2_full_wall: .* clean=1 ',engine,re.M),engine[-2000:]
    checks=re.search(r'^s4_multiply_stats: (.+)$',engine,re.M)[1]
    assert int(re.search(r'gmp_selftest_cases=(\d+)',checks)[1])>0
    assert int(re.search(r'gmp_checked=(\d+)',checks)[1])>0
    assert 'gmp_selftest_bad=0' in checks and 'gmp_check_bad=0' in checks
    checkpoint=Path(re.search(r'^stage2_plan: .* checkpoint=(.+)$',text,re.M)[1].strip())
    state=checkpoint.read_text(encoding='utf-8')
    assert re.match(rf'^2 {args.done} {args.total} "[^"]+" "" "[0-9a-f]{{64}}" ""\s*$',state),state
    assert sha(queue)==before['queue_sha256'] and sha(ini)==before['ini_sha256']
    result={'passed':True,**before,'record':args.record,'done':args.done,'total':args.total,
            'checkpoint':str(checkpoint),'checkpoint_bytes':checkpoint.stat().st_size,
            'carrier_exponent':args.carrier,'curves_this_run':1,'plan_phase_lines':1,
            'execution':execution,'receipt':receipt}
    (out/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('receipt','command')},indent=2))


if __name__=='__main__':main()
