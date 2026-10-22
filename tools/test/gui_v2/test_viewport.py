"""Minimal acquire-before-use regression for the pinned Vulkan viewport backend."""
from pathlib import Path
import argparse
import json
import os
import subprocess

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--exe',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--layers',type=Path,required=True)
    args=parser.parse_args(); out=args.out.resolve(); out.mkdir()
    env=os.environ.copy(); env['VK_LAYER_PATH']=str(args.layers.resolve())
    startup=subprocess.STARTUPINFO(); startup.dwFlags=subprocess.STARTF_USESHOWWINDOW; startup.wShowWindow=0
    log=out/'viewport.log'
    run=subprocess.run([str(args.exe.resolve()),'--frames','10','--multi-viewport','--detach-probe','--no-nvml','--renderer','vulkan','--validation','--audit-app','--ui-state',str(out/'ui.toml'),'--diagnostics',str(log)],cwd=out,env=env,startupinfo=startup,timeout=20)
    text=log.read_text(encoding='utf-8')
    assert run.returncode==0 and 'validation_summary: errors=0 warnings=0' in text,text
    (out/'result.json').write_text(json.dumps(dict(passed=True,acquire_before_use=True)))
    print('PASS: viewport acquire-before-use')

if __name__=='__main__':main()
