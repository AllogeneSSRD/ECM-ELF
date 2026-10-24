"""NVCC launcher for native CMake and independently maintained build probes."""
import argparse
import datetime
import json
import subprocess
import time
import uuid
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args();command=args.command
    if command and command[0]=='--':command=command[1:]
    if not command:parser.error('missing nvcc command')
    if any(v in ('--time','-time') or v.startswith(('--time=','-time=')) for v in command):
        return subprocess.call(command)
    folder=args.root.resolve()/'timing'/uuid.uuid4().hex;folder.mkdir(parents=True)
    csv=folder/'nvcc.csv';actual=[*command,'--time',str(csv)]
    record=dict(schema=1,semantic_command=command,command=actual,started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    start=time.perf_counter()
    try:code=subprocess.call(actual)
    except OSError as error:
        code=127;record['error']=str(error)
    present=csv.is_file() and csv.stat().st_size>0
    record.update(elapsed_seconds=time.perf_counter()-start,exit_code=code,csv=str(csv),csv_present=present,
                  status='failed' if code else 'complete' if present else 'timing_missing')
    (folder/'command.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    if code==0 and not present:
        print(f'NVCC phase timing missing: {folder}',flush=True);return 125
    return code


if __name__=='__main__':raise SystemExit(main())
