"""Execute frozen compile command files, with bounded process concurrency."""
import argparse
import concurrent.futures
import datetime
import json
import re
import subprocess
import time
from pathlib import Path


def run(job):
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    timer = time.perf_counter()
    with open(job['log'], 'wb') as log:
        # NVCC's host-compiler probe can crash with a regular-file stdout
        # handle in restricted Windows environments. Drain a pipe into the log
        # instead; preserve bounded memory, complete diagnostics and exit code.
        result = subprocess.Popen(['cmd.exe', '/d', '/c', job['command_file']], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  creationflags=subprocess.CREATE_NO_WINDOW)
        with result.stdout:
            for chunk in iter(lambda: result.stdout.read(65536), b''):log.write(chunk)
        result.wait()
    if job.get('msvc_includes') and result.returncode == 0:
        # /showIncludes is the compiler's dependency output on MSVC 14.16,
        # which predates /sourceDependencies. Keep SDK and macro includes too.
        text = Path(job['log']).read_text(encoding='mbcs', errors='replace')
        includes = []
        for line in text.splitlines():
            match = re.search(r':\s+([A-Za-z]:[\\/].+?)\s*$', line)
            if match and Path(match[1]).is_file():
                includes.append(str(Path(match[1]).resolve()))
        if not includes:
            with open(job['log'],'ab') as log:
                log.write(b'\nECM: successful MSVC compile produced no dependency includes\n')
            return dict(source=job['source'],started_utc=started,elapsed_seconds=time.perf_counter()-timer,
                        exit_code=126,log=job['log'])
        Path(job['msvc_includes']).write_text(json.dumps(dict(Version='1.2',Data=dict(Source=str(Path(job['source']).resolve()),Includes=sorted(set(includes))))), encoding='utf-8')
    return dict(source=job['source'], started_utc=started, elapsed_seconds=time.perf_counter()-timer,
                exit_code=result.returncode, log=job['log'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--jobs', type=int, required=True)
    args = parser.parse_args()
    if not 1 <= args.jobs <= 64:
        parser.error('jobs must be between 1 and 64')
    plan = json.loads(args.plan.read_text(encoding='utf-8-sig'))
    timer = time.perf_counter()
    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for record in pool.map(run, plan):
            records.append(record)
            print(f'{"FAIL" if record["exit_code"] else "ok"} {record["source"]}: {record["elapsed_seconds"]:.2f}s', flush=True)
    args.out.write_text(json.dumps(dict(wall_seconds=time.perf_counter()-timer, units=records), indent=2), encoding='utf-8')
    raise SystemExit(any(r['exit_code'] for r in records))
