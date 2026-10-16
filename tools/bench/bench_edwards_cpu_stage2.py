"""Alternate libecm NTT/plain on identical completed Edwards saves; no projections."""
import argparse
import ctypes
from ctypes import wintypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('gates', ROOT/'tools/test/test_edwards_cpu_pipeline.py')
gates = importlib.util.module_from_spec(spec);spec.loader.exec_module(gates)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage2', type=Path, required=True)
    p.add_argument('--saves', type=Path, nargs='+', required=True)
    p.add_argument('--b2', type=int, nargs='+', default=[10**6, 10**8])
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--threads', type=int, default=1)
    args = p.parse_args()
    out = args.output.resolve();out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):raise ValueError('use a fresh evidence directory')
    sha = lambda f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
    exe = args.stage2.resolve()
    report = dict(exe_sha256=sha(exe), script_sha256=sha(__file__), platform=platform.platform(),
                  processor=os.environ.get('PROCESSOR_IDENTIFIER'), cpu_power='not measured',
                  threads=args.threads, memory_mib=512, warmups_per_cell=1, repeats=args.repeats,
                  timers={'engine': 'curve setup plus full libecm call; excludes JSON publication',
                          'process': 'process launch through exit including save read and JSON publication'},
                  rows=[], complete=False)
    def persist():(out/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD)]+[(n, ctypes.c_size_t) for n in
                    ('peak_ws', 'ws', 'peak_pool_paged', 'pool_paged', 'peak_pool_nonpaged', 'pool_nonpaged', 'pagefile', 'peak_pagefile')]
    memory = ctypes.WinDLL('psapi').GetProcessMemoryInfo
    memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    for save in args.saves:
        save = save.resolve();saved = gates.oracle.records(save)[0]
        assert saved[-1] == 'POINT'
        bits = saved[0].bit_length()
        for b2 in args.b2:
            samples = {b: [] for b in ('ntt', 'plain')}
            for repeat in range(-1, args.repeats):
                for backend in (('ntt', 'plain') if repeat % 2 else ('plain', 'ntt')):
                    tag = f'm{bits}_b{b2}_{backend}_{repeat+1}'
                    result = out/(tag+'.jsonl');start = time.perf_counter()
                    with (out/(tag+'.log')).open('wb') as log:
                        proc = subprocess.Popen(list(map(str, [exe, '--save', save, '--b2', b2,
                            '--curves', 1, '--backend', backend, '--threads', args.threads, '--results', result])),
                            stdout=log, stderr=subprocess.STDOUT, cwd=out)
                        rc = proc.wait(timeout=1800)
                        duration = time.perf_counter()-start
                        counters = Counters();counters.cb = ctypes.sizeof(counters)
                        assert memory(proc._handle, ctypes.byref(counters), counters.cb)
                        peak = counters.peak_ws
                    assert rc == 0, tag
                    r = json.loads(result.read_text());gates.verify_json(r)
                    assert r['status'] == 'NF' and r['stage2-engine']['backend'] == backend
                    samples[backend].append(dict(warmup=repeat == -1, process_seconds=duration,
                        engine_seconds=r['stage2-engine']['seconds'], peak_working_set_bytes=peak,
                        log=tag+'.log', result_sha256=sha(result)))
                    print(tag, round(duration, 4), flush=True)
            for backend in ('ntt', 'plain'):
                formal = [r for r in samples[backend] if not r['warmup']]
                report['rows'].append(dict(bits=bits, n=str(saved[0]), b1=saved[3], sigma=saved[2], b2=b2,
                    save_sha256=sha(save), backend=backend, samples=samples[backend],
                    engine_median=statistics.median(r['engine_seconds'] for r in formal),
                    process_median=statistics.median(r['process_seconds'] for r in formal)))
            persist()
    report['complete'] = True;persist()


if __name__ == '__main__':main()
