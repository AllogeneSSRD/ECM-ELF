"""Interleaved Stage1 process costs with real save/result history and handoffs.

Consumes timing seeds from test_stage1_inputs.py. Repeats intact
historical records; new work uses different sigmas. IO counters measure process
API transfer bytes, not physical disk traffic. Run alone for performance data.
"""
import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import threading
import time


def run(exe, case):
    start = time.perf_counter()
    proc = subprocess.Popen([str(exe), '--config', str(case / 'ecm.toml')], cwd=case,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding='utf-8', errors='replace')
    kernel = ctypes.windll.kernel32
    kernel.OpenProcess.restype = ctypes.c_void_p
    handle = kernel.OpenProcess(0x410, False, proc.pid)
    assert handle, 'cannot measure process'
    class Memory(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_uint32), ('faults', ctypes.c_uint32)] + [
            (name, ctypes.c_size_t) for name in ('peak_ws', 'ws', 'peak_paged', 'paged',
                                               'peak_nonpaged', 'nonpaged', 'pagefile', 'peak_pagefile')]
    class IO(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in
                    ('read_ops', 'write_ops', 'other_ops', 'read_bytes', 'write_bytes', 'other_bytes')]
    peak = [0]
    stopped = threading.Event()
    def sample():
        while not stopped.is_set():
            memory = Memory(); memory.cb = ctypes.sizeof(memory)
            if ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.c_void_p(handle), ctypes.byref(memory), memory.cb):
                peak[0] = max(peak[0], memory.peak_ws)
            stopped.wait(.02)
    monitor = threading.Thread(target=sample, daemon=True); monitor.start()
    timer = threading.Timer(120, proc.kill); timer.start()
    events = {}; lines = []
    try:
        for line in proc.stdout:
            lines.append(line); now = time.perf_counter() - start
            if ('Saved ' in line and 'curve line(s)' in line) or 'Stage1 save:' in line:
                events['save'] = now
            if 'Stage1 results:' in line: events['result'] = now
            if 'queue done' in line: events['queue_done'] = now
        code = proc.wait(timeout=10)
        io = IO()
        assert kernel.GetProcessIoCounters(ctypes.c_void_p(handle), ctypes.byref(io))
        times = [ctypes.c_uint64() for _ in range(4)]
        assert kernel.GetProcessTimes(ctypes.c_void_p(handle), *[ctypes.byref(t) for t in times])
    finally:
        timer.cancel(); stopped.set(); monitor.join(); kernel.CloseHandle(ctypes.c_void_p(handle))
    (case / 'captured.log').write_text(''.join(lines), encoding='utf-8')
    assert code == 0 and all(key in events for key in ('save', 'result', 'queue_done')), case
    assert not (case / 'worktodo.txt').read_text().strip(), case
    return dict(seconds=time.perf_counter()-start, events=events,
                save_to_result=events['result']-events['save'],
                result_to_queue_done=events['queue_done']-events['result'],
                process_cpu_seconds=(times[2].value+times[3].value)/10000000,
                peak_working_set_bytes=peak[0], io={name:getattr(io, name) for name, _ in IO._fields_})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('before', 'after', 'runtime', 'seeds', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--samples', type=int, default=2)
    parser.add_argument('--history-repeats', type=int, default=40)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--bits', type=int, choices=(521, 10267), default=521)
    parser.add_argument('--only-history', action='store_true')
    args = parser.parse_args(); out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    binaries = {}; identities = {}
    for label in ('before', 'after'):
        folder = out / label; folder.mkdir()
        source = getattr(args, label).resolve(); binaries[label] = folder / 'ecm_cuda.exe'
        shutil.copy2(source, binaries[label]); shutil.copy2(args.runtime, folder / 'gmp-10.dll')
        identities[label] = hashlib.sha256(source.read_bytes()).hexdigest()
    report = dict(complete=False, identities=identities, b1=100, bits=args.bits, curves=1000,
                  history_repeats=args.history_repeats, device=args.device, samples=[])
    for curve in ('mont', 'ed'):
        seed = args.seeds / f'timing_current_{curve}_{args.bits}_0'
        source_save = next(seed.glob('*.save'))
        histories = (True,) if args.only_history else ((False, True) if args.history_repeats else (False,))
        for historical in histories:
            for index in range(args.samples):
                for label in (('before', 'after') if index % 2 == 0 else ('after', 'before')):
                    name = f'{curve}_{"history" if historical else "fresh"}_{index}_{label}'
                    case = out / name; case.mkdir(); copies = args.history_repeats if historical else 0
                    for filename in (source_save.name, 'stage1_results.jsonl', 'results.json.txt'):
                        (case / filename).write_bytes((seed / filename).read_bytes() * copies)
                    before_bytes = sum((case / filename).stat().st_size for filename in
                                       (source_save.name, 'stage1_results.jsonl', 'results.json.txt'))
                    (case / 'worktodo.txt').write_text(f'ECM=1,2,{args.bits},-1,100,10000,1000,200000\n')
                    quote = lambda path: json.dumps(str(path))
                    config = ('[shared]\nuser=""\ncomputer=""\nworktodo='+quote(case/'worktodo.txt')+
                              '\n[stage1]\ncurve='+quote(curve)+'\ntarget="gpu"\ngpu_param=0\ndevice='+str(args.device)+
                              '\nexp_cache="off"\nckpt_seconds=0\ntmp_dir='+quote(case)+
                              '\nlog_file='+quote(case/'program.log')+'\nfinished='+quote(case/'finished.txt')+
                              '\nsave_sync_dir_1='+quote(case/'backup')+'\n')
                    if curve == 'mont':
                        config += ('p95_worktodo_path='+quote(case/'prime95/worktodo.txt')+
                                   '\np95_sync_dir='+quote(case/'prime95/saves')+'\n')
                    (case / 'ecm.toml').write_text(config, encoding='utf-8')
                    result = run(binaries[label], case)
                    assert sum(bool(line.strip()) for line in (case/'stage1_results.jsonl').read_text().splitlines()) == 1000*(copies+1)
                    assert (case/'backup'/source_save.name).is_file()
                    if curve == 'mont':
                        assert (case/'prime95/worktodo.add').is_file()
                        assert (case/'prime95/saves'/source_save.name).is_file()
                    result.update(name=name, curve=curve, historical=historical, sample=index,
                                  binary=label, initial_history_bytes=before_bytes)
                    report['samples'].append(result)
                    (out/'result.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
                    print(name, json.dumps(result), flush=True)
    report['complete'] = True
    (out/'result.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
