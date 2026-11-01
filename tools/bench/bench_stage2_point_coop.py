"""Frozen serial scalar/CGBN field, chain or ladder experiment with GMP gates."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

from bench_stage2_version import check_gpu_processes, stamp, write
from gpu_telemetry import TelemetryRecorder


def integers(text):
    return [int(v) for v in text.split(',')]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', type=int, default=1)
    p.add_argument('--allow-gpu-pid', type=int, action='append', default=[])
    p.add_argument('--suite', choices=('correctness', 'performance'), required=True)
    p.add_argument('--probe', choices=('field', 'chain', 'ladder'), default='field')
    p.add_argument('--bits', default='4096,8192,16381')
    p.add_argument('--tpis', default='0,16', help='0: production scalar; match the probe build TPI')
    p.add_argument('--counts', default='24,4096')
    p.add_argument('--kinds', default='0,1')
    p.add_argument('--operations', default='0,1', help='Field: 0 multiply / 1 square; ladder: 0 internal / 1 normal output')
    p.add_argument('--lengths', default='32,64', help='Complete point-chain lengths')
    p.add_argument('--scalar-threads', type=int, default=32, help='Production wide chain reference block size')
    p.add_argument('--tails', default='0', help='0: full last chain; otherwise its point count')
    p.add_argument('--seed-strides', default='1', help='1: separate seed arrays; 2: interleaved production layout')
    p.add_argument('--correct-count', type=int, choices=(1,37), default=37)
    p.add_argument('--rounds', type=int, default=1)
    p.add_argument('--recurrence', type=int, default=4)
    p.add_argument('--scalar-bits',type=int,default=18,help='Ladder performance scalars; correctness uses 0 and 64-bit boundaries')
    p.add_argument('--threads', type=int, default=64)
    p.add_argument('--exploratory', action='store_true', help='Exclude from qualification, e.g. while CPU builds run')
    a = p.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    binary = out / 'bin'
    binary.mkdir()
    (binary / '.gitignore').write_text('*\n')
    for f in (a.exe.resolve(), a.exe.resolve().parent / 'gmp-10.dll', a.exe.resolve().parent / 'manifest.json'):
        shutil.copy2(f, binary / f.name)
    shutil.copy2(__file__, out / 'collector_snapshot.py')
    exe = binary / a.exe.name
    allowed = []
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name',
                                         '--format=csv,noheader'], text=True)
    for pid in a.allow_gpu_pid:
        names = {line.split(',', 2)[2].strip() for line in processes.splitlines()
                 if line.split(',')[1].strip() == str(pid)}
        if len(names) != 1 or not Path(next(iter(names))).is_absolute():
            raise ValueError('Cannot bind allowed process')
        allowed.append(dict(pid=pid, process_name=next(iter(names))))
    sha = lambda f: hashlib.sha256(f.read_bytes()).hexdigest()
    report = dict(complete=False, started_utc=stamp(), exploratory=a.exploratory, exe_sha256=sha(exe),
                  dll_sha256=sha(binary / 'gmp-10.dll'), conditions={
                      k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
                  allowed_processes=allowed, rows=[], failures=[])
    write(out / 'results.json', report)
    telemetry = TelemetryRecorder(a.device, out / 'telemetry.jsonl', interval=.2)
    report['device'] = telemetry.identity
    env = {k: v for k, v in os.environ.items() if not k.startswith(('NTT_', 'ECM_'))
           and k not in ('CUDA_VISIBLE_DEVICES', 'CUDA_LAUNCH_BLOCKING')}
    signatures = {}

    def call(bits, count, repeats, tpi, operation, alias, kind, fault=0, tail=0, stride=1):
        directory = out / f'call_{len(report["rows"]):04d}'
        directory.mkdir()
        if a.probe == 'field':
            values = (bits, count, repeats, tpi, operation, alias, a.threads, a.device, kind, fault, 5)
        elif a.probe == 'ladder':
            values=(bits,count,tpi,repeats,kind,operation,5 if a.suite=='performance' else 1,a.device,fault)
        else:
            values = (bits, count, repeats, tpi, a.threads if tpi else a.scalar_threads, a.device, kind, 5)
            tail = tail or repeats
            if tail != repeats or stride != 1:
                values = (*values, tail, stride)
        command = [str(exe), *map(str, values)]
        write(directory / 'command.json', command)
        check_gpu_processes(telemetry.monitor, out, directory / 'before.json', allowed=allowed)
        started = time.perf_counter()
        result = subprocess.run(command, env=env, capture_output=True, timeout=120)
        text = (result.stdout + result.stderr).decode('utf-8', errors='replace')
        (directory / 'console.log').write_text(text, encoding='utf-8')
        check_gpu_processes(telemetry.monitor, out, directory / 'after.json', allowed=allowed)
        fields = dict(re.findall(r'(\w+)=([^\s]+)', text))
        row = dict(command=command, returncode=result.returncode, fault=fault, fields=fields,
                   wall_seconds=time.perf_counter()-started, directory=str(directory))
        report['rows'].append(row)
        if result.returncode != int(bool(fault)) or 'bad' not in fields or bool(int(fields['bad'])) != bool(fault):
            raise RuntimeError('Arithmetic/process failure: ' + str(directory))
        if a.probe in ('chain','ladder') and not fault:
            if a.suite == 'correctness' and int(fields['checked_words']) != int(fields['output_words']):
                raise ValueError('Incomplete GMP point coverage')
            key = (bits,count,repeats,kind,operation) if a.probe=='ladder' else (bits, count, repeats, kind, tail, stride)
            if key in signatures and signatures[key] != fields['sha256']:
                raise ValueError('Complete point outputs differ from scalar control')
            signatures[key] = fields['sha256']
        write(out / 'results.json', report)
        if len(report['rows']) % 8 == 0 or fault:
            print(f'point_coop_progress: calls={len(report["rows"])} bits={bits} count={count} '
                  f'tpi={tpi} operation={operation} kind={kind} ms={fields["median_ms"]}', flush=True)

    try:
        if a.probe=='ladder' and a.suite=='correctness':
            for bits in integers(a.bits):
                for kind in integers(a.kinds):
                    for normal in (0,1):
                        for tpi in integers(a.tpis):call(bits,37,0,tpi,normal,0,kind)
            call(integers(a.bits)[0],1,0,integers(a.tpis)[-1],0,0,1,fault=1)
        elif a.probe == 'chain' and a.suite == 'correctness':
            for bits in integers(a.bits):
                for kind in integers(a.kinds):
                    for length in integers(a.lengths):
                        for tail in integers(a.tails):
                            for stride in integers(a.seed_strides):
                                for tpi in integers(a.tpis):
                                    call(bits, a.correct_count, length, tpi, 0, 0, kind, tail=tail, stride=stride)
        elif a.suite == 'correctness':
            for bits in integers(a.bits):
                for tpi in integers(a.tpis):
                    for kind in integers(a.kinds):
                        for operation in integers(a.operations):
                            for alias in (0, 1):
                                for recurrence in (1, 3):
                                    call(bits, 37, recurrence, tpi, operation, alias, kind)
            for tpi in integers(a.tpis):
                call(integers(a.bits)[0], 1, 1, tpi, 1, 1, 1, fault=1)
        else:
            for round in range(a.rounds):
                tpis = integers(a.tpis)
                if round % 2:
                    tpis.reverse()
                for bits in integers(a.bits):
                    for count in integers(a.counts):
                        for kind in integers(a.kinds):
                            for operation in integers(a.operations if a.probe in ('field','ladder') else a.lengths):
                                for stride in integers(a.seed_strides if a.probe=='chain' else '1'):
                                    for tpi in tpis:
                                        call(bits, count, a.scalar_bits if a.probe=='ladder' else a.recurrence if a.probe == 'field' else operation,
                                             tpi, operation, 1, kind, stride=stride)
        report['complete'] = True
        report['finished_utc'] = stamp()
    except Exception as error:
        report['failures'].append(dict(utc=stamp(), error=str(error)))
        raise
    finally:
        telemetry.close()
        write(out / 'results.json', report)


if __name__ == '__main__':
    main()
