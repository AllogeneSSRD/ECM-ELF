"""Administrator NCU capture of one retained, verified point-chain invocation.

Reference is a completed bench_stage2_point_coop.py chain call directory.
The capture is diagnostic only and never enters the performance summary.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

from bench_stage2_version import check_gpu_processes, write
from gpu_telemetry import TelemetryRecorder
from profile_stage2_graph import safe

NCU = Path('C:/Program Files/NVIDIA Corporation/Nsight Compute 2026.2.1/target/windows-desktop-win7-x64/ncu.exe')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--allow-gpu-pid', type=int, action='append', default=[])
    p.add_argument('--run-admin', action='store_true')
    p.add_argument('--collect-only', action='store_true')
    a = p.parse_args()
    out = a.output.resolve()
    if a.collect_only:
        meta = json.loads((out / 'command.json').read_text())
    else:
        out.mkdir(parents=True, exist_ok=False)
        reference = a.reference.resolve()
        command = json.loads((reference / 'command.json').read_text())
        fields = dict(re.findall(r'(\w+)=([^\s]+)', (reference / 'console.log').read_text()))
        if 'chains' not in fields or int(fields['bad']):
            raise ValueError('Need a verified completed point-chain reference')
        binary = out / 'bin'
        binary.mkdir()
        (binary / '.gitignore').write_text('*\n')
        exe = Path(command[0])
        for source in (exe, exe.parent / 'gmp-10.dll', exe.parent / 'manifest.json'):
            shutil.copy2(source, binary / source.name)
        command[0] = str(binary / exe.name)
        tpi = int(fields['tpi'])
        kernel = 'regex:stage2_coop_chain' if tpi else 'regex:s2g_chain_kernel'
        profile = [str(NCU), '--devices', '1', '--target-processes', 'all', '--clock-control', 'none',
                   '--cache-control', 'none', '--kernel-name', kernel, '--launch-skip', '1', '--launch-count', '1']
        for section in ('LaunchStats', 'Occupancy', 'SchedulerStats', 'WarpStateStats', 'SpeedOfLight'):
            profile += ['--section', section]
        profile += ['--metrics', 'l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum',
                    '--export', str(out / 'trace'), *command]
        batch = out / 'profile.cmd'
        settings = 'for /f "tokens=1 delims==" %%v in (\'set NTT_ 2^>nul\') do set "%%v="\n'
        batch.write_text('@echo off\n' + settings + safe(profile) + ' > "' + str(out / 'run.log') +
                         '" 2>&1\nexit /b %errorlevel%\n', encoding='utf-8')
        admin = "$pointProfile = Start-Process -FilePath 'C:/Windows/System32/cmd.exe' "
        admin += "-ArgumentList @('/d','/c','\"" + str(batch) + "\"') -Verb RunAs -WindowStyle Hidden -PassThru\n"
        admin += "$pointProfile.WaitForExit()\n$pointProfile.ExitCode | Set-Content -LiteralPath '" + str(out / 'exit.txt') + "'\nexit $pointProfile.ExitCode\n"
        (out / 'admin.ps1').write_text(admin, encoding='utf-8')
        meta = dict(reference=str(reference), command=profile, app=command, fields=fields, exe_sha256=sha(Path(command[0])),
                    dll_sha256=sha(binary / 'gmp-10.dll'), tool_sha256=sha(NCU), diagnostic=True)
        write(out / 'command.json', meta)
        shutil.copy2(__file__, out / 'collector_snapshot.py')
    if a.run_admin:
        allowed = []
        text = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name', '--format=csv,noheader'], text=True)
        for pid in a.allow_gpu_pid:
            names = {line.split(',', 2)[2].strip() for line in text.splitlines() if line.split(',')[1].strip() == str(pid)}
            if len(names) != 1 or not Path(next(iter(names))).is_absolute():
                raise ValueError('Cannot bind allowed process')
            allowed.append(dict(pid=pid, process_name=next(iter(names))))
        telemetry = TelemetryRecorder(1, out / 'telemetry.jsonl', interval=.2)
        meta.update(allowed_processes=allowed,device=telemetry.identity)
        write(out / 'command.json',meta)
        try:
            check_gpu_processes(telemetry.monitor, out, out / 'before.json', allowed=allowed)
            subprocess.run(['powershell', '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(out / 'admin.ps1')], check=True, timeout=600)
            check_gpu_processes(telemetry.monitor, out, out / 'after.json', allowed=allowed)
        finally:
            telemetry.close()
    if a.collect_only or a.run_admin:
        if int((out / 'exit.txt').read_text(encoding='utf-8-sig').strip()):
            raise ValueError('Administrator capture failed')
        text = (out / 'run.log').read_text(encoding='utf-8', errors='replace')
        fields = dict(re.findall(r'(\w+)=([^\s]+)', text))
        for key in ('sha256', 'checked_words', 'output_words', 'bad'):
            if fields.get(key) != meta['fields'][key]:
                raise ValueError('Profiled point output differs: ' + key)
        if sha(Path(meta['app'][0])) != meta['exe_sha256'] or sha(Path(meta['app'][0]).parent/'gmp-10.dll') != meta['dll_sha256']:
            raise ValueError('Profile binary changed')
        with (out / 'metrics.csv').open('wb') as dest:
            subprocess.run([str(NCU), '--import', str(out / 'trace.ncu-rep'), '--csv', '--page', 'raw'],
                           stdout=dest, stderr=subprocess.STDOUT, check=True, timeout=60)
        write(out / 'summary.json', dict(complete=True, diagnostic=True, reference=meta,
              report_sha256=sha(out / 'trace.ncu-rep'), metrics_sha256=sha(out / 'metrics.csv')))
        print('Collected', out)
    else:
        print(out / 'admin.ps1')


if __name__ == '__main__':
    main()
