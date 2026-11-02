"""Replay one checked fixed-D curve with CPU NVTX and CUDA activity tracing.

Administrator diagnostic only. Trace durations never enter timing matrices.
The selected executable must have its build manifest and frozen sources beside it.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess

from bench_stage2_n_scaling import fields
from bench_stage2_point_routes import sha
from bench_stage2_runtime_matrix import completed_records
from bench_stage2_version import check_gpu_processes,load,write
from gpu_telemetry import TelemetryRecorder
from result_format import normalize_result

NSYS=Path('C:/Program Files/NVIDIA Corporation/Nsight Systems 2026.1.3/target-windows-x64/nsys.exe')


def quoted(args):
    if any(any(c in str(s) for c in '&|<>%!^\r\n"\'') for s in args):
        raise ValueError('Unsupported shell character in diagnostic command')
    return ' '.join('"'+str(s)+'"' for s in args)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('reference','exe','output','study'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--background-snapshot',type=Path)
    a=p.parse_args();out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    reference=a.reference.resolve();original=load(reference/'command.json');cmd=list(original)
    manifest=load(a.exe.parent/'build_manifest.json')
    if sha(a.exe)!=manifest['sha256'].lower():raise ValueError('Frozen binary differs')
    for name,digest in manifest['source_hashes'].items():
        if sha(a.exe.parent/'sources'/name)!=digest.lower():raise ValueError('Frozen source differs: '+name)
    def arg(flag):return cmd[cmd.index(flag)+1]
    cmd[0]=str(a.exe.resolve());inputs={}
    for flag,name in (('--config','ecm.toml'),('--save','points.save')):
        source=Path(arg(flag));shutil.copy2(source,out/name);inputs[name]=sha(source);cmd[cmd.index(flag)+1]=str(out/name)
    for flag,name in (('--log','engine.log'),('--debug-log-file','debug.log'),('--results','result.jsonl'),
                      ('--submission','submission.txt'),('--tune-profile','unused_tune.toml')):
        cmd[cmd.index(flag)+1]=str(out/name)
    device=int(arg('--device'));source=load(a.study/'plan.json');allowed=source['allowed_processes']
    expected=completed_records(reference/'result.jsonl')
    if len(expected)!=1 or expected[0]['status']!='NF':raise ValueError('Need one completed NF reference')
    if a.background_snapshot:
        background=load(a.background_snapshot)
        if not background.get('approved_background_by_user') or background['device']!=device:raise ValueError('Unapproved background')
        allowed=allowed+background['allowed_processes'];shutil.copy2(a.background_snapshot,out/'background_snapshot.json')
    (out/'app.cmd').write_text('@echo off\nfor /f "tokens=1 delims==" %%v in (\'set NTT_ 2^>nul\') do set "%%v="\n'
        'set "NTT_HOST_TRACE=1"\n'+quoted(cmd)+' > "'+str(out/'console.log')+'" 2>&1\nexit /b %errorlevel%\n')
    profile=[str(NSYS),'profile','--trace=cuda,nvtx','--sample=none','--cpuctxsw=none','--cuda-graph-trace=node',
        '--cuda-trace-all-apis=true','--force-overwrite=false','--output',str(out/'trace'),
        'C:/Windows/System32/cmd.exe','/d','/c',str(out/'app.cmd')]
    (out/'profile.cmd').write_text('@echo off\n'+quoted(profile)+' > "'+str(out/'run.log')+'" 2>&1\nexit /b %errorlevel%\n')
    admin=out/'admin.ps1';admin.write_text("$treeProfile = Start-Process -FilePath 'C:/Windows/System32/cmd.exe' -ArgumentList @('/d','/c','\""+str(out/'profile.cmd')+"\"') -Verb RunAs -WindowStyle Hidden -PassThru\n$treeProfile.WaitForExit()\n$treeProfile.ExitCode | Set-Content -LiteralPath '"+str(out/'exit.txt')+"'\nexit $treeProfile.ExitCode\n")
    meta=dict(complete=False,diagnostic=True,app=cmd,profile=profile,device=device,inputs=inputs,
        exe_sha256=sha(a.exe),reference=str(reference),reference_result_sha256=sha(reference/'result.jsonl'))
    write(out/'summary.json',meta);shutil.copy2(__file__,out/'collector_snapshot.py')
    monitor=TelemetryRecorder(device,out/'telemetry.jsonl',interval=.2)
    try:
        if a.background_snapshot and monitor.identity['uuid']!=background['uuid']:raise ValueError('Background UUID differs')
        check_gpu_processes(monitor.monitor,out,out/'before.json',allowed=allowed)
        r=subprocess.run(['powershell','-NoProfile','-ExecutionPolicy','Bypass','-File',str(admin)],capture_output=True,text=True,timeout=900)
        (out/'launcher.log').write_text(r.stdout+r.stderr)
        check_gpu_processes(monitor.monitor,out,out/'after.json',allowed=allowed)
    finally:monitor.close()
    if r.returncode:raise ValueError('Nsight Systems failed; inspect diagnostic logs')
    rows=completed_records(out/'result.jsonl')
    if len(rows)!=1 or rows[0]['status']!='NF' or rows[0]['source-build']!=meta['exe_sha256']:
        raise ValueError('Profile result/binary differs')
    actual,wanted=normalize_result(rows[0]),normalize_result(expected[0])
    for key in ('N_hex','sigma','B1','B2','curve','param','carrier_exponent','factors','bad_factors'):
        if actual[key]!=wanted[key]:raise ValueError('Profile input/result differs: '+key)
    for key in ('d','poly-size'):
        if rows[0][key]!=expected[0][key]:raise ValueError('Geometry differs')
    texts=['\n'.join((folder/name).read_text(encoding='utf-8') for name in ('engine.log','debug.log')) for folder in (reference,out)]
    work=[]
    for text in texts:
        s=fields(text,'s4_multiply_stats')
        if min(int(s[k]) for k in ('gmp_selftest_cases','gmp_checked'))<=0 or any(int(s[k]) for k in ('gmp_check_bad','gmp_selftest_bad')):
            raise ValueError('Arithmetic checks missing/failed')
        work.append({k:s[k] for k in ('launches','poly_muls','coeffs_reduced')})
    counts=[[s for s in text.splitlines() if s.startswith(('ntt_length_counts:','ntt_convolution_counts:'))] for text in texts]
    if work[0]!=work[1] or not counts[0] or counts[0]!=counts[1]:raise ValueError('Trace work differs')
    if sha(a.exe)!=meta['exe_sha256'] or any(sha(out/name)!=digest for name,digest in inputs.items()):
        raise ValueError('Diagnostic inputs changed')
    with (out/'export.log').open('wb') as log:
        subprocess.run([str(NSYS),'export','--type=sqlite','--output',str(out/'trace.sqlite'),str(out/'trace.nsys-rep')],
                       stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
    meta.update(complete=True,trace_sha256=sha(out/'trace.sqlite'),work=work[1]);write(out/'summary.json',meta)
    print('Checked tree trace:',out,flush=True)


if __name__=='__main__':main()
