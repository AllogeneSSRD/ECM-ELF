"""Reject HostOnly reuse when a recorded CUDA input differs (no compilation).

The phase-times header belongs to host code; changing it is not a reason to
reject CUDA reuse. Mutate an actual CUDA input so the rejection is specific,
and happens before the copied fixture's different object paths are examined.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
CUDA_INPUT = 'src/ecm/stage2/cuda/device/arithmetic_kernels.cu'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build', type=Path, required=True)
    p.add_argument('--pwsh', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--vcvars', type=Path, default=Path(
        r'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat'))
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    manifest_path = a.build/'build_manifest.json'
    original = manifest_path.read_bytes()
    manifest = json.loads(original.decode('utf-8-sig'))
    assert CUDA_INPUT in manifest['cuda_sources']
    assert manifest['source_hashes'][CUDA_INPUT].lower() == hashlib.sha256((ROOT/CUDA_INPUT).read_bytes()).hexdigest()
    manifest['source_hashes'][CUDA_INPUT] = '0'*64
    fake = a.output/'mismatched_build'
    fake.mkdir()
    (fake/'build_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    command = [str(a.pwsh), '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(ROOT/'tools/build/build_stage2_local.ps1'),
               '-HostOnly', '-Build', str(fake), '-Engine', manifest['engine'],
               '-Arch', manifest['architecture'], '-GlBackend', manifest['gl_backend'],
               '-OuterUnrollU', str(manifest['outer_unroll_u']), '-AddSubMask', str(manifest['add_sub_mask']),
               '-SplitCompile', str(manifest['split_compile']), '-Gmp', manifest['gmp_root'],
               '-CudaRoot', manifest['cuda_root'], '-VcVars', str(a.vcvars)]
    env = dict(os.environ)
    if a.pwsh.name.lower() == 'powershell.exe':
        # Do not let an inherited PS7 module path hide WinPS 5.1's cmdlets.
        env['PSModulePath'] = str(Path(os.environ['WINDIR'])/'System32/WindowsPowerShell/v1.0/Modules')
    proc = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, errors='replace', timeout=60)
    log = proc.stdout+proc.stderr
    (a.output/'rejection.log').write_text(log, encoding='utf-8')
    assert proc.returncode != 0 and 'HostOnly CUDA dependency changed: '+CUDA_INPUT in log, log
    assert '== compile ' not in log and not (fake/'ecm_cuda_stage2.exe').exists()
    assert manifest_path.read_bytes() == original
    result = dict(complete=True, rejected_before_compile=True, source_build_unchanged=True)
    (a.output/'result.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
