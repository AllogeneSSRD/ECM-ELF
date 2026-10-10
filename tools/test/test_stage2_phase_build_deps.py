"""Reject HostOnly reuse when the recorded phase header differs (no compilation)."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
HEADER = 'src/core/ecm_stage2_phase_times.h'


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
    assert HEADER in manifest['source_hashes']
    assert manifest['source_hashes'][HEADER].lower() == hashlib.sha256((ROOT/HEADER).read_bytes()).hexdigest()
    manifest['source_hashes'][HEADER] = '0'*64
    fake = a.output/'mismatched_build'
    fake.mkdir()
    (fake/'build_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    command = [str(a.pwsh), '-NoProfile', '-File', str(ROOT/'tools/build/build_stage2_local.ps1'),
               '-HostOnly', '-Build', str(fake), '-Engine', manifest['engine'],
               '-Arch', manifest['architecture'], '-GlBackend', manifest['gl_backend'],
               '-OuterUnrollU', str(manifest['outer_unroll_u']), '-AddSubMask', str(manifest['add_sub_mask']),
               '-SplitCompile', str(manifest['split_compile']), '-Gmp', manifest['gmp_root'],
               '-CudaRoot', manifest['cuda_root'], '-VcVars', str(a.vcvars)]
    proc = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, errors='replace', timeout=60)
    log = proc.stdout+proc.stderr
    (a.output/'rejection.log').write_text(log, encoding='utf-8')
    assert proc.returncode != 0 and 'HostOnly CUDA dependency changed: '+HEADER in log, log
    assert '== compile ' not in log and not (fake/'ecm_cuda_stage2.exe').exists()
    assert manifest_path.read_bytes() == original
    result = dict(complete=True, rejected_before_compile=True, source_build_unchanged=True)
    (a.output/'result.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
