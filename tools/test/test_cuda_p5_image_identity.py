"""Reject incomplete or substituted CUDA image inputs before GPU initialization."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    sources = {}
    for name in ('native-stage1','native-stage2','small-save','fixtures'):
        path = out/name; path.write_bytes(b'identity fixture: '+name.encode())
        sources[name] = path
    images = []
    for stage in ('stage1','stage2'):
        for arch in (60,70,75,86,89,120):
            folder = out/stage/f'cuda-{stage}-sm{arch}'; folder.mkdir(parents=True)
            exe = folder/('ecm_cuda.exe' if stage == 'stage1' else 'ecm_cuda_stage2.exe')
            exe.write_bytes(b'never executed image identity fixture')
            images.append(dict(stage=stage, architecture=arch, exe_sha256=sha(exe)))
    static = dict(complete=True, images=images)
    native = dict(complete=True, inputs={str(path):sha(path) for path in sources.values()})
    checks = []
    for name, message in [('native_incomplete','native gate incomplete'),
                           ('native_substituted','native reference identity changed'),
                           ('image_substituted','release image identity changed'),
                           ('architecture_duplicate','incomplete release architecture matrix')]:
        s = json.loads(json.dumps(static)); n = json.loads(json.dumps(native))
        if name == 'native_incomplete': n['complete'] = False
        if name == 'native_substituted': n['inputs'][str(sources['native-stage1'])] = '0'*64
        if name == 'image_substituted': s['images'][-1]['exe_sha256'] = '0'*64
        if name == 'architecture_duplicate': s['images'][-1] = s['images'][-2]
        case = out/name; case.mkdir()
        sg = case/'static.json'; sg.write_text(json.dumps(s))
        ng = case/'native.json'; ng.write_text(json.dumps(n))
        target = case/'gpu_study'
        command = [sys.executable, str(ROOT/'tools/test/test_cuda_p5_images.py'),
            '--static-gate', str(sg), '--native-gate', str(ng),
            '--stage1-packages', str(out/'stage1'), '--stage2-packages', str(out/'stage2'),
            '--output', str(target)]
        for key,path in sources.items(): command += ['--'+key, str(path)]
        proc = subprocess.run(command, capture_output=True)
        text = (proc.stdout+proc.stderr).decode('utf-8','replace')
        (case/'stdout.log').write_text(text)
        assert proc.returncode and message in text and not target.exists(), (name, text)
        checks.append(name)
    collector = ROOT/'tools/test/test_cuda_p5_images.py'
    result = dict(complete=True, checks=checks, gpu_runs=0,
        inputs={str(path):sha(path) for path in (Path(__file__),collector)})
    (out/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print('PASS image identity:', len(checks), 'CPU negative gates')


if __name__ == '__main__': main()
