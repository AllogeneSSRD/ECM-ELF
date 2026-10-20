"""Exercise migrated optional Stage1 runtimes against their independent CPU backend."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

from test_cuda_prac import rows, run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', type=int, default=1)
    a = p.parse_args()
    source = a.exe.resolve()
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    binary = out/'bin'; binary.mkdir()
    for name in ('ecm_cuda.exe', 'gmp-10.dll'):
        shutil.copy2(source.parent/name, binary/name)
    for sidecar in source.parent.glob('ecm_cuda.exe.*'):
        shutil.copy2(sidecar, binary/sidecar.name)
    (binary/'ecm.toml').write_text('[stage1]\nauto_stage2 = 0\nedwards_stage2_transfer = false\n')
    exe = binary/source.name
    identity = hashlib.sha256(exe.read_bytes()).hexdigest()
    result = dict(complete=False, binary_sha256=identity, device=a.device, cases=[], performance_samples=0)

    def persist():
        (out/'result.json').write_text(json.dumps(result, indent=2)+'\n')

    for b1 in (20, 1000):
        for exponent in ('lcm', 'choose12'):
            name = f'b{b1}_{exponent}'
            settings = ('(2^127-1)', b1, 4, 26, exponent)
            reference = out/name/'cpu'
            run(exe, reference, *settings, 'cpu', a.device)
            expected = rows(reference)
            assert len(expected) == 4
            for algorithm in ('ladder', 'resident', 'prac'):
                folder = out/name/algorithm
                run(exe, folder, *settings, algorithm, a.device, registers=0)
                assert rows(folder) == expected, (name, algorithm)
                result['cases'].append(dict(case=name, algorithm=algorithm, records=len(expected), matched=True))
                persist(); print('PASS', name, algorithm, flush=True)

    settings = ('(2^127-1)', 10000, 4, 26, 'choose12')
    reference = out/'resume'/'reference'
    run(exe, reference, *settings, 'cpu', a.device)
    expected = rows(reference); assert len(expected) == 4
    for algorithm in ('resident', 'prac'):
        for corrupt in (False, True):
            folder = out/'resume'/f'{algorithm}_corrupt{int(corrupt)}'
            text = run(exe, folder, *settings, algorithm, a.device, sample=0.000001, registers=0)
            assert 'sample limit reached' in text and not rows(folder)
            checkpoints = list(folder.glob('.*.ckpt'))
            assert len(checkpoints) == 2 and sum('.bak.ckpt' in f.name for f in checkpoints) == 1, (folder, checkpoints)
            if corrupt:
                # Exhaust both binary rotation candidates: current behavior is rejection.
                for checkpoint in checkpoints:
                    payload = bytearray(checkpoint.read_bytes()); payload[-1] ^= 1
                    checkpoint.write_bytes(payload)
                if algorithm == 'prac':
                    cache = next((folder/'plan_cache').glob('*.bin'))
                    payload = bytearray(cache.read_bytes()); payload[-1] ^= 1; cache.write_bytes(payload)
            frozen = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in checkpoints}
            text = run(exe, folder, *settings, algorithm, a.device, registers=0, expect_records=not corrupt)
            if corrupt:
                assert 'no valid batch checkpoint generation' in text and not rows(folder)
                assert {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in checkpoints} == frozen
                retry = folder/'clean_retry'
                if algorithm == 'prac': shutil.copytree(folder/'plan_cache', retry/'plan_cache')
                text = run(exe, retry, *settings, algorithm, a.device, registers=0)
                if algorithm == 'prac': assert 'PRAC plan built' in text
                assert rows(retry) == expected
            else:
                assert 'checkpoint resumed' in text and rows(folder) == expected
                assert not list(folder.glob('.*.ckpt'))
            result['cases'].append(dict(case='resume', algorithm=algorithm, corruption=corrupt,
                                        records=len(expected), matched=True))
            persist(); print('PASS resume', algorithm, corrupt, flush=True)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == identity
    assert hashlib.sha256(exe.read_bytes()).hexdigest() == identity
    result['complete'] = True; persist()
    print(json.dumps(dict(complete=True, cases=len(result['cases']))))


if __name__ == '__main__': main()
