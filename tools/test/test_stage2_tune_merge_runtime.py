"""Verify production tune merge, compatibility gates and atomic publication."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tomllib


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--exe', type=Path, required=True)
    p.add_argument('--profile-a', type=Path, required=True)
    p.add_argument('--profile-b', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    sources = [a.exe.resolve(), a.profile_a.resolve(), a.profile_b.resolve()]
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    identities = {str(path): sha(path) for path in sources}
    target = out/'combined.toml'
    ini = out/'merge.ini'
    ini.write_text('verbose=false\nstage2_tune_profile='+str(target)+'\n', encoding='utf-8')
    # Deliberately invalid CUDA device: merging must only read the files.
    common = [str(a.exe.resolve()), '--ini', str(ini), '--device', '9999']

    def run(name, options, success=True):
        result = subprocess.run(common+options, capture_output=True, text=True,
                                errors='replace', timeout=60)
        (out/(name+'.log')).write_text(result.stdout+result.stderr, encoding='utf-8')
        assert (result.returncode == 0) == success, (name, result.stdout, result.stderr)

    command = ['--tune', 'ecm', '--tune-merge', str(a.profile_a.resolve()),
               '--tune-merge', str(a.profile_b.resolve()), '--tune-file', str(target)]
    run('merge', command)
    data = tomllib.loads(target.read_text(encoding='utf-8'))
    fields = ['target_bits', 'arithmetic_bits', 'carrier_exponent', 'modulus_kind', 'b1', 'b2', 'd']
    expected = {}
    for path in [a.profile_a, a.profile_b]:
        for sample in tomllib.loads(path.read_text(encoding='utf-8'))['ecm'].values():
            expected[tuple(sample[k] for k in fields)] = sample
    actual = {tuple(s[k] for k in fields): s for s in data['ecm'].values()}
    assert actual == expected
    assert data['profile']['format'] == 3
    assert data['summary']['measured'] == len(expected)
    assert data['summary']['merged_profiles'] == 2
    for forbidden in ['binary', 'manifest', 'sha256', 'path']:
        assert forbidden not in target.read_text(encoding='utf-8').lower()
    previous = target.read_bytes()
    broken = out/'broken.toml'
    broken.write_text(a.profile_b.read_text(encoding='utf-8').replace('batch_mb = 256', 'batch_mb = 64'), encoding='utf-8')
    run('policy_mismatch', ['--tune', 'ecm', '--tune-merge', str(a.profile_a.resolve()),
                           '--tune-merge', str(broken), '--tune-file', str(target)], False)
    assert target.read_bytes() == previous
    broken.write_text('[profile]\nformat = 3\n', encoding='utf-8')
    run('invalid_profile', ['--tune', 'ecm', '--tune-merge', str(broken), '--tune-file', str(target)], False)
    assert target.read_bytes() == previous
    run('inplace_rejected', ['--tune', 'ecm', '--tune-merge', str(a.profile_a.resolve()),
                            '--tune-file', str(a.profile_a.resolve())], False)
    run('grid_conflict', command+['--tune-b2', '26000000000'], False)
    run('ntt_conflict', ['--tune', 'ntt', '--tune-merge', str(a.profile_a.resolve()), '--tune-file', str(target)], False)
    run('ntt_cannot_overwrite_ecm', ['--tune', 'ntt', '--tune-file', str(target)], False)
    run('mode_required', ['--tune-merge', str(a.profile_a.resolve()), '--tune-file', str(target)], False)
    assert target.read_bytes() == previous
    assert identities == {str(path): sha(path) for path in sources}
    report = {'merged_samples': len(expected), 'failed_publication_retains_existing': True,
              'source_files_unchanged': True, 'no_cuda_device_required': True,
              'ini_configured_profile_can_be_refreshed': True, 'rejected_invocations': 7}
    (out/'result.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report))


if __name__ == '__main__':
    main()
