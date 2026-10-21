"""CPU-only release-leaf selection gates using generated PowerShell identities.

This does not build or qualify release programs, CUDA images or packages.
"""
import argparse
import json
from pathlib import Path
from test_cuda_p5_release import profile_hash, release_leaf


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fields', type=Path, required=True)
    p.add_argument('--expected-hash', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    fields = json.loads(a.fields.read_text(encoding='utf-8-sig'))
    checks = []

    def check(value, name):
        assert value, name
        checks.append(name)

    def rejects(action, name):
        try:
            action()
        except AssertionError:
            checks.append(name)
        else:
            raise AssertionError(name)

    check(profile_hash(fields) == a.expected_hash.read_text().strip(), 'Python agrees with the actual PowerShell profile digest')
    root = out/'build_cuda_release'
    fields.update(role='release', stage='stage2', engine='production', architectures='89')

    def marker(data):
        digest = profile_hash(data)
        leaf = root/'stage2/cuda13_3/sm89'/digest[:16]
        leaf.mkdir(parents=True, exist_ok=False)
        (leaf/'profile.json').write_text(json.dumps(dict(schema=1, profile_sha256=digest, fields=data)), encoding='utf-8')
        return leaf

    first = marker(dict(fields))
    check(release_leaf(root, 'stage2', 89) == first, 'one compatible default profile selected')
    alternative = dict(fields); alternative['option/split_compile'] = '4'
    marker(alternative)
    check(release_leaf(root, 'stage2', 89) == first, 'non-default schedule is not substituted for the requested default')
    second = dict(fields); second['gmp/lib/gmp.lib'] = 'different coherent library'
    second = marker(second)
    rejects(lambda: release_leaf(root, 'stage2', 89), 'ambiguous profiles fail instead of picking first/latest')
    check(release_leaf(root, 'stage2', 89, first) == first, 'explicit leaf resolves ambiguity')
    raw = (first/'profile.json').read_bytes()
    damaged = json.loads(raw); damaged['fields']['role'] = 'dev'
    (first/'profile.json').write_text(json.dumps(damaged), encoding='utf-8')
    rejects(lambda: release_leaf(root, 'stage2', 89, first), 'tampered marker rejected even with retained digest')
    (first/'profile.json').write_bytes(raw)
    rejects(lambda: release_leaf(root, 'stage1', 89, first), 'explicit leaf cannot cross stages')
    check(release_leaf(out/'legacy', 'stage2', 89) == out/'legacy/sm_89', 'frozen legacy Stage2 layout retained')
    check(release_leaf(out/'legacy', 'stage1', 89) == out/'legacy/cuda13_3', 'frozen legacy Stage1 layout retained')
    result = dict(complete=True, checks=len(checks), tests=checks, gpu_runs=0, release_builds=0)
    (out/'result.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
