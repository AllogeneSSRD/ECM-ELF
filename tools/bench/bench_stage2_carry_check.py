"""Same-binary fixed-save Stage2 comparison of optional fused carry diagnostics."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for option in ('exe', 'save', 'output'):
        p.add_argument('--' + option, type=Path, required=True)
    p.add_argument('--device', type=int, default=1)
    a = p.parse_args()
    repo = Path(__file__).resolve().parents[2]
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    assert not any(out.iterdir()), 'Use a fresh output directory'
    exe, saved = a.exe.resolve(), a.save.resolve()
    manifest = json.loads((exe.parent / 'build_manifest.json').read_text(encoding='utf-8-sig'))
    assert manifest['gl_fixed_mode'] == 3 and manifest['architecture'] == 'sm_89'
    assert manifest['outer_unroll_u'] == 0
    sources = {r[1]: r[2].lower() for line in manifest['sources']
               if (r := re.fullmatch(r'([^=]+\.(?:cu|cuh|cpp|h|ps1))=([A-Fa-f0-9]{64})', line))}
    assert len(sources) == 20
    closure = out / 'sources'
    for name in sources:
        target = closure / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((repo / name).read_bytes())
    saved_sha, driver_sha = digest(saved), digest(Path(__file__))
    saved_x = re.search(rb'\bX=(?:0x)?([0-9a-fA-F]+)', saved.read_bytes().splitlines()[0])[1].decode().lower().lstrip('0')
    env = {k: v for k, v in os.environ.items() if not k.startswith('NTT_')}
    env.update(NTT_NO_PROGRESS='1', NTT_D_MODEL='0', NTT_ARENA_CAP_KB='6451200',
               NTT_STAGE1_Q_DUMP='1', NTT_POINT_MERSENNE='1')
    rows, warmups = [], []

    def verify():
        assert digest(exe) == manifest['sha256'].lower()
        assert digest(saved) == saved_sha and digest(Path(__file__)) == driver_sha
        for name, want in sources.items():
            assert digest(repo / name) == want and digest(closure / name) == want, name

    # Prespecified, fully checked warmups precede all measured runs.
    order = [(True, 0), (True, 1)] + [(False, mode) for mode in (0, 1, 1, 0, 1, 0, 0, 1)]
    for warmup, mode in order:
        verify()
        name = f'warmup_{mode}' if warmup else f'{len(rows)+1}_f{mode}'
        cmd = [str(exe), '--save', str(saved), '--b2', '2011326186870', '--d', '1381380',
               '--device', str(a.device), '--results', str(out / (name + '.jsonl')),
               '--log', str(out / (name + '_engine.log'))]
        with (out / (name + '_driver.log')).open('wb') as log:
            run = subprocess.run(cmd, env=env | {'NTT_CARRY_CHECK_FUSED': str(mode)},
                                 stdout=log, stderr=subprocess.STDOUT, timeout=600)
        verify()
        assert run.returncode == 0, (name, run.returncode)
        text = (out / (name + '_engine.log')).read_text(encoding='utf-8', errors='replace')
        for token in ('stage1_skipped=1', 'gmp_selftest_bad=0', 'gmp_check_bad=0', 'pending=0',
                      'clean=1', 'point_arithmetic: xadd6=1', 'fixed=3', 'hash=4244971527793015097',
                      'signature=c85031f6149bae11', 'point_mersenne_mode: requested=1 enabled=1 bits=4423 nw=70',
                      'd_model: requested=0 enabled=0 version=legacy_56_1', 'ntt_outer_schedule: unroll_u=0'):
            assert token in text, (name, token)
        assert re.search(r'real_setup_Q_full: hex=([0-9a-f]+)', text)[1] == saved_x
        result = json.loads((out / (name + '.jsonl')).read_text(encoding='utf-8').splitlines()[-1])
        assert result['bad_factors'] == 0 and result['factors'] == []
        wall = {k: float(v) for k, v in re.findall(r'(init|main|total)=([0-9.]+)', re.search(r'stage2_full_wall: (.*)', text)[1])}
        s4 = dict(re.findall(r'(\w+)=([^ ]+)', re.search(r's4_multiply_stats: (.*)', text)[1]))
        coverage = {k: int(s4[k]) for k in ('launches', 'poly_muls', 'coeffs_reduced', 'gmp_selftest_cases', 'gmp_checked', 'full_checks')}
        assert coverage == dict(launches=397, poly_muls=1836241, coeffs_reduced=36615543,
                                gmp_selftest_cases=2400, gmp_checked=60474, full_checks=3)
        carry = {k: int(v) for k, v in re.findall(r'(\w+)=(\d+)', re.search(r'ntt_carry_check_stats: (.*)', text)[1])}
        assert carry['requested'] == mode
        assert (carry['fused_calls'] > 0 and carry['scratch_peak_bytes'] > 0) if mode else (carry['fused_calls'] == 0 and carry['scratch_peak_bytes'] == 0)
        row = dict(name=name, mode=mode, command=cmd, wall=wall, coverage=coverage, carry=carry)
        (warmups if warmup else rows).append(row)
        print(name, wall, carry, flush=True)
        (out / 'measurements.json').write_text(json.dumps(dict(warmups=warmups, runs=rows), indent=2), encoding='utf-8')
    means = {str(mode): {phase: statistics.mean(r['wall'][phase] for r in rows if r['mode'] == mode)
                        for phase in ('init', 'main', 'total')} for mode in (0, 1)}
    gains = {phase: 100 * (1 - means['1'][phase] / means['0'][phase]) for phase in ('init', 'main', 'total')}
    result = dict(manifest=manifest, sources=sources, driver_sha256=driver_sha, save_sha256=saved_sha,
                  Q_sha256=hashlib.sha256(saved_x.encode()).hexdigest(), device=a.device, env=env,
                  warmups=warmups, runs=rows, means=means, gain_percent=gains, passed=10, failed=0,
                  scope='Same binary, point1/unroll0, fixed Q/B2/D/checks, 2 prespecified warmups then ABBA+BAAB; no CI; D model off')
    (out / 'measurements.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(dict(means=means, gain_percent=gains)), flush=True)


if __name__ == '__main__':
    main()
