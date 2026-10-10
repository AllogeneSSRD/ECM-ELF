"""Compare the native no-Brent-Suyama probability with tools/ecm_prob/rho.py."""
import argparse
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--vcvars', type=Path, default=Path(
        r'C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat'))
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    exe = out/'probability.exe'
    cmd = out/'compile.cmd'
    cmd.write_text('@echo off\ncall "'+str(args.vcvars)+'" >nul 2>&1\n'
                   'if errorlevel 1 exit /b 1\ncl /nologo /std:c++17 /EHsc /O2 /utf-8 "'+
                   str(ROOT/'tools/test/stage2_probability_fixture.cpp')+'" /Fe:"'+str(exe)+
                   '" /Fo:"'+str(out/'probability.obj')+'"\n', encoding='utf-8')
    proc = subprocess.run(['cmd', '/c', str(cmd)], capture_output=True, text=True,
                          errors='replace', timeout=60)
    (out/'compile.log').write_text(proc.stdout+proc.stderr, encoding='utf-8')
    if proc.returncode:
        raise RuntimeError(proc.stdout+proc.stderr)
    sys.path.insert(0, str(ROOT/'tools/ecm_prob'))
    import rho
    cases = []
    for b1, ratio, bits, delta in itertools.product(
            [20, 256, 1000, 10000, 1e6, 1e7, 110e6, 260e6],
            [1, 10, 100, 10000], [20, 30, 60, 100, 130, 180, 225],
            [rho.ECM_EXTRA_SMOOTHNESS,
             rho.ECM_EXTRA_SMOOTHNESS+math.log(rho.EXTRA_SMOOTHNESS_32BITS_D)]):
        # The reference's integration assumes B2 <= effective group-order size.
        # Saturation beyond that bound is checked separately in the native model.
        b2 = b1*ratio
        n = 2.0**(bits-.5)
        if b2 <= n/math.exp(delta):
            cases.append((b1, b2, bits, delta))
    stdin = ''.join(' '.join(map(str, row))+'\n' for row in cases)
    values = list(map(float, subprocess.check_output([str(exe)], input=stdin, text=True).split()))
    assert len(values) == len(cases)
    maximum = 0.0
    rows = []
    for row, value in zip(cases, values):
        b1, b2, bits, delta = row
        expected = max(0.0, min(1.0, rho.prob(b1, b2, 2.0**(bits-.5), 0, 0, delta)))
        error = abs(value-expected)
        maximum = max(maximum, error)
        rows.append(dict(b1=b1, b2=b2, factor_bits=bits, delta=delta,
                         expected=expected, actual=value, absolute_error=error))
        assert math.isfinite(value) and 0 <= value <= 1
        assert error <= 2e-12, rows[-1]
    edges = [(1000, 1e15, 20, 3.134), (1e6, 1e18, 8192, 3.134),
             (1e6, 1e12, 10, 3.134), (20, 20, 30, 3.134)]
    edge_values = list(map(float, subprocess.check_output(
        [str(exe)], input=''.join(' '.join(map(str, row))+'\n' for row in edges), text=True).split()))
    assert all(math.isfinite(v) and 0 <= v <= 1 for v in edge_values)
    assert edge_values[2] == 1
    for bad in ['1 100 30 3.134\n', '100 99 30 3.134\n', '100 1000 1 3.134\n']:
        assert subprocess.run([str(exe)], input=bad, text=True, capture_output=True).returncode != 0
    report = dict(passed=True, cases=len(cases), max_absolute_error=maximum,
                  reference='tools/ecm_prob/rho.py', rows=rows,
                  edge_cases=[dict(input=row, actual=v) for row, v in zip(edges, edge_values)])
    (out/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({key: report[key] for key in ['passed', 'cases', 'max_absolute_error']}))


if __name__ == '__main__':
    main()
