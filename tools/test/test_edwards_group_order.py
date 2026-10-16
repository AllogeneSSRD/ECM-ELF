"""Cross-check PARI Edwards orders against C++ setup and independent field arithmetic."""
import argparse
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'tools/ecm_dataset'))
import edwards_order as order


def require(condition, message):
    if not condition:
        raise ValueError(message)


def multiply(point, scalar, p, a2, a4):
    """Affine Weierstrass oracle, independent of PARI and C++ Jacobian setup."""
    def add(left, right):
        if left is None: return right
        if right is None: return left
        x, y = left; u, v = right
        if x == u:
            if (y+v) % p == 0: return None
            slope = (3*x*x+2*a2*x+a4)*pow(2*y, -1, p) % p
        else:
            slope = (v-y)*pow(u-x, -1, p) % p
        nx = (slope*slope-a2-x-u) % p
        return nx, (slope*(x-nx)-y) % p
    result = None
    for bit in bin(scalar)[2:]:
        result = add(result, result)
        if bit == '1': result = add(result, point)
    return result


def cpp_curve(probe, p, sigma):
    run = subprocess.run([str(probe), str(p), str(sigma)], capture_output=True, check=True, timeout=10)
    return dict(line.split('|', 1) for line in run.stdout.decode().splitlines()), run.stdout


def verify_point(row):
    p = int(row['prime']);x = int(row['edwards_x']);y = int(row['edwards_y'])
    a, b = int(row['montgomery_a']), int(row['montgomery_b'])
    require(pow(b, (p-1)//2, p) == 1, 'Atkin-Morain B must be a square')
    point = None if y == 1 else (0, 0) if y == p-1 else (
        b*(1+y)*pow(1-y, -1, p) % p,
        b*b*(1+y)*pow((1-y)*x, -1, p) % p)
    n = int(row['point_order'])
    require(multiply(point, n, p, a*b % p, b*b % p) is None, 'Point order does not annihilate P')
    for q, _ in row['point_factorization']:
        require(multiply(point, n//int(q), p, a*b % p, b*b % p) is not None, 'Point order is not minimal')
    effective = int(row['order_after_48'])
    require(multiply(point, 48*effective, p, a*b % p, b*b % p) is None, '48 adjustment failed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, required=True, help='edwards_order.py fixture run')
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--gp', nargs='?')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args();gp = order.gp_path(args.gp)
    out = args.output.resolve();out.mkdir(parents=True, exist_ok=False)
    frozen = out/'bin';frozen.mkdir()
    probe = args.probe.resolve()
    for path in (probe, probe.parent/'gmp-10.dll', probe.parent/'manifest.json'):
        shutil.copy2(path, frozen/path.name)
    probe = frozen/probe.name
    reference = json.loads((args.reference/'summary.json').read_text())
    fixtures = json.loads((args.reference/'fixtures.json').read_text())
    require(reference['complete'] and len(reference['results']) == len(fixtures['cases']) == 17, 'Incomplete reference suite')
    require(reference['gp_sha256'][gp.name] == order.sha(gp), 'Different GP executable')
    require(all(reference['source_sha256'][name] == order.sha(args.reference/name)
                for name in reference['source_sha256']), 'Changed frozen sources')
    checked = []
    for i, row in enumerate(reference['results']):
        case = fixtures['cases'][i]
        require(all(row[k] == v for k, v in case.items()), 'Reference fixture row changed')
        folder = args.reference/f'case_{i:02d}'
        require(order.sha(folder/'stdout.log') == row['stdout_sha256'] and
                order.sha(folder/'stderr.log') == row['stderr_sha256'], 'Changed raw GP logs')
        parsed = order.parse_order((folder/'stdout.log').read_text(), int(row['prime']))
        require(all(row[k] == v for k, v in parsed.items()), 'Stale reference analysis')
        cpp, raw = cpp_curve(probe, row['prime'], row['sigma'])
        (out/f'cpp_{i:02d}.log').write_bytes(raw)
        require(cpp['STATUS'] == 'ready' and all(cpp[tag] == row[key] for tag, key in
                [('D', 'd'), ('EDWARDS_X', 'edwards_x'), ('EDWARDS_Y', 'edwards_y')]), 'C++/GP curve mismatch')
        require(int(cpp['A24'])*4 % int(row['prime']) == (int(row['montgomery_a'])+2) % int(row['prime']), 'a24 mismatch')
        verify_point(row);checked.append(dict(exponent=row['exponent'], sigma=row['sigma'], cpp_match=True, minimal_point_order=True))
    exception = reference['results'][11]
    require(exception['exponent'] == 13883 and int(exception['group_order']) == 3*int(exception['reported_order'])
            and exception['reported_in_hasse_interval'] is False, 'M13883 discrepancy changed')
    example = reference['results'][15];p = int(example['prime']);sigma = int(example['sigma'])
    # Test the independently claimed model; never relabel the original fixture.
    body = f'p={p};s={sigma};A=Mod(4*s,p)/Mod(2^32,p)-2;b=4*A+10;E=ellinit([0,b*A,0,b^2,0]);n=ellcard(E);f=factor(n);print("PARAM3|",n);for(i=1,matsize(f)[1],if(!isprime(f[i,1]),error("unproven factor"));print("F|",f[i,1],"|",f[i,2]));print("OK");quit();\n'
    (out/'param3.gp').write_text(body)
    run = subprocess.run([str(gp), '-q', '-f', '-s', '128M'], input=body.encode(), capture_output=True, check=True, timeout=120)
    (out/'param3_stdout.log').write_bytes(run.stdout);(out/'param3_stderr.log').write_bytes(run.stderr)
    require(order.integer_line(run.stdout.decode().replace('\r', ''), 'PARAM3') == int(example['reported_order']) and
            int(example['group_order']) != int(example['reported_order']), 'M15173 PARAM3 diagnosis changed')
    require(order.parse_factors(run.stdout.decode().replace('\r', ''), 'F') ==
            order.reported_factors(example['reported_order_factorization']) and b'OK' in run.stdout and
            b'***' not in run.stderr, 'PARAM3 factorization mismatch')
    small = []
    for p in [101, 103, 107, 109, 127, 131, 137, 139]:
        for sigma in [17, 19, (1 << 64)-1]:
            folder = out/f'small_{p}_{sigma}';cpp, raw = cpp_curve(probe, p, sigma)
            try:
                result = order.analyze(gp, dict(prime=str(p), sigma=str(sigma)), folder)
            except ValueError:
                require(cpp['STATUS'] != 'ready', 'GP rejected a valid C++ setup')
                (folder/'cpp.log').write_bytes(raw)
                small.append(dict(prime=p, sigma=str(sigma), setup='degenerate'));continue
            (folder/'cpp.log').write_bytes(raw)
            require(cpp['STATUS'] == 'ready', 'GP accepted a C++ setup failure')
            require(all(cpp[tag] == result[key] for tag, key in
                        [('D', 'd'), ('EDWARDS_X', 'edwards_x'), ('EDWARDS_Y', 'edwards_y')]), 'Small-field setup mismatch')
            a, b = int(result['montgomery_a']), int(result['montgomery_b'])
            count = 1
            for u in range(p):
                rhs = (u**3+a*u*u+u)*pow(b, -1, p) % p
                count += 1 if rhs == 0 else 2 if pow(rhs, (p-1)//2, p) == 1 else 0
            require(count == int(result['group_order']), 'Exhaustive small-field point count mismatch')
            verify_point(dict(result, prime=str(p)))
            small.append(dict(prime=p, sigma=str(sigma), setup='ready', group_order=str(count)))
    invalid = []
    for i, case in enumerate([dict(prime='15', sigma='17'), dict(prime='5', sigma='17'),
                              dict(prime='101', sigma='0'), dict(prime='101', sigma='-1'),
                              dict(prime='101', sigma=str(1 << 64)),
                              dict(prime='101', sigma='17', exponent=5)]):
        try:
            order.analyze(gp, case, out/f'invalid_{i}')
        except ValueError:
            invalid.append(case)
        else:
            raise ValueError('Accepted invalid input')
    original = (args.reference/'case_00/stdout.log').read_text();rejected = []
    for tag in ['GROUP', 'POINT', 'D', 'MONTGOMERY_B', 'OK']:
        bad = re.sub(r'^'+tag+r'(?:\|.*)?$', '', original, flags=re.M)
        try:
            order.parse_order(bad, int(reference['results'][0]['prime']))
        except ValueError:
            rejected.append(tag)
        else:
            raise ValueError('Accepted incomplete GP evidence: '+tag)
    report = dict(complete=True, reference_sha256=order.sha(args.reference/'summary.json'),
                  test_sha256=order.sha(Path(__file__)), gp_sha256=order.sha(gp),
                  cpp_probe_sha256=order.sha(probe), reference_checks=checked,
                  small_fields=small, invalid_inputs_rejected=invalid, missing_tags_rejected=rejected,
                  reference_group_matches=15, m13883_group_multiplier='3', m15173_reported_model='montgomery-param3')
    (out/'summary.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(complete=True, cpp_reference_cases=len(checked), small_fields=len(small),
                         small_valid=sum(r['setup'] == 'ready' for r in small), invalid_inputs=len(invalid), corrupt_evidence=len(rejected))))


if __name__ == '__main__':
    main()
