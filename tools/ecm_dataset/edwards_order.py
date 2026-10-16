"""Compute Edwards Atkin-Morain group/point orders without updating the PARAM0 DB."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import time

from dataset import bounds, gp_path, parse_factors, product

GP_SOURCE = Path(__file__).with_suffix('.gp')
FIXTURES = Path(__file__).with_name('fixtures')/'edwards_mersenne_ca.json'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reported_factors(expression):
    """Parse the supplied integer product, never evaluate GP/Python input."""
    parts = []
    for token in expression.split('*'):
        match = re.fullmatch(r'\s*(\d+)(?:\^(\d+))?\s*', token)
        if not match:
            raise ValueError('Invalid reported integer factorization')
        p, e = int(match[1]), int(match[2] or 1)
        if p < 2 or e < 1 or any(q == p for q, _ in parts):
            raise ValueError('Invalid or duplicate reported factor')
        parts.append((p, e))
    return sorted(parts)


def integer_line(text, tag):
    matches = re.findall(r'^'+tag+r'\|(\d+)$', text, re.M)
    if len(matches) != 1:
        raise ValueError('Missing or duplicate '+tag)
    return int(matches[0])


def parse_order(text, p):
    text = text.replace('\r', '')
    if re.findall(r'^MODEL\|(.*)$', text, re.M) != ['edwards-atkin-morain'] or text.splitlines().count('OK') != 1:
        raise ValueError('Incomplete Edwards GP evidence')
    group, point = integer_line(text, 'GROUP'), integer_line(text, 'POINT')
    gf, pf = parse_factors(text, 'G'), parse_factors(text, 'P')
    if point < 1 or product(gf) != group or product(pf) != point or group % point:
        raise ValueError('Order factorization mismatch')
    if (group-p-1)**2 > 4*p:
        raise ValueError('Group order outside Hasse interval')
    d, x, y, a, b = [integer_line(text, tag) for tag in
                     ['D', 'EDWARDS_X', 'EDWARDS_Y', 'MONTGOMERY_A', 'MONTGOMERY_B']]
    if any(v >= p for v in (d, x, y, a, b)) or d in (0, 1):
        raise ValueError('Invalid curve parameters')
    if (x*x+y*y-1-d*x*x*y*y) % p or (a*(1-d)-2*(1+d)) % p or (b*(1-d)-4) % p:
        raise ValueError('Curve equation/mapping mismatch')
    # Current Edwards Stage1 multiplies the original P by 48*lcm(1..B1).
    effective = point//math.gcd(point, 48)
    effective_parts = []
    for prime, exponent in pf:
        exponent -= min(exponent, 4 if prime == 2 else 1 if prime == 3 else 0)
        if exponent:
            effective_parts.append((prime, exponent))
    if product(effective_parts) != effective:
        raise ValueError('Incorrect 48*lcm adjustment')
    minimum, frontier = bounds(effective_parts)
    return dict(group_order=str(group), point_order=str(point),
                group_factorization=[[str(q), e] for q, e in gf],
                point_factorization=[[str(q), e] for q, e in pf],
                d=str(d), edwards_x=str(x), edwards_y=str(y),
                montgomery_a=str(a), montgomery_b=str(b),
                stage1_scalar='48LCM', order_after_48=str(effective),
                stage1_min_b1=str(minimum),
                single_prime_stage2_frontier=[[str(a), str(b)] for a, b in frontier])


def analyze(gp, case, folder, timeout=120, stack='128M'):
    p, sigma = int(case['prime']), int(case['sigma'])
    if p <= 3 or not 1 <= sigma < 1 << 64:
        raise ValueError('Require prime > 3 and nonzero uint64 sigma')
    if 'exponent' in case:
        exponent = int(case['exponent'])
        if exponent < 1 or pow(2, exponent, p) != 1:
            raise ValueError('Prime does not divide the stated Mersenne number')
    reported = reported_factors(case['reported_order_factorization']) if 'reported_order_factorization' in case else None
    folder.mkdir(parents=True, exist_ok=False)
    body = GP_SOURCE.read_text(encoding='utf-8')+f'\necm_edwards_order({p},{sigma});\nquit();\n'
    (folder/'input.gp').write_text(body, encoding='utf-8')
    start = time.monotonic()
    try:
        run = subprocess.run([str(gp), '-q', '-f', '-s', stack], input=body.encode(),
                             capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        (folder/'stdout.log').write_bytes(error.stdout or b'')
        (folder/'stderr.log').write_bytes(error.stderr or b'')
        raise
    (folder/'stdout.log').write_bytes(run.stdout)
    (folder/'stderr.log').write_bytes(run.stderr)
    text = run.stdout.decode('utf-8', errors='replace').replace('\r', '')
    err = run.stderr.decode('utf-8', errors='replace')
    if run.returncode or '***' in err:
        raise ValueError('GP failed: '+(err+text)[-1600:])
    result = parse_order(text, p)
    result.update(status='complete', seconds=time.monotonic()-start,
                  stdout_sha256=sha(folder/'stdout.log'), stderr_sha256=sha(folder/'stderr.log'))
    if reported is not None:
        claimed = product(reported); group = int(result['group_order']); point = int(result['point_order'])
        relation = ('group_order' if claimed == group else 'point_order' if claimed == point
                    else 'proper_divisor_of_group_order' if group % claimed == 0 else 'mismatch')
        result.update(reported_order=str(claimed), reported_relation=relation,
                      reported_group_multiplier=str(group//claimed) if group % claimed == 0 else None,
                      reported_in_hasse_interval=(claimed-p-1)**2 <= 4*p)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gp', nargs='?', help='Explicit executable; omitted or bare --gp searches PATH')
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--prime', type=int, help='Proven prime divisor, rather than the full composite target')
    source.add_argument('--fixtures', type=Path, help='JSON file with model and cases, e.g. '+str(FIXTURES))
    parser.add_argument('--sigma', type=int)
    parser.add_argument('--exponent', type=int, help='Optional Mersenne divisibility check')
    parser.add_argument('--timeout', type=float, default=120, help='Seconds per curve')
    parser.add_argument('--stack', default='128M', help='Initial PARI stack, e.g. 128M')
    parser.add_argument('--output', type=Path, required=True, help='New evidence directory; never overwritten')
    args = parser.parse_args()
    if args.timeout <= 0 or not re.fullmatch(r'[1-9]\d*[kKmMgG]?', args.stack):
        parser.error('Require a positive timeout and GP stack size')
    if args.prime is not None:
        if args.sigma is None:
            parser.error('--prime requires --sigma')
        cases = [dict(prime=str(args.prime), sigma=str(args.sigma))]
        if args.exponent is not None:
            cases[0]['exponent'] = args.exponent
    else:
        if args.sigma is not None or args.exponent is not None:
            parser.error('--fixtures supplies its own sigma/exponent')
        fixture = json.loads(args.fixtures.read_text(encoding='utf-8'))
        if fixture['model'] != 'edwards-atkin-morain' or not fixture['cases']:
            parser.error('Need nonempty Atkin-Morain Edwards cases')
        cases = fixture['cases']
    gp = gp_path(args.gp); out = args.output.resolve();out.mkdir(parents=True, exist_ok=False)
    for path in (GP_SOURCE, Path(__file__), Path(__file__).with_name('dataset.py')):
        shutil.copy2(path, out/path.name)
    if args.fixtures:
        shutil.copy2(args.fixtures, out/'fixtures.json')
    version = subprocess.run([str(gp), '--version-short'], capture_output=True, timeout=10, check=True)
    (out/'gp_version.log').write_bytes(version.stdout+version.stderr)
    runtime = {gp.name: sha(gp)}
    if (gp.parent/'libpari.dll').is_file():
        runtime['libpari.dll'] = sha(gp.parent/'libpari.dll')
    report = dict(complete=False, model='edwards-atkin-morain', gp=str(gp), gp_sha256=runtime,
                  source_sha256={p.name: sha(p) for p in (GP_SOURCE, Path(__file__), Path(__file__).with_name('dataset.py'))},
                  fixture_sha256=sha(args.fixtures) if args.fixtures else None,
                  timeout=args.timeout, stack=args.stack, results=[])
    def persist():
        (out/'summary.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    persist()
    for i, case in enumerate(cases):
        row = dict(case)
        try:
            row.update(analyze(gp, case, out/f'case_{i:02d}', args.timeout, args.stack))
        except (ValueError, subprocess.TimeoutExpired) as error:
            row.update(status='timeout' if isinstance(error, subprocess.TimeoutExpired) else 'error', error=str(error))
        report['results'].append(row);persist()
        print(f"{i+1}/{len(cases)} {row['status']} M{row.get('exponent','?')} sigma={row['sigma']} reference={row.get('reported_relation','none')}", flush=True)
    report['complete'] = all(r['status'] == 'complete' for r in report['results'])
    report['reported_relations'] = {name: sum(r.get('reported_relation') == name for r in report['results'])
                                    for name in ['group_order', 'point_order', 'proper_divisor_of_group_order', 'mismatch']}
    compared = [r for r in report['results'] if 'reported_order_factorization' in r]
    report['all_reported_group_orders_match'] = all(r.get('reported_relation') == 'group_order' for r in compared) if compared else None
    persist()
    with (out/'orders.csv').open('w', encoding='utf-8', newline='') as target:
        fields = ['exponent', 'prime', 'sigma', 'status', 'group_order', 'point_order',
                  'reported_order', 'reported_relation', 'reported_group_multiplier', 'stage1_min_b1']
        writer = csv.DictWriter(target, fields, extrasaction='ignore');writer.writeheader();writer.writerows(report['results'])
    print(json.dumps(dict(complete=report['complete'], reported_relations=report['reported_relations'], output=str(out))))
    return 0 if report['complete'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
