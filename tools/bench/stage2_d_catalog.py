"""Read the shared production D table; enumerate the former domain for reference."""
import argparse
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / 'src/core/ecm_stage2_d_catalog.h'
REFERENCE_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47)


def phi(n):
    result = n
    divisor = 2
    while divisor * divisor <= n:
        if n % divisor == 0:
            while n % divisor == 0:
                n //= divisor
            result -= result // divisor
        divisor += 1
    if n > 1:
        result -= result // n
    return result


def candidates(limit=200000000):
    text = HEADER.read_text(encoding='utf-8')
    body = re.search(r'inline constexpr Seed seeds\[\]\s*=\s*\{(.*?)\n\};', text, re.S)
    if not body:
        raise ValueError('shared D catalogue declaration not found')
    values = [int(v) for v in re.findall(r'\{\s*(\d+)\s*,\s*(?:true|false)\s*\}', body[1])]
    if not values or values != sorted(set(values)) or any(d < 6 or d % 2 for d in values):
        raise ValueError('invalid shared D catalogue')
    for d in values:
        if d <= limit:
            yield d, phi(d) // 2


def reference_candidates(limit=200000000):
    """Former 47-smooth domain, including odd D; never a runtime selector."""
    stack = [(0, 1, 1)]
    while stack:
        pos, d, totient = stack.pop()
        if pos == len(REFERENCE_PRIMES):
            if totient // 2:
                yield d, totient // 2
            continue
        prime = REFERENCE_PRIMES[pos]
        stack.append((pos + 1, d, totient))
        first = True
        while d <= limit // prime:
            d *= prime
            totient *= prime - 1 if first else prime
            first = False
            stack.append((pos + 1, d, totient))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--export-reference', type=Path, required=True)
    parser.add_argument('--d-limit', type=int, default=200000000)
    args = parser.parse_args()
    if args.d_limit < 3:
        raise ValueError('d-limit must be at least 3')
    rows = sorted(reference_candidates(args.d_limit))
    args.export_reference.parent.mkdir(parents=True, exist_ok=True)
    with args.export_reference.open('x', encoding='utf-8', newline='\n') as out:
        out.write('# Former native D model domain; reference only, no GPU measurements.\n')
        out.write(f'# primes={REFERENCE_PRIMES}; D<={args.d_limit}; rows={len(rows)}\n')
        out.write('D\tP=phi(D)/2\tportable_even_D_ge_6\n')
        for d, p in rows:
            out.write(f'{d}\t{p}\t{int(d >= 6 and d % 2 == 0)}\n')
    receipt = dict(rows=len(rows), even=sum(d >= 6 and d % 2 == 0 for d, _ in rows),
                   d_limit=args.d_limit, sha256=hashlib.sha256(args.export_reference.read_bytes()).hexdigest(),
                   tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with args.export_reference.with_suffix('.json').open('x', encoding='utf-8') as out:
        json.dump(receipt, out, indent=2)
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
