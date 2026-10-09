"""Prepare small carrier benchmark saves and independent target-ring oracles.

No GPU execution. Uses the existing Python Montgomery reference for Stage1,
every baby/giant coordinate, known denominator factors and monic leaf values.
M-domain denominator observations demonstrate why inverses must target N.
"""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fnv(values, words):
    h = 1469598103934665603
    mask = (1 << 64)-1
    for v in values:
        for i in range(words):
            h = ((h ^ ((v >> (64*i)) & mask))*1099511628211) & mask
    return str(h)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    out = a.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('use a fresh output directory')
    out.mkdir(parents=True, exist_ok=True)
    path = ROOT/'tools/stat/suyama_mont_ref.py'
    spec = importlib.util.spec_from_file_location('carrier_ref', path)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    d, count, b1 = 210, 65, 2
    cases = []
    # The final target is 2^8192+1, carried by M16384. This exercises maximum
    # limb count and the p%64==0 reduction boundary with a valid Stage1 point.
    for exponent, removed in ((37, 223), (67, 193707721), (29, 233), (253, 23),
                              (16384, (1 << 8192)-1)):
        m = (1 << exponent)-1
        if m % removed:
            raise ValueError('carrier/target factorization changed')
        n = m // removed
        point = None
        for sigma in range(26, 200):
            try:
                candidate = ref.stage1(sigma, b1, n)
                if candidate['gcd'] == 1 and math.gcd(candidate['x'], n) == 1:
                    point = candidate
                    break
            except ValueError:
                continue
        if point is None:
            raise ValueError(f'no valid Stage1 fixture for exponent {exponent}')
        save = out/f'm{exponent}_cofactor.save'
        save.write_text(f'METHOD=ECM; PARAM=0; SIGMA={sigma}; B1={b1}; N={n}; '
                        f'X=0x{point["x"]:x}; CHECKSUM={b1*sigma*n*point["x"]%4294967291};\n', encoding='utf-8')
        baby, giant, nonunits, removed_nonunits = [], [], [], []
        indices = [('baby', j) for j in range(1, d//2+1) if math.gcd(j, d) == 1]
        indices += [('giant', i*d) for i in range(1, count+1)]
        for kind, k in indices:
            x, z = ref.ladder(k, point['x'], 1, point['a24'], n)
            g = math.gcd(z, n)
            if g != 1:
                nonunits.append(dict(kind=kind, scalar=k, gcd=g))
            else:
                (baby if kind == 'baby' else giant).append(x*pow(z, -1, n) % n)
            # This is the same ordinary lifted curve/Q used by the carrier path.
            _, mz = ref.ladder(k, point['x'], 1, point['a24'], m)
            mg = math.gcd(mz, m)
            if g == 1 and mg != 1:
                removed_nonunits.append(dict(kind=kind, scalar=k, gcd_M=mg, gcd_N=g))
        expected = None
        if not nonunits:
            values = []
            for b in baby:
                v = 1
                for g in giant:
                    v = v*(b-g) % n
                values.append(v)
            expected = dict(target_bits=n.bit_length(), leaves=str(len(values)),
                            words=str(len(values)*((n.bit_length()+63)//64)),
                            nonzero=str(sum(v != 0 for v in values)),
                            hash=fnv(values, (n.bit_length()+63)//64))
        cases.append(dict(exponent=exponent, removed=removed, N_hex=format(n, 'x'),
                          B1=b1, B2=d*(count-2), D=d, sigma=sigma,
                          save=str(save), save_sha256=sha(save),
                          unit=not nonunits, nonunits=nonunits,
                          removed_only_nonunits=removed_nonunits, expected_leaf=expected))
    result = dict(complete=True, reference=str(path), reference_sha256=sha(path),
                  generator_sha256=sha(__file__), cases=cases)
    (out/'fixtures.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps([dict(exponent=c['exponent'], sigma=c['sigma'], unit=c['unit'],
                           target_nonunits=len(c['nonunits']),
                           removed_only_nonunits=len(c['removed_only_nonunits']),
                           leaf=c['expected_leaf']) for c in cases], indent=2))


if __name__ == '__main__':
    main()
