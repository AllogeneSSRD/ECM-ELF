"""Prepare validated near-bin cofactors from local PrimeNet HTML, without editing the corpus.

Every removed factor is divided with multiplicity and checked against M_p.
High-width Stage1 points are compared between Python and GMP-ECM. Optional
completed lower-width studies supply unchanged saves, not historical timings.
Outputs are ignored experimental evidence; no GPU execution or hardware changes.
"""
import argparse
from pathlib import Path
import re
import shutil
import sqlite3
import sys
from types import SimpleNamespace

import bench_stage2_n_scaling as scan

if hasattr(sys, 'set_int_max_str_digits'):
    sys.set_int_max_str_digits(0)


def select(html, bins):
    pairs, p = {}, None
    for m in re.finditer(r'id="M(\d+)"|href="/factor/(\d+)"', html.read_text(encoding='utf-8')):
        if m[1]:
            p = int(m[1]); pairs.setdefault(p, set())
        elif p:
            pairs[p].add(int(m[2]))
    candidates = []
    for p, factors in pairs.items():
        if not 2 <= p <= 16384 or not factors:
            continue
        n = (1 << p) - 1
        for f in sorted(factors):
            if f < 2 or n % f:
                raise ValueError(f'Invalid or overlapping divisor M{p}: {f}')
            while n % f == 0:
                n //= f
        if n > 3:
            candidates.append(dict(exponent=p, bits=n.bit_length(), factors=sorted(factors)))
    chosen = []
    for b in bins:
        eligible = [c for c in candidates if c['bits'] <= b and c['exponent'] not in {x['exponent'] for x in chosen}]
        if not eligible:
            raise ValueError('No eligible cofactor at or below bin ' + str(b))
        c = min(eligible, key=lambda c: (b-c['bits'], c['exponent']-c['bits']))
        chosen.append(dict(c, bin=b))
    return chosen


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--html', type=Path, default=scan.ROOT/'.refactor/Mersenne_exponent_factor_10000-19999.html')
    p.add_argument('--bins', type=int, nargs='+', default=[10240, 12288, 14336, 16384])
    p.add_argument('--exe', type=Path, required=True)
    p.add_argument('--gmp-ecm', type=Path, default=Path('D:/code/GIMPS/gmp-ecm/ecm-7.0.5-znver3/ecm.exe'))
    p.add_argument('--lower-study', type=Path, help='Completed, validated study whose cofactor saves are reused')
    p.add_argument('--b2', type=int, nargs='+', default=[26_000_000, 260_000_000])
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if any(not 2 <= b <= 16384 for b in a.bins) or any(b <= 20 for b in a.b2):
        p.error('bins must be 2..16384 and B2 must exceed B1=20')
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    chosen = select(a.html, a.bins)
    # A temporary database lets the existing independent Stage1 preparer be reused.
    # The maintained ecm_dataset database is never opened or changed.
    db_path = out.parent / (out.name + '_input_catalog.sqlite')
    if db_path.exists():
        raise ValueError('Temporary catalog already exists')
    with sqlite3.connect(db_path) as db:
        db.executescript('CREATE TABLE mersennes(exponent INTEGER PRIMARY KEY);'
                         'CREATE TABLE factors(exponent INTEGER,value TEXT);')
        db.executemany('INSERT INTO mersennes VALUES(?)', [(c['exponent'],) for c in chosen])
        db.executemany('INSERT INTO factors VALUES(?,?)',
                       [(c['exponent'], str(f)) for c in chosen for f in c['factors']])
    mon = scan.Monitor(a.device)
    try:
        opts = SimpleNamespace(exe=a.exe, gmp_ecm=a.gmp_ecm, db=db_path,
                               b1=20, sigma=26, b2=a.b2, repeats=1, full_controls=False,
                               exponents=[c['exponent'] for c in chosen], device=a.device,
                               arena_mb=6300, fold_mb=640, batch_mb=256, tune_profile=None)
        data = scan.prepare(opts, out, mon)
        data.update(preparation_only=True, html=str(a.html.resolve()), html_sha256=scan.sha(a.html),
                    selected_bins=chosen, generator_sha256=scan.sha(__file__))
        if a.lower_study:
            old = scan.read(a.lower_study/'measurements.json')
            if not old['complete'] or old['failures'] or old['device'] != mon.identity:
                raise ValueError('Need a completed lower study from this GPU')
            for c in old['cases']:
                if c['variant'] != 'cofactor' or c['exponent'] >= 10000:
                    continue
                if scan.sha(c['save']) != c['save_sha256'] or c['B1'] != 20:
                    raise ValueError('Lower save changed or B1 differs')
                folder = out/'inputs'/c['id']; folder.mkdir()
                save = folder/'stage1.save'; shutil.copy2(c['save'], save)
                data['cases'].append(dict(c, save=str(save)))
            data['lower_reference'] = dict(path=str(a.lower_study.resolve()),
                                           manifest_sha256=scan.sha(a.lower_study/'measurements.json'))
        data['sequence'] = []  # This is preparation, not an unexecuted timing matrix.
        scan.write(out/'measurements.json', data)
        scan.write(out/'selected_bins.json', chosen)
        print('PREPARED', len(data['cases']), 'validated cofactor saves', flush=True)
    finally:
        mon.close()


if __name__ == '__main__':
    main()
