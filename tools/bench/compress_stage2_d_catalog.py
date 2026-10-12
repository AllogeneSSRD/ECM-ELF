"""Propose a smaller literal D catalogue; never run curves or deploy it.

The envelope and boundary rules are coverage heuristics, not dominance proofs.
Inputs, dropped candidates, reasons and the generated header are preserved in
a fresh evidence directory. Native admission and full-curve A/B remain required.
"""
import argparse
import bisect
import csv
import hashlib
import json
from pathlib import Path
import re
import tomllib

from analyze_stage2_d_catalog import prime95_table
from calibrate_stage2_d import phi, shape

ROOT = Path(__file__).resolve().parents[2]
WIDTHS = sorted(set([128, 256, 318, 383, 503, 512, 1009, 1939, 2003,
                     2726, 3001, 3600, 4001, 4667, 5003, 5872, 6011,
                     6797, 7001, 7995, 8011] + list(range(1024, 16385, 512))))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-header', type=Path, default=ROOT / 'src/core/ecm_stage2_d_catalog.h')
    p.add_argument('--prime95-source', type=Path, default=ROOT / '.refactor/p95v3106b01.source/ecm.cpp')
    p.add_argument('--protect-profile', type=Path, action='append', default=[])
    p.add_argument('--protect-analysis', type=Path, action='append', default=[])
    p.add_argument('--bits', type=int, nargs='+', default=WIDTHS)
    p.add_argument('--fold-budgets-mb', type=int, nargs='+', default=[48, 128, 256, 640, 1024])
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if min(a.bits) < 2 or max(a.bits) > 16384 or min(a.fold_budgets_mb) <= 0:
        p.error('bits must be 2..16384; budgets must be positive')
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    text = a.input_header.read_text(encoding='utf-8-sig')
    declaration = re.search(r'(inline constexpr Seed seeds\[\]\s*=\s*\{)(.*?)(\n\};)', text, re.S)
    if not declaration:
        raise ValueError('literal shared catalogue not found')
    seeds = [(int(d), flag == 'true') for d, flag in
             re.findall(r'\{\s*(\d+)\s*,\s*(true|false)\s*\}', declaration[2])]
    ds = [d for d, _ in seeds]
    if not ds or ds != sorted(set(ds)) or any(d < 6 or d % 2 or d > 200000000 for d in ds):
        raise ValueError('input catalogue must be unique ascending even D >= 6')
    flags = dict(seeds)
    degrees = {d: phi(d) // 2 for d in flags}
    representatives = {}
    for d, degree in degrees.items():
        representatives[degree] = max(d, representatives.get(degree, 0))
    ps = sorted(representatives)
    reasons, excluded_protected = {}, []
    protected = set()
    sources = [a.input_header, a.prime95_source, *a.protect_profile, *a.protect_analysis,
               Path(__file__), Path(__file__).with_name('calibrate_stage2_d.py'),
               Path(__file__).with_name('analyze_stage2_d_catalog.py')]

    def include(d, reason, protect=False):
        if d not in flags:
            excluded_protected.append(dict(D=d, reason=reason))
            return
        reasons.setdefault(d, set()).add(reason)
        if protect:
            protected.add(d)

    for d, flag in flags.items():
        if flag:
            include(d, 'default_plateau', True)
    for row in prime95_table(a.prime95_source):
        include(row['D'], 'prime95_active', True)
    for file in a.protect_profile:
        profile = tomllib.loads(file.read_text(encoding='utf-8-sig'))
        for width in profile['width'].values():
            for table in width['table'].values():
                for row in table['matrix'][1:]:
                    fields = {k: v for k, v in table.items() if k != 'matrix'}
                    fields.update(zip(table['matrix'][0], row))
                    if (fields.get('source') == 'measured' and fields.get('clean') == 1
                            and fields.get('bad') == 0 and fields.get('hits') == 0
                            and fields.get('checked', 0) > 0 and fields.get('selftest_cases', 0) > 0
                            and fields.get('repeats', 0) > 0):
                        include(int(fields['d']), 'validated_profile', True)
    for file in a.protect_analysis:
        report = json.loads(file.read_text(encoding='utf-8-sig'))
        if not report['complete'] or report['counts']['failures']:
            raise ValueError('protected analysis must be complete and successful')
        for row in report['summary']:
            include(int(row['D']), 'verified_matrix', True)

    # For every P ceiling retain the greatest D available up to that ceiling.
    # This preserves the geometric memory frontier, not all elapsed-time winners.
    largest_d = 0
    for degree in ps:
        d = representatives[degree]
        if d > largest_d:
            include(d, 'geometric_envelope')
            largest_d = d
    edges = []

    def boundary(predicate, kind, **context):
        lo, hi = 0, len(ps)
        while lo < hi:
            mid = (lo + hi) // 2
            if predicate(ps[mid]):
                lo = mid + 1
            else:
                hi = mid
        adjacent = ps[max(0, lo - 1):min(len(ps), lo + 1)]
        for degree in adjacent:
            include(representatives[degree], kind)
        edges.append(dict(kind=kind, **context, adjacent_P=adjacent))

    def ntt_fits(degree, bits, k, offset):
        try:
            return shape(degree + offset, bits)[0] <= 1 << k
        except ValueError:
            return False

    for bits in sorted(set(a.bits)):
        for k in range(14, 28):
            for offset in (0, 1):
                boundary(lambda degree: ntt_fits(degree, bits, k, offset),
                         'ntt_boundary', bits=bits, log2_length=k, operand_offset=offset)
        words = (bits + 63) // 64
        for budget in sorted(set(a.fold_budgets_mb)):
            for frontier in (False, True):
                # Exact FoldOwnerLayout reuse=3 payload; frontier adds 24*P.
                boundary(lambda degree: 8 * words * (7 * degree + 7) + 48
                         + (24 * degree if frontier else 0) <= budget * 1048576,
                         'owner_boundary', bits=bits, budget_mb=budget,
                         reuse=3, frontier=frontier)
    for k in range(3, 24):
        boundary(lambda degree: degree < 1 << k, 'padded_tree_boundary', log2_degree=k)

    selected = sorted(reasons)
    generated = '\n' + ''.join(f'    {{{d}, {str(flags[d]).lower()}}},\n' for d in selected)
    header = text[:declaration.start(2)] + generated.rstrip('\n') + text[declaration.end(2):]
    (out / 'ecm_stage2_d_catalog.h').write_text(header, encoding='utf-8')
    for name, ds in (('retained', selected), ('removed', sorted(flags.keys() - reasons.keys()))):
        with (out / (name + '.csv')).open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=['D', 'P', 'default_plateau', 'reason'])
            writer.writeheader()
            for d in ds:
                writer.writerow(dict(D=d, P=degrees[d], default_plateau=int(flags[d]),
                                     reason=';'.join(sorted(reasons.get(d, [])))))
    frozen = out / 'sources'
    for file in sources:
        relative = file.resolve().relative_to(ROOT)
        target = frozen / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(file.read_bytes())
    report = dict(semantics='offline coverage heuristic, not strict domination or GPU performance proof',
                  input_count=len(flags), output_count=len(selected), protected_count=len(protected),
                  excluded_protected=excluded_protected, widths=sorted(set(a.bits)),
                  fold_budgets_mb=sorted(set(a.fold_budgets_mb)), owner_reuse=3,
                  sources={str(file.resolve().relative_to(ROOT)): sha(file) for file in sources},
                  generated_header_sha256=sha(out / 'ecm_stage2_d_catalog.h'),
                  origin_counts={reason: sum(reason in r for r in reasons.values())
                                 for reason in sorted(set().union(*reasons.values()))},
                  protected_D=sorted(protected), retained_D=selected, edges=edges,
                  limitations=['finite representative widths and default owner layout',
                               'NTT/owner payload boundaries are not joint process admission',
                               'tail trees and giant-route boundaries depend on B2',
                               'requires native plans and full-curve holdout matrix'])
    (out / 'analysis.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('input_count', 'output_count', 'protected_count',
                                           'excluded_protected', 'origin_counts')}))


if __name__ == '__main__':
    main()
