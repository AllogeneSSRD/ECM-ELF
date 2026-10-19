"""Audit D geometry and search regions without running CUDA or predicting seconds.

The geometric envelope is not a wall-time dominance proof. Packing uses the
existing offline integer reproduction; native admission remains authoritative.
All outputs are written to a fresh evidence directory. Production is unchanged.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from calibrate_stage2_d import shape
from stage2_d_catalog import HEADER, ROOT, candidates


def envelope(rows):
    result, largest = [], 0
    for d, p in sorted(rows, key=lambda v: (v[1], -v[0])):
        if d > largest:
            result.append((d, p))
            largest = d
    return result


def geometry(d, p, bits, b2, owner_mb):
    n, bpw, slots = shape(p + 1, bits)
    child = 1 << ((p - 1).bit_length() - 1) if p > 1 else 0
    tree_n = shape(child + 1, bits)[0]
    points = b2 // d + 2
    groups = (points + p - 1) // p
    owner = 8 * ((bits + 63) // 64) * (7 * p + 7) + 48
    return dict(D=d, P=p, D_per_P=d / p, S=bits, B2=b2, I=points, G=groups,
                fold_length=n, fold_bpw=bpw, fold_digits=slots,
                tree_top_length=tree_n, tree_levels=(p - 1).bit_length(),
                owner_bytes=owner, fold_payload_fits=owner <= owner_mb * 1048576,
                frontier_payload_fits=owner + 24 * p <= owner_mb * 1048576,
                proxy_fold_work=(groups - 1) * n * (n.bit_length() - 1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bits', type=int, nargs='+', default=[512, 1939, 4667, 7995, 12011, 16319])
    parser.add_argument('--b2', type=int, nargs='+', default=[260000000, 26000000000, 2600000000000])
    parser.add_argument('--owner-mb', type=int, default=640)
    parser.add_argument('--figures', type=Path, help='Figure prefix; default data/figures/<output name>')
    args = parser.parse_args()
    if any(not 2 <= b <= 16384 for b in args.bits) or min(args.b2) < 2 or args.owner_mb < 0:
        parser.error('invalid arithmetic width, B2 or owner budget')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    rows = list(candidates())
    front = envelope(rows)
    front_ds = {d for d, _ in front}
    summary = dict(semantics='offline exact integer geometry and coverage only; no GPU timings, '
                   'full request costs or native lifecycle admission',
                   git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                   owner_mb=args.owner_mb,
                   counts=dict(catalog=len(rows), distinct_P=len({p for _, p in rows}),
                               envelope=len(front), outside_envelope=len(rows)-len(front),
                               P_drops_in_D_order=sum(a[1] > b[1] for a, b in zip(rows, rows[1:]))),
                   envelope=[dict(D=d, P=p) for d, p in front], scenarios=[], counterexamples=[],
                   frontier_counterexamples=[])
    with (out / 'catalog.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['D', 'P', 'D_per_P', 'geometric_envelope'])
        writer.writerows((d, p, d/p, int(d in front_ds)) for d, p in rows)
    example = []
    for bits in args.bits:
        for b2 in args.b2:
            qualified, regions = [], {}
            for d, p in rows:
                if p < 512 or b2 // d + 2 <= p:
                    continue
                try:
                    g = geometry(d, p, bits, b2, args.owner_mb)
                except ValueError:
                    # The offline packing helper also refuses huge shapes.
                    continue
                if max(g['fold_length'], g['tree_top_length']) > 1 << 27:
                    continue
                qualified.append(g)
                key = (g['fold_length'], g['tree_top_length'], g['tree_levels'],
                       g['fold_payload_fits'], g['frontier_payload_fits'])
                regions.setdefault(key, []).append(g)
            summary['scenarios'].append(dict(S=bits, B2=b2, eligible_geometry=len(qualified),
                geometric_envelope=len(envelope([(g['D'], g['P']) for g in qualified])),
                coarse_regions=len(regions), regions=[dict(key=list(k), size=len(v),
                    min_P=min(g['P'] for g in v), max_P=max(g['P'] for g in v)) for k, v in regions.items()]))
            if bits == 4667 and b2 == 2600000000000:
                example = qualified
                # A geometrically dominates B yet performs more full-size folds.
                # This refutes a componentwise-count argument, not elapsed time.
                for b in qualified:
                    for a in qualified:
                        if (a['D'] > b['D'] and .8*b['P'] < a['P'] < b['P'] and
                                a['fold_length'] == b['fold_length'] and a['G'] > b['G'] >= 3):
                            summary['counterexamples'].append(dict(A=a, B=b,
                                fold_work_ratio=a['proxy_fold_work']/b['proxy_fold_work']))
                summary['counterexamples'].sort(key=lambda e: -e['fold_work_ratio'])
                summary['frontier_counterexamples'] = [e for e in summary['counterexamples']
                    if e['A']['D'] in front_ds and 50000 <= e['A']['P'] < e['B']['P'] <= 140000][:10]
                summary['counterexamples'] = summary['counterexamples'][:10]
    sources = [HEADER, Path(__file__), Path(__file__).with_name('calibrate_stage2_d.py'),
               Path(__file__).with_name('stage2_d_catalog.py'),
               ROOT / 'src/core/ecm_stage2_portable_model.h',
               ROOT / 'src/core/ecm_stage2_runtime_tune.h',
               ROOT / 'src/core/ecm_stage2_portable_driver.inl']
    summary['sources'] = {}
    for src in sources:
        dst = out / 'sources' / src.resolve().relative_to(ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        summary['sources'][str(src.resolve().relative_to(ROOT))] = hashlib.sha256(src.read_bytes()).hexdigest()
    (out / 'analysis.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    if example:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
        ax = axes[0]
        ax.scatter([p for _, p in rows], [d for d, _ in rows], s=9, color='0.7', label=f'Full catalog ({len(rows)})')
        ax.plot([p for _, p in front], [d for d, _ in front], color='#2468a2', label=f'Geometric envelope ({len(front)})')
        ax.set(xscale='log', yscale='log', xlabel='P = phi(D)/2 (coefficients)', ylabel='D (giant step)',
               title=f'Geometry alone retains {len(front)} candidates')
        ax.legend(fontsize=8)
        ax = axes[1]
        local = sorted((g for g in example if 40000 <= g['P'] <= 140000), key=lambda g: g['P'])
        scale = min(g['proxy_fold_work'] for g in local)
        ax.scatter([g['P'] for g in local], [g['proxy_fold_work']/scale for g in local],
                   s=15, color='0.6', label='Catalog geometry')
        selected = [g for g in local if g['D'] in front_ds]
        ax.scatter([g['P'] for g in selected], [g['proxy_fold_work']/scale for g in selected],
                   s=25, color='#2468a2', label='Geometric envelope')
        ax.set(xlabel='P = phi(D)/2 (coefficients)', ylabel='(G-1) L log2(L) / minimum (relative work)',
               title='Fold-only proxy has steps and competing basins')
        ax.legend(fontsize=8)
        for ax in axes:
            ax.grid(alpha=.2)
        fig.text(.01, .01, 'Source: catalog at recorded git HEAD; S=4667 bits, B2=2.6e12 on right. '
                 'No device timing. Proxy omits tail-fold savings, all other phases and memory admission.', fontsize=8)
        fig.tight_layout(rect=(0, .06, 1, 1))
        figures = args.figures or ROOT / 'data/figures' / out.name
        figures.parent.mkdir(parents=True, exist_ok=True)
        for ext in ('png', 'svg'):
            fig.savefig(str(figures) + '_geometry.' + ext, dpi=170)
        plt.close(fig)
    print(json.dumps(dict(counts=summary['counts'], scenarios=[{k: v for k, v in s.items() if k != 'regions'}
          for s in summary['scenarios']], counterexample=summary['counterexamples'][:1])))


if __name__ == '__main__':
    main()
