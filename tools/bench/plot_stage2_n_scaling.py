"""Plot N bit length (x, ticks every 1000 bits) versus complete Stage2 seconds.

Reads the grouped CSV made by analyze_stage2_n_scaling.py. Produces PNG and SVG
for three B2 curves, one-shot Mersenne controls, measured device reduction, and
100% phase charts for cofactor means and intact-Mersenne single-run controls.
"""
import argparse
import csv
from pathlib import Path


def plot_phase_percent(rows, bounds, output, title):
    """Disjoint wall-time phases; nested NTT/S4 counters must not be stacked."""
    import matplotlib.pyplot as plt
    import numpy as np
    phases = [
        ('baby_seconds', 'Baby points + normalization', '#4e79a7'),
        ('f_tree_init_seconds', 'F tree + init setup/checks', '#a0cbe8'),
        ('inv_seconds', 'Polynomial inverse', '#b07aa1'),
        ('giant_seconds', 'Giant points', '#59a14f'),
        ('gtrees_seconds', 'G trees', '#f28e2b'),
        ('fold_seconds', 'Fold', '#76b7b2'),
        ('gleaves_seconds', 'G-leaf preparation / fallback', '#e15759'),
        ('descent_seconds', 'F-tree descent', '#edc949'),
        ('accum_seconds', 'Leaf product + GCD', '#9c755f'),
        ('other_seconds', 'Other / accounting remainder', '#bab0ac'),
    ]
    if not all(all(key in r for key, _, _ in phases) for r in rows):
        return []
    outputs = []
    for variant in ('cofactor', 'mersenne'):
        available = [b2 for b2 in bounds if any(r['variant'] == variant and r['B2'] == b2 for r in rows)]
        if not available:
            continue
        fig, axes = plt.subplots(len(available), 1, figsize=(12, 3.2 * len(available) + 1.3), squeeze=False)
        for ax, b2 in zip(axes[:, 0], available):
            group = sorted([r for r in rows if r['variant'] == variant and r['B2'] == b2], key=lambda r: r['bits'])
            x = np.arange(len(group))
            bottom = np.zeros(len(group))
            for key, label, color in phases:
                seconds = np.array([float(r[key]) for r in group])
                if key == 'accum_seconds':
                    seconds += np.array([float(r.get('name_seconds', 0)) for r in group])
                if np.any(seconds < -0.003):
                    raise ValueError('Negative wall-time partition: ' + key)
                share = 100 * seconds / np.array([r['mean_seconds'] for r in group])
                # Rounded 3-decimal module logs can give tiny negative remainders.
                share = np.maximum(share, 0)
                ax.bar(x, share, bottom=bottom, color=color, width=0.78, label=label)
                rgb = [int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
                text_color = 'white' if sum(c * w for c, w in zip(rgb, (0.2126, 0.7152, 0.0722))) < 0.53 else '#111111'
                for xx, base, value in zip(x, bottom, share):
                    if value >= 9:
                        ax.text(xx, base + value / 2, f'{value:.0f}%', ha='center', va='center', fontsize=8,
                                color=text_color)
                bottom += share
            if np.any(np.abs(bottom - 100) > 0.05):
                raise ValueError('Phase shares do not sum to 100%')
            counts = '/'.join(str(n) for n in sorted({r['samples'] for r in group}))
            ax.set_title(f'B2={b2:.1e}; n={counts}; mean full time ' +
                         f'{min(r["mean_seconds"] for r in group):.2f}–{max(r["mean_seconds"] for r in group):.2f} s',
                         loc='left', fontsize=10)
            ax.set_xticks(x, [str(r['bits']) for r in group])
            ax.set_xlabel('Actual N bits (categorical spacing)')
            ax.set_ylabel('Full Stage2 time (%)')
            ax.set_ylim(0, 100)
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.grid(axis='y', color='#eeeeee', linewidth=0.6)
            ax.set_axisbelow(True)
            ax.spines[['top', 'right']].set_visible(False)
        kind = 'cofactor means' if variant == 'cofactor' else 'intact Mersenne, single-run controls'
        b1s = '/'.join(sorted({r['B1'] for r in rows if r['variant'] == variant}))
        fig.suptitle(title + '\nPhase share — ' + kind + '; B1=' + b1s, fontsize=13)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, fontsize=8,
                   bbox_to_anchor=(0.5, 0.015))
        fig.text(0.015, 0.005, 'Percent of mean seconds, not mean of per-run percentages. '
                 'NTT/S4 are nested and excluded. Other closes the wall-time ledger.', fontsize=8)
        fig.subplots_adjust(top=0.92, bottom=0.13, left=0.08, right=0.985, hspace=0.7)
        suffix = '_phases_percent' + ('_controls' if variant == 'mersenne' else '')
        for extension in ('png', 'svg'):
            name = str(output) + suffix + '.' + extension
            fig.savefig(name, dpi=180, bbox_inches='tight')
            outputs.append(name)
        plt.close(fig)
    return outputs


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='Output filename prefix')
    p.add_argument('--x', choices=('bits', 'exponent'), default='bits')
    p.add_argument('--tick', type=int, default=1000)
    p.add_argument('--title', default='ECM Stage2 on RTX 4060 Laptop GPU')
    a = p.parse_args()
    if a.tick <= 0:
        p.error('--tick must be positive')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    with a.summary.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for key in ('bits', 'exponent', 'B2', 'samples'):
            r[key] = int(r[key])
        for key in ('mean_seconds', 'min_seconds', 'max_seconds', 's4_reduce_seconds'):
            r[key] = float(r[key])
    bounds = sorted({r['B2'] for r in rows})
    output = a.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    colors = ['#2468a2', '#bd7319', '#49576d']
    kinds = ['time', 'reduction']
    if any(r['variant'] == 'mersenne' for r in rows):
        kinds.insert(1, 'controls')
    for kind in kinds:
        fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
        for i, b2 in enumerate(bounds):
            color = colors[i % len(colors)]
            group = sorted([r for r in rows if r['variant'] == 'cofactor' and r['B2'] == b2], key=lambda r: r[a.x])
            x = [r[a.x] for r in group]
            key = 's4_reduce_seconds' if kind == 'reduction' else 'mean_seconds'
            y = [r[key] for r in group]
            counts = '/'.join(str(n) for n in sorted({r['samples'] for r in group}))
            label = f'B2={b2:.1e}, cofactor mean (n={counts})'
            ax.plot(x, y, 'o-', color=color, label=label, linewidth=1.6, markersize=5)
            if kind != 'reduction':
                ax.fill_between(x, [r['min_seconds'] for r in group], [r['max_seconds'] for r in group],
                                color=color, alpha=0.12)
            if kind == 'controls':
                control = sorted([r for r in rows if r['variant'] == 'mersenne' and r['B2'] == b2], key=lambda r: r[a.x])
                ax.plot([r[a.x] for r in control], [r['mean_seconds'] for r in control], 'x--', color=color,
                        label=f'B2={b2:.1e}, intact Mersenne (n=1)', linewidth=1.1, markersize=6)
        xmax = max(r[a.x] for r in rows)
        ax.set_xlim(0, (xmax // a.tick + 1) * a.tick)
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_locator(MultipleLocator(a.tick))
        ax.set_xlabel('N bit length (bits)' if a.x == 'bits' else 'Original Mersenne exponent p (bits before division)')
        ax.set_ylabel('Device S4 reduction event time (s)' if kind == 'reduction' else 'Complete Stage2 time: init + main (s)')
        ax.set_title(a.title + (' — S4 reduction' if kind == 'reduction' else ''))
        ax.grid(axis='both', color='#dddddd', linewidth=0.6)
        ax.spines[['top', 'right']].set_visible(False)
        ax.legend(fontsize=8, frameon=False, loc='upper left')
        b1s = '/'.join(sorted({r['B1'] for r in rows}))
        caption = f'B1={b1s}; GPU1; cofactor x uses actual bits after removing catalog factors. '
        caption += 'Bands show observed min/max, not confidence intervals. ' if kind != 'reduction' else 'S4 is nested inside Stage2; do not add it to wall time. '
        caption += 'Source: ' + a.summary.name
        fig.text(0.01, -0.035, caption, fontsize=8, wrap=True)
        for extension in ('png', 'svg'):
            fig.savefig(str(output) + f'_{kind}.{extension}', dpi=180, bbox_inches='tight')
        plt.close(fig)
    print('Plots:', str(output) + '_{' + ','.join(kinds) + '}.{png,svg}')
    print('Phase shares:', ', '.join(plot_phase_percent(rows, bounds, output, a.title)))


if __name__ == '__main__':
    main()
