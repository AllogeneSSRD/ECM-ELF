"""Plot N bit length (x, ticks every 1000 bits) versus complete Stage2 seconds.

Reads the grouped CSV made by analyze_stage2_n_scaling.py. Produces PNG and SVG
for three B2 curves, one-shot Mersenne controls, measured device reduction, and
100% phase charts for cofactor means and intact-Mersenne single-run controls.
"""
import argparse
import csv
import json
from pathlib import Path


def plot_version_comparison(args):
    """Use audited exact-input pairs; CPU points retain their different scope."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    report = json.loads(args.version_comparison.read_text(encoding='utf-8'))
    if not report.get('exact_inputs_verified') or not report.get('equal_scope_verified'):
        raise ValueError('Need audited exact-input version comparison')
    rows = [r for r in report['rows'] if r['variant'] == 'cofactor']
    bounds = sorted({r['B2'] for r in rows})
    xmax = ((max(r['bits'] for r in rows) + args.tick / 2) // args.tick + 1) * args.tick
    cpu = []
    if args.cpu_comparison:
        cpu_report = json.loads(args.cpu_comparison.read_text(encoding='utf-8'))
        cpu = [r for r in cpu_report['exact_pairs'] if r['gpu_variant'] == 'cofactor']
        for pair in cpu:
            row = next(r for r in rows if r['exponent'] == pair['exponent'] and r['B2'] == pair['B2'])
            if row['bits'] != pair['bits'] or abs(row['current_seconds'] - pair['gpu']['mean_seconds']) > 1e-9:
                raise ValueError('CPU comparison belongs to a different GPU study')
    output = args.output.resolve()
    outputs = []
    def save(fig, suffix):
        for ext in ('png', 'svg'):
            path = str(output) + suffix + '.' + ext
            fig.savefig(path, dpi=180, bbox_inches='tight')
            outputs.append(path)
        plt.close(fig)
    fig, axes = plt.subplots(1, len(bounds), figsize=(15, 5), squeeze=False)
    for ax, b2 in zip(axes[0], bounds):
        group = sorted([r for r in rows if r['B2'] == b2], key=lambda r: r['bits'])
        for version, label, color, marker in [('previous', args.previous_label, '#858585', 'o'),
                                               ('current', args.current_label, '#2468a2', 's')]:
            ax.errorbar([r['bits'] for r in group], [r[version + '_seconds'] for r in group],
                        yerr=[r[version + '_std_seconds'] or 0 for r in group],
                        label=label, color=color, marker=marker, markersize=4, capsize=2)
        matched = sorted([r for r in cpu if r['B2'] == b2], key=lambda r: r['bits'])
        if matched:
            ax.errorbar([r['bits'] for r in matched], [r['cpu']['mean'] for r in matched],
                        yerr=[r['cpu']['stdev'] or 0 for r in matched], fmt='^', color='#b65b28',
                        markersize=7, capsize=3, label='Prime95 (exact N only)')
        ax.set_title(f'B2={b2:.1e}')
        ax.set_xlabel('Actual cofactor N (bits)')
        ax.set_ylabel('Mean complete Stage2 time (s)')
        ax.xaxis.set_major_locator(MultipleLocator(args.tick))
        ax.set_xlim(0, xmax)
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Complete Stage2: previous GPU / current GPU / Prime95', fontsize=13)
    fig.text(.01, -.025, 'GPU: identical N, B1, sigma, Stage1 saves and memory budgets; separate sessions, clocks/power may differ. '
             'Independent y scales; error bars: sample SD.\nPrime95: exact N and nominal B2 matched; B1, sigma and actual B2 differ. Source: '
             + args.version_comparison.name, fontsize=8)
    fig.tight_layout()
    save(fig, '_versions_times')
    fig, ax = plt.subplots(figsize=(10, 5))
    for b2, color in zip(bounds, ['#2468a2', '#bd7319', '#49576d']):
        group = sorted([r for r in rows if r['B2'] == b2], key=lambda r: r['bits'])
        ax.plot([r['bits'] for r in group], [r['speedup'] for r in group], 'o-', color=color,
                label=f'B2={b2:.1e}')
    ax.axhline(1, color='#858585', linestyle='--', linewidth=1)
    ax.set_xlabel('Actual cofactor N (bits)')
    ax.set_ylabel('GPU speedup = previous mean / current mean')
    ax.set_title('Current production versus previous production; above 1 means faster')
    ax.xaxis.set_major_locator(MultipleLocator(args.tick))
    ax.set_xlim(0, xmax)
    ax.grid(alpha=.2)
    ax.legend()
    fig.text(.01, -.025, 'Ratios of independent-session sample means, not confidence intervals or isolated optimization effects. '
             'Source: ' + args.version_comparison.name, fontsize=8)
    fig.tight_layout()
    save(fig, '_versions_speedup')
    fig, ax = plt.subplots(figsize=(10, 5))
    for version, label, color, marker in [('previous', args.previous_label, '#858585', 'o'),
                                         ('current', args.current_label, '#2468a2', 's')]:
        for index, b2 in enumerate(bounds):
            group = sorted([r for r in rows if r['B2'] == b2], key=lambda r: r['bits'])
            ax.plot([r['bits'] for r in group], [r[version + '_sm_clock_mean'] for r in group],
                    marker=marker, color=color, linestyle=['-', '--', ':'][index % 3],
                    label=f'{label}, B2={b2:.1e}')
    ax.set_xlabel('Actual cofactor N (bits)')
    ax.set_ylabel('NVML observed SM clock (MHz; sample mean)')
    ax.set_title('Measured clock conditions: no frequency normalization of timings')
    ax.xaxis.set_major_locator(MultipleLocator(args.tick))
    ax.set_xlim(0, xmax)
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
    fig.text(.01, -.025, 'Mean of observed invocation sample means; includes startup/idle observations. '
             'Not a pure-kernel or time-weighted clock. Source: ' + args.version_comparison.name, fontsize=8)
    fig.tight_layout()
    save(fig, '_versions_clocks')
    phases = [
        ('baby_seconds', 'Baby + normalization', '#4e79a7'),
        ('f_tree_init_seconds', 'F tree + init/checks', '#a0cbe8'),
        ('inv_seconds', 'Inverse', '#b07aa1'),
        ('giant_seconds', 'Giant points', '#59a14f'),
        ('gtrees_seconds', 'G trees', '#f28e2b'),
        ('fold_seconds', 'Fold', '#76b7b2'),
        ('gleaves_seconds', 'G-leaf preparation', '#e15759'),
        ('descent_seconds', 'Descent', '#edc949'),
        ('accum_seconds', 'Leaf product / GCD', '#9c755f'),
        ('other_seconds', 'Other', '#bab0ac'),
    ]
    import numpy as np
    fig, axes = plt.subplots(len(bounds), 1, figsize=(12, 10), squeeze=False)
    for ax, b2 in zip(axes[:, 0], bounds):
        group = sorted([r for r in rows if r['B2'] == b2], key=lambda r: r['bits'])
        x = np.arange(len(group))
        positive = np.zeros(len(group))
        negative = np.zeros(len(group))
        phase_total = np.zeros(len(group))
        for field, label, color in phases:
            values = np.array([r[field + '_delta'] for r in group])
            if field == 'accum_seconds':
                values += np.array([r.get('name_seconds_delta', 0) for r in group])
            bottom = np.where(values >= 0, positive, negative)
            ax.bar(x, values, bottom=bottom, color=color, label=label, width=.75)
            positive += np.maximum(values, 0)
            negative += np.minimum(values, 0)
            phase_total += values
        total = np.array([r['current_seconds'] - r['previous_seconds'] for r in group])
        if np.max(np.abs(phase_total - total)) > .003:
            raise ValueError('Version phase deltas do not close to total difference')
        ax.scatter(x, total, marker='_', color='#111111', s=140, label='Net full-time difference', zorder=3)
        ax.axhline(0, color='#777777', linewidth=.7)
        ax.set_xticks(x, [str(r['bits']) for r in group])
        ax.set_xlabel('Actual cofactor N (bits; categorical spacing)')
        ax.set_ylabel('Current minus previous time (s)')
        ax.set_title(f'B2={b2:.1e}; positive = slower, negative = faster', loc='left', fontsize=10)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .015), ncol=4, frameon=False, fontsize=8)
    fig.suptitle('Stage2 module wall-time differences; GPU clock constraints differ', fontsize=13)
    fig.text(.01, -.025, 'Disjoint phases; NTT/S4 event counters remain nested. Black ticks show the net difference. '
             'Source: ' + args.version_comparison.name, fontsize=8)
    fig.subplots_adjust(top=.93, bottom=.17, hspace=.65)
    save(fig, '_versions_phase_delta')
    return outputs


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
    p.add_argument('--version-comparison', type=Path, help='Audited _comparison.json from the existing analysis script')
    p.add_argument('--cpu-comparison', type=Path, help='Exact-N comparison.json for the current GPU study')
    p.add_argument('--previous-label', default='GPU previous')
    p.add_argument('--current-label', default='GPU current')
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
    if a.version_comparison:
        print('Version comparison:', ', '.join(plot_version_comparison(a)))


if __name__ == '__main__':
    main()
