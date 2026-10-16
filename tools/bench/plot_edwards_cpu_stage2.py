"""Plot completed CPU Stage2 samples, preserving min/max and process peak RSS."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summaries', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, default=Path('data/figures/edwards_cpu_stage2'))
    args = p.parse_args();rows = [];identity = {}
    for path in args.summaries:
        r = json.loads(path.read_text())
        assert r['complete'] and r['repeats'] >= 3
        identity[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.extend(r['rows'])
    cells = sorted({(r['bits'], r['b2']) for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)
    for bi, (backend, color) in enumerate([('ntt', '#28677b'), ('plain', '#b06c36')]):
        subset = {(r['bits'], r['b2']): r for r in rows if r['backend'] == backend}
        for metric, ax in zip(('engine_seconds', 'peak_working_set_bytes'), axes):
            scale = 1 if metric == 'engine_seconds' else 2**20
            samples = [[s[metric]/scale for s in subset[c]['samples'] if not s['warmup']] for c in cells]
            values = [statistics.median(s) for s in samples]
            y = [i+(bi-.5)*.28 for i in range(len(cells))]
            ax.barh(y, values, height=.26, color=color, label=backend,
                    xerr=[[v-min(s) for v, s in zip(values, samples)], [max(s)-v for v, s in zip(values, samples)]], capsize=2)
    axes[0].set_xscale('log');axes[0].set_xlabel('Full curve engine time (seconds, log scale)')
    axes[1].set_xlabel('Process peak working set (MiB)')
    axes[0].set_yticks(range(len(cells)), [f'{bits} bits / B2={b2:.0e}' for bits, b2 in cells])
    for ax in axes:
        ax.grid(axis='x', alpha=.2);ax.set_axisbelow(True)
    axes[0].legend()
    fig.suptitle('Edwards CPU Stage2: libecm NTT vs plain polynomial arithmetic')
    fig.text(.5, .01, 'B1=1000, sigma=17, 1 curve, 1 thread, 512 MiB planner budget; median of 3, whiskers min/max; CPU power not measured', ha='center', fontsize=8)
    fig.tight_layout(rect=(0,.035,1,.94));args.output.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'svg'):fig.savefig(args.output/f'backends.{ext}', dpi=170)
    (args.output/'manifest.json').write_text(json.dumps(dict(inputs=identity, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2)+'\n')


if __name__ == '__main__':main()
