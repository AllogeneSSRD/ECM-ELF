"""Audit D sources and propose a static catalogue, without executing GPU curves.

The generated header is a review artifact, not a deployed selector. Breakpoint
calculations reproduce integer choose_cfg; native complete plans remain the
authority for allocation, route and arithmetic qualification.
"""
import argparse
import ast
import bisect
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tomllib

from calibrate_stage2_d import phi, shape
from stage2_d_catalog import candidates, reference_candidates, HEADER


def product(text):
    def visit(n):
        if isinstance(n, ast.Constant) and type(n.value) is int:
            return n.value
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Mult):
            return visit(n.left) * visit(n.right)
        raise ValueError('expected an integer product in Prime95 D table')
    return visit(ast.parse(text.strip(), mode='eval').body)


def prime95_table(path):
    text = path.read_text(encoding='utf-8-sig')
    # In this upstream snapshot the last declaration is the active table;
    # the preceding declaration is inside ALTERNATE_D_DATA_FOR_POLYS... .
    starts = list(re.finditer(r'struct D_data poly_D_data\[\]\s*=\s*\{', text))
    if len(starts) != 2 or '#endif' not in text[starts[0].end():starts[1].start()]:
        raise ValueError('inspect upstream conditional layout before using this parser')
    start = starts[-1].end()
    rows = []
    for m in re.finditer(r'\{([^{}]+)\}', text[start:text.index('};', start)]):
        fields = m.group(1).split(',')
        if len(fields) != 4:
            raise ValueError('unexpected Prime95 table row')
        d, p, first, second = map(product, fields)
        if p != phi(d)//2:
            raise ValueError('Prime95 numrels does not match phi(D)/2')
        line = text.count('\n', 0, start+m.start())+1
        rows.append(dict(D=d, P=p, first_missing_prime=first,
                         second_missing_prime=second, line=line))
    return rows


def breakpoint(bits, k, limit):
    # Highest polynomial degree P whose fold operand P+1 fits length 2^k.
    lo, hi = 0, limit+1
    while lo+1 < hi:
        mid = (lo+hi)//2
        try:
            fits = shape(mid+1, bits)[0] <= 1 << k
        except ValueError:
            fits = False
        if fits:
            lo = mid
        else:
            hi = mid
    return lo


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prime95-source', type=Path, default=Path('.refactor/p95v3106b01.source/ecm.cpp'))
    p.add_argument('--profile', type=Path, default=Path('build_cuda_cmake/production_stage2/stage2_ecm_tune.toml'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--figures', type=Path, help='Figure prefix; default data/figures/<output name>')
    p.add_argument('--d-limit', type=int, default=200000000)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('use a fresh output directory; retain previous evidence')
    a.output.mkdir(parents=True)
    upstream = prime95_table(a.prime95_source)
    header = HEADER.relative_to(Path.cwd().resolve())
    current = [d for d, _ in candidates(a.d_limit)]
    profile = tomllib.loads(a.profile.read_text(encoding='utf-8'))
    observed = set()
    for width in profile['width'].values():
        for table in width['table'].values():
            for row in table['matrix'][1:]:
                fields = {k: v for k, v in table.items() if k != 'matrix'}
                fields.update(zip(table['matrix'][0], row))
                if fields.get('source') == 'measured':
                    observed.add(int(fields['d']))
    pool = [(d, pb) for d, pb in reference_candidates(a.d_limit) if d >= 6 and d % 2 == 0]
    # This is a candidate-generation heuristic, NOT a proof that other equal-P
    # choices are dominated: baby j magnitude and small-prime handling differ.
    representatives = {}
    for d, pb in pool:
        representatives[pb] = max(d, representatives.get(pb, 0))
    ps = sorted(representatives)
    origins = {}

    def include(d, reason):
        if 6 <= d <= a.d_limit and d % 2 == 0:
            origins.setdefault(d, set()).add(reason)

    for d in current:
        include(d, 'current_catalog')
    for d in observed:
        include(d, 'retained_measurement')
    for row in upstream:
        include(row['D'], 'prime95_active')
        d = row['D']*2
        while d <= a.d_limit:
            include(d, 'prime95_power2_extension')
            d *= 2
    widths = sorted(set([128,256,318,383,503,512,1009,1939,2003,2726,3001,
                         3600,4001,4667,5003,5872,6011,6797,7001,7995,8011]
                        + list(range(1024,16385,512))))
    edges = []
    for bits in widths:
        for k in range(14,28):
            last = breakpoint(bits, k, ps[-1])
            index = bisect.bisect_right(ps, last)
            around = ps[max(0,index-2):min(len(ps),index+2)]
            for pb in around:
                include(representatives[pb], 'fold_ntt_boundary')
            edges.append(dict(arithmetic_bits=bits, log2_ntt=k, max_P=last,
                              adjacent=[dict(D=representatives[pb], P=pb,
                                             fold_log2_ntt=shape(pb+1,bits)[0].bit_length()-1)
                                        for pb in around]))
    for k in range(3,24):
        index = bisect.bisect_left(ps, 1 << k)
        for pb in ps[max(0,index-2):min(len(ps),index+3)]:
            include(representatives[pb], 'padded_tree_boundary')
    rows = [dict(D=d, P=phi(d)//2, origin=sorted(origins[d])) for d in sorted(origins)]
    with (a.output/'catalogue.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['D','P','origin']);writer.writeheader();writer.writerows(rows)
    # A literal table is convenient for review; production adoption requires
    # shared consumers, shape-aware search and separate performance validation.
    generated = ['// Research proposal; not connected to production selection.',
                 '// D and P=phi(D)/2; provenance in catalogue.csv.',
                 'static constexpr struct { unsigned long long d, p; } d_catalogue[] = {']
    generated += [f'    {{{r["D"]}, {r["P"]}}},' for r in rows]
    generated += ['};','']
    (a.output/'catalogue_proposal.h').write_text('\n'.join(generated),encoding='utf-8')
    sources = [a.prime95_source, header, a.profile, Path(__file__),
               Path('src/cuda/ecm_cuda_stage2.cu'),Path('src/cuda/stage2/ntt_runtime.cuh'),
               Path('tools/bench/calibrate_stage2_d.py'),Path('tools/bench/stage2_d_catalog.py')]
    report=dict(semantics='offline catalogue/shape study; no GPU timings or process admission proof',
                git_head=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                sources={str(s):hashlib.sha256(s.read_bytes()).hexdigest() for s in sources},
                current=current,prime95=upstream,observed=sorted(observed),
                counts=dict(current=len(current),prime95=len(upstream),even_smooth=len(pool),
                            distinct_P=len(ps),proposal=len(rows),
                            prime95_overlap=len(set(current)&{r['D'] for r in upstream})),
                widths=widths,edges=edges,proposal=rows)
    for source in sources:
        relative=source.resolve().relative_to(Path.cwd().resolve())
        target=a.output/'sources'/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(source.read_bytes())
    (a.output/'analysis.json').write_text(json.dumps(report,indent=2),encoding='utf-8')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    fig, axes=plt.subplots(1,3,figsize=(15,4.5))
    for ax,(s,c,lo,hi) in zip(axes,[(318,503,20000,250000),(4667,5003,40000,155000),(7995,8011,45000,115000)]):
        points=list(range(lo,hi+1,200))
        for bits,label,color in [(s,f'Target {s} bits','#2468a2'),(c,f'Carrier {c} bits','#d07a22')]:
            ax.step(points,[shape(pb+1,bits)[0].bit_length()-1 for pb in points],where='post',label=label,color=color)
        for d in current:
            pb=phi(d)//2
            if lo<=pb<=hi:
                ax.axvline(pb,color='#888888',alpha=.16,linewidth=.7)
        ax.set_xlabel('P = phi(D)/2');ax.set_ylabel('Fold NTT log2(length)')
        ax.yaxis.set_major_locator(MultipleLocator(1));ax.grid(alpha=.2);ax.legend(fontsize=8)
        if ax.get_ylim()[1]>27.5:
            ax.axhspan(27.5,ax.get_ylim()[1],color='#999999',alpha=.15)
            ax.text(.02,.92,'k=28: outside portable cost range',transform=ax.transAxes,fontsize=8)
    fig.suptitle('Integer fold shape: NTT steps versus polynomial degree')
    fig.text(.01,-.03,f'Grey lines: current {len(current)} D candidates. Offline choose_cfg reproduction for fold operand P+1 only; '
             'inverse/tree, residency, joint memory and runtime costs require native planning.',fontsize=8)
    fig.tight_layout()
    figures=a.figures or Path('data/figures')/a.output.name
    figures.parent.mkdir(parents=True,exist_ok=True)
    for extension in ('png','svg'):
        fig.savefig(str(figures)+'_fold_ntt_steps.'+extension,dpi=180,bbox_inches='tight')
    plt.close(fig)
    print(json.dumps(report['counts']))


if __name__=='__main__':
    main()
