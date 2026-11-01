"""Re-read frozen CUDA matrices and describe bit-width scaling, without GPU work.

Empirical log-log slopes describe the measured range, not an asymptotic proof.
The production-shaped held-out off/on calls remain separate populations. The
small fixed-D matrix is a separate control, including per-process self-tests.
"""
import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import re
import shutil
import statistics as stats
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/ecm_report'))
from result_format import normalize_result
PHASES = ['baby', 'giant_combined', 'ftree_init', 'inv', 'gtrees', 'fold', 'descent', 'accum', 'other']
LABELS = ['Baby', 'Giant + leaf prep / waits', 'F tree / init other', 'Inverse',
          'G trees', 'Fold', 'Descent', 'Accum / GCD', 'Other']
COLORS = ['#43aa8b', '#8d72b4', '#b8d8ba', '#d9bd6a', '#397db6', '#de9858', '#acc977', '#d58d9f', '#bec7d0']


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fields(text, label):
    rows = re.findall(r'^' + re.escape(label) + r': (.*)$', text, re.M)
    if not rows:
        raise ValueError('Missing evidence: ' + label)
    return dict(re.findall(r'(\w+)=([^\s]+)', rows[-1]))


def partitions(total, init, baby, split, leaves):
    p = dict(baby=float(baby['ladder']) + float(baby['affine']),
             inv=float(split['inv']), gtrees=float(split['gtrees']),
             fold=float(split['fold']), descent=float(split['descent']), accum=float(split['accum']),
             giant_combined=float(split['giant']) + leaves)
    p['ftree_init'] = init - p['baby']
    p['other'] = total - sum(p.values())
    if min(p.values()) < -.012:
        raise ValueError('Phase timers overlap beyond their printed precision')
    # Preserve exact closure when rounded timers leave a few negative millisec.
    for key in p:
        if p[key] < 0:
            deficit = -p[key]
            p[key] = 0
            donor = max(p, key=p.get)
            p[donor] -= deficit
    if abs(sum(p.values()) - total) > 1e-7:
        raise ValueError('Phase partition does not close')
    return p


def workload(text, bits):
    rows = re.findall(r'^ntt_length_counts: log2_length:num_of_ntt (.*)$', text, re.M)
    if len(rows) != 1:
        raise ValueError('Need exactly one completed NTT distribution')
    counts = {int(k): int(v) for k, v in re.findall(r'(\d+):(\d+)', rows[0])}
    if not counts or any(v % 3 for v in counts.values()):
        raise ValueError('Invalid forward/inverse distribution')
    w = (bits + 63) // 64
    memory = fields(text, 'ntt_workspace_stats')
    s4 = fields(text, 's4_multiply_stats')
    point = fields(text, 'point_mersenne_mode')
    return dict(words=w, capacity_words=next(n for n in (4, 8, 16, 32, 64, 128, 256) if n >= w),
                ntt_counts=counts, max_log2_length=max(counts),
                butterflies=sum(v * (1 << k) * k / 2 for k, v in counts.items()),
                transforms=sum(counts.values()), coeffs_reduced=int(s4['coeffs_reduced']),
                reduce_device_seconds=float(s4['t_reduce']),
                ntt_module_peak_bytes=int(memory['full_peak_bytes']),
                point_mersenne=bool(int(point['enabled'])))


def primary(study):
    plan = load(study / 'plan.json')
    if not plan['complete'] or plan['failures'] or len(plan['evaluations']) != 52:
        raise ValueError('Need the completed 26-cell held-out point matrix')
    if sha(study / 'bin/ecm_cuda_stage2.exe') != plan['exe_sha256'] or sha(study / 'frozen_tune.toml') != plan['frozen_sha256']:
        raise ValueError('Frozen primary program/profile changed')
    cases = {c['id']: c for c in plan['cases']}
    cells = {c['id']: c for c in plan['cells']}
    result = []
    for entry in plan['evaluations']:
        cell = cells[entry['cell']]
        case = cases[cell['case']]
        folder = Path(entry['folder'])
        row = entry['row']
        raw = [json.loads(l) for l in (folder / 'result.jsonl').read_text(encoding='utf-8-sig').splitlines() if l.strip()]
        if len(raw) != 1 or raw[0]['source-build'] != plan['exe_sha256'] or raw[0]['status'] != 'NF':
            raise ValueError('Unverified primary receipt')
        if raw[0]['program']['version'] != '0.56.0':
            raise ValueError('Unexpected primary version; keep populations separate')
        receipt = normalize_result(raw[0])
        expected = dict(N_hex=case['N_hex'], sigma=case['sigma'], B1=case['b1'],
                        B2=cell['B2'], carrier_exponent=cell['carrier'])
        if any(receipt[k] != v for k, v in expected.items()) or receipt['factors'] or receipt['bad_factors']:
            raise ValueError('Primary input or factor receipt differs')
        if raw[0]['d'] != cell['D'] or raw[0]['poly-size'] != cell['P']:
            raise ValueError('Primary D/P changed')
        if sha(case['save']) != case['save_sha256']:
            raise ValueError('Primary save changed')
        text = '\n'.join((folder / n).read_text(encoding='utf-8') for n in ('engine.log', 'debug.log'))
        total = float(row['seconds'])
        clock = fields(text, 'stage2_full_wall')
        if abs(float(clock['total']) - total) > 1e-8:
            raise ValueError('Captured clock differs from raw log')
        phase = partitions(total, float(clock['init']), row['stats']['real_baby'],
                           row['stats']['real_batched_split'], float(row['stats']['real_batched_wall']['gleaves']))
        arithmetic = cell['carrier'] or case['bits']
        work = workload(text, arithmetic)
        if bool(work['point_mersenne']) != bool(cell['carrier']):
            raise ValueError('Point arithmetic mode differs from the family label')
        s4 = fields(text, 's4_multiply_stats')
        if int(s4['gmp_check_bad']) or int(s4['gmp_selftest_bad']) or not int(s4['gmp_checked']):
            raise ValueError('Arithmetic gate failed')
        result.append(dict(family='carrier' if cell['carrier'] else 'ordinary', target_bits=case['bits'],
                           arithmetic_bits=arithmetic, B1=case['b1'], B2=cell['B2'], D=cell['D'], P=cell['P'],
                           I=cell['B2'] // cell['D'] + 2, chain=entry['chain'],
                           tail_permission=entry['balance'], seconds=total, phases=phase,
                           point_chunks=len(row['policies']), policies=row['policies'],
                           point_seconds=phase['baby'] + phase['giant_combined'],
                           polynomial_seconds=sum(phase[k] for k in ('ftree_init', 'inv', 'gtrees', 'fold', 'descent')),
                           **work, evidence=str(folder)))
    for r in result:
        other = next(x for x in result if x['family'] == r['family'] and x['target_bits'] == r['target_bits'] and x['tail_permission'] != r['tail_permission'])
        if r['policies'] != other['policies'] or r['ntt_counts'] != other['ntt_counts']:
            raise ValueError('The no-op sensitivity control changed work')
    return plan, sorted(result, key=lambda r: (r['family'], r['target_bits'], r['tail_permission']))


def fixed_control(study):
    manifest = load(study / 'completed.json')
    if not manifest['complete'] or manifest['D'] != 2310 or manifest['repeats'] != 3:
        raise ValueError('Unqualified fixed-D study')
    result = []
    for line in (study / 'measurements.jsonl').read_text(encoding='utf-8').splitlines():
        r = json.loads(line)
        if r['backend'] != 'cuda':
            continue
        log = ROOT / r['log']
        if sha(log) != r['log_sha256'] or sha(log.parent / 'results.jsonl') != r['result_sha256']:
            raise ValueError('Fixed-D evidence changed')
        text = log.read_text(encoding='utf-8')
        raw = [json.loads(l) for l in (log.parent / 'results.jsonl').read_text(encoding='utf-8-sig').splitlines() if l.strip()]
        if len(raw) != 1 or raw[0]['status'] != 'NF' or raw[0]['source-build'] != r['source_build']:
            raise ValueError('Fixed-D result identity differs')
        receipt = normalize_result(raw[0])
        if receipt['B2'] != r['B2'] or receipt['carrier_exponent'] != r['carrier_exponent'] or receipt['factors'] or receipt['bad_factors']:
            raise ValueError('Fixed-D input/factors differ')
        if raw[0]['d'] != r['D'] or raw[0]['poly-size'] != r['P'] or abs(float(fields(text, 'stage2_full_wall')['total']) - r['engine_seconds']) > 1e-8:
            raise ValueError('Fixed-D geometry/clock differs')
        b = fields(text, 'real_baby')
        split = fields(text, 'real_batched_split')
        phase = partitions(r['engine_seconds'], r['init_seconds'], b, split, 0)
        result.append(dict(family='carrier' if r['mode'] == 'mersenne' else 'ordinary',
                           target_bits=r['bits'], arithmetic_bits=r['carrier_exponent'] or r['bits'],
                           B2=r['B2'], D=r['D'], P=r['P'], seconds=r['engine_seconds'],
                           phases=phase, source_build=r['source_build'], evidence=str(log)))
    if len(result) != 156:
        raise ValueError('Need all 156 CUDA fixed-D observations')
    for family in ('ordinary', 'carrier'):
        for bound in (260000, 2600000):
            group = [r for r in result if r['family'] == family and r['B2'] == bound]
            counts = {b: sum(r['target_bits'] == b for r in group) for b in {r['target_bits'] for r in group}}
            if len(counts) != 13 or set(counts.values()) != {3}:
                raise ValueError('Missing/duplicate fixed-D repetitions')
    return manifest, result


def slope(rows, key, lower):
    r = [x for x in rows if x['arithmetic_bits'] >= lower]
    x = np.log([v['arithmetic_bits'] for v in r])
    y = np.log([v[key] for v in r])
    p, a = np.polyfit(x, y, 1)
    residual = y - (p * x + a)
    return dict(exponent=float(p), intercept=float(a), n=len(r), widths=len(set(v['arithmetic_bits'] for v in r)),
                r2_log=float(1 - np.sum(residual ** 2) / np.sum((y - y.mean()) ** 2)),
                minimum_bits=min(v['arithmetic_bits'] for v in r), maximum_bits=max(v['arithmetic_bits'] for v in r))


def describe(rows, fixed):
    result = dict(fits=[], pairs=[], highest=[])
    for family in ('ordinary', 'carrier'):
        r = [x for x in rows if x['family'] == family and not x['tail_permission']]
        on = [x for x in rows if x['family'] == family and x['tail_permission']]
        for lower in (1939, 7995, 10000):
            for key in ('seconds', 'point_seconds', 'polynomial_seconds'):
                result['fits'].append(dict(study='production-shaped off', family=family, metric=key,
                                           **slope(r, key, lower)))
            result['fits'].append(dict(study='production-shaped on sensitivity', family=family, metric='seconds',
                                       **slope(on, 'seconds', lower)))
        a = next(x for x in r if x['target_bits'] == 7995)
        b = next(x for x in r if x['target_bits'] == 16272)
        result['pairs'].append(dict(family=family, from_bits=a['arithmetic_bits'], to_bits=b['arithmetic_bits'],
                                   time_ratio=b['seconds'] / a['seconds'], word_squared_ratio=(b['words'] / a['words']) ** 2,
                                   ntt_butterfly_ratio=b['butterflies'] / a['butterflies'],
                                   coeff_ratio=b['coeffs_reduced'] / a['coeffs_reduced'],
                                   empirical_exponent=math.log(b['seconds'] / a['seconds']) / math.log(b['arithmetic_bits'] / a['arithmetic_bits'])))
        result['highest'].append(dict(family=family, total=b['seconds'], phases=b['phases'],
                                     point_share=b['point_seconds'] / b['seconds'],
                                     polynomial_share=b['polynomial_seconds'] / b['seconds'],
                                     nested_reduce_share=b['reduce_device_seconds'] / b['seconds'],
                                     ideal_twice_points=1 / (1 - .5 * b['point_seconds'] / b['seconds'])))
        for bound in sorted({x['B2'] for x in fixed}):
            for lower in (1939, 7995, 10000):
                result['fits'].append(dict(study='fixed-D separate control', family=family, B2=bound, metric='seconds',
                                           **slope([x for x in fixed if x['family'] == family and x['B2'] == bound], 'seconds', lower)))
    return result


def plots(rows, fixed, figures):
    figures.mkdir(parents=True, exist_ok=True)
    files = []
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, family in zip(axes, ('ordinary', 'carrier')):
        r = [x for x in rows if x['family'] == family and not x['tail_permission']]
        x = np.array([v['target_bits'] for v in r])
        y = np.array([v['seconds'] for v in r])
        ax.plot(x, y, '-o', label='Frozen D/layout/chain; n=1')
        anchor = next(v for v in r if v['target_bits'] == 7995)
        quadratic = anchor['seconds'] * (np.array([v['words'] for v in r]) / anchor['words']) ** 2
        ax.plot(x, quadratic, '--', label='W squared, anchored at 7995 target bits', color='#999')
        ax.set(xlabel='Target modulus bits (carrier arithmetic uses p)', ylabel='Engine init + main (s)', title=family)
        ax.set_xticks(range(0, 17000, 1000));ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('RTX4060 Laptop | 0.56.0 held-out small-B2 matrix | B2 ~ 2.70e10')
    files.append(('times', fig))
    fig, axes = plt.subplots(2, 2, figsize=(16, 9))
    for col, family in enumerate(('ordinary', 'carrier')):
        r = [x for x in rows if x['family'] == family and not x['tail_permission']]
        for row in (0, 1):
            ax = axes[row, col];bottom = np.zeros(len(r))
            for key, label, color in zip(PHASES, LABELS, COLORS):
                values = np.array([v['phases'][key] for v in r])
                if row:
                    values = values / np.array([v['seconds'] for v in r]) * 100
                ax.bar(range(len(r)), values, bottom=bottom, color=color, label=label);bottom += values
            ax.set_xticks(range(len(r)), [str(v['target_bits']) for v in r], rotation=50, fontsize=8)
            ax.set(ylabel='Time share (%)' if row else 'Engine seconds', title=family if not row else '')
            if row:ax.set_ylim(0, 100)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper center', ncol=5, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, .93));files.append(('phases', fig))
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for family, color in [('ordinary', '#2a75ab'), ('carrier', '#d0873a')]:
        r = [x for x in rows if x['family'] == family and not x['tail_permission'] and x['target_bits'] >= 1939]
        x = [v['arithmetic_bits'] for v in r]
        anchor = next(v for v in r if v['target_bits'] == 7995)
        for key, style, label in [('point_seconds', '-o', 'point pipeline'), ('butterflies', '--', 'NTT butterfly work')]:
            axes[0].plot(x, [v[key] / anchor[key] for v in r], style, color=color, label=family + ' ' + label)
        axes[1].step(x, [v['max_log2_length'] for v in r], where='mid', label=family, color=color)
        bound = 2600000
        control = [v for v in fixed if v['family'] == family and v['B2'] == bound and v['target_bits'] >= 1939]
        bits = sorted({v['arithmetic_bits'] for v in control})
        times = [stats.mean(v['seconds'] for v in control if v['arithmetic_bits'] == b) for b in bits]
        axes[2].loglog(bits, times, '-o', color=color, label=family + ' n=3')
    axes[0].set(xlabel='Arithmetic bits', ylabel='Ratio to same-family 7995-bit target', title='Work and point-pipeline growth')
    axes[1].set(xlabel='Arithmetic bits', ylabel='Largest log2 NTT length', title='Actual packing length steps')
    axes[2].set(xlabel='Arithmetic bits (log)', ylabel='Complete engine s (log)', title='Separate fixed D=2310 / P=240 control')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.tight_layout();files.append(('work', fig))
    for name, fig in files:
        for ext in ('png', 'svg'):
            fig.savefig(figures / ('stage2-bit-complexity-' + name + '.' + ext), dpi=160)
        plt.close(fig)


def webpage(summary, rows, figures, destination):
    highest = summary['analysis']['highest']
    cards = ''.join('<article><h2>' + h['family'] + ' · 16272 target bits</h2><strong>' +
                    f"{h['total']:.2f} s</strong><p>Points + leaf preparation: {100*h['point_share']:.1f}%</p>" +
                    f"<p>2x faster point pipeline: theoretical whole-curve speedup {h['ideal_twice_points']:.2f}x</p></article>" for h in highest)
    body = ''
    for key, title in [('times', '时间与平方参照'), ('phases', '阶段秒数与百分比'), ('work', '工作量、长度拐点与固定D对照')]:
        svg = (figures / ('stage2-bit-complexity-' + key + '.svg')).read_text(encoding='utf-8')
        body += '<section><h2>' + title + '</h2>' + svg[svg.index('<svg'):] + '</section>'
    table = ''.join('<tr>' + ''.join('<td>' + html.escape(str(v)) + '</td>' for v in
                    (r['family'], r['target_bits'], r['arithmetic_bits'], r['D'], r['P'], r['chain'], f"{r['seconds']:.3f}",
                     f"{100*r['point_seconds']/r['seconds']:.1f}%", '2^' + str(r['max_log2_length']))) + '</tr>'
                    for r in rows if not r['tail_permission'])
    document = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width">' + \
        '<title>Stage2 位宽复杂度分析</title><style>body{font:16px system-ui;background:#f3f5f9;color:#19263c;margin:0 auto;max-width:1350px;padding:28px}' + \
        'article,section{background:white;border-radius:12px;padding:20px;margin:18px 0}.cards{display:flex;gap:20px}.cards article{flex:1}strong{font-size:30px}' + \
        'svg{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:right}a{color:#2567a6}</style>' + \
        '<h1>Stage2 为什么在高位宽变慢？</h1><p>由已有冻结证据生成；不启动GPU，不调整设备。正式说明：' + \
        '<a href="../performance/STAGE2_BIT_SCALING.md">位宽性能与优化</a>；<a href="../reference/ECM_STAGE2_BIT_COMPLEXITY.md">数学工作量</a>。</p>' + \
        '<p>点模乘仍为 O(W²)。梅森承载减少归约工作，但没有消除乘法的平方项。NTT为O(L log L)，不是O(L²)。' + \
        '实际耗时还受模板容量、分块、驻留、等待和D/P变化影响。拟合指数仅描述已测范围。</p><div class="cards">' + cards + '</div>' + body + \
        '<section><h2>生产形状独立样本</h2><p>0.56.0，4060LP；每格off侧n=1，不与学习样本或小B2固定D样本合并。</p><table>' + \
        '<thead><tr><th>Family</th><th>Target bits</th><th>Arithmetic bits</th><th>D</th><th>P</th><th>Chain</th><th>Seconds</th><th>Point share</th><th>Largest NTT</th></tr></thead><tbody>' + table + '</tbody></table></section></html>'
    destination.parent.mkdir(parents=True, exist_ok=True)
    # SVG path attributes tolerate newlines; strip Matplotlib's trailing blanks
    # so the embedded, versioned HTML also passes repository whitespace checks.
    document = '\n'.join(line.rstrip() for line in document.splitlines()) + '\n'
    destination.write_text(document, encoding='utf-8', newline='\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study', type=Path, required=True)
    p.add_argument('--fixed-study', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--figures', type=Path, required=True)
    p.add_argument('--html', type=Path)
    p.add_argument('--public-figures', type=Path)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    plan, rows = primary(a.study)
    manifest, fixed = fixed_control(a.fixed_study)
    summary = dict(complete=True, timing='engine init+main; outer selection/read/startup/publication excluded',
                   primary_study=str(a.study), primary_sha256=plan['exe_sha256'], primary_version='0.56.0',
                   primary_calls=len(rows), fixed_study=str(a.fixed_study), fixed_calls=len(fixed),
                   fixed_sha256=sorted({r['source_build'] for r in fixed}),
                   analysis=describe(rows, fixed), rows=rows, fixed_rows=fixed)
    (a.output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    columns = ['family', 'target_bits', 'arithmetic_bits', 'words', 'capacity_words', 'B2', 'D', 'P', 'I',
               'chain', 'point_chunks', 'tail_permission', 'seconds', 'point_seconds', 'polynomial_seconds',
               'butterflies', 'coeffs_reduced', 'reduce_device_seconds', 'max_log2_length', 'ntt_module_peak_bytes']
    with (a.output / 'cells.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns);writer.writeheader()
        writer.writerows({k: r[k] for k in columns} for r in rows)
    plots(rows, fixed, a.figures)
    if a.public_figures:
        a.public_figures.mkdir(parents=True, exist_ok=True)
        for name in ('times', 'phases', 'work'):
            shutil.copy2(a.figures / ('stage2-bit-complexity-' + name + '.png'), a.public_figures / ('stage2-bit-complexity-' + name + '.png'))
    if a.html:webpage(summary, rows, a.figures, a.html)
    shutil.copy2(__file__, a.output / 'analyzer_snapshot.py')
    print(json.dumps(dict(complete=True, primary_calls=len(rows), fixed_calls=len(fixed),
                          output=str(a.output), figures=str(a.figures), highest=summary['analysis']['highest']), indent=2))


if __name__ == '__main__':
    main()
