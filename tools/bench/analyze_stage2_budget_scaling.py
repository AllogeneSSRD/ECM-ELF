"""Summarize the serial budget study; run after all timed GPU work completes."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
from statistics import mean

MIB = 1 << 20


def alpha(a, b):
    return math.log(b['seconds'] / a['seconds']) / math.log(b['B2'] / a['B2'])


def linear(rows):
    # At fixed D/P, batch count is the relevant repeated-work coordinate.
    xs = [r['batches'] for r in rows]
    ys = [r['seconds'] for r in rows]
    xm, ym = mean(xs), mean(ys)
    slope = sum((x-xm)*(y-ym) for x,y in zip(xs,ys)) / sum((x-xm)**2 for x in xs)
    intercept = ym-slope*xm
    residual = sum((y-intercept-slope*x)**2 for x,y in zip(xs,ys))
    total = sum((y-ym)**2 for y in ys)
    return dict(intercept_seconds=intercept, seconds_per_batch=slope, r_squared=1-residual/total)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--study', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    data = json.loads((a.study/'measurements.json').read_text())
    assert data['passed'] == len(data['cases']) == len(data['runs']) and data['failed'] == 0
    a.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in data['runs']:
        c = r['case']
        log=(a.study/(c['name']+'_engine.log')).read_text(encoding='utf-8',errors='replace')
        root=dict(re.findall(r'(\w+)=([^\s]+)',re.search(r'real_batched_rootfold: (.*)',log)[1]))
        baby=dict(re.findall(r'(\w+)=([^\s]+)',re.search(r'baby_device: (.*)',log)[1]))
        host=[(int(x),int(y)) for x,y in re.findall(r'host private=(\d+) MB peak=(\d+) MB',log)]
        rows.append(dict(name=c['name'], group=c['group'], bits=c['bits'], B2=c['B2'],
                         config=c['config'], rep=c['rep'], D=c['plan']['D'], P=c['plan']['P'],
                         giant_points=int(r['shape']['giant_points']), batches=int(r['shape']['num_poly_g']),
                         big_budget_mib=c['big_mb'], owner_budget_mib=c['fold_mb'], arena_budget_mib=c['arena_mb'],
                         seconds=r['wall']['total'], init=r['wall']['init'], main=r['wall']['main'],
                         process_seconds=r['process_seconds'], owner_enabled=int(r['fold']['enabled']),
                         owner_reason=r['fold']['fallback'], owner_mib=int(r['fold']['peak_bytes'])/MIB,
                         big_mib=int(r['ntt']['big_peak_bytes'])/MIB,
                         small_mib=int(r['ntt']['small_peak_bytes'])/MIB,
                         table_mib=int(r['ntt']['table_peak_bytes'])/MIB,
                         fuse_base_mib=int(r['ntt']['fuse_base_peak_bytes'])/MIB,
                         ntt_full_mib=int(r['ntt']['full_peak_bytes'])/MIB,
                         raw_a_mib=int(r['gmemory']['rawA_peak_bytes'])/MIB,
                         raw_b_mib=int(r['gmemory']['rawB_peak_bytes'])/MIB,
                         coordinates_mib=int(r['gleaf']['coord_peak_bytes'])/MIB,
                         output_mib=int(r['output']['device_peak_bytes'])/MIB,
                         baby_payload_mib=int(baby['payload_bytes'])/MIB,
                         baby_root_d2h_bytes=int(baby['root_d2h_bytes']),
                         baby_leaf_d2h_bytes=int(baby['leaf_d2h_bytes']),
                         baby_coordinate_d2h_bytes=int(baby['coordinate_d2h_bytes']),
                         host_output_pinned_mib=int(r['output']['pinned_peak_bytes'])/MIB,
                         host_commit_snapshot_max_mib=max((x for x,_ in host),default=None),
                         host_peak_commit_snapshot_max_mib=max((y for _,y in host),default=None),
                         output_d2h_bytes=int(r['output']['d2h_words'])*8,
                         fold_h2d_bytes=int(r['fold']['h2d_bytes']), fold_d2h_bytes=int(r['fold']['d2h_bytes']),
                         fold_avoided_h2d_bytes=int(r['fold']['avoided_h2d_bytes']),
                         fold_avoided_d2h_bytes=int(r['fold']['avoided_d2h_bytes']),
                         rootfold_d2d_bytes=int(root['words'])*8,
                         rootfold_avoided_h2d_bytes=int(root['avoided_h2d_bytes']),
                         rootfold_avoided_d2h_bytes=int(root['avoided_d2h_bytes']),
                         gleaf_group_d2h_bytes=int(r['gleaf']['group_d2h_bytes']),
                         gleaf_bad_segment_d2h_bytes=int(r['gleaf']['bad_segment_d2h_bytes']),
                         gleaf_bad_point_d2h_bytes=int(r['gleaf']['bad_point_d2h_bytes']),
                         gleaf_avoided_point_d2h_bytes=int(r['gleaf']['avoided_point_d2h_bytes']),
                         gleaf_avoided_leaf_h2d_bytes=int(r['gleaf']['avoided_leaf_h2d_bytes']),
                         ntt_seconds=float(r['arena']['ntt_seconds']),fold_seconds=float(r['phases']['fold']),
                         gtrees_seconds=float(r['phases']['gtrees']),giant_seconds=float(r['phases']['giant']),
                         observed_card_mib=(r['device_samples']['observed_max_used']/MIB
                                            if r['device_samples']['observed_max_used'] is not None else None),
                         baseline_card_mib=(r['device_samples']['before_used']/MIB
                                            if r['device_samples']['before_used'] is not None else None),
                         arena_overflow=int(r['arena']['arena_overflow']), leaf_hash=r['leaf']['hash'],
                         gmp_checked=int(r['s4']['gmp_checked']), full_checks=int(r['s4']['full_checks']),
                         factors=';'.join(r['result']['factors'])))
    with (a.output/'runs.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    policy=[r for r in rows if r['group']=='shape_policy']
    slopes=[]
    for bits in sorted({r['bits'] for r in policy}):
        for config in data['configs']:
            series=sorted((r for r in policy if r['bits']==bits and r['config']==config),key=lambda r:r['B2'])
            for x,y in zip(series,series[1:]):
                slopes.append(dict(bits=bits,config=config,B2_low=x['B2'],B2_high=y['B2'],alpha=alpha(x,y),
                                   D_low=x['D'],D_high=y['D'],resident_low=x['owner_enabled'],resident_high=y['owner_enabled']))
    fixed=[]
    fits={}
    for config in ('large_resident','large_owner128'):
        for b2 in sorted({r['B2'] for r in rows if r['group']=='fixed_D'}):
            rep=[r for r in rows if r['group']=='fixed_D' and r['B2']==b2 and r['config']==config]
            assert len(rep)==2
            fixed.append(dict(config=config,B2=b2,D=rep[0]['D'],P=rep[0]['P'],batches=rep[0]['batches'],
                              seconds=mean(r['seconds'] for r in rep),min_seconds=min(r['seconds'] for r in rep),
                              max_seconds=max(r['seconds'] for r in rep),owner_enabled=rep[0]['owner_enabled']))
        fits[config]=linear([r for r in fixed if r['config']==config])
    boundary=[]
    for cfg in ('large_resident','large_owner640'):
        rep=[r for r in rows if r['group']=='owner640_boundary' and r['config']==cfg]
        if rep:
            assert len(rep)==2
            boundary.append(dict(config=cfg,seconds=mean(r['seconds'] for r in rep),
                                 samples_seconds=[r['seconds'] for r in rep],owner_enabled=rep[0]['owner_enabled'],
                                 B2=rep[0]['B2'],D=rep[0]['D'],P=rep[0]['P']))
    summary=dict(exe_sha256=data['exe_sha256'],gpu_uuid=data['gpu_uuid'],
                 runs=len(rows),warmup=sum(r['group']=='warmup' for r in rows),
                 policy_runs=len(policy),fixed_runs=sum(r['group']=='fixed_D' for r in rows),
                 local_exponents=slopes,fixed_means=fixed,fixed_batch_linear_fit=fits,owner640_boundary=boundary,
                 maximum_observed_card_mib=max(r['observed_card_mib'] for r in rows if r['observed_card_mib'] is not None),
                 maximum_logged_host_peak_commit_mib=max((r['host_peak_commit_snapshot_max_mib'] for r in rows
                                                          if r['host_peak_commit_snapshot_max_mib'] is not None),default=None),
                 all_arena_overflows=sum(r['arena_overflow'] for r in rows),
                 scope='Unreplicated policy grid; fixed-D ABBA two samples per mode per B2; no CI. Card maxima are 200ms observations, not process peaks.')
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2))
    text=['# Generated measurements (not a standalone interpretation)', '',
          '## Shape-policy grid', '',
          '|bits|B2|config|D|P|G|resident|T full (s)|NTT big MiB|owner MiB|card max observed MiB|',
          '|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in sorted(policy,key=lambda r:(r['bits'],r['B2'],r['config'])):
        text.append(f"|{r['bits']}|{r['B2']:.2e}|{r['config']}|{r['D']}|{r['P']}|{r['batches']}|{r['owner_enabled']}|{r['seconds']:.6f}|{r['big_mib']:.0f}|{r['owner_mib']:.2f}|{r['observed_card_mib']:.1f}|")
    text += ['', '## Fixed D means (two samples)', '',
             '|B2|config|G|mean s|min s|max s|', '|---:|---|---:|---:|---:|---:|']
    for r in sorted(fixed,key=lambda r:(r['B2'],r['config'])):
        text.append(f"|{r['B2']:.2e}|{r['config']}|{r['batches']}|{r['seconds']:.6f}|{r['min_seconds']:.6f}|{r['max_seconds']:.6f}|")
    text += ['', '## Local exponent', '', '|bits|config|B2 low|B2 high|alpha|D low|D high|', '|---:|---|---:|---:|---:|---:|---:|']
    for r in slopes:
        text.append(f"|{r['bits']}|{r['config']}|{r['B2_low']:.2e}|{r['B2_high']:.2e}|{r['alpha']:.4f}|{r['D_low']}|{r['D_high']}|")
    if boundary:
        text+=['','## Owner640 boundary', '', '|config|owner enabled|mean s|samples s|','|---|---:|---:|---|']
        for r in boundary:text.append(f"|{r['config']}|{r['owner_enabled']}|{r['seconds']:.6f}|{', '.join(f'{s:.6f}' for s in r['samples_seconds'])}|")
    (a.output/'tables.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    # Static scientific figures; intentionally after timed work, never alongside it.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter
    colors={'large_resident':'#2364aa','large_owner128':'#d55e00','small_big':'#009e73'}
    def capacity(n):return f'{n//1024} GiB' if n>=1024 and n%1024==0 else f'{n} MiB'
    labels={name:f"big {capacity(cfg['big_mb'])} / owner {capacity(cfg['fold_mb'])}" for name,cfg in data['configs'].items()}
    fig,axes=plt.subplots(1,3,figsize=(14,4.8))
    for ax,bits in zip(axes,(2203,4423,8191)):
        for cfg in data['configs']:
            s=sorted((r for r in policy if r['bits']==bits and r['config']==cfg),key=lambda r:r['B2'])
            ax.loglog([r['B2'] for r in s],[r['seconds'] for r in s],label=labels[cfg],color=colors[cfg],marker='o')
            for r in s:
                if not r['owner_enabled']:ax.scatter(r['B2'],r['seconds'],marker='x',s=70,color='black',zorder=4)
        reference=sorted((r for r in policy if r['bits']==bits and r['config']=='large_resident'),key=lambda r:r['B2'])
        x0,y0=reference[0]['B2'],reference[0]['seconds']
        for exponent,style in ((.5,'--'),(1,':')):
            ax.loglog([r['B2'] for r in reference],[y0*(r['B2']/x0)**exponent for r in reference],
                      linestyle=style,color='#888888',alpha=.6,label=f'reference slope {exponent:g}')
        ax.set_xticks([r['B2'] for r in reference],[f"{r['B2']:.1e}".replace('.0e','e').replace('e+0','e').replace('e+','e') for r in reference])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_title(f'{bits} bits');ax.set_xlabel('B2');ax.set_ylabel('Stage2 full wall (s)');ax.grid(True,which='both',alpha=.2)
    axes[0].legend(fontsize=7);fig.suptitle('Budget-filtered D policy; x = owner budget fallback; grid: one curve per point')
    fig.text(.5,.015,'2026-10-05 | production 893F6E90 | RTX 4060 Laptop GPU1 | B1=1000, sigma=26, lcm, default checks | big: external shape filter | guides aligned at first B2',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.055,1,.95));fig.savefig(a.output/'b2_runtime.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.8))
    for cfg in ('large_resident','large_owner128'):
        s=[r for r in fixed if r['config']==cfg]
        axes[0].plot([r['batches'] for r in s],[r['seconds'] for r in s],marker='o',label=cfg,color=colors[cfg])
        axes[0].fill_between([r['batches'] for r in s],[r['min_seconds'] for r in s],[r['max_seconds'] for r in s],alpha=.2,color=colors[cfg])
    axes[0].set_xlabel('G = ceil((floor(B2/D)+2)/P)');axes[0].set_ylabel('Stage2 full wall (s)')
    axes[0].set_title(f"4423 bits, fixed D={fixed[0]['D']}; ABBA (2/mode/B2)");axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    s=sorted((r for r in policy if r['bits']==4423 and r['config']=='large_resident'),key=lambda r:r['B2'])
    for field,label in [('big_mib','NTT big payload'),('ntt_full_mib','NTT full_peak payload'),('owner_mib','Owner payload'),('observed_card_mib','Whole-card observed maximum')]:
        axes[1].semilogx([r['B2'] for r in s],[r[field] for r in s],marker='o',label=label)
    axes[1].set_xlabel('B2');axes[1].set_ylabel('MiB');axes[1].set_title('Different memory scopes; curves cannot be summed')
    axes[1].xaxis.set_minor_formatter(NullFormatter())
    axes[1].legend(fontsize=8);axes[1].grid(alpha=.2)
    fig.text(.5,.015,'2026-10-05 | production 893F6E90 | RTX 4060 Laptop GPU1 | default checks | band: observed min/max, not CI | whole-card: 200ms samples, not process peak',ha='center',fontsize=7)
    fig.tight_layout(rect=(0,.055,1,1));fig.savefig(a.output/'fixed_d_memory.png',dpi=170);plt.close(fig)
    manifest={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in a.output.iterdir() if p.is_file() and p.name!='analysis_manifest.json'}
    manifest['analysis_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    manifest['measurements_sha256']=hashlib.sha256((a.study/'measurements.json').read_bytes()).hexdigest()
    (a.output/'analysis_manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':
    main()
