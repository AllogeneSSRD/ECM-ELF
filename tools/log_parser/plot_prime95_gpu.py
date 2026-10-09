"""Render measured CPU/GPU Stage2 comparisons (PNG/SVG and optional Canvas).

Input: comparison.json from compare_prime95_gpu.py. No benchmark jobs are run.
"""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output-prefix', type=Path, required=True)
    p.add_argument('--canvas', type=Path)
    args = p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    import numpy as np
    a = json.loads(args.input.read_text(encoding='utf-8'))
    pairs = [r for r in a['exact_pairs'] if r['gpu_variant']=='cofactor']
    bounds = sorted({r['B2'] for r in pairs})
    widths = sorted({r['bits'] for r in pairs})
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False,
                         'svg.fonttype':'none'})
    cpu_color, gpu_color = '#b65b28', '#287f8e'
    outputs = []

    def save(fig, name):
        for ext in ('png','svg'):
            target = Path(str(args.output_prefix)+'_'+name+'.'+ext)
            fig.savefig(target,dpi=165,bbox_inches='tight')
            outputs.append(str(target.resolve()))
        plt.close(fig)

    fig,axes = plt.subplots(1,len(widths),figsize=(12,4.4),squeeze=False)
    for ax,bits in zip(axes[0],widths):
        rows = sorted([r for r in pairs if r['bits']==bits],key=lambda r:r['B2'])
        x=np.arange(len(rows))
        for offset,color,label,values,errs in [
            (-.19,cpu_color,'Prime95 CPU (init + main + GCD)',[r['cpu']['mean'] for r in rows],[r['cpu']['stdev'] or 0 for r in rows]),
            (.19,gpu_color,'CUDA GPU (full Stage2, 55 W)',[r['gpu']['mean_seconds'] for r in rows],[r['gpu']['std_seconds'] or 0 for r in rows])]:
            bars=ax.bar(x+offset,values,.36,color=color,label=label,yerr=errs,capsize=3)
            label_pad=max(max(r['cpu']['max'],r['gpu']['max_seconds']) for r in rows)*.025
            for bar,val,err in zip(bars,values,errs): ax.text(bar.get_x()+bar.get_width()/2,val+err+label_pad,f'{val:.2f}',ha='center',va='bottom',fontsize=9)
        ax.set_xticks(x,[f'{r["B2"]:.1e}' for r in rows])
        ax.set_xlabel('Nominal B2 tier')
        ax.set_ylabel('Mean complete Stage2 time (s)')
        ax.set_title(f'Exact same cofactor N: {bits} bits; CPU n='+','.join(str(r['samples']) for r in rows)+'; GPU n=3')
        ax.set_ylim(0,max(max(r['cpu']['max'],r['gpu']['max_seconds']) for r in rows)*1.22)
        ax.grid(axis='y',alpha=.22);ax.set_axisbelow(True)
    axes[0,0].legend(fontsize=8,loc='upper left')
    fig.suptitle('CPU / GPU Stage2 comparison — exact N, nominal B2 matched',fontsize=13)
    fig.text(.01,-.035,'Source: Prime95 screen.log + results.json.txt (2026-10-09), GPU completed study (2026-10-08). Error bars: sample SD.\nCPU B1=100000; GPU B1=20. CPU rounds actual B2 up by 3.1–7.3%; sigma and native phase boundaries differ.',fontsize=9)
    fig.tight_layout();save(fig,'exact_times')

    fig,axes=plt.subplots(1,len(bounds),figsize=(15,4.5),sharey=True)
    for ax,b2 in zip(axes,bounds):
        cpu=[r for r in a['groups'] if r['B2']==b2]
        gpu=sorted([r for r in a['GPU_summary'] if r['B2']==b2 and r['variant']=='cofactor'],key=lambda r:r['bits'])
        ax.scatter([r['bits'] for r in cpu],[r['cpu']['mean'] for r in cpu],facecolors='none',edgecolors=cpu_color,s=40,label='CPU exact-N groups (incl. unmatched)')
        matched=[r for r in pairs if r['B2']==b2]
        ax.scatter([r['bits'] for r in matched],[r['cpu']['mean'] for r in matched],color=cpu_color,s=55,label='CPU with exact cofactor GPU match')
        ax.errorbar([r['bits'] for r in gpu],[r['mean_seconds'] for r in gpu],yerr=[r['std_seconds'] or 0 for r in gpu],color=gpu_color,marker='s',markersize=4,label='GPU all-known-factors removed (n=3)')
        ax.set_title(f'B2={b2:.1e}')
        ax.set_xlabel('Actual factoring target N (bits)')
        ax.xaxis.set_major_locator(MultipleLocator(1000));ax.set_xlim(0,8500)
        ax.set_yscale('log');ax.grid(alpha=.2)
    axes[0].set_ylabel('Mean complete Stage2 time (s, log scale)')
    axes[0].legend(loc='upper left',fontsize=7.4)
    fig.suptitle('Width scaling — hollow CPU points are context, not paired speedups',fontsize=13)
    fig.text(.01,-.035,'Source: 100 complete CPU curves, 44 exact-N groups; GPU 81 timed cofactor curves. 2026-10-08/09.\nAll x coordinates use recovered integer bit length. Equal exponent or similar bit length does not establish equal N.',fontsize=9)
    fig.tight_layout();save(fig,'N_scaling')

    fig,ax=plt.subplots(figsize=(9,4.3))
    for bits,color,marker in zip(widths,[gpu_color,cpu_color],['s','o']):
        rows=sorted([r for r in pairs if r['bits']==bits],key=lambda r:r['B2'])
        ratios=[r['gpu_speedup'] for r in rows]
        ax.plot(range(len(rows)),ratios,marker=marker,color=color,label=f'Exact cofactor N: {bits} bits')
        for i,v in enumerate(ratios): ax.annotate(f'{v:.2f}x',(i,v),xytext=(0,8),textcoords='offset points',ha='center')
    ax.axhline(1,color='#777777',ls='--',lw=1)
    ax.set_xticks(range(len(bounds)),[f'{b:.1e}' for b in bounds]);ax.set_xlabel('Nominal B2 tier')
    ax.set_ylabel('GPU speedup = mean CPU time / mean GPU time')
    ax.set_title('Above 1: GPU faster; below 1: CPU faster (55 W GPU baseline)')
    ax.legend();ax.grid(alpha=.18)
    fig.text(.01,-.025,'Source: the six exact-cofactor comparisons; 2026-10-08/09. Ratios of sample means; no B2/power correction applied.',fontsize=9)
    fig.tight_layout();save(fig,'speedup')

    ordered=sorted(pairs,key=lambda r:(r['bits'],r['B2']))
    labels=[f'{r["bits"]} bits\nB2={r["B2"]:.1e}' for r in ordered]
    fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True)
    cpu_phases=[('CPU init','#5589ab',lambda r:r['phase_seconds']['init']),
                ('CPU PolyG (includes point generation)','#d48b40',lambda r:r['detail_seconds']['PolyG built']),
                ('CPU PolyH','#47968b',lambda r:r['detail_seconds']['PolyH built']),
                ('CPU other main + GCD (residual)','#aaaaaa',lambda r:r['cpu']['mean']-r['phase_seconds']['init']-r['detail_seconds']['PolyG built']-r['detail_seconds']['PolyH built'])]
    gpu_phases=[('GPU baby + F tree/init','#5589ab',lambda r:r['gpu']['init_seconds']),
                ('GPU inverse','#a78ab0',lambda r:r['gpu']['inv_seconds']),
                ('GPU giant + leaf preparation','#8d899e',lambda r:r['gpu']['giant_seconds']+r['gpu']['gleaves_seconds']),
                ('GPU G trees','#d48b40',lambda r:r['gpu']['gtrees_seconds']),
                ('GPU fold','#47968b',lambda r:r['gpu']['fold_seconds']),
                ('GPU descent','#b6a15f',lambda r:r['gpu']['descent_seconds']),
                ('GPU other / leaf product / GCD','#aaaaaa',lambda r:r['gpu']['other_seconds']+r['gpu']['accum_seconds']+r['gpu']['name_seconds'])]
    for ax,phases,cpu_mode in zip(axes,[cpu_phases,gpu_phases],[True,False]):
        bottom=np.zeros(len(ordered))
        for name,color,get in phases:
            values=np.array([100*get(r)/(r['cpu']['mean'] if cpu_mode else r['gpu']['mean_seconds']) for r in ordered])
            if np.min(values)<-.01: raise ValueError('Nonadditive phase partition')
            ax.bar(range(len(ordered)),values,bottom=bottom,color=color,label=name,width=.7)
            for i,v in enumerate(values):
                if v>=10: ax.text(i,bottom[i]+v/2,f'{v:.0f}%',ha='center',va='center',fontsize=9)
            bottom+=values
        if np.max(np.abs(bottom-100))>.01: raise ValueError('Phase accounting mismatch')
        ax.set_ylabel('Mean Stage2 time (%)');ax.set_ylim(0,100);ax.set_yticks([0,25,50,75,100])
        ax.set_title('Prime95: native timers; PolyF slice estimates excluded' if cpu_mode else 'GPU: disjoint wall-time phases; S4/NTT nested event counters excluded',loc='left',fontsize=10)
        ax.legend(ncol=2 if cpu_mode else 3,fontsize=8,loc='upper left',bbox_to_anchor=(0,-.18 if cpu_mode else -.35))
    axes[1].set_xticks(range(len(ordered)),labels);axes[1].set_xlabel('Exact cofactor N / nominal B2 tier')
    fig.suptitle('Where time goes — native phase shares (boundaries differ between implementations)',fontsize=13)
    fig.subplots_adjust(hspace=.67,bottom=.23,top=.92)
    fig.text(.01,.01,'Source: six exact-cofactor groups, 2026-10-08/09. Each panel sums to 100%; CPU PolyG includes work GPU reports separately.\nPolyF up/down extrapolations are not stacked. GPU S4 reduction is nested in several phases, not another additive segment.',fontsize=9)
    save(fig,'phases_percent')
    if args.canvas:
        write_canvas(a,args.canvas)
        outputs.append(str(args.canvas.resolve()))
    print(json.dumps(outputs,ensure_ascii=False,indent=2))


def write_canvas(a,path):
    data=dict(counts=a['counts'],pairs=[dict(bits=r['bits'],p=r['exponent'],b2=r['B2'],
        cpu=r['cpu']['mean'],cpu_sd=r['cpu']['stdev'],cpu_n=r['samples'],gpu=r['gpu']['mean_seconds'],
        gpu_sd=r['gpu']['std_seconds'],gpu_n=r['gpu']['samples'],ratio=r['gpu_speedup'],
        overshoot=r['cpu_B2_overshoot_percent'],cpu_D=r['CPU_D'],cpu_P=r['CPU_degree'],
        gpu_D=r['gpu']['D'],gpu_P=r['gpu']['P'],gpu_G=r['gpu']['G'],cpu_fft=r['CPU_fft'],
        variant=r['gpu_variant'],cpu_init=r['phase_seconds']['init'],cpu_main=r['phase_seconds']['main'],
        gpu_init=r['gpu']['init_seconds'],gpu_main=r['gpu']['main_seconds']) for r in a['exact_pairs']],
        groups=[dict(p=r['exponent'],bits=r['bits'],b2=r['B2'],n=r['samples'],seconds=r['cpu']['mean'],matched=r['gpu_exact_match']) for r in a['groups']])
    source='''import {Stack, Row, Grid, H1, H2, Text, Stat, Callout, Select, BarChart, Table, Divider, useCanvasState} from "cursor/canvas";
const DATA = __DATA__;
const fmt = (n:number) => n.toFixed(3);
export default function ECMComparison() {
 const [tier,setTier]=useCanvasState("ecm-cpu-gpu-20261009-B2","2600000000000");
 const b2=Number(tier);
 const exact=DATA.pairs.filter(r=>r.b2===b2&&r.variant==="cofactor").sort((a,b)=>a.bits-b.bits);
 const groups=DATA.groups.filter(r=>r.b2===b2).sort((a,b)=>a.p-b.p||a.bits-b.bits);
 const labels=exact.map(r=>`${r.bits} bits (M${r.p} cofactor)`);
 return <Stack gap={20} style={{padding:24,maxWidth:1180,margin:"0 auto"}}>
  <H1>ECM Stage2：Prime95 CPU / CUDA GPU</H1>
  <Text tone="secondary">2026-10-08/09 · Ryzen AI 9 HX 370 / RTX 4060 Laptop · 原始日志、结果 JSON 与完整 GPU 测量</Text>
  <Grid columns={4} gap={20}><Stat value="100" label="完整 CPU Stage2 曲线"/><Stat value="34" label="命中因子提前结束，不计均值"/><Stat value="134/134" label="实际输入 N 已恢复"/><Stat value="6" label="相同余因子 N / B2 档位"/></Grid>
  <Callout tone="neutral" title="约 2k bits 时 GPU 更快；约 8k bits 时 Prime95 更快">
   1939 bits：GPU 1.38–2.30×；7995 bits：GPU 0.42–0.76×。速度比 = CPU 秒数 / GPU 秒数，大于 1 表示 GPU 更快。
  </Callout>
  <Row gap={12}><Text weight="semibold">B2 档位</Text><Select value={tier} onChange={setTier} options={[{value:"26000000000",label:"2.6 × 10¹⁰"},{value:"260000000000",label:"2.6 × 10¹¹"},{value:"2600000000000",label:"2.6 × 10¹²"}]}/></Row>
  <H2>相同整数 N：完整 Stage2 耗时均值</H2>
  <Text size="small" tone="secondary">横轴：实际余因子位宽（bits，分类）；纵轴：平均耗时（秒）。图例区分 CPU / GPU；源数据：CPU 2026-10-09、GPU 2026-10-08。</Text>
  <BarChart categories={labels} series={[{name:"Prime95 CPU (init + main + GCD)",data:exact.map(r=>r.cpu),tone:"warning"},{name:"CUDA GPU (full Stage2, 55 W)",data:exact.map(r=>r.gpu),tone:"info"}]} height={310} valueSuffix=" s" showValues/>
  <Table headers={["实际 N","CPU 均值 ± SD / s","CPU n","GPU 均值 ± SD / s","GPU n","GPU 速度比"]} rows={exact.map(r=>[`${r.bits} bits`,`${fmt(r.cpu)} ± ${fmt(r.cpu_sd??0)}`,r.cpu_n,`${fmt(r.gpu)} ± ${fmt(r.gpu_sd??0)}`,r.gpu_n,`${r.ratio.toFixed(2)}×`])}/>
  <Grid columns={2} gap={24}>
   <Stack gap={10}><H2>原生初始化与主阶段占比</H2><Text size="small" tone="secondary">横轴：实现与位宽；纵轴：完整 Stage2 百分比。各组均值；两实现计时边界不同，不能当作相同内核成本。</Text>
    <BarChart categories={exact.flatMap(r=>[`${r.bits} CPU`,`${r.bits} GPU`])} series={[{name:"Native init",data:exact.flatMap(r=>[r.cpu_init,r.gpu_init]),tone:"info"},{name:"Native main + GCD",data:exact.flatMap(r=>[r.cpu_main,r.gpu_main]),tone:"neutral"}]} normalized height={250}/>
   </Stack>
   <Stack gap={10}><H2>选形差异</H2><Table headers={["位宽","CPU D / P","GPU D / P","GPU G"]} rows={exact.map(r=>[r.bits,`${r.cpu_D.join(",")} / ${r.cpu_P.join(",")}`,`${r.gpu_D} / ${r.gpu_P}`,r.gpu_G])}/>
    <Text>7995 bits / 最大 B2：CPU 12 个 PolyG、11 个 PolyH；GPU 42 个 G 多项式、41 次 fold。计数概念存在实现差异，仍显示 GPU 批次压力。</Text>
   </Stack>
  </Grid>
  <Callout tone="neutral" title="归约域是重要算法差异，尚未独立量化收益">
   Prime95 参考源码对目标余因子 N 剥离已知因子，但 gwsetup 仍使用原始梅森形式。GPU 本轮余因子使用通用 division 归约。
   应研究“以完整梅森数承载算术、以真实 N 做求逆/GCD”的实现；直接改为完整梅森输入会重新引入已知因子及非单位回退。
  </Callout>
  <Divider/>
  <H2>全位宽 CPU 记录：仅作上下文</H2>
  <Text tone="secondary">B2 当前档位的全部精确 N 分组。除已配对行外，不能与相同指数的 GPU 时间计算速度比。</Text>
  <Table headers={["原指数 p","实际 N bits","CPU 完整 S2 / s","样本 n","现有 GPU 是否有同 N"]} rows={groups.map(r=>[r.p,r.bits,fmt(r.seconds),r.n,r.matched?"相同整数 N":"未匹配"])} striped/>
  <H2>比较边界</H2>
  <Text>CPU B1=100000，GPU B1=20；sigma 不同；CPU 实际 B2 比档位高约 3.1–7.3%。CPU 使用主线程及 3 个 polymult helper，并有 PRP 日志活动；GPU 数据采于 55 W，未与修复后的功耗混合。1939 bits 两个较大 B2 的 CPU CV 约 15–24%，需保留误差范围。</Text>
  <Text>功耗修复后只有 7995 bits / B2=2.6e11 有 3 次对照：GPU 34.920 s；CPU 29.620 s，GPU 速度比 0.85×。其余档位没有修复后数据，未外推。</Text>
  <Text size="small" tone="secondary">明细与行号：仓库 data/prime95_ecm_20261009/comparison/exact_pairs.csv、modulus_runs.csv；报告：docs/PRIME95_GPU_STAGE2_COMPARISON_20261009.md。最后一条曲线的停机后 Resuming 消息已排除，不影响已完成计时。</Text>
 </Stack>;
}
'''
    # A single self-contained canvas; all measurements are embedded, no network.
    path.write_text(source.replace('__DATA__',json.dumps(data,ensure_ascii=False,separators=(',',':'))),encoding='utf-8')


if __name__=='__main__':
    main()
