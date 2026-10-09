"""Plot offline Stage2 carrier geometry; all displayed alternatives are estimates."""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, BoundaryNorm


def write_canvas(d, path):
    fields = ('D', 'P', 'G', 'bits', 'fold_log2', 'fold_big_mib', 'owner_mib',
              'legacy_owner_mib', 'raw_g_mib', 'coord_mib', 'concurrent_lower_mib')
    embedded = {k: d[k] for k in ('target_bits', 'carrier_bits', 'b2', 'free_mib',
                                  'reserve_mib', 'fold_boundaries', 'carrier_cases')}
    embedded['rows'] = [{k: r[k] for k in fields} for r in d['rows']]
    source = '''import { Stack, Row, Grid, H1, H2, Text, Pill, Stat, Table, BarChart, Link, useCanvasState, useHostTheme } from "cursor/canvas";
const data = __DATA__;
export default function Stage2CarrierPlan() {
  const theme = useHostTheme();
  const [choice, setChoice] = useCanvasState<number>("stage2-carrier-bits", data.carrier_bits);
  const bits = choice === data.target_bits ? data.target_bits : data.carrier_bits;
  const rows = data.rows.filter(r => r.bits === bits);
  const cases: Array<{exponent:number,bits:number,B2:number,ntt_length_ratio:number}> = data.carrier_cases;
  const boundaries: Record<string, Record<string, number>> = data.fold_boundaries;
  const exps = [...new Set(cases.map(r => r.exponent))].sort((a,b) => a-b);
  const bounds = [...new Set(cases.map(r => r.B2))].sort((a,b) => a-b);
  const f = (n:number) => n.toLocaleString("en-US", {maximumFractionDigits:2});
  const cap = data.free_mib-data.reserve_mib;
  return <Stack gap={20} style={{padding:24, background:theme.bg.editor, color:theme.text.primary}}>
    <H1>GPU Stage2：梅森承载与显存规划</H1>
    <Text weight="semibold">首选 M8011/80111：保持目标 N，算术使用 M；求逆、非单位判定和 GCD 使用 N。</Text>
    <Text tone="secondary">2026-10-09 · 当前源码整数公式分析，不是新增性能实测。较大 P 的 NTT 台阶要求降低工作集，单纯提高配额无法解决。</Text>
    <Grid columns={cases.length ? 3 : 2}>
      <Stat value={`${data.target_bits} → ${data.carrier_bits}`} label="目标位宽 → 承载位宽 / bits" />
      <Stat value={f(boundaries[String(data.carrier_bits)]["27"])} label="承载 fold L≤2^27 的最大 P" />
      {cases.length>0 && <Stat value={`${cases.filter(r=>r.ntt_length_ratio>1).length}/${cases.length}`} label="已有形状固定 D 后 NTT 变大的格数" />}
    </Grid>
    <Row gap={8}>{[...new Set([data.target_bits,data.carrier_bits])].map(s=><Pill key={s} active={s===bits} onClick={()=>setChoice(s)}>{s} bits</Pill>)}</Row>
    <H2>重复 G 树/fold：同时存活显存下界</H2>
    <Text size="small" tone="secondary">横轴：候选 D/P；纵轴：MiB。B2={data.b2.toExponential(1)}。四项布局同时存活，仍遗漏表、S4 输出、seed 等。</Text>
    <BarChart categories={rows.map(r=>`D=${r.D} / P=${r.P}`)} stacked height={360} valueSuffix=" MiB"
      series={[
        {name:"NTT A/B/Q 池",data:rows.map(r=>r.fold_big_mib)},
        {name:"fold owner",data:rows.map(r=>r.owner_mib)},
        {name:"G raw A/B",data:rows.map(r=>r.raw_g_mib)},
        {name:"giant X/Z",data:rows.map(r=>r.coord_mib)}]}
      referenceLines={[{value:data.free_mib,label:`历史 free ${data.free_mib} MiB`},{value:cap,label:`free−reserve ${cap} MiB`}]}/>
    <Text size="small" tone="secondary">来源：2026-10-09 源码布局公式；free 取自先前 GPU1 日志，未查询实时显存。下界通过不能证明可分配。</Text>
    <Table headers={["D","P","G 数量","fold L","owner reuse=3 / MiB","默认 reuse=0 / MiB","并存下界 / MiB","对 free−reserve"]}
      rows={rows.map(r=>[f(r.D),f(r.P),r.G,`2^${r.fold_log2}`,f(r.owner_mib),f(r.legacy_owner_mib),f(r.concurrent_lower_mib),r.concurrent_lower_mib>cap?"排除当前布局":"仅下界通过"])} />
    {cases.length>0 && <>
      <H2>固定原 D：承载 / 目标 fold NTT 长度比</H2>
      <Text size="small" tone="secondary">行：目标 bits → 原梅森 p；列：请求 B2；单元：长度倍数，不是速度比。来源：完整2026-10-08 GPU 形状 + 当前打包公式重算。</Text>
      <Table headers={["N bits → M bits",...bounds.map(b=>`B2=${b.toExponential(1)}`)]}
        rows={exps.map(e=>[`${cases.find(r=>r.exponent===e)!.bits} → ${e}`,...bounds.map(b=>{const c=cases.find(r=>r.exponent===e&&r.B2===b);return c?`${c.ntt_length_ratio}×`:"—";})])}/>
    </>}
    <H2>实施次序</H2>
    <Text>1. target/carrier 上下文分离，固定 D 完成正确性与归因。2. 共用 MemoryPlan，统一 owner reuse 与真实调用形状。3. tune 后联合选择 backend/D/chunk。4. 研究 short/middle product 和 scratch 复用后再跨到更大 P。</Text>
    <Link href="D:/code/MPA-OpenCl/docs/STAGE2_MERSENNE_CARRIER_MEMORY_PLAN_20261009.md">完整报告：证明、公式、源码行号与原论文</Link>
  </Stack>;
}
'''
    # Inline all data. The managed canvas directory is supplied by the caller.
    path.write_text(source.replace('__DATA__', json.dumps(embedded, ensure_ascii=False)), encoding='utf-8')
    print(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, required=True)
    ap.add_argument('--output-prefix', type=Path, required=True)
    ap.add_argument('--canvas', type=Path, help='Optional standalone .canvas.tsx output')
    a = ap.parse_args()
    d = json.loads(a.input.read_text(encoding='utf-8'))
    cases = d['carrier_cases']
    rows = [r for r in d['rows'] if r['bits'] == d['carrier_bits']]
    x = np.arange(len(rows))
    fig = plt.figure(figsize=(13.8, 10.3), layout='constrained')
    grid = fig.add_gridspec(2, 2, height_ratios=[1.1, 1])
    mem = fig.add_subplot(grid[0, :])
    base = np.zeros(len(rows))
    for field, label, color in [
        ('fold_big_mib','NTT A/B/Q pool','#386a95'),
        ('owner_mib','Fold owner','#d39248'),
        ('raw_g_mib','G raw A/B','#8d9fae'),
        ('coord_mib','Giant X/Z chunk','#bac9b5'),
    ]:
        values = np.array([r[field] for r in rows])
        mem.bar(x, values, bottom=base, label=label, color=color, width=.65)
        base += values
    for j, v in enumerate(base):
        mem.text(j, v+80, f'{v:,.0f}', ha='center', fontsize=10)
    mem.axhline(d['free_mib'], color='#4e4e4e', linestyle='--', label=f"Historical free: {d['free_mib']} MiB")
    usable = d['free_mib'] - d['reserve_mib']
    mem.axhline(usable, color='#ac4b4b', linestyle=':', label=f'Free minus reserve: {usable} MiB')
    mem.set_xticks(x, [f"D={r['D']}\nP={r['P']}" for r in rows])
    mem.set_ylabel('Concurrent payload lower bound (MiB)')
    mem.set_xlabel('Candidate D and P, ascending P')
    mem.set_ylim(0, max(base)*1.18)
    mem.set_title(f"M{d['carrier_bits']} carrier: repeated G-tree/fold allocations", loc='left', fontsize=14)
    mem.legend(ncol=3, fontsize=9, loc='upper left')
    mem.spines[['top','right']].set_visible(False)
    counts = fig.add_subplot(grid[1, 0] if cases else grid[1, :])
    counts.plot([r['P']/1000 for r in rows], [r['G'] for r in rows], marker='o', color='#386a95')
    for r in rows:
        counts.annotate(str(r['G']), (r['P']/1000,r['G']), xytext=(0,7), textcoords='offset points', ha='center')
    counts.axvline(d['fold_boundaries'][str(d['carrier_bits'])]['27']/1000,
                   color='#ac4b4b', linestyle='--', label='Largest P at NTT length 2^27')
    counts.set(xlabel='Baby polynomial degree P (thousand coefficients)',
               ylabel='G polynomial batches (count)', title=f"B2={d['b2']:.1e}: geometry, not a time forecast")
    counts.legend(fontsize=9);counts.spines[['top','right']].set_visible(False)
    if cases:
        heat = fig.add_subplot(grid[1, 1])
        exps=sorted({r['exponent'] for r in cases});bounds=sorted({r['B2'] for r in cases})
        lookup={(r['exponent'],r['B2']):r['ntt_length_ratio'] for r in cases}
        matrix=np.array([[lookup.get((e,b),np.nan) for b in bounds] for e in exps])
        heat.imshow(np.ma.masked_invalid(matrix), cmap=ListedColormap(['#dae3e9','#e8c99f','#d69a89']),
                    norm=BoundaryNorm([.5,1.5,2.5,4.5],3),aspect='auto')
        for i in range(len(exps)):
            for j in range(len(bounds)):
                if np.isfinite(matrix[i,j]):
                    heat.text(j,i,f'{matrix[i,j]:g}x',ha='center',va='center',color='#232323')
        widths={r['exponent']:r['bits'] for r in cases}
        heat.set_xticks(range(len(bounds)),[f'{b:.1e}' for b in bounds])
        heat.set_yticks(range(len(exps)),[f"{widths[e]} -> {e}" for e in exps])
        heat.set(xlabel='Requested B2',ylabel='Target bits -> carrier bits',title='Fixed-D fold NTT length ratio (carrier / target)')
    fig.suptitle('Stage2 carrier and memory planning — analytical geometry, 2026-10-09',fontsize=16)
    source = 'current integer packing/layout formulas' + (' + completed 2026-10-08 GPU shapes' if cases else '')
    fig.supxlabel('Source: ' + source + '.\n'
                  'Lower bound omits tables, S4 outputs and other buffers. Passing the line does not prove feasibility.',fontsize=10)
    a.output_prefix.parent.mkdir(parents=True,exist_ok=True)
    for ext in ('png','svg'):
        path=a.output_prefix.with_suffix('.'+ext);fig.savefig(path,dpi=170);print(path)
    plt.close(fig)
    if a.canvas:
        write_canvas(d, a.canvas)


if __name__ == '__main__':
    main()
