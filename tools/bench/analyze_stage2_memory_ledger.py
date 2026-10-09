"""Plot simultaneous owned CUDA payloads and interval peaks from a checked matrix.

Never combines independent module peaks or labels tracked payload as driver usage.
Allocation sites refer to the frozen binary's source lines, not guessed modules.
"""
import argparse
import json
from pathlib import Path
from bench_stage2_production import sha
from stage2_memory_ledger import parse


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--figure',type=Path)
    a=p.parse_args()
    data=json.loads(a.input.read_text(encoding='utf-8'))
    if not data['complete'] or not data.get('memory_ledger'):
        raise ValueError('a completed allocation-ledger matrix is required')
    result=dict(input_sha256=sha(a.input),source_sha256=sha(__file__),parser_sha256=sha(Path(__file__).with_name('stage2_memory_ledger.py')),
                identity=data['identity'],input=data['input'],payload_only=True,runs=[])
    for row in data['runs']:
        if sha(row['debug_log'])!=row['debug_sha256']:
            raise ValueError('raw debug evidence changed')
        native=parse(Path(row['debug_log']).read_text(encoding='utf-8'))
        if native!=row['memory_ledger']:
            raise ValueError('ledger parser differs from the retained native evidence')
        peak=sorted((s for s in native['sites'] if s['scope']=='global_peak'),
                    key=lambda s:int(s['bytes']),reverse=True)
        result['runs'].append(dict(name=row['name'],key=row['key'],category=row['category'],
            final=native['final'],checkpoints=native['checkpoints'],global_peak_sites=peak))
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    if a.figure:
        import matplotlib.pyplot as plt
        rows=[r for r in result['runs'] if r['category']!='warmup']
        fig,axes=plt.subplots(1,len(rows),figsize=(7*len(rows),5.8),layout='constrained',squeeze=False,sharey=True)
        labels={'before_baby':'Before baby','before_ftree':'Before F tree','before_inverse':'After F tree',
                'after_inverse':'After inverse','after_fold_admission':'Fold admitted',
                'after_giant_loop':'After giant loop','after_frontier_admission':'Frontier admitted',
                'after_descent':'After descent','final':'Final'}
        for ax,row in zip(axes[0],rows):
            records=row['checkpoints'];x=range(len(records))
            ax.plot(x,[int(r['live_bytes'])/(1<<20) for r in records],'o-',label='Live owned payload')
            ax.plot(x,[int(r['interval_peak_bytes'])/(1<<20) for r in records],'s--',label='Peak since preceding checkpoint')
            ax.axhline(int(row['final']['peak_bytes'])/(1<<20),color='#aa7044',alpha=.6,label='Complete owned payload peak')
            ax.set_xticks(list(x),[labels.get(r['snapshot'],r['snapshot']) for r in records],rotation=45,ha='right')
            ax.set_ylabel('MiB of owned CUDA allocations');ax.set_title(row['key'])
            ax.spines[['top','right']].set_visible(False)
        ymax=max(int(row['final']['peak_bytes'])/(1<<20) for row in rows)
        for ax in axes[0]:ax.set_ylim(0,max(1,ymax*1.08))
        fig.legend(*axes[0,0].get_legend_handles_labels(),loc='outside lower center',ncol=3,frameon=False)
        fig.suptitle(f'D={data["input"]["D"]}, B2={data["input"]["B2"]:.1e} · carrier M{data["input"]["carrier_exponent"]}\n'
                     'Exploratory lifecycle measurement; context/module overhead and other GPU users excluded')
        a.figure.parent.mkdir(parents=True,exist_ok=True)
        for ext in ('png','svg'):fig.savefig(a.figure.with_suffix('.'+ext),dpi=170)
        plt.close(fig)
    print(json.dumps({r['name']:{k:r['final'][k] for k in ('peak_bytes','live_bytes','persistent_bytes','unknown_frees')}
                      for r in result['runs']}))


if __name__=='__main__':main()
