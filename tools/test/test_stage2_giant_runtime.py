"""Check giant retained capacities against completed curve allocation ledgers.

Reads existing evidence and plan-only output. Does not run CUDA or alter logs.
The ledger's full-process peak is deliberately not compared to a component peak.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/bench'))
from bench_stage2_production import sha
from stage2_memory_ledger import parse
from test_stage2_giant_memory import workspace_sites


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--curves', type=Path, required=True)
    parser.add_argument('--plans', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--joint-peak', action='store_true',
                        help='Also require complete normal no-hit curve peak equality (plan version >=2)')
    args = parser.parse_args()
    curves = json.loads(args.curves.read_text(encoding='utf-8'))
    plans = json.loads(args.plans.read_text(encoding='utf-8'))
    if not curves['complete'] or not plans['complete']:
        raise ValueError('completed evidence required')
    by_name = {row['name']: row['plan'] for row in plans['cases']}
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    rows = []
    for curve in curves['cases']:
        command = curve['command']; exe = Path(command[0])
        if sha(exe) != curves['identity']['binary_sha256']:
            raise ValueError('curve binary identity changed')
        source = exe.parent / 'sources/src/cuda/ecm_cuda_stage2.cu'
        if sha(source) != curves['identity']['sources']['src/cuda/ecm_cuda_stage2.cu']:
            raise ValueError('compiled source identity changed')
        debug = Path(command[command.index('--debug-log-file') + 1])
        ledger = parse(debug.read_text(encoding='utf-8-sig'))
        sites = workspace_sites(source.read_text(encoding='utf-8'))
        plan = by_name[curve['name']]
        gm = plan['giant_memory']
        predicted = {'before_inverse': gm['initial_bytes'],
                     'after_giant_loop': gm['after_giant_bytes'],
                     'after_frontier_admission': gm['after_giant_bytes'],
                     'after_descent': gm['after_giant_bytes'],
                     'after_block_products': gm['accumulation_bytes']}
        if 'enabled=0 ' in curve['frontier']:
            # Native nonresident descent calls need_vals(P) before descending;
            # compact products still wait for their consumer. This verifies
            # S3 capacity only, not the refused joint model's fallback timeline.
            predicted['after_descent'] += gm['value_bytes']
            if not gm['policy']['compact_products']:
                predicted['after_descent'] += gm['product_bytes']
        checked = {}
        for checkpoint in ledger['checkpoints']:
            name = checkpoint['snapshot']
            if name not in predicted:
                continue
            actual = sum(int(row['bytes']) for row in ledger['sites']
                         if row['scope'] == 'live' and row['snapshot'] == name and row['site'] in sites)
            if actual != predicted[name]:
                raise ValueError(f"{curve['name']} {name}: {actual} != {predicted[name]}")
            checked[name] = actual
        mandatory = {'before_inverse', 'after_giant_loop', 'after_descent', 'after_block_products'}
        if not mandatory <= checked.keys():
            raise ValueError('missing required curve checkpoints')
        row=dict(name=curve['name'],debug_sha256=sha(debug),checked_live_bytes=checked,
                 owned_process_peak_bytes=int(ledger['final']['peak_bytes']))
        if args.joint_peak:
            model=plan['curve_workspace_memory']
            if 'enabled=1 ' in curve['fold'] and 'enabled=1 ' in curve['frontier']:
                if model['version']<2 or not model['valid'] or not model['finished']:
                    raise ValueError('joint peak comparison requires covered initial and resident timeline')
                if curve['record']['hits'] or curve['record']['bad_factors']:
                    raise ValueError('joint peak check currently requires the normal no-hit topology')
                if row['owned_process_peak_bytes']!=model['peak_bytes']:
                    raise ValueError(f"{curve['name']}: predicted joint peak differs from runtime owned peak")
                row['verified_joint_peak_bytes']=model['peak_bytes']
            elif model['finished']:
                raise ValueError('refused owner path unexpectedly claims a finished joint model')
        rows.append(row)
    result = dict(complete=True,curves_sha256=sha(args.curves),plans_sha256=sha(args.plans),
                  cases=rows,gpu_calls=0)
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(f'PASS: {len(rows)} completed curves, giant retained capacities, closed device ledgers, '
          f"joint peak checks={sum('verified_joint_peak_bytes' in row for row in rows)}")


if __name__ == '__main__':
    main()
