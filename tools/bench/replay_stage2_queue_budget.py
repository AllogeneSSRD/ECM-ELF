"""Counterfactual finite-queue costs from checked, ordered D observations.

One qualified baseline is supplied to both policies. Replay preserves the
observed order within an arithmetic/layout family. Unexecuted curves use the
recorded mean parent wall for their D; the output is an estimate.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean


def replay(routes, horizon, repeats, allowance, adaptive):
    incumbent = routes[0]
    remaining = horizon
    seconds = 0.0
    tested = []
    for candidate in routes[1:5]:
        if remaining < 2 * repeats:
            break
        # The first alternative is always allowed. For later trials use only
        # the candidate's recorded pre-measurement prediction and past timings.
        prediction = candidate["prediction"]
        trial_cost = repeats * max(0, prediction * (1 + allowance) - incumbent["engine"])
        possible_saving = (remaining - repeats) * max(0, incumbent["engine"] - prediction * (1 - allowance))
        if adaptive and tested and trial_cost > possible_saving:
            # Later predictions were produced after this actual trial. Stop
            # here to avoid using evidence unavailable on the replayed path.
            break
        tested.append(candidate["d"])
        seconds += candidate["trial_wall"]
        remaining -= repeats
        if candidate["engine"] < incumbent["engine"]:
            incumbent = candidate
    return dict(estimated_wall=seconds + remaining * incumbent["wall"], trials=tested, winner=incumbent["d"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--allowance', type=float, default=.25)
    args = parser.parse_args()
    if args.repeats < 1 or not 0 < args.allowance < 1:
        parser.error('positive repeats and allowance between zero and one required')
    plan = json.loads(args.plan.read_text(encoding='utf-8-sig'))
    if not plan['complete'] or plan['failures']:
        raise ValueError('Incomplete source study')
    groups = defaultdict(lambda: defaultdict(list))
    for row in plan['runs']:
        obs = row.get('observation')
        if not obs or not obs['clean'] or obs['hits'] or obs['bad'] or not obs['checked'] or not obs['selftest_cases']:
            continue
        if obs.get('prediction_basis') != 'before_measurement' or obs['predicted_seconds'] <= 0:
            continue
        scope = tuple(obs.get(k,0) for k in ('groot_balanced','giant_chain_block','giant_tail_balance','baby_coop','giant_coop','seed_coop','reduce_coop','reduce_threads','reduce_kernel','fold_resident','frontier_resident'))
        groups[row['cell'],row['carrier_exponent'],scope][row['D']].append(row)
    results = []
    for (cell,carrier,scope), observations in groups.items():
        routes = []
        for d, rows in observations.items():
            if len(rows) < args.repeats:
                continue
            trial = rows[:args.repeats]
            routes.append(dict(d=d, engine=mean(r['engine_seconds'] for r in trial),
                trial_wall=sum(r['parent_seconds'] for r in trial),
                wall=mean(r['parent_seconds'] for r in rows), prediction=trial[0]['observation']['predicted_seconds']))
        if len(routes) < 3:
            continue
        for h in (2,5,20):
            results.append(dict(cell=cell,carrier=carrier,scope=scope,horizon=h,
                bounded=replay(routes,h,args.repeats,args.allowance,False),
                adaptive=replay(routes,h,args.repeats,args.allowance,True)))
    if not results:
        raise ValueError('No comparable families with three qualified Ds')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as stream:
        json.dump(dict(source=str(args.plan.resolve()),repeats=args.repeats,allowance=args.allowance,rows=results),stream,indent=2)


if __name__ == '__main__':
    main()
