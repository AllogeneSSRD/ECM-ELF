"""Independent route-work reference and performance fit (NumPy least squares)."""
import itertools
import math
import numpy as np

MODEL = 'giant_route_cost_v2'


def route_work(points, d, chunk, minimum, force=False):
    full, tail = divmod(points, chunk)
    if force or chunk < minimum:
        ladder_points, ladder_chunks, first = points, full+bool(tail), 1
        chain_points, chain_chunks = 0, 0
    elif tail and tail < minimum:
        ladder_points, ladder_chunks, first = tail, 1, points-tail+1
        chain_points, chain_chunks = points-tail, full
    else:
        ladder_points, ladder_chunks, first = 0, 0, points+1
        chain_points, chain_chunks = points, full+bool(tail)
    steps = sum(max(0, points-max(first, ((1 << bit)+d-1)//d)+1) for bit in range(1, 64))
    return dict(giant_chain_points=chain_points, giant_ladder_points=ladder_points,
                giant_chain_chunks=chain_chunks, giant_ladder_chunks=int(ladder_chunks), giant_ladder_steps=steps)


def annotate(sample, chunk, minimum=32768, force=False):
    return dict(sample, giant_work_model='chunk_routes_v1', giant_chunk_points=chunk,
                giant_chain_min=minimum, giant_force_ladder=int(force),
                **route_work(sample['giant_points'], sample['d'], chunk, minimum, force))


def predict_route(samples, b2, model):
    ordered = sorted(samples, key=lambda s: s['b2'])
    if model != MODEL or not 3 <= len(ordered) <= 128 or not ordered[0]['b2'] < b2 < ordered[-1]['b2']:
        return None
    scope = ('target_bits', 'arithmetic_bits', 'carrier_exponent', 'modulus_kind', 'b1', 'd',
             'giant_chunk_points', 'giant_chain_min', 'giant_force_ladder', 'giant_work_model')
    if any(any(k not in s or s[k] != ordered[0].get(k) for k in scope) or
           not s['fold_resident'] or not s['frontier_resident'] for s in ordered):
        return None
    points = [s['giant_points'] for s in ordered]
    if any(a >= b for a, b in zip(points, points[1:])) or points[-1] < 2*points[0] or points[-1] > 2**53:
        return None
    ladder = [s['giant_ladder_steps'] for s in ordered]
    chain_count = ladder.count(0)
    if not chain_count or any(v > 2**53 for v in ladder):
        return None
    dimensions = 4 if any(ladder) else 2
    if dimensions == 4 and (chain_count < 3 or len(ladder)-chain_count < 3 or len(ladder) < 7):
        return None
    x = np.array([[1., s['giant_points'], s['giant_ladder_steps'], s['giant_ladder_chunks']]
                  for s in ordered])[:, :dimensions]
    y = np.array([s['median_seconds'] for s in ordered])

    def fit(rows, targets):
        scale = np.max(x, axis=0)
        best, best_error = None, math.inf
        subsets = [(0, 1)] if dimensions == 2 else (subset for n in range(1, dimensions+1)
                    for subset in itertools.combinations(range(dimensions), n))
        for subset in subsets:
            if any(scale[j] <= 0 for j in subset):
                continue
            design = rows[:, subset]/scale[list(subset)]
            values, _, rank, singular = np.linalg.lstsq(design, targets, rcond=1e-10)
            if rank != len(subset) or np.any(values < -1e-10):
                continue
            coeff = np.zeros(dimensions)
            coeff[list(subset)] = np.maximum(values, 0)/scale[list(subset)]
            error = float(np.sum((rows@coeff-targets)**2))
            if error < best_error:
                best, best_error = coeff, error
        return best

    errors = []
    for i in range(len(x)):
        coefficients = fit(np.delete(x, i, axis=0), np.delete(y, i))
        if coefficients is None:
            return None
        errors.append(abs(float(x[i]@coefficients)-y[i]))
    relative = max(e/t for e, t in zip(errors, y))
    if relative > .08:
        return None
    first = ordered[0]
    work = route_work(b2//first['d']+2, first['d'], first['giant_chunk_points'],
                      first['giant_chain_min'], bool(first['giant_force_ladder']))
    positive = [v for v in ladder if v]
    steps = work['giant_ladder_steps']
    if steps and (not positive or not min(positive) <= steps <= max(positive)):
        return None
    coefficients = fit(x, y)
    if coefficients is None:
        return None
    features = np.array([1., b2//first['d']+2, steps, work['giant_ladder_chunks']])[:dimensions]
    seconds = float(features@coefficients)
    if not math.isfinite(seconds) or seconds <= 0:
        return None
    noise = max(s['mad_seconds'] for s in ordered)
    return dict(model=MODEL, seconds=seconds, rank=seconds+max(errors)+2*noise,
                fit_relative_error=relative, coefficients=coefficients.tolist(), **work)
