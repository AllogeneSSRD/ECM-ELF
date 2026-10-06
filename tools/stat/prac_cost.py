#!/usr/bin/env python3
"""Analyze legal Stage1 PRAC/Lucas plans against the current param0 ladder.

Examples:
  python tools/stat/prac_cost.py 1000 10000 100000 --cgbn
  python tools/stat/prac_cost.py 10000 --lucas-codes .refactor/ecm/Lchain_codes.dat
  python tools/stat/prac_cost.py 1000 --emit-plan docs/data/prac_1000_plan.json

This is an offline work-count study. Ratios are not GPU timing or cycle counts.
Every selected plan checks its exact differential scalar relations automatically.
"""

from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import time

from ecm_prac_plan import (load_lucas_codes, lucas_plan, prac_plan,
                           prime_multiplicity, prime_window, scalar_bits, stage1_terms)


def bound(value: str) -> int:
    try:
        number = Decimal(value)
        if not number.is_finite() or number != number.to_integral_value() or number < 2:
            raise ValueError
        return int(number)
    except (ValueError, InvalidOperation) as exc:
        raise argparse.ArgumentTypeError('Expected an integer B1 >= 2') from exc


def analyze(limit: int, args: argparse.Namespace, codes: dict[int, int]) -> tuple[dict, list]:
    started = time.perf_counter()
    if args.prime_window:
        low, high = args.prime_window
        if high > limit:
            raise ValueError('--prime-window upper bound cannot exceed B1')
        terms = [(p, prime_multiplicity(p, limit)) for p in prime_window(low, high)]
        if not terms:
            raise ValueError('Prime window contains no prime multipliers')
    else:
        terms = list(stage1_terms(limit))
    bits = scalar_bits(terms, args.torsion)
    dbl_cost, add_cost = 3 + 2 * args.sqr, 4 + 2 * args.sqr
    totals = Counter()
    hist = Counter()
    rows = []
    lucas_total = Counter()
    covered_primes = 0
    max_slots = 1
    max_lucas_slots = 1
    max_peak = 1
    plan_bytes = 4096
    for p, multiplicity in terms:
        plan = prac_plan(p, search=args.search, sqr=args.sqr)
        dbl, add = plan.counts
        totals['prac_dbl'] += multiplicity * dbl
        totals['prac_dadd'] += multiplicity * add
        lowered = plan.lower()
        max_slots = max(max_slots, lowered['point_slots'])
        max_peak = max(max_peak, lowered['peak_live_points'])
        chosen = plan
        if p < 11 or p in codes:
            try:
                lucas = lucas_plan(p, codes.get(p, 0))
            except ValueError as exc:
                raise ValueError(f'Lucas record p={p}, code={codes.get(p, 0):016x}: {exc}') from exc
            ld, la = lucas.counts
            lucas_total['dbl'] += multiplicity * ld
            lucas_total['dadd'] += multiplicity * la
            lucas_total['covered_repetitions'] += multiplicity
            covered_primes += 1
            lucas_lowered = lucas.lower()
            max_lucas_slots = max(max_lucas_slots, lucas_lowered['point_slots'])
            fits = args.point_budget == 0 or lucas_lowered['point_slots'] <= args.point_budget
            if fits and ld * dbl_cost + la * add_cost < dbl * dbl_cost + add * add_cost:
                chosen = lucas
                lowered = lucas_lowered
        cd, ca = chosen.counts
        totals['hybrid_dbl'] += multiplicity * cd
        totals['hybrid_dadd'] += multiplicity * ca
        hist[f'{chosen.source}:{lowered["point_slots"]}slots'] += multiplicity
        if args.emit_plan:
            row = dict(prime=p, repetitions=multiplicity, source=chosen.source,
                       d=chosen.initial_d, **lowered)
            plan_bytes += len(json.dumps(row, indent=6).encode('utf-8')) + 16
            if plan_bytes > args.max_plan_mib * 1024**2:
                raise ValueError('Expanded plan exceeds --max-plan-mib; use compact per-prime descriptors')
            rows.append(row)
    if args.torsion == 12:
        for key in ('prac', 'hybrid'):
            totals[f'{key}_dbl'] += 3
            totals[f'{key}_dadd'] += 1
        tplan = lucas_plan(3)
        tlower = tplan.lower()
        max_slots = max(max_slots, tlower['point_slots'])
        max_peak = max(max_peak, tlower['peak_live_points'])
        hist['torsion12:2slots'] += 1
        hist['torsion12:1slots'] += 2
        if args.emit_plan:
            rows.append(dict(prime=3, repetitions=1, source='torsion12', **tlower))
            tplan = prac_plan(2)
            rows.append(dict(prime=2, repetitions=2, source='torsion12', **tplan.lower()))
    ladder = (bits - 1) * (6 + 4 * args.sqr)
    result = dict(B1=limit, scope='prime-window-only' if args.prime_window else 'full-stage1',
                  prime_window=args.prime_window, torsion=args.torsion, scalar_bits=bits, ladder_bits=bits - 1,
                  point_budget=args.point_budget,
                  prime_count=len(terms), prime_repetitions=sum(e for _, e in terms),
                  sqr_cost=args.sqr, dbl_cost=dbl_cost, projective_dadd_cost=add_cost,
                  ladder_cost=ladder, **dict(totals),
                  prac_point_slots=max_slots, prac_peak_live_points=max_peak,
                  lucas_point_slots=max_lucas_slots, selected_slot_histogram=dict(hist),
                  covered_primes=covered_primes, covered_lucas=dict(lucas_total))
    for kind in ('prac', 'hybrid'):
        cost = totals[f'{kind}_dbl'] * dbl_cost + totals[f'{kind}_dadd'] * add_cost
        result[f'{kind}_cost'] = cost
        result[f'{kind}_ratio'] = cost / ladder
        result[f'{kind}_arithmetic_reduction_percent'] = 100 * (1 - cost / ladder)
    result['analysis_seconds'] = time.perf_counter() - started
    return result, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('B1', nargs='*', type=bound, default=[1000, 10000, 100000])
    parser.add_argument('--sqr', type=float, default=1.0, help='S/M; current CGBN default 1')
    parser.add_argument('--cgbn', action='store_true', help='Use S/M=1, param0 ladder=10')
    parser.add_argument('--check', action='store_true', help='Compatibility flag; scalar checks are always enabled')
    parser.add_argument('--search', type=int, default=7, help='Prime95 PracSearch, 1..50')
    parser.add_argument('--torsion', type=int, choices=[1, 12], default=1)
    parser.add_argument('--prime-window', nargs=2, type=bound, metavar=('LOW', 'HIGH'),
                        help='Analyze only primes in an interval; NOT a full-B1 extrapolation')
    parser.add_argument('--lucas-codes', type=Path)
    parser.add_argument('--point-budget', type=int, default=3,
                        help='Maximum point slots when selecting Lucas over PRAC (default 3; 0 unlimited)')
    parser.add_argument('--output', type=Path, help='Detailed cost analysis JSON')
    parser.add_argument('--emit-plan', type=Path, help='Single-B1 experimental slot plan JSON')
    parser.add_argument('--max-plan-mib', type=float, default=64.0)
    args = parser.parse_args()
    if args.cgbn:
        args.sqr = 1.0
    if not 0 < args.sqr < float('inf') or not 1 <= args.search <= 50:
        parser.error('Expected a finite positive --sqr and --search in 1..50')
    if args.point_budget != 0 and args.point_budget < 3:
        parser.error('--point-budget must be 0 or at least 3 for PRAC fallback')
    if args.emit_plan and len(args.B1) != 1:
        parser.error('--emit-plan requires exactly one B1')
    if args.prime_window and (len(args.B1) != 1 or args.emit_plan or args.torsion != 1):
        parser.error('--prime-window requires one B1, torsion=1 and no --emit-plan')
    if not 0 < args.max_plan_mib < float('inf'):
        parser.error('--max-plan-mib must be finite and positive')
    codes, metadata = {}, None
    if args.lucas_codes:
        codes, metadata = load_lucas_codes(args.lucas_codes)
    repo = Path(__file__).resolve().parents[2]
    sources = {}
    for name in ('tools/stat/prac_cost.py', 'tools/stat/ecm_prac_plan.py',
                 '.refactor/p95v3106b01.source/ecm.cpp', '.refactor/ecm/ecm.c'):
        path = repo / name
        if path.exists():
            sources[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = dict(schema=1, model='param0-projective-dadd-v1',
                  measurement='offline-operation-counts-not-gpu-timing',
                  sources=sources, search=args.search, point_budget=args.point_budget,
                  lucas_codes=metadata, results=[])
    print('Model: param0 6M+4S; chain DBL=3M+2S, DADD=4M+2S. GPU timing is not measured.')
    if metadata:
        print(f'Lucas input: {metadata["records"]} records, primes 11..{metadata["last_prime"]}.')
    for limit in args.B1:
        result, rows = analyze(limit, args, codes)
        report['results'].append(result)
        print(f'B1={limit} bits={result["scalar_bits"]} D={result["prac_dbl"]} '
              f'A={result["prac_dadd"]} PRAC/ladder={result["prac_ratio"]:.6f} '
              f'hybrid/ladder={result["hybrid_ratio"]:.6f} '
              f'PRAC slots={result["prac_point_slots"]} '
              f'analysis={result["analysis_seconds"]:.3f}s', flush=True)
        if args.emit_plan:
            payload = json.dumps(dict(schema=1, experimental=True, B1=limit,
                                     torsion=args.torsion, scalar_bits=result['scalar_bits'],
                                     sources=sources, lucas_codes=metadata, blocks=rows),
                                 indent=2).encode('utf-8')
            if len(payload) > args.max_plan_mib * 1024**2:
                raise ValueError('Expanded plan exceeds --max-plan-mib; use compact per-prime descriptors')
            args.emit_plan.parent.mkdir(parents=True, exist_ok=True)
            args.emit_plan.write_bytes(payload + b'\n')
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
