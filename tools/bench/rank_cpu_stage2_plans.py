"""Shortlist fixed-D CPU experiments from an existing GWpoly run at the same N."""
import argparse
import re
from pathlib import Path


def phi(d):
    n, value, prime = d, d, 2
    while prime * prime <= n:
        if n % prime == 0:
            value -= value // prime
            while n % prime == 0:
                n //= prime
        prime = 3 if prime == 2 else prime + 2
    return value - value // n if n > 1 else value


def units(p):
    return p * ((p - 1).bit_length() + 18) + 256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("worker_log", type=Path)
    parser.add_argument("--b2", type=int, required=True)
    parser.add_argument("--memory-mib", type=int, required=True)
    parser.add_argument("--limit", type=int, default=8)
    args = parser.parse_args()
    if min(args.b2, args.memory_mib, args.limit) <= 0:
        parser.error("B2, memory budget and limit must be positive")
    log = args.worker_log.read_text()
    lines = [line for line in log.splitlines() if line.startswith("gwpoly: ")]
    if len(lines) != 1:
        parser.error("expected a single completed GWpoly curve")
    fields = dict(re.findall(r"(\w+)=([^ ]+)", lines[0]))
    selected = int(fields["D"])
    actual = re.search(r"^gwpoly-plan: ftree=\w+ bytes_per_value=(\d+)$", log, re.M)
    mib_per_value = (int(actual[1]) / 1048576 if actual else
                     float(fields["planned_mib"]) / units(int(fields["P"])))
    catalog = Path(__file__).resolve().parents[2] / "src/ecm/stage2/planning/d_catalog.h"
    candidates = {int(d) for d in re.findall(r"\{(\d+), (?:true|false)\}", catalog.read_text())}
    candidates.update(w * k for w in (30, 210, 2310, 30030, 510510)
                      for k in (1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 13, 15, 21, 27, 35, 45, 63, 81))
    candidates.add(selected)
    plans = []
    for d in candidates:
        if not 6 <= d <= 20000000 or d % 2:
            continue
        p = phi(d) // 2
        levels = (p - 1).bit_length()
        mib = mib_per_value * units(p)
        rebuild = mib > args.memory_mib and levels > 3
        if rebuild:
            segment = (p + 7) // 8
            scratch = max(18*p, 12*p + segment*((segment-1).bit_length()+4))
            mib = mib_per_value * (5*p + 16 + scratch + 256)
        cost = (8 if rebuild else 6)*p*levels + (args.b2/d + p)*(levels+8)
        plans.append((cost, d, p, mib, "segmented" if rebuild else "full"))
    ranked = sorted(plan for plan in plans if plan[3] <= args.memory_mib)
    shown = ranked[:args.limit]
    original = next(plan for plan in plans if plan[1] == selected)
    if original not in shown:
        shown.append(original)
    print("D\tP\testimated MiB at input FFT\tF tree\tmodel score\tinput D")
    for cost, d, p, mib, tree in shown:
        print(f"{d}\t{p}\t{mib:.1f}\t{tree}\t{cost:.0f}\t{'*' if d == selected else ''}")


if __name__ == "__main__":
    main()
