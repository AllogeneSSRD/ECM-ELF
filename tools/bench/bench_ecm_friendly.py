"""Reproducible GKL(iv)/Atkin-Morain group and point-order experiment.

Finite rosters and uniformly sampled primes are explicit. No production files
or curve defaults are changed; all raw data live below the requested output.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/ecm_prob"))
from ecmath import primes_upto
from friendly_curves import EXPERIMENT_VERSION, from_t, is_square

FAMILIES = {1: "gkl_iv", 2: "atkin_morain", 3: "gkl_generic", 4: "paper_c6", 5: "legacy_fixed_z2x8"}


def prime32(n):
    if n < 2:
        return False
    for q in (2, 3, 5, 7, 11):
        if n % q == 0:
            return n == q
    d, s = n-1, 0
    while not d & 1:
        d >>= 1
        s += 1
    for a in (2, 3, 5, 7, 11):
        x = pow(a, d, n)
        if x in (1, n-1):
            continue
        for _ in range(s-1):
            x = x*x % n
            if x == n-1:
                break
        else:
            return False
    return True


def sample_primes(bits, count, seed):
    if bits < 4 or bits > 31:
        raise ValueError("This experiment supports proven/rechecked primes of 4..31 bits")
    lo, hi = 1 << (bits-1), 1 << bits
    if bits <= 16:
        population = [p for p in primes_upto(hi-1) if p >= lo]
        return sorted(random.Random(seed+bits).sample(population, min(count, len(population))))
    rng = random.Random(seed+bits)
    chosen = set()
    while len(chosen) < count:
        n = rng.randrange(lo//2, hi//2)*2+1
        if prime32(n):
            chosen.add(n)
    return sorted(chosen)


def generic_seeds(count):
    result, t = [], 3
    while len(result) < count:
        c = from_t(t)
        if not any(is_square(x) for x in (c.e, 2*c.e, c.e*c.e-1, c.e*c.e+1)):
            result.append(t)
        t += 1
    return result


def measure(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    roster = {1: list(range(1, args.curves+1)), 2: list(range(1, args.curves+1)),
              3: generic_seeds(args.curves), 4: [1], 5: [1]}
    gp_raw = (ROOT / "tools/ecm_prob/friendly_curves.gp").read_bytes()
    (out / "model.snapshot.gp").write_bytes(gp_raw)
    report = dict(experiment_version=EXPERIMENT_VERSION, program_version=re.search(r'version\[\] = "([^"]+)"', (ROOT/"include/ecm/platform/program_version.h").read_text()).group(1),
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  gp=str(args.gp.resolve()), seed=args.seed, curves=args.curves,
                  families=FAMILIES, roster=roster, conditions=[], complete=False)
    def persist():
        (out / "measurement.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    persist()
    for bits in args.bits:
        primes = sample_primes(bits, args.primes, args.seed)
        folder = out / f"bits{bits}"
        folder.mkdir()
        (folder / "primes.json").write_text(json.dumps(primes)+"\n", encoding="utf-8")
        body = gp_raw.decode("utf-8") + f'\nprint("VERSION|",version());\nps={primes};\n'
        body += f'print("POPULATION|",primepi(2^{bits}-1)-primepi(2^{bits-1}-1));\n'
        for family, seeds in roster.items():
            body += f'seeds{family}={seeds};\n'
        body += 'for(i=1,#ps,if(!isprime(ps[i]),error("input is not proven prime"));'
        for family in roster:
            body += f'for(j=1,#seeds{family},ecmf_order({family},seeds{family}[j],ps[i]));'
        body += ');\nprint("COMPLETE");\nquit();\n'
        (folder / "input.gp").write_text(body, encoding="utf-8")
        print(f"Measuring bits={bits}, primes={len(primes)}, curves={sum(map(len,roster.values()))}", flush=True)
        start = time.perf_counter()
        with (folder / "orders.log").open("wb") as stdout, (folder / "errors.log").open("wb") as stderr:
            p = subprocess.run([str(args.gp), "-q", "-f", "-s", "256M"], input=body.encode(),
                               stdout=stdout, stderr=stderr, timeout=args.timeout)
        seconds = time.perf_counter()-start
        errors = (folder / "errors.log").read_text(encoding="utf-8", errors="replace")
        with (folder / "orders.log").open("rb") as log:
            log.seek(max(0, log.seek(0, 2)-100))
            finished = b"COMPLETE" in log.read()
        if p.returncode or "***" in errors or not finished:
            raise RuntimeError(f"PARI failed for bits={bits}: {errors[-1500:]}")
        report["conditions"].append(dict(bits=bits, primes=len(primes), seconds=seconds,
                                         raw_orders=str(folder / "orders.log")))
        persist()
        print(f"Completed bits={bits} in {seconds:.3f}s", flush=True)
    report["complete"] = True
    persist()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gp", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--curves", type=int, default=100)
    p.add_argument("--primes", type=int, default=4096)
    p.add_argument("--bits", type=int, nargs="+", default=[15, 20, 25, 30])
    p.add_argument("--seed", type=int, default=20261024)
    p.add_argument("--timeout", type=int, default=1800)
    a = p.parse_args()
    if a.curves < 1 or a.primes < 1:
        p.error("curve and prime counts must be positive")
    measure(a)


if __name__ == "__main__":
    main()
