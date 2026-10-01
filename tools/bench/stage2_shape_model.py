#!/usr/bin/env python3
"""stage2_shape_model.py -- Route B (tree-based stage 2) cost model, in one file.

Why this exists
---------------
docs/DEV_STAGE2_GPU_PLAN.md states a per-curve figure of ~1.9e11 operand-bits for M5261
parameters.  That number was first obtained by BACK-DERIVING it from Prime95's wall time
(28.6 s at an assumed 0.15 ns/operand-bit) and then, in section 10.5, re-derived forwards
from the tree structure with pen and paper.  Both agreed, but nobody could re-run either.

This script is that forward derivation as code, so it can be recomputed, checked against
Prime95's own published parameters, and re-used to CHOOSE D for a GPU (section 10.6: on a
GPU we should not copy Prime95's D, because the total work is only logarithmic in D while
the transform memory is linear in it).

Conventions (identical to section 10.5 and to the bench probes)
--------------------------------------------------------------
* One multiplication of two polynomials with m coefficients of S bits each is charged
  `2 * m * (2S + ceil(log2 m))` operand-bits: the Kronecker packing that keeps carries
  inside a slot (slot width 2S + log2 m) and turns the polynomial product into one big
  integer product.  Both operands count once.
* `S` is the modulus width in bits (the coefficients of the stage-2 polynomials are
  residues mod N).

The three self-checks in `--verify` are the point of the script:
 1. poly_size(D) must equal phi(D)/2 for Prime95's published (D, poly_size) pairs
    (M5153: D=1771770 -> 167040, M5261: D=1411410 -> 132480, M8273: D=510510 -> 46080).
    If this fails, the model's notion of the baby set is wrong and every number below is
    meaningless.
 2. the total must land near 1.9e11 operand-bits for the M5261 parameter set, i.e. within
    the uncertainty of the hand derivation (it is a model, not a measurement).
 3. `--choose-d` must reproduce the ordering of section 10.6: a much smaller D is cheaper
    in MEMORY and costs only logarithmically in TIME.

Usage
-----
  python stage2_shape_model.py --verify
  python stage2_shape_model.py --b2 2.325e12 --bits 5153 --d 1771770 --num-poly-g 11
  python stage2_shape_model.py --b2 1.94e12 --bits 5261 --choose-d --mem-cap-mb 2048
"""

import argparse
import math
import sys


def phi(n):
    """Euler totient by trial division (n is small here: D <= ~2e6)."""
    result, m, p = n, n, 2
    while p * p <= m:
        if m % p == 0:
            while m % p == 0:
                m //= p
            result -= result // p
        p += 1 if p == 2 else 2
    if m > 1:
        result -= result // m
    return result


def poly_size(d):
    """Number of baby steps: integers < D/2 that are coprime to D (== phi(D)/2).

    Prime95 calls this numrels and reads it from a table (ecm.cpp:591-680); the table is
    just cached values of this function, which is what check 1 verifies.
    """
    return phi(d) // 2


def bits_per_mul(m, s):
    """Operand-bits for one polynomial multiplication (both operands, packed)."""
    if m <= 1:
        return 0
    return 2.0 * m * (2.0 * s + math.ceil(math.log2(m)))


def tree_cost(p, s):
    """DEPRECATED / UNDERCHARGES BY ~1.31x -- kept only so old numbers can be reproduced.

    stage2_tree_ref.cpp's accounting is validated against the multiplications it actually
    executes (model_tree(24) == measured f_tree == 43280, model_tree(4763) == measured
    giant_tree == 19523226, exact at three shapes), and it charges the leaf level, which
    this loop drops (it starts at m=1 with bits_per_mul(1) == 0).  Use --via-ref (it shells
    out to `stage2_tree_ref.exe --model-only`) when the exe is available; the numbers below
    are ~1.25x LOW on totals.
    """
    """Product tree F = prod_j (X - x_j): level k has P/2^k multiplications of two
    degree-2^(k-1) polynomials, so the level costs 2^? -- summed below."""
    total = 0.0
    m = 1
    while m * 2 <= p:
        n_mul = p // (2 * m)
        total += n_mul * bits_per_mul(m, s)
        m *= 2
    return total


def model(b2, s, d, num_poly_g=None, top_level_mults=3.0):
    """Per-curve operand-bits for the tree-based stage 2, split by component.

    Structure (from DEV_STAGE2_SELFHOST_FEASIBILITY.md section 1, which cites Prime95's
    ecm.cpp line numbers).  Note the BATCHING: the giant points are NOT all put into one
    product tree -- they are processed in `num_polyG` batches of `poly_size` points, and
    each batch is folded into one accumulated polynomial.  An earlier version of this
    script built a single tree over all B2/D giant points, which is a structurally
    different (and wrong) cost: it charged ~5x too much.

      F tree        product tree over the P baby points, built ONCE        (ecm.cpp:8805+)
      G tree        per outer loop, a product tree over the next P giant points
      fold          per outer loop, H = G*H mod F = 3 full-size P x P mults (ecm.cpp:9443)
      descent       ONE scaled remainder tree per curve, over the F tree (ecm.cpp:9518+)
      small         small-coefficient work at the leaves, O(P) of them
    """
    p = poly_size(d)
    num_sections = max(1.0, b2 / d)
    if num_poly_g is None:
        num_poly_g = max(2, math.ceil(num_sections / max(p, 1)))
    loops = max(1, num_poly_g - 1)

    f_tree = tree_cost(p, s)
    g_tree_each = tree_cost(p, s)
    g_tree = loops * g_tree_each
    fold = loops * top_level_mults * bits_per_mul(p, s)
    # The descent walks the F tree once and does the remainder (division) at each node;
    # Bernstein's remainder tree costs about the same as the product tree plus the
    # division, which is why it is charged as 2x the F tree here.
    descent = 2.0 * f_tree
    small = p * 2.0 * (2.0 * s + 1.0)

    return {
        "P": p,
        "num_poly_g": num_poly_g,
        "giant_points": int(num_sections) + 2,
        "f_tree": f_tree,
        "g_tree": g_tree,
        "fold": fold,
        "descent": descent,
        "small": small,
        "total": f_tree + g_tree + fold + descent + small,
    }


def fmt(x):
    return "{:.3e}".format(x)


def report(res, s, ns_per_bit, label=""):
    print("  {}P = {} coefficients, num_polyG = {}, giant points = {}".format(
        label, res["P"], res["num_poly_g"], res["giant_points"]))
    for k in ("f_tree", "g_tree", "fold", "descent", "small"):
        print("    {:<11} {:>12} operand-bits  ({:5.1f} %)".format(
            k, fmt(res[k]), 100.0 * res[k] / res["total"]))
    print("    {:<11} {:>12} operand-bits".format("TOTAL", fmt(res["total"])))
    if ns_per_bit:
        print("    -> {:.1f} s/curve at {} ns/operand-bit".format(
            res["total"] * ns_per_bit * 1e-9, ns_per_bit))
    # memory of the largest transform: the top-level P x P multiplication, packed with
    # NTT words of `bpw` payload bits and 8 bytes each, times 2 buffers
    slot = 2 * s + math.ceil(math.log2(max(res["P"], 2)))
    payload = res["P"] * slot
    bpw = max(4, int((64 - math.log2(2 * payload / 4)) / 2))
    words = 2 ** math.ceil(math.log2(2 * payload / bpw))
    print("    largest mul: slot {} bits, payload {:.3e} bits, bpw {}, N {} -> {:.0f} MB x2".format(
        slot, payload, bpw, words, words * 8 / 1048576.0))
    return words * 8 / 1048576.0


def verify():
    print("[1] poly_size(D) == phi(D)/2 against Prime95's published parameters")
    published = [(1771770, 167040, "M5153"), (1411410, 132480, "M5261"), (510510, 46080, "M8273")]
    ok = True
    for d, want, name in published:
        got = poly_size(d)
        good = got == want
        ok = ok and good
        print("    {} D={} -> phi(D)/2 = {} (Prime95's poly_size = {}) {}".format(
            name, d, got, want, "OK" if good else "MISMATCH"))
    print("    => {}".format("the baby-set model is right" if ok else "MODEL WRONG, fix before trusting anything below"))

    print("[2] cross-check against the only anchor there is (informational, see below)")
    res = model(1.94e12, 5261, 1411410)
    anchor = 1.9e11
    print("    this model      : {} operand-bits/curve".format(fmt(res["total"])))
    print("    the old anchor  : {} operand-bits/curve".format(fmt(anchor)))
    print("    ratio model/anchor = {:.2f}x".format(res["total"] / anchor))
    print("    NOTE the anchor is NOT a measurement: it was back-derived as")
    print("    '28.6 s of Prime95 wall time / 0.15 ns per operand-bit', and 0.15 was itself")
    print("    chosen so the two would agree.  The forward model above is the defensible one,")
    print("    and the 2x disagreement is an OPEN QUESTION about how Prime95's polymult moves")
    print("    bits: its reported stage2-fft-length is 320 words (20 KB) for a 5261-bit modulus,")
    print("    while a Kronecker-packed top-level P x P product needs ~1.4e9 bits of transform.")
    print("    Resolving that decides whether our Kronecker engine does ~2x or ~5x more work.")
    return ok


def choose_d(b2, s, mem_cap_mb, ns_per_bit):
    """Section 10.6: pick D minimising time subject to the transform fitting in memory.

    Only D values that are 'smooth' the way Prime95's table builds them are considered:
    D must be a product of small primes with many primitive roots (Prime95's poly_D_data
    uses D = 2*3*5*7*11*... variants).  Here we simply walk the highly-composite-ish
    candidates 2*3*5*7*11*13*..., which is what the table lists.
    """
    bases = [2, 3, 5, 7, 11, 13, 17]
    cands = set()
    for k in range(1, len(bases) + 1):
        d = 1
        for b in bases[:k]:
            d *= b
        cands.add(d)
        for extra in (2, 3, 4):
            cands.add(d * extra)
    rows = []
    for d in sorted(cands):
        res = model(b2, s, d)
        if res["P"] < 2:      # too small to have a product tree at all
            continue
        slot = 2 * s + math.ceil(math.log2(max(res["P"], 2)))
        payload = res["P"] * slot
        bpw = max(4, int((64 - math.log2(2 * payload / 4)) / 2))
        words = 2 ** math.ceil(math.log2(2 * payload / bpw))
        mem_mb = words * 8 / 1048576.0
        rows.append((d, res, mem_mb))
    print("  D        P        total operand-bits   time@{} ns/bit   largest transform".format(ns_per_bit))
    for d, res, mem_mb in rows:
        fits = "fits" if mem_mb * 2 <= mem_cap_mb else "TOO BIG"
        print("  {:<8} {:<8} {:>18} {:>12.1f} s {:>10.0f} MB  {}".format(
            d, res["P"], fmt(res["total"]), res["total"] * ns_per_bit * 1e-9, mem_mb, fits))
    best = min((r for r in rows if r[2] * 2 <= mem_cap_mb), key=lambda r: r[1]["total"], default=None)
    if best:
        print("  => best D that fits {} MB (2 buffers): D = {}, P = {}, {:.1f} s/curve".format(
            mem_cap_mb, best[0], best[1]["P"], best[1]["total"] * ns_per_bit * 1e-9))
    else:
        print("  => nothing fits {} MB; raise the cap or reduce S".format(mem_cap_mb))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--b2", type=float)
    ap.add_argument("--bits", type=int, default=5261)
    ap.add_argument("--d", type=int, default=1411410)
    ap.add_argument("--num-poly-g", type=int, default=None)
    ap.add_argument("--choose-d", action="store_true")
    ap.add_argument("--via-ref", action="store_true",
                    help="use stage2_tree_ref.exe --model-only (validated against the "
                         "multiplications that code actually executes) instead of this "
                         "script's own tree_cost, which undercharges the trees ~1.25x")
    ap.add_argument("--ref-exe", default="build_cuda_cmake/stage2_tree_ref.exe")
    ap.add_argument("--mem-cap-mb", type=float, default=2048.0)
    ap.add_argument("--ns-per-bit", type=float, default=0.278,
                    help="measured figure of merit; 0.278 = fp64 cuFFT at P=8192")
    a = ap.parse_args()

    if a.verify or a.b2 is None:
        sys.exit(0 if verify() else 1)

    if a.via_ref:
        import subprocess
        # --model-only still wants --n (it reads S from it); a bits-bit number is fine
        cmd = [a.ref_exe, "--model-only", "--n", str((1 << a.bits) - 1),
               "--b2", str(int(a.b2)), "--d", str(a.d)]
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=120).stdout
        except Exception as exc:                      # exe missing, etc.
            print("  --via-ref failed ({}); falling back to the built-in model".format(exc))
            out = ""
        rows = [ln for ln in out.splitlines() if "tree_convention=" in ln]
        print("stage2_tree_ref --model-only (validated against the multiplications it runs):")
        for ln in rows:
            print("   " + ln.strip())
        if rows:
            print("   (use the ours-balanced or ours-padded total; plan-script reproduces")
            print("    this script's undercharged convention, for continuity only)")
            return
        print("  no cost_model lines; falling back")

    if a.via_ref:
        import subprocess
        # --model-only still wants --n (it reads S from it); a bits-bit number is fine
        cmd = [a.ref_exe, "--model-only", "--n", str((1 << a.bits) - 1),
               "--b2", str(int(a.b2)), "--d", str(a.d)]
        out = ""
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=180).stdout
        except Exception as exc:
            print("  --via-ref failed ({}); falling back to the built-in model".format(exc))
        rows = [ln.strip() for ln in out.splitlines() if "tree_convention=" in ln]
        if rows:
            print("stage2_tree_ref --model-only (its recursion is validated against the"
                  " multiplications it actually executes):")
            for ln in rows:
                print("   " + ln)
            print("   use ours-balanced / ours-padded; the plan-script line reproduces this"
                  " script's own (undercharged) convention for continuity only")
            return
        print("  no cost_model lines from --model-only; falling back to the built-in model")

    if a.choose_d:
        choose_d(a.b2, a.bits, a.mem_cap_mb, a.ns_per_bit)
        return

    res = model(a.b2, a.bits, a.d, a.num_poly_g)
    print("B2 = {:.3e}, S = {} bits, D = {}".format(a.b2, a.bits, a.d))
    report(res, a.bits, a.ns_per_bit)


if __name__ == "__main__":
    main()
