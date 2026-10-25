"""Analyze exact finite-field orders and native GMP ECM-friendly timings.

Success is point-order divisibility, not group-order smoothness. Confidence
intervals cluster by prime; the finite seed roster is not a random family sample.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/ecm_prob"))
sys.path.insert(0, str(ROOT / "tools/bench"))
from ecmath import primes_upto
from model import fit_delta
from bench_ecm_friendly import prime32

NAMES = {1: "GKL (iv)", 2: "Atkin-Morain", 3: "Generic Z2 x Z4", 4: "Paper C6 (fixed)", 5: "Legacy Z2 x Z8 (fixed)"}
COLORS = {1: "#bd3e31", 2: "#1c64a0", 3: "#368c4d", 4: "#9d6b9e", 5: "#9d7a32"}


def estimate(numerators, denominators, population):
    """Ratio estimator; finite-population correction for sampled prime clusters."""
    h, v = np.asarray(numerators, float), np.asarray(denominators, float)
    n = len(h)
    rate = h.sum()/v.sum()
    influence = (h-rate*v)/v.mean()
    fpc = math.sqrt(max(0.0, (population-n)/(population-1))) if population > 1 else 0
    se = float(influence.std(ddof=1)/math.sqrt(n)*fpc)
    return dict(value=float(rate), se=se, ci95=[max(0.0, float(rate-1.96*se)), float(rate+1.96*se)]), influence


def read_orders(folder, roster):
    ps = json.loads((folder/"primes.json").read_text())
    pi = {p: i for i, p in enumerate(ps)}
    si = {f: {s: j for j, s in enumerate(seeds)} for f, seeds in roster.items()}
    arrays = {f: np.zeros((len(ps), len(seeds), 5), dtype=np.int64) for f, seeds in roster.items()}
    seen = {f: np.zeros(a.shape[:2], dtype=bool) for f, a in arrays.items()}
    population, version, complete = None, None, False
    for line in (folder/"orders.log").open(encoding="utf-8"):
        v = line.strip().split("|")
        if v[0] == "VERSION": version = v[1]; continue
        if v[0] == "POPULATION": population = int(v[1]); continue
        if v[0] == "COMPLETE": complete = True; continue
        f, seed, p = map(int, v[:3]); i, j = pi[p], si[f][seed]
        assert not seen[f][i, j], (f, seed, p)
        seen[f][i, j] = True
        if v[3] == "bad": continue
        n, r, v2, v3, v2p = map(int, v[3:])
        assert n % r == 0 and (n-p-1)**2 <= 4*p
        assert n % (16 if f in (1, 2, 5) else 8 if f == 3 else 6) == 0
        arrays[f][i, j] = [n, r, v2, v3, v2p]
    assert complete and population and version and all(s.all() for s in seen.values())
    return ps, arrays, population, version


def hits(orders, scalar, b1, b2):
    unique, inverse = np.unique(orders, return_inverse=True)
    remnants = [int(r)//math.gcd(int(r), scalar) if r else 0 for r in unique]
    s1 = np.array([r == 1 for r in remnants])
    s2 = np.array([r == 1 or (b1 < r <= b2 and prime32(r)) for r in remnants])
    return s1[inverse].reshape(orders.shape), s2[inverse].reshape(orders.shape)


def cpu_summary(path):
    data = json.loads(path.read_text(encoding="utf-8")); rows = data.pop("rows")
    groups, result = {}, []
    for r in rows:
        assert r["valid_endpoints"] == r["curves"]
        groups.setdefault((r["bits"], r["B1"], r["family"]), []).append(r)
    for (bits, b1, f), samples in sorted(groups.items()):
        rec = dict(bits=bits, B1=b1, family=f, samples=len(samples), curves_per_sample=samples[0]["curves"],
                   loop_multiplications_per_curve=samples[0]["loop_multiplications"]/samples[0]["curves"],
                   loop_squares_per_curve=samples[0]["loop_squares"]/samples[0]["curves"])
        for key in ("setup", "dictionary", "loop", "check", "total"):
            v = [r[key+"_seconds"]/r["curves"] for r in samples]
            rec[key+"_ms"] = dict(median=float(np.median(v)*1000), minimum=min(v)*1000, maximum=max(v)*1000)
        result.append(rec)
    pairs = []
    for bits, b1 in sorted(set((r["bits"], r["B1"]) for r in result)):
        a, b = [next(r for r in result if (r["bits"], r["B1"], r["family"]) == (bits, b1, f)) for f in (1, 2)]
        pairs.append(dict(bits=bits, B1=b1, total_time_ratio_iv_over_atkin=a["total_ms"]["median"]/b["total_ms"]["median"],
                          loop_time_ratio_iv_over_atkin=a["loop_ms"]["median"]/b["loop_ms"]["median"]))
    return dict(metadata=data, conditions=result, comparisons=pairs)


def gpu_summary(path):
    data = json.loads(path.read_text(encoding="utf-8")); rows = data.pop("rows")
    result = []
    for bits in sorted(set(r["bits"] for r in rows)):
        for family in (1, 2):
            samples = [r for r in rows if r["bits"] == bits and r["family"] == family]
            assert all(r["bad"] == 0 and r["selftest_cases"] > 0 and r["checked"] > 0 and not r["factors"] for r in samples)
            rec = dict(bits=bits, family=family, samples=len(samples), B1=samples[0]["B1"], B2=samples[0]["B2"],
                       d=samples[0]["d"], p=samples[0]["p"], giant_points=samples[0]["giant_points"])
            for key in ("engine_seconds", "wall_seconds"):
                times = [r[key] for r in samples]
                rec[key] = dict(median=float(np.median(times)), minimum=min(times), maximum=max(times))
            result.append(rec)
    return dict(metadata=data, conditions=result)


def plots(report, folder):
    folder.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    for log in (False, True):
        suffix = "log" if log else "linear"
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
        for f in NAMES:
            rows = [r for r in report["success"] if r["family"] == f and r["B1"] == 256 and r["multiplier"] == 48]
            for ax, key in zip(axes, ("stage1", "stage1_plus_single_prime_stage2")):
                x, y = [r["bits"] for r in rows], [100*r[key]["value"] for r in rows]
                err = [196*r[key]["se"] for r in rows]
                ax.errorbar(x, y, yerr=err, marker="o", capsize=3, label=NAMES[f], color=COLORS[f])
                ax.set_xticks(x); ax.set_xlabel("Prime bit length"); ax.set_ylabel("Exact point-order success (%)")
                ax.grid(alpha=.25)
                if log: ax.set_yscale("log")
        axes[0].set_title("Stage1: 48 x lcm(1..256)")
        axes[1].set_title("Stage1 + ideal single-prime Stage2, B2=25,600")
        axes[0].legend(fontsize=8)
        fig.suptitle("Finite seed rosters; 95% prime-cluster intervals; 15-bit primes fully enumerated", fontsize=10)
        fig.tight_layout(); fig.savefig(folder/f"gkl_iv_success_{suffix}.png", dpi=180); plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
        for f in NAMES:
            for ax, mult in zip(axes, (1, 48)):
                rows = [r for r in report["success"] if r["family"] == f and r["B1"] == 256 and r["multiplier"] == mult]
                ax.plot([r["bits"] for r in rows], [r["D_eff"] for r in rows], marker="o", color=COLORS[f], label=NAMES[f])
                ax.set_xticks([15, 20, 25, 30]); ax.set_xlabel("Prime bit length"); ax.set_ylabel("Fitted D_eff")
                ax.set_title(f"Stage1 scalar: {mult} x lcm(1..256)"); ax.grid(alpha=.25)
                if log: ax.set_yscale("log")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
        fig.suptitle("Point-order fit at p_ref=2^(bits-0.5); conditional model parameters, not torsion orders", fontsize=10)
        fig.tight_layout(rect=[0, .10, 1, 1]); fig.savefig(folder/f"gkl_iv_deff_{suffix}.png", dpi=180); plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
        for ax, b1 in zip(axes, (256, 1024)):
            for f in (1, 2):
                rows = [r for r in report["cpu"]["conditions"] if r["family"] == f and r["B1"] == b1]
                y = [r["total_ms"]["median"] for r in rows]
                err = [[y[i]-r["total_ms"]["minimum"] for i, r in enumerate(rows)],
                       [r["total_ms"]["maximum"]-y[i] for i, r in enumerate(rows)]]
                ax.errorbar([r["bits"] for r in rows], y, yerr=err, marker="o", capsize=3, label=NAMES[f], color=COLORS[f])
            ax.set_xticks([521, 2203, 4423]); ax.set_xlabel("Modulus bit length"); ax.set_ylabel("GMP total time per curve (ms)")
            ax.set_title(f"B1={b1}, 48LCM, NAF w6"); ax.grid(alpha=.25)
            if log: ax.set_yscale("log")
        sizes = sorted(set((r["curves_per_sample"], r["samples"]) for r in report["cpu"]["conditions"]))
        counts = "; ".join(f"{c} curves x {n} batches" for c, n in sizes)
        axes[0].legend(); fig.suptitle(f"Single pinned logical CPU; {counts}; median and observed range", fontsize=10)
        fig.tight_layout(); fig.savefig(folder/f"gkl_iv_cpu_{suffix}.png", dpi=180); plt.close(fig)
        if "gpu" in report:
            fig, ax = plt.subplots(figsize=(7.6, 4.6))
            for f in (1, 2):
                rows = [r for r in report["gpu"]["conditions"] if r["family"] == f]
                y = [r["engine_seconds"]["median"] for r in rows]
                err = [[y[i]-r["engine_seconds"]["minimum"] for i, r in enumerate(rows)],
                       [r["engine_seconds"]["maximum"]-y[i] for i, r in enumerate(rows)]]
                ax.errorbar([r["bits"] for r in rows], y, yerr=err, marker="o", capsize=3, label=NAMES[f], color=COLORS[f])
            ax.set_xticks([521, 2203, 4423]); ax.set_xlabel("Modulus bit length"); ax.set_ylabel("GPU Stage2 engine time (s)")
            first = report["gpu"]["conditions"][0]
            ax.set_title(f"RTX 4060 Laptop: identical NTT pipeline, B1={first['B1']}, B2={first['B2']:,}\n"
                         f"D={first['d']:,}, P={first['p']:,}; {first['samples']} curves per condition; observed range", fontsize=10)
            ax.grid(alpha=.25); ax.legend()
            if log: ax.set_yscale("log")
            fig.tight_layout(); fig.savefig(folder/f"gkl_iv_gpu_stage2_{suffix}.png", dpi=180); plt.close(fig)


def main(args):
    if args.summary:
        report = json.loads(args.summary.read_text(encoding="utf-8"))
        report["cpu"] = cpu_summary(args.cpu)
        if args.gpu: report["gpu"] = gpu_summary(args.gpu)
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output/"summary.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
        plots(report, args.figures)
        return
    measurement = json.loads((args.input/"measurement.json").read_text())
    assert measurement["complete"]
    roster = {int(k): v for k, v in measurement["roster"].items()}
    report = dict(experiment_version=measurement["experiment_version"], program_version=measurement["program_version"],
                  measurement=str(args.input.resolve()), statistics="fixed seed roster; ratio estimates clustered by sampled prime; finite-population correction",
                  D_eff_convention="fit_delta(Stage1 point hit fraction, B1, 2**(bits-.5)); scalar multiplier explicit",
                  groups=[], success=[], comparisons=[], trials=0)
    for condition in measurement["conditions"]:
        bits = condition["bits"]
        ps, arrays, population, gp_version = read_orders(args.input/f"bits{bits}", roster)
        report["gp_version"] = gp_version
        scratch = {}
        for f, a in arrays.items():
            valid = a[:, :, 0] > 0; v = valid.sum(axis=1)
            row = dict(bits=bits, family=f, name=NAMES[f], primes=len(ps), prime_population=population, curves=len(roster[f]),
                       trials=valid.size, valid=int(valid.sum()), bad=int((~valid).sum()))
            report["trials"] += valid.size
            for name, column in (("mean_v2_group", 2), ("mean_v3_group", 3), ("mean_v2_point", 4)):
                row[name], _ = estimate(a[:, :, column].sum(axis=1), v, population)
            row["v2_group_histogram"] = {str(k): int(((a[:, :, 2] == k)&valid).sum()) for k in range(1, int(a[:, :, 2].max())+1)}
            row["counterexamples"] = {}
            for modulus in (3, 32):
                locations = np.argwhere(valid & (a[:, :, 0] % modulus != 0))
                if len(locations):
                    i, j = locations[0]
                    row["counterexamples"][str(modulus)] = dict(seed=roster[f][j], prime=ps[i], group_order=int(a[i, j, 0]), point_order=int(a[i, j, 1]))
            report["groups"].append(row)
            for b1 in (256, 1024):
                base = math.lcm(*range(1, b1+1))
                for multiplier in (1, 48):
                    h1, h2 = hits(a[:, :, 1], base*multiplier, b1, 100*b1)
                    rec = dict(bits=bits, family=f, name=NAMES[f], B1=b1, B2=100*b1, multiplier=multiplier, valid=row["valid"], bad=row["bad"])
                    for name, h in (("stage1", h1), ("stage1_plus_single_prime_stage2", h2)):
                        rec[name], influence = estimate(h.sum(axis=1), v, population)
                        scratch[(f, b1, multiplier, name)] = (rec[name]["value"], influence)
                        rec[name]["hits"] = int(h.sum())
                    rec["D_eff"] = math.exp(fit_delta(rec["stage1"]["value"], b1, 2**(bits-.5)))
                    # Group smoothness is a sufficient, different condition, retained explicitly.
                    gh, _ = hits(a[:, :, 0], base*multiplier, b1, 100*b1)
                    rec["group_stage1_fraction"] = float(gh.sum()/valid.sum())
                    report["success"].append(rec)
        for b1 in (256, 1024):
            for mult in (1, 48):
                for name in ("stage1", "stage1_plus_single_prime_stage2"):
                    x, xi = scratch[(1, b1, mult, name)]; y, yi = scratch[(2, b1, mult, name)]
                    infl = xi/y - x*yi/(y*y)
                    fpc = math.sqrt(max(0., (population-len(ps))/(population-1)))
                    se = float(infl.std(ddof=1)/math.sqrt(len(ps))*fpc)
                    report["comparisons"].append(dict(bits=bits, B1=b1, multiplier=mult, metric=name,
                                                      ratio_iv_over_atkin=x/y, ci95=[x/y-1.96*se, x/y+1.96*se]))
        print(f"Analyzed {bits}-bit primes", flush=True)
    report["cpu"] = cpu_summary(args.cpu)
    if args.gpu: report["gpu"] = gpu_summary(args.gpu)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"summary.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    with (args.output/"success.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f); writer.writerow(["bits", "family", "B1", "B2", "multiplier", "valid", "bad", "stage1_fraction", "stage1_se", "stage2_fraction", "D_eff"])
        for r in report["success"]:
            writer.writerow([r[k] for k in ("bits", "name", "B1", "B2", "multiplier", "valid", "bad")]+[r["stage1"]["value"], r["stage1"]["se"], r["stage1_plus_single_prime_stage2"]["value"], r["D_eff"]])
    plots(report, args.figures)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path)
    p.add_argument("--summary", type=Path, help="Reuse verified order statistics; refresh timings and figures")
    p.add_argument("--cpu", type=Path, required=True)
    p.add_argument("--gpu", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--figures", type=Path, required=True)
    args = p.parse_args()
    if not args.input and not args.summary: p.error("--input or --summary required")
    main(args)
