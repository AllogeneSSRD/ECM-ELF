"""Compare one single-thread Edwards Stage2 run per exponent to Prime95's verbose timers."""
import argparse
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt

from plot_stage2_cpu_wide import PRIME, ROOT, SOURCE, prime_curves


PHASES = (
    ("baby", "Baby points", "#80b5c8"),
    ("ftree", "F construction", "#477d98"),
    ("reciprocal", "Reciprocal / prepare", "#334e6e"),
    ("g", "Giant points + G", "#188779"),
    ("fold", "H fold", "#df9145"),
    ("scale", "H scaling", "#c2a2ca"),
    ("tail", "Descent + other main", "#896c9e"),
    ("leaf", "Leaf product + GCD", "#a69978"),
    ("other", "Setup / cleanup", "#bac0c7"),
)


def validate_stack(name, stack, total):
    if not all(math.isfinite(v) and v >= 0 for v in stack.values()):
        raise ValueError(f"{name}: negative or invalid phase: {stack}")
    if not math.isclose(sum(stack.values()), total, abs_tol=1e-5):
        raise ValueError(f"{name}: phases do not sum to {total}")


def read_prime(path):
    rows = {}
    patterns = {
        "baby": r"nQx complete\.", "ftree": r"Round off:[^\n]*avg poly_size:",
        "reciprocal": r"PolyR built\.", "compress": r"Poly compress\.",
        "g": r"PolyG built\.", "fold": r"PolyH built\.", "scale": r"H\(X\) scaled\.",
        "up_estimate": r"PolyF up\.", "down_estimate": r"PolyF down\.",
        "leaf": r"gg = mul H\(X\)\.",
    }
    for e, requested, actual, timers, block in prime_curves(path):
        init, main, gcd = timers
        samples = {key: [float(v) for v in re.findall(pattern + r"[^\n]*?Time: ([\d.]+) sec", block)]
                   for key, pattern in patterns.items()}
        if any(len(samples[key]) != 1 for key in ("baby", "reciprocal", "compress", "scale", "leaf")):
            raise ValueError(f"M{e}: missing or duplicated Prime95 timer")
        if not all(samples[key] for key in ("ftree", "g", "up_estimate", "down_estimate")):
            raise ValueError(f"M{e}: incomplete Prime95 tree timers")
        if len(samples["fold"]) != len(samples["g"]) - 1:
            raise ValueError(f"M{e}: unexpected number of Prime95 folds")
        s = {key: sum(values) for key, values in samples.items()}
        prepare = s["reciprocal"] + s["compress"]
        # PolyF up/down extrapolate the first slice. Use the main timer's
        # remaining wall time here and retain the printed estimates separately.
        stack = dict(baby=s["baby"], ftree=s["ftree"], reciprocal=prepare, g=s["g"],
                     fold=s["fold"], scale=s["scale"], leaf=s["leaf"] + gcd,
                     tail=main-s["g"]-s["fold"]-s["scale"]-s["leaf"],
                     other=init-s["baby"]-s["ftree"]-prepare)
        total = sum(timers)
        validate_stack(f"Prime95 M{e}", stack, total)
        geometry = re.search(r"D: (\d+), degree-(\d+) polynomials", block)
        if not geometry:
            raise ValueError(f"M{e}: missing Prime95 geometry")
        rows[e] = dict(requested=requested, actual=actual, fresh="Stage 1 complete." in block,
                       d=int(geometry[1]), p=int(geometry[2]), blocks=len(samples["g"]),
                       init=init, main=main, gcd=gcd, total=total, stack=stack, samples=samples)
    return rows


def read_cpu(folder, row):
    result = json.loads((folder / "result.jsonl").read_bytes())
    engine = result["stage2-engine"]
    expected = ("NF", row["e"], 1000, row["b2"], 26, "requested-B2")
    if (result["status"], result["exponent"], result["b1"], result["b2"],
        result["Edwards"]["sigma"], engine["coverage"]) != expected:
        raise ValueError(f"Different curve or incomplete run in {folder}")
    args = json.loads((folder / "cmd.json").read_bytes())
    factors = [int(args[i+1]) for i, arg in enumerate(args) if arg == "--known-factor"]
    if (args[args.index("--threads")+1], args[args.index("--memory-mib")+1], args[args.index("--d")+1],
        factors, engine["checked-values"]) != ("1", "4096", "0", row["factors"], 0):
        raise ValueError(f"Unexpected CPU benchmark configuration in {folder}")
    log = (folder / "run.log").read_text()
    lines = re.findall(r"^gwpoly-phases: (.*)$", log, re.M)
    if len(lines) != 1:
        raise ValueError(f"Expected one CPU phase record in {folder}")
    s = {k: float(v) for k, v in (field.split("=") for field in lines[0].split())}
    init, main, total = (engine[k] for k in ("init-seconds", "main-seconds", "seconds"))
    stack = dict(baby=s["baby"], ftree=s["ftree"], reciprocal=s["reciprocal"],
                 g=s["giant"]+s["gtree"], fold=s["fold"], scale=s["scale"], leaf=s["leaf_gcd"])
    stack["tail"] = main-stack["g"]-s["fold"]-s["scale"]-s["leaf_gcd"]
    stack["other"] = total-main-s["baby"]-s["ftree"]-s["reciprocal"]
    if stack["tail"] + 1e-5 < s["descent"] or init + main > total + 1e-5:
        raise ValueError(f"CPU phase boundaries overlap in {folder}")
    validate_stack(f"ECM-ELF M{row['e']}", stack, total)
    return dict(version=result["program"]["version"], dll=engine["gw-build"], d=result["d"],
                p=result["poly-size"], blocks=int(re.search(r"\bblocks=(\d+)", log)[1]),
                fft=engine["fft-length"], init=init, main=main, total=total, stack=stack, measured=s)


def draw(rows, dest):
    fig, ax = plt.subplots(figsize=(11.6, 8.1))
    fig.subplots_adjust(left=.18, right=.945, top=.83, bottom=.085)
    ymax = len(rows)-1
    longest = max(r["cpu"]["total"] for r in rows)
    for i, row in enumerate(rows):
        for key, offset, label in (("cpu", -.19, "ELF"), ("prime", .19, "P95")):
            y = i + offset
            ax.text(-.012, y, label, transform=ax.get_yaxis_transform(), ha="right", va="center",
                    color="#475569", fontsize=8)
            if key == "prime" and not row["comparable"]:
                ax.text(.3, y, "resumed curve", va="center", fontsize=8, color="#667080")
                continue
            source = row[key]
            longest = max(longest, source["total"])
            left = 0
            for phase, title, color in PHASES:
                value = source["stack"][phase]
                ax.barh(y, value, left=left, height=.30, color=color,
                        label=title if i == 0 and key == "cpu" else None)
                left += value
            ax.text(left + .5, y, f"{left:.1f}", fontsize=8, va="center")
        if i and row["b2"] != rows[i-1]["b2"]:
            ax.axhline(i-.5, color="#87929e", ls="--", lw=.9)
    ax.set_yticks(range(len(rows)), [f"M{r['e']}" for r in rows], fontsize=10)
    ax.tick_params(axis="y", length=0, pad=31)
    ax.set(xlim=(0, longest*1.1), ylim=(ymax+.56, -.56), xlabel="Seconds per curve")
    ax.grid(axis="x", alpha=.19)
    ax.set_axisbelow(True)
    for edge in ("left", "top", "right"):
        ax.spines[edge].set_visible(False)
    fig.text(.055, .966, "Single-thread Edwards Stage 2", fontsize=18, weight="bold")
    fig.text(.055, .931, f"ECM-ELF {rows[0]['cpu']['version']} and Prime95  |  Ryzen AI 9 HX 370  |  "
             "B1 = 1,000  |  Same final B2 within each pair", fontsize=10)
    fig.legend(*ax.get_legend_handles_labels(), loc="upper left", bbox_to_anchor=(.052, .919),
               frameon=False, ncol=5, fontsize=9)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=ROOT / "data/experiments/stage2_cpu_inplace")
    parser.add_argument("--figure", type=Path, default=ROOT / "data/figures/stage2-cpu-prime-phases.png")
    parser.add_argument("--summary", type=Path, default=ROOT / "data/benchmarks/stage2-cpu-prime-phases.json")
    args = parser.parse_args()
    prime = read_prime(PRIME / "screen_ed_stage2_matrix.log")
    matrix = json.loads((SOURCE / "summary.json").read_bytes())
    if {r["e"] for r in matrix} != set(prime):
        raise ValueError("Prime95 and CPU matrix cover different exponents")
    rows = []
    for case in matrix:
        e, b2 = case["e"], case["b2"]
        ref = prime[e]
        comparable = ref["fresh"] and ref["requested"] == b2
        requested_cpu = read_cpu(args.runs / f"{e}_{b2}_0_1_inplace_1", case)
        target = ref["actual"] if comparable else b2
        cpu = requested_cpu if target == b2 else read_cpu(
            args.runs / f"{e}_{target}_0_1_inplace_1", dict(case, b2=target))
        rows.append(dict(e=e, bits=case["bits"], b2=b2, cpu_b2=target, cpu=cpu,
                         cpu_requested=requested_cpu, prime=ref, comparable=comparable))
    if len({r[key]["dll"] for r in rows for key in ("cpu", "cpu_requested")}) != 1:
        raise ValueError("Mixed CPU builds")
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    draw(rows, args.figure)


if __name__ == "__main__":
    main()
