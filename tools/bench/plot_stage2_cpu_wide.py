"""Plot historical or current CPU wide Stage2 timings with the Prime95 reference."""
import argparse
import hashlib
import json
import math
import re
import shutil
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

from bench_cpu_stage2_full_width import AFFINITY

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/experiments/stage2_wide_b1000"
DOC = ROOT / "docs/performance"
PRIME = ROOT / "tools/benchmarks/prime95"


def prime_curves(path):
    """Read each completed curve with its own Stage2 timers and verbose log."""
    blocks = path.read_text(encoding="utf-8-sig").split("ECM on M")[1:]
    seen = set()
    for block in blocks:
        header = re.match(r"(\d+): Edwards curve #1 with s=26, B1=1000, B2=(\d+)", block)
        if not header:
            raise ValueError("Unexpected Prime95 curve header")
        e, requested = map(int, header.groups())
        end = re.search(rf"M{e} completed 1 ECM Edwards curve, B1=1000, B2=(\d+)", block)
        if not end or e in seen:
            raise ValueError(f"Incomplete or duplicate Prime95 curve M{e}")
        seen.add(e)
        block = block[:end.end()]
        timers = []
        for phase in ("init complete", "complete", "GCD complete"):
            matches = re.findall(rf"Stage 2 {phase}\.[^\n]*?[Tt]ime: ([\d.]+) sec", block)
            if len(matches) != 1:
                raise ValueError(f"Missing or duplicate {phase} timer for M{e}")
            timers.append(float(matches[0]))
        yield e, requested, int(end[1]), timers, block


def prime_samples(path):
    return {e: (requested, actual, sum(timers), "Stage 1 complete." in block)
            for e, requested, actual, timers, block in prime_curves(path)}


def current_samples(path, rows):
    report = json.loads(path.read_bytes())
    version = re.search(r"^ECM-ELF (\d+\.\d+\.\d+) CPU Stage2\b", report["version"])
    repeats = report["repeats"]
    if (not version or not report["complete"] or report["matrix"] != "wide" or
            repeats < 1 or len(report["runs"]) != 2 * len(rows) * repeats):
        raise ValueError("Expected a complete, versioned wide CPU matrix")
    timing = {}
    for row in rows:
        save = SOURCE / row["directory"] / f"m{row['e']}.save"
        saved = hashlib.sha256(save.read_bytes()).hexdigest()
        for threads, cpus in AFFINITY.items():
            runs = [r for r in report["runs"] if (r["bits"], r["threads"]) == (row["bits"], threads)]
            if sorted(run["repeat"] for run in runs) != list(range(1, repeats + 1)):
                raise ValueError(f"Incomplete current samples for M{row['e']}, {threads}T")
            for run in runs:
                if (tuple(run["cpus"]), run["save_sha256"], run["exponent"], run["b1"], run["b2"],
                    run["actual"], run["sigma"]) != (cpus, saved, row["e"], 1000, row["b2"], row["b2"], 26):
                    raise ValueError(f"Current input mismatch for M{row['e']}, {threads}T")
                if (not run["dll"] or run["dll"] not in report["version"] or
                        not math.isfinite(run["seconds"]) or run["seconds"] <= 0):
                    raise ValueError(f"Invalid current sample for M{row['e']}, {threads}T")
            timing[row["e"], threads] = [run["seconds"] for run in runs]
    return version[1], timing


def timing_bar(ax, x, values, width, color, label, historical=False):
    center = median(values)
    appearance = (dict(facecolor=to_rgba(color, .22), edgecolor=color, linestyle="--", linewidth=1)
                  if historical else dict(color=color))
    ax.bar(x, center, width, label=label, yerr=[[center-min(values)], [max(values)-center]],
           capsize=3, **appearance)
    return center


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", type=Path, help="Complete --matrix wide summary.json from the CPU collector")
    args = parser.parse_args()
    rows = json.loads((SOURCE / "summary.json").read_bytes())
    if [r["e"] for r in rows] != [16001, 32749, 58631, 65521, 106501, 131071, 225287, 262139]:
        raise ValueError("Incomplete CPU matrix")
    prime = prime_samples(PRIME / "screen_ed_stage2_matrix.log")
    if set(prime) != {r["e"] for r in rows}:
        raise ValueError("Prime95 input set differs from CPU matrix")
    version, timings = current_samples(args.current, rows) if args.current else (None, {})
    width = .16 if version else .25
    lines, table = [], []
    fig, ax = plt.subplots(figsize=(13, 5.2) if version else (12, 5.2), layout="constrained")
    for i, row in enumerate(rows):
        e, b, factors = row["e"], row["b2"], row["factors"]
        n, rem = divmod((1 << e) - 1, math.prod(factors))
        if rem or n.bit_length() != row["bits"]:
            raise ValueError(f"Invalid cofactor for M{e}")
        suffix = ',"' + ",".join(map(str, factors)) + '"' if factors else ""
        lines.append(f"ECM2=1,2,{e},-1,1000,{b},1,26{suffix}")
        times = []
        for threads, key, offset, color in ((1, "one", -.26, "#1856a4"), (4, "four", 0, "#008391")):
            values = row[key]
            for repeat, seconds in enumerate(values, 1):
                p = SOURCE / row["directory"] / f"{e}_{b}_{threads}_{repeat}" / "result.jsonl"
                result = json.loads(p.read_bytes())
                if (result["status"], result["b1"], result["b2"], result["exponent"],
                    result["Edwards"]["sigma"], result["stage2-engine"]["seconds"]) != (
                        "NF", 1000, b, e, 26, seconds):
                    raise ValueError(f"Result mismatch in {p}")
            if len(values) != 2:
                raise ValueError("Expected two complete curves per condition")
            if version:
                offset = -.34 if threads == 1 else 0
            label = f"GWpoly {threads}T"
            if version:
                label = f"0.66.1   {label}"
            times.append(timing_bar(ax, i + offset, values, width, color, label if i == 0 else None,
                                    historical=bool(version)))
            if version:
                current_label = f"{version}   GWpoly {threads}T" if i == 0 else None
                times.append(timing_bar(ax, i + offset + .17, timings[e, threads], width, color, current_label))
        requested, actual, elapsed, fresh = prime[e]
        display = "—"
        if fresh and requested == b:
            prime_label = "Prime95 1T (rounded B2)" if version else "Prime95 1T"
            ax.bar(i + (.34 if version else .26), elapsed, width, color="#333333",
                   label=prime_label if i == 0 else None)
            display = f"{elapsed:.3f}"
        cpu_cells = " | ".join(f"{value:.3f}" for value in times)
        table.append(f"| {e} | {n.bit_length()} | {b:,} | {cpu_cells} | {actual:,} | {display} |")
        if i and rows[i-1]["b2"] != b:
            ax.axvline(i-.5, ls="--", lw=1, color="0.48")
    start = 0
    for i, row in enumerate(rows):
        if i == len(rows)-1 or rows[i+1]["b2"] != row["b2"]:
            label = f"Requested B2 = {row['b2']:,}" if version else f"B2={row['b2']:.3g}"
            ax.text((start+i)/2, .97, label, transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=9)
            start = i+1
    ax.set_xticks(range(len(rows)), [f"{r['bits']:,}" for r in rows])
    ax.set(xlabel="Cofactor N (bits)", ylabel="Seconds per curve",
           title="Edwards Stage 2   |   Ryzen AI 9 HX 370   |   B1 = 1,000")
    ax.set_ylim(0, ax.get_ylim()[1]*1.14)
    if version:
        ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.10), ncol=5, frameon=False, fontsize=9)
    else:
        ax.legend(loc="upper left", bbox_to_anchor=(0,1.13), ncol=3, frameon=False)
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)
    if version:
        figures = ROOT / f"data/figures/stage2_cpu_wide_{version.replace('.', '')}"
        figures.mkdir(parents=True, exist_ok=True)
        output = figures / f"stage2-cpu-wide-b1000-{version}.png"
        fig.savefig(output, dpi=160)
        fig.savefig(output.with_suffix(".svg"))
        shutil.copyfile(output, DOC / "attachments" / output.name)
    else:
        fig.savefig(DOC / "attachments/stage2-cpu-wide-b1000.png", dpi=160)
        (PRIME / "worktodo_ecm2_stage2_wide_b1000.txt").write_text("\n".join(lines)+"\n", encoding="ascii")
    plt.close(fig)
    print("\n".join(table))


if __name__ == "__main__":
    main()
