"""Publish the CPU wide Stage2 matrix and matching Prime95 work lines."""
import json
import math
import re
from pathlib import Path
from statistics import median

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/experiments/stage2_wide_b1000"
DOC = ROOT / "docs/performance"
PRIME = ROOT / "tools/benchmarks/prime95"


def prime_samples(path):
    """Keep only fresh curves with complete stage timers and a matching completion."""
    blocks = path.read_text(encoding="utf-8-sig").split("ECM on M")[1:]
    samples = {}
    for block in blocks:
        header = re.match(r"(\d+): Edwards curve #1 with s=26, B1=1000, B2=(\d+)", block)
        if not header:
            raise ValueError("Unexpected Prime95 curve header")
        e, requested = map(int, header.groups())
        end = re.search(rf"M{e} completed 1 ECM Edwards curve, B1=1000, B2=(\d+)", block)
        if not end or e in samples:
            raise ValueError(f"Incomplete or duplicate Prime95 curve M{e}")
        timers = []
        for phase in ("init complete", "complete", "GCD complete"):
            matches = re.findall(rf"Stage 2 {phase}\.[^\n]*?[Tt]ime: ([\d.]+) sec", block)
            if len(matches) != 1:
                raise ValueError(f"Missing or duplicate {phase} timer for M{e}")
            timers.append(float(matches[0]))
        samples[e] = requested, int(end[1]), sum(timers), "Stage 1 complete." in block
    return samples


def main():
    rows = json.loads((SOURCE / "summary.json").read_bytes())
    if [r["e"] for r in rows] != [16001, 32749, 58631, 65521, 106501, 131071, 225287, 262139]:
        raise ValueError("Incomplete CPU matrix")
    prime = prime_samples(PRIME / "screen_ed_stage2_matrix.log")
    if set(prime) != {r["e"] for r in rows}:
        raise ValueError("Prime95 input set differs from CPU matrix")
    lines, table = [], []
    fig, ax = plt.subplots(figsize=(12, 5.2), layout="constrained")
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
            m = median(values)
            times.append(m)
            ax.bar(i + offset, m, .25, color=color, label=f"GWpoly {threads}T" if i == 0 else None,
                   yerr=[[m-min(values)], [max(values)-m]], capsize=3)
        requested, actual, elapsed, fresh = prime[e]
        display = "—"
        if fresh and requested == b:
            ax.bar(i+.26, elapsed, .25, color="#333333", label="Prime95 1T" if i == 0 else None)
            display = f"{elapsed:.3f}"
        table.append(f"| {e} | {n.bit_length()} | {b:,} | {times[0]:.3f} | {times[1]:.3f} | {actual:,} | {display} |")
        if i and rows[i-1]["b2"] != b:
            ax.axvline(i-.5, ls="--", lw=1, color="0.48")
    start = 0
    for i, row in enumerate(rows):
        if i == len(rows)-1 or rows[i+1]["b2"] != row["b2"]:
            ax.text((start+i)/2, .97, f"B2={row['b2']:.3g}", transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=9)
            start = i+1
    ax.set_xticks(range(len(rows)), [f"{r['bits']:,}" for r in rows])
    ax.set(xlabel="Cofactor N (bits)", ylabel="Seconds per curve",
           title="Edwards Stage 2   |   Ryzen AI 9 HX 370   |   B1 = 1,000")
    ax.set_ylim(0, ax.get_ylim()[1]*1.14)
    ax.legend(loc="upper left", bbox_to_anchor=(0,1.13), ncol=3, frameon=False)
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)
    fig.savefig(DOC / "attachments/stage2-cpu-wide-b1000.png", dpi=160)
    plt.close(fig)
    (PRIME / "worktodo_ecm2_stage2_wide_b1000.txt").write_text("\n".join(lines)+"\n", encoding="ascii")
    print("\n".join(table))


if __name__ == "__main__":
    main()
