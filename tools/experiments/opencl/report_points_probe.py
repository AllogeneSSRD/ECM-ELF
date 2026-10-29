"""Summarize component timings; not a complete production Stage2 benchmark."""
from __future__ import annotations
import argparse
import json
import statistics
from pathlib import Path


def summarize(path: Path) -> dict:
    report = json.loads(path.read_text(encoding="utf-8-sig"))
    if report.get("status") != "passed":
        raise ValueError(f"Unsuccessful probe: {path}")
    manifest = json.loads((path.parent / "build_manifest.json").read_text(encoding="utf-8-sig"))
    groups: dict[tuple, list] = {}
    for row in report["rows"]:
        key = row["bits"], row.get("count"), row.get("gcd_count")
        groups.setdefault(key, []).append(row)
    rows = []
    for key, samples in sorted(groups.items(), key=lambda item: tuple(v or 0 for v in item[0])):
        if len(samples) != 3 or {r["repeat"] for r in samples} != {0, 1, 2}:
            raise ValueError(f"Incomplete samples: {key}")
        metrics = {}
        for name in samples[0]:
            if name in {"bits", "count", "gcd_count", "repeat"}:
                continue
            values = [r[name] for r in samples]
            metrics[name] = {"median": statistics.median(values), "min": min(values), "max": max(values)}
        rows.append({"bits": key[0], "count": key[1], "gcd_count": key[2], "metrics": metrics})
    return {"source": str(path), "device": report["device"], "driver": report["driver"],
            "executable_sha256": manifest["points_executable_sha256"],
            "cuda_comparison": report["cuda_comparison"], "repeats": 3, "rows": rows}


def plot(devices: list[dict], output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    for index, device in enumerate(devices):
        data = [r for r in device["rows"] if r["count"]]
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
        x = np.arange(len(data))
        labels = [str(r["bits"]) for r in data]
        specs = [
            ("Projective ladder: 1024 scattered 20-bit scalars", "ladder",
             [("cpu_ladder_mean", "CPU GMP mean"), ("cl_ladder_kernel", "OpenCL kernel"), ("cl_ladder_wall", "OpenCL call"), ("cuda_ladder_kernel", "CUDA production kernel")]),
            ("Segmented normalization: 32 points / inverse", "normalize",
             [("cpu_normalize_mean", "CPU GMP mean"), ("cl_normalize_kernel", "OpenCL kernel"), ("cl_normalize_wall", "OpenCL call")]),
            ("Products: 1024 values -> 16 blocks", "products",
             [("cpu_products_mean", "CPU GMP mean"), ("cl_products_kernel", "OpenCL kernel"), ("cl_products_wall", "OpenCL call"), ("cuda_products_kernel", "CUDA production kernel")]),
        ]
        for ax, (title, _, series) in zip(axes.flat, specs):
            series = [(key, name) for key, name in series if key in data[0]["metrics"]]
            width = .8 / len(series)
            for j, (key, name) in enumerate(series):
                values = [r["metrics"][key]["median"] * 1000 for r in data]
                ax.bar(x + (j - (len(series) - 1) / 2) * width, values, width, label=name)
            ax.set(title=title, ylabel="Milliseconds (log scale)", xlabel="Target bits", xticks=x, xticklabels=labels, yscale="log")
            ax.grid(axis="y", alpha=.2)
            ax.legend(fontsize=8)
        ax = axes[1, 1]
        for bits in labels:
            gcds = [r for r in device["rows"] if r["gcd_count"] and str(r["bits"]) == bits]
            counts = [r["gcd_count"] for r in gcds]
            ax.plot(counts, [r["metrics"]["cpu_gcd_seconds"]["median"] * 1000 for r in gcds], "o--", label=f"CPU {bits}b")
            ax.plot(counts, [r["metrics"]["cl_gcd_wall"]["median"] * 1000 for r in gcds], "s-", label=f"OpenCL {bits}b")
        ax.set(title="GCD: input resident; OpenCL includes verdict drain", xlabel="Independent GCDs", ylabel="Milliseconds", xscale="log", yscale="log")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, ncol=2)
        fig.suptitle(f"{device['device']} | n=3 | clocks not locked\nComponent experiment; not production Stage2 / Prime95", fontsize=13)
        target = output if len(devices) == 1 else output.with_name(f"{output.stem}_{index}{output.suffix}")
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=160)
        plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("inputs", type=Path, nargs="+")
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--figure", type=Path, required=True)
    args = p.parse_args()
    devices = [summarize(path) for path in args.inputs]
    if len({d["executable_sha256"] for d in devices}) != 1:
        raise ValueError("Mixed binary identities")
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps({"devices": devices}, ensure_ascii=False, indent=2), encoding="utf-8")
    plot(devices, args.figure)


if __name__ == "__main__":
    main()
