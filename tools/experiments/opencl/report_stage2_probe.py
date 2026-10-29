"""Summarize the small native OpenCL Stage2 probe A/B (not production timings)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SETTINGS = ("serial", "batched_transient", "batched_copy", "batched_reuse")
LABELS = ("Serial", "Batched\ntransient", "Batched\ncopy", "Batched\nborrow + reuse")


def summarize(paths: list[Path]) -> dict:
    devices, identities = [], set()
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8-sig"))
        if report.get("status") != "passed" or report.get("bad") != 0:
            raise ValueError(f"Not a passed probe: {path}")
        manifest = json.loads((path.parent / "build_manifest.json").read_text(encoding="utf-8-sig"))
        identity = manifest["stage2_executable_sha256"]
        identities.add(identity)
        rows = report["benchmarks"]
        if not rows:
            raise ValueError(f"No benchmark rows: {path}")
        groups: dict[tuple[int, int, int], dict] = {}
        for row in rows:
            key = row["bits"], row["P"], row["giants"]
            if row["setting"] in groups.setdefault(key, {}):
                raise ValueError(f"Duplicate setting for {key}: {path}")
            groups[key][row["setting"]] = row
        comparisons = []
        for key, settings in sorted(groups.items()):
            if set(settings) != set(SETTINGS):
                raise ValueError(f"Incomplete settings for {key}: {path}")
            pairs = {row["pairs_per_run"] for row in settings.values()}
            if len(pairs) != 1:
                raise ValueError(f"Different mathematical work for {key}: {path}")
            time = {name: settings[name]["median_seconds"] for name in SETTINGS}
            if min(time.values()) <= 0:
                raise ValueError(f"Invalid timing for {key}: {path}")
            comparisons.append({
                "bits": key[0], "P": key[1], "giants": key[2],
                "serial_to_batched_speedup": time["serial"] / time["batched_reuse"],
                "workspace_time_reduction_pct": 100 * (1 - time["batched_reuse"] / time["batched_transient"]),
                "borrow_time_reduction_pct": 100 * (1 - time["batched_reuse"] / time["batched_copy"]),
                "settings": settings,
            })
        devices.append({"device": report["device"], "driver": report["driver"],
                        "source_report": str(path), "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "comparisons": comparisons})
    if len(identities) != 1:
        raise ValueError("Use the same executable for the within-revision A/B")
    return {"scope": "small polynomial pipeline; CPU points/GCD and initial uploads excluded",
            "executable_sha256": identities.pop(), "devices": devices}


def plot(summary: dict, output: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    devices = summary["devices"]
    heights = max(len(d["comparisons"]) for d in devices)
    fig, axes = plt.subplots(heights, len(devices), squeeze=False,
                             figsize=(6.2 * len(devices), 3.7 * heights), constrained_layout=True)
    for col, device in enumerate(devices):
        for row, case in enumerate(device["comparisons"]):
            ax = axes[row][col]
            values = [case["settings"][name] for name in SETTINGS]
            median = [v["median_seconds"] * 1000 for v in values]
            low = [max(0, t - v["min_seconds"] * 1000) for t, v in zip(median, values)]
            high = [max(0, v["max_seconds"] * 1000 - t) for t, v in zip(median, values)]
            bars = ax.bar(range(4), median, yerr=[low, high], capsize=3,
                          color=("#8994a6", "#d9a655", "#5587b2", "#228476"))
            ax.bar_label(bars, labels=[f"{v:.2f}" for v in median], padding=6, fontsize=9)
            ax.set_xticks(range(4), LABELS, fontsize=9)
            ax.set_ylabel("Polynomial pipeline wall time (ms)")
            ax.set_title(f"{device['device']}\n{case['bits']} bits, P={case['P']}, giants={case['giants']}", fontsize=11)
            ax.set_ylim(0, max(median[i] + high[i] for i in range(4)) * 1.23)
            ax.grid(axis="y", alpha=.2)
            ax.set_axisbelow(True)
    fig.suptitle("Native OpenCL Stage2: fixed-input quick A/B\n"
                 "Median of 3 runs; whiskers=min/max\nExplicit priming excluded", fontsize=13)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--summary", type=Path, default=Path("data/benchmarks/opencl_stage2_probe.json"))
    parser.add_argument("--figure", type=Path, default=Path("data/figures/opencl_stage2_probe.png"))
    args = parser.parse_args()
    summary = summarize(args.reports)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    plot(summary, args.figure)
    for device in summary["devices"]:
        for row in device["comparisons"]:
            print(f"{device['device']} {row['bits']}b: "
                  f"serial/batched={row['serial_to_batched_speedup']:.2f}x "
                  f"workspace={row['workspace_time_reduction_pct']:.1f}% "
                  f"borrow={row['borrow_time_reduction_pct']:.1f}%")


if __name__ == "__main__":
    main()
