"""Shared layout for Stage1 and Stage2 version throughput figures."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


WIDTHS = {2003: (2, 1939), 4001: (4, 3600), 8011: (8, 7995), 16319: (16, 16272)}
PRIME95 = "Prime95"


def version_key(version):
    return (0, *map(int, version.split("."))) if version else (1,)


def plot_version_panels(samples, path, *, title, ylabel, colors, hatch, hatch_label, reference_label, note):
    max_versions = max(len({row.version for row in samples if row.exponent == p}) for p in WIDTHS)
    dense = max_versions > 10
    with plt.style.context("seaborn-v0_8-whitegrid"):
        fig, axes = plt.subplots(2, 2, figsize=(20, 9.4) if dense else (14, 8.4))
        fig.suptitle(title, fontsize=20, fontweight="semibold", y=.965)
        fig.supylabel(ylabel, fontsize=12, x=.036)
        for ax, (p, (size, cofactor)) in zip(axes.flat, WIDTHS.items()):
            panel = [row for row in samples if row.exponent == p]
            if sum(row.series == PRIME95 and not row.version for row in panel) != 1:
                raise ValueError(f"Expected one unversioned Prime95 reference for M{p}")
            versions = sorted({row.version for row in panel}, key=version_key)
            peak = max(row.throughput for row in panel)
            ax.set_title(f"~{size}K  ·  M{p}  |  {cofactor:,}-bit cofactor", fontsize=12, pad=9)
            for x, version in enumerate(versions):
                group = sorted((row for row in panel if row.version == version),
                               key=lambda row: list(colors).index(row.series))
                if not version:
                    ax.axvline(x - .5, color="#b0b8c2", linewidth=.9, linestyle="--")
                for i, row in enumerate(group):
                    color = colors[row.series]
                    offset = .32 * (i - (len(group) - 1) / 2)
                    hatched = row.timing == hatch
                    bars = ax.bar(x + offset, row.throughput, width=.21, color="white" if hatched else color,
                                  edgecolor=color, hatch="////" if hatched else None, zorder=3)
                    digits = 2 if row.throughput >= 10 else (4 if row.throughput < .1 else 3)
                    near_previous = i and abs(row.throughput - group[i - 1].throughput) < .05 * peak
                    stagger = (len(versions) > 20 and x % 2 == 1) or near_previous
                    ax.bar_label(bars, labels=[f"{row.throughput:.{digits}f}"], padding=12 if stagger else 4,
                                 fontsize=7 if len(versions) > 14 else 8, color="#242831", zorder=5)
            ax.set_xticks(range(len(versions)), [v or PRIME95 for v in versions], fontsize=9)
            if len(versions) > 10:
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            ax.set_ylim(0, 1.16 * peak)
            ax.set_xlim(-.57, len(versions) - .43)
            ax.grid(axis="x", visible=False)
            ax.spines[["top", "right"]].set_visible(False)
        legend = [Patch(facecolor=color, label=reference_label if series == PRIME95 else series)
                  for series, color in colors.items()]
        if any(row.timing == hatch for row in samples):
            legend.append(Patch(facecolor="white", edgecolor="#4c5362", hatch="////", label=hatch_label))
        fig.legend(handles=legend, loc="lower center", bbox_to_anchor=(.5, .054), ncol=len(legend),
                   fontsize=10, frameon=False)
        fig.text(.5, .035, note, ha="center", fontsize=9, color="#4c5362")
        fig.subplots_adjust(left=.094, right=.981, top=.87, bottom=.17 if dense else .145,
                            hspace=.42 if dense else .31, wspace=.19)
        fig.savefig(path, dpi=180)
        plt.close(fig)
