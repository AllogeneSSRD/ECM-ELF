#!/usr/bin/env python3
"""Plot two-date ECM progress changes and submission growth by date/range.

History is sparse: the state on D is the last row with report_date <= D.
The heatmap assigns each positive change to the date it was first reported.
"""

from __future__ import annotations

import argparse
import math
import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm
from matplotlib.patches import Patch

from plot_ecm import (CURVES_TO_TEST, LEVELS, apply_xticks, draw_vgrid,
                      load_rows, parse_xtick_gap, progress_y)


ROOT = Path(__file__).parent
DEFAULT_DB = Path(__file__).parent / "ecm_progress.db"
DEFAULT_OUT_DIR = ROOT / "figures"
LEVEL_PREFIX = {
    level: sum(CURVES_TO_TEST[prior] for prior in LEVELS if prior < level)
    for level in LEVELS
}


def parse_date(value: str) -> str:
    if len(value) != 8 or not value.isdigit():
        raise argparse.ArgumentTypeError("date must be YYYYMMDD")
    try:
        datetime.strptime(value, "%Y%m%d")
    except ValueError as exc:
        raise argparse.ArgumentTypeError("date must be YYYYMMDD") from exc
    return value


def history_dates(conn: sqlite3.Connection) -> list[str]:
    return [
        str(row[0])
        for row in conn.execute(
            "SELECT DISTINCT report_date FROM progress_history ORDER BY report_date"
        )
    ]


def cumulative_curves(t_level: int, curves: int | None) -> int:
    """Completed lower levels plus work recorded in the current level."""
    if t_level not in CURVES_TO_TEST:
        raise ValueError(f"Unknown t_level {t_level}")
    cap = CURVES_TO_TEST[t_level]
    if curves is not None and curves < 0:
        raise ValueError(f"Negative curves at t_level {t_level}: {curves}")
    return LEVEL_PREFIX[t_level] + (cap if curves is None else min(curves, cap))


def progress_changes(
    conn: sqlite3.Connection, lo: int, hi: int, start: str, end: str
) -> list[tuple[int, float | None, float]]:
    """End-date exponents with optional start-date progress, in exponent order."""
    before = {exp: progress_y(t, c) for exp, t, c, _ in load_rows(conn, lo, hi, start)}
    after = load_rows(conn, lo, hi, end)
    return [(exp, before.get(exp), progress_y(t, c)) for exp, t, c, _ in after]


@dataclass(frozen=True)
class ChangeBucket:
    exp_rep: int
    old: float | None
    new: float
    first_y: float | None
    has_gain: bool
    has_loss: bool


def compress_progress_changes(
    rows: list[tuple[int, float | None, float]], compress: int, agg: str = "avg"
) -> list[ChangeBucket]:
    """Group consecutive end-date exponents without mixing first-seen into deltas."""
    if compress < 1 or agg not in {"avg", "max"}:
        raise ValueError("compress must be >= 1 and agg must be avg or max")
    reduce_values = max if agg == "max" else lambda values: sum(values) / len(values)
    buckets: list[ChangeBucket] = []
    for i in range(0, len(rows), compress):
        chunk = rows[i : i + compress]
        matched = [(old, new) for _, old, new in chunk if old is not None]
        first = [new for _, old, new in chunk if old is None]
        first_y = float(reduce_values(first)) if first else None
        buckets.append(ChangeBucket(
            exp_rep=chunk[0][0],
            old=float(reduce_values([old for old, _ in matched])) if matched else None,
            new=float(reduce_values([new for _, new in matched])) if matched else first_y,
            first_y=first_y,
            has_gain=any(new > old for old, new in matched),
            has_loss=any(new < old for old, new in matched),
        ))
    return buckets


def submission_heatmap(
    conn: sqlite3.Connection,
    lo: int,
    hi: int,
    dates: list[str],
    end: str,
    range_size: int,
    metric: str = "curves",
) -> tuple[np.ndarray, int, int]:
    """Positive event deltas per report date and fixed exponent range.

    First observations have no measured delta. Decreases are counted separately
    and never cancel genuine growth in the same cell.
    """
    if not dates or range_size < 1 or metric not in {"curves", "progress"}:
        raise ValueError("invalid heatmap dates, range_size, or metric")
    cells = np.zeros((len(dates), (hi - lo) // range_size + 1), dtype=float)
    date_index = {date: index for index, date in enumerate(dates)}
    previous: dict[int, float] = {}
    corrections = first_seen = 0
    measure = cumulative_curves if metric == "curves" else progress_y
    for exp, date, t_level, curves in conn.execute(
        """
        SELECT exponent, report_date, t_level, curves
        FROM progress_history
        WHERE exponent BETWEEN ? AND ? AND report_date <= ?
        ORDER BY report_date, exponent
        """,
        (lo, hi, end),
    ):
        exp = int(exp)
        value = float(measure(int(t_level), curves))
        old = previous.get(exp)
        if date in date_index:
            if old is None:
                first_seen += 1
            elif value > old:
                cells[date_index[date], (exp - lo) // range_size] += value - old
            elif value < old:
                corrections += 1
        previous[exp] = value
    return cells, corrections, first_seen


def heatmap_height_px(dates: list[str], max_height_px: int, height_span_days: int) -> int:
    """Scale the image to its calendar span, capped at max_height_px."""
    if not dates or max_height_px < 1 or height_span_days < 1:
        raise ValueError("invalid heatmap geometry")
    span_days = (datetime.strptime(dates[-1], "%Y%m%d")
                 - datetime.strptime(dates[0], "%Y%m%d")).days + 1
    return min(max_height_px, max(min(320, max_height_px),
                                  math.ceil(max_height_px * span_days / height_span_days)))


def heatmap_width_px(n_ranges: int, max_width_px: int) -> int:
    if n_ranges < 1 or max_width_px < 500:
        raise ValueError("heatmap width limit must be at least 500 pixels")
    return min(max_width_px, max(900, 900 + n_ranges * 10))


def selected_heatmap_dates(dates: list[str], start: str | None, end: str,
                           last_days: int | None) -> list[str]:
    if last_days is not None:
        window_start = (datetime.strptime(end, "%Y%m%d")
                        - timedelta(days=last_days - 1)).strftime("%Y%m%d")
        start = max(start or dates[0], window_start)
    else:
        start = start or dates[0]
    return [date for date in dates if start <= date <= end]


def progress_start_date(dates: list[str], start: str | None, end: str,
                        last_days: int | None) -> str | None:
    """Use the day before a lookback window as its comparison baseline."""
    if last_days is None:
        return start or next((date for date in reversed(dates) if date < end), None)
    baseline = (datetime.strptime(end, "%Y%m%d")
                - timedelta(days=last_days)).strftime("%Y%m%d")
    return max(start or dates[0], baseline)


def heatmap_xticks(lo: int, hi: int, range_size: int,
                   xtick_gap: str | int) -> tuple[list[int], list[str]]:
    """Exponent-space ticks mapped to heatmap bins, with one label per bin."""
    n_ranges = (hi - lo) // range_size + 1
    if n_ranges == 1:
        return [0], [str(lo)]
    if xtick_gap == "auto":
        marks = list(dict.fromkeys(int(i) for i in np.linspace(0, n_ranges - 1,
                                                               min(8, n_ranges))))
        return marks, [str(hi if i == n_ranges - 1 else lo + i * range_size)
                       for i in marks]

    gap = int(xtick_gap)
    if gap <= 0:
        raise ValueError("--xtick-gap must be positive or 'auto'")
    positions: list[int] = []
    labels: list[str] = []
    for i in range(n_ranges):
        left = lo + i * range_size
        right = min(hi, left + range_size - 1)
        multiple = ((left + gap - 1) // gap) * gap
        if multiple <= right:
            positions.append(i)
            labels.append(str(multiple))
    if positions[0:1] != [0]:
        positions.insert(0, 0)
        labels.insert(0, str(lo))
    if positions[-1] != n_ranges - 1:
        positions.append(n_ranges - 1)
        labels.append(str(hi))
    return positions, labels


def draw_progress_change(
    rows: list[tuple[int, float | None, float]],
    lo: int,
    hi: int,
    start: str,
    end: str,
    out: Path,
    dpi: int,
    xtick_gap: str | int = 1000,
    vgrid: bool = False,
    compress: int = 1,
    compress_agg: str = "avg",
) -> None:
    if not rows:
        raise ValueError("No end-date exponents in the selected range")
    buckets = compress_progress_changes(rows, compress, compress_agg)
    exponents = [bucket.exp_rep for bucket in buckets]
    n = len(buckets)
    x = np.arange(n)
    old = np.array([bucket.old if bucket.old is not None else 0.0 for bucket in buckets])
    new = np.array([bucket.new for bucket in buckets])
    first = np.array([bucket.old is None for bucket in buckets])
    first_y = np.array([bucket.first_y if bucket.first_y is not None else 0.0
                        for bucket in buckets])
    first_mark = np.array([bucket.first_y is not None for bucket in buckets])
    gain = np.maximum(new - old, 0.0)
    loss = np.maximum(old - new, 0.0)
    gain[first] = 0.0
    loss[first] = 0.0

    fig, ax = plt.subplots(figsize=(max(10, min(60, n / dpi)), 8), dpi=dpi)
    ax.bar(x, old, width=1, align="edge", color="#9ca3af", linewidth=0)
    ax.bar(x, gain, bottom=old, width=1, align="edge", color="#0d9488", linewidth=0)
    ax.bar(x, loss, bottom=new, width=1, align="edge", color="#dc2626", linewidth=0)
    if first.any():
        ax.bar(x[first], new[first], width=1, align="edge", color="#7c3aed", linewidth=0)
    low = min(min(old[~first], default=float(new.min())), float(new.min()))
    high = max(float(old.max()), float(new.max()), float(first_y.max()))
    floor = max((level for level in LEVELS if level <= low), default=LEVELS[0])
    ceil = min((level for level in LEVELS if level >= high), default=LEVELS[-1])
    ax.set_ylim(floor - 0.5, max(high, ceil) + 0.5)
    ax.set_xlim(0, n)
    # The measured delta can be much thinner than one screen pixel. Mark its
    # location without changing the segment's actual height.
    for mask, ys, color, marker, offset in (
        (np.array([bucket.has_gain for bucket in buckets]), new, "#0f766e", "o", 0.35),
        (np.array([bucket.has_loss for bucket in buckets]), new, "#b91c1c", "v", 0.5),
        (first_mark, first_y, "#6d28d9", "^", 0.65),
    ):
        if mask.any():
            ax.scatter(x[mask] + (offset if compress > 1 else 0.5), ys[mask],
                       s=28, c=color, marker=marker,
                       edgecolors="white", linewidths=0.5, zorder=12)
    for level in LEVELS:
        if floor <= level <= max(high, ceil):
            ax.axhline(level, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    tick_pos = apply_xticks(ax, exponents, xtick_gap)
    if vgrid:
        draw_vgrid(ax, tick_pos)
    xlabel = "Mersenne exponent (dense order)"
    if compress > 1:
        xlabel += f", {compress} exponents/bar ({compress_agg})"
    ax.set_xlabel(xlabel)
    ax.set_ylabel("ECM progress (digit-level equivalent)")
    ax.set_title(f"ECM progress change  [{lo}, {hi}]  {start} → {end}")
    ax.legend(
        handles=[
            Patch(color="#9ca3af", label=f"as of {start}"),
            Patch(color="#0d9488", label="growth"),
            Patch(color="#dc2626", label="correction / decrease"),
            Patch(color="#7c3aed", label="first observed by end date"),
        ],
        loc="upper right",
        fontsize=8,
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(
        f"Wrote {out}: bars={n} from {len(rows)} exponents, "
        f"increased={sum(new > old for _, old, new in rows if old is not None)}, "
        f"decreased={sum(new < old for _, old, new in rows if old is not None)}, "
        f"first_seen={sum(old is None for _, old, _ in rows)}"
    )


def draw_submission_heatmap(
    cells: np.ndarray,
    dates: list[str],
    lo: int,
    hi: int,
    range_size: int,
    metric: str,
    out: Path,
    dpi: int,
    max_height_px: int = 800,
    height_span_days: int = 60,
    max_width_px: int = 2800,
    xtick_gap: str | int = 1000,
    vgrid: bool = False,
    hide_zero: bool = False,
) -> None:
    n_dates, n_ranges = cells.shape
    height_px = heatmap_height_px(dates, max_height_px, height_span_days)
    width_px = heatmap_width_px(n_ranges, max_width_px)
    fig, ax = plt.subplots(
        figsize=(width_px / dpi, height_px / dpi),
        dpi=dpi,
    )
    cmap = plt.get_cmap("YlOrRd").copy()
    if hide_zero:
        cmap.set_bad((1, 1, 1, 0))
        ax.set_facecolor("white")
    image = ax.imshow(
        np.ma.masked_equal(cells, 0) if hide_zero else cells,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=PowerNorm(gamma=0.5, vmin=0, vmax=max(1.0, float(cells.max()))),
    )
    x_marks, x_labels = heatmap_xticks(lo, hi, range_size, xtick_gap)
    ax.set_xticks(x_marks, labels=x_labels, rotation=60, ha="right")
    if vgrid:
        draw_vgrid(ax, x_marks)
    y_marks = np.linspace(0, n_dates - 1, min(n_dates, max(2, height_px // 55)), dtype=int)
    ax.set_yticks(y_marks, labels=[dates[i] for i in y_marks])
    ax.set_xlabel("Mersenne exponent (range bins)")
    ax.set_ylabel("Report date")
    ax.set_title(f"ECM submission growth  {dates[0]} → {dates[-1]}")
    unit = "new curves" if metric == "curves" else "T-level"
    if not hide_zero or np.any(cells > 0):
        fig.colorbar(image, ax=ax, label=f"Growth ({unit})")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(
        f"Wrote {out}: size={width_px}x{height_px}px, "
        f"report_dates={n_dates}, ranges={n_ranges}, "
        f"positive_growth={cells.sum():.3f} {unit}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", choices=("change", "progress", "heatmap", "both"),
                        default="change", help="Plot to write (default: change)")
    parser.add_argument("--db", type=Path, default=Path(os.environ.get("ECM_DB", DEFAULT_DB)))
    parser.add_argument("--lo", type=int, default=1)
    parser.add_argument("--hi", type=int, default=20000)
    parser.add_argument("--from", dest="start", type=parse_date, metavar="YYYYMMDD")
    parser.add_argument("--to", dest="end", type=parse_date, metavar="YYYYMMDD")
    parser.add_argument("--range-size", type=int, default=1000,
                        help="Exponent width of one heatmap column (default 1000)")
    parser.add_argument("--xtick-gap", type=parse_xtick_gap, default=1000,
                        metavar="N|auto",
                        help="X ticks at exponent multiples of N (default 1000), or auto")
    parser.add_argument("--vgrid", action="store_true",
                        help="Draw vertical dotted lines at X tick marks")
    parser.add_argument("--compress", type=int, default=1,
                        help="Change plot: merge N consecutive exponents per bar (default 1)")
    parser.add_argument("--compress-agg", choices=("avg", "max"), default="avg",
                        help="Change plot: aggregate compared progress within each bar")
    parser.add_argument("--last-days", type=int,
                        help="Use the last N calendar days ending at --to")
    parser.add_argument("--max-height-px", type=int, default=800,
                        help="Maximum heatmap image height in pixels (default 800)")
    parser.add_argument("--max-width-px", type=int, default=2800,
                        help="Maximum heatmap image width in pixels (default 2800)")
    parser.add_argument("--height-span-days", type=int, default=60,
                        help="Calendar days corresponding to maximum height (default 60)")
    parser.add_argument("--metric", choices=("curves", "progress"), default="curves",
                        help="Heatmap growth unit (default: cumulative curves)")
    parser.add_argument("--hide-zero", action="store_true",
                        help="Heatmap: leave zero-growth cells unpainted on white background")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--dpi", type=int, default=120)
    args = parser.parse_args()
    if (args.lo > args.hi or args.range_size < 1 or args.compress < 1 or args.dpi < 1
            or args.max_height_px < 1 or args.max_width_px < 500
            or args.height_span_days < 1
            or (args.last_days is not None and args.last_days < 1)):
        parser.error("require --lo <= --hi, width >= 500, and positive range, compress, day, height, and dpi values")
    if args.plot == "heatmap" and (args.compress != 1 or args.compress_agg != "avg"):
        parser.error("--compress applies to change; use --range-size for heatmap")
    if args.hide_zero and args.plot not in {"heatmap", "both"}:
        parser.error("--hide-zero requires --plot heatmap or both")
    if args.start and args.end and args.start > args.end:
        parser.error("--from must not be after --to")
    if not args.db.is_file():
        parser.error(f"database not found: {args.db}")

    # Read-only URI also prevents an accidental database creation/write.
    conn = sqlite3.connect(f"file:{args.db.resolve().as_posix()}?mode=ro", uri=True)
    try:
        dates = history_dates(conn)
        if not dates:
            parser.error("progress_history is empty")
        end = args.end or dates[-1]
        if end < dates[0]:
            parser.error(f"--to is before the first report ({dates[0]})")
        if args.plot in {"change", "progress", "both"}:
            start = progress_start_date(dates, args.start, end, args.last_days)
            if start is None or start < dates[0] or start >= end:
                parser.error("progress plot needs two ordered dates with a start snapshot")
            rows = progress_changes(conn, args.lo, args.hi, start, end)
            suffix = (f"_compress{args.compress}_{args.compress_agg}"
                      if args.compress > 1 else "")
            out = args.out_dir / (
                f"ecm_progress_change_{args.lo}-{args.hi}_{start}_{end}{suffix}.png"
            )
            draw_progress_change(rows, args.lo, args.hi, start, end, out,
                                 args.dpi, args.xtick_gap, args.vgrid,
                                 args.compress, args.compress_agg)
        if args.plot in {"heatmap", "both"}:
            selected = selected_heatmap_dates(dates, args.start, end, args.last_days)
            if not selected:
                parser.error("no report dates in the selected heatmap interval")
            cells, corrections, first_seen = submission_heatmap(
                conn, args.lo, args.hi, selected, end, args.range_size, args.metric
            )
            out = args.out_dir / (
                f"ecm_submission_heatmap_{args.lo}-{args.hi}_"
                f"{selected[0]}_{selected[-1]}_step{args.range_size}_{args.metric}"
                f"{'_hidezero' if args.hide_zero else ''}.png"
            )
            draw_submission_heatmap(
                cells, selected, args.lo, args.hi, args.range_size, args.metric,
                out, args.dpi, args.max_height_px, args.height_span_days,
                args.max_width_px, args.xtick_gap, args.vgrid, args.hide_zero
            )
            print(f"Heatmap excluded first_seen={first_seen}, corrections={corrections} events")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
