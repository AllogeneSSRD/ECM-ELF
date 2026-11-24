#!/usr/bin/env python3
"""Plot two consecutive Stage1 NTT batches from monotonic span logs."""

import argparse
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np


LANES = ("I/O", "Curve prefetch", "CUDA path", "Host")
STYLE = {
    "select_batch": (3, "#4c78a8", "Batch selection"),
    "curve_setup": (3, "#4c78a8", "Curve construction"),
    "curve_reuse": (3, "#4c78a8", "Curves"),
    "select_plan": (3, "#4c78a8", "Plan selection"),
    "plan_reuse": (3, "#4c78a8", "Plan"),
    "cuda_init": (2, "#258073", "CUDA init"),
    "scalar_loop": (2, "#258073", "Main loop"),
    "prefetch_ready": (1, "#9664a9", "Next batch curves"),
    "normalize": (3, "#4c78a8", "Normalize"),
    "final_checkpoint": (0, "#b27333", "Final checkpoint"),
    "complete": (0, "#b27333", "Commit"),
    "reserve": (0, "#b27333", "Reserve"),
    "tune_write": (0, "#b27333", "Write tune"),
}


def read_spans(path):
    pattern = r"stage1_ntt_span: phase=(\w+) start_us=(\d+) end_us=(\d+)"
    spans = [(phase, int(start), int(end)) for phase, start, end in
             re.findall(pattern, path.read_text(encoding="utf-8", errors="replace"))]
    if any(end < start for _, start, end in spans):
        raise ValueError("Monotonic span ended before it started")
    return spans


def panel(ax, records, loop, origin, title):
    phase, begin, end = loop
    loop_start, loop_end = (begin-origin)/1e6, (end-origin)/1e6
    if phase != "scalar_loop" or loop_end-loop_start < 1.6:
        raise ValueError("Expected a main loop longer than 1.6 seconds")
    cut_left, cut_right = loop_start + .75, loop_end - .75
    ready_times = [(end-origin)/1e6 for phase, _, end in records if phase == "prefetch_ready"]
    for ready in ready_times:
        if cut_left < ready < cut_right:
            cut_right = max(cut_left + .12, ready - .10)
    break_width = .22
    removed = cut_right - cut_left - break_width

    def x(t):
        return t - removed if t >= cut_right else t

    def draw_gap(y, color):
        # Two complete waves represent the cut, independently of its duration.
        height = np.linspace(-.20, .20, 70)
        for frac in (.27, .73):
            wave = cut_left + break_width*frac + .027*np.sin(np.linspace(0, 4*np.pi, len(height)))
            ax.plot(wave, y+height, color=color, lw=1.1, clip_on=False)

    for phase, start, end in records:
        lane, color, label = STYLE[phase]
        left, right = (start-origin)/1e6, (end-origin)/1e6
        duration = right-left
        segments = [(left, min(right,cut_left))] if left < cut_left else []
        if right > cut_right:
            segments.append((max(left,cut_right),right))
        for lo, hi in segments:
            ax.barh(lane,x(hi)-x(lo),left=x(lo),height=.40,color=color,zorder=3)
        if left < cut_left < cut_right < right:
            draw_gap(lane,color)
        if phase == "prefetch_ready":
            ax.scatter(x(right),lane,color=color,marker="D",s=19,zorder=5)
            ax.annotate(f"Ready  {right:.3f} s", (x(right), lane), xytext=(4,10),
                        textcoords="offset points", fontsize=8)
            ax.text(x(left),lane-.26,f"{label}  {duration:.3f} s",
                    ha="left",va="top",fontsize=8)
        if phase == "final_checkpoint":
            ax.annotate(f"Checkpoint  {duration * 1000:.0f} ms",(x(right),lane),
                        xytext=(13,21),textcoords="offset points",fontsize=8,
                        arrowprops={"arrowstyle":"-", "lw":.65})
        if phase == "scalar_loop":
            ax.text((x(left)+x(right))/2, lane+.28, f"Main loop  {duration:.3f} s",
                    ha="center",va="bottom",fontsize=9)
        elif phase in {"select_batch", "select_plan"}:
            ax.text((left+right)/2,lane+.25,f"{label}\n{duration:.3f} s",
                    ha="center",va="bottom",fontsize=7)
        elif duration >= .25 and phase != "prefetch_ready":
            ax.text((x(left)+x(right))/2,lane+.27,f"{label}  {duration:.3f} s",
                    ha="center",va="bottom",fontsize=8)

    # Label the two sub-millisecond hits together. Their starts can be separated
    # by small uninstrumented host operations and are shown at their real times.
    reused = [(p,a,b) for p,a,b in records if p in {"curve_reuse","plan_reuse"}]
    if reused:
        values = ", ".join(f"{STYLE[p][2]} {(b-a)/1000:.3f} ms" for p,a,b in reused)
        place = (reused[-1][1]-origin)/1e6
        ax.annotate(values,(x(place),3),xytext=(12,18),textcoords="offset points",
                    arrowprops={"arrowstyle":"-", "lw":.65},fontsize=8)
        for _,start,_ in reused:
            ax.scatter(x((start-origin)/1e6),3,marker="|",s=90,color="#18314e",zorder=5)

    for pos in (cut_left,cut_left+break_width):
        ax.axvline(pos,lw=.6,color="#7b8490",ls=":",zorder=2)
    ticks = [t for t in range(0,int(loop_end)+1) if t < cut_left or t > cut_right]
    ax.set_xticks([x(t) for t in ticks], [str(t) for t in ticks])
    ax.set_yticks(range(len(LANES)),LANES)
    ax.set_ylim(-.44,3.93)
    ax.set_xlim(-.06,max(7.4,x(loop_end)+.65))
    ax.set_title(title,loc="left",fontweight="bold",pad=12)
    ax.grid(axis="x",color="#dce1e7",lw=.6)
    ax.spines[["top","right","left"]].set_visible(False)
    ax.tick_params(axis="y",length=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    spans = read_spans(args.log)
    loops = sorted((row for row in spans if row[0] == "scalar_loop"), key=lambda row: row[1])
    if len(loops) < 2:
        raise ValueError("The log must contain at least two main-loop spans")
    first, second = loops[:2]
    origin = min(a for p,a,b in spans if p=="select_batch")
    before = [row for row in spans if row[0] in STYLE and origin <= row[1] < first[2]]
    after = [row for row in spans if row[0] in STYLE and first[2] <= row[1] < second[2]]
    if not {"curve_reuse","plan_reuse","prefetch_ready"} <= {row[0] for row in after}:
        raise ValueError("Second batch must reuse curves and plan and complete a third-batch prefetch")
    plt.rcParams.update({"font.size":10, "font.family":"DejaVu Sans", "axes.edgecolor":"#536273"})
    fig, axes = plt.subplots(2,1,figsize=(10.6,5.4))
    fig.subplots_adjust(left=.15,right=.98,top=.83,bottom=.11,hspace=.75)
    fig.suptitle("Stage 1 NTT · Handoff between three-curve batches",x=.15,ha="left",fontweight="bold")
    fig.text(.15,.896,"RTX 4070 Ti  |  M3608153 cofactor  |  B1 = 10,000  |  NAF w8  |  163,840-point IBDWT",fontsize=9)
    panel(axes[0],before,first,origin,"(a) First batch · time since batch selection (s)")
    panel(axes[1],after,second,first[2],"(b) Reused batch · time since previous loop ended (s)")
    right = max(ax.get_xlim()[1] for ax in axes)
    for ax in axes:
        ax.set_xlim(-.06,right)
    fig.text(.15,.04,"Paired wavy marks omit the middle of the main loops. Tick labels retain measured elapsed time.",fontsize=8)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.output,dpi=220,facecolor="white")


if __name__ == "__main__":
    main()
