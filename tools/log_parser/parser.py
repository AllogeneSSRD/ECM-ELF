"""ECM (Prime95/mprime) worker log parser.

Segments a screen log into individual ECM *runs*. A run is opened by every
``ECM on M...: Edwards/Montgomery curve #N`` line and closed by the next ``ECM on`` line
for the same worker (or end of file). All Stage / memory / FFT lines that fall
inside that window are attributed to the run.
"""

import re
from datetime import datetime

LINE_RE = re.compile(
    r"^\[(?P<worker>[^\]]+)\]\s+\[(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]\s?(?P<msg>.*)$"
)

WORKER_NUM_RE = re.compile(r"Worker #(\d+)")


def worker_number(label):
    """Prime95 uses [Worker] when only one worker exists."""
    if label == "Worker":
        return 1
    match = WORKER_NUM_RE.fullmatch(label)
    return int(match.group(1)) if match else None

ECM_RE = re.compile(
    r"ECM on M(?P<exp>\d+):\s*(?P<curve_type>Edwards|Montgomery) curve #(?P<curve>\d+)\s*"
    r"with s=(?P<s>\d+),\s*B1=(?P<b1>\d+),\s*B2=(?P<b2tbd>TBD|\d+)"
)

# FFT lines. The FFT "type" may contain hyphens/plus (e.g. AVX-512, FMA3).
# "Using" sets the stage-1 FFT at worker/exponent start; "Switching to" is the
# (usually larger) stage-2 FFT; "Switching back to" restores the stage-1 FFT
# for the next curve. A worker only prints "Using" the first time it runs an
# exponent, so we must track the worker's *current* FFT across all three kinds.
FFT_LINE_RE = re.compile(
    r"(?P<kind>Using|Switching to|Switching back to) "
    r"(?P<type>[\w+-]+) FFT length (?P<len>\d+)"
)

B2_RE = re.compile(
    r"Actual B2 will be (?P<b2>\d+).*?Curve is worth (?P<worth>\d+(?:\.\d+)?)"
)
AVAIL_MEM_RE = re.compile(r"Available memory is (?P<mem>\d+)MB")
USING_MEM_RE = re.compile(r"Using (?P<mem>\d+)MB of memory")
STAGE1_RE = re.compile(r"Stage 1 complete\..*?Total time:\s*(?P<t>\d+(?:\.\d+)?)\s*sec")
STAGE2_INIT_RE = re.compile(
    r"Stage 2 init complete\..*?Time:\s*(?P<t>\d+(?:\.\d+)?)\s*sec"
)
STAGE2_RE = re.compile(
    r"Stage 2 complete\..*?Total time:\s*(?P<t>\d+(?:\.\d+)?)\s*sec"
)
STAGE2_GCD_RE = re.compile(
    r"Stage 2 GCD complete\..*?Time:\s*(?P<t>\d+(?:\.\d+)?)\s*sec"
)

TS_FMT = "%Y-%m-%d %H:%M:%S"


def _new_run(worker_label, worker_num, ts, m, cur_fft, line_number):
    return {
        "worker": worker_num,
        "worker_label": worker_label,
        "exponent": int(m.group("exp")),
        "curve": int(m.group("curve")),
        "curve_type": m.group("curve_type"),
        "s": m.group("s"),
        "b1": int(m.group("b1")),
        "b2_requested": None if m.group("b2tbd") == "TBD" else int(m.group("b2tbd")),
        "b2": None,
        "worth": None,
        "avail_mem": None,
        "using_mem": None,
        "s1_time": None,
        "s2_init_time": None,
        "s2_time": None,
        "s2_gcd_time": None,
        "s1_fft": cur_fft[1] if cur_fft else None,
        "s1_fft_type": cur_fft[0] if cur_fft else None,
        "s2_fft": None,
        "s2_fft_type": None,
        "start_ts": ts,
        "end_ts": ts,
        "start_line": line_number,
        "end_line": line_number,
        # timestamps used for the gantt breakdown (not exported columns)
        "s1_end_ts": None,
        "s2_start_ts": None,
        "s2_end_ts": None,
        "status": "interrupted",
    }


def _finalize(run):
    if run["s2_time"] is not None:
        run["status"] = "complete"
    elif run["s1_time"] is not None:
        run["status"] = "stage1-only"
    else:
        run["status"] = "interrupted"
    return run


def parse_log(text):
    """Parse raw log text -> list of run dicts (in start order)."""
    runs = []
    open_run = {}       # worker_num -> run dict
    current_fft = {}    # worker_num -> (type, length)  (worker's live FFT)

    for line_number, raw in enumerate(text.splitlines(), 1):
        m = LINE_RE.match(raw.rstrip("\r"))
        if not m:
            continue
        worker_label = m.group("worker")
        ts = m.group("ts")
        msg = m.group("msg")

        wnum = worker_number(worker_label)
        if wnum is None:
            # [Main window] / [Comm window] never produce ECM runs.
            continue

        # FFT change line: update the worker's live FFT. If it's a stage-2
        # "Switching to" inside an open run, record the run's S2 FFT.
        fft = FFT_LINE_RE.search(msg)
        if fft:
            ftype, flen = fft.group("type"), int(fft.group("len"))
            current_fft[wnum] = (ftype, flen)
            run = open_run.get(wnum)
            if (
                fft.group("kind") == "Switching to"
                and run is not None
                and run["s1_time"] is not None
                and run["s2_fft"] is None
            ):
                run["s2_fft"] = flen
                run["s2_fft_type"] = ftype
                run["end_ts"] = ts
            continue

        # New ECM run.
        ecm = ECM_RE.search(msg)
        if ecm:
            if wnum in open_run:
                runs.append(_finalize(open_run.pop(wnum)))
            open_run[wnum] = _new_run(
                worker_label, wnum, ts, ecm, current_fft.get(wnum), line_number
            )
            continue

        run = open_run.get(wnum)
        if run is None:
            continue

        run["end_ts"] = ts
        run["end_line"] = line_number

        b2m = B2_RE.search(msg)
        if b2m:
            run["b2"] = int(b2m.group("b2"))
            run["worth"] = float(b2m.group("worth"))

        am = AVAIL_MEM_RE.search(msg)
        if am:
            run["avail_mem"] = int(am.group("mem"))

        umem = USING_MEM_RE.search(msg)
        if umem:
            run["using_mem"] = int(umem.group("mem"))

        s1 = STAGE1_RE.search(msg)
        if s1:
            run["s1_time"] = float(s1.group("t"))
            run["s1_end_ts"] = ts

        s2i = STAGE2_INIT_RE.search(msg)
        if s2i:
            run["s2_init_time"] = float(s2i.group("t"))
            run["s2_start_ts"] = ts

        s2 = STAGE2_RE.search(msg)
        if s2:
            run["s2_time"] = float(s2.group("t"))
            run["s2_end_ts"] = ts

        s2g = STAGE2_GCD_RE.search(msg)
        if s2g:
            run["s2_gcd_time"] = float(s2g.group("t"))
            run["end_ts"] = ts

    for run in open_run.values():
        runs.append(_finalize(run))

    # When S1 FFT == S2 FFT the program prints no "Switching to" line, so a run
    # that entered stage 2 without a switch keeps the same FFT as stage 1.
    for run in runs:
        entered_stage2 = run["s2_time"] is not None or run["s2_init_time"] is not None
        if run["s2_fft"] is None and entered_stage2:
            run["s2_fft"] = run["s1_fft"]
            run["s2_fft_type"] = run["s1_fft_type"]

    runs.sort(key=lambda r: (r["start_ts"], r["worker"]))
    return runs


# Column definitions shared with the export layer. (key, header)
COLUMNS = [
    ("worker_label", "Worker"),
    ("exponent", "Exponent"),
    ("curve", "Curve"), # Curve #
    ("s", "s"),
    ("b1", "B1"),
    ("b2", "B2 (Actual)"),
    ("worth", "Worth"),
    ("avail_mem", "mem (MB)"), # Available mem (MB)
    ("using_mem", "Using (MB)"), # Using mem (MB)
    ("s1_time", "S1 time (s)"), # Stage 1 time (s)
    ("s2_init_time", "S2 init (s)"), # Stage 2 init time (s)
    ("s2_time", "S2 time (s)"), # Stage 2 time (s)
    ("s2_gcd_time", "GCD (s)"), # Stage 2 GCD time (s)
    ("s1_fft", "S1 FFT"),
    ("s1_fft_type", "type"),
    ("s2_fft", "S2 FFT"),
    ("s2_fft_type", "type"),
    ("start_ts", "Start"),
    ("end_ts", "End"),
    ("status", "Status"),
]


if __name__ == "__main__":
    import sys
    import json

    path = sys.argv[1] if len(sys.argv) > 1 else "screen_example.log"
    with open(path, encoding="utf-8", errors="replace") as fh:
        rows = parse_log(fh.read())
    print(f"Parsed {len(rows)} runs")
    print(json.dumps(rows[:3], indent=2, ensure_ascii=False))
