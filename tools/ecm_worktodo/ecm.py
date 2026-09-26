#!/usr/bin/env python3
"""worktodo pipeline -- Windows-native generator for the ECM stage-1 / stage-2 split.

GENERATION ONLY.  This tool never runs stage 1 or stage 2 and never rewrites a file that a
running Prime95 owns; it only turns assignment lines into the work files both consumers read.

  stage 1 : our driver (ecm_cuda.exe, queue mode) reads the ECMSTAGE2= lines
  stage 2 : Prime95 reads the ECM=/ECM2= lines (and/or the same ECMSTAGE2= lines, which is
            the format the stage-2 handoff was designed around -- the driver merely *also*
            accepts ECMSTAGE2= to run stage 1, a convenience feature with different
            semantics)

INPUT (both prefixes are equivalent; the prefix is only a spelling)

    ECM=/ECM2=  [<AID>,|N/A,|<empty>][FFT2=<fftl>,]<k>,<b>,<n>,<c>,<B1>[,<B2>][,<curves>]
                [,<sigma>][,"f1,f2,..."]

    defaults match Prime95 and our C++ parser (src/core/ecm_worktodo.cpp): B2 = 0,
    curves = 100.  NOTE: B2 = 0 means "let Prime95 pick B2 automatically", it does NOT mean
    "skip stage 2".

OUTPUT

    --out-ecmstage2 FILE   ECMSTAGE2=[<aid>,]<k>,<b>,<n>,<c>,"<save>",<B2>,<skip>,<curves>
                           [,"f1,f2,..."]
                           This single artifact is BOTH our stage-1 queue file and the
                           stage-2 handoff line, which is why there is no separate
                           --out-stage1 switch: copy it wherever it is needed.
                           The driver takes B1 from the save name, so the rendered name must
                           end in "_<B1>.save" (enforced, see --allow-invalid-save-name).

    --out-ecm FILE         ECM=/ECM2= lines for native Prime95 worktodo files, selected by
                           --ecm-prefix (default ECM2).  Plain line list by default;
                           --worker N wraps the batch in one "[Worker #N]" section, --append
                           appends.  Existing content is never parsed or preserved.

    --emit-cli FILE        One command line per task, for manual / batched runs.
      --emit-cli-kind      sh (gmp-ecm, LF) | ps1 | bat (our ecm_cuda.exe, CRLF)

PIPELINE

    parse -> filter -> dedup -> rewrite -> sort -> emit

    Filtering happens BEFORE dedup on purpose: filters pick the candidate rows the operator
    wants, and dedup then collapses duplicates among the survivors.  (Deduping first can
    silently drop a whole number when the "winning" row is filtered out afterwards.)

    Dedup identity is the mathematical object (k, b, n, c), compared numerically.  On a
    collision the winner is: largest B1 -> largest curves -> real AID over N/A or empty ->
    first occurrence.  Known-factor lists are UNIONED across the group (known factors only
    shrink N, and both output lines must agree on N).

EXAMPLES

    # native Prime95 worktodo + our stage-1 queue, same content, two destinations
    python ecm.py --input sorted.csv --set-b1 110e6 --gpu-curves 192 --sort-by n \\
        --out-ecmstage2 ..\\ECM\\pipeline\\worktodo_add.csv \\
        --out-ecm      ..\\prmers\\worktodo.txt --ecm-prefix ECM2

    # inspect without writing anything
    python ecm.py --input assignment.csv --min-curves 100 --dry-run

    # per-task command lines for our GPU driver
    python ecm.py --input sorted.csv --emit-cli worktodo.ps1 --emit-cli-kind ps1 --device 1
"""

from __future__ import annotations

import argparse
import csv
import io
import re
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

VERSION = "2.0"

# --------------------------------------------------------------------------------------
# token helpers (mirror the C++ parser so that emit -> parse is exactly reversible)
# --------------------------------------------------------------------------------------

_INT_RE = re.compile(r"^[+-]?\d+$")
_NUM_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
_SAVE_RE = re.compile(r"^(?P<stem>.+)_(?P<b1>[^_]+)\.save$")


def is_int_like(s: str) -> bool:
    """Same rule as the C++ is_int_like(): optional sign, then digits only."""
    return bool(_INT_RE.match(s.strip()))


def strtod(s: str) -> Optional[float]:
    """Parse a whole token as a C strtod() would (no underscores, no inf/nan)."""
    t = s.strip()
    if not _NUM_RE.match(t):
        return None
    try:
        return float(t)
    except ValueError:
        return None


def split_csv(text: str) -> List[str]:
    row = next(csv.reader(io.StringIO(text), skipinitialspace=True), None)
    if row is None:
        return []
    return [item.strip() for item in row]


def csv_quote(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


# --------------------------------------------------------------------------------------
# record
# --------------------------------------------------------------------------------------


@dataclass
class Task:
    """One assignment. k/b/c/b1/b2 keep the *original* spelling (110e6 stays 110e6)."""

    idx: int = 0                       # input order, for stable tie-breaks
    aid: str = ""
    fft2: str = ""
    k: str = ""
    b: str = ""
    c: str = ""
    n: int = 0
    b1: str = ""
    b2: str = "0"
    curves: int = 100
    sigma: str = ""
    factors: List[str] = field(default_factory=list)

    # -- identity / ordering ---------------------------------------------------------
    def number_key(self) -> Tuple[int, int, int, int]:
        """Numerical identity of k*b^n+c (so k=01 and k=1 are the same work)."""
        return (int(self.k), int(self.b), self.n, int(self.c))

    def b1_value(self) -> float:
        v = strtod(self.b1)
        return v if v is not None else 0.0

    def b2_value(self) -> float:
        v = strtod(self.b2)
        return v if v is not None else 0.0

    def has_real_aid(self) -> bool:
        return bool(self.aid) and self.aid.upper() != "N/A"

    def expression(self) -> str:
        c_int = int(self.c)
        sign = "+" if c_int >= 0 else "-"
        body = f"({self.k}*{self.b}^{self.n}{sign}{abs(c_int)})"
        if self.factors:
            return f"{body}/({'*'.join(self.factors)})"
        return body


# --------------------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------------------


class LineError(ValueError):
    pass


def parse_ecm_line(line: str, idx: int) -> Task:
    """Parse one ECM=/ECM2= line. Raises LineError with a precise reason."""
    s = line.strip()
    if s.startswith("ECM2="):
        body = s[5:]
    elif s.startswith("ECM="):
        body = s[4:]
    else:
        raise LineError("line does not start with ECM= or ECM2=")

    # The known-factor list is the only quoted field; strip it before CSV splitting, exactly
    # like the C++ parser does.
    factors_raw = ""
    q0 = body.find('"')
    unquoted = body
    if q0 != -1:
        unquoted = body[:q0]
        q1 = body.rfind('"')
        if q1 > q0:
            factors_raw = body[q0 + 1:q1]

    cols = split_csv(unquoted)
    if not cols or (len(cols) == 1 and not cols[0]):
        raise LineError("no fields after ECM=/ECM2=")

    i = 0
    aid = ""
    fft2 = ""
    if not is_int_like(cols[i]) and not cols[i].startswith("FFT2="):
        aid = cols[i]
        i += 1
    if i < len(cols) and cols[i].startswith("FFT2="):
        fft2 = cols[i][5:]
        i += 1

    if len(cols) < i + 5:
        raise LineError("not enough fields (need k,b,n,c,B1)")

    k, b = cols[i], cols[i + 1]
    n_raw, c = cols[i + 2], cols[i + 3]
    b1 = cols[i + 4]

    if not is_int_like(n_raw) or n_raw.startswith("-"):
        raise LineError(f"invalid exponent n: '{n_raw}'")
    n = int(n_raw)
    if n < 0:
        raise LineError(f"invalid exponent n: '{n_raw}'")
    if strtod(b1) is None or strtod(b1) <= 0.0:
        raise LineError(f"invalid B1: '{b1}'")

    p = i + 5
    b2 = "0"
    if len(cols) > p:
        if cols[p]:
            v = strtod(cols[p])
            if v is None or v < 0.0:
                raise LineError(f"invalid B2: '{cols[p]}'")
            b2 = cols[p]
        p += 1

    curves = 100
    if len(cols) > p:
        if cols[p] and is_int_like(cols[p]):
            v = int(cols[p])
            if 0 < v <= 0xFFFFFFFF:
                curves = v
        p += 1

    sigma = ""
    if len(cols) > p:
        if cols[p] and is_int_like(cols[p]) and int(cols[p]) > 0:
            sigma = cols[p]
        p += 1

    factors = [f for f in split_csv(factors_raw) if f] if factors_raw else []

    return Task(idx=idx, aid=aid, fft2=fft2, k=k, b=b, c=c, n=n,
                b1=b1, b2=b2, curves=curves, sigma=sigma, factors=factors)


# --------------------------------------------------------------------------------------
# pipeline stages
# --------------------------------------------------------------------------------------


def read_tasks(paths: Sequence[Path], stats: dict) -> List[Task]:
    tasks: List[Task] = []
    for path in paths:
        try:
            fh = _open_text(path)
        except OSError as exc:
            raise SystemExit(f"error: cannot read {path}: {exc.strerror or exc}")
        with fh:
            for raw in fh:
                line = raw.strip().lstrip("\ufeff")
                if not line or line.startswith("#"):
                    stats["skipped_blank_or_comment"] += 1
                    continue
                if not (line.startswith("ECM2=") or line.startswith("ECM=")):
                    if line.startswith("ECMSTAGE2="):
                        # ECMSTAGE2= is an OUTPUT format: our driver accepts it to run
                        # stage 1, but this generator only takes assignments.
                        stats["skipped_other_prefix"] += 1
                    else:
                        stats["skipped_unknown"] += 1
                    continue
                try:
                    tasks.append(parse_ecm_line(line, len(tasks)))
                except LineError as exc:
                    raise SystemExit(f"{path}: {exc}\n  line: {line}")
    stats["read"] = len(tasks)
    return tasks


def apply_filters(tasks: List[Task], args, stats: dict) -> List[Task]:
    out = []
    for t in tasks:
        if args.min_n is not None and t.n < args.min_n:
            stats["filtered_n"] += 1
            continue
        if args.max_n is not None and t.n > args.max_n:
            stats["filtered_n"] += 1
            continue
        if args.min_curves is not None and t.curves < args.min_curves:
            stats["filtered_curves"] += 1
            continue
        if args.max_curves is not None and t.curves > args.max_curves:
            stats["filtered_curves"] += 1
            continue
        out.append(t)
    return out


def dedup_tasks(tasks: List[Task], stats: dict) -> List[Task]:
    """Identity = (k,b,n,c) numerically; winner = B1 -> curves -> real AID -> first seen."""
    groups: dict = {}
    for t in tasks:
        groups.setdefault(t.number_key(), []).append(t)

    out: List[Task] = []
    for key, rows in groups.items():
        if len(rows) == 1:
            out.append(rows[0])
            continue
        stats["duplicates_removed"] += len(rows) - 1
        winner = sorted(
            rows,
            key=lambda t: (-t.b1_value(), -t.curves, 0 if t.has_real_aid() else 1, t.idx),
        )[0]
        # Union of known factors across the group: known factors only shrink N, and the
        # stage-1 and stage-2 lines must agree on N.
        merged: List[str] = []
        for t in rows:
            for f in t.factors:
                if f not in merged:
                    merged.append(f)
        winner.factors = merged
        out.append(winner)
    return out


def apply_rewrites(tasks: List[Task], args, stats: dict) -> List[Task]:
    out = []
    for t in tasks:
        if args.set_b1 is not None:
            t.b1 = args.set_b1
        if args.set_b2 is not None:
            t.b2 = args.set_b2
        if args.set_has_na and not t.aid:
            t.aid = "N/A"
        out.append(t)
    return out


_SORT_KEYS = {
    "aid": lambda t: (t.aid or ""),
    "k": lambda t: int(t.k),
    "b": lambda t: int(t.b),
    "n": lambda t: t.n,
    "c": lambda t: int(t.c),
    "b1": lambda t: t.b1_value(),
    "b2": lambda t: t.b2_value(),
    "curves": lambda t: t.curves,
}


def sort_tasks(tasks: List[Task], sort_by: str, desc: bool) -> List[Task]:
    keys = [k.strip() for k in sort_by.split(",") if k.strip()] if sort_by else []
    if not keys:
        return tasks
    for k in keys:
        if k not in _SORT_KEYS:
            raise SystemExit(f"unsupported --sort-by field: '{k}' (choose from {', '.join(_SORT_KEYS)})")
    # stable: idx is the final tie-break, so two runs are byte-identical
    return sorted(tasks, key=lambda t: tuple(_SORT_KEYS[k](t) for k in keys) + (t.idx,),
                  reverse=desc)


# --------------------------------------------------------------------------------------
# save-name / factor validation
# --------------------------------------------------------------------------------------


def render_save_name(pattern: str, t: Task) -> str:
    return (pattern.replace("{k}", t.k).replace("{b}", t.b).replace("{c}", t.c)
                   .replace("{n}", str(t.n)).replace("{b1}", t.b1).replace("{b2}", t.b2))


def check_save_name(name: str) -> Optional[str]:
    """Mirror ecm_extract_b1_from_save_name(): must end .save with a _<B1> token."""
    m = _SAVE_RE.match(name)
    if not m:
        return f"save name does not match <...>_<B1>.save: '{name}'"
    v = strtod(m.group("b1"))
    if v is None or v <= 0.0:
        return f"cannot parse B1 from save-name token '{m.group('b1')}' (in '{name}')"
    return None


def verify_factors(tasks: Iterable[Task]) -> List[str]:
    """Exact divisibility check: every known factor must divide k*b^n + c."""
    problems: List[str] = []
    for t in tasks:
        if not t.factors:
            continue
        value = int(t.k) * pow(int(t.b), t.n) + int(t.c)
        for f in t.factors:
            if not is_int_like(f) or int(f) <= 1:
                problems.append(f"n={t.n}: factor '{f}' is not an integer > 1")
                continue
            if value % int(f) != 0:
                problems.append(f"n={t.n}: factor {f} does not divide {t.k}*{t.b}^{t.n}{t.c}")
    return problems


def normalize_factors(tasks: Iterable[Task], do_sort: bool) -> None:
    for t in tasks:
        seen: List[str] = []
        for f in t.factors:
            if f not in seen:
                seen.append(f)
        t.factors = sorted(seen, key=lambda f: (int(f) if is_int_like(f) else 0)) if do_sort else seen


# --------------------------------------------------------------------------------------
# emitting
# --------------------------------------------------------------------------------------


def emit_ecmstage2(t: Task, save: str, skip_curves: int, curves: int) -> str:
    parts: List[str] = []
    if t.aid:
        parts.append(t.aid)
    parts += [t.k, t.b, str(t.n), t.c, csv_quote(save), t.b2, str(skip_curves), str(curves)]
    if t.factors:
        parts.append(csv_quote(",".join(t.factors)))
    return "ECMSTAGE2=" + ",".join(parts)


def emit_ecm(t: Task, prefix: str, curves: int) -> str:
    parts: List[str] = []
    if t.aid:
        parts.append(t.aid)
    if t.fft2:
        parts.append(f"FFT2={t.fft2}")
    parts += [t.k, t.b, str(t.n), t.c, t.b1, t.b2, str(curves)]
    if t.sigma:
        parts.append(t.sigma)
    if t.factors:
        parts.append(csv_quote(",".join(t.factors)))
    return f"{prefix}=" + ",".join(parts)


def emit_cli(t: Task, save: str, args) -> str:
    kind = args.emit_cli_kind
    expr = t.expression()
    if kind == "sh":
        return (f"echo '{expr}' | {args.exe} -v -savea {save} -gpu "
                f"-gpuckpt {args.gpuckpt} -gpucurves {args.gpu_curves} {t.b1} {t.b2}")
    cmd = (f"{args.exe} -v -gpu -d {args.device} -gpuckpt {args.gpuckpt} "
           f"-gpucurves {args.gpu_curves} -savea {save} {t.b1} {t.b2}")
    if kind == "bat":
        # ^ & | < > need escaping inside a .bat file; the N expression has none of these
        # except ^ (in "2^3500017"), which must be doubled.
        return f"echo {expr.replace('^', '^^')}| {cmd}"
    return f"echo '{expr}' | {cmd}"          # ps1: single quotes, no escaping needed


# --------------------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------------------


def _open_text(path: Path):
    """Read UTF-8 (with or without BOM) and fall back to GBK, like the operator's files."""
    data = path.read_bytes()
    if data[:3] == b"\xef\xbb\xbf":
        data = data[3:]
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("gbk", errors="replace")
    return io.StringIO(text)


def write_lines(path: Path, lines: Sequence[str], append: bool, newline: str) -> None:
    if not lines:
        return
    nl = "\r\n" if newline == "crlf" else "\n"
    payload = nl.join(lines) + nl
    path.parent.mkdir(parents=True, exist_ok=True)
    if append and path.exists() and path.stat().st_size > 0:
        existing = path.read_bytes()
        if not existing.endswith((b"\n", b"\r")):
            payload = nl + payload
        with path.open("ab") as fh:
            fh.write(payload.encode("utf-8"))
    else:
        with path.open("wb") as fh:
            fh.write(payload.encode("utf-8"))


def write_section(path: Path, worker: int, lines: Sequence[str], append: bool, newline: str) -> None:
    """Plain list, or one [Worker #N] section (never touches other sections)."""
    if worker is None:
        write_lines(path, lines, append, newline)
        return
    body = [f"[Worker #{worker}]"] + list(lines) + [""]
    write_lines(path, body, append, newline)


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ecm.py",
        description="ECM worktodo pipeline: assignments -> stage-1 queue (ECMSTAGE2=) and "
                    "Prime95 worktodo (ECM=/ECM2=).  Generation only.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="B2=0 means 'let Prime95 choose B2 automatically', not 'skip stage 2'.")

    p.add_argument("--version", action="version", version=f"worktodo pipeline {VERSION}")

    g = p.add_argument_group("input")
    g.add_argument("--input", action="append", default=[], metavar="FILE",
                   help="assignment file (ECM=/ECM2= lines); repeatable")

    g = p.add_argument_group("output (all optional; none = summary only)")
    g.add_argument("--out-ecmstage2", metavar="FILE",
                   help="ECMSTAGE2= lines: our stage-1 queue AND the stage-2 handoff")
    g.add_argument("--out-ecm", metavar="FILE",
                   help="ECM=/ECM2= lines for native Prime95 worktodo files")
    g.add_argument("--emit-cli", metavar="FILE",
                   help="per-task command lines (see --emit-cli-kind)")
    g.add_argument("--emit-cli-kind", choices=["sh", "ps1", "bat"], default="sh")
    g.add_argument("--worker", type=int, metavar="N",
                   help="wrap --out-ecm in a single [Worker #N] section")
    g.add_argument("--ecm-prefix", choices=["ECM2", "ECM"], default="ECM2",
                   help="prefix for --out-ecm (default ECM2; both are equivalent for Prime95)")
    g.add_argument("--newline", choices=["crlf", "lf"], default="crlf",
                   help="output line ending (default crlf)")
    g.add_argument("--append-ecmstage2", action="store_true")
    g.add_argument("--append-ecm", action="store_true")
    g.add_argument("--append-cli", action="store_true")

    g = p.add_argument_group("parameters")
    g.add_argument("--save-pattern", default="m{n}_{b1}.save",
                   help="save-file template; {n}{k}{b}{c}{b1}{b2} (default m{n}_{b1}.save)")
    g.add_argument("--gpu-curves", type=int, default=192, help="curves per task (both stages)")
    g.add_argument("--skip-curves", type=int, default=0, help="ECMSTAGE2 skip_curves field")
    g.add_argument("--set-b1", metavar="VAL", help="rewrite B1 (original spelling kept)")
    g.add_argument("--set-b2", metavar="VAL",
                   help="rewrite B2; 0 = let Prime95 pick B2 automatically")
    g.add_argument("--set-has-na", action="store_true",
                   help="write N/A where the AID is missing")

    g = p.add_argument_group("processing")
    g.add_argument("--no-dedup", action="store_true", help="keep duplicate (k,b,n,c) rows")
    g.add_argument("--sort-by", default="n",
                   help="comma list of n,k,b,c,b1,b2,curves,aid (default n); empty = input order")
    g.add_argument("--desc", action="store_true", help="descending sort")
    g.add_argument("--sort-factors", action="store_true",
                   help="normalize known-factor order (numerically ascending); default keeps "
                        "input order so output stays byte-comparable")
    g.add_argument("--verify-factors", action="store_true",
                   help="exact check that every known factor divides k*b^n+c (off by default)")

    g = p.add_argument_group("filters")
    g.add_argument("--min-n", type=int)
    g.add_argument("--max-n", type=int)
    g.add_argument("--min-curves", type=int)
    g.add_argument("--max-curves", type=int)

    g = p.add_argument_group("command-line emitter")
    g.add_argument("--exe", help="executable for --emit-cli (default: ./ecm for sh, "
                                 "ecm_cuda.exe otherwise)")
    g.add_argument("--device", type=int, default=0, help="-d index for ps1/bat emitters")
    g.add_argument("--gpuckpt", type=int, default=300, help="-gpuckpt seconds (default 300)")

    g = p.add_argument_group("safety")
    g.add_argument("--allow-invalid-save-name", action="store_true",
                   help="do not fail when the rendered save name breaks <...>_<B1>.save")
    g.add_argument("--dry-run", action="store_true", help="print the summary, write nothing")

    # ---- legacy aliases (the Linux-era tool's switch names) -------------------------
    g = p.add_argument_group("legacy aliases")
    g.add_argument("--out-windows", metavar="FILE", help="alias of --out-ecmstage2")
    g.add_argument("--out-prmers", metavar="FILE", help="alias of --out-ecm")
    g.add_argument("--out-linux", metavar="FILE", help="alias of --emit-cli (kind sh)")
    g.add_argument("--append-windows", action="store_true", help="alias of --append-ecmstage2")
    g.add_argument("--append-prmers", action="store_true", help="alias of --append-ecm")
    g.add_argument("--append-linux", action="store_true", help="alias of --append-cli")
    g.add_argument("--gmpecm-b2", metavar="VAL", help="legacy: B2 for the stage-1 command line")
    g.add_argument("--p95-b2", metavar="VAL", help="legacy: B2 for the stage-2 line")
    return p


def resolve_aliases(args) -> None:
    if args.out_windows and not args.out_ecmstage2:
        args.out_ecmstage2 = args.out_windows
    if args.out_prmers and not args.out_ecm:
        args.out_ecm = args.out_prmers
    if args.out_linux and not args.emit_cli:
        args.emit_cli, args.emit_cli_kind = args.out_linux, "sh"
    args.append_ecmstage2 = args.append_ecmstage2 or args.append_windows
    args.append_ecm = args.append_ecm or args.append_prmers
    args.append_cli = args.append_cli or args.append_linux
    if args.set_b2 is None:
        for legacy in (args.gmpecm_b2, args.p95_b2):
            if legacy is not None:
                args.set_b2 = legacy
    if args.exe is None:
        args.exe = "./ecm" if args.emit_cli_kind == "sh" else "ecm_cuda.exe"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    resolve_aliases(args)

    if not args.input:
        print("error: --input is required", file=sys.stderr)
        return 2

    stats = {"read": 0, "skipped_blank_or_comment": 0, "skipped_other_prefix": 0,
             "skipped_unknown": 0, "filtered_n": 0, "filtered_curves": 0,
             "duplicates_removed": 0}

    tasks = read_tasks([Path(p) for p in args.input], stats)
    tasks = apply_filters(tasks, args, stats)
    before_dedup = len(tasks)
    if not args.no_dedup:
        tasks = dedup_tasks(tasks, stats)
    else:
        for t in tasks:
            t.factors = list(dict.fromkeys(t.factors))
    tasks = apply_rewrites(tasks, args, stats)
    normalize_factors(tasks, args.sort_factors)
    tasks = sort_tasks(tasks, args.sort_by, args.desc)

    # ---- validation ------------------------------------------------------------------
    problems: List[str] = []
    if args.verify_factors:
        problems += verify_factors(tasks)
    if problems:
        for msg in problems:
            print(f"error: {msg}", file=sys.stderr)
        return 1

    need_save = bool(args.out_ecmstage2) or (bool(args.emit_cli) and args.emit_cli_kind != "sh")
    save_names: List[str] = []
    for t in tasks:
        name = render_save_name(args.save_pattern, t)
        save_names.append(name)
        if need_save:
            why = check_save_name(name)
            if why and not args.allow_invalid_save_name:
                print(f"error: {why}", file=sys.stderr)
                print("       the driver takes B1 from the save name for ECMSTAGE2= lines; "
                      "see --allow-invalid-save-name", file=sys.stderr)
                return 1

    # ---- emit ------------------------------------------------------------------------
    stage2_lines = [emit_ecmstage2(t, s, args.skip_curves, args.gpu_curves)
                    for t, s in zip(tasks, save_names)]
    ecm_lines = [emit_ecm(t, args.ecm_prefix, args.gpu_curves) for t in tasks]
    cli_lines = [emit_cli(t, s, args) for t, s in zip(tasks, save_names)]

    written: List[str] = []
    if not args.dry_run:
        try:
            if args.out_ecmstage2:
                write_lines(Path(args.out_ecmstage2), stage2_lines, args.append_ecmstage2, args.newline)
                written.append(f"ECMSTAGE2=    {len(stage2_lines):6d} lines -> {args.out_ecmstage2}")
            if args.out_ecm:
                write_section(Path(args.out_ecm), args.worker, ecm_lines, args.append_ecm, args.newline)
                written.append(f"{args.ecm_prefix}=  {len(ecm_lines):6d} lines -> {args.out_ecm}")
            if args.emit_cli:
                write_lines(Path(args.emit_cli), cli_lines, args.append_cli, args.newline)
                written.append(f"CLI ({args.emit_cli_kind})  {len(cli_lines):6d} lines -> {args.emit_cli}")
        except OSError as exc:
            target = getattr(exc, "filename", "?")
            print(f"error: cannot write {target}: {exc.strerror or exc}", file=sys.stderr)
            print("       (a running Prime95 or a read-only/locked target will do this)",
                  file=sys.stderr)
            return 1

    # ---- summary (ASCII on purpose: survives any console codepage) --------------------
    print(f"worktodo pipeline {VERSION}")
    print(f"  input files        : {len(args.input)}")
    print(f"  tasks read         : {stats['read']}")
    print(f"  skipped (comment)  : {stats['skipped_blank_or_comment']}")
    print(f"  skipped (ECMSTAGE2): {stats['skipped_other_prefix']}")
    print(f"  skipped (unknown)  : {stats['skipped_unknown']}")
    print(f"  filtered by n      : {stats['filtered_n']}")
    print(f"  filtered by curves : {stats['filtered_curves']}")
    print(f"  after filters      : {before_dedup}")
    print(f"  duplicates removed : {stats['duplicates_removed']}")
    print(f"  tasks to emit      : {len(tasks)}")
    if args.dry_run:
        print("  dry run: nothing written")
    for line in written:
        print(f"  wrote {line}")
    if not written and not args.dry_run:
        print("  no output requested (use --out-ecmstage2 / --out-ecm / --emit-cli)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
