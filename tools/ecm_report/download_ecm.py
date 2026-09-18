#!/usr/bin/env python3
"""
download_ecm.py  –  Periodically download PrimeNet ECM Progress HTML reports.

Saves to tools/ecm_report/html/YYYY-MM-DD.html using the page's Current time date.
Same-day files are overwritten.

Usage:
    python download_ecm.py [--lo 1] [--hi 100000] [--interval 2] [--once]
                           [--out-dir PATH]

Default URL pattern:
    https://www.mersenne.org/report_ecm/?txt=1&ecmnof_lo=LO&ecmnof_hi=HI&ecm_lo=LO&ecm_hi=HI
"""

from __future__ import annotations

import argparse
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_OUT_DIR = Path(__file__).parent / "html"
BASE_URL = "https://www.mersenne.org/report_ecm/"
CURRENT_TIME_RE = re.compile(
    r"Current time:\s*(\d{4}-\d{2}-\d{2})\s+\d{2}:\d{2}\s+UTC"
)
USER_AGENT = (
    "MPA-OpenCl-ecm_report/1.0 (+local mirror of PrimeNet ECM progress)"
)


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def build_url(lo: int, hi: int) -> str:
    return (
        f"{BASE_URL}?txt=1"
        f"&ecmnof_lo={lo}&ecmnof_hi={hi}"
        f"&ecm_lo={lo}&ecm_hi={hi}"
    )


def fetch_html(url: str, timeout: float) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
        charset = resp.headers.get_content_charset() or "utf-8"
    return raw.decode(charset, errors="replace")


def extract_date(html: str) -> str:
    m = CURRENT_TIME_RE.search(html)
    if not m:
        raise ValueError("Could not find 'Current time: YYYY-MM-DD … UTC' in HTML")
    return m.group(1)  # YYYY-MM-DD


def download_once(lo: int, hi: int, out_dir: Path, timeout: float) -> Path:
    url = build_url(lo, hi)
    print(f"[{now_utc()}] GET {url}")
    html = fetch_html(url, timeout=timeout)
    date = extract_date(html)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{date}.html"
    path.write_text(html, encoding="utf-8", newline="\n")
    print(f"[{now_utc()}] Saved {path}  ({len(html)} chars, date={date})")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download PrimeNet ECM Progress HTML on a schedule."
    )
    parser.add_argument("--lo", type=int, default=1, help="Exponent range low (default 1)")
    parser.add_argument(
        "--hi", type=int, default=100000, help="Exponent range high (default 100000)"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=2.0,
        metavar="HOURS",
        help="Hours between downloads (default 2)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Download once and exit (no loop)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP timeout in seconds (default 120)",
    )
    args = parser.parse_args()

    if args.lo > args.hi:
        sys.exit("ERROR: --lo must be <= --hi")
    if args.interval <= 0:
        sys.exit("ERROR: --interval must be > 0")

    if args.once:
        try:
            download_once(args.lo, args.hi, args.out_dir, args.timeout)
        except (urllib.error.URLError, ValueError, OSError) as e:
            sys.exit(f"ERROR: {e}")
        return

    interval_sec = args.interval * 3600.0
    print(
        f"[{now_utc()}] Looping every {args.interval} h  "
        f"range=[{args.lo}, {args.hi}]  out={args.out_dir}"
    )
    while True:
        try:
            download_once(args.lo, args.hi, args.out_dir, args.timeout)
        except (urllib.error.URLError, ValueError, OSError) as e:
            print(f"[{now_utc()}] ERROR: {e}", file=sys.stderr)
        print(f"[{now_utc()}] Sleeping {args.interval} h …")
        try:
            time.sleep(interval_sec)
        except KeyboardInterrupt:
            print(f"\n[{now_utc()}] Stopped.")
            break


if __name__ == "__main__":
    main()
