"""Generate and cache exhaustive prime sets per bit-width (local, reproducible).

Uses primesieve.exe (`.refactor/primesieve-12.15-win-x64/primesieve.exe`) to
enumerate all primes in [2^(b-1), 2^b - 1] for each bit-width b, stores them as
little-endian uint64 binary (`data/primes/bits{b}.bin`), and records provenance
in `data/primes/manifest.json` (range, count, sha256, generator command,
timestamp).

Idempotent: if the cached file's sha256 matches the manifest, regeneration is
skipped.

Usage:
    python gen_primes.py            # bits 15..20 (default)
    python gen_primes.py 20 25      # explicit bit range
"""

from __future__ import annotations

import array
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

TOOL_DIR = Path(__file__).resolve().parent
PRIME_DIR = TOOL_DIR / "data" / "primes"
PRIMESIEVE = Path(r"D:\code\MPA-OpenCl\.refactor\primesieve-12.15-win-x64\primesieve.exe")
MANIFEST = PRIME_DIR / "manifest.json"


def sieve_range(lo: int, hi: int) -> list[int]:
    """All primes in [lo, hi] via primesieve.exe."""
    cmd = [str(PRIMESIEVE), str(lo), str(hi), "-p", "--no-status"]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True)
    primes = [int(line) for line in out.stdout.split() if line.strip()]
    return primes


def sha256_of_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def write_primes(bit: int) -> dict:
    lo = 1 << (bit - 1)
    hi = (1 << bit) - 1
    primes = sieve_range(lo, hi)
    data = array.array("Q", primes).tobytes()  # uint64 little-endian
    fpath = PRIME_DIR / f"bits{bit}.bin"
    fpath.write_bytes(data)
    entry = {
        "bit": bit,
        "lo": lo,
        "hi": hi,
        "count": len(primes),
        "sha256": sha256_of_bytes(data),
        "generator": "primesieve 12.15",
        "cmd": f"primesieve {lo} {hi} -p",
        "format": "uint64 little-endian, one prime per 8 bytes",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": "exhaustive (all primes in [2^(b-1), 2^b-1])",
    }
    return entry


def load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {}


def save_manifest(manifest: dict) -> None:
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def gen(bits: list[int]) -> None:
    PRIME_DIR.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    for b in bits:
        key = str(b)
        fpath = PRIME_DIR / f"bits{b}.bin"
        prev = manifest.get(key)
        if fpath.exists() and prev and prev.get("sha256") == sha256_of_bytes(fpath.read_bytes()):
            print(f"bits{b}: cached ({prev['count']} primes, sha256 ok)")
            continue
        entry = write_primes(b)
        manifest[key] = entry
        save_manifest(manifest)
        print(f"bits{b}: generated {entry['count']} primes -> {fpath.name} "
              f"(sha256 {entry['sha256'][:12]}...)")
    # ensure the manifest is saved even if nothing changed
    save_manifest(manifest)


def main():
    bits = [int(a) for a in sys.argv[1:]] or list(range(15, 21))
    gen(bits)


if __name__ == "__main__":
    main()
