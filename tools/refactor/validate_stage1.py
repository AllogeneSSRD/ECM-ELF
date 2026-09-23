#!/usr/bin/env python3
"""Stage1 validation — tests auto-mode ECM on all available GPUs."""
import subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ECM = str(ROOT / "build" / "Debug" / "ecm.exe")
ENV = {"ECM_KERNEL_ROOT": str(ROOT / "kernels" / "opencl")}

CASES = [
    ("(2^151-1)", "1e4", "0", "3:2026",     "1",  "151-bit Mersenne"),
    ("(2^347-1)", "1e4", "0", "3:561219477","8",  "347-bit Mersenne"),
    ("(2^421-1)", "1e4", "0", "3:20260611", "8",  "421-bit Mersenne"),
    ("(2^641-1)", "1e4", "0", "3:20260611", "8",  "641-bit Mersenne"),
]

def test(dev, n_expr, b1, b2, sigma, curves, desc):
    tag = f"[dev={dev}] {desc}"
    t0 = time.time()
    try:
        r = subprocess.run(
            [ECM, "-v", "-gpu", "-d", str(dev), "-sigma", sigma,
             "-gpucurves", curves, b1, b2],
            input=n_expr + "\n", capture_output=True, text=True, timeout=120, env=ENV,
        )
        dt = time.time() - t0
        combined = r.stdout + r.stderr
        # Check for critical errors
        for line in combined.splitlines():
            ll = line.lower()
            if "opencl build error" in ll:
                print(f"  FAIL {tag}  ({dt:.1f}s)\n    {line.strip()}")
                return False
            if ("error:" in ll and "factor" not in ll
                    and "build error" not in ll):
                print(f"  FAIL {tag}  ({dt:.1f}s)\n    {line.strip()}")
                return False
            if "failed to build" in ll:
                print(f"  FAIL {tag}  ({dt:.1f}s)\n    {line.strip()}")
                return False
            if "opencl_ecm_stage1 returned:" in line:
                parts = line.split()
                idx = parts.index("returned:") + 1
                ret = parts[idx]
                gt = next((w for w in parts if "gputime=" in w), "")
                print(f"  OK   {tag}  ret={ret} {gt} ({dt:.1f}s)")
                return True
        print(f"  OK   {tag}  no-return-line ({dt:.1f}s)")
        return True
    except subprocess.TimeoutExpired:
        print(f"  FAIL {tag}  timeout")
        return False
    except Exception as e:
        print(f"  FAIL {tag}  {e}")
        return False


def main():
    # Detect GPU count
    try:
        r = subprocess.run(
            [ECM, "-gpu", "-d", "0", "-gpucurves", "1", "-sigma", "3:1", "1", "0"],
            input="(2^127-1)\n", capture_output=True, text=True, timeout=30, env=ENV,
        )
    except Exception:
        print("Cannot detect GPUs.")
        return 1
    combined = r.stdout + r.stderr
    ndev = 1
    for ln in combined.splitlines():
        if ln.strip().startswith("[") and "GPU" in ln:
            try:
                n = int(ln.strip()[1])
                ndev = max(ndev, n + 1)
            except ValueError:
                pass
    print(f"Running validation on {ndev} GPU device(s)\n")

    ok = fail = 0
    for dev in range(ndev):
        for n_expr, b1, b2, sigma, curves, desc in CASES:
            if test(dev, n_expr, b1, b2, sigma, curves, desc):
                ok += 1
            else:
                fail += 1

    print(f"\n{'='*50}\n{ok} passed, {fail} failed, {ok+fail} total")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
