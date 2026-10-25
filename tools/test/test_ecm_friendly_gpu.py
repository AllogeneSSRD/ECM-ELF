"""Actual CUDA Stage2 factor recovery for GKL(iv) and Atkin-Morain.

Uses an explicit a24 and a normalized experimental endpoint, never a forged
PARAM0/Atkin-Morain save identity. Defaults to the user's GPU1 test device.
"""
import argparse
import json
import math
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main(args):
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    p, cofactor = 537033121, 2**127-1
    n = p*cofactor; scalar = 48*math.lcm(*range(1, 257))
    body = (ROOT/"tools/ecm_prob/friendly_curves.gp").read_text(encoding="utf-8")
    body += f'\nif(!isprime({p}) || !isprime({cofactor}),error("unproven test factors"));\n'
    body += f'for(f=1,2,ecmf_order(f,1,{p}));\nprint("COMPLETE");quit();\n'
    (out/"oracle.gp").write_text(body, encoding="utf-8")
    oracle = subprocess.run([str(args.gp), "-q", "-f"], input=body, text=True, capture_output=True, check=True)
    (out/"oracle.log").write_text(oracle.stdout, encoding="utf-8")
    (out/"oracle.errors.log").write_text(oracle.stderr, encoding="utf-8")
    assert "***" not in oracle.stderr and "COMPLETE" in oracle.stdout
    orders = {}
    for line in oracle.stdout.splitlines():
        v = line.split("|")
        if len(v) != 8: continue
        f, r = int(v[0]), int(v[4]); remainder = r//math.gcd(r, scalar)
        assert remainder == {1: 593, 2: 281}[f]
        orders[f] = dict(point_order=r, remaining_prime=remainder)
    assert set(orders) == {1, 2}
    request = "".join(f"{f} 1 {n} 256 48 25600 210 {args.device}\n" for f in (1, 2))
    (out/"input.txt").write_text(request, encoding="utf-8")
    native = subprocess.run([str(args.probe), "--stage2"], input=request, text=True, capture_output=True, timeout=300)
    (out/"stdout.log").write_text(native.stdout, encoding="utf-8")
    (out/"stderr.log").write_text(native.stderr, encoding="utf-8")
    assert native.returncode == 0, native.stderr
    rows = [json.loads(line.split("|", 1)[1]) for line in native.stdout.splitlines() if line.startswith("FRIENDLY_GPU|")]
    assert len(rows) == 2
    for row in rows:
        assert row["factors"] == [str(p)], row
        assert 1 < p < n and n % p == 0
        assert row["bad"] == row["bad_factors"] == 0 and row["checked"] > 0 and row["selftest_cases"] > 0
    result = dict(prime=p, cofactor=str(cofactor), N=str(n), B1=256, B2=25600, device=args.device,
                  oracle_orders=orders, actual_gpu_results=rows, passed=True)
    (out/"validation.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(f"PASS: both curve families recovered {p}; independent remaining orders 593 and 281")


if __name__ == "__main__":
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--probe", type=Path, required=True)
    a.add_argument("--gp", type=Path, required=True)
    a.add_argument("--output", type=Path, required=True)
    a.add_argument("--device", type=int, default=1)
    main(a.parse_args())
