"""Compare the standalone GMP experiment with independent PARI/GP arithmetic.

Requires the probe built by tools/build/test/build_ecm_friendly_probe.ps1.
All generated evidence goes to --output; no production saves are synthesized.
"""
import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/ecm_prob"))
from friendly_curves import family_iv, atkin_morain


def run(args):
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    primes = [1009, 65537, 1000003, 2147483647, 2305843009213693951]
    seeds = [1, 2, 3, 4, 7, 8, 31, 100, 2**53-1, 2**53+1, 2**64-1]
    cases = [(f, m, p) for f in (1, 2) for m in seeds for p in primes]
    body = (ROOT / "tools/ecm_prob/friendly_curves.gp").read_text(encoding="utf-8")
    body += '\nprint("VERSION|",version());\n'
    for f, m, p in cases:
        body += (f'c=ecmf_model({f},{m},{p});if(#c, '
                 f'print("C|{f}|{m}|{p}|",lift(c[3][2]),"|",lift(c[3][3]),"|",lift(c[3][4]),"|",'
                 'lift(c[3][1]/(c[3][1]-c[3][2]))),'
                 f'print("B|{f}|{m}|{p}"));\n')
    # Small fields exercise torsion/endpoints; large fields avoid universal hits.
    evals = [(f, m, p, b1, mult) for f in (1, 2) for m in (1, 2, 3, 7)
             for p in (1009, 1000003, 2147483647) for b1 in (256, 1024) for mult in (1, 48)]
    for f, m, p, b1, mult in evals:
        body += (f'c=ecmf_model({f},{m},{p});if(#c,'
                 f'S={mult};forprime(q=2,{b1},S*=q^logint({b1},q));R=ellmul(c[1],c[2],S);'
                 f'print("E|{f}|{m}|{p}|{b1}|{mult}|",if(R==[0],"infinity",lift(R[1]/(4/(c[3][1]-c[3][2]))))),'
                 f'print("EB|{f}|{m}|{p}|{b1}|{mult}"));\n')
    for f in (1, 2, 4, 5):
        for m in (1, 2, 3) if f <= 2 else (1,):
            body += f'c=ecmf_model({f},{m});T=elltors(c[1]);print("T|{f}|{m}|",T[1],"|",T[2]);\n'
    body += 'print("COMPLETE");quit();\n'
    (out / "oracle.gp").write_text(body, encoding="utf-8")
    gp = subprocess.run([str(args.gp), "-q", "-f", "-s", "256M"], input=body,
                        text=True, capture_output=True, check=True, timeout=300)
    (out / "oracle.log").write_text(gp.stdout, encoding="utf-8")
    (out / "oracle.errors.log").write_text(gp.stderr, encoding="utf-8")
    assert "***" not in gp.stderr and "COMPLETE" in gp.stdout, gp.stderr
    construction, endpoints, torsion = {}, {}, []
    for line in gp.stdout.splitlines():
        v = line.split("|")
        if v[0] in ("C", "B"):
            construction[tuple(map(int, v[1:4]))] = list(map(int, v[4:])) if v[0] == "C" else None
        elif v[0] == "E":
            endpoints[tuple(map(int, v[1:6]))] = v[6]
        elif v[0] == "T":
            torsion.append(v[1:])
            expected = {1: (8, "[4, 2]"), 2: (16, "[8, 2]"), 4: (6, "[6]"), 5: (16, "[8, 2]")}
            assert (int(v[3]), v[4]) == expected[int(v[1])], line
    native_body = "".join(f"{f} {m} {p}\n" for f, m, p in cases)
    assert len(endpoints) == len(evals), "all selected endpoint oracle constructions must be usable"
    native = subprocess.run([str(args.probe), "--construct"], input=native_body,
                            text=True, capture_output=True, check=True)
    (out / "construct.log").write_text(native.stdout, encoding="utf-8")
    checked = 0
    for line in native.stdout.splitlines():
        v = line.split("|")
        key = tuple(map(int, v[:3]))
        expected = construction[key]
        if expected is None:
            assert v[3] == "degenerate", line
        else:
            assert v[3] == "ready" and list(map(int, v[4:8])) == expected, (line, expected)
            checked += 1
    assert len(native.stdout.splitlines()) == len(cases)
    # Independent exact rational check, separate from GP and native formula evaluation.
    rational_checks = 0
    for f in (1, 2):
        for m in (1, 2, 3, 4, 7, 8):
            c = (family_iv if f == 1 else atkin_morain)(m)
            for p in primes:
                def residue(q):
                    return q.numerator * pow(q.denominator, -1, p) % p
                expected = construction[(f, m, p)]
                if expected is not None:
                    assert expected == [residue(q) for q in (c.d, c.x, c.y, c.a/(c.a-c.d))]
                    rational_checks += 1
    # One request per endpoint: incomplete Edwards exceptions are retained as failures.
    failures, endpoint_checks, replays = [], 0, 0
    for key, expected in endpoints.items():
        native = subprocess.run([str(args.probe), "--evaluate"], input=" ".join(map(str, key))+"\n",
                                text=True, capture_output=True)
        if native.returncode:
            failures.append(dict(case=key, error=native.stderr.strip()))
            continue
        v = native.stdout.strip().split("|")
        assert v[3] == "ready", native.stdout
        x, z = map(int, v[9:11])
        p = key[2]
        assert (z == 0 and x != 0) if expected == "infinity" else (z != 0 and x == int(expected)*z % p), (key, expected, x, z)
        endpoint_checks += 1
        replays += int(v[11])
    composite_cases = [(f, m, p*q) for f in (1, 2) for m in range(1, 17)
                       for p, q in ((5, 1009), (7, 1009), (11, 1009), (13, 1009), (1009, 65537))]
    native = subprocess.run([str(args.probe), "--construct"],
                            input="".join(" ".join(map(str, c))+"\n" for c in composite_cases),
                            text=True, capture_output=True, check=True)
    (out / "composites.log").write_text(native.stdout, encoding="utf-8")
    assert len(native.stdout.splitlines()) == len(composite_cases)
    factors = 0
    for line in native.stdout.splitlines():
        v = line.split("|")
        if v[3] == "factor":
            n, factor = int(v[2]), int(v[8])
            assert 1 < factor < n and n % factor == 0
            factors += 1
    result = dict(construct_cases=len(cases), ready_construct_checks=checked,
                  rational_checks=rational_checks, endpoint_checks=endpoint_checks,
                  exceptional_endpoints=failures, endpoint_replays=replays, composite_cases=len(composite_cases),
                  proper_setup_factors=factors, torsion=torsion)
    (out / "validation.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit("Endpoint exceptions require handling before claiming complete Stage1 support")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gp", type=Path, required=True)
    p.add_argument("--probe", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    run(p.parse_args())
