"""Compare complete Stage2 curves with the baby wheel and short NTT overrides.

Default coverage regenerates its Montgomery Stage1 points with the independent
Python oracle. The retained Edwards save suites are optional.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "tools" / name) for name in ("stat", "ecm_report")]
from result_format import normalize_result
import suyama_mont_ref as mont


@dataclass(frozen=True)
class Case:
    name: str
    save_name: str | None
    d: int
    p: int
    b2: int
    factors: tuple[int, ...] = ()
    carrier: int = 0
    weighted: bool = False
    wide: bool = False
    mont_input: tuple[int, int, int] | None = None

    def input_bytes(self, save_dir: Path | None) -> bytes:
        if self.save_name:
            assert save_dir is not None
            return (save_dir / self.save_name).read_bytes()
        assert self.mont_input is not None
        n, sigma, b1 = self.mont_input
        stage1 = mont.stage1(sigma, b1, n)
        assert stage1["gcd"] == 1
        checksum = b1 * sigma * n * stage1["x"] % 4294967291
        return (f"METHOD=ECM; PARAM=0; SIGMA={sigma}; B1={b1}; N={n}; "
                f"X={stage1['x']:#x}; CHECKSUM={checksum};\n").encode("ascii")


@dataclass(frozen=True)
class Arm:
    name: str
    wheel: bool | None = False
    conv: int = 0
    wheel_state: str = ""


BASE = Arm("base")
AUTO_SMALL = Arm("auto_small", wheel=None)
WHEEL = Arm("wheel", True, wheel_state="applied")
WARP = Arm("warp", conv=1)
FUSED = Arm("fused", conv=2)
COMBINED = Arm("combined", True, 2, "applied")
COMMON = Arm("wheel_nonunit", True, wheel_state="common_nonunit")
REPLAY = Arm("wheel_segment", True, wheel_state="segment")
DENIED = Arm("wheel_alloc_fail", True, wheel_state="denied")

CASES = {
    "factor10007": (
        "small", Case("factor10007", None, 210, 24, 13_230, (10007,),
                      mont_input=(10007 * 65537, 26, 20)),
        (BASE, COMBINED)),
    "nonunit103": (
        "small", Case("nonunit103", None, 210, 24, 13_230, (103,),
                      mont_input=(103 * 65537, 26, 2)),
        (BASE, COMMON)),
    "segment137": (
        "small", Case("segment137", None, 210, 24, 13_230, (137,),
                      mont_input=(137 * ((1 << 127) - 1), 28, 2)),
        (BASE, REPLAY)),
    "nf127": (
        "small", Case("nf127", None, 210, 24, 13_230, mont_input=((1 << 127) - 1, 26, 20)),
        (BASE, AUTO_SMALL, WHEEL, DENIED)),
    "real_ordinary": ("real", Case("real_ordinary", "m4001_n3600_s26.save", 180180, 17280, 2_701_597_990),
                      (BASE, WARP, COMBINED)),
    "real_carrier_weighted": (
        "real", Case("real_carrier_weighted", "m4001_n3600_s26.save", 180180, 17280, 2_701_597_990,
                     carrier=4001, weighted=True),
        (BASE, WARP, COMBINED)),
    "real_factor10007": (
        "real", Case("real_factor10007", None, 60060, 5760, 2_600_000_000, (10007,),
                     mont_input=(10007 * ((1 << 4423) - 1), 26, 20)),
        (BASE, FUSED)),
    "wide_carrier": ("wide", Case("wide_carrier", "m16319_n16272_s26.save", 180180, 17280, 2_701_597_990,
                                  carrier=16319, wide=True),
                     (BASE, WHEEL)),
}


def json_file(path: Path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def log_fields(trace: str, label: str) -> dict[str, str]:
    pattern = rf"^(?:\[[^\]\n]+\] )*{re.escape(label)}: ([^\n]+)$"
    rows = set(re.findall(pattern, trace, flags=re.M))
    assert len(rows) == 1, (label, len(rows))
    return dict(re.findall(r"([\w-]+)=([^\s]+)", rows.pop()))


def saved_fields(input_data: bytes) -> dict[str, str]:
    return dict(token.strip().split("=", 1)
                for token in input_data.decode("utf-8-sig").split(";") if token.strip())


def validate_input(case: Case, fields: dict[str, str], raw: dict):
    result = normalize_result(raw)
    n = int(fields["N"], 0)
    expected = {
        "N_hex": f"{n:x}",
        "B1": int(fields["B1"]),
        "B2": case.b2,
        "d": case.d,
        "poly-size": case.p,
        "sigma": int(fields["SIGMA"]),
        "status": "stage2_completed",
        "carrier_exponent": case.carrier,
        "requested_D": case.d,
        "requested_B2": case.b2,
        "bad_factors": 0,
    }
    assert all(result[k] == value for k, value in expected.items()), (
        case.name, {k: (result[k], v) for k, v in expected.items() if result[k] != v})
    assert raw["status"] == ("F" if case.factors else "NF")
    factors = tuple(sorted(int(f) for f in result["factors"]))
    assert factors == case.factors, (case.name, factors, case.factors)
    assert all(1 < f < n and n % f == 0 for f in factors)
    assert all(int(e) == 0 for e in raw["errors"].values()), raw["errors"]
    return dict(expected, factors=factors)


def validate_small_dump(case: Case, fields: dict[str, str], dump: Path):
    n = int(fields["N"], 0)
    q = int(fields["X"], 0)
    sigma = int(fields["SIGMA"])
    a24 = mont.suyama_curve(sigma, n)[1]
    lines = dump.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "stage2_tree_F_dump v1" and lines[-1] == "end"
    babies = [tuple(line.split()[1:]) for line in lines if line.startswith("baby ")]
    coefficients = [int(line[2:], 16) for line in lines if line.startswith("F ")]
    indices = [j for j in range(1, case.d // 2 + 1) if math.gcd(j, case.d) == 1]
    assert [int(j) for j, _ in babies] == indices and len(babies) == case.p
    coeffs = [1]
    nonunit = 0
    for j, actual in babies:
        x, z = mont.ladder(int(j), q, 1, a24, n)
        y = int(actual, 16)
        g = math.gcd(z, n)
        if g == 1:
            assert y == x * pow(z, -1, n) % n, (case.name, j)
        elif g == n:
            assert y == 0, (case.name, j)
        else:
            nonunit += 1
        # Rebuild exact F from these canonical x or nonunit fallback leaves.
        updated = [0] * (len(coeffs) + 1)
        for i, c in enumerate(coeffs):
            updated[i] = (updated[i] - y * c) % n
            updated[i + 1] = c
        coeffs = updated
    assert coefficients == coeffs, case.name
    if case.name == "segment137":
        assert nonunit > 0, case.name


def run(args, case: Case, case_input: bytes, arm: Arm, directory: Path) -> dict:
    directory.mkdir(parents=True)
    source = directory / "input.save"
    source.write_bytes(case_input)
    config = directory / "ecm.toml"
    config.write_text("config_version=4\n[stage2]\ndevice=\"gpu\"\nauto_b2 = false\n", encoding="utf-8")
    engine_log = directory / "engine.log"
    debug_log = directory / "debug.log"
    results = directory / "result.jsonl"
    # The CLI condition descriptor accepts numeric NTT_ values. Each working
    # directory is isolated, so use "1" as the dump's relative path.
    f_dump = directory / "1"
    command = [str(args.exe), "--config", str(config), "--save", str(source),
               "--b2", str(case.b2), "--d", str(case.d), "--device", str(args.device),
               "--carrier-exponent", str(case.carrier), "--auto-carrier", "0",
               "--giant-chain", "64", "--giant-tail-balance", "0", "--b2-round-chain", "0",
               "--baby-backend", "coop32" if case.wide else "scalar",
               "--giant-backend", "coop32" if case.wide else "scalar", "--groot-layout", "0",
               "--tune-runtime", "0", "--short-calibration", "0", "--short-ntt-batch-tune", "0",
               "--arena-mb", "6000", "--owner-budget-mb", "640", "--batch-mb", "64",
               "--weighted-ntt", str(int(case.weighted)), "--pack-zero", "1",
               "--relocation", "0", "--factor-only", "--curves", "1", "--log-level", "phases",
               "--log", str(engine_log), "--debug-log-file", str(debug_log),
               "--results", str(results), "--submission", str(directory / "submission.txt")]
    controls = {
        "NTT_BABY_DEVICE": "1",
        "NTT_BABY_DEVICE_CHECK": "1",
        "NTT_SMALL_PRIME_CHECK": "1",
        "NTT_TARGET_LEAF_HASH": "1",
        "NTT_NO_PROGRESS": "1",
        "NTT_SHORT_GRAPH": str(int(args.graph)),
    }
    if arm.wheel is not None:
        controls["NTT_BABY_WHEEL"] = str(int(arm.wheel))
        controls["NTT_STAGE2_SHORT_CONV"] = str(arm.conv)
    if case.p < 4096:
        controls["NTT_REAL_F_DUMP"] = f_dump.name
        if arm.wheel:
            controls["NTT_BABY_WHEEL_STEP"] = "30"
    if arm.wheel_state == "denied":
        controls["NTT_BABY_WHEEL_ALLOC_FAIL"] = "1"
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("NTT_", "ECM_", "CUDA_VISIBLE_DEVICES", "CUDA_LAUNCH_BLOCKING"))}
    json_file(directory / "command.json", {"argv": command, "overrides": controls})
    with (directory / "console.log").open("wb") as log:
        process = subprocess.run(command, cwd=directory, env=env | controls,
                                 stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
    assert process.returncode == 0, (directory, process.returncode)
    [raw] = [json.loads(line) for line in results.read_text(encoding="utf-8-sig").splitlines() if line]
    saved = saved_fields(case_input)
    identity = validate_input(case, saved, raw)
    trace = "\n".join(path.read_text(encoding="utf-8-sig", errors="replace")
                      for path in (engine_log, debug_log, directory / "console.log"))
    leaves = log_fields(trace, "target_descent_values")
    modulus_bits = int(saved["N"], 0).bit_length()
    words = (modulus_bits + 63) // 64
    arithmetic_words = ((case.carrier or modulus_bits) + 63) // 64
    assert (int(leaves["target_bits"]), int(leaves["leaves"]), int(leaves["words"])) == (
        modulus_bits, case.p, case.p * words), directory
    device = log_fields(trace, "baby_device")
    assert device["enabled"] == "1" and int(device["checked_words"]) == case.p * arithmetic_words, directory
    checks = log_fields(trace, "s4_multiply_stats")
    assert checks["gmp_selftest_bad"] == checks["gmp_check_bad"] == "0", directory
    assert int(checks["gmp_selftest_cases"]) and int(checks["gmp_checked"]), directory
    cache = log_fields(trace, "small_prime_reuse") if case.p < 4096 else {}
    if cache:
        assert cache["bad"] == "0", directory
    wheel = log_fields(trace, "baby_wheel") if arm.wheel else {}
    if not arm.wheel:
        assert "baby_wheel: requested=1" not in trace, directory
    if wheel:
        assert wheel["requested"] == "1", directory
        if arm.wheel_state in ("applied", "segment"):
            assert wheel["enabled"] == "1" and int(wheel["segments"]) > 0, directory
        else:
            reason = "allocation" if arm.wheel_state == "denied" else arm.wheel_state
            assert (wheel["enabled"], wheel["reason"]) == ("0", reason), directory
        if arm.wheel_state == "segment":
            assert (int(wheel["replayed_segments"]), int(wheel["replayed_points"])) == (1, 4), directory
            assert int(device["bad_groups"]) > 0, directory
        if arm.wheel_state == "common_nonunit":
            assert int(wheel["replayed_points"]) == case.p, directory
    conv = log_fields(trace, "ntt_short_conv_stats")
    assert int(conv["requested"]) == arm.conv, directory
    if arm.conv and case.p >= 4096:
        calls = "warp_calls" if arm.conv == 1 else "fused_calls"
        assert int(conv[calls]) > 0, (directory, conv)
        if case.weighted:
            assert int(conv["weighted_calls"]) > 0, (directory, conv)
        if args.graph:
            assert int(log_fields(trace, "ntt_graph_stats")["launches"]) > 0, directory
    if case.p < 4096:
        validate_small_dump(case, saved, f_dump)
    return {"identity": identity, "target_leaves": leaves, "cache": cache, "baby_device": device,
            "baby_wheel": wheel, "ntt_short_conv": conv, "s4_checks": checks,
            "program": raw["program"], "directory": str(directory)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--suite", action="append", choices=("small", "real", "wide"))
    parser.add_argument("--case", action="append", choices=tuple(CASES))
    parser.add_argument("--save-dir", type=Path, help="directory with the optional retained Edwards saves")
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--graph", action="store_true")
    args = parser.parse_args()
    suites = args.suite or (("small", "real", "wide") if args.case else ("small",))
    selected = [(case, arms) for suite, case, arms in CASES.values()
                if suite in suites and (not args.case or case.name in args.case)]
    if not selected:
        parser.error("no matching cases")
    if any(case.save_name for case, _ in selected) and args.save_dir is None:
        parser.error("selected retained Edwards cases require --save-dir")
    args.exe = args.exe.resolve(strict=True)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    summary = {"device": args.device, "graph": args.graph, "complete": False, "runs": []}
    try:
        for case, arms in selected:
            reference = None
            baseline_dump = out / case.name / "base" / "1"
            case_input = case.input_bytes(args.save_dir)
            for arm in arms:
                directory = out / case.name / arm.name
                result = run(args, case, case_input, arm, directory)
                comparison = (result["identity"], result["target_leaves"],
                              {k: result["cache"][k] for k in ("matched", "reused", "bad")
                               if k in result["cache"]})
                if reference is None:
                    reference = comparison
                else:
                    assert comparison == reference, (case.name, arm.name, comparison, reference)
                    if baseline_dump.exists():
                        assert baseline_dump.read_bytes() == (directory / "1").read_bytes(), directory
                summary["runs"].append({"case": case.name, "arm": arm.name, **result})
                json_file(out / "summary.json", summary)
                print(f"PASS {case.name}/{arm.name} P={case.p} "
                      f"target_digest={result['target_leaves']['hash']} "
                      f"wheel={result['baby_wheel'].get('enabled', '0')} "
                      f"warp_calls={result['ntt_short_conv']['warp_calls']} "
                      f"fused_calls={result['ntt_short_conv']['fused_calls']}", flush=True)
        summary["complete"] = True
    finally:
        json_file(out / "summary.json", summary)


if __name__ == "__main__":
    main()
