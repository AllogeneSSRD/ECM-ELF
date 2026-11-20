"""Exercise budget changes, native dispatch reuse and exhaustive search on one CUDA device."""
import argparse
import re
import subprocess
import tomllib
from pathlib import Path

from test_stage1_ntt_production import saved_points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    args = parser.parse_args()
    exe, out = args.exe.resolve(strict=True), args.output.resolve()
    out.mkdir(parents=True)
    database = out / "tune.toml"

    def run(name, limit, flags, expression=None, extra=""):
        work = out / name
        work.mkdir()
        cfg = work / "ecm.toml"
        cfg.write_text(f'config_version = 4\n[shared]\ntune_file = "{database.as_posix()}"\nlog_file = ""\n'
                       f'[stage1]\ntarget = "gpu"\nparam = "e"\ngpu_algorithm = "ntt"\nstage1_gpu_device = {args.device}\n'
                       f'edwards_naf_window_bits = 6\nnaf_memory_mb = {limit}\nntt_memory_mib = 256\n'
                       'handoff_mode = "off"\nexponent_cache_dir = "off"\n' + extra, encoding="utf-8")
        result = subprocess.run([str(exe), "--config", str(cfg), *flags], input=expression, cwd=work,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8",
                                errors="replace", timeout=300)
        (work / "run.log").write_text(result.stdout, encoding="utf-8")
        assert result.returncode == 0, (work, result.stdout[-2000:])
        return result.stdout

    def tune(name, p, limit, exhaustive=False):
        flags = ["--tune", "ntt,batch=1-128", "--bits", str(p), "--naf-w", "6", "32"]
        if exhaustive:
            flags.insert(2, "--tune-exhaustive")
        log = run(name, limit, flags)
        measured, reused = map(int, re.search(r"measured=(\d+) reused=(\d+)", log).groups())
        print(name, measured, reused, flush=True)
        return measured, reused

    def rows():
        columns, *values = tomllib.loads(database.read_text(encoding="utf-8"))["ntt"]["matrix"]
        return [dict(zip(columns, row)) for row in values]

    p = 63487
    assert tune("cold", p, 6)[0] > 0
    cold = rows()
    assert all(row["naf_window"] == 6 and row["b1"] == 32 for row in cold)
    assert {row["length"] for row in cold} == {2560,4096,8192}
    assert {row["dataflow"] & 384 for row in cold if row["length"] == 2560} == {0,128,384}
    for row in cold:
        assert row["sample_target_ms"] == 100 and row["repeats"] == 2
        rate = 1000 * row["batch"] * row["timed_steps"] * row["repeats"] / row["total_device_ms"]
        assert abs(row["iter_per_s"] / rate - 1) < 1e-7
    snapshot = database.read_bytes()
    assert tune("hot", p, 6) == (1,len(cold))
    assert ("status=updated" in (out/"hot/run.log").read_text()) == (snapshot != database.read_bytes())
    measured, reused = tune("grow", p, 12)
    assert measured and reused
    assert tune("shrink", p, 4)[1] > 0
    snapshot = database.read_bytes()
    assert tune("compatible", 63485, 6)[0] == 1
    assert ("status=updated" in (out/"compatible/run.log").read_text()) == (snapshot != database.read_bytes())
    measured, reused = tune("fusion_gate", 63493, 6)
    assert measured and reused
    assert any(r["reference_bits"] == 63493 and r["length"] == 2560 and not r["dataflow"] & 32768
               for r in rows())
    tune("exhaustive", p, 6, True)
    log = (out / "exhaustive/run.log").read_text(encoding="utf-8")
    caps = {int(n): int(b) for n,b in re.findall(r"length=(\d+) bpw=\d+ naf=6 capacity=(\d+)", log)}
    for length, cap in caps.items():
        layouts = (0,128,384) if length == 2560 else (0,)
        for layout in layouts:
            covered = {r["batch"] for r in rows() if r["length"] == length and r["dataflow"] & 384 == layout
                       and r["sample_phase"] == 2}
            assert set(range(1,cap+1)) <= covered
    flags = ["-sigma","26","-gpucurves","1","-save","out.save","32","32"]
    expression = f"2^{p}-1\n"
    log = run("production", 6, flags, expression)
    assert "source=tune" in log
    winner = max((r for r in rows() if r["reference_bits"] == p and r["b1"] == 32
                  and r["batch"] == 1 and r["sample_phase"] == 2), key=lambda r: r["iter_per_s"])
    actual = dict(re.findall(r"(\w+)=([^ ]+)", re.search(r"stage1_ntt_plan: (.*)", log)[1]))
    for key, column in (("length","length"), ("tile_layers","tile"), ("graph_bits","graph_bits"),
                        ("outer_max_layers","max_radix_bits"), ("dataflow","dataflow")):
        assert int(actual[key]) == winner[column]
    run("linear", 6, flags, expression, 'ntt_convolution = "linear"\n')
    assert saved_points(out/"production/out.save",p,32,True) == saved_points(out/"linear/out.save",p,32,True)
    stronger = ["--tune","ntt,batch=1-128,time=2","--bits",str(p),"--naf-w","6"]
    limited = run("wall_budget", 6, [*stronger,"--tune-seconds","1","64"])
    assert "status=time_limit" in limited
    partial = [r for r in rows() if r["b1"] == 64]
    assert partial
    continued = run("continue_budget", 6, [*stronger,"64"])
    found = re.search(r"measured=(\d+) reused=(\d+)", continued)
    assert found and all(int(value) > 0 for value in found.groups())
    print("PASS native NAF, budget, mixed layouts, dispatch gate and production endpoint", flush=True)


if __name__ == "__main__":
    main()
