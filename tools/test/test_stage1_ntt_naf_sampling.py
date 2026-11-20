"""Compare independent NAF tune matrices and cross-B1 cache hits on one GPU."""
import argparse
import math
import re
import subprocess
import tomllib
from pathlib import Path


def sample_shape(b1, window):
    scalar = 48 * math.lcm(*range(1, b1 + 1))
    digits = []
    while scalar:
        digit = scalar % (1 << window) if scalar & 1 else 0
        if digit >= 1 << (window - 1):
            digit -= 1 << window
        digits.append(digit)
        scalar = (scalar - digit) // 2
    digits.reverse()
    n = len(digits)
    spans = min(8, (n + 63) // 64)
    sample = []
    for i in range(spans):
        lo, hi = i*n//spans, (i+1)*n//spans
        length = min(64, hi-lo)
        start = lo + (hi-lo-length)//2
        sample.extend(digits[start:start+length])
    return n, len(sample), sum(bool(d) for d in sample)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, required=True)
    args = parser.parse_args()
    exe, out = args.exe.resolve(strict=True), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)

    def run(source, b1):
        work = out / f"from{source}_b{b1}"
        work.mkdir()
        db = out / f"b{source}.toml"
        cfg = work / "ecm.toml"
        cfg.write_text(f'config_version = 4\n[shared]\nlog_file = ""\ntune_file = "{db.as_posix()}"\n'
                       f'[stage1]\ntarget = "gpu"\ngpu_algorithm = "ntt"\nstage1_gpu_device = {args.device}\n'
                       'ntt_convolution = "linear"\nntt_tile = 12\nntt_outer_radix_limit = 4\n'
                       'ntt_warp_tail = true\n', encoding="utf-8")
        before = db.read_bytes() if db.exists() else None
        flags = ["--config", str(cfg), "-tune", "ntt,exp=63487,naf=6-8,batch=1-2", "--tune-exhaustive", str(b1)]
        result = subprocess.run([str(exe), *flags], cwd=work, capture_output=True, text=True, timeout=240)
        log = result.stdout + result.stderr
        (work / "run.log").write_text(log, encoding="utf-8")
        assert result.returncode == 0, log[-2000:]
        summaries = re.findall(r"measured=(\d+) reused=(\d+)", log)
        assert len(summaries) == 3
        measured = sum(int(m) for m, _ in summaries)
        reused = sum(int(r) for _, r in summaries)
        if source != b1:
            assert measured == 0 and all(int(r) for _, r in summaries) and before == db.read_bytes()
            assert "stage1_ntt_naf_sample:" not in log
            cold = (out / f"from{source}_b{source}" / "run.log").read_text(encoding="utf-8")
            winners = re.findall(r"\bBest [^\r\n]+", cold)
            assert len(winners) == 3 and winners == re.findall(r"\bBest [^\r\n]+", log)
            return
        assert before is None and measured and reused == 0
        assert log.count("stage1_ntt_naf_sample:") == 3
        fields, *values = tomllib.loads(db.read_text(encoding="utf-8"))["ntt"]["matrix"]
        rows = [dict(zip(fields, row)) for row in values]
        assert len(rows) == measured
        shapes = {w: sample_shape(b1, w) for w in (6, 7, 8)}
        rates = {}
        for row in rows:
            assert row["benchmark_kind"] == "stage1_naf_stratified" and row["b1"] == b1
            w = row["naf_window"]
            assert (row["naf_columns"], row["block_steps"], row["sample_nonzero"]) == shapes[w]
            expected = 1000 * row["batch"] * row["timed_steps"] * row["repeats"] / row["total_device_ms"]
            assert math.isclose(row["iter_per_s"], expected, rel_tol=1e-9)
            if row["repeats"] == 5:
                rates[w, row["batch"], row["graph_bits"]] = row["iter_per_s"]
        assert set(rates) == {(w, b, g) for w in shapes for b in (1, 2) for g in (4, 8, 16, 32, 64)}
        print(f"B1={b1} columns/sample/nonzero={shapes} measured={measured}", flush=True)
        return rates

    bounds = (100, 500, 10000)
    matrices = {b: run(b, b) for b in bounds}
    for source, target in ((100, 500), (100, 10000), (10000, 100)):
        run(source, target)
        trained, actual = matrices[source], matrices[target]
        winner = max(trained, key=trained.get)
        best = max(actual, key=actual.get)
        print(f"{source}->{target}: reused naf/batch/graph={winner}, independent best={best}, "
              f"actual={actual[winner]:.1f}/{actual[best]:.1f} iter/s "
              f"({100*actual[winner]/actual[best]:.2f}%)", flush=True)


if __name__ == "__main__":
    main()
