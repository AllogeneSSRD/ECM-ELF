#!/usr/bin/env python3
"""Generate worktodo files for Android batch execution.

Outputs:
  worktodo_selftest.txt   - 6 correctness tests with expected factors
  worktodo_benchmark.txt  - performance benchmark matching ecm_benchmark.ps1

Format: shell-residue lines compatible with desktop ecm.exe and Android parser.
Lines starting with # are expected factors (parsed by selftest validator).
"""
import os, sys

OUTDIR = sys.argv[1] if len(sys.argv) > 1 else "."

# ── Selftest ────────────────────────────────────────────────────────────
SELFTEST_LINES = [
    ("(2^151-1)", "3:2026",          "1e4", 1,  "391612124215324515959"),
    ("(2^241-1)", "3:3334796807",    "1e4", 1,  "22000409"),
    ("(2^347-1)", "3:561219477",     "1e4", 1,  "14143189112952632419639"),
    ("(2^421-1)", "3:268526266",     "1e4", 1,  "614002928307599"),
    ("(2^677-1)", "3:4001686290",    "1e3", 1,  "1943118631"),
    ("(2^991-1)", "3:822692423",     "1e3", 1,  "231620367206687"),
]

SELFTEST_HEADER = """# ── ECM Android Selftest ──
#  验证 GPU stage1 因子分解正确性。
#  格式: echo '<N>' | ecm.exe [flags] B1 B2
#  下一行 # 开头的注释为期望因子。
#  echo 和 .exe 路径前缀在 Android 上被自适应剥离。
"""

def make_selftest():
    lines = [SELFTEST_HEADER]
    for n_expr, sigma, b1, curves, expected in SELFTEST_LINES:
        lines.append(
            f"echo '{n_expr}' | .\\build\\Debug\\ecm.exe "
            f"-v -d 0 -gpu -sigma {sigma} -gpucurves {curves} {b1} 0"
        )
        lines.append(f"# {expected}")
    return "\n".join(lines) + "\n"

# ── Benchmark ──────────────────────────────────────────────────────────
# Parameters matching ecm_benchmark.ps1 defaults
BENCH_N      = "2^421-1"
BENCH_SIGMA  = "3:2026"
BENCH_DEVICE = 0   # Android single-GPU default; desktop dGPU users may edit this in the generated file
BENCH_CURVES = [1, 32, 64, 128, 256, 384, 512, 1024, 1536, 2048, 3072, 4096, 6144, 9216, 12288, 16384]
BENCH_B1     = "1e4"

BENCH_HEADER = """# ── ECM Android Benchmark ──
#  性能测试，匹配 ecm_benchmark.ps1 参数。
#  N={N}, sigma={sigma}, device={device}
#  B1={B1} 固定，遍历 16 个 gpucurves。
""".format(N=BENCH_N, sigma=BENCH_SIGMA, device=BENCH_DEVICE, B1=BENCH_B1)

def make_benchmark():
    lines = [BENCH_HEADER]
    for curves in BENCH_CURVES:
        lines.append(
            f"echo '({BENCH_N})' | .\\build\\Debug\\ecm.exe "
            f"-v -d {BENCH_DEVICE} -gpu -sigma {BENCH_SIGMA} "
            f"-gpucurves {curves} {BENCH_B1} 0"
        )
    return "\n".join(lines) + "\n"

# ── Write ──────────────────────────────────────────────────────────────
os.makedirs(OUTDIR, exist_ok=True)

selftest_path = os.path.join(OUTDIR, "worktodo_selftest.txt")
bench_path    = os.path.join(OUTDIR, "worktodo_benchmark.txt")

with open(selftest_path, "w", encoding="utf-8") as f:
    f.write(make_selftest())
print(f"wrote {selftest_path} ({len(SELFTEST_LINES)} tests)")

with open(bench_path, "w", encoding="utf-8") as f:
    f.write(make_benchmark())
print(f"wrote {bench_path} ({len(BENCH_CURVES)} curves @ B1={BENCH_B1})")
