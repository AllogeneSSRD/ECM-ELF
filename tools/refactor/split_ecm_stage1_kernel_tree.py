#!/usr/bin/env python3
"""Split cgbn ecm_stage1.cl into kernels/opencl/ modular layout (one-time / regen)."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "cgbn/backends/opencl/kernels/ecm_stage1.cl"
OUT = ROOT / "kernels/opencl"


def lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines(keepends=True)


def write(rel: str, content: str) -> None:
    p = OUT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    print("wrote", p.relative_to(ROOT))


def main() -> None:
    src = lines(SRC)
    n = len(src)

    def slice_lines(a: int, b: int) -> str:
        # 1-based inclusive line numbers
        return "".join(src[a - 1 : b])

    # Shared config + limb helpers used before entry dispatch.
    write(
        "common/stage1_config.h.cl",
        "// ECM stage1 shared compile-time configuration (prepended first).\n"
        + slice_lines(1, 71),
    )
    write(
        "common/mp_priv.h.cl",
        "// ECM stage1 multi-precision limb primitives.\n" + slice_lines(540, 592),
    )

    write(
        "mont_mul/unroll512.cl",
        "// Stage1 Montgomery mul — 512-bit unroll-only.\n" + slice_lines(73, 123),
    )
    write(
        "mont_mul/unroll384.cl",
        "// Stage1 Montgomery mul — 384-bit unroll-only.\n" + slice_lines(125, 194),
    )
    write(
        "mont_mul/mont4096/unroll64.cl",
        "// Stage1 Montgomery mul — 4096-bit unroll64.\n" + slice_lines(196, 241),
    )
    write(
        "mont_mul/mont4096/unroll64_mt2.cl",
        "// Stage1 Montgomery mul — 4096-bit unroll64 MT2 local.\n" + slice_lines(243, 369),
    )
    write(
        "mont_mul/priv_opt.cl",
        "// Stage1 Montgomery mul — generic priv_opt.\n" + slice_lines(371, 434),
    )
    write(
        "mont_mul/unroll32.cl",
        "// Stage1 Montgomery mul — unroll32.\n" + slice_lines(436, 484),
    )

    # Entry: preamble guards + mont dispatch + sqr + addsub/ladder (mont bodies removed).
    removed = set(range(73, 485))  # mont implementations
    removed.update(range(540, 593))  # mp_priv (moved to common)
    entry: list[str] = []
    for i, line in enumerate(src, start=1):
        if i in removed:
            continue
        entry.append(line)
    write("ecm_stage1.cl", "".join(entry))

    # FIPS 4096 supplement (from legacy mont4096 paths file).
    fips_src = ROOT / "cgbn/backends/opencl/kernels/ecm_stage1_mont4096_paths.cl"
    write(
        "mont_mul/mont4096/fips4096.cl",
        fips_src.read_text(encoding="utf-8"),
    )


if __name__ == "__main__":
    main()
