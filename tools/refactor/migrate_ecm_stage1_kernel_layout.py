#!/usr/bin/env python3
"""Migrate kernels/opencl to unified mont_mul / mont_sqr / add_mod / sub_mod layout."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
K = ROOT / "kernels/opencl"
SRC_ECM = K / "ecm_stage1.cl"
LEGACY_ECM = ROOT / "cgbn/backends/opencl/kernels/ecm_stage1.cl"


def write(rel: str, content: str) -> None:
    p = K / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    print("wrote", p.relative_to(ROOT))


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def slice_lines(path: Path, start: int, end: int) -> str:
    lines = read_text(path).splitlines(keepends=True)
    return "".join(lines[start - 1 : end])


def append_mont_wrappers(body: str, mul_call: str, sqr_uses_mul: bool = True) -> str:
    out = body.rstrip() + "\n\n"
    out += (
        "static inline void ecm_stage1_mont_mul(uint *out, const uint *a, const uint *b,\n"
        "                                       const uint *N, uint np0, uint limbs) {\n"
        f"    {mul_call};\n"
        "    (void)limbs;\n"
        "}\n\n"
    )
    if sqr_uses_mul:
        out += (
            "static inline void ecm_stage1_mont_sqr(uint *out, const uint *a, const uint *N,\n"
            "                                       uint np0, uint limbs) {\n"
            "    ecm_stage1_mont_mul(out, a, a, N, np0, limbs);\n"
            "}\n"
        )
    return out


def migrate_mont_mul() -> None:
    renames = {
        "mont_mul/unroll384.cl": ("mont_mul/unroll_384.cl", "mont_mul_stage1_unroll_only_384(out, a, b, N, np0)"),
        "mont_mul/unroll512.cl": ("mont_mul/unroll_512.cl", "mont_mul_stage1_unroll_only_512(out, a, b, N, np0)"),
        "mont_mul/unroll32.cl": ("mont_mul/unroll_32.cl", "mont_mul_stage1_unroll32(out, a, b, N, np0, limbs)"),
        "mont_mul/priv_opt.cl": ("mont_mul/priv_opt.cl", "mont_mul_stage1_priv_opt(out, a, b, N, np0, limbs)"),
        "mont_mul/mont4096/unroll64.cl": (
            "mont_mul/4096/unroll_64.cl",
            "mont_mul_stage1_unroll64_4096(out, a, b, N, np0)",
        ),
        "mont_mul/mont4096/unroll64_mt2.cl": (
            "mont_mul/4096/unroll_64_mt2.cl",
            "mont_mul_stage1_unroll64_4096(out, a, b, N, np0)",
        ),
        "mont_mul/mont4096/fips4096.cl": (
            "mont_mul/4096/fips_4096.cl",
            "mont_mul_stage1_fips4096(out, a, b, N, np0)",
        ),
    }
    for old, (new, call) in renames.items():
        old_p = K / old
        if not old_p.exists():
            continue
        body = read_text(old_p)
        if "ecm_stage1_mont_mul" not in body:
            body = append_mont_wrappers(body, call)
        write(new, body)
        if old != new:
            old_p.unlink(missing_ok=True)

    shutil.rmtree(K / "mont_mul/mont4096", ignore_errors=True)

    write(
        "mont_mul/dispatch.cl",
        "// Montgomery mul dispatch shell (implementation from loaded operator file).\n"
        "static inline void mont_mul_stage1(uint *out, const uint *a, const uint *b,\n"
        "                                   const uint *N, uint np0, uint limbs) {\n"
        "    ecm_stage1_mont_mul(out, a, b, N, np0, limbs);\n"
        "}\n",
    )
    write(
        "mont_sqr/dispatch.cl",
        "// Montgomery sqr dispatch shell (implementation from loaded operator file).\n"
        "static inline void mont_sqr_stage1(uint *out, const uint *a, const uint *N, uint np0,\n"
        "                                   uint limbs) {\n"
        "    ecm_stage1_mont_sqr(out, a, N, np0, limbs);\n"
        "}\n",
    )


def extract_addsub_from_entry() -> None:
    if not SRC_ECM.exists():
        return
    blocks = {
        "add_mod/fused_unroll.cl": (179, 201),
        "sub_mod/fused_unroll.cl": (203, 224),
        "add_mod/unroll_512b.cl": (226, 230),
        "sub_mod/unroll_512b.cl": (232, 235),
        "add_mod/unroll_4096b.cl": (238, 266),
        "sub_mod/unroll_4096b.cl": (268, 295),
        "add_mod/asm_4096b.cl": (297, 314),
        "sub_mod/asm_4096b.cl": (316, 334),
        "add_mod/asm_512b.cl": (337, 341),
        "add_mod/fused.cl": (477, 519),
        "sub_mod/fused.cl": (553, 567),
    }
    for rel, (start, end) in blocks.items():
        chunk = slice_lines(SRC_ECM, start, end).strip() + "\n"
        if rel.startswith("add_mod/"):
            wrapper = (
                "\nstatic inline void ecm_stage1_add_mod(uint *r, const uint *a, const uint *b,\n"
                "                                        const uint *N, uint limbs) {\n"
            )
            if "asm_4096b" in rel:
                wrapper += "    if (limbs == 128u) { mp_add_mod_asm_b32_4096(r, a, b, N); return; }\n"
            elif "asm_512b" in rel:
                wrapper += "    if (limbs == 16u) { mp_add_mod_asm_b16_512(r, a, b, N); return; }\n"
            elif "unroll_4096b" in rel:
                wrapper += "    if (limbs == 128u) { mp_add_mod_fused_unroll_b32_4096(r, a, b, N); return; }\n"
            elif "unroll_512b" in rel:
                wrapper += "    if (limbs == 16u) { mp_add_mod_fused_unroll_b16_512(r, a, b, N); return; }\n"
            elif "fused_unroll" in rel:
                wrapper += "    if (limbs == MAX_LIMBS) { mp_add_mod_fused_unroll(r, a, b, N); return; }\n"
            else:
                wrapper += "    mp_add_mod_fused_unroll(r, a, b, N);\n"
            wrapper += "    (void)limbs;\n}\n"
        else:
            wrapper = (
                "\nstatic inline int ecm_stage1_sub_mod(uint *r, const uint *a, const uint *b,\n"
                "                                       const uint *N, uint limbs) {\n"
            )
            if "asm_4096b" in rel:
                wrapper += "    if (limbs == 128u) { return mp_sub_mod_asm_b32_4096(r, a, b, N); }\n"
            elif "unroll_4096b" in rel:
                wrapper += "    if (limbs == 128u) { return mp_sub_mod_fused_unroll_b32_4096(r, a, b, N); }\n"
            elif "unroll_512b" in rel:
                wrapper += "    if (limbs == 16u) { return mp_sub_mod_fused_unroll_b16_512(r, a, b, N); }\n"
            elif "fused_unroll" in rel:
                wrapper += "    if (limbs == MAX_LIMBS) { return mp_sub_mod_fused_unroll(r, a, b, N); }\n"
            else:
                wrapper += "    return mp_sub_mod_fused_unroll(r, a, b, N);\n"
            wrapper += "    return 0;\n}\n"
        write(rel, f"// Extracted stage1 {rel}\n{chunk}{wrapper}")

    write(
        "add_mod/dispatch.cl",
        "static inline void mp_add_mod(uint *r, const uint *a, const uint *b, const uint *N, uint limbs) {\n"
        "    ecm_stage1_add_mod(r, a, b, N, limbs);\n"
        "}\n",
    )
    write(
        "sub_mod/dispatch.cl",
        "static inline int mp_sub_mod(uint *r, const uint *a, const uint *b, const uint *N, uint limbs) {\n"
        "    return ecm_stage1_sub_mod(r, a, b, N, limbs);\n"
        "}\n",
    )


def slim_ecm_stage1() -> None:
    if not SRC_ECM.exists():
        return
    text = read_text(SRC_ECM)
    lines = text.splitlines(keepends=True)
    remove_ranges = [
        (73, 126),   # mont dispatch (now in mont_* /dispatch.cl)
        (128, 567),  # addsub bodies + mp_add_mod/mp_sub_mod
    ]
    remove = set()
    for a, b in remove_ranges:
        remove.update(range(a, b + 1))
    kept = [line for i, line in enumerate(lines, start=1) if i not in remove]
    # Remove addsub path enum block if still present
    out = "".join(kept)
    out = re.sub(
        r"#ifndef ECM_STAGE1_ADDMOD_PATH.*?#define ECM_ADDSUB_PATH_ASM_384B 13\n",
        "",
        out,
        flags=re.S,
    )
    write("ecm_stage1.cl", out)


def main() -> None:
    migrate_mont_mul()
    extract_addsub_from_entry()
    slim_ecm_stage1()
    shutil.rmtree(K / "addsub", ignore_errors=True)


if __name__ == "__main__":
    main()
