#!/usr/bin/env python3
"""Rename stage1 operators to mont_mul_unroll_384b style and strip dispatch wrappers."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
K = ROOT / "kernels/opencl"

MONT_RENAMES = {
    "mont_mul/unroll_384.cl": ("mont_mul/mont_mul_unroll_384b.cl", "mont_mul_unroll_384b", "mont_sqr_unroll_384b"),
    "mont_mul/unroll_512.cl": ("mont_mul/mont_mul_unroll_512b.cl", "mont_mul_unroll_512b", "mont_sqr_unroll_512b"),
    "mont_mul/unroll_32.cl": ("mont_mul/mont_mul_unroll_32.cl", "mont_mul_unroll_32", "mont_sqr_unroll_32"),
    "mont_mul/priv_opt.cl": ("mont_mul/mont_mul_priv_opt.cl", "mont_mul_priv_opt", "mont_sqr_priv_opt"),
    "mont_mul/4096/unroll_64.cl": ("mont_mul/mont_mul_unroll_4096b.cl", "mont_mul_unroll_4096b", "mont_sqr_unroll_4096b"),
    "mont_mul/4096/unroll_64_mt2.cl": (
        "mont_mul/mont_mul_unroll_4096b_mt2.cl",
        "mont_mul_unroll_4096b_mt2",
        "mont_sqr_unroll_4096b_mt2",
    ),
    "mont_mul/4096/fips_4096.cl": ("mont_mul/mont_mul_fips_4096b.cl", "mont_mul_fips_4096b", "mont_sqr_fips_4096b"),
}

ADD_RENAMES = {
    "add_mod/fused_unroll.cl": "add_mod/add_mod_fused_unroll.cl",
    "add_mod/fused.cl": "add_mod/add_mod_fused.cl",
    "add_mod/unroll_512b.cl": "add_mod/add_mod_unroll_512b.cl",
    "add_mod/unroll_4096b.cl": "add_mod/add_mod_unroll_4096b.cl",
    "add_mod/asm_512b.cl": "add_mod/add_mod_asm_512b.cl",
    "add_mod/asm_4096b.cl": "add_mod/add_mod_asm_4096b.cl",
}

SUB_RENAMES = {
    "sub_mod/fused_unroll.cl": "sub_mod/sub_mod_fused_unroll.cl",
    "sub_mod/fused.cl": "sub_mod/sub_mod_fused.cl",
    "sub_mod/unroll_512b.cl": "sub_mod/sub_mod_unroll_512b.cl",
    "sub_mod/unroll_4096b.cl": "sub_mod/sub_mod_unroll_4096b.cl",
    "sub_mod/asm_512b.cl": "sub_mod/sub_mod_asm_512b.cl",
    "sub_mod/asm_4096b.cl": "sub_mod/sub_mod_asm_4096b.cl",
}


def strip_ecm_wrappers(text: str) -> str:
    text = re.sub(
        r"\nstatic inline void ecm_stage1_mont_mul\([^)]*\)\s*\{[^}]*\}\n",
        "\n",
        text,
        flags=re.S,
    )
    text = re.sub(
        r"\nstatic inline void ecm_stage1_mont_sqr\([^)]*\)\s*\{[^}]*\}\n",
        "\n",
        text,
        flags=re.S,
    )
    text = re.sub(
        r"\nstatic inline void ecm_stage1_add_mod\([^)]*\)\s*\{[^}]*\}\n",
        "\n",
        text,
        flags=re.S,
    )
    text = re.sub(
        r"\nstatic inline int ecm_stage1_sub_mod\([^)]*\)\s*\{[^}]*\}\n",
        "\n",
        text,
        flags=re.S,
    )
    return text


def add_limb_param_to_mont_core(text: str, new_name: str, old_names: tuple[str, ...]) -> str:
    for old in old_names:
        text = text.replace(f"static inline void {old}(", f"static inline void {new_name}(")
    # Append limbs param if 5-arg mont core
    text = re.sub(
        rf"(static inline void {re.escape(new_name)}\(uint \*out, const uint \*a, const uint \*b,\s*const uint \*N, uint np0)\)",
        r"\1, uint limbs)",
        text,
        count=1,
    )
    if f"(void)limbs" not in text and "uint limbs)" in text:
        # insert (void)limbs at end of function if no use
        pass
    return text


def rename_addsub_file(src: Path, dst: Path, fn_prefix: str) -> None:
    if not src.exists():
        return
    text = strip_ecm_wrappers(src.read_text(encoding="utf-8"))
    # mp_add_mod_unroll_384b -> add_mod_unroll_384b
    text = re.sub(r"\bmp_add_mod_(\w+)", rf"{fn_prefix}_\1", text)
    text = re.sub(r"\bmp_sub_mod_(\w+)", lambda m: f"sub_mod_{m.group(1)}", text)
  # fix sub if add file - handled per side
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text, encoding="utf-8")
    src.unlink(missing_ok=True)
    print("renamed", dst.relative_to(ROOT))


def process_mont(old_rel: str, new_rel: str, mul_name: str, sqr_name: str) -> None:
    src = K / old_rel
    if not src.exists():
        return
    text = strip_ecm_wrappers(src.read_text(encoding="utf-8"))
    old_mul_names = (
        "mont_mul_stage1_unroll_only_384",
        "mont_mul_stage1_unroll_only_512",
        "mont_mul_stage1_unroll32",
        "mont_mul_stage1_priv_opt",
        "mont_mul_stage1_unroll64_4096",
        "mont_mul_stage1_unroll64_4096_mt2_local",
        "mont_mul_stage1_fips4096",
    )
    for old in old_mul_names:
        if f"static inline void {old}(" in text:
            text = text.replace(f"static inline void {old}(", f"static inline void {mul_name}(")
            if ", uint limbs)" not in text.split(f"static inline void {mul_name}(")[1].split(")")[0]:
                text = re.sub(
                    rf"(static inline void {re.escape(mul_name)}\([^)]*uint np0)\)",
                    r"\1, uint limbs)",
                    text,
                    count=1,
                )
            break
    sqr_body = (
        f"\nstatic inline void {sqr_name}(uint *out, const uint *a, const uint *N, uint np0, uint limbs) {{\n"
        f"    {mul_name}(out, a, a, N, np0, limbs);\n"
        f"}}\n"
    )
    if f"static inline void {sqr_name}(" not in text:
        text = text.rstrip() + sqr_body
    dst = K / new_rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text, encoding="utf-8")
    src.unlink(missing_ok=True)
    print("mont", dst.relative_to(ROOT))


def process_bits(side: str, bits: int, kind: str) -> None:
    old = K / f"{side}_mod/bits/{kind}_{bits}b.cl"
    if not old.exists():
        return
    prefix = "add_mod" if side == "add" else "sub_mod"
    fn = f"{prefix}_{kind}_{bits}b"
    text = strip_ecm_wrappers(old.read_text(encoding="utf-8"))
    text = re.sub(rf"\bmp_{side}_mod_{kind}_{bits}b", fn, text)
    if side == "add":
        wrapper = (
            f"\nstatic inline void {fn}(uint *r, const uint *a, const uint *b, const uint *N, uint limbs) {{\n"
            f"    if (limbs == 16u) {{ {fn}_core(r, a, b, N); }}\n"
            f"    (void)limbs;\n"
            f"}}\n"
        )
        text = text.replace(f"static inline void {fn}(", f"static inline void {fn}_core(", 1)
        if f"{fn}(" not in wrapper or f"static inline void {fn}(" not in text:
            text = text.rstrip() + wrapper.replace(f"{fn}_core", fn).replace(f"{fn}(", f"{fn}_core(")
    else:
        text = text.replace(f"static inline int {fn}(", f"static inline int {fn}_core(", 1)
        wrapper = (
            f"\nstatic inline int {fn}(uint *r, const uint *a, const uint *b, const uint *N, uint limbs) {{\n"
            f"    if (limbs == 16u) {{ return {fn}_core(r, a, b, N); }}\n"
            f"    return 0;\n"
            f"}}\n"
        )
        text = text.rstrip() + wrapper
    new = K / f"{side}_mod/{fn}.cl"
    new.write_text(text, encoding="utf-8")
    old.unlink(missing_ok=True)
    print("bits", new.relative_to(ROOT))


def main() -> None:
    for old, (new, mul, sqr) in MONT_RENAMES.items():
        process_mont(old, new, mul, sqr)
    for old, new in ADD_RENAMES.items():
        src, dst = K / old, K / new
        if src.exists():
            text = strip_ecm_wrappers(src.read_text(encoding="utf-8"))
            fn = Path(new).stem
            text = re.sub(r"\bmp_add_mod_(\w+)", rf"add_mod_\1", text)
            text = re.sub(r"\bmp_sub_mod_(\w+)", rf"add_mod_\1", text)
            if f"static inline void {fn}(" not in text:
                text += (
                    f"\nstatic inline void {fn}(uint *r, const uint *a, const uint *b, const uint *N, uint limbs) {{\n"
                    f"    add_mod_fused_unroll(r, a, b, N);\n"
                    f"    (void)limbs;\n"
                    f"}}\n"
                )
            dst.write_text(text, encoding="utf-8")
            src.unlink(missing_ok=True)
    for old, new in SUB_RENAMES.items():
        src, dst = K / old, K / new
        if src.exists():
            text = strip_ecm_wrappers(src.read_text(encoding="utf-8"))
            fn = Path(new).stem
            text = re.sub(r"\bmp_sub_mod_(\w+)", rf"sub_mod_\1", text)
            text = re.sub(r"\bmp_add_mod_(\w+)", rf"sub_mod_\1", text)
            dst.write_text(text, encoding="utf-8")
            src.unlink(missing_ok=True)
    for bits in (128, 192, 256, 384):
        process_bits("add", bits, "unroll")
        process_bits("sub", bits, "unroll")
        process_bits("add", bits, "asm")
        process_bits("sub", bits, "asm")
    for p in ["mont_mul/dispatch.cl", "mont_sqr/dispatch.cl", "add_mod/dispatch.cl", "sub_mod/dispatch.cl"]:
        (K / p).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
