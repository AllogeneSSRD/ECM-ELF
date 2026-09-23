#!/usr/bin/env python3
"""Migrate stage1 operator .cl files to mont_mul_unroll_384b / add_mod_unroll_384b naming."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
K = ROOT / "kernels/opencl"

MONT = [
    ("mont_mul/unroll_384.cl", "mont_mul/mont_mul_unroll_384b.cl", "mont_mul_unroll_384b", "mont_sqr_unroll_384b",
     ("mont_mul_stage1_unroll_only_384",)),
    ("mont_mul/unroll_512.cl", "mont_mul/mont_mul_unroll_512b.cl", "mont_mul_unroll_512b", "mont_sqr_unroll_512b",
     ("mont_mul_stage1_unroll_only_512",)),
    ("mont_mul/unroll_32.cl", "mont_mul/mont_mul_unroll_32.cl", "mont_mul_unroll_32", "mont_sqr_unroll_32",
     ("mont_mul_stage1_unroll32",)),
    ("mont_mul/priv_opt.cl", "mont_mul/mont_mul_priv_opt.cl", "mont_mul_priv_opt", "mont_sqr_priv_opt",
     ("mont_mul_stage1_priv_opt",)),
    ("mont_mul/4096/unroll_64.cl", "mont_mul/mont_mul_unroll_4096b.cl", "mont_mul_unroll_4096b", "mont_sqr_unroll_4096b",
     ("mont_mul_stage1_unroll64_4096",)),
    ("mont_mul/4096/unroll_64_mt2.cl", "mont_mul/mont_mul_unroll_4096b_mt2.cl", "mont_mul_unroll_4096b_mt2",
     "mont_sqr_unroll_4096b_mt2", ("mont_mul_stage1_unroll64_4096_mt2_local", "mont_mul_stage1_unroll64_4096_mt2")),
    ("mont_mul/4096/fips_4096.cl", "mont_mul/mont_mul_fips_4096b.cl", "mont_mul_fips_4096b", "mont_sqr_fips_4096b",
     ("mont_mul_stage1_fips4096",)),
]

ADD_MAP = {
    "add_mod/fused_unroll.cl": ("add_mod/add_mod_fused_unroll.cl", "add_mod_fused_unroll", "mp_add_mod_fused_unroll"),
    "add_mod/fused.cl": ("add_mod/add_mod_fused.cl", "add_mod_fused", None),
    "add_mod/unroll_512b.cl": ("add_mod/add_mod_unroll_512b.cl", "add_mod_unroll_512b", "mp_add_mod_fused_unroll_b16_512"),
    "add_mod/unroll_4096b.cl": ("add_mod/add_mod_unroll_4096b.cl", "add_mod_unroll_4096b", "mp_add_mod_fused_unroll_b32_4096"),
    "add_mod/asm_512b.cl": ("add_mod/add_mod_asm_512b.cl", "add_mod_asm_512b", "mp_add_mod_asm_b16_512"),
    "add_mod/asm_4096b.cl": ("add_mod/add_mod_asm_4096b.cl", "add_mod_asm_4096b", "mp_add_mod_asm_b32_4096"),
}

SUB_MAP = {
    "sub_mod/fused_unroll.cl": ("sub_mod/sub_mod_fused_unroll.cl", "sub_mod_fused_unroll", "mp_sub_mod_fused_unroll"),
    "sub_mod/fused.cl": ("sub_mod/sub_mod_fused.cl", "sub_mod_fused", None),
    "sub_mod/unroll_512b.cl": ("sub_mod/sub_mod_unroll_512b.cl", "sub_mod_unroll_512b", "mp_sub_mod_fused_unroll_b16_512"),
    "sub_mod/unroll_4096b.cl": ("sub_mod/sub_mod_unroll_4096b.cl", "sub_mod_unroll_4096b", "mp_sub_mod_fused_unroll_b32_4096"),
    "sub_mod/asm_512b.cl": ("sub_mod/sub_mod_asm_512b.cl", "sub_mod_asm_512b", None),
    "sub_mod/asm_4096b.cl": ("sub_mod/sub_mod_asm_4096b.cl", "sub_mod_asm_4096b", "mp_sub_mod_asm_b32_4096"),
}


def strip_wrappers(text: str) -> str:
    for pat in (
        r"\nstatic inline void ecm_stage1_mont_mul\([^)]*\)\s*\{[^}]*\}\n",
        r"\nstatic inline void ecm_stage1_mont_sqr\([^)]*\)\s*\{[^}]*\}\n",
        r"\nstatic inline void ecm_stage1_add_mod\([^)]*\)\s*\{[^}]*\}\n",
        r"\nstatic inline int ecm_stage1_sub_mod\([^)]*\)\s*\{[^}]*\}\n",
    ):
        text = re.sub(pat, "\n", text, flags=re.S)
    return text


def mont_migrate(src_rel: str, dst_rel: str, mul: str, sqr: str, old_names: tuple[str, ...]) -> None:
    src = K / src_rel
    if not src.exists():
        return
    text = strip_wrappers(src.read_text(encoding="utf-8"))
    for old in old_names:
        if f"void {old}(" in text:
            text = text.replace(f"void {old}(", f"void {mul}(", 1)
            if ", uint limbs)" not in text.split(f"void {mul}(")[1].split("{")[0]:
                text = re.sub(
                    rf"(static inline void {re.escape(mul)}\([^)]*uint np0)\)",
                    r"\1, uint limbs)",
                    text,
                    count=1,
                )
            if "(void)limbs;" not in text:
                text = text.replace(f"void {mul}(", f"void {mul}(", 1)
            break
    if f"void {sqr}(" not in text:
        text = (
            text.rstrip()
            + f"\n\nstatic inline void {sqr}(uint *out, const uint *a, const uint *N, uint np0, uint limbs) {{\n"
            + f"    {mul}(out, a, a, N, np0, limbs);\n"
            + "}\n"
        )
    dst = K / dst_rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text, encoding="utf-8")
    src.unlink(missing_ok=True)


def addsub_migrate(src_rel: str, dst_rel: str, fn: str, core: str | None, ret_int: bool) -> None:
    src = K / src_rel
    if not src.exists():
        return
    text = strip_wrappers(src.read_text(encoding="utf-8"))
    if core:
        text = text.replace(f"void {core}(", f"static inline void {fn}_body(", 1)
        text = text.replace(f"int {core}(", f"static inline int {fn}_body(", 1)
    text = re.sub(r"\bmp_add_mod_(\w+)", r"add_mod_\1", text)
    text = re.sub(r"\bmp_sub_mod_(\w+)", r"sub_mod_\1", text)
    if f"void {fn}(" not in text and f"int {fn}(" not in text:
        if ret_int:
            text += (
                f"\nstatic inline int {fn}(uint *r, const uint *a, const uint *b, const uint *N, uint limbs) {{\n"
                f"    return {fn}_body(r, a, b, N, limbs);\n"
                f"}}\n"
            )
        else:
            text += (
                f"\nstatic inline void {fn}(uint *r, const uint *a, const uint *b, const uint *N, uint limbs) {{\n"
                f"    {fn}_body(r, a, b, N, limbs);\n"
                f"    (void)limbs;\n"
                f"}}\n"
            )
    dst = K / dst_rel
    dst.write_text(text, encoding="utf-8")
    src.unlink(missing_ok=True)


def bits_migrate(side: str, kind: str, bits: int) -> None:
    old = K / f"{side}_mod/bits/{kind}_{bits}b.cl"
    if not old.exists():
        old = K / f"{side}_mod/{kind}_{bits}b.cl"
    if not old.exists():
        return
    fn = f"{'add_mod' if side == 'add' else 'sub_mod'}_{kind}_{bits}b"
    text = strip_wrappers(old.read_text(encoding="utf-8"))
    text = re.sub(rf"\bmp_{side}_mod_{kind}_{bits}b", f"{fn}_body", text)
    if side == "add":
        text += (
            f"\nstatic inline void {fn}(uint *r, const uint *a, const uint *b, const uint *N, uint limbs) {{\n"
            f"    if (limbs == 16u) {{ {fn}_body(r, a, b, N); }}\n"
            f"    (void)limbs;\n"
            f"}}\n"
        )
    else:
        text += (
            f"\nstatic inline int {fn}(uint *r, const uint *a, const uint *b, const uint *N, uint limbs) {{\n"
            f"    if (limbs == 16u) {{ return {fn}_body(r, a, b, N); }}\n"
            f"    return 0;\n"
            f"}}\n"
        )
    dst = K / f"{'add_mod' if side == 'add' else 'sub_mod'}/{fn}.cl"
    dst.write_text(text, encoding="utf-8")
    old.unlink(missing_ok=True)
    bits_dir = K / f"{side}_mod/bits"
    if bits_dir.exists() and not any(bits_dir.iterdir()):
        bits_dir.rmdir()


def main() -> None:
    for item in MONT:
        mont_migrate(*item)
    for src, (dst, fn, core) in ADD_MAP.items():
        addsub_migrate(src, dst, fn, core, False)
    for src, (dst, fn, core) in SUB_MAP.items():
        addsub_migrate(src, dst, fn, core, True)
    for bits in (128, 192, 256, 384):
        bits_migrate("add", "unroll", bits)
        bits_migrate("sub", "unroll", bits)
        bits_migrate("add", "asm", bits)
        bits_migrate("sub", "asm", bits)
    for p in ("mont_mul/dispatch.cl", "mont_sqr/dispatch.cl", "add_mod/dispatch.cl", "sub_mod/dispatch.cl"):
        (K / p).unlink(missing_ok=True)
    shutil.rmtree(K / "mont_mul/4096", ignore_errors=True)
    print("done")


if __name__ == "__main__":
    main()
