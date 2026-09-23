"""Canonical paths for mp_addsub OpenCL kernels and generators."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KERNELS = ROOT / "cgbn" / "backends" / "opencl" / "kernels"
MP_ADDSUB = KERNELS / "mp_addsub"
MP_GEN = MP_ADDSUB / "generated"

# Hand-written / shared
ASM_BASE = MP_ADDSUB / "asm_base.cl"

# Generated add/sub manual unroll (scalar full unroll per MAX_LIMBS)
ADD_FUSED_UNROLL_MANUAL = MP_GEN / "add_fused_unroll_manual.cl"
SUB_FUSED_UNROLL_MANUAL = MP_GEN / "sub_fused_unroll_manual.cl"

# Generated pragma unroll (ECM fused_unroll_auto)
FUSED_UNROLL_AUTO = MP_GEN / "fused_unroll_auto.cl"

# ASM building blocks and entry kernels
ASM_BLOCK = {
    8: MP_GEN / "asm_block08.cl",
    16: MP_GEN / "asm_block16.cl",
    32: MP_GEN / "asm_block32.cl",
    64: MP_GEN / "asm_block64.cl",
}
ASM_ADD_KERNELS = MP_GEN / "asm_add_kernels.cl"
ASM_SUB_KERNELS = MP_GEN / "asm_sub_kernels.cl"

# ECM stage1-only (private asm blocks)
STAGE1_ASM_BLOCK32 = MP_ADDSUB / "stage1" / "asm_block32_stage1.cl"

# Deprecated kernel root copies (generators write to MP_GEN only)
LEGACY = {
    "add_unroll": ADD_FUSED_UNROLL_MANUAL,
    "sub_unroll": SUB_FUSED_UNROLL_MANUAL,
    "auto": FUSED_UNROLL_AUTO,
    "asm_base": ASM_BASE,
}
