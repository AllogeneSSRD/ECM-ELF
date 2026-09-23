#!/usr/bin/env python3
"""Generate mp_submod_asm_block64_generated.cl (64-limb fused sub-mod asm block)."""

from pathlib import Path

from mp_asm_block_gen import write_sub_block_file

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "kernels/opencl/bench/mp_addsub/generated/asm_sub_block64.cl"


def main() -> None:
    write_sub_block_file(OUT, 64, "gen_mp_submod_asm_block64.py")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
