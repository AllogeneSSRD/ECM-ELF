#!/usr/bin/env python3
"""Regenerate all mp_addsub generated OpenCL sources."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Generator scripts live next to this driver (tools/gen); the repo root is two
# levels up. Keep these derived from __file__ so the driver works from any CWD.
GEN = Path(__file__).resolve().parent
ROOT = GEN.parents[1]

SCRIPTS = [
    "gen_mp_add_mod_unroll.py",
    "gen_mp_sub_mod_unroll.py",
    "gen_mp_addmod_asm_block16.py",
    "gen_mp_addmod_asm_block32.py",
    "gen_mp_addmod_asm_block64.py",
    "gen_mp_submod_asm_block32.py",
    "gen_mp_submod_asm_block64.py",
    "gen_mp_addmod_asm_fused.py",
    "gen_mp_submod_asm_fused.py",
    "gen_mp_addsub_asm_block32_stage1.py",
    "gen_mp_addsub_asm_block16_stage1.py",
]


def main() -> None:
    MP_GEN = ROOT / "kernels/opencl/bench/mp_addsub/generated"
    MP_GEN.mkdir(parents=True, exist_ok=True)
    (ROOT / "kernels/opencl/bench/mp_addsub/stage1").mkdir(parents=True, exist_ok=True)

    for name in SCRIPTS:
        path = GEN / name
        if not path.is_file():
            print(f"skip missing {name}")
            continue
        print(f"running {name}...")
        subprocess.check_call([sys.executable, str(path)], cwd=str(ROOT))
    print("done.")


if __name__ == "__main__":
    main()
