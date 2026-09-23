#!/usr/bin/env python3
"""Patch CONTENT in tools/gen_npu_addsub.py with optimized mod kernels."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GEN = ROOT / "tools" / "gen_npu_addsub.py"

MOD_BLOCK = '''
def numpy_mp_add_mod_legacy(a: np.ndarray, b: np.ndarray, n: np.ndarray) -> np.ndarray:
    batch, limbs = a.shape
    r = np.zeros((batch, limbs), dtype=np.uint32)
    for i in range(batch):
        carry_add = 0
        carry_sub = 1
        ri = np.zeros(limbs, dtype=np.uint64)
        ai = a[i].astype(np.uint64)
        bi = b[i].astype(np.uint64)
        ni = n[i].astype(np.uint64)
        for j in range(limbs):
            s = int(ai[j]) + int(bi[j]) + carry_add
            carry_add = s >> 32
            t = (s & 0xFFFFFFFF) + ((~int(ni[j])) & 0xFFFFFFFF) + carry_sub
            carry_sub = t >> 32
            ri[j] = t & 0xFFFFFFFF
        if (carry_add | carry_sub) == 0:
            c = 0
            for j in range(limbs):
                s = int(ri[j]) + int(ni[j]) + c
                ri[j] = s & 0xFFFFFFFF
                c = s >> 32
        r[i] = ri.astype(np.uint32)
    return r


def numpy_mp_sub_mod_legacy(a: np.ndarray, b: np.ndarray, n: np.ndarray) -> np.ndarray:
    batch, limbs = a.shape
    r = np.zeros((batch, limbs), dtype=np.uint32)
    for i in range(batch):
        borrow = 0
        ri = np.zeros(limbs, dtype=np.uint64)
        ai = a[i].astype(np.uint64)
        bi = b[i].astype(np.uint64)
        ni = n[i].astype(np.uint64)
        for j in range(limbs):
            av = int(ai[j])
            bv = int(bi[j])
            w = av - bv - borrow
            ri[j] = w & 0xFFFFFFFF
            borrow = 1 if av < bv + borrow else 0
        if borrow:
            c = 0
            for j in range(limbs):
                s = int(ri[j]) + int(ni[j]) + c
                ri[j] = s & 0xFFFFFFFF
                c = s >> 32
        r[i] = ri.astype(np.uint32)
    return r


def numpy_mp_add_mod_vec(a: np.ndarray, b: np.ndarray, n: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.uint64)
    b = np.asarray(b, dtype=np.uint64)
    n = np.asarray(n, dtype=np.uint64)
    batch, limbs = a.shape
    r = np.zeros((batch, limbs), dtype=np.uint64)
    carry_add = np.zeros(batch, dtype=np.uint64)
    carry_sub = np.ones(batch, dtype=np.uint64)
    mask32 = np.uint64(0xFFFFFFFF)
    for j in range(limbs):
        s = a[:, j] + b[:, j] + carry_add
        carry_add = s >> 32
        t = (s & mask32) + ((~n[:, j]) & mask32) + carry_sub
        carry_sub = t >> 32
        r[:, j] = t & mask32
    need_fix = (carry_add | carry_sub) == 0
    if np.any(need_fix):
        c = np.zeros(batch, dtype=np.uint64)
        for j in range(limbs):
            s = r[:, j] + n[:, j] + c
            new_r = s & mask32
            new_c = s >> 32
            r[:, j] = np.where(need_fix, new_r, r[:, j])
            c = np.where(need_fix, new_c, np.uint64(0))
    return r.astype(np.uint32)


def numpy_mp_sub_mod_vec(a: np.ndarray, b: np.ndarray, n: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.uint64)
    b = np.asarray(b, dtype=np.uint64)
    n = np.asarray(n, dtype=np.uint64)
    batch, limbs = a.shape
    r = np.zeros((batch, limbs), dtype=np.uint64)
    borrow = np.zeros(batch, dtype=np.uint64)
    mask32 = np.uint64(0xFFFFFFFF)
    for j in range(limbs):
        av = a[:, j]
        bv = b[:, j]
        w = av - bv - borrow
        r[:, j] = w & mask32
        borrow = (av < bv + borrow).astype(np.uint64)
    need_fix = borrow.astype(bool)
    if np.any(need_fix):
        c = np.zeros(batch, dtype=np.uint64)
        for j in range(limbs):
            s = r[:, j] + n[:, j] + c
            new_r = s & mask32
            new_c = s >> 32
            r[:, j] = np.where(need_fix, new_r, r[:, j])
            c = np.where(need_fix, new_c, np.uint64(0))
    return r.astype(np.uint32)


numpy_mp_add_mod = numpy_mp_add_mod_vec
numpy_mp_sub_mod = numpy_mp_sub_mod_vec


'''


def patch_content(content: str) -> str:
    if "def numpy_mp_add_mod_vec" not in content:
        if "def numpy_mp_add_mod_legacy" in content:
            old_start = content.index("def numpy_mp_add_mod_legacy")
        else:
            old_start = content.index("def numpy_mp_add_mod")
        old_end = content.index("class NPUAddSubBackend")
        content = content[:old_start] + MOD_BLOCK + content[old_end:]

    for old_body, new_body in (
        (
            "    def mp_add_mod(self, a: np.ndarray, b: np.ndarray, n: np.ndarray) -> np.ndarray:\n"
            "        return numpy_mp_add_mod(a, b, n)\n\n"
            "    def mp_sub_mod(self, a: np.ndarray, b: np.ndarray, n: np.ndarray) -> np.ndarray:\n"
            "        return numpy_mp_sub_mod(a, b, n)",
            "    def mp_add_mod(self, a: np.ndarray, b: np.ndarray, n: np.ndarray) -> np.ndarray:\n"
            "        return numpy_mp_add_mod_vec(a, b, n)\n\n"
            "    def mp_sub_mod(self, a: np.ndarray, b: np.ndarray, n: np.ndarray) -> np.ndarray:\n"
            "        return numpy_mp_sub_mod_vec(a, b, n)",
        ),
        (
            "    def mp_add_mod(self, a: np.ndarray, b: np.ndarray, n: np.ndarray) -> np.ndarray:\n"
            "        return bigint_mp_add_mod(a, b, n)\n\n"
            "    def mp_sub_mod(self, a: np.ndarray, b: np.ndarray, n: np.ndarray) -> np.ndarray:\n"
            "        return bigint_mp_sub_mod(a, b, n)",
            "    def mp_add_mod(self, a: np.ndarray, b: np.ndarray, n: np.ndarray) -> np.ndarray:\n"
            "        return numpy_mp_add_mod_vec(a, b, n)\n\n"
            "    def mp_sub_mod(self, a: np.ndarray, b: np.ndarray, n: np.ndarray) -> np.ndarray:\n"
            "        return numpy_mp_sub_mod_vec(a, b, n)",
        ),
    ):
        if old_body in content:
            content = content.replace(old_body, new_body)
            break

    content = content.replace(
        "class NumpyBackend:\n"
        "    mp_add_n = staticmethod(numpy_mp_add_n)\n"
        "    mp_sub_n = staticmethod(numpy_mp_sub_n)\n"
        "    mp_add_mod = staticmethod(numpy_mp_add_mod)\n"
        "    mp_sub_mod = staticmethod(numpy_mp_sub_mod)",
        "class NumpyBackend:\n"
        "    mp_add_n = staticmethod(numpy_mp_add_n)\n"
        "    mp_sub_n = staticmethod(numpy_mp_sub_n)\n"
        "    mp_add_mod = staticmethod(numpy_mp_add_mod_vec)\n"
        "    mp_sub_mod = staticmethod(numpy_mp_sub_mod_vec)",
    )

    old_bench = """        timings.append(
            (
                "mp_add_mod",
                "npu_fused",
                bench_op(
                    lambda: npu_backend.mp_add_mod(a_batch, b_batch, n_batch),
                    warmup,
                    kernel_iterations,
                    launch_repeats,
                ),
            )
        )
        timings.append(
            (
                "mp_sub_mod",
                "npu_fused",
                bench_op(
                    lambda: npu_backend.mp_sub_mod(a_batch, b_batch, n_batch),
                    warmup,
                    kernel_iterations,
                    launch_repeats,
                ),
            )
        )"""

    new_bench = """        timings.append(
            (
                "mp_add_mod",
                "npu_fused",
                bench_op(
                    lambda: npu_backend.mp_add_mod(a_batch, b_batch, n_batch),
                    warmup,
                    kernel_iterations,
                    launch_repeats,
                ),
            )
        )
        timings.append(
            (
                "mp_sub_mod",
                "npu_fused",
                bench_op(
                    lambda: npu_backend.mp_sub_mod(a_batch, b_batch, n_batch),
                    warmup,
                    kernel_iterations,
                    launch_repeats,
                ),
            )
        )"""

    if old_bench in content:
        content = content.replace(old_bench, new_bench)

    already_bench = """        timings.append(
            (
                "mp_add_mod",
                "bigint",
                bench_op(
                    lambda: bigint_mp_add_mod(a_batch, b_batch, n_batch),
                    warmup,
                    kernel_iterations,
                    launch_repeats,
                ),
            )
        )
        timings.append(
            (
                "mp_add_mod",
                "fused_vec",
                bench_op(
                    lambda: numpy_mp_add_mod_vec(a_batch, b_batch, n_batch),
                    warmup,
                    kernel_iterations,
                    launch_repeats,
                ),
            )
        )
        timings.append(
            (
                "mp_sub_mod",
                "bigint",
                bench_op(
                    lambda: bigint_mp_sub_mod(a_batch, b_batch, n_batch),
                    warmup,
                    kernel_iterations,
                    launch_repeats,
                ),
            )
        )
        timings.append(
            (
                "mp_sub_mod",
                "fused_vec",
                bench_op(
                    lambda: numpy_mp_sub_mod_vec(a_batch, b_batch, n_batch),
                    warmup,
                    kernel_iterations,
                    launch_repeats,
                ),
            )
        )"""

    if already_bench in content:
        content = content.replace(already_bench, new_bench)

    old_numpy_mod = """    timings.append(
        (
            "mp_add_mod",
            "numpy_fused",
            bench_op(lambda: numpy_be.mp_add_mod(a_batch, b_batch, n_batch), warmup, kernel_iterations, launch_repeats),
        )
    )
    timings.append(
        (
            "mp_sub_mod",
            "numpy_fused",
            bench_op(lambda: numpy_be.mp_sub_mod(a_batch, b_batch, n_batch), warmup, kernel_iterations, launch_repeats),
        )
    )"""

    new_numpy_mod = """    timings.append(
        (
            "mp_add_mod",
            "numpy_fused",
            bench_op(
                lambda: numpy_be.mp_add_mod(a_batch, b_batch, n_batch),
                warmup,
                kernel_iterations,
                launch_repeats,
            ),
        )
    )
    timings.append(
        (
            "mp_sub_mod",
            "numpy_fused",
            bench_op(
                lambda: numpy_be.mp_sub_mod(a_batch, b_batch, n_batch),
                warmup,
                kernel_iterations,
                launch_repeats,
            ),
        )
    )
    timings.append(
        (
            "mp_add_mod",
            "fused_legacy",
            bench_op(
                lambda: numpy_mp_add_mod_legacy(a_batch, b_batch, n_batch),
                warmup,
                kernel_iterations,
                launch_repeats,
            ),
        )
    )
    timings.append(
        (
            "mp_sub_mod",
            "fused_legacy",
            bench_op(
                lambda: numpy_mp_sub_mod_legacy(a_batch, b_batch, n_batch),
                warmup,
                kernel_iterations,
                launch_repeats,
            ),
        )
    )"""

    if old_numpy_mod in content:
        content = content.replace(old_numpy_mod, new_numpy_mod)

    already_numpy_mod = """    timings.append(
        (
            "mp_add_mod",
            "fused_legacy",
            bench_op(
                lambda: numpy_mp_add_mod_legacy(a_batch, b_batch, n_batch),
                warmup,
                kernel_iterations,
                launch_repeats,
            ),
        )
    )
    timings.append(
        (
            "mp_sub_mod",
            "fused_legacy",
            bench_op(
                lambda: numpy_mp_sub_mod_legacy(a_batch, b_batch, n_batch),
                warmup,
                kernel_iterations,
                launch_repeats,
            ),
        )
    )"""

    if already_numpy_mod in content:
        content = content.replace(already_numpy_mod, new_numpy_mod)

    return content


def main() -> None:
    text = GEN.read_text(encoding="utf-8")
    marker = "CONTENT = r'''"
    start = text.index(marker) + len(marker)
    end = text.rindex("'''\n\n\ndef main")
    inner = text[start:end]
    inner = patch_content(inner)
    text = text[:start] + inner + text[end:]
    GEN.write_text(text, encoding="utf-8", newline="\n")
    print(f"Patched CONTENT in {GEN}")


if __name__ == "__main__":
    main()
