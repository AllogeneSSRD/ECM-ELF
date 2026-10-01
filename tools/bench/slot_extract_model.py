#!/usr/bin/env python3
"""Isolate the slot-assembly step from the NTT: feed it the EXACT product words.

Why: with the transform now verified against a direct-DFT oracle, the remaining failure of
tools/bench/ntt_poly_probe.cu is that `poly 4 64 1 1` still returns wrong slots.  Two
possibilities, and they need very different fixes:
  (a) the extraction (slot_assemble_kernel) mangles a correct word array, or
  (b) the extraction is fine and the word array it is fed is not the exact product
      (carry stage, transform pairing, or packing/offset mismatch).
This script decides it without the GPU: build the packed operands, compute the exact product
with Python integers, lay it out as little-endian 64-bit words, and run the CURRENT
extraction logic on those words.  If every slot comes back equal to the true coefficient
mod 4294967291, the extraction is correct and the bug is upstream.

Mirrors the probe's own choices: compact packing at bit offset i*slot_bits, slot_bits =
2S + ceil(log2 P), SLOT_MOD = 4294967291, and the bpw the probe printed for that shape.
"""

import random

SLOT_MOD = 4294967291
MASK64 = (1 << 64) - 1


def pack(coeffs, slot):
    v = 0
    for i, c in enumerate(coeffs):
        v |= c << (i * slot)
    return v


def words_of(value, nwords, bpw):
    """The probe's `c` array is a DIGIT array: element j holds bits [j*bpw, (j+1)*bpw) of
    the packed product (that is what the transform convolves and what the carry stage
    reduces back to bpw bits).  It is NOT a base-2^64 word array."""
    m = (1 << bpw) - 1
    return [(value >> (bpw * j)) & m for j in range(nwords)]


def extract(i, c, slot, bpw, stride=None):
    """The current slot_assemble_kernel logic (as fixed 2026-09-30).

    `slot` is the coefficient width in bits (the window width), `stride` is the packing
    stride in bits (coefficient i starts at bit i*stride of the packed product).  The probe
    uses a WORD-ALIGNED stride = ceil(slot/bpw)*bpw; the compact stride (stride == slot,
    which is NOT a multiple of bpw) is kept here as the counter-example: it is what broke the
    probe, and this model shows the difference is upstream of the extraction (the extraction
    handles both, but the compact packing makes the convolution coefficients exceed p, so the
    digits it is fed are already wrong).
    """
    if stride is None:
        stride = slot
    wps = (slot + bpw - 1) // bpw
    base = (i * stride) // bpw
    shift = (i * stride) % bpw
    top_bits = slot - (wps - 1) * bpw
    v = 0
    for j in range(wps - 1, -1, -1):
        x = c[base + j] >> shift
        if shift:
            x |= c[base + j + 1] << (bpw - shift)   # digit array: elements hold bpw bits, NOT 64
        keep = top_bits if j == wps - 1 else bpw
        if keep < 64:
            x &= (1 << keep) - 1
        v = (((v << bpw) % SLOT_MOD) + (x % SLOT_MOD)) % SLOT_MOD
    return v


def main():
    import math
    for (P, S, bpw) in [(4, 64, 27), (64, 64, 25), (128, 257, 24), (512, 1024, 22)]:
        slot = 2 * S + math.ceil(math.log2(P))
        slot_words = (slot + bpw - 1) // bpw
        stride = slot_words * bpw          # word-aligned, what the probe packs with now
        rnd = random.Random(12345 + P)
        a = [rnd.getrandbits(S) for _ in range(P)]
        b = [rnd.getrandbits(S) for _ in range(P)]
        # true polynomial product coefficients
        prod = [0] * (2 * P - 1)
        for i in range(P):
            for j in range(P):
                prod[i + j] += a[i] * b[j]
        # The probe packs coefficient i at bit i*stride, so digit j of the packed operand is
        # < 2^bpw exactly when stride is a multiple of bpw (see the packing comment in
        # ntt_poly_probe.cu) -- that is what makes the exact product's digit array canonical
        # and N*(2^bpw)^2 < p a valid bound.
        nwords = (stride * (2 * P - 1) + slot + bpw) // bpw + 8
        for label, st in (("word-aligned", stride), ("compact", slot)):
            A = pack(a, st)
            B = pack(b, st)
            C = A * B
            c = words_of(C, nwords, bpw)
            inmax = max(max(words_of(A, nwords, bpw)), max(words_of(B, nwords, bpw)))
            bad = 0
            first = None
            for k in range(2 * P - 1):
                got = extract(k, c, slot, bpw, st)
                want = prod[k] % SLOT_MOD
                if got != want:
                    bad += 1
                    if first is None:
                        first = (k, got, want, prod[k].bit_length())
            print("P={:<4} S={:<5} slot={:<6} bpw={:<3} {:>12} stride={:<6} words={:<8} "
                  "slots={:<5} max_input_digit_bits={:<3} bad={}{}".format(
                      P, S, slot, bpw, label, st, nwords, 2 * P - 1, inmax.bit_length(), bad,
                      "" if first is None else
                      "  first_bad: slot={} got={} want={} (true bitlen {})".format(*first)))


if __name__ == "__main__":
    main()
