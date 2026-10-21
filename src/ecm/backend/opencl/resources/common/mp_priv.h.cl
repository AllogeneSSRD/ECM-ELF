// ECM stage1 multi-precision limb primitives.
// ---------------------------------------------------------------------------
// Private multi-limb helpers (one curve per work-item)
//
// Naming convention:
//   `mp_*` means "multi-precision integer primitive". These are the low-level
//   bignum building blocks used by higher-level curve operations.
//
// Why keep the `mp_` prefix:
// 1) Distinguish bignum operators from point/curve operators (double_add_v2, etc.).
// 2) Make call-sites read like arithmetic formulas over Z and Z/NZ.
// 3) Avoid ambiguity with OpenCL scalar/vector add/sub intrinsics.
// ---------------------------------------------------------------------------

static inline void mp_copy(uint *dst, const uint *src, uint limbs) {
    for (uint i = 0u; i < limbs; ++i) {
        dst[i] = src[i];
    }
}

static inline void mp_zero(uint *dst, uint limbs) {
    for (uint i = 0u; i < limbs; ++i) {
        dst[i] = 0u;
    }
}

static inline int mp_ge(const uint *a, const uint *N, uint limbs) {
    for (int i = (int)limbs - 1; i >= 0; --i) {
        if (a[(uint)i] > N[(uint)i]) return 1;
        if (a[(uint)i] < N[(uint)i]) return 0;
    }
    return 1;
}

static inline void mp_sub_n(uint *r, const uint *a, const uint *N, uint limbs) {
    ulong borrow = 0ul;
    for (uint i = 0u; i < limbs; ++i) {
        ulong av = (ulong)a[i];
        ulong nv = (ulong)N[i];
        ulong w = av - nv - borrow;
        r[i] = (uint)w;
        borrow = (av < nv + borrow) ? 1ul : 0ul;
    }
}

static inline uint mp_add_n(uint *r, const uint *a, const uint *b, uint limbs) {
    ulong carry = 0ul;
    for (uint i = 0u; i < limbs; ++i) {
        ulong sum = (ulong)a[i] + (ulong)b[i] + carry;
        r[i] = (uint)sum;
        carry = sum >> 32;
    }
    return (uint)carry;
}
