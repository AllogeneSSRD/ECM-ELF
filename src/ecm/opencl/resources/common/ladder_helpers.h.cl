// Curve ladder helpers (independent of operator selection).
// 依赖 common/mp_priv.h.cl 中的 mp_ge / mp_sub_n 等基础 limb 原语。

static inline uint mul_ui32_limbs(uint *r, uint m, uint limbs) {
    ulong carry = 0ul;
    for (uint i = 0u; i < limbs; ++i) {
        ulong prod = (ulong)r[i] * (ulong)m + carry;
        r[i] = (uint)prod;
        carry = prod >> 32;
    }
    return (uint)carry;
}

static inline void shift_right_32_limbs(uint *r, uint limbs) {
    for (uint i = 0u; i + 1u < limbs; ++i) {
        r[i] = r[i + 1u];
    }
    r[limbs - 1u] = 0u;
}

static inline void mp_shift_left_1_mod(uint *r, const uint *a, const uint *N, uint limbs) {
    uint carry = 0u;
    for (uint i = 0u; i < limbs; ++i) {
        uint old = a[i];
        r[i] = (old << 1) | carry;
        carry = old >> 31;
    }
    if (carry || mp_ge(r, N, limbs)) {
        mp_sub_n(r, r, N, limbs);
    }
}

static inline void mont_normalize(uint *r, const uint *N, uint limbs) {
    if (mp_ge(r, N, limbs)) {
        mp_sub_n(r, r, N, limbs);
    }
}

static inline void maybe_mont_normalize(uint *r, const uint *N, uint limbs) {
#if ECM_STAGE1_FORCE_NORMALIZE
    mont_normalize(r, N, limbs);
#else
    (void)r;
    (void)N;
    (void)limbs;
#endif
}
