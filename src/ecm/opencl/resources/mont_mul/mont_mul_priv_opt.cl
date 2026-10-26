// Stage1 Montgomery mul — generic priv_opt.
// Generic Montgomery mul: cached B + speculative final subtract (mont_mul_priv_opt_core ABI).
static inline void mont_mul_priv_opt(uint *out, const uint *a, const uint *b,
                                            const uint *N, uint np0, uint limbs) {
    if (limbs == 0u || limbs > MAX_LIMBS) {
        return;
    }

    uint t[MAX_LIMBS + 2u];
    for (uint i = 0u; i < limbs + 2u; ++i) {
        t[i] = 0u;
    }

    uint B[MAX_LIMBS];
    for (uint j = 0u; j < limbs; ++j) {
        B[j] = b[j];
    }

    for (uint i = 0u; i < limbs; ++i) {
        uint ai = a[i];

        ulong carry = 0ul;
        for (uint j = 0u; j < limbs; ++j) {
            ulong uv = (ulong)t[j] + (ulong)ai * (ulong)B[j] + carry;
            t[j] = (uint)uv;
            carry = uv >> 32;
        }
        ulong top = (ulong)t[limbs] + carry;
        t[limbs] = (uint)top;
        t[limbs + 1u] = (uint)(top >> 32);

        uint m = (uint)((ulong)t[0] * (ulong)np0);
        carry = 0ul;
        for (uint j = 0u; j < limbs; ++j) {
            ulong uv = (ulong)t[j] + (ulong)m * (ulong)N[j] + carry;
            if (j > 0u) {
                t[j - 1u] = (uint)uv;
            }
            carry = uv >> 32;
        }
        top = (ulong)t[limbs] + carry;
        t[limbs - 1u] = (uint)top;
        top = (ulong)t[limbs + 1u] + (top >> 32);
        t[limbs] = (uint)top;
        t[limbs + 1u] = (uint)(top >> 32);
    }

    ulong borrow = 0ul;
    uint D[MAX_LIMBS];
    for (uint i = 0u; i < limbs; ++i) {
        ulong tv = (ulong)t[i];
        ulong nv = (ulong)N[i];
        ulong w = tv - nv - borrow;
        D[i] = (uint)w;
        borrow = (tv < nv + borrow) ? 1ul : 0ul;
    }

    uint need_sub = (t[limbs] != 0u || t[limbs + 1u] != 0u) ? 1u : 0u;
    need_sub = (borrow == 0u) ? 1u : need_sub;
    uint mask = 0u - need_sub;

    for (uint i = 0u; i < limbs; ++i) {
        out[i] = (D[i] & mask) | (t[i] & ~mask);
    }
}

static inline void mont_sqr_priv_opt(uint *out, const uint *a, const uint *N, uint np0, uint limbs) {
    mont_mul_priv_opt(out, a, a, N, np0, limbs);
}
