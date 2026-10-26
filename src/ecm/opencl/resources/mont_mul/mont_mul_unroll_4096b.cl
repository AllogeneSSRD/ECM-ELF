// Stage1 Montgomery mul — 4096-bit unroll64.
static inline void mont_mul_unroll_4096b(uint *out, const uint *a, const uint *b,
                                                  const uint *N, uint np0, uint limbs) {
    uint t[128u + 2u];
    for (uint i = 0u; i < 130u; ++i) t[i] = 0u;
    uint B[128u];
    for (uint j = 0u; j < 128u; ++j) B[j] = b[j];
    for (uint i = 0u; i < 128u; ++i) {
        uint ai = a[i];
        ulong carry = 0ul;
        #pragma unroll 64
        for (uint j = 0u; j < 128u; ++j) {
            ulong uv = (ulong)t[j] + (ulong)ai * (ulong)B[j] + carry;
            t[j] = (uint)uv;
            carry = uv >> 32;
        }
        ulong top = (ulong)t[128u] + carry;
        t[128u] = (uint)top;
        t[129u] = (uint)(top >> 32);
        uint m = (uint)((ulong)t[0] * (ulong)np0);
        ulong uv0 = (ulong)t[0] + (ulong)m * (ulong)N[0];
        carry = uv0 >> 32;
        #pragma unroll 64
        for (uint j = 1u; j < 128u; ++j) {
            ulong uv = (ulong)t[j] + (ulong)m * (ulong)N[j] + carry;
            t[j - 1u] = (uint)uv;
            carry = uv >> 32;
        }
        top = (ulong)t[128u] + carry;
        t[127u] = (uint)top;
        top = (ulong)t[129u] + (top >> 32);
        t[128u] = (uint)top;
        t[129u] = (uint)(top >> 32);
    }
    ulong borrow = 0ul;
    uint D[128u];
    for (uint i = 0u; i < 128u; ++i) {
        ulong tv = (ulong)t[i], nv = (ulong)N[i];
        ulong w = tv - nv - borrow;
        D[i] = (uint)w;
        borrow = (tv < nv + borrow) ? 1ul : 0ul;
    }
    uint need_sub = (t[128u] != 0u || t[129u] != 0u) ? 1u : 0u;
    need_sub = (borrow == 0u) ? 1u : need_sub;
    uint mask = 0u - need_sub;
    for (uint i = 0u; i < 128u; ++i) out[i] = (D[i] & mask) | (t[i] & ~mask);
}

static inline void mont_sqr_unroll_4096b(uint *out, const uint *a, const uint *N, uint np0, uint limbs) {
    mont_mul_unroll_4096b(out, a, a, N, np0, limbs);
}
