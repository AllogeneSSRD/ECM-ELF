// Stage1 special_mult generic fallback (R = 2^32).
// Noinline CIOS — single outer iteration because m is one 32-bit limb.
// Uses MAX_LIMBS (compile-time variable set by Host).
#ifdef __GNUC__
__attribute__((noinline))
#endif
static void special_mult_ui32_generic(uint *r, uint m, const uint *N, uint np0, uint limbs) {
    uint t[MAX_LIMBS + 1u];
    for (uint i = 0u; i <= limbs; ++i) t[i] = 0u;

    // 1) product: t = r * m
    {
        ulong carry = 0ul;
        for (uint j = 0u; j < limbs; ++j) {
            ulong uv = (ulong)t[j] + (ulong)m * (ulong)r[j] + carry;
            t[j] = (uint)uv;
            carry = uv >> 32;
        }
        t[limbs] = (uint)carry;
    }

    // 2) reduction: t = (t + q*N) >> 32  where q = t[0]*np0
    {
        uint mp = (uint)((ulong)t[0] * (ulong)np0);
        ulong carry = 0ul;
        for (uint j = 0u; j < limbs; ++j) {
            ulong uv = (ulong)t[j] + (ulong)mp * (ulong)N[j] + carry;
            if (j > 0u) t[j - 1u] = (uint)uv;
            carry = uv >> 32;
        }
        ulong top = (ulong)t[limbs] + carry;
        t[limbs - 1u] = (uint)top;
        t[limbs] = (uint)(top >> 32);
    }

    // 3) branchless conditional subtraction (same pattern as mont_mul_unroll_512b)
    {
        ulong borrow = 0ul;
        uint D[MAX_LIMBS];
        for (uint i = 0u; i < limbs; ++i) {
            ulong tv = (ulong)t[i];
            ulong nv = (ulong)N[i];
            ulong w = tv - nv - borrow;
            D[i] = (uint)w;
            borrow = (tv < nv + borrow) ? 1ul : 0ul;
        }
        uint need_sub = (t[limbs] != 0u) ? 1u : (borrow == 0ul ? 1u : 0u);
        uint mask = 0u - need_sub;
        for (uint i = 0u; i < limbs; ++i) {
            r[i] = (D[i] & mask) | (t[i] & ~mask);
        }
    }
}
