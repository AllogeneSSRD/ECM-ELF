// Generic fused sub-mod: full compile-time unroll for MAX_LIMBS.

static inline int sub_mod_fused_unroll_body(uint *r, const uint *a, const uint *b, const uint *N) {
    ulong br = 0ul;
    #pragma unroll ECM_ADDSUB_UNROLL_HINT
    for (uint i = 0u; i < MAX_LIMBS; ++i) {
        ulong av = (ulong)a[i];
        ulong bv = (ulong)b[i];
        ulong w = av - bv - br;
        r[i] = (uint)w;
        br = (av < bv + br) ? 1ul : 0ul;
    }
    if (br != 0ul) {
        ulong c = 0ul;
        #pragma unroll ECM_ADDSUB_UNROLL_HINT
        for (uint i = 0u; i < MAX_LIMBS; ++i) {
            ulong s = (ulong)r[i] + (ulong)N[i] + c;
            r[i] = (uint)s;
            c = s >> 32;
        }
        return 1;
    }
    return 0;
}

static inline int sub_mod_fused_unroll(uint *r, const uint *a, const uint *b, const uint *N, uint limbs) {
    if (limbs == MAX_LIMBS) {
        return sub_mod_fused_unroll_body(r, a, b, N);
    }
    (void)limbs;
    return 0;
}
