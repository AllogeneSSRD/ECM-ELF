// Generic fused add-mod: full compile-time unroll for MAX_LIMBS.

static inline void add_mod_fused_unroll_body(uint *r, const uint *a, const uint *b, const uint *N) {
    ulong carry_add = 0ul;
    ulong carry_sub = 1ul;
    #pragma unroll ECM_ADDSUB_UNROLL_HINT
    for (uint i = 0u; i < MAX_LIMBS; ++i) {
        ulong sum = (ulong)a[i] + (ulong)b[i] + carry_add;
        carry_add = sum >> 32;
        ulong temp = (ulong)(uint)sum + (ulong)(~N[i]) + carry_sub;
        carry_sub = temp >> 32;
        r[i] = (uint)temp;
    }
    if ((carry_add | carry_sub) != 0ul) {
        return;
    }
    ulong c = 0ul;
    #pragma unroll ECM_ADDSUB_UNROLL_HINT
    for (uint i = 0u; i < MAX_LIMBS; ++i) {
        ulong s = (ulong)r[i] + (ulong)N[i] + c;
        r[i] = (uint)s;
        c = s >> 32;
    }
}

static inline void add_mod_fused_unroll(uint *r, const uint *a, const uint *b, const uint *N, uint limbs) {
    if (limbs == MAX_LIMBS) {
        add_mod_fused_unroll_body(r, a, b, N);
    }
    (void)limbs;
}
