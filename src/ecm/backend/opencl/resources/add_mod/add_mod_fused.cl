// Generic modular add (scalar / pair-unroll fallback).
static inline void add_mod_fused_body(uint *r, const uint *a, const uint *b, const uint *N,
                                      uint limbs) {
    ulong carry_add = 0ul;
    ulong carry_sub = 1ul;
#if MP_ADD_MOD_FUSED_UNROLL == 2
    uint j = 0u;
    for (; j + 1u < limbs; j += 2u) {
        ulong sum0 = (ulong)a[j] + (ulong)b[j] + carry_add;
        carry_add = sum0 >> 32;
        ulong temp0 = (ulong)(uint)sum0 + (ulong)(~N[j]) + carry_sub;
        carry_sub = temp0 >> 32;
        r[j] = (uint)temp0;

        ulong sum1 = (ulong)a[j + 1u] + (ulong)b[j + 1u] + carry_add;
        carry_add = sum1 >> 32;
        ulong temp1 = (ulong)(uint)sum1 + (ulong)(~N[j + 1u]) + carry_sub;
        carry_sub = temp1 >> 32;
        r[j + 1u] = (uint)temp1;
    }
    if (limbs & 1u) {
        ulong sum = (ulong)a[j] + (ulong)b[j] + carry_add;
        carry_add = sum >> 32;
        ulong temp = (ulong)(uint)sum + (ulong)(~N[j]) + carry_sub;
        carry_sub = temp >> 32;
        r[j] = (uint)temp;
    }
#else
    for (uint i = 0u; i < limbs; ++i) {
        ulong sum = (ulong)a[i] + (ulong)b[i] + carry_add;
        carry_add = sum >> 32;
        ulong temp = (ulong)(uint)sum + (ulong)(~N[i]) + carry_sub;
        carry_sub = temp >> 32;
        r[i] = (uint)temp;
    }
#endif
    if ((carry_add | carry_sub) != 0ul) {
        return;
    }
    ulong c = 0ul;
    for (uint i = 0u; i < limbs; ++i) {
        ulong s = (ulong)r[i] + (ulong)N[i] + c;
        r[i] = (uint)s;
        c = s >> 32;
    }
}

static inline void add_mod_fused(uint *r, const uint *a, const uint *b, const uint *N, uint limbs) {
    add_mod_fused_body(r, a, b, N, limbs);
    (void)limbs;
}
