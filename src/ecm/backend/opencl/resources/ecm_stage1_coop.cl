// 4096-bit cooperative work-group supplement.
#if ECM_STAGE1_USE_COOP_WG

#ifndef ECM_STAGE1_MUL_PATH
#define ECM_STAGE1_MUL_PATH 0
#endif
#ifndef ECM_STAGE1_SQR_PATH
#define ECM_STAGE1_SQR_PATH 0
#endif
#define MONT_FIXED_4096_LIMBS MAX_LIMBS

#if ECM_STAGE1_USE_COOP_WG
static inline void mp_add_mod_fused_unroll_b32_4096_local(__local uint *r, __local const uint *a,
                                                          __local const uint *b,
                                                          __local const uint *N) {
    ulong carry_add = 0ul;
    ulong carry_sub = 1ul;
    #pragma unroll
    for (uint blk = 0u; blk < 4u; ++blk) {
        uint off = blk * 32u;
        #pragma unroll 32
        for (uint j = 0u; j < 32u; ++j) {
            uint i = off + j;
            ulong sum = (ulong)a[i] + (ulong)b[i] + carry_add;
            carry_add = sum >> 32;
            ulong temp = (ulong)(uint)sum + (ulong)(~N[i]) + carry_sub;
            carry_sub = temp >> 32;
            r[i] = (uint)temp;
        }
    }
    if ((carry_add | carry_sub) != 0ul) {
        return;
    }
    ulong c = 0ul;
    #pragma unroll 32
    for (uint i = 0u; i < 128u; ++i) {
        ulong s = (ulong)r[i] + (ulong)N[i] + c;
        r[i] = (uint)s;
        c = s >> 32;
    }
}

static inline void mp_add_mod_local(__local uint *r, __local const uint *a, __local const uint *b,
                                    __local const uint *N, uint limbs) {
    mp_add_mod_fused_unroll_b32_4096_local(r, a, b, N);
    (void)limbs;
}

static inline uint mp_add_n_local(__local uint *r, __local const uint *a, __local const uint *b,
                                  uint limbs) {
    ulong carry = 0ul;
    for (uint i = 0u; i < limbs; ++i) {
        ulong sum = (ulong)a[i] + (ulong)b[i] + carry;
        r[i] = (uint)sum;
        carry = sum >> 32;
    }
    return (uint)carry;
}

static inline void mp_sub_n_local(__local uint *r, __local const uint *a, __local const uint *N,
                                  uint limbs) {
    ulong borrow = 0ul;
    for (uint i = 0u; i < limbs; ++i) {
        ulong av = (ulong)a[i];
        ulong nv = (ulong)N[i];
        ulong w = av - nv - borrow;
        r[i] = (uint)w;
        borrow = (av < nv + borrow) ? 1ul : 0ul;
    }
}

static inline int mp_sub_mod_local(__local uint *r, __local const uint *a, __local const uint *b,
                                   __local const uint *N, uint limbs) {
    ulong borrow = 0ul;
    for (uint i = 0u; i < limbs; ++i) {
        ulong av = (ulong)a[i];
        ulong bv = (ulong)b[i];
        ulong w = av - bv - borrow;
        r[i] = (uint)w;
        borrow = (av < bv + borrow) ? 1ul : 0ul;
    }
    if (borrow) {
        ulong c = 0ul;
        for (uint i = 0u; i < limbs; ++i) {
            ulong s = (ulong)r[i] + (ulong)N[i] + c;
            r[i] = (uint)s;
            c = s >> 32;
        }
        return 1;
    }
    return 0;
}

static inline void mp_copy_local(__local uint *dst, __local const uint *src, uint limbs) {
    for (uint i = 0u; i < limbs; ++i) {
        dst[i] = src[i];
    }
}

static inline int mp_ge_local(__local const uint *a, __local const uint *N, uint limbs) {
    for (int i = (int)limbs - 1; i >= 0; --i) {
        if (a[(uint)i] > N[(uint)i]) return 1;
        if (a[(uint)i] < N[(uint)i]) return 0;
    }
    return 1;
}

static inline void mont_normalize_local(__local uint *r, __local const uint *N, uint limbs) {
    if (mp_ge_local(r, N, limbs)) {
        mp_sub_n_local(r, r, N, limbs);
    }
}

static inline void maybe_mont_normalize_local(__local uint *r, __local const uint *N, uint limbs) {
#if ECM_STAGE1_FORCE_NORMALIZE
    mont_normalize_local(r, N, limbs);
#else
    (void)r;
    (void)N;
    (void)limbs;
#endif
}

static inline uint mul_ui32_limbs_local(__local uint *r, uint m, uint limbs) {
    ulong carry = 0ul;
    for (uint i = 0u; i < limbs; ++i) {
        ulong prod = (ulong)r[i] * (ulong)m + carry;
        r[i] = (uint)prod;
        carry = prod >> 32;
    }
    return (uint)carry;
}

static inline void shift_right_32_limbs_local(__local uint *r, uint limbs) {
    for (uint i = 0u; i + 1u < limbs; ++i) {
        r[i] = r[i + 1u];
    }
    r[limbs - 1u] = 0u;
}

static inline void special_mult_ui32_local(__local uint *r, uint m, __local const uint *N,
                                           uint np0, uint limbs) {
    uint carry_t1 = mul_ui32_limbs_local(r, m, limbs);
    uint t1_0 = r[0];
    uint q = (uint)((ulong)t1_0 * (ulong)np0);
    uint temp[MAX_LIMBS];
    for (uint i = 0u; i < limbs; ++i) {
        temp[i] = N[i];
    }
    uint carry_t2 = mul_ui32_limbs(temp, q, limbs);
    shift_right_32_limbs_local(r, limbs);
    shift_right_32_limbs(temp, limbs);
    r[limbs - 1u] = carry_t1;
    temp[limbs - 1u] = carry_t2;
    {
        uint carry = 0u;
        for (uint i = 0u; i < limbs; ++i) {
            ulong sum = (ulong)r[i] + (ulong)temp[i] + (ulong)carry;
            r[i] = (uint)sum;
            carry = (uint)(sum >> 32);
        }
        int carry_q = (int)carry;
        if (t1_0 != 0u) {
            carry = 0u;
            for (uint i = 0u; i < limbs; ++i) {
                ulong sum = (ulong)r[i] + (ulong)carry;
                r[i] = (uint)sum;
                carry = (uint)(sum >> 32);
            }
            carry_q += (int)carry;
        }
        if (carry_q > 0) {
            mp_sub_n_local(r, r, N, limbs);
        }
        if (mp_ge_local(r, N, limbs)) {
            mp_sub_n_local(r, r, N, limbs);
        }
    }
}

static inline void mp_shift_left_1_mod_local(__local uint *r, __local const uint *a,
                                             __local const uint *N, uint limbs) {
    uint carry = 0u;
    for (uint i = 0u; i < limbs; ++i) {
        uint old = a[i];
        r[i] = (old << 1) | carry;
        carry = old >> 31;
    }
    if (carry || mp_ge_local(r, N, limbs)) {
        mp_sub_n_local(r, r, N, limbs);
    }
}
#endif

#if ECM_STAGE1_USE_COOP_WG
static inline void mont_mul_stage1_coop(
    uint *out, const uint *a, const uint *b, const uint *N, __local const uint *N_loc,
    uint np0, __local uint *op_a, __local uint *op_b, __local uint *op_out,
    __local uint *mont_scratch, uint lid)
{
#if ECM_STAGE1_MUL_PATH == 2
    if (lid == 0u) {
        mont_mul_fips_4096b(out, a, b, N, np0, MONT_FIXED_4096_LIMBS);
    }
#elif ECM_STAGE1_MUL_PATH == 3
    if (lid == 0u) {
        for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
            op_a[i] = a[i];
            op_b[i] = b[i];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    mont_mul_stage1_fips4096_mtn_local(op_out, op_a, op_b, N_loc, np0, mont_scratch, lid, 8u);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0u) {
        for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
            out[i] = op_out[i];
        }
    }
#elif ECM_STAGE1_MUL_PATH == 4
    if (lid == 0u) {
        for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
            op_a[i] = a[i];
            op_b[i] = b[i];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    mont_mul_stage1_fips4096_mtn_local(op_out, op_a, op_b, N_loc, np0, mont_scratch, lid, 16u);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0u) {
        for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
            out[i] = op_out[i];
        }
    }
#elif ECM_STAGE1_MUL_PATH == 5 || ECM_STAGE1_MUL_PATH == (5 + 63)
    mont_mul_stage1_karatsuba_2048_coop(out, a, b, N, mont_scratch, lid);
#else
    if (lid == 0u) {
        mont_mul_unroll_4096b(out, a, b, N, np0, MONT_FIXED_4096_LIMBS);
    }
#endif
    barrier(CLK_LOCAL_MEM_FENCE);
}

static inline void mont_sqr_stage1_coop(
    uint *out, const uint *a, const uint *N, __local const uint *N_loc, uint np0,
    __local uint *op_a, __local uint *op_b, __local uint *op_out, __local uint *mont_scratch,
    uint lid)
{
#if ECM_STAGE1_SQR_PATH == 2
    if (lid == 0u) {
        mont_mul_fips_4096b(out, a, a, N, np0, MONT_FIXED_4096_LIMBS);
    }
#elif ECM_STAGE1_SQR_PATH == 3
    if (lid == 0u) {
        for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
            op_a[i] = a[i];
            op_b[i] = a[i];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    mont_mul_stage1_fips4096_mtn_local(op_out, op_a, op_b, N_loc, np0, mont_scratch, lid, 8u);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0u) {
        for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
            out[i] = op_out[i];
        }
    }
#elif ECM_STAGE1_SQR_PATH == 4
    if (lid == 0u) {
        for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
            op_a[i] = a[i];
            op_b[i] = a[i];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    mont_mul_stage1_fips4096_mtn_local(op_out, op_a, op_b, N_loc, np0, mont_scratch, lid, 16u);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0u) {
        for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
            out[i] = op_out[i];
        }
    }
#elif ECM_STAGE1_SQR_PATH == 5
    mont_mul_stage1_karatsuba_2048_coop(out, a, a, N, mont_scratch, lid);
#else
    if (lid == 0u) {
        mont_mul_unroll_4096b(out, a, a, N, np0, MONT_FIXED_4096_LIMBS);
    }
#endif
    barrier(CLK_LOCAL_MEM_FENCE);
}
#endif

#endif // ECM_STAGE1_USE_COOP_WG