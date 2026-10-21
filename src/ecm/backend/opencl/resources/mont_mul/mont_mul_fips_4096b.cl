// ECM stage1-only 4096-bit FIPS Montgomery paths (included from host when ECM_STAGE1_HAS_FIPS4096=1)

#ifndef MONT_FIXED_4096_LIMBS
#define MONT_FIXED_4096_LIMBS 128u
#endif

#ifndef ECM_STAGE1_HAS_FIPS4096
#define ECM_STAGE1_HAS_FIPS4096 0
#endif

#if ECM_STAGE1_HAS_FIPS4096

#define FIPS4096_T_WORDS (2u * MONT_FIXED_4096_LIMBS + 2u)
#define FIPS4096_P_WORDS (MONT_FIXED_4096_LIMBS + 1u)
#define FIPS4096_MT_PROD_WORDS (2u * MONT_FIXED_4096_LIMBS)
#define ECM_STAGE1_FIPS4096_MT_LOCAL_U32 (MONT_FIXED_4096_LIMBS + MONT_FIXED_4096_LIMBS + FIPS4096_MT_PROD_WORDS + FIPS4096_P_WORDS)

static inline void fips512_csa_add_stage1(uint *t, uint *u, uint *v, ulong prod) {
    ulong val = (ulong)(*v) + prod;
    *v = (uint)val;
    val = (val >> 32) + (ulong)(*u);
    *u = (uint)val;
    val = (val >> 32) + (ulong)(*t);
    *t = (uint)val;
}

static inline void fips512_csa_shift_stage1(uint *t, uint *u, uint *v) {
    uint tv = *t;
    uint uv = *u;
    *v = uv;
    *u = tv;
    *t = 0u;
}

static inline void fips4096_finalize_p_stage1(uint *out, const uint *P, const uint *n) {
    uint ps = P[MONT_FIXED_4096_LIMBS];
    ulong borrow = 0ul;
    uint D[MONT_FIXED_4096_LIMBS];
    for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
        ulong tv = (ulong)P[i];
        ulong nv = (ulong)n[i];
        ulong w = tv - nv - borrow;
        D[i] = (uint)w;
        borrow = (tv < nv + borrow) ? 1ul : 0ul;
    }
    uint any_high = (ps != 0u) ? 1u : 0u;
    uint need_sub = any_high | ((borrow == 0u) ? 1u : 0u);
    uint mask = 0u - need_sub;
    for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
        out[i] = (D[i] & mask) | (P[i] & ~mask);
    }
}

static inline void fips4096_finalize_p_stage1_local(__local uint *out, __local uint *P,
                                                      __local const uint *n) {
    uint ps = P[MONT_FIXED_4096_LIMBS];
    ulong borrow = 0ul;
    uint D[MONT_FIXED_4096_LIMBS];
    for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
        ulong tv = (ulong)P[i];
        ulong nv = (ulong)n[i];
        ulong w = tv - nv - borrow;
        D[i] = (uint)w;
        borrow = (tv < nv + borrow) ? 1ul : 0ul;
    }
    uint any_high = (ps != 0u) ? 1u : 0u;
    uint need_sub = any_high | ((borrow == 0u) ? 1u : 0u);
    uint mask = 0u - need_sub;
    for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
        out[i] = (D[i] & mask) | (P[i] & ~mask);
    }
}

static inline void mont_mul_fips_4096b(uint *out, const uint *a, const uint *b, const uint *N,
                                            uint np0, uint limbs) {
    uint P[FIPS4096_P_WORDS];
    uint t = 0u;
    uint u = 0u;
    uint v = 0u;

    for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
        for (uint j = 0u; j < i; ++j) {
            fips512_csa_add_stage1(&t, &u, &v, (ulong)a[j] * (ulong)b[i - j]);
            fips512_csa_add_stage1(&t, &u, &v, (ulong)P[j] * (ulong)N[i - j]);
        }
        fips512_csa_add_stage1(&t, &u, &v, (ulong)a[i] * (ulong)b[0]);
        uint pi = (uint)((ulong)v * (ulong)np0);
        fips512_csa_add_stage1(&t, &u, &v, (ulong)pi * (ulong)N[0]);
        P[i] = pi;
        fips512_csa_shift_stage1(&t, &u, &v);
    }

    for (uint i = MONT_FIXED_4096_LIMBS; i < 2u * MONT_FIXED_4096_LIMBS; ++i) {
        for (uint j = i - MONT_FIXED_4096_LIMBS + 1u; j < MONT_FIXED_4096_LIMBS; ++j) {
            fips512_csa_add_stage1(&t, &u, &v, (ulong)a[j] * (ulong)b[i - j]);
            fips512_csa_add_stage1(&t, &u, &v, (ulong)P[j] * (ulong)N[i - j]);
        }
        P[i - MONT_FIXED_4096_LIMBS] = v;
        fips512_csa_shift_stage1(&t, &u, &v);
    }
    P[MONT_FIXED_4096_LIMBS] = v;

    fips4096_finalize_p_stage1(out, P, N);
    (void)np0;
}

static inline void mont_mul_stage1_fips4096_mtn_local(
    __local uint *out, __local const uint *a, __local const uint *b, __local const uint *N,
    uint np0, __local uint *local_mem, uint lid, uint mt)
{
    __local uint *A = local_mem;
    __local uint *B = A + MONT_FIXED_4096_LIMBS;
    __local ulong *prods = (__local ulong *)(B + MONT_FIXED_4096_LIMBS);
    __local uint *P = (__local uint *)(prods + FIPS4096_MT_PROD_WORDS);

    for (uint j = lid; j < MONT_FIXED_4096_LIMBS; j += mt) {
        A[j] = a[j];
        B[j] = b[j];
    }
    if (lid == 0u) {
        for (uint i = 0u; i < FIPS4096_P_WORDS; ++i) {
            P[i] = 0u;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint t = 0u;
    uint u = 0u;
    uint v = 0u;

    for (uint i = 0u; i < MONT_FIXED_4096_LIMBS; ++i) {
        for (uint j = lid; j < i; j += mt) {
            prods[2u * j] = (ulong)A[j] * (ulong)B[i - j];
            prods[2u * j + 1u] = (ulong)P[j] * (ulong)N[i - j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (lid == 0u) {
            for (uint j = 0u; j < i; ++j) {
                fips512_csa_add_stage1(&t, &u, &v, prods[2u * j]);
                fips512_csa_add_stage1(&t, &u, &v, prods[2u * j + 1u]);
            }
            fips512_csa_add_stage1(&t, &u, &v, (ulong)A[i] * (ulong)B[0]);
            uint pi = (uint)((ulong)v * (ulong)np0);
            fips512_csa_add_stage1(&t, &u, &v, (ulong)pi * (ulong)N[0]);
            P[i] = pi;
            fips512_csa_shift_stage1(&t, &u, &v);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (uint i = MONT_FIXED_4096_LIMBS; i < 2u * MONT_FIXED_4096_LIMBS; ++i) {
        for (uint j = i - MONT_FIXED_4096_LIMBS + 1u + lid; j < MONT_FIXED_4096_LIMBS; j += mt) {
            prods[2u * j] = (ulong)A[j] * (ulong)B[i - j];
            prods[2u * j + 1u] = (ulong)P[j] * (ulong)N[i - j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (lid == 0u) {
            for (uint j = i - MONT_FIXED_4096_LIMBS + 1u; j < MONT_FIXED_4096_LIMBS; ++j) {
                fips512_csa_add_stage1(&t, &u, &v, prods[2u * j]);
                fips512_csa_add_stage1(&t, &u, &v, prods[2u * j + 1u]);
            }
            P[i - MONT_FIXED_4096_LIMBS] = v;
            fips512_csa_shift_stage1(&t, &u, &v);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0u) {
        P[MONT_FIXED_4096_LIMBS] = v;
        fips4096_finalize_p_stage1_local(out, P, N);
    }
}

#endif

static inline void mont_sqr_fips_4096b(uint *out, const uint *a, const uint *N, uint np0, uint limbs) {
    mont_mul_fips_4096b(out, a, a, N, np0, limbs);
}
