#pragma once
// ecm_edwards_mont.h — fixed-size Montgomery field arithmetic (mpn-based).
//
// 目标: <10000-bit 合数 N。用 GMP mpn 的 64-bit limb (mp_bits_per_limb=64),
// 160 limbs = 10240 bits, 足够覆盖 <10000 bit。
// 所有中间量栈上分配, 无逐次 mpz 堆分配; REDC 用 mpn_addmul_1 自实现。

#include <gmp.h>
#include <stdint.h>
#include <string.h>

#ifndef ED_MONT_MAX_LIMBS
#define ED_MONT_MAX_LIMBS 160   // 160*64 = 10240 bits (覆盖 <10000 bit 目标)
#endif

typedef struct { mp_limb_t l[ED_MONT_MAX_LIMBS]; } mont_t;

typedef struct {
    mp_limb_t N[ED_MONT_MAX_LIMBS];   // 模数 limbs (nlimbs 有效, 其余 0)
    size_t nlimbs;                     // N 的实际 limb 数
    mp_limb_t nprime0;                 // -N^{-1} mod 2^64
    mont_t one;                        // R mod N (Montgomery 域中的 1)
    mpz_t Nz;                          // N (mpz, 供转换用)
    mpz_t R;                           // 2^(nlimbs*64)
    mpz_t Rinv;                        // R^{-1} mod N (from_mont 用)
} mont_ctx_t;

// 初始化: 若 N 超过 ED_MONT_MAX_LIMBS 返回 -1, 否则 0。
static inline int mont_init(mont_ctx_t *ctx, const mpz_t N) {
    size_t bits = mpz_sizeinbase(N, 2);
    size_t nlimbs = (bits + GMP_NUMB_BITS - 1) / GMP_NUMB_BITS;
    if (nlimbs > ED_MONT_MAX_LIMBS) return -1;
    ctx->nlimbs = nlimbs;

    memset(ctx->N, 0, sizeof(ctx->N));
    size_t count = 0;
    mpz_export(ctx->N, &count, -1, sizeof(mp_limb_t), 0, 0, N);

    // nprime0 = -N0^{-1} mod 2^64
    {
        mpz_t n0, mod;
        mpz_init(n0);
        mpz_import(n0, 1, -1, sizeof(mp_limb_t), 0, 0, &ctx->N[0]);  // n0 = N 的低 limb (64-bit)
        mpz_init(mod);
        mpz_set_ui(mod, 0);
        mpz_setbit(mod, GMP_NUMB_BITS);
        mpz_invert(n0, n0, mod);
        ctx->nprime0 = (mp_limb_t)(0ULL - (unsigned long long)mpz_getlimbn(n0, 0));
        mpz_clear(n0);
        mpz_clear(mod);
    }

    // R = 2^(nlimbs*64), one = R mod N, Rinv = R^{-1} mod N
    mpz_init(ctx->Nz);
    mpz_set(ctx->Nz, N);
    mpz_init(ctx->R);
    mpz_set_ui(ctx->R, 1);
    mpz_mul_2exp(ctx->R, ctx->R, (unsigned long)(nlimbs * GMP_NUMB_BITS));
    {
        mpz_t t;
        mpz_init(t);
        mpz_mod(t, ctx->R, N);
        memset(ctx->one.l, 0, sizeof(ctx->one.l));
        count = 0;
        mpz_export(ctx->one.l, &count, -1, sizeof(mp_limb_t), 0, 0, t);
        mpz_clear(t);
    }
    mpz_init(ctx->Rinv);
    mpz_invert(ctx->Rinv, ctx->R, N);
    return 0;
}

static inline void mont_clear(mont_ctx_t *ctx) {
    mpz_clear(ctx->Nz);
    mpz_clear(ctx->R);
    mpz_clear(ctx->Rinv);
}

// 转换: r = a*R mod N (进 Montgomery 域)
static inline void mont_to(mont_t *r, const mpz_t a, const mont_ctx_t *ctx) {
    mpz_t t;
    mpz_init(t);
    mpz_mul(t, a, ctx->R);
    mpz_mod(t, t, ctx->Nz);
    memset(r->l, 0, sizeof(r->l));
    size_t count = 0;
    mpz_export(r->l, &count, -1, sizeof(mp_limb_t), 0, 0, t);
    mpz_clear(t);
}

// 转换: r = a*R^{-1} mod N (出 Montgomery 域)
static inline void mont_from(mpz_t r, const mont_t *a, const mont_ctx_t *ctx) {
    mpz_t t;
    mpz_init(t);
    mpz_import(t, ctx->nlimbs, -1, sizeof(mp_limb_t), 0, 0, a->l);
    mpz_mul(t, t, ctx->Rinv);
    mpz_mod(t, t, ctx->Nz);
    mpz_set(r, t);
    mpz_clear(t);
}

// REDC: t (2n limbs) -> r (n limbs), r = t * R^{-1} mod N
static inline void mont_redc(mp_limb_t *r, mp_limb_t *t, const mont_ctx_t *ctx) {
    const size_t n = ctx->nlimbs;
    const mp_limb_t *N = ctx->N;
    for (size_t i = 0; i < n; i++) {
        const mp_limb_t m = t[i] * ctx->nprime0;      // low limb
        const mp_limb_t cy = mpn_addmul_1(t + i, N, n, m);
        if (cy) {
            mpn_add_1(t + i + n, t + i + n, n - i, cy);
        }
    }
    mpn_copyi(r, t + n, n);
    if (mpn_cmp(r, N, n) >= 0) {
        mpn_sub_n(r, r, N, n);
    }
}

static inline void mont_mul(mont_t *r, const mont_t *a, const mont_t *b,
                            const mont_ctx_t *ctx) {
    mp_limb_t t[2 * ED_MONT_MAX_LIMBS];
    mpn_mul_n(t, a->l, b->l, ctx->nlimbs);
    mont_redc(r->l, t, ctx);
}

static inline void mont_sqr(mont_t *r, const mont_t *a, const mont_ctx_t *ctx) {
    mp_limb_t t[2 * ED_MONT_MAX_LIMBS];
    mpn_sqr(t, a->l, ctx->nlimbs);
    mont_redc(r->l, t, ctx);
}

static inline void mont_add(mont_t *r, const mont_t *a, const mont_t *b,
                            const mont_ctx_t *ctx) {
    const size_t n = ctx->nlimbs;
    mp_limb_t c = mpn_add_n(r->l, a->l, b->l, n);
    if (c || mpn_cmp(r->l, ctx->N, n) >= 0) {
        mpn_sub_n(r->l, r->l, ctx->N, n);
    }
}

static inline void mont_sub(mont_t *r, const mont_t *a, const mont_t *b,
                            const mont_ctx_t *ctx) {
    const size_t n = ctx->nlimbs;
    mp_limb_t b_ = mpn_sub_n(r->l, a->l, b->l, n);
    if (b_) {
        mpn_add_n(r->l, r->l, ctx->N, n);
    }
}

static inline void mont_neg(mont_t *r, const mont_t *a, const mont_ctx_t *ctx) {
    const size_t n = ctx->nlimbs;
    if (mpn_zero_p(a->l, n)) {
        mpn_zero(r->l, n);
    } else {
        mpn_sub_n(r->l, ctx->N, a->l, n);
    }
}

static inline void mont_set_ui(mont_t *r, unsigned long v, const mont_ctx_t *ctx) {
    const size_t n = ctx->nlimbs;
    if (v == 0) {
        mpn_zero(r->l, n);
    } else {
        mpn_copyi(r->l, ctx->one.l, n);   // 1 in Montgomery = R mod N
    }
}

static inline void mont_set(mont_t *r, const mont_t *a, const mont_ctx_t *ctx) {
    mpn_copyi(r->l, a->l, ctx->nlimbs);
}

static inline int mont_is_zero(const mont_t *a, const mont_ctx_t *ctx) {
    return mpn_zero_p(a->l, ctx->nlimbs);
}
