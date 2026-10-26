#pragma once
// ecm_edwards_mont.h — fixed-size Montgomery field arithmetic (mpn-based).
//
// 目标: <10000-bit 合数 N。用 GMP mpn 的 64-bit limb (mp_bits_per_limb=64),
// 160 limbs = 10240 bits, 足够覆盖 <10000 bit。
// 所有中间量栈上分配, 无逐次 mpz 堆分配; REDC 用 mpn_addmul_1 自实现。

#include <gmp.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// GMP 的 REDC 内核 (internal 接口) 分派
// ---------------------------------------------------------------------------
//
// mpn_redc_1 / mpn_redc_n / mpn_binvert 不在 gmp.h 里 (GMP 明确声明这些是
// internal、接口可变)，但每个我们使用的 GMP 构建都导出了它们 (vcpkg/zen3 x64 DLL
// 的 .def 与 Android 静态库)。GMP 6.3.0 自 2023 年起未再更新。
//
// 本机实测 (tools/bench/mont_redc_ab.c，zen3 GMP 6.3.0，mul+redc 总时间):
//   limbs   我们原来的二次循环   mpn_redc_1    mpn_redc_n
//      20        0.36 us          0.32 (1.10x)   0.38
//      47        1.59             1.47 (1.08x)   1.54
//      63        2.80             2.53 (1.11x)   2.60
//      94        5.64             5.19           4.60 (1.22x)
//     125        9.54             8.83           7.68 (1.24x)
//     154       13.83            13.12          10.67 (1.30x)
// 阈值照抄 GMP 自己的调度 (mpn/x86_64/gmp-mparam.h: REDC_2_TO_REDC_N_THRESHOLD=79)。
//
// 置 ECM_MONT_USE_GMP_REDC=0 可退回原来的二次实现 (自检/对照用)。
#ifndef ECM_MONT_USE_GMP_REDC
#define ECM_MONT_USE_GMP_REDC 1
#endif
#define ECM_MONT_REDC_N_THRESHOLD 79
#define ECM_MPN_redc_1       __MPN(redc_1)
#define ECM_MPN_redc_n       __MPN(redc_n)
#define ECM_MPN_binvert      __MPN(binvert)
#define ECM_MPN_binvert_itch __MPN(binvert_itch)
#if ECM_MONT_USE_GMP_REDC
// C linkage: the DLL exports these with plain C names.
#ifdef __cplusplus
extern "C" {
#endif
extern mp_limb_t ECM_MPN_redc_1(mp_ptr, mp_ptr, mp_srcptr, mp_size_t, mp_limb_t);
extern void      ECM_MPN_redc_n(mp_ptr, mp_ptr, mp_srcptr, mp_size_t, mp_srcptr);
extern void      ECM_MPN_binvert(mp_ptr, mp_srcptr, mp_size_t, mp_ptr);
extern mp_size_t ECM_MPN_binvert_itch(mp_size_t);
#ifdef __cplusplus
}
#endif
#endif

#ifndef ED_MONT_MAX_LIMBS
#define ED_MONT_MAX_LIMBS 160   // 160*64 = 10240 bits (覆盖 <10000 bit 目标)
#endif

typedef struct { mp_limb_t l[ED_MONT_MAX_LIMBS]; } mont_t;

typedef struct {
    mp_limb_t N[ED_MONT_MAX_LIMBS];   // 模数 limbs (nlimbs 有效, 其余 0)
    size_t nlimbs;                     // N 的实际 limb 数
    mp_limb_t nprime0;                 // -N^{-1} mod 2^64
#if ECM_MONT_USE_GMP_REDC
    mp_limb_t *ip;                     // N^{-1} mod B^nlimbs (供 mpn_redc_n)
    int use_redc_n;                    // nlimbs >= ECM_MONT_REDC_N_THRESHOLD
#endif
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
#if ECM_MONT_USE_GMP_REDC
    // n >= 79 时用 GMP 的次二次归约 mpn_redc_n, 需要 ip = N^{-1} mod B^n。
    ctx->ip = nullptr;
    ctx->use_redc_n = (nlimbs >= ECM_MONT_REDC_N_THRESHOLD);
    if (ctx->use_redc_n) {
        const mp_size_t n = (mp_size_t)nlimbs;
        const mp_size_t itch = ECM_MPN_binvert_itch(n);
        mp_limb_t *scratch = (mp_limb_t *)malloc((size_t)itch * sizeof(mp_limb_t));
        ctx->ip = (mp_limb_t *)malloc((size_t)n * sizeof(mp_limb_t));
        if (!scratch || !ctx->ip) {
            free(scratch);
            free(ctx->ip);
            ctx->ip = nullptr;
            ctx->use_redc_n = 0;
            return -1;
        }
        ECM_MPN_binvert(ctx->ip, ctx->N, n, scratch);
        free(scratch);
    }
#endif
    return 0;
}

static inline void mont_clear(mont_ctx_t *ctx) {
#if ECM_MONT_USE_GMP_REDC
    free(ctx->ip);
    ctx->ip = nullptr;
#endif
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

// REDC: t (2n limbs) -> r (n limbs), r = t * R^{-1} mod N.  Clobbers t.
static inline void mont_redc(mp_limb_t *r, mp_limb_t *t, const mont_ctx_t *ctx) {
    const size_t n = ctx->nlimbs;
#if ECM_MONT_USE_GMP_REDC
    if (ctx->use_redc_n) {
        // GMP 自己的调度在 n >= 79 时用次二次的 mpn_redc_n; 结果已规范化 (< N)。
        ECM_MPN_redc_n(r, t, ctx->N, (mp_size_t)n, ctx->ip);
        return;
    }
    {
        // mpn_redc_1 的返回值是结果**最高位的 limb**（GMP 里另一种用法是
        // MPN_INCR_U(rp, n+1, cy)），并**不是**「结果 ≥ N」的标志位。
        // 只按 cy != 0 去减 N 的话，REDC 结果落在 [N, 2N) 时就被原样留下了 ——
        // 值仍然 ≡ 正确值 (mod N)，但 limbb 表示不是规范的，而 mont_add/mont_sub
        // 都假定算子 < N，于是后续运算被污染（和 SIMD 侧 §15.9 的 bug 同一类）。
        // 实测: canary 在 bits=3001 (nlimbs=47, redc_1 路径) 上能抓到 ≥ N 的结果。
        const mp_limb_t cy = ECM_MPN_redc_1(r, t, ctx->N, (mp_size_t)n, ctx->nprime0);
        if (cy != 0 || mpn_cmp(r, ctx->N, n) >= 0) mpn_sub_n(r, r, ctx->N, n);
        return;
    }
#else
    for (size_t i = 0; i < n; i++) {
        const mp_limb_t m = t[i] * ctx->nprime0;      // low limb
        const mp_limb_t cy = mpn_addmul_1(t + i, ctx->N, n, m);
        if (cy) {
            mpn_add_1(t + i + n, t + i + n, n - i, cy);
        }
    }
    mpn_copyi(r, t + n, n);
    if (mpn_cmp(r, ctx->N, n) >= 0) {
        mpn_sub_n(r, r, ctx->N, n);
    }
#endif
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
