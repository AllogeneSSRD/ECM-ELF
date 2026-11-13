#include "ecm/cpu/arithmetic/ifma/modular_ifma.h"
#include "ecm/cpu/arithmetic/scalar/modular_mpn.h"
#include <immintrin.h>
#include <cstdio>
#include <initializer_list>
#include <new>
#include <stdexcept>
#ifdef _MSC_VER
#include <intrin.h>
#endif

static bool has_ifma() {
#ifdef _MSC_VER
    int regs[4];
    __cpuidex(regs, 1, 0);
    if (!(regs[2] & (1 << 27)) || (_xgetbv(0) & 0xe6) != 0xe6) return false;
    __cpuidex(regs, 7, 0);
    constexpr int required = (1 << 16) | (1 << 17) | (1 << 21);
    return (regs[1] & required) == required;
#else
    return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq") &&
           __builtin_cpu_supports("avx512ifma");
#endif
}

static void scalar() {
    mpz_t m,a,b,expected,actual;
    mpz_inits(m,a,b,expected,actual,nullptr);
    gmp_randstate_t rng;
    gmp_randinit_default(rng);
    constexpr unsigned widths[] = {
        2, 61, 63, 64, 65, 127, 128, 129, 991, 1279, 2203, 4003, 4253, 4423, 5056, 8191, 9689, 10240
    };
    unsigned checks = 0;
    for (unsigned p : widths) {
        mpz_set_ui(m,0);
        mpz_setbit(m,p);
        mpz_sub_ui(m,m,1);
        mont_ctx_t ctx;
        if (mont_init(&ctx,m) || !ctx.mersenne) throw std::runtime_error("scalar fold init");
        for (unsigned i=0;i<200;++i) {
            mpz_urandomm(a,rng,m);
            mpz_urandomm(b,rng,m);
            if (i == 0) { mpz_sub_ui(a,m,1); mpz_set(b,a); }
            if (i == 1) { mpz_sub_ui(a,m,1); mpz_set_ui(b,2); }
            if (i == 2) { mpz_set_ui(a,0); }
            if (i == 3) { mpz_sub_ui(a,m,1); mpz_set_ui(b,1); }
            if (i == 4 && p > 2 && p % 2 == 0) {
                mpz_set_ui(a,3);
                mpz_divexact_ui(b,m,3); // All-ones folded product must encode zero.
            }
            if (i == 5 && p > 65) {
                mpz_sub_ui(a,m,1);
                mpz_set_ui(b,1); mpz_setbit(b,64); // End-around carry crosses a limb.
            }
            if (i == 6 && p > 65) {
                mpz_set_ui(a,0);
                mpz_set_ui(b,0); mpz_setbit(b,64); // Borrow correction crosses a limb.
            }
            mont_t x,y;
            mont_to(&x,a,&ctx); mont_to(&y,b,&ctx);
            mont_add(&x,&x,&y,&ctx);
            mont_from(actual,&x,&ctx);
            mpz_add(expected,a,b); mpz_mod(expected,expected,m);
            if (mpz_cmp(expected,actual)) throw std::runtime_error("scalar add");
            mont_to(&x,a,&ctx);
            mont_sub(&y,&x,&y,&ctx);
            mont_from(actual,&y,&ctx);
            mpz_sub(expected,a,b); mpz_mod(expected,expected,m);
            if (mpz_cmp(expected,actual)) throw std::runtime_error("scalar subtract");
            mont_to(&x,a,&ctx); mont_to(&y,b,&ctx);
            mont_mul(&x,&x,&y,&ctx);
            mont_from(actual,&x,&ctx);
            mpz_mul(expected,a,b); mpz_mod(expected,expected,m);
            if (mpz_cmp(expected,actual)) throw std::runtime_error("scalar multiply");
            mont_sqr(&y,&y,&ctx);
            mont_from(actual,&y,&ctx);
            mpz_mul(expected,b,b); mpz_mod(expected,expected,m);
            if (mpz_cmp(expected,actual)) throw std::runtime_error("scalar square");
            checks += 4;
        }
        mont_clear(&ctx);
    }
    mpz_clears(m,a,b,expected,actual,nullptr);
    gmp_randclear(rng);
    printf("PASS %u scalar modular operations at %zu widths\n",
           checks, sizeof(widths) / sizeof(*widths));
}

static void simd() {
    mpz_t m,a[8],b[8],expected,actual;
    mpz_inits(m,expected,actual,nullptr);
    for (unsigned k=0;k<8;++k) mpz_inits(a[k],b[k],nullptr);
    gmp_randstate_t rng;
    gmp_randinit_default(rng);
    for (unsigned p : {991u,4212u,4213u,4253u,8164u,8165u,8191u,16432u,16433u,
                       106496u,106497u,212993u,262139u}) {
        mpz_set_ui(m,0);
        mpz_setbit(m,p);
        mpz_sub_ui(m,m,1);
        ifma_ctx_t ctx;
        if (ifma_ctx_init_ex(&ctx,m,IFMA_FIELD_MERS)) throw std::runtime_error("IFMA fold init");
        const unsigned mul_depth = p >= 4213 && p <= 16432 ? 2 : 0;
        const unsigned sqr_depth = p >= 8165 && p <= 16432 ? 2 : 0;
        if (ctx.mul_depth != mul_depth || ctx.sqr_depth != sqr_depth)
            throw std::runtime_error("IFMA Karatsuba dispatch boundary");
        const auto bytes=ctx.n*IFMA_LANES*sizeof(uint64_t);
        auto* x=(uint64_t*)_mm_malloc(bytes,64);
        auto* y=(uint64_t*)_mm_malloc(bytes,64);
        auto* z=(uint64_t*)_mm_malloc(bytes,64);
        if (!x || !y || !z) throw std::bad_alloc();
        for (unsigned k=0;k<8;++k) {
            mpz_urandomm(a[k],rng,m); mpz_urandomm(b[k],rng,m);
            if (k==0) {
                mpz_set_ui(a[k],0);
                mpz_setbit(a[k],52*((p-1)/52));
                mpz_sub_ui(a[k],a[k],1); mpz_set(b[k],a[k]);
            }
            if (k==1) { mpz_sub_ui(a[k],m,1); mpz_set(b[k],a[k]); }
            if (k==2) { mpz_set_ui(a[k],0); mpz_set_ui(b[k],1); }
            if (k==3 && p%2==0) { mpz_set_ui(a[k],3); mpz_divexact_ui(b[k],m,3); }
            ifma_from_mpz_lane(x,k,a[k],&ctx); ifma_from_mpz_lane(y,k,b[k],&ctx);
        }
        for (bool square : {false,true}) {
            const auto words = bytes / sizeof(uint64_t);
            for (size_t i=0;i<words;++i) z[i]=x[i];
            if (square) ifma_mont_sqr(z,z,&ctx);
            else ifma_mont_mul(z,z,y,&ctx);
            for (unsigned k=0;k<8;++k) {
                ifma_to_mpz_lane(actual,z,k,&ctx);
                mpz_mul(expected,a[k],square?a[k]:b[k]); mpz_mod(expected,expected,m);
                if (mpz_cmp(expected,actual)) {
                    fprintf(stderr,"IFMA mismatch p=%u lane=%u square=%d\n",p,k,square);
                    throw std::runtime_error("IFMA modular product");
                }
            }
        }
        printf("PASS IFMA p=%u multiply and square, 8 lanes\n",p);
        _mm_free(x); _mm_free(y); _mm_free(z);
        ifma_ctx_clear(&ctx);
    }
    for (unsigned k=0;k<8;++k) mpz_clears(a[k],b[k],nullptr);
    mpz_clears(m,expected,actual,nullptr);
    gmp_randclear(rng);
}

int main() {
    try {
        scalar();
        if (has_ifma()) simd();
        else puts("SKIP wide IFMA checks (unsupported ISA or OS state)");
    }
    catch (const std::exception& e) { fprintf(stderr,"%s\n",e.what()); return 1; }
}
