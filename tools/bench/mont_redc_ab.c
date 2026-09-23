// mont_redc_ab.c 鈥?A/B: our quadratic REDC vs a sub-quadratic "SOS" Montgomery
// reduction, on the same GMP build, at the limb counts this project actually uses
// (3000-8000 bit => 47-125 x 64-bit limbs).
//
// Method A (status quo, mirrors ecm_edwards_mont.h:mont_redc):
//     t = a*b                      (mpn_mul_n, sub-quadratic)
//     for i in 0..n-1:  m = t[i]*np0; t += m*N << 64i      (n x mpn_addmul_1 => O(n^2))
//     r = t[n..2n-1]; if r >= N: r -= N
//
// Method B (SOS, separated operand scanning):
//     t  = a*b                        (mpn_mul_n)
//     m  = (t mod B^n) * Np mod B^n   (mpn_mullo_n)
//     mn = m*N                        (mpn_mul_n)
//     V  = t + mn  < 2*N*B^n  =>  r = V >> 64n  < 2N, one conditional subtract.
//   Both multiplications are sub-quadratic; mullo_n is cheap.
//
// Build (gcc):   gcc -O2 -I<gmp>/include mont_redc_ab.c -L<gmp>/lib -lgmp
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <gmp.h>

static double now_s(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static uint64_t g_state = 0x243F6A8885A308D3ULL;
static uint64_t xrand(void) {
    g_state ^= g_state << 13;
    g_state ^= g_state >> 7;
    g_state ^= g_state << 17;
    return g_state;
}

/* GMP keeps these out of gmp.h (internal interface) but the vcpkg DLL exports
   them; prototypes taken verbatim from gmp-impl.h of GMP 6.3.0.  The exported
   symbols carry GMP's __gmpn_ prefix, so declare them through __MPN(). */
#define ECMPN_redc_1     __MPN(redc_1)
#define ECMPN_redc_2     __MPN(redc_2)
#define ECMPN_redc_n     __MPN(redc_n)
#define ECMPN_binvert    __MPN(binvert)
#define ECMPN_binvert_itch __MPN(binvert_itch)
#define ECMPN_sbpi1_bdiv_r __MPN(sbpi1_bdiv_r)
#define ECMPN_mullo_n    __MPN(mullo_n)
extern mp_limb_t ECMPN_redc_1(mp_ptr, mp_ptr, mp_srcptr, mp_size_t, mp_limb_t);
extern mp_limb_t ECMPN_redc_2(mp_ptr, mp_ptr, mp_srcptr, mp_size_t, mp_srcptr);
extern void      ECMPN_redc_n(mp_ptr, mp_ptr, mp_srcptr, mp_size_t, mp_srcptr);
extern void      ECMPN_binvert(mp_ptr, mp_srcptr, mp_size_t, mp_ptr);
extern mp_size_t ECMPN_binvert_itch(mp_size_t);
extern mp_limb_t ECMPN_sbpi1_bdiv_r(mp_ptr, mp_size_t, mp_srcptr, mp_size_t, mp_limb_t);
extern void      ECMPN_mullo_n(mp_ptr, mp_srcptr, mp_srcptr, mp_size_t);

/* ---- variants under test ---- */

/* A: status-quo quadratic REDC (ecm_edwards_mont.h:mont_redc) */
static void redc_quadratic(mp_limb_t *r, mp_limb_t *t, const mp_limb_t *N,
                           mp_limb_t np0, size_t n) {
    for (size_t i = 0; i < n; i++) {
        const mp_limb_t m = t[i] * np0;
        const mp_limb_t cy = mpn_addmul_1(t + i, N, n, m);
        if (cy) mpn_add_1(t + i + n, t + i + n, n - i, cy);
    }
    mpn_copyi(r, t + n, n);
    if (mpn_cmp(r, N, n) >= 0) mpn_sub_n(r, r, N, n);
}

/* A1: GMP's MPN_REDC_1 -- assembly sbpi1_bdiv_r division kernel */
static void redc_bdiv(mp_limb_t *r, mp_limb_t *t, const mp_limb_t *N,
                      mp_limb_t np0, size_t n) {
    const mp_limb_t cy = ECMPN_sbpi1_bdiv_r(t, 2 * n, N, n, np0);
    if (cy != 0) mpn_sub_n(r, t + n, N, n);
    else         mpn_copyi(r, t + n, n);
}

/* A2: GMP's mpn_redc_1 (generic wrapper, same convention) */
static void redc_gmp1(mp_limb_t *r, mp_limb_t *t, const mp_limb_t *N,
                      mp_limb_t np0, size_t n) {
    const mp_limb_t cy = ECMPN_redc_1(r, t, N, n, np0);
    if (cy != 0) mpn_sub_n(r, r, N, n);
}

/* A3: GMP's mpn_redc_2 (needs 2-limb inverse) */
static void redc_gmp2(mp_limb_t *r, mp_limb_t *t, const mp_limb_t *N,
                      const mp_limb_t *mip2, size_t n) {
    const mp_limb_t cy = ECMPN_redc_2(r, t, N, n, mip2);
    if (cy != 0) mpn_sub_n(r, r, N, n);
}

/* B: GMP's sub-quadratic mpn_redc_n (n >= 79 per REDC_2_TO_REDC_N_THRESHOLD) */
static void redc_n(mp_limb_t *r, mp_limb_t *t, const mp_limb_t *N,
                   const mp_limb_t *ip, size_t n) {
    ECMPN_redc_n(r, t, N, n, ip);
}

/* B2: explicit SOS (two sub-quadratic multiplications) */
static void redc_sos(mp_limb_t *r, mp_limb_t *t, const mp_limb_t *N,
                     const mp_limb_t *Np, size_t n,
                     mp_limb_t *m, mp_limb_t *mn) {
    ECMPN_mullo_n(m, t, Np, n);
    mpn_mul_n(mn, m, N, n);
    const mp_limb_t hi = mpn_add_n(t, t, mn, 2 * n);
    mpn_copyi(r, t + n, n);
    if (hi) mpn_sub_n(r, r, N, n);
    else if (mpn_cmp(r, N, n) >= 0) mpn_sub_n(r, r, N, n);
}

int main(int argc, char **argv) {
    const size_t sizes[] = {20, 47, 63, 94, 125, 154};
    const int nsz = (int)(sizeof(sizes) / sizeof(sizes[0]));
    const long iters = (argc > 1) ? atol(argv[1]) : 20000;

    printf("iters per size = %ld, GMP %d.%d.%d mp_bits_per_limb=%d\n\n",
           iters, __GNU_MP_VERSION, __GNU_MP_VERSION_MINOR,
           __GNU_MP_VERSION_PATCHLEVEL, (int)GMP_NUMB_BITS);
    printf("%-6s %12s %12s %8s  %s\n", "limbs", "A_us", "B_us", "B/A", "check");
    fflush(stdout);

    for (int si = 0; si < nsz; si++) {
        const size_t n = sizes[si];
        mp_limb_t *a  = calloc(n + 4, sizeof(mp_limb_t));
        mp_limb_t *b  = calloc(n + 4, sizeof(mp_limb_t));
        mp_limb_t *N  = calloc(n + 4, sizeof(mp_limb_t));
        mp_limb_t *Np = calloc(n + 4, sizeof(mp_limb_t));
        mp_limb_t *tA = calloc(2 * n + 8, sizeof(mp_limb_t));
        mp_limb_t *tB = calloc(2 * n + 8, sizeof(mp_limb_t));
        mp_limb_t *m  = calloc(2 * n + 8, sizeof(mp_limb_t));
        mp_limb_t *mn = calloc(2 * n + 8, sizeof(mp_limb_t));
        mp_limb_t *rA = calloc(n + 4, sizeof(mp_limb_t));
        mp_limb_t *rB = calloc(n + 4, sizeof(mp_limb_t));

        for (size_t i = 0; i < n; i++) { a[i] = xrand(); N[i] = xrand(); }
        N[n - 1] |= (mp_limb_t)1 << 63;      /* full-size modulus */
        N[0] |= 1u;                          /* odd -> Montgomery valid */
        for (size_t i = 0; i < n; i++) b[i] = xrand();
        a[n - 1] %= N[n - 1];
        b[n - 1] %= N[n - 1];

        /* Np = -N^-1 mod B^n */
        {
            mpz_t zN, zR, zInv;
            mpz_inits(zN, zR, zInv, NULL);
            mpz_import(zN, n, -1, sizeof(mp_limb_t), 0, 0, N);
            mpz_set_ui(zR, 1);
            mpz_mul_2exp(zR, zR, (unsigned long)(n * GMP_NUMB_BITS));
            mpz_invert(zInv, zN, zR);
            mpz_sub(zInv, zR, zInv);
            mpz_mod(zInv, zInv, zR);
            size_t cnt = 0;
            mpz_export(Np, &cnt, -1, sizeof(mp_limb_t), 0, 0, zInv);
            mpz_clears(zN, zR, zInv, NULL);
        }
        /* REDC wants -N^-1 mod B; Np already holds -N^-1 mod B^n, so Np[0] IS np0. */
        const mp_limb_t np0 = Np[0];

        /* ip = N^-1 mod B^n (for mpn_redc_n), and the 2-limb inverse (for redc_2) */
        mp_limb_t *ip  = calloc(n + 4, sizeof(mp_limb_t));
        mp_limb_t *mip2 = calloc(4, sizeof(mp_limb_t));
        {
            const mp_size_t itch = ECMPN_binvert_itch((mp_size_t)n);
            mp_limb_t *scr = calloc((size_t)itch + 8, sizeof(mp_limb_t));
            ECMPN_binvert(ip, N, (mp_size_t)n, scr);
            free(scr);
            mp_limb_t tm[4] = {N[0], n > 1 ? N[1] : 0, 0, 0};
            if (n >= 2) ECMPN_binvert(mip2, tm, 2, mip2 + 2);   /* scratch overlaps tail */
        }

        /* correctness against the mpz reference, for every variant */
        mpn_mul_n(tA, a, b, n);
        mpz_t za, zb, zN, zR, zr, zt, zgot;
        mpz_inits(za, zb, zN, zR, zr, zt, zgot, NULL);
        mpz_import(za, n, -1, sizeof(mp_limb_t), 0, 0, a);
        mpz_import(zb, n, -1, sizeof(mp_limb_t), 0, 0, b);
        mpz_import(zN, n, -1, sizeof(mp_limb_t), 0, 0, N);
        mpz_set_ui(zR, 1);
        mpz_mul_2exp(zR, zR, (unsigned long)(n * GMP_NUMB_BITS));
        mpz_invert(zt, zR, zN);
        mpz_mul(zr, za, zb);
        mpz_mod(zr, zr, zN);
        mpz_mul(zr, zr, zt);
        mpz_mod(zr, zr, zN);

        char chk[64] = "";
        struct { const char *name; void (*fn)(void); } dummy; (void)dummy;
        #define CHECK(tag, call)                                                  \
            do {                                                                  \
                memcpy(tB, tA, 2 * n * sizeof(mp_limb_t));                        \
                call;                                                             \
                mpz_import(zgot, n, -1, sizeof(mp_limb_t), 0, 0, rB);              \
                strncat(chk, mpz_cmp(zgot, zr) == 0 ? "ok " : "BAD ",             \
                        sizeof(chk) - strlen(chk) - 1);                           \
            } while (0)
        CHECK("quad", redc_quadratic(rB, tB, N, np0, n));
        CHECK("bdiv", redc_bdiv(rB, tB, N, np0, n));
        CHECK("g1",   redc_gmp1(rB, tB, N, np0, n));
        CHECK("g2",   redc_gmp2(rB, tB, N, mip2, n));
        CHECK("rn",   redc_n(rB, tB, N, ip, n));
        CHECK("sos",  redc_sos(rB, tB, N, Np, n, m, mn));
        #undef CHECK
        mpz_clears(za, zb, zN, zR, zr, zt, zgot, NULL);

        /* timing */
        #define TIME(expr, out)                                                   \
            do {                                                                  \
                double _t0 = now_s();                                             \
                for (long it = 0; it < iters; it++) {                             \
                    mpn_mul_n(tA, a, b, n);                                       \
                    tA[0] ^= (mp_limb_t)it;                                       \
                    expr;                                                         \
                }                                                                 \
                out = (now_s() - _t0) / iters * 1e6;                              \
            } while (0)
        double u_quad, u_bdiv, u_g1, u_g2, u_rn, u_sos;
        TIME(redc_quadratic(rA, tA, N, np0, n), u_quad);
        TIME(redc_bdiv(rA, tA, N, np0, n), u_bdiv);
        TIME(redc_gmp1(rA, tA, N, np0, n), u_g1);
        TIME(redc_gmp2(rA, tA, N, mip2, n), u_g2);
        TIME(redc_n(rA, tA, N, ip, n), u_rn);
        TIME(redc_sos(rA, tA, N, Np, n, m, mn), u_sos);
        #undef TIME

        printf("%-5zu %8.2f | %8.2f %8.2f %8.2f %8.2f %8.2f | %5.2f %5.2f %5.2f %5.2f\n",
               n, u_quad, u_bdiv, u_g1, u_g2, u_rn, u_sos,
               u_quad / u_bdiv, u_quad / u_g1, u_quad / u_g2, u_quad / u_rn);
        printf("      check (quad bdiv g1 g2 rn sos): %s\n", chk);
        fflush(stdout);

        free(a); free(b); free(N); free(Np); free(tA); free(tB); free(ip); free(mip2);
        free(m); free(mn); free(rA); free(rB);
    }
    return 0;
}
