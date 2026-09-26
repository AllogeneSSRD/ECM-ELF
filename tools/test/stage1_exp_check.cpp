/* ---------------------------------------------------------------------------
 * stage1_exp_check.cpp -- the shared stage-1 exponent builder, verified.
 *
 * 1. EXACTNESS against the old implementation: the quadratic loop that used to
 *    live in mont_build_s() (multiply each prime power into a growing
 *    accumulator) is reproduced here verbatim and compared bit for bit with
 *    ecm_build_lcm_exponent() for a sweep of B1 values and both torsions.
 * 2. EXACTNESS independently of any implementation: for every prime p <= B1,
 *    p^floor(log_p B1) must divide s, and p^(that+1) must NOT, and the quotient
 *    after removing all of them must be exactly the torsion factor.
 * 3. TIMING: what the startup cost was and is, for B1 = 1e6 / 1e7 / 1.1e8
 *    (the user's queue task used 1e7; their GPU runs use 1.1e8).
 *
 * build: tools\build_tool.bat tools\test\stage1_exp_check.cpp src\core\ecm_stage1_exp.cpp
 * run:   build_vs18\tools\stage1_exp_check.exe
 * ------------------------------------------------------------------------- */
#include <gmp.h>
#include <stdio.h>
#include <stdlib.h>

#include <chrono>
#include <vector>

#include "ecm_stage1_exp.h"
#include "ecm_mont_cpu.h"     /* mont_expand_bits */

static int g_fail = 0, g_pass = 0;

static void check(bool ok, const char *what)
{
    printf("  [%s] %s\n", ok ? "PASS" : "FAIL", what);
    if (ok) g_pass++; else g_fail++;
}

/* the pre-2026-09-24 implementation, kept here as the reference */
static size_t old_build_s(mpz_t s, uint64_t B1, uint64_t torsion)
{
    std::vector<char> composite(B1 + 1, 0);
    mpz_set_ui(s, torsion ? torsion : 1);
    for (uint64_t p = 2; p <= B1; ++p) {
        if (composite[p]) continue;
        for (uint64_t q = p * 2; q <= B1; q += p) composite[q] = 1;
        uint64_t v = p;
        while (v <= B1 / p) v *= p;
        mpz_mul_ui(s, s, (unsigned long)v);
    }
    return (size_t)mpz_sizeinbase(s, 2);
}

/* true when r^j > B1, in saturating 64-bit arithmetic */
static bool ipow_over(uint64_t r, unsigned j, uint64_t B1)
{
    uint64_t p = 1;
    for (unsigned t = 0; t < j; t++) {
        if (r == 0 || p > B1 / r) return true;
        p *= r;
    }
    return false;
}

/* p-adic valuation of s at p */
static uint64_t valuation(const mpz_t s, uint64_t p)
{
    mpz_t t;
    mpz_init_set(t, s);
    uint64_t e = 0;
    while (mpz_divisible_ui_p(t, (unsigned long)(p & 0xFFFFFFFFu)) && e < 1000000u) {
        mpz_divexact_ui(t, t, (unsigned long)(p & 0xFFFFFFFFu));
        e++;
    }
    mpz_clear(t);
    return e;
}

int main(void)
{
    printf("stage1_exp_check: s = torsion * lcm(1..B1)\n\n");

    printf("1. identical to the old quadratic builder (bit for bit)\n");
    {
        /* small B1 values exercise the carry chain in the binary counter, plus a
           couple of larger ones that a wrong merge order could break */
        const uint64_t bounds[] = {0, 1, 2, 3, 4, 10, 97, 1000, 4096, 65536, 100000, 1000003};
        const uint64_t tors[] = {1, 12};
        bool all_ok = true;
        for (unsigned bi = 0; bi < sizeof(bounds) / sizeof(bounds[0]) && all_ok; bi++) {
            for (unsigned ti = 0; ti < 2; ti++) {
                mpz_t a, b;
                mpz_inits(a, b, NULL);
                const size_t ob = old_build_s(b, bounds[bi], tors[ti]);
                ecm_build_lcm_exponent(a, bounds[bi], tors[ti]);
                const size_t nb = (size_t)mpz_sizeinbase(a, 2);
                if (mpz_cmp(a, b) != 0 || ob != nb) {
                    printf("      mismatch at B1=%llu torsion=%llu (%zu vs %zu bits)\n",
                           (unsigned long long)bounds[bi], (unsigned long long)tors[ti], nb, ob);
                    all_ok = false;
                }
                mpz_clears(a, b, NULL);
            }
        }
        check(all_ok, "12 B1 values x 2 torsions agree with the reference");
    }

    printf("\n2. independent p-adic check (divisibility, not implementation)\n");
    {
        const uint64_t B1 = 20000;
        mpz_t s;
        mpz_init(s);
        ecm_build_lcm_exponent(s, B1, 12);
        bool ok = true;
        for (uint64_t p = 2; p <= B1 && ok; p++) {
            bool prime = true;
            for (uint64_t d = 2; d * d <= p; d++) if (p % d == 0) { prime = false; break; }
            if (!prime) continue;
            /* exponent of p in s = v_p(lcm(1..B1)) + v_p(torsion), where
               v_p(lcm(1..B1)) = floor(log_p B1) */
            uint64_t want = 1, q = p;
            while (q <= B1 / p) { q *= p; want++; }
            for (uint64_t t = 12; t % p == 0; t /= p) want++;
            const uint64_t got = valuation(s, p);
            if (got != want) {
                printf("      p=%llu: v_p(s)=%llu, expected %llu\n",
                       (unsigned long long)p, (unsigned long long)got,
                       (unsigned long long)want);
                ok = false;
            }
        }
        /* the remainder of s after dividing out every prime power must be 1: s must
           have no other prime factor (the torsion factor 12 = 2^2*3 is covered by
           the primes <= B1, so for torsion 12 the remainder is 1 as well) */
        mpz_t rem;
        mpz_init_set(rem, s);
        for (uint64_t p = 2; p <= B1; p++) {
            bool prime = true;
            for (uint64_t d = 2; d * d <= p; d++) if (p % d == 0) { prime = false; break; }
            if (!prime) continue;
            uint64_t e = valuation(s, p);
            while (e--) mpz_divexact_ui(rem, rem, (unsigned long)p);
        }
        check(ok, "every prime p <= 20000 has v_p(s) = floor(log_p B1) + v_p(12)");
        check(mpz_cmp_ui(rem, 1) == 0, "dividing out every prime power leaves exactly 1");
        mpz_clear(rem);
        mpz_clear(s);
    }

    printf("\n2b. cross-check with a completely different algorithm (primorials)\n");
    {
        /* lcm(1..B1) = prod_j primorial(floor(B1^(1/j))) -- a prime p <= B1 occurs
           once per j with p^j <= B1.  GMP computes each primorial with its own
           product tree, so agreeing here is an independent confirmation (and it
           was measured to be only ~8% faster, which is why it was not adopted). */
        const uint64_t bounds[] = {2, 1000, 65536, 1000003, 10000000};
        bool all_ok = true;
        for (unsigned i = 0; i < sizeof(bounds) / sizeof(bounds[0]); i++) {
            mpz_t ours, prim, tot;
            mpz_inits(ours, prim, tot, NULL);
            ecm_build_lcm_exponent(ours, bounds[i], 1);
            mpz_set_ui(tot, 1);
            for (unsigned j = 1; j < 64; j++) {
                uint64_t r = 1;
                while (!ipow_over(r + 1, j, bounds[i])) r++;
                if (r < 2) break;
                mpz_primorial_ui(prim, (unsigned long)r);
                if (mpz_cmp_ui(prim, 1) == 0) break;
                mpz_mul(tot, tot, prim);
            }
            if (mpz_cmp(tot, ours) != 0) {
                printf("      mismatch at B1=%llu\n", (unsigned long long)bounds[i]);
                all_ok = false;
            }
            mpz_clears(ours, prim, tot, NULL);
        }
        check(all_ok, "5 B1 values equal the primorial-chain product");
    }

    printf("\n3. startup cost (the number the user measured as ~30 s)\n");
    {
        const uint64_t bounds[] = {1000000ull, 10000000ull, 110000000ull};
        for (unsigned i = 0; i < 3; i++) {
            const uint64_t B1 = bounds[i];
            mpz_t s;
            mpz_init(s);
            const auto t0 = std::chrono::steady_clock::now();
            const bool ok = ecm_build_lcm_exponent(s, B1, 1);
            const double dt = std::chrono::duration<double>(
                                  std::chrono::steady_clock::now() - t0).count();
            printf("  B1 = %-10llu  %7.3f s   s_bits = %zu   %s\n",
                   (unsigned long long)B1, dt, (size_t)mpz_sizeinbase(s, 2),
                   ok ? "" : "(FAILED)");
            if (!ok) g_fail++;
            mpz_clear(s);
        }
        printf("  (for reference: the old loop measured ~29 s at B1=1e7)\n");

        /* The ladder does not read the mpz: it reads a byte-per-bit array, so the
           whole fixed startup cost is build + expand.  Both are measured here. */
        mpz_t s;
        mpz_init(s);
        ecm_build_lcm_exponent(s, 10000000ull, 1);
        size_t nb = 0;
        const auto t1 = std::chrono::steady_clock::now();
        uint8_t *bits = mont_expand_bits(s, &nb);
        const double dt1 = std::chrono::duration<double>(
                               std::chrono::steady_clock::now() - t1).count();
        printf("  mont_expand_bits(1e7)  %6.3f s   %zu bits\n", dt1, nb);
        free(bits);
        mpz_clear(s);
    }

    printf("\n4. where the remaining large-B1 cost sits (sieve vs product tree)\n");
    {
        /* Same sieve as the builder, timed on its own, so the split between
           "mark the primes" and "multiply them together" is visible. */
        const uint64_t B1 = 110000000ull;
        const auto t0 = std::chrono::steady_clock::now();
        std::vector<char> sieve((size_t)B1 + 1u, 1);
        sieve[0] = sieve[1] = 0;
        for (uint64_t p = 2; p * p <= B1; ++p) {
            if (!sieve[(size_t)p]) continue;
            for (uint64_t q = p * p; q <= B1; q += p) sieve[(size_t)q] = 0;
        }
        const double t_sieve = std::chrono::duration<double>(
                                   std::chrono::steady_clock::now() - t0).count();
        uint64_t primes = 0;
        for (size_t i = 2; i <= (size_t)B1; i++) if (sieve[i]) primes++;
        printf("  B1 = 1.1e8: sieve %.3f s (%llu primes, %zu MB), "
               "product tree = total - sieve\n",
               t_sieve, (unsigned long long)primes, ((size_t)B1 + 1u) / (1024 * 1024));
        printf("  (before this fix the same B1 needed minutes: 620k primes x ~1e5 limbs "
               "at B1=1e7 already cost 29 s)\n");
    }

    printf("\n%s: %d passed, %d failed\n", g_fail ? "FAILURE" : "ALL OK", g_pass, g_fail);
    return g_fail ? 1 : 0;
}
