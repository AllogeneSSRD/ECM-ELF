/* helper_canary.cpp — can the field helpers (add/sub/neg) return a
 * non-canonical value (>= N)?  mpz_mod() in ifma_to_mpz_lane() hides that, but
 * the next multiplication then gets an operand >= N and produces garbage.
 */
#include "simd_edwards.h"
#include <gmp.h>
#include <stdio.h>
#include <vector>

static void add52(mpz_t out, uint64_t limb)
{
    mpz_mul_2exp(out, out, 26);
    mpz_add_ui(out, out, (unsigned long)((limb >> 26) & 0x3FFFFFFu));
    mpz_mul_2exp(out, out, 26);
    mpz_add_ui(out, out, (unsigned long)(limb & 0x3FFFFFFu));
}

static void raw_to_mpz(mpz_t out, const uint64_t *e, size_t n)
{
    mpz_set_ui(out, 0);
    for (size_t i = n; i-- > 0; ) add52(out, e[8 * i] & 0xFFFFFFFFFFFFFULL);
}

int main(int argc, char **argv)
{
    const size_t k = (argc > 1) ? (size_t)atoi(argv[1]) : 4003;
    const int mode = (argc > 2) ? atoi(argv[2]) : IFMA_FIELD_MONT;
    const int w = (argc > 3) ? atoi(argv[3]) : 12;
    const int trials = (argc > 4) ? atoi(argv[4]) : 300;

    mpz_t N;
    mpz_init(N);
    mpz_set_ui(N, 1); mpz_mul_2exp(N, N, (mp_bitcnt_t)k); mpz_sub_ui(N, N, 1);
    ed_soa_ctx_t c;
    if (ed_soa_init_ex(&c, N, w, mode) != 0) { printf("init failed\n"); return 2; }
    uint64_t sg[8];
    for (int i = 0; i < 8; i++) sg[i] = 20260922u + (uint64_t)i;
    ed_soa_set_curves(&c, sg, 8);
    const size_t n = c.n, lw = 8 * n;
    printf("%s n=%zu trials=%d\n", ed_soa_field_name(&c), n, trials);

    std::vector<uint64_t> A(lw), B(lw), R(lw), S(lw);
    mpz_t ra, rb, rr, want;
    mpz_inits(ra, rb, rr, want, NULL);
    gmp_randstate_t rs; gmp_randinit_mt(rs); gmp_randseed_ui(rs, 4242u);

    int bad_canon = 0, bad_val = 0;
    for (int t = 0; t < trials; t++) {
        mpz_t x, y; mpz_inits(x, y, NULL);
        mpz_urandomm(x, rs, N); mpz_urandomm(y, rs, N);
        for (unsigned lane = 0; lane < 8; lane++) {
            ifma_from_mpz_lane(A.data(), lane, x, &c.mc);
            ifma_from_mpz_lane(B.data(), lane, y, &c.mc);
        }
        raw_to_mpz(ra, A.data(), n);       /* domain value of A */
        raw_to_mpz(rb, B.data(), n);

        for (int op = 0; op < 4; op++) {
            if (op == 0) { ed_soa_debug_add_field(&c, R.data(), A.data(), B.data());
                           mpz_add(want, ra, rb); }
            else if (op == 1) { ed_soa_debug_sub_field(&c, R.data(), A.data(), B.data(), S.data());
                                mpz_sub(want, ra, rb); }
            else if (op == 2) { ed_soa_debug_neg_field(&c, R.data(), A.data());
                                mpz_neg(want, ra); }
            else { ed_soa_debug_add_field(&c, R.data(), A.data(), A.data());
                   mpz_add(want, ra, ra); }
            mpz_mod(want, want, N);
            raw_to_mpz(rr, R.data(), n);
            if (mpz_cmp(rr, N) >= 0) {
                bad_canon++;
                if (bad_canon <= 3)
                    gmp_printf("  op=%d NON-CANONICAL raw=%Zx\n    a=%Zx\n    b=%Zx\n", op, rr, ra, rb);
            }
            if (mpz_cmp(rr, want) != 0) {
                bad_val++;
                if (bad_val <= 3)
                    gmp_printf("  op=%d WRONG raw=%Zx want=%Zx\n", op, rr, want);
            }
        }
        mpz_clears(x, y, NULL);
    }
    printf("k=%zu mode=%s -> non-canonical=%d wrong=%d\n", k, mode ? "mersenne" : "montgomery",
           bad_canon, bad_val);
    mpz_clears(ra, rb, rr, want, NULL);
    ed_soa_clear(&c);
    mpz_clear(N);
    gmp_randclear(rs);
    return 0;
}
