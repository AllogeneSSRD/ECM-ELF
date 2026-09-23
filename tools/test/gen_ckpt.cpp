// gen_ckpt.cpp — 生成一个中途 STAGE1 checkpoint (M991 @ bitnum=100000) 供驱动恢复测试.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <gmp.h>
#include "ecm_edwards_cpu.h"
#include "ecm_edwards_save.h"

static void compute_s(mpz_t s, unsigned long B1) {
    mpz_set_ui(s, 48);
    std::vector<char> sieve(B1 + 1, 1);
    sieve[0] = sieve[1] = 0;
    for (unsigned long p = 2; p * p <= B1; p++) {
        if (!sieve[p]) continue;
        for (unsigned long q = p * p; q <= B1; q += p) sieve[q] = 0;
    }
    for (unsigned long p = 2; p <= B1; p++) {
        if (!sieve[p]) continue;
        unsigned long v = p;
        while (v <= B1 / p) v *= p;
        mpz_mul_ui(s, s, v);
    }
}

struct GenCtx { const mpz_t *N; const mpz_t *s; uint64_t sigma; const char *path; bool done; };

static int gen_progress(void *p, const edwards_checkpoint_t *cur) {
    GenCtx *g = (GenCtx*)p;
    if (g->done) return 0;
    mpz_t d, Px, Py;
    mpz_inits(d, Px, Py, NULL);
    edwards_atkin_morain(d, Px, Py, g->sigma, *g->N);
    ecm_save_common cm;
    cm.k = 1.0; cm.b = 2; cm.n = 991; cm.c = -1; cm.curve = 1;
    cm.B1 = 1000000; cm.B2 = 0; cm.sigma = g->sigma;
    uint32_t expbuf = (uint32_t)mpz_sizeinbase(*g->s, 2);
    uint32_t dict = (uint32_t)1u << (edwards_get_naf_w() - 2);
    ecm_edwards_write_stage1(g->path, cm, 2, expbuf, cur->bitnum, dict,
                             Px, Py, cur->Rx, cur->Ry, cur->Rz);
    mpz_clears(d, Px, Py, NULL);
    printf("wrote STAGE1 checkpoint @ bitnum=%u\n", cur->bitnum);
    g->done = true;
    return 0;  // abort
}

int main(int argc, char **argv) {
    const char *path = (argc >= 2) ? argv[1] : "e0000991";
    mpz_t N, s;
    mpz_inits(N, s, NULL);
    mpz_ui_pow_ui(N, 2, 991); mpz_sub_ui(N, N, 1);
    compute_s(s, 1000000);
    const uint64_t sigma = 105413044550089ULL;
    GenCtx g;
    g.N = &N; g.s = &s; g.sigma = sigma; g.path = path; g.done = false;
    int rc = edwards_stage1_curve_progress(nullptr, nullptr, nullptr, N, sigma, s,
                                           100000, nullptr, gen_progress, &g);
    printf("rc=%d (expect 2=aborted)\n", rc);
    mpz_clears(N, s, NULL);
    return 0;
}
