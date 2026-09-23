// ecm_edwards_checkpoint_test.cpp — 分块可恢复标量乘单测.
// 1) 完整跑一条曲线 → 参考 Qx/Qz.
// 2) 分块跑, 在第一个 chunk 处中止 → 捕获 checkpoint.
// 3) 从 checkpoint 恢复 → 完成, 比对 Qx/Qz 是否与完整结果一致.
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <gmp.h>
#include "ecm_edwards_cpu.h"

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

struct abort_ctx {
    edwards_checkpoint_t ckpt;
    bool captured;
};

static int progress_abort(void *p, const edwards_checkpoint_t *cur) {
    abort_ctx *a = (abort_ctx *)p;
    if (!a->captured) {
        a->ckpt.bitnum = cur->bitnum;
        mpz_set(a->ckpt.Rx, cur->Rx);
        mpz_set(a->ckpt.Ry, cur->Ry);
        mpz_set(a->ckpt.Rz, cur->Rz);
        a->captured = true;
        return 0;   // abort
    }
    return 1;
}

int main() {
    mpz_t N, s, Qx, Qz, Qx2, Qz2;
    mpz_inits(N, s, Qx, Qz, Qx2, Qz2, NULL);
    mpz_ui_pow_ui(N, 2, 991);
    mpz_sub_ui(N, N, 1);
    compute_s(s, 1000000);
    const uint64_t sigma = 105413044550089ULL;

    // 1) 完整跑
    int rc = edwards_stage1_curve(nullptr, Qx, Qz, N, sigma, s);
    printf("full run: rc=%d\n", rc);

    // 2) 分块跑, 第一个 chunk 中止
    abort_ctx ac;
    ac.captured = false;
    edwards_checkpoint_init(&ac.ckpt);
    int rc2 = edwards_stage1_curve_progress(nullptr, nullptr, nullptr, N, sigma, s,
                                            100000, nullptr, progress_abort, &ac);
    printf("chunked abort: rc=%d captured=%d bitnum=%u\n", rc2, ac.captured, ac.ckpt.bitnum);

    // 3) 从 checkpoint 恢复
    int rc3 = edwards_stage1_curve_progress(nullptr, Qx2, Qz2, N, sigma, s,
                                            100000, &ac.ckpt, nullptr, nullptr);
    printf("resume: rc=%d\n", rc3);
    printf("Qx match: %s\n", mpz_cmp(Qx2, Qx) == 0 ? "YES" : "NO");
    printf("Qz match: %s\n", mpz_cmp(Qz2, Qz) == 0 ? "YES" : "NO");

    edwards_checkpoint_clear(&ac.ckpt);
    mpz_clears(N, s, Qx, Qz, Qx2, Qz2, NULL);
    return 0;
}
