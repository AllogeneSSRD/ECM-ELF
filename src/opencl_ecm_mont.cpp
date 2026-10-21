#include "opencl_ecm_mont.h"
#include "ecm/platform/logging.h"

#include <cstdlib>

void ecm_from_mpz(const mpz_t s, uint32_t *x, uint32_t count) {
    size_t words;
    if (mpz_sizeinbase(s, 2) > (size_t)count * 32) {
        ecm_ts_fprintf(stderr, "from_mpz: value does not fit in %u limbs\n", count);
        exit(EXIT_FAILURE);
    }
    mpz_export(x, &words, -1, sizeof(uint32_t), 0, 0, s);
    while (words < count) {
        x[words++] = 0;
    }
}

void ecm_to_mpz(mpz_t r, const uint32_t *x, uint32_t count) {
    mpz_import(r, count, -1, sizeof(uint32_t), 0, 0, x);
}

uint32_t ecm_find_np0(const mpz_t N) {
    mpz_t temp;
    mpz_init(temp);
    mpz_ui_pow_ui(temp, 2, 32);
    if (!mpz_invert(temp, N, temp)) {
        mpz_clear(temp);
        return 0;
    }
    uint32_t np0 = 0u - (uint32_t)mpz_get_ui(temp);
    mpz_clear(temp);
    return np0;
}

void ecm_to_montgomery(uint32_t *out, const mpz_t bn, const mpz_t N, uint32_t bits,
                       uint32_t limbs) {
    mpz_t t;
    mpz_init(t);
    mpz_mul_2exp(t, bn, bits);
    mpz_fdiv_r(t, t, N);
    ecm_from_mpz(t, out, limbs);
    mpz_clear(t);
}

void ecm_from_montgomery(mpz_t out, const mpz_t mont, const mpz_t N, uint32_t np0,
                         uint32_t limbs) {
    mpz_t prod, add;
    mpz_init(prod);
    mpz_init(add);
    mpz_set(prod, mont);
    for (uint32_t index = 0; index < limbs; index++) {
        uint32_t low = np0 * (uint32_t)mpz_get_ui(prod);
        mpz_mul_ui(add, N, low);
        mpz_add(prod, prod, add);
        mpz_fdiv_q_2exp(prod, prod, 32);
    }
    if (mpz_cmp(prod, N) < 0) {
        mpz_set(out, prod);
    } else {
        mpz_sub(out, prod, N);
    }
    mpz_clear(prod);
    mpz_clear(add);
}

void ecm_curves_to_montgomery(uint32_t *data, uint32_t curves, uint32_t limbs, const mpz_t N,
                              uint32_t bits) {
    const uint32_t stride = 5u * limbs;
    mpz_t t;
    mpz_init(t);
    for (uint32_t c = 0; c < curves; c++) {
        uint32_t *datum = data + c * stride;
        for (uint32_t slot = 1; slot <= 4; slot++) {
            ecm_to_mpz(t, datum + slot * limbs, limbs);
            ecm_to_montgomery(datum + slot * limbs, t, N, bits, limbs);
        }
    }
    mpz_clear(t);
}

void ecm_curves_from_montgomery(uint32_t *data, uint32_t curves, uint32_t limbs, const mpz_t N,
                                uint32_t np0) {
    const uint32_t stride = 5u * limbs;
    mpz_t t, mont;
    mpz_init(t);
    mpz_init(mont);
    for (uint32_t c = 0; c < curves; c++) {
        uint32_t *datum = data + c * stride;
        for (uint32_t slot = 1; slot <= 4; slot++) {
            ecm_to_mpz(mont, datum + slot * limbs, limbs);
            ecm_from_montgomery(t, mont, N, np0, limbs);
            ecm_from_mpz(t, datum + slot * limbs, limbs);
        }
    }
    mpz_clear(t);
    mpz_clear(mont);
}
