#include "limb24.h"

#include "../../../../include/ecm/platform/logging.h"

#include <cstring>

uint32_t ecm_inv24_odd(uint32_t x) {
    uint64_t y = 1;
    for (int i = 0; i < 4; ++i) {
        y = y * (2ull - static_cast<uint64_t>(x & ECM_LIMB24_MASK) * y);
        y &= 0xFFFFFFull;
    }
    return static_cast<uint32_t>(y) & ECM_LIMB24_MASK;
}

void ecm_mpz_to_limb24(uint32_t *out, uint32_t limbs, const mpz_t z) {
    std::memset(out, 0, sizeof(uint32_t) * limbs);
    mpz_t t;
    mpz_init_set(t, z);
    for (uint32_t i = 0; i < limbs; ++i) {
        out[i] = static_cast<uint32_t>(mpz_get_ui(t)) & ECM_LIMB24_MASK;
        mpz_fdiv_q_2exp(t, t, 24);
        if (mpz_sgn(t) == 0) {
            break;
        }
    }
    mpz_clear(t);
}

void ecm_limb24_from_mpz(uint32_t *out, uint32_t limbs, const mpz_t s) {
    if (mpz_sizeinbase(s, 2) > static_cast<size_t>(limbs) * 24u) {
        ecm_ts_fprintf(stderr, "limb24: value does not fit in %u limbs\n", limbs);
        std::memset(out, 0, sizeof(uint32_t) * limbs);
        return;
    }
    ecm_mpz_to_limb24(out, limbs, s);
}

void ecm_limb24_to_mpz(mpz_t r, const uint32_t *x, uint32_t limbs) {
    mpz_set_ui(r, 0);
    mpz_t limb;
    mpz_init(limb);
    for (uint32_t i = 0u; i < limbs; ++i) {
        mpz_set_ui(limb, x[i] & ECM_LIMB24_MASK);
        mpz_mul_2exp(limb, limb, 24u * i);
        mpz_add(r, r, limb);
    }
    mpz_clear(limb);
}

uint32_t ecm_find_np0_limb24(const uint32_t *n_limbs) {
    return 0u - ecm_inv24_odd(n_limbs[0] | 1u);
}

void ecm_to_montgomery_limb24(uint32_t *out, const mpz_t bn, const mpz_t N, uint32_t mont_bits,
                              uint32_t limbs) {
    mpz_t t;
    mpz_init(t);
    mpz_mul_2exp(t, bn, mont_bits);
    mpz_fdiv_r(t, t, N);
    ecm_mpz_to_limb24(out, limbs, t);
    mpz_clear(t);
}

void ecm_from_montgomery_limb24(mpz_t out, const mpz_t mont, const mpz_t N, uint32_t np0,
                                uint32_t limbs) {
    (void)np0;
    const uint32_t mont_bits = limbs * 24u;
    mpz_t r_inv, prod;
    mpz_init(r_inv);
    mpz_init(prod);
    mpz_set_ui(r_inv, 2);
    mpz_pow_ui(r_inv, r_inv, mont_bits);
    if (!mpz_invert(r_inv, r_inv, N)) {
        ecm_ts_fprintf(stderr, "limb24: R^-1 mod N does not exist\n");
        mpz_set_ui(out, 0);
        mpz_clear(r_inv);
        mpz_clear(prod);
        return;
    }
    mpz_mul(prod, mont, r_inv);
    mpz_mod(prod, prod, N);
    mpz_set(out, prod);
    mpz_clear(r_inv);
    mpz_clear(prod);
}
