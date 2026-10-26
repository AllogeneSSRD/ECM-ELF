#pragma once

#include <gmp.h>

#include <cstdint>

constexpr uint32_t ECM_LIMB24_MASK = 0xFFFFFFu;
constexpr uint32_t ECM_I24_384_LIMBS = 16u;
constexpr uint32_t ECM_I24_384_BITS = ECM_I24_384_LIMBS * 24u;
/** Stage-1 ladder headroom (must match cgbn_stage1_opencl.cpp CARRY_BITS). */
constexpr uint32_t ECM_STAGE1_CARRY_BITS = 6u;

uint32_t ecm_inv24_odd(uint32_t x);
void ecm_mpz_to_limb24(uint32_t *out, uint32_t limbs, const mpz_t z);
void ecm_limb24_from_mpz(uint32_t *out, uint32_t limbs, const mpz_t s);
void ecm_limb24_to_mpz(mpz_t r, const uint32_t *x, uint32_t limbs);
uint32_t ecm_find_np0_limb24(const uint32_t *n_limbs);
void ecm_to_montgomery_limb24(uint32_t *out, const mpz_t bn, const mpz_t N, uint32_t mont_bits,
                              uint32_t limbs);
void ecm_from_montgomery_limb24(mpz_t out, const mpz_t mont, const mpz_t N, uint32_t np0,
                                uint32_t limbs);

inline bool ecm_limb24_fits_n(size_t n_bit_size, uint32_t limbs) {
    return n_bit_size + ECM_STAGE1_CARRY_BITS <= static_cast<size_t>(limbs) * 24u;
}

inline uint32_t ecm_limb24_mont_bits(uint32_t limbs) {
    return limbs * 24u;
}

/** i24 stage-1 kernel/data limb count: ceil((N + carry) / 24), independent of 32-bit CGBN BITS. */
inline uint32_t ecm_limb24_stage1_limbs(size_t n_bit_size) {
    return static_cast<uint32_t>((n_bit_size + ECM_STAGE1_CARRY_BITS + 23u) / 24u);
}

/** Detect i24 checkpoint layout (BITS = limbs×24, not limbs×32). */
inline bool ecm_checkpoint_is_i24_layout(uint32_t bits, uint32_t limbs) {
    return ecm_limb24_mont_bits(limbs) == bits && limbs != bits / 32u;
}
