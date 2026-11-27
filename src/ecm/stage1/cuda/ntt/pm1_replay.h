#pragma once
#include "pm1.h"
#include "ecm/integer.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ecm_mersenne_ntt::pm1_detail {

enum Slot : unsigned { Current, BlockStart, Product, BlockEnd, Left, Three, Scratch, SlotCount };

inline uint64_t exponent_bits(const std::vector<uint32_t>& e) {
    if (e.empty() || !e.back()) throw std::invalid_argument("invalid P-1 exponent words");
    uint64_t bits = uint64_t(e.size() - 1) * 32;
    for (uint32_t top = e.back(); top; top >>= 1) ++bits;
    return bits;
}

// Sum one L-bit chunk without converting the whole exponent for each block.
inline void add_chunk(mpz_ptr sum, mpz_ptr scratch, const std::vector<uint32_t>& e,
                      uint64_t low, unsigned length) {
    const unsigned shift = unsigned(low % 32);
    mpz_import(scratch, (shift + length + 31) / 32, -1, 4, 0, 0, e.data() + low / 32);
    mpz_fdiv_q_2exp(scratch, scratch, shift);
    mpz_fdiv_r_2exp(scratch, scratch, length);
    mpz_add(sum, sum, scratch);
}

// For subblock endpoints u_j and chunks c_j:
// u_0 * P^(2^L) * 3^(sum c_j) = P * u_m, where P = product(u_0 ... u_(m-1)).
// 'step' maps Current to Current^2 or 3*Current^2; the remaining operations
// are infrequent copies, canonical modular products and checkpoint readbacks.
template<class Step, class Copy, class Mul, class Read, class Restore>
void replay(const std::vector<uint32_t>& exponent, Pm1Progress& saved,
            const std::function<bool(const Pm1Progress&)>& verified,
            Step step, Copy copy, Mul mul, Read read, Restore restore) {
    const uint64_t bits = exponent_bits(exponent);
    // Bound the delay to the next verified checkpoint even for long assignments.
    const auto full_length = std::clamp(unsigned(std::sqrt(double(bits) / 20)), 2u, 256u);
    unsigned length_limit = full_length;
    unsigned consecutive_errors = 0;
    ecm_edwards::Integer sum, chunk;
    while (saved.k < bits) {
        const uint64_t left = bits - saved.k;
        const unsigned length = unsigned(std::min<uint64_t>(left, length_limit));
        const unsigned chunks = unsigned(std::min<uint64_t>(left / length, length));
        copy(BlockStart, Current);
        copy(Product, Current);
        mpz_set_ui(sum.v, 0);
        uint64_t k = saved.k;
        for (unsigned j = 0; j < chunks; ++j) {
            if (j) mul(Product, Product, Current);
            add_chunk(sum.v, chunk.v, exponent, bits - k - length, length);
            for (unsigned i = 0; i < length; ++i) {
                const uint64_t at = bits - ++k;
                step((exponent[at / 32] >> (at % 32)) & 1u);
            }
        }
        copy(BlockEnd, Current);
        mul(Left, Product, BlockEnd);

        // Split the chunk sum at bit L. Start from P * 3^(sum >> L),
        // then consume exactly L low bits with the same square/optional-3 step.
        mpz_fdiv_q_2exp(chunk.v, sum.v, length);
        if (mpz_sgn(chunk.v)) {
            copy(Current, Three);
            for (size_t i = mpz_sizeinbase(chunk.v, 2) - 1; i > 0; --i)
                step(mpz_tstbit(chunk.v, i - 1));
            mul(Current, Current, Product);
        } else copy(Current, Product);
        for (unsigned i = length; i; --i) step(mpz_tstbit(sum.v, i - 1));
        mul(Current, Current, BlockStart);
        const auto check = read(Current);
        const bool nonzero = std::any_of(check.begin(), check.end(), [](uint32_t word){ return word != 0; });
        if (!nonzero || check != read(Left)) {
            ++saved.errors;
            if (++consecutive_errors >= 3) throw std::runtime_error("P-1 Gerbicz check repeatedly failed");
            length_limit = std::max(1u, length_limit / 2);
            restore(saved.residue);
            continue;
        }
        saved.residue = read(BlockEnd);
        saved.k = k;
        if (verified && !verified(saved)) break;
        copy(Current, BlockEnd);
        consecutive_errors = 0;
        length_limit = std::min(full_length, length_limit + 1);
    }
}
}
