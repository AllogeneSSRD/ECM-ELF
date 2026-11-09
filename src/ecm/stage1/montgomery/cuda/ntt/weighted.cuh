#pragma once

#include "../../../../cuda/arithmetic/goldilocks/goldilocks.cuh"
#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace ecm_mersenne_ntt {

using WeightedWord = unsigned long long;
struct WeightedDigit {
    WeightedWord weight;
    unsigned bit, width;
};
struct WeightedLimb {
    unsigned first[3], end[3];
};
struct WeightedTables {
    std::vector<WeightedDigit> digits;
    std::vector<WeightedWord> inverse;
    std::vector<WeightedLimb> limbs;
};

inline WeightedTables weighted_tables(unsigned p, unsigned length) {
    using U = WeightedWord;
    // 2 has order 192 in Goldilocks. The generator of that subgroup is
    // 7^h, where h is divisible by every supported power-of-two length.
    constexpr U h = (GL_P - 1) / 192;
    const U generator = gl_pow_host(7, h);
    const U target = gl_pow_host(2, p);
    U r = 1, exponent = 0;
    for (; exponent < 192 && r != target; ++exponent)
        r = gl_mul_host(r, generator);
    if (exponent == 192 || h % length)
        throw std::logic_error("Goldilocks weight root unavailable");
    const U rho = gl_pow_host(7, (h / length) * exponent);
    const U rho_inv = gl_pow_host(rho, GL_P - 2);
    const auto at = [=](unsigned i) { return unsigned((U(p) * i + length - 1) / length); };

    WeightedTables out;
    out.digits.reserve(length);
    out.inverse.reserve(length);
    const unsigned small = p / length;
    const U forward_step = gl_mul_host(U(1) << small, rho_inv);
    const U inverse_step = gl_mul_host(gl_pow_host(2, GL_P - 1 - small), rho);
    const U forward_steps[] = {forward_step, gl_mul_host(forward_step, 2)};
    const U inverse_steps[] = {inverse_step, gl_mul_host(inverse_step, (GL_P + 1) / 2)};
    U w = 1, iw = 1;
    for (unsigned i = 0; i < length; ++i) {
        const unsigned bit = at(i), width = at(i + 1) - bit;
        out.digits.push_back({w, bit, width});
        out.inverse.push_back(iw);
        // Moving to the next digit multiplies the weight by 2^width / rho.
        w = gl_mul_host(w, forward_steps[width - small]);
        iw = gl_mul_host(iw, inverse_steps[width - small]);
    }

    const auto lower = [&](unsigned bit) {
        return unsigned(std::lower_bound(out.digits.begin(), out.digits.end(), bit,
            [](const WeightedDigit& digit, unsigned bound) { return digit.bit < bound; })
            - out.digits.begin());
    };
    out.limbs.resize((p + 31) / 32);
    for (unsigned i = 0; i < out.limbs.size(); ++i) {
        const unsigned bit = i * 32, width = std::min(32u, p - bit);
        for (unsigned wrap = 0; wrap < 3; ++wrap) {
            const unsigned lo = bit + wrap * p;
            // A coefficient has at most 64 bits. Include each coefficient
            // that overlaps this word, including wrap at p and at 2*p.
            out.limbs[i].first[wrap] = lower(lo < 63 ? 0 : lo - 63);
            out.limbs[i].end[wrap] = lower(lo + width);
        }
    }
    return out;
}

__global__ void unweight_products(WeightedWord* out, const WeightedWord* in,
                                 const WeightedWord* weights, unsigned length) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < length) {
        const size_t index = size_t(blockIdx.y) * length + i;
        out[index] = gl_mul(in[index], weights[i]);
    }
}

__global__ void weighted_limb_sums(WeightedWord* out, const WeightedWord* coefficients,
    const WeightedDigit* digits, const WeightedLimb* spans, unsigned p,
    unsigned words, unsigned length) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= words) return;
    const unsigned bit = i * 32, width = min(32u, p - bit);
    const auto mask = (WeightedWord(1) << width) - 1;
    const auto* input = coefficients + size_t(blockIdx.y) * length;
    const auto span = spans[i];
    WeightedWord sum = 0;
    for (unsigned wrap = 0; wrap < 3; ++wrap) {
        const unsigned lo = bit + wrap * p;
        for (unsigned j = span.first[wrap]; j < span.end[wrap]; ++j) {
            const auto start = digits[j].bit;
            const auto c = input[j];
            sum += (start <= lo ? c >> (lo - start) : c << (start - lo)) & mask;
        }
    }
    out[size_t(blockIdx.y) * words + i] = sum;
}

// Transfer every multi-bit carry once. The only narrow limb is the last;
// any extra carry from it is fed into limb zero before the one-bit scan.
__device__ inline WeightedWord weighted_limb(const WeightedWord* sums,
    unsigned words, unsigned tail_bits, unsigned i, uint32_t mask) {
    const unsigned last = words - 1;
    const WeightedWord tail_mask = (WeightedWord(1) << tail_bits) - 1;
    const WeightedWord tail = (sums[last] & tail_mask) +
        (last ? sums[last - 1] >> 32 : sums[last] >> tail_bits);
    const WeightedWord value = i == last ? tail & tail_mask :
        (sums[i] & mask) + (i ? sums[i - 1] >> 32 : sums[last] >> tail_bits);
    return value + (i == 0 ? tail >> tail_bits : 0);
}

} // namespace ecm_mersenne_ntt
