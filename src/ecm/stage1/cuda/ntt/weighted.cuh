#pragma once

#include "../../../cuda/arithmetic/goldilocks/goldilocks.cuh"
#include <algorithm>
#include <cstdint>
#include <numeric>
#include <optional>
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

// For canonical p-bit inputs, digit widths are floor(p/L) or one bit larger.
// The exact weighted coefficient envelope is largest at output zero. There
// each short/long digit pair has factor 1 or 2 after removal of the weight.
inline std::optional<WeightedWord> weighted_unsigned_bound(unsigned p, unsigned length) {
    const unsigned period_count = std::gcd(p, length);
    const unsigned period = length / period_count;
    const unsigned wide = (p % length) / period_count;
    const unsigned has_wide = wide != 0;
    const auto step = WeightedWord{1} << (p / length);
    const auto narrow_max = step - 1;
    const unsigned wide_pairs = has_wide +
        (2 * wide > period + 1 ? 2 * (2 * wide - period - 1) : 0);

    // Expand (narrow_max + step * is_wide) for both digits. Each product
    // fits in one word (bpw <= 31); check multiplicities before multiplying.
    auto remaining = GL_P - 1;
    const auto fits = [&](WeightedWord count, WeightedWord value) {
        if (count && value > remaining / count) return false;
        remaining -= count * value;
        return true;
    };
    if (!fits(WeightedWord(period_count) * (2 * period - 1), narrow_max * narrow_max) ||
        !fits(WeightedWord(period_count) * (2 * wide - has_wide), 2 * narrow_max * step) ||
        !fits(WeightedWord(period_count) * wide_pairs, step * step))
        return std::nullopt;
    return GL_P - 1 - remaining;
}

inline WeightedTables weighted_tables(unsigned p, unsigned length, WeightedWord inverse_scale,
                                      unsigned coefficient_bits = 64) {
    using U = WeightedWord;
    // 2 has order 192 in Goldilocks. The generator of that subgroup is
    // 7^h, where h is divisible by every supported 2^k and 5*2^k length.
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
    U w = 1, iw = inverse_scale;
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
    const unsigned reach = coefficient_bits - 1;
    out.limbs.resize((p + 31) / 32);
    for (unsigned i = 0; i < out.limbs.size(); ++i) {
        const unsigned bit = i * 32, width = std::min(32u, p - bit);
        for (unsigned wrap = 0; wrap < 3; ++wrap) {
            const unsigned lo = bit + wrap * p;
            // Include each coefficient whose proved bit range overlaps this
            // word, including wrap at p and at 2*p.
            out.limbs[i].first[wrap] = lower(lo < reach ? 0 : lo - reach);
            out.limbs[i].end[wrap] = lower(lo + width);
        }
    }
    return out;
}

constexpr unsigned weighted_limb_threads = 128;

__device__ __forceinline__ WeightedWord weighted_limb_fragment(WeightedWord coefficient,
    unsigned start, unsigned bit, WeightedWord mask) {
    return (start <= bit ? coefficient >> (bit - start) : coefficient << (start - bit)) & mask;
}

inline unsigned weighted_limb_smem(const std::vector<WeightedLimb>& limbs) {
    unsigned capacity = 0;
    for (unsigned begin = 0; begin < limbs.size(); begin += weighted_limb_threads) {
        const auto& first = limbs[begin];
        const auto& last = limbs[std::min<size_t>(limbs.size(), begin + weighted_limb_threads) - 1];
        unsigned count = 0;
        for (unsigned wrap = 0; wrap < 3; ++wrap)
            count += last.end[wrap] - first.first[wrap];
        capacity = std::max(capacity, count);
    }
    return capacity * sizeof(WeightedWord);
}

static __global__ __launch_bounds__(weighted_limb_threads)
void weighted_limb_sums(WeightedWord* out, const WeightedWord* coefficients,
    const WeightedWord* inverse_weights, const WeightedDigit* digits, const WeightedLimb* spans,
    unsigned p, unsigned words, unsigned length) {
    const unsigned begin = blockIdx.x * weighted_limb_threads;
    const unsigned end = min(words, begin + weighted_limb_threads);
    const auto first = spans[begin];
    const auto last = spans[end - 1];
    const auto* input = coefficients + size_t(blockIdx.y) * length;
    extern __shared__ WeightedWord local[];
    unsigned offset = 0;
    for (unsigned wrap = 0; wrap < 3; ++wrap) {
        const unsigned lo = first.first[wrap], hi = last.end[wrap];
        for (unsigned j = lo + threadIdx.x; j < hi; j += weighted_limb_threads)
            local[offset + j - lo] = gl_mul(input[j], inverse_weights[j]);
        offset += hi - lo;
    }
    __syncthreads();

    const unsigned i = begin + threadIdx.x;
    if (i >= end) return;
    const unsigned bit = i * 32, width = min(32u, p - bit);
    const auto mask = (WeightedWord(1) << width) - 1;
    const auto span = spans[i];
    WeightedWord sum = 0;
    offset = 0;
    for (unsigned wrap = 0; wrap < 3; ++wrap) {
        const unsigned lo = bit + wrap * p;
        for (unsigned j = span.first[wrap]; j < span.end[wrap]; ++j) {
            const unsigned start = digits[j].bit;
            const auto c = local[offset + j - first.first[wrap]];
            sum += weighted_limb_fragment(c,start,lo,mask);
        }
        offset += last.end[wrap] - first.first[wrap];
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
