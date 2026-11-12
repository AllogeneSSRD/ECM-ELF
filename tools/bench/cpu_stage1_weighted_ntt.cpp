// Single-core probe for exact 5x2^k weighted cyclic multiplication modulo 2^p-1.
// Compile this diagnostic translation unit for AVX-512F, DQ and IFMA.
#include <gmp.h>
#include <gwnum.h>
#include <immintrin.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

#ifdef _MSC_VER
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <intrin.h>
#endif

namespace {
using Word = uint64_t;
using Vec = __m512i;
constexpr Word q = 1125899893734401ULL;
constexpr Word mask52 = (Word{1} << 52) - 1;
constexpr Word radix_mod_q = (Word{1} << 52) % q;

constexpr Word montgomery_inverse() {
    Word inv = 1;
    for (int i = 0; i < 6; ++i) inv *= 2 - q * inv;
    return (0 - inv) & mask52;
}

Word mul_mod(Word a, Word b) {
#ifdef _MSC_VER
    Word hi, remainder;
    const Word lo = _umul128(a, b, &hi);
    _udiv128(hi, lo, q, &remainder);
    return remainder;
#else
    return static_cast<Word>((__uint128_t(a) * b) % q);
#endif
}

Word power(Word x, Word n) {
    Word r = 1;
    for (; n; n >>= 1, x = mul_mod(x, x))
        if (n & 1) r = mul_mod(r, x);
    return r;
}

struct Multiplier {
    Word value;
    Word quotient;
    static Multiplier make(Word w) {
#ifdef _MSC_VER
        Word rem;
        return {w, _udiv128(w >> 12, w << 52, q, &rem)};
#else
        return {w, static_cast<Word>((__uint128_t(w) << 52) / q)};
#endif
    }
};

Vec splat(Word a) { return _mm512_set1_epi64(static_cast<int64_t>(a)); }
Vec load(const Word* a) { return _mm512_loadu_si512(a); }
void store(Word* a, Vec x) { _mm512_storeu_si512(a, x); }

Vec reduce(Vec a) {
    const Vec mod = splat(q);
    return _mm512_mask_sub_epi64(a, _mm512_cmp_epu64_mask(a, mod, _MM_CMPINT_NLT), a, mod);
}
Vec add(Vec a, Vec b) { return reduce(_mm512_add_epi64(a, b)); }
Vec sub(Vec a, Vec b) {
    return _mm512_mask_add_epi64(_mm512_sub_epi64(a, b),
                                _mm512_cmp_epu64_mask(a, b, _MM_CMPINT_LT),
                                _mm512_sub_epi64(a, b), splat(q));
}
Vec multiply_fixed(Vec a, Vec w, Vec precon) {
    const Vec hi = _mm512_madd52hi_epu64(_mm512_setzero_si512(), a, precon);
    const Vec lo = _mm512_madd52lo_epu64(_mm512_setzero_si512(), a, w);
    return reduce(_mm512_and_si512(_mm512_madd52lo_epu64(lo, hi, splat(0 - q)), splat(mask52)));
}
Vec multiply_fixed(Vec a, Multiplier m) {
    return multiply_fixed(a, splat(m.value), splat(m.quotient));
}
Vec multiply(Vec a, Vec b) {
    const Vec zero = _mm512_setzero_si512();
    const Vec lo = _mm512_and_si512(_mm512_madd52lo_epu64(zero, a, b), splat(mask52));
    const Vec hi = _mm512_madd52hi_epu64(zero, a, b);
    const Vec m = _mm512_madd52lo_epu64(zero, lo, splat(montgomery_inverse()));
    const Vec correction = _mm512_madd52lo_epu64(lo, m, splat(q));
    return reduce(_mm512_madd52hi_epu64(_mm512_add_epi64(hi, _mm512_srli_epi64(correction, 52)),
                                      m, splat(q)));
}

struct Stage {
    unsigned half = 0;
    std::vector<Word> weights, quotients;
    std::array<Word, 8> peer{};
    __mmask8 lower = 0;

    Stage(unsigned h, Word root) : half(h), weights(h < 8 ? 8 : h), quotients(weights.size()) {
        for (unsigned i = 0; i < weights.size(); ++i) {
            const auto m = Multiplier::make(power(root, i % h));
            weights[i] = m.value;
            quotients[i] = m.quotient;
        }
        if (h < 8) {
            for (unsigned i = 0; i < 8; ++i) {
                peer[i] = i ^ h;
                if (!(i & h)) lower |= __mmask8(1u << i);
            }
        }
    }

    template<bool inverse>
    void apply(Word* row, unsigned n) const {
        if (half < 8) {
            const Vec indices = load(peer.data());
            const Vec w = load(weights.data()), wp = load(quotients.data());
            for (unsigned i = 0; i < n; i += 8) {
                const Vec a = load(row + i);
                if constexpr (inverse) {
                    const Vec u = _mm512_mask_blend_epi64(lower, multiply_fixed(a, w, wp), a);
                    const Vec v = _mm512_permutexvar_epi64(indices, u);
                    store(row + i, _mm512_mask_blend_epi64(lower, sub(v, u), add(u, v)));
                } else {
                    const Vec b = _mm512_permutexvar_epi64(indices, a);
                    const Vec upper = multiply_fixed(sub(b, a), w, wp);
                    store(row + i, _mm512_mask_blend_epi64(lower, upper, add(a, b)));
                }
            }
            return;
        }
        for (unsigned base = 0; base < n; base += 2 * half) {
            for (unsigned j = 0; j < half; j += 8) {
                Word* x = row + base + j;
                Word* y = x + half;
                const Vec a = load(x), b = load(y);
                const Vec w = load(weights.data() + j), wp = load(quotients.data() + j);
                if constexpr (inverse) {
                    const Vec v = multiply_fixed(b, w, wp);
                    store(x, add(a, v));
                    store(y, sub(a, v));
                } else {
                    store(x, add(a, b));
                    store(y, multiply_fixed(sub(a, b), w, wp));
                }
            }
        }
    }
};

class WeightedPlan {
    unsigned p_, row_, n_;
    std::vector<Stage> forward_, inverse_;
    std::array<Multiplier, 4> root5_{}, inverse5_{};
    std::vector<unsigned> position_, offset_;
    std::vector<Word> digit_mask_, weights_, weight_precon_, unweights_, unweight_precon_;
    std::vector<Word> x_, y_, digit_;

    static std::array<Multiplier, 4> radix5_constants(Word root) {
        const Word r2 = mul_mod(root, root), r3 = mul_mod(r2, root), r4 = mul_mod(r3, root);
        const Word half = (q + 1) / 2;
        return {Multiplier::make(mul_mod((root + r4) % q, half)),
                Multiplier::make(mul_mod((r2 + r3) % q, half)),
                Multiplier::make(mul_mod((root + q - r4) % q, half)),
                Multiplier::make(mul_mod((r2 + q - r3) % q, half))};
    }

    void radix5(Word* x, const std::array<Multiplier, 4>& c) const {
        for (unsigned v = 0; v < row_; v += 8) {
            Word* a = x + v;
            Word* b = a + row_;
            Word* d = b + row_;
            Word* e = d + row_;
            Word* f = e + row_;
            const Vec x0 = load(a), x1 = load(b), x2 = load(d), x3 = load(e), x4 = load(f);
            const Vec ab = add(x1, x4), cd = add(x2, x3), ac = sub(x1, x4), bd = sub(x2, x3);
            const Vec z1 = add(x0, add(multiply_fixed(ab, c[0]), multiply_fixed(cd, c[1])));
            const Vec z2 = add(x0, add(multiply_fixed(ab, c[1]), multiply_fixed(cd, c[0])));
            const Vec s1 = add(multiply_fixed(ac, c[2]), multiply_fixed(bd, c[3]));
            const Vec s2 = sub(multiply_fixed(ac, c[3]), multiply_fixed(bd, c[2]));
            store(a, add(x0, add(ab, cd)));
            store(b, add(z1, s1));
            store(d, add(z2, s2));
            store(e, sub(z2, s2));
            store(f, sub(z1, s1));
        }
    }

    void encode(const std::vector<Word>& a, Word* x) const {
        for (unsigned i = 0; i < n_; ++i) {
            const unsigned j = position_[i];
            const unsigned limb = offset_[j] / 64, shift = offset_[j] % 64;
            Word d = a[limb] >> shift;
            if (shift) d |= a[limb + 1] << (64 - shift);
            x[i] = d & digit_mask_[j];
        }
        for (unsigned i = 0; i < n_; i += 8)
            store(x + i, multiply_fixed(load(x + i), load(weights_.data() + i), load(weight_precon_.data() + i)));
        for (unsigned u = 0; u < 5; ++u)
            for (const auto& stage : forward_) stage.apply<false>(x + u * row_, row_);
        radix5(x, root5_);
    }

    void decode(std::vector<Word>& out) {
        radix5(x_.data(), inverse5_);
        for (unsigned u = 0; u < 5; ++u)
            for (const auto& stage : inverse_) stage.apply<true>(x_.data() + u * row_, row_);
        for (unsigned i = 0; i < n_; i += 8) {
            std::array<Word, 8> words;
            store(words.data(), multiply_fixed(load(x_.data() + i), load(unweights_.data() + i),
                                               load(unweight_precon_.data() + i)));
            for (unsigned j = 0; j < words.size(); ++j) digit_[position_[i + j]] = words[j];
        }
        Word carry = 0;
        do {
            for (unsigned i = 0; i < n_; ++i) {
                const Word sum = digit_[i] + carry;
                digit_[i] = sum & digit_mask_[i];
                carry = sum >> (offset_[i + 1] - offset_[i]);
            }
        } while (carry);
        std::fill(out.begin(), out.end(), 0);
        if (digit_ == digit_mask_) return; // The all-ones representative is zero.
        for (unsigned i = 0; i < n_; ++i) {
            const unsigned limb = offset_[i] / 64, shift = offset_[i] % 64;
            out[limb] |= digit_[i] << shift;
            if (shift) out[limb + 1] |= digit_[i] >> (64 - shift);
        }
    }

public:
    explicit WeightedPlan(unsigned p, unsigned row)
        : p_(p), row_(row), n_(5 * row), position_(n_), offset_(n_ + 1), digit_mask_(n_),
          weights_(n_), weight_precon_(n_), unweights_(n_), unweight_precon_(n_),
          x_(n_), y_(n_), digit_(n_) {
        if (!((p == 6011 && row == 64) || (p == 12511 && row == 128)))
            throw std::invalid_argument("unverified p/length");
        for (unsigned j = 0; j <= n_; ++j) offset_[j] = (p_ * j + n_ - 1) / n_;
        for (unsigned j = 0; j < n_; ++j)
            digit_mask_[j] = (Word{1} << (offset_[j + 1] - offset_[j])) - 1;
        // Sum the actual unequal digit bounds. The coarse uniform-digit
        // bound rejects 640 for M12511 even though every coefficient fits.
        for (unsigned k = 0; k < n_; ++k) {
            Word bound = 0;
            for (unsigned i = 0; i < n_; ++i) {
                const unsigned wrap = i > k ? n_ : 0;
                const unsigned j = k + wrap - i;
                const unsigned shift = offset_[i] + offset_[j] - offset_[k] - (wrap ? p_ : 0);
                if (shift > 1) throw std::runtime_error("invalid weighted exponent");
                bound += (digit_mask_[i] * digit_mask_[j]) << shift;
            }
            if (bound >= q) throw std::runtime_error("coefficient bound exceeds the field");
        }
        const Word rho = power(row == 64 ? 5915116383046ULL : 523974791127ULL, p_);
        if (power(rho, n_) != power(2, p_)) throw std::runtime_error("invalid weight root");
        const Word root = power(6, (q - 1) / n_);
        if (power(root, n_) != 1 || power(root, n_ / 2) == 1 || power(root, n_ / 5) == 1)
            throw std::runtime_error("invalid transform root");
        const Word wrow = power(root, 5), w5 = power(root, row_);
        for (unsigned half = row_ / 2; half; half /= 2)
            forward_.emplace_back(half, power(wrow, row_ / (2 * half)));
        for (unsigned half = 1; half < row_; half *= 2)
            inverse_.emplace_back(half, power(power(wrow, q - 2), row_ / (2 * half)));
        root5_ = radix5_constants(w5);
        inverse5_ = radix5_constants(power(w5, q - 2));
        const Word inv_scale = power(mul_mod(n_, radix_mod_q), q - 2);
        const Word rho_inv = power(rho, q - 2);
        for (unsigned u = 0; u < 5; ++u) {
            for (unsigned v = 0; v < row_; ++v) {
                const unsigned i = u * row_ + v;
                // PFA input permutation. Keep the resulting spectrum order
                // through the pointwise product and into the inverse.
                const unsigned j = (u * row_ + 5 * v) % n_;
                position_[i] = j;
                const Word w = mul_mod(power(2, offset_[j]), power(rho_inv, j));
                const auto enc = Multiplier::make(mul_mod(w, radix_mod_q));
                const auto dec = Multiplier::make(mul_mod(power(w, q - 2), inv_scale));
                weights_[i] = enc.value;
                weight_precon_[i] = enc.quotient;
                unweights_[i] = dec.value;
                unweight_precon_[i] = dec.quotient;
            }
        }
    }

    void product(const std::vector<Word>& a, const std::vector<Word>& b, std::vector<Word>& out) {
        encode(a, x_.data());
        const Word* rhs = x_.data();
        if (&a != &b) {
            encode(b, y_.data());
            rhs = y_.data();
        }
        for (unsigned i = 0; i < n_; i += 8)
            store(x_.data() + i, multiply(load(x_.data() + i), load(rhs + i)));
        decode(out);
    }
};

class GwControl {
    gwhandle gw_{};
    gwnum a_ = nullptr, b_ = nullptr;
public:
    explicit GwControl(unsigned p) {
        gwinit(&gw_);
        gwset_num_threads(&gw_, 1);
        gwclear_use_benchmarks(&gw_);
        if (gwsetup(&gw_, 1.0, 2, p, -1) || !(a_ = gwalloc(&gw_)) || !(b_ = gwalloc(&gw_))) {
            gwdone(&gw_);
            throw std::runtime_error("GWNUM setup failed");
        }
    }
    ~GwControl() { gwdone(&gw_); }
    void reset(const std::vector<Word>& a, const std::vector<Word>& b) {
        binary64togw(&gw_, a.data(), a.size(), a_);
        binary64togw(&gw_, b.data(), b.size(), b_);
    }
    void step(bool square) {
        if (square) gwsquare2(&gw_, a_, a_, 0);
        else gwmul3(&gw_, a_, b_, a_, GWMUL_PRESERVE_S2);
    }
    bool equal(const std::vector<Word>& expected) {
        std::vector<Word> result(expected.size() + 4);
        if (gwtobinary64(&gw_, a_, result.data(), static_cast<uint32_t>(result.size())) < 0) return false;
        return std::equal(expected.begin(), expected.end(), result.begin());
    }
    unsigned length() const { return static_cast<unsigned>(gwfftlen(&gw_)); }
};

template<class Step> double chain(Step step, int repeats) {
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeats; ++i) step();
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() / repeats;
}

void probe(unsigned p, unsigned row) {
    WeightedPlan plan(p, row);
    const size_t words = (p + 63) / 64 + 1; // Zero sentinel for the last digit load.
    std::vector<Word> a(words), b(words), out(words);
    mpz_t mod, x, y, expected, actual;
    mpz_inits(mod, x, y, expected, actual, nullptr);
    mpz_setbit(mod, p);
    mpz_sub_ui(mod, mod, 1);
    std::mt19937_64 random(1);
    for (int i = 0; i < 128; ++i) {
        for (auto* operand : {&a, &b}) {
            for (size_t j = 0; j + 1 < words; ++j) (*operand)[j] = random();
            (*operand)[words - 2] &= (Word{1} << (p % 64)) - 1;
        }
        if (i < 3) {
            mpz_sub_ui(x, mod, static_cast<unsigned>(i + 1));
            mpz_export(a.data(), nullptr, -1, sizeof(Word), 0, 0, x);
            if (i == 0) b = a;
            if (i == 1) std::fill(b.begin(), b.end(), 0);
            if (i == 2) { std::fill(b.begin(), b.end(), 0); b[0] = 1; }
        }
        mpz_import(x, words, -1, sizeof(Word), 0, 0, a.data());
        mpz_import(y, words, -1, sizeof(Word), 0, 0, b.data());
        for (bool square : {false, true}) {
            plan.product(a, square ? a : b, out);
            mpz_mul(expected, x, square ? x : y);
            mpz_mod(expected, expected, mod);
            mpz_import(actual, words, -1, sizeof(Word), 0, 0, out.data());
            if (mpz_cmp(expected, actual)) throw std::runtime_error("GMP mismatch");
        }
    }
    GwControl control(p);
    const auto seed = a;
    for (bool square : {false, true}) {
        const auto ntt_step = [&] { plan.product(a, square ? a : b, out); out.swap(a); };
        const auto gw_step = [&] { control.step(square); };
        for (int sample = 0; sample < 3; ++sample) {
            constexpr int repeats = 30000;
            a = seed;
            control.reset(seed, b);
            chain(ntt_step, 100);
            chain(gw_step, 100);
            a = seed;
            control.reset(seed, b);
            double ntt, gw;
            if (sample == 1) {
                gw = chain(gw_step, repeats);
                ntt = chain(ntt_step, repeats);
            } else {
                ntt = chain(ntt_step, repeats);
                gw = chain(gw_step, repeats);
            }
            if (!control.equal(a)) throw std::runtime_error("GWNUM chain endpoint mismatch");
            std::printf("M%u %s sample=%d NTT L=%u %.3f us GWNUM L=%u %.3f us\n",
                        p, square ? "square" : "multiply", sample + 1, 5 * row, ntt, control.length(), gw);
        }
    }
    mpz_clears(mod, x, y, expected, actual, nullptr);
}
} // namespace

int main() {
    try {
#ifdef _MSC_VER
        int regs[4];
        __cpuidex(regs, 1, 0);
        if (!(regs[2] & (1 << 27)) || (_xgetbv(0) & 0xe6) != 0xe6)
            throw std::runtime_error("AVX-512 OS state is unavailable");
        __cpuidex(regs, 7, 0);
        constexpr int required = (1 << 16) | (1 << 17) | (1 << 21);
        if ((regs[1] & required) != required)
            throw std::runtime_error("AVX-512F/DQ/IFMA is required");
        if (!SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR(1) << 1))
            throw std::runtime_error("cannot pin to logical CPU 1");
#endif
        probe(6011, 64);
        probe(12511, 128);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
