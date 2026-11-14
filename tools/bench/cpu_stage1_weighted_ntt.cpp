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
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>
#include "cpu_stage1_ntt_ifma_control.h"

#ifdef _MSC_VER
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <intrin.h>
#endif

namespace {
using Word = uint64_t;
using Clock = std::chrono::steady_clock;
using Digits = std::vector<Word>;
using Chains = std::vector<Digits>;
enum class Operation { Multiply, Square, Fixed };
enum Phase { Pack, Forward, Pointwise, Inverse, Unpack, Carry, PhaseCount };
using PhaseTimes = std::array<Clock::duration, PhaseCount>;

struct Shape { unsigned p, row; Word rho_base; };
constexpr Shape shapes[] = {
    {4001, 64, 5915116383046ULL}, {6011, 64, 5915116383046ULL},
    {8191, 128, 523974791127ULL}, {12511, 128, 523974791127ULL},
    {16001, 256, 479836011024ULL}, {32749, 512, 770574134522ULL},
    {65521, 1024, 188872533427ULL},
};
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

#include "cpu_stage1_ntt_kernels.h"

template<unsigned Lanes, bool Lazy, bool Fused>
class WeightedPlan {
    using O = cpu_ntt::Ops<Lanes>;
    using V = typename O::V;
    unsigned p_, row_, n_;
    std::vector<Stage> forward_, inverse_;
    std::array<Multiplier, 4> root5_{}, inverse5_{};
    std::vector<unsigned> position_, offset_;
    std::vector<Word> digit_mask_, weights_, weight_precon_, unweights_, unweight_precon_;
    std::vector<Word> x_, y_, digit_;

    template<bool Timed, class Fn>
    static void phase(PhaseTimes* times, Phase index, Fn fn) {
        if constexpr (Timed) {
            const auto start = Clock::now();
            fn();
            (*times)[index] += Clock::now() - start;
        } else {
            fn();
        }
    }

    static V mul_constant(V value, Multiplier m) {
        return O::fixed(value, O::splat(m.value), O::splat(m.quotient));
    }

    static std::array<Multiplier, 4> radix5_constants(Word root) {
        const Word r2 = mul_mod(root, root), r3 = mul_mod(r2, root), r4 = mul_mod(r3, root);
        const Word half = (q + 1) / 2;
        return {Multiplier::make(mul_mod((root + r4) % q, half)),
                Multiplier::make(mul_mod((r2 + r3) % q, half)),
                Multiplier::make(mul_mod((root + q - r4) % q, half)),
                Multiplier::make(mul_mod((r2 + q - r3) % q, half))};
    }

    void radix5(Word* x, const std::array<Multiplier, 4>& c) const {
        for (unsigned v = 0; v < row_; v += Lanes) {
            Word* a = x + v;
            Word* b = a + row_;
            Word* d = b + row_;
            Word* e = d + row_;
            Word* f = e + row_;
            const V x0 = O::load(a), x1 = O::load(b), x2 = O::load(d), x3 = O::load(e), x4 = O::load(f);
            const V ab = O::add(x1, x4), cd = O::add(x2, x3), ac = O::sub(x1, x4), bd = O::sub(x2, x3);
            const V z1 = O::add(x0, O::add(mul_constant(ab, c[0]), mul_constant(cd, c[1])));
            const V z2 = O::add(x0, O::add(mul_constant(ab, c[1]), mul_constant(cd, c[0])));
            const V s1 = O::add(mul_constant(ac, c[2]), mul_constant(bd, c[3]));
            const V s2 = O::sub(mul_constant(ac, c[3]), mul_constant(bd, c[2]));
            O::store(a, O::add(x0, O::add(ab, cd)));
            O::store(b, O::add(z1, s1));
            O::store(d, O::add(z2, s2));
            O::store(e, O::sub(z2, s2));
            O::store(f, O::sub(z1, s1));
        }
    }

    void pack(const Digits& a, Word* x) const {
        for (unsigned i = 0; i < n_; ++i) {
            const unsigned j = position_[i];
            const unsigned limb = offset_[j] / 64, shift = offset_[j] % 64;
            Word d = a[limb] >> shift;
            if (shift) d |= a[limb + 1] << (64 - shift);
            x[i] = d & digit_mask_[j];
        }
        for (unsigned i = 0; i < n_; i += Lanes)
            O::store(x + i, O::fixed(O::load(x + i), O::load(weights_.data() + i), O::load(weight_precon_.data() + i)));
    }

    void forward(Word* x) const {
        for (unsigned u = 0; u < 5; ++u) {
            auto* row = x + u * row_;
            if constexpr (Lanes == 8 && !Lazy && !Fused)
                for (const auto& stage : forward_) stage.apply<false>(row, row_);
            else
                cpu_ntt::transform_row<Lanes, false, Lazy, Fused>(row, row_, forward_);
        }
        radix5(x, root5_);
    }

    void inverse() {
        radix5(x_.data(), inverse5_);
        for (unsigned u = 0; u < 5; ++u) {
            auto* row = x_.data() + u * row_;
            if constexpr (Lanes == 8 && !Lazy && !Fused)
                for (const auto& stage : inverse_) stage.apply<true>(row, row_);
            else
                cpu_ntt::transform_row<Lanes, true, Lazy, Fused>(row, row_, inverse_);
        }
    }

    void unpack() {
        for (unsigned i = 0; i < n_; i += Lanes) {
            std::array<Word, Lanes> words;
            O::store(words.data(), O::fixed(O::load(x_.data() + i), O::load(unweights_.data() + i),
                                           O::load(unweight_precon_.data() + i)));
            for (unsigned j = 0; j < words.size(); ++j) digit_[position_[i + j]] = words[j];
        }
    }

    void carry(Digits& out) {
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
    explicit WeightedPlan(Shape shape, bool verify_bound = false)
        : p_(shape.p), row_(shape.row), n_(5 * row_), position_(n_), offset_(n_ + 1), digit_mask_(n_),
          weights_(n_), weight_precon_(n_), unweights_(n_), unweight_precon_(n_),
          x_(n_), y_(n_), digit_(n_) {
        for (unsigned j = 0; j <= n_; ++j) offset_[j] = (p_ * j + n_ - 1) / n_;
        for (unsigned j = 0; j < n_; ++j)
            digit_mask_[j] = (Word{1} << (offset_[j + 1] - offset_[j])) - 1;
        // Sum the actual unequal digit bounds. The coarse uniform-digit
        // bound rejects 640 for M12511 even though every coefficient fits.
        for (unsigned k = 0; verify_bound && k < n_; ++k) {
            Word bound = 0;
            for (unsigned i = 0; i < n_; ++i) {
                const unsigned wrap = i > k ? n_ : 0;
                const unsigned j = k + wrap - i;
                const unsigned shift = offset_[i] + offset_[j] - offset_[k] - (wrap ? p_ : 0);
                if (shift > 1) throw std::runtime_error("invalid weighted exponent");
                const Word term = (digit_mask_[i] * digit_mask_[j]) << shift;
                if (term >= q - bound) throw std::runtime_error("coefficient bound exceeds the field");
                bound += term;
            }
        }
        const Word rho = power(shape.rho_base, p_);
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

    Digits fixed_spectrum(const Digits& b) const {
        Digits spectrum(n_);
        pack(b, spectrum.data());
        forward(spectrum.data());
        return spectrum;
    }

    template<bool Timed = false>
    void product(const Digits& a, const Digits& b, Digits& out, Operation op, PhaseTimes* times = nullptr) {
        phase<Timed>(times, Pack, [&] {
            pack(a, x_.data());
            if (op == Operation::Multiply) pack(b, y_.data());
        });
        phase<Timed>(times, Forward, [&] {
            forward(x_.data());
            if (op == Operation::Multiply) forward(y_.data());
        });
        const Word* rhs = op == Operation::Square ? x_.data() :
                          op == Operation::Fixed ? b.data() : y_.data();
        phase<Timed>(times, Pointwise, [&] {
            for (unsigned i = 0; i < n_; i += Lanes)
                O::store(x_.data() + i, O::mul(O::load(x_.data() + i), O::load(rhs + i)));
        });
        phase<Timed>(times, Inverse, [&] { inverse(); });
        phase<Timed>(times, Unpack, [&] { unpack(); });
        phase<Timed>(times, Carry, [&] { carry(out); });
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
    void reset(const Digits& a, const Digits& b, Operation op) {
        binary64togw(&gw_, a.data(), a.size(), a_);
        binary64togw(&gw_, b.data(), b.size(), b_);
        if (op == Operation::Fixed) gwfft(&gw_, b_, b_);
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
    uint64_t fft_count() { return gw_get_fft_count(&gw_); }
    void clear_fft_count() { gw_clear_fft_count(&gw_); }
};

struct Integer {
    mpz_t v;
    Integer() { mpz_init(v); }
    ~Integer() { mpz_clear(v); }
    Integer(const Integer&) = delete;
    Integer& operator=(const Integer&) = delete;
};

const char* name(Operation op) {
    return op == Operation::Square ? "square" : op == Operation::Fixed ? "fixed" : "multiply";
}

Digits random_digits(unsigned p, std::mt19937_64& rng) {
    Digits out((p + 63) / 64 + 1); // Zero sentinel for a digit crossing the last limb.
    for (size_t i = 0; i + 1 < out.size(); ++i) out[i] = rng();
    if (p % 64) out[out.size() - 2] &= (Word{1} << (p % 64)) - 1;
    return out;
}

Digits gmp_chain(const Digits& a, const Digits& b, mpz_srcptr modulus, Operation op, unsigned count) {
    Integer x, y;
    mpz_import(x.v, a.size(), -1, sizeof(Word), 0, 0, a.data());
    mpz_import(y.v, b.size(), -1, sizeof(Word), 0, 0, b.data());
    for (unsigned i = 0; i < count; ++i) {
        mpz_mul(x.v, x.v, op == Operation::Square ? x.v : y.v);
        mpz_mod(x.v, x.v, modulus);
    }
    Digits result(a.size());
    mpz_export(result.data(), nullptr, -1, sizeof(Word), 0, 0, x.v);
    return result;
}

template<class Step, class Reset>
double chain(Step step, Reset reset, unsigned repeats, unsigned batch) {
    reset();
    for (unsigned i = 0; i < 100; ++i) step();
    reset();
    const auto start = Clock::now();
    for (unsigned i = 0; i < repeats; ++i) step();
    return std::chrono::duration<double, std::micro>(Clock::now() - start).count() / (repeats * batch);
}

void emit(Shape shape, Operation op, unsigned batch, unsigned sample, const char* backend,
          unsigned length, unsigned depth, double us) {
    std::printf("time,%u,%s,%u,%u,%s,%u,%u,%.6f\n", shape.p, name(op), batch, sample, backend, length, depth, us);
}

template<unsigned Lanes, bool Lazy, bool Fused>
void measure_ntt(Shape shape, const Chains& seed, const Chains& rhs, const Chains& expected,
                 Operation op, unsigned repeats, unsigned sample, mpz_srcptr modulus, bool profile_only) {
    const unsigned batch = static_cast<unsigned>(seed.size());
    WeightedPlan<Lanes, Lazy, Fused> plan(shape);
    Digits out(seed[0].size());
    if (sample == 1 && batch == 1) {
        std::mt19937_64 rng(2);
        for (unsigned i = 0; i < 32; ++i) {
            auto a = random_digits(shape.p, rng), b = random_digits(shape.p, rng);
            if (i < 3) {
                Integer near_limit;
                mpz_sub_ui(near_limit.v, modulus, i + 1);
                mpz_export(a.data(), nullptr, -1, sizeof(Word), 0, 0, near_limit.v);
                if (i == 0) b = a;
                else { std::fill(b.begin(), b.end(), 0); b[0] = i - 1; }
            }
            if (op == Operation::Fixed) plan.product(a, plan.fixed_spectrum(b), out, op);
            else plan.product(a, b, out, op);
            if (out != gmp_chain(a, b, modulus, op, 1))
                throw std::runtime_error("NTT GMP product mismatch at M" + std::to_string(shape.p));
        }
    }
    Chains current = seed;
    auto reset = [&] { current = seed; };
    // Independent coefficient-SIMD chains share this plan and are interleaved on one core.
    Chains spectra;
    if (op == Operation::Fixed)
        for (const auto& b : rhs) spectra.push_back(plan.fixed_spectrum(b));
    const auto& operand = op == Operation::Fixed ? spectra : rhs;
    auto step = [&] {
        for (unsigned k = 0; k < batch; ++k) {
            plan.product(current[k], operand[k], out, op);
            current[k].swap(out);
        }
    };
    char label[32];
    std::snprintf(label, sizeof(label), "%s-%s%s", Lanes == 4 ? "ymm" : "zmm",
                  Lazy ? "lazy" : "canonical", Fused ? "-fused8" : "");
    if (!profile_only) {
        const double us = chain(step, reset, repeats, batch);
        if (current != expected) throw std::runtime_error("NTT dependent chain GMP mismatch");
        emit(shape, op, batch, sample, label, 5 * shape.row, 0, us);
    }
    if (batch == 1) {
        PhaseTimes times{};
        reset();
        for (unsigned i = 0; i < 100; ++i) step();
        reset();
        for (unsigned i = 0; i < repeats; ++i) {
            plan.template product<true>(current[0], operand[0], out, op, &times);
            current[0].swap(out);
        }
        if (current != expected) throw std::runtime_error("profiled NTT dependent chain GMP mismatch");
        std::printf("phase,%u,%s,%u,%u,%s", shape.p, name(op), batch, sample, label);
        for (const auto elapsed : times)
            std::printf(",%.6f", std::chrono::duration<double, std::micro>(elapsed).count() / repeats);
        std::printf("\n");
        std::fflush(stdout);
    }
}

void probe(Shape shape, bool profile_only) {
    // All vector variants use identical digits, roots and coefficient bounds.
    const WeightedPlan<8, false, false> verified(shape, true);
    Integer modulus;
    mpz_setbit(modulus.v, shape.p);
    mpz_sub_ui(modulus.v, modulus.v, 1);
    std::mt19937_64 rng(1);
    Chains seeds, rhs;
    for (unsigned lane = 0; lane < 8; ++lane) {
        seeds.push_back(random_digits(shape.p, rng));
        rhs.push_back(random_digits(shape.p, rng));
    }
    for (unsigned batch : {1u, 2u, 4u, 8u}) {
        if (profile_only && batch != 1) break;
        const Chains a(seeds.begin(), seeds.begin() + batch), b(rhs.begin(), rhs.begin() + batch);
        const unsigned repeats = shape.p <= 16001 ? 2000 : 700;
        for (Operation op : {Operation::Multiply, Operation::Square, Operation::Fixed}) {
            Chains expected;
            for (unsigned i = 0; i < batch; ++i)
                expected.push_back(gmp_chain(a[i], b[i], modulus.v, op, repeats));
            using Runner = decltype(&measure_ntt<4, false, false>);
            constexpr Runner runners[] = {
                &measure_ntt<8, false, false>, &measure_ntt<8, true, false>,
                &measure_ntt<8, false, true>, &measure_ntt<8, true, true>,
                &measure_ntt<4, false, false>, &measure_ntt<4, true, false>,
                &measure_ntt<4, false, true>, &measure_ntt<4, true, true>,
            };
            for (unsigned sample = 1; sample <= 3; ++sample) {
                if (profile_only) {
                    runners[0](shape, a, b, expected, op, repeats, sample, modulus.v, true);
                    continue;
                }
                const bool square = op == Operation::Square;
                auto controls = [&] {
                    ecm_cpu_ntt_bench::IfmaControl ifma(shape.p);
                    const double t_ifma = chain([&] { ifma.step(square); }, [&] { ifma.reset(a, b); }, repeats, batch);
                    if (!ifma.equal(expected)) throw std::runtime_error("IFMA dependent chain GMP mismatch");
                    emit(shape, op, batch, sample, "ifma", ifma.limbs(), ifma.karatsuba_depth(square), t_ifma);
                    std::vector<std::unique_ptr<GwControl>> gw;
                    for (unsigned i = 0; i < batch; ++i) gw.push_back(std::make_unique<GwControl>(shape.p));
                    const double t_gw = chain([&] { for (auto& control : gw) control->step(square); },
                        [&] {
                            for (unsigned i = 0; i < batch; ++i) {
                                gw[i]->reset(a[i], b[i], op);
                                gw[i]->clear_fft_count();
                            }
                        }, repeats, batch);
                    uint64_t transforms = 0;
                    for (auto& control : gw) transforms += control->fft_count();
                    for (unsigned i = 0; i < batch; ++i)
                        if (!gw[i]->equal(expected[i])) throw std::runtime_error("GWNUM dependent chain GMP mismatch");
                    emit(shape, op, batch, sample, "gwnum", gw[0]->length(), 0, t_gw);
                    std::printf("gwfft,%u,%s,%u,%u,%.6f\n", shape.p, name(op), batch, sample,
                                double(transforms) / (repeats * batch));
                };
                if (sample == 2) controls();
                for (unsigned index = 0; index < std::size(runners); ++index) {
                    const auto selected = sample == 2 ? std::size(runners) - 1 - index :
                                          sample == 3 ? (index + 4) % std::size(runners) : index;
                    runners[selected](shape, a, b, expected, op, repeats, sample, modulus.v, false);
                }
                if (sample != 2) controls();
                std::fflush(stdout);
            }
        }
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
#ifdef _MSC_VER
        int regs[4];
        __cpuidex(regs, 1, 0);
        if (!(regs[2] & (1 << 27)) || (_xgetbv(0) & 0xe6) != 0xe6)
            throw std::runtime_error("AVX-512 OS state is unavailable");
        __cpuidex(regs, 7, 0);
        constexpr unsigned required = (1u << 16) | (1u << 17) | (1u << 21) | (1u << 31);
        if ((static_cast<unsigned>(regs[1]) & required) != required)
            throw std::runtime_error("AVX-512F/DQ/IFMA/VL is required");
        if (!SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR(1) << 1))
            throw std::runtime_error("cannot pin to logical CPU 1");
#endif
        const bool profile_only = argc > 1 && std::string_view(argv[1]) == "--profile-only";
        if (argc > 2 || (argc == 2 && !profile_only))
            throw std::invalid_argument("usage: cpu_stage1_weighted_ntt [--profile-only]");
        for (const auto shape : shapes) probe(shape, profile_only);
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
