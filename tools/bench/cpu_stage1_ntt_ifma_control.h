#pragma once

#include "ecm/cpu/arithmetic/ifma/modular_ifma.h"

#include <cstdint>
#include <immintrin.h>
#include <memory>
#include <new>
#include <stdexcept>
#include <vector>

namespace ecm_cpu_ntt_bench {

// The production arithmetic always executes eight lanes. With fewer active
// chains the remaining lanes stay zero; count only the occupied lanes in ops/s.
class IfmaControl {
    using Words = std::vector<uint64_t>;
    using Chains = std::vector<Words>;
    struct AlignedDelete {
        void operator()(uint64_t* ptr) const { _mm_free(ptr); }
    };

    ifma_ctx_t context_{};
    std::unique_ptr<uint64_t, AlignedDelete> a_, b_;
    size_t occupied_ = 0;

    static void import(mpz_ptr value, const Words& words) {
        mpz_import(value, words.size(), -1, sizeof(uint64_t), 0, 0, words.data());
    }

public:
    explicit IfmaControl(unsigned p) {
        mpz_t modulus;
        mpz_init(modulus);
        mpz_setbit(modulus, p);
        mpz_sub_ui(modulus, modulus, 1);
        const int result = ifma_ctx_init_ex(&context_, modulus, IFMA_FIELD_MERS);
        mpz_clear(modulus);
        if (result != 0) throw std::runtime_error("IFMA Mersenne setup failed");

        const auto bytes = context_.n * IFMA_LANES * sizeof(uint64_t);
        a_.reset(static_cast<uint64_t*>(_mm_malloc(bytes, 64)));
        b_.reset(static_cast<uint64_t*>(_mm_malloc(bytes, 64)));
        if (!a_ || !b_) {
            ifma_ctx_clear(&context_);
            throw std::bad_alloc();
        }
    }

    IfmaControl(const IfmaControl&) = delete;
    IfmaControl& operator=(const IfmaControl&) = delete;
    ~IfmaControl() { ifma_ctx_clear(&context_); }

    void reset(const Chains& a, const Chains& b) {
        if (a.empty() || a.size() > IFMA_LANES || a.size() != b.size())
            throw std::invalid_argument("invalid IFMA control batch");
        occupied_ = a.size();
        ifma_set_zero(a_.get(), &context_);
        ifma_set_zero(b_.get(), &context_);
        mpz_t value;
        mpz_init(value);
        for (unsigned lane = 0; lane < occupied_; ++lane) {
            import(value, a[lane]);
            ifma_from_mpz_lane(a_.get(), lane, value, &context_);
            import(value, b[lane]);
            ifma_from_mpz_lane(b_.get(), lane, value, &context_);
        }
        mpz_clear(value);
    }

    void step(bool square) {
        if (square)
            ifma_mont_sqr(a_.get(), a_.get(), &context_);
        else
            ifma_mont_mul(a_.get(), a_.get(), b_.get(), &context_);
    }

    bool equal(const Chains& expected) const {
        if (!occupied_ || expected.size() != occupied_) return false;
        mpz_t actual, reference;
        mpz_inits(actual, reference, nullptr);
        bool matches = true;
        for (unsigned lane = 0; lane < IFMA_LANES; ++lane) {
            ifma_to_mpz_lane(actual, a_.get(), lane, &context_);
            if (lane < occupied_)
                import(reference, expected[lane]);
            else
                mpz_set_ui(reference, 0);
            matches &= mpz_cmp(actual, reference) == 0;
        }
        mpz_clears(actual, reference, nullptr);
        return matches;
    }

    size_t limbs() const { return context_.n; }
    unsigned karatsuba_depth(bool square) const {
        return square ? context_.sqr_depth : context_.mul_depth;
    }
};

} // namespace ecm_cpu_ntt_bench
