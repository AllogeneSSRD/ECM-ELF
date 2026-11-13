// Compare the production IFMA Mersenne kernels to one or two Karatsuba
// levels over signed redundant columns. Eight lanes, radix 2^52. The
// production Mersenne fold is shared by both paths.
//
// mers_karatsuba [bits|--matrix|--sweep] [max_depth] [iterations] [--verify-only]
// Probe only, MSVC /arch:AVX512 or GCC -mavx512f -mavx512dq -mavx512ifma.
#include "../../src/ecm/cpu/arithmetic/ifma/modular_ifma.cpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <memory>
#include <stdexcept>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__linux__)
#include <sched.h>
#endif

namespace {
struct AlignedDelete {
    void operator()(uint64_t *p) const { _mm_free(p); }
};
using Aligned = std::unique_ptr<uint64_t, AlignedDelete>;

static Aligned make_columns(size_t n)
{
    Aligned p(static_cast<uint64_t *>(_mm_malloc(n * 64, 64)));
    if (!p) throw std::bad_alloc();
    return p;
}

static void pin_cpu_1()
{
#ifdef _WIN32
    if (!SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR(1) << 1))
        throw std::runtime_error("could not pin to CPU 1");
#elif defined(__linux__)
    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(1, &mask);
    if (sched_setaffinity(0, sizeof(mask), &mask))
        throw std::runtime_error("could not pin to CPU 1");
#endif
}

template<bool square>
static void run_kernel(uint64_t *out, const uint64_t *a, const uint64_t *b, const ifma_ctx_t *c,
                       uint64_t *scratch, int depth)
{
    if (depth) {
        const __m512i top = ecm_ifma_karatsuba::normalized_product<square>(
            c->scratch, a, b, c->n, scratch, depth);
        ifma_mersenne_fold_top(out, c->scratch, c, top);
    } else if constexpr (square) {
        ifma_mersenne_sqr(out, a, c);
    } else {
        ifma_mersenne_mul(out, a, b, c);
    }
}

static void fill_operands(uint64_t *a, uint64_t *b, const ifma_ctx_t *c, gmp_randstate_t rng,
                          unsigned trial)
{
    mpz_t x, y;
    mpz_inits(x, y, nullptr);
    for (unsigned lane = 0; lane < IFMA_LANES; ++lane) {
        mpz_urandomm(x, rng, c->N);
        mpz_urandomm(y, rng, c->N);
        if (trial == 0) {
            switch (lane) {
            case 0: mpz_set_ui(x, 0); break;
            case 1: mpz_set_ui(y, 0); break;
            case 2: mpz_set_ui(x, 1); break;
            case 3: mpz_set_ui(y, 1); break;
            case 4: mpz_sub_ui(x, c->N, 1); mpz_set(y, x); break;
            case 5: mpz_set_ui(x, 0); mpz_setbit(x, 52 * (c->n / 2)); mpz_sub_ui(x, x, 1); break;
            case 6: mpz_sub_ui(x, c->N, 1); mpz_set_ui(y, 2); break;
            case 7:
                if (c->k % 2 == 0) {
                    mpz_set_ui(x, 3);
                    mpz_divexact_ui(y, c->N, 3);
                }
                break;
            }
        }
        ifma_from_mpz_lane(a, lane, x, c);
        ifma_from_mpz_lane(b, lane, y, c);
    }
    mpz_clears(x, y, nullptr);
}

template<bool square>
static bool verify(const uint64_t *a, const uint64_t *b, uint64_t *expected, uint64_t *out,
                   const ifma_ctx_t *c, uint64_t *scratch, int max_depth)
{
    mpz_t x, y;
    mpz_inits(x, y, nullptr);
    for (unsigned lane = 0; lane < IFMA_LANES; ++lane) {
        ifma_to_mpz_lane(x, a, lane, c);
        if constexpr (square) mpz_mul(x, x, x);
        else {
            ifma_to_mpz_lane(y, b, lane, c);
            mpz_mul(x, x, y);
        }
        mpz_mod(x, x, c->N);
        ifma_from_mpz_lane(expected, lane, x, c);
    }
    mpz_clears(x, y, nullptr);
    const size_t bytes = c->n * 64;
    for (int depth = 0; depth <= max_depth; ++depth) {
        // In-place update matches the dependent chain and also ensures the
        // input buffer is not accidentally read after output publication.
        memcpy(out, a, bytes);
        run_kernel<square>(out, out, b, c, scratch, depth);
        if (memcmp(out, expected, bytes) != 0) {
            fprintf(stderr, "GMP mismatch bits=%zu square=%d depth=%d\n", c->k, int(square), depth);
            return false;
        }
    }
    return true;
}

template<bool square>
static bool measure(const uint64_t *a, const uint64_t *b, uint64_t *out, uint64_t *expected,
                    const ifma_ctx_t *c, uint64_t *scratch, int max_depth, int iters)
{
    constexpr int repeats = 9;
    std::array<std::array<double, repeats>, 3> samples{};
    const size_t bytes = c->n * 64;
    // Use a production result to check the entire dependency chain endpoint.
    memcpy(expected, a, bytes);
    for (int q = 0; q < iters; ++q)
        run_kernel<square>(expected, expected, b, c, scratch, 0);

    for (int r = 0; r < repeats; ++r) {
        for (int j = 0; j <= max_depth; ++j) {
            const int depth = (j + r) % (max_depth + 1);
            memcpy(out, a, bytes);
            for (int q = 0; q < 100; ++q)
                run_kernel<square>(out, out, b, c, scratch, depth);
            memcpy(out, a, bytes);
            const auto start = std::chrono::steady_clock::now();
            for (int q = 0; q < iters; ++q)
                run_kernel<square>(out, out, b, c, scratch, depth);
            const auto elapsed = std::chrono::steady_clock::now() - start;
            samples[depth][r] = std::chrono::duration<double, std::nano>(elapsed).count() / iters;
            if (memcmp(out, expected, bytes) != 0) {
                fprintf(stderr, "Chain mismatch bits=%zu square=%d depth=%d\n", c->k, int(square), depth);
                return false;
            }
        }
    }
    for (int depth = 0; depth <= max_depth; ++depth) {
        auto &row = samples[depth];
        printf("samples %zu %s %d", c->k, square ? "sqr" : "mul", depth);
        for (double v : row) printf(",%.2f", v);
        puts("");
        std::sort(row.begin(), row.end());
        printf("median %zu %zu %s %d %.2f ns/batch %+.2f%%\n", c->k, c->n,
               square ? "sqr" : "mul", depth, row[repeats / 2],
               (row[repeats / 2] / samples[0][repeats / 2] - 1.0) * 100.0);
    }
    return true;
}

static bool run(size_t bits, int depth, int iters)
{
    mpz_t modulus;
    mpz_init_set_ui(modulus, 1);
    mpz_mul_2exp(modulus, modulus, static_cast<mp_bitcnt_t>(bits));
    mpz_sub_ui(modulus, modulus, 1);
    ifma_ctx_t c;
    const int rc = ifma_ctx_init_ex(&c, modulus, IFMA_FIELD_MERS);
    mpz_clear(modulus);
    if (rc) throw std::runtime_error("cannot initialize Mersenne context");

    const auto a = make_columns(c.n), b = make_columns(c.n);
    const auto expected = make_columns(c.n), out = make_columns(c.n);
    const auto scratch = make_columns(ecm_ifma_karatsuba::workspace_columns(c.n, depth));
    gmp_randstate_t rng;
    gmp_randinit_default(rng);
    gmp_randseed_ui(rng, 2601);
    constexpr unsigned trials = 64;
    bool ok = true;
    for (unsigned trial = 0; ok && trial < trials; ++trial) {
        fill_operands(a.get(), b.get(), &c, rng, trial);
        ok = verify<false>(a.get(), b.get(), expected.get(), out.get(), &c, scratch.get(), depth)
          && verify<true>(a.get(), b.get(), expected.get(), out.get(), &c, scratch.get(), depth);
    }
    gmp_randclear(rng);
    if (ok) printf("verified %zu bits %zu limbs, %u GMP lane results per depth\n",
                   c.k, c.n, trials * IFMA_LANES * 2);
    if (ok && iters > 0) {
        ok = measure<false>(a.get(), b.get(), out.get(), expected.get(), &c, scratch.get(), depth, iters)
          && measure<true>(a.get(), b.get(), out.get(), expected.get(), &c, scratch.get(), depth, iters);
    }
    ifma_ctx_clear(&c);
    return ok;
}
} // namespace

int main(int argc, char **argv)
{
    try {
        const bool matrix = argc > 1 && strcmp(argv[1], "--matrix") == 0;
        const bool sweep = argc > 1 && strcmp(argv[1], "--sweep") == 0;
        const int depth = argc > 2 ? atoi(argv[2]) : 2;
        const int iters = argc > 3 ? atoi(argv[3]) : 5000;
        const bool verify_only = argc > 4 && strcmp(argv[4], "--verify-only") == 0;
        if (argc > 5 || (argc > 4 && !verify_only) || depth < 1 || depth > 2 || iters < 1)
            throw std::runtime_error("usage: mers_karatsuba [bits|--matrix|--sweep] [max_depth 1|2] [iterations] [--verify-only]");
        pin_cpu_1();
        printf("CPU 1, eight IFMA lanes, %d iterations x 9 repeats per variant\n", verify_only ? 0 : iters);
        if (matrix) {
            for (size_t bits : {991u, 2203u, 4253u, 8191u, 12511u, 16319u})
                if (!run(bits, depth, verify_only ? 0 : iters)) return 1;
            return 0;
        }
        if (sweep) {
            for (size_t n : {60u, 64u, 68u, 71u, 72u, 73u, 76u, 79u, 80u, 81u, 82u, 83u, 84u,
                             88u, 92u, 104u, 112u, 119u, 120u, 121u, 128u, 158u, 316u})
                if (!run(n * 52 - 1, depth, verify_only ? 0 : iters)) return 1;
            return 0;
        }
        const int bits = argc > 1 ? atoi(argv[1]) : 4253;
        if (bits < 64 || bits > 16432)
            throw std::runtime_error("supported bit range is 64..16432");
        return run(static_cast<size_t>(bits), depth, verify_only ? 0 : iters) ? 0 : 1;
    } catch (const std::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        return 2;
    }
}
