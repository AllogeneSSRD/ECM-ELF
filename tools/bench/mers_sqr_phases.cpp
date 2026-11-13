/* CPU Stage1 IFMA phases for both operations and reduction modes.
 * Build with tools/build/test/build_cpu_stage1_ifma_phases.ps1.
 * The generated diagnostic copy comes from the current production source.
 * Run on an AVX-512 IFMA host. Every reported time covers a batch of 8 lanes.
 */
#include "ifma_phase_source.inc"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#else
#include <sched.h>
#endif

#if defined(_MSC_VER)
#define PHASE_NOINLINE __declspec(noinline)
#else
#define PHASE_NOINLINE __attribute__((noinline))
#endif

enum class Phase : size_t { clear, product, carry, fold_high, fold_carry, canonical, count };
constexpr size_t phase_count = size_t(Phase::count);
using Samples = std::array<uint64_t, phase_count>;

static Samples elapsed;
static uint64_t last_tick;

static inline uint64_t tick()
{
    // Serialize the timestamp against both adjacent phases.
    _mm_lfence();
    const uint64_t t = __rdtsc();
    _mm_lfence();
    return t;
}

static PHASE_NOINLINE void mark(Phase phase)
{
    const uint64_t now = tick();
    elapsed[size_t(phase)] += now - last_tick;
    last_tick = now;
}

#include "ifma_phase_kernels.inc"

using Kernel = void (*)(uint64_t*, const uint64_t*, const uint64_t*, const ifma_ctx_t*);

static void square(uint64_t* out, const uint64_t* a, const uint64_t*, const ifma_ctx_t* c)
{
    ifma_mersenne_sqr(out, a, c);
}

static void profile_square(uint64_t* out, const uint64_t* a, const uint64_t*, const ifma_ctx_t* c)
{
    profile_ifma_mersenne_sqr(out, a, c);
}

static void profile_cios_square(uint64_t* out, const uint64_t* a, const uint64_t*, const ifma_ctx_t* c)
{
    profile_ifma_cios(out, a, a, c);
}

static void cios_square(uint64_t* out, const uint64_t* a, const uint64_t*, const ifma_ctx_t* c)
{
    ifma_cios(out, a, a, c);
}

struct Buffer {
    uint64_t* data;

    explicit Buffer(size_t n) : data(static_cast<uint64_t*>(_mm_malloc(n * sizeof(uint64_t), 64)))
    {
        if (!data) throw std::bad_alloc();
    }

    ~Buffer() { _mm_free(data); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

static void pin_cpu(int cpu)
{
#if defined(_WIN32)
    if (cpu >= int(sizeof(DWORD_PTR) * 8) ||
        !SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR(1) << cpu))
        throw std::runtime_error("Cannot pin probe to requested logical CPU");
#else
    cpu_set_t set;
    CPU_ZERO(&set);
    if (cpu >= CPU_SETSIZE) throw std::runtime_error("Logical CPU is out of range");
    CPU_SET(cpu, &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0)
        throw std::runtime_error("Cannot pin probe to requested logical CPU");
#endif
}

using Clock = std::chrono::steady_clock;

static double calibrate_ns_per_tick()
{
    const auto a = Clock::now();
    const uint64_t t0 = tick();
    while (Clock::now() - a < std::chrono::milliseconds(40)) {}
    const uint64_t t1 = tick();
    return std::chrono::duration<double, std::nano>(Clock::now() - a).count() / (t1 - t0);
}

static double timer_ticks(size_t reps)
{
    elapsed = {};
    for (size_t i = 0; i < reps; ++i) {
        last_tick = tick();
        mark(Phase::clear);
        mark(Phase::product);
    }
    return double(elapsed[size_t(Phase::product)]) / reps;
}

static double median(std::vector<double> values)
{
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

static void measure(unsigned p, const ifma_ctx_t& ctx, const uint64_t* a, const uint64_t* b,
                    bool sqr, size_t reps, size_t samples, double ns_per_tick)
{
    const bool fold = ctx.mode == IFMA_FIELD_MERS;
    const Kernel native = fold ? (sqr ? square : ifma_mersenne_mul)
                              : (sqr ? cios_square : ifma_cios);
    const Kernel profile = fold ? (sqr ? profile_square : profile_ifma_mersenne_mul)
                               : (sqr ? profile_cios_square : profile_ifma_cios);
    const size_t words = IFMA_LANES * ctx.n;
    Buffer x(words), y(words);
    native(x.data, a, b, &ctx);
    last_tick = tick();
    profile(y.data, a, b, &ctx);
    if (memcmp(x.data, y.data, words * sizeof(uint64_t)) != 0)
        throw std::runtime_error("Instrumented kernel differs from production");

    for (size_t i = 0; i < 200; ++i)
        native(x.data, a, b, &ctx);

    const char* mode = fold ? "fold_schoolbook" : "cios";
    const char* op = sqr ? "sqr" : "mul";
    using Values = std::array<double, phase_count>;
    std::array<std::vector<double>, phase_count> phase_values;
    std::vector<double> native_values, marked_values;
    for (size_t sample = 0; sample < samples; ++sample) {
        const double overhead = timer_ticks(reps);
        auto run_native = [&] {
            const auto start = Clock::now();
            for (size_t i = 0; i < reps; ++i) native(x.data, a, b, &ctx);
            return std::chrono::duration<double, std::nano>(Clock::now() - start).count() / reps;
        };
        double total = 0;
        if (sample % 2 == 0) total = run_native();
        elapsed = {};
        for (size_t i = 0; i < reps; ++i) {
            last_tick = tick();
            profile(y.data, a, b, &ctx);
        }
        Values phase{};
        double marked = 0;
        // Markers perturb code generation and execution. Native timings are
        // the whole-kernel baseline and need not equal this phase sum.
        for (size_t i = 0; i < phase_count; ++i) {
            phase[i] = elapsed[i] ? (double(elapsed[i]) / reps - overhead) * ns_per_tick : 0;
            phase_values[i].push_back(phase[i]);
            marked += phase[i];
        }
        if (sample % 2 != 0) total = run_native();
        native_values.push_back(total);
        marked_values.push_back(marked);
        printf("sample,%u,%zu,%s,%s,%zu,%.1f,%.1f,%.1f", p, ctx.n, mode, op,
               sample + 1, total, marked, overhead * ns_per_tick);
        for (double value : phase) printf(",%.1f", value);
        putchar('\n');
    }
    printf("median,%u,%zu,%s,%s,%zu,%.1f,%.1f,", p, ctx.n, mode, op, samples,
           median(native_values), median(marked_values));
    for (const auto& values : phase_values) printf(",%.1f", median(values));
    putchar('\n');
}

static void run(unsigned p, size_t reps, size_t samples, double ns_per_tick)
{
    mpz_t modulus, value;
    mpz_inits(modulus, value, nullptr);
    mpz_setbit(modulus, p);
    mpz_sub_ui(modulus, modulus, 1);
    gmp_randstate_t rng;
    gmp_randinit_mt(rng);
    for (int mode : {IFMA_FIELD_MERS, IFMA_FIELD_MONT}) {
        ifma_ctx_t ctx;
        if (ifma_ctx_init_ex(&ctx, modulus, mode) != 0)
            throw std::runtime_error("IFMA context initialization failed");
        gmp_randseed_ui(rng, 2026);
        Buffer a(IFMA_LANES * ctx.n), b(IFMA_LANES * ctx.n);
        for (unsigned lane = 0; lane < IFMA_LANES; ++lane) {
            mpz_urandomm(value, rng, modulus);
            ifma_from_mpz_lane(a.data, lane, value, &ctx);
            mpz_urandomm(value, rng, modulus);
            ifma_from_mpz_lane(b.data, lane, value, &ctx);
        }
        for (bool sqr : {false, true})
            measure(p, ctx, a.data, b.data, sqr, reps, samples, ns_per_tick);
        ifma_ctx_clear(&ctx);
    }
    gmp_randclear(rng);
    mpz_clears(modulus, value, nullptr);
}

int main(int argc, char** argv)
{
    try {
        const int cpu = argc > 1 ? atoi(argv[1]) : 1;
        const int reps = argc > 2 ? atoi(argv[2]) : 4000;
        const int samples = argc > 3 ? atoi(argv[3]) : 9;
        if (cpu < 0 || reps < 100 || samples < 3 || samples % 2 == 0 || argc > 4)
            throw std::runtime_error("Usage: mers_sqr_phases [logical_cpu] [reps >= 100] [odd samples >= 3]");
        pin_cpu(cpu);
        const double ns_per_tick = calibrate_ns_per_tick();
        printf("# CPU=%d, reps/sample=%d, samples=%d, ns/TSC tick=%.6f, lanes=%d\n",
               cpu, reps, samples, ns_per_tick, IFMA_LANES);
        printf("stat,p,n52,mode,op,sample,native_ns,marked_sum_ns,timer_ns,clear_ns,product_ns,"
               "carry_ns,%s,%s,canonical_ns\n", fold_columns[0], fold_columns[1]);
        for (unsigned p : {991u, 2203u, 4253u, 8191u})
            run(p, size_t(reps), size_t(samples), ns_per_tick);
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
