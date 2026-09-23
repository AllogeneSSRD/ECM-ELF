/* ──────────────────────────────────────────────────────────────────────────
 *  cpu_mont_bench — standalone CPU Montgomery multiplication benchmark
 *
 *  Build: g++ -O3 -mavx512f -mavx512dq -mavx2 -mfma -mf16c src/cpu_mont_bench.cpp
 *              src/cpu_mont_avx.cpp src/cpu_mont_scalar.cpp
 *              -Iinclude -Isrc -lgmp -o cpu_mont_bench
 *
 *  Run:   cpu_mont_bench --bits 512 --iterations 10000 --instances 64
 * ──────────────────────────────────────────────────────────────────────── */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <chrono>
#include <string>
#include <vector>
#include <algorithm>
#include <thread>
#include <mutex>
#include <atomic>

#include <gmp.h>

#include "cpu_mont_avx.h"
#include "cpu_mont_scalar.h"

#ifdef _WIN32
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
#else
    #include <pthread.h>
    #include <sched.h>
    #include <unistd.h>
#endif

/* ══════════════════════════════════════════════════════════════════════════
 *  CLI helpers
 * ════════════════════════════════════════════════════════════════════════ */

#include <cmath>
#include <sstream>

static constexpr int DEFAULT_IPT_AVX512 = 16;
static constexpr int DEFAULT_IPT_AVX2   = 8;

/// Parse a CLI value that may be in scientific notation (e.g. 1e6, 5e5).
static bool parse_cli_count(const char *s, const char *label, int &out) {
    if (s == nullptr || *s == '\0') return true;
    try {
        double d = std::stod(s);
        if (!std::isfinite(d) || d < 0.0) {
            std::fprintf(stderr, "Invalid %s: %s\n", label, s);
            return false;
        }
        out = (int)(d + 0.5);
        return true;
    } catch (...) {
        std::fprintf(stderr, "Invalid %s: %s\n", label, s);
        return false;
    }
}

static void print_usage(const char *prog)
{
    std::printf(
        "Usage: %s [options] [bits] [iterations] [ipt] [repeats]\n"
        "  Positional args:\n"
        "    bits                    Bit-width (default: 512)\n"
        "    iterations              Montgomery multiplications per thread; supports 1e6 (default: 1000)\n"
        "    ipt                     Instances per thread, auto=16 AVX512 / 8 AVX2 (default: auto)\n"
        "    repeats                 Launch repeats per timing sample (default: 1)\n"
        "  Options:\n"
        "  -b, --bits <N>           Alias for 1st positional\n"
        "  -k, --kernel-iters <N>   Alias for 2nd positional; supports 1e6\n"
        "  -i, --ipt <N>            Alias for 3rd positional\n"
        "  -t, --threads <N>        Number of threads (default: 1)\n"
        "  -r, --repeats <N>        Alias for 4th positional\n"
        "  -a, --affinity MODE      CPU affinity: auto (default), none, or comma-sep logical CPUs\n"
        "  --no-overflow            Use small inputs (less conditional sub triggers)\n"
        "  --avx2                   Force AVX2 code path (even on AVX512-capable CPU)\n"
        "  --sync-barrier           Barrier-sync every iteration across threads\n"
        "  --no-verify              Skip correctness self-test\n"
        "  -v, --verbose\n"
        "\nExamples:\n"
        "  %s 512 1e6 16                                    # latency: 1 thread\n"
        "  %s 512 1e6 16 5 -t 12 -a 1,3,5,7,9,11,13,15,17,19,21,23\n"
        "                                                    # throughput: 12 threads\n",
        prog, prog, prog);
}

/* ══════════════════════════════════════════════════════════════════════════
 *  High-res timer
 * ════════════════════════════════════════════════════════════════════════ */

static double now_sec()
{
#ifdef _WIN32
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / (double)freq.QuadPart;
#else
    auto now = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double>(now.time_since_epoch()).count();
#endif
}

/* ══════════════════════════════════════════════════════════════════════════
 *  CPU topology detection
 * ════════════════════════════════════════════════════════════════════════ */

static uint32_t get_logical_cpu_count()
{
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    return si.dwNumberOfProcessors;
#else
    return (uint32_t)sysconf(_SC_NPROCESSORS_CONF);
#endif
}

#ifdef _WIN32
/* Detect physical core count on Windows by using GetLogicalProcessorInformation.
 * Returns number of physical cores (packages × cores per package, ignoring HT),
 * or falls back to logical count. */
static uint32_t get_physical_core_count()
{
    DWORD len = 0;
    GetLogicalProcessorInformation(nullptr, &len);
    if (len == 0) return get_logical_cpu_count();

    std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buf(len / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
    if (!GetLogicalProcessorInformation(buf.data(), &len))
        return get_logical_cpu_count();

    uint32_t phys_cores = 0;
    for (auto &info : buf) {
        if (info.Relationship == RelationProcessorCore) {
            phys_cores++;
        }
    }
    return phys_cores > 0 ? phys_cores : get_logical_cpu_count();
}
#else
static uint32_t get_physical_core_count()
{
    /* Linux: count unique physical core ids via /sys/devices/system/cpu/cpuN/topology/core_id */
    FILE *f = fopen("/sys/devices/system/cpu/present", "r");
    if (!f) return get_logical_cpu_count();
    uint32_t start, end;
    if (fscanf(f, "%u-%u", &start, &end) != 2) { fclose(f); return get_logical_cpu_count(); }
    fclose(f);

    /* count unique physical core ids */
    std::vector<uint32_t> core_ids;
    char path[256];
    for (uint32_t cpu = start; cpu <= end; cpu++) {
        snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%u/topology/core_id", cpu);
        f = fopen(path, "r");
        if (!f) continue;
        uint32_t core_id;
        if (fscanf(f, "%u", &core_id) == 1) {
            bool found = false;
            for (auto &id : core_ids) if (id == core_id) { found = true; break; }
            if (!found) core_ids.push_back(core_id);
        }
        fclose(f);
    }
    return core_ids.empty() ? get_logical_cpu_count() : (uint32_t)core_ids.size();
}
#endif

/* ══════════════════════════════════════════════════════════════════════════
 *  CPU affinity
 * ════════════════════════════════════════════════════════════════════════ */

#ifdef _WIN32
static bool apply_affinity_any()
{
    return SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)-1) != 0;
}

static bool apply_affinity_core(uint32_t core_idx)
{
    /* core_idx is zero-based physical core index.
     * On Windows, we map it to the first logical processor of that physical core.
     * For simplicity, if core_idx < physical_cores, bind to logical CPU = core_idx
     * (first thread of each physical core). */
    DWORD_PTR mask = (DWORD_PTR)1 << core_idx;
    return SetThreadAffinityMask(GetCurrentThread(), mask) != 0;
}

static bool apply_affinity_list(const std::vector<uint32_t> &cpus, uint32_t thread_idx)
{
    if (thread_idx >= cpus.size()) return false;
    DWORD_PTR mask = (DWORD_PTR)1 << cpus[thread_idx];
    return SetThreadAffinityMask(GetCurrentThread(), mask) != 0;
}

#else
static bool apply_affinity_any()
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (int i = 0; i < CPU_SETSIZE; i++)
        CPU_SET(i, &cpuset);
    return pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset) == 0;
}

static bool apply_affinity_core(uint32_t core_idx)
{
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core_idx, &cpuset);
    return pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset) == 0;
}

static bool apply_affinity_list(const std::vector<uint32_t> &cpus, uint32_t thread_idx)
{
    if (thread_idx >= cpus.size()) return false;
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpus[thread_idx], &cpuset);
    return pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset) == 0;
}
#endif

static std::vector<uint32_t> parse_affinity_list(const std::string &s)
{
    std::vector<uint32_t> result;
    std::string token;
    for (size_t i = 0; i <= s.size(); ++i) {
        if (i == s.size() || s[i] == ',') {
            if (!token.empty()) {
                result.push_back((uint32_t)std::atoi(token.c_str()));
                token.clear();
            }
        } else if (s[i] >= '0' && s[i] <= '9') {
            token += s[i];
        }
    }
    return result;
}

/* ══════════════════════════════════════════════════════════════════════════
 *  GMP ↔ uint32_t[] conversion for benchmark data
 * ════════════════════════════════════════════════════════════════════════ */

static void fill_from_gmp(const mpz_t v, uint32_t *out, uint32_t words)
{
    mpz_t tmp, mod;
    mpz_init(tmp);
    mpz_init(mod);
    mpz_ui_pow_ui(mod, 2ul, (unsigned long)(words * 32));
    mpz_mod(tmp, v, mod);
    size_t count = 0;
    mpz_export(out, &count, -1, sizeof(uint32_t), 0, 0, tmp);
    for (size_t i = count; i < words; ++i) out[i] = 0u;
    mpz_clear(tmp);
    mpz_clear(mod);
}

/* ══════════════════════════════════════════════════════════════════════════
 *  Seeded LCG for reproducible random fill  (used by self-test only)
 * ════════════════════════════════════════════════════════════════════════ */

static uint32_t lcg_state[4] = { 0x12345678u, 0x9abcdef0u, 0xdeadbeefu, 0xcafebabeu };

static inline uint32_t lcg_rand_u32()
{
    uint64_t s1 = ((uint64_t)lcg_state[0] << 32) | lcg_state[1];
    uint64_t s0 = ((uint64_t)lcg_state[2] << 32) | lcg_state[3];
    uint64_t result = s0 + s0;
    s1 ^= s1 << 23;
    uint64_t ns = s1 ^ s0 ^ (s1 >> 18) ^ (s0 >> 5);
    lcg_state[0] = (uint32_t)(ns >> 32);
    lcg_state[1] = (uint32_t)(ns);
    lcg_state[2] = (uint32_t)(s0 >> 32);
    lcg_state[3] = (uint32_t)(s0);
    return (uint32_t)(result >> 32);
}

static void fill_random_limbs(uint32_t *dst, uint32_t limbs)
{
    for (uint32_t i = 0; i < limbs; ++i)
        dst[i] = lcg_rand_u32();
}

static void generate_modulus(uint32_t *N, uint32_t *np0_out, uint32_t limbs)
{
    fill_random_limbs(N, limbs);
    N[0] |= 1u;
    N[limbs - 1] |= (1u << 31);
    cpu_mont_np0_compute(np0_out, N, limbs);
}

/* ══════════════════════════════════════════════════════════════════════════
 *  Self-test helpers
 * ════════════════════════════════════════════════════════════════════════ */

static bool selftest_one(uint32_t LIM, uint32_t K, uint32_t stride,
                          const uint32_t *N_data, uint32_t np0, bool use_avx512)
{
    std::vector<uint32_t> a(K * stride);
    std::vector<uint32_t> b(K * stride);

    for (uint32_t k = 0; k < K; ++k) {
        fill_random_limbs(&a[k * stride], LIM);
        fill_random_limbs(&b[k * stride], LIM);
        for (uint32_t j = 0; j < LIM; ++j) {
            a[k * stride + j] &= N_data[j];
            b[k * stride + j] &= N_data[j];
        }
    }

    /* scalar reference */
    std::vector<uint32_t> ref(K * stride);
    for (uint32_t k = 0; k < K; ++k) {
        cpu_mont_scalar_cios(&ref[k * stride], &a[k * stride], &b[k * stride],
                             N_data, np0, LIM);
    }

    /* AVX batch */
    std::vector<uint32_t> out(K * stride);
    if (use_avx512) {
        avx512_mont_cios_batch(out.data(), a.data(), b.data(),
                                N_data, np0, LIM, stride, K);
    } else {
        avx2_mont_cios_batch(out.data(), a.data(), b.data(),
                              N_data, np0, LIM, stride, K);
    }

    for (uint32_t k = 0; k < K; ++k) {
        for (uint32_t j = 0; j < LIM; ++j) {
            if (out[k * stride + j] != ref[k * stride + j]) {
                std::printf("  FAIL: instance %u limb %u: got 0x%08x expected 0x%08x\n",
                            k, j, out[k * stride + j], ref[k * stride + j]);
                return false;
            }
        }
    }
    return true;
}

static bool run_selftest(uint32_t LIM, uint32_t K, uint32_t stride,
                          const uint32_t *N_data, uint32_t np0,
                          bool use_avx512, int thread_idx, int num_tests)
{
    char label[64];
    // if (thread_idx >= 0)
    //     snprintf(label, sizeof(label), "  Self-test thread %d (%s, %u curves, %u-bit)... ",
    //              thread_idx, use_avx512 ? "AVX512" : "AVX2", K, LIM * 32);
    // else
    //     snprintf(label, sizeof(label), "  Self-test (%s, %u curves, %u-bit)... ",
    //              use_avx512 ? "AVX512" : "AVX2", K, LIM * 32);

    std::printf("%s", label);
    fflush(stdout);

    lcg_state[0] = 0x11111111u + (thread_idx >= 0 ? (uint32_t)thread_idx : 0u);
    lcg_state[1] = 0x22222222u;
    lcg_state[2] = 0x33333333u;
    lcg_state[3] = 0x44444444u;

    for (int t = 0; t < num_tests; ++t) {
        if (!selftest_one(LIM, K, stride, N_data, np0, use_avx512)) {
            std::printf("FAILED at test %d\n", t);
            return false;
        }
    }
    // std::printf("PASSED (%d tests)\n", num_tests);
    return true;
}

/* ══════════════════════════════════════════════════════════════════════════
 *  Benchmarking data structures
 * ════════════════════════════════════════════════════════════════════════ */

struct ThreadResult {
    double elapsed_sec = 0.0;
    uint64_t ops = 0;
    bool selftest_passed = true;
};

/* Context passed to each worker thread */
struct ThreadCtx {
    uint32_t thread_idx;
    uint32_t K;            /* instances per thread */
    uint32_t LIM;          /* limbs */
    uint32_t stride;       /* = LIM */
    uint32_t iterations;
    uint32_t repeats;
    bool use_avx512;
    bool verify;
    bool sync_barrier;
    const uint32_t *shared_N;
    uint32_t np0;

    /* worker local buffers */
    std::vector<uint32_t> a;
    std::vector<uint32_t> b;
    std::vector<uint32_t> out;

    /* shared barrier */
    std::atomic<int> *barrier_count;
    int num_threads;
    std::mutex *barrier_mtx;
#ifdef _WIN32
    CONDITION_VARIABLE *barrier_cv;
#else
    pthread_cond_t *barrier_cv;
    pthread_mutex_t *barrier_pmtx;
#endif

    ThreadResult result;
};

static void thread_benchmark(ThreadCtx &ctx)
{
    const uint32_t K   = ctx.K;
    const uint32_t LIM = ctx.LIM;
    const uint32_t stride = ctx.stride;
    const uint32_t *N  = ctx.shared_N;
    uint32_t np0       = ctx.np0;

    /* self-test (each thread uses own seed offset) */
    if (ctx.verify) {
        ctx.result.selftest_passed = run_selftest(LIM, K, stride, N, np0,
                                                   ctx.use_avx512, (int)ctx.thread_idx, 10);
        if (!ctx.result.selftest_passed) return;
    }

    /* warm-up */
    const uint32_t WARMUP = 10;
    for (uint32_t w = 0; w < WARMUP; ++w) {
        if (ctx.use_avx512) {
            avx512_mont_cios_batch(ctx.out.data(), ctx.a.data(), ctx.b.data(),
                                   N, np0, LIM, stride, K);
        } else {
            avx2_mont_cios_batch(ctx.out.data(), ctx.a.data(), ctx.b.data(),
                                  N, np0, LIM, stride, K);
        }
    }

    /* timed run */
    double t0 = now_sec();
    for (uint32_t r = 0; r < ctx.repeats; ++r) {
        for (uint32_t i = 0; i < ctx.iterations; ++i) {
            if (ctx.use_avx512) {
                avx512_mont_cios_batch(ctx.out.data(), ctx.a.data(), ctx.b.data(),
                                       N, np0, LIM, stride, K);
            } else {
                avx2_mont_cios_batch(ctx.out.data(), ctx.a.data(), ctx.b.data(),
                                      N, np0, LIM, stride, K);
            }

            if (ctx.sync_barrier && ctx.num_threads > 1) {
                /* barrier: all threads wait here after each iteration */
                int old_val = (*ctx.barrier_count)++;
                if (old_val + 1 < ctx.num_threads) {
#ifdef _WIN32
                    std::unique_lock<std::mutex> lock(*ctx.barrier_mtx);
                    ctx.barrier_cv; // placeholder for cond var wait
                    /* Wait for last thread to arrive */
                    while (*ctx.barrier_count < ctx.num_threads) {
                        // spin-wait is simpler but less efficient
                        // For brief waits this is acceptable
                    }
#else
                    // Simple spin-barrier for cross-platform
                    while (*ctx.barrier_count < ctx.num_threads) {
                        /* spin */
                    }
#endif
                } else {
                    /* last thread resets barrier */
                    *ctx.barrier_count = 0;
                }
            }
        }
    }
    double t1 = now_sec();

    ctx.result.elapsed_sec = t1 - t0;
    ctx.result.ops = (uint64_t)ctx.iterations * ctx.repeats * K;
}

/* ══════════════════════════════════════════════════════════════════════════
 *  Main
 * ════════════════════════════════════════════════════════════════════════ */

int main(int argc, char *argv[])
{
    /* Default values */
    uint32_t bits       = 512;
    uint32_t iterations = 1000;
    uint32_t ipt        = 0;           // 0 = auto (16 AVX512, 8 AVX2)
    uint32_t repeats    = 1;
    bool     force_avx2 = false;
    bool     do_verify  = true;
    bool     verbose    = false;
    bool     sync_barrier = false;
    bool     no_overflow  = false;
    std::string affinity = "auto";
    uint32_t num_threads = 1;

    /* Parse CLI */
    const char *prog = argv[0];
    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-h" || a == "--help") { print_usage(prog); return 0; }
        if ((a == "-b" || a == "--bits") && i + 1 < argc) {
            if (!parse_cli_count(argv[++i], "--bits", (int&)bits)) return 1;
            continue;
        }
        if ((a == "-k" || a == "--kernel-iters" || a == "--iterations") && i + 1 < argc) {
            if (!parse_cli_count(argv[++i], a.c_str(), (int&)iterations)) return 1;
            continue;
        }
        if ((a == "-i" || a == "--ipt" || a == "--instances") && i + 1 < argc) {
            if (!parse_cli_count(argv[++i], a.c_str(), (int&)ipt)) return 1;
            continue;
        }
        if ((a == "-t" || a == "--threads") && i + 1 < argc) {
            if (!parse_cli_count(argv[++i], "--threads", (int&)num_threads)) return 1;
            continue;
        }
        if ((a == "-r" || a == "--repeats") && i + 1 < argc) {
            if (!parse_cli_count(argv[++i], "--repeats", (int&)repeats)) return 1;
            continue;
        }
        if ((a == "-a" || a == "--affinity" || a == "--aff") && i + 1 < argc) {
            affinity = argv[++i];
            continue;
        }
        if (a == "--avx2") { force_avx2 = true; continue; }
        if (a == "--no-overflow") { no_overflow = true; continue; }
        if (a == "--sync-barrier") { sync_barrier = true; continue; }
        if (a == "--no-verify") { do_verify = false; continue; }
        if (a == "-v" || a == "--verbose") { verbose = true; continue; }
        if (!a.empty() && a[0] == '-') {
            std::printf("Unknown option: %s\n", a.c_str());
            print_usage(prog);
            return 1;
        }
        pos.push_back(a);
    }
    if (pos.size() >= 1 && !parse_cli_count(pos[0].c_str(), "bits", (int&)bits)) return 1;
    if (pos.size() >= 2 && !parse_cli_count(pos[1].c_str(), "iterations", (int&)iterations)) return 1;
    if (pos.size() >= 3 && !parse_cli_count(pos[2].c_str(), "ipt", (int&)ipt)) return 1;
    if (pos.size() >= 4 && !parse_cli_count(pos[3].c_str(), "repeats", (int&)repeats)) return 1;

    /* CPU feature detection */
    bool have_avx512 = cpu_has_avx512f();
    bool have_avx2   = cpu_has_avx2();
    bool use_avx512  = have_avx512 && !force_avx2;
    bool use_avx2    = have_avx2;

    if (verbose) {
        std::printf("CPU features: AVX512F=%d  AVX2=%d\n", (int)have_avx512, (int)have_avx2);
        std::printf("Physical cores: %u  Logical CPUs: %u\n",
                    get_physical_core_count(), get_logical_cpu_count());
    }

    if (!use_avx512 && !use_avx2) {
        std::printf("Error: CPU does not support AVX2 or AVX512. Cannot run.\n");
        return 1;
    }

    const uint32_t LIM = bits / 32;
    if (LIM > CPU_MONT_MAX_LIMBS) {
        std::printf("Error: bits=%u exceeds CPU_MONT_MAX_LIMBS=%u (max %u bits)\n",
                    bits, CPU_MONT_MAX_LIMBS, CPU_MONT_MAX_LIMBS * 32);
        return 1;
    }

    /* Determine instances per thread and total instances */
    const uint32_t DEFAULT_K = use_avx512 ? (uint32_t)DEFAULT_IPT_AVX512 : (uint32_t)DEFAULT_IPT_AVX2;
    if (ipt == 0) ipt = DEFAULT_K;
    uint32_t total_instances = ipt * num_threads;

    /* Validate thread count against physical cores */
    uint32_t phys_cores = get_physical_core_count();
    if (num_threads > phys_cores && affinity == "auto") {
        std::printf("  Warning: %u threads requested but only %u physical cores detected. "
                    "Performance may degrade.\n", num_threads, phys_cores);
    }

    /* Parse affinity */
    std::vector<uint32_t> affinity_cpus;
    bool affinity_auto = (affinity == "auto");
    bool affinity_none = (affinity == "none");
    if (!affinity_auto && !affinity_none) {
        affinity_cpus = parse_affinity_list(affinity);
        if (affinity_cpus.size() < num_threads) {
            std::printf("Error: --affinity specified %zu CPUs but %u threads needed\n",
                        affinity_cpus.size(), num_threads);
            return 1;
        }
    }

    /* Shared data and thread contexts */
    std::vector<uint32_t> shared_N(LIM);
    uint32_t np0;
    std::atomic<int> barrier_count(0);
    std::mutex barrier_mtx;
#ifdef _WIN32
    CONDITION_VARIABLE barrier_cv;
#else
    pthread_cond_t barrier_cv = PTHREAD_COND_INITIALIZER;
    pthread_mutex_t barrier_pmtx = PTHREAD_MUTEX_INITIALIZER;
#endif
    std::vector<ThreadCtx> thread_ctxs(num_threads);
    std::vector<std::thread> threads;

    /* Two test cases (same seed scheme as other benches):
     *   0 = large inputs (a,b large) — more likely to trigger conditional sub
     *   1 = small inputs (a,b small) — less likely to trigger conditional sub
     */
    {
        gmp_randstate_t rng;
        gmp_randinit_default(rng);
        const int case_idx = no_overflow ? 1 : 0;
        gmp_randseed_ui(rng, (unsigned long)(bits * 31337u + (unsigned)case_idx * 0x9e3779b9u));

        mpz_t gN, ga, gb;
        mpz_init(gN); mpz_init(ga); mpz_init(gb);

        mpz_urandomb(gN, rng, (unsigned long)bits);
        mpz_setbit(gN, (unsigned long)(bits - 1));
        mpz_setbit(gN, 0ul);

        if (case_idx == 0) {  // large-inputs: a,b in [N/2, N)
            mpz_t half; mpz_init(half); mpz_tdiv_q_ui(half, gN, 2u);
            mpz_urandomm(ga, rng, half); mpz_add(ga, ga, half);
            mpz_urandomm(gb, rng, half); mpz_add(gb, gb, half);
            mpz_clear(half);
        } else {  // small-inputs: both < N/4
            mpz_t quar; mpz_init(quar); mpz_tdiv_q_ui(quar, gN, 4u);
            mpz_urandomm(ga, rng, quar); mpz_urandomm(gb, rng, quar);
            mpz_clear(quar);
        }
        gmp_randclear(rng);

        /* Convert to word arrays */
        std::vector<uint32_t> a_words(LIM), b_words(LIM);
        fill_from_gmp(gN, shared_N.data(), LIM);
        fill_from_gmp(ga, a_words.data(), LIM);
        fill_from_gmp(gb, b_words.data(), LIM);
        mpz_clear(gN); mpz_clear(ga); mpz_clear(gb);

        /* Compute np0 from N */
        cpu_mont_np0_compute(&np0, shared_N.data(), LIM);

        /* Print header */
        const char *isa_name = use_avx512 ? "AVX512F" : "AVX2";
        std::printf("=== CPU Montgomery Mul Benchmark ===\n");
        std::printf("  Case:         %s\n", case_idx == 0 ? "large-inputs" : "small-inputs");
        std::printf("  ISA:          %s\n", isa_name);
        std::printf("  Bit-width:    %u\n", bits);
        std::printf("  Limbs:        %u\n", LIM);
        std::printf("  Instances:    %u\n", total_instances);
        std::printf("  Threads:      %u\n", num_threads);
        std::printf("  K/thread:     %u (ipt)\n", ipt);
        std::printf("  Iterations:   %u\n", iterations);
        std::printf("  Repeats:      %u\n", repeats);
        std::printf("  Total batches: %u\n", iterations * repeats);
        std::printf("  Affinity:     %s\n", affinity.c_str());
        if (sync_barrier) std::printf("  Barrier sync: enabled\n");
        std::printf("\n");

        /* Prepare thread contexts and fill per-thread buffers with broadcast a,b */
        for (uint32_t t = 0; t < num_threads; ++t) {
            ThreadCtx &ctx = thread_ctxs[t];
            ctx.thread_idx = t;
            ctx.K = ipt;
            ctx.LIM = LIM;
            ctx.stride = LIM;
            ctx.iterations = iterations;
            ctx.repeats = repeats;
            ctx.use_avx512 = use_avx512;
            ctx.verify = do_verify;
            ctx.sync_barrier = sync_barrier;
            ctx.shared_N = shared_N.data();
            ctx.np0 = np0;
            ctx.num_threads = (int)num_threads;
            ctx.barrier_count = &barrier_count;
            ctx.barrier_mtx = &barrier_mtx;
#ifdef _WIN32
            ctx.barrier_cv = &barrier_cv;
#else
            ctx.barrier_cv = &barrier_cv;
            ctx.barrier_pmtx = &barrier_pmtx;
#endif

            ctx.a.resize((size_t)ipt * LIM);
            ctx.b.resize((size_t)ipt * LIM);
            ctx.out.resize((size_t)ipt * LIM);
            for (uint32_t k = 0; k < ipt; ++k) {
                std::memcpy(&ctx.a[k * LIM], a_words.data(), LIM * sizeof(uint32_t));
                std::memcpy(&ctx.b[k * LIM], b_words.data(), LIM * sizeof(uint32_t));
            }
        }
    }

    /* Apply main thread affinity */
    if (affinity_auto) {
        if (apply_affinity_core(0)) {
            if (verbose) std::printf("  Main thread affinity: core 0\n");
        }
    } else if (affinity_none) {
        apply_affinity_any();
    } else {
        apply_affinity_list(affinity_cpus, 0);
    }

    /* self-test (single-threaded, then multi-threaded) */
    if (do_verify) {
        if (!run_selftest(LIM, ipt, LIM, shared_N.data(), np0, use_avx512, -1, 10))
            return 1;
        std::printf("\n");
    }

    /* Launch worker threads */
    double t_wall_start = now_sec();

    for (uint32_t t = 1; t < num_threads; ++t) {
        threads.emplace_back([&thread_ctxs, affinity_auto, affinity_none, &affinity_cpus](uint32_t idx) {
            if (affinity_auto) {
                apply_affinity_core(idx);
            } else if (affinity_none) {
                apply_affinity_any();
            } else {
                apply_affinity_list(affinity_cpus, idx);
            }
#ifdef _WIN32
            SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
#endif
            thread_benchmark(thread_ctxs[idx]);
        }, t);
    }

    /* Thread 0 runs in main thread */
    {
        ThreadCtx &ctx = thread_ctxs[0];
#ifdef _WIN32
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
#endif
        thread_benchmark(ctx);
    }

    for (auto &th : threads) th.join();

    double t_wall_end = now_sec();
    double wall_elapsed = t_wall_end - t_wall_start;

    /* Collect results */
    bool all_passed = true;
    double total_thread_elapsed = 0.0;
    uint64_t total_ops = 0;

    for (uint32_t t = 0; t < num_threads; ++t) {
        ThreadResult &res = thread_ctxs[t].result;
        if (!res.selftest_passed) all_passed = false;
        total_ops += res.ops;
        if (res.elapsed_sec > total_thread_elapsed)
            total_thread_elapsed = res.elapsed_sec;
    }

    if (do_verify && !all_passed) {
        std::printf("  Self-test FAILED in one or more threads.\n");
        return 1;
    }

    /* Print results */
    std::printf("  === Results ===\n");
    std::printf("  Wall-clock:    %.3f s\n", wall_elapsed);
    std::printf("  Max thread:    %.3f s\n", total_thread_elapsed);
    std::printf("  Total ops:     %llu mont_mul\n", (unsigned long long)total_ops);
    std::printf("  Throughput:    %.3f M mont_mul/s\n", (double)total_ops / total_thread_elapsed / 1e6);

    /* Per-thread stats */
    if (num_threads > 1) {
        std::printf("\n  Per-thread:\n");
        for (uint32_t t = 0; t < num_threads; ++t) {
            ThreadResult &res = thread_ctxs[t].result;
            double thr_mops = (double)res.ops / res.elapsed_sec / 1e6;
            std::printf("    Thread %2u: %7.3f s, %7.3f M/s, %6llu ops",
                        t, res.elapsed_sec, thr_mops, (unsigned long long)res.ops);
            if (do_verify) std::printf(" [%s]", res.selftest_passed ? "PASS" : "FAIL");
            std::printf("\n");
        }
    }

    std::printf("\n");
    return 0;
}