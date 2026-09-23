// ============================================================================
// cpu_addsub_bench.cpp — portable CPU modular add/sub micro-benchmark.
//
// Runs `cpu_add_mod_unroll_<W>b` and `cpu_sub_mod_unroll_<W>b` for the
// width selected via --bits, plus `cpu_add_mod` (fused fallback) for all
// widths.  Reports ops/s per variant.
//
// Multi-threading & affinity:
//   -t, --threads N          Number of threads (default: 1, or affinity count)
//   -a, --affinity 1,3,5,7   Pin thread t to core c_t
//
// Instances-per-thread (IPT):
//   -i, --ipt N              Instances per thread (default: 16, avx512 lane width)
//   total_instances = ipt * threads
//
// Kernel iterations support scientific notation: -k 1e6
//
// No OpenCL dependency.  Requires GMP for operand generation.
// ============================================================================

#include "cpu_addsub_impl.h"

#include "opencl_ecm_runtime_config.h"

#include "cpu_info.h"

#include <gmp.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#else
#include <pthread.h>
#include <sched.h>
#endif

namespace {

constexpr uint32_t MAX_BENCH_BITS  = 16384;
constexpr uint32_t MAX_BENCH_WORDS = MAX_BENCH_BITS / 32;
constexpr int DEFAULT_IPT          = 16; // matches avx2/avx512 lane width

void fill_from_gmp(const mpz_t v, uint32_t *out, size_t words) {
    mpz_t tmp, mod;
    mpz_init(tmp);
    mpz_init(mod);
    mpz_ui_pow_ui(mod, 2, (unsigned long)(words * 32));
    mpz_mod(tmp, v, mod);
    size_t count = 0;
    mpz_export(out, &count, -1, sizeof(uint32_t), 0, 0, tmp);
    for (size_t i = count; i < words; ++i) out[i] = 0u;
    mpz_clear(tmp);
    mpz_clear(mod);
}

// ── Affinity helpers ──────────────────────────────────────────────────

std::vector<int> parse_affinity(const std::string &s) {
    std::vector<int> cores;
    std::istringstream ss(s);
    std::string token;
    while (std::getline(ss, token, ',')) {
        try { cores.push_back(std::stoi(token)); }
        catch (...) { return {}; }
    }
    return cores;
}

bool pin_thread_to_core(int core) {
#ifdef _WIN32
    DWORD_PTR mask = 1ull << core;
    return SetThreadAffinityMask(GetCurrentThread(), mask) != 0;
#else
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(core, &cpuset);
    return pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset) == 0;
#endif
}

// ── CPU info query (verbose) ──────────────────────────────────────────

static void print_cpu_info_wrapper(bool verbose) {
    print_cpu_info(verbose,
        "  Benchmark ISA requirements:\n"
        "    scalar:     SSE2 (always)\n"
        "    AVX2 SoA:   AVX2+FMA (8-lane SIMD)\n"
        "    AVX512 SoA: AVX512F+AVX512DQ (16-lane SIMD)");
}

// ── Multi-threaded scalar runner (inline, returns by ref) ──────────────

struct ThreadRange {
    int inst_start;
    int inst_end;
    int core;
};

void thread_addsub_run(bool is_add,
                       uint32_t *r, const uint32_t *a, const uint32_t *b,
                       const uint32_t *N, int iters, int width_words,
                       ThreadRange range, double &thread_ms) {
    if (range.core >= 0) pin_thread_to_core(range.core);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int inst_idx = range.inst_start; inst_idx < range.inst_end; ++inst_idx) {
        uint32_t *r_i = r + inst_idx * width_words;
        const uint32_t *a_i = a + inst_idx * width_words;
        const uint32_t *b_i = b + inst_idx * width_words;
        for (int it = 0; it < iters; ++it) {
            if (is_add)
                cpu_add_fused_scalar(r_i, (it == 0) ? a_i : r_i, b_i, N, (uint32_t)width_words);
            else
                cpu_sub_fused_scalar(r_i, (it == 0) ? a_i : r_i, b_i, N, (uint32_t)width_words);
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    thread_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
}

void run_addsub_mt_inline(bool is_add,
                          uint32_t *r_buf, const uint32_t *a_buf, const uint32_t *b_buf,
                          const uint32_t *N, int iters, int ipt, int width_words,
                          int repeats, int num_threads, const std::vector<int> &affinity,
                          double &best_ms) {
    int total_inst = ipt * num_threads;
    best_ms = 1e12;
    for (int rep = 0; rep < repeats; ++rep) {
        std::memcpy(r_buf, a_buf, (size_t)width_words * (size_t)total_inst * sizeof(uint32_t));
        std::vector<ThreadRange> ranges(num_threads);
        int base = 0;
        for (int t = 0; t < num_threads; ++t) {
            int share = (total_inst - base + num_threads - t - 1) / (num_threads - t);
            ranges[t].inst_start = base;
            ranges[t].inst_end   = base + share;
            ranges[t].core       = (t < (int)affinity.size()) ? affinity[t] : -1;
            base += share;
        }
        std::vector<std::thread> threads;
        std::vector<double> thread_times(num_threads, 0.0);
        for (int t = 1; t < num_threads; ++t)
            threads.emplace_back(thread_addsub_run, is_add, r_buf, a_buf, b_buf, N,
                                 iters, width_words, ranges[t], std::ref(thread_times[t]));
        thread_addsub_run(is_add, r_buf, a_buf, b_buf, N,
                          iters, width_words, ranges[0], thread_times[0]);
        for (auto &th : threads) th.join();
        double max_ms = 0;
        for (int t = 0; t < num_threads; ++t)
            if (thread_times[t] > max_ms) max_ms = thread_times[t];
        if (max_ms < best_ms) best_ms = max_ms;
    }
}

// ── Generic SoA runner (function-pointer dispatch) ──────────────────
// All AVX2/AVX512 SoA add/sub kernels share this signature.  The caller
// selects the kernel and must guarantee the CPU supports its ISA at runtime
// (AVX512 kernels are illegal instructions on AVX2-only CPUs).
using SoaKernelFn = void (*)(uint32_t *, const uint32_t *, const uint32_t *,
                             const uint32_t *, uint32_t);

void run_soa_kernel(SoaKernelFn kernel, int K,
                    uint32_t *r_buf, uint32_t *a_buf, uint32_t *b_buf,
                    const uint32_t *n_buf,
                    int iters, int ipt, int width_words,
                    int repeats, int num_threads, const std::vector<int> &affinity,
                    double &best_ms) {
    int total_inst = ipt * num_threads;
    int n_batches = total_inst / K;
    best_ms = 1e12;
    for (int rep = 0; rep < repeats; ++rep) {
        std::memcpy(r_buf, a_buf, (size_t)width_words * total_inst * sizeof(uint32_t));
        std::vector<int> thread_nbatches(num_threads);
        std::vector<int> thread_cores(num_threads);
        int batch_base = 0;
        for (int t = 0; t < num_threads; ++t) {
            int share = (n_batches - batch_base + num_threads - t - 1) / (num_threads - t);
            thread_nbatches[t] = share;
            thread_cores[t] = (t < (int)affinity.size()) ? affinity[t] : -1;
            batch_base += share;
        }
        auto thread_soa_run = [&](int t, double &t_ms) {
            if (thread_cores[t] >= 0) pin_thread_to_core(thread_cores[t]);
            int my_start = 0;
            for (int pt = 0; pt < t; ++pt) my_start += thread_nbatches[pt];
            int my_n = thread_nbatches[t];
            if (my_n == 0) { t_ms = 0; return; }
            size_t per_batch = (size_t)width_words * K;
            uint32_t *al = new uint32_t[per_batch * my_n];
            uint32_t *bl = new uint32_t[per_batch * my_n];
            uint32_t *rl = new uint32_t[per_batch * my_n];
            for (int b = 0; b < my_n; ++b) {
                int gb = my_start + b, base = gb * K;
                uint32_t *ap = al + b * width_words * K;
                uint32_t *bp = bl + b * width_words * K;
                for (uint32_t limb = 0; limb < (uint32_t)width_words; ++limb) {
                    for (int inst = 0; inst < K; ++inst)
                        ap[limb * K + inst] = a_buf[(base + inst) * width_words + limb];
                    for (int inst = 0; inst < K; ++inst)
                        bp[limb * K + inst] = b_buf[(base + inst) * width_words + limb];
                }
            }
            std::memcpy(rl, al, per_batch * my_n * sizeof(uint32_t));
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int it = 0; it < iters; ++it) {
                for (int b = 0; b < my_n; ++b) {
                    uint32_t *rp = rl + b * width_words * K;
                    const uint32_t *ap = al + b * width_words * K;
                    const uint32_t *bp = bl + b * width_words * K;
                    kernel(rp, (it == 0) ? ap : rp, bp, n_buf, (uint32_t)width_words);
                }
            }
            auto t1 = std::chrono::high_resolution_clock::now();
            t_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            for (int b = 0; b < my_n; ++b) {
                int gb = my_start + b, base = gb * K;
                uint32_t *rp = rl + b * width_words * K;
                for (uint32_t limb = 0; limb < (uint32_t)width_words; ++limb)
                    for (int inst = 0; inst < K; ++inst)
                        r_buf[(base + inst) * width_words + limb] = rp[limb * K + inst];
            }
            delete[] al; delete[] bl; delete[] rl;
        };
        std::vector<std::thread> threads;
        std::vector<double> thread_ms(num_threads, 0.0);
        for (int t = 1; t < num_threads; ++t)
            threads.emplace_back(thread_soa_run, t, std::ref(thread_ms[t]));
        thread_soa_run(0, thread_ms[0]);
        for (auto &th : threads) th.join();
        double max_ms = 0;
        for (int t = 0; t < num_threads; ++t)
            if (thread_ms[t] > max_ms) max_ms = thread_ms[t];
        if (max_ms < best_ms) best_ms = max_ms;
    }
}
} // namespace

bool runCpuAddSubBenchmark(int bits, int kernel_iterations, int ipt, int launch_repeats,
                           int num_threads, const std::vector<int> &affinity,
                           bool no_overflow = false, bool verbose = false,
                           bool run_scalar = true, bool run_avx2 = true,
                           bool run_avx512 = true) {
    if (bits <= 0 || (bits % 32) != 0 || (uint32_t)bits > MAX_BENCH_BITS) {
        std::cerr << "bits must be a positive multiple of 32 and <= " << MAX_BENCH_BITS << std::endl;
        return false;
    }
    const uint32_t WORDS = (uint32_t)bits / 32u;
    const int effective_threads = num_threads > 0 ? num_threads : 1;
    const int total_inst = ipt * effective_threads;
    const bool use_affinity = !affinity.empty();

    std::cout << "CPU add/sub microbench: " << bits << " bits (" << WORDS << " words), "
              << kernel_iterations << " kernel_iters, "
              << ipt << " ipt * " << effective_threads << "t = " << total_inst << " instances, "
              << launch_repeats << " repeats";
    if (use_affinity) {
        std::cout << ", affinity=";
        for (size_t i = 0; i < affinity.size(); ++i) {
            if (i > 0) std::cout << ",";
            std::cout << affinity[i];
        }
    }
    std::cout << "\n";

    print_cpu_info_wrapper(verbose);

    // Runtime ISA detection — the binary is compiled with an AVX2 baseline and
    // dispatches to the isolated AVX512 kernels only when the CPU supports them.
    const CpuIsa isa = detect_cpu_isa();
    const bool has_avx2   = isa.has(ISA_AVX2) && isa.has(ISA_FMA);
    const bool has_avx512 = isa.has(ISA_AVX512F) && isa.has(ISA_AVX512DQ);

    // SoA batch sizes (always available for display; ISA guards inside runner)
    const int K_AVX512 = 16;
    const int K_AVX2 = 8;

    // Two test cases per bit-width (same seed scheme as OpenCL):
    //   0 = a+b >= N (overflow — fused speculative subtract triggers)
    //   1 = a+b <  N (no-overflow — sum stays below N)
    struct BenchCase {
        const char *label;
        mpz_t a_gmp, b_gmp, n_gmp;
    };
    BenchCase cases[2] = {};
    {
        gmp_randstate_t rng;
        gmp_randinit_default(rng);
        for (int ic = 0; ic < 2; ++ic) {
            gmp_randseed_ui(rng, (unsigned long)((uint32_t)bits * 31337u + (unsigned)ic * 0x9e3779b9u));
            cases[ic].label = (ic == 0) ? "overflow (a+b>=N)" : "no-overflow (a+b<N)";
            mpz_init(cases[ic].a_gmp); mpz_init(cases[ic].b_gmp); mpz_init(cases[ic].n_gmp);
            mpz_t &N = cases[ic].n_gmp, &a = cases[ic].a_gmp, &b = cases[ic].b_gmp;
            mpz_urandomb(N, rng, (unsigned long)bits); mpz_setbit(N, bits-1); mpz_setbit(N, 0);
            if (ic == 0) {  // overflow
                mpz_t half; mpz_init(half); mpz_tdiv_q_ui(half, N, 2u);
                mpz_urandomm(a, rng, half); mpz_add(a, a, half);
                mpz_urandomm(b, rng, N);
                mpz_t sum; mpz_init(sum); mpz_add(sum, a, b);
                if (mpz_cmp(sum, N) < 0) { mpz_sub(b, N, a); mpz_sub_ui(b, b, 1u); }
                mpz_clear(sum); mpz_clear(half);
            } else {  // no-overflow
                mpz_t quar; mpz_init(quar); mpz_tdiv_q_ui(quar, N, 4u);
                mpz_urandomm(a, rng, quar); mpz_urandomm(b, rng, quar);
                mpz_clear(quar);
            }
        }
        gmp_randclear(rng);
    }

    const int case_idx = no_overflow ? 1 : 0;
    std::cout << "  [" << cases[case_idx].label << "]\n";
    mpz_t &a_gmp = cases[case_idx].a_gmp;
    mpz_t &b_gmp = cases[case_idx].b_gmp;
    mpz_t &n_gmp = cases[case_idx].n_gmp;

    size_t buf_words = (size_t)total_inst * WORDS;
    uint32_t *a_buf = new uint32_t[buf_words];
    uint32_t *b_buf = new uint32_t[buf_words];
    uint32_t *n_buf = new uint32_t[WORDS];
    uint32_t *r_buf = new uint32_t[buf_words];
    fill_from_gmp(n_gmp, n_buf, WORDS);
    // AoS fill: [inst * WORDS + limb]
    uint32_t *aos_template = new uint32_t[WORDS];
    uint32_t *bos_template = new uint32_t[WORDS];
    fill_from_gmp(a_gmp, aos_template, WORDS);
    fill_from_gmp(b_gmp, bos_template, WORDS);
    for (int i = 0; i < total_inst; ++i) {
        std::memcpy(a_buf + i * WORDS, aos_template, WORDS * sizeof(uint32_t));
        std::memcpy(b_buf + i * WORDS, bos_template, WORDS * sizeof(uint32_t));
        std::memcpy(r_buf + i * WORDS, aos_template, WORDS * sizeof(uint32_t));
    }
    delete[] aos_template;
    delete[] bos_template;

    // op_count per timing sample = kernel_iters × total_inst (single repeat)
    const double op_count = (double)kernel_iterations * (double)total_inst;

    auto print_line = [&](const std::string &label, double ms) {
        if (ms <= 0.0) {
            std::cout << "  " << label << ": N/A (no ISA)\n";
        } else {
            double ops_s = op_count / (ms / 1000.0);
            std::cout << "  " << label << ": " << ms << " ms, " << ops_s << " ops/s";
            if (effective_threads > 1) std::cout << " (" << effective_threads << "t)";
            std::cout << "\n";
        }
    };

    auto print_skip = [&](const std::string &label, int req_k) {
        std::cout << "  " << label << ": skipped (need multiple of " << req_k
                  << " instances, got " << total_inst << ")\n";
    };

    std::cout << "\n";

    // ── Scalar fused ─────────────────────────────────────────────────

    if (run_scalar) {
        double t_add = 0, t_sub = 0;
        run_addsub_mt_inline(true, r_buf, a_buf, b_buf, n_buf,
                             kernel_iterations, ipt, (int)WORDS,
                             launch_repeats, effective_threads, affinity, t_add);
        run_addsub_mt_inline(false, r_buf, a_buf, b_buf, n_buf,
                             kernel_iterations, ipt, (int)WORDS,
                             launch_repeats, effective_threads, affinity, t_sub);
        std::cout << "  [scalar]\n";
        print_line("cpu_add_fused",       t_add);
        print_line("cpu_sub_fused",       t_sub);
    }

    // Per-ISA test matrix: add/sub × {soa, mask, blend, mb(=mask+blend)}.
    struct SoaItem { const char *label; SoaKernelFn fn; };

    auto run_soa_group = [&](const SoaItem *items, int n, int K,
                             bool has_isa, const char *isa_note) {
        bool can = (total_inst % K == 0);
        if (has_isa && can) {
            for (int i = 0; i < n; ++i) {
                double t = 0;
                run_soa_kernel(items[i].fn, K, r_buf, a_buf, b_buf, n_buf,
                               kernel_iterations, ipt, (int)WORDS,
                               launch_repeats, effective_threads, affinity, t);
                print_line(items[i].label, t);
            }
        } else if (!has_isa) {
            for (int i = 0; i < n; ++i)
                std::cout << "  " << items[i].label << ": skipped (" << isa_note << ")\n";
        } else {
            for (int i = 0; i < n; ++i) print_skip(items[i].label, K);
        }
    };

    // ── AVX2 SoA (requires 8 | total_inst) ──────────────────────────

    if (run_avx2) {
        std::cout << "  [AVX2 SoA]\n";
        const SoaItem avx2_items[] = {
            {"cpu_add_avx2_soa",   cpu_add_fused_avx2_soa},
            {"cpu_sub_avx2_soa",   cpu_sub_fused_avx2_soa},
            {"cpu_add_avx2_mask",  cpu_add_fused_avx2_soa_mask},
            {"cpu_sub_avx2_mask",  cpu_sub_fused_avx2_soa_mask},
            {"cpu_add_avx2_blend", cpu_add_fused_avx2_soa_blend},
            {"cpu_sub_avx2_blend", cpu_sub_fused_avx2_soa_blend},
            {"cpu_add_avx2_mb",    cpu_add_fused_avx2_soa_mask_blend},
            {"cpu_sub_avx2_mb",    cpu_sub_fused_avx2_soa_mask_blend},
        };
        run_soa_group(avx2_items, 8, K_AVX2, has_avx2, "CPU lacks AVX2/FMA");
    }

    // ── AVX512 SoA (requires 16 | total_inst) ───────────────────────

    if (run_avx512) {
        std::cout << "  [AVX512 SoA]\n";
        const SoaItem avx512_items[] = {
            {"cpu_add_avx512_soa",   cpu_add_fused_avx512_soa},
            {"cpu_sub_avx512_soa",   cpu_sub_fused_avx512_soa},
            {"cpu_add_avx512_mask",  cpu_add_fused_avx512_soa_mask},
            {"cpu_sub_avx512_mask",  cpu_sub_fused_avx512_soa_mask},
            {"cpu_add_avx512_blend", cpu_add_fused_avx512_soa_blend},
            {"cpu_sub_avx512_blend", cpu_sub_fused_avx512_soa_blend},
            {"cpu_add_avx512_mb",    cpu_add_fused_avx512_soa_mask_blend},
            {"cpu_sub_avx512_mb",    cpu_sub_fused_avx512_soa_mask_blend},
        };
        run_soa_group(avx512_items, 8, K_AVX512, has_avx512, "CPU lacks AVX512F/DQ");
    }

    // ── Optional CSV ─────────────────────────────────────────────────
    const std::string &csv_path = ecm_runtime_config().bench_csv;
    if (!csv_path.empty()) {
        std::ofstream csv(csv_path, std::ios::app);
        // CSV disabled for immediate-output mode (restore when needed)
        (void)csv;
    }

    delete[] a_buf;
    delete[] b_buf;
    delete[] n_buf;
    delete[] r_buf;
    for (int ic = 0; ic < 2; ++ic) {
        mpz_clear(cases[ic].a_gmp);
        mpz_clear(cases[ic].b_gmp);
        mpz_clear(cases[ic].n_gmp);
    }
    return true;
}

// ── main ────────────────────────────────────────────────────────────────
#ifdef BUILD_CPU_ADDSUB_MAIN

namespace {

/// Parse a CLI value that may be in scientific notation (e.g. 1e6, 5e5).
/// Returns the rounded integer, or the parsed int when no exponent.
bool parse_cli_count(const char *s, const char *label, int &out) {
    if (s == nullptr || *s == '\0') return true;
    try {
        double d = std::stod(s); // handles 1e6, 5e5, 1000
        if (!std::isfinite(d) || d < 0.0) {
            std::cerr << "Invalid " << label << ": " << s << std::endl;
            return false;
        }
        out = (int)(d + 0.5);
        return true;
    } catch (...) {
        std::cerr << "Invalid " << label << ": " << s << std::endl;
        return false;
    }
}

bool parse_cli_int(const char *s, const char *label, int &out) {
    return parse_cli_count(s, label, out);
}

} // namespace

int main(int argc, char **argv) {
    int bits = 1024;
    int kernel_iterations = 1000;
    int ipt = DEFAULT_IPT;
    int launch_repeats = 10;
    bool no_overflow = false;
    bool verbose = false;
    int num_threads = 0; // 0 = auto (1 if no affinity, else aff.size())
    std::string affinity_str;
    bool only_avx2 = false;    // --avx2  → run only AVX2 group
    bool only_avx512 = false;  // --avx512 → run only AVX512 group

    auto print_usage = [&]() {
        std::cout
            << "Usage: cpu_addsub_bench [options] [bits] [kernel_iterations] [ipt] [repeats]\n"
            << "  Positional args:\n"
            << "    bits                    Benchmark bit width (mult of 32, <= " << MAX_BENCH_BITS << ", default: 1024)\n"
            << "    kernel_iterations       Kernel inner-loop count; supports 1e6 notation (default: 1000)\n"
            << "    ipt                     Instances per thread (default: " << DEFAULT_IPT << ")\n"
            << "    repeats                 Measurement repeats for averaging (default: 10)\n"
            << "  Options:\n"
            << "  -b, --bits <bits>        Alias for 1st positional\n"
            << "  -k, --kernel-iters <N>   Alias for 2nd positional; supports 1e6\n"
            << "  -i, --ipt <N>            Alias for 3rd positional\n"
            << "  -t, --threads <N>        Number of threads (default: 1, or affinity count)\n"
            << "  -r, --repeats <N>        Alias for 4th positional\n"
            << "  -a, --affinity c1,c2,... Pin thread t to core c_t\n"
            << "  --no-overflow            Use a+b < N test data (default: a+b >= N)\n"
            << "  --avx2                   Run only the AVX2 group (skip scalar/AVX512)\n"
            << "  --avx512                 Run only the AVX512 group (skip scalar/AVX2)\n"
            << "                           (combine --avx2 --avx512 to run both, skip scalar)\n"
            << "  -v, --verbose            Print CPU info and ISA details\n"
            << "  --csv <file>             Append results CSV\n"
            << "  -h, --help               Show this help\n"
            << "\nExamples:\n"
            << "  cpu_addsub_bench 512 1e6 16                          # latency + overflow case\n"
            << "  cpu_addsub_bench 512 1e6 16 5 --no-overflow          # latency + no-overflow case\n"
            << "  cpu_addsub_bench 512 5e4 16 5 -t 12 -a 1,3,5,7,9,11,13,15,17,19,21,23\n"
            << "                                                        # throughput: 12 threads, 192 total instances\n";
    };

    std::vector<std::string> pos;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-h" || a == "--help") { print_usage(); return EXIT_SUCCESS; }
        if (a == "-b" || a == "--bits") {
            if (i + 1 >= argc) { std::cerr << "Missing value for " << a << "\n"; return EXIT_FAILURE; }
            if (!parse_cli_int(argv[++i], "bits", bits)) return EXIT_FAILURE;
            continue;
        }
        if (a == "-k" || a == "--kernel-iters") {
            if (i + 1 >= argc) { std::cerr << "Missing value for " << a << "\n"; return EXIT_FAILURE; }
            if (!parse_cli_count(argv[++i], "--kernel-iters", kernel_iterations)) return EXIT_FAILURE;
            continue;
        }
        if (a == "-i" || a == "--ipt") {
            if (i + 1 >= argc) { std::cerr << "Missing value for " << a << "\n"; return EXIT_FAILURE; }
            if (!parse_cli_int(argv[++i], "--ipt", ipt)) return EXIT_FAILURE;
            continue;
        }
        if (a == "-t" || a == "--threads") {
            if (i + 1 >= argc) { std::cerr << "Missing value for " << a << "\n"; return EXIT_FAILURE; }
            if (!parse_cli_int(argv[++i], "--threads", num_threads)) return EXIT_FAILURE;
            continue;
        }
        if (a == "-r" || a == "--repeats") {
            if (i + 1 >= argc) { std::cerr << "Missing value for " << a << "\n"; return EXIT_FAILURE; }
            if (!parse_cli_int(argv[++i], "--repeats", launch_repeats)) return EXIT_FAILURE;
            continue;
        }
        if (a == "-a" || a == "--affinity" || a == "--aff") {
            if (i + 1 >= argc) { std::cerr << "Missing value for " << a << "\n"; return EXIT_FAILURE; }
            affinity_str = argv[++i];
            continue;
        }
        if (a == "--no-overflow") { no_overflow = true; continue; }
        if (a == "--avx2") { only_avx2 = true; continue; }
        if (a == "--avx512") { only_avx512 = true; continue; }
        if (a == "-v" || a == "--verbose") { verbose = true; continue; }
        if (a == "--csv") { if (i + 1 < argc) ecm_runtime_config().bench_csv = argv[++i]; continue; }
        if (!a.empty() && a[0] == '-') { std::cerr << "Unknown option: " << a << " (use --help)\n"; return EXIT_FAILURE; }
        pos.push_back(a);
    }
    if (pos.size() >= 1 && !parse_cli_count(pos[0].c_str(), "bits", bits)) return EXIT_FAILURE;
    if (pos.size() >= 2 && !parse_cli_count(pos[1].c_str(), "kernel_iterations", kernel_iterations)) return EXIT_FAILURE;
    if (pos.size() >= 3 && !parse_cli_count(pos[2].c_str(), "ipt", ipt)) return EXIT_FAILURE;
    if (pos.size() >= 4 && !parse_cli_count(pos[3].c_str(), "launch_repeats", launch_repeats)) return EXIT_FAILURE;

    // Resolve threads/affinity
    std::vector<int> affinity_cores = affinity_str.empty() ? std::vector<int>() : parse_affinity(affinity_str);
    if (!affinity_str.empty() && affinity_cores.empty()) {
        std::cerr << "Invalid --affinity format (expect comma-separated ints)\n";
        return EXIT_FAILURE;
    }
    if (num_threads <= 0) num_threads = affinity_cores.empty() ? 1 : (int)affinity_cores.size();

    // ISA-group selection: if any of --avx2/--avx512 is given, run ONLY the
    // selected group(s) and skip the rest.  Otherwise run everything.
    const bool select_active = only_avx2 || only_avx512;
    const bool run_scalar = !select_active;
    const bool run_avx2   = select_active ? only_avx2   : true;
    const bool run_avx512 = select_active ? only_avx512 : true;

    bool ok = runCpuAddSubBenchmark(bits, kernel_iterations, ipt, launch_repeats,
                                    num_threads, affinity_cores,
                                    no_overflow, verbose,
                                    run_scalar, run_avx2, run_avx512);
    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
#endif // BUILD_CPU_ADDSUB_MAIN
