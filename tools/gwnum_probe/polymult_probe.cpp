/*----------------------------------------------------------------------
| polymult_probe.cpp -- measures prime95's polymult library at the shapes
| an ECM stage 2 for a SMALL modulus (N <= 10 000 bits) actually uses.
|
| Why: Prime95's ECM stage 2 is built from gwnum modular arithmetic plus
| this polynomial-multiplication library (ecm.cpp includes polymult.h and
| calls polymult/polymult_fma). Our own stage 2 would have to run at least
| this fast, so this probe measures the engine directly instead of guessing:
|
|   * gwfftlen for the modulus, and polymult_fft_size(n) for the poly size
|   * polymult_safety_margin / polymult_mem_required (how much memory one
|     polymult of this shape needs -- Prime95 runs with Memory=12288)
|   * time per polymult: first call (includes plan building) vs steady state
|     with POLYMULT_SAVE_PLAN/POLYMULT_USE_PLAN (what stage 2 does in a loop)
|   * 1 / 4 / 8 / 24 threads
|   * a correctness check of the convolution against GMP for a small size
|
| Build/run: tools/gwnum_probe/build_and_run.ps1 -Polymult <p> <size> [iters]
+---------------------------------------------------------------------*/

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

extern "C" {
#include "cpuid.h"
#include "gwnum.h"
#include "polymult.h"
}

#include "gmp.h"

namespace {

double now_us() {
    using clock = std::chrono::steady_clock;
    static const clock::time_point t0 = clock::now();
    return std::chrono::duration<double, std::micro>(clock::now() - t0).count();
}

} /* namespace */

int main(int argc, char **argv) {
    /* argv: <p> <poly_size> [iters] [max_threads] -- one shape per run, so a huge shape
       cannot turn the whole run into a hang (measured: the 132480-coefficient shape with
       four thread settings needs minutes, not seconds). */
    const int p = (argc > 1) ? std::atoi(argv[1]) : 5153;      /* Mersenne exponent */
    const uint64_t req_size = (argc > 2) ? std::strtoull(argv[2], nullptr, 10) : 720;
    const uint64_t sizes[] = {req_size};
    const int iters = (argc > 3) ? std::atoi(argv[3]) : 2;
    const int max_threads = (argc > 4) ? std::atoi(argv[4]) : 4;

    guessCpuType();
    guessCpuSpeed();

    gwhandle gwdata;
    gwinit(&gwdata);
    /* These MUST come after gwinit and BEFORE gwsetup (measured the hard way: without
       gwset_using_polymult the per-gwnum header size differs from what polymult assumes, and
       the first polymult hangs instead of failing loudly).  Mirrors ecm.cpp:7951-7971. */
    /* The documented protocol (polymult.h:28-37): the FFT length must be large enough that
       EXTRA_BITS/2 > polymult_safety_margin(P,P).  Prime95 binary-searches this
       (max_safe_poly2_size, ecm.cpp:5453-5471); if it is not satisfied the planner does not
       fail -- it hangs (measured 2026-09-30). */
    const float need_margin = polymult_safety_margin(req_size, req_size);
    int setup_rc = 0;
    for (int extra = 0; extra < 6; ++extra) {
        gwinit(&gwdata);
        gwset_using_polymult(&gwdata);      /* gwnum.h:217 -> GW_LARGE_HEADER_SIZE */
        gwset_num_threads(&gwdata, 1);
        gwset_polymult_safety_margin(&gwdata, need_margin);
        gwset_larger_fftlen_count(&gwdata, extra);
        setup_rc = gwsetup(&gwdata, 1.0, 2, static_cast<unsigned long>(p), -1);
        if (setup_rc != 0) break;
        std::printf("  gwsetup try %d: fftlen=%lu EXTRA_BITS=%.3f  need=%.3f  %s\n", extra,
                    (unsigned long)gwfftlen(&gwdata), (double)gwdata.EXTRA_BITS,
                    (double)need_margin, gw_passes_safety_margin(&gwdata, need_margin) ? "OK" : "not enough");
        std::fflush(stdout);
        if (gw_passes_safety_margin(&gwdata, need_margin)) break;
    }
    if (setup_rc != 0) {
        std::printf("gwsetup failed (%d)\n", setup_rc);
        return 1;
    }
    char desc[512];
    gwfft_description(&gwdata, desc);
    std::printf("polymult probe -- modulus 2^%d-1 (%d bits)\n", p, p);
    std::printf("  gwnum: %s  |  gwfftlen=%lu  |  gwnum_size=%lu B  gwmemused=%lu B\n",
                desc, (unsigned long)gwfftlen(&gwdata), (unsigned long)gwnum_size(&gwdata),
                (unsigned long)gwmemused(&gwdata));
    std::printf("  cpu: %s  threads=%u\n\n", CPU_BRAND, CPU_CORES);

    /* FFT(1) is used by the monic post-processing; polymult_init only creates it when k != 1
       or the modulus is a general mod, so for a Mersenne modulus WE must create it. */
    gwuser_init_FFT1(&gwdata);
    std::printf("[step] gwuser_init_FFT1 done\n"); std::fflush(stdout);
    pmhandle pm;
    std::printf("[step] polymult_init...\n"); std::fflush(stdout);
    polymult_init(&pm, &gwdata);
    std::printf("[step] polymult_init done\n"); std::fflush(stdout);
    /* REQUIRED: polymult_init memsets the handle, so every tuning threshold
       (two_pass_start / mt_ffts_start / KARAT_BREAK / FFT_BREAK / strided_writes_end ...)
       is zero until this is called -- with zeros the planner never terminates and the first
       polymult hangs forever (measured 2026-09-30; prime95 calls this at ecm.cpp:7834-7835).
       Cache numbers are this machine's: 1 MB L2 per core, 24 MB L3. */
    polymult_default_tuning(&pm, 1024, 24576);
    polymult_set_cpu_flags(&pm, CPU_FLAGS);
    /* Also REQUIRED before the first polymult: polymult_init memsets num_threads/max_num_threads
       to 0, and a polymult with 0 threads never returns (measured). */
    polymult_set_max_num_threads(&pm, max_threads);
    /* polymult does NOT create its own worker threads: it dispatches LINE work to the helpers
       launched here (HELPER_POLYMULT_LINE, polymult.c:4463).  Without this call the first
       polymult dispatches work and never returns -- which is exactly the hang this probe hit
       (prime95 calls it at ecm.cpp:9148/9366/9748 before its first polymult of each phase). */
    polymult_launch_helpers(&pm);
    std::printf("[step] polymult_launch_helpers done\n"); std::fflush(stdout);
    std::printf("[step] default_tuning(L2=1024KB L3=24576KB) done\n"); std::fflush(stdout);
    /* The shipped default tuning assumes L2=256KB / L3=6MB; this machine's HX 370 has
       much larger caches, so report both so the difference is visible. */
    std::printf("  polymult default tuning: L2=256 KB L3=6144 KB (library default)\n\n");

    for (uint64_t n : sizes) {
        const uint64_t outn = 2 * n; /* monic: in1(in2) + in2 - 1 == outn */
        std::printf("[step] mem_required...\n"); std::fflush(stdout);
        uint64_t need = polymult_mem_required(&pm, n, n, POLYMULT_INVEC1_MONIC);
        std::printf("[step] mem_required=%llu B\n", (unsigned long long)need); std::fflush(stdout);
        const uint64_t pmfft = polymult_fft_size(outn);
        const float margin = polymult_safety_margin(n, n);
        const double mb = static_cast<double>(need) / (1024.0 * 1024.0);
        std::fflush(stdout);
        std::printf("== poly_size=%llu -> polymult_fft_size(2n)=%llu  safety_margin=%.3f  mem~%.0f MB\n",
                    (unsigned long long)n, (unsigned long long)pmfft, (double)margin, mb);
        if (mb > 20000.0) {
            std::printf("   skipped (memory guard)\n\n");
            continue;
        }

        gwnum *A = gwalloc_array(&gwdata, n);
        gwnum *B = gwalloc_array(&gwdata, n);
        gwnum *C = gwalloc_array(&gwdata, outn);
        if (A == nullptr || B == nullptr || C == nullptr) {
            std::printf("   allocation failed\n\n");
            continue;
        }
        std::printf("[step] filling %llu coefficients...\n", (unsigned long long)n); std::fflush(stdout);
        gw_random_number(&gwdata, A[0]);
        for (uint64_t i = 0; i < n; ++i) {
            gw_random_number(&gwdata, A[i]);
            gw_random_number(&gwdata, B[i]);
        }
        /* The first call builds a plan -- stage 2 pays this once, then loops
           with POLYMULT_USE_PLAN, so both numbers matter. */
        std::printf("[step] first polymult (incl. plan)...\n"); std::fflush(stdout);
        const double t_plan0 = now_us();
        polymult(&pm, A, n, B, n, C, outn, POLYMULT_INVEC1_MONIC | POLYMULT_SAVE_PLAN);
        const double t_first = now_us() - t_plan0;

        for (int threads : {1, 2, 4, 8}) {
            if (threads > max_threads || static_cast<unsigned>(threads) > CPU_CORES) continue;
            polymult_set_max_num_threads(&pm, threads);
            /* warm up with the plan, then time a batch */
            polymult(&pm, A, n, B, n, C, outn,
                     POLYMULT_INVEC1_MONIC | POLYMULT_USE_PLAN | POLYMULT_STARTNEXTFFT);
            const double t0 = now_us();
            for (int i = 0; i < iters; ++i) {
                polymult(&pm, A, n, B, n, C, outn,
                         POLYMULT_INVEC1_MONIC | POLYMULT_USE_PLAN | POLYMULT_STARTNEXTFFT);
            }
            const double per = (now_us() - t0) / iters;
            std::printf("   threads=%-3d %9.0f us/polymult  (%6.2f M coeff-ops/s)   [first call incl. plan: %.0f us]\n",
                        threads, per,
                        (2.0 * static_cast<double>(n) * static_cast<double>(n)) / per,
                        t_first);
        }
        gwfree_array(&gwdata, C);
        gwfree_array(&gwdata, B);
        gwfree_array(&gwdata, A);
        std::printf("\n");
    }

    /* Correctness: a small polymult must equal the schoolbook convolution of the
       coefficients (done in GMP).  Without this the timings prove nothing. */
    {
        const uint64_t n = 32;
        gwnum *A = gwalloc_array(&gwdata, n);
        gwnum *B = gwalloc_array(&gwdata, n);
        gwnum *C = gwalloc_array(&gwdata, 2 * n);
        std::vector<uint32_t> buf(static_cast<size_t>(p / 32) + 8);
        mpz_t *ca = new mpz_t[n], *cb = new mpz_t[n], *cc = new mpz_t[2 * n];
        for (uint64_t i = 0; i < n; ++i) {
            mpz_init(ca[i]);
            mpz_init(cb[i]);
            u64togw(&gwdata, 1000003ull * (i + 1), A[i]);
            u64togw(&gwdata, 7919ull * (i + 3), B[i]);
            mpz_set_ui(ca[i], 1000003ull * (i + 1));
            mpz_set_ui(cb[i], 7919ull * (i + 3));
        }
        for (uint64_t i = 0; i < 2 * n; ++i) mpz_init(cc[i]);
        mpz_t N;
        mpz_init(N);
        mpz_setbit(N, static_cast<mp_bitcnt_t>(p));
        mpz_sub_ui(N, N, 1); /* modulus of the handle: 2^p - 1 */
        polymult(&pm, A, n, B, n, C, 2 * n - 1, 0);
        int bad = 0;
        for (uint64_t i = 0; i < 2 * n - 1; ++i) {
            const long got = gwtobinary(&gwdata, C[i], buf.data(), static_cast<uint32_t>(buf.size()));
            mpz_t g;
            mpz_init(g);
            if (got > 0) mpz_import(g, static_cast<size_t>(got), -1, 4, 0, 0, buf.data());
            for (uint64_t j = 0; j < n; ++j) {
                if (i >= j && i - j < n) mpz_addmul(cc[i], ca[j], cb[i - j]);
            }
            mpz_mod(cc[i], cc[i], N); /* polymult coefficients are reduced mod the handle's modulus */
            if (mpz_cmp(g, cc[i]) != 0) ++bad;
            mpz_clear(g);
        }
        std::printf("correctness (n=32, vs GMP schoolbook convolution mod N): %s (%d bad of %llu)\n",
                    bad == 0 ? "OK" : "MISMATCHES PRESENT", bad, (unsigned long long)(2 * n - 1));
        mpz_clear(N);
        for (uint64_t i = 0; i < 2 * n; ++i) mpz_clear(cc[i]);
        for (uint64_t i = 0; i < n; ++i) { mpz_clear(ca[i]); mpz_clear(cb[i]); }
        delete[] ca; delete[] cb; delete[] cc;
        gwfree_array(&gwdata, C);
        gwfree_array(&gwdata, B);
        gwfree_array(&gwdata, A);
    }

    polymult_done(&pm);
    gwdone(&gwdata);
    return 0;
}
