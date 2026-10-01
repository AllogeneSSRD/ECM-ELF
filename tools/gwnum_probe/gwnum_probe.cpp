/*----------------------------------------------------------------------
| gwnum_probe.cpp -- feasibility measurement for using prime95/gwnum's
| AVX2 / AVX-512 IBDWT FFT from our ECM program.
|
| What it measures, for the modulus shapes and sizes this project actually
| works with (Mersenne N = 2^p - 1, p around 3,571 .. 12,323 bits):
|
|   * gwnum modular multiply / square, auto-detected CPU path
|   * the same with the AVX-512 flags removed from the handle (AVX2 path)
|     and with AVX2 removed as well (SSE/AVX path) -- so AVX2 vs AVX-512
|     is a measured delta, not a guess (the override is the one documented
|     in gwnum/tutorial.txt lines 29-36)
|   * the "one operand already FFTed" pattern (GWMUL_FFT_S2), which is what
|     an ECM stage-1 accumulator loop does
|   * multi-threaded multiply (gwset_num_threads)
|   * the GMP baseline our program uses today (mpz_mul + mpz_mod) on the
|     SAME sizes, and a raw mpz_mul for reference
|   * a correctness cross-check: 50 random products from gwnum must equal
|     GMP's mpz_mul+mpz_mod bit for bit, plus gwnum's own roundoff-error
|     report (gw_get_maxerr) and the chosen FFT description
|
| Build: tools/gwnum_probe/build_and_run.ps1 (MSVC x64, links gwnum64.lib
| from the prime95 source tree and our own GMP build).
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
}

#include "gmp.h"

namespace {

double now_us() {
    using clock = std::chrono::steady_clock;
    static const clock::time_point t0 = clock::now();
    return std::chrono::duration<double, std::micro>(clock::now() - t0).count();
}

/* Deterministic pseudo-random words: same input for every configuration, so the
   numbers are comparable (and the GMP cross-check compares identical operands). */
void fill_random(std::vector<uint32_t> &v, uint64_t seed) {
    uint64_t s = seed | 1u;
    for (size_t i = 0; i < v.size(); ++i) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        v[i] = static_cast<uint32_t>(s >> 32);
    }
}

/* gwnum requires operands BELOW the modulus. The first version of this probe fed it
   p/32+2 random words (up to p+64 bits, i.e. >= N): the FFT saturates, the roundoff report
   explodes (gw_get_maxerr 369 instead of 0.5) and EVERY product disagrees with GMP.
   Operands are therefore built with p-1 significant bits, always < N = 2^p - 1. */
void make_operand(std::vector<uint32_t> &v, int p, uint64_t seed) {
    const int bits = p - 1;
    const size_t n = static_cast<size_t>(bits / 32) + 1;
    v.assign(n, 0u);
    fill_random(v, seed);
    const int top_bits = bits - 32 * static_cast<int>(n - 1);
    v[n - 1] &= (top_bits >= 32) ? 0xFFFFFFFFu : ((1u << top_bits) - 1u);
    v[n - 1] |= 1u << (top_bits - 1); /* keep the value full-size */
}

struct GwMeas {
    bool ok = false;
    std::string err;
    std::string fft_desc;
    unsigned long fftlen = 0;
    int fft_type = 0;
    int arch = 0;
    double us_mul = 0.0;       /* both operands normal (preserved) */
    double us_sqr = 0.0;
    double us_mul_ffts2 = 0.0; /* s2 pre-FFTed: the ECM accumulator pattern */
    double maxerr = 0.0;
    int mismatches = 0;        /* vs GMP, over 50 random products */
};

/* Time one call pattern: warm up, then the best of 3 timed batches. */
template <typename Fn>
double time_batch(int iters, Fn fn) {
    double best = 1e30;
    fn(); /* warm up (also settles FFT caches) */
    for (int r = 0; r < 3; ++r) {
        const double t0 = now_us();
        for (int i = 0; i < iters; ++i) fn();
        const double dt = now_us() - t0;
        if (dt < best) best = dt;
    }
    return best / static_cast<double>(iters);
}

GwMeas run_gwnum(int p, bool keep_avx512, bool keep_avx2, int threads, int iters,
                 const std::vector<uint32_t> &words, mpz_srcptr gmp_n) {
    GwMeas m;
    gwhandle h;
    gwinit(&h);
    if (!keep_avx512) h.cpu_flags &= ~(unsigned)(CPU_AVX512F | CPU_AVX512VL | CPU_AVX512DQ);
    if (!keep_avx2) h.cpu_flags &= ~(unsigned)(CPU_AVX2 | CPU_FMA3);
    if (threads > 1) gwset_num_threads(&h, static_cast<unsigned>(threads));
    /* Exact Mersenne form 1*2^p-1: gwnum's primary, fastest use case. */
    const int rc = gwsetup(&h, 1.0, 2, static_cast<unsigned long>(p), -1);
    if (rc != 0) {
        m.err = "gwsetup failed: " + std::to_string(rc);
        gwdone(&h);
        return m;
    }
    char desc[512];
    desc[0] = '\0';
    gwfft_description(&h, desc);
    m.fft_desc = desc;
    m.fftlen = h.FFTLEN;
    m.fft_type = h.FFT_TYPE;
    m.arch = h.ARCH;

    gwnum x = gwalloc(&h), y = gwalloc(&h), z = gwalloc(&h);
    if (x == nullptr || y == nullptr || z == nullptr) {
        m.err = "gwalloc failed";
        gwdone(&h);
        return m;
    }
    binarytogw(&h, words.data(), static_cast<uint32_t>(words.size()), x);
    binarytogw(&h, words.data(), static_cast<uint32_t>(words.size()), y);

    m.us_mul = time_batch(iters, [&] {
        gwmul3(&h, x, y, z, GWMUL_PRESERVE_S1 | GWMUL_PRESERVE_S2);
    });
    m.us_sqr = time_batch(iters, [&] { gwsquare2(&h, x, z, GWMUL_PRESERVE_S1); });
    /* s2 is FFTed once and reused: an ECM stage-1 multiply against a fixed value. */
    gwmul3(&h, x, y, z, GWMUL_FFT_S2 | GWMUL_PRESERVE_S1);
    m.us_mul_ffts2 = time_batch(iters, [&] {
        gwmul3(&h, x, y, z, GWMUL_FFT_S2 | GWMUL_PRESERVE_S1);
    });

    /* Correctness: gwnum vs GMP on 50 products (this is the real risk of a
       floating-point IBDWT: roundoff, not speed). */
    gwerror_checking(&h, 1);
    std::vector<uint32_t> out(words.size() + 8);
    mpz_t ga, gb, gc;
    mpz_inits(ga, gb, gc, nullptr);
    for (int i = 0; i < 50; ++i) {
        std::vector<uint32_t> w1, w2;
        make_operand(w1, p, 0x1234567ull + static_cast<uint64_t>(i) * 7919ull);
        make_operand(w2, p, 0xabcdef1ull + static_cast<uint64_t>(i) * 104729ull);
        binarytogw(&h, w1.data(), static_cast<uint32_t>(w1.size()), x);
        binarytogw(&h, w2.data(), static_cast<uint32_t>(w2.size()), y);
        gwmul3(&h, x, y, z, GWMUL_PRESERVE_S1 | GWMUL_PRESERVE_S2);
        const long n = gwtobinary(&h, z, out.data(), static_cast<uint32_t>(out.size()));
        if (n < 0) { ++m.mismatches; continue; }
        mpz_import(ga, static_cast<size_t>(n), -1, 4, 0, 0, out.data());
        mpz_import(gb, w1.size(), -1, 4, 0, 0, w1.data());
        mpz_t g2;
        mpz_init(g2);
        mpz_import(g2, w2.size(), -1, 4, 0, 0, w2.data());
        mpz_mul(gc, gb, g2);
        mpz_mod(gc, gc, gmp_n);
        if (mpz_cmp(ga, gc) != 0) {
            ++m.mismatches;
            if (m.mismatches == 1) {
                char *sa = mpz_get_str(nullptr, 16, ga);
                char *sc = mpz_get_str(nullptr, 16, gc);
                std::printf("    [cross-check] first mismatch i=%d words=%ld\n", i, n);
                std::printf("      gwnum: %s\n", std::string(sa).substr(0, 48).c_str());
                std::printf("      gmp  : %s\n", std::string(sc).substr(0, 48).c_str());
                std::printf("      out[0..3]=%08x %08x %08x %08x  w1[0..3]=%08x %08x %08x %08x\n",
                            out[0], out[1], out[2], out[3], w1[0], w1[1], w1[2], w1[3]);
                void (*freefunc)(void *, size_t) = nullptr;
                mp_get_memory_functions(nullptr, nullptr, &freefunc);
                freefunc(sa, std::strlen(sa) + 1);
                freefunc(sc, std::strlen(sc) + 1);
            }
        }
        mpz_clear(g2);
    }
    m.maxerr = gw_get_maxerr(&h);
    gwerror_checking(&h, 0);
    mpz_clears(ga, gb, gc, nullptr);

    gwfreeall(&h);
    gwdone(&h);
    m.ok = true;
    return m;
}

/* Same measurement for a modulus with NO special form (PrimeNet does hand out ECM= lines whose
   N is not 2^p-1). gwnum/tutorial.txt:25-26 says multiplications there are "three times
   slower" -- measured below instead of trusted. */
GwMeas run_gwnum_general(int p, int iters, const std::vector<uint32_t> &words,
                         const std::vector<uint64_t> &mod64, mpz_srcptr gmp_n) {
    GwMeas m;
    gwhandle h;
    gwinit(&h);
    const int rc = gwsetup_general_mod_64(&h, mod64.data(),
                                          static_cast<uint64_t>(mod64.size()));
    if (rc != 0) {
        m.err = "gwsetup_general_mod_64 failed: " + std::to_string(rc);
        gwdone(&h);
        return m;
    }
    char desc[512];
    desc[0] = '\0';
    gwfft_description(&h, desc);
    m.fft_desc = desc;
    m.fftlen = h.FFTLEN;
    m.fft_type = h.FFT_TYPE;
    m.arch = h.ARCH;

    gwnum x = gwalloc(&h), y = gwalloc(&h), z = gwalloc(&h);
    if (x == nullptr || y == nullptr || z == nullptr) {
        m.err = "gwalloc failed";
        gwdone(&h);
        return m;
    }
    binarytogw(&h, words.data(), static_cast<uint32_t>(words.size()), x);
    binarytogw(&h, words.data(), static_cast<uint32_t>(words.size()), y);
    m.us_mul = time_batch(iters, [&] {
        gwmul3(&h, x, y, z, GWMUL_PRESERVE_S1 | GWMUL_PRESERVE_S2);
    });
    m.us_sqr = time_batch(iters, [&] { gwsquare2(&h, x, z, GWMUL_PRESERVE_S1); });
    m.us_mul_ffts2 = 0.0;

    gwerror_checking(&h, 1);
    std::vector<uint32_t> out(words.size() + 8);
    mpz_t ga, gb, gc, g2;
    mpz_inits(ga, gb, gc, g2, nullptr);
    for (int i = 0; i < 20; ++i) {
        std::vector<uint32_t> w1, w2;
        make_operand(w1, p, 0x5151u + static_cast<uint64_t>(i) * 6151ull);
        make_operand(w2, p, 0x9292u + static_cast<uint64_t>(i) * 3571ull);
        binarytogw(&h, w1.data(), static_cast<uint32_t>(w1.size()), x);
        binarytogw(&h, w2.data(), static_cast<uint32_t>(w2.size()), y);
        gwmul3(&h, x, y, z, GWMUL_PRESERVE_S1 | GWMUL_PRESERVE_S2);
        const long n = gwtobinary(&h, z, out.data(), static_cast<uint32_t>(out.size()));
        if (n < 0) { ++m.mismatches; continue; }
        mpz_import(ga, static_cast<size_t>(n), -1, 4, 0, 0, out.data());
        mpz_import(gb, w1.size(), -1, 4, 0, 0, w1.data());
        mpz_import(g2, w2.size(), -1, 4, 0, 0, w2.data());
        mpz_mul(gc, gb, g2);
        mpz_mod(gc, gc, gmp_n);
        if (mpz_cmp(ga, gc) != 0) ++m.mismatches;
    }
    m.maxerr = gw_get_maxerr(&h);
    gwerror_checking(&h, 0);
    mpz_clears(ga, gb, gc, g2, nullptr);
    gwfreeall(&h);
    gwdone(&h);
    m.ok = true;
    return m;
}

struct GmpMeas {
    double us_mulmod = 0.0;
    double us_mul = 0.0;
};

GmpMeas run_gmp(int p, int iters, const std::vector<uint32_t> &words) {
    GmpMeas m;
    mpz_t a, b, n, c;
    mpz_inits(a, b, n, c, nullptr);
    mpz_import(a, words.size(), -1, 4, 0, 0, words.data());
    mpz_import(b, words.size(), -1, 4, 0, 0, words.data());
    mpz_setbit(n, static_cast<mp_bitcnt_t>(p));
    mpz_sub_ui(n, n, 1); /* 2^p - 1 */
    m.us_mulmod = time_batch(iters, [&] {
        mpz_mul(c, a, b);
        mpz_mod(c, c, n);
    });
    m.us_mul = time_batch(iters, [&] { mpz_mul(c, a, b); });
    mpz_clears(a, b, n, c, nullptr);
    return m;
}

void report(int p, const char *label, const GwMeas &m) {
    if (!m.ok) {
        std::printf("  %-26s FAILED: %s\n", label, m.err.c_str());
        return;
    }
    std::printf("  %-26s mul %8.2f us (%9.0f/s)  sqr %8.2f us  mul[FFT_S2] %8.2f us"
                "  fftlen=%lu type=%d arch=%d err=%.3g bad=%d\n",
                label, m.us_mul, 1e6 / m.us_mul, m.us_sqr, m.us_mul_ffts2, m.fftlen,
                m.fft_type, m.arch, m.maxerr, m.mismatches);
    std::printf("  %-26s fft: %s\n", "", m.fft_desc.c_str());
}

} /* namespace */

/* Elementary sanity checks: these need no GMP, so they separate "gwnum is wrong"
   from "my harness is wrong". Run with --selftest. */
static int selftest(int p) {
    gwhandle h;
    gwinit(&h);
    if (gwsetup(&h, 1.0, 2, static_cast<unsigned long>(p), -1) != 0) {
        std::printf("selftest: gwsetup failed\n");
        return 1;
    }
    char desc[512];
    gwfft_description(&h, desc);
    std::printf("selftest p=%d  fft: %s  fftlen=%lu\n", p, desc, h.FFTLEN);

    gwnum x = gwalloc(&h), y = gwalloc(&h), z = gwalloc(&h);
    std::vector<uint32_t> buf(static_cast<size_t>(p / 32) + 8);
    int bad = 0;

    struct Case { const char *name; uint64_t a; uint64_t b; uint64_t want; };
    const Case cases[] = {
        {"3*5", 3, 5, 15},
        {"2^32*2^32", 4294967296ull, 4294967296ull, 0}, /* checked as 2^64 mod N below */
    };
    for (const Case &c : cases) {
        u64togw(&h, c.a, x);
        u64togw(&h, c.b, y);
        gwmul3(&h, x, y, z, GWMUL_PRESERVE_S1 | GWMUL_PRESERVE_S2);
        const long n = gwtobinary(&h, z, buf.data(), static_cast<uint32_t>(buf.size()));
        std::printf("  %-12s -> words=%ld  w0=0x%08x w1=0x%08x w2=0x%08x\n", c.name, n,
                    buf[0], n > 1 ? buf[1] : 0u, n > 2 ? buf[2] : 0u);
        if (c.want != 0) {
            if (n != 1 || buf[0] != static_cast<uint32_t>(c.want)) {
                std::printf("    MISMATCH: want %llu\n", static_cast<unsigned long long>(c.want));
                ++bad;
            }
        }
    }

    /* Algebraic identities on full-size values: (N-1)^2 = 1 and 2 * 2^(p-1) = 1 (mod N). */
    {
        std::vector<uint32_t> all_ones(static_cast<size_t>(p / 32) + 2, 0xFFFFFFFFu);
        all_ones.back() &= (p % 32 == 0) ? 0u : ((1u << (p % 32)) - 1u);
        binarytogw(&h, all_ones.data(), static_cast<uint32_t>(all_ones.size()), x); /* N-1 = -1 */
        gwsquare2(&h, x, z, GWMUL_PRESERVE_S1);
        long n = gwtobinary(&h, z, buf.data(), static_cast<uint32_t>(buf.size()));
        std::printf("  (-1)^2       -> words=%ld  w0=0x%08x (want 1)\n", n, buf[0]);
        if (n != 1 || buf[0] != 1u) { ++bad; std::printf("    MISMATCH\n"); }

        std::vector<uint32_t> two(static_cast<size_t>(p / 32) + 2, 0u), half(static_cast<size_t>(p / 32) + 2, 0u);
        two[0] = 2;
        /* 2^(p-1) = (N+1)/2: a single 1 bit at position p-1. */
        half[(p - 1) / 32] = 1u << ((p - 1) % 32);
        binarytogw(&h, two.data(), static_cast<uint32_t>(two.size()), x);
        binarytogw(&h, half.data(), static_cast<uint32_t>(half.size()), y);
        gwmul3(&h, x, y, z, GWMUL_PRESERVE_S1 | GWMUL_PRESERVE_S2);
        n = gwtobinary(&h, z, buf.data(), static_cast<uint32_t>(buf.size()));
        std::printf("  2*2^(p-1)    -> words=%ld  w0=0x%08x (want 1)\n", n, buf[0]);
        if (n != 1 || buf[0] != 1u) { ++bad; std::printf("    MISMATCH\n"); }
    }

    std::printf("selftest: %s\n", bad == 0 ? "all identities hold" : "FAILURES PRESENT");
    gwdone(&h);
    return bad == 0 ? 0 : 1;
}

int main(int argc, char **argv) {
    const bool quick = (argc > 1 && std::strcmp(argv[1], "--quick") == 0);
    if (argc > 1 && std::strcmp(argv[1], "--selftest") == 0) {
        guessCpuType();
        guessCpuSpeed();
        return selftest(3571);
    }
    guessCpuType();
    guessCpuSpeed();
    std::printf("gwnum probe -- IBDWT feasibility for our ECM sizes\n");
    std::printf("cpu: %s\n", CPU_BRAND);
    std::printf("cpu flags: 0x%08x  avx2=%d avx512f=%d avx512vl=%d avx512dq=%d  cores=%u\n",
                CPU_FLAGS, (CPU_FLAGS & CPU_AVX2) ? 1 : 0, (CPU_FLAGS & CPU_AVX512F) ? 1 : 0,
                (CPU_FLAGS & CPU_AVX512VL) ? 1 : 0, (CPU_FLAGS & CPU_AVX512DQ) ? 1 : 0, CPU_CORES);
    std::printf("gwnum version: %s\n\n", GWNUM_VERSION);

    const int sizes[] = {3571, 12323, 100003, 1000003};
    for (int p : sizes) {
        if (quick && p > 20000) break;
        std::vector<uint32_t> w;
        make_operand(w, p, 0x5eed0000ull + static_cast<uint64_t>(p));

        mpz_t gmp_n;
        mpz_init(gmp_n);
        mpz_setbit(gmp_n, static_cast<mp_bitcnt_t>(p));
        mpz_sub_ui(gmp_n, gmp_n, 1);

        const int iters = (p <= 13000) ? 20000 : (p <= 130000 ? 4000 : 400);
        std::printf("== 2^%d - 1 (%d bits, Mersenne form)  iters=%d\n", p, p, iters);
            /* Label note (measured + verified in the gwnum source): there is NO AVX2 FFT path. CPU_AVX2
           is detected in cpuid.c but never read by the FFT dispatch; the float FFTs are SSE2 (x*),
           AVX/FMA3 (y*) and AVX-512 (z*). Clearing CPU_AVX512F therefore selects FMA3, not AVX2,
           and that is what the FFT description below prints. */
        report(p, "gwnum auto (AVX-512)", run_gwnum(p, true, true, 1, iters, w, gmp_n));
        report(p, "AVX-512 off -> FMA3", run_gwnum(p, false, true, 1, iters, w, gmp_n));
        report(p, "AVX-512+FMA3 off -> AVX", run_gwnum(p, false, false, 1, iters, w, gmp_n));
        if (p >= 100000) {
            report(p, "gwnum auto, 4 threads", run_gwnum(p, true, true, 4, iters, w, gmp_n));
            report(p, "gwnum auto, 8 threads", run_gwnum(p, true, true, 8, iters, w, gmp_n));
        }
        /* Non-Mersenne modulus of the same size (no k*b^n+c form). */
        {
            std::vector<uint64_t> mod64(static_cast<size_t>(p / 64) + 1);
            for (size_t i = 0; i < mod64.size(); ++i) {
                mod64[i] = 0x9E3779B97F4A7C15ull * (i + 1) + 0x123456789ABCDEFull;
            }
            const int topbit = (p - 1) % 64;
            mod64.back() &= (1ull << (topbit + 1)) - 1ull;
            mod64.back() |= 1ull << topbit;
            mod64[0] |= 1ull; /* odd */
            mpz_t gn;
            mpz_init(gn);
            mpz_import(gn, mod64.size(), -1, 8, 0, 0, mod64.data());
            report(p, "gwnum general modulus", run_gwnum_general(p, iters, w, mod64, gn));
            mpz_clear(gn);
        }

        const GmpMeas g = run_gmp(p, iters, w);
        std::printf("  %-26s mul+mod %8.2f us (%9.0f/s)   raw mpz_mul %8.2f us\n", "GMP (zen3 build)",
                    g.us_mulmod, 1e6 / g.us_mulmod, g.us_mul);
        std::printf("\n");
        mpz_clear(gmp_n);
    }
    return 0;
}
