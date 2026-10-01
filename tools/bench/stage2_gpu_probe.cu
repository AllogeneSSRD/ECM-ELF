/* stage2_gpu_probe.cu — standalone driver for the CUDA/CGBN stage 2 (M1).

   Nothing in the production path is involved: this links
   kernels/cuda/cgbn_stage2.cu (+ the per-TPI instantiations) directly, so stage 2 can be
   built, measured and compared against the CPU reference
   (tools/bench/stage2_ref.cpp --algorithm pairing) before any driver/backend plumbing
   exists.  It is the executable that tools/test/test_stage2_gpu.ps1 drives.

   Everything the driver would have to supply is a command-line flag here:
     --n <decimal>       the number to factor (default: the frozen test vector 2^128+1)
     --sigma <u64>       first curve's sigma (Suyama / gmp-ecm -param 0)
     --curves <k>        run k curves with sigma, sigma+1, ... sigma+k-1
     --b1 --b2 --d       stage-1 bound, stage-2 bound, baby/giant span
     --segs <k>          pairing instances per curve
     --device <n>        CUDA device (default 0; the tests pass 1 to spare a busy card)
     --save <file>       take (SIGMA, B1, X=0x..) per curve from a stage-1 save file
                         instead of running the stage-1 ladder on the device
     --tiers             print the (bits, tpi) tiers this build carries and exit
     --selftest          run the frozen-vector checks (see the header of the test script)
     --verbose <n>       OUTPUT_* verbosity (0 always, 1 normal, 2 verbose)

   Machine-readable line (parsed by the test):
     stage2gpu: curves=1 hits=1 factors=59649589127497217 tier=192 tpi=4 segs=1
                gpu=0.123 elapsed=0.456 stage1_done=0 bad_factors=0

   Build (from the repo root, CUDA 13.3 + MSVC):
     nvcc -std=c++17 -O3 -arch=sm_89 -I kernels/cuda -I cgbn/include -I include
          -Xcompiler /wd4819 -o build_cuda_cmake/stage2_gpu_probe.exe
          tools/bench/stage2_gpu_probe.cu kernels/cuda/cgbn_stage2.cu
          kernels/cuda/cgbn_stage2_kernels_tpi4.cu kernels/cuda/cgbn_stage2_kernels_tpi8.cu
          -L third_party/gmp-zen3/dist/lib -lgmp
   (and copy third_party/gmp-zen3/dist/bin/gmp-10.dll next to the exe).
*/

#include "cgbn_stage2_cuda.h"
#include "cuda_ecm_shim.h"

#include <gmp.h>

#include <chrono>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

/* ── logging shims ───────────────────────────────────────────────────────────────
   In ecm_cuda these live in src/cuda/ecm_cuda_backend.cu and route through the project's
   timestamped logger; here they print plainly, so the probe stays a single self-contained
   executable. */
static int g_verbose = OUTPUT_NORMAL;

extern "C" void ecm_cuda_set_verbose(int level) { g_verbose = level; }

extern "C" int test_verbose(int level) { return g_verbose >= level; }

extern "C" void outputf(int verbosity, const char *format, ...) {
    if (verbosity != OUTPUT_ERROR && g_verbose < verbosity) return;
    FILE *stream = (verbosity == OUTPUT_ERROR) ? stderr : stdout;
    va_list ap;
    va_start(ap, format);
    vfprintf(stream, format, ap);
    va_end(ap);
    fflush(stream);
}

/* ── helpers ─────────────────────────────────────────────────────────────────── */

static double now_s() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

/* The save format is the driver's: one curve per line, "SIGMA=<dec>;B1=<dec>;...;X=0x<hex>;..." */
struct SaveCurve {
    uint64_t sigma = 0;
    uint64_t b1 = 0;
    std::string x_hex;
};

static std::vector<SaveCurve> parse_save(const std::string &path) {
    std::vector<SaveCurve> out;
    FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) return out;
    char line[8192];
    while (std::fgets(line, sizeof(line), f)) {
        const char *ps = std::strstr(line, "SIGMA=");
        const char *pb = std::strstr(line, "B1=");
        const char *px = std::strstr(line, "X=0x");
        if (!ps || !pb) continue;
        SaveCurve c;
        c.sigma = std::strtoull(ps + 6, nullptr, 10);
        c.b1 = std::strtoull(pb + 3, nullptr, 10);
        if (px) {
            /* px points at "X=0x": skip FOUR characters -- mpz_set_str(...,16) rejects the
               "0x" prefix, and keeping it silently produced x = 0 in the CPU reference. */
            c.x_hex = px + 4;
            const size_t stop = c.x_hex.find_first_of(";\r\n ");
            if (stop != std::string::npos) c.x_hex.resize(stop);
        }
        out.push_back(c);
    }
    std::fclose(f);
    return out;
}

static void print_tiers() {
    std::printf("stage2gpu_tiers:");
    uint32_t last = 0;
    for (uint32_t nb = 1; nb <= 2100; ++nb) {
        uint32_t bits = 0, tpi = 0, tpb = 0;
        if (!cgbn_stage2_tier_for(nb, &bits, &tpi, &tpb)) continue;
        if (bits == last) continue;                 /* only the tier boundaries */
        last = bits;
        std::printf(" %u/%u/%u", bits, tpi, tpb);
    }
    std::printf("\n");
}

/* One run of stage 2 over `curves` curves; returns the number of distinct factors. */
struct RunResult {
    uint64_t hits = 0;
    uint32_t bad = 0;
    uint32_t stage1_done = 0;
    double gpu = 0.0;
    uint32_t tier = 0, tpi = 0, segs = 1;
    std::vector<std::string> factors;
};

static int run(const mpz_t N, uint64_t sigma0, uint32_t curves, uint64_t B1, uint64_t B2,
               uint64_t D, uint32_t segs, int device, const std::vector<SaveCurve> &sv,
               int verbose, RunResult &res) {
    std::vector<uint64_t> sigmas(curves);
    std::vector<std::string> xstore(curves);
    std::vector<const char *> xptr(curves, nullptr);
    for (uint32_t i = 0; i < curves; ++i) {
        sigmas[i] = sv.empty() ? (sigma0 + i) : sv[i].sigma;
        if (!sv.empty() && !sv[i].x_hex.empty()) {
            xstore[i] = sv[i].x_hex;
            xptr[i] = xstore[i].c_str();
        }
    }

    ecm_stage2_opts opts;
    std::memset(&opts, 0, sizeof(opts));
    opts.b1 = sv.empty() ? B1 : sv[0].b1;
    opts.b2 = B2;
    opts.d = D;
    opts.segs = segs;
    opts.device_index = device;
    opts.torsion = 1;

    const int max_factors = 16;
    /* std::vector<mpz_t> is illegal (mpz_t is an array type), so this is a plain array. */
    mpz_t *fac = new mpz_t[max_factors];
    std::vector<int> found(max_factors, 0);
    for (int i = 0; i < max_factors; ++i) mpz_init(fac[i]);

    float gpu = 0.0f;
    uint32_t s1 = 0;
    const int rc = cgbn_ecm_stage2(fac, found.data(), max_factors, N, curves,
                                   sigmas.data(), xptr.empty() ? nullptr : xptr.data(),
                                   &opts, &gpu, &s1, verbose);
    res.gpu = gpu;
    res.stage1_done = s1;
    res.segs = opts.segs ? opts.segs : 1;
    cgbn_stage2_tier_for((uint32_t)mpz_sizeinbase(N, 2), &res.tier, &res.tpi, nullptr);

    if (rc == 0) {
        for (int i = 0; i < max_factors; ++i) {
            if (!found[i]) continue;
            /* soundness: a reported factor must divide N */
            if (mpz_divisible_p(N, fac[i]))
                ++res.hits;
            else
                ++res.bad;
            char *s = mpz_get_str(nullptr, 10, fac[i]);
            res.factors.push_back(std::string(s));
            void (*ff)(void *, size_t) = nullptr;
            mp_get_memory_functions(nullptr, nullptr, &ff);
            ff(s, std::strlen(s) + 1);
        }
    }
    for (int i = 0; i < max_factors; ++i) mpz_clear(fac[i]);
    delete[] fac;
    return rc;
}

static void print_summary(const RunResult &r, uint32_t curves, double elapsed) {
    std::printf("stage2gpu: curves=%u hits=%llu factors=", curves,
                (unsigned long long)r.hits);
    for (size_t i = 0; i < r.factors.size(); ++i)
        std::printf("%s%s", i ? "," : "", r.factors[i].c_str());
    std::printf(" tier=%u tpi=%u segs=%u gpu=%.3f elapsed=%.2f stage1_done=%u bad_factors=%u\n",
                r.tier, r.tpi, r.segs, r.gpu, elapsed, r.stage1_done, r.bad);
}

/* ── selftest: the frozen vector, on the device ───────────────────────────────── */

static int g_checks = 0, g_failed = 0;
static void check(const char *name, bool ok, const std::string &detail = "") {
    if (ok) {
        ++g_checks;
        std::printf("  [ok]   %s\n", name);
    } else {
        ++g_checks;
        ++g_failed;
        std::printf("  [FAIL] %s%s%s\n", name, detail.empty() ? "" : " -- ",
                    detail.c_str());
    }
}

/* 2^128+1 = 59649589127497217 * 5704689200685129054721.  sigma=26 with B1=1e3/B2=1e6/D=210
   is the FROZEN configuration of tools/test/test_stage2_ref.ps1: stage 1 cannot find it and
   stage 2 hits it through the prime 114713 (the group order has 114713 as its largest
   prime).  B2=114000 must therefore find nothing: the hit is sharp, not accidental. */
static const char *N128 = "340282366920938463463374607431768211457";
static const char *F17 = "59649589127497217";

static int selftest(int device) {
    mpz_t N;
    mpz_init(N);
    if (mpz_set_str(N, N128, 10) != 0) {
        std::printf("stage2gpu_selftest: cannot parse the frozen N\n");
        mpz_clear(N);
        return 2;
    }

    {
        uint32_t bits = 0, tpi = 0, tpb = 0;
        const bool ok = cgbn_stage2_tier_for(129, &bits, &tpi, &tpb) != 0;
        check("a 129-bit N has a tier", ok);
        check("the tier covers N + carry bits", bits >= 129u + 6u,
              "bits=" + std::to_string(bits));
    }

    {
        RunResult r;
        const double t0 = now_s();
        const int rc = run(N, 26, 1, 1000, 1000000, 210, 1, device, {}, OUTPUT_NORMAL, r);
        check("the frozen vector runs", rc == 0);
        check("the known 17-digit factor is found", r.hits == 1 &&
              !r.factors.empty() && r.factors[0] == F17,
              r.factors.empty() ? "no factor" : r.factors[0]);
        check("stage 1 did not already find it", r.stage1_done == 0,
              "stage1_done=" + std::to_string(r.stage1_done));
        check("no reported factor fails to divide N", r.bad == 0);
        /* Machine-readable: the test script compares this against the CPU reference's
           `stage2: ... factors=...` line, so the two must use the same field name. */
        std::printf("stage2gpu_selftest_vector: curves=1 hits=%llu factors=%s gpu=%.3f\n",
                    (unsigned long long)r.hits,
                    r.factors.empty() ? "" : r.factors[0].c_str(), r.gpu);
        std::printf("  (frozen vector: gpu=%.3f s, cpu wall=%.2f s)\n", r.gpu, now_s() - t0);
    }

    {
        RunResult r;
        const int rc = run(N, 26, 1, 1000, 114000, 210, 1, device, {}, OUTPUT_NORMAL, r);
        check("B2 just below the hit's prime finds nothing",
              rc == 0 && r.hits == 0,
              "hits=" + std::to_string(r.hits));
    }

    {
        /* segs > 1 must not change the answer: the accumulator is split, the host
           multiplies the pieces back together before the gcd. */
        RunResult r;
        const int rc = run(N, 26, 1, 1000, 1000000, 210, 4, device, {}, OUTPUT_NORMAL, r);
        check("segs=4 agrees with segs=1", rc == 0 && r.hits == 1 &&
              !r.factors.empty() && r.factors[0] == F17);
    }

    {
        /* A sigma sweep: the reference finds exactly one hit in sigma 2..200 for this
           B2 (test_stage2_ref.ps1 proves the sweep is meaningful). */
        RunResult r;
        const int rc = run(N, 2, 199, 1000, 1000000, 210, 1, device, {}, OUTPUT_NORMAL, r);
        check("sigma sweep 2..200 runs", rc == 0);
        check("the sweep finds exactly the known factor", r.hits >= 1 &&
              !r.factors.empty() && r.factors[0] == F17,
              "hits=" + std::to_string(r.hits));
        std::printf("stage2gpu_selftest_sweep: curves=199 hits=%llu factors=%s gpu=%.3f\n",
                    (unsigned long long)r.hits,
                    r.factors.empty() ? "" : r.factors[0].c_str(), r.gpu);
    }

    mpz_clear(N);
    std::printf("stage2gpu_selftest: checks=%d failed=%d\n", g_checks, g_failed);
    return g_failed == 0 ? 0 : 1;
}

/* ── main ─────────────────────────────────────────────────────────────────────── */

int main(int argc, char **argv) {
    const char *n_str = N128;
    const char *save = nullptr;
    uint64_t B1 = 1000, B2 = 1000000, D = 210, sigma = 26;
    uint32_t curves = 1, segs = 1;
    int device = 0, verbose = OUTPUT_NORMAL;
    bool do_selftest = false, do_tiers = false;

    for (int i = 1; i < argc; ++i) {
        const char *a = argv[i];
        auto next = [&]() -> const char * { return (i + 1 < argc) ? argv[++i] : ""; };
        if (!std::strcmp(a, "--selftest")) do_selftest = true;
        else if (!std::strcmp(a, "--tiers")) do_tiers = true;
        else if (!std::strcmp(a, "--n")) n_str = next();
        else if (!std::strcmp(a, "--save")) save = next();
        else if (!std::strcmp(a, "--b1")) B1 = std::strtoull(next(), nullptr, 10);
        else if (!std::strcmp(a, "--b2")) B2 = std::strtoull(next(), nullptr, 10);
        else if (!std::strcmp(a, "--d")) D = std::strtoull(next(), nullptr, 10);
        else if (!std::strcmp(a, "--sigma")) sigma = std::strtoull(next(), nullptr, 10);
        else if (!std::strcmp(a, "--curves")) curves = (uint32_t)std::strtoul(next(), nullptr, 10);
        else if (!std::strcmp(a, "--segs")) segs = (uint32_t)std::strtoul(next(), nullptr, 10);
        else if (!std::strcmp(a, "--device")) device = (int)std::strtol(next(), nullptr, 10);
        else if (!std::strcmp(a, "--verbose")) verbose = (int)std::strtol(next(), nullptr, 10);
        else {
            std::printf("stage2_gpu_probe: unknown argument '%s'\n", a);
            return 2;
        }
    }

    ecm_cuda_set_verbose(verbose);

    if (do_tiers) {
        print_tiers();
        return 0;
    }
    if (do_selftest) return selftest(device);

    mpz_t N;
    mpz_init(N);
    if (mpz_set_str(N, n_str, 10) != 0) {
        std::printf("stage2_gpu_probe: cannot parse --n '%s'\n", n_str);
        mpz_clear(N);
        return 2;
    }

    std::vector<SaveCurve> sv;
    if (save) {
        sv = parse_save(save);
        if (sv.empty()) {
            std::printf("stage2_gpu_probe: no curve line in the save '%s'\n", save);
            mpz_clear(N);
            return 2;
        }
        if (sv.size() < curves) {
            std::printf("stage2_gpu_probe: the save has %zu curves but --curves %u was asked "
                        "for\n", sv.size(), curves);
            mpz_clear(N);
            return 2;
        }
    }

    RunResult r;
    const double t0 = now_s();
    const int rc = run(N, sigma, curves, B1, B2, D, segs, device, sv, verbose, r);
    const double elapsed = now_s() - t0;
    print_summary(r, curves, elapsed);
    mpz_clear(N);
    return rc == 0 ? 0 : 1;
}
