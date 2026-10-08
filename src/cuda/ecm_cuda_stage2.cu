// Production ECM Stage2, forked from the verified 58db21f tree engine.
// Owned source closure: no dependency on tools/bench. Development variants live
// in tools/bench/ecm_cuda_stage2_dev.cu. Keep arithmetic checks and nonunit,
// allocation and generic-modulus fallbacks when changing production policy.
#include "../core/ecm_stage2_logging.h"
#if !defined(NTT_GL_FIXED_MODE) || NTT_GL_FIXED_MODE != 3
#error "Production Stage2 requires fixed PTX3 Goldilocks"
#endif
#if !defined(NTT_OUTER_UNROLL_U) || NTT_OUTER_UNROLL_U != 0
#error "Production Stage2 requires the verified outer schedule"
#endif
static constexpr int s4_tail_mont_mode() { return 0; }
#include <cstdlib>

namespace {
// Set before the engine's static flag snapshots. Explicit environment overrides
// remain available for comparison with the experiment executable.
struct ProductionDefaults {
    ProductionDefaults() {
        const char *keys[] = {"NTT_S4_MERSENNE", "NTT_SMALL_PRIME_REUSE",
            "NTT_GIANT_SEED_DEVICE", "NTT_GFINV_SEG_EXACT", "NTT_GFINV_BATCH",
            "NTT_FOLD_FLAT", "NTT_FOLD_DEVICE", "NTT_GROOT_DEVICE", "NTT_SCALED_DESCENT",
            "NTT_S4_OUTPUT_WINDOW", "NTT_S4_CHUNK_OUTPUT", "NTT_DEVICE_GLEAF",
            "NTT_GROOT_TO_FOLD", "NTT_S4_ORACLE_ASYNC", "NTT_S4_CARRY_BATCH",
            "NTT_FUSE_WARP_TAIL", "NTT_XADD6", "NTT_D_MODEL", "NTT_BABY_DEVICE",
            "NTT_GIANT_SEED_PAIR"};
        for (const char *key : keys) set_default(key, "1");
#if defined(NTT_GL_FIXED_MODE) && NTT_GL_FIXED_MODE >= 0
        set_default("NTT_GL_SHORT_REDUCE", (NTT_GL_FIXED_MODE&1) ? "1" : "0");
#else
        set_default("NTT_GL_SHORT_REDUCE", "1");
#endif
#if defined(NTT_GL_FIXED_MODE) && NTT_GL_FIXED_MODE == 3
        set_default("NTT_POINT_MERSENNE", "1");
#endif
        set_default("NTT_FUSE_COOP_OUTER", "2");
        set_default("NTT_GIANT_BASE_CPU", "0");
        set_default("NTT_DEVICE_GLEAF_MAX_MB", "512");
        set_default("NTT_FOLD_DEVICE_MAX_MB", "640");
        set_default("NTT_FOLD_OWNER_REUSE", "3");
        set_default("NTT_S4_BATCH_MB", "64");
        set_default("NTT_S4_SAMPLE", "96");
        set_default("NTT_S4_CHECK_EVERY", "8");
        set_default("NTT_NAME_MAX", "1");
    }
    static void set_default(const char *key, const char *value) {
        if (std::getenv(key)) return;
#ifdef _WIN32
        _putenv_s(key, value);
#else
        setenv(key, value, 0);
#endif
    }
} production_defaults;
}

#include "stage2/ntt_runtime.cuh"
#include "../core/ecm_stage2_geometry.h"

#include <string>
#include <utility>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <array>
#include <functional>

/* ===================================================================================== *
 *  a crash must never look like a silent exit again.  The real-shape failures of section
 *  18.3 were reported by the harness as a bare "exit=1" with an EMPTY stderr, which is what
 *  a host access violation looks like once the WER dialog is suppressed (section 20): no
 *  stack, no code, no address.  This filter prints the exception code, the faulting address
 *  and the instruction/RIP (i.e. enough to map a fault back to a function with a map file),
 *  then lets the default handler terminate the process.  It is diagnostic only -- no
 *  arithmetic, no allocation, no CUDA call.
 *
 *  STATUS (verified, not assumed): with this filter installed the real shape (D = 570570,
 *  B2 = 1.94e12) STILL dies with exit=1 and a STILL-EMPTY stderr at 40-56 lines (just after the
 *  P = 32768/32769 G-tree shape selftest), so if that death were a host access violation this
 *  filter would have printed.  It does not print, and an independent vectored handler
 *  (AddVectoredExceptionHandler + DbgHelp, run against a PDB build) did not fire either, and
 *  the death is not deterministic (113 s once, ~4 min another time, several shorter).
 *
 *  RESOLVED (2026-10-01, after the above): it is NOT a clean exit and NOT an exception -- it is
 *  the display driver's watchdog.  Evidence, three independent pieces:
 *    1. `Get-WinEvent System -ProviderName nvlddmkm` carries id 13 (error) + id 153 (device
 *       reset/recovery) at 2026-10-01 03:15:59, i.e. the wall-clock instant of the death of the
 *       run that had printed `batched_progress: batch=4/66 ... t=90.7 s`; the same provider
 *       logged a burst of id 13 in the minutes before.  A TDR reset tears down the CUDA context
 *       and terminates the host WITHOUT any exception, which is exactly the "empty stderr" seen
 *       here (and why the WER LocalDumps folder stayed empty: no exception ever happens).
 *    2. The trigger is LAUNCH LENGTH, not memory: `s2g_launch_ladder` used to launch one thread
 *       per giant point, i.e. a single 207360-thread kernel in which every thread runs a serial
 *       2*S = 10522-step Montgomery ladder (~12 s of GPU time per launch at S = 5261).  The
 *       frozen shapes never exposed it because there the whole giant set is 4763 points.
 *    3. The fix below (a bounded grid-stride ladder launch, g_ladder_cap points per launch)
 *       makes the real shape run past that point; the failure is not a CUDA allocation error
 *       either, because every CUDA call goes through CK(), which prints
 *       "CUDA error <text> at <file>:<line>" and exits 2 -- and stderr was empty.
 *  The nets below stay, because a non-exception death must never be silent again: std::terminate
 *  is routed to a handler that names the active exception, CK() now prints the failing
 *  expression, the requested bytes and the device free/total, and a watchdog keeps the last
 *  measured state in memory (and on stderr when it can).
 * ===================================================================================== */
#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")     /* GetProcessMemoryInfo: host commit, for the last-state line */
static LONG WINAPI s2g_unhandled_filter(EXCEPTION_POINTERS *ep)
{
    const EXCEPTION_RECORD *r = ep->ExceptionRecord;
    const void *rip = nullptr;
#if defined(_M_X64) || defined(_M_AMD64)
    rip = (const void *)ep->ContextRecord->Rip;
#endif
    std::fprintf(stderr, "CRASH: code=0x%08lX fault_addr=%p rip=%p (host exception; the "
                         "exception code and the address above are the diagnosis)\n",
                 (unsigned long)r->ExceptionCode, r->ExceptionAddress, rip);
    std::fflush(stderr);
    return EXCEPTION_CONTINUE_SEARCH;
}
static void s2g_install_crash_handler(void)
{
    SetUnhandledExceptionFilter(s2g_unhandled_filter);
}
/* host commit charge + device free bytes, one line; also the "last known state" the watchdog
   leaves behind when the driver's watchdog kills us (see the header comment) */
static char g_last_state[512] = "none";
static void s2g_state(const char *what)
{
    PROCESS_MEMORY_COUNTERS pmc{};
    pmc.cb = sizeof(pmc);
    GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
    size_t dfree = 0, dtotal = 0;
    cudaMemGetInfo(&dfree, &dtotal);
    std::snprintf(g_last_state, sizeof(g_last_state),
                  "%s | host private=%.0f MB peak=%.0f MB | device free=%.0f MB of %.0f MB",
                  what, (double)pmc.PagefileUsage / 1048576.0,
                  (double)pmc.PeakPagefileUsage / 1048576.0, (double)dfree / 1048576.0,
                  (double)dtotal / 1048576.0);
}
static void s2g_print_last_state(const char *why)
{
    std::fprintf(stderr, "LAST_STATE (%s): %s\n", why, g_last_state);
    std::fflush(stderr);
}
/* every death that is not an exception ends up here: an escaping exception, a host bad_alloc,
   a throwing destructor.  Prints the type and what() before terminating. */
static void s2g_terminate_handler(void)
{
    std::fprintf(stderr, "TERMINATE: %s\n", g_last_state);
    if (std::current_exception()) {
        try { std::rethrow_exception(std::current_exception()); }
        catch (const std::exception &e) {
            std::fprintf(stderr, "TERMINATE: exception: %s\n", e.what());
        }
        catch (...) { std::fprintf(stderr, "TERMINATE: non-std exception\n"); }
    } else {
        std::fprintf(stderr, "TERMINATE: no active exception (pure virtual call, or terminate() "
                             "called directly)\n");
    }
    std::fflush(stderr);
    std::abort();
}
static void s2g_install_terminate(void) { std::set_terminate(s2g_terminate_handler); }
#else
static void s2g_install_crash_handler(void) {}
static void s2g_install_terminate(void) {}
static void s2g_state(const char *) {}
static void s2g_print_last_state(const char *) {}
#endif

/* ===================================================================================== *
 *  the dump file written by tools/bench/stage2_tree_ref.cpp --dump-F
 * ===================================================================================== */

struct FDump {
    bool ok = false;
    std::string err;
    std::string n_dec, n_hex, a24_hex, q_hex;
    long n_bits = 0;
    unsigned long long D = 0, B1 = 0, B2 = 0, sigma = 0, degree = 0;
    std::vector<unsigned long long> baby_j;      /* ascending, coprime to D, j <= D/2 */
    std::vector<std::string> baby_x;             /* the CPU's affine x of [j]Q, hex */
    std::vector<std::string> F;                  /* coefficients of F, ascending, hex */
};

static bool read_hex_mpz(mpz_t out, const std::string &hex)
{
    return !hex.empty() && mpz_set_str(out, hex.c_str(), 16) == 0;
}

static FDump read_f_dump(const char *path)
{
    FDump d;
    FILE *f = std::fopen(path, "rb");
    if (!f) { d.err = "cannot open"; return d; }
    char line[1 << 16];
    if (!std::fgets(line, sizeof(line), f)) { std::fclose(f); d.err = "empty file"; return d; }
    std::string hdr = line;
    while (!hdr.empty() && (hdr.back() == '\n' || hdr.back() == '\r')) hdr.pop_back();
    if (hdr != "stage2_tree_F_dump v1") {
        std::fclose(f);
        d.err = "not a stage2_tree_F_dump v1 file (got: " + hdr + ")";
        return d;
    }
    bool saw_end = false;
    while (std::fgets(line, sizeof(line), f)) {
        std::string s = line;
        while (!s.empty() && (s.back() == '\n' || s.back() == '\r')) s.pop_back();
        if (s.empty()) continue;
        const size_t sp = s.find(' ');
        const std::string key = (sp == std::string::npos) ? s : s.substr(0, sp);
        const std::string rest = (sp == std::string::npos) ? std::string() : s.substr(sp + 1);
        auto u64 = [&]() -> unsigned long long { return std::strtoull(rest.c_str(), nullptr, 10); };
        auto first_token = [&]() -> std::string {
            const size_t sp2 = rest.find(' ');
            return (sp2 == std::string::npos) ? rest : rest.substr(0, sp2);
        };
        if (key == "N_dec") d.n_dec = rest;
        else if (key == "N_hex") d.n_hex = rest;
        else if (key == "N_bits") d.n_bits = std::atol(rest.c_str());
        else if (key == "D") d.D = u64();
        else if (key == "B1") d.B1 = u64();
        else if (key == "B2") d.B2 = u64();
        else if (key == "sigma") d.sigma = u64();
        else if (key == "a24_hex") d.a24_hex = first_token();
        else if (key == "Q_hex") d.q_hex = first_token();
        else if (key == "baby_count") { /* implied by the baby lines; checked below */ }
        else if (key == "baby") {
            const size_t sp2 = rest.find(' ');
            if (sp2 == std::string::npos) { std::fclose(f); d.err = "bad baby line"; return d; }
            d.baby_j.push_back(std::strtoull(rest.substr(0, sp2).c_str(), nullptr, 10));
            d.baby_x.push_back(rest.substr(sp2 + 1));
        } else if (key == "F_degree") d.degree = u64();
        else if (key == "F") d.F.push_back(first_token());
        else if (key == "end") saw_end = true;
    }
    std::fclose(f);
    if (!saw_end) { d.err = "no end marker"; return d; }
    if (d.n_dec.empty() || d.baby_j.empty()) { d.err = "missing N or baby points"; return d; }
    if (d.baby_j.size() != d.baby_x.size()) { d.err = "baby j/x count mismatch"; return d; }
    if (d.F.size() != d.degree + 1) {
        d.err = "F has " + std::to_string(d.F.size()) + " coefficients but F_degree=" +
                std::to_string(d.degree);
        return d;
    }
    d.ok = true;
    return d;
}

/* the same file format, written by this tool (so the two dumps can be compared line by line) */
static bool write_f_dump(const char *path, const mpz_t N, const mpz_t a24, const mpz_t Q,
                         unsigned long long D, unsigned long long B1, unsigned long long B2,
                         unsigned long long sigma, const std::vector<unsigned long long> &bj,
                         const std::vector<std::string> &bx,
                         const std::vector<std::string> &F, unsigned long long degree)
{
    FILE *f = std::fopen(path, "wb");
    if (!f) return false;
    char *nh = mpz_get_str(nullptr, 16, N);
    char *nd = mpz_get_str(nullptr, 10, N);
    char *ah = mpz_get_str(nullptr, 16, a24);
    char *qh = mpz_get_str(nullptr, 16, Q);
    std::fprintf(f, "stage2_tree_F_dump v1\n");
    std::fprintf(f, "N_dec %s\n", nd);
    std::fprintf(f, "N_hex %s\n", nh);
    std::fprintf(f, "N_bits %ld\n", (long)mpz_sizeinbase(N, 2));
    std::fprintf(f, "D %llu\nB1 %llu\nB2 %llu\nsigma %llu\n",
                 (unsigned long long)D, (unsigned long long)B1, (unsigned long long)B2,
                 (unsigned long long)sigma);
    std::fprintf(f, "a24_hex %s\n", ah);
    std::fprintf(f, "Q_hex %s\n", qh);
    std::fprintf(f, "baby_count %llu\n", (unsigned long long)bj.size());
    for (size_t i = 0; i < bj.size(); ++i)
        std::fprintf(f, "baby %llu %s\n", (unsigned long long)bj[i], bx[i].c_str());
    std::fprintf(f, "F_degree %llu\n", (unsigned long long)degree);
    for (const std::string &c : F) std::fprintf(f, "F %s\n", c.c_str());
    std::fprintf(f, "end\n");
    std::fclose(f);
    void (*freefunc)(void *, size_t) = nullptr;
    mp_get_memory_functions(nullptr, nullptr, &freefunc);
    freefunc(nh, std::strlen(nh) + 1);
    freefunc(nd, std::strlen(nd) + 1);
    freefunc(ah, std::strlen(ah) + 1);
    freefunc(qh, std::strlen(qh) + 1);
    return true;
}

/* ===================================================================================== *
 *  host helpers: mpz <-> the coefficient representation
 * ===================================================================================== */

static size_t words_for_bits(size_t bits) { return (bits + 63) / 64; }

/* w must already be sized W.  Returns false when the value does not fit in W words (a
   caller bug -- mpz_export would otherwise write past the end). */
static bool mpz_to_words(std::vector<unsigned long long> &w, size_t W, const mpz_t v)
{
    if (w.size() != W) w.assign(W, 0ull);
    else std::fill(w.begin(), w.end(), 0ull);
    if ((size_t)mpz_sizeinbase(v, 2) > W * 64) return false;
    size_t cnt = 0;
    mpz_export(w.data(), &cnt, -1, 8, 0, 0, v);
    return cnt <= W;
}

static void words_to_mpz(mpz_t out, const unsigned long long *w, size_t W)
{
    mpz_import(out, W, -1, 8, 0, 0, w);
}

/* mirror of stage2_tree_ref.cpp's affine_x for a point given as (X, Z):
   x = X/Z mod N, 0 for the identity, and the un-invertible case keeps X -- that is the
   "Z shares a factor with N" case, which the reference passes through unchanged.

   `_checked` returns false for that last case instead of hiding it (objective 3 / section 31.4):
   gcd(Z, N) > 1 means the point is the IDENTITY modulo a factor of N -- i.e. exactly the hit
   stage 2 is looking for -- and the host can then record that factor explicitly instead of
   letting an arbitrary representative X become a leaf of the giant product. */
static bool affine_x_gmp_checked(mpz_t out, const mpz_t X, const mpz_t Z, const mpz_t N)
{
    if (mpz_cmp_ui(Z, 0) == 0) { mpz_set_ui(out, 0); return true; }
    mpz_t inv;
    mpz_init(inv);
    const bool ok = (mpz_invert(inv, Z, N) != 0);
    if (!ok) {
        mpz_set(out, X);
    } else {
        mpz_mul(out, X, inv);
        mpz_mod(out, out, N);
    }
    mpz_clear(inv);
    return ok;
}

static void affine_x_gmp(mpz_t out, const mpz_t X, const mpz_t Z, const mpz_t N)
{
    (void)affine_x_gmp_checked(out, X, Z, N);
}

/* ---- THE PROJECTIVE LEAF (objective 4, section 42) -----------------------------------------
 *  The scale of the giant-point segment: S2G_GFINV_SEG points share ONE invertibility test and
 *  one inverse, and that is also the granularity at which the two leaf forms are mixed.
 *  Measured (section 41): an mpz_invert costs ~70 us for the production modulus while a modular
 *  multiply costs ~6.5 us, and a segment of 16 contains one of the ~0.9% DEGENERATE giant
 *  points with probability 13.4% -- which is what makes 16 the right size (8/16/32 give
 *  45.0/42.4/43.4 us per point on the OLD scheme). */
static const size_t S2G_GFINV_SEG = 16;

// Batch inverses of the EXISTING segment products, not of a larger point segment.
// Only a bounded 64-segment window is retained; no chunk-sized inverse array.
static const bool g_fold_flat = [] {
    const char *e=std::getenv("NTT_FOLD_FLAT");
    return e && *e && std::atoi(e)!=0;
}();
// Exact product of the actual returned Z words, not a Montgomery image product.
static const bool g_gfinv_seg_exact = [] {
    const char *e=std::getenv("NTT_GFINV_SEG_EXACT");return !e || std::atoi(e)!=0;
}();
static const bool g_gfinv_seg_check = [] {
    const char *e=std::getenv("NTT_GFINV_SEG_CHECK");return e && std::atoi(e)!=0;
}();
static const bool g_giant_seed_device = [] {
    const char *e=std::getenv("NTT_GIANT_SEED_DEVICE");return e && std::atoi(e)!=0;
}();
static constexpr bool g_giant_seed_pair = true;
static constexpr bool g_giant_base_cpu = false;
static const bool g_giant_seed_check = [] {
    const char *e=std::getenv("NTT_GIANT_SEED_CHECK");return e && std::atoi(e)!=0;
}();
struct GiantSeedStats {
    unsigned long long chunks=0,points=0,avoided_d2h_bytes=0,avoided_h2d_bytes=0;
    unsigned long long avoided_cpu_modmuls=0,avoided_montmuls=0,checked_words=0;
    unsigned long long segments=0,segment_checks=0,segment_fix_muls=0,fix_table_peak_bytes=0;
    unsigned long long base_builds=0,base_nonunits=0,paired_chunks=0,paired_ladders=0;
    unsigned long long base_cpu_builds=0,base_gpu_builds=0,base_h2d_bytes=0,base_checked_words=0;
    double base_cpu_seconds=0,base_build_seconds=0;
    unsigned long long scalar_h2d_avoided=0,base_d2h_bytes=0;
} g_giant_seed;
static const bool g_gfinv_batch = [] {
    const char *e=std::getenv("NTT_GFINV_BATCH"); return e && std::atoi(e)!=0;
}();
struct GfinvStats {
    unsigned long long requests=0,cache_hits=0,groups=0,segments=0,group_attempts=0,
        group_failures=0,individual_attempts=0,good=0,nonunits=0,scratch_peak_bytes=0;
    double t_prepare=0;
};
static GfinvStats g_gfinv;
struct GfinvBatch {
    static constexpr size_t GROUP=64;
    const std::vector<unsigned long long> &products;
    size_t W,base=(size_t)-1,count=0;
    mpz_srcptr N;
    bool enabled,unit[GROUP]{};
    mpz_t prefix[GROUP+1],inverse[GROUP],acc,value;
    GfinvStats &stats;
    GfinvBatch(const std::vector<unsigned long long> &p,size_t w,mpz_srcptr n,bool on,
               GfinvStats &s=g_gfinv):products(p),W(w),N(n),enabled(on),stats(s) {
        if(enabled) {
            for(size_t j=0;j<=GROUP;++j)mpz_init(prefix[j]);
            for(size_t j=0;j<GROUP;++j)mpz_init(inverse[j]);
            mpz_inits(acc,value,nullptr);
        }
    }
    GfinvBatch(const GfinvBatch &)=delete;
    GfinvBatch &operator=(const GfinvBatch &)=delete;
    ~GfinvBatch() {
        if(enabled) {
            for(size_t j=0;j<=GROUP;++j)mpz_clear(prefix[j]);
            for(size_t j=0;j<GROUP;++j)mpz_clear(inverse[j]);
            mpz_clears(acc,value,nullptr);
        }
    }
    void prepare(size_t index) {
        const double t0=now_s();base=index/GROUP*GROUP;
        count=std::min(GROUP,products.size()/W-base);
        ++stats.groups;stats.segments+=count;
        mpz_set_ui(prefix[0],1);
        for(size_t j=0;j<count;++j) {
            words_to_mpz(value,products.data()+(base+j)*W,W);
            mpz_mul(prefix[j+1],prefix[j],value);mpz_mod(prefix[j+1],prefix[j+1],N);
        }
        ++stats.group_attempts;
        if(mpz_invert(acc,prefix[count],N)) {
            // Product is a unit iff every factor is a unit, including composite N.
            for(size_t j=count;j--;) {
                mpz_mul(inverse[j],acc,prefix[j]);mpz_mod(inverse[j],inverse[j],N);
                words_to_mpz(value,products.data()+(base+j)*W,W);
                mpz_mul(acc,acc,value);mpz_mod(acc,acc,N);unit[j]=true;
            }
            stats.good+=count;
        } else {
            // Recover the EXACT original per-segment classification, not a group-wide failure.
            ++stats.group_failures;
            for(size_t j=0;j<count;++j) {
                words_to_mpz(value,products.data()+(base+j)*W,W);
                ++stats.individual_attempts;unit[j]=mpz_invert(inverse[j],value,N)!=0;
                if(unit[j])++stats.good;else {++stats.nonunits;mpz_set_ui(inverse[j],0);}
            }
        }
        unsigned long long bytes=sizeof(mp_limb_t)*(acc[0]._mp_alloc+value[0]._mp_alloc);
        for(size_t j=0;j<=GROUP;++j)bytes+=sizeof(mp_limb_t)*prefix[j][0]._mp_alloc;
        for(size_t j=0;j<GROUP;++j)bytes+=sizeof(mp_limb_t)*inverse[j][0]._mp_alloc;
        stats.scratch_peak_bytes=std::max(stats.scratch_peak_bytes,bytes);
        stats.t_prepare+=now_s()-t0;
    }
    bool get(mpz_t out,size_t index) {
        if(!enabled || !W || products.size()%W || index>=products.size()/W) {
            std::fprintf(stderr,"%s: FATAL: segment inverse cache index outside grid\n",NTT_PROBE_NAME);std::exit(3);
        }
        ++stats.requests;
        if(base==(size_t)-1 || index<base || index>=base+count)prepare(index);else ++stats.cache_hits;
        const size_t j=index-base;mpz_set(out,inverse[j]);
        static bool poisoned=false;
        const char *fault=std::getenv("NTT_GFINV_BATCH_TEST_BAD");
        if(unit[j] && fault && std::atoi(fault) && !poisoned) {mpz_add_ui(out,out,1);poisoned=true;}
        return unit[j];
    }
};

static void gfinv_batch_fixture(const mpz_t actual,size_t actualW)
{
    unsigned long long cases=0,checks=0,nonunits=0;
    GfinvStats stats;mpz_t N,value,want,got,test;mpz_inits(N,value,want,got,test,nullptr);
    const size_t sizes[]={0,1,15,16,17,63,64,65,127,129};
    for(int modulus=0;modulus<3;++modulus) {
        if(!modulus)mpz_set(N,actual);else mpz_set_ui(N,modulus==1?15:35);
        const size_t W=modulus?1:actualW;
        for(size_t size:sizes)for(int pattern=0;pattern<3;++pattern) {
            std::vector<unsigned long long> words(size*W),tmp;
            for(size_t j=0;j<size;++j) {
                mpz_set_ui(value,pattern?2+37*j:1);
                if(pattern==2 && (j==0 || j==63 || j==64 || j+1==size))mpz_set_ui(value,0);
                mpz_mod(value,value,N);mpz_to_words(tmp,W,value);
                std::copy(tmp.begin(),tmp.end(),words.begin()+j*W);
            }
            GfinvBatch cache(words,W,N,true,stats);++cases;
            for(size_t j=0;j<size;++j)for(int repeat=0;repeat<(j%17==0?2:1);++repeat) {
                words_to_mpz(value,words.data()+j*W,W);
                const bool expected=mpz_invert(want,value,N)!=0,unit=cache.get(got,j);++checks;
                if(!expected)++nonunits;
                if(unit!=expected || (unit && mpz_cmp(want,got)) || (!unit && mpz_sgn(got))) {
                    std::fprintf(stderr,"%s: FATAL: segment inverse GMP mismatch modulus=%d pattern=%d size=%llu index=%llu\n",
                        NTT_PROBE_NAME,modulus,pattern,(unsigned long long)size,(unsigned long long)j);std::exit(3);
                }
                if(unit) {
                    mpz_mul(test,value,got);mpz_mod(test,test,N);
                    if(mpz_cmp_ui(test,1)) {std::fprintf(stderr,"%s: FATAL: segment inverse identity mismatch\n",NTT_PROBE_NAME);std::exit(3);}
                }
            }
        }
    }
    stage2_log::print(stage2_log::debug, "gfinv_fixture: cases=%llu checks=%llu nonunits=%llu group_failures=%llu cache_hits=%llu scratch_peak_bytes=%llu bad=0\n",
        cases,checks,nonunits,stats.group_failures,stats.cache_hits,stats.scratch_peak_bytes);
    mpz_clears(N,value,want,got,test,nullptr);
}


/* dst = -x mod n, W words, for x < n.  Pure word arithmetic: no GMP objects at all, which is
   the whole point of the projective leaf.  x == 0 gives 0 (the old affine path's value for the
   identity). */
static void words_neg_mod_n(std::vector<unsigned long long> &dst, size_t off,
                            const unsigned long long *x, const unsigned long long *n, size_t W)
{
    bool zero = true;
    for (size_t i = 0; i < W; ++i)
        if (x[i]) { zero = false; break; }
    if (zero) {
        for (size_t i = 0; i < W; ++i) dst[off + i] = 0ull;
        return;
    }
    unsigned long long borrow = 0;
    for (size_t i = 0; i < W; ++i) {
        const unsigned long long t = n[i] - x[i];
        const unsigned long long b1 = (n[i] < x[i]) ? 1ull : 0ull;
        const unsigned long long d = t - borrow;
        const unsigned long long b2 = (t < borrow) ? 1ull : 0ull;
        dst[off + i] = d;
        borrow = b1 | b2;
    }
}

/* the per-segment product of the z values, mod N, computed ON THE HOST.  Used by the LADDER
   path (whose outputs are in the normal domain, where a device Montgomery product would be
   wrong); the chain path gets the same quantity from the device, out of the images it already
   has.  Either way it is the product of the values AS RETURNED -- the scale the projective
   leaves carry -- so the host cannot tell the two apart. */
static void gfinv_segprod_host(std::vector<unsigned long long> &out, size_t npts, size_t nw,
                               const std::vector<unsigned long long> &gz, const mpz_t N)
{
    const size_t nseg = (npts + S2G_GFINV_SEG - 1) / S2G_GFINV_SEG;
    out.assign(nseg * nw, 0ull);
    mpz_t p, t;
    mpz_inits(p, t, nullptr);
    std::vector<unsigned long long> tmp(nw, 0ull);
    for (size_t s = 0; s < nseg; ++s) {
        const size_t a = s * S2G_GFINV_SEG;
        const size_t b = ((a + S2G_GFINV_SEG) < npts) ? (a + S2G_GFINV_SEG) : npts;
        words_to_mpz(p, &gz[a * nw], nw);
        for (size_t i = a + 1; i < b; ++i) {
            words_to_mpz(t, &gz[i * nw], nw);
            mpz_mul(p, p, t);
            mpz_mod(p, p, N);
        }
        mpz_to_words(tmp, nw, p);
        std::copy(tmp.begin(), tmp.end(), out.begin() + (long)(s * nw));
    }
    mpz_clears(p, t, nullptr);
}

/* ===================================================================================== *
 *  device: x-only Montgomery arithmetic mod N and the reference's ladder
 *
 *  Every formula below is the reference's, with each value replaced by its MONTGOMERY IMAGE
 *  (x -> x*R mod N, R = 2^(64*nw)).  That is legitimate because xdbl/xadd are polynomial
 *  identities in (X, Z, a24) built only from additions, subtractions and multiplications, and
 *  Montgomery multiplication is a ring homomorphism (Mont(a,b) = a*b*R^-1, hence
 *  Mont(aR,bR) = abR), while plain scalar multipliers such as the 4 in "4XZ" keep their value.
 *  So the ladder computes the Montgomery image of the same projective point and the affine
 *  ratio X/Z is identical mod N; converting back is one Mont(x, 1) per coordinate.  (Working
 *  in the Montgomery domain is also why the projective scalings of xdbl (c^4) and xadd (c^5)
 *  are not a problem here: any nonzero scaling of a pair represents the same point, and X/Z
 *  is what the host sees.)
 * ===================================================================================== */

/* (hi, lo) = x*y + z + c, exactly */
__device__ __forceinline__ void s2g_mac(unsigned long long x, unsigned long long y,
                                        unsigned long long z, unsigned long long c,
                                        unsigned long long &lo, unsigned long long &hi)
{
    const unsigned long long p0 = x * y;
    const unsigned long long p1 = __umul64hi(x, y);
    unsigned long long s = p0 + z;
    unsigned long long k = (s < p0) ? 1ull : 0ull;
    s += c;
    k += (s < c) ? 1ull : 0ull;
    lo = s;
    hi = p1 + k;
}

/* r = (r - N) mod 2^(64*nw); returns the borrow */
__device__ __forceinline__ unsigned long long s2g_sub_n(unsigned long long *r,
                                                        const unsigned long long *n, int nw)
{
    unsigned long long borrow = 0;
    for (int i = 0; i < nw; ++i) {
        const unsigned long long d = r[i] - n[i];
        const unsigned long long b1 = (r[i] < n[i]) ? 1ull : 0ull;
        const unsigned long long d2 = d - borrow;
        const unsigned long long b2 = (d < borrow) ? 1ull : 0ull;
        r[i] = d2;
        borrow = b1 + b2;
    }
    return borrow;
}

__device__ __forceinline__ bool s2g_ge_n(const unsigned long long *a,
                                         const unsigned long long *n, int nw)
{
    for (int i = nw - 1; i >= 0; --i)
        if (a[i] != n[i]) return a[i] > n[i];
    return true;                                          /* equal counts as >= */
}

/* r = a + b mod N  (a, b < N) */
template <int NW>
__device__ __forceinline__ void s2g_addmod(unsigned long long *r, const unsigned long long *a,
                                           const unsigned long long *b,
                                           const unsigned long long *n, int nw)
{
    unsigned long long c = 0;
    for (int i = 0; i < nw; ++i) {
        const unsigned long long s1 = a[i] + b[i];
        const unsigned long long c1 = (s1 < a[i]) ? 1ull : 0ull;
        const unsigned long long s2 = s1 + c;
        const unsigned long long c2 = (s2 < c) ? 1ull : 0ull;
        r[i] = s2;
        c = c1 + c2;
    }
    if (c != 0 || s2g_ge_n(r, n, nw)) s2g_sub_n(r, n, nw);   /* a+b < 2N: one subtraction */
}

/* r = a - b mod N  (a, b < N) */
template <int NW>
__device__ __forceinline__ void s2g_submod(unsigned long long *r, const unsigned long long *a,
                                           const unsigned long long *b,
                                           const unsigned long long *n, int nw)
{
    unsigned long long borrow = 0;
    for (int i = 0; i < nw; ++i) {
        const unsigned long long d = a[i] - b[i];
        const unsigned long long b1 = (a[i] < b[i]) ? 1ull : 0ull;
        const unsigned long long d2 = d - borrow;
        const unsigned long long b2 = (d < borrow) ? 1ull : 0ull;
        r[i] = d2;
        borrow = b1 + b2;
    }
    if (borrow) {
        unsigned long long c = 0;
        for (int i = 0; i < nw; ++i) {
            const unsigned long long s1 = r[i] + n[i];
            const unsigned long long c1 = (s1 < r[i]) ? 1ull : 0ull;
            const unsigned long long s2 = s1 + c;
            const unsigned long long c2 = (s2 < c) ? 1ull : 0ull;
            r[i] = s2;
            c = c1 + c2;
        }
    }
}

/* One symbol makes the indexed divisor the first field, independently of the
   compiler's ordering of separate constant symbols. The point flag is updated
   separately; normalized-divisor uploads must never overwrite it. */
struct S2GDeviceConstants {
    unsigned long long divisor[ecm_stage2::max_words];
    int point_mersenne_bits;
};
static_assert(offsetof(S2GDeviceConstants, divisor)==0, "divisor must lead constant storage");
__device__ __constant__ S2GDeviceConstants g_s2g_device_constants={};
#include "stage2/stage2_point_mersenne.cuh"

/* r = a*b*R^-1 mod N, R = 2^(64*nw), a,b < N.  Schoolbook product + Montgomery reduction
   (SOS/REDC).  The result is < 2N, so ONE conditional subtraction of N finishes it.
   ninv = -N^-1 mod 2^64 (N odd).  Checked against GMP for every shape at startup. */
template <int NW>
__device__ __forceinline__ void s2g_mont_mul(unsigned long long *r,
                                             const unsigned long long *a,
                                             const unsigned long long *b,
                                             const unsigned long long *n,
                                             unsigned long long ninv, int nw)
{
    unsigned long long t[2 * NW + 2];
    for (int i = 0; i < 2 * nw + 2; ++i) t[i] = 0;
    for (int i = 0; i < nw; ++i) {
        unsigned long long c = 0;
        const unsigned long long bi = b[i];
        for (int j = 0; j < nw; ++j) {
            unsigned long long lo, hi;
            s2g_mac(a[j], bi, t[i + j], c, lo, hi);
            t[i + j] = lo;
            c = hi;
        }
        t[i + nw] = c;                 /* untouched before this iteration (SOS invariant) */
    }
    unsigned long long out[NW];
    const int mersenne_bits=g_s2g_device_constants.point_mersenne_bits;
    if(mersenne_bits){s2g_mersenne_mont_reduce<NW>(r,t,n,nw,mersenne_bits,out);return;}
    for (int i = 0; i < nw; ++i) {
        const unsigned long long m = t[i] * ninv;
        unsigned long long c = 0;
        for (int j = 0; j < nw; ++j) {
            unsigned long long lo, hi;
            s2g_mac(m, n[j], t[i + j], c, lo, hi);
            t[i + j] = lo;
            c = hi;
        }
        int k = i + nw;
        while (c != 0) {
            const unsigned long long s = t[k] + c;
            c = (s < c) ? 1ull : 0ull;
            t[k] = s;
            ++k;
        }
    }
    for (int i = 0; i < nw; ++i) out[i] = t[nw + i];
    if (t[2 * nw] != 0 || s2g_ge_n(out, n, nw)) s2g_sub_n(out, n, nw);
    for (int i = 0; i < nw; ++i) r[i] = out[i];
}

/* r = 2p: the reference's xdbl, in Montgomery images */
template <int NW>
__device__ __forceinline__ void s2g_xdbl(unsigned long long *rx, unsigned long long *rz,
                                         const unsigned long long *px, const unsigned long long *pz,
                                         const unsigned long long *a24, const unsigned long long *n,
                                         unsigned long long ninv, int nw)
{
    unsigned long long t1[NW], t2[NW], t3[NW], t4[NW];
    s2g_addmod<NW>(t1, px, pz, n, nw);
    s2g_submod<NW>(t2, px, pz, n, nw);
    s2g_mont_mul<NW>(t3, t1, t1, n, ninv, nw);            /* (X+Z)^2 */
    s2g_mont_mul<NW>(t4, t2, t2, n, ninv, nw);            /* (X-Z)^2 */
    s2g_mont_mul<NW>(rx, t3, t4, n, ninv, nw);            /* X2 = (X+Z)^2 (X-Z)^2 */
    s2g_submod<NW>(t1, t3, t4, n, nw);                    /* 4XZ */
    s2g_mont_mul<NW>(t2, a24, t1, n, ninv, nw);
    s2g_addmod<NW>(t2, t2, t4, n, nw);                    /* a24*4XZ + (X-Z)^2 */
    s2g_mont_mul<NW>(rz, t1, t2, n, ninv, nw);
}

/* r = a/2 mod odd N, canonical a<N. Add N when odd BEFORE shifting;
   the carry out of the top limb belongs to the (nw+1)-limb sum, even when N
   almost fills the radix. Linear in the Montgomery image too. In-place safe. */
template <int NW>
__device__ __forceinline__ void s2g_halfmod(unsigned long long *r,
                                            const unsigned long long *a,
                                            const unsigned long long *n, int nw)
{
    const bool odd=(a[0]&1ull)!=0;
    unsigned long long carry=0;
    for(int i=0;i<nw;++i) {
        const auto v=a[i], add=odd ? n[i] : 0ull;
        const auto s=v+add, c1=(s<v) ? 1ull : 0ull;
        const auto u=s+carry, c2=(u<s) ? 1ull : 0ull;
        r[i]=u; carry=c1+c2;
    }
    for(int i=nw-1;i>=0;--i) {
        const auto v=r[i];
        r[i]=(v>>1)|(carry<<63); carry=v&1ull;
    }
}

/* r = p + q with diff = p - q: the reference's xadd, in Montgomery images.
   The output is written only at the end, so the ladder may alias it with p and q (the
   reference documents exactly this trap: writing r.X early corrupted Z3). */
template <int NW, bool XADD6=false>
__device__ __forceinline__ void s2g_xadd(unsigned long long *rx, unsigned long long *rz,
                                         const unsigned long long *px, const unsigned long long *pz,
                                         const unsigned long long *qx, const unsigned long long *qz,
                                         const unsigned long long *dx, const unsigned long long *dz,
                                         const unsigned long long *n, unsigned long long ninv,
                                         int nw)
{
    unsigned long long a[NW], b[NW], tx[NW], tz[NW];
    if(XADD6) {
        // u=(Xp+Zp)*(Xq-Zq), v=(Xp-Zp)*(Xq+Zq), all in Montgomery images.
        s2g_addmod<NW>(a,px,pz,n,nw); s2g_submod<NW>(b,px,pz,n,nw);
        s2g_addmod<NW>(tx,qx,qz,n,nw); s2g_submod<NW>(tz,qx,qz,n,nw);
        s2g_mont_mul<NW>(a,a,tz,n,ninv,nw);
        s2g_mont_mul<NW>(b,b,tx,n,ninv,nw);
        s2g_addmod<NW>(tx,a,b,n,nw); s2g_submod<NW>(tz,a,b,n,nw);
        // Halves recover the old coordinate scale exactly; the Z sign is squared.
        s2g_halfmod<NW>(tx,tx,n,nw); s2g_halfmod<NW>(tz,tz,n,nw);
        s2g_mont_mul<NW>(tx,tx,tx,n,ninv,nw);
        s2g_mont_mul<NW>(tz,tz,tz,n,ninv,nw);
        s2g_mont_mul<NW>(a,dz,tx,n,ninv,nw);
        s2g_mont_mul<NW>(b,dx,tz,n,ninv,nw);
        // Delay both writes, including when the output aliases the difference.
        for(int i=0;i<nw;++i) {rx[i]=a[i];rz[i]=b[i];}
        return;
    }
    s2g_mont_mul<NW>(a, px, qx, n, ninv, nw);
    s2g_mont_mul<NW>(b, pz, qz, n, ninv, nw);
    s2g_submod<NW>(a, a, b, n, nw);                /* X_P X_Q - Z_P Z_Q */
    s2g_mont_mul<NW>(a, a, a, n, ninv, nw);
    s2g_mont_mul<NW>(tx, dz, a, n, ninv, nw);
    s2g_mont_mul<NW>(a, px, qz, n, ninv, nw);
    s2g_mont_mul<NW>(b, pz, qx, n, ninv, nw);
    s2g_submod<NW>(a, a, b, n, nw);                /* X_P Z_Q - Z_P X_Q */
    s2g_mont_mul<NW>(a, a, a, n, ninv, nw);
    s2g_mont_mul<NW>(tz, dx, a, n, ninv, nw);
    for (int i = 0; i < nw; ++i) { rx[i] = tx[i]; rz[i] = tz[i]; }
}

/* the reference's ladder(): MSB first, the difference point is p itself */
template <int NW, bool XADD6=false>
__device__ void s2g_ladder(unsigned long long k, const unsigned long long *px,
                           const unsigned long long *pz, const unsigned long long *a24,
                           const unsigned long long *n, unsigned long long ninv, int nw,
                           const unsigned long long *mone,          /* Montgomery image of 1 */
                           unsigned long long *rx, unsigned long long *rz)
{
    if (k == 0) {                                              /* identity (1 : 0) */
        for (int i = 0; i < nw; ++i) { rx[i] = mone[i]; rz[i] = 0; }
        return;                                                /* affine_x then gives 0 */
    }
    int top = 63;
    while (((k >> top) & 1ull) == 0) --top;
    unsigned long long r0x[NW], r0z[NW], r1x[NW], r1z[NW];
    for (int i = 0; i < nw; ++i) { r0x[i] = px[i]; r0z[i] = pz[i]; }
    s2g_xdbl<NW>(r1x, r1z, px, pz, a24, n, ninv, nw);
    for (int i = top - 1; i >= 0; --i) {
        if (((k >> i) & 1ull) != 0) {
            s2g_xadd<NW,XADD6>(r0x, r0z, r0x, r0z, r1x, r1z, px, pz, n, ninv, nw);
            s2g_xdbl<NW>(r1x, r1z, r1x, r1z, a24, n, ninv, nw);
        } else {
            s2g_xadd<NW,XADD6>(r1x, r1z, r0x, r0z, r1x, r1z, px, pz, n, ninv, nw);
            s2g_xdbl<NW>(r0x, r0z, r0x, r0z, a24, n, ninv, nw);
        }
    }
    for (int i = 0; i < nw; ++i) { rx[i] = r0x[i]; rz[i] = r0z[i]; }
}

/* one thread per point: (X, Z) = [j]Q, written back in the NORMAL domain so the host sees the
   same pair the CPU ladder produces */
template <int NW, bool XADD6=false>
__global__ void s2g_ladder_kernel(const unsigned long long *n, unsigned long long ninv, int nw,
                                  const unsigned long long *qx, const unsigned long long *qz,
                                  const unsigned long long *a24, const unsigned long long *mone,
                                  const unsigned long long *js, int npts,
                                  unsigned long long *out_x, unsigned long long *out_z, bool normal_output)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= npts) return;
    unsigned long long rx[NW], rz[NW], one[NW], nx[NW], nz[NW];
    /* GRID-STRIDE: the launcher bounds the number of threads per launch (a ladder is a long
       serial chain, and a multi-second kernel is killed by the display driver's watchdog), so
       one launch may have to cover more points than it has threads.  The stride version is the
       same arithmetic, point by point, with a different work assignment. */
    for (int i = t; i < npts; i += gridDim.x * blockDim.x) {
        s2g_ladder<NW,XADD6>(js[i], qx, qz, a24, n, ninv, nw, mone, rx, rz);
        if(normal_output) {
            for (int j = 0; j < NW; ++j) one[j] = 0;
            one[0] = 1;
            s2g_mont_mul<NW>(nx, rx, one, n, ninv, nw);
            s2g_mont_mul<NW>(nz, rz, one, n, ninv, nw);
        }
        for (int j = 0; j < nw; ++j) {
            out_x[(size_t)i * nw + j] = normal_output ? nx[j] : rx[j];
            out_z[(size_t)i * nw + j] = normal_output ? nz[j] : rz[j];
        }
    }
}

/* device mod-N arithmetic selftest: out[i] = a[i]*b[i]*R^-1 mod N, compared with GMP */
template <int NW>
__global__ void s2g_mont_test_kernel(const unsigned long long *a, const unsigned long long *b,
                                     const unsigned long long *n, unsigned long long ninv,
                                     int nw, int cases, unsigned long long *out)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= cases) return;
    unsigned long long r[NW];
    /* the caller's arrays are packed with stride nw (not NW: NW is only the template's
       upper bound, chosen from the word-count dispatch) */
    s2g_mont_mul<NW>(r, a + (size_t)t * nw, b + (size_t)t * nw, n, ninv, nw);
    for (int i = 0; i < nw; ++i) out[(size_t)t * nw + i] = r[i];
}

/* Optional algebra gate: arbitrary canonical coordinates, including nonunits and
   zero Z, and independent p/q/difference/cross-output aliases. Both template variants
   are checked against GMP's old polynomial coordinates, without projective division. */
template <int NW, bool XADD6>
__global__ void s2g_xadd_test_kernel(const unsigned long long *input,
                                     const unsigned long long *n, unsigned long long ninv,
                                     int nw,int cases,unsigned long long *output)
{
    const int t=blockIdx.x*blockDim.x+threadIdx.x;
    if(t>=cases*5) return;
    const int c=t/5, alias=t%5;
    const auto *s=input+(size_t)c*6*nw;
    unsigned long long px[NW],pz[NW],qx[NW],qz[NW],dx[NW],dz[NW],tx[NW],tz[NW];
    for(int i=0;i<nw;++i) {px[i]=s[i];pz[i]=s[nw+i];qx[i]=s[2*nw+i];
                          qz[i]=s[3*nw+i];dx[i]=s[4*nw+i];dz[i]=s[5*nw+i];}
    auto *rx=alias==1 || alias==4 ? px : alias==2 ? qx : alias==3 ? dx : tx;
    auto *rz=alias==1 ? pz : alias==2 || alias==4 ? qz : alias==3 ? dz : tz;
    s2g_xadd<NW,XADD6>(rx,rz,px,pz,qx,qz,dx,dz,n,ninv,nw);
    auto *out=output+(size_t)t*3*nw;
    for(int i=0;i<nw;++i) {out[i]=rx[i];out[nw+i]=rz[i];}
    // Out-of-place here; candidate xADD also exercises in-place modular halves.
    s2g_halfmod<NW>(out+2*nw,s,n,nw);
}

/* the word-count dispatch: seven instantiations cover every N up to 16384 bits */
#define S2G_DISPATCH(NWD, FN, ...)                                                      \
    do {                                                                                \
        const int _nwd = (NWD);                                                         \
        if (_nwd <= 4) { FN<4>(__VA_ARGS__); }                                          \
        else if (_nwd <= 8) { FN<8>(__VA_ARGS__); }                                     \
        else if (_nwd <= 16) { FN<16>(__VA_ARGS__); }                                   \
        else if (_nwd <= 32) { FN<32>(__VA_ARGS__); }                                   \
        else if (_nwd <= 64) { FN<64>(__VA_ARGS__); }                                   \
        else if (_nwd <= 128) { FN<128>(__VA_ARGS__); }                                 \
        else if (_nwd <= 256) { FN<256>(__VA_ARGS__); }                                 \
        else {                                                                          \
            std::fprintf(stderr, "%s: N has %d words (%d bits); the ladder supports up " \
                                 "to 256 words (16384 bits)\n",                          \
                         NTT_PROBE_NAME, _nwd, _nwd * 64);                              \
            std::exit(2);                                                               \
        }                                                                               \
    } while (0)

template <int NW>
static void s2g_launch_mont_test(int nw, int cases, const unsigned long long *da,
                                 const unsigned long long *db, const unsigned long long *dn,
                                 unsigned long long ninv, unsigned long long *dout)
{
    const unsigned int th = 128;
    const unsigned int bl = (unsigned int)((cases + th - 1) / th);
    s2g_mont_test_kernel<NW><<<bl, th>>>(da, db, dn, ninv, nw, cases, dout);
}

template <int NW>
static void s2g_launch_xadd_test(int nw,int cases,const unsigned long long *in,
                                 const unsigned long long *n,unsigned long long ninv,
                                 unsigned long long *out,bool candidate)
{
    const unsigned int th=64,bl=(cases*5+th-1)/th;
    if(candidate) s2g_xadd_test_kernel<NW,true><<<bl,th>>>(in,n,ninv,nw,cases,out);
    else s2g_xadd_test_kernel<NW,false><<<bl,th>>>(in,n,ninv,nw,cases,out);
}

/* ONE ladder launch must stay SHORT.  A ladder is a serial 5261-bit chain (~2*S steps), so on
   the real shape a single launch over a whole 207k-point chunk is a ~12-second kernel -- and a
   kernel that long gets the process KILLED with no diagnostic at all: the display driver's
   watchdog (TDR) resets the device and terminates the host process, which is why the real-shape
   runs of section 18.3 died 60-270 s in with an empty stderr and a bare "exit=1" and why the
   Event Log shows nvlddmkm id 13/153 at exactly those moments.  The cap makes every launch
   bounded; the grid-stride loop keeps the same total work and the same result.  Override with
   NTT_LADDER_CAP (0 or unset = the default), e.g. NTT_LADDER_CAP=2048 for a slower/safer run. */
static constexpr bool g_xadd6 = true;
static int g_ladder_cap = 8192;
static void ladder_cap_init(void)
{
    const char *e = std::getenv("NTT_LADDER_CAP");
    if (e && *e) g_ladder_cap = std::atoi(e);
    if (g_ladder_cap <= 0) g_ladder_cap = 8192;
    if (g_ladder_cap > (1 << 24)) g_ladder_cap = 1 << 24;   /* 2^24 threads = 2^18 blocks:
                                                               far inside the grid limit */
}

template <int NW>
static void s2g_launch_ladder(int nw, int npts, const unsigned long long *dn,
                              unsigned long long ninv, const unsigned long long *dqx,
                              const unsigned long long *dqz, const unsigned long long *da24,
                              const unsigned long long *dmone, const unsigned long long *djs,
                              unsigned long long *dx, unsigned long long *dz, bool normal_output=true)
{
    const unsigned int th = 64;
    /* the same kernel, launched once per <= g_ladder_cap points (grid-stride inside).
       THE GRID IS SIZED TO THE CHUNK, NOT TO THE CAP.  This one line was worth 100x: the launch
       used to be `ceil(g_ladder_cap/64)` blocks whatever the chunk was, i.e. 8192 threads for a
       ONE-point chain, and this kernel's arrays are indexed with the runtime `nw` and therefore
       live in LOCAL memory (see the note on s2g_mont_mul) -- so every launch committed
       ~8192 x 5 KB = 40 MB of local memory for nothing.  Measured before the fix: a 1-point
       launch (the setup chain, 25 sequential prime powers) cost 0.51 s and a 2768-point batch
       cost 1.29 s, i.e. almost all of it launch overhead; the naming loop's 664388 candidate
       ladders cost 309 s of a 318 s run (section 26).  The thread->point mapping and the
       grid-stride arithmetic are unchanged, so the result is bit-identical. */
    for (int p0 = 0; p0 < npts; p0 += g_ladder_cap) {
        const int m = ((npts - p0) < g_ladder_cap) ? (npts - p0) : g_ladder_cap;
        const unsigned int bl = (unsigned int)((m + th - 1) / th);
        if constexpr(g_xadd6)
            s2g_ladder_kernel<NW,true><<<bl,th>>>(dn,ninv,nw,dqx,dqz,da24,dmone,djs+p0,m,
                                                dx+(size_t)p0*nw,dz+(size_t)p0*nw,normal_output);
        else s2g_ladder_kernel<NW><<<bl, th>>>(dn, ninv, nw, dqx, dqz, da24, dmone, djs + p0, m,
                                          dx + (size_t)p0 * nw, dz + (size_t)p0 * nw, normal_output);
    }
}

/* H=[D]Q is already a Montgomery pair. A ladder on H ends with the two
   adjacent points [i]H and [i+1]H; retain both instead of running it twice.
   This deliberately leaves the original ladder/kernel unchanged for A/B. */
template <int NW, bool XADD6>
__global__ void s2g_seed_pair_kernel(const unsigned long long *n,unsigned long long ninv,int nw,
                                    const unsigned long long *hx,const unsigned long long *hz,
                                    const unsigned long long *a24,const unsigned long long *mone,
                                    unsigned long long clo,unsigned long long npts,
                                    unsigned long long per_block,unsigned long long blocks,
                                    unsigned long long offset,unsigned long long count,
                                    unsigned long long *ox,unsigned long long *oz)
{
    const unsigned long long local=blockIdx.x*(unsigned long long)blockDim.x+threadIdx.x;
    if(local>=count)return;
    const unsigned long long b=offset+local,k=clo+b*per_block;
    unsigned long long ax[NW],az[NW],bx[NW],bz[NW];
    if(k==0) {
        for(int j=0;j<nw;++j){ax[j]=mone[j];az[j]=0;bx[j]=hx[j];bz[j]=hz[j];}
    } else {
        int top=63;while(((k>>top)&1ull)==0)--top;
        for(int j=0;j<nw;++j){ax[j]=hx[j];az[j]=hz[j];}
        s2g_xdbl<NW>(bx,bz,hx,hz,a24,n,ninv,nw);
        for(int bit=top-1;bit>=0;--bit) {
            if((k>>bit)&1ull) {
                s2g_xadd<NW,XADD6>(ax,az,ax,az,bx,bz,hx,hz,n,ninv,nw);
                s2g_xdbl<NW>(bx,bz,bx,bz,a24,n,ninv,nw);
            } else {
                s2g_xadd<NW,XADD6>(bx,bz,ax,az,bx,bz,hx,hz,n,ninv,nw);
                s2g_xdbl<NW>(ax,az,ax,az,a24,n,ninv,nw);
            }
        }
    }
    const bool single=b*per_block+1>=npts;
    for(int j=0;j<nw;++j) {
        ox[(size_t)(2*b)*nw+j]=ax[j];oz[(size_t)(2*b)*nw+j]=az[j];
        ox[(size_t)(2*b+1)*nw+j]=single?ax[j]:bx[j];
        oz[(size_t)(2*b+1)*nw+j]=single?az[j]:bz[j];
        if(b==0){ox[(size_t)(2*blocks)*nw+j]=hx[j];oz[(size_t)(2*blocks)*nw+j]=hz[j];}
    }
}

template <int NW>
static void s2g_launch_seed_pair(int nw,unsigned long long clo,unsigned long long npts,
                               unsigned long long per_block,unsigned long long blocks,
                               const unsigned long long *dn,unsigned long long ninv,
                               const unsigned long long *hx,const unsigned long long *hz,
                               const unsigned long long *a24,const unsigned long long *mone,
                               unsigned long long *ox,unsigned long long *oz)
{
    for(unsigned long long offset=0;offset<blocks;offset+=g_ladder_cap) {
        const unsigned long long count=std::min(blocks-offset,(unsigned long long)g_ladder_cap);
        const unsigned int th=64,bl=(unsigned int)((count+th-1)/th);
        if constexpr(g_xadd6)s2g_seed_pair_kernel<NW,true><<<bl,th>>>(dn,ninv,nw,hx,hz,a24,mone,
            clo,npts,per_block,blocks,offset,count,ox,oz);
        else s2g_seed_pair_kernel<NW,false><<<bl,th>>>(dn,ninv,nw,hx,hz,a24,mone,
            clo,npts,per_block,blocks,offset,count,ox,oz);
    }
}

/* ===================================================================================== *
 *  THE DIFFERENTIAL-ADDITION CHAIN FOR THE GIANT POINTS (objective 3, section 31)
 *
 *  The giant points are CONSECUTIVE MULTIPLES of one point: x_{(i+1)D} = xADD(x_{iD}, x_D,
 *  x_{(i-1)D}), so every point after the first two costs ONE differential addition -- 8 Montgomery
 *  multiplications -- instead of a full ~bitlen(i*D)-step ladder (~574 at the real shape).
 *
 *  WHY IT MATTERS: measured at A=1.94e12/D=570570 the ladder version spent 901 s of a 1957 s
 *  curve on this phase alone (46%): 3.4e6 points x 41 ladder steps x 14 multiplications.
 *
 *  A chain is SEQUENTIAL, so it is parallelised ACROSS the points, not inside one: one thread owns
 *  a block of `per_block` consecutive multiples and every block is seeded by two ladder points
 *  (a chain needs two consecutive values to start).  The points stay PROJECTIVE (X:Z) in the
 *  Montgomery domain -- the only consumer is the host's affine_x = X/Z mod N, which is invariant
 *  under projective scaling, so no inversion and no conversion is needed.
 * ===================================================================================== */
template <int NW, bool XADD6=false>
__global__ void s2g_chain_kernel(const unsigned long long *dn, unsigned long long ninv, int nw,
                                 const unsigned long long *ddx, const unsigned long long *ddz,
                                 const unsigned long long *dsx, const unsigned long long *dsz,
                                 const unsigned long long *esx, const unsigned long long *esz,
                                 unsigned long long npts, unsigned long long per_block,
                                 unsigned long long blocks,
                                 unsigned long long *out_x, unsigned long long *out_z,
                                 size_t seed_stride)
{
    const unsigned long long t = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (t >= blocks) return;
    const unsigned long long start = t * per_block;
    if (start >= npts) return;
    const unsigned long long end = ((start + per_block) < npts) ? (start + per_block) : npts;
    unsigned long long xa[NW], za[NW], xb[NW], zb[NW], xc[NW], zc[NW];
    for (int j = 0; j < nw; ++j) {
        xa[j] = dsx[(size_t)t * seed_stride * nw + j];
        za[j] = dsz[(size_t)t * seed_stride * nw + j];
        xb[j] = esx[(size_t)t * seed_stride * nw + j];
        zb[j] = esz[(size_t)t * seed_stride * nw + j];
    }
    for (int j = 0; j < nw; ++j) {
        out_x[(size_t)start * nw + j] = xa[j];
        out_z[(size_t)start * nw + j] = za[j];
    }
    if (end > start + 1)
        for (int j = 0; j < nw; ++j) {
            out_x[(size_t)(start + 1) * nw + j] = xb[j];
            out_z[(size_t)(start + 1) * nw + j] = zb[j];
        }
    for (unsigned long long k = start + 2; k < end; ++k) {
        /* x_k = x_{k-1} + x_1, with the difference x_{k-2}: p = x_{k-1}, q = x_D, diff = x_{k-2} */
        s2g_xadd<NW,XADD6>(xc, zc, xb, zb, ddx, ddz, xa, za, dn, ninv, nw);
        for (int j = 0; j < nw; ++j) {
            out_x[(size_t)k * nw + j] = xc[j];
            out_z[(size_t)k * nw + j] = zc[j];
            xa[j] = xb[j];
            za[j] = zb[j];
            xb[j] = xc[j];
            zb[j] = zc[j];
        }
    }
}

template <int NW>
static void s2g_launch_chain(int nw, unsigned long long blocks, unsigned long long npts,
                             unsigned long long per_block, unsigned long long ninv,
                             const unsigned long long *dn, const unsigned long long *ddx,
                             const unsigned long long *ddz, const unsigned long long *dsx,
                             const unsigned long long *dsz, const unsigned long long *esx,
                             const unsigned long long *esz, unsigned long long *ox,
                             unsigned long long *oz, size_t seed_stride=1)
{
    const unsigned int th = 64;
    const unsigned int bl = (unsigned int)((blocks + th - 1) / th);
    if constexpr(g_xadd6)
        s2g_chain_kernel<NW,true><<<bl,th>>>(dn,ninv,nw,ddx,ddz,dsx,dsz,esx,esz,npts,per_block,
                                           blocks,ox,oz,seed_stride);
    else s2g_chain_kernel<NW><<<bl, th>>>(dn, ninv, nw, ddx, ddz, dsx, dsz, esx, esz, npts, per_block,
                                    blocks, ox, oz, seed_stride);
}

/* ===================================================================================== *
 *  THE SEGMENT PRODUCTS OF THE GIANT z-COORDINATES (objective 4, section 42)
 *
 *  One thread per segment of `seg` consecutive giant points; ONE Montgomery multiplication per
 *  point after the first. The uncorrected result is Gamma/R^(m-1), where Gamma is the
 *  ordinary product of the actual returned Z WORDS. Default correction by Mont(p,R^m)
 *  returns exactly Gamma, including short tails; legacy mode is diagnostic only. The host asks
 *  ONE question per segment -- is this product invertible mod N? -- and the answer decides
 *  whether the segment's leaves can be written in the PROJECTIVE form [ -X, Z ] (two word-level
 *  operations, no GMP at all) instead of being converted to affine [ -x, 1 ] with a modular
 *  inversion per point.  See the note at the leaf loop for why the two forms may be MIXED and
 *  why the tree is then bit-identical to the old one after one global scaling.
 * ===================================================================================== */
template <int NW>
__global__ void s2g_segprod_kernel(const unsigned long long *dn, unsigned long long ninv, int nw,
                                   const unsigned long long *dz, unsigned long long npts,
                                   unsigned long long seg, unsigned long long nseg,
                                   unsigned long long *out, const unsigned long long *fix)
{
    const unsigned long long s = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (s >= nseg) return;
    const unsigned long long start = s * seg;
    if (start >= npts) return;
    const unsigned long long end = ((start + seg) < npts) ? (start + seg) : npts;
    unsigned long long p[NW], tmp[NW];
    for (int i = 0; i < nw; ++i) p[i] = dz[(size_t)start * nw + i];
    for (unsigned long long j = start + 1; j < end; ++j) {
        s2g_mont_mul<NW>(tmp, p, dz + (size_t)j * nw, dn, ninv, nw);
        for (int i = 0; i < nw; ++i) p[i] = tmp[i];
    }
    const size_t m=(size_t)(end-start);
    if(fix && m>1) {
        s2g_mont_mul<NW>(tmp,p,fix+m*nw,dn,ninv,nw);
        for(int i=0;i<nw;++i)p[i]=tmp[i];
    }
    for (int i = 0; i < nw; ++i) out[(size_t)s * nw + i] = p[i];
}

template <int NW>
static void s2g_launch_segprod(int nw, unsigned long long npts, unsigned long long seg,
                               unsigned long long ninv, const unsigned long long *dn,
                               const unsigned long long *dz, unsigned long long *out, const unsigned long long *fix=nullptr)
{
    if(!npts)return;
    const unsigned long long nseg = (npts + seg - 1) / seg;
    const unsigned int th = 64;
    const unsigned int bl = (unsigned int)((nseg + th - 1) / th);
    s2g_segprod_kernel<NW><<<bl, th>>>(dn, ninv, nw, dz, npts, seg, nseg, out, fix);
}

/* Actual returned X/Z words -> the same projective [-X,Z] linear leaf as the
   CPU path. No domain conversion: Gamma is a product of these exact Z words. */
__global__ void s2g_projective_leaf_kernel(const unsigned long long *n,int nw,
    const unsigned long long *x,const unsigned long long *z,size_t first,size_t count,
    unsigned long long *out)
{
    const size_t k=blockIdx.x*(size_t)blockDim.x+threadIdx.x;if(k>=count)return;
    const auto *xx=x+(first+k)*nw,*zz=z+(first+k)*nw;
    bool zero=true;for(int j=0;j<nw;++j)if(xx[j])zero=false;
    unsigned long long borrow=0;
    for(int j=0;j<nw;++j) {
        const unsigned long long t=n[j]-xx[j],b1=n[j]<xx[j],v=t-borrow,b2=t<borrow;
        borrow=b1|b2;out[(2*k)*nw+j]=zero?0:v;out[(2*k+1)*nw+j]=zz[j];
    }
}

/* ===================================================================================== *
 *  the polynomial layer: PolyN = m*W words, coefficient-major (see the header)
 * ===================================================================================== */

/* ===================================================================================== *
 *  THE BATCHED COST ACCOUNTING (slice S3)
 *
 *  Every multiplication of the batched structure is charged with the CPU reference's own
 *  formula (tools/bench/stage2_tree_ref.cpp, Cost::add):
 *
 *      bits = 2 * max(m1,m2) * (2*S + ceil(log2 max(m1,m2)))
 *
 *  with m1,m2 the COEFFICIENT COUNTS of the two operands -- so the numbers below are
 *  directly comparable with `stage2_tree_ref --cost`'s cost_model line, which is what
 *  acceptance gate 3 asks for.  The categories are the batched structure's own:
 *  f_tree (built once), g_tree (one per outer loop), fold (3 full-size multiplies per loop:
 *  G*H, rev(T)*rev(F)^-1, q*F), descent (H against the F tree) and inv (the Newton inverse
 *  of rev(F), computed ONCE and reused by every fold -- the reference's model has no such
 *  term, it is charged separately so the comparison stays honest).
 * ===================================================================================== */

enum { BC_FTREE = 0, BC_GTREE = 1, BC_FOLD = 2, BC_DESCENT = 3, BC_FINV = 4, BC_NCAT = 5 };
static const char *const kBatchedCat[BC_NCAT] = { "f_tree", "g_tree", "fold", "descent", "inv" };

static int ceil_log2_u64(unsigned long long m)
{
    int l = 0;
    while (l < 63 && (1ull << l) < m) ++l;
    return l;
}

struct MulCost {
    long S = 0;
    unsigned long long muls[BC_NCAT] = {0, 0, 0, 0, 0};
    unsigned long long bits[BC_NCAT] = {0, 0, 0, 0, 0};
    unsigned long long coeff[BC_NCAT] = {0, 0, 0, 0, 0};
    unsigned long long hcnt[64] = {}, hbits[64] = {};
    unsigned long long max_m1 = 0, max_m2 = 0, max_bits = 0;
    unsigned long long cat_mul_max[BC_NCAT] = {0, 0, 0, 0, 0};   /* widest operand per category */

    void add(int cat, size_t m1, size_t m2)
    {
        const unsigned long long m = (unsigned long long)((m1 > m2) ? m1 : m2);
        const int l = ceil_log2_u64(m);
        const unsigned long long b =
            2ull * m * (2ull * (unsigned long long)S + (unsigned long long)l);
        bits[cat] += b;
        ++muls[cat];
        coeff[cat] += (unsigned long long)m1 * (unsigned long long)m2;
        if (l < 64) { ++hcnt[l]; hbits[l] += b; }
        if (m > cat_mul_max[cat]) cat_mul_max[cat] = m;
        if (b > max_bits) { max_bits = b; max_m1 = m1; max_m2 = m2; }
    }
    unsigned long long tot_muls() const
    {
        unsigned long long t = 0;
        for (int i = 0; i < BC_NCAT; ++i) t += muls[i];
        return t;
    }
    unsigned long long tot_bits() const
    {
        unsigned long long t = 0;
        for (int i = 0; i < BC_NCAT; ++i) t += bits[i];
        return t;
    }
};

struct S4Reduce;

/* slice S4: the batched/device-reduced multiply path, shared by every multiply the tree makes.
   `red` is the device reduction (nullptr = do not use it, i.e. the pre-S4 host-GMP path);
   `d_out` is the reusable device buffer the reduced coefficients land in. */
struct S4Ctx {
    S4Reduce *red = nullptr;
    unsigned long long *d_out = nullptr;
    size_t d_out_cap = 0;
    unsigned long long launches = 0, muls = 0, groups = 0, level_calls = 0;
    long long sample_limit = 4096;      /* a full GMP check below this many coefficients */
    bool selftested = false;
    /* OBJECTIVE 4 (section 33): the DEVICE-side operand packing.  Before this, every batched
       multiply packed its operands in a host loop (36.8 us per polynomial multiplication,
       measured) and then uploaded the PACKED operands (20.2 us) -- 33% of the whole NTT budget.
       Packing on the device needs the raw coefficients on the device (P*W words per slice, which
       is ~9x SMALLER than the packed form at S=5261) and a digit buffer per operand. */
    unsigned long long *d_rawA = nullptr, *d_rawB = nullptr, *d_packA = nullptr, *d_packB = nullptr;
    size_t d_rawA_cap = 0, d_rawB_cap = 0, d_pack_cap = 0;
    size_t raw_capacity(const unsigned long long *p) const {
        return p==d_rawA ? d_rawA_cap : p==d_rawB ? d_rawB_cap : 0;
    }
    void raw_reserve(size_t a,size_t b) {
        // Each operand/frontier keeps its own capacity; a larger A never implies a larger B.
        if(a>d_rawA_cap) {
            if(d_rawA)CK(cudaFree(d_rawA));d_rawA=nullptr;d_rawA_cap=0;
            CK(cudaMalloc(&d_rawA,a*8));d_rawA_cap=a;
        }
        if(b>d_rawB_cap) {
            if(d_rawB)CK(cudaFree(d_rawB));d_rawB=nullptr;d_rawB_cap=0;
            CK(cudaMalloc(&d_rawB,b*8));d_rawB_cap=b;
        }
    }
    double t_h2d_raw = 0.0, t_packdev = 0.0;
    unsigned long long raw_words = 0, pack_launches = 0;
    /* All S4 device-packed calls, including F-tree/inverse before the main-loop timers. */
    unsigned long long input_direct = 0, input_copied = 0, input_d2d_bytes = 0;
    unsigned long long input_avoided_bytes = 0, packed_peak_bytes = 0, temp_peak_bytes = 0;
    double t_input_copy_host = 0.0;
    ~S4Ctx()
    {
        if (d_out) cudaFree(d_out);
        if (d_rawA) cudaFree(d_rawA);
        if (d_rawB) cudaFree(d_rawB);
        if (d_packA) cudaFree(d_packA);
        if (d_packB) cudaFree(d_packB);
    }
};

struct PolyLayer {
    mpz_t N;
    size_t W = 0;               /* words per coefficient */
    size_t S = 0;               /* bits(N) */
    int device = 1;
    /* slice S3: the persistent multiply arena (buffers + fusion plan + per-pass twiddle
       tables built once per SHAPE instead of once per call).  nullptr = the original
       per-call path, unchanged. */
    NttArena *arena = nullptr;
    /* slice S4: the batched + device-reduced multiply.  nullptr = the per-call path. */
    S4Ctx *s4 = nullptr;
    /* slice S3: the batched structure's multiplication accounting.  `cat` is the CURRENT
       category: every multiply that does not name one explicitly (the ones inside
       cp_divmod/cp_inv_series, whose caller knows what phase it is in) is charged to it.
       -1 = do not account (the S1/S2 paths, which keep their own counters). */
    MulCost cost;
    int cat = -1;
    /* statistics */
    unsigned long long ntt_calls = 0, muls = 0, slot_checks = 0;
    /* slice S4: batched multiplies are ONE launch for `nbatch` poly multiplies, so the launch
       count is reported separately from the multiply count (which keeps the S3 numbers
       comparable) */
    unsigned long long ntt_launches = 0;
    unsigned long long max_ntt_words = 0, max_ntt_coeffs = 0, max_slot_bits = 0;
    /* the BINDING exactness shape: the call whose L*(2^bpw-1)^2 is the largest, i.e. the one
       that comes closest to p -- that is the shape the bound has to be re-proved for */
    unsigned long long bind_P = 0, bind_L = 0, bind_slot_bits = 0, bind_slot_words = 0;
    int bind_bpw = 0;
    double bind_bound_bits = 0.0;
    double ntt_seconds = 0.0;
    /* slice S3: where the time inside the multiply goes.  t_fwd/t_inv/t_slot are the NTT
       implementation's own measurements (two forward transforms, the inverse pass with the
       pointwise product and the 1/N scale, then carry + slot assembly + readback); whatever
       is left of ntt_seconds is the host side of the call (packing into bpw digits, the
       memcpys, the exact-coefficient extraction and the two-extraction cross-check). */
    double t_fwd = 0.0, t_inv = 0.0, t_slot = 0.0;
    /* slice S4: the per-call fixed cost, split (see NttMulStats).  host_side = everything in
       ntt_seconds that is not t_fwd/t_inv/t_slot; these four are the measured parts of it. */
    double t_setup = 0.0, t_pack = 0.0, t_maxc = 0.0, t_h2d = 0.0, t_d2h = 0.0, t_ext = 0.0,
           t_xchk = 0.0;
    /* OBJECTIVE 4 (section 32): the copy of the reduced PRODUCT back to the host that
       poly_mul_batch_modN does once per chunk -- the volume and the time of it, measured here
       because no timer in the probe covers it (the probe's host-side timers belong to the host
       implementation, which the batched path does not use). */
    double t_d2h_coeff = 0.0;
    unsigned long long d2h_coeff_words = 0;
    /* OBJECTIVE 4: the three phases of the BATCHED entry point that its timers never covered:
       the host-side packing of the operands, the max-coefficient scan over the packed batch, and
       the upload of the packed operands (section 32). */
    double t_hpack = 0.0, t_scan = 0.0, t_h2d_batch = 0.0;
    /* the carry-convergence assert's own pass (section 34) */
    double t_check = 0.0;
    /* the three parts of t_check (section 41): the reset launch, the kernel's own GPU time and
       the blocking 16-byte readback that drains the pipeline */
    double t_check_reset = 0.0, t_check_kernel = 0.0, t_check_d2h = 0.0;
    double t_plan = 0.0, t_opcopy = 0.0;
    /* the slot assembly + its D2H, skipped when a reduction hook is installed (section 35) */
    double t_hout = 0.0;

    PolyLayer() { mpz_init(N); }
    ~PolyLayer() { mpz_clear(N); }
    PolyLayer(const PolyLayer &) = delete;
    PolyLayer &operator=(const PolyLayer &) = delete;
};

static bool poly_is_one(const unsigned long long *p, size_t W)
{
    if (p[0] != 1) return false;
    for (size_t i = 1; i < W; ++i)
        if (p[i] != 0) return false;
    return true;
}

/* c = a*b mod N for degrees da and db.  The operands are ZERO-PADDED to P = max(ma,mb)
   coefficients (ma = da+1) and multiplied by the shared NTT multiply, which returns EXACT
   integer coefficients; each is then reduced mod N here.  Padding is what lets an unbalanced
   product tree use one balanced multiply: the exactness bound L = P*slot_words is computed
   from the PADDED length and only grows under padding, so the assertion inside
   ntt_poly_mul_host stays valid (it never assumes the coefficients are nonzero). */
static std::vector<unsigned long long> poly_mul_modN(PolyLayer &L,
                                                     const std::vector<unsigned long long> &a,
                                                     size_t da,
                                                     const std::vector<unsigned long long> &b,
                                                     size_t db,
                                                     int cat = -1)
{
    const size_t W = L.W;
    const size_t ma = da + 1, mb = db + 1;
    const unsigned long long P = (unsigned long long)((ma > mb) ? ma : mb);
    std::vector<unsigned long long> wa((size_t)P * W, 0ull), wb((size_t)P * W, 0ull);
    std::copy(a.begin(), a.begin() + (long)(ma * W), wa.begin());
    std::copy(b.begin(), b.begin() + (long)(mb * W), wb.begin());

    std::vector<unsigned long long> slots;
    std::vector<std::vector<unsigned long long>> exact;
    NttMulStats st{};
    const double t0 = now_s();
    const int rc = ntt_poly_mul_host(P, (int)L.S, L.device, /*verbose=*/false, /*dump=*/0,
                                     wa.data(), wb.data(), &slots, &exact, &st, L.arena);
    if (rc != 0) {
        std::fprintf(stderr, "%s: ntt_poly_mul_host failed (rc=%d) at P=%llu S=%d\n",
                     NTT_PROBE_NAME, rc, (unsigned long long)P, (int)L.S);
        std::exit(3);
    }
    L.ntt_seconds += now_s() - t0;
    L.t_fwd += st.t_fwd;
    L.t_inv += st.t_inv;
    L.t_slot += st.t_slot;
    L.t_setup += st.t_setup;
    L.t_pack += st.t_pack;
    L.t_maxc += st.t_maxc;
    L.t_h2d += st.t_h2d;
    L.t_d2h += st.t_d2h;
    L.t_ext += st.t_ext;
    L.t_xchk += st.t_xchk;
    ++L.ntt_calls;
    ++L.muls;
    {
        const int c = (cat >= 0) ? cat : L.cat;       /* explicit, else the layer's current one */
        if (c >= 0) L.cost.add(c, ma, mb);
    }
    L.max_ntt_words = std::max(L.max_ntt_words, st.N);
    L.max_ntt_coeffs = std::max(L.max_ntt_coeffs, P);
    L.max_slot_bits = std::max(L.max_slot_bits, st.slot_bits);
    /* The exactness bound is proven and asserted inside ntt_poly_mul_host for THIS (P, S);
       re-assert the headline inequality here from the values that came back, so the tree's
       own shapes are visibly covered:  L*(2^bpw-1)^2 < p. */
    if (!exact_ok_terms(st.L_terms, st.bpw)) {
        std::fprintf(stderr, "%s: EXACTNESS VIOLATED for the tree's shape: L=%llu bpw=%d\n",
                     NTT_PROBE_NAME, (unsigned long long)st.L_terms, st.bpw);
        std::exit(3);
    }
    {
        const double bb = coeff_bound_bits_terms(st.L_terms, st.bpw);
        if (bb > L.bind_bound_bits) {
            L.bind_bound_bits = bb;
            L.bind_P = P;
            L.bind_L = st.L_terms;
            L.bind_slot_bits = st.slot_bits;
            L.bind_slot_words = st.slot_words;
            L.bind_bpw = st.bpw;
        }
    }
    if (exact.size() != st.out_slots) {
        std::fprintf(stderr, "%s: expected %llu exact coefficients, got %llu\n", NTT_PROBE_NAME,
                     (unsigned long long)st.out_slots, (unsigned long long)exact.size());
        std::exit(3);
    }
    const size_t nc = ma + mb - 1;
    std::vector<unsigned long long> out(nc * W, 0ull);
    mpz_t t;
    mpz_init(t);
    for (size_t k = 0; k < nc; ++k) {
        mpz_import(t, exact[k].size(), -1, 8, 0, 0, exact[k].data());
        mpz_mod(t, t, L.N);
        size_t cnt = 0;
        mpz_export(&out[k * W], &cnt, -1, 8, 0, 0, t);
        if (cnt > W) {
            std::fprintf(stderr, "%s: FATAL: reduced coefficient needs %llu words > W=%llu\n",
                         NTT_PROBE_NAME, (unsigned long long)cnt, (unsigned long long)W);
            std::exit(3);
        }
        ++L.slot_checks;
    }
    mpz_clear(t);
    return out;
}

struct FTreeStats {
    size_t leaves = 0, padded = 0;
    unsigned long long muls = 0;
    unsigned long long node_peak_bytes = 0, node_retained_bytes = 0, node_released_bytes = 0;
    unsigned long long nodes_released = 0, passthrough_moves = 0;
    double t_release = 0.0;
};

/* the stage-2 parameters both tails need, independent of where they came from (a CPU dump for
   the frozen vector, or the curve setup for a real shape whose dump cannot exist) */
struct Stage2Params {
    unsigned long long D = 0, B1 = 0, B2 = 0;
    std::vector<unsigned long long> baby_j;
};

/* ===================================================================================== *
 *  SLICE S4 (A) -- THE DEVICE mod-N REDUCTION OF THE EXACT PRODUCT COEFFICIENTS
 *
 *  WHERE THE TIME WENT (S3, frozen vector, 5334 multiplies of 1.07 s): 97% inside the NTT
 *  multiply, and inside that, `host_side` = 0.535 s = 100 us per call against 1.4 ms of
 *  butterfly arithmetic for the WHOLE run.  Part of that host side is GMP work that scales
 *  with the shape and is not launch overhead at all: the exact coefficients come back as
 *  2S+log2(P)-bit integers (10539 bits at the real shape) and are reduced mod N one mpz_mod
 *  at a time, 2P-1 of them per multiply, 3 multiplies per outer loop.  At the real shape
 *  that is 1.8e5 reductions of a 10539-bit value each fold, and it also forces the WHOLE
 *  canonical digit array (N words = 1 GB at P=92160) back to the host on every call.
 *
 *  THE REDUCTION ITSELF is Montgomery REDC, done on the digits the carry stage leaves on the
 *  device -- no host round trip, no mpz at all in the hot path:
 *
 *    (1) the slot_words bpw-bit digits of coefficient k of slice s ARE the exact integer
 *        C = sum_j d[j] 2^(bpw*j), because the packing stride is word-aligned (the same
 *        property the slot assembler relies on);
 *    (2) C is converted to base-2^64 limbs and reduced by L Montgomery elimination steps:
 *        each step makes the low limb zero and shifts down by 2^64, so after L steps the
 *        array holds exactly (C + q*N)/2^(64 L) = C*2^(-64 L) mod N, provided
 *            C < N * 2^(64 L)                                                  [S4-1]
 *        which is ASSERTED ON THE HOST with exact integers for the ACTUAL N and shape (never
 *        inherited).  L is chosen minimal for [S4-1], so the work is L*nw limb multiply-
 *        accumulates instead of the ~nw*nlimb of a Horner loop -- the difference between
 *        6.9e3 and 1.2e6 limb ops per coefficient at S=5261;
 *    (3) one Montgomery multiply by the host-precomputed Y = 2^(64 (L+nw)) mod N undoes the
 *        2^(-64 L) and lands back in the PLAIN domain (Y multiplies by 2^(64L) once the
 *        multiply's own R^-1 is accounted for), so the tree's coefficients stay in the same
 *        representation as before.
 *
 *  WHAT IS ASSERTED, from the values that are actually there (section 14.10's lesson):
 *    * the probe re-derives L*(2^bpw-1)^2 < p for the shape and asserts every operand digit
 *      < 2^bpw on the packed values;
 *    * s4_slot_canonical_kernel checks, over EVERY coefficient of EVERY slice, that the digits
 *      above slot_bits in the slot window are zero -- together with "every digit < 2^bpw"
 *      (asserted by the probe's carry-residual check) that is exactly C < 2^slot_bits;
 *    * [S4-1] is then proved on the host from that bound and the actual N.
 *  If the bound were violated the reduction would silently return a wrong residue, which is
 *  why it is a proof plus a value check rather than an assumption.
 *
 *  s4_reduce_selftest() additionally runs the kernel against GMP on random AND adversarial
 *  digit patterns of the actual shape (all-ones, top-bit-only, zero), and every batched
 *  multiply that uses the reduction also verifies a sample of its own coefficients against
 *  GMP -- all of them when the batch is small enough to make a full check cheap.
 * ===================================================================================== */

/* -------- slice S4: the device mod-N reduction of exact product coefficients ---------- */
/* how many coefficients per batch are compared against GMP in-run: a full check when the
   batch is at most this many, otherwise this many in 16 contiguous runs.  0 = no in-run
   check (the startup selftest and the acceptance gates still run).  NTT_S4_SAMPLE=n. */
static long long g_s4_sample_limit = 96;
/* check one call in `g_s4_check_every` (the first call of every shape is always checked), so
   the in-run oracle costs a bounded fraction of the run instead of dominating it */
static unsigned long long g_s4_check_every = 8;
/* diagnostic: print a per-level hash of the descent's polynomials from BOTH implementations
   (NTT_S4_DESCENT_TRACE=1) */
static bool g_s4_descent_trace = false;
/* progress lines: one per G-tree batch and one per descent LEVEL.  On by default, because at
   the real shape these phases run for tens of minutes and used to print nothing at all (a hung
   run and a working run were indistinguishable); NTT_NO_PROGRESS=1 turns them off. */
static bool g_s4_batched_progress = true;

/* THE DEFERRED CARRY CHECK, COUNTED (section 29): how many chunk round-trips skipped the
   probe's per-chunk residual readback, and how many single-readback finishes replaced them.
   Printed so tools/test can assert the deferral actually happened. */
static unsigned long long g_defer_chunks = 0, g_defer_finishes = 0, g_defer_slices = 0;
static unsigned long long g_defer_checked_chunks = 0, g_defer_max_group = 0;
static double g_carry_group_readback = 0.0, g_carry_chunk_readback = 0.0;

/* ---- PINNED STAGING FOR THE CHUNK ROUND-TRIPS (section 30) --------------------------------
   Section 28 measured what the per-chunk cost really is: a PAGEABLE transfer cannot be overlapped,
   because the driver has to wait for the stream before it can stage the buffer -- so each chunk
   paid a full pipeline drain twice (upload, readback) on top of the one the carry check used to
   add.  Section 29 removed the carry check's drain; this removes the other two by staging through
   PAGE-LOCKED host memory: the host memcpy into the pinned buffer costs no device time at all, and
   the following cudaMemcpyAsync needs no implicit sync, so the kernels of the next chunk stay
   queued instead of waiting for the host.  The output side is double buffered so the host consumes
   chunk k-1 while chunk k is still copying and computing.

   Host RAM only -- the device footprint is unchanged (the arena's 5779 MB is the binding
   constraint, section 27.3), and the buffers are small: at the production shape one chunk's raw
   coefficients are 0.43 MB per operand and its reduced output 1.66 MB.  A failed pinning falls back
   to the old blocking path rather than aborting. */
/* Two upload slots, each with its OWN A/B capacities. Removing a carry drain must not let
   the CPU overwrite pinned input while a preceding H2D still reads it. */
static unsigned long long *g_pin_raw[2][2] = {};
static size_t g_pin_raw_cap[2][2] = {};
static cudaEvent_t g_pin_raw_ev[2] = {};
static bool g_pin_raw_pending[2] = {};
static unsigned long long *g_pin_out[2] = {nullptr, nullptr};
static size_t g_pin_out_cap[2] = {0, 0};
static cudaEvent_t g_pin_ev[2] = {nullptr, nullptr};

/* THE A/B KNOBS OF SECTIONS 29 AND 30.  Both techniques are timing-sensitive and the card's SM
   clock swings between 1000 and 1772 MHz from run to run (read from the GPU-Z sensor log), which is
   8-10% of a wall clock -- the same order as the effects being measured.  Cross-BUILD comparisons
   are therefore confounded by clock drift, so each technique has a runtime switch and the honest
   A/B is four back-to-back runs of ONE binary.  Defaults: both ON. */
static bool opt_off(const char *name)
{
    const char *e = std::getenv(name);
    return e && *e && std::atoi(e) == 0;
}
static const bool g_s4_defer_carry = !opt_off("NTT_S4_DEFER_CARRY");
static const bool g_s4_async = !opt_off("NTT_S4_ASYNC");
/* Default direct-to-scratch: production removes 2 GiB of temporary inputs and 579 GiB of
   D2D. The ~2% ABBA time gain is smaller than control drift, so speed remains unproven. */
static const bool g_s4_pack_direct = !opt_off("NTT_S4_PACK_DIRECT");
/* Borrow already padded flat operands; 0 retains both host materializations for A/B. */
static const bool g_s4_flat_direct = !opt_off("NTT_S4_FLAT_DIRECT");
/* Opt-in until the full numerical/production gate validates the compact contract. */
static const bool g_s4_output_window = [] {
    const char *e=std::getenv("NTT_S4_OUTPUT_WINDOW");
    return e && *e && std::atoi(e)!=0;
}();
struct OutputWindowStats {
    unsigned long long calls=0, source_coeffs=0, reduced_coeffs=0, returned_coeffs=0;
    unsigned long long skipped_coeffs=0, d2h_words=0, device_peak_bytes=0, pinned_peak_bytes=0;
};
static OutputWindowStats g_output_window;
/* Each chunk already reaches OUT through its blocking/pinned readback. Retain the redundant
   whole-call copy only for a same-binary control, never as a correctness fallback. */
static const bool g_s4_final_readback = [] {
    const char *e=std::getenv("NTT_S4_FINAL_READBACK");
    return e && *e && std::atoi(e)!=0;
}();
struct FinalReadbackStats {
    unsigned long long calls=0, copied_words=0, avoided_words=0, host_peak_bytes=0;
    double t_copy=0.0;
};
static FinalReadbackStats g_final_readback;
/* Output snapshots/copies are queued before the next reduction on the SAME default stream.
   Legacy whole-call readback requires the original per-slice device layout. */
static const bool g_s4_chunk_output = !opt_off("NTT_S4_CHUNK_OUTPUT");
struct ChunkOutputStats {
    unsigned long long calls=0, reused_calls=0, whole_calls=0, multi_chunk_reused=0;
    unsigned long long reused_chunks=0, legacy_calls=0, grows=0;
    unsigned long long request_peak_bytes=0, whole_peak_bytes=0, retained_peak_bytes=0;
};
static ChunkOutputStats g_chunk_output;
/* Opt-in algorithm candidate: ordinary-coefficient scaled remainder descent. */
static constexpr bool g_scaled_descent = true;
static const bool g_scaled_check = [] {
    const char *v=std::getenv("NTT_SCALED_CHECK");
    return v && std::atoi(v)!=0;
}();
/* Main-loop G trees only need the root; 0 retains the full heap for same-binary A/B. */
static const bool g_groot_device = [] {
    const char *v=std::getenv("NTT_GROOT_DEVICE");return v && std::atoi(v)!=0;
}();
static const bool g_groot_device_check = [] {
    const char *v=std::getenv("NTT_GROOT_DEVICE_CHECK");return v && std::atoi(v)!=0;
}();
struct GDeviceStats {
    unsigned long long trees=0, fallbacks=0, levels=0, groups=0, pairs=0, copies=0;
    unsigned long long leaf_words=0, root_words=0, resident_words=0, trace_words=0;
    unsigned long long metadata_words=0, metadata_peak_bytes=0, raw_peak_bytes=0;
    unsigned long long logical_frontier_peak_bytes=0, host_staging_peak_bytes=0;
    unsigned long long checked_nodes=0, checked_words=0;
};
static GDeviceStats g_gdevice;
// Existing pinned output staging is borrowed only after its event, never grown for leaves.
static const bool g_groot_leaf_staging = !opt_off("NTT_GROOT_LEAF_STAGING");
static const bool g_groot_compact_raw = !opt_off("NTT_GROOT_COMPACT_RAW");
static const size_t g_groot_leaf_chunk = [] {
    const char *e=std::getenv("NTT_GROOT_LEAF_CHUNK");return e?(size_t)std::strtoull(e,nullptr,10):0;
}();
struct GMemoryStats {
    unsigned long long pinned_trees=0,pinned_slices=0,pinned_words=0,pageable_words=0;
    unsigned long long leaf_fallbacks=0,legacy_trees=0,pinned_borrow_peak_bytes=0;
    unsigned long long rawA_peak_bytes=0,rawB_peak_bytes=0;
};
static GMemoryStats g_gmemory;
static const bool g_s4_groot_only = !opt_off("NTT_S4_GROOT_ONLY");
struct GRootStats {
    unsigned long long builds = 0, nodes_released = 0, passthrough_moves = 0;
    unsigned long long peak_node_bytes = 0, peak_retained_bytes = 0, released_bytes = 0;
    unsigned long long input_released_bytes = 0, root_words = 0;
    unsigned long long root_hash = 1469598103934665603ull;
    bool root_hash_complete=true;
    double t_release = 0.0;
};
static GRootStats g_groot;
/* Same-binary candidate: accumulate identical interior chunks, check BEFORE the tail reset.
   Opt-in until production ABBA establishes a gain. */
static const bool g_s4_carry_batch = [] {
    const char *e = std::getenv("NTT_S4_CARRY_BATCH");
    return e && *e && std::atoi(e) != 0;
}();
/* Gate-only chunk cap and counter fault; production explicitly disables both. */
static const unsigned long long g_s4_chunk_max = [] {
    const char *e = std::getenv("NTT_S4_CHUNK_MAX");
    return e && *e ? std::strtoull(e, nullptr, 10) : 0ull;
}();
static const bool g_s4_carry_test_bad = [] {
    const char *e = std::getenv("NTT_S4_CARRY_TEST_BAD");
    return e && *e && std::atoi(e) != 0;
}();
static bool g_s4_carry_injected = false;
static const bool g_s4_carry_trace = [] {
    const char *e = std::getenv("NTT_S4_CARRY_TRACE");
    return e && *e && std::atoi(e) != 0;
}();
static unsigned long long g_carry_output_hash = 1469598103934665603ull, g_carry_output_words = 0;
static unsigned long long g_pin_raw_waits = 0;
static double g_pin_raw_wait = 0.0;
/* counted so tools/test can assert the asynchronous path was TAKEN (section 30): the timing
   difference between the blocking and the pinned path is small enough that a run which silently
   fell back to blocking would look like a normal result */
static unsigned long long g_pin_raw_used = 0, g_pin_out_used = 0, g_pin_fallbacks = 0;

/* ONE slot, ONE capacity.  Sharing a capacity between the two slots was a real bug: when slot 0
   grew it raised the shared cap, so slot 1 looked big enough while still pointing at the smaller
   buffer, and the upload died with "CUDA error invalid argument" at the second tree level
   (measured, P=5). */
static unsigned long long *pin_words(unsigned long long **slot, size_t *cap, size_t words)
{
    if (*slot && *cap >= words) return *slot;
    if (*slot) { (void)cudaFreeHost(*slot); *slot = nullptr; *cap = 0; }
    void *q = nullptr;
    if (cudaHostAlloc(&q, words * sizeof(unsigned long long), cudaHostAllocDefault) != cudaSuccess) {
        (void)cudaGetLastError();       /* no pinned memory: the caller keeps the blocking path */
        return nullptr;
    }
    *slot = (unsigned long long *)q;
    *cap = words;
    return *slot;
}

/* per-chunk device-buffer budget for a batched multiply (MB): see poly_mul_batch_modN.
   THE ONE-SHAPE LADDER (section 27) says bigger is faster: at D=1231230/B2=1e11, P=115200 the whole
   run is 107.88 s (16 MB) / 91.27 s (32 MB) / 83.68 s (64 MB) with identical factor sets, and the
   two chunk sizes move NO transfer volume at all (D2H 13.67 GB and H2D 14.29 GB in every run; only
   pack_launches changes, 37006 / 20256 / 11130) -- the win is the per-chunk fixed cost (~2 ms per
   chunk: removing 8375 chunks bought 19 s), not bandwidth.  96 MB and above die outright with a real
   "CUDA error out of memory" even though the arena's cap check passed, because the cap comes from
   the free memory at startup while the engine also holds its own pools.
   THE OLD COPIED-INPUT PRODUCTION PATH REVERSES THE RESULT:
   at D=1231230/B2=1.94e12 the 64 MB budget reports "device free=0 MB of 8188 MB" at descent start
   and the host-side naming ladder goes from t_ladder=1.33 s to 142.00 s -- the phase after the
   descent is starved of device memory, and the run takes 373.98 s instead of 253.53 s (measured, both
   with the current binary).  A per-chunk budget is therefore NOT a free lever: it trades tree time
   for whatever needs the device later.  The arena allocations now degrade instead of aborting (see
   NttArena::try_malloc), so a too-large value costs time rather than the run.
   AFTER DIRECT PACK (section 37, 2026-10-03), the SAME production shape/binary ABBA gives
   213.225 -> 208.075 s at 32/64 MB, pack launches 44942 -> 24462, arena 5779 -> 6215.6 MiB,
   no overflow and naming ladder 1.429 -> 1.453 s. Keep 64 ONLY for direct device packing;
   copied inputs and host packing retain 32. This is a scratch budget, NOT a total VRAM cap.
   The modest wall-time gain still has run-to-run noise; GPU-Z busy time is unchanged. */
static const unsigned long long g_s4_batch_budget_mb = 64;

struct S4Reduce;
static void s4_oracle_release(S4Reduce &R);

struct S4Reduce {
    /* the modulus (shared by every shape of the run) */
    int nw = 0;                          /* words of N */
    int mersenne_bits = 0;               /* verified N=2^S-1, owned by this modulus */
    unsigned long long ninv = 0;         /* -N^-1 mod 2^64 */
    size_t w = 0;                        /* words per coefficient in the tree (W = ceil(S/64)) */
    mpz_t N{};                           /* the actual modulus, for the GMP oracle */
    std::vector<unsigned long long> hn;
    unsigned long long *dn = nullptr;
    unsigned long long *dbad = nullptr;      /* the reduction's slot-canonical counter (device) */
    unsigned long long dbad_host = 0;
    /* the per-SHAPE parameters: L, the limb count and the domain-correction constant all
       depend on slot_bits, i.e. on P, so one entry per (P) the run multiplies at */
    struct Shape {
        unsigned long long slot_bits = 0, slot_stride = 0, slot_words = 0;
        int bpw = 0, L = 0, nlimb = 0;
        double bound_bits = 0.0;
        unsigned long long P = 0;
        std::vector<unsigned long long> hy;
        unsigned long long *dy = nullptr;
        unsigned long long calls = 0, coeffs = 0, canon_bad = 0;
        /* 1 once the window canonicality of this shape has been checked (section 49) */
        unsigned long long canon_checked = 0;
        unsigned long long checked = 0, check_bad = 0, check_first = 0, full_checks = 0;
        double t_hookd2h = 0.0, t_hooksample = 0.0;
        unsigned long long selftest_cases = 0, selftest_bad = 0;
        long long selftest_first = -1;
        double t_reduce = 0.0;
        /* ---- THE HOOK'S OWN TIMING MUST NOT DRAIN THE PIPELINE (docs section 41) ----------
           A `cudaDeviceSynchronize()` used to sit right after the reduction kernel purely so
           that `t_reduce` could be accumulated.  That made `t_reduce` a mixture of kernel time
           and a full pipeline drain, and at the production shape it read 88.229 s out of the
           252.300 s of `ntt_seconds` -- the largest single item on the books.  The marks are
           CUDA events now, resolved on a LATER call once the device has caught up (so the host
           never has to wait for work it just queued), which makes `t_reduce` the kernels' own
           GPU time and `t_reduce_host` what the host actually paid.  The ring is bounded, so a
           host that is more than DTRING calls ahead still blocks -- but on the OLDEST mark,
           once, instead of once per call. */
        static const int DTRING = 8;
        cudaEvent_t dt_ev[DTRING][2] = {};
        bool dt_used[DTRING] = {};
        double t_reduce_host = 0.0;
        unsigned long long dt_blocks = 0;
        ~Shape() { if (dy) cudaFree(dy); dt_destroy(); }
        void dt_destroy()
        {
            for (int i = 0; i < DTRING; ++i)
                if (dt_used[i]) {
                    cudaEventDestroy(dt_ev[i][0]);
                    cudaEventDestroy(dt_ev[i][1]);
                    dt_ev[i][0] = dt_ev[i][1] = nullptr;
                    dt_used[i] = false;
                }
        }
        /* resolve every mark the device has already passed; NEVER blocks */
        void dt_reap()
        {
            float ms = 0.0f;
            for (int i = 0; i < DTRING; ++i) {
                if (!dt_used[i]) continue;
                if (cudaEventQuery(dt_ev[i][1]) != cudaSuccess) continue;
                if (cudaEventElapsedTime(&ms, dt_ev[i][0], dt_ev[i][1]) == cudaSuccess)
                    t_reduce += (double)ms * 1e-3;
                cudaEventDestroy(dt_ev[i][0]);
                cudaEventDestroy(dt_ev[i][1]);
                dt_ev[i][0] = dt_ev[i][1] = nullptr;
                dt_used[i] = false;
            }
        }
        /* take a slot for a new mark: reuse a resolved one, else wait for what is already
           queued.  The wait happens only when the host is a whole ring ahead of the device,
           which is the one place the deferred timing can still cost anything. */
        int dt_slot()
        {
            dt_reap();
            for (int i = 0; i < DTRING; ++i)
                if (!dt_used[i]) return i;
            ++dt_blocks;
            dt_flush();
            return 0;
        }
        /* resolve everything, waiting if necessary -- called once, when the run is over and the
           stream has drained anyway, so it never costs anything */
        void dt_flush()
        {
            for (int i = 0; i < DTRING; ++i) {
                if (!dt_used[i]) continue;
                cudaEventSynchronize(dt_ev[i][1]);
                float ms = 0.0f;
                if (cudaEventElapsedTime(&ms, dt_ev[i][0], dt_ev[i][1]) == cudaSuccess)
                    t_reduce += (double)ms * 1e-3;
                cudaEventDestroy(dt_ev[i][0]);
                cudaEventDestroy(dt_ev[i][1]);
                dt_ev[i][0] = dt_ev[i][1] = nullptr;
                dt_used[i] = false;
            }
        }
        Shape() = default;
        Shape(const Shape &) = delete;
        Shape &operator=(const Shape &) = delete;
    };
    std::vector<Shape *> shapes;
    unsigned long long reduce_calls = 0, coeffs_total = 0;
    /* the asynchronous `dbad` readback (section 43): one copy in flight at a time, compared on
       the following call, and resolved for good by s4_dbad_resolve() */
    unsigned long long *h_dbad = nullptr;
    cudaEvent_t ev_dbad = nullptr;
    bool dbad_inflight = false;
    Shape *dbad_shape = nullptr;
    unsigned long long dbad_expected = 0;

    S4Reduce() { mpz_init(N); }
    ~S4Reduce()
    {
        s4_oracle_release(*this);       /* snapshots reference shapes and N: drain FIRST */
        for (Shape *s : shapes) delete s;
        if (dn) cudaFree(dn);
        if (dbad) cudaFree(dbad);
        if (h_dbad) cudaFreeHost(h_dbad);
        if (ev_dbad) cudaEventDestroy(ev_dbad);
        mpz_clear(N);
    }
    /* the key must be the WHOLE shape, not slot_bits: slot_bits = 2S + ceil(log2 P) is the same
       for P = 5, 6, 7 and 8, while bpw/slot_words/slot_stride differ between them (found the
       hard way: the lookup returned the P=5 entry for the P=8 multiply) */
    Shape *find(unsigned long long slot_bits, unsigned long long slot_words, int bpw) const
    {
        for (Shape *s : shapes)
            if (s->slot_bits == slot_bits && s->slot_words == slot_words && s->bpw == bpw)
                return s;
        return nullptr;
    }
};

/* Modulus-level oracle state; leave Shape's layout alone.  Capture before device scratch is
   reused, compare a completed HOST snapshot while later GPU work runs.  No device pointer is
   retained past the capture call.  Memory and delayed validation are bounded by this ring. */
/* Deferred validation remains opt-in: its first production A/B did not establish a win. */
static const bool g_s4_oracle_async = [] {
    const char *e = std::getenv("NTT_S4_ORACLE_ASYNC");
    return e && *e && std::atoi(e) != 0;
}();
static const bool g_s4_oracle_pack = !opt_off("NTT_S4_ORACLE_PACK");
static const int g_s4_oracle_ring = [] {
    const char *e = std::getenv("NTT_S4_ORACLE_RING");
    return e && *e ? std::max(1, std::min(8, std::atoi(e))) : 4;
}();
struct S4OracleSlot {
    unsigned long long *digits = nullptr, *reduced = nullptr;
    size_t digcap = 0, redcap = 0;
    cudaEvent_t ready = nullptr;
    S4Reduce::Shape *shape = nullptr;
    unsigned long long slice = 0, k0 = 0, count = 0, out_slots = 0;
};
static S4OracleSlot g_oracle_slot[8];
static S4Reduce *g_oracle_owner = nullptr;
static cudaEvent_t g_oracle_block_ev = nullptr;
struct S4OracleStats {
    unsigned long long selected = 0, queued = 0, compared = 0, samples = 0;
    unsigned long long head = 0, tail = 0, ring_waits = 0, fallbacks = 0;
    unsigned long long signature = 1469598103934665603ull, pinned_peak = 0;
    double t_wait = 0.0, t_copy = 0.0, t_gmp = 0.0, t_alloc = 0.0;
    double t_capture = 0.0, t_reap = 0.0, t_drain = 0.0;
    double t_num = 0.0, t_mod = 0.0;
};
static S4OracleStats g_oracle;
static void s4_oracle_reap(S4Reduce &R, bool wait);
static void s4_oracle_drain(S4Reduce &R);
static void s4_gmp_pack_check();

static void s4_oracle_reset(S4Reduce &R)
{
    if (g_oracle_owner && g_oracle_owner != &R) {
        std::fprintf(stderr, "%s: FATAL: overlapping modulus oracle owners\n", NTT_PROBE_NAME);
        std::exit(3);
    }
    g_oracle_owner = &R;
    g_oracle = S4OracleStats{};
    stage2_log::print(stage2_log::debug, "s4_oracle_mode: async=%d ring=%d pack=%d (same samples and GMP predicate)\n",
                (int)g_s4_oracle_async, g_s4_oracle_ring, (int)g_s4_oracle_pack);
}

static void s4_oracle_report()
{
    stage2_log::print(stage2_log::debug, "s4_oracle_stats: async=%d selected=%llu queued=%llu compared=%llu samples=%llu "
                "pending=%llu ring_waits=%llu fallbacks=%llu signature=%016llx pinned_peak=%llu "
                "t_wait=%.6f t_copy_host=%.6f t_gmp=%.6f t_alloc=%.6f "
                "t_capture=%.6f t_reap=%.6f t_drain=%.6f host_total=%.6f "
                "pack=%d t_num=%.6f t_mod=%.6f\n",
                (int)g_s4_oracle_async, g_oracle.selected, g_oracle.queued, g_oracle.compared,
                g_oracle.samples, g_oracle.tail - g_oracle.head, g_oracle.ring_waits,
                g_oracle.fallbacks, g_oracle.signature, g_oracle.pinned_peak,
                g_oracle.t_wait, g_oracle.t_copy, g_oracle.t_gmp, g_oracle.t_alloc,
                g_oracle.t_capture, g_oracle.t_reap, g_oracle.t_drain,
                g_oracle.t_capture + g_oracle.t_reap + g_oracle.t_drain,
                (int)g_s4_oracle_pack, g_oracle.t_num, g_oracle.t_mod);
}

/* Resolve the pending asynchronous `dbad` readback, if there is one (section 43).  NON-BLOCKING
   by default: the point of the whole exercise is that the host must not wait for the work it has
   just queued.  `wait=true` is used only where a violation could otherwise be lost -- before the
   counter is reset for a new shape, and once at the end of the run.  A violation is FATAL, in the
   same terms as the old synchronous check (the counter is monotone, so a late read is complete:
   it cannot miss a violation, it can only report it later). */
static void s4_dbad_resolve(S4Reduce &R, bool wait = false)
{
    if (!R.dbad_inflight) return;
    if (wait) CK(cudaEventSynchronize(R.ev_dbad));
    else if (cudaEventQuery(R.ev_dbad) != cudaSuccess) return;
    R.dbad_inflight = false;
    const unsigned long long hbad = *R.h_dbad;
    S4Reduce::Shape *S = R.dbad_shape;
    if (S && hbad != R.dbad_expected) {
        const unsigned long long added = (hbad > R.dbad_expected) ? (hbad - R.dbad_expected)
                                                                 : hbad;
        std::fprintf(stderr, "%s: FATAL: %llu (of %llu so far) slot windows of P=%llu have "
                             "nonzero digits above slot_bits=%llu -- the reduction bound "
                             "C < 2^slot_bits does NOT apply to these values\n", NTT_PROBE_NAME,
                     added, S->coeffs, S->P, S->slot_bits);
        std::exit(3);
    }
    if (S) S->canon_bad = hbad;
}

/* Modulus-level constants, deliberately outside the per-shape objects.  The normalized
   divisor is broadcast from constant memory; its width is bounded by S2G_DISPATCH. */
static std::vector<unsigned long long> g_div_hns;
static unsigned long long g_div_recip = 0;
static int g_div_shift = 0, g_div_nw = 0;
__host__ __device__ __forceinline__ void s2g_mul64(unsigned long long, unsigned long long,
                                                unsigned long long &, unsigned long long &);
__host__ __device__ __forceinline__ unsigned long long s2g_udiv_2by1(
    unsigned long long, unsigned long long, unsigned long long, unsigned long long);

/* Plain long division of an UNNORMALIZED coefficient by N.  ns = N << shift is normalized,
   and t has two spare high words.  No Montgomery domain enters this path: the result is C mod N.
   The exact top-two-word quotient can overestimate the full quotient by at most two when the
   divisor's high bit is set.  Subtract the full product, then add the divisor back on underflow.
   Return the repair count (used by the independent GMP fixtures), or -1 on an invariant failure.
   The host and device execute this same helper, including the 64-bit multiplication primitive. */
template <int NW>
__host__ __device__ __forceinline__ int s4_div_rem(unsigned long long *t, int limbs,
    const unsigned long long *ns, int nw, int shift, unsigned long long recip,
    unsigned long long *out)
{
    for (int i = 0; i < nw; ++i) out[i] = 0;
    if (shift != 0) {
        unsigned long long carry = 0;
        for (int i = 0; i < limbs; ++i) {
            const unsigned long long v = t[i];
            t[i] = (v << shift) | carry;
            carry = v >> (64 - shift);
        }
        t[limbs++] = carry;
    }
    while (limbs > 0 && t[limbs - 1] == 0) --limbs;
    t[limbs] = 0;                        /* high sentinel, including the shift == 0 case */
    int repairs = 0;
    const unsigned long long top = ns[nw - 1];
    for (int j = limbs - nw; j >= 0; --j) {
        const unsigned long long u1 = t[j + nw], u0 = t[j + nw - 1];
        if (u1 > top) return -1;
        const unsigned long long q = (u1 == top) ? ~0ull
                                                  : s2g_udiv_2by1(u1, u0, top, recip);
        unsigned long long carry = 0;
        for (int i = 0; i < nw; ++i) {
            unsigned long long lo, hi;
            s2g_mul64(q, ns[i], lo, hi);
            const unsigned long long sub = lo + carry;
            hi += (sub < lo) ? 1ull : 0ull;
            const unsigned long long v = t[j + i];
            t[j + i] = v - sub;
            carry = hi + ((v < sub) ? 1ull : 0ull);
        }
        bool under = t[j + nw] < carry;
        t[j + nw] -= carry;
        for (int correction = 0; under && correction < 2; ++correction) {
            unsigned long long c = 0;
            for (int i = 0; i < nw; ++i) {
                const unsigned long long v = t[j + i], sum = v + ns[i];
                const unsigned long long next = sum + c;
                c = ((sum < v) || (next < sum)) ? 1ull : 0ull;
                t[j + i] = next;
            }
            const unsigned long long v = t[j + nw];
            t[j + nw] = v + c;
            under = (t[j + nw] >= v);    /* wrapping the negative high word ends the borrow */
            ++repairs;
        }
        if (under) return -1;
    }
    for (int i = 0; i < nw; ++i)
        out[i] = (shift == 0) ? t[i]
            : ((t[i] >> shift) | (i + 1 < nw ? (t[i + 1] << (64 - shift)) : 0ull));
    return repairs;
}

/* Plain remainder modulo an EXACT Mersenne modulus. Each fold replaces v by
   low_S(v)+high_S(v), congruent because 2^S == 1. Read high words before overwriting
   low words: this is safe even at S<64 (q=0). No Montgomery scaling enters here.
   Input has at least one spare word; trailing words are ignored beyond live. */
template <int NW>
__host__ __device__ __forceinline__ int s4_mersenne_rem(unsigned long long *t, int live,
    int nw, int bits, unsigned long long *out)
{
    const int q=bits/64,b=bits%64;
    const unsigned long long mask=b ? ((1ull<<b)-1) : ~0ull;
    while(live>0 && t[live-1]==0)--live;
    int folds=0;
    while(live>nw || (b && live==nw && (t[nw-1]>>b)!=0)) {
        const int len=(live-q)>nw ? live-q : nw;
        unsigned long long carry=0;
        for(int i=0;i<len;++i) {
            const int src=i+q;
            const unsigned long long h0=src<live ? t[src] : 0ull;
            const unsigned long long h1=(b && src+1<live) ? t[src+1] : 0ull;
            const unsigned long long hi=b ? ((h0>>b)|(h1<<(64-b))) : h0;
            const unsigned long long lo=i<nw ? (i==nw-1 ? (t[i]&mask) : t[i]) : 0ull;
            const unsigned long long sum=lo+hi,next=sum+carry;
            carry=((sum<lo)||(next<sum)) ? 1ull : 0ull;
            t[i]=next;
        }
        live=len;
        if(carry)t[live++]=carry;
        while(live>0 && t[live-1]==0)--live;
        ++folds;
    }
    bool equal=true;
    for(int i=0;i<nw;++i) {
        out[i]=i<live ? t[i] : 0ull;
        if(out[i]!=(i==nw-1 ? mask : ~0ull))equal=false;
    }
    if(equal)for(int i=0;i<nw;++i)out[i]=0; // N itself is zero, canonical [0,N)
    return folds;
}

static const bool g_s4_mersenne=[] {
    const char *e=std::getenv("NTT_S4_MERSENNE");return e && std::atoi(e)!=0;
}();

/* Gate-only GPU primitive, independently checked against ordinary GMP remainders. */
template <int NW>
__global__ void s4_mersenne_fixture_kernel(const unsigned long long *input,int stride,
    const int *lengths,int cases,int nw,int bits,unsigned long long *out,int *counts)
{
    const int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=cases)return;
    unsigned long long t[2*NW+4]={},u[NW];
    for(int i=0;i<lengths[k];++i)t[i]=input[(size_t)k*stride+i];
    counts[k]=s4_mersenne_rem<NW>(t,lengths[k],nw,bits,u);
    for(int i=0;i<nw;++i)out[(size_t)k*nw+i]=u[i];
}

static void s4_mersenne_check()
{
    mpz_t N,v,want,got;mpz_inits(N,v,want,got,nullptr);
    unsigned long long cases=0,words=0,bad=0,folds=0,seed=0x493caf523ull;
    for(int bits : {2,3,31,63,64,65,127,128,129,255,256,4423,5261,8191,8192,8193,16381,16383,16384}) {
        const int nw=(bits+63)/64,stride=2*nw+4,ncase=48;
        mpz_set_ui(N,1);mpz_mul_2exp(N,N,bits);mpz_sub_ui(N,N,1);
        std::vector<unsigned long long> input(ncase*stride),expected(ncase*nw),gpu(expected.size());
        std::vector<int> lengths(ncase),counts(ncase);
        for(int k=0;k<ncase;++k) {
            auto *t=input.data()+(size_t)k*stride;
            if(k==0)mpz_set_ui(v,0);
            else if(k<4){mpz_set(v,N);if(k==1)mpz_sub_ui(v,v,1);if(k==3)mpz_add_ui(v,v,1);}
            else if(k<7){mpz_mul(v,N,N);if(k==4)mpz_sub_ui(v,v,1);if(k==6)mpz_add_ui(v,v,1);}
            else if(k<10){mpz_set_ui(v,1);mpz_mul_2exp(v,v,2*bits+20);if(k==7)mpz_sub_ui(v,v,1);if(k==9)mpz_add_ui(v,v,1);}
            else {
                for(int i=0;i<stride-1;++i){seed=seed*6364136223846793005ull+1;t[i]=seed;}
                if(k==10)for(int i=0;i<stride-1;++i)t[i]=~0ull;
                mpz_import(v,stride-1,-1,8,0,0,t);
            }
            std::fill(t,t+stride,0ull);size_t len=0;mpz_export(t,&len,-1,8,0,0,v);lengths[k]=(int)len;
            mpz_mod(want,v,N);mpz_export(expected.data()+(size_t)k*nw,nullptr,-1,8,0,0,want);
            unsigned long long scratch[2*ecm_stage2::max_words+4]={},u[ecm_stage2::max_words];std::copy(t,t+stride,scratch);
            const int f=nw<=128 ? s4_mersenne_rem<128>(scratch,(int)len,nw,bits,u)
                : s4_mersenne_rem<256>(scratch,(int)len,nw,bits,u);
            mpz_import(got,nw,-1,8,0,0,u);if(mpz_cmp(want,got))++bad;
            folds+=f;
        }
        unsigned long long *di=nullptr,*doo=nullptr;int *dl=nullptr,*dc=nullptr;
        CK(cudaMalloc(&di,input.size()*8));CK(cudaMalloc(&doo,gpu.size()*8));
        CK(cudaMalloc(&dl,ncase*sizeof(int)));CK(cudaMalloc(&dc,ncase*sizeof(int)));
        CK(cudaMemcpy(di,input.data(),input.size()*8,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dl,lengths.data(),ncase*sizeof(int),cudaMemcpyHostToDevice));
        if(nw<=128)s4_mersenne_fixture_kernel<128><<<1,64>>>(di,stride,dl,ncase,nw,bits,doo,dc);
        else s4_mersenne_fixture_kernel<256><<<1,64>>>(di,stride,dl,ncase,nw,bits,doo,dc);
        CK(cudaGetLastError());
        CK(cudaMemcpy(gpu.data(),doo,gpu.size()*8,cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(counts.data(),dc,ncase*sizeof(int),cudaMemcpyDeviceToHost));
        CK(cudaFree(di));CK(cudaFree(doo));CK(cudaFree(dl));CK(cudaFree(dc));
        const char *fault=std::getenv("NTT_S4_MERSENNE_TEST_BAD");if(fault && std::atoi(fault))gpu[0]^=1;
        for(size_t i=0;i<gpu.size();++i)if(gpu[i]!=expected[i])++bad;
        cases+=ncase;words+=gpu.size();
    }
    mpz_clears(N,v,want,got,nullptr);
    stage2_log::print(stage2_log::debug, "s4_mersenne_check: cases=%llu words=%llu folds=%llu bad=%llu (CPU/GPU vs GMP, S=2..16384)\n",cases,words,folds,bad);
    if(bad){std::fprintf(stderr,"FATAL: Mersenne remainder GMP mismatch\n");std::exit(3);}
}

/* base-2^64 limbs of the slot window -> mod N, on the device.  One thread per coefficient. */
template <int NW, bool MERSENNE=false>
__global__ void s4_reduce_kernel(const unsigned long long *digits, unsigned long long n,
                                 int bpw, unsigned long long slot_words,
                                 unsigned long long out_slots, unsigned long long nbatch,
                                 const unsigned long long *dn, unsigned long long ninv,
                                 int nw, int L, const unsigned long long *dy,
                                 unsigned long long w, unsigned long long *out,
                                 unsigned long long slot_bits, unsigned long long *bad,
                                 unsigned long long *s4_dbg, int tail_mont,
                                 int div_shift, unsigned long long div_recip,
                                 unsigned long long first, unsigned long long count, int mersenne_bits)
{
    const unsigned long long total = out_slots * nbatch;
    const unsigned long long gid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (gid >= total) return;
    const unsigned long long s = gid / out_slots;
    const unsigned long long k = gid - s * out_slots;
    const unsigned long long *d = digits + s * n + k * slot_words;
    /* SAME source-slot bound assertion, even for omitted coefficients. No NTT truncation.
       Separate the source index from compact output slice*count+(k-first). */
    const unsigned long long top_bits=slot_bits-(slot_words-1)*(unsigned long long)bpw;
    if(top_bits<64 && bad && (d[slot_words-1]>>top_bits)!=0) atomicAdd(bad,1ull);
    if(k<first || k-first>=count) return;
    const unsigned long long out_gid=s*count+k-first;

    /* ---- (1) bpw-bit digits -> base-2^64 limbs, least significant first.  acc holds the
       bits not yet flushed; v << nacc may spill past bit 63, and the spill is recovered as
       v >> (64-nacc) (nacc is never 0 when the spill happens, because bpw < 64). ------- */
    unsigned long long t[2 * NW + 4];
#pragma unroll 1
    for (int i = 0; i < 2 * NW + 4; ++i) t[i] = 0;
    {
        unsigned long long acc = 0;
        int nacc = 0, limb = 0;
        for (unsigned long long j = 0; j < slot_words; ++j) {
            const unsigned long long v = d[j];
            acc |= v << nacc;
            if (nacc + bpw >= 64) {
                t[limb++] = acc;
                acc = (nacc == 0) ? 0ull : (v >> (64 - nacc));
                nacc = nacc + bpw - 64;
            } else {
                nacc += bpw;
            }
        }
        if (nacc > 0) t[limb++] = acc;
    }
    unsigned long long u[NW];
    if constexpr(MERSENNE) {
        const int limbs=(int)((slot_words*(unsigned long long)bpw+63)/64);
        s4_mersenne_rem<NW>(t,limbs,nw,mersenne_bits,u);
        for(unsigned long long i=0;i<w;++i)out[out_gid*w+i]=i<(unsigned long long)nw ? u[i] : 0ull;
        return;
    }
        const int limbs = (int)((slot_words * (unsigned long long)bpw + 63) / 64);
        const int rc = s4_div_rem<NW>(t, limbs, g_s2g_device_constants.divisor, nw, div_shift, div_recip, u);
        if (rc < 0 && bad != nullptr) atomicAdd(bad, 1ull);
        for (unsigned long long i = 0; i < w; ++i)
            out[out_gid * w + i] = (i < (unsigned long long)nw) ? u[i] : 0ull;
        return;
    
}

/* (the slot-canonical check is folded into s4_reduce_kernel above: the digits are in registers
   there, and a separate kernel would add one launch and one full read per batched multiply) */

/* Reduction mode: 0 = direct plain long division of C (default); 1 = old REDC followed by
   Montgomery restoration with Y.  Read once from NTT_S4_OLDTAIL for same-binary A/B. */


template <int NW>
static void s4_launch_reduce(int nw, int L, unsigned long long nbatch,
                             unsigned long long out_slots, unsigned long long total,
                             const unsigned long long *ddig, unsigned long long n, int bpw,
                             unsigned long long slot_words, const unsigned long long *dn,
                             unsigned long long ninv, const unsigned long long *dy,
                             unsigned long long w, unsigned long long *dout,
                             unsigned long long slot_bits, unsigned long long *dbad,
                             unsigned long long *s4_dbg = nullptr,
                             unsigned long long first = 0, unsigned long long count = ~0ull, int mersenne_bits=0)
{
    const unsigned int th = 128;
    const unsigned int bl = (unsigned int)((total + th - 1) / th);
    if(mersenne_bits && !s4_tail_mont_mode() && s4_dbg==nullptr)
        s4_reduce_kernel<NW,true><<<bl,th>>>(ddig,n,bpw,slot_words,out_slots,nbatch,dn,ninv,nw,
            L,dy,w,dout,slot_bits,dbad,s4_dbg,0,g_div_shift,g_div_recip,first,count==~0ull ? out_slots : count,mersenne_bits);
    else
        s4_reduce_kernel<NW><<<bl, th>>>(ddig, n, bpw, slot_words, out_slots, nbatch, dn, ninv, nw,
            L, dy, w, dout, slot_bits, dbad, s4_dbg, s4_tail_mont_mode(),
            g_div_shift, g_div_recip, first, count==~0ull ? out_slots : count,0);
}

/* ---- THE 2-BY-1 DIVISION PRIMITIVE (objective 4, docs/DEV_GPUOWL_NTT_NOTES.md section 31) -----
   The reduction's tail spends 2*nw^2 MACs putting the R factor back, but with L == nw the whole
   reduction can be ONE plain long division of C by N at about nw^2 MACs.  The measurements say that
   is the right thing to want: t_reduce is 21.6 s at the B2=1e11 shape, it does NOT move when the
   run's wall clock moves between 83.8 s and 94.9 s, and it does not move when the kernel's local
   arrays shrink by a third -- so it is LATENCY-bound on the elimination's serial carry chain, and
   only fewer dependent MACs help.

   A long division needs an exact 2-by-1 quotient digit.  The first version of this function was a
   half-remembered published code sequence and was wrong in 256 of 256 GMP comparisons; it is now
   estimate-plus-exact-remainder-correction, right by construction: Moeller-Granlund's estimate
   floor((v*u1)/2^64) + u1 is never above the true quotient and at most 2 below it, so the loop steps
   down on underflow and up while the remainder is still >= d. */
__host__ __device__ __forceinline__ void s2g_mul64(unsigned long long a, unsigned long long b,
                                                   unsigned long long &lo, unsigned long long &hi)
{
#if defined(__CUDA_ARCH__)
    lo = a * b;
    hi = __umul64hi(a, b);
#else
    /* host code has no 128-bit type here (MSVC), so the high half comes from the 32-bit split:
       a*b = p11*2^64 + K*2^32 + p00 with K = p01 + p10, and floor((K*2^32 + p00)/2^64) =
       (K>>32) + ((K&m32) + (p00>>32))>>32 -- the two carry terms below. */
    const unsigned long long m32 = 0xffffffffull;
    const unsigned long long a0 = a & m32, a1 = a >> 32, b0 = b & m32, b1 = b >> 32;
    const unsigned long long p00 = a0 * b0, p01 = a0 * b1, p10 = a1 * b0, p11 = a1 * b1;
    const unsigned long long ks = (p01 & m32) + (p10 & m32);
    const unsigned long long lo_mid = ks + (p00 >> 32);
    lo = (lo_mid << 32) | (p00 & m32);
    hi = p11 + (p01 >> 32) + (p10 >> 32) + (lo_mid >> 32);
#endif
}

__host__ __device__ __forceinline__ unsigned long long s2g_udiv_2by1(unsigned long long u1,
                                                                    unsigned long long u0,
                                                                    unsigned long long d,
                                                                    unsigned long long v)
{
    unsigned long long lo = 0, hi = 0;
    s2g_mul64(v, u1, lo, hi);
    unsigned long long q = hi + u1;
    for (int guard = 0; guard < 8; ++guard) {
        unsigned long long plo = 0, phi = 0;
        s2g_mul64(q, d, plo, phi);
        const unsigned long long rlo = u0 - plo;
        const unsigned long long b1 = (u0 < plo) ? 1ull : 0ull;
        const unsigned long long rhi = u1 - phi - b1;
        if ((u1 < phi) || (u1 == phi && b1 != 0)) { --q; continue; }   /* estimate too high */
        if (rhi != 0 || rlo >= d) { ++q; continue; }                   /* still >= d: step up */
        return q;                                                      /* exact */
    }
    return q;
}

/* the modulus-level division constants, computed once: FILE-STATICS ON PURPOSE.  An earlier attempt
   kept them in S4Reduce::Shape and the --check-F path then died silently -- exit 3 with EMPTY
   stderr, the gate falling from 47/47 to 16 passed / 32 failed, and a stash-and-rebuild bisect put
   the blame on that round's changes.  Here they change no struct layout and touch no per-shape
   code; the 64-bit values cross the GMP boundary through mpz_import/mpz_export, so no assumption
   about the width of `unsigned long` is involved. */
/* Test the SAME host/device long-division helper against GMP on full multiword remainders.
   Synthetic moduli exercise shift=0/63, nw=1/128/129/256, saturated quotient estimates and the
   add-back branch; those events are too rare for random product coefficients alone. */
static void s4_div_check(const std::vector<unsigned long long> &actual)
{
    mpz_t den, num, want, got, norm, q, dtop, radix;
    mpz_inits(den, num, want, got, norm, q, dtop, radix, nullptr);
    mpz_set_ui(radix, 1);
    mpz_mul_2exp(radix, radix, 64);
    unsigned long long cases = 0, bad = 0, repairs = 0, seed = 0x89abcdef01234567ull;
    const int widths[] = {1, 2, 3, 8, 83, 128, 129, 256};
    for (int fixture = 0; fixture < 1+3*(int)(sizeof(widths)/sizeof(widths[0])); ++fixture) {
        const int nw = (fixture == 0) ? (int)actual.size() : widths[(fixture - 1) / 3];
        std::vector<unsigned long long> hn((size_t)nw), ns((size_t)nw);
        for (int i = 0; i < nw; ++i) {
            seed = seed * 6364136223846793005ull + 1442695040888963407ull;
            hn[(size_t)i] = seed;
        }
        if (fixture == 0) hn = actual;
        else {
            const int kind = (fixture - 1) % 3;
            hn[(size_t)nw - 1] = (kind == 0) ? 0x8000000000000000ull
                : (kind == 1 ? 1ull : 0x1ffffffffull);
            hn[0] |= 1ull;
        }
        mpz_import(den, (size_t)nw, -1, 8, 0, 0, hn.data());
        int shift = 0;
        while ((hn[(size_t)nw - 1] << shift) < 0x8000000000000000ull) ++shift;
        mpz_mul_2exp(norm, den, (unsigned)shift);
        mpz_export(ns.data(), nullptr, -1, 8, 0, 0, norm);
        mpz_import(dtop, 1, -1, 8, 0, 0, &ns[(size_t)nw - 1]);
        mpz_set_ui(num, 1);
        mpz_mul_2exp(num, num, 128);
        mpz_sub_ui(num, num, 1);
        mpz_fdiv_q(q, num, dtop);
        mpz_sub(q, q, radix);
        unsigned long long recip = 0;
        mpz_export(&recip, nullptr, -1, 8, 0, 0, q);
        for (int c = 0; c < 32; ++c) {
            unsigned long long t[2*ecm_stage2::max_words+4] = {}, out[ecm_stage2::max_words] = {};
            if (c < 3) { mpz_set(num, den); if (c == 0) mpz_sub_ui(num, num, 1);
                          if (c == 2) mpz_add_ui(num, num, 1); }
            else if (c == 3) {
                mpz_sub_ui(num, den, 1); mpz_mul_2exp(num, num, 64);
                mpz_add(num, num, radix); mpz_sub_ui(num, num, 1);
            } else if (c < 7) {
                mpz_mul(num, den, den);
                if (c == 4) mpz_sub_ui(num, num, 1);
                if (c == 6) { mpz_add(num, num, den); mpz_sub_ui(num, num, 1); }
            } else if (c == 7) mpz_set_ui(num, 0);
            else {
                for (int i = 0; i < 2 * nw; ++i) {
                    seed = seed * 6364136223846793005ull + 1442695040888963407ull;
                    t[i] = seed;
                }
                mpz_import(num, (size_t)2 * nw, -1, 8, 0, 0, t);
            }
            std::memset(t, 0, sizeof(t));
            size_t limbs = 0;
            mpz_export(t, &limbs, -1, 8, 0, 0, num);
            const int rc = nw<=128 ? s4_div_rem<128>(t, (int)limbs, ns.data(), nw, shift, recip, out)
                : s4_div_rem<256>(t, (int)limbs, ns.data(), nw, shift, recip, out);
            mpz_mod(want, num, den);
            mpz_import(got, (size_t)nw, -1, 8, 0, 0, out);
            ++cases;
            if (rc < 0 || mpz_cmp(want, got) != 0) {
                ++bad;
                if (bad <= 3) std::fprintf(stderr, "%s: div_rem MISMATCH fixture=%d case=%d "
                                                    "nw=%d shift=%d rc=%d\n",
                                           NTT_PROBE_NAME, fixture, c, nw, shift, rc);
            } else repairs += (unsigned long long)rc;
        }
    }
    mpz_clears(den, num, want, got, norm, q, dtop, radix, nullptr);
    stage2_log::print(stage2_log::debug, "s4_div_check: cases=%llu bad=%llu repairs=%llu (full remainder vs GMP, "
                "widths=1..256 shifts=0..63)\n", cases, bad, repairs);
    if (bad || repairs == 0) {
        std::fprintf(stderr, "%s: FATAL: long division failed its GMP/borrow-repair fixtures\n",
                     NTT_PROBE_NAME);
        std::exit(3);
    }
}

/* the modulus-level part: N, its Montgomery constants and the shared device copy */
static int s4_reduce_init(S4Reduce &R, const mpz_t N, size_t W,
                          const std::vector<unsigned long long> &hn, unsigned long long ninv)
{
    R.nw = (int)W;
    R.w = W;
    R.ninv = ninv;
    R.hn = hn;
    mpz_set(R.N, N);
    mpz_t plus;mpz_init(plus);mpz_add_ui(plus,N,1);
    const bool exact_mersenne=mpz_cmp_ui(N,1)>0 && mpz_popcount(plus)==1;
    R.mersenne_bits=(g_s4_mersenne && exact_mersenne) ? (int)mpz_sizeinbase(N,2) : 0;
    mpz_clear(plus);
    s4_oracle_reset(R);
    s4_gmp_pack_check();
    CK(cudaMalloc(&R.dn, (size_t)R.nw * sizeof(unsigned long long)));
    CK(cudaMemcpy(R.dn, R.hn.data(), (size_t)R.nw * sizeof(unsigned long long),
                  cudaMemcpyHostToDevice));
    /* ---- THE DIVISION CONSTANTS, AND THE PRIMITIVE CHECKED AGAINST GMP (section 31) ---------
       N << dshift (top bit set) and its reciprocal word, then 256 numerators per modulus compared
       against GMP -- random ones plus the saturating extreme u1 = d-1, u0 = all ones.  A mismatch is
       FATAL: section 35.1 of the plan document records that this area was once optimised with wrong
       maths and caught by a test, so the arithmetic is checked before anything depends on it. */
    {
        const unsigned long long top = R.hn[(size_t)R.nw - 1];
        int sh = 0;
        while (sh < 64 && ((top << sh) & (1ull << 63)) == 0) ++sh;
        if (sh == 64) sh = 0;
        g_div_shift = sh;
        g_div_nw = R.nw;
        g_div_hns.assign((size_t)R.nw, 0ull);
        for (int i = R.nw - 1; i >= 0; --i) {
            const unsigned long long v = R.hn[(size_t)i];
            g_div_hns[(size_t)i] = (sh == 0)
                ? v : ((v << sh) | (i > 0 ? (R.hn[(size_t)i - 1] >> (64 - sh)) : 0ull));
        }
        CK(cudaMemcpyToSymbol(g_s2g_device_constants, g_div_hns.data(),
                              (size_t)R.nw * sizeof(unsigned long long)));
        const unsigned long long d = g_div_hns[(size_t)R.nw - 1];
        mpz_t num, den, q, t64;
        mpz_inits(num, den, q, t64, nullptr);
        mpz_set_ui(num, 1);
        mpz_mul_2exp(num, num, 128);
        mpz_sub_ui(num, num, 1);
        mpz_import(den, 1, -1, 8, 0, 0, &d);          /* exact, whatever sizeof(unsigned long) is */
        mpz_fdiv_q(q, num, den);
        mpz_set_ui(t64, 1);
        mpz_mul_2exp(t64, t64, 64);
        mpz_sub(q, q, t64);
        g_div_recip = 0;
        mpz_export(&g_div_recip, nullptr, -1, 8, 0, 0, q);
        {
            unsigned long long cases = 0, bad = 0, seed = 0x9e3779b97f4a7c15ull;
            mpz_t n2, qq, rr, gq;
            mpz_inits(n2, qq, rr, gq, nullptr);
            for (int t = 0; t < 256; ++t) {
                seed = seed * 6364136223846793005ull + 1442695040888963407ull;
                const unsigned long long u1 = (t == 1) ? (d - 1) : (seed % d);
                seed = seed * 6364136223846793005ull + 1442695040888963407ull;
                const unsigned long long u0 = (t == 1) ? ~0ull : seed;
                if (u1 >= d) continue;
                const unsigned long long got = s2g_udiv_2by1(u1, u0, d, g_div_recip);
                unsigned long long two[2] = {u0, u1};      /* little-endian: low word first */
                mpz_import(n2, 2, -1, 8, 0, 0, two);
                mpz_fdiv_qr(qq, rr, n2, den);
                mpz_import(gq, 1, -1, 8, 0, 0, &got);
                ++cases;
                if (mpz_cmp(gq, qq) != 0) {
                    ++bad;
                    if (bad <= 3)
                        std::fprintf(stderr, "%s: udiv_2by1 MISMATCH u1=%llu u0=%llu got=%llu\n",
                                     NTT_PROBE_NAME, u1, u0, got);
                }
            }
            mpz_clears(n2, qq, rr, gq, nullptr);
            stage2_log::print(stage2_log::debug, "s4_udiv_check: cases=%llu bad=%llu (2-by-1 division vs GMP, dshift=%d)\n",
                        cases, bad, g_div_shift);
            if (bad) {
                std::fprintf(stderr, "%s: FATAL: the 2-by-1 division primitive disagrees with GMP "
                                     "-- refusing to continue\n", NTT_PROBE_NAME);
                std::exit(3);
            }
        }
        mpz_clears(num, den, q, t64, nullptr);
    }
    s4_div_check(R.hn);
    stage2_log::print(stage2_log::debug, "s4_reduce_mode: algorithm=%s nw=%d dshift=%d (NTT_S4_OLDTAIL=%d)\n",
                s4_tail_mont_mode() ? "montgomery" : (R.mersenne_bits ? "mersenne" : "division"), R.nw, g_div_shift,
                s4_tail_mont_mode());
    stage2_log::print(stage2_log::debug, "s4_mersenne_mode: requested=%d eligible=%d bits=%d enabled=%d\n",(int)g_s4_mersenne,
                (int)exact_mersenne,R.mersenne_bits,(int)(R.mersenne_bits && !s4_tail_mont_mode()));
    const char *fixture=std::getenv("NTT_S4_MERSENNE_TEST");
    if(fixture && std::atoi(fixture))s4_mersenne_check();
    return 0;
}

/* L = the smallest number of elimination steps with 2^slot_bits <= N * 2^(64 L), proved with
   mpz against the ACTUAL N (never against bits(N) as a proxy).  Called ONCE per shape.
 *
 * TWO conditions constrain L, and this function keeps the one the rest of the engine is built on:
 *
 *   [R2] MAGNITUDE -- after L eliminations the value is (v + N*sum)/2^(64L) < N, so the nw
 *        returned words hold it.  This is the condition the formula below solves.
 *   [R1] CONTAINMENT -- the kernel converts the window's digits into limbs t[0..nlimb-1] and
 *        returns t[L..L+nw-1] as v >> 64L, so the bits of v below limb L must already be zero for
 *        the returned words to be the whole of v.  L = nlimb - nw is what makes this exact.
 *
 * For the F-tree / fold / fold-loop shapes the two agree (S=129 gives nlimb=5, nw=3: the formula
 * gives L=3 and nlimb-nw=2, and L=3 >= nlimb-nw=2 still contains the value -- measured: forcing
 * L=nlimb-nw=2 there makes the magnitude test fail and aborts the run, so L=3 is the correct one
 * and the gates stay green).  For the S5 SLOT shapes the two DISAGREE (see the S5Shape comment):
 * at S=129/slot_bits=259 the window bits lands at limb 1, L=3 in the formula, and the reduction
 * then returns 0 for every coefficient.  That is a real, still-open defect of the S5 path -- it is
 * recorded in build_cuda_cmake/_S5_HANDOFF.md, and it is why NTT_S5_ON is still opt-in. */
static S4Reduce::Shape *s4_shape_init(S4Reduce &R, unsigned long long P,
                                      unsigned long long slot_bits,
                                      unsigned long long slot_stride,
                                      unsigned long long slot_words, int bpw)
{
    S4Reduce::Shape *S = R.find(slot_bits, slot_words, bpw);
    if (S) return S;
    S = new S4Reduce::Shape();
    S->slot_bits = slot_bits;
    S->slot_stride = slot_stride;
    S->slot_words = slot_words;
    S->bpw = bpw;
    S->P = P;
    S->nlimb = (int)((slot_stride + 63) / 64);
    mpz_t Cmax, Rl, prod;
    mpz_inits(Cmax, Rl, prod, nullptr);
    mpz_set_ui(Cmax, 1);
    mpz_mul_2exp(Cmax, Cmax, (unsigned long)slot_bits);   /* Cmax = 2^slot_bits (exclusive) */
    int L = 1;
    for (;; ++L) {
        mpz_set_ui(Rl, 1);
        mpz_mul_2exp(Rl, Rl, (unsigned long)(64 * L));
        mpz_mul(prod, R.N, Rl);
        if (mpz_cmp(Cmax, prod) <= 0) break;
        if (L > 4096) {
            std::fprintf(stderr, "%s: FATAL: cannot find L with 2^slot_bits <= N*2^(64L) "
                                 "(slot_bits=%llu, N has %d words)\n", NTT_PROBE_NAME,
                         slot_bits, R.nw);
            std::exit(3);
        }
    }
    S->L = L;
    S->bound_bits = (double)mpz_sizeinbase(prod, 2) - (double)slot_bits;
    /* Y = 2^(64 (L+nw)) mod N: goes from the 2^(-64L) domain to the plain one in one multiply */
    {
        mpz_t Y;
        mpz_init(Y);
        mpz_set_ui(Y, 1);
        mpz_mul_2exp(Y, Y, (unsigned long)(64 * (L + R.nw)));
        mpz_mod(Y, Y, R.N);
        S->hy.assign((size_t)R.nw, 0ull);
        if (!mpz_to_words(S->hy, (size_t)R.nw, Y)) {
            std::fprintf(stderr, "%s: FATAL: Y does not fit %d words\n", NTT_PROBE_NAME, R.nw);
            std::exit(3);
        }
        mpz_clear(Y);
    }
    mpz_clears(Cmax, Rl, prod, nullptr);
    /* the device array is 2*NW+4 limbs with NW the dispatched template width: the digit->limb
       conversion must fit nlimb+1 of them and the elimination loop reaches L+nw+2 */
    int nwmax = ecm_stage2::max_words;
    for (int v : {4, 8, 16, 32, 64, 128, 256}) if (R.nw <= v) { nwmax = v; break; }
    const long long need = (S->nlimb + 1) > (L + R.nw + 2) ? (S->nlimb + 1) : (L + R.nw + 2);
    if (need > 2 * nwmax + 4) {
        std::fprintf(stderr, "%s: FATAL: the reduction needs %lld limbs but the %d-word "
                             "template only has %d\n", NTT_PROBE_NAME, need, nwmax,
                     2 * nwmax + 4);
        std::exit(3);
    }
    CK(cudaMalloc(&S->dy, (size_t)R.nw * sizeof(unsigned long long)));
    CK(cudaMemcpy(S->dy, S->hy.data(), (size_t)R.nw * sizeof(unsigned long long),
                  cudaMemcpyHostToDevice));
    R.shapes.push_back(S);
    stage2_log::print(stage2_log::debug, "s4_reduce_shape: P=%llu slot_bits=%llu slot_stride=%llu nlimb=%d L=%d "
                "(per-coefficient preprint: %d digits base 2^%d -> %d limbs, then %d "
                "elimination steps of %d limbs) ; precondition C < 2^slot_bits <= N*2^(64L) "
                "proved with GMP for THIS N: margin=%.1f bits\n",
                P, slot_bits, slot_stride, S->nlimb, S->L, (int)slot_words, bpw, S->nlimb,
                S->L, R.nw, S->bound_bits);
    return S;
}

/* (the slot-canonical check is folded INTO the reduction kernel: the digits are already in
   registers there, so the precondition costs no extra launch and no extra read) */

/* the hook the probe calls: reduce `nbatch*out_slots` coefficients of the digit buffer */
static void s4_check_reduced(S4Reduce &R, S4Reduce::Shape *S, const unsigned long long *ddig,
                             unsigned long long n, unsigned long long out_slots,
                             unsigned long long nbatch, const unsigned long long *d_out,
                             long long sample_limit, unsigned long long first = 0,
                             unsigned long long count = ~0ull);

static void s4_reduce_hook(void *ctx, const unsigned long long *digits, unsigned long long n,
                           int bpw, unsigned long long slot_words, unsigned long long slot_bits,
                           unsigned long long out_slots, unsigned long long nbatch,
                           unsigned long long *out, unsigned long long w,
                           unsigned long long first, unsigned long long count)
{
    S4Reduce &R = *(S4Reduce *)ctx;
    S4Reduce::Shape *S = R.find(slot_bits, slot_words, bpw);
    if (!S) {
        std::fprintf(stderr, "%s: FATAL: the reduction was never initialised for this shape "
                             "(slot_bits=%llu bpw=%d slot_words=%llu)\n", NTT_PROBE_NAME,
                     slot_bits, bpw, slot_words);
        std::exit(3);
    }
    if (w != R.w) {
        std::fprintf(stderr, "%s: FATAL: the hook was given w=%llu words per coefficient but "
                             "the modulus has %d\n", NTT_PROBE_NAME, w, R.nw);
        std::exit(3);
    }
    const unsigned long long total = out_slots * nbatch;
    if (!R.dbad) {
        CK(cudaMalloc(&R.dbad, sizeof(unsigned long long)));
        R.dbad_host = 0;
    }
    const double t0 = now_s();
    /* THE COUNTER'S RESET IS NOT PER-CALL WORK (objective 4, section 36).  `dbad` counts slot
       windows whose digits overflowed the window: it is MONOTONE and purely diagnostic, yet the
       old code reset it with a cudaMemset AND read it back with an 8-byte device-to-host copy on
       EVERY call.  A small pageable D2H is pure latency here (~10-20 us; bulk D2H runs at
       2.9 GB/s), so at 589960 calls that is seconds and at the production shape's 3.8e6 calls it
       is minutes -- all of it inside "the unattributed part of ntt_seconds".  Now it is reset
       once per shape and read every g_s4_check_every calls (the GMP oracle's cadence), so a
       violation is still reported within 8 calls of happening. */
    if (S->calls == 0) {
        /* ... AND THE PENDING READBACK MUST BE RESOLVED BEFORE THE RESET (section 43): `dbad` is
           shared by every shape and is zeroed when a shape starts, so a copy still in flight
           would read the post-reset zero and look exactly like a violation.  Once per shape, so
           the wait is free. */
        s4_dbad_resolve(R);
        CK(cudaMemset(R.dbad, 0, sizeof(unsigned long long)));
    }
    /* MARK THE KERNEL IN-STREAM AND RESOLVE IT LATER (section 41): no cudaDeviceSynchronize()
       here any more -- see the note on `t_reduce` in S4Reduce::Shape. */
    const int dts = S->dt_slot();
    CK(cudaEventCreate(&S->dt_ev[dts][0]));
    CK(cudaEventCreate(&S->dt_ev[dts][1]));
    CK(cudaEventRecord(S->dt_ev[dts][0]));
    S2G_DISPATCH(R.nw, s4_launch_reduce, (int)R.nw, S->L, nbatch, out_slots, total, digits, n,
                 bpw, slot_words, R.dn, R.ninv, S->dy, w, out, slot_bits, R.dbad, nullptr, first, count,R.mersenne_bits);
    CK(cudaGetLastError());
    CK(cudaEventRecord(S->dt_ev[dts][1]));
    S->dt_used[dts] = true;
    S->t_reduce_host += now_s() - t0;
    const double th0 = now_s();
    /* ---- THE DIAGNOSTIC READBACK MUST NOT DRAIN EITHER (objective 4, section 43) -----------
       This is the SAME trap section 41 found in the carry-residual check, in the hook instead of
       the multiply: an 8-byte blocking copy issued right after a chunk's worth of kernels has
       been queued waits for all of them, and it MEASURED 19.2 s at the production shape (1946
       calls, ~10 ms each) for a counter that is never read as a control input.  `dbad` is
       monotone, so reading it LATE is exactly as good: the copy is asynchronous into pinned
       memory and the PREVIOUS one is compared on the next call (and at the end of the run).
       A violation is therefore still reported -- one call later instead of immediately. */
    s4_dbad_resolve(R);
    if (!R.dbad_inflight && (S->calls % g_s4_check_every) == 0) {
        if (!R.h_dbad) {
            CK(cudaHostAlloc((void **)&R.h_dbad, sizeof(unsigned long long), cudaHostAllocDefault));
            *R.h_dbad = 0;
            CK(cudaEventCreate(&R.ev_dbad));
        }
        CK(cudaMemcpyAsync(R.h_dbad, R.dbad, sizeof(unsigned long long),
                           cudaMemcpyDeviceToHost));
        CK(cudaEventRecord(R.ev_dbad));
        R.dbad_inflight = true;
        R.dbad_shape = S;
        R.dbad_expected = S->canon_bad;
    }
    ++R.reduce_calls;
    S->t_hookd2h += now_s() - th0;
    ++S->calls;
    S->coeffs += count*nbatch;
    R.coeffs_total += count*nbatch;
    /* This reduction is already queued: comparing an OLD snapshot can overlap it. */
    const double tr0 = now_s();
    s4_oracle_reap(R, false);
    g_oracle.t_reap += now_s() - tr0;
    /* THE IN-RUN ORACLE, here rather than at the call site: the digit buffer belongs to the
       multiply and is released when it returns, so a check done afterwards would read freed
       device memory (that was a real crash, "CUDA error invalid argument", the first time the
       S2 tail ran without the arena).  Reading it here is also the stronger check: the raw
       digits GMP sees are the ones the transform and the carry just produced. */
    if (g_s4_sample_limit > 0 && (S->calls <= 1 || (S->calls % g_s4_check_every) == 0)) {
        const double ts0 = now_s();
        s4_check_reduced(R, S, digits, n, out_slots, nbatch, out, g_s4_sample_limit, first, count);
        S->t_hooksample += now_s() - ts0;
    }
}

/* one coefficient, reduced on the host with GMP the way the pre-S4 code did it: the digits
   are the exact integer C = sum_j d[j] 2^(bpw j), so C mod N is the oracle. */
/* Exact SUM of arbitrary 64-bit digits at bpw-bit offsets, including NONCANONICAL digits.
   Use addition with carry, not OR: OR would reproduce the GPU assembler's failure for bad
   carry output and turn an independent oracle into a false agreement. Two spare words bound
   the possible carry beyond the last digit's 64-bit value. */
static void s4_gmp_assemble(mpz_t v, const unsigned long long *digits,
                           unsigned long long nslots, int bpw)
{
    std::vector<unsigned long long> words((size_t)((nslots * (unsigned long long)bpw + 63) / 64) + 2, 0ull);
    for (unsigned long long j = 0; j < nslots; ++j) {
        const unsigned long long bit = j * (unsigned long long)bpw, value = digits[j];
        const size_t k = (size_t)(bit / 64);
        const int shift = (int)(bit % 64);
        const unsigned long long low = value << shift, old = words[k];
        words[k] = old + low;
        const unsigned long long carry = (words[k] < old) ? 1ull : 0ull;
        const unsigned long long high = shift ? (value >> (64 - shift)) : 0ull;
        const unsigned long long before = words[k + 1], sum = before + high;
        const unsigned long long next = sum + carry;
        words[k + 1] = next;
        bool c = (sum < before) || (next < sum);
        for (size_t i = k + 2; c; ++i) {
            if (i >= words.size()) {
                std::fprintf(stderr, "%s: FATAL: oracle assembly carry escaped its bound\n", NTT_PROBE_NAME);
                std::exit(3);
            }
            c = (++words[i] == 0);
        }
    }
    mpz_import(v, words.size(), -1, sizeof(unsigned long long), 0, 0, words.data());
}

static void s4_gmp_assemble_slow(mpz_t v, const unsigned long long *digits,
                                unsigned long long nslots, int bpw)
{
    mpz_set_ui(v, 0);
    for (unsigned long long j = nslots; j-- > 0;) {
        mpz_mul_2exp(v, v, (unsigned)bpw);
        mpz_add_u64(v, digits[j]);
    }
}

static void s4_gmp_pack_check()
{
    const int counts[] = {0, 1, 2, 3, 63, 64, 65, 502, 1024};
    const int bases[] = {1, 19, 21, 31, 32, 63, 64};
    unsigned long long seed = 0x1234fedcba987654ull, cases = 0, bad = 0;
    mpz_t want, got;
    mpz_inits(want, got, nullptr);
    for (int n : counts) for (int b : bases) for (int pattern = 0; pattern < 6; ++pattern) {
        std::vector<unsigned long long> digits((size_t)n, 0ull);
        for (int i = 0; i < n; ++i) {
            seed = seed * 6364136223846793005ull + 1442695040888963407ull;
            const unsigned long long mask = (b == 64) ? ~0ull : ((1ull << b) - 1ull);
            digits[(size_t)i] = pattern == 0 ? 0ull : pattern == 1 ? (seed & mask)
                : pattern == 2 ? ~0ull : pattern == 3 ? (i == n - 1 ? ~0ull : 0ull)
                : pattern == 4 ? seed : ((i & 1) ? ~0ull : 1ull);
        }
        s4_gmp_assemble_slow(want, digits.data(), (unsigned long long)n, b);
        s4_gmp_assemble(got, digits.data(), (unsigned long long)n, b);
        ++cases;
        if (mpz_cmp(want, got) != 0) ++bad;
    }
    mpz_clears(want, got, nullptr);
    stage2_log::print(stage2_log::debug, "s4_oracle_pack_check: cases=%llu bad=%llu (exact integers, bpw=1..64, "
                "canonical AND noncanonical digits)\n", cases, bad);
    if (bad) {
        std::fprintf(stderr, "%s: FATAL: packed oracle assembly disagrees with GMP\n", NTT_PROBE_NAME);
        std::exit(3);
    }
}

static void s4_gmp_reduce(mpz_t out, const unsigned long long *digits, unsigned long long nslots,
                          int bpw, const mpz_t N, bool timed = false)
{
    mpz_t v;
    mpz_init(v);
    double t0 = timed ? now_s() : 0.0;
    if (g_s4_oracle_pack) s4_gmp_assemble(v, digits, nslots, bpw);
    else s4_gmp_assemble_slow(v, digits, nslots, bpw);
    if (timed) { g_oracle.t_num += now_s() - t0; t0 = now_s(); }
    mpz_mod(out, v, N);
    if (timed) g_oracle.t_mod += now_s() - t0;
    mpz_clear(v);
}

/* the reduction kernel against GMP, on the ACTUAL shape, with adversarial patterns included */
static int s4_reduce_selftest(S4Reduce &R, S4Reduce::Shape *S)
{
    const unsigned long long cases = 96;
    const unsigned long long slot_words = S->slot_words;
    const int bpw = S->bpw;
    const unsigned long long maxd = (1ull << bpw) - 1ull;
    std::vector<unsigned long long> dig((size_t)(cases * slot_words), 0),
                                    got((size_t)(cases * R.w), 0);
    uint64_t s = 0x243f6a8885a308d3ull;
    /* THE EXACT WINDOW THE S5 DESCENT FAILED ON (section 47): 0xfc861cfcb724b25400000001, the
       one value for which the REDC invariant `r == V*2^-(64L) mod N` was measured to break, in
       limb 0 only.  It is put in as a CASE so that "the same shape passes on synthetic values and
       fails on the real one" stops being a paradox and becomes a one-second reproduction. */
    const bool have_real_case = (S->slot_bits == 259ull && bpw == 7 && slot_words == 37);
    unsigned long long real_dig[64] = {0};
    if (have_real_case) {
        mpz_t v, two, q, rem;
        mpz_inits(v, two, q, rem, nullptr);
        mpz_set_str(v, "fc861cfcb724b25400000001", 16);
        mpz_set_ui(two, 1);
        mpz_mul_2exp(two, two, (unsigned)bpw);
        /* GMP's fdiv_qr requires all four variables to be DISTINCT (the first version aliased
           the remainder with the dividend, so this case did not contain the window it claimed
           to -- which is why "the same shape passes in isolation" survived one round of being
           turned into a unit test at all, section 48). */
        for (unsigned long long j = 0; j < slot_words; ++j) {
            mpz_fdiv_qr(q, rem, v, two);
            real_dig[j] = mpz_get_ui(rem);
            mpz_set(v, q);
        }
        mpz_clears(v, two, q, rem, nullptr);
        /* prove the decomposition, digit by digit: rebuilt must equal the hex we started from */
        {
            mpz_t back;
            mpz_init_set_ui(back, 0);
            for (unsigned long long j = slot_words; j-- > 0;) {
                mpz_mul_2exp(back, back, (unsigned)bpw);
                mpz_add_ui(back, back, real_dig[j]);
            }
            char *bs = mpz_get_str(nullptr, 16, back);
            stage2_log::print(stage2_log::debug, "s4_reduce_realcase: rebuilt=%s ok=%d\n", bs,
                        (std::strcmp(bs, "fc861cfcb724b25400000001") == 0) ? 1 : 0);
            void (*ff)(void *, size_t) = nullptr;
            mp_get_memory_functions(nullptr, nullptr, &ff);
            ff(bs, std::strlen(bs) + 1);
            mpz_clear(back);
        }
    }
    for (unsigned long long c = 0; c < cases; ++c) {
        for (unsigned long long j = 0; j < slot_words; ++j) {
            s = s * 6364136223846793005ull + 1442695040888963407ull;
            unsigned long long v;
            switch (c % 7) {
                case 0: v = maxd; break;                                  /* all ones */
                case 1: v = 0; break;                                     /* zero */
                case 2: v = (j + 1 == slot_words) ? 1ull : 0ull; break;   /* single top bit */
                case 3: v = (j == 0) ? maxd : 0ull; break;                /* low digit full */
                /* ---- A DENSE WINDOW WHOSE VALUE IS BELOW N (section 46) --------------------
                   The S5 descent failed on exactly this shape of window: 96 bits, i.e. TWO limbs
                   both fully populated, and below N so the reduction must hand the value back
                   unchanged.  Every other pattern here is either >= N (all ones, single top bit,
                   the random default) or occupies one limb (zero, low digit full), so the one
                   case that was wrong was the one case not covered.  A test suite is only as
                   good as the shapes it contains. */
                case 6: v = (j < (slot_words * 2ull) / 5ull) ? (s & maxd) : 0ull; break;
                /* the recorded real failure (section 47) */
                case 5: v = have_real_case ? real_dig[j] : (s & maxd); break;
                default: v = s & maxd; break;
            }
            dig[(size_t)(c * slot_words + j)] = v;
        }
    }
    /* the last digit must respect the slot window, exactly like a real product coefficient */
    {
        const unsigned long long top_bits = S->slot_bits - (slot_words - 1) *
                                            (unsigned long long)bpw;
        if (top_bits < 64)
            for (unsigned long long c = 0; c < cases; ++c)
                dig[(size_t)(c * slot_words + slot_words - 1)] &= ((1ull << top_bits) - 1ull);
    }
    unsigned long long *dd = nullptr, *dout = nullptr, *ddbg = nullptr;
    /* ---- THE GEOMETRY A/B (section 48) ----------------------------------------------------
       Every launch argument here is identical to the S5 multiply's EXCEPT `n` (the slice stride),
       `out_slots` and `total`: this selftest uses `cases*slot_words / cases / cases`, the real call
       uses `N / out_slots / out_slots*nbatch`.  NTT_S5_SELFTEST_N makes this selftest run with the
       REAL stride, which is the last variable left between "the same window passes in isolation"
       and "it fails in situ". */
    unsigned long long sel_n = cases * slot_words;
    {
        const char *e = std::getenv("NTT_S5_SELFTEST_N");
        if (e && *e) {
            const unsigned long long v = std::strtoull(e, nullptr, 10);
            if (v >= slot_words) sel_n = v;
        }
    }
    const unsigned long long sel_os = (sel_n < cases * slot_words) ? (sel_n / slot_words) : cases;
    if (sel_os < cases)
        stage2_log::print(stage2_log::debug, "s4_reduce_selftest_geom: n=%llu out_slots=%llu (the real call's stride)\n",
                    sel_n, sel_os);
    constexpr bool dbg=false; // Legacy REDC forensic dump is development-only.
    CK(cudaMalloc(&dd, dig.size() * sizeof(unsigned long long)));
    CK(cudaMalloc(&dout, (size_t)(cases * R.w) * sizeof(unsigned long long)));
    if (dbg) CK(cudaMalloc(&ddbg, 4 * 24 * sizeof(unsigned long long)));
    CK(cudaMemcpy(dd, dig.data(), dig.size() * sizeof(unsigned long long),
                  cudaMemcpyHostToDevice));
    S2G_DISPATCH(R.nw, s4_launch_reduce, (int)R.nw, S->L, 1ull, sel_os, sel_os, dd,
                 sel_n, bpw, slot_words, R.dn, R.ninv, S->dy,
                 (unsigned long long)R.w, dout, S->slot_bits, (unsigned long long *)nullptr, ddbg,0,~0ull,R.mersenne_bits);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(got.data(), dout, got.size() * sizeof(unsigned long long),
                  cudaMemcpyDeviceToHost));
    if (dbg) {
        std::vector<unsigned long long> hb(64, 0ull);
        CK(cudaMemcpy(hb.data(), ddbg, hb.size() * sizeof(unsigned long long),
                      cudaMemcpyDeviceToHost));
        for (int c = 0; c < 4; ++c) {
            char *ds = nullptr;
            {
                mpz_t dv;
                mpz_init(dv);
                for (unsigned long long j = slot_words; j-- > 0;) {
                    mpz_mul_2exp(dv, dv, (unsigned)bpw);
                    mpz_add_u64(dv, dig[(size_t)(c * slot_words + j)]);
                }
                ds = mpz_get_str(nullptr, 16, dv);
                mpz_clear(dv);
            }
            std::fprintf(stderr, "s4_dbg: case=%d L=%llu bpw=%llu sw=%llu d0..2=[%llu,%llu,%llu] "
                                 "dlast=%llu t=[%llx,%llx,%llx,%llx,%llx,%llx] "
                                 "r=[%llx,%llx,%llx] window=%s\n", c, hb[(size_t)(c * 16 + 9)],
                         hb[(size_t)(c * 16 + 15)], hb[(size_t)(c * 16 + 10)],
                         hb[(size_t)(c * 16 + 11)], hb[(size_t)(c * 16 + 12)],
                         hb[(size_t)(c * 16 + 13)], hb[(size_t)(c * 16 + 14)],
                         hb[(size_t)(c * 16 + 0)], hb[(size_t)(c * 16 + 1)],
                         hb[(size_t)(c * 16 + 2)], hb[(size_t)(c * 16 + 3)],
                         hb[(size_t)(c * 16 + 4)], hb[(size_t)(c * 16 + 5)],
                         hb[(size_t)(c * 16 + 6)], hb[(size_t)(c * 16 + 7)],
                         hb[(size_t)(c * 16 + 8)], ds);
            /* the limbs the kernel's own conversion would produce from these digits, computed on
               the host with the same rule, so a mismatch localises to the conversion */
            {
                std::vector<unsigned long long> tl(2 * (size_t)R.nw + 4, 0ull);
                unsigned long long acc = 0, nacc = 0, limb = 0;
                for (unsigned long long j = 0; j < slot_words; ++j) {
                    const unsigned long long v = dig[(size_t)(c * slot_words + j)];
                    acc |= (v << nacc);
                    if ((unsigned long long)bpw + nacc >= 64) {
                        tl[(size_t)limb++] = acc;
                        acc = (nacc == 0) ? 0ull : (v >> (64 - nacc));
                        nacc = nacc + (unsigned long long)bpw - 64;
                    } else {
                        nacc += (unsigned long long)bpw;
                    }
                }
                if (nacc) tl[(size_t)limb++] = acc;
                std::fprintf(stderr, "s4_dbg_hostconv: case=%d limbs_written=%llu "
                                     "tl=[%llx,%llx,%llx,%llx,%llx,%llx]\n", c, limb,
                             tl[0], tl[1], tl[2], tl[3], tl[4], tl[5]);
            }
            void (*ff)(void *, size_t) = nullptr;
            mp_get_memory_functions(nullptr, nullptr, &ff);
            ff(ds, std::strlen(ds) + 1);
        }
        cudaFree(ddbg);
    }
    cudaFree(dd);
    cudaFree(dout);
    unsigned long long bad = 0;
    long long first = -1;
    mpz_t want, mine;
    mpz_inits(want, mine, nullptr);
    for (unsigned long long c = 0; c < sel_os; ++c) {
        s4_gmp_reduce(want, &dig[(size_t)(c * slot_words)], slot_words, bpw, R.N);
        mpz_import(mine, (size_t)R.w, -1, 8, 0, 0, &got[(size_t)(c * R.w)]);
        if (mpz_cmp(want, mine) != 0) {
            if (!bad) {
                first = (long long)c;
                char *ws = mpz_get_str(nullptr, 16, want);
                char *ms = mpz_get_str(nullptr, 16, mine);
                stage2_log::print(stage2_log::debug, "s4_reduce_selftest_bad: case=%llu pattern=%llu gmp=%s gpu=%s\n",
                            (unsigned long long)c, (unsigned long long)(c % 6), ws, ms);
                void (*ff)(void *, size_t) = nullptr;
                mp_get_memory_functions(nullptr, nullptr, &ff);
                ff(ws, std::strlen(ws) + 1);
                ff(ms, std::strlen(ms) + 1);
            }
            ++bad;
        }
    }
    mpz_clears(want, mine, nullptr);
    S->selftest_cases = cases;
    S->selftest_bad = bad;
    S->selftest_first = first;
    if (std::getenv("NTT_S5_REDDUMP") && *std::getenv("NTT_S5_REDDUMP")
        && std::atoi(std::getenv("NTT_S5_REDDUMP")) != 0) {
        char *ws = mpz_get_str(nullptr, 16, want);
        char *ms = mpz_get_str(nullptr, 16, mine);
        std::fprintf(stderr, "s4_selftest_dump: P=%llu slot_bits=%llu slot_words=%llu bpw=%d "
                             "L=%d nlimb=%d nw=%d w=%d last_case gmp=%s gpu=%s\n", S->P,
                     S->slot_bits, S->slot_words, S->bpw, S->L, S->nlimb, R.nw, (int)R.w, ws, ms);
        void (*ff)(void *, size_t) = nullptr;
        mp_get_memory_functions(nullptr, nullptr, &ff);
        ff(ws, std::strlen(ws) + 1);
        ff(ms, std::strlen(ms) + 1);
    }
    stage2_log::print(stage2_log::debug, "s4_reduce_selftest: P=%llu cases=%llu mismatches=%llu first_bad=%lld (device "
                "REDC of a %llu-bit coefficient mod the actual N vs GMP; six digit patterns "
                "incl. all-ones/zero/single-top-bit/single-low-digit)\n", S->P, cases, bad,
                first, S->slot_bits);
    return bad ? 1 : 0;
}

/* the in-run check: verify the device's reduced coefficients against GMP, from the batch's OWN
   digits -- so it checks the transform, the carry, the window convention and the reduction
   together.  BOUNDED ON PURPOSE: at most TWO device->host copies and at most `sample_limit`
   mpz reductions per call, whatever the batch size (a copy per coefficient, or per slice, costs
   more than the multiply it is checking -- measured: 16357 calls x 4096 runs of one slice each
   turned a 1.05 s run into a 15 s one).  Coverage over the run comes from ROTATING the slice
   and the start coefficient with the call counter, so consecutive calls look at different
   places.  A run always lies INSIDE ONE SLICE: the digits are slice-major at stride n, so the
   coefficient order is contiguous within a slice and NOT across slices. */
static void s4_compare_snapshot(S4Reduce &R, S4Reduce::Shape *S,
    unsigned long long s, unsigned long long k0, unsigned long long cnt,
    unsigned long long out_slots, const unsigned long long *digbuf,
    const unsigned long long *redbuf)
{
    const double t0 = now_s();
    mpz_t want, mine;
    mpz_inits(want, mine, nullptr);
        for (unsigned long long q = 0; q < cnt; ++q) {
            s4_gmp_reduce(want, &digbuf[(size_t)(q * S->slot_words)], S->slot_words, S->bpw,
                          R.N, true);
            mpz_import(mine, (size_t)R.w, -1, 8, 0, 0, &redbuf[(size_t)(q * R.w)]);
            ++S->checked;
            if (mpz_cmp(want, mine) != 0) {
                if (!S->check_bad) {
                    S->check_first = s * out_slots + k0 + q;
                    char *ws = mpz_get_str(nullptr, 16, want);
                    char *ms = mpz_get_str(nullptr, 16, mine);
                    /* ---- AND THE TWO THINGS THAT SPLIT THE CAUSE (section 51) ------------
                       A dictionary of the failure without its context is what cost rounds 14-17:
                       report whether THIS window's digits are canonical (the kernel's conversion
                       ORs them in, so a digit >= 2^bpw loses its high bits) and what the REDC
                       invariant requires r to be, so "the carry did not canonicalise" and "the
                       elimination is wrong" are distinguishable from one line. */
                    unsigned long long mx = 0, mj = 0;
                    for (unsigned long long jj = 0; jj < S->slot_words; ++jj) {
                        const unsigned long long v = digbuf[(size_t)(q * S->slot_words + jj)];
                        if (v > mx) { mx = v; mj = jj; }
                    }
                    mpz_t tv, rexp, tw;
                    mpz_inits(tv, rexp, tw, nullptr);
                    for (unsigned long long jj = S->slot_words; jj-- > 0;) {
                        mpz_mul_2exp(tv, tv, (unsigned)S->bpw);
                        mpz_add_u64(tv, digbuf[(size_t)(q * S->slot_words + jj)]);
                    }
                    mpz_set_ui(tw, 1);
                    mpz_mul_2exp(tw, tw, (mp_bitcnt_t)(64 * S->L));
                    mpz_invert(tw, tw, R.N);
                    mpz_mul(rexp, tv, tw);
                    mpz_mod(rexp, rexp, R.N);
                    char *rs = mpz_get_str(nullptr, 16, rexp);
                    stage2_log::print(stage2_log::debug, "s4_reduce_CHECK_bad: slice=%llu k=%llu gmp=%s gpu=%s "
                                "max_digit=%llu/%llu(mj=%llu)%s rexp=%s\n",
                                s, k0 + q, ws, ms, mx, (1ull << S->bpw) - 1ull, mj,
                                (mx < (1ull << S->bpw)) ? " CANON" : " NONCANON", rs);
                    void (*ff)(void *, size_t) = nullptr;
                    mp_get_memory_functions(nullptr, nullptr, &ff);
                    ff(ws, std::strlen(ws) + 1);
                    ff(ms, std::strlen(ms) + 1);
                    ff(rs, std::strlen(rs) + 1);
                    mpz_clears(tv, rexp, tw, nullptr);
                }
                ++S->check_bad;
                std::fprintf(stderr, "%s: FATAL: the device reduction disagrees with GMP at "
                                     "slice %llu coefficient %llu\n", NTT_PROBE_NAME, s, k0 + q);
                std::exit(3);
            }
        }
    mpz_clears(want, mine, nullptr);
    ++g_oracle.compared;
    g_oracle.samples += cnt;
    g_oracle.t_gmp += now_s() - t0;
}

/* Resolve ONE oldest snapshot.  Only ring pressure and final drain may wait. */
static bool s4_oracle_one(S4Reduce &R, bool wait)
{
    if (g_oracle.head == g_oracle.tail) return false;
    S4OracleSlot &slot = g_oracle_slot[g_oracle.head % g_s4_oracle_ring];
    const cudaError_t status = cudaEventQuery(slot.ready);
    if (status == cudaErrorNotReady && !wait) return false;
    if (status == cudaErrorNotReady) {
        const double t0 = now_s();
        CK(cudaEventSynchronize(slot.ready));
        g_oracle.t_wait += now_s() - t0;
    } else CK(status);
    s4_compare_snapshot(R, slot.shape, slot.slice, slot.k0, slot.count, slot.out_slots,
                        slot.digits, slot.reduced);
    ++g_oracle.head;
    return true;
}

static void s4_oracle_reap(S4Reduce &R, bool wait)
{
    if (g_oracle_owner != &R) return;
    while (s4_oracle_one(R, wait)) {}
}

static void s4_oracle_drain(S4Reduce &R)
{
    if (g_oracle_owner != &R) return;
    const double t0 = now_s();
    /* Gate-only fault injection: corrupt the LAST pending host snapshot, specifically proving
       that the final drain validates it before success or destruction. Never alter device data. */
    const char *bad = std::getenv("NTT_S4_ORACLE_TEST_BAD");
    if (bad && std::atoi(bad) != 0 && g_oracle.head != g_oracle.tail) {
        S4OracleSlot &slot = g_oracle_slot[(g_oracle.tail - 1) % g_s4_oracle_ring];
        CK(cudaEventSynchronize(slot.ready));
        slot.reduced[0] ^= 1ull;
    }
    s4_oracle_reap(R, true);
    if (g_oracle.selected != g_oracle.compared) {
        std::fprintf(stderr, "%s: FATAL: oracle lost snapshots (%llu selected, %llu compared)\n",
                     NTT_PROBE_NAME, g_oracle.selected, g_oracle.compared);
        std::exit(3);
    }
    g_oracle.t_drain += now_s() - t0;
}

static void s4_oracle_release(S4Reduce &R)
{
    if (g_oracle_owner != &R) return;
    s4_oracle_drain(R);
    for (S4OracleSlot &slot : g_oracle_slot) {
        if (slot.digits) CK(cudaFreeHost(slot.digits));
        if (slot.reduced) CK(cudaFreeHost(slot.reduced));
        if (slot.ready) CK(cudaEventDestroy(slot.ready));
        slot = S4OracleSlot{};
    }
    if (g_oracle_block_ev) CK(cudaEventDestroy(g_oracle_block_ev));
    g_oracle_block_ev = nullptr;
    g_oracle_owner = nullptr;
}

static void s4_oracle_block_ready()
{
    if (!g_oracle_block_ev) CK(cudaEventCreateWithFlags(&g_oracle_block_ev, cudaEventDisableTiming));
    CK(cudaEventRecord(g_oracle_block_ev));
    const double t0 = now_s();
    CK(cudaEventSynchronize(g_oracle_block_ev));
    g_oracle.t_wait += now_s() - t0;
}

static void s4_check_reduced(S4Reduce &R, S4Reduce::Shape *S, const unsigned long long *ddig,
                             unsigned long long n, unsigned long long out_slots,
                             unsigned long long nbatch, const unsigned long long *d_out,
                             long long sample_limit, unsigned long long first,
                             unsigned long long count)
{
    if(count==~0ull) count=out_slots;
    const unsigned long long total = count * nbatch;
    if (!total || sample_limit == 0) return;
    const double tc0 = now_s();
    const unsigned long long lim = (unsigned long long)sample_limit;
    const bool small_batch = (nbatch <= 4 && total <= lim);
    std::vector<std::array<unsigned long long, 3>> runs;
    if (small_batch) {
        for (unsigned long long s = 0; s < nbatch; ++s) runs.push_back({s, 0ull, count});
        ++S->full_checks;
    } else {
        const unsigned long long cnt = std::min(lim, count);
        uint64_t seed = 0x9e3779b97f4a7c15ull * (S->calls + 1) + total;
        seed = seed * 6364136223846793005ull + 1442695040888963407ull;
        const unsigned long long s = (seed >> 11) % nbatch;
        seed = seed * 6364136223846793005ull + 1442695040888963407ull;
        const unsigned long long room = (count > cnt) ? (count - cnt + 1) : 1;
        runs.push_back({s, (seed >> 11) % room, cnt});
    }
    if (!g_s4_oracle_async) s4_oracle_block_ready();
    for (const auto &rr : runs) {
        const unsigned long long s = rr[0], k0 = rr[1], source_k=first+k0, cnt = rr[2];
        const size_t ndig = (size_t)(cnt * S->slot_words), nred = (size_t)(cnt * R.w);
        ++g_oracle.selected;
        for (const auto v : {S->P, S->slot_bits, S->slot_words,
                            (unsigned long long)S->bpw, S->calls, s, source_k, cnt})
            g_oracle.signature = (g_oracle.signature ^ v) * 1099511628211ull;
        bool captured = false;
        if (g_s4_oracle_async) {
            if (g_oracle.tail - g_oracle.head == (unsigned long long)g_s4_oracle_ring) {
                ++g_oracle.ring_waits;
                s4_oracle_one(R, true);     /* never overwrite a pending HOST snapshot */
            }
            S4OracleSlot &slot = g_oracle_slot[g_oracle.tail % g_s4_oracle_ring];
            const double ta0 = now_s();
            auto *dig = pin_words(&slot.digits, &slot.digcap, ndig);
            auto *red = pin_words(&slot.reduced, &slot.redcap, nred);
            g_oracle.t_alloc += now_s() - ta0;
            unsigned long long bytes = 0;
            for (const auto &q : g_oracle_slot) bytes += 8ull * (q.digcap + q.redcap);
            g_oracle.pinned_peak = std::max(g_oracle.pinned_peak, bytes);
            if (dig && red) {
                if (!slot.ready) CK(cudaEventCreateWithFlags(&slot.ready, cudaEventDisableTiming));
                slot.shape = S; slot.slice = s; slot.k0 = source_k; slot.count = cnt;
                slot.out_slots = out_slots;
                const double t0 = now_s();
                CK(cudaMemcpyAsync(dig, ddig + s * n + source_k * S->slot_words,
                                   ndig * 8, cudaMemcpyDeviceToHost));
                CK(cudaMemcpyAsync(red, d_out + (s * count + k0) * R.w,
                                   nred * 8, cudaMemcpyDeviceToHost));
                CK(cudaEventRecord(slot.ready));
                g_oracle.t_copy += now_s() - t0;
                ++g_oracle.tail; ++g_oracle.queued;
                captured = true;
            } else { ++g_oracle.fallbacks; s4_oracle_block_ready(); }
        }
        if (!captured) {
            std::vector<unsigned long long> dig(ndig), red(nred);
            const double t0 = now_s();
            CK(cudaMemcpy(dig.data(), ddig + s * n + source_k * S->slot_words,
                          ndig * 8, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(red.data(), d_out + (s * count + k0) * R.w,
                          nred * 8, cudaMemcpyDeviceToHost));
            g_oracle.t_copy += now_s() - t0;
            s4_compare_snapshot(R, S, s, source_k, cnt, out_slots, dig.data(), red.data());
        }
    }
    g_oracle.t_capture += now_s() - tc0;
}

/* ===================================================================================== *
 *  DEVICE-SIDE OPERAND PACKING for the default (S4) batched multiply -- objective 4, §33
 *
 *  `ntt_pack_operand` (the probe's host packer) writes coefficient i's base-2^bpw digits at the
 *  digit indices  i*slot_words + k,  k = 0 .. ceil(S/bpw)-1:  its bit offset is
 *  i*slot_stride + b with slot_stride a multiple of bpw and b itself a multiple of bpw, so the
 *  packing is a pure digit permutation with no carries across digits.  This kernel does exactly
 *  that, one thread per (slice, coefficient); the destination is pre-zeroed, which is also what
 *  makes the digits between ceil(S/bpw) and slot_words (the window tail the reduction asserts to
 *  be zero) zero.  It is therefore the SAME format the host packer produced, and the S4
 *  reduction's own in-run GMP sample is what checks it.
 *
 *  WHY: measured at B2=1e11, the host packing cost 36.8 us per polynomial multiplication and the
 *  upload of its PACKED result another 20.2 us -- 33% of the entire NTT budget (section 32).  The
 *  raw coefficients are P*W words per slice, about 9x fewer than the packed form at S=5261, so
 *  this replaces both with one small upload and one device pass.
 * ===================================================================================== */
__global__ void s4_pack_batch_kernel(const unsigned long long *src, int S, int bpw,
                                     unsigned long long slot_words, unsigned long long N,
                                     unsigned long long ds, unsigned long long ma, int W,
                                     unsigned long long *dst)
{
    const unsigned long long gid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (gid >= ds * ma) return;
    const unsigned long long s = gid / ma, i = gid - s * ma;
    const unsigned long long *c = src + gid * (unsigned long long)W;  /* slice stride = ma*W */
    unsigned long long *d = dst + s * N + i * slot_words;
    const unsigned long long mask = (bpw >= 64) ? ~0ull : ((1ull << bpw) - 1ull);
    const int ndig = (S + bpw - 1) / bpw;
    for (int k = 0; k < ndig; ++k) {
        const int bit = k * bpw;
        const int w = bit >> 6, sh = bit & 63;
        unsigned long long v = c[w] >> sh;
        if (sh + bpw > 64 && (w + 1) < W) v |= c[w + 1] << (64 - sh);   /* bits >= S stay zero */
        d[k] = v & mask;
    }
}

static void s4_launch_pack_batch(const unsigned long long *src, int S, int bpw,
                                 unsigned long long slot_words, unsigned long long N,
                                 unsigned long long ds, unsigned long long ma, int W,
                                 unsigned long long *dst)
{
    const unsigned long long total = ds * ma;
    if (total == 0) return;
    const unsigned int th = 256;
    const unsigned int bl = (unsigned int)((total + th - 1) / th);
    s4_pack_batch_kernel<<<bl, th>>>(src, S, bpw, slot_words, N, ds, ma, W, dst);
    CK(cudaGetLastError());
}

/* Gather canonical coefficients from a tight resident frontier directly into the
   engine's digit buffers. Metadata is in WORD offsets, never polynomial indices. */
// Explicit owner for non-staging resident arithmetic. No borrowed arena/raw
// pointer can masquerade as an independent lease; each range is bounded below.
struct S4ResidentOwner {
    unsigned long long *data[2]={nullptr,nullptr};
    size_t words[2]={0,0};
    S4ResidentOwner()=default;
    S4ResidentOwner(const S4ResidentOwner&)=delete;
    S4ResidentOwner& operator=(const S4ResidentOwner&)=delete;
    ~S4ResidentOwner(){release();}
    void release(){for(int i=0;i<2;++i)if(data[i]){CK(cudaFree(data[i]));data[i]=nullptr;words[i]=0;}}
    bool owns(const unsigned long long *p,size_t n)const {
        return (p==data[0] && n<=words[0]) || (p==data[1] && n<=words[1]);
    }
};
struct S4DeviceBatch {
    const unsigned long long *src=nullptr,*meta=nullptr;
    unsigned long long *dst=nullptr;
    const unsigned long long *host_meta=nullptr;
    size_t src_words=0,dst_words=0,nbatch=0;
    const S4ResidentOwner *owner=nullptr; // nullptr = the original exclusive raw frontier lease
};
__global__ void s4_pack_gather_kernel(const unsigned long long *src,
    const unsigned long long *offsets, size_t m, size_t nc, int W, int bits,
    int bpw, unsigned long long sw, unsigned long long N, unsigned long long *dst)
{
    const size_t gid=blockIdx.x*(size_t)blockDim.x+threadIdx.x;
    if(gid>=m*nc)return;
    const size_t slice=gid/nc,coef=gid%nc;
    const auto *c=src+offsets[slice]+coef*W;
    auto *d=dst+slice*N+coef*sw;
    const unsigned long long mask=(1ull<<bpw)-1;
    for(int k=0;k<(bits+bpw-1)/bpw;++k) {
        const int bit=k*bpw,w=bit>>6,shift=bit&63;
        unsigned long long v=c[w]>>shift;
        if(shift+bpw>64 && w+1<W)v|=c[w+1]<<(64-shift);
        d[k]=v&mask;
    }
}
struct S4GatherPack {
    const S4DeviceBatch *batch=nullptr;S4Ctx *C=nullptr;size_t s0=0,ma=0,mb=0;
    unsigned long long P=0,N=0,sw=0;int bits=0,bpw=0,W=0;
};
static int s4_gather_final_input(void *ctx,const NttShape &sh,unsigned long long m,
    unsigned long long *a,unsigned long long *b)
{
    const auto &p=*(const S4GatherPack*)ctx;
    if(sh.P!=p.P || sh.N!=p.N || sh.S!=p.bits || sh.bpw!=p.bpw ||
       sh.slot_words!=p.sw || sh.W!=(unsigned long long)p.W)return 3;
    const double start=now_s();
    CK(cudaMemsetAsync(a,0,(size_t)m*sh.N*8));
    CK(cudaMemsetAsync(b,0,(size_t)m*sh.N*8));
    s4_pack_gather_kernel<<<(unsigned int)((m*p.ma+255)/256),256>>>(p.batch->src,
        p.batch->meta+p.s0,m,p.ma,p.W,p.bits,p.bpw,p.sw,p.N,a);
    CK(cudaGetLastError());
    s4_pack_gather_kernel<<<(unsigned int)((m*p.mb+255)/256),256>>>(p.batch->src,
        p.batch->meta+p.batch->nbatch+p.s0,m,p.mb,p.W,p.bits,p.bpw,p.sw,p.N,b);
    CK(cudaGetLastError());
    p.C->t_packdev+=now_s()-start;
    return 0;
}
__global__ void s4_scatter_result_kernel(const unsigned long long *src,
    size_t stride,size_t first,size_t count,size_t m,int W,
    const unsigned long long *offsets,unsigned long long *dst)
{
    const size_t gid=blockIdx.x*(size_t)blockDim.x+threadIdx.x;
    const size_t words=count*(size_t)W;
    if(gid>=m*words)return;
    const size_t slice=gid/words,j=gid%words;
    dst[offsets[slice]+j]=src[(slice*stride+first)*W+j];
}

struct S4InputPack {
    S4Ctx *C = nullptr;
    unsigned long long P = 0, N = 0, slot_words = 0;
    int S = 0, bpw = 0, W = 0;
};

static int s4_pack_final_input(void *ctx, const NttShape &sh, unsigned long long nbatch,
                              unsigned long long *dA, unsigned long long *dB)
{
    const S4InputPack &p = *(const S4InputPack *)ctx;
    /* The query that budgeted/padded raw operands MUST agree with the engine's actual plan. */
    if (sh.P != p.P || sh.S != p.S || sh.N != p.N || sh.bpw != p.bpw ||
        sh.slot_words != p.slot_words || sh.W != (unsigned long long)p.W) {
        std::fprintf(stderr, "%s: S4 input pack shape disagrees with the NTT plan\n", NTT_PROBE_NAME);
        return 3;
    }
    const double t0 = now_s();
    const size_t bytes = (size_t)nbatch * sh.N * sizeof(unsigned long long);
    /* Reused scratch contains spectra/previous outputs. Clear gaps, window tails and unused
       coefficients on EVERY call, then overwrite the occupied digits with the original packer. */
    CK(cudaMemsetAsync(dA, 0, bytes));
    CK(cudaMemsetAsync(dB, 0, bytes));
    s4_launch_pack_batch(p.C->d_rawA, p.S, p.bpw, p.slot_words, p.N, nbatch, p.P, p.W, dA);
    s4_launch_pack_batch(p.C->d_rawB, p.S, p.bpw, p.slot_words, p.N, nbatch, p.P, p.W, dB);
    p.C->t_packdev += now_s() - t0;   /* host enqueue time, never a pipeline drain */
    return 0;
}

/* ===================================================================================== *
 *  SLICE S4 (A) -- THE BATCHED MULTIPLY OF ONE SHAPE, WITH THE DEVICE REDUCTION
 *
 *  poly_mul_modN() multiplies ONE pair and reduces the exact coefficients on the host with
 *  GMP; this multiplies `nbatch` pairs of the SAME shape in one launch per pass and reduces
 *  them on the device (S4Reduce).  It is the ONLY multiply the tree uses when L.s4 is set,
 *  including the single-pair call sites (the fold, the Newton inverse, the descent) -- those
 *  pass nbatch = 1 and still get the device reduction, so the whole engine has one reduction
 *  path and cannot drift from itself.
 *
 *  ma/mb are the operand coefficient counts (deg+1, NOT padded); P = max(ma,mb) is what the
 *  NTT sees, exactly as in poly_mul_modN, and every slice of one call must share the shape
 *  (that is what makes the batch a single NTT shape).
 * ===================================================================================== */
static void poly_mul_batch_modN(PolyLayer &L,
                                const unsigned long long *wa, const unsigned long long *wb,
                                size_t ma, size_t mb, size_t nbatch,
                                std::vector<unsigned long long> &out, int cat = -1,
                                NttMulStats *st_out = nullptr,
                                size_t first = 0, size_t count = (size_t)-1,
                                const S4DeviceBatch *resident = nullptr)
{
    const size_t W = L.W;
    const size_t P = (ma > mb) ? ma : mb;
    const size_t full_nc=ma+mb-1;
    if(count==(size_t)-1) count=full_nc;
    if(first>full_nc || count>full_nc-first) {
        std::fprintf(stderr,"%s: FATAL: invalid polynomial output window\n",NTT_PROBE_NAME);
        std::exit(3);
    }
    const size_t nc=count;
    const bool host_output=!resident || g_s4_carry_trace;
    if(host_output) out.assign(nbatch*nc*W,0ull);else out.clear();
    if (nbatch == 0) return;
    if(resident) {
        if(!resident->src || !resident->dst || resident->src==resident->dst ||
           !resident->meta || !resident->host_meta || resident->nbatch!=nbatch ||
           (!resident->owner && (first!=0 || count!=full_nc)) || g_s4_final_readback || !g_s4_pack_direct) {
            std::fprintf(stderr,"%s: FATAL: invalid resident batch lifetime/shape\n",NTT_PROBE_NAME);std::exit(3);
        }
        for(size_t i=0;i<nbatch;++i) {
            const auto *o=resident->host_meta;
            if(o[i]>resident->src_words || ma*W>resident->src_words-o[i] ||
               o[nbatch+i]>resident->src_words || mb*W>resident->src_words-o[nbatch+i] ||
               o[2*nbatch+i]>resident->dst_words || nc*W>resident->dst_words-o[2*nbatch+i]) {
                std::fprintf(stderr,"%s: FATAL: resident offset outside frontier\n",NTT_PROBE_NAME);std::exit(3);
            }
        }
    }
    S4Ctx &C = *L.s4;
    if(resident && resident->owner &&
       (!resident->owner->owns(resident->src,resident->src_words) ||
        !resident->owner->owns(resident->dst,resident->dst_words))) {
        std::fprintf(stderr,"%s: FATAL: resident range is not held by its owner\n",NTT_PROBE_NAME);std::exit(3);
    }
    if(resident && !resident->owner && ((resident->src!=C.d_rawA && resident->src!=C.d_rawB) ||
                   (resident->dst!=C.d_rawA && resident->dst!=C.d_rawB) ||
                   resident->src_words>C.raw_capacity(resident->src) || resident->dst_words>C.raw_capacity(resident->dst))) {
        std::fprintf(stderr,"%s: FATAL: resident frontier is not owned raw staging\n",NTT_PROBE_NAME);std::exit(3);
    }
    const unsigned long long out_slots = 2 * P - 1; // FULL shape and canonical assertion
    const unsigned long long output_slots=g_s4_output_window ? count : out_slots;
    const unsigned long long output_first=g_s4_output_window ? first : 0;
    const size_t host_first=g_s4_output_window ? 0 : first;
    /* the per-shape reduction (L, the domain constant and the GMP selftest) must exist BEFORE
       the hook fires, and the hook fires inside the multiply -- so the shape is queried here,
       without touching the device.  Never inferred: ntt_shape_query runs the SAME choose_cfg
       the multiply will run. */
    unsigned long long qN = 0, qsb = 0, qsw = 0, qss = 0, qos = 0;
    int qbpw = 0;
    {
        if (!ntt_shape_query(P, (int)L.S, &qN, &qbpw, &qsb, &qsw, &qss, &qos)) {
            std::fprintf(stderr, "%s: the shape P=%llu S=%d is refused by the multiply\n",
                         NTT_PROBE_NAME, (unsigned long long)P, (int)L.S);
            std::exit(3);
        }
        if (qos != out_slots || qss != qsw * (unsigned long long)qbpw) {
            std::fprintf(stderr, "%s: FATAL: the queried shape disagrees with the multiply's\n",
                         NTT_PROBE_NAME);
            std::exit(3);
        }
        if (!C.red->find(qsb, qsw, qbpw)) {
            S4Reduce::Shape *S0 = s4_shape_init(*C.red, (unsigned long long)P, qsb, qss, qsw,
                                                qbpw);
            if (s4_reduce_selftest(*C.red, S0)) {
                std::fprintf(stderr, "%s: FATAL: the device reduction disagrees with GMP on the "
                                     "actual shape -- refusing to continue\n", NTT_PROBE_NAME);
                std::exit(3);
            }
            C.selftested = true;
        }
    }
    const size_t need = nbatch * output_slots * W;
    NttReduceHook hook;
    hook.ctx = C.red;
    hook.run = s4_reduce_hook;
    hook.w = (unsigned long long)W;
    hook.first=output_first; hook.count=output_slots;

    std::vector<unsigned long long> slots;
    NttMulStats st{};
    const double t0 = now_s();
    /* CHUNKING (slice S4, learned from the real shape): a level's whole batch needs
       nbatch*N words, and nbatch*N is roughly CONSTANT per level (~P*slot_words), so material-
       ising every level at once asked the arena for ~600 MB per level and 17 levels blew the
       cap -- after which the per-call fallback could not allocate and the run died with
       "out of memory".  A per-chunk device budget keeps every (N, chunk) buffer small; the
       arena caches one entry per (N, chunk) because the chunk size is a deterministic function
       of N.  Bigger budget = fewer launches; smaller = less memory. */
    /* ---- NTT_S4_BATCH_MB: THE PER-CHUNK DEVICE BUDGET (section 27) --------------------------
       This was a hard-coded 32 MB.  The per-level timers showed a ~0.3-0.4 s FIXED cost per tree
       level that does not depend on the multiply count at all (3600 tiny multiplies cost 0.375 s,
       and ONE multiply of the full degree costs 0.618 s), and it lands inside
       poly_mul_batch_modN -- whose own sub-timers the dev path never fills.  The chunking below is
       the first suspect: a level's batch is split into ceil(nbatch/chunk) chunk round-trips, each
       with its own H2D, pack, NTT, reduce and D2H.  This knob makes that testable. */
    /* The legacy controls have not gained the direct packer's VRAM headroom. */
    static const bool host_pack = [] {
        const char *e = std::getenv("NTT_S4_HOSTPACK");
        return e && *e && std::atoi(e) != 0;
    }();
    static const unsigned long long s4_batch_mb = [] {
        const char *e = std::getenv("NTT_S4_BATCH_MB");
        return (e && *e) ? std::strtoull(e, nullptr, 10) : 0ull;
    }();
    static const unsigned long long default_mb =
        (!host_pack && g_s4_pack_direct) ? g_s4_batch_budget_mb : 32ull;
    const unsigned long long effective_mb = s4_batch_mb ? s4_batch_mb : default_mb;
    const unsigned long long budget_bytes = effective_mb << 20;
    static bool budget_reported = false;
    if (!budget_reported) {
        stage2_log::print(stage2_log::debug, "s4_batch_budget: mb=%llu env_override=%d direct=%d host_pack=%d\n",
                    effective_mb, (int)(s4_batch_mb != 0), (int)g_s4_pack_direct, (int)host_pack);
        budget_reported = true;
    }
    unsigned long long chunk = 1;
    {
        unsigned long long qN2 = 0;
        int qbpw2 = 0;
        if (!ntt_shape_query(P, (int)L.S, &qN2, &qbpw2, nullptr, nullptr, nullptr, nullptr))
            qN2 = 1;
        for (unsigned long long c = nbatch; c >= 1; c /= 2) {
            const unsigned long long bytes =
                (3 * qN2 * c + out_slots * c) * (unsigned long long)sizeof(unsigned long long);
            if (bytes <= budget_bytes) { chunk = c; break; }
        }
    }
    if (g_s4_chunk_max) chunk = std::min(chunk, g_s4_chunk_max);
    const bool chunk_output=g_s4_chunk_output && !g_s4_final_readback;
    const size_t allocation_need=std::max((size_t)1,
        (size_t)(chunk_output ? chunk : nbatch)*output_slots*W);
    if(allocation_need>C.d_out_cap) {
        if(C.d_out) { CK(cudaFree(C.d_out)); C.d_out=nullptr; C.d_out_cap=0; }
        CK(cudaMalloc(&C.d_out,allocation_need*sizeof(unsigned long long)));
        C.d_out_cap=allocation_need;
        ++g_chunk_output.grows;
    }
    hook.out=C.d_out;
    ++g_chunk_output.calls;
    if(chunk_output) {
        ++g_chunk_output.reused_calls;
        g_chunk_output.reused_chunks+=(nbatch+chunk-1)/chunk;
        if(chunk<nbatch) ++g_chunk_output.multi_chunk_reused;
    } else ++g_chunk_output.whole_calls;
    if(g_s4_chunk_output && g_s4_final_readback) ++g_chunk_output.legacy_calls;
    g_chunk_output.request_peak_bytes=std::max(g_chunk_output.request_peak_bytes,8ull*allocation_need);
    g_chunk_output.whole_peak_bytes=std::max(g_chunk_output.whole_peak_bytes,8ull*std::max((size_t)1,need));
    g_chunk_output.retained_peak_bytes=std::max(g_chunk_output.retained_peak_bytes,8ull*C.d_out_cap);
    int rc = 0;
    /* THE DEVICE PACKING SWITCH (objective 4, section 33): default is the device packer; set
       NTT_S4_HOSTPACK=1 to run the old host-packing path, which is kept as the A/B oracle. */
    /* ---- THE DEFERRED CARRY CHECK (section 29) ----------------------------------------------
       Every chunk ends with the probe's carry check, which is a pageable D2H of 2*nbatch words and
       therefore waits for the device to finish everything queued before it: a full pipeline drain
       per chunk.  Measured at the production shape that is 22471 x 1.64 ms = 36.75 s of a 253.53 s
       run (`carrysplt d2h`), the largest host-side item on the books, and it is what makes the
       chunk size matter so much (fewer chunks = fewer drains, section 27).

       Deferring it needs the counters to survive between chunks, which requires the arena (the
       buffer has to outlive the call), so it is on whenever an arena exists and the device packer
       is in use.  The accumulation lives in the arena entry of the chunk's shape, which is why the
       readback happens BEFORE a non-deferred chunk resets the counters. Identical deferred chunks
       skip that reset and can accumulate their atomicAdd/atomicMax counters in place. The original
       per-interior finish remains the CARRY_BATCH=0 control. Interior chunks lose their fwd/inv
       event attribution (the events are
       destroyed unread, since reading them would need the drain we are removing) -- the last chunk
       of each call still reports the sample. */
    const bool defer_ok = (!host_pack && L.arena != nullptr && g_s4_defer_carry);
    /* the doubly-buffered PINNED output staging (section 30): reserved once per call at the largest
       chunk this call can use, so every chunk of the call follows the same path (mixing a blocking
       chunk with the deferred-consumption scheme would break the pending bookkeeping) */
    const size_t out_words_max = (size_t)(chunk * output_slots * W);
    if (!g_pin_ev[0]) CK(cudaEventCreateWithFlags(&g_pin_ev[0], cudaEventDisableTiming));
    if (!g_pin_ev[1]) CK(cudaEventCreateWithFlags(&g_pin_ev[1], cudaEventDisableTiming));
    const bool async_out = (host_output && g_s4_async &&
                            pin_words(&g_pin_out[0], &g_pin_out_cap[0], out_words_max) != nullptr &&
                            pin_words(&g_pin_out[1], &g_pin_out_cap[1], out_words_max) != nullptr);
    unsigned long long ci = 0, pend_m = 0, pend_s0 = 0;
    bool out_pending = false;
    bool carry_pending = false;
    unsigned long long carry_pending_m = 0, carry_pending_chunks = 0;
    double carry_d2h_acc = 0.0, chunk_d2h_acc = 0.0;
    unsigned long long carry_res_acc = 0, carry_bits_acc = 0;
    auto finish_carry = [&] {
        NttMulStats fin{};
        const int rf = ntt_batch_carry_finish(L.arena, qN, carry_pending_m, &fin);
        carry_d2h_acc += fin.t_check_d2h;
        g_carry_group_readback += fin.t_check_d2h;
        carry_res_acc += fin.carry_residual;
        carry_bits_acc = std::max(carry_bits_acc, fin.carry_max_bits);
        ++g_defer_finishes;
        if (rf != 0) {
            std::fprintf(stderr, "%s: the deferred carry check of the chunked multiply failed "
                                 "(rc=%d) at P=%llu nbatch=%llu chunk=%llu accumulated=%llu\n",
                         NTT_PROBE_NAME, rf, (unsigned long long)P,
                         (unsigned long long)nbatch, carry_pending_m, carry_pending_chunks);
            std::exit(3);
        }
        g_defer_checked_chunks += carry_pending_chunks;
        g_defer_max_group = std::max(g_defer_max_group, carry_pending_chunks);
        carry_pending = false;
        carry_pending_chunks = 0;
    };
    for (unsigned long long s0 = 0; s0 < nbatch; s0 += chunk) {
        const unsigned long long m = ((nbatch - s0) < chunk) ? (nbatch - s0) : chunk;
        const bool last_chunk = (s0 + m >= nbatch);
        /* s0 != 0 is NOT optional: the first chunk still has to run the probe's non-deferred
           path because that is what memsets the residual counters.  Skipping it would leave the
           PREVIOUS call's counters in the arena entry, and since a successful check leaves zeros
           the result would not be a false alarm but a silently VACUOUS check for those slices. */
        const bool defer_this = defer_ok && !last_chunk && (s0 != 0);
        /* SAME (N,m), no reset: atomic counters retain every interior chunk's verdict. Finish
           before a last/short/non-deferred chunk, or at function exit, never after its memset. */
        if (carry_pending && (!g_s4_carry_batch || !defer_this || carry_pending_m != m))
            finish_carry();
        NttReduceHook h2 = hook;
        if (hook.out) h2.out = hook.out + (chunk_output ? 0 : (size_t)(s0 * output_slots) * W);
        int r1 = 0;
        if(resident) {
            if(host_pack) {std::fprintf(stderr,"%s: FATAL: resident multiply cannot use host packing\n",NTT_PROBE_NAME);std::exit(3);}
            S4GatherPack pack{resident,&C,(size_t)s0,ma,mb,(unsigned long long)P,qN,qsw,(int)L.S,qbpw,(int)W};
            NttInputHook input{&pack,s4_gather_final_input};
            const double tp=C.t_packdev;
            r1=ntt_poly_mul_batch_dev(P,(int)L.S,L.device,m,nullptr,nullptr,&st,L.arena,&h2,
                nullptr,qbpw,defer_this,&input);
            // Input pack remains a direct engine write; no temporary packed inputs or D2D.
            if(!r1) {++C.input_direct;C.input_avoided_bytes+=16ull*m*qN;}
            C.packed_peak_bytes=std::max(C.packed_peak_bytes,16ull*m*qN);
            C.pack_launches+=2;
            C.t_h2d_raw+=C.t_packdev-tp;
            if(!r1 && st.carry_deferred) {
                carry_pending=true;carry_pending_m=m;++carry_pending_chunks;
                ++g_defer_chunks;g_defer_slices+=m;
                if(g_s4_carry_test_bad && !g_s4_carry_injected) {
                    CK(cudaMemsetAsync(L.arena->cur.dRes,1,sizeof(unsigned long long)));
                    g_s4_carry_injected=true;
                    stage2_log::print(stage2_log::debug, "s4_carry_fault: first resident interior diagnostic poisoned P=%llu m=%llu\n",
                        (unsigned long long)P,m);
                }
            }
        } else if (host_pack) {
            r1 = ntt_poly_mul_batch_host(P, (int)L.S, L.device, m,
                                         wa + s0 * P * W, wb + s0 * P * W,
                                         (s0 == 0) ? &slots : nullptr, &st, L.arena, &h2,
                                         nullptr);
        } else {
            /* 1. the RAW coefficients to the device (P*W words per slice per operand) ... */
            const size_t raw_words = (size_t)m * P * W;
            C.raw_reserve(raw_words,raw_words);
            const double th0 = now_s();
            /* ASYNC UPLOAD VIA PINNED STAGING (section 30): the host memcpy into pinned memory
               touches no device state, so it cannot drain the pipeline, and the pinned
               cudaMemcpyAsync needs no implicit sync either.  Without pinned memory the old
               blocking pair runs unchanged. */
            const size_t raw_bytes = raw_words * sizeof(unsigned long long);
            const size_t upload_slot = g_s4_carry_batch ? (size_t)(ci & 1) : 0;
            if (g_s4_async && g_pin_raw_pending[upload_slot]) {
                const cudaError_t ready = cudaEventQuery(g_pin_raw_ev[upload_slot]);
                if (ready == cudaErrorNotReady) {
                    const double tw0 = now_s();
                    CK(cudaEventSynchronize(g_pin_raw_ev[upload_slot]));
                    g_pin_raw_wait += now_s() - tw0;
                    ++g_pin_raw_waits;
                } else CK(ready);
                g_pin_raw_pending[upload_slot] = false;
            }
            unsigned long long *pa = g_s4_async ? pin_words(&g_pin_raw[upload_slot][0], &g_pin_raw_cap[upload_slot][0], raw_words)
                                                : nullptr;
            unsigned long long *pb = g_s4_async ? pin_words(&g_pin_raw[upload_slot][1], &g_pin_raw_cap[upload_slot][1], raw_words)
                                                : nullptr;
            if (pa && pb) {
                std::memcpy(pa, wa + s0 * P * W, raw_bytes);
                std::memcpy(pb, wb + s0 * P * W, raw_bytes);
                CK(cudaMemcpyAsync(C.d_rawA, pa, raw_bytes, cudaMemcpyHostToDevice));
                CK(cudaMemcpyAsync(C.d_rawB, pb, raw_bytes, cudaMemcpyHostToDevice));
                if (!g_pin_raw_ev[upload_slot])
                    CK(cudaEventCreateWithFlags(&g_pin_raw_ev[upload_slot], cudaEventDisableTiming));
                CK(cudaEventRecord(g_pin_raw_ev[upload_slot]));
                g_pin_raw_pending[upload_slot] = true;
                ++g_pin_raw_used;
            } else {
                CK(cudaMemcpy(C.d_rawA, wa + s0 * P * W, raw_bytes, cudaMemcpyHostToDevice));
                CK(cudaMemcpy(C.d_rawB, wb + s0 * P * W, raw_bytes, cudaMemcpyHostToDevice));
                if (g_s4_async) ++g_pin_fallbacks;
            }
            /* 2. ... packed into digits ON the device (both operands, one launch each) ... */
            const size_t pack_words = (size_t)m * (size_t)qN;
            if (!g_s4_pack_direct && pack_words > C.d_pack_cap) {
                if (C.d_packA) { cudaFree(C.d_packA); cudaFree(C.d_packB); C.d_packA = C.d_packB = nullptr; }
                CK(cudaMalloc(&C.d_packA, pack_words * sizeof(unsigned long long)));
                CK(cudaMalloc(&C.d_packB, pack_words * sizeof(unsigned long long)));
                C.d_pack_cap = pack_words;
                C.temp_peak_bytes = std::max(C.temp_peak_bytes, 16ull * C.d_pack_cap);
            }
            C.packed_peak_bytes = std::max(C.packed_peak_bytes, 16ull * pack_words);
            S4InputPack pack{&C, (unsigned long long)P, qN, qsw, (int)L.S, qbpw, (int)W};
            NttInputHook input{&pack, s4_pack_final_input};
            const double tp0 = C.t_packdev;
            if (!g_s4_pack_direct) {
                /* Control: identical digits go through the retained temporary buffers + D2D. */
                CK(cudaMemset(C.d_packA, 0, pack_words * sizeof(unsigned long long)));
                CK(cudaMemset(C.d_packB, 0, pack_words * sizeof(unsigned long long)));
                const double tpack0 = now_s();
                s4_launch_pack_batch(C.d_rawA, (int)L.S, qbpw, qsw, qN, m, P, (int)W, C.d_packA);
                s4_launch_pack_batch(C.d_rawB, (int)L.S, qbpw, qsw, qN, m, P, (int)W, C.d_packB);
                C.t_packdev += now_s() - tpack0;
            }
            C.t_h2d_raw += now_s() - th0;
            C.raw_words += (unsigned long long)raw_words * 2;
            C.pack_launches += 2;
            /* 3. ... and the multiply itself, device-to-device (the same passes, the same carry,
               the same exactness assertions, the same reduction hook) */
            NttMulStats nst{};
            r1 = ntt_poly_mul_batch_dev(P, (int)L.S, L.device, m, C.d_packA, C.d_packB, &nst,
                                        L.arena, &h2, nullptr, qbpw, defer_this,
                                        g_s4_pack_direct ? &input : nullptr);
            /* Keep rawupload's historical upload+packing host total comparable. The direct
               pack is queued INSIDE the engine, so add just its own enqueue time here. */
            if (g_s4_pack_direct) C.t_h2d_raw += C.t_packdev - tp0;
            if (r1 == 0) {
                if (g_s4_pack_direct) { ++C.input_direct; C.input_avoided_bytes += 16ull * pack_words; }
                else { ++C.input_copied; C.input_d2d_bytes += 16ull * pack_words; }
                C.t_input_copy_host += nst.t_opcopy;
            }
            st = nst;                     /* the caller's stats are the dev path's */
            if (r1 == 0 && nst.carry_deferred) {
                carry_pending = true; carry_pending_m = m; ++carry_pending_chunks;
                ++g_defer_chunks; g_defer_slices += m;
                if (g_s4_carry_test_bad && !g_s4_carry_injected) {
                    /* Corrupt ONLY the diagnostic counter after the first interior chunk.
                       Later good chunks and the tail must never hide this error. */
                    CK(cudaMemsetAsync(L.arena->cur.dRes, 1, sizeof(unsigned long long)));
                    g_s4_carry_injected = true;
                    stage2_log::print(stage2_log::debug, "s4_carry_fault: first interior diagnostic poisoned P=%llu m=%llu\n",
                                (unsigned long long)P, m);
                }
            }
            if (s0 == 0) slots.clear();   /* the host path filled this; the dev path does not */
        }
        if (r1 != 0) { rc = r1; break; }
        /* Readbacks of ALL non-deferred chunks, not only the final overwritten `st`.
           Interior deferred chunks contribute zero here and are charged by finish_carry. */
        chunk_d2h_acc += st.t_check_d2h;
        g_carry_chunk_readback += st.t_check_d2h;
        carry_bits_acc = std::max(carry_bits_acc, st.carry_max_bits);
        /* the reduced coefficients of this chunk, back to the host -- AND THIS TRANSFER IS THE
           POINT OF OBJECTIVE 4 (section 32): it is `m * output_slots * W` words, i.e. the WHOLE
           product of every slice at full slot width, and it is neither inside the probe's timers
           (they belong to the host implementation) nor inside the tree's own phases.  It is
           therefore timed and counted HERE, so "the 111 us per call that nobody measured" can be
           attributed instead of guessed.

           SECTION 30: it is now an ASYNC copy into pinned staging, and the chunk that is copied
           here is consumed on the NEXT iteration -- so the host never waits for the device unless
           the device is genuinely behind, which is the difference between "the transfer is
           overlapped" and "the pipeline is drained per chunk".  `out_pending` marks the one chunk
           that has been copied but not yet written into `out`; the end of the call drains it. */
        const size_t out_words = (size_t)(m * output_slots * W);
        if(resident) {
            s4_scatter_result_kernel<<<(unsigned int)((m*nc*W+255)/256),256>>>(h2.out,
                output_slots,host_first,nc,m,(int)W,resident->meta+2*nbatch+s0,resident->dst);
            CK(cudaGetLastError());
            if(!host_output) {if(!resident->owner)g_gdevice.resident_words+=out_words;continue;}
            if(!resident->owner)g_gdevice.trace_words+=out_words;
        }
        if (!async_out) {
            /* NO PINNED MEMORY: the original blocking readback, so a failed pinning costs speed
               and never correctness (and never a half-filled `out`) */
            const double td0 = now_s();
            std::vector<unsigned long long> all(out_words, 0ull);
            if(out_words) CK(cudaMemcpy(all.data(), h2.out, out_words * sizeof(unsigned long long),
                                       cudaMemcpyDeviceToHost));
            for (unsigned long long s = 0; s < m; ++s)
                std::copy(all.begin() + (long)((s * output_slots + host_first) * W),
                          all.begin() + (long)((s * output_slots + host_first + nc) * W),
                          out.begin() + (long)((s0 + s) * nc * W));
            L.t_d2h_coeff += now_s() - td0;
            L.d2h_coeff_words += (unsigned long long)out_words;
            continue;
        }
        unsigned long long *po = g_pin_out[(size_t)(ci & 1)];
        const double td0 = now_s();
        if(out_words) CK(cudaMemcpyAsync(po, h2.out, out_words * sizeof(unsigned long long),
                                        cudaMemcpyDeviceToHost));
        CK(cudaEventRecord(g_pin_ev[ci & 1]));
        ++g_pin_out_used;
        L.t_d2h_coeff += now_s() - td0;
        L.d2h_coeff_words += (unsigned long long)out_words;
        /* NOW consume the previous chunk's buffer: its copy has had this whole chunk's work to
           finish, so this wait is normally already satisfied */
        if (out_pending) {
            const double tw0 = now_s();
            CK(cudaEventSynchronize(g_pin_ev[(ci - 1) & 1]));
            const unsigned long long *prev = g_pin_out[(ci - 1) & 1];
            for (unsigned long long s = 0; s < pend_m; ++s)
                std::copy(prev + ((size_t)s * output_slots + host_first) * W,
                          prev + ((size_t)s * output_slots + host_first) * W + nc * W,
                          out.begin() + (long)((pend_s0 + s) * nc * W));
            L.t_d2h_coeff += now_s() - tw0;
        }
        pend_m = m;
        pend_s0 = s0;
        out_pending = true;
        ++ci;
    }
    if (carry_pending) finish_carry();
    /* THE ONE DRAIN OF THE COEFFICIENT READBACK: the last chunk copied but not yet consumed */
    if (out_pending) {
        const double tw0 = now_s();
        CK(cudaEventSynchronize(g_pin_ev[(ci - 1) & 1]));
        const unsigned long long *prev = g_pin_out[(ci - 1) & 1];
        for (unsigned long long s = 0; s < pend_m; ++s)
            std::copy(prev + ((size_t)s * output_slots + host_first) * W,
                      prev + ((size_t)s * output_slots + host_first) * W + nc * W,
                      out.begin() + (long)((pend_s0 + s) * nc * W));
        L.t_d2h_coeff += now_s() - tw0;
        out_pending = false;
    }
    L.ntt_seconds += now_s() - t0;
    /* the deferred chunks' carry verdict, folded back into the caller's account: their counters
       were read ONCE by ntt_batch_carry_finish instead of once per chunk (section 29), so this is
       the only place their time and residual totals can be reported */
    /* Charge through `st` ONCE below. Directly adding to L here as well double-counted every
       finish; keeping only the last chunk also omitted the first non-deferred readback. */
    st.t_check_d2h = chunk_d2h_acc + carry_d2h_acc;
    st.carry_residual += carry_res_acc;
    st.carry_max_bits = std::max(st.carry_max_bits, carry_bits_acc);
    /* the caller's copy of the per-call account (section 27): the tree needs it PER LEVEL to say
       whether its floor is host packing, H2D/D2H traffic, the transforms, the carry check or the
       exact coefficient extraction -- `t0` above already covers the whole call including the
       host-side copies, so `total` = now - t0 minus the sum of the parts is the unattributed rest */
    if (st_out) {
        *st_out = st;
        st_out->t_total_call = now_s() - t0;
    }
    if (rc != 0) {
        std::fprintf(stderr, "%s: ntt_poly_mul_batch_host failed (rc=%d) at P=%llu S=%d "
                             "nbatch=%llu\n", NTT_PROBE_NAME, rc, (unsigned long long)P,
                     (int)L.S, (unsigned long long)nbatch);
        std::exit(3);
    }
    ++L.ntt_launches;
    L.ntt_calls += nbatch;
    L.muls += nbatch;
    L.t_fwd += st.t_fwd;
    L.t_inv += st.t_inv;
    L.t_slot += st.t_slot;
    L.t_hpack += st.t_hpack;
    L.t_scan += st.t_scan;
    L.t_h2d_batch += st.t_h2d_batch;
    L.t_check += st.t_check;
    L.t_check_reset += st.t_check_reset;
    L.t_check_kernel += st.t_check_kernel;
    L.t_check_d2h += st.t_check_d2h;
    L.t_plan += st.t_plan;
    L.t_opcopy += st.t_opcopy;
    L.t_hout += st.t_hout;
    L.max_ntt_words = std::max(L.max_ntt_words, st.N);
    L.max_ntt_coeffs = std::max(L.max_ntt_coeffs, (unsigned long long)P);
    L.max_slot_bits = std::max(L.max_slot_bits, st.slot_bits);
    /* the tree's OWN exactness bound, re-derived for THIS shape from the values that came
       back -- never inherited from the probe's shapes nor from another tree level */
    if (!exact_ok_terms(st.L_terms, st.bpw)) {
        std::fprintf(stderr, "%s: EXACTNESS VIOLATED for the tree's shape: L=%llu bpw=%d\n",
                     NTT_PROBE_NAME, (unsigned long long)st.L_terms, st.bpw);
        std::exit(3);
    }
    {
        const double bb = coeff_bound_bits_terms(st.L_terms, st.bpw);
        if (bb > L.bind_bound_bits) {
            L.bind_bound_bits = bb;
            L.bind_P = P;
            L.bind_L = st.L_terms;
            L.bind_slot_bits = st.slot_bits;
            L.bind_slot_words = st.slot_words;
            L.bind_bpw = st.bpw;
        }
    }
    const int c = (cat >= 0) ? cat : L.cat;
    if (c >= 0)
        for (size_t s = 0; s < nbatch; ++s) L.cost.add(c, ma, mb);
    L.slot_checks += nbatch * nc;
    /* Shape lookup remains checked even when the redundant readback is disabled. */
    S4Reduce::Shape *S = C.red->find(st.slot_bits, st.slot_words, st.bpw);
    if (!S) {
        std::fprintf(stderr, "%s: FATAL: the reduction shape changed under the multiply\n",
                     NTT_PROBE_NAME);
        std::exit(3);
    }
    ++g_output_window.calls;
    g_output_window.source_coeffs+=nbatch*out_slots;
    g_output_window.reduced_coeffs+=nbatch*output_slots;
    g_output_window.returned_coeffs+=nbatch*count;
    g_output_window.skipped_coeffs+=nbatch*(out_slots-output_slots);
    if(host_output) g_output_window.d2h_words+=nbatch*output_slots*W;
    g_output_window.device_peak_bytes=std::max(g_output_window.device_peak_bytes,8ull*C.d_out_cap);
    g_output_window.pinned_peak_bytes=std::max(g_output_window.pinned_peak_bytes,8ull*(g_pin_out_cap[0]+g_pin_out_cap[1]));
    ++g_final_readback.calls;
    if (g_s4_final_readback) {
        const double tfinal=now_s();
        std::vector<unsigned long long> all(need, 0ull);
        if(need) CK(cudaMemcpy(all.data(), C.d_out, need * sizeof(unsigned long long),
                               cudaMemcpyDeviceToHost));
        for (size_t s = 0; s < nbatch; ++s)
            std::copy(all.begin() + (long)((s * output_slots + host_first) * W),
                      all.begin() + (long)((s * output_slots + host_first + nc) * W),
                      out.begin() + (long)(s * nc * W));
        const double elapsed=now_s()-tfinal;
        g_final_readback.copied_words+=need;
        g_final_readback.host_peak_bytes=std::max(g_final_readback.host_peak_bytes,8ull*need);
        g_final_readback.t_copy+=elapsed;
        /* Historical ntt_seconds stopped BEFORE this copy. Include it in the control's call
           time, while keeping coefficient D2H's existing chunk-only ledger separately labelled. */
        L.ntt_seconds+=elapsed;
    } else {
        g_final_readback.avoided_words+=need;
    }
    if (st_out) st_out->t_total_call=now_s()-t0;
    /* Fingerprint the actual FINAL output, after either path; the old trace preceded the
       whole-call overwrite and could not detect a fault introduced by that last copy. */
    if (g_s4_carry_trace) {
        for (const auto v : {(unsigned long long)ma, (unsigned long long)mb,
                            (unsigned long long)nbatch, (unsigned long long)out.size()})
            g_carry_output_hash = (g_carry_output_hash ^ v) * 1099511628211ull;
        for (const auto v : out)
            g_carry_output_hash = (g_carry_output_hash ^ v) * 1099511628211ull;
        g_carry_output_words += (unsigned long long)out.size();
    }
    /* the in-run oracle (the device's coefficients against GMP) runs INSIDE the hook, where
       the digit buffer is still alive */
    ++C.launches;
    C.muls += nbatch;
}

/* the batched product tree: a power-of-two padded heap exactly as before (t[1] = root,
   t[pad+i] = leaf i, padding leaves are the constant 1, and a node with a constant-1 child IS
   the other child -- the CPU reference's build_product_tree + poly_is_one shortcut), but built
   LEVEL BY LEVEL so that every multiply of a level shares one batched launch: the number of
   launches per tree drops from (leaves-1) to (levels x distinct shapes). */
static std::vector<std::vector<unsigned long long>> build_tree_flat(
    PolyLayer &L, const std::vector<std::vector<unsigned long long>> &leaf,
    std::vector<size_t> &deg, size_t &pad_out, FTreeStats &fs, int cat = -1,
    bool keep_children = true)
{
    const size_t W = L.W;
    const size_t n = leaf.size();
    size_t pad = 1;
    while (pad < n) pad *= 2;
    fs.leaves = n;
    fs.padded = pad;
    pad_out = pad;
    /* Root-only callers never read a consumed child again. Internal placeholders need no
       payload; leaves, including constant-one padding, are defined before the first level. */
    std::vector<std::vector<unsigned long long>> t = keep_children
        ? std::vector<std::vector<unsigned long long>>(2 * pad, std::vector<unsigned long long>(W, 0ull))
        : std::vector<std::vector<unsigned long long>>(2 * pad);
    deg.assign(2 * pad, 0);
    for (size_t i = keep_children ? 0 : pad; i < 2 * pad; ++i) {
        if (!keep_children) t[i].assign(W, 0ull);
        t[i][0] = 1;                                        /* the constant 1 */
    }
    for (size_t i = 0; i < n; ++i) { t[pad + i] = leaf[i]; deg[pad + i] = 1; }
    size_t live_words = 0;
    for (const auto &v : t) live_words += v.capacity();
    fs.node_peak_bytes = 8ull * live_words;
    auto update_parent = [&](size_t i, size_t old_cap) {
        live_words -= old_cap; live_words += t[i].capacity();
        fs.node_peak_bytes = std::max(fs.node_peak_bytes, 8ull * live_words);
    };
    auto passthrough = [&](size_t i, size_t child) {
        const size_t old = t[i].capacity(), source = t[child].capacity();
        if (keep_children) { t[i] = t[child]; update_parent(i, old); }
        else {
            t[i] = std::move(t[child]);
            live_words -= old + source;
            live_words += t[i].capacity() + t[child].capacity();
            fs.node_peak_bytes = std::max(fs.node_peak_bytes, 8ull * live_words);
            ++fs.passthrough_moves;
        }
        deg[i] = deg[child];
    };
    auto release_children = [&](size_t first, size_t last) {
        if (keep_children) return;
        const double tr0 = now_s();
        for (size_t i = first; i < last; ++i) {
            const size_t old = t[i].capacity();
            std::vector<unsigned long long>().swap(t[i]);
            live_words -= old; fs.node_released_bytes += 8ull * old;
            ++fs.nodes_released;
        }
        fs.t_release += now_s() - tr0;
    };
    auto finish = [&] {
        fs.node_retained_bytes = 8ull * live_words;
        if (g_s4_carry_trace) {
            size_t actual = 0;
            for (size_t i=0; i<t.size(); ++i) {
                actual += t[i].capacity();
                if (!keep_children && i>1 && !t[i].empty()) {
                    std::fprintf(stderr, "%s: FATAL: root-only tree retains child %llu\n", NTT_PROBE_NAME,
                                 (unsigned long long)i); std::exit(3);
                }
            }
            if (actual != live_words) {
                std::fprintf(stderr, "%s: FATAL: tree capacity accounting mismatch\n", NTT_PROBE_NAME);
                std::exit(3);
            }
        }
    };
    /* A one-leaf tree has no parent level (base=0 would never reach the loop's base=1 stop). */
    if (pad == 1) { finish(); return t; }
    if (!L.s4) {                                               /* the pre-S4 path, unchanged */
        for (size_t i = pad; i-- > 1; ) {
            if (deg[2*i] == 0 && poly_is_one(t[2 * i].data(), W)) { passthrough(i, 2*i+1); }
            else if (deg[2*i+1] == 0 && poly_is_one(t[2 * i + 1].data(), W)) { passthrough(i, 2*i); }
            else {
                const size_t old = t[i].capacity();
                t[i] = poly_mul_modN(L, t[2 * i], deg[2 * i], t[2 * i + 1], deg[2 * i + 1], cat);
                update_parent(i, old);
                deg[i] = deg[2 * i] + deg[2 * i + 1];
                ++fs.muls;
            }
            release_children(2*i, 2*i+2);
        }
        finish();
        return t;
    }
    ++L.s4->level_calls;
    /* ---- PER-LEVEL TIMES OF A TREE (section 26) ---------------------------------------------
       The cost model says every level of a product tree carries the SAME operand bits (P*slot_bits
       per level, whatever the degree), and the measured per-phase rates (g_tree 0.32 ns per operand
       bit, fold 0.38, descent 0.33) say the phases are uniformly efficient -- which would make the
       G tree's 101 s inherent rather than wasteful.  That is a big claim to leave on arithmetic, so
       each level is timed and the first two builds (the F tree, then the first G tree) print their
       ladder: if the levels are equal, the tree has no cheap half to attack. */
    static int lvl_print = 0;
    const bool lvl_trace = (lvl_print < 2);
    if (lvl_trace) ++lvl_print;
    for (size_t base = pad / 2; ; base /= 2) {
        const double tl0 = now_s();
        unsigned long long lvl_muls = 0, lvl_groups = 0;
        double lvl_fwd = 0, lvl_inv = 0, lvl_slot = 0, lvl_hpack = 0, lvl_h2d = 0, lvl_check = 0;
        double lvl_plan = 0, lvl_opcopy = 0, lvl_hout = 0, lvl_total = 0;
        /* nodes [base, 2*base), children at [2*base, 4*base) -- all children are already built */
        std::map<std::pair<size_t, size_t>, std::vector<size_t>> groups;
        for (size_t i = base; i < 2 * base; ++i) {
            const size_t c0 = 2 * i, c1 = 2 * i + 1;
            if (deg[c0] == 0 && poly_is_one(t[c0].data(), W)) { passthrough(i, c1); }
            else if (deg[c1] == 0 && poly_is_one(t[c1].data(), W)) { passthrough(i, c0); }
            else {
                const size_t ma = deg[c0] + 1, mb = deg[c1] + 1;
                groups[std::make_pair((ma < mb) ? ma : mb, (ma < mb) ? mb : ma)].push_back(i);
            }
        }
        for (auto &g : groups) {
            const size_t ma = g.first.first, mb = g.first.second;
            const size_t P = (ma > mb) ? ma : mb;
            const size_t nbatch = g.second.size();
            const size_t nc = ma + mb - 1;
            std::vector<unsigned long long> wa(nbatch * P * W, 0ull), wb(nbatch * P * W, 0ull);
            for (size_t s = 0; s < nbatch; ++s) {
                const size_t i = g.second[s];
                /* the operand lengths are the CHILDREN's own (the group key is normalised to
                   (min,max) for the shape, so it must NOT be used to size the copies) */
                const size_t na = deg[2 * i] + 1, nb2 = deg[2 * i + 1] + 1;
                std::copy(t[2 * i].begin(), t[2 * i].begin() + (long)(na * W),
                          wa.begin() + (long)(s * P * W));
                std::copy(t[2 * i + 1].begin(), t[2 * i + 1].begin() + (long)(nb2 * W),
                          wb.begin() + (long)(s * P * W));
            }
            std::vector<unsigned long long> res;
            NttMulStats gst{};
            poly_mul_batch_modN(L, wa.data(), wb.data(), ma, mb, nbatch, res, cat, &gst);
            ++L.s4->groups;
            ++lvl_groups;
            lvl_muls += nbatch;
            if (lvl_trace) {
                lvl_fwd += gst.t_fwd; lvl_inv += gst.t_inv; lvl_slot += gst.t_slot;
                lvl_hpack += gst.t_hpack; lvl_h2d += gst.t_h2d_batch;
                lvl_check += gst.t_check; lvl_plan += gst.t_plan;
                lvl_opcopy += gst.t_opcopy; lvl_hout += gst.t_hout;
                lvl_total += gst.t_total_call;
            }
            for (size_t s = 0; s < nbatch; ++s) {
                const size_t i = g.second[s];
                const size_t old = t[i].capacity();
                t[i].assign(res.begin() + (long)(s * nc * W), res.begin() + (long)((s + 1) * nc * W));
                update_parent(i, old);
                deg[i] = ma + mb - 2;
                ++fs.muls;
            }
        }
        /* Inputs were copied into wa/wb before the call; all device readers use that owned
           staging. Every sibling group has finished consuming this level before reclamation. */
        release_children(2*base, 4*base);
        if (lvl_trace)
            stage2_log::print(stage2_log::debug, "tree_level: base=%llu groups=%llu muls=%llu t=%.3f s | ntt total=%.3f "
                        "fwd=%.3f inv=%.3f slot=%.3f hpack=%.3f h2d=%.3f check=%.3f plan=%.3f "
                        "opcopy=%.3f hout=%.3f\n",
                        (unsigned long long)base, lvl_groups, lvl_muls, now_s() - tl0, lvl_total,
                        lvl_fwd, lvl_inv, lvl_slot, lvl_hpack, lvl_h2d, lvl_check, lvl_plan,
                        lvl_opcopy, lvl_hout);
        if (base == 1) break;
    }
    finish();
    return t;
}

/* (build_F_tree() -- the old "root only" wrapper -- is gone: the batched engine needs the
   WHOLE F heap for its descent, so run_check_F keeps build_tree_flat's return value and uses
   t[1] as the root where the earlier slices wanted only that.) */



static int g_device = 1;

static void hex_of_words(std::string &out, const unsigned long long *w, size_t W, const mpz_t N)
{
    mpz_t v;
    mpz_init(v);
    words_to_mpz(v, w, W);
    mpz_mod(v, v, N);
    char *s = mpz_get_str(nullptr, 16, v);
    out = s;
    void (*ff)(void *, size_t) = nullptr;
    mp_get_memory_functions(nullptr, nullptr, &ff);
    ff(s, std::strlen(s) + 1);
    mpz_clear(v);
}

/* ---- device mod-N arithmetic vs GMP, for this N (the ladder's foundation) ------------- */
static int mont_selftest(const std::vector<unsigned long long> &hn, size_t nw,
                         unsigned long long ninv, const mpz_t N, const mpz_t R)
{
    const bool requested=fuse_env_ull("NTT_POINT_MERSENNE",0)!=0;
    const int bits=(int)mpz_sizeinbase(N,2);
    const bool exact_shape=bits>=2 && mpz_popcount(N)==(mp_bitcnt_t)bits && (size_t)((bits+63)/64)==nw;
    const int enabled_bits=requested && exact_shape ? bits : 0;
    CK(cudaMemcpyToSymbol(g_s2g_device_constants,&enabled_bits,sizeof(enabled_bits),
                          offsetof(S2GDeviceConstants, point_mersenne_bits)));
    stage2_log::print(stage2_log::debug, "point_mersenne_mode: requested=%d enabled=%d bits=%d nw=%llu reduction=fold_rotate\n",
                (int)requested,enabled_bits!=0,bits,(unsigned long long)nw);
    stage2_log::print(stage2_log::debug, "point_arithmetic: xadd6=%d xadd_mont_muls=%d coordinate_scale=legacy_exact\n",
                (int)g_xadd6,g_xadd6 ? 6 : 8);
    const int cases = 2048;
    std::vector<unsigned long long> ha((size_t)cases * nw, 0ull), hb((size_t)cases * nw, 0ull);
    uint64_t s = 0x2468ace13579bdfull;
    mpz_t x;
    mpz_init(x);
    for (int side = 0; side < 2; ++side) {
        std::vector<unsigned long long> &dst = (side == 0) ? ha : hb;
        for (int i = 0; i < cases; ++i) {
            std::vector<unsigned long long> w(nw, 0ull);
            for (size_t j = 0; j < nw; ++j) {
                s = s * 6364136223846793005ull + 1442695040888963407ull;
                w[j] = s;
            }
            words_to_mpz(x, w.data(), nw);
            mpz_mod(x, x, N);
            mpz_to_words(w, nw, x);
            std::copy(w.begin(), w.end(), dst.begin() + (long)((size_t)i * nw));
        }
    }
    unsigned long long *da = nullptr, *db = nullptr, *dn = nullptr, *dout = nullptr;
    CK(cudaMalloc(&da, ha.size() * 8));
    CK(cudaMalloc(&db, hb.size() * 8));
    CK(cudaMalloc(&dn, nw * 8));
    CK(cudaMalloc(&dout, ha.size() * 8));
    CK(cudaMemcpy(da, ha.data(), ha.size() * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(db, hb.data(), hb.size() * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dn, hn.data(), nw * 8, cudaMemcpyHostToDevice));
    S2G_DISPATCH((int)nw, s2g_launch_mont_test, (int)nw, cases, da, db, dn, ninv, dout);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    std::vector<unsigned long long> ho(ha.size(), 0ull);
    CK(cudaMemcpy(ho.data(), dout, ho.size() * 8, cudaMemcpyDeviceToHost));
    cudaFree(da); cudaFree(db); cudaFree(dn); cudaFree(dout);

    unsigned long long bad = 0, first = 0;
    mpz_t am, bm, acc, rinv, want;
    mpz_inits(am, bm, acc, rinv, want, nullptr);
    mpz_invert(rinv, R, N);                                /* R^-1 mod N */
    for (int i = 0; i < cases; ++i) {
        words_to_mpz(am, &ha[(size_t)i * nw], nw);
        words_to_mpz(bm, &hb[(size_t)i * nw], nw);
        mpz_mul(acc, am, bm);
        mpz_mul(acc, acc, rinv);
        mpz_mod(acc, acc, N);
        words_to_mpz(want, &ho[(size_t)i * nw], nw);
        if (mpz_cmp(acc, want) != 0) { if (!bad) first = (unsigned long long)i; ++bad; }
    }
    mpz_clears(am, bm, acc, rinv, want, x, nullptr);
    stage2_log::print(stage2_log::debug, "mont_selftest: cases=%d mismatches=%llu first_bad=%llu "
                "(device Mont(a,b) vs GMP a*b*R^-1 mod N, nw=%llu)\n",
                cases, bad, first, (unsigned long long)nw);
    return bad ? 1 : 0;
}

static int xadd_selftest(const std::vector<unsigned long long> &hn,size_t nw,
                          unsigned long long ninv,const mpz_t N,const mpz_t R)
{
    const char *flag=std::getenv("NTT_XADD6_TEST");
    if(!flag || !*flag || !std::atoi(flag)) return 0;
    const int cases=128; // 64 ordinary-domain inputs + their 64 Montgomery images
    std::vector<unsigned long long> input((size_t)cases*6*nw),got((size_t)cases*5*3*nw);
    mpz_t x,rinv,a[6],tx,tz,v,half,observed;
    mpz_inits(x,rinv,tx,tz,v,half,observed,nullptr);
    for(auto &item:a) mpz_init(item);
    mpz_invert(rinv,R,N);
    uint64_t seed=0x85c529913ba8c742ull;
    for(int c=0;c<cases/2;++c) for(int j=0;j<6;++j) {
        std::vector<unsigned long long> w(nw);
        for(auto &word:w) {seed=seed*6364136223846793005ull+1442695040888963407ull;word=seed;}
        words_to_mpz(x,w.data(),nw);mpz_mod(x,x,N);
        if(c<8) {
            switch((c+j)%8) {
                case 0:mpz_set_ui(x,0);break;
                case 1:mpz_set_ui(x,1);break;
                case 2:mpz_sub_ui(x,N,1);break;
                case 3:mpz_sub_ui(x,N,2);break;
                case 4:mpz_fdiv_q_2exp(x,N,1);break;
                case 5:mpz_fdiv_q_2exp(x,N,1);mpz_add_ui(x,x,1);break;
                case 6:mpz_set_ui(x,3);mpz_mod(x,x,N);break;
                default:break;
            }
        }
        mpz_to_words(w,nw,x);
        std::copy(w.begin(),w.end(),input.begin()+((size_t)c*6+j)*nw);
        mpz_mul(x,x,R);mpz_mod(x,x,N);mpz_to_words(w,nw,x);
        std::copy(w.begin(),w.end(),input.begin()+((size_t)(c+cases/2)*6+j)*nw);
    }
    unsigned long long *di=nullptr,*dn=nullptr,*dout=nullptr;
    CK(cudaMalloc(&di,input.size()*8));CK(cudaMalloc(&dn,nw*8));CK(cudaMalloc(&dout,got.size()*8));
    CK(cudaMemcpy(di,input.data(),input.size()*8,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dn,hn.data(),nw*8,cudaMemcpyHostToDevice));
    unsigned long long bad=0,first=0;
    auto mont=[&](mpz_t dst,const mpz_t u,const mpz_t w) {
        mpz_mul(dst,u,w);mpz_mul(dst,dst,rinv);mpz_mod(dst,dst,N);
    };
    for(int candidate=0;candidate<2;++candidate) {
        S2G_DISPATCH((int)nw,s2g_launch_xadd_test,(int)nw,cases,di,dn,ninv,dout,candidate!=0);
        CK(cudaGetLastError());CK(cudaDeviceSynchronize());
        const char *fault=std::getenv("NTT_XADD6_TEST_BAD");
        if(candidate && fault && std::atoi(fault)) CK(cudaMemset(dout,0xff,8));
        CK(cudaMemcpy(got.data(),dout,got.size()*8,cudaMemcpyDeviceToHost));
        for(int c=0;c<cases;++c) {
            for(int j=0;j<6;++j) words_to_mpz(a[j],input.data()+((size_t)c*6+j)*nw,nw);
            mont(tx,a[0],a[2]);mont(v,a[1],a[3]);mpz_sub(tx,tx,v);mpz_mod(tx,tx,N);
            mont(tx,tx,tx);mont(tx,a[5],tx);
            mont(tz,a[0],a[3]);mont(v,a[1],a[2]);mpz_sub(tz,tz,v);mpz_mod(tz,tz,N);
            mont(tz,tz,tz);mont(tz,a[4],tz);
            mpz_set(half,a[0]);if(mpz_odd_p(half)) mpz_add(half,half,N);mpz_fdiv_q_2exp(half,half,1);
            for(int alias=0;alias<5;++alias) for(int coord=0;coord<3;++coord) {
                words_to_mpz(observed,got.data()+(((size_t)c*5+alias)*3+coord)*nw,nw);
                if(mpz_cmp(observed,coord==0 ? tx : coord==1 ? tz : half)!=0) {
                    if(!bad) first=((candidate*cases+c)*5+alias)*3+coord;
                    ++bad;
                }
            }
        }
    }
    CK(cudaFree(di));CK(cudaFree(dn));CK(cudaFree(dout));
    for(auto &item:a) mpz_clear(item);
    mpz_clears(x,rinv,tx,tz,v,half,observed,nullptr);
    stage2_log::print(stage2_log::debug, "xadd6_selftest: cases=%d aliases=5 domains=2 modes=2 coordinates=%d halves=%d bad=%llu first_bad=%llu nw=%llu\n",
                cases*5*2,cases*5*2*2,cases*5*2,bad,first,(unsigned long long)nw);
    return bad ? 1 : 0;
}

/* the ladder's parameters, kept together so the giant points and the culprit naming reuse the
   very same kernel the baby points were computed with */
struct LadderCtx {
    std::vector<unsigned long long> hn, hqx, hqz, ha24, hmone;
    unsigned long long ninv = 0;
    size_t nw = 0;
};


/* (X_i, Z_i) = [js[i]]Q on the device, written back in the NORMAL domain */
static void ladder_points(const LadderCtx &C, const std::vector<unsigned long long> &js,
                          std::vector<unsigned long long> &outx,
                          std::vector<unsigned long long> &outz)
{
    const size_t nw = C.nw, n = js.size();
    outx.assign(n * nw, 0ull);
    outz.assign(n * nw, 0ull);
    if (n == 0) return;
    /* ONE set of device buffers for every ladder call in the process.  This used to be eight
       cudaMalloc + eight cudaFree per CALL, and at nw=83 (5261 bits) the driver overhead
       dominated the arithmetic completely: the setup chain is 25 calls with ONE point each and
       it measured 12.693 s of every real run (~0.5 s per call), while each of those ladders is a
       <=10-bit chain, i.e. a few hundred Montgomery multiplications (section 26.6).  The buffers
       grow on demand and are keyed by nw, so the only thing that changes is when memory is
       requested: the kernel, the packing, the values and the results are bit-identical. */
    static unsigned long long *dn = nullptr, *dqx = nullptr, *dqz = nullptr, *da24 = nullptr,
                              *dmone = nullptr, *djs = nullptr, *dx = nullptr, *dz = nullptr;
    static size_t cap_nw = 0, cap_n = 0;
    if (cap_nw != nw) {
        cudaFree(dn); cudaFree(dqx); cudaFree(dqz); cudaFree(da24); cudaFree(dmone);
        cudaFree(djs); cudaFree(dx); cudaFree(dz);
        dn = dqx = dqz = da24 = dmone = djs = dx = dz = nullptr;
        cap_nw = nw;
        cap_n = 0;
        CK(cudaMalloc(&dn, nw * 8));
        CK(cudaMalloc(&dqx, nw * 8));
        CK(cudaMalloc(&dqz, nw * 8));
        CK(cudaMalloc(&da24, nw * 8));
        CK(cudaMalloc(&dmone, nw * 8));
    }
    if (n > cap_n) {
        cudaFree(djs); cudaFree(dx); cudaFree(dz);
        djs = dx = dz = nullptr;
        CK(cudaMalloc(&djs, n * 8));
        CK(cudaMalloc(&dx, n * nw * 8));
        CK(cudaMalloc(&dz, n * nw * 8));
        cap_n = n;
    }
    CK(cudaMemcpy(dn, C.hn.data(), nw * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dqx, C.hqx.data(), nw * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dqz, C.hqz.data(), nw * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(da24, C.ha24.data(), nw * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dmone, C.hmone.data(), nw * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(djs, js.data(), n * 8, cudaMemcpyHostToDevice));
    S2G_DISPATCH((int)nw, s2g_launch_ladder, (int)nw, (int)n, dn, C.ninv, dqx, dqz, da24, dmone,
                 djs, dx, dz);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(outx.data(), dx, outx.size() * 8, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(outz.data(), dz, outz.size() * 8, cudaMemcpyDeviceToHost));
}

/* ===================================================================================== *
 *  THE OPTIONAL TAIL: giant points, remainder tree, accumulation, gcd.
 *
 *  This is slice S2's ALGORITHM running on slice S1's machinery.  It exists so the probe
 *  closes the loop the plan asks for -- "the frozen vector must give
 *  factors=59649589127497217 and hit_primes=114713, the same as the pairing reference and the
 *  CPU tree reference" -- and, more importantly, so the slice-1 polynomial layer is exercised
 *  where it actually matters.  Every multiplication is the same GPU NTT call; the tree WALK is
 *  host-driven here, which is exactly what S2/S3 must replace with a batched device descent.
 *
 *  The division is the reference's reversed-polynomial Newton over the composite N: every
 *  divisor is MONIC, so the leading coefficient is 1, whose inverse mod N is 1 -- no inversion
 *  can ever fail and the algorithm is valid without knowing the factors of N.  Coefficients
 *  use the same representation as everywhere else (W words, reduced mod N), held one vector
 *  per coefficient for readability.
 * ===================================================================================== */

/* Independent ordinary-coefficient product, used only by small-tree gates. */
static std::vector<unsigned long long> groot_product_gmp(PolyLayer &L,
    const std::vector<unsigned long long> &a,const std::vector<unsigned long long> &b)
{
    const size_t W=L.W,ma=a.size()/W,mb=b.size()/W;
    std::vector<unsigned long long> out((ma+mb-1)*W,0),word(W);
    mpz_t x,y,z;mpz_inits(x,y,z,nullptr);
    for(size_t k=0;k<ma+mb-1;++k) {
        mpz_set_ui(z,0);
        for(size_t i=0;i<ma;++i) if(k>=i && k-i<mb) {
            words_to_mpz(x,a.data()+i*W,W);words_to_mpz(y,b.data()+(k-i)*W,W);mpz_addmul(z,x,y);
        }
        mpz_mod(z,z,L.N);mpz_to_words(word,W,z);std::copy(word.begin(),word.end(),out.begin()+k*W);
    }
    mpz_clears(x,y,z,nullptr);return out;
}

static std::vector<std::vector<unsigned long long>> build_groot_device(
    PolyLayer &L,const std::vector<std::vector<unsigned long long>> &leaf,
    std::vector<size_t> &deg,size_t &pad_out,FTreeStats &fs,int cat,
    size_t device_n=0,const std::function<void(unsigned long long*)> &fill={},
    const std::function<void(const unsigned long long*,size_t)> &root_sink={})
{
    const size_t W=L.W,n=fill?device_n:leaf.size();size_t pad=1;while(pad<n)pad*=2;
    const bool check=g_groot_device_check;
    if(check && n>512) {std::fprintf(stderr,"%s: FATAL: resident GMP check limited to 512 leaves\n",NTT_PROBE_NAME);std::exit(3);}
    ++g_gdevice.trees;fs.leaves=n;fs.padded=pad;pad_out=pad;
    deg.assign(2*pad,0);
    for(size_t i=0;i<n;++i) {
        if(!fill && leaf[i].size()!=2*W) {std::fprintf(stderr,"%s: FATAL: resident tree requires linear leaves\n",NTT_PROBE_NAME);std::exit(3);}
        deg[pad+i]=1;
    }
    for(size_t i=pad;--i;)deg[i]=deg[2*i]+deg[2*i+1];
    std::vector<std::vector<unsigned long long>> tree(2*pad),expected;
    if(!n) {tree[1].assign(W,0);tree[1][0]=1;fs.node_peak_bytes=fs.node_retained_bytes=8*W;return tree;}
    S4Ctx &C=*L.s4;
    // Exclusive lease of the existing raw staging pair. Resident calls never upload raw inputs.
    const size_t need=2*n*W;
    size_t next_need=0;
    for(size_t i=pad/2;pad>1 && i<pad;++i)if(deg[i])next_need+=(deg[i]+1)*W;
    // Dense linear leaves: frontier words = (n + nonempty nodes)*W, decreasing each level.
    // A holds the largest (leaf) frontier; B only needs the first parent frontier.
    C.raw_reserve(need,g_groot_compact_raw?next_need:need);
    g_gdevice.raw_peak_bytes=std::max(g_gdevice.raw_peak_bytes,8ull*(C.d_rawA_cap+C.d_rawB_cap));
    g_gmemory.rawA_peak_bytes=std::max(g_gmemory.rawA_peak_bytes,8ull*C.d_rawA_cap);
    g_gmemory.rawB_peak_bytes=std::max(g_gmemory.rawB_peak_bytes,8ull*C.d_rawB_cap);
    const double upload=now_s();
    bool available[2]={false,false},used[2]={false,false};
    if(g_groot_leaf_staging && g_s4_async)for(int k=0;k<2;++k)
        available[k]=g_pin_out[k] && g_pin_out_cap[k]>=2*W && g_pin_ev[k];
    if(fill)fill(C.d_rawA);
    else if(available[0] || available[1]) {
        ++g_gmemory.pinned_trees;size_t offset=0,turn=0;
        while(offset<n) {
            size_t k=turn++&1;if(!available[k])k^=1;
            // The last D2H or our previous H2D must consume this pinned slot before CPU writes it.
            CK(cudaEventSynchronize(g_pin_ev[k]));
            size_t take=std::min(n-offset,g_pin_out_cap[k]/(2*W));
            if(g_groot_leaf_chunk)take=std::min(take,g_groot_leaf_chunk);
            for(size_t i=0;i<take;++i)std::copy(leaf[offset+i].begin(),leaf[offset+i].end(),g_pin_out[k]+2*i*W);
            CK(cudaMemcpyAsync(C.d_rawA+2*offset*W,g_pin_out[k],take*2*W*8,cudaMemcpyHostToDevice));
            CK(cudaEventRecord(g_pin_ev[k]));used[k]=true;
            ++g_gmemory.pinned_slices;g_gmemory.pinned_words+=take*2*W;
            g_gmemory.pinned_borrow_peak_bytes=std::max(g_gmemory.pinned_borrow_peak_bytes,16ull*take*W);
            offset+=take;
        }
        // Trace multiplies may grow/free output staging at entry: finish EVERY leaf H2D first.
        // n=1 has no multiply, so it also needs this explicit endpoint fence.
        for(int k=0;k<2;++k)if(used[k])CK(cudaEventSynchronize(g_pin_ev[k]));
    } else {
        if(g_groot_leaf_staging)++g_gmemory.leaf_fallbacks;else ++g_gmemory.legacy_trees;
        std::vector<unsigned long long> initial(need);
        for(size_t i=0;i<n;++i)std::copy(leaf[i].begin(),leaf[i].end(),initial.begin()+2*i*W);
        g_gdevice.host_staging_peak_bytes=std::max(g_gdevice.host_staging_peak_bytes,8ull*initial.capacity());
        CK(cudaMemcpy(C.d_rawA,initial.data(),need*8,cudaMemcpyHostToDevice));g_gmemory.pageable_words+=need;
    }
    if(!fill){C.t_h2d_raw+=now_s()-upload;C.raw_words+=need;}
    g_gdevice.leaf_words+=need;
    if(check) {
        std::vector<unsigned long long> device_leaf;
        if(fill){device_leaf.resize(need);CK(cudaMemcpy(device_leaf.data(),C.d_rawA,need*8,cudaMemcpyDeviceToHost));}
        expected.resize(2*pad);
        for(size_t i=0;i<pad;++i) {
            expected[pad+i].assign(W,0);expected[pad+i][0]=1;
            if(i<n) {
                if(fill)expected[pad+i].assign(device_leaf.begin()+2*i*W,device_leaf.begin()+2*(i+1)*W);
                else expected[pad+i]=leaf[i];
            }
        }
        for(size_t i=pad;--i;)expected[i]=groot_product_gmp(L,expected[2*i],expected[2*i+1]);
    }
    auto *cur=C.d_rawA,*next=C.d_rawB;size_t cur_words=need;
    std::vector<unsigned long long> offsets(pad);
    for(size_t i=0;i<pad;++i)offsets[i]=std::min(i,n)*2*W;
    struct Metadata {
        unsigned long long *device=nullptr,*pinned=nullptr;size_t capacity=0;
        ~Metadata(){if(device)CK(cudaFree(device));if(pinned)CK(cudaFreeHost(pinned));}
    } meta;
    if(pad>1) {
        meta.capacity=3*(pad/2);CK(cudaMalloc(&meta.device,meta.capacity*8));
        if(g_s4_async) {
            cudaError_t err=cudaHostAlloc(&meta.pinned,meta.capacity*8,cudaHostAllocDefault);
            if(err!=cudaSuccess){meta.pinned=nullptr;cudaGetLastError();}
        }
        g_gdevice.metadata_peak_bytes=std::max(g_gdevice.metadata_peak_bytes,8ull*meta.capacity);
    }
    auto validate=[&](size_t base) {
        if(!check)return;
        std::vector<unsigned long long> words(cur_words);
        CK(cudaMemcpy(words.data(),cur,cur_words*8,cudaMemcpyDeviceToHost));
        for(size_t i=0;i<base;++i) {
            const size_t d=deg[base+i];++g_gdevice.checked_nodes;g_gdevice.checked_words+=(d+1)*W;
            if(d && !std::equal(expected[base+i].begin(),expected[base+i].end(),words.begin()+offsets[i])) {
                std::fprintf(stderr,"%s: FATAL: resident GMP node mismatch node=%llu degree=%llu\n",NTT_PROBE_NAME,
                    (unsigned long long)(base+i),(unsigned long long)d);std::exit(3);
            }
        }
    };
    validate(pad);
    const double start=now_s();
    static bool poisoned=false;
    for(size_t base=pad/2; base; base/=2) {
        const double level_start=now_s();
        std::vector<unsigned long long> target(base);size_t next_words=0;
        for(size_t i=0;i<base;++i) {target[i]=next_words;if(deg[base+i])next_words+=(deg[base+i]+1)*W;}
        if(next_words>C.raw_capacity(next)) {std::fprintf(stderr,"%s: FATAL: resident next frontier exceeds raw lease\n",NTT_PROBE_NAME);std::exit(3);}
        g_gdevice.logical_frontier_peak_bytes=std::max(g_gdevice.logical_frontier_peak_bytes,8ull*(cur_words+next_words));
        std::map<std::pair<size_t,size_t>,std::vector<size_t>> groups;
        for(size_t i=0;i<base;++i) {
            const size_t da=deg[2*base+2*i],db=deg[2*base+2*i+1];
            if(!da && !db)continue;
            if(!da || !db) {
                const size_t child=2*i+(da?0:1),d=da?da:db;
                CK(cudaMemcpyAsync(next+target[i],cur+offsets[child],(d+1)*W*8,cudaMemcpyDeviceToDevice));
                ++g_gdevice.copies;continue;
            }
            groups[{std::min(da,db)+1,std::max(da,db)+1}].push_back(i);
        }
        for(const auto &group:groups) {
            const size_t ma=group.first.first,mb=group.first.second,nb=group.second.size();
            std::vector<unsigned long long> map(3*nb);
            for(size_t j=0;j<nb;++j) {
                const size_t i=group.second[j],left=2*i,right=left+1;
                const bool swap=deg[2*base+left]>deg[2*base+right];
                map[j]=offsets[swap?right:left];map[nb+j]=offsets[swap?left:right];map[2*nb+j]=target[i];
            }
            // Previous metadata upload is complete when the multiply's final carry check returns.
            // All scatter reads of DEVICE metadata precede this update in the default stream.
            if(meta.pinned) {
                std::copy(map.begin(),map.end(),meta.pinned);
                CK(cudaMemcpyAsync(meta.device,meta.pinned,map.size()*8,cudaMemcpyHostToDevice));
            } else CK(cudaMemcpy(meta.device,map.data(),map.size()*8,cudaMemcpyHostToDevice));
            g_gdevice.metadata_words+=map.size();
            S4DeviceBatch batch{cur,meta.device,next,map.data(),cur_words,next_words,nb};
            std::vector<unsigned long long> unused;
            poly_mul_batch_modN(L,nullptr,nullptr,ma,mb,nb,unused,cat,nullptr,0,ma+mb-1,&batch);
            ++g_gdevice.groups;g_gdevice.pairs+=nb;fs.muls+=nb;
            ++L.s4->groups;
        }
        const char *fault=std::getenv("NTT_GROOT_DEVICE_TEST_BAD");
        if(fault && std::atoi(fault) && !poisoned && next_words) {
            CK(cudaMemsetAsync(next,0xff,8));poisoned=true;
            stage2_log::print(stage2_log::debug, "gdevice_fault: first computed frontier poisoned\n");
        }
        std::swap(cur,next);offsets.swap(target);cur_words=next_words;++g_gdevice.levels;validate(base);
        if(g_s4_batched_progress)
            stage2_log::print(stage2_log::debug, "gdevice_progress: base=%llu groups=%llu t_level=%.3f t_total=%.3f\n",
                (unsigned long long)base,(unsigned long long)groups.size(),now_s()-level_start,now_s()-start);
    }
    if(root_sink) {
        // The consumer copies before any subsequent call can recycle raw A/B.
        root_sink(cur,n+1);g_gdevice.root_words+=(n+1)*W;return tree;
    }
    tree[1].resize((n+1)*W);
    const double tr=now_s();CK(cudaMemcpy(tree[1].data(),cur,tree[1].size()*8,cudaMemcpyDeviceToHost));
    L.t_d2h_coeff+=now_s()-tr;L.d2h_coeff_words+=tree[1].size();g_gdevice.root_words+=tree[1].size();
    fs.node_peak_bytes=fs.node_retained_bytes=8ull*tree[1].capacity();
    return tree;
}

static std::vector<std::vector<unsigned long long>> build_groot_select(
    PolyLayer &L,const std::vector<std::vector<unsigned long long>> &leaf,
    std::vector<size_t> &deg,size_t &pad,FTreeStats &fs,int cat,bool keep_children,
    const std::function<void(const unsigned long long*,size_t)> &root_sink={})
{
    const char *host=std::getenv("NTT_S4_HOSTPACK");
    if(g_groot_device && !keep_children && L.s4 && g_s4_pack_direct && !g_s4_final_readback &&
       !(host && std::atoi(host)))return build_groot_device(L,leaf,deg,pad,fs,cat,0,{},root_sink);
    if(g_groot_device)++g_gdevice.fallbacks;
    return build_tree_flat(L,leaf,deg,pad,fs,cat,keep_children);
}

static void groot_device_fixture(PolyLayer &L)
{
    const size_t W=L.W;unsigned long long cases=0,words=0;
    const auto nodes0=g_gdevice.checked_nodes,words0=g_gdevice.checked_words;
    mpz_t c,z;mpz_inits(c,z,nullptr);
    for(size_t n:{0u,1u,2u,3u,5u,7u,8u,9u,17u,23u,257u,511u})for(size_t mode=0;mode<4;++mode) {
        std::vector<std::vector<unsigned long long>> leaf;
        std::vector<unsigned long long> expected(W,0),word(W);expected[0]=1;
        for(size_t i=0;i<n;++i) {
            if(mode==0){mpz_set_ui(c,1);mpz_set_ui(z,1);}
            else if(mode==1){mpz_set_ui(c,13+i*17);mpz_set_ui(z,2+i*7);}
            else if(mode==2){mpz_sub_ui(c,L.N,1);mpz_sub_ui(z,L.N,1);}
            else {mpz_set_ui(c,i%3==0?0:i%3==1?1:i);mpz_set_ui(z,i%3==2?i+1:0);}
            mpz_mod(c,c,L.N);mpz_mod(z,z,L.N);
            std::vector<unsigned long long> f(2*W,0);
            mpz_to_words(word,W,c);std::copy(word.begin(),word.end(),f.begin());
            mpz_to_words(word,W,z);std::copy(word.begin(),word.end(),f.begin()+W);
            leaf.push_back(f);expected=groot_product_gmp(L,expected,f);
        }
        FTreeStats fs;std::vector<size_t> degree;size_t pad=0;
        auto tree=build_groot_select(L,leaf,degree,pad,fs,-1,false);
        if(tree[1]!=expected || degree[1]!=n) {std::fprintf(stderr,"%s: FATAL: resident fixture root mismatch n=%llu\n",NTT_PROBE_NAME,(unsigned long long)n);std::exit(3);}
        for(size_t i=2;i<tree.size();++i)if(!tree[i].empty())std::exit(3);
        ++cases;words+=expected.size();
    }
    mpz_clears(c,z,nullptr);
    stage2_log::print(stage2_log::debug, "gdevice_fixture: cases=%llu words=%llu checked_nodes=%llu checked_words=%llu bad=0 (GMP nonmonic, zero/constant-one, empty/single/padded/large trees)\n",
        cases,words,g_gdevice.checked_nodes-nodes0,g_gdevice.checked_words-words0);
}

using CPoly = std::vector<std::vector<unsigned long long>>;

static std::vector<unsigned long long> cp_zero(size_t W)
{
    return std::vector<unsigned long long>(W, 0ull);
}

static void cp_resize(CPoly &a, size_t n, size_t W)
{
    a.resize(n, cp_zero(W));
    for (std::vector<unsigned long long> &c : a)
        if (c.size() != W) c.assign(W, 0ull);
}

static CPoly cp_from_flat(const std::vector<unsigned long long> &f, size_t deg, size_t W)
{
    CPoly p;
    cp_resize(p, deg + 1, W);
    for (size_t k = 0; k <= deg; ++k)
        std::copy(f.begin() + (long)(k * W), f.begin() + (long)((k + 1) * W), p[k].begin());
    return p;
}

static std::vector<unsigned long long> cp_to_flat(const CPoly &p, size_t W)
{
    std::vector<unsigned long long> f(p.size() * W, 0ull);
    for (size_t k = 0; k < p.size(); ++k)
        std::copy(p[k].begin(), p[k].end(), f.begin() + (long)(k * W));
    return f;
}

static bool cp_coeff_zero(const std::vector<unsigned long long> &c)
{
    for (unsigned long long w : c)
        if (w) return false;
    return true;
}

static void cp_trim(CPoly &a)
{
    while (a.size() > 1 && cp_coeff_zero(a.back())) a.pop_back();
}

static bool cp_is_one(const CPoly &a, size_t W)
{
    return a.size() == 1 && poly_is_one(a[0].data(), W);
}

/* dst = x - y mod N (one coefficient); x or y may be absent (empty vector = 0) */
static void cp_coeff_sub(std::vector<unsigned long long> &dst,
                         const std::vector<unsigned long long> &x,
                         const std::vector<unsigned long long> &y, const mpz_t N, size_t W)
{
    mpz_t a, b;
    mpz_inits(a, b, nullptr);
    if (x.size() == W) words_to_mpz(a, x.data(), W); else mpz_set_ui(a, 0);
    if (y.size() == W) words_to_mpz(b, y.data(), W); else mpz_set_ui(b, 0);
    mpz_sub(a, a, b);
    mpz_mod(a, a, N);
    mpz_to_words(dst, W, a);
    mpz_clears(a, b, nullptr);
}

/* a +/- b coefficient-wise mod N */
static CPoly cp_addsub(const CPoly &a, const CPoly &b, PolyLayer &L, bool sub)
{
    const size_t W = L.W;
    CPoly r;
    cp_resize(r, std::max(a.size(), b.size()), W);
    mpz_t x, y;
    mpz_inits(x, y, nullptr);
    std::vector<unsigned long long> tmp(W, 0ull);
    for (size_t i = 0; i < r.size(); ++i) {
        if (i < a.size()) words_to_mpz(x, a[i].data(), W); else mpz_set_ui(x, 0);
        if (i < b.size()) words_to_mpz(y, b[i].data(), W); else mpz_set_ui(y, 0);
        if (sub) mpz_sub(x, x, y); else mpz_add(x, x, y);
        mpz_mod(x, x, L.N);
        mpz_to_words(tmp, W, x);
        r[i] = tmp;
    }
    mpz_clears(x, y, nullptr);
    cp_trim(r);
    return r;
}

/* a*b mod N through the SAME GPU multiply the F tree uses.  With slice S4's batched multiply
   in place (L.s4) this is the batched entry point with nbatch == 1, i.e. the same kernels AND
   the device-side mod-N reduction -- the tree has one multiply path, never two. */
static CPoly cp_mul(const CPoly &a, const CPoly &b, PolyLayer &L, size_t keep = (size_t)-1)
{
    const size_t W = L.W;
    if(a.empty() || b.empty()) return {};
    const size_t nc=a.size()+b.size()-1;
    const size_t wanted=keep==(size_t)-1 ? nc : std::min(keep,nc);
    if (L.s4) {
        /* the batched entry point takes nbatch slices of P*W words, ZERO-PADDED: an operand
           shorter than P must not be handed over as a tight array, or the packer reads past
           its end (that is a real bug this slice shipped for one build: cp_divmod multiplies
           unequal lengths constantly, and the garbage above the operand landed in the top
           coefficients -- the multiply's own GMP check cannot see it, because it verifies the
           digits it produced, not what was fed in). */
        const size_t P = (a.size() > b.size()) ? a.size() : b.size();
        std::vector<unsigned long long> fa(P * W, 0ull), fb(P * W, 0ull);
        for (size_t i = 0; i < a.size(); ++i)
            std::copy(a[i].begin(), a[i].end(), fa.begin() + (long)(i * W));
        for (size_t i = 0; i < b.size(); ++i)
            std::copy(b[i].begin(), b[i].end(), fb.begin() + (long)(i * W));
        std::vector<unsigned long long> fc;
        poly_mul_batch_modN(L, fa.data(), fb.data(), a.size(), b.size(), 1, fc,-1,nullptr,0,wanted);
        return wanted ? cp_from_flat(fc,wanted-1,W) : CPoly{};
    }
    const std::vector<unsigned long long> fa = cp_to_flat(a, W);
    const std::vector<unsigned long long> fb = cp_to_flat(b, W);
    const std::vector<unsigned long long> fc = poly_mul_modN(L, fa, a.size() - 1, fb, b.size() - 1);
    return wanted ? cp_from_flat(fc,wanted-1,W) : CPoly{};
}

/* g = 1/a mod X^k.  The reference's Newton doubling, with its two hard-won details kept:
   the intermediate g is NEVER trimmed (a trimmed inverse whose last coefficient is zero can
   stop growing and spin forever) and a non-invertible a[0] is fatal rather than silent. */
static CPoly cp_inv_series(const CPoly &a, size_t k, PolyLayer &L)
{
    const size_t W = L.W;
    CPoly g;
    cp_resize(g, 1, W);
    {
        mpz_t av, inv;
        mpz_inits(av, inv, nullptr);
        words_to_mpz(av, a[0].data(), W);
        if (mpz_invert(inv, av, L.N) == 0) {
            std::fprintf(stderr, "%s: FATAL: divisor leading coefficient not invertible mod N "
                                 "(the divisor was not monic)\n", NTT_PROBE_NAME);
            std::exit(4);
        }
        mpz_mod(inv, inv, L.N);
        mpz_to_words(g[0], W, inv);
        mpz_clears(av, inv, nullptr);
    }
    while (g.size() < k) {
        const size_t nxt = std::min(2 * g.size(), k);
        const CPoly at(a.begin(), a.begin() + (long)std::min(a.size(), nxt));
        CPoly ag = cp_mul(at, g, L, nxt);
        cp_resize(ag, nxt, W);                        /* mod X^nxt */
        CPoly h;
        cp_resize(h, nxt, W);
        {
            mpz_t two;
            mpz_init(two);
            mpz_set_ui(two, 2);
            mpz_mod(two, two, L.N);
            mpz_to_words(h[0], W, two);
            mpz_clear(two);
        }
        h = cp_addsub(h, ag, L, /*sub=*/true);        /* h = 2 - ag */
        cp_resize(h, nxt, W);
        CPoly gn = cp_mul(g, h, L, nxt);
        cp_resize(gn, nxt, W);                        /* mod X^nxt */
        g = gn;
    }
    return g;
}

/* a = q*b + r with deg r < deg b; b is MONIC everywhere in this file */
static void cp_divmod(CPoly &q, CPoly &r, const CPoly &a, const CPoly &b, PolyLayer &L)
{
    const size_t W = L.W;
    const long da = (long)a.size() - 1, db = (long)b.size() - 1;
    if (da < db) {
        cp_resize(q, 1, W);
        r = a;
        cp_trim(r);
        return;
    }
    const size_t k = (size_t)(da - db + 1);
    CPoly ra, rb;
    cp_resize(ra, k, W);
    cp_resize(rb, (size_t)db + 1, W);
    for (size_t i = 0; i < k; ++i) ra[i] = a[(size_t)(da - (long)i)];
    for (size_t i = 0; i <= (size_t)db; ++i) rb[i] = b[(size_t)(db - (long)i)];
    const CPoly rbi = cp_inv_series(rb, k, L);
    CPoly qrev = cp_mul(ra, rbi, L, k);
    cp_resize(qrev, k, W);
    cp_resize(q, k, W);
    for (size_t i = 0; i < k; ++i) q[i] = qrev[k - 1 - i];
    cp_trim(q);

    const CPoly qb = cp_mul(q, b, L, (size_t)db);
    if (db == 0) {
        cp_resize(r, 1, W);
    } else {
        cp_resize(r, (size_t)db, W);
        for (size_t i = 0; i < (size_t)db; ++i) {
            const std::vector<unsigned long long> &x = a[i];
            const std::vector<unsigned long long> &y =
                (i < qb.size()) ? qb[i] : cp_zero(W);
            cp_coeff_sub(r[i], x, y, L.N, W);
        }
    }
    cp_trim(r);
}

/* the reference descent, one cp_mod per node (the pre-S4 code, kept as the oracle the batched
   version is checked against) */
static CPoly cp_mod(const CPoly &a, const CPoly &b, PolyLayer &L);

static void descent_slow(PolyLayer &L,
                         const std::vector<std::vector<unsigned long long>> &Ft,
                         const std::vector<size_t> &Fdeg, size_t Fpad, const CPoly &H,
                         std::vector<std::vector<unsigned long long>> &values,
                         unsigned long long &divmods)
{
    const size_t W = L.W;
    /* the leaf count is the F tree's degree, not H's coefficient count (section 54.5): H mod F
       has deg < deg F, so H may be SHORTER than the leaf count -- and at the first batch, where
       H = T with no reduction, it usually is.  Reading `cur[i]` for i in [H.size(), P) is exactly
       the right thing: the tree walk has already reduced H (implicitly zero-extended) down to
       every leaf, so those rows are H(x_i), not zeros. */
    const size_t P = (Fdeg.size() > 1 && Fdeg[1] > 0) ? Fdeg[1] : H.size();
    std::vector<CPoly> cur(1);
    cur[0] = H;
    size_t base = 1, cnt = 1;
    while (base < Fpad) {
        const size_t nbase = base * 2;
        std::vector<CPoly> nxt(cnt * 2);
        for (size_t j = 0; j < cnt; ++j) {
            for (int sgn = 0; sgn < 2; ++sgn) {
                const size_t ci = nbase + 2 * j + (size_t)sgn;
                if (Fdeg[ci] == 0 && poly_is_one(Ft[ci].data(), W)) {
                    cp_resize(nxt[2 * j + (size_t)sgn], 1, W);      /* H mod 1 = 0 */
                } else {
                    if (cur[j].size() >= Fdeg[ci] + 1) ++divmods;
                    nxt[2 * j + (size_t)sgn] =
                        cp_mod(cur[j], cp_from_flat(Ft[ci], Fdeg[ci], W), L);
                }
            }
        }
        cur.swap(nxt);
        base = nbase;
        cnt *= 2;
        if (g_s4_descent_trace) {
            unsigned long long hh = 1469598103934665603ull;
            for (size_t j = 0; j < cnt; ++j)
                for (const std::vector<unsigned long long> &c : cur[j])
                    for (unsigned long long v : c) hh = (hh ^ v) * 1099511628211ull;
            stage2_log::print(stage2_log::debug, "descent_trace: slow base=%llu nodes=%llu hash=%llu\n",
                        (unsigned long long)base, (unsigned long long)cnt, hh);
        }
    }
    values.assign(P, std::vector<unsigned long long>(W, 0ull));
    for (size_t i = 0; i < P; ++i) values[i] = cp_to_flat(cur[i], W);
}

/* the batched descent: H mod every node of the F tree, one level at a time.  `values` comes
   back as the leaf remainders (the H(x_j)); `divmods` counts the cp_divmod calls the CPU
   reference's cost model counts. */
static void divmod_batch(PolyLayer &L, const std::vector<unsigned long long> &A,
                         const std::vector<unsigned long long> &B, size_t da, size_t db,
                         size_t nbatch, std::vector<unsigned long long> &out, int cat);

/* the linear-divisor leaf specialisation of the batched descent (defined below, next to
   flat_truncate); declared here because the descent's group loop calls it */
static void leaf_mod_linear_batch(const std::vector<unsigned long long> &A, size_t da,
                                  const std::vector<unsigned long long> &B, size_t nbatch,
                                  size_t W, const mpz_t N,
                                  std::vector<unsigned long long> &out);



/* a mod b for a MONIC b; the degree fast path is what makes a descent over a tree whose top
   nodes are much larger than F almost free (no division, no multiply). */
static CPoly cp_mod(const CPoly &a, const CPoly &b, PolyLayer &L)
{
    if (cp_is_one(b, L.W)) { CPoly z; cp_resize(z, 1, L.W); return z; }
    if (a.size() < b.size()) return a;
    CPoly q, r;
    cp_divmod(q, r, a, b, L);
    return r;
}

/* (a mod (X + b0)) for nbatch pairs whose divisor IS monic and linear: the answer is the single
   coefficient a(-b0), i.e. Horner evaluated at the divisor's constant term -- note the tree's
   leaf is (X - x_j), so b0 = -x_j and the evaluation point is x_j.

   The generic Newton division computes exactly this (at k = 1 it is q = a[0], r = a[1] -
   root*a[0]), but the LAST level of the descent is P/2 nodes of this one shape, and running the
   full machinery there made the real shape's final descent level its longest single phase.  This
   is the same arithmetic with one multiply-add per coefficient.

   The divisor shape is CHECKED, not assumed: a divisor whose degree is not exactly 1 in this
   group would silently return a wrong leaf value, so it is fatal. */
static void leaf_mod_linear_batch(const std::vector<unsigned long long> &A, size_t da,
                                  const std::vector<unsigned long long> &B, size_t nbatch,
                                  size_t W, const mpz_t N, std::vector<unsigned long long> &out)
{
    out.assign(nbatch * W, 0ull);
    mpz_t h, x, t;
    mpz_inits(h, x, t, nullptr);
    std::vector<unsigned long long> w(W, 0ull);
    for (size_t s = 0; s < nbatch; ++s) {
        const unsigned long long *b = &B[s * 2 * W];
        if (b[W] != 1) {
            std::fprintf(stderr, "%s: FATAL: leaf_mod_linear_batch: divisor %llu is not monic "
                                 "(lead=%llu)\n", NTT_PROBE_NAME, (unsigned long long)s, b[W]);
            std::exit(3);
        }
        for (size_t q = 2; q < W; ++q)
            if (b[q] != 0) {
                std::fprintf(stderr, "%s: FATAL: leaf_mod_linear_batch: divisor %llu has degree "
                                     "> 1 (word %llu = %llu)\n", NTT_PROBE_NAME,
                             (unsigned long long)s, (unsigned long long)q, b[q]);
                std::exit(3);
            }
        words_to_mpz(x, b, W);                       /* b = [ -root, 1 ] */
        mpz_neg(x, x);                               /* the evaluation point IS the root */
        mpz_mod(x, x, N);
        /* Horner over REVERSED coefficient order, low coefficient inside: h = a_da, then
           h = a_i + h*x for i = da-1 .. 0, so the constant coefficient is applied last.  (The
           other way round is NOT equivalent: for da = 1 it gives a1 + a0*x instead of
           a0 + a1*x, which is what the first version of this function did -- the coefficients
           are stored low-first, so a direct loop walks them in the wrong order.) */
        words_to_mpz(h, &A[s * (da + 1) * W + da * W], W);
        for (size_t i = da; i-- > 0;) {
            mpz_mul(t, h, x);
            words_to_mpz(h, &A[s * (da + 1) * W + i * W], W);
            mpz_add(h, h, t);
            mpz_mod(h, h, N);
        }
        mpz_to_words(w, W, h);
        std::copy(w.begin(), w.end(), out.begin() + (long)(s * W));
    }
    mpz_clears(h, x, t, nullptr);
}

/* ===================================================================================== *
 *  SLICE S4 (B) -- THE BATCHED DESCENT
 *
 *  The descent is where the call count of a REAL shape explodes: H is reduced against every
 *  node of the F tree, so at P = 92160 there are ~1.8e5 divmods, and the bottom two levels
 *  (k = da-db+1 <= 2, i.e. no Newton step at all, just a handful of coefficient products) are
 *  1.4e5 of them.  At the measured single-multiply fixed cost (~370 us per LAUNCH) that is
 *  ~70 s of pure launch overhead -- more than the whole rest of the curve.
 *
 *  There is nothing sequential about a descent level: every node of one level is reduced
 *  independently, against divisors of the SAME degree, so the whole level is one batched
 *  multiply per step of the division algorithm.  These three functions are that: flat_mul_batch
 *  (one batched multiply over nbatch pairs of one shape), inv_series_batch (the reference's
 *  Newton doubling, with every step batched) and divmod_batch (the reference's cp_divmod with
 *  every multiply batched).  The ALGORITHM is byte-for-byte the reference's -- same reversal,
 *  same Newton recurrence (g is never trimmed, a non-invertible lead is fatal), same
 *  truncation points -- only the arithmetic is grouped.
 * ===================================================================================== */

/* Truncate every slice of `v` to its low `keep` coefficients: `v` holds nbatch slices whose
   slice STRIDE (in coefficients) is `nc` and each of whose coefficients is W words, and it comes
   back TIGHTLY packed at nbatch*keep*W.  It REPACKS: a plain resize() keeps the old slice
   stride, so every slice after the first would be read from the wrong offset -- that is exactly
   the bug that made the batched descent wrong for k > 1 while k = 1 (where 2k-1 == k, no repack
   needed) looked fine.

   THE STRIDE IS A PARAMETER, NOT A CONVENTION.  The caller states, in `nc`, the stride it
   actually built: the operand's length must be nbatch*nc*W.  The version before this one derived
   nothing and checked `v.size() < nbatch*nc*W` anyway, so a caller that passed a buffer allocated
   for a SHORTER slice -- which is what a batch-multiply operand used to be, because
   flat_mul_batch reads its operands at the stride its ARGUMENTS declare while the result is
   written at ma+mb-1 -- walked off the end inside the memcpy below.  That is the real-shape
   host access violation (0xC0000005 in memcpy, called from here, 60-270 s in, on this function's
   first big repack) that the G-tree descent hit.

   The `caller` argument is the third line of defence: the guard below names the caller that
   built a short operand, so the next occurrence is a one-line diagnosis rather than a stack
   walk.  On every call site in this file the stride is now taken from the SAME expression that
   sized the buffer (flat_mul_batch returns exactly nbatch*(ma+mb-1)*W with its own assertion),
   so this cannot fire on a correct call. */
static void flat_truncate(std::vector<unsigned long long> &v, size_t nbatch, size_t nc,
                          size_t keep, size_t W, const char *caller)
{
    if (keep == nc) return;                            /* already tight at `keep` coefficients */
    const size_t need = nbatch * nc * W;               /* the last slice ends exactly here */
    if (keep > nc || v.size() < need) {
        std::fprintf(stderr, "%s: FATAL: %s: flat_truncate source too short (%llu words, needs "
                             "%llu = nbatch(%llu)*stride_nc(%llu)*W(%llu)) keep=%llu "
                             "per-slice-words-declared=%llu\n", NTT_PROBE_NAME, caller,
                     (unsigned long long)v.size(), (unsigned long long)need,
                     (unsigned long long)nbatch, (unsigned long long)nc, (unsigned long long)W,
                     (unsigned long long)keep, (unsigned long long)(nc * W));
        std::exit(3);
    }
    std::vector<unsigned long long> out(nbatch * keep * W, 0ull);
    for (size_t s = 0; s < nbatch; ++s) {
        const size_t src0 = s * nc * W;
        std::copy(v.begin() + (long)src0, v.begin() + (long)(src0 + keep * W),
                  out.begin() + (long)(s * keep * W));
    }
    v.swap(out);
}

/* out = A*B, where A holds nbatch slices of ma coefficients and B nbatch slices of mb, each
   slice W words per coefficient.  The slices are zero-padded to P = max(ma,mb) because the
   multiply takes a single shape per launch. */
struct FlatInputStats {
    unsigned long long calls = 0, borrowed = 0, padded = 0, alias_clones = 0;
    unsigned long long copy_bytes = 0, zero_bytes = 0, avoided_copy_bytes = 0, avoided_zero_bytes = 0;
    unsigned long long temp_peak_bytes = 0, control_peak_bytes = 0;
    double t_prepare = 0.0;
};
static FlatInputStats g_flat_input;

static void flat_mul_batch(PolyLayer &L, const std::vector<unsigned long long> &A, size_t ma,
                           const std::vector<unsigned long long> &B, size_t mb, size_t nbatch,
                           std::vector<unsigned long long> &out, int cat,
                           size_t first = 0, size_t count = (size_t)-1)
{
    const size_t W = L.W;
    const size_t P = (ma > mb) ? ma : mb;           /* the multiply takes ONE shape per launch */
    const size_t nc = count==(size_t)-1 ? ma+mb-1 : count;
    /* CONTRACT: every operand holds exactly nbatch slices of MA (MB) coefficients -- so its
       length is nbatch*ma*W (nbatch*mb*W) and its slice stride IS ma (mb) -- and the result is
       written TIGHTLY at the requested count (default ma+mb-1) coefficients per slice.
       FIRST addresses the full source convolution. All three are load-bearing, and
       all three were assumed rather than checked until the real shape broke:
         * a SHORTER operand (or one whose real slice is shorter than ma) is read past its end;
         * a LONGER one used to make the result `A.size()/ma * nc` words -- inflating each
           slice to k coefficients in the Newton step of inv_series_batch, after which
           flat_truncate was handed a slice stride it had never declared and read off the end.
       Both were host access violations at P = 4096 / W = 83.  The stride is stated once, here,
       and every caller now has to satisfy it: Newton's two multiplies plus divmod_batch's
       qrev = ra*rbi and qb = q*B. */
    if (A.size() != nbatch * ma * W || B.size() != nbatch * mb * W) {
        std::fprintf(stderr, "%s: FATAL: flat_mul_batch operand shape mismatch: A=%llu (must be "
                             "%llu = nbatch*ma*W), B=%llu (must be %llu = nbatch*mb*W) for "
                             "ma=%llu mb=%llu nbatch=%llu W=%llu\n", NTT_PROBE_NAME,
                     (unsigned long long)A.size(), (unsigned long long)(nbatch * ma * W),
                     (unsigned long long)B.size(), (unsigned long long)(nbatch * mb * W),
                     (unsigned long long)ma, (unsigned long long)mb, (unsigned long long)nbatch,
                     (unsigned long long)W);
        std::exit(3);
    }
    const double tp0 = now_s();
    std::vector<unsigned long long> wa, wb;
    const size_t padded_words = nbatch * P * W;
    auto input = [&](const std::vector<unsigned long long> &v, size_t m,
                     std::vector<unsigned long long> &scratch) -> const unsigned long long * {
        const unsigned long long bytes = 8ull * v.size();
        /* poly_mul_batch_modN assigns OUT before reading input. Clone aliases even when
           their stride already matches P. Borrowed sources stay alive for this whole call;
           async uploads copy them to owned pinned staging before returning to this function. */
        if (g_s4_flat_direct && m == P && &v != &out) {
            ++g_flat_input.borrowed;
            g_flat_input.avoided_copy_bytes += bytes;
            g_flat_input.avoided_zero_bytes += 8ull * padded_words;
            return v.data();
        }
        ++g_flat_input.padded;
        if (&v == &out) ++g_flat_input.alias_clones;
        scratch.assign(padded_words, 0ull);
        for (size_t s = 0; s < nbatch; ++s)
            std::copy(v.begin() + (long)(s * m * W), v.begin() + (long)((s + 1) * m * W),
                      scratch.begin() + (long)(s * P * W));
        g_flat_input.copy_bytes += bytes;
        g_flat_input.zero_bytes += 8ull * padded_words;
        return scratch.data();
    };
    const auto *pa = input(A, ma, wa);
    const auto *pb = input(B, mb, wb);
    ++g_flat_input.calls;
    g_flat_input.temp_peak_bytes = std::max(g_flat_input.temp_peak_bytes,
        8ull * (wa.capacity() + wb.capacity()));
    g_flat_input.control_peak_bytes = std::max(g_flat_input.control_peak_bytes, 16ull * padded_words);
    g_flat_input.t_prepare += now_s() - tp0;
    poly_mul_batch_modN(L, pa, pb, ma, mb, nbatch, out, cat, nullptr, first, count);
    /* The multiply must return the exact compact stride. A mismatch must never turn into
       a silently zero-filled polynomial. */
    if (out.size() != nbatch * nc * W) {
        std::fprintf(stderr,"%s: FATAL: compact batch output size mismatch\n",NTT_PROBE_NAME);
        std::exit(3);
    }
}

/* Host-contiguous fold; the GPU multiply and its checks are shared with CPoly.
   H keeps the SAME declared coefficient count as the legacy path: only the remainder
   is trimmed, never T or the inverse. This matters when a leading coefficient is zero. */
struct FoldFlatStats {
    bool enabled=false;
    unsigned long long folds=0, muls=0, sub_coeffs=0, peak_bytes=0;
    double t_prepare=0, t_multiply=0, t_subtract=0, t_bridge=0;
};
static void fold_flat_step(PolyLayer &L, const std::vector<unsigned long long> &G,
                           std::vector<unsigned long long> &H,
                           const std::vector<unsigned long long> &F,
                           const std::vector<unsigned long long> &finv, FoldFlatStats &st)
{
    const size_t W=L.W, P=F.size()/W-1;
    if(!L.s4 || !W || !P || F.size()%W || G.size()%W || H.size()%W || finv.size()%W) {
        std::fprintf(stderr,"%s: FATAL: flat fold shape mismatch\n",NTT_PROBE_NAME);std::exit(3);
    }
    ++st.folds;
    if(G.empty() || H.empty()) {H.clear();return;}
    std::vector<unsigned long long> T,qrev,qb;
    auto multiply=[&](const std::vector<unsigned long long> &a,
                      const std::vector<unsigned long long> &b,
                      std::vector<unsigned long long> &out,size_t count=(size_t)-1) {
        const double t=now_s();flat_mul_batch(L,a,a.size()/W,b,b.size()/W,1,out,BC_FOLD,0,count);
        st.t_multiply+=now_s()-t;++st.muls;
    };
    multiply(G,H,T);
    const size_t degT=T.size()/W-1;
    if(degT<P) {H=std::move(T);return;}
    const size_t k=degT-P+1;
    if(finv.size()<k*W) {std::fprintf(stderr,"%s: FATAL: flat fold inverse too short\n",NTT_PROBE_NAME);std::exit(3);}
    {
        const double tp=now_s();
        std::vector<unsigned long long> ra(k*W),rbi(finv.begin(),finv.begin()+k*W);
        for(size_t i=0;i<k;++i)std::copy_n(T.data()+(degT-i)*W,W,ra.data()+i*W);
        st.t_prepare+=now_s()-tp;
        multiply(ra,rbi,qrev,k);
        st.peak_bytes=std::max(st.peak_bytes,8ull*(H.capacity()+T.capacity()+ra.capacity()+rbi.capacity()+qrev.capacity()));
    } // reverse inputs no longer live when q*F is formed
    const double tq=now_s();
    for(size_t i=0;i<k/2;++i)for(size_t w=0;w<W;++w)std::swap(qrev[i*W+w],qrev[(k-1-i)*W+w]);
    st.t_prepare+=now_s()-tq;
    multiply(qrev,F,qb,P);
    H.resize(P*W);
    st.peak_bytes=std::max(st.peak_bytes,8ull*(H.capacity()+T.capacity()+qrev.capacity()+qb.capacity()));
    const double ts=now_s();
    // Reuse GMP limbs for the whole remainder, preserving general composite-N mod semantics.
    mpz_t a,b;mpz_inits(a,b,nullptr);
    std::vector<unsigned long long> word(W);
    for(size_t i=0;i<P;++i) {
        words_to_mpz(a,T.data()+i*W,W);words_to_mpz(b,qb.data()+i*W,W);
        mpz_sub(a,a,b);mpz_mod(a,a,L.N);mpz_to_words(word,W,a);
        std::copy_n(word.data(),W,H.data()+i*W);
    }
    mpz_clears(a,b,nullptr);st.sub_coeffs+=P;
    const char *fault=std::getenv("NTT_FOLD_FLAT_TEST_BAD");
    if(fault && std::atoi(fault) && !H.empty()) H[0]^=1;
    while(H.size()>W && std::all_of(H.end()-W,H.end(),[](unsigned long long x){return x==0;}))H.resize(H.size()-W);
    st.t_subtract+=now_s()-ts;
}

/* Independent schoolbook GMP product + classical monic long division, NOT the
   reversed-inverse division used by the candidate and the legacy GPU path. */
static std::vector<unsigned long long> fold_gmp_reference(PolyLayer &L,
    const std::vector<unsigned long long> &G,const std::vector<unsigned long long> &H,
    const std::vector<unsigned long long> &F)
{
    const size_t W=L.W,P=F.size()/W-1;
    auto out=groot_product_gmp(L,G,H);
    if(out.size()/W<=P)return out;
    mpz_t a,b,c;mpz_inits(a,b,c,nullptr);std::vector<unsigned long long> word(W);
    for(size_t i=out.size()/W;i-->P;) {
        words_to_mpz(c,out.data()+i*W,W);
        for(size_t j=0;j<P;++j) {
            words_to_mpz(a,out.data()+(i-P+j)*W,W);words_to_mpz(b,F.data()+j*W,W);
            mpz_submul(a,c,b);mpz_mod(a,a,L.N);mpz_to_words(word,W,a);
            std::copy_n(word.data(),W,out.data()+(i-P+j)*W);
        }
        std::fill_n(out.data()+i*W,W,0ull);
    }
    mpz_clears(a,b,c,nullptr);out.resize(P*W);
    while(out.size()>W && std::all_of(out.end()-W,out.end(),[](unsigned long long x){return x==0;}))out.resize(out.size()-W);
    return out;
}
static bool fold_device_flag(const char *key) {
    const char *v=std::getenv(key);return v && std::atoi(v)!=0;
}
// Two order-sensitive 64-bit checksums of the canonical root input stream.
// Indices are global across roots; this is a diagnostic, not a cryptographic hash.
__host__ __device__ static inline unsigned long long groot_mix64(unsigned long long x) {
    x^=x>>30;x*=0xbf58476d1ce4e5b9ull;x^=x>>27;x*=0x94d049bb133111ebull;return x^(x>>31);
}
__global__ void groot_input_digest_kernel(const unsigned long long *input,size_t count,
    unsigned long long base,unsigned long long *digest) {
    unsigned long long sum=0,xorv=0;
    for(size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;i<count;i+=gridDim.x*(size_t)blockDim.x) {
        const auto index=base+i,value=input[i];
        sum+=groot_mix64(value^groot_mix64(index+0x9e3779b97f4a7c15ull));
        xorv^=groot_mix64(value+groot_mix64(index+0xd1b54a32d192ed03ull));
    }
    __shared__ unsigned long long sums[256],xors[256];
    sums[threadIdx.x]=sum;xors[threadIdx.x]=xorv;__syncthreads();
    for(unsigned d=128;d;d/=2) {
        if(threadIdx.x<d){sums[threadIdx.x]+=sums[threadIdx.x+d];xors[threadIdx.x]^=xors[threadIdx.x+d];}
        __syncthreads();
    }
    if(!threadIdx.x){atomicAdd(digest,sums[0]);atomicXor(digest+1,xors[0]);}
}
static void groot_input_digest_host(const std::vector<unsigned long long> &input,
    unsigned long long &words,unsigned long long *digest) {
    for(const auto value:input) {
        digest[0]+=groot_mix64(value^groot_mix64(words+0x9e3779b97f4a7c15ull));
        digest[1]^=groot_mix64(value+groot_mix64(words+0xd1b54a32d192ed03ull));++words;
    }
}

// Production retains only the validated q/qb and G/reverse reuse policy.
static constexpr unsigned kFoldOwnerReuse=3;
struct FoldDeviceStats {
    bool requested=false,enabled=false;
    unsigned reuse=kFoldOwnerReuse;
    unsigned long long layout_bytes=0,saved_bytes=0;
    std::string fallback="none";
    unsigned long long folds=0,muls=0,sub_coeffs=0,peak_bytes=0,h2d_bytes=0,d2h_bytes=0;
    unsigned long long avoided_h2d_bytes=0,avoided_d2h_bytes=0,checked_words=0;
    bool root_requested=false;
    unsigned long long root_device_trees=0,root_device_words=0,root_checked_words=0;
    unsigned long long root_h2d_avoided=0,root_d2h_avoided=0,root_digest_words=0,root_digest[2]={0,0};
    double root_t_handoff=0;
    double t_setup=0,t_upload=0,t_reverse=0,t_subtract=0,t_readback=0;
};
__global__ void fold_reverse_kernel(const unsigned long long *src,size_t top,
    size_t count,int W,unsigned long long *out) {
    const size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;
    if(i<count*(size_t)W)out[i]=src[(top-i/W)*W+i%W];
}
// Ordinary canonical subtraction; no Montgomery factor enters the remainder.
__global__ void fold_subtract_kernel(const unsigned long long *a,const unsigned long long *b,
    const unsigned long long *n,int W,size_t count,unsigned long long *out,
    unsigned long long *length) {
    const size_t i=blockIdx.x*(size_t)blockDim.x+threadIdx.x;if(i>=count)return;
    a+=i*W;b+=i*W;out+=i*W;
    unsigned long long borrow=0;
    for(int j=0;j<W;++j) {
        const unsigned long long t=a[j]-b[j],b1=a[j]<b[j],v=t-borrow,b2=t<borrow;
        out[j]=v;borrow=b1|b2;
    }
    if(borrow) {
        unsigned long long carry=0;
        for(int j=0;j<W;++j) {
            const unsigned long long t=out[j]+n[j],c1=t<out[j],v=t+carry,c2=v<t;
            out[j]=v;carry=c1|c2;
        }
    }
    unsigned long long any=0;for(int j=0;j<W;++j)any|=out[j];
    if(any)atomicMax(length,(unsigned long long)i+1);
}
#include "stage2/scale_plain.cuh"

static bool gscale_flag(const char *key)
{
    const char *v=std::getenv(key);if(!v)return false;
    if((v[0]!='0' && v[0]!='1') || v[1]){std::fprintf(stderr,"FATAL: %s must be 0 or 1\n",key);std::exit(3);}
    return v[0]=='1';
}

struct FoldDeviceState {
    S4ResidentOwner memory;
    unsigned long long *map=nullptr,*length=nullptr,*modulus=nullptr,*digest=nullptr;
    PolyLayer *layer=nullptr;FoldDeviceStats *stats=nullptr;FoldFlatStats *flat=nullptr;
    size_t P=0,W=0,hcount=0,gcount=0,f=0,inv=0,h=0,g=0,reverse=0,t=0,q=0,qb=0;
    bool active=false,check=false;
    const std::vector<unsigned long long> *host_F=nullptr;
    ~FoldDeviceState(){release();}
    void release() {
        memory.release();
        if(map){CK(cudaFree(map));map=nullptr;}
        if(length){CK(cudaFree(length));length=nullptr;}
        if(modulus){CK(cudaFree(modulus));modulus=nullptr;}
        if(digest){CK(cudaFree(digest));digest=nullptr;}
        active=false;
    }
    bool init(PolyLayer &L,const std::vector<unsigned long long> &F,
              const std::vector<unsigned long long> &inverse,FoldDeviceStats &st,FoldFlatStats &fs) {
        layer=&L;stats=&st;flat=&fs;st.requested=fold_device_flag("NTT_FOLD_DEVICE");
        if(!st.requested)return false;
        const double begin=now_s();W=L.W;P=F.size()/W-1;host_F=&F;
        check=fold_device_flag("NTT_FOLD_DEVICE_CHECK");
        const char *hostpack=std::getenv("NTT_S4_HOSTPACK");
        if(!g_fold_flat || !L.s4 || !L.arena || !g_s4_pack_direct || g_s4_final_readback ||
           g_s4_carry_trace || (hostpack && std::atoi(hostpack))) {st.fallback="backend";return false;}
        if(!P || F.size()!=(P+1)*W || inverse.size()!=(P+1)*W) {st.fallback="shape";return false;}
        if(check && P>64){std::fprintf(stderr,"FATAL: device fold GMP check limited to P<=64\n");std::exit(3);}
        ecm_stage2::FoldOwnerLayout layout;
        if(!ecm_stage2::fold_owner_layout(P,W,kFoldOwnerReuse,layout)) {
            std::fprintf(stderr,"FATAL: device fold owner layout overflow\n");std::exit(3);
        }
        const unsigned long long bytes=layout.bytes;st.layout_bytes=bytes;
        unsigned long long max_mb=640;
        if(const char *e=std::getenv("NTT_FOLD_DEVICE_MAX_MB"))max_mb=std::strtoull(e,nullptr,10);
        if(max_mb>(~0ull>>20) || bytes>(max_mb<<20)) {st.fallback="budget";return false;}
        unsigned long long qN=0;
        if(!ntt_shape_query(P+1,(int)L.S,&qN,nullptr,nullptr,nullptr,nullptr,nullptr)) {st.fallback="ntt_shape";return false;}
        size_t available=0,total=0;CK(cudaMemGetInfo(&available,&total));
        // Finv has normally already built the largest workspace. Reserve its
        // possible remaining growth and 1 GiB for coordinates/frontier/context.
        const size_t target=3ull*qN*8,current=L.arena->workspace.words*8;
        const size_t growth=target>current?target-current:0;
        if(bytes>available || available-bytes<growth+(1ull<<30)) {st.fallback="headroom";return false;}
        if(fold_device_flag("NTT_FOLD_DEVICE_ALLOC_FAIL")){st.fallback="allocation_fixture";return false;}
        memory.words[0]=layout.source_words;memory.words[1]=layout.result_words;
        for(int i=0;i<2;++i) {
            const auto error=cudaMalloc(&memory.data[i],memory.words[i]*8);
            if(error==cudaErrorMemoryAllocation){cudaGetLastError();memory.release();st.fallback="allocation";return false;}
            CK(error);
        }
        CK(cudaMalloc(&map,24));CK(cudaMalloc(&length,8));CK(cudaMalloc(&modulus,W*8));
        CK(cudaMalloc(&digest,16));CK(cudaMemsetAsync(digest,0,16));
        f=layout.f;inv=layout.inv;h=layout.h;g=layout.g;reverse=layout.reverse;
        t=layout.t;q=layout.q;qb=layout.qb;
        upload(f,F);upload(inv,inverse);
        std::vector<unsigned long long> hn(W);mpz_to_words(hn,W,L.N);
        CK(cudaMemcpy(modulus,hn.data(),W*8,cudaMemcpyHostToDevice));st.h2d_bytes+=W*8;
        st.saved_bytes=ecm_stage2::owner_bytes(P,W,0)-bytes;
        st.peak_bytes=bytes;st.enabled=active=true;st.t_setup=now_s()-begin;
        return true;
    }
    void upload(size_t offset,const std::vector<unsigned long long> &v) {
        if(v.empty())return;
        const double begin=now_s();CK(cudaMemcpy(memory.data[0]+offset,v.data(),v.size()*8,cudaMemcpyHostToDevice));
        stats->h2d_bytes+=v.size()*8;stats->t_upload+=now_s()-begin;
    }
    void digest_root(size_t offset,size_t words) {
        if(words) {
            const unsigned blocks=(unsigned)std::min((size_t)1024,(words+255)/256);
            groot_input_digest_kernel<<<blocks,256>>>(memory.data[0]+offset,words,stats->root_digest_words,digest);
            CK(cudaGetLastError());stats->root_digest_words+=words;
        }
    }
    void read_digest() {
        CK(cudaMemcpy(stats->root_digest,digest,16,cudaMemcpyDeviceToHost));stats->d2h_bytes+=16;
    }
    void accept_device_root(const unsigned long long *src,size_t count,bool seed_root) {
        if(!active || !src || count>P+1 || !count ||
           (src!=layer->s4->d_rawA && src!=layer->s4->d_rawB) || count*W>layer->s4->raw_capacity(src)) {
            std::fprintf(stderr,"FATAL: root-to-fold source is not a live bounded raw lease\n");std::exit(3);
        }
        const bool verify=fold_device_flag("NTT_GROOT_TO_FOLD_CHECK");
        if(verify && P>64){std::fprintf(stderr,"FATAL: root-to-fold check limited to P<=64\n");std::exit(3);}
        const double begin=now_s();const size_t dest=seed_root?h:g,words=count*W;
        std::vector<unsigned long long> expected;
        if(verify) {
            expected.resize(words);CK(cudaMemcpy(expected.data(),src,words*8,cudaMemcpyDeviceToHost));
            stats->d2h_bytes+=words*8;
        }
        CK(cudaMemcpyAsync(memory.data[0]+dest,src,words*8,cudaMemcpyDeviceToDevice));
        if(fold_device_flag("NTT_GROOT_TO_FOLD_TEST_BAD") && !stats->root_device_trees)
            CK(cudaMemsetAsync(memory.data[0]+dest,0xff,8));
        if(verify) {
            auto actual=read(dest,count);stats->root_checked_words+=2*words;
            if(actual!=expected){std::fprintf(stderr,"FATAL: root-to-fold word mismatch\n");std::exit(3);}
        }
        if(seed_root)hcount=count;else gcount=count;
        digest_root(dest,words);++stats->root_device_trees;stats->root_device_words+=words;
        stats->root_h2d_avoided+=words*8;stats->root_d2h_avoided+=words*8;
        stats->root_t_handoff+=now_s()-begin;
    }
    void seed(const std::vector<unsigned long long> &H) {
        if(H.size()>(P+1)*W || H.size()%W){std::fprintf(stderr,"FATAL: device fold seed shape\n");std::exit(3);}
        upload(h,H);hcount=H.size()/W;digest_root(h,H.size());
    }
    std::vector<unsigned long long> read(size_t offset,size_t count) {
        std::vector<unsigned long long> v(count*W);const double begin=now_s();
        if(!v.empty())CK(cudaMemcpy(v.data(),memory.data[0]+offset,v.size()*8,cudaMemcpyDeviceToHost));
        stats->d2h_bytes+=v.size()*8;stats->t_readback+=now_s()-begin;return v;
    }
    void mul(size_t a,size_t ma,size_t b,size_t mb,size_t dest,size_t count) {
        unsigned long long metadata[3]={(unsigned long long)a,(unsigned long long)b,(unsigned long long)dest};
        CK(cudaMemcpy(map,metadata,24,cudaMemcpyHostToDevice));stats->h2d_bytes+=24;
        S4DeviceBatch batch{memory.data[0],map,memory.data[1],metadata,memory.words[0],memory.words[1],1,&memory};
        std::vector<unsigned long long> unused;
        const double begin=now_s();poly_mul_batch_modN(*layer,nullptr,nullptr,ma,mb,1,unused,BC_FOLD,nullptr,0,count,&batch);
        flat->t_multiply+=now_s()-begin;++flat->muls;++stats->muls;
        stats->avoided_h2d_bytes+=16ull*std::max(ma,mb)*W;
        stats->avoided_d2h_bytes+=8ull*(g_s4_output_window?count:2*std::max(ma,mb)-1)*W;
    }
    void reverse_copy(size_t src,size_t top,size_t count) {
        const double begin=now_s();fold_reverse_kernel<<<(unsigned)((count*W+255)/256),256>>>(
            memory.data[1]+src,top,count,(int)W,memory.data[0]+reverse);CK(cudaGetLastError());
        stats->t_reverse+=now_s()-begin;
    }
    void step(const std::vector<unsigned long long> &G) {
        if(G.size()>(P+1)*W || G.size()%W){std::fprintf(stderr,"FATAL: device fold G shape\n");std::exit(3);}
        upload(g,G);digest_root(g,G.size());step_loaded(G.size()/W,check?&G:nullptr);
    }
    void step_loaded(size_t ng,const std::vector<unsigned long long> *host_G=nullptr) {
        ++stats->folds;++flat->folds;
        std::vector<unsigned long long> expected;
        if(check) {
            auto G=host_G?*host_G:read(g,ng);expected=fold_gmp_reference(*layer,G,read(h,hcount),*host_F);
        }
        if(!ng || !hcount){hcount=0;return;}
        const size_t nT=ng+hcount-1;
        // Digest and gather read G before reverse overwrites its slot, on the
        // same default stream. Diagnostic GMP inputs are already independent.
        mul(g,ng,h,hcount,t,nT);
        if(nT<=P) {
            CK(cudaMemcpyAsync(memory.data[0]+h,memory.data[1]+t,nT*W*8,cudaMemcpyDeviceToDevice));hcount=nT;
        } else {
            const size_t k=nT-P;
            reverse_copy(t,nT-1,k);mul(reverse,k,inv,k,q,k);
            // Reverse reads all k<=P+1 q coefficients before qb overwrites q.
            // Oracle slots hold independent digit/result snapshots.
            reverse_copy(q,k-1,k);mul(reverse,k,f,P+1,qb,P);
            const double begin=now_s();CK(cudaMemsetAsync(length,0,8));
            fold_subtract_kernel<<<(unsigned)((P+127)/128),128>>>(memory.data[1]+t,memory.data[1]+qb,
                modulus,(int)W,P,memory.data[0]+h,length);CK(cudaGetLastError());
            unsigned long long n=0;CK(cudaMemcpy(&n,length,8,cudaMemcpyDeviceToHost));
            hcount=std::max((size_t)1,(size_t)n);stats->d2h_bytes+=8;stats->sub_coeffs+=P;flat->sub_coeffs+=P;
            stats->t_subtract+=now_s()-begin;flat->t_subtract+=now_s()-begin;
        }
        if(fold_device_flag("NTT_FOLD_DEVICE_TEST_BAD"))CK(cudaMemset(memory.data[0]+h,0xff,8));
        if(check) {
            auto actual=read(h,hcount);stats->checked_words+=actual.size();
            if(actual!=expected){std::fprintf(stderr,"FATAL: device fold GMP mismatch P=%llu\n",(unsigned long long)P);std::exit(3);}
        }
    }
    void scale(mpz_srcptr factor,PlainScaleStats &st,bool verify,bool poison) {
        if(!active || !hcount || q+W>memory.words[1] || h+hcount*W>memory.words[0]) {
            std::fprintf(stderr,"FATAL: Gamma device owner lease invalid\n");std::exit(3);
        }
        s2g_plain_scale(W,hcount,memory.data[0]+h,memory.data[1]+q,modulus,layer->N,factor,st,verify,poison);
        stats->h2d_bytes+=st.h2d_bytes;stats->d2h_bytes+=st.check_d2h_bytes;
    }
    void finish(std::vector<unsigned long long> &H) {H=read(h,hcount);read_digest();release();}
};

static void fold_flat_fixture(PolyLayer &L)
{
    const size_t W=L.W;unsigned long long cases=0,words=0;
    mpz_t z;mpz_init(z);std::vector<unsigned long long> word(W);
    for(size_t P:{1u,2u,3u,8u,17u}) {
        std::vector<unsigned long long> F(W,0);F[0]=1;
        for(size_t j=0;j<P;++j) {
            mpz_set_ui(z,j+2);mpz_neg(z,z);mpz_mod(z,z,L.N);mpz_to_words(word,W,z);
            std::vector<unsigned long long> leaf(2*W);std::copy_n(word.data(),W,leaf.data());leaf[W]=1;
            F=groot_product_gmp(L,F,leaf);
        }
        CPoly rev=cp_from_flat(F,P,W);std::reverse(rev.begin(),rev.end());
        auto inverse=cp_to_flat(cp_inv_series(rev,P+1,L),W);
        for(size_t mode=0;mode<4;++mode) {
            const size_t nh=mode==1?1:mode==2?P:P+1,ng=mode==1?1:P+1;
            auto make=[&](size_t n,bool zero){
                std::vector<unsigned long long> v(n*W);
                for(size_t j=0;j<n;++j) {
                    if(zero)mpz_set_ui(z,0);
                    else if(mode==2)mpz_sub_ui(z,L.N,j%3+1);
                    else {mpz_set_ui(z,17*j+3);mpz_mul_2exp(z,z,(unsigned long)((j*31)%L.S));}
                    mpz_mod(z,z,L.N);mpz_to_words(word,W,z);std::copy_n(word.data(),W,v.data()+j*W);
                }return v;
            };
            auto H=make(nh,mode==0),G=make(ng,false);
            FoldFlatStats stats;
            // mode 3 exercises repeated folds and a shorter final G with the same inverse.
            for(size_t round=0;round<(mode==3?3u:1u);++round) {
                if(round==2)G.resize(std::min(G.size(),2*W));
                auto expected=fold_gmp_reference(L,G,H,F);
                fold_flat_step(L,G,H,F,inverse,stats);
                if(H!=expected) {std::fprintf(stderr,"%s: FATAL: flat fold GMP mismatch P=%llu mode=%llu round=%llu\n",NTT_PROBE_NAME,(unsigned long long)P,(unsigned long long)mode,(unsigned long long)round);std::exit(3);}
                ++cases;words+=H.size();
            }
        }
    }
    mpz_clear(z);
    stage2_log::print(stage2_log::debug, "fold_flat_fixture: cases=%llu words=%llu bad=0 (GMP schoolbook/monic long division, zero/near-N/short/padded/repeated folds)\n",cases,words);
}

static void fold_device_fixture(PolyLayer &L)
{
    const size_t W=L.W;unsigned long long cases=0,words=0;
    mpz_t z;mpz_init(z);std::vector<unsigned long long> word(W);
    for(size_t P:{1u,2u,3u,8u,17u}) {
        std::vector<unsigned long long> F(W,0);F[0]=1;
        for(size_t j=0;j<P;++j) {
            mpz_set_ui(z,j+2);mpz_neg(z,z);mpz_mod(z,z,L.N);mpz_to_words(word,W,z);
            std::vector<unsigned long long> leaf(2*W);std::copy_n(word.data(),W,leaf.data());leaf[W]=1;
            F=groot_product_gmp(L,F,leaf);
        }
        CPoly rev=cp_from_flat(F,P,W);std::reverse(rev.begin(),rev.end());
        auto inverse=cp_to_flat(cp_inv_series(rev,P+1,L),W);
        for(size_t mode=0;mode<4;++mode) {
            const size_t nh=mode==1?1:mode==2?P:P+1,ng=mode==1?1:P+1;
            auto make=[&](size_t n,bool zero){
                std::vector<unsigned long long> v(n*W);
                for(size_t j=0;j<n;++j) {
                    if(zero)mpz_set_ui(z,0);
                    else if(mode==2)mpz_sub_ui(z,L.N,j%3+1);
                    else {mpz_set_ui(z,17*j+3);mpz_mul_2exp(z,z,(unsigned long)((j*31)%L.S));}
                    mpz_mod(z,z,L.N);mpz_to_words(word,W,z);std::copy_n(word.data(),W,v.data()+j*W);
                }return v;
            };
            auto H=make(nh,mode==0),G=make(ng,false);
            FoldFlatStats stats;FoldDeviceStats fd;FoldDeviceState state;
            if(!state.init(L,F,inverse,fd,stats)){std::fprintf(stderr,"FATAL: device fold fixture fell back: %s\n",fd.fallback.c_str());std::exit(3);}
            state.seed(H);unsigned long long digest_words=0,expected_digest[2]={0,0};
            groot_input_digest_host(H,digest_words,expected_digest);
            // mode 3 exercises repeated folds and a shorter final G with the same inverse.
            for(size_t round=0;round<(mode==3?3u:1u);++round) {
                if(round==2)G.resize(std::min(G.size(),2*W));
                auto expected=fold_gmp_reference(L,G,H,F);
                groot_input_digest_host(G,digest_words,expected_digest);
                state.step(G);H=state.read(state.h,state.hcount);
                if(H!=expected) {std::fprintf(stderr,"%s: FATAL: flat fold GMP mismatch P=%llu mode=%llu round=%llu\n",NTT_PROBE_NAME,(unsigned long long)P,(unsigned long long)mode,(unsigned long long)round);std::exit(3);}
                ++cases;words+=H.size();
            }
            state.read_digest();
            if(fd.root_digest_words!=digest_words || fd.root_digest[0]!=expected_digest[0] || fd.root_digest[1]!=expected_digest[1]) {
                std::fprintf(stderr,"FATAL: device root input digest mismatch\n");std::exit(3);
            }
        }
    }
    mpz_clear(z);
    stage2_log::print(stage2_log::debug, "fold_device_fixture: cases=%llu words=%llu bad=0 (GMP schoolbook/monic long division, zero/near-N/short/padded/repeated folds)\n",cases,words);
}


/* Scaled remainder descent: one sibling multiply per nontrivial child.
   Independent GMP node/Horner checks are opt-in, excluded from clean timings. */
struct ScaledStats {
    unsigned long long levels=0, mul_calls=0, mul_pairs=0, copies=0, zeros=0;
    unsigned long long states=0, words=0, leaves=0, checked_states=0, checked_words=0;
    unsigned long long frontier_peak_bytes=0, pack_peak_bytes=0, root_inverse_reused=0;
    unsigned long long root_divisions=0;
};

/* Independent GMP oracle: original H mod monic F by long division, then a triangular
   solve of rev_(d-1)(remainder)/rev_d(F). No NTT or scaled child recurrence is used. */
static std::vector<unsigned long long> scaled_state_gmp(
    const CPoly &H, const std::vector<unsigned long long> &F, size_t d, PolyLayer &L)
{
    const size_t W=L.W;
    if(F.size()!=(d+1)*W || !poly_is_one(F.data()+d*W,W)) {
        std::fprintf(stderr,"%s: FATAL: scaled oracle needs a monic node\n",NTT_PROBE_NAME);
        std::exit(3);
    }
    if(!d) return {};
    CPoly r=H;
    if(r.size()<d) cp_resize(r,d,W);
    mpz_t q,f,t,u;
    mpz_inits(q,f,t,u,nullptr);
    for(size_t i=r.size();i-- >d;) {
        words_to_mpz(q,r[i].data(),W);
        for(size_t j=0;j<=d;++j) {
            words_to_mpz(f,F.data()+j*W,W);
            words_to_mpz(t,r[i-d+j].data(),W);
            mpz_submul(t,q,f);mpz_mod(t,t,L.N);
            mpz_to_words(r[i-d+j],W,t);
        }
    }
    std::vector<unsigned long long> state(d*W,0);
    std::vector<unsigned long long> word(W,0);
    for(size_t i=0;i<d;++i) {
        words_to_mpz(t,r[d-1-i].data(),W);
        for(size_t j=1;j<=i;++j) {
            words_to_mpz(f,F.data()+(d-j)*W,W);
            words_to_mpz(u,state.data()+(i-j)*W,W);
            mpz_submul(t,f,u);
        }
        mpz_mod(t,t,L.N);mpz_to_words(word,W,t);
        std::copy(word.begin(),word.end(),state.begin()+i*W);
    }
    mpz_clears(q,f,t,u,nullptr);
    return state;
}

static void descent_scaled(PolyLayer &L,
    const std::vector<std::vector<unsigned long long>> &Ft,
    const std::vector<size_t> &Fdeg, size_t Fpad, const CPoly &H,
    const CPoly *cached_finv, std::vector<std::vector<unsigned long long>> &values,
    ScaledStats &st, int cat, bool check)
{
    const size_t W=L.W,P=Fdeg[1];
    if(!L.s4 || !P || Ft.size()<2*Fpad || Fdeg.size()<2*Fpad ||
       Ft[1].size()!=(P+1)*W || !poly_is_one(Ft[1].data()+P*W,W) || (check && P>512)) {
        std::fprintf(stderr,"%s: FATAL: invalid scaled descent shape (GMP check limited to 512 leaves)\n",NTT_PROBE_NAME);
        std::exit(3);
    }
    CPoly local_inv, h=H;
    if(h.size()>P) {
        h=cp_mod(h,cp_from_flat(Ft[1],P,W),L); ++st.root_divisions;
    }
    const CPoly *inv=cached_finv;
    if(!inv || inv->size()<P) {
        CPoly rev;
        cp_resize(rev,P+1,W);
        for(size_t i=0;i<=P;++i)
            std::copy_n(Ft[1].data()+(P-i)*W,W,rev[i].begin());
        local_inv=cp_inv_series(rev,P,L);inv=&local_inv;
    } else ++st.root_inverse_reused;
    std::vector<unsigned long long> A(P*W,0),B(P*W,0),root;
    for(size_t i=0;i<P;++i) {
        if(P-1-i<h.size()) std::copy(h[P-1-i].begin(),h[P-1-i].end(),A.begin()+i*W);
        std::copy((*inv)[i].begin(),(*inv)[i].end(),B.begin()+i*W);
    }
    flat_mul_batch(L,A,P,B,P,1,root,cat,0,P);
    ++st.mul_calls;++st.mul_pairs;
    st.pack_peak_bytes=std::max(st.pack_peak_bytes,8ull*(A.capacity()+B.capacity()+root.capacity()));
    std::vector<unsigned long long>().swap(A);std::vector<unsigned long long>().swap(B);
    std::vector<std::vector<unsigned long long>> cur(1);
    cur[0].swap(root);
    auto validate=[&](const std::vector<std::vector<unsigned long long>> &front,size_t base) {
        for(size_t i=0;i<front.size();++i) {
            const size_t node=base+i,d=Fdeg[node];
            if(front[i].size()!=d*W) {std::fprintf(stderr,"%s: FATAL: scaled state length\n",NTT_PROBE_NAME);std::exit(3);}
            ++st.states;st.words+=d*W;
            if(check) {
                auto want=scaled_state_gmp(H,Ft[node],d,L);
                if(front[i]!=want) {std::fprintf(stderr,"%s: FATAL: scaled GMP node mismatch node=%llu degree=%llu\n",NTT_PROBE_NAME,(unsigned long long)node,(unsigned long long)d);std::exit(3);}
                ++st.checked_states;st.checked_words+=d*W;
            }
        }
    };
    validate(cur,1);
    const double t0=now_s();
    for(size_t base=1;base<Fpad;base*=2) {
        const double tl0=now_s();
        const size_t nbase=base*2;
        std::vector<std::vector<unsigned long long>> next(cur.size()*2);
        std::map<std::pair<size_t,size_t>,std::vector<std::pair<size_t,size_t>>> groups;
        for(size_t j=0;j<cur.size();++j) for(size_t side=0;side<2;++side) {
            const size_t slot=2*j+side,ci=nbase+slot,si=ci^1;
            const size_t a=Fdeg[ci],b=Fdeg[si];
            if(a+b!=Fdeg[base+j]) {std::fprintf(stderr,"%s: FATAL: scaled degree conservation\n",NTT_PROBE_NAME);std::exit(3);}
            if(!a) {++st.zeros;continue;}
            if(!b) {next[slot]=cur[j];++st.copies;continue;}
            groups[{a,b}].push_back({j,slot});
        }
        for(const auto &g:groups) {
            const size_t a=g.first.first,b=g.first.second,ma=a+b,mb=b+1,nb=g.second.size();
            A.assign(nb*ma*W,0);B.assign(nb*mb*W,0);
            std::vector<unsigned long long> product;
            for(size_t s=0;s<nb;++s) {
                const size_t j=g.second[s].first,slot=g.second[s].second,si=(nbase+slot)^1;
                std::copy(cur[j].begin(),cur[j].end(),A.begin()+s*ma*W);
                if(Ft[si].size()!=mb*W) {std::fprintf(stderr,"%s: FATAL: scaled sibling length\n",NTT_PROBE_NAME);std::exit(3);}
                for(size_t q=0;q<mb;++q)
                    std::copy_n(Ft[si].data()+(b-q)*W,W,B.begin()+(s*mb+q)*W);
            }
            flat_mul_batch(L,A,ma,B,mb,nb,product,cat,b,a);
            ++st.mul_calls;st.mul_pairs+=nb;
            st.pack_peak_bytes=std::max(st.pack_peak_bytes,8ull*(A.capacity()+B.capacity()+product.capacity()));
            for(size_t s=0;s<nb;++s)
                next[g.second[s].second].assign(product.begin()+s*a*W,product.begin()+(s+1)*a*W);
        }
        unsigned long long bytes=0;
        for(const auto &v:cur) bytes+=8ull*v.capacity();
        for(const auto &v:next) bytes+=8ull*v.capacity();
        st.frontier_peak_bytes=std::max(st.frontier_peak_bytes,bytes);
        cur.swap(next);++st.levels;validate(cur,nbase);
        if(g_s4_batched_progress)
            stage2_log::print(stage2_log::debug, "scaled_progress: level=%llu nodes=%llu groups=%llu t_level=%.3f t_total=%.3f\n",st.levels,(unsigned long long)cur.size(),(unsigned long long)groups.size(),now_s()-tl0,now_s()-t0);
    }
    values.assign(P,std::vector<unsigned long long>(W,0));
    size_t leaf=0;
    for(size_t i=0;i<Fpad;++i) if(Fdeg[Fpad+i]) {
        if(Fdeg[Fpad+i]!=1 || leaf>=P) {std::fprintf(stderr,"%s: FATAL: scaled non-linear leaf\n",NTT_PROBE_NAME);std::exit(3);}
        values[leaf++]=std::move(cur[i]);
    }
    if(leaf!=P) {std::fprintf(stderr,"%s: FATAL: scaled leaf count\n",NTT_PROBE_NAME);std::exit(3);}
    st.leaves+=leaf;
}

static void scaled_descent_fixture(PolyLayer &L)
{
    if(!L.s4) {std::fprintf(stderr,"%s: FATAL: scaled fixture requires S4\n",NTT_PROBE_NAME);std::exit(3);}
    const size_t W=L.W;
    ScaledStats sum;
    unsigned long long cases=0;
    mpz_t x;mpz_init(x);
    for(size_t P:{1u,2u,3u,5u,7u,8u,9u,13u,16u,23u}) for(size_t padding=0;padding<3;++padding) {
        size_t pad=1;while(pad<P) pad*=2;
        std::vector<std::vector<unsigned long long>> Ft(2*pad);
        std::vector<size_t> deg(2*pad,0);
        for(size_t i=0;i<pad;++i) {Ft[pad+i].assign(W,0);Ft[pad+i][0]=1;}
        for(size_t i=0;i<P;++i) {
            const size_t slot=padding==0?i:padding==1?pad-P+i:(i*5)%pad;
            auto &f=Ft[pad+slot];f.assign(2*W,0);
            mpz_set_ui(x,1+i*17);mpz_neg(x,x);mpz_mod(x,x,L.N);
            std::vector<unsigned long long> word(W,0);mpz_to_words(word,W,x);
            std::copy(word.begin(),word.end(),f.begin());f[W]=1;deg[pad+slot]=1;
        }
        /* Build the fixture tree with independent GMP convolution. */
        mpz_t a,b,t;mpz_inits(a,b,t,nullptr);
        for(size_t node=pad;--node;) {
            const size_t dl=deg[2*node],dr=deg[2*node+1],d=dl+dr;
            deg[node]=d;Ft[node].assign((d+1)*W,0);
            for(size_t i=0;i<=d;++i) {
                mpz_set_ui(t,0);
                for(size_t j=0;j<=dl;++j) if(i>=j && i-j<=dr) {
                    words_to_mpz(a,Ft[2*node].data()+j*W,W);
                    words_to_mpz(b,Ft[2*node+1].data()+(i-j)*W,W);mpz_addmul(t,a,b);
                }
                mpz_mod(t,t,L.N);std::vector<unsigned long long> word(W,0);mpz_to_words(word,W,t);
                std::copy(word.begin(),word.end(),Ft[node].begin()+i*W);
            }
        }
        CPoly rev;cp_resize(rev,P+1,W);
        for(size_t i=0;i<=P;++i) std::copy_n(Ft[1].data()+(P-i)*W,W,rev[i].begin());
        const CPoly inv=cp_inv_series(rev,P+1,L);
        for(size_t kind=0;kind<5;++kind) {
            CPoly H;cp_resize(H,kind==4?P+1:P,W);
            for(size_t i=0;i<H.size();++i) {
                if(kind==0) mpz_set_ui(x,0);
                else if(kind==1) mpz_set_ui(x,i==0?1:0);
                else if(kind==2) mpz_sub_ui(x,L.N,1);
                else {mpz_set_ui(x,13+i*191);mpz_mod(x,x,L.N);}
                mpz_to_words(H[i],W,x);
            }
            if(kind==1) cp_trim(H);
            std::vector<std::vector<unsigned long long>> values;
            descent_scaled(L,Ft,deg,pad,H,kind==0?nullptr:&inv,values,sum,-1,true);
            /* Each degree-one node's independent remainder is also Horner(H,x). */
            size_t leaf=0;
            for(size_t i=0;i<pad;++i) if(deg[pad+i]) {
                words_to_mpz(a,Ft[pad+i].data(),W);mpz_neg(a,a);mpz_mod(a,a,L.N);mpz_set_ui(t,0);
                for(size_t k=H.size();k--;) {mpz_mul(t,t,a);words_to_mpz(b,H[k].data(),W);mpz_add(t,t,b);mpz_mod(t,t,L.N);}
                words_to_mpz(b,values[leaf++].data(),W);
                if(mpz_cmp(t,b)) {std::fprintf(stderr,"%s: FATAL: scaled Horner mismatch\n",NTT_PROBE_NAME);std::exit(3);}
            }
            ++cases;
        }
        mpz_clears(a,b,t,nullptr);
    }
    mpz_clear(x);
    stage2_log::print(stage2_log::debug, "scaled_fixture: cases=%llu states=%llu words=%llu leaves=%llu reused=%llu root_divisions=%llu bad=0\n",cases,sum.checked_states,sum.checked_words,sum.leaves,sum.root_inverse_reused,sum.root_divisions);
}

static void s4_flat_input_check(PolyLayer &L, size_t slices = 3)
{
    /* Regression for a unit carry crossing many warps/blocks.  Compare every digit,
       not just the final modular remainder, and keep batched slices independent. */
    const size_t carry_n = 8205, carry_nb = 3;
    unsigned long long carry_bad = 0;
    for (const int bpw : {14, 26}) {
        const unsigned long long mask = (1ull << bpw) - 1;
        std::vector<unsigned long long> raw(carry_n*carry_nb), expected(raw.size()), actual(raw.size());
        for (size_t s=0; s<carry_nb; ++s) {
            const size_t start = s==0 ? 0 : s==1 ? 31 : 255;
            for (size_t j=start; j<start+4097; ++j) raw[s*carry_n+j] = mask;
            raw[s*carry_n+start] = 1ull << 60;
            /* A second generator after a kill and a partial warp at the array end. */
            raw[s*carry_n+7001] = mask+1;
            for (size_t j=7002; j<carry_n-1; ++j) raw[s*carry_n+j] = mask;
            unsigned long long carry = 0;
            for (size_t j=0; j<carry_n; ++j) {
                const auto v = raw[s*carry_n+j] + carry;
                expected[s*carry_n+j] = v & mask;
                carry = v >> bpw;
            }
        }
        unsigned long long *di = nullptr, *doo = nullptr;
        CK(cudaMalloc(&di, raw.size()*8)); CK(cudaMalloc(&doo, raw.size()*8));
        CK(cudaMemcpy(di, raw.data(), raw.size()*8, cudaMemcpyHostToDevice));
        carry_cone_kernel<5><<<dim3((unsigned int)((carry_n+255)/256), (unsigned int)carry_nb), 256>>>(
            di, doo, carry_n, bpw, carry_n);
        CK(cudaGetLastError());
        CK(cudaMemcpy(actual.data(), doo, actual.size()*8, cudaMemcpyDeviceToHost));
        CK(cudaFree(di)); CK(cudaFree(doo));
        for (size_t i=0; i<actual.size(); ++i) if (actual[i] != expected[i]) ++carry_bad;
    }
    stage2_log::print(stage2_log::debug, "carry_chain_check: cases=6 words=49230 bad=%llu (14/26-bit radix, 4097-digit chains)\n", carry_bad);
    if (carry_bad) std::exit(3);
    unsigned long long cases = 0, words = 0, bad = 0;
    mpz_t av, bv, sum, term;
    mpz_inits(av, bv, sum, term, nullptr);
    const size_t W = L.W;
    for (const auto dims : {std::pair<size_t,size_t>{3,3}, {2,5}, {5,2}, {1,4}, {4,1}}) {
        const size_t ma = dims.first, mb = dims.second, nb = slices, nc = ma + mb - 1;
        std::vector<unsigned long long> a(nb*ma*W), b(nb*mb*W), expected(nb*nc*W);
        auto fill = [&](std::vector<unsigned long long> &v) {
            for (size_t i=0; i<v.size()/W; ++i) {
                mpz_sub_ui(av, L.N, (unsigned long)(i+1));
                mpz_mod(av, av, L.N);
                std::vector<unsigned long long> coef(W);
                mpz_to_words(coef, W, av);
                std::copy(coef.begin(), coef.end(), v.begin() + (long)(i*W));
            }
        };
        fill(a); fill(b);
        for (size_t s=0; s<nb; ++s) for (size_t k=0; k<nc; ++k) {
            mpz_set_ui(sum, 0);
            for (size_t i=0; i<ma; ++i) if (k>=i && k-i<mb) {
                words_to_mpz(av, &a[(s*ma+i)*W], W);
                words_to_mpz(bv, &b[(s*mb+k-i)*W], W);
                mpz_mul(term, av, bv); mpz_add(sum, sum, term);
            }
            mpz_mod(sum, sum, L.N);
            std::vector<unsigned long long> coef(W);
            mpz_to_words(coef, W, sum);
            std::copy(coef.begin(), coef.end(), expected.begin() + (long)((s*nc+k)*W));
        }
        for (int alias=0; alias<4; ++alias) {
            if (alias==3 && ma!=mb) continue;
            auto aa=a, bb=b;
            std::vector<unsigned long long> separate;
            auto &result = alias==1 || alias==3 ? aa : alias==2 ? bb : separate;
            flat_mul_batch(L, aa, ma, alias==3 ? aa : bb, mb, nb, result, -1);
            ++cases; words += result.size();
            if (result != expected) ++bad;
        }
    }
    mpz_clears(av, bv, sum, term, nullptr);
    stage2_log::print(stage2_log::debug, "%s: cases=%llu words=%llu bad=%llu (GMP, short/full strides, A/B/both aliases)\n",
                slices==3 ? "s4_flat_input_check" : "s4_final_readback_check", cases, words, bad);
    if (bad) std::exit(3);
}

/* Gate-only arbitrary windows, including nonzero FIRST and empty output. The reference is
   an independent coefficient convolution from the original inputs, never assembled NTT digits. */
static void s4_output_window_check(PolyLayer &L)
{
    const size_t W=L.W, nb=5;
    unsigned long long cases=0,words=0,bad=0,canonical_cases=0;
    mpz_t av,bv,term,sum,inv;
    mpz_inits(av,bv,term,sum,inv,nullptr);
    for(const auto dims : {std::pair<size_t,size_t>{3,3},{2,5},{5,2},{1,4},{4,1},{1,1}}) {
        const size_t ma=dims.first,mb=dims.second,nc=ma+mb-1;
        std::vector<unsigned long long> a(nb*ma*W),b(nb*mb*W);
        for(int operand=0;operand<2;++operand) {
            auto &v=operand ? b : a;
            for(size_t i=0;i<v.size()/W;++i) {
                mpz_sub_ui(av,L.N,(unsigned long)(operand ? 2*i+3 : i+1));
                if(i%7==0) mpz_set_ui(av,0);
                mpz_mod(av,av,L.N);
                std::vector<unsigned long long> c(W);mpz_to_words(c,W,av);
                std::copy(c.begin(),c.end(),v.begin()+(long)(i*W));
            }
        }
        std::vector<std::pair<size_t,size_t>> windows{{0,nc},{0,0},{nc,0},{0,1},{nc-1,1}};
        if(nc>1) windows.push_back({1,nc-1});
        if(nc>=4) windows.push_back({2,2});
        for(int alias=0;alias<4;++alias) {
            if(alias==3 && ma!=mb) continue;
            const auto &oracle_b=alias==3 ? a : b;
            std::vector<unsigned long long> expected(nb*nc*W);
            for(size_t s=0;s<nb;++s) for(size_t k=0;k<nc;++k) {
                mpz_set_ui(sum,0);
                for(size_t i=0;i<ma;++i) if(k>=i && k-i<mb) {
                    words_to_mpz(av,&a[(s*ma+i)*W],W);
                    words_to_mpz(bv,&oracle_b[(s*mb+k-i)*W],W);
                    mpz_mul(term,av,bv);mpz_add(sum,sum,term);
                }
                mpz_mod(sum,sum,L.N);
                std::vector<unsigned long long> c(W);mpz_to_words(c,W,sum);
                std::copy(c.begin(),c.end(),expected.begin()+(long)((s*nc+k)*W));
            }
            for(const auto window : windows) {
                const size_t first=window.first,count=window.second;
                auto aa=a,bb=b;
                std::vector<unsigned long long> separate,want(nb*count*W);
                auto &out=alias==1 || alias==3 ? aa : alias==2 ? bb : separate;
                for(size_t s=0;s<nb;++s)
                    std::copy(expected.begin()+(long)((s*nc+first)*W),
                              expected.begin()+(long)((s*nc+first+count)*W),want.begin()+(long)(s*count*W));
                flat_mul_batch(L,aa,ma,alias==3 ? aa : bb,mb,nb,out,-1,first,count);
                ++cases;words+=out.size();if(out!=want) ++bad;
            }
        }
        std::vector<unsigned long long> empty_a,empty_b,empty_out(7,123);
        flat_mul_batch(L,empty_a,ma,empty_b,mb,0,empty_out,-1,0,0);
        ++cases;if(!empty_out.empty()) ++bad;
    }
    /* A constant series is shorter than the requested Newton prefix at its first step. */
    CPoly constant;cp_resize(constant,1,W);constant[0][0]=2;
    const CPoly inverse=cp_inv_series(constant,5,L);
    mpz_set_ui(av,2);mpz_invert(inv,av,L.N);
    std::vector<unsigned long long> ci(W);mpz_to_words(ci,W,inv);
    ++cases;words+=inverse.size()*W;
    if(inverse.size()!=5 || inverse[0]!=ci) ++bad;
    for(size_t i=1;i<inverse.size();++i) if(!cp_coeff_zero(inverse[i])) ++bad;

    /* A bound violation in an OMITTED source slot must still be counted, including count=0.
       Use a separate diagnostic buffer so this intentional fault cannot poison the real run. */
    unsigned long long n=0,sb=0,sw=0,ss=0,os=0;int bpw=0;
    if(!ntt_shape_query(5,(int)L.S,&n,&bpw,&sb,&sw,&ss,&os)) std::exit(3);
    S4Reduce &R=*L.s4->red;
    S4Reduce::Shape *S=s4_shape_init(R,5,sb,ss,sw,bpw);
    const unsigned long long source_slots=9,slices=2,raw_stride=source_slots*sw;
    std::vector<unsigned long long> digits(slices*raw_stride,0);
    const auto top=sb-(sw-1)*(unsigned long long)bpw;
    digits[raw_stride+2*sw+sw-1]=1ull<<top; // source k=2, outside both tested windows
    unsigned long long *dd=nullptr,*dout=nullptr,*dbad=nullptr;
    CK(cudaMalloc(&dd,digits.size()*8));CK(cudaMalloc(&dout,(slices*W+2)*8));CK(cudaMalloc(&dbad,8));
    CK(cudaMemcpy(dd,digits.data(),digits.size()*8,cudaMemcpyHostToDevice));
    for(const auto window : {std::pair<unsigned long long,unsigned long long>{7,1},{0,0}}) {
        const unsigned long long first=window.first,count=window.second;
        std::vector<unsigned long long> canary(slices*W+2,0xfeedfacedeadbeefull),got(canary.size());
        CK(cudaMemcpy(dout,canary.data(),canary.size()*8,cudaMemcpyHostToDevice));CK(cudaMemset(dbad,0,8));
        S2G_DISPATCH(R.nw,s4_launch_reduce,(int)R.nw,S->L,slices,source_slots,source_slots*slices,
                     dd,raw_stride,bpw,sw,R.dn,R.ninv,S->dy,R.w,dout,sb,dbad,nullptr,first,count,R.mersenne_bits);
        unsigned long long hb=0;
        CK(cudaMemcpy(&hb,dbad,8,cudaMemcpyDeviceToHost));CK(cudaMemcpy(got.data(),dout,got.size()*8,cudaMemcpyDeviceToHost));
        ++canonical_cases;if(hb!=1) ++bad;
        for(size_t i=0;i<got.size();++i) if(got[i]!=(i<slices*count*W ? 0ull : canary[i])) ++bad;
    }
    CK(cudaFree(dd));CK(cudaFree(dout));CK(cudaFree(dbad));
    mpz_clears(av,bv,term,sum,inv,nullptr);
    stage2_log::print(stage2_log::debug, "s4_output_window_check: cases=%llu words=%llu canonical_cases=%llu bad=%llu "
                "(GMP convolution, arbitrary/empty windows, strides, aliases, 5 slices, constant inverse)\n",
                cases,words,canonical_cases,bad);
    if(bad) std::exit(3);
}

/* Independent GMP product oracle, including X+1 and an internal polynomial with constant
   coefficient one. A scalar limb test alone must never classify either as the identity. */
static void groot_lifetime_check(PolyLayer &L)
{
    const size_t W = L.W;
    S4Ctx *saved_s4 = L.s4;
    unsigned long long cases=0, words=0, descent_cases=0, bad=0;
    mpz_t c, a, b, sum;
    mpz_inits(c, a, b, sum, nullptr);
    for (int backend=0; backend<2; ++backend) {
        if (backend && !saved_s4) continue;
        L.s4 = backend ? saved_s4 : nullptr;
        for (size_t n : {0u, 1u, 2u, 3u, 5u, 8u, 9u, 17u}) {
            std::vector<std::vector<unsigned long long>> leaves;
            std::vector<unsigned long long> expected(W, 0ull);
            expected[0]=1;
            for (size_t i=0; i<n; ++i) {
                if (i%4==2) mpz_sub_ui(c, L.N, 1);
                else mpz_set_ui(c, i%4==3 ? 2 : 1);
                std::vector<unsigned long long> leaf(2*W, 0ull), next((i+2)*W, 0ull);
                mpz_to_words(leaf, W, c); leaf.resize(2*W, 0ull); leaf[W]=1;
                leaves.push_back(leaf);
                for (size_t k=0; k<i+2; ++k) {
                    mpz_set_ui(sum, 0);
                    if (k<=i) {
                        words_to_mpz(a, &expected[k*W], W);
                        mpz_mul(sum, a, c);
                    }
                    if (k>0) { words_to_mpz(b, &expected[(k-1)*W], W); mpz_add(sum, sum, b); }
                    mpz_mod(sum, sum, L.N);
                    std::vector<unsigned long long> coef(W);
                    mpz_to_words(coef, W, sum);
                    std::copy(coef.begin(), coef.end(), next.begin()+(long)(k*W));
                }
                expected.swap(next);
            }
            for (bool keep : {true, false}) {
                FTreeStats fs;
                size_t pad=0; std::vector<size_t> deg;
                auto tree=build_tree_flat(L, leaves, deg, pad, fs, -1, keep);
                ++cases; words+=expected.size();
                if (deg[1]!=n || tree[1]!=expected) ++bad;
                if (!keep) for (size_t i=2; i<tree.size(); ++i) if (!tree[i].empty()) ++bad;
                if (keep && n>=2) {
                    /* H(X)=X+5 evaluated at each leaf root -c; independent of cp_mod. */
                    CPoly H(2, std::vector<unsigned long long>(W, 0ull)); H[0][0]=5; H[1][0]=1;
                    {
                        // Independent monic division oracle; the production descent uses scaled states.
                        L.s4 = backend ? saved_s4 : nullptr;
                        std::vector<std::vector<unsigned long long>> values;
                        unsigned long long divmods=0;
                        descent_slow(L, tree, deg, pad, H, values, divmods);
                        L.s4 = backend ? saved_s4 : nullptr;
                        ++descent_cases;
                        if (values.size()!=n) { ++bad; continue; }
                        for (size_t i=0; i<n; ++i) {
                            words_to_mpz(c, leaves[i].data(), W);
                            mpz_ui_sub(sum, 5, c); mpz_mod(sum, sum, L.N);
                            std::vector<unsigned long long> coef(W); mpz_to_words(coef, W, sum);
                            if (values[i]!=coef) ++bad;
                        }
                    }
                }
            }
        }
    }
    L.s4=saved_s4;
    mpz_clears(c, a, b, sum, nullptr);
    stage2_log::print(stage2_log::debug, "groot_lifetime_check: cases=%llu words=%llu descent_cases=%llu bad=%llu "
                "(GMP, empty/single/padded trees, constant-one nonconstant polynomials)\n",
                cases, words, descent_cases, bad);
    if (bad) std::exit(3);
}

/* nbatch slices of `n` coefficients each, coefficient-wise A - B mod N (the flat twin of
   cp_addsub, used by the batched Newton step h = 2 - a*g) */
static std::vector<unsigned long long> cp_addsub_flat(const std::vector<unsigned long long> &A,
                                                      const std::vector<unsigned long long> &B,
                                                      size_t n, size_t W, const mpz_t N,
                                                      bool sub)
{
    std::vector<unsigned long long> out(A.size(), 0ull);
    std::vector<unsigned long long> wa(W, 0ull), wb(W, 0ull);
    mpz_t x, y;
    mpz_inits(x, y, nullptr);
    for (size_t i = 0; i < n; ++i) {
        words_to_mpz(x, &A[i * W], W);
        words_to_mpz(y, &B[i * W], W);
        if (sub) mpz_sub(x, x, y); else mpz_add(x, x, y);
        mpz_mod(x, x, N);
        mpz_to_words(wa, W, x);
        std::copy(wa.begin(), wa.end(), out.begin() + (long)(i * W));
    }
    mpz_clears(x, y, nullptr);
    return out;
}

/* pointer twin of cp_coeff_sub (the batched code works on flat buffers, and the vector version
   reads .size() to decide whether a coefficient is zero) */
static void cp_coeff_sub_p(unsigned long long *dst, const unsigned long long *x,
                           const unsigned long long *y, const mpz_t N, size_t W)
{
    mpz_t a, b;
    mpz_inits(a, b, nullptr);
    words_to_mpz(a, x, W);
    words_to_mpz(b, y, W);
    mpz_sub(a, a, b);
    mpz_mod(a, a, N);
    std::vector<unsigned long long> tmp(W, 0ull);
    mpz_to_words(tmp, W, a);
    std::copy(tmp.begin(), tmp.end(), dst);
    mpz_clears(a, b, nullptr);
}

/* g = 1/a mod X^k for nbatch series of m <= k coefficients each (the reference's cp_inv_series,
   batched).  g is NEVER trimmed (a trimmed inverse can stop growing and spin forever) and a
   non-invertible leading coefficient is fatal rather than silent. */
static void inv_series_batch(PolyLayer &L, const std::vector<unsigned long long> &A, size_t m,
                             size_t k, size_t nbatch, std::vector<unsigned long long> &g, int cat)
{
    const size_t W = L.W;
    /* g holds EXACTLY `len` coefficients per slice at every step, because flat_mul_batch reads
       it at the stride the caller declares and derives its result from the operand's element
       count.  The Newton loop below starts at len = 1 and REPLACES g with a freshly truncated
       buffer at the end of each round, so `nbatch*W` would be enough -- but only if the first
       multiply's result were not inflated, and the chain is easier to trust when the first
       allocation already holds the full k (the first round only writes coefficients 0..nxt-1,
       so nothing about the answer changes). */
    g.assign(nbatch * W, 0ull);
    /* 1/a[0] per slice.  cp_divmod always hands us a REVERSED MONIC divisor, so a[0] == 1 and
       this is the identity -- but it is CHECKED, and the general path is GMP. */
    bool all_one = true;
    for (size_t s = 0; s < nbatch && all_one; ++s) {
        const unsigned long long *c = &A[s * m * W];
        if (c[0] != 1) all_one = false;
        for (size_t j = 1; j < W; ++j) if (c[j] != 0) all_one = false;
    }
    if (all_one) {
        for (size_t s = 0; s < nbatch; ++s) g[s * W] = 1;
    } else {
        mpz_t av, inv;
        std::vector<unsigned long long> gtmp(W, 0ull);
        mpz_inits(av, inv, nullptr);
        for (size_t s = 0; s < nbatch; ++s) {
            words_to_mpz(av, &A[s * m * W], W);
            if (mpz_invert(inv, av, L.N) == 0) {
                std::fprintf(stderr, "%s: FATAL: divisor leading coefficient not invertible "
                                     "mod N (the divisor was not monic)\n", NTT_PROBE_NAME);
                std::exit(4);
            }
            mpz_mod(inv, inv, L.N);
            mpz_to_words(gtmp, W, inv);
            std::copy(gtmp.begin(), gtmp.end(), g.begin() + (long)(s * W));
        }
        mpz_clears(av, inv, nullptr);
    }
    size_t len = 1;
    std::vector<unsigned long long> at, ag, h, gn;
    std::vector<unsigned long long> gpad;      /* g zero-extended to the current nxt */
    while (len < k) {
        const size_t nxt = ((2 * len) < k) ? (2 * len) : k;
        const size_t am = (m < nxt) ? m : nxt;
        at.assign(nbatch * am * W, 0ull);
        for (size_t s = 0; s < nbatch; ++s)
            std::copy(A.begin() + (long)(s * m * W), A.begin() + (long)(s * m * W + am * W),
                      at.begin() + (long)(s * am * W));
        /* g zero-extended to nxt coefficients with the extension FORCED to zero: g is the
           inverse of a mod X^len, so coefficients len..nxt-1 are identically zero.  (The
           extension is also what lets the products below reach degree nxt-1 regardless of how
           short g currently is.) */
        gpad.assign(nbatch * nxt * W, 0ull);
        for (size_t s = 0; s < nbatch; ++s)
            std::copy(g.begin() + (long)(s * len * W),
                      g.begin() + (long)((s + 1) * len * W),
                      gpad.begin() + (long)(s * nxt * W));
        /* ---- ag = (a*g) mod X^nxt ---------------------------------------------------
           EVERY buffer below is nbatch slices of exactly the coefficient count its own `n`
           argument declares, because each consumer (flat_mul_batch, flat_truncate,
           cp_addsub_flat) strides by that declaration.  The multiply's two operands must also
           be the arguments' OWN lengths: this call site used to pass (at, am, g, len) while `g`
           actually held 2*len, so the multiply read the low `len` of `at` and the whole 2*len
           of `g`, returning 3*len-1 coefficients where the code assumed am+len-1 -- the slice
           strides below stopped matching their buffers, and at the real shape (P = 4096,
           W = 83) that walked off the heap inside flat_truncate's memcpy. */
        flat_mul_batch(L, at, am, (len == nxt) ? g : gpad, nxt, nbatch, ag, cat, 0, nxt);
        /* The returned prefix is already tight at nxt coefficients per slice; the input and
           transform shape still cover the complete product. */
        if (ag.size() != nbatch * nxt * W) ag.resize(nbatch * nxt * W, 0ull);
        h.assign(nbatch * nxt * W, 0ull);
        {
            mpz_t two;
            mpz_init(two);
            mpz_set_ui(two, 2);
            mpz_mod(two, two, L.N);
            std::vector<unsigned long long> w2(W, 0ull);
            mpz_to_words(w2, W, two);
            for (size_t s = 0; s < nbatch; ++s)
                std::copy(w2.begin(), w2.end(), h.begin() + (long)(s * nxt * W));
            mpz_clear(two);
        }
        h = cp_addsub_flat(h, ag, nbatch * nxt, L.W, L.N, /*sub=*/true);   /* h = 2 - ag */
        h.resize(nbatch * nxt * W);
        /* ---- g = (g*h) mod X^nxt ---------------------------------------------------- */
        flat_mul_batch(L, gpad, nxt, h, nxt, nbatch, gn, cat, 0, nxt);
        if (gn.size() != nbatch * nxt * W) gn.resize(nbatch * nxt * W, 0ull);
        g.swap(gn);
        len = nxt;
    }
}

/* a = q*b + r with deg r < deg b, for nbatch pairs of the SAME shape (the reference's
   cp_divmod, batched).  A: nbatch slices of da+1 coefficients, B: nbatch of db+1 (monic),
   out: nbatch of db coefficients.  k = da-db+1 > 0. */
static void divmod_batch(PolyLayer &L, const std::vector<unsigned long long> &A,
                         const std::vector<unsigned long long> &B, size_t da, size_t db,
                         size_t nbatch, std::vector<unsigned long long> &out, int cat)
{
    const size_t W = L.W;
    const size_t k = da - db + 1;
    std::vector<unsigned long long> ra(nbatch * k * W, 0ull), rb(nbatch * (db + 1) * W, 0ull);
    for (size_t s = 0; s < nbatch; ++s) {
        for (size_t i = 0; i < k; ++i)
            std::copy(A.begin() + (long)((s * (da + 1) + (da - i)) * W),
                      A.begin() + (long)((s * (da + 1) + (da - i) + 1) * W),
                      ra.begin() + (long)((s * k + i) * W));
        for (size_t i = 0; i <= db; ++i)
            std::copy(B.begin() + (long)((s * (db + 1) + (db - i)) * W),
                      B.begin() + (long)((s * (db + 1) + (db - i) + 1) * W),
                      rb.begin() + (long)((s * (db + 1) + i) * W));
    }
    std::vector<unsigned long long> rbi;
    inv_series_batch(L, rb, db + 1, k, nbatch, rbi, cat);
    if (rbi.size() < nbatch * k * W) {
        std::fprintf(stderr, "%s: FATAL: the batched inverse returned %llu words, expected "
                             "%llu (nbatch=%llu k=%llu W=%llu)\n", NTT_PROBE_NAME,
                     (unsigned long long)rbi.size(), (unsigned long long)(nbatch * k * W),
                     (unsigned long long)nbatch, (unsigned long long)k, (unsigned long long)W);
        std::exit(3);
    }
    std::vector<unsigned long long> qrev, q, qb;
    /* ra holds k coefficients (the reversed top k of the dividend) and the inverse holds k, so
       the product qrev = ra*rbi has 2k-1: flat_truncate below reads exactly that stride and
       keeps the top k, which is the reversed quotient.  inv_series_batch already returns k per
       slice, so this is only an assertion of the contract the truncate depends on. */
    if (rbi.size() != nbatch * k * W) {
        std::fprintf(stderr, "%s: FATAL: the batched inverse has %llu words, expected %llu "
                             "(nbatch=%llu k=%llu W=%llu)\n", NTT_PROBE_NAME,
                     (unsigned long long)rbi.size(), (unsigned long long)(nbatch * k * W),
                     (unsigned long long)nbatch, (unsigned long long)k, (unsigned long long)W);
        std::exit(3);
    }
    flat_mul_batch(L, ra, k, rbi, k, nbatch, qrev, cat, 0, k);
    q.assign(nbatch * k * W, 0ull);
    for (size_t s = 0; s < nbatch; ++s)
        for (size_t i = 0; i < k; ++i)
            std::copy(qrev.begin() + (long)((s * k + (k - 1 - i)) * W),
                      qrev.begin() + (long)((s * k + (k - 1 - i) + 1) * W),
                      q.begin() + (long)((s * k + i) * W));
    flat_mul_batch(L, q, k, B, db + 1, nbatch, qb, cat, 0, db);
    out.assign(nbatch * db * W, 0ull);
    for (size_t s = 0; s < nbatch; ++s)
        for (size_t i = 0; i < db; ++i)
            cp_coeff_sub_p(&out[(s * db + i) * W], &A[(s * (da + 1) + i) * W],
                           &qb[(s * db + i) * W], L.N, W);
}

/* the exact primality test below needs a*b mod m for 64-bit a,b,m.  __int128 is NOT usable here:
   nvcc parses this whole translation unit for the device too and rejects the type (measured:
   "expected a )" at the __int128 cast), so 64x64 -> 128 is assembled from 32-bit halves --
   a*b = a*bh*2^32 + a*bl with a*bl < 2^96 and each piece reduced mod m before it is combined, so
   no intermediate ever leaves 64 bits. */
static inline unsigned long long mulmod_u64(unsigned long long a, unsigned long long b,
                                            unsigned long long m)
{
    a %= m;
    b %= m;
    const unsigned long long bh = b >> 32, bl = b & 0xffffffffull;
    const unsigned long long ah = a >> 32, al = a & 0xffffffffull;
    /* a*bh < 2^96: reduce it as (ah*bh << 32) + al*bh, step by step */
    const unsigned long long p_hh = (ah * bh) % m;
    const unsigned long long p_hl = (al * bh) % m;
    const unsigned long long p_lh = (ah * bl) % m;
    const unsigned long long p_ll = (al * bl) % m;
    unsigned long long r = (((p_hh << 32) % m) + p_hl) % m;   /* the 2^64 part */
    r = (((r << 32) % m) + p_lh) % m;                         /* the 2^32 part */
    r = (r + p_ll) % m;
    return r;
}

static unsigned long long powmod_u64(unsigned long long a, unsigned long long e,
                                     unsigned long long m)
{
    unsigned long long r = 1ull % m;
    a %= m;
    while (e) {
        if (e & 1ull) r = mulmod_u64(r, a, m);
        a = mulmod_u64(a, a, m);
        e >>= 1;
    }
    return r;
}

/* ---- the 64-bit primality test the naming loop lives or dies by ----------------------------
 *
 * The culprit-naming loop calls this ~2*(B2/D) times PER HIT LEAF, so at the real shape
 * (B2=4e10, D=570570) one hit leaf costs 1.4e5 calls.  The old implementation was
 * mpz_probab_prime_p(z, 25): a GMP object created and destroyed, plus 25 Miller-Rabin rounds,
 * for a number that is already known to be < 2^64.  It measured ~150 us per call, which is what
 * made the naming loop (not the algorithm) the pole of the whole run -- see the split timers
 * `t_scan`/`t_ladder` in the `batched_naming:` line, and docs/DEV_STAGE2_GPU_PLAN.md section 26.
 *
 * This version is EXACT, not probabilistic, and much cheaper:
 *   * trial division by every prime <= 101 (there are 26 of them) rejects ~88% of all candidates
 *     with one 64-bit remainder each, before any modular exponentiation happens;
 *   * the survivors go through a deterministic Miller-Rabin whose 7-base set
 *     {2, 325, 9375, 28178, 450775, 9780504, 1795265022} is PROVEN complete for n < 3.317e24 >
 *     2^64 (Sinclair/Jimenez-da-Silva -- the older "first 12 primes" set is only proven to
 *     3.18e23), so the answer is true primality, never "probably prime".  A false POSITIVE here
 *     would put a bogus prime into `hit_primes`, and the acceptance gate compares that list
 *     against the CPU reference, so exactness is a requirement, not a nicety.
 *   * no GMP object is created or destroyed on this path at all.
 * The GMP call was `p < 2` -> false, `>= 2` -> the test; the same boundary is kept. */
static bool is_prime_u64(unsigned long long p)
{
    if (p < 2) return false;
    /* 1. the small primes, by trial division (cheap and exact).  NOT named `small`: MSVC keeps the
       legacy keyword `small` (== char) and parses `unsigned small[]` as a structured binding. */
    {
        static const unsigned kTrialPrimes[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43,
                                                47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101};
        const int nsmall = (int)(sizeof(kTrialPrimes) / sizeof(kTrialPrimes[0]));
        for (int i = 0; i < nsmall; ++i) {
            const unsigned long long q = kTrialPrimes[i];
            if (p % q == 0) return p == q;
        }
    }
    /* 2. exact Miller-Rabin for the whole 64-bit range */
    unsigned long long d = p - 1;
    int s = 0;
    while ((d & 1ull) == 0) { d >>= 1; ++s; }
    static const unsigned long long base[] = {2ull, 325ull, 9375ull, 28178ull, 450775ull,
                                              9780504ull, 1795265022ull};
    const int nbase = (int)(sizeof(base) / sizeof(base[0]));
    for (int ib = 0; ib < nbase; ++ib) {
        const unsigned long long b = base[ib];
        if (b % p == 0) continue;
        unsigned long long x = powmod_u64(b % p, d, p);
        if (x == 1ull || x == p - 1ull) continue;
        bool composite = true;
        for (int i = 1; i < s; ++i) {
            x = mulmod_u64(x, x, p);
            if (x == p - 1ull) { composite = false; break; }
        }
        if (composite) return false;
    }
    return true;
}

/* NTT_NAME_MAX=n: name at most the first n hit leaves (a hit leaf = a leaf whose value shares a
   factor with N).  0 or unset = name EVERY hit leaf, which is what the validation runs use
   (the gate compares `hit_primes` against the CPU reference).  The scan is pure DIAGNOSTICS --
   the factor set no longer depends on it -- so a timing run sets this to keep it out of the
   wall clock.  Naming is bounded per RUN, not per block: the leaf indices are stable and the
   first leaves are the same ones every run. */
static long long g_name_max = -1;
static long long name_max(void)
{
    if (g_name_max < 0) {
        const char *e = std::getenv("NTT_NAME_MAX");
        g_name_max = (e && *e) ? std::atoll(e) : 0;
        if (g_name_max < 0) g_name_max = 0;
    }
    return g_name_max;
}

struct Stage2Tail {
    unsigned long long hits = 0, bad_factors = 0;
    /* hits whose prime was counted but not named (the naming budget stopped the scan): hits and
       bad_factors stay EXACTLY comparable with the CPU reference either way, only the
       hit_primes list is short, and that is reported rather than hidden */
    unsigned long long unnamed_hits = 0;
    std::vector<std::string> factors;
    std::vector<unsigned long long> hit_primes;
};

/* the giant points x_i = affine_x([i*D]Q), i = 1 .. B2/D+2, on the device.  The CPU
   reference reaches the same points with a differential-addition chain; both are [i*D]Q, and
   independent ladders are both simpler and parallel, so the DEFINITION (not the reference's
   optimisation) is what is mirrored here. */


/* the S2 algorithm: remainder tree + accumulate + gcd, naming the culprit prime the way the
   reference does (an independent device ladder for every candidate p = i*D -+ j, gcd of Z) */


/* ===================================================================================== *
 *  SLICE S3 -- THE BATCHED STRUCTURE, WITH THE ORCHESTRATION ON THE DEVICE
 *  (docs/DEV_STAGE2_GPU_PLAN.md sections 13.1, 18 and 18.3)
 *
 *  WHAT S2 DID (the simple structure): ONE product tree over all B2/D giant points, one
 *  remainder-tree descent against it, one accumulation.  The CPU cost model says that costs
 *  6.46e7 operand-bits at the frozen shape; the BATCHED structure (Prime95's shape, which
 *  tools/bench/stage2_tree_ref.cpp models as `ours-balanced` / `ours-padded`) costs 1.65e7 --
 *  3.9x less work -- because a giant point's cost drops from 3*2S*log2(B2/D) to
 *  2S*(log2 P + 6):
 *
 *     F(X) = prod_j (X - x_j) mod N over the P baby points      built ONCE (slice S1/S2 code)
 *     1/F via Newton on the REVERSED F                          ONCE, cached, reused by every
 *                                                               mod-F reduction below
 *     for each batch of P giant points:
 *         G(X) = prod_i (X - x_i) mod N                         one G tree
 *         H <- (G*H) mod F                                      3 full-size multiplies:
 *                                                               T = G*H,
 *                                                               rev(T)*rev(F)^-1 (truncated),
 *                                                               q*F (subtracted from T)
 *     ONE descent of H against the F tree                       H(x_j) for every baby point
 *     accumulate prod_j H(x_j) mod N                            ON THE DEVICE, block products
 *
 *  WHY THE ANSWER IS THE SAME AS S2's (this is the claim the acceptance gate tests):
 *  F(X) = prod_j (X - x_j) vanishes at every baby point, so reducing mod F and then
 *  evaluating at x_j gives H(x_j) = prod_b G_b(x_j) = prod over ALL (giant, baby) pairs of
 *  (x_j - x_i).  S2 accumulated prod_i F(x_i) = prod over the same pairs of (x_i - x_j).  The
 *  two products are equal up to the unit (-1)^(I*P) and, more importantly, have the SAME set
 *  of prime divisors -- which is all a GCD can see.  So the factor set and the named hit
 *  primes are identical by construction, and the gate checks it empirically against S2, the
 *  CPU tree reference and the pairing reference.  (Only the GROUPING of the accumulation and
 *  the leaf a culprit is attributed to change: S2 blames a giant point and then searches the
 *  baby set, this engine blames a baby point and searches the giant set.  The candidate set
 *  {i*D +- j : p prime, B1 < p <= B2} is the same one in both directions.')
 *
 *  WHAT IS ON THE DEVICE (the point of the slice):
 *    * ONE NttArena for the whole curve: every buffer and every twiddle table of a shape is
 *      allocated/built once and reused by all ~5e3 multiplies (see the arena comment in
 *      ntt_poly_probe.cu -- the per-call cudaMalloc + fuse_init was measured as the dominant
 *      cost of the S2 run);
 *    * the giant points are ONE ladder launch for all of them, with the ladder's device
 *      buffers allocated once instead of per batch;
 *    * the descent's OUTPUT -- the H(x_j) leaf values -- is uploaded and multiplied on the
 *      device (s2g_block_prod_kernel, the ladder's own Montgomery multiplier), so the host
 *      GMP sees one value per BLOCK and then a gcd, not one value per leaf.  The per-leaf
 *      values are only read back for a block whose product actually shares a factor with N,
 *      which is where the culprit has to be named.
 *    * the per-multiply exact coefficient extraction is still host GMP: it is part of the
 *      verified NTT multiply, and moving the 2S-bit mod-N reduction of the extracted
 *      coefficients onto the device is slice S4's job (section 18.1 item 2).
 * ===================================================================================== */

/* product mod N of every value in a block: out[b] = prod_{t in block b} vals[t]
   (Montgomery-chained: acc <- acc*v*R^-1, so the block product comes back as prod*R^-(k-1)
   mod N.  R is invertible mod N, so the GCD with N -- the only thing stage 2 asks of this
   number -- is unchanged, every intermediate is a proper value in [0,N), and the reference
   records a gcd rather than the raw product, so nothing that is printed changes.) */
template <int NW>
__global__ void s2g_block_prod_kernel(const unsigned long long *vals, int nvals, int nw,
                                      int per_block, const unsigned long long *n,
                                      unsigned long long ninv, unsigned long long *out)
{
    const int b = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    const int nb = (nvals + per_block - 1) / per_block;
    if (b >= nb) return;
    const int lo = b * per_block;
    int hi = lo + per_block;
    if (hi > nvals) hi = nvals;
    unsigned long long acc[NW];
    for (int i = 0; i < nw; ++i) acc[i] = vals[(size_t)lo * nw + i];
    for (int t = lo + 1; t < hi; ++t)
        s2g_mont_mul<NW>(acc, acc, vals + (size_t)t * nw, n, ninv, nw);
    for (int i = 0; i < nw; ++i) out[(size_t)b * nw + i] = acc[i];
}

template <int NW>
static void s2g_launch_block_prod(int nw, int nvals, int per_block,
                                  const unsigned long long *dv, const unsigned long long *dn,
                                  unsigned long long ninv, unsigned long long *dout)
{
    const int nb = (nvals + per_block - 1) / per_block;
    const unsigned int th = 64;
    const unsigned int bl = (unsigned int)((nb + th - 1) / th);
    s2g_block_prod_kernel<NW><<<bl, th>>>(dv, nvals, nw, per_block, dn, ninv, dout);
}

/* every device buffer the batched engine needs, allocated ONCE for the whole curve: the
   ladder's inputs (uploaded once), its scratch, and the accumulation buffers. */
struct S3Workspace {
    const LadderCtx *C = nullptr;
    unsigned long long *dn = nullptr, *dqx = nullptr, *dqz = nullptr, *da24 = nullptr,
                       *dmone = nullptr;
    unsigned long long *djs = nullptr;
    unsigned long long *dx = nullptr, *dz = nullptr;
    unsigned long long *giant_base=nullptr;
    unsigned long long giant_base_d=0;
    bool giant_base_unit=false;
    unsigned long long *dvals = nullptr, *dprod = nullptr;
    unsigned long long *dsegfix=nullptr;
    size_t segfix_cap=0;
    size_t js_cap = 0, pt_cap = 0, val_cap = 0, prod_cap = 0;
    unsigned long long ladder_calls = 0, ladder_points_total = 0, prod_launches = 0;
    size_t bytes = 0;

    ~S3Workspace()
    {
        if (dn) cudaFree(dn);
        if (dqx) cudaFree(dqx);
        if (dqz) cudaFree(dqz);
        if (da24) cudaFree(da24);
        if (dmone) cudaFree(dmone);
        if (djs) cudaFree(djs);
        if (dx) cudaFree(dx);
        if (dz) cudaFree(dz);
        if (giant_base) cudaFree(giant_base);
        if (dvals) cudaFree(dvals);
        if (dprod) cudaFree(dprod);
        if (dsegfix) cudaFree(dsegfix);
    }
    void init(const LadderCtx &c)
    {
        C = &c;
        const size_t nw = c.nw;
        CK(cudaMalloc(&dn, nw * 8));
        CK(cudaMalloc(&dqx, nw * 8));
        CK(cudaMalloc(&dqz, nw * 8));
        CK(cudaMalloc(&da24, nw * 8));
        CK(cudaMalloc(&dmone, nw * 8));
        CK(cudaMemcpy(dn, c.hn.data(), nw * 8, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dqx, c.hqx.data(), nw * 8, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dqz, c.hqz.data(), nw * 8, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(da24, c.ha24.data(), nw * 8, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dmone, c.hmone.data(), nw * 8, cudaMemcpyHostToDevice));
        bytes = 5 * nw * 8;
    }
    void need_segfix(size_t seg)
    {
        if(seg<=segfix_cap)return;
        const size_t nw=C->nw;
        std::vector<unsigned long long> powers((seg+1)*nw,0),word(nw);
        mpz_t N,R,p;mpz_inits(N,R,p,nullptr);
        words_to_mpz(N,C->hn.data(),nw);words_to_mpz(R,C->hmone.data(),nw);
        mpz_set_ui(p,1);
        for(size_t m=0;m<=seg;++m) {
            mpz_to_words(word,nw,p);std::copy(word.begin(),word.end(),powers.begin()+m*nw);
            mpz_mul(p,p,R);mpz_mod(p,p,N);
        }
        mpz_clears(N,R,p,nullptr);
        if(dsegfix){CK(cudaFree(dsegfix));bytes-=(segfix_cap+1)*nw*8;}
        dsegfix=nullptr;CK(cudaMalloc(&dsegfix,powers.size()*8));
        CK(cudaMemcpy(dsegfix,powers.data(),powers.size()*8,cudaMemcpyHostToDevice));
        segfix_cap=seg;bytes+=powers.size()*8;
    }
    bool need_giant_base(unsigned long long d,mpz_srcptr modulus)
    {
        if(giant_base_d==d)return giant_base_unit;
        const double begin=now_s();const size_t nw=C->nw;
        if(!giant_base){CK(cudaMalloc(&giant_base,16*nw));bytes+=16*nw;}
        {
            need_pts(1);CK(cudaMemcpy(djs,&d,8,cudaMemcpyHostToDevice));
            S2G_DISPATCH((int)nw,s2g_launch_ladder,(int)nw,1,dn,C->ninv,dqx,dqz,da24,dmone,
                         djs,giant_base,giant_base+nw,false);
            CK(cudaGetLastError());++ladder_calls;++ladder_points_total;
            std::vector<unsigned long long> z(nw);
            CK(cudaMemcpy(z.data(),giant_base+nw,8*nw,cudaMemcpyDeviceToHost));
            mpz_t value,gcd;mpz_inits(value,gcd,nullptr);
            words_to_mpz(value,z.data(),nw);mpz_gcd(gcd,value,modulus);
            giant_base_unit=mpz_cmp_ui(gcd,1)==0;mpz_clears(value,gcd,nullptr);
            ++g_giant_seed.base_gpu_builds;g_giant_seed.base_d2h_bytes+=8*nw;
        }
        giant_base_d=d;++g_giant_seed.base_builds;
        g_giant_seed.base_build_seconds+=now_s()-begin;
        if(!giant_base_unit)++g_giant_seed.base_nonunits;
        return giant_base_unit; // Preserve the original factor path for nonunits/infinity.
    }
    void need_pts(size_t n)
    {
        if (n <= pt_cap) return;
        if (djs) cudaFree(djs);
        if (dx) cudaFree(dx);
        if (dz) cudaFree(dz);
        djs = nullptr; dx = nullptr; dz = nullptr;
        const size_t nw = C->nw;
        CK(cudaMalloc(&djs, n * 8));
        CK(cudaMalloc(&dx, n * nw * 8));
        CK(cudaMalloc(&dz, n * nw * 8));
        bytes += n * 8 + 2 * n * nw * 8;
        pt_cap = n;
    }
    void need_vals(size_t n)
    {
        if (n <= val_cap) return;
        if (dvals) cudaFree(dvals);
        if (dprod) cudaFree(dprod);
        dvals = nullptr; dprod = nullptr;
        const size_t nw = C->nw;
        CK(cudaMalloc(&dvals, n * nw * 8));
        CK(cudaMalloc(&dprod, n * nw * 8));
        bytes += 2 * n * nw * 8;
        val_cap = n;
        prod_cap = n;
    }
};

static void segment_product_fixture(const LadderCtx &C)
{
    S3Workspace ws;ws.init(C);const size_t nw=C.nw;
    mpz_t N,R,ri,p,z,want,tmp;mpz_inits(N,R,ri,p,z,want,tmp,nullptr);
    words_to_mpz(N,C.hn.data(),nw);words_to_mpz(R,C.hmone.data(),nw);
    if(!mpz_invert(ri,R,N)){std::fprintf(stderr,"FATAL: R not invertible\n");std::exit(3);}
    unsigned long long cases=0,checks=0,legacy_different=0,state=0x982dded45;
    for(size_t n : {0u,1u,2u,15u,16u,17u,31u,32u,33u,63u,64u,65u,127u})
    for(size_t seg : {1u,2u,3u,7u,16u,17u,32u}) {
        ++cases;ws.need_segfix(seg);
        if(!n){S2G_DISPATCH((int)nw,s2g_launch_segprod,(int)nw,0,seg,C.ninv,ws.dn,nullptr,nullptr,ws.dsegfix);continue;}
        const size_t ns=(n+seg-1)/seg;std::vector<unsigned long long> input(n*nw),a(ns*nw),b(ns*nw),word(nw);
        for(size_t i=0;i<n;++i) {
            for(size_t j=0;j<nw;++j){state=state*6364136223846793005ull+1;word[j]=state;}
            words_to_mpz(z,word.data(),nw);mpz_mod(z,z,N);
            if(i%19==0)mpz_set_ui(z,0);else if(i%19==1)mpz_sub_ui(z,N,1);else if(i%19==2)mpz_set_ui(z,1);
            mpz_to_words(word,nw,z);std::copy(word.begin(),word.end(),input.begin()+i*nw);
        }
        unsigned long long *din=nullptr,*dout=nullptr;CK(cudaMalloc(&din,input.size()*8));CK(cudaMalloc(&dout,a.size()*8));
        CK(cudaMemcpy(din,input.data(),input.size()*8,cudaMemcpyHostToDevice));
        S2G_DISPATCH((int)nw,s2g_launch_segprod,(int)nw,n,seg,C.ninv,ws.dn,din,dout,ws.dsegfix);
        CK(cudaGetLastError());CK(cudaMemcpy(a.data(),dout,a.size()*8,cudaMemcpyDeviceToHost));
        S2G_DISPATCH((int)nw,s2g_launch_segprod,(int)nw,n,seg,C.ninv,ws.dn,din,dout,nullptr);
        CK(cudaGetLastError());CK(cudaMemcpy(b.data(),dout,b.size()*8,cudaMemcpyDeviceToHost));
        CK(cudaFree(din));CK(cudaFree(dout));
        const char *fault=std::getenv("NTT_GFINV_SEG_TEST_BAD");if(fault && std::atoi(fault))a[0]^=1;
        for(size_t j=0;j<ns;++j) {
            const size_t lo=j*seg,hi=std::min(n,lo+seg);mpz_set_ui(p,1);
            for(size_t i=lo;i<hi;++i){words_to_mpz(z,input.data()+i*nw,nw);mpz_mul(p,p,z);mpz_mod(p,p,N);}
            words_to_mpz(z,a.data()+j*nw,nw);
            if(mpz_cmp(p,z)){std::fprintf(stderr,"FATAL: segment fixture GMP mismatch n=%llu seg=%llu\n",(unsigned long long)n,(unsigned long long)seg);std::exit(3);}
            mpz_powm_ui(tmp,ri,(unsigned long)(hi-lo-1),N);mpz_mul(want,p,tmp);mpz_mod(want,want,N);
            words_to_mpz(z,b.data()+j*nw,nw);
            if(mpz_cmp(want,z)){std::fprintf(stderr,"FATAL: legacy segment formula mismatch\n");std::exit(3);}
            if(mpz_cmp(p,z))++legacy_different;++checks;
        }
    }
    mpz_clears(N,R,ri,p,z,want,tmp,nullptr);
    stage2_log::print(stage2_log::debug, "segment_product_fixture: cases=%llu checks=%llu legacy_different=%llu bad=0\n",cases,checks,legacy_different);
}

/* (X_i, Z_i) = [js[i]]Q through the SAME ladder kernel as S1/S2, with the buffers reused */
static void ladder_points_ws(S3Workspace &W, const std::vector<unsigned long long> &js,
                             std::vector<unsigned long long> &outx,
                             std::vector<unsigned long long> &outz)
{
    const LadderCtx &C = *W.C;
    const size_t nw = C.nw, n = js.size();
    outx.assign(n * nw, 0ull);
    outz.assign(n * nw, 0ull);
    if (n == 0) return;
    W.need_pts(n);
    CK(cudaMemcpy(W.djs, js.data(), n * 8, cudaMemcpyHostToDevice));
    S2G_DISPATCH((int)nw, s2g_launch_ladder, (int)nw, (int)n, W.dn, C.ninv, W.dqx, W.dqz,
                 W.da24, W.dmone, W.djs, W.dx, W.dz);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(outx.data(), W.dx, outx.size() * 8, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(outz.data(), W.dz, outz.size() * 8, cudaMemcpyDeviceToHost));
    ++W.ladder_calls;
    W.ladder_points_total += n;
}

/* one chunk of giant points through the DIFFERENTIAL-ADDITION CHAIN (see s2g_chain_kernel):
   seeds from the existing ladder, then one xADD per point.  `gx`/`gz` come back as PROJECTIVE
   (X:Z) Montgomery pairs, which is all the caller's affine_x = X/Z mod N needs.
   `per_block` points per thread; `check` also computes the chunk the old way and compares the
   affine x values (the ladder and the chain give different projective representatives of the
   same point, so the affine value is the only thing that can be compared). */
struct DeviceGLeafStats {
    unsigned long long requested_chunks=0,chunks=0,fallback_chunks=0,groups=0,bad_groups=0,
        good_segments=0,device_trees=0,device_leaf_words=0,patch_words=0,
        group_d2h_bytes=0,bad_segment_d2h_bytes=0,bad_point_d2h_bytes=0,
        avoided_point_d2h_bytes=0,avoided_segment_d2h_bytes=0,avoided_leaf_h2d_bytes=0,
        coord_peak_bytes=0,group_peak_bytes=0,checked_groups=0,checked_leaf_words=0;
    double t_prepare=0,t_invert=0,t_fill=0;
};
static bool device_gleaf_flag(const char *name) {
    const char *e=std::getenv(name);return e && std::atoi(e)!=0;
}
/* Own chain outputs or borrow the ladder workspace until ALL trees in this chunk finish.
   Neither pointer aliases S4 raw frontiers. Bad-group coordinates are sparse host objects. */
struct ResidentGiant {
    static constexpr size_t GROUP=64,SEG=S2G_GFINV_SEG;
    unsigned long long *x=nullptr,*z=nullptr,*segments=nullptr,*groups_device=nullptr,*group_fix=nullptr;
    size_t n=0,w=0;bool own=false;
    std::vector<unsigned char> group_good;
    std::vector<std::vector<unsigned long long>> bad_x,bad_z;
    ~ResidentGiant() {
        if(own){if(x)CK(cudaFree(x));if(z)CK(cudaFree(z));}
        if(segments)CK(cudaFree(segments));if(groups_device)CK(cudaFree(groups_device));
        if(group_fix)CK(cudaFree(group_fix));
    }
    const unsigned long long *coord(size_t q,bool use_z) const {
        const size_t group=q/(GROUP*SEG),local=q%(GROUP*SEG);
        const auto &v=(use_z?bad_z:bad_x)[group];
        if(local*w+w>v.size()){std::fprintf(stderr,"FATAL: sparse giant coordinate outside bad group\n");std::exit(3);}
        return v.data()+local*w;
    }
    bool good(size_t segment) const {return group_good[segment/GROUP]!=0;}
    void prepare(S3Workspace &ws,PolyLayer &L,std::vector<unsigned long long> &gseg,
        mpz_t Ginv,unsigned long long &gamma_points,unsigned long long &projective_segments,DeviceGLeafStats &st) {
        const double begin=now_s();const size_t ns=(n+SEG-1)/SEG,ng=(ns+GROUP-1)/GROUP;
        const auto bad_before=st.bad_point_d2h_bytes,seg_before=st.bad_segment_d2h_bytes;
        ++st.chunks;st.groups+=ng;st.coord_peak_bytes=std::max(st.coord_peak_bytes,16ull*n*w);
        group_good.assign(ng,0);bad_x.resize(ng);bad_z.resize(ng);
        CK(cudaMalloc(&groups_device,ng*w*8));CK(cudaMalloc(&group_fix,(GROUP+1)*w*8));
        std::vector<unsigned long long> fix((GROUP+1)*w),word(w),products(ng*w);
        mpz_t R,p,v,inv;mpz_inits(R,p,v,inv,nullptr);words_to_mpz(R,ws.C->hmone.data(),w);mpz_set_ui(p,1);
        for(size_t j=0;j<=GROUP;++j) {
            mpz_to_words(word,w,p);std::copy(word.begin(),word.end(),fix.begin()+j*w);
            mpz_mul(p,p,R);mpz_mod(p,p,L.N);
        }
        CK(cudaMemcpy(group_fix,fix.data(),fix.size()*8,cudaMemcpyHostToDevice));
        S2G_DISPATCH((int)w,s2g_launch_segprod,(int)w,ns,GROUP,ws.C->ninv,ws.dn,segments,groups_device,group_fix);
        CK(cudaGetLastError());CK(cudaMemcpy(products.data(),groups_device,products.size()*8,cudaMemcpyDeviceToHost));
        st.group_d2h_bytes+=products.size()*8;st.group_peak_bytes=std::max(st.group_peak_bytes,8ull*(ng+GROUP+1)*w);
        std::vector<unsigned long long> all_segments;
        if(device_gleaf_flag("NTT_DEVICE_GLEAF_CHECK")) {
            all_segments.resize(ns*w);CK(cudaMemcpy(all_segments.data(),segments,all_segments.size()*8,cudaMemcpyDeviceToHost));
        }
        for(size_t h=0;h<ng;++h) {
            const size_t lo=h*GROUP,hi=std::min(ns,lo+GROUP);
            words_to_mpz(p,products.data()+h*w,w);
            if(!all_segments.empty()) {
                mpz_set_ui(v,1);
                for(size_t k=lo;k<hi;++k){words_to_mpz(inv,all_segments.data()+k*w,w);mpz_mul(v,v,inv);mpz_mod(v,v,L.N);}
                if(mpz_cmp(p,v)){std::fprintf(stderr,"FATAL: GPU Gamma group GMP mismatch\n");std::exit(3);}++st.checked_groups;
            }
            const double ti=now_s();const bool unit=mpz_invert(inv,p,L.N)!=0;st.t_invert+=now_s()-ti;
            if(unit) {
                group_good[h]=1;mpz_mul(Ginv,Ginv,inv);mpz_mod(Ginv,Ginv,L.N);
                gamma_points+=std::min(n,hi*SEG)-lo*SEG;projective_segments+=hi-lo;st.good_segments+=hi-lo;
            } else {
                ++st.bad_groups;CK(cudaMemcpy(gseg.data()+lo*w,segments+lo*w,(hi-lo)*w*8,cudaMemcpyDeviceToHost));
                st.bad_segment_d2h_bytes+=(hi-lo)*w*8;
                const size_t first=lo*SEG,count=std::min(n,hi*SEG)-first;
                bad_x[h].resize(count*w);bad_z[h].resize(count*w);
                CK(cudaMemcpy(bad_x[h].data(),x+first*w,count*w*8,cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(bad_z[h].data(),z+first*w,count*w*8,cudaMemcpyDeviceToHost));
                st.bad_point_d2h_bytes+=count*w*16;
            }
        }
        st.avoided_point_d2h_bytes+=16ull*n*w-(st.bad_point_d2h_bytes-bad_before);
        st.avoided_segment_d2h_bytes+=8ull*ns*w-(st.bad_segment_d2h_bytes-seg_before);
        mpz_clears(R,p,v,inv,nullptr);st.t_prepare+=now_s()-begin;
    }
};

static void giant_chunk_chain(PolyLayer &L, const LadderCtx &C, S3Workspace &W, unsigned long long D,
                              unsigned long long clo, unsigned long long chi,
                              unsigned long long per_block, std::vector<unsigned long long> &gx,
                              std::vector<unsigned long long> &gz, bool check,
                              unsigned long long &seed_points,
                              std::vector<unsigned long long> *gseg = nullptr,
                              unsigned long long seg = 16,ResidentGiant *resident=nullptr)
{
    const size_t nw = C.nw;
    const unsigned long long npts = chi - clo + 1;
    const unsigned long long blocks = (npts + per_block - 1) / per_block;
    mpz_t Rm, X, Z;
    mpz_inits(Rm, X, Z, nullptr);
    /* R = the Montgomery constant (the image of 1) is exactly what the ladder context holds as
       `hmone`, so the normal->image conversion needs no extra constant */
    words_to_mpz(Rm, C.hmone.data(), nw);
    /* the ladder points we need: for every block its first two multiples, plus x_D itself */
    std::vector<unsigned long long> js;
    js.reserve((size_t)(2 * blocks + 1));
    for (unsigned long long b = 0; b < blocks; ++b) {
        const unsigned long long i0 = clo + b * per_block;
        js.push_back(i0 * D);
        js.push_back((i0 + 1 <= chi ? (i0 + 1) : i0) * D);
    }
    js.push_back(D);                                    /* the difference point x_D */
    seed_points = (unsigned long long)js.size();
    std::vector<unsigned long long> lx, lz;
    if(g_giant_seed_device) {
        W.need_pts(js.size());
        const bool paired=g_giant_seed_pair && W.need_giant_base(D,L.N);
        if(paired) {
            S2G_DISPATCH((int)nw,s2g_launch_seed_pair,(int)nw,clo,npts,per_block,blocks,
                         W.dn,C.ninv,W.giant_base,W.giant_base+nw,W.da24,W.dmone,W.dx,W.dz);
            ++g_giant_seed.paired_chunks;g_giant_seed.paired_ladders+=blocks;
            g_giant_seed.scalar_h2d_avoided+=js.size()*8;
        } else {
            CK(cudaMemcpy(W.djs,js.data(),js.size()*8,cudaMemcpyHostToDevice));
            S2G_DISPATCH((int)nw,s2g_launch_ladder,(int)nw,(int)js.size(),W.dn,C.ninv,
                         W.dqx,W.dqz,W.da24,W.dmone,W.djs,W.dx,W.dz,false);
        }
        CK(cudaGetLastError());
        ++W.ladder_calls;W.ladder_points_total+=paired?blocks:js.size();
        ++g_giant_seed.chunks;g_giant_seed.points+=js.size();
        g_giant_seed.avoided_d2h_bytes+=16ull*nw*js.size();
        g_giant_seed.avoided_h2d_bytes+=16ull*nw*js.size();
        g_giant_seed.avoided_cpu_modmuls+=2*js.size();
        g_giant_seed.avoided_montmuls+=2*js.size();
        if(g_giant_seed_check) {
            lx.resize(js.size()*nw);lz.resize(js.size()*nw);
            CK(cudaMemcpy(lx.data(),W.dx,lx.size()*8,cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(lz.data(),W.dz,lz.size()*8,cudaMemcpyDeviceToHost));
            std::vector<unsigned long long> cx,cz,expect(nw);
            ladder_points(C,js,cx,cz); // independent normal-output endpoint, not W's lease
            mpz_t old_z,new_z;mpz_inits(old_z,new_z,nullptr);
            for(size_t i=0;i<js.size();++i)for(int z=0;z<2;++z) {
                if(paired) {
                    if(z)continue; // Projective scale differs; compare x by cross products.
                    words_to_mpz(X,cx.data()+i*nw,nw);words_to_mpz(new_z,lz.data()+i*nw,nw);
                    mpz_mul(X,X,new_z);mpz_mod(X,X,L.N);
                    words_to_mpz(Z,lx.data()+i*nw,nw);words_to_mpz(old_z,cz.data()+i*nw,nw);
                    mpz_mul(Z,Z,old_z);mpz_mod(Z,Z,L.N);
                    if(mpz_cmp(X,Z)!=0){std::fprintf(stderr,"FATAL: paired seed GMP projective mismatch point=%llu\n",(unsigned long long)i);std::exit(3);}
                    g_giant_seed.checked_words+=2*nw;continue;
                }
                words_to_mpz(X,(z?cz:cx).data()+i*nw,nw);mpz_mul(X,X,Rm);mpz_mod(X,X,L.N);
                mpz_to_words(expect,nw,X);
                if(!std::equal(expect.begin(),expect.end(),(z?lz:lx).begin()+i*nw)) {
                    std::fprintf(stderr,"%s: FATAL: device seed GMP image mismatch point=%llu\n",NTT_PROBE_NAME,(unsigned long long)i);std::exit(3);
                }
                g_giant_seed.checked_words+=nw;
            }
            mpz_clears(old_z,new_z,nullptr);
        }
    } else ladder_points_ws(W, js, lx, lz);
    /* the seeds and the difference point as MONTGOMERY IMAGES (x*R mod N) */
    std::vector<unsigned long long> hdsx,hdsz,hesx,hesz,hdx,hdz;
    if(!g_giant_seed_device) {
        hdsx.resize((size_t)blocks*nw);hdsz.resize((size_t)blocks*nw);
        hesx.resize((size_t)blocks*nw);hesz.resize((size_t)blocks*nw);hdx.resize(nw);hdz.resize(nw);
        std::vector<unsigned long long> tmpw(nw, 0ull);
        auto img = [&](std::vector<unsigned long long> &dst, size_t off, size_t e, bool use_z) {
            words_to_mpz(X, use_z ? &lz[e * nw] : &lx[e * nw], nw);
            mpz_mul(X, X, Rm);
            mpz_mod(X, X, L.N);
            mpz_to_words(tmpw, nw, X);
            std::copy(tmpw.begin(), tmpw.end(), dst.begin() + (long)off);
        };
        for (unsigned long long b = 0; b < blocks; ++b) {
            img(hdsx, (size_t)b * nw, (size_t)(2 * b), false);
            img(hdsz, (size_t)b * nw, (size_t)(2 * b), true);
            img(hesx, (size_t)b * nw, (size_t)(2 * b + 1), false);
            img(hesz, (size_t)b * nw, (size_t)(2 * b + 1), true);
        }
        img(hdx, 0, (size_t)(2 * blocks), false);
        img(hdz, 0, (size_t)(2 * blocks), true);
    }
    unsigned long long *ddsx = nullptr, *ddsz = nullptr, *desx = nullptr, *desz = nullptr,
                       *ddx = nullptr, *ddz = nullptr, *ox = nullptr, *oz = nullptr;
    const size_t nseed = (size_t)blocks * nw;
    if(g_giant_seed_device) {
        ddsx=W.dx;ddsz=W.dz;desx=W.dx+nw;desz=W.dz+nw;
        ddx=W.dx+2*nseed;ddz=W.dz+2*nseed;
    } else {
    CK(cudaMalloc(&ddsx, nseed * 8));
    CK(cudaMalloc(&ddsz, nseed * 8));
    CK(cudaMalloc(&desx, nseed * 8));
    CK(cudaMalloc(&desz, nseed * 8));
    CK(cudaMalloc(&ddx, nw * 8));
    CK(cudaMalloc(&ddz, nw * 8));

    CK(cudaMemcpy(ddsx, hdsx.data(), nseed * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(ddsz, hdsz.data(), nseed * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(desx, hesx.data(), nseed * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(desz, hesz.data(), nseed * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(ddx, hdx.data(), nw * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(ddz, hdz.data(), nw * 8, cudaMemcpyHostToDevice));
    }
    CK(cudaMalloc(&ox, (size_t)npts * nw * 8));
    CK(cudaMalloc(&oz, (size_t)npts * nw * 8));
    S2G_DISPATCH((int)nw, s2g_launch_chain, (int)nw, blocks, npts, per_block, C.ninv, W.dn, ddx,
                 ddz, ddsx, ddsz, desx, desz, ox, oz,g_giant_seed_device?2:1);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    if(!resident || check || g_gfinv_seg_check) {
        gx.assign((size_t)npts*nw,0);gz.assign((size_t)npts*nw,0);
        CK(cudaMemcpy(gx.data(),ox,gx.size()*8,cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(gz.data(),oz,gz.size()*8,cudaMemcpyDeviceToHost));
    } else {std::vector<unsigned long long>().swap(gx);std::vector<unsigned long long>().swap(gz);}
    /* the per-segment z-products, while the point buffers are still on the device (section 42) */
    unsigned long long *dsp = nullptr;
    if (gseg) {
        const unsigned long long nseg = (npts + seg - 1) / seg;
        gseg->assign((size_t)nseg * nw, 0ull);
        CK(cudaMalloc(&dsp, (size_t)nseg * nw * 8));
        if(g_gfinv_seg_exact)W.need_segfix((size_t)seg);
        S2G_DISPATCH((int)nw, s2g_launch_segprod, (int)nw, npts, seg, C.ninv, W.dn, oz, dsp,
                    g_gfinv_seg_exact?W.dsegfix:nullptr);
        g_giant_seed.segments+=nseg;
        if(g_gfinv_seg_exact) {
            g_giant_seed.segment_fix_muls+=nseg-((npts%seg)==1?1:0);
            g_giant_seed.fix_table_peak_bytes=std::max(g_giant_seed.fix_table_peak_bytes,8ull*(W.segfix_cap+1)*nw);
        }
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        if(!resident || g_gfinv_seg_check)CK(cudaMemcpy(gseg->data(),dsp,gseg->size()*8,cudaMemcpyDeviceToHost));
        if(!resident)cudaFree(dsp);
        if(g_gfinv_seg_check) {
            if(seg!=S2G_GFINV_SEG){std::fprintf(stderr,"FATAL: segment check grid mismatch\n");std::exit(3);}
            std::vector<unsigned long long> expected;
            gfinv_segprod_host(expected,(size_t)npts,nw,gz,L.N);
            if(expected!=*gseg){std::fprintf(stderr,"%s: FATAL: segment product GMP mismatch\n",NTT_PROBE_NAME);std::exit(3);}
            g_giant_seed.segment_checks+=nseg;
        }
    }
    if(!g_giant_seed_device) {
        cudaFree(ddsx);cudaFree(ddsz);cudaFree(desx);cudaFree(desz);cudaFree(ddx);cudaFree(ddz);
    }
    if(resident){resident->x=ox;resident->z=oz;resident->segments=dsp;resident->n=npts;resident->w=nw;resident->own=true;}
    else {cudaFree(ox);cudaFree(oz);}
    /* THE CHECK: the same chunk through the ladder, compared on the AFFINE x value (the chain and
       the ladder hold different projective representatives of the same point, so X and Z cannot
       be compared -- X/Z mod N can, and that is the only quantity the caller uses). */
    if (check) {
        std::vector<unsigned long long> gjs((size_t)npts, 0ull);
        for (unsigned long long i = 0; i < npts; ++i) gjs[(size_t)i] = (clo + i) * D;
        std::vector<unsigned long long> cx, cz;
        ladder_points_ws(W, gjs, cx, cz);
        std::vector<unsigned long long> a1(nw, 0ull), a2(nw, 0ull);
        unsigned long long bad = 0, first = 0;
        for (unsigned long long i = 0; i < npts; ++i) {
            mpz_t X1, Z1;
            mpz_inits(X1, Z1, nullptr);
            words_to_mpz(X1, &gx[(size_t)i * nw], nw);
            words_to_mpz(Z1, &gz[(size_t)i * nw], nw);
            affine_x_gmp(X, X1, Z1, L.N);
            mpz_to_words(a1, nw, X);
            words_to_mpz(X1, &cx[(size_t)i * nw], nw);
            words_to_mpz(Z1, &cz[(size_t)i * nw], nw);
            affine_x_gmp(X, X1, Z1, L.N);
            mpz_to_words(a2, nw, X);
            if (a1 != a2) {
                if (!bad) first = i;
                /* the PATTERN matters: sparse mismatches inside blocks mean an arithmetic edge
                   case, a whole block means a seed problem, and everything after one index means
                   a truncated transfer.  Print the first eight. */
                if (bad < 8)
                    std::fprintf(stderr, "giant_chain_mismatch: i=%llu block=%llu off=%llu\n",
                                 i, i / per_block, i % per_block);
                ++bad;
            }
            /* THE DEGENERACY WINDOW: a giant point whose Z is not invertible mod N is a point
               where stage 2 has effectively FOUND a factor (gcd(Z, N) > 1), and affine_x_gmp
               deliberately falls back to X for it.  Such a point cannot be compared between two
               different projective representatives -- and, more importantly, the CHAIN must not
               run through it (the xADD formula is undefined with Z = 0 mod q).  Print the window
               around the first mismatch so the two explanations can be told apart by inspection. */
            if (bad && i <= first + 12) {
                mpz_t Zc, Zl, g;
                mpz_inits(Zc, Zl, g, nullptr);
                words_to_mpz(Zc, &gz[(size_t)i * nw], nw);
                words_to_mpz(Zl, &cz[(size_t)i * nw], nw);
                mpz_gcd(g, Zc, L.N);
                const bool zc_bad = (mpz_cmp_ui(g, 1) > 0);
                mpz_gcd(g, Zl, L.N);
                const bool zl_bad = (mpz_cmp_ui(g, 1) > 0);
                std::fprintf(stderr, "giant_chain_window: i=%llu equal=%d chain_Z_noninvertible=%d "
                                     "ladder_Z_noninvertible=%d\n", i, (a1 == a2) ? 1 : 0,
                             zc_bad ? 1 : 0, zl_bad ? 1 : 0);
                mpz_clears(Zc, Zl, g, nullptr);
            }
            mpz_clears(X1, Z1, nullptr);
        }
        stage2_log::print(stage2_log::debug, "giant_chain_check: points=%llu blocks=%llu per_block=%llu seed_points=%llu "
                    "mismatches=%llu first=%llu\n", npts, blocks, per_block, seed_points, bad,
                    first);
    }
    mpz_clears(Rm, X, Z, nullptr);
}

/* the block product of nvals values, computed ON the device from a device buffer */
static void dev_block_products(S3Workspace &W, int nvals, int per_block,
                               std::vector<std::vector<unsigned long long>> &out)
{
    const LadderCtx &C = *W.C;
    const size_t nw = C.nw;
    const int nb = (nvals + per_block - 1) / per_block;
    out.assign((size_t)nb, std::vector<unsigned long long>(nw, 0ull));
    if (nvals == 0) return;
    S2G_DISPATCH((int)nw, s2g_launch_block_prod, (int)nw, nvals, per_block, W.dvals, W.dn,
                 C.ninv, W.dprod);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    std::vector<unsigned long long> flat((size_t)nb * nw, 0ull);
    CK(cudaMemcpy(flat.data(), W.dprod, flat.size() * 8, cudaMemcpyDeviceToHost));
    for (int b = 0; b < nb; ++b)
        for (size_t i = 0; i < nw; ++i) out[(size_t)b][i] = flat[(size_t)b * nw + i];
    ++W.prod_launches;
}

/* the reference's record(): keep a factor that really divides N, and its hit prime.
   `count_hit` = whether this record also counts as an ATTRIBUTED hit.  The batched engine's
   fallback (a hit leaf whose stage-2 prime could not be identified) records the factor but must
   NOT inflate `hits`: `hits`/`hit_primes` are compared against the CPU reference by the
   acceptance gate, while the factor set is reported independently. */
static void s3_record(Stage2Tail &out, const mpz_t f, unsigned long long prime, const mpz_t N,
                      bool count_hit = true)
{
    if (mpz_cmp_ui(f, 1) <= 0 || mpz_cmp(f, N) == 0) return;
    mpz_t tq;
    mpz_init(tq);
    mpz_mod(tq, N, f);
    if (mpz_cmp_ui(tq, 0) != 0) ++out.bad_factors;
    mpz_clear(tq);
    char *s = mpz_get_str(nullptr, 10, f);
    bool seen = false;
    for (const std::string &v : out.factors)
        if (v == s) seen = true;
    if (!seen) out.factors.push_back(s);
    void (*ff)(void *, size_t) = nullptr;
    mp_get_memory_functions(nullptr, nullptr, &ff);
    ff(s, std::strlen(s) + 1);
    if (prime) out.hit_primes.push_back(prime);
    if (count_hit) ++out.hits;
}

/* ===================================================================================== *
 *  THE CURVE SETUP FOR A REAL SHAPE (slice S4 / section 17's P2)
 *
 *  The frozen vector takes a24 and Q from the CPU reference's dump, and that is the right
 *  oracle for it.  A REAL shape (S = 5261, B2 = 1.94e12) cannot: the CPU reference would have
 *  to build F over ~9e4 baby points with GMP schoolbook.  This engine therefore builds its own
 *  input point, with the reference's own formulas:
 *
 *      a24 = (A+2)/4,  A = (v-u)^3 (3u+v) / (4 u^3 v) - 2,  u = sigma^2-5, v = 4 sigma
 *      P0  = (u^3 : v^3)                       (the reference's suyama_curve, to the letter)
 *      s   = lcm(1..B1) = prod p^floor(log_p B1)
 *      Q   = [s] P0
 *
 *  [s]P0 is computed as a CHAIN of device ladders, one per prime power p^e <= B1: the ladder
 *  is [k] on its own base point, so applying [p1^e1], then [p2^e2] to the result, ... gives
 *  exactly [prod p^e]P0 -- and every step goes through the SAME s2g_ladder kernels the baby
 *  points use (one implementation of the curve arithmetic, no new formulas).  The Montgomery
 *  hand-off between steps is two host mpz multiplications by R.
 *
 *  The whole path is CHECKED against the CPU reference wherever a dump exists
 *  (setup_check below): recomputing a24 and Q from (N, sigma, B1) must reproduce the dump's
 *  a24_hex and the affine x of its Q_hex.
 * ===================================================================================== */

static void set_u64_mpz(mpz_t r, unsigned long long v)
{
    mpz_set_ui(r, (unsigned long)(v >> 32));
    mpz_mul_2exp(r, r, 32);
    mpz_add_ui(r, r, (unsigned long)(v & 0xFFFFFFFFull));
}

/* the reference's suyama_curve: a24 = (A+2)/4 and P0 = (u^3 : v^3). */
static int suyama_curve(mpz_t a24, mpz_t px, mpz_t pz, unsigned long long sigma, const mpz_t N)
{
    mpz_t sig, u, v, num, den, inv, A, t;
    mpz_inits(sig, u, v, num, den, inv, A, t, nullptr);
    set_u64_mpz(sig, sigma);
    mpz_mul(u, sig, sig);
    mpz_sub_ui(u, u, 5);                        /* u = sigma^2 - 5 */
    mpz_mul_ui(v, sig, 4);                      /* v = 4 sigma */
    mpz_sub(num, v, u);
    mpz_powm_ui(num, num, 3, N);                /* (v-u)^3 */
    mpz_mul_ui(t, u, 3);
    mpz_add(t, t, v);                           /* 3u+v */
    mpz_mul(num, num, t);
    mpz_powm_ui(den, u, 3, N);
    mpz_mul(den, den, v);
    mpz_mul_ui(den, den, 4);                    /* 4 u^3 v */
    int rc = 0;
    if (mpz_invert(inv, den, N) == 0) { rc = 1; goto done; }
    mpz_mul(A, num, inv);
    mpz_sub_ui(A, A, 2);                        /* A = (v-u)^3(3u+v)/(4u^3v) - 2 */
    mpz_add_ui(t, A, 2);
    mpz_set_ui(den, 4);
    if (mpz_invert(inv, den, N) == 0) { rc = 1; goto done; }
    mpz_mul(a24, t, inv);
    mpz_mod(a24, a24, N);                       /* a24 = (A+2)/4 */
    mpz_powm_ui(px, u, 3, N);                   /* X0 = u^3 */
    mpz_powm_ui(pz, v, 3, N);                   /* Z0 = v^3 */
done:
    mpz_clears(sig, u, v, num, den, inv, A, t, nullptr);
    return rc;
}

/* the prime powers p^e <= B1 (their product is lcm(1..B1)), GROUPED so that one device ladder
   covers as many as fit in a u64 multiplier: the scalar is the same product either way (the
   ladder is applied to the same total exponent), but 168 separate one-point launches cost
   11.7 s at S=5261 -- the per-LAUNCH cost again -- while ~24 grouped ones cost ~1.7 s. */
struct BatchedRun {
    DeviceGLeafStats device_leaf;
    FoldFlatStats fold_flat;
    FoldDeviceStats fold_device;
    Stage2Tail tail;
    unsigned long long giant_points = 0, num_poly_g = 0, loops = 0, P = 0;
    unsigned long long descent_divmods = 0, leaf_values = 0, apply_blocks = 0, block_per = 0;
    unsigned long long small_primes = 0;
    unsigned long long hit_blocks = 0, named_searches = 0, candidates_tested = 0;
    /* the naming loop's own accounting: a HIT LEAF is a leaf whose value shares a factor with N
       (there can be many of them -- the batched engine evaluates H at the BABY points, and one
       stage-2 prime can be covered by several baby points, so `hit_leaves` is NOT comparable
       with the CPU reference's `hits`, whose leaves are the GIANT points).  `unnamed` counts hit
       leaves whose scan found no confirming prime: their factor is recorded from the leaf gcd
       itself, exactly as the reference's name_culprit does. */
    unsigned long long hit_leaves = 0, unnamed = 0;
    double t_scan = 0.0, t_ladder = 0.0;
    /* the giant-point chain accounting: how many chunks went through the chain and how many
       LADDER points the seeds cost (a chain needs two seeds per block, so this is the part of the
       phase that is not one-xADD-per-point -- section 31) */
    unsigned long long giant_chain_chunks = 0, giant_seed_points = 0;
    /* objective 3: giant points whose Z shares a factor with N (the hit made explicit) */
    unsigned long long giant_degenerate = 0;
    unsigned long long unnamed_hits = 0, cand_lists = 0;
    /* device-side bookkeeping, so the breakdown line reports measured work, not a guess */
    unsigned long long ladder_calls = 0, ladder_points = 0, prod_launches = 0;
    unsigned long long arena_fuse_builds = 0, arena_fuse_reuse = 0, arena_buf_builds = 0,
                       arena_buf_reuse = 0, arena_overflow = 0;
    double arena_mb = 0.0;
    unsigned long long ntt_calls = 0;
    /* slice S5: the device descent's own accounting (its leaf values live on the device) */
    unsigned long long s5_divmods = 0, s5_generic = 0, s5_linear = 0, s5_copies = 0, s5_zeros = 0,
                       s5_ntt_launches = 0, s5_forest_bytes = 0, s5_frontier_bytes = 0;
    double s5_t_ntt = 0.0, s5_t_pack = 0.0, s5_t_generic = 0.0, s5_t_copy = 0.0;
    bool s5_readback = false;
    double t_giant = 0.0, t_gtrees = 0.0, t_fold = 0.0, t_descent = 0.0, t_inv = 0.0,
           t_accum = 0.0, t_name = 0.0;
    /* ---- THE BOOKS MUST BALANCE (section 41) -------------------------------------------
       `batched_progress: t=` is the SUM of the three loop timers, not a wall clock, so a run
       can look fully accounted while the loop is really much longer: at the production shape
       those three add up to 308.6 s and the curve's wall is 555.3 s, and the missing ~186 s
       was invisible for several rounds because nothing timed the loop ITSELF.  These four
       numbers close the loop: pre + loop_wall + post must equal the caller's `elapsed`, and
       `gleaves` is the one part of the loop body that no phase timer owned (the host GMP
       affine conversion of every giant point, section 31.4/38). */
    double t_pre_loop = 0.0, t_loop_wall = 0.0, t_post_loop = 0.0, t_gleaves = 0.0;
    /* the one pass over H that removes the projective scale (section 42) */
    double t_gscale = 0.0;
    PlainScaleStats gscale;
    /* ---- THE PROJECTIVE BOOKKEEPING MUST BE EXACTLY SELF-CONSISTENT ----------------------
       The factor set CANNOT validate Gamma: any invertible Gamma gives the same gcds, so a
       doubled or missing factor would still "pass".  The invariant that can be checked is
       combinatorial and exact: every projective leaf must be covered by exactly one accumulated
       segment product, i.e. gamma_points == proj_points at the end of the run.  A missed segment
       makes it too small, a double count makes it too large, and either way it is a hard error
       rather than a silently different scale. */
    unsigned long long proj_points = 0, proj_gamma_points = 0, proj_segments = 0, proj_fallbacks = 0;
    /* ... and the four parts of `gleaves`, because "the host affine conversion" is 36% of the
       whole curve and each part has a different fix (section 41) */
    double t_gin = 0.0, t_ginv = 0.0, t_gmul = 0.0, t_gout = 0.0;
    bool dbg_progress = false;              /* one phase line per G-tree batch (long shapes) */
};

/* Baby normalization proves unit Zs by batch inversion and computes every nonunit gcd.
   Keep that proof for this curve, not coordinates. Exact input keys and actual sorted baby
   indices prevent reuse across a different Q, curve, modulus, bounds, or incomplete baby set. */
static bool small_prime_flag(const char *name) {
    const char *e=std::getenv(name);return e && std::atoi(e)!=0;
}
struct SmallPrimeBabyCache {
    LadderCtx key;
    unsigned long long D=0,B1=0,B2=0;
    std::vector<unsigned long long> js;
    std::vector<std::pair<size_t,std::string>> nonunit;
    bool ready=false;
    void begin(const LadderCtx &C,unsigned long long d,unsigned long long b1,unsigned long long b2,
               const std::vector<unsigned long long> &indices) {
        key=C;D=d;B1=b1;B2=b2;js=indices;nonunit.clear();ready=false;
    }
    void record(size_t index,const mpz_t gcd) {
        char *v=mpz_get_str(nullptr,10,gcd);nonunit.emplace_back(index,v);
        void (*release)(void*,size_t)=nullptr;mp_get_memory_functions(nullptr,nullptr,&release);
        release(v,std::strlen(v)+1);
    }
    bool matches(const LadderCtx &C,const Stage2Params &SP) const {
        return ready && D==SP.D && B1==SP.B1 && B2==SP.B2 && js==SP.baby_j &&
            key.nw==C.nw && key.ninv==C.ninv && key.hn==C.hn && key.hqx==C.hqx &&
            key.hqz==C.hqz && key.ha24==C.ha24 && key.hmone==C.hmone;
    }
    void gcd_at(size_t index,mpz_t g) const {
        const auto it=std::lower_bound(nonunit.begin(),nonunit.end(),index,
            [](const auto &v,size_t i){return v.first<i;});
        if(it!=nonunit.end() && it->first==index)mpz_set_str(g,it->second.c_str(),10);
        else mpz_set_ui(g,1);
    }
    size_t payload_bytes() const {
        size_t v=js.capacity()*8+(key.hn.capacity()+key.hqx.capacity()+key.hqz.capacity()+
            key.ha24.capacity()+key.hmone.capacity())*8+nonunit.capacity()*sizeof(nonunit[0]);
        for(const auto &e:nonunit)v+=e.second.capacity()+1;
        return v; // capacity ledger, not process private bytes or allocator overhead
    }
};

#include "stage2/stage2_baby_host.cuh"

/* the batched structure itself.  Ft/Fdeg/Fpad is the F product tree (heap, degrees, padded
   leaf count) that run_check_F already built and verified coefficient by coefficient. */
static BatchedRun run_batched(PolyLayer &L, const LadderCtx &C, const Stage2Params &SP,
                              const std::vector<std::vector<unsigned long long>> &Ft,
                              const std::vector<size_t> &Fdeg, size_t Fpad,
                              const SmallPrimeBabyCache *baby_cache=nullptr)
{
    const size_t W = L.W;
    const unsigned long long D = SP.D, B1 = SP.B1, B2 = SP.B2;
    const unsigned long long imax = B2 / D + 2;          /* the SAME giant set S2/the CPU ref use */
    const double t_entry = now_s();
    const size_t P = Fdeg[1];                            /* deg F = the baby count = poly_size */
    if(!L.s4) {
        std::fprintf(stderr,"%s: FATAL: scaled descent needs S4 and cannot be combined with S5\n",NTT_PROBE_NAME);
        std::exit(3);
    }
    const bool fold_flat_enabled=g_fold_flat && L.s4;
    BatchedRun R;
    R.fold_device.root_requested=fold_device_flag("NTT_GROOT_TO_FOLD");
    R.fold_flat.enabled=fold_flat_enabled;
    R.P = P;
    R.giant_points = imax;
    R.dbg_progress = true;                  /* one line per G-tree batch: gate-able, machine-readable */
    L.cost.S = (long)L.S;                              /* (already set by the caller) */

    mpz_t g, pg;
    mpz_inits(g, pg, nullptr);

    /* every device buffer this engine needs, allocated ONCE for the whole curve */
    S3Workspace ws;
    ws.init(C);

    /* ---- 0. Small primes cannot be reached by iD+-j for i>=1. Reuse only the
       exact baby normalization proof; missing p (including p|D) uses the original ladder. */
    {
        const double ts=now_s();
        const bool requested=small_prime_flag("NTT_SMALL_PRIME_REUSE");
        const bool matched=requested && baby_cache && baby_cache->matches(C,SP);
        const bool check=small_prime_flag("NTT_SMALL_PRIME_CHECK");
        const unsigned long long half=(D<2)?1:D/2;
        std::vector<unsigned long long> smalljs,missing;
        std::vector<size_t> cached;
        size_t cursor=0;
        unsigned long long reused=0,avoided_mont=0,checked=0,bad=0;
        for(unsigned long long p=2;p<=half;++p) {
            if(p<=B1 || p>B2 || !is_prime_u64(p))continue;
            smalljs.push_back(p);
            if(matched)while(cursor<baby_cache->js.size() && baby_cache->js[cursor]<p)++cursor;
            const bool covered=matched && cursor<baby_cache->js.size() && baby_cache->js[cursor]==p;
            cached.push_back(covered?cursor:~size_t(0));
            if(covered) {
                ++reused;unsigned long long bits=0,v=p;while(v){++bits;v>>=1;}
                avoided_mont+=13*bits-6;
            } else missing.push_back(p);
        }
        R.small_primes=smalljs.size();
        std::vector<unsigned long long> sx,sz,cx,cz;
        ladder_points_ws(ws,missing,sx,sz);
        if(check && reused)ladder_points_ws(ws,smalljs,cx,cz);
        mpz_t z,expected;mpz_inits(z,expected,nullptr);
        size_t fallback=0;
        for(size_t k=0;k<smalljs.size();++k) {
            if(cached[k]!=~size_t(0)) {
                baby_cache->gcd_at(cached[k],pg);
                if(check) {
                    words_to_mpz(z,cz.data()+k*W,W);mpz_gcd(expected,z,L.N);
                    if(small_prime_flag("NTT_SMALL_PRIME_TEST_BAD") && checked==0)mpz_add_ui(expected,expected,1);
                    if(mpz_cmp(pg,expected))++bad;
                    ++checked;
                }
            } else {
                words_to_mpz(z,sz.data()+fallback++*W,W);mpz_gcd(pg,z,L.N);
            }
            if(mpz_cmp_ui(pg,1)>0 && mpz_cmp(pg,L.N)<0)s3_record(R.tail,pg,smalljs[k],L.N);
        }
        mpz_clears(z,expected,nullptr);
        stage2_log::print(stage2_log::debug, "small_prime_reuse: requested=%d available=%d matched=%d primes=%llu reused=%llu fallback=%llu checked=%llu bad=%llu avoided_montmuls=%llu avoided_h2d_bytes=%llu avoided_d2h_bytes=%llu cache_bytes=%llu elapsed=%.6f\n",
            (int)requested,(int)(baby_cache && baby_cache->ready),(int)matched,R.small_primes,reused,
            (unsigned long long)missing.size(),checked,bad,avoided_mont,reused*8,reused*16*W,
            (unsigned long long)(baby_cache?baby_cache->payload_bytes():0),now_s()-ts);
        if(bad){std::fprintf(stderr,"FATAL: small-prime baby GCD proof mismatch\n");std::exit(3);}
    }

    /* ---- 1. THE GIANT POINTS, ONE BATCH AT A TIME -------------------------------------
       S3 computed ALL of them up front (one ladder launch + a device buffer per point).  At the
       real shape that is imax = 3.4e6 points x 83 words x 2 coordinates = 4.5 GB of device
       memory AND the same again on the host -- "out of memory" before the first G tree.  The
       engine only ever needs ONE batch of P giant points at a time (that is what the G tree is
       built from), so the points are computed per batch and the buffers are reused.  The
       per-batch cost is the same affine_x conversion as before (host GMP, ~10 us per point);
       what disappears is the peak memory. */
    std::vector<unsigned long long> gjs;
    std::vector<unsigned long long> gx, gz;
    /* the per-segment z-products (section 42), one chunk at a time */
    std::vector<unsigned long long> gseg;
    unsigned long long proj_points = 0, proj_gamma_points = 0, proj_segments = 0,
                       proj_fallbacks = 0;
    /* GAMMA^-1: the ONE global correction the projective leaves need.  The G tree of a batch
       whose leaves are projective is GAMMA_b * prod(X - x_i) instead of prod(X - x_i), and the
       folds carry that constant through (H <- (G*H) mod F scales by the same constant), so H
       comes out of the loop as GAMMA * prod_b f_b.  GAMMA = prod of the z values of exactly the
       points whose leaf is projective, i.e. a product of INVERTIBLE elements -- which is why it
       is invertible, and why multiplying H by GAMMA^-1 before the descent reproduces the old
       polynomial EXACTLY (mod N, coefficient by coefficient). */
    mpz_t Ginv;
    mpz_init_set_ui(Ginv, 1);

    /* ---- 2. the outer loop: ceil(I/P) batches, the first one SEEDS H ------------------- */
    R.num_poly_g = (imax + (unsigned long long)P - 1) / (unsigned long long)P;
    if (R.num_poly_g == 0) R.num_poly_g = 1;
    R.loops = R.num_poly_g - 1;

    /* 1/F: the Newton inverse of the REVERSED, MONIC F, to the length a full-size mod-F
       reduction needs (deg T = 2P => k = P+1).  Computed ONCE and reused by every fold --
       the reference's cost model has no such term, so it is charged to its own category. */
    CPoly finv;
    const CPoly Fpoly = fold_flat_enabled ? CPoly{} : cp_from_flat(Ft[1], Fdeg[1], W);
    const double ti0 = now_s();
    if (R.loops > 0) {
        CPoly revF;
        cp_resize(revF, P + 1, W);
        for (size_t i = 0; i <= P; ++i)                 /* revF[i] = F[P-i] (F is monic) */
            std::copy(Ft[1].begin() + (long)((P - i) * W),
                      Ft[1].begin() + (long)((P - i + 1) * W), revF[i].begin());
        L.cat = BC_FINV;                               /* the Newton inverse of rev(F) */
        finv = cp_inv_series(revF, P + 1, L);
        L.cat = -1;
    }
    R.t_inv = now_s() - ti0;
    std::vector<unsigned long long> Hflat,finvflat;
    if(fold_flat_enabled && !finv.empty()) {
        const double tb=now_s();finvflat=cp_to_flat(finv,W);CPoly{}.swap(finv);
        R.fold_flat.t_bridge+=now_s()-tb;
    }
    FoldDeviceState device_fold;
    if(fold_flat_enabled && !finvflat.empty())device_fold.init(L,Ft[1],finvflat,R.fold_device,R.fold_flat);
    CPoly H;
    FTreeStats gs;
    /* the giant points are computed in POINT CHUNKS that are a whole number of G-tree batches
       and bounded in memory (2 coordinates x nw words x 8 bytes each): one chunk for the frozen
       vector (4763 points -> the single ladder launch S3 used), ~17 for the real shape. */
    const unsigned long long pts_budget_bytes = (unsigned long long)256 << 20;
    unsigned long long pts_per_chunk = P;
    {
        const unsigned long long per_point = 2 * (unsigned long long)C.nw * 8;
        unsigned long long k = pts_budget_bytes / (per_point ? per_point : 1);
        if (k < P) k = P;
        pts_per_chunk = P * ((k + P - 1) / P);
    }
    for (unsigned long long c0 = 0; c0 < imax; c0 += pts_per_chunk) {
        if (c0 == 0) R.t_pre_loop = now_s() - t_entry;
        const double tloop0 = now_s();
        const unsigned long long c1 = ((imax - c0) < pts_per_chunk) ? imax : (c0 + pts_per_chunk);
        const size_t clo = (size_t)(c0 + 1), chi = (size_t)c1;
        const double tgp = now_s();
        s2g_state("giant ladder chunk: about to launch");   /* the last state on a driver kill */
        /* THE CHAIN BY DEFAULT (see s2g_chain_kernel): one differential addition per giant point
           instead of a full ladder.  NTT_GIANT_LADDER=1 forces the old per-point ladder (kept as
           the oracle), NTT_GIANT_CHAIN_CHECK=1 runs BOTH and compares the affine x values,
           NTT_GIANT_CHAIN_BLOCK=n sets the points per thread (default 256: at the real shape that
           is 786 blocks per chunk, one seed launch, and ~10-30 ms of chain work per chunk). */
        /* NTT_GIANT_CHAIN_BLOCK: points per thead.  The default is deliberately SMALL (64) even
           though a larger block is cheaper in arithmetic: a chain that passes through a point
           with Z not invertible mod N (i.e. a point that is the identity modulo one of N's
           factors -- exactly the hit stage 2 is looking for!) carries that degeneracy in its Z
           into every following point of the block, and those leaves are then garbage.  Measured
           at rung 2: the single degenerate index 3511 (= the residual order of the factor 42089)
           contaminates the rest of whatever block contains it and nothing else -- the mismatch
           count is exactly (block_end - 3511) for every block size tried (9/73/89/73/489 for
           64/128/200/512/1000).  A small block bounds that damage; the seed cost is 2 ladder
           points per block, i.e. ~18 Montgomery multiplications per point at 64 (still ~32x less
           than the ladder's ~574).  Set NTT_GIANT_CHAIN_BLOCK to trade the two. */
        static const unsigned long long chain_block = [] {
            const char *e = std::getenv("NTT_GIANT_CHAIN_BLOCK");
            unsigned long long v = (e && *e) ? std::strtoull(e, nullptr, 10) : 64ull;
            if (v < 4) v = 4;
            if (v > 1u << 20) v = 1u << 20;
            return v;
        }();
        /* Opt-in short-chunk policy: more independent chains, shorter serial
           xADD loops. The global block remains unchanged for large chunks. */
        static const unsigned long long short_chain_block = [] {
            const char *e = std::getenv("NTT_GIANT_CHAIN_SMALL_BLOCK");
            unsigned long long v = (e && *e) ? std::strtoull(e, nullptr, 10) : 0ull;
            if(v && v < 4)v=4;
            return std::min(v,64ull);
        }();
        static const unsigned long long short_chain_max = [] {
            const char *e = std::getenv("NTT_GIANT_CHAIN_SMALL_MAX");
            unsigned long long v = (e && *e) ? std::strtoull(e, nullptr, 10) : 8192ull;
            return std::min(v,1ull<<20);
        }();
        static const bool force_ladder = [] {
            const char *e = std::getenv("NTT_GIANT_LADDER");
            return e && *e && std::atoi(e) != 0;
        }();
        static const bool chain_check = [] {
            const char *e = std::getenv("NTT_GIANT_CHAIN_CHECK");
            return e && *e && std::atoi(e) != 0;
        }();
        /* WHICH PATH, DECIDED BY SIZE -- measured, not guessed.  The chain's arithmetic per point
           is ~70x smaller (one xADD vs a full ladder), but it carries FIXED costs the ladder path
           does not: a second launch, six small uploads and a host-side Montgomery conversion of
           the seeds.  At rung 2 (imax=4331) that makes the chain SLOWER (giant=1.671 s vs
           1.097 s, measured), while at the production shape (imax=3.4e6) the ladder's arithmetic
           is 901 s and the chain's fixed costs are irrelevant.  So: below `chain_min` points per
           chunk use the ladder, above it use the chain.  NTT_GIANT_CHAIN_MIN=0 forces the chain
           (for the checks), NTT_GIANT_CHAIN_MIN=<big> forces the ladder. */
        static const unsigned long long chain_min = [] {
            const char *e = std::getenv("NTT_GIANT_CHAIN_MIN");
            return (e && *e) ? std::strtoull(e, nullptr, 10) : 32768ull;
        }();
        const unsigned long long npts = (unsigned long long)(chi - clo + 1);
        /* the per-SEGMENT z-product grid is shared by the two paths (section 42): the chain
           computes it on the device out of the Montgomery images it already produced, the
           ladder path computes the very same product on the host.  Both are the product of the
           z values AS RETURNED, which is exactly the scale the projective leaves carry. */
        ResidentGiant resident;
        const char *hp=std::getenv("NTT_S4_HOSTPACK"),*mb=std::getenv("NTT_DEVICE_GLEAF_MAX_MB");
        const unsigned long long maxbytes=(mb?std::strtoull(mb,nullptr,10):512ull)*1048576ull;
        const bool requested=device_gleaf_flag("NTT_DEVICE_GLEAF");
        if(requested)++R.device_leaf.requested_chunks;
        const bool device_leaf=requested && g_groot_device && g_s4_groot_only && L.s4 &&
            g_s4_pack_direct && !g_s4_final_readback && !(hp && std::atoi(hp)) && g_gfinv_seg_exact &&
            16ull*npts*W<=maxbytes;
        if(requested && !device_leaf)++R.device_leaf.fallback_chunks;
        gseg.assign(((size_t)npts + S2G_GFINV_SEG - 1) / S2G_GFINV_SEG * (size_t)C.nw, 0ull);
        const unsigned long long effective_chain_block=ecm_stage2::giant_chain_block(
            npts,chain_block,short_chain_block,short_chain_max);
        if(short_chain_block)stage2_log::print(stage2_log::debug, "giant_chain_policy: npts=%llu route=%s base_block=%llu block=%llu short_block=%llu short_max=%llu\n",
            npts,(force_ladder || npts<chain_min)?"ladder":"chain",chain_block,
            (force_ladder || npts<chain_min)?0ull:effective_chain_block,short_chain_block,short_chain_max);
        if (force_ladder || npts < chain_min) {
            gjs.resize(chi - clo + 1);
            for (size_t i = clo; i <= chi; ++i) gjs[i - clo] = (unsigned long long)i * D;
            if(device_leaf) {
                ws.need_pts(npts);CK(cudaMemcpy(ws.djs,gjs.data(),npts*8,cudaMemcpyHostToDevice));
                S2G_DISPATCH((int)W,s2g_launch_ladder,(int)W,(int)npts,ws.dn,C.ninv,ws.dqx,ws.dqz,ws.da24,ws.dmone,ws.djs,ws.dx,ws.dz);
                CK(cudaGetLastError());++ws.ladder_calls;ws.ladder_points_total+=npts;
                resident.x=ws.dx;resident.z=ws.dz;resident.n=npts;resident.w=W;
                CK(cudaMalloc(&resident.segments,gseg.size()*8));ws.need_segfix(S2G_GFINV_SEG);
                S2G_DISPATCH((int)W,s2g_launch_segprod,(int)W,npts,S2G_GFINV_SEG,C.ninv,ws.dn,ws.dz,resident.segments,ws.dsegfix);
                CK(cudaGetLastError());
                if(g_gfinv_seg_check) {
                    gx.resize(npts*W);gz.resize(npts*W);
                    CK(cudaMemcpy(gx.data(),ws.dx,gx.size()*8,cudaMemcpyDeviceToHost));
                    CK(cudaMemcpy(gz.data(),ws.dz,gz.size()*8,cudaMemcpyDeviceToHost));
                    std::vector<unsigned long long> expected;gfinv_segprod_host(expected,npts,W,gz,L.N);
                    CK(cudaMemcpy(gseg.data(),resident.segments,gseg.size()*8,cudaMemcpyDeviceToHost));
                    if(expected!=gseg){std::fprintf(stderr,"FATAL: ladder device segment GMP mismatch\n");std::exit(3);}
                } else {std::vector<unsigned long long>().swap(gx);std::vector<unsigned long long>().swap(gz);}
            } else {
                ladder_points_ws(ws,gjs,gx,gz);gfinv_segprod_host(gseg,npts,W,gz,L.N);
            }
        } else {
            unsigned long long seed_points = 0;
            giant_chunk_chain(L, C, ws, D, (unsigned long long)clo, (unsigned long long)chi,
                              effective_chain_block, gx, gz, chain_check, seed_points, &gseg,
                              S2G_GFINV_SEG,device_leaf?&resident:nullptr);
            R.giant_seed_points += seed_points;
            ++R.giant_chain_chunks;
        }
        R.t_giant += now_s() - tgp;
        if(device_leaf) {
            const double t=now_s(),ti=R.device_leaf.t_invert;resident.prepare(ws,L,gseg,Ginv,proj_gamma_points,proj_segments,R.device_leaf);
            R.t_gleaves+=now_s()-t;R.t_ginv+=R.device_leaf.t_invert-ti;
        }
        GfinvBatch segment_inverse(gseg,W,L.N,g_gfinv_batch);
        /* the G trees of the batches that lie inside this point chunk */
        for (unsigned long long b = c0 / P; b < R.num_poly_g && b * P < c1; ++b) {
        const size_t lo = (size_t)(b * (unsigned long long)P);
        size_t hi = lo + P;
        if (hi > (size_t)imax) hi = (size_t)imax;
        std::vector<std::vector<unsigned long long>> bleaf;
        if(!device_leaf)bleaf.assign(hi-lo,std::vector<unsigned long long>(2*W,0));
        std::map<size_t,std::vector<unsigned long long>> patches;
        auto leaf_at=[&](size_t i)->std::vector<unsigned long long>& {
            if(!device_leaf)return bleaf[i];
            auto &v=patches[i];if(v.empty())v.assign(2*W,0);return v;
        };
        auto coord_at=[&](size_t i,bool z)->const unsigned long long* {
            return device_leaf?resident.coord(i,z):(z?gz:gx).data()+i*W;
        };
        std::vector<size_t> bdeg;
        size_t bpad = 0;
        FTreeStats bs;
        {
            const double tgl0 = now_s();
            mpz_t X, Z, ax, neg, gq;
            mpz_inits(X, Z, ax, neg, gq, nullptr);
            /* ---- THE PROJECTIVE LEAF, AND WHY THE TREE CANNOT CHANGE (section 42) ------------
               x_i = X_i / Z_i mod N used to be one mpz_invert per giant point: MEASURED at 61.5 us
               for the production modulus while the two input conversions and the output conversion
               together are 0.75 us, so 224.8 s of a 621 s curve -- the largest single phase in the
               engine, larger than the whole NTT (234 s).  Section 41 got it to 132.8 s with
               Montgomery's trick on the host; this removes the inversion from the CLEAN path
               entirely.
               The leaf does not have to be monic.  If it is written PROJECTIVELY as
                   [ -X_i , Z_i ]      instead of      [ -x_i , 1 ],
               then the factor is  Z_i*X - X_i = Z_i*(X - x_i), i.e. exactly Z_i times the old
               leaf, and a product tree over a MIXED set of leaves is
                   (prod over the projective ones of Z_i) * prod over ALL of them (X - x_i).
               So a segment whose z-product is INVERTIBLE mod N writes its leaves projectively at
               word level (two word operations, no GMP), and a segment whose product is NOT
               invertible -- it contains one of the ~0.9% DEGENERATE giant points, gcd(Z_i,N) > 1,
               the hit of objective 3 -- falls back to the affine path VERBATIM, degeneracy record
               included.  Gamma = the product of exactly the projective segments' z values is
               therefore a product of invertible elements, and multiplying H by Gamma^-1 once
               before the descent reproduces the old polynomial coefficient for coefficient.
               ONE mpz_invert per SEGMENT, not per point, and that invert is ALSO the Gamma^-1
               factor it contributes -- nothing is computed twice.  The segment products come from
               the device for the chain path (s2g_segprod_kernel, out of the Montgomery images the
               chain already has) and from the host for the ladder path. */
            const size_t SEG = S2G_GFINV_SEG;
            /* plain arrays, not std::vector: mpz_t is an ARRAY type (__mpz_struct[1]) and cannot
               be held in a std::vector */
            mpz_t pv[32 + 1], zv[32];
            for (int s = 0; s <= 32; ++s) mpz_init(pv[s]);
            for (int s = 0; s < 32; ++s) mpz_init(zv[s]);
            mpz_t pinv, zinv, pseg, invp;
            mpz_inits(pinv, zinv, pseg, invp, nullptr);
            std::vector<unsigned long long> w(W, 0ull);
            /* THE SEGMENT GRID IS IN CHUNK-LOCAL INDICES, because that is the grid the device
               built `gseg` on (`giant_chunk_chain` segments [0,SEG), [SEG,2SEG), ... over the
               chunk's own points).  The point indices here are GLOBAL (lo = b*P), so the local
               index is `l - (clo-1)`; using the global one walked `gseg` off its end for every
               point after the first chunk -- silently at 1e11 (the read landed inside the heap)
               and as an access violation at the production shape, which is how it was found.
               `first_touch` keeps Gamma from counting a segment twice when a BATCH boundary
               falls inside it: only the batch that owns the segment's first point adds its
               inverse, and the other half's projective leaves are covered by that one factor. */
            const size_t lbase = clo - 1;               /* local 0 == global lbase */
            const size_t lhi = hi - lbase;
            for (size_t la = lo - lbase; la < lhi;) {
                const size_t sidx = la / SEG;
                const size_t send = (sidx + 1) * SEG;
                const size_t lend = (send < lhi) ? send : lhi;
                const size_t nb = lend - la;
                const bool first_touch = (la == sidx * SEG);
                /* a hard guard, because getting this wrong the first time was SILENT at 1e11 (the
                   out-of-range read landed inside the heap and only Gamma was wrong) and an access
                   violation at the production shape -- an index that must be in range is asserted,
                   not trusted */
                if ((sidx + 1) * W > gseg.size()) {
                    std::fprintf(stderr, "%s: FATAL: giant segment %llu is outside the %llu "
                                         "segment products of this chunk (local index %llu of "
                                         "%llu)\n", NTT_PROBE_NAME, (unsigned long long)sidx,
                                 (unsigned long long)(gseg.size() / W), (unsigned long long)la,
                                 (unsigned long long)lhi);
                    std::exit(3);
                }
                if(device_leaf && resident.good(sidx)) {
                    proj_points+=nb;la=lend;continue; // Gamma counted once by the exact whole group
                }
                const double t1 = now_s();
                words_to_mpz(pseg, &gseg[sidx * W], W);
                const double t2 = now_s();
                bool clean;
                if(g_gfinv_batch)clean=segment_inverse.get(invp,sidx);
                else {
                    ++g_gfinv.requests;++g_gfinv.individual_attempts;
                    clean=mpz_invert(invp,pseg,L.N)!=0;
                }
                const double t3 = now_s();
                R.t_gin += t2 - t1;
                R.t_ginv += t3 - t2;
                if (clean) {
                    if (first_touch) {
                        mpz_mul(Ginv, Ginv, invp);
                        mpz_mod(Ginv, Ginv, L.N);
                        /* the points the DEVICE's segment product covers (the last segment of a
                           chunk can be short) */
                        const size_t seglen = ((sidx + 1) * SEG < npts) ? SEG
                                                                        : (npts - sidx * SEG);
                        proj_gamma_points += seglen;
                        ++proj_segments;
                    }
                    if(device_leaf)proj_points+=nb;
                    else for (size_t q = la; q < lend; ++q) {
                        words_neg_mod_n(leaf_at(q + lbase - lo), 0, coord_at(q,false), C.hn.data(), W);
                        std::copy(coord_at(q,true), coord_at(q,true) + (long)W,
                                  leaf_at(q + lbase - lo).begin() + (long)W);
                        ++proj_points;
                    }
                    R.t_gout += now_s() - t3;
                    la = lend;
                    continue;
                }
                /* ---- the affine path, VERBATIM (Montgomery's trick inside the segment, and the
                   old per-point path for whatever still refuses to invert) ------------------ */
                bool seg_clean = true;
                mpz_set_ui(pv[0], 1);
                proj_fallbacks += nb;
                for (size_t j = 0; j < nb; ++j) {
                    const size_t q = la + j;
                    words_to_mpz(zv[j], coord_at(q,true), W);
                    if (mpz_sgn(zv[j]) == 0) { seg_clean = false; break; }
                    mpz_mul(pv[j + 1], pv[j], zv[j]);
                    mpz_mod(pv[j + 1], pv[j + 1], L.N);
                }
                if (seg_clean && mpz_invert(pinv, pv[nb], L.N) == 0) seg_clean = false;
                if (!seg_clean) {
                    for (size_t j = 0; j < nb; ++j) {
                        const size_t q = la + j, bi = q + lbase - lo;
                        words_to_mpz(X, coord_at(q,false), W);
                        words_to_mpz(Z, coord_at(q,true), W);
                        if (!affine_x_gmp_checked(ax, X, Z, L.N)) {
                            ++R.giant_degenerate;
                            mpz_gcd(gq, Z, L.N);
                            s3_record(R.tail, gq, 0, L.N, /*count_hit=*/false);
                        }
                        mpz_neg(neg, ax);
                        mpz_mod(neg, neg, L.N);
                        mpz_to_words(w, W, neg);
                        std::copy(w.begin(), w.end(), leaf_at(bi).begin());
                        leaf_at(bi)[W] = 1;
                    }
                    R.t_gout += now_s() - t3;
                    la = lend;
                    continue;
                }
                for (size_t j = nb; j-- > 0;) {
                    const size_t q = la + j, bi = q + lbase - lo;
                    mpz_mul(zinv, pinv, pv[j]);
                    mpz_mod(zinv, zinv, L.N);        /* Z_j^-1 */
                    mpz_mul(pinv, pinv, zv[j]);
                    mpz_mod(pinv, pinv, L.N);        /* inverse of the shorter product */
                    words_to_mpz(X, coord_at(q,false), W);
                    mpz_mul(ax, X, zinv);
                    mpz_mod(ax, ax, L.N);
                    mpz_neg(neg, ax);
                    mpz_mod(neg, neg, L.N);          /* the leaf (X - x_i) = [ -x_i, 1 ] */
                    mpz_to_words(w, W, neg);
                    std::copy(w.begin(), w.end(), leaf_at(bi).begin());
                    leaf_at(bi)[W] = 1;
                }
                R.t_gout += now_s() - t3;
                la = lend;
            }
            for (int s = 0; s <= 32; ++s) mpz_clear(pv[s]);
            for (int s = 0; s < 32; ++s) mpz_clear(zv[s]);
            mpz_clears(pinv, zinv, pseg, invp, nullptr);
            mpz_clears(X, Z, ax, neg, nullptr);
            R.t_gleaves += now_s() - tgl0;
        }
        const double tg0 = now_s();
        std::vector<std::vector<unsigned long long>> gt;
        bool root_handed_off=false;
        std::function<void(const unsigned long long*,size_t)> root_sink;
        if(device_fold.active && R.fold_device.root_requested && g_s4_groot_only) {
            root_sink=[&](const unsigned long long *src,size_t count) {
                device_fold.accept_device_root(src,count,b==0);root_handed_off=true;
            };
        }
        if(device_leaf) {
            auto fill=[&](unsigned long long *out) {
                const double t=now_s();const size_t first=lo-(clo-1),count=hi-lo;
                s2g_projective_leaf_kernel<<<(unsigned)((count+127)/128),128>>>(ws.dn,(int)W,resident.x,resident.z,first,count,out);
                CK(cudaGetLastError());
                for(auto it=patches.begin();it!=patches.end();) {
                    const size_t start=it->first;size_t next=start;std::vector<unsigned long long> words;
                    while(it!=patches.end() && it->first==next) {words.insert(words.end(),it->second.begin(),it->second.end());++it;++next;}
                    CK(cudaMemcpy(out+2*start*W,words.data(),words.size()*8,cudaMemcpyHostToDevice));
                    R.device_leaf.patch_words+=words.size();
                }
                if(device_gleaf_flag("NTT_DEVICE_GLEAF_TEST_BAD")){CK(cudaMemset(out,0xff,8));}
                if(device_gleaf_flag("NTT_DEVICE_GLEAF_CHECK")) {
                    std::vector<unsigned long long> x(count*W),z(count*W),actual(count*2*W),expected(2*W);
                    CK(cudaMemcpy(x.data(),resident.x+first*W,x.size()*8,cudaMemcpyDeviceToHost));
                    CK(cudaMemcpy(z.data(),resident.z+first*W,z.size()*8,cudaMemcpyDeviceToHost));
                    CK(cudaMemcpy(actual.data(),out,actual.size()*8,cudaMemcpyDeviceToHost));
                    for(size_t i=0;i<count;++i) {
                        const auto patch=patches.find(i);
                        if(patch!=patches.end())expected=patch->second;
                        else {words_neg_mod_n(expected,0,x.data()+i*W,C.hn.data(),W);std::copy(z.data()+i*W,z.data()+(i+1)*W,expected.begin()+W);}
                        if(!std::equal(expected.begin(),expected.end(),actual.begin()+2*i*W)) {
                            std::fprintf(stderr,"FATAL: device giant leaf CPU mismatch\n");std::exit(3);
                        }
                    }
                    R.device_leaf.checked_leaf_words+=actual.size();
                }
                ++R.device_leaf.device_trees;R.device_leaf.device_leaf_words+=count*2*W;
                R.device_leaf.avoided_leaf_h2d_bytes+=16ull*count*W;
                R.device_leaf.t_fill+=now_s()-t;
            };
            gt=build_groot_device(L,bleaf,bdeg,bpad,bs,BC_GTREE,hi-lo,fill,root_sink);
        } else gt=build_groot_select(L,bleaf,bdeg,bpad,bs,BC_GTREE,!g_s4_groot_only,root_sink);
        if (g_s4_groot_only) {
            const double tr0 = now_s();
            for (const auto &v : bleaf) g_groot.input_released_bytes += 8ull * v.capacity();
            std::vector<std::vector<unsigned long long>>().swap(bleaf);
            bs.t_release += now_s() - tr0;
        }
        R.t_gtrees += now_s() - tg0;
        ++g_groot.builds;
        g_groot.nodes_released += bs.nodes_released;
        g_groot.passthrough_moves += bs.passthrough_moves;
        g_groot.released_bytes += bs.node_released_bytes;
        g_groot.peak_node_bytes = std::max(g_groot.peak_node_bytes, bs.node_peak_bytes);
        g_groot.peak_retained_bytes = std::max(g_groot.peak_retained_bytes, bs.node_retained_bytes);
        g_groot.t_release += bs.t_release;
        g_groot.root_words += (bdeg[1]+1)*W;
        // FNV is complete only when every root is materialized. Resident consumers
        // fingerprint their actual inputs separately on the device.
        if(root_handed_off)g_groot.root_hash_complete=false;
        else for (const auto w : gt[1]) g_groot.root_hash = (g_groot.root_hash ^ w) * 1099511628211ull;
        gs.leaves += bs.leaves;
        gs.padded += bs.padded;
        gs.muls += bs.muls;
        if (b == 0) {
            if(!root_handed_off) {
                if(device_fold.active)device_fold.seed(gt[1]);
                else if(fold_flat_enabled)Hflat=std::move(gt[1]);else H=cp_from_flat(gt[1],bdeg[1],W);
            }
            continue;
        }
        /* ---- the fold: H <- (G*H) mod F, three full-size multiplies ---- */
        const double tf0 = now_s();
        if(root_handed_off)device_fold.step_loaded(device_fold.gcount);
        else if(device_fold.active)device_fold.step(gt[1]);
        else if(fold_flat_enabled) {
            fold_flat_step(L,gt[1],Hflat,Ft[1],finvflat,R.fold_flat);
        } else {
        const CPoly G = cp_from_flat(gt[1], bdeg[1], W);
        L.cat = BC_FOLD;
        const CPoly T = cp_mul(G, H, L);
        const size_t degT = (T.size() ? T.size() - 1 : 0);
        if (degT < P) { H = T; }                       /* T mod F = T when deg T < deg F */
        else {
            const size_t k = degT - P + 1;             /* <= P+1 */
            CPoly ra, rbi;
            cp_resize(ra, k, W);                       /* rev_k(T) */
            for (size_t i = 0; i < k; ++i) ra[i] = T[degT - i];
            cp_resize(rbi, k, W);                      /* the cached inverse, truncated to k */
            for (size_t i = 0; i < k; ++i) rbi[i] = finv[i];
            CPoly qrev = cp_mul(ra, rbi, L, k);
            cp_resize(qrev, k, W);
            CPoly q;
            cp_resize(q, k, W);
            for (size_t i = 0; i < k; ++i) q[i] = qrev[k - 1 - i];
            const CPoly qb = cp_mul(q, Fpoly, L, P);
            cp_resize(H, P, W);                        /* r = T - q*F has degree < P */
            for (size_t i = 0; i < P; ++i)
                cp_coeff_sub(H[i], T[i], (i < qb.size()) ? qb[i] : cp_zero(W), L.N, W);
            cp_trim(H);
        }
        }
        L.cat = -1;
        R.t_fold += now_s() - tf0;
        /* the fold loop is minutes long at the real shape and used to print NOTHING between
           `exactness:` and the final summary, which is why a hung or crashed real-shape run
           could not be told from a slow one.  One line per batch: progress, and the two
           numbers that say whether the arena is doing its job (fuse_reuse must grow,
           arena_overflow must stay 0 -- a nonzero overflow means per-call cudaMalloc is back
           in the hot path, which is exactly the 82% of section 18.2). */
        if (R.dbg_progress) {
            const double done = R.t_giant + R.t_gtrees + R.t_fold;
            const size_t arena_mb = (L.arena ? L.arena->mb() : 0);
            stage2_log::print(stage2_log::batches, "batched_progress: batch=%llu/%llu (%.1f%%) t=%.1f s left=%.1f s "
                        "giant=%.1f gtrees=%.1f fold=%.1f | ntt_calls=%llu launches=%llu "
                        "arena_mb=%llu fuse_builds=%llu fuse_reuse=%llu overflow=%llu\n",
                        b + 1, R.num_poly_g, 100.0 * (double)(b + 1) / (double)R.num_poly_g,
                        done, ((double)R.num_poly_g / (double)(b + 1) - 1.0) * done,
                        R.t_giant, R.t_gtrees, R.t_fold, L.ntt_calls, L.ntt_launches,
                        (unsigned long long)arena_mb,
                        L.arena ? L.arena->fuse_builds : 0ull,
                        L.arena ? L.arena->fuse_hits : 0ull,
                        L.arena ? L.arena->overflow : 0ull);
        }
        }
        R.t_loop_wall += now_s() - tloop0;
    }
    R.gscale.requested=true;
    const bool gscale_check=gscale_flag("NTT_GSCALE_DEVICE_CHECK"),gscale_bad=gscale_flag("NTT_GSCALE_DEVICE_TEST_BAD");
    if(gscale_bad && !gscale_check){std::fprintf(stderr,"FATAL: Gamma poison requires full GMP check\n");std::exit(3);}
    if(R.gscale.requested && device_fold.active && device_fold.hcount && mpz_cmp_ui(Ginv,1)!=0) {
        device_fold.scale(Ginv,R.gscale,gscale_check,gscale_bad);R.t_gscale=R.gscale.seconds;
    } else if(R.gscale.requested)R.gscale.fallback=!device_fold.active?"owner":!device_fold.hcount?"empty":"unit";
    if(device_fold.active)device_fold.finish(Hflat);
    if(fold_flat_enabled && !Hflat.empty()) {
        const double tb=now_s();H=cp_from_flat(Hflat,Hflat.size()/W-1,W);
        if(!finvflat.empty())finv=cp_from_flat(finvflat,finvflat.size()/W-1,W);
        std::vector<unsigned long long>().swap(Hflat);std::vector<unsigned long long>().swap(finvflat);
        R.fold_flat.t_bridge+=now_s()-tb;
    }
    /* ---- UNDO THE PROJECTIVE SCALE (section 42) ------------------------------------------
       The projective leaves multiplied the tree by Gamma, and the fold carries a constant
       straight through (H <- (G*H) mod F scales by the same constant), so H left the loop as
       Gamma * prod_b f_b mod F.  ONE pass over H's coefficients by Gamma^-1 restores the exact
       polynomial the old monic leaves produced -- not approximately: every step of the tree is
       reduced mod N coefficient by coefficient, so the scaling is exact in that ring, and
       gcd(v, N) is unchanged by an invertible factor either way. */
    if (!R.gscale.enabled && mpz_cmp_ui(Ginv, 1) != 0 && !H.empty()) {
        const double tg0 = now_s();
        mpz_t c;
        mpz_init(c);
        for (size_t i = 0; i < H.size(); ++i) {
            words_to_mpz(c, H[i].data(), W);
            mpz_mul(c, c, Ginv);
            mpz_mod(c, c, L.N);
            mpz_to_words(H[i], W, c);
        }
        mpz_clear(c);
        R.t_gscale = now_s() - tg0;
    }
    /* THE INVARIANT (section 42): one accumulated segment product per projective leaf set, no
       more and no less.  The factor set cannot see a wrong Gamma (any INVERTIBLE one gives the
       same gcds), so this is asserted instead of trusted. */
    if (proj_points != proj_gamma_points) {
        std::fprintf(stderr, "%s: FATAL: the projective scale covers %llu points but %llu leaves "
                             "were written projectively -- Gamma is not the product of exactly "
                             "those z values\n", NTT_PROBE_NAME,
                     (unsigned long long)proj_gamma_points, (unsigned long long)proj_points);
        std::exit(3);
    }
    {
        mpz_t t;
        mpz_init(t);
        if (mpz_invert(t, Ginv, L.N) == 0) {
            std::fprintf(stderr, "%s: FATAL: the projective scale Gamma is not invertible mod N "
                                 "-- H cannot be unscaled\n", NTT_PROBE_NAME);
            std::exit(3);
        }
        mpz_clear(t);
    }
    R.proj_points = proj_points;
    R.proj_gamma_points = proj_gamma_points;
    R.proj_segments = proj_segments;
    R.proj_fallbacks = proj_fallbacks;
    mpz_clear(Ginv);
    R.t_post_loop = now_s();

    /* ---- 3. ONE descent of H against the F tree: H(x_j) at every baby point ------------ */
    const double td0 = now_s();
    stage2_log::print(stage2_log::phases, "descent_begin: P=%llu levels=%d last_state=(%s)\n", (unsigned long long)P,
                (int)ceil_log2_u64((unsigned long long)Fpad), g_last_state);
    s2g_state("the batched descent");            /* a driver kill here leaves this behind */
    ws.need_vals((size_t)P);
    std::vector<std::vector<unsigned long long>> values;
    bool dev_leaves = false;
    {
            L.cat=BC_DESCENT;
            ScaledStats st;
            descent_scaled(L,Ft,Fdeg,Fpad,H,&finv,values,st,BC_DESCENT,g_scaled_check);
            L.cat=-1;
            stage2_log::print(stage2_log::phases, "scaled_descent: enabled=1 levels=%llu mul_calls=%llu mul_pairs=%llu copies=%llu zeros=%llu states=%llu words=%llu leaves=%llu checked_states=%llu checked_words=%llu frontier_peak_bytes=%llu pack_peak_bytes=%llu root_inverse_reused=%llu root_divisions=%llu\n",
                st.levels,st.mul_calls,st.mul_pairs,st.copies,st.zeros,st.states,st.words,st.leaves,st.checked_states,st.checked_words,st.frontier_peak_bytes,st.pack_peak_bytes,st.root_inverse_reused,st.root_divisions);
        }
    /* NTT_S4_DESCENT_CHECK=1 runs BOTH descents on the same H and compares every leaf value:
       the batched descent is a rearrangement of the same divisions, so any difference is a bug
       in the rearrangement.  With S5 on, the device descent's leaf values are the ones compared
       (read back once, because the check itself is the point). */
    {
        const char *envc = std::getenv("NTT_S4_DESCENT_CHECK");
        if (L.s4 && envc && *envc && std::atoi(envc) != 0) {
            if (dev_leaves) {
                values.assign((size_t)P, std::vector<unsigned long long>(W, 0ull));
                std::vector<unsigned long long> flat((size_t)P * W, 0ull);
                CK(cudaMemcpy(flat.data(), ws.dvals, flat.size() * 8, cudaMemcpyDeviceToHost));
                for (size_t i = 0; i < (size_t)P; ++i)
                    std::copy(flat.begin() + (long)(i * W), flat.begin() + (long)((i + 1) * W),
                              values[i].begin());
                R.s5_readback = true;
            }
            std::vector<std::vector<unsigned long long>> ref((size_t)P,
                std::vector<unsigned long long>(W, 0ull));
            unsigned long long dm = 0;
            L.cat = BC_DESCENT;
            descent_slow(L, Ft, Fdeg, Fpad, H, ref, dm);
            L.cat = -1;
            unsigned long long bad = 0, first = 0;
            unsigned long long badleaves = 0, firsts[8] = {0}, nfirst = 0;
            for (size_t i = 0; i < (size_t)P; ++i) {
                bool dl = false;
                for (size_t q = 0; q < W; ++q)
                    if (values[i][q] != ref[i][q]) {
                        if (!bad) {
                            first = (unsigned long long)i;
                            stage2_log::print(stage2_log::debug, "descent_check_bad: i=%llu word=%llu batched=%llu "
                                        "slow=%llu\n", (unsigned long long)i,
                                        (unsigned long long)q, values[i][q], ref[i][q]);
                        }
                        ++bad;
                        dl = true;
                    }
                if (dl) { ++badleaves; if (nfirst < 8) firsts[nfirst++] = (unsigned long long)i; }
            }
            stage2_log::print(stage2_log::debug, "descent_check_leaves: P=%llu differing_leaves=%llu first8=", (unsigned long long)P,
                        badleaves);
            for (unsigned long long z = 0; z < nfirst; ++z)
                stage2_log::print(stage2_log::debug, "%s%llu", z ? "," : "", firsts[z]);
            stage2_log::print(stage2_log::debug, "\n");
            /* ---- WHAT IS THE DEVICE'S VALUE, EXACTLY? (section 44) -------------------------
               A DOMAIN mistake is a constant factor, and a constant factor is NAMEABLE: compute
               ref/device mod N and ask whether it is R^k for a small k (R = 2^(64*nw), the
               Montgomery radix).  That turns "the numbers disagree" into "the device returns the
               plain value times R^-2", which is a diagnosis instead of a search. */
            if (bad) {
                mpz_t a, b, ai, ratio, R, pw, Rinv;
                mpz_inits(a, b, ai, ratio, R, pw, Rinv, nullptr);
                words_to_mpz(a, values[first].data(), W);
                words_to_mpz(b, ref[first].data(), W);
                mpz_set_ui(R, 1);
                for (size_t i = 0; i < W * 64; ++i) { mpz_mul_2exp(R, R, 1); mpz_mod(R, R, L.N); }
                char *sb = mpz_get_str(nullptr, 16, b), *sa = mpz_get_str(nullptr, 16, a);
                stage2_log::print(stage2_log::debug, "descent_check_values: i=%llu slow=%s batched=%s\n",
                            first, sb, sa);
                free(sb); free(sa);
                if (mpz_invert(ai, a, L.N) != 0) {
                    mpz_mul(ratio, b, ai); mpz_mod(ratio, ratio, L.N);
                    int kbest = 1000;
                    mpz_set_ui(pw, 1);
                    for (int k = 0; k <= 8 && kbest == 1000; ++k) {
                        if (mpz_cmp(pw, ratio) == 0) kbest = k;
                        mpz_mul(pw, pw, R); mpz_mod(pw, pw, L.N);
                    }
                    if (mpz_invert(Rinv, R, L.N) != 0) {
                        mpz_set_ui(pw, 1);
                        for (int k = 1; k <= 8 && kbest == 1000; ++k) {
                            mpz_mul(pw, pw, Rinv); mpz_mod(pw, pw, L.N);
                            if (mpz_cmp(pw, ratio) == 0) kbest = -k;
                        }
                    }
                    char *sr = mpz_get_str(nullptr, 16, ratio);
                    stage2_log::print(stage2_log::debug, "descent_check_domain: slow/batched = %s = R^%d\n", sr, kbest);
                    free(sr);
                } else {
                    stage2_log::print(stage2_log::debug, "descent_check_domain: the device's leaf value is NOT invertible "
                                "mod N\n");
                }
                mpz_clears(a, b, ai, ratio, R, pw, Rinv, nullptr);
            }
            stage2_log::print(stage2_log::debug, "descent_check: P=%llu divmods_batched=%llu divmods_slow=%llu "
                        "mismatching_coefficients=%llu first=%llu\n", (unsigned long long)P,
                        R.descent_divmods, dm, bad, first);
        }
    }
    R.leaf_values = P;
    /* A diagnostic S5 readback is also a complete plain-domain leaf vector. */
    if(!dev_leaves || R.s5_readback) {
        unsigned long long hash=1469598103934665603ull,words=0;
        for(const auto &v:values) for(auto word:v) {hash=(hash^word)*1099511628211ull;++words;}
        stage2_log::print(stage2_log::debug, "descent_values: leaves=%llu words=%llu hash=%llu\n",(unsigned long long)values.size(),words,hash);
    }
    R.t_descent = now_s() - td0;
    /* a phase marker, because the descent is where a long shape can look hung: everything after
       it used to print nothing until the final summary line */
    if (R.dbg_progress)
        stage2_log::print(stage2_log::phases, "batched_phase: descent_done t=%.1f s divmods=%llu\n", R.t_descent,
                    R.descent_divmods);

    /* ---- 4. accumulate prod_j H(x_j) mod N ON THE DEVICE, one gcd per block ------------ */
    const double ta0 = now_s();
    const size_t BLOCK = 64;                            /* small, so a hit localises to <=64 */
    std::vector<std::vector<unsigned long long>> bprod;
    ws.need_vals((size_t)P);
    {
        if (!dev_leaves) {                  /* the S5 path already left its leaves in ws.dvals */
            std::vector<unsigned long long> flat((size_t)P * W, 0ull);
            for (size_t i = 0; i < P; ++i)
                std::copy(values[i].begin(), values[i].end(), flat.begin() + (long)(i * W));
            CK(cudaMemcpy(ws.dvals, flat.data(), flat.size() * 8, cudaMemcpyHostToDevice));
        }
        dev_block_products(ws, (int)P, (int)BLOCK, bprod);
    }
    if (R.dbg_progress)
        stage2_log::print(stage2_log::phases, "batched_phase: block_products_done blocks=%llu t=%.1f s\n",
                    (unsigned long long)bprod.size(), now_s() - ta0);
    R.apply_blocks = bprod.size();
    R.block_per = BLOCK;
    /* ---- THE NAMING BUDGET ---------------------------------------------------------------------
     * Naming a hit is DIAGNOSTICS: it says which stage-2 prime produced the factor.  It must not
     * be on the critical path at scale, but the reporting semantics the gates assert must keep
     * working, so the DEFAULT depends on the shape rather than on a flag:
     *   * candidate arithmetic is 2*imax is_prime_u64 calls per hit leaf, and 2*imax = 2*(B2/D+2).
     *     The shapes the gates use have B2 <= 1e8 and D >= 210, and their hit count is small, so
     *     the budget below (2^19 primality tests, ~1 s at the exact test's measured speed) covers
     *     them: the frozen vector still prints hit_primes=114713 and rung 2/3 still print 3511,
     *     which is what `check_stage2_tree_gpu.ps1` compares against the CPU reference.
     *   * MEASURED at rung 3 (S=5261, D=510510, B2=1e8): 351 hit blocks and 22490 individual hits.
     *     Even with the exact primality test and with the scan stopping at the first confirmed
     *     prime, naming all of them costs ~195 s of a 233 s run (a 5261-bit GMP ladder point + gcd
     *     per hit is the irreducible part), while the same run with naming off takes 35.7 s and
     *     produces the same factor set.  So the budget is a TIME bound on a diagnostics phase, and
     *     a shape that exceeds it gets a full factor list and a `stage2_naming:` line that says how
     *     many hits were counted but not named.  NTT_NAME_HITS=1 forces full naming for a shape
     *     whose hit primes are actually wanted.
     * NTT_NAME_HITS=1 forces naming everywhere (for a deliberate deep run); NTT_NAME_HITS=0 forces
     * it off everywhere; NTT_NAME_BUDGET_BLOCKS=n overrides the block budget. */
    /* ---- THE CANDIDATE SET: THE TWO ENGINES ARE TRANSPOSED, AND THAT MATTERS ------------------
     * The recovered S5 patch (section 28) carried a `leaf_candidates()` lambda that attributed a
     * hit at "leaf L" to the primes p = off -+ j with off = (L+1)*D over the BABY VALUES j, i.e.
     * exactly the CPU reference's rule (stage2_tree_ref.cpp's name_culprit).  That rule is right
     * for the reference because ITS leaf values are F evaluated at the GIANT points, and it is
     * WRONG here: this engine's `values`/ws.dvals hold H evaluated at the BABY points (H is the
     * folded giant product, reduced mod F, and the descent walks F's tree, whose leaves are the
     * baby factors (x - x_j)).  The evidence for the baby reading is direct, not a reading of the
     * code: at rung 2 (D=2310, B2=1e7) this engine reports hit_leaves=240 = P, i.e. EVERY leaf
     * value shares a factor with N, which is exactly what the baby reading predicts -- the only
     * factor q=42089 has residual order m=3511 after stage 1, so
     * "D*i -+ j = 0 (mod 3511)" is solvable for every baby value j (i ranges over 4331 > 3511
     * values), while only one of those 240 leaves has a PRIME witness, p = 2310 + 1201 = 3511.
     * Under the giant reading those 240 hits would have to come from 240 distinct giant leaves,
     * and the count would not be P.  (Both readings name p=3511 for the leaf that confirms, which
     * is why the transposed rule looked like a fix: it was validated against a run whose modulus
     * was not the one the oracle used -- section 25.)
     * The lambda is therefore NOT used; the sound rule for THIS engine, used below, is: a hit
     * leaf j (a baby point) is witnessed by the primes p = i*D -+ j over the whole giant range,
     * because q | H(x_j) => q | (x_j - x_i) for some i => (i*D -+ j)*Q = O (mod q). */
    bool name_hits = true;
    unsigned long long name_budget_blocks = 0;
    {
        const char *eh = std::getenv("NTT_NAME_HITS");
        const char *eb = std::getenv("NTT_NAME_BUDGET_BLOCKS");
        if (eh && *eh) name_hits = (std::atoi(eh) != 0);
        unsigned long long tests_per_block = 0;
        if (imax < (~0ull) / (2ull * (unsigned long long)BLOCK)) {
            tests_per_block = 2ull * (unsigned long long)BLOCK * imax;
        } else {
            tests_per_block = ~0ull;                 /* overflow: treat as unbounded */
        }
        const unsigned long long budget = 1ull << 22;      /* ~4.2e6 primality tests */
        name_budget_blocks = (tests_per_block == 0) ? ~0ull : (budget / tests_per_block);
        if (name_budget_blocks == 0) name_budget_blocks = 1;
        if (eb && *eb) name_budget_blocks = std::strtoull(eb, nullptr, 10);
        stage2_log::print(stage2_log::debug, "naming_policy: name_hits=%d budget_blocks=%llu (2*imax=%llu primality tests "
                    "per leaf, %llu per block of %llu)\n", name_hits ? 1 : 0, name_budget_blocks,
                    2ull * imax, tests_per_block, (unsigned long long)BLOCK);
    }
    mpz_t bv, bg;
    mpz_inits(bv, bg, nullptr);
    std::vector<std::string> bstr((size_t)bprod.size());
    for (size_t b = 0; b < bprod.size(); ++b) {
        words_to_mpz(bv, bprod[b].data(), W);
        mpz_gcd(bg, bv, L.N);
        if (mpz_cmp_ui(bg, 1) > 0 && mpz_cmp(bg, L.N) < 0) {
            ++R.hit_blocks;                             /* a real hit: only NOW do we look at
                                                           the individual leaf values */
            if (dev_leaves && values.empty()) {
                /* the S5 leaves were never on the host: read them back ONCE, here, because a
                   hit means they are about to be inspected one by one anyway */
                values.assign((size_t)P, std::vector<unsigned long long>(W, 0ull));
                std::vector<unsigned long long> flat((size_t)P * W, 0ull);
                CK(cudaMemcpy(flat.data(), ws.dvals, flat.size() * 8, cudaMemcpyDeviceToHost));
                for (size_t i = 0; i < (size_t)P; ++i)
                    std::copy(flat.begin() + (long)(i * W), flat.begin() + (long)((i + 1) * W),
                              values[i].begin());
                R.s5_readback = true;
            }
            const size_t lo = b * BLOCK;
            const size_t hi = std::min(lo + BLOCK, (size_t)P);
            if (R.dbg_progress)
                stage2_log::print(stage2_log::phases, "batched_phase: naming_begin block=%llu leaves=[%llu,%llu) t=%.1f s\n",
                            (unsigned long long)b, (unsigned long long)lo, (unsigned long long)hi,
                            now_s() - ta0);
            /* THE WORK BUDGET OF THE NAMING PHASE (see the comment above `name_hits`).
               Candidate arithmetic is exactly 2*imax is_prime_u64 calls per LEAF, and this loop is
               entered once per hit block, so the phase costs
                   blocks * BLOCK * 2 * imax
               primality tests whatever the shape is.  When that product exceeds the budget the
               phase counts the hits (so hits/bad_factors stay exactly comparable with the CPU
               reference) and says so, instead of spending hours naming primes nobody will read:
               measured at B2=4e10/D=570570, one 64-leaf block of the old loop took 762.8 s and
               there are 810 such blocks, i.e. ~2.7 h of the ~2.75 h run. */
            const bool name_here = name_hits && (R.hit_blocks <= name_budget_blocks);
            if (R.dbg_progress && !name_here && R.hit_blocks == name_budget_blocks + 1) {
                stage2_log::print(stage2_log::phases, "batched_phase: naming_budget_exhausted at block=%llu hit_blocks=%llu "
                            "-- hits are still counted, hit_primes are no longer named\n",
                            (unsigned long long)b, R.hit_blocks);
            }
            unsigned long long blk_leaves = 0, blk_cands = 0;   /* this block's naming work */
            unsigned long long blk_leafhits = 0, blk_rec = 0;   /* per-leaf hits, and records */
            for (size_t j = lo; j < hi; ++j) {
                mpz_t v, lg;
                mpz_init(v);
                mpz_init(lg);
                words_to_mpz(v, values[j].data(), W);
                /* `lg` (this leaf's own gcd) is kept SEPARATE from `pg` (a later candidate's
                   gcd), because the fallback below needs the leaf's value after the scan */
                mpz_gcd(lg, v, L.N);
                if (mpz_cmp_ui(lg, 1) > 0 && mpz_cmp(lg, L.N) < 0) {
                    /* the baby point j = SP.baby_j[j] is the culprit's BABY half:
                       p | H(x_j) => p | (x_j - x_i) for some giant i, so search i */
                    ++R.hit_leaves;
                    ++blk_leafhits;
                    bool named = false;
                    const double tn0 = now_s();
                    const double ts0 = tn0;
                    std::vector<unsigned long long> cand;
                    /* TWO gates, both DIAGNOSTIC-only: NTT_NAME_MAX caps the naming per RUN
                       (the leaf indices are stable, so the same first leaves are named every
                       time) and `name_here` keeps the block budget of the naming phase.
                       Neither can change the factor set -- see the fallback below. */
                    const long long nmax = name_max();
                    if (name_here && (nmax == 0 || R.hit_leaves <= (unsigned long long)nmax)) {
                        const unsigned long long jj = SP.baby_j[j];
                        for (unsigned long long i = 1; i <= imax; ++i) {
                            const unsigned long long off = i * D;
                            for (int sgn = 0; sgn < 2; ++sgn) {
                                if (sgn == 0 && off < jj) continue;
                                const unsigned long long p = (sgn == 0) ? (off - jj) : (off + jj);
                                if (p <= B1 || p > B2) continue;
                                if (!is_prime_u64(p)) continue;
                                cand.push_back(p);
                            }
                        }
                        ++R.named_searches;
                        ++blk_leaves;
                        blk_cands += cand.size();
                        R.candidates_tested += cand.size();
                    }
                    R.t_scan += now_s() - ts0;
                    if (!cand.empty()) {
                        const double tl0 = now_s();
                        std::vector<unsigned long long> cx, cz;
                        ladder_points_ws(ws, cand, cx, cz);
                        for (size_t k = 0; k < cand.size(); ++k) {
                            mpz_t z;
                            mpz_init(z);
                            words_to_mpz(z, &cz[k * W], W);
                            mpz_gcd(pg, z, L.N);
                            if (mpz_cmp_ui(pg, 1) > 0 && mpz_cmp(pg, L.N) < 0) {
                                ++blk_rec;
                                s3_record(R.tail, pg, cand[k], L.N);
                                named = true;
                            }
                            mpz_clear(z);
                        }
                        R.t_ladder += now_s() - tl0;
                    }
                    if (!named) {
                        /* THE REFERENCE'S OWN CONVENTION (stage2_tree_ref.cpp name_culprit): a
                           hit whose stage-2 prime cannot be identified is STILL a real factor,
                           so record the leaf gcd itself with prime = 0 instead of dropping it.
                           Without this the reported factor SET would depend on the naming scan:
                           a hit leaf whose candidates happen not to confirm used to contribute
                           NOTHING, which is a silently lost factor.  `unnamed` counts these and
                           `hits` is left as the number of ATTRIBUTED hits, so `hits`/`hit_primes`
                           stay comparable with the CPU reference. */
                        s3_record(R.tail, lg, 0, L.N, /*count_hit=*/false);
                        ++R.unnamed;
                    }
                    R.t_name += now_s() - tn0;
                }
                mpz_clear(lg);
                mpz_clear(v);
            }
            if (R.dbg_progress)
                stage2_log::print(stage2_log::phases, "batched_phase: naming_end block=%llu hit_leaves=%llu candidates=%llu "
                            "leaf_hits=%llu records=%llu hit_blocks=%llu t_name=%.1f s t=%.1f s\n",
                            (unsigned long long)b, blk_leaves, blk_cands, blk_leafhits, blk_rec,
                            (unsigned long long)R.hit_blocks, R.t_name, now_s() - ta0);
        }
    }
    if (R.dbg_progress)
        stage2_log::print(stage2_log::debug, "tail_counts: hits=%llu unnamed=%llu factors=%llu hit_primes=%llu "
                    "hit_blocks=%llu named_searches=%llu candidates=%llu cand_lists=%llu\n",
                    (unsigned long long)R.tail.hits, (unsigned long long)R.tail.unnamed_hits,
                    (unsigned long long)R.tail.factors.size(),
                    (unsigned long long)R.tail.hit_primes.size(),
                    (unsigned long long)R.hit_blocks, (unsigned long long)R.named_searches,
                    (unsigned long long)R.candidates_tested, (unsigned long long)R.cand_lists);
    mpz_clears(bv, bg, nullptr);
    R.t_accum = now_s() - ta0;
    /* WHERE the naming time goes: with the old GMP primality test one hit leaf cost ~1.3 s of
       pure `is_prime_u64` calls, which is 98.7% of a rung-2 run (measured, section 26). */
    stage2_log::print(stage2_log::debug, "batched_naming: hit_blocks=%llu hit_leaves=%llu named_searches=%llu "
                "candidates_tested=%llu unnamed=%llu t_scan=%.3f t_ladder=%.3f t_name=%.3f "
                "name_max=%lld\n",
                R.hit_blocks, R.hit_leaves, R.named_searches, R.candidates_tested, R.unnamed,
                R.t_scan, R.t_ladder, R.t_name, name_max());
    mpz_clears(g, pg, nullptr);
    /* close the books (section 41): pre + loop_wall + post must equal the caller's `elapsed` */
    R.t_post_loop = now_s() - R.t_post_loop;
    /* the device-side counters: how much of the orchestration actually happened once */
    const auto &dl=R.device_leaf;
    stage2_log::print(stage2_log::debug, "device_gleaf: requested_chunks=%llu chunks=%llu fallback_chunks=%llu groups=%llu bad_groups=%llu good_segments=%llu device_trees=%llu device_leaf_words=%llu patch_words=%llu group_d2h_bytes=%llu bad_segment_d2h_bytes=%llu bad_point_d2h_bytes=%llu avoided_point_d2h_bytes=%llu avoided_segment_d2h_bytes=%llu avoided_leaf_h2d_bytes=%llu coord_peak_bytes=%llu group_peak_bytes=%llu checked_groups=%llu checked_leaf_words=%llu t_prepare=%.6f t_invert=%.6f t_fill=%.6f\n",
        dl.requested_chunks,dl.chunks,dl.fallback_chunks,dl.groups,dl.bad_groups,dl.good_segments,dl.device_trees,
        dl.device_leaf_words,dl.patch_words,dl.group_d2h_bytes,dl.bad_segment_d2h_bytes,dl.bad_point_d2h_bytes,
        dl.avoided_point_d2h_bytes,dl.avoided_segment_d2h_bytes,dl.avoided_leaf_h2d_bytes-dl.patch_words*8,
        dl.coord_peak_bytes,dl.group_peak_bytes,dl.checked_groups,dl.checked_leaf_words,dl.t_prepare,dl.t_invert,dl.t_fill);
    R.ladder_calls = ws.ladder_calls;
    R.ladder_points = ws.ladder_points_total;
    R.prod_launches = ws.prod_launches;
    if (L.arena) {
        R.arena_fuse_builds = L.arena->fuse_builds;
        R.arena_fuse_reuse = L.arena->fuse_hits;
        R.arena_buf_builds = L.arena->buf_builds;
        R.arena_buf_reuse = L.arena->buf_hits;
        R.arena_overflow = L.arena->overflow;
        R.arena_mb = L.arena->mb();
    }
    return R;
}

/* ===================================================================================== *
 *  SLICE S4 (B) -- THE REAL SHAPE (section 17's P2), WITH AN EXPLICIT MEMORY BUDGET
 *
 *  Section 10.6: the total work of the tree structure is proportional to 1/D while the
 *  transform memory is proportional to P = phi(D)/2, so the right D is the LARGEST one whose
 *  largest transform still fits.  This code decides that from the ACTUAL free memory and the
 *  ACTUAL shape plan (ntt_shape_query runs the multiply's own choose_cfg), prints the decision,
 *  and then runs the engine at that shape.
 * ===================================================================================== */

static unsigned long long gcd_u64(unsigned long long a, unsigned long long b)
{
    while (b) { const unsigned long long t = a % b; a = b; b = t; }
    return a;
}

static unsigned long long phi_u64(unsigned long long n){
    unsigned long long r = n, m = n;
    for (unsigned long long p = 2; p * p <= m; ++p) {
        if (m % p) continue;
        while (m % p == 0) m /= p;
        r -= r / p;
    }
    if (m > 1) r -= r / m;
    return r;
}

/* the arena footprint of one transform length: the shared big buffers (3*N words) plus dOut.
   THE TABLE CACHES ARE NOT COUNTED (section 20): ntt_fuse_cache_tables stores ~N + N/2 words per
   cached shape, but they are a PURE CACHE and NttArena::drop_table_caches evicts the ones belonging
   to other shapes before the arena refuses an allocation, so what a shape must be able to hold is
   the big buffers and the small ones.  Counting the caches in made the fold's shape look like 4.5N
   and that is what rejected every D above ~600000 in the section 19 scan -- i.e. it was a
   feasibility test that no longer matched the allocator. */
static unsigned long long real_shape_words(unsigned long long P, int S, bool *ok)
{
    unsigned long long N = 0, sw = 0, ss = 0, os = 0;
    int bpw = 0;
    if (!ntt_shape_query(P, S, &N, &bpw, nullptr, &sw, &ss, &os)) { *ok = false; return 0; }
    *ok = true;
    return 3 * N + os;
}

/* the two transform lengths a D actually needs: the fold/inverse at (P+1) coefficients and the
   tree's own top nodes at (P/2+1) */
static bool real_run_geometry(unsigned long long p,int bits,ecm_stage2::Geometry &g)
{
    return ecm_stage2::geometry(p,bits,[](unsigned long long m,int s,
        unsigned long long *n,unsigned long long *out) {
        return ntt_shape_query(m,s,n,nullptr,nullptr,nullptr,nullptr,out);
    },g,kFoldOwnerReuse);
}
static bool real_run_words(unsigned long long P,int S,unsigned long long *out_words,
                           unsigned long long *n_fold,unsigned long long *n_tree)
{
    ecm_stage2::Geometry g;
    if(!real_run_geometry(P,S,g))return false;
    *out_words=g.arena_estimate_bytes/8;
    if(n_fold)*n_fold=g.fold_length;
    if(n_tree)*n_tree=g.tree_length;
    return true;
}

/* the D scan's candidate record lives with the scan itself (section 59); the old `DChoice`, which
   only carried a memory footprint, was replaced by a cost+coverage model */

#include "stage2/stage2_d_model.cuh"

static int run_real(const char *n_str, bool n_is_hex, unsigned long long sigma,
                    unsigned long long B1, unsigned long long B2, unsigned long long D_in,
                    bool choose_d, bool run_s2, int curves,
                    const char *saved_qx_hex = nullptr, Stage2Tail *saved_result = nullptr, bool d_plan_only = false,
                    ecm_stage2::Plan *plan_result = nullptr)
{
    PolyLayer L;
    L.device = g_device;
    if (n_is_hex) { if (mpz_set_str(L.N, n_str, 16) != 0) { std::fprintf(stderr, "%s: bad hex N\n", NTT_PROBE_NAME); return 2; } }
    else if (mpz_set_str(L.N, n_str, 10) != 0) { std::fprintf(stderr, "%s: bad decimal N\n", NTT_PROBE_NAME); return 2; }
    if (mpz_odd_p(L.N) == 0) { std::fprintf(stderr, "%s: N must be odd\n", NTT_PROBE_NAME); return 2; }
    L.S = (size_t)mpz_sizeinbase(L.N, 2);
    L.W = words_for_bits(L.S);
    L.cost.S = (long)L.S;
    const size_t nw = L.W;

    CK(cudaSetDevice(g_device));
    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, g_device));
    CK(cudaFree(0));
    size_t freeb = 0, totalb = 0;
    CK(cudaMemGetInfo(&freeb, &totalb));
    size_t reserve = (size_t)768 * 1024 * 1024;
    {
        const char *envr = std::getenv("NTT_ARENA_RESERVE_MB");
        if (envr && *envr) reserve = (size_t)std::strtoull(envr, nullptr, 10) << 20;
    }
    size_t cap = (freeb > reserve) ? (freeb - reserve) : (freeb / 2);
    // Planning must respect the user arena cap as well as live free VRAM.
    const auto user_cap=fuse_env_ull("NTT_ARENA_CAP_KB",0);
    if(user_cap) cap=std::min(cap,(size_t)user_cap*1024);
    stage2_log::print(stage2_log::phases, "stage2_real: device=%d (%s) N_bits=%ld sigma=%llu B1=%llu B2=%llu "
                "requested_D=%llu choose_d=%d\n", g_device, prop.name, (long)L.S, sigma, B1, B2,
                D_in, choose_d ? 1 : 0);

    /* ---- THE D SEARCH (section 59) ----------------------------------------------------------
       `--choose-d` used to take the LARGEST D whose transforms fit memory, on the grounds that
       "work is proportional to 1/D".  That is half the story and the wrong half to optimise:

         * the MAIN LOOP is proportional to 1/D -- imax = B2/D giant steps, and this engine ties
           the batch size to P exactly (`num_poly_g = ceil(imax/P)`, section 27/S3), so every
           multiply in the loop is a degree-P x degree-P multiply and the loop costs
           imax * log2(min(P, imax));
         * the F TREE, the Newton inverse, the remainder DESCENT and the block ACCUMULATION are all
           proportional to P = phi(D)/2, which GROWS with D.

       D balances giant steps against the baby/F tree, inverse and descent. The baby set
       is j<=D/2 with gcd(j,D)=1; its +/- residues enumerate the units. A prime p>D is
       automatically coprime to D. phi(D)/D is residue density, not prime-coverage probability.
       Different D still changes the lower-bound/overshoot geometry, so rank by measured
       cost without claiming identical prime sets. The calibrated resident model above
       handles actual NTT lengths; the legacy rates below remain the unsupported-scope fallback.

       THE MODEL IS FITTED TO MEASURED PHASE TIMES, NOT GUESSED.  The reference is the production
       run of section 56.1 (D=570570, P=51840, imax=3400110, giant_points=3400110):
           gtrees 192.920 + fold 72.836          = 265.756   loop,  per imax*log2(P)
           pre 6.335 + descent 31.134 + accum 13.068 = 50.537  tree, per P
           giant 50.655                                  per imax
           gleaves 20.722 + loop_host 9.617      = 30.339   per batch
           face value plus the ~25 s setup that lives outside `elapsed`
       Turned into RATES (so the model does not depend on the B2 being planned) and re-applied, it
       reproduces that run to 0.1%: 265.7 + 50.5 + 50.7 + 30.3 + 25.0 = 422.3 s against the measured
       397.83 + 25 = 422.8 s.  Its job is ranking, not prediction to the second.

       Memory feasibility stays a HARD filter: a candidate whose shapes do not fit the arena is not
       a candidate at all. */
    unsigned long long D = D_in, P_baby = 0;
    const char *env4 = std::getenv("NTT_S4_OFF");
    const bool s4_on = !(env4 && *env4 && std::atoi(env4) != 0);
    const bool model_requested=fuse_env_ull("NTT_D_MODEL",0)!=0;
    const auto outer_mode=fuse_env_ull("NTT_FUSE_COOP_OUTER",0);
    const bool shape_ntt=outer_mode==2 && d_shape_rates_valid && fuse_shape_policy_supported(12);
    const bool baby_requested=fuse_env_ull("NTT_BABY_DEVICE",0)!=0;
    const auto baby_budget=fuse_env_ull("NTT_BABY_DEVICE_MAX_MB",512)*1024*1024;
    const auto baby_live_cap=freeb>64ull*1024*1024 ? freeb-64ull*1024*1024 : 0ull;
    const auto baby_cap=std::min((unsigned long long)baby_budget,(unsigned long long)baby_live_cap);
    // Frozen phase rates were measured under the legacy cache admission ledger.
    // Payload accounting v2 needs a new calibration before those rates can be enabled.
    const bool cache_rates_valid=false;
    bool calibrated=model_requested && cache_rates_valid && s4_on && !s4_tail_mont_mode() && g_xadd6 && g_s4_mersenne && g_groot_device &&
        g_s4_output_window && g_s4_chunk_output && g_s4_groot_only && g_s4_pack_direct &&
        g_s4_oracle_async && g_s4_oracle_pack && g_s4_carry_batch && !g_s4_final_readback &&
        L.S==4423 && mpz_popcount(L.N)==4423 && curves==1 && B1==1000 &&
        B2>=100000000000ull && B2<=2011326186870ull &&
        std::strstr(prop.name,"RTX 4060 Laptop") &&
        fuse_env_ull("NTT_FUSE_T",12)==12 && fuse_env_ull("NTT_FUSE_M",4)==4 &&
        fuse_env_ull("NTT_FUSE_WARP_TAIL",0)!=0 && (outer_mode==0 || shape_ntt) &&
        fuse_env_ull("NTT_S4_BATCH_MB",64)==64 &&
        fuse_env_ull("NTT_GIANT_CHAIN_BLOCK",64)==64 && fuse_env_ull("NTT_GIANT_CHAIN_MIN",32768)==32768 &&
        fuse_env_ull("NTT_GIANT_CHAIN_SMALL_BLOCK",0)==0 && !g_giant_seed_pair &&
        fuse_env_ull("NTT_ARENA_WORKSPACE_POOL",1)!=0 &&
        fuse_env_ull("NTT_S4_SAMPLE",96)==96 && fuse_env_ull("NTT_S4_CHECK_EVERY",8)==8;
    for(const char *key:{"NTT_SMALL_PRIME_REUSE","NTT_GIANT_SEED_DEVICE","NTT_GFINV_SEG_EXACT",
                         "NTT_GFINV_BATCH","NTT_FOLD_FLAT","NTT_FOLD_DEVICE","NTT_DEVICE_GLEAF",
                         "NTT_GROOT_TO_FOLD","NTT_SCALED_DESCENT"})
        if(!fuse_env_ull(key,0))calibrated=false;
    if(fuse_env_ull("NTT_S4_HOSTPACK",0) || !fuse_compact_scratch() ||
       !fuse_env_ull("NTT_S4_FLAT_DIRECT",1))calibrated=false;
    // Empirical rates must match the selected arithmetic and NTT policy.
    const bool gl_short=(ntt_gl_reduce_requested_mode()&1u)!=0;
    const bool gl_shift_scale=fuse_env_ull("NTT_GL_SHIFT_SCALE",0)!=0;
    const bool gl_ptx=NTT_GL_FIXED_MODE==3 || fuse_env_ull("NTT_GL_PTX_REDUCE",0)!=0;
    const bool point_requested=fuse_env_ull("NTT_POINT_MERSENNE",0)!=0;
    // Only the measured immutable PTX/GPU baby backend has profile5 rates.
    const bool fixed_ptx=NTT_GL_FIXED_MODE==3 && d_fixed_ptx_rates_valid && gl_short && shape_ntt && baby_requested;
    const bool point_fold=point_requested && fixed_ptx && d_point_fold_rates_valid &&
        L.S==4423 && mpz_popcount(L.N)==4423;
    // Outer ILP changes NTT timings; frozen profiles require the original schedule.
    if(NTT_OUTER_UNROLL_U!=0)calibrated=false;
    if(ntt_carry_check_requested())calibrated=false;
    if(point_requested && !point_fold)calibrated=false;
    if(gl_shift_scale || (gl_ptx && !fixed_ptx) || (NTT_GL_FIXED_MODE>=0 && !fixed_ptx))calibrated=false;
    if(gl_short && !(shape_ntt && (fixed_ptx ? d_fixed_ptx_rates_valid : d_short_rates_valid)))calibrated=false;
    if(baby_requested && !(gl_short && shape_ntt && (fixed_ptx ? d_fixed_ptx_rates_valid : d_baby_rates_valid) && baby_cap))calibrated=false;
    if(baby_requested)for(const char *key:{"NTT_BABY_DEVICE_CHECK","NTT_BABY_DEVICE_TEST",
                                         "NTT_BABY_DEVICE_TEST_BAD","NTT_BABY_DEVICE_ALLOC_FAIL"})
        if(fuse_env_ull(key,0))calibrated=false;
    const auto fold_budget=fuse_env_ull("NTT_FOLD_DEVICE_MAX_MB",640)*1024*1024;
    auto owner_bytes=[&](unsigned long long p){return ecm_stage2::owner_bytes(p,nw,kFoldOwnerReuse);};
    if(D_in && (phi_u64(D_in)/2==0 || B2/D_in+2<=phi_u64(D_in)/2 ||
                owner_bytes(phi_u64(D_in)/2)>fold_budget ||
                (baby_requested && d_baby_payload_bytes(phi_u64(D_in)/2,nw)>baby_cap)))calibrated=false;
    if(model_requested && !cache_rates_valid)
        stage2_log::print(stage2_log::debug, "d_model_scope: calibrated=0 reason=cache_payload_v2_unmeasured\n");
    const double d_scan_begin=now_s();
    double selected_seconds=0;
    DPhaseModel phase_model((int)L.S,B2,point_fold ? 6 : fixed_ptx ? 5 : baby_requested ? 4 : gl_short && shape_ntt ? 3 : shape_ntt ? 2 : 0);
    const char *model_version=calibrated ? (point_fold ? "resident_point_fold_v1" : fixed_ptx ? "resident_fixed_ptx_v1" : baby_requested ? "resident_baby_v1" : gl_short ? "resident_short_v1" : shape_ntt ? "resident_shape_v1" : "resident_xadd6_v1") : "legacy_56_1";
    stage2_log::print(stage2_log::debug, "ntt_outer_schedule: unroll_u=%d (0=compiler-default, 4=experimental ILP)\n",NTT_OUTER_UNROLL_U);
    stage2_log::print(stage2_log::debug, "d_model: requested=%d enabled=%d version=%s arena_cap_bytes=%llu fold_budget_bytes=%llu gl_short=%d gl_shift_scale=%d gl_ptx=%d "
                "(calibrated scope: RTX4060 Laptop M4423 B1=1000 B2=1e11..2011326186870, batch64/chain64; estimates)\n",
                (int)model_requested,(int)calibrated,model_version,
                (unsigned long long)cap,fold_budget,(int)gl_short,(int)gl_shift_scale,(int)gl_ptx);
    {
        struct DCand {
            unsigned long long D = 0, P = 0, imax = 0, batches = 0;
            double units = 0.0, loop = 0.0, tree = 0.0, giant = 0.0, glue = 0.0, total = 0.0;
            double rate = 0.0;                       /* covered values per second (coverage ~ 1) */
            bool fits = false;
            unsigned long long total_words = 0;
        };
        const double RP = 51840.0, RIM = 3400110.0, RB = 66.0;
        const double RLOOP = 265.756, RTREE = 50.537, RGIANT = 50.655, RGLUE = 30.339, RSETUP = 25.0;
        auto lg2 = [](double x) { return (x > 1.0) ? (std::log(x) / std::log(2.0)) : 1.0; };
        auto model = [&](DCand &c) {
            c.imax = B2 / c.D + 2;
            const unsigned long long bsz = (c.P < c.imax) ? c.P : c.imax;
            c.batches = bsz ? (c.imax + bsz - 1) / bsz : 1;
            c.units = (double)c.P * 2.0 / (double)c.D;      /* phi(D)/D: informational only */
            c.loop = (RLOOP / (RIM * lg2(RP))) * (double)c.imax * lg2((double)bsz);
            c.tree = (RTREE / RP) * (double)c.P;
            c.giant = (RGIANT / RIM) * (double)c.imax;
            c.glue = (RGLUE / RB) * (double)c.batches;
            c.total = c.loop + c.tree + c.giant + c.glue + RSETUP;
            if(calibrated) {
                if(c.P==0 || c.imax<=c.P || owner_bytes(c.P)>fold_budget ||
                   (baby_requested && d_baby_payload_bytes(c.P,nw)>baby_cap)) {c.total=1e100;return;}
                const auto e=phase_model.cost(c.D,c.P);
                c.loop=e.gtrees+e.fold;c.tree=e.init+e.descent+e.inv+e.accum;
                c.giant=e.giant;c.glue=e.glue;c.total=e.total;
            }
            const double span = (B2 > B1) ? (double)(B2 - B1) : 1.0;
            c.rate = span / (c.total > 1e-9 ? c.total : 1e-9);
        };
        /* every 47-smooth D (products of the primes below with ANY exponents): the step size need
           not be squarefree, and a repeated factor multiplies D (fewer steps) without changing
           phi(D)/D, which is exactly the trade the model is about. */
        static const unsigned long long prim[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41,
                                                  43, 47};
        const size_t np = sizeof(prim) / sizeof(prim[0]);
        const unsigned long long dlim = 200000000ull;
        std::vector<DCand> cands;
        cands.reserve(60000);
        /* iterative enumeration of every 47-smooth D <= dlim: an explicit stack visits each
           candidate exactly once, needs no recursion and no <functional> */
        {
            std::vector<std::pair<size_t, unsigned long long>> stk;
            stk.push_back({0, 1ull});
            while (!stk.empty()) {
                const size_t i = stk.back().first;
                const unsigned long long cur = stk.back().second;
                stk.pop_back();
                if (i == np) {
                    DCand c;
                    c.D = cur;
                    c.P = phi_u64(cur) / 2;
                    if (c.P == 0) continue;
                    model(c);
                    cands.push_back(c);
                    continue;
                }
                unsigned long long v = cur;
                for (;;) {
                    stk.push_back({i + 1, v});
                    if (v > dlim / prim[i]) break;
                    v *= prim[i];
                }
            }
        }
        std::sort(cands.begin(), cands.end(),
                  [](const DCand &a, const DCand &b) { return a.total < b.total; });
        if(calibrated) stage2_log::print(stage2_log::debug, "d_scan: candidates=%llu (47-smooth D <= %llu) ; "
                                  "model=%s ranked by empirical full Stage2 cost; "
                                  "actual NTT lengths, partial trees/Newton, owner and arena filters\n",
                                  (unsigned long long)cands.size(),dlim,model_version);
        else stage2_log::print(stage2_log::debug, "d_scan: candidates=%llu (47-smooth D <= %llu) ; model fitted to the section "
                    "56.1 run (rates: loop=%.3f s per imax*log2, tree=%.1f us per leaf, "
                    "giant=%.2f us per step, glue=%.2f ms per batch) ; ranked by TOTAL COST "
                    "(coverage is ~1 for every candidate: a prime p > D is coprime to D, so its "
                    "residue is a unit and the baby set IS the units)\n",
                    (unsigned long long)cands.size(), dlim,
                    RLOOP / (RIM * lg2(RP)), RTREE * 1e6 / RP, RGIANT * 1e6 / RIM, RGLUE * 1e3 / RB);
        DCand best;
        unsigned long long checked = 0, shown = 0;
        for (DCand &c : cands) {
            if (shown >= 14 && best.D) break;
            if (checked++ > 4000 && best.D) break;
            unsigned long long tw = 0;
            if(calibrated && (owner_bytes(c.P)>fold_budget || c.total>=1e80))continue;
            if (!real_run_words(c.P, (int)L.S, &tw, nullptr, nullptr)) continue;
            c.total_words = tw;
            c.fits = ((double)tw * 8.0 <= (double)cap);
            if (!c.fits) continue;
            if (!best.D || c.total < best.total) best = c;
            if (shown < 14) {
                // i>=1 and j<=D/2: the first giant covers from about D/2.
                // This is a rough lower-bound diagnostic, not a coverage weight.
                auto pi_est=[](double x){return x>2 ? x/std::log(x) : x>=2 ? 1.0 : 0.0;};
                const double missed=std::max(0.0,pi_est((double)c.D/2)-pi_est((double)B1));
                stage2_log::print(stage2_log::debug, "d_scan_choice: D=%llu P=phi/2=%llu imax=%llu batches=%llu "
                            "units=%.4f below_first_giant~%.0f fit=%.0f MB | loop=%.1f tree=%.1f "
                            "giant=%.1f glue=%.1f total=%.1f rate=%.3g vals/s\n",
                            c.D, c.P, c.imax, c.batches, c.units, missed,
                            (double)tw * 8.0 / 1048576.0, c.loop, c.tree, c.giant, c.glue, c.total,
                            c.rate);
                if(calibrated)phase_model.print(c.D,c.P,owner_bytes(c.P));
                ++shown;
            }
        }
        /* the requested D, for comparison (even when it is not the winner) */
        if (D_in) {
            DCand r;
            r.D = D_in;
            r.P = phi_u64(D_in) / 2;
            model(r);
            unsigned long long tw = 0;
            const bool okm = real_run_words(r.P, (int)L.S, &tw, nullptr, nullptr) && tw*8<=cap;
            if(calibrated)phase_model.print(r.D,r.P,owner_bytes(r.P));
            stage2_log::print(stage2_log::debug, "d_scan_reference: D=%llu P=%llu imax=%llu batches=%llu units=%.4f "
                        "fit=%s | loop=%.1f tree=%.1f giant=%.1f glue=%.1f total=%.1f "
                        "rate=%.3g vals/s\n",
                        r.D, r.P, r.imax, r.batches, r.units, okm ? "yes" : "NO",
                        r.loop, r.tree, r.giant, r.glue, r.total, r.rate);
        }
        if (choose_d) {
            if (!best.D) {
                std::fprintf(stderr, "%s: no candidate D fits the budget (cap=%.0f MB)\n",
                             NTT_PROBE_NAME, cap / 1048576.0);
                return 3;
            }
            D = best.D;
            selected_seconds=best.total;
            stage2_log::print(stage2_log::debug, "d_scan_decision: D=%llu P=phi(D)/2=%llu (the CHEAPEST admissible shape by "
                        "the fitted cost model -- NOT the largest D that fits, which is what this "
                        "flag used to pick; see the header comment)\n", D, best.P);
        } else {
            P_baby = phi_u64(D) / 2;
            DCand explicit_plan; explicit_plan.D=D; explicit_plan.P=P_baby;
            model(explicit_plan); selected_seconds=explicit_plan.total;
            bool ok1 = false, ok2 = false;
            const unsigned long long w1 = real_shape_words(P_baby, (int)L.S, &ok1);
            const unsigned long long w2 = real_shape_words(P_baby / 2 + 1, (int)L.S, &ok2);
            stage2_log::print(stage2_log::debug, "d_budget: free=%.0f MB reserve=%.0f MB cap=%.0f MB ; D=%llu "
                        "P=phi(D)/2=%llu largest_transform=%.0f MB total_needed=%.0f MB "
                        "fits=%d%s\n", freeb / 1048576.0, reserve / 1048576.0,
                        cap / 1048576.0, D, P_baby,
                        (double)w1 * 8.0 / 1048576.0, (double)(w1 + 2 * w2) * 8.0 / 1048576.0,
                        ((double)(w1 + (ok2 ? 2 * w2 : 0)) * 8.0 <= (double)cap) ? 1 : 0,
                        ok1 ? "" : " (SHAPE REFUSED BY THE MULTIPLY)");
        }
    }
    P_baby = phi_u64(D) / 2;
    stage2_log::print(stage2_log::debug, "d_scan_wall: seconds=%.6f selected_D=%llu (excluded from stage2_full_wall)\n",
                now_s()-d_scan_begin,D);
    if(plan_result) {
        ecm_stage2::Plan &p=*plan_result;
        if(!real_run_geometry(P_baby,(int)L.S,p.geometry))return 3;
        p.d=D; p.b1=B1; p.b2=B2; p.giant_points=B2/D+2;
        p.batches=p.giant_points/P_baby+(p.giant_points%P_baby!=0);
        p.free_bytes=freeb; p.arena_cap_bytes=cap; p.owner_budget_bytes=fold_budget;
        p.baby_bytes=d_baby_payload_bytes(P_baby,nw);
        p.owner_budget_fits=p.geometry.fold_owner_bytes<=fold_budget;
        p.arena_estimate_fits=p.geometry.arena_estimate_bytes<=cap;
        p.estimated_seconds=selected_seconds; p.calibrated=calibrated; p.model=model_version;
    }
    if(d_plan_only) {
        stage2_log::print(stage2_log::debug, "d_plan_only: D=%llu P=%llu curves_executed=0 model=%s\n",
                    D,phi_u64(D)/2,model_version);
        return 0;
    }

    /* the baby set: j COPRIME TO D, j <= D/2, ascending (the CPU reference's own rule).  Not
       "coprime to N": the first version of this filtered by gcd(N,j) and produced 1155 points
       for D=2310 instead of 240. */
    const double stage2_shape_begin=now_s();
    std::vector<unsigned long long> baby_j;
    for (unsigned long long j = 1; j <= D / 2; ++j)
        if (gcd_u64(j, D) == 1) baby_j.push_back(j);
    if (baby_j.size() != P_baby) {
        std::fprintf(stderr, "%s: FATAL: baby set has %llu points but phi(D)/2 = %llu\n",
                     NTT_PROBE_NAME, (unsigned long long)baby_j.size(), P_baby);
        return 3;
    }
    const double stage2_shape_seconds=now_s()-stage2_shape_begin;
    const unsigned long long imax = B2 / D + 2;
    /* B1 and B2 are printed RAW, not as a derived count: --b2 used to be read with strtoull
       base 10, so "1e11" silently became 1 and the run quietly computed a B2=1 shape
       (giant_points=2) while every other line looked normal -- caught only because
       giant_points was printed at all (section 27).  Both are parsed as floating point now. */
    stage2_log::print(stage2_log::phases, "real_shape: D=%llu P=phi(D)/2=%llu baby_points=%llu giant_points=%llu "
                "num_poly_g=%llu loops=%llu B1=%llu B2=%llu S_bits=%ld\n", D, P_baby,
                (unsigned long long)baby_j.size(), imax, (P_baby + imax - 1) / P_baby,
                (imax + P_baby - 1) / P_baby - 1, B1, B2, (long)L.S);

    /* ---- Montgomery parameters (R, ninv, a24, Q) ---- */
    mpz_t R, tmp, a24, ax, az;
    mpz_inits(R, tmp, a24, ax, az, nullptr);
    mpz_set_ui(tmp, 1);
    mpz_mul_2exp(tmp, tmp, (unsigned long)(64 * nw));
    mpz_mod(R, tmp, L.N);
    unsigned long long ninv = 0;
    {
        mpz_t nmod, inv, pow2;
        mpz_inits(nmod, inv, pow2, nullptr);
        mpz_set_ui(pow2, 1);
        mpz_mul_2exp(pow2, pow2, 64);
        mpz_mod(nmod, L.N, pow2);
        if (mpz_invert(inv, nmod, pow2) == 0) {
            std::fprintf(stderr, "%s: N is not invertible mod 2^64\n", NTT_PROBE_NAME);
            return 2;
        }
        mpz_sub(inv, pow2, inv);
        unsigned long long buf[2] = {0, 0};
        size_t cnt = 0;
        mpz_export(buf, &cnt, -1, 8, 0, 0, inv);
        ninv = buf[0];
        mpz_clears(nmod, inv, pow2, nullptr);
    }
    std::vector<unsigned long long> hn(nw), ha24(nw), hmone(nw);
    mpz_to_words(hn, nw, L.N);
    mpz_to_words(hmone, nw, R);
    if (suyama_curve(a24, ax, az, sigma, L.N) != 0) {
        std::fprintf(stderr, "%s: degenerate sigma for this N\n", NTT_PROBE_NAME);
        return 2;
    }
    mpz_mul(tmp, a24, R);
    mpz_mod(tmp, tmp, L.N);
    mpz_to_words(ha24, nw, tmp);
    if (mont_selftest(hn, nw, ninv, L.N, R) != 0) return 1;
    if (xadd_selftest(hn,nw,ninv,L.N,R) != 0) return 1;

    /* A Stage1 text save contains ordinary affine X. Do not regenerate the
       exponent or apply the optional Stage1 torsion multiplier on resume. */
    std::vector<unsigned long long> qx(nw), qz(nw);
    if (saved_qx_hex) {
        if (mpz_set_str(tmp, saved_qx_hex, 16) != 0 || mpz_sgn(tmp) < 0 ||
            mpz_cmp(tmp, L.N) >= 0) {
            std::fprintf(stderr, "%s: invalid saved affine X\n", NTT_PROBE_NAME);
            return 2;
        }
        mpz_to_words(qx, nw, tmp);
        qz[0] = 1;
        stage2_log::print(stage2_log::phases, "stage1_save_resume: sigma=%llu B1=%llu normalized_Z=1 stage1_skipped=1\n",
                    sigma, B1);
    } else {
        std::fprintf(stderr,"production Stage2 requires a normalized Stage1 save\n");
        return 2;
    }
    {
        mpz_t X, Z;
        mpz_inits(X, Z, nullptr);
        words_to_mpz(X, qx.data(), nw);
        words_to_mpz(Z, qz.data(), nw);
        affine_x_gmp(tmp, X, Z, L.N);
        char *s = mpz_get_str(nullptr, 16, tmp);
        stage2_log::print(stage2_log::debug, "real_setup_Q: Q_x_hex=%.64s... (%zu hex digits)\n", s, std::strlen(s));
        const char *qfull=std::getenv("NTT_STAGE1_Q_DUMP");
        if(qfull && std::atoi(qfull)!=0) stage2_log::print(stage2_log::debug, "real_setup_Q_full: hex=%s\n",s);
        void (*ff)(void *, size_t) = nullptr;
        mp_get_memory_functions(nullptr, nullptr, &ff);
        ff(s, std::strlen(s) + 1);
        mpz_clears(X, Z, nullptr);
    }
    /* Stage1 Q is ready. Include context/reducer setup, baby/F-tree, then the entire tail.
       Baby index enumeration ran before Stage1; account its measured work once separately. */
    const double stage2_init_begin=now_s();
    LadderCtx C;
    C.hn = hn;
    C.nw = nw;
    C.ninv = ninv;
    C.ha24 = ha24;
    C.hmone = hmone;
    C.hqx.assign(nw, 0ull);
    C.hqz.assign(nw, 0ull);
    {
        mpz_t X, Z;
        mpz_inits(X, Z, nullptr);
        words_to_mpz(X, qx.data(), nw);
        mpz_mul(X, X, R);
        mpz_mod(X, X, L.N);
        mpz_to_words(C.hqx, nw, X);
        words_to_mpz(Z, qz.data(), nw);
        mpz_mul(Z, Z, R);
        mpz_mod(Z, Z, L.N);
        mpz_to_words(C.hqz, nw, Z);
        mpz_clears(X, Z, nullptr);
    }

    /* ---- the arena + the device reduction, with the budget printed ---- */
    NttArena arena;
    arena.device = g_device;
    arena.cap_bytes = cap;
    {
        const char *envh = std::getenv("NTT_ARENA_CAP_KB");
        if (envh && *envh) arena.cap_bytes = (size_t)std::strtoull(envh, nullptr, 10) << 10;
    }
    S4Ctx s4;
    S4Reduce red;
    {
        const char *es = std::getenv("NTT_S4_SAMPLE");
        if (es && *es) g_s4_sample_limit = std::atoll(es);
        const char *ee = std::getenv("NTT_S4_CHECK_EVERY");
        if (ee && *ee) g_s4_check_every = (unsigned long long)std::strtoull(ee, nullptr, 10);
    }
    L.arena = &arena;
    double stage2_fixture_begin=0.0;
    if (s4_on) {
        s4_reduce_init(red, L.N, nw, hn, ninv);
        s4.red = &red;
        L.s4 = &s4;
        stage2_fixture_begin=now_s();
        const char *flat_test = std::getenv("NTT_S4_FLAT_TEST");
        if (flat_test && std::atoi(flat_test) != 0) s4_flat_input_check(L);
        const char *readback_test = std::getenv("NTT_S4_FINAL_READBACK_TEST");
        if (readback_test && std::atoi(readback_test) != 0) s4_flat_input_check(L,5);
        const char *window_test=std::getenv("NTT_S4_OUTPUT_WINDOW_TEST");
        if(window_test && std::atoi(window_test)!=0) s4_output_window_check(L);
        const char *scaled_test=std::getenv("NTT_SCALED_TEST");
        if(scaled_test && std::atoi(scaled_test)!=0) scaled_descent_fixture(L);
    }
    if(!s4_on) stage2_fixture_begin=now_s();
    const char *fold_test=std::getenv("NTT_FOLD_FLAT_TEST");
    if(fold_test && std::atoi(fold_test))fold_flat_fixture(L);
    gscale_flag("NTT_GSCALE_DEVICE_CHECK");gscale_flag("NTT_GSCALE_DEVICE_TEST_BAD");
    const char *fold_device_test=std::getenv("NTT_FOLD_DEVICE_TEST");
    if(fold_device_test && std::atoi(fold_device_test))fold_device_fixture(L);
    if(fuse_env_ull("NTT_BABY_DEVICE_TEST",0))device_baby_fixture(nw);
    const char *seg_test=std::getenv("NTT_GFINV_SEG_TEST");
    if(seg_test && std::atoi(seg_test))segment_product_fixture(C);
    const char *ginv_test=std::getenv("NTT_GFINV_BATCH_TEST");
    if(ginv_test && std::atoi(ginv_test))gfinv_batch_fixture(L.N,nw);
    const char *gdevice_test=std::getenv("NTT_GROOT_DEVICE_TEST");
    if(gdevice_test && std::atoi(gdevice_test))groot_device_fixture(L);
    const char *groot_test = std::getenv("NTT_S4_GROOT_TEST");
    if (groot_test && std::atoi(groot_test) != 0) groot_lifetime_check(L);
    const char *workspace_test = std::getenv("NTT_ARENA_WORKSPACE_TEST");
    if (workspace_test && std::atoi(workspace_test) != 0) ntt_workspace_check(g_device);
    if(fuse_env_ull("NTT_FUSE_COOP_TEST",0))ntt_fuse_coop_check(g_device);
    const char *fuse_test = std::getenv("NTT_FUSE_LIFETIME_TEST");
    if (fuse_test && std::atoi(fuse_test) != 0) {
        ntt_fuse_lifetime_check(g_device);
        ntt_fuse_capacity_check(g_device);
    }
    const double stage2_fixture_seconds=now_s()-stage2_fixture_begin;
    const char *real_dump=std::getenv("NTT_REAL_F_DUMP");
    bool stage2_extra_fixtures=real_dump && *real_dump;
    for(const char *key : {"NTT_BABY_DEVICE_TEST","NTT_BABY_DEVICE_CHECK","NTT_BABY_DEVICE_TEST_BAD","NTT_BABY_DEVICE_ALLOC_FAIL","NTT_FUSE_COOP_TEST","NTT_FUSE_COOP_BAD","NTT_XADD6_TEST","NTT_XADD6_TEST_BAD","NTT_S4_FLAT_TEST","NTT_S4_FINAL_READBACK_TEST","NTT_S4_OUTPUT_WINDOW_TEST",
                           "NTT_S4_GROOT_TEST","NTT_ARENA_WORKSPACE_TEST","NTT_FUSE_LIFETIME_TEST",
                           "NTT_SCALED_TEST","NTT_SCALED_CHECK","NTT_GROOT_DEVICE_TEST","NTT_GROOT_DEVICE_CHECK","NTT_GROOT_DEVICE_TEST_BAD","NTT_GROOT_LEAF_CHUNK","NTT_GFINV_BATCH_TEST","NTT_GFINV_BATCH_TEST_BAD","NTT_FOLD_FLAT_TEST","NTT_FOLD_FLAT_TEST_BAD","NTT_GSCALE_DEVICE_CHECK","NTT_GSCALE_DEVICE_TEST_BAD","NTT_FOLD_DEVICE_TEST","NTT_FOLD_DEVICE_CHECK","NTT_FOLD_DEVICE_TEST_BAD","NTT_FOLD_DEVICE_ALLOC_FAIL","NTT_GROOT_TO_FOLD_CHECK","NTT_GROOT_TO_FOLD_TEST_BAD","NTT_GFINV_SEG_TEST","NTT_GFINV_SEG_TEST_BAD","NTT_GFINV_SEG_CHECK","NTT_GIANT_SEED_CHECK","NTT_S4_MERSENNE_TEST","NTT_S4_MERSENNE_TEST_BAD","NTT_SMALL_PRIME_CHECK","NTT_SMALL_PRIME_TEST_BAD","NTT_SMALL_PRIME_CACHE_STALE","NTT_DEVICE_GLEAF_CHECK","NTT_DEVICE_GLEAF_TEST_BAD"}) {
        const char *v=std::getenv(key);
        if(v && std::atoi(v)!=0) stage2_extra_fixtures=true;
    }
    {
        bool ok1 = false, ok2 = false;
        const double w1 = (double)real_shape_words(P_baby, (int)L.S, &ok1);
        const double w2 = (double)real_shape_words(P_baby / 2 + 1, (int)L.S, &ok2);
        stage2_log::print(stage2_log::phases, "mem_budget: free=%.0f MB total=%.0f MB reserve=%.0f MB arena_cap=%.0f MB ; "
                    "P=%llu => largest transform (the fold, operand %llu coeffs) = %.0f MB ; "
                    "tree top (operand %llu coeffs) = %.0f MB ; the arena holds one entry per "
                    "shape and a shape that does not fit falls back to per-call cudaMalloc "
                    "(same kernels and same result, slower) -- arena_overflow counts those\n",
                    freeb / 1048576.0, totalb / 1048576.0, reserve / 1048576.0,
                    arena.cap_bytes / 1048576.0, P_baby, P_baby + 1,
                    w1 * 8.0 / 1048576.0, P_baby / 2 + 1, w2 * 8.0 / 1048576.0);
    }

    /* ---- the baby points and F, exactly as the dump path does ---- */
    double t_f = 0.0;
    std::vector<std::vector<unsigned long long>> Ft;
    std::vector<size_t> Fdeg;
    size_t Fpad = 0, fdeg = 0;
    /* OBJECTIVE 3 (section 31.4): factors revealed by DEGENERATE BABY POINTS -- points whose Z
       shares a factor with N, i.e. points that are the identity modulo that factor.  Collected
       here (the batched run does not exist yet) and merged into the reported factor set below. */
    std::vector<std::string> baby_deg;
    SmallPrimeBabyCache small_cache;
    const bool reuse_small=small_prime_flag("NTT_SMALL_PRIME_REUSE");
    if(reuse_small)small_cache.begin(C,D,B1,B2,baby_j);
    {
        const double t0 = now_s();
        std::vector<unsigned long long> bx, bz;
        unsigned long long noninv = 0;              /* printed as real_baby: ... degenerate= */
        const double tb0 = now_s();
        std::vector<std::vector<unsigned long long>> leaf(baby_j.size());
        DeviceBabyStats bs;
        const bool baby_requested=fuse_env_ull("NTT_BABY_DEVICE",0)!=0;
        const bool baby_used=baby_requested && device_baby_generate(C,L.N,baby_j,leaf,
            reuse_small?&small_cache:nullptr,baby_deg,noninv,bs);
        if(!baby_used)ladder_points(C,baby_j,bx,bz);
        const double tb1=baby_used?tb0+bs.ladder_seconds:now_s();
        if(!baby_used) {
            mpz_t X, Z, xj, neg, gq;
            mpz_inits(X, Z, xj, neg, gq, nullptr);
            std::vector<unsigned long long> w(nw, 0ull);
            /* ---- ONE INVERSION PER SEGMENT, NOT PER POINT (section 25) --------------------------
               x_j = X_j/Z_j mod N used to cost one mpz_invert per BABY point: 115200 of them at
               ~61.5 us each is ~7 s, and the whole pre-`elapsed` setup -- this ladder plus its
               conversion and real_setup's 12.1 s -- is ~44 s of the 317 s wall clock, the second
               largest improvable item after the G tree.  Section 42 already fixed exactly this for
               the GIANT points (224.8 s -> 20.3 s) with Montgomery's trick: invert the segment's
               PRODUCT once, then walk back.  Same code shape here, same fallback: if any Z in the
               segment is not invertible (gcd(Z,N) > 1 -- the degeneracy that IS a factor, objective
               3) the whole segment drops to the per-point path, which records the gcd. */
            const size_t BS = 256;
            /* plain C arrays, NOT std::vector<mpz_t>: mpz_t is an ARRAY type, so a vector of it
               cannot be constructed (measured: "a new-initializer may not be specified for an
               array" from MSVC's xmemory) -- the same trap this project recorded once before */
            mpz_t pv[BS + 1];                            /* prefix products */
            mpz_t zv[BS];
            for (size_t j = 0; j <= BS; ++j) mpz_init(pv[j]);
            for (size_t j = 0; j < BS; ++j) mpz_init(zv[j]);
            mpz_t iprod, tmul;
            mpz_inits(iprod, tmul, nullptr);
            const size_t nb = baby_j.size();
            for (size_t lo = 0; lo < nb; lo += BS) {
                const size_t hi = ((lo + BS) < nb) ? (lo + BS) : nb;
                const size_t seg = hi - lo;
                bool clean = true;
                mpz_set_ui(pv[0], 1);
                for (size_t j = 0; j < seg; ++j) {
                    words_to_mpz(zv[j], &bz[(lo + j) * nw], nw);
                    if (mpz_sgn(zv[j]) == 0) { clean = false; break; }
                    mpz_mul(pv[j + 1], pv[j], zv[j]);
                    mpz_mod(pv[j + 1], pv[j + 1], L.N);
                }
                if (clean && mpz_invert(iprod, pv[seg], L.N) == 0) clean = false;
                if (!clean) {
                    /* the affine path, VERBATIM: whatever refuses to invert is a hit, not an
                       accident, and affine_x_gmp_checked records its gcd as a factor */
                    for (size_t j = 0; j < seg; ++j) {
                        words_to_mpz(X, &bx[(lo + j) * nw], nw);
                        words_to_mpz(Z, &bz[(lo + j) * nw], nw);
                        /* The established affine helper accepts Z=0 as leaf x=0. Its GCD
                           proof is N, not 1; retain saturation without changing leaf semantics. */
                        if(reuse_small && mpz_sgn(Z)==0)small_cache.record(lo+j,L.N);
                        if (!affine_x_gmp_checked(xj, X, Z, L.N)) {
                            ++noninv;
                            mpz_gcd(gq, Z, L.N);
                            if(reuse_small)small_cache.record(lo+j,gq);
                            char *gs = mpz_get_str(nullptr, 10, gq);
                            baby_deg.push_back(gs);
                            void (*ff)(void *, size_t) = nullptr;
                            mp_get_memory_functions(nullptr, nullptr, &ff);
                            ff(gs, std::strlen(gs) + 1);
                        }
                        mpz_neg(neg, xj);
                        mpz_mod(neg, neg, L.N);
                        mpz_to_words(w, nw, neg);
                        leaf[lo + j].assign(2 * nw, 0ull);
                        std::copy(w.begin(), w.end(), leaf[lo + j].begin());
                        leaf[lo + j][nw] = 1;
                    }
                    continue;
                }
                /* At step j, iprod = 1/(z_0..z_j). Multiply by pv[j] = z_0..z_{j-1}
                   to get 1/z_j, then update iprod for the shorter prefix. */
                for (size_t j = seg; j-- > 0;) {
                    words_to_mpz(X, &bx[(lo + j) * nw], nw);
                    mpz_mul(tmul, iprod, pv[j]);
                    mpz_mod(tmul, tmul, L.N);
                    mpz_mul(tmul, X, tmul);
                    mpz_mod(xj, tmul, L.N);
                    mpz_mul(iprod, iprod, zv[j]);
                    mpz_mod(iprod, iprod, L.N);
                    mpz_neg(neg, xj);
                    mpz_mod(neg, neg, L.N);
                    mpz_to_words(w, nw, neg);
                    leaf[lo + j].assign(2 * nw, 0ull);
                    std::copy(w.begin(), w.end(), leaf[lo + j].begin());
                    leaf[lo + j][nw] = 1;
                }
            }
            for (size_t j = 0; j <= BS; ++j) mpz_clear(pv[j]);
            for (size_t j = 0; j < BS; ++j) mpz_clear(zv[j]);
            mpz_clears(iprod, tmul, nullptr);
            if (noninv)
                stage2_log::print(stage2_log::debug, "baby_degenerate: points=%llu of %llu have Z sharing a factor with N "
                            "-> their gcd was recorded as a factor\n", noninv,
                            (unsigned long long)baby_j.size());
            mpz_clears(X, Z, xj, neg, gq, nullptr);
        }
        stage2_log::print(stage2_log::debug, "baby_device: requested=%d enabled=%d points=%llu groups=%llu bad_groups=%llu "
                    "payload_bytes=%llu root_d2h_bytes=%llu leaf_d2h_bytes=%llu coordinate_d2h_bytes=%llu "
                    "seed_h2d_bytes=%llu checked_words=%llu invert=%.6f total=%.6f\n",
                    (int)baby_requested,(int)baby_used,(unsigned long long)bs.points,(unsigned long long)bs.groups,
                    (unsigned long long)bs.bad_groups,(unsigned long long)bs.payload_bytes,
                    (unsigned long long)bs.root_d2h_bytes,(unsigned long long)bs.leaf_d2h_bytes,
                    (unsigned long long)bs.coordinate_d2h_bytes,(unsigned long long)bs.seed_h2d_bytes,
                    (unsigned long long)bs.checked_words,bs.invert_seconds,bs.total_seconds);
        stage2_log::print(stage2_log::debug, "ladder: baby_points=%llu (device x_j; there is no CPU reference at this "
                    "shape, see the report)\n", (unsigned long long)baby_j.size());
        /* the setup pieces that live OUTSIDE `elapsed` had no timer at all (section 25): they are
           ~44 s of the 317 s wall clock, so naming them is the first step */
        stage2_log::print(stage2_log::phases, "real_baby: points=%llu ladder=%.3f s affine=%.3f s degenerate=%llu\n",
                    (unsigned long long)baby_j.size(), tb1 - tb0, now_s() - tb1, noninv);
        FTreeStats fs;
        Ft = build_tree_flat(L, leaf, Fdeg, Fpad, fs, BC_FTREE);
        s4_oracle_drain(red);    /* include final F-tree validation in its phase timer */
        fdeg = Fdeg[1];
        if(real_dump && *real_dump) {
            if(baby_j.size()>4096) {std::fprintf(stderr,"%s: FATAL: real F dump limited to 4096 babies\n",NTT_PROBE_NAME);std::exit(3);}
            std::vector<std::string> bx_hex,F_hex;
            mpz_t x,z,q;mpz_inits(x,z,q,nullptr);
            auto as_hex=[&](const mpz_t v) {
                char *s=mpz_get_str(nullptr,16,v);std::string result=s;
                void (*release)(void*,size_t)=nullptr;mp_get_memory_functions(nullptr,nullptr,&release);
                release(s,result.size()+1);return result;
            };
            for(const auto &v:leaf) {
                words_to_mpz(x,v.data(),nw);mpz_neg(x,x);mpz_mod(x,x,L.N);bx_hex.push_back(as_hex(x));
            }
            for(size_t i=0;i<=fdeg;++i) {words_to_mpz(x,Ft[1].data()+i*nw,nw);F_hex.push_back(as_hex(x));}
            words_to_mpz(x,qx.data(),nw);words_to_mpz(z,qz.data(),nw);affine_x_gmp(q,x,z,L.N);
            if(!write_f_dump(real_dump,L.N,a24,q,D,B1,B2,sigma,baby_j,bx_hex,F_hex,fdeg)) {
                std::fprintf(stderr,"%s: FATAL: cannot write real F dump\n",NTT_PROBE_NAME);std::exit(3);
            }
            mpz_clears(x,z,q,nullptr);
            stage2_log::print(stage2_log::debug, "real_F_dump: babies=%llu coefficients=%llu path=%s\n",
                (unsigned long long)bx_hex.size(),(unsigned long long)F_hex.size(),real_dump);
        }
        stage2_log::print(stage2_log::phases, "ftree_real: leaves=%llu padded=%llu muls=%llu ntt_calls=%llu "
                    "ntt_seconds=%.3f\n", (unsigned long long)fs.leaves,
                    (unsigned long long)fs.padded, (unsigned long long)fs.muls, L.ntt_calls,
                    L.ntt_seconds);
        t_f = now_s() - t0;
    }
    stage2_log::print(stage2_log::debug, "exactness: binding_shape P=%llu S=%llu slot_bits=%llu slot_words=%llu "
                "L=P*slot_words=%llu bpw=%d ; L*(2^bpw-1)^2 = 2^%.3f < p = 2^%.3f -> %s\n",
                L.bind_P, (unsigned long long)L.S, L.bind_slot_bits, L.bind_slot_words,
                L.bind_L, L.bind_bpw, L.bind_bound_bits, mpz_log2d_p(),
                exact_ok_terms(L.bind_L, L.bind_bpw) ? "OK" : "VIOLATED");

    if(reuse_small) {
        small_cache.ready=std::is_sorted(small_cache.js.begin(),small_cache.js.end()) &&
            std::adjacent_find(small_cache.js.begin(),small_cache.js.end())==small_cache.js.end();
        if(small_prime_flag("NTT_SMALL_PRIME_CACHE_STALE"))++small_cache.D;
    }
    const double stage2_init_seconds=stage2_shape_seconds+now_s()-stage2_init_begin;
    /* ---- the tails ---- */
    Stage2Params SP;
    SP.D = D; SP.B1 = B1; SP.B2 = B2; SP.baby_j = baby_j;
    for (int cv = 0; cv < curves; ++cv) {
        g_gfinv={};g_giant_seed={};

        const unsigned long long nb = L.ntt_calls, nl = L.ntt_launches;
        const double ns = L.ntt_seconds;
        const double fw0 = L.t_fwd, iv0 = L.t_inv, sl0 = L.t_slot, st0 = L.t_setup, pk0 = L.t_pack,
                     mc0 = L.t_maxc, h20 = L.t_h2d, d20 = L.t_d2h, ex0 = L.t_ext, xc0 = L.t_xchk;
        const double dc0 = L.t_d2h_coeff;
        const unsigned long long dw0 = L.d2h_coeff_words;
        const double hp0 = L.t_hpack, sc0 = L.t_scan, hb0 = L.t_h2d_batch, ck0 = L.t_check;
        const double cr0 = L.t_check_reset, ck0b = L.t_check_kernel, cd0 = L.t_check_d2h;
        const double pl0b = L.t_plan, oc0 = L.t_opcopy, ho0 = L.t_hout;
        const double ry0 = L.s4 ? L.s4->t_h2d_raw : 0.0;
        const unsigned long long rw0 = L.s4 ? L.s4->raw_words : 0ull;
        const unsigned long long pl0 = L.s4 ? L.s4->pack_launches : 0ull;
        const double t0 = now_s();
        BatchedRun BR = run_batched(L, C, SP, Ft, Fdeg, Fpad,reuse_small?&small_cache:nullptr);
        s4_oracle_drain(red);    /* validation must finish BEFORE elapsed and success */
        /* merge the factors that degenerate BABY points revealed (objective 3): each already
           divides N by construction, and they are deduplicated against what the naming stage
           found, so the reported fact set cannot change for a shape where the naming stage
           already reaches them. */
        for (const std::string &s : baby_deg)
            if (std::find(BR.tail.factors.begin(), BR.tail.factors.end(), s) ==
                BR.tail.factors.end())
                BR.tail.factors.push_back(s);
        if (saved_result) *saved_result = BR.tail;
        const double el = now_s() - t0;
        std::string fs2, ps2;
        for (size_t i = 0; i < BR.tail.factors.size(); ++i) { if (i) fs2 += ","; fs2 += BR.tail.factors[i]; }
        for (size_t i = 0; i < BR.tail.hit_primes.size(); ++i) { if (i) ps2 += ","; ps2 += std::to_string(BR.tail.hit_primes[i]); }
        stage2_log::print(stage2_log::curve, "stage2: algorithm=tree_gpu_batched curves=1 hits=%llu bad_factors=%llu "
                    "factors=%s hit_primes=%s elapsed=%.2f\n", BR.tail.hits, BR.tail.bad_factors,
                    fs2.c_str(), ps2.c_str(), el);
        stage2_log::print(stage2_log::curve, "stage2_full_wall: curve=%d shape=%.6f init=%.6f main=%.6f total=%.6f "
                    "fixture=%.6f clean=%d (init includes baby/F-tree and mandatory selftests; "
                    "main includes GCD, naming and oracle drain; Stage1 excluded; init shared)\n",
                    cv+1,stage2_shape_seconds,stage2_init_seconds,el,stage2_init_seconds+el,
                    stage2_fixture_seconds,(int)(!stage2_extra_fixtures && !run_s2 && curves==1));
        stage2_log::print(stage2_log::debug, "real_batched_foldflat: enabled=%d folds=%llu muls=%llu sub_coeffs=%llu peak_bytes=%llu prepare=%.6f multiply=%.6f subtract=%.6f bridge=%.6f\n",
            (int)BR.fold_flat.enabled,BR.fold_flat.folds,BR.fold_flat.muls,BR.fold_flat.sub_coeffs,BR.fold_flat.peak_bytes,
            BR.fold_flat.t_prepare,BR.fold_flat.t_multiply,BR.fold_flat.t_subtract,BR.fold_flat.t_bridge);
        const auto &fd=BR.fold_device;
        stage2_log::print(stage2_log::phases, "real_batched_folddevice: requested=%d enabled=%d fallback=%s folds=%llu muls=%llu sub_coeffs=%llu peak_bytes=%llu h2d_bytes=%llu d2h_bytes=%llu avoided_h2d_bytes=%llu avoided_d2h_bytes=%llu checked_words=%llu setup=%.6f upload=%.6f reverse=%.6f subtract=%.6f readback=%.6f reuse=%u layout_bytes=%llu saved_bytes=%llu\n",
            (int)fd.requested,(int)fd.enabled,fd.fallback.c_str(),fd.folds,fd.muls,fd.sub_coeffs,fd.peak_bytes,
            fd.h2d_bytes,fd.d2h_bytes,fd.avoided_h2d_bytes,fd.avoided_d2h_bytes,fd.checked_words,
            fd.t_setup,fd.t_upload,fd.t_reverse,fd.t_subtract,fd.t_readback,fd.reuse,fd.layout_bytes,fd.saved_bytes);
        stage2_log::print(stage2_log::phases, "real_batched_rootfold: requested=%d trees=%llu words=%llu avoided_h2d_bytes=%llu avoided_d2h_bytes=%llu checked_words=%llu digest_words=%llu digest_sum=%016llx digest_xor=%016llx digest_kind=mixsum_xor_v1 t_handoff=%.6f\n",
            (int)fd.root_requested,fd.root_device_trees,fd.root_device_words,fd.root_h2d_avoided,fd.root_d2h_avoided,
            fd.root_checked_words,fd.root_digest_words,fd.root_digest[0],fd.root_digest[1],fd.root_t_handoff);
        stage2_log::print(stage2_log::debug, "real_batched_gfinv: enabled=%d requests=%llu cache_hits=%llu groups=%llu segments=%llu group_attempts=%llu group_failures=%llu individual_attempts=%llu good=%llu nonunits=%llu scratch_peak_bytes=%llu t_prepare=%.6f\n",
            (int)g_gfinv_batch,g_gfinv.requests,g_gfinv.cache_hits,g_gfinv.groups,g_gfinv.segments,
            g_gfinv.group_attempts,g_gfinv.group_failures,g_gfinv.individual_attempts,g_gfinv.good,
            g_gfinv.nonunits,g_gfinv.scratch_peak_bytes,g_gfinv.t_prepare);
        if (BR.tail.unnamed_hits)
            stage2_log::print(stage2_log::debug, "stage2_naming: named_hits=%llu unnamed_hits=%llu (the naming budget "
                        "stopped the culprit scan; hits and bad_factors are complete, the "
                        "hit_primes list is not -- NTT_NAME_HITS=1 forces naming)\n",
                        BR.tail.hits - BR.tail.unnamed_hits, BR.tail.unnamed_hits);
        stage2_log::print(stage2_log::debug, "real_batched_shape: P=%llu giant_points=%llu num_poly_g=%llu loops=%llu "
                    "descent_divmods=%llu\n", BR.P, BR.giant_points, BR.num_poly_g, BR.loops,
                    BR.descent_divmods);
        stage2_log::print(stage2_log::debug, "real_batched_gdevice: enabled=%d trees=%llu fallbacks=%llu levels=%llu groups=%llu pairs=%llu copies=%llu leaf_words=%llu root_words=%llu resident_words=%llu trace_words=%llu metadata_words=%llu metadata_peak_bytes=%llu raw_peak_bytes=%llu logical_frontier_peak_bytes=%llu host_staging_peak_bytes=%llu checked_nodes=%llu checked_words=%llu\n",
            (int)g_groot_device,g_gdevice.trees,g_gdevice.fallbacks,g_gdevice.levels,g_gdevice.groups,g_gdevice.pairs,g_gdevice.copies,
            g_gdevice.leaf_words,g_gdevice.root_words,g_gdevice.resident_words,g_gdevice.trace_words,g_gdevice.metadata_words,
            g_gdevice.metadata_peak_bytes,g_gdevice.raw_peak_bytes,g_gdevice.logical_frontier_peak_bytes,
            g_gdevice.host_staging_peak_bytes,g_gdevice.checked_nodes,g_gdevice.checked_words);
        stage2_log::print(stage2_log::debug, "real_batched_gmemory: compact_raw=%d leaf_staging=%d pinned_trees=%llu pinned_slices=%llu pinned_words=%llu pageable_words=%llu leaf_fallbacks=%llu legacy_trees=%llu pinned_borrow_peak_bytes=%llu rawA_peak_bytes=%llu rawB_peak_bytes=%llu\n",
            (int)g_groot_compact_raw,(int)g_groot_leaf_staging,g_gmemory.pinned_trees,g_gmemory.pinned_slices,
            g_gmemory.pinned_words,g_gmemory.pageable_words,g_gmemory.leaf_fallbacks,g_gmemory.legacy_trees,
            g_gmemory.pinned_borrow_peak_bytes,g_gmemory.rawA_peak_bytes,g_gmemory.rawB_peak_bytes);
        stage2_log::print(stage2_log::debug, "real_batched_groot: root_only=%d builds=%llu nodes_released=%llu moves=%llu "
                    "node_peak_bytes=%llu retained_peak_bytes=%llu released_bytes=%llu input_released_bytes=%llu "
                    "root_words=%llu trace=%d root_hash=%016llx t_release=%.6f root_hash_complete=%d (node capacities, not process peak)\n",
                    (int)g_s4_groot_only, g_groot.builds, g_groot.nodes_released, g_groot.passthrough_moves,
                    g_groot.peak_node_bytes, g_groot.peak_retained_bytes, g_groot.released_bytes,
                    g_groot.input_released_bytes, g_groot.root_words, (int)g_s4_carry_trace,
                    g_groot.root_hash, g_groot.t_release,(int)g_groot.root_hash_complete);
        stage2_log::print(stage2_log::debug, "real_batched_cost: poly_muls=%llu operand_bits=%llu f_tree=%llu g_tree=%llu "
                    "fold=%llu descent=%llu inv=%llu total=%llu\n", L.cost.tot_muls(),
                    L.cost.tot_bits(), L.cost.bits[BC_FTREE], L.cost.bits[BC_GTREE],
                    L.cost.bits[BC_FOLD], L.cost.bits[BC_DESCENT], L.cost.bits[BC_FINV],
                    L.cost.tot_bits());
        stage2_log::print(stage2_log::phases, "real_batched_split: giant=%.3f gtrees=%.3f fold=%.3f descent=%.3f inv=%.3f "
                    "accum=%.3f name=%.3f f_tree_incl=%.3f\n", BR.t_giant, BR.t_gtrees, BR.t_fold,
                    BR.t_descent, BR.t_inv, BR.t_accum, BR.t_name, t_f);
        /* THE BOOKS, CLOSED (section 41): pre + loop_wall + post == `el` above, and the only
           part of the loop body that no phase timer owned is the host affine conversion of the
           giant points (`gleaves`).  loop_host = loop_wall - giant - gtrees - fold - gleaves is
           then everything else the host does inside the loop with the device idle. */
        stage2_log::print(stage2_log::debug, "real_batched_wall: pre=%.3f loop_wall=%.3f post=%.3f sum=%.3f (elapsed=%.2f) "
                    "| gleaves=%.3f loop_host=%.3f\n", BR.t_pre_loop, BR.t_loop_wall, BR.t_post_loop,
                    BR.t_pre_loop + BR.t_loop_wall + BR.t_post_loop, el, BR.t_gleaves,
                    BR.t_loop_wall - BR.t_giant - BR.t_gtrees - BR.t_fold - BR.t_gleaves);
        stage2_log::print(stage2_log::debug, "real_batched_gleaves: in=%.3f invert=%.3f out=%.3f (us_per_point: in=%.2f "
                    "invert=%.2f out=%.2f) gscale=%.3f s\n", BR.t_gin, BR.t_ginv, BR.t_gout,
                    BR.giant_points ? 1e6 * BR.t_gin / (double)BR.giant_points : 0.0,
                    BR.giant_points ? 1e6 * BR.t_ginv / (double)BR.giant_points : 0.0,
                    BR.giant_points ? 1e6 * BR.t_gout / (double)BR.giant_points : 0.0,
                    BR.t_gscale);
        stage2_log::print(stage2_log::phases,"real_gscale_device: requested=%d enabled=%d coefficients=%llu h2d_bytes=%llu check_d2h_bytes=%llu checked_words=%llu seconds=%.6f fallback=%s\n",
            (int)BR.gscale.requested,(int)BR.gscale.enabled,BR.gscale.coefficients,BR.gscale.h2d_bytes,
            BR.gscale.check_d2h_bytes,BR.gscale.checked_words,BR.gscale.seconds,BR.gscale.fallback);
        /* section 42: the projective leaves and the segment products that cover them must match
           EXACTLY (asserted in run_batched); the number is reported so a drift is visible */
        stage2_log::print(stage2_log::debug, "real_batched_projective: leaves=%llu gamma_points=%llu segments=%llu "
                    "affine_fallback_points=%llu\n", BR.proj_points, BR.proj_gamma_points,
                    BR.proj_segments, BR.proj_fallbacks);
        stage2_log::print(stage2_log::phases, "real_giant_seed: enabled=%d exact_segments=%d chunks=%llu points=%llu avoided_d2h_bytes=%llu avoided_h2d_bytes=%llu avoided_cpu_modmuls=%llu avoided_montmuls=%llu checked_words=%llu segments=%llu segment_checks=%llu segment_fix_muls=%llu fix_table_peak_bytes=%llu\n",
                    (int)g_giant_seed_device,(int)g_gfinv_seg_exact,g_giant_seed.chunks,g_giant_seed.points,
                    g_giant_seed.avoided_d2h_bytes,g_giant_seed.avoided_h2d_bytes,g_giant_seed.avoided_cpu_modmuls,
                    g_giant_seed.avoided_montmuls,g_giant_seed.checked_words,g_giant_seed.segments,
                    g_giant_seed.segment_checks,g_giant_seed.segment_fix_muls,g_giant_seed.fix_table_peak_bytes);
        stage2_log::print(stage2_log::debug, "real_giant_seed_pair: requested=%d base_builds=%llu base_nonunits=%llu chunks=%llu paired_ladders=%llu scalar_h2d_avoided=%llu base_d2h_bytes=%llu\n",
                    (int)g_giant_seed_pair,g_giant_seed.base_builds,g_giant_seed.base_nonunits,
                    g_giant_seed.paired_chunks,g_giant_seed.paired_ladders,
                    g_giant_seed.scalar_h2d_avoided,g_giant_seed.base_d2h_bytes);
        stage2_log::print(stage2_log::phases, "real_giant_base: cpu_requested=%d cpu_builds=%llu gpu_builds=%llu h2d_bytes=%llu checked_words=%llu cpu_seconds=%.6f build_seconds=%.6f\n",
                    (int)g_giant_base_cpu,g_giant_seed.base_cpu_builds,g_giant_seed.base_gpu_builds,
                    g_giant_seed.base_h2d_bytes,g_giant_seed.base_checked_words,
                    g_giant_seed.base_cpu_seconds,g_giant_seed.base_build_seconds);
        stage2_log::print(stage2_log::debug, "real_giant_chain: chunks=%llu seed_points=%llu chunks_per_ladder=%llu\n",
                    BR.giant_chain_chunks, BR.giant_seed_points,
                    BR.giant_chain_chunks ? 0ull : 1ull);
        /* objective 3: giant points that are the IDENTITY modulo a factor of N -- the hit made
           explicit.  Their gcd(Z, N) was recorded as a factor (credited to no hit). */
        stage2_log::print(stage2_log::debug, "real_giant_degenerate: points=%llu\n", BR.giant_degenerate);
        stage2_log::print(stage2_log::debug, "real_batched_breakdown: wall=%.2f ntt_calls=%llu ntt_launches=%llu "
                    "ntt_seconds=%.3f (%.1f%%) arena_mb=%.1f arena_overflow=%llu\n", el,
                    L.ntt_calls - nb, L.ntt_launches - nl, L.ntt_seconds - ns,
                    el > 0 ? 100.0 * (L.ntt_seconds - ns) / el : 0.0, arena.mb(),
                    BR.arena_overflow);
        arena.print_workspace_stats();
        /* WHERE the NTT-attributed time goes, PER CALL (objective 4).  The phase columns are only
           filled when NTT_HOST_BREAK=1 (two clock reads per phase); without it they print 0 and
           only per_call is meaningful.  The point of the line: `ntt_seconds/ntt_calls` at the real
           shape is ~165 us per call, while the GPU phases measured on the frozen shape are ~50 us,
           so the rest is wrapper/host work -- and these columns say which part. */
        if (L.ntt_calls > nb) {
            const double inv_c = 1.0 / (double)(L.ntt_calls - nb);
            stage2_log::print(stage2_log::debug, "real_batched_ntt_usecall: per_call=%.1f setup=%.1f pack=%.1f maxcoeff=%.1f "
                        "h2d=%.1f fwd=%.1f inv=%.1f carry_slot_out=%.1f d2h=%.1f extract=%.1f "
                        "xcheck=%.1f hostside=%.1f | calls=%llu\n",
                        inv_c * (L.ntt_seconds - ns) * 1e6, inv_c * (L.t_setup - st0) * 1e6,
                        inv_c * (L.t_pack - pk0) * 1e6, inv_c * (L.t_maxc - mc0) * 1e6,
                        inv_c * (L.t_h2d - h20) * 1e6, inv_c * (L.t_fwd - fw0) * 1e6,
                        inv_c * (L.t_inv - iv0) * 1e6, inv_c * (L.t_slot - sl0) * 1e6,
                        inv_c * (L.t_d2h - d20) * 1e6, inv_c * (L.t_ext - ex0) * 1e6,
                        inv_c * (L.t_xchk - xc0) * 1e6,
                        inv_c * ((L.t_setup - st0) + (L.t_pack - pk0) + (L.t_maxc - mc0) +
                                 (L.t_h2d - h20) + (L.t_d2h - d20) + (L.t_ext - ex0) +
                                 (L.t_xchk - xc0)) * 1e6,
                        L.ntt_calls - nb);
            /* SECTION 29: the deferred carry check, counted so a test can prove it RAN rather
               than merely that it did no harm -- the whole point of the change is that the
               per-chunk readback disappears, and a run that silently stopped checking would look
               identical in the timing. */
            stage2_log::print(stage2_log::debug, "real_batched_carrydefer: chunks_deferred=%llu finishes=%llu "
                        "deferred_slices=%llu batch_enabled=%d checked_chunks=%llu max_group=%llu\n",
                        g_defer_chunks, g_defer_finishes, g_defer_slices, (int)g_s4_carry_batch,
                        g_defer_checked_chunks, g_defer_max_group);
            stage2_log::print(stage2_log::debug, "real_batched_carrytrace: enabled=%d words=%llu signature=%016llx\n",
                        (int)g_s4_carry_trace, g_carry_output_words, g_carry_output_hash);
            stage2_log::print(stage2_log::debug, "real_batched_outputwindow: enabled=%d calls=%llu source_coeffs=%llu reduced_coeffs=%llu "
                        "returned_coeffs=%llu skipped_coeffs=%llu d2h_words=%llu device_peak_bytes=%llu "
                        "pinned_peak_bytes=%llu (all S4 calls incl F-tree; shape/carry full)\n",
                        (int)g_s4_output_window,g_output_window.calls,g_output_window.source_coeffs,
                        g_output_window.reduced_coeffs,g_output_window.returned_coeffs,
                        g_output_window.skipped_coeffs,g_output_window.d2h_words,
                        g_output_window.device_peak_bytes,g_output_window.pinned_peak_bytes);
            stage2_log::print(stage2_log::debug, "real_batched_chunkoutput: enabled=%d calls=%llu reused_calls=%llu whole_calls=%llu "
                        "multi_chunk_reused=%llu reused_chunks=%llu legacy_calls=%llu grows=%llu "
                        "request_peak_bytes=%llu whole_peak_bytes=%llu retained_peak_bytes=%llu\n",
                        (int)g_s4_chunk_output,g_chunk_output.calls,g_chunk_output.reused_calls,
                        g_chunk_output.whole_calls,g_chunk_output.multi_chunk_reused,g_chunk_output.reused_chunks,
                        g_chunk_output.legacy_calls,g_chunk_output.grows,g_chunk_output.request_peak_bytes,
                        g_chunk_output.whole_peak_bytes,g_chunk_output.retained_peak_bytes);
            stage2_log::print(stage2_log::debug, "real_batched_finalreadback: enabled=%d calls=%llu copied_words=%llu "
                        "avoided_words=%llu host_peak_bytes=%llu t_copy=%.6f "
                        "(all S4 calls incl F-tree; additional to chunk coeffback)\n",
                        (int)g_s4_final_readback,g_final_readback.calls,g_final_readback.copied_words,
                        g_final_readback.avoided_words,g_final_readback.host_peak_bytes,g_final_readback.t_copy);
            if (L.s4) stage2_log::print(stage2_log::debug, "real_batched_input: direct_enabled=%d direct_chunks=%llu copied_chunks=%llu "
                        "d2d_bytes=%llu avoided_bytes=%llu packed_peak_bytes=%llu "
                        "temp_peak_bytes=%llu temp_current_bytes=%llu pack_host=%.6f copy_host=%.6f "
                        "(all S4 device-packed calls)\n",
                        (int)g_s4_pack_direct, L.s4->input_direct, L.s4->input_copied,
                        L.s4->input_d2d_bytes, L.s4->input_avoided_bytes, L.s4->packed_peak_bytes,
                        L.s4->temp_peak_bytes, 16ull * L.s4->d_pack_cap,
                        L.s4->t_packdev, L.s4->t_input_copy_host);
            stage2_log::print(stage2_log::debug, "real_batched_carrytime: group_readback=%.6f chunk_readback=%.6f "
                        "total=%.6f (all S4 calls, each readback charged once)\n",
                        g_carry_group_readback, g_carry_chunk_readback,
                        g_carry_group_readback + g_carry_chunk_readback);
            stage2_log::print(stage2_log::debug, "real_batched_flatinput: direct_enabled=%d calls=%llu borrowed=%llu padded=%llu "
                        "alias_clones=%llu copy_bytes=%llu zero_bytes=%llu avoided_copy_bytes=%llu "
                        "avoided_zero_bytes=%llu temp_peak_bytes=%llu control_peak_bytes=%llu t_prepare=%.6f "
                        "(all flat_mul_batch calls, host padding only)\n", (int)g_s4_flat_direct,
                        g_flat_input.calls, g_flat_input.borrowed, g_flat_input.padded, g_flat_input.alias_clones,
                        g_flat_input.copy_bytes, g_flat_input.zero_bytes, g_flat_input.avoided_copy_bytes,
                        g_flat_input.avoided_zero_bytes, g_flat_input.temp_peak_bytes,
                        g_flat_input.control_peak_bytes, g_flat_input.t_prepare);
            /* SECTION 30: the pinned/async path, counted for the same reason as above -- and the
               fallback count, so a run that silently lost pinned memory cannot look normal */
            stage2_log::print(stage2_log::debug, "real_batched_asyncxfer: raw_async=%llu out_async=%llu fallbacks=%llu "
                        "(async_enabled=%d) raw_reuse_waits=%llu raw_reuse_wait=%.6f "
                        "raw_pinned_bytes=%llu\n",
                        g_pin_raw_used, g_pin_out_used, g_pin_fallbacks, g_s4_async ? 1 : 0,
                        g_pin_raw_waits, g_pin_raw_wait,
                        8ull * (g_pin_raw_cap[0][0] + g_pin_raw_cap[0][1] +
                                g_pin_raw_cap[1][0] + g_pin_raw_cap[1][1]));
            /* OBJECTIVE 4: the one phase the probe cannot see -- the reduced product coming back
               to the host once per chunk in the batched path (section 32). */
            const double gb = (double)(L.d2h_coeff_words - dw0) * 8.0 / 1073741824.0;
            const double tdc = L.t_d2h_coeff - dc0;
            stage2_log::print(stage2_log::debug, "real_batched_coeffback: us_per_call=%.1f total=%.3f s volume=%.2f GB "
                        "effective_GBps=%.2f share_of_ntt=%.1f%%\n",
                        inv_c * tdc * 1e6, tdc, gb, tdc > 0 ? gb / tdc : 0.0,
                        (L.ntt_seconds - ns) > 0 ? 100.0 * tdc / (L.ntt_seconds - ns) : 0.0);
            /* and the three phases the batched entry point never had timers for */
            const double thp = L.t_hpack - hp0, tsc = L.t_scan - sc0, th2 = L.t_h2d_batch - hb0;
            const double tck = L.t_check - ck0;
            const double acc = inv_c * (thp + tsc + th2) * 1e6;
            stage2_log::print(stage2_log::debug, "real_batched_hostbatch: us_per_call=%.1f (pack=%.1f scan=%.1f h2d=%.1f) "
                        "totals pack=%.3f scan=%.3f h2d=%.3f s share_of_ntt=%.1f%%\n", acc,
                        inv_c * thp * 1e6, inv_c * tsc * 1e6, inv_c * th2 * 1e6, thp, tsc, th2,
                        (L.ntt_seconds - ns) > 0 ? 100.0 * (thp + tsc + th2) / (L.ntt_seconds - ns)
                                                 : 0.0);
            /* THE CARRY-CONVERGENCE ASSERT (section 34): a whole extra pass over the digit array
               plus a D2H, run on EVERY call and never timed before now. */
            stage2_log::print(stage2_log::debug, "real_batched_carrycheck: us_per_call=%.1f total=%.3f s "
                        "share_of_ntt=%.1f%%\n", inv_c * tck * 1e6, tck,
                        (L.ntt_seconds - ns) > 0 ? 100.0 * tck / (L.ntt_seconds - ns) : 0.0);
            /* WHICH PART of that is the pass and which is the drain (section 41): the kernel's
               own GPU time comes from in-stream events, the readback is the blocking D2H of the
               two 8-byte counters.  A readback far larger than the kernel is pure latency. */
            {
                const double trs = L.t_check_reset - cr0, tke = L.t_check_kernel - ck0b,
                             td2 = L.t_check_d2h - cd0;
                stage2_log::print(stage2_log::debug, "real_batched_carrysplit: reset_us_per_call=%.1f kernel_us_per_call="
                            "%.1f d2h_us_per_call=%.1f | totals reset=%.3f kernel=%.3f "
                            "d2h=%.3f s\n", inv_c * trs * 1e6, inv_c * tke * 1e6, inv_c * td2 * 1e6,
                            trs, tke, td2);
            }
            /* ... and the two remaining pieces of the device entry point: the shape plan
               (choose_cfg re-proving the exactness bound per call) and the device-to-device copy
               of the packed operands into the arena's scratch. */
            const double tpl = L.t_plan - pl0b, toc = L.t_opcopy - oc0;
            const double tho = L.t_hout - ho0;
            stage2_log::print(stage2_log::debug, "real_batched_devpath: plan_us_per_call=%.1f (total=%.3f s) "
                        "opcopy_us_per_call=%.1f (total=%.3f s) hout_us_per_call=%.1f "
                        "(total=%.3f s) share_of_ntt=%.1f%%\n",
                        inv_c * tpl * 1e6, tpl, inv_c * toc * 1e6, toc, inv_c * tho * 1e6, tho,
                        (L.ntt_seconds - ns) > 0 ? 100.0 * (tpl + toc + tho) / (L.ntt_seconds - ns)
                                                 : 0.0);
            /* OBJECTIVE 4, the DEVICE-pack version of the same two items: the raw coefficient
               upload and the packing pass are now on the device, so what is left here is one
               small H2D per chunk plus two kernel launches (section 33). */
            if (L.s4) {
                const double traw = L.s4->t_h2d_raw - ry0;
                const double gbr = (double)(L.s4->raw_words - rw0) * 8.0 / 1073741824.0;
                stage2_log::print(stage2_log::debug, "real_batched_rawupload: us_per_call=%.1f total=%.3f s volume=%.2f GB "
                            "effective_GBps=%.2f pack_launches=%llu share_of_ntt=%.1f%%\n",
                            inv_c * traw * 1e6, traw, gbr, traw > 0 ? gbr / traw : 0.0,
                            L.s4->pack_launches - pl0,
                            (L.ntt_seconds - ns) > 0 ? 100.0 * traw / (L.ntt_seconds - ns) : 0.0);
            }
        }
        if (s4_on) {
            unsigned long long sel_cases = 0, sel_bad = 0, checked = 0, check_bad = 0,
                               full = 0, canon = 0, coeffs = 0;
            double t_red = 0.0, t_red_host = 0.0;
            unsigned long long dt_blocked = 0;
            s4_dbad_resolve(red, /*wait=*/true);   /* section 43: the last readback must be checked */
            s4_oracle_drain(red);
            s4_oracle_report();
            for (S4Reduce::Shape *S : red.shapes) {
                S->dt_flush();     /* the run is over and the stream has drained: resolve the
                                      deferred marks now (section 41) */
                sel_cases += S->selftest_cases; sel_bad += S->selftest_bad;
                checked += S->checked; check_bad += S->check_bad; full += S->full_checks;
                canon += S->canon_bad; coeffs += S->coeffs; t_red += S->t_reduce;
                t_red_host += S->t_reduce_host; dt_blocked += S->dt_blocks;
                stage2_log::print(stage2_log::debug, "s4_reduce_stats: P=%llu slot_bits=%llu L=%d nlimb=%d launches=%llu "
                            "coeffs=%llu gmp_checked=%llu gmp_bad=%llu slot_canonical_bad=%llu "
                            "t_reduce=%.3f\n", S->P, S->slot_bits, S->L, S->nlimb, S->calls,
                            S->coeffs, S->checked, S->check_bad, S->canon_bad, S->t_reduce);
                /* section 41: how much of the hook is the KERNEL and how much is the host */
                stage2_log::print(stage2_log::debug, "s4_reduce_split: P=%llu kernel_us_per_call=%.2f "
                            "host_us_per_call=%.2f ring_waits=%llu | t_reduce_host=%.3f s\n",
                            S->P, S->calls ? 1e6 * S->t_reduce / (double)S->calls : 0.0,
                            S->calls ? 1e6 * S->t_reduce_host / (double)S->calls : 0.0,
                            S->dt_blocks, S->t_reduce_host);
                /* the hook's out-of-timer work on the REAL shape too (section 36): the canonical
                   counter's readback and the in-run GMP oracle */
                stage2_log::print(stage2_log::debug, "s4_reduce_hook_tail: P=%llu d2h_bad_us_per_call=%.2f "
                            "sample_us_per_call=%.2f | t_hookd2h=%.3f s t_hooksample=%.3f s\n",
                            S->P, S->calls ? 1e6 * S->t_hookd2h / (double)S->calls : 0.0,
                            S->calls ? 1e6 * S->t_hooksample / (double)S->calls : 0.0,
                            S->t_hookd2h, S->t_hooksample);
            }
            stage2_log::print(stage2_log::phases, "s4_multiply_stats: enabled=1 launches=%llu poly_muls=%llu "
                        "coeffs_reduced=%llu t_reduce=%.3f t_reduce_host=%.3f ring_waits=%llu "
                        "gmp_selftest_cases=%llu "
                        "gmp_selftest_bad=%llu gmp_checked=%llu gmp_check_bad=%llu "
                        "full_checks=%llu\n", s4.launches, s4.muls, coeffs, t_red, t_red_host,
                        dt_blocked, sel_cases,
                        sel_bad, checked, check_bad, full);
        }
    }
    mpz_clears(R, tmp, a24, ax, az, nullptr);
    return 0;
}

#include "../core/ecm_cuda_stage2.h"
#include "ecm_stage2_tune.cuh"

bool ecm_cuda_stage2_shape_query(uint64_t p,int bits,uint64_t *length,uint64_t *output_slots)
{
    if(!p || p>(1ull<<28) || bits<2 || bits>ecm_stage2::max_input_bits)return false;
    unsigned long long n=0,out=0;
    if(!ntt_shape_query(p,bits,&n,nullptr,nullptr,nullptr,nullptr,&out))return false;
    if(length)*length=n;if(output_slots)*output_slots=out;return true;
}

int ecm_cuda_stage2_device_info(int device,EcmStage2DeviceInfo *info,const char *expected_uuid)
{
    if(!info || device<0)return 2;
    *info=EcmStage2DeviceInfo{};
    cudaDeviceProp prop{};size_t free=0,total=0;
    if(cudaGetDeviceProperties(&prop,device)!=cudaSuccess)return 2;
    const char *hex="0123456789abcdef";
    for(int i=0;i<16;++i){const unsigned char b=(unsigned char)prop.uuid.bytes[i];info->uuid_hex[2*i]=hex[b>>4];info->uuid_hex[2*i+1]=hex[b&15];}
    if(expected_uuid && std::strcmp(expected_uuid,info->uuid_hex))return 3;
    if(cudaSetDevice(device)!=cudaSuccess || cudaMemGetInfo(&free,&total)!=cudaSuccess)return 2;
    info->free_bytes=free;info->total_bytes=total;info->major=prop.major;info->minor=prop.minor;
    if(cudaRuntimeGetVersion(&info->runtime)!=cudaSuccess || cudaDriverGetVersion(&info->driver)!=cudaSuccess)return 2;
#ifdef NTT_GL_FIXED_MODE
    info->fixed_mode=NTT_GL_FIXED_MODE;
#endif
#ifdef NTT_OUTER_UNROLL_U
    info->outer_unroll_u=NTT_OUTER_UNROLL_U;
#endif
    return 0;
}

int ecm_cuda_stage2_tune_ntt(int device,int min_log2,int max_log2,int repeats,
                            uint64_t memory,void (*report)(const char*,void*),void *context)
{
    if(ecm_cuda_stage2_check_configuration())return 2;
    return stage2_tune::run(device,min_log2,max_log2,repeats,memory,report,context);
}

int ecm_cuda_stage2_plan(const char *n_hex,uint64_t sigma,uint64_t b1,uint64_t b2,
                        uint64_t d,int device,void (*report)(const char*,void*),void *context)
{
    if(!report || !n_hex || device<0 || !b1 || b2<=b1 ||
       b2>(uint64_t)INT64_MAX-8192 || (d && (d<6 || d%2)))return 2;
    if(ecm_cuda_stage2_check_configuration())return 2;
    g_device=device;
    ecm_stage2::Plan p;
    const int code=run_real(n_hex,true,sigma,b1,b2,d,d==0,false,1,nullptr,nullptr,true,&p);
    if(code)return code;
    const auto &g=p.geometry;
    std::ostringstream json;json<<std::setprecision(17)
        <<"{\"type\":\"stage2_plan\",\"schema\":1,\"curves_executed\":0,\"bits\":"<<g.bits
        <<",\"words\":"<<g.words<<",\"B1\":"<<p.b1<<",\"B2\":"<<p.b2<<",\"D\":"<<p.d
        <<",\"P\":"<<g.p<<",\"I\":"<<p.giant_points<<",\"G\":"<<p.batches
        <<",\"fold_length\":"<<g.fold_length<<",\"tree_length\":"<<g.tree_length
        <<",\"fold_big_bytes\":"<<g.fold_big_bytes<<",\"arena_estimate_bytes\":"<<g.arena_estimate_bytes
        <<",\"owner_bytes\":"<<g.fold_owner_bytes<<",\"owner_reuse\":"<<kFoldOwnerReuse<<",\"baby_payload_bytes\":"<<p.baby_bytes
        <<",\"free_bytes\":"<<p.free_bytes<<",\"arena_cap_bytes\":"<<p.arena_cap_bytes
        <<",\"owner_budget_bytes\":"<<p.owner_budget_bytes
        <<",\"owner_budget_fits\":"<<(p.owner_budget_fits ? "true" : "false")
        <<",\"arena_estimate_fits\":"<<(p.arena_estimate_fits ? "true" : "false")
        <<",\"residency_guaranteed\":false,\"process_peak_estimated\":false,\"accounting_version\":2"
        <<",\"model\":"<<stage2_tune::quote(p.model)<<",\"calibrated\":"<<(p.calibrated ? "true" : "false")
        <<",\"stage2_seconds_estimate\":"<<p.estimated_seconds<<"}";
    report(json.str().c_str(),context);return 0;
}

int ecm_cuda_stage2_run(const char *n_hex, const char *x_hex, uint64_t sigma,
                       uint64_t b1, uint64_t b2, uint64_t d, int device,
                       void (*report)(const char *, void *), void *context)
{
    if(ecm_cuda_stage2_check_configuration())return 2;
    s2g_install_crash_handler();
    s2g_install_terminate();
    ladder_cap_init();
    g_device = device;
    const char *progress = std::getenv("NTT_NO_PROGRESS");
    if (progress && std::atoi(progress)) g_s4_batched_progress = false;
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    Stage2Tail tail;
    const int code = run_real(n_hex, true, sigma, b1, b2, d, d == 0,
                              false, 1, x_hex, &tail);
    if (code || tail.bad_factors) return code ? code : 1;
    std::string json = "\"hits\":" + std::to_string(tail.hits) +
        ",\"bad_factors\":" + std::to_string(tail.bad_factors) + ",\"factors\":[";
    for (size_t i = 0; i < tail.factors.size(); ++i) {
        if (i) json += ',';
        json += '"' + tail.factors[i] + '"';
    }
    json += ']';
    report(json.c_str(), context);
    return 0;
}

int ecm_cuda_stage2_default_log_level() { return stage2_log::batches; }
int ecm_cuda_stage2_set_log_level(int level) {
    if(level<stage2_log::quiet || level>stage2_log::debug)return 2;
    stage2_log::level=level;return 0;
}
int ecm_cuda_stage2_check_configuration() {
    struct FixedOption { const char *key; const char *value; };
    if(std::getenv("NTT_FUSE_COOP_M")) {
        std::fprintf(stderr,"production Stage2 requires automatic NTT_FUSE_COOP_M; use -Engine development for radix comparisons\n");
        return 2;
    }
    const FixedOption options[] = {
        {"NTT_XADD6", "1"},
        {"NTT_POINT_MERSENNE", "1"},
        {"NTT_GIANT_SEED_DEVICE", "1"},
        {"NTT_GIANT_SEED_PAIR", "1"},
        {"NTT_GSCALE_DEVICE", "1"},
        {"NTT_GSCALE_DEVICE_TEST", "0"},
        {"NTT_GFINV_SEG_EXACT", "1"},
        {"NTT_GFINV_BATCH", "1"},
        {"NTT_SMALL_PRIME_REUSE", "1"},
        {"NTT_FOLD_FLAT", "1"},
        {"NTT_FOLD_DEVICE", "1"},
        {"NTT_FOLD_OWNER_REUSE", "3"},
        {"NTT_GROOT_DEVICE", "1"},
        {"NTT_SCALED_DESCENT", "1"},
        {"NTT_S4_OUTPUT_WINDOW", "1"},
        {"NTT_S4_CHUNK_OUTPUT", "1"},
        {"NTT_DEVICE_GLEAF", "1"},
        {"NTT_GROOT_TO_FOLD", "1"},
        {"NTT_S4_ORACLE_ASYNC", "1"},
        {"NTT_S4_CARRY_BATCH", "1"},
        {"NTT_FUSE_WARP_TAIL", "1"},
        {"NTT_BABY_DEVICE", "1"},
        {"NTT_S4_MERSENNE", "1"},
        {"NTT_S4_PACK_DIRECT", "1"},
        {"NTT_S4_FLAT_DIRECT", "1"},
        {"NTT_GROOT_COMPACT_RAW", "1"},
        {"NTT_GROOT_LEAF_STAGING", "1"},
        {"NTT_GL_SHORT_REDUCE", "1"},
        {"NTT_S4_OLDTAIL", "0"},
        {"NTT_S5_ON", "0"},
        {"NTT_S4_OFF", "0"},
        {"NTT_GIANT_BASE_CPU", "0"},
        {"NTT_CARRY_CHECK_FUSED", "0"},
        {"NTT_GL_SHIFT_SCALE", "0"},
        {"NTT_GIANT_CHAIN_SMALL_BLOCK", "0"},
        {"NTT_S4_FINAL_READBACK", "0"},
        {"NTT_S4_HOSTPACK", "0"},
        {"NTT_FUSE_COOP_OUTER", "2"},
        {"NTT_FUSE_T", "12"},
        {"NTT_FUSE_M", "4"},
        {"NTT_FUSE_COMPACT_SCRATCH", "1"},
        {"NTT_S4_GROOT_ONLY", "1"},
        {"NTT_S5_REDDUMP", "0"},
        {"NTT_GIANT_CHAIN_BLOCK", "64"},
        {"NTT_GIANT_CHAIN_MIN", "32768"},
    };
    for(const auto &option:options) {
        const char *value=std::getenv(option.key);
        if(value && std::strcmp(value,option.value)) {
            std::fprintf(stderr,"production Stage2 requires %s=%s; use -Engine development for algorithm comparisons\n",option.key,option.value);
            return 2;
        }
    }
    return 0;
}
