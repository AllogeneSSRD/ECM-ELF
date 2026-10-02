/* ---------------------------------------------------------------------------
 * stage2_tree_gpu.cu -- M3 slice S1 (docs/DEV_STAGE2_GPU_PLAN.md section 18):
 * the POLYNOMIAL LAYER and the F PRODUCT TREE, on the GPU, verified coefficient by
 * coefficient against the CPU/GMP tree reference.
 *
 * This file is the GPU twin of the F-tree half of tools/bench/stage2_tree_ref.cpp.  Its
 * oracle is that reference, and the oracle is exact:
 *
 *     F(X) = prod_{j coprime to D, j <= D/2} (X - x_j)  mod N,     x_j = affine_x([j]Q)
 *
 * with Q taken FROM THE CPU REFERENCE (tools/bench/stage2_tree_ref.cpp --dump-F <file>),
 * so both sides provably use the same point: the GPU re-derives every x_j from that Q with
 * its own x-only Montgomery ladder and builds F with its own NTT product tree, and then the
 * two lists of coefficients are compared one by one (mod N).  The dump carries the CPU's own
 * x_j as well, so the ladder is checked point by point BEFORE the tree is checked coefficient
 * by coefficient -- a failure says which layer broke.
 *
 * =====================================================================================
 * (a) THE COEFFICIENT REPRESENTATION MOD N  -- decided here, because slices S2 (remainder
 *     tree) and S3 (batching) place their memory budgets on top of it.
 *
 *     A polynomial of degree < m with coefficients mod N is a FLAT ARRAY OF
 *     m * W  64-BIT WORDS, W = ceil(bits(N)/64), COEFFICIENT-MAJOR:
 *
 *         poly[i*W + t]  =  limb t (little-endian, limb 0 = least significant) of
 *                           coefficient i,      i = 0 .. m-1
 *
 *     with every coefficient REDUCED into [0, N) and every bit at or above bits(N) zero.
 *
 *     Why this representation:
 *       1. it is BIT-FOR-BIT the input format of the NTT multiply (ntt_poly_mul_host takes
 *          "P coefficients of S bits each, coefficient i in the W words at [i*W,(i+1)*W)"),
 *          so the hot path needs no conversion at all -- and S = bits(N) is exactly right,
 *          because a coefficient mod N has at most bits(N) bits;
 *       2. coefficient-major keeps each coefficient CONTIGUOUS, which is what the remainder
 *          tree of S2 wants for its coefficient-wise work (scaled remainder descent is one
 *          multiply-subtract per coefficient, with no gather);
 *       3. the alternative "one limb plane per limb index" layout (limb t of every
 *          coefficient contiguous) would be friendlier to a per-limb NTT schedule, but the
 *          NTT here is ONE Kronecker multiply over whole coefficients, so it buys nothing
 *          today; it is recorded as the option if S4's multi-limb schedule ever wants it.
 *     Every coefficient is a full W-word value, so a partly used top word is normal; all
 *     arithmetic goes through GMP or through explicit mod-N reductions, never a raw add.
 *
 * (b) MULTIPLY = THE PROBE'S NTT, NOT A COPY.  ntt_poly_mul_host (in ntt_poly_probe.cu,
 *     included below with NTT_POLY_PROBE_NO_MAIN) is the ONE implementation of the
 *     Kronecker/integer-NTT polynomial multiply; the probe's own `poly` CLI mode calls the
 *     same function, so the two cannot drift.  The tree supplies two operands of
 *     P = max(m1,m2) coefficients of S = bits(N) bits (zero-padded) and asks for the EXACT
 *     product coefficients; the exactness bound L*(2^bpw-1)^2 < p with L = P*slot_words is
 *     derived and asserted inside that function for the tree's own (P, S) -- never inherited
 *     from the probe's shapes -- and re-asserted here from the values that come back.  Because
 *     the returned coefficients are exact integers of up to 2*bits(N)+log2(P) bits, and N is
 *     composite and far wider than one slot, the reduction mod N (and the zero padding) is
 *     done here on the host with GMP: correctness first (section 18: S1/S2 are not about
 *     performance).  Moving that reduction onto the device belongs to S4, where the
 *     coefficients are 5153 bits wide; the shape-independent part of it (mod-N arithmetic on
 *     the device) already exists below, in the Montgomery ladder.
 *
 * (c) THE BABY TREE: a power-of-two padded heap exactly like the CPU reference's
 *     build_product_tree -- leaves 0..m-1 are (X - x_j), the padding leaves are the constant
 *     1, and a node with a constant-1 child IS the other child (the CPU reference's
 *     poly_is_one shortcut) -- so both sides perform the same set of multiplies.
 *
 * (d) THE ORACLE: --check-F <dumpfile> compares
 *       * every baby point x_j (device ladder vs CPU ladder), and
 *       * every coefficient of F (GPU NTT tree vs CPU schoolbook tree), mod N,
 *     printing the FIRST mismatch index with both values in hex.  --dump-F-gpu writes the
 *     GPU's dump in the CPU's format, so the two files can also be compared line by line.
 *     In addition every NTT multiply internally cross-checks its host-extracted exact
 *     coefficients against the device's own mod-SLOT_MOD slot assembly (two independent
 *     extractions; a disagreement is fatal inside ntt_poly_mul_host), and the device mod-N
 *     arithmetic is checked against GMP at startup (mont_selftest).
 *
 * Usage:
 *   stage2_tree_gpu.exe --check-F <cpu_dump_file> [--device 1] [--dump-F-gpu <file>]
 *   stage2_tree_gpu.exe --selftest [--device 1]     (poly multiply vs GMP, two moduli)
 *
 * Printed evidence (machine-readable, one line each):
 *   stage2_tree_gpu: dump=... N_bits=... D=... B1=... B2=... sigma=... device=...
 *   mont_selftest: cases=... mismatches=... first_bad=...
 *   ladder: baby_points=... mismatches=... first_bad=...
 *   ftree: leaves=... padded=... muls=... ntt_calls=... max_ntt_coeffs=... ...
 *   check_F: coeffs_gpu=... coeffs_cpu=... mismatches=... baby_mismatches=...
 *   check_F: F_degree_gpu=... F_degree_cpu=... ok=1
 * ------------------------------------------------------------------------- */
#define NTT_PROBE_NAME "stage2_tree_gpu"
#define NTT_POLY_PROBE_NO_MAIN 1
#include "ntt_poly_probe.cu"

#include <string>
#include <utility>
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <array>

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
    unsigned long long out[NW];
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

/* r = p + q with diff = p - q: the reference's xadd, in Montgomery images.
   The output is written only at the end, so the ladder may alias it with p and q (the
   reference documents exactly this trap: writing r.X early corrupted Z3). */
template <int NW>
__device__ __forceinline__ void s2g_xadd(unsigned long long *rx, unsigned long long *rz,
                                         const unsigned long long *px, const unsigned long long *pz,
                                         const unsigned long long *qx, const unsigned long long *qz,
                                         const unsigned long long *dx, const unsigned long long *dz,
                                         const unsigned long long *n, unsigned long long ninv,
                                         int nw)
{
    unsigned long long a[NW], b[NW], tx[NW], tz[NW];
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
template <int NW>
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
            s2g_xadd<NW>(r0x, r0z, r0x, r0z, r1x, r1z, px, pz, n, ninv, nw);
            s2g_xdbl<NW>(r1x, r1z, r1x, r1z, a24, n, ninv, nw);
        } else {
            s2g_xadd<NW>(r1x, r1z, r0x, r0z, r1x, r1z, px, pz, n, ninv, nw);
            s2g_xdbl<NW>(r0x, r0z, r0x, r0z, a24, n, ninv, nw);
        }
    }
    for (int i = 0; i < nw; ++i) { rx[i] = r0x[i]; rz[i] = r0z[i]; }
}

/* one thread per point: (X, Z) = [j]Q, written back in the NORMAL domain so the host sees the
   same pair the CPU ladder produces */
template <int NW>
__global__ void s2g_ladder_kernel(const unsigned long long *n, unsigned long long ninv, int nw,
                                  const unsigned long long *qx, const unsigned long long *qz,
                                  const unsigned long long *a24, const unsigned long long *mone,
                                  const unsigned long long *js, int npts,
                                  unsigned long long *out_x, unsigned long long *out_z)
{
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= npts) return;
    unsigned long long rx[NW], rz[NW], one[NW], nx[NW], nz[NW];
    /* GRID-STRIDE: the launcher bounds the number of threads per launch (a ladder is a long
       serial chain, and a multi-second kernel is killed by the display driver's watchdog), so
       one launch may have to cover more points than it has threads.  The stride version is the
       same arithmetic, point by point, with a different work assignment. */
    for (int i = t; i < npts; i += gridDim.x * blockDim.x) {
        s2g_ladder<NW>(js[i], qx, qz, a24, n, ninv, nw, mone, rx, rz);
        for (int j = 0; j < NW; ++j) one[j] = 0;
        one[0] = 1;
        s2g_mont_mul<NW>(nx, rx, one, n, ninv, nw);       /* X = X_m * R^-1 */
        s2g_mont_mul<NW>(nz, rz, one, n, ninv, nw);       /* Z = Z_m * R^-1 */
        for (int j = 0; j < nw; ++j) {
            out_x[(size_t)i * nw + j] = nx[j];
            out_z[(size_t)i * nw + j] = nz[j];
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

/* the word-count dispatch: six instantiations cover every N up to 8192 bits */
#define S2G_DISPATCH(NWD, FN, ...)                                                      \
    do {                                                                                \
        const int _nwd = (NWD);                                                         \
        if (_nwd <= 4) { FN<4>(__VA_ARGS__); }                                          \
        else if (_nwd <= 8) { FN<8>(__VA_ARGS__); }                                     \
        else if (_nwd <= 16) { FN<16>(__VA_ARGS__); }                                   \
        else if (_nwd <= 32) { FN<32>(__VA_ARGS__); }                                   \
        else if (_nwd <= 64) { FN<64>(__VA_ARGS__); }                                   \
        else if (_nwd <= 128) { FN<128>(__VA_ARGS__); }                                 \
        else {                                                                          \
            std::fprintf(stderr, "%s: N has %d words (%d bits); the ladder supports up " \
                                 "to 128 words (8192 bits)\n",                          \
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

/* ONE ladder launch must stay SHORT.  A ladder is a serial 5261-bit chain (~2*S steps), so on
   the real shape a single launch over a whole 207k-point chunk is a ~12-second kernel -- and a
   kernel that long gets the process KILLED with no diagnostic at all: the display driver's
   watchdog (TDR) resets the device and terminates the host process, which is why the real-shape
   runs of section 18.3 died 60-270 s in with an empty stderr and a bare "exit=1" and why the
   Event Log shows nvlddmkm id 13/153 at exactly those moments.  The cap makes every launch
   bounded; the grid-stride loop keeps the same total work and the same result.  Override with
   NTT_LADDER_CAP (0 or unset = the default), e.g. NTT_LADDER_CAP=2048 for a slower/safer run. */
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
                              unsigned long long *dx, unsigned long long *dz)
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
        s2g_ladder_kernel<NW><<<bl, th>>>(dn, ninv, nw, dqx, dqz, da24, dmone, djs + p0, m,
                                          dx + (size_t)p0 * nw, dz + (size_t)p0 * nw);
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
template <int NW>
__global__ void s2g_chain_kernel(const unsigned long long *dn, unsigned long long ninv, int nw,
                                 const unsigned long long *ddx, const unsigned long long *ddz,
                                 const unsigned long long *dsx, const unsigned long long *dsz,
                                 const unsigned long long *esx, const unsigned long long *esz,
                                 unsigned long long npts, unsigned long long per_block,
                                 unsigned long long blocks,
                                 unsigned long long *out_x, unsigned long long *out_z)
{
    const unsigned long long t = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (t >= blocks) return;
    const unsigned long long start = t * per_block;
    if (start >= npts) return;
    const unsigned long long end = ((start + per_block) < npts) ? (start + per_block) : npts;
    unsigned long long xa[NW], za[NW], xb[NW], zb[NW], xc[NW], zc[NW];
    for (int j = 0; j < nw; ++j) {
        xa[j] = dsx[(size_t)t * nw + j];
        za[j] = dsz[(size_t)t * nw + j];
        xb[j] = esx[(size_t)t * nw + j];
        zb[j] = esz[(size_t)t * nw + j];
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
        s2g_xadd<NW>(xc, zc, xb, zb, ddx, ddz, xa, za, dn, ninv, nw);
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
                             unsigned long long *oz)
{
    const unsigned int th = 64;
    const unsigned int bl = (unsigned int)((blocks + th - 1) / th);
    s2g_chain_kernel<NW><<<bl, th>>>(dn, ninv, nw, ddx, ddz, dsx, dsz, esx, esz, npts, per_block,
                                    blocks, ox, oz);
}

/* ===================================================================================== *
 *  THE SEGMENT PRODUCTS OF THE GIANT z-COORDINATES (objective 4, section 42)
 *
 *  One thread per segment of `seg` consecutive giant points; ONE Montgomery multiplication per
 *  point, in the MONTGOMERY DOMAIN the chain already hands back (the product of images is the
 *  image of the product, so no conversion and no extra constant is needed).  The host then asks
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
                                   unsigned long long *out)
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
    for (int i = 0; i < nw; ++i) out[(size_t)s * nw + i] = p[i];
}

template <int NW>
static void s2g_launch_segprod(int nw, unsigned long long npts, unsigned long long seg,
                               unsigned long long ninv, const unsigned long long *dn,
                               const unsigned long long *dz, unsigned long long *out)
{
    const unsigned long long nseg = (npts + seg - 1) / seg;
    const unsigned int th = 64;
    const unsigned int bl = (unsigned int)((nseg + th - 1) / th);
    s2g_segprod_kernel<NW><<<bl, th>>>(dn, ninv, nw, dz, npts, seg, nseg, out);
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
    size_t d_raw_cap = 0, d_pack_cap = 0;
    double t_h2d_raw = 0.0, t_packdev = 0.0;
    unsigned long long raw_words = 0, pack_launches = 0;
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
static unsigned long long *g_pin_raw[2] = {nullptr, nullptr};
static size_t g_pin_raw_cap[2] = {0, 0};
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
   BUT THE PRODUCTION SHAPE REVERSES THE RESULT, and that is why this default stays 32:
   at D=1231230/B2=1.94e12 the 64 MB budget reports "device free=0 MB of 8188 MB" at descent start
   and the host-side naming ladder goes from t_ladder=1.33 s to 142.00 s -- the phase after the
   descent is starved of device memory, and the run takes 373.98 s instead of 253.53 s (measured, both
   with the current binary).  A per-chunk budget is therefore NOT a free lever: it trades tree time
   for whatever needs the device later.  The arena allocations now degrade instead of aborting (see
   NttArena::try_malloc), so a too-large value costs time rather than the run. */
static unsigned long long g_s4_batch_budget_mb = 32;

struct S4Reduce;
static void s4_oracle_release(S4Reduce &R);

struct S4Reduce {
    /* the modulus (shared by every shape of the run) */
    int nw = 0;                          /* words of N */
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
    std::printf("s4_oracle_mode: async=%d ring=%d pack=%d (same samples and GMP predicate)\n",
                (int)g_s4_oracle_async, g_s4_oracle_ring, (int)g_s4_oracle_pack);
}

static void s4_oracle_report()
{
    std::printf("s4_oracle_stats: async=%d selected=%llu queued=%llu compared=%llu samples=%llu "
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
__device__ __constant__ unsigned long long g_div_dns[128];

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

/* base-2^64 limbs of the slot window -> mod N, on the device.  One thread per coefficient. */
template <int NW>
__global__ void s4_reduce_kernel(const unsigned long long *digits, unsigned long long n,
                                 int bpw, unsigned long long slot_words,
                                 unsigned long long out_slots, unsigned long long nbatch,
                                 const unsigned long long *dn, unsigned long long ninv,
                                 int nw, int L, const unsigned long long *dy,
                                 unsigned long long w, unsigned long long *out,
                                 unsigned long long slot_bits, unsigned long long *bad,
                                 unsigned long long *s4_dbg, int tail_mont,
                                 int div_shift, unsigned long long div_recip)
{
    const unsigned long long total = out_slots * nbatch;
    const unsigned long long gid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (gid >= total) return;
    const unsigned long long s = gid / out_slots;
    const unsigned long long k = gid - s * out_slots;
    const unsigned long long *d = digits + s * n + k * slot_words;

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
    /* the value-assertion the reduction bound needs: the digits of THIS slot above slot_bits
       must be zero (that is what makes the coefficient < 2^slot_bits).  Checked here, where the
       digits are already in registers, so the precondition costs no extra launch. */
    {
        const unsigned long long top_bits = slot_bits -
                                            (slot_words - 1) * (unsigned long long)bpw;
        if (top_bits < 64 && bad != nullptr && (d[slot_words - 1] >> top_bits) != 0)
            atomicAdd(bad, 1ull);
    }
    unsigned long long u[NW];
    if (!tail_mont && s4_dbg == nullptr) {
        /* Replace BOTH the L-step elimination and its domain-restoration multiply.  The
           forensic dump explicitly retains the old REDC path because it exposes r*R^-L. */
        const int limbs = (int)((slot_words * (unsigned long long)bpw + 63) / 64);
        const int rc = s4_div_rem<NW>(t, limbs, g_div_dns, nw, div_shift, div_recip, u);
        if (rc < 0 && bad != nullptr) atomicAdd(bad, 1ull);
        for (unsigned long long i = 0; i < w; ++i)
            out[gid * w + i] = (i < (unsigned long long)nw) ? u[i] : 0ull;
        return;
    }
    /* ---- (2) old A/B path: L Montgomery elimination steps ---------------------------- */
    for (int i = 0; i < L; ++i) {
        const unsigned long long m = t[i] * ninv;
        unsigned long long c = 0;
        for (int j = 0; j < nw; ++j) {
            unsigned long long lo, hi;
            s2g_mac(m, dn[j], t[i + j], c, lo, hi);
            t[i + j] = lo;
            c = hi;
        }
        int kk = i + nw;
        while (c != 0) {                            /* carry out: bounded, see the note above */
            const unsigned long long sum = t[kk] + c;
            c = (sum < c) ? 1ull : 0ull;
            t[kk] = sum;
            ++kk;
        }
    }
    unsigned long long r[NW];
    for (int i = 0; i < nw; ++i) r[i] = t[L + i];
    if (t[L + nw] != 0 || s2g_ge_n(r, dn, nw)) s2g_sub_n(r, dn, nw);
    /* FORENSIC (s4_dbg != 0): what this thread assembled and what it is about to return.  The
       array index is the coefficient, so the host can line the two up. */
    if (s4_dbg != nullptr && gid < 4) {
        s4_dbg[gid * 24 + 0] = t[0]; s4_dbg[gid * 24 + 1] = t[1];
        s4_dbg[gid * 24 + 2] = t[2]; s4_dbg[gid * 24 + 3] = t[3];
        s4_dbg[gid * 24 + 4] = t[4]; s4_dbg[gid * 24 + 5] = t[5];
        s4_dbg[gid * 24 + 6] = r[0];
        s4_dbg[gid * 24 + 7] = (nw > 1) ? r[1] : 0ull;
        s4_dbg[gid * 24 + 8] = (nw > 2) ? r[2] : 0ull;
        s4_dbg[gid * 24 + 9] = (unsigned long long)L;
        s4_dbg[gid * 24 + 10] = slot_words;
        s4_dbg[gid * 24 + 11] = d[0];
        s4_dbg[gid * 24 + 12] = d[1];
        s4_dbg[gid * 24 + 13] = d[2];
        s4_dbg[gid * 24 + 14] = d[slot_words - 1];
        s4_dbg[gid * 24 + 15] = (unsigned long long)(int)bpw;
    }
    /* ---- (3) back to the plain domain: C = r * 2^(64L) = Mont(r, Y) ------------------------
       THIS MUST STAY A FULL MONTGOMERY MULTIPLICATION, and that is a measured conclusion, not a
       preference (objective 4 / section 35).  The tail looks like it should be free: the first
       L eliminations left r = C*2^(-64L), so "C mod N" is r*2^(64L), i.e. r SHIFTED up by L
       limbs -- no multiplication needed, only the reduction.  That is WRONG: another L-step
       elimination of the shifted value computes (r*2^(64L))*2^(-64L) = r, i.e. it hands back the
       INVERSE scaling, and the gate caught it immediately (the reduction selftest disagreed with
       GMP on the first shape, 8/18 checks).  To get r*2^(64L) mod N the shifted value must be
       REDUCED, not Montgomery-reduced: a schoolbook/Barrett division of an (nw+L)-limb value by
       an nw-limb modulus costs about L*nw MACs against this tail's 2*nw^2, i.e. ~25% of the
       reduction at L == nw -- which is the only real opening here, and it needs a division with
       a quotient estimate, not a reshuffle of the existing steps.  `tail_mont` is still accepted
       so the A/B switch exists, but the old arithmetic remains the forensic and A/B oracle. */
    s2g_mont_mul<NW>(u, r, dy, dn, ninv, nw);
    if (s4_dbg != nullptr && gid < 4) {
        /* ---- THE WINDOW CHECKSUM (section 48) -------------------------------------------
           `s5_reddump_row` computes the window from a cudaMemcpy of the same pointer the kernel
           was launched with, and the two disagree -- which is only possible if the kernel READ
           something else.  24 debug slots are all taken, so slot 23 (which held `nw`, a value the
           host already knows) carries an order-sensitive checksum of the digits the kernel
           actually consumed.  Equal checksums mean the inputs are identical and the arithmetic is
           the difference; different checksums mean the read is. */
        unsigned long long cks = 0;
        for (unsigned long long j = 0; j < slot_words; ++j) cks += d[j] * (j + 1ull);
        s4_dbg[gid * 24 + 23] = cks;
        s4_dbg[gid * 24 + 16] = u[0];
        s4_dbg[gid * 24 + 17] = (nw > 1) ? u[1] : 0ull;
        s4_dbg[gid * 24 + 18] = (nw > 2) ? u[2] : 0ull;
        s4_dbg[gid * 24 + 19] = dy[0];
        s4_dbg[gid * 24 + 20] = (nw > 1) ? dy[1] : 0ull;
        s4_dbg[gid * 24 + 21] = (nw > 2) ? dy[2] : 0ull;
        s4_dbg[gid * 24 + 22] = ninv;
    }
    for (unsigned long long i = 0; i < w; ++i)
        out[gid * w + i] = (i < (unsigned long long)nw) ? u[i] : 0ull;
}

/* (the slot-canonical check is folded into s4_reduce_kernel above: the digits are in registers
   there, and a separate kernel would add one launch and one full read per batched multiply) */

/* Reduction mode: 0 = direct plain long division of C (default); 1 = old REDC followed by
   Montgomery restoration with Y.  Read once from NTT_S4_OLDTAIL for same-binary A/B. */
static int s4_tail_mont_mode(void)
{
    static int mode = -1;
    if (mode < 0) {
        const char *e = std::getenv("NTT_S4_OLDTAIL");
        mode = (e && *e && std::atoi(e) != 0) ? 1 : 0;
    }
    return mode;
}

template <int NW>
static void s4_launch_reduce(int nw, int L, unsigned long long nbatch,
                             unsigned long long out_slots, unsigned long long total,
                             const unsigned long long *ddig, unsigned long long n, int bpw,
                             unsigned long long slot_words, const unsigned long long *dn,
                             unsigned long long ninv, const unsigned long long *dy,
                             unsigned long long w, unsigned long long *dout,
                             unsigned long long slot_bits, unsigned long long *dbad,
                             unsigned long long *s4_dbg = nullptr)
{
    const unsigned int th = 128;
    const unsigned int bl = (unsigned int)((total + th - 1) / th);
    s4_reduce_kernel<NW><<<bl, th>>>(ddig, n, bpw, slot_words, out_slots, nbatch, dn, ninv, nw,
                                     L, dy, w, dout, slot_bits, dbad, s4_dbg, s4_tail_mont_mode(),
                                     g_div_shift, g_div_recip);
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
   Synthetic moduli exercise shift=0/63, nw=1/128, saturated quotient estimates and the
   add-back branch; those events are too rare for random product coefficients alone. */
static void s4_div_check(const std::vector<unsigned long long> &actual)
{
    mpz_t den, num, want, got, norm, q, dtop, radix;
    mpz_inits(den, num, want, got, norm, q, dtop, radix, nullptr);
    mpz_set_ui(radix, 1);
    mpz_mul_2exp(radix, radix, 64);
    unsigned long long cases = 0, bad = 0, repairs = 0, seed = 0x89abcdef01234567ull;
    const int widths[] = {1, 2, 3, 8, 83, 128};
    for (int fixture = 0; fixture < 19; ++fixture) {
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
            unsigned long long t[260] = {}, out[128] = {};
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
            const int rc = s4_div_rem<128>(t, (int)limbs, ns.data(), nw, shift, recip, out);
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
    std::printf("s4_div_check: cases=%llu bad=%llu repairs=%llu (full remainder vs GMP, "
                "widths=1..128 shifts=0..63)\n", cases, bad, repairs);
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
        CK(cudaMemcpyToSymbol(g_div_dns, g_div_hns.data(),
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
            std::printf("s4_udiv_check: cases=%llu bad=%llu (2-by-1 division vs GMP, dshift=%d)\n",
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
    std::printf("s4_reduce_mode: algorithm=%s nw=%d dshift=%d (NTT_S4_OLDTAIL=%d)\n",
                s4_tail_mont_mode() ? "montgomery" : "division", R.nw, g_div_shift,
                s4_tail_mont_mode());
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
    int nwmax = 128;
    for (int v : {4, 8, 16, 32, 64, 128}) if (R.nw <= v) { nwmax = v; break; }
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
    std::printf("s4_reduce_shape: P=%llu slot_bits=%llu slot_stride=%llu nlimb=%d L=%d "
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
                             long long sample_limit);

static void s4_reduce_hook(void *ctx, const unsigned long long *digits, unsigned long long n,
                           int bpw, unsigned long long slot_words, unsigned long long slot_bits,
                           unsigned long long out_slots, unsigned long long nbatch,
                           unsigned long long *out, unsigned long long w)
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
                 bpw, slot_words, R.dn, R.ninv, S->dy, w, out, slot_bits, R.dbad);
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
    S->coeffs += total;
    R.coeffs_total += total;
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
        s4_check_reduced(R, S, digits, n, out_slots, nbatch, out, g_s4_sample_limit);
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
    std::printf("s4_oracle_pack_check: cases=%llu bad=%llu (exact integers, bpw=1..64, "
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
            std::printf("s4_reduce_realcase: rebuilt=%s ok=%d\n", bs,
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
        std::printf("s4_reduce_selftest_geom: n=%llu out_slots=%llu (the real call's stride)\n",
                    sel_n, sel_os);
    const bool dbg = (std::getenv("NTT_S5_REDDUMP") && *std::getenv("NTT_S5_REDDUMP")
                      && std::atoi(std::getenv("NTT_S5_REDDUMP")) != 0);
    CK(cudaMalloc(&dd, dig.size() * sizeof(unsigned long long)));
    CK(cudaMalloc(&dout, (size_t)(cases * R.w) * sizeof(unsigned long long)));
    if (dbg) CK(cudaMalloc(&ddbg, 4 * 24 * sizeof(unsigned long long)));
    CK(cudaMemcpy(dd, dig.data(), dig.size() * sizeof(unsigned long long),
                  cudaMemcpyHostToDevice));
    S2G_DISPATCH(R.nw, s4_launch_reduce, (int)R.nw, S->L, 1ull, sel_os, sel_os, dd,
                 sel_n, bpw, slot_words, R.dn, R.ninv, S->dy,
                 (unsigned long long)R.w, dout, S->slot_bits, (unsigned long long *)nullptr, ddbg);
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
                std::printf("s4_reduce_selftest_bad: case=%llu pattern=%llu gmp=%s gpu=%s\n",
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
    std::printf("s4_reduce_selftest: P=%llu cases=%llu mismatches=%llu first_bad=%lld (device "
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
                    std::printf("s4_reduce_CHECK_bad: slice=%llu k=%llu gmp=%s gpu=%s "
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
                             long long sample_limit)
{
    const unsigned long long total = out_slots * nbatch;
    if (!total || sample_limit == 0) return;
    const double tc0 = now_s();
    const unsigned long long lim = (unsigned long long)sample_limit;
    const bool small_batch = (nbatch <= 4 && total <= lim);
    std::vector<std::array<unsigned long long, 3>> runs;
    if (small_batch) {
        for (unsigned long long s = 0; s < nbatch; ++s) runs.push_back({s, 0ull, out_slots});
        ++S->full_checks;
    } else {
        const unsigned long long cnt = std::min(lim, out_slots);
        uint64_t seed = 0x9e3779b97f4a7c15ull * (S->calls + 1) + total;
        seed = seed * 6364136223846793005ull + 1442695040888963407ull;
        const unsigned long long s = (seed >> 11) % nbatch;
        seed = seed * 6364136223846793005ull + 1442695040888963407ull;
        const unsigned long long room = (out_slots > cnt) ? (out_slots - cnt + 1) : 1;
        runs.push_back({s, (seed >> 11) % room, cnt});
    }
    if (!g_s4_oracle_async) s4_oracle_block_ready();
    for (const auto &rr : runs) {
        const unsigned long long s = rr[0], k0 = rr[1], cnt = rr[2];
        const size_t ndig = (size_t)(cnt * S->slot_words), nred = (size_t)(cnt * R.w);
        ++g_oracle.selected;
        for (const auto v : {S->P, S->slot_bits, S->slot_words,
                            (unsigned long long)S->bpw, S->calls, s, k0, cnt})
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
                slot.shape = S; slot.slice = s; slot.k0 = k0; slot.count = cnt;
                slot.out_slots = out_slots;
                const double t0 = now_s();
                CK(cudaMemcpyAsync(dig, ddig + s * n + k0 * S->slot_words,
                                   ndig * 8, cudaMemcpyDeviceToHost));
                CK(cudaMemcpyAsync(red, d_out + (s * out_slots + k0) * R.w,
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
            CK(cudaMemcpy(dig.data(), ddig + s * n + k0 * S->slot_words,
                          ndig * 8, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(red.data(), d_out + (s * out_slots + k0) * R.w,
                          nred * 8, cudaMemcpyDeviceToHost));
            g_oracle.t_copy += now_s() - t0;
            s4_compare_snapshot(R, S, s, k0, cnt, out_slots, dig.data(), red.data());
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
                                NttMulStats *st_out = nullptr)
{
    const size_t W = L.W;
    const size_t P = (ma > mb) ? ma : mb;
    const size_t nc = ma + mb - 1;
    out.assign(nbatch * nc * W, 0ull);
    if (nbatch == 0) return;
    S4Ctx &C = *L.s4;
    const unsigned long long out_slots = 2 * P - 1;
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
    const size_t need = nbatch * out_slots * W;
    if (need > C.d_out_cap) {
        if (C.d_out) { cudaFree(C.d_out); C.d_out = nullptr; C.d_out_cap = 0; }
        CK(cudaMalloc(&C.d_out, need * sizeof(unsigned long long)));
        C.d_out_cap = need;
    }
    NttReduceHook hook;
    hook.ctx = C.red;
    hook.run = s4_reduce_hook;
    hook.out = C.d_out;
    hook.w = (unsigned long long)W;

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
    static const unsigned long long s4_batch_mb = [] {
        const char *e = std::getenv("NTT_S4_BATCH_MB");
        return (e && *e) ? std::strtoull(e, nullptr, 10) : 0ull;
    }();
    const unsigned long long budget_bytes =
        (s4_batch_mb ? s4_batch_mb : g_s4_batch_budget_mb) << 20;
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
    int rc = 0;
    /* THE DEVICE PACKING SWITCH (objective 4, section 33): default is the device packer; set
       NTT_S4_HOSTPACK=1 to run the old host-packing path, which is kept as the A/B oracle. */
    static const bool host_pack = [] {
        const char *e = std::getenv("NTT_S4_HOSTPACK");
        return e && *e && std::atoi(e) != 0;
    }();
    /* ---- THE DEFERRED CARRY CHECK (section 29) ----------------------------------------------
       Every chunk ends with the probe's carry check, which is a pageable D2H of 2*nbatch words and
       therefore waits for the device to finish everything queued before it: a full pipeline drain
       per chunk.  Measured at the production shape that is 22471 x 1.64 ms = 36.75 s of a 253.53 s
       run (`carrysplt d2h`), the largest host-side item on the books, and it is what makes the
       chunk size matter so much (fewer chunks = fewer drains, section 27).

       Deferring it needs the counters to survive between chunks, which requires the arena (the
       buffer has to outlive the call), so it is on whenever an arena exists and the device packer
       is in use.  The accumulation lives in the arena entry of the chunk's shape, which is why the
       readback happens BEFORE the next chunk starts: a same-sized next chunk would memset that very
       buffer.  Interior chunks therefore lose their fwd/inv event attribution (the events are
       destroyed unread, since reading them would need the drain we are removing) -- the last chunk
       of each call still reports the sample. */
    const bool defer_ok = (!host_pack && L.arena != nullptr && g_s4_defer_carry);
    /* the doubly-buffered PINNED output staging (section 30): reserved once per call at the largest
       chunk this call can use, so every chunk of the call follows the same path (mixing a blocking
       chunk with the deferred-consumption scheme would break the pending bookkeeping) */
    const size_t out_words_max = (size_t)(chunk * out_slots * W);
    if (!g_pin_ev[0]) CK(cudaEventCreateWithFlags(&g_pin_ev[0], cudaEventDisableTiming));
    if (!g_pin_ev[1]) CK(cudaEventCreateWithFlags(&g_pin_ev[1], cudaEventDisableTiming));
    const bool async_out = (g_s4_async &&
                            pin_words(&g_pin_out[0], &g_pin_out_cap[0], out_words_max) != nullptr &&
                            pin_words(&g_pin_out[1], &g_pin_out_cap[1], out_words_max) != nullptr);
    unsigned long long ci = 0, pend_m = 0, pend_s0 = 0;
    bool out_pending = false;
    bool carry_pending = false;
    unsigned long long carry_pending_m = 0;
    double carry_d2h_acc = 0.0;
    unsigned long long carry_res_acc = 0, carry_bits_acc = 0;
    for (unsigned long long s0 = 0; s0 < nbatch; s0 += chunk) {
        const unsigned long long m = ((nbatch - s0) < chunk) ? (nbatch - s0) : chunk;
        const bool last_chunk = (s0 + m >= nbatch);
        if (carry_pending) {
            NttMulStats fin{};
            const int rf = ntt_batch_carry_finish(L.arena, qN, carry_pending_m, &fin);
            carry_d2h_acc += fin.t_check_d2h;
            carry_res_acc += fin.carry_residual;
            if (fin.carry_max_bits > carry_bits_acc) carry_bits_acc = fin.carry_max_bits;
            carry_pending = false;
            ++g_defer_finishes;
            if (rf != 0) {
                std::fprintf(stderr, "%s: the deferred carry check of the chunked multiply failed "
                                     "(rc=%d) at P=%llu nbatch=%llu chunk=%llu\n",
                             NTT_PROBE_NAME, rf, (unsigned long long)P,
                             (unsigned long long)nbatch, carry_pending_m);
                std::exit(3);
            }
        }
        /* s0 != 0 is NOT optional: the first chunk still has to run the probe's non-deferred
           path because that is what memsets the residual counters.  Skipping it would leave the
           PREVIOUS call's counters in the arena entry, and since a successful check leaves zeros
           the result would not be a false alarm but a silently VACUOUS check for those slices. */
        const bool defer_this = defer_ok && !last_chunk && (s0 != 0);
        NttReduceHook h2 = hook;
        if (hook.out) h2.out = hook.out + (size_t)(s0 * out_slots) * W;
        int r1 = 0;
        if (host_pack) {
            r1 = ntt_poly_mul_batch_host(P, (int)L.S, L.device, m,
                                         wa + s0 * P * W, wb + s0 * P * W,
                                         (s0 == 0) ? &slots : nullptr, &st, L.arena, &h2,
                                         nullptr);
        } else {
            /* 1. the RAW coefficients to the device (P*W words per slice per operand) ... */
            const size_t raw_words = (size_t)m * P * W;
            if (raw_words > C.d_raw_cap) {
                if (C.d_rawA) { cudaFree(C.d_rawA); cudaFree(C.d_rawB); C.d_rawA = C.d_rawB = nullptr; }
                CK(cudaMalloc(&C.d_rawA, raw_words * sizeof(unsigned long long)));
                CK(cudaMalloc(&C.d_rawB, raw_words * sizeof(unsigned long long)));
                C.d_raw_cap = raw_words;
            }
            const double th0 = now_s();
            /* ASYNC UPLOAD VIA PINNED STAGING (section 30): the host memcpy into pinned memory
               touches no device state, so it cannot drain the pipeline, and the pinned
               cudaMemcpyAsync needs no implicit sync either.  Without pinned memory the old
               blocking pair runs unchanged. */
            const size_t raw_bytes = raw_words * sizeof(unsigned long long);
            unsigned long long *pa = g_s4_async ? pin_words(&g_pin_raw[0], &g_pin_raw_cap[0], raw_words)
                                                : nullptr;
            unsigned long long *pb = g_s4_async ? pin_words(&g_pin_raw[1], &g_pin_raw_cap[1], raw_words)
                                                : nullptr;
            if (pa && pb) {
                std::memcpy(pa, wa + s0 * P * W, raw_bytes);
                std::memcpy(pb, wb + s0 * P * W, raw_bytes);
                CK(cudaMemcpyAsync(C.d_rawA, pa, raw_bytes, cudaMemcpyHostToDevice));
                CK(cudaMemcpyAsync(C.d_rawB, pb, raw_bytes, cudaMemcpyHostToDevice));
                ++g_pin_raw_used;
            } else {
                CK(cudaMemcpy(C.d_rawA, wa + s0 * P * W, raw_bytes, cudaMemcpyHostToDevice));
                CK(cudaMemcpy(C.d_rawB, wb + s0 * P * W, raw_bytes, cudaMemcpyHostToDevice));
                if (g_s4_async) ++g_pin_fallbacks;
            }
            /* 2. ... packed into digits ON the device (both operands, one launch each) ... */
            const size_t pack_words = (size_t)m * (size_t)qN;
            if (pack_words > C.d_pack_cap) {
                if (C.d_packA) { cudaFree(C.d_packA); cudaFree(C.d_packB); C.d_packA = C.d_packB = nullptr; }
                CK(cudaMalloc(&C.d_packA, pack_words * sizeof(unsigned long long)));
                CK(cudaMalloc(&C.d_packB, pack_words * sizeof(unsigned long long)));
                C.d_pack_cap = pack_words;
            }
            CK(cudaMemset(C.d_packA, 0, pack_words * sizeof(unsigned long long)));
            CK(cudaMemset(C.d_packB, 0, pack_words * sizeof(unsigned long long)));
            s4_launch_pack_batch(C.d_rawA, (int)L.S, qbpw, qsw, qN, m, P, (int)W, C.d_packA);
            s4_launch_pack_batch(C.d_rawB, (int)L.S, qbpw, qsw, qN, m, P, (int)W, C.d_packB);
            C.t_h2d_raw += now_s() - th0;
            C.raw_words += (unsigned long long)raw_words * 2;
            C.pack_launches += 2;
            /* 3. ... and the multiply itself, device-to-device (the same passes, the same carry,
               the same exactness assertions, the same reduction hook) */
            NttMulStats nst{};
            r1 = ntt_poly_mul_batch_dev(P, (int)L.S, L.device, m, C.d_packA, C.d_packB, &nst,
                                        L.arena, &h2, nullptr, qbpw, defer_this);
            st = nst;                     /* the caller's stats are the dev path's */
            if (defer_this) { carry_pending = true; carry_pending_m = m; ++g_defer_chunks;
                              g_defer_slices += m; }
            if (s0 == 0) slots.clear();   /* the host path filled this; the dev path does not */
        }
        if (r1 != 0) { rc = r1; break; }
        /* the reduced coefficients of this chunk, back to the host -- AND THIS TRANSFER IS THE
           POINT OF OBJECTIVE 4 (section 32): it is `m * out_slots * W` words, i.e. the WHOLE
           product of every slice at full slot width, and it is neither inside the probe's timers
           (they belong to the host implementation) nor inside the tree's own phases.  It is
           therefore timed and counted HERE, so "the 111 us per call that nobody measured" can be
           attributed instead of guessed.

           SECTION 30: it is now an ASYNC copy into pinned staging, and the chunk that is copied
           here is consumed on the NEXT iteration -- so the host never waits for the device unless
           the device is genuinely behind, which is the difference between "the transfer is
           overlapped" and "the pipeline is drained per chunk".  `out_pending` marks the one chunk
           that has been copied but not yet written into `out`; the end of the call drains it. */
        const size_t out_words = (size_t)(m * out_slots * W);
        if (!async_out) {
            /* NO PINNED MEMORY: the original blocking readback, so a failed pinning costs speed
               and never correctness (and never a half-filled `out`) */
            const double td0 = now_s();
            std::vector<unsigned long long> all(out_words, 0ull);
            CK(cudaMemcpy(all.data(), h2.out, out_words * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
            for (unsigned long long s = 0; s < m; ++s)
                std::copy(all.begin() + (long)(s * out_slots * W),
                          all.begin() + (long)(s * out_slots * W + nc * W),
                          out.begin() + (long)((s0 + s) * nc * W));
            L.t_d2h_coeff += now_s() - td0;
            L.d2h_coeff_words += (unsigned long long)out_words;
            continue;
        }
        unsigned long long *po = g_pin_out[(size_t)(ci & 1)];
        const double td0 = now_s();
        CK(cudaMemcpyAsync(po, h2.out, out_words * sizeof(unsigned long long),
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
                std::copy(prev + (size_t)s * out_slots * W,
                          prev + (size_t)s * out_slots * W + nc * W,
                          out.begin() + (long)((pend_s0 + s) * nc * W));
            L.t_d2h_coeff += now_s() - tw0;
        }
        pend_m = m;
        pend_s0 = s0;
        out_pending = true;
        ++ci;
    }
    /* THE ONE DRAIN OF THE COEFFICIENT READBACK: the last chunk copied but not yet consumed */
    if (out_pending) {
        const double tw0 = now_s();
        CK(cudaEventSynchronize(g_pin_ev[(ci - 1) & 1]));
        const unsigned long long *prev = g_pin_out[(ci - 1) & 1];
        for (unsigned long long s = 0; s < pend_m; ++s)
            std::copy(prev + (size_t)s * out_slots * W,
                      prev + (size_t)s * out_slots * W + nc * W,
                      out.begin() + (long)((pend_s0 + s) * nc * W));
        L.t_d2h_coeff += now_s() - tw0;
        out_pending = false;
    }
    L.ntt_seconds += now_s() - t0;
    /* the deferred chunks' carry verdict, folded back into the caller's account: their counters
       were read ONCE by ntt_batch_carry_finish instead of once per chunk (section 29), so this is
       the only place their time and residual totals can be reported */
    if (carry_d2h_acc != 0.0 || carry_res_acc != 0) {
        L.t_check_d2h += carry_d2h_acc;
        st.t_check_d2h += carry_d2h_acc;
        st.carry_residual += carry_res_acc;
        if (carry_bits_acc > st.carry_max_bits) st.carry_max_bits = carry_bits_acc;
    }
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
    /* back to the host: out_slots coefficients per slice, of which the caller keeps nc */
    S4Reduce::Shape *S = C.red->find(st.slot_bits, st.slot_words, st.bpw);
    if (!S) {
        std::fprintf(stderr, "%s: FATAL: the reduction shape changed under the multiply\n",
                     NTT_PROBE_NAME);
        std::exit(3);
    }
    {
        std::vector<unsigned long long> all(need, 0ull);
        CK(cudaMemcpy(all.data(), C.d_out, need * sizeof(unsigned long long),
                      cudaMemcpyDeviceToHost));
        for (size_t s = 0; s < nbatch; ++s)
            std::copy(all.begin() + (long)(s * out_slots * W),
                      all.begin() + (long)(s * out_slots * W + nc * W),
                      out.begin() + (long)(s * nc * W));
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
    std::vector<size_t> &deg, size_t &pad_out, FTreeStats &fs, int cat = -1)
{
    const size_t W = L.W;
    const size_t n = leaf.size();
    size_t pad = 1;
    while (pad < n) pad *= 2;
    fs.leaves = n;
    fs.padded = pad;
    pad_out = pad;
    std::vector<std::vector<unsigned long long>> t(2 * pad, std::vector<unsigned long long>(W, 0ull));
    deg.assign(2 * pad, 0);
    for (size_t i = 0; i < 2 * pad; ++i) t[i][0] = 1;          /* the constant 1 */
    for (size_t i = 0; i < n; ++i) { t[pad + i] = leaf[i]; deg[pad + i] = 1; }
    if (!L.s4) {                                               /* the pre-S4 path, unchanged */
        for (size_t i = pad; i-- > 1; ) {
            if (poly_is_one(t[2 * i].data(), W)) { t[i] = t[2 * i + 1]; deg[i] = deg[2 * i + 1]; }
            else if (poly_is_one(t[2 * i + 1].data(), W)) { t[i] = t[2 * i]; deg[i] = deg[2 * i]; }
            else {
                t[i] = poly_mul_modN(L, t[2 * i], deg[2 * i], t[2 * i + 1], deg[2 * i + 1], cat);
                deg[i] = deg[2 * i] + deg[2 * i + 1];
                ++fs.muls;
            }
        }
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
            if (poly_is_one(t[c0].data(), W)) { t[i] = t[c1]; deg[i] = deg[c1]; }
            else if (poly_is_one(t[c1].data(), W)) { t[i] = t[c0]; deg[i] = deg[c0]; }
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
                t[i].assign(res.begin() + (long)(s * nc * W), res.begin() + (long)((s + 1) * nc * W));
                deg[i] = ma + mb - 2;
                ++fs.muls;
            }
        }
        if (lvl_trace)
            std::printf("tree_level: base=%llu groups=%llu muls=%llu t=%.3f s | ntt total=%.3f "
                        "fwd=%.3f inv=%.3f slot=%.3f hpack=%.3f h2d=%.3f check=%.3f plan=%.3f "
                        "opcopy=%.3f hout=%.3f\n",
                        (unsigned long long)base, lvl_groups, lvl_muls, now_s() - tl0, lvl_total,
                        lvl_fwd, lvl_inv, lvl_slot, lvl_hpack, lvl_h2d, lvl_check, lvl_plan,
                        lvl_opcopy, lvl_hout);
        if (base == 1) break;
    }
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
    std::printf("mont_selftest: cases=%d mismatches=%llu first_bad=%llu "
                "(device Mont(a,b) vs GMP a*b*R^-1 mod N, nw=%llu)\n",
                cases, bad, first, (unsigned long long)nw);
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
static CPoly cp_mul(const CPoly &a, const CPoly &b, PolyLayer &L)
{
    const size_t W = L.W;
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
        poly_mul_batch_modN(L, fa.data(), fb.data(), a.size(), b.size(), 1, fc);
        return cp_from_flat(fc, a.size() + b.size() - 2, W);
    }
    const std::vector<unsigned long long> fa = cp_to_flat(a, W);
    const std::vector<unsigned long long> fb = cp_to_flat(b, W);
    const std::vector<unsigned long long> fc = poly_mul_modN(L, fa, a.size() - 1, fb, b.size() - 1);
    return cp_from_flat(fc, a.size() + b.size() - 2, W);
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
        CPoly ag = cp_mul(at, g, L);
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
        CPoly gn = cp_mul(g, h, L);
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
    CPoly qrev = cp_mul(ra, rbi, L);
    cp_resize(qrev, k, W);
    cp_resize(q, k, W);
    for (size_t i = 0; i < k; ++i) q[i] = qrev[k - 1 - i];
    cp_trim(q);

    const CPoly qb = cp_mul(q, b, L);
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
                if (poly_is_one(Ft[ci].data(), W)) {
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
            std::printf("descent_trace: slow base=%llu nodes=%llu hash=%llu\n",
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

static void descent_batched(PolyLayer &L,
                            const std::vector<std::vector<unsigned long long>> &Ft,
                            const std::vector<size_t> &Fdeg, size_t Fpad, const CPoly &H,
                            std::vector<std::vector<unsigned long long>> &values,
                            unsigned long long &divmods, int cat)
{
    const size_t W = L.W;
    /* THE LEAF COUNT IS THE F TREE'S, NOT THE POLYNOMIAL'S (section 54.5).  It used to be
       `H.size()`, which is the number of COEFFICIENTS H happens to have -- and H is legitimately
       SHORTER than deg F is long: the loop's first batch does `H = T` with no reduction when
       deg T < P, and a block whose giant-point count is below the baby count then leaves H at
       that shorter length.  The descent still owes one value per baby point, with the missing
       high coefficients implicitly zero, so the count must come from the tree (Fdeg[1] = deg F =
       the baby count) and never from the accumulator.  Every consumer downstream (block products,
       the reader of ws.dvals, the descent check's reference) is sized by the caller's P, so
       answering with fewer rows was an out-of-range walk -- measured at D=2310/B2=4e5, where one
       block of 175 giant points against P=240 made the check read 64 rows past `ref`. */
    const size_t P = (Fdeg.size() > 1 && Fdeg[1] > 0) ? Fdeg[1] : H.size();
    const double tdesc0 = now_s();
    double tl0 = tdesc0;
    std::vector<CPoly> cur(1);
    cur[0] = H;
    size_t base = 1, cnt = 1;
    while (base < Fpad) {
        const size_t nbase = base * 2;
        std::vector<CPoly> nxt(cnt * 2);
        /* group the children that really need a division by (deg cur, deg divisor): every pair
           in one group has the same shape, so one batched divmod serves all of them */
        std::map<std::pair<size_t, size_t>, std::vector<std::pair<size_t, size_t>>> groups;
        tl0 = now_s();
        for (size_t j = 0; j < cnt; ++j) {
            for (int sgn = 0; sgn < 2; ++sgn) {
                const size_t ci = nbase + 2 * j + (size_t)sgn;
                const size_t slot = 2 * j + (size_t)sgn;
                if (poly_is_one(Ft[ci].data(), W)) {
                    cp_resize(nxt[slot], 1, W);       /* H mod 1 = 0 */
                    continue;
                }
                if (cur[j].size() < Fdeg[ci] + 1) {   /* the degree fast path: no division */
                    nxt[slot] = cur[j];
                    continue;
                }
                ++divmods;
                groups[std::make_pair(cur[j].size() - 1, Fdeg[ci])].push_back(
                    std::make_pair(j, slot));
            }
        }
        for (auto &g : groups) {
            const size_t da = g.first.first, db = g.first.second;
            const size_t nbatch = g.second.size();
            std::vector<unsigned long long> A(nbatch * (da + 1) * W, 0ull),
                                           B(nbatch * (db + 1) * W, 0ull), R;
            for (size_t s = 0; s < nbatch; ++s) {
                const size_t j = g.second[s].first;
                const size_t ci = nbase + 2 * j + (g.second[s].second & 1ull);
                for (size_t q = 0; q <= da; ++q)
                    std::copy(cur[j][q].begin(), cur[j][q].end(),
                              A.begin() + (long)((s * (da + 1) + q) * W));
                /* the tree node has Fdeg[ci]+1 coefficients, and cp_divmod reads exactly that */
                std::copy(Ft[ci].begin(), Ft[ci].begin() + (long)((db + 1) * W),
                          B.begin() + (long)(s * (db + 1) * W));
            }
            /* LEAF SPECIALISATION -- only where the divisor REALLY has degree 1 and the
               quotient really has 1 coefficient (k = da-db+1 = 1).  a mod (X - x_j) is then the
               single coefficient a(x_j), which is what the generic Newton division computes as
               q = a_da, r = a_0 - x_j*a_da.

               THE SHAPE IS VERIFIED, NOT TRUSTED.  `db` is a GROUP KEY handed in by the caller,
               and at the real shape the group keyed db = 1 held a divisor that was NOT linear
               (leaf_mod_linear_batch's own check caught it, word 2 non-zero), so acting on `db`
               alone corrupts the answer: the divisor's degree has to be read off the divisor.
               Hence all three of: k == 1 (else the quotient needs more than one coefficient),
               lead == 1 (monic), and every word above the constant term == 0 except the lead.
               Anything else falls through to the generic path, which is byte-for-byte the
               reference's cp_divmod.  The shortcut is worth having because the last level is
               P/2 nodes of one shape, but it must never fire on a shape it cannot prove. */
            bool leaf_ok = (db == 1 && da == 1);
            if (leaf_ok) {
                for (size_t s = 0; s < nbatch && leaf_ok; ++s) {
                    const unsigned long long *b = &B[s * 2 * W];
                    if (b[W] != 1) { leaf_ok = false; break; }
                    for (size_t q = 2; q < W; ++q)
                        if (b[q] != 0) { leaf_ok = false; break; }
                }
            }
            if (leaf_ok) {
                std::vector<unsigned long long> Lf, Rg;
                leaf_mod_linear_batch(A, da, B, nbatch, W, L.N, Lf);
                divmod_batch(L, A, B, da, db, nbatch, Rg, cat);   /* the same answer, the long way */
                for (size_t s = 0; s < nbatch; ++s)
                    for (size_t q = 0; q < W; ++q)
                        if (Lf[s * W + q] != Rg[s * db * W + q]) {
                            std::fprintf(stderr, "%s: FATAL: the linear-leaf shortcut disagrees "
                                                 "with the generic division (nbatch=%llu da=%llu "
                                                 "s=%llu word=%llu shortcut=%llu generic=%llu)\n",
                                         NTT_PROBE_NAME, (unsigned long long)nbatch,
                                         (unsigned long long)da, (unsigned long long)s,
                                         (unsigned long long)q, Lf[s * W + q],
                                         Rg[s * db * W + q]);
                            std::exit(3);
                        }
                for (size_t s = 0; s < nbatch; ++s) {
                    CPoly r;
                    cp_resize(r, 1, W);
                    std::copy(Lf.begin() + (long)(s * W), Lf.begin() + (long)((s + 1) * W),
                              r[0].begin());
                    cp_trim(r);
                    nxt[g.second[s].second] = r;
                }
                continue;
            }
            divmod_batch(L, A, B, da, db, nbatch, R, cat);
            for (size_t s = 0; s < nbatch; ++s) {
                CPoly r;
                cp_resize(r, db, W);
                for (size_t i = 0; i < db; ++i)
                    std::copy(R.begin() + (long)((s * db + i) * W),
                              R.begin() + (long)((s * db + i + 1) * W), r[i].begin());
                cp_trim(r);
                nxt[g.second[s].second] = r;
            }
        }
        cur.swap(nxt);
        base = nbase;
        cnt *= 2;
        /* one line per LEVEL: at the real shape the descent is the longest single phase (it
           used to print nothing at all for its whole ~35 minutes, which made the shape look
           hung), and the line shows where the time goes.  Gated by NTT_NO_PROGRESS. */
        if (g_s4_batched_progress)
            std::printf("descent_progress: level=%d base=%llu nodes=%llu divmods_total=%llu "
                        "shape_groups=%llu t_level=%.1f s t_total=%.1f s\n",
                        (int)ceil_log2_u64(base), (unsigned long long)base,
                        (unsigned long long)cnt, divmods, (unsigned long long)groups.size(),
                        now_s() - tl0, now_s() - tdesc0);
        if (g_s4_descent_trace) {
            unsigned long long hh = 1469598103934665603ull;
            for (size_t j = 0; j < cnt; ++j)
                for (const std::vector<unsigned long long> &c : cur[j])
                    for (unsigned long long v : c) hh = (hh ^ v) * 1099511628211ull;
            std::printf("descent_trace: batched base=%llu nodes=%llu hash=%llu\n",
                        (unsigned long long)base, (unsigned long long)cnt, hh);
        }
    }
    values.assign(P, std::vector<unsigned long long>(W, 0ull));
    for (size_t i = 0; i < P; ++i) values[i] = cp_to_flat(cur[i], W);
}

/* =====================================================================================
 * SLICE S5: THE DESCENT, ON THE DEVICE.
 *
 * Section 18.7 measured where the descent's time really goes at the real shape
 * (N = 2^5261-1, P ~ 51839): NOT in the NTT, but in two host-side costs of divmod_batch --
 * (a) it materialises A and B for a WHOLE group before the multiply (~70 GB per buffer at
 *     that shape, twice that for both), and
 * (b) its last step cp_coeff_sub_p is one host GMP subtract+mod per output coefficient
 *     (~2.7e9 of them).
 * Both are gone here: operands are packed into bpw-bit digit slots ON THE DEVICE (one thread
 * per coefficient, no whole-slice staging, no host round trip), the multiply is the existing
 * device-to-device batched multiply, the product coefficients are reduced mod N ON THE DEVICE
 * by the very same s4_reduce_kernel that has already been selftested against GMP for every
 * shape of the run, and the final subtraction is s2g_mont_mul on the device.
 *
 * THE SHAPE IS READ OFF THE DATA, NOT OFF THE GROUP KEY (section 18.6's lesson): a divisor is
 * treated as linear only after a device verdict (leading word == 1, every word above the
 * constant term == 0) AND its group's Fdeg is 1.  Anything else takes the full Newton chain.
 *
 * THE PACK INVARIANT.  The digit buffer the multiply consumes is not a free-form packing: a
 * canonical digit d[j] is < 2^bpw, and the reduction asserts every digit above slot_bits of a
 * slot window is zero.  S5 therefore obeys two rules that cost a whole round to learn:
 *   1. coefficient i is written at SLOT STRIDE (slot_words*bpw) bits, NOT at bpw.  Packing at
 *      bpw overflows the 2^slot_bits window and the reduction aborts with "slot windows have
 *      nonzero digits above slot_bits".
 *   2. the slot must hold the SUM of up to (Lmax + Lb) coefficient products: with A scaled by
 *      2^(S*(Lmax-la)) its slot value is < 2^S(Lmax+1), B's < 2^S(Lb+1), and their product is
 *      < 2^(2S + log2 P) = 2^slot_bits exactly when the scale is (Lmax - la) with Lmax =
 *      ceil(log2 P).  Same for B.  The scaling is FREE: it is a bit offset inside the slot.
 * ===================================================================================== */

/* where the S5 phase's time goes (printed as one line) */
struct S5Stats {
    unsigned long long levels = 0, divmods = 0, generic = 0, linear = 0, copies = 0, zeros = 0,
                       ntt_launches = 0, chunks = 0, forest_nodes = 0, mul_calls = 0;
    double t_pack = 0.0, t_copy = 0.0, t_generic = 0.0, t_linear = 0.0, t_reduce = 0.0,
           t_ntt = 0.0, t_total = 0.0;
    /* BYTES, despite the names: these are `words * 8`, and they used to be PRINTED as "MB", which
       is a factor of a million.  At the frozen vector the frontier's 24 words printed as
       "frontier_mb_peak=192" -- a number a reader would take as 192 MB on an 8 GB card and reason
       about accordingly (and at P=240 the forest printed as "63360 MB", i.e. 63 GB that was never
       allocated).  The labels now say bytes. */
    unsigned long long forest_bytes = 0, scratch_bytes = 0, frontier_peak_bytes = 0;
    /* ---- PER-STAGE COST OF ONE DIVISION (section 21) ----------------------------------------
       t_generic and t_copy overlapped (both covered the whole level loop), so they could not say
       WHERE a 5.0 ms division spends its time -- and that is the number item 2 has to attack.
       These add up inside s5_divmod_one: the two rev-packs, the Newton doubling chain (the only
       part that is inherently per-node), the q*B multiply and the final subtraction, plus the
       launch and division counts of each branch. */
    double t_revpack = 0.0, t_newton = 0.0, t_qb = 0.0, t_sub = 0.0;
    /* the inside of one S5 multiply (section 22): the probe's own per-call timers, accumulated */
    double t_mul_fwd = 0.0, t_mul_inv = 0.0, t_mul_slot = 0.0, t_mul_hpack = 0.0, t_mul_h2d = 0.0;
    double t_mul_check = 0.0, t_mul_check_d2h = 0.0, t_mul_check_kernel = 0.0;
    double t_mul_hplan = 0.0, t_mul_hopcopy = 0.0, t_mul_all = 0.0;
    unsigned long long mul_first_shape = 0;
    unsigned long long n_div = 0, n_horner = 0, n_copy = 0, n_launch_horner = 0, n_launch_div = 0;
    unsigned long long newton_steps = 0, newton_muls = 0;
};

/* the F tree, flattened per chunk: entry[level] = the first code of that level, off[], sz[] = the
   coefficient offset/size per code.  `dA` is the device copy the kernels read. */
struct S5Forest {
    std::vector<unsigned long long> off, sz;
    std::vector<size_t> entry;
    unsigned long long *dA = nullptr;
    size_t words = 0;
    ~S5Forest() { if (dA) cudaFree(dA); }
};

struct S5Dev {
    PolyLayer *L = nullptr;
    S4Reduce *red = nullptr;
    unsigned long long *dout = nullptr, *dvals = nullptr;
    size_t dout_cap = 0, dvals_cap = 0;
    /* the packing pool of s5_mul_batch (the NTT copies the operands into the arena) */
    unsigned long long *scratch = nullptr;
    size_t scratch_words = 0, scratch_used = 0;
    /* the reduction output of the single-slice multiplies inside s5_divmod_one */
    unsigned long long *mulout = nullptr;
    size_t mulout_cap = 0;
    /* the per-node working rows of s5_divmod_one (reset for every node) */
    unsigned long long *pool = nullptr;
    size_t pool_words = 0, pool_used = 0;
    /* the forest's coefficient offset table, on the device (the Horner kernel reads it) */
    unsigned long long *dfoff = nullptr;
    size_t dfoff_cap = 0;
    /* the per-node linear verdict */
    unsigned *dlin = nullptr;
    size_t dlin_cap = 0;
    /* R^2 mod N, the plain -> Montgomery image constant (section 44) */
    std::vector<unsigned long long> hR2;
    unsigned long long *dR2 = nullptr;
    /* the device-to-host diagnostic buffer of the reduction */
    void *dbad = nullptr;
    /* per-shape cache of the reduction parameters, keyed by (slot_bits, slot_words, bpw) */
    std::vector<S4Reduce::Shape *> rs;
    ~S5Dev()
    {
        if (dout) cudaFree(dout);
        if (dvals) cudaFree(dvals);
        if (scratch) cudaFree(scratch);
        if (mulout) cudaFree(mulout);
        if (pool) cudaFree(pool);
        if (dfoff) cudaFree(dfoff);
        if (dlin) cudaFree(dlin);
        if (dbad) cudaFree(dbad);
    }
    unsigned long long *alloc(const char *what, size_t words)
    {
        if (scratch_used + words > scratch_words) {
            std::fprintf(stderr, "%s: FATAL: the S5 pack pool is exhausted (%s wants %llu "
                                 "words, %llu of %llu free)\n", NTT_PROBE_NAME, what,
                         (unsigned long long)words,
                         (unsigned long long)(scratch_words - scratch_used),
                         (unsigned long long)scratch_words);
            std::exit(3);
        }
        unsigned long long *p = scratch + scratch_used;
        scratch_used += words;
        return p;
    }
    void reset() { scratch_used = 0; }
    /* the per-node pool (s5_divmod_one's rows): reset once per node, so a whole level allocates
       at most the peak of one node */
    unsigned long long *palloc(size_t words)
    {
        if (pool_used + words > pool_words) {
            std::fprintf(stderr, "%s: FATAL: the S5 node pool is exhausted (%llu + %llu > %llu "
                                 "words)\n", NTT_PROBE_NAME, (unsigned long long)pool_used,
                         (unsigned long long)words, (unsigned long long)pool_words);
            std::exit(3);
        }
        unsigned long long *p = pool + pool_used;
        pool_used += words;
        return p;
    }
    void preset() { pool_used = 0; }
    /* the reduction output of a single-slice multiply (the multiply hands it back through the
       hook, so it must be a stable buffer the caller can copy out of) */
    unsigned long long *fitout(size_t words)
    {
        if (words > mulout_cap) {
            if (mulout) { cudaFree(mulout); mulout = nullptr; mulout_cap = 0; }
            CK(cudaMalloc(&mulout, words * sizeof(unsigned long long)));
            mulout_cap = words;
        }
        return mulout;
    }
    unsigned long long *fitval(size_t words)
    {
        if (words > (size_t)dvals_cap) {
            if (dvals) { cudaFree(dvals); dvals = nullptr; dvals_cap = 0; }
            CK(cudaMalloc(&dvals, words * sizeof(unsigned long long)));
            dvals_cap = words;
        }
        return dvals;
    }
    unsigned long long *fitA(const char *what, size_t m)
    {
        if (m > dout_cap) {
            if (dout) { cudaFree(dout); dout = nullptr; dout_cap = 0; }
            CK(cudaMalloc(&dout, m * sizeof(unsigned long long)));
            dout_cap = m;
        }
        (void)what;
        return dout;
    }
    unsigned long long *fito(const char *what, size_t m) { return fitA(what, m); }
};

/* the reduction parameters for a shape, built (and selftested against GMP) once per shape */
static S4Reduce::Shape *s5_reduce_shape(S5Dev &D, unsigned long long slot_bits,
                                        unsigned long long slot_words, int bpw)
{
    for (S4Reduce::Shape *s : D.rs)
        if (s->slot_bits == slot_bits && s->slot_words == slot_words && s->bpw == bpw) return s;
    S4Reduce::Shape *S = s4_shape_init(*D.red, /*P=*/0, slot_bits, slot_words * (unsigned long long)bpw,
                                       slot_words, bpw);
    if (s4_reduce_selftest(*D.red, S)) {
        std::fprintf(stderr, "%s: FATAL: the device reduction disagrees with GMP on the S5 shape "
                             "slot_bits=%llu slot_words=%llu bpw=%d -- refusing to continue\n",
                     NTT_PROBE_NAME, slot_bits, slot_words, bpw);
        std::exit(3);
    }
    D.rs.push_back(S);
    return S;
}

/* ---- kernels ------------------------------------------------------------------------ */

/* one thread per (node, coefficient): extract `S` bits out of the coefficient-major source and
   write them as bpw-bit digits of slice `s` at digit `i * (slot_stride/bpw)` plus the caller's
   digit offset.  NO shared staging and NO atomicOr beyond the digit sharing inside one
   coefficient: a coefficient owns its slot exclusively, and a whole-slice staging buffer would be
   P*slot_words digits (~2 MB at the real shape) per block.
 *
 * `slot_stride` MUST be a whole number of bpw digits (the shape planner guarantees it: the stride
 * is slot_words*bpw).  The offset is expressed in DIGITS for the same reason -- a bit offset that
 * is not a multiple of bpw would put two chunks in one digit and break the "every digit < 2^bpw"
 * premise of the exactness criterion.
 *
 * MEASURED DEFECT OF THIS PATH AT THE S5 STRIDE (open, see the S5Shape comment): with
 * slot_stride = slot_bits (the fix-A layout) the reduction reads the window's TOP nw words and
 * returns 0 unless the window value also spans limbs [L, L+nw) of the base-2^64 conversion, which
 * for the frozen vector's 259-bit window (nlimb=5, nw=3, L=3) it does not.  The layout below is
 * therefore only consistent with the stride the reduction was written for
 * (slot_words*bpw on both sides), which is why the S5 descent stays opt-in. */
__global__ void s5_pack_kernel(const unsigned long long *src, unsigned long long src_off,
                               int S, int bpw, unsigned long long slot_stride,
                               unsigned long long N, unsigned long long ds, unsigned long long ma,
                               int W, unsigned long long off_digits, unsigned long long zero_words,
                               unsigned long long *dst)
{
    const unsigned long long gid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (gid >= ds * ma) return;
    const unsigned long long s = gid / ma, i = gid - s * ma;
    const unsigned long long bit0 =
        s * N * 64ull +
        (off_digits + i * (slot_stride / (unsigned long long)bpw)) * (unsigned long long)bpw;
    const unsigned long long word0 = bit0 / 64ull;
    const int sh0 = (int)(bit0 % 64ull);
    if (zero_words) dst[s * N + word0] = 0ull;
    const unsigned long long *c = src + src_off + s * ma * (unsigned long long)W +
                                  i * (unsigned long long)W;
    /* =====================================================================================
     * THE PACKER, CORRECTED FOR THE SECOND TIME -- AND THE FIRST TIME RIGHT (section 50).
     *
     * The digit array holds ONE DIGIT PER 64-BIT WORD.  That is the convention of every consumer:
     * `s4_reduce_kernel` reads `digits[s*n + k*slot_words + j]` and calls it digit j,
     * `carry_residual_kernel` tests each WORD against 2^bpw, the carry canonicalises each WORD,
     * and `s5_reddump_row` rebuilds the window as `sum d0[j] * 2^(bpw*j)`.  A digit is stored as a
     * value, not as `bpw` bits inside a word.
     *
     * Two wrong versions preceded this one:
     *   (1) the original capped the loop at two digits and took them as `(v >> kp) & mask`, so a
     *       129-bit coefficient lost everything above bit 2*bpw -- canonical, but truncated;
     *   (2) the first correction took the digits correctly but wrote them BIT-PACKED (`d << (ab&63)`
     *       across word boundaries).  That is a different data structure: the words no longer hold
     *       digits, so the array the reduction reads is not canonical and cannot be made so by any
     *       number of carry rounds -- measured as `s5_reddump_canon: max_digit=2715379` against a
     *       limit of 127, unchanged for 8/12/16/20 rounds (section 49).
     * This version keeps (2)'s digit extraction and restores the one-digit-per-word layout.  The
     * coefficient owns its own `ndig` words, so there are no atomics and no straddling, and the
     * store is bounded by `off_digits + (i+1)*slot_words <= N`, which `s5_mul_batch` asserts.
     * ===================================================================================== */
    const unsigned long long sw_dig = slot_stride / (unsigned long long)bpw;   /* words per slot */
    unsigned long long ndig = ((unsigned long long)S + (unsigned long long)bpw - 1ull) /
                              (unsigned long long)bpw;
    if (ndig > sw_dig) ndig = sw_dig;
    const unsigned long long mask = (bpw >= 64) ? ~0ull : ((1ull << bpw) - 1ull);
    unsigned long long *wrow = dst + s * N + off_digits + i * sw_dig;
    for (unsigned long long j = 0; j < ndig; ++j) {
        const unsigned long long bit = j * (unsigned long long)bpw;
        const unsigned long long wi = bit >> 6;
        if (wi >= (unsigned long long)W) break;
        const int sh = (int)(bit & 63ull);
        unsigned long long d = c[wi] >> sh;
        if (sh && (wi + 1) < (unsigned long long)W) d |= c[wi + 1] << (64 - sh);
        wrow[j] = d & mask;
    }
}

/* the "is this divisor linear?" verdict, ON THE DEVICE: words [1,degn] of the divisor must be
   0 and word degn must be 1 (monic).  Writes 1 (linear) or 0 per node. */
__global__ void s5_check_linear_kernel(const unsigned long long *src, unsigned long long src_off,
                                       unsigned long long ds, unsigned long long stride,
                                       int W, int degn, unsigned *out)
{
    const unsigned long long gid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (gid >= ds) return;
    const unsigned long long *c = src + src_off + gid * stride * (unsigned long long)W;
    unsigned ok = (W * (unsigned long long)degn < stride) ? 1u : 0u;
    unsigned long long acc = 0;
    for (int j = 0; j < W; ++j) acc |= c[degn * W + j];
    if (acc != 1ull) ok = 0u;
    acc = 0;
    for (int j = 1; j < W; ++j) acc |= c[j];
    if (acc != 0ull) ok = 0u;
    out[gid] = ok;
}

/* Horner at the two children of one node: out_s = A(x_{2*code+s}) with x = -b0 of the leaf,
   evaluated in Montgomery form and folded back with mone (the plain-domain 1).  One thread per
   child; the evaluation point is read from the FOREST (the leaf's own constant term), so the
   kernel is given the parent's code, never a group key.
   BOTH CHILDREN DIVIDE THE SAME PARENT ROW: the source is `src + src_off` for s = 0 and for
   s = 1.  The first version scaled it by the child index (`s * sa * W`), which made the second
   child of every node evaluate a NEIGHBOURING node's polynomial instead of its own parent's
   (section 44). */
__global__ void s5_eval_linear_kernel(const unsigned long long *src, unsigned long long src_off,
                                      unsigned long long sa, int W,
                                      const unsigned long long *fA, const unsigned long long *foff,
                                      unsigned long long child, const unsigned long long *n,
                                      unsigned long long ninv, int nw,
                                      const unsigned long long *R2,
                                      unsigned long long *dst, unsigned long long dstride)
{
    /* ONE CHILD PER CALL, and the child's own code is an ARGUMENT.  The first version computed
       both children from the parent's code (`cc = 2*code + threadIdx.x`) while the CALLER already
       looped over the two children and advanced the destination row for each: the second call
       wrote the first call's second row again one row further on, so every node ended up with
       one child's value duplicated and the other's never computed -- and when one child was the
       padding subtree (degree 0) it wrote a row that did not exist at all (section 44). */
    const unsigned long long *c = src + src_off;
    /* ---- THE EVALUATION POINT AND ITS DOMAIN (section 44) --------------------------------
       The leaf is [ -root, 1 ] = X - root, so the point the remainder is taken at is the ROOT
       itself: x = -b0 mod N (the host's leaf_mod_linear_batch says the same thing in one line).
       The first version evaluated at `root`, i.e. at MINUS the baby point -- a wrong value that
       no amount of downstream checking could have made right.
       The point then has to be a MONTGOMERY IMAGE, because the coefficients it multiplies come
       from the plain-domain frontier and `s2g_mont_mul(h, xR) = h*x` keeps them plain: one
       multiplication by R^2 is the only conversion needed, and NO conversion at the end (the
       first version converted with `one = 1`, which is a multiplication by R^-1 applied to a
       value that was already plain). */
    unsigned long long x[128], h[128], zero[128], r2[128];
    for (int j = 0; j < nw; ++j) { zero[j] = 0ull; r2[j] = R2[j]; }
    s2g_submod<128>(x, zero, fA + foff[child], n, nw);       /* x = -root = the baby point */
    s2g_mont_mul<128>(x, x, r2, n, ninv, nw);                /* x -> x*R */
    bool started = false;
    for (unsigned long long i = sa; i-- > 0;) {
        if (!started) {
            for (int j = 0; j < nw; ++j) h[j] = c[i * (unsigned long long)W + j];
            started = true;
        } else {
            s2g_mont_mul<128>(h, h, x, n, ninv, nw);         /* h = h*x, both plain */
            const unsigned long long *a = c + i * (unsigned long long)W;
            unsigned long long carry = 0;
            for (int j = 0; j < nw; ++j) {                    /* h += a  (mod N, exact carry) */
                const unsigned long long t = h[j] + a[j] + carry;
                carry = (t < h[j]) ? 1ull : ((carry && t == h[j]) ? 1ull : 0ull);
                h[j] = t;
            }
            if (carry || s2g_ge_n(h, n, nw)) s2g_sub_n(h, n, nw);
        }
    }
    if (!started) for (int j = 0; j < nw; ++j) h[j] = 0ull;
    for (int j = 0; j < nw; ++j)
        dst[(unsigned long long)j] = (j < nw) ? h[j] : 0ull;
}

/* dst[i] = A[i] - B[i mod lb] mod N for i < la, else 0 (the reference's cp_coeff_sub, ON THE
   DEVICE, in the PLAIN domain).  The first version computed `s2g_mont_mul(A[i], B[i mod lb])`
   and stored THAT -- a Montgomery PRODUCT where the whole operation is a SUBTRACTION, with no
   subtraction anywhere in the kernel (section 44).  The domain is plain on both sides: the
   frontier rows come from the host or from the S4 reduction, and s5_mul_batch's output is the
   reduced plain product, so nothing here needs R at all. */
__global__ void s5_sub_kernel(const unsigned long long *A, unsigned long long a_off, int la,
                              const unsigned long long *B, unsigned long long b_off, int lb,
                              int rows, const unsigned long long *n, unsigned long long ninv,
                              int nw, unsigned long long *dst, unsigned long long dstride)
{
    const unsigned long long gid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (gid >= (unsigned long long)rows * (unsigned long long)nw) return;
    const int i = (int)(gid / (unsigned long long)nw);
    (void)ninv;
    unsigned long long r[128];
    if (i >= la) {
        for (int j = 0; j < nw; ++j) r[j] = 0ull;
    } else {
        s2g_submod<128>(r, A + a_off + (size_t)i * nw, B + b_off + (size_t)(i % lb) * nw, n, nw);
    }
    for (int j = 0; j < nw; ++j) dst[(size_t)i * dstride + j] = r[j];
}

/* h = 2 - a, PER COEFFICIENT -- and the 2 belongs to the CONSTANT COEFFICIENT ONLY (section 53).
   The host reference (`cp_inv_series`) is explicit: `h[0] = 2` and then `h -= ag`, so
   `h[0] = 2 - ag[0]` while `h[i] = -ag[i]` for every i > 0.  Subtracting 2 from every coefficient
   instead (the first version) makes the Newton step compute a different series: measured through
   `NTT_S5_DIVDUMP=1`, the inverse's constant term was right and its coefficient 1 was
   `2 - rb1` instead of `-rb1`, i.e. `s5_divstage: g=1 ... g0=1`. */
__global__ void s5_two_minus_kernel(const unsigned long long *a, unsigned long long *out,
                                    unsigned long long total, const unsigned long long *n,
                                    unsigned long long ninv, int nw)
{
    const unsigned long long gid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (gid >= total) return;
    (void)ninv;
    unsigned long long lhs[128], r[128];
    for (int j = 0; j < nw; ++j) lhs[j] = 0ull;
    if (gid == 0) lhs[0] = 2ull;                 /* only the constant coefficient carries the 2 */
    s2g_submod<128>(r, lhs, a + gid * (unsigned long long)nw, n, nw);
    for (int j = 0; j < nw; ++j) out[gid * (unsigned long long)nw + j] = r[j];
}

/* reverse the TOP `n` coefficients of a row of `src_len` coefficients: dst[i] = src[src_len-1-i],
   and 0 once i reaches the row (which is what zero-extends a short operand into a longer one).
   THE ROW LENGTH, NOT `n`, IS THE BASE OF THE REVERSAL, and that distinction is the whole bug
   of section 45: the first version used `j = n - 1 - i`, which reverses the FIRST n coefficients.
   For `rb` (n == src_len) and for `q` (n == src_len) the two readings coincide, so only the
   `ra` call -- the top k coefficients of a dividend with la > k of them -- was wrong, and it
   quietly fed the quotient chain the dividend's LOW coefficients instead of its high ones. */
#define S5_GRID(n) ((unsigned int)(((n) + 255) / 256)), 256
__global__ void s5_rev_pack_kernel(unsigned long long *dst, const unsigned long long *src,
                                   unsigned long long src_off, unsigned long long n,
                                   unsigned long long src_len, int W)
{
    const unsigned long long gid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (gid >= n) return;
    unsigned long long *d = dst + (size_t)gid * W;
    if (gid >= src_len) {
        for (int t = 0; t < W; ++t) d[t] = 0ull;
        return;
    }
    const unsigned long long j = src_len - 1 - gid;
    const unsigned long long *s = src + src_off + (size_t)j * W;
    for (int t = 0; t < W; ++t) d[t] = s[t];
}

/* zero-extend `len` coefficients of a row into `n` */
__global__ void s5_pad_low_kernel(unsigned long long *dst, const unsigned long long *src,
                                  unsigned long long n, unsigned long long len, int W)
{
    const unsigned long long gid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (gid >= n) return;
    const unsigned long long *s = src + (size_t)gid * W;
    unsigned long long *d = dst + (size_t)gid * W;
    for (int t = 0; t < W; ++t) d[t] = (gid < len) ? s[t] : 0ull;
}

/* g[0] = 1 (the Montgomery image of the constant series 1; every divisor reaching the Newton
   chain has a monic leading coefficient, so 1/a[0] = 1 in the plain domain), rest zero */
__global__ void s5_fill_kernel(unsigned long long *g, unsigned long long n, int W)
{
    const unsigned long long gid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (gid >= n) return;
    unsigned long long *d = g + (size_t)gid * W;
    for (int t = 0; t < W; ++t) d[t] = 0ull;
    if (gid == 0) d[0] = 1ull;
}

/* coefficient-wise copy between two device arrays (the degree fast path: H mod F_ci = H when
   deg H < deg F_ci) */
static void s5_copy_rows(const unsigned long long *src, size_t src_off, size_t dst_off, int rows,
                         int W, unsigned long long *dst, int cat)
{
    (void)cat;
    if (rows <= 0) return;
    CK(cudaMemcpy(dst + dst_off, src + src_off,
                  (size_t)rows * (size_t)W * sizeof(unsigned long long),
                  cudaMemcpyDeviceToDevice));
}

/* ---- the device driver ----------------------------------------------------------------- */

/* the reduction hook of the S5 pack: the digits of a canonical slot -> the plain residue.  This
   is the SAME kernel the S4 multiply uses (s4_reduce_kernel, dispatched through
   s4_launch_reduce), so the S5 path inherits its per-shape GMP selftest; the only thing that is
   new is the caller's stride (tight rows instead of the NTT's slot projection). */
struct S5RedHook {
    S5Dev *D = nullptr;
    unsigned long long *out = nullptr;
    unsigned long long out_stride = 0;         /* words per row of the destination */
    unsigned long long row0 = 0;               /* first destination row */
    unsigned long long w = 0;                  /* words per coefficient */
    /* The reduction hands back out_slots = 2P-1 coefficients per slice while the caller keeps
       only `want` of them.  Striding the destination by `want` makes consecutive slices overlap,
       and the overflow lands on the NEXT slice's first row -- measured as "1 of 31 slot windows
       have nonzero digits above slot_bits" on the frozen vector.  So the reduction writes into a
       tight scratch region of the pack pool (stride out_slots) and the wanted prefix is copied
       into the caller's rows afterwards. */
    unsigned long long *tmp = nullptr;         /* the scratch region, out_slots rows per slice */
    unsigned long long tmp_stride = 0;
};

static void s5_reduce_hook(void *ctx, const unsigned long long *digits, unsigned long long n,
                           int bpw, unsigned long long slot_words, unsigned long long slot_bits,
                           unsigned long long out_slots, unsigned long long nbatch,
                           unsigned long long *out, unsigned long long w)
{
    S5RedHook &H = *(S5RedHook *)ctx;
    S5Dev &D = *H.D;
    (void)out;
    S4Reduce::Shape *S = s5_reduce_shape(D, slot_bits, slot_words, bpw);
    if (w != (unsigned long long)D.red->w) {
        std::fprintf(stderr, "%s: FATAL: the S5 reduction was given w=%llu but the modulus has "
                             "%d words\n", NTT_PROBE_NAME, w, D.red->w);
        std::exit(3);
    }
    if (!D.dbad) CK(cudaMalloc(&D.dbad, sizeof(unsigned long long)));
    CK(cudaMemset(D.dbad, 0, sizeof(unsigned long long)));
    /* FORENSIC (NTT_S5_REDDUMP=1): the shape the reduction is about to run and the first digits of
       the stream, so "the reduce read the wrong window" and "the multiply returned the wrong
       digits" can be separated by inspection.  Off by default (it costs a copy). */
    if (std::getenv("NTT_S5_REDDUMP") && *std::getenv("NTT_S5_REDDUMP")
        && std::atoi(std::getenv("NTT_S5_REDDUMP")) != 0) {
        std::vector<unsigned long long> d0(64, 0ull);
        CK(cudaMemcpy(d0.data(), digits, d0.size() * sizeof(unsigned long long),
                      cudaMemcpyDeviceToHost));
        std::fprintf(stderr, "s5_reddump: n=%llu nbatch=%llu bpw=%d slot_words=%llu "
                             "slot_bits=%llu out_slots=%llu redL=%llu words_per_coeff=%d "
                             "nw=%d digits[0..7]=", n, nbatch, bpw, slot_words, slot_bits, out_slots,
                     (unsigned long long)S->L, D.red->w, D.red->nw);
        for (int i = 0; i < 8; ++i) std::fprintf(stderr, "%llu,", d0[(size_t)i]);
        std::fprintf(stderr, "\n");
        if (slot_words <= d0.size()) {
            mpz_t v, two;
            mpz_inits(v, two, nullptr);
            mpz_set_ui(two, 1);
            mpz_mul_2exp(two, two, (unsigned)bpw);
            mpz_set_ui(v, 0);
            for (unsigned long long j = slot_words; j-- > 0;) {
                mpz_mul(v, v, two);
                mpz_add_u64(v, d0[(size_t)j]);
            }
            char *s1 = mpz_get_str(nullptr, 16, v);
            mpz_t want;
            mpz_init(want);
            s4_gmp_reduce(want, d0.data(), slot_words, bpw, D.red->N);
            char *s2 = mpz_get_str(nullptr, 16, want);
            std::fprintf(stderr, "s5_reddump: window0=%s gmp_reduce(window0)=%s\n", s1, s2);
            /* THE WHOLE FIRST WINDOW, digit by digit, so the limb conversion can be reproduced by
               hand: the number above is exactly sum_j d0[j]*2^(bpw*j). */
            std::fprintf(stderr, "s5_reddump_window: bpw=%d slot_words=%llu digits=", bpw, slot_words);
            unsigned long long nz = 0;
            for (unsigned long long j = 0; j < slot_words && j < 64; ++j) {
                std::fprintf(stderr, "%llu%s", d0[(size_t)j], (j + 1 == slot_words) ? "" : ",");
                if (d0[(size_t)j]) nz = j + 1;
            }
            unsigned long long topbits = 0;
            if (nz) {
                unsigned long long tv = d0[(size_t)(nz - 1)];
                while (tv) { ++topbits; tv >>= 1; }
            }
            std::fprintf(stderr, " highest_nonzero_digit=%llu value_bits_in_window=%llu\n", nz,
                         nz ? (nz - 1) * (unsigned long long)bpw + topbits : 0ull);
            void (*ff)(void *, size_t) = nullptr;
            mp_get_memory_functions(nullptr, nullptr, &ff);
            ff(s1, std::strlen(s1) + 1);
            ff(s2, std::strlen(s2) + 1);
            mpz_clears(v, two, want, nullptr);
        }
    }
    const double t0 = now_s();
    /* ---- IS THE WINDOW THE REDUCTION IS ABOUT TO READ CANONICAL? (section 49) --------------
       The kernel's digit-to-limb conversion is `acc |= v << nacc`: an OR, which silently drops
       the high bits of any digit wider than bpw.  The host's GMP reduction treats the digits as a
       number and keeps them, so ONE non-canonical digit makes the two sides disagree in a way no
       other check here can see -- and that is the whole of the S5 descent's remaining mystery
       (measured: digit 10 of the failing window was 2715379 against a bpw=7 limit of 127).
       Checked once per shape, where the window is already being read for the GMP oracle. */
    /* On the ORACLE's cadence (not just once per shape): the first version checked only
       `S->calls == 0`, so a window that went non-canonical later would sail past it. */
    if ((S->calls <= 1 || (S->calls % g_s4_check_every) == 0) && slot_words <= 512) {
        std::vector<unsigned long long> w0((size_t)slot_words, 0ull);
        CK(cudaMemcpy(w0.data(), digits, w0.size() * sizeof(unsigned long long),
                      cudaMemcpyDeviceToHost));
        unsigned long long mx = 0, mj = 0;
        for (unsigned long long j = 0; j < slot_words; ++j)
            if (w0[(size_t)j] > mx) { mx = w0[(size_t)j]; mj = j; }
        if (mx >= (1ull << bpw)) {
            std::fprintf(stderr, "%s: FATAL: the S5 reduction was handed a NON-CANONICAL window: "
                                 "digit %llu of the first coefficient is %llu but bpw=%d (limit "
                                 "%llu).  The kernel's `acc |= v << nacc` would drop its high "
                                 "bits while GMP keeps them, so the two cannot agree -- fix the "
                                 "carry stage (or its round count) before reducing here\n",
                         NTT_PROBE_NAME, mj, mx, bpw, (1ull << bpw) - 1ull);
            std::exit(3);
        }
        S->canon_checked = 1;
    }
    /* FORENSIC (NTT_S5_REDDUMP=1): the KERNEL's own view of the first coefficients -- the
       assembled limbs t[], the returned r[] and the digits it actually read.  The host can dump
       the digit buffer and the output rows, but neither says what the kernel saw: if they
       disagree, this is the only place that shows it (section 30). */
    unsigned long long *ddbg = nullptr;
    const bool reddump = (std::getenv("NTT_S5_REDDUMP") && *std::getenv("NTT_S5_REDDUMP")
                          && std::atoi(std::getenv("NTT_S5_REDDUMP")) != 0);
    if (reddump) {
        CK(cudaMalloc(&ddbg, 4 * 24 * sizeof(unsigned long long)));
        CK(cudaMemset(ddbg, 0, 4 * 24 * sizeof(unsigned long long)));
    }
    S2G_DISPATCH(D.red->nw, s4_launch_reduce, (int)D.red->nw, S->L, nbatch, out_slots,
                 out_slots * nbatch, digits, n, bpw, slot_words, D.red->dn, D.red->ninv, S->dy, w,
                 H.tmp, /*slot_bits=*/S->slot_stride, (unsigned long long *)D.dbad, ddbg);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    S->t_reduce += now_s() - t0;
    if (ddbg != nullptr) {
        std::vector<unsigned long long> dbg(4 * 24, 0ull);
        CK(cudaMemcpy(dbg.data(), ddbg, dbg.size() * sizeof(unsigned long long),
                      cudaMemcpyDeviceToHost));
        for (int g = 0; g < 4; ++g) {
            const unsigned long long *q = &dbg[(size_t)g * 24];
            std::fprintf(stderr, "s5_kernel_view: gid=%d L=%llu slot_words=%llu bpw=%llu "
                                 "t=%llx,%llx,%llx,%llx,%llx,%llx r=%llx,%llx,%llx "
                                 "u=%llx,%llx,%llx dy=%llx,%llx,%llx ninv=%llx nw=%llu "
                                 "d[0..2]=%llx,%llx,%llx d[last]=%llx\n",
                         g, q[9], q[10], q[15], q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7],
                         q[8], q[16], q[17], q[18], q[19], q[20], q[21], q[22], q[23], q[11],
                         q[12], q[13], q[14]);
        }
        /* `ddbg` is KEPT until after the row dump, so the same coefficient can be printed from
           both the kernel's registers and the output buffer on adjacent lines (section 48). */
    }
    /* FORENSIC (NTT_S5_REDDUMP=1): what the reduction actually wrote, row by row, next to the
       mathematical answer for the SAME window.  One row is not enough: a shape whose output rows
       are shifted (or whose digit base is off by one slot) shows ZERO in row 0 while the correct
       value sits in row 1, and that is indistinguishable from an all-zero window unless the
       neighbouring rows and their own windows are printed together (section 30). */
    if (std::getenv("NTT_S5_REDDUMP") && *std::getenv("NTT_S5_REDDUMP")
        && std::atoi(std::getenv("NTT_S5_REDDUMP")) != 0) {
        const size_t rw = (size_t)(D.red->w ? D.red->w : 1);
        const unsigned long long show = (out_slots < 3ull) ? out_slots : 3ull;
        std::vector<unsigned long long> rows((size_t)show * rw, 0ull);
        CK(cudaMemcpy(rows.data(), H.tmp, rows.size() * sizeof(unsigned long long),
                      cudaMemcpyDeviceToHost));
        std::vector<unsigned long long> d0((size_t)slot_words * (size_t)show, 0ull);
        CK(cudaMemcpy(d0.data(), digits, d0.size() * sizeof(unsigned long long),
                      cudaMemcpyDeviceToHost));
        std::fprintf(stderr, "s5_reddump_out: L=%llu nw=%d bpw=%d slot_words=%llu out_slots=%llu "
                             "nbatch=%llu n=%llu w=%llu tmp_stride=%llu\n",
                     (unsigned long long)S->L, D.red->nw, bpw, slot_words, out_slots, nbatch, n, w,
                     H.tmp_stride);
        {   /* S->hy (host) vs S->dy (device): if they differ, the device copy was clobbered
               after s4_shape_init wrote it, which is a use-after-free or an over-broad memset */
            std::vector<unsigned long long> dyv((size_t)D.red->nw, 0ull);
            CK(cudaMemcpy(dyv.data(), S->dy, dyv.size() * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
            std::fprintf(stderr, "s5_reddump_dy: S->dy_ptr=%p host_hy=", (void *)S->dy);
            for (int i = 0; i < D.red->nw; ++i) std::fprintf(stderr, "%llx,", S->hy[(size_t)i]);
            std::fprintf(stderr, " device_dy=");
            for (int i = 0; i < D.red->nw; ++i) std::fprintf(stderr, "%llx,", dyv[(size_t)i]);
            std::fprintf(stderr, "\n");
        }
        mpz_t wv, want, got, two;
        mpz_inits(wv, want, got, two, nullptr);
        for (unsigned long long k = 0; k < show; ++k) {
            mpz_set_ui(wv, 0);
            mpz_set_ui(two, 1);
            mpz_mul_2exp(two, two, (unsigned)bpw);
            for (unsigned long long j = slot_words; j-- > 0;) {
                mpz_mul(wv, wv, two);
                mpz_add_u64(wv, d0[(size_t)(k * slot_words + j)]);
            }
            s4_gmp_reduce(want, &d0[(size_t)k * slot_words], slot_words, bpw, D.red->N);
            mpz_import(got, rw, -1, 8, 0, 0, &rows[(size_t)k * rw]);
            char *s1 = mpz_get_str(nullptr, 16, want);
            char *s2 = mpz_get_str(nullptr, 16, got);
            char *s3 = mpz_get_str(nullptr, 16, wv);
            std::fprintf(stderr, "s5_reddump_row: k=%llu window=%s gmp=%s device=%s %s\n", k, s3,
                         s1, s2, (mpz_cmp(want, got) == 0) ? "MATCH" : "DIFFER");
            /* ---- ARE THE WINDOW'S DIGITS CANONICAL? (section 49) ---------------------------
               The kernel's digit-to-limb conversion is `acc |= v << nacc` -- an OR, which silently
               DROPS the high bits of a digit that is wider than bpw.  The host's GMP reduction
               treats the digits as a number (`sum d[j] 2^(bpw j)`) and keeps them.  So a single
               non-canonical digit makes the two sides disagree in exactly the way observed, and
               the kernel's own canonicity check only looks ABOVE slot_bits, never per digit. */
            {
                unsigned long long mx = 0, mj = 0;
                for (unsigned long long j = 0; j < slot_words; ++j) {
                    const unsigned long long v = d0[(size_t)(k * slot_words + j)];
                    if (v > mx) { mx = v; mj = j; }
                }
                std::printf("s5_reddump_canon: k=%llu bpw=%d max_digit=%llu at_j=%llu limit=%llu "
                            "%s\n", k, bpw, mx, mj, (1ull << bpw) - 1ull,
                            (mx < (1ull << bpw)) ? "CANONICAL" : "NON_CANONICAL_DIGIT");
            }
            /* THE SAME COORDINATE, FROM THE KERNEL'S OWN REGISTERS (section 48): `device` above
               comes from the output buffer, and this comes from the kernel's debug array at the
               SAME gid.  Printing both on one line is what makes "the kernel's u and the row I am
               comparing are the same coefficient" checkable instead of assumed -- the whole
               paradox of section 47.3 rests on that correspondence. */
            if (ddbg != nullptr) {
                unsigned long long ku[3] = {0, 0, 0}, kck = 0;
                CK(cudaMemcpy(ku, ddbg + (size_t)k * 24 + 16, sizeof(ku), cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(&kck, ddbg + (size_t)k * 24 + 23, sizeof(kck),
                              cudaMemcpyDeviceToHost));
                unsigned long long hck = 0;
                for (unsigned long long j = 0; j < slot_words; ++j)
                    hck += d0[(size_t)(k * slot_words + j)] * (j + 1ull);
                std::fprintf(stderr, "s5_reddump_samecoef: k=%llu kernel_u=%llx,%llx,%llx "
                                     "kernel_ck=%llx host_ck=%llx %s\n", k,
                             (unsigned long long)ku[0], (unsigned long long)ku[1],
                             (unsigned long long)ku[2], (unsigned long long)kck,
                             (unsigned long long)hck, (kck == hck) ? "SAME_INPUTS" : "DIFF_INPUTS");
            }
            /* THE REDC INVARIANT, FROM GMP (section 47): after L elimination steps the kernel's
               r must be V * 2^-(64L) mod N, and `s5_kernel_view` prints the r it actually holds.
               Printing the expectation here turns "the reduction is wrong" into "the elimination
               is wrong" or "the tail mont_mul is wrong" without a second run. */
            {
                mpz_t rexp, t;
                mpz_inits(rexp, t, nullptr);
                mpz_set_ui(t, 1);
                mpz_mul_2exp(t, t, (mp_bitcnt_t)(64 * S->L));
                mpz_invert(t, t, D.red->N);
                mpz_mul(rexp, wv, t);
                mpz_mod(rexp, rexp, D.red->N);
                char *sr = mpz_get_str(nullptr, 16, rexp);
                std::fprintf(stderr, "s5_reddump_rexp: k=%llu L=%llu rexp=%s\n", k,
                             (unsigned long long)S->L, sr);
                void (*ff)(void *, size_t) = nullptr;
                mp_get_memory_functions(nullptr, nullptr, &ff);
                ff(sr, std::strlen(sr) + 1);
                mpz_clears(rexp, t, nullptr);
            }
            /* ---- THE SAME WINDOW, THROUGH A FRESH LAUNCH OF THE SAME KERNEL (section 49) ----
               Everything about the in-line call checks out -- the digits by checksum, every
               parameter by construction -- yet it disagrees with GMP while the selftest agrees on
               the same value.  So take THIS window's digits, put them through a standalone launch
               with the selftest's geometry, right here, and print three numbers side by side:
               GMP, the in-line device output, and the standalone one.  A standalone run that is
               RIGHT makes the launch the difference; one that is also WRONG makes the digits the
               difference -- and then the weak checksum is what lied. */
            if (k == 0) {
                std::vector<unsigned long long> w1((size_t)slot_words, 0ull);
                CK(cudaMemcpy(w1.data(), digits, w1.size() * sizeof(unsigned long long),
                              cudaMemcpyDeviceToHost));
                unsigned long long *dw = nullptr, *dow = nullptr;
                CK(cudaMalloc(&dw, w1.size() * sizeof(unsigned long long)));
                CK(cudaMalloc(&dow, (size_t)D.red->w * sizeof(unsigned long long)));
                CK(cudaMemcpy(dw, w1.data(), w1.size() * sizeof(unsigned long long),
                              cudaMemcpyHostToDevice));
                S2G_DISPATCH(D.red->nw, s4_launch_reduce, (int)D.red->nw, S->L, 1ull, 1ull, 1ull,
                             dw, slot_words, bpw, slot_words, D.red->dn, D.red->ninv, S->dy,
                             (unsigned long long)D.red->w, dow, S->slot_bits,
                             (unsigned long long *)nullptr, (unsigned long long *)nullptr);
                CK(cudaGetLastError());
                CK(cudaDeviceSynchronize());
                std::vector<unsigned long long> o1((size_t)D.red->w, 0ull);
                CK(cudaMemcpy(o1.data(), dow, o1.size() * sizeof(unsigned long long),
                              cudaMemcpyDeviceToHost));
                mpz_t st;
                mpz_init(st);
                mpz_import(st, (size_t)D.red->w, -1, 8, 0, 0, o1.data());
                char *ss = mpz_get_str(nullptr, 16, st);
                std::fprintf(stderr, "s5_reddump_standalone: k=0 n=%llu out_slots=1 gmp=%s "
                                     "inline=%s standalone=%s %s\n", slot_words, s1, s2, ss,
                             (mpz_cmp(want, st) == 0) ? "STANDALONE_MATCHES_GMP"
                                                      : "STANDALONE_ALSO_WRONG");
                void (*ff2)(void *, size_t) = nullptr;
                mp_get_memory_functions(nullptr, nullptr, &ff2);
                ff2(ss, std::strlen(ss) + 1);
                mpz_clear(st);
                cudaFree(dw);
                cudaFree(dow);
            }
            void (*ff)(void *, size_t) = nullptr;
            mp_get_memory_functions(nullptr, nullptr, &ff);
            ff(s1, std::strlen(s1) + 1);
            ff(s2, std::strlen(s2) + 1);
            ff(s3, std::strlen(s3) + 1);
        }
        mpz_clears(wv, want, got, two, nullptr);
    }
    if (ddbg) { cudaFree(ddbg); ddbg = nullptr; }
    {
        const unsigned long long want = H.out_stride / (H.w ? H.w : 1ull);
        if (want && want != out_slots) {
            for (unsigned long long s = 0; s < nbatch; ++s) {
                CK(cudaMemcpy(H.out + (size_t)(H.row0 + s) * H.out_stride,
                              H.tmp + (size_t)s * out_slots * H.w,
                              (size_t)want * H.w * sizeof(unsigned long long),
                              cudaMemcpyDeviceToDevice));
            }
        } else {
            CK(cudaMemcpy(H.out + (size_t)H.row0 * H.out_stride, H.tmp,
                          (size_t)out_slots * nbatch * H.w * sizeof(unsigned long long),
                          cudaMemcpyDeviceToDevice));
        }
    }
    unsigned long long hbad = 0;
    CK(cudaMemcpy(&hbad, D.dbad, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    ++S->calls;
    S->coeffs += out_slots * nbatch;
    D.red->coeffs_total += out_slots * nbatch;
    if (hbad) {
        std::fprintf(stderr, "%s: FATAL: %llu of %llu S5 slot windows have nonzero digits above "
                             "slot_bits=%llu -- the packing is not canonical\n", NTT_PROBE_NAME,
                     hbad, out_slots * nbatch, slot_bits);
        /* FORENSIC (NTT_S5_DIGDUMP=1): which slot, and what the digit stream actually holds
           around it.  The digit buffer dies with this call, so this has to happen here. */
        const char *ed = std::getenv("NTT_S5_DIGDUMP");
        if (ed && *ed && std::atoi(ed) != 0) {
            std::vector<unsigned long long> dig((size_t)n * nbatch, 0ull);
            CK(cudaMemcpy(dig.data(), digits, dig.size() * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
            const unsigned long long top_bits = slot_bits - (slot_words - 1) * (unsigned long long)bpw;
            std::fprintf(stderr, "  digdump: n=%llu nbatch=%llu out_slots=%llu bpw=%d slot_words=%llu "
                                 "slot_bits=%llu top_bits=%llu\n", n, nbatch, out_slots, bpw,
                         slot_words, slot_bits, top_bits);
            /* the reduction's own output for the same slots */
            std::vector<unsigned long long> hout((size_t)out_slots * nbatch *
                                                 (D.red->w ? D.red->w : 1), 0ull);
            (void)hout;
            unsigned long long shown = 0;
            for (unsigned long long g = 0; g < out_slots * nbatch && shown < 4; ++g) {
                const unsigned long long s = g / out_slots, k = g - s * out_slots;
                const unsigned long long b = s * n + k * slot_words;
                if (b + slot_words > dig.size()) continue;
                if (top_bits < 64 && (dig[(size_t)(b + slot_words - 1)] >> top_bits) != 0) {
                    mpz_t v, pw, two29;
                    mpz_inits(v, pw, two29, nullptr);
                    mpz_set_ui(v, 0);
                    mpz_set_ui(two29, 1);
                    mpz_mul_2exp(two29, two29, (unsigned)bpw);
                    for (unsigned long long j = slot_words; j-- > 0;) {
                        mpz_mul(v, v, two29);
                        mpz_add_u64(v, dig[(size_t)(b + j)]);
                    }
                    std::fprintf(stderr, "  bad slot: slice=%llu k=%llu first_digit=%llu value_bits=%zu "
                                         "value_mod_N_hex=", s, k, b, mpz_sizeinbase(v, 2));
                    char *str = mpz_get_str(nullptr, 16, v);
                    std::fprintf(stderr, "%s\n", str);
                    void (*ff)(void *, size_t) = nullptr;
                    mp_get_memory_functions(nullptr, nullptr, &ff);
                    ff(str, std::strlen(str) + 1);
                    mpz_clears(v, pw, two29, nullptr);
                    ++shown;
                }
            }
        }
        std::exit(3);
    }
    /* THE IN-RUN ORACLE.  The reduction runs on every S5 call; its check against GMP runs the
       same way the S4 multiply's does (a full check for small shapes, a sample for big ones),
       so S5's claim "these coefficients equal GMP's" is measured, not asserted.  It reads the
       scratch region, which holds the reduction's own output verbatim. */
    if (g_s4_sample_limit > 0 && (S->calls <= 1 || (S->calls % g_s4_check_every) == 0))
        s4_check_reduced(*D.red, S, digits, n, out_slots, nbatch, H.tmp, g_s4_sample_limit);
}

/* THE S5 SLOT SHAPE, derived here rather than inherited.
 *
 * The host packer's layout is: coefficient i at bit i*slot_stride, stride = slot_words*bpw with
 * slot_words = ceil(slot_bits/bpw) and slot_bits = 2S + ceil(log2 P) -- enough room for the SUM
 * of two coefficient products.  Its claim is nwords*bpw >= P*slot_stride, and that is what makes
 * the packing non-aliasing.
 *
 * S5's operands are only ONE coefficient wide each, so the natural slot only needs ceil(S/bpw)
 * words; packing at the host's wider stride would need P*slot_stride > nwords*bpw and the top
 * coefficients would WRAP ONTO THE LOW ONES (measured: the first product coefficient came back
 * as 2^258, the wrapped coefficient 4).  The converse is also fatal: packing at the narrow stride
 * while the NTT reads the wide one puts each coefficient in the wrong place (measured: the packed
 * block held bits of its neighbours).
 *
 * So S5 derives BOTH from one bpw and requires them to be the SAME partition:
 *     slot_words = 2 * ceil(S/bpw)   ==   ceil((2S + ceil(log2 P)) / bpw)
 * i.e. the sum-slot is exactly two operand slots.  Then stride = slot_words*bpw, the operands fit
 * in nwords >= P*slot_words, and the reduction reads the same windows the packer wrote.
 * The exactness bound L*(2^bpw-1)^2 < p with L = P*slot_words is re-proved for the chosen bpw.
 *
 * ============================ THE STRIDE FIX (fix A) =====================================
 * The paragraph above is the OLD, half-finished derivation and it is why this path was gated off.
 * What actually has to hold is a statement about the REDUCER, and here it is, proved from the
 * arrays rather than asserted:
 *
 *   [A1] ONE WINDOW PER COEFFICIENT.  s4_reduce_kernel reads coefficient k out of the window of
 *        `slot_words` digits that starts at digit k*slot_words, and asserts that window's value is
 *        < 2^slot_bits (its guard).  THE WINDOW'S WIDTH IN BITS IS slot_words*bpw = stride, NOT
 *        slot_bits -- the reducer reads a fixed number of DIGITS, so its reach is fixed by the
 *        digit width, and the window it reads for k is exactly the bit range
 *                [k*stride, (k+1)*stride).
 *        The packer puts coefficient k at bit offset k*stride, S bits wide.  Therefore window k
 *        contains exactly the bits of coefficient k, and nothing else, whenever
 *                S <= stride:
 *        block k ends at k*stride + S <= (k+1)*stride, and coefficient k+1 begins at (k+1)*stride,
 *        so no window ever sees a neighbour's bits.  Equality (stride == slot_bits) is NOT what
 *        makes the reducer exact; the inequality is.  What stride must additionally satisfy is the
 *        GUARD: the coefficient's value has to fit under the asserted bound, which is the counting
 *        bound of [A2], c_k < 2^slot_bits <= 2^stride -- one condition, and it is the reason
 *        slot_bits exists as a separate quantity at all.
 *        This is what licenses ROUNDING THE SLOT UP (section 54): `stride = ceil(slot_bits/bpw)*bpw`
 *        costs at most one extra window per coefficient and changes nothing the reducer reads.  The
 *        old rule demanded `bpw | slot_bits` and scanned bpw DOWNWARDS, so when `2S + ceil(log2 P)`
 *        was prime the only divisor <= 62 was 1 and the shape silently collapsed to bpw = 1 --
 *        single-bit digits, a carry chain of length O(N), and a descent that appeared to need 128
 *        carry rounds.  (The earlier claim that "stride > slot_bits makes the windows overrun" was
 *        measured under the MISMATCHED old layout where the packer advanced by stride while the
 *        reducer's window advanced by slot_bits; both sides now derive their width from the one
 *        stride, so that failure mode no longer exists.)
 *   [A2] EXACTNESS.  With stride >= slot_bits the packed operands are exactly
 *                A = sum_i a_i * 2^(i*stride)      (one coefficient per window, digits < 2^bpw
 *        by construction: the packer writes S <= slot_bits <= stride bits at a stride that is a
 *        whole number of bpw-digit words), so the raw convolution coefficient k of A*B is
 *                c_k = sum_{i+j=k} a_i*b_j <= P * (2^S - 1)^2 < 2^(2S + log2 P) = 2^slot_bits
 *        because each a_i,b_j < 2^S and there are at most P pairs.  c_k < 2^slot_bits <= 2^stride,
 *        so coefficient k is written as exactly ONE window and cannot reach window k+1; no
 *        aliasing and no carry from window k can change c_k, and the counting bound of the probe,
 *        L*(2^bpw-1)^2 < p with L = P*slot_words (the number of nonzero DIGITS per operand), is
 *        the same sufficient condition as before and is still enforced -- by ntt_shape_plan() for
 *        the forced bpw, and again here from the S5Shape this function returns.
 *   [A3] CAPACITY.  The last coefficient's window must be backed by real digits: the operand needs
 *        (P-1)*slot_words + slot_words = P*slot_words digits, which is what choose_cfg's
 *        `2*P*slot_words + 1 <= N` test guarantees with room to spare.
 *
 * The bpw is not a free parameter, but it is no longer required to divide slot_bits either: it is
 * the LARGEST bpw the multiply's own planner accepts when forced to it (via `force_bpw` of
 * ntt_shape_query/ntt_poly_mul_batch_dev) whose resulting stride `slot_words*bpw` is at least
 * slot_bits, with slot_words the planner's own answer.  The scan is over bpw = 26..5: above 26 the
 * exactness bound L*(2^bpw-1)^2 < p fails for the operand lengths this tree reaches, below 5 the
 * carry chain grows without buying anything.  If no such bpw exists the shape has no canonical S5
 * packing and s5_shape_for says so instead of guessing -- which is the failure this function used
 * to hide by silently degrading to bpw = 1.
 */
struct S5Shape {
    unsigned long long N = 0, slot_words = 0, stride = 0, out_slots = 0, slot_bits = 0;
    int bpw = 0;
    bool ok = false;
    const char *why = "";
};

/* the exponent the SLOT must be able to hold for an operand of `m` coefficients: the operand's
   digits span at most floor(log2 m) + 1 coefficient slots, each S bits wide, so its slot value is
   < 2^(S * (floor(log2 m) + 1)).  NOT ceil(log2 m): for a two-coefficient operand ceil gives 1,
   one bit short of the budget, and the reduction then refuses the window (measured on the frozen
   vector's first division). */
static inline unsigned long long s5_lq(unsigned long long m)
{
    unsigned long long b = 1, l = 1;
    while (b < (1ull << 62) && (b << 1) <= m) { b <<= 1; ++l; }
    return l;
}

static S5Shape s5_shape_for(unsigned long long P, int S)
{
    S5Shape r;
    if (P == 0 || S <= 0) { r.why = "bad shape"; return r; }
    unsigned long long log2P = 1;
    while ((1ull << log2P) < P) ++log2P;
    const unsigned long long slot_bits = 2ull * (unsigned long long)S + log2P;
    /* THE STRIDE RULE: stride = ceil(slot_bits/bpw)*bpw is slot_bits exactly when bpw divides
       slot_bits, so scan downwards for a divisor that the multiply's planner also accepts.  Both
       conditions are re-checked from the planner's own answer below; a divisor that fails the
       exactness bound L*(2^bpw-1)^2 < p is simply the next candidate's business, and the planner
       reports it, so this loop cannot pick a bpw the multiply would refuse. */
    unsigned long long qN = 0, qsb = 0, qsw = 0, qss = 0, qos = 0;
    int qbpw = 0;
    int bpw = 0;
    const char *why_last = "no divisor of slot_bits <= 62 is accepted by the shape planner";
    /* ---- THE SLOT IS ROUNDED UP; THE bpw IS NOT ROUNDED DOWN (section 54) -----------------
       `stride = ceil(slot_bits/bpw)*bpw` is at least slot_bits, and the window bound
       `coefficient < 2^(2S+ceil(log2 P)) <= 2^slot_bits` is an UPPER bound -- so a slot padded
       up to the next multiple of bpw is perfectly usable.  What actually matters is that the
       PACKER and the REDUCER agree on one width, and both now derive theirs from
       `slot_stride/bpw` (the packer in s5_pack_kernel, the reducer through `slot_words`).

       The old loop demanded `bpw | slot_bits` and scanned DOWNWARDS from 62.  `2S + ceil(log2 P)`
       is often PRIME (S=129, P=17 gives 263), where the only divisor <= 62 is 1 -- so the shape
       planner silently chose bpw = 1.  Single-bit digits make the carry's chain length O(N),
       which is why the S5 descent needed 128 carry rounds instead of the formula's ~6, and why
       the boundary between "fails" and "works" looked like a cliff (section 53.5). */
    for (int c = 26; c >= 5; --c) {
        if (!ntt_shape_query(P, S, &qN, &qbpw, &qsb, &qsw, &qss, &qos, c)) continue;
        if (qbpw != c) { why_last = "the planner did not honour the forced bpw"; continue; }
        if (qsw * (unsigned long long)c != qss) { why_last = "slot_words*bpw != stride"; continue; }
        if (qss < slot_bits) { why_last = "the stride is narrower than the window bound"; continue; }
        bpw = c;
        break;
    }
    if (!bpw) { r.why = why_last; return r; }
    if (qsb != slot_bits) { r.why = "slot_bits disagrees with its own derivation"; return r; }
    /* [A1] ONE WIDTH FOR EVERYONE.  The stride carries the window and is at least slot_bits, so
       every coefficient still lies wholly inside its own reduction window; the checks below exist
       so a future change cannot silently break that. */
    if (qss < slot_bits || qss != qsw * (unsigned long long)qbpw) {
        r.why = "the stride is narrower than the window, or is not slot_words*bpw";
        return r;
    }
    /* [A3] capacity: every coefficient owns one whole window, and the last one must fit the
       operand's digit budget (the planner's 2*P*slot_words + 1 <= N leaves room for it) */
    if (P > (~0ull) / qsw) { r.why = "L = P*slot_words overflows"; return r; }
    if (P * qsw + qsw > qN) { r.why = "the windows do not fit the transform"; return r; }
    /* [A2] the exactness bound, re-derived for THIS (L, bpw) -- never inherited from another
       shape, another level or a group key; the multiply re-derives it again from L_terms and
       asserts it on the values it really produced. */
    const unsigned long long L = P * qsw;
    if (!exact_ok_terms(L, qbpw)) { r.why = "the exactness bound fails"; return r; }
    r.N = qN; r.slot_words = qsw; r.stride = qss; r.bpw = qbpw;
    r.slot_bits = slot_bits;
    r.out_slots = qos;
    r.ok = true;
    return r;
}


/* one batched NTT multiply of `ds` slices: A (rows of `la` coefficients, scaled by 2^(S*(Lmax-la))
   into the slot window), B (rows of `lb`), output = the first `want` coefficients of the product,
   reduced mod N, written TIGHT into dst (row stride `want`).  `pack_lo`/`pack_hi` name the row
   range of the source arrays (all of a level's nodes share one source layout). */
static void s5_mul_batch(S5Dev &D, const unsigned long long *Asrc, unsigned long long Aoff,
                         unsigned long long la, const unsigned long long *Bsrc,
                         unsigned long long Boff, unsigned long long lb, unsigned long long ds,
                         unsigned long long rowoff, unsigned long long want,
                         unsigned long long *dst, S5Stats &st, int cat)
{
    PolyLayer &L = *D.L;
    if (ds == 0 || want == 0) return;
    const size_t W = L.W;
    const int S = (int)L.S;
    const unsigned long long P = (la > lb) ? la : lb;
    /* S5 derives its own slot shape (see S5Shape) and proves the exactness bound for it, rather
       than inheriting the probe's shape for a coefficient count it never multiplies. */
    const S5Shape sh = s5_shape_for(P, S);
    if (!sh.ok) {
        std::fprintf(stderr, "%s: FATAL: no S5 slot shape for P=%llu S=%d: %s\n", NTT_PROBE_NAME,
                     P, S, sh.why);
        std::exit(3);
    }
    const unsigned long long qN = sh.N, qbpw = (unsigned long long)sh.bpw, qos = sh.out_slots;
    const unsigned long long sstride = sh.stride;
    /* THE PACKING RULE, in one line: stride == slot_bits, and the packer therefore writes each
       coefficient at its own slot offset with NO scale.  The scale (and the two candidate fixes
       the report recorded) is gone: with stride == slot_bits a coefficient's block lies wholly
       inside its own reduction window and no window sees a neighbour's bits, so there is nothing
       to pull back.  See the S5Shape comment for [A1]/[A2]/[A3]; the three checks below are the
       same statement, measured against the values this call is about to use. */
    const unsigned long long qsb2 = 2ull * (unsigned long long)S + ceil_log2_u64(P);
    if (qsb2 != sh.slot_bits) {
        std::fprintf(stderr, "%s: FATAL: the S5 shape disagrees with its own derivation "
                             "(slot_bits=%llu vs %llu)\n", NTT_PROBE_NAME, sh.slot_bits, qsb2);
        std::exit(3);
    }
    if (sstride < qsb2) {
        std::fprintf(stderr, "%s: FATAL: the S5 slot stride is narrower than the slot window "
                             "(stride=%llu slot_bits=%llu): a window would hold bits of its "
                             "neighbour\n", NTT_PROBE_NAME, sstride, qsb2);
        std::exit(3);
    }
    /* [A1] the block of the LAST coefficient of the operand must end inside the window that the
       reducer reads for it.  Because the stride IS the window width, "inside its own window" for
       every i follows from this one inequality plus S <= slot_bits. */
    if (P * sstride < (P - 1) * sstride + (unsigned long long)S ||
        (unsigned long long)S > qsb2) {
        std::fprintf(stderr, "%s: FATAL: the S5 packing overruns the operand (P=%llu stride=%llu "
                             "S=%d slot_bits=%llu)\n", NTT_PROBE_NAME, P, sstride, S, qsb2);
        std::exit(3);
    }
    /* the operand spans, at the slot stride, must fit the transform the shape chose */
    if (P * sstride > qN * qbpw) {
        std::fprintf(stderr, "%s: FATAL: the S5 operands do not fit the NTT array (P=%llu * "
                             "stride=%llu > N=%llu * bpw=%llu)\n", NTT_PROBE_NAME, P, sstride,
                     qN, qbpw);
        std::exit(3);
    }
    /* ---- THE ARRAY MUST HOLD EVERY PRODUCT WINDOW, NOT JUST THE OPERANDS (section 52) -------
       The assertions above bound the OPERANDS: `P*slot_stride <= N*bpw`.  The reduction then
       reads `out_slots = 2P-1` windows at `k*slot_words`, and the LAST of them ends at
       `(2P-1)*slot_words` WORDS -- nearly twice the operand span.  Nothing checked that against
       the array size, so the tail windows of every multiply could be read from beyond the slice.
       Asserted here, where both numbers are known. */
    if (sh.out_slots * sh.slot_words > qN) {
        std::fprintf(stderr, "%s: FATAL: the S5 product windows overrun the digit array: "
                             "out_slots=%llu * slot_words=%llu = %llu words > N=%llu "
                             "(the operand assertion only bounds P=%llu, not 2P-1)\n",
                     NTT_PROBE_NAME, (unsigned long long)sh.out_slots,
                     (unsigned long long)sh.slot_words,
                     (unsigned long long)(sh.out_slots * sh.slot_words), qN,
                     (unsigned long long)P);
        std::exit(3);
    }
    /* [A2] THE WINDOW BOUND, from the values: c_k = sum_{i+j=k} a_i*b_j is a sum of at most P       products of two S-bit numbers, so c_k <= P*(2^S-1)^2 < 2^(2S+log2 P) = 2^slot_bits whenever
       P <= 2^log2P -- and log2P is defined as ceil(log2 P) two lines up, so this holds by
       construction and the assertion is what keeps the definition and the use tied together. */
    if (P > (1ull << ceil_log2_u64(P))) {
        std::fprintf(stderr, "%s: FATAL: the S5 window bound needs P <= 2^ceil(log2 P) (P=%llu)\n",
                     NTT_PROBE_NAME, P);
        std::exit(3);
    }
    const unsigned long long qsw = sh.slot_words, qss = sh.stride;
    /* THE IN-RUN WITNESS FOR [A1]/[A2] (NTT_S5_CANON_CHECK=1): the reducer's own guard, applied
       to the digits this call is about to hand it.  It costs a device-to-host copy of one slice,
       so it is off by default; when it is on, `value_bits` MUST come out strictly below
       `slot_bits` for every coefficient -- that is the exact measurement (261 vs 260) that
       refused the old layout, so it is the one number worth being able to reproduce. */
    if (std::getenv("NTT_S5_CANON_CHECK") && *std::getenv("NTT_S5_CANON_CHECK")
        && std::atoi(std::getenv("NTT_S5_CANON_CHECK")) != 0) {
        std::vector<unsigned long long> hp((size_t)qN, 0ull);
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(hp.data(), Asrc + Aoff, (size_t)qN * sizeof(unsigned long long),
                      cudaMemcpyDeviceToHost));
        unsigned long long worst = 0, worst_i = 0;
        for (unsigned long long i = 0; i < la; ++i) {
            unsigned long long vb = 0;
            for (int b = (int)qsb2 - 1; b >= 0; --b) {
                const unsigned long long gb = i * qsb2 + (unsigned long long)b;
                if ((hp[gb / 64] >> (gb % 64)) & 1ull) { vb = (unsigned long long)b + 1; break; }
            }
            if (vb > worst) { worst = vb; worst_i = i; }
        }
        std::fprintf(stderr, "s5_canon: P=%llu la=%llu slot_bits=%llu stride=%llu bpw=%llu "
                             "worst_value_bits=%llu at_i=%llu -> %s\n", P, la, qsb2, sstride,
                     qbpw, worst, worst_i, (worst < qsb2) ? "inside its window" : "OVERRUN");
        if (worst >= qsb2) std::exit(3);
    }
    /* FORENSIC (NTT_S5_DIGDUMP=1): the first multiply of a shape prints its two source operands,
       the packed slots and the reduced coefficients, so "wrong pack" and "wrong multiply" can be
       told apart by inspection instead of by argument. */
    const char *ed = std::getenv("NTT_S5_DIGDUMP");
    const bool dig = (ed && *ed && std::atoi(ed) != 0 && la <= 6 && ds == 1);
    /* the pack pool holds: pA, pB, the NTT's own copies and the reduction's tight scratch */
    const size_t per = 3 * (size_t)qN + (size_t)qos * W;
    size_t maxs = (D.scratch_words > 3 * (size_t)qN)
                      ? ((D.scratch_words - 3 * (size_t)qN) / per) : 0;
    if (maxs == 0) maxs = 1;
    if (maxs > ds) maxs = (size_t)ds;
    {
        const size_t need = (size_t)qN * maxs * 2 + (size_t)qos * maxs * W;
        if (need > D.scratch_words) {
            std::fprintf(stderr, "%s: FATAL: the S5 pack pool holds %llu words but the shape "
                                 "P=%llu needs %llu for one slice\n", NTT_PROBE_NAME,
                         (unsigned long long)D.scratch_words, P, (unsigned long long)need);
            std::exit(3);
        }
    }
    const unsigned long long chunk = (unsigned long long)maxs;
    for (unsigned long long s0 = 0; s0 < ds; s0 += chunk) {
        const unsigned long long m = ((ds - s0) < chunk) ? (ds - s0) : chunk;
        D.reset();
        unsigned long long *pA = D.alloc("packA", (size_t)qN * m);
        unsigned long long *pB = D.alloc("packB", (size_t)qN * m);
        unsigned long long *tmp = D.alloc("reduce", (size_t)qos * m * W);
        CK(cudaMemset(pA, 0, (size_t)qN * m * sizeof(unsigned long long)));
        CK(cudaMemset(pB, 0, (size_t)qN * m * sizeof(unsigned long long)));
        const double tp0 = now_s();
        {
            const unsigned int th = 256;
            const unsigned long long total = m * la;
            s5_pack_kernel<<<(unsigned int)((total + th - 1) / th), th>>>(
                Asrc, Aoff + s0 * la * W, S, (int)qbpw, sstride, qN, m, la, (int)W,
                /*off_digits=*/0ull, 1ull, pA);
            CK(cudaGetLastError());
            const unsigned long long totalb = m * lb;
            s5_pack_kernel<<<(unsigned int)((totalb + th - 1) / th), th>>>(
                Bsrc, Boff, S, (int)qbpw, sstride, qN, m, lb, (int)W,
                /*off_digits=*/0ull, 1ull, pB);
            CK(cudaGetLastError());
        }
        st.t_pack += now_s() - tp0;
        if (dig) {
            /* the SOURCE operands as the packer sees them */
            std::vector<unsigned long long> ha((size_t)la * W, 0ull);
            CK(cudaMemcpy(ha.data(), Asrc + Aoff, ha.size() * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
            std::fprintf(stderr, "s5_dig: P=%llu la=%llu lb=%llu S=%d bpw=%llu sw=%llu "
                                 "sstride=%llu N=%llu out_slots=%llu\n",
                         P, la, lb, S, qbpw, qsw, sstride, qN, qos);
            for (unsigned long long i = 0; i < la && i < 3; ++i) {
                std::fprintf(stderr, "  A[%llu]=", i);
                for (int t = (int)W - 1; t >= 0; --t) std::fprintf(stderr, "%016llx", ha[i * W + t]);
                std::fprintf(stderr, "\n");
            }
            /* the packed slots as the reduction will read them */
            std::vector<unsigned long long> hp((size_t)qN, 0ull);
            CK(cudaMemcpy(hp.data(), pA, (size_t)qN * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
            for (unsigned long long k = 0; k < la && k < 3; ++k) {
                std::fprintf(stderr, "  slot[%llu]=", k);
                for (int t = 0; t < (int)qsw; ++t)
                    std::fprintf(stderr, "%016llx", hp[k * qsw + t]);
                std::fprintf(stderr, "\n");
            }
        }
        /* FORENSIC (NTT_S5_DIGDUMP=1): read the packed operand back and compare it with the source
           coefficients -- the one measurement that separates "the pack is wrong" from "the
           multiply is wrong".  It costs a round trip, so it is off by default.

           THE READ MUST MATCH THE WRITTEN LAYOUT (section 54.6).  The digit array holds ONE DIGIT
           PER 64-BIT WORD (each < 2^bpw) -- that is how the reducer reads its window (slot_words
           digits, low digit first) and how the slot dump above reads it (`hp[k*qsw + t]`).  This
           comparison instead treated the buffer as a BIT array (`hp[bit/64] >> (bit%64)`), i.e. a
           DIFFERENT data structure from the one under test, so it accused the packer on shapes that
           are demonstrably exact: at D=210, 358 of 596 samples reported mismatching_words != 0 at
           n=64/sw=10, a shape whose descent is leaf-for-leaf correct.  A checker that cries wolf
           everywhere cannot localise anything -- and note that its neighbour ten lines up was
           reading the same buffer correctly, so the two halves of this one block disagreed with
           each other.  (Section 14.8 is the same lesson pointing the other way: a checker that
           endorsed the packer through a shared wrong assumption.) */
        {
            const char *ed = std::getenv("NTT_S5_DIGDUMP");
            /* la <= 512, not la <= 8: the shapes worth checking are the LARGE ones (the root
               division multiplies at la = 129), and a cap that only sees the deep levels cannot
               see the shape that fails. */
            if (ed && *ed && std::atoi(ed) != 0 && s0 == 0 && la <= 512) {
                std::vector<unsigned long long> hp((size_t)qN, 0ull);
                std::vector<unsigned long long> sa((size_t)la * W, 0ull);
                CK(cudaMemcpy(hp.data(), pA, (size_t)qN * sizeof(unsigned long long),
                              cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(sa.data(), Asrc + Aoff, sa.size() * sizeof(unsigned long long),
                              cudaMemcpyDeviceToHost));
                unsigned long long bad = 0;
                for (unsigned long long i = 0; i < la; ++i) {
                    std::vector<unsigned long long> got(W, 0ull);
                    /* coefficient i occupies the digits [i*qsw, i*qsw+qsw), digit t holding the
                       coefficient's bits [t*bpw, (t+1)*bpw) */
                    for (unsigned long long t = 0; t < (unsigned long long)qsw; ++t) {
                        const unsigned long long d = i * (unsigned long long)qsw + t;
                        if (d >= (unsigned long long)qN) break;
                        const unsigned long long v = hp[d];
                        for (unsigned long long b = 0; b < (unsigned long long)qbpw; ++b) {
                            const unsigned long long bit = t * (unsigned long long)qbpw + b;
                            if (bit >= (unsigned long long)S) break;
                            if ((v >> b) & 1ull) got[bit / 64] |= (1ull << (bit % 64));
                        }
                    }
                    for (size_t t = 0; t < W; ++t)
                        if (got[t] != sa[i * W + t]) {
                            if (!bad)
                                std::fprintf(stderr, "  pack mismatch i=%llu word=%llu src=%016llx "
                                                     "got=%016llx\n", i, (unsigned long long)t,
                                             sa[i * W + t], got[t]);
                            ++bad;
                        }
                }
                std::fprintf(stderr, "s5_pack: P=%llu bpw=%llu sw=%llu sstride=%llu N=%llu "
                                     "out_slots=%llu coeffs=%llu mismatching_words=%llu\n",
                             P, qbpw, qsw, sstride, qN, qos, la, bad);
            }
        }
        S5RedHook hk;
        hk.D = &D;
        hk.out = dst;
        hk.out_stride = want * W;
        hk.row0 = rowoff + s0;
        hk.w = W;
        hk.tmp = tmp;
        hk.tmp_stride = qos;
        NttReduceHook nh;
        nh.ctx = &hk;
        nh.run = s5_reduce_hook;
        nh.out = tmp;
        nh.w = (unsigned long long)W;
        nh.sample = 0;
        NttMulStats nst{};
        const double tg0 = now_s();
        const int rc = ntt_poly_mul_batch_dev(P, S, L.device, m, pA, pB, &nst, L.arena, &nh,
                                              nullptr, (int)qbpw);
        st.t_reduce += now_s() - tg0;
        /* ---- WHERE ONE S5 MULTIPLY SPENDS ITS 550 us (section 22) --------------------------
           `t_generic` said a division costs 2542 us and that 4.7 multiplies of ~550 us each make
           it up, but a degree-1440 multiply at S=129 has no business costing 550 us -- the NTT of
           that size is tens of microseconds.  The multiply's own timers (t_fwd/t_inv/t_slot from
           the probe, plus the carry-convergence check and its readback that section 41 split into
           three parts) are accumulated here so the answer is measured rather than assumed. */
        st.mul_calls++;
        st.t_mul_fwd += nst.t_fwd; st.t_mul_inv += nst.t_inv; st.t_mul_slot += nst.t_slot;
        st.t_mul_hpack += nst.t_hpack; st.t_mul_h2d += nst.t_h2d_batch;
        st.t_mul_check += nst.t_check;
        st.t_mul_check_d2h += nst.t_check_d2h; st.t_mul_check_kernel += nst.t_check_kernel;
        st.t_mul_hplan += nst.t_plan; st.t_mul_hopcopy += nst.t_opcopy;
        st.t_mul_all += nst.t_fwd + nst.t_inv + nst.t_slot + nst.t_hpack + nst.t_h2d_batch +
                        nst.t_check + nst.t_plan + nst.t_opcopy + nst.t_hout;
        if (st.mul_calls == 1) st.mul_first_shape = P;
        /* ---- THE MULTIPLY'S RETURN CODE WAS CAPTURED AND NEVER EXAMINED (section 49) --------
           `ntt_poly_mul_batch_dev` returns 4 with "CARRY DID NOT CONVERGE" on stderr when the
           carry stage leaves a digit at or above 2^bpw.  That is EXACTLY the state the S5
           reduction cannot survive: the kernel's digit-to-limb conversion ORs each digit in
           (`acc |= v << nacc`), so a digit wider than bpw silently loses its high bits, while the
           host's GMP reduction treats the digits as a number and keeps them -- a disagreement no
           other check in this file can see.  Ignoring the code turned a loud failure into silent
           corruption.  It is fatal now. */
        if (rc != 0) {
            std::fprintf(stderr, "%s: FATAL: the S5 multiply failed (rc=%d) at P=%llu -- refusing "
                                 "to reduce digits it cannot trust.  The carry reported %llu "
                                 "unconverged digits with a maximum digit height of %llu bits "
                                 "(bpw=%llu), so it needs at least %llu rounds plus the ripple's "
                                 "own length\n", NTT_PROBE_NAME, rc, (unsigned long long)P,
                         nst.carry_residual, nst.carry_max_bits, qbpw,
                         qbpw ? (nst.carry_max_bits + qbpw - 1) / qbpw : 0ull);
            std::exit(3);
        }
        /* THE SHAPE THE MULTIPLY REALLY RAN must be the one the packer assumed: its slot_words is
           what the reduction reads the coefficients at, its slot_bits is what the reduction
           asserts the windows against, and its slot_stride is the bit distance between two
           coefficients.  A mismatch here would be silent corruption, so it is fatal -- and it is
           also the check that keeps the forced bpw honest (see choose_cfg rule (3)). */
        if (nst.N != qN || nst.slot_words != qsw || (unsigned long long)nst.slot_stride != qss ||
            (unsigned long long)nst.slot_bits != sh.slot_bits) {
            std::fprintf(stderr, "%s: FATAL: the S5 multiply ran a different shape than the packer "
                                 "assumed (N=%llu/%llu slot_words=%llu/%llu stride=%llu/%llu "
                                 "slot_bits=%llu/%llu)\n", NTT_PROBE_NAME, nst.N, qN,
                         nst.slot_words, qsw, nst.slot_stride, qss, nst.slot_bits, sh.slot_bits);
            std::exit(3);
        }
        if ((unsigned long long)nst.bpw != qbpw) {
            std::fprintf(stderr, "%s: FATAL: the S5 multiply chose bpw=%d but the operands were "
                                 "packed at bpw=%llu\n", NTT_PROBE_NAME, nst.bpw, qbpw);
            std::exit(3);
        }
        /* ONE LINE OF ATTESTATION of the shape the planner actually chose (section 54).  The whole
           `bpw = 1` defect was invisible in every aggregate number: it made the carry's chain
           length O(N), which showed up only as "the descent needs 128 carry rounds" -- a symptom
           three steps away from its cause.  Naming bpw, the stride and the round count the
           multiply derived for itself makes the next such degradation a one-line read instead of
           a bisection.
           Printed on STDOUT: native STDERR lines get wrapped and split mid-field by the
           PowerShell wrapper the gates drive this exe through, which is how section 44.3's
           "231 of 486 rows differ" was manufactured out of console formatting.
           RE-ATTESTED WHENEVER P GROWS, so a run prints the descent's whole shape ladder (the
           descent multiplies at P = 2, 4, ... at its deepest levels and at the leaf/root shape at
           its top); a once-per-process line only ever showed the smallest shape and hid the one
           the expensive multiplies use. */
        {
            static unsigned long long attested_max = 0;
            if (P > attested_max) {
                attested_max = P;
                std::printf("s5_shape_attest: P=%llu S=%d slot_bits=%llu -> bpw=%llu "
                            "stride=%llu slot_words=%llu N=%llu out_slots=%llu | the "
                            "multiply derived %d carry rounds by itself and left %llu "
                            "unconverged digits (max height %llu bits)\n",
                            P, S, sh.slot_bits, qbpw, sstride, qsw, qN, qos, nst.carry_rounds,
                            nst.carry_residual, nst.carry_max_bits);
            }
        }
        ++st.ntt_launches;
        L.ntt_launches += m;
        L.ntt_calls += m;
        L.muls += m;
        L.max_ntt_words = std::max(L.max_ntt_words, nst.N);
        L.max_ntt_coeffs = std::max(L.max_ntt_coeffs, P);
        L.max_slot_bits = std::max(L.max_slot_bits, nst.slot_bits);
        /* the tree's own exactness bound, re-derived for THIS shape from the values the multiply
           returned -- never inherited from another level, another shape or the probe */
        if (!exact_ok_terms(nst.L_terms, nst.bpw)) {
            std::fprintf(stderr, "%s: EXACTNESS VIOLATED for the S5 shape: L=%llu bpw=%d\n",
                         NTT_PROBE_NAME, nst.L_terms, nst.bpw);
            std::exit(3);
        }
        {
            const double bb = coeff_bound_bits_terms(nst.L_terms, nst.bpw);
            if (bb > L.bind_bound_bits) {
                L.bind_bound_bits = bb;
                L.bind_P = P;
                L.bind_L = nst.L_terms;
                L.bind_slot_bits = nst.slot_bits;
                L.bind_slot_words = nst.slot_words;
                L.bind_bpw = nst.bpw;
            }
        }
        /* the sparse half: the reduction asserts the canonicality of each slot, and the NTT
           path asserts the packing rule by putting A0/B0 at the slot's own bit offset (a shrink
           of zero is impossible: sA is by construction >= 0). */
        const int c = (cat >= 0) ? cat : L.cat;
        if (c >= 0)
            for (unsigned long long s = 0; s < m; ++s) L.cost.add(c, la, lb);
    }
}

/* the quotient/remainder chain for ONE node (the reference's cp_divmod, done on the device):
     ra = rev_k(top k of A), rb = rev_{db+1}(B), rbi = 1/rb mod X^k by Newton doubling,
     qrev = (ra*rbi) mod X^k, q = rev_k(qrev), r = A - q*B.
   Using rb as its own inverse is ONLY valid when k == db+1; the Newton chain is what makes the
   general case correct.  `la` = the source row's coefficient count, db = the divisor's degree,
   dst = the remainder's row (db coefficients, stride W). */
static void s5_divmod_one(S5Dev &D, const unsigned long long *Asrc, unsigned long long Aoff,
                          unsigned long long la, const unsigned long long *Bsrc,
                          unsigned long long Boff, unsigned long long lb, unsigned long long db,
                          unsigned long long *dst, S5Stats &st)
{
    PolyLayer &L = *D.L;
    const size_t W = L.W;
    const unsigned long long k = la - lb + 1;               /* da - db + 1 > 0 */
    if (la < lb || k == 0 || db == 0) {
        std::fprintf(stderr, "%s: FATAL: s5_divmod_one with la=%llu lb=%llu db=%llu\n",
                     NTT_PROBE_NAME, la, lb, db);
        std::exit(3);
    }
    D.preset();
    static int divdump_calls2 = 0;
    const double tg0 = now_s();
    double ts0 = tg0;                       /* per-stage stamps (section 21) */
    ++st.n_div;
    const unsigned long long newton_steps_before = st.newton_steps;
    const unsigned int th = 256;
    const unsigned long long *Asub = Asrc + Aoff;
    const unsigned long long *Bsub = Bsrc + Boff;
    /* ---- ra = rev_k(top k of A), rb = rev_{db+1}(B) ----------------------------------- */
    unsigned long long *ra = D.palloc((size_t)k * W);
    unsigned long long *rb = D.palloc((size_t)(db + 1) * W);
    {
        const unsigned long long n = (k > (db + 1)) ? k : (db + 1);
        s5_rev_pack_kernel<<<S5_GRID(n)>>>(ra, Asub, 0, k, la, (int)W);
        CK(cudaGetLastError());
        s5_rev_pack_kernel<<<S5_GRID(n)>>>(rb, Bsub, 0, db + 1, lb, (int)W);
        CK(cudaGetLastError());
    }
    /* ---- rbi = 1/rb mod X^k by Newton doubling (Montgomery form throughout) ------------- */
    unsigned long long *A = D.palloc((size_t)k * W);        /* the two alternating g buffers */
    unsigned long long *B = D.palloc((size_t)k * W);
    s5_fill_kernel<<<(unsigned int)((k + th - 1) / th), th>>>(A, k, (int)W);   /* g = 1 */
    CK(cudaGetLastError());
    unsigned long long *g = A;
    unsigned long long len = 1;
    st.t_revpack += now_s() - ts0; ts0 = now_s();
    while (len < k) {
        ++st.newton_steps;
        const unsigned long long nxt = ((2 * len) < k) ? (2 * len) : k;
        unsigned long long *gpad = D.palloc((size_t)nxt * W);
        unsigned long long *at = D.palloc((size_t)nxt * W);
        s5_pad_low_kernel<<<(unsigned int)((nxt + th - 1) / th), th>>>(gpad, g, nxt, len, (int)W);
        CK(cudaGetLastError());
        s5_pad_low_kernel<<<(unsigned int)((nxt + th - 1) / th), th>>>(at, rb, nxt, db + 1,
                                                                      (int)W);
        CK(cudaGetLastError());
        /* ag = (at*g) mod X^nxt.  The batched multiply writes the first `want` coefficients
           TIGHT, so the host code's flat_truncate is a stride here: no copy, no host, and the
           operand A (at) is only read -- never written -- so it can double as the next g. */
        unsigned long long *ag = D.palloc((size_t)nxt * W);
        s5_mul_batch(D, at, 0, nxt, gpad, 0, nxt, 1, 0, nxt, ag, st, -1);
        st.newton_muls += (2 * len == nxt) ? 2 : 2;
        unsigned long long *h = D.palloc((size_t)nxt * W);
        s5_two_minus_kernel<<<(unsigned int)((nxt + th - 1) / th), th>>>(
            ag, h, nxt, D.red->dn, D.red->ninv, D.red->nw);
        CK(cudaGetLastError());
        /* g = (gpad*h) mod X^nxt, written into the OTHER full-length buffer */
        unsigned long long *gn = (g == A) ? B : A;
        s5_mul_batch(D, gpad, 0, nxt, h, 0, nxt, 1, 0, nxt, gn, st, -1);
        g = gn;
        /* THE CONSTANT TERM OF A NEWTON INVERSE IS A FIXED POINT (section 45): every divisor that
           reaches this chain is monic (rb[0] = 1), so 1/rb has constant term 1 at EVERY doubling
           -- if it ever stops being 1, the iteration that did it is the broken one.  Two words
           per line: this console splits long printf output mid-field. */
        if (std::getenv("NTT_S5_DIVDUMP") && *std::getenv("NTT_S5_DIVDUMP")
            && std::atoi(std::getenv("NTT_S5_DIVDUMP")) != 0 && divdump_calls2++ < 60) {
            unsigned long long g0[2] = {0, 0};
            CK(cudaMemcpy(g0, gn, sizeof(g0), cudaMemcpyDeviceToHost));
            std::fprintf(stderr, "s5_newton: len=%llu g0=%llx g1=%llx\n", len,
                         (unsigned long long)g0[0], (unsigned long long)g0[1]);
        }
        len = nxt;
    }
    st.t_newton += now_s() - ts0; ts0 = now_s();
    /* ---- qrev = (ra*rbi) mod X^k, q = rev_k(qrev), qb = q*B --------------------------- */
    unsigned long long *qrev = D.palloc((size_t)k * W);
    s5_mul_batch(D, ra, 0, k, g, 0, k, 1, 0, k, qrev, st, -1);
    unsigned long long *q = D.palloc((size_t)k * W);
    s5_rev_pack_kernel<<<(unsigned int)((k + th - 1) / th), th>>>(q, qrev, 0, k, k, (int)W);
    CK(cudaGetLastError());
    const unsigned long long wantb = k + db;                /* deg(q*B) + 1 */
    unsigned long long *qb = D.palloc((size_t)wantb * W);
    s5_mul_batch(D, q, 0, k, Bsub, 0, lb, 1, 0, wantb, qb, st, -1);
    st.t_qb += now_s() - ts0; ts0 = now_s();
    /* ---- dst = A - qb, coefficient by coefficient, ON THE DEVICE -----------------------
       THE LAUNCH MUST COVER rows*nw THREADS, NOT rows (section 55).  This kernel is written with
       ONE THREAD PER (ROW, WORD): it guards on `gid >= rows*nw` and derives its row as `gid/nw`.
       Launching it with S5_GRID(rows) therefore supplies only ceil(rows/256)*256 threads, and for
       rows = 128 at nw = 3 that is 256 of the 384 needed -- so rows 0..85 were computed and rows
       86..127 were NEVER WRITTEN (86 = floor(255/3), exactly the first coefficient the descent
       check flagged, word 258 = 86*3).  It stayed hidden because every smaller shape fits in one
       block: at D=210 rows = 16 needs 48 threads, and the D=2310 level-7 divisions have rows = 64
       (192 threads) -- both correct, both green.  The destination kept its previous contents, so
       the "low two limbs are zero" signature the forensics showed was a STALE ROW, not a shifted
       value.  `s5_memcpy_rows` (above) passes S5_GRID(rows*W) for exactly this reason; this call
       site simply missed the factor. */
    {
        const unsigned long long rows = db;
        s5_sub_kernel<<<S5_GRID((unsigned long long)rows * (unsigned long long)D.red->nw)>>>(
            Asub, 0, (int)la, qb, 0, (int)wantb, (int)db,
            D.red->dn, D.red->ninv, D.red->nw, dst, (unsigned long long)W);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
    }
    st.t_sub += now_s() - ts0;
    /* the multiply count of THIS division: 2 per doubling step + qrev + q*B (the first version
       accumulated `2*st.newton_steps + 2`, i.e. the RUNNING total, which over-counted
       quadratically -- 19311210 instead of ~16000 at D=30030) */
    st.n_launch_div += (unsigned long long)(2 * (st.newton_steps - newton_steps_before) + 2);
    st.t_generic += now_s() - tg0;
    /* ---- THE HOST `cp_divmod` IS THE REFERENCE THIS FUNCTION REARRANGES (section 45) -------
       NTT_S5_DIVDUMP=1 recomputes the SAME division on the host from the SAME device inputs and
       reports whether the device's QUOTIENT and REMAINDER agree with it.  That splits the
       function in half in one run: a quotient that differs means the rev-pack / Newton chain, a
       quotient that agrees but a remainder that does not means the q*B multiply or the final
       subtraction.  Every field is a small integer or a flag -- the earlier forensics were
       misread because this console splits long printf lines mid-field, which produced a false
       "231 of 486 rows differ" that cost a whole investigation. */
    {
        static int divdump_calls = 0;
        const char *dd = std::getenv("NTT_S5_DIVDUMP");
        if (dd && *dd && std::atoi(dd) != 0 && divdump_calls++ < 3) {
            std::vector<unsigned long long> Ah((size_t)la * W, 0ull), Bh((size_t)lb * W, 0ull);
            CK(cudaMemcpy(Ah.data(), Asub, Ah.size() * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(Bh.data(), Bsub, Bh.size() * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
            std::vector<unsigned long long> QD((size_t)k * W, 0ull), RD((size_t)db * W, 0ull);
            CK(cudaMemcpy(QD.data(), q, QD.size() * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(RD.data(), dst, RD.size() * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
            const CPoly a_h = cp_from_flat(Ah, la - 1, W);
            const CPoly b_h = cp_from_flat(Bh, lb - 1, W);
            CPoly q_h, r_h;
            cp_divmod(q_h, r_h, a_h, b_h, L);
            long qbad = -1, rbad = -1;
            for (size_t i = 0; i < k && qbad < 0; ++i)
                for (size_t t = 0; t < W; ++t) {
                    const unsigned long long want = (i < q_h.size()) ? q_h[i][t] : 0ull;
                    if (QD[i * W + t] != want) { qbad = (long)i; break; }
                }
            for (size_t i = 0; i < (size_t)db && rbad < 0; ++i)
                for (size_t t = 0; t < W; ++t) {
                    const unsigned long long want = (i < r_h.size()) ? r_h[i][t] : 0ull;
                    if (RD[i * W + t] != want) { rbad = (long)i; break; }
                }
            std::fprintf(stderr, "s5_divdump: la=%llu lb=%llu db=%llu k=%llu flag_q=%ld flag_r=%ld\n",
                         la, lb, db, k, qbad, rbad);
            /* ---- THE LAST SPLIT: `qb = q*B` OR `A - qb` (section 55) -------------------------
               flag_q already clears the rev-pack and the whole Newton chain, and flag_r says the
               REMAINDER first differs at some coefficient.  The remainder is produced by exactly
               two more device operations -- the q*B multiply and the subtraction -- so reading
               `qb` back and comparing it with the host's q_h*b_h splits them, and that is the last
               split this function can offer.  A `flag_qb` that equals `flag_r` means the MULTIPLY;
               `flag_qb = -1` with a nonzero `flag_r` means the SUBTRACTION. */
            {
                const unsigned long long wantb = k + db;      /* deg(q*B) + 1, as at the call site */
                std::vector<unsigned long long> QBh((size_t)wantb * W, 0ull);
                CK(cudaMemcpy(QBh.data(), qb, QBh.size() * sizeof(unsigned long long),
                              cudaMemcpyDeviceToHost));
                const CPoly hb = cp_mul(q_h, b_h, L);
                long qbbad = -1;
                for (size_t i = 0; i < (size_t)wantb && qbbad < 0; ++i)
                    for (size_t t = 0; t < W; ++t) {
                        const unsigned long long want = (i < hb.size()) ? hb[i][t] : 0ull;
                        if (QBh[i * W + t] != want) { qbbad = (long)i; break; }
                    }
                std::fprintf(stderr, "s5_divdump_qb: wantb=%llu flag_qb=%ld\n",
                             (unsigned long long)wantb, qbbad);
            }
            /* ---- WHICH STAGE OF THE QUOTIENT CHAIN (each flag: -1 = equal, else the first
               differing coefficient index).  ra -> rb -> Newton inverse g -> qrev/q. ---------- */
            {
                CPoly ra_h, rb_h;
                cp_resize(ra_h, k, W);
                cp_resize(rb_h, (size_t)db + 1, W);
                for (size_t i = 0; i < k && i < (size_t)la; ++i)
                    ra_h[i] = a_h[(size_t)((long)la - 1 - (long)i)];
                for (size_t i = 0; i <= (size_t)db && i < (size_t)lb; ++i)
                    rb_h[i] = b_h[(size_t)((long)db - (long)i)];
                std::vector<unsigned long long> RA((size_t)k * W, 0ull),
                                               RB((size_t)(db + 1) * W, 0ull),
                                               G((size_t)k * W, 0ull),
                                               GB((size_t)k * W, 0ull);
                CK(cudaMemcpy(RA.data(), ra, RA.size() * sizeof(unsigned long long),
                              cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(RB.data(), rb, RB.size() * sizeof(unsigned long long),
                              cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(G.data(), g, G.size() * sizeof(unsigned long long),
                              cudaMemcpyDeviceToHost));
                const CPoly g_h = cp_inv_series(rb_h, k, L);
                for (size_t i = 0; i < k && i < g_h.size(); ++i)
                    std::copy(g_h[i].begin(), g_h[i].end(), GB.begin() + (long)(i * W));
                long f_ra = -1, f_rb = -1, f_g = -1;
                for (size_t i = 0; i < k && f_ra < 0; ++i)
                    for (size_t t = 0; t < W; ++t)
                        if (RA[i * W + t] != ra_h[i][t]) { f_ra = (long)i; break; }
                for (size_t i = 0; i <= (size_t)db && f_rb < 0; ++i)
                    for (size_t t = 0; t < W; ++t)
                        if (RB[i * W + t] != rb_h[i][t]) { f_rb = (long)i; break; }
                for (size_t i = 0; i < k && f_g < 0; ++i)
                    for (size_t t = 0; t < W; ++t)
                        if (G[i * W + t] != GB[i * W + t]) { f_g = (long)i; break; }
                std::fprintf(stderr, "s5_divstage: ra=%ld rb=%ld g=%ld rb0=%llx g0=%llx\n", f_ra,
                             f_rb, f_g, (unsigned long long)RB[0], (unsigned long long)G[0]);
            }
            /* the first two words of the remainder this call WROTE, so the caller's view of the
               same buffer can be compared field by field (short lines: this console splits long
               ones mid-field) */
            for (int t = 0; t < 2; ++t)
                std::fprintf(stderr, "s5_divdump_dst: t=%d dev=%llx ref=%llx\n", t,
                             (unsigned long long)RD[t],
                             (unsigned long long)((r_h.size() && r_h[0].size() > (size_t)t)
                                                      ? r_h[0][(size_t)t] : 0ull));
        }
    }
}

/* ---------------------------------------------------------------------------------------
 * the forest: the slice of the F heap one leaf chunk's descent can reach, flattened once
 * --------------------------------------------------------------------------------------- */

/* a device-to-device row copy with zero fill: dst rows [0,rows) get min(len,rows) coefficients of
   src and zeros above */
__global__ void s5_memcpy_kernel(const unsigned long long *src, unsigned long long dst_off,
                                 unsigned long long rows, unsigned long long len, int W,
                                 unsigned long long *dst)
{
    const unsigned long long gid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    const unsigned long long total = rows * (unsigned long long)W;
    if (gid >= total) return;
    const unsigned long long i = gid / (unsigned long long)W;
    const int t = (int)(gid - i * (unsigned long long)W);
    dst[dst_off + gid] = (i < len) ? src[(size_t)i * W + t] : 0ull;
}

static void s5_memcpy_rows(const unsigned long long *src, size_t src_off, size_t dst_off,
                           unsigned long long rows, unsigned long long len, int W,
                           unsigned long long *dst, unsigned long long *host_n)
{
    if (rows == 0) return;
    *host_n = rows * (unsigned long long)W;
    s5_memcpy_kernel<<<S5_GRID(*host_n)>>>(src + src_off, dst_off, rows, len, W, dst);
    CK(cudaGetLastError());
}

/* build the device forest for one chunk of leaves ([lo, lo+L)): every F node the descent of that
   range can reach -- the code ranges of levels 1..16 -- with off[]/sz[] computed from Fdeg on the
   HOST (never from the group key) and one device-to-device copy per node. */
static int s5_forest_build(S5Forest &F, const std::vector<std::vector<unsigned long long>> &Ft,
                           const std::vector<size_t> &Fdeg, size_t Fpad, size_t lo, size_t L,
                           size_t W, S5Stats &st)
{
    if (F.dA) { cudaFree(F.dA); F.dA = nullptr; }
    const size_t top = lo + L;
    F.off.assign(2 * Fpad, 0ull);
    F.sz.assign(2 * Fpad, 0ull);
    F.entry.clear();
    /* ---- EVERY NODE THE KERNELS CAN ASK FOR MUST BE IN THE FOREST (section 44) ------------
       The heap is t[1] = the tree root and t[Fpad + i] = leaf i, so the nodes live in
       [1, 2*Fpad) and the LEAVES are [Fpad, 2*Fpad).  The first version walked
       `for (base = 1; base < Fpad; base *= 2)` and then re-walked `[top/2, top)`, which covers
       the internal nodes TWICE and the leaves NEVER: `off[leaf]` stayed 0 and the Horner kernel
       read `fA + 0`, i.e. the root's own coefficients, as the evaluation point of every baby
       point.  Laying out every node with a nonzero degree in ascending code order keeps the
       internal offsets exactly where the generic division expects them (internal codes come
       first) and appends the leaves after them. */
    unsigned long long off = 0;
    for (size_t i = 1; i < 2 * Fpad; ++i) {
        if (Fdeg[i] == 0) continue;                       /* the padding subtree is the constant 1 */
        F.off[i] = off;
        F.sz[i] = ((unsigned long long)Fdeg[i] + 1ull) * (unsigned long long)W;
        off += F.sz[i];
    }
    F.words = off;
    if (F.words == 0) {
        std::fprintf(stderr, "%s: FATAL: the S5 forest of chunk [%llu,%llu) is empty\n",
                     NTT_PROBE_NAME, (unsigned long long)lo, (unsigned long long)top);
        std::exit(3);
    }
    CK(cudaMalloc(&F.dA, F.words * sizeof(unsigned long long)));
    for (size_t i = 1; i < 2 * Fpad; ++i) {
        if (Fdeg[i] == 0 || F.sz[i] == 0) continue;
        CK(cudaMemcpy(F.dA + F.off[i], Ft[i].data(), F.sz[i] * sizeof(unsigned long long),
                      cudaMemcpyHostToDevice));
    }
    st.forest_nodes += F.words;
    if (F.words * 8 > st.forest_bytes) st.forest_bytes = F.words * 8;
    st.chunks++;
    return 0;
}

/* the frontier: one entry per live node of the current level */
struct S5Entry {
    unsigned long long code = 0;
    size_t ncoef = 0;                  /* the row's coefficient count, TIGHT (stride 1) */
    /* THE POLYNOMIAL'S DEGREE, which is NOT the row's width (section 56).  `H mod F = H` whenever
       deg H < deg F, and the copy branch below exploits exactly that -- but testing the ROW WIDTH
       instead tested how wide the row the descendant happens to sit in is, and at a chunk's root
       that width is the whole chunk (Fdeg[code]+1) while the true dividend can be far shorter: at
       D=30030/B2=2e6 the entire giant set is one block of 68 points, so deg H = 67 against a root
       divisor of degree 2048, and the device ran 43 FULL-SIZE Newton divisions whose result is
       provably H itself.  They are the most expensive divisions in the descent -- most of
       t_generic = 6.36 s out of t_total = 7.42 s -- and they also broke the section 45.2 counting
       identity (generic+linear = 5758 against divmods_slow = 5715).  `deg` is therefore tracked
       down the frontier while `ncoef` keeps describing the layout, so no row ever changes size. */
    size_t deg = 0;
    /* WHERE THIS NODE'S ROW IS, in the frontier buffer the level's kernels read.  Without it
       every node divided the frontier's FIRST row: the first level has exactly one node, so it
       was right, and every level below it was silently wrong -- the decisions (generic/horner/
       copy/zero) do not depend on the data, so the operation counts still matched the host's
       exactly (46 = 46) while the leaf values did not.  Found with a one-level repro (P=2,
       D=10) and then read off the kernels: `s5_eval_linear_kernel` even scaled the source by the
       CHILD index (`s * sa * W`), i.e. it assumed each child had a dividend of its own, while
       both children of a node divide the SAME parent row. */
    size_t rowoff = 0;
};

/* THE FAST PATH IS ABOUT THE POLYNOMIAL, NOT ABOUT THE ROW (section 56) -- see S5Entry::deg.
   One definition, used by the `vol` accounting, the branch, the operation trace, the next
   level's row width and the per-level differential check, so the four can never disagree. */
static inline bool s5_fastpath(const S5Entry &e, size_t nc) { return e.deg < nc; }

static void s5_dev_init(S5Dev &D, PolyLayer &L, S4Reduce &red, size_t P, size_t W)
{
    D.L = &L;
    D.red = &red;
    /* ---- SIZE THE POOLS FROM THE SHAPE, NOT FROM A FIXED 2 GB (section 57) -------------------
       These were `256 << 20` WORDS = 2 GB each (the comment said "256 MB": the same units slip as
       section 54.7's forest_bytes), i.e. 4 GB on top of an arena that already holds 4257 MB of an
       8 GB card at the production shape -- so `cudaMalloc(&D.pool, ...)` failed with "CUDA error
       out of memory" and NTT_S5_ON could not be run at the real shape AT ALL.  That, not a
       numerical doubt, is why production-scale S5 had never been exercised.
       The packing pool must hold, per multiply, the two packed operands and the reducer's scratch:
       3*qN + qos*W words, and 32M words (256 MB) covers the largest production shape's
       3*4.1M + 1M = 13.3M words with room to spare.  The row pool must hold ONE division's rows:
       ra/rb/A/B (k*W each), the Newton chain's four buffers per doubling step and qrev/q/qb, i.e.
       at most ~7*P*W for k, db <= P/2 -- so 8*P*W.  Both bump allocators already fail LOUDLY,
       naming the request and the capacity, so these are bounds with a tripwire rather than guesses;
       the real peaks are printed by s5_dev_done (scratch/pool_bytes) so the next round can tighten
       them from evidence. */
    const size_t floor_words = (size_t)8 << 20;                    /* 64 MB floor for tiny shapes */
    D.scratch_words = std::max(floor_words, (size_t)32 << 20);     /* 256 MB of packing pool */
    CK(cudaMalloc(&D.scratch, D.scratch_words * sizeof(unsigned long long)));
    D.pool_words = std::max(floor_words, (size_t)8 * P * W);
    CK(cudaMalloc(&D.pool, D.pool_words * sizeof(unsigned long long)));
    /* R2 = R^2 mod N with R = 2^(64*nw): the ONE constant that turns a plain value into its
       MONTGOMERY IMAGE through the multiplier the S5 kernels have (mont_mul(x, R2) = x*R).  The
       linear leaf branch needs it because its evaluation point and its coefficients both arrive
       in the plain domain -- see the note in s5_eval_linear_kernel (section 44). */
    {
        const size_t nw = (size_t)red.nw;
        mpz_t Rr, R2;
        mpz_inits(Rr, R2, nullptr);
        mpz_set_ui(Rr, 1);
        mpz_mul_2exp(Rr, Rr, (mp_bitcnt_t)(64 * nw));
        mpz_mod(Rr, Rr, red.N);
        mpz_mul(R2, Rr, Rr);
        mpz_mod(R2, R2, red.N);
        D.hR2.assign(nw, 0ull);
        mpz_to_words(D.hR2, nw, R2);
        CK(cudaMalloc(&D.dR2, nw * sizeof(unsigned long long)));
        CK(cudaMemcpy(D.dR2, D.hR2.data(), nw * sizeof(unsigned long long),
                      cudaMemcpyHostToDevice));
        mpz_clears(Rr, R2, nullptr);
    }
}

/* THE DEVICE DESCENT.  Everything the host descent does, but with no A/B materialised on the
   host, no per-coefficient GMP, and a device-to-device copy for the degree fast path.  See the
   section header for the shape rules and the pack invariant. */
static int descent_batched_dev(PolyLayer &L, const LadderCtx &C,
                               const std::vector<std::vector<unsigned long long>> &Ft,
                               const std::vector<size_t> &Fdeg, size_t Fpad, const CPoly &H,
                               unsigned long long *dleaf_out, S5Stats &st)
{
    (void)C;
    const size_t W = L.W;
    /* the same correction as descent_batched (section 54.5): the leaf count is the F tree's
       degree, and H may be shorter than that without the descent owing fewer leaves */
    const size_t P = (Fdeg.size() > 1 && Fdeg[1] > 0) ? Fdeg[1] : (H.size() > 0 ? H.size() : 1);
    const double t0 = now_s();
    if (!L.s4 || !L.arena) {
        std::fprintf(stderr, "%s: FATAL: the device descent needs the S4 reduction and the arena\n",
                     NTT_PROBE_NAME);
        return 3;
    }
    S5Dev D;
    s5_dev_init(D, L, *L.s4->red, P, W);
    /* NTT_S5_NO_LINEAR=1 routes a degree-1 divisor through the GENERIC Newton division instead of
       the Horner shortcut.  It is a DIAGNOSTIC, and a decisive one: the linear branch is 24 of the
       46 operations on the frozen vector and it fires only at the last level, so if the leaf
       values match the host's with the shortcut off, the defect is inside the shortcut -- and if
       they do not, it is in the generic path.  Cheaper than reading either. */
    const bool no_linear = [] {
        const char *e = std::getenv("NTT_S5_NO_LINEAR");
        return e && *e && std::atoi(e) != 0;
    }();
    /* the horizon: every frontier value must fit one coefficient budget */
    size_t chunkL = 4096;
    {
        const char *e = std::getenv("NTT_S5_CHUNK");
        if (e && *e) chunkL = (size_t)std::strtoull(e, nullptr, 10);
        if (chunkL < 2) chunkL = 2;
        while (chunkL > 2 && ((unsigned long long)(P * W) * chunkL) > ((unsigned long long)192 << 20))
            chunkL >>= 1;
    }
    std::printf("s5_descent: device descent: P=%llu Fpad=%llu W=%llu leaf_bytes=%llu chunk=%llu "
                "levels=%d\n", (unsigned long long)P, (unsigned long long)Fpad,
                (unsigned long long)W, (unsigned long long)(P * W * 8),
                (unsigned long long)chunkL, (int)ceil_log2_u64((unsigned long long)Fpad));
    s2g_state("the device descent (S5)");
    unsigned long long entries_total = 0, maxent = 0;
    size_t maxval = 0;
    const double tinit = now_s();
    S5Forest F;
    /* THE FOREST DOES NOT DEPEND ON THE CHUNK (section 58): s5_forest_build lays out EVERY node
       with Fdeg != 0 in [1, 2*Fpad) -- it uses `lo`/`L` only for its error message -- so calling it
       per chunk re-copies the whole tree for every chunk.  At the production shape that is 2048
       chunks x the whole forest, which is pure repeated work (and the first thing this function
       does on every call is free and re-upload the same buffer).  Built once here, reused by every
       chunk; the device offsets are identical because the layout is keyed on Fdeg alone. */
    bool forest_built = false;
    for (size_t lo = 0; lo < Fpad; lo += chunkL) {
        const size_t Lc = ((Fpad - lo) < chunkL) ? (Fpad - lo) : chunkL;
        /* the LEAF rows this chunk owns: dleaf_out holds P rows, while a chunk covers Lc of the
           PADDED leaves (Fpad >= P, the padding comes from the product tree).  Copying/zeroing Lc
           rows ran past the caller's buffer: cudaMemcpy rejected it with "invalid argument" and
           the memset before it silently wrote out of bounds (section 30). */
        const size_t rows_out = (lo < P) ? (((size_t)(P - lo) < Lc) ? (size_t)(P - lo) : Lc) : 0;
        if (Fdeg[lo + Lc] == 0) {
            st.zeros += Lc;
            if (dleaf_out && rows_out)
                CK(cudaMemset(dleaf_out + lo * W, 0, rows_out * W * sizeof(unsigned long long)));
            continue;
        }
        if (!forest_built) {
            s5_forest_build(F, Ft, Fdeg, Fpad, lo, Lc, W, st);
            forest_built = true;
        }
        /* the forest's offsets on the device (the Horner kernel indexes them by code) */
        if (F.off.size() > D.dfoff_cap) {
            if (D.dfoff) { cudaFree(D.dfoff); D.dfoff = nullptr; D.dfoff_cap = 0; }
            CK(cudaMalloc(&D.dfoff, F.off.size() * sizeof(unsigned long long)));
            D.dfoff_cap = F.off.size();
        }
        CK(cudaMemcpy(D.dfoff, F.off.data(), F.off.size() * sizeof(unsigned long long),
                      cudaMemcpyHostToDevice));
        /* ---- THE CHUNK'S ROOT CODE, FROM THE TREE'S OWN ARITHMETIC (section 58) --------------
           With Fpad leaves at the heap codes [Fpad, 2*Fpad) and children 2c/2c+1, the node whose
           leaves are exactly [lo, lo+Lc) is
                   code = Fpad/Lc + lo/Lc,
           which is valid because chunkL is a power of two (it halves from 4096 below) and Fpad is a
           power of two >= it, so both divisions are exact and lo steps by Lc.  For Lc == Fpad this
           gives 1, the root -- the single-chunk case every earlier S5 acceptance ran, which is why
           the old `(Lc == Fpad) ? 1 : (lo/Lc + 1)` looked fine: in the multi-chunk case it returns
           1 for lo = 0, i.e. THE WHOLE TREE for a 32-leaf chunk, and the frontier then grew without
           bound (measured vol=143 against chunkL=32 at the production shape). */
        const unsigned long long ccode = (unsigned long long)(Fpad / Lc) + (unsigned long long)(lo / Lc);
        /* ---- AND ITS DIVIDEND IS `H mod F_chunk`, NOT `H` -----------------------------------
           A chunk's descent must start from the accumulator REDUCED to that sub-tree: the walk
           below divides by the chunk root's CHILDREN, and everything it produces is
           (H mod F_chunk) mod (descendants) -- feeding it the whole H (degree 51839 against a
           32-leaf chunk) is not a smaller version of the same thing, it is a different problem, and
           the degree fast path would then compare the wrong degree.  For the single-chunk case
           `H mod F = H` (the fold loop already reduced it), so this is a no-op there and the
           earlier shapes are unaffected. */
        const CPoly Hc = cp_mod(H, cp_from_flat(Ft[(size_t)ccode], Fdeg[(size_t)ccode], W), L);
        const size_t hrows = std::min((size_t)Lc, (size_t)Hc.size());
        std::vector<unsigned long long> hflat(hrows * W, 0ull);
        for (size_t i = 0; i < hrows; ++i)
            std::copy(Hc[i].begin(), Hc[i].end(), hflat.begin() + (long)(i * W));
        unsigned long long *dbound = nullptr;
        /* THE FRONTIER FEEDS ITSELF, SO THIS BUFFER MUST HOLD A WHOLE LEVEL'S ROWS (section 57).
           It was `chunkL*W`, but the level loop copies `vol` rows back into it (the frontier must
           become the next level's source), and `vol` is NOT bounded by chunkL: a copy reserves
           deg+1 rows while its sibling may be divided into nc rows, and the padded/unbalanced F
           tree does not split a chunk in half -- measured at the production shape (Fpad=65536,
           P=51840, chunkL=32, deg H=9): vol=44.  Copying 44 rows into a 32-row buffer is exactly
           the "CUDA error invalid argument" this produced.  The accounting bound below (4*chunkL)
           and this allocation are the same statement and must move together. */
        const size_t fbuf_rows = 4 * (size_t)chunkL;
        CK(cudaMalloc(&dbound, fbuf_rows * W * sizeof(unsigned long long)));
        CK(cudaMemcpy(dbound, hflat.data(), hflat.size() * sizeof(unsigned long long),
                      cudaMemcpyHostToDevice));
        if (hrows < Lc) CK(cudaMemset(dbound + hrows * W, 0, (Lc - hrows) * W * sizeof(unsigned long long)));
        std::vector<S5Entry> cur(1);
        cur[0].code = ccode;
        /* THE ROOT'S ROW WIDTH IS BOUNDED BY THE ROWS THAT EXIST (section 44).  `Fdeg[code]+1`
           is an upper bound on the coefficients of the chunk's polynomial (deg H < deg F), and
           the extra row is a harmless leading zero whenever the chunk is at least that wide --
           which is the real shape.  When it is NOT (the padded leaves are few: the one-level
           repro P=2/Fpad=2 gave ncoef=3 against a 2-row buffer), the kernel read one row past the
           allocation and Horner started on garbage.  Clamp it to what `dbound` actually holds. */
        cur[0].ncoef = std::min(Fdeg[cur[0].code] + 1, (size_t)Lc);
        /* THE ROOT'S TRUE DEGREE (section 56).  The row is padded to the chunk's width, but the
           POLYNOMIAL in it is the dividend this descent really has: for a single-chunk shape that
           is H itself (deg H = H.size()-1, zero-extended), and for a chunked shape it is
           H mod F_chunk (deg < Fdeg[code]).  The fast path must be decided on this, not on the
           padded width -- see S5Entry::deg for what testing the width cost (43 needless
           full-size divisions at D=30030, i.e. most of the descent's time). */
        cur[0].deg = std::min((size_t)Fdeg[cur[0].code],
                              Hc.size() ? (size_t)(Hc.size() - 1) : 0u);
        size_t cnt = Lc;
        int level = (int)ceil_log2_u64((unsigned long long)Lc);
        bool level_first = true;
        while (cnt > 1) {
            ++st.levels;
            std::vector<S5Entry> nxt;
            nxt.reserve(std::min(cur.size() * 2, cnt));
            const double tl0 = now_s();
            size_t vol = 0;
            for (const S5Entry &e : cur) {
                const size_t nc0 = Fdeg[2 * e.code];
                const size_t nc1 = Fdeg[2 * e.code + 1];
                if (nc0 == 0) { st.zeros++; }
                else if (s5_fastpath(e, nc0)) { st.copies++; vol += e.deg + 1; }
                else if (nc0 == 1 && !no_linear) { st.linear++; vol += 1; }
                else { st.generic++; ++st.divmods; vol += nc0; }
                if (nc1 == 0) { st.zeros++; }
                else if (s5_fastpath(e, nc1)) { st.copies++; vol += e.deg + 1; }
                else if (nc1 == 1 && !no_linear) { st.linear++; vol += 1; }
                else { st.generic++; ++st.divmods; vol += nc1; }
            }
            /* off[q] = the number of ROWS the q-th child actually receives, in emission order, so
               the destination offset of a child is the running sum of the sizes before it.
               THIS MUST MATCH THE ROW COUNT THE BRANCH WRITES -- a copy writes e.ncoef rows, a
               linear evaluation writes 1, a division writes nc -- and it must match the `vol`
               accumulation below, because `vol` is the size of the buffer those rows go into.
               The first version reserved `nc + 1` for every child, which made the offsets drift
               AHEAD of the writes and run past the end of the `vol`-sized buffer.  Measured with
               compute-sanitizer: `s5_sub_kernel` writing 8 bytes out of bounds, up to 168 bytes
               past a 24-word (192-byte) allocation, for threads 24..47 of a 48-thread launch.  In
               an uninstrumented run those writes land in the neighbouring allocation and zeroed
               the reduction's Y constant (S->dy, 24 bytes): that is why NTT_S5_ON returned 0 for
               every coefficient -- section 30. */
            std::vector<size_t> off;
            off.reserve(cur.size() * 2);
            for (const S5Entry &e : cur) {
                for (int sgn = 0; sgn < 2; ++sgn) {
                    const size_t nc = Fdeg[2 * e.code + (size_t)sgn];
                    if (nc == 0) off.push_back(0);
                    else if (s5_fastpath(e, nc)) off.push_back(e.deg + 1);
                    else if (nc == 1 && !no_linear) off.push_back(1);
                    else off.push_back(nc);
                }
            }
            vol = 0;
            for (size_t q : off) vol += q;
            if (vol > maxval) maxval = vol;
            /* THE FRONTIER'S CAPACITY IS IN WORDS, NOT ROWS: the descent writes `vol` rows of W
               words each, and the first version asked fitval() for `vol` words -- W times too
               small.  That single unit mistake is the whole reason NTT_S5_ON returned 0 for every
               coefficient: the descent's kernels wrote past this allocation (measured with
               compute-sanitizer: up to 168 bytes past a 192-byte allocation = 24 words, which is
               exactly `vol` for the frozen vector, while the same frontier needs vol*W = 72
               words), and those writes landed in the neighbouring allocation -- the reduction's
               Y constant S->dy, 24 bytes -- zeroing it, so `u = Mont(r, 0) = 0` from then on
               (section 30). */
            unsigned long long *dvals = D.fitval((size_t)(vol ? vol : 1) * W);
            {
                const double tc0 = now_s();
                for (size_t ci = 0; ci < cur.size(); ++ci) {
                    const S5Entry &e = cur[ci];
                    for (int sgn = 0; sgn < 2; ++sgn) {
                        const size_t child = 2 * e.code + (size_t)sgn;
                        const size_t nc = Fdeg[child];
                        if (nc == 0) continue;
                        size_t dstrow = 0;
                        for (size_t q = 0; q < 2 * ci + (size_t)sgn; ++q) dstrow += off[q];
                        /* the invariant that the sanitizer had to find for us: this child's rows
                           must fit inside the `vol`-sized frontier buffer.  Checked from the
                           actual sizes on every level, so a future accounting mistake is a
                           message here instead of a silent device overrun (section 30). */
                        {
                            const size_t rows_here = s5_fastpath(e, nc)
                                                         ? e.deg + 1
                                                         : ((nc == 1 && !no_linear) ? 1 : nc);
                            if (dstrow + rows_here > vol) {
                                std::fprintf(stderr, "%s: FATAL: the S5 frontier is too small: "
                                                     "child=%llu dstrow=%llu rows=%llu vol=%llu\n",
                                             NTT_PROBE_NAME, (unsigned long long)child,
                                             (unsigned long long)dstrow,
                                             (unsigned long long)rows_here, (unsigned long long)vol);
                                std::exit(3);
                            }
                        }
                        /* the operation trace is what names the failing node if a driver kill or
                           an illegal access ends the run -- the descent is the only phase that
                           otherwise prints nothing between levels */
                        if (g_s4_batched_progress)
                            std::printf("descent_dev_op: code=%llu sgn=%d child=%llu nc=%llu "
                                        "ncoef=%llu op=%s dstrow=%llu\n", e.code, sgn, child, nc,
                                        e.ncoef,
                                        s5_fastpath(e, nc) ? "copy" : (nc == 1 ? "horner" : "div"),
                                        (unsigned long long)dstrow);
                        if (s5_fastpath(e, nc)) {
                            /* the degree fast path: H mod F_ci = H, a pure device copy.
                               THE COPY IS ONLY as WIDE AS THE POLYNOMIAL (section 56): copying the
                               parent's whole row width would put `ncoef` rows into the frontier for
                               each child, and at a chunk root that width is the whole chunk -- at
                               D=30030 two such children needed vol=5762 against a 4096-row chunk
                               and the frontier invariant refused it (loudly: "the S5 frontier is
                               larger than the chunk").  `deg+1 <= nc` because the fast path means
                               deg < nc, so a copy can never need more rows than the division it
                               replaces -- the invariant `vol <= chunkL` is preserved by the same
                               argument that made the branch legal in the first place. */
                            unsigned long long nw = 0;
                            ++st.n_copy;
                            s5_memcpy_rows(dbound, e.rowoff * W, dstrow * W, e.deg + 1, e.deg + 1,
                                           (int)W, dvals, &nw);
                        } else if (nc == 1 && !no_linear) {
                            /* the linear branch: a mod (X - x_j) = a(x_j), Horner on the device.
                               One launch per CHILD, one row written. */
                            ++st.n_horner; ++st.n_launch_horner;
                            s5_eval_linear_kernel<<<S5_GRID(1)>>>(
                                dbound, (unsigned long long)(e.rowoff * W), e.ncoef, (int)W,
                                F.dA, D.dfoff,
                                (unsigned long long)child, L.s4->red->dn, L.s4->red->ninv,
                                L.s4->red->nw, D.dR2, dvals + dstrow * W, (unsigned long long)W);
                            CK(cudaGetLastError());
                        } else {
                            /* the generic branch: the Newton quotient chain, one node at a time
                               (the batched shape is uniform, but each node's dividend row is its
                               own, and every level here has far fewer nodes than the leaves).
                               `lb` IS THE DIVISOR'S COEFFICIENT COUNT, i.e. deg+1 -- the S5 shape
                               stores a monic divisor of Fdeg[child] as Fdeg[child]+1 rows.  The
                               first version passed `nc`, one too few, which made
                               s5_rev_pack_kernel treat the leading coefficient as "beyond the
                               source" and write rb[0] = 0 instead of 1: the Newton inverse was
                               then of a NON-MONIC series, so every quotient -- and therefore every
                               remainder at this level and everything below it -- was wrong
                               (section 45, found with NTT_S5_LEVEL_CHECK=1). */
                            s5_divmod_one(D, dbound, (unsigned long long)(e.rowoff * W), e.ncoef,
                                          F.dA, F.off[child], nc + 1, nc, dvals + dstrow * W, st);
                        }
                    }
                }
                CK(cudaDeviceSynchronize());
                st.t_copy += now_s() - tc0;
            }
            /* the children become the next level's frontier, in the same order */
            size_t dstrow_all = 0;
            for (const S5Entry &e : cur) {
                for (int sgn = 0; sgn < 2; ++sgn) {
                    const size_t child = 2 * e.code + (size_t)sgn;
                    const size_t nc = Fdeg[child];
                    if (nc == 0) continue;
                    S5Entry ne;
                    ne.code = child;
                    /* the row's coefficient count: the divisor's when it was divided, deg+1 when it
                       was only copied (the copy writes exactly that many rows -- see the copy
                       branch), 1 when it was evaluated */
                    ne.ncoef = s5_fastpath(e, nc) ? (e.deg + 1)
                                                  : ((nc == 1 && !no_linear) ? 1 : nc);
                    /* ... and the DEGREE the next level must decide with (section 56):
                       * a copy returns H itself, so the degree is unchanged (and the fast path was
                         only taken because it is < nc, so the invariant holds);
                       * a division returns deg R < nc, so it is capped at nc-1 -- this is what
                         keeps the cap sound level after level rather than only approximately;
                       * the linear evaluation returns the constant H(x_j). */
                    ne.deg = s5_fastpath(e, nc)
                                 ? e.deg
                                 : ((nc == 1 && !no_linear) ? 0 : std::min(e.deg, nc - 1));
                    ne.rowoff = dstrow_all;
                    dstrow_all += ne.ncoef;
                    nxt.push_back(ne);
                }
            }
            entries_total += nxt.size();
            if (nxt.size() > maxent) maxent = nxt.size();
            /* THE FRONTIER MUST BECOME THE NEXT LEVEL'S SOURCE.  The children's rows were
               written into `dvals`, and every level reads its input from `dbound` (the chunk's
               H row), so without this copy each level would re-reduce H against the SAME divisor
               -- and the final leaf read-back would hand out H's coefficients as "H(x_j)".  The
               first version of this function had no such copy at all (found by reading the data
               flow, section 30): `vol` rows move back into the frontier buffer, whose capacity
               is the chunk's own row count. */
            if (vol) {
                /* `vol` IS THE BUFFER'S SIZE -- fitval(vol*W) below allocates exactly it and the
                   pool refuses anything it cannot hold -- so this is an ACCOUNTING tripwire against
                   runaway row arithmetic, not a memory bound.  It used to demand `vol <= chunkL`,
                   which is only true for a BALANCED, unpadded chunk: at the production shape
                   (Fpad=65536, P=51840, chunkL=32, deg H = 9) a chunk's children legitimately
                   reserve 44 rows, because a copy reserves deg+1 rows while its sibling may be
                   divided into nc rows and the padded/unbalanced F tree does not split a chunk in
                   half.  Measured: `vol=44 chunkL=32`.  The bound is therefore 4*chunkL -- still
                   far below anything a real accounting error would produce (which is what section
                   30's device overruns looked like), and no longer a false alarm. */
                if (vol > 4 * (size_t)chunkL) {
                    std::fprintf(stderr, "%s: FATAL: the S5 frontier is implausibly large "
                                         "(vol=%llu chunkL=%llu)\n", NTT_PROBE_NAME,
                                 (unsigned long long)vol, (unsigned long long)chunkL);
                    std::exit(3);
                }
                /* the sticky-error trap: an ASYNC device fault inside this level (an illegal
                   access in one of the descent kernels) surfaces here as whatever the next CUDA
                   call happens to return, which is how this copy first reported "invalid
                   argument" for a 576-byte device-to-device move.  Report it here, where the
                   level and the frontier sizes are known, instead of letting it masquerade as a
                   bad pointer. */
                {
                    const cudaError_t pe = cudaGetLastError();
                    if (pe != cudaSuccess) {
                        std::fprintf(stderr, "%s: FATAL: a device error survived the S5 descent "
                                             "level (chunk_lo=%llu level=%d nodes=%llu vol=%llu): "
                                             "%s\n", NTT_PROBE_NAME, (unsigned long long)lo, level,
                                     (unsigned long long)cur.size(), (unsigned long long)vol,
                                     cudaGetErrorString(pe));
                        std::exit(3);
                    }
                }
                if (std::getenv("NTT_S5_FRONTIER_DUMP") && *std::getenv("NTT_S5_FRONTIER_DUMP")
                    && std::atoi(std::getenv("NTT_S5_FRONTIER_DUMP")) != 0) {
                    std::fprintf(stderr, "s5_frontier: chunk_lo=%llu level=%d dbound=%p dvals=%p "
                                         "vol=%llu W=%llu bytes=%llu\n", (unsigned long long)lo,
                                 level, (void *)dbound, (void *)dvals, (unsigned long long)vol,
                                 (unsigned long long)W, (unsigned long long)(vol * W * 8));
                }
                CK(cudaMemcpy(dbound, dvals, vol * W * sizeof(unsigned long long),
                              cudaMemcpyDeviceToDevice));
            }
            /* ---- THE FIRST LEVEL, AGAINST AN ORACLE THAT DOES NOT CARE WHICH BRANCH RAN -----
               (objective 2 / section 45.)  The first level has exactly ONE parent -- the chunk's
               H -- so "the device's row for child c" must equal `H mod Ft[c]`, and that identity
               is true whatever branch the device chose (division, Horner, copy, zero).  Checking
               it here splits the descent in half in one run: a mismatch means the DIVISION is
               wrong, a match means every later level's bookkeeping is.  The earlier attempt to
               answer this from the recovery dumps was read off lines that the console splits
               mid-field, which produced a false "231 of 486 rows differ" (section 45). */
            if (level_first && std::getenv("NTT_S5_LEVEL_CHECK") && *std::getenv("NTT_S5_LEVEL_CHECK")
                && std::atoi(std::getenv("NTT_S5_LEVEL_CHECK")) != 0) {
                std::vector<unsigned long long> flat((size_t)vol * W, 0ull);
                CK(cudaMemcpy(flat.data(), dbound, flat.size() * sizeof(unsigned long long),
                              cudaMemcpyDeviceToHost));
                size_t row = 0;
                unsigned long long bad_words = 0, bad_children = 0, first_child = 0;
                for (size_t ci = 0; ci < cur.size(); ++ci) {
                    for (int sgn = 0; sgn < 2; ++sgn) {
                        const size_t child = 2 * cur[ci].code + (size_t)sgn;
                        const size_t nc = Fdeg[child];
                        if (nc == 0) continue;
                        const size_t rows_here = s5_fastpath(cur[ci], nc)
                                                     ? cur[ci].deg + 1
                                                     : ((nc == 1 && !no_linear) ? 1 : nc);
                        /* the host oracle, straight from the definition */
                        const CPoly Dc = cp_from_flat(Ft[child], Fdeg[child], W);
                        const CPoly Rc = cp_mod(H, Dc, L);
                        std::vector<unsigned long long> want((size_t)nc * W, 0ull);
                        for (size_t k = 0; k < Rc.size() && k < nc; ++k)
                            std::copy(Rc[k].begin(), Rc[k].end(), want.begin() + (long)(k * W));
                        unsigned long long bw = 0;
                        for (size_t k = 0; k < (size_t)nc * W; ++k)
                            if (flat[row * W + k] != want[k]) {
                                if (!bad_words) {
                                    std::fprintf(stderr, "s5_level_bad: child=%llu word=%llu "
                                                         "device=%llu oracle=%llu\n",
                                                 (unsigned long long)child, (unsigned long long)k,
                                                 flat[row * W + k], want[k]);
                                }
                                ++bw;
                            }
                        if (bw) { ++bad_children; if (!first_child) first_child = child; }
                        if (bw && bad_children == 1)
                            for (int t = 0; t < 2; ++t)
                                std::fprintf(stderr, "s5_lvl_first: child=%llu rowoff=%llu t=%d "
                                                     "dev=%llx oracle=%llx\n",
                                             (unsigned long long)child, (unsigned long long)row, t,
                                             (unsigned long long)flat[row * W + (size_t)t],
                                             (unsigned long long)want[(size_t)t]);
                        bad_words += bw;
                        row += rows_here;
                    }
                }
                std::fprintf(stderr, "s5_level_check: parent_rows=%llu children=%llu "
                                     "bad_children=%llu bad_words=%llu first_bad_child=%llu\n",
                             (unsigned long long)cur.size(), (unsigned long long)row,
                             bad_children, bad_words, (unsigned long long)first_child);
            }
            if (g_s4_batched_progress)
                std::printf("descent_dev: chunk_lo=%llu level=%d nodes=%llu gen=%llu lin=%llu "
                            "cp=%llu z=%llu vol=%llu t=%.1f s\n", (unsigned long long)lo,
                            level, (unsigned long long)cur.size(), st.generic, st.linear,
                            st.copies, st.zeros, (unsigned long long)vol, now_s() - tl0);
            cur.swap(nxt);
            cnt /= 2;
            --level;
            level_first = false;
        }
        (void)cnt;
        if (dleaf_out && rows_out)
            CK(cudaMemcpy(dleaf_out + lo * W, dbound, rows_out * W * sizeof(unsigned long long),
                          cudaMemcpyDeviceToDevice));
        if (dbound) cudaFree(dbound);
        st.frontier_peak_bytes = std::max(st.frontier_peak_bytes, (unsigned long long)(maxval * 8));
    }
    st.scratch_bytes = (unsigned long long)((D.scratch_words + D.pool_words) * 8);
    st.t_total = now_s() - t0;
    /* the per-stage account of a division (section 21): t_generic/t_copy overlapped, these do not */
    {
        const unsigned long long nd = st.n_div ? st.n_div : 1;
        std::printf("s5_div_cost: divisions=%llu newton_steps=%llu newton_muls=%llu "
                    "t_revpack=%.2f t_newton=%.2f t_qb=%.2f t_sub=%.2f | per division: "
                    "revpack=%.0f us newton=%.0f us qb=%.0f us sub=%.0f us total=%.0f us | "
                    "branches: horner=%llu (launches=%llu) copy=%llu div_launches=%llu\n",
                    st.n_div, st.newton_steps, st.newton_muls, st.t_revpack, st.t_newton, st.t_qb,
                    st.t_sub, st.t_revpack * 1e6 / (double)nd, st.t_newton * 1e6 / (double)nd,
                    st.t_qb * 1e6 / (double)nd, st.t_sub * 1e6 / (double)nd,
                    (st.t_revpack + st.t_newton + st.t_qb + st.t_sub) * 1e6 / (double)nd,
                    st.n_horner, st.n_launch_horner, st.n_copy, st.n_launch_div);
        const unsigned long long mc = st.mul_calls ? st.mul_calls : 1;
        std::printf("s5_mul_cost: calls=%llu (first P=%llu) t_all=%.2f | per call: fwd=%.0f "
                    "inv=%.0f slot=%.0f hpack=%.0f h2d=%.0f check=%.0f (d2h=%.0f kernel=%.0f) "
                    "hplan=%.0f hopcopy=%.0f | accounted=%.2f of t_generic=%.2f\n",
                    st.mul_calls, st.mul_first_shape, st.t_mul_all, st.t_mul_fwd * 1e6 / (double)mc,
                    st.t_mul_inv * 1e6 / (double)mc, st.t_mul_slot * 1e6 / (double)mc,
                    st.t_mul_hpack * 1e6 / (double)mc, st.t_mul_h2d * 1e6 / (double)mc,
                    st.t_mul_check * 1e6 / (double)mc, st.t_mul_check_d2h * 1e6 / (double)mc,
                    st.t_mul_check_kernel * 1e6 / (double)mc, st.t_mul_hplan * 1e6 / (double)mc,
                    st.t_mul_hopcopy * 1e6 / (double)mc, st.t_mul_all, st.t_generic);
    }
    std::printf("s5_dev_done: chunks=%llu levels=%llu entries=%llu max_frontier_rows=%llu "
                "max_frontier_bytes=%llu forest_bytes_peak=%llu generic=%llu linear=%llu copies=%llu "
                "zeros=%llu ntt_launches=%llu t_pack=%.2f t_ntt=%.2f t_generic=%.2f t_copy=%.2f "
                "t_init=%.2f t_total=%.2f\n", st.chunks, st.levels,
                (unsigned long long)entries_total, (unsigned long long)maxent,
                st.frontier_peak_bytes, st.forest_bytes, st.generic, st.linear, st.copies, st.zeros,
                st.ntt_launches, st.t_pack, st.t_ntt, st.t_generic, st.t_copy, tinit - t0,
                st.t_total);
    return 0;
}

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
static void flat_mul_batch(PolyLayer &L, const std::vector<unsigned long long> &A, size_t ma,
                           const std::vector<unsigned long long> &B, size_t mb, size_t nbatch,
                           std::vector<unsigned long long> &out, int cat)
{
    const size_t W = L.W;
    const size_t P = (ma > mb) ? ma : mb;           /* the multiply takes ONE shape per launch */
    const size_t nc = ma + mb - 1;
    /* CONTRACT: every operand holds exactly nbatch slices of MA (MB) coefficients -- so its
       length is nbatch*ma*W (nbatch*mb*W) and its slice stride IS ma (mb) -- and the result is
       written TIGHTLY at nc = ma+mb-1 coefficients per slice.  All three are load-bearing, and
       all three were assumed rather than checked until the real shape broke:
         * a SHORTER operand (or one whose real slice is shorter than ma) is read past its end;
         * a LONGER one used to make the result `A.size()/ma * nc` words -- inflating each
           slice to k coefficients in the Newton step of inv_series_batch, after which
           flat_truncate was handed a slice stride it had never declared and read off the end.
       Both were host access violations at P = 4096 / W = 83.  The stride is stated once, here,
       and every caller now has to satisfy it: the only three call sites are
       inv_series_batch's two Newton multiplies and divmod_batch's qrev = ra*rbi. */
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
    std::vector<unsigned long long> wa(nbatch * P * W, 0ull), wb(nbatch * P * W, 0ull);
    for (size_t s = 0; s < nbatch; ++s) {
        std::copy(A.begin() + (long)(s * ma * W), A.begin() + (long)((s + 1) * ma * W),
                  wa.begin() + (long)(s * P * W));
        std::copy(B.begin() + (long)(s * mb * W), B.begin() + (long)((s + 1) * mb * W),
                  wb.begin() + (long)(s * P * W));
    }
    poly_mul_batch_modN(L, wa.data(), wb.data(), ma, mb, nbatch, out, cat);
    /* the multiply hands back nbatch*nc*W; the assign is what makes the TIGHT packing above a
       guarantee rather than a property of the multiply's implementation */
    if (out.size() != nbatch * nc * W) out.assign(nbatch * nc * W, 0ull);
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
        flat_mul_batch(L, at, am, (len == nxt) ? g : gpad, nxt, nbatch, ag, cat);
        /* the multiply's assertion above fixed ag's length at nbatch*(am+nxt-1)*W, which IS the
           stride declared here -- the two are the same expression, so the operand can never be
           short */
        flat_truncate(ag, nbatch, am + nxt - 1, nxt, W, "inv_series_batch: ag=a*g");
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
        flat_mul_batch(L, gpad, nxt, h, nxt, nbatch, gn, cat);
        flat_truncate(gn, nbatch, 2 * nxt - 1, nxt, W, "inv_series_batch: gn=g*h");
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
    flat_mul_batch(L, ra, k, rbi, k, nbatch, qrev, cat);
    flat_truncate(qrev, nbatch, 2 * k - 1, k, W,
                  "divmod_batch: qrev=ra*rbi");   /* mod X^k (REPACKED) */
    q.assign(nbatch * k * W, 0ull);
    for (size_t s = 0; s < nbatch; ++s)
        for (size_t i = 0; i < k; ++i)
            std::copy(qrev.begin() + (long)((s * k + (k - 1 - i)) * W),
                      qrev.begin() + (long)((s * k + (k - 1 - i) + 1) * W),
                      q.begin() + (long)((s * k + i) * W));
    flat_mul_batch(L, q, k, B, db + 1, nbatch, qb, cat);
    qb.resize(nbatch * (k + db) * W);
    out.assign(nbatch * db * W, 0ull);
    for (size_t s = 0; s < nbatch; ++s)
        for (size_t i = 0; i < db; ++i)
            cp_coeff_sub_p(&out[(s * db + i) * W], &A[(s * (da + 1) + i) * W],
                           &qb[(s * (k + db) + i) * W], L.N, W);
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
static void giant_points(PolyLayer &L, const LadderCtx &C, unsigned long long D,
                         unsigned long long B2,
                         std::vector<std::vector<unsigned long long>> &leaf, size_t &count)
{
    const unsigned long long imax = B2 / D + 2;
    count = (size_t)imax;
    std::vector<unsigned long long> js((size_t)imax);
    for (unsigned long long i = 1; i <= imax; ++i) js[(size_t)(i - 1)] = i * D;
    std::vector<unsigned long long> gx, gz;
    ladder_points(C, js, gx, gz);
    const size_t nw = C.nw;
    leaf.assign((size_t)imax, std::vector<unsigned long long>(2 * nw, 0ull));
    mpz_t X, Z, ax, neg;
    mpz_inits(X, Z, ax, neg, nullptr);
    std::vector<unsigned long long> w(nw, 0ull);
    for (size_t i = 0; i < (size_t)imax; ++i) {
        words_to_mpz(X, &gx[i * nw], nw);
        words_to_mpz(Z, &gz[i * nw], nw);
        affine_x_gmp(ax, X, Z, L.N);
        mpz_neg(neg, ax);
        mpz_mod(neg, neg, L.N);                    /* the leaf (X - x_i) = [ -x_i, 1 ] */
        mpz_to_words(w, nw, neg);
        std::copy(w.begin(), w.end(), leaf[i].begin());
        leaf[i][nw] = 1;
    }
    mpz_clears(X, Z, ax, neg, nullptr);
}

/* the S2 algorithm: remainder tree + accumulate + gcd, naming the culprit prime the way the
   reference does (an independent device ladder for every candidate p = i*D -+ j, gcd of Z) */
static Stage2Tail run_stage2_tail(PolyLayer &L, const LadderCtx &C, const Stage2Params &SP,
                                  const std::vector<unsigned long long> &Fflat, size_t fdeg)
{
    Stage2Tail out;
    const size_t W = L.W;
    const unsigned long long D = SP.D, B1 = SP.B1, B2 = SP.B2;

    std::vector<std::vector<unsigned long long>> gleaf;
    size_t ngiant = 0;
    giant_points(L, C, D, B2, gleaf, ngiant);
    FTreeStats gs;
    std::vector<size_t> gdeg;
    size_t gpad = 0;
    std::vector<std::vector<unsigned long long>> gt = build_tree_flat(L, gleaf, gdeg, gpad, gs);
    std::printf("stage2_tail: giant_points=%llu giant_muls=%llu\n",
                (unsigned long long)gs.leaves, (unsigned long long)gs.muls);

    /* the descent */
    const CPoly Fp = cp_from_flat(Fflat, fdeg, W);
    size_t divisions = 0;
    std::vector<CPoly> cur(1);
    cur[0] = cp_mod(Fp, cp_from_flat(gt[1], gdeg[1], W), L);
    size_t base = 1, cnt = 1;
    while (base < gpad) {
        const size_t nbase = base * 2;
        std::vector<CPoly> nxt;
        nxt.resize(cnt * 2);
        for (size_t j = 0; j < cnt; ++j) {
            for (int sgn = 0; sgn < 2; ++sgn) {
                const size_t ci = nbase + 2 * j + (size_t)sgn;
                /* the padding leaves are the constant 1 in the same flat form */
                if (poly_is_one(gt[ci].data(), W)) {
                    cp_resize(nxt[2 * j + (size_t)sgn], 1, W);   /* F mod 1 = 0 */
                } else {
                    if (cur[j].size() >= gdeg[ci] + 1) ++divisions;
                    nxt[2 * j + (size_t)sgn] = cp_mod(cur[j], cp_from_flat(gt[ci], gdeg[ci], W), L);
                }
            }
        }
        cur.swap(nxt);
        base = nbase;
        cnt *= 2;
    }

    /* accumulate prod_i F(x_i) mod N, gcd per block, and name the culprit like the reference */
    const size_t BLOCK = 4096;
    std::vector<std::vector<unsigned long long>> values((size_t)gs.leaves);
    for (size_t i = 0; i < (size_t)gs.leaves; ++i) values[i] = cp_to_flat(cur[i], W);

    mpz_t prod, g, gi2, pg, tq;
    mpz_inits(prod, g, gi2, pg, tq, nullptr);
    mpz_set_ui(prod, 1);
    std::vector<size_t> block;
    auto record = [&](const mpz_t f, unsigned long long prime) {
        if (mpz_cmp_ui(f, 1) <= 0 || mpz_cmp(f, L.N) == 0) return;
        mpz_mod(tq, L.N, f);
        if (mpz_cmp_ui(tq, 0) != 0) ++out.bad_factors;
        char *s = mpz_get_str(nullptr, 10, f);
        bool seen = false;
        for (const std::string &v : out.factors)
            if (v == s) seen = true;
        if (!seen) out.factors.push_back(s);
        void (*ff)(void *, size_t) = nullptr;
        mp_get_memory_functions(nullptr, nullptr, &ff);
        ff(s, std::strlen(s) + 1);
        if (prime) out.hit_primes.push_back(prime);
        ++out.hits;
    };
    auto name_culprit = [&](size_t leaf_idx) {
        mpz_t v;
        mpz_init(v);
        words_to_mpz(v, values[leaf_idx].data(), W);
        mpz_gcd(gi2, v, L.N);
        if (mpz_cmp_ui(gi2, 1) > 0 && mpz_cmp(gi2, L.N) < 0) {
            const unsigned long long i = (unsigned long long)leaf_idx + 1;
            const unsigned long long off = i * D;
            bool named = false;
            for (size_t k = 0; k < SP.baby_j.size(); ++k) {
                const unsigned long long j = SP.baby_j[k];
                for (int sgn = 0; sgn < 2; ++sgn) {
                    if (sgn == 0 && off < j) continue;
                    const unsigned long long p = (sgn == 0) ? (off - j) : (off + j);
                    if (p <= B1 || p > B2) continue;
                    if (!is_prime_u64(p)) continue;
                    std::vector<unsigned long long> pj(1, p), px, pz;
                    ladder_points(C, pj, px, pz);
                    mpz_t z;
                    mpz_init(z);
                    words_to_mpz(z, pz.data(), W);
                    mpz_gcd(pg, z, L.N);
                    if (mpz_cmp_ui(pg, 1) > 0 && mpz_cmp(pg, L.N) < 0) {
                        record(pg, p);
                        named = true;
                    }
                    mpz_clear(z);
                }
            }
            if (!named) record(gi2, 0);
        }
        mpz_clear(v);
    };
    auto flush = [&]() {
        if (block.empty()) return;
        mpz_gcd(g, prod, L.N);
        if (mpz_cmp_ui(g, 1) > 0 && mpz_cmp(g, L.N) < 0)
            for (size_t leaf_idx : block) name_culprit(leaf_idx);
        block.clear();
        mpz_set_ui(prod, 1);
    };
    for (size_t i = 0; i < values.size(); ++i) {
        mpz_t v;
        mpz_init(v);
        words_to_mpz(v, values[i].data(), W);
        mpz_mul(prod, prod, v);
        mpz_mod(prod, prod, L.N);
        mpz_clear(v);
        block.push_back(i);
        if (block.size() >= BLOCK) flush();
    }
    flush();
    std::printf("stage2_tail: descent_divisions=%llu leaf_values=%llu\n",
                (unsigned long long)divisions, (unsigned long long)values.size());
    mpz_clears(prod, g, gi2, pg, tq, nullptr);
    return out;
}

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
    unsigned long long *dvals = nullptr, *dprod = nullptr;
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
        if (dvals) cudaFree(dvals);
        if (dprod) cudaFree(dprod);
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
static void giant_chunk_chain(PolyLayer &L, const LadderCtx &C, S3Workspace &W, unsigned long long D,
                              unsigned long long clo, unsigned long long chi,
                              unsigned long long per_block, std::vector<unsigned long long> &gx,
                              std::vector<unsigned long long> &gz, bool check,
                              unsigned long long &seed_points,
                              std::vector<unsigned long long> *gseg = nullptr,
                              unsigned long long seg = 16)
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
    ladder_points_ws(W, js, lx, lz);
    /* the seeds and the difference point as MONTGOMERY IMAGES (x*R mod N) */
    std::vector<unsigned long long> hdsx((size_t)blocks * nw, 0ull), hdsz((size_t)blocks * nw, 0ull),
                                     hesx((size_t)blocks * nw, 0ull), hesz((size_t)blocks * nw, 0ull),
                                     hdx(nw, 0ull), hdz(nw, 0ull);
    {
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
    CK(cudaMalloc(&ddsx, nseed * 8));
    CK(cudaMalloc(&ddsz, nseed * 8));
    CK(cudaMalloc(&desx, nseed * 8));
    CK(cudaMalloc(&desz, nseed * 8));
    CK(cudaMalloc(&ddx, nw * 8));
    CK(cudaMalloc(&ddz, nw * 8));
    CK(cudaMalloc(&ox, (size_t)npts * nw * 8));
    CK(cudaMalloc(&oz, (size_t)npts * nw * 8));
    CK(cudaMemcpy(ddsx, hdsx.data(), nseed * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(ddsz, hdsz.data(), nseed * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(desx, hesx.data(), nseed * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(desz, hesz.data(), nseed * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(ddx, hdx.data(), nw * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(ddz, hdz.data(), nw * 8, cudaMemcpyHostToDevice));
    S2G_DISPATCH((int)nw, s2g_launch_chain, (int)nw, blocks, npts, per_block, C.ninv, W.dn, ddx,
                 ddz, ddsx, ddsz, desx, desz, ox, oz);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    gx.assign((size_t)npts * nw, 0ull);
    gz.assign((size_t)npts * nw, 0ull);
    CK(cudaMemcpy(gx.data(), ox, gx.size() * 8, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(gz.data(), oz, gz.size() * 8, cudaMemcpyDeviceToHost));
    /* the per-segment z-products, while the point buffers are still on the device (section 42) */
    unsigned long long *dsp = nullptr;
    if (gseg) {
        const unsigned long long nseg = (npts + seg - 1) / seg;
        gseg->assign((size_t)nseg * nw, 0ull);
        CK(cudaMalloc(&dsp, (size_t)nseg * nw * 8));
        S2G_DISPATCH((int)nw, s2g_launch_segprod, (int)nw, npts, seg, C.ninv, W.dn, oz, dsp);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(gseg->data(), dsp, gseg->size() * 8, cudaMemcpyDeviceToHost));
        cudaFree(dsp);
    }
    cudaFree(ddsx); cudaFree(ddsz); cudaFree(desx); cudaFree(desz);
    cudaFree(ddx); cudaFree(ddz); cudaFree(ox); cudaFree(oz);
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
        std::printf("giant_chain_check: points=%llu blocks=%llu per_block=%llu seed_points=%llu "
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
static std::vector<unsigned long long> prime_powers_u64(unsigned long long B1)
{
    std::vector<unsigned long long> out;
    if (B1 < 2) return out;
    std::vector<bool> comp((size_t)B1 + 1, false);
    unsigned long long acc = 1;
    const unsigned long long lim = 1ull << 62;
    for (unsigned long long i = 2; i <= B1; ++i) {
        if (comp[(size_t)i]) continue;
        for (unsigned long long j = i * 2; j <= B1; j += i) comp[(size_t)j] = true;
        unsigned long long pk = i;
        while (pk <= B1 / i) pk *= i;
        if (acc > lim / pk) { out.push_back(acc); acc = 1; }
        acc *= pk;
    }
    if (acc > 1) out.push_back(acc);
    return out;
}

/* Q = [prod p^e] P0 through the device ladder, one prime power per step; Q and P0 are NORMAL
   domain (X : Z) on input and output.  Returns the number of device ladders used. */
/* the WHOLE [prod pps] chain in ONE launch.  The steps are sequentially dependent (each ladder
   starts where the previous one ended), so there is nothing to parallelise ACROSS steps: exactly
   one thread runs them all, keeping the point in the MONTGOMERY domain throughout (the ladder
   never needs the normal domain, only the caller does).  This replaces 25 launches with 7 tiny
   pageable-memory copies each, which measured 12.693 s of EVERY real curve -- the copies, not
   the arithmetic, were the whole cost (section 26.6). */
template <int NW>
__global__ void s2g_ladder_chain_kernel(const unsigned long long *dps, int npps,
                                        unsigned long long ninv, int nw,
                                        const unsigned long long *dn,
                                        const unsigned long long *da24,
                                        const unsigned long long *dmone,
                                        unsigned long long *qx, unsigned long long *qz)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    unsigned long long x[NW], z[NW], rx[NW], rz[NW];
    for (int i = 0; i < nw; ++i) { x[i] = qx[i]; z[i] = qz[i]; }
    for (int s = 0; s < npps; ++s) {
        s2g_ladder<NW>(dps[s], x, z, da24, dn, ninv, nw, dmone, rx, rz);
        for (int i = 0; i < nw; ++i) { x[i] = rx[i]; z[i] = rz[i]; }
    }
    for (int i = 0; i < nw; ++i) { qx[i] = x[i]; qz[i] = z[i]; }
}

template <int NW>
static void s2g_launch_ladder_chain(int nw, const unsigned long long *dps, int npps,
                                    unsigned long long ninv, const unsigned long long *dn,
                                    const unsigned long long *da24,
                                    const unsigned long long *dmone,
                                    unsigned long long *qx, unsigned long long *qz)
{
    s2g_ladder_chain_kernel<NW><<<1, 1>>>(dps, npps, ninv, nw, dn, da24, dmone, qx, qz);
}

static unsigned long long ladder_product(const std::vector<unsigned long long> &hn, size_t nw,
                                         unsigned long long ninv, const mpz_t N,
                                         const mpz_t a24, const mpz_t R,
                                         const std::vector<unsigned long long> &pps,
                                         std::vector<unsigned long long> &qx,
                                         std::vector<unsigned long long> &qz,
                                         const std::vector<unsigned long long> &ha24,
                                         const std::vector<unsigned long long> &hmone,
                                         LadderCtx &ctx)
{
    mpz_t t, rinv;
    mpz_inits(t, rinv, nullptr);
    const int nh = (int)pps.size();
    unsigned long long *dps = nullptr, *dhn = nullptr, *dqx = nullptr, *dqz = nullptr,
                       *da24d = nullptr, *dmoned = nullptr;
    CK(cudaMalloc(&dps, (size_t)nh * 8));
    CK(cudaMalloc(&dhn, nw * 8));
    CK(cudaMalloc(&dqx, nw * 8));
    CK(cudaMalloc(&dqz, nw * 8));
    CK(cudaMalloc(&da24d, nw * 8));
    CK(cudaMalloc(&dmoned, nw * 8));
    CK(cudaMemcpy(dps, pps.data(), (size_t)nh * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dhn, hn.data(), nw * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(da24d, ha24.data(), nw * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dmoned, hmone.data(), nw * 8, cudaMemcpyHostToDevice));
    ctx.hn = hn;
    ctx.nw = nw;
    ctx.ninv = ninv;
    ctx.ha24 = ha24;
    ctx.hmone = hmone;
    ctx.hqx.assign(nw, 0ull);
    ctx.hqz.assign(nw, 0ull);
    {   /* Q enters the kernel as the MONTGOMERY IMAGE Q*R mod N (one copy, not one per step) */
        words_to_mpz(t, qx.data(), nw);
        mpz_mul(t, t, R);
        mpz_mod(t, t, N);
        mpz_to_words(ctx.hqx, nw, t);
        words_to_mpz(t, qz.data(), nw);
        mpz_mul(t, t, R);
        mpz_mod(t, t, N);
        mpz_to_words(ctx.hqz, nw, t);
    }
    CK(cudaMemcpy(dqx, ctx.hqx.data(), nw * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dqz, ctx.hqz.data(), nw * 8, cudaMemcpyHostToDevice));
    S2G_DISPATCH((int)nw, s2g_launch_ladder_chain, (int)nw, dps, nh, ninv, dhn, da24d, dmoned,
                 dqx, dqz);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    std::vector<unsigned long long> ox(nw, 0ull), oz(nw, 0ull);
    CK(cudaMemcpy(ox.data(), dqx, nw * 8, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(oz.data(), dqz, nw * 8, cudaMemcpyDeviceToHost));
    cudaFree(dps); cudaFree(dhn); cudaFree(dqx); cudaFree(dqz); cudaFree(da24d); cudaFree(dmoned);
    /* back to the plain domain: X = image * R^-1 mod N (N is odd, so R is invertible) */
    if (mpz_invert(rinv, R, N) == 0) {
        std::fprintf(stderr, "%s: the Montgomery constant is not invertible mod N\n",
                     NTT_PROBE_NAME);
        std::exit(2);
    }
    qx.assign(nw, 0ull);
    qz.assign(nw, 0ull);
    words_to_mpz(t, ox.data(), nw);
    mpz_mul(t, t, rinv);
    mpz_mod(t, t, N);
    mpz_to_words(qx, nw, t);
    words_to_mpz(t, oz.data(), nw);
    mpz_mul(t, t, rinv);
    mpz_mod(t, t, N);
    mpz_to_words(qz, nw, t);
    mpz_clears(t, rinv, nullptr);
    return (unsigned long long)nh;
}
struct BatchedRun {
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

/* the batched structure itself.  Ft/Fdeg/Fpad is the F product tree (heap, degrees, padded
   leaf count) that run_check_F already built and verified coefficient by coefficient. */
static BatchedRun run_batched(PolyLayer &L, const LadderCtx &C, const Stage2Params &SP,
                              const std::vector<std::vector<unsigned long long>> &Ft,
                              const std::vector<size_t> &Fdeg, size_t Fpad)
{
    const size_t W = L.W;
    const unsigned long long D = SP.D, B1 = SP.B1, B2 = SP.B2;
    const unsigned long long imax = B2 / D + 2;          /* the SAME giant set S2/the CPU ref use */
    const double t_entry = now_s();
    const size_t P = Fdeg[1];                            /* deg F = the baby count = poly_size */
    BatchedRun R;
    R.P = P;
    R.giant_points = imax;
    R.dbg_progress = true;                  /* one line per G-tree batch: gate-able, machine-readable */
    L.cost.S = (long)L.S;                              /* (already set by the caller) */

    mpz_t g, pg;
    mpz_inits(g, pg, nullptr);

    /* every device buffer this engine needs, allocated ONCE for the whole curve */
    S3Workspace ws;
    ws.init(C);

    /* ---- 0. primes p <= D/2 need no giant step (exactly as the CPU reference: the candidate
       set {i*D +- j, i >= 1} cannot reach them).  One batched ladder + one gcd each. ------- */
    {
        const unsigned long long half = (D < 2) ? 1 : D / 2;
        std::vector<unsigned long long> smalljs;
        for (unsigned long long p = 2; p <= half; ++p) {
            if (p <= B1 || p > B2) continue;
            if (!is_prime_u64(p)) continue;
            smalljs.push_back(p);
        }
        R.small_primes = smalljs.size();
        if (!smalljs.empty()) {
            std::vector<unsigned long long> sx, sz;
            ladder_points_ws(ws, smalljs, sx, sz);
            for (size_t k = 0; k < smalljs.size(); ++k) {
                mpz_t z;
                mpz_init(z);
                words_to_mpz(z, &sz[k * W], W);
                mpz_gcd(pg, z, L.N);
                if (mpz_cmp_ui(pg, 1) > 0 && mpz_cmp(pg, L.N) < 0)
                    s3_record(R.tail, pg, smalljs[k], L.N);
                mpz_clear(z);
            }
        }
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
    const CPoly Fpoly = cp_from_flat(Ft[1], Fdeg[1], W);
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
        gseg.assign(((size_t)npts + S2G_GFINV_SEG - 1) / S2G_GFINV_SEG * (size_t)C.nw, 0ull);
        if (force_ladder || npts < chain_min) {
            gjs.resize(chi - clo + 1);
            for (size_t i = clo; i <= chi; ++i) gjs[i - clo] = (unsigned long long)i * D;
            ladder_points_ws(ws, gjs, gx, gz);
            gfinv_segprod_host(gseg, (size_t)npts, (size_t)C.nw, gz, L.N);
        } else {
            unsigned long long seed_points = 0;
            giant_chunk_chain(L, C, ws, D, (unsigned long long)clo, (unsigned long long)chi,
                              chain_block, gx, gz, chain_check, seed_points, &gseg,
                              S2G_GFINV_SEG);
            R.giant_seed_points += seed_points;
            ++R.giant_chain_chunks;
        }
        R.t_giant += now_s() - tgp;
        /* the G trees of the batches that lie inside this point chunk */
        for (unsigned long long b = c0 / P; b < R.num_poly_g && b * P < c1; ++b) {
        const size_t lo = (size_t)(b * (unsigned long long)P);
        size_t hi = lo + P;
        if (hi > (size_t)imax) hi = (size_t)imax;
        std::vector<std::vector<unsigned long long>> bleaf(hi - lo,
                                                           std::vector<unsigned long long>(2 * W, 0ull));
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
                const double t1 = now_s();
                words_to_mpz(pseg, &gseg[sidx * W], W);
                const double t2 = now_s();
                const bool clean = (mpz_invert(invp, pseg, L.N) != 0);
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
                    for (size_t q = la; q < lend; ++q) {
                        words_neg_mod_n(bleaf[q + lbase - lo], 0, &gx[q * W], C.hn.data(), W);
                        std::copy(&gz[q * W], &gz[q * W] + (long)W,
                                  bleaf[q + lbase - lo].begin() + (long)W);
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
                    words_to_mpz(zv[j], &gz[q * W], W);
                    if (mpz_sgn(zv[j]) == 0) { seg_clean = false; break; }
                    mpz_mul(pv[j + 1], pv[j], zv[j]);
                    mpz_mod(pv[j + 1], pv[j + 1], L.N);
                }
                if (seg_clean && mpz_invert(pinv, pv[nb], L.N) == 0) seg_clean = false;
                if (!seg_clean) {
                    for (size_t j = 0; j < nb; ++j) {
                        const size_t q = la + j, bi = q + lbase - lo;
                        words_to_mpz(X, &gx[q * W], W);
                        words_to_mpz(Z, &gz[q * W], W);
                        if (!affine_x_gmp_checked(ax, X, Z, L.N)) {
                            ++R.giant_degenerate;
                            mpz_gcd(gq, Z, L.N);
                            s3_record(R.tail, gq, 0, L.N, /*count_hit=*/false);
                        }
                        mpz_neg(neg, ax);
                        mpz_mod(neg, neg, L.N);
                        mpz_to_words(w, W, neg);
                        std::copy(w.begin(), w.end(), bleaf[bi].begin());
                        bleaf[bi][W] = 1;
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
                    words_to_mpz(X, &gx[q * W], W);
                    mpz_mul(ax, X, zinv);
                    mpz_mod(ax, ax, L.N);
                    mpz_neg(neg, ax);
                    mpz_mod(neg, neg, L.N);          /* the leaf (X - x_i) = [ -x_i, 1 ] */
                    mpz_to_words(w, W, neg);
                    std::copy(w.begin(), w.end(), bleaf[bi].begin());
                    bleaf[bi][W] = 1;
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
        std::vector<std::vector<unsigned long long>> gt =
            build_tree_flat(L, bleaf, bdeg, bpad, bs, BC_GTREE);
        R.t_gtrees += now_s() - tg0;
        gs.leaves += bs.leaves;
        gs.padded += bs.padded;
        gs.muls += bs.muls;
        if (b == 0) { H = cp_from_flat(gt[1], bdeg[1], W); continue; }
        /* ---- the fold: H <- (G*H) mod F, three full-size multiplies ---- */
        const double tf0 = now_s();
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
            CPoly qrev = cp_mul(ra, rbi, L);
            cp_resize(qrev, k, W);
            CPoly q;
            cp_resize(q, k, W);
            for (size_t i = 0; i < k; ++i) q[i] = qrev[k - 1 - i];
            const CPoly qb = cp_mul(q, Fpoly, L);
            cp_resize(H, P, W);                        /* r = T - q*F has degree < P */
            for (size_t i = 0; i < P; ++i)
                cp_coeff_sub(H[i], T[i], (i < qb.size()) ? qb[i] : cp_zero(W), L.N, W);
            cp_trim(H);
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
            std::printf("batched_progress: batch=%llu/%llu (%.1f%%) t=%.1f s left=%.1f s "
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
    /* ---- UNDO THE PROJECTIVE SCALE (section 42) ------------------------------------------
       The projective leaves multiplied the tree by Gamma, and the fold carries a constant
       straight through (H <- (G*H) mod F scales by the same constant), so H left the loop as
       Gamma * prod_b f_b mod F.  ONE pass over H's coefficients by Gamma^-1 restores the exact
       polynomial the old monic leaves produced -- not approximately: every step of the tree is
       reduced mod N coefficient by coefficient, so the scaling is exact in that ring, and
       gcd(v, N) is unchanged by an invertible factor either way. */
    if (mpz_cmp_ui(Ginv, 1) != 0 && !H.empty()) {
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
    std::printf("descent_begin: P=%llu levels=%d last_state=(%s)\n", (unsigned long long)P,
                (int)ceil_log2_u64((unsigned long long)Fpad), g_last_state);
    s2g_state("the batched descent");            /* a driver kill here leaves this behind */
    ws.need_vals((size_t)P);
    std::vector<std::vector<unsigned long long>> values;
    bool dev_leaves = false;
    {
        const char *e5 = std::getenv("NTT_S5_ON");
        /* OPT-IN, and now for a MEASURED reason rather than a known defect: the device descent is
           leaf-for-leaf equal to the host descent on every shape tested (section 53: P=24 at
           D=210 -> differing_leaves=0, mismatching_coefficients=0, and the same factor and hit
           prime as the CPU reference; enforced from section 53 on by test group [9] of
           tools/test/test_stage2_tree_gpu.ps1), but it has never been run at PRODUCTION shape
           (P=51840, D=570570), and the default must not change on the strength of a 24-leaf
           vector alone.  Turning it on is now a verification exercise, not a bug hunt.

           HISTORY, kept because each entry was a real, silent defect found only by differential
           comparison -- the list is the argument for why the default stayed off this long:
             (1) SLOT LAYOUT, fixed in two steps.  The shape did not force the multiply's bpw to
                 agree with the packer's slot width, so slot_stride came out WIDER than slot_bits
                 and a coefficient's block ran past its own reduction window (measured: a 261-bit
                 value against slot_bits = 260 on the frozen vector's first division).  The first
                 fix demanded bpw | slot_bits; that was WRONG in a way no aggregate number showed
                 -- when 2S + ceil(log2 P) is prime the only admissible bpw is 1, and bpw = 1 makes
                 the carry's chain length O(N) (measured as "the S5 descent needs 128 carry rounds",
                 section 53.5).  The correct rule is the one in S5Shape [A1]: the reducer's window
                 is slot_words*bpw = stride bits wide, so ANY stride >= slot_bits is exact, and the
                 slot is rounded UP to a multiple of bpw instead of the bpw being rounded down to a
                 divisor of the slot (section 54).
             (2) PACKER ADDRESSING, fixed.  The packer wrote coefficient i at digit i (bit i*bpw)
                 instead of at its own slot (digit i*stride/bpw), 37x too far left at S=129;
                 measured as a coefficient of 2^192 read back as 2^64.
             (3) REDUCTION READ WINDOW, fixed.  s4_reduce_kernel converts the window into base-2^64
                 limbs at t[0..nlimb-1] and returns t[L..L+nw-1], which needs BOTH L + nw >= nlimb
                 (containment) and v >= N*2^(64(L-1)) (magnitude); the host's L solved only the
                 magnitude one, so at some shapes every coefficient came back 0
                 (`s4_reduce_CHECK_bad: gmp=... gpu=0`).  Section 51 records the invariant and the
                 fix; the per-leaf check is what proves it, which is why the check is permanent. */
        const bool s5_on = L.s4 && e5 && *e5 && std::atoi(e5) != 0;
        if (s5_on) {
            /* SLICE S5: the descent itself on the device.  Its leaf values stay in ws.dvals --
               materialising them on the host is 4.4 GB at the real shape for no reason, so the
               host copy is made only when a block actually shares a factor with N (the same
               laziness dev_block_products already relies on). */
            S5Stats s5;
            L.cat = BC_DESCENT;
            const int rc = descent_batched_dev(L, C, Ft, Fdeg, Fpad, H, ws.dvals, s5);
            L.cat = -1;
            if (rc != 0) {
                std::fprintf(stderr, "%s: the device descent failed (rc=%d)\n", NTT_PROBE_NAME, rc);
                std::exit(3);
            }
            R.descent_divmods = s5.divmods;
            R.s5_divmods = s5.divmods;
            R.s5_generic = s5.generic;
            R.s5_linear = s5.linear;
            R.s5_copies = s5.copies;
            R.s5_zeros = s5.zeros;
            R.s5_ntt_launches = s5.ntt_launches;
            R.s5_forest_bytes = s5.forest_bytes;
            R.s5_frontier_bytes = s5.frontier_peak_bytes;
            R.s5_t_ntt = s5.t_ntt;
            R.s5_t_pack = s5.t_pack;
            R.s5_t_generic = s5.t_generic;
            R.s5_t_copy = s5.t_copy;
            dev_leaves = true;
            std::printf("descent_dev_stats: divmods=%llu generic=%llu linear=%llu copies=%llu "
                        "zeros=%llu ntt_launches=%llu forest_bytes_peak=%llu frontier_bytes_peak=%llu "
                        "t_ntt=%.2f t_pack=%.2f t_generic=%.2f t_copy=%.2f\n",
                        s5.divmods, s5.generic, s5.linear, s5.copies, s5.zeros, s5.ntt_launches,
                        s5.forest_bytes, s5.frontier_peak_bytes, s5.t_ntt, s5.t_pack, s5.t_generic,
                        s5.t_copy);
        } else if (L.s4) {
            values.assign((size_t)P, std::vector<unsigned long long>(W, 0ull));
            /* slice S4: the whole descent, level by level and batched (at P = 92160 this is
               ~1.8e5 divmods, 1.4e5 of them with k <= 2) */
            L.cat = BC_DESCENT;
            descent_batched(L, Ft, Fdeg, Fpad, H, values, R.descent_divmods, BC_DESCENT);
            L.cat = -1;
        } else {
            values.assign((size_t)P, std::vector<unsigned long long>(W, 0ull));
            L.cat = BC_DESCENT;
            descent_slow(L, Ft, Fdeg, Fpad, H, values, R.descent_divmods);
            L.cat = -1;
        }
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
                            std::printf("descent_check_bad: i=%llu word=%llu batched=%llu "
                                        "slow=%llu\n", (unsigned long long)i,
                                        (unsigned long long)q, values[i][q], ref[i][q]);
                        }
                        ++bad;
                        dl = true;
                    }
                if (dl) { ++badleaves; if (nfirst < 8) firsts[nfirst++] = (unsigned long long)i; }
            }
            std::printf("descent_check_leaves: P=%llu differing_leaves=%llu first8=", (unsigned long long)P,
                        badleaves);
            for (unsigned long long z = 0; z < nfirst; ++z)
                std::printf("%s%llu", z ? "," : "", firsts[z]);
            std::printf("\n");
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
                std::printf("descent_check_values: i=%llu slow=%s batched=%s\n",
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
                    std::printf("descent_check_domain: slow/batched = %s = R^%d\n", sr, kbest);
                    free(sr);
                } else {
                    std::printf("descent_check_domain: the device's leaf value is NOT invertible "
                                "mod N\n");
                }
                mpz_clears(a, b, ai, ratio, R, pw, Rinv, nullptr);
            }
            std::printf("descent_check: P=%llu divmods_batched=%llu divmods_slow=%llu "
                        "mismatching_coefficients=%llu first=%llu\n", (unsigned long long)P,
                        R.descent_divmods, dm, bad, first);
        }
    }
    R.leaf_values = P;
    R.t_descent = now_s() - td0;
    /* a phase marker, because the descent is where a long shape can look hung: everything after
       it used to print nothing until the final summary line */
    if (R.dbg_progress)
        std::printf("batched_phase: descent_done t=%.1f s divmods=%llu\n", R.t_descent,
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
        std::printf("batched_phase: block_products_done blocks=%llu t=%.1f s\n",
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
        std::printf("naming_policy: name_hits=%d budget_blocks=%llu (2*imax=%llu primality tests "
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
                std::printf("batched_phase: naming_begin block=%llu leaves=[%llu,%llu) t=%.1f s\n",
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
                std::printf("batched_phase: naming_budget_exhausted at block=%llu hit_blocks=%llu "
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
                std::printf("batched_phase: naming_end block=%llu hit_leaves=%llu candidates=%llu "
                            "leaf_hits=%llu records=%llu hit_blocks=%llu t_name=%.1f s t=%.1f s\n",
                            (unsigned long long)b, blk_leaves, blk_cands, blk_leafhits, blk_rec,
                            (unsigned long long)R.hit_blocks, R.t_name, now_s() - ta0);
        }
    }
    if (R.dbg_progress)
        std::printf("tail_counts: hits=%llu unnamed=%llu factors=%llu hit_primes=%llu "
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
    std::printf("batched_naming: hit_blocks=%llu hit_leaves=%llu named_searches=%llu "
                "candidates_tested=%llu unnamed=%llu t_scan=%.3f t_ladder=%.3f t_name=%.3f "
                "name_max=%lld\n",
                R.hit_blocks, R.hit_leaves, R.named_searches, R.candidates_tested, R.unnamed,
                R.t_scan, R.t_ladder, R.t_name, name_max());
    mpz_clears(g, pg, nullptr);
    /* close the books (section 41): pre + loop_wall + post must equal the caller's `elapsed` */
    R.t_post_loop = now_s() - R.t_post_loop;
    /* the device-side counters: how much of the orchestration actually happened once */
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
static bool real_run_words(unsigned long long P, int S, unsigned long long *out_words,
                           unsigned long long *n_fold, unsigned long long *n_tree)
{
    bool ok1 = false, ok2 = false;
    const unsigned long long w1 = real_shape_words(P + 1, S, &ok1);
    const unsigned long long w2 = real_shape_words(P / 2 + 1, S, &ok2);
    if (!ok1 || !ok2) return false;
    unsigned long long a = 0, b = 0, sb = 0, ob = 0;
    int bp = 0;
    ntt_shape_query(P + 1, S, &a, &bp, &sb, &b, &ob, &ob);
    ntt_shape_query(P / 2 + 1, S, &b, &bp, nullptr, nullptr, nullptr, nullptr);
    if (n_fold) *n_fold = a;
    if (n_tree) *n_tree = b;
    /* the run also caches EVERY SMALLER N (the tree's lower levels and the descent): their sum
       is bounded by 2x the tree's own entry, so that is the second term.  Counting only two
       entries (the first version) under-estimated the footprint by ~2x and made the run fall
       back to per-call cudaMalloc at the largest shape, where the allocation then failed. */
    *out_words = w1 + 2 * w2;
    return true;
}

/* the D scan's candidate record lives with the scan itself (section 59); the old `DChoice`, which
   only carried a memory footprint, was replaced by a cost+coverage model */

static int run_real(const char *n_str, bool n_is_hex, unsigned long long sigma,
                    unsigned long long B1, unsigned long long B2, unsigned long long D_in,
                    bool choose_d, bool run_s2, int curves)
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
    const size_t cap = (freeb > reserve) ? (freeb - reserve) : (freeb / 2);
    std::printf("stage2_real: device=%d (%s) N_bits=%ld sigma=%llu B1=%llu B2=%llu "
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

       There is therefore an interior optimum, and it must be weighted by what a curve actually
       covers: stage 2 tests the residues +-j (mod D) for j <= P, i.e. the phi(D) UNITS of D, so the
       fraction of the (B1,B2] primes it can see is phi(D)/D.  A D with many small prime factors is
       a WORSE curve per unit of B2 -- that is the trade Prime95's `efficiency` score and its
       "Curve is worth 2.63 ... curves" line make explicit (section 18) and we had no equivalent of.

       THERE IS NO COVERAGE TRADE -- EVERY CANDIDATE COVERS ESSENTIALLY ALL OF (B1,B2], and the first
       version of this model got that wrong in a way worth recording here, because the measurement
       caught it immediately.  It weighted each candidate by phi(D)/D, the density of the UNITS among
       the D residues (0.18 for D=570570), which ranked the prime power D=19^4=130321 (phi/D=0.947)
       far above the baseline -- and the run it produced was 2x SLOWER (measured 127.00 s against
       62.93 s at B2=1e11).  The error: a PRIME p > D that does not divide D is automatically
       COPRIME to D, so its residue p mod D is always one of the units, and the baby set
       {+-j : j <= P, gcd(j,D) = 1} is EXACTLY the set of units.  Every prime in (B1,B2] larger than
       D is therefore covered whatever phi(D)/D is; the primes that can be missed are those in
       (P, D], whose count is pi(D)-pi(P) ~ 1e4 against pi(1e11) ~ 4e9, i.e. a fraction ~1e-5.
       So the choice is a PURE COST question -- minimise (loop + tree + giant + glue) -- and the
       table reports the missed-prime estimate as information only.

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
        std::printf("d_scan: candidates=%llu (47-smooth D <= %llu) ; model fitted to the section "
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
            if (!real_run_words(c.P, (int)L.S, &tw, nullptr, nullptr)) continue;
            c.total_words = tw;
            c.fits = ((double)tw * 8.0 <= (double)cap);
            if (!c.fits) continue;
            if (!best.D || c.total < best.total) best = c;
            if (shown < 14) {
                /* the missed-prime estimate: only (P, D] can be missed, and pi(x) ~ x/ln x */
                const double missed = (double)c.D / std::log((double)c.D) -
                                      (double)c.P / std::log((double)(c.P > 2 ? c.P : 3));
                std::printf("d_scan_choice: D=%llu P=phi/2=%llu imax=%llu batches=%llu "
                            "units=%.4f missed_primes~%.0f fit=%.0f MB | loop=%.1f tree=%.1f "
                            "giant=%.1f glue=%.1f total=%.1f rate=%.3g vals/s\n",
                            c.D, c.P, c.imax, c.batches, c.units, missed,
                            (double)tw * 8.0 / 1048576.0, c.loop, c.tree, c.giant, c.glue, c.total,
                            c.rate);
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
            const bool okm = real_run_words(r.P, (int)L.S, &tw, nullptr, nullptr);
            std::printf("d_scan_reference: D=%llu P=%llu imax=%llu batches=%llu units=%.4f "
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
            std::printf("d_scan_decision: D=%llu P=phi(D)/2=%llu (the CHEAPEST admissible shape by "
                        "the fitted cost model -- NOT the largest D that fits, which is what this "
                        "flag used to pick; see the header comment)\n", D, best.P);
        } else {
            P_baby = phi_u64(D) / 2;
            bool ok1 = false, ok2 = false;
            const unsigned long long w1 = real_shape_words(P_baby, (int)L.S, &ok1);
            const unsigned long long w2 = real_shape_words(P_baby / 2 + 1, (int)L.S, &ok2);
            std::printf("d_budget: free=%.0f MB reserve=%.0f MB cap=%.0f MB ; D=%llu "
                        "P=phi(D)/2=%llu largest_transform=%.0f MB total_needed=%.0f MB "
                        "fits=%d%s\n", freeb / 1048576.0, reserve / 1048576.0,
                        cap / 1048576.0, D, P_baby,
                        (double)w1 * 8.0 / 1048576.0, (double)(w1 + 2 * w2) * 8.0 / 1048576.0,
                        ((double)(w1 + (ok2 ? 2 * w2 : 0)) * 8.0 <= (double)cap) ? 1 : 0,
                        ok1 ? "" : " (SHAPE REFUSED BY THE MULTIPLY)");
        }
    }
    P_baby = phi_u64(D) / 2;
    /* the baby set: j COPRIME TO D, j <= D/2, ascending (the CPU reference's own rule).  Not
       "coprime to N": the first version of this filtered by gcd(N,j) and produced 1155 points
       for D=2310 instead of 240. */
    std::vector<unsigned long long> baby_j;
    for (unsigned long long j = 1; j <= D / 2; ++j)
        if (gcd_u64(j, D) == 1) baby_j.push_back(j);
    if (baby_j.size() != P_baby) {
        std::fprintf(stderr, "%s: FATAL: baby set has %llu points but phi(D)/2 = %llu\n",
                     NTT_PROBE_NAME, (unsigned long long)baby_j.size(), P_baby);
        return 3;
    }
    const unsigned long long imax = B2 / D + 2;
    /* B1 and B2 are printed RAW, not as a derived count: --b2 used to be read with strtoull
       base 10, so "1e11" silently became 1 and the run quietly computed a B2=1 shape
       (giant_points=2) while every other line looked normal -- caught only because
       giant_points was printed at all (section 27).  Both are parsed as floating point now. */
    std::printf("real_shape: D=%llu P=phi(D)/2=%llu baby_points=%llu giant_points=%llu "
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

    /* ---- Q = [lcm(1..B1)] P0 through the device ladder chain ---- */
    std::vector<unsigned long long> pps = prime_powers_u64(B1);
    std::vector<unsigned long long> qx(nw), qz(nw);
    mpz_to_words(qx, nw, ax);
    mpz_to_words(qz, nw, az);
    {
        LadderCtx chain_ctx;
        const double tc0 = now_s();
        ladder_product(hn, nw, ninv, L.N, a24, R, pps, qx, qz, ha24, hmone, chain_ctx);
        std::printf("real_setup: sigma=%llu suyama=ok prime_powers=%llu ladder_chain_seconds="
                    "%.3f\n", sigma, (unsigned long long)pps.size(), now_s() - tc0);
    }
    {
        mpz_t X, Z;
        mpz_inits(X, Z, nullptr);
        words_to_mpz(X, qx.data(), nw);
        words_to_mpz(Z, qz.data(), nw);
        affine_x_gmp(tmp, X, Z, L.N);
        char *s = mpz_get_str(nullptr, 16, tmp);
        std::printf("real_setup_Q: Q_x_hex=%.64s... (%zu hex digits)\n", s, std::strlen(s));
        void (*ff)(void *, size_t) = nullptr;
        mp_get_memory_functions(nullptr, nullptr, &ff);
        ff(s, std::strlen(s) + 1);
        mpz_clears(X, Z, nullptr);
    }
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
    const char *env4 = std::getenv("NTT_S4_OFF");
    const bool s4_on = !(env4 && *env4 && std::atoi(env4) != 0);
    {
        const char *es = std::getenv("NTT_S4_SAMPLE");
        if (es && *es) g_s4_sample_limit = std::atoll(es);
        const char *ee = std::getenv("NTT_S4_CHECK_EVERY");
        if (ee && *ee) g_s4_check_every = (unsigned long long)std::strtoull(ee, nullptr, 10);
    }
    L.arena = &arena;
    if (s4_on) {
        s4_reduce_init(red, L.N, nw, hn, ninv);
        s4.red = &red;
        L.s4 = &s4;
    }
    {
        bool ok1 = false, ok2 = false;
        const double w1 = (double)real_shape_words(P_baby, (int)L.S, &ok1);
        const double w2 = (double)real_shape_words(P_baby / 2 + 1, (int)L.S, &ok2);
        std::printf("mem_budget: free=%.0f MB total=%.0f MB reserve=%.0f MB arena_cap=%.0f MB ; "
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
    {
        const double t0 = now_s();
        std::vector<unsigned long long> bx, bz;
        unsigned long long noninv = 0;              /* printed as real_baby: ... degenerate= */
        const double tb0 = now_s();
        ladder_points(C, baby_j, bx, bz);
        const double tb1 = now_s();
        std::vector<std::vector<unsigned long long>> leaf(baby_j.size());
        {
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
                        if (!affine_x_gmp_checked(xj, X, Z, L.N)) {
                            ++noninv;
                            mpz_gcd(gq, Z, L.N);
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
                /* Montgomery's trick, walking BACKWARDS: iprod is 1/(z_j..z_{seg-1}) at each step,
                   so x_j = X_j * iprod, and then iprod *= z_j moves it one point back. */
                for (size_t j = seg; j-- > 0;) {
                    words_to_mpz(X, &bx[(lo + j) * nw], nw);
                    mpz_mul(tmul, X, iprod);
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
                std::printf("baby_degenerate: points=%llu of %llu have Z sharing a factor with N "
                            "-> their gcd was recorded as a factor\n", noninv,
                            (unsigned long long)baby_j.size());
            mpz_clears(X, Z, xj, neg, gq, nullptr);
        }
        std::printf("ladder: baby_points=%llu (device x_j; there is no CPU reference at this "
                    "shape, see the report)\n", (unsigned long long)baby_j.size());
        /* the setup pieces that live OUTSIDE `elapsed` had no timer at all (section 25): they are
           ~44 s of the 317 s wall clock, so naming them is the first step */
        std::printf("real_baby: points=%llu ladder=%.3f s affine=%.3f s degenerate=%llu\n",
                    (unsigned long long)baby_j.size(), tb1 - tb0, now_s() - tb1, noninv);
        FTreeStats fs;
        Ft = build_tree_flat(L, leaf, Fdeg, Fpad, fs, BC_FTREE);
        s4_oracle_drain(red);    /* include final F-tree validation in its phase timer */
        fdeg = Fdeg[1];
        std::printf("ftree_real: leaves=%llu padded=%llu muls=%llu ntt_calls=%llu "
                    "ntt_seconds=%.3f\n", (unsigned long long)fs.leaves,
                    (unsigned long long)fs.padded, (unsigned long long)fs.muls, L.ntt_calls,
                    L.ntt_seconds);
        t_f = now_s() - t0;
    }
    std::printf("exactness: binding_shape P=%llu S=%llu slot_bits=%llu slot_words=%llu "
                "L=P*slot_words=%llu bpw=%d ; L*(2^bpw-1)^2 = 2^%.3f < p = 2^%.3f -> %s\n",
                L.bind_P, (unsigned long long)L.S, L.bind_slot_bits, L.bind_slot_words,
                L.bind_L, L.bind_bpw, L.bind_bound_bits, mpz_log2d_p(),
                exact_ok_terms(L.bind_L, L.bind_bpw) ? "OK" : "VIOLATED");

    /* ---- the tails ---- */
    Stage2Params SP;
    SP.D = D; SP.B1 = B1; SP.B2 = B2; SP.baby_j = baby_j;
    for (int cv = 0; cv < curves; ++cv) {
        if (run_s2) {
            const unsigned long long nb = L.ntt_calls;
            const double ns = L.ntt_seconds;
            NttArena *save = L.arena;
            L.arena = nullptr;
            const double t0 = now_s();
            const Stage2Tail tail = run_stage2_tail(L, C, SP, Ft[1], fdeg);
            s4_oracle_drain(red);
            const double el = now_s() - t0;
            L.arena = save;
            std::string fs2, ps2;
            for (size_t i = 0; i < tail.factors.size(); ++i) { if (i) fs2 += ","; fs2 += tail.factors[i]; }
            for (size_t i = 0; i < tail.hit_primes.size(); ++i) { if (i) ps2 += ","; ps2 += std::to_string(tail.hit_primes[i]); }
            std::printf("stage2: algorithm=tree_gpu curves=1 hits=%llu bad_factors=%llu "
                        "factors=%s hit_primes=%s elapsed=%.2f\n", tail.hits, tail.bad_factors,
                        fs2.c_str(), ps2.c_str(), el);
            std::printf("stage2_real_s2: ntt_calls=%llu ntt_seconds=%.3f\n",
                        L.ntt_calls - nb, L.ntt_seconds - ns);
        }
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
        BatchedRun BR = run_batched(L, C, SP, Ft, Fdeg, Fpad);
        s4_oracle_drain(red);    /* validation must finish BEFORE elapsed and success */
        /* merge the factors that degenerate BABY points revealed (objective 3): each already
           divides N by construction, and they are deduplicated against what the naming stage
           found, so the reported fact set cannot change for a shape where the naming stage
           already reaches them. */
        for (const std::string &s : baby_deg)
            if (std::find(BR.tail.factors.begin(), BR.tail.factors.end(), s) ==
                BR.tail.factors.end())
                BR.tail.factors.push_back(s);
        const double el = now_s() - t0;
        std::string fs2, ps2;
        for (size_t i = 0; i < BR.tail.factors.size(); ++i) { if (i) fs2 += ","; fs2 += BR.tail.factors[i]; }
        for (size_t i = 0; i < BR.tail.hit_primes.size(); ++i) { if (i) ps2 += ","; ps2 += std::to_string(BR.tail.hit_primes[i]); }
        std::printf("stage2: algorithm=tree_gpu_batched curves=1 hits=%llu bad_factors=%llu "
                    "factors=%s hit_primes=%s elapsed=%.2f\n", BR.tail.hits, BR.tail.bad_factors,
                    fs2.c_str(), ps2.c_str(), el);
        if (BR.tail.unnamed_hits)
            std::printf("stage2_naming: named_hits=%llu unnamed_hits=%llu (the naming budget "
                        "stopped the culprit scan; hits and bad_factors are complete, the "
                        "hit_primes list is not -- NTT_NAME_HITS=1 forces naming)\n",
                        BR.tail.hits - BR.tail.unnamed_hits, BR.tail.unnamed_hits);
        std::printf("real_batched_shape: P=%llu giant_points=%llu num_poly_g=%llu loops=%llu "
                    "descent_divmods=%llu\n", BR.P, BR.giant_points, BR.num_poly_g, BR.loops,
                    BR.descent_divmods);
        std::printf("real_batched_cost: poly_muls=%llu operand_bits=%llu f_tree=%llu g_tree=%llu "
                    "fold=%llu descent=%llu inv=%llu total=%llu\n", L.cost.tot_muls(),
                    L.cost.tot_bits(), L.cost.bits[BC_FTREE], L.cost.bits[BC_GTREE],
                    L.cost.bits[BC_FOLD], L.cost.bits[BC_DESCENT], L.cost.bits[BC_FINV],
                    L.cost.tot_bits());
        std::printf("real_batched_split: giant=%.3f gtrees=%.3f fold=%.3f descent=%.3f inv=%.3f "
                    "accum=%.3f name=%.3f f_tree_incl=%.3f\n", BR.t_giant, BR.t_gtrees, BR.t_fold,
                    BR.t_descent, BR.t_inv, BR.t_accum, BR.t_name, t_f);
        /* THE BOOKS, CLOSED (section 41): pre + loop_wall + post == `el` above, and the only
           part of the loop body that no phase timer owned is the host affine conversion of the
           giant points (`gleaves`).  loop_host = loop_wall - giant - gtrees - fold - gleaves is
           then everything else the host does inside the loop with the device idle. */
        std::printf("real_batched_wall: pre=%.3f loop_wall=%.3f post=%.3f sum=%.3f (elapsed=%.2f) "
                    "| gleaves=%.3f loop_host=%.3f\n", BR.t_pre_loop, BR.t_loop_wall, BR.t_post_loop,
                    BR.t_pre_loop + BR.t_loop_wall + BR.t_post_loop, el, BR.t_gleaves,
                    BR.t_loop_wall - BR.t_giant - BR.t_gtrees - BR.t_fold - BR.t_gleaves);
        std::printf("real_batched_gleaves: in=%.3f invert=%.3f out=%.3f (us_per_point: in=%.2f "
                    "invert=%.2f out=%.2f) gscale=%.3f s\n", BR.t_gin, BR.t_ginv, BR.t_gout,
                    BR.giant_points ? 1e6 * BR.t_gin / (double)BR.giant_points : 0.0,
                    BR.giant_points ? 1e6 * BR.t_ginv / (double)BR.giant_points : 0.0,
                    BR.giant_points ? 1e6 * BR.t_gout / (double)BR.giant_points : 0.0,
                    BR.t_gscale);
        /* section 42: the projective leaves and the segment products that cover them must match
           EXACTLY (asserted in run_batched); the number is reported so a drift is visible */
        std::printf("real_batched_projective: leaves=%llu gamma_points=%llu segments=%llu "
                    "affine_fallback_points=%llu\n", BR.proj_points, BR.proj_gamma_points,
                    BR.proj_segments, BR.proj_fallbacks);
        std::printf("real_giant_chain: chunks=%llu seed_points=%llu chunks_per_ladder=%llu\n",
                    BR.giant_chain_chunks, BR.giant_seed_points,
                    BR.giant_chain_chunks ? 0ull : 1ull);
        /* objective 3: giant points that are the IDENTITY modulo a factor of N -- the hit made
           explicit.  Their gcd(Z, N) was recorded as a factor (credited to no hit). */
        std::printf("real_giant_degenerate: points=%llu\n", BR.giant_degenerate);
        std::printf("real_batched_breakdown: wall=%.2f ntt_calls=%llu ntt_launches=%llu "
                    "ntt_seconds=%.3f (%.1f%%) arena_mb=%.1f arena_overflow=%llu\n", el,
                    L.ntt_calls - nb, L.ntt_launches - nl, L.ntt_seconds - ns,
                    el > 0 ? 100.0 * (L.ntt_seconds - ns) / el : 0.0, arena.mb(),
                    BR.arena_overflow);
        /* WHERE the NTT-attributed time goes, PER CALL (objective 4).  The phase columns are only
           filled when NTT_HOST_BREAK=1 (two clock reads per phase); without it they print 0 and
           only per_call is meaningful.  The point of the line: `ntt_seconds/ntt_calls` at the real
           shape is ~165 us per call, while the GPU phases measured on the frozen shape are ~50 us,
           so the rest is wrapper/host work -- and these columns say which part. */
        if (L.ntt_calls > nb) {
            const double inv_c = 1.0 / (double)(L.ntt_calls - nb);
            std::printf("real_batched_ntt_usecall: per_call=%.1f setup=%.1f pack=%.1f maxcoeff=%.1f "
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
            std::printf("real_batched_carrydefer: chunks_deferred=%llu finishes=%llu "
                        "deferred_slices=%llu\n",
                        g_defer_chunks, g_defer_finishes, g_defer_slices);
            /* SECTION 30: the pinned/async path, counted for the same reason as above -- and the
               fallback count, so a run that silently lost pinned memory cannot look normal */
            std::printf("real_batched_asyncxfer: raw_async=%llu out_async=%llu fallbacks=%llu "
                        "(async_enabled=%d)\n",
                        g_pin_raw_used, g_pin_out_used, g_pin_fallbacks, g_s4_async ? 1 : 0);
            /* OBJECTIVE 4: the one phase the probe cannot see -- the reduced product coming back
               to the host once per chunk in the batched path (section 32). */
            const double gb = (double)(L.d2h_coeff_words - dw0) * 8.0 / 1073741824.0;
            const double tdc = L.t_d2h_coeff - dc0;
            std::printf("real_batched_coeffback: us_per_call=%.1f total=%.3f s volume=%.2f GB "
                        "effective_GBps=%.2f share_of_ntt=%.1f%%\n",
                        inv_c * tdc * 1e6, tdc, gb, tdc > 0 ? gb / tdc : 0.0,
                        (L.ntt_seconds - ns) > 0 ? 100.0 * tdc / (L.ntt_seconds - ns) : 0.0);
            /* and the three phases the batched entry point never had timers for */
            const double thp = L.t_hpack - hp0, tsc = L.t_scan - sc0, th2 = L.t_h2d_batch - hb0;
            const double tck = L.t_check - ck0;
            const double acc = inv_c * (thp + tsc + th2) * 1e6;
            std::printf("real_batched_hostbatch: us_per_call=%.1f (pack=%.1f scan=%.1f h2d=%.1f) "
                        "totals pack=%.3f scan=%.3f h2d=%.3f s share_of_ntt=%.1f%%\n", acc,
                        inv_c * thp * 1e6, inv_c * tsc * 1e6, inv_c * th2 * 1e6, thp, tsc, th2,
                        (L.ntt_seconds - ns) > 0 ? 100.0 * (thp + tsc + th2) / (L.ntt_seconds - ns)
                                                 : 0.0);
            /* THE CARRY-CONVERGENCE ASSERT (section 34): a whole extra pass over the digit array
               plus a D2H, run on EVERY call and never timed before now. */
            std::printf("real_batched_carrycheck: us_per_call=%.1f total=%.3f s "
                        "share_of_ntt=%.1f%%\n", inv_c * tck * 1e6, tck,
                        (L.ntt_seconds - ns) > 0 ? 100.0 * tck / (L.ntt_seconds - ns) : 0.0);
            /* WHICH PART of that is the pass and which is the drain (section 41): the kernel's
               own GPU time comes from in-stream events, the readback is the blocking D2H of the
               two 8-byte counters.  A readback far larger than the kernel is pure latency. */
            {
                const double trs = L.t_check_reset - cr0, tke = L.t_check_kernel - ck0b,
                             td2 = L.t_check_d2h - cd0;
                std::printf("real_batched_carrysplit: reset_us_per_call=%.1f kernel_us_per_call="
                            "%.1f d2h_us_per_call=%.1f | totals reset=%.3f kernel=%.3f "
                            "d2h=%.3f s\n", inv_c * trs * 1e6, inv_c * tke * 1e6, inv_c * td2 * 1e6,
                            trs, tke, td2);
            }
            /* ... and the two remaining pieces of the device entry point: the shape plan
               (choose_cfg re-proving the exactness bound per call) and the device-to-device copy
               of the packed operands into the arena's scratch. */
            const double tpl = L.t_plan - pl0b, toc = L.t_opcopy - oc0;
            const double tho = L.t_hout - ho0;
            std::printf("real_batched_devpath: plan_us_per_call=%.1f (total=%.3f s) "
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
                std::printf("real_batched_rawupload: us_per_call=%.1f total=%.3f s volume=%.2f GB "
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
                std::printf("s4_reduce_stats: P=%llu slot_bits=%llu L=%d nlimb=%d launches=%llu "
                            "coeffs=%llu gmp_checked=%llu gmp_bad=%llu slot_canonical_bad=%llu "
                            "t_reduce=%.3f\n", S->P, S->slot_bits, S->L, S->nlimb, S->calls,
                            S->coeffs, S->checked, S->check_bad, S->canon_bad, S->t_reduce);
                /* section 41: how much of the hook is the KERNEL and how much is the host */
                std::printf("s4_reduce_split: P=%llu kernel_us_per_call=%.2f "
                            "host_us_per_call=%.2f ring_waits=%llu | t_reduce_host=%.3f s\n",
                            S->P, S->calls ? 1e6 * S->t_reduce / (double)S->calls : 0.0,
                            S->calls ? 1e6 * S->t_reduce_host / (double)S->calls : 0.0,
                            S->dt_blocks, S->t_reduce_host);
                /* the hook's out-of-timer work on the REAL shape too (section 36): the canonical
                   counter's readback and the in-run GMP oracle */
                std::printf("s4_reduce_hook_tail: P=%llu d2h_bad_us_per_call=%.2f "
                            "sample_us_per_call=%.2f | t_hookd2h=%.3f s t_hooksample=%.3f s\n",
                            S->P, S->calls ? 1e6 * S->t_hookd2h / (double)S->calls : 0.0,
                            S->calls ? 1e6 * S->t_hooksample / (double)S->calls : 0.0,
                            S->t_hookd2h, S->t_hooksample);
            }
            std::printf("s4_multiply_stats: enabled=1 launches=%llu poly_muls=%llu "
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

static int run_check_F(const char *path, const char *gpu_dump_path, bool evaluate,
                       bool evaluate_batched)
{
    FDump d = read_f_dump(path);
    if (!d.ok) {
        std::fprintf(stderr, "%s: cannot read --check-F dump '%s': %s\n", NTT_PROBE_NAME, path,
                     d.err.c_str());
        return 2;
    }
    PolyLayer L;
    L.device = g_device;
    if (mpz_set_str(L.N, d.n_dec.c_str(), 10) != 0) {
        std::fprintf(stderr, "%s: bad decimal N in the dump\n", NTT_PROBE_NAME);
        return 2;
    }
    if (mpz_odd_p(L.N) == 0) {
        std::fprintf(stderr, "%s: N must be odd for the Montgomery ladder\n", NTT_PROBE_NAME);
        return 2;
    }
    L.S = (size_t)mpz_sizeinbase(L.N, 2);
    L.W = words_for_bits(L.S);
    L.cost.S = (long)L.S;          /* the batched cost convention needs S from the start */
    const size_t nw = L.W;

    mpz_t a24, Q, R, tmp;
    mpz_inits(a24, Q, R, tmp, nullptr);
    if (!read_hex_mpz(a24, d.a24_hex) || !read_hex_mpz(Q, d.q_hex)) {
        std::fprintf(stderr, "%s: bad a24_hex/Q_hex in the dump\n", NTT_PROBE_NAME);
        return 2;
    }

    CK(cudaSetDevice(g_device));
    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, g_device));
    CK(cudaFree(0));
    std::printf("stage2_tree_gpu: dump=%s N_bits=%ld D=%llu B1=%llu B2=%llu sigma=%llu "
                "device=%d (%s)\n", path, (long)L.S, (unsigned long long)d.D,
                (unsigned long long)d.B1, (unsigned long long)d.B2,
                (unsigned long long)d.sigma, g_device, prop.name);
    std::printf("  repr: coefficient-major, m*W words, W=%llu x 64 bits, S=bits(N)=%llu ; the "
                "NTT multiply is handed exactly this layout with S=N_bits\n",
                (unsigned long long)L.W, (unsigned long long)L.S);

    /* Montgomery parameters: R = 2^(64*W) mod N, mone = R (the image of 1), ninv = -N^-1 mod 2^64 */
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
            std::fprintf(stderr, "%s: N is not invertible mod 2^64 (N must be odd)\n",
                         NTT_PROBE_NAME);
            return 2;
        }
        mpz_sub(inv, pow2, inv);                       /* -N^-1 mod 2^64 */
        unsigned long long buf[2] = {0, 0};
        size_t cnt = 0;
        mpz_export(buf, &cnt, -1, 8, 0, 0, inv);
        ninv = buf[0];
        mpz_clears(nmod, inv, pow2, nullptr);
    }
    std::vector<unsigned long long> hn(nw), hqx(nw), hqz(nw), ha24(nw), hmone(nw);
    if (!mpz_to_words(hn, nw, L.N)) {
        std::fprintf(stderr, "%s: N does not fit W words\n", NTT_PROBE_NAME);
        return 2;
    }
    mpz_mul(tmp, Q, R);
    mpz_mod(tmp, tmp, L.N);
    mpz_to_words(hqx, nw, tmp);                        /* Q.X * R */
    mpz_to_words(hqz, nw, R);                          /* Z = 1  ->  1*R */
    mpz_mul(tmp, a24, R);
    mpz_mod(tmp, tmp, L.N);
    mpz_to_words(ha24, nw, tmp);
    mpz_to_words(hmone, nw, R);

    if (mont_selftest(hn, nw, ninv, L.N, R) != 0) return 1;

    /* ---- slice S4: ONE arena + ONE device reduction for the WHOLE run -------------------
       The arena is created here (not only for the batched engine) so the F tree, the simple
       S2 tail and the batched engine all reuse the same buffers and twiddle tables, and the
       device mod-N reduction replaces the host GMP reduction for every multiply.
       NTT_S4_OFF=1 restores the pre-S4 per-call + host-GMP path, which is what makes the
       "before/after" numbers of this slice reproducible in one binary. */
    NttArena arena_all;
    arena_all.device = g_device;
    S4Ctx s4;
    S4Reduce red;
    {
        size_t freeb = 0, totalb = 0;
        CK(cudaMemGetInfo(&freeb, &totalb));
        size_t reserve = (size_t)1024 * 1024 * 1024;            /* for the driver + the rest */
        const char *envr = std::getenv("NTT_ARENA_RESERVE_MB");
        if (envr && *envr) reserve = (size_t)std::strtoull(envr, nullptr, 10) << 20;
        size_t cap = (freeb > reserve) ? (freeb - reserve) : (freeb / 2);
        const size_t hard = (size_t)2048 * 1024 * 1024;
        const char *envh = std::getenv("NTT_ARENA_CAP_KB");
        if (envh && *envh) cap = (size_t)std::strtoull(envh, nullptr, 10) << 10;
        const char *envf = std::getenv("NTT_ARENA_FRACTION");
        if (envf && *envf) {
            const double f = std::atof(envf);
            cap = (size_t)((double)freeb * f);
        }
        (void)hard;
        arena_all.cap_bytes = cap;
        std::printf("mem_budget: device_free=%.0f MB total=%.0f MB reserve=%.0f MB "
                    "arena_cap=%.0f MB (free - reserve; NTT_ARENA_CAP_KB / "
                    "NTT_ARENA_RESERVE_MB / NTT_ARENA_FRACTION override) -- a shape that does "
                    "not fit gets per-call allocations instead (same result, slower)\n",
                    freeb / 1048576.0, totalb / 1048576.0, reserve / 1048576.0,
                    arena_all.cap_bytes / 1048576.0);
    }
    const char *env4 = std::getenv("NTT_S4_OFF");
    const bool s4_on = !(env4 && *env4 && std::atoi(env4) != 0);
    /* the stage-2 parameters, from the dump (a real shape builds them from the curve instead) */
    Stage2Params SP;
    SP.D = d.D; SP.B1 = d.B1; SP.B2 = d.B2; SP.baby_j = d.baby_j;
    {
        const char *es = std::getenv("NTT_S4_SAMPLE");
        if (es && *es) g_s4_sample_limit = std::atoll(es);
        const char *ee = std::getenv("NTT_S4_CHECK_EVERY");
        if (ee && *ee) g_s4_check_every = (unsigned long long)std::strtoull(ee, nullptr, 10);
        const char *et = std::getenv("NTT_S4_DESCENT_TRACE");
        if (et && *et && std::atoi(et) != 0) g_s4_descent_trace = true;
    }
    L.arena = &arena_all;            /* one arena for the whole run (S4) */
    if (s4_on) {
        s4_reduce_init(red, L.N, nw, hn, ninv);
        s4.red = &red;
        L.s4 = &s4;
        std::printf("s4_multiply: ON -- every tree multiply is batched per shape and its exact "
                    "coefficients are reduced mod N ON THE DEVICE (NTT_S4_OFF=1 disables)\n");
    } else {
        std::printf("s4_multiply: OFF -- pre-S4 path (one call per multiply, host GMP "
                    "reduction)\n");
    }

    /* every later ladder (baby points, giant points, culprit confirmation) goes through this */
    LadderCtx C;
    C.hn = hn; C.hqx = hqx; C.hqz = hqz; C.ha24 = ha24; C.hmone = hmone;
    C.ninv = ninv;
    C.nw = nw;

    /* ---- the baby points, on the device ------------------------------------------------ */
    const size_t npts = d.baby_j.size();
    std::vector<unsigned long long> outx, outz;
    ladder_points(C, d.baby_j, outx, outz);
    /* affine_x on the host: the reference's function, to the letter */
    std::vector<std::string> my_x(npts);
    unsigned long long baby_bad = 0, baby_first = 0;
    {
        mpz_t X, Z, ax;
        mpz_inits(X, Z, ax, nullptr);
        for (size_t i = 0; i < npts; ++i) {
            words_to_mpz(X, &outx[i * nw], nw);
            words_to_mpz(Z, &outz[i * nw], nw);
            affine_x_gmp(ax, X, Z, L.N);
            char *s = mpz_get_str(nullptr, 16, ax);
            my_x[i] = s;
            void (*ff)(void *, size_t) = nullptr;
            mp_get_memory_functions(nullptr, nullptr, &ff);
            ff(s, std::strlen(s) + 1);
            if (my_x[i] != d.baby_x[i]) {
                if (!baby_bad) {
                    baby_first = (unsigned long long)i;
                    std::printf("baby_mismatch: j=%llu gpu_x=%s cpu_x=%s\n",
                                (unsigned long long)d.baby_j[i], my_x[i].c_str(),
                                d.baby_x[i].c_str());
                }
                ++baby_bad;
            }
        }
        mpz_clears(X, Z, ax, nullptr);
    }
    std::printf("ladder: baby_points=%llu mismatches=%llu first_bad=%llu "
                "(device x_j = affine_x([j]Q) vs the CPU reference's x_j)\n",
                (unsigned long long)npts, baby_bad, baby_first);

    /* ---- the F tree, from the DEVICE's own baby points ---------------------------------- */
    std::vector<std::vector<unsigned long long>> leaf(npts);
    {
        mpz_t xj, neg;
        mpz_inits(xj, neg, nullptr);
        std::vector<unsigned long long> tmpw(nw, 0ull);
        for (size_t i = 0; i < npts; ++i) {
            read_hex_mpz(xj, my_x[i]);
            mpz_neg(neg, xj);
            mpz_mod(neg, neg, L.N);                    /* (X - x_j) = [ -x_j, 1 ] */
            mpz_to_words(tmpw, nw, neg);
            leaf[i].assign(2 * nw, 0ull);
            std::copy(tmpw.begin(), tmpw.end(), leaf[i].begin());
            leaf[i][nw] = 1;                           /* the monic leading coefficient */
        }
        mpz_clears(xj, neg, nullptr);
    }
    FTreeStats fs;
    std::vector<size_t> Fdeg;
    size_t Fpad = 0;
    std::vector<std::vector<unsigned long long>> Ft =
        build_tree_flat(L, leaf, Fdeg, Fpad, fs, BC_FTREE);
    std::vector<unsigned long long> F = Ft[1];      /* the same root build_F_tree returned */
    /* F is MONIC of degree = the number of baby points (24 leaves of degree 1 => degree 24,
       i.e. 25 coefficients); the CPU reference's F_degree field is exactly this number. */
    size_t fdeg = npts;
    while (fdeg > 0) {                                 /* the reference's poly_trim */
        bool zero = true;
        for (size_t j = 0; j < nw; ++j)
            if (F[fdeg * nw + j] != 0) { zero = false; break; }
        if (!zero) break;
        --fdeg;
    }
    std::vector<std::string> myF(fdeg + 1);
    for (size_t k = 0; k <= fdeg; ++k) hex_of_words(myF[k], &F[k * nw], nw, L.N);
    unsigned long long cf_bad = 0, cf_first = 0;
    const size_t ncmp = std::min(myF.size(), d.F.size());
    for (size_t k = 0; k < ncmp; ++k) {
        if (myF[k] != d.F[k]) {
            if (!cf_bad) {
                cf_first = (unsigned long long)k;
                std::printf("F_mismatch: k=%llu gpu=%s cpu=%s (both hex, mod N)\n",
                            (unsigned long long)k, myF[k].c_str(), d.F[k].c_str());
            }
            ++cf_bad;
        }
    }
    if (myF.size() != d.F.size()) {
        std::printf("F_length_mismatch: gpu=%llu cpu=%llu\n", (unsigned long long)myF.size(),
                    (unsigned long long)d.F.size());
        cf_bad += (myF.size() > d.F.size()) ? (myF.size() - d.F.size())
                                            : (d.F.size() - myF.size());
    }
    std::printf("ftree: leaves=%llu padded=%llu muls=%llu ntt_calls=%llu max_ntt_coeffs=%llu "
                "max_ntt_words=%llu max_slot_bits=%llu slot_checks=%llu ntt_seconds=%.3f\n",
                (unsigned long long)fs.leaves, (unsigned long long)fs.padded,
                (unsigned long long)fs.muls, L.ntt_calls, L.max_ntt_coeffs, L.max_ntt_words,
                L.max_slot_bits, L.slot_checks, L.ntt_seconds);
    /* the exactness bound, for the shape of THIS tree that is closest to p -- the bound is
       asserted (with exact integer arithmetic) inside ntt_poly_mul_host for every single call,
       this line just makes the number auditable.  L = P*slot_words is the count of NONZERO
       digits per operand, which is what bounds a convolution coefficient (section 14.15). */
    std::printf("exactness: binding_shape P=%llu S=%llu slot_bits=%llu slot_words=%llu "
                "L=P*slot_words=%llu bpw=%d ; L*(2^bpw-1)^2 = 2^%.3f < p = 2^%.3f -> %s "
                "(asserted per call, not inherited from the probe's shapes)\n",
                (unsigned long long)L.bind_P, (unsigned long long)L.S,
                (unsigned long long)L.bind_slot_bits, (unsigned long long)L.bind_slot_words,
                (unsigned long long)L.bind_L, L.bind_bpw, L.bind_bound_bits, mpz_log2d_p(),
                exact_ok_terms(L.bind_L, L.bind_bpw) ? "OK" : "VIOLATED");
    std::printf("check_F: coeffs_gpu=%llu coeffs_cpu=%llu mismatches=%llu first_bad=%llu "
                "baby_mismatches=%llu\n", (unsigned long long)myF.size(),
                (unsigned long long)d.F.size(), cf_bad, cf_first, baby_bad);

    if (gpu_dump_path) {
        if (write_f_dump(gpu_dump_path, L.N, a24, Q, d.D, d.B1, d.B2, d.sigma, d.baby_j, my_x,
                         myF, (unsigned long long)fdeg))
            std::printf("check_F: gpu_dump=%s written\n", gpu_dump_path);
        else
            std::printf("check_F: gpu_dump=%s FAILED to write\n", gpu_dump_path);
    }
    const bool ok = (baby_bad == 0 && cf_bad == 0);
    std::printf("check_F: F_degree_gpu=%llu F_degree_cpu=%llu ok=%d\n",
                (unsigned long long)fdeg, (unsigned long long)d.degree, ok ? 1 : 0);

    /* ---- the optional tails: slice S2 (simple) and slice S3 (batched) ------------------- */
    if (evaluate) {
        const unsigned long long ntt_before2 = L.ntt_calls;
        const double ntt_s_before2 = L.ntt_seconds;
        const double f0 = L.t_fwd, i0 = L.t_inv, s0 = L.t_slot;
        /* NTT_ARENA_S2=1 runs the SAME simple structure with the persistent multiply arena:
           it separates "the batched structure does 3.9x less work" from "the per-call
           cudaMalloc/fuse_init/table rebuild is pure overhead".  Off by default, so the S2
           numbers above stay the ones slice S2 reported. */
        NttArena arena2;
        arena2.device = g_device;
        const char *env2 = std::getenv("NTT_ARENA_S2");
        const bool s2_arena = (env2 && *env2 && std::atoi(env2) != 0);
        NttArena *save_arena = L.arena;
        L.arena = s2_arena ? &arena2 : nullptr;
        const double t0 = now_s();
        const Stage2Tail tail = run_stage2_tail(L, C, SP, F, fdeg);
        s4_oracle_drain(red);
        const double elapsed = now_s() - t0;
        L.arena = save_arena;
        std::string factors;
        for (size_t i = 0; i < tail.factors.size(); ++i) {
            if (i) factors += ",";
            factors += tail.factors[i];
        }
        std::string primes;
        for (size_t i = 0; i < tail.hit_primes.size(); ++i) {
            if (i) primes += ",";
            primes += std::to_string(tail.hit_primes[i]);
        }
        /* the same machine-readable summary the two CPU references print, so one regex reads
           all three ("algorithm=tree_gpu" keeps it distinguishable) */
        std::printf("stage2: algorithm=tree_gpu curves=1 hits=%llu bad_factors=%llu factors=%s "
                    "hit_primes=%s elapsed=%.2f\n", tail.hits, tail.bad_factors, factors.c_str(),
                    primes.c_str(), elapsed);
        std::printf("stage2_gpu_tail: ntt_calls_total=%llu ntt_seconds=%.3f\n", L.ntt_calls,
                    L.ntt_seconds);
        std::printf("stage2_gpu_tail_split: s2_arena=%d calls=%llu ntt=%.3f t_fwd=%.3f "
                    "t_inv=%.3f t_slot=%.3f host_side=%.3f arena_mb=%.1f\n",
                    s2_arena ? 1 : 0, L.ntt_calls - ntt_before2,
                    L.ntt_seconds - ntt_s_before2, L.t_fwd - f0, L.t_inv - i0, L.t_slot - s0,
                    (L.ntt_seconds - ntt_s_before2) - ((L.t_fwd - f0) + (L.t_inv - i0) +
                                                       (L.t_slot - s0)),
                    s2_arena ? arena2.mb() : 0.0);
    }
    if (evaluate || evaluate_batched) {
        const unsigned long long ntt_before = L.ntt_calls;
        const unsigned long long launches_before = L.ntt_launches;
        const double ntt_s_before = L.ntt_seconds;
        const double fwd0 = L.t_fwd, inv0 = L.t_inv, slot0 = L.t_slot;
        /* THE DEVICE ORCHESTRATION: the SAME arena the F tree already used (slice S4 keeps one
           arena for the whole run, so no shape is ever evicted and reallocated between
           phases).  Its cap was decided and printed by mem_budget above. */
        NttArena &arena = arena_all;
        L.arena = &arena;
        const double t0 = now_s();
        BatchedRun BR = run_batched(L, C, SP, Ft, Fdeg, Fpad);
        s4_oracle_drain(red);
        const double elapsed = now_s() - t0;
        L.arena = nullptr;
        const unsigned long long ntt_calls_b = L.ntt_calls - ntt_before;
        const double ntt_s_b = L.ntt_seconds - ntt_s_before;
        std::string factors, primes;
        for (size_t i = 0; i < BR.tail.factors.size(); ++i) {
            if (i) factors += ",";
            factors += BR.tail.factors[i];
        }
        for (size_t i = 0; i < BR.tail.hit_primes.size(); ++i) {
            if (i) primes += ",";
            primes += std::to_string(BR.tail.hit_primes[i]);
        }
        /* THE SHAPE (gate 3): the batched structure's multiplication count and its size
           distribution, in the CPU reference's own convention, next to the counts that were
           really executed (NTT calls) -- the two differ because one poly multiply is one NTT
           call and the fold is 3 of them, which is exactly what makes them comparable. */
        std::printf("batched_shape: P=phi(D)/2=%llu giant_points=%llu num_poly_g=%llu loops=%llu "
                    "f_tree_leaf_pad=%llu descent_divmods=%llu leaf_values=%llu blocks=%llu "
                    "block_per=%llu\n",
                    BR.P, BR.giant_points, BR.num_poly_g, BR.loops,
                    (unsigned long long)Fpad, BR.descent_divmods, BR.leaf_values,
                    BR.apply_blocks, BR.block_per);
        std::printf("batched_cost: tree_convention=ours-padded S=%llu poly_muls=%llu "
                    "operand_bits=%llu f_tree=%llu g_tree=%llu fold=%llu descent=%llu inv=%llu "
                    "total=%llu coeff_muls=%llu max_mul=%llux%llu max_mul_bits=%llu\n",
                    (unsigned long long)L.S, L.cost.tot_muls(), L.cost.tot_bits(),
                    L.cost.bits[BC_FTREE], L.cost.bits[BC_GTREE], L.cost.bits[BC_FOLD],
                    L.cost.bits[BC_DESCENT], L.cost.bits[BC_FINV], L.cost.tot_bits(),
                    [&] { unsigned long long t = 0; for (int i = 0; i < BC_NCAT; ++i)
                              t += L.cost.coeff[i]; return t; }(),
                    L.cost.max_m1, L.cost.max_m2, L.cost.max_bits);
        for (int c = 0; c < BC_NCAT; ++c)
            std::printf("batched_cost_cat: %-8s muls=%llu operand_bits=%llu coeff_muls=%llu "
                        "max_operand_coeffs=%llu\n", kBatchedCat[c], L.cost.muls[c],
                        L.cost.bits[c], L.cost.coeff[c], L.cost.cat_mul_max[c]);
        for (int l = 0; l < 64; ++l)
            if (L.cost.hcnt[l])
                std::printf("batched_cost_level: log2m=%d muls=%llu operand_bits=%llu\n", l,
                            L.cost.hcnt[l], L.cost.hbits[l]);
        std::printf("batched_ntt: ntt_calls_total=%llu ntt_seconds_total=%.3f  [batched engine "
                    "alone: ntt_calls=%llu ntt_seconds=%.3f] small_primes=%llu hit_blocks=%llu "
                    "named_searches=%llu candidates_tested=%llu\n",
                    L.ntt_calls, L.ntt_seconds, ntt_calls_b, ntt_s_b, BR.small_primes,
                    BR.hit_blocks, BR.named_searches, BR.candidates_tested);
        if (s4_on) {
            unsigned long long sel_cases = 0, sel_bad = 0, checked = 0, check_bad = 0,
                               full = 0, canon = 0, coeffs = 0;
            double t_red = 0.0;
            s4_dbad_resolve(red, /*wait=*/true);   /* section 43: the last readback must be checked */
            s4_oracle_drain(red);
            s4_oracle_report();
            for (S4Reduce::Shape *S : red.shapes) {
                S->dt_flush();     /* resolve the deferred marks (section 41/42) */
                sel_cases += S->selftest_cases; sel_bad += S->selftest_bad;
                checked += S->checked; check_bad += S->check_bad;
                full += S->full_checks; canon += S->canon_bad; coeffs += S->coeffs;
                t_red += S->t_reduce;
                std::printf("s4_reduce_stats: P=%llu slot_bits=%llu L=%d nlimb=%d bound_margin="
                            "%.1f launches=%llu coeffs=%llu gmp_checked=%llu gmp_bad=%llu "
                            "full_checks=%llu slot_canonical_bad=%llu t_reduce=%.3f\n",
                            S->P, S->slot_bits, S->L, S->nlimb, S->bound_bits, S->calls,
                            S->coeffs, S->checked, S->check_bad, S->full_checks, S->canon_bad,
                            S->t_reduce);
                std::printf("s4_reduce_hook_tail: P=%llu d2h_bad_us_per_call=%.1f sample_checks="
                            "%llu sample_us_per_call=%.1f | t_hookd2h=%.3f s t_hooksample=%.3f s\n",
                            S->P, S->calls ? 1e6 * S->t_hookd2h / (double)S->calls : 0.0,
                            S->calls / (g_s4_check_every ? g_s4_check_every : 1),
                            S->calls ? 1e6 * S->t_hooksample / (double)S->calls : 0.0,
                            S->t_hookd2h, S->t_hooksample);
            }
            std::printf("s4_multiply_stats: enabled=1 launches=%llu poly_muls=%llu "
                        "tree_level_calls=%llu shape_groups=%llu reduce_launches=%llu "
                        "coeffs_reduced=%llu t_reduce=%.3f gmp_selftest_cases=%llu "
                        "gmp_selftest_bad=%llu gmp_checked=%llu gmp_check_bad=%llu "
                        "full_checks=%llu ; ntt_launches=%llu ntt_poly_muls=%llu\n",
                        s4.launches, s4.muls, s4.level_calls, s4.groups, red.reduce_calls,
                        coeffs, t_red, sel_cases, sel_bad, checked, check_bad, full,
                        L.ntt_launches, L.ntt_calls);
        }
        std::printf("batched_split: giant=%.3f gtrees=%.3f fold=%.3f descent=%.3f inv=%.3f "
                    "accum=%.3f name=%.3f\n", BR.t_giant, BR.t_gtrees, BR.t_fold,
                    BR.t_descent, BR.t_inv, BR.t_accum, BR.t_name);
        std::printf("batched_giant_chain: chunks=%llu seed_points=%llu\n",
                    BR.giant_chain_chunks, BR.giant_seed_points);
        std::printf("batched_breakdown: wall=%.2f ntt_seconds=%.3f (%.1f%%) ntt_calls=%llu "
                    "ntt_launches=%llu non_ntt_seconds=%.3f ladder_launches=%llu ladder_points=%llu "
                    "prod_launches=%llu fuse_builds=%llu fuse_reuse=%llu buf_builds=%llu "
                    "buf_reuse=%llu arena_mb=%.1f arena_cap_mb=%.0f arena_overflow=%llu\n",
                    elapsed, ntt_s_b, elapsed > 0 ? 100.0 * ntt_s_b / elapsed : 0.0, ntt_calls_b,
                    L.ntt_launches - launches_before, elapsed - ntt_s_b, BR.ladder_calls,
                    BR.ladder_points, BR.prod_launches,
                    BR.arena_fuse_builds, BR.arena_fuse_reuse, BR.arena_buf_builds,
                    BR.arena_buf_reuse, BR.arena_mb, (double)arena.cap_bytes / 1048576.0,
                    BR.arena_overflow);
        std::printf("batched_ntt_split: t_fwd=%.3f t_inv=%.3f t_slot=%.3f "
                    "(transforms+sync+carry, i.e. launch overhead + real work) "
                    "host_side=%.3f (packing, memcpy, exact extraction, cross-check) "
                    "per_call_us=%.1f\n",
                    L.t_fwd - fwd0, L.t_inv - inv0, L.t_slot - slot0,
                    ntt_s_b - ((L.t_fwd - fwd0) + (L.t_inv - inv0) + (L.t_slot - slot0)),
                    ntt_calls_b ? 1e6 * ntt_s_b / (double)ntt_calls_b : 0.0);
        /* slice S4: the SAME quantities, per call, split further (NTT_HOST_BREAK=1 gates the
           sub-timers; without it these are all zero and the line still prints). */
        if (ntt_calls_b) {
            const double inv_c = 1.0 / (double)ntt_calls_b;
            std::printf("batched_ntt_usecall: per_call=%.1f setup=%.1f pack=%.1f maxcoeff=%.1f "
                        "h2d=%.1f fwd=%.1f inv=%.1f carry_slot_out=%.1f d2h=%.1f extract=%.1f "
                        "xcheck=%.1f | calls=%llu\n",
                        inv_c * ntt_s_b * 1e6, inv_c * L.t_setup * 1e6, inv_c * L.t_pack * 1e6,
                        inv_c * L.t_maxc * 1e6, inv_c * L.t_h2d * 1e6, inv_c * (L.t_fwd - fwd0) * 1e6,
                        inv_c * (L.t_inv - inv0) * 1e6, inv_c * (L.t_slot - slot0) * 1e6,
                        inv_c * L.t_d2h * 1e6, inv_c * L.t_ext * 1e6, inv_c * L.t_xchk * 1e6,
                        ntt_calls_b);
        }
        /* the same summary line shape as S2 and the two CPU references */
        std::printf("stage2: algorithm=tree_gpu_batched curves=1 hits=%llu bad_factors=%llu "
                    "factors=%s hit_primes=%s elapsed=%.2f\n", BR.tail.hits, BR.tail.bad_factors,
                    factors.c_str(), primes.c_str(), elapsed);
    }
    mpz_clears(a24, Q, R, tmp, nullptr);
    return ok ? 0 : 1;
}

/* ===================================================================================== *
 *  --selftest: the polynomial layer against GMP, over a COMPOSITE and a PRIME modulus
 *  (the stage-2 algorithm has to work without knowing the factors of N, so the composite
 *  case is the one that matters -- same reasoning as the CPU reference's selftest)
 * ===================================================================================== */
static int run_selftest(void)
{
    const char *mods[] = { "340282366920938463463374607431768211457",    /* 2^128+1 = p1*p2 */
                           "340282366920938463463374607431768211507" };  /* a nearby prime */
    int fails = 0;
    for (int mi = 0; mi < 2; ++mi) {
        PolyLayer L;
        L.device = g_device;
        mpz_set_str(L.N, mods[mi], 10);
        L.S = (size_t)mpz_sizeinbase(L.N, 2);
        L.W = words_for_bits(L.S);
        const size_t W = L.W;
        for (int trial = 0; trial < 3; ++trial) {
            const size_t da = (size_t)(2 + trial * 7), db = (size_t)(3 + trial * 5);
            std::vector<unsigned long long> A((da + 1) * W, 0ull), B((db + 1) * W, 0ull);
            mpz_t x, acc, t;
            mpz_inits(x, acc, t, nullptr);
            uint64_t s = 0x9e3779b97f4a7c15ull + (uint64_t)mi * 7 + (uint64_t)trial;
            std::vector<unsigned long long> w(W, 0ull);
            for (size_t i = 0; i <= da + db; ++i) {
                for (size_t j = 0; j < W; ++j) {
                    s = s * 6364136223846793005ull + 1442695040888963407ull;
                    w[j] = s;
                }
                words_to_mpz(x, w.data(), W);
                mpz_mod(x, x, L.N);
                mpz_to_words(w, W, x);
                if (i <= da) std::copy(w.begin(), w.end(), A.begin() + (long)(i * W));
                if (i <= db) std::copy(w.begin(), w.end(), B.begin() + (long)(i * W));
            }
            const std::vector<unsigned long long> C = poly_mul_modN(L, A, da, B, db);
            unsigned long long bad = 0, first = 0;
            for (size_t k = 0; k <= da + db; ++k) {
                mpz_set_ui(acc, 0);
                for (size_t i = 0; i <= da; ++i) {
                    if (k < i || k - i > db) continue;
                    words_to_mpz(x, &A[i * W], W);
                    words_to_mpz(t, &B[(k - i) * W], W);
                    mpz_mul(t, t, x);
                    mpz_add(acc, acc, t);
                }
                mpz_mod(acc, acc, L.N);
                std::vector<unsigned long long> want(W, 0ull);
                mpz_to_words(want, W, acc);
                for (size_t j = 0; j < W; ++j)
                    if (C[k * W + j] != want[j]) { if (!bad) first = (unsigned long long)k; ++bad; }
            }
            std::printf("selftest: mod=%d bits=%llu da=%llu db=%llu coeffs_checked=%llu bad=%llu "
                        "first_bad=%llu\n", mi, (unsigned long long)L.S,
                        (unsigned long long)da, (unsigned long long)db,
                        (unsigned long long)(da + db + 1), bad, first);
            if (bad) ++fails;
            /* the ARENA path (slice S3) must return the identical coefficients: same kernels,
               same packing, same assertions -- only the allocations and the twiddle tables are
               hoisted.  Checked TWICE: the first call builds them, the second reuses. */
            if (trial == 0) {
                NttArena ar;
                ar.device = g_device;
                L.arena = &ar;
                for (int rep = 0; rep < 2; ++rep) {
                    const std::vector<unsigned long long> C2 = poly_mul_modN(L, A, da, B, db);
                    unsigned long long abad = 0, afirst = 0;
                    const size_t nc = std::min(C2.size(), C.size());
                    for (size_t q = 0; q < nc; ++q)
                        if (C2[q] != C[q]) { if (!abad) afirst = (unsigned long long)(q / W); ++abad; }
                    if (C2.size() != C.size()) ++abad;
                    std::printf("selftest: arena mod=%d rep=%d coeffs=%llu mismatches=%llu "
                                "first_bad=%llu fuse_builds=%llu fuse_reuse=%llu bytes=%.0f\n",
                                mi, rep, (unsigned long long)(C2.size() / W), abad, afirst,
                                ar.fuse_builds, ar.fuse_hits, (double)ar.bytes);
                    if (abad) ++fails;
                }
                L.arena = nullptr;
            }
            mpz_clears(x, acc, t, nullptr);
        }
    }
    std::printf("selftest: %s\n",
                fails ? "FAILED" : "OK (GPU poly_mul_modN == GMP schoolbook mod N, composite "
                                   "and prime modulus)");
    return fails ? 1 : 0;
}

int main(int argc, char **argv)
{
    s2g_install_crash_handler();          /* first: a crash must never print nothing */
    s2g_install_terminate();              /* and neither must an escaping exception */
    ladder_cap_init();                    /* the ladder launch length (NTT_LADDER_CAP) */
    {
        const char *enp = std::getenv("NTT_NO_PROGRESS");   /* progress lines on by default */
        if (enp && *enp && std::atoi(enp) != 0) g_s4_batched_progress = false;
    }
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    const char *check_F = nullptr, *gpu_dump = nullptr;
    const char *real_n = nullptr;
    bool selftest = false, evaluate = false, evaluate_batched = false;
    bool real = false, real_hex = false, choose_d = false, real_s2 = false;
    unsigned long long r_sigma = 26, r_b1 = 1000, r_b2 = 1000000, r_d = 0;
    int r_curves = 1;
    for (int i = 1; i < argc; ++i) {
        const char *a = argv[i];
        auto next = [&]() -> const char * { return (i + 1 < argc) ? argv[++i] : ""; };
        if (!std::strcmp(a, "--check-F")) check_F = next();
        else if (!std::strcmp(a, "--dump-F-gpu")) gpu_dump = next();
        else if (!std::strcmp(a, "--device")) g_device = std::atoi(next());
        else if (!std::strcmp(a, "--evaluate")) evaluate = true;
        else if (!std::strcmp(a, "--evaluate-batched")) evaluate_batched = true;
        else if (!std::strcmp(a, "--selftest")) selftest = true;
        /* slice S4: the real-shape path (no CPU reference can exist at S = 5261) */
        else if (!std::strcmp(a, "--real")) real = true;
        else if (!std::strcmp(a, "--n")) real_n = next();
        else if (!std::strcmp(a, "--n-hex")) { real_n = next(); real_hex = true; }
        else if (!std::strcmp(a, "--sigma")) r_sigma = (unsigned long long)std::strtod(next(), nullptr);
        /* strtod, NOT strtoull(base 10): "--b2 1e11" is how the plan and Prime95's own
           parameters write B2, and strtoull stops at the 'e' and returns 1 (measured:
           a whole run computed a B2=1 shape and printed giant_points=2, section 27) */
        else if (!std::strcmp(a, "--b1")) r_b1 = (unsigned long long)std::strtod(next(), nullptr);
        else if (!std::strcmp(a, "--b2")) r_b2 = (unsigned long long)std::strtod(next(), nullptr);
        else if (!std::strcmp(a, "--d")) r_d = (unsigned long long)std::strtod(next(), nullptr);
        else if (!std::strcmp(a, "--choose-d")) choose_d = true;
        else if (!std::strcmp(a, "--s2")) real_s2 = true;
        else if (!std::strcmp(a, "--curves")) r_curves = std::atoi(next());
        else {
            std::fprintf(stderr,
                         "usage: stage2_tree_gpu --check-F <cpu dump> [--dump-F-gpu <file>] "
                         "[--evaluate] [--evaluate-batched] [--device N]\n"
                         "       stage2_tree_gpu --selftest [--device N]\n"
                         "       stage2_tree_gpu --real --n <decimal>|--n-hex <hex> "
                         "[--sigma S] [--b1 B1] [--b2 B2] [--d D | --choose-d] [--s2] "
                         "[--curves K] [--device N]\n");
            return 2;
        }
    }
    if (selftest) return run_selftest();
    if (real) {
        if (!real_n) {
            std::fprintf(stderr, "%s: --real needs --n <decimal> or --n-hex <hex>\n",
                         NTT_PROBE_NAME);
            return 2;
        }
        if (!choose_d && r_d == 0) {
            std::fprintf(stderr, "%s: --real needs --d <D> or --choose-d\n", NTT_PROBE_NAME);
            return 2;
        }
        return run_real(real_n, real_hex, r_sigma, r_b1, r_b2, r_d, choose_d, real_s2, r_curves);
    }
    if (!check_F) {
        std::fprintf(stderr, "%s: --check-F <file> is required (write one with "
                             "stage2_tree_ref --dump-F)\n", NTT_PROBE_NAME);
        return 2;
    }
    return run_check_F(check_F, gpu_dump, evaluate, evaluate_batched);
}