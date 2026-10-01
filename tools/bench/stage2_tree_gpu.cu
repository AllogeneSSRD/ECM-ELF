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
   "Z shares a factor with N" case, which the reference passes through unchanged. */
static void affine_x_gmp(mpz_t out, const mpz_t X, const mpz_t Z, const mpz_t N)
{
    if (mpz_cmp_ui(Z, 0) == 0) { mpz_set_ui(out, 0); return; }
    mpz_t inv;
    mpz_init(inv);
    if (mpz_invert(inv, Z, N) == 0) {
        mpz_set(out, X);
    } else {
        mpz_mul(out, X, inv);
        mpz_mod(out, out, N);
    }
    mpz_clear(inv);
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
    /* the same kernel, launched once per <= g_ladder_cap points (grid-stride inside) */
    for (int p0 = 0; p0 < npts; p0 += g_ladder_cap) {
        const int m = ((npts - p0) < g_ladder_cap) ? (npts - p0) : g_ladder_cap;
        const unsigned int bl = (unsigned int)((g_ladder_cap + th - 1) / th);
        s2g_ladder_kernel<NW><<<bl, th>>>(dn, ninv, nw, dqx, dqz, da24, dmone, djs + p0, m,
                                          dx + (size_t)p0 * nw, dz + (size_t)p0 * nw);
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
    ~S4Ctx() { if (d_out) cudaFree(d_out); }
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

/* per-chunk device-buffer budget for a batched multiply (MB): see poly_mul_batch_modN */
static unsigned long long g_s4_batch_budget_mb = 32;

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
        unsigned long long checked = 0, check_bad = 0, check_first = 0, full_checks = 0;
        unsigned long long selftest_cases = 0, selftest_bad = 0;
        long long selftest_first = -1;
        double t_reduce = 0.0;
        ~Shape() { if (dy) cudaFree(dy); }
        Shape() = default;
        Shape(const Shape &) = delete;
        Shape &operator=(const Shape &) = delete;
    };
    std::vector<Shape *> shapes;
    unsigned long long reduce_calls = 0, coeffs_total = 0;

    S4Reduce() { mpz_init(N); }
    ~S4Reduce()
    {
        for (Shape *s : shapes) delete s;
        if (dn) cudaFree(dn);
        if (dbad) cudaFree(dbad);
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

/* base-2^64 limbs of the slot window -> mod N, on the device.  One thread per coefficient. */
template <int NW>
__global__ void s4_reduce_kernel(const unsigned long long *digits, unsigned long long n,
                                 int bpw, unsigned long long slot_words,
                                 unsigned long long out_slots, unsigned long long nbatch,
                                 const unsigned long long *dn, unsigned long long ninv,
                                 int nw, int L, const unsigned long long *dy,
                                 unsigned long long w, unsigned long long *out,
                                 unsigned long long slot_bits, unsigned long long *bad)
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
    /* ---- (2) L Montgomery elimination steps: each makes limb i zero and divides by 2^64 - */
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
    /* ---- (3) back to the plain domain: c = r * 2^(64L) = Mont(r, Y) ---------------- */
    unsigned long long u[NW];
    s2g_mont_mul<NW>(u, r, dy, dn, ninv, nw);
    for (unsigned long long i = 0; i < w; ++i)
        out[gid * w + i] = (i < (unsigned long long)nw) ? u[i] : 0ull;
}

/* (the slot-canonical check is folded into s4_reduce_kernel above: the digits are in registers
   there, and a separate kernel would add one launch and one full read per batched multiply) */

template <int NW>
static void s4_launch_reduce(int nw, int L, unsigned long long nbatch,
                             unsigned long long out_slots, unsigned long long total,
                             const unsigned long long *ddig, unsigned long long n, int bpw,
                             unsigned long long slot_words, const unsigned long long *dn,
                             unsigned long long ninv, const unsigned long long *dy,
                             unsigned long long w, unsigned long long *dout,
                             unsigned long long slot_bits, unsigned long long *dbad)
{
    const unsigned int th = 128;
    const unsigned int bl = (unsigned int)((total + th - 1) / th);
    s4_reduce_kernel<NW><<<bl, th>>>(ddig, n, bpw, slot_words, out_slots, nbatch, dn, ninv, nw,
                                     L, dy, w, dout, slot_bits, dbad);
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
    CK(cudaMalloc(&R.dn, (size_t)R.nw * sizeof(unsigned long long)));
    CK(cudaMemcpy(R.dn, R.hn.data(), (size_t)R.nw * sizeof(unsigned long long),
                  cudaMemcpyHostToDevice));
    return 0;
}

/* L = the smallest number of elimination steps with 2^slot_bits <= N * 2^(64 L), proved with
   mpz against the ACTUAL N (never against bits(N) as a proxy).  Called ONCE per shape. */
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
    const double t0 = now_s();
    if (!R.dbad) {
        CK(cudaMalloc(&R.dbad, sizeof(unsigned long long)));
        R.dbad_host = 0;
    }
    CK(cudaMemset(R.dbad, 0, sizeof(unsigned long long)));
    S2G_DISPATCH(R.nw, s4_launch_reduce, (int)R.nw, S->L, nbatch, out_slots, total, digits, n,
                 bpw, slot_words, R.dn, R.ninv, S->dy, w, out, slot_bits, R.dbad);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    S->t_reduce += now_s() - t0;
    unsigned long long hbad = 0;
    CK(cudaMemcpy(&hbad, R.dbad, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    ++R.reduce_calls;
    S->canon_bad += hbad;
    if (hbad) {
        std::fprintf(stderr, "%s: FATAL: %llu of %llu slot windows have nonzero digits above "
                             "slot_bits=%llu -- the reduction bound C < 2^slot_bits does NOT "
                             "apply to these values\n", NTT_PROBE_NAME, hbad, total, slot_bits);
        std::exit(3);
    }
    ++S->calls;
    S->coeffs += total;
    R.coeffs_total += total;
    /* THE IN-RUN ORACLE, here rather than at the call site: the digit buffer belongs to the
       multiply and is released when it returns, so a check done afterwards would read freed
       device memory (that was a real crash, "CUDA error invalid argument", the first time the
       S2 tail ran without the arena).  Reading it here is also the stronger check: the raw
       digits GMP sees are the ones the transform and the carry just produced. */
    if (g_s4_sample_limit > 0 && (S->calls <= 1 || (S->calls % g_s4_check_every) == 0))
        s4_check_reduced(R, S, digits, n, out_slots, nbatch, out, g_s4_sample_limit);
}

/* one coefficient, reduced on the host with GMP the way the pre-S4 code did it: the digits
   are the exact integer C = sum_j d[j] 2^(bpw j), so C mod N is the oracle. */
static void s4_gmp_reduce(mpz_t out, const unsigned long long *digits, unsigned long long nslots,
                          int bpw, const mpz_t N)
{
    mpz_t v;
    mpz_init(v);
    mpz_set_ui(v, 0);
    for (unsigned long long j = nslots; j-- > 0;) {
        mpz_mul_2exp(v, v, (unsigned)bpw);
        mpz_add_u64(v, digits[j]);       /* NOT mpz_add_ui: on Windows that truncates to 32 bits */
    }
    mpz_mod(out, v, N);
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
    for (unsigned long long c = 0; c < cases; ++c) {
        for (unsigned long long j = 0; j < slot_words; ++j) {
            s = s * 6364136223846793005ull + 1442695040888963407ull;
            unsigned long long v;
            switch (c % 6) {
                case 0: v = maxd; break;                                  /* all ones */
                case 1: v = 0; break;                                     /* zero */
                case 2: v = (j + 1 == slot_words) ? 1ull : 0ull; break;   /* single top bit */
                case 3: v = (j == 0) ? maxd : 0ull; break;                /* low digit full */
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
    unsigned long long *dd = nullptr, *dout = nullptr;
    CK(cudaMalloc(&dd, dig.size() * sizeof(unsigned long long)));
    CK(cudaMalloc(&dout, (size_t)(cases * R.w) * sizeof(unsigned long long)));
    CK(cudaMemcpy(dd, dig.data(), dig.size() * sizeof(unsigned long long),
                  cudaMemcpyHostToDevice));
    S2G_DISPATCH(R.nw, s4_launch_reduce, (int)R.nw, S->L, 1ull, cases, cases, dd,
                 cases * slot_words, bpw, slot_words, R.dn, R.ninv, S->dy,
                 (unsigned long long)R.w, dout, S->slot_bits, (unsigned long long *)nullptr);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(got.data(), dout, got.size() * sizeof(unsigned long long),
                  cudaMemcpyDeviceToHost));
    cudaFree(dd);
    cudaFree(dout);
    unsigned long long bad = 0;
    long long first = -1;
    mpz_t want, mine;
    mpz_inits(want, mine, nullptr);
    for (unsigned long long c = 0; c < cases; ++c) {
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
static void s4_check_reduced(S4Reduce &R, S4Reduce::Shape *S, const unsigned long long *ddig,
                             unsigned long long n, unsigned long long out_slots,
                             unsigned long long nbatch, const unsigned long long *d_out,
                             long long sample_limit)
{
    const unsigned long long total = out_slots * nbatch;
    if (!total || sample_limit == 0) return;
    const unsigned long long lim = (unsigned long long)sample_limit;
    const bool small_batch = (nbatch <= 4 && total <= lim);
    std::vector<std::array<unsigned long long, 3>> runs;   /* (slice, k0, count) */
    if (small_batch) {
        for (unsigned long long s = 0; s < nbatch; ++s) runs.push_back({s, 0ull, out_slots});
        ++S->full_checks;
    } else {
        unsigned long long cnt = lim;
        if (cnt > out_slots) cnt = out_slots;
        uint64_t seed = 0x9e3779b97f4a7c15ull * (S->calls + 1) + total;
        seed = seed * 6364136223846793005ull + 1442695040888963407ull;
        const unsigned long long s = (seed >> 11) % nbatch;
        seed = seed * 6364136223846793005ull + 1442695040888963407ull;
        const unsigned long long room = (out_slots > cnt) ? (out_slots - cnt + 1) : 1;
        const unsigned long long k0 = (seed >> 11) % room;
        runs.push_back({s, k0, cnt});
    }
    mpz_t want, mine;
    mpz_inits(want, mine, nullptr);
    std::vector<unsigned long long> digbuf, redbuf;
    for (const std::array<unsigned long long, 3> &rr : runs) {
        const unsigned long long s = rr[0], k0 = rr[1], cnt = rr[2];
        digbuf.assign((size_t)(cnt * S->slot_words), 0ull);
        redbuf.assign((size_t)(cnt * R.w), 0ull);
        CK(cudaMemcpy(digbuf.data(), ddig + s * n + k0 * S->slot_words,
                      digbuf.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(redbuf.data(), d_out + (s * out_slots + k0) * R.w,
                      redbuf.size() * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        for (unsigned long long q = 0; q < cnt; ++q) {
            s4_gmp_reduce(want, &digbuf[(size_t)(q * S->slot_words)], S->slot_words, S->bpw,
                          R.N);
            mpz_import(mine, (size_t)R.w, -1, 8, 0, 0, &redbuf[(size_t)(q * R.w)]);
            ++S->checked;
            if (mpz_cmp(want, mine) != 0) {
                if (!S->check_bad) {
                    S->check_first = s * out_slots + k0 + q;
                    char *ws = mpz_get_str(nullptr, 16, want);
                    char *ms = mpz_get_str(nullptr, 16, mine);
                    std::printf("s4_reduce_CHECK_bad: slice=%llu k=%llu gmp=%s gpu=%s\n",
                                s, k0 + q, ws, ms);
                    void (*ff)(void *, size_t) = nullptr;
                    mp_get_memory_functions(nullptr, nullptr, &ff);
                    ff(ws, std::strlen(ws) + 1);
                    ff(ms, std::strlen(ms) + 1);
                }
                ++S->check_bad;
                std::fprintf(stderr, "%s: FATAL: the device reduction disagrees with GMP at "
                                     "slice %llu coefficient %llu\n", NTT_PROBE_NAME, s, k0 + q);
                std::exit(3);
            }
        }
    }
    mpz_clears(want, mine, nullptr);
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
                                std::vector<unsigned long long> &out, int cat = -1)
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
    {
        unsigned long long qN = 0, qsb = 0, qsw = 0, qss = 0, qos = 0;
        int qbpw = 0;
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
    const unsigned long long budget_bytes = g_s4_batch_budget_mb << 20;
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
    for (unsigned long long s0 = 0; s0 < nbatch; s0 += chunk) {
        const unsigned long long m = ((nbatch - s0) < chunk) ? (nbatch - s0) : chunk;
        NttReduceHook h2 = hook;
        if (hook.out) h2.out = hook.out + (size_t)(s0 * out_slots) * W;
        const int r1 = ntt_poly_mul_batch_host(P, (int)L.S, L.device, m,
                                               wa + s0 * P * W, wb + s0 * P * W,
                                               (s0 == 0) ? &slots : nullptr, &st, L.arena, &h2,
                                               nullptr);
        if (r1 != 0) { rc = r1; break; }
        /* the reduced coefficients of this chunk, back to the host */
        std::vector<unsigned long long> all((size_t)(m * out_slots * W), 0ull);
        CK(cudaMemcpy(all.data(), h2.out, all.size() * sizeof(unsigned long long),
                      cudaMemcpyDeviceToHost));
        for (unsigned long long s = 0; s < m; ++s)
            std::copy(all.begin() + (long)(s * out_slots * W),
                      all.begin() + (long)(s * out_slots * W + nc * W),
                      out.begin() + (long)((s0 + s) * nc * W));
    }
    L.ntt_seconds += now_s() - t0;
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
    for (size_t base = pad / 2; ; base /= 2) {
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
            poly_mul_batch_modN(L, wa.data(), wb.data(), ma, mb, nbatch, res, cat);
            ++L.s4->groups;
            for (size_t s = 0; s < nbatch; ++s) {
                const size_t i = g.second[s];
                t[i].assign(res.begin() + (long)(s * nc * W), res.begin() + (long)((s + 1) * nc * W));
                deg[i] = ma + mb - 2;
                ++fs.muls;
            }
        }
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
    unsigned long long *dn = nullptr, *dqx = nullptr, *dqz = nullptr, *da24 = nullptr,
                       *dmone = nullptr, *djs = nullptr, *dx = nullptr, *dz = nullptr;
    CK(cudaMalloc(&dn, nw * 8));
    CK(cudaMalloc(&dqx, nw * 8));
    CK(cudaMalloc(&dqz, nw * 8));
    CK(cudaMalloc(&da24, nw * 8));
    CK(cudaMalloc(&dmone, nw * 8));
    CK(cudaMalloc(&djs, n * 8));
    CK(cudaMalloc(&dx, outx.size() * 8));
    CK(cudaMalloc(&dz, outz.size() * 8));
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
    cudaFree(dn); cudaFree(dqx); cudaFree(dqz); cudaFree(da24); cudaFree(dmone);
    cudaFree(djs); cudaFree(dx); cudaFree(dz);
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
    const size_t P = H.size();
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
    const size_t P = H.size();                        /* deg H < deg F <= P */
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

static bool is_prime_u64(unsigned long long p)
{
    if (p < 2) return false;
    mpz_t z;
    mpz_init(z);
    mpz_import(z, 1, -1, 8, 0, 0, &p);            /* NOT mpz_set_ui/mpz_add_ui: on Windows
                                                     unsigned long is 32 bits and those
                                                     truncate silently (section 14.10) */
    const int r = mpz_probab_prime_p(z, 25);
    mpz_clear(z);
    return r != 0;
}

struct Stage2Tail {
    unsigned long long hits = 0, bad_factors = 0;
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

/* the reference's record(): keep a factor that really divides N, and its hit prime */
static void s3_record(Stage2Tail &out, const mpz_t f, unsigned long long prime, const mpz_t N)
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
    ++out.hits;
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
    mpz_t t, rmul;
    mpz_inits(t, rmul, nullptr);
    std::vector<unsigned long long> hr(nw, 0ull);
    mpz_to_words(hr, nw, R);
    unsigned long long calls = 0;
    for (unsigned long long pk : pps) {
        LadderCtx &C = ctx;
        C.hn = hn;
        C.nw = nw;
        C.ninv = ninv;
        C.ha24 = ha24;
        C.hmone = hmone;
        C.hqx.assign(nw, 0ull);
        C.hqz.assign(nw, 0ull);
        {   /* Q * R mod N */
            mpz_t x;
            mpz_init(x);
            words_to_mpz(x, qx.data(), nw);
            mpz_mul(x, x, R);
            mpz_mod(x, x, N);
            mpz_to_words(C.hqx, nw, x);
            words_to_mpz(x, qz.data(), nw);
            mpz_mul(x, x, R);
            mpz_mod(x, x, N);
            mpz_to_words(C.hqz, nw, x);
            mpz_clear(x);
        }
        std::vector<unsigned long long> ox, oz;
        ladder_points(C, std::vector<unsigned long long>(1, pk), ox, oz);
        qx.assign(ox.begin(), ox.begin() + (long)nw);
        qz.assign(oz.begin(), oz.begin() + (long)nw);
        ++calls;
    }
    mpz_clears(t, rmul, nullptr);
    return calls;
}
struct BatchedRun {
    Stage2Tail tail;
    unsigned long long giant_points = 0, num_poly_g = 0, loops = 0, P = 0;
    unsigned long long descent_divmods = 0, leaf_values = 0, apply_blocks = 0, block_per = 0;
    unsigned long long small_primes = 0;
    unsigned long long hit_blocks = 0, named_searches = 0, candidates_tested = 0;
    /* device-side bookkeeping, so the breakdown line reports measured work, not a guess */
    unsigned long long ladder_calls = 0, ladder_points = 0, prod_launches = 0;
    unsigned long long arena_fuse_builds = 0, arena_fuse_reuse = 0, arena_buf_builds = 0,
                       arena_buf_reuse = 0, arena_overflow = 0;
    double arena_mb = 0.0;
    unsigned long long ntt_calls = 0;
    double t_giant = 0.0, t_gtrees = 0.0, t_fold = 0.0, t_descent = 0.0, t_inv = 0.0,
           t_accum = 0.0, t_name = 0.0;
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
        const unsigned long long c1 = ((imax - c0) < pts_per_chunk) ? imax : (c0 + pts_per_chunk);
        const size_t clo = (size_t)(c0 + 1), chi = (size_t)c1;
        const double tgp = now_s();
        gjs.resize(chi - clo + 1);
        for (size_t i = clo; i <= chi; ++i) gjs[i - clo] = (unsigned long long)i * D;
        s2g_state("giant ladder chunk: about to launch");   /* the last state on a driver kill */
        ladder_points_ws(ws, gjs, gx, gz);
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
            mpz_t X, Z, ax, neg;
            mpz_inits(X, Z, ax, neg, nullptr);
            std::vector<unsigned long long> w(W, 0ull);
            for (size_t i = lo; i < hi; ++i) {
                const size_t q = i - (clo - 1);        /* index inside this chunk */
                words_to_mpz(X, &gx[q * W], W);
                words_to_mpz(Z, &gz[q * W], W);
                affine_x_gmp(ax, X, Z, L.N);
                mpz_neg(neg, ax);
                mpz_mod(neg, neg, L.N);               /* the leaf (X - x_i) = [ -x_i, 1 ] */
                mpz_to_words(w, W, neg);
                std::copy(w.begin(), w.end(), bleaf[i - lo].begin());
                bleaf[i - lo][W] = 1;
            }
            mpz_clears(X, Z, ax, neg, nullptr);
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
    }

    /* ---- 3. ONE descent of H against the F tree: H(x_j) at every baby point ------------ */
    const double td0 = now_s();
    std::printf("descent_begin: P=%llu levels=%d last_state=(%s)\n", (unsigned long long)P,
                (int)ceil_log2_u64((unsigned long long)Fpad), g_last_state);
    s2g_state("the batched descent");            /* a driver kill here leaves this behind */
    std::vector<std::vector<unsigned long long>> values((size_t)P,
                                                       std::vector<unsigned long long>(W, 0ull));
    if (L.s4) {
        /* slice S4: the whole descent, level by level and batched (at P = 92160 this is
           ~1.8e5 divmods, 1.4e5 of them with k <= 2) */
        L.cat = BC_DESCENT;
        descent_batched(L, Ft, Fdeg, Fpad, H, values, R.descent_divmods, BC_DESCENT);
        L.cat = -1;
    } else {
        L.cat = BC_DESCENT;
        descent_slow(L, Ft, Fdeg, Fpad, H, values, R.descent_divmods);
        L.cat = -1;
    }
    /* NTT_S4_DESCENT_CHECK=1 runs BOTH descents on the same H and compares every leaf value:
       the batched descent is a rearrangement of the same divisions, so any difference is a bug
       in the rearrangement. */
    {
        const char *envc = std::getenv("NTT_S4_DESCENT_CHECK");
        if (L.s4 && envc && *envc && std::atoi(envc) != 0) {
            std::vector<std::vector<unsigned long long>> ref((size_t)P,
                std::vector<unsigned long long>(W, 0ull));
            unsigned long long dm = 0;
            L.cat = BC_DESCENT;
            descent_slow(L, Ft, Fdeg, Fpad, H, ref, dm);
            L.cat = -1;
            unsigned long long bad = 0, first = 0;
            for (size_t i = 0; i < (size_t)P; ++i)
                for (size_t q = 0; q < W; ++q)
                    if (values[i][q] != ref[i][q]) {
                        if (!bad) {
                            first = (unsigned long long)i;
                            std::printf("descent_check_bad: i=%llu word=%llu batched=%llu "
                                        "slow=%llu\n", (unsigned long long)i,
                                        (unsigned long long)q, values[i][q], ref[i][q]);
                        }
                        ++bad;
                    }
            std::printf("descent_check: P=%llu divmods_batched=%llu divmods_slow=%llu "
                        "mismatching_coefficients=%llu first=%llu\n", (unsigned long long)P,
                        R.descent_divmods, dm, bad, first);
        }
    }
    R.leaf_values = P;
    R.t_descent = now_s() - td0;

    /* ---- 4. accumulate prod_j H(x_j) mod N ON THE DEVICE, one gcd per block ------------ */
    const double ta0 = now_s();
    const size_t BLOCK = 64;                            /* small, so a hit localises to <=64 */
    std::vector<std::vector<unsigned long long>> bprod;
    ws.need_vals((size_t)P);
    {
        std::vector<unsigned long long> flat((size_t)P * W, 0ull);
        for (size_t i = 0; i < P; ++i)
            std::copy(values[i].begin(), values[i].end(), flat.begin() + (long)(i * W));
        CK(cudaMemcpy(ws.dvals, flat.data(), flat.size() * 8, cudaMemcpyHostToDevice));
        dev_block_products(ws, (int)P, (int)BLOCK, bprod);
    }
    R.apply_blocks = bprod.size();
    R.block_per = BLOCK;
    mpz_t bv, bg;
    mpz_inits(bv, bg, nullptr);
    std::vector<std::string> bstr((size_t)bprod.size());
    for (size_t b = 0; b < bprod.size(); ++b) {
        words_to_mpz(bv, bprod[b].data(), W);
        mpz_gcd(bg, bv, L.N);
        if (mpz_cmp_ui(bg, 1) > 0 && mpz_cmp(bg, L.N) < 0) {
            ++R.hit_blocks;                             /* a real hit: only NOW do we look at
                                                           the individual leaf values */
            const size_t lo = b * BLOCK;
            const size_t hi = std::min(lo + BLOCK, (size_t)P);
            for (size_t j = lo; j < hi; ++j) {
                mpz_t v;
                mpz_init(v);
                words_to_mpz(v, values[j].data(), W);
                mpz_gcd(pg, v, L.N);
                if (mpz_cmp_ui(pg, 1) > 0 && mpz_cmp(pg, L.N) < 0) {
                    /* the baby point j = SP.baby_j[j] is the culprit's BABY half:
                       p | H(x_j) => p | (x_j - x_i) for some giant i, so search i */
                    const double tn0 = now_s();
                    const unsigned long long jj = SP.baby_j[j];
                    std::vector<unsigned long long> cand;
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
                    R.candidates_tested += cand.size();
                    if (!cand.empty()) {
                        std::vector<unsigned long long> cx, cz;
                        ladder_points_ws(ws, cand, cx, cz);
                        for (size_t k = 0; k < cand.size(); ++k) {
                            mpz_t z;
                            mpz_init(z);
                            words_to_mpz(z, &cz[k * W], W);
                            mpz_gcd(pg, z, L.N);
                            if (mpz_cmp_ui(pg, 1) > 0 && mpz_cmp(pg, L.N) < 0)
                                s3_record(R.tail, pg, cand[k], L.N);
                            mpz_clear(z);
                        }
                    }
                    R.t_name += now_s() - tn0;
                }
                mpz_clear(v);
            }
        }
    }
    mpz_clears(bv, bg, nullptr);
    R.t_accum = now_s() - ta0;
    mpz_clears(g, pg, nullptr);
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

/* the arena footprint of one transform length: the shared big buffers (3*N words) plus the
   FuseCtx tables of that N (N + N/2 + the per-pass tables).  Shapes that share N share the
   buffers and the tables, so the run's footprint is the sum over DISTINCT N, not over shapes. */
static unsigned long long real_shape_words(unsigned long long P, int S, bool *ok)
{
    unsigned long long N = 0, sw = 0, ss = 0, os = 0;
    int bpw = 0;
    if (!ntt_shape_query(P, S, &N, &bpw, nullptr, &sw, &ss, &os)) { *ok = false; return 0; }
    *ok = true;
    return 3 * N + (N + N / 2 + 4 * 8 * 64) + os;
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

struct DChoice {
    unsigned long long D = 0, P = 0, largest_words = 0, total_words = 0;
    bool fits = false;
};

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

    /* ---- the D search: the largest D whose largest transform still fits the budget ---- */
    unsigned long long D = D_in, P_baby = 0;
    {
        static const unsigned long long prim[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41,
                                                  43, 47};
        std::vector<DChoice> cands;
        if (choose_d) {
            const size_t np = sizeof(prim) / sizeof(prim[0]);
            for (unsigned long long mask = 1; mask < (1ull << np); ++mask) {
                unsigned long long d = 1;
                bool over = false;
                for (size_t i = 0; i < np; ++i)
                    if (mask & (1ull << i)) {
                        if (d > 200000000ull / prim[i]) { over = true; break; }
                        d *= prim[i];
                    }
                if (over || d > 200000000ull) continue;
                DChoice c;
                c.D = d;
                c.P = phi_u64(d) / 2;
                if (c.P == 0) continue;
                unsigned long long ww = 0;
                if (!real_run_words(c.P, (int)L.S, &ww, &c.largest_words, nullptr)) continue;
                c.largest_words = ww;
                c.total_words = ww;
                c.fits = (double)c.total_words * 8.0 <= (double)cap;
                cands.push_back(c);
            }
            std::sort(cands.begin(), cands.end(),
                      [](const DChoice &a, const DChoice &b) { return a.D < b.D; });
            const DChoice *best = nullptr;
            for (const DChoice &c : cands)
                if (c.fits && (!best || c.D > best->D)) best = &c;
            std::printf("d_budget: free=%.0f MB reserve=%.0f MB cap=%.0f MB ; candidates=%llu ; "
                        "the largest transform of a candidate is 3*N+out_slots+N+N/2 words\n",
                        freeb / 1048576.0, reserve / 1048576.0, cap / 1048576.0,
                        (unsigned long long)cands.size());
            {
                unsigned long long shown = 0;
                for (size_t i = cands.size(); i-- > 0 && shown < 12; ++shown) {
                    const DChoice &c = cands[i];
                    std::printf("d_budget_choice: D=%llu P=phi/2=%llu largest_transform=%.0f MB "
                                "total_needed=%.0f MB fits=%d\n", c.D, c.P,
                                (double)c.largest_words * 8.0 / 1048576.0,
                                (double)c.total_words * 8.0 / 1048576.0, c.fits ? 1 : 0);
                }
            }
            if (!best) {
                std::fprintf(stderr, "%s: no candidate D fits the budget (cap=%.0f MB)\n",
                             NTT_PROBE_NAME, cap / 1048576.0);
                return 3;
            }
            D = best->D;
            std::printf("d_budget_decision: D=%llu P=phi(D)/2=%llu (the largest D whose tree + "
                        "fold + descent shapes fit; work is proportional to 1/D, so this is the "
                        "fastest admissible shape)\n", D, best->P);
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
    std::printf("real_shape: D=%llu P=phi(D)/2=%llu baby_points=%llu giant_points=%llu "
                "num_poly_g=%llu loops=%llu S_bits=%ld\n", D, P_baby,
                (unsigned long long)baby_j.size(), imax, (P_baby + imax - 1) / P_baby,
                (imax + P_baby - 1) / P_baby - 1, (long)L.S);

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
    {
        const double t0 = now_s();
        std::vector<unsigned long long> bx, bz;
        ladder_points(C, baby_j, bx, bz);
        std::vector<std::vector<unsigned long long>> leaf(baby_j.size());
        {
            mpz_t X, Z, xj, neg;
            mpz_inits(X, Z, xj, neg, nullptr);
            std::vector<unsigned long long> w(nw, 0ull);
            for (size_t i = 0; i < baby_j.size(); ++i) {
                words_to_mpz(X, &bx[i * nw], nw);
                words_to_mpz(Z, &bz[i * nw], nw);
                affine_x_gmp(xj, X, Z, L.N);
                mpz_neg(neg, xj);
                mpz_mod(neg, neg, L.N);
                mpz_to_words(w, nw, neg);
                leaf[i].assign(2 * nw, 0ull);
                std::copy(w.begin(), w.end(), leaf[i].begin());
                leaf[i][nw] = 1;
            }
            mpz_clears(X, Z, xj, neg, nullptr);
        }
        std::printf("ladder: baby_points=%llu (device x_j; there is no CPU reference at this "
                    "shape, see the report)\n", (unsigned long long)baby_j.size());
        FTreeStats fs;
        Ft = build_tree_flat(L, leaf, Fdeg, Fpad, fs, BC_FTREE);
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
        const double t0 = now_s();
        BatchedRun BR = run_batched(L, C, SP, Ft, Fdeg, Fpad);
        const double el = now_s() - t0;
        std::string fs2, ps2;
        for (size_t i = 0; i < BR.tail.factors.size(); ++i) { if (i) fs2 += ","; fs2 += BR.tail.factors[i]; }
        for (size_t i = 0; i < BR.tail.hit_primes.size(); ++i) { if (i) ps2 += ","; ps2 += std::to_string(BR.tail.hit_primes[i]); }
        std::printf("stage2: algorithm=tree_gpu_batched curves=1 hits=%llu bad_factors=%llu "
                    "factors=%s hit_primes=%s elapsed=%.2f\n", BR.tail.hits, BR.tail.bad_factors,
                    fs2.c_str(), ps2.c_str(), el);
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
        std::printf("real_batched_breakdown: wall=%.2f ntt_calls=%llu ntt_launches=%llu "
                    "ntt_seconds=%.3f (%.1f%%) arena_mb=%.1f arena_overflow=%llu\n", el,
                    L.ntt_calls - nb, L.ntt_launches - nl, L.ntt_seconds - ns,
                    el > 0 ? 100.0 * (L.ntt_seconds - ns) / el : 0.0, arena.mb(),
                    BR.arena_overflow);
        if (s4_on) {
            unsigned long long sel_cases = 0, sel_bad = 0, checked = 0, check_bad = 0,
                               full = 0, canon = 0, coeffs = 0;
            double t_red = 0.0;
            for (S4Reduce::Shape *S : red.shapes) {
                sel_cases += S->selftest_cases; sel_bad += S->selftest_bad;
                checked += S->checked; check_bad += S->check_bad; full += S->full_checks;
                canon += S->canon_bad; coeffs += S->coeffs; t_red += S->t_reduce;
                std::printf("s4_reduce_stats: P=%llu slot_bits=%llu L=%d nlimb=%d launches=%llu "
                            "coeffs=%llu gmp_checked=%llu gmp_bad=%llu slot_canonical_bad=%llu "
                            "t_reduce=%.3f\n", S->P, S->slot_bits, S->L, S->nlimb, S->calls,
                            S->coeffs, S->checked, S->check_bad, S->canon_bad, S->t_reduce);
            }
            std::printf("s4_multiply_stats: enabled=1 launches=%llu poly_muls=%llu "
                        "coeffs_reduced=%llu t_reduce=%.3f gmp_selftest_cases=%llu "
                        "gmp_selftest_bad=%llu gmp_checked=%llu gmp_check_bad=%llu "
                        "full_checks=%llu\n", s4.launches, s4.muls, coeffs, t_red, sel_cases,
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
            for (S4Reduce::Shape *S : red.shapes) {
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
        else if (!std::strcmp(a, "--sigma")) r_sigma = std::strtoull(next(), nullptr, 10);
        else if (!std::strcmp(a, "--b1")) r_b1 = std::strtoull(next(), nullptr, 10);
        else if (!std::strcmp(a, "--b2")) r_b2 = std::strtoull(next(), nullptr, 10);
        else if (!std::strcmp(a, "--d")) r_d = std::strtoull(next(), nullptr, 10);
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
