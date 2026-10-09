/* ---------------------------------------------------------------------------
 * stage2_tree_ref.cpp -- ECM stage 2, TREE (polynomial-BSGS) CORRECTNESS REFERENCE.
 *
 * Sibling of tools/bench/stage2_ref.cpp: that one is the ORACLE for the arithmetic
 * and the save-file conventions, this one is the ORACLE for the *tree* structure
 * that the GPU stage-2 engine (docs/architecture/STAGE2.md, Route B / §2.2) is
 * going to implement with an NTT.  Correctness first, speed last: every polynomial
 * multiplication here is SCHOOLBOOK and every modulus reduction is mpz_mod, so the
 * code is inspectable by hand.  It is NOT fast and is not meant to be.
 *
 * What it computes (standard polynomial / Bernstein BSGS stage 2, the structure
 * Prime95 and GMP-ECM use to get O(P log^2 P) instead of O(P^2)):
 *
 *   1. baby set  R = { j : 1 <= j <= D/2, gcd(j, D) = 1 }     (|R| = phi(D)/2)
 *      baby points x_j = affine_x([j]Q)
 *   2. F(X) = prod_{j in R} (X - x_j) mod N, built with a PRODUCT TREE whose
 *      leaves are the linear polynomials (X - x_j); internal nodes multiply their
 *      two children.  F is monic by construction.
 *   3. giant points x_i = affine_x([i*D]Q), i = 1 .. B2/D + 2, built with the same
 *      differential-addition chain as the pairing path (xadd(x_{i-1}, x_1, x_{i-2})).
 *   4. MULTIPOINT EVALUATION of F at the giant points via a REMAINDER TREE: a
 *      second product tree is built over the giant points, then F mod node is
 *      descended; at the leaves the (degree < 1) remainder is the constant F(x_i).
 *      Polynomial division mod N must work for a COMPOSITE N: it uses the
 *      reversed-polynomial Newton iteration, and because F and every tree node are
 *      MONIC the leading coefficient is 1, whose inverse mod N is 1 -- so no
 *      inversion can ever fail and the algorithm is valid without knowing the
 *      factors of N.  (This is the whole reason the algorithm works in stage 2.)
 *   5. accumulate prod_i F(x_i) mod N, gcd per block of 4096 leaves (to bound the
 *      work of naming a culprit) and once more at the end; a reported factor is
 *      always verified to divide N (bad_factors).
 *   6. --naive-check: evaluate F at every giant point by direct Horner and compare
 *      with the remainder tree.  This is the unit-level oracle for steps 2-4 and it
 *      runs the production code path.
 *   7. cost accounting: every polynomial multiplication is counted with its operand
 *      sizes and turned into "operand bits moved" with exactly the convention of
 *      docs/architecture/STAGE2.md:
 *          one multiplication of two polynomials with m coefficients of S bits
 *              costs   2 * m * (2S + ceil(log2 m))   bits,   S = sizeinbase(N,2)
 *      For an unbalanced product (m1 != m2) the cost is charged with m = max(m1,m2)
 *      (documented deviation; the plan's model only ever multiplies balanced
 *      operands).  The total is broken down into F tree / giant tree / remainder
 *      tree / top-level, where top-level is the first division F mod root(GiantTree)
 *      -- the counterpart of the plan's "full-size top-level P x P" step.  It is 0
 *      whenever deg F < the giant-tree root, which is the case for every unbalanced
 *      (small D, large B2/D) shape run here; that is a property of the shape, not a
 *      missing bucket.
 *
 * Conventions are pinned to stage2_ref.cpp (and through it to our stage 1 and to
 * gmp-ecm -param 0): the shared curve-math block below (Pt / xdbl / xadd / ladder /
 * bits_of_u64 / xmul_u64 / affine_x / set_u64 / suyama_curve / build_s_bits /
 * primes_up_to) is copied from that file with its CODE unchanged -- checked mechanically
 * by comparing every code line of the two files; only two comment lines inside xadd were
 * reworded to say where the lesson came from.  The conventions are:
 *   u = sigma^2-5, v = 4*sigma, A = (v-u)^3(3u+v)/(4u^3v) - 2, a24 = (A+2)/4,
 *   start point (X:Z) = (u^3:v^3), stage 1 = [s]P with s = lcm(1..B1) (torsion 1),
 *   Q = affine_x([s]P) is the stage-2 input, --save reads "X=0x..." (0x skipped).
 *
 * Deliberate deviations from stage2_ref.cpp's pairing path, all documented at the point
 * of use:
 *   (a) the baby index set is { j coprime to D, j <= D/2 } (phi(D)/2 points, the
 *       plan's |Rs| ~ 0.2 D), while the pairing path keeps a table for EVERY
 *       j <= D/2.  For a prime p > D/2 with r = p mod D, j = min(r, D-r) is
 *       automatically coprime to D because gcd(r,D) = gcd(p,D) = 1, so both paths
 *       use exactly the same (i, j) for every prime the polynomial path can reach.
 *   (b) primes p <= D/2 (which have no i >= 1) are therefore NOT covered by the
 *       polynomial product; they are handled by the same direct ladder + gcd the
 *       pairing path uses, so the two factor sets stay comparable.  With B1 >= D/2
 *       (the frozen configuration) this loop is empty.
 *   (c) there is no G/H Newton machinery and no scaled remainder descent.  Prime95
 *       rebuilds G(X) per outer loop over poly_size giant points and folds it in with
 *       H = G*H mod F (3 full-size multiplications, ecm.cpp:9334-9461), then runs ONE
 *       scaled remainder descent per curve (ecm.cpp:9518+).  That batched structure is
 *       what the GPU engine will implement, but it is NOT what this reference executes
 *       -- here it is only ACCOUNTED (see the cost_model lines and the model section),
 *       because the point of this file is to pin the multipoint-evaluation core against
 *       direct Horner evaluation, which the simple structure does leaf by leaf.
 *
 * Usage:
 *   stage2_tree_ref.exe --selftest
 *   stage2_tree_ref.exe --n <decimal> --sigma <u64> [--curves <k>] --b1 <B1>
 *                       --b2 <B2> --d <D> [--naive-check] [--cost] [--verbose-hits]
 *                       [--dump-F <file>]
 *   stage2_tree_ref.exe --n <decimal> --save <file> [--save-first <i>] [--curves <k>]
 *                       --b2 <B2> --d <D> [--naive-check] [--cost]
 *   stage2_tree_ref.exe --n <decimal> --b2 <B2> --d <D> --model-only [--num-poly-g <k>]
 *
 * --dump-F <file> writes the modulus, a24, the stage-2 input point Q, the baby points and
 * the coefficients of F in the format the GPU tree engine reads (tools/bench/stage2_tree_gpu.cu,
 * docs section 18 slice S1); it changes NOTHING on stdout.  With --curves > 1 the last curve
 * processed is the one dumped.
 *
 * Every run that executes stage 2 prints, in this order:
 *   stage2_tree:      the shape (baby_j, giant_i, F_degree)
 *   stage2_naive:     with --naive-check (points, mismatches; a mismatch makes exit 1)
 *   cost:             the MEASURED structure (a): poly_muls, operand_bits + breakdown
 *   cost_detail:/cost_level:  with --cost: per-category and per-level-size detail
 *   cost_model:       the ACCOUNTED batched structure (b), under three tree conventions
 *                     (ours-balanced / ours-padded / plan-script)
 *   cost_model_check: the model's tree recursion vs the measured tree cost (must MATCH)
 *   stage2:           the machine-readable summary line (last, as in stage2_ref.exe)
 *
 * Machine-readable summary line (same field names as stage2_ref.exe, plus the new
 * operand_bits=; parsed by tools/test/test_stage2_tree_ref.ps1):
 *   stage2: algorithm=tree curves=1 hits=1 bad_factors=0 factors=59649589127497217
 *           hit_primes=114713 elapsed=0.12 operand_bits=12345678
 * ------------------------------------------------------------------------- */
#include <gmp.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

using std::string;
using std::vector;

double now_s()
{
    using clock = std::chrono::steady_clock;
    static const clock::time_point t0 = clock::now();
    return std::chrono::duration<double>(clock::now() - t0).count();
}

/* ======================================================================== *
 *  x-only Montgomery arithmetic mod N -- VERBATIM from tools/bench/stage2_ref.cpp
 *  (the conventions are pinned there; do not "improve" them here, the two files
 *  must stay bit-identical in their curve math)
 * ======================================================================== */

struct Pt {
    mpz_t X, Z;
    Pt() { mpz_inits(X, Z, nullptr); }
    ~Pt() { mpz_clears(X, Z, nullptr); }
    Pt(const Pt &) = delete;
    Pt &operator=(const Pt &) = delete;
};

/* r = 2p:  X2 = (X+Z)^2 (X-Z)^2,  Z2 = 4XZ ((X-Z)^2 + a24*4XZ),  a24 = (A+2)/4. */
void xdbl(Pt &r, const Pt &p, const mpz_t a24, const mpz_t N)
{
    mpz_t t1, t2, t3, t4;
    mpz_inits(t1, t2, t3, t4, nullptr);
    mpz_add(t1, p.X, p.Z);
    mpz_sub(t2, p.X, p.Z);
    mpz_mul(t3, t1, t1);
    mpz_mul(t4, t2, t2);
    mpz_mul(r.X, t3, t4);
    mpz_sub(t1, t3, t4);                 /* 4XZ */
    mpz_mul(t2, a24, t1);
    mpz_add(t2, t2, t4);
    mpz_mul(r.Z, t1, t2);
    mpz_mod(r.X, r.X, N);
    mpz_mod(r.Z, r.Z, N);
    mpz_clears(t1, t2, t3, t4, nullptr);
}

/* r = p + q with diff = p - q:  X3 = Z_D (X_P X_Q - Z_P Z_Q)^2, Z3 = X_D (X_P Z_Q - Z_P X_Q)^2.
   Everything is computed into temporaries and written back at the END: the ladder calls
   this as xadd(r0, r0, r1, P), i.e. the output aliases the first input AND the difference,
   so writing r.X early destroyed X_P and silently corrupted Z3 (measured in stage2_ref:
   k >= 3 wrong, k = 2 right -- the affine oracle caught it). */
void xadd(Pt &r, const Pt &p, const Pt &q, const Pt &diff, const mpz_t N)
{
    mpz_t a, b, tX, tZ;
    mpz_inits(a, b, tX, tZ, nullptr);
    mpz_mul(a, p.X, q.X);
    mpz_mul(b, p.Z, q.Z);
    mpz_sub(a, a, b);                 /* X_P X_Q - Z_P Z_Q */
    mpz_mul(a, a, a);
    mpz_mul(tX, diff.Z, a);
    mpz_mul(a, p.X, q.Z);
    mpz_mul(b, p.Z, q.X);
    mpz_sub(a, a, b);                 /* X_P Z_Q - Z_P X_Q */
    mpz_mul(a, a, a);
    mpz_mul(tZ, diff.X, a);
    mpz_mod(tX, tX, N);
    mpz_mod(tZ, tZ, N);
    mpz_set(r.X, tX);
    mpz_set(r.Z, tZ);
    mpz_clears(a, b, tX, tZ, nullptr);
}

/* Montgomery ladder over a bit array (MSB first); p is the fixed difference. */
void ladder(Pt &r, const vector<uint8_t> &bits, const Pt &p, const mpz_t a24, const mpz_t N)
{
    size_t first = 0;
    while (first < bits.size() && !bits[first]) ++first;
    if (first == bits.size()) {                     /* k == 0: identity */
        mpz_set_ui(r.X, 1);
        mpz_set_ui(r.Z, 0);
        return;
    }
    Pt r0, r1;
    mpz_set(r0.X, p.X);
    mpz_set(r0.Z, p.Z);
    xdbl(r1, p, a24, N);
    for (size_t i = first + 1; i < bits.size(); ++i) {
        if (bits[i]) {
            xadd(r0, r0, r1, p, N);
            xdbl(r1, r1, a24, N);
        } else {
            xadd(r1, r0, r1, p, N);
            xdbl(r0, r0, a24, N);
        }
    }
    mpz_set(r.X, r0.X);
    mpz_set(r.Z, r0.Z);
}

vector<uint8_t> bits_of_u64(uint64_t k)
{
    vector<uint8_t> b;
    if (k == 0) return b;
    int top = 63;
    while (((k >> top) & 1u) == 0) --top;
    for (int i = top; i >= 0; --i) b.push_back((uint8_t)((k >> i) & 1u));
    return b;
}

void xmul_u64(Pt &r, uint64_t k, const Pt &p, const mpz_t a24, const mpz_t N)
{
    ladder(r, bits_of_u64(k), p, a24, N);
}

/* Normalised affine x = X/Z (0 when the point is the identity). */
void affine_x(mpz_t out, const Pt &p, const mpz_t N)
{
    if (mpz_cmp_ui(p.Z, 0) == 0) {
        mpz_set_ui(out, 0);
        return;
    }
    mpz_t inv;
    mpz_init(inv);
    if (mpz_invert(inv, p.Z, N) == 0) {
        mpz_set(out, p.X);
    } else {
        mpz_mul(out, p.X, inv);
        mpz_mod(out, out, N);
    }
    mpz_clear(inv);
}

void set_u64(mpz_t r, uint64_t v)
{
    mpz_set_ui(r, (unsigned long)(v >> 32));
    mpz_mul_2exp(r, r, 32);
    mpz_add_ui(r, r, (unsigned long)(v & 0xFFFFFFFFull));
}

/* Returns 1 (and a factor) when the sigma is degenerate mod N, 0 on success. */
int suyama_curve(mpz_t a24, Pt &P, uint64_t sigma, const mpz_t N, mpz_t factor)
{
    mpz_t sig, u, v, num, den, inv, A, t;
    mpz_inits(sig, u, v, num, den, inv, A, t, nullptr);
    set_u64(sig, sigma);
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
    if (mpz_invert(inv, den, N) == 0) {
        mpz_gcd(factor, den, N);
        mpz_clears(sig, u, v, num, den, inv, A, t, nullptr);
        return 1;
    }
    mpz_mul(A, num, inv);
    mpz_sub_ui(A, A, 2);                        /* A = (v-u)^3(3u+v)/(4u^3v) - 2 */
    mpz_add_ui(t, A, 2);
    mpz_set_ui(den, 4);
    if (mpz_invert(inv, den, N) == 0) {
        mpz_gcd(factor, den, N);
        mpz_clears(sig, u, v, num, den, inv, A, t, nullptr);
        return 1;
    }
    mpz_mul(a24, t, inv);
    mpz_mod(a24, a24, N);                       /* a24 = (A+2)/4 */
    mpz_powm_ui(P.X, u, 3, N);                  /* X0 = u^3 */
    mpz_powm_ui(P.Z, v, 3, N);                  /* Z0 = v^3 */
    mpz_clears(sig, u, v, num, den, inv, A, t, nullptr);
    return 0;
}

/* s = torsion * lcm(1..B1), as a bit array (MSB first). */
vector<uint8_t> build_s_bits(uint64_t B1, uint64_t torsion)
{
    mpz_t s;
    mpz_init_set_ui(s, (unsigned long)torsion);
    vector<bool> composite((size_t)B1 + 1, false);
    for (uint64_t i = 2; i <= B1; ++i) {
        if (composite[(size_t)i]) continue;
        for (uint64_t j = i * 2; j <= B1; j += i) composite[(size_t)j] = true;
        uint64_t pk = i;
        while (pk <= B1 / i) pk *= i;
        mpz_mul_ui(s, s, (unsigned long)pk);
    }
    const size_t nbits = mpz_sizeinbase(s, 2);
    vector<uint8_t> bits(nbits);
    for (size_t i = 0; i < nbits; ++i) bits[nbits - 1 - i] = (uint8_t)mpz_tstbit(s, i);
    mpz_clear(s);
    return bits;
}

vector<uint64_t> primes_up_to(uint64_t n)
{
    vector<uint64_t> out;
    if (n < 2) return out;
    vector<bool> composite((size_t)n + 1, false);
    for (uint64_t i = 2; i <= n; ++i) {
        if (composite[(size_t)i]) continue;
        out.push_back(i);
        for (uint64_t j = i * 2; j <= n; j += i) composite[(size_t)j] = true;
    }
    return out;
}

uint64_t gcd_u64(uint64_t a, uint64_t b)
{
    while (b) { const uint64_t t = a % b; a = b; b = t; }
    return a;
}

/* Primality of a 64-bit candidate.  Only used to NAME a hit (the recorded factor is
   always confirmed by an independent ladder + gcd of Z, so a primality mistake could
   at worst change which prime is printed, never the factor).  The tree path never
   needs the full sieve up to B2 -- that is exactly why it can be run with B2 = 1e8
   without a 100 MB table. */
bool is_prime_u64(uint64_t p)
{
    if (p < 2) return false;
    mpz_t z;
    mpz_init_set_ui(z, 0);
    set_u64(z, p);
    const int r = mpz_probab_prime_p(z, 25);
    mpz_clear(z);
    return r != 0;
}

/* ======================================================================== *
 *  results (same shape as stage2_ref.cpp so both summary lines are parsed
 *  by one regex)
 * ======================================================================== */

struct Result {
    uint64_t hits = 0;
    uint64_t bad_factors = 0;                   /* a "factor" that does not divide N */
    vector<string> factors;
    vector<uint64_t> hit_primes;
};

uint64_t g_cur_sigma = 0;      /* sigma of the curve being processed (for hit lines) */
bool g_verbose_hits = false;
const char *g_dump_F = nullptr; /* --dump-F <file>: see dump_F_file below (no stdout change) */

void record(Result &res, const mpz_t f, uint64_t prime, const mpz_t N)
{
    if (mpz_cmp_ui(f, 1) <= 0 || mpz_cmp(f, N) == 0) return;
    mpz_t r;
    mpz_init(r);
    mpz_mod(r, N, f);
    if (mpz_cmp_ui(r, 0) != 0) ++res.bad_factors;
    mpz_clear(r);
    char *s = mpz_get_str(nullptr, 10, f);
    bool seen = false;
    for (const string &v : res.factors)
        if (v == s) seen = true;
    if (!seen) res.factors.push_back(s);
    void (*freefunc)(void *, size_t) = nullptr;
    mp_get_memory_functions(nullptr, nullptr, &freefunc);
    freefunc(s, std::strlen(s) + 1);
    if (prime) res.hit_primes.push_back(prime);
    ++res.hits;
    if (g_verbose_hits && prime) {
        std::printf("stage2_hit: sigma=%llu factor=%s prime=%llu\n",
                    (unsigned long long)g_cur_sigma, s, (unsigned long long)prime);
    }
}

/* ======================================================================== *
 *  cost accounting -- "operand bits moved", the convention of
 *  docs/architecture/STAGE2.md
 *
 *  The tool's whole reason for existing is that the plan's per-curve figure
 *  (1.9e11 bits for M5261) is currently derived from Prime95's wall clock and
 *  needs an independent number computed from a real tree.  So EVERY polynomial
 *  multiplication performed anywhere in this file goes through cost_add().
 * ======================================================================== */

enum { CAT_FTREE = 0, CAT_GTREE = 1, CAT_REM = 2, CAT_TOP = 3, CAT_N = 4 };
const char *const kCatName[CAT_N] = { "f_tree", "giant_tree", "remainder", "top_level" };
const int kHistLevels = 24;                  /* histograms keyed by ceil(log2 m) */

int ceil_log2_bits(unsigned long long m)
{
    int l = 0;
    while (l < 63 && (1ull << l) < m) ++l;
    return l;
}

struct Cost {
    long S = 0;                                            /* sizeinbase(N, 2) */
    unsigned long long muls[CAT_N] = { 0, 0, 0, 0 };
    unsigned long long bits[CAT_N] = { 0, 0, 0, 0 };
    unsigned long long coeff_muls[CAT_N] = { 0, 0, 0, 0 };
    unsigned long long hist_cnt[CAT_N][kHistLevels] = {};
    unsigned long long hist_bits[CAT_N][kHistLevels] = {};
    unsigned long long max_m1 = 0, max_m2 = 0, max_bits = 0;
    int max_cat = 0;

    void add(int cat, size_t m1, size_t m2)
    {
        const unsigned long long m = (unsigned long long)std::max(m1, m2);
        const int l = ceil_log2_bits(m);
        const unsigned long long b = 2ull * m * (2ull * (unsigned long long)S +
                                                  (unsigned long long)l);
        bits[cat] += b;
        ++muls[cat];
        coeff_muls[cat] += (unsigned long long)m1 * (unsigned long long)m2;
        if (l < kHistLevels) { ++hist_cnt[cat][l]; hist_bits[cat][l] += b; }
        if (b > max_bits) {
            max_bits = b;
            max_m1 = m1;
            max_m2 = m2;
            max_cat = cat;
        }
    }
    unsigned long long total_bits() const
    {
        unsigned long long t = 0;
        for (int i = 0; i < CAT_N; ++i) t += bits[i];
        return t;
    }
    unsigned long long total_muls() const
    {
        unsigned long long t = 0;
        for (int i = 0; i < CAT_N; ++i) t += muls[i];
        return t;
    }
    unsigned long long total_coeff_muls() const
    {
        unsigned long long t = 0;
        for (int i = 0; i < CAT_N; ++i) t += coeff_muls[i];
        return t;
    }
};

Cost g_cost;                                 /* accumulated over all curves of a run */

/* ======================================================================== *
 *  BATCHED-structure cost MODEL (structure (b)).
 *
 *  This tool VERIFIES and MEASURES the simple structure (a): ONE product tree over
 *  all B2/D giant points, one remainder tree.  That is the structure whose leaves can
 *  be checked one by one against direct Horner evaluation, which is why it is the
 *  correctness reference.  The structure the GPU engine will implement is different
 *  (from Prime95's ecm.cpp, line references in tools/bench/stage2_shape_model.py):
 *
 *    F tree + its Newton inverse   built ONCE per curve          (ecm.cpp:9077, 9215)
 *    per outer loop (num_polyG):   a G tree over poly_size giant points
 *                                  + H = G*H mod F = 3 full-size mults (ecm.cpp:9443)
 *    scaled remainder descent      ONCE per curve, over the F tree (ecm.cpp:9518+)
 *
 *  There is exactly one such model, and it is a DERIVATION, not a measurement --
 *  this reference deliberately has no G/H machinery (see the header), so here the
 *  batched structure is only ACCOUNTED, with the same convention as the measured
 *  numbers above so the two are directly comparable.
 *
 *  Two conventions are printed, because they disagree and the disagreement matters:
 *    code : model_tree() below, which mirrors build_product_tree EXACTLY (power-of-two
 *           padding, the identity early-out, operand sizes as they really are, cost
 *           2*max(m1,m2)*(2S + ceil(log2 max(m1,m2)))).  Validated against the measured
 *           f_tree and giant_tree of a real run (cost_model_check line).
 *    plan : tools/bench/stage2_shape_model.py tree_cost(), which starts at m = 1 (so it
 *           charges the leaves as size-1 polynomials and every operand as 2^level).
 *           Printed so this tool's number can be compared with the plan's 3.75e11.
 *
 *  NOTE ON UNITS: these are OUR Kronecker operand-bits (the quantity our engine moves),
 *  NOT Prime95's internal cost -- Prime95 uses a two-level scheme (an outer ~2*poly_size
 *  complex FFT of full-width residues plus a per-coefficient inner FFT), so the two are
 *  not comparable numbers.
 * ======================================================================== */

unsigned long long bits_for_mul(unsigned long long m, long S)
{
    if (m <= 1) return 0;
    return 2ull * m * (2ull * (unsigned long long)S +
                       (unsigned long long)ceil_log2_bits(m));
}

/* Cost of building a product tree over n linear leaves.
     balanced = split n into ceil(n/2) and floor(n/2): what a real implementation (and
                the GPU engine) builds, so the operands are as even as possible;
     padded   = split at the power-of-two boundary, i.e. EXACTLY what
                build_product_tree() below does (n = 24 -> 16 + 8 -> an unbalanced
                17x9 root multiplication).  This is the recursion validated against the
                measured f_tree / giant_tree of a real run in cost_model_check.
   A node covering k real leaves has k+1 coefficients, and the multiplication is skipped
   when a side is empty (the poly_is_one early-out in poly_mul). */
unsigned long long model_tree(unsigned long long n, long S, bool balanced)
{
    if (n <= 1) return 0;
    unsigned long long left, right;
    if (balanced) {
        left = (n + 1) / 2;
        right = n / 2;
    } else {
        unsigned long long L = 1;
        while (L < n) L *= 2;
        const unsigned long long half = L / 2;
        left = std::min(n, half);
        right = (n > half) ? (n - half) : 0;
    }
    unsigned long long cost = 0;
    if (left >= 1 && right >= 1) cost += bits_for_mul(std::max(left, right) + 1, S);
    return cost + model_tree(left, S, balanced) + model_tree(right, S, balanced);
}

/* The same tree under the plan's convention (tools/bench/stage2_shape_model.py
   tree_cost()): it starts at m = 1 -- so it charges the leaves as size-1 polynomials and
   drops the leaf level entirely -- and charges every operand as 2^level.  Reproduced
   here so the difference between the conventions is visible instead of buried: for
   M5261 this function reproduces the plan's 3.753e11 exactly. */
unsigned long long model_tree_plan(unsigned long long p, long S)
{
    unsigned long long total = 0;
    for (unsigned long long m = 2; m * 2 <= p; m *= 2)
        total += (p / (2 * m)) * bits_for_mul(m, S);
    return total;
}

enum { TREE_BALANCED = 0, TREE_PADDED = 1, TREE_PLAN = 2 };
const char *const kTreeConvName[3] = { "ours-balanced", "ours-padded", "plan-script" };

struct BatchedModel {
    unsigned long long P = 0, giant_points = 0, num_poly_g = 0, loops = 0;
    unsigned long long f_tree = 0, g_tree = 0, fold = 0, descent = 0, small = 0;
    unsigned long long total() const { return f_tree + g_tree + fold + descent + small; }
};

/* P = poly_size = the baby count = phi(D)/2 (verified against Prime95's three published
   (D, poly_size) pairs in tools/bench/stage2_shape_model.py; here it is simply the size
   of the same set stage2_tree() builds). */
unsigned long long baby_count(uint64_t D)
{
    const uint64_t half = (D < 2) ? 1 : D / 2;
    unsigned long long n = 0;
    for (uint64_t j = 1; j <= half; ++j)
        if (gcd_u64(j, D) == 1) ++n;
    return n;
}

BatchedModel model_batched(unsigned long long P, unsigned long long giant_points, long S,
                           unsigned long long num_poly_g_override, int conv)
{
    BatchedModel m;
    m.P = P;
    m.giant_points = giant_points;
    const unsigned long long num_sections = (giant_points > 2) ? giant_points - 2 : 1;
    if (num_poly_g_override)
        m.num_poly_g = num_poly_g_override;
    else
        m.num_poly_g = std::max(2ull, (num_sections + P - 1) / (P ? P : 1));
    m.loops = std::max(1ull, m.num_poly_g - 1);

    if (conv == TREE_PLAN) {
        m.f_tree = model_tree_plan(P, S);
        /* "full-size" P x P multiplication: the plan charges m = P; our conventions
           charge the P+1 coefficients a degree-P polynomial really has */
        m.fold = m.loops * 3ull * bits_for_mul(P, S);
        m.small = P * 2ull * (2ull * (unsigned long long)S + 1ull);
    } else {
        m.f_tree = model_tree(P, S, conv == TREE_BALANCED);
        m.fold = m.loops * 3ull * bits_for_mul(P + 1, S);
        m.small = 0;
    }
    m.g_tree = m.loops * m.f_tree;
    /* ONE scaled remainder descent per curve.  Charged as 2x the F tree, as the plan does
       (this reference implements the SIMPLE remainder tree over the giant points, not the
       scaled descent, so it cannot measure this term). */
    m.descent = 2 * m.f_tree;
    return m;
}

void print_batched_model(const char *tag, const BatchedModel &m, long S)
{
    std::printf("cost_model: tree_convention=%s S=%ld P=phi(D)/2=%llu giant_points=%llu "
                "num_poly_g=%llu loops=%llu f_tree=%llu g_tree=%llu fold=%llu descent=%llu "
                "small=%llu total=%llu operand_bits_per_curve=%llu\n",
                tag, S, m.P, m.giant_points, m.num_poly_g, m.loops, m.f_tree, m.g_tree,
                m.fold, m.descent, m.small, m.total(), m.total());
}

/* ======================================================================== *
 *  polynomials mod a COMPOSITE N.  Coefficients are always kept reduced into
 *  [0, N).  The only two operations the algorithm needs are
 *      multiply      (schoolbook, one mpz_mul + one mpz_add per coefficient pair)
 *      divide        (reversed-polynomial Newton; the divisor is always MONIC)
 *  and both are counted for the cost model above.
 * ======================================================================== */

struct Zp {                                  /* one coefficient; RAII over mpz_t */
    mpz_t v;
    Zp() { mpz_init(v); }
    Zp(const Zp &o) { mpz_init_set(v, o.v); }
    Zp(Zp &&o) noexcept { mpz_init(v); mpz_swap(v, o.v); }
    Zp &operator=(const Zp &o) { if (this != &o) mpz_set(v, o.v); return *this; }
    Zp &operator=(Zp &&o) noexcept { mpz_swap(v, o.v); return *this; }
    ~Zp() { mpz_clear(v); }
};

using Poly = vector<Zp>;                     /* c[0] = constant term */

void poly_trim(Poly &a)                      /* drop high zero coefficients, keep 1 */
{
    while (a.size() > 1 && mpz_cmp_ui(a.back().v, 0) == 0) a.pop_back();
}

int poly_deg(const Poly &a) { return (int)a.size() - 1; }

bool poly_is_one(const Poly &a)
{
    return a.size() == 1 && mpz_cmp_ui(a[0].v, 1) == 0;
}

bool poly_is_zero(const Poly &a)
{
    return a.size() == 1 && mpz_cmp_ui(a[0].v, 0) == 0;
}

bool poly_equal(const Poly &a, const Poly &b)
{
    Poly x = a, y = b;
    poly_trim(x);
    poly_trim(y);
    if (x.size() != y.size()) return false;
    for (size_t i = 0; i < x.size(); ++i)
        if (mpz_cmp(x[i].v, y[i].v) != 0) return false;
    return true;
}

/* r = a*b mod N, schoolbook.  The identity polynomial 1 is special-cased (exact, and
   it is what the padding of a power-of-two product tree produces -- charging a
   multiplication by 1 would inflate the plan's cost model). */
Poly poly_mul(const Poly &a, const Poly &b, const mpz_t N, int cat)
{
    if (poly_is_one(a)) return b;
    if (poly_is_one(b)) return a;
    Poly r(a.size() + b.size() - 1);
    mpz_t t;
    mpz_init(t);
    for (size_t i = 0; i < a.size(); ++i) {
        if (mpz_cmp_ui(a[i].v, 0) == 0) continue;         /* zero coefficients are free */
        for (size_t j = 0; j < b.size(); ++j) {
            mpz_mul(t, a[i].v, b[j].v);
            mpz_add(r[i + j].v, r[i + j].v, t);
        }
    }
    mpz_clear(t);
    for (size_t k = 0; k < r.size(); ++k) mpz_mod(r[k].v, r[k].v, N);
    poly_trim(r);
    g_cost.add(cat, a.size(), b.size());
    return r;
}

Poly poly_add(const Poly &a, const Poly &b, const mpz_t N)
{
    Poly r(std::max(a.size(), b.size()));
    for (size_t i = 0; i < r.size(); ++i) {
        if (i < a.size()) mpz_set(r[i].v, a[i].v);
        if (i < b.size()) mpz_add(r[i].v, r[i].v, b[i].v);
        mpz_mod(r[i].v, r[i].v, N);
    }
    poly_trim(r);
    return r;
}

/* g = 1/a mod X^k, with coefficients mod N.  Requires a[0] invertible mod N.
   Newton doubling: g <- g * (2 - a*g) mod X^(2*len(g)).  For this file a[0] is
   always 1 (the reversed divisor starts with the MONIC leading coefficient), so
   the inversion below cannot fail -- mod a composite N that is the entire point. */
Poly poly_inv_series(const Poly &a, size_t k, const mpz_t N, int cat)
{
    Poly g(1);
    if (mpz_invert(g[0].v, a[0].v, N) == 0) {
        std::printf("stage2_tree_ref: FATAL: leading coefficient not invertible mod N "
                    "(the divisor was not monic)\n");
        std::exit(4);
    }
    while (g.size() < k) {
        const size_t nxt = std::min(2 * g.size(), k);
        Poly at(a.begin(), a.begin() + (long)std::min(a.size(), nxt));
        Poly ag = poly_mul(at, g, N, cat);       /* mod X^nxt */
        ag.resize(nxt);
        Poly h(nxt);                             /* h = 2 - ag */
        mpz_set_ui(h[0].v, 2);
        for (size_t i = 0; i < nxt; ++i) {
            mpz_sub(h[i].v, h[i].v, ag[i].v);
            mpz_mod(h[i].v, h[i].v, N);          /* mpz_mod is non-negative */
        }
        Poly gn = poly_mul(g, h, N, cat);        /* mod X^nxt */
        /* pad/truncate to EXACTLY nxt and do NOT trim: the inverse of a series can have
           zero coefficients anywhere, so trimming here would shrink g back and make the
           while loop above spin forever (measured: --selftest hung for > 300 s on the
           first random divisor whose inverse ended in a zero coefficient -- the frozen
           end-to-end case never hit it, which is exactly why a random divisor must be
           part of the selftest). */
        gn.resize(nxt);
        g = gn;
    }
    return g;
}

/* a = q*b + r with deg r < deg b.  b must have an invertible leading coefficient;
   everywhere in this file b is MONIC.  Standard reversed-polynomial Newton:
       rev(q) = rev(a) * rev(b)^(-1)   mod X^(deg a - deg b + 1)
   which is exactly why a composite N is no obstacle as long as the divisor is monic
   (no leading-coefficient inversion ever has to be attempted).  The low part of q*b
   is what the remainder needs; the full product is computed anyway (we do NOT use the
   middle-product optimisation, so the cost figure for this step is an upper bound). */
void poly_divmod(Poly &q, Poly &r, const Poly &a, const Poly &b, const mpz_t N, int cat)
{
    const int da = poly_deg(a), db = poly_deg(b);
    if (da < db) {
        q.assign(1, Zp());
        r = a;
        poly_trim(r);
        return;
    }
    const size_t k = (size_t)(da - db + 1);
    Poly ra(k), rb(b.size());
    for (size_t i = 0; i < k; ++i) mpz_set(ra[i].v, a[(size_t)(da - (int)i)].v);
    for (size_t i = 0; i < b.size(); ++i) mpz_set(rb[i].v, b[(size_t)(db - (int)i)].v);
    Poly rbi = poly_inv_series(rb, k, N, cat);
    Poly qrev = poly_mul(ra, rbi, N, cat);
    qrev.resize(k);
    q.assign(k, Zp());
    for (size_t i = 0; i < k; ++i) mpz_set(q[i].v, qrev[k - 1 - i].v);
    poly_trim(q);

    Poly qb = poly_mul(q, b, N, cat);
    if (db == 0) {
        r.assign(1, Zp());
    } else {
        r.assign((size_t)db, Zp());
        for (size_t i = 0; i < (size_t)db; ++i) {
            mpz_sub(r[i].v, a[i].v, qb[i].v);
            mpz_mod(r[i].v, r[i].v, N);
        }
    }
    poly_trim(r);
}

/* r = a mod b (b monic).  Fast path when the degree already fits: no division, no
   multiplication -- this is what makes a shape with deg F << number of giant points
   cheap in the descent. */
Poly poly_mod(const Poly &a, const Poly &b, const mpz_t N, int cat)
{
    if (poly_is_one(b)) return Poly(1);              /* a mod 1 = 0 */
    if (poly_deg(a) < poly_deg(b)) return a;
    Poly q, r;
    poly_divmod(q, r, a, b, N, cat);
    return r;
}

/* Direct Horner evaluation -- the independent oracle for the remainder tree. */
void poly_eval_horner(mpz_t out, const Poly &f, const mpz_t x, const mpz_t N)
{
    mpz_set(out, f.back().v);
    for (int k = (int)f.size() - 2; k >= 0; --k) {
        mpz_mul(out, out, x);
        mpz_add(out, out, f[k].v);
        mpz_mod(out, out, N);
    }
}

/* ======================================================================== *
 *  product tree (power-of-two padded heap; t[1] is the root, t[L + i] the
 *  i-th leaf).  Padding leaves are the constant 1, so they change nothing
 *  mathematically and are skipped in the descent.
 * ======================================================================== */

struct PolyTree {
    size_t n = 0;                    /* real leaves */
    size_t leaves = 0;               /* padded leaf count = 2^ceil(log2 n) */
    vector<Poly> t;                  /* heap, t[0] unused */
};

void build_product_tree(PolyTree &T, const vector<Poly> &leaf, const mpz_t N, int cat)
{
    const size_t n = leaf.size();
    size_t L = 1;
    while (L < n) L *= 2;
    T.n = n;
    T.leaves = L;
    T.t.assign(2 * L, Poly(1));
    for (size_t i = 0; i < 2 * L; ++i) mpz_set_ui(T.t[i][0].v, 1);
    for (size_t i = 0; i < n; ++i) T.t[L + i] = leaf[i];
    for (size_t i = L; i-- > 1; ) T.t[i] = poly_mul(T.t[2 * i], T.t[2 * i + 1], N, cat);
}

/* Descend F through the tree; out[i] = F(x_i) as a constant for the real leaves.
   The root leftover F mod root is charged to the top-level bucket (the counterpart of
   the plan's full-size top-level step), everything below it to the remainder tree. */
vector<Zp> remainder_tree(const Poly &F, const PolyTree &T, const mpz_t N)
{
    vector<Zp> out(T.n);
    vector<Poly> cur(1);
    cur[0] = poly_mod(F, T.t[1], N, CAT_TOP);
    size_t base = 1, cnt = 1;
    while (base < T.leaves) {
        const size_t nbase = base * 2;
        vector<Poly> nxt(cnt * 2);
        for (size_t j = 0; j < cnt; ++j) {
            for (int s = 0; s < 2; ++s) {
                const size_t ci = nbase + 2 * j + (size_t)s;
                if (poly_is_one(T.t[ci])) {
                    /* padding subtree: F mod 1 = 0 (and no real leaf is in here) */
                    nxt[2 * j + (size_t)s] = Poly(1);
                } else {
                    nxt[2 * j + (size_t)s] = poly_mod(cur[j], T.t[ci], N, CAT_REM);
                }
            }
        }
        cur.swap(nxt);
        base = nbase;
        cnt *= 2;
    }
    for (size_t i = 0; i < T.n; ++i)
        if (!cur[i].empty()) mpz_set(out[i].v, cur[i][0].v);
    return out;
}

/* ======================================================================== *
 *  stage 2 proper: F tree + giant tree + remainder tree + accumulation
 * ======================================================================== */

struct TreeStats {
    size_t baby_j = 0;
    size_t giant_i = 0;
    size_t f_degree = 0;
    unsigned long long naive_points = 0;
    unsigned long long naive_mismatches = 0;
    double naive_elapsed = 0.0;
};

/* ======================================================================== *
 *  --dump-F <file>: everything the GPU tree engine (tools/bench/stage2_tree_gpu.cu, M3
 *  slice S1 of docs/architecture/STAGE2.md) needs to reproduce THIS F
 *  independently -- the modulus, the curve's a24, the stage-2 input point Q (the affine x
 *  of [s]P, i.e. exactly what a save file carries), the baby index set with its x_j, and
 *  the coefficients of F.
 *
 *  The GPU side re-derives every x_j from that same Q with its own x-only Montgomery
 *  ladder and rebuilds F with its own NTT product tree, then compares the two
 *  coefficient lists one by one mod N -- and the two files can also be compared line by
 *  line, since the GPU writes the identical format.  Handing over Q (rather than the
 *  points) is what makes the comparison prove something about the LADDER too, and it is
 *  the convention the plan asks for: one entry point for the points, section 18.3.
 *
 *  Nothing is printed here, so a run with --dump-F has byte-identical stdout.
 * ======================================================================== */

void dump_F_file(const char *path, const mpz_t N, const mpz_t a24, const mpz_t Qx, uint64_t D,
                 uint64_t B1, uint64_t B2, uint64_t sigma, const vector<uint64_t> &baby_idx,
                 const vector<Poly> &f_leaves, const Poly &F)
{
    FILE *f = std::fopen(path, "wb");
    if (!f) {
        std::printf("stage2_tree_ref: cannot write --dump-F file %s\n", path);
        return;
    }
    auto put_hex = [&](const char *tag, const mpz_t v) {
        char *s = mpz_get_str(nullptr, 16, v);
        std::fprintf(f, "%s %s\n", tag, s);
        void (*ff)(void *, size_t) = nullptr;
        mp_get_memory_functions(nullptr, nullptr, &ff);
        ff(s, std::strlen(s) + 1);
    };
    std::fprintf(f, "stage2_tree_F_dump v1\n");
    {
        /* N in DECIMAL under the N_dec key (the GPU reader parses that key base 10) and in
           hex under N_hex; everything else in the file is hex. */
        char *s = mpz_get_str(nullptr, 10, N);
        std::fprintf(f, "N_dec %s\n", s);
        void (*ff)(void *, size_t) = nullptr;
        mp_get_memory_functions(nullptr, nullptr, &ff);
        ff(s, std::strlen(s) + 1);
    }
    put_hex("N_hex", N);
    std::fprintf(f, "N_bits %ld\n", (long)mpz_sizeinbase(N, 2));
    std::fprintf(f, "D %llu\nB1 %llu\nB2 %llu\nsigma %llu\n", (unsigned long long)D,
                 (unsigned long long)B1, (unsigned long long)B2, (unsigned long long)sigma);
    put_hex("a24_hex", a24);
    put_hex("Q_hex", Qx);
    std::fprintf(f, "baby_count %llu\n", (unsigned long long)baby_idx.size());
    mpz_t x;
    mpz_init(x);
    for (size_t k = 0; k < baby_idx.size(); ++k) {
        mpz_neg(x, f_leaves[k][0].v);          /* the leaf is (X - x_j) */
        mpz_mod(x, x, N);
        char *s = mpz_get_str(nullptr, 16, x);
        std::fprintf(f, "baby %llu %s\n", (unsigned long long)baby_idx[k], s);
        void (*ff)(void *, size_t) = nullptr;
        mp_get_memory_functions(nullptr, nullptr, &ff);
        ff(s, std::strlen(s) + 1);
    }
    mpz_clear(x);
    std::fprintf(f, "F_degree %d\n", poly_deg(F));
    for (int k = 0; k <= poly_deg(F); ++k) {
        char *s = mpz_get_str(nullptr, 16, F[(size_t)k].v);
        std::fprintf(f, "F %s\n", s);
        void (*ff)(void *, size_t) = nullptr;
        mp_get_memory_functions(nullptr, nullptr, &ff);
        ff(s, std::strlen(s) + 1);
    }
    std::fprintf(f, "end\n");
    std::fclose(f);
}

Result stage2_tree(const mpz_t N, uint64_t B1, uint64_t B2, uint64_t D, const Pt &Q,
                   const mpz_t a24, bool naive_check, TreeStats &st)
{
    Result res;
    const uint64_t half = (D < 2) ? 1 : D / 2;
    const uint64_t imax = B2 / D + 2;

    /* ---- 0. primes p <= D/2: no i >= 1 exists for them, so the polynomial product
       cannot reach them.  Handled exactly like the pairing path (direct ladder + gcd).
       Empty whenever B1 >= D/2, which includes the frozen configuration. ---------- */
    {
        const vector<uint64_t> smallp = primes_up_to(half);
        mpz_t g;
        mpz_init(g);
        for (uint64_t p : smallp) {
            if (p <= B1 || p > B2) continue;
            Pt R;
            xmul_u64(R, p, Q, a24, N);
            mpz_gcd(g, R.Z, N);
            if (mpz_cmp_ui(g, 1) > 0 && mpz_cmp(g, N) < 0) record(res, g, p, N);
        }
        mpz_clear(g);
    }

    /* ---- 1. baby set and baby points ------------------------------------------ */
    vector<uint64_t> baby_idx;
    for (uint64_t j = 1; j <= half; ++j)
        if (gcd_u64(j, D) == 1) baby_idx.push_back(j);
    st.baby_j = baby_idx.size();

    vector<Poly> f_leaves(baby_idx.size());
    for (size_t k = 0; k < baby_idx.size(); ++k) {
        Pt R;
        xmul_u64(R, baby_idx[k], Q, a24, N);
        mpz_t x;
        mpz_init(x);
        affine_x(x, R, N);                            /* x_j */
        f_leaves[k].assign(2, Zp());
        mpz_neg(f_leaves[k][0].v, x);                 /* (X - x_j) */
        mpz_mod(f_leaves[k][0].v, f_leaves[k][0].v, N);
        mpz_set_ui(f_leaves[k][1].v, 1);
        mpz_clear(x);
    }

    /* ---- 2. F(X) = prod (X - x_j) mod N --------------------------------------- */
    PolyTree FT;
    build_product_tree(FT, f_leaves, N, CAT_FTREE);
    const Poly F = FT.t[1];
    st.f_degree = (size_t)poly_deg(F);
    /* --dump-F: hand F, Q, a24 and the baby points to the GPU twin BEFORE the leaves are
       released (they are the only copy of x_j; F is rebuilt from them). */
    if (g_dump_F) dump_F_file(g_dump_F, N, a24, Q.X, D, B1, B2, g_cur_sigma, baby_idx, f_leaves, F);
    vector<Poly>().swap(f_leaves);                    /* the tree owns the leaves now */

    /* ---- 3. giant points (same differential-addition chain as the pairing path) - */
    vector<Pt *> giant((size_t)imax + 1, nullptr);
    for (uint64_t i = 1; i <= imax; ++i) {
        giant[(size_t)i] = new Pt();
        if (i == 1) {
            xmul_u64(*giant[1], D, Q, a24, N);
        } else if (i == 2) {
            xdbl(*giant[2], *giant[1], a24, N);
        } else {
            xadd(*giant[(size_t)i], *giant[(size_t)(i - 1)], *giant[1],
                 *giant[(size_t)(i - 2)], N);
        }
    }
    st.giant_i = (size_t)imax;

    vector<Poly> g_leaves((size_t)imax);
    vector<Zp> gx((size_t)imax);                      /* affine x_i, for --naive-check */
    for (uint64_t i = 1; i <= imax; ++i) {
        mpz_t x;
        mpz_init(x);
        affine_x(x, *giant[(size_t)i], N);
        mpz_set(gx[(size_t)(i - 1)].v, x);
        g_leaves[(size_t)(i - 1)].assign(2, Zp());
        mpz_neg(g_leaves[(size_t)(i - 1)][0].v, x);
        mpz_mod(g_leaves[(size_t)(i - 1)][0].v, g_leaves[(size_t)(i - 1)][0].v, N);
        mpz_set_ui(g_leaves[(size_t)(i - 1)][1].v, 1);
        mpz_clear(x);
    }
    for (uint64_t i = 1; i <= imax; ++i) { delete giant[(size_t)i]; giant[(size_t)i] = nullptr; }

    /* ---- 4. product tree over the giant points + remainder-tree evaluation ----- */
    PolyTree GT;
    build_product_tree(GT, g_leaves, N, CAT_GTREE);
    vector<Poly>().swap(g_leaves);
    vector<Zp> values = remainder_tree(F, GT, N);

    /* ---- 4b. --naive-check: the remainder tree against direct Horner ---------- */
    if (naive_check) {
        const double t0 = now_s();
        mpz_t h;
        mpz_init(h);
        for (size_t k = 0; k < values.size(); ++k) {
            poly_eval_horner(h, F, gx[k].v, N);
            if (mpz_cmp(h, values[k].v) != 0) {
                if (st.naive_mismatches < 3) {
                    char *a = mpz_get_str(nullptr, 16, h);
                    char *b = mpz_get_str(nullptr, 16, values[k].v);
                    std::printf("naive_mismatch: sigma=%llu i=%llu horner=0x%s tree=0x%s\n",
                                (unsigned long long)g_cur_sigma,
                                (unsigned long long)(k + 1), a, b);
                    void (*ff)(void *, size_t) = nullptr;
                    mp_get_memory_functions(nullptr, nullptr, &ff);
                    ff(a, std::strlen(a) + 1);
                    ff(b, std::strlen(b) + 1);
                }
                ++st.naive_mismatches;
            }
        }
        mpz_clear(h);
        st.naive_points = values.size();
        st.naive_elapsed = now_s() - t0;
        std::printf("stage2_naive: sigma=%llu giant_points=%llu horner_calls=%llu "
                    "mismatches=%llu elapsed=%.2f\n",
                    (unsigned long long)g_cur_sigma,
                    (unsigned long long)values.size(),
                    (unsigned long long)values.size(),
                    (unsigned long long)st.naive_mismatches, st.naive_elapsed);
    }

    /* ---- 5. accumulate + gcd, block by block (naming the culprit like pairing) - */
    const size_t BLOCK = 4096;
    mpz_t prod, g;
    mpz_inits(prod, g, nullptr);
    mpz_set_ui(prod, 1);
    vector<size_t> block;

    auto name_culprit = [&](size_t leaf) {
        /* leaf index -> giant index i = leaf + 1; the value F(x_i) shares a factor with
           N.  Attribute it to every stage-2 prime i*D -+ j that maps to this giant index
           and CONFIRM each with an independent ladder + gcd of Z, exactly as the pairing
           path names its culprit.  If no prime maps here (a composite candidate), the
           factor is still real -- record it with prime = 0. */
        mpz_t gi2, pg;
        mpz_inits(gi2, pg, nullptr);
        mpz_gcd(gi2, values[leaf].v, N);
        if (mpz_cmp_ui(gi2, 1) > 0 && mpz_cmp(gi2, N) < 0) {
            const uint64_t i = (uint64_t)leaf + 1;
            const uint64_t off = i * D;
            bool named = false;
            for (uint64_t j : baby_idx) {
                for (int s = 0; s < 2; ++s) {
                    if (s == 0 && off < j) continue;
                    const uint64_t p = (s == 0) ? (off - j) : (off + j);
                    if (p <= B1 || p > B2) continue;
                    if (!is_prime_u64(p)) continue;
                    Pt R;
                    xmul_u64(R, p, Q, a24, N);
                    mpz_gcd(pg, R.Z, N);
                    if (mpz_cmp_ui(pg, 1) > 0 && mpz_cmp(pg, N) < 0) {
                        record(res, pg, p, N);
                        named = true;
                    }
                }
            }
            if (!named) record(res, gi2, 0, N);
        }
        mpz_clears(gi2, pg, nullptr);
    };

    auto flush = [&]() {
        if (block.empty()) return;
        mpz_gcd(g, prod, N);
        if (mpz_cmp_ui(g, 1) > 0 && mpz_cmp(g, N) < 0)
            for (size_t leaf : block) name_culprit(leaf);
        block.clear();
        mpz_set_ui(prod, 1);
    };

    for (size_t k = 0; k < values.size(); ++k) {
        mpz_mul(prod, prod, values[k].v);
        mpz_mod(prod, prod, N);
        block.push_back(k);
        if (block.size() >= BLOCK) flush();
    }
    flush();
    mpz_clears(prod, g, nullptr);
    return res;
}

/* ======================================================================== *
 *  checks (same [ok]/[FAIL] convention as stage2_ref.cpp --selftest)
 * ======================================================================== */

int g_checks = 0, g_fails = 0;
void check(const char *name, bool ok, const string &detail = "")
{
    ++g_checks;
    if (ok) {
        std::printf("  [ok]   %s\n", name);
    } else {
        ++g_fails;
        std::printf("  [FAIL] %s%s%s\n", name, detail.empty() ? "" : " -- ", detail.c_str());
    }
}

/* deterministic pseudo-random numbers for the selftest (reproducible failure) */
uint64_t g_rng = 0x123456789ABCDEFull;
uint64_t rnd()
{
    g_rng = g_rng * 6364136223846793005ull + 1442695040888963407ull;
    return g_rng >> 11;
}

Poly poly_from_u64(const vector<uint64_t> &c, const mpz_t N)
{
    Poly p(c.size());
    for (size_t i = 0; i < c.size(); ++i) {
        mpz_set_ui(p[i].v, 0);
        set_u64(p[i].v, c[i]);
        mpz_mod(p[i].v, p[i].v, N);
    }
    poly_trim(p);
    return p;
}

/* Leaves (X - root) for a list of roots, then the product tree over them. */
Poly tree_product(const vector<Zp> &roots, const mpz_t N, int cat)
{
    vector<Poly> leaves(roots.size());
    for (size_t i = 0; i < roots.size(); ++i) {
        leaves[i].assign(2, Zp());
        mpz_neg(leaves[i][0].v, roots[i].v);
        mpz_mod(leaves[i][0].v, leaves[i][0].v, N);
        mpz_set_ui(leaves[i][1].v, 1);
    }
    PolyTree T;
    build_product_tree(T, leaves, N, cat);
    return T.t[1];
}

/* One full "tree stage 2" of the polynomial core over an arbitrary modulus and an
   arbitrary (roots, points) pair -- the unit-level oracle for steps 2-4.  Returns the
   number of mismatches against direct Horner and sets *all_roots_ok. */
unsigned long long tree_vs_horner(const mpz_t N, const vector<Zp> &roots,
                                  const vector<Zp> &points, bool &all_roots_ok)
{
    const Poly F = tree_product(roots, N, CAT_FTREE);
    vector<Poly> leaves(points.size());
    for (size_t i = 0; i < points.size(); ++i) {
        leaves[i].assign(2, Zp());
        mpz_neg(leaves[i][0].v, points[i].v);
        mpz_mod(leaves[i][0].v, leaves[i][0].v, N);
        mpz_set_ui(leaves[i][1].v, 1);
    }
    PolyTree T;
    build_product_tree(T, leaves, N, CAT_GTREE);

    /* an independent invariant that needs no reference: F vanishes at every root */
    all_roots_ok = true;
    {
        vector<Poly> rl(roots.size());
        for (size_t i = 0; i < roots.size(); ++i) {
            rl[i].assign(2, Zp());
            mpz_neg(rl[i][0].v, roots[i].v);
            mpz_mod(rl[i][0].v, rl[i][0].v, N);
            mpz_set_ui(rl[i][1].v, 1);
        }
        PolyTree RT;
        build_product_tree(RT, rl, N, CAT_GTREE);
        const vector<Zp> rv = remainder_tree(F, RT, N);
        for (size_t i = 0; i < rv.size(); ++i)
            if (mpz_cmp_ui(rv[i].v, 0) != 0) all_roots_ok = false;
    }

    const vector<Zp> vals = remainder_tree(F, T, N);
    unsigned long long mism = 0;
    mpz_t h;
    mpz_init(h);
    for (size_t i = 0; i < vals.size(); ++i) {
        poly_eval_horner(h, F, points[i].v, N);
        if (mpz_cmp(h, vals[i].v) != 0) ++mism;
    }
    mpz_clear(h);
    return mism;
}

/* ======================================================================== *
 *  one curve: curve setup + stage 1 + stage 2 (mirrors stage2_ref.cpp's main
 *  loop so both tools can be driven from the same script)
 * ======================================================================== */

struct CurveRun {
    Result res;
    TreeStats st;
    bool processed = false;      /* false: nothing to do (stage 1 already finished) */
};

CurveRun run_curve(const mpz_t N, uint64_t sigma, uint64_t B1, uint64_t B2, uint64_t D,
                   const vector<uint8_t> &s_bits, const string *save_x_hex,
                   bool naive_check)
{
    CurveRun cr;
    mpz_t a24, factor;
    mpz_inits(a24, factor, nullptr);
    Pt P;
    if (suyama_curve(a24, P, sigma, N, factor) == 1) {
        record(cr.res, factor, 0, N);                 /* degenerate sigma: it IS a factor */
        cr.processed = true;
        mpz_clears(a24, factor, nullptr);
        return cr;
    }
    Pt Qa;
    if (save_x_hex && !save_x_hex->empty()) {
        if (mpz_set_str(Qa.X, save_x_hex->c_str(), 16) != 0) {
            std::printf("stage2_tree_ref: unparsable X in the save: %s\n", save_x_hex->c_str());
            mpz_clears(a24, factor, nullptr);
            return cr;
        }
        mpz_set_ui(Qa.Z, 1);
    } else {
        Pt Q;
        ladder(Q, s_bits, P, a24, N);                 /* [s]P = the save's point */
        if (mpz_cmp_ui(Q.Z, 0) == 0) {                /* stage 1 already found it */
            mpz_clears(a24, factor, nullptr);
            return cr;
        }
        mpz_t x;
        mpz_init(x);
        affine_x(x, Q, N);
        mpz_set(Qa.X, x);
        mpz_set_ui(Qa.Z, 1);
        mpz_clear(x);
    }
    cr.res = stage2_tree(N, B1, B2, D, Qa, a24, naive_check, cr.st);
    cr.processed = true;
    mpz_clears(a24, factor, nullptr);
    return cr;
}

/* ---- save file parsing (identical convention to stage2_ref.cpp) ---------- */

struct SaveCurve {
    uint64_t sigma = 0;
    uint64_t b1 = 0;
    string x_hex;
    bool have_x = false;
};

vector<SaveCurve> parse_save(const string &path, uint64_t first, uint64_t count)
{
    vector<SaveCurve> out;
    FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) return out;
    char line[8192];
    uint64_t seen = 0;
    while (std::fgets(line, sizeof(line), f)) {
        const char *ps = std::strstr(line, "SIGMA=");
        const char *pb = std::strstr(line, "B1=");
        const char *px = std::strstr(line, "X=0x");
        if (!ps || !pb) continue;
        if (seen++ < first) continue;
        if (out.size() >= count) break;
        SaveCurve c;
        c.sigma = std::strtoull(ps + 6, nullptr, 10);
        c.b1 = std::strtoull(pb + 3, nullptr, 10);
        if (px) {
            /* px points at "X=0x", so skip FOUR characters: keeping the "0x" prefix makes
               mpz_set_str(..., 16) fail and the point silently becomes 0 (that trap cost
               a debugging round in stage2_ref -- measured: the save-driven run found
               nothing while the identical synthetic run found the factor). */
            c.x_hex = px + 4;
            const size_t stop = c.x_hex.find_first_of(";\r\n ");
            if (stop != string::npos) c.x_hex.resize(stop);
            c.have_x = !c.x_hex.empty();
        }
        out.push_back(c);
    }
    std::fclose(f);
    return out;
}

void print_cost_line()
{
    std::printf("cost: S=%ld poly_muls=%llu operand_bits=%llu f_tree=%llu giant_tree=%llu "
                "remainder=%llu top_level=%llu coeff_muls=%llu max_mul=%llux%llu "
                "max_mul_bits=%llu\n",
                g_cost.S, g_cost.total_muls(), g_cost.total_bits(),
                g_cost.bits[CAT_FTREE], g_cost.bits[CAT_GTREE], g_cost.bits[CAT_REM],
                g_cost.bits[CAT_TOP], g_cost.total_coeff_muls(),
                g_cost.max_m1, g_cost.max_m2, g_cost.max_bits);
}

void print_cost_detail()
{
    std::printf("cost_detail: category muls operand_bits coeff_muls\n");
    for (int c = 0; c < CAT_N; ++c)
        std::printf("cost_detail: %-11s %llu %llu %llu\n", kCatName[c], g_cost.muls[c],
                    g_cost.bits[c], g_cost.coeff_muls[c]);
    for (int c = 0; c < CAT_N; ++c)
        for (int l = 0; l < kHistLevels; ++l)
            if (g_cost.hist_cnt[c][l])
                std::printf("cost_level: cat=%s log2m=%d muls=%llu operand_bits=%llu\n",
                            kCatName[c], l, g_cost.hist_cnt[c][l], g_cost.hist_bits[c][l]);
}

void print_result(const char *algo, const Result &r, uint64_t curves, double elapsed,
                  unsigned long long operand_bits)
{
    std::printf("stage2: algorithm=%s curves=%llu hits=%llu bad_factors=%llu factors=",
                algo, (unsigned long long)curves, (unsigned long long)r.hits,
                (unsigned long long)r.bad_factors);
    for (size_t i = 0; i < r.factors.size(); ++i)
        std::printf("%s%s", i ? "," : "", r.factors[i].c_str());
    std::printf(" hit_primes=");
    for (size_t i = 0; i < r.hit_primes.size() && i < 32; ++i)
        std::printf("%s%llu", i ? "," : "", (unsigned long long)r.hit_primes[i]);
    std::printf(" elapsed=%.2f operand_bits=%llu\n", elapsed, operand_bits);
}

/* ---- end to end on 2^128+1 (same frozen configuration as stage2_ref) ------ */

struct E2E {
    Result res;
    TreeStats st;
    uint64_t curves = 0;
};

E2E run_known_factor(uint64_t B1, uint64_t B2, uint64_t D, uint64_t sigma, bool naive)
{
    E2E e;
    mpz_t N;
    mpz_init(N);
    mpz_set_str(N, "340282366920938463463374607431768211457", 10);   /* 2^128+1 */
    const vector<uint8_t> s_bits = build_s_bits(B1, 1);
    g_cur_sigma = sigma;
    const CurveRun cr = run_curve(N, sigma, B1, B2, D, s_bits, nullptr, naive);
    e.res = cr.res;
    e.st = cr.st;
    e.curves = cr.processed ? 1 : 0;
    mpz_clear(N);
    return e;
}

/* ======================================================================== *
 *  selftest
 * ======================================================================== */

int selftest()
{
    g_cost.S = 61;
    std::printf("stage2_tree_ref --selftest\n\n"
                "[1] polynomial core (product tree + remainder tree) against independent "
                "oracles\n");

    /* (a) schoolbook multiplication vs pointwise evaluation, over a prime field.
       Two different computations of the same thing: coefficient convolution and
       evaluation -- a wrong convolution cannot survive this. */
    {
        const char *pp = "2305843009213693951";        /* M61, prime */
        mpz_t p;
        mpz_init_set_str(p, pp, 10);
        bool ok = true;
        string detail;
        for (int trial = 0; trial < 20 && ok; ++trial) {
            vector<uint64_t> ca(1 + (rnd() % 6)), cb(1 + (rnd() % 6));
            for (uint64_t &v : ca) v = rnd() % 1000000007ull;
            for (uint64_t &v : cb) v = rnd() % 1000000007ull;
            const Poly a = poly_from_u64(ca, p), b = poly_from_u64(cb, p);
            const Poly ab = poly_mul(a, b, p, CAT_FTREE);
            for (int k = 0; k < 5; ++k) {
                Zp x;
                set_u64(x.v, rnd() % 1000000007ull);
                mpz_t ea, eb, eab;
                mpz_inits(ea, eb, eab, nullptr);
                poly_eval_horner(ea, a, x.v, p);
                poly_eval_horner(eb, b, x.v, p);
                poly_eval_horner(eab, ab, x.v, p);
                mpz_mul(ea, ea, eb);
                mpz_mod(ea, ea, p);
                if (mpz_cmp(ea, eab) != 0) {
                    ok = false;
                    char *s1 = mpz_get_str(nullptr, 10, ea);
                    char *s2 = mpz_get_str(nullptr, 10, eab);
                    detail = string("mul vs eval: ") + s1 + " != " + s2;
                    void (*ff)(void *, size_t) = nullptr;
                    mp_get_memory_functions(nullptr, nullptr, &ff);
                    ff(s1, std::strlen(s1) + 1);
                    ff(s2, std::strlen(s2) + 1);
                }
                mpz_clears(ea, eb, eab, nullptr);
            }
        }
        check("poly_mul == pointwise evaluation of the two factors (prime modulus)", ok, detail);

        /* (b) division: a == q*b + r with deg r < deg b, b monic.  The divisor must have
           degree >= 1 for that statement to mean anything: the zero polynomial is
           represented here as the single coefficient 0, so poly_deg() reports 0 for it
           (measured: with a degree-0 divisor the check below failed on a CORRECT
           division -- r = 0 with deg b = 0).*/
        bool ok2 = true;
        string d2;
        for (int trial = 0; trial < 20 && ok2; ++trial) {
            vector<uint64_t> ca(1 + (rnd() % 16)), cb(2 + (rnd() % 7));
            for (uint64_t &v : ca) v = rnd() % 1000000007ull;
            for (uint64_t &v : cb) v = rnd() % 1000000007ull;
            Poly a = poly_from_u64(ca, p);
            Poly b = poly_from_u64(cb, p);
            mpz_set_ui(b.back().v, 1);                  /* force the divisor monic */
            Poly q, r;
            poly_divmod(q, r, a, b, p, CAT_REM);
            /* Every failure message carries the degrees and the coefficient counts: the
               first version of this check reported only "deg r >= deg b", which left the
               next reader guessing whether the division or the check was wrong. */
            const string dims = "deg a=" + std::to_string(poly_deg(a)) +
                                " deg b=" + std::to_string(poly_deg(b)) +
                                " deg q=" + std::to_string(poly_deg(q)) +
                                " deg r=" + std::to_string(poly_deg(r)) +
                                " (sizes " + std::to_string(a.size()) + "," +
                                std::to_string(b.size()) + "," + std::to_string(q.size()) +
                                "," + std::to_string(r.size()) + ")";
            if (poly_deg(r) >= poly_deg(b)) { ok2 = false; d2 = "deg r >= deg b: " + dims; break; }
            Poly recon = poly_add(poly_mul(q, b, p, CAT_REM), r, p);
            if (!poly_equal(recon, a)) { ok2 = false; d2 = "q*b + r != a: " + dims; break; }
        }
        check("poly_divmod: a == q*b + r and deg r < deg b (b monic)", ok2, d2);

        /* (c) exact multiples give a zero remainder */
        bool ok3 = true;
        for (int trial = 0; trial < 10 && ok3; ++trial) {
            vector<uint64_t> ca(1 + (rnd() % 6)), cb(1 + (rnd() % 6));
            for (uint64_t &v : ca) v = rnd() % 1000000007ull;
            for (uint64_t &v : cb) v = rnd() % 1000000007ull;
            Poly b = poly_from_u64(cb, p);
            mpz_set_ui(b.back().v, 1);
            const Poly a = poly_mul(poly_from_u64(ca, p), b, p, CAT_REM);
            if (!poly_is_zero(poly_mod(a, b, p, CAT_REM))) ok3 = false;
        }
        check("poly_mod of an exact multiple is the zero polynomial", ok3);
        mpz_clear(p);
    }

    /* (d) the WHOLE core (F tree -> giant tree -> remainder tree) against Horner, over
       a prime modulus and over a COMPOSITE modulus (N = p1*p2, where the reversion
       trick is the only reason division works at all). */
    {
        const char *p1s = "2305843009213693951";       /* M61 */
        const char *p2s = "2147483647";                /* 2^31-1 */
        mpz_t p1, p2, comp;
        mpz_inits(p1, p2, comp, nullptr);
        mpz_set_str(p1, p1s, 10);
        mpz_set_str(p2, p2s, 10);
        mpz_mul(comp, p1, p2);

        for (int mode = 0; mode < 2; ++mode) {
            const char *tag = (mode == 0) ? "prime modulus" : "COMPOSITE modulus p1*p2";
            auto run_mode = [&](const mpz_t mod) {
                vector<Zp> roots(1 + (rnd() % 40)), points(1 + (rnd() % 60));
                for (Zp &z : roots) set_u64(z.v, rnd() % 1000000007ull);
                for (Zp &z : points) set_u64(z.v, rnd() % 1000000007ull);
                bool roots_ok = false;
                const unsigned long long mism = tree_vs_horner(mod, roots, points, roots_ok);
                check(("F(x_j) == 0 at every root (" + string(tag) + ")").c_str(), roots_ok);
                check(("remainder tree == Horner at every giant point (" + string(tag) +
                       ")").c_str(),
                      mism == 0, "mismatches=" + std::to_string(mism));
            };
            if (mode == 0) run_mode(p1); else run_mode(comp);
        }
        mpz_clears(p1, p2, comp, nullptr);
    }

    /* (e) the frozen end-to-end case, including the naive check of the very run
       that produces the factor. */
    std::printf("\n[2] end to end on 2^128+1 (known factors), B1 small so only stage 2 "
                "can find it\n");
    {
        /* FROZEN configuration, identical to stage2_ref.cpp's: sigma = 26 with
           B1 = 1000, B2 = 1e6, D = 210 finds the 17-digit factor 59649589127497217
           through the stage-2 prime 114713 (p = 546*210 + 53). */
        g_cost = Cost();
        g_cost.S = 128;
        const E2E e = run_known_factor(1000, 1000000, 210, 26, true);
        check("tree finds a factor of 2^128+1 (sigma=26, B1=1e3, B2=1e6, D=210)",
              e.res.factors.size() > 0,
              "factors=" + std::to_string(e.res.factors.size()));
        check("the 17-digit factor 59649589127497217 is among them",
              e.res.factors.size() == 1 && e.res.factors[0] == "59649589127497217",
              e.res.factors.empty() ? "none" : e.res.factors[0]);
        check("no bogus factor (every factor divides N)", e.res.bad_factors == 0,
              "bad=" + std::to_string(e.res.bad_factors));
        check("the naive check of that run has no mismatch",
              e.st.naive_points > 0 && e.st.naive_mismatches == 0,
              "points=" + std::to_string(e.st.naive_points) +
                  " mismatches=" + std::to_string(e.st.naive_mismatches));
        check("the hit is attributed to stage-2 prime 114713",
              e.res.hit_primes.size() > 0 && e.res.hit_primes[0] == 114713);
        std::printf("  (shape: baby_j=%llu giant_i=%llu F_degree=%llu, operand_bits=%llu)\n",
                    (unsigned long long)e.st.baby_j, (unsigned long long)e.st.giant_i,
                    (unsigned long long)e.st.f_degree, g_cost.total_bits());
    }

    std::printf("\nselftest: %d checks, %d failed\n", g_checks, g_fails);
    return g_fails == 0 ? 0 : 1;
}

} /* namespace */

int main(int argc, char **argv)
{
    /* Unbuffered stdout: this tool can be long-running and is meant to be killable, so
       progress must not sit in a 4 KB stdio buffer (the first --selftest hang produced
       NO output at all, which cost a debugging round). */
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    const char *n_str = nullptr;
    const char *save = nullptr;
    uint64_t B1 = 0, B2 = 0, D = 210, sigma = 2, curves = 1, save_first = 0, torsion = 1;
    uint64_t num_poly_g = 0;                      /* 0 = derive it from the shape */
    bool do_selftest = false, naive_check = false, cost_detail = false, model_only = false;

    for (int i = 1; i < argc; ++i) {
        const char *a = argv[i];
        auto next = [&]() -> const char * { return (i + 1 < argc) ? argv[++i] : ""; };
        if (!std::strcmp(a, "--selftest")) do_selftest = true;
        else if (!std::strcmp(a, "--n")) n_str = next();
        else if (!std::strcmp(a, "--save")) save = next();
        else if (!std::strcmp(a, "--b1")) B1 = std::strtoull(next(), nullptr, 10);
        else if (!std::strcmp(a, "--b2")) B2 = std::strtoull(next(), nullptr, 10);
        else if (!std::strcmp(a, "--d")) D = std::strtoull(next(), nullptr, 10);
        else if (!std::strcmp(a, "--sigma")) sigma = std::strtoull(next(), nullptr, 10);
        else if (!std::strcmp(a, "--curves")) curves = std::strtoull(next(), nullptr, 10);
        else if (!std::strcmp(a, "--save-first")) save_first = std::strtoull(next(), nullptr, 10);
        else if (!std::strcmp(a, "--torsion")) torsion = std::strtoull(next(), nullptr, 10);
        else if (!std::strcmp(a, "--num-poly-g")) num_poly_g = std::strtoull(next(), nullptr, 10);
        else if (!std::strcmp(a, "--naive-check")) naive_check = true;
        else if (!std::strcmp(a, "--cost")) cost_detail = true;
        else if (!std::strcmp(a, "--model-only")) model_only = true;
        else if (!std::strcmp(a, "--dump-F")) g_dump_F = next();
        else if (!std::strcmp(a, "--verbose-hits")) g_verbose_hits = true;
    }

    if (do_selftest || argc == 1) return selftest();
    if (!n_str) {
        std::printf("stage2_tree_ref: --n <decimal> is required\n");
        return 2;
    }
    if (D == 0 || B2 == 0) {
        std::printf("stage2_tree_ref: --b2 <B2> and --d <D> must be non-zero\n");
        return 2;
    }

    mpz_t N;
    mpz_init(N);
    if (mpz_set_str(N, n_str, 10) != 0) {
        std::printf("stage2_tree_ref: bad decimal for --n\n");
        mpz_clear(N);
        return 2;
    }
    g_cost.S = (long)mpz_sizeinbase(N, 2);

    vector<uint64_t> sigmas;
    vector<string> save_x;
    if (save) {
        const vector<SaveCurve> sc = parse_save(save, save_first, curves);
        for (const SaveCurve &c : sc) {
            sigmas.push_back(c.sigma);
            save_x.push_back(c.have_x ? c.x_hex : string());
            if (B1 == 0) B1 = c.b1;
        }
        if (sc.empty()) {
            std::printf("stage2_tree_ref: no curves parsed from %s\n", save);
            mpz_clear(N);
            return 2;
        }
    } else {
        for (uint64_t c = 0; c < curves; ++c) sigmas.push_back(sigma + c);
    }

    const vector<uint8_t> s_bits = (save == nullptr) ? build_s_bits(B1, torsion)
                                                     : vector<uint8_t>();

    std::printf("stage2_tree: N_bits=%ld D=%llu B1=%llu B2=%llu curves=%llu\n",
                (long)mpz_sizeinbase(N, 2), (unsigned long long)D,
                (unsigned long long)B1, (unsigned long long)B2,
                (unsigned long long)sigmas.size());

    /* --model-only: account the BATCHED structure for this shape and stop.  This is how
       a shape far too large to actually run (M5261's D = 1411410, B2 = 1.9e12) can still
       be cross-checked against the plan's cost model. */
    if (model_only) {
        const unsigned long long P = baby_count(D);
        const unsigned long long giant_points = B2 / D + 2;
        const long S = (long)mpz_sizeinbase(N, 2);
        std::printf("cost_model: shape model-only P=phi(D)/2=%llu giant_points=%llu\n",
                    P, giant_points);
        for (int c = 0; c < 3; ++c)
            print_batched_model(kTreeConvName[c],
                                model_batched(P, giant_points, S, num_poly_g, c), S);
        mpz_clear(N);
        return 0;
    }

    Result all;
    uint64_t used_curves = 0, tree_curves = 0;
    unsigned long long naive_points = 0, naive_mismatches = 0;
    const double t0 = now_s();
    for (size_t idx = 0; idx < sigmas.size(); ++idx) {
        g_cur_sigma = sigmas[idx];
        const string *sx = save_x.empty() ? nullptr : &save_x[idx];
        const CurveRun cr = run_curve(N, sigmas[idx], B1, B2, D, s_bits, sx, naive_check);
        if (cr.processed) {
            /* merge this curve's factors into the run total, re-verifying each one */
            for (const string &f : cr.res.factors) {
                mpz_t fz;
                mpz_init(fz);
                mpz_set_str(fz, f.c_str(), 10);
                record(all, fz, 0, N);
                mpz_clear(fz);
            }
            for (uint64_t p : cr.res.hit_primes) all.hit_primes.push_back(p);
            naive_points += cr.st.naive_points;
            naive_mismatches += cr.st.naive_mismatches;
            ++used_curves;
            if (cr.st.f_degree > 0) ++tree_curves;
            std::printf("stage2_tree: sigma=%llu D=%llu B1=%llu B2=%llu baby_j=%llu "
                        "giant_i=%llu F_degree=%llu factors=%llu\n",
                        (unsigned long long)sigmas[idx], (unsigned long long)D,
                        (unsigned long long)B1, (unsigned long long)B2,
                        (unsigned long long)cr.st.baby_j,
                        (unsigned long long)cr.st.giant_i,
                        (unsigned long long)cr.st.f_degree,
                        (unsigned long long)cr.res.factors.size());
        }
    }
    const double elapsed = now_s() - t0;

    if (naive_check)
        std::printf("naive_check: points=%llu mismatches=%llu\n", naive_points,
                    naive_mismatches);
    print_cost_line();
    if (cost_detail) print_cost_detail();

    /* ---- structure (b): the batched accounting (see the model section above) ------- */
    {
        const long S = (long)mpz_sizeinbase(N, 2);
        const unsigned long long P = baby_count(D);
        const unsigned long long giant_points = B2 / D + 2;
        const BatchedModel mb = model_batched(P, giant_points, S, num_poly_g, TREE_BALANCED);
        std::printf("cost_model: structure=batched (one F tree + num_polyG x [G tree over "
                    "poly_size points + 3 full-size mults] + one descent), P=%llu "
                    "giant_points=%llu num_poly_g=%llu loops=%llu; units are OUR Kronecker "
                    "operand-bits, NOT Prime95's internal cost\n",
                    mb.P, mb.giant_points, mb.num_poly_g, mb.loops);
        for (int c = 0; c < 3; ++c)
            print_batched_model(kTreeConvName[c],
                                model_batched(P, giant_points, S, num_poly_g, c), S);
        /* The model's product-tree recursion reproduces the MEASURED product-tree cost of
           a real run exactly (both trees are plain builds over known leaf counts), which
           is what makes the batched numbers above trustworthy rather than a guess. */
        if (tree_curves > 0) {
            const unsigned long long mf =
                model_tree((unsigned long long)P, S, false) * tree_curves;
            const unsigned long long mg =
                model_tree(giant_points, S, false) * tree_curves;
            const bool ok = (g_cost.bits[CAT_FTREE] == mf) && (g_cost.bits[CAT_GTREE] == mg);
            std::printf("cost_model_check: %s recursion=our-padded(power-of-two, exactly "
                        "what build_product_tree builds) f_tree measured=%llu model=%llu "
                        "giant_tree measured=%llu model=%llu over=%llu curves\n",
                        ok ? "MATCH" : "MISMATCH", g_cost.bits[CAT_FTREE], mf,
                        g_cost.bits[CAT_GTREE], mg, (unsigned long long)tree_curves);
        }
    }

    print_result("tree", all, used_curves, elapsed, g_cost.total_bits());

    mpz_clear(N);
    /* A --naive-check run that disagrees with Horner is a FAILURE of the tool, so it
       must not look like a successful run to a script. */
    if (naive_mismatches > 0) return 1;
    return 0;
}
