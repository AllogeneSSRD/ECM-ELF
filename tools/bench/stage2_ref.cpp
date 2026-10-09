/* ---------------------------------------------------------------------------
 * stage2_ref.cpp -- ECM stage 2, CORRECTNESS-FIRST reference implementation.
 *
 * Purpose (docs/architecture/STAGE2.md, milestone M1): prove the algorithm and the
 * save-file conventions before any GPU work.  Three independent checks:
 *
 *   (a) x-only curve arithmetic  vs  naive AFFINE arithmetic over a prime field
 *       (--selftest, section 1 of the output);
 *   (b) two stage-2 algorithms that must agree:
 *         brute   -- for every prime p in (B1,B2]: [p]Q by the x-only ladder, gcd(Z,N).
 *                    Obviously correct, O(pi(B2)) ladders: only for small B2, which is
 *                    exactly what an oracle needs.
 *         pairing -- classic BSGS standard continuation: baby table x([j]Q), j <= D/2;
 *                    giant steps x([iD]Q); for every prime p > D/2 write r = p mod D,
 *                    j = min(r, D-r), i = (p -+ j)/D and accumulate (X_i Z_j - X_j Z_i).
 *                    If f | N with [p]Q = O (mod f) then [iD]Q = -+[j]Q (mod f), so
 *                    x_i == x_j (mod f) and f divides the product.  Block gcds narrow
 *                    the culprit, per-prime gcds inside the block name it.
 *       The pairing candidate set {iD +- j} cap (B1,B2] is a superset of that range's
 *       primes (composites only add coverage, never a wrong factor), so brute's hits
 *       must be a SUBSET of pairing's -- that is the agreement assertion.
 *   (c) end to end on a number with KNOWN factors: 2^128+1 = 59649589127497217 *
 *       5704689200685129054721, small B1 so that only stage 2 can find it.
 *
 * Conventions are pinned to our stage 1 (src/cpu/ecm_mont_cpu.h):
 *   u = sigma^2-5, v = 4*sigma, A = (v-u)^3(3u+v)/(4u^3v) - 2, a24 = (A+2)/4,
 *   start point (X:Z) = (u^3:v^3), stage 1 = [s]P with s = torsion*lcm(1..B1),
 *   and a save file stores the NORMALISED x = Qx/Qz.
 *
 * Usage:
 *   stage2_ref.exe --selftest
 *   stage2_ref.exe --n <decimal> --b1 <B1> --b2 <B2> --d <D> [--sigma <u64>]
 *                  [--curves <k>] [--algorithm brute|pairing|both] [--torsion <1|12>]
 *   stage2_ref.exe --n <decimal> --save <file> [--save-curves <k>] [--save-first <i>]
 *                  --b2 <B2> --d <D> [--algorithm ...]
 *   stage2_ref.exe --n <decimal> --sigma <u64> --b1 <B1> [--torsion <1|12>] --print-stage1-x
 *
 * Machine-readable summary line per algorithm run (parsed by
 * tools/test/test_stage2_ref.ps1):
 *   stage2: algorithm=pairing curves=16 hits=2 factors=59649589127497217
 *           hit_primes=... bad_factors=0 elapsed=1.23
 * ------------------------------------------------------------------------- */
#include <gmp.h>

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

/* ---- x-only Montgomery arithmetic mod N ---------------------------------- */

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
   so writing r.X early destroyed X_P and silently corrupted Z3 (measured: k >= 3 wrong,
   k = 2 right -- the affine oracle caught it). */
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

/* ---- curve setup: Suyama sigma, identical conventions to our stage 1 ------ */

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

/* ---- results ------------------------------------------------------------- */

struct Result {
    uint64_t hits = 0;
    uint64_t bad_factors = 0;                   /* a "factor" that does not divide N */
    vector<string> factors;
    vector<uint64_t> hit_primes;
};

uint64_t g_cur_sigma = 0;      /* sigma of the curve being processed (for hit lines) */
bool g_verbose_hits = false;

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

/* ---- stage 2: brute-force oracle ---------------------------------------- */

Result stage2_brute(const mpz_t N, const vector<uint64_t> &primes, uint64_t B1, uint64_t B2,
                    const Pt &Q, const mpz_t a24)
{
    Result res;
    mpz_t g;
    mpz_init(g);
    for (uint64_t p : primes) {
        if (p <= B1 || p > B2) continue;
        Pt R;
        xmul_u64(R, p, Q, a24, N);
        mpz_gcd(g, R.Z, N);
        if (mpz_cmp_ui(g, 1) > 0 && mpz_cmp(g, N) < 0) record(res, g, p, N);
    }
    mpz_clear(g);
    return res;
}

/* ---- stage 2: classic BSGS / pairing ------------------------------------ */

Result stage2_pairing(const mpz_t N, const vector<uint64_t> &primes, uint64_t B1, uint64_t B2,
                      uint64_t D, const Pt &Q, const mpz_t a24)
{
    Result res;
    const uint64_t half = (D < 2) ? 1 : D / 2;
    const uint64_t imax = B2 / D + 2;

    vector<Pt *> baby(half + 1, nullptr);
    for (uint64_t j = 1; j <= half; ++j) {
        baby[j] = new Pt();
        xmul_u64(*baby[j], j, Q, a24, N);
    }
    vector<Pt *> giant(imax + 1, nullptr);
    for (uint64_t i = 1; i <= imax; ++i) {
        giant[i] = new Pt();
        if (i == 1) {
            xmul_u64(*giant[1], D, Q, a24, N);
        } else if (i == 2) {
            xdbl(*giant[2], *giant[1], a24, N);
        } else {
            xadd(*giant[i], *giant[i - 1], *giant[1], *giant[i - 2], N);
        }
    }

    const size_t BLOCK = 4096;
    mpz_t prod, t1, t2, g;
    mpz_inits(prod, t1, t2, g, nullptr);
    mpz_set_ui(prod, 1);
    vector<uint64_t> block;
    auto flush = [&]() {
        if (block.empty()) return;
        mpz_gcd(g, prod, N);
        if (mpz_cmp_ui(g, 1) > 0 && mpz_cmp(g, N) < 0) {
            for (uint64_t p : block) {                 /* name the culprit(s) */
                Pt R;
                xmul_u64(R, p, Q, a24, N);
                mpz_t pg;
                mpz_init(pg);
                mpz_gcd(pg, R.Z, N);
                if (mpz_cmp_ui(pg, 1) > 0 && mpz_cmp(pg, N) < 0) record(res, pg, p, N);
                mpz_clear(pg);
            }
        }
        block.clear();
        mpz_set_ui(prod, 1);
    };

    for (uint64_t p : primes) {
        if (p <= B1 || p > B2) continue;
        if (p <= half) {                               /* too small to pair */
            Pt R;
            xmul_u64(R, p, Q, a24, N);
            mpz_gcd(g, R.Z, N);
            if (mpz_cmp_ui(g, 1) > 0 && mpz_cmp(g, N) < 0) record(res, g, p, N);
            continue;
        }
        const uint64_t r = p % D;
        if (r == 0) continue;                          /* p | D: covered by stage 1 */
        const uint64_t j = (r <= half) ? r : D - r;
        const uint64_t i = (r <= half) ? (p - j) / D : (p + j) / D;
        if (i == 0 || i > imax) continue;
        mpz_mul(t1, giant[i]->X, baby[j]->Z);           /* X_i Z_j */
        mpz_mul(t2, baby[j]->X, giant[i]->Z);           /* X_j Z_i */
        mpz_sub(t1, t1, t2);
        mpz_mul(prod, prod, t1);
        mpz_mod(prod, prod, N);
        block.push_back(p);
        if (block.size() >= BLOCK) flush();
    }
    flush();

    mpz_clears(prod, t1, t2, g, nullptr);
    for (uint64_t j = 1; j <= half; ++j) delete baby[j];
    for (uint64_t i = 1; i <= imax; ++i) delete giant[i];
    return res;
}

/* ---- affine oracle over a prime field (used by --selftest) --------------- */

struct Affine {
    mpz_t A, p;
    Affine() { mpz_inits(A, p, nullptr); }
    ~Affine() { mpz_clears(A, p, nullptr); }
    /* doubling: lam = (3x^2 + 2Ax + 1) / (2y);  x3 = lam^2 - A - 2x */
    void dbl(mpz_t xr, const mpz_t x, const mpz_t y) const
    {
        mpz_t num, den, inv, lam, t;
        mpz_inits(num, den, inv, lam, t, nullptr);
        mpz_mul(num, x, x);
        mpz_mul_ui(num, num, 3);
        mpz_mul(t, A, x);
        mpz_mul_ui(t, t, 2);
        mpz_add(num, num, t);
        mpz_add_ui(num, num, 1);
        mpz_mul_ui(den, y, 2);
        mpz_invert(inv, den, p);
        mpz_mul(lam, num, inv);
        mpz_mod(lam, lam, p);
        mpz_mul(xr, lam, lam);
        mpz_sub(xr, xr, A);
        mpz_sub(xr, xr, x);
        mpz_sub(xr, xr, x);
        mpz_mod(xr, xr, p);
        mpz_clears(num, den, inv, lam, t, nullptr);
    }
    /* addition: lam = (y2-y1)/(x2-x1);  x3 = lam^2 - A - x1 - x2 */
    void add(mpz_t xr, const mpz_t x1, const mpz_t y1, const mpz_t x2, const mpz_t y2) const
    {
        mpz_t num, den, inv, lam;
        mpz_inits(num, den, inv, lam, nullptr);
        mpz_sub(num, y2, y1);
        mpz_sub(den, x2, x1);
        mpz_invert(inv, den, p);
        mpz_mul(lam, num, inv);
        mpz_mod(lam, lam, p);
        mpz_mul(xr, lam, lam);
        mpz_sub(xr, xr, A);
        mpz_sub(xr, xr, x1);
        mpz_sub(xr, xr, x2);
        mpz_mod(xr, xr, p);
        mpz_clears(num, den, inv, lam, nullptr);
    }
};

/* [k]P affinely, tracking x and y (small k only).
   The first step MUST be a doubling: adding P to itself hits the x1 == x2 case and
   the inversion below fails, which silently produced garbage for every k >= 2 in the
   first version of this oracle (measured: "xADD matches affine" FAIL). */
void affine_dbl_xy(mpz_t xr, mpz_t yr, const mpz_t x, const mpz_t y, const Affine &aff)
{
    mpz_t num, den, inv, lam, t;
    mpz_inits(num, den, inv, lam, t, nullptr);
    mpz_mul(num, x, x);
    mpz_mul_ui(num, num, 3);
    mpz_mul(t, aff.A, x);
    mpz_mul_ui(t, t, 2);
    mpz_add(num, num, t);
    mpz_add_ui(num, num, 1);
    mpz_mul_ui(den, y, 2);
    mpz_invert(inv, den, aff.p);
    mpz_mul(lam, num, inv);
    mpz_mod(lam, lam, aff.p);
    mpz_mul(xr, lam, lam);
    mpz_sub(xr, xr, aff.A);
    mpz_sub(xr, xr, x);
    mpz_sub(xr, xr, x);
    mpz_mod(xr, xr, aff.p);
    mpz_sub(yr, x, xr);
    mpz_mul(yr, yr, lam);
    mpz_sub(yr, yr, y);
    mpz_mod(yr, yr, aff.p);
    mpz_clears(num, den, inv, lam, t, nullptr);
}

void affine_mul(mpz_t xr, mpz_t yr, uint64_t k, const mpz_t x, const mpz_t y, const Affine &aff)
{
    if (k == 1) {
        mpz_set(xr, x);
        mpz_set(yr, y);
        return;
    }
    mpz_t sx, sy, nx, ny, num, den, inv, lam;
    mpz_inits(sx, sy, nx, ny, num, den, inv, lam, nullptr);
    affine_dbl_xy(sx, sy, x, y, aff);                 /* [2]P */
    for (uint64_t i = 3; i <= k; ++i) {               /* += P */
        mpz_sub(num, sy, y);
        mpz_sub(den, sx, x);
        mpz_invert(inv, den, aff.p);
        mpz_mul(lam, num, inv);
        mpz_mod(lam, lam, aff.p);
        mpz_mul(nx, lam, lam);
        mpz_sub(nx, nx, aff.A);
        mpz_sub(nx, nx, sx);
        mpz_sub(nx, nx, x);
        mpz_mod(nx, nx, aff.p);
        mpz_sub(ny, sx, nx);
        mpz_mul(ny, ny, lam);
        mpz_sub(ny, ny, sy);
        mpz_mod(ny, ny, aff.p);
        mpz_set(sx, nx);
        mpz_set(sy, ny);
    }
    mpz_set(xr, sx);
    mpz_set(yr, sy);
    mpz_clears(sx, sy, nx, ny, num, den, inv, lam, nullptr);
}

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

/* ---- known-factor end to end (2^128+1) ---------------------------------- */

struct E2E {
    Result brute, pairing;
    uint64_t curves = 0;
    bool ok = false;
};

E2E run_known_factor(uint64_t B1, uint64_t B2, uint64_t D, uint64_t first_sigma, uint64_t ncurves,
                     uint64_t torsion)
{
    E2E e2e;
    mpz_t N;
    mpz_init(N);
    mpz_set_str(N, "340282366920938463463374607431768211457", 10);   /* 2^128+1 */
    const vector<uint64_t> primes = primes_up_to(B2);
    const vector<uint8_t> s_bits = build_s_bits(B1, torsion);
    for (uint64_t c = 0; c < ncurves; ++c) {
        const uint64_t sigma = first_sigma + c;
        mpz_t a24, factor;
        mpz_inits(a24, factor, nullptr);
        Pt P;
        if (suyama_curve(a24, P, sigma, N, factor) == 1) {
            record(e2e.pairing, factor, 0, N);
            mpz_clears(a24, factor, nullptr);
            continue;
        }
        Pt Q;
        ladder(Q, s_bits, P, a24, N);                 /* [s]P = the save's point */
        if (mpz_cmp_ui(Q.Z, 0) == 0) {                /* stage 1 already done */
            mpz_clears(a24, factor, nullptr);
            continue;
        }
        /* make the point affine so both algorithms start from the same x */
        mpz_t xq;
        mpz_init(xq);
        affine_x(xq, Q, N);
        Pt Qa;
        mpz_set(Qa.X, xq);
        mpz_set_ui(Qa.Z, 1);
        Result rb = stage2_brute(N, primes, B1, B2, Qa, a24);
        Result rp = stage2_pairing(N, primes, B1, B2, D, Qa, a24);
        for (const string &f : rb.factors) {
            bool in_pairing = false;
            for (const string &g : rp.factors)
                if (g == f) in_pairing = true;
            if (!in_pairing) ++e2e.pairing.bad_factors;   /* brute found, pairing missed */
        }
        for (const string &f : rb.factors) {
            char *dummy = nullptr;
            (void)dummy;
            mpz_t fz;
            mpz_init(fz);
            mpz_set_str(fz, f.c_str(), 10);
            record(e2e.brute, fz, 0, N);
            mpz_clear(fz);
        }
        for (const string &f : rp.factors) {
            mpz_t fz;
            mpz_init(fz);
            mpz_set_str(fz, f.c_str(), 10);
            record(e2e.pairing, fz, 0, N);
            mpz_clear(fz);
        }
        ++e2e.curves;
        mpz_clear(xq);
        mpz_clears(a24, factor, nullptr);
    }
    mpz_clear(N);
    e2e.ok = true;
    return e2e;
}

int selftest()
{
    std::printf("stage2_ref --selftest\n\n[1] x-only arithmetic vs affine arithmetic over a prime field\n");
    {
        /* M61 = 2^61-1 is prime and == 3 (mod 4), so a modular square root is
           rhs^((p+1)/4) -- no Tonelli-Shanks needed.  (The first attempt used the
           largest 64-bit prime, which is == 1 (mod 4) AND searched for y by brute
           force: the smallest root of a random residue is uniform in [0,p), so that
           search could never succeed.  Measured: "affine oracle found a point" FAIL.) */
        const char *pp = "2305843009213693951";
        Affine aff;
        mpz_set_str(aff.p, pp, 10);
        mpz_set_ui(aff.A, 6);
        mpz_t a24, inv4;
        mpz_inits(a24, inv4, nullptr);
        mpz_add_ui(a24, aff.A, 2);
        mpz_set_ui(inv4, 4);
        mpz_invert(inv4, inv4, aff.p);
        mpz_mul(a24, a24, inv4);
        mpz_mod(a24, a24, aff.p);                     /* a24 = (A+2)/4 */

        /* find a point on y^2 = x^3 + A x^2 + x over F_p */
        mpz_t x, y, rhs, t;
        mpz_inits(x, y, rhs, t, nullptr);
        mpz_set_ui(x, 2);
        mpz_powm_ui(rhs, x, 3, aff.p);
        mpz_mul(t, aff.A, x);
        mpz_mul(t, t, x);
        mpz_add(rhs, rhs, t);
        mpz_add(rhs, rhs, x);
        mpz_mod(rhs, rhs, aff.p);
        bool found = false;
        mpz_t exp, xc;
        mpz_inits(exp, xc, nullptr);
        mpz_add_ui(exp, aff.p, 1);
        mpz_fdiv_q_2exp(exp, exp, 2);                 /* (p+1)/4 */
        for (uint64_t cand = 2; cand < 200 && !found; ++cand) {
            mpz_set_ui(xc, (unsigned long)cand);
            mpz_powm_ui(rhs, xc, 3, aff.p);
            mpz_mul(t, aff.A, xc);
            mpz_mul(t, t, xc);
            mpz_add(rhs, rhs, t);
            mpz_add(rhs, rhs, xc);
            mpz_mod(rhs, rhs, aff.p);
            if (mpz_legendre(rhs, aff.p) == 1) {
                mpz_powm(y, rhs, exp, aff.p);         /* square root mod p */
                if (mpz_cmp_ui(y, 0) != 0) {
                    mpz_set(x, xc);
                    found = true;
                }
            }
        }
        mpz_clears(exp, xc, nullptr);
        check("affine oracle found a point on the test curve", found);
        if (found) {
            Pt P1;
            mpz_set(P1.X, x);
            mpz_set_ui(P1.Z, 1);

            /* xDBL vs affine doubling */
            Pt d;
            xdbl(d, P1, a24, aff.p);
            mpz_t xa, xo;
            mpz_inits(xa, xo, nullptr);
            aff.dbl(xa, x, y);
            affine_x(xo, d, aff.p);
            check("xDBL matches affine doubling", mpz_cmp(xo, xa) == 0);

            /* xADD vs affine addition: (2P) + P, difference P */
            Pt two, three;
            xdbl(two, P1, a24, aff.p);
            xadd(three, two, P1, P1, aff.p);
            mpz_t x2a, y2a, x3a, y3a;
            mpz_inits(x2a, y2a, x3a, y3a, nullptr);
            affine_mul(x2a, y2a, 2, x, y, aff);
            affine_mul(x3a, y3a, 3, x, y, aff);
            affine_x(xo, three, aff.p);
            check("xADD matches affine (2P)+P", mpz_cmp(xo, x3a) == 0);

            /* ladder [k]P vs affine [k]P */
            bool all_ok = true;
            string detail;
            for (uint64_t k : {2ull, 3ull, 5ull, 17ull, 1000ull, 12345ull}) {
                Pt R;
                xmul_u64(R, k, P1, a24, aff.p);
                mpz_t xk, yk;
                mpz_inits(xk, yk, nullptr);
                affine_mul(xk, yk, k, x, y, aff);
                affine_x(xo, R, aff.p);
                const bool ok = (mpz_cmp(xo, xk) == 0);
                if (!ok) {
                    all_ok = false;
                    char *a1 = mpz_get_str(nullptr, 16, xo);
                    char *a2 = mpz_get_str(nullptr, 16, xk);
                    detail += "k=" + std::to_string(k) + " ladder=" + a1 + " affine=" + a2 + " ";
                    void (*ff)(void *, size_t) = nullptr;
                    mp_get_memory_functions(nullptr, nullptr, &ff);
                    ff(a1, std::strlen(a1) + 1);
                    ff(a2, std::strlen(a2) + 1);
                }
                mpz_clears(xk, yk, nullptr);
            }
            check("ladder [k]P matches affine for k = 2,3,5,17,1000,12345", all_ok, detail);
            mpz_clears(x2a, y2a, x3a, y3a, nullptr);
            mpz_clears(xa, xo, nullptr);

            /* a wrong a24 must give a different answer (guards against a no-op test) */
            mpz_t bad;
            mpz_init(bad);
            mpz_add_ui(bad, a24, 1);
            Pt dbad;
            xdbl(dbad, P1, bad, aff.p);
            mpz_t xbad;
            mpz_init(xbad);
            affine_x(xbad, dbad, aff.p);
            check("a wrong a24 changes the result (test is not vacuous)",
                  mpz_cmp(xbad, xa) != 0);
            mpz_clear(bad);
            mpz_clear(xbad);
        }
        mpz_clears(x, y, rhs, t, nullptr);
        mpz_clears(a24, inv4, nullptr);
    }

    std::printf("\n[2] end to end on 2^128+1 (known factors), B1 small so only stage 2 can find it\n");
    {
        /* FROZEN configuration (measured 2026-09-30): sigma = 26 with B1 = 1000,
           B2 = 1e6, D = 210 finds the 17-digit factor 59649589127497217 through the
           stage-2 prime 114713.  Everything here is deterministic, so the check is
           reproducible; the sigma sweep that found it covered 2..201 (199 curves
           processed, exactly this one hit). */
        const E2E e = run_known_factor(1000, 1000000, 210, 26, 1, 1);
        check("pairing finds a factor of 2^128+1 (sigma=26, B1=1e3, B2=1e6)",
              e.pairing.factors.size() > 0,
              "factors=" + std::to_string(e.pairing.factors.size()));
        check("no bogus factor (every factor divides N)", e.pairing.bad_factors == 0);
        bool known = e.pairing.factors.size() > 0;
        for (const string &f : e.pairing.factors) {
            if (f != "59649589127497217" && f != "5704689200685129054721") known = false;
        }
        check("every factor is one of the two known factors of 2^128+1", known);
        check("brute (independent algorithm) finds the same factor(s)",
              e.brute.factors.size() > 0 && e.brute.factors.size() <= e.pairing.factors.size(),
              "brute=" + std::to_string(e.brute.factors.size()) +
                  " pairing=" + std::to_string(e.pairing.factors.size()));
    }

    std::printf("\nselftest: %d checks, %d failed\n", g_checks, g_fails);
    return g_fails == 0 ? 0 : 1;
}

/* ---- save file parsing --------------------------------------------------- */

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
               mpz_set_str(..., 16) fail (it does not accept a 0x prefix) and the point
               silently became 0 -- which cost a debugging round (measured: save-driven
               stage 2 found nothing while the identical synthetic run found the factor). */
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

void print_result(const char *algo, const Result &r, uint64_t curves, double elapsed)
{
    std::printf("stage2: algorithm=%s curves=%llu hits=%llu bad_factors=%llu factors=",
                algo, (unsigned long long)curves, (unsigned long long)r.hits,
                (unsigned long long)r.bad_factors);
    for (size_t i = 0; i < r.factors.size(); ++i)
        std::printf("%s%s", i ? "," : "", r.factors[i].c_str());
    std::printf(" hit_primes=");
    for (size_t i = 0; i < r.hit_primes.size() && i < 32; ++i)
        std::printf("%s%llu", i ? "," : "", (unsigned long long)r.hit_primes[i]);
    std::printf(" elapsed=%.2f\n", elapsed);
}

} /* namespace */

int main(int argc, char **argv)
{
    const char *n_str = nullptr;
    const char *save = nullptr;
    uint64_t B1 = 0, B2 = 0, D = 210, sigma = 2, curves = 1, save_first = 0, torsion = 1;
    string algorithm = "both";
    bool print_stage1_x = false, do_selftest = false;

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
        else if (!std::strcmp(a, "--algorithm")) algorithm = next();
        else if (!std::strcmp(a, "--print-stage1-x")) print_stage1_x = true;
        else if (!std::strcmp(a, "--verbose-hits")) g_verbose_hits = true;
    }

    if (do_selftest || argc == 1) return selftest();
    if (!n_str) {
        std::printf("stage2_ref: --n <decimal> is required\n");
        return 2;
    }

    mpz_t N;
    mpz_init(N);
    if (mpz_set_str(N, n_str, 10) != 0) {
        std::printf("stage2_ref: bad decimal for --n\n");
        mpz_clear(N);
        return 2;
    }

    /* --print-stage1-x: recompute x([s]P) and print it (convention check vs a save) */
    if (print_stage1_x) {
        vector<uint8_t> bits = build_s_bits(B1, torsion);
        mpz_t a24, factor;
        mpz_inits(a24, factor, nullptr);
        Pt P;
        const int st = suyama_curve(a24, P, sigma, N, factor);
        if (st == 1) {
            char *s = mpz_get_str(nullptr, 10, factor);
            std::printf("stage1_x: sigma=%llu B1=%llu degenerate factor=%s\n",
                        (unsigned long long)sigma, (unsigned long long)B1, s);
            void (*ff)(void *, size_t) = nullptr;
            mp_get_memory_functions(nullptr, nullptr, &ff);
            ff(s, std::strlen(s) + 1);
        } else {
            Pt Q;
            ladder(Q, bits, P, a24, N);
            mpz_t x;
            mpz_init(x);
            affine_x(x, Q, N);
            char *s = mpz_get_str(nullptr, 16, x);
            std::printf("stage1_x: sigma=%llu B1=%llu torsion=%llu s_bits=%llu x=0x%s z_zero=%d\n",
                        (unsigned long long)sigma, (unsigned long long)B1,
                        (unsigned long long)torsion, (unsigned long long)bits.size(), s,
                        mpz_cmp_ui(Q.Z, 0) == 0 ? 1 : 0);
            mpz_clear(x);
        }
        mpz_clears(a24, factor, nullptr);
        mpz_clear(N);
        return 0;
    }

    const vector<uint64_t> primes = primes_up_to(B2);
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
            std::printf("stage2_ref: no curves parsed from %s\n", save);
            mpz_clear(N);
            return 2;
        }
    } else {
        for (uint64_t c = 0; c < curves; ++c) sigmas.push_back(sigma + c);
    }

    const vector<uint8_t> s_bits = (save == nullptr) ? build_s_bits(B1, torsion) : vector<uint8_t>();

    Result brute_all, pairing_all;
    uint64_t used_curves = 0;
    const double t0 = now_s();
    for (size_t idx = 0; idx < sigmas.size(); ++idx) {
        mpz_t a24, factor;
        mpz_inits(a24, factor, nullptr);
        g_cur_sigma = sigmas[idx];
        Pt P;
        if (suyama_curve(a24, P, sigmas[idx], N, factor) == 1) {
            if (algorithm != "brute") record(pairing_all, factor, 0, N);
            if (algorithm != "pairing") record(brute_all, factor, 0, N);
            mpz_clears(a24, factor, nullptr);
            ++used_curves;
            continue;
        }
        Pt Qa;
        if (!save_x.empty() && !save_x[idx].empty()) {
            if (mpz_set_str(Qa.X, save_x[idx].c_str(), 16) != 0) {
                std::printf("stage2_ref: unparsable X in the save: %s\n", save_x[idx].c_str());
                mpz_clears(a24, factor, nullptr);
                continue;
            }
            mpz_set_ui(Qa.Z, 1);
            if (g_verbose_hits) {
                char *dbg = mpz_get_str(nullptr, 16, Qa.X);
                std::printf("stage2_point: sigma=%llu source=save x=0x%s\n",
                            (unsigned long long)sigmas[idx], dbg);
                void (*ff)(void *, size_t) = nullptr;
                mp_get_memory_functions(nullptr, nullptr, &ff);
                ff(dbg, std::strlen(dbg) + 1);
            }
        } else {
            Pt Q;
            ladder(Q, s_bits, P, a24, N);
            if (mpz_cmp_ui(Q.Z, 0) == 0) {           /* stage 1 already found it */
                mpz_clears(a24, factor, nullptr);
                continue;
            }
            mpz_t x;
            mpz_init(x);
            affine_x(x, Q, N);
            mpz_set(Qa.X, x);
            mpz_set_ui(Qa.Z, 1);
            if (g_verbose_hits) {
                char *dbg = mpz_get_str(nullptr, 16, Qa.X);
                std::printf("stage2_point: sigma=%llu source=ladder x=0x%s\n",
                            (unsigned long long)sigmas[idx], dbg);
                void (*ff)(void *, size_t) = nullptr;
                mp_get_memory_functions(nullptr, nullptr, &ff);
                ff(dbg, std::strlen(dbg) + 1);
            }
            mpz_clear(x);
        }
        if (algorithm != "pairing") {
            const Result r = stage2_brute(N, primes, B1, B2, Qa, a24);
            for (const string &f : r.factors) {
                mpz_t fz;
                mpz_init(fz);
                mpz_set_str(fz, f.c_str(), 10);
                record(brute_all, fz, 0, N);
                mpz_clear(fz);
            }
            brute_all.hit_primes.insert(brute_all.hit_primes.end(), r.hit_primes.begin(),
                                        r.hit_primes.end());
        }
        if (algorithm != "brute") {
            const Result r = stage2_pairing(N, primes, B1, B2, D, Qa, a24);
            for (const string &f : r.factors) {
                mpz_t fz;
                mpz_init(fz);
                mpz_set_str(fz, f.c_str(), 10);
                record(pairing_all, fz, 0, N);
                mpz_clear(fz);
            }
            pairing_all.hit_primes.insert(pairing_all.hit_primes.end(), r.hit_primes.begin(),
                                          r.hit_primes.end());
        }
        ++used_curves;
        mpz_clears(a24, factor, nullptr);
    }
    const double elapsed = now_s() - t0;

    if (algorithm != "pairing") print_result("brute", brute_all, used_curves, elapsed);
    if (algorithm != "brute") print_result("pairing", pairing_all, used_curves, elapsed);
    mpz_clear(N);
    return 0;
}
