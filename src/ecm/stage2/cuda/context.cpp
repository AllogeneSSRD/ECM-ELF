// Modulus metadata, GMP word conversion and integer helpers shared by planning and execution.
#include "context.h"

bool mpz_to_words(std::vector<unsigned long long> &w,size_t W,const mpz_t v)
{
    if (w.size() != W) w.assign(W, 0ull);
    else std::fill(w.begin(), w.end(), 0ull);
    if ((size_t)mpz_sizeinbase(v, 2) > W * 64) return false;
    size_t cnt = 0;
    mpz_export(w.data(), &cnt, -1, 8, 0, 0, v);
    return cnt <= W;
}

void words_to_mpz(mpz_t out,const unsigned long long *w,size_t W)
{
    mpz_import(out, W, -1, 8, 0, 0, w);
}

int ceil_log2_u64(unsigned long long m)
{
    int l = 0;
    while (l < 63 && (1ull << l) < m) ++l;
    return l;
}

void set_u64_mpz(mpz_t r,unsigned long long v)
{
    mpz_set_ui(r, (unsigned long)(v >> 32));
    mpz_mul_2exp(r, r, 32);
    mpz_add_ui(r, r, (unsigned long)(v & 0xFFFFFFFFull));
}

unsigned long long mulmod_u64(unsigned long long a,unsigned long long b,unsigned long long m)
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

unsigned long long powmod_u64(unsigned long long a,unsigned long long e,unsigned long long m)
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

bool is_prime_u64(unsigned long long p)
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
