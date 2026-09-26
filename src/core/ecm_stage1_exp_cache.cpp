/* ecm_stage1_exp_cache.cpp -- see the header for the contract and the validation rules. */
#include "ecm_stage1_exp_cache.h"
#include "ecm_stage1_exp.h"          /* ecm_build_lcm_exponent(), the thing being cached */

#include <stddef.h>
#include <stdio.h>
#include <string.h>

#include <chrono>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

using Clock = std::chrono::steady_clock;
static double secs_since(const Clock::time_point &t0)
{
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

const uint64_t ECM_EXP_CACHE_MIN_B1 = 1000000ull;   /* below this a build is ~10 ms */

namespace {
std::string g_cache_dir;
}

void ecm_exp_cache_set_dir(const std::string &dir)
{
    if (dir == "off" || dir == "none" || dir == "0" || dir == "-") g_cache_dir.clear();
    else g_cache_dir = dir;
}

const std::string &ecm_exp_cache_get_dir(void) { return g_cache_dir; }

namespace {

const char     EXP_CACHE_MAGIC[8] = { 'E', 'C', 'M', 'E', 'X', 'P', 'C', '1' };
const uint32_t EXP_CACHE_VERSION  = 1u;
const uint32_t EXP_CACHE_SEMANTICS = 1u;            /* 1 = torsion * lcm(1..B1) */

/* Fixed 64-byte header, little-endian. */
#pragma pack(push, 1)
struct ExpCacheHeader {
    char     magic[8];
    uint32_t version;
    uint32_t semantics;
    uint64_t B1;
    uint64_t torsion;
    uint64_t nbits;
    uint64_t nwords;        /* 8-byte words in the payload */
    uint64_t hash;          /* FNV-1a 64 over header (minus this field) + payload */
    uint64_t csum;          /* additive checksum over the payload words */
};
#pragma pack(pop)

static uint64_t fnv1a64(const void *data, size_t len, uint64_t h)
{
    const unsigned char *p = (const unsigned char *)data;
    for (size_t i = 0; i < len; ++i) {
        h ^= (uint64_t)p[i];
        h *= 1099511628211ull;
    }
    return h;
}

static uint64_t payload_hash(const uint64_t *w, size_t n)
{
    return fnv1a64(w, n * sizeof(uint64_t), 1469598103934665603ull);
}

static uint64_t payload_csum(const uint64_t *w, size_t n)
{
    uint64_t s = 0;
    for (size_t i = 0; i < n; ++i) s += w[i];
    return s;
}

/* Header fields that must be hashed (everything before the hash field). */
static const size_t EXP_CACHE_HASHED_HEADER_BYTES = offsetof(ExpCacheHeader, hash);

static unsigned bitlen_u64(uint64_t v)
{
    unsigned n = 0;
    while (v) { ++n; v >>= 1; }
    return n;
}

/* Deterministic Miller-Rabin for 64-bit values (bases below are a proven set). */
static bool is_prime_u64(uint64_t n)
{
    if (n < 2) return false;
    /* NOTE: do not name this small -- <windows.h> (rpcndr.h) defines the macro
       small as char, so uint64_t small[] turns into uint64_t char[]. */
    static const uint64_t mr_bases[] = { 2,3,5,7,11,13,17,19,23,29,31,37 };
    for (size_t i = 0; i < sizeof(mr_bases) / sizeof(mr_bases[0]); ++i) {
        if (n % mr_bases[i] == 0) return n == mr_bases[i];
    }
    uint64_t d = n - 1;
    unsigned s = 0;
    while ((d & 1ull) == 0) { d >>= 1; ++s; }
    for (size_t i = 0; i < sizeof(mr_bases) / sizeof(mr_bases[0]); ++i) {
        uint64_t x = mr_bases[i], r = 1;
        uint64_t e = d;
        while (e) {                                  /* modular exponentiation mod n */
            if (e & 1ull) r = (uint64_t)((unsigned __int64)r * x % n);
            x = (uint64_t)((unsigned __int64)x * x % n);
            e >>= 1;
        }
        if (r == 1 || r == n - 1) continue;
        bool composite = true;
        for (unsigned k = 1; k < s; ++k) {
            r = (uint64_t)((unsigned __int64)r * r % n);
            if (r == n - 1) { composite = false; break; }
        }
        if (composite) return false;
    }
    return true;
}

static uint64_t highest_power(uint64_t p, uint64_t B1)
{
    uint64_t pp = p;
    while (pp <= B1 / p) pp *= p;
    return pp;
}

static void mul_u64_into(mpz_t acc, uint64_t v)
{
    /* mpz_mul_ui takes unsigned long (32-bit on Windows): split large factors. */
    if (v <= 0xFFFFFFFFull) {
        mpz_mul_ui(acc, acc, (unsigned long)v);
    } else {
        mpz_t t;
        mpz_init(t);
        mpz_set_ui(t, (unsigned long)(v >> 32));
        mpz_mul_2exp(t, t, 32);
        mpz_add_ui(t, t, (unsigned long)(v & 0xFFFFFFFFull));
        mpz_mul(acc, acc, t);
        mpz_clear(t);
    }
}

} /* namespace */

bool ecm_exp_semantic_check(const mpz_t s, uint64_t B1, std::string *why)
{
    if (B1 < 3) return true;

    mpz_t m;
    mpz_init_set_ui(m, 1);

    /* (a) the first 32 primes <= B1 (or every prime <= B1 when B1 is small) */
    unsigned got = 0;
    for (uint64_t p = 2; p <= B1 && got < 32; ++p) {
        if (!is_prime_u64(p)) continue;
        mul_u64_into(m, highest_power(p, B1));
        ++got;
    }
    if (!mpz_divisible_p(s, m)) {
        if (why) *why = "not divisible by the first primes <= B1";
        mpz_clear(m);
        return false;
    }

    /* (b) the 32 primes just below B1: this is the part that proves the payload matches THIS
       B1 (a cache built for a smaller B1 cannot be divisible by them). */
    mpz_set_ui(m, 1);
    got = 0;
    for (uint64_t p = (B1 & 1ull) ? B1 - 2 : B1 - 1; p >= 3 && got < 32; p -= 2) {
        if (!is_prime_u64(p)) continue;
        mul_u64_into(m, p);
        ++got;
    }
    if (!mpz_divisible_p(s, m)) {
        if (why) *why = "not divisible by the primes just below B1";
        mpz_clear(m);
        return false;
    }
    mpz_clear(m);
    return true;
}

std::string ecm_exp_cache_path(const std::string &dir, uint64_t B1, uint64_t torsion)
{
    char name[128];
    snprintf(name, sizeof name, "ecm_exp_lcm_%llu_t%llu_v%u.bin",
             (unsigned long long)B1, (unsigned long long)(torsion ? torsion : 1u),
             (unsigned)EXP_CACHE_VERSION);
    if (dir.empty()) return std::string(name);
    if (dir[dir.size() - 1] == '/' || dir[dir.size() - 1] == '\\') return dir + name;
    return dir + "/" + name;
}

bool ecm_exp_cache_load(const std::string &path, mpz_t s, uint64_t B1, uint64_t torsion,
                        std::string *why)
{
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) {
        if (why) *why = "no cache file";
        return false;
    }
    ExpCacheHeader h;
    if (fread(&h, 1, sizeof h, f) != sizeof h) {
        fclose(f);
        if (why) *why = "header truncated";
        return false;
    }
    if (memcmp(h.magic, EXP_CACHE_MAGIC, sizeof h.magic) != 0) {
        fclose(f);
        if (why) *why = "bad magic";
        return false;
    }
    if (h.version != EXP_CACHE_VERSION || h.semantics != EXP_CACHE_SEMANTICS) {
        fclose(f);
        if (why) *why = "format/semantics version mismatch";
        return false;
    }
    if (h.B1 != B1 || h.torsion != (torsion ? torsion : 1u)) {
        fclose(f);
        if (why) *why = "cached parameters differ (B1/torsion)";
        return false;
    }
    if (h.nwords == 0 || h.nbits == 0) {
        fclose(f);
        if (why) *why = "empty payload";
        return false;
    }
    /* lcm(1..B1) has 1.4427*B1 bits; allow a small band, plus room for the torsion factor. */
    const double est = 1.4426950408889634 * (double)B1;
    if ((double)h.nbits < est * 0.99 || (double)h.nbits > est * 1.01 + 96.0) {
        fclose(f);
        if (why) *why = "bit length implausible for this B1";
        return false;
    }
    std::vector<uint64_t> w((size_t)h.nwords);
    const size_t got = fread(&w[0], sizeof(uint64_t), (size_t)h.nwords, f);
    if (got != (size_t)h.nwords) {
        fclose(f);
        if (why) *why = "payload truncated";
        return false;
    }
    fclose(f);

    uint64_t hh = fnv1a64(&h, EXP_CACHE_HASHED_HEADER_BYTES, 1469598103934665603ull);
    hh = fnv1a64(&w[0], (size_t)h.nwords * sizeof(uint64_t), hh);
    if (hh != h.hash) {
        if (why) *why = "payload/header checksum mismatch";
        return false;
    }
    if (payload_csum(&w[0], (size_t)h.nwords) != h.csum) {
        if (why) *why = "payload additive checksum mismatch";
        return false;
    }

    mpz_import(s, (size_t)h.nwords, -1, sizeof(uint64_t), 0, 0, &w[0]);
    if ((uint64_t)mpz_sizeinbase(s, 2) != h.nbits) {
        if (why) *why = "bit length does not match the payload";
        return false;
    }
    if (!ecm_exp_semantic_check(s, B1, why)) return false;
    return true;
}

bool ecm_exp_cache_save(const std::string &path, const mpz_t s, uint64_t B1, uint64_t torsion,
                        std::string *why)
{
    size_t count = 0;
    uint64_t *w = (uint64_t *)mpz_export(nullptr, &count, -1, sizeof(uint64_t), 0, 0, s);
    if (!w || count == 0) {
        if (w) free(w);
        if (why) *why = "cannot export the value";
        return false;
    }
    ExpCacheHeader h;
    memset(&h, 0, sizeof h);
    memcpy(h.magic, EXP_CACHE_MAGIC, sizeof h.magic);
    h.version = EXP_CACHE_VERSION;
    h.semantics = EXP_CACHE_SEMANTICS;
    h.B1 = B1;
    h.torsion = torsion ? torsion : 1u;
    h.nbits = (uint64_t)mpz_sizeinbase(s, 2);
    h.nwords = (uint64_t)count;
    h.hash = fnv1a64(&h, EXP_CACHE_HASHED_HEADER_BYTES, 1469598103934665603ull);
    h.hash = fnv1a64(w, count * sizeof(uint64_t), h.hash);
    h.csum = payload_csum(w, count);

    const std::string tmp = path + ".tmp";
    FILE *f = fopen(tmp.c_str(), "wb");
    if (!f) {
        free(w);
        if (why) *why = "cannot open the cache file for writing";
        return false;
    }
    bool ok = fwrite(&h, 1, sizeof h, f) == sizeof h &&
              fwrite(w, sizeof(uint64_t), count, f) == count;
    fclose(f);
    free(w);
    if (!ok) {
        remove(tmp.c_str());
        if (why) *why = "write failed";
        return false;
    }
    /* Atomic replace: rename() overwrites on POSIX; on Windows use MoveFileEx. */
#ifdef _WIN32
    if (!MoveFileExA(tmp.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        remove(tmp.c_str());
        if (why) *why = "cannot rename the cache file into place";
        return false;
    }
#else
    if (rename(tmp.c_str(), path.c_str()) != 0) {
        remove(tmp.c_str());
        if (why) *why = "cannot rename the cache file into place";
        return false;
    }
#endif
    return true;
}

bool ecm_build_lcm_exponent_cached(mpz_t s, uint64_t B1, uint64_t torsion,
                                   const std::string &cache_dir, std::string *detail)
{
    char buf[256];
    const std::string path = (cache_dir.empty() || B1 < ECM_EXP_CACHE_MIN_B1)
                             ? std::string() : ecm_exp_cache_path(cache_dir, B1, torsion);

    if (!path.empty()) {
        std::string why;
        const Clock::time_point t0 = Clock::now();
        if (ecm_exp_cache_load(path, s, B1, torsion, &why)) {
            if (detail) {
                snprintf(buf, sizeof buf, "cache hit (%.2f s, %s)",
                         secs_since(t0), path.c_str());
                *detail = buf;
            }
            return true;
        }
        /* Not a hard error: rebuild, then overwrite the stale/invalid entry. */
        const bool built = ecm_build_lcm_exponent(s, B1, torsion);
        const double secs = secs_since(t0);
        if (!built) {
            if (detail) *detail = "build failed";
            return false;
        }
        std::string werr;
        const bool saved = ecm_exp_cache_save(path, s, B1, torsion, &werr);
        if (detail) {
            snprintf(buf, sizeof buf, "built in %.2f s (cache %s; previous entry rejected: %s)",
                     secs, saved ? "written" : "write failed", why.c_str());
            *detail = buf;
        }
        return true;
    }

    const Clock::time_point t0 = Clock::now();
    const bool built = ecm_build_lcm_exponent(s, B1, torsion);
    if (detail) {
        snprintf(buf, sizeof buf, built ? "built in %.2f s (cache off)" : "build failed",
                 secs_since(t0));
        *detail = buf;
    }
    return built;
}
