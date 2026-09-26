/* ecm_stage1_exp.cpp -- see ecm_stage1_exp.h for the contract and the history.
 *
 * HOW s = torsion * lcm(1..B1) IS BUILT  (all three choices are measured decisions)
 *
 *  1. Sieve: odds-only and SEGMENTED (2^19 odds = 512 KiB of flags per segment, hot in L2).
 *     The original full `std::vector<char> sieve(B1+1)` was 260 MB at B1 = 260e6.  Measured
 *     with -DECM_EXP_TIMING: the segmented marking itself is only ~0.13 s, so this is about
 *     memory/cache behaviour, not about the marking count (~257M marks).
 *
 *  2. Product: primes are folded into a SMALL BLOCK accumulator (`mpz_mul_ui` on a value of
 *     at most EXP_BLOCK_BITS bits), and a finished block is folded into a binary-counter
 *     product tree.  Multiplying every prime power straight into the tree costs a GMP call
 *     chain (set_ui/mul_2exp/add_ui/cmp/mul) per prime: measured 0.55 us per prime, i.e.
 *     7.8 s of the 14.2 s at B1 = 260e6.  With the block accumulator the per-prime work is a
 *     single in-place `mpz_mul_ui` on <= 2048 bits, which is ~10-20 ns.
 *
 *     `highest_power()` is only needed for p <= sqrt(B1) (2,001 primes at 260e6); every larger
 *     prime contributes p itself, so the per-prime 64-bit division disappears from the hot loop.
 *
 *  3. Combine: the counter's occupied slots are reduced PAIRWISE.  A sequential combine from
 *     the largest slot first (the obvious loop) still costs ~6.3 s at B1 = 260e6 because it
 *     multiplies the growing 47 MB accumulator by each remaining slot; pairwise keeps every
 *     multiply between operands of comparable size.
 *
 * The reference implementation (PrMers, PrMers-main/src/modes/RunEcmTwistedEdwards.cpp
 * "Building K", lines ~1399-1532) does the same three things: odds-only segmented primes, a
 * fixed-size chunk accumulator, and a pairwise chunk reduction.
 *
 * Exactness is unaffected: multiplication is exact and commutative, so the result is the SAME
 * integer; only the order changes.  tools/test/stage1_exp_check.cpp verifies that against the
 * naive loop and against p-adic valuations, and bench_lcm.cpp compares builders bit for bit.
 */
#include "ecm_stage1_exp.h"

#include <stddef.h>
#include <string.h>

#include <vector>

#ifdef ECM_EXP_TIMING
/* Diagnostic: compile with -DECM_EXP_TIMING and call ecm_exp_timing_report(). */
#include <chrono>
#include <cstdio>
static double g_t_base = 0.0, g_t_mark = 0.0, g_t_block = 0.0, g_t_fold = 0.0, g_t_final = 0.0;
static unsigned long long g_primes = 0, g_marks = 0, g_folds = 0;
void ecm_exp_timing_report(void)
{
    printf("   [timing] base-primes=%.3f mark=%.3f block=%.3f fold=%.3f final=%.3f s"
           "  (primes=%llu marks=%llu folds=%llu)\n",
           g_t_base, g_t_mark, g_t_block, g_t_fold, g_t_final, g_primes, g_marks, g_folds);
}
#define EXP_CLOCK_NOW() std::chrono::steady_clock::now()
#define EXP_SECS(a, b) std::chrono::duration<double>((b) - (a)).count()
#else
#define EXP_CLOCK_NOW() 0
#define EXP_SECS(a, b) 0.0
#endif

/* mpz_set_ui()/mpz_mul_ui() take `unsigned long`, which is 32 bits on Windows, so a 64-bit
   value must be assembled from its halves (same trap as mont_set_sigma: sigma = 2^62 silently
   became 0 through mpz_set_ui). */
static void set_u64(mpz_t r, uint64_t v)
{
    mpz_set_ui(r, (unsigned long)(v >> 32));
    mpz_mul_2exp(r, r, 32);
    mpz_add_ui(r, r, (unsigned long)(v & 0xFFFFFFFFull));
}

/* Binary counter height: one slot per bit of the folded-block count.  With 2048-bit blocks a
   B1 of 5e9 needs ~1.8e6 blocks, i.e. 21 slots; 40 leaves a wide margin. */
#define EXP_SLOTS 40

/* Block accumulator size, in 32-bit limbs.  The hot operation is "multiply the block by one
   prime power", which happens once per prime (14.2M times at B1 = 260e6), so the block must be
   SMALL: cost is O(limbs) per prime plus one tree fold per (32*limbs*~1.39) primes.
   Measured trade-off at B1 = 260e6 (see the table in ECM_CGBN_OPTIMIZATION / the header):
     8 limbs (256 bits) -> 0.14 s multiply + ~1.05 s folds
    16 limbs (512 bits) -> 0.27 s        + ~0.52 s folds   <- used
    64 limbs (2048 bits) -> 1.08 s       + ~0.13 s folds  */
#define EXP_BLOCK_LIMBS 16u

/* Odd numbers per sieve segment: 2^19 odds = 512 KiB of flags (1 MiB of integers). */
#define EXP_SEGMENT_ODDS (1u << 19)

namespace {

/* Hand-rolled little-endian 32-bit limb accumulator.
   GMP's mpz_mul_ui() allocates a temporary limb array and copies the result back on EVERY call
   (measured ~600 ns per prime here, 8.5 s of the 11.1 s build at B1 = 260e6); multiplying in
   place costs L*~1.2 ns.  32-bit limbs are deliberate: MSVC has no __int128, and a 32x32->64
   product is exactly what the loop needs. */
struct BlockAcc {
    uint32_t limb[EXP_BLOCK_LIMBS];
    unsigned n;                       /* used limbs; n == 0 means "1" */

    void reset() { n = 0; }

    /* True when multiplying by another 32-bit value may not fit: the caller must flush first.
       (An earlier version dropped the top carry limb instead, which silently LOST prime powers:
       measured 1,437,860 bits instead of 1,442,099 at B1 = 1e6.) */
    bool needs_flush() const { return n + 1u >= EXP_BLOCK_LIMBS; }

    void mul_u32(uint32_t v)
    {
        if (v <= 1) return;
        if (n == 0) { limb[0] = v; n = 1; return; }    /* 1 * v = v (n == 0 encodes 1) */
        uint64_t carry = 0;
        for (unsigned i = 0; i < n; ++i) {
            const uint64_t t = (uint64_t)limb[i] * (uint64_t)v + carry;
            limb[i] = (uint32_t)t;
            carry = t >> 32;
        }
        if (carry) limb[n++] = (uint32_t)carry;        /* guaranteed to fit by needs_flush() */
    }

    /* out = current value (1 when empty) */
    void export_to(mpz_t out) const
    {
        if (n == 0) { mpz_set_ui(out, 1); return; }
        mpz_import(out, (size_t)n, -1, sizeof(uint32_t), 0, 0, limb);
    }
};

struct ProductTree {
    mpz_t slot[EXP_SLOTS];

    void init()
    {
        for (unsigned j = 0; j < EXP_SLOTS; ++j) mpz_init_set_ui(slot[j], 1);
    }
    void clear()
    {
        for (unsigned j = 0; j < EXP_SLOTS; ++j) mpz_clear(slot[j]);
    }
    /* Fold one finished block into the counter: carry it up until an empty slot accepts it.
       The carry MUST go through a temporary (product of the slots it passes) and then land in
       the first empty slot.  An earlier version multiplied slot[j] into slot[j+1] as it walked
       up, which destroys the "slot j holds 2^j blocks" invariant: the values then grow by
       merging ever-larger slots and the build becomes quadratic (observed as a >10 minute run
       at B1 = 260e6 -- do not "optimise" this into an in-place carry again). */
    void add(const mpz_t block)
    {
        if (mpz_cmp_ui(block, 1) == 0) return;
        mpz_t carry;
        mpz_init_set(carry, block);
        unsigned j = 0;
        while (j + 1 < EXP_SLOTS && mpz_cmp_ui(slot[j], 1) != 0) {
            mpz_mul(carry, carry, slot[j]);
            mpz_set_ui(slot[j], 1);
            ++j;
        }
        mpz_set(slot[j], carry);
        mpz_clear(carry);
    }
    /* Rare path (B1 > 4.29e9 only): fold a prime power that does not fit a uint32 multiplier. */
    void add_u64(uint64_t v)
    {
        mpz_t t;
        mpz_init(t);
        set_u64(t, v);
        add(t);
        mpz_clear(t);
    }
    /* Pairwise reduction of the occupied slots: never multiply a huge accumulator by a much
       smaller slot in sequence (that is what cost 6.3 s). */
    void finish_into(mpz_t s)
    {
        std::vector<unsigned> occ;
        for (unsigned j = 0; j < EXP_SLOTS; ++j) {
            if (mpz_cmp_ui(slot[j], 1) != 0) occ.push_back(j);
        }
        if (occ.empty()) return;
        while (occ.size() > 1) {
            std::vector<unsigned> next;
            for (size_t i = 0; i + 1 < occ.size(); i += 2) {
                mpz_mul(slot[occ[i]], slot[occ[i]], slot[occ[i + 1]]);
                next.push_back(occ[i]);
            }
            if (occ.size() & 1u) next.push_back(occ.back());
            occ.swap(next);
        }
        mpz_mul(s, s, slot[occ[0]]);
    }
};

static uint64_t isqrt_u64(uint64_t n)
{
    uint64_t r = 0;
    uint64_t bit = 1ull << 62;
    while (bit > n) bit >>= 2;
    while (bit) {
        if (n >= r + bit) {
            n -= r + bit;
            r = (r >> 1) + bit;
        } else {
            r >>= 1;
        }
        bit >>= 2;
    }
    return r;
}

/* Highest power of p that is <= B1 (the prime power that belongs in lcm(1..B1)). */
static uint64_t highest_power(uint64_t p, uint64_t B1)
{
    uint64_t pp = p;
    while (pp <= B1 / p) pp *= p;
    return pp;
}

/* Bit length of a nonzero uint64. */
static unsigned bitlen_u64(uint64_t v)
{
    unsigned n = 0;
    while (v) { ++n; v >>= 1; }
    return n;
}

} /* namespace */

bool ecm_build_lcm_exponent(mpz_t s, uint64_t B1, uint64_t torsion)
{
    mpz_set_ui(s, torsion ? (unsigned long)torsion : 1);
    if (B1 < 2) return true;
    if (B1 > 5000000000ull) return false;

    ProductTree tree;
    tree.init();
    BlockAcc block;
    block.reset();
    bool ok = true;

    const uint64_t root = isqrt_u64(B1);

    mpz_t scratch;                     /* only used to hand a block over to the tree */
    mpz_init2(scratch, EXP_BLOCK_LIMBS * 32u + 64u);

    /* Fold the block accumulator into the tree and start a new block. */
    struct Folder {
        static void flush(BlockAcc *b, ProductTree *tree, mpz_t scratch)
        {
            if (b->n == 0) return;
            b->export_to(scratch);
            tree->add(scratch);
            b->reset();
#ifdef ECM_EXP_TIMING
            ++g_folds;
#endif
        }
    };

    do {   /* one-pass body, so the failure path has a single exit */
        /* 2 is the only even prime: handle it directly so the sieve can be odds-only. */
        {
            const uint64_t pp = highest_power(2, B1);
            block.mul_u32((uint32_t)pp);            /* pp <= B1 <= 5e9 fits 32 bits */
        }

        if (root < 3) {
            for (uint64_t p = 3; p <= B1; p += 2) {
                const uint64_t pp = highest_power(p, B1);
                if (block.needs_flush()) Folder::flush(&block, &tree, scratch);
                if (pp <= 0xFFFFFFFFull) {
                    block.mul_u32((uint32_t)pp);
                } else {
                    tree.add_u64(pp);
                }
            }
            break;
        }

        const auto tb0 = EXP_CLOCK_NOW();
        std::vector<char> base;
        try {
            base.assign((size_t)root + 1u, 1);
        } catch (...) {
            ok = false;
            break;
        }
        base[0] = base[1] = 0;
        for (uint64_t p = 2; p * p <= root; ++p) {
            if (!base[(size_t)p]) continue;
            for (uint64_t q = p * p; q <= root; q += p) base[(size_t)q] = 0;
        }
        std::vector<uint64_t> base_primes;
        for (uint64_t p = 3; p <= root; p += 2) {
            if (base[(size_t)p]) base_primes.push_back(p);
        }
        const auto tb1 = EXP_CLOCK_NOW();
#ifdef ECM_EXP_TIMING
        g_t_base += EXP_SECS(tb0, tb1);
#endif

        std::vector<char> seg;
        try {
            seg.assign(EXP_SEGMENT_ODDS, 1);
        } catch (...) {
            ok = false;
            break;
        }

        uint64_t lo = 3;
        while (lo <= B1) {
            const uint64_t span = (uint64_t)(EXP_SEGMENT_ODDS - 1u) * 2ull;
            /* NOTE: clamp with a comparison, never with `B1 - span` -- that underflows for
               B1 < span and silently admits primes above B1 (measured: +70,719 bits at
               B1 = 1e6, exactly the primes in (1e6, 1.048e6]). */
            uint64_t hi = lo + span;
            if (hi > B1) hi = B1;
            if ((hi & 1ull) == 0) hi -= 1;                  /* keep the right end odd */
            const size_t cnt = (size_t)((hi - lo) / 2ull) + 1u;
            memset(&seg[0], 1, cnt);

            const auto tm0 = EXP_CLOCK_NOW();
            for (size_t i = 0; i < base_primes.size(); ++i) {
                const uint64_t p = base_primes[i];
                uint64_t start = p * p;
                if (start < lo) {
                    start = ((lo + p - 1ull) / p) * p;
                    if ((start & 1ull) == 0) start += p;    /* p is odd: step to the odd multiple */
                }
                for (uint64_t q = start; q <= hi; q += 2ull * p) {
                    seg[(size_t)((q - lo) / 2ull)] = 0;
#ifdef ECM_EXP_TIMING
                    ++g_marks;
#endif
                }
            }
            const auto tm1 = EXP_CLOCK_NOW();

            for (size_t i = 0; i < cnt; ++i) {
                if (!seg[i]) continue;
                const uint64_t p = lo + 2ull * (uint64_t)i;
                /* Only p <= sqrt(B1) can occur with a power > 1. */
                const uint64_t pp = (p <= root) ? highest_power(p, B1) : p;
                /* Exact flush trigger: keep room for the carry limb instead of estimating the
                   block size from bit lengths (an estimate can be a bit short and then the
                   carry would be dropped). */
                if (block.needs_flush()) {
#ifdef ECM_EXP_TIMING
                    const auto tf0 = EXP_CLOCK_NOW();
#endif
                    Folder::flush(&block, &tree, scratch);
#ifdef ECM_EXP_TIMING
                    g_t_fold += EXP_SECS(tf0, EXP_CLOCK_NOW());
#endif
                }
                if (pp <= 0xFFFFFFFFull) {
                    block.mul_u32((uint32_t)pp);
                } else {
                    /* B1 > 4.29e9 (only reachable with an explicit huge B1): rare path. */
                    Folder::flush(&block, &tree, scratch);
                    tree.add_u64(pp);
                }

#ifdef ECM_EXP_TIMING
                ++g_primes;
#endif
            }
            const auto tm2 = EXP_CLOCK_NOW();
#ifdef ECM_EXP_TIMING
            g_t_mark += EXP_SECS(tm0, tm1);
            g_t_block += EXP_SECS(tm1, tm2);
#endif

            if (hi >= B1) break;
            lo = hi + 2ull;
        }
    } while (false);

    if (ok) {
        Folder::flush(&block, &tree, scratch);
        const auto tf0 = EXP_CLOCK_NOW();
        tree.finish_into(s);
        const auto tf1 = EXP_CLOCK_NOW();
#ifdef ECM_EXP_TIMING
        g_t_final += EXP_SECS(tf0, tf1);
#endif
    }

    mpz_clear(scratch);
    tree.clear();
    return ok;
}
