/* ---------------------------------------------------------------------------
 * ecm_stage1_exp_cache.h -- persistent cache for s = torsion * lcm(1..B1).
 *
 * WHY: at B1 = 260e6 the product is 375 Mbit (47 MB) and building it costs ~10.6 s even after
 * the optimisations in ecm_stage1_exp.cpp (the remaining time is inherent: GMP's FFT multiply
 * of two ~190 Mbit operands alone is 0.87 s, and a product tree over 375 Mbit needs several of
 * those, see docs/ECM_CGBN_OPTIMIZATION.md).  Every queue task repeats that build, so a
 * worktodo full of tasks at the same B1 pays it over and over.  Loading a validated cache file
 * costs ~0.3 s instead (hash + import + two divisibility checks).
 *
 * VALIDATION (a cache that cannot be trusted is worse than no cache):
 *   1. header: magic / format version / semantics id / exact B1 / exact torsion;
 *   2. sanity: bit length of lcm(1..B1) lies in a narrow band around 1.4427*B1, and the payload
 *      size matches the declared word count exactly (catches truncation);
 *   3. integrity: FNV-1a 64 + additive checksum over header fields AND payload (catches
 *      corruption and any edit of the header, e.g. renaming a cache built for another B1);
 *   4. SEMANTIC: the loaded value must be divisible by the product of the highest prime powers
 *      of (a) the first 32 primes <= B1 and (b) the 32 primes just below B1.  (b) is what proves
 *      the payload really belongs to this B1 -- a cache built for a smaller B1 fails it, and a
 *      cache built for a larger one is caught by the exact B1 comparison in (1).
 *   A cache that fails any check is ignored (the caller rebuilds and overwrites it).
 *
 * The file is written to "<dir>/<name>.tmp" and then renamed over the target, so a reader never
 * sees a partial file and two concurrent writers simply produce two identical valid files.
 * ------------------------------------------------------------------------- */
#ifndef ECM_STAGE1_EXP_CACHE_H
#define ECM_STAGE1_EXP_CACHE_H

#include <gmp.h>
#include <stdint.h>

#include <string>
#include <vector>

/* Build s = torsion * lcm(1..B1), reusing `cache_dir` when it holds a valid entry.
 * `cache_dir` empty (or B1 below the minimum) means "build every time".
 * `detail` receives a one-line human-readable status for the log, e.g.
 *   "cache hit 0.31 s (ecm_exp_lcm_260000000_t1_v1.bin)"
 *   "built in 10.6 s, cached"
 *   "cache ignored: payload checksum mismatch"
 * Returns false only when the value could not be produced at all. */
bool ecm_build_lcm_exponent_cached(mpz_t s, uint64_t B1, uint64_t torsion,
                                   const std::string &cache_dir, std::string *detail);

/* Path of the cache entry for (B1, torsion) inside `dir`. */
std::string ecm_exp_cache_path(const std::string &dir, uint64_t B1, uint64_t torsion);

/* Process-wide cache directory, set once from the CLI (--exp-cache) or ecm.ini (exp_cache).
   Empty disables caching; the string "off"/"none"/"0" passed to the setter also clears it.
   It is a global because the two producers of s live in different translation units
   (ecm_driver.cpp for the GPU batch, src/cpu/ecm_mont_cpu.cpp for the Montgomery path). */
void ecm_exp_cache_set_dir(const std::string &dir);
const std::string &ecm_exp_cache_get_dir(void);

/* Lower bound of B1 for which caching makes sense (below it the build is instantaneous). */
extern const uint64_t ECM_EXP_CACHE_MIN_B1;

/* Low-level pieces, exposed for the tests in tools/test/stage1_exp_check.cpp. */
bool ecm_exp_cache_load(const std::string &path, mpz_t s, uint64_t B1, uint64_t torsion,
                        std::string *why);
bool ecm_exp_cache_save(const std::string &path, const mpz_t s, uint64_t B1, uint64_t torsion,
                        std::string *why);

/* Validation helpers used by the loader (also exercised directly by the tests). */
bool ecm_exp_semantic_check(const mpz_t s, uint64_t B1, std::string *why);

#endif /* ECM_STAGE1_EXP_CACHE_H */
