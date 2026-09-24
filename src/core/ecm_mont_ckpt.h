/* ---------------------------------------------------------------------------
 * ecm_mont_ckpt.h -- mid-stage-1 checkpoints for the Montgomery (Suyama sigma)
 * CPU paths, scalar mpn *and* AVX512-IFMA batch (docs/ECM_Montgomery_STAGE1.md §17).
 *
 * Scope, deliberately narrow:
 *   * INTERNAL format.  Only this program writes and reads it; the artifact that
 *     has to interoperate with gmp-ecm / Prime95 stays the .save file produced
 *     at the END of stage 1, which this module does not touch at all.
 *   * One text file per curve:  <dir>/<stem>_c0000017.ckpt , where <stem> is the
 *     .save stem (m3001_1e6).  The Edwards path uses the same <tmp_dir> idea with
 *     a per-curve file, so the two CPU methods behave alike for the queue manager.
 *   * Text rather than binary because it is written by several worker threads at
 *     unpredictable moments (interval timer, Ctrl+C) and it is the artifact a human
 *     stares at after a crash: key=value lines stay readable and greppable.  A
 *     truncated write is detected by the CHECKSUM/END pair at the end of the file.
 *
 * Why so little has to be saved: the ladder invariant is
 *     p0 = [k]P, p1 = [k+1]P       (k = exponent bits consumed)
 * and the curve constants (a24, xdiff) are functions of sigma alone, so a resume
 * point is (k, p0, p1) plus the parameters that identify the work (N, B1,
 * torsion, sigma).  A finished curve additionally stores its result (x, or the
 * factor for a hit) -- otherwise an interrupted run would have to recompute every
 * curve it had already finished.
 * ------------------------------------------------------------------------- */
#ifndef ECM_MONT_CKPT_H
#define ECM_MONT_CKPT_H

#include <gmp.h>
#include <stddef.h>
#include <stdint.h>

#include <string>

#define MONT_CKPT_VERSION 1

enum {
    MONT_CKPT_INFLIGHT = 0,
    MONT_CKPT_DONE     = 1
};

typedef struct {
    int      status;      /* MONT_CKPT_INFLIGHT | MONT_CKPT_DONE */
    int      hit;         /* DONE only: gcd(Z,N) yielded a non-trivial factor */
    size_t   bitnum;      /* INFLIGHT: exponent bits consumed (multiple of the chunk) */
    size_t   sbits;       /* total exponent bits (identity + progress check) */
    uint64_t sigma;       /* identifies the curve; adopted on resume */
    uint32_t curve;       /* 0-based curve index within the task */
    double   B1;
    int      torsion;     /* 1 = lcm (gmp-ecm param 0), 12 = Prime95 choose12 */
    char     field[32];   /* field layer token: "mpn" | "ifma" (informational) */

    /* INFLIGHT state, plain domain mod N (converted from lane-SoA by the caller) */
    mpz_t X0, Z0, X1, Z1;

    /* DONE payload: xout for a miss, factor for a hit */
    mpz_t xout, factor;
} mont_ckpt_t;

void mont_ckpt_init(mont_ckpt_t *ck);
void mont_ckpt_clear(mont_ckpt_t *ck);

/* Fill the identity fields every record carries (curve, sigma, B1, torsion,
   sbits, field token).  Callers only have to add status/bitnum/payload. */
void mont_ckpt_ident(mont_ckpt_t *ck, uint32_t curve, uint64_t sigma, double B1,
                     int torsion, size_t sbits, const char *field);
/* <stem>_c<curve, 7 digits>.ckpt -- one name per curve, so a checkpoint write
   never races with another worker (a curve belongs to exactly one batch). */
std::string mont_ckpt_path(const std::string &stem, uint32_t curve);

/* 0 on success.  The file is written in place: a crash mid-write leaves no
   CHECKSUM/END, which the reader rejects (no stale-but-valid half state). */
int mont_ckpt_write(const std::string &path, const mpz_t N, const mont_ckpt_t *ck);

/* 1 = loaded and consistent with (N, B1, torsion, sbits), 0 = absent, unreadable,
   truncated or belonging to different work.  `why` (optional) receives the reason
   for a rejection, for the startup log. */
int mont_ckpt_read(const std::string &path, const mpz_t N, double B1, int torsion,
                   size_t sbits, mont_ckpt_t *ck, std::string *why);

void mont_ckpt_remove(const std::string &path);

#endif /* ECM_MONT_CKPT_H */
