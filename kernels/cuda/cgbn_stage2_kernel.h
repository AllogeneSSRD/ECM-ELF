/* cgbn_stage2_kernel.h — device side of the CUDA/CGBN ECM stage 2 (pairing / BSGS).

   M1 of docs/DEV_STAGE2_GPU_PLAN.md.  This is the *reference* stage 2
   (tools/bench/stage2_ref.cpp, algorithm=pairing) moved onto the GPU with the same
   algorithm and the same hit semantics, so the two paths can be compared directly:

     baby[j]  = [j]Q           j = 1 .. D/2            (Q = the stage-1 result)
     giant[i] = [iD]Q          i = 1 .. B2/D + 2
     for every prime p in (B1, B2]:
       p <= D/2 : the baby table already holds [p]Q, so the test is gcd(Z_[p]Q, N) > 1
                  -- no ladder, no arithmetic, just accumulate baby[p].Z
       otherwise: r = p mod D, j = min(r, D-r), i = (p -+ j)/D, and the test is
                  accumulate X_i Z_j - X_j Z_i   (== 0 mod p  iff  p | iD -+ j)

   The accumulated product is gcd'ed with N on the host.  The candidate set is a
   SUPERSET of the primes that can actually pay off (the same superset the CPU
   reference uses), which is why the two hit sets are directly comparable.

   ── Domain conventions (do not "optimise" them away) ─────────────────────────────

   Global memory holds plain residues; the kernel converts to CGBN's Montgomery domain
   (R = 2^BITS) with cgbn_bn2mont, exactly like cgbn_stage1_kernel.h does.  a24 MUST be
   converted as well: xDBL computes BB + a24*K with K = AA - BB (domain R^-1) and BB
   (domain R^-1), so a24*K has to land in the same domain -- that needs a24 in
   Montgomery form.  (Feeding a plain a24 silently produces a wrong curve, which is the
   kind of bug that only shows up as "no factor found".)

   Everything else is domain-free: X_i Z_j - X_j Z_i is homogeneous of degree 2 in each
   point, both terms carry the SAME factor (projective scale and one R^-1 per mont_mul),
   so the factor cancels in the difference and the host gcd only ever asks "is this
   0 mod p".  In particular the stage-1 point may be fed as (x : 1) with the affine x
   straight from a save file: no inversion and no domain conversion is needed anywhere
   in stage 2.

   ── Why chains, not ladders, for the tables ─────────────────────────────────────

   The tables are built with the serial chain  v[j] = xADD(v[j-1], v[1], v[j-2])
   (ONE differential addition per entry) rather than one ladder per entry.  Besides
   being far cheaper (xADD = 8 mults vs ~8*log2(j) for a ladder), it is UNIFORM: every
   instance walks the same trip count and takes the same branches, so no instance can
   diverge inside a CGBN collective operation.  Ladders are used only for quantities
   that are identical for every curve (the stage-1 scalar s and the giant step D),
   where uniformity holds exactly.
*/

#ifndef CGBN_STAGE2_KERNEL_H
#define CGBN_STAGE2_KERNEL_H 1

/* gmp.h MUST precede cgbn.h: cgbn.h picks cgbn_mpz.h when __GMP_H__ is defined and
   otherwise falls back to cgbn_cpu.h, which is an unconditional #error.  The stage-1 TUs
   do the same thing in that order. */
#include <gmp.h>

#include <cgbn.h>
#include <cuda.h>
#include <stdint.h>

#include <assert.h>   /* the kernel header is also parsed by the host TU */

#ifndef FORCE_INLINE
#define FORCE_INLINE __forceinline__
#endif

#ifndef CHECK_ERROR
#define CHECK_ERROR 0
#endif

/* Container overhead: N must fit in BITS - S2_CARRY_BITS.  Same value and rationale as
   the stage-1 header (intermediate (X+Z)/(X-Z) style adds must not wrap). */
#define S2_CARRY_BITS 6

#ifndef ECM_MAX_ROTATION
#define ECM_MAX_ROTATION 1
#endif

template<uint32_t tpi, uint32_t bits>
struct cgbn_s2_params_t {
  static const uint32_t TPB = 128;    /* launch block size (matches ECM_TPB's default) */
  static const uint32_t TPI = tpi;    /* threads per instance; one instance = one curve */
  static const uint32_t BITS = bits;  /* CGBN container size */
  /* CGBN's cgbn_context_t reads these four off the parameters type (cgbn_cuda.h:58-62);
     MAX_ROTATION is the limb rotation in the multiply, 1 = measured best/cheapest for the
     stage-1 family (docs/ECM_CGBN_OPTIMIZATION.md 5.5). */
  static const uint32_t MAX_ROTATION = ECM_MAX_ROTATION;
  static const uint32_t SHM_LIMIT = 0;
  static const bool     CONSTANT_TIME = false;
  /* Same per-tier register budget as the stage-1 family (docs/ECM_CGBN_OPTIMIZATION.md). */
  static const uint32_t REG_TARGET = (bits <= 2048u) ? 56u : ((bits <= 5120u) ? 128u : 255u);
};

/* ── Kernel arguments ────────────────────────────────────────────────────────────
   Every buffer is an array of 32-bit limbs in CGBN's little-endian order, so one big
   number is exactly LIMBS = BITS/32 words and a point is 2*LIMBS words.  Per-curve
   buffers are indexed by (curve * LIMBS); the two tables are indexed by
   ((curve * points + index) * 2 + 0/1) with points = half+1 and imax+1. */
struct s2_args {
  uint32_t curves;      /* curves in this batch (= number of table instances) */
  uint32_t segs;        /* pairing instances per curve, each with its own accumulator */
  uint32_t half;        /* D/2: baby table size */
  uint32_t imax;        /* B2/D + 2: giant table size */
  uint32_t n_pair;      /* number of pairing primes */
  uint32_t n_small;     /* number of primes <= D/2 */
  uint32_t np0;         /* -N^-1 mod 2^32 */
  uint32_t have_x;      /* 1 = every curve has an affine x in x_in (skip the stage-1 ladder) */
  uint32_t s_bits;      /* bit length of s = torsion*lcm(1..B1) */
  uint32_t d_bits;      /* bit length of D */

  const uint8_t  *sbits;   /* MSB-first bit array, s_bits long */
  const uint8_t  *dbits;   /* MSB-first bit array, d_bits long */

  const uint32_t *modulus;  /* LIMBS: N (the same for every curve) */
  const uint32_t *a24;      /* curves * LIMBS: (A+2)/4 mod N */
  const uint32_t *start_x;  /* curves * LIMBS: u^3 (used only when have_x = 0) */
  const uint32_t *start_z;  /* curves * LIMBS: v^3 (used only when have_x = 0) */
  const uint32_t *x_in;     /* curves * LIMBS: affine x of the stage-1 result */

  const uint32_t *p_i;      /* n_pair: giant index i of each pairing prime */
  const uint32_t *p_j;      /* n_pair: baby index j of each pairing prime */
  const uint32_t *p_small;  /* n_small: baby index j (= the prime) of each small prime */

  uint32_t *baby;       /* curves * (half+1) * 2 * LIMBS */
  uint32_t *giant;      /* curves * (imax+1) * 2 * LIMBS */
  uint32_t *z_stage1;   /* curves * LIMBS: Montgomery Z of [s]Q (the stage-1 gcd) */
  uint32_t *acc;        /* curves * segs * LIMBS: one product residue per instance */
};

/* ── x-only Montgomery arithmetic on one curve ───────────────────────────────── */
template<class params>
class s2_curve_t {
  public:
  typedef cgbn_context_t<params::TPI, params> context_t;
  typedef cgbn_env_t<context_t, params::BITS> env_t;
  typedef typename env_t::cgbn_t              bn_t;
  typedef cgbn_mem_t<params::BITS>            mem_t;

  context_t _context;
  env_t     _env;
  uint32_t  _np0;

  __device__ FORCE_INLINE s2_curve_t(cgbn_monitor_t monitor, cgbn_error_report_t *report,
                                     int32_t instance, uint32_t np0)
      : _context(monitor, report, (uint32_t)instance), _env(_context), _np0(np0) {}

  /* r <- r mod n; valid after cgbn_add of two values < n (so r < 2n). */
  __device__ FORCE_INLINE void norm_add(bn_t &r, const bn_t &n) {
    if (cgbn_compare(_env, r, n) >= 0)
      cgbn_sub(_env, r, r, n);
  }

  /* r <- a - b mod n */
  __device__ FORCE_INLINE void sub_mod(bn_t &r, const bn_t &a, const bn_t &b, const bn_t &n) {
    if (cgbn_sub(_env, r, a, b))
      cgbn_add(_env, r, r, n);
  }

  /* (X2:Z2) = [2](X:Z) -- 2*S + 2*M.  Alias-safe: X2 may alias X and Z2 may alias Z,
     because X and Z are both consumed before the first write to X2/Z2. */
  __device__ FORCE_INLINE void xdbl(bn_t &X2, bn_t &Z2, const bn_t &X, const bn_t &Z,
                                    const bn_t &a24, const bn_t &n) {
    bn_t t, u, AA, BB, K, dK;
    cgbn_add(_env, t, Z, X);
    norm_add(t, n);
    sub_mod(u, Z, X, n);
    cgbn_mont_sqr(_env, AA, t, n, _np0);          /* (X+Z)^2 */
    cgbn_mont_sqr(_env, BB, u, n, _np0);          /* (X-Z)^2 */
    cgbn_mont_mul(_env, X2, AA, BB, n, _np0);     /* X2 = (X+Z)^2 (X-Z)^2 */
    sub_mod(K, AA, BB, n);                        /* K = 4XZ */
    cgbn_mont_mul(_env, dK, K, a24, n, _np0);     /* a24 * 4XZ */
    cgbn_add(_env, u, BB, dK);
    norm_add(u, n);
    cgbn_mont_mul(_env, Z2, K, u, n, _np0);       /* Z2 = 4XZ ((X-Z)^2 + a24 4XZ) */
  }

  /* (X3:Z3) = (XP:ZP) + (XQ:ZQ), given the difference point (XD:ZD).
     4*M + 2*S.  Fully alias-safe: every product goes to a local and X3/Z3 are written
     last (writing X3 early would clobber XP when the caller passes the same object). */
  __device__ FORCE_INLINE void xadd(bn_t &X3, bn_t &Z3,
                                    const bn_t &XP, const bn_t &ZP,
                                    const bn_t &XQ, const bn_t &ZQ,
                                    const bn_t &XD, const bn_t &ZD, const bn_t &n) {
    bn_t a, b, tX, tZ;
    cgbn_mont_mul(_env, tX, XP, XQ, n, _np0);
    cgbn_mont_mul(_env, b, ZP, ZQ, n, _np0);
    sub_mod(a, tX, b, n);                         /* XP XQ - ZP ZQ */
    cgbn_mont_sqr(_env, a, a, n, _np0);
    cgbn_mont_mul(_env, tX, ZD, a, n, _np0);      /* ZD (XP XQ - ZP ZQ)^2 */
    cgbn_mont_mul(_env, tZ, XP, ZQ, n, _np0);
    cgbn_mont_mul(_env, b, ZP, XQ, n, _np0);
    sub_mod(b, tZ, b, n);                         /* XP ZQ - ZP XQ */
    cgbn_mont_sqr(_env, b, b, n, _np0);
    cgbn_mont_mul(_env, tZ, XD, b, n, _np0);      /* XD (XP ZQ - ZP XQ)^2 */
    cgbn_set(_env, X3, tX);
    cgbn_set(_env, Z3, tZ);
  }

  /* r = [k]Q for k given MSB-first in bits[0..n); the fixed difference point is Q.
     Leading zeros are skipped exactly like the CPU reference does; that makes the trip
     count depend on k, so this is only used for quantities that are identical across
     curves (the stage-1 scalar s, and the giant step D). */
  __device__ FORCE_INLINE void ladder(bn_t &rX, bn_t &rZ, const uint8_t *bits, uint32_t n,
                                      const bn_t &QX, const bn_t &QZ,
                                      const bn_t &a24, const bn_t &N) {
    uint32_t first = 0;
    while (first < n && bits[first] == 0)
      ++first;
    if (first >= n) {                       /* k == 0 -> the point at infinity (1:0) */
      cgbn_set_ui32(_env, rX, 1);
      cgbn_set_ui32(_env, rZ, 0);
      return;
    }
    bn_t r0X, r0Z, r1X, r1Z;
    cgbn_set(_env, r0X, QX);
    cgbn_set(_env, r0Z, QZ);
    xdbl(r1X, r1Z, QX, QZ, a24, N);          /* (r0,r1) = (Q, 2Q) */
    for (uint32_t i = first + 1; i < n; ++i) {
      if (bits[i]) {
        xadd(r0X, r0Z, r0X, r0Z, r1X, r1Z, QX, QZ, N);
        xdbl(r1X, r1Z, r1X, r1Z, a24, N);
      } else {
        xadd(r1X, r1Z, r0X, r0Z, r1X, r1Z, QX, QZ, N);
        xdbl(r0X, r0Z, r0X, r0Z, a24, N);
      }
    }
    cgbn_set(_env, rX, r0X);
    cgbn_set(_env, rZ, r0Z);
  }
};

/* ── Kernel 1: the stage-1 point (optional), the baby table and the giant table ────
   One instance per curve; every instance walks the same chains, so there is no
   divergence inside the CGBN collectives. */
template<class params>
__global__ void __maxnreg__(params::REG_TARGET)
kernel_s2_tables(cgbn_error_report_t *report, s2_args a) {
  const uint32_t LIMBS = params::BITS / 32;
  int32_t inst = (int32_t)((blockIdx.x * blockDim.x + threadIdx.x) / params::TPI);
  if (inst < 0 || (uint32_t)inst >= a.curves)
    return;
  const uint32_t curve = (uint32_t)inst;

  typedef s2_curve_t<params> curve_t;
  typename curve_t::mem_t *baby  = (typename curve_t::mem_t *)a.baby;
  typename curve_t::mem_t *giant = (typename curve_t::mem_t *)a.giant;

  cgbn_monitor_t monitor = CHECK_ERROR ? cgbn_report_monitor : cgbn_no_checks;
  curve_t c(monitor, report, inst, a.np0);

  typename curve_t::bn_t N, a24, QX, QZ, g1X, g1Z, pX, pZ, tX, tZ;
  cgbn_load(c._env, N, (typename curve_t::mem_t *)a.modulus);
  cgbn_load(c._env, a24, (typename curve_t::mem_t *)(a.a24 + (size_t)curve * LIMBS));
  /* a24 must be in Montgomery form (see the head comment).  bn2mont also hands back np0,
     so this is where a host/kernel disagreement about np0 would show up. */
  {
    const uint32_t np0_test = cgbn_bn2mont(c._env, a24, a24, N);
    assert(np0_test == a.np0);
    (void)np0_test;
  }

  /* ---- the stage-1 point Q ---- */
  if (a.have_x) {
    cgbn_load(c._env, QX, (typename curve_t::mem_t *)(a.x_in + (size_t)curve * LIMBS));
    cgbn_set_ui32(c._env, QZ, 1);
  } else {
    typename curve_t::bn_t sX, sZ;
    cgbn_load(c._env, sX, (typename curve_t::mem_t *)(a.start_x + (size_t)curve * LIMBS));
    cgbn_load(c._env, sZ, (typename curve_t::mem_t *)(a.start_z + (size_t)curve * LIMBS));
    c.ladder(QX, QZ, a.sbits, a.s_bits, sX, sZ, a24, N);   /* Q = [s]P0 */
  }
  cgbn_bn2mont(c._env, QX, QX, N);
  cgbn_bn2mont(c._env, QZ, QZ, N);
  cgbn_store(c._env, (typename curve_t::mem_t *)(a.z_stage1 + (size_t)curve * LIMBS), QZ);

  /* ---- baby[j] = [j]Q, j = 1..half: a chain of differential additions ---- */
  {
    const size_t bs = (size_t)(a.half + 1);
    typename curve_t::mem_t *my = baby + (size_t)curve * bs * 2;
    cgbn_store(c._env, &my[1 * 2 + 0], QX);
    cgbn_store(c._env, &my[1 * 2 + 1], QZ);
    if (a.half >= 2) {
      c.xdbl(tX, tZ, QX, QZ, a24, N);                 /* [2]Q */
      cgbn_store(c._env, &my[2 * 2 + 0], tX);
      cgbn_store(c._env, &my[2 * 2 + 1], tZ);
    }
    for (uint32_t j = 3; j <= a.half; ++j) {
      cgbn_load(c._env, pX, &my[(size_t)(j - 1) * 2 + 0]);   /* [j-1]Q */
      cgbn_load(c._env, pZ, &my[(size_t)(j - 1) * 2 + 1]);
      cgbn_load(c._env, tX, &my[(size_t)(j - 2) * 2 + 0]);   /* [j-2]Q = the difference */
      cgbn_load(c._env, tZ, &my[(size_t)(j - 2) * 2 + 1]);
      c.xadd(pX, pZ, pX, pZ, QX, QZ, tX, tZ, N);             /* [j]Q = [j-1]Q + Q */
      cgbn_store(c._env, &my[(size_t)j * 2 + 0], pX);
      cgbn_store(c._env, &my[(size_t)j * 2 + 1], pZ);
    }
  }

  /* ---- giant[i] = [iD]Q, i = 1..imax: one ladder + a chain ---- */
  {
    const size_t gs = (size_t)(a.imax + 1);
    typename curve_t::mem_t *my = giant + (size_t)curve * gs * 2;
    c.ladder(g1X, g1Z, a.dbits, a.d_bits, QX, QZ, a24, N);    /* giant[1] = [D]Q */
    cgbn_store(c._env, &my[1 * 2 + 0], g1X);
    cgbn_store(c._env, &my[1 * 2 + 1], g1Z);
    if (a.imax >= 2) {
      c.xdbl(tX, tZ, g1X, g1Z, a24, N);                       /* giant[2] = [2D]Q */
      cgbn_store(c._env, &my[2 * 2 + 0], tX);
      cgbn_store(c._env, &my[2 * 2 + 1], tZ);
    }
    for (uint32_t i = 3; i <= a.imax; ++i) {
      cgbn_load(c._env, pX, &my[(size_t)(i - 1) * 2 + 0]);    /* giant[i-1] */
      cgbn_load(c._env, pZ, &my[(size_t)(i - 1) * 2 + 1]);
      cgbn_load(c._env, tX, &my[(size_t)(i - 2) * 2 + 0]);    /* giant[i-2] = the difference */
      cgbn_load(c._env, tZ, &my[(size_t)(i - 2) * 2 + 1]);
      c.xadd(pX, pZ, pX, pZ, g1X, g1Z, tX, tZ, N);            /* giant[i] = giant[i-1] + giant[1] */
      cgbn_store(c._env, &my[(size_t)i * 2 + 0], pX);
      cgbn_store(c._env, &my[(size_t)i * 2 + 1], pZ);
    }
  }
}

/* ── Kernel 2: accumulate the pairing product ─────────────────────────────────────
   One instance per (curve, segment).  Every instance runs the same instruction
   sequence (only the prime index differs), so there is no divergent branch inside the
   loop -- that is why the "small prime" case is a separate loop rather than an if. */
template<class params>
__global__ void __maxnreg__(params::REG_TARGET)
kernel_s2_pair(cgbn_error_report_t *report, s2_args a) {
  const uint32_t LIMBS = params::BITS / 32;
  int32_t inst = (int32_t)((blockIdx.x * blockDim.x + threadIdx.x) / params::TPI);
  const uint32_t total = a.curves * a.segs;
  if (inst < 0 || (uint32_t)inst >= total)
    return;
  const uint32_t curve = (uint32_t)inst / a.segs;
  const uint32_t seg   = (uint32_t)inst % a.segs;

  typedef s2_curve_t<params> curve_t;
  typename curve_t::mem_t *baby  = (typename curve_t::mem_t *)a.baby;
  typename curve_t::mem_t *giant = (typename curve_t::mem_t *)a.giant;
  typename curve_t::mem_t *my_baby  = baby  + (size_t)curve * (size_t)(a.half + 1) * 2;
  typename curve_t::mem_t *my_giant = giant + (size_t)curve * (size_t)(a.imax + 1) * 2;

  cgbn_monitor_t monitor = CHECK_ERROR ? cgbn_report_monitor : cgbn_no_checks;
  curve_t c(monitor, report, inst, a.np0);

  typename curve_t::bn_t N, acc, Xi, Zi, Xj, Zj, t1, t2, d;
  cgbn_load(c._env, N, (typename curve_t::mem_t *)a.modulus);
  cgbn_set_ui32(c._env, acc, 1);
  cgbn_bn2mont(c._env, acc, acc, N);         /* acc = R: the first product is exactly d */

  for (uint32_t k = seg; k < a.n_pair; k += a.segs) {
    const uint32_t i = a.p_i[k], j = a.p_j[k];
    cgbn_load(c._env, Xi, &my_giant[(size_t)i * 2 + 0]);
    cgbn_load(c._env, Zi, &my_giant[(size_t)i * 2 + 1]);
    cgbn_load(c._env, Xj, &my_baby[(size_t)j * 2 + 0]);
    cgbn_load(c._env, Zj, &my_baby[(size_t)j * 2 + 1]);
    cgbn_mont_mul(c._env, t1, Xi, Zj, N, a.np0);
    cgbn_mont_mul(c._env, t2, Xj, Zi, N, a.np0);
    c.sub_mod(d, t1, t2, N);                 /* X_i Z_j - X_j Z_i */
    cgbn_mont_mul(c._env, acc, acc, d, N, a.np0);
  }
  for (uint32_t k = seg; k < a.n_small; k += a.segs) {
    const uint32_t j = a.p_small[k];         /* the prime itself: baby[p] = [p]Q */
    cgbn_load(c._env, Zj, &my_baby[(size_t)j * 2 + 1]);
    cgbn_mont_mul(c._env, acc, acc, Zj, N, a.np0);
  }

  cgbn_store(c._env, (typename curve_t::mem_t *)(a.acc + (size_t)inst * LIMBS), acc);
}

/* ── Tier dispatch (one entry per instantiated (TPI, BITS) pair) ───────────────── */
struct cgbn_s2_kernels_t {
  void (*tables)(cgbn_error_report_t *, s2_args);
  void (*pair)(cgbn_error_report_t *, s2_args);
  uint32_t bits;   /* container size of the tier, 0 when invalid */
  uint32_t tpi;
  uint32_t tpb;
};

/* Each returns the smallest tier of its family with bits >= want_bits, or a struct with
   bits == 0 when that family has none.  Implemented by cgbn_stage2_kernels_tpi<N>.cu. */
cgbn_s2_kernels_t cgbn_stage2_kernels_tpi4(uint32_t want_bits);
cgbn_s2_kernels_t cgbn_stage2_kernels_tpi8(uint32_t want_bits);

static inline bool cgbn_s2_valid(const cgbn_s2_kernels_t &k) { return k.bits != 0u; }

#endif /* CGBN_STAGE2_KERNEL_H */
