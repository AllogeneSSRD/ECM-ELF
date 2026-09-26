#ifndef _CGBN_STAGE1_KERNEL_H
#define _CGBN_STAGE1_KERNEL_H 1

/* cgbn_stage1_kernel.h — device-side CGBN ECM stage-1 kernel templates and the
   per-TPI dispatch seam.

   Included by:
     - the per-TPI kernel translation units (cgbn_stage1_kernels_tpi*.cu), which
       instantiate the __global__ kernel_double_add template for a subset of
       (TPI, BITS);
     - the host TU (kernels/cuda/cgbn_stage1.cu), which only references the
       typedefs and calls the dispatch functions (no kernel instantiation).

   The includer must have already #included <gmp.h> then <cgbn.h> (in that
   order, as required by CGBN). */

#include <cassert>
#include <cstdint>

// See cgbn_error_t enum (cgbn.h:39)
#define cgbn_normalized_error ((cgbn_error_t) 14)
#define cgbn_positive_overflow ((cgbn_error_t) 15)
#define cgbn_negative_overflow ((cgbn_error_t) 16)

// Seems to adds very small overhead (1-10%)
#define VERIFY_NORMALIZED 0
// Adds even less overhead (<1%)
#define CHECK_ERROR 1

// Tested with check_gpuecm.sage
#define CARRY_BITS 6

// Can dramatically change compile time
#if 1
    #define FORCE_INLINE __forceinline__
#else
    #define FORCE_INLINE
#endif

// MPA-OpenCl port: dev build (kernels up to 1024 bits) is the default. Define
// ECM_CUDA_FULL_BUILD at compile time to build the full kernel set instead.
#ifndef ECM_CUDA_FULL_BUILD
#define IS_DEV_BUILD
#endif

// ---------------------------------------------------------------------------
// PROBE ONLY -- do not enable in production (docs/ECM_CGBN_OPTIMIZATION.md §5)
//
// ECM_PROBE_ADD_DENSITY = k makes the fused double-and-add execute its *addition* half only
// once every k-th bit.  THE RESULT IS MATHEMATICALLY WRONG for k > 1: it exists purely to
// measure the timing ceiling of an add-chain schedule (PRAC / NAF + dictionary), where the
// ladder does one doubling per bit but only ~1/3..1/6 of the additions.  k = 1 (default) is
// the correct kernel.  Build with -DECM_PROBE_ADD_DENSITY=k to run the probe.
// ---------------------------------------------------------------------------
#ifndef ECM_PROBE_ADD_DENSITY
#define ECM_PROBE_ADD_DENSITY 1
#endif

// ---------------------------------------------------------------------------
// PROBE ONLY -- do not enable in production (docs/ECM_CGBN_OPTIMIZATION.md §5.4)
//
// ECM_PROBE_CHAIN_W = M executes, per bit, ONE xz doubling plus one REAL
// differential addition every M-th bit, where that addition uses a PROJECTIVE
// difference point (EFD dadd-1987-m-3, 4M+2S).  That is the price any
// dictionary / window / co-Z chain has to pay, because a chain's difference is a
// maintained (projective) point, not our affine-normalised start point (2M+2S).
//   M = 1 -> PRAC-like op mix (one chain addition per bit)
//   M = w+1 -> ideal width-w NAF density (1/(w+1) additions per bit) with FREE
//              invariant maintenance -- an upper bound no real chain reaches
// THE RESULT IS MATHEMATICALLY WRONG (the difference point is fake).  Timing only.
// M = 0 (default) disables the probe and the correct fused double-and-add runs.
// ---------------------------------------------------------------------------
#ifndef ECM_PROBE_CHAIN_W
#define ECM_PROBE_CHAIN_W 0
#endif

// ---------------------------------------------------------------------------
// A/B experiment (docs/ECM_CGBN_OPTIMIZATION.md §8 item 3): scheduling variant of the
// fused step.  ECM_STEP_VARIANT = 1 (default) keeps the historical double_add_v2;
// 2 selects double_add_v2_ssa, which computes exactly the same arithmetic with explicit
// prep registers (no write-after-read hazard).  Both must give bit-identical saves.
// ---------------------------------------------------------------------------
#ifndef ECM_STEP_VARIANT
#define ECM_STEP_VARIANT 1
#endif

// ---------------------------------------------------------------------------
// PROBE ONLY (docs/ECM_CGBN_OPTIMIZATION.md §5.6): run the *param2-shaped* step (affine
// difference folded into a shift, full-width a24) on the param0 data path to price the
// param2 kernel.  RESULTS ARE WRONG (a24 is truncated to 32 bits); timing only.
// ---------------------------------------------------------------------------
#ifndef ECM_PARAM2SHAPE
#define ECM_PARAM2SHAPE 0
#endif

/* TODO test how this changes gpu_throughput_test */
/* NOTE: >= 512 may not be supported for > 2048 bit kernels */
//
// Tunables (docs/ECM_CGBN_OPTIMIZATION.md §5.5/§8).  Both are COMPILE-TIME on purpose:
// CGBN's shuffles assume the block really is params::TPB threads wide, so the launch
// config in kernels/cuda/cgbn_stage1.cu reads the same constant.  Override with
// -DECM_TPB=<n> / -DECM_MAX_ROTATION=<n> to sweep them.
//
// Defaults changed 2026-09-25 from 256/4 after a measured sweep (M511 and M761, 8192
// curves, B1=1e5, param3, GPU 1, median of 2):
//   * blocks = curves / (TPB/TPI) is the throughput driver: TPB=512 (half the blocks)
//     measured -15%, TPB=128 +1.2%, TPB=64 +1.1%.
//   * MAX_ROTATION 1/2/4 differ by <=0.4% (noise); 1 is the cheapest to be sure about.
//   * The bigger single lever is the register cap, applied per source file in
//     CMakeLists (tpi4/tpi8 only, i.e. <=2048 bits): 72 -> 56 registers gives +4.7%
//     at M511/M761 (48 overflows and loses it again).  The >=2560-bit tiers are left
//     at the compiler default because a 72+ register allocation there is unmeasured.
#ifndef ECM_TPB
#define ECM_TPB 128
#endif
#ifndef ECM_MAX_ROTATION
#define ECM_MAX_ROTATION 1
#endif

// A/B switch for the register target (0 = use the per-tier value encoded in
// cgbn_params_t::REG_TARGET).  Set it to e.g. 255 to reproduce the "let ptxas decide"
// allocation the ncu profile of the 4070 Ti run showed (140-141 registers).
#ifndef ECM_REG_TARGET_FORCE
#define ECM_REG_TARGET_FORCE 0
#endif

// Compile-time switch for the param2 kernel family (cgbn_stage1_kernels_param2.cu).
// `-DECM_NO_PARAM2=1` compiles NO param2 instantiation at all: the family is a second
// full copy of every tier, so skipping it cuts the kernel compile time noticeably while
// working on the param0/param3 paths.  `--gpu-param 2` then fails with an explicit error
// instead of silently falling back to another parametrization.
#ifndef ECM_NO_PARAM2
#define ECM_NO_PARAM2 0
#endif

const uint32_t TPB_DEFAULT = ECM_TPB;

template<uint32_t tpi, uint32_t bits>
class cgbn_params_t {
  public:
  // parameters used by the CGBN context
  static const uint32_t TPB=TPB_DEFAULT;           // Reasonable default
  static const uint32_t MAX_ROTATION=ECM_MAX_ROTATION; // good default value
  static const uint32_t SHM_LIMIT=0;               // no shared mem available
  // MPA-OpenCl port: CONSTANT_TIME is required by CGBN's cgbn_context_t on all
  // compilers (was previously mis-guarded behind #ifndef _MSC_VER).
  static const bool     CONSTANT_TIME=false;       // not implemented

  // parameters used locally in the application
  static const uint32_t TPI=tpi;                   // threads per instance
  static const uint32_t BITS=bits;                 // instance size

  /* Per-tier register target, ENCODED INTO THE INSTANTIATION.
     __maxnreg__(N) constrains ONE kernel, so unlike --maxrregcount (which is per source
     FILE) the register budget can differ per bit width -- which matters because the
     suyama/param2 files contain every tier.  N must be >= 1 for every instantiation, so
     "do not cap" is spelled 255 (= the sm_89 per-thread maximum, i.e. no constraint).

     The table below is measured, not guessed (docs/ECM_CGBN_OPTIMIZATION.md 5.7).
     ptxas -v register counts for the suyama (param0) family, TPB=128:

       bits :  2560  3072  3584  4096  4608  5120  5632  6144  7168  8192
       regs :    86    98   109   117   129   141   163   174   186   211   (uncapped)
       blk/SM:   5     5     4     4     3     3     3     2     2     2

       <=2048 bits : 56.  M511/M761 sweep: 56 registers beat the compiler's 72 by +4.7%;
                      48 spilled and lost it again (the full-build check at M1021 gave +2.8%).
       2560..5120  : 128.  For 2560-4096 the natural allocation is already 86-124 registers,
                      so this cap does NOT bind (same code, still 4-5 blocks/SM, no spill).
                      For 4608/5120 it binds: 129/141 -> 128 registers moves 3 -> 4 blocks/SM
                      for 48/144 B of spill stores, and it MEASURABLY wins:
                        tier 4608 (M4423, 4060, TPB=128): 576 curves +1.7%, 768 +3.2%,
                                                         1152 +2.8%, 1920 +2.2%
                        tier 5120 (5000-bit prime, same GPU): 768 curves +2.5%, 1920 +1.4%
                      (768 curves on the 24-SM 4060 is the same "1.0 vs 1.33 waves" shape as
                      the 1920-curve batch on the 60-SM 4070 Ti, i.e. the production shape.)
                      WHAT the win actually is: removing a PARTIAL LAST WAVE, not the extra
                      resident warps.  At an equal wave shape the two allocations tie (576
                      curves = 1.0 wave for the 3-block allocation), and the 60-SM 4070 Ti
                      runs at B1=260e6 (2026-09-25, user data) measure 120 blocks = 2
                      blocks/SM = 8 warps/SM at 72.90 s/curve vs 240 blocks = 4 blocks/SM =
                      16 warps at 73.41 s/curve -- i.e. occupancy is neutral-to-negative
                      because the kernel is issue bound (ncu: Compute SOL ~83%, DRAM ~0.5%).
                      Use the cap to keep a batch wave-aligned; do not expect occupancy to pay.
                      Correctness: both allocations produce the identical stage-1 X.
       5632+ bits  : 255 (uncapped).  Here the cap is expensive -- 560 B of spill stores at
                      5632, 1028 B at 6144, 3356 B at 8192 -- and no measurement says the
                      extra blocks pay for it, so this stays an explicitly open item.
     NOTE: this table describes registers, which is TPB independent, so changing ECM_TPB
     does not invalidate it (the resulting blocks/SM does change: N blocks/SM needs
     TPB*regs*N <= 65536). */
  static const uint32_t REG_TARGET =
      (ECM_REG_TARGET_FORCE > 0) ? ECM_REG_TARGET_FORCE
                                 : ((bits <= 2048u) ? 56u : ((bits <= 5120u) ? 128u : 255u));
};


template<class params>
class curve_t {
  public:

  typedef cgbn_context_t<params::TPI, params>   context_t;
  typedef cgbn_env_t<context_t, params::BITS>   env_t;
  typedef typename env_t::cgbn_t                bn_t;
  typedef cgbn_mem_t<params::BITS>              mem_t;

  context_t _context;
  env_t     _env;
  int32_t   _instance; // which curve instance is this

  // Constructor
  __device__ FORCE_INLINE curve_t(cgbn_monitor_t monitor, cgbn_error_report_t *report, int32_t instance) :
      _context(monitor, report, (uint32_t)instance), _env(_context), _instance(instance) {}

  // Verify 0 <= r < modulus
  __device__ FORCE_INLINE void assert_normalized(bn_t &r, const bn_t &modulus) {
    //if (VERIFY_NORMALIZED && _context.check_errors())
    if (VERIFY_NORMALIZED && CHECK_ERROR) {

        // Negative overflow
        if (cgbn_extract_bits_ui32(_env, r, params::BITS-1, 1)) {
            _context.report_error(cgbn_negative_overflow);
        }
        // Positive overflow
        if (cgbn_compare(_env, r, modulus) >= 0) {
            _context.report_error(cgbn_positive_overflow);
        }
    }
  }

  // Normalize after addition
  __device__ FORCE_INLINE void normalize_addition(bn_t &r, const bn_t &modulus) {
      if (cgbn_compare(_env, r, modulus) >= 0) {
          cgbn_sub(_env, r, r, modulus);
      }
  }

  /**
   * Calculate (r * m) / 2^32 mod modulus
   *
   * This removes a factor of 2^32 which is not present in m.
   * Otherwise m (really d) needs to be passed as a bigint not a uint32
   */
  __device__ FORCE_INLINE void special_mult_ui32(bn_t &r, uint32_t m, const bn_t &modulus, uint32_t np0) {
    //uint32_t thread_i = (blockIdx.x*blockDim.x + threadIdx.x)%params::TPI;
    bn_t temp;

    uint32_t carry_t1 = cgbn_mul_ui32(_env, r, r, m);
    uint32_t t1_0 = cgbn_extract_bits_ui32(_env, r, 0, 32);
    uint32_t q = t1_0 * np0;
    uint32_t carry_t2 = cgbn_mul_ui32(_env, temp, modulus, q);

    // Can I call dshift_right(1) directly?
    cgbn_shift_right(_env, r, r, 32);
    cgbn_shift_right(_env, temp, temp, 32);
    // Add back overflow carry
    cgbn_insert_bits_ui32(_env, r, r, params::BITS-32, 32, carry_t1);
    cgbn_insert_bits_ui32(_env, temp, temp, params::BITS-32, 32, carry_t2);

    if (VERIFY_NORMALIZED) {
        // (uint32 * X) >> 32 is always less than X
        assert_normalized(r, modulus);
        assert_normalized(temp, modulus);
    }

    // Can't overflow because of CARRY_BITS
    int32_t carry_q = cgbn_add(_env, r, r, temp);
    carry_q += cgbn_add_ui32(_env, r, r, t1_0 != 0); // add 1

    if (carry_q > 0) {
        // This should never happen,
        // if CHECK_ERROR, no need for the conditional call to cgbn_sub
        if (CHECK_ERROR) {
            _context.report_error(cgbn_positive_overflow);
        } else {
            cgbn_sub(_env, r, r, modulus);
        }
    }

    // 0 <= r, temp < modulus => r + temp + 1 < 2*modulus
    if (cgbn_compare(_env, r, modulus) >= 0) {
        cgbn_sub(_env, r, r, modulus);
    }
  }

  /**
   * Compute simultaneously
   * (q : u) <- [2](q : u)
   * (w : v) <- (q : u) + (w : v)
   * A second implementation previously existed in cudakernel_default.cu
   * See dup_add_batch1 in batch.c
   */
  __device__ FORCE_INLINE void double_add_v2(
          bn_t &q, bn_t &u,
          bn_t &w, bn_t &v,
          uint32_t d,
          const bn_t &modulus,
          const uint32_t np0,
          const bool do_add = true) {
    // q = xA = aX
    // u = zA = aZ
    // w = xB = bX
    // v = zB = bZ

    /* Doesn't seem to be a large cost to using many extra variables */
    bn_t t, CB, DA, AA, BB, K, dK;

    /* Can maybe use one more bit if cgbn_add subtracts when carry happens */
    /* Might be nice to add a macro that verifies no carry out of cgbn_add */

    // Is there anything interesting like only one of these can overflow?
    if (do_add) {                                  // PROBE: skipped for k > 1 (see top of file)
      cgbn_add(_env, t, v, w); // t = (bZ + bX)
      normalize_addition(t, modulus);
      if (cgbn_sub(_env, v, v, w)) // v = (bZ - bX)
          cgbn_add(_env, v, v, modulus);
    }


    cgbn_add(_env, w, u, q); // w = (aZ + aX)
    normalize_addition(w, modulus);
    if (cgbn_sub(_env, u, u, q)) // u = (aZ - aX)
        cgbn_add(_env, u, u, modulus);
    if (VERIFY_NORMALIZED && do_add) {   // PROBE: t/v are stale when the add half is skipped
        assert_normalized(t, modulus);
        assert_normalized(v, modulus);
        assert_normalized(w, modulus);
        assert_normalized(u, modulus);
    }

    // NOTE (2026-09-24, CGBN optimization): NO normalize_addition() after a
    // cgbn_mont_mul / cgbn_mont_sqr.  CGBN's Montgomery multiply already ends with a
    // full conditional subtraction (core_mont_wmad.cu:178-189), so its result is < n
    // and the extra cgbn_compare + conditional cgbn_sub is pure overhead.  Measured
    // with tools/bench/cgbn_op_probe.cu: that compare+sub costs 43% of a mont_mul at
    // the 512-bit tier (15% at 1024, 2.6% at 3072, 1.7% at 4096), and this kernel did
    // 8 of them per bit -> removing them is a double-digit win at small N.
    // Keep the normalize after cgbn_add/cgbn_sub/cgbn_shift_left (those CAN exceed n).
    // See docs/ECM_CGBN_OPTIMIZATION.md.
    if (do_add) {                                  // PROBE: CB/DA exist only for the addition
      cgbn_mont_mul(_env, CB, t, u, modulus, np0); // C*B
      cgbn_mont_mul(_env, DA, v, w, modulus, np0); // D*A
    }

    /* Roughly 40% of time is spent in these two calls */
    cgbn_mont_sqr(_env, AA, w, modulus, np0);    // AA
    cgbn_mont_sqr(_env, BB, u, modulus, np0);    // BB
    if (VERIFY_NORMALIZED) {
        assert_normalized(CB, modulus);
        assert_normalized(DA, modulus);
        assert_normalized(AA, modulus);
        assert_normalized(BB, modulus);
    }

    // q = aX is finalized
    cgbn_mont_mul(_env, q, AA, BB, modulus, np0); // AA*BB
        assert_normalized(q, modulus);

    if (cgbn_sub(_env, K, AA, BB)) // K = AA-BB
        cgbn_add(_env, K, K, modulus);

    // By definition of d = (sigma / 2^32) % MODN
    // K = k*R
    // dK = d*k*R = (K * R * sigma) >> 32
    cgbn_set(_env, dK, K);
    special_mult_ui32(dK, d, modulus, np0); // dK = K*d
        assert_normalized(dK, modulus);

    cgbn_add(_env, u, BB, dK); // BB + dK
    normalize_addition(u, modulus);
    if (VERIFY_NORMALIZED) {
        assert_normalized(K, modulus);
        assert_normalized(dK, modulus);
        assert_normalized(u, modulus);
    }

    // u = aZ is finalized
    cgbn_mont_mul(_env, u, K, u, modulus, np0); // K(BB+dK)
        assert_normalized(u, modulus);

    if (do_add) {                     // PROBE: the (w:v) output is the addition's result
      cgbn_add(_env, w, DA, CB); // DA + CB
      normalize_addition(w, modulus);   // kept: DA + CB can reach 2n
      if (cgbn_sub(_env, v, DA, CB)) // DA - CB
          cgbn_add(_env, v, v, modulus);
      if (VERIFY_NORMALIZED) {
          assert_normalized(w, modulus);
          assert_normalized(v, modulus);
      }

      // w = bX is finalized
      cgbn_mont_sqr(_env, w, w, modulus, np0); // (DA+CB)^2 mod N
          assert_normalized(w, modulus);

      cgbn_mont_sqr(_env, v, v, modulus, np0); // (DA-CB)^2 mod N
          assert_normalized(v, modulus);

      // v = bZ is finalized
      cgbn_shift_left(_env, v, v, 1); // double
      normalize_addition(v, modulus);   // kept: the shift can exceed n
          assert_normalized(v, modulus);
    }
  }

  /**
   * A/B experiment (docs/ECM_CGBN_OPTIMIZATION.md §8 item 3): the *same* arithmetic as
   * double_add_v2, but written with explicit prep registers so that the source has no
   * write-after-read hazard between the addition half and the doubling half.  In
   * double_add_v2 the prep (aZ+aX, aZ-aX) is kept in the `w`/`u` registers, and the
   * doubling overwrites `u`; that forces the addition's two multiplies to issue early and
   * lengthens the live ranges.  Selected with -DECM_STEP_VARIANT=2.  Correctness is
   * identical (same formulas, same order of reduction), so the A/B is pure scheduling.
   */
  __device__ FORCE_INLINE void double_add_v2_ssa(
          bn_t &q, bn_t &u,
          bn_t &w, bn_t &v,
          uint32_t d,
          const bn_t &modulus,
          const uint32_t np0) {
    bn_t t, CB, DA, AA, BB, K, dK, ax, az, bx, bz;

    // ---- independent prep, no register is read after being written ----
    cgbn_add(_env, az, u, q); // aZ + aX
    normalize_addition(az, modulus);
    if (cgbn_sub(_env, ax, u, q)) // aZ - aX
        cgbn_add(_env, ax, ax, modulus);

    cgbn_add(_env, bz, v, w); // bZ + bX
    normalize_addition(bz, modulus);
    if (cgbn_sub(_env, bx, v, w))
        cgbn_add(_env, bx, bx, modulus);

    // ---- addition half: CB = (bZ+bX)(aZ-aX), DA = (bZ-bX)(aZ+aX) ----
    cgbn_mont_mul(_env, CB, bz, ax, modulus, np0);
    cgbn_mont_mul(_env, DA, bx, az, modulus, np0);

    // ---- doubling half ----
    cgbn_mont_sqr(_env, AA, az, modulus, np0);
    cgbn_mont_sqr(_env, BB, ax, modulus, np0);
    cgbn_mont_mul(_env, q, AA, BB, modulus, np0);

    if (cgbn_sub(_env, K, AA, BB))
        cgbn_add(_env, K, K, modulus);

    cgbn_set(_env, dK, K);
    special_mult_ui32(dK, d, modulus, np0);

    cgbn_add(_env, t, BB, dK);
    normalize_addition(t, modulus);
    cgbn_mont_mul(_env, u, K, t, modulus, np0);

    // ---- addition half tail ----
    cgbn_add(_env, w, DA, CB);
    normalize_addition(w, modulus);
    if (cgbn_sub(_env, v, DA, CB))
        cgbn_add(_env, v, v, modulus);

    cgbn_mont_sqr(_env, w, w, modulus, np0);
    cgbn_mont_sqr(_env, v, v, modulus, np0);
    cgbn_shift_left(_env, v, v, 1);
    normalize_addition(v, modulus);
  }

  /* -------------------------------------------------------------------------
   * PROBE ONLY: windowed-chain arithmetic (docs/ECM_CGBN_OPTIMIZATION.md §5.4).
   *
   * One xz doubling per bit, plus -- every M-th bit -- a REAL differential
   * addition whose difference point is PROJECTIVE (EFD dadd-1987-m-3, 4M+2S).
   * The operands/difference are fake (we reuse the live state), so the point
   * coordinates are meaningless; ONLY the operator mix and its cost are real:
   *   per bit: 2S + 2M + special_mult_ui32,   plus (do_dadd ? 2S + 4M : 0)
   * which is exactly what a width-w NAF / PRAC / co-Z chain would execute.
   * ------------------------------------------------------------------------- */
  __device__ FORCE_INLINE void chain_probe_step(
          bn_t &q, bn_t &u,
          bn_t &w, bn_t &v,
          uint32_t d,
          const bn_t &modulus,
          const uint32_t np0,
          const bool do_dadd) {
    bn_t t, CB, DA, AA, BB, K, dK, XD, ZD;

    // ---- (q,u) <- [2](q,u): same op mix as the production doubling half ----
    cgbn_add(_env, w, u, q); // w = (aZ + aX)
    normalize_addition(w, modulus);
    if (cgbn_sub(_env, u, u, q)) // u = (aZ - aX)
        cgbn_add(_env, u, u, modulus);

    cgbn_mont_sqr(_env, AA, w, modulus, np0);    // AA
    cgbn_mont_sqr(_env, BB, u, modulus, np0);    // BB
    cgbn_mont_mul(_env, q, AA, BB, modulus, np0); // q = AA*BB

    if (cgbn_sub(_env, K, AA, BB)) // K = AA-BB
        cgbn_add(_env, K, K, modulus);

    cgbn_set(_env, dK, K);
    special_mult_ui32(dK, d, modulus, np0); // dK = K*d (32-bit multiply)

    cgbn_add(_env, u, BB, dK); // BB + dK
    normalize_addition(u, modulus);
    cgbn_mont_mul(_env, u, K, u, modulus, np0); // u = K(BB+dK)

    if (!do_dadd)
      return;

    // ---- (w,v) <- dadd((w,v), (q,u)) with a PROJECTIVE difference (XD:ZD) ----
    // EFD dadd-1987-m-3:
    //   X5 = Z1*((X2-Z2)(X3+Z3) + (X2+Z2)(X3-Z3))^2
    //   Z5 = X1*((X2-Z2)(X3+Z3) - (X2+Z2)(X3-Z3))^2
    cgbn_set(_env, XD, q); // projective difference: reuses the live state on
    cgbn_set(_env, ZD, u); // purpose (timing probe only -- value is wrong)

    cgbn_add(_env, t, v, w); // (X2 + Z2)
    normalize_addition(t, modulus);
    if (cgbn_sub(_env, v, v, w)) // (X2 - Z2)
        cgbn_add(_env, v, v, modulus);

    cgbn_add(_env, CB, u, q); // (X3 + Z3)
    normalize_addition(CB, modulus);
    if (cgbn_sub(_env, DA, u, q)) // (X3 - Z3)
        cgbn_add(_env, DA, DA, modulus);

    cgbn_mont_mul(_env, v, v, CB, modulus, np0); // (X2-Z2)(X3+Z3)
    cgbn_mont_mul(_env, t, t, DA, modulus, np0); // (X2+Z2)(X3-Z3)

    cgbn_add(_env, CB, t, v); // sum
    normalize_addition(CB, modulus);
    if (cgbn_sub(_env, DA, t, v)) // difference
        cgbn_add(_env, DA, DA, modulus);

    cgbn_mont_sqr(_env, CB, CB, modulus, np0);   // sum^2
    cgbn_mont_sqr(_env, DA, DA, modulus, np0);   // difference^2

    cgbn_mont_mul(_env, w, ZD, CB, modulus, np0); // X5 = Z1*sum^2
    cgbn_mont_mul(_env, v, XD, DA, modulus, np0); // Z5 = X1*diff^2
  }

  /* -------------------------------------------------------------------------
   * Suyama param0 variant of the same fused double-and-add
   * (docs/ECM_Montgomery_STAGE1.md §19/§20).
   *
   * The param3 path above is cheap for TWO reasons, both coming from its fixed
   * shape "P_a = (2:1), P_b = 2P, difference x = 2" (upstream gmp-ecm
   * batch.c:167: "assume (x2:z2) - (x1:z1) = (2:1)"):
   *
   *   * its curve constant is a 32-bit value that plays the role of a24, so the
   *     doubling uses special_mult_ui32() (32xN multiply + one-word reduction)
   *     instead of a full-width multiply;
   *   * its differential addition needs no multiplication by the difference
   *     coordinate: x_D = 2 is folded into shift_left(v,1).
   *
   * Suyama param0 has P = (u^3 : v^3) and a full-width a24 = (A+2)/4, so both
   * shortcuts have to be given back -- and that is ALL that differs.  The op
   * count becomes 6M+4S per bit, exactly the CPU reference's, and the kernel
   * becomes sigma-agnostic (everything sigma-dependent is prepared on the host).
   * ------------------------------------------------------------------------- */
  __device__ FORCE_INLINE void double_add_v2_suyama(
          bn_t &q, bn_t &u,
          bn_t &w, bn_t &v,
          const bn_t &a24,
          const bn_t &xdiff,
          const bn_t &modulus,
          const uint32_t np0,
          // const_diff = true means "the ladder difference point is the constant 2",
          // i.e. the affine xdiff multiply becomes the shift_left(v,1) of the batch
          // family.  That is exactly gmp-ecm's param 2 (get_curve_from_param2 ends with
          // mpres_set_ui(x0, 2, n), parametrizations.c:373): full-width a24 (unlike
          // param3's 32-bit d) with a constant difference (unlike param0's xdiff).
          // 5M+4S per bit instead of param0's 6M+4S -- see
          // docs/ECM_CGBN_OPTIMIZATION.md §5.6.  Uniform across the warp, so free.
          const bool const_diff = false) {
    // q = xA = aX
    // u = zA = aZ
    // w = xB = bX
    // v = zB = bZ

    bn_t t, CB, DA, AA, BB, K, dK;

    cgbn_add(_env, t, v, w); // t = (bZ + bX)
    normalize_addition(t, modulus);
    if (cgbn_sub(_env, v, v, w)) // v = (bZ - bX)
        cgbn_add(_env, v, v, modulus);

    cgbn_add(_env, w, u, q); // w = (aZ + aX)
    normalize_addition(w, modulus);
    if (cgbn_sub(_env, u, u, q)) // u = (aZ - aX)
        cgbn_add(_env, u, u, modulus);

    cgbn_mont_mul(_env, CB, t, u, modulus, np0); // C*B
    cgbn_mont_mul(_env, DA, v, w, modulus, np0); // D*A

    cgbn_mont_sqr(_env, AA, w, modulus, np0);    // AA
    cgbn_mont_sqr(_env, BB, u, modulus, np0);    // BB

    // q = aX is finalized
    cgbn_mont_mul(_env, q, AA, BB, modulus, np0); // AA*BB

    if (cgbn_sub(_env, K, AA, BB)) // K = AA-BB = 4XZ
        cgbn_add(_env, K, K, modulus);

    // dK = a24 * K  (full width -- a24 is NOT a 32-bit batch parameter here)
    cgbn_mont_mul(_env, dK, K, a24, modulus, np0);
        assert_normalized(dK, modulus);

    cgbn_add(_env, u, BB, dK); // BB + a24*K
    normalize_addition(u, modulus);   // kept: BB + dK can reach 2n

    // u = aZ is finalized
    cgbn_mont_mul(_env, u, K, u, modulus, np0); // K(BB + a24*K)

    cgbn_add(_env, w, DA, CB); // DA + CB
    normalize_addition(w, modulus);   // kept: DA + CB can reach 2n
    if (cgbn_sub(_env, v, DA, CB)) // DA - CB
        cgbn_add(_env, v, v, modulus);

    // w = bX is finalized (Z_D = 1: the host normalises the difference point)
    cgbn_mont_sqr(_env, w, w, modulus, np0); // (DA+CB)^2 mod N

    cgbn_mont_sqr(_env, v, v, modulus, np0); // (DA-CB)^2 mod N

    // v = bZ is finalized: x_D * (DA-CB)^2, where x_D = X0/Z0 is the affine x of
    // the ladder difference point (param3 folds the constant 2 in here instead)
    if (const_diff) {
      // param2: x_D = 2 -- same shortcut as the batch family, no multiply at all
      cgbn_shift_left(_env, v, v, 1);
    } else {
      cgbn_mont_mul(_env, v, v, xdiff, modulus, np0);
    }
        assert_normalized(v, modulus);
  }
};


/**
 * Double-and-add, index decreasing algorithm.
 */
template<class params>
__global__ void __maxnreg__(params::REG_TARGET) kernel_double_add(
        cgbn_error_report_t *report,
        uint64_t s_bits,
        uint64_t s_bits_start,
        uint64_t s_bits_interval,
        uint32_t *gpu_s_bits,
        uint32_t *data,
        uint32_t count,
        uint32_t sigma_0,
        uint32_t np0
        ) {
  // decode an instance_i number from the blockIdx and threadIdx
  int32_t instance_i = (blockIdx.x*blockDim.x + threadIdx.x)/params::TPI;
  if(instance_i >= count)
    return;

  /* Cast uint32_t array to mem_t */
  typename curve_t<params>::mem_t *data_cast = (typename curve_t<params>::mem_t*) data;

  cgbn_monitor_t monitor = CHECK_ERROR ? cgbn_report_monitor : cgbn_no_checks;

  curve_t<params> curve(monitor, report, instance_i);
  typename curve_t<params>::bn_t  aX, aZ, bX, bZ, modulus;

  { // Setup
      cgbn_load(curve._env, modulus, &data_cast[5*instance_i+0]);
      cgbn_load(curve._env, aX, &data_cast[5*instance_i+1]);
      cgbn_load(curve._env, aZ, &data_cast[5*instance_i+2]);
      cgbn_load(curve._env, bX, &data_cast[5*instance_i+3]);
      cgbn_load(curve._env, bZ, &data_cast[5*instance_i+4]);

      /* Convert points to mont, has a miniscule bit of overhead with batching. */
      uint32_t np0_test = cgbn_bn2mont(curve._env, aX, aX, modulus);
      assert(np0 == np0_test);

      cgbn_bn2mont(curve._env, aZ, aZ, modulus);
      cgbn_bn2mont(curve._env, bX, bX, modulus);
      cgbn_bn2mont(curve._env, bZ, bZ, modulus);

      {
        curve.assert_normalized(aX, modulus);
        curve.assert_normalized(aZ, modulus);
        curve.assert_normalized(bX, modulus);
        curve.assert_normalized(bZ, modulus);
      }
  }

  /* Initially
     P_a = (aX, aZ) contains P
     P_b = (bX, bZ) contains 2P */

  // d = (sigma / 2^32) mod N BUT 2^32 handled by special_mult_ui32
  uint32_t d = sigma_0 + instance_i;

  int swapped = 0;
  for (uint64_t b = s_bits_start; b < s_bits_start + s_bits_interval; b++) {
    /* Process bits from MSB to LSB, last index to first index
     * b counts from 0 to s_num_bits */
    uint64_t nth = s_bits - 1 - b;

    int bit = (gpu_s_bits[nth/32] >> (nth&31)) & 1;
    if (bit != swapped) {
        swapped = !swapped;
        cgbn_swap(curve._env, aX, bX);
        cgbn_swap(curve._env, aZ, bZ);
    }
#if ECM_PROBE_CHAIN_W > 0
    // PROBE: replace the fused ladder step with the chain op mix (timing only).
    curve.chain_probe_step(aX, aZ, bX, bZ, d, modulus, np0,
                           ((b % (uint64_t)ECM_PROBE_CHAIN_W) == 0));
#elif ECM_STEP_VARIANT == 2
    // A/B: same arithmetic, explicit prep registers (production only; the add-density
    // probe above is disabled in this variant).
    curve.double_add_v2_ssa(aX, aZ, bX, bZ, d, modulus, np0);
#else
    curve.double_add_v2(aX, aZ, bX, bZ, d, modulus, np0,
                        (ECM_PROBE_ADD_DENSITY <= 1) || ((b % (uint64_t)ECM_PROBE_ADD_DENSITY) == 0));
#endif
  }

  if (swapped) {
    cgbn_swap(curve._env, aX, bX);
    cgbn_swap(curve._env, aZ, bZ);
  }

  { // Final output
    // Convert everything back to bn
    cgbn_mont2bn(curve._env, aX, aX, modulus, np0);
    cgbn_mont2bn(curve._env, aZ, aZ, modulus, np0);
    cgbn_mont2bn(curve._env, bX, bX, modulus, np0);
    cgbn_mont2bn(curve._env, bZ, bZ, modulus, np0);

    {
      curve.assert_normalized(aX, modulus);
      curve.assert_normalized(aZ, modulus);
      curve.assert_normalized(bX, modulus);
      curve.assert_normalized(bZ, modulus);
    }
    cgbn_store(curve._env, &data_cast[5*instance_i+1], aX);
    cgbn_store(curve._env, &data_cast[5*instance_i+2], aZ);
    cgbn_store(curve._env, &data_cast[5*instance_i+3], bX);
    cgbn_store(curve._env, &data_cast[5*instance_i+4], bZ);
  }
}


/**
 * Suyama param0 double-and-add, index decreasing (same ladder, different curve).
 *
 * Data layout per curve -- SEVEN words, prepared entirely on the host
 * (set_p_2p_suyama):   N, a24, xdiff, aX, aZ, bX, bZ
 *
 *   a24   = (A+2)/4                       full width, the curve constant
 *   xdiff = X0/Z0                         affine x of the ladder difference point
 *   (aX,aZ) = P = (u^3 : v^3)             start point
 *   (bX,bZ) = 2P                          second ladder point
 *
 * `sigma_0` is unused here (kept only because the kernel-pointer type is shared
 * with the param3 kernel): with param0 everything sigma-dependent is already
 * baked into the seven words above.
 */
/**
 * Suyama param0 double-and-add kernel, and -- with CONST_DIFF = true -- the param2
 * ("batch 2", 6-torsion) one.  The two differ ONLY in how the ladder difference enters
 * the differential addition:
 *
 *   CONST_DIFF = false : difference is the affine x0 -> one mont_mul by xdiff   (6M+4S)
 *   CONST_DIFF = true  : difference is the constant 2 -> shift_left, free       (5M+4S)
 *
 * gmp-ecm's param 2 sets x0 = 2 (parametrizations.c:373) with a full-width a24, which is
 * exactly this combination; param0 has a full-width a24 and a genuine xdiff.  Both live in
 * the same 7-word buffer layout.  CONST_DIFF is a TEMPLATE parameter on purpose: passing
 * it as a runtime flag instead cost 34% of throughput (registers 71 -> 92, docs §5.6).
 * See docs/ECM_CGBN_OPTIMIZATION.md §5.6.
 */
template<class params, bool CONST_DIFF = false>
__global__ void __maxnreg__(params::REG_TARGET) kernel_double_add_suyama(
        cgbn_error_report_t *report,
        uint64_t s_bits,
        uint64_t s_bits_start,
        uint64_t s_bits_interval,
        uint32_t *gpu_s_bits,
        uint32_t *data,
        uint32_t count,
        uint32_t sigma_0,
        uint32_t np0
        ) {
  (void)sigma_0;
  int32_t instance_i = (blockIdx.x*blockDim.x + threadIdx.x)/params::TPI;
  if(instance_i >= count)
    return;

  typename curve_t<params>::mem_t *data_cast = (typename curve_t<params>::mem_t*) data;

  cgbn_monitor_t monitor = CHECK_ERROR ? cgbn_report_monitor : cgbn_no_checks;

  curve_t<params> curve(monitor, report, instance_i);
  typename curve_t<params>::bn_t aX, aZ, bX, bZ, a24, xdiff, modulus;

  { // Setup -- 7 words per instance
      cgbn_load(curve._env, modulus, &data_cast[7*instance_i+0]);
      cgbn_load(curve._env, a24,     &data_cast[7*instance_i+1]);
      cgbn_load(curve._env, xdiff,   &data_cast[7*instance_i+2]);
      cgbn_load(curve._env, aX,      &data_cast[7*instance_i+3]);
      cgbn_load(curve._env, aZ,      &data_cast[7*instance_i+4]);
      cgbn_load(curve._env, bX,      &data_cast[7*instance_i+5]);
      cgbn_load(curve._env, bZ,      &data_cast[7*instance_i+6]);

      /* Convert the values that participate in field multiplications to the
         Montgomery domain.  N stays as it is (it is the modulus). */
      uint32_t np0_test = cgbn_bn2mont(curve._env, aX, aX, modulus);
      assert(np0 == np0_test);
      cgbn_bn2mont(curve._env, aZ, aZ, modulus);
      cgbn_bn2mont(curve._env, bX, bX, modulus);
      cgbn_bn2mont(curve._env, bZ, bZ, modulus);
      cgbn_bn2mont(curve._env, a24, a24, modulus);
      cgbn_bn2mont(curve._env, xdiff, xdiff, modulus);
  }

  /* P_a = (aX, aZ) holds P, P_b = (bX, bZ) holds 2P */
  int swapped = 0;
  for (uint64_t b = s_bits_start; b < s_bits_start + s_bits_interval; b++) {
    uint64_t nth = s_bits - 1 - b;
    int bit = (gpu_s_bits[nth/32] >> (nth&31)) & 1;
    if (bit != swapped) {
        swapped = !swapped;
        cgbn_swap(curve._env, aX, bX);
        cgbn_swap(curve._env, aZ, bZ);
    }
#if ECM_PARAM2SHAPE
    // PROBE: force the *param2-shaped* step (full-width a24 + constant difference 2) on
    // the param0 data path.  Timing only -- docs/ECM_CGBN_OPTIMIZATION.md §5.6.
    curve.double_add_v2_suyama(aX, aZ, bX, bZ, a24, xdiff, modulus, np0, true);
#else
    /* NOTE (2026-09-25): do NOT try to pick the shift/multiply with a runtime flag here.
       A warp-uniform `xdiff == 2` test looked free, but keeping xdiff live across the bit
       loop pushed this kernel from 71 to 92 registers and cost 34% of throughput (the
       param0 path fell from 71 M to 47 M curve-bits/s, docs §5.6).  param2 therefore gets
       its OWN kernel family (same instantiations, const_diff = true) and the dispatch
       chooses the family -- like param0's family already is.  See §8 item 5. */
    curve.double_add_v2_suyama(aX, aZ, bX, bZ, a24, xdiff, modulus, np0, CONST_DIFF);
#endif
  }

  if (swapped) {
    cgbn_swap(curve._env, aX, bX);
    cgbn_swap(curve._env, aZ, bZ);
  }

  { // Final output -- points go back to plain form, at the same 7-word stride
    cgbn_mont2bn(curve._env, aX, aX, modulus, np0);
    cgbn_mont2bn(curve._env, aZ, aZ, modulus, np0);
    cgbn_mont2bn(curve._env, bX, bX, modulus, np0);
    cgbn_mont2bn(curve._env, bZ, bZ, modulus, np0);

    cgbn_store(curve._env, &data_cast[7*instance_i+3], aX);
    cgbn_store(curve._env, &data_cast[7*instance_i+4], aZ);
    cgbn_store(curve._env, &data_cast[7*instance_i+5], bX);
    cgbn_store(curve._env, &data_cast[7*instance_i+6], bZ);
  }
}


// ── kernel param typedefs ─────────────────────────────────────────────────
// TPI=4 (always compiled)
typedef cgbn_params_t<4, 128>   cgbn_params_128;
typedef cgbn_params_t<4, 192>   cgbn_params_192;
typedef cgbn_params_t<4, 256>   cgbn_params_256;
typedef cgbn_params_t<4, 384>   cgbn_params_384;
typedef cgbn_params_t<4, 512>   cgbn_params_small;

// TPI=8 (768/1024 in dev build; 1280..2048 full build only)
typedef cgbn_params_t<8, 768>   cgbn_params_768;
typedef cgbn_params_t<8, 1024>  cgbn_params_medium;
typedef cgbn_params_t<8, 1280>  cgbn_params_1280;
typedef cgbn_params_t<8, 1536>  cgbn_params_1536;
typedef cgbn_params_t<8, 1792>  cgbn_params_1792;
typedef cgbn_params_t<8, 2048>  cgbn_params_2048;

// TPI=16 (512 interval; full build only)
typedef cgbn_params_t<16, 2560> cgbn_params_2560;
typedef cgbn_params_t<16, 2816> cgbn_params_2816;
typedef cgbn_params_t<16, 3072> cgbn_params_3072;
typedef cgbn_params_t<16, 3328> cgbn_params_3328;
typedef cgbn_params_t<16, 3584> cgbn_params_3584;
typedef cgbn_params_t<16, 3840> cgbn_params_3840;
typedef cgbn_params_t<16, 4096> cgbn_params_4096;
typedef cgbn_params_t<16, 4352> cgbn_params_4352;
typedef cgbn_params_t<16, 4608> cgbn_params_4608;
typedef cgbn_params_t<16, 4864> cgbn_params_4864;
typedef cgbn_params_t<16, 5120> cgbn_params_5120;
typedef cgbn_params_t<16, 5376> cgbn_params_5376;
typedef cgbn_params_t<16, 5632> cgbn_params_5632;
typedef cgbn_params_t<16, 5888> cgbn_params_5888;
typedef cgbn_params_t<16, 6144> cgbn_params_6144;
typedef cgbn_params_t<16, 6400> cgbn_params_6400;
typedef cgbn_params_t<16, 6656> cgbn_params_6656;
typedef cgbn_params_t<16, 6912> cgbn_params_6912;
typedef cgbn_params_t<16, 7168> cgbn_params_7168;
typedef cgbn_params_t<16, 7424> cgbn_params_7424;
typedef cgbn_params_t<16, 7680> cgbn_params_7680;
typedef cgbn_params_t<16, 7936> cgbn_params_7936;
typedef cgbn_params_t<16, 8192> cgbn_params_8192;

// TPI=32 (full build only)
typedef cgbn_params_t<32, 9216>  cgbn_params_9216;
typedef cgbn_params_t<32, 10240> cgbn_params_10240;
typedef cgbn_params_t<32, 11264> cgbn_params_11264;
typedef cgbn_params_t<32, 12288> cgbn_params_12288;
typedef cgbn_params_t<32, 13312> cgbn_params_13312;
typedef cgbn_params_t<32, 14336> cgbn_params_14336;
typedef cgbn_params_t<32, 15360> cgbn_params_15360;
typedef cgbn_params_t<32, 16384> cgbn_params_16384;


// ── per-TPI dispatch seam ─────────────────────────────────────────────────
// Each function maps a kernel BITS to the instantiated __global__ function
// pointer (and the matching TPI), or returns nullptr when BITS is not in that
// TPI group. Implemented in the per-TPI kernel translation units.
typedef void (*cgbn_stage1_kernel_fn)(cgbn_error_report_t *, uint64_t, uint64_t,
                                      uint64_t, uint32_t*, uint32_t*, uint32_t,
                                      uint32_t, uint32_t);

cgbn_stage1_kernel_fn cgbn_stage1_kernel_tpi4(uint32_t BITS, uint32_t *TPI_out);
cgbn_stage1_kernel_fn cgbn_stage1_kernel_tpi8(uint32_t BITS, uint32_t *TPI_out);
cgbn_stage1_kernel_fn cgbn_stage1_kernel_tpi16(uint32_t BITS, uint32_t *TPI_out);
cgbn_stage1_kernel_fn cgbn_stage1_kernel_tpi32(uint32_t BITS, uint32_t *TPI_out);

/* Suyama param0 variants (method = gpu + gpu_param = 0).  Kept in their own TU so
   the extra template instantiations do not slow down the param3 kernel builds; the
   instantiated grid mirrors the param3 one exactly (see the answer to "can the two
   share instantiations?" in docs/ECM_Montgomery_STAGE1.md §21). */
cgbn_stage1_kernel_fn cgbn_stage1_kernel_suyama_tpi4(uint32_t BITS, uint32_t *TPI_out);
cgbn_stage1_kernel_fn cgbn_stage1_kernel_suyama_tpi8(uint32_t BITS, uint32_t *TPI_out);
cgbn_stage1_kernel_fn cgbn_stage1_kernel_suyama_tpi16(uint32_t BITS, uint32_t *TPI_out);
cgbn_stage1_kernel_fn cgbn_stage1_kernel_suyama_tpi32(uint32_t BITS, uint32_t *TPI_out);

/* param2 ("batch 2", 6-torsion, method = gpu + gpu_param = 2) variants: the same kernel
   body instantiated with CONST_DIFF = true, so the differential addition drops the xdiff
   multiply (5M+4S vs param0's 6M+4S).  Its own TU for the same reason as the Suyama set,
   and its own family rather than a runtime flag (that cost 34%, docs §5.6). */
cgbn_stage1_kernel_fn cgbn_stage1_kernel_param2_tpi4(uint32_t BITS, uint32_t *TPI_out);
cgbn_stage1_kernel_fn cgbn_stage1_kernel_param2_tpi8(uint32_t BITS, uint32_t *TPI_out);
cgbn_stage1_kernel_fn cgbn_stage1_kernel_param2_tpi16(uint32_t BITS, uint32_t *TPI_out);
cgbn_stage1_kernel_fn cgbn_stage1_kernel_param2_tpi32(uint32_t BITS, uint32_t *TPI_out);

#endif  /* _CGBN_STAGE1_KERNEL_H */
