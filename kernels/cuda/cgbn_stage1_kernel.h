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

// ---------------------------------------------------------------------------
// MERSENNE FOLD DOMAIN (probe, 2026-09-25: docs/ECM_CGBN_OPTIMIZATION.md §9)
//
// For N = 2^k - 1 the Montgomery reduction of every modular multiply can be
// replaced by a fold: 2^k == 1, so for the 2*k-bit product P = hi*2^k + lo we
// have P == hi + lo (mod N) and no Q*N chain has to be evaluated.  CGBN's
// mont_mul spends half its madds on that reduction (core_mont_wmad.cu chains
// 5-8), so the fold is the GPU analogue of the CPU IFMA fold domain
// (src/cpu/simd_mont_ifma.cpp, ifma_mersenne_mul: "madds/模乘减半").
//
// Measured per-modmul cost on the 4060 Laptop (GPU 1), tools/bench/cgbn_mers_fold_probe.cu,
// 4096 instances, dependency chain, 2 passes (ns/op, second pass; both passes agree):
//
//   tier (TPI,BITS)  mont_mul  fold_gen   ratio      verdict
//    4,  512           0.77      2.02     2.62   LOSS
//    8, 1024           2.36      3.06     1.30   LOSS
//    8, 2048           7.53      7.91     1.05   tie
//   16, 3072          17.31     17.22     0.995  tie
//   16, 4096          29.20     27.29     0.935  win
//   16, 4608          35.55     28.06     0.789  WIN
//   16, 5120          44.45     35.22     0.792  WIN
//   16, 8192         106.81     93.57     0.876  win
//
// (`fold_gen` is the general-k fold the kernel needs; for k == BITS exactly, the
// fold degenerates to ones'-complement addition and measured 0.687..0.815.)
//
// This macro switches the WHOLE suyama kernel family (param0 and param2) to the
// fold domain, which is only valid for a Mersenne N.  A fold build therefore
// REFUSES to run anything else (kernels/cuda/cgbn_stage1.cu checks N and errors
// out), and `--gpu-param 3` is rejected as well.  When ECM_MERS_FOLD = 0 (default)
// the generated code is bit-identical to the historical Montgomery kernel.
//
// The fold needs the shift t = BITS - k at runtime; it is smuggled through the
// kernel's `sigma_0` argument (unused by the suyama family) so the shared
// kernel-pointer typedef does not have to change.
// ---------------------------------------------------------------------------
#ifndef ECM_MERS_FOLD
#define ECM_MERS_FOLD 0
#endif

// ---------------------------------------------------------------------------
// PROBE ONLY -- do not enable in production: timing only, RESULTS ARE WRONG.
//
// ECM_MERS_FOLD_PROBE_ALIGN = 1 prices the fold WITHOUT its runtime shift/mask, i.e.
// as if the modulus were 2^BITS - 1 instead of 2^k - 1 (it folds at the wrong bit
// position, so every curve is wrong; only the operator mix is real).  It answers the
// one question left after the end-to-end A/B:
//
//   * aligned fold faster than mont_mul  -> the runtime shift (t up to 255 bits) and
//     the runtime mask are what kill the general fold, and a hand-written word-level
//     fold (lane rotation + one sub-word shift per word) is worth writing;
//   * still not faster -> the fold's serial dependency chain, replacing mont_mul's
//     eight independent madd chains, is the problem and the direction is closed.
//
// See docs/ECM_CGBN_OPTIMIZATION.md 9.6/9.7.
// ---------------------------------------------------------------------------
#ifndef ECM_MERS_FOLD_PROBE_ALIGN
#define ECM_MERS_FOLD_PROBE_ALIGN 0
#endif

// ---------------------------------------------------------------------------
// ECM_SBITS_CACHE = 1 caches the stage-1 exponent word in a register instead of
// re-loading gpu_s_bits[nth/32] (plus its address arithmetic) on EVERY bit.
//
// The bit loop walks the exponent MSB-first, so one word stays live for 32
// consecutive iterations; the historical form issues a *dependent* global load per
// bit and the warp waits on it before it can decide the ladder swap.  That load is
// a much bigger share of the time when a kernel has few other instructions to hide
// it with -- which is exactly the Mersenne-fold kernel's situation (ncu, user-run:
// Compute (SM) Throughput -19.7% vs the Montgomery kernel, i.e. idle issue slots
// rather than a saturated machine).  Default 0 keeps the historical path
// byte-identical; the A/B is in docs/ECM_CGBN_OPTIMIZATION.md 9.9.
// ---------------------------------------------------------------------------
#ifndef ECM_SBITS_CACHE
#define ECM_SBITS_CACHE 0
#endif

// ---------------------------------------------------------------------------
// ECM_MERS_FOLD_FUSED = 1 (goal round 1, docs 9.10) replaces the fold's
// "cgbn_mul_wide(...) then fold the two halves" with a FUSED core: CGBN's product
// accumulation with the fold applied to its own internal accumulators (rl/ra/carry),
// so the 2*BITS-bit product never becomes a live object in the caller.
//
// Why: the fold kernel spills (356 B stores / 652 B loads at the 128-register cap vs
// 144/172 for the Montgomery kernel) because it must keep low+high+temp (3*LIMBS words)
// alive on top of the ladder state, and keeping the caller's wide pair live also stops
// ptxas from interleaving two independent modmuls.  Probe (tools/bench/cgbn_mers_fold_probe.cu,
// 4060, 2048 instances, per-op ns; fold_gen = the two-pass production fold): 
//
//   parallel chains   tier 4608: gen / fused        tier 5120: gen / fused
//        1            28.06 / 28.52  (-1.6%)          35.23 / 33.95  (+3.8%)
//        2            28.87 / 29.54  (-2.3%)          37.86 / 35.91  (+5.2%)
//        4            35.70 / 29.91  (+16.3%)         40.75 / 40.70  (+0.1%)
//
// i.e. the fused core is FLAT across chain counts while the two-pass fold degrades --
// the register-pressure explanation seen from the throughput side.  Correctness: the
// accumulation loops are copied verbatim from core_mul_wmad.cu::mul_wide (only the
// epilogue differs), and the probe verifies the 1000-step chain and the 18-value edge
// battery of every operator against GMP.
//
// Default 0: the two-pass fold stays the reference until the kernel A/B confirms this.
// ---------------------------------------------------------------------------
#ifndef ECM_MERS_FOLD_FUSED
#define ECM_MERS_FOLD_FUSED 0
#endif

#if ECM_MERS_FOLD_FUSED
#include "cgbn_mers_fused_core.h"
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

  // CGBN's WMAD core subtracts N on radix overflow, not on r >= N.
  // Canonical operands give REDC(a*b) < 2*N; one subtraction restores the
  // [0,N) invariant required by our additions and borrow/add-back subtractions.
  // In-place destinations are supported by CGBN and by these wrappers.
  __device__ FORCE_INLINE void mont_mul_normalized(
          bn_t &r, const bn_t &a, const bn_t &b,
          const bn_t &modulus, uint32_t np0) {
      cgbn_mont_mul(_env, r, a, b, modulus, np0);
      normalize_addition(r, modulus);
  }

  __device__ FORCE_INLINE void mont_sqr_normalized(
          bn_t &r, const bn_t &a, const bn_t &modulus, uint32_t np0) {
      cgbn_mont_sqr(_env, r, a, modulus, np0);
      normalize_addition(r, modulus);
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

    // Keep Montgomery products canonical before reusing them in the ladder.
    // Removing these subtractions produced wrong Q at M4423/B1=4 (scalar 12);
    // see docs/DEV_GPUOWL_NTT_NOTES.md section 45.
    if (do_add) {                                  // PROBE: CB/DA exist only for the addition
      mont_mul_normalized(CB, t, u, modulus, np0); // C*B
      mont_mul_normalized(DA, v, w, modulus, np0); // D*A
    }

    /* Roughly 40% of time is spent in these two calls */
    mont_sqr_normalized(AA, w, modulus, np0);    // AA
    mont_sqr_normalized(BB, u, modulus, np0);    // BB
    if (VERIFY_NORMALIZED) {
        assert_normalized(CB, modulus);
        assert_normalized(DA, modulus);
        assert_normalized(AA, modulus);
        assert_normalized(BB, modulus);
    }

    // q = aX is finalized
    mont_mul_normalized(q, AA, BB, modulus, np0); // AA*BB
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
    mont_mul_normalized(u, K, u, modulus, np0); // K(BB+dK)
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
      mont_sqr_normalized(w, w, modulus, np0); // (DA+CB)^2 mod N
          assert_normalized(w, modulus);

      mont_sqr_normalized(v, v, modulus, np0); // (DA-CB)^2 mod N
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
    mont_mul_normalized(CB, bz, ax, modulus, np0);
    mont_mul_normalized(DA, bx, az, modulus, np0);

    // ---- doubling half ----
    mont_sqr_normalized(AA, az, modulus, np0);
    mont_sqr_normalized(BB, ax, modulus, np0);
    mont_mul_normalized(q, AA, BB, modulus, np0);

    if (cgbn_sub(_env, K, AA, BB))
        cgbn_add(_env, K, K, modulus);

    cgbn_set(_env, dK, K);
    special_mult_ui32(dK, d, modulus, np0);

    cgbn_add(_env, t, BB, dK);
    normalize_addition(t, modulus);
    mont_mul_normalized(u, K, t, modulus, np0);

    // ---- addition half tail ----
    cgbn_add(_env, w, DA, CB);
    normalize_addition(w, modulus);
    if (cgbn_sub(_env, v, DA, CB))
        cgbn_add(_env, v, v, modulus);

    mont_sqr_normalized(w, w, modulus, np0);
    mont_sqr_normalized(v, v, modulus, np0);
    cgbn_shift_left(_env, v, v, 1);
    normalize_addition(v, modulus);
  }

  /* -------------------------------------------------------------------------
   * MERSENNE FOLD DOMAIN (ECM_MERS_FOLD builds only -- see the macro's comment).
   *
   * N = 2^k - 1, t = BITS - k >= 1 (the tier always carries CARRY_BITS of headroom
   * above n_log2, so t >= 6 in practice; t == 0 would need a shift by the whole
   * word width and is rejected by the host).
   *
   *   P = a*b (2*BITS bits) = high*2^BITS + low        a, b < 2^k
   *   P >> k = high*2^t + (low >> k)                   (k <= BITS)
   *   P mod 2^k = low mod 2^k
   *   fold = (P mod 2^k) + (P >> k)  < 3*2^k           -> fold again
   *                                   < 2^k + 3       -> two conditional subtracts
   *
   * Verified against GMP in tools/bench/cgbn_mers_fold_probe.cu: an 18-value edge
   * battery (0, 1, 2, n-1, n-2, 2^(k-1)+-1, R, ...) x itself, 324 products, plus a
   * 1000-step chain A*B^1000, on every tier -- no mismatches.
   *
   * Note that everything the kernel stores/loads stays in the PLAIN domain: the
   * bn2mont/mont2bn conversions disappear and no np0 is needed.  Both domains are
   * exact modular arithmetic, so a fold run and a Montgomery run produce the SAME
   * stage-1 X (that is the A/B acceptance test).
   * ------------------------------------------------------------------------- */
  __device__ FORCE_INLINE void fold_mul(bn_t &r,
                                        const bn_t &a, const bn_t &b,
                                        const bn_t &modulus,
                                        const uint32_t k, const uint32_t t) {
    /* ONE fold + TWO conditional subtractions, not two folds + one subtraction:
         m = (low mod 2^k) + (low >> k) + (high << t)     ==  P  (mod 2^k - 1)
       Each term is bounded (a, b < 2^k => low < 2^(k+t), high < 2^(k-t)):
         low mod 2^k <= N,  high << t <= N,  low >> k <= 2^t - 1 <= (N-1)/2   [t < k]
       => m <= 2.5N - 0.5, so TWO conditional subtractions canonicalise it (< N).
       The second fold (mask + shift + add) the first version ran instead is therefore
       redundant: it was 3 ops and 2 serial steps per modmul, on the fold's critical
       path.  t < k is enforced on the host (BITS < 2k).
       NOTE the separate result register: several call sites alias the destination (r)
       with an input (u = K*u, x = x*xdiff), and the first draft's in-place reuse of the
       wide pair would clobber those inputs. */
#if ECM_MERS_FOLD_FUSED && defined(__CUDA_ARCH__)
    /* goal round 1 (docs 9.10): fuse the fold into CGBN's product accumulators, so no
       2*BITS-bit wide object is ever live in this caller (that is what makes the fold
       kernel spill, and what stops ptxas interleaving two independent modmuls). */
    mers_core::fused_mul(_env, r, a, b, modulus, k, t);
#else
    typename env_t::cgbn_wide_t p;
    bn_t m;
    cgbn_mul_wide(_env, p, a, b);
#if ECM_MERS_FOLD_PROBE_ALIGN
    /* PROBE ONLY (results wrong): fold at the BITS boundary -- ones' complement
       addition of the two halves, i.e. NO mask and NO runtime shift. */
    (void)k; (void)t;
    int32_t c = cgbn_add(_env, r, p._low, p._high);
    c = cgbn_add_ui32(_env, r, r, (uint32_t)c);       /* end-around carry */
    c = cgbn_add_ui32(_env, r, r, (uint32_t)c);       /* only if r was all ones */
    if (cgbn_compare(_env, r, modulus) >= 0) cgbn_sub(_env, r, r, modulus);
#else
    cgbn_bitwise_mask_and(_env, m, p._low, (int32_t)k);   /* m    = low mod 2^k     */
    cgbn_shift_right(_env, p._low, p._low, k);            /* low  = low >> k  (< 2^t) */
    cgbn_shift_left(_env, p._high, p._high, t);           /* high = high << t (< 2^k) */
    cgbn_add(_env, m, m, p._low);
    cgbn_add(_env, m, m, p._high);                        /* m <= 2.5N - 0.5 */
    if (cgbn_compare(_env, m, modulus) >= 0) cgbn_sub(_env, m, m, modulus);
    if (cgbn_compare(_env, m, modulus) >= 0) cgbn_sub(_env, m, m, modulus);
    cgbn_set(_env, r, m);
#endif
#endif
  }

  /* CGBN implements mont_sqr as mont_mul(a,a) (alpha == 1.0 today), so the fold
     square is the same call -- kept as a named function so the kernel body reads
     like the Montgomery one. */
  __device__ FORCE_INLINE void fold_sqr(bn_t &r, const bn_t &a, const bn_t &modulus,
                                        const uint32_t k, const uint32_t t) {
    fold_mul(r, a, a, modulus, k, t);
  }

  /* -------------------------------------------------------------------------
   * Suyama param0 fold-domain variant of the fused double-and-add: the SAME
   * arithmetic and the same op mix (6M+4S, or 5M+4S with const_diff) as
   * double_add_v2_suyama, with every mont_mul/mont_sqr replaced by fold_mul and
   * the Montgomery-specific normalization dropped (the fold already returns a
   * value < n, exactly like CGBN's mont_mul).
   * ------------------------------------------------------------------------- */
  __device__ FORCE_INLINE void double_add_v2_suyama_fold(
          bn_t &q, bn_t &u,
          bn_t &w, bn_t &v,
          const bn_t &a24,
          const bn_t &xdiff,
          const bn_t &modulus,
          const uint32_t k, const uint32_t t,
          const bool const_diff = false) {
    bn_t tmp, CB, DA, AA, BB, K, dK;

    cgbn_add(_env, tmp, v, w); // t = (bZ + bX)
    normalize_addition(tmp, modulus);
    if (cgbn_sub(_env, v, v, w)) // v = (bZ - bX)
        cgbn_add(_env, v, v, modulus);

    cgbn_add(_env, w, u, q); // w = (aZ + aX)
    normalize_addition(w, modulus);
    if (cgbn_sub(_env, u, u, q)) // u = (aZ - aX)
        cgbn_add(_env, u, u, modulus);

    fold_mul(CB, tmp, u, modulus, k, t); // C*B
    fold_mul(DA, v, w, modulus, k, t);   // D*A

    fold_sqr(AA, w, modulus, k, t);
    fold_sqr(BB, u, modulus, k, t);

    fold_mul(q, AA, BB, modulus, k, t);  // q = aX

    if (cgbn_sub(_env, K, AA, BB))
        cgbn_add(_env, K, K, modulus);

    fold_mul(dK, K, a24, modulus, k, t); // dK = a24*K (full width)

    cgbn_add(_env, u, BB, dK);
    normalize_addition(u, modulus);

    fold_mul(u, K, u, modulus, k, t);    // u = aZ

    cgbn_add(_env, w, DA, CB);
    normalize_addition(w, modulus);
    if (cgbn_sub(_env, v, DA, CB))
        cgbn_add(_env, v, v, modulus);

    fold_sqr(w, w, modulus, k, t);       // (DA+CB)^2
    fold_sqr(v, v, modulus, k, t);       // (DA-CB)^2

    if (const_diff) {
      /* param2: difference x = 2.  Both arithmetic domains keep v < N;
         doubling v can reach 2N, so normalize before the next ladder step. */
      cgbn_shift_left(_env, v, v, 1);
      normalize_addition(v, modulus);
    } else {
      fold_mul(v, v, xdiff, modulus, k, t);
    }
    assert_normalized(v, modulus);
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

    mont_sqr_normalized(AA, w, modulus, np0);    // AA
    mont_sqr_normalized(BB, u, modulus, np0);    // BB
    mont_mul_normalized(q, AA, BB, modulus, np0); // q = AA*BB

    if (cgbn_sub(_env, K, AA, BB)) // K = AA-BB
        cgbn_add(_env, K, K, modulus);

    cgbn_set(_env, dK, K);
    special_mult_ui32(dK, d, modulus, np0); // dK = K*d (32-bit multiply)

    cgbn_add(_env, u, BB, dK); // BB + dK
    normalize_addition(u, modulus);
    mont_mul_normalized(u, K, u, modulus, np0); // u = K(BB+dK)

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

    mont_mul_normalized(v, v, CB, modulus, np0); // (X2-Z2)(X3+Z3)
    mont_mul_normalized(t, t, DA, modulus, np0); // (X2+Z2)(X3-Z3)

    cgbn_add(_env, CB, t, v); // sum
    normalize_addition(CB, modulus);
    if (cgbn_sub(_env, DA, t, v)) // difference
        cgbn_add(_env, DA, DA, modulus);

    mont_sqr_normalized(CB, CB, modulus, np0);   // sum^2
    mont_sqr_normalized(DA, DA, modulus, np0);   // difference^2

    mont_mul_normalized(w, ZD, CB, modulus, np0); // X5 = Z1*sum^2
    mont_mul_normalized(v, XD, DA, modulus, np0); // Z5 = X1*diff^2
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

    mont_mul_normalized(CB, t, u, modulus, np0); // C*B
    mont_mul_normalized(DA, v, w, modulus, np0); // D*A

    mont_sqr_normalized(AA, w, modulus, np0);    // AA
    mont_sqr_normalized(BB, u, modulus, np0);    // BB

    // q = aX is finalized
    mont_mul_normalized(q, AA, BB, modulus, np0); // AA*BB

    if (cgbn_sub(_env, K, AA, BB)) // K = AA-BB = 4XZ
        cgbn_add(_env, K, K, modulus);

    // dK = a24 * K  (full width -- a24 is NOT a 32-bit batch parameter here)
    mont_mul_normalized(dK, K, a24, modulus, np0);
        assert_normalized(dK, modulus);

    cgbn_add(_env, u, BB, dK); // BB + a24*K
    normalize_addition(u, modulus);   // kept: BB + dK can reach 2n

    // u = aZ is finalized
    mont_mul_normalized(u, K, u, modulus, np0); // K(BB + a24*K)

    cgbn_add(_env, w, DA, CB); // DA + CB
    normalize_addition(w, modulus);   // kept: DA + CB can reach 2n
    if (cgbn_sub(_env, v, DA, CB)) // DA - CB
        cgbn_add(_env, v, v, modulus);

    // w = bX is finalized (Z_D = 1: the host normalises the difference point)
    mont_sqr_normalized(w, w, modulus, np0); // (DA+CB)^2 mod N

    mont_sqr_normalized(v, v, modulus, np0); // (DA-CB)^2 mod N

    // v = bZ is finalized: x_D * (DA-CB)^2, where x_D = X0/Z0 is the affine x of
    // the ladder difference point (param3 folds the constant 2 in here instead)
    if (const_diff) {
      // param2: x_D = 2 -- same shortcut as the batch family, no multiply at all
      cgbn_shift_left(_env, v, v, 1);
      normalize_addition(v, modulus);
    } else {
      mont_mul_normalized(v, v, xdiff, modulus, np0);
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

#if ECM_MERS_FOLD
  /* Mersenne fold build: `sigma_0` is not a sigma here, it is the fold shift
     t = BITS - k (see the ECM_MERS_FOLD comment at the top of this file).  The
     host guarantees N = 2^k - 1, k = n_log2 and 1 <= t < BITS. */
  const uint32_t fold_t = sigma_0;
  const uint32_t fold_k = (uint32_t)params::BITS - fold_t;
#endif

  { // Setup -- 7 words per instance
      cgbn_load(curve._env, modulus, &data_cast[7*instance_i+0]);
      cgbn_load(curve._env, a24,     &data_cast[7*instance_i+1]);
      cgbn_load(curve._env, xdiff,   &data_cast[7*instance_i+2]);
      cgbn_load(curve._env, aX,      &data_cast[7*instance_i+3]);
      cgbn_load(curve._env, aZ,      &data_cast[7*instance_i+4]);
      cgbn_load(curve._env, bX,      &data_cast[7*instance_i+5]);
      cgbn_load(curve._env, bZ,      &data_cast[7*instance_i+6]);

#if ECM_MERS_FOLD
      /* Fold domain: the host already computed every value mod N, so there is no
         conversion to do at all (no R, no np0). */
      curve.assert_normalized(aX, modulus);
      curve.assert_normalized(aZ, modulus);
      curve.assert_normalized(bX, modulus);
      curve.assert_normalized(bZ, modulus);
#else
      /* Convert the values that participate in field multiplications to the
         Montgomery domain.  N stays as it is (it is the modulus). */
      uint32_t np0_test = cgbn_bn2mont(curve._env, aX, aX, modulus);
      assert(np0 == np0_test);
      cgbn_bn2mont(curve._env, aZ, aZ, modulus);
      cgbn_bn2mont(curve._env, bX, bX, modulus);
      cgbn_bn2mont(curve._env, bZ, bZ, modulus);
      cgbn_bn2mont(curve._env, a24, a24, modulus);
      cgbn_bn2mont(curve._env, xdiff, xdiff, modulus);
#endif
  }

  /* P_a = (aX, aZ) holds P, P_b = (bX, bZ) holds 2P */
  int swapped = 0;
#if ECM_SBITS_CACHE
  /* ECM_SBITS_CACHE: the exponent is consumed MSB-first, so one 32-bit word serves
     32 consecutive iterations.  Keep it in a register instead of issuing a dependent
     global load (plus its `nth/32` address arithmetic) on every bit; refresh at the
     word boundary, which in this descending walk is (nth & 31) == 31. */
  const uint64_t b_end = s_bits_start + s_bits_interval;
  uint32_t s_word = 0;
  if (s_bits_start < b_end) {
      s_word = gpu_s_bits[(s_bits - 1 - s_bits_start) >> 5];
  }
#endif
  for (uint64_t b = s_bits_start; b < s_bits_start + s_bits_interval; b++) {
    uint64_t nth = s_bits - 1 - b;
#if ECM_SBITS_CACHE
    if ((nth & 31) == 31 && b != s_bits_start) {
        s_word = gpu_s_bits[nth >> 5];
    }
    int bit = (s_word >> (nth & 31)) & 1;
#else
    int bit = (gpu_s_bits[nth/32] >> (nth&31)) & 1;
#endif
    if (bit != swapped) {
        swapped = !swapped;
        cgbn_swap(curve._env, aX, bX);
        cgbn_swap(curve._env, aZ, bZ);
    }
#if ECM_PARAM2SHAPE
    // PROBE: force the *param2-shaped* step (full-width a24 + constant difference 2) on
    // the param0 data path.  Timing only -- docs/ECM_CGBN_OPTIMIZATION.md §5.6.
    curve.double_add_v2_suyama(aX, aZ, bX, bZ, a24, xdiff, modulus, np0, true);
#elif ECM_MERS_FOLD
    curve.double_add_v2_suyama_fold(aX, aZ, bX, bZ, a24, xdiff, modulus,
                                    fold_k, fold_t, CONST_DIFF);
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
#if !ECM_MERS_FOLD
    /* Fold builds are already in plain form -- there is no R to remove. */
    cgbn_mont2bn(curve._env, aX, aX, modulus, np0);
    cgbn_mont2bn(curve._env, aZ, aZ, modulus, np0);
    cgbn_mont2bn(curve._env, bX, bX, modulus, np0);
    cgbn_mont2bn(curve._env, bZ, bZ, modulus, np0);
#endif

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
