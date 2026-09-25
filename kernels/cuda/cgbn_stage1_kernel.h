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

/* TODO test how this changes gpu_throughput_test */
/* NOTE: >= 512 may not be supported for > 2048 bit kernels */
const uint32_t TPB_DEFAULT = 256;

template<uint32_t tpi, uint32_t bits>
class cgbn_params_t {
  public:
  // parameters used by the CGBN context
  static const uint32_t TPB=TPB_DEFAULT;           // Reasonable default
  static const uint32_t MAX_ROTATION=4;            // good default value
  static const uint32_t SHM_LIMIT=0;               // no shared mem available
  // MPA-OpenCl port: CONSTANT_TIME is required by CGBN's cgbn_context_t on all
  // compilers (was previously mis-guarded behind #ifndef _MSC_VER).
  static const bool     CONSTANT_TIME=false;       // not implemented

  // parameters used locally in the application
  static const uint32_t TPI=tpi;                   // threads per instance
  static const uint32_t BITS=bits;                 // instance size
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
          const uint32_t np0) {
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
    cgbn_mont_mul(_env, v, v, xdiff, modulus, np0);
        assert_normalized(v, modulus);
  }
};


/**
 * Double-and-add, index decreasing algorithm.
 */
template<class params>
__global__ void kernel_double_add(
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
    curve.double_add_v2(aX, aZ, bX, bZ, d, modulus, np0,
                        (ECM_PROBE_ADD_DENSITY <= 1) || ((b % (uint64_t)ECM_PROBE_ADD_DENSITY) == 0));
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
template<class params>
__global__ void kernel_double_add_suyama(
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
    curve.double_add_v2_suyama(aX, aZ, bX, bZ, a24, xdiff, modulus, np0);
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

#endif  /* _CGBN_STAGE1_KERNEL_H */
