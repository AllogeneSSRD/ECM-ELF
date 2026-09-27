#ifndef _CGBN_MERS_FUSED_CORE_H
#define _CGBN_MERS_FUSED_CORE_H 1

/* cgbn_mers_fused_core.h -- hand-written MERSENNE modular multiplication for the fold
 * domain (N = 2^k - 1), goal round 1, see docs/ECM_CGBN_OPTIMIZATION.md 9.10.
 *
 * WHAT IT IS
 *   CGBN's product accumulation with the fold applied to its OWN internal accumulators.
 *   The accumulation loops are copied verbatim from
 *     cgbn/include/cgbn/core/core_mul_wmad.cu :: core_t<env>::mul_wide (lines 142-320)
 *   (names qualified; that routine's `add[]` input is zero in every CGBN caller of
 *   cgbn_mul_wide, hence mpzero(ra) here).  Only the EPILOGUE differs.
 *
 * WHY
 *   The two-pass fold (cgbn_mul_wide + fold the halves with cgbn_* operators) keeps
 *   low + high + temp = 3*LIMBS words alive on top of the ladder state, and the caller's
 *   cgbn_wide_t blocks ptxas from interleaving two independent modmuls.  Measured
 *   consequence: 356 B spill stores / 652 B spill loads at the 128-register cap (vs
 *   144/172 for the Montgomery kernel) and ncu showing L2 Active Cycles +641%.
 *   The fused epilogue never materializes the 2*BITS-bit product:
 *       m = (rl mod 2^k) + (rl >> k) + (ra << t)  ==  a*b  (mod 2^k - 1)
 *   with ra already carrying the low half's carry into word 0 (that is what mul_wide's
 *   final fast_propagate_add does), and t < k bounds m <= 2.5N - 0.5 so two conditional
 *   subtractions canonicalise it.
 *
 * MEASURED (tools/bench/cgbn_mers_fold_probe.cu, 4060, 2048 instances, ns/op)
 *   parallel chains   tier 4608: fold_gen / fold_fused    tier 5120: fold_gen / fold_fused
 *        1            28.06 / 28.52  (-1.6%)                35.23 / 33.95  (+3.8%)
 *        2            28.87 / 29.54  (-2.3%)                37.86 / 35.91  (+5.2%)
 *        4            35.70 / 29.91  (+16.3%)               40.75 / 40.70  (+0.1%)
 *   i.e. flat in the number of parallel chains while the two-pass fold degrades.
 *   Correctness: the probe checks a 1000-step chain and an 18-value edge battery
 *   (0/1/2/n-1/n-2/2^(k-1)+-1/R/all-ones words ...) against GMP on every tier.
 *
 * NOTE the guard: cgbn.h pulls in cgbn_cuda.h (and therefore cgbn::core_t) only when
 * __CUDA_ARCH__ is defined -- in the host pass it includes cgbn_mpz.h, where
 * cgbn::core_t does not exist at all.  The whole file is therefore device-only.
 */

#if defined(__CUDA_ARCH__)

namespace mers_core {

template<class env_t>
__device__ __forceinline__ void fused_mul(env_t env,
                                          typename env_t::cgbn_t &r_out,
                                          const typename env_t::cgbn_t &a,
                                          const typename env_t::cgbn_t &b,
                                          const typename env_t::cgbn_t &n,
                                          const uint32_t k, const uint32_t t) {
  typedef cgbn::core_t<env_t> core;
  const uint32_t LIMBS = env_t::LIMBS;
  const uint32_t TPI = env_t::TPI;
  const uint32_t BITS = env_t::BITS;
  const uint32_t PADDING = env_t::PADDING;

  uint32_t sync = core::sync_mask(), group_thread = threadIdx.x & (TPI - 1);
  uint32_t tmp, t0, t1, term0, term1, carry;
  uint32_t rl[LIMBS], ra[LIMBS + 2], ru[LIMBS + 1];
  int32_t threads = (PADDING != 0) ? (BITS / 32) / LIMBS & 0xFFFE : (int32_t)TPI;

  if (PADDING != 0) cgbn::mpzero<LIMBS>(rl);
  cgbn::mpzero<LIMBS>(ra);
  ra[LIMBS] = 0; ra[LIMBS + 1] = 0;
  cgbn::mpzero<LIMBS>(ru);
  ru[LIMBS] = 0;

  carry = 0;
  #pragma nounroll
  for (int32_t row = 0; row < threads; row += 2) {
    #pragma unroll
    for (int32_t l = 0; l < (int32_t)(LIMBS * 2); l += 2) {
      if (l < (int32_t)LIMBS) term0 = __shfl_sync(sync, b._limbs[l], row, TPI);
      else                    term0 = __shfl_sync(sync, b._limbs[l - LIMBS], row + 1, TPI);
      if (l + 1 < (int32_t)LIMBS) term1 = __shfl_sync(sync, b._limbs[l + 1], row, TPI);
      else                        term1 = __shfl_sync(sync, b._limbs[l + 1 - LIMBS], row + 1, TPI);

      cgbn::chain_t<> chain1;                     /* aligned:   T0 * A_even */
      #pragma unroll
      for (int32_t index = 0; index < (int32_t)LIMBS; index += 2) {
        ra[index] = chain1.madlo(a._limbs[index], term0, ra[index]);
        ra[index + 1] = chain1.madhi(a._limbs[index], term0, ra[index + 1]);
      }
      if (LIMBS % 2 == 0) ra[LIMBS] = chain1.add(ra[LIMBS], 0);

      cgbn::chain_t<> chain2;                     /* unaligned: T0 * A_odd */
      t0 = chain2.add(ra[0], carry);
      #pragma unroll
      for (int32_t index = 0; index < (int32_t)(LIMBS - 1); index += 2) {
        ru[index] = chain2.madlo(a._limbs[index + 1], term0, ru[index]);
        ru[index + 1] = chain2.madhi(a._limbs[index + 1], term0, ru[index + 1]);
      }
      if (LIMBS % 2 == 1) ru[LIMBS - 1] = chain2.add(0, 0);

      cgbn::chain_t<> chain3;                     /* unaligned: T1 * A_even */
      t1 = chain3.madlo(a._limbs[0], term1, ru[0]);
      carry = chain3.madhi(a._limbs[0], term1, ru[1]);
      #pragma unroll
      for (int32_t index = 0; index < (int32_t)(LIMBS - 2); index += 2) {
        ru[index] = chain3.madlo(a._limbs[index + 2], term1, ru[index + 2]);
        ru[index + 1] = chain3.madhi(a._limbs[index + 2], term1, ru[index + 3]);
      }
      if (LIMBS % 2 == 1) ru[LIMBS - 1] = 0;
      else                ru[LIMBS - 2] = chain3.add(0, 0);
      ru[LIMBS - 1 + LIMBS % 2] = 0;

      cgbn::chain_t<> chain4;                     /* aligned:   T1 * A_odd */
      t1 = chain4.add(t1, ra[1]);
      #pragma unroll
      for (int32_t index = 0; index < (int32_t)(LIMBS - 3); index += 2) {
        ra[index] = chain4.madlo(a._limbs[index + 1], term1, ra[index + 2]);
        ra[index + 1] = chain4.madhi(a._limbs[index + 1], term1, ra[index + 3]);
      }
      ra[LIMBS - 2 - LIMBS % 2] = chain4.madlo(a._limbs[LIMBS - 1 - LIMBS % 2], term1, ra[LIMBS - LIMBS % 2]);
      ra[LIMBS - 1 - LIMBS % 2] = chain4.madhi(a._limbs[LIMBS - 1 - LIMBS % 2], term1, ra[LIMBS + 1 - LIMBS % 2]);
      if (LIMBS % 2 == 1) ra[LIMBS - 1] = chain4.add(0, 0);

      if (l < (int32_t)LIMBS) {
        tmp = __shfl_sync(sync, t0, 0, TPI);
        if (group_thread == (uint32_t)row) rl[l] = tmp;
      } else {
        tmp = __shfl_sync(sync, t0, 0, TPI);
        if (group_thread == (uint32_t)(row + 1)) rl[l - LIMBS] = tmp;
      }
      if (l + 1 < (int32_t)LIMBS) {
        tmp = __shfl_sync(sync, t1, 0, TPI);
        if (group_thread == (uint32_t)row) rl[l + 1] = tmp;
      } else {
        tmp = __shfl_sync(sync, t1, 0, TPI);
        if (group_thread == (uint32_t)(row + 1)) rl[l - LIMBS + 1] = tmp;
      }
      t0 = __shfl_sync(sync, t0, threadIdx.x + 1, TPI);
      t1 = __shfl_sync(sync, t1, threadIdx.x + 1, TPI);

      ra[LIMBS] = 0;
      if (group_thread != TPI - 1) {
        ra[LIMBS - 2] = cgbn::add_cc(ra[LIMBS - 2], t0);
        ra[LIMBS - 1] = cgbn::addc_cc(ra[LIMBS - 1], t1);
        ra[LIMBS] = cgbn::addc(0, 0);
      }
    }
  }

  cgbn::chain_t<> chainXX;
  ra[0] = chainXX.add(ra[0], carry);
  #pragma unroll
  for (int32_t index = 1; index < (int32_t)LIMBS; index++)
    ra[index] = chainXX.add(ra[index], ru[index - 1]);
  carry = chainXX.add(ra[LIMBS], 0);

  /* imad-algorithm tails: dead for every instantiated tier (PADDING == 0 and
     BITS/32 == TPI*LIMBS) but kept so the copy stays faithful to mul_wide */
  if (BITS / 32 >= (uint32_t)(threads * LIMBS + LIMBS)) {
    #pragma unroll
    for (uint32_t l = 0; l < LIMBS; l++) {
      tmp = __shfl_sync(sync, b._limbs[l], threads, TPI);
      cgbn::chain_t<> c3;
      #pragma unroll
      for (uint32_t index = 0; index < LIMBS; index++)
        ra[index] = c3.madlo(a._limbs[index], tmp, ra[index]);
      carry = c3.add(carry, 0);
      uint32_t s0 = __shfl_sync(sync, ra[0], 0, TPI);
      if (group_thread == (uint32_t)threads) rl[l] = s0;
      uint32_t s1 = __shfl_down_sync(sync, ra[0], 1, TPI);
      s1 = (group_thread == TPI - 1) ? 0 : s1;
      cgbn::chain_t<> c4;
      #pragma unroll
      for (uint32_t index = 0; index < LIMBS - 1; index++)
        ra[index] = c4.madhi(a._limbs[index], tmp, ra[index + 1]);
      ra[LIMBS - 1] = c4.madhi(a._limbs[LIMBS - 1], tmp, carry);
      carry = c4.add(0, 0);
      ra[LIMBS - 1] = cgbn::add_cc(ra[LIMBS - 1], s1);
      carry = cgbn::addc(carry, 0);
    }
  }
  if ((BITS / 32) % LIMBS != 0) {
    uint32_t r2 = threads + (BITS / 32 >= (uint32_t)(threads * LIMBS + LIMBS) ? 1u : 0u);
    #pragma unroll
    for (uint32_t l = 0; l < (BITS / 32) % LIMBS; l++) {
      tmp = __shfl_sync(sync, b._limbs[l], r2, TPI);
      cgbn::chain_t<> c3;
      #pragma unroll
      for (uint32_t index = 0; index < LIMBS; index++)
        ra[index] = c3.madlo(a._limbs[index], tmp, ra[index]);
      carry = c3.add(carry, 0);
      uint32_t s0 = __shfl_sync(sync, ra[0], 0, TPI);
      if (group_thread == r2) rl[l] = s0;
      uint32_t s1 = __shfl_down_sync(sync, ra[0], 1, TPI);
      s1 = (group_thread == TPI - 1) ? 0 : s1;
      cgbn::chain_t<> c4;
      #pragma unroll
      for (uint32_t index = 0; index < LIMBS - 1; index++)
        ra[index] = c4.madhi(a._limbs[index], tmp, ra[index + 1]);
      ra[LIMBS - 1] = c4.madhi(a._limbs[LIMBS - 1], tmp, carry);
      carry = c4.add(0, 0);
      ra[LIMBS - 1] = cgbn::add_cc(ra[LIMBS - 1], s1);
      carry = cgbn::addc(carry, 0);
    }
  }

  /* ---- FUSED EPILOGUE (the only difference from mul_wide) ---- */
  core::fast_propagate_add(carry, ra);
  uint32_t m[LIMBS];
  core::bitwise_mask_and(m, rl, (int32_t)k);      /* m  = rl mod 2^k       */
  core::shift_right(rl, rl, k);                   /* rl = rl >> k  (< 2^t) */
  core::shift_left(ra, ra, t);                    /* ra = ra << t  (< 2^k) */
  core::add(m, m, rl);
  core::add(m, m, ra);
  if (core::compare(m, n._limbs) >= 0) core::sub(m, m, n._limbs);
  if (core::compare(m, n._limbs) >= 0) core::sub(m, m, n._limbs);
  cgbn::mpset<LIMBS>(r_out._limbs, m);
}

} /* namespace mers_core */

#endif /* __CUDA_ARCH__ */

#endif /* _CGBN_MERS_FUSED_CORE_H */
