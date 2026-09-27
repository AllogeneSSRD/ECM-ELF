// cgbn_mers_fold_probe.cu -- can a GPU ECM stage-1 modmul for a MERSENNE modulus
// (N = 2^k - 1) beat CGBN's Montgomery multiplication?
//
// Background (docs/ECM_CGBN_OPTIMIZATION.md, "Mersenne fold" item):
//   * the CPU IFMA stage-1 path has a Mersenne fold domain: for N = 2^k-1 the
//     Montgomery reduction is replaced by a fold (2^k == 1), which halves the
//     madds per modular multiply (src/cpu/simd_mont_ifma.cpp, ifma_mersenne_mul);
//   * CGBN's mont_mul is "full product (T*A) + full reduction (Q*N)" -- see
//     cgbn/include/cgbn/core/core_mont_wmad.cu: chains 1-4 do T*A, chains 5-8 do
//     Q*N, i.e. twice the word products of a bare product;
//   * cgbn_mul_wide (core_mul_wmad.cu) computes ONLY the T*A half.
// So the fold candidate is   r = fold(mul_wide(a,b))  and the question is purely
// quantitative: how much of mont_mul does the reduction actually cost, and how
// much does the fold cost?
//
// Fold arithmetic (derived):
//   P = a*b (2*BITS bits) = _high*2^BITS + _low,  a,b < 2^k, k <= BITS, t = BITS-k
//   P >> k = _high*2^t + (_low >> k)          P mod 2^k = _low mod 2^k
//   fold(P) = (P mod 2^k) + (P >> k)          [re-fold while >= 2^k]
// For k == BITS (t == 0) this collapses to  fold = _low + _high  with the carry
// out of bit BITS added back in (ones' complement), i.e. no shifts at all.
//
// Every op below is a dependency chain  a <- a (x) b  starting from the same
// plain-domain A < N with b = B (plain domain), so ALL of them have the SAME
// closed form  a_iters = A * B^iters mod N  and are all checkable with GMP
// (mpz_powm).  mont_mul gets b in Montgomery form and is therefore expected to
// land on the very same value -- that cross-check is the point of the harness.
//
// NOTE: keep this file ASCII-only.  nvcc reads a BOM-less UTF-8 file as ANSI
// (GBK on this box) and a multi-byte CJK character can swallow the newline,
// commenting out the next line of code.
//
// usage: cgbn_mers_fold_probe.exe [tier 0..7] [instances] [iters] [device] [ktest]
//   tier 0..7 = (4,512) (8,1024) (8,2048) (16,3072) (16,4096) (16,8192) (16,4608) (16,5120)
//   ktest     = "align" (k = BITS: the cheap fold) | "gen" (k < BITS: exact tier,
//               e.g. M4423 in the 4608 tier) | "both"

#include <gmp.h>          // must precede cgbn.h (CGBN host side requires GMP)
#include <cgbn/cgbn.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

#define CUDA_CHECK(call)                                                            \
  do {                                                                              \
    cudaError_t e_ = (call);                                                        \
    if (e_ != cudaSuccess) {                                                        \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
      exit(2);                                                                      \
    }                                                                               \
  } while (0)

template <uint32_t tpi, uint32_t bits>
struct params_t {
  static const uint32_t TPB = 128;
  static const uint32_t MAX_ROTATION = 1;
  static const uint32_t SHM_LIMIT = 0;
  static const bool CONSTANT_TIME = false;
  static const uint32_t TPI = tpi;
  static const uint32_t BITS = bits;
};

/* PROBE_ROT=1 also runs fold_rot: the rotating-accumulator transplant of CGBN's own
   `mul` placement (core_mul_wmad.cu:100-123) into mul_wide's accumulation loops.
   IT IS INCORRECT -- 265/324 edge-battery mismatches -- and therefore OFF by default;
   see docs/ECM_CGBN_OPTIMIZATION.md 9.10.5.  Reason: mul stores the chain carry-out
   under a RANGE predicate, so on lanes outside that range rl[] keeps a STALE value.
   mul tolerates that (its carries are consumed directly), but mul_wide merges the
   rotated t0/t1 into its ru[]/ra[] carry structure, which then produces a wrong
   product.  The two routines' bookkeeping is coupled to their chain architectures,
   so this is not a transplant that can be dropped in. */
#ifndef PROBE_ROT
#define PROBE_ROT 0
#endif

enum { OP_MONT_MUL, OP_MUL_WIDE, OP_REDUCE_WIDE, OP_FOLD_ALIGN, OP_FOLD_ALIGN_CANON,
       OP_FOLD_GEN, OP_FOLD_FUSED,
#if PROBE_ROT
       OP_FOLD_ROT,
#endif
       OP_COUNT };

/* How many INDEPENDENT chains run per instance.  A dependency chain measures latency;
   the real kernel has several modmuls in flight, and the earlier rounds showed the chain
   view ranks mul_wide above mont_mul while the kernel measures the opposite.  With
   PROBE_CHAINS=2 each instance runs two chains on identical values (chain 0 is verified),
   which is the cheapest way to expose throughput instead of latency -- docs 9.9(g). */
#ifndef PROBE_CHAINS
#define PROBE_CHAINS 1
#endif

#define SLOTS 4          /* per instance: a, b_plain, b_mont, n */

/* ---------------------------------------------------------------------------
 * MERSENNE FUSED CORE (goal round 1, docs 9.10): CGBN's mul_wide with the fold applied
 * to its INTERNAL accumulators.
 *
 * The product accumulation below is copied VERBATIM from
 *   cgbn/include/cgbn/core/core_mul_wmad.cu :: core_t<env>::mul_wide (lines 142-320)
 * (names qualified; the `add[]` input of that function is zero in every CGBN caller of
 * cgbn_mul_wide, hence mpzero(ra) here).  Only the EPILOGUE differs: instead of
 * publishing (lo, hi) as two cgbn_t and then folding them with cgbn_* operators, the
 * fold is applied to rl/ra/carry in place.  That removes the 2*BITS-bit wide object
 * from the caller's live set -- which is what makes this kernel spill (356 B stores /
 * 652 B loads at 128 regs vs 144/172 for the Montgomery kernel).
 *
 * Why it may be faster: fewer live registers, no mpset of lo/hi, and the fold's
 * mask/shift/add run on the same arrays the accumulation just produced.
 * Why it may not be: the accumulation loops (and their broadcast+place bookkeeping for
 * the low half) are unchanged -- that bookkeeping was ~400 instructions per modmul per
 * thread, i.e. as expensive as the ~320 madds of arithmetic (docs 9.9(f)).
 * NOTE: this block must only be compiled in the DEVICE pass -- cgbn.h pulls in cgbn_cuda.h
 * (and therefore cgbn::core_t) only when __CUDA_ARCH__ is defined; the host pass includes
 * cgbn_mpz.h instead, where cgbn::core_t does not exist at all.  Hence the guard below,
 * and the matching guard around the call site in the probe kernel.
 * ------------------------------------------------------------------------- */
#if defined(__CUDA_ARCH__)
namespace mers_core {

/* ROT = true selects CGBN's `mul`-style ROTATING low-half accumulator instead of
   mul_wide's "broadcast + predicated store" (docs 9.10).  Per word pair that trades
   2 x SHFL for 2 x MOV; SHFL issues through MIO at roughly a quarter of the ALU rate
   on Ada, so the rotating form should be the cheaper one -- measured below. */
template<bool ROT, class env_t>
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
  for (int32_t r = 0; r < threads; r += 2) {
    #pragma unroll
    for (int32_t l = 0; l < (int32_t)(LIMBS * 2); l += 2) {
      if (l < (int32_t)LIMBS) term0 = __shfl_sync(sync, b._limbs[l], r, TPI);
      else                    term0 = __shfl_sync(sync, b._limbs[l - LIMBS], r + 1, TPI);
      if (l + 1 < (int32_t)LIMBS) term1 = __shfl_sync(sync, b._limbs[l + 1], r, TPI);
      else                        term1 = __shfl_sync(sync, b._limbs[l + 1 - LIMBS], r + 1, TPI);

      cgbn::chain_t<> chain1;                       // aligned:   T0 * A_even
      #pragma unroll
      for (int32_t index = 0; index < (int32_t)LIMBS; index += 2) {
        ra[index] = chain1.madlo(a._limbs[index], term0, ra[index]);
        ra[index + 1] = chain1.madhi(a._limbs[index], term0, ra[index + 1]);
      }
      if (LIMBS % 2 == 0) ra[LIMBS] = chain1.add(ra[LIMBS], 0);

      cgbn::chain_t<> chain2;                       // unaligned: T0 * A_odd
      t0 = chain2.add(ra[0], carry);
      #pragma unroll
      for (int32_t index = 0; index < (int32_t)(LIMBS - 1); index += 2) {
        ru[index] = chain2.madlo(a._limbs[index + 1], term0, ru[index]);
        ru[index + 1] = chain2.madhi(a._limbs[index + 1], term0, ru[index + 1]);
      }
      if (LIMBS % 2 == 1) ru[LIMBS - 1] = chain2.add(0, 0);

      cgbn::chain_t<> chain3;                       // unaligned: T1 * A_even
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

      cgbn::chain_t<> chain4;                       // aligned:   T1 * A_odd
      t1 = chain4.add(t1, ra[1]);
      #pragma unroll
      for (int32_t index = 0; index < (int32_t)(LIMBS - 3); index += 2) {
        ra[index] = chain4.madlo(a._limbs[index + 1], term1, ra[index + 2]);
        ra[index + 1] = chain4.madhi(a._limbs[index + 1], term1, ra[index + 3]);
      }
      ra[LIMBS - 2 - LIMBS % 2] = chain4.madlo(a._limbs[LIMBS - 1 - LIMBS % 2], term1, ra[LIMBS - LIMBS % 2]);
      ra[LIMBS - 1 - LIMBS % 2] = chain4.madhi(a._limbs[LIMBS - 1 - LIMBS % 2], term1, ra[LIMBS + 1 - LIMBS % 2]);
      if (LIMBS % 2 == 1) ra[LIMBS - 1] = chain4.add(0, 0);

      if (ROT) {
        /* CGBN core_mul_wmad.cu :: core_t<env>::mul (lines 100-123), verbatim: store the
           carry-out under a range predicate, rotate the accumulator by one lane, then read
           the rotated value back as the chain input of the next iteration. */
        if (l < (int32_t)LIMBS) {
          if (group_thread < (uint32_t)(threads - r)) rl[l] = t0;
          rl[l] = __shfl_sync(sync, rl[l], threadIdx.x + 1, TPI);
          t0 = rl[l];
        } else {
          if (group_thread < (uint32_t)(threads - 1 - r)) rl[l - LIMBS] = t0;
          rl[l - LIMBS] = __shfl_sync(sync, rl[l - LIMBS], threadIdx.x + 1, TPI);
          t0 = rl[l - LIMBS];
        }
        if (l + 1 < (int32_t)LIMBS) {
          if (group_thread < (uint32_t)(threads - r)) rl[l + 1] = t1;
          rl[l + 1] = __shfl_sync(sync, rl[l + 1], threadIdx.x + 1, TPI);
          t1 = rl[l + 1];
        } else {
          if (group_thread < (uint32_t)(threads - 1 - r)) rl[l + 1 - LIMBS] = t1;
          rl[l + 1 - LIMBS] = __shfl_sync(sync, rl[l + 1 - LIMBS], threadIdx.x + 1, TPI);
          t1 = rl[l + 1 - LIMBS];
        }
      } else {
        if (l < (int32_t)LIMBS) {
          tmp = __shfl_sync(sync, t0, 0, TPI);
          if (group_thread == (uint32_t)r) rl[l] = tmp;
        } else {
          tmp = __shfl_sync(sync, t0, 0, TPI);
          if (group_thread == (uint32_t)(r + 1)) rl[l - LIMBS] = tmp;
        }
        if (l + 1 < (int32_t)LIMBS) {
          tmp = __shfl_sync(sync, t1, 0, TPI);
          if (group_thread == (uint32_t)r) rl[l + 1] = tmp;
        } else {
          tmp = __shfl_sync(sync, t1, 0, TPI);
          if (group_thread == (uint32_t)(r + 1)) rl[l - LIMBS + 1] = tmp;
        }
        t0 = __shfl_sync(sync, t0, threadIdx.x + 1, TPI);
        t1 = __shfl_sync(sync, t1, threadIdx.x + 1, TPI);
      }

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

  /* imad-algorithm tails (dead for our tiers: PADDING==0 and BITS/32 == TPI*LIMBS) */
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
    uint32_t r2 = threads + (BITS / 32 >= (uint32_t)(threads * LIMBS + LIMBS));
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

  /* ---- FUSED EPILOGUE (the only difference from mul_wide) ---------------------------
     mul_wide would do: mpset(lo, rl); mpset(hi, ra); fast_propagate_add(carry, hi);
     We instead fold straight out of rl/ra:
         m = (rl mod 2^k) + (rl >> k) + (ra << t)   ==  a*b  (mod 2^k - 1)
     with ra already carrying the top carry, then the same two conditional subtractions
     the production fold uses (t < k makes m <= 2.5N - 0.5). */
  core::fast_propagate_add(carry, ra);            /* carry from the low half into ra[0] */
  uint32_t m[LIMBS];
  core::bitwise_mask_and(m, rl, (int32_t)k);      /* m  = rl mod 2^k        */
  core::shift_right(rl, rl, k);                   /* rl = rl >> k  (< 2^t)  */
  core::shift_left(ra, ra, t);                    /* ra = ra << t  (< 2^k)  */
  core::add(m, m, rl);
  core::add(m, m, ra);
  if (core::compare(m, n._limbs) >= 0) core::sub(m, m, n._limbs);
  if (core::compare(m, n._limbs) >= 0) core::sub(m, m, n._limbs);
  cgbn::mpset<LIMBS>(r_out._limbs, m);
}

} /* namespace mers_core */
#endif /* __CUDA_ARCH__ */


/* k == BITS: ones' complement addition of the two halves. */
template<class env_t>
__device__ __forceinline__ void mers_fold_mul_align(env_t env,
                                                    typename env_t::cgbn_t &r,
                                                    const typename env_t::cgbn_t &a,
                                                    const typename env_t::cgbn_t &b,
                                                    const typename env_t::cgbn_t &n,
                                                    const bool canon) {
  typename env_t::cgbn_wide_t p;
  cgbn_mul_wide(env, p, a, b);
  int32_t c = cgbn_add(env, r, p._low, p._high);   /* r = low+high mod 2^BITS */
  c = cgbn_add_ui32(env, r, r, (uint32_t)c);       /* end-around carry: 2^BITS == 1 */
  c = cgbn_add_ui32(env, r, r, (uint32_t)c);       /* only fires if r was all ones */
  if (canon) {
    if (cgbn_compare(env, r, n) >= 0) cgbn_sub(env, r, r, n);   /* r == all ones -> 0 */
  }
}

/* General k <= BITS with runtime t = BITS - k. */
template<class env_t>
__device__ __forceinline__ void mers_fold_mul_gen(env_t env,
                                                  typename env_t::cgbn_t &r,
                                                  const typename env_t::cgbn_t &a,
                                                  const typename env_t::cgbn_t &b,
                                                  const typename env_t::cgbn_t &n,
                                                  const uint32_t k, const uint32_t t) {
  typename env_t::cgbn_wide_t p;
  typename env_t::cgbn_t m, s, hh;
  cgbn_mul_wide(env, p, a, b);
  cgbn_bitwise_mask_and(env, m, p._low, (int32_t)k);   /* low mod 2^k      */
  cgbn_shift_right(env, s, p._low, k);                 /* (low >> k) < 2^t */
  cgbn_shift_left(env, hh, p._high, t);                /* (high << t) < 2^k */
  cgbn_add(env, r, m, s);
  cgbn_add(env, r, r, hh);                             /* r < 3*2^k */
  cgbn_bitwise_mask_and(env, m, r, (int32_t)k);        /* second fold: handles the carry */
  cgbn_shift_right(env, s, r, k);
  cgbn_add(env, r, m, s);                              /* r < 2^k + 3 */
  if (cgbn_compare(env, r, n) >= 0) cgbn_sub(env, r, r, n);
  if (cgbn_compare(env, r, n) >= 0) cgbn_sub(env, r, r, n);
}

/* ---------------------------------------------------------------------------
 * Probe kernel.  The timed op is a dependency chain: every iteration consumes
 * the previous result, so the measurement is latency-bound like the ladder is.
 * ------------------------------------------------------------------------- */
template <class params, int OP>
__global__ void k_probe(cgbn_error_report_t *report,
                        cgbn_mem_t<params::BITS> *data,
                        uint32_t np0, uint32_t k, uint32_t t, uint32_t b_slot,
                        int iters, uint64_t *sink) {
  typedef cgbn_context_t<params::TPI, params> context_t;
  typedef cgbn_env_t<context_t, params::BITS> env_t;
  typedef typename env_t::cgbn_t bn_t;

  context_t ctx(cgbn_no_checks, report, (uint32_t)blockIdx.x);
  env_t env(ctx);

  const int32_t ipb = blockDim.x / params::TPI;
  const int32_t instance = blockIdx.x * ipb + (threadIdx.x / params::TPI);

  cgbn_mem_t<params::BITS> *slot = &data[SLOTS * instance];
  /* PROBE_CHAINS independent chains on identical values: the real kernel keeps several
     modmuls in flight, and a single dependency chain only measures latency (docs 9.9g).
     Chain 0 is the one the host verifies; the others are exact copies of it. */
  bn_t av[PROBE_CHAINS], b, n;
  cgbn_load(env, b, &slot[b_slot]);
  cgbn_load(env, n, &slot[3]);
  #pragma unroll
  for (int c = 0; c < PROBE_CHAINS; c++) cgbn_load(env, av[c], &slot[0]);

  if (OP == OP_MONT_MUL) {
    for (int i = 0; i < iters; i++)
      #pragma unroll
      for (int c = 0; c < PROBE_CHAINS; c++) env.mont_mul(av[c], av[c], b, n, np0);
  } else if (OP == OP_MUL_WIDE) {
    for (int i = 0; i < iters; i++)
      #pragma unroll
      for (int c = 0; c < PROBE_CHAINS; c++) {
        typename env_t::cgbn_wide_t p;
        cgbn_mul_wide(env, p, av[c], b);
        /* both halves must stay live, otherwise ptxas deletes the low-half stores
           and the "product price" comes out too low (that artifact was measured:
           a = p._high gave exactly fold_align's number) */
        cgbn_add(env, av[c], p._low, p._high);
      }
  } else if (OP == OP_REDUCE_WIDE) {
    for (int i = 0; i < iters; i++)
      #pragma unroll
      for (int c = 0; c < PROBE_CHAINS; c++) {
        typename env_t::cgbn_wide_t p;
        cgbn_mul_wide(env, p, av[c], b);
        env.mont_reduce_wide(av[c], p, n, np0);
      }
  } else if (OP == OP_FOLD_ALIGN) {
    for (int i = 0; i < iters; i++)
      #pragma unroll
      for (int c = 0; c < PROBE_CHAINS; c++) mers_fold_mul_align(env, av[c], av[c], b, n, false);
  } else if (OP == OP_FOLD_ALIGN_CANON) {
    for (int i = 0; i < iters; i++)
      #pragma unroll
      for (int c = 0; c < PROBE_CHAINS; c++) mers_fold_mul_align(env, av[c], av[c], b, n, true);
  } else if (OP == OP_FOLD_GEN) {
    for (int i = 0; i < iters; i++)
      #pragma unroll
      for (int c = 0; c < PROBE_CHAINS; c++) mers_fold_mul_gen(env, av[c], av[c], b, n, k, t);
#if PROBE_ROT
  } else if (OP == OP_FOLD_ROT) {
#if defined(__CUDA_ARCH__)
    for (int i = 0; i < iters; i++)
      #pragma unroll
      for (int c = 0; c < PROBE_CHAINS; c++) mers_core::fused_mul<true>(env, av[c], av[c], b, n, k, t);
#endif
#endif
  } else if (OP == OP_FOLD_FUSED) {
#if defined(__CUDA_ARCH__)
    for (int i = 0; i < iters; i++)
      #pragma unroll
      for (int c = 0; c < PROBE_CHAINS; c++) mers_core::fused_mul<false>(env, av[c], av[c], b, n, k, t);
#endif
  }

  cgbn_store(env, &slot[0], av[0]);
  uint64_t acc = cgbn_get_ui32(env, av[0]);
  if (acc == 0xdeadbeefull) sink[instance] = acc;   /* defeat DCE */
}

template <class params>
static void launch(int op, int blocks, int iters, cgbn_error_report_t *report,
                   cgbn_mem_t<params::BITS> *dev, uint32_t np0, uint32_t k,
                   uint32_t t, uint64_t *sink) {
  const uint32_t bslot = (op == OP_MONT_MUL || op == OP_REDUCE_WIDE) ? 2u : 1u;
  switch (op) {
    case OP_MONT_MUL:         k_probe<params, OP_MONT_MUL><<<blocks, params::TPB>>>(report, dev, np0, k, t, bslot, iters, sink); break;
    case OP_MUL_WIDE:         k_probe<params, OP_MUL_WIDE><<<blocks, params::TPB>>>(report, dev, np0, k, t, bslot, iters, sink); break;
    case OP_REDUCE_WIDE:      k_probe<params, OP_REDUCE_WIDE><<<blocks, params::TPB>>>(report, dev, np0, k, t, bslot, iters, sink); break;
    case OP_FOLD_ALIGN:       k_probe<params, OP_FOLD_ALIGN><<<blocks, params::TPB>>>(report, dev, np0, k, t, bslot, iters, sink); break;
    case OP_FOLD_ALIGN_CANON: k_probe<params, OP_FOLD_ALIGN_CANON><<<blocks, params::TPB>>>(report, dev, np0, k, t, bslot, iters, sink); break;
    case OP_FOLD_GEN:         k_probe<params, OP_FOLD_GEN><<<blocks, params::TPB>>>(report, dev, np0, k, t, bslot, iters, sink); break;
    case OP_FOLD_FUSED:       k_probe<params, OP_FOLD_FUSED><<<blocks, params::TPB>>>(report, dev, np0, k, t, bslot, iters, sink); break;
#if PROBE_ROT
    case OP_FOLD_ROT:         k_probe<params, OP_FOLD_ROT><<<blocks, params::TPB>>>(report, dev, np0, k, t, bslot, iters, sink); break;
#endif
  }
}

static uint32_t np0_for(uint32_t n0) {
  uint32_t inv = 1;
  for (int i = 0; i < 5; i++) inv *= 2u - n0 * inv;
  return (uint32_t)(0u - inv);
}

/* splitmix64: deterministic, so every number in this report is reproducible */
static uint64_t sm64(uint64_t *s) {
  uint64_t z = (*s += 0x9E3779B97F4A7C15ull);
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

static const char *op_name(int op) {
  switch (op) {
    case OP_MONT_MUL:         return "mont_mul";
    case OP_MUL_WIDE:         return "mul_wide";
    case OP_REDUCE_WIDE:      return "mul_wide+reduce";
    case OP_FOLD_ALIGN:       return "fold_align";
    case OP_FOLD_ALIGN_CANON: return "fold_align+canon";
    case OP_FOLD_GEN:         return "fold_gen";
    case OP_FOLD_FUSED:       return "fold_fused";
#if PROBE_ROT
    case OP_FOLD_ROT:         return "fold_rot";
#endif
  }
  return "?";
}

static void mpz_from_mem(mpz_t out, const uint32_t *limbs, int nlimbs) {
  mpz_set_ui(out, 0);
  for (int l = nlimbs - 1; l >= 0; l--) {
    mpz_mul_2exp(out, out, 32);
    mpz_add_ui(out, out, (unsigned long)limbs[l]);
  }
}

/* store mpz -> little-endian 32-bit limbs (value must be < 2^(32*nlimbs)) */
static void mpz_to_mem(uint32_t *limbs, int nlimbs, mpz_t v) {
  mpz_t t;
  mpz_init_set(t, v);
  for (int l = 0; l < nlimbs; l++) {
    limbs[l] = (uint32_t)mpz_get_ui(t);
    mpz_fdiv_q_2exp(t, t, 32);
  }
  mpz_clear(t);
}

template <class params>
static int run_case(int instances, int iters, uint32_t k, const char *tag,
                    int device, bool aligned) {
  typedef cgbn_mem_t<params::BITS> mem_t;
  const int32_t ipb = params::TPB / params::TPI;
  const int32_t blocks = (instances + ipb - 1) / ipb;
  const int32_t total = blocks * ipb;
  const uint32_t bits = params::BITS;
  const int nl = bits / 32;

  printf("== %s  N = 2^%u - 1 (%s)  instances=%d iters=%d\n",
         tag, k, aligned ? "k == BITS" : "k < BITS", total, iters);

  /* ---- host reference values ---- */
  mpz_t N, R, A, B, B_mont, got, exp;
  mpz_inits(N, R, A, B, B_mont, got, exp, NULL);
  mpz_set_ui(N, 1);
  mpz_mul_2exp(N, N, k);
  mpz_sub_ui(N, N, 1);                          /* N = 2^k - 1 */
  mpz_set_ui(R, 1);
  mpz_mul_2exp(R, R, bits);
  mpz_mod(R, R, N);

  mem_t *host = (mem_t *)malloc((size_t)total * SLOTS * sizeof(mem_t));
  mem_t *out = (mem_t *)malloc((size_t)total * SLOTS * sizeof(mem_t));
  memset(host, 0, (size_t)total * SLOTS * sizeof(mem_t));

  for (int i = 0; i < total; i++) {
    uint64_t s = 0x123456789ABCDEFull + (uint64_t)i * 0x9E3779B97F4A7C15ull;
    mpz_t a, b;
    mpz_inits(a, b, NULL);
    mpz_set_ui(a, 0);
    mpz_set_ui(b, 0);
    for (int l = nl - 1; l >= 0; l--) {
      mpz_mul_2exp(a, a, 32);
      mpz_add_ui(a, a, (unsigned long)(sm64(&s) & 0xFFFFFFFFu));
      mpz_mul_2exp(b, b, 32);
      mpz_add_ui(b, b, (unsigned long)(sm64(&s) & 0xFFFFFFFFu));
    }
    mpz_mod(a, a, N);
    mpz_mod(b, b, N);
    if (mpz_sgn(a) == 0) mpz_set_ui(a, 1);
    if (mpz_sgn(b) == 0) mpz_set_ui(b, 1);
    if (i == 0) { mpz_set(A, a); mpz_set(B, b); }

    mpz_to_mem(host[SLOTS * i + 0]._limbs, nl, a);        /* a, plain */
    mpz_to_mem(host[SLOTS * i + 1]._limbs, nl, b);        /* b, plain */
    mpz_mul(b, b, R);
    mpz_mod(b, b, N);
    mpz_to_mem(host[SLOTS * i + 2]._limbs, nl, b);        /* b, Montgomery form */
    mpz_to_mem(host[SLOTS * i + 3]._limbs, nl, N);        /* n */
    mpz_clears(a, b, NULL);
  }
  mpz_mul(B_mont, B, R);
  mpz_mod(B_mont, B_mont, N);

  const uint32_t np0 = np0_for(host[3]._limbs[0]);
  const uint32_t tshift = bits - k;

  mem_t *dev = nullptr;
  uint64_t *sink = nullptr;
  cgbn_error_report_t *report = nullptr;
  CUDA_CHECK(cudaMalloc(&dev, (size_t)total * SLOTS * sizeof(mem_t)));
  CUDA_CHECK(cudaMalloc(&sink, (size_t)total * sizeof(uint64_t)));
  CUDA_CHECK(cudaMemset(sink, 0, (size_t)total * sizeof(uint64_t)));
  CUDA_CHECK(cgbn_error_report_alloc(&report));

  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));

  int ops[OP_COUNT];
  int nops = 0;
  if (aligned) { ops[nops++] = OP_FOLD_ALIGN; ops[nops++] = OP_FOLD_ALIGN_CANON; }
  ops[nops++] = OP_MONT_MUL;
  ops[nops++] = OP_MUL_WIDE;
  ops[nops++] = OP_REDUCE_WIDE;
  /* the general fold needs shift_right(low, k); with k == BITS that is a shift by
     the whole word width, so it is only defined for the exact-tier case */
  if (tshift > 0) { ops[nops++] = OP_FOLD_GEN; ops[nops++] = OP_FOLD_FUSED;
#if PROBE_ROT
    ops[nops++] = OP_FOLD_ROT;
#endif
  }

  double ns[2][OP_COUNT];
  bool okv[OP_COUNT];
  memset(ns, 0, sizeof(ns));
  memset(okv, 0, sizeof(okv));

  for (int pass = 0; pass < 2; pass++) {
  for (int oi = 0; oi < nops; oi++) {
    /* second pass runs the list backwards: a per-op number that moves between
       the two passes is an ordering / clock artifact, not a code difference */
    const int op = ops[pass ? (nops - 1 - oi) : oi];

    CUDA_CHECK(cudaMemcpy(dev, host, (size_t)total * SLOTS * sizeof(mem_t), cudaMemcpyHostToDevice));
    launch<params>(op, blocks, 4, report, dev, np0, k, tshift, sink);   /* warmup */
    CUDA_CHECK(cudaDeviceSynchronize());
    if (cgbn_error_report_check(report)) {
      fprintf(stderr, "  CGBN error (op=%s, warmup)\n", op_name(op));
      cgbn_error_report_reset(report);
    }
    CUDA_CHECK(cudaMemcpy(dev, host, (size_t)total * SLOTS * sizeof(mem_t), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(e0));
    launch<params>(op, blocks, iters, report, dev, np0, k, tshift, sink);
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    if (cgbn_error_report_check(report)) {
      fprintf(stderr, "  CGBN error (op=%s)\n", op_name(op));
      cgbn_error_report_reset(report);
    }
    ns[pass][op] = (double)ms * 1e6 / ((double)total * (double)iters * (double)PROBE_CHAINS);

    /* verify the final state of the timed chain against GMP */
    CUDA_CHECK(cudaMemcpy(out, dev, (size_t)total * SLOTS * sizeof(mem_t), cudaMemcpyDeviceToHost));
    mpz_from_mem(got, out[0]._limbs, nl);
    mpz_mod(got, got, N);
    mpz_powm_ui(exp, B, (unsigned long)iters, N);
    mpz_mul(exp, exp, A);
    mpz_mod(exp, exp, N);
    okv[op] = true;
    if (op == OP_MUL_WIDE) {
      okv[op] = true;                              /* no closed form for this chain */
    } else if (mpz_cmp(exp, got) != 0) {
      okv[op] = false;
      gmp_fprintf(stderr, "  VERIFY FAIL (%s): expected %Zx\n                     got      %Zx\n",
                  op_name(op), exp, got);
    }
  }
  }

  const double base = ns[1][OP_MONT_MUL];
  int fails = 0;
  for (int oi = 0; oi < nops; oi++) {
    const int op = ops[oi];
    if (!okv[op] && op != OP_MUL_WIDE) fails++;
    printf("  %-18s %8.2f / %8.2f ns/op (1st/2nd pass) %8.2f Mops/s  vs mont_mul %5.3f  %s\n",
           op_name(op), ns[0][op], ns[1][op], 1e3 / ns[1][op],
           base > 0.0 ? ns[1][op] / base : 1.0,
           (op == OP_MUL_WIDE) ? "(timing only)" : (okv[op] ? "GMP OK" : "GMP FAIL"));
  }
  printf("  N=%s (k=%u, t=%u), %d failures\n",
         aligned ? "2^BITS-1" : "2^k-1", k, tshift, fails);

  cgbn_error_report_free(report);
  cudaFree(dev);
  cudaFree(sink);
  free(host);
  free(out);
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  mpz_clears(N, R, A, B, B_mont, got, exp, NULL);
  return fails;
}

/* ---------------------------------------------------------------------------
 * Edge-case battery: fold_mul must be exact on the values a real ladder feeds
 * it (0, 1, small, n-1, powers of two, all-ones words, R) -- the ones'-complement
 * end-around carry only ever fires on the "sum of halves overflows" pattern, so
 * a random-only test is not enough.  Runs the cross product of ~18 special
 * values once per op and checks EVERY instance against GMP.
 * ------------------------------------------------------------------------- */
template <class params>
static int run_edge(uint32_t k, int device, bool aligned) {
  typedef cgbn_mem_t<params::BITS> mem_t;
  const uint32_t bits = params::BITS;
  const int nl = bits / 32;
  const int32_t ipb = params::TPB / params::TPI;

  mpz_t N, R;
  mpz_inits(N, R, NULL);
  mpz_set_ui(N, 1); mpz_mul_2exp(N, N, k); mpz_sub_ui(N, N, 1);
  mpz_set_ui(R, 1); mpz_mul_2exp(R, R, bits); mpz_mod(R, R, N);

  /* ---- special values ---- */
  enum { NV = 18 };
  mpz_t v[NV];
  for (int i = 0; i < NV; i++) mpz_init(v[i]);
  mpz_set_ui(v[0], 0);
  mpz_set_ui(v[1], 1);
  mpz_set_ui(v[2], 2);
  mpz_set_ui(v[3], 3);
  mpz_set_ui(v[4], 4);
  mpz_set_ui(v[5], 0xFFFFFFFFull);
  mpz_set_ui(v[6], 1); mpz_mul_2exp(v[6], v[6], 32);
  mpz_set_ui(v[7], 1); mpz_mul_2exp(v[7], v[7], 31); mpz_sub_ui(v[7], v[7], 1);
  mpz_sub_ui(v[8], N, 1);              /* n-1 */
  mpz_sub_ui(v[9], N, 2);              /* n-2 */
  mpz_sub_ui(v[10], N, 0xFFFFFFFFull);
  mpz_set_ui(v[11], 1); mpz_mul_2exp(v[11], v[11], k - 1);          /* 2^(k-1) */
  mpz_sub_ui(v[12], v[11], 1);
  mpz_add_ui(v[13], v[11], 1);
  mpz_fdiv_q_2exp(v[14], N, 1);        /* floor(n/2) */
  mpz_fdiv_q_ui(v[15], N, 3);
  mpz_set(v[16], R);                   /* R mod N */
  mpz_sub_ui(v[17], R, 1);
  mpz_mod(v[16], v[16], N);
  mpz_mod(v[17], v[17], N);
  for (int i = 0; i < NV; i++) {
    if (mpz_cmp_ui(v[i], 0) < 0) mpz_set_ui(v[i], 0);
    mpz_mod(v[i], v[i], N);            /* keep the contract: input < n */
  }

  const int instances = NV * NV;
  const int32_t blocks = (instances + ipb - 1) / ipb;
  const int32_t total = blocks * ipb;

  mem_t *host = (mem_t *)malloc((size_t)total * SLOTS * sizeof(mem_t));
  mem_t *out = (mem_t *)malloc((size_t)total * SLOTS * sizeof(mem_t));
  memset(host, 0, (size_t)total * SLOTS * sizeof(mem_t));
  for (int i = 0; i < total; i++) {
    const int ia = (i / NV) % NV, ib = i % NV;
    mpz_to_mem(host[SLOTS * i + 0]._limbs, nl, v[ia]);
    mpz_to_mem(host[SLOTS * i + 1]._limbs, nl, v[ib]);
    mpz_t bm;
    mpz_init(bm);
    mpz_mul(bm, v[ib], R); mpz_mod(bm, bm, N);
    mpz_to_mem(host[SLOTS * i + 2]._limbs, nl, bm);
    mpz_to_mem(host[SLOTS * i + 3]._limbs, nl, N);
    mpz_clear(bm);
  }

  const uint32_t np0 = np0_for(host[3]._limbs[0]);
  const uint32_t tshift = bits - k;

  mem_t *dev = nullptr;
  uint64_t *sink = nullptr;
  cgbn_error_report_t *report = nullptr;
  CUDA_CHECK(cudaMalloc(&dev, (size_t)total * SLOTS * sizeof(mem_t)));
  CUDA_CHECK(cudaMalloc(&sink, (size_t)total * sizeof(uint64_t)));
  CUDA_CHECK(cudaMemset(sink, 0, (size_t)total * sizeof(uint64_t)));
  CUDA_CHECK(cgbn_error_report_alloc(&report));

  printf("== edge battery: %s  N = 2^%u - 1, %d special values, %d products\n",
         aligned ? "k == BITS" : "k < BITS", k, (int)NV, instances);

  int ops[3];
  int nops = 0;
  if (aligned) ops[nops++] = OP_FOLD_ALIGN_CANON;
  if (tshift > 0) { ops[nops++] = OP_FOLD_GEN; ops[nops++] = OP_FOLD_FUSED;
#if PROBE_ROT
    ops[nops++] = OP_FOLD_ROT;
#endif
  }
  ops[nops++] = OP_MONT_MUL;

  int fails = 0;
  for (int oi = 0; oi < nops; oi++) {
    const int op = ops[oi];
    CUDA_CHECK(cudaMemcpy(dev, host, (size_t)total * SLOTS * sizeof(mem_t), cudaMemcpyHostToDevice));
    launch<params>(op, blocks, 1, report, dev, np0, k, tshift, sink);
    CUDA_CHECK(cudaDeviceSynchronize());
    if (cgbn_error_report_check(report)) {
      fprintf(stderr, "  CGBN error (op=%s, edge)\n", op_name(op));
      cgbn_error_report_reset(report);
    }
    CUDA_CHECK(cudaMemcpy(out, dev, (size_t)total * SLOTS * sizeof(mem_t), cudaMemcpyDeviceToHost));

    int bad = 0;
    for (int i = 0; i < instances; i++) {
      const int ia = (i / NV) % NV, ib = i % NV;
      mpz_t got, exp;
      mpz_inits(got, exp, NULL);
      mpz_from_mem(got, out[SLOTS * i + 0]._limbs, nl);
      mpz_mod(got, got, N);
      /* mont_mul got b in Montgomery form, so its plain result is a*b as well
         (a*b*R*R^-1); every op must land on a*b mod n. */
      mpz_mul(exp, v[ia], v[ib]);
      mpz_mod(exp, exp, N);
      if (mpz_cmp(got, exp) != 0) {
        bad++;
        if (bad <= 3) {
          gmp_fprintf(stderr, "  EDGE FAIL (%s) i=%d: a=%Zx b=%Zx\n    expected %Zx\n    got      %Zx\n",
                      op_name(op), i, v[ia], v[ib], exp, got);
        }
      }
      mpz_clears(got, exp, NULL);
    }
    printf("  %-18s %d products checked, %d mismatches  %s\n",
           op_name(op), instances, bad, bad ? "FAIL" : "OK");
    fails += bad ? 1 : 0;
  }

  cgbn_error_report_free(report);
  cudaFree(dev);
  cudaFree(sink);
  free(host);
  free(out);
  for (int i = 0; i < NV; i++) mpz_clear(v[i]);
  mpz_clears(N, R, NULL);
  return fails;
}

int main(int argc, char **argv) {
  /* one alias per tier: a params_t<a,b> template-id cannot be passed through a
     macro argument (the comma would split the argument list) */
  typedef params_t<4, 512>   t0_t;
  typedef params_t<8, 1024>  t1_t;
  typedef params_t<8, 2048>  t2_t;
  typedef params_t<16, 3072> t3_t;
  typedef params_t<16, 4096> t4_t;
  typedef params_t<16, 8192> t5_t;
  typedef params_t<16, 4608> t6_t;
  typedef params_t<16, 5120> t7_t;
  int tier = (argc > 1) ? atoi(argv[1]) : 7;
  int instances = (argc > 2) ? atoi(argv[2]) : 4096;
  int iters = (argc > 3) ? atoi(argv[3]) : 1000;
  int device = (argc > 4) ? atoi(argv[4]) : 1;
  const char *kmode = (argc > 5) ? argv[5] : "both";

  CUDA_CHECK(cudaSetDevice(device));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
  printf("Mersenne fold probe -- device=%d (%s, %d SMs), instances=%d, iters=%d, ktest=%s\n",
         device, prop.name, prop.multiProcessorCount, instances, iters, kmode);

  const bool do_align = (strcmp(kmode, "align") == 0 || strcmp(kmode, "both") == 0);
  const bool do_gen   = (strcmp(kmode, "gen") == 0 || strcmp(kmode, "both") == 0);
  const bool do_edge  = (strcmp(kmode, "edge") == 0);
  if (!do_align && !do_gen && !do_edge) { fprintf(stderr, "ktest = align|gen|both|edge\n"); return 1; }
  int fails = 0;

#define RUN_TIER(TP, KAL, KGN, TAGAL, TAGGN)                                        \
  do {                                                                              \
    if (do_edge) {                                                                  \
      fails += run_edge<TP>(KAL, device, true);                                     \
      fails += run_edge<TP>(KGN, device, false);                                    \
    } else {                                                                        \
      if (do_align) fails += run_case<TP>(instances, iters, KAL, TAGAL, device, true);  \
      if (do_gen)   fails += run_case<TP>(instances, iters, KGN, TAGGN, device, false); \
    }                                                                               \
  } while (0)

  switch (tier) {
    case 0: RUN_TIER(t0_t, 512,  480, "TPI=4 BITS=512",   "TPI=4 BITS=512");   break;
    case 1: RUN_TIER(t1_t, 1024, 960, "TPI=8 BITS=1024",  "TPI=8 BITS=1024");  break;
    case 2: RUN_TIER(t2_t, 2048, 1863, "TPI=8 BITS=2048", "TPI=8 BITS=2048");  break;
    case 3: RUN_TIER(t3_t, 3072, 2887, "TPI=16 BITS=3072", "TPI=16 BITS=3072"); break;
    case 4: RUN_TIER(t4_t, 4096, 3911, "TPI=16 BITS=4096", "TPI=16 BITS=4096"); break;
    case 5: RUN_TIER(t5_t, 8192, 8007, "TPI=16 BITS=8192", "TPI=16 BITS=8192"); break;
    case 6: RUN_TIER(t6_t, 4608, 4423, "TPI=16 BITS=4608", "TPI=16 BITS=4608 (M4423)"); break;
    case 7: RUN_TIER(t7_t, 5120, 4999, "TPI=16 BITS=5120", "TPI=16 BITS=5120 (M4999)"); break;
    default: fprintf(stderr, "tier 0..7\n"); return 1;
  }
#undef RUN_TIER
  printf("done, %d verification failures\n", fails);
  return fails ? 1 : 0;
}
