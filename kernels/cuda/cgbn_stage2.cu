/* cgbn_stage2.cu — host side of the native CUDA/CGBN ECM stage 2 (pairing / BSGS).

   Everything that depends only on (N, sigma, B1, B2, D) happens here with GMP, exactly
   as the stage-1 host TU does for its curves, so the device never divides or inverts:

     - the Suyama a24 and the start point (u^3 : v^3) per curve,
     - the stage-1 scalar s = torsion * lcm(1..B1) as an MSB-first bit array,
     - the candidate primes in (B1, B2] split into the two stage-2 kinds,
     - np0 = -N^-1 mod 2^32 (must agree with what cgbn_bn2mont returns in the kernel),
     - the memory budget for the two tables,
     - the final gcd of every accumulator with N.

   The two kernels and the arithmetic they use are documented in cgbn_stage2_kernel.h.
   Correctness oracle: tools/bench/stage2_ref.cpp --algorithm pairing (same candidate
   set, same hit test); tools/test/test_stage2_gpu.ps1 compares the two directly.
*/

#include "cgbn_stage2_cuda.h"
#include "cgbn_stage2_kernel.h"
#include "cuda_ecm_shim.h"      /* outputf / test_verbose */

#include <cuda_runtime.h>
#include <gmp.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

/* ── small helpers ─────────────────────────────────────────────────────────────── */

/* Every failure path returns ECM_ERROR.  (Device buffers allocated before the failure
   are not freed here; the process that got a CUDA error this deep is not expected to
   keep using the context, and the contexts we run in are per-process.) */
#define S2_CHECK(call)                                                                 \
  do {                                                                                 \
    cudaError_t e_ = (call);                                                            \
    if (e_ != cudaSuccess) {                                                            \
      outputf(OUTPUT_ERROR, "stage2: %s failed: %s\n", #call, cudaGetErrorString(e_));  \
      return ECM_ERROR;                                                                 \
    }                                                                                   \
  } while (0)

/* np0 = -N^-1 mod 2^32.  Same computation as find_np0() in cgbn_stage1.cu, and the
   value cgbn_bn2mont returns on the device (asserted in the stage-2 selftest). */
static uint32_t s2_find_np0(const mpz_t N) {
  mpz_t t;
  mpz_init(t);
  mpz_ui_pow_ui(t, 2, 32);
  if (!mpz_invert(t, N, t)) {          /* N even (or 1): no Montgomery form exists */
    mpz_clear(t);
    return 0;
  }
  const uint32_t np0 = (uint32_t)(0u - (uint32_t)mpz_get_ui(t));
  mpz_clear(t);
  return np0;
}

/* mpz_set_ui only takes 32 bits on Windows: assemble a uint64 from its halves. */
static void s2_set_u64(mpz_t r, uint64_t v) {
  mpz_set_ui(r, (unsigned long)(v >> 32));
  mpz_mul_2exp(r, r, 32);
  mpz_add_ui(r, r, (unsigned long)(v & 0xFFFFFFFFull));
}

/* Suyama (gmp-ecm -param 0) curve setup: a24 = (A+2)/4 and P0 = (u^3 : v^3).
   Returns 1 when the sigma is degenerate mod N and *factor carries the gcd. */
static int s2_suyama(mpz_t a24, mpz_t X0, mpz_t Z0, uint64_t sigma, const mpz_t N,
                     mpz_t factor) {
  mpz_t sig, u, v, num, den, inv, A, t;
  mpz_inits(sig, u, v, num, den, inv, A, t, nullptr);
  s2_set_u64(sig, sigma);
  mpz_mul(u, sig, sig);
  mpz_sub_ui(u, u, 5);                          /* u = sigma^2 - 5 */
  mpz_mul_ui(v, sig, 4);                        /* v = 4 sigma   */
  mpz_sub(num, v, u);
  mpz_powm_ui(num, num, 3, N);                  /* (v-u)^3 */
  mpz_mul_ui(t, u, 3);
  mpz_add(t, t, v);                             /* 3u+v */
  mpz_mul(num, num, t);
  mpz_powm_ui(den, u, 3, N);
  mpz_mul(den, den, v);
  mpz_mul_ui(den, den, 4);                      /* 4 u^3 v */
  if (mpz_invert(inv, den, N) == 0) {
    mpz_gcd(factor, den, N);
    mpz_clears(sig, u, v, num, den, inv, A, t, nullptr);
    return 1;
  }
  mpz_mul(A, num, inv);
  mpz_sub_ui(A, A, 2);                          /* A = (v-u)^3(3u+v)/(4u^3v) - 2 */
  mpz_add_ui(t, A, 2);
  mpz_set_ui(den, 4);
  if (mpz_invert(inv, den, N) == 0) {
    mpz_gcd(factor, den, N);
    mpz_clears(sig, u, v, num, den, inv, A, t, nullptr);
    return 1;
  }
  mpz_mul(a24, t, inv);
  mpz_mod(a24, a24, N);                         /* a24 = (A+2)/4 */
  mpz_powm_ui(X0, u, 3, N);                     /* X0 = u^3 */
  mpz_powm_ui(Z0, v, 3, N);                     /* Z0 = v^3 */
  mpz_clears(sig, u, v, num, den, inv, A, t, nullptr);
  return 0;
}

/* s = torsion * lcm(1..B1), MSB-first, trailing (low) zero bits dropped from the front.
   Identical to build_s_bits() in tools/bench/stage2_ref.cpp. */
static void s2_build_s_bits(uint64_t B1, uint64_t torsion, std::vector<uint8_t> &bits) {
  mpz_t s;
  mpz_init_set_ui(s, (unsigned long)(torsion ? torsion : 1));
  std::vector<uint8_t> composite((size_t)B1 + 1, 0);
  for (uint64_t i = 2; i <= B1; ++i) {
    if (composite[(size_t)i]) continue;
    for (uint64_t j = i * 2; j <= B1; j += i) composite[(size_t)j] = 1;
    uint64_t pk = i;
    while (pk <= B1 / i) pk *= i;               /* highest power of i <= B1 */
    mpz_mul_ui(s, s, (unsigned long)pk);
  }
  const size_t n = mpz_sizeinbase(s, 2);
  bits.assign(n, 0);
  for (size_t i = 0; i < n; ++i) bits[n - 1 - i] = (uint8_t)mpz_tstbit(s, i);
  mpz_clear(s);
}

static void s2_bits_of_u64(uint64_t k, std::vector<uint8_t> &bits) {
  bits.clear();
  if (k == 0) return;
  int top = 63;
  while (((k >> top) & 1u) == 0) --top;
  for (int i = top; i >= 0; --i) bits.push_back((uint8_t)((k >> i) & 1u));
}

/* ── candidate primes ────────────────────────────────────────────────────────────
   A segmented sieve, so B2 in the 1e9 range does not need a B2-sized bitmap.  Each
   prime p in (B1, B2] is classified exactly like the CPU reference:
     p <= D/2  -> "small": baby[p] = [p]Q is the stage-2 test
     else      -> r = p mod D, j = min(r, D-r), i = (p -+ j)/D, and the pair (i, j)
   Primes with p | D are skipped (stage 1 covers them), and so are pairs outside the
   giant table. */
struct s2_primes {
  std::vector<uint32_t> p_i, p_j;      /* pairing candidates */
  std::vector<uint32_t> small;         /* baby indices for the small primes */
  uint64_t total = 0;                  /* primes seen in (B1, B2] */
  uint64_t skipped_d = 0;              /* p | D */
  uint64_t skipped_range = 0;          /* pair outside 1..imax */
};

static void s2_build_primes(uint64_t B1, uint64_t B2, uint64_t D, uint32_t half,
                            uint32_t imax, s2_primes &out) {
  const uint64_t lo = B1 + 1;                  /* candidates are p > B1 */
  const uint64_t hi = B2;                      /* and p <= B2 */
  if (hi < lo) return;

  /* base primes up to sqrt(hi) */
  uint64_t root = 1;
  while ((root + 1) * (root + 1) <= hi) ++root;
  std::vector<uint32_t> base;
  {
    std::vector<uint8_t> comp((size_t)root + 1, 0);
    for (uint64_t i = 2; i <= root; ++i) {
      if (comp[(size_t)i]) continue;
      base.push_back((uint32_t)i);
      for (uint64_t j = i * i; j <= root; j += i) comp[(size_t)j] = 1;
    }
  }

  const uint64_t BLOCK = 1u << 21;
  std::vector<uint8_t> seg((size_t)BLOCK);
  for (uint64_t b = lo; b <= hi; b += BLOCK) {
    const uint64_t end = std::min(b + BLOCK - 1, hi);
    const uint64_t len = end - b + 1;
    std::fill(seg.begin(), seg.begin() + (size_t)len, 1);
    for (uint32_t q : base) {
      const uint64_t qq = (uint64_t)q * q;
      if (qq > end) break;
      uint64_t start = (qq > b) ? qq : ((b + q - 1) / q) * q;
      for (uint64_t m = start; m <= end; m += q) seg[(size_t)(m - b)] = 0;
    }
    for (uint64_t p = b; p <= end; ++p) {
      if (!seg[(size_t)(p - b)]) continue;
      ++out.total;
      if (p <= half) {
        out.small.push_back((uint32_t)p);
        continue;
      }
      const uint64_t r = p % D;
      if (r == 0) { ++out.skipped_d; continue; }
      const uint64_t j = (r <= half) ? r : D - r;
      const uint64_t i = (r <= half) ? (p - j) / D : (p + j) / D;
      if (i == 0 || i > imax) { ++out.skipped_range; continue; }
      out.p_i.push_back((uint32_t)i);
      out.p_j.push_back((uint32_t)j);
    }
  }
}

/* ── tier selection ────────────────────────────────────────────────────────────── */

int cgbn_stage2_tier_for(uint32_t nbits, uint32_t *bits, uint32_t *tpi, uint32_t *tpb) {
  const uint32_t want = nbits + (uint32_t)S2_CARRY_BITS;
  cgbn_s2_kernels_t best;
  best.bits = 0; best.tpi = 0; best.tpb = 0; best.tables = nullptr; best.pair = nullptr;
  cgbn_s2_kernels_t cands[2] = { cgbn_stage2_kernels_tpi4(want),
                                 cgbn_stage2_kernels_tpi8(want) };
  for (int i = 0; i < 2; ++i) {
    if (!cgbn_s2_valid(cands[i])) continue;
    if (best.bits == 0 || cands[i].bits < best.bits) best = cands[i];
  }
  if (!cgbn_s2_valid(best)) return 0;
  if (bits) *bits = best.bits;
  if (tpi) *tpi = best.tpi;
  if (tpb) *tpb = best.tpb;
  return 1;
}

/* ── the entry point ──────────────────────────────────────────────────────────── */

extern "C" int cgbn_ecm_stage2(mpz_t *factors, int *array_found, int max_factors,
                               const mpz_t N, uint32_t curves, const uint64_t *sigma,
                               const char *const *x_hex, const ecm_stage2_opts *opts,
                               float *gputime, uint32_t *stage1_done, int verbose) {
  if (gputime) *gputime = 0.0f;
  if (stage1_done) *stage1_done = 0;
  if (curves == 0) return ECM_ERROR;
  if (opts == nullptr) return ECM_ERROR;

  const uint64_t B1 = opts->b1, B2 = opts->b2, D = opts->d;
  if (D < 2 || B1 >= B2) {
    outputf(OUTPUT_ERROR, "stage2: need 2 <= D and B1 < B2 (got D=%llu B1=%llu B2=%llu)\n",
            (unsigned long long)D, (unsigned long long)B1, (unsigned long long)B2);
    return ECM_ERROR;
  }
  if (mpz_even_p(N)) {
    outputf(OUTPUT_ERROR, "stage2: N must be odd (CGBN Montgomery needs an odd modulus)\n");
    return ECM_ERROR;
  }

  const uint32_t half = (uint32_t)(D / 2);
  const uint32_t imax = (uint32_t)(B2 / D + 2);
  const uint32_t segs = (opts->segs == 0u) ? 1u : opts->segs;
  const uint32_t torsion = (opts->torsion == 0u) ? 1u : opts->torsion;

  const uint32_t nbits = (uint32_t)mpz_sizeinbase(N, 2);
  uint32_t bits = 0, tpi = 0, tpb = 0;
  if (!cgbn_stage2_tier_for(nbits, &bits, &tpi, &tpb)) {
    outputf(OUTPUT_ERROR,
            "stage2: no instantiated tier covers a %u-bit N (largest is 2048 bits in this "
            "build); add the tier to kernels/cuda/cgbn_stage2_kernels_tpi*.cu\n", nbits);
    return ECM_ERROR;
  }
  const uint32_t LIMBS = bits / 32;

  /* ---- host-side per-curve parameters ---- */
  std::vector<uint8_t> sbits, dbits;
  s2_build_s_bits(B1, torsion, sbits);
  s2_bits_of_u64(D, dbits);

  s2_primes pr;
  s2_build_primes(B1, B2, D, half, imax, pr);
  if (pr.p_i.size() == 0 && pr.small.size() == 0) {
    outputf(OUTPUT_ERROR, "stage2: no candidate primes in (%llu, %llu]\n",
            (unsigned long long)B1, (unsigned long long)B2);
    return ECM_ERROR;
  }

  std::vector<uint32_t> h_a24((size_t)curves * LIMBS, 0);
  std::vector<uint32_t> h_sx((size_t)curves * LIMBS, 0), h_sz((size_t)curves * LIMBS, 0);
  std::vector<uint32_t> h_x((size_t)curves * LIMBS, 0);
  std::vector<uint32_t> h_mod((size_t)LIMBS, 0);
  uint32_t have_x = 1;
  {
    mpz_t t;
    mpz_init(t);
    mpz_export(h_mod.data(), nullptr, -1, 4, 0, 0, N);
    const size_t exp = mpz_sizeinbase(N, 2) / 32 + 1;
    if (exp > LIMBS) {
      outputf(OUTPUT_ERROR, "stage2: internal error: N does not fit the tier\n");
      mpz_clear(t);
      return ECM_ERROR;
    }
    mpz_clear(t);
  }

  uint32_t degenerate = 0;
  for (uint32_t i = 0; i < curves; ++i) {
    const char *xs = x_hex ? x_hex[i] : nullptr;
    if (xs == nullptr || xs[0] == '\0') have_x = 0;
  }

  for (uint32_t i = 0; i < curves; ++i) {
    const uint64_t sg = sigma ? sigma[i] : 2;
    mpz_t a24, x0, z0, f;
    mpz_inits(a24, x0, z0, f, nullptr);
    if (!have_x) {
      if (s2_suyama(a24, x0, z0, sg, N, f) == 1) {
        /* Degenerate curve: the sigma itself splits N.  Report it like the reference. */
        ++degenerate;
        if (f && mpz_cmp_ui(f, 1) > 0 && mpz_cmp(f, N) < 0) {
          int slot = -1;
          for (int k = 0; k < max_factors; ++k)
            if (array_found[k] && mpz_cmp(factors[k], f) == 0) { slot = k; break; }
          if (slot < 0)
            for (int k = 0; k < max_factors; ++k)
              if (!array_found[k]) { mpz_set(factors[k], f); array_found[k] = 1; slot = k; break; }
        }
      }
      mpz_export(h_a24.data() + (size_t)i * LIMBS, nullptr, -1, 4, 0, 0, a24);
      mpz_export(h_sx.data() + (size_t)i * LIMBS, nullptr, -1, 4, 0, 0, x0);
      mpz_export(h_sz.data() + (size_t)i * LIMBS, nullptr, -1, 4, 0, 0, z0);
    } else {
      if (s2_suyama(a24, x0, z0, sg, N, f) == 1) ++degenerate;
      mpz_export(h_a24.data() + (size_t)i * LIMBS, nullptr, -1, 4, 0, 0, a24);
      if (mpz_set_str(x0, x_hex[i], 16) != 0) {
        outputf(OUTPUT_ERROR, "stage2: curve %u: unparsable affine x '%s'\n", i, x_hex[i]);
        mpz_clears(a24, x0, z0, f, nullptr);
        return ECM_ERROR;
      }
      mpz_mod(x0, x0, N);
      mpz_export(h_x.data() + (size_t)i * LIMBS, nullptr, -1, 4, 0, 0, x0);
    }
    mpz_clears(a24, x0, z0, f, nullptr);
  }

  /* ---- memory budget ---- */
  const size_t limb_bytes = sizeof(uint32_t);
  const size_t baby_words = (size_t)curves * ((size_t)half + 1) * 2 * LIMBS;
  const size_t giant_words = (size_t)curves * ((size_t)imax + 1) * 2 * LIMBS;
  const size_t other_words = (size_t)curves * LIMBS * 3u /* a24,x,z */
                           + (size_t)LIMBS
                           + (size_t)curves * (size_t)segs * LIMBS
                           + (size_t)pr.p_i.size() * 2u + (size_t)pr.small.size()
                           + sbits.size() + dbits.size();
  const size_t need_bytes = (baby_words + giant_words + other_words) * limb_bytes;
  size_t free_bytes = 0, total_bytes = 0;
  S2_CHECK(cudaSetDevice(opts->device_index < 0 ? 0 : opts->device_index));
  S2_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
  if (need_bytes > free_bytes) {
    outputf(OUTPUT_ERROR,
            "stage2: needs %.1f MB (baby %.1f + giant %.1f) but only %.1f MB are free on "
            "this device; reduce curves, D or B2\n",
            (double)need_bytes / 1048576.0, (double)(baby_words * limb_bytes) / 1048576.0,
            (double)(giant_words * limb_bytes) / 1048576.0, (double)free_bytes / 1048576.0);
    return ECM_ERROR;
  }

  /* ---- device buffers ---- */
  uint32_t *d_mod = nullptr, *d_a24 = nullptr, *d_sx = nullptr, *d_sz = nullptr;
  uint32_t *d_x = nullptr, *d_baby = nullptr, *d_giant = nullptr;
  uint32_t *d_z1 = nullptr, *d_acc = nullptr, *d_pi = nullptr, *d_pj = nullptr;
  uint32_t *d_ps = nullptr;
  uint8_t *d_sbits = nullptr, *d_dbits = nullptr;
  cgbn_error_report_t *report = nullptr;

  auto cleanup = [&]() {
    if (d_mod) cudaFree(d_mod);
    if (d_a24) cudaFree(d_a24);
    if (d_sx) cudaFree(d_sx);
    if (d_sz) cudaFree(d_sz);
    if (d_x) cudaFree(d_x);
    if (d_baby) cudaFree(d_baby);
    if (d_giant) cudaFree(d_giant);
    if (d_z1) cudaFree(d_z1);
    if (d_acc) cudaFree(d_acc);
    if (d_pi) cudaFree(d_pi);
    if (d_pj) cudaFree(d_pj);
    if (d_ps) cudaFree(d_ps);
    if (d_sbits) cudaFree(d_sbits);
    if (d_dbits) cudaFree(d_dbits);
    if (report) cgbn_error_report_free(report);
  };

  S2_CHECK(cudaMalloc((void **)&d_mod, (size_t)LIMBS * limb_bytes));
  S2_CHECK(cudaMalloc((void **)&d_a24, (size_t)curves * LIMBS * limb_bytes));
  S2_CHECK(cudaMalloc((void **)&d_baby, baby_words * limb_bytes));
  S2_CHECK(cudaMalloc((void **)&d_giant, giant_words * limb_bytes));
  S2_CHECK(cudaMalloc((void **)&d_z1, (size_t)curves * LIMBS * limb_bytes));
  S2_CHECK(cudaMalloc((void **)&d_acc, (size_t)curves * (size_t)segs * LIMBS * limb_bytes));
  if (!have_x) {
    S2_CHECK(cudaMalloc((void **)&d_sx, (size_t)curves * LIMBS * limb_bytes));
    S2_CHECK(cudaMalloc((void **)&d_sz, (size_t)curves * LIMBS * limb_bytes));
  } else {
    S2_CHECK(cudaMalloc((void **)&d_x, (size_t)curves * LIMBS * limb_bytes));
  }
  if (pr.p_i.size()) {
    S2_CHECK(cudaMalloc((void **)&d_pi, pr.p_i.size() * sizeof(uint32_t)));
    S2_CHECK(cudaMalloc((void **)&d_pj, pr.p_j.size() * sizeof(uint32_t)));
  }
  if (pr.small.size())
    S2_CHECK(cudaMalloc((void **)&d_ps, pr.small.size() * sizeof(uint32_t)));
  S2_CHECK(cudaMalloc((void **)&d_sbits, std::max<size_t>(sbits.size(), 1)));
  S2_CHECK(cudaMalloc((void **)&d_dbits, std::max<size_t>(dbits.size(), 1)));
  S2_CHECK(cgbn_error_report_alloc(&report));

  S2_CHECK(cudaMemcpy(d_mod, h_mod.data(), (size_t)LIMBS * limb_bytes, cudaMemcpyHostToDevice));
  S2_CHECK(cudaMemcpy(d_a24, h_a24.data(), (size_t)curves * LIMBS * limb_bytes,
                      cudaMemcpyHostToDevice));
  if (have_x) {
    S2_CHECK(cudaMemcpy(d_x, h_x.data(), (size_t)curves * LIMBS * limb_bytes,
                        cudaMemcpyHostToDevice));
  } else {
    S2_CHECK(cudaMemcpy(d_sx, h_sx.data(), (size_t)curves * LIMBS * limb_bytes,
                        cudaMemcpyHostToDevice));
    S2_CHECK(cudaMemcpy(d_sz, h_sz.data(), (size_t)curves * LIMBS * limb_bytes,
                        cudaMemcpyHostToDevice));
  }
  if (pr.p_i.size()) {
    S2_CHECK(cudaMemcpy(d_pi, pr.p_i.data(), pr.p_i.size() * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));
    S2_CHECK(cudaMemcpy(d_pj, pr.p_j.data(), pr.p_j.size() * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));
  }
  if (pr.small.size())
    S2_CHECK(cudaMemcpy(d_ps, pr.small.data(), pr.small.size() * sizeof(uint32_t),
                        cudaMemcpyHostToDevice));
  S2_CHECK(cudaMemcpy(d_sbits, sbits.data(), sbits.size(), cudaMemcpyHostToDevice));
  S2_CHECK(cudaMemcpy(d_dbits, dbits.data(), dbits.size(), cudaMemcpyHostToDevice));

  /* ---- launch ---- */
  /* The tier was already chosen by cgbn_stage2_tier_for; ask each family for that exact
     size so we get the kernel pointers that match `bits` (2^BITS is baked into the
     Montgomery domain, so a mismatched tier would silently compute a different curve). */
  const cgbn_s2_kernels_t k4 = cgbn_stage2_kernels_tpi4(bits);
  const cgbn_s2_kernels_t k8 = cgbn_stage2_kernels_tpi8(bits);
  cgbn_s2_kernels_t K;
  std::memset(&K, 0, sizeof(K));
  if (cgbn_s2_valid(k4) && k4.bits == bits)      K = k4;
  else if (cgbn_s2_valid(k8) && k8.bits == bits) K = k8;
  if (!cgbn_s2_valid(K)) {
    outputf(OUTPUT_ERROR, "stage2: internal error: no kernel for tier %u\n", bits);
    cleanup();
    return ECM_ERROR;
  }

  s2_args a;
  std::memset(&a, 0, sizeof(a));
  a.curves = curves;
  a.segs = segs;
  a.half = half;
  a.imax = imax;
  a.n_pair = (uint32_t)pr.p_i.size();
  a.n_small = (uint32_t)pr.small.size();
  a.np0 = s2_find_np0(N);
  a.have_x = have_x;
  a.s_bits = (uint32_t)sbits.size();
  a.d_bits = (uint32_t)dbits.size();
  a.sbits = d_sbits;
  a.dbits = d_dbits;
  a.modulus = d_mod;
  a.a24 = d_a24;
  a.start_x = have_x ? nullptr : d_sx;
  a.start_z = have_x ? nullptr : d_sz;
  a.x_in = have_x ? d_x : nullptr;
  a.p_i = d_pi;
  a.p_j = d_pj;
  a.p_small = d_ps;
  a.baby = d_baby;
  a.giant = d_giant;
  a.z_stage1 = d_z1;
  a.acc = d_acc;

  cudaEvent_t ev0, ev1;
  S2_CHECK(cudaEventCreate(&ev0));
  S2_CHECK(cudaEventCreate(&ev1));
  S2_CHECK(cudaEventRecord(ev0));

  {
    const uint32_t threads = curves * tpi;
    K.tables<<<(threads + tpb - 1) / tpb, tpb>>>(report, a);
    S2_CHECK(cudaGetLastError());
  }
  {
    const uint32_t threads = curves * segs * tpi;
    K.pair<<<(threads + tpb - 1) / tpb, tpb>>>(report, a);
    S2_CHECK(cudaGetLastError());
  }
  S2_CHECK(cudaDeviceSynchronize());
  S2_CHECK(cudaEventRecord(ev1));
  S2_CHECK(cudaEventSynchronize(ev1));
  float ms = 0.0f;
  S2_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
  if (gputime) *gputime = ms / 1000.0f;
  S2_CHECK(cudaEventDestroy(ev0));
  S2_CHECK(cudaEventDestroy(ev1));

  if (cgbn_error_report_check(report)) {
    outputf(OUTPUT_ERROR, "stage2: CGBN reported an error (see cgbn_error_report)\n");
    cleanup();
    return ECM_ERROR;
  }

  /* ---- host reduction: one gcd per curve ---- */
  std::vector<uint32_t> h_acc((size_t)curves * (size_t)segs * LIMBS);
  std::vector<uint32_t> h_z1((size_t)curves * LIMBS);
  S2_CHECK(cudaMemcpy(h_acc.data(), d_acc, h_acc.size() * limb_bytes, cudaMemcpyDeviceToHost));
  S2_CHECK(cudaMemcpy(h_z1.data(), d_z1, h_z1.size() * limb_bytes, cudaMemcpyDeviceToHost));

  {
    mpz_t prod, g, t, z;
    mpz_inits(prod, g, t, z, nullptr);
    uint32_t n_stage1 = 0;
    for (uint32_t i = 0; i < curves; ++i) {
      mpz_set_ui(prod, 1);
      for (uint32_t s = 0; s < segs; ++s) {
        mpz_import(t, LIMBS, -1, 4, 0, 0,
                   h_acc.data() + ((size_t)i * segs + s) * LIMBS);
        mpz_mul(prod, prod, t);
        mpz_mod(prod, prod, N);
      }
      mpz_gcd(g, prod, N);
      if (mpz_cmp_ui(g, 1) > 0 && mpz_cmp(g, N) < 0) {
        int slot = -1;
        for (int k = 0; k < max_factors; ++k)
          if (array_found[k] && mpz_cmp(factors[k], g) == 0) { slot = k; break; }
        if (slot < 0)
          for (int k = 0; k < max_factors; ++k)
            if (!array_found[k]) { mpz_set(factors[k], g); array_found[k] = 1; slot = k; break; }
        if (test_verbose(OUTPUT_NORMAL))
          outputf(OUTPUT_NORMAL, "stage2: curve %u (sigma=%llu) gcd found a factor\n", i,
                  (unsigned long long)(sigma ? sigma[i] : 0));
      } else if (mpz_cmp(g, N) == 0) {
        outputf(OUTPUT_NORMAL,
                "stage2: curve %u: the accumulated product is 0 mod N (stage 2 hit every "
                "factor at once); lower B2 for this curve\n", i);
      }
      /* The stage-1 gcd the reference does before stage 2 (Z of [s]Q). */
      mpz_import(z, LIMBS, -1, 4, 0, 0, h_z1.data() + (size_t)i * LIMBS);
      mpz_gcd(g, z, N);
      if (mpz_cmp_ui(g, 1) > 0 && mpz_cmp(g, N) < 0) ++n_stage1;
    }
    if (stage1_done) *stage1_done = n_stage1;
    mpz_clears(prod, g, t, z, nullptr);
  }

  outputf(OUTPUT_NORMAL,
          "GPU stage2: curves=%u tier=%u/%u tpb=%u segs=%u B1=%llu B2=%llu D=%llu "
          "cand=%zu (+%zu small, %llu skipped) stage1_point=%s time=%.3f s\n",
          curves, bits, tpi, K.tpb, segs, (unsigned long long)B1, (unsigned long long)B2,
          (unsigned long long)D, pr.p_i.size(), pr.small.size(),
          (unsigned long long)(pr.skipped_d + pr.skipped_range),
          have_x ? "save" : "ladder", (double)(gputime ? *gputime : 0.0f));
  if (test_verbose(OUTPUT_VERBOSE) && degenerate)
    outputf(OUTPUT_VERBOSE, "stage2: %u degenerate sigma(s)\n", degenerate);

  cleanup();
  return 0;
}
