/* cgbn_stage1.h: header for CGBN (GPU) based ecm stage 1.

Copyright 2021 Seth Troisi

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or (at your
option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
more details.

You should have received a copy of the GNU General Public License
along with this program; see the file COPYING.  If not, see
http://www.gnu.org/licenses/ or write to the Free Software Foundation, Inc.,
51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA.
*/

#include "host_math.h"
#include "../../runtime/cuda_ecm_shim.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <algorithm>
#include <stdexcept>
#ifndef ECM_CUDA_FULL_BUILD
#define IS_DEV_BUILD
#endif

namespace ecm_cuda_mont_host {
int verify_size_of_n(mpz_srcptr N,size_t max_bits,size_t carry_bits) {
  size_t n_log2 = mpz_sizeinbase(N, 2);

  /* Using check_gpuecm.sage it looks like 4 bits would suffice. */
  if(max_bits<carry_bits)return ECM_ERROR;
  size_t max_usable_bits = max_bits - carry_bits;

  if (n_log2 <= max_usable_bits)
    return ECM_NO_FACTOR_FOUND;

  outputf (OUTPUT_ERROR, "GPU: N(%zu bits) + carry(%zu bits) > BITS(%zu)\n",
      n_log2, carry_bits, max_bits);
  outputf (OUTPUT_ERROR, "GPU: Error, input number should be stricly lower than 2^%zu\n",
      max_usable_bits);
  return ECM_ERROR;
}



static
void to_mpz(mpz_t r, const uint32_t *x, uint32_t count) {
  mpz_import (r, count, -1, sizeof(uint32_t), 0, 0, x);
}

static
void from_mpz(const mpz_t s, uint32_t *x, uint32_t count) {
  size_t words;

  if(mpz_sizeinbase (s, 2) > count * 32) {
    fprintf (stderr, "from_mpz failed -- result does not fit\n");
    throw std::runtime_error("Montgomery packed value exceeds container");
  }

  mpz_export (x, &words, -1, sizeof(uint32_t), 0, 0, s);
  while(words<count)
    x[words++]=0;
}



static
int findfactor(mpz_t factor, const mpz_t N, const mpz_t x_final, const mpz_t z_final) {
    // XXX: combine / refactor logic with cudawrapper.c findfactor

    /* Check if factor found */

    bool inverted = mpz_invert(factor, z_final, N);    // aZ ^ (N-2) % N
    if (inverted) {
        mpz_mul(factor, x_final, factor);         // aX * aZ^-1
        mpz_mod(factor, factor, N);             // "Residual"
        return ECM_NO_FACTOR_FOUND;
    }

    mpz_gcd(factor, z_final, N);
    if (mpz_cmp_ui(factor, 1) <= 0 || mpz_cmp(factor, N) >= 0)
        return ECM_ERROR; // A whole-modulus gcd has no valid affine output/factor.
    return ECM_FACTOR_FOUND_STEP1;
}


// A failed inverse is a mathematical result, never a zero coefficient.
static bool inverse_or_factor(mpz_ptr inverse,mpz_srcptr denominator,mpz_srcptr n,
        uint32_t lane,uint64_t sigma,int param,const ConstructionResults *result) {
    if(mpz_invert(inverse,denominator,n))return true;
    mpz_gcd(inverse,denominator,n);
    if(mpz_cmp_ui(inverse,1)<=0 || mpz_cmp(inverse,n)>=0) {
        if(result && result->degenerate_curve)*result->degenerate_curve=lane;
        throw ecm_cuda_stage1::RunError(ecm_cuda_stage1::Failure::Preparation,
            "degenerate construction denominator: param="+std::to_string(param)+" sigma="+std::to_string(sigma));
    }
    if(!result || !result->factors || !result->found)
        throw ecm_cuda_stage1::RunError(ecm_cuda_stage1::Failure::Preparation,
            "construction factor requires a per-curve result sink");
    mpz_set(result->factors[lane],inverse);result->found[lane]=ECM_FACTOR_FOUND_STEP1;
    if(result->publish && !result->publish(lane,inverse,result->context))
        throw ecm_cuda_stage1::RunError(ecm_cuda_stage1::Failure::Publication,"cannot publish construction factor");
    outputf(OUTPUT_NORMAL,"GPU: construction factor found: param=%d sigma=%llu curve=%u\n",param,
            static_cast<unsigned long long>(sigma),lane);
    return false;
}
static void identity_lane(uint32_t *lane,size_t limbs,mpz_srcptr n) {
    // The lane's real factor is held separately. No fabricated curve is exposed.
    // Identity arithmetic keeps indexing stable without device division.
    std::fill(lane,lane+7*limbs,0u);from_mpz(n,lane,uint32_t(limbs));
    lane[limbs]=1;lane[2*limbs]=2;lane[3*limbs]=1;lane[5*limbs]=1;
}

static size_t validate_layout(mpz_srcptr n,uint32_t curves,uint32_t bits,size_t *bytes) {
    if(!bytes || !curves || bits<32 || bits%32 || mpz_cmp_ui(n,3)<=0 ||
       !mpz_odd_p(n) || mpz_sizeinbase(n,2)>bits)
        throw std::invalid_argument("invalid Montgomery host buffer layout/modulus");
    return bits/32;
}

uint32_t find_np0(const mpz_t N) {
  uint32_t np0;
  mpz_t temp;
  mpz_init(temp);
  mpz_ui_pow_ui(temp, 2, 32);
  // NB: must not put mpz_invert() inside assert() — NDEBUG (Release) would drop
  // the call and leave temp = 2^32, giving np0 = 0 on this platform.
  int inv_ok = mpz_invert(temp, N, temp);
  if(!inv_ok){mpz_clear(temp);throw std::invalid_argument("Montgomery modulus must be odd");}
  np0 = -mpz_get_ui(temp);
  mpz_clear(temp);
  return np0;
}


uint32_t* allocate_and_set_s_bits(const mpz_t s, uint64_t *nbits) {
  if(!nbits || mpz_sgn(s)<=0)throw std::invalid_argument("positive Stage1 scalar required");
  uint64_t num_bits = *nbits = mpz_sizeinbase (s, 2);

  uint64_t allocated = num_bits/32+(num_bits%32!=0);
  uint32_t *s_bits = (uint32_t*) malloc (ecm_cuda_stage1::checked_size({sizeof(uint32_t),size_t(allocated)}));

  if(!s_bits)throw std::bad_alloc();
  size_t countp;
  mpz_export (s_bits, &countp, -1, sizeof(uint32_t), 0, 0, s);
  assert (countp == allocated);

  return s_bits;
}


uint32_t* set_p_2p(const mpz_t N,
                   uint32_t curves, uint32_t sigma,
                   uint32_t BITS, size_t *data_size) {
  /**
   * Store 5 numbers per curve:
   * N, P_a (x, z), P_b (x, z)
   *
   * P_a is initialized with (2, 1)
   * P_b (for the doubled terms) is initialized with (9, 64 * d + 8)
   */

  const size_t limbs_per = validate_layout(N,curves,BITS,data_size);
  *data_size = ecm_cuda_stage1::checked_size({5,size_t(curves),limbs_per,sizeof(uint32_t)});
  uint32_t *data = (uint32_t*) malloc(*data_size);
  if(!data)throw std::bad_alloc();
  uint32_t *datum = data;

  mpz_t x;
  mpz_init(x);
  for(uint32_t index = 0; index < curves; index++) {
      // d = (sigma / 2^32) mod N BUT 2^32 handled by special_mul_ui32
      uint32_t d = sigma + index;

      // Modulo (N)
      from_mpz(N, datum + 0 * limbs_per, BITS/32);

      // P1 (X, Z)
      mpz_set_ui(x, 2);
      from_mpz(x, datum + 1 * limbs_per, BITS/32);
      mpz_set_ui(x, 1);
      from_mpz(x, datum + 2 * limbs_per, BITS/32);

      // 2P = P2 (X, Z)
      // P2_x = 9
      mpz_set_ui(x, 9);
      from_mpz(x, datum + 3 * limbs_per, BITS/32);

      // d = sigma * mod_inverse(2 ** 32, N)
      mpz_ui_pow_ui(x, 2, 32);
      mpz_invert(x, x, N);
      mpz_mul_ui(x, x, d);
      // P2_x = 64 * d + 8;
      mpz_mul_ui(x, x, 64);
      mpz_add_ui(x, x, 8);
      mpz_mod(x, x, N);

      outputf (OUTPUT_TRACE, "sigma %d => P2_y: %Zd\n", d, x);
      from_mpz(x, datum + 4 * limbs_per, BITS/32);
      datum += 5 * limbs_per;
  }
  mpz_clear(x);
  return data;
}


/* ---------------------------------------------------------------------------
 * Suyama param0 curve/point setup (method = gpu + gpu_param = 0).
 *
 * Same math as the CPU reference (src/cpu/ecm_mont_cpu.cpp:mont_suyama_curve) and
 * as gmp-ecm's -param 0, so the two paths run IDENTICAL curves for the same sigma:
 *
 *   u = sigma^2 - 5        v = 4*sigma
 *   A = (v-u)^3 (3u+v) / (4 u^3 v) - 2      [Montgomery coefficient]
 *   a24 = (A+2)/4                            [doubling constant, full width]
 *   P = (u^3 : v^3)                          [start point]
 *   xdiff = X0/Z0                            [affine x of the ladder difference]
 *   2P via one xDBL(a24)                     [second ladder point]
 *
 * Everything sigma-dependent (including the three inversions) happens HERE, on the
 * host; the device kernel only sees the seven words below and needs no inversion.
 * That is why the param0 port costs nothing extra in the kernel besides giving back
 * the two shortcuts the batch parametrization enjoys (docs §19.3).
 *
 * Layout per curve (7 * BITS/32 words):  N, a24, xdiff, aX, aZ, bX, bZ
 * ------------------------------------------------------------------------- */
uint32_t* set_p_2p_suyama(const mpz_t N, uint32_t curves, uint64_t sigma0,
                          uint32_t BITS, size_t *data_size,const ConstructionResults *results) {
  if(mpz_cmp_ui(N,3)<=0 || !mpz_odd_p(N))throw std::invalid_argument("odd modulus > 3 required");
  const size_t limbs_per = validate_layout(N,curves,BITS,data_size);
  *data_size = ecm_cuda_stage1::checked_size({7,size_t(curves),limbs_per,sizeof(uint32_t)});
  uint32_t *data = (uint32_t*) malloc(*data_size);
  if(!data)throw std::bad_alloc();
  uint32_t *datum = data;

  mpz_t sigma, u, v, t, num, den, inv, A, a24, X0, Z0, xdiff;
  mpz_t aA, aB, AA, BB, E, X2, Z2;
  mpz_inits(sigma, u, v, t, num, den, inv, A, a24, X0, Z0, xdiff,
            aA, aB, AA, BB, E, X2, Z2, NULL);

  try {
  for(uint32_t index = 0; index < curves; index++) {
      /* sigma is 64-bit here (unlike the batch parametrization's 32-bit d), and
         mpz_set_ui only takes a 32-bit unsigned long on Windows -- assemble from
         the halves, exactly like mont_set_sigma() does in the CPU path. */
      const uint64_t sg = sigma0 + (uint64_t)index;
      mpz_set_ui(sigma, (unsigned long)(sg >> 32));
      mpz_mul_2exp(sigma, sigma, 32);
      mpz_add_ui(sigma, sigma, (unsigned long)(sg & 0xFFFFFFFFull));

      mpz_mul(u, sigma, sigma);
      mpz_sub_ui(u, u, 5);                          /* u = sigma^2 - 5 */
      mpz_mul_ui(v, sigma, 4);                      /* v = 4*sigma */

      mpz_sub(t, v, u);
      mpz_powm_ui(num, t, 3, N);                    /* (v-u)^3 */
      mpz_mul_ui(t, u, 3);
      mpz_add(t, t, v);                             /* 3u+v */
      mpz_mul(num, num, t);
      mpz_mod(num, num, N);

      mpz_powm_ui(den, u, 3, N);                    /* u^3 */
      mpz_mul_ui(den, den, 4);
      mpz_mul(den, den, v);
      mpz_mod(den, den, N);                         /* 4 u^3 v */

      if (!inverse_or_factor(inv,den,N,index,sg,0,results)) {
          identity_lane(datum,limbs_per,N);goto next_curve;
      }
      mpz_mul(A, num, inv);
      mpz_sub_ui(A, A, 2);
      mpz_mod(A, A, N);                             /* A */

      mpz_add_ui(a24, A, 2);
      mpz_set_ui(t, 4);
      mpz_invert(t, t, N);
      mpz_mul(a24, a24, t);
      mpz_mod(a24, a24, N);                         /* a24 = (A+2)/4 */

      mpz_powm_ui(X0, u, 3, N);
      mpz_powm_ui(Z0, v, 3, N);

      if (!inverse_or_factor(inv,Z0,N,index,sg,0,results)) {
          identity_lane(datum,limbs_per,N);goto next_curve;
      }
      mpz_mul(xdiff, X0, inv);
      mpz_mod(xdiff, xdiff, N);                     /* affine x of P */

      /* 2P, i.e. one xDBL with a24 -- the identical formula the CPU ladder uses:
         A = X+Z, B = X-Z, AA = A^2, BB = B^2, E = AA-BB,
         X2 = AA*BB, Z2 = E*(BB + a24*E)                                        */
      mpz_add(aA, X0, Z0);
      mpz_sub(aB, X0, Z0);
      mpz_mul(AA, aA, aA);  mpz_mod(AA, AA, N);
      mpz_mul(BB, aB, aB);  mpz_mod(BB, BB, N);
      mpz_sub(E, AA, BB);   mpz_mod(E, E, N);
      mpz_mul(X2, AA, BB);  mpz_mod(X2, X2, N);
      mpz_mul(t, a24, E);   mpz_mod(t, t, N);
      mpz_add(t, t, BB);    mpz_mod(t, t, N);
      mpz_mul(Z2, E, t);    mpz_mod(Z2, Z2, N);

      from_mpz(N,     datum + 0 * limbs_per, limbs_per);
      from_mpz(a24,   datum + 1 * limbs_per, limbs_per);
      from_mpz(xdiff, datum + 2 * limbs_per, limbs_per);
      from_mpz(X0,    datum + 3 * limbs_per, limbs_per);
      from_mpz(Z0,    datum + 4 * limbs_per, limbs_per);
      from_mpz(X2,    datum + 5 * limbs_per, limbs_per);
      from_mpz(Z2,    datum + 6 * limbs_per, limbs_per);

      outputf (OUTPUT_TRACE,
               "sigma %llu => a24 %Zd, xdiff %Zd, P (%Zd,%Zd) 2P (%Zd,%Zd)\n",
               (unsigned long long)sg, a24, xdiff, X0, Z0, X2, Z2);

next_curve:
      datum += 7 * limbs_per;
  }

  } catch(...) {
  mpz_clears(sigma, u, v, t, num, den, inv, A, a24, X0, Z0, xdiff,
             aA, aB, AA, BB, E, X2, Z2, NULL);
    free(data);throw;
  }
  mpz_clears(sigma, u, v, t, num, den, inv, A, a24, X0, Z0, xdiff,
             aA, aB, AA, BB, E, X2, Z2, NULL);
  return data;
}


/* ---------------------------------------------------------------------------
 * param2 ("batch 2", 6-torsion) curve/point setup (method = gpu + gpu_param = 2).
 *
 * Transcribed from gmp-ecm's get_curve_from_param2() (parametrizations.c:296-386):
 *
 *   P  = sigma * (-3 : 3 : 1) on the FIXED curve y^2 = x^3 + 36, sigma >= 2 a scalar
 *   (x:y:z) -> affine
 *   x3 = (3x + y + 6) / (2(y - 3))
 *   A  = -(3 x3^4 + 6 x3^2 - 1) / (4 x3^3)        [Montgomery coefficient]
 *   x0 = 2                                         [parametrizations.c:373]
 *
 * Consequences for our kernel (docs/performance/STAGE1.md): the ladder difference
 * is the CONSTANT 2, so the differential addition needs no multiply (the shift), while
 * a24 = (A+2)/4 is a full-width residue -- exactly the 5M+4S shape of
 * cgbn_stage1_kernels_param2.cu.  Measured +10.7% over param0 at M511.
 *
 * Layout per curve (7 * BITS/32 words), identical to the Suyama path:
 *   N, a24, xdiff (= 2), aX (= 2), aZ (= 1), bX (= 9), bZ (= 2P's Z)
 *
 * The Jacobian helpers below are the standard a=0 / mixed formulas for y^2 = x^3 + b;
 * in trace mode the code also verifies its own scalar multiply (6*P == O, which holds
 * because -3:3:1 is a 6-torsion point).
 * ------------------------------------------------------------------------- */

/* Jacobian doubling, a = 0 (dbl-2009-l) */
static void p2_jac_dbl(mpz_t X3, mpz_t Y3, mpz_t Z3,
                       const mpz_t X1, const mpz_t Y1, const mpz_t Z1, const mpz_t n) {
  /* All three outputs may alias the inputs (the caller passes X,Y,Z for both), so every
     intermediate is computed into a temporary and stored at the very end.  The earlier
     version stored Y3 before computing Z3 = 2*Y1*Z1, which silently corrupted Z3. */
  mpz_t A, B, C, D, E, F, t, xo, yo, zo;
  mpz_inits(A, B, C, D, E, F, t, xo, yo, zo, NULL);
  mpz_mul(A, X1, X1); mpz_mod(A, A, n);                     /* A = X1^2 */
  mpz_mul(B, Y1, Y1); mpz_mod(B, B, n);                     /* B = Y1^2 */
  mpz_mul(C, B, B);   mpz_mod(C, C, n);                     /* C = B^2 */
  mpz_add(D, X1, B);  mpz_mul(D, D, D); mpz_mod(D, D, n);
  mpz_sub(D, D, A);   mpz_sub(D, D, C); mpz_mod(D, D, n);
  mpz_mul_ui(D, D, 2); mpz_mod(D, D, n);                    /* D = 2((X1+B)^2 - A - C) */
  mpz_mul_ui(E, A, 3); mpz_mod(E, E, n);                    /* E = 3A */
  mpz_mul(F, E, E);   mpz_mod(F, F, n);                     /* F = E^2 */
  mpz_mul_ui(t, D, 2); mpz_mod(t, t, n);
  mpz_sub(xo, F, t);  mpz_mod(xo, xo, n);                   /* X3 = F - 2D */
  mpz_sub(t, D, xo);  mpz_mod(t, t, n);
  mpz_mul(t, t, E);   mpz_mod(t, t, n);                     /* E(D - X3) */
  mpz_mul_ui(C, C, 8); mpz_mod(C, C, n);
  mpz_sub(yo, t, C);  mpz_mod(yo, yo, n);                   /* Y3 = E(D-X3) - 8C */
  mpz_mul(t, Y1, Z1); mpz_mod(t, t, n);
  mpz_mul_ui(zo, t, 2); mpz_mod(zo, zo, n);                 /* Z3 = 2 Y1 Z1 */
  mpz_set(X3, xo); mpz_set(Y3, yo); mpz_set(Z3, zo);
  mpz_clears(A, B, C, D, E, F, t, xo, yo, zo, NULL);
}

/* Jacobian + affine mixed addition (madd-2007-bl), a = 0 */
static void p2_jac_add_affine(mpz_t X1, mpz_t Y1, mpz_t Z1,
                              const mpz_t x2, const mpz_t y2, const mpz_t n) {
  mpz_t Z1Z1, U2, S2, H, HH, I, J, r, V, t;
  mpz_inits(Z1Z1, U2, S2, H, HH, I, J, r, V, t, NULL);
  mpz_mul(Z1Z1, Z1, Z1); mpz_mod(Z1Z1, Z1Z1, n);            /* Z1Z1 = Z1^2 */
  mpz_mul(U2, x2, Z1Z1); mpz_mod(U2, U2, n);                /* U2 = X2 Z1^2 */
  mpz_mul(t, Z1, Z1Z1);  mpz_mod(t, t, n);
  mpz_mul(S2, y2, t);    mpz_mod(S2, S2, n);                /* S2 = Y2 Z1^3 */
  mpz_sub(H, U2, X1);    mpz_mod(H, H, n);                  /* H = U2 - X1 */
  mpz_mul(HH, H, H);     mpz_mod(HH, HH, n);                /* HH = H^2 */
  mpz_mul_ui(I, HH, 4);  mpz_mod(I, I, n);                  /* I = 4 HH */
  mpz_mul(J, H, I);      mpz_mod(J, J, n);                  /* J = H I */
  mpz_sub(r, S2, Y1);    mpz_mod(r, r, n);
  mpz_mul_ui(r, r, 2);   mpz_mod(r, r, n);                  /* r = 2(S2 - Y1) */
  mpz_mul(V, X1, I);     mpz_mod(V, V, n);                  /* V = X1 I */
  mpz_mul(t, r, r);      mpz_mod(t, t, n);
  mpz_sub(t, t, J);      mpz_sub(t, t, V); mpz_sub(t, t, V); mpz_mod(t, t, n);
  /* t = X3 */
  mpz_sub(V, V, t);      mpz_mod(V, V, n);
  mpz_mul(V, V, r);      mpz_mod(V, V, n);
  mpz_mul(J, Y1, J);     mpz_mod(J, J, n);
  mpz_mul_ui(J, J, 2);   mpz_mod(J, J, n);
  mpz_sub(V, V, J);      mpz_mod(V, V, n);                  /* Y3 = r(V - X3) - 2 Y1 J */
  mpz_add(H, Z1, H);     mpz_mul(H, H, H); mpz_mod(H, H, n);
  mpz_sub(H, H, Z1Z1);   mpz_sub(H, H, HH); mpz_mod(H, H, n);/* Z3 = (Z1+H)^2 - Z1Z1 - HH */
  mpz_set(X1, t); mpz_set(Y1, V); mpz_set(Z1, H);
  mpz_clears(Z1Z1, U2, S2, H, HH, I, J, r, V, t, NULL);
}

/* k * (-3:3:1) on y^2 = x^3 + 36, left-to-right double-and-add (the point is unique, so
   any valid chain gives the same result as gmp-ecm's addchain_param). */
static void p2_scalar_mul_base(mpz_t rx, mpz_t ry, mpz_t rz, const mpz_t k, const mpz_t N) {
  mpz_t X, Y, Z, bx, by;
  mpz_inits(X, Y, Z, bx, by, NULL);
  mpz_set_si(bx, -3); mpz_set_ui(by, 3);
  mpz_set_ui(X, 0); mpz_set_ui(Y, 1); mpz_set_ui(Z, 0);      /* point at infinity */
  int started = 0;
  for (int i = mpz_sizeinbase(k, 2) - 1; i >= 0; i--) {
    if (started) p2_jac_dbl(X, Y, Z, X, Y, Z, N);
    if (mpz_tstbit(k, i)) {
      if (!started) { mpz_set(X, bx); mpz_set(Y, by); mpz_set_ui(Z, 1); started = 1; }
      else          { p2_jac_add_affine(X, Y, Z, bx, by, N); }
    }
  }
  mpz_set(rx, X); mpz_set(ry, Y); mpz_set(rz, Z);
  mpz_clears(X, Y, Z, bx, by, NULL);
}

uint32_t* set_p_2p_param2(const mpz_t N, uint32_t curves, uint32_t sigma0,
                          uint32_t BITS, size_t *data_size,const ConstructionResults *results) {
  if(mpz_cmp_ui(N,3)<=0 || !mpz_odd_p(N) || sigma0<2)throw std::invalid_argument("param2 needs odd modulus > 3 and sigma >= 2");
  const size_t limbs_per = validate_layout(N,curves,BITS,data_size);
  *data_size = ecm_cuda_stage1::checked_size({7,size_t(curves),limbs_per,sizeof(uint32_t)});
  uint32_t *data = (uint32_t*) malloc(*data_size);
  if(!data)throw std::bad_alloc();
  uint32_t *datum = data;

  mpz_t k, x, y, z, t, u, v, w, A, a24, two, X2, Z2;
  mpz_inits(k, x, y, z, t, u, v, w, A, a24, two, X2, Z2, NULL);
  mpz_set_ui(two, 2);

  try {
  for (uint32_t index = 0; index < curves; index++) {
      /* sigma is the scalar multiplier here (gmp-ecm requires sigma >= 2; a sigma of 0
         or 1 is rejected before construction). */
      const uint32_t sg = sigma0 + index;
      mpz_set_ui(k, sg);

      p2_scalar_mul_base(x, y, z, k, N);

      /* affine normalisation of (x:y:z) */
      if (!inverse_or_factor(u,z,N,index,sg,2,results)) {
          identity_lane(datum,limbs_per,N);goto next_curve;
      }
      mpz_mul(v, u, u);   mpz_mod(v, v, N);
      mpz_mul(u, v, u);   mpz_mod(u, u, N);
      mpz_mul(x, x, v);   mpz_mod(x, x, N);
      mpz_mul(y, y, u);   mpz_mod(y, y, N);

      /* x3 = (3x + y + 6) / (2(y - 3)) */
      mpz_sub_ui(t, y, 3); mpz_mod(t, t, N);
      mpz_mul_ui(t, t, 2); mpz_mod(t, t, N);
      if (!inverse_or_factor(u,t,N,index,sg,2,results)) {
          identity_lane(datum,limbs_per,N);goto next_curve;
      }
      mpz_mul_ui(w, x, 3); mpz_mod(w, w, N);
      mpz_add(w, w, y);    mpz_mod(w, w, N);
      mpz_add_ui(w, w, 6); mpz_mod(w, w, N);
      mpz_mul(x, w, u);    mpz_mod(x, x, N);                  /* x3 */

      /* A = -(3 x3^4 + 6 x3^2 - 1) / (4 x3^3) */
      mpz_mul(u, x, x);    mpz_mod(u, u, N);                  /* x3^2 */
      mpz_mul(v, u, x);    mpz_mod(v, v, N);                  /* x3^3 */
      mpz_mul(w, u, u);    mpz_mod(w, w, N);                  /* x3^4 */
      mpz_mul_ui(u, u, 6); mpz_mod(u, u, N); mpz_neg(u, u);
      mpz_mul_ui(v, v, 4); mpz_mod(v, v, N);
      mpz_mul_ui(w, w, 3); mpz_mod(w, w, N); mpz_neg(w, w);
      if (!inverse_or_factor(t,v,N,index,sg,2,results)) {
          identity_lane(datum,limbs_per,N);goto next_curve;
      }
      mpz_add(w, w, u);    mpz_mod(w, w, N);
      mpz_add_ui(w, w, 1); mpz_mod(w, w, N);
      mpz_mul(A, w, t);    mpz_mod(A, A, N);

      /* a24 = (A+2)/4 (full width -- this is why param2 is 5M+4S, not 4M+4S).
         NOTE: stage 1 runs on the curve with coefficient A, NOT on the rescaled
         a/b form of gmp-ecm's FindGroupOrderParam2 comment: an independent
         reference ladder reproduces gmp-ecm's saved x with (A, x0=2) exactly
         (see docs/performance/STAGE1.md 5.6). */
      mpz_add_ui(a24, A, 2); mpz_mod(a24, a24, N);
      mpz_set_ui(t, 4);
      if (mpz_invert(t, t, N) == 0) mpz_set_ui(t, 0);
      mpz_mul(a24, a24, t); mpz_mod(a24, a24, N);

      /* Start point x0 = 2 ON THAT CURVE: P = (2:1) and 2P by one xDBL with the general
         formula: X+Z = 3, X-Z = 1, AA = 9, BB = 1, E = 8 => X2 = AA*BB = 9 and
         Z2 = E*(BB + a24*E) = 8 + 64*a24. */
      mpz_set_ui(X2, 9);
      mpz_mul_ui(Z2, a24, 64); mpz_mod(Z2, Z2, N);
      mpz_add_ui(Z2, Z2, 8);   mpz_mod(Z2, Z2, N);

      from_mpz(N,     datum + 0 * limbs_per, limbs_per);
      from_mpz(a24,   datum + 1 * limbs_per, limbs_per);
      from_mpz(two,   datum + 2 * limbs_per, limbs_per);      /* xdiff = 2 (constant) */
      from_mpz(two,   datum + 3 * limbs_per, limbs_per);      /* aX = 2 */
      mpz_set_ui(x, 1);
      from_mpz(x,     datum + 4 * limbs_per, limbs_per);      /* aZ = 1 */
      from_mpz(X2,    datum + 5 * limbs_per, limbs_per);      /* bX = 9 */
      from_mpz(Z2,    datum + 6 * limbs_per, limbs_per);      /* bZ */

next_curve:
      datum += 7 * limbs_per;
  }

  } catch(...) {
  mpz_clears(k, x, y, z, t, u, v, w, A, a24, two, X2, Z2, NULL);
    free(data);throw;
  }
  mpz_clears(k, x, y, z, t, u, v, w, A, a24, two, X2, Z2, NULL);
  return data;
}


int process_results(mpz_t *factors, int *array_found,
                    const mpz_t N,
                    const uint32_t *data, uint32_t cgbn_bits,
                    int curves, uint64_t sigma, uint32_t words_per_curve,
                    int p1_word, int p2_word, int curve_param,bool construction_results,uint32_t *degenerate_curve) {
  mpz_t x_final, z_final, modulo;
  mpz_init(modulo);
  mpz_init(x_final);
  mpz_init(z_final);

  const uint32_t limbs_per = cgbn_bits / 32;

  int youpi = ECM_NO_FACTOR_FOUND;
  int errors = 0;
  for(size_t i = 0; i < curves; i++) {
    if(construction_results && array_found[i]==ECM_FACTOR_FOUND_STEP1) {
        if(mpz_cmp_ui(factors[i],1)<=0 || mpz_cmp(factors[i],N)>=0 || !mpz_divisible_p(N,factors[i])) {
            ++errors;continue;
        }
        youpi=ECM_FACTOR_FOUND_STEP1;continue;
    }
    const uint32_t *datum = data + (words_per_curve * i * limbs_per);

    if (test_verbose (OUTPUT_TRACE) && i == 0) {
      to_mpz(modulo, datum + 0 * limbs_per, limbs_per);
      outputf (OUTPUT_TRACE, "index: 0 modulo: %Zd\n", modulo);

      to_mpz(x_final, datum + p1_word * limbs_per, limbs_per);
      to_mpz(z_final, datum + (p1_word + 1) * limbs_per, limbs_per);
      outputf (OUTPUT_TRACE, "index: 0 pA: (%Zd, %Zd)\n", x_final, z_final);

      to_mpz(x_final, datum + p2_word * limbs_per, limbs_per);
      to_mpz(z_final, datum + (p2_word + 1) * limbs_per, limbs_per);
      outputf (OUTPUT_TRACE, "index: 0 pB: (%Zd, %Zd)\n", x_final, z_final);
    }

    // Make sure we were testing the right number.
    to_mpz(modulo, datum + 0 * limbs_per, limbs_per);
    assert(mpz_cmp(modulo, N) == 0);

    to_mpz(x_final, datum + p1_word * limbs_per, limbs_per);
    to_mpz(z_final, datum + (p1_word + 1) * limbs_per, limbs_per);

    /* Suspicious only for param3, whose start point is literally (2, 1): see below.
       For param0 the start point is (u^3 : v^3), so the check cannot be applied --
       a "didn't compute" curve is caught by cgbn's error report instead. */
    if (p1_word == 1 &&
        mpz_cmp_ui (x_final, 2) == 0 && mpz_cmp_ui (z_final, 1) == 0) {
      errors += 1;
      if (errors < 10 || errors % 100 == 1)
        outputf (OUTPUT_ERROR, "GPU: curve %d didn't compute?\n", i);
    }

    array_found[i] = findfactor(factors[i], N, x_final, z_final);
    if (array_found[i] == ECM_ERROR) {
      if(degenerate_curve)*degenerate_curve=uint32_t(i);
      ++errors;
      array_found[i] = ECM_NO_FACTOR_FOUND;
      outputf(OUTPUT_ERROR, "GPU: degenerate final denominator on curve %zu (sigma %d:%llu); refusing final save\n",
              i, curve_param, (unsigned long long)(sigma + i));
      continue;
    }
    if (array_found[i] != ECM_NO_FACTOR_FOUND) {
      youpi = array_found[i];
      /* NOTE: the project's logger is plain vfprintf(), so gmp-ecm's %Zd is NOT
         supported -- it used to print a literal 'd' and drop the value.  Render the
         factor explicitly. */
      char *fac_str = mpz_get_str(NULL, 10, factors[i]);
      outputf (OUTPUT_NORMAL, "GPU: factor %s found in Step 1 with curve %zu (sigma %d:%llu)\n",
          fac_str ? fac_str : "?", i, curve_param, (unsigned long long)(sigma + i));
      free(fac_str);
    }
  }

  mpz_clear(modulo);
  mpz_clear(x_final);
  mpz_clear(z_final);

#ifdef IS_DEV_BUILD
  if (errors)
        outputf (OUTPUT_ERROR, "Had %d errors. Try `make clean; make` or reducing TPB_DEFAULT\n",
            errors);
#endif

  if (errors > 0)
      return ECM_ERROR;

  return youpi;
}


} // namespace ecm_cuda_mont_host
