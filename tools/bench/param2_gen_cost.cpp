/* param2_gen_cost.cpp -- what does per-curve curve GENERATION cost on the host?
 *
 * Motivating question (docs/ECM_CGBN_OPTIMIZATION.md §5.6): our CUDA stage-1 kernel
 * prices the *param2*-shaped step at 78.37 M curve-bits/s against param0's 71.01 M
 * (+10.4%), so param2 ("batch 2", the 6-torsion batch family) would be the fastest of
 * the three parametrizations per curve.  But gmp-ecm's get_curve_from_param2()
 * (parametrizations.c:296) is by far the most expensive generator: it computes
 * sigma*(-3:3:1) on the fixed curve y^2 = x^3 + 36 with a recursive addition chain and
 * then needs THREE modular inversions (lines 323, 338, 361), against param3's single
 * 32-bit constant multiply.  This tool measures that host-side cost per curve at a
 * given N size so the kernel gain can be weighed against it.
 *
 * It is a *cost model*, not a transcription: the scalar multiply uses textbook Jacobian
 * double-and-add on y^2 = x^3 + 36 instead of gmp-ecm's specialised addchain_param(),
 * while the derivation of A from x3 (parametrizations.c:330-371) is transcribed op for op.
 *
 * usage: param2_gen_cost.exe [curves] [repeats]        (N in decimal on stdin)
 */
#include <gmp.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

static double now_ms(void) {
  struct timespec ts;
  timespec_get(&ts, TIME_UTC);
  return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

/* ---- y^2 = x^3 + 36, Jacobian (X:Y:Z), a = 0 ------------------------------------ */
/* Point at infinity is Z = 0.  P = (-3 : 3 : 1) is the fixed 6-torsion base point. */

static void jac_dbl(mpz_t X3, mpz_t Y3, mpz_t Z3,
                    const mpz_t X1, const mpz_t Y1, const mpz_t Z1, const mpz_t n) {
  /* dbl-2009-l style, a = 0: 2M + 5S */
  mpz_t A, B, C, D, t1, t2;
  mpz_inits(A, B, C, D, t1, t2, NULL);
  mpz_mul(A, X1, X1); mpz_mod(A, A, n);                    /* A = X1^2 */
  mpz_mul(B, Y1, Y1); mpz_mod(B, B, n);                    /* B = Y1^2 */
  mpz_mul(C, B, B); mpz_mod(C, C, n);                      /* C = B^2 */
  mpz_add(D, X1, B); mpz_mul(D, D, D); mpz_mod(D, D, n);
  mpz_sub(D, D, A); mpz_sub(D, D, C); mpz_mod(D, D, n);
  mpz_mul_ui(D, D, 2); mpz_mod(D, D, n);                   /* D = 2((X1+B)^2 - A - C) */
  mpz_mul_ui(t1, A, 3); mpz_mod(t1, t1, n);                /* 3A */
  mpz_mul(t2, t1, t1); mpz_mod(t2, t2, n);                 /* 9A^2 */
  mpz_mul_ui(t1, D, 2); mpz_mod(t1, t1, n);
  mpz_sub(X3, t2, t1); mpz_mod(X3, X3, n);                 /* X3 = 9A^2 - 2D */
  mpz_sub(t1, D, X3); mpz_mod(t1, t1, n);
  mpz_mul(t1, t1, t2); mpz_mod(t1, t1, n);
  mpz_mul_ui(t2, C, 8); mpz_mod(t2, t2, n);
  mpz_sub(Y3, t1, t2); mpz_mod(Y3, Y3, n);                 /* Y3 = D(9A^2-X3) - 8C */
  mpz_mul(t1, Y1, Z1); mpz_mod(Y3, Y3, n);                 /* Y1*Z1 ... */
  mpz_mul(Y3, Y3, t1); mpz_mod(Y3, Y3, n);                 /* Y3 = Y1*Z1*(...) */
  mpz_mul(t1, Y1, Y1); mpz_mod(t1, t1, n);
  mpz_mul_ui(Z3, t1, 2); mpz_mod(Z3, Z3, n);               /* Z3 = 2B */
  mpz_clears(A, B, C, D, t1, t2, NULL);
}

/* mixed addition: (X1:Y1:Z1) += (x2:y2:1), a = 0, standard formulas */
static void jac_add_affine(mpz_t X1, mpz_t Y1, mpz_t Z1,
                           const mpz_t x2, const mpz_t y2, const mpz_t n) {
  mpz_t Z1Z1, U2, S2, H, r, t1, t2, t3, t4;
  mpz_inits(Z1Z1, U2, S2, H, r, t1, t2, t3, t4, NULL);
  mpz_mul(Z1Z1, Z1, Z1); mpz_mod(Z1Z1, Z1Z1, n);
  mpz_mul(U2, x2, Z1Z1); mpz_mod(U2, U2, n);
  mpz_mul(S2, y2, Z1); mpz_mod(S2, S2, n);
  mpz_mul(S2, S2, Z1Z1); mpz_mod(S2, S2, n);
  mpz_sub(H, U2, X1); mpz_mod(H, H, n);
  mpz_sub(r, S2, Y1); mpz_mod(r, r, n);
  mpz_mul(t1, H, H); mpz_mod(t1, t1, n);                   /* H^2 */
  mpz_mul(t2, t1, H); mpz_mod(t2, t2, n);                  /* H^3 */
  mpz_mul(t3, t1, X1); mpz_mod(t3, t3, n);
  mpz_mul_ui(t3, t3, 2); mpz_mod(t3, t3, n);
  mpz_mul(t4, r, r); mpz_mod(t4, t4, n);
  mpz_sub(t4, t4, t2); mpz_mod(t4, t4, n);
  mpz_sub(t4, t4, t3); mpz_mod(t4, t4, n);                 /* X3 */
  mpz_sub(t3, t3, t4); mpz_mod(t3, t3, n);
  mpz_mul(t3, t3, r); mpz_mod(t3, t3, n);
  mpz_mul(t2, Y1, t2); mpz_mod(t2, t2, n);
  mpz_sub(t3, t3, t2); mpz_mod(t3, t3, n);                 /* Y3 */
  mpz_mul(t1, Z1, H); mpz_mod(Z1, t1, n);                  /* Z3 */
  mpz_set(X1, t4); mpz_set(Y1, t3);
  mpz_clears(Z1Z1, U2, S2, H, r, t1, t2, t3, t4, NULL);
}

/* k * (-3:3:1) on y^2 = x^3 + 36, plain left-to-right double-and-add */
static void scalar_mul_base(mpz_t rx, mpz_t ry, mpz_t rz, mpz_t k,
                            const mpz_t n, mpz_t bx, mpz_t by) {
  mpz_t X, Y, Z;
  mpz_inits(X, Y, Z, NULL);
  mpz_set_ui(X, 0); mpz_set_ui(Y, 1); mpz_set_ui(Z, 0);    /* infinity */
  bool set = false;
  for (int i = mpz_sizeinbase(k, 2) - 1; i >= 0; i--) {
    if (set) jac_dbl(X, Y, Z, X, Y, Z, n);
    if (mpz_tstbit(k, i)) {
      if (!set) {
        /* first set bit: initialise from the affine base point */
        mpz_set(X, bx); mpz_set(Y, by); mpz_set_ui(Z, 1); set = true;
      } else {
        jac_add_affine(X, Y, Z, bx, by, n);
      }
    }
  }
  mpz_set(rx, X); mpz_set(ry, Y); mpz_set(rz, Z);
  mpz_clears(X, Y, Z, NULL);
}

/* ---- transcribed from gmp-ecm parametrizations.c:323-373 ------------------------ */

static void invert_or_die(mpz_t r, const mpz_t a, const mpz_t n) {
  if (mpz_invert(r, a, n) == 0) mpz_set_ui(r, 1);          /* gcd != 1 would stop the real code */
}

static void gen_param2_once(mpz_t A_out, mpz_t x0_out, const mpz_t N, unsigned long sigma) {
  mpz_t x, y, z, t, u, v, w, k;
  mpz_inits(x, y, z, t, u, v, w, k, NULL);
  mpz_set_ui(k, sigma);
  mpz_set_si(x, -3); mpz_set_ui(y, 3); mpz_set_ui(z, 1);

  scalar_mul_base(x, y, z, k, N, x, y);                    /* (x:y:z) = sigma*P */
  mpz_mod(x, x, N); mpz_mod(y, y, N); mpz_mod(z, z, N);

  invert_or_die(u, z, N);                                  /* inversion 1 */
  mpz_mul(v, u, u); mpz_mod(v, v, N);
  mpz_mul(u, v, u); mpz_mod(u, u, N);
  mpz_mul(x, x, v); mpz_mod(x, x, N);
  mpz_mul(y, y, u); mpz_mod(y, y, N);

  mpz_sub_ui(t, y, 3); mpz_mod(t, t, N);
  mpz_mul_ui(t, t, 2); mpz_mod(t, t, N);
  invert_or_die(u, t, N);                                  /* inversion 2 */

  mpz_mul_ui(w, x, 3); mpz_mod(w, w, N);
  mpz_add(w, w, y); mpz_mod(w, w, N);
  mpz_add_ui(w, w, 6); mpz_mod(w, w, N);
  mpz_mul(x, w, u); mpz_mod(x, x, N);                      /* x3 */

  /* A = -(3*x3^4 + 6*x3^2 - 1) / (4*x3^3) */
  mpz_mul(u, x, x); mpz_mod(u, u, N);                      /* x3^2 */
  mpz_mul(v, u, x); mpz_mod(v, v, N);                      /* x3^3 */
  mpz_mul(w, u, u); mpz_mod(w, w, N);                      /* x3^4 */
  mpz_mul_ui(u, u, 6); mpz_mod(u, u, N); mpz_neg(u, u);
  mpz_mul_ui(v, v, 4); mpz_mod(v, v, N);
  mpz_mul_ui(w, w, 3); mpz_mod(w, w, N); mpz_neg(w, w);
  invert_or_die(t, v, N);                                  /* inversion 3 */
  mpz_add(w, w, u); mpz_mod(w, w, N);
  mpz_add_ui(w, w, 1); mpz_mod(w, w, N);
  mpz_mul(A_out, w, t); mpz_mod(A_out, A_out, N);
  mpz_set_ui(x0_out, 2);                                   /* parametrizations.c:373 */

  mpz_clears(x, y, z, t, u, v, w, k, NULL);
}

/* param0-shaped generation, a COST MODEL of gmp-ecm's get_curve_from_param0(): the Suyama
 * map u = sigma^2-5, v = 4*sigma followed by ~10 mults/squares, plus ONE modular
 * inversion (mpres_invert of b*z for the affine normalisation) and a gcd check.  It is
 * here only to give the param2 comparison a measured baseline; it is not a transcription. */
static void gen_param0_shaped_once(mpz_t A_out, mpz_t x0_out, const mpz_t N, unsigned long sigma) {
  mpz_t u, v, w, t, inv;
  mpz_inits(u, v, w, t, inv, NULL);
  mpz_set_ui(u, sigma); mpz_mul(u, u, u); mpz_mod(u, u, N); mpz_sub_ui(u, u, 5);
  mpz_set_ui(v, sigma); mpz_mul_ui(v, v, 4);
  for (int i = 0; i < 10; i++) {                 /* ~10 field ops of the parametrisation */
    mpz_mul(w, u, v); mpz_mod(w, w, N);
    mpz_mul(t, v, v); mpz_mod(t, t, N);
    mpz_add(u, u, w); mpz_mod(u, u, N);
    mpz_set(v, t);
  }
  invert_or_die(inv, v, N);                      /* the single inversion */
  mpz_mul(A_out, u, inv); mpz_mod(A_out, A_out, N);
  mpz_set_ui(x0_out, 2);
  mpz_clears(u, v, w, t, inv, NULL);
}

/* param3-shaped generation, for the comparison: A = 4*(sigma/2^32) - 2, x0 = 2 */
static void gen_param3_once(mpz_t A_out, mpz_t x0_out, const mpz_t N, unsigned long sigma) {
  mpz_t d, inv;
  mpz_inits(d, inv, NULL);
  mpz_set_ui(d, 1);
  mpz_mul_2exp(d, d, 32);
  invert_or_die(inv, d, N);
  mpz_mul_ui(d, inv, sigma);
  mpz_mod(d, d, N);
  mpz_mul_ui(d, d, 4); mpz_mod(d, d, N);
  mpz_sub_ui(A_out, d, 2); mpz_mod(A_out, A_out, N);
  mpz_set_ui(x0_out, 2);
  mpz_clears(d, inv, NULL);
}

int main(int argc, char **argv) {
  long curves = (argc > 1) ? atol(argv[1]) : 2000;
  int  reps   = (argc > 2) ? atoi(argv[2]) : 3;

  mpz_t N; mpz_init(N);
  if (mpz_inp_str(N, stdin, 10) == 0) { fprintf(stderr, "no N on stdin\n"); return 2; }

  mpz_t A, x0, sink; mpz_inits(A, x0, sink, NULL);
  printf("param2_gen_cost: N = %zu bits, %ld curves per repeat, %d repeats\n",
         mpz_sizeinbase(N, 2), curves, reps);

  const char *names[3] = {"param3-shaped (1 constant inversion + 1 mult)",
                          "param0-shaped model (1 inversion + ~12 mults)",
                          "param2 (gmp-ecm get_curve_from_param2)"};
  for (int which = 0; which < 3; which++) {
    double best = 1e30;
    for (int r = 0; r < reps; r++) {
      double t0 = now_ms();
      for (long i = 0; i < curves; i++) {
        unsigned long sigma = 2 + (unsigned long)((i * 2654435761u + r * 40503u) % 4000000000u);
        switch (which) {
          case 0: gen_param3_once(A, x0, N, sigma); break;
          case 1: gen_param0_shaped_once(A, x0, N, sigma); break;
          default: gen_param2_once(A, x0, N, sigma); break;
        }
        mpz_xor(sink, sink, A);          /* defeat dead-code elimination */
      }
      double dt = now_ms() - t0;
      if (dt < best) best = dt;
    }
    printf("  %-52s %9.3f ms/curve   [checksum %lu]\n",
           names[which], best / (double)curves, mpz_get_ui(sink));
  }
  mpz_clears(N, A, x0, sink, NULL);
  return 0;
}
