#pragma once
#include "ecm/integer.h"
#include <array>
#include <cstdint>
#include <cstddef>
#include <stdexcept>

// Checked Atkin-Morain construction for x^2 + y^2 = 1 + d*x^2*y^2 (a = 1).
// Reference: .refactor/p95v3106b01.source/ecm truncated P-1.cpp,
// choose_atkin_morain(), especially its STRAIGHT_FORWARD branch.
// The affine parameter formulas agree; the auxiliary point here is Jacobian,
// whereas Prime95's ws_add/ws_dbl use ordinary projective coordinates.
// All inverse/singularity checks use the actual target, even with a Stage2 carrier.
namespace ecm_edwards {
enum class Setup { Ready, Factor, Degenerate };

// Invalid inputs and failed equation checks do not report gcd = n.
inline bool construction_is_degenerate(Setup setup, mpz_srcptr gcd, mpz_srcptr n) {
    return setup == Setup::Degenerate && mpz_cmp_ui(gcd, 1) > 0 && mpz_cmp(gcd, n) == 0;
}

inline bool point_is_identity(const mpz_t* p) {
    return !mpz_sgn(p[0]) && !mpz_sgn(p[3]) && mpz_sgn(p[2]) && !mpz_cmp(p[1], p[2]);
}

// A failed inverse may expose a proper factor. gcd = n instead denotes a
// degenerate construction, not a factor suitable for publication.
inline Setup inverse(mpz_t out, const mpz_t value, const mpz_t n, mpz_t factor) {
    if (mpz_invert(out, value, n))
        return Setup::Ready;
    mpz_gcd(factor, value, n);
    return mpz_cmp(factor, n) < 0 ? Setup::Factor : Setup::Degenerate;
}

// One worker owns this state for a single immutable Mersenne modulus.
struct MersenneConstruction {
    explicit MersenneConstruction(mpz_srcptr modulus) : n(modulus), bits(mpz_sizeinbase(n, 2)) {
        if (mpz_sgn(n) <= 0 || mpz_scan0(n, 0) != bits)
            throw std::invalid_argument("Edwards Mersenne construction requires a full Mersenne modulus");
    }

    struct Jacobian {
        Integer x, y, z;
    };
    const mpz_srcptr n;
    const mp_bitcnt_t bits;
    Integer high;
    std::array<Jacobian, 64> before_bit;
    uint64_t previous_sigma = 0;

    void reduce(mpz_ptr value) {
        if (bits < 8192) {
            mpz_mod(value, value, n);
            return;
        }
        const bool negative = mpz_sgn(value) < 0;
        if (negative) mpz_neg(value, value);
        while (mpz_sizeinbase(value, 2) > bits) {
            mpz_tdiv_q_2exp(high.v, value, bits);
            mpz_tdiv_r_2exp(value, value, bits);
            mpz_add(value, value, high.v);
        }
        if (mpz_cmp(value, n) == 0) mpz_set_ui(value, 0);
        if (negative && mpz_sgn(value)) mpz_sub(value, n, value);
    }
};

inline Setup construct(mpz_t d, mpz_t px, mpz_t py, mpz_t a24, mpz_t factor, uint64_t sigma,
                       const mpz_t n, MersenneConstruction* mersenne = nullptr) {
    if (mersenne && mersenne->n != n)
        throw std::invalid_argument("Edwards Mersenne construction modulus differs from its scratch state");
    mpz_set_ui(factor, 0);
    if (!sigma || mpz_cmp_ui(n, 3) <= 0 || !mpz_odd_p(n))
        return Setup::Degenerate;
    Integer X, Y, Z,                            // Auxiliary Jacobian point.
            A, B, C, S, M, U, V, H, J, K, T,    // Reused point-operation temporaries.
            inv, s, t, alpha, beta, h, num, den;
    auto reduce = [&](mpz_t r) {
        if (mersenne) mersenne->reduce(r);
        else mpz_mod(r, r, n);
    };
    auto mul = [&](mpz_t r, const mpz_t a, const mpz_t b) {
        mpz_mul(r, a, b);
        reduce(r);
    };
    auto sqr = [&](mpz_t r, const mpz_t a) {
        mul(r, a, a);
    };

    // Auxiliary curve: t^2 = s^3 - 8*s - 32, with base point Q = (12, 40).
    // Compute [sigma]Q in Jacobian coordinates: s = X/Z^2, t = Y/Z^3.
    // Z = 0 represents infinity. Arithmetic is modulo n; affine normalization
    // is delayed to avoid inverses during double-and-add.
    mpz_set_ui(Z.v, 0);
    auto double_point = [&]() {
        if (mpz_sgn(Z.v)) {
            // A = X^2, B = Y^2, C = Y^4, S = 4*X*Y^2.
            sqr(A.v, X.v);
            sqr(B.v, Y.v);
            sqr(C.v, B.v);
            mul(S.v, X.v, B.v);
            mpz_mul_ui(S.v, S.v, 4);
            reduce(S.v);

            // M = 3*X^2 + a*Z^4 = 3*A - 8*Z^4 (Weierstrass a = -8).
            sqr(M.v, Z.v);
            sqr(M.v, M.v);
            mpz_mul_ui(M.v, M.v, 8);
            mpz_mul_ui(T.v, A.v, 3);
            mpz_sub(M.v, T.v, M.v);
            reduce(M.v);

            // X' = M^2 - 2*S, kept in U until the old inputs are consumed.
            sqr(U.v, M.v);
            mpz_sub(U.v, U.v, S.v);
            mpz_sub(U.v, U.v, S.v);
            reduce(U.v);

            // Y' = M*(S - X') - 8*C.
            mpz_sub(V.v, S.v, U.v);
            mul(V.v, M.v, V.v);
            mpz_mul_ui(T.v, C.v, 8);
            mpz_sub(V.v, V.v, T.v);
            reduce(V.v);

            // Z' = 2*Y*Z; update X/Y only after using the old Y.
            mul(Z.v, Y.v, Z.v);
            mpz_mul_ui(Z.v, Z.v, 2);
            reduce(Z.v);
            mpz_set(X.v, U.v);
            mpz_set(Y.v, V.v);
        }
    };

    int bit = 63;
    if (mersenne && mersenne->previous_sigma) {
        const uint64_t diff = sigma ^ mersenne->previous_sigma;
        while (bit && !(diff >> bit)) --bit;
        const auto& shared = mersenne->before_bit[bit];
        mpz_set(X.v, shared.x.v);
        mpz_set(Y.v, shared.y.v);
        mpz_set(Z.v, shared.z.v);
    }
    // An early gcd inside the loop can leave later snapshots stale.
    if (mersenne) mersenne->previous_sigma = 0;
    // Left-to-right binary multiplication: P <- 2*P, then P <- P+Q for bit 1.
    for (; bit >= 0; --bit) {
        if (mersenne) {
            auto& state = mersenne->before_bit[bit];
            mpz_set(state.x.v, X.v);
            mpz_set(state.y.v, Y.v);
            mpz_set(state.z.v, Z.v);
        }
        double_point();
        if (!((sigma >> bit) & 1))
            continue;

        // Infinity + Q = Q, represented by (12, 40, 1).
        if (!mpz_sgn(Z.v)) {
            mpz_set_ui(X.v, 12);
            reduce(X.v);
            mpz_set_ui(Y.v, 40);
            reduce(Y.v);
            mpz_set_ui(Z.v, 1);
            continue;
        }

        // Mixed Jacobian/affine addition with Q = (12, 40):
        // U = 12*Z^2, V = 40*Z^3, H = U-X, J = V-Y.
        sqr(A.v, Z.v);
        mpz_mul_ui(U.v, A.v, 12);
        reduce(U.v);
        mul(B.v, A.v, Z.v);
        mpz_mul_ui(V.v, B.v, 40);
        reduce(V.v);
        mpz_sub(H.v, U.v, X.v);
        reduce(H.v);
        mpz_sub(J.v, V.v, Y.v);
        reduce(J.v);
        if (!mpz_sgn(H.v)) {
            // Same point: use doubling instead of generic addition.
            if (!mpz_sgn(J.v)) {
                double_point();
                continue;
            }

            // A nonunit J can expose a proper factor of the target.
            // Otherwise P and Q are opposite, so their sum is infinity.
            mpz_gcd(factor, J.v, n);
            if (mpz_cmp_ui(factor, 1) > 0 && mpz_cmp(factor, n) < 0)
                return Setup::Factor;
            mpz_set_ui(Z.v, 0);
            continue;
        }

        // A = H^2, B = H^3, K = X*H^2.
        sqr(A.v, H.v);
        mul(B.v, A.v, H.v);
        mul(K.v, X.v, A.v);

        // X' = J^2 - B - 2*K.
        sqr(U.v, J.v);
        mpz_sub(U.v, U.v, B.v);
        mpz_sub(U.v, U.v, K.v);
        mpz_sub(U.v, U.v, K.v);
        reduce(U.v);

        // Y' = J*(K - X') - Y*B; Z' = Z*H.
        mpz_sub(V.v, K.v, U.v);
        mul(V.v, J.v, V.v);
        mul(T.v, Y.v, B.v);
        mpz_sub(V.v, V.v, T.v);
        reduce(V.v);
        mul(Z.v, Z.v, H.v);
        mpz_set(X.v, U.v);
        mpz_set(Y.v, V.v);
    }
    if (mersenne) mersenne->previous_sigma = sigma;

    // Normalize once: (s, t) = (X/Z^2, Y/Z^3).
    Setup rc = inverse(inv.v, Z.v, n, factor);
    if (rc != Setup::Ready)
        return rc;
    sqr(T.v, inv.v);
    mul(s.v, X.v, T.v);
    mul(T.v, T.v, inv.v);
    mul(t.v, Y.v, T.v);

    // alpha = ((t+25)/(s-9) + 1)^(-1) = (s-9)/(t+s+16).
    mpz_sub_ui(num.v, s.v, 9);
    mpz_add(den.v, t.v, s.v);
    mpz_add_ui(den.v, den.v, 16);
    rc = inverse(inv.v, den.v, n, factor);
    if (rc != Setup::Ready)
        return rc;
    mul(alpha.v, num.v, inv.v);

    // beta = 2*alpha*(4*alpha+1)/(8*alpha^2-1).
    mpz_mul_ui(num.v, alpha.v, 4);
    mpz_add_ui(num.v, num.v, 1);
    mul(num.v, num.v, alpha.v);
    mpz_mul_ui(num.v, num.v, 2);
    sqr(den.v, alpha.v);
    mpz_mul_ui(den.v, den.v, 8);
    mpz_sub_ui(den.v, den.v, 1);
    rc = inverse(inv.v, den.v, n, factor);
    if (rc != Setup::Ready)
        return rc;
    mul(beta.v, num.v, inv.v);

    // h = 2*beta-1; Edwards d = (2*h^2-1)/h^4.
    mpz_mul_ui(h.v, beta.v, 2);
    mpz_sub_ui(h.v, h.v, 1);
    reduce(h.v);
    sqr(T.v, h.v);
    mpz_mul_ui(num.v, T.v, 2);
    mpz_sub_ui(num.v, num.v, 1);
    sqr(den.v, T.v);

    // Edwards starting x = h*(4*beta-3)/(6*beta-5).
    const auto x_numerator = [&] {
        mpz_mul_ui(num.v, beta.v, 4);
        mpz_sub_ui(num.v, num.v, 3);
        mul(num.v, num.v, h.v);
    };
    const auto x_denominator = [&](mpz_ptr r) {
        mpz_mul_ui(r, beta.v, 6);
        mpz_sub_ui(r, r, 5);
    };

    // Edwards starting y = h*(t^2 + 50*t - 2*s^3 + 27*s^2 - 104)
    //                      / ((t + 3*s - 2)*(t + s + 16)).
    const auto y_numerator = [&] {
        sqr(num.v, t.v);
        mpz_mul_ui(T.v, t.v, 50);
        mpz_add(num.v, num.v, T.v);
        sqr(T.v, s.v);
        mul(U.v, T.v, s.v);
        mpz_mul_ui(U.v, U.v, 2);
        mpz_sub(num.v, num.v, U.v);
        mpz_mul_ui(T.v, T.v, 27);
        mpz_add(num.v, num.v, T.v);
        mpz_sub_ui(num.v, num.v, 104);
        mul(num.v, num.v, h.v);
    };
    const auto y_denominator = [&](mpz_ptr r) {
        mpz_mul_ui(r, s.v, 3);
        mpz_add(r, r, t.v);
        mpz_sub_ui(r, r, 2);
        mpz_add(T.v, t.v, s.v);
        mpz_add_ui(T.v, T.v, 16);
        mul(r, r, T.v);
    };

    bool mapped = false;
    if (mersenne) {
        // With d = num/den, a24 = den/(den-num). The product also includes num
        // to check that d is a unit; den-num checks that 1-d is a unit.
        Integer a24_den, x_den, y_den;
        mpz_sub(a24_den.v, den.v, num.v);
        x_denominator(x_den.v);
        y_denominator(y_den.v);
        const std::array<mpz_srcptr, 4> divisors{den.v, a24_den.v, x_den.v, y_den.v};
        std::array<Integer, divisors.size()> products;
        mpz_set(products.front().v, divisors.front());
        for (size_t i = 1; i < divisors.size(); ++i)
            mul(products[i].v, products[i - 1].v, divisors[i]);
        mul(T.v, products.back().v, num.v);
        if (mpz_invert(inv.v, T.v, n)) {
            // Cancel num, then recover each reciprocal using the prefix products.
            mul(inv.v, inv.v, num.v);
            for (size_t i = divisors.size() - 1; i != 0; --i) {
                mul(products[i].v, inv.v, products[i - 1].v);
                mul(inv.v, inv.v, divisors[i]);
            }
            mul(d, num.v, inv.v);
            mul(a24, den.v, products[1].v);
            x_numerator();
            mul(px, num.v, products[2].v);
            y_numerator();
            mul(py, num.v, products[3].v);
            mapped = true;
        }
    }

    if (!mapped) {
        // Check nonunit products in the original order to preserve the result
        // and the exact reported gcd, including when different divisors fail.
        rc = inverse(inv.v, den.v, n, factor);
        if (rc != Setup::Ready)
            return rc;
        mul(d, num.v, inv.v);

        // For a = 1 and odd n, nonsingularity requires d*(1-d) to be a unit.
        mpz_ui_sub(den.v, 1, d);
        mul(T.v, d, den.v);
        mpz_gcd(factor, T.v, n);
        if (mpz_cmp_ui(factor, 1) != 0)
            return mpz_cmp(factor, n) < 0 ? Setup::Factor : Setup::Degenerate;

        // xDBL uses a24 = (A+2)/4 = 1/(1-d).
        rc = inverse(a24, den.v, n, factor);
        if (rc != Setup::Ready)
            return rc;

        x_numerator();
        x_denominator(den.v);
        rc = inverse(inv.v, den.v, n, factor);
        if (rc != Setup::Ready)
            return rc;
        mul(px, num.v, inv.v);

        y_numerator();
        y_denominator(den.v);
        rc = inverse(inv.v, den.v, n, factor);
        if (rc != Setup::Ready)
            return rc;
        mul(py, num.v, inv.v);
    }

    // Verify px^2 + py^2 - 1 - d*px^2*py^2 = 0 (mod n).
    sqr(U.v, px);
    sqr(V.v, py);
    mpz_add(T.v, U.v, V.v);
    mpz_sub_ui(T.v, T.v, 1);
    mul(U.v, U.v, V.v);
    mul(U.v, U.v, d);
    mpz_sub(T.v, T.v, U.v);
    reduce(T.v);
    if (mpz_sgn(T.v))
        return Setup::Degenerate;
    mpz_set_ui(factor, 0);
    return Setup::Ready;
}

// Extended Edwards point p = (X, Y, Z, T). For invertible Z, x = X/Z, y = Y/Z.
// This differs from the auxiliary Jacobian coordinate convention above.
inline bool point_valid(const mpz_t n, const mpz_t d, mpz_t *p) {
    for (unsigned j = 0; j < 4; ++j)
        if (mpz_sgn(p[j]) < 0 || mpz_cmp(p[j], n) >= 0)
            return false;
    if (!mpz_sgn(p[0]) && !mpz_sgn(p[1]) && !mpz_sgn(p[2]) && !mpz_sgn(p[3]))
        return false;
    Integer a, b, c;

    // Extended-coordinate consistency: X*Y = Z*T (mod n).
    mpz_mul(a.v, p[0], p[1]);
    mpz_mul(b.v, p[2], p[3]);
    mpz_sub(a.v, a.v, b.v);
    mpz_mod(a.v, a.v, n);
    if (mpz_sgn(a.v))
        return false;

    // Extended curve relation: X^2 + Y^2 = Z^2 + d*T^2 (mod n).
    // For invertible Z, dividing by Z^2 and using XY = ZT recovers
    // x^2 + y^2 = 1 + d*x^2*y^2. No cancellation by Z is performed here.
    mpz_mul(a.v, p[0], p[0]);
    mpz_mul(b.v, p[1], p[1]);
    mpz_add(a.v, a.v, b.v);
    mpz_mul(b.v, p[2], p[2]);
    mpz_sub(a.v, a.v, b.v);
    mpz_mul(c.v, p[3], p[3]);
    mpz_mul(c.v, c.v, d);
    mpz_sub(a.v, a.v, c.v);
    mpz_mod(a.v, a.v, n);
    return !mpz_sgn(a.v);
}

} // namespace ecm_edwards
