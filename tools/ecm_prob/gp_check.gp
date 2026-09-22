\\ Cross-validation of the pure-Python Edwards arithmetic against PARI/GP.
\\
\\ For an Edwards curve x^2 + y^2 = 1 + d x^2 y^2 with base point (x1, y1),
\\ map it to Weierstrass and compute the point order (and its factorization),
\\ then check B1-powersmoothness.  Used to independently confirm
\\ tools/ecm_prob/ecmath.py.
\\
\\ Birational map (Bernstein-Lange):
\\   Edwards -> Montgomery:  A = 2(1+d)/(1-d), B = 4/(1-d),
\\                           u = (1+y)/(1-y), v = u/x
\\   Montgomery -> Weierstrass y^2 = x^3 + a*x + b:
\\       a = (3 - A^2)/(3 B^2), b = (2 A^3 - 9 A)/(27 B^3),
\\       X = u/B + A/(3 B), Y = v/B
\\
\\ Usage (from gp):
\\   \\r gp_check.gp
\\   edwards_order(524287, -24167, 25, 5, 23, -1, 7)
\\
\\ d = dn/dd,  x1 = xn/xd,  y1 = yn/yd  (exact rationals)

edwards_to_weier(p, dn, dd, xn, xd, yn, yd) = {
  my(d, x1, y1, A, B, u, v, a, b, X, Y);
  d  = Mod(dn, p) / Mod(dd, p);
  x1 = Mod(xn, p) / Mod(xd, p);
  y1 = Mod(yn, p) / Mod(yd, p);
  A  = 2*(1+d) / (1-d);
  B  = 4 / (1-d);
  u  = (1+y1) / (1-y1);
  v  = u / x1;
  a  = (3 - A^2) / (3*B^2);
  b  = (2*A^3 - 9*A) / (27*B^3);
  X  = u/B + A/(3*B);
  Y  = v/B;
  [a, b, X, Y]
};

edwards_order(p, dn, dd, xn, xd, yn, yd) = {
  my(w, E, P, o);
  w = edwards_to_weier(p, dn, dd, xn, xd, yn, yd);
  E = ellinit([w[1], w[2]]);
  P = [w[3], w[4]];
  o = ellorder(E, P);
  [o, factor(o), ellcard(E), factor(ellcard(E))]
};

\\ Four paper Section 9.1 curves
edwards_order_Z4(p)     = edwards_order(p, 1, 3,   2, 1,  3, 1);
edwards_order_Z2xZ4(p)  = edwards_order(p, 1, 36,  8, 1,  9, 1);
edwards_order_Z12(p)    = edwards_order(p, -24167, 25, 5, 23, -1, 7);
edwards_order_Z2xZ8(p)  = edwards_order(p, 25921, 83521, 13, 7, 289, 49);

\\ Example (matches cross_check.py):
\\   edwards_order_Z12(524287)  ->  order 18732 = 2^2 * 3 * 7 * 223
