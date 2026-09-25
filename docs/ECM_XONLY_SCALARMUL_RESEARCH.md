# X-only scalar multiplication for fixed public scalars: can anything beat 1 doubling + 1 addition per bit?

**Scope.** Targets a CUDA ECM stage-1 kernel computing [s]P, s = lcm(1..B1), B1 = 1e5
(bits(s) = 144352, not 144000 — see §9), on a Montgomery curve in x-only XZ
coordinates, fixed public scalar (no SIMT divergence), CGBN field ops with S = M.

**Verdict up front.** No. For x-only Montgomery arithmetic with a fixed affine
difference, 1 doubling + 1 differential addition per bit is not merely what the
plain ladder happens to cost — it is the structural floor, and every published
alternative (PRAC/Lucas chains, co-Z, windowed, double-base) is *worse* in this
cost model. The single real lever is the constant multiply inside the doubling,
which you already exploit. Details, exact counts, and citations below.

---

## 1. Notation, cost model, and a calibration warning

**EFD formula names and costs that actually exist** (I verified these against the
rendered EFD page; see §10 for what does *not* exist):

| EFD name | Assumption | Cost | Meaning for you |
|---|---|---|---|
| `mdbl-1987-m` | Z1=1 | 1M+2S+1\*a (+3add+1\*4) | doubling an affine point, using curve `a` |
| `dbl-1987-m-3` | 4·a24=a+2 | 2M+2S+1\*a24+4add | general doubling |
| `dbl-1987-m-2` | 4·a24=a+2 | 4M+3S+1\*a24 | same, no CSE |
| `mdadd-1987-m` | Z1=1 | 3M+2S+6add | **affine** difference (Z_D=1) |
| `dadd-1987-m-3` | — | 4M+2S+6add | **projective** difference |
| `dadd-1987-m` | — | 6M+2S+2add | projective difference, alternative form |
| `mladd-1987-m` | Z1=1, 4·a24=a+2 | 5M+4S+1\*a24 | fused dbl+dadd, affine diff |
| `ladd-1987-m-3` | 4·a24=a+2 | 6M+4S+1\*a24 | fused dbl+dadd, projective diff |

**Two corrections to your premises here, and they matter.**

1. Your §2 says "EFD `dadd-1987-m-3` with a projective difference is 4M+2S = 6 M-units
   vs my 3.86". The *number* 4M+2S is right for the projective difference, but the
   naming is inverted: in EFD, `dadd-1987-m-3` **is** the 4M+2S projective-difference
   form (it begins `X5 = Z1*(DA+CB)^2`), and the **affine**-difference variant is
   `mdadd-1987-m` at **3M+2S**. The extra 1M is exactly the `Z1*`/`X1*` multiply you
   save by keeping X_D = Z_D = 1.

2. **Your addition is more expensive than 3.86 M-units.** Verbatim `mdadd-1987-m`:

   ```
   A = X2+Z2 ; B = X2-Z2 ; C = X3+Z3 ; D = X3-Z3
   DA = D*A ; CB = C*B
   X5 = (DA+CB)^2
   Z5 = X1*(DA-CB)^2
   ```

   That is DA, CB, (DA+CB)², (DA−CB)², **and X1·** — five field multiplications
   (5 M-units at S=M), not four. Getting 3.86 requires treating the `X1*` multiply
   as free, which is only legitimate if you pre-scale so that X1 = 1 *and* fold the
   multiply into a neighbouring squaring (possible, but then you owe one extra
   multiply elsewhere, or you have made an algebraic claim that needs checking).
   Likewise your doubling at 4.24 = 2M+2S+0.24m assumes a24 costs a quarter of an M.

   **Consequence:** your measured 8.10 M-units/bit is, under a strict S=M, no-free-lunch
   reading of EFD, an *optimistic* figure — strict EFD accounting for
   `mdbl-1987-m` (1M+2S+1a) + `mdadd-1987-m` (3M+2S) gives **9.0 M-units/bit**.
   I could not reproduce 3.86 from any EFD formula without assuming a free multiply.
   **Action item: re-derive your two measured constants against the formula text above
   before trusting any conclusion that depends on the 8.10 vs 9.0 gap.** Every result
   below is therefore stated twice where it matters: in *your* units (baseline 8.10)
   and in *strict EFD* units (baseline 9.0).

**Cost model used for the candidates** (M = 1, S = 1):

- x-only doubling, affine input (Z1=1): `mdbl` = 1M+2S+1m ≈ **3.0–4.24** (yours)
- x-only differential add, **affine** difference: `mdadd` = 3M+2S = **5.0** strict
  (you measure **3.86**)
- x-only differential add, **projective** difference: `dadd-1987-m-3` = 4M+2S = **6.0**
- co-Z ZADDU: 5M+2S = **7.0** (Goundar et al. 2011, Table 1)
- co-Z ZADDU′ (discard Z): 4M+2S = **6.0**
- co-Z ZADDC: 6M+3S = **9.0**; ZADDC′: 5M+3S = **8.0**
- co-Z ZDAU / ZACAU: 9M+7S = **16.0**; ZDAU′/ZACAU′: 8M+6S = **14.0**
- GMP-ECM's own internal pricing of its own primitives:
  `#define ADD 6.0 /* multiplications in an addition */`,
  `#define DUP 5.0 /* multiplications in a duplicate */`

---

## 2. Why x-only windowing is structurally impossible (the core structural result)

I want to state this as a lemma, because it kills most of the candidate list outright.

> **Lemma (x-only difference requirement).** In XZ coordinates over a Montgomery
> curve, x(P+Q) is computable from x(P), x(Q) **and nothing else** only if you also
> supply x(P−Q). There is no x-only "plain addition".

This is Montgomery 1987's whole point, and every implementation obeys it. GMP-ECM's
`add3()` header comment states the contract verbatim:

> "adds Q=(x2:z2) and R=(x1:z1) and puts the result in (x3:z3), using 6 muls (4 muls
> and 2 squares), and 6 add/sub. **One assumes that Q-R=P or R-Q=P where P=(x:z).**"

Bernstein (2006) states it as the defining property of a *differential* addition chain:

> "A differential addition chain is an addition chain in which each sum is already
> accompanied by a difference: i.e., whenever a new chain element P + Q is formed by
> adding P and Q, the difference P − Q was already in the chain."

> **Corollary (window illegality).** A w-bit window / w-NAF scheme adds
> `Q + dP` where `d ∈ {1,3,…,2^w−1}` is chosen per window. The required difference is
> `x(Q − dP)`. During the w shifts the accumulator becomes `2^j·k`, so the needed
> difference is `(2^j k − d)P` — a new, position-dependent point for every window and
> every digit. It is in no precomputed table, and it cannot be derived from x-only
> data. Precomputing {dP} does **not** help: you would need the whole difference set
> {(d−d′)P}, and you would still need x(2^j k − d) after the shifts.

**This is a corpus-wide negative result, not just my argument.** A systematic search
of EFD, eprint, JAIST, HAL, theses and the co-Z literature found **no paper doing
x-only windowed / w-NAF scalar multiplication with a precomputed table on Montgomery
curves.** The literature does the opposite: Goundar–Joye–Miyaji–Rivain–Venelli
*explicitly reject* NAF — they note NAF's "average density of non-zero digits of only
1/3" and discard it because it has a zero digit, replacing it with a **zeroless**
signed-digit (ZSD) representation with digits in {−1,+1}: still **one digit per bit,
no table**. And Meloni's co-Z windowing work is all done in full (X,Y,Z) Jacobian
coordinates, never x-only.

**So the answer to "is w-NAF legal in x-only?" is no, and the published record agrees.**

---

## 3. Claim-by-claim verification of your analysis

### Claim 1 — "Any 2-value x-only state with a fixed affine difference needs exactly 1 doubling + 1 differential addition per bit." **CONFIRMED.**

For the left-to-right ladder on digit i, from state (A, B) = (kP, (k+1)P) with B−A = P:
- i = 0: next state is (2A, 2A+P) = (2kP, (2k+1)P) → 1 doubling (2A) + 1 differential add (2A+P, difference P).
- i = 1: next state is (A+B, 2B) = ((2k+1)P, (2k+2)P) → 1 differential add (A+B, difference P) + 1 doubling (2B).

Both branches: exactly 1 doubling + 1 differential addition. And this is not just
forced for the ladder — it is forced for **any** 2-point state:
(a) one doubling per bit is a hard information-theoretic floor (each step must at
least double the working magnitude to cover one more bit), and
(b) since x-only general addition does not exist, the second point can only be
produced by a *differential* addition, which costs at least `mdadd` = 3M+2S.
So the floor for a 2-point x-only state is `mdbl + mdadd` = 4M+4S+1m.

The only escape from (b) would be a second **doubling**, but then the state contains
only {2A, 2B} whose difference is 2P — you have lost the ability to form A+B at all
(you would need x(2P) as the difference of 2A and 2B, and then you are tracking a
different chain). Signed digits do not help: NAF reduces the *number of nonzero
digits*, but in a 2-value state you must materialise both new values every iteration
regardless of the digit, so the addition count stays exactly 1/bit. **Your claim 1 stands.**

### Claim 2 — "To get fewer additions per bit you need a larger state; then differences become arbitrary projective points, costing 4M+2S = 6 M-units." **CONFIRMED, and stronger than you stated.**

Confirmed by the Corollary above: a larger state is *not merely expensive*, it is
**illegal** unless every difference you ever need is itself a maintained chain value
(a Lucas / differential-addition chain). And in a real Lucas chain the difference
genuinely is projective, never affine — Montgomery's PRAC invariant, verbatim
(eq. 5.1 of the 1992 manuscript): `A = Xa(P); B = Xb(P); C = X_{a−b}(P)` — `C` is a
live register with an arbitrary Z. Bernstein, on exactly this point:

> "Computing the x-coordinate of P + Q from the x-coordinates of P, Q, P − Q takes only
> 5 field multiplications **if the x-coordinate of P − Q has denominator 1**. This
> observation motivates considering differential addition chains with a limited set of
> differences P − Q, allowing the denominators to be replaced by 1 at the cost of a
> limited number of field divisions."

That is precisely your ladder: it is the *only* interesting chain whose difference set
is a single affine point. **PRAC does not satisfy it** — its differences are arbitrary
chain terms (2A, 3A, A−B, X_{−1}), never affine. So your claim 2 is right, and the
projective penalty is unavoidable in every large-state scheme.

### Claim 3 — "Pricing PRAC gives ~1 doubling + ~0.7 additions per bit, additions ~6 M-units, total ~8.4 M-units/bit, a small loss vs 8.1." **REFUTED on the density, CONFIRMED on the conclusion.**

Your density is wrong and your conclusion is right — for a different reason than you gave.

**Measured density (best available data, from the PRAC research pass).** Bernstein's
2009 ECM speed-record slides, measuring GMP-ECM 6.2.3 on the real stage-1 scalar at
B1 = 10^6 (b = 1,442,099 bits):

> "P → sP is computed using 2001915 (1.38820 b) DADD + 194155 (0.13463 b) DBL.
> These DADDs use 8590140 M + 4392140 S + 12788124 add."

The labels are swapped relative to the true counts, verified arithmetically:
6·2001915 + 5·194155 = 12,982,265 ≈ 8,590,140 + 4,392,140 = 12,982,280, and
structurally (≈78,567 `prac()` calls → 2,001,915/78,567 = 25.48 additions per call vs
average log2 p ≈ 18.37). So:

- **additions/bit = 1.388**, doublings/bit = 0.135 — i.e. **1 DBL + 1.39 ADD per bit**, not 1 + 0.7.
- Cross-check: Montgomery's own 1992 §11: "For large random n, Algorithm PRAC seems to
  generate a Lucas chain of length about 1.6 log2 n, but its length occasionally exceeds
  2 log2 n." For primes < 10^6 Montgomery's Table 5 gives PRAC 2,278,430 f-evaluations
  vs binary 2,755,571 (PRAC 17.3% fewer), the Theorem-8 lower bound being 2,114,698.

**Cost with that density:**

| pricing | per bit |
|---|---|
| your units (4.24/3.86) | 1.39·3.86 + 0.135·4.24 = **5.94** ← *naive, WRONG (assumes affine difference)* |
| your units, projective addition at 6.0 | 1.39·**6.0** + 0.135·4.24 = **8.92** |
| strict EFD (6.0 add, 3.0 dbl) | 1.39·6 + 0.135·3 = **8.75** |
| GMP-ECM's own pricing (ADD=6, DUP=5) | 1.39·6 + 0.135·5 = **9.02** |

So PRAC ≈ 8.9–9.0 M-units/bit vs your baseline 8.10, i.e. **~10–11% slower**, or roughly
break-even against a strict-EFD baseline of 9.0. **Your conclusion (small loss) is right;
your stated mechanism (0.7 additions/bit) is not. The real reason PRAC loses is that its
1.39 additions/bit are all projective-difference additions at 6 M-units, versus your 1
addition/bit at ~3.9–5.0.**

**Break-even threshold.** With d = 0.135 doublings/bit you win only if
`6·a + 4.24·0.135 < 8.10`, i.e. **a < 0.86 additions/bit**. PRAC is at 1.39 — you would
need a chain 60% shorter than PRAC, and the one-dimensional lower bound is ~1.44
additions/bit (Montgomery's Fibonacci/Theorem-7 bound, restated by Bernstein as
"(log 2)/log((1+√5)/2) ≈ 1.44042 additions per bit" for one-dimensional differential
chains). **The floor is 1.44 additions/bit, so a < 0.86 is unreachable by any Lucas chain.**
That is the clean proof that no chain-restructuring beats you.

### Claim 4 — "Skipping additions during zero-digit runs is pointless because the difference must itself be advanced." **CONFIRMED.**

With z skipped additions (pure doublings) the pair `(A, B) = (kP, (k+1)P)` becomes
`(2^z k P, (2^z k + 2^z)P)`, whose difference is `2^z P`, not `P`. To restore the affine
difference you must supply `x(2^z P)`, and since `2^z P = 2^z · (2^{z-1} P)`, the cheapest
route is z successive doublings at ~4.24 M-units each — the same order as the z additions
(3.86–5.0 each) you skipped. **No gain.** (Note the extra irony: in the ladder the
"difference is advanced" for free, because the doubling *is* the advancing and the
difference point itself never moves — that is why the ladder is efficient, and why
breaking the run structure destroys the free lunch.)

---

## 4. Candidate schemes: exact enumeration with all invariant-restoring costs

For each: avg doublings/bit (D), avg additions/bit (A), cost per addition, whether the
difference is affine, total M-units/bit, and speedup vs baseline 8.10 (your units).

### 4.1 Plain binary ladder, affine difference — **BASELINE**
- **D = 1.0, A = 1.0.** Doubling `mdbl-1987-m` (Z1=1) = 1M+2S+1m. Add `mdadd-1987-m`
  (Z1=1, X_D=Z_D=1) = 3M+2S (strict) — **difference AFFINE**.
- Invariant maintenance: **zero**. The difference `B − A = P` is provably invariant across
  both ladder branches: if `B − A = P` then `2B − (A+B) = B − A = P` and
  `(A+B) − 2A = B − A = P`. No normalization, no dictionary, no reconstruction.
- Cost: **8.10 (yours) / 9.0 (strict EFD)** M-units/bit. Speedup 1.00×.

### 4.2 PRAC / 1-D Lucas chains — **WORSE**
- **D = 0.135, A = 1.388** (measured, Bernstein 2009, B1=10^6). Difference **PROJECTIVE**
  (arbitrary chain terms).
- Invariant maintenance: PRAC carries **three** live XZ registers (A, B, C with
  `C = X_{a−b}`, `A = [d]P + [e]B`), plus temporaries. GMP-ECM's newer precomputed-table
  path makes the backward reach a data-structure field and ASSERTs it:
  `ASSERT((dif == Lchain[chain_length-k].value) && (k < 15))` — i.e. **16 XZ registers
  (32 mpz values) is the proven minimum register file** for a replayable precomputed chain.
  This is a hard memory cost per CUDA thread, on top of the arithmetic.
- Cost: `1.388·6 + 0.135·(4.24 or 5)` = **8.9 / 9.0** M-units/bit. Speedup **0.90–0.91×**
  (i.e. 10% slower).
- Source-verified details: `prac()` and `lucas_cost()` are **static functions in ecm.c**
  (there is no `prac.c`; no `prac_chain`/`prac_chain2`), GMP-ECM tries 10 α values
  `{1/φ, …}` and keeps the cheapest, and drives stage 1 as: plain doublings for 2^k,
  plain ×3 for 3^k, then `prac()` per prime, then repeated `prac()` per prime power.
  Prime95's PRAC is `lucas_mul()` in **`gwnum/ecmstag1.c`**, not `ecm.cpp` (whose
  `ecm.cpp` only holds an unreferenced `int PRAC_SEARCH = 7;`).
- GMP-ECM's own `lucas.c` (P+1) notes:
  > "we used to use several (4) values of 'val', but: (1) the code to estimate the best
  > value was buggy; (2) even after fixing the bug, the overhead to choose the best value
  > was larger than the corresponding gain (for a c155 and B1=10^7)."
  (Zimmermann measures the 10-α gain as only **3.72%** at B1=10^6.)

### 4.3 Montgomery / Tsuruoka / Bleichenbacher / Euclid / CFRC chains (short 1-D chains) — **WORSE**
- Best published one-dimensional densities: **1.56 additions/bit** for 256-bit primes
  (Bernstein 2006 experiments: 399.286 additions on average for 256-bit primes), with the
  one-dimensional lower bound at 1.440 additions/bit. Bernstein's own comparison table row
  reads `1 | standard | 1.533 | 1.560 | 8.885 | 8.983 | no`.
- Note Bernstein's "field mults per bit" weights for the elliptic-curve context are
  `(1336, 1093, 905)` ≈ `905(1.476, 1.208, 1)` from Curve25519 numbers — i.e. his 8.885/bit
  figure is **already** for a difference-with-denominator-1 addition. That is exactly your
  affine-difference optimization, and even with it the best is 8.9 vs 9.0 strict — a ~1%
  gain, not 20%.
- Cost: `1.44–1.56 · (5 or 6) + 1·(3–4.24)` = **10–13** M-units/bit. Speedup **0.6–0.8×**.
- Also: finding these chains is *expensive*. Meloni reports random Euclidean addition
  chains average ~7000 steps for 160 bits in theory, ~2500 in practice, and that a 160-bit
  chain of length 320 needs ~30 trial gcd's while length 270 needs >45,000 — hence
  "one should not expect to use EAC whose length is shorter than 320", and his own table
  shows the Montgomery ladder (1463 field mults) beating EAC-320 (1512–1792). For a large
  state you also pay large **memory per thread**, which a CUDA ECM kernel cannot afford.

### 4.4 co-Z addition / co-Z ladders (Meloni; Goundar–Joye–Miyaji; GJM+Rivain+Venelli) — **WORSE, and to be handled carefully**
**Important framing correction: co-Z is the *non-Montgomery* competitor to XZ.** All
co-Z literature is defined for short Weierstrass / Jacobian coordinates. There is **no
co-Z-for-Montgomery-curves body of work**, and EFD has **no co-Z/ZADD entries at all**
(see §10).

- Co-Z costs (GJM+Rivain+Venelli, *J. Cryptogr. Eng.* 1(2):161–176, 2011, Table 1):
  ZADDU 5M+2S, (X,Y)-only ZADDU′ 4M+2S, ZADDC 6M+3S, ZADDC′ 5M+3S, ZDAU 9M+7S,
  ZDAU′ 8M+6S, ZACAU 9M+7S, ZACAU′ 8M+6S, DBLU 1M+5S, TPLU 6M+7S.
- Their own ladder comparison (Table 2): X-only Montgomery ladder `n(9M+7S) + 1I + 14M + 3S`;
  (X,Y)-only co-Z ladder `n(8M+6S) + 1I + 1M`. Since S=M here, co-Z gives 14 M-units/bit
  vs the Montgomery ladder's 16 — but your affine-difference ladder is **9.0**. So **co-Z
  is ~1.5× worse than what you already have.** Meloni says so himself:
  > "The computational cost of this addition is 4M and 2S, which is lower than with our formula."
- The genuinely interesting co-Z facts, for completeness: (i) the ZADDU output point *and*
  the updated base point share one Z3, so the invariant `Z(P) = Z(Q)` costs nothing to
  maintain; (ii) ZADDC yields P+Q **and** P−Q sharing one Z3 for only 1M+1S extra, which is
  what makes a regular co-Z binary ladder possible; (iii) the is `(X,Y)`-only variants
  (ZADDU′, ZDAU′, ZACAU′) save 1M by never computing Z3 in the loop; (iv) you pay exactly
  one inversion at the end (`Jac2aff` 1I+3M+1S; x-only recovery 1I+8M+1S).
- **Do not claim co-Z beats Montgomery XZ** — on M-count it does not (5M+2S vs 4M+2S).

### 4.5 Windowed / fixed-window / signed-window / w-NAF in x-only — **ILLEGAL**
See §2. Not a cost question; the operations cannot be defined. A w-bit window costs
`w·(doubling) + 1·(addition)` per window = `1 + 1/w` ops/bit, plus an unattainable
difference reconstruction; and the "dictionary coherence" cost is unbounded because the
required differences `(2^j k − d)P` grow without limit.

### 4.6 Double-base / multi-base (2,3) and 2-dimensional tricks — **NOT APPLICABLE**
- Double-base chains need general point addition (no known x-only addition) — the reason
  ECM uses Lucas chains at all is stated verbatim by Bouvier–Imbert:
  > "As seen in Sect. 2.2, Montgomery curves only admit a differential addition.
  > Therefore the previous constructions (double-base expansions and chains) cannot be
  > used to perform scalar multiplication. Instead, one uses Lucas chains."
- x-only tripling *is* definable (you have x(2P), x(3P), x(P)) but it needs 2 doublings
  + 1 addition to establish and then a 3+-point state, so per bit it is ≥ 1.6 ops/bit at
  worse cost than 1 + 1. Meloni's tripling, Zeckendorf/Fibonacci and Euclidean chains all
  land at `(11.18n)M + (4.07n)S`, i.e. ~15 M-units/bit.
- Multi-scalar / 2-dimensional differential chains (Bernstein 2006's new chains) are all
  **stage-2** methods (they compute `mP + nQ` from `P, Q, P−Q`); they do not apply to
  `[s]P`. Bernstein's famous "PRAC 1.82 additions/bit, 0.33 doublings/bit" is the
  **two-dimensional** figure and must not be quoted for stage 1.

### 4.7 The one legitimate lever: kill the constant multiply in the doubling — **SMALL WIN, you already have it**
- `mdbl-1987-m` is `1M + 2S + 1*a` with **general `a`**, versus `dbl-1987-m-3`'s
  `2M + 2S + 1*a24`. Using the `a`-form with Z1=1 and **small `a`** turns the curve-constant
  multiply into a cheap constant multiply (`*const`), saving up to one full M per bit.
- In CGBN terms the win depends on whether `a`/`a24` is representable as a small limb
  vector. If you can force `a24` to be a tiny constant mod N (e.g. by choosing the curve
  via an isogeny, or by using Suyama's parametrization, which already yields small `a`),
  this is the difference between 4 M-units and 3 M-units of doubling — **up to ~1 M-unit/bit,
  i.e. up to ~1.12× on your 8.1 baseline.**
- **Flagged as the only actionable micro-optimization in this entire report, and it is not
  an algorithmic restructuring — it is a representation choice.** Montgomery made exactly
  this point in 1987: "In the binary method, 11% of the multiplications can be replaced by
  additions if (A+2)/4 is sufficiently small."

---

## 5. Comparison table (averages per bit of s)

Baseline = your measured 8.10 M-units/bit. "Strict" column uses EFD costs at S=M with a
full-price constant multiply (baseline 9.0). Speedups are ×(baseline/candidate).

| # | Scheme | D/bit | A/bit | Diff point | Cost/add | Your units /bit | Strict /bit | Speedup (yours) | Legal x-only? |
|---|---|---|---|---|---|---|---|---|---|
| 1 | **Plain binary ladder, affine diff (yours)** | 1.00 | 1.000 | **affine** | 3.86–5.0 | **8.10** | 9.00 | **1.00×** | yes |
| 2 | PRAC / 1-D Lucas chain (measured) | 0.135 | 1.388 | projective | 6.0 | 8.92 | 8.75 | 0.91× | yes |
| 3 | Best 1-D chain (Bernstein/Tsuruoka) | ~0.6 | 1.56 | projective | 6.0 | 10.7 | 11.4 | 0.76× | yes |
| 4 | Montgomery ladder, projective diff (GMP-ECM `ecm_mul`) | 1.00 | 1.000 | projective | 6.0 | 10.24 | 11.0 | 0.79× | yes |
| 5 | Fused `mladd-1987-m` (affine diff, not decomposed) | 1.00 | 1.000 | affine | in-fused | 8.10 (measured) | 9.00 | 1.00× | yes |
| 6 | Fused `ladd-1987-m-3` (projective diff) | 1.00 | 1.000 | projective | in-fused | 10.24 | 11.0 | 0.79× | yes |
| 7 | co-Z ZADDU-based ladder (X,Y-only, GJM) | 1.00 | 1.000 | shared Z | 6.0 | 14.0 | 14.0 | 0.58× | yes (Weierstrass coords) |
| 8 | co-Z ZDAU′ doubling-addition ladder | ~0.5 | ~0.5 | shared Z | 14.0 | 14.0 | 14.0 | 0.58× | yes (Weierstrass) |
| 9 | x-only w-NAF / sliding window | — | 0.2–0.5 | undefined | — | **illegal** | **illegal** | — | **no** |
| 10 | x-only fixed window (w=4) | 1.00 | 0.25 | undefined | — | **illegal** | **illegal** | — | **no** |
| 11 | Double-base (2,3) chain | ~0.6 | ~0.8 | undefined | — | **illegal** | **illegal** | — | **no** |
| 12 | Meloni EAC / Zeckendorf (11.18n M) | — | — | projective | 6.0 | ~15 | ~15 | 0.54× | yes |
| 13 | **Yours + small-`a` doubling (`mdbl`, a free)** | 1.00 | 1.000 | affine | 3.86–5.0 | **~7.1–7.6** | ~8.0 | **~1.07–1.14×** | yes |

**Provenance.** Rows 1, 5, 13: your measurements (I could not reproduce 3.86 from EFD text
— see §1). Row 2: measured op counts (Bernstein 2009 slides, GMP-ECM 6.2.3, B1=10^6),
*my* pricing. Rows 3, 4, 6, 12: my op accounting from EFD/paper costs and published chain
densities. Rows 7, 8: published M/S from GJM+Rivain+Venelli Table 1/2, my conversion.
Rows 9–11: structural illegality, not a cost estimate.

**Your own timing probe, re-derived.** Your numbers imply doubling : addition ≈ 4.24 : 3.86
= 1.10 : 1. Removing 1/k of the additions gives cost `4.24 + (1 − 1/k)·3.86`, i.e. speedup
`8.10 / (8.10 − 3.86/k)` = 1.91, 1.32, 1.18, 1.12, 1.09 for k = 1, 2, 3, 4, 5. Your measured
1.32 / 1.47 / 1.56 / 1.62 / 1.66 / 1.91 are consistently higher than this model — which
means **your probe is measuring more than the arithmetic** (register pressure, scheduling,
occupancy, or the a24 multiply being cheaper than M). Either way, the probe's ceiling, 1.91×,
is the *unattainable* "delete all additions" bound, and §2–§3 show the attainable value is 1.00×.

---

## 6. Verdict

1. **No correct x-only algorithm achieves >1.2× over 8.1 M-units/bit in this setting.**
   The plain ladder is optimal, and the reason is structural, not incidental: x-only
   arithmetic admits no general addition, so every second point must come from a
   differential addition; a fixed affine difference is the cheapest possible difference
   (it removes the `Z1*`/`X1*` multiply, worth 1M); and *any* scheme that reduces the
   number of additions must use variable differences, which are projective and cost 6
   M-units instead of ~4–5, plus it needs a bigger state and the whole chain live.
2. **The break-even arithmetic is decisive.** To win with projective-difference additions
   you need fewer than **0.86 additions/bit**. The one-dimensional differential-chain
   lower bound is **1.44042 additions/bit** (Montgomery's Theorem 7 / Fibonacci bound).
   So the best conceivable chain cannot beat you on addition count, and it loses on
   addition *cost*. Two independent walls.
3. **PRAC is not the answer.** Your ~8.4 estimate was right by luck; the measured density
   is 1 DBL + **1.39** ADD/bit (not 1 + 0.7). At 6 M-units per projective addition that is
   8.9–9.0 M-units/bit, i.e. **~10% slower**, not faster.
4. **The single best available improvement is representation-level, not algorithmic:**
   make the curve constant in the doubling genuinely free (use the `a`-form
   `mdbl-1987-m` = 1M+2S+1a with a small/near-constant `a`, or force `a24` to be a small
   constant), worth up to ~1 M-unit/bit (~1.07–1.14×). Everything else on the table loses.
5. **What to do instead of chasing an algorithm:** spend the effort on the actual kernel —
   occupancy/register budget, avoiding the `Z1*` multiply in `mdadd-1987-m` (i.e. get the
   3.86 down to a *provably* 4M+2S-with-free-constant-multiply rather than an assumed one),
   batching-friendly scheduling, and the a24 constant multiply. And **re-derive your two
   measured constants** (§1) — your baseline may really be 9.0, in which case PRAC-style
   schemes stop being a loss and start being a wash, which is worth knowing before you
   invest in any chain machinery.

---

## 7. Implementable pseudocode for the single best candidate

There is no winning *new* algorithm, so the "best candidate" is a hardening of what you
have: (a) make the doubling's constant multiply provably free, (b) get the addition to a
provable 4M+2S free-constant form, (c) keep the affine-difference invariant explicit and
asserted. Per-thread state is minimal (4 field elements + 1 constant), which is what a CUDA
ECM kernel needs.

```
# ---- Per-curve state (one thread), Montgomery curve b*y^2 = x^3 + a*x^2 + x ----
# a24  : (a+2)/4 mod N, chosen so that a24 is a SMALL constant (Suyama param gives small a)
#        OR use a directly and force a to be small.
# P    : base point, AFFINE and normalized:  X_P = 1, Z_P = 1
#        (this is the FIXED affine difference for the whole run -- it never changes)
# state for the ladder, invariant  B - A = P  i.e.  (B - A) has x = 1
A  = (XA, ZA)      # currently [k]P
B  = (XB, ZB)      # currently [k+1]P
# Note: the "difference" is NOT stored. It is the compile-time constant (1:1).

def xDBL_Q(X, Z):                       # mdbl-1987-m, Z1=1 path, 1M+2S+1*const
    # Requires: caller keeps Z arbitrary; this is the *affine-input* doubling.
    # For arbitrary Z use dbl-1987-m-3 = 2M+2S+1*a24 (no Z1=1 assumption).
    AA = X*X                           # 1S
    C  = AA + a*X + 1                  # a*X is a CONSTANT multiply (small a) -> cheap
    X2 = (AA - 1)^2                    # 1S
    Z2 = 4*X*C                         # 1M
    return (X2, Z2)

def xDBL(X, Z):                         # dbl-1987-m-3 with CSE, 2M+2S+1*a24
    t0 = X + Z ; AA = t0*t0            # 1S
    t1 = X - Z ; BB = t1*t1            # 1S
    C  = AA - BB
    X2 = AA * BB                       # 1M
    Z2 = C * (BB + a24*C)              # 1M + 1*a24  (fold a24*C into a cheap const mul)
    return (X2, Z2)

def xDADD(X2,Z2, X3,Z3):                # mdadd-1987-m, affine difference (1:1)
    # returns (X5,Z5) = x(P2+P3) where the difference has x = 1
    A = X2 + Z2 ; Bv = X2 - Z2
    C = X3 + Z3 ; D  = X3 - Z3
    DA = D * A                         # 1M
    CB = C * Bv                        # 1M
    s  = DA + CB ; d = DA - CB
    X5 = s * s                         # 1S
    Z5 = d * d                         # 1S    <-- X1 = 1, so the X1* multiply is GONE
    return (X5, Z5)
    # NOTE: this is the 4M form only if the X1* multiply is genuinely eliminated.
    # EFD's mdadd-1987-m prints Z5 = X1*(DA-CB)^2, i.e. 5M. Verify or pay 1M.

# ---- Main loop: left-to-right over the bits of the FIXED, PUBLIC scalar s ----
# Precompute s once, shared by every thread: bit-array of s = lcm(1..B1).
assert X_P == 1 and Z_P == 1
(A, B) = (xDBL(X_P, Z_P), (1, 1))       # A = 2P, B = P  ->  (k,k+1) = (2,1)
# convention below: maintain (A,B) = ([k]P, [k+1]P), k = prefix of s read so far

for bit in s_bits[1:]:                  # skip the leading bit
    if bit == 0:
        # (k, k+1) -> (2k, 2k+1)   = (2A, A+B)
        B = xDADD(A, B)                 # A+B ; difference B-A = P (affine)  [1 add]
        A = xDBL(A)                     # 2A                                    [1 dbl]
        # invariant: B - A = (A+B) - 2A = B_old - A_old = P    -> still affine, FREE
    else:
        # (k, k+1) -> (2k+1, 2k+2) = (A+B, 2B)
        A = xDADD(A, B)                 # A+B ; difference B-A = P (affine)  [1 add]
        B = xDBL(B)                     # 2B                                    [1 dbl]
        # invariant: new B - new A = 2B - (A+B) = B_old - A_old = P  -> FREE
    # NO normalization, NO dictionary, NO difference update. The affine difference
    # is preserved *algebraically* by both branches: this is the whole trick.

# Final: [s]P = A.  Nothing to normalize; A is projective (X_A : Z_A) as required by
# ECM stage 2, which never needs Z_A = 1.
```

**Per-curve memory:** 4 field elements of state (`XA, ZA, XB, ZB`) + `a24`/`a` + `N` + the
base point (which is the constant (1:1)). That is the entire footprint — versus 16 XZ
registers (32 mpz) for a replayable PRAC chain, 3 live XZ points + 4 temporaries for plain
PRAC, and large tables for any window. On a GPU where every register costs occupancy, this
is a decisive second reason your structure is the right one.

**Digit choice:** there is none — the scalar is fixed and public, so the branch is
*thread-uniform* (`if bit == 0 ... else ...` on a precomputed shared bit array). No SIMT
divergence, exactly as you noted. Do **not** introduce signed digits: NAF would not change
the per-bit operation count in a 2-point state (both branches still need 1 dbl + 1 add), and
it destroys the "difference is the fixed affine point P" invariant.

**What to add as asserts while porting:** after each step, check (debug build only, on a CPU
model) that the pair tracks `([k]P, [k+1]P)` for the prefix `k`, and that the difference is
`P` — i.e. mirror GMP-ECM's approach of making the Lucas-chain invariant an explicit,
asserted structure field rather than a comment.

---

## 8. What would change the answer

- **If `s` were secret or per-thread**, the whole analysis is unchanged (your scalar is
  fixed anyway), but constant-time requirements would forbid nothing here — the ladder is
  already uniform.
- **If the difference could be made affine for a variable point at O(1) cost**, windowing
  would become legal and you could win. There is no such technique in the literature, and
  §2 explains why there cannot be: making `Z_D = 1` requires a division, and a division
  costs an inversion.
- **If `a24` can be a small constant**, you win ~1 M-unit/bit. This is the only live lead.
- **If you are willing to move off Montgomery x-only** (e.g. to a curve shape with cheap
  complete addition, or to (X,Y)-only co-Z), the arithmetic changes entirely — but x-only
  is what makes ECM stage 1 cheap, and co-Z is 1.5× worse here, so this is not attractive.

---

## 9. Two factual corrections to your framing

- **bitlength(s) for B1 = 1e5 is 144352, not ~144000.** Independently reproduced by sieving
  prime powers; b(10^3)=1438, b(10^4)=14447, b(10^5)=144352, b(10^6)=1442099 (Bernstein's
  2009 slides state 1442099 for B1=10^6). The reason is ψ(B1) ≈ B1, so
  b ≈ B1/ln 2 ≈ 1.4427·B1. Your 144000 is 0.24% low — harmless, but the scaling law is
  b ≈ 1.4427·B1, not anything logarithmic.
- **You are not at the GMP-ECM `ADD=6/DUP=5` baseline advantage you may think.** GMP-ECM's
  `ecm_mul()` ladder uses `duplicate` = 5M and `add3` = 6M = **11 M-units/bit**. Your
  affine-difference ladder at 8.1–9.0 is genuinely better than a standard x-only
  Montgomery ladder — you have already banked the `Z_D = 1` and `Z1 = 1` optimizations
  (Montgomery 1987 §10.3.1: "If one starts with X1 = 2 and Z1 = 1, then this cost reduces
  to 9 log2 n multiplications and 9 log2 n additions"). That is why nothing else wins.

---

## 10. Corrections to names and citations (things that do not exist)

Flagging these because they were in the premise and would propagate into your code comments:

- **EFD has no co-Z / ZADD / ZADDU / ZADDC / ZDBL entries at all.** Searched the genus-1
  index (471 formulas, 33 representations, 10 curve shapes): EFD's vocabulary is
  DBL/ADD/reADD/preADD/mADD/mreADD/mpreADD/mDBL/mmADD/DADD/mDADD/LADD/mLADD/TPL/SUNI/SCALE.
- **`dbl-2005-11` and `dadd-2005-11` do not exist.** The EFD Montgomery XZ page lists only
  1987 Montgomery formulas; `.../auto-code/montgom/xz/doubling/dbl-2005-11.op3` returns 404.
  The 2002 entries (Brier–Joye, Izu–Takagi) are on the **short Weierstrass** XZ page.
- **`add-1987-m-3` does not exist**, and there is no projective *full* addition in Montgomery
  XZ at all — x-only cannot do general addition.
- **Your projective/affine mapping was inverted** — see §1 item 1. `dadd-1987-m-3` (4M+2S) is
  the *projective*-difference one; `mdadd-1987-m` (3M+2S) is the *affine* one.
- **There is no `prac.c` in GMP-ECM** and no `prac_chain`/`prac_chain2`. `prac()` and
  `lucas_cost()` are static in **`ecm.c`**; `pp1_mul_prac()` is public in `lucas.c`.
- **Prime95 has no `lucas_mul` in `ecm.cpp`.** The working PRAC is `lucas_mul()` in
  **`gwnum/ecmstag1.c`**; `ecm.cpp` only defines an apparently unreferenced
  `int PRAC_SEARCH = 7;` (I could not establish its semantics — do not assert it).
- Meloni's paper is **WAIFI 2007, "New point addition formulae for ECC applications",
  LNCS 4547:189–201** — not a longer "co-Z … XZ-only scalar multiplication" title.

---

## 11. Citations and links

**Primary formulas / EFD**
- Explicit-Formulas Database, Montgomery curves, XZ coordinates — the only Montgomery XZ
  page, containing `mdbl-1987-m` (1M+2S+1a), `dbl-1987-m-3` (2M+2S+1a24),
  `mdadd-1987-m` (3M+2S), `dadd-1987-m-3` (4M+2S), `dadd-1987-m` (6M+2S),
  `mladd-1987-m` (5M+4S), `ladd-1987-m-3` (6M+4S), `scale` (1I+1M):
  <https://www.hyperelliptic.org/EFD/g1p/auto-montgom-xz.html>
- EFD Montgomery shape page: <https://www.hyperelliptic.org/EFD/g1p/auto-montgom.html>
- EFD genus-1 index / speed tables: <https://www.hyperelliptic.org/EFD/g1p/index.html>

**Montgomery's original work**
- P. L. Montgomery, "Speeding the Pollard and elliptic curve methods of factorization",
  *Math. Comp.* **48**(177):243–264, 1987 — the x-only formulas (p. 261, 3rd–6th displays),
  the §10.3.1 cost discussion quoted above, and the parametrization.
  <https://www.ams.org/journals/mcom/1987-48-177/S0025-5718-1987-0866113-7/S0025-5718-1987-0866113-7.pdf>
- P. L. Montgomery, "Evaluating recurrences of form X_{m+n} = f(X_m, X_n, X_{m−n}) via Lucas
  chains" (1983, rev. 1991/1992), unpublished — **Algorithm PRAC**, Table 4 (the 9 rules),
  invariant (5.1) `C = X_{a−b}`, the ≤ 4 log2 n bound, the simplified rules {3,4,5,9}, the
  "1.6 log2 n" density and "PRAC occasionally does worse" (Tables 5 and 6).
  <http://cr.yp.to/bib/1992/montgomery-lucas.pdf>

**Differential addition chains**
- D. J. Bernstein, "Differential addition chains", 2006 — the definition quoted in §2, the
  one-dimensional comparison table (`1 | standard | 1.533 | 1.560 | 8.885 | 8.983`), the
  1.44042 additions/bit lower bound, the "denominator 1 ⇒ 5 field mults" observation, the
  Curve25519-based weights (1336, 1093, 905), and the Euclid/S/Bleichenbacher/Tsuruoka
  chain definitions.
  <http://cr.yp.to/ecdh/diffchain-20060219.pdf>
  (also reachable via <https://r.jina.ai/http://cr.yp.to/ecdh/diffchain-20060219.pdf>)
- D. J. Bernstein, "ECM speed records on CPU and GPU", slides, 2009-09-12 — the measured
  B1=10^6 and B1=10^3 stage-1 op counts quoted in §3.
  <https://cr.yp.to/talks/2009.09.12/slides.pdf>
- M. Stam, "Speeding up subgroup cryptosystems", PhD thesis, TU Eindhoven, 2003
  (Conjecture 3.29, differential-addition-chain theory) — cited via Bernstein 2006.
- Y. Tsuruoka, "Computing short Lucas chains for elliptic curve cryptosystems", *IEICE
  Trans. Fundamentals* E84-A(5):1227–1233, 2001.
- M. Kutz, "Lower bounds for Lucas chains", *SIAM J. Comput.* 31:1896–1908, 2002.

**co-Z arithmetic**
- N. Meloni, "New point addition formulae for ECC applications", WAIFI 2007, LNCS 4547:189–201
  — "This addition involves 5M and 2S"; "The computational cost of this addition is 4M and
  2S, which is lower than with our formula"; the `(5s − 7)M + (2s − 1)S` EAC cost; x-recovery
  8M+4S+1 inversion.
  <https://dl.acm.org/doi/10.1007/978-3-540-73074-3_15>
- R. Goundar, M. Joye, A. Miyaji, "Co-Z addition formulae and binary ladders on elliptic
  curves", CHES 2010 / eprint 2010/309 — ZADDU 5M+2S, ZADDC 6M+3S, DBLU 1M+5S,
  ladder 9M+7S/bit, the (X,Y)-only n(8M+6S)+1I+1M results, and the explicit rejection of NAF
  in favour of zeroless signed digits. <https://eprint.iacr.org/2010/309>
- R. Goundar, M. Joye, A. Miyaji, F. Rivain, A. Venelli, "Scalar multiplication on
  Weierstrass elliptic curves using co-Z addition formulae", *J. Cryptographic Engineering*
  1(2):161–176, 2011, DOI 10.1007/s13389-011-0012-0 — **the authoritative co-Z cost table**
  (ZADDU 5M+2S, ZADDU′ 4M+2S, ZADDC 6M+3S, ZADDC′ 5M+3S, ZDAU 9M+7S, ZDAU′ 8M+6S,
  ZACAU 9M+7S, ZACAU′ 8M+6S, DBLU 1M+5S, TPLU 6M+7S; Jac2aff 1I+3M+1S).
  <https://doi.org/10.1007/s13389-011-0012-0>
- F. Rivain, "Fast and regular algorithms for scalar multiplication over elliptic curves",
  eprint 2011/338. <https://eprint.iacr.org/2011/338>
- A. Venelli, S. Dassance, "Faster side-channel resistant elliptic curve scalar
  multiplication", *Contemp. Math.* 521:29–40, 2010 — origin of the (X,Y)-only variants.

**ECM chain practice and implementations**
- GMP-ECM source: `prac()` and `lucas_cost()` in **ecm.c** (`#define ADD 6.0`,
  `#define DUP 5.0`, the 10 `val[]` constants, the "do the first line of Table 4 whose
  condition qualifies" comment), `ecm_mul()` (the textbook ladder, invariant comment
  `/* invariant: (P1,P0) = ((k+1)P, kP) */`), `add3()`/`duplicate()` (the cost comments
  quoted in §2), the 2023 McLaughlin precomputed-Lucas-chain path
  (`generate_Lucas_chain`, `Lchain_codes.dat`, `ASSERT(dif == Lchain[...].value && k < 15)`,
  `mpres_t LCS_x[16]`), and `lucas.c` (the P+1 `val` note).
  <https://gitlab.inria.fr/zimmerma/ecm> ·
  raw mirror used: <https://raw.githubusercontent.com/sethtroisi/gmp-ecm/main/ecm.c>
- prime95/MPrime: `ell_mul()`, `lucas_mul()`, `bin_ell_mul()`, `ell_dbl`, `ell_add_fft`,
  `lucas_cost` in **gwnum/ecmstag1.c**; the verbatim comment "Try a series of Lucas chains
  to find the cheapest… This is much faster than bin_ell_mul, but uses more memory."
  <https://raw.githubusercontent.com/shafferjohn/Prime95/master/gwnum/ecmstag1.c>
  (note: prime95's `lucas_cost` charges `ell_dbl` and `ell_add` both 12, which contradicts
  its own per-routine comments of 10 and 12 FFTs — a real inconsistency in that source.)
- P. Zimmermann, "20 years of ECM", ANTS 2006 — the 10 α constants table, the "gain using
  those 10 values instead of α = φ only is 3.72% for B1 = 10^6", and the Lucas-chain
  framing. <https://members.loria.fr/PZimmermann/papers/ecm.pdf>
- C. Bouvier, L. Imbert, "Faster cofactorization with ECM using mixed representations",
  PKC 2020 — "Montgomery curves only admit a differential addition. Therefore the
  previous constructions (double-base expansions and chains) cannot be used"; "When an
  addition step is encountered, the definition ensures that the difference of the two
  operands is already available"; PRAC invariants `±C = A − B`, `[n]P = [d]A + [e]B`;
  Bos–Kleinjung B1=256 at "361 doublings and only 38 additions"; Table 1 XZ costs
  (dADD 4M+2S = 6M, dDBL 3M+2S = 5M); "The arithmetic cost per bit of our implementation
  is relatively stable, around 7.6 M."
  <https://link.springer.com/content/pdf/10.1007/978-3-030-45388-6_17.pdf>
- J. W. Bos, T. Kleinjung, "ECM at work", ASIACRYPT 2012 (paywalled; described via
  Bouvier–Imbert above).
- M. Hamburg, "Faster Montgomery and double-add ladders for short Weierstrass curves",
  TCHES 2020 / eprint 2020/437 — the reference point for what per-bit ladder costs look
  like **off** Montgomery curves (Montgomery ladder "5M+4S+1m+8A per scalar bit" on
  Montgomery curves; 8M+3S+7A on short Weierstrass). Included to show that no x-only
  Montgomery improvement is hiding in this line of work.
  <https://eprint.iacr.org/2020/437> · code: <https://github.com/bitwiseshiftleft/ladder_formulas>

---

## 12. Explicit uncertainties (do not treat as established)

1. **I could not reproduce your measured 3.86 from any EFD formula.** `mdadd-1987-m` prints
   3M+2S = 5 M-units at S=M. Your 3.86 requires a free 1M. Either you have a genuine extra
   optimization the EFD text does not show, or your micro-benchmark is mis-attributing work.
   **This is the highest-priority thing to re-verify**, because it sets the baseline
   everything else is compared against. If the true baseline is 9.0, PRAC at 8.75–9.0 is a
   wash rather than a loss.
2. **I could not fetch the EFD `.op3` / `.sage` sub-pages** (non-text content type to my
   fetcher); all EFD costs above come from the rendered page text, which I read in full.
3. **No published D/bit and A/bit curve as a function of B1 exists for stage-1 PRAC.** Only
   Montgomery's aggregate f-evaluation counts and the Bernstein 2009 measured pair
   (1.38820, 0.13463) at B1 = 10^6 — two data points, not a function. I flagged the
   B1 = 1e5 extrapolation as an estimate, not a measurement.
4. **`PRAC_SEARCH = 7` semantics in prime95 are unverified** (the variable is not referenced
   in the fetched `ecm.cpp`; the mersenneforum thread documenting it returns 403). Do not
   assert it.
5. **My "no x-only windowed paper exists" claim is a search-based negative result**, not a
   proof of non-existence. The structural argument in §2 is a proof; the corpus claim is
   evidence. Also note one paper I could not read (a 2024 "Searching for differential
   addition chains", *Res. Number Theory*, DOI 10.1007/s40993-024-00604-8) was blocked by
   Cloudflare — it appears to be about one-dimensional differential addition chains and
   *might* contain relevant newer numbers. Flagging as unread rather than dismissing it.
6. **Meloni's exact co-Z costs are quoted from a full-text mirror** (the HAL copy is behind
   a bot wall), though the numbers are independently corroborated by both GJM papers.
7. I could not retrieve the HAL PhD thesis `tel-00477005v2` that a search snippet quoted
   ("Both chains involve an addition step that references a difference that occurred 5 steps
   before the new term") — the snippet is consistent with §2 but I could not read the source,
   so I do not cite it as a source.
