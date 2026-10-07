#ifndef CGBN_STAGE1_PRAC_SINGLE_ADD_CUH
#define CGBN_STAGE1_PRAC_SINGLE_ADD_CUH

// Private to the single-site candidate. Both outputs must be distinct from
// every input (and each other): ox holds V before xd/zd are consumed. The only
// call below uses fresh tx/tz, never point-coordinate or modulus aliases.
// Do not substitute this for the public, alias-safe prac_add.
template<class P>
__device__ FORCE_INLINE void prac_add_disjoint(curve_t<P> &c,
    typename curve_t<P>::bn_t &ox, typename curve_t<P>::bn_t &oz,
    const typename curve_t<P>::bn_t &x1, const typename curve_t<P>::bn_t &z1,
    const typename curve_t<P>::bn_t &x2, const typename curve_t<P>::bn_t &z2,
    const typename curve_t<P>::bn_t &xd, const typename curve_t<P>::bn_t &zd,
    const typename curve_t<P>::bn_t &n, uint32_t np0) {
    typename curve_t<P>::bn_t t, u;
    cgbn_add(c._env, t, x1, z1); c.normalize_addition(t, n);
    if (cgbn_sub(c._env, u, x2, z2)) cgbn_add(c._env, u, u, n);
    c.mont_mul_normalized(u, u, t, n, np0);
    if (cgbn_sub(c._env, t, x1, z1)) cgbn_add(c._env, t, t, n);
    cgbn_add(c._env, ox, x2, z2); c.normalize_addition(ox, n);
    c.mont_mul_normalized(ox, ox, t, n, np0);
    cgbn_add(c._env, t, u, ox); c.normalize_addition(t, n);
    if (cgbn_sub(c._env, u, u, ox)) cgbn_add(c._env, u, u, n);
    c.mont_sqr_normalized(t, t, n, np0);
    c.mont_sqr_normalized(u, u, n, np0);
    c.mont_mul_normalized(ox, zd, t, n, np0);
    c.mont_mul_normalized(u, xd, u, n, np0);
    cgbn_set(c._env, oz, u);
}

// Normalize every odd-prime rule to T = ADD(A, B, C), including the final ADD.
// All point coordinates stay in fixed bn variables; no dynamic bn pointers and
// no device-call ABI. The two T coordinates are the candidate's added liveness.
template<class P, bool COMPACT>
__device__ FORCE_INLINE void prac_odd_single_add(curve_t<P> &c,
    typename curve_t<P>::bn_t &ax, typename curve_t<P>::bn_t &az,
    typename curve_t<P>::bn_t &bx, typename curve_t<P>::bn_t &bz,
    typename curve_t<P>::bn_t &cx, typename curve_t<P>::bn_t &cz,
    const typename curve_t<P>::bn_t &a24, const typename curve_t<P>::bn_t &n,
    uint32_t np0, uint32_t p, uint32_t initial_d) {
    typename curve_t<P>::bn_t tx, tz;
    cgbn_set(c._env, cx, ax); cgbn_set(c._env, cz, az);
    prac_dbl<P, COMPACT>(c, bx, bz, ax, az, a24, n, np0);
    uint32_t e = p - initial_d, d = initial_d - e;
    for (;;) {
        // In the compact candidate, carry the final state through the ADD in
        // rule itself (4), rather than a separate live finish flag. Preserve
        // the old single-add control flow for same-binary comparisons.
        const bool finish = !COMPACT && d == e;
        int rule = COMPACT && d == e ? 4 : 0;
        if (COMPACT ? rule != 4 : !finish) {
            if (d < e) {
                uint32_t tmp = d; d = e; e = tmp;
                cgbn_swap(c._env, ax, bx); cgbn_swap(c._env, az, bz);
            }
            if (uint64_t(d) * 100 > uint64_t(e) * 296) {
                rule = ((d & 1) == (e & 1)) ? 1 : !(d & 1) ? 2 : 3;
                if (rule == 3) {
                    cgbn_swap(c._env, ax, bx); cgbn_swap(c._env, az, bz);
                }
                if (rule == 2 || rule == 3) {
                    cgbn_swap(c._env, bx, cx); cgbn_swap(c._env, bz, cz);
                }
            }
        }
        if constexpr (COMPACT)
            prac_add_disjoint(c, tx, tz, ax, az, bx, bz, cx, cz, n, np0);
        else
            prac_add(c, tx, tz, ax, az, bx, bz, cx, cz, n, np0);
        if (COMPACT ? rule == 4 : finish) {
            cgbn_set(c._env, ax, tx); cgbn_set(c._env, az, tz);
            break;
        }
        if (rule == 0) {
            // Subtract: (A, B, C) -> (A, A+B, B).
            cgbn_set(c._env, cx, bx); cgbn_set(c._env, cz, bz);
            cgbn_set(c._env, bx, tx); cgbn_set(c._env, bz, tz);
            d -= e;
        } else {
            prac_dbl<P, COMPACT>(c, ax, az, ax, az, a24, n, np0);
            if (rule == 1) {
                // Same parity: C survives, B becomes A+B, A doubles.
                cgbn_set(c._env, bx, tx); cgbn_set(c._env, bz, tz);
                d = (d - e) / 2;
            } else {
                // Before ADD: rule2 (A,C,B), rule3 (B,C,A). Restore the
                // logical roles using dead inputs after ADD/DBL consumed them.
                if (rule == 2) {
                    cgbn_set(c._env, bx, cx); cgbn_set(c._env, bz, cz);
                    d /= 2;
                } else {
                    cgbn_set(c._env, bx, ax); cgbn_set(c._env, bz, az);
                    cgbn_set(c._env, ax, cx); cgbn_set(c._env, az, cz);
                    e /= 2;
                }
                cgbn_set(c._env, cx, tx); cgbn_set(c._env, cz, tz);
            }
        }
    }
}
#endif
