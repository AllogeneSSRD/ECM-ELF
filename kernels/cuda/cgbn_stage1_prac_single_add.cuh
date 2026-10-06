#ifndef CGBN_STAGE1_PRAC_SINGLE_ADD_CUH
#define CGBN_STAGE1_PRAC_SINGLE_ADD_CUH

// Normalize every odd-prime rule to T = ADD(A, B, C), including the final ADD.
// All point coordinates stay in fixed bn variables; no dynamic bn pointers and
// no device-call ABI. The two T coordinates are the candidate's added liveness.
template<class P>
__device__ FORCE_INLINE void prac_odd_single_add(curve_t<P> &c,
    typename curve_t<P>::bn_t &ax, typename curve_t<P>::bn_t &az,
    typename curve_t<P>::bn_t &bx, typename curve_t<P>::bn_t &bz,
    typename curve_t<P>::bn_t &cx, typename curve_t<P>::bn_t &cz,
    const typename curve_t<P>::bn_t &a24, const typename curve_t<P>::bn_t &n,
    uint32_t np0, uint32_t p, uint32_t initial_d) {
    typename curve_t<P>::bn_t tx, tz;
    cgbn_set(c._env, cx, ax); cgbn_set(c._env, cz, az);
    prac_dbl<P>(c, bx, bz, ax, az, a24, n, np0);
    uint32_t e = p - initial_d, d = initial_d - e;
    for (;;) {
        const bool finish = d == e;
        int rule = 0;
        if (!finish) {
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
        prac_add(c, tx, tz, ax, az, bx, bz, cx, cz, n, np0);
        if (finish) {
            cgbn_set(c._env, ax, tx); cgbn_set(c._env, az, tz);
            break;
        }
        if (rule == 0) {
            // Subtract: (A, B, C) -> (A, A+B, B).
            cgbn_set(c._env, cx, bx); cgbn_set(c._env, cz, bz);
            cgbn_set(c._env, bx, tx); cgbn_set(c._env, bz, tz);
            d -= e;
        } else {
            prac_dbl<P>(c, ax, az, ax, az, a24, n, np0);
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
