#ifndef CGBN_STAGE1_PRAC_KERNEL_CUH
#define CGBN_STAGE1_PRAC_KERNEL_CUH
#include "ecm_prac_plan.h"

enum { ECM_DOMAIN_INIT = 0, ECM_DOMAIN_EXPORT = 1, ECM_DOMAIN_LADDER = 2,
       ECM_DOMAIN_PRAC = 3, ECM_DOMAIN_PRAC_NATURAL = 4, ECM_DOMAIN_PRAC_168 = 5,
       ECM_DOMAIN_PRAC_COMPACT = 6, ECM_DOMAIN_PRAC_COMPACT_168 = 7,
       ECM_DOMAIN_PRAC_OUTLINE_ADD = 8, ECM_DOMAIN_PRAC_OUTLINE_ADD_168 = 9,
       ECM_DOMAIN_PRAC_SINGLE_ADD = 10, ECM_DOMAIN_PRAC_SINGLE_ADD_168 = 11,
       ECM_DOMAIN_PRAC_SINGLE_COMPACT = 12, ECM_DOMAIN_PRAC_SINGLE_COMPACT_168 = 13 };
cgbn_stage1_kernel_fn cgbn_stage1_domain_dispatch(uint32_t bits, uint32_t *tpi, int mode,
                                                uint32_t requested_tpi = 0);

// Outputs may overwrite either input, including the difference. Read before commit.
template<class P, bool COMPACT = false>
__device__ FORCE_INLINE void prac_dbl(curve_t<P> &c,
    typename curve_t<P>::bn_t &ox, typename curve_t<P>::bn_t &oz,
    const typename curve_t<P>::bn_t &x, const typename curve_t<P>::bn_t &z,
    const typename curve_t<P>::bn_t &a24, const typename curve_t<P>::bn_t &n, uint32_t np0) {
    typename curve_t<P>::bn_t a, b;
    cgbn_add(c._env, a, x, z); c.normalize_addition(a, n);
    if (cgbn_sub(c._env, b, x, z)) cgbn_add(c._env, b, b, n);
    c.mont_sqr_normalized(a, a, n, np0);
    c.mont_sqr_normalized(b, b, n, np0);
    if constexpr (COMPACT) {
        // X consumes AA/BB before AA becomes E. Z is free after input sums
        // are formed and can hold a24*E. Only two local field values remain.
        c.mont_mul_normalized(ox, a, b, n, np0);
        if (cgbn_sub(c._env, a, a, b)) cgbn_add(c._env, a, a, n);
        c.mont_mul_normalized(oz, a24, a, n, np0);
        cgbn_add(c._env, b, b, oz); c.normalize_addition(b, n);
        c.mont_mul_normalized(oz, a, b, n, np0);
    } else {
        typename curve_t<P>::bn_t e;
        if (cgbn_sub(c._env, e, a, b)) cgbn_add(c._env, e, e, n);
        c.mont_mul_normalized(ox, a, b, n, np0);
        c.mont_mul_normalized(a, a24, e, n, np0);
        cgbn_add(c._env, a, a, b); c.normalize_addition(a, n);
        c.mont_mul_normalized(oz, e, a, n, np0);
    }
}
template<class P>
__device__ FORCE_INLINE void prac_add(curve_t<P> &c,
    typename curve_t<P>::bn_t &ox, typename curve_t<P>::bn_t &oz,
    const typename curve_t<P>::bn_t &x1, const typename curve_t<P>::bn_t &z1,
    const typename curve_t<P>::bn_t &x2, const typename curve_t<P>::bn_t &z2,
    const typename curve_t<P>::bn_t &xd, const typename curve_t<P>::bn_t &zd,
    const typename curve_t<P>::bn_t &n, uint32_t np0) {
    typename curve_t<P>::bn_t t, u, v;
    cgbn_add(c._env, t, x1, z1); c.normalize_addition(t, n);
    if (cgbn_sub(c._env, u, x2, z2)) cgbn_add(c._env, u, u, n);
    c.mont_mul_normalized(u, u, t, n, np0);
    if (cgbn_sub(c._env, t, x1, z1)) cgbn_add(c._env, t, t, n);
    cgbn_add(c._env, v, x2, z2); c.normalize_addition(v, n);
    c.mont_mul_normalized(v, v, t, n, np0);
    cgbn_add(c._env, t, u, v); c.normalize_addition(t, n);
    if (cgbn_sub(c._env, u, u, v)) cgbn_add(c._env, u, u, n);
    c.mont_sqr_normalized(t, t, n, np0);
    c.mont_sqr_normalized(u, u, n, np0);
    c.mont_mul_normalized(v, zd, t, n, np0);
    c.mont_mul_normalized(u, xd, u, n, np0);
    cgbn_set(c._env, ox, v); cgbn_set(c._env, oz, u);
}

// One shared xADD body trades repeated instruction text for the CUDA call ABI.
// Keep the alias-safe arithmetic above identical to the baseline.
template<class P>
__device__ __noinline__ void prac_add_outlined(curve_t<P> &c,
    typename curve_t<P>::bn_t &ox, typename curve_t<P>::bn_t &oz,
    const typename curve_t<P>::bn_t &x1, const typename curve_t<P>::bn_t &z1,
    const typename curve_t<P>::bn_t &x2, const typename curve_t<P>::bn_t &z2,
    const typename curve_t<P>::bn_t &xd, const typename curve_t<P>::bn_t &zd,
    const typename curve_t<P>::bn_t &n, uint32_t np0) {
    prac_add(c, ox, oz, x1, z1, x2, z2, xd, zd, n, np0);
}

template<class P, bool OUTLINE>
__device__ FORCE_INLINE void prac_add_selected(curve_t<P> &c,
    typename curve_t<P>::bn_t &ox, typename curve_t<P>::bn_t &oz,
    const typename curve_t<P>::bn_t &x1, const typename curve_t<P>::bn_t &z1,
    const typename curve_t<P>::bn_t &x2, const typename curve_t<P>::bn_t &z2,
    const typename curve_t<P>::bn_t &xd, const typename curve_t<P>::bn_t &zd,
    const typename curve_t<P>::bn_t &n, uint32_t np0) {
    if constexpr (OUTLINE) prac_add_outlined(c, ox, oz, x1, z1, x2, z2, xd, zd, n, np0);
    else prac_add(c, ox, oz, x1, z1, x2, z2, xd, zd, n, np0);
}

// The definition lives only in the candidate TU. Baseline instantiations discard
// this call, so editing the candidate body need not recompile every PRAC tier.
template<class P, bool COMPACT = false>
__device__ FORCE_INLINE void prac_odd_single_add(curve_t<P> &c,
    typename curve_t<P>::bn_t &ax, typename curve_t<P>::bn_t &az,
    typename curve_t<P>::bn_t &bx, typename curve_t<P>::bn_t &bz,
    typename curve_t<P>::bn_t &cx, typename curve_t<P>::bn_t &cz,
    const typename curve_t<P>::bn_t &a24, const typename curve_t<P>::bn_t &n,
    uint32_t np0, uint32_t p, uint32_t initial_d);

template<class P, int MODE>
__global__ void __maxnreg__((MODE == ECM_DOMAIN_PRAC_NATURAL || MODE == ECM_DOMAIN_PRAC_COMPACT || MODE == ECM_DOMAIN_PRAC_OUTLINE_ADD || MODE == ECM_DOMAIN_PRAC_SINGLE_ADD || MODE == ECM_DOMAIN_PRAC_SINGLE_COMPACT) ? 255 :
                         (MODE == ECM_DOMAIN_PRAC_168 || MODE == ECM_DOMAIN_PRAC_COMPACT_168 || MODE == ECM_DOMAIN_PRAC_OUTLINE_ADD_168 || MODE == ECM_DOMAIN_PRAC_SINGLE_ADD_168 || MODE == ECM_DOMAIN_PRAC_SINGLE_COMPACT_168) ? 168 : P::REG_TARGET) kernel_suyama_domain(
    cgbn_error_report_t *report, uint64_t total, uint64_t start, uint64_t length,
    uint32_t *control, uint32_t *data, uint32_t count, uint32_t unused, uint32_t np0) {
    (void)unused;
    int instance = (blockIdx.x * blockDim.x + threadIdx.x) / P::TPI;
    if (instance >= count) return;
    curve_t<P> c(CHECK_ERROR ? cgbn_report_monitor : cgbn_no_checks, report, instance);
    using bn = typename curve_t<P>::bn_t;
    auto *mem = reinterpret_cast<typename curve_t<P>::mem_t *>(data) + 7 * instance;
    bn n; cgbn_load(c._env, n, mem);
    if constexpr (MODE == ECM_DOMAIN_INIT || MODE == ECM_DOMAIN_EXPORT) {
        bn value;
        for (int word = 1; word < 7; ++word) {
            cgbn_load(c._env, value, mem + word);
            if constexpr (MODE == ECM_DOMAIN_INIT) cgbn_bn2mont(c._env, value, value, n);
            else cgbn_mont2bn(c._env, value, value, n, np0);
            cgbn_store(c._env, mem + word, value);
        }
    } else {
        bn ax, az, bx, bz, a24;
        cgbn_load(c._env, a24, mem + 1);
        cgbn_load(c._env, ax, mem + 3); cgbn_load(c._env, az, mem + 4);
        if constexpr (MODE == ECM_DOMAIN_LADDER) {
            bn diff;
            cgbn_load(c._env, diff, mem + 2);
            cgbn_load(c._env, bx, mem + 5); cgbn_load(c._env, bz, mem + 6);
            int swapped = 0;
            for (uint64_t b = start; b < start + length; ++b) {
                uint64_t nth = total - 1 - b;
                int bit = (control[nth / 32] >> (nth & 31)) & 1;
                if (bit != swapped) {
                    swapped = bit; cgbn_swap(c._env, ax, bx); cgbn_swap(c._env, az, bz);
                }
                c.double_add_v2_suyama(ax, az, bx, bz, a24, diff, n, np0);
            }
            if (swapped) { cgbn_swap(c._env, ax, bx); cgbn_swap(c._env, az, bz); }
            cgbn_store(c._env, mem + 5, bx); cgbn_store(c._env, mem + 6, bz);
        } else {
            constexpr bool compact = MODE == ECM_DOMAIN_PRAC_COMPACT || MODE == ECM_DOMAIN_PRAC_COMPACT_168 ||
                                     MODE == ECM_DOMAIN_PRAC_SINGLE_COMPACT || MODE == ECM_DOMAIN_PRAC_SINGLE_COMPACT_168;
            constexpr bool outlined = MODE == ECM_DOMAIN_PRAC_OUTLINE_ADD || MODE == ECM_DOMAIN_PRAC_OUTLINE_ADD_168;
            bn cx, cz;
            auto *primes = reinterpret_cast<const EcmPracPrime *>(control);
            for (uint64_t i = start; i < start + length; ++i) {
                EcmPracPrime entry = primes[i];
                for (uint32_t repeat = 0; repeat < entry.repetitions; ++repeat) {
                    if (entry.p == 2) { prac_dbl<P, compact>(c, ax, az, ax, az, a24, n, np0); continue; }
                    if constexpr (MODE == ECM_DOMAIN_PRAC_SINGLE_ADD || MODE == ECM_DOMAIN_PRAC_SINGLE_ADD_168 ||
                                  MODE == ECM_DOMAIN_PRAC_SINGLE_COMPACT || MODE == ECM_DOMAIN_PRAC_SINGLE_COMPACT_168) {
                        prac_odd_single_add<P, compact>(c, ax, az, bx, bz, cx, cz, a24, n, np0, entry.p, entry.d);
                    } else {
                    cgbn_set(c._env, cx, ax); cgbn_set(c._env, cz, az);
                    prac_dbl<P, compact>(c, bx, bz, ax, az, a24, n, np0);
                    uint32_t e = entry.p - entry.d, d = entry.d - e;
                    while (d != e) {
                        if (d < e) {
                            uint32_t tmp = d; d = e; e = tmp;
                            cgbn_swap(c._env, ax, bx); cgbn_swap(c._env, az, bz);
                        }
                        if (uint64_t(d) * 100 <= uint64_t(e) * 296) {
                            prac_add_selected<P, outlined>(c, cx, cz, ax, az, bx, bz, cx, cz, n, np0);
                            cgbn_swap(c._env, bx, cx); cgbn_swap(c._env, bz, cz);
                            d -= e;
                        } else {
                            // Reuse one ADD+DBL body. Fixed role swaps avoid dynamic bn pointers.
                            int rule = ((d & 1) == (e & 1)) ? 1 : !(d & 1) ? 2 : 3;
                            if (rule == 1) { cgbn_swap(c._env, bx, cx); cgbn_swap(c._env, bz, cz); }
                            if (rule == 3) { cgbn_swap(c._env, ax, bx); cgbn_swap(c._env, az, bz); }
                            prac_add_selected<P, outlined>(c, cx, cz, ax, az, cx, cz, bx, bz, n, np0);
                            prac_dbl<P, compact>(c, ax, az, ax, az, a24, n, np0);
                            if (rule == 1) {
                                cgbn_swap(c._env, bx, cx); cgbn_swap(c._env, bz, cz); d = (d - e) / 2;
                            } else if (rule == 2) d /= 2;
                            else { cgbn_swap(c._env, ax, bx); cgbn_swap(c._env, az, bz); e /= 2; }
                        }
                    }
                    prac_add_selected<P, outlined>(c, ax, az, bx, bz, ax, az, cx, cz, n, np0);
                    }
                }
            }
        }
        cgbn_store(c._env, mem + 3, ax); cgbn_store(c._env, mem + 4, az);
    }
}

template<class P>
static cgbn_stage1_kernel_fn domain_kernel(int mode) {
    switch (mode) {
    case ECM_DOMAIN_INIT: return kernel_suyama_domain<P, ECM_DOMAIN_INIT>;
    case ECM_DOMAIN_EXPORT: return kernel_suyama_domain<P, ECM_DOMAIN_EXPORT>;
    case ECM_DOMAIN_LADDER: return kernel_suyama_domain<P, ECM_DOMAIN_LADDER>;
    case ECM_DOMAIN_PRAC: return kernel_suyama_domain<P, ECM_DOMAIN_PRAC>;
    case ECM_DOMAIN_PRAC_NATURAL:
        if constexpr (P::BITS > 2048 && P::BITS <= 8192)
            return kernel_suyama_domain<P, ECM_DOMAIN_PRAC_NATURAL>;
        else return kernel_suyama_domain<P, ECM_DOMAIN_PRAC>;
    case ECM_DOMAIN_PRAC_168:
        if constexpr (P::BITS == 4608 && P::TPI == 16)
            return kernel_suyama_domain<P, ECM_DOMAIN_PRAC_168>;
        else return nullptr;
    case ECM_DOMAIN_PRAC_COMPACT:
        if constexpr (P::BITS == 4608 && P::TPI == 16)
            return kernel_suyama_domain<P, ECM_DOMAIN_PRAC_COMPACT>;
        else return nullptr;
    case ECM_DOMAIN_PRAC_COMPACT_168:
        if constexpr (P::BITS == 4608 && P::TPI == 16)
            return kernel_suyama_domain<P, ECM_DOMAIN_PRAC_COMPACT_168>;
        else return nullptr;
    case ECM_DOMAIN_PRAC_OUTLINE_ADD:
        if constexpr (P::BITS == 4608 && P::TPI == 16)
            return kernel_suyama_domain<P, ECM_DOMAIN_PRAC_OUTLINE_ADD>;
        else return nullptr;
    case ECM_DOMAIN_PRAC_OUTLINE_ADD_168:
        if constexpr (P::BITS == 4608 && P::TPI == 16)
            return kernel_suyama_domain<P, ECM_DOMAIN_PRAC_OUTLINE_ADD_168>;
        else return nullptr;
    default: return nullptr;
    }
}
#endif
