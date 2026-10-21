#include "runtime.h"
bool exact_ok_terms(unsigned long long L, int bpw)
{
    if (bpw <= 0 || bpw > 62) return false;
    if (L == 0) return true;
    /* L*(2^bpw-1)^2 < L*2^(2bpw) < 2^(bitlen(L)+2bpw) and >= 2^(bitlen(L)+2bpw-3), so the
       bit length settles everything outside [2^62, 2^68).  Inside it, do it with GMP. */
    const int bl = bitlen_u64(L) + 2 * bpw;
    if (bl <= 62) return true;                            /* < 2^62 < p */
    if (bl >= 68) return false;                           /* >= 2^65 > p */
    mpz_t Lz, bz, wz, one, prod, p;
    mpz_inits(Lz, bz, wz, one, prod, p, nullptr);
    mpz_import(Lz, 1, -1, 8, 0, 0, &L);
    mpz_set_ui(one, 1);
    mpz_mul_2exp(bz, one, (unsigned long)bpw);
    mpz_sub_ui(bz, bz, 1);
    mpz_mul(wz, bz, bz);
    mpz_mul(prod, Lz, wz);
    const unsigned long long glp = GL_P;
    mpz_import(p, 1, -1, 8, 0, 0, &glp);
    const bool r = (mpz_cmp(prod, p) < 0);
    mpz_clears(Lz, bz, wz, one, prod, p, nullptr);
    return r;
}

bool exact_ok_nterms(unsigned long long n, int bpw)
{
    return exact_ok_terms(n, bpw);
}

double mpz_log2d(const mpz_t x)
{
    if (mpz_sgn(x) <= 0) return 0.0;
    long e = 0;
    const double d = mpz_get_d_2exp(&e, x);           /* d in [0.5, 1) */
    return (double)e + std::log2(d);
}

double mpz_log2d_p()
{
    mpz_t p;
    mpz_init(p);
    const unsigned long long glp = GL_P;
    mpz_import(p, 1, -1, 8, 0, 0, &glp);
    const double r = mpz_log2d(p);
    mpz_clear(p);
    return r;
}

double coeff_bound_bits_terms(unsigned long long L, int bpw)
{
    mpz_t Lz, bz, wz, one, prod;
    mpz_inits(Lz, bz, wz, one, prod, nullptr);
    mpz_import(Lz, 1, -1, 8, 0, 0, &L);
    mpz_set_ui(one, 1);
    mpz_mul_2exp(bz, one, (unsigned long)bpw);
    mpz_sub_ui(bz, bz, 1);
    mpz_mul(wz, bz, bz);
    mpz_mul(prod, Lz, wz);
    const double r = mpz_log2d(prod);
    mpz_clears(Lz, bz, wz, one, prod, nullptr);
    return r;
}

double bound_over_p_terms(unsigned long long L, int bpw)
{
    mpz_t Lz, bz, wz, one, prod, p;
    mpz_inits(Lz, bz, wz, one, prod, p, nullptr);
    mpz_import(Lz, 1, -1, 8, 0, 0, &L);
    mpz_set_ui(one, 1);
    mpz_mul_2exp(bz, one, (unsigned long)bpw);
    mpz_sub_ui(bz, bz, 1);
    mpz_mul(wz, bz, bz);
    mpz_mul(prod, Lz, wz);
    const unsigned long long glp = GL_P;
    mpz_import(p, 1, -1, 8, 0, 0, &glp);
    const double r = mpz_get_d(prod) / mpz_get_d(p);
    mpz_clears(Lz, bz, wz, one, prod, p, nullptr);
    return r;
}

double coeff_bound_bits(unsigned long long n, int bpw)
{
    mpz_t nz, bz, wz, one, prod;
    mpz_inits(nz, bz, wz, one, prod, nullptr);
    mpz_import(nz, 1, -1, 8, 0, 0, &n);
    mpz_set_ui(one, 1);
    mpz_mul_2exp(bz, one, (unsigned long)bpw);
    mpz_sub_ui(bz, bz, 1);
    mpz_mul(wz, bz, bz);
    mpz_mul(prod, nz, wz);
    const double r = mpz_log2d(prod);
    mpz_clears(nz, bz, wz, one, prod, nullptr);
    return r;
}

NttCfg choose_cfg(unsigned long long payload_bits, unsigned long long slot_bits,
                         unsigned long long P, int force_bpw)
{
    NttCfg c{};
    c.ok = false;
    c.L = 0;
    if (force_bpw != 0) {
        if (force_bpw < 1 || force_bpw > 62) {
            c.why = "the forced bpw is outside 1..62";
            return c;
        }
        const unsigned long long s =
            (slot_bits + (unsigned long long)force_bpw - 1) / (unsigned long long)force_bpw;
        if (P > 0 && P > (~0ull) / s) {
            c.why = "the forced bpw overflows L = P * ceil(slot_bits/bpw)";
            return c;
        }
        const unsigned long long l =
            (P > 0) ? (P * s)
                    : ((payload_bits + (unsigned long long)force_bpw - 1) /
                       (unsigned long long)force_bpw);
        if (!exact_ok_terms(l, force_bpw)) {
            c.why = "the forced bpw fails the exactness bound L*(2^bpw-1)^2 < p";
            return c;
        }
        c.slot_words = s;
        c.L = l;
        c.bpw = force_bpw;
        for (int k = 1; k <= 32; ++k) {
            const unsigned long long n = 1ull << k;
            if (n > (1ull << 32) / 8) break;
            const unsigned long long need_words =
                (P > 0) ? (2 * P * c.slot_words + 1)
                        : ((2 * payload_bits + (unsigned long long)force_bpw - 1) /
                           (unsigned long long)force_bpw + 1);
            if (n >= need_words) {
                c.nwords = n;
                c.k = k;
                c.ok = true;
                return c;
            }
        }
        c.bpw = 0;
        c.slot_words = 0;
        c.L = 0;
        c.why = "no power-of-two transform length holds the forced bpw's slot words";
        return c;
    }
    for (int k = 1; k <= 32; ++k) {
        const unsigned long long n = 1ull << k;
        if (n > (1ull << 32) / 8) break;
        /* largest bpw with L*(2^bpw - 1)^2 < p.  (The old code used N*(2^bpw)^2 < p and then
           subtracted one bit "for safety"; that is what forced bpw=18/N=2^24 on the P=8192
           shape.  With the term count L the comparison is exact and no fudge is needed.) */
        int bpw = 0;
        unsigned long long sw = 0, L = 0;
        for (int b = 62; b >= 1; --b) {
            const unsigned long long s =
                (slot_bits + (unsigned long long)b - 1) / (unsigned long long)b;
            if (P > 0 && P > (~0ull) / s) continue;          /* L would overflow: skip b */
            const unsigned long long l =
                (P > 0) ? (P * s)
                        : ((payload_bits + (unsigned long long)b - 1) / (unsigned long long)b);
            if (!exact_ok_terms(l, b)) continue;
            bpw = b;
            sw = s;
            L = l;
            break;
        }
        if (bpw <= 0) continue;
        c.slot_words = sw;
        c.L = L;
        /* word-aligned footprint: 2*P*slot_words digits, plus the top digit the assembler
           peeks at.  (With P unknown -- the generic `bench` mode -- fall back to the
           bit-based estimate.) */
        const unsigned long long need_words =
            (P > 0) ? (2 * P * c.slot_words + 1)
                    : ((2 * payload_bits + (unsigned long long)bpw - 1) /
                       (unsigned long long)bpw + 1);
        if (n >= need_words) {
            c.nwords = n;
            c.k = k;
            c.bpw = bpw;
            c.ok = true;
            return c;
        }
    }
    c.why = "no power-of-two transform length up to 2^32 fits this payload: the exactness "
            "bound L*(2^bpw-1)^2 < p cannot be met for ANY bpw, or no transform length "
            "reaches 2*P*ceil(slot_bits/bpw)+1 digits";
    return c;
}

bool max_conv_coeff_gmp(const std::vector<unsigned long long> &hA,
                               const std::vector<unsigned long long> &hB, unsigned long long L,
                               unsigned long long *out_max, unsigned long long *out_k)
{
    *out_max = 0;
    *out_k = 0;
    mpz_t A, B, C;
    mpz_inits(A, B, C, nullptr);
    mpz_import(A, (size_t)L, -1, 8, 0, 0, hA.data());
    mpz_import(B, (size_t)L, -1, 8, 0, 0, hB.data());
    mpz_mul(C, A, B);
    const size_t nc = mpz_size(C);
    std::vector<unsigned long long> limbs(nc + 1, 0);
    size_t cnt = 0;
    mpz_export(limbs.data(), &cnt, -1, 8, 0, 0, C);
    unsigned long long best = 0, bk = 0;
    for (size_t i = 0; i < cnt; ++i) {
        if (limbs[i] > best) { best = limbs[i]; bk = (unsigned long long)i; }
    }
    mpz_clears(A, B, C, nullptr);
    *out_max = best;
    *out_k = bk;
    return true;
}

double digit_sq_double(int bpw)
{
    const long double d = (long double)((1ull << bpw) - 1ull);
    return (double)(d * d);
}

void mpz_add_u64(mpz_t acc, unsigned long long v)
{
    if (v == 0) return;
    mpz_t t;
    mpz_init(t);
    mpz_import(t, 1, -1, 8, 0, 0, &v);
    mpz_add(acc, acc, t);
    mpz_clear(t);
}

void dump_cmp(const char *tag, const std::vector<unsigned long long> &d, const mpz_t C,
                     unsigned long long n, int bpw, int show)
{
    mpz_t x, t;
    mpz_inits(x, t, nullptr);
    mpz_set_ui(x, 0);
    for (unsigned long long j = n; j-- > 0;) {
        mpz_mul_2exp(x, x, (unsigned long)bpw);
        mpz_add_u64(x, d[(size_t)j]);
    }
    const bool value_ok = (mpz_cmp(x, C) == 0);
    /* difference as an integer: x - C (small if only the low digits are off) */
    mpz_sub(t, x, C);
    const long dbl = (mpz_sgn(t) == 0) ? 0 : (long)mpz_sizeinbase(t, 2);
    unsigned long long first = 0, bad = 0;
    mpz_t mask, cj;
    mpz_inits(mask, cj, nullptr);
    mpz_set_ui(mask, 1);
    mpz_mul_2exp(mask, mask, (unsigned long)bpw);
    mpz_sub_ui(mask, mask, 1);
    mpz_t cur;
    mpz_init_set(cur, C);
    for (unsigned long long j = 0; j < n; ++j) {
        mpz_and(cj, cur, mask);
        if (mpz_get_ui(cj) != d[(size_t)j]) {
            if (bad == 0) first = j;
            ++bad;
        }
        mpz_fdiv_q_2exp(cur, cur, (unsigned long)bpw);
    }
    stage2_log::print(stage2_log::debug, "  %s: digits_bad=%llu/%llu first_bad_digit=%llu value_eq_exact=%d "
                "|value_diff|_bits=%ld\n", tag, bad, n, first, value_ok ? 1 : 0, dbl);
    stage2_log::print(stage2_log::debug, "    exact");
    {
        mpz_t cur2, m2;
        mpz_init_set(cur2, C);
        mpz_init(m2);
        for (unsigned long long j = 0; j < (unsigned long long)show && j < n; ++j) {
            mpz_and(m2, cur2, mask);
            stage2_log::print(stage2_log::debug, " [%llu]=%llu", j, (unsigned long long)mpz_get_ui(m2));
            mpz_fdiv_q_2exp(cur2, cur2, (unsigned long)bpw);
        }
        stage2_log::print(stage2_log::debug, "\n");
        mpz_clears(cur2, m2, nullptr);
    }
    stage2_log::print(stage2_log::debug, "    %s   ", tag);
    for (unsigned long long j = 0; j < (unsigned long long)show && j < n; ++j)
        stage2_log::print(stage2_log::debug, " [%llu]=%llu", j, d[(size_t)j]);
    stage2_log::print(stage2_log::debug, "\n");
    mpz_clears(x, t, mask, cj, cur, nullptr);
}

bool ntt_shape_query(unsigned long long P, int S, unsigned long long *N_out, int *bpw_out,
                            unsigned long long *slot_bits_out, unsigned long long *slot_words_out,
                            unsigned long long *slot_stride_out, unsigned long long *out_slots_out,
                            int force_bpw)
{
    unsigned long long log2P = 1;
    while ((1ull << log2P) < P) ++log2P;
    const unsigned long long slot_bits = 2ull * (unsigned long long)S + log2P;
    const NttCfg cfg = choose_cfg(P * slot_bits, slot_bits, P, force_bpw);
    if (!cfg.ok) return false;
    if (N_out) *N_out = cfg.nwords;
    if (bpw_out) *bpw_out = cfg.bpw;
    if (slot_bits_out) *slot_bits_out = slot_bits;
    if (slot_words_out) *slot_words_out = cfg.slot_words;
    if (slot_stride_out) *slot_stride_out = cfg.slot_words * (unsigned long long)cfg.bpw;
    if (out_slots_out) *out_slots_out = 2 * P - 1;
    return true;
}

int ntt_shape_plan(unsigned long long P, int S, int device, NttArena *arena,
                          FuseCtx &fc, NttShape &sh, int force_bpw)
{
    stage2_trace::Range trace("NTT.plan");
    static std::vector<NttShapeCacheEntry> memo;
    for (const NttShapeCacheEntry &e : memo) {
        if (e.P == P && e.S == S && e.device == device && e.force_bpw == force_bpw) {
            sh = e.sh;
            ntt_arena_fuse(arena, sh.N, sh.k, sh.omega, sh.omega_inv, fc);
            return 0;
        }
    }
    const int rc = ntt_shape_plan_uncached(P, S, device, arena, fc, sh, force_bpw);
    if (rc == 0) {
        NttShapeCacheEntry e;
        e.P = P;
        e.S = S;
        e.device = device;
        e.force_bpw = force_bpw;
        e.sh = sh;
        memo.push_back(e);
    }
    return rc;
}

int ntt_shape_plan_uncached(unsigned long long P, int S, int device, NttArena *arena,
                                   FuseCtx &fc, NttShape &sh, int force_bpw)
{
    if (P == 0 || S <= 0) {
        std::fprintf(stderr, NTT_PROBE_NAME ": bad shape P=%llu S=%d\n",
                     (unsigned long long)P, S);
        return 2;
    }
    CK(cudaSetDevice(device));
    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, device));
    CK(cudaFree(0));
    std::snprintf(sh.dev_name, sizeof(sh.dev_name), "%s", prop.name);
    sh.device = device;
    sh.P = P;
    sh.S = S;
    unsigned long long log2P = 1;
    while ((1ull << log2P) < P) ++log2P;
    sh.slot_bits = 2ull * (unsigned long long)S + log2P;
    sh.payload_bits = P * sh.slot_bits;

    const NttCfg cfg = choose_cfg(sh.payload_bits, sh.slot_bits, P, force_bpw);
    if (!cfg.ok) {
        std::fprintf(stderr, NTT_PROBE_NAME ": %s\n", cfg.why.c_str());
        return 2;
    }
    sh.N = cfg.nwords;
    sh.bpw = cfg.bpw;
    sh.k = cfg.k;
    /* WORD-ALIGNED slot stride: coefficient i is packed at bit i*slot_stride, and
       slot_stride = ceil(slot_bits/bpw)*bpw keeps every operand digit < 2^bpw. */
    sh.slot_words = cfg.slot_words;
    sh.slot_stride = sh.slot_words * (unsigned long long)sh.bpw;
    sh.out_slots = 2 * P - 1;

    if (sh.slot_bits > sh.slot_stride || sh.slot_stride > (unsigned long long)sh.bpw * sh.N ||
        (2 * P - 1) * sh.slot_words + sh.slot_words > sh.N) {
        std::fprintf(stderr,
                     NTT_PROBE_NAME ": ASSERT FAILED slot_bits=%llu slot_stride=%llu needs "
                     "%llu words of N=%llu\n", (unsigned long long)sh.slot_bits,
                     (unsigned long long)sh.slot_stride,
                     (unsigned long long)((2 * P - 1) * sh.slot_words + sh.slot_words),
                     (unsigned long long)sh.N);
        return 3;
    }
    sh.W = (unsigned long long)(S + 63) / 64;

    /* twiddle base: omega = 7^((p-1)/N).  Checked: omega^N == 1 and omega^(N/2) == p-1
       (i.e. omega really is a primitive N-th root), and N^{-1} mod p really inverts N. */
    const unsigned long long root = 7ull;
    sh.omega = gl_pow_host(root, (GL_P - 1ull) / sh.N);
    sh.omega_inv = gl_pow_host(sh.omega, GL_P - 2ull);            /* inverse root */
    if (gl_pow_host(sh.omega, sh.N) != 1ull || gl_pow_host(sh.omega, sh.N / 2) != GL_P - 1ull) {
        std::fprintf(stderr, NTT_PROBE_NAME ": omega is not a primitive %llu-th root\n",
                     (unsigned long long)sh.N);
        return 3;
    }
    if (gl_mul_host(sh.omega, sh.omega_inv) != 1ull) {
        std::fprintf(stderr, NTT_PROBE_NAME ": omega * omega_inv != 1\n");
        return 3;
    }
    sh.n_scale = gl_pow_host(sh.N % GL_P, GL_P - 2ull);           /* N^-1 mod p */
    if (gl_mul_host(sh.N % GL_P, sh.n_scale) != 1ull) {
        std::fprintf(stderr, NTT_PROBE_NAME ": 1/N failed\n");
        return 3;
    }

    sh.mem_mb = (double)(4 * sh.N * sizeof(unsigned long long) +
                         sh.out_slots * sizeof(unsigned long long)) / 1048576.0;
    if (P > (~0ull) / sh.slot_words) {
        std::fprintf(stderr, NTT_PROBE_NAME ": L = P*slot_words overflows 64 bits\n");
        return 3;
    }
    sh.L_terms = P * sh.slot_words;
    if (!exact_ok_terms(sh.L_terms, sh.bpw)) {
        std::fprintf(stderr, NTT_PROBE_NAME ": EXACTNESS VIOLATED: L=%llu (P=%llu * "
                             "slot_words=%llu) * (2^%d - 1)^2 >= p -- the convolution would "
                             "wrap mod p silently\n", (unsigned long long)sh.L_terms,
                     (unsigned long long)sh.P, (unsigned long long)sh.slot_words, sh.bpw);
        return 6;
    }
    sh.bound_bits = coeff_bound_bits_terms(sh.L_terms, sh.bpw);
    sh.bound_ratio = bound_over_p_terms(sh.L_terms, sh.bpw);
    sh.old_bound_bits = coeff_bound_bits(sh.N, sh.bpw);           /* the old N-based form */
    sh.bound_ok = exact_ok_terms(sh.L_terms, sh.bpw);
    sh.old_bound_ok = exact_ok_nterms(sh.N, sh.bpw);
    sh.p_bits = mpz_log2d_p();

    /* fusion plan (section 12): t stages in the shared-memory tile pass, the rest fused into
       radix-2^M register passes.  PASS COUNT = number of FULL ARRAY PASSES. */
    ntt_arena_fuse(arena, sh.N, sh.k, sh.omega, sh.omega_inv, fc);
    sh.passes_fwd = fc.passes_fwd;
    sh.passes_total = 2 * fc.passes_fwd + fc.passes_fwd + 1;      /* +1 = the carry cone */
    sh.carry_rounds = 2 + (sh.k + 2 * sh.bpw + sh.bpw - 1) / sh.bpw;
    /* NTT_CARRY_ROUNDS raises the height-reduction rounds for diagnostic comparisons.
       Long binary carry chains are resolved exactly, independently of this override. */
    {
        const char *e = std::getenv("NTT_CARRY_ROUNDS");
        if (e && *e) {
            const int v = std::atoi(e);
            if (v > sh.carry_rounds) sh.carry_rounds = v;
        }
    }
    return 0;
}

int ntt_shape_exactness(const NttShape &sh)
{
    if (sh.P > (~0ull) / sh.slot_words) {
        std::fprintf(stderr, NTT_PROBE_NAME ": L = P*slot_words overflows 64 bits\n");
        return 3;
    }
    if (!exact_ok_terms(sh.L_terms, sh.bpw)) {
        std::fprintf(stderr, NTT_PROBE_NAME ": EXACTNESS VIOLATED: L=%llu (P=%llu * "
                             "slot_words=%llu) * (2^%d - 1)^2 >= p -- the convolution would "
                             "wrap mod p silently\n", (unsigned long long)sh.L_terms,
                     (unsigned long long)sh.P, (unsigned long long)sh.slot_words, sh.bpw);
        return 6;
    }
    return 0;
}

int ntt_shape_maxcoeff(const NttShape &sh, const std::vector<uint64_t> &hA,
                              const std::vector<uint64_t> &hB, unsigned long long *maxcoeff,
                              unsigned long long *maxcoeff_k, bool *ran,
                              unsigned long long *limbs)
{
    /* the empirical maximum is a check on the BOUND DERIVATION (are the digits really < 2^bpw,
       is the convolution really below p), so it is run once per SHAPE: at the frozen shape the
       S4 engine launches ~1.8e3 batches and running a GMP multiply in every one of them is pure
       cost.  The counting bound itself is still asserted on EVERY call, and the canonical-digit
       property is checked on every call by the reduction kernel. */
    static std::vector<unsigned long long> done;
    const unsigned long long maxcoeff_limbs = fuse_env_ull("NTT_MAXCOEFF_LIMBS", 8000000ull);
    if (limbs) *limbs = maxcoeff_limbs;
    *maxcoeff = 0;
    *maxcoeff_k = 0;
    for (unsigned long long key : done)
        if (key == sh.L_terms * 4096ull + (unsigned long long)sh.bpw) { *ran = false; return 0; }
    *ran = (maxcoeff_limbs > 0 && sh.L_terms <= maxcoeff_limbs);
    if (!*ran) return 0;
    done.push_back(sh.L_terms * 4096ull + (unsigned long long)sh.bpw);
    max_conv_coeff_gmp(hA, hB, sh.L_terms, maxcoeff, maxcoeff_k);
    if (*maxcoeff >= GL_P) {
        std::fprintf(stderr, NTT_PROBE_NAME ": EXACTNESS VIOLATED (measured): the true "
                             "maximum convolution coefficient is %llu >= p = %llu "
                             "(k=%llu, L=%llu, bpw=%d) -> the NTT would wrap silently\n",
                     (unsigned long long)*maxcoeff, (unsigned long long)GL_P,
                     (unsigned long long)*maxcoeff_k, (unsigned long long)sh.L_terms, sh.bpw);
        return 7;
    }
    return 0;
}
