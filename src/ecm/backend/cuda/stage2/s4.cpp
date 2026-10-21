// S4 modulus/shape setup and reduction hooks.
#include "s4.h"
#include "options.h"
#include "checks.h"
#include "point_launch.h"

unsigned long long *pin_words(unsigned long long **slot,size_t *cap,size_t words)
{
    stage2_trace::Range trace("host.pinned_reserve");
    if (*slot && *cap >= words) return *slot;
    if (*slot) { (void)cudaFreeHost(*slot); *slot = nullptr; *cap = 0; }
    void *q = nullptr;
    if (cudaHostAlloc(&q, words * sizeof(unsigned long long), cudaHostAllocDefault) != cudaSuccess) {
        (void)cudaGetLastError();       /* no pinned memory: the caller keeps the blocking path */
        return nullptr;
    }
    *slot = (unsigned long long *)q;
    *cap = words;
    return *slot;
}

void S4Reduce::Shape::dt_destroy()
{
        for (int i = 0; i < DTRING; ++i)
            if (dt_used[i]) {
                cudaEventDestroy(dt_ev[i][0]);
                cudaEventDestroy(dt_ev[i][1]);
                dt_ev[i][0] = dt_ev[i][1] = nullptr;
                dt_used[i] = false;
            }
    }

void S4Reduce::Shape::dt_reap()
{
        float ms = 0.0f;
        for (int i = 0; i < DTRING; ++i) {
            if (!dt_used[i]) continue;
            if (cudaEventQuery(dt_ev[i][1]) != cudaSuccess) continue;
            if (cudaEventElapsedTime(&ms, dt_ev[i][0], dt_ev[i][1]) == cudaSuccess)
                t_reduce += (double)ms * 1e-3;
            cudaEventDestroy(dt_ev[i][0]);
            cudaEventDestroy(dt_ev[i][1]);
            dt_ev[i][0] = dt_ev[i][1] = nullptr;
            dt_used[i] = false;
        }
    }

int S4Reduce::Shape::dt_slot()
{
        dt_reap();
        for (int i = 0; i < DTRING; ++i)
            if (!dt_used[i]) return i;
        ++dt_blocks;
        dt_flush();
        return 0;
    }

void S4Reduce::Shape::dt_flush()
{
        for (int i = 0; i < DTRING; ++i) {
            if (!dt_used[i]) continue;
            cudaEventSynchronize(dt_ev[i][1]);
            float ms = 0.0f;
            if (cudaEventElapsedTime(&ms, dt_ev[i][0], dt_ev[i][1]) == cudaSuccess)
                t_reduce += (double)ms * 1e-3;
            cudaEventDestroy(dt_ev[i][0]);
            cudaEventDestroy(dt_ev[i][1]);
            dt_ev[i][0] = dt_ev[i][1] = nullptr;
            dt_used[i] = false;
        }
    }

S4Reduce::S4Reduce()
{ mpz_init(N); }

S4Reduce::Shape *S4Reduce::find(unsigned long long slot_bits,unsigned long long slot_words,int bpw) const
{
    for (Shape *s : shapes)
        if (s->slot_bits == slot_bits && s->slot_words == slot_words && s->bpw == bpw)
            return s;
    return nullptr;
}

int s4_reduce_init(
    S4Reduce &R,
    const mpz_t N,
    size_t W,
    const std::vector<unsigned long long> &hn,
    unsigned long long ninv)
{
    stage2_trace::Range trace("init.S4_constants");
    R.nw = (int)W;
    R.w = W;
    R.ninv = ninv;
    R.hn = hn;
    mpz_set(R.N, N);
    mpz_t plus;mpz_init(plus);mpz_add_ui(plus,N,1);
    const bool exact_mersenne=mpz_cmp_ui(N,1)>0 && mpz_popcount(plus)==1;
    R.mersenne_bits=(g_s4_mersenne && exact_mersenne) ? (int)mpz_sizeinbase(N,2) : 0;
    mpz_clear(plus);
    s4_oracle_reset(R);
    s4_gmp_pack_check();
    CK(cudaMalloc(&R.dn, (size_t)R.nw * sizeof(unsigned long long)));
    CK(cudaMemcpy(R.dn, R.hn.data(), (size_t)R.nw * sizeof(unsigned long long),
                  cudaMemcpyHostToDevice));
    /* ---- THE DIVISION CONSTANTS, AND THE PRIMITIVE CHECKED AGAINST GMP (section 31) ---------
       N << dshift (top bit set) and its reciprocal word, then 256 numerators per modulus compared
       against GMP -- random ones plus the saturating extreme u1 = d-1, u0 = all ones.  A mismatch is
       FATAL: section 35.1 of the plan document records that this area was once optimised with wrong
       maths and caught by a test, so the arithmetic is checked before anything depends on it. */
    {
        const unsigned long long top = R.hn[(size_t)R.nw - 1];
        int sh = 0;
        while (sh < 64 && ((top << sh) & (1ull << 63)) == 0) ++sh;
        if (sh == 64) sh = 0;
        g_div_shift = sh;
        g_div_nw = R.nw;
        g_div_hns.assign((size_t)R.nw, 0ull);
        for (int i = R.nw - 1; i >= 0; --i) {
            const unsigned long long v = R.hn[(size_t)i];
            g_div_hns[(size_t)i] = (sh == 0)
                ? v : ((v << sh) | (i > 0 ? (R.hn[(size_t)i - 1] >> (64 - sh)) : 0ull));
        }
        CK(ecm_stage2_copy_constants( g_div_hns.data(),
                              (size_t)R.nw * sizeof(unsigned long long)));
        const unsigned long long d = g_div_hns[(size_t)R.nw - 1];
        mpz_t num, den, q, t64;
        mpz_inits(num, den, q, t64, nullptr);
        mpz_set_ui(num, 1);
        mpz_mul_2exp(num, num, 128);
        mpz_sub_ui(num, num, 1);
        mpz_import(den, 1, -1, 8, 0, 0, &d);          /* exact, whatever sizeof(unsigned long) is */
        mpz_fdiv_q(q, num, den);
        mpz_set_ui(t64, 1);
        mpz_mul_2exp(t64, t64, 64);
        mpz_sub(q, q, t64);
        g_div_recip = 0;
        mpz_export(&g_div_recip, nullptr, -1, 8, 0, 0, q);
        {
            unsigned long long cases = 0, bad = 0, seed = 0x9e3779b97f4a7c15ull;
            mpz_t n2, qq, rr, gq;
            mpz_inits(n2, qq, rr, gq, nullptr);
            for (int t = 0; t < 256; ++t) {
                seed = seed * 6364136223846793005ull + 1442695040888963407ull;
                const unsigned long long u1 = (t == 1) ? (d - 1) : (seed % d);
                seed = seed * 6364136223846793005ull + 1442695040888963407ull;
                const unsigned long long u0 = (t == 1) ? ~0ull : seed;
                if (u1 >= d) continue;
                const unsigned long long got = s2g_udiv_2by1(u1, u0, d, g_div_recip);
                unsigned long long two[2] = {u0, u1};      /* little-endian: low word first */
                mpz_import(n2, 2, -1, 8, 0, 0, two);
                mpz_fdiv_qr(qq, rr, n2, den);
                mpz_import(gq, 1, -1, 8, 0, 0, &got);
                ++cases;
                if (mpz_cmp(gq, qq) != 0) {
                    ++bad;
                    if (bad <= 3)
                        std::fprintf(stderr, "%s: udiv_2by1 MISMATCH u1=%llu u0=%llu got=%llu\n",
                                     NTT_PROBE_NAME, u1, u0, got);
                }
            }
            mpz_clears(n2, qq, rr, gq, nullptr);
            stage2_log::print(stage2_log::debug, "s4_udiv_check: cases=%llu bad=%llu (2-by-1 division vs GMP, dshift=%d)\n",
                        cases, bad, g_div_shift);
            if (bad) {
                std::fprintf(stderr, "%s: FATAL: the 2-by-1 division primitive disagrees with GMP "
                                     "-- refusing to continue\n", NTT_PROBE_NAME);
                std::exit(3);
            }
        }
        mpz_clears(num, den, q, t64, nullptr);
    }
    s4_div_check(R.hn);
    stage2_log::print(stage2_log::debug, "s4_reduce_mode: algorithm=%s nw=%d dshift=%d (NTT_S4_OLDTAIL=%d)\n",
                s4_tail_mont_mode() ? "montgomery" : (R.mersenne_bits ? "mersenne" : "division"), R.nw, g_div_shift,
                s4_tail_mont_mode());
    stage2_log::print(stage2_log::debug, "s4_mersenne_mode: requested=%d eligible=%d bits=%d enabled=%d\n",(int)g_s4_mersenne,
                (int)exact_mersenne,R.mersenne_bits,(int)(R.mersenne_bits && !s4_tail_mont_mode()));
    const char *fixture=std::getenv("NTT_S4_MERSENNE_TEST");
    if(fixture && std::atoi(fixture))s4_mersenne_check();
    return 0;
}

S4Reduce::Shape *s4_shape_init(
    S4Reduce &R,
    unsigned long long P,
    unsigned long long slot_bits,
    unsigned long long slot_stride,
    unsigned long long slot_words,
    int bpw)
{
    S4Reduce::Shape *S = R.find(slot_bits, slot_words, bpw);
    if (S) return S;
    S = new S4Reduce::Shape();
    S->slot_bits = slot_bits;
    S->slot_stride = slot_stride;
    S->slot_words = slot_words;
    S->bpw = bpw;
    S->P = P;
    S->nlimb = (int)((slot_stride + 63) / 64);
    mpz_t Cmax, Rl, prod;
    mpz_inits(Cmax, Rl, prod, nullptr);
    mpz_set_ui(Cmax, 1);
    mpz_mul_2exp(Cmax, Cmax, (unsigned long)slot_bits);   /* Cmax = 2^slot_bits (exclusive) */
    int L = 1;
    for (;; ++L) {
        mpz_set_ui(Rl, 1);
        mpz_mul_2exp(Rl, Rl, (unsigned long)(64 * L));
        mpz_mul(prod, R.N, Rl);
        if (mpz_cmp(Cmax, prod) <= 0) break;
        if (L > 4096) {
            std::fprintf(stderr, "%s: FATAL: cannot find L with 2^slot_bits <= N*2^(64L) "
                                 "(slot_bits=%llu, N has %d words)\n", NTT_PROBE_NAME,
                         slot_bits, R.nw);
            std::exit(3);
        }
    }
    S->L = L;
    S->bound_bits = (double)mpz_sizeinbase(prod, 2) - (double)slot_bits;
    /* Y = 2^(64 (L+nw)) mod N: goes from the 2^(-64L) domain to the plain one in one multiply */
    {
        mpz_t Y;
        mpz_init(Y);
        mpz_set_ui(Y, 1);
        mpz_mul_2exp(Y, Y, (unsigned long)(64 * (L + R.nw)));
        mpz_mod(Y, Y, R.N);
        S->hy.assign((size_t)R.nw, 0ull);
        if (!mpz_to_words(S->hy, (size_t)R.nw, Y)) {
            std::fprintf(stderr, "%s: FATAL: Y does not fit %d words\n", NTT_PROBE_NAME, R.nw);
            std::exit(3);
        }
        mpz_clear(Y);
    }
    mpz_clears(Cmax, Rl, prod, nullptr);
    /* the device array is 2*NW+4 limbs with NW the dispatched template width: the digit->limb
       conversion must fit nlimb+1 of them and the elimination loop reaches L+nw+2 */
    int nwmax = ecm_stage2::max_words;
    for (int v : {4, 8, 16, 32, 64, 128, 256}) if (R.nw <= v) { nwmax = v; break; }
    const long long need = (S->nlimb + 1) > (L + R.nw + 2) ? (S->nlimb + 1) : (L + R.nw + 2);
    if (need > 2 * nwmax + 4) {
        std::fprintf(stderr, "%s: FATAL: the reduction needs %lld limbs but the %d-word "
                             "template only has %d\n", NTT_PROBE_NAME, need, nwmax,
                     2 * nwmax + 4);
        std::exit(3);
    }
    CK(cudaMalloc(&S->dy, (size_t)R.nw * sizeof(unsigned long long)));
    CK(cudaMemcpy(S->dy, S->hy.data(), (size_t)R.nw * sizeof(unsigned long long),
                  cudaMemcpyHostToDevice));
    R.shapes.push_back(S);
    stage2_log::print(stage2_log::debug, "s4_reduce_shape: P=%llu slot_bits=%llu slot_stride=%llu nlimb=%d L=%d "
                "(per-coefficient preprint: %d digits base 2^%d -> %d limbs, then %d "
                "elimination steps of %d limbs) ; precondition C < 2^slot_bits <= N*2^(64L) "
                "proved with GMP for THIS N: margin=%.1f bits\n",
                P, slot_bits, slot_stride, S->nlimb, S->L, (int)slot_words, bpw, S->nlimb,
                S->L, R.nw, S->bound_bits);
    return S;
}

void s4_reduce_hook(
    void *ctx,
    const unsigned long long *digits,
    unsigned long long n,
    int bpw,
    unsigned long long slot_words,
    unsigned long long slot_bits,
    unsigned long long out_slots,
    unsigned long long nbatch,
    unsigned long long *out,
    unsigned long long w,
    unsigned long long first,
    unsigned long long count)
{
    S4Reduce &R = *(S4Reduce *)ctx;
    S4Reduce::Shape *S = R.find(slot_bits, slot_words, bpw);
    if (!S) {
        std::fprintf(stderr, "%s: FATAL: the reduction was never initialised for this shape "
                             "(slot_bits=%llu bpw=%d slot_words=%llu)\n", NTT_PROBE_NAME,
                     slot_bits, bpw, slot_words);
        std::exit(3);
    }
    if (w != R.w) {
        std::fprintf(stderr, "%s: FATAL: the hook was given w=%llu words per coefficient but "
                             "the modulus has %d\n", NTT_PROBE_NAME, w, R.nw);
        std::exit(3);
    }
    const unsigned long long total = out_slots * nbatch;
    if (!R.dbad) {
        CK(cudaMalloc(&R.dbad, sizeof(unsigned long long)));
        R.dbad_host = 0;
    }
    const double t0 = now_s();
    /* THE COUNTER'S RESET IS NOT PER-CALL WORK (objective 4, section 36).  `dbad` counts slot
       windows whose digits overflowed the window: it is MONOTONE and purely diagnostic, yet the
       old code reset it with a cudaMemset AND read it back with an 8-byte device-to-host copy on
       EVERY call.  A small pageable D2H is pure latency here (~10-20 us; bulk D2H runs at
       2.9 GB/s), so at 589960 calls that is seconds and at the production shape's 3.8e6 calls it
       is minutes -- all of it inside "the unattributed part of ntt_seconds".  Now it is reset
       once per shape and read every g_s4_check_every calls (the GMP oracle's cadence), so a
       violation is still reported within 8 calls of happening. */
    if (S->calls == 0) {
        /* ... AND THE PENDING READBACK MUST BE RESOLVED BEFORE THE RESET (section 43): `dbad` is
           shared by every shape and is zeroed when a shape starts, so a copy still in flight
           would read the post-reset zero and look exactly like a violation.  Once per shape, so
           the wait is free. */
        s4_dbad_resolve(R);
        CK(cudaMemset(R.dbad, 0, sizeof(unsigned long long)));
    }
    /* MARK THE KERNEL IN-STREAM AND RESOLVE IT LATER (section 41): no cudaDeviceSynchronize()
       here any more -- see the note on `t_reduce` in S4Reduce::Shape. */
    const int dts = S->dt_slot();
    CK(cudaEventCreate(&S->dt_ev[dts][0]));
    CK(cudaEventCreate(&S->dt_ev[dts][1]));
    CK(cudaEventRecord(S->dt_ev[dts][0]));
    S2G_DISPATCH(R.nw, s4_launch_reduce, (int)R.nw, S->L, nbatch, out_slots, total, digits, n,
                 bpw, slot_words, R.dn, R.ninv, S->dy, w, out, slot_bits, R.dbad, nullptr, first, count,R.mersenne_bits);
    CK(cudaGetLastError());
    CK(cudaEventRecord(S->dt_ev[dts][1]));
    S->dt_used[dts] = true;
    S->t_reduce_host += now_s() - t0;
    const double th0 = now_s();
    /* ---- THE DIAGNOSTIC READBACK MUST NOT DRAIN EITHER (objective 4, section 43) -----------
       This is the SAME trap section 41 found in the carry-residual check, in the hook instead of
       the multiply: an 8-byte blocking copy issued right after a chunk's worth of kernels has
       been queued waits for all of them, and it MEASURED 19.2 s at the production shape (1946
       calls, ~10 ms each) for a counter that is never read as a control input.  `dbad` is
       monotone, so reading it LATE is exactly as good: the copy is asynchronous into pinned
       memory and the PREVIOUS one is compared on the next call (and at the end of the run).
       A violation is therefore still reported -- one call later instead of immediately. */
    s4_dbad_resolve(R);
    if (!R.dbad_inflight && (S->calls % g_s4_check_every) == 0) {
        if (!R.h_dbad) {
            CK(cudaHostAlloc((void **)&R.h_dbad, sizeof(unsigned long long), cudaHostAllocDefault));
            *R.h_dbad = 0;
            CK(cudaEventCreate(&R.ev_dbad));
        }
        CK(cudaMemcpyAsync(R.h_dbad, R.dbad, sizeof(unsigned long long),
                           cudaMemcpyDeviceToHost));
        CK(cudaEventRecord(R.ev_dbad));
        R.dbad_inflight = true;
        R.dbad_shape = S;
        R.dbad_expected = S->canon_bad;
    }
    ++R.reduce_calls;
    S->t_hookd2h += now_s() - th0;
    ++S->calls;
    S->coeffs += count*nbatch;
    R.coeffs_total += count*nbatch;
    /* This reduction is already queued: comparing an OLD snapshot can overlap it. */
    const double tr0 = now_s();
    s4_oracle_reap(R, false);
    g_oracle.t_reap += now_s() - tr0;
    /* THE IN-RUN ORACLE, here rather than at the call site: the digit buffer belongs to the
       multiply and is released when it returns, so a check done afterwards would read freed
       device memory (that was a real crash, "CUDA error invalid argument", the first time the
       S2 tail ran without the arena).  Reading it here is also the stronger check: the raw
       digits GMP sees are the ones the transform and the carry just produced. */
    if (g_s4_sample_limit > 0 && (S->calls <= 1 || (S->calls % g_s4_check_every) == 0)) {
        const double ts0 = now_s();
        s4_check_reduced(R, S, digits, n, out_slots, nbatch, out, g_s4_sample_limit, first, count);
        S->t_hooksample += now_s() - ts0;
    }
}
