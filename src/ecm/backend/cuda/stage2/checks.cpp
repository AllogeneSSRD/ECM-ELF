#include "host_wait.h"
// Mandatory arithmetic checks, samples, asynchronous snapshots and draining.
#include "checks.h"
#include "options.h"
#include "point_launch.h"

void s4_oracle_reset(S4Reduce &R)
{
    if (g_oracle_owner && g_oracle_owner != &R) {
        std::fprintf(stderr, "%s: FATAL: overlapping modulus oracle owners\n", NTT_PROBE_NAME);
        std::exit(3);
    }
    g_oracle_owner = &R;
    g_oracle = S4OracleStats{};
    stage2_log::print(stage2_log::debug, "s4_oracle_mode: async=%d ring=%d pack=%d (same samples and GMP predicate)\n",
                (int)g_s4_oracle_async, g_s4_oracle_ring, (int)g_s4_oracle_pack);
}

void s4_oracle_report()
{
    stage2_log::print(stage2_log::debug, "s4_oracle_stats: async=%d selected=%llu queued=%llu compared=%llu samples=%llu "
                "pending=%llu ring_waits=%llu fallbacks=%llu signature=%016llx pinned_peak=%llu "
                "t_wait=%.6f t_copy_host=%.6f t_gmp=%.6f t_alloc=%.6f "
                "t_capture=%.6f t_reap=%.6f t_drain=%.6f host_total=%.6f "
                "pack=%d t_num=%.6f t_mod=%.6f\n",
                (int)g_s4_oracle_async, g_oracle.selected, g_oracle.queued, g_oracle.compared,
                g_oracle.samples, g_oracle.tail - g_oracle.head, g_oracle.ring_waits,
                g_oracle.fallbacks, g_oracle.signature, g_oracle.pinned_peak,
                g_oracle.t_wait, g_oracle.t_copy, g_oracle.t_gmp, g_oracle.t_alloc,
                g_oracle.t_capture, g_oracle.t_reap, g_oracle.t_drain,
                g_oracle.t_capture + g_oracle.t_reap + g_oracle.t_drain,
                (int)g_s4_oracle_pack, g_oracle.t_num, g_oracle.t_mod);
}

void s4_dbad_resolve(S4Reduce &R,bool wait)
{
    if (!R.dbad_inflight) return;
    if (wait) CK(stage2_wait::event_sync(R.ev_dbad,stage2_wait::Kind::canonical));
    else {const auto status=cudaEventQuery(R.ev_dbad);if(status==cudaErrorNotReady)return;CK(status);}
    R.dbad_inflight = false;
    const unsigned long long hbad = *R.h_dbad;
    S4Reduce::Shape *S = R.dbad_shape;
    if (S && hbad != R.dbad_expected) {
        const unsigned long long added = (hbad > R.dbad_expected) ? (hbad - R.dbad_expected)
                                                                 : hbad;
        std::fprintf(stderr, "%s: FATAL: %llu (of %llu so far) slot windows of P=%llu have "
                             "nonzero digits above slot_bits=%llu -- the reduction bound "
                             "C < 2^slot_bits does NOT apply to these values\n", NTT_PROBE_NAME,
                     added, S->coeffs, S->P, S->slot_bits);
        std::exit(3);
    }
    if (S) S->canon_bad = hbad;
}

void s4_mersenne_check()
{
    mpz_t N,v,want,got;mpz_inits(N,v,want,got,nullptr);
    unsigned long long cases=0,words=0,bad=0,folds=0,seed=0x493caf523ull;
    for(int bits : {2,3,31,63,64,65,127,128,129,255,256,4423,5261,8191,8192,8193,16381,16383,16384}) {
        const int nw=(bits+63)/64,stride=2*nw+4,ncase=48;
        mpz_set_ui(N,1);mpz_mul_2exp(N,N,bits);mpz_sub_ui(N,N,1);
        std::vector<unsigned long long> input(ncase*stride),expected(ncase*nw),gpu(expected.size());
        std::vector<int> lengths(ncase),counts(ncase);
        for(int k=0;k<ncase;++k) {
            auto *t=input.data()+(size_t)k*stride;
            if(k==0)mpz_set_ui(v,0);
            else if(k<4){mpz_set(v,N);if(k==1)mpz_sub_ui(v,v,1);if(k==3)mpz_add_ui(v,v,1);}
            else if(k<7){mpz_mul(v,N,N);if(k==4)mpz_sub_ui(v,v,1);if(k==6)mpz_add_ui(v,v,1);}
            else if(k<10){mpz_set_ui(v,1);mpz_mul_2exp(v,v,2*bits+20);if(k==7)mpz_sub_ui(v,v,1);if(k==9)mpz_add_ui(v,v,1);}
            else {
                for(int i=0;i<stride-1;++i){seed=seed*6364136223846793005ull+1;t[i]=seed;}
                if(k==10)for(int i=0;i<stride-1;++i)t[i]=~0ull;
                mpz_import(v,stride-1,-1,8,0,0,t);
            }
            std::fill(t,t+stride,0ull);size_t len=0;mpz_export(t,&len,-1,8,0,0,v);lengths[k]=(int)len;
            mpz_mod(want,v,N);mpz_export(expected.data()+(size_t)k*nw,nullptr,-1,8,0,0,want);
            unsigned long long scratch[2*ecm_stage2::max_words+4]={},u[ecm_stage2::max_words];std::copy(t,t+stride,scratch);
            const int f=nw<=128 ? s4_mersenne_rem<128>(scratch,(int)len,nw,bits,u)
                : s4_mersenne_rem<256>(scratch,(int)len,nw,bits,u);
            mpz_import(got,nw,-1,8,0,0,u);if(mpz_cmp(want,got))++bad;
            folds+=f;
        }
        unsigned long long *di=nullptr,*doo=nullptr;int *dl=nullptr,*dc=nullptr;
        CK(cudaMalloc(&di,input.size()*8));CK(cudaMalloc(&doo,gpu.size()*8));
        CK(cudaMalloc(&dl,ncase*sizeof(int)));CK(cudaMalloc(&dc,ncase*sizeof(int)));
        CK(cudaMemcpy(di,input.data(),input.size()*8,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dl,lengths.data(),ncase*sizeof(int),cudaMemcpyHostToDevice));
        if(nw<=128)ecm_launch_s4_mersenne_fixture_kernel<128>(EcmCudaLaunch{1,64}, di,stride,dl,ncase,nw,bits,doo,dc);
        else ecm_launch_s4_mersenne_fixture_kernel<256>(EcmCudaLaunch{1,64}, di,stride,dl,ncase,nw,bits,doo,dc);
        CK(cudaGetLastError());
        CK(cudaMemcpy(gpu.data(),doo,gpu.size()*8,cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(counts.data(),dc,ncase*sizeof(int),cudaMemcpyDeviceToHost));
        CK(cudaFree(di));CK(cudaFree(doo));CK(cudaFree(dl));CK(cudaFree(dc));
        const char *fault=std::getenv("NTT_S4_MERSENNE_TEST_BAD");if(fault && std::atoi(fault))gpu[0]^=1;
        for(size_t i=0;i<gpu.size();++i)if(gpu[i]!=expected[i])++bad;
        cases+=ncase;words+=gpu.size();
    }
    mpz_clears(N,v,want,got,nullptr);
    stage2_log::print(stage2_log::debug, "s4_mersenne_check: cases=%llu words=%llu folds=%llu bad=%llu (CPU/GPU vs GMP, S=2..16384)\n",cases,words,folds,bad);
    if(bad){std::fprintf(stderr,"FATAL: Mersenne remainder GMP mismatch\n");std::exit(3);}
}

void s4_div_check(const std::vector<unsigned long long> &actual)
{
    mpz_t den, num, want, got, norm, q, dtop, radix;
    mpz_inits(den, num, want, got, norm, q, dtop, radix, nullptr);
    mpz_set_ui(radix, 1);
    mpz_mul_2exp(radix, radix, 64);
    unsigned long long cases = 0, bad = 0, repairs = 0, seed = 0x89abcdef01234567ull;
    const int widths[] = {1, 2, 3, 8, 83, 128, 129, 256};
    for (int fixture = 0; fixture < 1+3*(int)(sizeof(widths)/sizeof(widths[0])); ++fixture) {
        const int nw = (fixture == 0) ? (int)actual.size() : widths[(fixture - 1) / 3];
        std::vector<unsigned long long> hn((size_t)nw), ns((size_t)nw);
        for (int i = 0; i < nw; ++i) {
            seed = seed * 6364136223846793005ull + 1442695040888963407ull;
            hn[(size_t)i] = seed;
        }
        if (fixture == 0) hn = actual;
        else {
            const int kind = (fixture - 1) % 3;
            hn[(size_t)nw - 1] = (kind == 0) ? 0x8000000000000000ull
                : (kind == 1 ? 1ull : 0x1ffffffffull);
            hn[0] |= 1ull;
        }
        mpz_import(den, (size_t)nw, -1, 8, 0, 0, hn.data());
        int shift = 0;
        while ((hn[(size_t)nw - 1] << shift) < 0x8000000000000000ull) ++shift;
        mpz_mul_2exp(norm, den, (unsigned)shift);
        mpz_export(ns.data(), nullptr, -1, 8, 0, 0, norm);
        mpz_import(dtop, 1, -1, 8, 0, 0, &ns[(size_t)nw - 1]);
        mpz_set_ui(num, 1);
        mpz_mul_2exp(num, num, 128);
        mpz_sub_ui(num, num, 1);
        mpz_fdiv_q(q, num, dtop);
        mpz_sub(q, q, radix);
        unsigned long long recip = 0;
        mpz_export(&recip, nullptr, -1, 8, 0, 0, q);
        for (int c = 0; c < 32; ++c) {
            unsigned long long t[2*ecm_stage2::max_words+4] = {}, out[ecm_stage2::max_words] = {};
            if (c < 3) { mpz_set(num, den); if (c == 0) mpz_sub_ui(num, num, 1);
                          if (c == 2) mpz_add_ui(num, num, 1); }
            else if (c == 3) {
                mpz_sub_ui(num, den, 1); mpz_mul_2exp(num, num, 64);
                mpz_add(num, num, radix); mpz_sub_ui(num, num, 1);
            } else if (c < 7) {
                mpz_mul(num, den, den);
                if (c == 4) mpz_sub_ui(num, num, 1);
                if (c == 6) { mpz_add(num, num, den); mpz_sub_ui(num, num, 1); }
            } else if (c == 7) mpz_set_ui(num, 0);
            else {
                for (int i = 0; i < 2 * nw; ++i) {
                    seed = seed * 6364136223846793005ull + 1442695040888963407ull;
                    t[i] = seed;
                }
                mpz_import(num, (size_t)2 * nw, -1, 8, 0, 0, t);
            }
            std::memset(t, 0, sizeof(t));
            size_t limbs = 0;
            mpz_export(t, &limbs, -1, 8, 0, 0, num);
            const int rc = nw<=128 ? s4_div_rem<128>(t, (int)limbs, ns.data(), nw, shift, recip, out)
                : s4_div_rem<256>(t, (int)limbs, ns.data(), nw, shift, recip, out);
            mpz_mod(want, num, den);
            mpz_import(got, (size_t)nw, -1, 8, 0, 0, out);
            ++cases;
            if (rc < 0 || mpz_cmp(want, got) != 0) {
                ++bad;
                if (bad <= 3) std::fprintf(stderr, "%s: div_rem MISMATCH fixture=%d case=%d "
                                                    "nw=%d shift=%d rc=%d\n",
                                           NTT_PROBE_NAME, fixture, c, nw, shift, rc);
            } else repairs += (unsigned long long)rc;
        }
    }
    mpz_clears(den, num, want, got, norm, q, dtop, radix, nullptr);
    stage2_log::print(stage2_log::debug, "s4_div_check: cases=%llu bad=%llu repairs=%llu (full remainder vs GMP, "
                "widths=1..256 shifts=0..63)\n", cases, bad, repairs);
    if (bad || repairs == 0) {
        std::fprintf(stderr, "%s: FATAL: long division failed its GMP/borrow-repair fixtures\n",
                     NTT_PROBE_NAME);
        std::exit(3);
    }
}

void s4_gmp_assemble(mpz_t v,const unsigned long long *digits,unsigned long long nslots,int bpw)
{
    std::vector<unsigned long long> words((size_t)((nslots * (unsigned long long)bpw + 63) / 64) + 2, 0ull);
    for (unsigned long long j = 0; j < nslots; ++j) {
        const unsigned long long bit = j * (unsigned long long)bpw, value = digits[j];
        const size_t k = (size_t)(bit / 64);
        const int shift = (int)(bit % 64);
        const unsigned long long low = value << shift, old = words[k];
        words[k] = old + low;
        const unsigned long long carry = (words[k] < old) ? 1ull : 0ull;
        const unsigned long long high = shift ? (value >> (64 - shift)) : 0ull;
        const unsigned long long before = words[k + 1], sum = before + high;
        const unsigned long long next = sum + carry;
        words[k + 1] = next;
        bool c = (sum < before) || (next < sum);
        for (size_t i = k + 2; c; ++i) {
            if (i >= words.size()) {
                std::fprintf(stderr, "%s: FATAL: oracle assembly carry escaped its bound\n", NTT_PROBE_NAME);
                std::exit(3);
            }
            c = (++words[i] == 0);
        }
    }
    mpz_import(v, words.size(), -1, sizeof(unsigned long long), 0, 0, words.data());
}

void s4_gmp_assemble_slow(mpz_t v,const unsigned long long *digits,unsigned long long nslots,int bpw)
{
    mpz_set_ui(v, 0);
    for (unsigned long long j = nslots; j-- > 0;) {
        mpz_mul_2exp(v, v, (unsigned)bpw);
        mpz_add_u64(v, digits[j]);
    }
}

void s4_gmp_pack_check()
{
    stage2_trace::Range trace("check.GMP_pack");
    const int counts[] = {0, 1, 2, 3, 63, 64, 65, 502, 1024};
    const int bases[] = {1, 19, 21, 31, 32, 63, 64};
    unsigned long long seed = 0x1234fedcba987654ull, cases = 0, bad = 0;
    mpz_t want, got;
    mpz_inits(want, got, nullptr);
    for (int n : counts) for (int b : bases) for (int pattern = 0; pattern < 6; ++pattern) {
        std::vector<unsigned long long> digits((size_t)n, 0ull);
        for (int i = 0; i < n; ++i) {
            seed = seed * 6364136223846793005ull + 1442695040888963407ull;
            const unsigned long long mask = (b == 64) ? ~0ull : ((1ull << b) - 1ull);
            digits[(size_t)i] = pattern == 0 ? 0ull : pattern == 1 ? (seed & mask)
                : pattern == 2 ? ~0ull : pattern == 3 ? (i == n - 1 ? ~0ull : 0ull)
                : pattern == 4 ? seed : ((i & 1) ? ~0ull : 1ull);
        }
        s4_gmp_assemble_slow(want, digits.data(), (unsigned long long)n, b);
        s4_gmp_assemble(got, digits.data(), (unsigned long long)n, b);
        ++cases;
        if (mpz_cmp(want, got) != 0) ++bad;
    }
    mpz_clears(want, got, nullptr);
    stage2_log::print(stage2_log::debug, "s4_oracle_pack_check: cases=%llu bad=%llu (exact integers, bpw=1..64, "
                "canonical AND noncanonical digits)\n", cases, bad);
    if (bad) {
        std::fprintf(stderr, "%s: FATAL: packed oracle assembly disagrees with GMP\n", NTT_PROBE_NAME);
        std::exit(3);
    }
}

void s4_gmp_reduce(
    mpz_t out,
    const unsigned long long *digits,
    unsigned long long nslots,
    int bpw,
    const mpz_t N,
    bool timed)
{
    mpz_t v;
    mpz_init(v);
    double t0 = timed ? now_s() : 0.0;
    if (g_s4_oracle_pack) s4_gmp_assemble(v, digits, nslots, bpw);
    else s4_gmp_assemble_slow(v, digits, nslots, bpw);
    if (timed) { g_oracle.t_num += now_s() - t0; t0 = now_s(); }
    mpz_mod(out, v, N);
    if (timed) g_oracle.t_mod += now_s() - t0;
    mpz_clear(v);
}

int s4_reduce_selftest(S4Reduce &R,S4Reduce::Shape *S)
{
    stage2_trace::Range trace("check.S4_selftest");
    const unsigned long long cases = 96;
    const unsigned long long slot_words = S->slot_words;
    const int bpw = S->bpw;
    const unsigned long long maxd = (1ull << bpw) - 1ull;
    std::vector<unsigned long long> dig((size_t)(cases * slot_words), 0),
                                    got((size_t)(cases * R.w), 0);
    uint64_t s = 0x243f6a8885a308d3ull;
    /* THE EXACT WINDOW THE S5 DESCENT FAILED ON (section 47): 0xfc861cfcb724b25400000001, the
       one value for which the REDC invariant `r == V*2^-(64L) mod N` was measured to break, in
       limb 0 only.  It is put in as a CASE so that "the same shape passes on synthetic values and
       fails on the real one" stops being a paradox and becomes a one-second reproduction. */
    const bool have_real_case = (S->slot_bits == 259ull && bpw == 7 && slot_words == 37);
    unsigned long long real_dig[64] = {0};
    if (have_real_case) {
        mpz_t v, two, q, rem;
        mpz_inits(v, two, q, rem, nullptr);
        mpz_set_str(v, "fc861cfcb724b25400000001", 16);
        mpz_set_ui(two, 1);
        mpz_mul_2exp(two, two, (unsigned)bpw);
        /* GMP's fdiv_qr requires all four variables to be DISTINCT (the first version aliased
           the remainder with the dividend, so this case did not contain the window it claimed
           to -- which is why "the same shape passes in isolation" survived one round of being
           turned into a unit test at all, section 48). */
        for (unsigned long long j = 0; j < slot_words; ++j) {
            mpz_fdiv_qr(q, rem, v, two);
            real_dig[j] = mpz_get_ui(rem);
            mpz_set(v, q);
        }
        mpz_clears(v, two, q, rem, nullptr);
        /* prove the decomposition, digit by digit: rebuilt must equal the hex we started from */
        {
            mpz_t back;
            mpz_init_set_ui(back, 0);
            for (unsigned long long j = slot_words; j-- > 0;) {
                mpz_mul_2exp(back, back, (unsigned)bpw);
                mpz_add_ui(back, back, real_dig[j]);
            }
            char *bs = mpz_get_str(nullptr, 16, back);
            stage2_log::print(stage2_log::debug, "s4_reduce_realcase: rebuilt=%s ok=%d\n", bs,
                        (std::strcmp(bs, "fc861cfcb724b25400000001") == 0) ? 1 : 0);
            void (*ff)(void *, size_t) = nullptr;
            mp_get_memory_functions(nullptr, nullptr, &ff);
            ff(bs, std::strlen(bs) + 1);
            mpz_clear(back);
        }
    }
    for (unsigned long long c = 0; c < cases; ++c) {
        for (unsigned long long j = 0; j < slot_words; ++j) {
            s = s * 6364136223846793005ull + 1442695040888963407ull;
            unsigned long long v;
            switch (c % 7) {
                case 0: v = maxd; break;                                  /* all ones */
                case 1: v = 0; break;                                     /* zero */
                case 2: v = (j + 1 == slot_words) ? 1ull : 0ull; break;   /* single top bit */
                case 3: v = (j == 0) ? maxd : 0ull; break;                /* low digit full */
                /* ---- A DENSE WINDOW WHOSE VALUE IS BELOW N (section 46) --------------------
                   The S5 descent failed on exactly this shape of window: 96 bits, i.e. TWO limbs
                   both fully populated, and below N so the reduction must hand the value back
                   unchanged.  Every other pattern here is either >= N (all ones, single top bit,
                   the random default) or occupies one limb (zero, low digit full), so the one
                   case that was wrong was the one case not covered.  A test suite is only as
                   good as the shapes it contains. */
                case 6: v = (j < (slot_words * 2ull) / 5ull) ? (s & maxd) : 0ull; break;
                /* the recorded real failure (section 47) */
                case 5: v = have_real_case ? real_dig[j] : (s & maxd); break;
                default: v = s & maxd; break;
            }
            dig[(size_t)(c * slot_words + j)] = v;
        }
    }
    /* the last digit must respect the slot window, exactly like a real product coefficient */
    {
        const unsigned long long top_bits = S->slot_bits - (slot_words - 1) *
                                            (unsigned long long)bpw;
        if (top_bits < 64)
            for (unsigned long long c = 0; c < cases; ++c)
                dig[(size_t)(c * slot_words + slot_words - 1)] &= ((1ull << top_bits) - 1ull);
    }
    unsigned long long *dd = nullptr, *dout = nullptr, *ddbg = nullptr;
    /* ---- THE GEOMETRY A/B (section 48) ----------------------------------------------------
       Every launch argument here is identical to the S5 multiply's EXCEPT `n` (the slice stride),
       `out_slots` and `total`: this selftest uses `cases*slot_words / cases / cases`, the real call
       uses `N / out_slots / out_slots*nbatch`.  NTT_S5_SELFTEST_N makes this selftest run with the
       REAL stride, which is the last variable left between "the same window passes in isolation"
       and "it fails in situ". */
    unsigned long long sel_n = cases * slot_words;
    {
        const char *e = std::getenv("NTT_S5_SELFTEST_N");
        if (e && *e) {
            const unsigned long long v = std::strtoull(e, nullptr, 10);
            if (v >= slot_words) sel_n = v;
        }
    }
    const unsigned long long sel_os = (sel_n < cases * slot_words) ? (sel_n / slot_words) : cases;
    if (sel_os < cases)
        stage2_log::print(stage2_log::debug, "s4_reduce_selftest_geom: n=%llu out_slots=%llu (the real call's stride)\n",
                    sel_n, sel_os);
    constexpr bool dbg=false; // Legacy REDC forensic dump is development-only.
    CK(cudaMalloc(&dd, dig.size() * sizeof(unsigned long long)));
    CK(cudaMalloc(&dout, (size_t)(cases * R.w) * sizeof(unsigned long long)));
    if (dbg) CK(cudaMalloc(&ddbg, 4 * 24 * sizeof(unsigned long long)));
    CK(cudaMemcpy(dd, dig.data(), dig.size() * sizeof(unsigned long long),
                  cudaMemcpyHostToDevice));
    S2G_DISPATCH(R.nw, s4_launch_reduce, (int)R.nw, S->L, 1ull, sel_os, sel_os, dd,
                 sel_n, bpw, slot_words, R.dn, R.ninv, S->dy,
                 (unsigned long long)R.w, dout, S->slot_bits, (unsigned long long *)nullptr, ddbg,0,~0ull,R.mersenne_bits);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(got.data(), dout, got.size() * sizeof(unsigned long long),
                  cudaMemcpyDeviceToHost));
    if (dbg) {
        std::vector<unsigned long long> hb(64, 0ull);
        CK(cudaMemcpy(hb.data(), ddbg, hb.size() * sizeof(unsigned long long),
                      cudaMemcpyDeviceToHost));
        for (int c = 0; c < 4; ++c) {
            char *ds = nullptr;
            {
                mpz_t dv;
                mpz_init(dv);
                for (unsigned long long j = slot_words; j-- > 0;) {
                    mpz_mul_2exp(dv, dv, (unsigned)bpw);
                    mpz_add_u64(dv, dig[(size_t)(c * slot_words + j)]);
                }
                ds = mpz_get_str(nullptr, 16, dv);
                mpz_clear(dv);
            }
            std::fprintf(stderr, "s4_dbg: case=%d L=%llu bpw=%llu sw=%llu d0..2=[%llu,%llu,%llu] "
                                 "dlast=%llu t=[%llx,%llx,%llx,%llx,%llx,%llx] "
                                 "r=[%llx,%llx,%llx] window=%s\n", c, hb[(size_t)(c * 16 + 9)],
                         hb[(size_t)(c * 16 + 15)], hb[(size_t)(c * 16 + 10)],
                         hb[(size_t)(c * 16 + 11)], hb[(size_t)(c * 16 + 12)],
                         hb[(size_t)(c * 16 + 13)], hb[(size_t)(c * 16 + 14)],
                         hb[(size_t)(c * 16 + 0)], hb[(size_t)(c * 16 + 1)],
                         hb[(size_t)(c * 16 + 2)], hb[(size_t)(c * 16 + 3)],
                         hb[(size_t)(c * 16 + 4)], hb[(size_t)(c * 16 + 5)],
                         hb[(size_t)(c * 16 + 6)], hb[(size_t)(c * 16 + 7)],
                         hb[(size_t)(c * 16 + 8)], ds);
            /* the limbs the kernel's own conversion would produce from these digits, computed on
               the host with the same rule, so a mismatch localises to the conversion */
            {
                std::vector<unsigned long long> tl(2 * (size_t)R.nw + 4, 0ull);
                unsigned long long acc = 0, nacc = 0, limb = 0;
                for (unsigned long long j = 0; j < slot_words; ++j) {
                    const unsigned long long v = dig[(size_t)(c * slot_words + j)];
                    acc |= (v << nacc);
                    if ((unsigned long long)bpw + nacc >= 64) {
                        tl[(size_t)limb++] = acc;
                        acc = (nacc == 0) ? 0ull : (v >> (64 - nacc));
                        nacc = nacc + (unsigned long long)bpw - 64;
                    } else {
                        nacc += (unsigned long long)bpw;
                    }
                }
                if (nacc) tl[(size_t)limb++] = acc;
                std::fprintf(stderr, "s4_dbg_hostconv: case=%d limbs_written=%llu "
                                     "tl=[%llx,%llx,%llx,%llx,%llx,%llx]\n", c, limb,
                             tl[0], tl[1], tl[2], tl[3], tl[4], tl[5]);
            }
            void (*ff)(void *, size_t) = nullptr;
            mp_get_memory_functions(nullptr, nullptr, &ff);
            ff(ds, std::strlen(ds) + 1);
        }
        cudaFree(ddbg);
    }
    cudaFree(dd);
    cudaFree(dout);
    unsigned long long bad = 0;
    long long first = -1;
    mpz_t want, mine;
    mpz_inits(want, mine, nullptr);
    for (unsigned long long c = 0; c < sel_os; ++c) {
        s4_gmp_reduce(want, &dig[(size_t)(c * slot_words)], slot_words, bpw, R.N);
        mpz_import(mine, (size_t)R.w, -1, 8, 0, 0, &got[(size_t)(c * R.w)]);
        if (mpz_cmp(want, mine) != 0) {
            if (!bad) {
                first = (long long)c;
                char *ws = mpz_get_str(nullptr, 16, want);
                char *ms = mpz_get_str(nullptr, 16, mine);
                stage2_log::print(stage2_log::debug, "s4_reduce_selftest_bad: case=%llu pattern=%llu gmp=%s gpu=%s\n",
                            (unsigned long long)c, (unsigned long long)(c % 6), ws, ms);
                void (*ff)(void *, size_t) = nullptr;
                mp_get_memory_functions(nullptr, nullptr, &ff);
                ff(ws, std::strlen(ws) + 1);
                ff(ms, std::strlen(ms) + 1);
            }
            ++bad;
        }
    }
    mpz_clears(want, mine, nullptr);
    S->selftest_cases = cases;
    S->selftest_bad = bad;
    S->selftest_first = first;
    if (std::getenv("NTT_S5_REDDUMP") && *std::getenv("NTT_S5_REDDUMP")
        && std::atoi(std::getenv("NTT_S5_REDDUMP")) != 0) {
        char *ws = mpz_get_str(nullptr, 16, want);
        char *ms = mpz_get_str(nullptr, 16, mine);
        std::fprintf(stderr, "s4_selftest_dump: P=%llu slot_bits=%llu slot_words=%llu bpw=%d "
                             "L=%d nlimb=%d nw=%d w=%d last_case gmp=%s gpu=%s\n", S->P,
                     S->slot_bits, S->slot_words, S->bpw, S->L, S->nlimb, R.nw, (int)R.w, ws, ms);
        void (*ff)(void *, size_t) = nullptr;
        mp_get_memory_functions(nullptr, nullptr, &ff);
        ff(ws, std::strlen(ws) + 1);
        ff(ms, std::strlen(ms) + 1);
    }
    stage2_log::print(stage2_log::debug, "s4_reduce_selftest: P=%llu cases=%llu mismatches=%llu first_bad=%lld (device "
                "REDC of a %llu-bit coefficient mod the actual N vs GMP; six digit patterns "
                "incl. all-ones/zero/single-top-bit/single-low-digit)\n", S->P, cases, bad,
                first, S->slot_bits);
    return bad ? 1 : 0;
}

void s4_compare_snapshot(
    S4Reduce &R,
    S4Reduce::Shape *S,
    unsigned long long s,
    unsigned long long k0,
    unsigned long long cnt,
    unsigned long long out_slots,
    const unsigned long long *digbuf,
    const unsigned long long *redbuf)
{
    stage2_trace::Range trace("check.GMP_snapshot");
    const double t0 = now_s();
    mpz_t want, mine;
    mpz_inits(want, mine, nullptr);
        for (unsigned long long q = 0; q < cnt; ++q) {
            s4_gmp_reduce(want, &digbuf[(size_t)(q * S->slot_words)], S->slot_words, S->bpw,
                          R.N, true);
            mpz_import(mine, (size_t)R.w, -1, 8, 0, 0, &redbuf[(size_t)(q * R.w)]);
            ++S->checked;
            if (mpz_cmp(want, mine) != 0) {
                if (!S->check_bad) {
                    S->check_first = s * out_slots + k0 + q;
                    char *ws = mpz_get_str(nullptr, 16, want);
                    char *ms = mpz_get_str(nullptr, 16, mine);
                    /* ---- AND THE TWO THINGS THAT SPLIT THE CAUSE (section 51) ------------
                       A dictionary of the failure without its context is what cost rounds 14-17:
                       report whether THIS window's digits are canonical (the kernel's conversion
                       ORs them in, so a digit >= 2^bpw loses its high bits) and what the REDC
                       invariant requires r to be, so "the carry did not canonicalise" and "the
                       elimination is wrong" are distinguishable from one line. */
                    unsigned long long mx = 0, mj = 0;
                    for (unsigned long long jj = 0; jj < S->slot_words; ++jj) {
                        const unsigned long long v = digbuf[(size_t)(q * S->slot_words + jj)];
                        if (v > mx) { mx = v; mj = jj; }
                    }
                    mpz_t tv, rexp, tw;
                    mpz_inits(tv, rexp, tw, nullptr);
                    for (unsigned long long jj = S->slot_words; jj-- > 0;) {
                        mpz_mul_2exp(tv, tv, (unsigned)S->bpw);
                        mpz_add_u64(tv, digbuf[(size_t)(q * S->slot_words + jj)]);
                    }
                    mpz_set_ui(tw, 1);
                    mpz_mul_2exp(tw, tw, (mp_bitcnt_t)(64 * S->L));
                    mpz_invert(tw, tw, R.N);
                    mpz_mul(rexp, tv, tw);
                    mpz_mod(rexp, rexp, R.N);
                    char *rs = mpz_get_str(nullptr, 16, rexp);
                    stage2_log::print(stage2_log::debug, "s4_reduce_CHECK_bad: slice=%llu k=%llu gmp=%s gpu=%s "
                                "max_digit=%llu/%llu(mj=%llu)%s rexp=%s\n",
                                s, k0 + q, ws, ms, mx, (1ull << S->bpw) - 1ull, mj,
                                (mx < (1ull << S->bpw)) ? " CANON" : " NONCANON", rs);
                    void (*ff)(void *, size_t) = nullptr;
                    mp_get_memory_functions(nullptr, nullptr, &ff);
                    ff(ws, std::strlen(ws) + 1);
                    ff(ms, std::strlen(ms) + 1);
                    ff(rs, std::strlen(rs) + 1);
                    mpz_clears(tv, rexp, tw, nullptr);
                }
                ++S->check_bad;
                std::fprintf(stderr, "%s: FATAL: the device reduction disagrees with GMP at "
                                     "slice %llu coefficient %llu\n", NTT_PROBE_NAME, s, k0 + q);
                std::exit(3);
            }
        }
    mpz_clears(want, mine, nullptr);
    ++g_oracle.compared;
    g_oracle.samples += cnt;
    g_oracle.t_gmp += now_s() - t0;
}

bool s4_oracle_one(S4Reduce &R,bool wait)
{
    if (g_oracle.head == g_oracle.tail) return false;
    S4OracleSlot &slot = g_oracle_slot[g_oracle.head % g_s4_oracle_ring];
    const cudaError_t status = cudaEventQuery(slot.ready);
    if (status == cudaErrorNotReady && !wait) return false;
    if (status == cudaErrorNotReady) {
        const double t0 = now_s();
        CK(stage2_wait::event_sync(slot.ready,stage2_wait::Kind::oracle));
        g_oracle.t_wait += now_s() - t0;
    } else CK(status);
    s4_compare_snapshot(R, slot.shape, slot.slice, slot.k0, slot.count, slot.out_slots,
                        slot.digits, slot.reduced);
    ++g_oracle.head;
    return true;
}

void s4_oracle_reap(S4Reduce &R,bool wait)
{
    if (g_oracle_owner != &R) return;
    while (s4_oracle_one(R, wait)) {}
}

void s4_oracle_drain(S4Reduce &R)
{
    if (g_oracle_owner != &R) return;
    const double t0 = now_s();
    /* Gate-only fault injection: corrupt the LAST pending host snapshot, specifically proving
       that the final drain validates it before success or destruction. Never alter device data. */
    const char *bad = std::getenv("NTT_S4_ORACLE_TEST_BAD");
    if (bad && std::atoi(bad) != 0 && g_oracle.head != g_oracle.tail) {
        S4OracleSlot &slot = g_oracle_slot[(g_oracle.tail - 1) % g_s4_oracle_ring];
        CK(stage2_wait::event_sync(slot.ready,stage2_wait::Kind::oracle));
        slot.reduced[0] ^= 1ull;
    }
    s4_oracle_reap(R, true);
    if (g_oracle.selected != g_oracle.compared) {
        std::fprintf(stderr, "%s: FATAL: oracle lost snapshots (%llu selected, %llu compared)\n",
                     NTT_PROBE_NAME, g_oracle.selected, g_oracle.compared);
        std::exit(3);
    }
    g_oracle.t_drain += now_s() - t0;
}

void s4_oracle_release(S4Reduce &R)
{
    if (g_oracle_owner != &R) return;
    s4_oracle_drain(R);
    for (S4OracleSlot &slot : g_oracle_slot) {
        if (slot.digits) CK(cudaFreeHost(slot.digits));
        if (slot.reduced) CK(cudaFreeHost(slot.reduced));
        if (slot.ready) CK(cudaEventDestroy(slot.ready));
        slot = S4OracleSlot{};
    }
    if (g_oracle_block_ev) CK(cudaEventDestroy(g_oracle_block_ev));
    g_oracle_block_ev = nullptr;
    g_oracle_owner = nullptr;
}

void s4_oracle_block_ready()
{
    if (!g_oracle_block_ev) CK(cudaEventCreateWithFlags(&g_oracle_block_ev, cudaEventDisableTiming));
    CK(cudaEventRecord(g_oracle_block_ev));
    const double t0 = now_s();
    CK(stage2_wait::event_sync(g_oracle_block_ev,stage2_wait::Kind::oracle));
    g_oracle.t_wait += now_s() - t0;
}

void s4_check_reduced(
    S4Reduce &R,
    S4Reduce::Shape *S,
    const unsigned long long *ddig,
    unsigned long long n,
    unsigned long long out_slots,
    unsigned long long nbatch,
    const unsigned long long *d_out,
    long long sample_limit,
    unsigned long long first,
    unsigned long long count)
{
    if(count==~0ull) count=out_slots;
    const unsigned long long total = count * nbatch;
    if (!total || sample_limit == 0) return;
    const double tc0 = now_s();
    const unsigned long long lim = (unsigned long long)sample_limit;
    const bool small_batch = (nbatch <= 4 && total <= lim);
    std::vector<std::array<unsigned long long, 3>> runs;
    if (small_batch) {
        for (unsigned long long s = 0; s < nbatch; ++s) runs.push_back({s, 0ull, count});
        ++S->full_checks;
    } else {
        const unsigned long long cnt = std::min(lim, count);
        uint64_t seed = 0x9e3779b97f4a7c15ull * (S->calls + 1) + total;
        seed = seed * 6364136223846793005ull + 1442695040888963407ull;
        const unsigned long long s = (seed >> 11) % nbatch;
        seed = seed * 6364136223846793005ull + 1442695040888963407ull;
        const unsigned long long room = (count > cnt) ? (count - cnt + 1) : 1;
        runs.push_back({s, (seed >> 11) % room, cnt});
    }
    if (!g_s4_oracle_async) s4_oracle_block_ready();
    for (const auto &rr : runs) {
        const unsigned long long s = rr[0], k0 = rr[1], source_k=first+k0, cnt = rr[2];
        const size_t ndig = (size_t)(cnt * S->slot_words), nred = (size_t)(cnt * R.w);
        ++g_oracle.selected;
        for (const auto v : {S->P, S->slot_bits, S->slot_words,
                            (unsigned long long)S->bpw, S->calls, s, source_k, cnt})
            g_oracle.signature = (g_oracle.signature ^ v) * 1099511628211ull;
        bool captured = false;
        if (g_s4_oracle_async) {
            if (g_oracle.tail - g_oracle.head == (unsigned long long)g_s4_oracle_ring) {
                ++g_oracle.ring_waits;
                s4_oracle_one(R, true);     /* never overwrite a pending HOST snapshot */
            }
            S4OracleSlot &slot = g_oracle_slot[g_oracle.tail % g_s4_oracle_ring];
            const double ta0 = now_s();
            auto *dig = pin_words(&slot.digits, &slot.digcap, ndig);
            auto *red = pin_words(&slot.reduced, &slot.redcap, nred);
            g_oracle.t_alloc += now_s() - ta0;
            unsigned long long bytes = 0;
            for (const auto &q : g_oracle_slot) bytes += 8ull * (q.digcap + q.redcap);
            g_oracle.pinned_peak = std::max(g_oracle.pinned_peak, bytes);
            if (dig && red) {
                if (!slot.ready) CK(cudaEventCreateWithFlags(&slot.ready, cudaEventDisableTiming));
                slot.shape = S; slot.slice = s; slot.k0 = source_k; slot.count = cnt;
                slot.out_slots = out_slots;
                const double t0 = now_s();
                CK(cudaMemcpyAsync(dig, ddig + s * n + source_k * S->slot_words,
                                   ndig * 8, cudaMemcpyDeviceToHost));
                CK(cudaMemcpyAsync(red, d_out + (s * count + k0) * R.w,
                                   nred * 8, cudaMemcpyDeviceToHost));
                CK(cudaEventRecord(slot.ready));
                g_oracle.t_copy += now_s() - t0;
                ++g_oracle.tail; ++g_oracle.queued;
                captured = true;
            } else { ++g_oracle.fallbacks; s4_oracle_block_ready(); }
        }
        if (!captured) {
            std::vector<unsigned long long> dig(ndig), red(nred);
            const double t0 = now_s();
            CK(cudaMemcpy(dig.data(), ddig + s * n + source_k * S->slot_words,
                          ndig * 8, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(red.data(), d_out + (s * count + k0) * R.w,
                          nred * 8, cudaMemcpyDeviceToHost));
            g_oracle.t_copy += now_s() - t0;
            s4_compare_snapshot(R, S, s, source_k, cnt, out_slots, dig.data(), red.data());
        }
    }
    g_oracle.t_capture += now_s() - tc0;
}

int mont_selftest(
    const std::vector<unsigned long long> &hn,
    size_t nw,
    unsigned long long ninv,
    const mpz_t N,
    const mpz_t R)
{
    stage2_trace::Range trace("check.Montgomery");
    const bool requested=fuse_env_ull("NTT_POINT_MERSENNE",0)!=0;
    const int bits=(int)mpz_sizeinbase(N,2);
    const bool exact_shape=bits>=2 && mpz_popcount(N)==(mp_bitcnt_t)bits && (size_t)((bits+63)/64)==nw;
    const int enabled_bits=requested && exact_shape ? bits : 0;
    CK(ecm_stage2_copy_constants(&enabled_bits,sizeof(enabled_bits),
                          offsetof(S2GDeviceConstants, point_mersenne_bits)));
    stage2_log::print(stage2_log::debug, "point_mersenne_mode: requested=%d enabled=%d bits=%d nw=%llu reduction=fold_rotate\n",
                (int)requested,enabled_bits!=0,bits,(unsigned long long)nw);
    stage2_log::print(stage2_log::debug, "point_arithmetic: xadd6=%d xadd_mont_muls=%d coordinate_scale=legacy_exact\n",
                (int)g_xadd6,g_xadd6 ? 6 : 8);
    const int cases = 2048;
    std::vector<unsigned long long> ha((size_t)cases * nw, 0ull), hb((size_t)cases * nw, 0ull);
    uint64_t s = 0x2468ace13579bdfull;
    mpz_t x;
    mpz_init(x);
    for (int side = 0; side < 2; ++side) {
        std::vector<unsigned long long> &dst = (side == 0) ? ha : hb;
        for (int i = 0; i < cases; ++i) {
            std::vector<unsigned long long> w(nw, 0ull);
            for (size_t j = 0; j < nw; ++j) {
                s = s * 6364136223846793005ull + 1442695040888963407ull;
                w[j] = s;
            }
            words_to_mpz(x, w.data(), nw);
            mpz_mod(x, x, N);
            mpz_to_words(w, nw, x);
            std::copy(w.begin(), w.end(), dst.begin() + (long)((size_t)i * nw));
        }
    }
    unsigned long long *da = nullptr, *db = nullptr, *dn = nullptr, *dout = nullptr;
    CK(cudaMalloc(&da, ha.size() * 8));
    CK(cudaMalloc(&db, hb.size() * 8));
    CK(cudaMalloc(&dn, nw * 8));
    CK(cudaMalloc(&dout, ha.size() * 8));
    CK(cudaMemcpy(da, ha.data(), ha.size() * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(db, hb.data(), hb.size() * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dn, hn.data(), nw * 8, cudaMemcpyHostToDevice));
    S2G_DISPATCH((int)nw, s2g_launch_mont_test, (int)nw, cases, da, db, dn, ninv, dout);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    std::vector<unsigned long long> ho(ha.size(), 0ull);
    CK(cudaMemcpy(ho.data(), dout, ho.size() * 8, cudaMemcpyDeviceToHost));
    cudaFree(da); cudaFree(db); cudaFree(dn); cudaFree(dout);

    unsigned long long bad = 0, first = 0;
    mpz_t am, bm, acc, rinv, want;
    mpz_inits(am, bm, acc, rinv, want, nullptr);
    mpz_invert(rinv, R, N);                                /* R^-1 mod N */
    for (int i = 0; i < cases; ++i) {
        words_to_mpz(am, &ha[(size_t)i * nw], nw);
        words_to_mpz(bm, &hb[(size_t)i * nw], nw);
        mpz_mul(acc, am, bm);
        mpz_mul(acc, acc, rinv);
        mpz_mod(acc, acc, N);
        words_to_mpz(want, &ho[(size_t)i * nw], nw);
        if (mpz_cmp(acc, want) != 0) { if (!bad) first = (unsigned long long)i; ++bad; }
    }
    mpz_clears(am, bm, acc, rinv, want, x, nullptr);
    stage2_log::print(stage2_log::debug, "mont_selftest: cases=%d mismatches=%llu first_bad=%llu "
                "(device Mont(a,b) vs GMP a*b*R^-1 mod N, nw=%llu)\n",
                cases, bad, first, (unsigned long long)nw);
    return bad ? 1 : 0;
}

int xadd_selftest(
    const std::vector<unsigned long long> &hn,
    size_t nw,
    unsigned long long ninv,
    const mpz_t N,
    const mpz_t R)
{
    stage2_trace::Range trace("check.xADD");
    const char *flag=std::getenv("NTT_XADD6_TEST");
    if(!flag || !*flag || !std::atoi(flag)) return 0;
    const int cases=128; // 64 ordinary-domain inputs + their 64 Montgomery images
    std::vector<unsigned long long> input((size_t)cases*6*nw),got((size_t)cases*5*3*nw);
    mpz_t x,rinv,a[6],tx,tz,v,half,observed;
    mpz_inits(x,rinv,tx,tz,v,half,observed,nullptr);
    for(auto &item:a) mpz_init(item);
    mpz_invert(rinv,R,N);
    uint64_t seed=0x85c529913ba8c742ull;
    for(int c=0;c<cases/2;++c) for(int j=0;j<6;++j) {
        std::vector<unsigned long long> w(nw);
        for(auto &word:w) {seed=seed*6364136223846793005ull+1442695040888963407ull;word=seed;}
        words_to_mpz(x,w.data(),nw);mpz_mod(x,x,N);
        if(c<8) {
            switch((c+j)%8) {
                case 0:mpz_set_ui(x,0);break;
                case 1:mpz_set_ui(x,1);break;
                case 2:mpz_sub_ui(x,N,1);break;
                case 3:mpz_sub_ui(x,N,2);break;
                case 4:mpz_fdiv_q_2exp(x,N,1);break;
                case 5:mpz_fdiv_q_2exp(x,N,1);mpz_add_ui(x,x,1);break;
                case 6:mpz_set_ui(x,3);mpz_mod(x,x,N);break;
                default:break;
            }
        }
        mpz_to_words(w,nw,x);
        std::copy(w.begin(),w.end(),input.begin()+((size_t)c*6+j)*nw);
        mpz_mul(x,x,R);mpz_mod(x,x,N);mpz_to_words(w,nw,x);
        std::copy(w.begin(),w.end(),input.begin()+((size_t)(c+cases/2)*6+j)*nw);
    }
    unsigned long long *di=nullptr,*dn=nullptr,*dout=nullptr;
    CK(cudaMalloc(&di,input.size()*8));CK(cudaMalloc(&dn,nw*8));CK(cudaMalloc(&dout,got.size()*8));
    CK(cudaMemcpy(di,input.data(),input.size()*8,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dn,hn.data(),nw*8,cudaMemcpyHostToDevice));
    unsigned long long bad=0,first=0;
    auto mont=[&](mpz_t dst,const mpz_t u,const mpz_t w) {
        mpz_mul(dst,u,w);mpz_mul(dst,dst,rinv);mpz_mod(dst,dst,N);
    };
    for(int candidate=0;candidate<2;++candidate) {
        S2G_DISPATCH((int)nw,s2g_launch_xadd_test,(int)nw,cases,di,dn,ninv,dout,candidate!=0);
        CK(cudaGetLastError());CK(cudaDeviceSynchronize());
        const char *fault=std::getenv("NTT_XADD6_TEST_BAD");
        if(candidate && fault && std::atoi(fault)) CK(cudaMemset(dout,0xff,8));
        CK(cudaMemcpy(got.data(),dout,got.size()*8,cudaMemcpyDeviceToHost));
        for(int c=0;c<cases;++c) {
            for(int j=0;j<6;++j) words_to_mpz(a[j],input.data()+((size_t)c*6+j)*nw,nw);
            mont(tx,a[0],a[2]);mont(v,a[1],a[3]);mpz_sub(tx,tx,v);mpz_mod(tx,tx,N);
            mont(tx,tx,tx);mont(tx,a[5],tx);
            mont(tz,a[0],a[3]);mont(v,a[1],a[2]);mpz_sub(tz,tz,v);mpz_mod(tz,tz,N);
            mont(tz,tz,tz);mont(tz,a[4],tz);
            mpz_set(half,a[0]);if(mpz_odd_p(half)) mpz_add(half,half,N);mpz_fdiv_q_2exp(half,half,1);
            for(int alias=0;alias<5;++alias) for(int coord=0;coord<3;++coord) {
                words_to_mpz(observed,got.data()+(((size_t)c*5+alias)*3+coord)*nw,nw);
                if(mpz_cmp(observed,coord==0 ? tx : coord==1 ? tz : half)!=0) {
                    if(!bad) first=((candidate*cases+c)*5+alias)*3+coord;
                    ++bad;
                }
            }
        }
    }
    CK(cudaFree(di));CK(cudaFree(dn));CK(cudaFree(dout));
    for(auto &item:a) mpz_clear(item);
    mpz_clears(x,rinv,tx,tz,v,half,observed,nullptr);
    stage2_log::print(stage2_log::debug, "xadd6_selftest: cases=%d aliases=5 domains=2 modes=2 coordinates=%d halves=%d bad=%llu first_bad=%llu nw=%llu\n",
                cases*5*2,cases*5*2*2,cases*5*2,bad,first,(unsigned long long)nw);
    return bad ? 1 : 0;
}
