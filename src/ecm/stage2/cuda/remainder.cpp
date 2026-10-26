#include "host_wait.h"
// Newton inverse, folds, scaled root and resident frontier/descent.
#include "point_launch.h"
#include "remainder.h"
#include "options.h"
#include "diagnostics.h"
#include "scale_plain.cuh"

std::vector<unsigned long long> cp_zero(size_t W)
{
    return std::vector<unsigned long long>(W, 0ull);
}

void cp_resize(CPoly &a,size_t n,size_t W)
{
    a.resize(n, cp_zero(W));
    for (std::vector<unsigned long long> &c : a)
        if (c.size() != W) c.assign(W, 0ull);
}

CPoly cp_from_flat(const std::vector<unsigned long long> &f,size_t deg,size_t W)
{
    CPoly p;
    cp_resize(p, deg + 1, W);
    for (size_t k = 0; k <= deg; ++k)
        std::copy(f.begin() + (long)(k * W), f.begin() + (long)((k + 1) * W), p[k].begin());
    return p;
}

std::vector<unsigned long long> cp_to_flat(const CPoly &p,size_t W)
{
    std::vector<unsigned long long> f(p.size() * W, 0ull);
    for (size_t k = 0; k < p.size(); ++k)
        std::copy(p[k].begin(), p[k].end(), f.begin() + (long)(k * W));
    return f;
}

bool cp_coeff_zero(const std::vector<unsigned long long> &c)
{
    for (unsigned long long w : c)
        if (w) return false;
    return true;
}

void cp_trim(CPoly &a)
{
    while (a.size() > 1 && cp_coeff_zero(a.back())) a.pop_back();
}

bool cp_is_one(const CPoly &a,size_t W)
{
    return a.size() == 1 && poly_is_one(a[0].data(), W);
}

void cp_coeff_sub(
    std::vector<unsigned long long> &dst,
    const std::vector<unsigned long long> &x,
    const std::vector<unsigned long long> &y,
    const mpz_t N,
    size_t W)
{
    mpz_t a, b;
    mpz_inits(a, b, nullptr);
    if (x.size() == W) words_to_mpz(a, x.data(), W); else mpz_set_ui(a, 0);
    if (y.size() == W) words_to_mpz(b, y.data(), W); else mpz_set_ui(b, 0);
    mpz_sub(a, a, b);
    mpz_mod(a, a, N);
    mpz_to_words(dst, W, a);
    mpz_clears(a, b, nullptr);
}

CPoly cp_addsub(const CPoly &a,const CPoly &b,PolyLayer &L,bool sub)
{
    const size_t W = L.W;
    CPoly r;
    cp_resize(r, std::max(a.size(), b.size()), W);
    mpz_t x, y;
    mpz_inits(x, y, nullptr);
    std::vector<unsigned long long> tmp(W, 0ull);
    for (size_t i = 0; i < r.size(); ++i) {
        if (i < a.size()) words_to_mpz(x, a[i].data(), W); else mpz_set_ui(x, 0);
        if (i < b.size()) words_to_mpz(y, b[i].data(), W); else mpz_set_ui(y, 0);
        if (sub) mpz_sub(x, x, y); else mpz_add(x, x, y);
        mpz_mod(x, x, L.N);
        mpz_to_words(tmp, W, x);
        r[i] = tmp;
    }
    mpz_clears(x, y, nullptr);
    cp_trim(r);
    return r;
}

CPoly cp_mul(const CPoly &a,const CPoly &b,PolyLayer &L,size_t keep)
{
    const size_t W = L.W;
    if(a.empty() || b.empty()) return {};
    const size_t nc=a.size()+b.size()-1;
    const size_t wanted=keep==(size_t)-1 ? nc : std::min(keep,nc);
    if (L.s4) {
        /* the batched entry point takes nbatch slices of P*W words, ZERO-PADDED: an operand
           shorter than P must not be handed over as a tight array, or the packer reads past
           its end (that is a real bug this slice shipped for one build: cp_divmod multiplies
           unequal lengths constantly, and the garbage above the operand landed in the top
           coefficients -- the multiply's own GMP check cannot see it, because it verifies the
           digits it produced, not what was fed in). */
        const size_t P = (a.size() > b.size()) ? a.size() : b.size();
        std::vector<unsigned long long> fa(P * W, 0ull), fb(P * W, 0ull);
        for (size_t i = 0; i < a.size(); ++i)
            std::copy(a[i].begin(), a[i].end(), fa.begin() + (long)(i * W));
        for (size_t i = 0; i < b.size(); ++i)
            std::copy(b[i].begin(), b[i].end(), fb.begin() + (long)(i * W));
        std::vector<unsigned long long> fc;
        poly_mul_batch_modN(L, fa.data(), fb.data(), a.size(), b.size(), 1, fc,-1,nullptr,0,wanted);
        return wanted ? cp_from_flat(fc,wanted-1,W) : CPoly{};
    }
    const std::vector<unsigned long long> fa = cp_to_flat(a, W);
    const std::vector<unsigned long long> fb = cp_to_flat(b, W);
    const std::vector<unsigned long long> fc = poly_mul_modN(L, fa, a.size() - 1, fb, b.size() - 1);
    return wanted ? cp_from_flat(fc,wanted-1,W) : CPoly{};
}

CPoly cp_inv_series(const CPoly &a,size_t k,PolyLayer &L)
{
    const size_t W = L.W;
    CPoly g;
    cp_resize(g, 1, W);
    {
        mpz_t av, inv;
        mpz_inits(av, inv, nullptr);
        words_to_mpz(av, a[0].data(), W);
        if (mpz_invert(inv, av, L.N) == 0) {
            std::fprintf(stderr, "%s: FATAL: divisor leading coefficient not invertible mod N "
                                 "(the divisor was not monic)\n", NTT_PROBE_NAME);
            std::exit(4);
        }
        mpz_mod(inv, inv, L.N);
        mpz_to_words(g[0], W, inv);
        mpz_clears(av, inv, nullptr);
    }
    while (g.size() < k) {
        const size_t nxt = std::min(2 * g.size(), k);
        const CPoly at(a.begin(), a.begin() + (long)std::min(a.size(), nxt));
        CPoly ag = cp_mul(at, g, L, nxt);
        cp_resize(ag, nxt, W);                        /* mod X^nxt */
        CPoly h;
        cp_resize(h, nxt, W);
        {
            mpz_t two;
            mpz_init(two);
            mpz_set_ui(two, 2);
            mpz_mod(two, two, L.N);
            mpz_to_words(h[0], W, two);
            mpz_clear(two);
        }
        h = cp_addsub(h, ag, L, /*sub=*/true);        /* h = 2 - ag */
        cp_resize(h, nxt, W);
        CPoly gn = cp_mul(g, h, L, nxt);
        cp_resize(gn, nxt, W);                        /* mod X^nxt */
        g = gn;
    }
    return g;
}

void cp_divmod(CPoly &q,CPoly &r,const CPoly &a,const CPoly &b,PolyLayer &L)
{
    const size_t W = L.W;
    const long da = (long)a.size() - 1, db = (long)b.size() - 1;
    if (da < db) {
        cp_resize(q, 1, W);
        r = a;
        cp_trim(r);
        return;
    }
    const size_t k = (size_t)(da - db + 1);
    CPoly ra, rb;
    cp_resize(ra, k, W);
    cp_resize(rb, (size_t)db + 1, W);
    for (size_t i = 0; i < k; ++i) ra[i] = a[(size_t)(da - (long)i)];
    for (size_t i = 0; i <= (size_t)db; ++i) rb[i] = b[(size_t)(db - (long)i)];
    const CPoly rbi = cp_inv_series(rb, k, L);
    CPoly qrev = cp_mul(ra, rbi, L, k);
    cp_resize(qrev, k, W);
    cp_resize(q, k, W);
    for (size_t i = 0; i < k; ++i) q[i] = qrev[k - 1 - i];
    cp_trim(q);

    const CPoly qb = cp_mul(q, b, L, (size_t)db);
    if (db == 0) {
        cp_resize(r, 1, W);
    } else {
        cp_resize(r, (size_t)db, W);
        for (size_t i = 0; i < (size_t)db; ++i) {
            const std::vector<unsigned long long> &x = a[i];
            const std::vector<unsigned long long> &y =
                (i < qb.size()) ? qb[i] : cp_zero(W);
            cp_coeff_sub(r[i], x, y, L.N, W);
        }
    }
    cp_trim(r);
}

CPoly cp_mod(const CPoly &a,const CPoly &b,PolyLayer &L)
{
    if (cp_is_one(b, L.W)) { CPoly z; cp_resize(z, 1, L.W); return z; }
    if (a.size() < b.size()) return a;
    CPoly q, r;
    cp_divmod(q, r, a, b, L);
    return r;
}

void leaf_mod_linear_batch(
    const std::vector<unsigned long long> &A,
    size_t da,
    const std::vector<unsigned long long> &B,
    size_t nbatch,
    size_t W,
    const mpz_t N,
    std::vector<unsigned long long> &out)
{
    out.assign(nbatch * W, 0ull);
    mpz_t h, x, t;
    mpz_inits(h, x, t, nullptr);
    std::vector<unsigned long long> w(W, 0ull);
    for (size_t s = 0; s < nbatch; ++s) {
        const unsigned long long *b = &B[s * 2 * W];
        if (b[W] != 1) {
            std::fprintf(stderr, "%s: FATAL: leaf_mod_linear_batch: divisor %llu is not monic "
                                 "(lead=%llu)\n", NTT_PROBE_NAME, (unsigned long long)s, b[W]);
            std::exit(3);
        }
        for (size_t q = 2; q < W; ++q)
            if (b[q] != 0) {
                std::fprintf(stderr, "%s: FATAL: leaf_mod_linear_batch: divisor %llu has degree "
                                     "> 1 (word %llu = %llu)\n", NTT_PROBE_NAME,
                             (unsigned long long)s, (unsigned long long)q, b[q]);
                std::exit(3);
            }
        words_to_mpz(x, b, W);                       /* b = [ -root, 1 ] */
        mpz_neg(x, x);                               /* the evaluation point IS the root */
        mpz_mod(x, x, N);
        /* Horner over REVERSED coefficient order, low coefficient inside: h = a_da, then
           h = a_i + h*x for i = da-1 .. 0, so the constant coefficient is applied last.  (The
           other way round is NOT equivalent: for da = 1 it gives a1 + a0*x instead of
           a0 + a1*x, which is what the first version of this function did -- the coefficients
           are stored low-first, so a direct loop walks them in the wrong order.) */
        words_to_mpz(h, &A[s * (da + 1) * W + da * W], W);
        for (size_t i = da; i-- > 0;) {
            mpz_mul(t, h, x);
            words_to_mpz(h, &A[s * (da + 1) * W + i * W], W);
            mpz_add(h, h, t);
            mpz_mod(h, h, N);
        }
        mpz_to_words(w, W, h);
        std::copy(w.begin(), w.end(), out.begin() + (long)(s * W));
    }
    mpz_clears(h, x, t, nullptr);
}

void fold_flat_step(
    PolyLayer &L,
    const std::vector<unsigned long long> &G,
    std::vector<unsigned long long> &H,
    const std::vector<unsigned long long> &F,
    const std::vector<unsigned long long> &finv,
    FoldFlatStats &st)
{
    const size_t W=L.W, P=F.size()/W-1;
    if(!L.s4 || !W || !P || F.size()%W || G.size()%W || H.size()%W || finv.size()%W) {
        std::fprintf(stderr,"%s: FATAL: flat fold shape mismatch\n",NTT_PROBE_NAME);std::exit(3);
    }
    ++st.folds;
    if(G.empty() || H.empty()) {H.clear();return;}
    std::vector<unsigned long long> T,qrev,qb;
    auto multiply=[&](const std::vector<unsigned long long> &a,
                      const std::vector<unsigned long long> &b,
                      std::vector<unsigned long long> &out,size_t count=(size_t)-1) {
        const double t=now_s();flat_mul_batch(L,a,a.size()/W,b,b.size()/W,1,out,BC_FOLD,0,count);
        st.t_multiply+=now_s()-t;++st.muls;
    };
    multiply(G,H,T);
    const size_t degT=T.size()/W-1;
    if(degT<P) {H=std::move(T);return;}
    const size_t k=degT-P+1;
    if(finv.size()<k*W) {std::fprintf(stderr,"%s: FATAL: flat fold inverse too short\n",NTT_PROBE_NAME);std::exit(3);}
    {
        const double tp=now_s();
        std::vector<unsigned long long> ra(k*W),rbi(finv.begin(),finv.begin()+k*W);
        for(size_t i=0;i<k;++i)std::copy_n(T.data()+(degT-i)*W,W,ra.data()+i*W);
        st.t_prepare+=now_s()-tp;
        multiply(ra,rbi,qrev,k);
        st.peak_bytes=std::max(st.peak_bytes,8ull*(H.capacity()+T.capacity()+ra.capacity()+rbi.capacity()+qrev.capacity()));
    } // reverse inputs no longer live when q*F is formed
    const double tq=now_s();
    for(size_t i=0;i<k/2;++i)for(size_t w=0;w<W;++w)std::swap(qrev[i*W+w],qrev[(k-1-i)*W+w]);
    st.t_prepare+=now_s()-tq;
    multiply(qrev,F,qb,P);
    H.resize(P*W);
    st.peak_bytes=std::max(st.peak_bytes,8ull*(H.capacity()+T.capacity()+qrev.capacity()+qb.capacity()));
    const double ts=now_s();
    // Reuse GMP limbs for the whole remainder, preserving general composite-N mod semantics.
    mpz_t a,b;mpz_inits(a,b,nullptr);
    std::vector<unsigned long long> word(W);
    for(size_t i=0;i<P;++i) {
        words_to_mpz(a,T.data()+i*W,W);words_to_mpz(b,qb.data()+i*W,W);
        mpz_sub(a,a,b);mpz_mod(a,a,L.N);mpz_to_words(word,W,a);
        std::copy_n(word.data(),W,H.data()+i*W);
    }
    mpz_clears(a,b,nullptr);st.sub_coeffs+=P;
    const char *fault=std::getenv("NTT_FOLD_FLAT_TEST_BAD");
    if(fault && std::atoi(fault) && !H.empty()) H[0]^=1;
    while(H.size()>W && std::all_of(H.end()-W,H.end(),[](unsigned long long x){return x==0;}))H.resize(H.size()-W);
    st.t_subtract+=now_s()-ts;
}

bool FoldDeviceState::init(
    PolyLayer &L,
    const std::vector<unsigned long long> &F,
    const std::vector<unsigned long long> &inverse,
    FoldDeviceStats &st,
    FoldFlatStats &fs)
{
    layer=&L;stats=&st;flat=&fs;st.requested=fold_device_flag("NTT_FOLD_DEVICE");
    if(!st.requested)return false;
    const double begin=now_s();W=L.W;P=F.size()/W-1;host_F=&F;
    check=fold_device_flag("NTT_FOLD_DEVICE_CHECK");
    const char *hostpack=std::getenv("NTT_S4_HOSTPACK");
    if(!g_fold_flat || !L.s4 || !L.arena || !g_s4_pack_direct || g_s4_final_readback ||
       g_s4_carry_trace || (hostpack && std::atoi(hostpack))) {st.fallback="backend";return false;}
    if(!P || F.size()!=(P+1)*W || inverse.size()!=(P+1)*W) {st.fallback="shape";return false;}
    if(check && P>64){std::fprintf(stderr,"FATAL: device fold GMP check limited to P<=64\n");std::exit(3);}
    ecm_stage2::FoldOwnerLayout layout;
    if(!ecm_stage2::fold_owner_layout(P,W,kFoldOwnerReuse,layout)) {
        std::fprintf(stderr,"FATAL: device fold owner layout overflow\n");std::exit(3);
    }
    const unsigned long long bytes=layout.bytes;st.layout_bytes=bytes;
    unsigned long long max_mb=640;
    if(const char *e=std::getenv("NTT_FOLD_DEVICE_MAX_MB"))max_mb=std::strtoull(e,nullptr,10);
    if(max_mb>(~0ull>>20) || bytes>(max_mb<<20)) {st.fallback="budget";return false;}
    unsigned long long qN=0;
    if(!ntt_shape_query(P+1,(int)L.S,&qN,nullptr,nullptr,nullptr,nullptr,nullptr)) {st.fallback="ntt_shape";return false;}
    size_t available=0,total=0;CK(cudaMemGetInfo(&available,&total));
    // Finv has normally already built the largest workspace. Reserve its
    // possible remaining growth and 1 GiB for coordinates/frontier/context.
    const unsigned buffers=L.arena->big_buffer_count();
    const size_t target=(size_t)buffers*qN*8,current=L.arena->workspace.words*8;
    const size_t growth=target>current?target-current:0;
    const size_t future_reserve=1ull<<30;
    if(fold_device_flag("NTT_OWNER_TRIM_FUSE"))
        ntt_trim_fuse_headroom(L.arena,qN,bytes+growth+future_reserve,"inverse_to_fold",available);
    const bool fits=bytes<=available && available-bytes>=growth+future_reserve;
    stage2_log::print(stage2_log::debug,
        "fold_device_headroom: available_bytes=%llu owner_bytes=%llu physical_buffers=%u target_big_bytes=%llu current_big_bytes=%llu growth_bytes=%llu future_reserve_bytes=%llu fits=%d\n",
        (unsigned long long)available,bytes,buffers,(unsigned long long)target,
        (unsigned long long)current,(unsigned long long)growth,(unsigned long long)future_reserve,(int)fits);
    if(!fits) {st.fallback="headroom";return false;}
    if(fold_device_flag("NTT_FOLD_DEVICE_ALLOC_FAIL")){st.fallback="allocation_fixture";return false;}
    memory.words[0]=layout.source_words;memory.words[1]=layout.result_words;
    for(int i=0;i<2;++i) {
        const auto error=cudaMalloc(&memory.data[i],memory.words[i]*8);
        if(error==cudaErrorMemoryAllocation){cudaGetLastError();memory.release();st.fallback="allocation";return false;}
        CK(error);
    }
    CK(cudaMalloc(&map,24));CK(cudaMalloc(&length,8));CK(cudaMalloc(&modulus,W*8));
    CK(cudaMalloc(&digest,16));CK(cudaMemsetAsync(digest,0,16));
    f=layout.f;inv=layout.inv;h=layout.h;g=layout.g;reverse=layout.reverse;
    t=layout.t;q=layout.q;qb=layout.qb;
    upload(f,F);upload(inv,inverse);
    std::vector<unsigned long long> hn(W);mpz_to_words(hn,W,L.N);
    CK(stage2_wait::copy(modulus,hn.data(),W*8,cudaMemcpyHostToDevice));st.h2d_bytes+=W*8;
    st.saved_bytes=ecm_stage2::owner_bytes(P,W,0)-bytes;
    st.peak_bytes=bytes;st.enabled=active=true;st.t_setup=now_s()-begin;
    return true;
}

void FoldDeviceState::upload(size_t offset,const std::vector<unsigned long long> &v)
{
    if(v.empty())return;
    const double begin=now_s();CK(stage2_wait::copy(memory.data[0]+offset,v.data(),v.size()*8,cudaMemcpyHostToDevice));
    stats->h2d_bytes+=v.size()*8;stats->t_upload+=now_s()-begin;
}

void FoldDeviceState::digest_root(size_t offset,size_t words)
{
    if(words) {
        const unsigned blocks=(unsigned)std::min((size_t)1024,(words+255)/256);
        ecm_launch_groot_input_digest_kernel(EcmCudaLaunch{blocks,256}, memory.data[0]+offset,words,stats->root_digest_words,digest);
        CK(cudaGetLastError());stats->root_digest_words+=words;
    }
}

void FoldDeviceState::read_digest()
{
    CK(stage2_wait::copy(stats->root_digest,digest,16,cudaMemcpyDeviceToHost));stats->d2h_bytes+=16;
}

void FoldDeviceState::accept_device_root(const unsigned long long *src,size_t count,bool seed_root)
{
    if(!active || !src || count>P+1 || !count ||
       (src!=layer->s4->d_rawA && src!=layer->s4->d_rawB) || count*W>layer->s4->raw_capacity(src)) {
        std::fprintf(stderr,"FATAL: root-to-fold source is not a live bounded raw lease\n");std::exit(3);
    }
    const bool verify=fold_device_flag("NTT_GROOT_TO_FOLD_CHECK");
    if(verify && P>64){std::fprintf(stderr,"FATAL: root-to-fold check limited to P<=64\n");std::exit(3);}
    const double begin=now_s();const size_t dest=seed_root?h:g,words=count*W;
    std::vector<unsigned long long> expected;
    if(verify) {
        expected.resize(words);CK(stage2_wait::copy(expected.data(),src,words*8,cudaMemcpyDeviceToHost));
        stats->d2h_bytes+=words*8;
    }
    CK(cudaMemcpyAsync(memory.data[0]+dest,src,words*8,cudaMemcpyDeviceToDevice));
    if(fold_device_flag("NTT_GROOT_TO_FOLD_TEST_BAD") && !stats->root_device_trees)
        CK(cudaMemsetAsync(memory.data[0]+dest,0xff,8));
    if(verify) {
        auto actual=read(dest,count);stats->root_checked_words+=2*words;
        if(actual!=expected){std::fprintf(stderr,"FATAL: root-to-fold word mismatch\n");std::exit(3);}
    }
    if(seed_root)hcount=count;else gcount=count;
    digest_root(dest,words);++stats->root_device_trees;stats->root_device_words+=words;
    stats->root_h2d_avoided+=words*8;stats->root_d2h_avoided+=words*8;
    stats->root_t_handoff+=now_s()-begin;
}

void FoldDeviceState::seed(const std::vector<unsigned long long> &H)
{
    if(H.size()>(P+1)*W || H.size()%W){std::fprintf(stderr,"FATAL: device fold seed shape\n");std::exit(3);}
    upload(h,H);hcount=H.size()/W;digest_root(h,H.size());
}

std::vector<unsigned long long> FoldDeviceState::read(size_t offset,size_t count)
{
    std::vector<unsigned long long> v(count*W);const double begin=now_s();
    if(!v.empty())CK(stage2_wait::copy(v.data(),memory.data[0]+offset,v.size()*8,cudaMemcpyDeviceToHost));
    stats->d2h_bytes+=v.size()*8;stats->t_readback+=now_s()-begin;return v;
}

void FoldDeviceState::mul(size_t a,size_t ma,size_t b,size_t mb,size_t dest,size_t count)
{
    unsigned long long metadata[3]={(unsigned long long)a,(unsigned long long)b,(unsigned long long)dest};
    CK(stage2_wait::copy(map,metadata,24,cudaMemcpyHostToDevice));stats->h2d_bytes+=24;
    S4DeviceBatch batch{memory.data[0],map,memory.data[1],metadata,memory.words[0],memory.words[1],1,&memory};
    std::vector<unsigned long long> unused;
    const double begin=now_s();poly_mul_batch_modN(*layer,nullptr,nullptr,ma,mb,1,unused,BC_FOLD,nullptr,0,count,&batch);
    flat->t_multiply+=now_s()-begin;++flat->muls;++stats->muls;
    stats->avoided_h2d_bytes+=16ull*std::max(ma,mb)*W;
    stats->avoided_d2h_bytes+=8ull*(g_s4_output_window?count:2*std::max(ma,mb)-1)*W;
}

void FoldDeviceState::reverse_copy(size_t src,size_t top,size_t count)
{
    const double begin=now_s();ecm_launch_fold_reverse_kernel(EcmCudaLaunch{(unsigned)((count*W+255)/256),256},
        memory.data[1]+src,top,count,(int)W,memory.data[0]+reverse);CK(cudaGetLastError());
    stats->t_reverse+=now_s()-begin;
}

void FoldDeviceState::step(const std::vector<unsigned long long> &G)
{
    if(G.size()>(P+1)*W || G.size()%W){std::fprintf(stderr,"FATAL: device fold G shape\n");std::exit(3);}
    upload(g,G);digest_root(g,G.size());step_loaded(G.size()/W,check?&G:nullptr);
}

void FoldDeviceState::step_loaded(size_t ng,const std::vector<unsigned long long> *host_G)
{
    ++stats->folds;++flat->folds;
    std::vector<unsigned long long> expected;
    if(check) {
        auto G=host_G?*host_G:read(g,ng);expected=fold_gmp_reference(*layer,G,read(h,hcount),*host_F);
    }
    if(!ng || !hcount){hcount=0;return;}
    const size_t nT=ng+hcount-1;
    // Digest and gather read G before reverse overwrites its slot, on the
    // same default stream. Diagnostic GMP inputs are already independent.
    mul(g,ng,h,hcount,t,nT);
    if(nT<=P) {
        CK(cudaMemcpyAsync(memory.data[0]+h,memory.data[1]+t,nT*W*8,cudaMemcpyDeviceToDevice));hcount=nT;
    } else {
        const size_t k=nT-P;
        reverse_copy(t,nT-1,k);mul(reverse,k,inv,k,q,k);
        // Reverse reads all k<=P+1 q coefficients before qb overwrites q.
        // Oracle slots hold independent digit/result snapshots.
        reverse_copy(q,k-1,k);mul(reverse,k,f,P+1,qb,P);
        const double begin=now_s();CK(cudaMemsetAsync(length,0,8));
        ecm_launch_fold_subtract_kernel(EcmCudaLaunch{(unsigned)((P+127)/128),128}, memory.data[1]+t,memory.data[1]+qb,
            modulus,(int)W,P,memory.data[0]+h,length);CK(cudaGetLastError());
        unsigned long long n=0;CK(stage2_wait::copy(&n,length,8,cudaMemcpyDeviceToHost));
        hcount=std::max((size_t)1,(size_t)n);stats->d2h_bytes+=8;stats->sub_coeffs+=P;flat->sub_coeffs+=P;
        stats->t_subtract+=now_s()-begin;flat->t_subtract+=now_s()-begin;
    }
    if(fold_device_flag("NTT_FOLD_DEVICE_TEST_BAD"))CK(cudaMemset(memory.data[0]+h,0xff,8));
    if(check) {
        auto actual=read(h,hcount);stats->checked_words+=actual.size();
        if(actual!=expected){std::fprintf(stderr,"FATAL: device fold GMP mismatch P=%llu\n",(unsigned long long)P);std::exit(3);}
    }
}

void FoldDeviceState::scale(mpz_srcptr factor,PlainScaleStats &st,bool verify,bool poison)
{
    if(!active || !hcount || q+W>memory.words[1] || h+hcount*W>memory.words[0]) {
        std::fprintf(stderr,"FATAL: Gamma device owner lease invalid\n");std::exit(3);
    }
    s2g_plain_scale(W,hcount,memory.data[0]+h,memory.data[1]+q,modulus,layer->N,factor,st,verify,poison);
    stats->h2d_bytes+=st.h2d_bytes;stats->d2h_bytes+=st.check_d2h_bytes;
}

void FoldDeviceState::scaled_root(
    std::vector<unsigned long long> &root,
    ScaledRootDeviceStats &st,
    bool verify,
    bool poison,
    bool keep_resident)
{
    if(!active || !hcount || hcount>P || reverse+P*W>memory.words[0] ||
       inv+P*W>memory.words[0] || t+P*W>memory.words[1]) {
        std::fprintf(stderr,"FATAL: scaled root owner lease invalid\n");std::exit(3);
    }
    const double begin=now_s();
    std::vector<unsigned long long> expected;
    if(verify) {
        // Independent host packing of the entire H and cached inverse;
        // arithmetic remains the validated S4 path. Small gates also use
        // the independent GMP triangular oracle in descent_scaled.
        auto hc=read(h,hcount),ic=read(inv,P);
        st.check_d2h_bytes+=(hcount+P)*W*8ull;
        std::vector<unsigned long long> a(P*W,0);
        for(size_t i=0;i<P;++i)if(P-1-i<hcount)
            std::copy_n(hc.data()+(P-1-i)*W,W,a.data()+i*W);
        flat_mul_batch(*layer,a,P,ic,P,1,expected,BC_DESCENT,0,P);
    }
    // G/reverse is dead after the final fold; H and finv are disjoint.
    ecm_launch_scaled_root_reverse_kernel(EcmCudaLaunch{(unsigned)((P*W+255)/256),256},
        memory.data[0]+h,hcount,P,(int)W,memory.data[0]+reverse);
    CK(cudaGetLastError());
    unsigned long long metadata[3]={(unsigned long long)reverse,(unsigned long long)inv,(unsigned long long)t};
    CK(stage2_wait::copy(map,metadata,24,cudaMemcpyHostToDevice));stats->h2d_bytes+=24;st.h2d_bytes=24;
    S4DeviceBatch batch{memory.data[0],map,memory.data[1],metadata,memory.words[0],memory.words[1],1,&memory};
    std::vector<unsigned long long> unused;
    poly_mul_batch_modN(*layer,nullptr,nullptr,P,P,1,unused,BC_DESCENT,nullptr,0,P,&batch);
    if(poison) {
        unsigned long long v=0;CK(stage2_wait::copy(&v,memory.data[1]+t,8,cudaMemcpyDeviceToHost));
        v^=1;CK(stage2_wait::copy(memory.data[1]+t,&v,8,cudaMemcpyHostToDevice));
        stats->d2h_bytes+=8;stats->h2d_bytes+=8;st.check_d2h_bytes+=8;st.h2d_bytes+=8;
    }
    if(!keep_resident || verify) {
        root.resize(P*W);CK(stage2_wait::copy(root.data(),memory.data[1]+t,P*W*8,cudaMemcpyDeviceToHost));
        stats->d2h_bytes+=P*W*8;st.d2h_bytes=P*W*8;
    } else root.clear();
    if(verify) {
        st.checked_words=root.size();
        if(root!=expected){std::fprintf(stderr,"FATAL: scaled root device mismatch\n");std::exit(3);}
    }
    st.enabled=true;st.coefficients=P;st.avoided_h2d_bytes=16ull*P*W;
    st.avoided_h_readback_bytes=8ull*hcount*W;st.seconds=now_s()-begin;
}

void FoldDeviceState::finish(std::vector<unsigned long long> &H)
{H=read(h,hcount);read_digest();release();}

void descent_scaled(
    PolyLayer &L,
    const std::vector<std::vector<unsigned long long>> &Ft,
    const std::vector<size_t> &Fdeg,
    size_t Fpad,
    const CPoly &H,
    const CPoly *cached_finv,
    std::vector<std::vector<unsigned long long>> &values,
    ScaledStats &st,
    int cat,
    bool check,
    std::vector<unsigned long long> *prepared_root)
{
    const size_t W=L.W,P=Fdeg[1];
    if(!L.s4 || !P || Ft.size()<2*Fpad || Fdeg.size()<2*Fpad ||
       Ft[1].size()!=(P+1)*W || !poly_is_one(Ft[1].data()+P*W,W) || (check && P>512)) {
        std::fprintf(stderr,"%s: FATAL: invalid scaled descent shape (GMP check limited to 512 leaves)\n",NTT_PROBE_NAME);
        std::exit(3);
    }
    std::vector<unsigned long long> A,B,root;
    if(prepared_root) {
        if(prepared_root->size()!=P*W){std::fprintf(stderr,"FATAL: prepared scaled root length\n");std::exit(3);}
        root.swap(*prepared_root);++st.root_inverse_reused;++st.mul_calls;++st.mul_pairs;
        st.pack_peak_bytes=8ull*root.capacity();
    } else {
        CPoly local_inv, h=H;
        if(h.size()>P) {
            h=cp_mod(h,cp_from_flat(Ft[1],P,W),L); ++st.root_divisions;
        }
        const CPoly *inv=cached_finv;
        if(!inv || inv->size()<P) {
            CPoly rev;
            cp_resize(rev,P+1,W);
            for(size_t i=0;i<=P;++i)
                std::copy_n(Ft[1].data()+(P-i)*W,W,rev[i].begin());
            local_inv=cp_inv_series(rev,P,L);inv=&local_inv;
        } else ++st.root_inverse_reused;
        A.assign(P*W,0);B.assign(P*W,0);
        for(size_t i=0;i<P;++i) {
            if(P-1-i<h.size()) std::copy(h[P-1-i].begin(),h[P-1-i].end(),A.begin()+i*W);
            std::copy((*inv)[i].begin(),(*inv)[i].end(),B.begin()+i*W);
        }
        flat_mul_batch(L,A,P,B,P,1,root,cat,0,P);
        ++st.mul_calls;++st.mul_pairs;
        st.pack_peak_bytes=std::max(st.pack_peak_bytes,8ull*(A.capacity()+B.capacity()+root.capacity()));
        std::vector<unsigned long long>().swap(A);std::vector<unsigned long long>().swap(B);
    }
    std::vector<std::vector<unsigned long long>> cur(1);
    cur[0].swap(root);
    auto validate=[&](const std::vector<std::vector<unsigned long long>> &front,size_t base) {
        for(size_t i=0;i<front.size();++i) {
            const size_t node=base+i,d=Fdeg[node];
            if(front[i].size()!=d*W) {std::fprintf(stderr,"%s: FATAL: scaled state length\n",NTT_PROBE_NAME);std::exit(3);}
            ++st.states;st.words+=d*W;
            if(check) {
                auto want=scaled_state_gmp(H,Ft[node],d,L);
                if(front[i]!=want) {std::fprintf(stderr,"%s: FATAL: scaled GMP node mismatch node=%llu degree=%llu\n",NTT_PROBE_NAME,(unsigned long long)node,(unsigned long long)d);std::exit(3);}
                ++st.checked_states;st.checked_words+=d*W;
            }
        }
    };
    validate(cur,1);
    const double t0=now_s();
    for(size_t base=1;base<Fpad;base*=2) {
        const double tl0=now_s();
        const size_t nbase=base*2;
        std::vector<std::vector<unsigned long long>> next(cur.size()*2);
        std::map<std::pair<size_t,size_t>,std::vector<std::pair<size_t,size_t>>> groups;
        for(size_t j=0;j<cur.size();++j) for(size_t side=0;side<2;++side) {
            const size_t slot=2*j+side,ci=nbase+slot,si=ci^1;
            const size_t a=Fdeg[ci],b=Fdeg[si];
            if(a+b!=Fdeg[base+j]) {std::fprintf(stderr,"%s: FATAL: scaled degree conservation\n",NTT_PROBE_NAME);std::exit(3);}
            if(!a) {++st.zeros;continue;}
            if(!b) {next[slot]=cur[j];++st.copies;continue;}
            groups[{a,b}].push_back({j,slot});
        }
        for(const auto &g:groups) {
            const size_t a=g.first.first,b=g.first.second,ma=a+b,mb=b+1,nb=g.second.size();
            A.assign(nb*ma*W,0);B.assign(nb*mb*W,0);
            std::vector<unsigned long long> product;
            for(size_t s=0;s<nb;++s) {
                const size_t j=g.second[s].first,slot=g.second[s].second,si=(nbase+slot)^1;
                std::copy(cur[j].begin(),cur[j].end(),A.begin()+s*ma*W);
                if(Ft[si].size()!=mb*W) {std::fprintf(stderr,"%s: FATAL: scaled sibling length\n",NTT_PROBE_NAME);std::exit(3);}
                for(size_t q=0;q<mb;++q)
                    std::copy_n(Ft[si].data()+(b-q)*W,W,B.begin()+(s*mb+q)*W);
            }
            flat_mul_batch(L,A,ma,B,mb,nb,product,cat,b,a);
            ++st.mul_calls;st.mul_pairs+=nb;
            st.pack_peak_bytes=std::max(st.pack_peak_bytes,8ull*(A.capacity()+B.capacity()+product.capacity()));
            for(size_t s=0;s<nb;++s)
                next[g.second[s].second].assign(product.begin()+s*a*W,product.begin()+(s+1)*a*W);
        }
        unsigned long long bytes=0;
        for(const auto &v:cur) bytes+=8ull*v.capacity();
        for(const auto &v:next) bytes+=8ull*v.capacity();
        st.frontier_peak_bytes=std::max(st.frontier_peak_bytes,bytes);
        cur.swap(next);++st.levels;validate(cur,nbase);
        if(g_s4_batched_progress)
            stage2_log::print(stage2_log::debug, "scaled_progress: level=%llu nodes=%llu groups=%llu t_level=%.3f t_total=%.3f\n",st.levels,(unsigned long long)cur.size(),(unsigned long long)groups.size(),now_s()-tl0,now_s()-t0);
    }
    values.assign(P,std::vector<unsigned long long>(W,0));
    size_t leaf=0;
    for(size_t i=0;i<Fpad;++i) if(Fdeg[Fpad+i]) {
        if(Fdeg[Fpad+i]!=1 || leaf>=P) {std::fprintf(stderr,"%s: FATAL: scaled non-linear leaf\n",NTT_PROBE_NAME);std::exit(3);}
        values[leaf++]=std::move(cur[i]);
    }
    if(leaf!=P) {std::fprintf(stderr,"%s: FATAL: scaled leaf count\n",NTT_PROBE_NAME);std::exit(3);}
    st.leaves+=leaf;
}

std::vector<unsigned long long> cp_addsub_flat(
    const std::vector<unsigned long long> &A,
    const std::vector<unsigned long long> &B,
    size_t n,
    size_t W,
    const mpz_t N,
    bool sub)
{
    std::vector<unsigned long long> out(A.size(), 0ull);
    std::vector<unsigned long long> wa(W, 0ull), wb(W, 0ull);
    mpz_t x, y;
    mpz_inits(x, y, nullptr);
    for (size_t i = 0; i < n; ++i) {
        words_to_mpz(x, &A[i * W], W);
        words_to_mpz(y, &B[i * W], W);
        if (sub) mpz_sub(x, x, y); else mpz_add(x, x, y);
        mpz_mod(x, x, N);
        mpz_to_words(wa, W, x);
        std::copy(wa.begin(), wa.end(), out.begin() + (long)(i * W));
    }
    mpz_clears(x, y, nullptr);
    return out;
}

void cp_coeff_sub_p(
    unsigned long long *dst,
    const unsigned long long *x,
    const unsigned long long *y,
    const mpz_t N,
    size_t W)
{
    mpz_t a, b;
    mpz_inits(a, b, nullptr);
    words_to_mpz(a, x, W);
    words_to_mpz(b, y, W);
    mpz_sub(a, a, b);
    mpz_mod(a, a, N);
    std::vector<unsigned long long> tmp(W, 0ull);
    mpz_to_words(tmp, W, a);
    std::copy(tmp.begin(), tmp.end(), dst);
    mpz_clears(a, b, nullptr);
}

void inv_series_batch(
    PolyLayer &L,
    const std::vector<unsigned long long> &A,
    size_t m,
    size_t k,
    size_t nbatch,
    std::vector<unsigned long long> &g,
    int cat)
{
    const size_t W = L.W;
    /* g holds EXACTLY `len` coefficients per slice at every step, because flat_mul_batch reads
       it at the stride the caller declares and derives its result from the operand's element
       count.  The Newton loop below starts at len = 1 and REPLACES g with a freshly truncated
       buffer at the end of each round, so `nbatch*W` would be enough -- but only if the first
       multiply's result were not inflated, and the chain is easier to trust when the first
       allocation already holds the full k (the first round only writes coefficients 0..nxt-1,
       so nothing about the answer changes). */
    g.assign(nbatch * W, 0ull);
    /* 1/a[0] per slice.  cp_divmod always hands us a REVERSED MONIC divisor, so a[0] == 1 and
       this is the identity -- but it is CHECKED, and the general path is GMP. */
    bool all_one = true;
    for (size_t s = 0; s < nbatch && all_one; ++s) {
        const unsigned long long *c = &A[s * m * W];
        if (c[0] != 1) all_one = false;
        for (size_t j = 1; j < W; ++j) if (c[j] != 0) all_one = false;
    }
    if (all_one) {
        for (size_t s = 0; s < nbatch; ++s) g[s * W] = 1;
    } else {
        mpz_t av, inv;
        std::vector<unsigned long long> gtmp(W, 0ull);
        mpz_inits(av, inv, nullptr);
        for (size_t s = 0; s < nbatch; ++s) {
            words_to_mpz(av, &A[s * m * W], W);
            if (mpz_invert(inv, av, L.N) == 0) {
                std::fprintf(stderr, "%s: FATAL: divisor leading coefficient not invertible "
                                     "mod N (the divisor was not monic)\n", NTT_PROBE_NAME);
                std::exit(4);
            }
            mpz_mod(inv, inv, L.N);
            mpz_to_words(gtmp, W, inv);
            std::copy(gtmp.begin(), gtmp.end(), g.begin() + (long)(s * W));
        }
        mpz_clears(av, inv, nullptr);
    }
    size_t len = 1;
    std::vector<unsigned long long> at, ag, h, gn;
    std::vector<unsigned long long> gpad;      /* g zero-extended to the current nxt */
    while (len < k) {
        const size_t nxt = ((2 * len) < k) ? (2 * len) : k;
        const size_t am = (m < nxt) ? m : nxt;
        at.assign(nbatch * am * W, 0ull);
        for (size_t s = 0; s < nbatch; ++s)
            std::copy(A.begin() + (long)(s * m * W), A.begin() + (long)(s * m * W + am * W),
                      at.begin() + (long)(s * am * W));
        /* g zero-extended to nxt coefficients with the extension FORCED to zero: g is the
           inverse of a mod X^len, so coefficients len..nxt-1 are identically zero.  (The
           extension is also what lets the products below reach degree nxt-1 regardless of how
           short g currently is.) */
        gpad.assign(nbatch * nxt * W, 0ull);
        for (size_t s = 0; s < nbatch; ++s)
            std::copy(g.begin() + (long)(s * len * W),
                      g.begin() + (long)((s + 1) * len * W),
                      gpad.begin() + (long)(s * nxt * W));
        /* Every flat consumer uses the declared coefficient stride, including
       padding and unequal operand widths. Select outputs with that stride. */
        flat_mul_batch(L, at, am, (len == nxt) ? g : gpad, nxt, nbatch, ag, cat, 0, nxt);
        /* The returned prefix is already tight at nxt coefficients per slice; the input and
           transform shape still cover the complete product. */
        if (ag.size() != nbatch * nxt * W) ag.resize(nbatch * nxt * W, 0ull);
        h.assign(nbatch * nxt * W, 0ull);
        {
            mpz_t two;
            mpz_init(two);
            mpz_set_ui(two, 2);
            mpz_mod(two, two, L.N);
            std::vector<unsigned long long> w2(W, 0ull);
            mpz_to_words(w2, W, two);
            for (size_t s = 0; s < nbatch; ++s)
                std::copy(w2.begin(), w2.end(), h.begin() + (long)(s * nxt * W));
            mpz_clear(two);
        }
        h = cp_addsub_flat(h, ag, nbatch * nxt, L.W, L.N, /*sub=*/true);   /* h = 2 - ag */
        h.resize(nbatch * nxt * W);
        /* ---- g = (g*h) mod X^nxt ---------------------------------------------------- */
        flat_mul_batch(L, gpad, nxt, h, nxt, nbatch, gn, cat, 0, nxt);
        if (gn.size() != nbatch * nxt * W) gn.resize(nbatch * nxt * W, 0ull);
        g.swap(gn);
        len = nxt;
    }
}

void divmod_batch(
    PolyLayer &L,
    const std::vector<unsigned long long> &A,
    const std::vector<unsigned long long> &B,
    size_t da,
    size_t db,
    size_t nbatch,
    std::vector<unsigned long long> &out,
    int cat)
{
    const size_t W = L.W;
    const size_t k = da - db + 1;
    std::vector<unsigned long long> ra(nbatch * k * W, 0ull), rb(nbatch * (db + 1) * W, 0ull);
    for (size_t s = 0; s < nbatch; ++s) {
        for (size_t i = 0; i < k; ++i)
            std::copy(A.begin() + (long)((s * (da + 1) + (da - i)) * W),
                      A.begin() + (long)((s * (da + 1) + (da - i) + 1) * W),
                      ra.begin() + (long)((s * k + i) * W));
        for (size_t i = 0; i <= db; ++i)
            std::copy(B.begin() + (long)((s * (db + 1) + (db - i)) * W),
                      B.begin() + (long)((s * (db + 1) + (db - i) + 1) * W),
                      rb.begin() + (long)((s * (db + 1) + i) * W));
    }
    std::vector<unsigned long long> rbi;
    inv_series_batch(L, rb, db + 1, k, nbatch, rbi, cat);
    if (rbi.size() < nbatch * k * W) {
        std::fprintf(stderr, "%s: FATAL: the batched inverse returned %llu words, expected "
                             "%llu (nbatch=%llu k=%llu W=%llu)\n", NTT_PROBE_NAME,
                     (unsigned long long)rbi.size(), (unsigned long long)(nbatch * k * W),
                     (unsigned long long)nbatch, (unsigned long long)k, (unsigned long long)W);
        std::exit(3);
    }
    std::vector<unsigned long long> qrev, q, qb;
    /* Every flat consumer uses the declared coefficient stride, including
       padding and unequal operand widths. Select outputs with that stride. */
    if (rbi.size() != nbatch * k * W) {
        std::fprintf(stderr, "%s: FATAL: the batched inverse has %llu words, expected %llu "
                             "(nbatch=%llu k=%llu W=%llu)\n", NTT_PROBE_NAME,
                     (unsigned long long)rbi.size(), (unsigned long long)(nbatch * k * W),
                     (unsigned long long)nbatch, (unsigned long long)k, (unsigned long long)W);
        std::exit(3);
    }
    flat_mul_batch(L, ra, k, rbi, k, nbatch, qrev, cat, 0, k);
    q.assign(nbatch * k * W, 0ull);
    for (size_t s = 0; s < nbatch; ++s)
        for (size_t i = 0; i < k; ++i)
            std::copy(qrev.begin() + (long)((s * k + (k - 1 - i)) * W),
                      qrev.begin() + (long)((s * k + (k - 1 - i) + 1) * W),
                      q.begin() + (long)((s * k + i) * W));
    flat_mul_batch(L, q, k, B, db + 1, nbatch, qb, cat, 0, db);
    out.assign(nbatch * db * W, 0ull);
    for (size_t s = 0; s < nbatch; ++s)
        for (size_t i = 0; i < db; ++i)
            cp_coeff_sub_p(&out[(s * db + i) * W], &A[(s * (da + 1) + i) * W],
                           &qb[(s * db + i) * W], L.N, W);
}

bool ScaledFrontierDevice::init(FoldDeviceState &owner)
{
    const double begin=now_s();fold=&owner;
    const size_t p=owner.P,w=owner.W;
    unsigned long long pw=0,need=0,bytes=0;
    if(!owner.active || !p || !w || owner.t!=0 ||
       !ecm_stage2::multiply(p,w,pw) || !ecm_stage2::multiply(pw,3,need) ||
       !ecm_stage2::multiply(p,3,bytes) || !ecm_stage2::multiply(bytes,8,bytes) ||
       !ecm_stage2::add(owner.stats->layout_bytes,bytes,owner_and_metadata_bytes) ||
       need>owner.memory.words[0] || need>owner.memory.words[1]) {fallback="owner_shape";return false;}
    metadata_bytes=bytes;metadata_words=3*p;
    auto max_mb=fuse_env_ull("NTT_FOLD_DEVICE_MAX_MB",640);
    max_mb=std::min(max_mb,fuse_env_ull("NTT_SCALED_FRONTIER_MAX_MB",max_mb));
    if(max_mb>(~0ull>>20) || owner_and_metadata_bytes>(max_mb<<20)) {fallback="budget";return false;}
    size_t available=0,total=0;CK(cudaMemGetInfo(&available,&total));
    unsigned long long qn=0;
    if(!ntt_shape_query(p+1,(int)owner.layer->S,&qn,nullptr,nullptr,nullptr,nullptr,nullptr)) {fallback="ntt_shape";return false;}
    const unsigned buffers=owner.layer->arena->big_buffer_count();
    const size_t target=8ull*buffers*qn,current=8ull*owner.layer->arena->workspace.words;
    const size_t growth=target>current?target-current:0;
    const size_t future_reserve=1ull<<30;
    if(gscale_flag("NTT_OWNER_TRIM_FUSE"))
        ntt_trim_fuse_headroom(owner.layer->arena,qn,bytes+growth+future_reserve,"fold_to_descent",available);
    const bool fits=bytes<=available && available-bytes>=growth+future_reserve;
    stage2_log::print(stage2_log::debug,
        "frontier_device_headroom: available_bytes=%llu metadata_bytes=%llu physical_buffers=%u target_big_bytes=%llu current_big_bytes=%llu growth_bytes=%llu future_reserve_bytes=%llu fits=%d\n",
        (unsigned long long)available,bytes,buffers,(unsigned long long)target,
        (unsigned long long)current,(unsigned long long)growth,(unsigned long long)future_reserve,(int)fits);
    if(!fits) {fallback="headroom";return false;}
    if(gscale_flag("NTT_SCALED_FRONTIER_ALLOC_FAIL")) {fallback="allocation_fixture";return false;}
    const auto error=cudaMalloc(&metadata,(size_t)bytes);
    if(error==cudaErrorMemoryAllocation){cudaGetLastError();metadata=nullptr;fallback="allocation";return false;}
    CK(error);enabled=true;setup_seconds=now_s()-begin;return true;
}

void ScaledFrontierDevice::upload_siblings(
    unsigned long long *dst,
    const std::vector<std::vector<unsigned long long>> &Ft,
    size_t nbase,
    const std::vector<std::pair<size_t,
    size_t>> &group,
    size_t mb,
    size_t w)
{
    const double begin=now_s();const size_t slice=mb*w,words=group.size()*slice;
    bool available[2]={false,false},used[2]={false,false};
    if(g_s4_async)for(int i=0;i<2;++i)available[i]=g_pin_out[i] && g_pin_out_cap[i] && g_pin_ev[i];
    auto copy=[&](unsigned long long *out,size_t first,size_t count) {
        while(count) {
            const size_t s=first/slice,offset=first%slice,take=std::min(count,slice-offset);
            const auto &f=Ft[(nbase+group[s].second)^1];
            if(f.size()!=slice){std::fprintf(stderr,"FATAL: scaled frontier sibling length\n");std::exit(3);}
            std::copy_n(f.data()+offset,take,out);out+=take;first+=take;count-=take;
        }
    };
    if(available[0] || available[1]) {
        size_t offset=0,turn=0;
        while(offset<words) {
            size_t i=turn++&1;if(!available[i])i^=1;
            CK(stage2_wait::event_sync(g_pin_ev[i],stage2_wait::Kind::upload));
            const size_t take=std::min(words-offset,g_pin_out_cap[i]);
            copy(g_pin_out[i],offset,take);
            CK(cudaMemcpyAsync(dst+offset,g_pin_out[i],take*8,cudaMemcpyHostToDevice));
            CK(cudaEventRecord(g_pin_ev[i]));used[i]=true;offset+=take;
            pinned_borrow_peak_bytes=std::max(pinned_borrow_peak_bytes,8ull*take);
        }
        for(int i=0;i<2;++i)if(used[i])CK(stage2_wait::event_sync(g_pin_ev[i],stage2_wait::Kind::upload));
    } else {
        std::vector<unsigned long long> packed(words);copy(packed.data(),0,words);
        CK(stage2_wait::copy(dst,packed.data(),words*8,cudaMemcpyHostToDevice));
        host_staging_peak_bytes=std::max(host_staging_peak_bytes,8ull*packed.capacity());
    }
    f_h2d_bytes+=8ull*words;upload_seconds+=now_s()-begin;
}

void ScaledFrontierDevice::run(
    PolyLayer &L,
    const std::vector<std::vector<unsigned long long>> &Ft,
    const std::vector<size_t> &deg,
    size_t pad,
    const CPoly &H,
    std::vector<std::vector<unsigned long long>> &values,
    ScaledStats &st,
    int cat,
    bool check)
{
    const size_t p=fold->P,w=fold->W,pw=p*w;
    if(!enabled || !metadata || !fold->active || deg.size()<2*pad || Ft.size()<2*pad || deg[1]!=p ||
       (check && p>512)) {std::fprintf(stderr,"FATAL: scaled frontier live shape/check scope\n");std::exit(3);}
    auto *cur=fold->memory.data[1],*next=fold->memory.data[0];
    std::vector<unsigned long long> offsets(1,0);
    ++st.root_inverse_reused;++st.mul_calls;++st.mul_pairs;
    const bool poison=gscale_flag("NTT_SCALED_FRONTIER_TEST_BAD");
    if(poison && !check){std::fprintf(stderr,"FATAL: scaled frontier poison requires GMP check\n");std::exit(3);}
    auto corrupt=[&](unsigned long long *dst) {
        unsigned long long v=0;CK(stage2_wait::copy(&v,dst,8,cudaMemcpyDeviceToHost));v^=1;
        CK(stage2_wait::copy(dst,&v,8,cudaMemcpyHostToDevice));check_d2h_bytes+=8;
    };
    auto validate=[&](size_t base) {
        std::vector<unsigned long long> actual;
        if(check){actual.resize(pw);CK(stage2_wait::copy(actual.data(),cur,pw*8,cudaMemcpyDeviceToHost));check_d2h_bytes+=pw*8;}
        size_t words=0;
        for(size_t i=0;i<base;++i) {
            const size_t d=deg[base+i];
            if(offsets[i]!=words || d>p-words/w){std::fprintf(stderr,"FATAL: scaled frontier degree/offset\n");std::exit(3);}
            ++st.states;st.words+=d*w;words+=d*w;
            if(check) {
                const auto expected=scaled_state_gmp(H,Ft[base+i],d,L);
                if(!std::equal(expected.begin(),expected.end(),actual.begin()+offsets[i])) {
                    std::fprintf(stderr,"FATAL: scaled frontier GMP node mismatch node=%llu degree=%llu\n",
                        (unsigned long long)(base+i),(unsigned long long)d);std::exit(3);
                }
                ++st.checked_states;st.checked_words+=d*w;
            }
        }
        if(words!=pw){std::fprintf(stderr,"FATAL: scaled frontier degree conservation\n");std::exit(3);}
    };
    if(poison && pad==1)corrupt(cur);
    validate(1);
    for(size_t base=1;base<pad;base*=2) {
        const size_t nbase=2*base;
        std::vector<unsigned long long> target(nbase);size_t words=0;
        for(size_t i=0;i<nbase;++i){target[i]=words;words+=deg[nbase+i]*w;}
        if(words!=pw){std::fprintf(stderr,"FATAL: scaled frontier next length\n");std::exit(3);}
        logical_frontier_peak_bytes=std::max(logical_frontier_peak_bytes,16ull*pw);
        std::map<std::pair<size_t,size_t>,std::vector<std::pair<size_t,size_t>>> groups;
        for(size_t j=0;j<base;++j)for(size_t side=0;side<2;++side) {
            const size_t slot=2*j+side,ci=nbase+slot,a=deg[ci],b=deg[ci^1];
            if(a+b!=deg[base+j]){std::fprintf(stderr,"FATAL: scaled frontier child degrees\n");std::exit(3);}
            if(!a){++st.zeros;continue;}
            if(!b){CK(cudaMemcpyAsync(next+target[slot],cur+offsets[j],a*w*8,cudaMemcpyDeviceToDevice));++st.copies;continue;}
            groups[{a,b}].push_back({j,slot});
        }
        for(const auto &g:groups) {
            const size_t a=g.first.first,b=g.first.second,ma=a+b,mb=b+1,nb=g.second.size();
            if(nb>p || mb>2*p/nb){std::fprintf(stderr,"FATAL: scaled frontier sibling budget\n");std::exit(3);}
            std::vector<unsigned long long> map(3*nb);
            for(size_t s=0;s<nb;++s){map[s]=offsets[g.second[s].first];map[nb+s]=pw+s*mb*w;map[2*nb+s]=target[g.second[s].second];}
            upload_siblings(cur+pw,Ft,nbase,g.second,mb,w);
            CK(stage2_wait::copy(metadata,map.data(),map.size()*8,cudaMemcpyHostToDevice));metadata_h2d_bytes+=map.size()*8;
            S4DeviceBatch batch{cur,metadata,next,map.data(),pw+nb*mb*w,pw,nb,&fold->memory,true};
            std::vector<unsigned long long> unused;
            poly_mul_batch_modN(L,nullptr,nullptr,ma,mb,nb,unused,cat,nullptr,b,a,&batch);
            ++st.mul_calls;st.mul_pairs+=nb;
            avoided_parent_h2d_bytes+=8ull*nb*ma*w;avoided_state_d2h_bytes+=8ull*nb*a*w;
        }
        if(poison && base==1)corrupt(next);
        std::swap(cur,next);offsets.swap(target);++st.levels;validate(nbase);
    }
    std::vector<unsigned long long> leaves(pw);CK(stage2_wait::copy(leaves.data(),cur,pw*8,cudaMemcpyDeviceToHost));leaf_d2h_bytes=pw*8;
    values.assign(p,std::vector<unsigned long long>(w));size_t leaf=0;
    for(size_t i=0;i<pad;++i)if(deg[pad+i]) {
        if(deg[pad+i]!=1 || leaf>=p){std::fprintf(stderr,"FATAL: scaled frontier nonlinear leaf\n");std::exit(3);}
        std::copy_n(leaves.data()+offsets[i],w,values[leaf++].data());
    }
    if(leaf!=p){std::fprintf(stderr,"FATAL: scaled frontier leaf count\n");std::exit(3);}
    st.leaves+=leaf;
    st.frontier_peak_bytes=logical_frontier_peak_bytes;
    st.pack_peak_bytes=host_staging_peak_bytes;
}

void ScaledFrontierDevice::print()const
{
    stage2_log::print(stage2_log::phases, "scaled_frontier_device: requested=%d enabled=%d metadata_bytes=%llu owner_and_metadata_bytes=%llu f_h2d_bytes=%llu metadata_h2d_bytes=%llu leaf_d2h_bytes=%llu check_d2h_bytes=%llu avoided_parent_h2d_bytes=%llu avoided_state_d2h_bytes=%llu logical_frontier_peak_bytes=%llu host_staging_peak_bytes=%llu pinned_borrow_peak_bytes=%llu setup_seconds=%.6f upload_seconds=%.6f fallback=%s\n",
        (int)requested,(int)enabled,metadata_bytes,owner_and_metadata_bytes,f_h2d_bytes,metadata_h2d_bytes,leaf_d2h_bytes,check_d2h_bytes,
        avoided_parent_h2d_bytes,avoided_state_d2h_bytes,logical_frontier_peak_bytes,host_staging_peak_bytes,pinned_borrow_peak_bytes,setup_seconds,upload_seconds,fallback);
}
