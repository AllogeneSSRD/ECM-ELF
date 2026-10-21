#include "runtime.h"
int gl_selftest()
{
    ntt_gl_reduce_configure();
    mpz_t p, a, b, z, want, got;
    mpz_inits(p, a, b, z, want, got, nullptr);
    const unsigned long long glp = GL_P;
    mpz_import(p, 1, -1, 8, 0, 0, &glp);

    unsigned long long s = 0x243F6A8885A308D3ull;
    const unsigned long long edges[] = {0ull, 1ull, 2ull, GL_P - 1, GL_P - 2, GL_P >> 1,
                                        0xFFFFFFFFull, 0xFFFFFFFF00000000ull};
    const int nedges = (int)(sizeof(edges) / sizeof(edges[0]));
    int tested = 0;
    int reduce_cases = 0;
    for (int t = 0; t < 3000; ++t) {
        unsigned long long x, y;
        if (t < nedges * nedges) {
            x = edges[t / nedges];
            y = edges[t % nedges];
        } else {
            s = s * 6364136223846793005ull + 1442695040888963407ull;
            x = s;
            s = s * 6364136223846793005ull + 1442695040888963407ull;
            y = s;
        }
        x %= GL_P; y %= GL_P;
        const unsigned long long got_d = gl_mul_host(x, y);
        /* GMP: exact product then one mod */
        mpz_import(a, 1, -1, 8, 0, 0, &x);
        mpz_import(b, 1, -1, 8, 0, 0, &y);
        mpz_mul(z, a, b);
        mpz_mod(want, z, p);
        mpz_import(got, 1, -1, 8, 0, 0, &got_d);
        ++tested;
        if (mpz_cmp(want, got) != 0) {
            std::fprintf(stderr, "gl_selftest: FAILED at x=%llu y=%llu (got %llu, want %s)\n",
                         x, y, got_d, mpz_get_str(nullptr, 10, want));
            mpz_clears(p, a, b, z, want, got, nullptr);
            return 1;
        }
    }
    /* a few modular exponentiations too: gl_pow is what builds the omega / 1/N constants */
    for (int t = 0; t < 24; ++t) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        const unsigned long long base = s % GL_P;
        const unsigned long long e = (s >> 7) % (1ull << 40);
        const unsigned long long got_d = gl_pow_host(base, e);
        mpz_import(a, 1, -1, 8, 0, 0, &base);
        mpz_import(b, 1, -1, 8, 0, 0, &e);
        mpz_powm(want, a, b, p);
        mpz_import(got, 1, -1, 8, 0, 0, &got_d);
        ++tested;
        if (mpz_cmp(want, got) != 0) {
            std::fprintf(stderr, "gl_selftest: gl_pow FAILED base=%llu e=%llu\n", base, e);
            mpz_clears(p, a, b, z, want, got, nullptr);
            return 1;
        }
    }
    mpz_clears(p, a, b, z, want, got, nullptr);

    /*
     * Independent test of gl_reduce on ARBITRARY (lo, hi) pairs, including the full
     * hi < 2^64 range: this is the core that the two-step fold got wrong, and it is not
     * reachable through gl_mul alone (gl_mul always produces a genuine 64x64 product).
     */
    {
        mpz_t p2, lo2, hi2, got2, w2;
        mpz_inits(p2, lo2, hi2, got2, w2, nullptr);
        mpz_import(p2, 1, -1, 8, 0, 0, &glp);
        unsigned long long r = 0xC0FFEE123456789ull;
        int bad = 0;
        std::vector<unsigned long long> los, his;
        /* hand-picked boundaries plus random pairs */
        const unsigned long long los0[] = {0, 1, 2, GL_P - 1, GL_P - 2, 0xFFFFFFFFull,
                                           0xFFFFFFFF00000000ull, ~0ull};
        const unsigned long long his0[] = {0, 1, 2, 3, 0xFFFFFFFFull, 0x1FFFFFFFFull,
                                           0xFFFFFFFF00000000ull, ~0ull};
        for (unsigned long long x : los0) {
            for (unsigned long long y : his0) { los.push_back(x); his.push_back(y); }
        }
        for (int t = 0; t < 100000; ++t) {
            r = r * 6364136223846793005ull + 1442695040888963407ull;
            los.push_back(r);
            r = r * 6364136223846793005ull + 1442695040888963407ull;
            /* mix: sometimes a near-2^64 hi (what a product of two values < p produces) */
            his.push_back((t & 1) ? r : (0xFFFFFFFFFFFFFFFFull - (r & 0xFFFFFFFFull)));
        }
        for (size_t i = 0; i < los.size(); ++i) {
            const unsigned long long got_d = gl_reduce(los[i], his[i]);
            mpz_import(lo2, 1, -1, 8, 0, 0, &los[i]);
            mpz_import(hi2, 1, -1, 8, 0, 0, &his[i]);
            mpz_mul_2exp(hi2, hi2, 64);
            mpz_add(hi2, hi2, lo2);
            mpz_mod(w2, hi2, p2);   /* NOT `want`: it was mpz_clear()ed at the end of the block above, and writing through a cleared mpz_t is a use-after-free -- it corrupted the heap (0xC0000374 / 0xC0000005) in ~5 % of runs, before the first output line */
            mpz_import(got2, 1, -1, 8, 0, 0, &got_d);
            if (mpz_cmp(w2, got2) != 0) {
                if (bad == 0) {
                    std::fprintf(stderr,
                                 "gl_selftest: gl_reduce FAILED lo=%llu hi=%llu got=%llu\n",
                                 los[i], his[i], got_d);
                }
                ++bad;
            }
        }
        mpz_clears(p2, lo2, hi2, got2, w2, nullptr);
        if (bad) return 1;
        reduce_cases = (int)los.size();
        stage2_log::print(stage2_log::debug, NTT_PROBE_NAME ": gl_reduce selftest vs GMP on %d (lo,hi) pairs OK\n",
                    reduce_cases);
    }
    mpz_inits(p, a, b, z, want, got, nullptr);
    mpz_import(p, 1, -1, 8, 0, 0, &glp);

    /*
     * The kernels use the DEVICE gl_mul (the two-64-bit-halves version), which is a
     * different code path from the host's split version, so run the same comparison on the
     * device.  200k pairs, ~0.2 s.
     */
    const int nt = 200000;
    std::vector<unsigned long long> hx(nt), hy(nt);
    {
        unsigned long long r = 0x9E3779B97F4A7C15ull;
        for (int i = 0; i < nt; ++i) {
            if (i < nedges * nedges) {
                hx[(size_t)i] = edges[i / nedges];
                hy[(size_t)i] = edges[i % nedges];
            } else {
                r = r * 6364136223846793005ull + 1442695040888963407ull; hx[(size_t)i] = r;
                r = r * 6364136223846793005ull + 1442695040888963407ull; hy[(size_t)i] = r;
            }
            hx[(size_t)i] %= GL_P;
            hy[(size_t)i] %= GL_P;
        }
    }
    unsigned long long *dx = nullptr, *dy = nullptr, *dz = nullptr;
    CK(cudaMalloc(&dx, (size_t)nt * 8));
    CK(cudaMalloc(&dy, (size_t)nt * 8));
    CK(cudaMalloc(&dz, (size_t)nt * 8));
    CK(cudaMemcpy(dx, hx.data(), (size_t)nt * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dy, hy.data(), (size_t)nt * 8, cudaMemcpyHostToDevice));
    ecm_launch_gl_mul_test_kernel(EcmCudaLaunch{(nt + 255) / 256, 256}, dx, dy, dz, nt);
    CK(cudaGetLastError());
    std::vector<unsigned long long> hz(nt, 0);
    CK(cudaMemcpy(hz.data(), dz, (size_t)nt * 8, cudaMemcpyDeviceToHost));

    int dtested = 0, dbad = 0;
    for (int i = 0; i < nt; ++i) {
        mpz_import(a, 1, -1, 8, 0, 0, &hx[(size_t)i]);
        mpz_import(b, 1, -1, 8, 0, 0, &hy[(size_t)i]);
        mpz_mul(z, a, b);
        mpz_mod(want, z, p);
        mpz_import(got, 1, -1, 8, 0, 0, &hz[(size_t)i]);
        ++dtested;
        if (mpz_cmp(want, got) != 0) {
            if (dbad == 0) {
                std::fprintf(stderr, "gl_selftest(device): FAILED at x=%llu y=%llu got=%llu\n",
                             hx[(size_t)i], hy[(size_t)i], hz[(size_t)i]);
            }
            ++dbad;
        }
    }
    mpz_clears(p, a, b, z, want, got, nullptr);
    cudaFree(dx); cudaFree(dy); cudaFree(dz);
    if (dbad) return 1;
    stage2_log::print(stage2_log::debug, NTT_PROBE_NAME ": reduction selftest vs GMP: %d host + %d device cases and %d "
                "(lo,hi) folds OK (p=0x%llX)\n", tested, dtested, reduce_cases,
                (unsigned long long)GL_P);
    return 0;
}

void ntt_workspace_check(int device)
{
#ifdef ECM_STAGE2_MEMORY_LEDGER
    stage2_memory::fixture();
#endif
#ifdef ECM_STAGE2_GIANT_CHUNK_PLAN
    stage2_giant_chunk::fixture();
#endif
    unsigned long long checks=0, bad=0, words=0;
    auto check=[&](bool ok) { ++checks; if (!ok) ++bad; };
    for(bool reuse_bq : {false,true}) {
        NttArena ar; ar.device=device; ar.workspace_pool=true;ar.workspace_reuse_bq=reuse_bq;
        auto *first=ntt_arena_bufs(&ar,128,10,4);
        check(first!=nullptr); if (!first) std::exit(3);
        check(ar.workspace.capacity==512 && ar.workspace.words==512*(reuse_bq?2:3));
        check((first->dQ==first->dB)==reuse_bq && first->dQ!=first->dA);
        auto *a=first->dA; auto *res=first->dRes;
        unsigned long long poison=77, value=0;
        CK(cudaMemcpy(res,&poison,8,cudaMemcpyHostToDevice));
        auto *second=ntt_arena_bufs(&ar,256,10,2);
        check(second && second->dA==a && second->dRes!=res && ar.workspace_grows==1);
        auto *again=ntt_arena_bufs(&ar,128,10,4);
        check(again && again->dA==a && again->dRes==res);
        CK(cudaMemcpy(&value,res,8,cudaMemcpyDeviceToHost)); check(value==77);
        size_t span=0; check(ar.input_span(a+3,span) && span==509);
        check(ar.input_span(first->dQ+3,span) && span==509);
        const size_t small_charge=ar.bytes-ar.workspace.words*8;
        for (int index=1; index<=(reuse_bq?2:3); ++index) {
            ar.workspace_fail_alloc=index;
            check(ntt_arena_bufs(&ar,512,20,2)==nullptr);
            check(ar.workspace.words==0 && ar.bytes==small_charge);
        }
        ar.workspace_fail_alloc=0;
        check(ntt_arena_bufs(&ar,512,20,2)!=nullptr && ar.workspace.capacity==1024 &&
              ar.workspace.words==1024*(reuse_bq?2:3));
        CK(cudaMemcpy(&value,res,8,cudaMemcpyDeviceToHost)); check(value==77);
        ar.cap_bytes=ar.bytes;
        check(ntt_arena_bufs(&ar,4096,20,2)==nullptr && ar.workspace.words==0);
        ar.release(); check(ar.bytes==0 && ar.bigs.empty() && ar.smalls.empty());
        ar.release(); check(ar.bytes==0);
    }
    const unsigned long long P=3, nb=4;
    const int S=129;
    const size_t W=3;
    std::vector<unsigned long long> a(nb*P*W,0), b(a.size());
    for (size_t s=0; s<nb; ++s) for (size_t i=0; i<P; ++i) {
        a[(s*P+i)*W]=i+s+1; a[(s*P+i)*W+2]=1;
        b[(s*P+i)*W]=2*i+s+4; b[(s*P+i)*W+1]=s+1;
    }
    for (bool pool : {false,true}) for(bool reuse_bq : {false,true}) {
        NttArena ar; ar.device=device; ar.workspace_pool=pool;ar.workspace_reuse_bq=reuse_bq;
        std::vector<unsigned long long> output;
        unsigned long long *exported=nullptr;
        NttMulStats st;
        int rc=ntt_poly_mul_batch_host(P,S,device,nb,a.data(),b.data(),&output,&st,&ar,nullptr,&exported);
        check(rc==0 && exported && ar.workspace.words==0 && !ar.bigs.empty());
        if (rc || !exported) std::exit(3);
        check(ar.bigs.back().dQ!=ar.bigs.back().dB && !ar.bigs.back().q_alias_b);
        std::vector<unsigned long long> expected(nb*st.out_slots,0);
        mpz_t x,y,sum; mpz_inits(x,y,sum,nullptr);
        for (size_t s=0; s<nb; ++s) for (size_t k=0; k<2*P-1; ++k) {
            mpz_set_ui(sum,0);
            for (size_t i=0; i<P; ++i) if (k>=i && k-i<P) {
                mpz_import(x,W,-1,8,0,0,&a[(s*P+i)*W]);
                mpz_import(y,W,-1,8,0,0,&b[(s*P+k-i)*W]);
                mpz_mul(x,x,y); mpz_add(sum,sum,x);
            }
            expected[s*st.out_slots+k]=mpz_fdiv_ui(sum,4294967291ul);
        }
        mpz_clears(x,y,sum,nullptr);
        check(output==expected); words+=expected.size();
        NttShape sh; FuseCtx fc;
        check(ntt_shape_plan(P,S,device,&ar,fc,sh)==0 && sh.N==st.N);
        std::vector<unsigned long long> pa(nb*sh.N,0), pb(pa.size());
        unsigned long long maxdigit=0;
        for (size_t s=0; s<nb; ++s)
            check(ntt_pack_operand(sh,&a[s*P*W],&b[s*P*W],&pa[s*sh.N],&pb[s*sh.N],&maxdigit));
        /* Crossed inputs test overwrite hazards. Snapshot before either destination write. */
        auto *src_a=ar.cur.dB, *src_b=ar.cur.dA;
        CK(cudaMemcpy(src_a,pa.data(),pa.size()*8,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(src_b,pb.data(),pb.size()*8,cudaMemcpyHostToDevice));
        const auto aliases_before=ar.alias_snapshots;
        /* A shorter batch forces eviction of the nb=4 source when the pool grows to nb=3. */
        ar.cap_bytes=ar.bytes;
        rc=ntt_poly_mul_batch_dev(P,S,device,3,src_a,src_b,&st,&ar,nullptr,nullptr,sh.bpw);
        check(rc==0 && ar.alias_snapshots==aliases_before+2);
        check(pool ? ar.bigs.empty() : ar.bigs.size()==1 && ar.bigs[0].nbatch==3);
        if (rc) std::exit(3);
        output.resize(3*st.out_slots);
        CK(cudaMemcpy(output.data(),ar.cur.dOut,output.size()*8,cudaMemcpyDeviceToHost));
        expected.resize(output.size()); check(output==expected); words+=output.size();
        check(st.carry_deferred==false);
        check((ar.cur.dQ==ar.cur.dB)==(pool && reuse_bq));
        /* The previous digit output is an arena input too, including Q==B. Preserve both
           crossed sources BEFORE copying either destination or growing the workspace. */
        src_a=ar.cur.dQ;src_b=ar.cur.dA;
        CK(cudaMemcpy(src_a,pa.data(),3*sh.N*8,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(src_b,pb.data(),3*sh.N*8,cudaMemcpyHostToDevice));
        const auto q_aliases_before=ar.alias_snapshots;
        rc=ntt_poly_mul_batch_dev(P,S,device,3,src_a,src_b,&st,&ar,nullptr,nullptr,sh.bpw);
        check(rc==0 && ar.alias_snapshots==q_aliases_before+2);
        CK(cudaMemcpy(output.data(),ar.cur.dOut,output.size()*8,cudaMemcpyDeviceToHost));
        check(output==expected);words+=output.size();
        /* Force owned fallback. Its carry has already been checked, so never defer it. */
        ar.cap_bytes=1;
        unsigned long long *di_a=nullptr, *di_b=nullptr;
        CK(cudaMalloc(&di_a,3*sh.N*8)); CK(cudaMalloc(&di_b,3*sh.N*8));
        CK(cudaMemcpy(di_a,pa.data(),3*sh.N*8,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(di_b,pb.data(),3*sh.N*8,cudaMemcpyHostToDevice));
        rc=ntt_poly_mul_batch_dev(P,S,device,3,di_a,di_b,&st,&ar,nullptr,nullptr,sh.bpw,true);
        if (st.carry_deferred) { NttMulStats finish; check(ntt_batch_carry_finish(&ar,st.N,3,&finish)==0); }
        /* Existing cached buffers can still be reused under a reduced cap; evict first. */
        if (!ar.workspace.words && ar.bigs.empty()) check(rc==0 && !st.carry_deferred);
        ar.release();
        rc=ntt_poly_mul_batch_dev(P,S,device,3,di_a,di_b,&st,&ar,nullptr,nullptr,sh.bpw,true);
        check(rc==0 && !st.carry_deferred && ar.overflow>0);
        exported=(unsigned long long *)1;
        rc=ntt_poly_mul_batch_dev(P,S,device,3,di_a,di_b,&st,&ar,nullptr,&exported,sh.bpw);
        check(rc==3 && exported==nullptr);
        /* dRes must survive both deferred multiplies even when B becomes carry output.
           Its second word is a monotone maximum; a sentinel exposes an unwanted reset. */
        ar.cap_bytes=0;
        auto *deferred=ntt_arena_bufs(&ar,sh.N,st.out_slots,3);
        check(deferred!=nullptr);if(!deferred)std::exit(3);
        CK(cudaMemset(deferred->dRes,0,6*8));
        const unsigned long long sentinel=123;
        CK(cudaMemcpy(deferred->dRes+1,&sentinel,8,cudaMemcpyHostToDevice));
        for(int pass=0;pass<2;++pass) {
            rc=ntt_poly_mul_batch_dev(P,S,device,3,di_a,di_b,&st,&ar,nullptr,nullptr,sh.bpw,true);
            check(rc==0 && st.carry_deferred);
        }
        NttMulStats finish;
        check(ntt_batch_carry_finish(&ar,sh.N,3,&finish)==0 && finish.carry_max_bits==sentinel);
        CK(cudaMemcpy(output.data(),ar.cur.dOut,output.size()*8,cudaMemcpyDeviceToHost));
        check(output==expected);words+=output.size();
        CK(cudaFree(di_a)); CK(cudaFree(di_b));
    }
    {
        NttArena ar;ar.device=device;ar.workspace_pool=true;ar.workspace_reuse_bq=true;
        auto *buf=ntt_arena_bufs(&ar,128,10,4);check(buf!=nullptr);
        auto *verdict=buf->dRes;const unsigned long long marker=77;
        CK(cudaMemcpy(verdict,&marker,8,cudaMemcpyHostToDevice));
        NttShape sh;FuseCtx fc;unsigned long long keep=0;
        for(unsigned long long p:{3ull,17ull,65ull}) {
            check(ntt_shape_plan(p,129,device,&ar,fc,sh)==0);keep=sh.N;
            check(ar.bytes==ar.payload().total);
        }
        const auto before=ar.bytes;size_t removed=0;
        while(const auto freed=ar.drop_cold_fuse(keep)) {
            removed+=freed;check(ar.bytes==ar.payload().total);
            check(std::any_of(ar.fuses.begin(),ar.fuses.end(),[&](const auto &e){return e.n==keep;}));
        }
        check(removed>0 && before-ar.bytes==removed && ar.fuses.size()==1 && ar.drop_cold_fuse(keep)==0);
        unsigned long long value=0;CK(cudaMemcpy(&value,verdict,8,cudaMemcpyDeviceToHost));check(value==marker);
        const size_t p=17,w=3;std::vector<unsigned long long> a(p*w,0),b(a.size()),out,expected(2*p-1,0);
        for(size_t i=0;i<p;++i){a[i*w]=i+1;b[i*w]=2*i+1;}
        for(size_t i=0;i<p;++i)for(size_t j=0;j<p;++j)expected[i+j]+=a[i*w]*b[j*w];
        NttMulStats stats;
        check(ntt_poly_mul_batch_host(p,129,device,1,a.data(),b.data(),&out,&stats,&ar,nullptr)==0 && out==expected);
        check(ar.fuses.size()==2 && ar.bytes==ar.payload().total);
        ar.release();check(ar.payload().total==0 && ar.bytes==0);
    }
    stage2_log::print(stage2_log::debug, "ntt_workspace_check: checks=%llu words=%llu bad=%llu "
                "(GMP, two/three buffers, capacity reuse, dRes/deferred isolation, failure rollback, aliases/exports/fallback)\n",checks,words,bad);
    if (bad) std::exit(3);
}

void ntt_fuse_lifetime_check(int device)
{
    const unsigned long long P=17;
    const int S=129;
    std::vector<unsigned long long> a(P*3,0), b(P*3,0), output;
    for (size_t i=0;i<P;++i) { a[i*3]=i+1; b[i*3]=2*i+3; }
    NttArena ar; ar.device=device; ar.cap_bytes=1;
    const auto live_before=g_fuse_base.live_bytes;
    const auto allocations_before=g_fuse_base.allocations;
    const auto frees_before=g_fuse_base.frees;
    unsigned long long bad=0,calls=0;
    std::vector<unsigned long long> expected(2*P-1,0);
    for (size_t k=0;k<expected.size();++k)
        for (size_t i=0;i<P;++i) if(k>=i && k-i<P) expected[k]+=a[i*3]*b[(k-i)*3];
    for (int i=0;i<8;++i) {
        NttMulStats st;
        const int rc=ntt_poly_mul_batch_host(P,S,device,1,a.data(),b.data(),&output,&st,&ar,nullptr);
        ++calls;
        if (rc || output!=expected || g_fuse_base.live_bytes!=live_before || !ar.fuses.empty()) ++bad;
    }
    NttMulStats st;
    for (int i=0;i<2;++i) {
        const int rc=ntt_poly_mul_host(P,S,device,false,0,a.data(),b.data(),&output,nullptr,&st,&ar);
        ++calls; if(rc || output!=expected || g_fuse_base.live_bytes!=live_before) ++bad;
    }
    unsigned long long *exported=(unsigned long long *)1;
    int rc=ntt_poly_mul_batch_host(P,S,device,1,a.data(),b.data(),&output,&st,&ar,nullptr,&exported);
    ++calls; if(rc!=3 || exported || g_fuse_base.live_bytes!=live_before) ++bad;
    NttInputHook input;
    input.run=[](void *,const NttShape &,unsigned long long,unsigned long long *,unsigned long long *) { return 7; };
    rc=ntt_poly_mul_batch_dev(P,S,device,1,nullptr,nullptr,&st,&ar,nullptr,nullptr,0,false,&input);
    ++calls; if(rc!=7 || g_fuse_base.live_bytes!=live_before) ++bad;
    ar.cap_bytes=0;
    rc=ntt_poly_mul_batch_host(P,S,device,1,a.data(),b.data(),&output,&st,&ar,nullptr);
    ++calls; if(rc || output!=expected || ar.fuses.size()!=1 || g_fuse_base.live_bytes<=live_before) ++bad;
    const auto cached_allocs=g_fuse_base.allocations, cached_bytes=g_fuse_base.live_bytes;
    rc=ntt_poly_mul_batch_host(P,S,device,1,a.data(),b.data(),&output,&st,&ar,nullptr);
    ++calls; if(rc || output!=expected || g_fuse_base.allocations!=cached_allocs || g_fuse_base.live_bytes!=cached_bytes) ++bad;
    ar.release(); if(g_fuse_base.live_bytes!=live_before) ++bad;
    stage2_log::print(stage2_log::debug, "ntt_fuse_lifetime_check: calls=%llu bad=%llu allocations=%llu frees=%llu leaked_bytes=%llu\n",
                calls,bad,g_fuse_base.allocations-allocations_before,g_fuse_base.frees-frees_before,
                g_fuse_base.live_bytes-live_before);
    if (bad) std::exit(3);
}

void fuse_fixture_env(const char *key,const char *value)
{
#if defined(_WIN32)
    _putenv_s(key,value ? value : "");
#else
    if(value && *value) setenv(key,value,1); else unsetenv(key);
#endif
}

void ntt_fuse_capacity_check(int device)
{
    CK(cudaSetDevice(device));
    const char *et=std::getenv("NTT_FUSE_T"), *em=std::getenv("NTT_FUSE_M");
    const std::string saved_t=et ? et : "", saved_m=em ? em : "";
    const auto live_before=g_fuse_base.live_bytes;
    unsigned long long cases=0,words=0,bad=0;
    const bool extended=fuse_env_ull("NTT_FUSE_WARP_TEST",0)!=0;
    const std::vector<int> ks=extended ? std::vector<int>{4,5,6,7,8,11,12,13,15} : std::vector<int>{7,15,7};
    const std::vector<int> ts=extended ? std::vector<int>{4,5,6,7,8,11,12,12,12} : std::vector<int>{4,8,8};
    const unsigned long long nbatch=extended ? 3 : 1;
    for (size_t shape=0;shape<ks.size();++shape) {
        const int k=ks[shape]; const unsigned long long n=1ull<<k;
        const auto om=gl_pow_host(7ull,(GL_P-1)/n), omi=gl_pow_host(om,GL_P-2);
        const auto nsc=gl_pow_host(n,GL_P-2);
        std::vector<unsigned long long> original(n*nbatch,0),ones(n*nbatch,1),spectrum(n*nbatch,0),got(n*nbatch),expected(n*nbatch);
        original[0]=GL_P-1; original[1]=0x8000000000000000ull; original[n-1]=GL_P-2;
        /* Independent sparse DFT in GMP, stored in DIF bit-reversed order. */
        mpz_t p,w,wi,u,v,sum,tmp,ca,cb,cc;
        mpz_inits(p,w,wi,u,v,sum,tmp,ca,cb,cc,nullptr);
        const unsigned long long prime=GL_P;
        mpz_import(p,1,-1,8,0,0,&prime); mpz_import(w,1,-1,8,0,0,&om);
        mpz_import(wi,1,-1,8,0,0,&omi);
        mpz_import(ca,1,-1,8,0,0,&original[0]); mpz_import(cb,1,-1,8,0,0,&original[1]);
        mpz_import(cc,1,-1,8,0,0,&original[n-1]); mpz_set_ui(u,1); mpz_set_ui(v,1);
        for (unsigned long long j=0;j<n;++j) {
            mpz_mul(sum,cb,u); mpz_add(sum,sum,ca); mpz_mul(tmp,cc,v); mpz_add(sum,sum,tmp); mpz_mod(sum,sum,p);
            unsigned long long r=0,x=j; for(int bit=0;bit<k;++bit){r=(r<<1)|(x&1);x>>=1;}
            size_t count=0; mpz_export(&spectrum[r],&count,-1,8,0,0,sum);
            mpz_mul(u,u,w);mpz_mod(u,u,p);mpz_mul(v,v,wi);mpz_mod(v,v,p);
        }
        /* Slice-specific pointwise multiplier exercises inverse B and batch addressing.
           Forward spectra and inverse results are checked independently, not only roundtrip. */
        for(unsigned long long sl=0;sl<nbatch;++sl) for(unsigned long long j=0;j<n;++j) {
            original[sl*n+j]=original[j]; spectrum[sl*n+j]=spectrum[j]; ones[sl*n+j]=sl+1;
            mpz_import(tmp,1,-1,8,0,0,&original[j]); mpz_mul_ui(tmp,tmp,(unsigned long)(sl+1)); mpz_mod(tmp,tmp,p);
            size_t count=0; mpz_export(&expected[sl*n+j],&count,-1,8,0,0,tmp);
        }
        mpz_clears(p,w,wi,u,v,sum,tmp,ca,cb,cc,nullptr);
        unsigned long long *data=nullptr,*dones=nullptr;
        CK(cudaMalloc(&data,n*nbatch*8));CK(cudaMalloc(&dones,n*nbatch*8));
        CK(cudaMemcpy(dones,ones.data(),n*nbatch*8,cudaMemcpyHostToDevice));
        fuse_fixture_env("NTT_FUSE_T",std::to_string(ts[shape]).c_str());
        for (int m=1;m<=FUSE_MAX_M;++m) for(bool compact:{false,true}) {
            fuse_fixture_env("NTT_FUSE_M",std::to_string(m).c_str());
            FuseCtx fc; fuse_init(fc,n,k,om,omi,compact);
            for (int state=0;state<3;++state) {
                if(state==1) ntt_fuse_cache_tables(fc,om,omi);
                if(state==2) NttArena::fuse_drop_tables(fc);
                CK(cudaMemcpy(data,original.data(),n*nbatch*8,cudaMemcpyHostToDevice));
                ntt_forward_fused(data,fc,om,nbatch);
                CK(cudaMemcpy(got.data(),data,n*nbatch*8,cudaMemcpyDeviceToHost));
                for(size_t j=0;j<n*nbatch;++j){if(got[j]!=spectrum[j]) ++bad;} words+=n*nbatch;
                ntt_inverse_fused(data,dones,fc,omi,nsc,nbatch);
                CK(cudaMemcpy(got.data(),data,n*nbatch*8,cudaMemcpyDeviceToHost));
                for(size_t j=0;j<n*nbatch;++j){if(got[j]!=expected[j]) ++bad;} words+=n*nbatch;
                ++cases;
            }
            if(compact && fc.nms==0 && (fc.scr || fc.scr2 || fc.scrWords || fc.scr2Words)) ++bad;
            fuse_release(fc);
        }
        if(extended && k==12) {
            const std::string saved_w=std::getenv("NTT_FUSE_WARP_TAIL") ? std::getenv("NTT_FUSE_WARP_TAIL") : "";
            unsigned long long switch_bad=0, switch_words=0;
            {
                NttArena ar;
                for(int mode : {0,1,0,1}) {
                    fuse_fixture_env("NTT_FUSE_WARP_TAIL",std::to_string(mode).c_str());
                    fuse_fixture_env("NTT_FUSE_T","12");
                    FuseCtx large; ntt_arena_fuse(&ar,n,k,om,omi,large);
                    const auto om32=gl_pow_host(7ull,(GL_P-1)/32);
                    fuse_fixture_env("NTT_FUSE_T","5");
                    FuseCtx small; ntt_arena_fuse(&ar,32,5,om32,gl_pow_host(om32,GL_P-2),small);
                    /* Pin the cached large plan again: small init changes FUNCTION attributes. */
                    fuse_fixture_env("NTT_FUSE_T","12");
                    ntt_arena_fuse(&ar,n,k,om,omi,large);
                    if(large.warp_tail!=(mode!=0)) ++switch_bad;
                    CK(cudaMemcpy(data,original.data(),n*nbatch*8,cudaMemcpyHostToDevice));
                    ntt_forward_fused(data,large,om,nbatch);
                    CK(cudaMemcpy(got.data(),data,n*nbatch*8,cudaMemcpyDeviceToHost));
                    for(size_t j=0;j<n*nbatch;++j) if(got[j]!=spectrum[j]) ++switch_bad;
                    ntt_inverse_fused(data,dones,large,omi,nsc,nbatch);
                    CK(cudaMemcpy(got.data(),data,n*nbatch*8,cudaMemcpyDeviceToHost));
                    for(size_t j=0;j<n*nbatch;++j) if(got[j]!=expected[j]) ++switch_bad;
                    switch_words+=2*n*nbatch;
                }
            }
            fuse_fixture_env("NTT_FUSE_WARP_TAIL",saved_w.c_str());
            bad+=switch_bad;
            stage2_log::print(stage2_log::debug, "ntt_fuse_warp_switch_check: calls=4 words=%llu bad=%llu (cached 12/5/12, 0/1/0/1)\n",switch_words,switch_bad);
        }
        CK(cudaFree(data));CK(cudaFree(dones));
    }
    fuse_fixture_env("NTT_FUSE_T",saved_t.c_str());fuse_fixture_env("NTT_FUSE_M",saved_m.c_str());
    if(g_fuse_base.live_bytes!=live_before) ++bad;
    stage2_log::print(stage2_log::debug, "ntt_fuse_capacity_check: cases=%llu words=%llu bad=%llu (GMP DFT, inverse, radix 1..%d, cache/evict/tile-only)\n",
                cases,words,bad,FUSE_MAX_M);
    if(extended) {
        const bool warp=fuse_env_ull("NTT_FUSE_WARP_TAIL",0)!=0;
        stage2_log::print(stage2_log::debug, "ntt_fuse_warp_check: requested=%d cases=%llu batch=3 t=4,5,6,7,8,11,12 bad=%llu\n",(int)warp,cases,bad);
        ntt_tile_attributes(warp,32768,true);
        const void *fwd=warp ? ecm_ntt_tile_handle(false,true) : ecm_ntt_tile_handle(false,false);
        const void *inv=warp ? ecm_ntt_tile_handle(true,true) : ecm_ntt_tile_handle(true,false);
        cudaFuncAttributes fa,ia; int fb=0,ib=0;
        CK(cudaFuncGetAttributes(&fa,fwd)); CK(cudaFuncGetAttributes(&ia,inv));
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&fb,fwd,FUSE_TILE_THREADS,32768));
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&ib,inv,FUSE_TILE_THREADS,32768));
        stage2_log::print(stage2_log::debug, "ntt_fuse_warp_resources: requested=%d fwd_regs=%d inv_regs=%d fwd_local=%zu inv_local=%zu fwd_max_blocks=%d inv_max_blocks=%d block=512 dynamic_shared=32768\n",
                    (int)warp,fa.numRegs,ia.numRegs,fa.localSizeBytes,ia.localSizeBytes,fb,ib);
    }
    if(bad) std::exit(3);
}

void ntt_fuse_coop_check(int device)
{
    CK(cudaSetDevice(device));
    const char *et=std::getenv("NTT_FUSE_T"), *em=std::getenv("NTT_FUSE_COOP_M");
    const std::string saved_t=et ? et : "", saved_m=em ? em : "";
    const auto live_before=g_fuse_base.live_bytes;
    unsigned long long cases=0,words=0,bad=0;
    const bool extended=true;
    const std::string saved_coop=std::getenv("NTT_FUSE_COOP_OUTER") ? std::getenv("NTT_FUSE_COOP_OUTER") : "";
    fuse_fixture_env("NTT_FUSE_COOP_OUTER","1");
    const std::vector<int> ks=extended ? std::vector<int>{13,14,15,17} : std::vector<int>{7,15,7};
    const std::vector<int> ts=extended ? std::vector<int>{5,5,5,8} : std::vector<int>{4,8,8};
    const unsigned long long nbatch=extended ? 3 : 1;
    for (size_t shape=0;shape<ks.size();++shape) {
        const int k=ks[shape]; const unsigned long long n=1ull<<k;
        const auto om=gl_pow_host(7ull,(GL_P-1)/n), omi=gl_pow_host(om,GL_P-2);
        const auto nsc=gl_pow_host(n,GL_P-2);
        std::vector<unsigned long long> original(n*nbatch,0),ones(n*nbatch,1),spectrum(n*nbatch,0),got(n*nbatch),expected(n*nbatch);
        original[0]=GL_P-1; original[1]=0x8000000000000000ull; original[n-1]=GL_P-2;
        /* Independent sparse DFT in GMP, stored in DIF bit-reversed order. */
        mpz_t p,w,wi,u,v,sum,tmp,ca,cb,cc;
        mpz_inits(p,w,wi,u,v,sum,tmp,ca,cb,cc,nullptr);
        const unsigned long long prime=GL_P;
        mpz_import(p,1,-1,8,0,0,&prime); mpz_import(w,1,-1,8,0,0,&om);
        mpz_import(wi,1,-1,8,0,0,&omi);
        mpz_import(ca,1,-1,8,0,0,&original[0]); mpz_import(cb,1,-1,8,0,0,&original[1]);
        mpz_import(cc,1,-1,8,0,0,&original[n-1]); mpz_set_ui(u,1); mpz_set_ui(v,1);
        for (unsigned long long j=0;j<n;++j) {
            mpz_mul(sum,cb,u); mpz_add(sum,sum,ca); mpz_mul(tmp,cc,v); mpz_add(sum,sum,tmp); mpz_mod(sum,sum,p);
            unsigned long long r=0,x=j; for(int bit=0;bit<k;++bit){r=(r<<1)|(x&1);x>>=1;}
            size_t count=0; mpz_export(&spectrum[r],&count,-1,8,0,0,sum);
            mpz_mul(u,u,w);mpz_mod(u,u,p);mpz_mul(v,v,wi);mpz_mod(v,v,p);
        }
        /* Slice-specific pointwise multiplier exercises inverse B and batch addressing.
           Forward spectra and inverse results are checked independently, not only roundtrip. */
        for(unsigned long long sl=0;sl<nbatch;++sl) for(unsigned long long j=0;j<n;++j) {
            original[sl*n+j]=original[j]; spectrum[sl*n+j]=spectrum[j]; ones[sl*n+j]=sl+1;
            mpz_import(tmp,1,-1,8,0,0,&original[j]); mpz_mul_ui(tmp,tmp,(unsigned long)(sl+1)); mpz_mod(tmp,tmp,p);
            size_t count=0; mpz_export(&expected[sl*n+j],&count,-1,8,0,0,tmp);
        }
        mpz_clears(p,w,wi,u,v,sum,tmp,ca,cb,cc,nullptr);
        unsigned long long *data=nullptr,*dones=nullptr;
        CK(cudaMalloc(&data,n*nbatch*8));CK(cudaMalloc(&dones,n*nbatch*8));
        CK(cudaMemcpy(dones,ones.data(),n*nbatch*8,cudaMemcpyHostToDevice));
        fuse_fixture_env("NTT_FUSE_T",std::to_string(ts[shape]).c_str());
        for (int m=5;m<=8;++m) for(bool compact:{false,true}) {
            fuse_fixture_env("NTT_FUSE_COOP_M",std::to_string(m).c_str());
            FuseCtx fc; fuse_init(fc,n,k,om,omi,compact);
            for (int state=0;state<3;++state) {
                if(state==1) ntt_fuse_cache_tables(fc,om,omi);
                if(state==2) NttArena::fuse_drop_tables(fc);
                CK(cudaMemcpy(data,original.data(),n*nbatch*8,cudaMemcpyHostToDevice));
                ntt_forward_fused(data,fc,om,nbatch);
                CK(cudaMemcpy(got.data(),data,n*nbatch*8,cudaMemcpyDeviceToHost));
                if(fuse_env_ull("NTT_FUSE_COOP_BAD",0) && shape==0 && m==5 && !compact && state==0)got[0]^=1;
                for(size_t j=0;j<n*nbatch;++j){if(got[j]!=spectrum[j]) ++bad;} words+=n*nbatch;
                ntt_inverse_fused(data,dones,fc,omi,nsc,nbatch);
                CK(cudaMemcpy(got.data(),data,n*nbatch*8,cudaMemcpyDeviceToHost));
                for(size_t j=0;j<n*nbatch;++j){if(got[j]!=expected[j]) ++bad;} words+=n*nbatch;
                ++cases;
            }
            if(compact && fc.nms==0 && (fc.scr || fc.scr2 || fc.scrWords || fc.scr2Words)) ++bad;
            fuse_release(fc);
        }
        if(extended && k==17) {
            const std::string saved_w=std::getenv("NTT_FUSE_WARP_TAIL") ? std::getenv("NTT_FUSE_WARP_TAIL") : "";
            unsigned long long switch_bad=0, switch_words=0;
            {
                NttArena ar;
                for(int mode : {0,1,0,1}) {
                    fuse_fixture_env("NTT_FUSE_COOP_OUTER",std::to_string(mode).c_str());
                    fuse_fixture_env("NTT_FUSE_T","5");
                    FuseCtx large; ntt_arena_fuse(&ar,n,k,om,omi,large);
                    const auto om32=gl_pow_host(7ull,(GL_P-1)/32);
                    fuse_fixture_env("NTT_FUSE_T","5");
                    FuseCtx small; ntt_arena_fuse(&ar,32,5,om32,gl_pow_host(om32,GL_P-2),small);
                    /* Pin the cached large plan again: small init changes FUNCTION attributes. */
                    fuse_fixture_env("NTT_FUSE_T","5");
                    ntt_arena_fuse(&ar,n,k,om,omi,large);
                    if(large.coop_outer!=(mode!=0)) ++switch_bad;
                    CK(cudaMemcpy(data,original.data(),n*nbatch*8,cudaMemcpyHostToDevice));
                    ntt_forward_fused(data,large,om,nbatch);
                    CK(cudaMemcpy(got.data(),data,n*nbatch*8,cudaMemcpyDeviceToHost));
                    for(size_t j=0;j<n*nbatch;++j) if(got[j]!=spectrum[j]) ++switch_bad;
                    ntt_inverse_fused(data,dones,large,omi,nsc,nbatch);
                    CK(cudaMemcpy(got.data(),data,n*nbatch*8,cudaMemcpyDeviceToHost));
                    for(size_t j=0;j<n*nbatch;++j) if(got[j]!=expected[j]) ++switch_bad;
                    switch_words+=2*n*nbatch;
                }
            }
            fuse_fixture_env("NTT_FUSE_WARP_TAIL",saved_w.c_str());
            bad+=switch_bad;
            stage2_log::print(stage2_log::debug, "ntt_fuse_coop_switch_check: calls=4 words=%llu bad=%llu (cached mode 0/1/0/1)\n",switch_words,switch_bad);
        }
        CK(cudaFree(data));CK(cudaFree(dones));
    }
    fuse_fixture_env("NTT_FUSE_T",saved_t.c_str());fuse_fixture_env("NTT_FUSE_COOP_M",saved_m.c_str());
    if(g_fuse_base.live_bytes!=live_before) ++bad;
    stage2_log::print(stage2_log::debug, "ntt_fuse_coop_check: cases=%llu words=%llu bad=%llu (GMP DFT, inverse, radix 5..%d, cache/evict)\n",
                cases,words,bad,8);
    fuse_fixture_env("NTT_FUSE_COOP_OUTER",saved_coop.c_str());
    if(bad) std::exit(3);
}
