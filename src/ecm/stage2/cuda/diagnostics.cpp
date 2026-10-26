// Optional fixtures, GMP comparisons, dumps, fault injection and crash reporting.
#include "diagnostics.h"
#include "product.h"
#include "points.h"
#include "tree.h"
#include "remainder.h"
#include "factors.h"
#include "options.h"
#include "checks.h"
#include "point_launch.h"
#include "../planning/giant_chunk.h"
#include "stage2_baby_launch.h"
#include "scale_plain.cuh"

#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")     /* GetProcessMemoryInfo: host commit, for the last-state line */
static LONG WINAPI s2g_unhandled_filter(EXCEPTION_POINTERS *ep)
{
    const EXCEPTION_RECORD *r = ep->ExceptionRecord;
    const void *rip = nullptr;
#if defined(_M_X64) || defined(_M_AMD64)
    rip = (const void *)ep->ContextRecord->Rip;
#endif
    std::fprintf(stderr, "CRASH: code=0x%08lX fault_addr=%p rip=%p (host exception; the "
                         "exception code and the address above are the diagnosis)\n",
                 (unsigned long)r->ExceptionCode, r->ExceptionAddress, rip);
    std::fflush(stderr);
    return EXCEPTION_CONTINUE_SEARCH;
}
void s2g_install_crash_handler(void)
{
    SetUnhandledExceptionFilter(s2g_unhandled_filter);
}
/* host commit charge + device free bytes, one line; also the "last known state" the watchdog
   leaves behind when the driver's watchdog kills us (see the header comment) */
char g_last_state[512] = "none";
void s2g_state(const char *what)
{
    PROCESS_MEMORY_COUNTERS pmc{};
    pmc.cb = sizeof(pmc);
    GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc));
    size_t dfree = 0, dtotal = 0;
    cudaMemGetInfo(&dfree, &dtotal);
    std::snprintf(g_last_state, sizeof(g_last_state),
                  "%s | host private=%.0f MB peak=%.0f MB | device free=%.0f MB of %.0f MB",
                  what, (double)pmc.PagefileUsage / 1048576.0,
                  (double)pmc.PeakPagefileUsage / 1048576.0, (double)dfree / 1048576.0,
                  (double)dtotal / 1048576.0);
}
void s2g_print_last_state(const char *why)
{
    std::fprintf(stderr, "LAST_STATE (%s): %s\n", why, g_last_state);
    std::fflush(stderr);
}
/* every death that is not an exception ends up here: an escaping exception, a host bad_alloc,
   a throwing destructor.  Prints the type and what() before terminating. */
static void s2g_terminate_handler(void)
{
    std::fprintf(stderr, "TERMINATE: %s\n", g_last_state);
    if (std::current_exception()) {
        try { std::rethrow_exception(std::current_exception()); }
        catch (const std::exception &e) {
            std::fprintf(stderr, "TERMINATE: exception: %s\n", e.what());
        }
        catch (...) { std::fprintf(stderr, "TERMINATE: non-std exception\n"); }
    } else {
        std::fprintf(stderr, "TERMINATE: no active exception (pure virtual call, or terminate() "
                             "called directly)\n");
    }
    std::fflush(stderr);
    std::abort();
}
void s2g_install_terminate(void)
{ std::set_terminate(s2g_terminate_handler); }
#else
void s2g_install_crash_handler(void)
{}
void s2g_install_terminate(void)
{}
void s2g_state(const char *)
{}
void s2g_print_last_state(const char *)
{}
#endif

bool write_f_dump(
    const char *path,
    const mpz_t N,
    const mpz_t a24,
    const mpz_t Q,
    unsigned long long D,
    unsigned long long B1,
    unsigned long long B2,
    unsigned long long sigma,
    const std::vector<unsigned long long> &bj,
    const std::vector<std::string> &bx,
    const std::vector<std::string> &F,
    unsigned long long degree)
{
    FILE *f = std::fopen(path, "wb");
    if (!f) return false;
    char *nh = mpz_get_str(nullptr, 16, N);
    char *nd = mpz_get_str(nullptr, 10, N);
    char *ah = mpz_get_str(nullptr, 16, a24);
    char *qh = mpz_get_str(nullptr, 16, Q);
    std::fprintf(f, "stage2_tree_F_dump v1\n");
    std::fprintf(f, "N_dec %s\n", nd);
    std::fprintf(f, "N_hex %s\n", nh);
    std::fprintf(f, "N_bits %ld\n", (long)mpz_sizeinbase(N, 2));
    std::fprintf(f, "D %llu\nB1 %llu\nB2 %llu\nsigma %llu\n",
                 (unsigned long long)D, (unsigned long long)B1, (unsigned long long)B2,
                 (unsigned long long)sigma);
    std::fprintf(f, "a24_hex %s\n", ah);
    std::fprintf(f, "Q_hex %s\n", qh);
    std::fprintf(f, "baby_count %llu\n", (unsigned long long)bj.size());
    for (size_t i = 0; i < bj.size(); ++i)
        std::fprintf(f, "baby %llu %s\n", (unsigned long long)bj[i], bx[i].c_str());
    std::fprintf(f, "F_degree %llu\n", (unsigned long long)degree);
    for (const std::string &c : F) std::fprintf(f, "F %s\n", c.c_str());
    std::fprintf(f, "end\n");
    std::fclose(f);
    void (*freefunc)(void *, size_t) = nullptr;
    mp_get_memory_functions(nullptr, nullptr, &freefunc);
    freefunc(nh, std::strlen(nh) + 1);
    freefunc(nd, std::strlen(nd) + 1);
    freefunc(ah, std::strlen(ah) + 1);
    freefunc(qh, std::strlen(qh) + 1);
    return true;
}

void gfinv_batch_fixture(const mpz_t actual,size_t actualW)
{
    unsigned long long cases=0,checks=0,nonunits=0;
    GfinvStats stats;mpz_t N,value,want,got,test;mpz_inits(N,value,want,got,test,nullptr);
    const size_t sizes[]={0,1,15,16,17,63,64,65,127,129};
    for(int modulus=0;modulus<3;++modulus) {
        if(!modulus)mpz_set(N,actual);else mpz_set_ui(N,modulus==1?15:35);
        const size_t W=modulus?1:actualW;
        for(size_t size:sizes)for(int pattern=0;pattern<3;++pattern) {
            std::vector<unsigned long long> words(size*W),tmp;
            for(size_t j=0;j<size;++j) {
                mpz_set_ui(value,pattern?2+37*j:1);
                if(pattern==2 && (j==0 || j==63 || j==64 || j+1==size))mpz_set_ui(value,0);
                mpz_mod(value,value,N);mpz_to_words(tmp,W,value);
                std::copy(tmp.begin(),tmp.end(),words.begin()+j*W);
            }
            GfinvBatch cache(words,W,N,true,stats);++cases;
            for(size_t j=0;j<size;++j)for(int repeat=0;repeat<(j%17==0?2:1);++repeat) {
                words_to_mpz(value,words.data()+j*W,W);
                const bool expected=mpz_invert(want,value,N)!=0,unit=cache.get(got,j);++checks;
                if(!expected)++nonunits;
                if(unit!=expected || (unit && mpz_cmp(want,got)) || (!unit && mpz_sgn(got))) {
                    std::fprintf(stderr,"%s: FATAL: segment inverse GMP mismatch modulus=%d pattern=%d size=%llu index=%llu\n",
                        NTT_PROBE_NAME,modulus,pattern,(unsigned long long)size,(unsigned long long)j);std::exit(3);
                }
                if(unit) {
                    mpz_mul(test,value,got);mpz_mod(test,test,N);
                    if(mpz_cmp_ui(test,1)) {std::fprintf(stderr,"%s: FATAL: segment inverse identity mismatch\n",NTT_PROBE_NAME);std::exit(3);}
                }
            }
        }
    }
    stage2_log::print(stage2_log::debug, "gfinv_fixture: cases=%llu checks=%llu nonunits=%llu group_failures=%llu cache_hits=%llu scratch_peak_bytes=%llu bad=0\n",
        cases,checks,nonunits,stats.group_failures,stats.cache_hits,stats.scratch_peak_bytes);
    mpz_clears(N,value,want,got,test,nullptr);
}

void groot_device_fixture(PolyLayer &L)
{
    const size_t W=L.W;unsigned long long cases=0,words=0;
    const auto nodes0=g_gdevice.checked_nodes,words0=g_gdevice.checked_words;
    mpz_t c,z;mpz_inits(c,z,nullptr);
    for(size_t n:{0u,1u,2u,3u,5u,7u,8u,9u,17u,23u,257u,511u})for(size_t mode=0;mode<4;++mode) {
        std::vector<std::vector<unsigned long long>> leaf;
        std::vector<unsigned long long> expected(W,0),word(W);expected[0]=1;
        for(size_t i=0;i<n;++i) {
            if(mode==0){mpz_set_ui(c,1);mpz_set_ui(z,1);}
            else if(mode==1){mpz_set_ui(c,13+i*17);mpz_set_ui(z,2+i*7);}
            else if(mode==2){mpz_sub_ui(c,L.N,1);mpz_sub_ui(z,L.N,1);}
            else {mpz_set_ui(c,i%3==0?0:i%3==1?1:i);mpz_set_ui(z,i%3==2?i+1:0);}
            mpz_mod(c,c,L.N);mpz_mod(z,z,L.N);
            std::vector<unsigned long long> f(2*W,0);
            mpz_to_words(word,W,c);std::copy(word.begin(),word.end(),f.begin());
            mpz_to_words(word,W,z);std::copy(word.begin(),word.end(),f.begin()+W);
            leaf.push_back(f);expected=groot_product_gmp(L,expected,f);
        }
        FTreeStats fs;std::vector<size_t> degree;size_t pad=0;
        auto tree=build_groot_select(L,leaf,degree,pad,fs,-1,false,{},true);
        if(tree[1]!=expected || degree[1]!=n) {std::fprintf(stderr,"%s: FATAL: resident fixture root mismatch n=%llu\n",NTT_PROBE_NAME,(unsigned long long)n);std::exit(3);}
        for(size_t i=2;i<tree.size();++i)if(!tree[i].empty())std::exit(3);
        ++cases;words+=expected.size();
    }
    mpz_clears(c,z,nullptr);
    stage2_log::print(stage2_log::debug, "gdevice_fixture: cases=%llu words=%llu checked_nodes=%llu checked_words=%llu bad=0 (GMP nonmonic, zero/constant-one, empty/single/padded/large trees)\n",
        cases,words,g_gdevice.checked_nodes-nodes0,g_gdevice.checked_words-words0);
}

void descent_slow(
    PolyLayer &L,
    const std::vector<std::vector<unsigned long long>> &Ft,
    const std::vector<size_t> &Fdeg,
    size_t Fpad,
    const CPoly &H,
    std::vector<std::vector<unsigned long long>> &values,
    unsigned long long &divmods)
{
    const size_t W = L.W;
    /* the leaf count is the F tree's degree, not H's coefficient count (section 54.5): H mod F
       has deg < deg F, so H may be SHORTER than the leaf count -- and at the first batch, where
       H = T with no reduction, it usually is.  Reading `cur[i]` for i in [H.size(), P) is exactly
       the right thing: the tree walk has already reduced H (implicitly zero-extended) down to
       every leaf, so those rows are H(x_i), not zeros. */
    const size_t P = (Fdeg.size() > 1 && Fdeg[1] > 0) ? Fdeg[1] : H.size();
    std::vector<CPoly> cur(1);
    cur[0] = H;
    size_t base = 1, cnt = 1;
    while (base < Fpad) {
        const size_t nbase = base * 2;
        std::vector<CPoly> nxt(cnt * 2);
        for (size_t j = 0; j < cnt; ++j) {
            for (int sgn = 0; sgn < 2; ++sgn) {
                const size_t ci = nbase + 2 * j + (size_t)sgn;
                if (Fdeg[ci] == 0 && poly_is_one(Ft[ci].data(), W)) {
                    cp_resize(nxt[2 * j + (size_t)sgn], 1, W);      /* H mod 1 = 0 */
                } else {
                    if (cur[j].size() >= Fdeg[ci] + 1) ++divmods;
                    nxt[2 * j + (size_t)sgn] =
                        cp_mod(cur[j], cp_from_flat(Ft[ci], Fdeg[ci], W), L);
                }
            }
        }
        cur.swap(nxt);
        base = nbase;
        cnt *= 2;
        if (g_s4_descent_trace) {
            unsigned long long hh = 1469598103934665603ull;
            for (size_t j = 0; j < cnt; ++j)
                for (const std::vector<unsigned long long> &c : cur[j])
                    for (unsigned long long v : c) hh = (hh ^ v) * 1099511628211ull;
            stage2_log::print(stage2_log::debug, "descent_trace: slow base=%llu nodes=%llu hash=%llu\n",
                        (unsigned long long)base, (unsigned long long)cnt, hh);
        }
    }
    values.assign(P, std::vector<unsigned long long>(W, 0ull));
    for (size_t i = 0; i < P; ++i) values[i] = cp_to_flat(cur[i], W);
}

std::vector<unsigned long long> fold_gmp_reference(
    PolyLayer &L,
    const std::vector<unsigned long long> &G,
    const std::vector<unsigned long long> &H,
    const std::vector<unsigned long long> &F)
{
    const size_t W=L.W,P=F.size()/W-1;
    auto out=groot_product_gmp(L,G,H);
    if(out.size()/W<=P)return out;
    mpz_t a,b,c;mpz_inits(a,b,c,nullptr);std::vector<unsigned long long> word(W);
    for(size_t i=out.size()/W;i-->P;) {
        words_to_mpz(c,out.data()+i*W,W);
        for(size_t j=0;j<P;++j) {
            words_to_mpz(a,out.data()+(i-P+j)*W,W);words_to_mpz(b,F.data()+j*W,W);
            mpz_submul(a,c,b);mpz_mod(a,a,L.N);mpz_to_words(word,W,a);
            std::copy_n(word.data(),W,out.data()+(i-P+j)*W);
        }
        std::fill_n(out.data()+i*W,W,0ull);
    }
    mpz_clears(a,b,c,nullptr);out.resize(P*W);
    while(out.size()>W && std::all_of(out.end()-W,out.end(),[](unsigned long long x){return x==0;}))out.resize(out.size()-W);
    return out;
}

void fold_flat_fixture(PolyLayer &L)
{
    const size_t W=L.W;unsigned long long cases=0,words=0;
    mpz_t z;mpz_init(z);std::vector<unsigned long long> word(W);
    for(size_t P:{1u,2u,3u,8u,17u}) {
        std::vector<unsigned long long> F(W,0);F[0]=1;
        for(size_t j=0;j<P;++j) {
            mpz_set_ui(z,j+2);mpz_neg(z,z);mpz_mod(z,z,L.N);mpz_to_words(word,W,z);
            std::vector<unsigned long long> leaf(2*W);std::copy_n(word.data(),W,leaf.data());leaf[W]=1;
            F=groot_product_gmp(L,F,leaf);
        }
        CPoly rev=cp_from_flat(F,P,W);std::reverse(rev.begin(),rev.end());
        auto inverse=cp_to_flat(cp_inv_series(rev,P+1,L),W);
        for(size_t mode=0;mode<4;++mode) {
            const size_t nh=mode==1?1:mode==2?P:P+1,ng=mode==1?1:P+1;
            auto make=[&](size_t n,bool zero){
                std::vector<unsigned long long> v(n*W);
                for(size_t j=0;j<n;++j) {
                    if(zero)mpz_set_ui(z,0);
                    else if(mode==2)mpz_sub_ui(z,L.N,j%3+1);
                    else {mpz_set_ui(z,17*j+3);mpz_mul_2exp(z,z,(unsigned long)((j*31)%L.S));}
                    mpz_mod(z,z,L.N);mpz_to_words(word,W,z);std::copy_n(word.data(),W,v.data()+j*W);
                }return v;
            };
            auto H=make(nh,mode==0),G=make(ng,false);
            FoldFlatStats stats;
            // mode 3 exercises repeated folds and a shorter final G with the same inverse.
            for(size_t round=0;round<(mode==3?3u:1u);++round) {
                if(round==2)G.resize(std::min(G.size(),2*W));
                auto expected=fold_gmp_reference(L,G,H,F);
                fold_flat_step(L,G,H,F,inverse,stats);
                if(H!=expected) {std::fprintf(stderr,"%s: FATAL: flat fold GMP mismatch P=%llu mode=%llu round=%llu\n",NTT_PROBE_NAME,(unsigned long long)P,(unsigned long long)mode,(unsigned long long)round);std::exit(3);}
                ++cases;words+=H.size();
            }
        }
    }
    mpz_clear(z);
    stage2_log::print(stage2_log::debug, "fold_flat_fixture: cases=%llu words=%llu bad=0 (GMP schoolbook/monic long division, zero/near-N/short/padded/repeated folds)\n",cases,words);
}

void fold_device_fixture(PolyLayer &L)
{
    const size_t W=L.W;unsigned long long cases=0,words=0;
    mpz_t z;mpz_init(z);std::vector<unsigned long long> word(W);
    for(size_t P:{1u,2u,3u,8u,17u}) {
        std::vector<unsigned long long> F(W,0);F[0]=1;
        for(size_t j=0;j<P;++j) {
            mpz_set_ui(z,j+2);mpz_neg(z,z);mpz_mod(z,z,L.N);mpz_to_words(word,W,z);
            std::vector<unsigned long long> leaf(2*W);std::copy_n(word.data(),W,leaf.data());leaf[W]=1;
            F=groot_product_gmp(L,F,leaf);
        }
        CPoly rev=cp_from_flat(F,P,W);std::reverse(rev.begin(),rev.end());
        auto inverse=cp_to_flat(cp_inv_series(rev,P+1,L),W);
        for(size_t mode=0;mode<4;++mode) {
            const size_t nh=mode==1?1:mode==2?P:P+1,ng=mode==1?1:P+1;
            auto make=[&](size_t n,bool zero){
                std::vector<unsigned long long> v(n*W);
                for(size_t j=0;j<n;++j) {
                    if(zero)mpz_set_ui(z,0);
                    else if(mode==2)mpz_sub_ui(z,L.N,j%3+1);
                    else {mpz_set_ui(z,17*j+3);mpz_mul_2exp(z,z,(unsigned long)((j*31)%L.S));}
                    mpz_mod(z,z,L.N);mpz_to_words(word,W,z);std::copy_n(word.data(),W,v.data()+j*W);
                }return v;
            };
            auto H=make(nh,mode==0),G=make(ng,false);
            FoldFlatStats stats;FoldDeviceStats fd;FoldDeviceState state;
            if(!state.init(L,F,inverse,fd,stats)){std::fprintf(stderr,"FATAL: device fold fixture fell back: %s\n",fd.fallback.c_str());std::exit(3);}
            state.seed(H);unsigned long long digest_words=0,expected_digest[2]={0,0};
            groot_input_digest_host(H,digest_words,expected_digest);
            // mode 3 exercises repeated folds and a shorter final G with the same inverse.
            for(size_t round=0;round<(mode==3?3u:1u);++round) {
                if(round==2)G.resize(std::min(G.size(),2*W));
                auto expected=fold_gmp_reference(L,G,H,F);
                groot_input_digest_host(G,digest_words,expected_digest);
                state.step(G);H=state.read(state.h,state.hcount);
                if(H!=expected) {std::fprintf(stderr,"%s: FATAL: flat fold GMP mismatch P=%llu mode=%llu round=%llu\n",NTT_PROBE_NAME,(unsigned long long)P,(unsigned long long)mode,(unsigned long long)round);std::exit(3);}
                ++cases;words+=H.size();
            }
            state.read_digest();
            if(fd.root_digest_words!=digest_words || fd.root_digest[0]!=expected_digest[0] || fd.root_digest[1]!=expected_digest[1]) {
                std::fprintf(stderr,"FATAL: device root input digest mismatch\n");std::exit(3);
            }
        }
    }
    mpz_clear(z);
    stage2_log::print(stage2_log::debug, "fold_device_fixture: cases=%llu words=%llu bad=0 (GMP schoolbook/monic long division, zero/near-N/short/padded/repeated folds)\n",cases,words);
}

std::vector<unsigned long long> scaled_state_gmp(
    const CPoly &H,
    const std::vector<unsigned long long> &F,
    size_t d,
    PolyLayer &L)
{
    const size_t W=L.W;
    if(F.size()!=(d+1)*W || !poly_is_one(F.data()+d*W,W)) {
        std::fprintf(stderr,"%s: FATAL: scaled oracle needs a monic node\n",NTT_PROBE_NAME);
        std::exit(3);
    }
    if(!d) return {};
    CPoly r=H;
    if(r.size()<d) cp_resize(r,d,W);
    mpz_t q,f,t,u;
    mpz_inits(q,f,t,u,nullptr);
    for(size_t i=r.size();i-- >d;) {
        words_to_mpz(q,r[i].data(),W);
        for(size_t j=0;j<=d;++j) {
            words_to_mpz(f,F.data()+j*W,W);
            words_to_mpz(t,r[i-d+j].data(),W);
            mpz_submul(t,q,f);mpz_mod(t,t,L.N);
            mpz_to_words(r[i-d+j],W,t);
        }
    }
    std::vector<unsigned long long> state(d*W,0);
    std::vector<unsigned long long> word(W,0);
    for(size_t i=0;i<d;++i) {
        words_to_mpz(t,r[d-1-i].data(),W);
        for(size_t j=1;j<=i;++j) {
            words_to_mpz(f,F.data()+(d-j)*W,W);
            words_to_mpz(u,state.data()+(i-j)*W,W);
            mpz_submul(t,f,u);
        }
        mpz_mod(t,t,L.N);mpz_to_words(word,W,t);
        std::copy(word.begin(),word.end(),state.begin()+i*W);
    }
    mpz_clears(q,f,t,u,nullptr);
    return state;
}

void scaled_descent_fixture(PolyLayer &L)
{
    if(!L.s4) {std::fprintf(stderr,"%s: FATAL: scaled fixture requires S4\n",NTT_PROBE_NAME);std::exit(3);}
    const size_t W=L.W;
    ScaledStats sum;
    unsigned long long cases=0;
    mpz_t x;mpz_init(x);
    for(size_t P:{1u,2u,3u,5u,7u,8u,9u,13u,16u,23u}) for(size_t padding=0;padding<3;++padding) {
        size_t pad=1;while(pad<P) pad*=2;
        std::vector<std::vector<unsigned long long>> Ft(2*pad);
        std::vector<size_t> deg(2*pad,0);
        for(size_t i=0;i<pad;++i) {Ft[pad+i].assign(W,0);Ft[pad+i][0]=1;}
        for(size_t i=0;i<P;++i) {
            const size_t slot=padding==0?i:padding==1?pad-P+i:(i*5)%pad;
            auto &f=Ft[pad+slot];f.assign(2*W,0);
            mpz_set_ui(x,1+i*17);mpz_neg(x,x);mpz_mod(x,x,L.N);
            std::vector<unsigned long long> word(W,0);mpz_to_words(word,W,x);
            std::copy(word.begin(),word.end(),f.begin());f[W]=1;deg[pad+slot]=1;
        }
        /* Build the fixture tree with independent GMP convolution. */
        mpz_t a,b,t;mpz_inits(a,b,t,nullptr);
        for(size_t node=pad;--node;) {
            const size_t dl=deg[2*node],dr=deg[2*node+1],d=dl+dr;
            deg[node]=d;Ft[node].assign((d+1)*W,0);
            for(size_t i=0;i<=d;++i) {
                mpz_set_ui(t,0);
                for(size_t j=0;j<=dl;++j) if(i>=j && i-j<=dr) {
                    words_to_mpz(a,Ft[2*node].data()+j*W,W);
                    words_to_mpz(b,Ft[2*node+1].data()+(i-j)*W,W);mpz_addmul(t,a,b);
                }
                mpz_mod(t,t,L.N);std::vector<unsigned long long> word(W,0);mpz_to_words(word,W,t);
                std::copy(word.begin(),word.end(),Ft[node].begin()+i*W);
            }
        }
        CPoly rev;cp_resize(rev,P+1,W);
        for(size_t i=0;i<=P;++i) std::copy_n(Ft[1].data()+(P-i)*W,W,rev[i].begin());
        const CPoly inv=cp_inv_series(rev,P+1,L);
        for(size_t kind=0;kind<5;++kind) {
            CPoly H;cp_resize(H,kind==4?P+1:P,W);
            for(size_t i=0;i<H.size();++i) {
                if(kind==0) mpz_set_ui(x,0);
                else if(kind==1) mpz_set_ui(x,i==0?1:0);
                else if(kind==2) mpz_sub_ui(x,L.N,1);
                else {mpz_set_ui(x,13+i*191);mpz_mod(x,x,L.N);}
                mpz_to_words(H[i],W,x);
            }
            if(kind==1) cp_trim(H);
            std::vector<std::vector<unsigned long long>> values;
            descent_scaled(L,Ft,deg,pad,H,kind==0?nullptr:&inv,values,sum,-1,true);
            {
                FoldDeviceState owner;FoldDeviceStats device_stats;FoldFlatStats flat_stats;
                const auto inverse=cp_to_flat(inv,W);
                if(!owner.init(L,Ft[1],inverse,device_stats,flat_stats)) {
                    std::fprintf(stderr,"FATAL: scaled frontier fixture owner unavailable\n");std::exit(3);
                }
                CPoly reduced=H;
                if(reduced.size()>P)reduced=cp_mod(reduced,cp_from_flat(Ft[1],P,W),L);
                owner.seed(cp_to_flat(reduced,W));
                ScaledFrontierDevice frontier;frontier.requested=true;
                if(!frontier.init(owner)){std::fprintf(stderr,"FATAL: scaled frontier fixture lease unavailable\n");std::exit(3);}
                ScaledRootDeviceStats root_stats;std::vector<unsigned long long> root;
                owner.scaled_root(root,root_stats,true,false,true);
                ScaledStats checked;std::vector<std::vector<unsigned long long>> actual;
                frontier.run(L,Ft,deg,pad,H,actual,checked,-1,true);
                if(actual!=values){std::fprintf(stderr,"FATAL: scaled frontier fixture leaves disagree\n");std::exit(3);}
            }
            /* Each degree-one node's independent remainder is also Horner(H,x). */
            size_t leaf=0;
            for(size_t i=0;i<pad;++i) if(deg[pad+i]) {
                words_to_mpz(a,Ft[pad+i].data(),W);mpz_neg(a,a);mpz_mod(a,a,L.N);mpz_set_ui(t,0);
                for(size_t k=H.size();k--;) {mpz_mul(t,t,a);words_to_mpz(b,H[k].data(),W);mpz_add(t,t,b);mpz_mod(t,t,L.N);}
                words_to_mpz(b,values[leaf++].data(),W);
                if(mpz_cmp(t,b)) {std::fprintf(stderr,"%s: FATAL: scaled Horner mismatch\n",NTT_PROBE_NAME);std::exit(3);}
            }
            ++cases;
        }
        mpz_clears(a,b,t,nullptr);
    }
    mpz_clear(x);
    stage2_log::print(stage2_log::debug, "scaled_fixture: cases=%llu states=%llu words=%llu leaves=%llu reused=%llu root_divisions=%llu bad=0\n",cases,sum.checked_states,sum.checked_words,sum.leaves,sum.root_inverse_reused,sum.root_divisions);
    stage2_log::print(stage2_log::debug, "scaled_frontier_fixture: cases=%llu bad=0\n",cases);
}

void s4_flat_input_check(PolyLayer &L,size_t slices)
{
    /* Regression for a unit carry crossing many warps/blocks.  Compare every digit,
       not just the final modular remainder, and keep batched slices independent. */
    const size_t carry_n = 8205, carry_nb = 3;
    unsigned long long carry_bad = 0;
    for (const int bpw : {14, 26}) {
        const unsigned long long mask = (1ull << bpw) - 1;
        std::vector<unsigned long long> raw(carry_n*carry_nb), expected(raw.size()), actual(raw.size());
        for (size_t s=0; s<carry_nb; ++s) {
            const size_t start = s==0 ? 0 : s==1 ? 31 : 255;
            for (size_t j=start; j<start+4097; ++j) raw[s*carry_n+j] = mask;
            raw[s*carry_n+start] = 1ull << 60;
            /* A second generator after a kill and a partial warp at the array end. */
            raw[s*carry_n+7001] = mask+1;
            for (size_t j=7002; j<carry_n-1; ++j) raw[s*carry_n+j] = mask;
            unsigned long long carry = 0;
            for (size_t j=0; j<carry_n; ++j) {
                const auto v = raw[s*carry_n+j] + carry;
                expected[s*carry_n+j] = v & mask;
                carry = v >> bpw;
            }
        }
        unsigned long long *di = nullptr, *doo = nullptr;
        CK(cudaMalloc(&di, raw.size()*8)); CK(cudaMalloc(&doo, raw.size()*8));
        CK(cudaMemcpy(di, raw.data(), raw.size()*8, cudaMemcpyHostToDevice));
        ecm_launch_carry_cone_kernel<5>(EcmCudaLaunch{dim3((unsigned int)((carry_n+255)/256), (unsigned int)carry_nb), 256},
            di, doo, carry_n, bpw, carry_n);
        CK(cudaGetLastError());
        CK(cudaMemcpy(actual.data(), doo, actual.size()*8, cudaMemcpyDeviceToHost));
        CK(cudaFree(di)); CK(cudaFree(doo));
        for (size_t i=0; i<actual.size(); ++i) if (actual[i] != expected[i]) ++carry_bad;
    }
    stage2_log::print(stage2_log::debug, "carry_chain_check: cases=6 words=49230 bad=%llu (14/26-bit radix, 4097-digit chains)\n", carry_bad);
    if (carry_bad) std::exit(3);
    unsigned long long cases = 0, words = 0, bad = 0;
    mpz_t av, bv, sum, term;
    mpz_inits(av, bv, sum, term, nullptr);
    const size_t W = L.W;
    for (const auto dims : std::initializer_list<std::pair<size_t,size_t>>{{3,3}, {2,5}, {5,2}, {1,4}, {4,1}}) {
        const size_t ma = dims.first, mb = dims.second, nb = slices, nc = ma + mb - 1;
        std::vector<unsigned long long> a(nb*ma*W), b(nb*mb*W), expected(nb*nc*W);
        auto fill = [&](std::vector<unsigned long long> &v) {
            for (size_t i=0; i<v.size()/W; ++i) {
                mpz_sub_ui(av, L.N, (unsigned long)(i+1));
                mpz_mod(av, av, L.N);
                std::vector<unsigned long long> coef(W);
                mpz_to_words(coef, W, av);
                std::copy(coef.begin(), coef.end(), v.begin() + (long)(i*W));
            }
        };
        fill(a); fill(b);
        for (size_t s=0; s<nb; ++s) for (size_t k=0; k<nc; ++k) {
            mpz_set_ui(sum, 0);
            for (size_t i=0; i<ma; ++i) if (k>=i && k-i<mb) {
                words_to_mpz(av, &a[(s*ma+i)*W], W);
                words_to_mpz(bv, &b[(s*mb+k-i)*W], W);
                mpz_mul(term, av, bv); mpz_add(sum, sum, term);
            }
            mpz_mod(sum, sum, L.N);
            std::vector<unsigned long long> coef(W);
            mpz_to_words(coef, W, sum);
            std::copy(coef.begin(), coef.end(), expected.begin() + (long)((s*nc+k)*W));
        }
        for (int alias=0; alias<4; ++alias) {
            if (alias==3 && ma!=mb) continue;
            auto aa=a, bb=b;
            std::vector<unsigned long long> separate;
            auto &result = alias==1 || alias==3 ? aa : alias==2 ? bb : separate;
            flat_mul_batch(L, aa, ma, alias==3 ? aa : bb, mb, nb, result, -1);
            ++cases; words += result.size();
            if (result != expected) ++bad;
        }
    }
    mpz_clears(av, bv, sum, term, nullptr);
    stage2_log::print(stage2_log::debug, "%s: cases=%llu words=%llu bad=%llu (GMP, short/full strides, A/B/both aliases)\n",
                slices==3 ? "s4_flat_input_check" : "s4_final_readback_check", cases, words, bad);
    if (bad) std::exit(3);
}

void s4_output_window_check(PolyLayer &L)
{
    const size_t W=L.W, nb=5;
    unsigned long long cases=0,words=0,bad=0,canonical_cases=0;
    mpz_t av,bv,term,sum,inv;
    mpz_inits(av,bv,term,sum,inv,nullptr);
    for(const auto dims : std::initializer_list<std::pair<size_t,size_t>>{{3,3},{2,5},{5,2},{1,4},{4,1},{1,1}}) {
        const size_t ma=dims.first,mb=dims.second,nc=ma+mb-1;
        std::vector<unsigned long long> a(nb*ma*W),b(nb*mb*W);
        for(int operand=0;operand<2;++operand) {
            auto &v=operand ? b : a;
            for(size_t i=0;i<v.size()/W;++i) {
                mpz_sub_ui(av,L.N,(unsigned long)(operand ? 2*i+3 : i+1));
                if(i%7==0) mpz_set_ui(av,0);
                mpz_mod(av,av,L.N);
                std::vector<unsigned long long> c(W);mpz_to_words(c,W,av);
                std::copy(c.begin(),c.end(),v.begin()+(long)(i*W));
            }
        }
        std::vector<std::pair<size_t,size_t>> windows{{0,nc},{0,0},{nc,0},{0,1},{nc-1,1}};
        if(nc>1) windows.push_back({1,nc-1});
        if(nc>=4) windows.push_back({2,2});
        for(int alias=0;alias<4;++alias) {
            if(alias==3 && ma!=mb) continue;
            const auto &oracle_b=alias==3 ? a : b;
            std::vector<unsigned long long> expected(nb*nc*W);
            for(size_t s=0;s<nb;++s) for(size_t k=0;k<nc;++k) {
                mpz_set_ui(sum,0);
                for(size_t i=0;i<ma;++i) if(k>=i && k-i<mb) {
                    words_to_mpz(av,&a[(s*ma+i)*W],W);
                    words_to_mpz(bv,&oracle_b[(s*mb+k-i)*W],W);
                    mpz_mul(term,av,bv);mpz_add(sum,sum,term);
                }
                mpz_mod(sum,sum,L.N);
                std::vector<unsigned long long> c(W);mpz_to_words(c,W,sum);
                std::copy(c.begin(),c.end(),expected.begin()+(long)((s*nc+k)*W));
            }
            for(const auto window : windows) {
                const size_t first=window.first,count=window.second;
                auto aa=a,bb=b;
                std::vector<unsigned long long> separate,want(nb*count*W);
                auto &out=alias==1 || alias==3 ? aa : alias==2 ? bb : separate;
                for(size_t s=0;s<nb;++s)
                    std::copy(expected.begin()+(long)((s*nc+first)*W),
                              expected.begin()+(long)((s*nc+first+count)*W),want.begin()+(long)(s*count*W));
                flat_mul_batch(L,aa,ma,alias==3 ? aa : bb,mb,nb,out,-1,first,count);
                ++cases;words+=out.size();if(out!=want) ++bad;
            }
        }
        std::vector<unsigned long long> empty_a,empty_b,empty_out(7,123);
        flat_mul_batch(L,empty_a,ma,empty_b,mb,0,empty_out,-1,0,0);
        ++cases;if(!empty_out.empty()) ++bad;
    }
    /* A constant series is shorter than the requested Newton prefix at its first step. */
    CPoly constant;cp_resize(constant,1,W);constant[0][0]=2;
    const CPoly inverse=cp_inv_series(constant,5,L);
    mpz_set_ui(av,2);mpz_invert(inv,av,L.N);
    std::vector<unsigned long long> ci(W);mpz_to_words(ci,W,inv);
    ++cases;words+=inverse.size()*W;
    if(inverse.size()!=5 || inverse[0]!=ci) ++bad;
    for(size_t i=1;i<inverse.size();++i) if(!cp_coeff_zero(inverse[i])) ++bad;

    /* A bound violation in an OMITTED source slot must still be counted, including count=0.
       Use a separate diagnostic buffer so this intentional fault cannot poison the real run. */
    unsigned long long n=0,sb=0,sw=0,ss=0,os=0;int bpw=0;
    if(!ntt_shape_query(5,(int)L.S,&n,&bpw,&sb,&sw,&ss,&os)) std::exit(3);
    S4Reduce &R=*L.s4->red;
    S4Reduce::Shape *S=s4_shape_init(R,5,sb,ss,sw,bpw);
    const unsigned long long source_slots=9,slices=2,raw_stride=source_slots*sw;
    std::vector<unsigned long long> digits(slices*raw_stride,0);
    const auto top=sb-(sw-1)*(unsigned long long)bpw;
    digits[raw_stride+2*sw+sw-1]=1ull<<top; // source k=2, outside both tested windows
    unsigned long long *dd=nullptr,*dout=nullptr,*dbad=nullptr;
    CK(cudaMalloc(&dd,digits.size()*8));CK(cudaMalloc(&dout,(slices*W+2)*8));CK(cudaMalloc(&dbad,8));
    CK(cudaMemcpy(dd,digits.data(),digits.size()*8,cudaMemcpyHostToDevice));
    for(const auto window : std::initializer_list<std::pair<unsigned long long,unsigned long long>>{{7,1},{0,0}}) {
        const unsigned long long first=window.first,count=window.second;
        std::vector<unsigned long long> canary(slices*W+2,0xfeedfacedeadbeefull),got(canary.size());
        CK(cudaMemcpy(dout,canary.data(),canary.size()*8,cudaMemcpyHostToDevice));CK(cudaMemset(dbad,0,8));
        S2G_DISPATCH(R.nw,s4_launch_reduce,(int)R.nw,S->L,slices,source_slots,source_slots*slices,
                     dd,raw_stride,bpw,sw,R.dn,R.ninv,S->dy,R.w,dout,sb,dbad,nullptr,first,count,R.mersenne_bits);
        unsigned long long hb=0;
        CK(cudaMemcpy(&hb,dbad,8,cudaMemcpyDeviceToHost));CK(cudaMemcpy(got.data(),dout,got.size()*8,cudaMemcpyDeviceToHost));
        ++canonical_cases;if(hb!=1) ++bad;
        for(size_t i=0;i<got.size();++i) if(got[i]!=(i<slices*count*W ? 0ull : canary[i])) ++bad;
    }
    CK(cudaFree(dd));CK(cudaFree(dout));CK(cudaFree(dbad));
    mpz_clears(av,bv,term,sum,inv,nullptr);
    stage2_log::print(stage2_log::debug, "s4_output_window_check: cases=%llu words=%llu canonical_cases=%llu bad=%llu "
                "(GMP convolution, arbitrary/empty windows, strides, aliases, 5 slices, constant inverse)\n",
                cases,words,canonical_cases,bad);
    if(bad) std::exit(3);
}

void groot_lifetime_check(PolyLayer &L)
{
    const size_t W = L.W;
    S4Ctx *saved_s4 = L.s4;
    unsigned long long cases=0, words=0, descent_cases=0, bad=0;
    mpz_t c, a, b, sum;
    mpz_inits(c, a, b, sum, nullptr);
    for (int backend=0; backend<2; ++backend) {
        if (backend && !saved_s4) continue;
        L.s4 = backend ? saved_s4 : nullptr;
        for (size_t n : {0u, 1u, 2u, 3u, 5u, 8u, 9u, 17u}) {
            std::vector<std::vector<unsigned long long>> leaves;
            std::vector<unsigned long long> expected(W, 0ull);
            expected[0]=1;
            for (size_t i=0; i<n; ++i) {
                if (i%4==2) mpz_sub_ui(c, L.N, 1);
                else mpz_set_ui(c, i%4==3 ? 2 : 1);
                std::vector<unsigned long long> leaf(2*W, 0ull), next((i+2)*W, 0ull);
                mpz_to_words(leaf, W, c); leaf.resize(2*W, 0ull); leaf[W]=1;
                leaves.push_back(leaf);
                for (size_t k=0; k<i+2; ++k) {
                    mpz_set_ui(sum, 0);
                    if (k<=i) {
                        words_to_mpz(a, &expected[k*W], W);
                        mpz_mul(sum, a, c);
                    }
                    if (k>0) { words_to_mpz(b, &expected[(k-1)*W], W); mpz_add(sum, sum, b); }
                    mpz_mod(sum, sum, L.N);
                    std::vector<unsigned long long> coef(W);
                    mpz_to_words(coef, W, sum);
                    std::copy(coef.begin(), coef.end(), next.begin()+(long)(k*W));
                }
                expected.swap(next);
            }
            for (bool keep : {true, false}) {
                FTreeStats fs;
                size_t pad=0; std::vector<size_t> deg;
                auto tree=build_tree_flat(L, leaves, deg, pad, fs, -1, keep);
                ++cases; words+=expected.size();
                if (deg[1]!=n || tree[1]!=expected) ++bad;
                if (!keep) for (size_t i=2; i<tree.size(); ++i) if (!tree[i].empty()) ++bad;
                if (keep && n>=2) {
                    /* H(X)=X+5 evaluated at each leaf root -c; independent of cp_mod. */
                    CPoly H(2, std::vector<unsigned long long>(W, 0ull)); H[0][0]=5; H[1][0]=1;
                    {
                        // Independent monic division oracle; the production descent uses scaled states.
                        L.s4 = backend ? saved_s4 : nullptr;
                        std::vector<std::vector<unsigned long long>> values;
                        unsigned long long divmods=0;
                        descent_slow(L, tree, deg, pad, H, values, divmods);
                        L.s4 = backend ? saved_s4 : nullptr;
                        ++descent_cases;
                        if (values.size()!=n) { ++bad; continue; }
                        for (size_t i=0; i<n; ++i) {
                            words_to_mpz(c, leaves[i].data(), W);
                            mpz_ui_sub(sum, 5, c); mpz_mod(sum, sum, L.N);
                            std::vector<unsigned long long> coef(W); mpz_to_words(coef, W, sum);
                            if (values[i]!=coef) ++bad;
                        }
                    }
                }
            }
        }
    }
    L.s4=saved_s4;
    mpz_clears(c, a, b, sum, nullptr);
    stage2_log::print(stage2_log::debug, "groot_lifetime_check: cases=%llu words=%llu descent_cases=%llu bad=%llu "
                "(GMP, empty/single/padded trees, constant-one nonconstant polynomials)\n",
                cases, words, descent_cases, bad);
    if (bad) std::exit(3);
}

void segment_product_fixture(const LadderCtx &C)
{
    S3Workspace ws;ws.init(C);const size_t nw=C.nw;
    mpz_t N,R,ri,p,z,want,tmp;mpz_inits(N,R,ri,p,z,want,tmp,nullptr);
    words_to_mpz(N,C.hn.data(),nw);words_to_mpz(R,C.hmone.data(),nw);
    if(!mpz_invert(ri,R,N)){std::fprintf(stderr,"FATAL: R not invertible\n");std::exit(3);}
    unsigned long long cases=0,checks=0,legacy_different=0,state=0x982dded45;
    for(size_t n : {0u,1u,2u,15u,16u,17u,31u,32u,33u,63u,64u,65u,127u})
    for(size_t seg : {1u,2u,3u,7u,16u,17u,32u}) {
        ++cases;ws.need_segfix(seg);
        if(!n){S2G_DISPATCH((int)nw,s2g_launch_segprod,(int)nw,0,seg,C.ninv,ws.dn,nullptr,nullptr,ws.dsegfix);continue;}
        const size_t ns=(n+seg-1)/seg;std::vector<unsigned long long> input(n*nw),a(ns*nw),b(ns*nw),word(nw);
        for(size_t i=0;i<n;++i) {
            for(size_t j=0;j<nw;++j){state=state*6364136223846793005ull+1;word[j]=state;}
            words_to_mpz(z,word.data(),nw);mpz_mod(z,z,N);
            if(i%19==0)mpz_set_ui(z,0);else if(i%19==1)mpz_sub_ui(z,N,1);else if(i%19==2)mpz_set_ui(z,1);
            mpz_to_words(word,nw,z);std::copy(word.begin(),word.end(),input.begin()+i*nw);
        }
        unsigned long long *din=nullptr,*dout=nullptr;CK(cudaMalloc(&din,input.size()*8));CK(cudaMalloc(&dout,a.size()*8));
        CK(cudaMemcpy(din,input.data(),input.size()*8,cudaMemcpyHostToDevice));
        S2G_DISPATCH((int)nw,s2g_launch_segprod,(int)nw,n,seg,C.ninv,ws.dn,din,dout,ws.dsegfix);
        CK(cudaGetLastError());CK(cudaMemcpy(a.data(),dout,a.size()*8,cudaMemcpyDeviceToHost));
        S2G_DISPATCH((int)nw,s2g_launch_segprod,(int)nw,n,seg,C.ninv,ws.dn,din,dout,nullptr);
        CK(cudaGetLastError());CK(cudaMemcpy(b.data(),dout,b.size()*8,cudaMemcpyDeviceToHost));
        CK(cudaFree(din));CK(cudaFree(dout));
        const char *fault=std::getenv("NTT_GFINV_SEG_TEST_BAD");if(fault && std::atoi(fault))a[0]^=1;
        for(size_t j=0;j<ns;++j) {
            const size_t lo=j*seg,hi=std::min(n,lo+seg);mpz_set_ui(p,1);
            for(size_t i=lo;i<hi;++i){words_to_mpz(z,input.data()+i*nw,nw);mpz_mul(p,p,z);mpz_mod(p,p,N);}
            words_to_mpz(z,a.data()+j*nw,nw);
            if(mpz_cmp(p,z)){std::fprintf(stderr,"FATAL: segment fixture GMP mismatch n=%llu seg=%llu\n",(unsigned long long)n,(unsigned long long)seg);std::exit(3);}
            mpz_powm_ui(tmp,ri,(unsigned long)(hi-lo-1),N);mpz_mul(want,p,tmp);mpz_mod(want,want,N);
            words_to_mpz(z,b.data()+j*nw,nw);
            if(mpz_cmp(want,z)){std::fprintf(stderr,"FATAL: legacy segment formula mismatch\n");std::exit(3);}
            if(mpz_cmp(p,z))++legacy_different;++checks;
        }
    }
    mpz_clears(N,R,ri,p,z,want,tmp,nullptr);
    stage2_log::print(stage2_log::debug, "segment_product_fixture: cases=%llu checks=%llu legacy_different=%llu bad=0\n",cases,checks,legacy_different);
}

void block_product_fixture(const LadderCtx &C)
{
    S3Workspace ws;ws.init(C);const size_t w=C.nw;
    mpz_t N,R,ri,x,want,actual,power;mpz_inits(N,R,ri,x,want,actual,power,nullptr);
    words_to_mpz(N,C.hn.data(),w);words_to_mpz(R,C.hmone.data(),w);
    if(!mpz_invert(ri,R,N)){std::fprintf(stderr,"FATAL: product fixture radix is not a unit\n");std::exit(3);}
    unsigned long long cases=0,checked=0;
    for(int block:{1,2,7,64,128})for(int n:{0,1,2,63,64,65,127,128,129}) {
        ++cases;ws.need_vals((size_t)n);std::vector<unsigned long long> input((size_t)n*w),word(w);
        for(int i=0;i<n;++i) {
            if(i%11==0)mpz_set_ui(x,0);
            else if(i%11==1)mpz_sub_ui(x,N,1);
            else {mpz_set_ui(x,(unsigned long)(37*i+1));mpz_mod(x,x,N);}
            mpz_to_words(word,w,x);std::copy(word.begin(),word.end(),input.begin()+(size_t)i*w);
        }
        if(n)CK(cudaMemcpy(ws.dvals,input.data(),input.size()*8,cudaMemcpyHostToDevice));
        std::vector<std::vector<unsigned long long>> output;dev_block_products(ws,n,block,output);
        if(fuse_env_ull("NTT_S3_PRODUCTS_TEST_BAD",0) && !output.empty())output[0][0]^=1;
        if(output.size()!=(size_t)(n/block+(n%block!=0))) {std::fprintf(stderr,"FATAL: product fixture count\n");std::exit(3);}
        for(size_t b=0;b<output.size();++b) {
            const int lo=(int)b*block,hi=std::min(n,lo+block);mpz_set_ui(want,1);
            for(int i=lo;i<hi;++i) {
                words_to_mpz(x,input.data()+(size_t)i*w,w);mpz_mul(want,want,x);mpz_mod(want,want,N);
            }
            mpz_powm_ui(power,ri,(unsigned long)(hi-lo-1),N);mpz_mul(want,want,power);mpz_mod(want,want,N);
            words_to_mpz(actual,output[b].data(),w);
            if(mpz_cmp(actual,want)!=0) {std::fprintf(stderr,"FATAL: product fixture GMP mismatch n=%d block=%d index=%llu\n",n,block,(unsigned long long)b);std::exit(3);}
            ++checked;
        }
    }
    mpz_clears(N,R,ri,x,want,actual,power,nullptr);
    stage2_log::print(stage2_log::debug,"s3_product_fixture: compact=%d cases=%llu checked=%llu bad=0\n",
        s3_compact_products_requested(),cases,checked);
}

void device_baby_fixture(size_t w)
{
    mpz_t N,R,v;mpz_inits(N,R,v,nullptr);
    mpz_set_ui(N,1);mpz_mul_2exp(N,N,64*w-4);mpz_sub_ui(N,N,1);mpz_mul_ui(N,N,3);
    mpz_set_ui(R,1);mpz_mul_2exp(R,R,64*w);mpz_mod(R,R,N);
    LadderCtx c;c.nw=w;mpz_to_words(c.hn,w,N);mpz_to_words(c.hmone,w,R);
    unsigned long long inverse=1;for(int i=0;i<6;++i)inverse*=2-c.hn[0]*inverse;c.ninv=0-inverse;
    for(unsigned long long qz:{1ull,3ull}) {
        auto image=[&](std::vector<unsigned long long> &out,unsigned long long x){
            mpz_mul_ui(v,R,x);mpz_mod(v,v,N);mpz_to_words(out,w,v);};
        image(c.hqx,5);image(c.hqz,qz);image(c.ha24,3);
        std::vector<unsigned long long> js(513);for(size_t i=0;i<js.size();++i)js[i]=i;
        SmallPrimeBabyCache cache;cache.begin(c,210,20,1000,js);
        std::vector<std::vector<unsigned long long>> leaves(js.size());
        std::vector<std::string> factors;unsigned long long noninv=0;DeviceBabyStats st;
        if(!device_baby_generate(c,N,js,leaves,&cache,factors,noninv,st,true) ||
           st.checked_words!=js.size()*w || factors.size()!=noninv || !st.bad_groups) {
            std::fprintf(stderr,"FATAL: baby device host fixture incomplete\n");std::exit(3);
        }
        for(const auto &leaf:leaves) {
            if(leaf.size()!=2*w || leaf[w]!=1 ||
               std::any_of(leaf.begin()+w+1,leaf.end(),[](auto x){return x!=0;})) {
                std::fprintf(stderr,"FATAL: baby leaf is not monic\n");std::exit(3);
            }
        }
        stage2_log::print(stage2_log::debug, "baby_device_fixture: words=%llu qz=%llu checked_words=%llu noninv=%llu bad_groups=%llu bad=0\n",
            (unsigned long long)w,qz,(unsigned long long)st.checked_words,noninv,(unsigned long long)st.bad_groups);
    }
    mpz_clears(N,R,v,nullptr);
}

namespace stage2_giant_chunk {
void fixture() {
    uint64_t checks=0,bad=0;
    auto check=[&](bool value){++checks;if(!value)++bad;};
    for(uint64_t P:{1ull,48ull,126720ull,138240ull})for(uint64_t nw:{1ull,126ull,258ull}) {
        const uint64_t batch_bytes=16*nw*P;
        for(uint64_t budget:{1ull,batch_bytes-1,batch_bytes,batch_bytes+1,2*batch_bytes-1,2*batch_bytes}) {
            const auto f=plan(P,nw,budget,true),c=plan(P,nw,budget,false);
            check(f.valid && c.valid && f.points%P==0 && c.points%P==0);
            check(f.coordinate_bytes==std::max(uint64_t(1),budget/batch_bytes)*batch_bytes);
            const uint64_t k=budget/(16*nw);
            check(c.coordinate_bytes==std::max(uint64_t(1),k/P+uint64_t(k%P!=0))*batch_bytes);
            check(f.minimum_over_budget==(batch_bytes>budget));
        }
    }
    check(plan(126720,126,256ull<<20,true).points==126720);
    check(plan(126720,126,256ull<<20,false).points==253440);
    check(plan(138240,126,256ull<<20,true).points==138240);
    check(plan(138240,126,256ull<<20,false).points==138240);
    const auto maximum=std::numeric_limits<uint64_t>::max();
    check(!plan(0,1,1,true).valid);check(!plan(1,0,1,true).valid);
    check(!plan(1,1,0,true).valid);check(!plan(1,maximum,1,true).valid);
    check(!plan(maximum,1,1,true).valid);check(!plan(maximum,1,maximum,false).valid);
    stage2_log::print(stage2_log::debug,"giant_chunk_check: checks=%llu bad=%llu\n",
        (unsigned long long)checks,(unsigned long long)bad);
    if(bad)std::exit(3);
}
}
