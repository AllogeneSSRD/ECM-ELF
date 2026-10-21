// One saved curve, its setup/checks, relocation replay and final reporting.
#include "engine.h"
#include "options.h"
#include "plan.h"
#include "plan_runtime.h"
#include "checks.h"
#include "diagnostics.h"
#include "point_launch.h"
#include "../planning/ecm_stage2_giant_chunk.h"
#include "scale_plain.cuh"
#include "s4_pack_state.h"

int run_real(
    const char *n_str,
    bool n_is_hex,
    unsigned long long sigma,
    unsigned long long B1,
    unsigned long long B2,
    unsigned long long D_in,
    bool choose_d,
    const char *saved_qx_hex,
    Stage2Tail *saved_result,
    unsigned carrier_exponent,
    const char *saved_a24_hex)
{
    PolyLayer L;
    L.ntt_lengths.enabled=g_ntt_counts_enabled;
    L.device = g_device;
    stage2_memory::Session memory_session; // Destroyed after all curve-local device owners.
    stage2_log::phase("Plan Stage2 bounds and memory");
    std::string modulus_error;
    if(!L.configure(n_str,n_is_hex?16:10,carrier_exponent,modulus_error)) {
        std::fprintf(stderr,"%s: %s\n",NTT_PROBE_NAME,modulus_error.c_str());return 2;
    }
    L.cost.S = (long)L.S;
    const size_t nw = L.W;

    Stage2PlanRuntime planning;
    const int plan_code=prepare_stage2_plan(L,nw,sigma,B1,B2,D_in,choose_d,1,false,nullptr,planning);
    if(plan_code)return plan_code;
    const auto freeb=planning.freeb,totalb=planning.totalb,reserve=planning.reserve,cap=planning.cap;
    const auto D=planning.D,P_baby=planning.P_baby;
    const bool s4_on=planning.s4_on;

    /* the baby set: j COPRIME TO D, j <= D/2, ascending (the CPU reference's own rule).  Not
       "coprime to N": the first version of this filtered by gcd(N,j) and produced 1155 points
       for D=2310 instead of 240. */
    const double stage2_shape_begin=now_s();
    std::vector<unsigned long long> baby_j;
    for (unsigned long long j = 1; j <= D / 2; ++j)
        if (gcd_u64(j, D) == 1) baby_j.push_back(j);
    if (baby_j.size() != P_baby) {
        std::fprintf(stderr, "%s: FATAL: baby set has %llu points but phi(D)/2 = %llu\n",
                     NTT_PROBE_NAME, (unsigned long long)baby_j.size(), P_baby);
        return 3;
    }
    const double stage2_shape_seconds=now_s()-stage2_shape_begin;
    const unsigned long long imax = B2 / D + 2;
    /* B1 and B2 are printed RAW, not as a derived count: --b2 used to be read with strtoull
       base 10, so "1e11" silently became 1 and the run quietly computed a B2=1 shape
       (giant_points=2) while every other line looked normal -- caught only because
       giant_points was printed at all (section 27).  Both are parsed as floating point now. */
    stage2_log::print(stage2_log::phases, "real_shape: D=%llu P=phi(D)/2=%llu baby_points=%llu giant_points=%llu "
                "num_poly_g=%llu loops=%llu B1=%llu B2=%llu S_bits=%ld\n", D, P_baby,
                (unsigned long long)baby_j.size(), imax, (P_baby + imax - 1) / P_baby,
                (imax + P_baby - 1) / P_baby - 1, B1, B2, (long)L.S);

    /* ---- Montgomery parameters (R, ninv, a24, Q) ---- */
    mpz_t R, tmp, a24, ax, az;
    mpz_inits(R, tmp, a24, ax, az, nullptr);
    mpz_set_ui(tmp, 1);
    mpz_mul_2exp(tmp, tmp, (unsigned long)(64 * nw));
    mpz_mod(R, tmp, L.N);
    unsigned long long ninv = 0;
    {
        mpz_t nmod, inv, pow2;
        mpz_inits(nmod, inv, pow2, nullptr);
        mpz_set_ui(pow2, 1);
        mpz_mul_2exp(pow2, pow2, 64);
        mpz_mod(nmod, L.N, pow2);
        if (mpz_invert(inv, nmod, pow2) == 0) {
            std::fprintf(stderr, "%s: N is not invertible mod 2^64\n", NTT_PROBE_NAME);
            return 2;
        }
        mpz_sub(inv, pow2, inv);
        unsigned long long buf[2] = {0, 0};
        size_t cnt = 0;
        mpz_export(buf, &cnt, -1, 8, 0, 0, inv);
        ninv = buf[0];
        mpz_clears(nmod, inv, pow2, nullptr);
    }
    std::vector<unsigned long long> hn(nw), ha24(nw), hmone(nw);
    mpz_to_words(hn, nw, L.N);
    mpz_to_words(hmone, nw, R);
    if (saved_a24_hex) {
        if(!saved_qx_hex || mpz_set_str(a24,saved_a24_hex,16) || mpz_sgn(a24)<0 ||
           mpz_cmp(a24,L.target())>=0) {
            std::fprintf(stderr,"invalid saved curve a24\n");return 2;
        }
    } else if (suyama_curve(a24, ax, az, sigma, L.target()) != 0) {
        std::fprintf(stderr, "%s: degenerate sigma for this N\n", NTT_PROBE_NAME);
        return 2;
    }
    mpz_mul(tmp, a24, R);
    mpz_mod(tmp, tmp, L.N);
    mpz_to_words(ha24, nw, tmp);
    if (mont_selftest(hn, nw, ninv, L.N, R) != 0) return 1;
    if (xadd_selftest(hn,nw,ninv,L.N,R) != 0) return 1;

    /* A Stage1 text save contains ordinary affine X. Do not regenerate the
       exponent or apply the optional Stage1 torsion multiplier on resume. */
    std::vector<unsigned long long> qx(nw), qz(nw);
    if (saved_qx_hex) {
        if (mpz_set_str(tmp, saved_qx_hex, 16) != 0 || mpz_sgn(tmp) < 0 ||
            mpz_cmp(tmp, L.target()) >= 0) {
            std::fprintf(stderr, "%s: invalid saved affine X\n", NTT_PROBE_NAME);
            return 2;
        }
        mpz_to_words(qx, nw, tmp);
        qz[0] = 1;
        stage2_log::print(stage2_log::phases, "stage1_save_resume: sigma=%llu B1=%llu normalized_Z=1 stage1_skipped=1\n",
                    sigma, B1);
    } else {
        std::fprintf(stderr,"production Stage2 requires a normalized Stage1 save\n");
        return 2;
    }
    {
        mpz_t X, Z;
        mpz_inits(X, Z, nullptr);
        words_to_mpz(X, qx.data(), nw);
        words_to_mpz(Z, qz.data(), nw);
        affine_x_gmp(tmp, X, Z, L.target());
        char *s = mpz_get_str(nullptr, 16, tmp);
        stage2_log::print(stage2_log::debug, "real_setup_Q: Q_x_hex=%.64s... (%zu hex digits)\n", s, std::strlen(s));
        const char *qfull=std::getenv("NTT_STAGE1_Q_DUMP");
        if(qfull && std::atoi(qfull)!=0) stage2_log::print(stage2_log::debug, "real_setup_Q_full: hex=%s\n",s);
        void (*ff)(void *, size_t) = nullptr;
        mp_get_memory_functions(nullptr, nullptr, &ff);
        ff(s, std::strlen(s) + 1);
        mpz_clears(X, Z, nullptr);
    }
    /* Stage1 Q is ready. Include context/reducer setup, baby/F-tree, then the entire tail.
       Baby index enumeration ran before Stage1; account its measured work once separately. */
    const double stage2_init_begin=now_s();
    ecm_stage2::timing::Boundaries phase_boundaries;
    phase_boundaries.shape=stage2_shape_seconds;
    phase_boundaries.init_begin=stage2_init_begin;
    LadderCtx C;
    mpz_to_words(C.target_n,nw,L.target());
    C.hn = hn;
    C.nw = nw;
    C.ninv = ninv;
    C.ha24 = ha24;
    C.hmone = hmone;
    C.hqx.assign(nw, 0ull);
    C.hqz.assign(nw, 0ull);
    {
        mpz_t X, Z;
        mpz_inits(X, Z, nullptr);
        words_to_mpz(X, qx.data(), nw);
        mpz_mul(X, X, R);
        mpz_mod(X, X, L.N);
        mpz_to_words(C.hqx, nw, X);
        words_to_mpz(Z, qz.data(), nw);
        mpz_mul(Z, Z, R);
        mpz_mod(Z, Z, L.N);
        mpz_to_words(C.hqz, nw, Z);
        mpz_clears(X, Z, nullptr);
    }

    /* ---- the arena + the device reduction, with the budget printed ---- */
    NttArena arena;
    arena.device = g_device;
    arena.cap_bytes = cap;
    {
        const char *envh = std::getenv("NTT_ARENA_CAP_KB");
        if (envh && *envh) arena.cap_bytes = (size_t)std::strtoull(envh, nullptr, 10) << 10;
    }
    S4Ctx s4;
    S4Reduce red;
    {
        const char *es = std::getenv("NTT_S4_SAMPLE");
        if (es && *es) g_s4_sample_limit = std::atoll(es);
        const char *ee = std::getenv("NTT_S4_CHECK_EVERY");
        if (ee && *ee) g_s4_check_every = (unsigned long long)std::strtoull(ee, nullptr, 10);
    }
    L.arena = &arena;
    double stage2_fixture_begin=0.0;
    if (s4_on) {
        s4_reduce_init(red, L.N, nw, hn, ninv);
        s4.red = &red;
        L.s4 = &s4;
        stage2_fixture_begin=now_s();
        const char *flat_test = std::getenv("NTT_S4_FLAT_TEST");
        if (flat_test && std::atoi(flat_test) != 0) s4_flat_input_check(L);
        const char *readback_test = std::getenv("NTT_S4_FINAL_READBACK_TEST");
        if (readback_test && std::atoi(readback_test) != 0) s4_flat_input_check(L,5);
        const char *window_test=std::getenv("NTT_S4_OUTPUT_WINDOW_TEST");
        if(window_test && std::atoi(window_test)!=0) s4_output_window_check(L);
        const char *scaled_test=std::getenv("NTT_SCALED_TEST");
        if(scaled_test && std::atoi(scaled_test)!=0) scaled_descent_fixture(L);
    }
    if(!s4_on) stage2_fixture_begin=now_s();
    if(fuse_env_ull("NTT_S3_PRODUCTS_TEST",0))block_product_fixture(C);
    const char *fold_test=std::getenv("NTT_FOLD_FLAT_TEST");
    if(fold_test && std::atoi(fold_test))fold_flat_fixture(L);
    gscale_flag("NTT_GSCALE_DEVICE_CHECK");gscale_flag("NTT_GSCALE_DEVICE_TEST_BAD");
    gscale_flag("NTT_SCALED_ROOT_CHECK");gscale_flag("NTT_SCALED_ROOT_TEST_BAD");
    gscale_flag("NTT_SCALED_FRONTIER_CHECK");gscale_flag("NTT_SCALED_FRONTIER_TEST_BAD");gscale_flag("NTT_SCALED_FRONTIER_ALLOC_FAIL");
    const char *fold_device_test=std::getenv("NTT_FOLD_DEVICE_TEST");
    if(fold_device_test && std::atoi(fold_device_test))fold_device_fixture(L);
    if(fuse_env_ull("NTT_BABY_DEVICE_TEST",0))device_baby_fixture(nw);
    const char *seg_test=std::getenv("NTT_GFINV_SEG_TEST");
    if(seg_test && std::atoi(seg_test))segment_product_fixture(C);
    const char *ginv_test=std::getenv("NTT_GFINV_BATCH_TEST");
    if(ginv_test && std::atoi(ginv_test))gfinv_batch_fixture(L.N,nw);
    const char *gdevice_test=std::getenv("NTT_GROOT_DEVICE_TEST");
    if(gdevice_test && std::atoi(gdevice_test))groot_device_fixture(L);
    const char *groot_test = std::getenv("NTT_S4_GROOT_TEST");
    if (groot_test && std::atoi(groot_test) != 0) groot_lifetime_check(L);
    const char *workspace_test = std::getenv("NTT_ARENA_WORKSPACE_TEST");
    if (workspace_test && std::atoi(workspace_test) != 0) {
        ntt_workspace_check(g_device);
        unsigned long long cases=0,bad=0;
        for(unsigned buffers:{2u,3u})for(unsigned long long n:{128ull,4096ull,1048576ull})
        for(unsigned long long slots:{1ull,33ull,4095ull})
        for(unsigned long long batch:{1ull,3ull,65ull,1000ull})
        for(unsigned long long budget:{1ull,4096ull,1048576ull,268435456ull}) {
            unsigned long long want=1;
            for(auto c=batch;c;c/=2)
                if(8ull*(buffers*n+slots+2)*c<=budget) {want=c;break;}
            ++cases;
            if(ecm_stage2::chunk_slices(n,slots,batch,budget,buffers)!=want)++bad;
            unsigned long long bytes=0;
            ++cases;
            if(!ecm_stage2::chunk_request_bytes(n,slots,want,buffers,bytes) ||
               bytes!=8ull*(buffers*n+slots+2)*want)++bad;
        }
        unsigned long long bytes=0;
        for(auto n:{0ull,~0ull}) {
            ++cases;if(ecm_stage2::chunk_request_bytes(n,1,1,2,bytes))++bad;
        }
        ++cases;if(ecm_stage2::chunk_request_bytes(1,1,~0ull,3,bytes))++bad;
        ++cases;if(ecm_stage2::chunk_request_bytes(1,1,1,4,bytes))++bad;
        stage2_log::print(stage2_log::debug,"s4_chunk_budget_check: cases=%llu bad=%llu\n",cases,bad);
        if(bad)return 3;
        // Independent dense padded-tree oracle for the closed-form planner.
        // Enumerates children just as the real tree does; no geometry helper
        // is used to construct the expected groups or largest operand.
        cases=bad=0;
        std::vector<size_t> sizes;
        for(size_t p=1;p<=127;++p)sizes.push_back(p);
        for(size_t p:{511u,512u,513u,1023u,1024u,1025u,65535u,65536u,65537u,131071u,131072u,131073u})sizes.push_back(p);
        for(size_t p:sizes) {
            size_t pad=1;while(pad<p)pad*=2;
            std::vector<size_t> degree(2*pad,0);
            for(size_t i=0;i<p;++i)degree[pad+i]=1;
            for(size_t i=pad;--i;)degree[i]=degree[2*i]+degree[2*i+1];
            using Group=std::tuple<unsigned long long,unsigned long long,unsigned long long>;
            std::vector<Group> expected,actual;unsigned long long top=p==1?1:0;
            for(size_t base=pad/2;base;base/=2) {
                std::map<std::pair<size_t,size_t>,size_t> groups;
                for(size_t i=0;i<base;++i) {
                    size_t a=degree[2*base+2*i],b=degree[2*base+2*i+1];
                    if(!a || !b)continue;
                    auto key=std::make_pair(std::min(a,b)+1,std::max(a,b)+1);
                    ++groups[key];top=std::max(top,(unsigned long long)key.second);
                }
                for(const auto &g:groups)expected.emplace_back(g.first.first,g.first.second,g.second);
            }
            ++cases;
            if(!ecm_stage2::tree_multiply_groups(p,[&](auto a,auto b,auto n){actual.emplace_back(a,b,n);return true;}) ||
               actual!=expected || ecm_stage2::tree_operand_coefficients(p)!=top)++bad;
            for(unsigned buffers:{2u,3u})for(bool physical:{false,true}) {
                ecm_stage2::TreeWorkspacePlan plan;const unsigned long long budget=4096;
                auto query=[](auto m,int,auto *n,auto *out) {
                    *n=1;while(*n<4*m)*n*=2;*out=2*m-1;return true;
                };
                unsigned long long pairs=0,chunks=0,big=0,output=0,digits_bytes=0,keyed=0;
                std::map<std::pair<unsigned long long,unsigned long long>,unsigned long long> retained,large;
                for(const auto &g:expected) {
                    const auto a=std::get<0>(g),b=std::get<1>(g),nb=std::get<2>(g);
                    unsigned long long n=1;while(n<4*b)n*=2;const auto slots=2*b-1;
                    unsigned long long c=1;
                    for(auto test=nb;test;test/=2)if(8*test*((physical?buffers:3)*n+slots+(physical?2:0))<=budget){c=test;break;}
                    pairs+=nb;chunks+=(nb+c-1)/c;
                    big=std::max(big,8ull*buffers*n*c);output=std::max(output,8ull*(a+b-1)*c);
                    for(size_t offset=0;offset<nb;offset+=c) {
                        auto count=std::min(c,nb-offset);retained[{n,count}]=8ull*(slots+2)*count;large[{n,count}]=24ull*n*count;
                    }
                }
                for(const auto &e:retained)digits_bytes+=e.second;
                for(const auto &e:large)keyed+=e.second;
                ++cases;
                if(!ecm_stage2::tree_workspace_plan(p,64,query,budget,buffers,physical,0,plan) ||
                   plan.groups!=expected.size() || plan.pairs!=pairs || plan.chunks!=chunks ||
                   plan.big_peak_bytes!=big || plan.output_peak_bytes!=output ||
                   plan.digit_retained_bytes!=digits_bytes || plan.keyed_big_retained_bytes!=keyed)++bad;
            }
        }
        ecm_stage2::TreeWorkspacePlan invalid;
        ++cases;if(ecm_stage2::tree_multiply_groups(~0ull,[](auto,auto,auto){return true;}))++bad;
        ++cases;if(ecm_stage2::tree_workspace_plan(3,64,[](auto,int,auto *n,auto *out){*n=~0ull;*out=3;return true;},4096,2,false,0,invalid))++bad;
        stage2_log::print(stage2_log::debug,"s4_tree_memory_check: cases=%llu bad=%llu (dense padded-tree oracle, exact groups/chunks/retention)\n",cases,bad);
        if(bad)return 3;
    }
    if(fuse_env_ull("NTT_FUSE_COOP_TEST",0))ntt_fuse_coop_check(g_device);
    const char *fuse_test = std::getenv("NTT_FUSE_LIFETIME_TEST");
    if (fuse_test && std::atoi(fuse_test) != 0) {
        ntt_fuse_lifetime_check(g_device);
        ntt_fuse_capacity_check(g_device);
    }
    const double stage2_fixture_seconds=now_s()-stage2_fixture_begin;
    const char *real_dump=std::getenv("NTT_REAL_F_DUMP");
    bool stage2_extra_fixtures=(real_dump && *real_dump) || fuse_env_ull("NTT_TARGET_LEAF_HASH",0)!=0;
    for(const char *key : {"NTT_BABY_DEVICE_TEST","NTT_BABY_DEVICE_CHECK","NTT_BABY_DEVICE_TEST_BAD","NTT_BABY_DEVICE_ALLOC_FAIL","NTT_FUSE_COOP_TEST","NTT_FUSE_COOP_BAD","NTT_XADD6_TEST","NTT_XADD6_TEST_BAD","NTT_S4_FLAT_TEST","NTT_S4_FINAL_READBACK_TEST","NTT_S4_OUTPUT_WINDOW_TEST",
                           "NTT_S4_GROOT_TEST","NTT_ARENA_WORKSPACE_TEST","NTT_FUSE_LIFETIME_TEST","NTT_S3_PRODUCTS_TEST","NTT_S3_PRODUCTS_TEST_BAD",
                           "NTT_SCALED_TEST","NTT_SCALED_CHECK","NTT_SCALED_FRONTIER_CHECK","NTT_SCALED_FRONTIER_TEST_BAD","NTT_SCALED_FRONTIER_ALLOC_FAIL","NTT_SCALED_ROOT_CHECK","NTT_SCALED_ROOT_TEST_BAD","NTT_GROOT_DEVICE_TEST","NTT_GROOT_DEVICE_CHECK","NTT_GROOT_DEVICE_TEST_BAD","NTT_GROOT_LEAF_CHUNK","NTT_GFINV_BATCH_TEST","NTT_GFINV_BATCH_TEST_BAD","NTT_FOLD_FLAT_TEST","NTT_FOLD_FLAT_TEST_BAD","NTT_GSCALE_DEVICE_CHECK","NTT_GSCALE_DEVICE_TEST_BAD","NTT_FOLD_DEVICE_TEST","NTT_FOLD_DEVICE_CHECK","NTT_FOLD_DEVICE_TEST_BAD","NTT_FOLD_DEVICE_ALLOC_FAIL","NTT_GROOT_TO_FOLD_CHECK","NTT_GROOT_TO_FOLD_TEST_BAD","NTT_GFINV_SEG_TEST","NTT_GFINV_SEG_TEST_BAD","NTT_GFINV_SEG_CHECK","NTT_GIANT_SEED_CHECK","NTT_S4_MERSENNE_TEST","NTT_S4_MERSENNE_TEST_BAD","NTT_SMALL_PRIME_CHECK","NTT_SMALL_PRIME_TEST_BAD","NTT_SMALL_PRIME_CACHE_STALE","NTT_DEVICE_GLEAF_CHECK","NTT_DEVICE_GLEAF_TEST_BAD"}) {
        const char *v=std::getenv(key);
        if(v && std::atoi(v)!=0) stage2_extra_fixtures=true;
    }
    {
        bool ok1 = false, ok2 = false;
        const auto tree_coefficients=ecm_stage2::tree_operand_coefficients(P_baby);
        const double w1 = (double)real_shape_words(P_baby+1, (int)L.S, &ok1);
        const double w2 = (double)real_shape_words(tree_coefficients, (int)L.S, &ok2);
        stage2_log::print(stage2_log::phases, "mem_budget: free=%.0f MB total=%.0f MB reserve=%.0f MB arena_cap=%.0f MB ; "
                    "P=%llu => largest transform (the fold, operand %llu coeffs) = %.0f MB ; "
                    "tree top (operand %llu coeffs) = %.0f MB ; the arena holds one entry per "
                    "shape and a shape that does not fit falls back to per-call cudaMalloc "
                    "(same kernels and same result, slower) -- arena_overflow counts those\n",
                    freeb / 1048576.0, totalb / 1048576.0, reserve / 1048576.0,
                    arena.cap_bytes / 1048576.0, P_baby, P_baby + 1,
                    w1 * 8.0 / 1048576.0, tree_coefficients, w2 * 8.0 / 1048576.0);
    }

    /* ---- the baby points and F, exactly as the dump path does ---- */
    double t_f = 0.0;
    std::vector<std::vector<unsigned long long>> Ft;
    std::vector<size_t> Fdeg;
    size_t Fpad = 0, fdeg = 0;
    /* OBJECTIVE 3 (section 31.4): factors revealed by DEGENERATE BABY POINTS -- points whose Z
       shares a factor with N, i.e. points that are the identity modulo that factor.  Collected
       here (the batched run does not exist yet) and merged into the reported factor set below. */
    std::vector<std::string> baby_deg;
    unsigned long long baby_noninv=0;
    bool baby_nonunit_seen=false;
    phase_boundaries.baby_begin=now_s();
    stage2_log::phase("Generate and normalize baby points");
    stage2_memory::snapshot("before_baby");
    SmallPrimeBabyCache small_cache;
    const bool reuse_small=small_prime_flag("NTT_SMALL_PRIME_REUSE");
    if(reuse_small)small_cache.begin(C,D,B1,B2,baby_j);
    {
        const double t0 = now_s();
        std::vector<unsigned long long> bx, bz;
        auto &noninv=baby_noninv;                    /* legacy failed affine conversions; saturation tracked separately */
        const double tb0 = now_s();
        std::vector<std::vector<unsigned long long>> leaf(baby_j.size());
        DeviceBabyStats bs;
        const bool baby_requested=fuse_env_ull("NTT_BABY_DEVICE",0)!=0;
        const bool baby_used=baby_requested && device_baby_generate(C,L.target(),baby_j,leaf,
            reuse_small?&small_cache:nullptr,baby_deg,noninv,bs);
        if(baby_used && bs.bad_groups)baby_nonunit_seen=true;
        if(!baby_used)ladder_points(C,baby_j,bx,bz);
        const double tb1=baby_used?tb0+bs.ladder_seconds:now_s();
        if(!baby_used) {
            mpz_t X, Z, xj, neg, gq;
            mpz_inits(X, Z, xj, neg, gq, nullptr);
            std::vector<unsigned long long> w(nw, 0ull);
            /* ---- ONE INVERSION PER SEGMENT, NOT PER POINT (section 25) --------------------------
               x_j = X_j/Z_j mod N used to cost one mpz_invert per BABY point: 115200 of them at
               ~61.5 us each is ~7 s, and the whole pre-`elapsed` setup -- this ladder plus its
               conversion and real_setup's 12.1 s -- is ~44 s of the 317 s wall clock, the second
               largest improvable item after the G tree.  Section 42 already fixed exactly this for
               the GIANT points (224.8 s -> 20.3 s) with Montgomery's trick: invert the segment's
               PRODUCT once, then walk back.  Same code shape here, same fallback: if any Z in the
               segment is not invertible (gcd(Z,N) > 1 -- the degeneracy that IS a factor, objective
               3) the whole segment drops to the per-point path, which records the gcd. */
            const size_t BS = 256;
            /* plain C arrays, NOT std::vector<mpz_t>: mpz_t is an ARRAY type, so a vector of it
               cannot be constructed (measured: "a new-initializer may not be specified for an
               array" from MSVC's xmemory) -- the same trap this project recorded once before */
            mpz_t pv[BS + 1];                            /* prefix products */
            mpz_t zv[BS];
            for (size_t j = 0; j <= BS; ++j) mpz_init(pv[j]);
            for (size_t j = 0; j < BS; ++j) mpz_init(zv[j]);
            mpz_t iprod, tmul;
            mpz_inits(iprod, tmul, nullptr);
            const size_t nb = baby_j.size();
            for (size_t lo = 0; lo < nb; lo += BS) {
                const size_t hi = ((lo + BS) < nb) ? (lo + BS) : nb;
                const size_t seg = hi - lo;
                bool clean = true;
                mpz_set_ui(pv[0], 1);
                for (size_t j = 0; j < seg; ++j) {
                    words_to_mpz(zv[j], &bz[(lo + j) * nw], nw);
                    if (L.target_zero(zv[j])) { clean = false; break; }
                    mpz_mul(pv[j + 1], pv[j], zv[j]);
                    mpz_mod(pv[j + 1], pv[j + 1], L.N);
                }
                if (clean && mpz_invert(iprod, pv[seg], L.target()) == 0) clean = false;
                if (!clean) {
                    baby_nonunit_seen=true; // includes target-zero Z accepted as a legacy placeholder
                    /* the affine path, VERBATIM: whatever refuses to invert is a hit, not an
                       accident, and affine_x_gmp_checked records its gcd as a factor */
                    for (size_t j = 0; j < seg; ++j) {
                        words_to_mpz(X, &bx[(lo + j) * nw], nw);
                        words_to_mpz(Z, &bz[(lo + j) * nw], nw);
                        /* The established affine helper accepts Z=0 as leaf x=0. Its GCD
                           proof is N, not 1; retain saturation without changing leaf semantics. */
                        if(reuse_small && L.target_zero(Z))small_cache.record(lo+j,L.target());
                        if (!affine_x_gmp_checked(xj, X, Z, L.target())) {
                            ++noninv;
                            mpz_gcd(gq, Z, L.target());
                            if(reuse_small)small_cache.record(lo+j,gq);
                            char *gs = mpz_get_str(nullptr, 10, gq);
                            baby_deg.push_back(gs);
                            void (*ff)(void *, size_t) = nullptr;
                            mp_get_memory_functions(nullptr, nullptr, &ff);
                            ff(gs, std::strlen(gs) + 1);
                        }
                        mpz_neg(neg, xj);
                        mpz_mod(neg, neg, L.N);
                        mpz_to_words(w, nw, neg);
                        leaf[lo + j].assign(2 * nw, 0ull);
                        std::copy(w.begin(), w.end(), leaf[lo + j].begin());
                        leaf[lo + j][nw] = 1;
                    }
                    continue;
                }
                /* At step j, iprod = 1/(z_0..z_j). Multiply by pv[j] = z_0..z_{j-1}
                   to get 1/z_j, then update iprod for the shorter prefix. */
                for (size_t j = seg; j-- > 0;) {
                    words_to_mpz(X, &bx[(lo + j) * nw], nw);
                    mpz_mul(tmul, iprod, pv[j]);
                    mpz_mod(tmul, tmul, L.N);
                    mpz_mul(tmul, X, tmul);
                    mpz_mod(xj, tmul, L.N);
                    mpz_mul(iprod, iprod, zv[j]);
                    mpz_mod(iprod, iprod, L.N);
                    mpz_neg(neg, xj);
                    mpz_mod(neg, neg, L.N);
                    mpz_to_words(w, nw, neg);
                    leaf[lo + j].assign(2 * nw, 0ull);
                    std::copy(w.begin(), w.end(), leaf[lo + j].begin());
                    leaf[lo + j][nw] = 1;
                }
            }
            for (size_t j = 0; j <= BS; ++j) mpz_clear(pv[j]);
            for (size_t j = 0; j < BS; ++j) mpz_clear(zv[j]);
            mpz_clears(iprod, tmul, nullptr);
            if (noninv)
                stage2_log::print(stage2_log::debug, "baby_degenerate: points=%llu of %llu have Z sharing a factor with N "
                            "-> their gcd was recorded as a factor\n", noninv,
                            (unsigned long long)baby_j.size());
            mpz_clears(X, Z, xj, neg, gq, nullptr);
        }
        stage2_log::print(stage2_log::debug, "baby_device: requested=%d enabled=%d points=%llu groups=%llu bad_groups=%llu "
                    "payload_bytes=%llu root_d2h_bytes=%llu leaf_d2h_bytes=%llu coordinate_d2h_bytes=%llu "
                    "seed_h2d_bytes=%llu checked_words=%llu invert=%.6f total=%.6f\n",
                    (int)baby_requested,(int)baby_used,(unsigned long long)bs.points,(unsigned long long)bs.groups,
                    (unsigned long long)bs.bad_groups,(unsigned long long)bs.payload_bytes,
                    (unsigned long long)bs.root_d2h_bytes,(unsigned long long)bs.leaf_d2h_bytes,
                    (unsigned long long)bs.coordinate_d2h_bytes,(unsigned long long)bs.seed_h2d_bytes,
                    (unsigned long long)bs.checked_words,bs.invert_seconds,bs.total_seconds);
        stage2_log::print(stage2_log::debug, "ladder: baby_points=%llu (device x_j; there is no CPU reference at this "
                    "shape, see the report)\n", (unsigned long long)baby_j.size());
        /* the setup pieces that live OUTSIDE `elapsed` had no timer at all (section 25): they are
           ~44 s of the 317 s wall clock, so naming them is the first step */
        stage2_log::print(stage2_log::phases, "real_baby: points=%llu ladder=%.3f s affine=%.3f s degenerate=%llu\n",
                    (unsigned long long)baby_j.size(), tb1 - tb0, now_s() - tb1, noninv);
        FTreeStats fs;
        phase_boundaries.ftree_begin=now_s();
        stage2_log::phase("Build baby-point F tree");
        stage2_memory::snapshot("before_ftree");
        Ft = build_tree_flat(L, leaf, Fdeg, Fpad, fs, BC_FTREE);
        s4_oracle_drain(red);    /* include final F-tree validation in its phase timer */
        fdeg = Fdeg[1];
        if(real_dump && *real_dump) {
            if(baby_j.size()>4096) {std::fprintf(stderr,"%s: FATAL: real F dump limited to 4096 babies\n",NTT_PROBE_NAME);std::exit(3);}
            std::vector<std::string> bx_hex,F_hex;
            mpz_t x,z,q;mpz_inits(x,z,q,nullptr);
            auto as_hex=[&](const mpz_t v) {
                char *s=mpz_get_str(nullptr,16,v);std::string result=s;
                void (*release)(void*,size_t)=nullptr;mp_get_memory_functions(nullptr,nullptr,&release);
                release(s,result.size()+1);return result;
            };
            for(const auto &v:leaf) {
                words_to_mpz(x,v.data(),nw);mpz_neg(x,x);mpz_mod(x,x,L.target());bx_hex.push_back(as_hex(x));
            }
            for(size_t i=0;i<=fdeg;++i) {words_to_mpz(x,Ft[1].data()+i*nw,nw);mpz_mod(x,x,L.target());F_hex.push_back(as_hex(x));}
            words_to_mpz(x,qx.data(),nw);words_to_mpz(z,qz.data(),nw);affine_x_gmp(q,x,z,L.target());
            if(!write_f_dump(real_dump,L.target(),a24,q,D,B1,B2,sigma,baby_j,bx_hex,F_hex,fdeg)) {
                std::fprintf(stderr,"%s: FATAL: cannot write real F dump\n",NTT_PROBE_NAME);std::exit(3);
            }
            mpz_clears(x,z,q,nullptr);
            stage2_log::print(stage2_log::debug, "real_F_dump: babies=%llu coefficients=%llu path=%s\n",
                (unsigned long long)bx_hex.size(),(unsigned long long)F_hex.size(),real_dump);
        }
        stage2_log::print(stage2_log::phases, "ftree_real: leaves=%llu padded=%llu muls=%llu ntt_calls=%llu "
                    "ntt_seconds=%.3f\n", (unsigned long long)fs.leaves,
                    (unsigned long long)fs.padded, (unsigned long long)fs.muls, L.ntt_calls,
                    L.ntt_seconds);
        t_f = now_s() - t0;
    }
    stage2_log::print(stage2_log::debug, "exactness: binding_shape P=%llu S=%llu slot_bits=%llu slot_words=%llu "
                "L=P*slot_words=%llu bpw=%d ; L*(2^bpw-1)^2 = 2^%.3f < p = 2^%.3f -> %s\n",
                L.bind_P, (unsigned long long)L.S, L.bind_slot_bits, L.bind_slot_words,
                L.bind_L, L.bind_bpw, L.bind_bound_bits, mpz_log2d_p(),
                exact_ok_terms(L.bind_L, L.bind_bpw) ? "OK" : "VIOLATED");

    if(reuse_small) {
        small_cache.ready=std::is_sorted(small_cache.js.begin(),small_cache.js.end()) &&
            std::adjacent_find(small_cache.js.begin(),small_cache.js.end())==small_cache.js.end();
        if(small_prime_flag("NTT_SMALL_PRIME_CACHE_STALE"))++small_cache.D;
    }
    phase_boundaries.init_end=now_s();
    const double stage2_init_seconds=stage2_shape_seconds+(phase_boundaries.init_end-stage2_init_begin);
    /* ---- the tails ---- */
    Stage2Params SP;
    SP.D = D; SP.B1 = B1; SP.B2 = B2; SP.baby_j = baby_j;
    {
        g_gfinv={};g_giant_seed={};

        const unsigned long long nb = L.ntt_calls, nl = L.ntt_launches;
        const double ns = L.ntt_seconds;
        const double fw0 = L.t_fwd, iv0 = L.t_inv, sl0 = L.t_slot, st0 = L.t_setup, pk0 = L.t_pack,
                     mc0 = L.t_maxc, h20 = L.t_h2d, d20 = L.t_d2h, ex0 = L.t_ext, xc0 = L.t_xchk;
        const double dc0 = L.t_d2h_coeff;
        const unsigned long long dw0 = L.d2h_coeff_words;
        const double hp0 = L.t_hpack, sc0 = L.t_scan, hb0 = L.t_h2d_batch, ck0 = L.t_check;
        const double cr0 = L.t_check_reset, ck0b = L.t_check_kernel, cd0 = L.t_check_d2h;
        const double pl0b = L.t_plan, oc0 = L.t_opcopy, ho0 = L.t_hout;
        const double ry0 = L.s4 ? L.s4->t_h2d_raw : 0.0;
        const unsigned long long rw0 = L.s4 ? L.s4->raw_words : 0ull;
        const unsigned long long pl0 = L.s4 ? L.s4->pack_launches : 0ull;
        const double t0 = now_s();
        const auto relocation=ecm_stage2::giant_range(B2,D,g_relocation_enabled);
        const bool baby_nonunit=baby_nonunit_seen || baby_noninv!=0 || !baby_deg.empty();
        const bool attempted=relocation.active && !baby_nonunit;
        double probe_seconds=0,replay_seconds=0,final_begin=t0;
        bool replayed=relocation.active && baby_nonunit;
        ecm_stage2::timing::PhaseTimes probe_phases;
        BatchedRun BR=run_batched(L,C,SP,Ft,Fdeg,Fpad,reuse_small?&small_cache:nullptr,
                                  attempted?&relocation:nullptr);
        if(attempted) {
            s4_oracle_drain(red); // an unchecked probe cannot authorize NF or replay
            const double end=now_s();probe_seconds=end-t0;
            replayed=BR.tail.nonunit || !BR.tail.factors.empty();
            if(replayed) {
                auto bounds=BR.phase_boundaries;bounds.main_begin=t0;bounds.main_end=end;
                probe_phases=ecm_stage2::timing::partition(bounds);
                if(!probe_phases.complete){std::fprintf(stderr,"FATAL: incomplete relocation probe timing\n");std::exit(3);}
                final_begin=end;
                stage2_log::print(stage2_log::phases,"stage2_relocation_replay: reason=hit_or_nonunit first=1 last=%llu\n",relocation.last);
                BatchedRun original=run_batched(L,C,SP,Ft,Fdeg,Fpad,reuse_small?&small_cache:nullptr);
                s4_oracle_drain(red);replay_seconds=now_s()-end;
                original.t_giant+=BR.t_giant;original.t_gtrees+=BR.t_gtrees;original.t_fold+=BR.t_fold;
                original.t_descent+=BR.t_descent;original.t_inv+=BR.t_inv;original.t_accum+=BR.t_accum;
                BR=std::move(original);
            }
        } else if(replayed)replay_seconds=now_s()-t0;
        stage2_log::phase("Validate arithmetic and finalize factors");
        s4_oracle_drain(red);    /* validation must finish BEFORE elapsed and success */
        /* merge the factors that degenerate BABY points revealed (objective 3): each already
           divides N by construction, and they are deduplicated against what the naming stage
           found, so the reported fact set cannot change for a shape where the naming stage
           already reaches them. */
        for (const std::string &s : baby_deg)
            if (std::find(BR.tail.factors.begin(), BR.tail.factors.end(), s) ==
                BR.tail.factors.end())
                BR.tail.factors.push_back(s);
        if (saved_result) *saved_result = BR.tail;
        const double el = now_s() - t0;
        if(replayed && !attempted)replay_seconds=el; // early nonunit: include the mandatory oracle drain
        if(saved_result) {
            auto &m=saved_result->metrics;
            m.d=D;m.p=P_baby;m.giant_points=BR.giant_points;
            m.giant_first=BR.giant_first;m.giant_last=BR.giant_last;
            m.relocation=g_relocation_enabled;m.relocation_replayed=replayed;
            m.relocation_points=attempted?relocation.points:0;m.relocation_prime=relocation.missing;
            m.relocation_probe_seconds=probe_seconds;m.relocation_replay_seconds=replay_seconds;
            m.init_seconds=stage2_init_seconds;m.main_seconds=el;m.total_seconds=stage2_init_seconds+el;
            m.giant_seconds=BR.t_giant;m.gtrees_seconds=BR.t_gtrees;m.fold_seconds=BR.t_fold;
            m.descent_seconds=BR.t_descent;m.inverse_seconds=BR.t_inv;m.accum_seconds=BR.t_accum;
            auto boundaries=BR.phase_boundaries;
            boundaries.shape=phase_boundaries.shape;boundaries.init_begin=phase_boundaries.init_begin;
            boundaries.baby_begin=phase_boundaries.baby_begin;boundaries.ftree_begin=phase_boundaries.ftree_begin;
            boundaries.init_end=phase_boundaries.init_end;boundaries.main_begin=final_begin;boundaries.main_end=t0+el;
            m.phases=ecm_stage2::timing::partition(boundaries);
            if(replayed && attempted) {
                m.phases.complete=m.phases.complete && probe_phases.complete;
                for(size_t i=4;i<ecm_stage2::timing::count;++i)m.phases.seconds[i]+=probe_phases.seconds[i];
            }
            m.clean=!stage2_extra_fixtures;
            m.fold_resident=BR.fold_device.enabled;
        }
        BR.t_return_finalize = t0 + el - BR.t_post_end;
        BR.t_post_loop += BR.t_return_finalize;
        std::string fs2, ps2;
        for (size_t i = 0; i < BR.tail.factors.size(); ++i) { if (i) fs2 += ","; fs2 += BR.tail.factors[i]; }
        for (size_t i = 0; i < BR.tail.hit_primes.size(); ++i) { if (i) ps2 += ","; ps2 += std::to_string(BR.tail.hit_primes[i]); }
        stage2_log::print(stage2_log::curve, "stage2: algorithm=tree_gpu_batched curves=1 hits=%llu bad_factors=%llu "
                    "factors=%s hit_primes=%s elapsed=%.2f\n", BR.tail.hits, BR.tail.bad_factors,
                    fs2.c_str(), ps2.c_str(), el);
        stage2_log::print(stage2_log::curve, "stage2_full_wall: curve=%d shape=%.6f init=%.6f main=%.6f total=%.6f "
                    "fixture=%.6f clean=%d (init includes baby/F-tree and mandatory selftests; "
                    "main includes GCD, naming and oracle drain; Stage1 excluded; init shared)\n",
                    1,stage2_shape_seconds,stage2_init_seconds,el,stage2_init_seconds+el,
                    stage2_fixture_seconds,(int)(!stage2_extra_fixtures));
        L.print_ntt_lengths();
        if(g_relocation_enabled)stage2_log::print(stage2_log::curve,
            "stage2_relocation: active=%d first=%llu last=%llu probe_points=%llu missing=%llu replayed=%d "
            "probe=%.6f replay=%.6f main=%.6f final_first=%llu final_points=%llu reason=%s\n",
            (int)attempted,relocation.first,relocation.last,attempted?relocation.points:0ull,relocation.missing,
            (int)replayed,probe_seconds,replay_seconds,el,BR.giant_first,BR.giant_points,
            baby_nonunit?"baby_nonunit":replayed?"hit_or_nonunit":attempted?"clean_nf":"no_reduction");
        stage2_log::print(stage2_log::debug, "real_batched_foldflat: enabled=%d folds=%llu muls=%llu sub_coeffs=%llu peak_bytes=%llu prepare=%.6f multiply=%.6f subtract=%.6f bridge=%.6f\n",
            (int)BR.fold_flat.enabled,BR.fold_flat.folds,BR.fold_flat.muls,BR.fold_flat.sub_coeffs,BR.fold_flat.peak_bytes,
            BR.fold_flat.t_prepare,BR.fold_flat.t_multiply,BR.fold_flat.t_subtract,BR.fold_flat.t_bridge);
        const auto &fd=BR.fold_device;
        stage2_log::print(stage2_log::phases, "real_batched_folddevice: requested=%d enabled=%d fallback=%s folds=%llu muls=%llu sub_coeffs=%llu peak_bytes=%llu h2d_bytes=%llu d2h_bytes=%llu avoided_h2d_bytes=%llu avoided_d2h_bytes=%llu checked_words=%llu setup=%.6f upload=%.6f reverse=%.6f subtract=%.6f readback=%.6f reuse=%u layout_bytes=%llu saved_bytes=%llu\n",
            (int)fd.requested,(int)fd.enabled,fd.fallback.c_str(),fd.folds,fd.muls,fd.sub_coeffs,fd.peak_bytes,
            fd.h2d_bytes,fd.d2h_bytes,fd.avoided_h2d_bytes,fd.avoided_d2h_bytes,fd.checked_words,
            fd.t_setup,fd.t_upload,fd.t_reverse,fd.t_subtract,fd.t_readback,fd.reuse,fd.layout_bytes,fd.saved_bytes);
        stage2_log::print(stage2_log::phases, "real_batched_rootfold: requested=%d trees=%llu words=%llu avoided_h2d_bytes=%llu avoided_d2h_bytes=%llu checked_words=%llu digest_words=%llu digest_sum=%016llx digest_xor=%016llx digest_kind=mixsum_xor_v1 t_handoff=%.6f\n",
            (int)fd.root_requested,fd.root_device_trees,fd.root_device_words,fd.root_h2d_avoided,fd.root_d2h_avoided,
            fd.root_checked_words,fd.root_digest_words,fd.root_digest[0],fd.root_digest[1],fd.root_t_handoff);
        stage2_log::print(stage2_log::debug, "real_batched_gfinv: enabled=%d requests=%llu cache_hits=%llu groups=%llu segments=%llu group_attempts=%llu group_failures=%llu individual_attempts=%llu good=%llu nonunits=%llu scratch_peak_bytes=%llu t_prepare=%.6f\n",
            (int)g_gfinv_batch,g_gfinv.requests,g_gfinv.cache_hits,g_gfinv.groups,g_gfinv.segments,
            g_gfinv.group_attempts,g_gfinv.group_failures,g_gfinv.individual_attempts,g_gfinv.good,
            g_gfinv.nonunits,g_gfinv.scratch_peak_bytes,g_gfinv.t_prepare);
        if (BR.tail.unnamed_hits)
            stage2_log::print(stage2_log::debug, "stage2_naming: named_hits=%llu unnamed_hits=%llu (the naming budget "
                        "stopped the culprit scan; hits and bad_factors are complete, the "
                        "hit_primes list is not -- NTT_NAME_HITS=1 forces naming)\n",
                        BR.tail.hits - BR.tail.unnamed_hits, BR.tail.unnamed_hits);
        stage2_log::print(stage2_log::debug, "real_batched_shape: P=%llu giant_points=%llu num_poly_g=%llu loops=%llu "
                    "descent_divmods=%llu\n", BR.P, BR.giant_points, BR.num_poly_g, BR.loops,
                    BR.descent_divmods);
        stage2_log::print(stage2_log::debug, "real_batched_gdevice: enabled=%d trees=%llu fallbacks=%llu levels=%llu groups=%llu pairs=%llu copies=%llu leaf_words=%llu root_words=%llu resident_words=%llu trace_words=%llu metadata_words=%llu metadata_peak_bytes=%llu raw_peak_bytes=%llu logical_frontier_peak_bytes=%llu host_staging_peak_bytes=%llu checked_nodes=%llu checked_words=%llu\n",
            (int)g_groot_device,g_gdevice.trees,g_gdevice.fallbacks,g_gdevice.levels,g_gdevice.groups,g_gdevice.pairs,g_gdevice.copies,
            g_gdevice.leaf_words,g_gdevice.root_words,g_gdevice.resident_words,g_gdevice.trace_words,g_gdevice.metadata_words,
            g_gdevice.metadata_peak_bytes,g_gdevice.raw_peak_bytes,g_gdevice.logical_frontier_peak_bytes,
            g_gdevice.host_staging_peak_bytes,g_gdevice.checked_nodes,g_gdevice.checked_words);
        stage2_log::print(stage2_log::debug, "real_batched_gmemory: compact_raw=%d leaf_staging=%d pinned_trees=%llu pinned_slices=%llu pinned_words=%llu pageable_words=%llu leaf_fallbacks=%llu legacy_trees=%llu pinned_borrow_peak_bytes=%llu rawA_peak_bytes=%llu rawB_peak_bytes=%llu\n",
            (int)g_groot_compact_raw,(int)g_groot_leaf_staging,g_gmemory.pinned_trees,g_gmemory.pinned_slices,
            g_gmemory.pinned_words,g_gmemory.pageable_words,g_gmemory.leaf_fallbacks,g_gmemory.legacy_trees,
            g_gmemory.pinned_borrow_peak_bytes,g_gmemory.rawA_peak_bytes,g_gmemory.rawB_peak_bytes);
        stage2_log::print(stage2_log::debug, "real_batched_groot: root_only=%d builds=%llu nodes_released=%llu moves=%llu "
                    "node_peak_bytes=%llu retained_peak_bytes=%llu released_bytes=%llu input_released_bytes=%llu "
                    "root_words=%llu trace=%d root_hash=%016llx t_release=%.6f root_hash_complete=%d (node capacities, not process peak)\n",
                    (int)g_s4_groot_only, g_groot.builds, g_groot.nodes_released, g_groot.passthrough_moves,
                    g_groot.peak_node_bytes, g_groot.peak_retained_bytes, g_groot.released_bytes,
                    g_groot.input_released_bytes, g_groot.root_words, (int)g_s4_carry_trace,
                    g_groot.root_hash, g_groot.t_release,(int)g_groot.root_hash_complete);
        stage2_log::print(stage2_log::debug, "real_batched_cost: poly_muls=%llu operand_bits=%llu f_tree=%llu g_tree=%llu "
                    "fold=%llu descent=%llu inv=%llu total=%llu\n", L.cost.tot_muls(),
                    L.cost.tot_bits(), L.cost.bits[BC_FTREE], L.cost.bits[BC_GTREE],
                    L.cost.bits[BC_FOLD], L.cost.bits[BC_DESCENT], L.cost.bits[BC_FINV],
                    L.cost.tot_bits());
        stage2_log::print(stage2_log::phases, "real_batched_split: giant=%.3f gtrees=%.3f fold=%.3f descent=%.3f inv=%.3f "
                    "accum=%.3f name=%.3f f_tree_incl=%.3f\n", BR.t_giant, BR.t_gtrees, BR.t_fold,
                    BR.t_descent, BR.t_inv, BR.t_accum, BR.t_name, t_f);
        /* THE BOOKS, CLOSED (section 41): pre + loop_wall + post == `el` above, and the only
           part of the loop body that no phase timer owned is the host affine conversion of the
           giant points (`gleaves`).  loop_host = loop_wall - giant - gtrees - fold - gleaves is
           then everything else the host does inside the loop with the device idle. */
        stage2_log::print(stage2_log::phases,"real_batched_prepare: post_fold_to_descent=%.6f gamma=%.6f return_finalize=%.6f\n",BR.t_post_prepare,BR.t_gscale,BR.t_return_finalize);
        stage2_log::print(stage2_log::debug, "real_batched_wall: pre=%.3f loop_wall=%.3f post=%.3f sum=%.3f (elapsed=%.2f) "
                    "| gleaves=%.3f loop_host=%.3f\n", BR.t_pre_loop, BR.t_loop_wall, BR.t_post_loop,
                    BR.t_pre_loop + BR.t_loop_wall + BR.t_post_loop, el, BR.t_gleaves,
                    BR.t_loop_wall - BR.t_giant - BR.t_gtrees - BR.t_fold - BR.t_gleaves);
        stage2_log::print(stage2_log::debug, "real_batched_gleaves: in=%.3f invert=%.3f out=%.3f (us_per_point: in=%.2f "
                    "invert=%.2f out=%.2f) gscale=%.3f s\n", BR.t_gin, BR.t_ginv, BR.t_gout,
                    BR.giant_points ? 1e6 * BR.t_gin / (double)BR.giant_points : 0.0,
                    BR.giant_points ? 1e6 * BR.t_ginv / (double)BR.giant_points : 0.0,
                    BR.giant_points ? 1e6 * BR.t_gout / (double)BR.giant_points : 0.0,
                    BR.t_gscale);
        stage2_log::print(stage2_log::phases,"real_gscale_device: requested=%d enabled=%d coefficients=%llu h2d_bytes=%llu check_d2h_bytes=%llu checked_words=%llu seconds=%.6f fallback=%s\n",
            (int)BR.gscale.requested,(int)BR.gscale.enabled,BR.gscale.coefficients,BR.gscale.h2d_bytes,
            BR.gscale.check_d2h_bytes,BR.gscale.checked_words,BR.gscale.seconds,BR.gscale.fallback);
        /* section 42: the projective leaves and the segment products that cover them must match
           EXACTLY (asserted in run_batched); the number is reported so a drift is visible */
        stage2_log::print(stage2_log::debug, "real_batched_projective: leaves=%llu gamma_points=%llu segments=%llu "
                    "affine_fallback_points=%llu\n", BR.proj_points, BR.proj_gamma_points,
                    BR.proj_segments, BR.proj_fallbacks);
        stage2_log::print(stage2_log::phases, "real_giant_seed: enabled=%d exact_segments=%d chunks=%llu points=%llu avoided_d2h_bytes=%llu avoided_h2d_bytes=%llu avoided_cpu_modmuls=%llu avoided_montmuls=%llu checked_words=%llu segments=%llu segment_checks=%llu segment_fix_muls=%llu fix_table_peak_bytes=%llu\n",
                    (int)g_giant_seed_device,(int)g_gfinv_seg_exact,g_giant_seed.chunks,g_giant_seed.points,
                    g_giant_seed.avoided_d2h_bytes,g_giant_seed.avoided_h2d_bytes,g_giant_seed.avoided_cpu_modmuls,
                    g_giant_seed.avoided_montmuls,g_giant_seed.checked_words,g_giant_seed.segments,
                    g_giant_seed.segment_checks,g_giant_seed.segment_fix_muls,g_giant_seed.fix_table_peak_bytes);
        stage2_log::print(stage2_log::debug, "real_giant_seed_pair: requested=%d base_builds=%llu base_nonunits=%llu chunks=%llu paired_ladders=%llu scalar_h2d_avoided=%llu base_d2h_bytes=%llu\n",
                    (int)g_giant_seed_pair,g_giant_seed.base_builds,g_giant_seed.base_nonunits,
                    g_giant_seed.paired_chunks,g_giant_seed.paired_ladders,
                    g_giant_seed.scalar_h2d_avoided,g_giant_seed.base_d2h_bytes);
        stage2_log::print(stage2_log::phases, "real_giant_base: cpu_requested=%d cpu_builds=%llu gpu_builds=%llu h2d_bytes=%llu checked_words=%llu cpu_seconds=%.6f build_seconds=%.6f\n",
                    (int)g_giant_base_cpu,g_giant_seed.base_cpu_builds,g_giant_seed.base_gpu_builds,
                    g_giant_seed.base_h2d_bytes,g_giant_seed.base_checked_words,
                    g_giant_seed.base_cpu_seconds,g_giant_seed.base_build_seconds);
        stage2_log::print(stage2_log::debug, "real_giant_chain: chunks=%llu seed_points=%llu chunks_per_ladder=%llu\n",
                    BR.giant_chain_chunks, BR.giant_seed_points,
                    BR.giant_chain_chunks ? 0ull : 1ull);
        /* objective 3: giant points that are the IDENTITY modulo a factor of N -- the hit made
           explicit.  Their gcd(Z, N) was recorded as a factor (credited to no hit). */
        stage2_log::print(stage2_log::debug, "real_giant_degenerate: points=%llu\n", BR.giant_degenerate);
        stage2_log::print(stage2_log::debug, "real_batched_breakdown: wall=%.2f ntt_calls=%llu ntt_launches=%llu "
                    "ntt_seconds=%.3f (%.1f%%) arena_mb=%.1f arena_overflow=%llu\n", el,
                    L.ntt_calls - nb, L.ntt_launches - nl, L.ntt_seconds - ns,
                    el > 0 ? 100.0 * (L.ntt_seconds - ns) / el : 0.0, arena.mb(),
                    BR.arena_overflow);
        arena.print_workspace_stats();
        /* WHERE the NTT-attributed time goes, PER CALL (objective 4).  The phase columns are only
           filled when NTT_HOST_BREAK=1 (two clock reads per phase); without it they print 0 and
           only per_call is meaningful.  The point of the line: `ntt_seconds/ntt_calls` at the real
           shape is ~165 us per call, while the GPU phases measured on the frozen shape are ~50 us,
           so the rest is wrapper/host work -- and these columns say which part. */
        if (L.ntt_calls > nb) {
            const double inv_c = 1.0 / (double)(L.ntt_calls - nb);
            stage2_log::print(stage2_log::debug, "real_batched_ntt_usecall: per_call=%.1f setup=%.1f pack=%.1f maxcoeff=%.1f "
                        "h2d=%.1f fwd=%.1f inv=%.1f carry_slot_out=%.1f d2h=%.1f extract=%.1f "
                        "xcheck=%.1f hostside=%.1f | calls=%llu\n",
                        inv_c * (L.ntt_seconds - ns) * 1e6, inv_c * (L.t_setup - st0) * 1e6,
                        inv_c * (L.t_pack - pk0) * 1e6, inv_c * (L.t_maxc - mc0) * 1e6,
                        inv_c * (L.t_h2d - h20) * 1e6, inv_c * (L.t_fwd - fw0) * 1e6,
                        inv_c * (L.t_inv - iv0) * 1e6, inv_c * (L.t_slot - sl0) * 1e6,
                        inv_c * (L.t_d2h - d20) * 1e6, inv_c * (L.t_ext - ex0) * 1e6,
                        inv_c * (L.t_xchk - xc0) * 1e6,
                        inv_c * ((L.t_setup - st0) + (L.t_pack - pk0) + (L.t_maxc - mc0) +
                                 (L.t_h2d - h20) + (L.t_d2h - d20) + (L.t_ext - ex0) +
                                 (L.t_xchk - xc0)) * 1e6,
                        L.ntt_calls - nb);
            /* SECTION 29: the deferred carry check, counted so a test can prove it RAN rather
               than merely that it did no harm -- the whole point of the change is that the
               per-chunk readback disappears, and a run that silently stopped checking would look
               identical in the timing. */
            stage2_log::print(stage2_log::debug, "real_batched_carrydefer: chunks_deferred=%llu finishes=%llu "
                        "deferred_slices=%llu batch_enabled=%d checked_chunks=%llu max_group=%llu\n",
                        g_defer_chunks, g_defer_finishes, g_defer_slices, (int)g_s4_carry_batch,
                        g_defer_checked_chunks, g_defer_max_group);
            stage2_log::print(stage2_log::debug, "real_batched_carrytrace: enabled=%d words=%llu signature=%016llx\n",
                        (int)g_s4_carry_trace, g_carry_output_words, g_carry_output_hash);
            stage2_log::print(stage2_log::debug, "real_batched_outputwindow: enabled=%d calls=%llu source_coeffs=%llu reduced_coeffs=%llu "
                        "returned_coeffs=%llu skipped_coeffs=%llu d2h_words=%llu device_peak_bytes=%llu "
                        "pinned_peak_bytes=%llu (all S4 calls incl F-tree; shape/carry full)\n",
                        (int)g_s4_output_window,g_output_window.calls,g_output_window.source_coeffs,
                        g_output_window.reduced_coeffs,g_output_window.returned_coeffs,
                        g_output_window.skipped_coeffs,g_output_window.d2h_words,
                        g_output_window.device_peak_bytes,g_output_window.pinned_peak_bytes);
            stage2_log::print(stage2_log::debug, "real_batched_chunkoutput: enabled=%d calls=%llu reused_calls=%llu whole_calls=%llu "
                        "multi_chunk_reused=%llu reused_chunks=%llu legacy_calls=%llu grows=%llu "
                        "request_peak_bytes=%llu whole_peak_bytes=%llu retained_peak_bytes=%llu\n",
                        (int)g_s4_chunk_output,g_chunk_output.calls,g_chunk_output.reused_calls,
                        g_chunk_output.whole_calls,g_chunk_output.multi_chunk_reused,g_chunk_output.reused_chunks,
                        g_chunk_output.legacy_calls,g_chunk_output.grows,g_chunk_output.request_peak_bytes,
                        g_chunk_output.whole_peak_bytes,g_chunk_output.retained_peak_bytes);
            stage2_log::print(stage2_log::debug, "real_batched_finalreadback: enabled=%d calls=%llu copied_words=%llu "
                        "avoided_words=%llu host_peak_bytes=%llu t_copy=%.6f "
                        "(all S4 calls incl F-tree; additional to chunk coeffback)\n",
                        (int)g_s4_final_readback,g_final_readback.calls,g_final_readback.copied_words,
                        g_final_readback.avoided_words,g_final_readback.host_peak_bytes,g_final_readback.t_copy);
            if (L.s4) stage2_log::print(stage2_log::debug, "real_batched_input: direct_enabled=%d direct_chunks=%llu copied_chunks=%llu "
                        "d2d_bytes=%llu avoided_bytes=%llu packed_peak_bytes=%llu "
                        "temp_peak_bytes=%llu temp_current_bytes=%llu pack_host=%.6f copy_host=%.6f "
                        "(all S4 device-packed calls)\n",
                        (int)g_s4_pack_direct, L.s4->input_direct, L.s4->input_copied,
                        L.s4->input_d2d_bytes, L.s4->input_avoided_bytes, L.s4->packed_peak_bytes,
                        L.s4->temp_peak_bytes, 16ull * L.s4->d_pack_cap,
                        L.s4->t_packdev, L.s4->t_input_copy_host);
            stage2_log::print(stage2_log::debug, "real_batched_carrytime: group_readback=%.6f chunk_readback=%.6f "
                        "total=%.6f (all S4 calls, each readback charged once)\n",
                        g_carry_group_readback, g_carry_chunk_readback,
                        g_carry_group_readback + g_carry_chunk_readback);
            stage2_log::print(stage2_log::debug, "real_batched_flatinput: direct_enabled=%d calls=%llu borrowed=%llu padded=%llu "
                        "alias_clones=%llu copy_bytes=%llu zero_bytes=%llu avoided_copy_bytes=%llu "
                        "avoided_zero_bytes=%llu temp_peak_bytes=%llu control_peak_bytes=%llu t_prepare=%.6f "
                        "(all flat_mul_batch calls, host padding only)\n", (int)g_s4_flat_direct,
                        g_flat_input.calls, g_flat_input.borrowed, g_flat_input.padded, g_flat_input.alias_clones,
                        g_flat_input.copy_bytes, g_flat_input.zero_bytes, g_flat_input.avoided_copy_bytes,
                        g_flat_input.avoided_zero_bytes, g_flat_input.temp_peak_bytes,
                        g_flat_input.control_peak_bytes, g_flat_input.t_prepare);
            /* SECTION 30: the pinned/async path, counted for the same reason as above -- and the
               fallback count, so a run that silently lost pinned memory cannot look normal */
            stage2_log::print(stage2_log::debug, "real_batched_asyncxfer: raw_async=%llu out_async=%llu fallbacks=%llu "
                        "(async_enabled=%d) raw_reuse_waits=%llu raw_reuse_wait=%.6f "
                        "raw_pinned_bytes=%llu\n",
                        g_pin_raw_used, g_pin_out_used, g_pin_fallbacks, g_s4_async ? 1 : 0,
                        g_pin_raw_waits, g_pin_raw_wait,
                        8ull * (g_pin_raw_cap[0][0] + g_pin_raw_cap[0][1] +
                                g_pin_raw_cap[1][0] + g_pin_raw_cap[1][1]));
            /* OBJECTIVE 4: the one phase the probe cannot see -- the reduced product coming back
               to the host once per chunk in the batched path (section 32). */
            const double gb = (double)(L.d2h_coeff_words - dw0) * 8.0 / 1073741824.0;
            const double tdc = L.t_d2h_coeff - dc0;
            stage2_log::print(stage2_log::debug, "real_batched_coeffback: us_per_call=%.1f total=%.3f s volume=%.2f GB "
                        "effective_GBps=%.2f share_of_ntt=%.1f%%\n",
                        inv_c * tdc * 1e6, tdc, gb, tdc > 0 ? gb / tdc : 0.0,
                        (L.ntt_seconds - ns) > 0 ? 100.0 * tdc / (L.ntt_seconds - ns) : 0.0);
            /* and the three phases the batched entry point never had timers for */
            const double thp = L.t_hpack - hp0, tsc = L.t_scan - sc0, th2 = L.t_h2d_batch - hb0;
            const double tck = L.t_check - ck0;
            const double acc = inv_c * (thp + tsc + th2) * 1e6;
            stage2_log::print(stage2_log::debug, "real_batched_hostbatch: us_per_call=%.1f (pack=%.1f scan=%.1f h2d=%.1f) "
                        "totals pack=%.3f scan=%.3f h2d=%.3f s share_of_ntt=%.1f%%\n", acc,
                        inv_c * thp * 1e6, inv_c * tsc * 1e6, inv_c * th2 * 1e6, thp, tsc, th2,
                        (L.ntt_seconds - ns) > 0 ? 100.0 * (thp + tsc + th2) / (L.ntt_seconds - ns)
                                                 : 0.0);
            /* THE CARRY-CONVERGENCE ASSERT (section 34): a whole extra pass over the digit array
               plus a D2H, run on EVERY call and never timed before now. */
            stage2_log::print(stage2_log::debug, "real_batched_carrycheck: us_per_call=%.1f total=%.3f s "
                        "share_of_ntt=%.1f%%\n", inv_c * tck * 1e6, tck,
                        (L.ntt_seconds - ns) > 0 ? 100.0 * tck / (L.ntt_seconds - ns) : 0.0);
            /* WHICH PART of that is the pass and which is the drain (section 41): the kernel's
               own GPU time comes from in-stream events, the readback is the blocking D2H of the
               two 8-byte counters.  A readback far larger than the kernel is pure latency. */
            {
                const double trs = L.t_check_reset - cr0, tke = L.t_check_kernel - ck0b,
                             td2 = L.t_check_d2h - cd0;
                stage2_log::print(stage2_log::debug, "real_batched_carrysplit: reset_us_per_call=%.1f kernel_us_per_call="
                            "%.1f d2h_us_per_call=%.1f | totals reset=%.3f kernel=%.3f "
                            "d2h=%.3f s\n", inv_c * trs * 1e6, inv_c * tke * 1e6, inv_c * td2 * 1e6,
                            trs, tke, td2);
            }
            /* ... and the two remaining pieces of the device entry point: the shape plan
               (choose_cfg re-proving the exactness bound per call) and the device-to-device copy
               of the packed operands into the arena's scratch. */
            const double tpl = L.t_plan - pl0b, toc = L.t_opcopy - oc0;
            const double tho = L.t_hout - ho0;
            stage2_log::print(stage2_log::debug, "real_batched_devpath: plan_us_per_call=%.1f (total=%.3f s) "
                        "opcopy_us_per_call=%.1f (total=%.3f s) hout_us_per_call=%.1f "
                        "(total=%.3f s) share_of_ntt=%.1f%%\n",
                        inv_c * tpl * 1e6, tpl, inv_c * toc * 1e6, toc, inv_c * tho * 1e6, tho,
                        (L.ntt_seconds - ns) > 0 ? 100.0 * (tpl + toc + tho) / (L.ntt_seconds - ns)
                                                 : 0.0);
            /* OBJECTIVE 4, the DEVICE-pack version of the same two items: the raw coefficient
               upload and the packing pass are now on the device, so what is left here is one
               small H2D per chunk plus two kernel launches (section 33). */
            if (L.s4) {
                const double traw = L.s4->t_h2d_raw - ry0;
                const double gbr = (double)(L.s4->raw_words - rw0) * 8.0 / 1073741824.0;
                stage2_log::print(stage2_log::debug, "real_batched_rawupload: us_per_call=%.1f total=%.3f s volume=%.2f GB "
                            "effective_GBps=%.2f pack_launches=%llu share_of_ntt=%.1f%%\n",
                            inv_c * traw * 1e6, traw, gbr, traw > 0 ? gbr / traw : 0.0,
                            L.s4->pack_launches - pl0,
                            (L.ntt_seconds - ns) > 0 ? 100.0 * traw / (L.ntt_seconds - ns) : 0.0);
            }
        }
        if (s4_on) {
            unsigned long long sel_cases = 0, sel_bad = 0, checked = 0, check_bad = 0,
                               full = 0, canon = 0, coeffs = 0;
            double t_red = 0.0, t_red_host = 0.0;
            unsigned long long dt_blocked = 0;
            s4_dbad_resolve(red, /*wait=*/true);   /* section 43: the last readback must be checked */
            s4_oracle_drain(red);
            s4_oracle_report();
            for (S4Reduce::Shape *S : red.shapes) {
                S->dt_flush();     /* the run is over and the stream has drained: resolve the
                                      deferred marks now (section 41) */
                sel_cases += S->selftest_cases; sel_bad += S->selftest_bad;
                checked += S->checked; check_bad += S->check_bad; full += S->full_checks;
                canon += S->canon_bad; coeffs += S->coeffs; t_red += S->t_reduce;
                t_red_host += S->t_reduce_host; dt_blocked += S->dt_blocks;
                stage2_log::print(stage2_log::debug, "s4_reduce_stats: P=%llu slot_bits=%llu L=%d nlimb=%d launches=%llu "
                            "coeffs=%llu gmp_checked=%llu gmp_bad=%llu slot_canonical_bad=%llu "
                            "t_reduce=%.3f\n", S->P, S->slot_bits, S->L, S->nlimb, S->calls,
                            S->coeffs, S->checked, S->check_bad, S->canon_bad, S->t_reduce);
                /* section 41: how much of the hook is the KERNEL and how much is the host */
                stage2_log::print(stage2_log::debug, "s4_reduce_split: P=%llu kernel_us_per_call=%.2f "
                            "host_us_per_call=%.2f ring_waits=%llu | t_reduce_host=%.3f s\n",
                            S->P, S->calls ? 1e6 * S->t_reduce / (double)S->calls : 0.0,
                            S->calls ? 1e6 * S->t_reduce_host / (double)S->calls : 0.0,
                            S->dt_blocks, S->t_reduce_host);
                /* the hook's out-of-timer work on the REAL shape too (section 36): the canonical
                   counter's readback and the in-run GMP oracle */
                stage2_log::print(stage2_log::debug, "s4_reduce_hook_tail: P=%llu d2h_bad_us_per_call=%.2f "
                            "sample_us_per_call=%.2f | t_hookd2h=%.3f s t_hooksample=%.3f s\n",
                            S->P, S->calls ? 1e6 * S->t_hookd2h / (double)S->calls : 0.0,
                            S->calls ? 1e6 * S->t_hooksample / (double)S->calls : 0.0,
                            S->t_hookd2h, S->t_hooksample);
            }
            for(int category=0;category<BC_NCAT;++category) {
                const char *names[]={"ftree","gtrees","fold","descent","inverse"};
                const auto &p=s4.phase_requests[category];
                stage2_log::print(stage2_log::debug,
                    "s4_phase_memory: phase=%s groups=%llu pairs=%llu chunks=%llu request_big_peak_bytes=%llu "
                    "request_output_peak_bytes=%llu live_big_peak_bytes=%llu owned_subset_peak_bytes=%llu ntt_payload_peak_bytes=%llu process_peak_complete=0 subset_accounting_version=2\n",
                    names[category],p.groups,p.pairs,p.chunks,p.big_peak_bytes,p.output_peak_bytes,
                    s4.phase_big_live_peak[category],s4.phase_subset_peak[category],s4.phase_ntt_peak[category]);
                if(fuse_env_ull("NTT_REQUEST_AUDIT",0))stage2_log::print(stage2_log::debug,
                    "stage2_request_audit: version=1 phase=%s multiplier=%llu addend=%llu\n",
                    names[category],s4.request_signatures[category].multiplier,s4.request_signatures[category].addend);
            }
            if(fuse_env_ull("NTT_REQUEST_AUDIT",0))stage2_log::print(stage2_log::debug,
                "stage2_request_audit: version=1 phase=all multiplier=%llu addend=%llu\n",
                s4.request_signature.multiplier,s4.request_signature.addend);
            if(fuse_env_ull("NTT_REQUEST_AUDIT",0))stage2_log::print(stage2_log::debug,
                "stage2_s4_program_audit: version=1 multiplier=%llu addend=%llu\n",
                s4.s4_program_signature.multiplier,s4.s4_program_signature.addend);
            stage2_log::print(stage2_log::debug,
                "s4_chunk_plan: workspace_budget=%d calls=%llu changed_calls=%llu chunks=%llu legacy_chunks=%llu "
                "request_peak_bytes=%llu owned_subset_observed_peak_bytes=%llu single_over_budget_calls=%llu "
                "process_peak_complete=0\n",(int)(fuse_env_ull("NTT_S4_WORKSPACE_BUDGET",0)!=0),
                s4.chunk_plan_calls,s4.chunk_plan_changed,s4.chunk_plan_chunks,s4.chunk_plan_legacy_chunks,
                s4.chunk_plan_request_peak,s4.chunk_plan_subset_peak,s4.chunk_plan_over_budget);
            const auto &pz=stage2_pack::stats;
            stage2_log::print(stage2_log::phases,"s4_pack_zero_stats: requested=%d contiguous=%llu gather=%llu reverse=%llu "
                "launches=%llu words=%llu replaced_memset_bytes=%llu avoided_memsets=%llu scope=process\n",
                (int)stage2_pack::enabled(),pz.contiguous,pz.gather,pz.reverse,pz.launches,pz.words,
                8*pz.words,pz.avoided_memsets);
            stage2_log::print(stage2_log::phases, "s4_multiply_stats: enabled=1 launches=%llu poly_muls=%llu "
                        "coeffs_reduced=%llu t_reduce=%.3f t_reduce_host=%.3f ring_waits=%llu "
                        "gmp_selftest_cases=%llu "
                        "gmp_selftest_bad=%llu gmp_checked=%llu gmp_check_bad=%llu "
                        "full_checks=%llu\n", s4.launches, s4.muls, coeffs, t_red, t_red_host,
                        dt_blocked, sel_cases,
                        sel_bad, checked, check_bad, full);
            if(saved_result) {
                saved_result->metrics.selftest_cases=sel_cases;
                saved_result->metrics.checked=checked;
                saved_result->metrics.bad=sel_bad+check_bad+canon;
            }
        }
    }
    mpz_clears(R, tmp, a24, ax, az, nullptr);
    return 0;
}

int ecm_cuda_stage2_run(
    const char *n_hex,
    const char *x_hex,
    uint64_t sigma,
    uint64_t b1,
    uint64_t b2,
    uint64_t d,
    int device,
    void (*report)(const char *, void *),
    void *context,
    unsigned carrier_exponent,
    EcmStage2Metrics *metrics,
    const char *a24_hex)
{
    if(!n_hex || !x_hex || !report || device<0 || b1<2 || b2<=b1 ||
       b2>(uint64_t)INT64_MAX-8192 || (d && (d<6 || d%2)))return 2;
    if(ecm_cuda_stage2_check_configuration())return 2;
    s2g_install_crash_handler();
    s2g_install_terminate();
    ladder_cap_init();
    g_device = device;
    const char *progress = std::getenv("NTT_NO_PROGRESS");
    if (progress && std::atoi(progress)) g_s4_batched_progress = false;
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    Stage2Tail tail;
    const int code = run_real(n_hex, true, sigma, b1, b2, d, d == 0,
                              x_hex, &tail,carrier_exponent,a24_hex);
    if (code || tail.bad_factors) return code ? code : 1;
    if(metrics)*metrics=tail.metrics;
    std::string json = "\"hits\":" + std::to_string(tail.hits) +
        ",\"bad_factors\":" + std::to_string(tail.bad_factors) + ",\"factors\":[";
    for (size_t i = 0; i < tail.factors.size(); ++i) {
        if (i) json += ',';
        json += '"' + tail.factors[i] + '"';
    }
    json += ']';
    report(json.c_str(), context);
    return 0;
}
