#include "host_wait.h"
// Giant batches, fold/descent phases and factor scanning within one curve.
#include "point_launch.h"
#include "engine.h"
#include "options.h"
#include "checks.h"
#include "diagnostics.h"
#include "../planning/ecm_stage2_giant_chunk.h"

BatchedRun run_batched(
    PolyLayer &L,
    const LadderCtx &C,
    const Stage2Params &SP,
    const std::vector<std::vector<unsigned long long>> &Ft,
    const std::vector<size_t> &Fdeg,
    size_t Fpad,
    const SmallPrimeBabyCache *baby_cache,
    const ecm_stage2::GiantRange *requested_range)
{
    const size_t W = L.W;
    const unsigned long long D = SP.D, B1 = SP.B1, B2 = SP.B2;
    const auto range=requested_range?*requested_range:ecm_stage2::giant_range(B2,D,false);
    const unsigned long long imax = range.points;
    const double t_entry = now_s();
    const size_t P = Fdeg[1];                            /* deg F = the baby count = poly_size */
    const bool balanced_groot=ecm_stage2::giant_tree_balanced(W,imax,P);
    if(!L.s4) {
        std::fprintf(stderr,"%s: FATAL: scaled descent needs S4 and cannot be combined with S5\n",NTT_PROBE_NAME);
        std::exit(3);
    }
    const bool fold_flat_enabled=g_fold_flat && L.s4;
    BatchedRun R;
    R.fold_device.root_requested=fold_device_flag("NTT_GROOT_TO_FOLD");
    R.fold_flat.enabled=fold_flat_enabled;
    R.P = P;
    R.giant_points = imax;
    R.giant_first=range.first;R.giant_last=range.last;
    stage2_log::print(stage2_log::debug,
        "groot_layout: balanced=%d words=%llu points=%llu P=%llu full_groups=%llu policy=wide_repeated_v1\n",
        (int)balanced_groot,(unsigned long long)W,imax,(unsigned long long)P,imax/P);
    R.dbg_progress = true;                  /* one line per G-tree batch: gate-able, machine-readable */
    L.cost.S = (long)L.S;                              /* (already set by the caller) */

    mpz_t g, pg;
    mpz_inits(g, pg, nullptr);

    /* every device buffer this engine needs, allocated ONCE for the whole curve */
    S3Workspace ws;
    ws.init(C);

    /* ---- 0. Small primes cannot be reached by iD+-j for i>=1. Reuse only the
       exact baby normalization proof; missing p (including p|D) uses the original ladder. */
    {
        const double ts=now_s();
        const bool requested=small_prime_flag("NTT_SMALL_PRIME_REUSE");
        const bool matched=requested && baby_cache && baby_cache->matches(C,SP);
        const bool check=small_prime_flag("NTT_SMALL_PRIME_CHECK");
        const unsigned long long half=(D<2)?1:D/2;
        std::vector<unsigned long long> smalljs,missing;
        std::vector<size_t> cached;
        size_t cursor=0;
        unsigned long long reused=0,avoided_mont=0,checked=0,bad=0;
        for(unsigned long long p=2;p<=half;++p) {
            if(p<=B1 || p>B2 || !is_prime_u64(p))continue;
            smalljs.push_back(p);
            if(matched)while(cursor<baby_cache->js.size() && baby_cache->js[cursor]<p)++cursor;
            const bool covered=matched && cursor<baby_cache->js.size() && baby_cache->js[cursor]==p;
            cached.push_back(covered?cursor:~size_t(0));
            if(covered) {
                ++reused;unsigned long long bits=0,v=p;while(v){++bits;v>>=1;}
                avoided_mont+=13*bits-6;
            } else missing.push_back(p);
        }
        R.small_primes=smalljs.size();
        std::vector<unsigned long long> sx,sz,cx,cz;
        ladder_points_ws(ws,missing,sx,sz);
        if(check && reused)ladder_points_ws(ws,smalljs,cx,cz);
        mpz_t z,expected;mpz_inits(z,expected,nullptr);
        size_t fallback=0;
        for(size_t k=0;k<smalljs.size();++k) {
            if(cached[k]!=~size_t(0)) {
                baby_cache->gcd_at(cached[k],pg);
                if(check) {
                    words_to_mpz(z,cz.data()+k*W,W);mpz_gcd(expected,z,L.target());
                    if(small_prime_flag("NTT_SMALL_PRIME_TEST_BAD") && checked==0)mpz_add_ui(expected,expected,1);
                    if(mpz_cmp(pg,expected))++bad;
                    ++checked;
                }
            } else {
                words_to_mpz(z,sz.data()+fallback++*W,W);mpz_gcd(pg,z,L.target());
            }
            if(mpz_cmp_ui(pg,1)>0)R.tail.nonunit=true;
            if(mpz_cmp_ui(pg,1)>0 && mpz_cmp(pg,L.target())<0)s3_record(R.tail,pg,smalljs[k],L.target());
        }
        mpz_clears(z,expected,nullptr);
        stage2_log::print(stage2_log::debug, "small_prime_reuse: requested=%d available=%d matched=%d primes=%llu reused=%llu fallback=%llu checked=%llu bad=%llu avoided_montmuls=%llu avoided_h2d_bytes=%llu avoided_d2h_bytes=%llu cache_bytes=%llu elapsed=%.6f\n",
            (int)requested,(int)(baby_cache && baby_cache->ready),(int)matched,R.small_primes,reused,
            (unsigned long long)missing.size(),checked,bad,avoided_mont,reused*8,reused*16*W,
            (unsigned long long)(baby_cache?baby_cache->payload_bytes():0),now_s()-ts);
        if(bad){std::fprintf(stderr,"FATAL: small-prime baby GCD proof mismatch\n");std::exit(3);}
    }

    /* ---- 1. THE GIANT POINTS, ONE BATCH AT A TIME -------------------------------------
       S3 computed ALL of them up front (one ladder launch + a device buffer per point).  At the
       real shape that is imax = 3.4e6 points x 83 words x 2 coordinates = 4.5 GB of device
       memory AND the same again on the host -- "out of memory" before the first G tree.  The
       engine only ever needs ONE batch of P giant points at a time (that is what the G tree is
       built from), so the points are computed per batch and the buffers are reused.  The
       per-batch cost is the same affine_x conversion as before (host GMP, ~10 us per point);
       what disappears is the peak memory. */
    std::vector<unsigned long long> gjs;
    std::vector<unsigned long long> gx, gz;
    /* the per-segment z-products (section 42), one chunk at a time */
    std::vector<unsigned long long> gseg;
    unsigned long long proj_points = 0, proj_gamma_points = 0, proj_segments = 0,
                       proj_fallbacks = 0;
    /* GAMMA^-1: the ONE global correction the projective leaves need.  The G tree of a batch
       whose leaves are projective is GAMMA_b * prod(X - x_i) instead of prod(X - x_i), and the
       folds carry that constant through (H <- (G*H) mod F scales by the same constant), so H
       comes out of the loop as GAMMA * prod_b f_b.  GAMMA = prod of the z values of exactly the
       points whose leaf is projective, i.e. a product of INVERTIBLE elements -- which is why it
       is invertible, and why multiplying H by GAMMA^-1 before the descent reproduces the old
       polynomial EXACTLY (mod N, coefficient by coefficient). */
    mpz_t Ginv;
    mpz_init_set_ui(Ginv, 1);

    /* ---- 2. the outer loop: ceil(I/P) batches, the first one SEEDS H ------------------- */
    R.num_poly_g = (imax + (unsigned long long)P - 1) / (unsigned long long)P;
    if (R.num_poly_g == 0) R.num_poly_g = 1;
    R.loops = R.num_poly_g - 1;

    /* 1/F: the Newton inverse of the REVERSED, MONIC F, to the length a full-size mod-F
       reduction needs (deg T = 2P => k = P+1).  Computed ONCE and reused by every fold --
       the reference's cost model has no such term, so it is charged to its own category. */
    stage2_log::phase("Prepare polynomial inverse");
    stage2_memory::snapshot("before_inverse");
    CPoly finv;
    const CPoly Fpoly = fold_flat_enabled ? CPoly{} : cp_from_flat(Ft[1], Fdeg[1], W);
    const double ti0 = now_s();
    R.phase_boundaries.inverse_begin=ti0;
    if (R.loops > 0) {
        CPoly revF;
        cp_resize(revF, P + 1, W);
        for (size_t i = 0; i <= P; ++i)                 /* revF[i] = F[P-i] (F is monic) */
            std::copy(Ft[1].begin() + (long)((P - i) * W),
                      Ft[1].begin() + (long)((P - i + 1) * W), revF[i].begin());
        L.cat = BC_FINV;                               /* the Newton inverse of rev(F) */
        finv = cp_inv_series(revF, P + 1, L);
        L.cat = -1;
    }
    R.t_inv = now_s() - ti0;
    stage2_memory::snapshot("after_inverse");
    if(L.s4 && fuse_env_ull("NTT_REQUEST_AUDIT",0))
        ecm_stage2::s4_signature_boundary(L.s4->s4_program_signature,ecm_stage2::S4InverseDone);
    std::vector<unsigned long long> Hflat,finvflat;
    if(fold_flat_enabled && !finv.empty()) {
        const double tb=now_s();finvflat=cp_to_flat(finv,W);CPoly{}.swap(finv);
        R.fold_flat.t_bridge+=now_s()-tb;
    }
    FoldDeviceState device_fold;
    if(fold_flat_enabled && !finvflat.empty() && L.s4 &&
       (gscale_flag("NTT_PHASE_TRIM_RAW") || gscale_flag("NTT_PHASE_TRIM_OUTPUT"))) {
        // Newton/F-tree raw inputs are dead. Reclaim them before the owner
        // headroom query; the next G tree will reserve only its own raw shape.
        const double begin=now_s();
        const size_t freed=gscale_flag("NTT_PHASE_TRIM_RAW")?L.s4->raw_release():0;
        const size_t output=gscale_flag("NTT_PHASE_TRIM_OUTPUT")?L.s4->output_release():0;
        stage2_log::print(stage2_log::debug,
            "stage2_phase_trim: raw_released_bytes=%llu output_released_bytes=%llu seconds=%.6f boundary=inverse_to_fold\n",
            (unsigned long long)freed,(unsigned long long)output,now_s()-begin);
    }
    if(fold_flat_enabled && !finvflat.empty())device_fold.init(L,Ft[1],finvflat,R.fold_device,R.fold_flat);
    stage2_memory::snapshot("after_fold_admission");
    CPoly H;
    FTreeStats gs;
    /* Whole G-tree batches, minimum one P. The legacy ceiling may exceed the
       coordinate budget by one batch; the experimental floor stays within it
       unless the mandatory one-P working set itself exceeds the budget. */
    const char *point_kb_env=std::getenv("NTT_GIANT_POINT_BUDGET_KB");
    const unsigned long long point_kb=point_kb_env?std::strtoull(point_kb_env,nullptr,10):262144ull;
    const bool point_floor=fold_device_flag("NTT_GIANT_CHUNK_FLOOR");
    const auto point_plan=stage2_giant_chunk::plan(P,C.nw,
        point_kb<=std::numeric_limits<unsigned long long>::max()/1024?point_kb*1024:0,point_floor);
    if(!point_plan.valid){std::fprintf(stderr,"FATAL: invalid giant point chunk budget/shape\n");std::exit(3);}
    const unsigned long long pts_per_chunk=point_plan.points;
    const unsigned long long point_chunks=imax/pts_per_chunk+(imax%pts_per_chunk!=0);
    stage2_log::print(stage2_log::debug,
        "giant_chunk_plan: floor_requested=%d budget_bytes=%llu P=%llu nw=%llu points=%llu "
        "coordinate_cap_bytes=%llu minimum_over_budget=%d estimated_chunks=%llu\n",
        point_floor,point_kb*1024,(unsigned long long)P,(unsigned long long)C.nw,pts_per_chunk,
        (unsigned long long)point_plan.coordinate_bytes,point_plan.minimum_over_budget,point_chunks);
    unsigned long long actual_point_chunks=0;
    // One continuous interval includes chunk-local destruction between iterations.
    stage2_log::phase("Giant points, G trees and fold");
    const double t_loop_begin = now_s();
    R.phase_boundaries.loop_begin=t_loop_begin;
    R.t_pre_loop = t_loop_begin - t_entry;
    // Rebalance only the optional suffix. Admission retains the original full
    // range/capacity, so a replay never depends on a smaller point budget.
    const char *rebalance_hostpack=std::getenv("NTT_S4_HOSTPACK");
    const auto resident_limit=fuse_env_ull("NTT_DEVICE_GLEAF_MAX_MB",512)*1048576ull;
    const bool rebalance_resident=device_gleaf_flag("NTT_DEVICE_GLEAF") && g_groot_device &&
        g_s4_groot_only && L.s4 && g_s4_pack_direct && !g_s4_final_readback && g_gfinv_seg_exact &&
        !(rebalance_hostpack && std::atoi(rebalance_hostpack)) && 16ull*pts_per_chunk*W<=resident_limit;
    const auto rebalance_min=g_giant_seed_device && rebalance_resident?
        ecm_stage2::giant_chain_minimum(W,fuse_env_ull("NTT_GIANT_CHAIN_MIN",32768),
            fuse_env_ull("NTT_DEVICE_GLEAF_MAX_MB",512)*1048576ull):
        fuse_env_ull("NTT_GIANT_CHAIN_MIN",32768);
    for (unsigned long long c0 = 0; c0 < imax;) {
        ++actual_point_chunks;
        const auto take=ecm_stage2::giant_chunk_take(imax-c0,pts_per_chunk,P,rebalance_min,range.active);
        const unsigned long long c1=c0+take;
        const size_t clo=(size_t)(range.first+c0),chi=(size_t)(range.first+c1-1);
        if(range.active)stage2_log::print(stage2_log::debug,
            "relocation_chunk: offset=%llu first=%llu points=%llu capacity=%llu\n",
            c0,(unsigned long long)clo,take,pts_per_chunk);
        const double tgp = now_s();
        s2g_state("giant ladder chunk: about to launch");   /* the last state on a driver kill */
        /* THE CHAIN BY DEFAULT (see s2g_chain_kernel): one differential addition per giant point
           instead of a full ladder.  NTT_GIANT_LADDER=1 forces the old per-point ladder (kept as
           the oracle), NTT_GIANT_CHAIN_CHECK=1 runs BOTH and compares the affine x values,
           NTT_GIANT_CHAIN_BLOCK=n sets the points per thread (default 256: at the real shape that
           is 786 blocks per chunk, one seed launch, and ~10-30 ms of chain work per chunk). */
        /* NTT_GIANT_CHAIN_BLOCK: points per thead.  The default is deliberately SMALL (64) even
           though a larger block is cheaper in arithmetic: a chain that passes through a point
           with Z not invertible mod N (i.e. a point that is the identity modulo one of N's
           factors -- exactly the hit stage 2 is looking for!) carries that degeneracy in its Z
           into every following point of the block, and those leaves are then garbage.  Measured
           at rung 2: the single degenerate index 3511 (= the residual order of the factor 42089)
           contaminates the rest of whatever block contains it and nothing else -- the mismatch
           count is exactly (block_end - 3511) for every block size tried (9/73/89/73/489 for
           64/128/200/512/1000).  A small block bounds that damage; the seed cost is 2 ladder
           points per block, i.e. ~18 Montgomery multiplications per point at 64 (still ~32x less
           than the ladder's ~574).  Set NTT_GIANT_CHAIN_BLOCK to trade the two. */
        static const unsigned long long chain_block = [] {
            const char *e = std::getenv("NTT_GIANT_CHAIN_BLOCK");
            unsigned long long v = (e && *e) ? std::strtoull(e, nullptr, 10) : 64ull;
            if (v < 4) v = 4;
            if (v > 1u << 20) v = 1u << 20;
            return v;
        }();
        /* Opt-in short-chunk policy: more independent chains, shorter serial
           xADD loops. The global block remains unchanged for large chunks. */
        static const unsigned long long short_chain_block = [] {
            const char *e = std::getenv("NTT_GIANT_CHAIN_SMALL_BLOCK");
            unsigned long long v = (e && *e) ? std::strtoull(e, nullptr, 10) : 0ull;
            if(v && v < 4)v=4;
            return std::min(v,64ull);
        }();
        static const unsigned long long short_chain_max = [] {
            const char *e = std::getenv("NTT_GIANT_CHAIN_SMALL_MAX");
            unsigned long long v = (e && *e) ? std::strtoull(e, nullptr, 10) : 8192ull;
            return std::min(v,1ull<<20);
        }();
        static const bool force_ladder = [] {
            const char *e = std::getenv("NTT_GIANT_LADDER");
            return e && *e && std::atoi(e) != 0;
        }();
        static const bool chain_check = [] {
            const char *e = std::getenv("NTT_GIANT_CHAIN_CHECK");
            return e && *e && std::atoi(e) != 0;
        }();
        /* WHICH PATH, DECIDED BY SIZE -- measured, not guessed.  The chain's arithmetic per point
           is ~70x smaller (one xADD vs a full ladder), but it carries FIXED costs the ladder path
           does not: a second launch, six small uploads and a host-side Montgomery conversion of
           the seeds.  At rung 2 (imax=4331) that makes the chain SLOWER (giant=1.671 s vs
           1.097 s, measured), while at the production shape (imax=3.4e6) the ladder's arithmetic
           is 901 s and the chain's fixed costs are irrelevant.  So: below `chain_min` points per
           chunk use the ladder, above it use the chain.  NTT_GIANT_CHAIN_MIN=0 forces the chain
           (for the checks), NTT_GIANT_CHAIN_MIN=<big> forces the ladder. */
        static const unsigned long long configured_chain_min = [] {
            const char *e = std::getenv("NTT_GIANT_CHAIN_MIN");
            return (e && *e) ? std::strtoull(e, nullptr, 10) : 32768ull;
        }();
        const unsigned long long npts = (unsigned long long)(chi - clo + 1);
        /* the per-SEGMENT z-product grid is shared by the two paths (section 42): the chain
           computes it on the device out of the Montgomery images it already produced, the
           ladder path computes the very same product on the host.  Both are the product of the
           z values AS RETURNED, which is exactly the scale the projective leaves carry. */
        ResidentGiant resident;
        const char *hp=std::getenv("NTT_S4_HOSTPACK"),*mb=std::getenv("NTT_DEVICE_GLEAF_MAX_MB");
        const unsigned long long maxbytes=(mb?std::strtoull(mb,nullptr,10):512ull)*1048576ull;
        const bool requested=device_gleaf_flag("NTT_DEVICE_GLEAF");
        if(requested)++R.device_leaf.requested_chunks;
        const bool device_leaf=requested && g_groot_device && g_s4_groot_only && L.s4 &&
            g_s4_pack_direct && !g_s4_final_readback && !(hp && std::atoi(hp)) && g_gfinv_seg_exact &&
            16ull*npts*W<=maxbytes;
        const auto chain_min=device_leaf && g_giant_seed_device?
            ecm_stage2::giant_chain_minimum(W,configured_chain_min,maxbytes):configured_chain_min;
        const bool wide_tail=g_giant_seed_device && npts<configured_chain_min && chain_min<configured_chain_min;
        if(requested && !device_leaf)++R.device_leaf.fallback_chunks;
        gseg.assign(((size_t)npts + S2G_GFINV_SEG - 1) / S2G_GFINV_SEG * (size_t)C.nw, 0ull);
        const unsigned long long effective_chain_block=ecm_stage2::giant_chain_block(
            npts,chain_block,short_chain_block,short_chain_max);
        bool actual_chain=false;
        if (force_ladder || npts < chain_min) {
            gjs.resize(chi - clo + 1);
            for (size_t i = clo; i <= chi; ++i) gjs[i - clo] = (unsigned long long)i * D;
            if(device_leaf) {
                ws.need_pts(npts);CK(stage2_wait::copy(ws.djs,gjs.data(),npts*8,cudaMemcpyHostToDevice));
                S2G_DISPATCH((int)W,s2g_launch_ladder,(int)W,(int)npts,ws.dn,C.ninv,ws.dqx,ws.dqz,ws.da24,ws.dmone,ws.djs,ws.dx,ws.dz);
                CK(cudaGetLastError());++ws.ladder_calls;ws.ladder_points_total+=npts;
                resident.x=ws.dx;resident.z=ws.dz;resident.n=npts;resident.w=W;
                CK(cudaMalloc(&resident.segments,gseg.size()*8));ws.need_segfix(S2G_GFINV_SEG);
                S2G_DISPATCH((int)W,s2g_launch_segprod,(int)W,npts,S2G_GFINV_SEG,C.ninv,ws.dn,ws.dz,resident.segments,ws.dsegfix);
                CK(cudaGetLastError());
                if(g_gfinv_seg_check) {
                    gx.resize(npts*W);gz.resize(npts*W);
                    CK(stage2_wait::copy(gx.data(),ws.dx,gx.size()*8,cudaMemcpyDeviceToHost));
                    CK(stage2_wait::copy(gz.data(),ws.dz,gz.size()*8,cudaMemcpyDeviceToHost));
                    std::vector<unsigned long long> expected;gfinv_segprod_host(expected,npts,W,gz,L.N);
                    CK(stage2_wait::copy(gseg.data(),resident.segments,gseg.size()*8,cudaMemcpyDeviceToHost));
                    if(expected!=gseg){std::fprintf(stderr,"FATAL: ladder device segment GMP mismatch\n");std::exit(3);}
                } else {std::vector<unsigned long long>().swap(gx);std::vector<unsigned long long>().swap(gz);}
            } else {
                ladder_points_ws(ws,gjs,gx,gz);gfinv_segprod_host(gseg,npts,W,gz,L.N);
            }
        } else {
            unsigned long long seed_points = 0;
            actual_chain=giant_chunk_chain(L, C, ws, D, (unsigned long long)clo, (unsigned long long)chi,
                              effective_chain_block, gx, gz, chain_check, seed_points, &gseg,
                              S2G_GFINV_SEG,device_leaf?&resident:nullptr,wide_tail);
            R.giant_seed_points += seed_points;
            if(actual_chain)++R.giant_chain_chunks;
        }
        R.t_giant += now_s() - tgp;
        if(ws.giant_base_d && !ws.giant_base_unit)R.tail.nonunit=true;
        if(device_leaf) {
            const double t=now_s(),ti=R.device_leaf.t_invert;
            if(wide_tail && actual_chain) {
                // Any nonunit segment must keep the old per-point factor path,
                // including its non-affine placeholders. Retry in the SAME
                // coordinates/segment/group leases and restore logical Gamma.
                mpz_t saved;mpz_init_set(saved,Ginv);
                const auto gamma_before=proj_gamma_points,segments_before=proj_segments;
                const auto bad_before=R.device_leaf.bad_groups;
                resident.prepare(ws,L,gseg,Ginv,proj_gamma_points,proj_segments,R.device_leaf);
                if(R.device_leaf.bad_groups!=bad_before) {
                    mpz_set(Ginv,saved);proj_gamma_points=gamma_before;proj_segments=segments_before;
                    giant_ladder_into(ws,C,D,clo,(size_t)npts,resident.x,resident.z);
                    S2G_DISPATCH((int)W,s2g_launch_segprod,(int)W,npts,S2G_GFINV_SEG,C.ninv,
                        ws.dn,resident.z,resident.segments,ws.dsegfix);
                    CK(cudaGetLastError());
                    if(g_gfinv_seg_check) {
                        gx.resize(npts*W);gz.resize(npts*W);
                        CK(stage2_wait::copy(gx.data(),resident.x,gx.size()*8,cudaMemcpyDeviceToHost));
                        CK(stage2_wait::copy(gz.data(),resident.z,gz.size()*8,cudaMemcpyDeviceToHost));
                        CK(stage2_wait::copy(gseg.data(),resident.segments,gseg.size()*8,cudaMemcpyDeviceToHost));
                        std::vector<unsigned long long> expected;gfinv_segprod_host(expected,npts,W,gz,L.N);
                        if(expected!=gseg){std::fprintf(stderr,"FATAL: wide-tail fallback segment GMP mismatch\n");std::exit(3);}
                    }
                    ++R.device_leaf.retry_chunks;
                    resident.prepare(ws,L,gseg,Ginv,proj_gamma_points,proj_segments,R.device_leaf);
                    actual_chain=false;--R.giant_chain_chunks;
                    stage2_log::print(stage2_log::debug,"wide_tail_ladder_fallback: points=%llu reason=nonunit_segment index_capacity=%llu\n",
                        npts,(unsigned long long)ws.pt_cap);
                }
                mpz_clear(saved);
            } else resident.prepare(ws,L,gseg,Ginv,proj_gamma_points,proj_segments,R.device_leaf);
            if(R.device_leaf.bad_groups)R.tail.nonunit=true;
            R.t_gleaves+=now_s()-t;R.t_ginv+=R.device_leaf.t_invert-ti;
        }
        stage2_log::print(stage2_log::debug, "giant_chain_policy: npts=%llu route=%s base_block=%llu block=%llu short_block=%llu short_max=%llu configured_min=%llu effective_min=%llu words=%llu\n",
            npts,actual_chain?"chain":"ladder",chain_block,actual_chain?effective_chain_block:0ull,
            short_chain_block,short_chain_max,configured_chain_min,chain_min,(unsigned long long)W);
        GfinvBatch segment_inverse(gseg,W,L.target(),g_gfinv_batch);
        /* the G trees of the batches that lie inside this point chunk */
        for (unsigned long long b = c0 / P; b < R.num_poly_g && b * P < c1; ++b) {
        const size_t lo = (size_t)(b * (unsigned long long)P);
        size_t hi = lo + P;
        if (hi > (size_t)imax) hi = (size_t)imax;
        std::vector<std::vector<unsigned long long>> bleaf;
        if(!device_leaf)bleaf.assign(hi-lo,std::vector<unsigned long long>(2*W,0));
        std::map<size_t,std::vector<unsigned long long>> patches;
        auto leaf_at=[&](size_t i)->std::vector<unsigned long long>& {
            if(!device_leaf)return bleaf[i];
            auto &v=patches[i];if(v.empty())v.assign(2*W,0);return v;
        };
        auto coord_at=[&](size_t i,bool z)->const unsigned long long* {
            return device_leaf?resident.coord(i,z):(z?gz:gx).data()+i*W;
        };
        std::vector<size_t> bdeg;
        size_t bpad = 0;
        FTreeStats bs;
        {
            const double tgl0 = now_s();
            mpz_t X, Z, ax, neg, gq;
            mpz_inits(X, Z, ax, neg, gq, nullptr);
            /* ---- THE PROJECTIVE LEAF, AND WHY THE TREE CANNOT CHANGE (section 42) ------------
               x_i = X_i / Z_i mod N used to be one mpz_invert per giant point: MEASURED at 61.5 us
               for the production modulus while the two input conversions and the output conversion
               together are 0.75 us, so 224.8 s of a 621 s curve -- the largest single phase in the
               engine, larger than the whole NTT (234 s).  Section 41 got it to 132.8 s with
               Montgomery's trick on the host; this removes the inversion from the CLEAN path
               entirely.
               The leaf does not have to be monic.  If it is written PROJECTIVELY as
                   [ -X_i , Z_i ]      instead of      [ -x_i , 1 ],
               then the factor is  Z_i*X - X_i = Z_i*(X - x_i), i.e. exactly Z_i times the old
               leaf, and a product tree over a MIXED set of leaves is
                   (prod over the projective ones of Z_i) * prod over ALL of them (X - x_i).
               So a segment whose z-product is INVERTIBLE mod N writes its leaves projectively at
               word level (two word operations, no GMP), and a segment whose product is NOT
               invertible -- it contains one of the ~0.9% DEGENERATE giant points, gcd(Z_i,N) > 1,
               the hit of objective 3 -- falls back to the affine path VERBATIM, degeneracy record
               included.  Gamma = the product of exactly the projective segments' z values is
               therefore a product of invertible elements, and multiplying H by Gamma^-1 once
               before the descent reproduces the old polynomial coefficient for coefficient.
               ONE mpz_invert per SEGMENT, not per point, and that invert is ALSO the Gamma^-1
               factor it contributes -- nothing is computed twice.  The segment products come from
               the device for the chain path (s2g_segprod_kernel, out of the Montgomery images the
               chain already has) and from the host for the ladder path. */
            const size_t SEG = S2G_GFINV_SEG;
            /* plain arrays, not std::vector: mpz_t is an ARRAY type (__mpz_struct[1]) and cannot
               be held in a std::vector */
            mpz_t pv[32 + 1], zv[32];
            for (int s = 0; s <= 32; ++s) mpz_init(pv[s]);
            for (int s = 0; s < 32; ++s) mpz_init(zv[s]);
            mpz_t pinv, zinv, pseg, invp;
            mpz_inits(pinv, zinv, pseg, invp, nullptr);
            std::vector<unsigned long long> w(W, 0ull);
            /* THE SEGMENT GRID IS IN CHUNK-LOCAL INDICES, because that is the grid the device
               built `gseg` on (`giant_chunk_chain` segments [0,SEG), [SEG,2SEG), ... over the
               chunk's own points).  The point indices here are GLOBAL (lo = b*P), so the local
               index is `l - (clo-1)`; using the global one walked `gseg` off its end for every
               point after the first chunk -- silently at 1e11 (the read landed inside the heap)
               and as an access violation at the production shape, which is how it was found.
               `first_touch` keeps Gamma from counting a segment twice when a BATCH boundary
               falls inside it: only the batch that owns the segment's first point adds its
               inverse, and the other half's projective leaves are covered by that one factor. */
            const size_t lbase = (size_t)c0;             /* tree offsets, not absolute giant i */
            const size_t lhi = hi - lbase;
            for (size_t la = lo - lbase; la < lhi;) {
                const size_t sidx = la / SEG;
                const size_t send = (sidx + 1) * SEG;
                const size_t lend = (send < lhi) ? send : lhi;
                const size_t nb = lend - la;
                const bool first_touch = (la == sidx * SEG);
                /* a hard guard, because getting this wrong the first time was SILENT at 1e11 (the
                   out-of-range read landed inside the heap and only Gamma was wrong) and an access
                   violation at the production shape -- an index that must be in range is asserted,
                   not trusted */
                if ((sidx + 1) * W > gseg.size()) {
                    std::fprintf(stderr, "%s: FATAL: giant segment %llu is outside the %llu "
                                         "segment products of this chunk (local index %llu of "
                                         "%llu)\n", NTT_PROBE_NAME, (unsigned long long)sidx,
                                 (unsigned long long)(gseg.size() / W), (unsigned long long)la,
                                 (unsigned long long)lhi);
                    std::exit(3);
                }
                if(device_leaf && resident.good(sidx)) {
                    proj_points+=nb;la=lend;continue; // Gamma counted once by the exact whole group
                }
                const double t1 = now_s();
                words_to_mpz(pseg, &gseg[sidx * W], W);
                const double t2 = now_s();
                bool clean;
                if(g_gfinv_batch)clean=segment_inverse.get(invp,sidx);
                else {
                    ++g_gfinv.requests;++g_gfinv.individual_attempts;
                    clean=mpz_invert(invp,pseg,L.target())!=0;
                }
                const double t3 = now_s();
                R.t_gin += t2 - t1;
                R.t_ginv += t3 - t2;
                if(!clean)R.tail.nonunit=true;
                if (clean) {
                    if (first_touch) {
                        mpz_mul(Ginv, Ginv, invp);
                        mpz_mod(Ginv, Ginv, L.N);
                        /* the points the DEVICE's segment product covers (the last segment of a
                           chunk can be short) */
                        const size_t seglen = ((sidx + 1) * SEG < npts) ? SEG
                                                                        : (npts - sidx * SEG);
                        proj_gamma_points += seglen;
                        ++proj_segments;
                    }
                    if(device_leaf)proj_points+=nb;
                    else for (size_t q = la; q < lend; ++q) {
                        words_neg_mod_n(leaf_at(q + lbase - lo), 0, coord_at(q,false), C.hn.data(), W);
                        std::copy(coord_at(q,true), coord_at(q,true) + (long)W,
                                  leaf_at(q + lbase - lo).begin() + (long)W);
                        ++proj_points;
                    }
                    R.t_gout += now_s() - t3;
                    la = lend;
                    continue;
                }
                /* ---- the affine path, VERBATIM (Montgomery's trick inside the segment, and the
                   old per-point path for whatever still refuses to invert) ------------------ */
                bool seg_clean = true;
                mpz_set_ui(pv[0], 1);
                proj_fallbacks += nb;
                for (size_t j = 0; j < nb; ++j) {
                    const size_t q = la + j;
                    words_to_mpz(zv[j], coord_at(q,true), W);
                    if (L.target_zero(zv[j])) { seg_clean = false; break; }
                    mpz_mul(pv[j + 1], pv[j], zv[j]);
                    mpz_mod(pv[j + 1], pv[j + 1], L.N);
                }
                if (seg_clean && mpz_invert(pinv, pv[nb], L.target()) == 0) seg_clean = false;
                if (!seg_clean) {
                    for (size_t j = 0; j < nb; ++j) {
                        const size_t q = la + j, bi = q + lbase - lo;
                        words_to_mpz(X, coord_at(q,false), W);
                        words_to_mpz(Z, coord_at(q,true), W);
                        if (!affine_x_gmp_checked(ax, X, Z, L.target())) {
                            ++R.giant_degenerate;
                            mpz_gcd(gq, Z, L.target());
                            s3_record(R.tail, gq, 0, L.target(), /*count_hit=*/false);
                        }
                        mpz_neg(neg, ax);
                        mpz_mod(neg, neg, L.N);
                        mpz_to_words(w, W, neg);
                        std::copy(w.begin(), w.end(), leaf_at(bi).begin());
                        leaf_at(bi)[W] = 1;
                    }
                    R.t_gout += now_s() - t3;
                    la = lend;
                    continue;
                }
                for (size_t j = nb; j-- > 0;) {
                    const size_t q = la + j, bi = q + lbase - lo;
                    mpz_mul(zinv, pinv, pv[j]);
                    mpz_mod(zinv, zinv, L.N);        /* Z_j^-1 */
                    mpz_mul(pinv, pinv, zv[j]);
                    mpz_mod(pinv, pinv, L.N);        /* inverse of the shorter product */
                    words_to_mpz(X, coord_at(q,false), W);
                    mpz_mul(ax, X, zinv);
                    mpz_mod(ax, ax, L.N);
                    mpz_neg(neg, ax);
                    mpz_mod(neg, neg, L.N);          /* the leaf (X - x_i) = [ -x_i, 1 ] */
                    mpz_to_words(w, W, neg);
                    std::copy(w.begin(), w.end(), leaf_at(bi).begin());
                    leaf_at(bi)[W] = 1;
                }
                R.t_gout += now_s() - t3;
                la = lend;
            }
            for (int s = 0; s <= 32; ++s) mpz_clear(pv[s]);
            for (int s = 0; s < 32; ++s) mpz_clear(zv[s]);
            mpz_clears(pinv, zinv, pseg, invp, nullptr);
            mpz_clears(X, Z, ax, neg, nullptr);
            R.t_gleaves += now_s() - tgl0;
        }
        const double tg0 = now_s();
        std::vector<std::vector<unsigned long long>> gt;
        bool root_handed_off=false;
        std::function<void(const unsigned long long*,size_t)> root_sink;
        if(device_fold.active && R.fold_device.root_requested && g_s4_groot_only) {
            root_sink=[&](const unsigned long long *src,size_t count) {
                device_fold.accept_device_root(src,count,b==0);root_handed_off=true;
            };
        }
        if(device_leaf) {
            auto fill=[&](unsigned long long *out) {
                const double t=now_s();const size_t first=lo-c0,count=hi-lo;
                ecm_launch_s2g_projective_leaf_kernel(EcmCudaLaunch{(unsigned)((count+127)/128),128}, ws.dn,(int)W,resident.x,resident.z,first,count,out);
                CK(cudaGetLastError());
                for(auto it=patches.begin();it!=patches.end();) {
                    const size_t start=it->first;size_t next=start;std::vector<unsigned long long> words;
                    while(it!=patches.end() && it->first==next) {words.insert(words.end(),it->second.begin(),it->second.end());++it;++next;}
                    CK(stage2_wait::copy(out+2*start*W,words.data(),words.size()*8,cudaMemcpyHostToDevice));
                    R.device_leaf.patch_words+=words.size();
                }
                if(device_gleaf_flag("NTT_DEVICE_GLEAF_TEST_BAD")){CK(cudaMemset(out,0xff,8));}
                if(device_gleaf_flag("NTT_DEVICE_GLEAF_CHECK")) {
                    std::vector<unsigned long long> x(count*W),z(count*W),actual(count*2*W),expected(2*W);
                    CK(stage2_wait::copy(x.data(),resident.x+first*W,x.size()*8,cudaMemcpyDeviceToHost));
                    CK(stage2_wait::copy(z.data(),resident.z+first*W,z.size()*8,cudaMemcpyDeviceToHost));
                    CK(stage2_wait::copy(actual.data(),out,actual.size()*8,cudaMemcpyDeviceToHost));
                    for(size_t i=0;i<count;++i) {
                        const auto patch=patches.find(i);
                        if(patch!=patches.end())expected=patch->second;
                        else {words_neg_mod_n(expected,0,x.data()+i*W,C.hn.data(),W);std::copy(z.data()+i*W,z.data()+(i+1)*W,expected.begin()+W);}
                        if(!std::equal(expected.begin(),expected.end(),actual.begin()+2*i*W)) {
                            std::fprintf(stderr,"FATAL: device giant leaf CPU mismatch\n");std::exit(3);
                        }
                    }
                    R.device_leaf.checked_leaf_words+=actual.size();
                }
                ++R.device_leaf.device_trees;R.device_leaf.device_leaf_words+=count*2*W;
                R.device_leaf.avoided_leaf_h2d_bytes+=16ull*count*W;
                R.device_leaf.t_fill+=now_s()-t;
            };
            gt=build_groot_device(L,bleaf,bdeg,bpad,bs,BC_GTREE,hi-lo,fill,root_sink,balanced_groot);
        } else gt=build_groot_select(L,bleaf,bdeg,bpad,bs,BC_GTREE,!g_s4_groot_only,root_sink,balanced_groot);
        if (g_s4_groot_only) {
            const double tr0 = now_s();
            for (const auto &v : bleaf) g_groot.input_released_bytes += 8ull * v.capacity();
            std::vector<std::vector<unsigned long long>>().swap(bleaf);
            bs.t_release += now_s() - tr0;
        }
        R.t_gtrees += now_s() - tg0;
        ++g_groot.builds;
        g_groot.nodes_released += bs.nodes_released;
        g_groot.passthrough_moves += bs.passthrough_moves;
        g_groot.released_bytes += bs.node_released_bytes;
        g_groot.peak_node_bytes = std::max(g_groot.peak_node_bytes, bs.node_peak_bytes);
        g_groot.peak_retained_bytes = std::max(g_groot.peak_retained_bytes, bs.node_retained_bytes);
        g_groot.t_release += bs.t_release;
        g_groot.root_words += (bdeg[1]+1)*W;
        // FNV is complete only when every root is materialized. Resident consumers
        // fingerprint their actual inputs separately on the device.
        if(root_handed_off)g_groot.root_hash_complete=false;
        else for (const auto w : gt[1]) g_groot.root_hash = (g_groot.root_hash ^ w) * 1099511628211ull;
        gs.leaves += bs.leaves;
        gs.padded += bs.padded;
        gs.muls += bs.muls;
        if (b == 0) {
            if(!root_handed_off) {
                if(device_fold.active)device_fold.seed(gt[1]);
                else if(fold_flat_enabled)Hflat=std::move(gt[1]);else H=cp_from_flat(gt[1],bdeg[1],W);
            }
            continue;
        }
        /* ---- the fold: H <- (G*H) mod F, three full-size multiplies ---- */
        const double tf0 = now_s();
        if(root_handed_off)device_fold.step_loaded(device_fold.gcount);
        else if(device_fold.active)device_fold.step(gt[1]);
        else if(fold_flat_enabled) {
            fold_flat_step(L,gt[1],Hflat,Ft[1],finvflat,R.fold_flat);
        } else {
        const CPoly G = cp_from_flat(gt[1], bdeg[1], W);
        L.cat = BC_FOLD;
        const CPoly T = cp_mul(G, H, L);
        const size_t degT = (T.size() ? T.size() - 1 : 0);
        if (degT < P) { H = T; }                       /* T mod F = T when deg T < deg F */
        else {
            const size_t k = degT - P + 1;             /* <= P+1 */
            CPoly ra, rbi;
            cp_resize(ra, k, W);                       /* rev_k(T) */
            for (size_t i = 0; i < k; ++i) ra[i] = T[degT - i];
            cp_resize(rbi, k, W);                      /* the cached inverse, truncated to k */
            for (size_t i = 0; i < k; ++i) rbi[i] = finv[i];
            CPoly qrev = cp_mul(ra, rbi, L, k);
            cp_resize(qrev, k, W);
            CPoly q;
            cp_resize(q, k, W);
            for (size_t i = 0; i < k; ++i) q[i] = qrev[k - 1 - i];
            const CPoly qb = cp_mul(q, Fpoly, L, P);
            cp_resize(H, P, W);                        /* r = T - q*F has degree < P */
            for (size_t i = 0; i < P; ++i)
                cp_coeff_sub(H[i], T[i], (i < qb.size()) ? qb[i] : cp_zero(W), L.N, W);
            cp_trim(H);
        }
        }
        L.cat = -1;
        R.t_fold += now_s() - tf0;
        /* the fold loop is minutes long at the real shape and used to print NOTHING between
           `exactness:` and the final summary, which is why a hung or crashed real-shape run
           could not be told from a slow one.  One line per batch: progress, and the two
           numbers that say whether the arena is doing its job (fuse_reuse must grow,
           arena_overflow must stay 0 -- a nonzero overflow means per-call cudaMalloc is back
           in the hot path, which is exactly the 82% of section 18.2). */
        if (R.dbg_progress) {
            const double done = R.t_giant + R.t_gtrees + R.t_fold;
            const size_t arena_mb = (L.arena ? L.arena->mb() : 0);
            stage2_log::print(stage2_log::batches, "batched_progress: batch=%llu/%llu (%.1f%%) t=%.1f s left=%.1f s "
                        "giant=%.1f gtrees=%.1f fold=%.1f | ntt_calls=%llu launches=%llu "
                        "arena_mb=%llu fuse_builds=%llu fuse_reuse=%llu overflow=%llu\n",
                        b + 1, R.num_poly_g, 100.0 * (double)(b + 1) / (double)R.num_poly_g,
                        done, ((double)R.num_poly_g / (double)(b + 1) - 1.0) * done,
                        R.t_giant, R.t_gtrees, R.t_fold, L.ntt_calls, L.ntt_launches,
                        (unsigned long long)arena_mb,
                        L.arena ? L.arena->fuse_builds : 0ull,
                        L.arena ? L.arena->fuse_hits : 0ull,
                        L.arena ? L.arena->overflow : 0ull);
        }
        }
        c0=c1;
    }
    R.t_post_loop=now_s();
    R.phase_boundaries.loop_end=R.t_post_loop;
    stage2_log::print(stage2_log::debug,
        "giant_chunk_done: chunks=%llu chain_chunks=%llu ladder_chunks=%llu points=%llu\n",
        actual_point_chunks,R.giant_chain_chunks,actual_point_chunks-R.giant_chain_chunks,imax);
    R.t_loop_wall = R.t_post_loop - t_loop_begin;
    stage2_log::phase("Normalize fold and descend F tree");
    stage2_memory::snapshot("after_giant_loop");
    ScaledRootDeviceStats root_device;
    ScaledFrontierDevice frontier;
    frontier.requested=true;
    const bool frontier_check=gscale_flag("NTT_SCALED_FRONTIER_CHECK");
    if(frontier_check && P>512){std::fprintf(stderr,"FATAL: scaled frontier GMP check limited to 512 leaves\n");std::exit(3);}
    if(gscale_flag("NTT_SCALED_FRONTIER_TEST_BAD") && !(frontier_check || g_scaled_check)) {
        std::fprintf(stderr,"FATAL: scaled frontier poison requires GMP check\n");std::exit(3);
    }
    std::vector<unsigned long long> prepared_root;
    root_device.requested=true;
    const bool root_check=gscale_flag("NTT_SCALED_ROOT_CHECK"),root_bad=gscale_flag("NTT_SCALED_ROOT_TEST_BAD");
    if(root_bad && !root_check){std::fprintf(stderr,"FATAL: scaled root poison requires full root check\n");std::exit(3);}
    R.gscale.requested=true;
    const bool gscale_check=gscale_flag("NTT_GSCALE_DEVICE_CHECK"),gscale_bad=gscale_flag("NTT_GSCALE_DEVICE_TEST_BAD");
    if(gscale_bad && !gscale_check){std::fprintf(stderr,"FATAL: Gamma poison requires full GMP check\n");std::exit(3);}
    if(R.gscale.requested && device_fold.active && device_fold.hcount && mpz_cmp_ui(Ginv,1)!=0) {
        device_fold.scale(Ginv,R.gscale,gscale_check,gscale_bad);R.t_gscale=R.gscale.seconds;
    } else if(R.gscale.requested)R.gscale.fallback=!device_fold.active?"owner":!device_fold.hcount?"empty":"unit";
    if(device_fold.active && device_fold.hcount && device_fold.hcount<=P && (R.gscale.enabled || mpz_cmp_ui(Ginv,1)==0)) {
        if(L.s4 && fuse_env_ull("NTT_REQUEST_AUDIT",0))
            ecm_stage2::s4_signature_boundary(L.s4->s4_program_signature,ecm_stage2::S4FoldDone);
        if(gscale_flag("NTT_PHASE_TRIM_RAW") || gscale_flag("NTT_PHASE_TRIM_OUTPUT")) {
            const double begin=now_s();
            const size_t freed=gscale_flag("NTT_PHASE_TRIM_RAW")?L.s4->raw_release():0;
            const size_t output=gscale_flag("NTT_PHASE_TRIM_OUTPUT")?L.s4->output_release():0;
            stage2_log::print(stage2_log::debug,
                "stage2_phase_trim: raw_released_bytes=%llu output_released_bytes=%llu seconds=%.6f boundary=fold_to_descent\n",
                (unsigned long long)freed,(unsigned long long)output,now_s()-begin);
        }
        frontier.init(device_fold);
        stage2_memory::snapshot("after_frontier_admission");
        device_fold.scaled_root(prepared_root,root_device,root_check,root_bad,frontier.enabled);
        const char *dc=std::getenv("NTT_S4_DESCENT_CHECK");
        if(g_scaled_check || frontier_check || (dc && std::atoi(dc)))Hflat=device_fold.read(device_fold.h,device_fold.hcount);
        device_fold.read_digest();if(!frontier.enabled)device_fold.release();
    } else {
        root_device.fallback=!device_fold.active?"owner":device_fold.hcount>P?"root_division":!device_fold.hcount?"empty":"gamma_cpu";
        frontier.fallback="root_unavailable";
        if(device_fold.active)device_fold.finish(Hflat);
    }
    stage2_log::print(stage2_log::phases, "scaled_root_device: requested=%d enabled=%d coefficients=%llu h2d_bytes=%llu d2h_bytes=%llu check_d2h_bytes=%llu checked_words=%llu avoided_h2d_bytes=%llu avoided_h_readback_bytes=%llu seconds=%.6f fallback=%s\n",
        (int)root_device.requested,(int)root_device.enabled,(unsigned long long)root_device.coefficients,
        root_device.h2d_bytes,root_device.d2h_bytes,root_device.check_d2h_bytes,root_device.checked_words,
        root_device.avoided_h2d_bytes,root_device.avoided_h_readback_bytes,root_device.seconds,root_device.fallback);
    if(fold_flat_enabled && !Hflat.empty()) {
        const double tb=now_s();H=cp_from_flat(Hflat,Hflat.size()/W-1,W);
        if(!root_device.enabled && !finvflat.empty())finv=cp_from_flat(finvflat,finvflat.size()/W-1,W);
        std::vector<unsigned long long>().swap(Hflat);std::vector<unsigned long long>().swap(finvflat);
        R.fold_flat.t_bridge+=now_s()-tb;
    }
    if(root_device.enabled)std::vector<unsigned long long>().swap(finvflat);
    /* ---- UNDO THE PROJECTIVE SCALE (section 42) ------------------------------------------
       The projective leaves multiplied the tree by Gamma, and the fold carries a constant
       straight through (H <- (G*H) mod F scales by the same constant), so H left the loop as
       Gamma * prod_b f_b mod F.  ONE pass over H's coefficients by Gamma^-1 restores the exact
       polynomial the old monic leaves produced -- not approximately: every step of the tree is
       reduced mod N coefficient by coefficient, so the scaling is exact in that ring, and
       gcd(v, N) is unchanged by an invertible factor either way. */
    if (!R.gscale.enabled && mpz_cmp_ui(Ginv, 1) != 0 && !H.empty()) {
        const double tg0 = now_s();
        mpz_t c;
        mpz_init(c);
        for (size_t i = 0; i < H.size(); ++i) {
            words_to_mpz(c, H[i].data(), W);
            mpz_mul(c, c, Ginv);
            mpz_mod(c, c, L.N);
            mpz_to_words(H[i], W, c);
        }
        mpz_clear(c);
        R.t_gscale = now_s() - tg0;
    }
    /* THE INVARIANT (section 42): one accumulated segment product per projective leaf set, no
       more and no less.  The factor set cannot see a wrong Gamma (any INVERTIBLE one gives the
       same gcds), so this is asserted instead of trusted. */
    if (proj_points != proj_gamma_points) {
        std::fprintf(stderr, "%s: FATAL: the projective scale covers %llu points but %llu leaves "
                             "were written projectively -- Gamma is not the product of exactly "
                             "those z values\n", NTT_PROBE_NAME,
                     (unsigned long long)proj_gamma_points, (unsigned long long)proj_points);
        std::exit(3);
    }
    {
        mpz_t t;
        mpz_init(t);
        if (mpz_invert(t, Ginv, L.target()) == 0) {
            std::fprintf(stderr, "%s: FATAL: the projective scale Gamma is not invertible mod N "
                                 "-- H cannot be unscaled\n", NTT_PROBE_NAME);
            std::exit(3);
        }
        mpz_clear(t);
    }
    R.proj_points = proj_points;
    R.proj_gamma_points = proj_gamma_points;
    R.proj_segments = proj_segments;
    R.proj_fallbacks = proj_fallbacks;
    mpz_clear(Ginv);
    R.t_post_prepare = now_s()-R.t_post_loop;

    /* ---- 3. ONE descent of H against the F tree: H(x_j) at every baby point ------------ */
    const double td0 = now_s();
    stage2_log::print(stage2_log::phases, "descent_begin: P=%llu levels=%d last_state=(%s)\n", (unsigned long long)P,
                (int)ceil_log2_u64((unsigned long long)Fpad), g_last_state);
    s2g_state("the batched descent");            /* a driver kill here leaves this behind */
    if(!frontier.enabled)ws.need_vals((size_t)P);
    R.tail.metrics.frontier_resident=frontier.enabled;
    std::vector<std::vector<unsigned long long>> values;

    {
            L.cat=BC_DESCENT;
            ScaledStats st;
            if(frontier.enabled) {
                frontier.run(L,Ft,Fdeg,Fpad,H,values,st,BC_DESCENT,g_scaled_check || frontier_check);
                frontier.release();device_fold.release();
            } else descent_scaled(L,Ft,Fdeg,Fpad,H,&finv,values,st,BC_DESCENT,g_scaled_check,root_device.enabled?&prepared_root:nullptr);
            frontier.print();
            L.cat=-1;
            stage2_log::print(stage2_log::phases, "scaled_descent: enabled=1 levels=%llu mul_calls=%llu mul_pairs=%llu copies=%llu zeros=%llu states=%llu words=%llu leaves=%llu checked_states=%llu checked_words=%llu frontier_peak_bytes=%llu pack_peak_bytes=%llu root_inverse_reused=%llu root_divisions=%llu\n",
                st.levels,st.mul_calls,st.mul_pairs,st.copies,st.zeros,st.states,st.words,st.leaves,st.checked_states,st.checked_words,st.frontier_peak_bytes,st.pack_peak_bytes,st.root_inverse_reused,st.root_divisions);
        }
    /* NTT_S4_DESCENT_CHECK=1 runs BOTH descents on the same H and compares every leaf value:
       the batched descent is a rearrangement of the same divisions, so any difference is a bug
       in the rearrangement.  With S5 on, the device descent's leaf values are the ones compared
       (read back once, because the check itself is the point). */
    {
        const char *envc = std::getenv("NTT_S4_DESCENT_CHECK");
        if (L.s4 && envc && *envc && std::atoi(envc) != 0) {

            std::vector<std::vector<unsigned long long>> ref((size_t)P,
                std::vector<unsigned long long>(W, 0ull));
            unsigned long long dm = 0;
            L.cat = BC_DESCENT;
            descent_slow(L, Ft, Fdeg, Fpad, H, ref, dm);
            L.cat = -1;
            unsigned long long bad = 0, first = 0;
            unsigned long long badleaves = 0, firsts[8] = {0}, nfirst = 0;
            for (size_t i = 0; i < (size_t)P; ++i) {
                bool dl = false;
                for (size_t q = 0; q < W; ++q)
                    if (values[i][q] != ref[i][q]) {
                        if (!bad) {
                            first = (unsigned long long)i;
                            stage2_log::print(stage2_log::debug, "descent_check_bad: i=%llu word=%llu batched=%llu "
                                        "slow=%llu\n", (unsigned long long)i,
                                        (unsigned long long)q, values[i][q], ref[i][q]);
                        }
                        ++bad;
                        dl = true;
                    }
                if (dl) { ++badleaves; if (nfirst < 8) firsts[nfirst++] = (unsigned long long)i; }
            }
            stage2_log::print(stage2_log::debug, "descent_check_leaves: P=%llu differing_leaves=%llu first8=", (unsigned long long)P,
                        badleaves);
            for (unsigned long long z = 0; z < nfirst; ++z)
                stage2_log::print(stage2_log::debug, "%s%llu", z ? "," : "", firsts[z]);
            stage2_log::print(stage2_log::debug, "\n");
            /* ---- WHAT IS THE DEVICE'S VALUE, EXACTLY? (section 44) -------------------------
               A DOMAIN mistake is a constant factor, and a constant factor is NAMEABLE: compute
               ref/device mod N and ask whether it is R^k for a small k (R = 2^(64*nw), the
               Montgomery radix).  That turns "the numbers disagree" into "the device returns the
               plain value times R^-2", which is a diagnosis instead of a search. */
            if (bad) {
                mpz_t a, b, ai, ratio, R, pw, Rinv;
                mpz_inits(a, b, ai, ratio, R, pw, Rinv, nullptr);
                words_to_mpz(a, values[first].data(), W);
                words_to_mpz(b, ref[first].data(), W);
                mpz_set_ui(R, 1);
                for (size_t i = 0; i < W * 64; ++i) { mpz_mul_2exp(R, R, 1); mpz_mod(R, R, L.N); }
                char *sb = mpz_get_str(nullptr, 16, b), *sa = mpz_get_str(nullptr, 16, a);
                stage2_log::print(stage2_log::debug, "descent_check_values: i=%llu slow=%s batched=%s\n",
                            first, sb, sa);
                free(sb); free(sa);
                if (mpz_invert(ai, a, L.N) != 0) {
                    mpz_mul(ratio, b, ai); mpz_mod(ratio, ratio, L.N);
                    int kbest = 1000;
                    mpz_set_ui(pw, 1);
                    for (int k = 0; k <= 8 && kbest == 1000; ++k) {
                        if (mpz_cmp(pw, ratio) == 0) kbest = k;
                        mpz_mul(pw, pw, R); mpz_mod(pw, pw, L.N);
                    }
                    if (mpz_invert(Rinv, R, L.N) != 0) {
                        mpz_set_ui(pw, 1);
                        for (int k = 1; k <= 8 && kbest == 1000; ++k) {
                            mpz_mul(pw, pw, Rinv); mpz_mod(pw, pw, L.N);
                            if (mpz_cmp(pw, ratio) == 0) kbest = -k;
                        }
                    }
                    char *sr = mpz_get_str(nullptr, 16, ratio);
                    stage2_log::print(stage2_log::debug, "descent_check_domain: slow/batched = %s = R^%d\n", sr, kbest);
                    free(sr);
                } else {
                    stage2_log::print(stage2_log::debug, "descent_check_domain: the device's leaf value is NOT invertible "
                                "mod N\n");
                }
                mpz_clears(a, b, ai, ratio, R, pw, Rinv, nullptr);
            }
            stage2_log::print(stage2_log::debug, "descent_check: P=%llu divmods_batched=%llu divmods_slow=%llu "
                        "mismatching_coefficients=%llu first=%llu\n", (unsigned long long)P,
                        R.descent_divmods, dm, bad, first);
        }
    }
    R.leaf_values = P;
    /* A diagnostic S5 readback is also a complete plain-domain leaf vector. */
    {
        unsigned long long hash=1469598103934665603ull,words=0;
        for(const auto &v:values) for(auto word:v) {hash=(hash^word)*1099511628211ull;++words;}
        stage2_log::print(stage2_log::debug, "descent_values: leaves=%llu words=%llu hash=%llu\n",(unsigned long long)values.size(),words,hash);
        if(fuse_env_ull("NTT_TARGET_LEAF_HASH",0)) {
            // Compare generic/carrier runs in the target ring and target limb width.
            // This diagnostic never replaces the independent arithmetic oracles.
            const size_t target_words=(L.target_bits()+63)/64;
            std::vector<unsigned long long> projected(target_words);
            mpz_t value;mpz_init(value);
            unsigned long long digest=1469598103934665603ull,nonzero=0;
            for(const auto &v:values) {
                words_to_mpz(value,v.data(),W);mpz_mod(value,value,L.target());
                nonzero+=mpz_sgn(value)!=0;
                mpz_to_words(projected,target_words,value);
                for(auto word:projected)digest=(digest^word)*1099511628211ull;
            }
            mpz_clear(value);
            stage2_log::print(stage2_log::debug,
                "target_descent_values: target_bits=%llu leaves=%llu words=%llu nonzero=%llu hash=%llu\n",
                (unsigned long long)L.target_bits(),(unsigned long long)values.size(),
                (unsigned long long)(values.size()*target_words),nonzero,digest);
        }
    }
    R.t_descent = now_s() - td0 + root_device.seconds;
    /* a phase marker, because the descent is where a long shape can look hung: everything after
       it used to print nothing until the final summary line */
    if (R.dbg_progress)
        stage2_log::print(stage2_log::phases, "batched_phase: descent_done t=%.1f s divmods=%llu\n", R.t_descent,
                    R.descent_divmods);

    stage2_log::phase("Accumulate leaf products and GCD");
    stage2_memory::snapshot("after_descent");
    /* ---- 4. accumulate prod_j H(x_j) mod N ON THE DEVICE, one gcd per block ------------ */
    const double ta0 = now_s();
    R.phase_boundaries.accum_begin=ta0;
    const size_t BLOCK = 64;                            /* small, so a hit localises to <=64 */
    std::vector<std::vector<unsigned long long>> bprod;
    ws.need_vals((size_t)P);
    {
        {                  /* the S5 path already left its leaves in ws.dvals */
            std::vector<unsigned long long> flat((size_t)P * W, 0ull);
            for (size_t i = 0; i < P; ++i)
                std::copy(values[i].begin(), values[i].end(), flat.begin() + (long)(i * W));
            CK(stage2_wait::copy(ws.dvals, flat.data(), flat.size() * 8, cudaMemcpyHostToDevice));
        }
        dev_block_products(ws, (int)P, (int)BLOCK, bprod);
    }
    stage2_memory::snapshot("after_block_products");
    stage2_log::print(stage2_log::debug,
        "s3_product_workspace: compact=%d values_cap=%llu products_cap=%llu values_bytes=%llu "
        "products_bytes=%llu workspace_bytes=%llu block=%llu\n",s3_compact_products_requested(),
        (unsigned long long)ws.val_cap,(unsigned long long)ws.prod_cap,
        (unsigned long long)ws.val_cap*W*8,(unsigned long long)ws.prod_cap*W*8,
        (unsigned long long)ws.bytes,(unsigned long long)BLOCK);
    if (R.dbg_progress)
        stage2_log::print(stage2_log::phases, "batched_phase: block_products_done blocks=%llu t=%.1f s\n",
                    (unsigned long long)bprod.size(), now_s() - ta0);
    R.apply_blocks = bprod.size();
    R.block_per = BLOCK;
    /* A hit at baby j is witnessed by a prime i*D +/- j over the full giant
       range: q | H(x_j) implies some (i*D +/- j)*Q is zero modulo q.
       Naming has a block budget derived from 2*BLOCK*imax primality tests;
       NTT_NAME_HITS enables/disables it and NTT_NAME_BUDGET_BLOCKS overrides
       the cap. Every hit leaf's GCD and proper factors are retained even when
       candidate naming stops. Relocated probes do not name candidates. */
    bool name_hits = true;
    unsigned long long name_budget_blocks = 0;
    {
        const char *eh = std::getenv("NTT_NAME_HITS");
        const char *eb = std::getenv("NTT_NAME_BUDGET_BLOCKS");
        if (eh && *eh) name_hits = (std::atoi(eh) != 0);
        unsigned long long tests_per_block = 0;
        if (imax < (~0ull) / (2ull * (unsigned long long)BLOCK)) {
            tests_per_block = 2ull * (unsigned long long)BLOCK * imax;
        } else {
            tests_per_block = ~0ull;                 /* overflow: treat as unbounded */
        }
        const unsigned long long budget = 1ull << 22;      /* ~4.2e6 primality tests */
        name_budget_blocks = (tests_per_block == 0) ? ~0ull : (budget / tests_per_block);
        if (name_budget_blocks == 0) name_budget_blocks = 1;
        if (eb && *eb) name_budget_blocks = std::strtoull(eb, nullptr, 10);
        stage2_log::print(stage2_log::debug, "naming_policy: name_hits=%d budget_blocks=%llu (2*imax=%llu primality tests "
                    "per leaf, %llu per block of %llu)\n", name_hits ? 1 : 0, name_budget_blocks,
                    2ull * imax, tests_per_block, (unsigned long long)BLOCK);
    }
    mpz_t bv, bg;
    mpz_inits(bv, bg, nullptr);
    std::vector<std::string> bstr((size_t)bprod.size());
    for (size_t b = 0; b < bprod.size(); ++b) {
        words_to_mpz(bv, bprod[b].data(), W);
        mpz_gcd(bg, bv, L.target());
        if(mpz_cmp_ui(bg,1)>0)R.tail.nonunit=true; // includes gcd=N before the legacy extraction gate
        if (mpz_cmp_ui(bg, 1) > 0 && mpz_cmp(bg, L.target()) < 0) {
            ++R.hit_blocks;                             /* a real hit: only NOW do we look at
                                                           the individual leaf values */

            const size_t lo = b * BLOCK;
            const size_t hi = std::min(lo + BLOCK, (size_t)P);
            if (R.dbg_progress)
                stage2_log::print(stage2_log::phases, "batched_phase: naming_begin block=%llu leaves=[%llu,%llu) t=%.1f s\n",
                            (unsigned long long)b, (unsigned long long)lo, (unsigned long long)hi,
                            now_s() - ta0);
            /* The naming cap bounds candidate scans; leaf GCDs remain complete. */
            const bool name_here = !range.active && name_hits && (R.hit_blocks <= name_budget_blocks);
            if (R.dbg_progress && !name_here && R.hit_blocks == name_budget_blocks + 1) {
                stage2_log::print(stage2_log::phases, "batched_phase: naming_budget_exhausted at block=%llu hit_blocks=%llu "
                            "-- hits are still counted, hit_primes are no longer named\n",
                            (unsigned long long)b, R.hit_blocks);
            }
            unsigned long long blk_leaves = 0, blk_cands = 0;   /* this block's naming work */
            unsigned long long blk_leafhits = 0, blk_rec = 0;   /* per-leaf hits, and records */
            for (size_t j = lo; j < hi; ++j) {
                mpz_t v, lg;
                mpz_init(v);
                mpz_init(lg);
                words_to_mpz(v, values[j].data(), W);
                /* `lg` (this leaf's own gcd) is kept SEPARATE from `pg` (a later candidate's
                   gcd), because the fallback below needs the leaf's value after the scan */
                mpz_gcd(lg, v, L.target());
                if (mpz_cmp_ui(lg, 1) > 0 && mpz_cmp(lg, L.target()) < 0) {
                    /* the baby point j = SP.baby_j[j] is the culprit's BABY half:
                       p | H(x_j) => p | (x_j - x_i) for some giant i, so search i */
                    ++R.hit_leaves;
                    ++blk_leafhits;
                    bool named = false;
                    const double tn0 = now_s();
                    const double ts0 = tn0;
                    std::vector<unsigned long long> cand;
                    /* TWO gates, both DIAGNOSTIC-only: NTT_NAME_MAX caps the naming per RUN
                       (the leaf indices are stable, so the same first leaves are named every
                       time) and `name_here` keeps the block budget of the naming phase.
                       Neither can change the factor set -- see the fallback below. */
                    const long long nmax = name_max();
                    if (name_here && (nmax == 0 || R.hit_leaves <= (unsigned long long)nmax)) {
                        const unsigned long long jj = SP.baby_j[j];
                        for (unsigned long long i = 1; i <= imax; ++i) {
                            const unsigned long long off = i * D;
                            for (int sgn = 0; sgn < 2; ++sgn) {
                                if (sgn == 0 && off < jj) continue;
                                const unsigned long long p = (sgn == 0) ? (off - jj) : (off + jj);
                                if (p <= B1 || p > B2) continue;
                                if (!is_prime_u64(p)) continue;
                                cand.push_back(p);
                            }
                        }
                        ++R.named_searches;
                        ++blk_leaves;
                        blk_cands += cand.size();
                        R.candidates_tested += cand.size();
                    }
                    R.t_scan += now_s() - ts0;
                    if (!cand.empty()) {
                        const double tl0 = now_s();
                        std::vector<unsigned long long> cx, cz;
                        ladder_points_ws(ws, cand, cx, cz);
                        for (size_t k = 0; k < cand.size(); ++k) {
                            mpz_t z;
                            mpz_init(z);
                            words_to_mpz(z, &cz[k * W], W);
                            mpz_gcd(pg, z, L.target());
                            if (mpz_cmp_ui(pg, 1) > 0 && mpz_cmp(pg, L.target()) < 0) {
                                ++blk_rec;
                                s3_record(R.tail, pg, cand[k], L.target());
                                named = true;
                            }
                            mpz_clear(z);
                        }
                        R.t_ladder += now_s() - tl0;
                    }
                    if (!named) {
                        /* THE REFERENCE'S OWN CONVENTION (stage2_tree_ref.cpp name_culprit): a
                           hit whose stage-2 prime cannot be identified is STILL a real factor,
                           so record the leaf gcd itself with prime = 0 instead of dropping it.
                           Without this the reported factor SET would depend on the naming scan:
                           a hit leaf whose candidates happen not to confirm used to contribute
                           NOTHING, which is a silently lost factor.  `unnamed` counts these and
                           `hits` is left as the number of ATTRIBUTED hits, so `hits`/`hit_primes`
                           stay comparable with the CPU reference. */
                        s3_record(R.tail, lg, 0, L.target(), /*count_hit=*/false);
                        ++R.unnamed;
                    }
                    R.t_name += now_s() - tn0;
                }
                mpz_clear(lg);
                mpz_clear(v);
            }
            if (R.dbg_progress)
                stage2_log::print(stage2_log::phases, "batched_phase: naming_end block=%llu hit_leaves=%llu candidates=%llu "
                            "leaf_hits=%llu records=%llu hit_blocks=%llu t_name=%.1f s t=%.1f s\n",
                            (unsigned long long)b, blk_leaves, blk_cands, blk_leafhits, blk_rec,
                            (unsigned long long)R.hit_blocks, R.t_name, now_s() - ta0);
        }
    }
    if (R.dbg_progress)
        stage2_log::print(stage2_log::debug, "tail_counts: hits=%llu unnamed=%llu factors=%llu hit_primes=%llu "
                    "hit_blocks=%llu named_searches=%llu candidates=%llu cand_lists=%llu\n",
                    (unsigned long long)R.tail.hits, (unsigned long long)R.tail.unnamed_hits,
                    (unsigned long long)R.tail.factors.size(),
                    (unsigned long long)R.tail.hit_primes.size(),
                    (unsigned long long)R.hit_blocks, (unsigned long long)R.named_searches,
                    (unsigned long long)R.candidates_tested, (unsigned long long)R.cand_lists);
    mpz_clears(bv, bg, nullptr);
    R.t_accum = now_s() - ta0;
    /* WHERE the naming time goes: with the old GMP primality test one hit leaf cost ~1.3 s of
       pure `is_prime_u64` calls, which is 98.7% of a rung-2 run (measured, section 26). */
    stage2_log::print(stage2_log::debug, "batched_naming: hit_blocks=%llu hit_leaves=%llu named_searches=%llu "
                "candidates_tested=%llu unnamed=%llu t_scan=%.3f t_ladder=%.3f t_name=%.3f "
                "name_max=%lld\n",
                R.hit_blocks, R.hit_leaves, R.named_searches, R.candidates_tested, R.unnamed,
                R.t_scan, R.t_ladder, R.t_name, name_max());
    mpz_clears(g, pg, nullptr);
    // The caller extends post through local destruction, final oracle drain
    // and result merging. Keep its measured endpoint separate from subphases.
    /* the device-side counters: how much of the orchestration actually happened once */
    const auto &dl=R.device_leaf;
    stage2_log::print(stage2_log::debug, "device_gleaf: requested_chunks=%llu chunks=%llu retry_chunks=%llu fallback_chunks=%llu groups=%llu bad_groups=%llu good_segments=%llu device_trees=%llu device_leaf_words=%llu patch_words=%llu group_d2h_bytes=%llu bad_segment_d2h_bytes=%llu bad_point_d2h_bytes=%llu avoided_point_d2h_bytes=%llu avoided_segment_d2h_bytes=%llu avoided_leaf_h2d_bytes=%llu coord_peak_bytes=%llu group_peak_bytes=%llu checked_groups=%llu checked_leaf_words=%llu t_prepare=%.6f t_invert=%.6f t_fill=%.6f\n",
        dl.requested_chunks,dl.chunks,dl.retry_chunks,dl.fallback_chunks,dl.groups,dl.bad_groups,dl.good_segments,dl.device_trees,
        dl.device_leaf_words,dl.patch_words,dl.group_d2h_bytes,dl.bad_segment_d2h_bytes,dl.bad_point_d2h_bytes,
        dl.avoided_point_d2h_bytes,dl.avoided_segment_d2h_bytes,dl.avoided_leaf_h2d_bytes-dl.patch_words*8,
        dl.coord_peak_bytes,dl.group_peak_bytes,dl.checked_groups,dl.checked_leaf_words,dl.t_prepare,dl.t_invert,dl.t_fill);
    R.ladder_calls = ws.ladder_calls;
    R.ladder_points = ws.ladder_points_total;
    R.prod_launches = ws.prod_launches;
    if (L.arena) {
        R.arena_fuse_builds = L.arena->fuse_builds;
        R.arena_fuse_reuse = L.arena->fuse_hits;
        R.arena_buf_builds = L.arena->buf_builds;
        R.arena_buf_reuse = L.arena->buf_hits;
        R.arena_overflow = L.arena->overflow;
        R.arena_mb = L.arena->mb();
    }
    R.t_post_end = now_s();
    R.phase_boundaries.accum_end=R.t_post_end;
    R.t_post_loop = R.t_post_end - R.t_post_loop;
    return R;
}
