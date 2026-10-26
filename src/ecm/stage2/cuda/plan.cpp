// Bounds, D selection, memory/request queries and plan reporting. No curve execution.
#include "plan.h"
#include "options.h"
#include "plan_runtime.h"
#include "json_quote.h"
#include <sstream>
#include <iomanip>
#include "../planning/giant_chunk.h"

unsigned long long gcd_u64(unsigned long long a,unsigned long long b)
{
    while (b) { const unsigned long long t = a % b; a = b; b = t; }
    return a;
}

unsigned long long phi_u64(unsigned long long n)
{
    unsigned long long r = n, m = n;
    for (unsigned long long p = 2; p * p <= m; ++p) {
        if (m % p) continue;
        while (m % p == 0) m /= p;
        r -= r / p;
    }
    if (m > 1) r -= r / m;
    return r;
}

unsigned long long real_shape_words(unsigned long long P,int S,bool *ok)
{
    unsigned long long N = 0, sw = 0, ss = 0, os = 0;
    int bpw = 0;
    if (!ntt_shape_query(P, S, &N, &bpw, nullptr, &sw, &ss, &os)) { *ok = false; return 0; }
    *ok = true;
    return NttWorkspacePolicy{}.big_buffer_count() * N + os;
}

bool real_run_geometry(unsigned long long p,int bits,ecm_stage2::Geometry &g)
{
    return ecm_stage2::geometry(p,bits,[](unsigned long long m,int s,
        unsigned long long *n,unsigned long long *out) {
        return ntt_shape_query(m,s,n,nullptr,nullptr,nullptr,nullptr,out);
    },g,kFoldOwnerReuse,NttWorkspacePolicy{}.big_buffer_count());
}

bool real_run_words(
    unsigned long long P,
    int S,
    unsigned long long *out_words,
    unsigned long long *n_fold,
    unsigned long long *n_tree)
{
    ecm_stage2::Geometry g;
    if(!real_run_geometry(P,S,g))return false;
    *out_words=g.arena_estimate_bytes/8;
    if(n_fold)*n_fold=g.fold_length;
    if(n_tree)*n_tree=g.tree_length;
    return true;
}

int prepare_stage2_plan(ecm_stage2::ModulusContext &L,size_t nw,
    unsigned long long sigma,unsigned long long B1,unsigned long long B2,
    unsigned long long D_in,bool choose_d,int curves,bool d_plan_only,
    ecm_stage2::Plan *plan_result,Stage2PlanRuntime &runtime)
{
    CK(cudaSetDevice(g_device));
    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, g_device));
    CK(cudaFree(0));
    size_t freeb = 0, totalb = 0;
    CK(cudaMemGetInfo(&freeb, &totalb));
    size_t reserve = (size_t)768 * 1024 * 1024;
    {
        const char *envr = std::getenv("NTT_ARENA_RESERVE_MB");
        if (envr && *envr) reserve = (size_t)std::strtoull(envr, nullptr, 10) << 20;
    }
    size_t cap = (freeb > reserve) ? (freeb - reserve) : (freeb / 2);
    // Planning must respect the user arena cap as well as live free VRAM.
    const auto user_cap=fuse_env_ull("NTT_ARENA_CAP_KB",0);
    if(user_cap) cap=std::min(cap,(size_t)user_cap*1024);
    stage2_log::print(stage2_log::phases, "stage2_real: device=%d (%s) N_bits=%ld sigma=%llu B1=%llu B2=%llu "
                "requested_D=%llu choose_d=%d\n", g_device, prop.name, (long)L.target_bits(), sigma, B1, B2,
                D_in, choose_d ? 1 : 0);
    stage2_log::print(stage2_log::phases,
        "stage2_modulus: target_bits=%llu carrier_bits=%llu carrier_exponent=%u lifted=%d target_divides_carrier=1\n",
        (unsigned long long)L.target_bits(),(unsigned long long)L.S,L.carrier_exponent,(int)L.lifted());

    /* D balances giant work against the baby/F tree, inverse and descent.
       P=phi(D)/2 grows with D, while imax=B2/D+2 decreases. Baby indices are
       the units j<=D/2 modulo D; phi(D)/D is residue density.
       legacy_56_1 ranks by the phase rates below; estimates are for ranking.
       Actual transform shapes and the arena budget remain hard feasibility
       filters. The model/version text is part of the existing report format. */
    unsigned long long D = D_in, P_baby = 0;
    const char *env4 = std::getenv("NTT_S4_OFF");
    const bool s4_on = !(env4 && *env4 && std::atoi(env4) != 0);
    const bool model_requested=fuse_env_ull("NTT_D_MODEL",0)!=0;
    // Preserve the live NTT device-policy cache warmup at the planning boundary.
    if(fuse_env_ull("NTT_FUSE_COOP_OUTER",0)==2)(void)fuse_shape_policy_supported(12);
    const bool baby_requested=fuse_env_ull("NTT_BABY_DEVICE",0)!=0;
    const auto baby_budget=fuse_env_ull("NTT_BABY_DEVICE_MAX_MB",512)*1024*1024;
    const bool gl_short=(ntt_gl_reduce_requested_mode()&1u)!=0;
    const bool gl_shift_scale=fuse_env_ull("NTT_GL_SHIFT_SCALE",0)!=0;
    const bool gl_ptx=NTT_GL_FIXED_MODE==3 || fuse_env_ull("NTT_GL_PTX_REDUCE",0)!=0;
    const auto fold_budget=fuse_env_ull("NTT_FOLD_DEVICE_MAX_MB",640)*1024*1024;
    stage2_log::print(stage2_log::debug, "ntt_addsub_arithmetic: mask=1\n");
    if(model_requested)
        stage2_log::print(stage2_log::debug, "d_model_scope: calibrated=0 reason=cache_payload_v2_unmeasured\n");
    const double d_scan_begin=now_s();
    double selected_seconds=0;
    const char *model_version="legacy_56_1";
    stage2_log::print(stage2_log::debug, "ntt_outer_schedule: unroll_u=%d (0=compiler-default, 4=experimental ILP)\n",NTT_OUTER_UNROLL_U);
    stage2_log::print(stage2_log::debug, "d_model: requested=%d enabled=%d version=%s arena_cap_bytes=%llu fold_budget_bytes=%llu gl_short=%d gl_shift_scale=%d gl_ptx=%d "
                "(calibrated scope: RTX4060 Laptop M4423 B1=1000 B2=1e11..2011326186870, batch64/chain64; estimates)\n",
                (int)model_requested,0,model_version,
                (unsigned long long)cap,fold_budget,(int)gl_short,(int)gl_shift_scale,(int)gl_ptx);
    {
        struct DCand {
            unsigned long long D = 0, P = 0, imax = 0, batches = 0;
            double units = 0.0, loop = 0.0, tree = 0.0, giant = 0.0, glue = 0.0, total = 0.0;
            double rate = 0.0;                       /* covered values per second (coverage ~ 1) */
            bool fits = false;
            unsigned long long total_words = 0;
        };
        const double RP = 51840.0, RIM = 3400110.0, RB = 66.0;
        const double RLOOP = 265.756, RTREE = 50.537, RGIANT = 50.655, RGLUE = 30.339, RSETUP = 25.0;
        auto lg2 = [](double x) { return (x > 1.0) ? (std::log(x) / std::log(2.0)) : 1.0; };
        auto model = [&](DCand &c) {
            c.imax = B2 / c.D + 2;
            const unsigned long long bsz = (c.P < c.imax) ? c.P : c.imax;
            c.batches = bsz ? (c.imax + bsz - 1) / bsz : 1;
            c.units = (double)c.P * 2.0 / (double)c.D;      /* phi(D)/D: informational only */
            c.loop = (RLOOP / (RIM * lg2(RP))) * (double)c.imax * lg2((double)bsz);
            c.tree = (RTREE / RP) * (double)c.P;
            c.giant = (RGIANT / RIM) * (double)c.imax;
            c.glue = (RGLUE / RB) * (double)c.batches;
            c.total = c.loop + c.tree + c.giant + c.glue + RSETUP;

            const double span = (B2 > B1) ? (double)(B2 - B1) : 1.0;
            c.rate = span / (c.total > 1e-9 ? c.total : 1e-9);
        };
        /* Automatic geometry comes from the same literal catalogue as ECM tune.
           Explicit D remains a separately checked diagnostic override. */
        const unsigned long long dlim = 200000000ull;
        std::vector<DCand> cands;
        cands.reserve(ecm_stage2::d_catalog::count);
        for(const auto &entry:ecm_stage2::d_catalog::entries()) {
            DCand c;c.D=entry.d;c.P=entry.p;model(c);cands.push_back(c);
        }
        std::sort(cands.begin(), cands.end(),
                  [](const DCand &a, const DCand &b) { return a.total < b.total; });
stage2_log::print(stage2_log::debug, "d_scan: candidates=%llu (shared catalogue D <= %llu) ; model fitted to the section "
                    "56.1 run (rates: loop=%.3f s per imax*log2, tree=%.1f us per leaf, "
                    "giant=%.2f us per step, glue=%.2f ms per batch) ; ranked by TOTAL COST "
                    "(coverage is ~1 for every candidate: a prime p > D is coprime to D, so its "
                    "residue is a unit and the baby set IS the units)\n",
                    (unsigned long long)cands.size(), dlim,
                    RLOOP / (RIM * lg2(RP)), RTREE * 1e6 / RP, RGIANT * 1e6 / RIM, RGLUE * 1e3 / RB);
        DCand best;
        unsigned long long checked = 0, shown = 0;
        for (DCand &c : cands) {
            if (shown >= 14 && best.D) break;
            if (checked++ > 4000 && best.D) break;
            unsigned long long tw = 0;

            if (!real_run_words(c.P, (int)L.S, &tw, nullptr, nullptr)) continue;
            c.total_words = tw;
            c.fits = ((double)tw * 8.0 <= (double)cap);
            if (!c.fits) continue;
            if (!best.D || c.total < best.total) best = c;
            if (shown < 14) {
                // i>=1 and j<=D/2: the first giant covers from about D/2.
                // This is a rough lower-bound diagnostic, not a coverage weight.
                auto pi_est=[](double x){return x>2 ? x/std::log(x) : x>=2 ? 1.0 : 0.0;};
                const double missed=std::max(0.0,pi_est((double)c.D/2)-pi_est((double)B1));
                stage2_log::print(stage2_log::debug, "d_scan_choice: D=%llu P=phi/2=%llu imax=%llu batches=%llu "
                            "units=%.4f below_first_giant~%.0f fit=%.0f MB | loop=%.1f tree=%.1f "
                            "giant=%.1f glue=%.1f total=%.1f rate=%.3g vals/s\n",
                            c.D, c.P, c.imax, c.batches, c.units, missed,
                            (double)tw * 8.0 / 1048576.0, c.loop, c.tree, c.giant, c.glue, c.total,
                            c.rate);

                ++shown;
            }
        }
        /* the requested D, for comparison (even when it is not the winner) */
        if (D_in) {
            DCand r;
            r.D = D_in;
            r.P = phi_u64(D_in) / 2;
            model(r);
            unsigned long long tw = 0;
            const bool okm = real_run_words(r.P, (int)L.S, &tw, nullptr, nullptr) && tw*8<=cap;

            stage2_log::print(stage2_log::debug, "d_scan_reference: D=%llu P=%llu imax=%llu batches=%llu units=%.4f "
                        "fit=%s | loop=%.1f tree=%.1f giant=%.1f glue=%.1f total=%.1f "
                        "rate=%.3g vals/s\n",
                        r.D, r.P, r.imax, r.batches, r.units, okm ? "yes" : "NO",
                        r.loop, r.tree, r.giant, r.glue, r.total, r.rate);
        }
        if (choose_d) {
            if (!best.D) {
                std::fprintf(stderr, "%s: no candidate D fits the budget (cap=%.0f MB)\n",
                             NTT_PROBE_NAME, cap / 1048576.0);
                return 3;
            }
            D = best.D;
            selected_seconds=best.total;
            stage2_log::print(stage2_log::debug, "d_scan_decision: D=%llu P=phi(D)/2=%llu (the CHEAPEST admissible shape by "
                        "the fitted cost model -- NOT the largest D that fits, which is what this "
                        "flag used to pick; see the header comment)\n", D, best.P);
        } else {
            P_baby = phi_u64(D) / 2;
            DCand explicit_plan; explicit_plan.D=D; explicit_plan.P=P_baby;
            model(explicit_plan); selected_seconds=explicit_plan.total;
            bool ok1 = false, ok2 = false;
            const unsigned long long w1 = real_shape_words(P_baby+1, (int)L.S, &ok1);
            const unsigned long long w2 = real_shape_words(ecm_stage2::tree_operand_coefficients(P_baby), (int)L.S, &ok2);
            stage2_log::print(stage2_log::debug, "d_budget: free=%.0f MB reserve=%.0f MB cap=%.0f MB ; D=%llu "
                        "P=phi(D)/2=%llu largest_transform=%.0f MB total_needed=%.0f MB "
                        "fits=%d%s\n", freeb / 1048576.0, reserve / 1048576.0,
                        cap / 1048576.0, D, P_baby,
                        (double)w1 * 8.0 / 1048576.0, (double)(w1 + 2 * w2) * 8.0 / 1048576.0,
                        ((double)(w1 + (ok2 ? 2 * w2 : 0)) * 8.0 <= (double)cap) ? 1 : 0,
                        ok1 ? "" : " (SHAPE REFUSED BY THE MULTIPLY)");
        }
    }
    P_baby = phi_u64(D) / 2;
    stage2_log::print(stage2_log::debug, "d_scan_wall: seconds=%.6f selected_D=%llu (excluded from stage2_full_wall)\n",
                now_s()-d_scan_begin,D);
    if(plan_result) {
        ecm_stage2::Plan &p=*plan_result;
        p.target_bits=L.target_bits();p.carrier_exponent=L.carrier_exponent;
        if(!real_run_geometry(P_baby,(int)L.S,p.geometry))return 3;
        const bool host_pack=fuse_env_ull("NTT_S4_HOSTPACK",0)!=0;
        p.tree_payload_model_supported=s4_on && !host_pack && g_s4_pack_direct &&
            !g_s4_final_readback && g_s4_output_window && g_s4_chunk_output && !ntt_carry_check_requested();
        const auto batch_override=fuse_env_ull("NTT_S4_BATCH_MB",0);
        p.tree_batch_bytes=(batch_override?batch_override:(!host_pack && g_s4_pack_direct?g_s4_batch_budget_mb:32))<<20;
        p.tree_physical_chunks=fuse_env_ull("NTT_S4_WORKSPACE_BUDGET",0)!=0;
        p.tree_chunk_max=g_s4_chunk_max;
        if(!ecm_stage2::tree_workspace_plan(P_baby,(int)L.S,[](unsigned long long m,int s,
            unsigned long long *n,unsigned long long *out) {
                return ntt_shape_query(m,s,n,nullptr,nullptr,nullptr,nullptr,out);
            },p.tree_batch_bytes,(unsigned)p.geometry.workspace_buffers,p.tree_physical_chunks,
            p.tree_chunk_max,p.tree_workspace,&g_short_ntt_batches))return 3;
        if(!ecm_stage2::tree_cache_plan(P_baby,(int)L.S,[](unsigned long long m,int s,
            unsigned long long *n,unsigned long long *out) {
                return ntt_shape_query(m,s,n,nullptr,nullptr,nullptr,nullptr,out);
            },[](unsigned long long n,unsigned long long &table,unsigned long long &base) {
                int k=0;for(auto size=n;size>1;size>>=1)++k;
                FuseCtx description;fuse_describe(description,n,k,0,0);
                table=8ull*fuse_planned_table_words(description);
                base=8ull*fuse_planned_base_words(description);return true;
            },NttWorkspacePolicy{}.workspace_pool,p.tree_workspace))return 3;
        p.d=D; p.b1=B1; p.b2=B2; p.giant_points=B2/D+2;
        p.batches=p.giant_points/P_baby+(p.giant_points%P_baby!=0);
        p.requests=std::make_shared<ecm_stage2::RequestPlan>();
        if(!ecm_stage2::request_plan(P_baby,p.giant_points,(int)L.S,
            [](unsigned long long m,int s,unsigned long long *n,unsigned long long *out) {
                return ntt_shape_query(m,s,n,nullptr,nullptr,nullptr,nullptr,out);
            },[](unsigned long long n,unsigned long long &table,unsigned long long &base) {
                int k=0;for(auto size=n;size>1;size>>=1)++k;
                FuseCtx description;fuse_describe(description,n,k,0,0);
                table=8ull*fuse_planned_table_words(description);
                base=8ull*fuse_planned_base_words(description);return true;
            },p.tree_batch_bytes,(unsigned)p.geometry.workspace_buffers,p.tree_physical_chunks,
            p.tree_chunk_max,NttWorkspacePolicy{}.workspace_pool,*p.requests,&g_short_ntt_batches))return 3;
        p.ntt_memory=std::make_shared<ecm_stage2::NttMemoryPlan>();
        const NttWorkspacePolicy workspace_policy;
        const ecm_stage2::NttMemoryPolicy memory_policy{(unsigned long long)cap,
            workspace_policy.workspace_pool,workspace_policy.workspace_reuse_bq,ntt_carry_check_requested()};
        if(!ecm_stage2::ntt_memory_plan(p.requests->program,(int)L.S,
            [](unsigned long long m,int s,unsigned long long *n,unsigned long long *out) {
                return ntt_shape_query(m,s,n,nullptr,nullptr,nullptr,nullptr,out);
            },[](unsigned long long n,ecm_stage2::NttFuseMemoryLayout &layout) {
                int k=0;for(auto size=n;size>1;size>>=1)++k;
                FuseCtx description;fuse_describe(description,n,k,0,0);
                return ecm_stage2::ntt_fuse_memory_layout(description,layout);
            },p.tree_batch_bytes,p.tree_physical_chunks,p.tree_chunk_max,memory_policy,*p.ntt_memory,&g_short_ntt_batches))return 3;
        p.giant_memory=std::make_shared<ecm_stage2::GiantMemoryPlan>();
        p.s4_memory=std::make_shared<ecm_stage2::S4ProgramPlan>();
        ecm_stage2::S4ProgramPolicy s4_policy;
        s4_policy.batch_bytes=p.tree_batch_bytes;s4_policy.chunk_max=p.tree_chunk_max;
        s4_policy.short_ntt=&g_short_ntt_batches;
        s4_policy.buffers=(unsigned)p.geometry.workspace_buffers;s4_policy.physical_chunks=p.tree_physical_chunks;
        s4_policy.output_window=g_s4_output_window;s4_policy.chunk_output=g_s4_chunk_output;
        s4_policy.compact_raw=g_groot_compact_raw;
        s4_policy.trim_raw=gscale_flag("NTT_PHASE_TRIM_RAW");s4_policy.trim_output=gscale_flag("NTT_PHASE_TRIM_OUTPUT");
        p.s4_memory->policy=s4_policy;
        const bool resident_policy=p.tree_payload_model_supported && g_groot_device && g_s4_groot_only &&
            fold_device_flag("NTT_FOLD_FLAT") && fold_device_flag("NTT_FOLD_DEVICE") &&
            !opt_off("NTT_SCALED_FRONTIER_DEVICE");
        if(!resident_policy)p.s4_memory->reason="unsupported_resident_policy";
        else if(!ecm_stage2::s4_program_plan(p.requests->program,(int)L.S,
            [](unsigned long long m,int s,ecm_stage2::S4ShapeDescriptor &q) {
                int bpw=0;const bool ok=ntt_shape_query(m,s,&q.n,&bpw,&q.slot_bits,&q.slot_words,nullptr,&q.slots);
                q.bpw=(unsigned)bpw;return ok;
            },s4_policy,*p.s4_memory))return 3;
        p.workspace_memory=std::make_shared<ecm_stage2::WorkspaceMemoryPlan>();
        if(!resident_policy)p.workspace_memory->reason="unsupported_resident_policy";
        else if(!ecm_stage2::workspace_memory_plan(p.requests->program,(int)L.S,
            [](unsigned long long m,int s,ecm_stage2::S4ShapeDescriptor &q) {
                int bpw=0;const bool ok=ntt_shape_query(m,s,&q.n,&bpw,&q.slot_bits,&q.slot_words,nullptr,&q.slots);
                q.bpw=(unsigned)bpw;return ok;
            },[](unsigned long long n,ecm_stage2::NttFuseMemoryLayout &layout) {
                int k=0;for(auto size=n;size>1;size>>=1)++k;
                FuseCtx description;fuse_describe(description,n,k,0,0);
                return ecm_stage2::ntt_fuse_memory_layout(description,layout);
            },memory_policy,s4_policy,*p.workspace_memory))return 3;
        p.resident_workspace_memory=std::make_shared<ecm_stage2::WorkspaceMemoryPlan>();
        const auto frontier_mb=std::min(fold_budget>>20,fuse_env_ull("NTT_SCALED_FRONTIER_MAX_MB",fold_budget>>20));
        const ecm_stage2::OwnerMemoryPolicy owner_policy{P_baby,nw,fold_budget,kFoldOwnerReuse,frontier_mb<<20};
        if(!resident_policy)p.resident_workspace_memory->reason="unsupported_resident_policy";
        else if(!ecm_stage2::workspace_memory_plan(p.requests->program,(int)L.S,
            [](unsigned long long m,int s,ecm_stage2::S4ShapeDescriptor &q) {
                int bpw=0;const bool ok=ntt_shape_query(m,s,&q.n,&bpw,&q.slot_bits,&q.slot_words,nullptr,&q.slots);
                q.bpw=(unsigned)bpw;return ok;
            },[](unsigned long long n,ecm_stage2::NttFuseMemoryLayout &layout) {
                int k=0;for(auto size=n;size>1;size>>=1)++k;
                FuseCtx description;fuse_describe(description,n,k,0,0);
                return ecm_stage2::ntt_fuse_memory_layout(description,layout);
            },memory_policy,s4_policy,*p.resident_workspace_memory,true,{},&owner_policy))return 3;
        ecm_stage2::GiantMemoryPolicy giant_policy;
        giant_policy.compact_products=s3_compact_products_requested();
        giant_policy.chain_block=std::min(1ull<<20,std::max(4ull,fuse_env_ull("NTT_GIANT_CHAIN_BLOCK",64)));
        giant_policy.short_block=fuse_env_ull("NTT_GIANT_CHAIN_SMALL_BLOCK",0);
        if(giant_policy.short_block)giant_policy.short_block=std::min(64ull,std::max(4ull,giant_policy.short_block));
        giant_policy.short_max=std::min(1ull<<20,fuse_env_ull("NTT_GIANT_CHAIN_SMALL_MAX",8192));
        giant_policy.chain_min=fuse_env_ull("NTT_GIANT_CHAIN_MIN",32768);
        giant_policy.force_ladder=fold_device_flag("NTT_GIANT_LADDER");
        giant_policy.seed_device=g_giant_seed_device;giant_policy.seed_pair=g_giant_seed_pair;
        giant_policy.exact_segments=g_gfinv_seg_exact;
        giant_policy.resident_requested=device_gleaf_flag("NTT_DEVICE_GLEAF");
        giant_policy.resident_eligible=g_groot_device && g_s4_groot_only && s4_on &&
            g_s4_pack_direct && !g_s4_final_readback && !host_pack;
        const auto leaf_mb=fuse_env_ull("NTT_DEVICE_GLEAF_MAX_MB",512);
        const auto point_kb=fuse_env_ull("NTT_GIANT_POINT_BUDGET_KB",262144);
        const bool diagnostic=fold_device_flag("NTT_GIANT_CHAIN_CHECK") || g_giant_seed_check ||
            g_gfinv_seg_check || small_prime_flag("NTT_SMALL_PRIME_CHECK") ||
            small_prime_flag("NTT_SMALL_PRIME_CACHE_STALE") || fold_device_flag("NTT_S3_PRODUCTS_TEST");
        // Conditional on the normal baby proof cache matching this curve. The
        // actual small-prime loop then ladders only primes dividing D. Without
        // reuse, count its bounded interval directly, never use a guessed cap.
        const bool reuse_small=small_prime_flag("NTT_SMALL_PRIME_REUSE");
        const auto half=D/2,lo=std::min(B1,half),hi=std::min(B2,half);
        bool small_supported=true;
        if(reuse_small) {
            auto rest=D;
            for(unsigned long long q=2;q<=rest/q;++q)if(rest%q==0) {
                if(q>lo && q<=hi)++giant_policy.initial_points;
                do{rest/=q;}while(rest%q==0);
            }
            if(rest>1 && rest>lo && rest<=hi)++giant_policy.initial_points;
        } else if(hi>lo) {
            if(hi-lo>10000000ull)small_supported=false;
            else for(auto q=lo+1;q<=hi;++q)if(is_prime_u64(q))++giant_policy.initial_points;
        }
        unsigned long long point_bytes=0;
        if(diagnostic || !small_supported) {
            p.giant_memory->reason=diagnostic?"diagnostic_workspace_not_modeled":"small_prime_interval_not_modeled";
        } else if(!ecm_stage2::multiply(leaf_mb,1048576,giant_policy.resident_limit_bytes) ||
                  !ecm_stage2::multiply(point_kb,1024,point_bytes))return 3;
        else {
            if(giant_policy.seed_device && giant_policy.resident_requested && giant_policy.resident_eligible && giant_policy.exact_segments) {
                giant_policy.chain_min=ecm_stage2::giant_chain_minimum(nw,giant_policy.chain_min,giant_policy.resident_limit_bytes);
                giant_policy.wide_tail_chain=giant_policy.chain_min<fuse_env_ull("NTT_GIANT_CHAIN_MIN",32768);
            }
            const auto giant_chunk=stage2_giant_chunk::plan(P_baby,nw,point_bytes,
                fold_device_flag("NTT_GIANT_CHUNK_FLOOR"));
            if(!giant_chunk.valid || !ecm_stage2::giant_memory_plan(P_baby,p.giant_points,nw,
                giant_chunk.points,giant_policy,*p.giant_memory))return 3;
        }
        p.curve_workspace_memory=std::make_shared<ecm_stage2::WorkspaceMemoryPlan>();
        if(!resident_policy)p.curve_workspace_memory->reason="unsupported_resident_policy";
        else if(!p.giant_memory->valid)p.curve_workspace_memory->reason=p.giant_memory->reason;
        else {
            const ecm_stage2::GiantTimelinePolicy timeline{P_baby,p.giant_points,nw,p.giant_memory->chunk_points,giant_policy};
            const ecm_stage2::InitialMemoryPolicy initial_policy{P_baby,nw,baby_budget,baby_requested,
                initial_memory_diagnostics_requested()};
            if(!ecm_stage2::workspace_memory_plan(p.requests->program,(int)L.S,
                [](unsigned long long m,int s,ecm_stage2::S4ShapeDescriptor &q) {
                    int bpw=0;const bool ok=ntt_shape_query(m,s,&q.n,&bpw,&q.slot_bits,&q.slot_words,nullptr,&q.slots);
                    q.bpw=(unsigned)bpw;return ok;
                },[](unsigned long long n,ecm_stage2::NttFuseMemoryLayout &layout) {
                    int k=0;for(auto size=n;size>1;size>>=1)++k;
                    FuseCtx description;fuse_describe(description,n,k,0,0);
                    return ecm_stage2::ntt_fuse_memory_layout(description,layout);
                },memory_policy,s4_policy,*p.curve_workspace_memory,true,{},&owner_policy,&timeline,&initial_policy,true))return 3;
        }
        p.free_bytes=freeb; p.arena_cap_bytes=cap; p.owner_budget_bytes=fold_budget;
        p.baby_bytes=ecm_stage2::baby_payload_bytes(P_baby,nw);
        p.owner_budget_fits=p.geometry.fold_owner_bytes<=fold_budget;
        p.arena_estimate_fits=p.geometry.arena_estimate_bytes<=cap;
        p.estimated_seconds=selected_seconds; p.calibrated=false; p.model=model_version;
    }
    if(d_plan_only) {
        stage2_log::print(stage2_log::debug, "d_plan_only: D=%llu P=%llu curves_executed=0 model=%s\n",
                    D,phi_u64(D)/2,model_version);
        return 0;
    }

    runtime.freeb=freeb;runtime.totalb=totalb;runtime.reserve=reserve;runtime.cap=cap;
    runtime.D=D;runtime.P_baby=P_baby;runtime.s4_on=s4_on;
    return 0;
}

int query_stage2_plan(const char *n_str,unsigned long long sigma,unsigned long long B1,
    unsigned long long B2,unsigned long long D_in,ecm_stage2::Plan &result,unsigned carrier_exponent)
{
    ecm_stage2::ModulusContext L;
    stage2_memory::Session memory_session;
    stage2_log::phase("Plan Stage2 bounds and memory");
    std::string modulus_error;
    if(!L.configure(n_str,16,carrier_exponent,modulus_error)) {
        std::fprintf(stderr,"%s: %s\n",NTT_PROBE_NAME,modulus_error.c_str());return 2;
    }
    Stage2PlanRuntime planning;
    return prepare_stage2_plan(L,L.W,sigma,B1,B2,D_in,D_in==0,1,true,&result,planning);
}

bool ecm_cuda_stage2_shape_query(uint64_t p,int bits,uint64_t *length,uint64_t *output_slots)
{
    if(!p || p>(1ull<<28) || bits<2 || bits>ecm_stage2::max_input_bits)return false;
    unsigned long long n=0,out=0;
    if(!ntt_shape_query(p,bits,&n,nullptr,nullptr,nullptr,nullptr,&out))return false;
    if(length)*length=n;if(output_slots)*output_slots=out;return true;
}

int ecm_cuda_stage2_device_info(int device,EcmStage2DeviceInfo *info,const char *expected_uuid)
{
    if(!info || device<0)return 2;
    *info=EcmStage2DeviceInfo{};
    cudaDeviceProp prop{};size_t free=0,total=0;
    if(cudaGetDeviceProperties(&prop,device)!=cudaSuccess)return 2;
    const char *hex="0123456789abcdef";
    for(int i=0;i<16;++i){const unsigned char b=(unsigned char)prop.uuid.bytes[i];info->uuid_hex[2*i]=hex[b>>4];info->uuid_hex[2*i+1]=hex[b&15];}
    if(expected_uuid && std::strcmp(expected_uuid,info->uuid_hex))return 3;
    if(cudaSetDevice(device)!=cudaSuccess || cudaMemGetInfo(&free,&total)!=cudaSuccess)return 2;
    info->free_bytes=free;info->total_bytes=total;info->major=prop.major;info->minor=prop.minor;
    if(cudaRuntimeGetVersion(&info->runtime)!=cudaSuccess || cudaDriverGetVersion(&info->driver)!=cudaSuccess)return 2;
#ifdef NTT_GL_FIXED_MODE
    info->fixed_mode=NTT_GL_FIXED_MODE;
#endif
#ifdef NTT_OUTER_UNROLL_U
    info->outer_unroll_u=NTT_OUTER_UNROLL_U;
#endif
    return 0;
}

int ecm_cuda_stage2_plan(
    const char *n_hex,
    uint64_t sigma,
    uint64_t b1,
    uint64_t b2,
    uint64_t d,
    int device,
    void (*report)(const char*,void*),
    void *context,
    unsigned carrier_exponent)
{
    stage2_log::PlanningScope planning_logs;
    if(!report || !n_hex || device<0 || !b1 || b2<=b1 ||
       b2>(uint64_t)INT64_MAX-8192 || (d && (d<6 || d%2)))return 2;
    if(ecm_cuda_stage2_check_configuration())return 2;
    g_device=device;
    ecm_stage2::Plan p;
    const int code=query_stage2_plan(n_hex,sigma,b1,b2,d,p,carrier_exponent);
    if(code)return code;
    const auto &g=p.geometry;
    const auto relocation=ecm_stage2::giant_range(p.b2,p.d,g_relocation_enabled);
    std::ostringstream json;json<<std::setprecision(17)
        <<"{\"type\":\"stage2_plan\",\"schema\":1,\"curves_executed\":0,\"bits\":"<<g.bits
        <<",\"target_bits\":"<<p.target_bits<<",\"carrier_exponent\":"<<p.carrier_exponent
        <<",\"words\":"<<g.words<<",\"workspace_buffers\":"<<g.workspace_buffers<<",\"B1\":"<<p.b1<<",\"B2\":"<<p.b2<<",\"D\":"<<p.d
        <<",\"P\":"<<g.p<<",\"I\":"<<p.giant_points<<",\"G\":"<<p.batches
        <<",\"relocation\":"<<int(g_relocation_enabled)<<",\"relocation_first\":"<<relocation.first
        <<",\"relocation_points\":"<<relocation.points<<",\"relocation_prime\":"<<relocation.missing
        <<",\"relocation_estimate_scope\":\"original_replay_range\""
        <<",\"fold_length\":"<<g.fold_length<<",\"tree_length\":"<<g.tree_length
        <<",\"tree_operand_coefficients\":"<<g.tree_operand_coeffs
        <<",\"fold_big_bytes\":"<<g.fold_big_bytes<<",\"arena_estimate_bytes\":"<<g.arena_estimate_bytes
        <<",\"arena_estimate_kind\":\"legacy_additive\""
        <<",\"owner_bytes\":"<<g.fold_owner_bytes<<",\"owner_reuse\":"<<kFoldOwnerReuse<<",\"baby_payload_bytes\":"<<p.baby_bytes
        <<",\"free_bytes\":"<<p.free_bytes<<",\"arena_cap_bytes\":"<<p.arena_cap_bytes
        <<",\"owner_budget_bytes\":"<<p.owner_budget_bytes
        <<",\"owner_budget_fits\":"<<(p.owner_budget_fits ? "true" : "false")
        <<",\"arena_estimate_fits\":"<<(p.arena_estimate_fits ? "true" : "false")
        <<",\"residency_guaranteed\":false,\"process_peak_estimated\":false,\"accounting_version\":2,\"geometry_version\":3"
        <<",\"tree_workspace\":{\"pool\":"<<(NttWorkspacePolicy{}.workspace_pool?"true":"false")
        <<",\"supported\":"<<(p.tree_payload_model_supported?"true":"false")
        <<",\"no_eviction_model\":true,\"process_peak_complete\":false,\"batch_bytes\":"<<p.tree_batch_bytes
        <<",\"physical_chunks\":"<<(p.tree_physical_chunks?"true":"false")<<",\"chunk_max\":"<<p.tree_chunk_max
        <<",\"groups\":"<<p.tree_workspace.groups<<",\"pairs\":"<<p.tree_workspace.pairs<<",\"chunks\":"<<p.tree_workspace.chunks
        <<",\"shared_big_peak_bytes\":"<<p.tree_workspace.big_peak_bytes
        <<",\"digit_retained_bytes\":"<<p.tree_workspace.digit_retained_bytes
        <<",\"keyed_big_retained_bytes\":"<<p.tree_workspace.keyed_big_retained_bytes
        <<",\"output_peak_bytes\":"<<p.tree_workspace.output_peak_bytes
        <<",\"table_retained_bytes\":"<<p.tree_workspace.table_retained_bytes
        <<",\"base_retained_bytes\":"<<p.tree_workspace.base_retained_bytes
        <<",\"ntt_retained_bytes\":"<<p.tree_workspace.ntt_retained_bytes<<",\"cache_shapes\":[";
    bool first_cache=true;
    for(const auto &entry:p.tree_workspace.caches) {
        if(!first_cache)json<<',';first_cache=false;
        json<<"{\"N\":"<<entry.first<<",\"table_bytes\":"<<entry.second.table<<",\"base_bytes\":"<<entry.second.base<<'}';
    }
    json<<"]}";
    if(p.requests) {
        const auto &requests=*p.requests;const auto &payload=requests.retained;
        json<<",\"request_program\":{\"version\":2,\"valid\":"<<(requests.valid?"true":"false")
            <<",\"supported\":"<<(requests.valid && p.tree_payload_model_supported?"true":"false")
            <<",\"contract\":"<<stage2_tune::quote(requests.program.reason)
            <<",\"residency_required\":true,\"full_fold_degree_required\":true"
            <<",\"no_eviction_model\":true,\"process_peak_complete\":false,\"admission_model\":false"
            <<",\"multiplier\":"<<requests.signature.multiplier<<",\"addend\":"<<requests.signature.addend
            <<",\"shared_big_peak_bytes\":"<<payload.big_peak_bytes
            <<",\"digit_retained_bytes\":"<<payload.digit_retained_bytes
            <<",\"keyed_big_retained_bytes\":"<<payload.keyed_big_retained_bytes
            <<",\"output_peak_bytes\":"<<payload.output_peak_bytes
            <<",\"table_retained_bytes\":"<<payload.table_retained_bytes
            <<",\"base_retained_bytes\":"<<payload.base_retained_bytes
            <<",\"ntt_retained_bytes\":"<<payload.ntt_retained_bytes<<",\"phases\":[";
        for(unsigned i=0;i<ecm_stage2::RequestPhaseCount;++i) {
            if(i)json<<',';const auto &phase=requests.phases[i];const auto &sig=requests.signatures[i];
            json<<"{\"phase\":"<<stage2_tune::quote(ecm_stage2::request_phase_name(i))
                <<",\"groups\":"<<phase.groups<<",\"pairs\":"<<phase.pairs<<",\"chunks\":"<<phase.chunks
                <<",\"request_big_peak_bytes\":"<<phase.big_peak_bytes
                <<",\"request_output_peak_bytes\":"<<phase.output_peak_bytes
                <<",\"multiplier\":"<<sig.multiplier<<",\"addend\":"<<sig.addend<<'}';
        }
        json<<"],\"cache_shapes\":[";bool first=true;
        for(const auto &entry:payload.caches) {
            if(!first)json<<',';first=false;
            json<<"{\"N\":"<<entry.first<<",\"table_bytes\":"<<entry.second.table<<",\"base_bytes\":"<<entry.second.base<<'}';
        }
        json<<"],\"blocks\":[";first=true;
        for(const auto &block:requests.program.blocks) {
            if(!first)json<<',';first=false;json<<"{\"repeat\":"<<block.repeat<<",\"tree_leaves\":"<<block.tree_leaves<<",\"requests\":[";bool first_request=true;
            for(const auto &r:block.requests) {
                if(!first_request)json<<',';first_request=false;
                json<<"{\"phase\":"<<r.phase<<",\"ma\":"<<r.ma<<",\"mb\":"<<r.mb
                    <<",\"pairs\":"<<r.pairs<<",\"first\":"<<r.first<<",\"count\":"<<r.count
                    <<",\"input\":"<<stage2_tune::quote(ecm_stage2::request_input_name(r.input))<<'}';
            }
            json<<"]}";
        }
        json<<"]}";
    }
    if(p.ntt_memory) {
        const auto &memory=*p.ntt_memory;
        auto payload=[&](const ecm_stage2::NttMemoryPayload &x) {
            json<<"{\"big_bytes\":"<<x.big<<",\"digit_bytes\":"<<x.digits
                <<",\"table_bytes\":"<<x.table<<",\"base_bytes\":"<<x.base<<",\"total_bytes\":"<<x.total<<'}';
        };
        auto counters=[&](const ecm_stage2::NttMemoryCounters &x) {
            json<<"{\"calls\":"<<x.calls<<",\"fuse_builds\":"<<x.fuse_builds
                <<",\"fuse_hits\":"<<x.fuse_hits<<",\"workspace_grows\":"<<x.workspace_grows
                <<",\"cap_evictions\":"<<x.cap_evictions<<",\"cap_evicted_bytes\":"<<x.cap_evicted_bytes
                <<",\"carry_grows\":"<<x.carry_grows<<",\"carry_refusals\":"<<x.carry_refusals
                <<",\"allocations\":"<<x.allocations<<",\"frees\":"<<x.frees<<",\"grouped_events\":"<<x.grouped_events<<'}';
        };
        json<<",\"ntt_memory\":{\"version\":2,\"valid\":"<<(memory.valid?"true":"false")
            <<",\"supported\":"<<(memory.valid && p.tree_payload_model_supported?"true":"false")
            <<",\"finished\":"<<(memory.finished?"true":"false")<<",\"reason\":"<<stage2_tune::quote(memory.reason)
            <<",\"cap_bytes\":"<<memory.policy.cap_bytes<<",\"pool\":"<<(memory.policy.pool?"true":"false")
            <<",\"reuse_bq\":"<<(memory.policy.reuse_bq?"true":"false")
            <<",\"carry_check\":"<<(memory.policy.carry_check?"true":"false")
            <<",\"exact_allocation_events\":"<<(memory.exact_allocation_events?"true":"false")
            <<",\"cold_trim_modeled\":false,\"fallback_modeled\":false"
            <<",\"process_peak_complete\":false,\"admission_model\":false"
            <<",\"peak_bytes\":"<<memory.peak_bytes<<",\"executed_blocks\":"<<memory.executed_blocks
            <<",\"skipped_blocks\":"<<memory.skipped_blocks<<",\"final_payload\":";
        payload(memory.final_payload);json<<",\"counters\":";counters(memory.counters);
        json<<",\"stopped_at\":";
        if(memory.valid && !memory.finished) {
            const auto &s=memory.stop;const auto &r=s.request;
            json<<"{\"block\":"<<s.block<<",\"repeat_index\":"<<s.repeat_index
                <<",\"request_index\":"<<s.request_index<<",\"phase\":"<<stage2_tune::quote(ecm_stage2::request_phase_name(r.phase))
                <<",\"ma\":"<<r.ma<<",\"mb\":"<<r.mb<<",\"pairs\":"<<r.pairs
                <<",\"N\":"<<s.n<<",\"slots\":"<<s.slots<<",\"slices\":"<<s.slices<<'}';
        } else json<<"null";
        json<<",\"checkpoints\":[";bool first=true;
        for(const auto &checkpoint:memory.checkpoints) {
            if(!first)json<<',';first=false;
            json<<"{\"repeat\":"<<checkpoint.repeat<<",\"peak_bytes\":"<<checkpoint.peak_bytes<<",\"payload\":";
            payload(checkpoint.live);json<<",\"counters\":";counters(checkpoint.counters);json<<'}';
        }
        json<<"],\"fuse_layouts\":[";first=true;
        for(const auto &entry:memory.fuse_layouts) {
            if(!first)json<<',';first=false;
            json<<"{\"N\":"<<entry.first<<",\"grouped\":"<<(entry.second.grouped?"true":"false");
            auto allocations=[&](const char *name,const std::vector<ecm_stage2::NttFuseAllocation> &items) {
                json<<",\""<<name<<"\":[";bool comma=false;
                for(const auto &a:items) {if(comma)json<<',';comma=true;
                    json<<"{\"site\":"<<a.site<<",\"index\":"<<a.index<<",\"bytes\":"<<a.bytes<<'}';}
                json<<']';
            };
            allocations("base",entry.second.base);allocations("table",entry.second.table);json<<'}';
        }
        json<<"]}";
    }
    if(p.workspace_memory) {
        const auto &m=*p.workspace_memory;
        json<<",\"workspace_memory\":{\"version\":1,\"valid\":"<<(m.valid?"true":"false")
            <<",\"finished\":"<<(m.finished?"true":"false")<<",\"reason\":"<<stage2_tune::quote(m.reason)
            <<",\"components\":[\"ntt\",\"s4\"],\"process_peak_complete\":false,\"admission_model\":false"
            <<",\"peak_bytes\":"<<m.peak_bytes<<",\"ntt_at_peak\":"<<m.ntt_at_peak<<",\"s4_at_peak\":"<<m.s4_at_peak
            <<",\"ntt_peak_bytes\":"<<m.ntt_peak_bytes<<",\"s4_peak_bytes\":"<<m.s4_peak_bytes
            <<",\"final_bytes\":"<<m.final_bytes<<",\"released_bytes\":"<<m.released_bytes
            <<",\"executed_blocks\":"<<m.executed_blocks<<",\"skipped_blocks\":"<<m.skipped_blocks
            <<",\"simulated_events\":"<<m.simulated_events<<'}';
    }
    if(p.resident_workspace_memory) {
        const auto &m=*p.resident_workspace_memory;
        json<<",\"resident_workspace_memory\":{\"version\":1,\"valid\":"<<(m.valid?"true":"false")
            <<",\"finished\":"<<(m.finished?"true":"false")<<",\"reason\":"<<stage2_tune::quote(m.reason)
            <<",\"components\":[\"ntt\",\"s4\",\"fold_frontier\"],\"process_peak_complete\":false,\"admission_model\":false"
            <<",\"headroom_modeled\":false,\"fallback_modeled\":false,\"cold_trim_modeled\":false"
            <<",\"peak_bytes\":"<<m.peak_bytes<<",\"ntt_at_peak\":"<<m.ntt_at_peak<<",\"s4_at_peak\":"<<m.s4_at_peak
            <<",\"owner_at_peak\":"<<m.owner_at_peak<<",\"owner_peak_bytes\":"<<m.owner_peak_bytes
            <<",\"fold_bytes\":"<<m.fold_bytes<<",\"frontier_bytes\":"<<m.frontier_bytes
            <<",\"final_bytes\":"<<m.final_bytes<<",\"released_bytes\":"<<m.released_bytes
            <<",\"executed_blocks\":"<<m.executed_blocks<<",\"skipped_blocks\":"<<m.skipped_blocks
            <<",\"simulated_events\":"<<m.simulated_events<<'}';
    }
    if(p.curve_workspace_memory) {
        const auto &m=*p.curve_workspace_memory;
        const auto reserve=fuse_env_ull("NTT_ARENA_RESERVE_MB",768)*1048576ull;
        unsigned long long required_free=0;
        const bool required_valid=ecm_stage2::add(m.peak_bytes,reserve,required_free);
        required_free=std::max(required_free,std::max(m.baby_headroom_bytes,
            std::max(m.fold_headroom_bytes,m.frontier_headroom_bytes)));
        json<<",\"curve_workspace_memory\":{\"version\":3,\"valid\":"<<(m.valid?"true":"false")
            <<",\"finished\":"<<(m.finished?"true":"false")<<",\"reason\":"<<stage2_tune::quote(m.reason)
            <<",\"components\":[\"ntt\",\"s4\",\"fold_frontier\",\"giant\",\"initial\"],\"process_peak_complete\":false,\"admission_model\":false"
            <<",\"headroom_modeled\":false,\"fallback_modeled\":"<<(m.fallback_upper_bound?"true":"false")
            <<",\"payload_upper_bound\":"<<(m.fallback_upper_bound?"true":"false")
            <<",\"fold_enabled\":"<<(m.fold_enabled?"true":"false")<<",\"frontier_enabled\":"<<(m.frontier_enabled?"true":"false")
            <<",\"transient_peak_bytes\":"<<m.transient_peak_bytes<<",\"cold_trim_modeled\":false"
            <<",\"peak_bytes\":"<<m.peak_bytes<<",\"ntt_at_peak\":"<<m.ntt_at_peak<<",\"s4_at_peak\":"<<m.s4_at_peak
            <<",\"owner_at_peak\":"<<m.owner_at_peak<<",\"giant_at_peak\":"<<m.giant_at_peak
            <<",\"initial_at_peak\":"<<m.initial_at_peak<<",\"initial_peak_bytes\":"<<m.initial_peak_bytes
            <<",\"montgomery_bytes\":"<<m.montgomery_bytes<<",\"baby_bytes\":"<<m.baby_bytes
            <<",\"baby_headroom_bytes\":"<<m.baby_headroom_bytes<<",\"fold_headroom_bytes\":"<<m.fold_headroom_bytes
            <<",\"frontier_headroom_bytes\":"<<m.frontier_headroom_bytes<<",\"required_free_bytes\":"<<required_free
            <<",\"reserve_bytes\":"<<reserve
            <<",\"initial_free_snapshot_fits\":"<<(m.valid && m.finished && required_valid && required_free<=p.free_bytes?"true":"false")
            <<",\"giant_peak_bytes\":"<<m.giant_peak_bytes<<",\"giant_final_bytes\":"<<m.giant_final_bytes
            <<",\"point_chunks\":"<<m.point_chunks<<",\"points_consumed\":"<<m.points_consumed
            <<",\"final_bytes\":"<<m.final_bytes<<",\"released_bytes\":"<<m.released_bytes
            <<",\"executed_blocks\":"<<m.executed_blocks<<",\"skipped_blocks\":"<<m.skipped_blocks
            <<",\"simulated_events\":"<<m.simulated_events<<'}';
    }
    if(p.s4_memory) {
        const auto &m=*p.s4_memory;const auto &v=m.policy;
        auto payload=[&](const ecm_stage2::S4MemoryPayload &x) {
            json<<"{\"raw_a_bytes\":"<<x.raw_a<<",\"raw_b_bytes\":"<<x.raw_b
                <<",\"output_bytes\":"<<x.output<<",\"pack_a_bytes\":"<<x.pack_a<<",\"pack_b_bytes\":"<<x.pack_b
                <<",\"modulus_bytes\":"<<x.modulus<<",\"shape_constant_bytes\":"<<x.shape_constants
                <<",\"canonical_bytes\":"<<x.canonical_counter<<",\"selftest_digits_bytes\":"<<x.selftest_digits
                <<",\"selftest_output_bytes\":"<<x.selftest_output<<",\"tree_metadata_bytes\":"<<x.tree_metadata
                <<",\"total_bytes\":"<<x.total<<'}';
        };
        json<<",\"s4_memory\":{\"version\":1,\"valid\":"<<(m.valid?"true":"false")
            <<",\"reason\":"<<stage2_tune::quote(m.reason)
            <<",\"scope\":\"conditional_resident_s4\",\"residency_required\":true,\"full_fold_degree_required\":true"
            <<",\"process_peak_complete\":false,\"admission_model\":false,\"fallback_modeled\":false"
            <<",\"multiplier\":"<<m.signature.multiplier<<",\"addend\":"<<m.signature.addend
            <<",\"peak_bytes\":"<<m.peak_bytes<<",\"shape_count\":"<<m.shape_count
            <<",\"executed_blocks\":"<<m.executed_blocks<<",\"skipped_blocks\":"<<m.skipped_blocks
            <<",\"policy\":{\"batch_bytes\":"<<v.batch_bytes<<",\"chunk_max\":"<<v.chunk_max<<",\"buffers\":"<<v.buffers
            <<",\"physical_chunks\":"<<(v.physical_chunks?"true":"false")
            <<",\"output_window\":"<<(v.output_window?"true":"false")<<",\"chunk_output\":"<<(v.chunk_output?"true":"false")
            <<",\"compact_raw\":"<<(v.compact_raw?"true":"false")<<",\"trim_raw\":"<<(v.trim_raw?"true":"false")
            <<",\"trim_output\":"<<(v.trim_output?"true":"false")<<"},\"counters\":{\"requests\":"<<m.counters.requests
            <<",\"chunks\":"<<m.counters.chunks<<",\"trees\":"<<m.counters.trees
            <<",\"allocations\":"<<m.counters.allocations<<",\"frees\":"<<m.counters.frees<<"},\"after_inverse\":";
        payload(m.after_inverse);json<<",\"after_giant\":";payload(m.after_giant);
        json<<",\"final_payload\":";payload(m.final_payload);json<<",\"released_payload\":";payload(m.released_payload);
        json<<",\"shapes\":[";bool first=true;
        for(const auto &entry:m.descriptors){if(!first)json<<',';first=false;const auto &q=entry.second;
            json<<"{\"operand\":"<<entry.first<<",\"N\":"<<q.n<<",\"slots\":"<<q.slots
                <<",\"slot_bits\":"<<q.slot_bits<<",\"slot_words\":"<<q.slot_words<<",\"bpw\":"<<q.bpw<<'}';}
        json<<"]}";
    }
    if(p.giant_memory) {
        const auto &m=*p.giant_memory;const auto &v=m.policy;
        json<<",\"giant_memory\":{\"version\":2,\"valid\":"<<(m.valid?"true":"false")
            <<",\"reason\":"<<stage2_tune::quote(m.reason)
            <<",\"process_peak_complete\":false,\"admission_model\":false"
            <<",\"small_prime_cache_assumed\":"<<(small_prime_flag("NTT_SMALL_PRIME_REUSE")?"true":"false")
            <<",\"chunk_points\":"<<m.chunk_points<<",\"point_chunks\":"<<m.point_chunks
            <<",\"initial_bytes\":"<<m.initial_bytes<<",\"after_giant_bytes\":"<<m.after_giant_bytes
            <<",\"accumulation_bytes\":"<<m.accumulation_bytes<<",\"peak_bytes\":"<<m.peak_bytes
            <<",\"value_bytes\":"<<m.value_bytes<<",\"product_bytes\":"<<m.product_bytes
            <<",\"final_point_capacity\":"<<m.final_point_capacity
            <<",\"policy\":{\"chain_block\":"<<v.chain_block<<",\"short_block\":"<<v.short_block
            <<",\"short_max\":"<<v.short_max<<",\"chain_min\":"<<v.chain_min
            <<",\"wide_tail_chain\":"<<(v.wide_tail_chain?"true":"false")
            <<",\"segment\":"<<v.segment<<",\"group\":"<<v.group
            <<",\"resident_limit_bytes\":"<<v.resident_limit_bytes<<",\"initial_points\":"<<v.initial_points
            <<",\"compact_products\":"<<(v.compact_products?"true":"false")
            <<",\"accumulation_block\":"<<v.accumulation_block
            <<",\"force_ladder\":"<<(v.force_ladder?"true":"false")
            <<",\"seed_device\":"<<(v.seed_device?"true":"false")
            <<",\"seed_pair\":"<<(v.seed_pair?"true":"false")
            <<",\"exact_segments\":"<<(v.exact_segments?"true":"false")
            <<",\"resident_requested\":"<<(v.resident_requested?"true":"false")
            <<",\"resident_eligible\":"<<(v.resident_eligible?"true":"false")<<"},\"chunks\":[";
        bool first=true;
        for(const auto &c:m.chunks) {
            if(!first)json<<',';first=false;
            json<<"{\"repeat\":"<<c.repeat<<",\"points\":"<<c.points
                <<",\"route\":"<<stage2_tune::quote(c.chain?"chain":"ladder")
                <<",\"resident\":"<<(c.resident?"true":"false")<<",\"block\":"<<c.block
                <<",\"seeds\":"<<c.seeds<<",\"segments\":"<<c.segments<<",\"groups\":"<<c.groups
                <<",\"point_capacity\":"<<c.point_capacity<<",\"workspace_bytes\":"<<c.workspace_bytes
                <<",\"coordinate_bytes\":"<<c.coordinate_bytes<<",\"segment_bytes\":"<<c.segment_bytes
                <<",\"group_bytes\":"<<c.group_bytes<<",\"legacy_seed_bytes\":"<<c.legacy_seed_bytes
                <<",\"prepare_bytes\":"<<c.prepare_bytes<<",\"tree_bytes\":"<<c.tree_bytes<<'}';
        }
        json<<"]}";
    }
    json<<",\"model\":"<<stage2_tune::quote(p.model)<<",\"calibrated\":"<<(p.calibrated ? "true" : "false")
        <<",\"stage2_seconds_estimate\":"<<p.estimated_seconds<<"}";
    report(json.str().c_str(),context);return 0;
}
