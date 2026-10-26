#include "host_wait.h"
// Polynomial multiplication, packing, output windows and NTT/S4 accounting.
#include "product.h"
#include "options.h"
#include "checks.h"
#include "s4_pack_state.h"
#include "pack_launch.h"

// Host packing policy, accounting and launch geometry. Device kernels live in
// kernels/cuda/stage2/pack.cuh.
namespace stage2_pack {

static void account(unsigned long long m,unsigned long long N,bool gather,bool reverse=false) {
    if(gather)++stats.gather;else ++stats.contiguous;
    if(reverse)++stats.reverse;
    stats.launches+=(m+65534)/65535;stats.words+=m*N;++stats.avoided_memsets;
}
}

void MulCost::add(int cat,size_t m1,size_t m2)
{
    const unsigned long long m = (unsigned long long)((m1 > m2) ? m1 : m2);
    const int l = ceil_log2_u64(m);
    const unsigned long long b =
        2ull * m * (2ull * (unsigned long long)S + (unsigned long long)l);
    bits[cat] += b;
    ++muls[cat];
    coeff[cat] += (unsigned long long)m1 * (unsigned long long)m2;
    if (l < 64) { ++hcnt[l]; hbits[l] += b; }
    if (m > cat_mul_max[cat]) cat_mul_max[cat] = m;
    if (b > max_bits) { max_bits = b; max_m1 = m1; max_m2 = m2; }
}

unsigned long long MulCost::tot_muls() const
{
    unsigned long long t = 0;
    for (int i = 0; i < BC_NCAT; ++i) t += muls[i];
    return t;
}

unsigned long long MulCost::tot_bits() const
{
    unsigned long long t = 0;
    for (int i = 0; i < BC_NCAT; ++i) t += bits[i];
    return t;
}

void PolyLayer::count_ntt(unsigned long long n,unsigned long long pairs,unsigned long long chunks)
{
    ntt_lengths.add(n,pairs,chunks);
}

void PolyLayer::print_ntt_lengths()const
{
    if(!ntt_lengths.enabled)return;
    std::string counts,convs;unsigned long long pairs=0,chunks=0;double elements=0,butterflies=0;
    // Explicit opt-in output is independent of the progress log level.
    std::printf("ntt_count_table: scope=curve_polynomial includes=initialization excludes=arithmetic_selftests,stage1\n");
    std::printf("ntt_count_columns: length log2 convolutions forward inverse total\n");
    for(size_t k=0;k<ntt_lengths.entries.size();++k) {
        const auto &v=ntt_lengths.entries[k];if(!v.convolutions)continue;
        std::printf("ntt_count_row: %12llu %2zu %12llu %12llu %12llu %12llu\n",
            1ull<<k,k,(unsigned long long)v.convolutions,2ull*v.convolutions,
            (unsigned long long)v.convolutions,3ull*v.convolutions);
        if(!counts.empty()){counts+=' ';convs+=' ';}
        counts+=std::to_string(k)+":"+std::to_string(3*v.convolutions);
        convs+=std::to_string(k)+":"+std::to_string(v.convolutions);
        pairs+=v.convolutions;chunks+=v.chunks;
        const double work=3.*double(v.convolutions)*double(1ull<<k);
        elements+=work;butterflies+=work*double(k)/2.;
    }
    std::printf("ntt_length_counts: log2_length:num_of_ntt %s\n",counts.c_str());
    std::printf("ntt_work: scope=curve_polynomial convolutions=%llu transforms=%llu s4_chunks=%llu elements=%.6e logical_radix2_butterflies=%.6e short_batch_changed=%llu\n",
        pairs,3*pairs,chunks,elements,butterflies,short_batch_changed);
    stage2_log::print(stage2_log::debug,"ntt_convolution_counts: log2_length:num_of_conv %s\n",convs.c_str());
    std::fflush(stdout);
}

bool poly_is_one(const unsigned long long *p,size_t W)
{
    if (p[0] != 1) return false;
    for (size_t i = 1; i < W; ++i)
        if (p[i] != 0) return false;
    return true;
}

std::vector<unsigned long long> poly_mul_modN(
    PolyLayer &L,
    const std::vector<unsigned long long> &a,
    size_t da,
    const std::vector<unsigned long long> &b,
    size_t db,
    int cat)
{
    const size_t W = L.W;
    const size_t ma = da + 1, mb = db + 1;
    const unsigned long long P = (unsigned long long)((ma > mb) ? ma : mb);
    std::vector<unsigned long long> wa((size_t)P * W, 0ull), wb((size_t)P * W, 0ull);
    std::copy(a.begin(), a.begin() + (long)(ma * W), wa.begin());
    std::copy(b.begin(), b.begin() + (long)(mb * W), wb.begin());

    std::vector<unsigned long long> slots;
    std::vector<std::vector<unsigned long long>> exact;
    NttMulStats st{};
    const double t0 = now_s();
    const int rc = ntt_poly_mul_host(P, (int)L.S, L.device, /*verbose=*/false, /*dump=*/0,
                                     wa.data(), wb.data(), &slots, &exact, &st, L.arena);
    if (rc != 0) {
        std::fprintf(stderr, "%s: ntt_poly_mul_host failed (rc=%d) at P=%llu S=%d\n",
                     NTT_PROBE_NAME, rc, (unsigned long long)P, (int)L.S);
        std::exit(3);
    }
    L.ntt_seconds += now_s() - t0;
    L.t_fwd += st.t_fwd;
    L.t_inv += st.t_inv;
    L.t_slot += st.t_slot;
    L.t_setup += st.t_setup;
    L.t_pack += st.t_pack;
    L.t_maxc += st.t_maxc;
    L.t_h2d += st.t_h2d;
    L.t_d2h += st.t_d2h;
    L.t_ext += st.t_ext;
    L.t_xchk += st.t_xchk;
    ++L.ntt_calls;
    L.count_ntt(st.N,1,1);
    ++L.muls;
    {
        const int c = (cat >= 0) ? cat : L.cat;       /* explicit, else the layer's current one */
        if (c >= 0) L.cost.add(c, ma, mb);
    }
    L.max_ntt_words = std::max(L.max_ntt_words, st.N);
    L.max_ntt_coeffs = std::max(L.max_ntt_coeffs, P);
    L.max_slot_bits = std::max(L.max_slot_bits, st.slot_bits);
    /* The exactness bound is proven and asserted inside ntt_poly_mul_host for THIS (P, S);
       re-assert the headline inequality here from the values that came back, so the tree's
       own shapes are visibly covered:  L*(2^bpw-1)^2 < p. */
    if (!exact_ok_terms(st.L_terms, st.bpw)) {
        std::fprintf(stderr, "%s: EXACTNESS VIOLATED for the tree's shape: L=%llu bpw=%d\n",
                     NTT_PROBE_NAME, (unsigned long long)st.L_terms, st.bpw);
        std::exit(3);
    }
    {
        const double bb = coeff_bound_bits_terms(st.L_terms, st.bpw);
        if (bb > L.bind_bound_bits) {
            L.bind_bound_bits = bb;
            L.bind_P = P;
            L.bind_L = st.L_terms;
            L.bind_slot_bits = st.slot_bits;
            L.bind_slot_words = st.slot_words;
            L.bind_bpw = st.bpw;
        }
    }
    if (exact.size() != st.out_slots) {
        std::fprintf(stderr, "%s: expected %llu exact coefficients, got %llu\n", NTT_PROBE_NAME,
                     (unsigned long long)st.out_slots, (unsigned long long)exact.size());
        std::exit(3);
    }
    const size_t nc = ma + mb - 1;
    std::vector<unsigned long long> out(nc * W, 0ull);
    mpz_t t;
    mpz_init(t);
    for (size_t k = 0; k < nc; ++k) {
        mpz_import(t, exact[k].size(), -1, 8, 0, 0, exact[k].data());
        mpz_mod(t, t, L.N);
        size_t cnt = 0;
        mpz_export(&out[k * W], &cnt, -1, 8, 0, 0, t);
        if (cnt > W) {
            std::fprintf(stderr, "%s: FATAL: reduced coefficient needs %llu words > W=%llu\n",
                         NTT_PROBE_NAME, (unsigned long long)cnt, (unsigned long long)W);
            std::exit(3);
        }
        ++L.slot_checks;
    }
    mpz_clear(t);
    return out;
}

void poly_mul_batch_modN(
    PolyLayer &L,
    const unsigned long long *wa,
    const unsigned long long *wb,
    size_t ma,
    size_t mb,
    size_t nbatch,
    std::vector<unsigned long long> &out,
    int cat,
    NttMulStats *st_out,
    size_t first,
    size_t count,
    const S4DeviceBatch *resident)
{
    stage2_trace::Range trace("S4.request");
    const size_t W = L.W;
    const size_t P = (ma > mb) ? ma : mb;
    const size_t full_nc=ma+mb-1;
    if(count==(size_t)-1) count=full_nc;
    if(first>full_nc || count>full_nc-first) {
        std::fprintf(stderr,"%s: FATAL: invalid polynomial output window\n",NTT_PROBE_NAME);
        std::exit(3);
    }
    const size_t nc=count;
    const bool host_output=!resident || g_s4_carry_trace;
    if(host_output) out.assign(nbatch*nc*W,0ull);else out.clear();
    if (nbatch == 0) return;
    if(resident) {
        if(!resident->src || !resident->dst || resident->src==resident->dst ||
           !resident->meta || !resident->host_meta || resident->nbatch!=nbatch ||
           (!resident->owner && (first!=0 || count!=full_nc)) || g_s4_final_readback || !g_s4_pack_direct) {
            std::fprintf(stderr,"%s: FATAL: invalid resident batch lifetime/shape\n",NTT_PROBE_NAME);std::exit(3);
        }
        for(size_t i=0;i<nbatch;++i) {
            const auto *o=resident->host_meta;
            if(o[i]>resident->src_words || ma*W>resident->src_words-o[i] ||
               o[nbatch+i]>resident->src_words || mb*W>resident->src_words-o[nbatch+i] ||
               o[2*nbatch+i]>resident->dst_words || nc*W>resident->dst_words-o[2*nbatch+i]) {
                std::fprintf(stderr,"%s: FATAL: resident offset outside frontier\n",NTT_PROBE_NAME);std::exit(3);
            }
        }
    }
    S4Ctx &C = *L.s4;
    if(resident && resident->owner &&
       (!resident->owner->owns(resident->src,resident->src_words) ||
        !resident->owner->owns(resident->dst,resident->dst_words))) {
        std::fprintf(stderr,"%s: FATAL: resident range is not held by its owner\n",NTT_PROBE_NAME);std::exit(3);
    }
    if(resident && !resident->owner && ((resident->src!=C.d_rawA && resident->src!=C.d_rawB) ||
                   (resident->dst!=C.d_rawA && resident->dst!=C.d_rawB) ||
                   resident->src_words>C.raw_capacity(resident->src) || resident->dst_words>C.raw_capacity(resident->dst))) {
        std::fprintf(stderr,"%s: FATAL: resident frontier is not owned raw staging\n",NTT_PROBE_NAME);std::exit(3);
    }
    const unsigned long long out_slots = 2 * P - 1; // FULL shape and canonical assertion
    const unsigned long long output_slots=g_s4_output_window ? count : out_slots;
    const unsigned long long output_first=g_s4_output_window ? first : 0;
    const size_t host_first=g_s4_output_window ? 0 : first;
    /* the per-shape reduction (L, the domain constant and the GMP selftest) must exist BEFORE
       the hook fires, and the hook fires inside the multiply -- so the shape is queried here,
       without touching the device.  Never inferred: ntt_shape_query runs the SAME choose_cfg
       the multiply will run. */
    unsigned long long qN = 0, qsb = 0, qsw = 0, qss = 0, qos = 0;
    int qbpw = 0;
    {
        if (!ntt_shape_query(P, (int)L.S, &qN, &qbpw, &qsb, &qsw, &qss, &qos)) {
            std::fprintf(stderr, "%s: the shape P=%llu S=%d is refused by the multiply\n",
                         NTT_PROBE_NAME, (unsigned long long)P, (int)L.S);
            std::exit(3);
        }
        if (qos != out_slots || qss != qsw * (unsigned long long)qbpw) {
            std::fprintf(stderr, "%s: FATAL: the queried shape disagrees with the multiply's\n",
                         NTT_PROBE_NAME);
            std::exit(3);
        }
        if (!C.red->find(qsb, qsw, qbpw)) {
            S4Reduce::Shape *S0 = s4_shape_init(*C.red, (unsigned long long)P, qsb, qss, qsw,
                                                qbpw);
            if (s4_reduce_selftest(*C.red, S0)) {
                std::fprintf(stderr, "%s: FATAL: the device reduction disagrees with GMP on the "
                                     "actual shape -- refusing to continue\n", NTT_PROBE_NAME);
                std::exit(3);
            }
            C.selftested = true;
        }
    }
    const size_t need = nbatch * output_slots * W;
    NttReduceHook hook;
    hook.ctx = C.red;
    hook.run = s4_reduce_hook;
    hook.w = (unsigned long long)W;
    hook.first=output_first; hook.count=output_slots;

    std::vector<unsigned long long> slots;
    NttMulStats st{};
    const double t0 = now_s();
    /* CHUNKING (slice S4, learned from the real shape): a level's whole batch needs
       nbatch*N words, and nbatch*N is roughly CONSTANT per level (~P*slot_words), so material-
       ising every level at once asked the arena for ~600 MB per level and 17 levels blew the
       cap -- after which the per-call fallback could not allocate and the run died with
       "out of memory".  A per-chunk device budget keeps every (N, chunk) buffer small; the
       arena caches one entry per (N, chunk) because the chunk size is a deterministic function
       of N.  Bigger budget = fewer launches; smaller = less memory. */
    /* ---- NTT_S4_BATCH_MB: THE PER-CHUNK DEVICE BUDGET (section 27) --------------------------
       This was a hard-coded 32 MB.  The per-level timers showed a ~0.3-0.4 s FIXED cost per tree
       level that does not depend on the multiply count at all (3600 tiny multiplies cost 0.375 s,
       and ONE multiply of the full degree costs 0.618 s), and it lands inside
       poly_mul_batch_modN -- whose own sub-timers the dev path never fills.  The chunking below is
       the first suspect: a level's batch is split into ceil(nbatch/chunk) chunk round-trips, each
       with its own H2D, pack, NTT, reduce and D2H.  This knob makes that testable. */
    /* The legacy controls have not gained the direct packer's VRAM headroom. */
    static const bool host_pack = [] {
        const char *e = std::getenv("NTT_S4_HOSTPACK");
        return e && *e && std::atoi(e) != 0;
    }();
    static const unsigned long long s4_batch_mb = [] {
        const char *e = std::getenv("NTT_S4_BATCH_MB");
        return (e && *e) ? std::strtoull(e, nullptr, 10) : 0ull;
    }();
    static const unsigned long long default_mb =
        (!host_pack && g_s4_pack_direct) ? g_s4_batch_budget_mb : 32ull;
    const unsigned long long effective_mb = s4_batch_mb ? s4_batch_mb : default_mb;
    const unsigned long long budget_bytes = effective_mb << 20;
    static bool budget_reported = false;
    if (!budget_reported) {
        stage2_log::print(stage2_log::debug, "s4_batch_budget: mb=%llu env_override=%d direct=%d host_pack=%d\n",
                    effective_mb, (int)(s4_batch_mb != 0), (int)g_s4_pack_direct, (int)host_pack);
        budget_reported = true;
    }
    const bool workspace_budget=fuse_env_ull("NTT_S4_WORKSPACE_BUDGET",0)!=0;
    const unsigned buffers=L.arena ? L.arena->big_buffer_count() : 3;
    unsigned long long legacy_chunk=ecm_stage2::chunk_slices(qN,out_slots,nbatch,budget_bytes,3,false);
    unsigned long long chunk=workspace_budget ?
        ecm_stage2::chunk_slices(qN,out_slots,nbatch,budget_bytes,buffers) : legacy_chunk;
    if (g_s4_chunk_max) chunk = std::min(chunk, g_s4_chunk_max);
    if (g_s4_chunk_max) legacy_chunk = std::min(legacy_chunk, g_s4_chunk_max);
    const auto untuned_chunk=chunk;
    chunk=g_short_ntt_batches.select(qN,nbatch,chunk);
    if(chunk!=untuned_chunk)++L.short_batch_changed;
    ++C.chunk_plan_calls;
    C.chunk_plan_changed+=(chunk!=legacy_chunk);
    C.chunk_plan_chunks+=nbatch/chunk+(nbatch%chunk!=0);
    C.chunk_plan_legacy_chunks+=nbatch/legacy_chunk+(nbatch%legacy_chunk!=0);
    unsigned long long request_bytes=0;
    if(!ecm_stage2::chunk_request_bytes(qN,out_slots,chunk,buffers,request_bytes)) {
        std::fprintf(stderr,"%s: S4 chunk payload overflow\n",NTT_PROBE_NAME);std::exit(3);
    }
    C.chunk_plan_request_peak=std::max(C.chunk_plan_request_peak,request_bytes);
    C.chunk_plan_over_budget+=(request_bytes>budget_bytes);
    const bool chunk_output=g_s4_chunk_output && !g_s4_final_readback;
    const size_t allocation_need=std::max((size_t)1,
        (size_t)(chunk_output ? chunk : nbatch)*output_slots*W);
    const int memory_category=cat>=0?cat:L.cat;
    if(memory_category>=0 && memory_category<BC_NCAT) {
        auto &p=C.phase_requests[memory_category];
        ++p.groups;p.pairs+=nbatch;p.chunks+=nbatch/chunk+(nbatch%chunk!=0);
        p.big_peak_bytes=std::max(p.big_peak_bytes,8ull*buffers*qN*chunk);
        p.output_peak_bytes=std::max(p.output_peak_bytes,8ull*allocation_need);
        if(fuse_env_ull("NTT_REQUEST_AUDIT",0)) {
            static_assert(BC_FTREE==ecm_stage2::RequestFtree && BC_GTREE==ecm_stage2::RequestGtrees &&
                BC_FOLD==ecm_stage2::RequestFold && BC_DESCENT==ecm_stage2::RequestDescent &&
                BC_FINV==ecm_stage2::RequestInverse && BC_NCAT==ecm_stage2::RequestPhaseCount,"phase topology");
            const ecm_stage2::MultiplyRequest r{(unsigned)memory_category,ma,mb,nbatch,first,count};
            C.request_signatures[memory_category].append(r,qN,out_slots,chunk);
            C.request_signature.append(r,qN,out_slots,chunk);
            auto routed=r;
            routed.input=!resident?ecm_stage2::RequestHost:!resident->owner?ecm_stage2::RequestTreeRaw:
                resident->reverse_b?ecm_stage2::RequestFrontierOwner:ecm_stage2::RequestFoldOwner;
            ecm_stage2::s4_signature_request(C.s4_program_signature,routed,qN,out_slots,chunk);
        }
    }
    if(allocation_need>C.d_out_cap) {
        if(C.d_out) { CK(cudaFree(C.d_out)); C.d_out=nullptr; C.d_out_cap=0; }
        CK(cudaMalloc(&C.d_out,allocation_need*sizeof(unsigned long long)));
        C.d_out_cap=allocation_need;
        ++g_chunk_output.grows;
    }
    hook.out=C.d_out;
    ++g_chunk_output.calls;
    if(chunk_output) {
        ++g_chunk_output.reused_calls;
        g_chunk_output.reused_chunks+=(nbatch+chunk-1)/chunk;
        if(chunk<nbatch) ++g_chunk_output.multi_chunk_reused;
    } else ++g_chunk_output.whole_calls;
    if(g_s4_chunk_output && g_s4_final_readback) ++g_chunk_output.legacy_calls;
    g_chunk_output.request_peak_bytes=std::max(g_chunk_output.request_peak_bytes,8ull*allocation_need);
    g_chunk_output.whole_peak_bytes=std::max(g_chunk_output.whole_peak_bytes,8ull*std::max((size_t)1,need));
    g_chunk_output.retained_peak_bytes=std::max(g_chunk_output.retained_peak_bytes,8ull*C.d_out_cap);
    int rc = 0;
    /* THE DEVICE PACKING SWITCH (objective 4, section 33): default is the device packer; set
       NTT_S4_HOSTPACK=1 to run the old host-packing path, which is kept as the A/B oracle. */
    /* ---- THE DEFERRED CARRY CHECK (section 29) ----------------------------------------------
       Every chunk ends with the probe's carry check, which is a pageable D2H of 2*nbatch words and
       therefore waits for the device to finish everything queued before it: a full pipeline drain
       per chunk.  Measured at the production shape that is 22471 x 1.64 ms = 36.75 s of a 253.53 s
       run (`carrysplt d2h`), the largest host-side item on the books, and it is what makes the
       chunk size matter so much (fewer chunks = fewer drains, section 27).

       Deferring it needs the counters to survive between chunks, which requires the arena (the
       buffer has to outlive the call), so it is on whenever an arena exists and the device packer
       is in use.  The accumulation lives in the arena entry of the chunk's shape, which is why the
       readback happens BEFORE a non-deferred chunk resets the counters. Identical deferred chunks
       skip that reset and can accumulate their atomicAdd/atomicMax counters in place. The original
       per-interior finish remains the CARRY_BATCH=0 control. Interior chunks lose their fwd/inv
       event attribution (the events are
       destroyed unread, since reading them would need the drain we are removing) -- the last chunk
       of each call still reports the sample. */
    const bool defer_ok = (!host_pack && L.arena != nullptr && g_s4_defer_carry);
    /* the doubly-buffered PINNED output staging (section 30): reserved once per call at the largest
       chunk this call can use, so every chunk of the call follows the same path (mixing a blocking
       chunk with the deferred-consumption scheme would break the pending bookkeeping) */
    const size_t out_words_max = (size_t)(chunk * output_slots * W);
    if (!g_pin_ev[0]) CK(cudaEventCreateWithFlags(&g_pin_ev[0], cudaEventDisableTiming));
    if (!g_pin_ev[1]) CK(cudaEventCreateWithFlags(&g_pin_ev[1], cudaEventDisableTiming));
    const bool async_out = (host_output && g_s4_async &&
                            pin_words(&g_pin_out[0], &g_pin_out_cap[0], out_words_max) != nullptr &&
                            pin_words(&g_pin_out[1], &g_pin_out_cap[1], out_words_max) != nullptr);
    unsigned long long ci = 0, pend_m = 0, pend_s0 = 0;
    bool out_pending = false;
    bool carry_pending = false;
    unsigned long long carry_pending_m = 0, carry_pending_chunks = 0;
    double carry_d2h_acc = 0.0, chunk_d2h_acc = 0.0;
    unsigned long long carry_res_acc = 0, carry_bits_acc = 0;
    auto finish_carry = [&] {
        NttMulStats fin{};
        const int rf = ntt_batch_carry_finish(L.arena, qN, carry_pending_m, &fin);
        carry_d2h_acc += fin.t_check_d2h;
        g_carry_group_readback += fin.t_check_d2h;
        carry_res_acc += fin.carry_residual;
        carry_bits_acc = std::max(carry_bits_acc, fin.carry_max_bits);
        ++g_defer_finishes;
        if (rf != 0) {
            std::fprintf(stderr, "%s: the deferred carry check of the chunked multiply failed "
                                 "(rc=%d) at P=%llu nbatch=%llu chunk=%llu accumulated=%llu\n",
                         NTT_PROBE_NAME, rf, (unsigned long long)P,
                         (unsigned long long)nbatch, carry_pending_m, carry_pending_chunks);
            std::exit(3);
        }
        g_defer_checked_chunks += carry_pending_chunks;
        g_defer_max_group = std::max(g_defer_max_group, carry_pending_chunks);
        carry_pending = false;
        carry_pending_chunks = 0;
    };
    for (unsigned long long s0 = 0; s0 < nbatch; s0 += chunk) {
        const unsigned long long m = ((nbatch - s0) < chunk) ? (nbatch - s0) : chunk;
        const bool last_chunk = (s0 + m >= nbatch);
        /* s0 != 0 is NOT optional: the first chunk still has to run the probe's non-deferred
           path because that is what memsets the residual counters.  Skipping it would leave the
           PREVIOUS call's counters in the arena entry, and since a successful check leaves zeros
           the result would not be a false alarm but a silently VACUOUS check for those slices. */
        const bool defer_this = defer_ok && !last_chunk && (s0 != 0);
        /* SAME (N,m), no reset: atomic counters retain every interior chunk's verdict. Finish
           before a last/short/non-deferred chunk, or at function exit, never after its memset. */
        if (carry_pending && (!g_s4_carry_batch || !defer_this || carry_pending_m != m))
            finish_carry();
        NttReduceHook h2 = hook;
        if (hook.out) h2.out = hook.out + (chunk_output ? 0 : (size_t)(s0 * output_slots) * W);
        int r1 = 0;
        if(resident) {
            if(host_pack) {std::fprintf(stderr,"%s: FATAL: resident multiply cannot use host packing\n",NTT_PROBE_NAME);std::exit(3);}
            S4GatherPack pack{resident,&C,(size_t)s0,ma,mb,(unsigned long long)P,qN,qsw,(int)L.S,qbpw,(int)W};
            NttInputHook input{&pack,s4_gather_final_input};
            const double tp=C.t_packdev;
            r1=ntt_poly_mul_batch_dev(P,(int)L.S,L.device,m,nullptr,nullptr,&st,L.arena,&h2,
                nullptr,qbpw,defer_this,&input);
            // Input pack remains a direct engine write; no temporary packed inputs or D2D.
            if(!r1) {++C.input_direct;C.input_avoided_bytes+=16ull*m*qN;}
            C.packed_peak_bytes=std::max(C.packed_peak_bytes,16ull*m*qN);
            C.pack_launches+=2;
            C.t_h2d_raw+=C.t_packdev-tp;
            if(!r1 && st.carry_deferred) {
                carry_pending=true;carry_pending_m=m;++carry_pending_chunks;
                ++g_defer_chunks;g_defer_slices+=m;
                if(g_s4_carry_test_bad && !g_s4_carry_injected) {
                    CK(cudaMemsetAsync(L.arena->cur.dRes,1,sizeof(unsigned long long)));
                    g_s4_carry_injected=true;
                    stage2_log::print(stage2_log::debug, "s4_carry_fault: first resident interior diagnostic poisoned P=%llu m=%llu\n",
                        (unsigned long long)P,m);
                }
            }
        } else if (host_pack) {
            r1 = ntt_poly_mul_batch_host(P, (int)L.S, L.device, m,
                                         wa + s0 * P * W, wb + s0 * P * W,
                                         (s0 == 0) ? &slots : nullptr, &st, L.arena, &h2,
                                         nullptr);
        } else {
            /* 1. the RAW coefficients to the device (P*W words per slice per operand) ... */
            const size_t raw_words = (size_t)m * P * W;
            C.raw_reserve(raw_words,raw_words);
            const double th0 = now_s();
            /* ASYNC UPLOAD VIA PINNED STAGING (section 30): the host memcpy into pinned memory
               touches no device state, so it cannot drain the pipeline, and the pinned
               cudaMemcpyAsync needs no implicit sync either.  Without pinned memory the old
               blocking pair runs unchanged. */
            const size_t raw_bytes = raw_words * sizeof(unsigned long long);
            const size_t upload_slot = g_s4_carry_batch ? (size_t)(ci & 1) : 0;
            if (g_s4_async && g_pin_raw_pending[upload_slot]) {
                const cudaError_t ready = cudaEventQuery(g_pin_raw_ev[upload_slot]);
                if (ready == cudaErrorNotReady) {
                    const double tw0 = now_s();
                    CK(stage2_wait::event_sync(g_pin_raw_ev[upload_slot],stage2_wait::Kind::upload));
                    g_pin_raw_wait += now_s() - tw0;
                    ++g_pin_raw_waits;
                } else CK(ready);
                g_pin_raw_pending[upload_slot] = false;
            }
            unsigned long long *pa = g_s4_async ? pin_words(&g_pin_raw[upload_slot][0], &g_pin_raw_cap[upload_slot][0], raw_words)
                                                : nullptr;
            unsigned long long *pb = g_s4_async ? pin_words(&g_pin_raw[upload_slot][1], &g_pin_raw_cap[upload_slot][1], raw_words)
                                                : nullptr;
            if (pa && pb) {
                std::memcpy(pa, wa + s0 * P * W, raw_bytes);
                std::memcpy(pb, wb + s0 * P * W, raw_bytes);
                CK(cudaMemcpyAsync(C.d_rawA, pa, raw_bytes, cudaMemcpyHostToDevice));
                CK(cudaMemcpyAsync(C.d_rawB, pb, raw_bytes, cudaMemcpyHostToDevice));
                if (!g_pin_raw_ev[upload_slot])
                    CK(cudaEventCreateWithFlags(&g_pin_raw_ev[upload_slot], cudaEventDisableTiming));
                CK(cudaEventRecord(g_pin_raw_ev[upload_slot]));
                g_pin_raw_pending[upload_slot] = true;
                ++g_pin_raw_used;
            } else {
                CK(stage2_wait::copy(C.d_rawA, wa + s0 * P * W, raw_bytes, cudaMemcpyHostToDevice));
                CK(stage2_wait::copy(C.d_rawB, wb + s0 * P * W, raw_bytes, cudaMemcpyHostToDevice));
                if (g_s4_async) ++g_pin_fallbacks;
            }
            /* 2. ... packed into digits ON the device (both operands, one launch each) ... */
            const size_t pack_words = (size_t)m * (size_t)qN;
            if (!g_s4_pack_direct && pack_words > C.d_pack_cap) {
                if (C.d_packA) { cudaFree(C.d_packA); cudaFree(C.d_packB); C.d_packA = C.d_packB = nullptr; }
                CK(cudaMalloc(&C.d_packA, pack_words * sizeof(unsigned long long)));
                CK(cudaMalloc(&C.d_packB, pack_words * sizeof(unsigned long long)));
                C.d_pack_cap = pack_words;
                C.temp_peak_bytes = std::max(C.temp_peak_bytes, 16ull * C.d_pack_cap);
            }
            C.packed_peak_bytes = std::max(C.packed_peak_bytes, 16ull * pack_words);
            S4InputPack pack{&C, (unsigned long long)P, qN, qsw, (int)L.S, qbpw, (int)W};
            NttInputHook input{&pack, s4_pack_final_input};
            const double tp0 = C.t_packdev;
            if (!g_s4_pack_direct) {
                /* Control: identical digits go through the retained temporary buffers + D2D. */
                CK(cudaMemset(C.d_packA, 0, pack_words * sizeof(unsigned long long)));
                CK(cudaMemset(C.d_packB, 0, pack_words * sizeof(unsigned long long)));
                const double tpack0 = now_s();
                s4_launch_pack_batch(C.d_rawA, (int)L.S, qbpw, qsw, qN, m, P, (int)W, C.d_packA);
                s4_launch_pack_batch(C.d_rawB, (int)L.S, qbpw, qsw, qN, m, P, (int)W, C.d_packB);
                C.t_packdev += now_s() - tpack0;
            }
            C.t_h2d_raw += now_s() - th0;
            C.raw_words += (unsigned long long)raw_words * 2;
            C.pack_launches += 2;
            /* 3. ... and the multiply itself, device-to-device (the same passes, the same carry,
               the same exactness assertions, the same reduction hook) */
            NttMulStats nst{};
            r1 = ntt_poly_mul_batch_dev(P, (int)L.S, L.device, m, C.d_packA, C.d_packB, &nst,
                                        L.arena, &h2, nullptr, qbpw, defer_this,
                                        g_s4_pack_direct ? &input : nullptr);
            /* Keep rawupload's historical upload+packing host total comparable. The direct
               pack is queued INSIDE the engine, so add just its own enqueue time here. */
            if (g_s4_pack_direct) C.t_h2d_raw += C.t_packdev - tp0;
            if (r1 == 0) {
                if (g_s4_pack_direct) { ++C.input_direct; C.input_avoided_bytes += 16ull * pack_words; }
                else { ++C.input_copied; C.input_d2d_bytes += 16ull * pack_words; }
                C.t_input_copy_host += nst.t_opcopy;
            }
            st = nst;                     /* the caller's stats are the dev path's */
            if (r1 == 0 && nst.carry_deferred) {
                carry_pending = true; carry_pending_m = m; ++carry_pending_chunks;
                ++g_defer_chunks; g_defer_slices += m;
                if (g_s4_carry_test_bad && !g_s4_carry_injected) {
                    /* Corrupt ONLY the diagnostic counter after the first interior chunk.
                       Later good chunks and the tail must never hide this error. */
                    CK(cudaMemsetAsync(L.arena->cur.dRes, 1, sizeof(unsigned long long)));
                    g_s4_carry_injected = true;
                    stage2_log::print(stage2_log::debug, "s4_carry_fault: first interior diagnostic poisoned P=%llu m=%llu\n",
                                (unsigned long long)P, m);
                }
            }
            if (s0 == 0) slots.clear();   /* the host path filled this; the dev path does not */
        }
        if (r1 != 0) { rc = r1; break; }
        // Simultaneous owned SUBSET at this boundary, not a sum of module peaks.
        // Includes all live arena payload/base and retained S4 raw/pack/output;
        // excludes reduction constants/oracle, points, trees and resident owners.
        unsigned long long owned=L.arena ? L.arena->bytes : 0;
        owned+=8ull*(C.d_out_cap+C.d_rawA_cap+C.d_rawB_cap+2*C.d_pack_cap);
        C.chunk_plan_subset_peak=std::max(C.chunk_plan_subset_peak,owned);
        if(memory_category>=0 && memory_category<BC_NCAT) {
            unsigned long long big=L.arena?8ull*L.arena->workspace.words:0;
            if(L.arena)for(const auto &entry:L.arena->bigs)big+=8ull*entry.words;
            C.phase_big_live_peak[memory_category]=std::max(C.phase_big_live_peak[memory_category],big);
            C.phase_subset_peak[memory_category]=std::max(C.phase_subset_peak[memory_category],owned);
            C.phase_ntt_peak[memory_category]=std::max(C.phase_ntt_peak[memory_category],(unsigned long long)(L.arena?L.arena->bytes:0));
        }
        /* Readbacks of ALL non-deferred chunks, not only the final overwritten `st`.
           Interior deferred chunks contribute zero here and are charged by finish_carry. */
        chunk_d2h_acc += st.t_check_d2h;
        g_carry_chunk_readback += st.t_check_d2h;
        carry_bits_acc = std::max(carry_bits_acc, st.carry_max_bits);
        /* the reduced coefficients of this chunk, back to the host -- AND THIS TRANSFER IS THE
           POINT OF OBJECTIVE 4 (section 32): it is `m * output_slots * W` words, i.e. the WHOLE
           product of every slice at full slot width, and it is neither inside the probe's timers
           (they belong to the host implementation) nor inside the tree's own phases.  It is
           therefore timed and counted HERE, so "the 111 us per call that nobody measured" can be
           attributed instead of guessed.

           SECTION 30: it is now an ASYNC copy into pinned staging, and the chunk that is copied
           here is consumed on the NEXT iteration -- so the host never waits for the device unless
           the device is genuinely behind, which is the difference between "the transfer is
           overlapped" and "the pipeline is drained per chunk".  `out_pending` marks the one chunk
           that has been copied but not yet written into `out`; the end of the call drains it. */
        const size_t out_words = (size_t)(m * output_slots * W);
        if(resident) {
            ecm_launch_s4_scatter_result_kernel(EcmCudaLaunch{(unsigned int)((m*nc*W+255)/256),256}, h2.out,
                output_slots,host_first,nc,m,(int)W,resident->meta+2*nbatch+s0,resident->dst);
            CK(cudaGetLastError());
            if(!host_output) {if(!resident->owner)g_gdevice.resident_words+=out_words;continue;}
            if(!resident->owner)g_gdevice.trace_words+=out_words;
        }
        if (!async_out) {
            /* NO PINNED MEMORY: the original blocking readback, so a failed pinning costs speed
               and never correctness (and never a half-filled `out`) */
            const double td0 = now_s();
            std::vector<unsigned long long> all(out_words, 0ull);
            if(out_words) CK(stage2_wait::copy(all.data(), h2.out, out_words * sizeof(unsigned long long),
                                       cudaMemcpyDeviceToHost));
            for (unsigned long long s = 0; s < m; ++s)
                std::copy(all.begin() + (long)((s * output_slots + host_first) * W),
                          all.begin() + (long)((s * output_slots + host_first + nc) * W),
                          out.begin() + (long)((s0 + s) * nc * W));
            L.t_d2h_coeff += now_s() - td0;
            L.d2h_coeff_words += (unsigned long long)out_words;
            continue;
        }
        unsigned long long *po = g_pin_out[(size_t)(ci & 1)];
        const double td0 = now_s();
        if(out_words) CK(cudaMemcpyAsync(po, h2.out, out_words * sizeof(unsigned long long),
                                        cudaMemcpyDeviceToHost));
        CK(cudaEventRecord(g_pin_ev[ci & 1]));
        ++g_pin_out_used;
        L.t_d2h_coeff += now_s() - td0;
        L.d2h_coeff_words += (unsigned long long)out_words;
        /* NOW consume the previous chunk's buffer: its copy has had this whole chunk's work to
           finish, so this wait is normally already satisfied */
        if (out_pending) {
            const double tw0 = now_s();
            CK(stage2_wait::event_sync(g_pin_ev[(ci - 1) & 1],stage2_wait::Kind::output));
            const unsigned long long *prev = g_pin_out[(ci - 1) & 1];
            for (unsigned long long s = 0; s < pend_m; ++s)
                std::copy(prev + ((size_t)s * output_slots + host_first) * W,
                          prev + ((size_t)s * output_slots + host_first) * W + nc * W,
                          out.begin() + (long)((pend_s0 + s) * nc * W));
            L.t_d2h_coeff += now_s() - tw0;
        }
        pend_m = m;
        pend_s0 = s0;
        out_pending = true;
        ++ci;
    }
    if (carry_pending) finish_carry();
    /* THE ONE DRAIN OF THE COEFFICIENT READBACK: the last chunk copied but not yet consumed */
    if (out_pending) {
        const double tw0 = now_s();
        CK(stage2_wait::event_sync(g_pin_ev[(ci - 1) & 1],stage2_wait::Kind::output));
        const unsigned long long *prev = g_pin_out[(ci - 1) & 1];
        for (unsigned long long s = 0; s < pend_m; ++s)
            std::copy(prev + ((size_t)s * output_slots + host_first) * W,
                      prev + ((size_t)s * output_slots + host_first) * W + nc * W,
                      out.begin() + (long)((pend_s0 + s) * nc * W));
        L.t_d2h_coeff += now_s() - tw0;
        out_pending = false;
    }
    L.ntt_seconds += now_s() - t0;
    /* the deferred chunks' carry verdict, folded back into the caller's account: their counters
       were read ONCE by ntt_batch_carry_finish instead of once per chunk (section 29), so this is
       the only place their time and residual totals can be reported */
    /* Charge through `st` ONCE below. Directly adding to L here as well double-counted every
       finish; keeping only the last chunk also omitted the first non-deferred readback. */
    st.t_check_d2h = chunk_d2h_acc + carry_d2h_acc;
    st.carry_residual += carry_res_acc;
    st.carry_max_bits = std::max(st.carry_max_bits, carry_bits_acc);
    /* the caller's copy of the per-call account (section 27): the tree needs it PER LEVEL to say
       whether its floor is host packing, H2D/D2H traffic, the transforms, the carry check or the
       exact coefficient extraction -- `t0` above already covers the whole call including the
       host-side copies, so `total` = now - t0 minus the sum of the parts is the unattributed rest */
    if (st_out) {
        *st_out = st;
        st_out->t_total_call = now_s() - t0;
    }
    if (rc != 0) {
        std::fprintf(stderr, "%s: ntt_poly_mul_batch_host failed (rc=%d) at P=%llu S=%d "
                             "nbatch=%llu\n", NTT_PROBE_NAME, rc, (unsigned long long)P,
                     (int)L.S, (unsigned long long)nbatch);
        std::exit(3);
    }
    ++L.ntt_launches;
    L.ntt_calls += nbatch;
    L.count_ntt(qN,nbatch,nbatch/chunk+(nbatch%chunk!=0));
    L.muls += nbatch;
    L.t_fwd += st.t_fwd;
    L.t_inv += st.t_inv;
    L.t_slot += st.t_slot;
    L.t_hpack += st.t_hpack;
    L.t_scan += st.t_scan;
    L.t_h2d_batch += st.t_h2d_batch;
    L.t_check += st.t_check;
    L.t_check_reset += st.t_check_reset;
    L.t_check_kernel += st.t_check_kernel;
    L.t_check_d2h += st.t_check_d2h;
    L.t_plan += st.t_plan;
    L.t_opcopy += st.t_opcopy;
    L.t_hout += st.t_hout;
    L.max_ntt_words = std::max(L.max_ntt_words, st.N);
    L.max_ntt_coeffs = std::max(L.max_ntt_coeffs, (unsigned long long)P);
    L.max_slot_bits = std::max(L.max_slot_bits, st.slot_bits);
    /* the tree's OWN exactness bound, re-derived for THIS shape from the values that came
       back -- never inherited from the probe's shapes nor from another tree level */
    if (!exact_ok_terms(st.L_terms, st.bpw)) {
        std::fprintf(stderr, "%s: EXACTNESS VIOLATED for the tree's shape: L=%llu bpw=%d\n",
                     NTT_PROBE_NAME, (unsigned long long)st.L_terms, st.bpw);
        std::exit(3);
    }
    {
        const double bb = coeff_bound_bits_terms(st.L_terms, st.bpw);
        if (bb > L.bind_bound_bits) {
            L.bind_bound_bits = bb;
            L.bind_P = P;
            L.bind_L = st.L_terms;
            L.bind_slot_bits = st.slot_bits;
            L.bind_slot_words = st.slot_words;
            L.bind_bpw = st.bpw;
        }
    }
    const int c = (cat >= 0) ? cat : L.cat;
    if (c >= 0)
        for (size_t s = 0; s < nbatch; ++s) L.cost.add(c, ma, mb);
    L.slot_checks += nbatch * nc;
    /* Shape lookup remains checked even when the redundant readback is disabled. */
    S4Reduce::Shape *S = C.red->find(st.slot_bits, st.slot_words, st.bpw);
    if (!S) {
        std::fprintf(stderr, "%s: FATAL: the reduction shape changed under the multiply\n",
                     NTT_PROBE_NAME);
        std::exit(3);
    }
    ++g_output_window.calls;
    g_output_window.source_coeffs+=nbatch*out_slots;
    g_output_window.reduced_coeffs+=nbatch*output_slots;
    g_output_window.returned_coeffs+=nbatch*count;
    g_output_window.skipped_coeffs+=nbatch*(out_slots-output_slots);
    if(host_output) g_output_window.d2h_words+=nbatch*output_slots*W;
    g_output_window.device_peak_bytes=std::max(g_output_window.device_peak_bytes,8ull*C.d_out_cap);
    g_output_window.pinned_peak_bytes=std::max(g_output_window.pinned_peak_bytes,8ull*(g_pin_out_cap[0]+g_pin_out_cap[1]));
    ++g_final_readback.calls;
    if (g_s4_final_readback) {
        const double tfinal=now_s();
        std::vector<unsigned long long> all(need, 0ull);
        if(need) CK(stage2_wait::copy(all.data(), C.d_out, need * sizeof(unsigned long long),
                               cudaMemcpyDeviceToHost));
        for (size_t s = 0; s < nbatch; ++s)
            std::copy(all.begin() + (long)((s * output_slots + host_first) * W),
                      all.begin() + (long)((s * output_slots + host_first + nc) * W),
                      out.begin() + (long)(s * nc * W));
        const double elapsed=now_s()-tfinal;
        g_final_readback.copied_words+=need;
        g_final_readback.host_peak_bytes=std::max(g_final_readback.host_peak_bytes,8ull*need);
        g_final_readback.t_copy+=elapsed;
        /* Historical ntt_seconds stopped BEFORE this copy. Include it in the control's call
           time, while keeping coefficient D2H's existing chunk-only ledger separately labelled. */
        L.ntt_seconds+=elapsed;
    } else {
        g_final_readback.avoided_words+=need;
    }
    if (st_out) st_out->t_total_call=now_s()-t0;
    /* Fingerprint the actual FINAL output, after either path; the old trace preceded the
       whole-call overwrite and could not detect a fault introduced by that last copy. */
    if (g_s4_carry_trace) {
        for (const auto v : {(unsigned long long)ma, (unsigned long long)mb,
                            (unsigned long long)nbatch, (unsigned long long)out.size()})
            g_carry_output_hash = (g_carry_output_hash ^ v) * 1099511628211ull;
        for (const auto v : out)
            g_carry_output_hash = (g_carry_output_hash ^ v) * 1099511628211ull;
        g_carry_output_words += (unsigned long long)out.size();
    }
    /* the in-run oracle (the device's coefficients against GMP) runs INSIDE the hook, where
       the digit buffer is still alive */
    ++C.launches;
    C.muls += nbatch;
}

void flat_mul_batch(
    PolyLayer &L,
    const std::vector<unsigned long long> &A,
    size_t ma,
    const std::vector<unsigned long long> &B,
    size_t mb,
    size_t nbatch,
    std::vector<unsigned long long> &out,
    int cat,
    size_t first,
    size_t count)
{
    const size_t W = L.W;
    const size_t P = (ma > mb) ? ma : mb;           /* the multiply takes ONE shape per launch */
    const size_t nc = count==(size_t)-1 ? ma+mb-1 : count;
    /* Every flat consumer uses the declared coefficient stride, including
       padding and unequal operand widths. Select outputs with that stride. */
    if (A.size() != nbatch * ma * W || B.size() != nbatch * mb * W) {
        std::fprintf(stderr, "%s: FATAL: flat_mul_batch operand shape mismatch: A=%llu (must be "
                             "%llu = nbatch*ma*W), B=%llu (must be %llu = nbatch*mb*W) for "
                             "ma=%llu mb=%llu nbatch=%llu W=%llu\n", NTT_PROBE_NAME,
                     (unsigned long long)A.size(), (unsigned long long)(nbatch * ma * W),
                     (unsigned long long)B.size(), (unsigned long long)(nbatch * mb * W),
                     (unsigned long long)ma, (unsigned long long)mb, (unsigned long long)nbatch,
                     (unsigned long long)W);
        std::exit(3);
    }
    const double tp0 = now_s();
    std::vector<unsigned long long> wa, wb;
    const size_t padded_words = nbatch * P * W;
    auto input = [&](const std::vector<unsigned long long> &v, size_t m,
                     std::vector<unsigned long long> &scratch) -> const unsigned long long * {
        const unsigned long long bytes = 8ull * v.size();
        /* poly_mul_batch_modN assigns OUT before reading input. Clone aliases even when
           their stride already matches P. Borrowed sources stay alive for this whole call;
           async uploads copy them to owned pinned staging before returning to this function. */
        if (g_s4_flat_direct && m == P && &v != &out) {
            ++g_flat_input.borrowed;
            g_flat_input.avoided_copy_bytes += bytes;
            g_flat_input.avoided_zero_bytes += 8ull * padded_words;
            return v.data();
        }
        ++g_flat_input.padded;
        if (&v == &out) ++g_flat_input.alias_clones;
        scratch.assign(padded_words, 0ull);
        for (size_t s = 0; s < nbatch; ++s)
            std::copy(v.begin() + (long)(s * m * W), v.begin() + (long)((s + 1) * m * W),
                      scratch.begin() + (long)(s * P * W));
        g_flat_input.copy_bytes += bytes;
        g_flat_input.zero_bytes += 8ull * padded_words;
        return scratch.data();
    };
    const auto *pa = input(A, ma, wa);
    const auto *pb = input(B, mb, wb);
    ++g_flat_input.calls;
    g_flat_input.temp_peak_bytes = std::max(g_flat_input.temp_peak_bytes,
        8ull * (wa.capacity() + wb.capacity()));
    g_flat_input.control_peak_bytes = std::max(g_flat_input.control_peak_bytes, 16ull * padded_words);
    g_flat_input.t_prepare += now_s() - tp0;
    poly_mul_batch_modN(L, pa, pb, ma, mb, nbatch, out, cat, nullptr, first, count);
    /* The multiply must return the exact compact stride. A mismatch must never turn into
       a silently zero-filled polynomial. */
    if (out.size() != nbatch * nc * W) {
        std::fprintf(stderr,"%s: FATAL: compact batch output size mismatch\n",NTT_PROBE_NAME);
        std::exit(3);
    }
}

namespace stage2_pack {
Stats stats;
bool enabled(){return fuse_env_ull("NTT_S4_PACK_ZERO",0)!=0;}
}

int s4_gather_final_input(
    void *ctx,
    const NttShape &sh,
    unsigned long long m,
    unsigned long long *a,
    unsigned long long *b)
{
    const auto &p=*(const S4GatherPack*)ctx;
    if(sh.P!=p.P || sh.N!=p.N || sh.S!=p.bits || sh.bpw!=p.bpw ||
       sh.slot_words!=p.sw || sh.W!=(unsigned long long)p.W)return 3;
    const double start=now_s();
    if(stage2_pack::enabled() && stage2_pack::supported(p.N,p.sw,p.ma,p.bits,p.bpw,p.W) &&
       stage2_pack::supported(p.N,p.sw,p.mb,p.bits,p.bpw,p.W)) {
        stage2_pack::launch<true>(p.batch->src,p.batch->meta+p.s0,m,p.ma,p.W,p.bits,p.bpw,p.sw,p.N,a);
        const auto *offsets=p.batch->meta+p.batch->nbatch+p.s0;
        if(p.batch->reverse_b)
            stage2_pack::launch<true,true>(p.batch->src,offsets,m,p.mb,p.W,p.bits,p.bpw,p.sw,p.N,b);
        else stage2_pack::launch<true>(p.batch->src,offsets,m,p.mb,p.W,p.bits,p.bpw,p.sw,p.N,b);
        stage2_pack::account(m,p.N,true);stage2_pack::account(m,p.N,true,p.batch->reverse_b);
        p.C->t_packdev+=now_s()-start;return 0;
    }
    CK(cudaMemsetAsync(a,0,(size_t)m*sh.N*8));
    CK(cudaMemsetAsync(b,0,(size_t)m*sh.N*8));
    ecm_launch_s4_pack_gather_kernel(EcmCudaLaunch{(unsigned int)((m*p.ma+255)/256),256}, p.batch->src,
        p.batch->meta+p.s0,m,p.ma,p.W,p.bits,p.bpw,p.sw,p.N,a);
    CK(cudaGetLastError());
    if(p.batch->reverse_b)
        ecm_launch_s4_pack_gather_reverse_kernel(EcmCudaLaunch{(unsigned int)((m*p.mb+255)/256),256}, p.batch->src,
            p.batch->meta+p.batch->nbatch+p.s0,m,p.mb,p.W,p.bits,p.bpw,p.sw,p.N,b);
    else
        ecm_launch_s4_pack_gather_kernel(EcmCudaLaunch{(unsigned int)((m*p.mb+255)/256),256}, p.batch->src,
            p.batch->meta+p.batch->nbatch+p.s0,m,p.mb,p.W,p.bits,p.bpw,p.sw,p.N,b);
    CK(cudaGetLastError());
    p.C->t_packdev+=now_s()-start;
    return 0;
}

int s4_pack_final_input(
    void *ctx,
    const NttShape &sh,
    unsigned long long nbatch,
    unsigned long long *dA,
    unsigned long long *dB)
{
    const S4InputPack &p = *(const S4InputPack *)ctx;
    /* The query that budgeted/padded raw operands MUST agree with the engine's actual plan. */
    if (sh.P != p.P || sh.S != p.S || sh.N != p.N || sh.bpw != p.bpw ||
        sh.slot_words != p.slot_words || sh.W != (unsigned long long)p.W) {
        std::fprintf(stderr, "%s: S4 input pack shape disagrees with the NTT plan\n", NTT_PROBE_NAME);
        return 3;
    }
    const double t0 = now_s();
    const size_t bytes = (size_t)nbatch * sh.N * sizeof(unsigned long long);
    if(stage2_pack::enabled() && stage2_pack::supported(p.N,p.slot_words,p.P,p.S,p.bpw,p.W)) {
        stage2_pack::launch<false>(p.C->d_rawA,nullptr,nbatch,p.P,p.W,p.S,p.bpw,p.slot_words,p.N,dA);
        stage2_pack::launch<false>(p.C->d_rawB,nullptr,nbatch,p.P,p.W,p.S,p.bpw,p.slot_words,p.N,dB);
        stage2_pack::account(nbatch,p.N,false);stage2_pack::account(nbatch,p.N,false);
        p.C->t_packdev+=now_s()-t0;return 0;
    }
    /* Reused scratch contains spectra/previous outputs. Clear gaps, window tails and unused
       coefficients on EVERY call, then overwrite the occupied digits with the original packer. */
    CK(cudaMemsetAsync(dA, 0, bytes));
    CK(cudaMemsetAsync(dB, 0, bytes));
    s4_launch_pack_batch(p.C->d_rawA, p.S, p.bpw, p.slot_words, p.N, nbatch, p.P, p.W, dA);
    s4_launch_pack_batch(p.C->d_rawB, p.S, p.bpw, p.slot_words, p.N, nbatch, p.P, p.W, dB);
    p.C->t_packdev += now_s() - t0;   /* host enqueue time, never a pipeline drain */
    return 0;
}
