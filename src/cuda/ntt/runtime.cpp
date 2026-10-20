#include "runtime.h"
#define NTT_HBS(name,t0) do { if(ntt_host_break_on()) hb_##name += now_s()-(t0); } while(0)
FuseBaseStats g_fuse_base;
void NttArena::drop_carry_scratch() {
        if(carry_scratch) { CK(cudaFree(carry_scratch));bytes-=carry_scratch_bytes; }
        carry_scratch=nullptr;carry_scratch_bytes=0;
    }

unsigned int *NttArena::carry_workspace(unsigned long long n,unsigned long long batch) {
        if(!ntt_carry_check_requested())return nullptr;
        if(!batch || n>((size_t)-1)/batch || n*batch<(1ull<<20)) { ++carry_skipped_calls;return nullptr; }
        const size_t blocks=(size_t)((n+255)/256);
        if(blocks>((size_t)-1)/8/batch) { ++carry_refusals;return nullptr; }
        const size_t need=blocks*(size_t)batch*8;
        const char *fail=std::getenv("NTT_CARRY_CHECK_ALLOC_FAIL");
        if(fail && std::atoi(fail)) { ++carry_refusals;return nullptr; }
        if(need>carry_scratch_bytes) {
            if(cap_bytes && (need>cap_bytes || bytes-carry_scratch_bytes>cap_bytes-need)) { ++carry_refusals;return nullptr; }
            drop_carry_scratch();
            if(cudaMalloc((void**)&carry_scratch,need)!=cudaSuccess) { (void)cudaGetLastError();carry_scratch=nullptr;++carry_refusals;return nullptr; }
            carry_scratch_bytes=need;bytes+=need;++carry_grows;
            carry_scratch_peak=std::max(carry_scratch_peak,need);update_peaks();
        }
        ++carry_fused_calls;return carry_scratch;
    }

bool NttArena::try_malloc(unsigned long long **p, size_t bytes) {
        if (cudaMalloc((void **)p, bytes) == cudaSuccess) return true;
        (void)cudaGetLastError();             /* clear the sticky error before anything else runs */
        *p = nullptr;
        return false;
    }

void NttArena::drop_workspace() {
        short_graph.clear();
        /* cudaFree waits for queued default-stream readers, including oracle D2H copies.
           Drop before growth so old/new allocations never overlap in device memory. */
        if (workspace.dA) { CK(cudaFree(workspace.dA)); ++workspace_frees; }
        if (workspace.dB) { CK(cudaFree(workspace.dB)); ++workspace_frees; }
        if (workspace.dQ && !workspace.q_alias_b) { CK(cudaFree(workspace.dQ)); ++workspace_frees; }
        if (workspace.words) bytes -= workspace.words*8;
        workspace=BigEntry{};
    }

bool NttArena::input_span(const unsigned long long *p, size_t &remaining) const {
        if (!p) return false;
        const uintptr_t address=(uintptr_t)p;
        auto contains=[&](const unsigned long long *base, size_t words) {
            if (!base) return false;
            const uintptr_t start=(uintptr_t)base;
            if (address<start || address-start>=words*8) return false;
            if ((address-start)%8) { remaining=0; return true; }
            remaining=words-(address-start)/8; return true;
        };
        if (contains(workspace.dA,workspace.capacity) || contains(workspace.dB,workspace.capacity) ||
            contains(workspace.dQ,workspace.capacity)) return true;
        for (const auto &b : bigs)
            if (contains(b.dA,b.capacity) || contains(b.dB,b.capacity) || contains(b.dQ,b.capacity)) return true;
        for (const auto &b : smalls)
            if (contains(b.dOut,(size_t)(b.out_cap*b.nbatch)) || contains(b.dRes,(size_t)(2*b.nbatch))) return true;
        return false;
    }

NttArena::Payload NttArena::payload() const {
        Payload p;p.big=workspace.words*8;p.digits=carry_scratch_bytes;
        for(const auto &b:bigs)p.big+=b.words*8;
        for(const auto &s:smalls)p.digits+=s.words*8;
        for(const auto &f:fuses) {
            p.table+=(size_t)fuse_table_words(f.fc)*8;
            p.base+=fuse_base_words(f.fc)*8;
        }
        p.total=p.big+p.digits+p.table+p.base;return p;
    }

size_t NttArena::drop_cold_fuse(unsigned long long keep_n) {
        size_t chosen=fuses.size(),largest=0;
        for(size_t i=0;i<fuses.size();++i)if(fuses[i].n!=keep_n) {
            const size_t live=(fuse_table_words(fuses[i].fc)+fuse_base_words(fuses[i].fc))*8;
            if(live>largest){largest=live;chosen=i;}
        }
        if(chosen==fuses.size())return 0;
        short_graph.clear();
        fuse_release(fuses[chosen].fc); // cudaFree waits for all default-stream table readers.
        fuses.erase(fuses.begin()+(long)chosen);bytes-=largest;
        ++phase_fuse_evictions;phase_fuse_evicted_bytes+=largest;return largest;
    }

void NttArena::update_peaks() {
        size_t big=workspace.words*8, small=carry_scratch_bytes, table=0, base=0;
        for (const auto &b : bigs) big+=b.words*8;
        for (const auto &b : smalls) small+=b.words*8;
        for (const auto &f : fuses) { table+=(size_t)fuse_table_words(f.fc)*8; base+=fuse_base_words(f.fc)*8; }
        peak_big_bytes=std::max(peak_big_bytes,big); peak_small_bytes=std::max(peak_small_bytes,small);
        peak_table_bytes=std::max(peak_table_bytes,table);
        peak_owned_bytes=std::max(peak_owned_bytes,big+small+table);
        peak_fuse_base_bytes=std::max(peak_fuse_base_bytes,base);
        peak_full_bytes=std::max(peak_full_bytes,big+small+table+base);
    }

void NttArena::print_workspace_stats() const {
        stage2_log::print(stage2_log::debug,
                    "ntt_workspace_layout: reuse_bq_requested=%d active_alias=%d capacity_words=%llu physical_buffers=%d alias_calls=%llu saved_peak_bytes=%llu\n",
                    (int)workspace_reuse_bq,(int)workspace.q_alias_b,(unsigned long long)workspace.capacity,
                    workspace.words ? (workspace.q_alias_b ? 2 : 3) : 0,workspace_bq_calls,
                    (unsigned long long)workspace_bq_saved_peak);
        short_graph.print();
        const auto measured=payload();
        stage2_log::print(stage2_log::debug, "ntt_arena_accounting: version=2 payload_bytes=%llu cap_bytes=%llu calculated_bytes=%llu mismatch=%d\n",
                    (unsigned long long)bytes,(unsigned long long)cap_bytes,(unsigned long long)measured.total,(int)(bytes!=measured.total));
        stage2_log::print(stage2_log::debug,"ntt_phase_cache_stats: evictions=%llu evicted_bytes=%llu\n",
            phase_fuse_evictions,phase_fuse_evicted_bytes);
        stage2_log::print(stage2_log::debug, "ntt_carry_check_stats: requested=%d fused_calls=%llu skipped_calls=%llu refusals=%llu grows=%llu scratch_bytes=%llu scratch_peak_bytes=%llu\n",
                    (int)ntt_carry_check_requested(),carry_fused_calls,carry_skipped_calls,carry_refusals,carry_grows,
                    (unsigned long long)carry_scratch_bytes,(unsigned long long)carry_scratch_peak);
        size_t big=workspace.words*8, small=carry_scratch_bytes, table=0, base=0;
        for (const auto &b : bigs) big+=b.words*8;
        for (const auto &b : smalls) small+=b.words*8;
        for (const auto &f : fuses) { table+=(size_t)fuse_table_words(f.fc)*8; base+=fuse_base_words(f.fc)*8; }
        stage2_log::print(stage2_log::phases, "ntt_workspace_stats: pool=%d hits=%llu grows=%llu mallocs=%llu frees=%llu "
                    "workspace_bytes=%llu big_bytes=%llu small_bytes=%llu table_bytes=%llu "
                    "owned_bytes=%llu big_peak_bytes=%llu small_peak_bytes=%llu table_peak_bytes=%llu "
                    "owned_peak_bytes=%llu aliases=%llu alias_bytes=%llu evictions=%llu evicted_words=%llu "
                    "legacy_mallocs=%llu legacy_frees=%llu "
                    "fuse_base_bytes=%llu fuse_base_peak_bytes=%llu full_bytes=%llu full_peak_bytes=%llu\n",
                    (int)workspace_pool, workspace_hits, workspace_grows, workspace_mallocs, workspace_frees,
                    (unsigned long long)(workspace.words*8), (unsigned long long)big, (unsigned long long)small,
                    (unsigned long long)table, (unsigned long long)(big+small+table),
                    (unsigned long long)peak_big_bytes, (unsigned long long)peak_small_bytes,
                    (unsigned long long)peak_table_bytes, (unsigned long long)peak_owned_bytes,
                    alias_snapshots, alias_snapshot_bytes, tbl_evictions, tbl_words_freed, legacy_mallocs, legacy_frees,
                    (unsigned long long)base,(unsigned long long)peak_fuse_base_bytes,
                    (unsigned long long)(big+small+table+base),(unsigned long long)peak_full_bytes);
        stage2_log::print(stage2_log::debug, "ntt_fuse_base_stats: compact=%d allocations=%llu frees=%llu live_bytes=%llu peak_bytes=%llu\n",
                    (int)fuse_compact_scratch(),g_fuse_base.allocations,g_fuse_base.frees,
                    g_fuse_base.live_bytes,g_fuse_base.peak_bytes);
    }

unsigned long long NttArena::fuse_table_words(const FuseCtx &c) {
        return c.tables_cached ? fuse_planned_table_words(c) : 0;
    }

void NttArena::fuse_drop_tables(FuseCtx &c) {
        for (int p = 0; p < FUSE_MAX_PASSES; ++p) {
            if (c.passF[p]) cudaFree(c.passF[p]);
            if (c.radF[p]) cudaFree(c.radF[p]);
            if (c.passI[p]) cudaFree(c.passI[p]);
            if (c.radI[p]) cudaFree(c.radI[p]);
            c.passF[p] = c.radF[p] = c.passI[p] = c.radI[p] = nullptr;
        }
        c.tables_cached = false;             /* the per-call path rebuilds them on demand */
    }

unsigned long long NttArena::evict_other_shapes(unsigned long long keep_n,
                                          unsigned long long keep_nbatch) {
        // This default-stream scratch is never exported and contains no deferred verdict.
        // Only shape-local dRes below retains diagnostics across interior chunks.
        short_graph.clear();
        drop_carry_scratch();
        unsigned long long freed = 0;
        for (FuseEntry &e : fuses) {
            if (e.n == keep_n || !e.fc.tables_cached) continue;
            freed += fuse_table_words(e.fc);
            fuse_drop_tables(e.fc);
        }
        for (size_t i = bigs.size(); i-- > 0;) {
            BigEntry &b = bigs[i];
            if (b.n == keep_n && b.nbatch == keep_nbatch) continue;
            if (b.dA) { cudaFree(b.dA); ++legacy_frees; }
            if (b.dB) { cudaFree(b.dB); ++legacy_frees; }
            if (b.dQ) { cudaFree(b.dQ); ++legacy_frees; }
            freed += b.words;
            bigs.erase(bigs.begin() + (long)i);
        }
        for (size_t i = smalls.size(); i-- > 0;) {
            SmallEntry &s = smalls[i];
            if (s.n == keep_n && s.nbatch == keep_nbatch) continue;
            if (s.dOut) cudaFree(s.dOut);
            if (s.dRes) cudaFree(s.dRes);
            freed += s.words;
            smalls.erase(smalls.begin() + (long)i);
        }
        if (freed) { ++tbl_evictions; tbl_words_freed += freed; bytes -= freed * 8; }
        return freed;
    }

NttArena::~NttArena() { release(); }

void NttArena::release() {
        short_graph.release();
        drop_carry_scratch();
        drop_workspace();
        for (FuseEntry &e : fuses) fuse_release(e.fc);
        fuses.clear();
        for (BigEntry &b : bigs) {
            if (b.dA) { cudaFree(b.dA); ++legacy_frees; }
            if (b.dB) { cudaFree(b.dB); ++legacy_frees; }
            if (b.dQ) { cudaFree(b.dQ); ++legacy_frees; }
        }
        bigs.clear();
        for (SmallEntry &b : smalls) {
            if (b.dOut) cudaFree(b.dOut);
            if (b.dRes) cudaFree(b.dRes);
        }
        smalls.clear();
        bytes = 0;
    }

double NttArena::mb() const { return (double)bytes / 1048576.0; }
unsigned int ntt_gl_reduce_requested_mode()
{
    const char *env=std::getenv("NTT_GL_SHORT_REDUCE");
    const char *ptx=std::getenv("NTT_GL_PTX_REDUCE");
    const unsigned int default_short=NTT_GL_FIXED_MODE < 0 ? NTT_GL_SHORT_REDUCE_DEFAULT : NTT_GL_FIXED_MODE&1;
    const unsigned int default_ptx=NTT_GL_FIXED_MODE < 0 ? 0 : NTT_GL_FIXED_MODE>>1;
    const unsigned int short_enabled=env ? (std::strtoull(env,nullptr,10)!=0) : default_short;
    const unsigned int ptx_enabled=ptx ? (std::strtoull(ptx,nullptr,10)!=0) : default_ptx;
    const unsigned int mode=short_enabled ? 1u | (ptx_enabled ? 2u : 0u) : 0u;
#if NTT_GL_FIXED_MODE != -1
    if(mode!=NTT_GL_FIXED_MODE) {
        std::fprintf(stderr,"ntt_gl_backend_conflict: compiled=%d requested=%u\n",NTT_GL_FIXED_MODE,mode);
        std::exit(2);
    }
#endif
    return mode;
}

void ntt_gl_reduce_configure()
{
    const unsigned int enabled=ntt_gl_reduce_requested_mode();
    int device=0;CK(cudaGetDevice(&device));
    static std::map<int,unsigned int> configured;
    const auto it=configured.find(device);
    if(it!=configured.end() && it->second==enabled)return;
#if NTT_GL_FIXED_MODE == -1
    CK(ecm_ntt_copy_reduce_mode(enabled));
#endif
    configured[device]=enabled;
    stage2_log::print(stage2_log::debug, "ntt_gl_reduce_mode: device=%d short=%u ptx=%u fixed=%d\n",device,enabled&1u,enabled>>1,NTT_GL_FIXED_MODE);
}

double now_s()
{
    using clock = std::chrono::steady_clock;
    static const clock::time_point t0 = clock::now();
    return std::chrono::duration<double>(clock::now() - t0).count();
}

void ntt_bitrev_inplace(unsigned long long *d, unsigned long long n, int k)
{
    const unsigned int threads = 256;
    const unsigned int blocks = (unsigned int)((n + threads - 1) / threads);
    ecm_launch_bitrev_swap_kernel(EcmCudaLaunch(blocks, threads), d, (unsigned long long)k, n);
    CK(cudaGetLastError());
}

void ntt_dif(unsigned long long *d, unsigned long long n, int k,
                    unsigned long long omega)
{
    const unsigned int threads = 256;
    for (int s = k - 1; s >= 0; --s) {          /* DIF: half = N/2 ... 1 */
        const unsigned long long half = 1ull << s;
        const unsigned long long exp_step = n / (2 * half);
        const unsigned int bl = (unsigned int)((n / 2 + threads - 1) / threads);
        ecm_launch_ntt_stage_kernel(EcmCudaLaunch(bl, threads), d, half, exp_step, omega, n);
        CK(cudaGetLastError());
    }
}

void ntt_bitrev_perm(unsigned long long *d, unsigned long long *scratch,
                            unsigned long long n, int k)
{
    const unsigned int threads = 256;
    const unsigned int blocks = (unsigned int)((n + threads - 1) / threads);
    ecm_launch_bitrev_kernel(EcmCudaLaunch(blocks, threads), d, scratch, (unsigned long long)k, n);
    CK(cudaGetLastError());
    CK(cudaMemcpy(d, scratch, n * sizeof(unsigned long long), cudaMemcpyDeviceToDevice));
}

void ntt_forward(unsigned long long *d, unsigned long long n, int k,
                        unsigned long long omega, unsigned long long *scratch)
{
    (void)scratch;
    ntt_dif(d, n, k, omega);   /* halves DESCENDING: natural in, bit-reversed spectrum out */
}

void ntt_inverse(unsigned long long *d, unsigned long long n, int k,
                        unsigned long long omega_inv, unsigned long long n_scale,
                        unsigned long long *scratch)
{
    (void)scratch;
    const unsigned int threads = 256;
    const unsigned int blocks = (unsigned int)((n + threads - 1) / threads);
    /* DIT, halves ASCENDING, same index decomposition: consumes the bit-reversed frequency
       vector the forward produced, and gives the natural-order result. */
    for (int s = 0; s < k; ++s) {
        const unsigned long long half = 1ull << s;
        const unsigned long long exp_step = n / (2 * half);
        const unsigned int bl = (unsigned int)((n / 2 + threads - 1) / threads);
        ecm_launch_dit_stage_kernel(EcmCudaLaunch(bl, threads), d, half, exp_step, omega_inv, n);
        CK(cudaGetLastError());
    }
    ecm_launch_scale_kernel(EcmCudaLaunch(blocks, threads), d, n, n_scale);
    CK(cudaGetLastError());
}

bool ntt_carry_check_requested()
{
    const char *e=std::getenv("NTT_CARRY_CHECK_FUSED");return e && std::atoi(e)!=0;
}

void ntt_tile_attributes(bool warp, int smem, bool preferred)
{
    const void *fwd=warp ? ecm_ntt_tile_handle(false,true) : ecm_ntt_tile_handle(false,false);
    const void *inv=warp ? ecm_ntt_tile_handle(true,true) : ecm_ntt_tile_handle(true,false);
    if(preferred) {
        cudaFuncSetAttribute(fwd,cudaFuncAttributePreferredSharedMemoryCarveout,100);
        cudaFuncSetAttribute(inv,cudaFuncAttributePreferredSharedMemoryCarveout,100);
    }
    CK(cudaFuncSetAttribute(fwd,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
    CK(cudaFuncSetAttribute(inv,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
}

void fuse_base_allocate(unsigned long long **p, size_t words)
{
    if (!words) { *p=nullptr; return; }
    CK(cudaMalloc((void **)p,words*8));
    ++g_fuse_base.allocations; g_fuse_base.live_bytes+=words*8;
    g_fuse_base.peak_bytes=std::max(g_fuse_base.peak_bytes,g_fuse_base.live_bytes);
}

void fuse_base_free(unsigned long long *&p, size_t words)
{
    if (!p) return;
    CK(cudaFree(p)); p=nullptr;
    ++g_fuse_base.frees; g_fuse_base.live_bytes-=words*8;
}

void fuse_plan_stages(FuseCtx &c, int rem)
{
    static int best[64], bestm[64], bestlast[64];
    for (int r = 0; r <= rem; ++r) { best[r] = 1 << 20; bestm[r] = 0; bestlast[r] = 0; }
    best[0] = 0;
    for (int r = 1; r <= rem; ++r) {
        const int mx = (r < c.m_max) ? r : c.m_max;
        for (int m = mx; m >= 1; --m) {
            const int cand = best[r - m] + 1;
            const int last = (r - m == 0) ? m : bestlast[r - m];
            if (cand < best[r] || (cand == best[r] && last > bestlast[r])) {
                best[r] = cand;
                bestm[r] = m;
                bestlast[r] = last;
            }
        }
    }
    int tmp[8], n = 0, r = rem;
    while (r > 0 && n < 8) { tmp[n++] = bestm[r]; r -= bestm[r]; }
    /* the peel is already widest-first, which is the order the stages must be done in */
    for (int i = 0; i < n; ++i) c.ms[i] = tmp[i];
    int sum = 0;
    for (int i = 0; i < n; ++i) sum += c.ms[i];
    if (sum != rem) {                       /* cannot happen; loud if the DP is ever changed */
        std::fprintf(stderr, NTT_PROBE_NAME ": fuse plan does not cover %d stages (sum=%d)\n",
                     rem, sum);
        std::exit(3);
    }
    c.nms = n;
    c.outer_stages = rem;
}

unsigned long long fuse_env_ull(const char *name, unsigned long long dflt)
{
    const char *s = std::getenv(name);
    if (!s || !*s) return dflt;
    return std::strtoull(s, nullptr, 10);
}

bool fuse_compact_scratch()
{
    const char *e=std::getenv("NTT_FUSE_COMPACT_SCRATCH");
    return !e || !*e || std::atoi(e)!=0;
}

bool fuse_shape_device_supported()
{
    // The measured table is specific to this device. Cache properties by device,
    // rather than assuming the current context is the one seen on the first call.
    int device=0;CK(cudaGetDevice(&device));
    static std::map<int,bool> supported;
    auto it=supported.find(device);
    if(it!=supported.end())return it->second;
    cudaDeviceProp prop{};CK(cudaGetDeviceProperties(&prop,device));
    const bool ok=prop.major==8 && prop.minor==9 && std::strstr(prop.name,"RTX 4060 Laptop");
    supported.emplace(device,ok);return ok;
}

bool fuse_shape_policy_supported(int t)
{
    return t==12 && fuse_env_ull("NTT_FUSE_M",4)==4 &&
        fuse_env_ull("NTT_FUSE_WARP_TAIL",0)!=0 && fuse_compact_scratch() &&
        fuse_shape_device_supported();
}

bool fuse_joint_policy_supported()
{
    return fuse_env_ull("NTT_FUSE_JOINT",0)!=0 && fuse_env_ull("NTT_FUSE_COOP_OUTER",0)==2 &&
        fuse_env_ull("NTT_FUSE_T",12)==12 && fuse_shape_policy_supported(12);
}

int fuse_selected_tile(int k,int requested)
{
    return k==23 && requested==12 && fuse_joint_policy_supported() ? 11 : requested;
}

int fuse_outer_max(int t,int k,bool &coop)
{
    const auto mode=fuse_env_ull("NTT_FUSE_COOP_OUTER",0);
    if(mode==2) {
        // Deterministic measured policy: small/unmeasured shapes keep M4.
        // COOP_M is the forced-mode override only; it cannot alter this table.
        const bool joint=fuse_joint_policy_supported();
        const bool middle=k>=21 && k<=23 && (fuse_env_ull("NTT_FUSE_MID_COOP",0)!=0 || joint);
        coop=((k>=24 && k<=27) || middle) && (fuse_shape_policy_supported(t) || (joint && k==23 && t==11));
        // Experimental middle-length table. Keep the measured long table and
        // all unsupported devices/tiles unchanged. The arena keys m_max/coop.
        if(coop && middle)return 6;
        if(coop && joint && k==25)return 7;
        if(coop)return k==24 ? 6 : 8;
    } else coop=mode!=0 && t>=5;
    const int maximum=coop ? 8 : FUSE_MAX_M;
    const int requested=(int)fuse_env_ull(coop ? "NTT_FUSE_COOP_M" : "NTT_FUSE_M",maximum);
    return std::max(coop ? 5 : 1,std::min(maximum,requested));
}

void fuse_describe(FuseCtx &c, unsigned long long n, int k, unsigned long long omega,
                      unsigned long long omega_inv, bool compact)
{
    ntt_gl_reduce_configure();
    c.n = n;
    c.k = k;
    int t = (int)fuse_env_ull("NTT_FUSE_T", 12);
    if (t > k) t = k;
    if (t < 0) t = 0;
    t=fuse_selected_tile(k,t);
    c.t = t;
    c.warp_tail = fuse_env_ull("NTT_FUSE_WARP_TAIL",0)!=0;
    c.m_max=fuse_outer_max(t,k,c.coop_outer);
    c.compact_scratch=compact;
    fuse_plan_stages(c, k - t);

    c.scrWords = compact ? 0 : (size_t)(n >> 1) + 64;
    c.scr2Words = compact ? 0 : (c.coop_outer ? (size_t)(1ull<<c.m_max) : 64);
    if (compact) {
        int L=0;
        for (int p=0;p<c.nms;++p) {
            const int M=c.ms[p];
            c.scrWords=std::max(c.scrWords,(size_t)(n>>(L+M)));
            c.scr2Words=std::max(c.scr2Words,(size_t)(1ull<<M)); L+=M;
        }
        L=c.outer_stages;
        for (int p=c.nms-1;p>=0;--p) {
            const int M=c.ms[p]; L-=M;
            c.scrWords=std::max(c.scrWords,(size_t)(1ull<<(c.k-L-M)));
        }
    }
    c.passes_fwd = c.nms + (t > 0 ? 1 : 0);
}

unsigned long long fuse_planned_table_words(const FuseCtx &c)
{
    unsigned long long w=0; int L=0;
    for(int p=0;p<c.nms;++p) { const int M=c.ms[p]; w+=(c.n>>(L+M))+(1ull<<M); L+=M; }
    L=c.outer_stages;
    for(int p=c.nms-1;p>=0;--p) { const int M=c.ms[p]; L-=M; w+=(1ull<<(c.k-L-M))+(1ull<<M); }
    return w;
}

size_t fuse_planned_base_words(const FuseCtx &c)
{
    return 2*(size_t)(1ull<<c.t)+c.scrWords+c.scr2Words;
}

void fuse_init(FuseCtx &c, unsigned long long n, int k, unsigned long long omega,
                      unsigned long long omega_inv, bool compact)
{
    fuse_describe(c,n,k,omega,omega_inv,compact);
    const int t=c.t;
    fuse_base_allocate(&c.tblF,(size_t)(1ull << t));
    fuse_base_allocate(&c.tblI,(size_t)(1ull << t));
    if (t > 0) {
        const unsigned int tb = (unsigned int)(((1ull << t) + 255) / 256);
        ecm_launch_build_tile_table_kernel(EcmCudaLaunch(tb, 256), c.tblF, k, t, omega);
        CK(cudaGetLastError());
        ecm_launch_build_tile_table_kernel(EcmCudaLaunch(tb, 256), c.tblI, k, t, omega_inv);
        CK(cudaGetLastError());
    }
    fuse_base_allocate(&c.scr,c.scrWords);
    fuse_base_allocate(&c.scr2,c.scr2Words);
    /* 2^t words of dynamic shared memory for the tile pass (the twiddle table is in global).
       The carveout hint matters: without it the driver may leave the L1/shared split too
       small for more than one block per SM, and the tile pass is latency bound (its modular
       multiply chain), so the extra resident blocks are what hides that latency. */
    const int smem = (int)(1ull << t) * (int)sizeof(unsigned long long);
    ntt_tile_attributes(c.warp_tail && t>=5,smem,true);
    c.passes_fwd = c.nms + (t > 0 ? 1 : 0);
    c.ready = true;
}

void fuse_release(FuseCtx &c)
{
    for (int p = 0; p < FUSE_MAX_PASSES; ++p) {
        if (c.passF[p]) cudaFree(c.passF[p]);
        if (c.radF[p]) cudaFree(c.radF[p]);
        if (c.passI[p]) cudaFree(c.passI[p]);
        if (c.radI[p]) cudaFree(c.radI[p]);
        c.passF[p] = c.radF[p] = c.passI[p] = c.radI[p] = nullptr;
    }
    c.tables_cached = false;
    fuse_base_free(c.tblF,(size_t)(1ull << c.t));
    fuse_base_free(c.tblI,(size_t)(1ull << c.t));
    fuse_base_free(c.scr,c.scrWords);
    fuse_base_free(c.scr2,c.scr2Words);
    c.tblF = c.tblI = c.scr = c.scr2 = nullptr;
    c.ready = false;
    c.arena_borrowed = false;
}

size_t fuse_base_words(const FuseCtx &c)
{
    return (c.tblF ? (size_t)(1ull<<c.t) : 0) + (c.tblI ? (size_t)(1ull<<c.t) : 0)
         + (c.scr ? c.scrWords : 0) + (c.scr2 ? c.scr2Words : 0);
}

void ntt_fuse_cache_tables(FuseCtx &c, unsigned long long omega,
                                  unsigned long long omega_inv)
{
    if (c.tables_cached) return;
    {   /* forward: outer passes over stages k-1..t, ascending L; same for() as
           ntt_forward_fused so the (S, L, baseW) of pass p are identical. */
        int L = 0;
        for (int p = 0; p < c.nms; ++p) {
            const int M = c.ms[p];
            const unsigned long long S = c.n >> (L + M);
            CK(cudaMalloc(&c.passF[p], (size_t)S * sizeof(unsigned long long)));
            CK(cudaMalloc(&c.radF[p], (size_t)(1 << M) * sizeof(unsigned long long)));
            const unsigned int tb = (unsigned int)((S + 255) / 256);
            ecm_launch_build_pass_table_kernel(EcmCudaLaunch(tb, 256), c.passF[p], S, L, omega);
            CK(cudaGetLastError());
            const unsigned long long baseW = gl_pow_host(omega, S << L);
            ecm_launch_build_radix_table_kernel(EcmCudaLaunch(1, (1 << M)), c.radF[p], M, baseW, 0);
            CK(cudaGetLastError());
            L += M;
        }
    }
    {   /* inverse: same bookkeeping as ntt_inverse_fused (L walks back DOWN from k-t) */
        int L = c.outer_stages;
        for (int p = c.nms - 1; p >= 0; --p) {
            const int M = c.ms[p];
            L -= M;
            const int s0 = c.k - L - M;
            const unsigned long long S = 1ull << s0;
            CK(cudaMalloc(&c.passI[p], (size_t)S * sizeof(unsigned long long)));
            CK(cudaMalloc(&c.radI[p], (size_t)(1 << M) * sizeof(unsigned long long)));
            const unsigned int tb = (unsigned int)((S + 255) / 256);
            ecm_launch_build_pass_table_kernel(EcmCudaLaunch(tb, 256), c.passI[p], S, L, omega_inv);
            CK(cudaGetLastError());
            const unsigned long long Wbase = gl_pow_host(omega_inv, 1ull << (c.k - M));
            ecm_launch_build_radix_table_kernel(EcmCudaLaunch(1, (1 << M)), c.radI[p], M, Wbase, 1);
            CK(cudaGetLastError());
        }
    }
    CK(cudaDeviceSynchronize());
    c.tables_cached = true;
}

unsigned long long fuse_grid(unsigned long long groups, unsigned long long grp_per_blk)
{
    return (groups + grp_per_blk - 1) / grp_per_blk;
}

bool fuse_trace_on()
{
    static const bool on = (std::getenv("NTT_FUSE_TRACE") != nullptr);
    return on;
}

void fuse_mark(const char *tag, double &t0)
{
    if (!fuse_trace_on()) return;
    CK(cudaDeviceSynchronize());
    const double t1 = now_s();
    stage2_log::print(stage2_log::debug, "    [trace] %-28s %.3f ms\n", tag, (t1 - t0) * 1e3);
    t0 = t1;
}

void ntt_forward_fused(unsigned long long *d, const FuseCtx &c,
                              unsigned long long omega, unsigned long long nbatch, cudaStream_t stream)
{
    unsigned long long n = c.n;
    const unsigned long long stride = n;
    double ft0 = now_s();
    int L = 0;
    for (int p = 0; p < c.nms; ++p) {
        const int M = c.ms[p];
        const unsigned long long S = n >> (L + M);
        const unsigned long long groups = n >> M;
        const unsigned long long gpb = 1024;
        /* cached tables (S3): the same kernel wrote them once per SHAPE instead of once per
           pass of every multiply; when tables_cached is false this is exactly the old path */
        unsigned long long *scr = c.scr, *scr2 = c.scr2;
        if (c.tables_cached) { scr = c.passF[p]; scr2 = c.radF[p]; }
        else {
            ecm_launch_build_pass_table_kernel(EcmCudaLaunch((unsigned int)fuse_grid(S, 256), 256, 0, stream), scr, S, L, omega);
            CK(cudaGetLastError());
            const unsigned long long baseW = gl_pow_host(omega, S << L);
            ecm_launch_build_radix_table_kernel(EcmCudaLaunch(1, (1 << M), 0, stream), scr2, M, baseW, 0);
            CK(cudaGetLastError());
        }
        const unsigned int g = (unsigned int)fuse_grid(groups, gpb);
        dim3 gr(g, (unsigned int)nbatch);
        if(c.coop_outer && M>=5)
            launch_outer_coop<false>(M,d,n,L,scr,scr2,nbatch,stride,stream);
        else {
        switch (M) {
            case 1: ecm_launch_outer_fwd_kernel<1>(EcmCudaLaunch(gr, FUSE_OUTER_THREADS, 0, stream), d, n, L, scr, scr2, gpb, stride); break;
            case 2: ecm_launch_outer_fwd_kernel<2>(EcmCudaLaunch(gr, FUSE_OUTER_THREADS, 0, stream), d, n, L, scr, scr2, gpb, stride); break;
            case 3: ecm_launch_outer_fwd_kernel<3>(EcmCudaLaunch(gr, FUSE_OUTER_THREADS, 0, stream), d, n, L, scr, scr2, gpb, stride); break;
            case 4: ecm_launch_outer_fwd_kernel<4>(EcmCudaLaunch(gr, FUSE_OUTER_THREADS, 0, stream), d, n, L, scr, scr2, gpb, stride); break;
            case 5: ecm_launch_outer_fwd_kernel<5>(EcmCudaLaunch(gr, FUSE_OUTER_THREADS, 0, stream), d, n, L, scr, scr2, gpb, stride); break;
            case 6: ecm_launch_outer_fwd_kernel<6>(EcmCudaLaunch(gr, FUSE_OUTER_THREADS, 0, stream), d, n, L, scr, scr2, gpb, stride); break;
            default: std::fprintf(stderr, NTT_PROBE_NAME ": bad outer radix M=%d\n", M); std::exit(3);
        }
        }
        CK(cudaGetLastError());
        fuse_mark("fwd outer pass", ft0);
        L += M;
    }
    if (c.t > 0) {
        const unsigned int blocks = (unsigned int)(n >> c.t);
        const int smem = (int)(1ull << c.t) * (int)sizeof(unsigned long long);
        dim3 gr(blocks, (unsigned int)nbatch);
        if(c.warp_tail && c.t>=5)
            ecm_launch_tile_kernel<false,true>(EcmCudaLaunch(gr, FUSE_TILE_THREADS, smem, stream), d,nullptr,n,c.k,c.t,c.tblF,0ull,stride);
        else ecm_launch_tile_kernel<false>(EcmCudaLaunch(gr, FUSE_TILE_THREADS, smem, stream), d, nullptr, n, c.k, c.t,
                                                            c.tblF, 0ull, stride);
        CK(cudaGetLastError());
        fuse_mark("fwd tile pass", ft0);
    }
}

void ntt_inverse_fused(unsigned long long *d, const unsigned long long *b,
                              const FuseCtx &c, unsigned long long omega_inv,
                              unsigned long long n_scale, unsigned long long nbatch, cudaStream_t stream)
{
    unsigned long long n = c.n;
    const unsigned long long stride = n;
    double ft0 = now_s();
    if (c.t > 0) {
        const unsigned int blocks = (unsigned int)(n >> c.t);
        const int smem = (int)(1ull << c.t) * (int)sizeof(unsigned long long);
        dim3 gr(blocks, (unsigned int)nbatch);
        const bool shift_scale=fuse_env_ull("NTT_GL_SHIFT_SCALE",0)!=0;
        if(c.warp_tail && c.t>=5)
            ecm_launch_tile_kernel<true,true>(EcmCudaLaunch(gr, FUSE_TILE_THREADS, smem, stream), d,b,n,c.k,c.t,c.tblI,n_scale,stride,shift_scale);
        else ecm_launch_tile_kernel<true>(EcmCudaLaunch(gr, FUSE_TILE_THREADS, smem, stream), d, b, n, c.k, c.t, c.tblI,
                                                           n_scale, stride,shift_scale);
        CK(cudaGetLastError());
        fuse_mark("inv tile pass (+pw+scale)", ft0);
    }
    int L = c.outer_stages;                 /* stages already done walking DOWN from k-1 */
    for (int p = c.nms - 1; p >= 0; --p) {
        const int M = c.ms[p];
        L -= M;                             /* prefix sum of the passes before p */
        const int s0 = c.k - L - M;         /* lowest stage of this pass (ascending now) */
        const unsigned long long S = 1ull << s0;
        const unsigned long long groups = n >> M;
        const unsigned long long gpb = 1024;
        unsigned long long *scr = c.scr, *scr2 = c.scr2;
        if (c.tables_cached) { scr = c.passI[p]; scr2 = c.radI[p]; }
        else {
            ecm_launch_build_pass_table_kernel(EcmCudaLaunch((unsigned int)fuse_grid(S, 256), 256, 0, stream), scr, S, L,
                                                                             omega_inv);
            CK(cudaGetLastError());
            const unsigned long long Wbase = gl_pow_host(omega_inv, 1ull << (c.k - M));
            ecm_launch_build_radix_table_kernel(EcmCudaLaunch(1, (1 << M), 0, stream), scr2, M, Wbase, 1);
            CK(cudaGetLastError());
        }
        const unsigned int g = (unsigned int)fuse_grid(groups, gpb);
        dim3 gr(g, (unsigned int)nbatch);
        if(c.coop_outer && M>=5)
            launch_outer_coop<true>(M,d,n,s0,scr,scr2,nbatch,stride,stream);
        else {
        switch (M) {
            case 1: ecm_launch_outer_inv_kernel<1>(EcmCudaLaunch(gr, FUSE_OUTER_THREADS, 0, stream), d, n, s0, scr, scr2, gpb, stride); break;
            case 2: ecm_launch_outer_inv_kernel<2>(EcmCudaLaunch(gr, FUSE_OUTER_THREADS, 0, stream), d, n, s0, scr, scr2, gpb, stride); break;
            case 3: ecm_launch_outer_inv_kernel<3>(EcmCudaLaunch(gr, FUSE_OUTER_THREADS, 0, stream), d, n, s0, scr, scr2, gpb, stride); break;
            case 4: ecm_launch_outer_inv_kernel<4>(EcmCudaLaunch(gr, FUSE_OUTER_THREADS, 0, stream), d, n, s0, scr, scr2, gpb, stride); break;
            case 5: ecm_launch_outer_inv_kernel<5>(EcmCudaLaunch(gr, FUSE_OUTER_THREADS, 0, stream), d, n, s0, scr, scr2, gpb, stride); break;
            case 6: ecm_launch_outer_inv_kernel<6>(EcmCudaLaunch(gr, FUSE_OUTER_THREADS, 0, stream), d, n, s0, scr, scr2, gpb, stride); break;
            default: std::fprintf(stderr, NTT_PROBE_NAME ": bad outer radix M=%d\n", M); std::exit(3);
        }
        }
        CK(cudaGetLastError());
        fuse_mark("inv outer pass", ft0);
    }
}

NttArena::BufEntry *ntt_arena_bufs(NttArena *ar, unsigned long long n,
                                          unsigned long long out_slots,
                                          unsigned long long nbatch, bool allow_pool)
{
    if (!ar) return nullptr;
    if (nbatch == 0) nbatch = 1;
    if (!n || n>((size_t)-1)/24/nbatch || out_slots>((size_t)-1)/8/nbatch-2) {
        ++ar->overflow; return nullptr;
    }
    /* The three BUFFER-SIZE-INDEPENDENT buffers are cached by (N, nbatch) alone.  Keying them by
       (N, out_slots) -- as the first S4 version did -- multiplies the arena by the number of
       distinct out_slots that share one N, and at the real shape the fold's several multiplies
       at N = 2^27 with different out_slots would each have wanted their own 3*N words (3 GB).
       Only dOut (out_slots words) and dRes really depend on out_slots. */
    NttArena::BigEntry *big = nullptr;
    if (ar->workspace_pool && allow_pool) {
        if (!nbatch || n>((size_t)-1)/nbatch || n*nbatch>((size_t)-1)/24) { ++ar->overflow; return nullptr; }
        const size_t required=(size_t)(n*nbatch);
        const size_t buffers=ar->big_buffer_count(allow_pool);
        if (ar->workspace.capacity>=required && ar->workspace.q_alias_b==ar->workspace_reuse_bq) {
            big=&ar->workspace; ++ar->workspace_hits; ++ar->buf_hits;
        }
        else {
            ar->drop_workspace();
            const size_t need=required*buffers*8;
            if (ar->cap_bytes && (need>ar->cap_bytes || ar->bytes>ar->cap_bytes-need))
                ar->evict_other_shapes(n,nbatch);
            if (ar->cap_bytes && (need>ar->cap_bytes || ar->bytes>ar->cap_bytes-need)) {
                ++ar->overflow; return nullptr;
            }
            NttArena::BigEntry e;
            auto allocate=[&](unsigned long long **p, int index) {
                if (ar->workspace_fail_alloc==index) { *p=nullptr; return false; }
                ++ar->workspace_mallocs; return NttArena::try_malloc(p,required*8);
            };
            if (!allocate(&e.dA,1) || !allocate(&e.dB,2) ||
                (!ar->workspace_reuse_bq && !allocate(&e.dQ,3))) {
                if (e.dA) { CK(cudaFree(e.dA)); ++ar->workspace_frees; }
                if (e.dB) { CK(cudaFree(e.dB)); ++ar->workspace_frees; }
                if (e.dQ) { CK(cudaFree(e.dQ)); ++ar->workspace_frees; }
                ++ar->overflow; return nullptr;
            }
            e.n=n; e.nbatch=nbatch; e.capacity=required;
            e.q_alias_b=ar->workspace_reuse_bq;
            if(e.q_alias_b)e.dQ=e.dB;
            e.words=required*buffers;
            ar->workspace=e; ar->bytes+=need; big=&ar->workspace;
            ++ar->workspace_grows; ++ar->buf_builds; ar->update_peaks();
        }
        if(big->q_alias_b) {
            ++ar->workspace_bq_calls;
            ar->workspace_bq_saved_peak=std::max(ar->workspace_bq_saved_peak,big->capacity*8);
        }
    } else {
        /* Exported digits retain the keyed-cache lifetime. No shared scratch reader remains
           after host input snapshots and prior default-stream work have completed. */
        if (ar->workspace_pool) ar->drop_workspace();
    }
    for (NttArena::BigEntry &b : ar->bigs)
        if (!big)
        if (b.n == n && b.nbatch == nbatch) { big = &b; ++ar->buf_hits; break; }
    if (!big) {
        const size_t need = (size_t)(3 * n * nbatch) * sizeof(unsigned long long);
        /* BEFORE REFUSING, EVICT EVERYTHING THE ARENA CACHES FOR OTHER SHAPES (see the NttArena
           comment).  The table caches alone were not enough: at P=115200 the fold's shape needed
           3072 MB while the arena still held 5627 MB of earlier shapes' BIG buffers, so the run
           fell back to per-call cudaMalloc and died with "out of memory" (measured).  The hot
           shape keeps its own entry, and every evicted shape is rebuilt on demand. */
        if (ar->cap_bytes && (need>ar->cap_bytes || ar->bytes>ar->cap_bytes-need))
            ar->evict_other_shapes(n, nbatch);
        if (ar->cap_bytes && (need>ar->cap_bytes || ar->bytes>ar->cap_bytes-need)) {
            ++ar->overflow;
            if (ar->overflow <= 4)
                std::fprintf(stderr, "%s: arena refuses N=%llu nbatch=%llu (needs %.0f MB, has "
                                     "%.0f of %.0f MB): falling back to per-call cudaMalloc\n",
                             NTT_PROBE_NAME, n, nbatch, need / 1048576.0, ar->bytes / 1048576.0,
                             ar->cap_bytes / 1048576.0);
            return nullptr;
        }
        NttArena::BigEntry e;
        e.n = n;
        e.nbatch = nbatch;
        e.capacity = (size_t)(n*nbatch);
        /* the three big buffers are the whole cost of a shape, so this is where a real out-of-
           memory shows up first -- and it is a cache, so it degrades instead of aborting */
        auto allocate_legacy=[&](unsigned long long **p) {
            ++ar->legacy_mallocs; return NttArena::try_malloc(p,n*nbatch*sizeof(unsigned long long));
        };
        if (!allocate_legacy(&e.dA) || !allocate_legacy(&e.dB) || !allocate_legacy(&e.dQ)) {
            if (e.dA) { cudaFree(e.dA); ++ar->legacy_frees; }
            if (e.dB) { cudaFree(e.dB); ++ar->legacy_frees; }
            if (e.dQ) { cudaFree(e.dQ); ++ar->legacy_frees; }
            ++ar->overflow;
            if (ar->overflow <= 4)
                std::fprintf(stderr, "%s: arena could NOT allocate N=%llu nbatch=%llu (%.0f MB): "
                                     "the device is out of memory (the cap says %.0f of %.0f MB "
                                     "free) -- falling back to per-call cudaMalloc\n",
                             NTT_PROBE_NAME, n, nbatch, need / 1048576.0, ar->bytes / 1048576.0,
                             ar->cap_bytes / 1048576.0);
            return nullptr;
        }
        e.words = (size_t)(3 * n * nbatch);
        ar->bytes += need;
        ar->bigs.push_back(e);
        ++ar->buf_builds;
        big = &ar->bigs.back();
        ar->update_peaks();
    }
    /* The dOut buffer is cached by (N, nbatch) and merely GROWN to the largest out_slots at that
       N.  Keying it by (N, out_slots) -- the second version -- was catastrophic at the real
       shape: every P in a range maps to the SAME power-of-two N but a DIFFERENT out_slots, so
       one N accumulated thousands of small entries whose sum (GBs) blew the cap and pushed the
       largest shape onto the per-call path, where the allocation then failed. */
    NttArena::SmallEntry *small = nullptr;
    for (NttArena::SmallEntry &b : ar->smalls)
        if (b.n == n && b.nbatch == nbatch) { small = &b; ++ar->buf_hits; break; }
    if (small && out_slots > small->out_cap) {
        /* grow: the extra words are charged to the cap before anything is allocated */
        const size_t extra = (size_t)((out_slots - small->out_cap) * nbatch) *
                             sizeof(unsigned long long);
        if (ar->cap_bytes && (extra>ar->cap_bytes || ar->bytes>ar->cap_bytes-extra)) {
            ++ar->overflow;
            return nullptr;
        }
        const size_t old_bytes=(size_t)(small->out_cap*nbatch)*8;
        if (small->dOut) cudaFree(small->dOut);
        small->dOut = nullptr;
        small->out_cap = 0;
        small->words=(size_t)(2*nbatch);
        ar->bytes-=old_bytes;
        if (!NttArena::try_malloc(&small->dOut, out_slots * nbatch * sizeof(unsigned long long))) {
            /* out_cap is left at 0, so the next call at this (N, nbatch) simply retries the
               growth -- the entry stays valid, only unusable until an allocation succeeds */
            ++ar->overflow;
            return nullptr;
        }
        small->out_cap = out_slots;
        small->words = (size_t)(out_slots * nbatch + 2 * nbatch);
        ar->bytes += (size_t)(out_slots*nbatch)*8;
    } else if (!small) {
        const size_t need = (size_t)(out_slots * nbatch + 2 * nbatch) *
                            sizeof(unsigned long long);
        if (ar->cap_bytes && (need>ar->cap_bytes || ar->bytes>ar->cap_bytes-need)) {
            ++ar->overflow;
            return nullptr;
        }
        NttArena::SmallEntry e;
        e.n = n;
        e.nbatch = nbatch;
        e.out_cap = out_slots;
        if (!NttArena::try_malloc(&e.dOut, out_slots * nbatch * sizeof(unsigned long long)) ||
            !NttArena::try_malloc(&e.dRes, 2 * nbatch * sizeof(unsigned long long))) {
            if (e.dOut) cudaFree(e.dOut);
            if (e.dRes) cudaFree(e.dRes);
            ++ar->overflow;
            if (ar->overflow <= 4)
                std::fprintf(stderr, "%s: arena could NOT allocate the small buffers for N=%llu "
                                     "nbatch=%llu (%.0f MB): the device is out of memory -- "
                                     "falling back to per-call cudaMalloc\n",
                             NTT_PROBE_NAME, n, nbatch, need / 1048576.0);
            return nullptr;
        }
        e.words = (size_t)(out_slots * nbatch + 2 * nbatch);
        ar->bytes += need;
        ar->smalls.push_back(e);
        ++ar->buf_builds;
        small = &ar->smalls.back();
    }
    ar->cur.n = n;
    ar->cur.out_slots = out_slots;
    ar->cur.nbatch = nbatch;
    ar->cur.dA = big->dA; ar->cur.dB = big->dB; ar->cur.dQ = big->dQ; ar->cur.dC = nullptr;
    ar->cur.dOut = small->dOut; ar->cur.dRes = small->dRes;
    ar->update_peaks();
    return &ar->cur;
}

void ntt_arena_pin_smem(NttArena *ar, int t, bool warp)
{
    if (!ar) return;
    const size_t need = (size_t)1 << ((t < 0) ? 0 : t);
    if (need > ar->attr_words) ar->attr_words = need;
    const int smem = (int)(ar->attr_words * sizeof(unsigned long long));
    ntt_tile_attributes(warp && t>=5,smem);
}

void ntt_arena_fuse(NttArena *ar, unsigned long long n, int k, unsigned long long omega,
                           unsigned long long omega_inv, FuseCtx &out)
{
    ntt_gl_reduce_configure();
    if (!ar) { fuse_init(out, n, k, omega, omega_inv); return; }
    const int requested_t=fuse_selected_tile(k,std::max(0,std::min(k,(int)fuse_env_ull("NTT_FUSE_T",12))));
    bool requested_coop=false;
    const int requested_m=fuse_outer_max(requested_t,k,requested_coop);
    for (NttArena::FuseEntry &e : ar->fuses) {
        if (e.n == n && e.k == k && e.omega == omega && e.fc.t==requested_t &&
            e.fc.m_max==requested_m && e.fc.coop_outer==requested_coop &&
            e.fc.compact_scratch==fuse_compact_scratch()) {
            out = e.fc;                                  /* pointers are owned by the arena */
            out.warp_tail=fuse_env_ull("NTT_FUSE_WARP_TAIL",0)!=0; /* plan tables are identical */
            out.arena_borrowed = true;
            ntt_arena_pin_smem(ar, e.fc.t,out.warp_tail);
            ++ar->fuse_hits;
            return;
        }
    }
    NttArena::FuseEntry e;
    e.n = n;
    e.k = k;
    e.omega = omega;
    fuse_describe(e.fc,n,k,omega,omega_inv);
    const size_t need=(fuse_planned_base_words(e.fc)+fuse_planned_table_words(e.fc))*8;
    if (ar->cap_bytes && (need>ar->cap_bytes || ar->bytes>ar->cap_bytes-need)) {
        ++ar->overflow;
        fuse_init(out, n, k, omega, omega_inv);      /* over budget: per-call tables */
        ntt_arena_pin_smem(ar,out.t,out.warp_tail);
        return;
    }
    fuse_init(e.fc, n, k, omega, omega_inv);
    ntt_arena_pin_smem(ar, e.fc.t,e.fc.warp_tail);                      /* undo fuse_init's own (smaller) set */
    ntt_fuse_cache_tables(e.fc, omega, omega_inv);
    ar->bytes += need;
    ar->fuses.push_back(e);
    ar->update_peaks();
    ++ar->fuse_builds;
    out = ar->fuses.back().fc;
    out.arena_borrowed = true;
}

void ntt_trim_fuse_headroom(NttArena *arena,unsigned long long keep_n,
    size_t required,const char *boundary,size_t &available)
{
    if(!arena || available>=required)return;
    const double begin=now_s();const size_t before=available;
    size_t released=0,contexts=0,total=0;
    while(available<required) {
        const size_t freed=arena->drop_cold_fuse(keep_n);
        if(!freed)break;
        released+=freed;++contexts;CK(cudaMemGetInfo(&available,&total));
    }
    stage2_log::print(stage2_log::debug,
        "stage2_cache_trim: boundary=%s keep_n=%llu required_bytes=%llu before_bytes=%llu after_bytes=%llu released_bytes=%llu contexts=%llu seconds=%.6f fits=%d\n",
        boundary,keep_n,(unsigned long long)required,(unsigned long long)before,(unsigned long long)available,
        (unsigned long long)released,(unsigned long long)contexts,now_s()-begin,(int)(available>=required));
}

bool ntt_host_break_on()
{
    static const bool on = (std::getenv("NTT_HOST_BREAK") != nullptr);
    return on;
}

bool extract_exact_coeffs(const unsigned long long *digits, unsigned long long n,
                                 int bpw, unsigned long long slot_bits,
                                 unsigned long long slot_stride, unsigned long long out_slots,
                                 std::vector<std::vector<unsigned long long>> *out)
{
    if (bpw <= 0 || bpw >= 64 || slot_bits == 0) return false;
    mpz_t C, t;
    mpz_inits(C, t, nullptr);
    mpz_import(C, (size_t)n, -1, 8, 0, (size_t)(64 - bpw), digits);
    const size_t cw = (size_t)((slot_bits + 63) / 64);
    out->assign((size_t)out_slots, std::vector<unsigned long long>(cw, 0));
    std::vector<unsigned long long> tmp(cw + 1, 0);
    for (unsigned long long k = 0; k < out_slots; ++k) {
        mpz_tdiv_q_2exp(t, C, (mp_bitcnt_t)(k * slot_stride));
        mpz_fdiv_r_2exp(t, t, (mp_bitcnt_t)slot_bits);
        std::fill(tmp.begin(), tmp.end(), 0ull);
        size_t cnt = 0;
        mpz_export(tmp.data(), &cnt, -1, 8, 0, 0, t);
        for (size_t j = 0; j < cw && j < cnt; ++j) (*out)[(size_t)k][j] = tmp[j];
    }
    mpz_clears(C, t, nullptr);
    return true;
}

bool ntt_pack_operand(const NttShape &sh, const unsigned long long *wordsA,
                             const unsigned long long *wordsB, unsigned long long *hA,
                             unsigned long long *hB, unsigned long long *maxdigit_out)
{
    const unsigned long long P = sh.P, N = sh.N, W = sh.W;
    const int S = sh.S, bpw = sh.bpw;
    const unsigned long long slot_stride = sh.slot_stride;
    std::fill(hA, hA + N, 0ull);
    std::fill(hB, hB + N, 0ull);
    for (unsigned long long i = 0; i < P; ++i) {
        for (int b = 0; b < S; b += bpw) {
            const int nb = (b + bpw <= S) ? bpw : (S - b);
            if (nb <= 0) continue;
            const unsigned long long bit = i * slot_stride + (unsigned long long)b;
            const unsigned long long v = bits_of(wordsA, P * W, i * W * 64ull + (unsigned long long)b, nb);
            const unsigned long long v2 = bits_of(wordsB, P * W, i * W * 64ull + (unsigned long long)b, nb);
            hA[(size_t)(bit / bpw)] |= v << (int)(bit % bpw);
            hB[(size_t)(bit / bpw)] |= v2 << (int)(bit % bpw);
        }
    }
    const unsigned long long lim = 1ull << bpw;
    unsigned long long over = 0, maxd = 0;
    for (unsigned long long j = 0; j < N; ++j) {
        const unsigned long long m = (hA[(size_t)j] > hB[(size_t)j]) ? hA[(size_t)j]
                                                                     : hB[(size_t)j];
        if (m >= lim) ++over;
        if (m > maxd) maxd = m;
    }
    if (maxdigit_out) *maxdigit_out = maxd;
    if (over) {
        std::fprintf(stderr, NTT_PROBE_NAME ": PACKING NOT CANONICAL: %llu of %llu digits "
                             ">= 2^%d (max digit %llu) -> the exactness criterion "
                             "L*(2^bpw-1)^2 < p does not bound the convolution and every "
                             "coefficient would wrap mod p; slot_stride=%llu bpw=%d\n",
                     (unsigned long long)over, (unsigned long long)N, bpw,
                     (unsigned long long)maxd, (unsigned long long)slot_stride, bpw);
        return false;
    }
    return true;
}

void ntt_shape_print(const NttShape &sh, const FuseCtx &fc, bool verbose,
                            unsigned long long maxcoeff, unsigned long long maxcoeff_k,
                            bool maxcoeff_ran, unsigned long long maxcoeff_limbs)
{
    if (!verbose) return;
    stage2_log::print(stage2_log::phases, NTT_PROBE_NAME ": mode=poly device=%d (%s) P=%llu S=%d slot_bits=%llu "
                "bpw=%d nwords=%llu (log2=%d)\n",
                sh.device, sh.dev_name, (unsigned long long)sh.P, sh.S,
                (unsigned long long)sh.slot_bits, sh.bpw, (unsigned long long)sh.N, sh.k);
    stage2_log::print(stage2_log::debug, "  exactness: L*(2^bpw-1)^2 = 2^%.3f < p = 2^%.3f -> %s ; L=P*slot_words="
                "%llu*%llu=%llu terms (N=%llu would be the old count) ; bound/p=%.6f "
                "(headroom %.4fx) ; the old N-form N*(2^bpw)^2 = 2^%.3f would be %s ; "
                "payload=%llu bits capacity=%llu bits (%.2fx) ; mem=%.0f MB\n",
                sh.bound_bits, sh.p_bits, sh.bound_ok ? "OK" : "VIOLATED",
                (unsigned long long)sh.P, (unsigned long long)sh.slot_words,
                (unsigned long long)sh.L_terms, (unsigned long long)sh.N, sh.bound_ratio,
                (sh.bound_ratio > 0) ? 1.0 / sh.bound_ratio : 0.0, sh.old_bound_bits,
                sh.old_bound_ok ? "OK" : "VIOLATED (that form is what the L bound replaces: "
                                          "it is the reason bpw and N were smaller/larger "
                                          "before)",
                (unsigned long long)sh.payload_bits, (unsigned long long)sh.bpw * sh.N,
                (double)(sh.bpw * sh.N) / (double)sh.payload_bits, sh.mem_mb);
    if (maxcoeff_ran) {
        const double dsq = digit_sq_double(sh.bpw);
        stage2_log::print(stage2_log::debug, "  maxcoeff: EXACT GMP maximum over all %llu convolution coefficients of "
                    "this shape (the full sum, not a sample) = %llu at k=%llu = %.4f x "
                    "(2^bpw-1)^2, vs the proven bound L=%llu (max is %.4f of the bound) ; "
                    "max/p=%.4f -> < p OK\n",
                    (unsigned long long)(2 * (unsigned long long)sh.L_terms - 1),
                    (unsigned long long)maxcoeff, (unsigned long long)maxcoeff_k,
                    (double)maxcoeff / dsq, (unsigned long long)sh.L_terms,
                    (double)maxcoeff / (dsq * (double)sh.L_terms),
                    sh.bound_ratio * ((double)maxcoeff / (dsq * (double)sh.L_terms)));
    } else {
        stage2_log::print(stage2_log::debug, "  maxcoeff: NOT recomputed for this shape (L=%llu terms > "
                    "NTT_MAXCOEFF_LIMBS=%llu, the GMP product would need %.0f MB of host "
                    "memory) -- the coefficient bound above is the COUNTING bound "
                    "c_k <= L*(2^bpw-1)^2 (proof above exact_ok_terms), which the smaller "
                    "shapes of this run verify exactly against GMP\n",
                    (unsigned long long)sh.L_terms, (unsigned long long)maxcoeff_limbs,
                    3.0 * 8.0 * (double)sh.L_terms / 1048576.0);
    }
    stage2_log::print(stage2_log::debug, "  fusion: PASSES=%d total (each pass = one full read AND one full write of the "
                "N-word array) = %d forward + %d forward + %d inverse (+%d carry) ; "
                "transform_stages=%d ; plan: tile t=%d stages, then outer",
                sh.passes_total, sh.passes_fwd, sh.passes_fwd, sh.passes_fwd, 1, sh.k, fc.t);
    for (int p = 0; p < fc.nms; ++p) stage2_log::print(stage2_log::debug, " radix-%d", 1 << fc.ms[p]);
    stage2_log::print(stage2_log::debug, " ; warp_tail=%d ; slot assembly reads %.2f of a pass\n",
                (int)(fc.warp_tail && fc.t>=5), (double)(sh.out_slots * sh.slot_words) / (double)sh.N);
    stage2_log::print(stage2_log::debug, "  packing: word-aligned slot stride=%llu bits = %llu digits of %d bits "
                "(coefficients every %llu bits), max operand digit < 2^%d asserted\n",
                (unsigned long long)sh.slot_stride, (unsigned long long)sh.slot_words, sh.bpw,
                (unsigned long long)sh.slot_stride, sh.bpw);
}

NttPassResult ntt_run_passes(const NttShape &sh, const FuseCtx &fc,
                                    unsigned long long *dA, unsigned long long *dB,
                                    unsigned long long *dQ, unsigned long long *dOut,
                                    unsigned long long *dRes, unsigned long long nbatch,
                                    int dump, std::vector<unsigned long long> *hfa,
                                    std::vector<unsigned long long> *hfb,
                                    std::vector<unsigned long long> *hPre,
                                    std::vector<unsigned long long> *hPost,
                                    bool need_hout, bool defer_res, NttArena *arena)
{
    stage2_trace::Range trace("NTT.submit");
    NttPassResult r;
    const unsigned long long N = sh.N, out_slots = sh.out_slots;
    const unsigned int threads = 256;
    const unsigned int blocks = (unsigned int)((N + threads - 1) / threads);
    unsigned int *carry_partial=arena ? arena->carry_workspace(N,nbatch) : nullptr;
    const unsigned long long inv_n = ((~0ull) / N) + 1;   /* 2^64 / N: exact exponent mod */

    /* ---- forward, pointwise product, inverse ----
       FORWARD = DIF (Gentleman-Sande), halves DESCENDING: natural in, bit-reversed spectrum
       out.  INVERSE = DIT (Cooley-Tukey), halves ASCENDING, same index decomposition, root
       omega^{-1}: consumes that bit-reversed spectrum and yields natural order.  The pointwise
       product sits between them and needs NO permutation and NO twiddle factor (exp_scale 0).
       This pair is verified by `nttcheck` (roundtrip bit-exact and forward == direct DFT at
       every length 2..4096) -- there is no bit-reversal kernel anywhere in the data path.
       The FUSED implementation (section 12 above) runs the very same stages in the very same
       order; `fusecheck` compares it element by element against this per-stage code. */
    /* ---- THE TIMERS MUST NOT SERIALISE THE PIPELINE (objective 4, section 39) ---------------
       These two phases used to be followed by `cudaDeviceSynchronize()` purely so that t_fwd and
       t_inv could be measured.  The GPU-Z sensor log says what that costs: over a B2=1e11 run the
       GPU was at >=80% load only 33% of the samples and BELOW 20% for 26% of them (mean 58.7%,
       mean board power 19.8 W of a ~60 W budget) -- i.e. the engine leaves the device idle about
       two fifths of the time, and one cause is exactly this: the host drained the pipeline twice
       per multiply for no reason but the clock.
       The dependency the sync provided is already provided by the STREAM (the inverse reads what
       the forward wrote, the carry reads what the inverse wrote), so the timers move to CUDA
       events: recorded in-stream, read once at the end of the call, never blocking.  The numbers
       stay honest (events measure GPU time, not host time) and the host can queue the next
       phases while the device is still executing. */
    cudaEvent_t ev_f0, ev_f1, ev_i1;
    CK(cudaEventCreate(&ev_f0));
    CK(cudaEventCreate(&ev_f1));
    CK(cudaEventCreate(&ev_i1));
    NttGraphCache::Entry *graph=nullptr;
    if(arena && arena->short_graph.enabled && !dump && arena->workspace_pool && arena->workspace.dA && arena->workspace.dB) {
        const size_t required=(size_t)N*(size_t)nbatch;
        const auto contains=[&](const unsigned long long *base,const unsigned long long *value) {
            const auto start=(uintptr_t)base,address=(uintptr_t)value;
            return address>=start && (address-start)%8==0 && (address-start)/8<=arena->workspace.capacity &&
                required<=arena->workspace.capacity-(address-start)/8;
        };
        if(contains(arena->workspace.dA,dA) && contains(arena->workspace.dB,dB))
            graph=arena->short_graph.lookup(arena->device,fc,dA,dB,sh.omega,sh.omega_inv,sh.n_scale,nbatch);
    }
    CK(cudaEventRecord(ev_f0));
    if(graph)CK(cudaGraphLaunch(graph->forward,0));
    else {ntt_forward_fused(dA, fc, sh.omega, nbatch);ntt_forward_fused(dB, fc, sh.omega, nbatch);}
    if (dump && nbatch == 1 && hfa && hfb) {
        hfa->assign(N, 0);
        hfb->assign(N, 0);
        CK(cudaMemcpy(hfa->data(), dA, N * sizeof(unsigned long long),
                      cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hfb->data(), dB, N * sizeof(unsigned long long),
                      cudaMemcpyDeviceToHost));
    }
    /* the pointwise product AND the 1/N scale are elementwise, so they ride along inside the
       inverse's tile pass (which touches the array once anyway) -- two whole passes saved */
    CK(cudaEventRecord(ev_f1));

    if(graph) {
        CK(cudaGraphLaunch(graph->inverse,0));arena->short_graph.launches+=2;
        arena->short_graph.saved_launches+=(unsigned long long)fc.passes_fwd*3-2;
    } else ntt_inverse_fused(dA, dB, fc, sh.omega_inv, sh.n_scale, nbatch);
    CK(cudaEventRecord(ev_i1));

    if (dump && nbatch == 1 && hPre) {
        hPre->assign(N, 0);
        CK(cudaMemcpy(hPre->data(), dA, N * sizeof(unsigned long long),
                      cudaMemcpyDeviceToHost));
    }

    /* ---- carry stage: reduce every digit back to bpw bits and ripple the quotient into the
       next digit.  ONE PASS: the cone of the same recurrence is unfolded into registers
       (carry_cone_kernel), followed inside that kernel by exact binary propagation.
       If the round count exceeds the register cone, height reduction uses the old kernels
       and a final cone kernel resolves the remaining binary carry. */
    const double t2 = now_s();
    const int carry_rounds = sh.carry_rounds;
    unsigned long long *dDig = dA;
    if (carry_rounds <= FUSE_MAX_CARRY_ROUNDS) {
        const dim3 gr(blocks, (unsigned int)nbatch);
        if(carry_partial) {
            switch(carry_rounds) {
            case 1: ecm_launch_carry_cone_check_kernel<1>(EcmCudaLaunch(gr,threads), dA,dQ,N,sh.bpw,N,carry_partial);break;
            case 2: ecm_launch_carry_cone_check_kernel<2>(EcmCudaLaunch(gr,threads), dA,dQ,N,sh.bpw,N,carry_partial);break;
            case 3: ecm_launch_carry_cone_check_kernel<3>(EcmCudaLaunch(gr,threads), dA,dQ,N,sh.bpw,N,carry_partial);break;
            case 4: ecm_launch_carry_cone_check_kernel<4>(EcmCudaLaunch(gr,threads), dA,dQ,N,sh.bpw,N,carry_partial);break;
            case 5: ecm_launch_carry_cone_check_kernel<5>(EcmCudaLaunch(gr,threads), dA,dQ,N,sh.bpw,N,carry_partial);break;
            case 6: ecm_launch_carry_cone_check_kernel<6>(EcmCudaLaunch(gr,threads), dA,dQ,N,sh.bpw,N,carry_partial);break;
            case 7: ecm_launch_carry_cone_check_kernel<7>(EcmCudaLaunch(gr,threads), dA,dQ,N,sh.bpw,N,carry_partial);break;
            case 8: ecm_launch_carry_cone_check_kernel<8>(EcmCudaLaunch(gr,threads), dA,dQ,N,sh.bpw,N,carry_partial);break;
            case 9: ecm_launch_carry_cone_check_kernel<9>(EcmCudaLaunch(gr,threads), dA,dQ,N,sh.bpw,N,carry_partial);break;
            default: ecm_launch_carry_cone_check_kernel<10>(EcmCudaLaunch(gr,threads), dA,dQ,N,sh.bpw,N,carry_partial);break;
            }
        } else {
        switch (carry_rounds) {
            case 1: ecm_launch_carry_cone_kernel<1>(EcmCudaLaunch(gr, threads), dA, dQ, N, sh.bpw, N); break;
            case 2: ecm_launch_carry_cone_kernel<2>(EcmCudaLaunch(gr, threads), dA, dQ, N, sh.bpw, N); break;
            case 3: ecm_launch_carry_cone_kernel<3>(EcmCudaLaunch(gr, threads), dA, dQ, N, sh.bpw, N); break;
            case 4: ecm_launch_carry_cone_kernel<4>(EcmCudaLaunch(gr, threads), dA, dQ, N, sh.bpw, N); break;
            case 5: ecm_launch_carry_cone_kernel<5>(EcmCudaLaunch(gr, threads), dA, dQ, N, sh.bpw, N); break;
            case 6: ecm_launch_carry_cone_kernel<6>(EcmCudaLaunch(gr, threads), dA, dQ, N, sh.bpw, N); break;
            case 7: ecm_launch_carry_cone_kernel<7>(EcmCudaLaunch(gr, threads), dA, dQ, N, sh.bpw, N); break;
            case 8: ecm_launch_carry_cone_kernel<8>(EcmCudaLaunch(gr, threads), dA, dQ, N, sh.bpw, N); break;
            case 9: ecm_launch_carry_cone_kernel<9>(EcmCudaLaunch(gr, threads), dA, dQ, N, sh.bpw, N); break;
            default: ecm_launch_carry_cone_kernel<10>(EcmCudaLaunch(gr, threads), dA, dQ, N, sh.bpw, N); break;
        }
        }
        CK(cudaGetLastError());
        dDig = dQ;
    } else {
        for (unsigned long long s = 0; s < nbatch; ++s) {
            for (int rr = 0; rr < carry_rounds; ++rr) {
                ecm_launch_carry_extract_kernel(EcmCudaLaunch(blocks, threads), dA + s * N, dQ + s * N, N, sh.bpw);
                CK(cudaGetLastError());
                ecm_launch_carry_add_kernel(EcmCudaLaunch(blocks, threads), dA + s * N, dQ + s * N, N);
                CK(cudaGetLastError());
            }
        }
        const dim3 gr(blocks, (unsigned int)nbatch);
        if(carry_partial)ecm_launch_carry_cone_check_kernel<1>(EcmCudaLaunch(gr,threads), dA,dQ,N,sh.bpw,N,carry_partial);
        else ecm_launch_carry_cone_kernel<1>(EcmCudaLaunch(gr, threads), dA, dQ, N, sh.bpw, N);
        CK(cudaGetLastError());
        dDig = dQ;
    }
    r.digits = dDig;
    if (dump && nbatch == 1 && hPost) {
        hPost->assign(N, 0);
        CK(cudaMemcpy(hPost->data(), dDig, N * sizeof(unsigned long long),
                      cudaMemcpyDeviceToHost));
    }
    r.t_slot = now_s() - t2;
    /* THE SLOT PROJECTION IS ONLY FOR A CALLER WITHOUT A HOOK (objective 4, section 35): the
       caller that installed one reduces the coefficients on the device out of `digits` itself and
       never looks at `dOut`/`hOut`, so the assembly kernel and the D2H of out_slots*nbatch words
       are pure waste for it.  Measured and switchable (NTT_S4_KEEPHOUT=1 restores the old
       behaviour) rather than assumed. */
    const double t4 = now_s();
    if (need_hout) {
        const unsigned int blSlots = (unsigned int)((out_slots + threads - 1) / threads);
        {
            const dim3 gr(blSlots, (unsigned int)nbatch);
            ecm_launch_slot_assemble_kernel(EcmCudaLaunch(gr, threads), dDig, out_slots, sh.slot_bits, sh.slot_stride,
                                                  sh.bpw, dOut, N);
        }
        CK(cudaGetLastError());
        r.hOut.assign((size_t)(out_slots * nbatch), 0);
        CK(cudaMemcpy(r.hOut.data(), dOut, r.hOut.size() * sizeof(unsigned long long),
                      cudaMemcpyDeviceToHost));
    } else {
        r.hOut.clear();
    }
    r.t_hout = now_s() - t4;

    /* Carry convergence assert (UNTIMED until now, and always run -- a carry that stops early is
       silent: the digits are simply wrong above some index).  ONE PASS over the whole digit array
       plus a D2H per call, which is why it is now measured. */
    const double t3 = now_s();
    /* the kernel's OWN GPU time, from events recorded in-stream around it.  With defer_res the
       marks are handed back (r.dev_ev[3..4]) and read by the caller after its own drain. */
    cudaEvent_t ev_c0, ev_c1;
    CK(cudaEventCreate(&ev_c0));
    CK(cudaEventCreate(&ev_c1));
    if (!defer_res) {
        r.hRes.assign((size_t)(2 * nbatch), 0);
        CK(cudaMemset(dRes, 0, 2 * nbatch * sizeof(unsigned long long)));
    }
    const double t3a = now_s();
    r.t_check_reset = t3a - t3;
    CK(cudaEventRecord(ev_c0));
    {
        const dim3 gr(blocks, (unsigned int)nbatch);
        if(carry_partial)ecm_launch_carry_partial_finish_kernel(EcmCudaLaunch((unsigned int)nbatch,threads), carry_partial,blocks,dRes);
        else ecm_launch_carry_residual_kernel(EcmCudaLaunch(gr, threads), dDig, N, sh.bpw, dRes, N);
    }
    CK(cudaGetLastError());
    CK(cudaEventRecord(ev_c1));
    if (defer_res) {
        /* NOTHING HERE DRAINS THE PIPELINE (section 41).  `dRes` holds 2 words per slice and the
           kernel indexes them by slice, so the counters of every call in a chunk simply
           ACCUMULATE in place: one memset before the caller's chunk loop and one readback after
           it reproduce exactly the per-call values, at one drain per chunk instead of one per
           call.  Measured at the production shape: the whole carry check was 60.6 s (24% of
           ntt_seconds) of which the kernel itself is 0.2 us per call -- the rest was the 16-byte
           blocking readback draining the pipeline. */
        r.res_deferred = true;
        r.dev_ev[3] = ev_c0;
        r.dev_ev[4] = ev_c1;
        r.t_check = now_s() - t3;
        r.dev_ev[0] = ev_f0;
        r.dev_ev[1] = ev_f1;
        r.dev_ev[2] = ev_i1;
        return r;
    }
    const double tc0 = now_s();
    CK(cudaMemcpy(r.hRes.data(), dRes, r.hRes.size() * sizeof(unsigned long long),
                  cudaMemcpyDeviceToHost));
    r.t_check_d2h = now_s() - tc0;
    r.t_check_kernel = 0.0;
    {
        float ms = 0.0f;
        if (cudaEventElapsedTime(&ms, ev_c0, ev_c1) == cudaSuccess)
            r.t_check_kernel = (double)ms * 1e-3;       /* the GPU's own time */
    }
    cudaEventDestroy(ev_c0);
    cudaEventDestroy(ev_c1);
    r.t_check = now_s() - t3;
    /* the in-stream event timers, read once the stream has drained (the copies above guarantee
       that) -- see the note at ev_f0 */
    {
        float ms = 0.0f;
        CK(cudaEventElapsedTime(&ms, ev_f0, ev_f1));
        r.t_fwd = (double)ms * 1e-3;
        CK(cudaEventElapsedTime(&ms, ev_f1, ev_i1));
        r.t_inv = (double)ms * 1e-3;
        cudaEventDestroy(ev_f0);
        cudaEventDestroy(ev_f1);
        cudaEventDestroy(ev_i1);
    }
    return r;
}

void ntt_pass_read_marks(NttPassResult &r, double *t_fwd, double *t_inv, double *t_chk)
{
    float ms = 0.0f;
    if (r.dev_ev[0] && r.dev_ev[1] &&
        cudaEventElapsedTime(&ms, r.dev_ev[0], r.dev_ev[1]) == cudaSuccess)
        *t_fwd += (double)ms * 1e-3;
    if (r.dev_ev[1] && r.dev_ev[2] &&
        cudaEventElapsedTime(&ms, r.dev_ev[1], r.dev_ev[2]) == cudaSuccess)
        *t_inv += (double)ms * 1e-3;
    if (r.dev_ev[3] && r.dev_ev[4] &&
        cudaEventElapsedTime(&ms, r.dev_ev[3], r.dev_ev[4]) == cudaSuccess)
        *t_chk += (double)ms * 1e-3;
    for (int i = 0; i < 5; ++i)
        if (r.dev_ev[i]) { cudaEventDestroy(r.dev_ev[i]); r.dev_ev[i] = nullptr; }
}

bool ntt_hook_window(const NttReduceHook *hook, unsigned long long slots,
                             unsigned long long &count)
{
    count=hook && hook->count!=~0ull ? hook->count : slots;
    const auto first=hook ? hook->first : 0ull;
    return first<=slots && count<=slots-first;
}

int ntt_poly_mul_batch_host(unsigned long long P, int S, int device, unsigned long long nbatch,
                            const unsigned long long *wordsA, const unsigned long long *wordsB,
                            std::vector<unsigned long long> *out_slots_u64,
                            NttMulStats *st, NttArena *arena,
                            const NttReduceHook *hook,
                            unsigned long long **digits_out)
{
    if (nbatch == 0) {
        std::fprintf(stderr, NTT_PROBE_NAME ": batch of zero slices\n");
        return 2;
    }
    NttShape sh;
    FuseCtx fc;
    FuseCallGuard fuse_guard(fc);
    {
        const int rc = ntt_shape_plan(P, S, device, arena, fc, sh);
        if (rc) return rc;
    }
    sh.L_terms = sh.P * sh.slot_words;
    {
        const int rc = ntt_shape_exactness(sh);
        if (rc) return rc;
    }
    const unsigned long long N = sh.N, out_slots = sh.out_slots, W = sh.W;
    unsigned long long hook_count=out_slots;
    if (!ntt_hook_window(hook,out_slots,hook_count)) return 2;
    const size_t slice_in = (size_t)P * W;               /* u64 words per operand slice */
    std::vector<uint64_t> hA((size_t)(N * nbatch), 0), hB((size_t)(N * nbatch), 0);
    unsigned long long maxdigit = 0;
    /* with a reduction hook the caller never reads the slot projections, so the assembly kernel
       and its D2H are skipped (NTT_S4_KEEPHOUT=1 restores the old behaviour for the A/B) */
    const bool want_hout = (hook == nullptr || hook->run == nullptr) ? true : [] {
        const char *e = std::getenv("NTT_S4_KEEPHOUT");
        return e && *e && std::atoi(e) != 0;
    }();
    const double thp0 = now_s();
    for (unsigned long long s = 0; s < nbatch; ++s) {
        if (!ntt_pack_operand(sh, wordsA + s * slice_in, wordsB + s * slice_in,
                              hA.data() + s * N, hB.data() + s * N, &maxdigit))
            return 5;
    }
    const double t_hpack = now_s() - thp0;
    unsigned long long maxcoeff = 0, maxcoeff_k = 0, maxcoeff_limbs = 0;
    bool maxcoeff_ran = false;
    const double tsc0 = now_s();
    {
        const int rc = ntt_shape_maxcoeff(sh, hA, hB, &maxcoeff, &maxcoeff_k, &maxcoeff_ran,
                                          &maxcoeff_limbs);
        if (rc) return rc;
    }
    const double t_scan = now_s() - tsc0;
    /* Borrowed digits use the original keyed-cache lifetime, never shared scratch. */
    NttArena::BufEntry *ab = ntt_arena_bufs(arena, N, out_slots, nbatch, digits_out == nullptr);
    if (!ab && digits_out) {
        *digits_out=nullptr;

        return 3;                         /* an owned fallback would return a freed pointer */
    }
    unsigned long long *dA = nullptr, *dB = nullptr, *dQ = nullptr, *dOut = nullptr,
                       *dRes = nullptr;
    bool own = false;
    if (ab) {
        dA = ab->dA; dB = ab->dB; dQ = ab->dQ; dOut = ab->dOut; dRes = ab->dRes;
    } else {
        own = true;
        CK(cudaMalloc(&dA, (size_t)(N * nbatch) * sizeof(unsigned long long)));
        CK(cudaMalloc(&dB, (size_t)(N * nbatch) * sizeof(unsigned long long)));
        CK(cudaMalloc(&dQ, (size_t)(N * nbatch) * sizeof(unsigned long long)));
        CK(cudaMalloc(&dOut, (size_t)(out_slots * nbatch) * sizeof(unsigned long long)));
        CK(cudaMalloc(&dRes, 2 * (size_t)nbatch * sizeof(unsigned long long)));
    }
    const double h2d0 = now_s();
    CK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(unsigned long long),
                  cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(unsigned long long),
                  cudaMemcpyHostToDevice));
    const double t_h2d_batch = now_s() - h2d0;
    /* nbatch is a gridDim.y, and 65535 is the whole range of gridDim.y: a caller that asks for
       more is SPLIT here (chunk by chunk, the hook included) rather than silently truncated. */
    const unsigned long long max_y = 65535;
    const unsigned long long nchunk = (nbatch + max_y - 1) / max_y;
    NttPassResult r;
    r.hOut.assign((size_t)(out_slots * nbatch), 0);
    r.hRes.assign((size_t)(2 * nbatch), 0);
    for (unsigned long long ci = 0; ci < nchunk; ++ci) {
        const unsigned long long s0 = ci * max_y;
        const unsigned long long m = ((nbatch - s0) < max_y) ? (nbatch - s0) : max_y;
        NttPassResult rr = ntt_run_passes(sh, fc, dA + s0 * N, dB + s0 * N, dQ + s0 * N,
                                          dOut + s0 * out_slots, dRes + 2 * s0, m, 0,
                                          nullptr, nullptr, nullptr, nullptr, want_hout, false, ab ? arena : nullptr);
        if (want_hout)
            std::copy(rr.hOut.begin(), rr.hOut.end(), r.hOut.begin() + (long)(s0 * out_slots));
        std::copy(rr.hRes.begin(), rr.hRes.end(), r.hRes.begin() + (long)(2 * s0));
        r.t_fwd += rr.t_fwd; r.t_inv += rr.t_inv; r.t_slot += rr.t_slot;
        r.t_check += rr.t_check;
        r.t_check_reset += rr.t_check_reset;
        r.t_check_kernel += rr.t_check_kernel;
        r.t_check_d2h += rr.t_check_d2h;
        if (nchunk == 1) r.digits = rr.digits;
        if (hook && hook->run && hook->w && (hook->out || hook_count==0)) {
            hook->run(hook->ctx, rr.digits, N, sh.bpw, sh.slot_words, sh.slot_bits, out_slots,
                      m, hook->out ? hook->out + (size_t)(s0 * hook_count) * hook->w : nullptr,
                      hook->w, hook->first, hook_count);
            CK(cudaGetLastError());
            CK(cudaDeviceSynchronize());
        }
    }
    if (st) {
        st->P = P; st->S = S; st->N = N; st->k = sh.k; st->bpw = sh.bpw;
        st->slot_bits = sh.slot_bits; st->slot_stride = sh.slot_stride;
        st->slot_words = sh.slot_words; st->out_slots = out_slots; st->L_terms = sh.L_terms;
        st->cw = (unsigned long long)((sh.slot_bits + 63) / 64);
        st->passes_fwd = sh.passes_fwd; st->passes_total = sh.passes_total;
        st->carry_rounds = sh.carry_rounds;
        st->carry_residual = 0; st->carry_max_bits = 0;
        for (unsigned long long s = 0; s < nbatch; ++s) {
            st->carry_residual += r.hRes[(size_t)(2 * s)];
            if (r.hRes[(size_t)(2 * s + 1)] > st->carry_max_bits)
                st->carry_max_bits = r.hRes[(size_t)(2 * s + 1)];
        }
        st->mem_mb = sh.mem_mb * (double)nbatch;
        st->t_fwd = r.t_fwd; st->t_inv = r.t_inv; st->t_slot = r.t_slot;
        st->t_check += r.t_check;
        st->t_check_reset += r.t_check_reset;
        st->t_check_kernel += r.t_check_kernel;
        st->t_check_d2h += r.t_check_d2h;
        st->t_hpack += t_hpack;
        st->t_scan += t_scan;
        st->t_h2d_batch += t_h2d_batch;
        st->fuse_t = fc.t; st->fuse_nms = fc.nms;
        for (int q = 0; q < fc.nms && q < 8; ++q) st->fuse_ms[q] = fc.ms[q];
        st->exact_valid = true;
    }
    unsigned long long bad_res = 0;
    for (unsigned long long s = 0; s < nbatch; ++s)
        if (r.hRes[(size_t)(2 * s)] != 0) ++bad_res;
    if (bad_res) {
        std::fprintf(stderr, NTT_PROBE_NAME ": CARRY DID NOT CONVERGE in %llu of %llu slices "
                             "after %d rounds\n", bad_res, nbatch, sh.carry_rounds);
        if (own) { cudaFree(dA); cudaFree(dB); cudaFree(dQ); cudaFree(dOut); cudaFree(dRes); }
        return 4;
    }
    if (out_slots_u64) *out_slots_u64 = r.hOut;
    if (digits_out) {
        if (nchunk != 1) {
            std::fprintf(stderr, NTT_PROBE_NAME ": the digit buffer of a split batch is not "
                                 "one contiguous array\n");
            if (own) { cudaFree(dA); cudaFree(dB); cudaFree(dQ); cudaFree(dOut); cudaFree(dRes); }
            return 3;
        }
        *digits_out = r.digits;
    }
    int rc = 0;

    if (own) { cudaFree(dA); cudaFree(dB); cudaFree(dQ); cudaFree(dOut); cudaFree(dRes); }
    return rc;
}

int ntt_poly_mul_batch_dev(unsigned long long P, int S, int device, unsigned long long nbatch,
                           const unsigned long long *dAin, const unsigned long long *dBin,
                           NttMulStats *st, NttArena *arena, const NttReduceHook *hook,
                           unsigned long long **digits_out, int force_bpw,
                           bool defer_carry, const NttInputHook *input)
{
    stage2_trace::Range trace("NTT.request");
    if (nbatch == 0) {
        std::fprintf(stderr, NTT_PROBE_NAME ": batch of zero slices\n");
        return 2;
    }
    NttShape sh;
    FuseCtx fc;
    FuseCallGuard fuse_guard(fc);
    const double tplan0 = now_s();
    {
        const int rc = ntt_shape_plan(P, S, device, arena, fc, sh, force_bpw);
        if (rc) return rc;
    }
    const double t_plan = now_s() - tplan0;
    sh.L_terms = sh.P * sh.slot_words;
    {
        const int rc = ntt_shape_exactness(sh);
        if (rc) return rc;
    }
    const unsigned long long N = sh.N, out_slots = sh.out_slots, W = sh.W;
    unsigned long long hook_count=out_slots;
    if (!ntt_hook_window(hook,out_slots,hook_count)) return 2;
    std::vector<unsigned long long> alias_a, alias_b;
    auto preserve=[&](const unsigned long long *p, std::vector<unsigned long long> &saved) {
        size_t remaining=0;
        if (!arena || !arena->input_span(p,remaining)) return true;
        if ((size_t)(N*nbatch)>remaining) return false;
        saved.resize((size_t)(N*nbatch));
        CK(cudaMemcpy(saved.data(),p,saved.size()*8,cudaMemcpyDeviceToHost));
        ++arena->alias_snapshots; arena->alias_snapshot_bytes+=saved.size()*8;
        return true;
    };
    if (!input && (!preserve(dAin,alias_a) || !preserve(dBin,alias_b))) {
        std::fprintf(stderr, NTT_PROBE_NAME ": device input exceeds its arena allocation\n");

        return 3;
    }
    NttArena::BufEntry *ab = ntt_arena_bufs(arena, N, out_slots, nbatch, digits_out == nullptr);
    if (!ab && digits_out) {
        *digits_out=nullptr;

        return 3;
    }
    unsigned long long *dA = nullptr, *dB = nullptr, *dQ = nullptr, *dOut = nullptr,
                       *dRes = nullptr;
    bool own = false;
    if (ab) {
        dA = ab->dA; dB = ab->dB; dQ = ab->dQ; dOut = ab->dOut; dRes = ab->dRes;
    } else {
        own = true;
        CK(cudaMalloc(&dA, (size_t)(N * nbatch) * sizeof(unsigned long long)));
        CK(cudaMalloc(&dB, (size_t)(N * nbatch) * sizeof(unsigned long long)));
        CK(cudaMalloc(&dQ, (size_t)(N * nbatch) * sizeof(unsigned long long)));
        CK(cudaMalloc(&dOut, (size_t)(out_slots * nbatch) * sizeof(unsigned long long)));
        CK(cudaMalloc(&dRes, 2 * (size_t)nbatch * sizeof(unsigned long long)));
    }
    const double tcopy0 = now_s();
    if (input) {
        const int rc = input->run ? input->run(input->ctx, sh, nbatch, dA, dB) : 2;
        if (rc) {

            if (own) { cudaFree(dA); cudaFree(dB); cudaFree(dQ); cudaFree(dOut); cudaFree(dRes); }
            return rc;
        }
        CK(cudaGetLastError());
    } else {
        CK(cudaMemcpy(dA, alias_a.empty() ? dAin : alias_a.data(), (size_t)N * nbatch * sizeof(unsigned long long),
                      alias_a.empty() ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dB, alias_b.empty() ? dBin : alias_b.data(), (size_t)N * nbatch * sizeof(unsigned long long),
                      alias_b.empty() ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice));
    }
    const double t_copy = input ? 0.0 : now_s() - tcopy0;
    const unsigned long long max_y = 65535;
    const unsigned long long nchunk = (nbatch + max_y - 1) / max_y;
    /* with a reduction hook the caller never reads the slot projections, so the assembly kernel
       and its D2H are skipped here too (NTT_S4_KEEPHOUT=1 restores the old behaviour for the A/B) */
    const bool want_hout = (hook == nullptr || hook->run == nullptr) ? true : [] {
        const char *e = std::getenv("NTT_S4_KEEPHOUT");
        return e && *e && std::atoi(e) != 0;
    }();
    NttPassResult r;
    if (want_hout) r.hOut.assign((size_t)(out_slots * nbatch), 0);
    /* ONE reset and ONE readback for the whole chunk (section 41): the counters live at
       dRes[2*slice] and the kernel only ever touches its own slice's two words, so accumulating
       them across the chunk and reading once is exactly equivalent to reading after each
       call -- see the note where the call defers. */
    /* ---- DEFERRED CARRY CHECK (section 29) -------------------------------------------------
       A chunked caller pays this blocking readback ONCE PER CHUNK, and at the production shape
       that is 22471 chunks x 1.64 ms = 36.75 s of the 253.53 s run (measured: `carrysplt d2h`),
       even though the copy itself is 2*nbatch words -- the cost is the pipeline DRAIN, not the
       bytes.  With defer_carry the caller asks this call to leave the counters in place (no
       memset, no readback) and to read them once after the last deferred chunk, through
       ntt_batch_carry_finish.  The memset must be skipped for the same reason: it would wipe the
       interior chunks' counters before they are read.  Only legal with an arena, because the
       dRes buffer has to outlive this call. */
    const bool defer = defer_carry && (arena != nullptr) && (ab != nullptr) && !own;
    if (st) st->carry_deferred = defer;
    if (!defer)
        CK(cudaMemset(dRes, 0, 2 * (size_t)nbatch * sizeof(unsigned long long)));
    std::vector<NttPassResult> sub((size_t)nchunk);
    for (unsigned long long ci = 0; ci < nchunk; ++ci) {
        const unsigned long long s0 = ci * max_y;
        const unsigned long long m = ((nbatch - s0) < max_y) ? (nbatch - s0) : max_y;
        NttPassResult rr = ntt_run_passes(sh, fc, dA + s0 * N, dB + s0 * N, dQ + s0 * N,
                                          dOut + s0 * out_slots, dRes + 2 * s0, m, 0,
                                          nullptr, nullptr, nullptr, nullptr, want_hout, true, ab ? arena : nullptr);
        sub[(size_t)ci] = rr;
        if (!rr.hOut.empty())
            std::copy(rr.hOut.begin(), rr.hOut.end(), r.hOut.begin() + (long)(s0 * out_slots));
        r.t_slot += rr.t_slot;
        r.t_check += rr.t_check;
        r.t_check_reset += rr.t_check_reset;
        if (nchunk == 1) r.digits = rr.digits;
        if (hook && hook->run && hook->w && (hook->out || hook_count==0)) {
            hook->run(hook->ctx, rr.digits, N, sh.bpw, sh.slot_words, sh.slot_bits, out_slots,
                      m, hook->out ? hook->out + (size_t)(s0 * hook_count) * hook->w : nullptr,
                      hook->w, hook->first, hook_count);
            CK(cudaGetLastError());
        }
    }
    /* THE ONE DRAIN OF THE WHOLE CHUNK (section 41): the carry counters and the deferred
       in-stream marks are both read here, after a single blocking copy, instead of once per
       call.  This is also where an asynchronous kernel failure surfaces. */
    {
        const double tr0 = now_s();
        if (!defer) {
            r.hRes.assign((size_t)(2 * nbatch), 0);
            CK(cudaMemcpy(r.hRes.data(), dRes, r.hRes.size() * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
        }
        for (unsigned long long ci = 0; ci < nchunk; ++ci)
            ntt_pass_read_marks(sub[(size_t)ci], &r.t_fwd, &r.t_inv, &r.t_check_kernel);
        if (!defer) r.t_check_d2h += now_s() - tr0;
    }
    if (st) {
        st->P = P; st->S = S; st->N = N; st->k = sh.k; st->bpw = sh.bpw;
        st->slot_bits = sh.slot_bits; st->slot_stride = sh.slot_stride;
        st->slot_words = sh.slot_words; st->out_slots = out_slots; st->L_terms = sh.L_terms;
        st->cw = (unsigned long long)((sh.slot_bits + 63) / 64);
        st->passes_fwd = sh.passes_fwd; st->passes_total = sh.passes_total;
        st->carry_rounds = sh.carry_rounds;
        st->carry_residual = 0; st->carry_max_bits = 0;
        /* with a deferred carry check hRes is deliberately EMPTY here (the caller reads the whole
           accumulation once, through ntt_batch_carry_finish), so the per-slice scan must not run:
           it would index an empty vector */
        if (!defer)
            for (unsigned long long s = 0; s < nbatch; ++s) {
                st->carry_residual += r.hRes[(size_t)(2 * s)];
                if (r.hRes[(size_t)(2 * s + 1)] > st->carry_max_bits)
                    st->carry_max_bits = r.hRes[(size_t)(2 * s + 1)];
            }
        st->mem_mb = sh.mem_mb * (double)nbatch;
        st->t_fwd = r.t_fwd; st->t_inv = r.t_inv; st->t_slot = r.t_slot;
        st->t_check += r.t_check;
        st->t_check_reset += r.t_check_reset;
        st->t_check_kernel += r.t_check_kernel;
        st->t_check_d2h += r.t_check_d2h;
        st->t_plan += t_plan;
        st->t_opcopy += t_copy;
        st->t_hout += r.t_hout;
        /* the DEVICE entry point packs on the device: no host packing, no host scan and no
           operand upload to attribute, so those three fields stay 0 on this path */
        st->fuse_t = fc.t; st->fuse_nms = fc.nms;
        for (int q = 0; q < fc.nms && q < 8; ++q) st->fuse_ms[q] = fc.ms[q];
        st->exact_valid = true;
    }
    unsigned long long bad_res = 0;
    /* a deferred check has no hRes to scan: the caller's ntt_batch_carry_finish reads the whole
       accumulation and rejects it there, so scanning an empty vector here is both wrong (OOB)
       and redundant */
    if (!defer)
        for (unsigned long long s = 0; s < nbatch; ++s)
            if (r.hRes[(size_t)(2 * s)] != 0) ++bad_res;
    if (bad_res) {
        std::fprintf(stderr, NTT_PROBE_NAME ": CARRY DID NOT CONVERGE in %llu of %llu slices "
                             "after %d rounds\n", bad_res, nbatch, sh.carry_rounds);
        if (own) { cudaFree(dA); cudaFree(dB); cudaFree(dQ); cudaFree(dOut); cudaFree(dRes); }
        return 4;
    }
    if (digits_out) *digits_out = (nchunk == 1) ? r.digits : nullptr;

    if (own) { cudaFree(dA); cudaFree(dB); cudaFree(dQ); cudaFree(dOut); cudaFree(dRes); }
    return 0;
}

int ntt_batch_carry_finish(NttArena *arena, unsigned long long N, unsigned long long nbatch,
                           NttMulStats *st)
{
    if (!arena) {
        std::fprintf(stderr, NTT_PROBE_NAME ": ntt_batch_carry_finish without an arena\n");
        return 2;
    }
    /* the composite the last multiply left behind -- NOT a fresh lookup, because a fresh entry
       would be uninitialised memory and would pass a check it never earned */
    if (!arena->cur.dRes || arena->cur.n != N || arena->cur.nbatch != nbatch) {
        std::fprintf(stderr, NTT_PROBE_NAME ": ntt_batch_carry_finish: the arena's current buffers "
                             "are not the deferred shape (want N=%llu nbatch=%llu, have N=%llu "
                             "nbatch=%llu)\n", N, nbatch, arena->cur.n, arena->cur.nbatch);
        return 3;
    }
    const double t0 = now_s();
    std::vector<unsigned long long> hRes((size_t)(2 * nbatch), 0);
    if (cudaMemcpy(hRes.data(), arena->cur.dRes, hRes.size() * sizeof(unsigned long long),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, NTT_PROBE_NAME ": ntt_batch_carry_finish: the deferred readback "
                             "failed\n");
        (void)cudaGetLastError();
        return 5;
    }
    const double td = now_s() - t0;
    unsigned long long bad = 0;
    if (st) {
        st->t_check_d2h += td;                 /* charged to the one call that really paid it */
        st->carry_residual = 0;
        st->carry_max_bits = 0;
        for (unsigned long long s = 0; s < nbatch; ++s) {
            st->carry_residual += hRes[(size_t)(2 * s)];
            if (hRes[(size_t)(2 * s + 1)] > st->carry_max_bits)
                st->carry_max_bits = hRes[(size_t)(2 * s + 1)];
        }
    }
    for (unsigned long long s = 0; s < nbatch; ++s)
        if (hRes[(size_t)(2 * s)] != 0) ++bad;
    if (bad) {
        std::fprintf(stderr, NTT_PROBE_NAME ": CARRY DID NOT CONVERGE in %llu of %llu slices "
                             "(deferred check)\n", bad, nbatch);
        return 4;
    }
    return 0;
}

int ntt_poly_mul_host(unsigned long long P, int S, int device, bool verbose, int dump,
                      const unsigned long long *wordsA, const unsigned long long *wordsB,
                      std::vector<unsigned long long> *out_slots_u64,
                      std::vector<std::vector<unsigned long long>> *out_exact,
                      NttMulStats *st, NttArena *arena)
{
    const double thb_setup = now_s();
    double hb_t_setup = 0, hb_t_pack = 0, hb_t_maxc = 0, hb_t_h2d = 0, hb_t_d2h = 0,
           hb_t_ext = 0, hb_t_xchk = 0;
    /* THE SHAPE PLAN (shared with the batched entry point, so the two cannot drift) */
    NttShape sh;
    FuseCtx fc;
    FuseCallGuard fuse_guard(fc);
    {
        const int rc = ntt_shape_plan(P, S, device, arena, fc, sh);
        if (rc) return rc;
    }
    const unsigned long long N = sh.N, out_slots = sh.out_slots, W = sh.W;
    const int bpw = sh.bpw;
    const unsigned long long L_terms = sh.L_terms, slot_bits = sh.slot_bits;
    const unsigned long long slot_stride = sh.slot_stride, slot_words = sh.slot_words;
    const int k = sh.k;
    const unsigned long long omega = sh.omega;
    NTT_HBS(t_setup, thb_setup);

    /* ---- THE PACKING (shared): coefficient i at bit i*slot_stride, bpw bits per digit ----
       `slot_stride` MUST be a multiple of bpw.  Packing compactly at exactly i*slot_bits
       (slot_bits = 2S + log2 P is not a multiple of bpw) makes one DIGIT of the packed
       operand hold the tail of one bpw-bit chunk of the coefficient plus the head of the
       next chunk (the write `h[bit/bpw] |= v << (bit%bpw)` spills up to bpw-1 bits past the
       digit).  A digit then reaches 2^(2*bpw-1) instead of 2^bpw, the convolution
       coefficient bound becomes L*(2^(2bpw-1))^2 -- e.g. 2^66 for P=4/S=64 and 2^67 for
       P=1024/S=5153, both far ABOVE p -- so the exactness test L*(2^bpw-1)^2 < p keeps passing
       while EVERY coefficient of the product wraps mod p.  Symptom: every slot wrong
       (bad_slots = 2P-1, slot 0 included) with the geometry and the timings unchanged. */
    const double thb_pack = now_s();
    std::vector<uint64_t> hA(N, 0), hB(N, 0);
    if (!ntt_pack_operand(sh, wordsA, wordsB, hA.data(), hB.data(), nullptr)) return 5;
    NTT_HBS(t_pack, thb_pack);

    /* ---- INDEPENDENT empirical check of the bound, on the digits just packed ------------ */
    unsigned long long maxcoeff = 0, maxcoeff_k = 0, maxcoeff_limbs = 0;
    bool maxcoeff_ran = false;
    const double thb_maxc = now_s();
    {
        const int rc = ntt_shape_maxcoeff(sh, hA, hB, &maxcoeff, &maxcoeff_k, &maxcoeff_ran,
                                          &maxcoeff_limbs);
        if (rc) return rc;
    }
    NTT_HBS(t_maxc, thb_maxc);
    ntt_shape_print(sh, fc, verbose, maxcoeff, maxcoeff_k, maxcoeff_ran, maxcoeff_limbs);

    unsigned long long *dA = nullptr, *dB = nullptr, *dC = nullptr, *dQ = nullptr,
                       *dOut = nullptr, *dRes = nullptr;
    NttArena::BufEntry *ab = ntt_arena_bufs(arena, N, out_slots);   /* nullptr: allocate here */
    if (ab) {
        dA = ab->dA; dB = ab->dB; dC = ab->dC; dQ = ab->dQ; dOut = ab->dOut; dRes = ab->dRes;
    } else {
        CK(cudaMalloc(&dA, N * sizeof(unsigned long long)));
        CK(cudaMalloc(&dB, N * sizeof(unsigned long long)));
        CK(cudaMalloc(&dC, N * sizeof(unsigned long long)));
        CK(cudaMalloc(&dQ, N * sizeof(unsigned long long)));
        CK(cudaMalloc(&dOut, out_slots * sizeof(unsigned long long)));
        CK(cudaMalloc(&dRes, 2 * sizeof(unsigned long long)));
    }
    const double thb_h2d = now_s();
    CK(cudaMemcpy(dA, hA.data(), N * sizeof(unsigned long long), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hB.data(), N * sizeof(unsigned long long), cudaMemcpyHostToDevice));
    NTT_HBS(t_h2d, thb_h2d);

    /* ---- the device passes (shared runner; nbatch == 1) --------------------------------- */
    std::vector<unsigned long long> hfa, hfb, hPre, hPost;
    const NttPassResult pr = ntt_run_passes(sh, fc, dA, dB, dQ, dOut, dRes, 1,
                                           dump, &hfa, &hfb, &hPre, &hPost, true, false, ab ? arena : nullptr);
    const double t_fwd = pr.t_fwd, t_inv = pr.t_inv, t_slot = pr.t_slot;
    const std::vector<unsigned long long> &hOut = pr.hOut;
    const unsigned long long *dDig = pr.digits;
    const unsigned long long hRes[2] = {pr.hRes[0], pr.hRes[1]};
    if (dump) {
        /* host direct DFT of the same inputs, in the bit-reversed order the DIF produces */
        auto brev = [&](unsigned long long i) {
            unsigned long long r = 0, v = i;
            for (int q = 0; q < k; ++q) { r = (r << 1) | (v & 1ull); v >>= 1; }
            return r;
        };
        int b1 = 0, b2 = 0;
        for (unsigned long long i = 0; i < N; ++i) {
            unsigned long long acc = 0, acc2 = 0;
            for (unsigned long long j = 0; j < N; ++j) {
                const unsigned long long w = gl_pow_host(omega, (i * j) % N);
                acc = gl_add_host(acc, gl_mul_host(hA[(size_t)j], w));
                acc2 = gl_add_host(acc2, gl_mul_host(hB[(size_t)j], w));
            }
            if (acc != hfa[(size_t)brev(i)]) ++b1;
            if (acc2 != hfb[(size_t)brev(i)]) ++b2;
        }
        stage2_log::print(stage2_log::debug, "  dump: forward-vs-directDFT  A bad=%d/%llu  B bad=%d/%llu\n", b1,
                    (unsigned long long)N, b2, (unsigned long long)N);
    }
    if (dump && verbose) {
    stage2_log::print(stage2_log::debug, "  carry: %d ripple rounds computed in 1 pass, residual digits >= 2^%d: %llu "
                "(max digit height %llu bits)\n", sh.carry_rounds, bpw,
                (unsigned long long)hRes[0], (unsigned long long)hRes[1]);
    }
    if (hRes[0] != 0) {
        std::fprintf(stderr, NTT_PROBE_NAME ": CARRY DID NOT CONVERGE: %llu digits still >= "
                             "2^%d after %d rounds\n",
                     (unsigned long long)hRes[0], bpw, sh.carry_rounds);
        return 4;
    }
    /* ---- hand the result back, then release everything this call allocated -------------- */
    if (out_slots_u64) *out_slots_u64 = hOut;
    bool exact_ok = true;
    if (dump) {
        /* (a) the exact product's digit array, from the SAME packed operands, via GMP:
           A = sum_j hA[j]*2^(bpw*j), C = A*B, digit j = bits [j*bpw, (j+1)*bpw) of C. */
        mpz_t A, B, C;
        mpz_inits(A, B, C, nullptr);
        mpz_set_ui(A, 0);
        for (unsigned long long j = N; j-- > 0;) {
            mpz_mul_2exp(A, A, (unsigned)bpw);
            mpz_add_u64(A, hA[(size_t)j]);
        }
        mpz_set_ui(B, 0);
        for (unsigned long long j = N; j-- > 0;) {
            mpz_mul_2exp(B, B, (unsigned)bpw);
            mpz_add_u64(B, hB[(size_t)j]);
        }
        mpz_mul(C, A, B);
        stage2_log::print(stage2_log::debug, "  dump: P=%llu S=%d bpw=%d N=%llu slot_bits=%llu payload_digits=%llu "
                    "needed(2*payload-1)=%llu exact_product_bits=%zu\n",
                    (unsigned long long)P, S, bpw, (unsigned long long)N,
                    (unsigned long long)slot_bits,
                    (unsigned long long)((sh.payload_bits + (unsigned)bpw - 1) / (unsigned)bpw),
                    (unsigned long long)((2 * sh.payload_bits - 1) / (unsigned)bpw + 1),
                    mpz_sizeinbase(C, 2));
        stage2_log::print(stage2_log::debug, "  dump: input digits hA[0..3]=%llu %llu %llu %llu   hB[0..3]=%llu %llu "
                    "%llu %llu\n", hA[0], hA[1], hA[2], hA[3], hB[0], hB[1], hB[2], hB[3]);
        stage2_log::print(stage2_log::debug, "  dump: input coefficients (S-bit, %llu word(s) each):", (unsigned long long)W);
        for (unsigned long long i = 0; i < P && i < 6; ++i)
            stage2_log::print(stage2_log::debug, " A[%llu]=%llu B[%llu]=%llu", i, wordsA[(size_t)(i * W)], i,
                        wordsB[(size_t)(i * W)]);
        stage2_log::print(stage2_log::debug, "\n");
        dump_cmp("pre-carry (after inverse NTT)", hPre, C, N, bpw, 12);
        dump_cmp("post-carry", hPost, C, N, bpw, 12);
        /* What the inverse NTT MUST return: the cyclic convolution of the two input arrays,
           reduced mod p.  Compared element by element, and also up to a constant factor (a
           missing/incorrect 1/N shows up as a single global ratio). */
        std::vector<unsigned long long> conv(N, 0);
        for (unsigned long long i = 0; i < N; ++i)
            for (unsigned long long j = 0; j < N; ++j)
                conv[(size_t)((i + j) % N)] = gl_add_host(conv[(size_t)((i + j) % N)],
                                                          gl_mul_host(hA[(size_t)i],
                                                                      hB[(size_t)j]));
        unsigned long long cbad = 0, cfirst = 0;
        for (unsigned long long i = 0; i < N; ++i)
            if (conv[(size_t)i] != hPre[(size_t)i]) { if (!cbad) cfirst = i; ++cbad; }
        stage2_log::print(stage2_log::debug, "  dump: inverse-vs-host-cyclic-conv bad=%llu/%llu first=%llu\n", cbad,
                    (unsigned long long)N, cfirst);
        if (conv[0] != 0) {
            const unsigned long long ratio =
                gl_mul_host(hPre[0], gl_pow_host(conv[0], GL_P - 2ull));
            unsigned long long rbad = 0;
            for (unsigned long long i = 0; i < N; ++i)
                if (gl_mul_host(conv[(size_t)i], ratio) != hPre[(size_t)i]) ++rbad;
            stage2_log::print(stage2_log::debug, "  dump: pre-carry / conv ratio=%llu (uniform for %llu/%llu entries); "
                        "N^-1 mod p=%llu, N mod p=%llu\n", ratio, N - rbad,
                        (unsigned long long)N, sh.n_scale, N % GL_P);
        }
        stage2_log::print(stage2_log::debug, "  dump: conv[0..5]=%llu %llu %llu %llu %llu %llu  pre-carry[0..5]=%llu "
                    "%llu %llu %llu %llu %llu\n", conv[0], conv[1], conv[2], conv[3], conv[4],
                    conv[5], hPre[0], hPre[1], hPre[2], hPre[3], hPre[4], hPre[5]);
        mpz_clears(A, B, C, nullptr);
    }
    if (out_exact) {
        std::vector<unsigned long long> hDig(N, 0);
        double thb = now_s();
        CK(cudaMemcpy(hDig.data(), dDig, N * sizeof(unsigned long long),
                      cudaMemcpyDeviceToHost));
        NTT_HBS(t_d2h, thb);
        thb = now_s();
        exact_ok = extract_exact_coeffs(hDig.data(), N, bpw, slot_bits, slot_stride, out_slots,
                                        out_exact);
        NTT_HBS(t_ext, thb);
        thb = now_s();
        /* TWO INDEPENDENT EXTRACTIONS of the same coefficient must agree: the device's
           mod-SLOT_MOD projection (checked against GMP by run_poly's `verify` block) and this
           host extraction of the full integer.  A disagreement means one of them is wrong, so
           it is reported and turned into a non-zero return instead of being passed on. */
        for (unsigned long long kx = 0; kx < out_slots && exact_ok; ++kx) {
            mpz_t tx;
            mpz_init(tx);
            mpz_import(tx, (*out_exact)[(size_t)kx].size(), -1, 8, 0, 0,
                       (*out_exact)[(size_t)kx].data());
            mpz_mod_ui(tx, tx, (unsigned long)SLOT_MOD);
            const unsigned long long got = (unsigned long long)mpz_get_ui(tx);
            mpz_clear(tx);
            if (got != hOut[(size_t)kx]) {
                std::fprintf(stderr, "%s: EXACT SLOT MISMATCH at k=%llu: host extraction "
                                     "%llu != device slot assembly %llu\n",
                             NTT_PROBE_NAME, (unsigned long long)kx, got,
                             (unsigned long long)hOut[(size_t)kx]);
                exact_ok = false;
            }
        }
        NTT_HBS(t_xchk, thb);
    }
    if (st) {
        st->P = P;
        st->S = S;
        st->N = N;
        st->k = k;
        st->bpw = bpw;
        st->slot_bits = slot_bits;
        st->slot_stride = slot_stride;
        st->slot_words = slot_words;
        st->out_slots = out_slots;
        st->L_terms = L_terms;
        st->cw = (unsigned long long)((slot_bits + 63) / 64);
        st->passes_fwd = sh.passes_fwd;
        st->passes_total = sh.passes_total;
        st->carry_rounds = sh.carry_rounds;
        st->carry_residual = hRes[0];
        st->carry_max_bits = hRes[1];
        st->mem_mb = sh.mem_mb;
        st->t_fwd = t_fwd;
        st->t_inv = t_inv;
        st->t_slot = t_slot;
        st->fuse_t = fc.t;
        st->fuse_nms = fc.nms;
        for (int q = 0; q < fc.nms && q < 8; ++q) st->fuse_ms[q] = fc.ms[q];
        st->exact_valid = exact_ok;
        st->t_setup = hb_t_setup;
        st->t_pack = hb_t_pack;
        st->t_maxc = hb_t_maxc;
        st->t_h2d = hb_t_h2d;
        st->t_carry = 0.0;
        st->t_d2h = hb_t_d2h;
        st->t_ext = hb_t_ext;
        st->t_xchk = hb_t_xchk;
    }

    if (!ab) { cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dQ); cudaFree(dOut);
               cudaFree(dRes); }
    return exact_ok ? 0 : 8;
}
