#pragma once
#include "../../../platform/ecm_stage2_logging.h"
#include "../runtime/device_memory_ledger.h"
#include "parameters.h"
#include "launch.h"
#include "../arithmetic/goldilocks/goldilocks.cuh"
#include <gmp.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <map>
#include <vector>
#include <algorithm>
#include "host_trace.h"
#define CK(x)                                                                        \
    do {                                                                             \
        const cudaError_t e_ = (x);                                                  \
        if (e_ != cudaSuccess) {                                                      \
            size_t ck_free_ = 0, ck_total_ = 0;                                       \
            cudaMemGetInfo(&ck_free_, &ck_total_);                                    \
            std::fprintf(stderr,                                                      \
                         "CUDA error %s (%d) at %s:%d\n  call: %s\n  device free=%.1f " \
                         "MB of %.1f MB (%.1f MB in use)\n",                         \
                         cudaGetErrorString(e_), (int)e_, __FILE__, __LINE__, #x,     \
                         (double)ck_free_ / 1048576.0, (double)ck_total_ / 1048576.0, \
                         (double)(ck_total_ - ck_free_) / 1048576.0);                 \
            std::fflush(stderr);                                                      \
            std::exit(2);                                                            \
        }                                                                            \
    } while (0)

struct FuseCtx;
struct FuseBaseStats;
struct FuseCallGuard;
struct NttWorkspacePolicy;
struct NttArena;
struct NttCfg;
struct NttMulStats;
struct NttShape;
struct NttShapeCacheEntry;
struct NttPassResult;
struct NttReduceHook;
struct NttInputHook;
extern FuseBaseStats g_fuse_base;
unsigned int ntt_gl_reduce_requested_mode();
void ntt_gl_reduce_configure();
double now_s();
void ntt_bitrev_inplace(unsigned long long *d, unsigned long long n, int k);
void ntt_dif(unsigned long long *d, unsigned long long n, int k,
                    unsigned long long omega);
void ntt_bitrev_perm(unsigned long long *d, unsigned long long *scratch,
                            unsigned long long n, int k);
void ntt_forward(unsigned long long *d, unsigned long long n, int k,
                        unsigned long long omega, unsigned long long *scratch);
void ntt_inverse(unsigned long long *d, unsigned long long n, int k,
                        unsigned long long omega_inv, unsigned long long n_scale,
                        unsigned long long *scratch);
bool ntt_carry_check_requested();
void ntt_tile_attributes(bool warp, int smem, bool preferred=false);
void fuse_base_allocate(unsigned long long **p, size_t words);
void fuse_base_free(unsigned long long *&p, size_t words);
void fuse_plan_stages(FuseCtx &c, int rem);
unsigned long long fuse_env_ull(const char *name, unsigned long long dflt);
bool fuse_compact_scratch();
bool fuse_shape_device_supported();
bool fuse_shape_policy_supported(int t);
bool fuse_joint_policy_supported();
int fuse_selected_tile(int k,int requested);
int fuse_outer_max(int t,int k,bool &coop);
void fuse_describe(FuseCtx &c, unsigned long long n, int k, unsigned long long omega,
                      unsigned long long omega_inv, bool compact=fuse_compact_scratch());
unsigned long long fuse_planned_table_words(const FuseCtx &c);
size_t fuse_planned_base_words(const FuseCtx &c);
void fuse_init(FuseCtx &c, unsigned long long n, int k, unsigned long long omega,
                      unsigned long long omega_inv, bool compact=fuse_compact_scratch());
void fuse_release(FuseCtx &c);
size_t fuse_base_words(const FuseCtx &c);
void ntt_fuse_cache_tables(FuseCtx &c, unsigned long long omega,
                                  unsigned long long omega_inv);
unsigned long long fuse_grid(unsigned long long groups, unsigned long long grp_per_blk);
bool fuse_trace_on();
void fuse_mark(const char *tag, double &t0);
void ntt_forward_fused(unsigned long long *d, const FuseCtx &c,
                              unsigned long long omega, unsigned long long nbatch = 1, cudaStream_t stream = 0);
void ntt_inverse_fused(unsigned long long *d, const unsigned long long *b,
                              const FuseCtx &c, unsigned long long omega_inv,
                              unsigned long long n_scale, unsigned long long nbatch = 1, cudaStream_t stream = 0);
void ntt_arena_pin_smem(NttArena *ar, int t, bool warp);
void ntt_arena_fuse(NttArena *ar, unsigned long long n, int k, unsigned long long omega,
                           unsigned long long omega_inv, FuseCtx &out);
void ntt_trim_fuse_headroom(NttArena *arena,unsigned long long keep_n,
    size_t required,const char *boundary,size_t &available);
int gl_selftest();
bool exact_ok_terms(unsigned long long L, int bpw);
bool exact_ok_nterms(unsigned long long n, int bpw);
double mpz_log2d(const mpz_t x);
double mpz_log2d_p();
double coeff_bound_bits_terms(unsigned long long L, int bpw);
double bound_over_p_terms(unsigned long long L, int bpw);
double coeff_bound_bits(unsigned long long n, int bpw);
NttCfg choose_cfg(unsigned long long payload_bits, unsigned long long slot_bits,
                         unsigned long long P = 0, int force_bpw = 0);
bool max_conv_coeff_gmp(const std::vector<unsigned long long> &hA,
                               const std::vector<unsigned long long> &hB, unsigned long long L,
                               unsigned long long *out_max, unsigned long long *out_k);
double digit_sq_double(int bpw);
void mpz_add_u64(mpz_t acc, unsigned long long v);
void dump_cmp(const char *tag, const std::vector<unsigned long long> &d, const mpz_t C,
                     unsigned long long n, int bpw, int show);
bool ntt_host_break_on();
bool extract_exact_coeffs(const unsigned long long *digits, unsigned long long n,
                                 int bpw, unsigned long long slot_bits,
                                 unsigned long long slot_stride, unsigned long long out_slots,
                                 std::vector<std::vector<unsigned long long>> *out);
bool ntt_shape_query(unsigned long long P, int S, unsigned long long *N_out, int *bpw_out,
                            unsigned long long *slot_bits_out, unsigned long long *slot_words_out,
                            unsigned long long *slot_stride_out, unsigned long long *out_slots_out,
                            int force_bpw = 0);
int ntt_shape_plan(unsigned long long P, int S, int device, NttArena *arena,
                          FuseCtx &fc, NttShape &sh, int force_bpw = 0);
int ntt_shape_plan_uncached(unsigned long long P, int S, int device, NttArena *arena,
                                   FuseCtx &fc, NttShape &sh, int force_bpw);
int ntt_shape_exactness(const NttShape &sh);
bool ntt_pack_operand(const NttShape &sh, const unsigned long long *wordsA,
                             const unsigned long long *wordsB, unsigned long long *hA,
                             unsigned long long *hB, unsigned long long *maxdigit_out);
int ntt_shape_maxcoeff(const NttShape &sh, const std::vector<uint64_t> &hA,
                              const std::vector<uint64_t> &hB, unsigned long long *maxcoeff,
                              unsigned long long *maxcoeff_k, bool *ran,
                              unsigned long long *limbs);
void ntt_shape_print(const NttShape &sh, const FuseCtx &fc, bool verbose,
                            unsigned long long maxcoeff, unsigned long long maxcoeff_k,
                            bool maxcoeff_ran, unsigned long long maxcoeff_limbs);
NttPassResult ntt_run_passes(const NttShape &sh, const FuseCtx &fc,
                                    unsigned long long *dA, unsigned long long *dB,
                                    unsigned long long *dQ, unsigned long long *dOut,
                                    unsigned long long *dRes, unsigned long long nbatch,
                                    int dump, std::vector<unsigned long long> *hfa,
                                    std::vector<unsigned long long> *hfb,
                                    std::vector<unsigned long long> *hPre,
                                    std::vector<unsigned long long> *hPost,
                                    bool need_hout = true, bool defer_res = false, NttArena *arena=nullptr);
void ntt_pass_read_marks(NttPassResult &r, double *t_fwd, double *t_inv, double *t_chk);
bool ntt_hook_window(const NttReduceHook *hook, unsigned long long slots,
                             unsigned long long &count);
int ntt_poly_mul_batch_host(unsigned long long P, int S, int device, unsigned long long nbatch,
                            const unsigned long long *wordsA, const unsigned long long *wordsB,
                            std::vector<unsigned long long> *out_slots_u64,
                            NttMulStats *st, NttArena *arena,
                            const NttReduceHook *hook,
                            unsigned long long **digits_out = nullptr);
int ntt_poly_mul_batch_dev(unsigned long long P, int S, int device, unsigned long long nbatch,
                           const unsigned long long *dAin, const unsigned long long *dBin,
                           NttMulStats *st, NttArena *arena, const NttReduceHook *hook,
                           unsigned long long **digits_out = nullptr, int force_bpw = 0,
                           bool defer_carry = false, const NttInputHook *input = nullptr);
int ntt_batch_carry_finish(NttArena *arena, unsigned long long N, unsigned long long nbatch,
                           NttMulStats *st);
void ntt_workspace_check(int device);
void ntt_fuse_lifetime_check(int device);
void fuse_fixture_env(const char *key,const char *value);
void ntt_fuse_capacity_check(int device);
void ntt_fuse_coop_check(int device);
int ntt_poly_mul_host(unsigned long long P, int S, int device, bool verbose, int dump,
                      const unsigned long long *wordsA, const unsigned long long *wordsB,
                      std::vector<unsigned long long> *out_slots_u64,
                      std::vector<std::vector<unsigned long long>> *out_exact,
                      NttMulStats *st, NttArena *arena = nullptr);
template <bool INVERSE>
void launch_outer_coop(int m,unsigned long long *a,unsigned long long n,int stage,
                              const unsigned long long *coarse,const unsigned long long *radix,
                              unsigned long long nbatch,unsigned long long stride,cudaStream_t stream=0)
{
    const auto S=INVERSE ? (1ull<<stage) : (n>>(stage+m));
    const int v=m==8 ? 16 : 32;
    if(m<5 || m>8 || S<(unsigned long long)v) {
        std::fprintf(stderr,NTT_PROBE_NAME ": invalid cooperative outer M=%d S=%llu\n",m,S);
        std::exit(3);
    }
    dim3 grid((unsigned int)(n/((1ull<<m)*v)),(unsigned int)nbatch);
    switch(m) {
    case 5:ecm_launch_outer_coop_kernel<5,INVERSE>(EcmCudaLaunch{grid,256,0,stream}, a,n,stage,coarse,radix,stride);break;
    case 6:ecm_launch_outer_coop_kernel<6,INVERSE>(EcmCudaLaunch{grid,256,0,stream}, a,n,stage,coarse,radix,stride);break;
    case 7:ecm_launch_outer_coop_kernel<7,INVERSE>(EcmCudaLaunch{grid,256,0,stream}, a,n,stage,coarse,radix,stride);break;
    case 8:ecm_launch_outer_coop_kernel<8,INVERSE>(EcmCudaLaunch{grid,256,0,stream}, a,n,stage,coarse,radix,stride);break;
    }
}
struct FuseCtx {
    unsigned long long n = 0;
    int k = 0, t = 0;
    int ms[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int nms = 0;
    int outer_stages = 0;                                /* stages covered by outer passes */
    unsigned long long *tblF = nullptr, *tblI = nullptr; /* 2^t words each */
    unsigned long long *scr = nullptr;                   /* per-pass coarse tables */
    unsigned long long *scr2 = nullptr;                  /* per-pass radix tables */
    size_t scrWords = 0;
    size_t scr2Words = 64;
    int m_max = FUSE_MAX_M;
    bool ready = false;
    bool coop_outer = false, compact_scratch = true;
    bool warp_tail = false;                            /* kernel choice, no new tables/buffers */
    bool arena_borrowed = false;                         /* this copy must not release arena storage */
    int passes_fwd = 0;
    /* ---- OPTIONAL cache of the per-pass twiddle tables (section 18, slice S3) ------------
       build_pass_table_kernel / build_radix_table_kernel are pure functions of (shape, pass),
       yet the per-call path rebuilds them with two extra launches on EVERY multiply.  A
       caller that multiplies the same shape thousands of times (the batched stage-2 tree)
       fills these once with ntt_fuse_cache_tables() and every later multiply reuses them.
       When tables_cached is false nothing here is touched and the path is bit-identical to
       the original one. */
    bool tables_cached = false;
    unsigned long long *passF[FUSE_MAX_PASSES] = {nullptr};
    unsigned long long *radF[FUSE_MAX_PASSES] = {nullptr};
    unsigned long long *passI[FUSE_MAX_PASSES] = {nullptr};
    unsigned long long *radI[FUSE_MAX_PASSES] = {nullptr};
};

struct FuseBaseStats {
    unsigned long long allocations=0, frees=0, live_bytes=0, peak_bytes=0;
};

struct FuseCallGuard {
    FuseCtx &fc;
    explicit FuseCallGuard(FuseCtx &context):fc(context) {}
    ~FuseCallGuard() { if (!fc.arena_borrowed) fuse_release(fc); }
    FuseCallGuard(const FuseCallGuard &)=delete;
    FuseCallGuard &operator=(const FuseCallGuard &)=delete;
};

struct NttWorkspacePolicy {
    bool workspace_pool = [] {
        const char *e=std::getenv("NTT_ARENA_WORKSPACE_POOL");
        return !e || !*e || std::atoi(e)!=0;
    }();
    bool workspace_reuse_bq = [] {
        const char *e=std::getenv("NTT_WORKSPACE_REUSE_BQ");
        return e && *e && std::atoi(e)!=0;
    }();
    unsigned big_buffer_count(bool allow_pool=true) const {
        return workspace_pool && allow_pool && workspace_reuse_bq ? 2 : 3;
    }
};



// Host-only cache. Nodes borrow arena arrays/tables; clear before any owner frees
// them. Capture uses a private stream, replay keeps the original stream order.
struct NttGraphCache {
    struct Key {
        unsigned long long v[64]={};
        bool operator==(const Key &other)const{return std::memcmp(v,other.v,sizeof(v))==0;}
    };
    struct Entry {
        Key key;unsigned long long seen=0,last=0;
        cudaGraphExec_t forward=nullptr,inverse=nullptr;
    };
    bool enabled=fuse_env_ull("NTT_SHORT_GRAPH",0)!=0,failed=false;
    unsigned long long max_k=fuse_env_ull("NTT_SHORT_GRAPH_MAX_K",19);
    unsigned long long max_words=fuse_env_ull("NTT_SHORT_GRAPH_MAX_WORDS",1ull<<22);
    unsigned long long min_calls=std::max(1ull,fuse_env_ull("NTT_SHORT_GRAPH_MIN_CALLS",4));
    size_t limit=(size_t)std::max(1ull,std::min(256ull,fuse_env_ull("NTT_SHORT_GRAPH_ENTRIES",64)));
    cudaStream_t capture_stream=nullptr;
    std::vector<Entry> entries;
    unsigned long long candidates=0,hits=0,builds=0,launches=0,skipped=0,cold=0;
    unsigned long long failures=0,invalidations=0,evictions=0,peak_entries=0,nodes=0,saved_launches=0;
    double setup_seconds=0;
    static void destroy(Entry &entry);
    void clear();
    void release();
    ~NttGraphCache();
    NttGraphCache()=default;
    NttGraphCache(const NttGraphCache &)=delete;
    NttGraphCache &operator=(const NttGraphCache &)=delete;
    static bool recoverable(cudaError_t error);
    bool accept(cudaError_t error);
    bool capture(cudaGraphExec_t &exec,bool inverse,const FuseCtx &fc,
        unsigned long long *a,unsigned long long *b,unsigned long long omega,
        unsigned long long omega_inv,unsigned long long scale,unsigned long long batch);
    Entry *lookup(int device,const FuseCtx &fc,unsigned long long *a,unsigned long long *b,
        unsigned long long omega,unsigned long long omega_inv,unsigned long long scale,unsigned long long batch);
    void print()const;
};

struct NttArena : NttWorkspacePolicy {
    NttGraphCache short_graph;
    int device = -1;
    size_t cap_bytes = 0;                 /* 0 = unlimited */
    size_t bytes = 0; // exact arena-owned CUDA payload; accounting v2
    unsigned int *carry_scratch=nullptr;
    size_t carry_scratch_bytes=0,carry_scratch_peak=0;
    unsigned long long carry_fused_calls=0,carry_skipped_calls=0,carry_refusals=0,carry_grows=0;
    void drop_carry_scratch();
    unsigned int *carry_workspace(unsigned long long n,unsigned long long batch);
    size_t attr_words = 0;                /* the shared-memory ceiling pinned on the tile
                                             kernels (see ntt_arena_pin_smem) */
    unsigned long long fuse_hits = 0, fuse_builds = 0, buf_hits = 0, buf_builds = 0;

    struct FuseEntry {
        unsigned long long n = 0;
        int k = 0;
        unsigned long long omega = 0;
        FuseCtx fc;
    };
    struct BufEntry {
        unsigned long long n = 0, out_slots = 0;
        /* slice S4: a batched multiply of `nbatch` slices needs nbatch*N words per buffer,
           so the batch count is part of the key (0 and 1 mean the same thing: one slice) */
        unsigned long long nbatch = 1;
        unsigned long long *dA = nullptr, *dB = nullptr, *dC = nullptr, *dQ = nullptr,
                           *dOut = nullptr, *dRes = nullptr;
    };
    std::vector<FuseEntry> fuses;
    struct BigEntry {                       /* shape-independent A/B and optional separate Q */
        unsigned long long n = 0, nbatch = 1;
        unsigned long long *dA = nullptr, *dB = nullptr, *dQ = nullptr;
        size_t capacity = 0;                /* words in EACH physical allocation */
        bool q_alias_b = false;             /* Q borrows B after the inverse; never owns it */
        size_t words = 0;                   /* what this entry costs, for eviction accounting */
    };
    struct SmallEntry {                     /* only these depend on out_slots */
        unsigned long long n = 0, nbatch = 1;
        unsigned long long out_cap = 0;     /* GROWN, never shrunk: reused by every smaller
                                               out_slots at the same N */
        unsigned long long *dOut = nullptr, *dRes = nullptr;
        size_t words = 0;
    };
    std::vector<BigEntry> bigs;
    /* A/B/Q are temporary default-stream scratch. Preserve shape-local dRes below, and
       retain keyed buffers for calls that export a borrowed digit pointer. */
    BigEntry workspace;
    unsigned long long workspace_hits=0, workspace_grows=0, workspace_mallocs=0, workspace_frees=0;
    unsigned long long legacy_mallocs=0, legacy_frees=0;
    unsigned long long alias_snapshots=0, alias_snapshot_bytes=0;
    unsigned long long workspace_bq_calls=0;
    size_t workspace_bq_saved_peak=0;
    size_t peak_big_bytes=0, peak_small_bytes=0, peak_table_bytes=0, peak_owned_bytes=0;
    size_t peak_fuse_base_bytes=0, peak_full_bytes=0;
    unsigned long long phase_fuse_evictions=0,phase_fuse_evicted_bytes=0;
    /* Deterministic allocation-failure fixture; never set by the production engine. */
    int workspace_fail_alloc=0;
    std::vector<SmallEntry> smalls;
    BufEntry cur;                           /* the composite the caller is handed */
    unsigned long long overflow = 0;      /* shapes that did NOT fit the cap (per-call path) */
    /* ---- THE TABLE CACHES ARE EVICTABLE, AND THAT IS THE CHEAPEST MEMORY WE HAVE ------------
       ntt_fuse_cache_tables() stores one table pair per outer pass, summing to about N + N/2
       words for a cached shape -- 35% of what one transform length costs, ~1.2 GB of the 4.6 GB
       the real shape holds.  Every byte of it is a PURE CACHE: the per-call path rebuilds the same
       tables with two extra launches, and the kernels that read them check `tables_cached` first,
       so dropping them can only cost speed, never correctness.  When an allocation would exceed
       the cap we therefore evict the caches of every shape EXCEPT the one being allocated (the hot
       shape keeps its caching) and retry before falling back to the per-call path. */
    unsigned long long tbl_evictions = 0, tbl_words_freed = 0;

    /* A cudaMalloc THAT CAN FAIL WITHOUT KILLING THE RUN.  The cap accounting is in words the
       arena believes it owns, but the device's REAL free memory can be smaller -- the stage-2
       engine holds its own pools (the ladder, the frontier, the S5 forests) and the driver
       reserves some -- so an allocation that passes the cap check can still come back "out of
       memory".  Measured, not theorised: at NTT_S4_BATCH_MB=96 the arena's own check passed and
       the very next cudaMalloc failed, and because that site used CK() the whole run died
       (ntt_poly_probe.cu:1634 cudaMalloc dA, exit 2).  Every arena allocation is a CACHE, so
       failing one is recoverable: the caller falls back to the per-call path, which is exactly
       the `overflow` path that already exists for cap refusals. */
    static bool try_malloc(unsigned long long **p, size_t bytes);

    void drop_workspace();

    /* Returns the remaining allocation span for an arena-owned device input. Snapshot such
       inputs before lookup: it can overwrite scratch or evict an older exported keyed buffer. */
    bool input_span(const unsigned long long *p, size_t &remaining) const;

    struct Payload {
        size_t big=0,digits=0,table=0,base=0,total=0;
    };
    Payload payload() const;
    // Only use at a completed phase boundary. No digits/verdicts, big scratch
    // or owner pointers are touched. Drop the largest cold context first to
    // minimise the number of shapes rebuilt later; preserve the next hot N.
    size_t drop_cold_fuse(unsigned long long keep_n);
    void update_peaks();

    void print_workspace_stats() const;

    /* the words a cached shape's tables occupy: sum over outer passes of (S + 2^M) for the forward
       and again for the inverse, exactly as ntt_fuse_cache_tables allocates them */
    static unsigned long long fuse_table_words(const FuseCtx &c);

    static void fuse_drop_tables(FuseCtx &c);

    /* free the table caches of every shape except `keep_n`, AND the big/small buffers of every
       other (n, nbatch) -- see the long comment above: all of it is a cache, the multiply that is
       being set up now is the only live user, and without evicting the BIGS the arena still held
       5627 MB of earlier shapes when the fold's own 3072 MB shape asked for room at P=115200 and
       the run fell back to per-call cudaMalloc and died with "out of memory" (measured).  Returns
       the words freed. */
    unsigned long long evict_other_shapes(unsigned long long keep_n,
                                          unsigned long long keep_nbatch);

    /* a fused shape becomes unbounded work per call, so it must not be built while another
       shape's buffers are still live on a small card: the caller sets cap_bytes and the two
       out-of-memory paths below simply fall back to the per-call behaviour. */
    ~NttArena();

    void release();
    double mb() const;
};

struct NttCfg {
    unsigned long long nwords;      /* N = 2^k transform length */
    unsigned long long slot_words;  /* words occupied by one coefficient (padded) */
    unsigned long long L;           /* NONZERO digits per operand = the true term count */
    int k, bpw;
    bool ok;
    std::string why;
};

struct NttMulStats {
    unsigned long long P = 0, N = 0, out_slots = 0;      /* coefficients, transform length */
    unsigned long long slot_bits = 0, slot_stride = 0, slot_words = 0, L_terms = 0;
    unsigned long long cw = 0;                           /* words per exact coefficient */
    int S = 0, k = 0, bpw = 0;
    int passes_fwd = 0, passes_total = 0, carry_rounds = 0;
    double mem_mb = 0.0, t_fwd = 0.0, t_inv = 0.0, t_slot = 0.0;
    /* OBJECTIVE 4 (docs/architecture/STAGE2.md): the batched entry point packs its
       operands ON THE HOST, scans them for the max coefficient, and ships them to the device --
       none of which any of the timers above covers, because they were written for the other
       entry point.  Measured separately so the "111 us per call that nobody measured" can be
       attributed instead of guessed. */
    double t_hpack = 0.0, t_scan = 0.0, t_h2d_batch = 0.0;
    /* the extra pass the carry-convergence assert costs (section 34), and its three parts
       (section 41: the reset launch, the kernel, the blocking readback) */
    double t_check = 0.0;
    double t_check_reset = 0.0, t_check_kernel = 0.0, t_check_d2h = 0.0;
    /* where the REST of the device entry point's time goes: the shape plan (choose_cfg, which
       proves the exactness bound) and the device-to-device copy of the two operands into the
       arena's scratch (section 34) */
    double t_plan = 0.0, t_opcopy = 0.0;
    double t_hout = 0.0;
    /* the WHOLE call's wall time (section 27): the tree's per-level floor is whatever is left once
       the parts below are subtracted, so the total has to be reported next to them */
    double t_total_call = 0.0;
    unsigned long long carry_residual = 0, carry_max_bits = 0;
    bool carry_deferred = false;            /* actual path, including arena-refused fallback */
    int fuse_t = 0, fuse_nms = 0, fuse_ms[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    bool exact_valid = false;
    /* slice S4: where the per-CALL FIXED cost goes.  t_fwd/t_inv/t_slot already isolate the
       device passes; these seven split the rest (the "host_side" of the S3 breakdown) so the
       question "is the fixed cost launch overhead or host GMP?" is answered by measurement
       instead of by argument.  Zero unless NTT_HOST_BREAK=1 (the timers are pure clock reads,
       but keeping the default path free of them keeps the probe's timings comparable with the
       ones already recorded in the plan). */
    double t_setup = 0, t_pack = 0, t_maxc = 0, t_h2d = 0, t_carry = 0, t_d2h = 0,
           t_ext = 0, t_xchk = 0;
};

struct NttShape {
    int device = 0;
    char dev_name[256] = {0};
    unsigned long long P = 0, N = 0, out_slots = 0, slot_bits = 0, slot_stride = 0,
                       slot_words = 0, L_terms = 0, W = 0, payload_bits = 0;
    int S = 0, k = 0, bpw = 0;
    unsigned long long omega = 0, omega_inv = 0, n_scale = 0;
    double mem_mb = 0, bound_bits = 0, bound_ratio = 0, old_bound_bits = 0, p_bits = 0;
    bool bound_ok = false, old_bound_ok = false;
    int passes_fwd = 0, passes_total = 0, carry_rounds = 0;
};

struct NttShapeCacheEntry {
    unsigned long long P = 0;
    int S = 0, device = -1, force_bpw = 0;
    NttShape sh;
};

struct NttPassResult {
    double t_fwd = 0.0, t_inv = 0.0, t_slot = 0.0;
    /* the carry-convergence assert below is a WHOLE EXTRA PASS over the digit array (a memset, a
       kernel over N*nbatch digits and a device-to-host copy) and it was untimed, which is why
       more than half of ntt_seconds had no owner (docs/architecture/STAGE2.md). */
    double t_check = 0.0;
    /* ... and THAT total is itself split, because its three parts have three different fixes
       (section 41): the reset memset launch, the kernel itself, and the blocking 16-byte D2H
       that drains the whole pipeline before the host can look at the two counters. */
    double t_check_reset = 0.0, t_check_kernel = 0.0, t_check_d2h = 0.0;
    /* the slot assembly + its D2H: skipped entirely when the caller has a reduction hook, which
       never reads them (section 35) */
    double t_hout = 0.0;
    unsigned long long *digits = nullptr;      /* the buffer holding the canonical digits */
    std::vector<unsigned long long> hOut;      /* nbatch * out_slots slot projections */
    std::vector<unsigned long long> hRes;      /* 2 * nbatch carry diagnostics */
    /* DEFERRED IN-STREAM MARKS (docs section 41).  Reading an event's elapsed time is only
       possible once the event has completed, so a call that has no other reason to drain the
       pipeline must either block here (which is exactly the cost being removed) or hand the
       marks back untouched.  The batched device entry point does the latter: it owns the one
       blocking copy at the end of its chunk, and reads these five marks right afterwards.
       ev[0..1] = forward, ev[1..2] = inverse, ev[3..4] = carry-residual kernel. */
    cudaEvent_t dev_ev[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};
    /* the carry-residual counters, read in the same deferred step (empty until then) */
    bool res_deferred = false;
};

struct NttReduceHook {
    void *ctx = nullptr;
    /* digits: nbatch*N u64 of bpw-bit digits, laid out slice after slice.  Coefficient k of
       slice s occupies exactly the slot_words digits at [s*N + k*slot_words, +slot_words)
       (the packing stride is word-aligned, so a coefficient never straddles a slot boundary),
       and its value is < 2^slot_bits. Write nbatch*count coefficients of `w` u64 words
       each into `out`, slice-major. Shape/carry remain full. first/count select source slots
       and count is the compact output stride; ~0ull means full. count=0 still runs the hook
       for source diagnostics, with an optional null output pointer. */
    void (*run)(void *ctx, const unsigned long long *digits, unsigned long long n, int bpw,
                unsigned long long slot_words, unsigned long long slot_bits,
                unsigned long long out_slots, unsigned long long nbatch,
                unsigned long long *out, unsigned long long w,
                unsigned long long first, unsigned long long count) = nullptr;
    unsigned long long *out = nullptr;     /* device buffer: nbatch*resolved_count*w words */
    unsigned long long w = 0;              /* u64 words per reduced coefficient */
    /* how many (slice, coefficient) pairs to verify against the host on EVERY batch: a full
       check is cheap for small shapes, and a sample is the honest option for big ones.  0 =
       none (the caller then says so in its own report). */
    unsigned long long sample = 0;
    unsigned long long first = 0, count = ~0ull;
};

struct NttInputHook {
    void *ctx = nullptr;
    int (*run)(void *ctx, const NttShape &shape, unsigned long long nbatch,
               unsigned long long *dA, unsigned long long *dB) = nullptr;
};

NttArena::BufEntry *ntt_arena_bufs(NttArena *ar, unsigned long long n,
                                          unsigned long long out_slots,
                                          unsigned long long nbatch = 1, bool allow_pool = true);
