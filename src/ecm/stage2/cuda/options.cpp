// Single owner of defaults and dynamic snapshots, in their original initialization order.
#include "options.h"
#include "points.h"
#include "product.h"
#include "s4.h"
#include "tree.h"

namespace {
// Set before the engine's static flag snapshots. Explicit environment overrides
// remain available for comparison with the experiment executable.
struct ProductionDefaults {
    ProductionDefaults() {
        const char *keys[] = {"NTT_S4_MERSENNE", "NTT_SMALL_PRIME_REUSE",
            "NTT_GIANT_SEED_DEVICE", "NTT_GFINV_SEG_EXACT", "NTT_GFINV_BATCH",
            "NTT_FOLD_FLAT", "NTT_FOLD_DEVICE", "NTT_GROOT_DEVICE", "NTT_SCALED_DESCENT",
            "NTT_S4_OUTPUT_WINDOW", "NTT_S4_CHUNK_OUTPUT", "NTT_DEVICE_GLEAF",
            "NTT_GROOT_TO_FOLD", "NTT_S4_ORACLE_ASYNC", "NTT_S4_CARRY_BATCH",
            "NTT_FUSE_WARP_TAIL", "NTT_XADD6", "NTT_D_MODEL", "NTT_BABY_DEVICE",
            "NTT_GIANT_SEED_PAIR"};
        for (const char *key : keys) set_default(key, "1");
#if defined(NTT_GL_FIXED_MODE) && NTT_GL_FIXED_MODE >= 0
        set_default("NTT_GL_SHORT_REDUCE", (NTT_GL_FIXED_MODE&1) ? "1" : "0");
#else
        set_default("NTT_GL_SHORT_REDUCE", "1");
#endif
#if defined(NTT_GL_FIXED_MODE) && NTT_GL_FIXED_MODE == 3
        set_default("NTT_POINT_MERSENNE", "1");
#endif
        set_default("NTT_HOST_EVENT_WAIT", "0");
        set_default("NTT_HOST_READBACK_WAIT", "0");
        set_default("NTT_S4_TIMING_FIFO", "1");
        set_default("NTT_FUSE_COOP_OUTER", "2");
        set_default("NTT_GIANT_BASE_CPU", "0");
        set_default("NTT_DEVICE_GLEAF_MAX_MB", "512");
        set_default("NTT_FOLD_DEVICE_MAX_MB", "640");
        set_default("NTT_FOLD_OWNER_REUSE", "3");
        set_default("NTT_S4_BATCH_MB", "64");
        set_default("NTT_S4_SAMPLE", "96");
        set_default("NTT_S4_CHECK_EVERY", "8");
        set_default("NTT_NAME_MAX", "1");
    }
    static void set_default(const char *key, const char *value) {
        if (std::getenv(key)) return;
#ifdef _WIN32
        _putenv_s(key, value);
#else
        setenv(key, value, 0);
#endif
    }
} production_defaults;
}

ecm_stage2::ShortNttBatchPolicy g_short_ntt_batches;

const bool g_fold_flat = [] {
    const char *e=std::getenv("NTT_FOLD_FLAT");
    return e && *e && std::atoi(e)!=0;
}();

const bool g_gfinv_seg_exact = [] {
    const char *e=std::getenv("NTT_GFINV_SEG_EXACT");return !e || std::atoi(e)!=0;
}();

const bool g_gfinv_seg_check = [] {
    const char *e=std::getenv("NTT_GFINV_SEG_CHECK");return e && std::atoi(e)!=0;
}();

const bool g_giant_seed_device = [] {
    const char *e=std::getenv("NTT_GIANT_SEED_DEVICE");return e && std::atoi(e)!=0;
}();

const bool g_giant_seed_check = [] {
    const char *e=std::getenv("NTT_GIANT_SEED_CHECK");return e && std::atoi(e)!=0;
}();

GiantSeedStats g_giant_seed;

const bool g_gfinv_batch = [] {
    const char *e=std::getenv("NTT_GFINV_BATCH"); return e && std::atoi(e)!=0;
}();

GfinvStats g_gfinv;

int g_ladder_cap = 8192;

bool g_ntt_counts_enabled=false;

bool g_relocation_enabled=false;

long long g_s4_sample_limit = 96;

unsigned long long g_s4_check_every = 8;

bool g_s4_descent_trace = false;

bool g_s4_batched_progress = true;

unsigned long long g_defer_chunks = 0, g_defer_finishes = 0, g_defer_slices = 0;

unsigned long long g_defer_checked_chunks = 0, g_defer_max_group = 0;

double g_carry_group_readback = 0.0, g_carry_chunk_readback = 0.0;

unsigned long long *g_pin_raw[2][2] = {};

size_t g_pin_raw_cap[2][2] = {};

cudaEvent_t g_pin_raw_ev[2] = {};

bool g_pin_raw_pending[2] = {};

unsigned long long *g_pin_out[2] = {nullptr, nullptr};

size_t g_pin_out_cap[2] = {0, 0};

cudaEvent_t g_pin_ev[2] = {nullptr, nullptr};

const bool g_s4_defer_carry = !opt_off("NTT_S4_DEFER_CARRY");

const bool g_s4_async = !opt_off("NTT_S4_ASYNC");

const bool g_s4_pack_direct = !opt_off("NTT_S4_PACK_DIRECT");

const bool g_s4_flat_direct = !opt_off("NTT_S4_FLAT_DIRECT");

const bool g_s4_output_window = [] {
    const char *e=std::getenv("NTT_S4_OUTPUT_WINDOW");
    return e && *e && std::atoi(e)!=0;
}();

OutputWindowStats g_output_window;

const bool g_s4_final_readback = [] {
    const char *e=std::getenv("NTT_S4_FINAL_READBACK");
    return e && *e && std::atoi(e)!=0;
}();

FinalReadbackStats g_final_readback;

const bool g_s4_chunk_output = !opt_off("NTT_S4_CHUNK_OUTPUT");

ChunkOutputStats g_chunk_output;

const bool g_scaled_check = [] {
    const char *v=std::getenv("NTT_SCALED_CHECK");
    return v && std::atoi(v)!=0;
}();

const bool g_groot_device = [] {
    const char *v=std::getenv("NTT_GROOT_DEVICE");return v && std::atoi(v)!=0;
}();

const bool g_groot_device_check = [] {
    const char *v=std::getenv("NTT_GROOT_DEVICE_CHECK");return v && std::atoi(v)!=0;
}();

GDeviceStats g_gdevice;

const bool g_groot_leaf_staging = !opt_off("NTT_GROOT_LEAF_STAGING");

const bool g_groot_compact_raw = !opt_off("NTT_GROOT_COMPACT_RAW");

const size_t g_groot_leaf_chunk = [] {
    const char *e=std::getenv("NTT_GROOT_LEAF_CHUNK");return e?(size_t)std::strtoull(e,nullptr,10):0;
}();

GMemoryStats g_gmemory;

const bool g_s4_groot_only = !opt_off("NTT_S4_GROOT_ONLY");

GRootStats g_groot;

const bool g_s4_carry_batch = [] {
    const char *e = std::getenv("NTT_S4_CARRY_BATCH");
    return e && *e && std::atoi(e) != 0;
}();

const unsigned long long g_s4_chunk_max = [] {
    const char *e = std::getenv("NTT_S4_CHUNK_MAX");
    return e && *e ? std::strtoull(e, nullptr, 10) : 0ull;
}();

const bool g_s4_carry_test_bad = [] {
    const char *e = std::getenv("NTT_S4_CARRY_TEST_BAD");
    return e && *e && std::atoi(e) != 0;
}();

bool g_s4_carry_injected = false;

const bool g_s4_carry_trace = [] {
    const char *e = std::getenv("NTT_S4_CARRY_TRACE");
    return e && *e && std::atoi(e) != 0;
}();

unsigned long long g_carry_output_hash = 1469598103934665603ull, g_carry_output_words = 0;

unsigned long long g_pin_raw_waits = 0;

double g_pin_raw_wait = 0.0;

unsigned long long g_pin_raw_used = 0, g_pin_out_used = 0, g_pin_fallbacks = 0;

const bool g_s4_oracle_async = [] {
    const char *e = std::getenv("NTT_S4_ORACLE_ASYNC");
    return e && *e && std::atoi(e) != 0;
}();

const bool g_s4_oracle_pack = !opt_off("NTT_S4_ORACLE_PACK");

const int g_s4_oracle_ring = [] {
    const char *e = std::getenv("NTT_S4_ORACLE_RING");
    return e && *e ? std::max(1, std::min(8, std::atoi(e))) : 4;
}();

S4OracleSlot g_oracle_slot[8];

S4Reduce *g_oracle_owner = nullptr;

cudaEvent_t g_oracle_block_ev = nullptr;

S4OracleStats g_oracle;

std::vector<unsigned long long> g_div_hns;

unsigned long long g_div_recip = 0;

int g_div_shift = 0, g_div_nw = 0;

const bool g_s4_mersenne=[] {
    const char *e=std::getenv("NTT_S4_MERSENNE");return e && std::atoi(e)!=0;
}();

int g_device = 1;

FlatInputStats g_flat_input;

long long g_name_max = -1;

void ladder_cap_init(void)
{
    const char *e = std::getenv("NTT_LADDER_CAP");
    if (e && *e) g_ladder_cap = std::atoi(e);
    if (g_ladder_cap <= 0) g_ladder_cap = 8192;
    if (g_ladder_cap > (1 << 24)) g_ladder_cap = 1 << 24;   /* 2^24 threads = 2^18 blocks:
                                                               far inside the grid limit */
}

bool opt_off(const char *name)
{
    const char *e = std::getenv(name);
    return e && *e && std::atoi(e) == 0;
}

bool fold_device_flag(const char *key)
{
    const char *v=std::getenv(key);return v && std::atoi(v)!=0;
}

bool gscale_flag(const char *key)
{
    const char *v=std::getenv(key);if(!v)return false;
    if((v[0]!='0' && v[0]!='1') || v[1]){std::fprintf(stderr,"FATAL: %s must be 0 or 1\n",key);std::exit(3);}
    return v[0]=='1';
}

long long name_max(void)
{
    if (g_name_max < 0) {
        const char *e = std::getenv("NTT_NAME_MAX");
        g_name_max = (e && *e) ? std::atoll(e) : 0;
        if (g_name_max < 0) g_name_max = 0;
    }
    return g_name_max;
}

bool s3_compact_products_requested()
{
    const char *e=std::getenv("NTT_S3_COMPACT_PRODUCTS");return !e || std::atoi(e)!=0;
}

bool device_gleaf_flag(const char *name)
{
    const char *e=std::getenv(name);return e && std::atoi(e)!=0;
}

bool small_prime_flag(const char *name)
{
    const char *e=std::getenv(name);return e && std::atoi(e)!=0;
}

bool initial_memory_diagnostics_requested()
{
    for(const char *key:{"NTT_XADD6_TEST","NTT_S4_MERSENNE_TEST","NTT_BABY_DEVICE_TEST",
        "NTT_BABY_DEVICE_CHECK","NTT_BABY_DEVICE_ALLOC_FAIL","NTT_S4_FLAT_TEST",
        "NTT_S4_FINAL_READBACK_TEST","NTT_S4_OUTPUT_WINDOW_TEST","NTT_SCALED_TEST",
        "NTT_S3_PRODUCTS_TEST","NTT_FOLD_FLAT_TEST","NTT_FOLD_DEVICE_TEST",
        "NTT_GFINV_SEG_TEST","NTT_GFINV_BATCH_TEST","NTT_GROOT_DEVICE_TEST",
        "NTT_S4_GROOT_TEST","NTT_ARENA_WORKSPACE_TEST","NTT_FUSE_COOP_TEST",
        "NTT_FUSE_LIFETIME_TEST","NTT_GSCALE_DEVICE_CHECK"})
        if(fuse_env_ull(key,0))return true;
    return false;
}

int ecm_cuda_stage2_default_log_level()
{ return stage2_log::batches; }

int ecm_cuda_stage2_set_ntt_counts(bool enabled)
{ g_ntt_counts_enabled=enabled;return 0; }

int ecm_cuda_stage2_set_relocation(bool enabled)
{ g_relocation_enabled=enabled;return 0; }

int ecm_cuda_stage2_set_short_ntt_batches(const EcmStage2NttBatchSample *samples,size_t count)
{
    if((count && !samples) || count>65536)return 2;
    ecm_stage2::ShortNttBatchPolicy next;
    for(size_t i=0;i<count;++i) {
        const auto &s=samples[i];
        if(s.length<8 || s.length>(1ull<<19) || (s.length&(s.length-1)) ||
           !s.batch || s.batch>65535 || !std::isfinite(s.seconds) || s.seconds<=0)return 2;
        next.observe(s.length,s.batch,s.seconds);
    }
    g_short_ntt_batches=std::move(next);return 0;
}

int ecm_cuda_stage2_set_log_level(int level)
{
    if(level<stage2_log::quiet || level>stage2_log::debug)return 2;
    stage2_log::level=level;return 0;
}

int ecm_cuda_stage2_check_configuration()
{
    if(const char *v=std::getenv("NTT_GROOT_BALANCED"))if(std::strcmp(v,"0") && std::strcmp(v,"1")) {
        std::fprintf(stderr,"NTT_GROOT_BALANCED must be 0 or 1\n");return 2;
    }
    struct FixedOption { const char *key; const char *value; };
    if(std::getenv("NTT_FUSE_COOP_M")) {
        std::fprintf(stderr,"production Stage2 requires automatic NTT_FUSE_COOP_M; use -Engine development for radix comparisons\n");
        return 2;
    }
    const FixedOption options[] = {
        {"NTT_XADD6", "1"},
        {"NTT_POINT_MERSENNE", "1"},
        {"NTT_GIANT_SEED_DEVICE", "1"},
        {"NTT_GIANT_SEED_PAIR", "1"},
        {"NTT_GSCALE_DEVICE", "1"},
        {"NTT_GSCALE_DEVICE_TEST", "0"},
        {"NTT_SCALED_ROOT_DEVICE", "1"},
        {"NTT_SCALED_ROOT_TEST", "0"},
        {"NTT_SCALED_FRONTIER_DEVICE", "1"},
        {"NTT_SCALED_FRONTIER_TEST", "0"},
        {"NTT_GFINV_SEG_EXACT", "1"},
        {"NTT_GFINV_BATCH", "1"},
        {"NTT_SMALL_PRIME_REUSE", "1"},
        {"NTT_FOLD_FLAT", "1"},
        {"NTT_FOLD_DEVICE", "1"},
        {"NTT_FOLD_OWNER_REUSE", "3"},
        {"NTT_GROOT_DEVICE", "1"},
        {"NTT_SCALED_DESCENT", "1"},
        {"NTT_S4_OUTPUT_WINDOW", "1"},
        {"NTT_S4_CHUNK_OUTPUT", "1"},
        {"NTT_DEVICE_GLEAF", "1"},
        {"NTT_GROOT_TO_FOLD", "1"},
        {"NTT_S4_ORACLE_ASYNC", "1"},
        {"NTT_S4_CARRY_BATCH", "1"},
        {"NTT_FUSE_WARP_TAIL", "1"},
        {"NTT_BABY_DEVICE", "1"},
        {"NTT_S4_MERSENNE", "1"},
        {"NTT_S4_PACK_DIRECT", "1"},
        {"NTT_S4_FLAT_DIRECT", "1"},
        {"NTT_GROOT_COMPACT_RAW", "1"},
        {"NTT_GROOT_LEAF_STAGING", "1"},
        {"NTT_GL_SHORT_REDUCE", "1"},
        {"NTT_S4_OLDTAIL", "0"},
        {"NTT_S5_ON", "0"},
        {"NTT_S4_OFF", "0"},
        {"NTT_GIANT_BASE_CPU", "0"},
        {"NTT_CARRY_CHECK_FUSED", "0"},
        {"NTT_GL_SHIFT_SCALE", "0"},
        {"NTT_GIANT_CHAIN_SMALL_BLOCK", "0"},
        {"NTT_S4_FINAL_READBACK", "0"},
        {"NTT_S4_HOSTPACK", "0"},
        {"NTT_FUSE_COOP_OUTER", "2"},
        {"NTT_FUSE_T", "12"},
        {"NTT_FUSE_M", "4"},
        {"NTT_FUSE_COMPACT_SCRATCH", "1"},
        {"NTT_S4_GROOT_ONLY", "1"},
        {"NTT_S5_REDDUMP", "0"},
        {"NTT_GIANT_CHAIN_BLOCK", "64"},
        {"NTT_GIANT_CHAIN_MIN", "32768"},
    };
    for(const auto &option:options) {
        const char *value=std::getenv(option.key);
        if(value && std::strcmp(value,option.value)) {
            std::fprintf(stderr,"production Stage2 requires %s=%s; use -Engine development for algorithm comparisons\n",option.key,option.value);
            return 2;
        }
    }
    for(const char *key:{"NTT_S4_PACK_ZERO","NTT_FUSE_JOINT","NTT_HOST_TRACE","NTT_FUSE_MID_COOP","NTT_GSCALE_DEVICE_CHECK","NTT_GSCALE_DEVICE_TEST_BAD","NTT_SCALED_ROOT_CHECK","NTT_SCALED_ROOT_TEST_BAD","NTT_SCALED_FRONTIER_CHECK","NTT_SCALED_FRONTIER_TEST_BAD","NTT_SCALED_FRONTIER_ALLOC_FAIL"}) {
        const char *v=std::getenv(key);
        if(v && std::strcmp(v,"0") && std::strcmp(v,"1")) {
            std::fprintf(stderr,"production Stage2 requires %s=0 or 1\n",key);return 2;
        }
    }
#if !ECM_STAGE2_NVTX_AVAILABLE
    if(stage2_trace::enabled()) {std::fprintf(stderr,"NTT_HOST_TRACE requires CUDA NVTX3 headers\n");return 2;}
#endif
    for(const auto &pair:{std::pair<const char*,const char*>("NTT_GSCALE_DEVICE_TEST_BAD","NTT_GSCALE_DEVICE_CHECK"),
                         std::pair<const char*,const char*>("NTT_SCALED_ROOT_TEST_BAD","NTT_SCALED_ROOT_CHECK"),
                         std::pair<const char*,const char*>("NTT_SCALED_FRONTIER_TEST_BAD","NTT_SCALED_FRONTIER_CHECK")}) {
        const char *bad=std::getenv(pair.first),*check=std::getenv(pair.second);
        if(bad && !std::strcmp(bad,"1") && (!check || std::strcmp(check,"1"))) {
            std::fprintf(stderr,"production Stage2 requires %s=1 for %s\n",pair.second,pair.first);return 2;
        }
    }
    return 0;
}
