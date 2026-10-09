#include <cstdlib>

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
            "NTT_FUSE_WARP_TAIL", "NTT_XADD6", "NTT_D_MODEL", "NTT_BABY_DEVICE"};
        for (const char *key : keys) set_default(key, "1");
#if defined(NTT_GL_FIXED_MODE) && NTT_GL_FIXED_MODE >= 0
        set_default("NTT_GL_SHORT_REDUCE", (NTT_GL_FIXED_MODE&1) ? "1" : "0");
#else
        set_default("NTT_GL_SHORT_REDUCE", "1");
#endif
#if defined(NTT_GL_FIXED_MODE) && NTT_GL_FIXED_MODE == 3
        set_default("NTT_POINT_MERSENNE", "1");
#endif
        set_default("NTT_FUSE_COOP_OUTER", "2");
        set_default("NTT_DEVICE_GLEAF_MAX_MB", "512");
        set_default("NTT_FOLD_DEVICE_MAX_MB", "640");
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

// Reuse the experimental engine verbatim, including its exact NTT and checks.
#define STAGE2_TREE_GPU_NO_MAIN
#include "stage2_tree_gpu.cu"
#include "../../src/core/ecm_cuda_stage2.h"
#include "../../src/cuda/ecm_stage2_tune.cuh"

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

int ecm_cuda_stage2_tune_ntt(int device,int min_log2,int max_log2,int repeats,
                            uint64_t memory,void (*report)(const char*,void*),void *context)
{
    return stage2_tune::run(device,min_log2,max_log2,repeats,memory,report,context);
}

int ecm_cuda_stage2_plan(const char *n_hex,uint64_t sigma,uint64_t b1,uint64_t b2,
                        uint64_t d,int device,void (*report)(const char*,void*),void *context,unsigned carrier_exponent)
{
    if(carrier_exponent){std::fprintf(stderr,"Mersenne carrier requires the production engine source\n");return 2;}
    if(!report || !n_hex || device<0 || !b1 || b2<=b1 ||
       b2>(uint64_t)INT64_MAX-8192 || (d && (d<6 || d%2)))return 2;
    g_device=device;
    ecm_stage2::Plan p;
    const int code=run_real(n_hex,true,sigma,b1,b2,d,d==0,false,1,nullptr,nullptr,true,&p);
    if(code)return code;
    const auto &g=p.geometry;
    std::ostringstream json;json<<std::setprecision(17)
        <<"{\"type\":\"stage2_plan\",\"schema\":1,\"curves_executed\":0,\"bits\":"<<g.bits
        <<",\"words\":"<<g.words<<",\"B1\":"<<p.b1<<",\"B2\":"<<p.b2<<",\"D\":"<<p.d
        <<",\"P\":"<<g.p<<",\"I\":"<<p.giant_points<<",\"G\":"<<p.batches
        <<",\"fold_length\":"<<g.fold_length<<",\"tree_length\":"<<g.tree_length
        <<",\"fold_big_bytes\":"<<g.fold_big_bytes<<",\"arena_estimate_bytes\":"<<g.arena_estimate_bytes
        <<",\"owner_bytes\":"<<g.fold_owner_bytes<<",\"owner_reuse\":"<<fold_owner_reuse()<<",\"baby_payload_bytes\":"<<p.baby_bytes
        <<",\"free_bytes\":"<<p.free_bytes<<",\"arena_cap_bytes\":"<<p.arena_cap_bytes
        <<",\"owner_budget_bytes\":"<<p.owner_budget_bytes
        <<",\"owner_budget_fits\":"<<(p.owner_budget_fits ? "true" : "false")
        <<",\"arena_estimate_fits\":"<<(p.arena_estimate_fits ? "true" : "false")
        <<",\"residency_guaranteed\":false,\"process_peak_estimated\":false,\"accounting_version\":2"
        <<",\"model\":"<<stage2_tune::quote(p.model)<<",\"calibrated\":"<<(p.calibrated ? "true" : "false")
        <<",\"stage2_seconds_estimate\":"<<p.estimated_seconds<<"}";
    report(json.str().c_str(),context);return 0;
}

int ecm_cuda_stage2_run(const char *n_hex, const char *x_hex, uint64_t sigma,
                       uint64_t b1, uint64_t b2, uint64_t d, int device,
                       void (*report)(const char *, void *), void *context,unsigned carrier_exponent,EcmStage2Metrics *metrics)
{
    if(metrics){std::fprintf(stderr,"Structured ECM tune requires the production engine\n");return 2;}
    if(carrier_exponent){std::fprintf(stderr,"Mersenne carrier requires the production engine source\n");return 2;}
    s2g_install_crash_handler();
    s2g_install_terminate();
    ladder_cap_init();
    g_device = device;
    const char *progress = std::getenv("NTT_NO_PROGRESS");
    if (progress && std::atoi(progress)) g_s4_batched_progress = false;
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    Stage2Tail tail;
    const int code = run_real(n_hex, true, sigma, b1, b2, d, d == 0,
                              false, 1, x_hex, &tail);
    if (code || tail.bad_factors) return code ? code : 1;
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

int ecm_cuda_stage2_default_log_level() { return 4; }
int ecm_cuda_stage2_set_log_level(int level) { return level == 4 ? 0 : 2; }
int ecm_cuda_stage2_check_configuration() {
    if(fold_owner_reuse_option()<0) {
        std::fprintf(stderr,"NTT_FOLD_OWNER_REUSE must be 0..3\n");return 2;
    }
    return 0;
}
