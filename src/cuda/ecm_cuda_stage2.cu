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
            "NTT_FUSE_WARP_TAIL", "NTT_XADD6", "NTT_D_MODEL", "NTT_GL_SHORT_REDUCE"};
        for (const char *key : keys) set_default(key, "1");
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
#include "../../tools/bench/stage2_tree_gpu.cu"
#include "../core/ecm_cuda_stage2.h"

int ecm_cuda_stage2_run(const char *n_hex, const char *x_hex, uint64_t sigma,
                       uint64_t b1, uint64_t b2, uint64_t d, int device,
                       void (*report)(const char *, void *), void *context)
{
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
