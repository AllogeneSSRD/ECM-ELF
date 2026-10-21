#pragma once

// ============================================================================
// ECM runtime configuration - single source of truth.
//
// All formerly-custom environment variables (getenv) are consolidated here.
// Each executable's main() fills this struct from command-line arguments;
// deep / cross-platform shared code (cgbn_stage1_opencl, impl_opencl,
// debug_utils, opencl_ecm_log, ...) only READS it and never calls getenv.
//
// Defaults match the old "environment variable not set" behavior, so removing
// getenv does not change semantics. Android has no CLI: its JNI entry writes
// the fields directly; unset fields keep the defaults below.
//
// Note: standard system variables such as LOGNAME / USERNAME are NOT part of
// this struct (they are not program-custom) and are still read as usual.
// ============================================================================

#include <cstdint>
#include <string>

struct EcmRuntimeConfig {
    // --- device / kernel build (main solver) ---
    int device_index = 0;               // was CGBN_OPENCL_DEVICE_INDEX   / CLI: -d
    uint32_t tpi = 8u;                  // was ECM_OPENCL_TPI             / CLI: --tpi (1..32)
    int stage1_force_normalize = 1;     // was ECM_STAGE1_FORCE_NORMALIZE / CLI: --force-normalize
    int add_mod_fused_unroll = 2;       // was ECM_MP_ADD_MOD_FUSED_UNROLL/ CLI: --fused-unroll (1|2)

    // --- OpenCL backend (impl_opencl.cpp) ---
    std::string kernel_root;            // was ECM_KERNEL_ROOT / CGBN_KERNEL_ROOT / CLI: --kernel-root ("" = built-in default)
    std::string cache_dir;              // was CGBN_OPENCL_CACHE_DIR        / CLI: --cache-dir ("" = default)
    bool cache_disable = false;         // was CGBN_OPENCL_CACHE_DISABLE    / CLI: --no-kernel-cache
    bool cache_verbose = false;         // was CGBN_OPENCL_CACHE_VERBOSE    / CLI: --cache-verbose
    bool compile_verbose = false;       // was CGBN_OPENCL_COMPILE_VERBOSE  / CLI: --compile-verbose

    // --- logging / debug / verification (main solver) ---
    bool log_timestamp = true;          // was ECM_LOG_TIMESTAMP (default ON) / CLI: --no-log-timestamp
    bool gpu_dump = false;              // was ECM_GPU_DUMP             / CLI: --gpu-dump
    std::string gpu_dump_file = "dump.csv";               // was ECM_GPU_DUMP_FILE   / CLI: --gpu-dump-file
    bool profile_ops = false;           // was ECM_PROFILE_OPS          / CLI: --profile-ops
    std::string profile_ops_file = "ecm_ops_profile.csv"; // was ECM_PROFILE_OPS_FILE / CLI: --profile-ops-file
    bool sync_each_batch = false;       // was ECM_SYNC_EACH_BATCH      / CLI: --sync-each-batch
    bool verify_gpu_results = false;    // was ECM_VERIFY_GPU_RESULTS   / CLI: --verify-gpu
    bool verify_gpu_strict = false;     // was ECM_VERIFY_GPU_STRICT    / CLI: --verify-gpu-strict

    // --- external tools ---
    std::string gp_bin;                 // was ECM_GP_BIN / PARI_GP_BIN  / CLI: --gp ("" = default "gp")

    // --- bench / diagnostic tools ---
    bool addsub_asm_disable = false;    // was ECM_ADDSUB_ASM_DISABLE    / CLI: --no-asm
    bool addsub_asm_b64 = false;        // was ECM_ADDSUB_ASM_B64        / CLI: --asm-b64
    std::string bench_csv;              // was ECM_BENCH_CSV             / CLI: --csv
    int mont_wg_impl = -1;              // was ECM_MONT_WG_IMPL          / CLI: --wg-impl (-1 = unset)
    int mont_wg_impl4_unroll = -1;      // was ECM_MONT_WG_IMPL4_UNROLL  / CLI: --wg-impl4-unroll (-1 = unset)
    bool gpu_sliced = false;            // CLI: --sliced  → use sliced shuffle kernel (PoC, 32T)
    bool gpu_sliced_t16 = false;        // CLI: --sliced-t16 → 16 lanes × 2 limbs (PoC, lower VGPR)
    bool gpu_local = false;             // CLI: --local  → LDS-based kernel (avoid scratch spill at large bits)
    int wg_size = 0;                    // CLI: --wg <N>  → explicit work-group size (0=auto, 1,4,8,16,32)
};

// Process-wide mutable singleton. Each main() writes it after parsing argv; the
// rest of the code only reads it.
EcmRuntimeConfig &ecm_runtime_config();
