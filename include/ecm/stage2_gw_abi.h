#pragma once
#include <stdint.h>

/* Windows x64 ABI v1. All buffers belong to the caller. Each integer has
 * words unsigned 32-bit limbs, least significant first, zero padded. */
typedef struct {
    uint64_t d, poly_size, fft_length, blocks, giants, polymults, checked_values;
    double init_seconds, main_seconds, max_roundoff, estimated_memory_mib;
    double point_seconds, tree_seconds, fold_seconds, descent_seconds;
    char fft_description[256], polymult_isa[16];
} ecm_gw_stage2_stats;

typedef struct {
    uint32_t words, threads, check;
    uint64_t b1, b2, d, exponent, memory_mib;
    const uint32_t *n, *x, *a24;
    int (*stop)(void);
} ecm_gw_stage2_request;

typedef struct {
    uint32_t abi, size;
    const char *version, *build, *cpu;
    /* 0 = complete, 1 = factor, -1 = failure (including stop). */
    int (*run)(const ecm_gw_stage2_request*, uint32_t* factor,
               ecm_gw_stage2_stats*, char* error, uint32_t error_size);
    int (*selftest)(uint32_t threads, char* error, uint32_t error_size);
} ecm_gw_stage2_api;
typedef const ecm_gw_stage2_api* (*ecm_gw_stage2_get_api_fn)(uint32_t);

/* ABI 2 adds complete-curve phase observations and allocation preflight.
 * The ABI 1 layout and get_api(1) remain available to old workers. */
typedef struct {
    double baby_seconds, ftree_seconds, reciprocal_seconds, giant_seconds;
    double gtree_seconds, fold_seconds, scale_seconds, descent_seconds, leaf_gcd_seconds;
} ecm_gw_stage2_phases;

typedef struct {
    uint64_t d, poly_size, fft_length;
} ecm_gw_stage2_tune_candidate;

typedef struct {
    ecm_gw_stage2_api base;
    int (*run_with_phases)(const ecm_gw_stage2_request*, uint32_t* factor,
                           ecm_gw_stage2_stats*, ecm_gw_stage2_phases*,
                           char* error, uint32_t error_size);
    /* Uses n only. Returns the number of candidates written, or -1 on error. */
    int (*tune_candidates)(const ecm_gw_stage2_request*, ecm_gw_stage2_tune_candidate*,
                           uint32_t capacity, char* error, uint32_t error_size);
} ecm_gw_stage2_api_v2;

typedef enum {
    ECM_GW_PLAN, ECM_GW_BABY, ECM_GW_FTREE, ECM_GW_INVERSE,
    ECM_GW_GIANT, ECM_GW_BATCH, ECM_GW_DESCENT, ECM_GW_GCD, ECM_GW_REPLAY
} ecm_gw_stage2_progress_kind;

typedef struct {
    uint32_t kind;
    uint64_t batch, batches, d, poly_size;
    double elapsed;
} ecm_gw_stage2_progress;

/* Called synchronously on the calling thread. The event is valid only during
 * the callback, which must not throw across the DLL boundary. */
typedef void (*ecm_gw_stage2_progress_fn)(void*, const ecm_gw_stage2_progress*);

typedef struct {
    ecm_gw_stage2_api_v2 base;
    int (*run_with_progress)(const ecm_gw_stage2_request*, uint32_t* factor,
                             ecm_gw_stage2_stats*, ecm_gw_stage2_phases*,
                             ecm_gw_stage2_progress_fn, void*, char* error, uint32_t error_size);
} ecm_gw_stage2_api_v3;

/* ABI 4 exposes the admitted coefficient FFT and F-tree memory plan. */
typedef struct {
    uint64_t d, poly_size, fft_length, fft_exponent, max_exponent;
    uint32_t f_tree_splits;
    double planned_mib;
    char fft_layout[256], polymult_isa[16];
} ecm_gw_stage2_tune_plan;

typedef struct {
    ecm_gw_stage2_api_v3 base;
    int (*tune_plans)(const ecm_gw_stage2_request*, ecm_gw_stage2_tune_plan*,
                      uint32_t capacity, char* error, uint32_t error_size);
} ecm_gw_stage2_api_v4;

/* ABI 5 lets runtime tuning execute the exact F-tree layout it tested. */
typedef struct {
    ecm_gw_stage2_request base;
    int32_t f_tree_splits; /* -1 automatic, -2 segmented, 0 full, >0 fixed splits. */
    uint32_t tune_candidates, dense_d;
} ecm_gw_stage2_request_v5;

typedef struct {
    ecm_gw_stage2_api_v4 base;
    int (*run_plan)(const ecm_gw_stage2_request_v5*, uint32_t* factor,
                    ecm_gw_stage2_stats*, ecm_gw_stage2_phases*,
                    ecm_gw_stage2_progress_fn, void*, char* error, uint32_t error_size);
    int (*search_plans)(const ecm_gw_stage2_request_v5*, ecm_gw_stage2_tune_plan*,
                        uint32_t capacity, char* error, uint32_t error_size);
} ecm_gw_stage2_api_v5;
