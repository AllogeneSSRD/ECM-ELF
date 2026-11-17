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
