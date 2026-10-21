#pragma once

#include "context.h"

void ladder_cap_init(void);

bool opt_off(const char *name);

bool fold_device_flag(const char *key);

bool gscale_flag(const char *key);

long long name_max(void);

bool s3_compact_products_requested();

bool device_gleaf_flag(const char *name);

bool small_prime_flag(const char *name);

bool initial_memory_diagnostics_requested();

extern ecm_stage2::ShortNttBatchPolicy g_short_ntt_batches;

extern const bool g_fold_flat;

extern const bool g_gfinv_seg_exact;

extern const bool g_gfinv_seg_check;

extern const bool g_giant_seed_device;

extern const bool g_giant_seed_check;

extern const bool g_gfinv_batch;

extern int g_ladder_cap;

extern bool g_ntt_counts_enabled;

extern bool g_relocation_enabled;

extern long long g_s4_sample_limit;

extern unsigned long long g_s4_check_every;

extern bool g_s4_descent_trace;

extern bool g_s4_batched_progress;

extern unsigned long long g_defer_chunks, g_defer_finishes, g_defer_slices;

extern unsigned long long g_defer_checked_chunks, g_defer_max_group;

extern double g_carry_group_readback, g_carry_chunk_readback;

extern unsigned long long *g_pin_raw[2][2];

extern size_t g_pin_raw_cap[2][2];

extern cudaEvent_t g_pin_raw_ev[2];

extern bool g_pin_raw_pending[2];

extern unsigned long long *g_pin_out[2];

extern size_t g_pin_out_cap[2];

extern cudaEvent_t g_pin_ev[2];

extern const bool g_s4_defer_carry;

extern const bool g_s4_async;

extern const bool g_s4_pack_direct;

extern const bool g_s4_flat_direct;

extern const bool g_s4_output_window;

extern const bool g_s4_final_readback;

extern const bool g_s4_chunk_output;

extern const bool g_scaled_check;

extern const bool g_groot_device;

extern const bool g_groot_device_check;

extern const bool g_groot_leaf_staging;

extern const bool g_groot_compact_raw;

extern const size_t g_groot_leaf_chunk;

extern const bool g_s4_groot_only;

extern const bool g_s4_carry_batch;

extern const unsigned long long g_s4_chunk_max;

extern const bool g_s4_carry_test_bad;

extern bool g_s4_carry_injected;

extern const bool g_s4_carry_trace;

extern unsigned long long g_carry_output_hash, g_carry_output_words;

extern unsigned long long g_pin_raw_waits;

extern double g_pin_raw_wait;

extern unsigned long long g_pin_raw_used, g_pin_out_used, g_pin_fallbacks;

extern const bool g_s4_oracle_async;

extern const bool g_s4_oracle_pack;

extern const int g_s4_oracle_ring;

extern cudaEvent_t g_oracle_block_ev;

extern std::vector<unsigned long long> g_div_hns;

extern unsigned long long g_div_recip;

extern int g_div_shift, g_div_nw;

extern const bool g_s4_mersenne;

extern int g_device;

extern long long g_name_max;
