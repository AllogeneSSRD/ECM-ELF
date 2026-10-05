#pragma once
#include <cstdint>

// One normalized param0 Stage1 point per process. The callback is invoked only
// after Stage2 and its arithmetic checks have completed successfully.
int ecm_cuda_stage2_run(const char *n_hex, const char *x_hex, uint64_t sigma,
                       uint64_t b1, uint64_t b2, uint64_t d, int device,
                       void (*report)(const char *, void *), void *context);

// Planning initializes the CUDA context and queries live memory, but performs
// no curve arithmetic. All time/memory estimates are explicitly labelled.
int ecm_cuda_stage2_plan(const char *n_hex, uint64_t sigma, uint64_t b1,
                        uint64_t b2, uint64_t d, int device,
                        void (*report)(const char *, void *), void *context);
// One iteration is two forwards plus the fused product/scale/inverse.
// This measures field convolution only, not packing/carry/ECM reduction.
int ecm_cuda_stage2_tune_ntt(int device, int min_log2, int max_log2,
                            int repeats, uint64_t memory_bytes,
                            void (*report)(const char *, void *), void *context);
