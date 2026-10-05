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

// Cost planning queries the same integer packing backend. No CUDA allocation or
// curve arithmetic occurs in shape_query; device_info initializes a CUDA context.
bool ecm_cuda_stage2_shape_query(uint64_t coefficients, int bits,
                                uint64_t *length, uint64_t *output_slots);
struct EcmStage2DeviceInfo {
    uint64_t free_bytes=0,total_bytes=0;
    int major=0,minor=0,runtime=0,driver=0,fixed_mode=-1,outer_unroll_u=0;
    char uuid_hex[33]{};
};
int ecm_cuda_stage2_device_info(int device,EcmStage2DeviceInfo *info,const char *expected_uuid=nullptr);
