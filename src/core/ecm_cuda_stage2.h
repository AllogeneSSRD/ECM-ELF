#pragma once
#include <cstdint>

// Production defaults to batch progress (3); development keeps full diagnostics
// (4). Lower levels never disable mandatory arithmetic/error checks.
int ecm_cuda_stage2_default_log_level();
int ecm_cuda_stage2_set_log_level(int level);
int ecm_cuda_stage2_check_configuration();

// One normalized param0 Stage1 point per process. The callback is invoked only
// after Stage2 and its arithmetic checks have completed successfully.
// carrier_exponent=0 keeps arithmetic modulo saved N. An explicit p requires
// N | (2^p-1); arithmetic uses that carrier, inverses/GCDs still target saved N.
// Optional structured benchmark evidence, populated after the final oracle drain.
// Phase timers overlap; total_seconds is the ranking boundary, not their sum.
struct EcmStage2Metrics {
    uint64_t d=0,p=0,giant_points=0,selftest_cases=0,checked=0,bad=0;
    double total_seconds=0,init_seconds=0,main_seconds=0;
    double giant_seconds=0,gtrees_seconds=0,fold_seconds=0,descent_seconds=0;
    double inverse_seconds=0,accum_seconds=0;
    bool clean=false,fold_resident=false,frontier_resident=false;
};
int ecm_cuda_stage2_run(const char *n_hex, const char *x_hex, uint64_t sigma,
                       uint64_t b1, uint64_t b2, uint64_t d, int device,
                       void (*report)(const char *, void *), void *context,
                       unsigned carrier_exponent=0,EcmStage2Metrics *metrics=nullptr);

// Planning initializes the CUDA context and queries live memory, but performs
// no curve arithmetic. All time/memory estimates are explicitly labelled.
int ecm_cuda_stage2_plan(const char *n_hex, uint64_t sigma, uint64_t b1,
                        uint64_t b2, uint64_t d, int device,
                        void (*report)(const char *, void *), void *context,
                        unsigned carrier_exponent=0);
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
