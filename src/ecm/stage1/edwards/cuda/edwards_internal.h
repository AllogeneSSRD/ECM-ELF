#pragma once
#include "runtime.h"
#include "ecm/stage1/scalar.h"
namespace ecm_edwards_cuda_detail {
struct DeviceAllocationError : std::runtime_error {
    using std::runtime_error::runtime_error;
};
struct Request {
    mpz_srcptr n,scalar;
    mpz_t *states;
    uint32_t count;
    uint32_t &done;
    uint32_t &snapshot_cursor;
    uint32_t chunk;
    const ecm_cuda_stage1::EdwardsSliceProgress &progress;
    void *context;
    const EcmEdwardsNaf *plan;
    int doubling;
    EcmEdwardsNafMetrics *metrics;
    const std::function<void()> &prepared;
    bool pipeline;
    unsigned cuda_slice_depth=0;
    bool cuda_low_cpu=false;
    std::function<bool()> checkpoint_due;
};
#define ECM_CUDA_TIER(W,T,G,M,D,A,MT,ET) int run_##W(const Request &request);
#include "../../../cuda/dispatch/tiers.def"
#undef ECM_CUDA_TIER
size_t prepare_device(int device);
}
