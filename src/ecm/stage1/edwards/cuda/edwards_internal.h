#pragma once
#include "runtime.h"
#include "ecm/stage1/scalar.h"
namespace ecm_edwards_cuda_detail {
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
    EcmEdwardsPrepared prepared;
};
#define ECM_CUDA_TIER(W,T,G,M,D,A,MT,ET) int run_##W(const Request &request);
#include "../../../cuda/dispatch/tiers.def"
#undef ECM_CUDA_TIER
void prepare_device(int device);
}
