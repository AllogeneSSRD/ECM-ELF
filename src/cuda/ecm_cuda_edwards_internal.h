#pragma once
#include "ecm_cuda_edwards_naf.h"
#include "../core/ecm_edwards_naf.h"
namespace ecm_edwards_cuda_detail {
struct Request {
    mpz_srcptr n,scalar;
    mpz_t *states;
    uint32_t count;
    uint32_t &done;
    uint32_t chunk;
    EcmEdwardsProgress progress;
    void *context;
    const EcmEdwardsNaf *plan;
    int doubling;
    EcmEdwardsNafMetrics *metrics;
    EcmEdwardsPrepared prepared;
};
#define ECM_EDWARDS_TIER(W,T) int run_##W(const Request &request);
#include "ecm_cuda_edwards_tiers.inc"
#undef ECM_EDWARDS_TIER
int dispatch(const Request &request);
}
