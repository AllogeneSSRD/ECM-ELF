#pragma once
// Host-only diagnostic contract. The adapter compiles the real Stage1 host
// loop and kernels as separate modules; the probe supplies only timed SDK hooks.
#include "naf.h"
#include <cuda_runtime.h>
struct EcmEdwardsNaf;
namespace ecm_cuda_diagnostics {
struct HostWaitHooks {
    cudaError_t (*event_create)(cudaEvent_t *);
    cudaError_t (*event_record)(cudaEvent_t,cudaStream_t);
    cudaError_t (*event_sync)(cudaEvent_t);
    cudaError_t (*device_sync)();
    cudaError_t (*memcpy)(void *,const void *,size_t,cudaMemcpyKind);
    cudaError_t (*last_error)();
    cudaError_t (*elapsed)(float *,cudaEvent_t,cudaEvent_t);
};
int host_wait_run(const HostWaitHooks &hooks,mpz_srcptr n,mpz_srcptr scalar,
    mpz_t *states,uint32_t count,uint32_t &done,uint32_t chunk,
    const EcmEdwardsNaf &plan,int doubling,EcmEdwardsNafMetrics *metrics);
}
