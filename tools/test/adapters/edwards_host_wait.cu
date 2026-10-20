// Explicit diagnostic adapter: compile the unchanged production host template.
// SDK substitutions stay within this TU. Production targets never link it.
#include "ecm/cuda/diagnostics/host_wait.h"
namespace {
const ecm_cuda_diagnostics::HostWaitHooks *active_hooks=nullptr;
cudaError_t hook_create(cudaEvent_t *event){return active_hooks->event_create(event);}
cudaError_t hook_record(cudaEvent_t event,cudaStream_t stream=0){return active_hooks->event_record(event,stream);}
cudaError_t hook_event_sync(cudaEvent_t event){return active_hooks->event_sync(event);}
cudaError_t hook_device_sync(){return active_hooks->device_sync();}
cudaError_t hook_memcpy(void *dst,const void *src,size_t bytes,cudaMemcpyKind kind){return active_hooks->memcpy(dst,src,bytes,kind);}
cudaError_t hook_last_error(){return active_hooks->last_error();}
cudaError_t hook_elapsed(float *ms,cudaEvent_t start,cudaEvent_t stop){return active_hooks->elapsed(ms,start,stop);}
}
#define cudaEventCreate hook_create
#define cudaEventRecord hook_record
#define cudaEventSynchronize hook_event_sync
#define cudaDeviceSynchronize hook_device_sync
#define cudaMemcpy hook_memcpy
#define cudaGetLastError hook_last_error
#define cudaEventElapsedTime hook_elapsed
#include "../../../src/cuda/stage1/edwards_runtime.inl"
#undef cudaEventCreate
#undef cudaEventRecord
#undef cudaEventSynchronize
#undef cudaDeviceSynchronize
#undef cudaMemcpy
#undef cudaGetLastError
#undef cudaEventElapsedTime

#define ECM_DIAG_JOIN_(A,B) A##B
#define ECM_DIAG_JOIN(A,B) ECM_DIAG_JOIN_(A,B)
int ecm_cuda_diagnostics::host_wait_run(const HostWaitHooks &hooks,mpz_srcptr n,mpz_srcptr scalar,
    mpz_t *states,uint32_t count,uint32_t &done,uint32_t chunk,
    const EcmEdwardsNaf &plan,int doubling,EcmEdwardsNafMetrics *metrics) {
    struct Restore {
        const HostWaitHooks *previous;
        ~Restore(){active_hooks=previous;}
    } restore{active_hooks};
    active_hooks=&hooks;
    return ecm_edwards_cuda_detail::ECM_DIAG_JOIN(run_,ECM_HOST_WAIT_TIER)(
        {n,scalar,states,count,done,chunk,nullptr,nullptr,&plan,doubling,metrics,nullptr});
}
