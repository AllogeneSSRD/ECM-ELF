#include "ecm/cuda/arithmetic/cgbn/types.cuh"
#include <cstdlib>
#include <cstdio>
#include <stdexcept>
#include <type_traits>

// Substitute only CUDA deallocation at this private support seam. Production
// ownership and error helpers are compiled unchanged; no GPU/context is used.
namespace fixture {
int device_live=0,report_live=0,event_live=0;
cudaError_t release(void *p){--device_live;std::free(p);return cudaSuccess;}
cudaError_t release_report(cgbn_error_report_t *p){--report_live;delete p;return cudaSuccess;}
cudaError_t release_event(cudaEvent_t p){--event_live;delete reinterpret_cast<int *>(p);return cudaSuccess;}
}
#define cudaFree fixture::release
#define cgbn_error_report_free fixture::release_report
#define cudaEventDestroy fixture::release_event
#include "ecm/stage1/montgomery/cuda/resources.cuh"
#undef cudaFree
#undef cgbn_error_report_free
#undef cudaEventDestroy

int main() {
    using namespace ecm_cuda_stage1;
    static_assert(!std::is_copy_constructible_v<MontResources>);
    try {
        for(int fail=0;fail<=6;++fail) {
            try {
                MontResources owned;
                for(int step=0;step<6;++step) {
                    if(step==fail)throw std::runtime_error("injected allocation failure");
                    if(step<2){auto *p=static_cast<uint32_t *>(std::malloc(64));++fixture::device_live;
                        if(!step)owned.device_scalar=p;else owned.device_data=p;}
                    else if(step==2){owned.report=new cgbn_error_report_t{};++fixture::report_live;}
                    else {auto p=reinterpret_cast<cudaEvent_t>(new int);++fixture::event_live;
                        if(step==3)owned.start=p;else if(step==4)owned.batch=p;else owned.stop=p;}
                }
            }catch(const std::runtime_error &){}
            if(fixture::device_live || fixture::report_live || fixture::event_live)
                throw std::runtime_error("partial construction leaked resources");
        }
        bool cuda_failed=false,cgbn_failed=false;
        try {check_cuda(cudaErrorInvalidValue,"injected CUDA error");}
        catch(const RunError &error){cuda_failed=error.failure==Failure::Device;}
        cgbn_error_report_t report{};report._error=cgbn_division_by_zero_error;report._instance=0xFFFFFFFF;
        try {check_cgbn(&report,__FILE__,__LINE__);}
        catch(const RunError &error){cgbn_failed=error.failure==Failure::Device;}
        if(!cuda_failed || !cgbn_failed)throw std::runtime_error("runtime error escaped typed boundary");
        std::puts("cuda_resources: 7 lifetimes and CUDA/CGBN propagation passed; gpu_runs=0");return 0;
    }catch(const std::exception &error){std::fprintf(stderr,"FAIL: %s\n",error.what());return 1;}
}
