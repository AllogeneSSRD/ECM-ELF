#include "host_policy.h"
#ifndef ECM_STAGE2_OPENCL
#include <cuda_runtime_api.h>
#endif
#include "../planning/cost_profile.h"
#include "../../platform/compute_log.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
namespace s2prod {
void configure_groot_experiment(Options &o) {
    const char *value=std::getenv("NTT_GROOT_BALANCED");if(!value)return;
#ifdef ECM_STAGE2_OPENCL
    throw std::runtime_error("NTT_GROOT_BALANCED is a CUDA-only fixed-shape experiment");
#else
    if(std::strcmp(value,"0") && std::strcmp(value,"1"))
        throw std::runtime_error("NTT_GROOT_BALANCED must be 0 or 1");
    if(!o.d || o.auto_b2 || o.auto_carrier || o.tune_runtime || o.short_calibration ||
       o.short_batch_tune || !o.tune.empty() || !o.cost_profile.empty())
        throw std::runtime_error("G-root layout experiments require fixed D, explicit B2, auto-carrier=0, tune-runtime=0, short-calibration=0 and short-ntt-batch-tune=0");
    // Older profile rows identify the legacy layout. Until layout is an
    // explicit profile shape field, never read or write them for this override.
    o.tune_profile.clear();
#endif
}
void configure_host_wait(Options &o,const HostWaitDefaults *defaults) {
    if(!o.portable_cli.count("--low-cpu-wait")) {
        if(!defaults || !defaults->present)return;
        o.low_cpu_wait=defaults->value;
    }
    const char *value=o.low_cpu_wait?"1":"0";
    // Native blocking scheduling covers the existing synchronous copies more
    // cheaply on the measured WDDM host. Poll only explicit completion events;
    // pre-waiting every readback remains an independent experimental control.
    if(_putenv_s("NTT_CUDA_WAIT_MODE",o.low_cpu_wait?"4":"0") ||
       _putenv_s("NTT_HOST_EVENT_WAIT",value) || _putenv_s("NTT_HOST_READBACK_WAIT","0"))
        throw std::runtime_error("cannot configure Stage2 host waiting");
}
void configure_pack_zero(Options &o,const PackZeroDefaults *s) {
    // Resolve once in the parent; every curve/tune child receives the value.
    // Legacy cprof has an exact PACK_ZERO=0 contract, including absent env.
    if(!o.portable_cli.count("--pack-zero")) {
        if(s && s->has_pack_zero)o.pack_zero=s->pack_zero;
        else if(const char *v=std::getenv("NTT_S4_PACK_ZERO")) {
            if(std::strcmp(v,"0") && std::strcmp(v,"1"))
                throw std::runtime_error("NTT_S4_PACK_ZERO must be 0 or 1");
            o.pack_zero=std::strcmp(v,"1")==0;
        } else if(!o.cost_profile.empty() || (s && !s->cost_profile.empty()))o.pack_zero=false;
    }
    if(_putenv_s("NTT_S4_PACK_ZERO",o.pack_zero?"1":"0"))
        throw std::runtime_error("cannot configure Stage2 pack/zero fusion");
}
void configure_cuda_wait(int device) {
#ifndef ECM_STAGE2_OPENCL
    const char *value=std::getenv("NTT_CUDA_WAIT_MODE");
    if(!value)return;
    const uint64_t mode=ecm_stage2::cost::integer(value);
    if(mode!=0&&mode!=1&&mode!=2&&mode!=4)
        throw std::runtime_error("NTT_CUDA_WAIT_MODE must be 0(auto), 1(spin), 2(yield), or 4(blocking)");
    auto check=[](cudaError_t error){if(error!=cudaSuccess)throw std::runtime_error(std::string("CUDA wait configuration failed: ")+cudaGetErrorString(error));};
    check(cudaSetDevice(device));
    unsigned before=0,after=0;check(cudaGetDeviceFlags(&before));
    // MapHost (bit 3) is implicit in CUDA 13's returned flags and must not be
    // passed to cudaSetDeviceFlags. Preserve all other non-schedule flags.
    check(cudaSetDeviceFlags((before&~15u)|static_cast<unsigned>(mode)));check(cudaGetDeviceFlags(&after));
    if((after&7u)!=mode||(after&~7u)!=(before&~7u))
        throw std::runtime_error("CUDA wait flags did not match requested context state");
    if(stage2_log::enabled(stage2_log::debug))std::cout<<"stage2_cuda_wait: device="<<device<<" requested="<<mode
             <<" before="<<before<<" after="<<after<<std::endl;
#else
    (void)device; // OpenCL uses checked blocking transfers/events on its queue.
#endif
}
}
