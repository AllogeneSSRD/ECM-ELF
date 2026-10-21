#include "host_policy.h"
#include <cuda_runtime_api.h>
#include "../../../service/planning/legacy/ecm_stage2_cost_profile.h"
#include "../../../platform/legacy/ecm_stage2_logging.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
namespace s2prod {
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
}
}
