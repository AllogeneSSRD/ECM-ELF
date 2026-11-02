#include "host_policy.h"
#ifndef ECM_STAGE2_OPENCL
#include <cuda_runtime_api.h>
#endif
#include "../planning/cost_profile.h"
#include "../point_backend.h"
#include "../../platform/compute_log.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
namespace s2prod {
void apply_point_backends(const Options &o) {
#ifndef ECM_STAGE2_OPENCL
    if(_putenv_s("NTT_BABY_COOP",std::to_string(o.selected_baby_coop).c_str()) ||
       _putenv_s("NTT_GIANT_COOP",std::to_string(o.selected_giant_coop).c_str()) ||
       _putenv_s("NTT_GIANT_SEED_COOP",std::to_string(o.selected_seed_coop).c_str()) ||
       _putenv_s("NTT_S4_COOP_REDUCE",std::to_string(o.selected_reduce_coop).c_str()) ||
       _putenv_s("NTT_S4_COOP_TPB",std::to_string(o.reduce_threads).c_str()) ||
       _putenv_s("NTT_S4_COOP_KERNEL",std::to_string(o.selected_reduce_kernel?o.selected_reduce_kernel:(o.reduce_kernel?o.reduce_kernel:ecm_stage2::reduce_kernel_current)).c_str()))
        throw std::runtime_error("cannot configure Stage2 point backends");
#else
    (void)o;
#endif
}
void configure_point_backends(Options &o,const PointBackendDefaults *defaults) {
    const bool baby_cli=o.portable_cli.count("--baby-backend"),giant_cli=o.portable_cli.count("--giant-backend");
#ifndef ECM_STAGE2_OPENCL
    const auto overrides=ecm_cuda_stage2_point_environment_overrides();
    if(!baby_cli) {
        if(defaults && defaults->baby_present)o.baby_backend=defaults->baby;
        else if(overrides&1) {
            const std::string value=std::getenv("NTT_BABY_COOP");
            if(value!="0" && value!="32")throw std::runtime_error("NTT_BABY_COOP must be 0 or 32");
            o.baby_backend=value=="32"?"coop32":"scalar";
        }
    }
    if(!giant_cli) {
        if(defaults && defaults->giant_present)o.giant_backend=defaults->giant;
        else if(overrides&2) {
            const std::string value=std::getenv("NTT_GIANT_COOP");
            if(value!="0" && value!="16" && value!="32")throw std::runtime_error("NTT_GIANT_COOP must be 0, 16 or 32");
            o.giant_backend=ecm_stage2::point_backend_name(unsigned(std::stoul(value)));
        }
    }
    if(!o.portable_cli.count("--seed-backend")) {
        if(defaults && defaults->seed_present)o.seed_backend=defaults->seed;
        else if(overrides&4) {
            const std::string value=std::getenv("NTT_GIANT_SEED_COOP");
            if(value!="0" && value!="32")throw std::runtime_error("NTT_GIANT_SEED_COOP must be 0 or 32");
            o.seed_backend=value=="32"?"coop32":"scalar";
        }
    }

#endif
    const auto baby=ecm_stage2::baby_backend(o.baby_backend);
    o.selected_baby_coop=baby<0?0:unsigned(baby);
    const auto giant=ecm_stage2::giant_backend(o.giant_backend);
    o.selected_giant_coop=giant<0?0:unsigned(giant);
    const auto seed=ecm_stage2::seed_backend(o.seed_backend);
    o.selected_seed_coop=seed<0?0:unsigned(seed);
#ifdef ECM_STAGE2_OPENCL
    if(o.selected_baby_coop || o.selected_giant_coop || o.selected_seed_coop)throw std::runtime_error("cooperative point backends are CUDA-only");
#endif
    apply_point_backends(o);
}
void configure_s4_backend(Options &o,bool present,const std::string &value) {
    if(!o.portable_cli.count("--reduce-backend")) {
        if(present)o.reduce_backend=value;
        else if(const char *env=std::getenv("NTT_S4_COOP_REDUCE")) {
            if(std::strcmp(env,"0") && std::strcmp(env,"32"))throw std::runtime_error("NTT_S4_COOP_REDUCE must be 0 or 32");
            o.reduce_backend=std::strcmp(env,"32")==0?"coop32":"scalar";
        }
    }
    const auto mode=ecm_stage2::reduce_backend(o.reduce_backend);o.selected_reduce_coop=mode<0?0:unsigned(mode);
    const auto setting=[&](const char *cli,const char *env,unsigned &target) {
        if(!o.portable_cli.count(cli))if(const char *text=std::getenv(env)){const auto n=ecm_stage2::cost::integer(text);if(n>128)throw std::runtime_error("invalid S4 variant override");target=unsigned(n);}
    };
    setting("--reduce-threads","NTT_S4_COOP_TPB",o.reduce_threads);
    setting("--reduce-kernel","NTT_S4_COOP_KERNEL",o.reduce_kernel);
    if(!ecm_stage2::valid_reduce_variant(o.reduce_threads,o.reduce_kernel?o.reduce_kernel:ecm_stage2::reduce_kernel_current))throw std::runtime_error("S4 threads must be 64, 96 or 128 and kernel 1 or 2");
    o.selected_reduce_threads=o.selected_reduce_coop?o.reduce_threads:0;o.selected_reduce_kernel=o.selected_reduce_coop?(o.reduce_kernel?o.reduce_kernel:ecm_stage2::reduce_kernel_current):0;
#ifdef ECM_STAGE2_OPENCL
    if(o.selected_reduce_coop)throw std::runtime_error("cooperative S4 reduction is CUDA-only");
#endif
}
void configure_giant_policy(Options &o) {
    if(o.giant_chain!="64" && o.giant_chain!="32" && o.giant_chain!="auto")
        throw std::runtime_error("giant-chain must be 64, 32 or auto");
    if(o.b2_round_chain && o.relocation)
        throw std::runtime_error("B2 chain rounding requires the complete giant interval (relocation=0)");
#ifdef ECM_STAGE2_OPENCL
    if(o.giant_tail_balance || o.b2_round_chain || o.giant_chain!="64")
        throw std::runtime_error("giant point policies are CUDA-only");
#else
    _putenv_s("NTT_GIANT_TAIL_BALANCE",o.giant_tail_balance?"1":"0");
    _putenv_s("NTT_GIANT_CHAIN_BLOCK",o.giant_chain=="32"?"32":"64");
    o.selected_chain=o.giant_chain=="32"?32:64;
#endif
}
void configure_groot_experiment(Options &o) {
    const char *value=std::getenv("NTT_GROOT_BALANCED");
#ifdef ECM_STAGE2_OPENCL
    if(value || o.groot_layout>=0)throw std::runtime_error("G-root layout selection is CUDA-only");
#else
    if(!value)return;
    if(std::strcmp(value,"0") && std::strcmp(value,"1"))
        throw std::runtime_error("NTT_GROOT_BALANCED must be 0 or 1");
    if(o.groot_layout>=0 && o.groot_layout!=int(value[0]-'0'))
        throw std::runtime_error("conflicting G-root layout overrides");
    o.groot_layout=int(value[0]-'0');
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
