#include "edwards_internal.h"
#include <cuda_runtime.h>
#include <stdexcept>

namespace {
struct Tier {uint32_t bits,tpi;int (*execute)(const ecm_edwards_cuda_detail::Request &);};
const Tier *select_tier(size_t bits) {
    static const Tier tiers[]={
#define ECM_CUDA_TIER(W,T,G,M,D,A,MT,ET) {W,T,ecm_edwards_cuda_detail::run_##W},
#include "ecm/cuda/tiers.def"
#undef ECM_CUDA_TIER
        {0,0,nullptr}
    };
    for(const auto &tier:tiers)if(tier.bits && bits<tier.bits)return &tier;
    return nullptr;
}
}
bool ecm_cuda_edwards_tier(size_t bits,uint32_t &container,uint32_t &tpi) {
    const auto *tier=select_tier(bits);container=tier?tier->bits:0;tpi=tier?tier->tpi:0;
    return tier!=nullptr;
}
int ecm_edwards_cuda_detail::dispatch(const Request &r) {
    const auto *tier=select_tier(mpz_sizeinbase(r.n,2));
    if(!tier)throw std::runtime_error("no compiled GPU Edwards container with one spare bit for target");
    return tier->execute(r);
}
void ecm_edwards_cuda_detail::prepare_device(int device) {
    auto check=[](cudaError_t rc){if(rc!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(rc));};
    check(cudaSetDevice(device));
    unsigned flags=0;check(cudaGetDeviceFlags(&flags));
    check(cudaSetDeviceFlags((flags&~unsigned(cudaDeviceScheduleMask))|cudaDeviceScheduleBlockingSync));
}
int ecm_cuda_edwards_stage1(const mpz_t n,const mpz_t s,mpz_t *states,
        uint32_t count,uint32_t &done,uint32_t chunk,int device,
        EcmEdwardsProgress cb,void *context,std::string &error,EcmEdwardsPrepared prepared) {
    try {
        if(!states || !count || device<0 || mpz_cmp_ui(n,3)<=0 || !mpz_odd_p(n) ||
           mpz_sgn(s)<=0 || mpz_sizeinbase(s,2)>UINT32_MAX || done>mpz_sizeinbase(s,2))
            throw std::runtime_error("invalid GPU Edwards scalar state");
        for(size_t i=0;i<size_t(count)*9;++i)
            if(mpz_sgn(states[i])<0 || mpz_cmp(states[i],n)>=0)throw std::runtime_error("noncanonical GPU Edwards state");
        ecm_edwards_cuda_detail::prepare_device(device);
        return ecm_edwards_cuda_detail::dispatch({n,s,states,count,done,chunk,cb,context,nullptr,4,nullptr,prepared});
    }catch(const std::exception &e){error=e.what();return -1;}
}
