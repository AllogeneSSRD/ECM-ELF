#include <gmp.h>
#include <cgbn/cgbn.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>
#include "ecm_cuda_edwards_naf.h"
#include "../core/ecm_edwards_naf.h"
#include "../../kernels/cuda/cgbn_edwards_point.cuh"
#include "../../kernels/cuda/cgbn_edwards_naf.cuh"

namespace {
void check(cudaError_t rc){if(rc!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(rc));}
template<class T> struct Device {
    T *p=nullptr;
    explicit Device(size_t n){check(cudaMalloc(&p,n*sizeof(T)));}
    ~Device(){if(p)cudaFree(p);}
};
struct Report {
    cgbn_error_report_t *p=nullptr;
    Report(){check(cgbn_error_report_alloc(&p));}
    ~Report(){if(p)cgbn_error_report_free(p);}
};
template<uint32_t W,uint32_t T> struct Params {
    static const uint32_t BITS=W,TPI=T,TPB=128,MAX_ROTATION=1,SHM_LIMIT=0;
    static const bool CONSTANT_TIME=false;
};
template<class Mem> void put(Mem &r,const mpz_t x){
    std::memset(&r,0,sizeof(r));size_t used=0;mpz_export(r._limbs,&used,-1,4,0,0,x);
}
#include "ecm_cuda_edwards_naf.inl"
}

int ecm_cuda_edwards_stage1_naf(const mpz_t n,const mpz_t scalar,mpz_t *states,
        uint32_t count,uint32_t &done,uint32_t chunk,int device,
        EcmEdwardsProgress progress,void *context,std::string &error,
        unsigned window,int doubling,EcmEdwardsNafMetrics *metrics) {
    try {
        if(!states || !count || !chunk || device<0 || mpz_cmp_ui(n,3)<=0 || !mpz_odd_p(n) ||
           (doubling!=4 && doubling!=5))throw std::runtime_error("invalid GPU Edwards NAF state/algorithm");
        EcmEdwardsNaf plan(scalar,window);
        if(done>plan.bits)throw std::runtime_error("GPU Edwards progress exceeds NAF columns");
        for(size_t i=0;i<size_t(count)*9;++i)
            if(mpz_sgn(states[i])<0 || mpz_cmp(states[i],n)>=0)
                throw std::runtime_error("noncanonical GPU Edwards NAF state");
        check(cudaSetDevice(device));const size_t bits=mpz_sizeinbase(n,2);
#define ED_NAF_TIER(W,T) if(bits<W)return run_naf<Params<W,T>>(n,plan,states,count,done,chunk,progress,context,doubling,metrics)
        ED_NAF_TIER(256,4);ED_NAF_TIER(512,4);ED_NAF_TIER(1024,8);ED_NAF_TIER(2048,16);
        ED_NAF_TIER(4096,16);ED_NAF_TIER(8192,32);ED_NAF_TIER(17408,32);
#undef ED_NAF_TIER
        throw std::runtime_error("GPU Edwards NAF supports targets up to 17407 bits");
    }catch(const std::exception &e){error=e.what();return -1;}
}
