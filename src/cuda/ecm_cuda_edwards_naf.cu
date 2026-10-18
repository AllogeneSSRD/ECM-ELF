#include "ecm_cuda_edwards_internal.h"
#include <cuda_runtime.h>
#include <stdexcept>

int ecm_cuda_edwards_stage1_naf(const mpz_t n,const mpz_t scalar,mpz_t *states,
        uint32_t count,uint32_t &done,uint32_t chunk,int device,
        EcmEdwardsProgress progress,void *context,std::string &error,
        unsigned window,int doubling,EcmEdwardsNafMetrics *metrics,EcmEdwardsPrepared prepared) {
    try {
        if(!states || !count || device<0 || mpz_cmp_ui(n,3)<=0 || !mpz_odd_p(n) ||
           (doubling!=4 && doubling!=5))throw std::runtime_error("invalid GPU Edwards NAF state/algorithm");
        EcmEdwardsNaf plan(scalar,window);
        if(done>plan.bits)throw std::runtime_error("GPU Edwards progress exceeds NAF columns");
        for(size_t i=0;i<size_t(count)*9;++i)
            if(mpz_sgn(states[i])<0 || mpz_cmp(states[i],n)>=0)
                throw std::runtime_error("noncanonical GPU Edwards NAF state");
        ecm_edwards_cuda_detail::prepare_device(device);
        return ecm_edwards_cuda_detail::dispatch({n,scalar,states,count,done,chunk,progress,context,&plan,doubling,metrics,prepared});
    }catch(const std::exception &e){error=e.what();return -1;}
}
