#include "runtime.h"
#include "edwards_internal.h"
#include "ecm/core/curve/ecm_edwards_curve.h"
#include <memory>

namespace {
struct Tier {uint32_t bits,tpi;int (*execute)(const ecm_edwards_cuda_detail::Request &);};
const Tier *select_tier(size_t bits) {
    static const Tier tiers[]={
#define ECM_CUDA_TIER(W,T,G,M,D,A,MT,ET) {W,T,ecm_edwards_cuda_detail::run_##W},
#include "../../dispatch/tiers.def"
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
ecm_cuda_stage1::RunResult ecm_cuda_stage1::execute_edwards_batch(mpz_srcptr n,mpz_srcptr scalar,
    mpz_t *states,uint32_t count,uint32_t &done,const EdwardsOptions &options,
    EcmEdwardsProgress progress,void *context,EcmEdwardsNafMetrics *metrics,EcmEdwardsPrepared prepared) {
    RunResult result;result.cursor=done;result.snapshot_cursor=done;
    try {
        if(metrics)*metrics={};
        if(!n || !scalar || !states || !count || options.device<0 || mpz_cmp_ui(n,3)<=0 ||
           !mpz_odd_p(n) || mpz_sgn(scalar)<=0 || mpz_sizeinbase(scalar,2)>UINT32_MAX ||
           (options.window && (options.doubling!=4 && options.doubling!=5)))
            throw std::invalid_argument("invalid GPU Edwards request");
        std::unique_ptr<EcmEdwardsNaf> plan;
        if(options.window)plan=std::make_unique<EcmEdwardsNaf>(scalar,options.window);
        const auto total=plan?plan->bits:uint32_t(mpz_sizeinbase(scalar,2));result.total=total;
        if(done>total)throw std::invalid_argument("GPU Edwards progress exceeds scalar columns");
        const auto values=checked_size({size_t(count),9});
        for(size_t i=0;i<values;++i)
            if(mpz_sgn(states[i])<0 || mpz_cmp(states[i],n)>=0)
                throw std::invalid_argument("noncanonical GPU Edwards state");
        if(metrics){metrics->columns=total;metrics->additions=plan?plan->additions:0;}
        if(done==total) {
            for(uint32_t i=0;i<count;++i)if(!ecm_edwards::point_valid(n,states[size_t(i)*9],states+size_t(i)*9+5))
                throw std::invalid_argument("invalid completed GPU Edwards point");
            if(prepared)prepared(context);
            result.code=0;result.termination=Termination::Completed;result.failure=Failure::None;return result;
        }
        const auto *tier=select_tier(mpz_sizeinbase(n,2));
        if(!tier)throw std::invalid_argument("no compiled GPU Edwards container with one spare bit for target");
        const auto point_bytes=checked_size({size_t(count),size_t(tier->bits/8)});
        checked_size({point_bytes,10});
        const auto dictionary_bytes=plan?checked_size({point_bytes,size_t(1u<<(options.window-2)),4}):0;
        if(metrics){metrics->container_bits=tier->bits;metrics->tpi=tier->tpi;metrics->dictionary_bytes=dictionary_bytes;}
        ecm_edwards_cuda_detail::prepare_device(options.device);
        result.failure=Failure::Execution;
        result.code=tier->execute({n,scalar,states,count,done,options.chunk,progress,context,
            plan.get(),options.doubling,metrics,prepared});
        result.cursor=done;result.snapshot_cursor=done;
        if(result.code==0){result.termination=Termination::Completed;result.failure=Failure::None;}
        else if(result.code==2){result.termination=Termination::Stopped;result.failure=Failure::None;}
    }catch(const RunError &e){result.failure=e.failure;result.error=e.what();result.cursor=done;result.snapshot_cursor=done;}
    catch(const std::exception &e){result.error=e.what();result.cursor=done;result.snapshot_cursor=done;}
    return result;
}

// Compatibility entry points used by standalone arithmetic probes.
int ecm_cuda_edwards_stage1(const mpz_t n,const mpz_t scalar,mpz_t *states,
    uint32_t count,uint32_t &done,uint32_t chunk,int device,EcmEdwardsProgress progress,
    void *context,std::string &error,EcmEdwardsPrepared prepared,ecm_cuda_stage1::Failure *failure) {
    const auto result=ecm_cuda_stage1::execute_edwards_batch(n,scalar,states,count,done,
        {device,0,4,chunk},progress,context,nullptr,prepared);
    error=result.error;if(failure)*failure=result.failure;return result.code;
}
int ecm_cuda_edwards_stage1_naf(const mpz_t n,const mpz_t scalar,mpz_t *states,
    uint32_t count,uint32_t &done,uint32_t chunk,int device,EcmEdwardsProgress progress,
    void *context,std::string &error,unsigned window,int doubling,EcmEdwardsNafMetrics *metrics,
    EcmEdwardsPrepared prepared,ecm_cuda_stage1::Failure *failure) {
    if(window<2 || window>12) {
        error="GPU Edwards NAF window must be 2..12";
        if(failure)*failure=ecm_cuda_stage1::Failure::Preparation;
        return -1;
    }
    const auto result=ecm_cuda_stage1::execute_edwards_batch(n,scalar,states,count,done,
        {device,window,doubling,chunk},progress,context,metrics,prepared);
    error=result.error;if(failure)*failure=result.failure;return result.code;
}
