#include "runtime.h"
#include "edwards_internal.h"
#include "memory.h"
#include "ecm/curve/edwards.h"
#include <memory>

namespace {
struct Tier {uint32_t bits,tpi;int (*execute)(const ecm_edwards_cuda_detail::Request &);};
const Tier *select_tier(size_t bits) {
    static const Tier tiers[]={
#define ECM_CUDA_TIER(W,T,G,M,D,A,MT,ET) {W,T,ecm_edwards_cuda_detail::run_##W},
#include "../../../cuda/dispatch/tiers.def"
#undef ECM_CUDA_TIER
        {0,0,nullptr}
    };
    for(const auto &tier:tiers)if(tier.bits && bits<tier.bits)return &tier;
    return nullptr;
}
ecm_cuda_stage1::EdwardsSliceProgress legacy_progress(EcmEdwardsProgress callback, void *context) {
    if (!callback) return {};
    return [callback, context](uint32_t done, const ecm_cuda_stage1::EdwardsSnapshot &snapshot) {
        snapshot();
        return callback(done, context);
    };
}

}
bool ecm_cuda_edwards_tier(size_t bits,uint32_t &container,uint32_t &tpi) {
    const auto *tier=select_tier(bits);container=tier?tier->bits:0;tpi=tier?tier->tpi:0;
    return tier!=nullptr;
}
ecm_cuda_stage1::RunResult ecm_cuda_stage1::execute_edwards_batch(mpz_srcptr n,mpz_srcptr scalar,
    mpz_t *states,uint32_t count,uint32_t &done,const EdwardsOptions &options,
    const EdwardsSliceProgress &progress,void *context,EcmEdwardsNafMetrics *metrics,EcmEdwardsPrepared prepared) {
    uint32_t snapshot_cursor=done;
    RunResult result;result.cursor=done;result.snapshot_cursor=done;
    try {
        if(metrics)*metrics={};
        if(!n || !scalar || !states || !count || options.device<0 || mpz_cmp_ui(n,3)<=0 ||
           !mpz_odd_p(n) || mpz_sgn(scalar)<=0 || mpz_sizeinbase(scalar,2)>UINT32_MAX ||
           (options.window && (options.window<2 || options.window>12 ||
                               (options.doubling!=4 && options.doubling!=5))))
            throw std::invalid_argument("invalid GPU Edwards request");
        const auto scalar_bits = uint32_t(mpz_sizeinbase(scalar, 2));
        const auto values=checked_size({size_t(count),9});
        for(size_t i=0;i<values;++i)
            if(mpz_sgn(states[i])<0 || mpz_cmp(states[i],n)>=0)
                throw std::invalid_argument("noncanonical GPU Edwards state");

        const auto budget_host = edwards_host_budget();
        const bool can_lower = !done && options.allow_memory_downgrade;
        const bool can_use_binary = can_lower && options.allow_binary_fallback;
        unsigned window = options.window;
        std::unique_ptr<EcmEdwardsNaf> plan;
        if (done) {
            if (window) {
                if (edwards_runtime_host_bytes(0, 0, scalar_bits, true) > budget_host)
                    throw std::runtime_error("insufficient host memory to validate the GPU Edwards NAF cursor");
                plan = std::make_unique<EcmEdwardsNaf>(scalar, window);
            }
            const auto total = plan ? plan->bits : scalar_bits;
            result.total = total;
            if (done > total)
                throw std::invalid_argument("GPU Edwards progress exceeds scalar columns");
            if (done == total) {
                for(uint32_t i=0;i<count;++i)
                    if(!ecm_edwards::point_valid(n,states[size_t(i)*9],states+size_t(i)*9+5))
                        throw std::invalid_argument("invalid completed GPU Edwards point");
                if (metrics) {
                    metrics->columns = total;
                    metrics->additions = plan ? plan->additions : 0;
                }
                if (options.memory_selection) options.memory_selection(window, window, 0, 0);
                if (prepared) prepared(context);
                result.code=0;result.termination=Termination::Completed;result.failure=Failure::None;return result;
            }
        }

        const auto *tier = select_tier(mpz_sizeinbase(n,2));
        if(!tier)throw std::invalid_argument("no compiled GPU Edwards container with one spare bit for target");
        if (window && can_use_binary && edwards_runtime_host_bytes(tier->bits, count, scalar_bits, true) > budget_host)
            window = 0;
        if (const auto required = edwards_runtime_host_bytes(tier->bits, count, scalar_bits, window != 0);
            required > budget_host)
            throw std::runtime_error("insufficient host memory for GPU Edwards preparation (required " +
                                     std::to_string(required) + " bytes, usable " + std::to_string(budget_host) + " bytes)");

        for (;;) {
            const auto free = ecm_edwards_cuda_detail::prepare_device(options.device);
            using namespace ecm_edwards_memory;
            const auto budget = size_t(cap_budget(configured_budget(edwards_device_budget(free), options.memory_limit_mb),
                                                  "ECM_EDWARDS_GPU_MEMORY_MIB"));
            auto selected = edwards_device_window(tier->bits, count, scalar_bits, window, can_lower, budget);
            if (!selected && window && can_use_binary)
                selected = edwards_device_window(tier->bits, count, scalar_bits, 0, false, budget);
            if (!selected) {
                const auto required = edwards_device_bytes(tier->bits, count, scalar_bits,
                                                           !can_lower || !window ? window : can_use_binary ? 0 : 2);
                throw std::runtime_error("insufficient device memory for GPU Edwards (required " +
                    std::to_string(required) + " bytes, usable " + std::to_string(budget) + " bytes)" +
                    (done ? "; checkpoint window retained" : "; use a smaller curve batch"));
            }
            window = *selected;
            if (!window)
                plan.reset();
            else if (!plan || plan->window != window)
                plan = std::make_unique<EcmEdwardsNaf>(scalar, window);
            const auto total = plan ? plan->bits : scalar_bits;
            result.total = total;
            const auto required = edwards_device_bytes(tier->bits, count, scalar_bits, window);
            const auto point_bytes = checked_size({size_t(count), size_t(tier->bits / 8)});
            if (metrics) {
                *metrics = {};
                metrics->columns = total;
                metrics->additions = plan ? plan->additions : 0;
                metrics->container_bits = tier->bits;
                metrics->tpi = tier->tpi;
                metrics->dictionary_bytes = window ? checked_size({point_bytes, size_t(1) << (window - 2), 4}) : 0;
            }
            bool started = false;
            const std::function<void()> ready = [&] {
                started = true;
                if (options.memory_selection) options.memory_selection(options.window, window, required, budget);
                if (prepared) prepared(context);
            };
            try {
                result.code = tier->execute({n,scalar,states,count,done,snapshot_cursor,options.chunk,progress,context,
                    plan.get(),options.doubling,metrics,ready,options.pipeline,
                    options.cuda_slice_depth,options.cuda_low_cpu,options.checkpoint_due});
                break;
            } catch (const ecm_edwards_cuda_detail::DeviceAllocationError&) {
                if (started || done || !can_lower || !window || (window == 2 && !can_use_binary))
                    throw RunError(Failure::Device, "CUDA Edwards allocation failed after memory preflight; retry when more device memory is free");
                window = window > 2 ? window - 1 : 0;
                plan.reset();
            }
        }
        result.failure=Failure::Execution;
        if(result.code==0){result.termination=Termination::Completed;result.failure=Failure::None;}
        else if(result.code==2){result.termination=Termination::Stopped;result.failure=Failure::None;}
    }catch(const RunError &e){result.failure=e.failure;result.error=e.what();}
    catch(const std::exception &e){result.error=e.what();}
    result.cursor=done;result.snapshot_cursor=snapshot_cursor;
    return result;
}

// Compatibility entry points used by standalone arithmetic probes.
int ecm_cuda_edwards_stage1(const mpz_t n,const mpz_t scalar,mpz_t *states,
    uint32_t count,uint32_t &done,uint32_t chunk,int device,EcmEdwardsProgress progress,
    void *context,std::string &error,EcmEdwardsPrepared prepared,ecm_cuda_stage1::Failure *failure) {
    const auto result=ecm_cuda_stage1::execute_edwards_batch(n,scalar,states,count,done,
        {device,0,4,chunk},legacy_progress(progress,context),context,nullptr,prepared);
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
        {device,window,doubling,chunk},legacy_progress(progress,context),context,metrics,prepared);
    error=result.error;if(failure)*failure=result.failure;return result.code;
}
