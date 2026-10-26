#include "../../src/ecm/stage1/montgomery/cuda/host_math.h"
#include "../../src/ecm/stage1/execution.h"
#include "../../src/ecm/stage1/sigma.h"
#include "../../src/ecm/stage1/edwards/cuda/runtime.h"
#include "../../src/ecm/stage1/edwards/cuda/edwards_internal.h"
#include "../../src/ecm/curve/edwards.h"
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <limits>

extern "C" void outputf(int,const char *,...) {}
extern "C" int test_verbose(int){return 0;}
namespace {
int notifications=0;
bool accept(uint32_t lane,mpz_srcptr factor,void *) {
    if(lane || mpz_cmp_ui(factor,1)<=0)throw std::runtime_error("factor callback");
    ++notifications;return true;
}
bool reject(uint32_t,mpz_srcptr,void *) {return false;}
void require(bool value,const char *message){if(!value)throw std::runtime_error(message);}

void sigma_contract() {
    using namespace ecm_stage1;
    require(sigma_batch_fits(UINT64_MAX, 1), "last u64 sigma rejected");
    require(!sigma_batch_fits(UINT64_MAX, 2), "u64 batch wrapped");
    require(!sigma_batch_fits(1, 0), "empty sigma batch accepted");
    require(montgomery_sigma_batch_fits(UINT32_MAX, 1, 3), "last PARAM3 sigma rejected");
    require(montgomery_sigma_batch_fits(UINT32_MAX - 7, 8, 3), "last PARAM3 batch rejected");
    require(!montgomery_sigma_batch_fits(UINT32_MAX - 6, 8, 3), "PARAM3 batch wrapped");
    require(!montgomery_sigma_batch_fits(1, 1, 2), "PARAM2 sigma=1 accepted");
    require(montgomery_sigma_batch_fits(2, UINT32_MAX - 1, 2), "PARAM2 full range rejected");
    require(!montgomery_sigma_batch_fits(2, UINT32_MAX, 2), "PARAM2 range overflow");
    require(!montgomery_sigma_batch_fits(0, 1, 3), "PARAM3 sigma=0 accepted");

    // Singleton ranges deterministically exercise both RNG endpoints. Small
    // random samples cannot reliably hit these unsigned-width boundaries.
    require(random_sigma32(1, UINT32_MAX) == UINT32_MAX, "u32 sampler lost its upper endpoint");
    require(random_sigma32(UINT32_MAX - 1, 2) == 2, "PARAM2 sampler lost its lower endpoint");
    require(random_sigma32(UINT32_MAX) == 1, "full PARAM3 interval changed");
    require(gpu_pick_random_sigma(0) == 2 && gpu_pick_random_sigma(UINT32_MAX) == 2,
            "historical C sampler ABI changed");
    bool rejected = false;
    try { (void)random_sigma32(0); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "empty sampler range accepted");
    rejected = false;
    try { (void)random_sigma32(UINT32_MAX, 2); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "invalid PARAM2 sampler interval accepted");

    // Restored and explicit identities must not consume the process sampler.
    std::srand(137);
    const int next = std::rand();
    std::srand(137);
    require(select_curve_sigma(UINT64_MAX, false, 0, 7) == UINT64_MAX,
            "restored curve identity replaced");
    require(select_curve_sigma(0, true, UINT64_MAX, 0) == UINT64_MAX,
            "explicit full-u64 identity narrowed");
    require(select_batch_sigma(UINT64_MAX, false, 0, 1, 0) == UINT64_MAX,
            "restored PARAM0 identity narrowed");
    require(select_batch_sigma(0, true, UINT32_MAX - 7, 8, 3) == UINT32_MAX - 7,
            "explicit PARAM3 batch changed");
    require(std::rand() == next, "restored/fixed sigma consumed random state");
    rejected = false;
    try { (void)select_curve_sigma(0, true, UINT64_MAX, 1); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "fixed curve index overflow accepted");
    rejected = false;
    try { (void)select_batch_sigma(0, true, 1, 1, 2); }
    catch (const std::invalid_argument&) { rejected = true; }
    require(rejected, "invalid explicit PARAM2 seed silently changed");
    require(select_batch_sigma(0, false, 0, UINT32_MAX - 1, 2) == 2,
            "random PARAM2 batch did not apply construction minimum");
    for (unsigned i = 0; i < 256; ++i) {
        const auto sigma = select_curve_sigma(0, false, 0, i);
        require(sigma > 5 && sigma < (uint64_t(1) << 53), "independent R53 range changed");
        require(montgomery_sigma_batch_fits(select_batch_sigma(0, false, 0, 8, 2), 8, 2),
                "random PARAM2 batch rejected by its own construction policy");
    }

    mpz_t n, d, product;
    mpz_inits(n, d, product, nullptr);
    mpz_set_ui(n, 101);
    gpu_compute_batch_d(d, UINT32_MAX, n);
    mpz_mul_2exp(product, d, 32);
    mpz_mod(product, product, n);
    require(mpz_cmp_ui(product, UINT32_MAX % 101) == 0, "historical batch_d ABI changed");
    mpz_clears(n, d, product, nullptr);
    std::puts("sigma_contract=PASS u32_endpoints=1 param2_minimum=1 u64_overflow=1 restored_no_draw=1 legacy_abi=1");
}
unsigned device_calls=0,dispatch_calls=0;bool reject_device=false;
void edwards_runtime_contract() {
    using namespace ecm_cuda_stage1;using namespace ecm_edwards;
    Integer n,s;mpz_ui_pow_ui(n.v,2,521);mpz_sub_ui(n.v,n.v,1);mpz_set_ui(s.v,12345);
    Array state(9);mpz_set_ui(state[6],1);mpz_set_ui(state[7],1);
    for(unsigned window:{0u,6u}) {
        uint32_t done=window?EcmEdwardsNaf(s.v,window).bits:uint32_t(mpz_sizeinbase(s.v,2));
        EcmEdwardsNafMetrics metrics;unsigned prepared=0;
        const auto result=execute_edwards_batch(n.v,s.v,state.data(),1,done,{999,window,4,0},nullptr,&prepared,&metrics,
            [](void *p){++*static_cast<unsigned *>(p);});
        require(result.termination==Termination::Completed && result.cursor==result.total && prepared==1,
            "completed Edwards checkpoint did not finalize on host");
        require(!device_calls && !dispatch_calls && !metrics.dictionary_bytes && !metrics.container_bits,
            "completed Edwards checkpoint touched a device/tier");
    }
    uint32_t done=0;std::string error;Failure failure;
    require(ecm_cuda_edwards_stage1_naf(n.v,s.v,state.data(),1,done,0,0,nullptr,nullptr,error,0,4,nullptr,nullptr,&failure)==-1 &&
        failure==Failure::Preparation && !device_calls,"NAF compatibility wrapper accepted window 0");
    done=uint32_t(mpz_sizeinbase(s.v,2));mpz_set_ui(state[7],0);
    require(execute_edwards_batch(n.v,s.v,state.data(),1,done,{0,0,4,0},nullptr,nullptr,nullptr,nullptr).failure==Failure::Preparation,
        "completed invalid Edwards point accepted");
    require(!device_calls,"invalid complete state prepared device");mpz_set_ui(state[7],1);
    mpz_ui_pow_ui(n.v,2,127);mpz_sub_ui(n.v,n.v,1);done=0;reject_device=true;
    require(execute_edwards_batch(n.v,s.v,state.data(),1,done,{1,0,4,17},nullptr,nullptr,nullptr,nullptr).failure==Failure::Device,
        "Edwards device error lost");
    reject_device=false;EcmEdwardsNafMetrics metrics;
    const auto stopped=execute_edwards_batch(n.v,s.v,state.data(),1,done,{1,6,5,17},nullptr,nullptr,&metrics,nullptr);
    require(stopped.termination==Termination::Stopped && stopped.cursor==1 && device_calls==2 && dispatch_calls==1 &&
        metrics.container_bits==384 && metrics.tpi==4,"resolved Edwards request/stop mismatch");
    std::puts("edwards_runtime_contract=PASS complete_binary_naf_no_gpu_or_tier=1 invalid_state=1 compatibility=1 device_failure=1 resolved_dispatch=1");
}
}
// Substitute only the hardware boundary. Validation and resolution above are
// the real ordinary-CXX implementation; no CUDA SDK is linked by this test.
void ecm_edwards_cuda_detail::prepare_device(int) {
    ++device_calls;if(reject_device)throw ecm_cuda_stage1::RunError(ecm_cuda_stage1::Failure::Device,"injected device failure");
}
int ecm_edwards_cuda_detail::run_384(const Request &request) {
    ++dispatch_calls;require(request.chunk==17 && request.plan && request.plan->window==6 && request.doubling==5,
        "resolved request differs from requested NAF options");request.done=1;return 2;
}
int main() {
    using namespace ecm_cuda_mont_host;
    using namespace ecm_cuda_stage1;
    try {
        edwards_runtime_contract();
        sigma_contract();
        {
            using Result=ecm_stage1::LegacyExecutionResult;using Kind=Result::CurveKind;
            mpz_t n,value;mpz_init_set_ui(n,77);mpz_init_set_ui(value,7);
            Result result;result.allocate_curves(3);
            result.set_curve(0,Kind::Factor,value,26);
            mpz_set_ui(value,12);result.set_curve(1,Kind::Stage2Point,value,27);
            result.set_curve(2,Kind::CompletedWithoutPoint,nullptr,28);result.validate_curves(n);
            Result moved(std::move(result));require(moved.curve_kinds[2]==Kind::CompletedWithoutPoint && moved.sigmas[2]==28,"curve outcome move lost data");
            moved.set_curve(1,Kind::Degenerate,nullptr,27);bool rejected=false;
            try{moved.validate_curves(n);}catch(const std::runtime_error &){rejected=true;}
            require(rejected && moved.array_found[1]==ECM_NO_FACTOR_FOUND,"degenerate result published as factor/point");
            moved.set_curve(1,Kind::Stage2Point,value,27);mpz_set_ui(value,77);moved.set_curve(0,Kind::Factor,value,26);
            rejected=false;try{moved.validate_curves(n);}catch(const std::runtime_error &){rejected=true;}
            require(rejected,"whole-modulus outcome accepted as a factor");mpz_clears(n,value,nullptr);
            std::puts("curve_outcomes=PASS owned_move=1 point=1 no_point=1 degenerate=1 proper_factor=1");
        }
        mpz_t prime,n;mpz_inits(prime,n,nullptr);mpz_ui_pow_ui(prime,2,127);mpz_sub_ui(prime,prime,1);
        for(int param:{0,2}) {
            ecm_stage1::LegacyExecutionResult result;result.allocate_curves(1);
            const uint32_t divisor=param?3:101,sigma=param?2:101;
            mpz_mul_ui(n,prime,divisor);size_t bytes=0;notifications=0;
            ConstructionResults sink{result.factors,result.array_found,accept,nullptr};
            auto *data=param?set_p_2p_param2(n,1,sigma,384,&bytes,&sink):set_p_2p_suyama(n,1,sigma,384,&bytes,&sink);
            require(bytes==7*384/8 && notifications==1,"construction callback count/layout");
            require(result.array_found[0]==ECM_FACTOR_FOUND_STEP1 && mpz_cmp_ui(result.factors[0],divisor)==0,"proper construction factor");
            require(process_results(result.factors,result.array_found,n,data,384,1,sigma,7,3,5,param,true)==ECM_FACTOR_FOUND_STEP1,"identity lane finalization");
            std::free(data);
            sink.publish=reject;
            bool rejected=false;
            try {data=param?set_p_2p_param2(n,1,sigma,384,&bytes,&sink):set_p_2p_suyama(n,1,sigma,384,&bytes,&sink);std::free(data);}
            catch(const RunError &error){rejected=error.failure==Failure::Publication;}
            require(rejected,"construction publication failure classification");
        }
        ecm_stage1::LegacyExecutionResult result;result.allocate_curves(1);
        ConstructionResults sink{result.factors,result.array_found};size_t bytes=0;
        mpz_set_ui(n,101);bool degenerate=false;
        try {std::free(set_p_2p_suyama(n,1,101,384,&bytes,&sink));}
        catch(const RunError &error){degenerate=error.failure==Failure::Preparation;}
        require(degenerate && result.array_found[0]==ECM_NO_FACTOR_FOUND,"whole-modulus gcd is not a factor");
        bool overflow=false;
        try {(void)checked_size({std::numeric_limits<size_t>::max(),2});}
        catch(const std::length_error &){overflow=true;}
        require(overflow,"allocation multiplication overflow");
        bool invalid=false;
        try {std::free(set_p_2p(n,1,1,31,&bytes));}
        catch(const std::invalid_argument &){invalid=true;}
        require(invalid,"bad container rejected before allocation");
        mpz_clears(prime,n,nullptr);
        std::puts("cuda_host_contract: GCD factors, degenerate, callback failure, identity lanes, allocation checks passed");
        return 0;
    }catch(const std::exception &error){std::fprintf(stderr,"FAIL: %s\n",error.what());return 1;}
}
