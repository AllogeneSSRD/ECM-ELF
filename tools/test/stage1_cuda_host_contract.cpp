#include "../../src/ecm/backend/cuda/stage1/montgomery/host_math.h"
#include "../../src/ecm/service/stage1/execution_contract.h"
#include "../../src/ecm/backend/cuda/stage1/edwards/runtime.h"
#include "../../src/ecm/backend/cuda/stage1/edwards/edwards_internal.h"
#include "../../src/ecm/core/curve/ecm_edwards_curve.h"
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
        {
            using Result=ecm_stage1::ExecutionResult;using Kind=Result::CurveKind;
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
            ecm_stage1::ExecutionResult result;result.allocate_curves(1);
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
        ecm_stage1::ExecutionResult result;result.allocate_curves(1);
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
