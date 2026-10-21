#include "../../src/ecm/backend/cuda/stage1/montgomery/host_math.h"
#include "../../src/ecm/service/stage1/execution_contract.h"
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
}
int main() {
    using namespace ecm_cuda_mont_host;
    using namespace ecm_cuda_stage1;
    try {
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
