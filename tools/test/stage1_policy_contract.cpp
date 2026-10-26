#include "ecm/montgomery_stage1/cuda/host_policy.h"
#include "ecm/backend/cuda/runtime/cuda_ecm_shim.h"
#include "io/exponent_cache.h"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>
// The frozen pre-refactor oracle reads a global. Production receives this
// directory explicitly from the normalized request.
const std::string &ecm_exp_cache_get_dir();
#include "baseline_policy.inc"

namespace host=ecm_cuda_mont_host;
static std::string diagnostic;
extern "C" void outputf(int,const char *format,...){
    char buffer[2048];va_list args;va_start(args,format);vsnprintf(buffer,sizeof(buffer),format,args);va_end(args);
    diagnostic+=buffer;
}
const std::string &ecm_exp_cache_get_dir(){static const std::string value="fixture-cache";return value;}
static void env(const char *key,const char *value){if(_putenv_s(key,value)!=0)throw std::runtime_error("fixture environment");}
template<class F>static std::string observe(F f){
    diagnostic.clear();
    try{const auto value=f();return "OK:"+value+"|"+diagnostic;}catch(const std::exception &e){return std::string("ERROR:")+e.what()+"|"+diagnostic;}
}
static size_t checked=0;
template<class A,class B>static void same(A a,B b){
    const auto expected=observe(a),actual=observe(b);
    if(expected!=actual)throw std::runtime_error("policy mismatch: "+expected+" != "+actual);
    ++checked;
}
static std::string describe(const host::PracPolicy &p){return std::to_string(p.mode)+","+std::to_string(p.requested_tpi)+","+
    std::to_string(p.compact)+std::to_string(p.outlined)+std::to_string(p.single_add)+std::to_string(p.single_compact);}
static std::string describe(const host::PracWindow &w){std::ostringstream out;out<<w.position<<','<<w.first<<','<<w.count<<','<<w.chunk<<','<<w.warmup<<','<<w.seconds<<','<<w.dump;return out.str();}
int main(){
    try{
        for(auto algorithm:{"","ladder","resident","prac","bad"})for(auto position:{"","prefix","bad"})for(auto variant:{"","baseline","compact","bad"}){
            env("ECM_GPU_STAGE1_ALGO",algorithm);env("ECM_PRAC_WINDOW",position);env("ECM_PRAC_VARIANT",variant);
            same([]{const auto *s=baseline::stage1_algorithm();return s?std::string(s):"invalid";},[]{const auto request=host::capture_mont_request(nullptr,"fixture-cache");const auto *s=host::stage1_algorithm(request);return s?std::string(s):"invalid";});
        }
        for(bool prac:{false,true})for(auto cap:{"","0","128","168","255","bad"})for(auto variant:{"","baseline","compact","outline-add","single-add","single-compact","bad"})for(auto tpi:{"","0","16","32","bad"}){
            env("ECM_PRAC_REG_TARGET",cap);env("ECM_PRAC_VARIANT",variant);env("ECM_STAGE1_TPI",tpi);
            same([=]{return describe(baseline::read_prac_policy(prac));},[=]{return describe(host::read_prac_policy(host::capture_mont_request(nullptr,"fixture-cache"),prac));});
        }
        mpz_t n;mpz_init_set_ui(n,10403);
        for(auto constant:{"","none","runtime","np0","m4423","bad"})for(bool prac:{false,true})for(int original:{2,3,5,13})for(uint32_t bits:{384u,4608u})for(uint32_t tpi:{0u,32u}){
            env("ECM_PRAC_CONSTANTS",constant);
            same([&]{int mode=original;const std::string s=baseline::prac_constants_policy(prac,mode,bits,tpi,n,14);return s+":"+std::to_string(mode);},
                 [&]{int mode=original;const std::string s=host::prac_constants_policy(host::capture_mont_request(nullptr,"fixture-cache"),prac,mode,bits,tpi,n,14);return s+":"+std::to_string(mode);});
        }
        for(int kind=0;kind<2;++kind){
            mpz_set_ui(n,1);mpz_mul_2exp(n,n,4423);mpz_sub_ui(n,n,kind?3:1);env("ECM_PRAC_CONSTANTS","m4423");
            same([&]{int mode=13;const std::string s=baseline::prac_constants_policy(true,mode,4608,0,n,4423);return s+":"+std::to_string(mode);},
                 [&]{int mode=13;const std::string s=host::prac_constants_policy(host::capture_mont_request(nullptr,"fixture-cache"),true,mode,4608,0,n,4423);return s+":"+std::to_string(mode);});
        }
        mpz_clear(n);
        for(auto key:{"ECM_PRAC_WINDOW_COUNT","ECM_PRAC_WINDOW_CHUNK","ECM_PRAC_WINDOW_WARMUP","ECM_PRAC_WINDOW_DUMP","ECM_GPU_STAGE1_SAMPLE_SECONDS"}){
            for(auto value:{"","0","1","16","32","33","-1","nan","inf","18446744073709551616","600","601","1x"}){
                for(auto reset:{"ECM_PRAC_WINDOW_COUNT","ECM_PRAC_WINDOW_CHUNK","ECM_PRAC_WINDOW_WARMUP","ECM_PRAC_WINDOW_DUMP","ECM_GPU_STAGE1_SAMPLE_SECONDS"})env(reset,"");
                env(key,value);
                for(auto position:{"","prefix","middle","tail","bad"})for(bool prac:{false,true})for(uint64_t records:{0ull,3ull,100ull}){
                    env("ECM_PRAC_WINDOW",position);
                    same([=]{return describe(baseline::prac_window_settings(prac,records));},[=]{return describe(host::prac_window_settings(host::capture_mont_request(nullptr,"fixture-cache"),prac,records));});
                }
            }
        }
        for(auto value:{"","0","10","100","500","501","nan","inf","1x","-1"}){
            env("ECM_PRAC_TARGET_MS",value);env("ECM_GPU_STAGE1_SAMPLE_SECONDS",value);
            for(bool prac:{false,true})same([=]{return std::to_string(baseline::prac_target_ms(prac));},[=]{return std::to_string(host::prac_target_ms(host::capture_mont_request(nullptr,"fixture-cache"),prac));});
            same([]{return std::to_string(baseline::prac_sample_limit());},[]{return std::to_string(host::prac_sample_limit(host::capture_mont_request(nullptr,"fixture-cache")));});
        }
        for(auto cache:{"","C:/fixture-cache"}){env("ECM_PRAC_PLAN_CACHE",cache);same([]{return baseline::prac_plan_cache();},[]{return host::prac_plan_cache(host::capture_mont_request(nullptr,"fixture-cache"));});}
        for(auto key:{"ECM_PRAC_WINDOW","ECM_PRAC_VARIANT","ECM_PRAC_REG_TARGET","ECM_STAGE1_TPI"})env(key,"");
        env("ECM_GPU_STAGE1_ALGO","prac");env("ECM_PRAC_TARGET_MS","100");env("ECM_PRAC_PLAN_CACHE","snapshot-cache");
        env("ECM_GPU_STAGE1_SAMPLE_SECONDS","1");env("ECM_GPU_DUMP","1");
        const auto requested=host::capture_mont_request(nullptr,"fixture-cache");
        env("ECM_GPU_STAGE1_ALGO","bad");env("ECM_PRAC_TARGET_MS","bad");env("ECM_PRAC_PLAN_CACHE","changed");
        env("ECM_GPU_STAGE1_SAMPLE_SECONDS","bad");env("ECM_GPU_DUMP","0");
        if(std::string(host::stage1_algorithm(requested))!="prac" || host::prac_target_ms(requested,true)!=100 ||
           host::prac_plan_cache(requested)!="snapshot-cache" || host::prac_sample_limit(requested)!=1 || !host::gpu_dump_enabled(requested))
            throw std::runtime_error("owned run selection changed with process environment");
        auto explicit_empty=requested;explicit_empty.values["ECM_PRAC_TARGET_MS"]="";bool rejected=false;
        try{host::prac_target_ms(explicit_empty,true);}catch(const std::runtime_error &){rejected=true;}
        if(!rejected)throw std::runtime_error("explicit empty target confused with unset");
        std::cout<<"policy_snapshot: environment_isolation=1 explicit_empty=1\n";
        std::cout<<"policy_contract: compared="<<checked<<" passed=1\n";return 0;
    }catch(const std::exception &e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}
}
