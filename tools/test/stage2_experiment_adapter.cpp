// Diagnostic entry into an already built experimental worker. No engine inclusion.
#include "../../src/ecm/backend/cuda/stage2/ecm_cuda_stage2.h"
#include <cstdio>
#include <cstdlib>

static void result(const char *json,void *) {
    std::printf("{\"type\":\"adapter_result\",%s}\n",json);
}
int main(int argc,char **argv) {
    if(argc!=8)return 2;
    if(ecm_cuda_stage2_set_log_level(4))return 2;
    return ecm_cuda_stage2_run(argv[1],argv[2],std::strtoull(argv[3],nullptr,10),
        std::strtoull(argv[4],nullptr,10),std::strtoull(argv[5],nullptr,10),
        std::strtoull(argv[6],nullptr,10),std::atoi(argv[7]),result,nullptr,0,nullptr,nullptr);
}
