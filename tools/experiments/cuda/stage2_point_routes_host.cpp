// Test-only host adapter. Reuse the production implementation and every
// arithmetic guard; expose just three chain lengths in an isolated probe.
// The installed production executable and its option contract do not change.
#define ecm_cuda_stage2_check_configuration ecm_point_probe_canonical_configuration
#include "../../../src/ecm/stage2/cuda/options.cpp"
#undef ecm_cuda_stage2_check_configuration

int ecm_cuda_stage2_check_configuration()
{
    const char *raw=std::getenv("NTT_GIANT_CHAIN_BLOCK");
    const std::string value=raw?raw:"64";
    if(value!="32" && value!="64" && value!="128") {
        std::fprintf(stderr,"point_route_probe: chain block must be 32, 64 or 128\n");return 2;
    }
    _putenv_s("NTT_GIANT_CHAIN_BLOCK","64");
    const int result=ecm_point_probe_canonical_configuration();
    _putenv_s("NTT_GIANT_CHAIN_BLOCK",raw?value.c_str():"");
    std::fprintf(stderr,"point_route_probe: isolated host adapter chain_block=%s\n",value.c_str());
    return result;
}
