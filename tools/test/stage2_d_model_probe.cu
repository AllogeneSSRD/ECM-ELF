// Historical model gate. Frozen production profiles use the actual NTT module;
// the default retains the independent legacy engine's existing shape query.
#ifdef ECM_STAGE2_FROZEN_PRODUCTION_PROFILES
#include "../../src/core/ecm_stage2_logging.h"
#include "../../src/core/ecm_stage2_initial_memory.h"
#include "../../src/cuda/ntt/runtime.h"
#include "../experiments/cuda/stage2_tree/d_model_profiles.h"
#else
#include "../experiments/cuda/ntt_variants/engine.cuh"
#include "../experiments/cuda/stage2_tree/legacy_d_model_profiles.h"
#endif

int main(int argc,char **argv)
{
#ifdef ECM_STAGE2_FROZEN_PRODUCTION_PROFILES
    stage2_log::level=stage2_log::debug;
#endif
    if(argc!=6)return 2;
    const int bits=std::atoi(argv[1]),profile=std::atoi(argv[5]);
    const auto bound=std::strtoull(argv[2],nullptr,10),d=std::strtoull(argv[3],nullptr,10),p=std::strtoull(argv[4],nullptr,10);
    if(bits<1 || bits>8192 || !d || bound<d || !p || p>1000000 || (profile!=0 && profile!=2 && profile!=3 && profile!=4 && profile!=5 && profile!=6))return 2;
    if((profile==2 && !d_shape_rates_valid) || (profile==3 && !d_short_rates_valid) ||
       (profile==4 && !d_baby_rates_valid) || (profile==5 && !d_fixed_ptx_rates_valid) || (profile==6 && !d_point_fold_rates_valid))return 3;
    DPhaseModel model(bits,bound,profile);
    model.print(d,p,8ull*((bits+63)/64)*(9*p+8)+48);
    if(profile==4 || profile==5 || profile==6)std::printf("d_model_baby_payload: bytes=%llu\n",d_baby_payload_bytes(p,(bits+63)/64));
    std::printf("d_model_probe: profile=%d curves_executed=0\n",profile);
    return 0;
}
