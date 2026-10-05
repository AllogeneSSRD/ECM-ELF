// CPU-only model gate using the actual NTT shape query and production class.
#define NTT_POLY_PROBE_NO_MAIN
#include "../bench/ntt_poly_probe.cu"
#include "../bench/stage2_d_model.cuh"

int main(int argc,char **argv)
{
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
