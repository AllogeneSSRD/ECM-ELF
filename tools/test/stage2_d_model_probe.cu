// CPU-only model gate using the actual NTT shape query and production class.
#define NTT_POLY_PROBE_NO_MAIN
#include "../bench/ntt_poly_probe.cu"
#include "../bench/stage2_d_model.cuh"

int main(int argc,char **argv)
{
    if(argc!=6)return 2;
    const int bits=std::atoi(argv[1]),profile=std::atoi(argv[5]);
    const auto bound=std::strtoull(argv[2],nullptr,10),d=std::strtoull(argv[3],nullptr,10),p=std::strtoull(argv[4],nullptr,10);
    if(bits<1 || bits>8192 || !d || bound<d || !p || p>1000000 || (profile!=0 && profile!=2))return 2;
    if(profile==2 && !d_shape_rates_valid)return 3;
    DPhaseModel model(bits,bound,profile==2);
    model.print(d,p,8ull*((bits+63)/64)*(9*p+8)+48);
    std::printf("d_model_probe: profile=%d curves_executed=0\n",profile);
    return 0;
}
