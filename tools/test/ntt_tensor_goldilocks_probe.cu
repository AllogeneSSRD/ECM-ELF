#include "ntt_tensor_fixture.cuh"
int main(int argc,char **argv)
{
    const int device=argc>1 ? std::atoi(argv[1]) : 1;const char *mode=argc>2 ? argv[2] : "--check";
    CK(cudaSetDevice(device));cudaDeviceProp prop{};CK(cudaGetDeviceProperties(&prop,device));
    if(prop.major<8){std::fprintf(stderr,"requires sm80+ integer MMA\n");return 2;}
    std::setvbuf(stdout,nullptr,_IONBF,0);
    std::printf("tc_gold_device: device=%d name=%s sm=%d%d\n",device,prop.name,prop.major,prop.minor);
    for(int threads:{32,64,128,256}) {
        resources("tensor",tc_goldilocks_mat16,threads);resources("fft_forward",cuda_goldilocks_fft16<false>,threads);
        resources("fft_inverse",cuda_goldilocks_fft16<true>,threads);resources("scalar_dft",cuda_goldilocks_dft16,threads);
    }
    for(int threads:{128,256,512}) {
        tile_resources("tensor_forward",tc_goldilocks_tile<false>,threads);
        tile_resources("tensor_inverse",tc_goldilocks_tile<true>,threads);
    }
    tile_resources("cuda_forward",ecm_ntt_tile_handle(false,true),512);
    tile_resources("cuda_inverse",ecm_ntt_tile_handle(true,true),512);
    int code=2;
    if(!std::strcmp(mode,"--check") || !std::strcmp(mode,"--fault"))code=gate(!std::strcmp(mode,"--fault"));
    else if(!std::strcmp(mode,"--tile-check") || !std::strcmp(mode,"--tile-fault"))code=tile_gate(!std::strcmp(mode,"--tile-fault"));
    else if(!std::strcmp(mode,"--tile-bench") || !std::strcmp(mode,"--mixed-bench")) {
        const int k=argc>3 ? std::atoi(argv[3]) : 24,t=argc>4 ? std::atoi(argv[4]) : 12;
        const int threads=argc>5 ? std::atoi(argv[5]) : 256,operation=argc>6 ? std::atoi(argv[6]) : 0;
        const bool mixed=!std::strcmp(mode,"--mixed-bench");
        if(k>=t && k>=12 && k<=26 && t>=6 && t<=12 && (threads==128 || threads==256 || threads==512) &&
           operation>=0 && operation<=2 && (!mixed || operation==0))
            code=mixed ? mixed_bench(k,t,threads) : tile_bench(k,t,threads,operation);
    }
    else if(!std::strcmp(mode,"--bench")) {
        const Word batches=argc>3 ? std::strtoull(argv[3],nullptr,10) : 65536;
        const int threads=argc>4 ? std::atoi(argv[4]) : 128,operation=argc>5 ? std::atoi(argv[5]) : 0;
        if(batches && batches<=262144 && (threads==32 || threads==64 || threads==128 || threads==256) && operation>=0 && operation<=2)
            code=bench(batches,threads,operation);
    }
    CK(cudaDeviceSynchronize());
    std::printf("tc_gold_memory: requested_peak_bytes=%llu live_bytes=%llu\n",peak_bytes,live_bytes);
    return code ? code : live_bytes ? 4 : 0;
}
