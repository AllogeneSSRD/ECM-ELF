#include "../experiments/cuda/ntt_tensor/length_fixture.cuh"
int main(int argc,char **argv)
{
    if(argc<3){std::fprintf(stderr,"device --check|--fault|--bench [k batch tile threads operation repeats]\n");return 2;}
    try {
        std::setvbuf(stdout,nullptr,_IONBF,0);stage2_log::level=stage2_log::quiet;
        _putenv_s("NTT_FUSE_WARP_TAIL","1");_putenv_s("NTT_FUSE_M","4");_putenv_s("NTT_FUSE_COOP_OUTER","2");_putenv_s("NTT_FUSE_COMPACT_SCRATCH","1");
        const int device=std::atoi(argv[1]);CK(cudaSetDevice(device));cudaDeviceProp p{};CK(cudaGetDeviceProperties(&p,device));
        if(p.major<8)return 2;
        std::printf("tc_length_device: device=%d name=%s sm=%d%d fixed=%d production_runtime=1 sm_count=%d\n",device,p.name,p.major,p.minor,NTT_GL_FIXED_MODE,p.multiProcessorCount);
        for(int threads:{128,256,512}) {
            tile_resources("tensor64_forward",tc_goldilocks_tile<false>,threads);tile_resources("tensor64_inverse",tc_goldilocks_tile<true>,threads);
            tile_resources("tensor40_forward",tc_goldilocks_tile<false,true>,threads);tile_resources("tensor40_inverse",tc_goldilocks_tile<true,true>,threads);
        }
        tile_resources("production_forward",ecm_ntt_tile_handle(false,true),512);tile_resources("production_inverse",ecm_ntt_tile_handle(true,true),512);
        int code=2;
        if(!std::strcmp(argv[2],"--check") || !std::strcmp(argv[2],"--fault"))code=length_gate(!std::strcmp(argv[2],"--fault"));
        else if((!std::strcmp(argv[2],"--bench") || !std::strcmp(argv[2],"--bench-graph")) && (argc==9 || argc==10)) {
            const int k=std::atoi(argv[3]),batch=std::atoi(argv[4]),tile=std::atoi(argv[5]),threads=std::atoi(argv[6]),op=std::atoi(argv[7]),repeats=std::atoi(argv[8]);
            const int iterations=argc==10 ? std::atoi(argv[9]) : 1;
            if(k>=6 && k<=27 && batch>=1 && batch<=64 && tile>=6 && tile<=12 && (threads==128 || threads==256 || threads==512) && op>=0 && op<=2 && repeats>=1 && repeats<=101 && iterations>=1 && iterations<=256)
                code=length_bench(k,batch,tile,threads,op,repeats,!std::strcmp(argv[2],"--bench-graph"),iterations);
        }
        std::printf("tc_length_memory: requested_peak_bytes=%llu fuse_base_peak_bytes=%llu live_bytes=%llu fuse_live_bytes=%llu\n",peak_bytes,g_fuse_base.peak_bytes,live_bytes,g_fuse_base.live_bytes);
        return code;
    } catch(const std::exception &e){std::fprintf(stderr,"tc_length_error: %s\n",e.what());return 2;}
}
