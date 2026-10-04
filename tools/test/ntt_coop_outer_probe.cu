#define NTT_POLY_PROBE_NO_MAIN
#include "../bench/ntt_poly_probe.cu"

template<int M,bool INV> static void resources()
{
    cudaFuncAttributes a{};int blocks=0;
    CK(cudaFuncGetAttributes(&a,outer_coop_kernel<M,INV>));
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks,outer_coop_kernel<M,INV>,256,0));
    std::printf("ntt_coop_resources: M=%d inverse=%d regs=%d local=%zu shared=%zu max_blocks=%d\n",
                M,(int)INV,a.numRegs,a.localSizeBytes,a.sharedSizeBytes,blocks);
}
__global__ void sparse_input(unsigned long long *a,unsigned long long *b,unsigned long long n)
{
    const auto i=blockIdx.x*(unsigned long long)blockDim.x+threadIdx.x;
    if(i<n){a[i]=i==0 ? GL_P-1 : i==1 ? 0x8000000000000000ull : i==2 ? GL_P-2 : 0;
            b[i]=i==0 ? 3 : i==1 ? 5 : i==2 ? 7 : 0;}
}
__global__ void sparse_result(const unsigned long long *a,unsigned long long n,
                               const unsigned long long *want,unsigned long long *bad)
{
    const auto i=blockIdx.x*(unsigned long long)blockDim.x+threadIdx.x;
    if(i<n && a[i]!=(i<5 ? want[i] : 0))atomicAdd(bad,1ull);
}
static void benchmark(int device,int k)
{
    if(k<16 || k>27){std::fprintf(stderr,"benchmark k must be 16..27\n");std::exit(2);}
    const auto n=1ull<<k,om=gl_pow_host(7,(GL_P-1)/n),omi=gl_pow_host(om,GL_P-2),scale=gl_pow_host(n,GL_P-2);
    unsigned long long *a=nullptr,*b=nullptr,*want=nullptr,*bad=nullptr;
    CK(cudaMalloc(&a,n*8));CK(cudaMalloc(&b,n*8));CK(cudaMalloc(&want,5*8));CK(cudaMalloc(&bad,8));
    unsigned long long expected[5]={};
    const unsigned long long aa[3]={GL_P-1,0x8000000000000000ull,GL_P-2},bb[3]={3,5,7};
    mpz_t p,z,x,y;mpz_inits(p,z,x,y,nullptr);const unsigned long long prime=GL_P;mpz_import(p,1,-1,8,0,0,&prime);
    for(int c=0;c<5;++c){mpz_set_ui(z,0);for(int i=0;i<3;++i)if(c-i>=0 && c-i<3){
        mpz_import(x,1,-1,8,0,0,&aa[i]);mpz_import(y,1,-1,8,0,0,&bb[c-i]);mpz_addmul(z,x,y);}
        mpz_mod(z,z,p);size_t count=0;mpz_export(expected+c,&count,-1,8,0,0,z);}
    mpz_clears(p,z,x,y,nullptr);CK(cudaMemcpy(want,expected,5*8,cudaMemcpyHostToDevice));
    cudaEvent_t start,end;CK(cudaEventCreate(&start));CK(cudaEventCreate(&end));
    NttArena arena;arena.device=device;
    fuse_fixture_env("NTT_FUSE_T","12");fuse_fixture_env("NTT_FUSE_M","4");fuse_fixture_env("NTT_FUSE_WARP_TAIL","1");
    int run=0;
    for(int mode:{0,1,1,0,1,0,0,1}) {
        fuse_fixture_env("NTT_FUSE_COOP_OUTER",std::to_string(mode).c_str());
        FuseCtx fc;ntt_arena_fuse(&arena,n,k,om,omi,fc);
        double seconds=0;unsigned long long wrong=0;
        for(int repeat=0;repeat<4;++repeat) {
            sparse_input<<<(unsigned int)((n+255)/256),256>>>(a,b,n);
            CK(cudaEventRecord(start));
            ntt_forward_fused(a,fc,om);ntt_forward_fused(b,fc,om);ntt_inverse_fused(a,b,fc,omi,scale);
            CK(cudaEventRecord(end));CK(cudaEventSynchronize(end));float ms=0;CK(cudaEventElapsedTime(&ms,start,end));
            if(repeat)seconds+=ms/1000.0;
            CK(cudaMemset(bad,0,8));sparse_result<<<(unsigned int)((n+255)/256),256>>>(a,n,want,bad);
            CK(cudaMemcpy(&wrong,bad,8,cudaMemcpyDeviceToHost));if(wrong)std::exit(3);
        }
        std::printf("ntt_coop_bench: run=%d coop=%d k=%d N=%llu passes_fwd=%d seconds=%.9f bad=%llu\n",
                    ++run,mode,k,n,fc.passes_fwd,seconds/3,wrong);
    }
    arena.release();CK(cudaEventDestroy(start));CK(cudaEventDestroy(end));
    CK(cudaFree(a));CK(cudaFree(b));CK(cudaFree(want));CK(cudaFree(bad));
}
int main(int argc,char **argv)
{
    const int device=argc>1 ? std::atoi(argv[1]) : 1;
    CK(cudaSetDevice(device));
    if(argc>2 && !std::strcmp(argv[2],"--bench")) {benchmark(device,argc>3 ? std::atoi(argv[3]) : 27);return 0;}
    if(argc>2 && !std::strcmp(argv[2],"--legacy")) {
        fuse_fixture_env("NTT_FUSE_COOP_OUTER","0");fuse_fixture_env("NTT_FUSE_WARP_TEST","1");
        for(int mode:{0,1}) {
            fuse_fixture_env("NTT_FUSE_WARP_TAIL",std::to_string(mode).c_str());
            ntt_fuse_lifetime_check(device);ntt_fuse_capacity_check(device);
        }
        return 0;
    }
    ntt_fuse_coop_check(device);
    resources<5,false>();resources<5,true>();resources<6,false>();resources<6,true>();
    resources<7,false>();resources<7,true>();resources<8,false>();resources<8,true>();
    std::printf("ntt_coop_probe: device=%d bad=0\n",device);
    return 0;
}
