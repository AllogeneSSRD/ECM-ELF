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
static void benchmark(int device,int k,bool automatic=false,bool short_reduce=false,bool shift_scale=false,bool ptx_reduce=false,bool fixed=false)
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
        fuse_fixture_env("NTT_FUSE_COOP_OUTER",std::to_string(fixed || short_reduce || shift_scale || ptx_reduce ? 2 : mode && automatic ? 2 : mode).c_str());
        if(short_reduce)fuse_fixture_env("NTT_GL_SHORT_REDUCE",std::to_string(mode).c_str());
        if(shift_scale){fuse_fixture_env("NTT_GL_SHORT_REDUCE","1");fuse_fixture_env("NTT_GL_SHIFT_SCALE",std::to_string(mode).c_str());}
        if(ptx_reduce){fuse_fixture_env("NTT_GL_SHORT_REDUCE","1");fuse_fixture_env("NTT_GL_PTX_REDUCE",std::to_string(mode).c_str());}
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
        std::printf("%s: run=%d %s=%d k=%d N=%llu passes_fwd=%d seconds=%.9f bad=%llu policy=%d selected_M=%d selected_coop=%d\n",
                    fixed ? "ntt_fixed_bench" : ptx_reduce ? "ntt_ptx_bench" : shift_scale ? "ntt_scale_bench" : short_reduce ? "ntt_reduce_bench" : "ntt_coop_bench",
                    ++run,fixed ? "backend" : ptx_reduce ? "ptx" : shift_scale ? "shift" : short_reduce ? "short" : "coop",fixed ? NTT_GL_FIXED_MODE : mode,k,n,fc.passes_fwd,seconds/3,wrong,(int)automatic,fc.m_max,(int)fc.coop_outer);
    }
    arena.release();CK(cudaEventDestroy(start));CK(cudaEventDestroy(end));
    CK(cudaFree(a));CK(cudaFree(b));CK(cudaFree(want));CK(cudaFree(bad));
}

__global__ void scale_check_kernel(const unsigned long long *x,const unsigned long long *scale,
    const int *k,unsigned long long *out,size_t count)
{
    const auto i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count)out[i]=gl_scale_is_inverse_pow2(scale[i],k[i]) ?
        gl_scale_inverse_pow2(x[i],k[i]) : gl_mul(x[i],scale[i]);
}
static int scale_check()
{
    std::vector<unsigned long long> x,scale,want;std::vector<int> shifts;
    const unsigned long long edges[]={0,1,2,GL_P-1,GL_P-2,GL_P/2,0xffffffffull,1ull<<32,1ull<<63};
    mpz_t p,z,a,b;mpz_inits(p,z,a,b,nullptr);const auto prime=GL_P;mpz_import(p,1,-1,8,0,0,&prime);
    auto append=[&](unsigned long long v,unsigned long long s,int k) {
        unsigned long long w=0;size_t count=0;mpz_import(a,1,-1,8,0,0,&v);mpz_import(b,1,-1,8,0,0,&s);
        mpz_mul(z,a,b);mpz_mod(z,z,p);mpz_export(&w,&count,-1,8,0,0,z);
        x.push_back(v);scale.push_back(s);shifts.push_back(k);want.push_back(w);
    };
    unsigned long long seed=0x873197ull;
    for(int k=1;k<=32;++k) {
        const auto s=gl_pow_host(1ull<<k,GL_P-2);
        for(auto v:edges)append(v,s,k);
        // Boundaries on both sides of r=0 and of the correction's comparison.
        for(auto v:edges)for(int delta:{-1,0,1}) {
            auto v2=(v&~((1ull<<k)-1));v2+=delta;
            if(v2<GL_P)append(v2,s,k);
        }
        for(int i=0;i<8192;++i){seed^=seed<<13;seed^=seed>>7;seed^=seed<<17;append(seed%GL_P,s,k);}
    }
    for(int k:{-1,0,1,16,32,33,64})for(auto s:{0ull,1ull,GL_P-1})for(auto v:edges)append(v,s,k);
    mpz_clears(p,z,a,b,nullptr);
    unsigned long long *dx=nullptr,*ds=nullptr,*out=nullptr;int *dk=nullptr;
    CK(cudaMalloc(&dx,x.size()*8));CK(cudaMalloc(&ds,x.size()*8));CK(cudaMalloc(&dk,x.size()*4));CK(cudaMalloc(&out,x.size()*8));
    CK(cudaMemcpy(dx,x.data(),x.size()*8,cudaMemcpyHostToDevice));CK(cudaMemcpy(ds,scale.data(),x.size()*8,cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dk,shifts.data(),x.size()*4,cudaMemcpyHostToDevice));
    scale_check_kernel<<<(unsigned int)((x.size()+255)/256),256>>>(dx,ds,dk,out,x.size());
    std::vector<unsigned long long> got(x.size());CK(cudaMemcpy(got.data(),out,x.size()*8,cudaMemcpyDeviceToHost));
    size_t bad=0;for(size_t i=0;i<x.size();++i)if(got[i]!=want[i])++bad;
    CK(cudaFree(dx));CK(cudaFree(ds));CK(cudaFree(dk));CK(cudaFree(out));
    std::printf("ntt_scale_check: words=%zu bad=%zu\n",x.size(),bad);return bad ? 3 : 0;
}
static void policy_check()
{
    unsigned long long calls=0,bad=0;
    const bool supported=fuse_shape_device_supported();
    fuse_fixture_env("NTT_FUSE_COOP_OUTER","2");fuse_fixture_env("NTT_FUSE_M","4");
    // The automatic table, its boundaries and unsupported tile/warp/scratch modes.
    for(int t:{8,12})for(int warp:{0,1})for(int compact:{0,1})for(int k:{5,13,23,24,25,26,27,28,29}) {
        fuse_fixture_env("NTT_FUSE_WARP_TAIL",std::to_string(warp).c_str());
        fuse_fixture_env("NTT_FUSE_COMPACT_SCRATCH",std::to_string(compact).c_str());
        bool coop=false;const int m=fuse_outer_max(t,k,coop);
        const bool want=supported && t==12 && warp && compact && k>=24 && k<=27;
        if(coop!=want || m!=(want ? (k==24 ? 6 : 8) : 4))++bad;
        ++calls;
    }
    fuse_fixture_env("NTT_FUSE_WARP_TAIL","1");fuse_fixture_env("NTT_FUSE_COMPACT_SCRATCH","1");
    for(int mode:{0,1,2})for(int request:{1,4,5,6,8}) {
        fuse_fixture_env("NTT_FUSE_COOP_OUTER",std::to_string(mode).c_str());
        fuse_fixture_env("NTT_FUSE_COOP_M",std::to_string(request).c_str());
        bool coop=false;const int m=fuse_outer_max(12,24,coop);
        const bool want=mode==1 || (mode==2 && supported);
        const int wm=mode==1 ? std::max(5,request) : want ? 6 : 4;
        if(coop!=want || m!=wm)++bad;
        ++calls;
    }
    // An explicit original M override disables the measured automatic table.
    fuse_fixture_env("NTT_FUSE_COOP_OUTER","2");fuse_fixture_env("NTT_FUSE_M","3");
    bool coop=false;const int m=fuse_outer_max(12,27,coop);++calls;if(coop || m!=3)++bad;
    std::printf("ntt_shape_policy_check: calls=%llu supported=%d bad=%llu\n",calls,(int)supported,bad);
    if(bad)std::exit(3);
}
int main(int argc,char **argv)
{
    const int device=argc>1 ? std::atoi(argv[1]) : 1;
    CK(cudaSetDevice(device));
    if(argc>2 && !std::strcmp(argv[2],"--bench-fixed")) {
        if(NTT_GL_FIXED_MODE<0){std::fprintf(stderr,"--bench-fixed requires an immutable backend\n");return 2;}
        benchmark(device,argc>3 ? std::atoi(argv[3]) : 27,true,false,false,false,true);return 0;
    }
    if(argc>2 && !std::strcmp(argv[2],"--bench-ptx")) {benchmark(device,argc>3 ? std::atoi(argv[3]) : 27,true,false,false,true);return 0;}
    if(argc>2 && !std::strcmp(argv[2],"--scale-check")) {ntt_gl_reduce_configure();return scale_check();}
    if(argc>2 && !std::strcmp(argv[2],"--bench-scale")) {benchmark(device,argc>3 ? std::atoi(argv[3]) : 27,true,false,true);return 0;}
    if(argc>2 && !std::strcmp(argv[2],"--bench")) {benchmark(device,argc>3 ? std::atoi(argv[3]) : 27);return 0;}
    if(argc>2 && !std::strcmp(argv[2],"--bench-auto")) {benchmark(device,argc>3 ? std::atoi(argv[3]) : 27,true);return 0;}
    if(argc>2 && !std::strcmp(argv[2],"--bench-reduce")) {benchmark(device,argc>3 ? std::atoi(argv[3]) : 27,true,true);return 0;}
    if(argc>2 && !std::strcmp(argv[2],"--gl-selftest")) {return gl_selftest();}
    if(argc>2 && !std::strcmp(argv[2],"--policy")) {policy_check();return 0;}
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
