// Same-binary full-transform experiment; production arithmetic and outer passes.
#define NTT_TENSOR_GOLDILOCKS_PROBE_NO_MAIN
#define NTT_TENSOR_USE_PRODUCTION_RUNTIME
#include "../test/ntt_tensor_goldilocks_probe.cu"
#include <stdexcept>

static std::vector<unsigned int> signed_roots(bool inverse)
{
    Transform matrix(inverse,false);std::vector<unsigned int> packed(320);
    constexpr int planes[]={0,1,3,5,6};
    for(int plane=0;plane<5;++plane)for(int half=0;half<2;++half)for(int lane=0;lane<32;++lane) {
        const int row=(lane>>2)+8*half,col=(lane&3)*4;unsigned int word=0;
        for(int i=0;i<4;++i) {
            int digits[8]={};
            if(row/8==(col+i)/8) {
                Word value=matrix.matrix[(row%8)*2*16+(col+i)%8];const int sign=value>GL_P/2 ? -1 : 1;
                if(sign<0)value=GL_P-value;
                if(value==1)digits[0]=sign;
                else if(value==(1ull<<24))digits[3]=sign;
                else if(value==(1ull<<48))digits[6]=sign;
                else if(value==((1ull<<40)-(1ull<<8))){digits[5]=sign;digits[1]=-sign;}
                else throw std::runtime_error("DFT8 root outside signed-plane contract");
            }
            word|=(unsigned int)(digits[planes[plane]]&255)<<(8*i);
        }
        packed[plane*64+half*32+lane]=word;
    }
    return packed;
}
__global__ void length_fill(Word *a,Word *b,Word words,Word seed)
{
    const Word i=(Word)blockIdx.x*blockDim.x+threadIdx.x;if(i>=words)return;
    Word x=i+seed+0x9e3779b97f4a7c15ull;
    x=(x^(x>>30))*0xbf58476d1ce4e5b9ull;x=(x^(x>>27))*0x94d049bb133111ebull;x^=x>>31;
    a[i]=x>=GL_P ? x-GL_P : x;x^=0x369dea0f31a53f85ull;
    b[i]=x>=GL_P ? x-GL_P : x;
}
__global__ void length_compare(const Word *a,const Word *b,const Word *ra,const Word *rb,Word words,Word *bad)
{
    const Word i=(Word)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<words && (a[i]!=ra[i] || b[i]!=rb[i]))atomicAdd(bad,1ull);
}
struct LengthPlan {
    int k,t,threads;Word n,omega,omega_inverse,scale;FuseCtx ctx;
    std::vector<unsigned int> roots[4];Buffer rf,ri,hf,hi;
    LengthPlan(int bits,int tile,int cta):k(bits),t(std::min(bits,tile)),threads(cta),n(1ull<<bits),
        omega(gl_pow_host(7,(GL_P-1)/n)),omega_inverse(gl_pow_host(omega,GL_P-2)),scale(gl_pow_host(n,GL_P-2)),
        rf(2048),ri(2048),hf(1280),hi(1280)
    {
        char tile_text[16];std::snprintf(tile_text,sizeof(tile_text),"%d",t);_putenv_s("NTT_FUSE_T",tile_text);
        fuse_init(ctx,n,k,omega,omega_inverse);ntt_fuse_cache_tables(ctx,omega,omega_inverse);
        roots[0]=pack_roots(Transform(false,false).matrix);roots[1]=pack_roots(Transform(true,false).matrix);
        roots[2]=signed_roots(false);roots[3]=signed_roots(true);
        Buffer *dst[]={&rf,&ri,&hf,&hi};
        for(int i=0;i<4;++i)CK(cudaMemcpy(dst[i]->p,roots[i].data(),dst[i]->bytes,cudaMemcpyHostToDevice));
        const void *kernels[]={(const void *)tc_goldilocks_tile<false>,(const void *)tc_goldilocks_tile<true>,
            (const void *)tc_goldilocks_tile<false,true>,(const void *)tc_goldilocks_tile<true,true>};
        for(const auto kernel:kernels) {
            CK(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,(1u<<t)*8));
            CK(cudaFuncSetAttribute(kernel,cudaFuncAttributePreferredSharedMemoryCarveout,100));
        }
    }
    ~LengthPlan(){fuse_release(ctx);}
    void transform(int mode,bool inverse,Word *a,const Word *b,Word batch) {
        if(!mode) {
            if(inverse)ntt_inverse_fused(a,b,ctx,omega_inverse,scale,batch);
            else ntt_forward_fused(a,ctx,omega,batch);
        } else {
            // t only gates the tile launch in the production runners. Preserve
            // k, nms, outer_stages, roots and radix tables exactly.
            FuseCtx outer=ctx;outer.t=0;
            if(!inverse)ntt_forward_fused(a,outer,omega,batch);
            const dim3 grid((unsigned int)(n>>t),(unsigned int)batch);const int shared=(1u<<t)*8;
            if(mode==1) {
                if(inverse)tc_goldilocks_tile<true><<<grid,threads,shared>>>(a,b,n,t,ctx.tblI,scale,n,ri.get<unsigned int>());
                else tc_goldilocks_tile<false><<<grid,threads,shared>>>(a,nullptr,n,t,ctx.tblF,0,n,rf.get<unsigned int>());
            } else {
                if(inverse)tc_goldilocks_tile<true,true><<<grid,threads,shared>>>(a,b,n,t,ctx.tblI,scale,n,hi.get<unsigned int>());
                else tc_goldilocks_tile<false,true><<<grid,threads,shared>>>(a,nullptr,n,t,ctx.tblF,0,n,hf.get<unsigned int>());
            }
            CK(cudaGetLastError());
            if(inverse)ntt_inverse_fused(a,b,outer,omega_inverse,scale,batch);
        }
    }
    void operation(int mode,int operation,Word *a,Word *b,Word batch) {
        if(operation==2){transform(mode,false,a,nullptr,batch);transform(mode,false,b,nullptr,batch);}
        transform(mode,operation!=0,a,b,batch);
    }
};
static int length_gate(bool fault)
{
    Word total_bad=0,words_checked=0;bool injected=false;
    for(int k:{6,8,10,12,13,14})for(int batch:{1,3}) {
        const Word n=1ull<<k,words=n*batch;LengthPlan plan(k,12,256);
        Buffer a(words*8),b(words*8);Checks check[3];
        for(int pattern:{0,2,4,5,7}) {
            auto input=fixture(pattern,words,0x9134+k),multiplier=fixture(5,words,0x743+pattern);
            const auto forward=tile_reference(input,multiplier,k,false,1,batch,0);
            const auto inverse=tile_reference(input,multiplier,k,true,1,batch,0);
            const auto bf=tile_reference(multiplier,input,k,false,1,batch,0);
            const auto convolution=tile_reference(forward,bf,k,true,1,batch,0);
            const std::vector<Word> *expected[]={&forward,&inverse,&convolution};
            for(int operation=0;operation<3;++operation)for(int mode=0;mode<3;++mode) {
                CK(cudaMemcpy(a.p,input.data(),a.bytes,cudaMemcpyHostToDevice));CK(cudaMemcpy(b.p,multiplier.data(),b.bytes,cudaMemcpyHostToDevice));
                plan.operation(mode,operation,a.get(),b.get(),batch);
                if(fault && mode==2 && !injected){corrupt_first<<<1,32>>>(a.get());injected=true;}
                std::vector<Word> got(words),got_b(words);CK(cudaMemcpy(got.data(),a.p,a.bytes,cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(got_b.data(),b.p,b.bytes,cudaMemcpyDeviceToHost));
                compare(check[mode],got,*expected[operation]);compare(check[mode],got_b,operation==2 ? bf : multiplier);
            }
        }
        for(int mode=0;mode<3;++mode) {
            total_bad+=check[mode].bad;words_checked+=check[mode].words;
            std::printf("tc_length_check: k=%d batch=%d mode=%d cases=%llu words=%llu bad=%llu\n",k,batch,mode,check[mode].cases,check[mode].words,check[mode].bad);
        }
    }
    std::printf("tc_length_gate: fault=%d injected=%d checked_words=%llu bad=%llu\n",(int)fault,(int)injected,words_checked,total_bad);
    return total_bad ? 3 : 0;
}
static int length_bench(int k,int batch,int tile,int threads,int operation,int repeats,bool graph,int iterations)
{
    LengthPlan plan(k,tile,threads);const Word words=plan.n*batch;
    Buffer a(words*8),b(words*8),ra(words*8),rb(words*8),bad(8);
    length_fill<<<(unsigned int)((words+255)/256),256>>>(a.get(),b.get(),words,0x5193);
    for(int i=0;i<iterations;++i)plan.operation(0,operation,a.get(),b.get(),batch);
    CK(cudaMemcpy(ra.p,a.p,a.bytes,cudaMemcpyDeviceToDevice));CK(cudaMemcpy(rb.p,b.p,b.bytes,cudaMemcpyDeviceToDevice));
    cudaGraph_t captured[3]={};cudaGraphExec_t executable[3]={};
    if(graph)for(int mode=0;mode<3;++mode) {
        CK(cudaStreamBeginCapture(0,cudaStreamCaptureModeGlobal));
        for(int i=0;i<iterations;++i)plan.operation(mode,operation,a.get(),b.get(),batch);
        CK(cudaStreamEndCapture(0,&captured[mode]));
        CK(cudaGraphInstantiate(&executable[mode],captured[mode],nullptr,nullptr,0));
    }
    cudaEvent_t start,end;CK(cudaEventCreate(&start));CK(cudaEventCreate(&end));
    const auto tables=8*(fuse_planned_base_words(plan.ctx)+fuse_planned_table_words(plan.ctx));
    std::printf("tc_length_plan: k=%d batch=%d t=%d passes=%d coop=%d max_radix_bits=%d data_bytes=%llu table_base_bytes=%llu root_bytes=6656 setup_excluded=1 warmup=0 graph=%d iterations=%d\n",
        k,batch,plan.t,plan.ctx.passes_fwd,(int)plan.ctx.coop_outer,plan.ctx.m_max,words*32,tables,(int)graph,iterations);
    int run=0;
    for(int repeat=0;repeat<repeats;++repeat)for(int i=0;i<3;++i) {
        const int mode=(repeat%2 ? 2-i : i+repeat)%3;
        length_fill<<<(unsigned int)((words+255)/256),256>>>(a.get(),b.get(),words,0x5193);
        CK(cudaEventRecord(start));const double host_begin=now_s();
        if(graph)CK(cudaGraphLaunch(executable[mode],0));
        else for(int i=0;i<iterations;++i)plan.operation(mode,operation,a.get(),b.get(),batch);
        CK(cudaEventRecord(end));CK(cudaEventSynchronize(end));const double host_seconds=now_s()-host_begin;
        float ms=0;CK(cudaEventElapsedTime(&ms,start,end));
        CK(cudaMemset(bad.p,0,8));length_compare<<<(unsigned int)((words+255)/256),256>>>(a.get(),b.get(),ra.get(),rb.get(),words,bad.get());
        Word wrong=0;CK(cudaMemcpy(&wrong,bad.p,8,cudaMemcpyDeviceToHost));
        std::printf("tc_length_bench: run=%d repeat=%d mode=%d k=%d N=%llu batch=%d t=%d threads=%d operation=%d seconds=%.9f host_seconds=%.9f checked_words=%llu bad=%llu graph=%d iterations=%d\n",
            ++run,repeat+1,mode,k,plan.n,batch,plan.t,threads,operation,ms/(1000.0*iterations),host_seconds/iterations,words*2,wrong,(int)graph,iterations);
        if(wrong)return 3;
    }
    CK(cudaEventDestroy(start));CK(cudaEventDestroy(end));
    if(graph)for(int mode=0;mode<3;++mode){CK(cudaGraphExecDestroy(executable[mode]));CK(cudaGraphDestroy(captured[mode]));}
    return 0;
}
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
        tile_resources("production_forward",tile_kernel<false,true>,512);tile_resources("production_inverse",tile_kernel<true,true>,512);
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
