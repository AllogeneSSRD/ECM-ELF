// One context, private data, immutable cached roots, explicit nonblocking streams.
#include "../experiments/cuda/ntt_tensor/length_fixture.cuh"
#include <cuda_profiler_api.h>

struct ConcurrentSubmit {
    cudaStream_t control{},lane[2]{};cudaEvent_t fork{},done[2]{};
    ConcurrentSubmit() {
        CK(cudaStreamSynchronize(0)); // Plan/root uploads were issued on the default stream.
        CK(cudaStreamCreateWithFlags(&control,cudaStreamNonBlocking));
        CK(cudaEventCreateWithFlags(&fork,cudaEventDisableTiming));
        for(int i=0;i<2;++i) {
            CK(cudaStreamCreateWithFlags(&lane[i],cudaStreamNonBlocking));
            CK(cudaEventCreateWithFlags(&done[i],cudaEventDisableTiming));
        }
    }
    ~ConcurrentSubmit() {
        CK(cudaStreamSynchronize(control));CK(cudaEventDestroy(fork));
        for(int i=0;i<2;++i){CK(cudaStreamSynchronize(lane[i]));CK(cudaEventDestroy(done[i]));CK(cudaStreamDestroy(lane[i]));}
        CK(cudaStreamDestroy(control));
    }
    void submit(LengthPlan &plan,int first,int second,bool parallel,int jobs,
                Word *a,Word *b,Word *c,Word *d,Word batch,int iterations) {
        CK(cudaEventRecord(fork,control));
        for(int i=0;i<2;++i)CK(cudaStreamWaitEvent(lane[i],fork,0));
        for(int i=0;i<iterations;++i) {
            plan.operation(first,2,a,b,batch,lane[0]);
            if(jobs==2)plan.operation(second,2,c,d,batch,lane[parallel ? 1 : 0]);
        }
        for(int i=0;i<2;++i){CK(cudaEventRecord(done[i],lane[i]));CK(cudaStreamWaitEvent(control,done[i],0));}
    }
};
struct ConcurrentData {
    Word words;Buffer a,b,ra,rb;
    explicit ConcurrentData(Word n):words(n),a(n*8),b(n*8),ra(n*8),rb(n*8){}
    void fill(cudaStream_t stream,Word seed) {
        length_fill<<<(unsigned int)((words+255)/256),256,0,stream>>>(a.get(),b.get(),words,seed);CK(cudaGetLastError());
    }
    void reference(LengthPlan &plan,Word batch,int iterations,cudaStream_t stream,Word seed) {
        fill(stream,seed);
        for(int i=0;i<iterations;++i)plan.operation(0,2,a.get(),b.get(),batch,stream);
        CK(cudaMemcpyAsync(ra.p,a.p,a.bytes,cudaMemcpyDeviceToDevice,stream));
        CK(cudaMemcpyAsync(rb.p,b.p,b.bytes,cudaMemcpyDeviceToDevice,stream));
    }
    void check(Buffer &bad,cudaStream_t stream) {
        length_compare<<<(unsigned int)((words+255)/256),256,0,stream>>>(a.get(),b.get(),ra.get(),rb.get(),words,bad.get());CK(cudaGetLastError());
    }
};
struct ConcurrentScene {const char *name;int first,second,jobs;bool parallel;};
static const ConcurrentScene concurrent_scenes[]={
    {"single_C",0,0,1,false},{"single_T40",2,0,1,false},{"single_N40",3,0,1,false},
    {"CC_serial",0,0,2,false},{"CC_parallel",0,0,2,true},
    {"CT40_serial",0,2,2,false},{"CT40_parallel",0,2,2,true},
    {"TT40_serial",2,2,2,false},{"TT40_parallel",2,2,2,true},
    {"CN40_serial",0,3,2,false},{"CN40_parallel",0,3,2,true},
    {"NN40_serial",3,3,2,false},{"NN40_parallel",3,3,2,true}
};

static int concurrency_gate() {
    Word bad=0,checked=0;
    for(int k:{6,10,13}) {
        const Word batch=3,n=1ull<<k,words=n*batch;LengthPlan plan(k,12,256);ConcurrentSubmit submit;
        if(!plan.ctx.tables_cached)return 3;
        const auto input=fixture(4,words,13),multiplier=fixture(7,words,91);
        const auto fa=tile_reference(input,multiplier,k,false,1,batch,0);
        const auto fb=tile_reference(multiplier,input,k,false,1,batch,0);
        const auto expected=tile_reference(fa,fb,k,true,1,batch,0);
        Buffer a(words*8),b(words*8),c(words*8),d(words*8);
        for(const auto &scene:concurrent_scenes)if(scene.jobs==2) {
            for(auto *dest:{&a,&c})CK(cudaMemcpy(dest->p,input.data(),dest->bytes,cudaMemcpyHostToDevice));
            for(auto *dest:{&b,&d})CK(cudaMemcpy(dest->p,multiplier.data(),dest->bytes,cudaMemcpyHostToDevice));
            CK(cudaStreamSynchronize(0)); // Pageable H2D staging must finish before nonblocking readers.
            submit.submit(plan,scene.first,scene.second,scene.parallel,2,a.get(),b.get(),c.get(),d.get(),batch,1);
            CK(cudaStreamSynchronize(submit.control));
            for(auto *source:{&a,&b,&c,&d}) {
                std::vector<Word> got(words);CK(cudaMemcpy(got.data(),source->p,source->bytes,cudaMemcpyDeviceToHost));
                const auto &want=(source==&a || source==&c) ? expected : fb;
                for(size_t i=0;i<got.size();++i){++checked;if(got[i]!=want[i])++bad;}
            }
        }
    }
    std::printf("tc_concurrency_gate: checked_words=%llu bad=%llu\n",checked,bad);return bad?3:0;
}
static int concurrency_bench(int k,int batch,int tile,int threads,int repeats,int iterations,bool graph,int profile_scene) {
    LengthPlan plan(k,tile,threads);ConcurrentSubmit submit;const Word words=plan.n*batch;
    if(!plan.ctx.tables_cached)return 3;
    ConcurrentData x(words),y(words);Buffer bad(8);
    x.reference(plan,batch,iterations,submit.control,0x9134);y.reference(plan,batch,iterations,submit.control,0x1439);
    CK(cudaStreamSynchronize(submit.control));
    cudaGraph_t captured[13]={};cudaGraphExec_t executable[13]={};
    if(graph)for(int s=0;s<13;++s)if(profile_scene<0 || s==profile_scene) {
        const auto &scene=concurrent_scenes[s];
        CK(cudaStreamBeginCapture(submit.control,cudaStreamCaptureModeGlobal));
        submit.submit(plan,scene.first,scene.second,scene.parallel,scene.jobs,x.a.get(),x.b.get(),y.a.get(),y.b.get(),batch,iterations);
        CK(cudaStreamEndCapture(submit.control,&captured[s]));
        CK(cudaGraphInstantiate(&executable[s],captured[s],nullptr,nullptr,0));
    }
    cudaEvent_t begin,end;CK(cudaEventCreate(&begin));CK(cudaEventCreate(&end));
    const auto tables=8*(fuse_planned_base_words(plan.ctx)+fuse_planned_table_words(plan.ctx));
    std::printf("tc_concurrency_plan: k=%d batch=%d t=%d cta=%d iterations=%d graph=%d passes=%d coop=%d data_bytes=%llu table_base_bytes=%llu root_bytes=6656 private_data=1 cached_readonly_roots=1\n",
        k,batch,plan.t,threads,iterations,(int)graph,plan.ctx.passes_fwd,(int)plan.ctx.coop_outer,words*64,tables);
    Word wrong_total=0;int run=0;
    for(int r=0;r<repeats;++r)for(int q=0;q<13;++q) {
        const int s=(r%2 ? 12-q+r : q+r)%13;
        if(profile_scene>=0 && s!=profile_scene)continue;
        const auto &scene=concurrent_scenes[s];
        x.fill(submit.control,0x9134);y.fill(submit.control,0x1439);
        CK(cudaStreamSynchronize(submit.control)); // Fill is entirely outside both timers.
        if(profile_scene>=0)CK(cudaProfilerStart());
        const double host_begin=now_s();CK(cudaEventRecord(begin,submit.control));
        if(graph)CK(cudaGraphLaunch(executable[s],submit.control));
        else submit.submit(plan,scene.first,scene.second,scene.parallel,scene.jobs,x.a.get(),x.b.get(),y.a.get(),y.b.get(),batch,iterations);
        CK(cudaEventRecord(end,submit.control));CK(cudaEventSynchronize(end));
        const double host=now_s()-host_begin;float ms=0;CK(cudaEventElapsedTime(&ms,begin,end));
        if(profile_scene>=0)CK(cudaProfilerStop());
        CK(cudaMemsetAsync(bad.p,0,8,submit.control));x.check(bad,submit.control);
        if(scene.jobs==2)y.check(bad,submit.control);
        Word wrong=0;CK(cudaMemcpyAsync(&wrong,bad.p,8,cudaMemcpyDeviceToHost,submit.control));CK(cudaStreamSynchronize(submit.control));
        wrong_total+=wrong;
        std::printf("tc_concurrency_bench: run=%d repeat=%d scene=%d name=%s jobs=%d parallel=%d k=%d batch=%d cta=%d iterations=%d graph=%d seconds=%.9f host_seconds=%.9f checked_words=%llu bad=%llu profile=%d\n",
            ++run,r+1,s,scene.name,scene.jobs,(int)scene.parallel,k,batch,threads,iterations,(int)graph,ms/1000.0,host,
            words*2*scene.jobs,wrong,(int)(profile_scene>=0));
    }
    CK(cudaEventDestroy(begin));CK(cudaEventDestroy(end));
    if(graph)for(int s=0;s<13;++s)if(executable[s]){CK(cudaGraphExecDestroy(executable[s]));CK(cudaGraphDestroy(captured[s]));}
    return wrong_total?3:0;
}
int main(int argc,char **argv) {
    if(argc<3)return 2;
    try {
        std::setvbuf(stdout,nullptr,_IONBF,0);stage2_log::level=stage2_log::quiet;
        _putenv_s("NTT_FUSE_WARP_TAIL","1");_putenv_s("NTT_FUSE_M","4");_putenv_s("NTT_FUSE_COOP_OUTER","2");_putenv_s("NTT_FUSE_COMPACT_SCRATCH","1");
        const int device=std::atoi(argv[1]);CK(cudaSetDevice(device));cudaDeviceProp p{};CK(cudaGetDeviceProperties(&p,device));
        if(p.major<8)return 2;
        std::printf("tc_concurrency_device: device=%d name=%s sm=%d%d sms=%d concurrent=%d shared_sm=%llu registers_sm=%d threads_sm=%d fixed=%d\n",
            device,p.name,p.major,p.minor,p.multiProcessorCount,p.concurrentKernels,(Word)p.sharedMemPerMultiprocessor,p.regsPerMultiprocessor,p.maxThreadsPerMultiProcessor,NTT_GL_FIXED_MODE);
        for(int threads:{128,256,512}) {
            tile_resources("tensor40_forward",tc_goldilocks_tile<false,true>,threads);tile_resources("tensor40_inverse",tc_goldilocks_tile<true,true>,threads);
            tile_resources("narrow40_forward",tc_goldilocks_narrow64_tile<false>,threads);tile_resources("narrow40_inverse",tc_goldilocks_narrow64_tile<true>,threads);
        }
        tile_resources("production_forward",ecm_ntt_tile_handle(false,true),512);tile_resources("production_inverse",ecm_ntt_tile_handle(true,true),512);
        int code=2;
        if(!std::strcmp(argv[2],"--check")){code=length_gate(false,4);if(!code)code=concurrency_gate();}
        else if(!std::strcmp(argv[2],"--fault"))code=length_gate(true,4);
        else if(!std::strcmp(argv[2],"--bench") && (argc==10 || argc==11)) {
            const int k=std::atoi(argv[3]),b=std::atoi(argv[4]),t=std::atoi(argv[5]),cta=std::atoi(argv[6]),r=std::atoi(argv[7]),it=std::atoi(argv[8]),g=std::atoi(argv[9]);
            const int profile=argc==11 ? std::atoi(argv[10]) : -1;
            if(k>=6 && k<=26 && b>=1 && b<=4096 && t>=6 && t<=12 && (cta==128 || cta==256 || cta==512) &&
               r>=1 && r<=101 && it>=1 && it<=256 && (g==0 || g==1) && profile>=-1 && profile<13 && (1ull<<k)*(Word)b<=(1ull<<25))
                code=concurrency_bench(k,b,t,cta,r,it,g,profile);
        }
        std::printf("tc_concurrency_memory: peak_bytes=%llu live_bytes=%llu fuse_live_bytes=%llu\n",peak_bytes,live_bytes,g_fuse_base.live_bytes);
        return code;
    } catch(const std::exception &e){std::fprintf(stderr,"tc_concurrency_error: %s\n",e.what());return 2;}
}
