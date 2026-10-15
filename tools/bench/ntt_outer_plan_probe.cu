// Same-binary outer-plan study using the production runtime and arithmetic.
#include <algorithm>
#include <stdexcept>
#include "../../src/core/ecm_stage2_logging.h"
#include "../../src/cuda/stage2/ntt_runtime.cuh"
#include <cuda_profiler_api.h>

using Word=unsigned long long;
struct PlanBuffer {
    Word *p=nullptr;
    explicit PlanBuffer(Word words){CK(cudaMalloc(&p,words*8));}
    ~PlanBuffer(){CK(cudaFree(p));}
};
__global__ void outer_plan_fill(Word *a,Word *b,Word words) {
    const Word i=(Word)blockIdx.x*blockDim.x+threadIdx.x;if(i>=words)return;
    Word x=i+0x9837b912835ull;
    x=(x^(x>>30))*0xbf58476d1ce4e5b9ull;x=(x^(x>>27))*0x94d049bb133111ebull;x^=x>>31;
    a[i]=x>=GL_P ? x-GL_P : x;x^=0x369dea0f31a53f85ull;
    b[i]=x>=GL_P ? x-GL_P : x;
}
__global__ void outer_plan_compare(const Word *a,const Word *b,const Word *ra,const Word *rb,Word words,Word *bad) {
    const Word i=(Word)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<words){if(a[i]!=ra[i])atomicAdd(bad,1ull);if(b[i]!=rb[i])atomicAdd(bad,1ull);}
}
struct OuterPlan {
    FuseCtx ctx;Word n,omega,inverse,scale;int mode;
    OuterPlan(int k,int selected):n(1ull<<k),omega(gl_pow_host(7,(GL_P-1)/n)),
        inverse(gl_pow_host(omega,GL_P-2)),scale(gl_pow_host(n,GL_P-2)),mode(selected) {
        fuse_fixture_env("NTT_FUSE_COOP_OUTER",mode ? "1" : "2");
        fuse_fixture_env("NTT_FUSE_COOP_M",std::to_string(mode+4).c_str());
        fuse_init(ctx,n,k,omega,inverse);ntt_fuse_cache_tables(ctx,omega,inverse);
        std::string stages;
        for(int i=0;i<ctx.nms;++i){if(i)stages+=",";stages+=std::to_string(ctx.ms[i]);}
        std::printf("outer_plan: mode=%d k=%d t=%d max_m=%d coop=%d passes=%d stages=%s base_table_bytes=%llu\n",
            mode,k,ctx.t,ctx.m_max,(int)ctx.coop_outer,ctx.passes_fwd,stages.c_str(),
            8*(fuse_planned_base_words(ctx)+fuse_planned_table_words(ctx)));
    }
    ~OuterPlan(){fuse_release(ctx);}
    void run(int op,Word *a,Word *b,Word batch) {
        if(op==2){ntt_forward_fused(a,ctx,omega,batch);ntt_forward_fused(b,ctx,omega,batch);}
        if(op==0)ntt_forward_fused(a,ctx,omega,batch);
        else ntt_inverse_fused(a,b,ctx,inverse,scale,batch);
    }
};
static int bench(int k,int batch,int repeats,int iterations,int op,int profile_mode=-1) {
    const Word n=1ull<<k,words=n*batch;
    PlanBuffer a(words),b(words),ra(words),rb(words),bad(1);
    auto fill=[&](){outer_plan_fill<<<(unsigned)((words+255)/256),256>>>(a.p,b.p,words);};
    {
        OuterPlan reference(k,0);fill();
        for(int i=0;i<iterations;++i)reference.run(op,a.p,b.p,batch);
        CK(cudaMemcpy(ra.p,a.p,words*8,cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(rb.p,b.p,words*8,cudaMemcpyDeviceToDevice));
    }
    // Keep every candidate plan alive. Setup and table construction are excluded.
    std::vector<OuterPlan *> plans;
    for(int mode=0;mode<5;++mode)plans.push_back(new OuterPlan(k,mode));
    cudaEvent_t start,end;CK(cudaEventCreate(&start));CK(cudaEventCreate(&end));
    for(int repeat=0;repeat<repeats;++repeat)for(int position=0;position<5;++position) {
        const int mode=(repeat%2 ? 4-position+repeat/2 : position+repeat/2)%5;
        if(profile_mode>=0 && mode!=profile_mode)continue;
        fill();CK(cudaDeviceSynchronize());
        if(profile_mode>=0)CK(cudaProfilerStart());
        CK(cudaEventRecord(start));const auto host_start=std::chrono::steady_clock::now();
        for(int i=0;i<iterations;++i)plans[mode]->run(op,a.p,b.p,batch);
        CK(cudaEventRecord(end));CK(cudaEventSynchronize(end));
        const double host=std::chrono::duration<double>(std::chrono::steady_clock::now()-host_start).count();
        if(profile_mode>=0)CK(cudaProfilerStop());
        float ms=0;CK(cudaEventElapsedTime(&ms,start,end));
        CK(cudaMemset(bad.p,0,8));outer_plan_compare<<<(unsigned)((words+255)/256),256>>>(a.p,b.p,ra.p,rb.p,words,bad.p);
        Word wrong=0;CK(cudaMemcpy(&wrong,bad.p,8,cudaMemcpyDeviceToHost));
        std::printf("outer_plan_sample: repeat=%d position=%d mode=%d k=%d batch=%d operation=%d iterations=%d seconds=%.9f host_seconds=%.9f checked_words=%llu bad=%llu\n",
            repeat,position,mode,k,batch,op,iterations,ms/(1000.0*iterations),host/iterations,2*words,wrong);
        if(wrong)return 3;
    }
    CK(cudaEventDestroy(start));CK(cudaEventDestroy(end));
    for(auto plan:plans)delete plan;
    std::printf("outer_plan_done: live_bytes=%llu\n",g_fuse_base.live_bytes);
    return g_fuse_base.live_bytes ? 3 : 0;
}
static int policy_gate() {
    Word calls=0,bad=0;const bool device=fuse_shape_device_supported();
    for(int enabled:{0,1})for(int tile:{8,12})for(int warp:{0,1})for(int compact:{0,1})
        for(int original:{3,4})for(int mode:{0,1,2})for(int k:{20,21,22,23,24,27,28}) {
        fuse_fixture_env("NTT_FUSE_MID_COOP",std::to_string(enabled).c_str());
        fuse_fixture_env("NTT_FUSE_WARP_TAIL",std::to_string(warp).c_str());
        fuse_fixture_env("NTT_FUSE_COMPACT_SCRATCH",std::to_string(compact).c_str());
        fuse_fixture_env("NTT_FUSE_M",std::to_string(original).c_str());
        fuse_fixture_env("NTT_FUSE_COOP_OUTER",std::to_string(mode).c_str());
        fuse_fixture_env("NTT_FUSE_COOP_M","7");
        bool coop=false;const int maximum=fuse_outer_max(tile,k,coop);
        const bool supported=device && tile==12 && warp && compact && original==4;
        const bool middle=enabled && k>=21 && k<=23;
        const bool automatic=mode==2 && supported && ((k>=24 && k<=27) || middle);
        const bool want=mode==1 || automatic;
        const int m=mode==1 ? 7 : automatic ? ((middle || k==24) ? 6 : 8) : original;
        ++calls;if(coop!=want || maximum!=m)++bad;
    }
    std::printf("outer_policy_gate: calls=%llu bad=%llu\n",calls,bad);return bad ? 3 : 0;
}
int main(int argc,char **argv) {
    if(argc<3)return 2;
    std::setvbuf(stdout,nullptr,_IONBF,0);const int device=std::atoi(argv[1]);CK(cudaSetDevice(device));
    fuse_fixture_env("NTT_FUSE_T","12");fuse_fixture_env("NTT_FUSE_M","4");
    fuse_fixture_env("NTT_FUSE_WARP_TAIL","1");fuse_fixture_env("NTT_FUSE_COMPACT_SCRATCH","1");
    cudaDeviceProp prop{};CK(cudaGetDeviceProperties(&prop,device));
    std::printf("outer_plan_device: device=%d name=%s sm=%d%d sm_count=%d fixed=%d\n",device,prop.name,prop.major,prop.minor,prop.multiProcessorCount,NTT_GL_FIXED_MODE);
    if(!std::strcmp(argv[2],"--policy"))return policy_gate();
    if(!std::strcmp(argv[2],"--check") || !std::strcmp(argv[2],"--fault")) {
        stage2_log::level=stage2_log::debug;
        if(!std::strcmp(argv[2],"--fault"))fuse_fixture_env("NTT_FUSE_COOP_BAD","1");
        ntt_fuse_coop_check(device);return gl_selftest();
    }
    stage2_log::level=stage2_log::quiet;
    const bool profiling=!std::strcmp(argv[2],"--profile");
    if((!std::strcmp(argv[2],"--bench") && argc==8) || (profiling && argc==9)) {
        const int k=std::atoi(argv[3]),batch=std::atoi(argv[4]),repeats=std::atoi(argv[5]),iterations=std::atoi(argv[6]),op=std::atoi(argv[7]);
        const int profile=profiling ? std::atoi(argv[8]) : -1;
        if(k>=16 && k<=25 && batch>=1 && batch<=8 && repeats>=1 && repeats<=101 && iterations>=1 && iterations<=128 && op>=0 && op<=2 && profile>=-1 && profile<5)
            return bench(k,batch,repeats,iterations,op,profile);
    }
    return 2;
}
