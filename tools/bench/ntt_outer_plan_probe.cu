// Same-binary outer-plan study using the production runtime and arithmetic.
#include <algorithm>
#include <stdexcept>
#include "../../src/ecm/platform/legacy/ecm_stage2_logging.h"
#include "../../src/ecm/backend/cuda/ntt/runtime.h"
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
    OuterPlan(int k,int selected,int tile=12,int maximum=0):n(1ull<<k),omega(gl_pow_host(7,(GL_P-1)/n)),
        inverse(gl_pow_host(omega,GL_P-2)),scale(gl_pow_host(n,GL_P-2)),mode(selected) {
        fuse_fixture_env("NTT_FUSE_COOP_OUTER",mode ? "1" : "2");
        fuse_fixture_env("NTT_FUSE_T",std::to_string(std::min(k,tile)).c_str());
        fuse_fixture_env("NTT_FUSE_COOP_M",std::to_string(maximum ? maximum : mode+4).c_str());
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
static int bench(int k,int batch,int repeats,int iterations,int op,int profile_mode=-1,bool joint=false) {
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
    const int modes=joint ? 17 : 5;
    for(int mode=0;mode<modes;++mode)plans.push_back(joint && mode ?
        new OuterPlan(k,mode,10+(mode-1)/4,5+(mode-1)%4) : new OuterPlan(k,mode));
    // CUDA function attributes are shared by every plan. A smaller plan must
    // not invalidate a larger tile's launch; actual dynamic bytes stay per-CTA.
    ntt_tile_attributes(true,8*(1<<std::min(k,joint ? 13 : 12)),true);
    cudaEvent_t start,end;CK(cudaEventCreate(&start));CK(cudaEventCreate(&end));
    for(int repeat=0;repeat<repeats;++repeat)for(int position=0;position<modes;++position) {
        const int mode=joint ? (2*modes+5*repeat+(repeat%2 ? -position : position))%modes :
            (repeat%2 ? 4-position+repeat/2 : position+repeat/2)%5;
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
// Independent dense DIF/DIT oracle. Roots, products, sums and scaling use GMP.
static std::vector<Word> joint_gmp(std::vector<Word> a,const std::vector<Word> &b,int k,bool inverse,int batch) {
    mpz_t p,x,y,z,w,exponent;mpz_inits(p,x,y,z,w,exponent,nullptr);
    const Word prime=GL_P,n=1ull<<k;mpz_import(p,1,-1,8,0,0,&prime);mpz_set_ui(x,7);
    const Word e=(prime-1)/n;mpz_import(exponent,1,-1,8,0,0,&e);mpz_powm(w,x,exponent,p);
    if(inverse)mpz_invert(w,w,p);
    auto word=[&](mpz_t v){Word out=0;size_t count=0;mpz_export(&out,&count,-1,8,0,0,v);return out;};
    const Word root=word(w);
    auto mul=[&](Word u,Word v){mpz_import(x,1,-1,8,0,0,&u);mpz_import(y,1,-1,8,0,0,&v);
        mpz_mul(z,x,y);mpz_mod(z,z,p);return word(z);};
    std::vector<Word> powers(n/2,1);for(size_t j=1;j<powers.size();++j)powers[j]=mul(powers[j-1],root);
    if(inverse) {
        mpz_import(x,1,-1,8,0,0,&n);mpz_invert(z,x,p);const Word scale=word(z);
        for(size_t j=0;j<a.size();++j)a[j]=mul(mul(a[j],b[j]),scale);
    }
    for(int level=0;level<k;++level) {
        const int st=inverse ? level : k-level-1;const Word half=1ull<<st,len=2*half;
        for(int sl=0;sl<batch;++sl)for(Word base=0;base<n;base+=len)for(Word j=0;j<half;++j) {
            const size_t lo=sl*n+base+j,hi=lo+half;const Word u=a[lo],v=inverse ? mul(a[hi],powers[j*(n/len)]) : a[hi];
            mpz_import(x,1,-1,8,0,0,&u);mpz_import(y,1,-1,8,0,0,&v);
            mpz_add(z,x,y);mpz_mod(z,z,p);a[lo]=word(z);mpz_sub(z,x,y);mpz_mod(z,z,p);a[hi]=word(z);
            if(!inverse)a[hi]=mul(a[hi],powers[j*(n/len)]);
        }
    }
    mpz_clears(p,x,y,z,w,exponent,nullptr);return a;
}
static int joint_gate(bool fault) {
    Word bad=0,checked=0,cases=0,seed=0x639172385ull;bool injected=false;
    for(int k:{13,15}) {
        const int batch=3;const Word n=1ull<<k,words=n*batch;
        std::vector<Word> input(words),multiplier(words),got(words),got_b(words);
        for(size_t j=0;j<words;++j) {
            seed^=seed<<13;seed^=seed>>7;seed^=seed<<17;
            input[j]=j%17==0 ? GL_P-1 : j%19==0 ? 0 : seed%GL_P;
            multiplier[j]=(seed^0x729384987ull)%GL_P;
        }
        const auto forward=joint_gmp(input,multiplier,k,false,batch);
        const auto bf=joint_gmp(multiplier,input,k,false,batch);
        const auto inverse=joint_gmp(input,multiplier,k,true,batch);
        const auto convolution=joint_gmp(forward,bf,k,true,batch);
        const std::vector<Word> *expected[]={&forward,&inverse,&convolution};
        PlanBuffer a(words),b(words);
        for(int mode=0;mode<17;++mode) {
            OuterPlan plan(k,mode,mode ? 10+(mode-1)/4 : 12,mode ? 5+(mode-1)%4 : 0);
            for(int op=0;op<3;++op) {
                CK(cudaMemcpy(a.p,input.data(),words*8,cudaMemcpyHostToDevice));
                CK(cudaMemcpy(b.p,multiplier.data(),words*8,cudaMemcpyHostToDevice));plan.run(op,a.p,b.p,batch);
                CK(cudaMemcpy(got.data(),a.p,words*8,cudaMemcpyDeviceToHost));
                CK(cudaMemcpy(got_b.data(),b.p,words*8,cudaMemcpyDeviceToHost));
                if(fault && mode==16 && !injected){got[0]^=1;injected=true;}
                for(size_t j=0;j<words;++j){bad+=got[j]!=(*expected[op])[j];bad+=got_b[j]!=(op==2 ? bf[j] : multiplier[j]);}
                checked+=2*words;++cases;
            }
        }
    }
    std::printf("joint_gmp_gate: cases=%llu words=%llu bad=%llu fault=%d injected=%d live=%llu\n",cases,checked,bad,(int)fault,(int)injected,g_fuse_base.live_bytes);
    return bad || g_fuse_base.live_bytes ? 3 : 0;
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
__global__ void joint_poison(Word *p){if(!threadIdx.x && !blockIdx.x)p[0]^=1;}
static int joint_policy_gate(bool fault) {
    Word bad=0,calls=0,checked=0;const bool device=fuse_shape_device_supported();
    for(int joint:{0,1})for(int middle:{0,1})for(int tile:{8,11,12})for(int warp:{0,1})
        for(int compact:{0,1})for(int original:{3,4})for(int mode:{0,1,2})for(int k=20;k<=28;++k) {
        for(auto item:{std::pair<const char*,int>{"NTT_FUSE_JOINT",joint},{"NTT_FUSE_MID_COOP",middle},
            {"NTT_FUSE_T",tile},{"NTT_FUSE_WARP_TAIL",warp},{"NTT_FUSE_COMPACT_SCRATCH",compact},
            {"NTT_FUSE_M",original},{"NTT_FUSE_COOP_OUTER",mode},{"NTT_FUSE_COOP_M",7}})
            fuse_fixture_env(item.first,std::to_string(item.second).c_str());
        const bool supported=device && tile==12 && warp && compact && original==4;
        const bool candidate=joint && mode==2 && supported;
        const int want_t=candidate && k==23 ? 11 : tile;
        const bool automatic=mode==2 && supported && ((k>=24 && k<=27) || ((middle || joint) && k>=21 && k<=23));
        const bool coop=mode==1 || automatic;
        const int m=mode==1 ? 7 : automatic ? (k<=24 ? 6 : candidate && k==25 ? 7 : 8) : original;
        FuseCtx ctx;fuse_describe(ctx,1ull<<k,k,0,0);
        ++calls;if(ctx.t!=want_t || ctx.m_max!=m || ctx.coop_outer!=coop || ctx.outer_stages!=k-want_t)++bad;
    }
    fuse_fixture_env("NTT_FUSE_T","12");fuse_fixture_env("NTT_FUSE_M","4");
    fuse_fixture_env("NTT_FUSE_WARP_TAIL","1");fuse_fixture_env("NTT_FUSE_COMPACT_SCRATCH","1");
    fuse_fixture_env("NTT_FUSE_MID_COOP","1");bool injected=false;
    for(int k:{21,23,25}) {
        const Word n=1ull<<k;PlanBuffer a(n),b(n),ra(n),rb(n),wrong(1);
        auto fill=[&](){outer_plan_fill<<<(unsigned)((n+255)/256),256>>>(a.p,b.p,n);};
        fuse_fixture_env("NTT_FUSE_JOINT","0");
        {OuterPlan reference(k,0);fill();reference.run(2,a.p,b.p,1);
            CK(cudaMemcpy(ra.p,a.p,n*8,cudaMemcpyDeviceToDevice));CK(cudaMemcpy(rb.p,b.p,n*8,cudaMemcpyDeviceToDevice));}
        {
            NttArena arena;int dev=0;CK(cudaGetDevice(&dev));arena.device=dev;
            for(int selected:{0,1,0,1}) {
                fuse_fixture_env("NTT_FUSE_JOINT",std::to_string(selected).c_str());
                const Word omega=gl_pow_host(7,(GL_P-1)/n),inverse=gl_pow_host(omega,GL_P-2),scale=gl_pow_host(n,GL_P-2);
                FuseCtx ctx;ntt_arena_fuse(&arena,n,k,omega,inverse,ctx);
                if(!ctx.arena_borrowed || ctx.t!=(device && selected && k==23 ? 11 : 12))++bad;
                fill();ntt_forward_fused(a.p,ctx,omega);ntt_forward_fused(b.p,ctx,omega);ntt_inverse_fused(a.p,b.p,ctx,inverse,scale);
                if(fault && selected && k==23 && !injected){joint_poison<<<1,1>>>(a.p);injected=true;}
                CK(cudaMemset(wrong.p,0,8));outer_plan_compare<<<(unsigned)((n+255)/256),256>>>(a.p,b.p,ra.p,rb.p,n,wrong.p);
                Word count=0;CK(cudaMemcpy(&count,wrong.p,8,cudaMemcpyDeviceToHost));bad+=count;checked+=2*n;
            }
        }
    }
    std::printf("joint_policy_gate: calls=%llu words=%llu bad=%llu fault=%d injected=%d live=%llu\n",calls,checked,bad,(int)fault,(int)injected,g_fuse_base.live_bytes);
    return bad || g_fuse_base.live_bytes ? 3 : 0;
}
int main(int argc,char **argv) {
    if(argc<3)return 2;
    std::setvbuf(stdout,nullptr,_IONBF,0);const int device=std::atoi(argv[1]);CK(cudaSetDevice(device));
    fuse_fixture_env("NTT_FUSE_T","12");fuse_fixture_env("NTT_FUSE_M","4");
    fuse_fixture_env("NTT_FUSE_WARP_TAIL","1");fuse_fixture_env("NTT_FUSE_COMPACT_SCRATCH","1");
    cudaDeviceProp prop{};CK(cudaGetDeviceProperties(&prop,device));
    std::printf("outer_plan_device: device=%d name=%s sm=%d%d sm_count=%d fixed=%d\n",device,prop.name,prop.major,prop.minor,prop.multiProcessorCount,NTT_GL_FIXED_MODE);
    if(!std::strcmp(argv[2],"--policy"))return policy_gate();
    if(!std::strcmp(argv[2],"--joint-policy") || !std::strcmp(argv[2],"--joint-policy-fault"))return joint_policy_gate(!std::strcmp(argv[2],"--joint-policy-fault"));
    if(!std::strcmp(argv[2],"--joint-check") || !std::strcmp(argv[2],"--joint-fault"))return joint_gate(!std::strcmp(argv[2],"--joint-fault"));
    if(!std::strcmp(argv[2],"--check") || !std::strcmp(argv[2],"--fault")) {
        stage2_log::level=stage2_log::debug;
        if(!std::strcmp(argv[2],"--fault"))fuse_fixture_env("NTT_FUSE_COOP_BAD","1");
        ntt_fuse_coop_check(device);return gl_selftest();
    }
    stage2_log::level=stage2_log::quiet;
    const bool joint=!std::strcmp(argv[2],"--joint") || !std::strcmp(argv[2],"--joint-profile");
    const bool profiling=!std::strcmp(argv[2],"--profile") || !std::strcmp(argv[2],"--joint-profile");
    if(((!std::strcmp(argv[2],"--bench") || !std::strcmp(argv[2],"--joint")) && argc==8) || (profiling && argc==9)) {
        const int k=std::atoi(argv[3]),batch=std::atoi(argv[4]),repeats=std::atoi(argv[5]),iterations=std::atoi(argv[6]),op=std::atoi(argv[7]);
        const int profile=profiling ? std::atoi(argv[8]) : -1;
        if(profiling && profile<0)return 2;
        if(k>=16 && k<=27 && batch>=1 && batch<=8 && ((1ull<<k)*batch<=(1ull<<27)) && repeats>=1 && repeats<=101 && iterations>=1 && iterations<=128 && op>=0 && op<=2 && profile>=-1 && profile<(joint ? 17 : 5))
            return bench(k,batch,repeats,iterations,op,profile,joint);
    }
    return 2;
}
