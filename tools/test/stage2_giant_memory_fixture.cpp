// Runner inserts the production S3 allocator. All allocations are opaque CPU
// tokens. Arithmetic and kernel launches are stubbed, never tested by this file.
#include "../../src/core/ecm_stage2_giant_memory.h"
#include <iostream>
#include <string>
#include <stdexcept>
#include <cstdint>
#include <cstdlib>
using namespace ecm_stage2;
static std::map<void *,Word> heap;
static Word live=0,peak=0,checks=0;static uintptr_t address=4096;
static void check(bool v) {++checks;if(!v)throw std::runtime_error("check "+std::to_string(checks));}
template<class T>static int cudaMalloc(T **p,size_t bytes) {
    *p=reinterpret_cast<T *>(address);address+=4096;
    heap[*p]=bytes;live+=bytes;peak=std::max(peak,live);return 0;
}
static int cudaFree(void *p) {
    if(!p)return 0;auto it=heap.find(p);check(it!=heap.end());live-=it->second;heap.erase(it);return 0;
}
template<class... T>static int cudaMemcpy(T...){return 0;}
static int cudaGetLastError(){return 0;}
static constexpr int cudaMemcpyHostToDevice=0,cudaMemcpyDeviceToHost=1;
#define CK(x) do{check((x)==0);}while(0)
#define S2G_DISPATCH(...) do{}while(0)
using mpz_t=int[1];using mpz_srcptr=const int *;
template<class... T>static void mpz_inits(T...){}
template<class... T>static void mpz_clears(T...){}
template<class... T>static void mpz_set_ui(T...){}
template<class... T>static void mpz_mul(T...){}
template<class... T>static void mpz_mod(T...){}
template<class... T>static void mpz_gcd(T...){}
template<class... T>static int mpz_cmp_ui(T...){return 0;}
template<class... T>static void words_to_mpz(T...){}
template<class... T>static void mpz_to_words(T...){}
static double now_s(){return 0;}
static bool compact_products=true;
static bool s3_compact_products_requested(){return compact_products;}
struct LadderCtx {
    size_t nw;Word ninv=0;std::vector<Word> hn,hqx,hqz,ha24,hmone;
    explicit LadderCtx(size_t w):nw(w),hn(w),hqx(w),hqz(w),ha24(w),hmone(w){}
};
struct {Word base_gpu_builds=0,base_d2h_bytes=0,base_builds=0,base_nonunits=0;
    double base_build_seconds=0;} g_giant_seed;

// @NATIVE_S3@

static void workspace_check(const S3Workspace &ws) {
    Word bytes=0;check(giant_workspace_bytes(ws.C->nw,ws.pt_cap,ws.val_cap,ws.prod_cap,
        ws.segfix_cap,ws.giant_base!=nullptr,bytes));
    if(ws.bytes!=bytes)throw std::runtime_error("S3 bytes counter includes released capacities");
    check(ws.bytes==bytes);check(live==bytes);
}
static void one(Word p,Word n,Word w,Word q,GiantMemoryPolicy policy) {
    compact_products=policy.compact_products;
    check(heap.empty());live=peak=0;
    GiantMemoryPlan plan;check(giant_memory_plan(p,n,w,q,policy,plan));
    LadderCtx ctx{size_t(w)};
    {
        S3Workspace ws;ws.init(ctx);ws.need_pts(size_t(policy.initial_points));workspace_check(ws);
        check(live==plan.initial_bytes);
        Word offset=0;
        for(size_t index=0;index<plan.chunks.size();++index) {
            const auto &c=plan.chunks[index];
            for(Word rep=0;rep<c.repeat;++rep) {
                const Word points=std::min(q,n-offset);
                const bool chain=!policy.force_ladder && points>=policy.chain_min;
                const Word block=policy.short_block && points<policy.short_max && policy.short_block<policy.chain_block?
                    policy.short_block:policy.chain_block;
                const Word chains=points/block+(points%block!=0),seeds=chain?2*chains+1:points;
                ws.need_pts(size_t(seeds));
                if(chain && policy.seed_device && policy.seed_pair)ws.need_giant_base(1,nullptr);
                const Word ns=points/policy.segment+(points%policy.segment!=0);
                const bool resident=policy.resident_requested && policy.resident_eligible && policy.exact_segments &&
                    16*points*w<=policy.resident_limit_bytes;
                std::vector<void *> temporary,legacy,groups;
                auto alloc=[&](std::vector<void *> &v,Word bytes){void *ptr=nullptr;CK(cudaMalloc(&ptr,size_t(bytes)));v.push_back(ptr);};
                if(chain && !policy.seed_device) {
                    for(int k=0;k<4;++k)alloc(legacy,chains*w*8);
                    for(int k=0;k<2;++k)alloc(legacy,w*8);
                }
                if(chain){alloc(temporary,points*w*8);alloc(temporary,points*w*8);}
                if(chain || resident) {
                    alloc(temporary,ns*w*8);
                    if(policy.exact_segments)ws.need_segfix(size_t(policy.segment));
                }
                check(chain==c.chain && seeds==c.seeds && resident==c.resident);
                check(live==c.prepare_bytes);
                for(auto ptr:legacy)CK(cudaFree(ptr));
                if(!resident){for(auto ptr:temporary)CK(cudaFree(ptr));temporary.clear();}
                else {
                    const Word ng=ns/policy.group+(ns%policy.group!=0);
                    alloc(groups,ng*w*8);alloc(groups,(policy.group+1)*w*8);
                }
                check(live==c.tree_bytes);
                for(auto ptr:groups)CK(cudaFree(ptr));
                for(auto ptr:temporary)CK(cudaFree(ptr));
                workspace_check(ws);check(live==c.workspace_bytes);
                offset+=points;
            }
        }
        check(offset==n && live==plan.after_giant_bytes && ws.pt_cap==plan.final_point_capacity);
        ws.need_vals(size_t(p));workspace_check(ws);
#ifndef S3_TEST_LEGACY_SOURCE
        ws.need_products(size_t(policy.compact_products?(p+policy.accumulation_block-1)/policy.accumulation_block:p));
#endif
        workspace_check(ws);check(live==plan.accumulation_bytes);
        check(peak==plan.peak_bytes);
        // Grow repeatedly and ensure the production bytes counter drops old caps.
        ws.need_pts(size_t(ws.pt_cap+1));workspace_check(ws);
        ws.need_vals(size_t(p+1));workspace_check(ws);
#ifndef S3_TEST_LEGACY_SOURCE
        ws.need_products(size_t(ws.prod_cap+1));workspace_check(ws);
        ws.need_vals(size_t(ws.val_cap+2));workspace_check(ws);
#endif
        ws.need_pts(1);ws.need_vals(1);workspace_check(ws);
    }
    check(heap.empty() && !live);
}
int main(int argc,char **argv) {
    try {
        if(argc==7) {
            GiantMemoryPlan p;GiantMemoryPolicy v;v.initial_points=std::strtoull(argv[5],nullptr,10);
            v.compact_products=std::atoi(argv[6])!=0;
            check(giant_memory_plan(std::strtoull(argv[1],nullptr,10),std::strtoull(argv[2],nullptr,10),
                std::strtoull(argv[3],nullptr,10),std::strtoull(argv[4],nullptr,10),v,p));
            Word giant_peak=0;for(const auto &c:p.chunks)giant_peak=std::max(giant_peak,std::max(c.prepare_bytes,c.tree_bytes));
            std::cout<<"{\"initial\":"<<p.initial_bytes<<",\"after_giant\":"<<p.after_giant_bytes
                <<",\"accumulation\":"<<p.accumulation_bytes<<",\"peak\":"<<p.peak_bytes
                <<",\"giant_peak\":"<<giant_peak<<",\"point_capacity\":"<<p.final_point_capacity<<"}\n";return 0;
        }
        for(unsigned flags=0;flags<128;++flags)for(Word w:{1ull,7ull,126ull,256ull})
        for(Word p:{1ull,17ull,64ull,8192ull}) {
            GiantMemoryPolicy v;v.chain_min=128;v.chain_block=64;
            v.force_ladder=flags&1;v.seed_device=flags&2;v.exact_segments=flags&4;
            v.resident_requested=flags&8;v.resident_eligible=flags&16;
            v.short_block=(flags&32)?4:0;v.short_max=129;v.initial_points=flags%7;
            v.resident_limit_bytes=16*p*w;
            v.compact_products=flags&64;
            one(p,3*p+1,w,2*p,v);
        }
        GiantMemoryPolicy v;v.initial_points=1;
        one(126720,1882177,126,253440,v);
        one(126720,253440+32000,126,253440,v); // chain -> ladder tail retains larger seed cap
        v.seed_device=false;one(126720,1882177,126,126720,v);
        GiantMemoryPlan huge;v.seed_device=true;
        check(giant_memory_plan(1,9000000000000000000ull,1,32768,v,huge));check(huge.chunks.size()<=2);
        check(!giant_memory_plan(0,1,1,1,v,huge));check(!giant_memory_plan(1,1,0,1,v,huge));
        check(!giant_memory_plan(1,1,257,1,v,huge));check(!giant_memory_plan(17,20,1,18,v,huge));
        check(!giant_memory_plan(1,~Word(0),256,~Word(0),v,huge));
        v.segment=0;check(!giant_memory_plan(1,1,1,1,v,huge));
        std::cout<<"{\"checks\":"<<checks<<",\"bad\":0,\"gpu_calls\":0}\n";return 0;
    } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
