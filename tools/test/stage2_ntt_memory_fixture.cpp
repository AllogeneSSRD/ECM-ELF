// The runner inserts the production arena source below. CUDA allocation is
// replaced by an opaque CPU ledger; no driver, context or GPU work is invoked.
#include "../../src/core/ecm_stage2_ntt_memory.h"
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
using namespace ecm_stage2;
static std::map<void *,size_t> heap;
static uintptr_t next_address=4096;
static Word heap_live=0,heap_peak=0,checks=0;
static std::string context;
static void check(bool value) {++checks;if(!value)throw std::runtime_error(context+" at check "+std::to_string(checks));}
static int cudaMalloc(void **p,size_t bytes) {
    // Opaque aligned tokens; never dereferenced, including multi-GiB fixtures.
    *p=reinterpret_cast<void *>(next_address);next_address+=4096;
    heap.emplace(*p,bytes);heap_live+=bytes;heap_peak=std::max(heap_peak,heap_live);return 0;
}
static int cudaFree(void *p) {
    if(!p)return 0;const auto it=heap.find(p);check(it!=heap.end());
    heap_live-=it->second;heap.erase(it);return 0;
}
static int cudaGetLastError(){return 0;}
static int cudaMemGetInfo(size_t *free,size_t *total){*total=size_t(16)<<30;*free=*total-heap_live;return 0;}
static constexpr int cudaSuccess=0,FUSE_MAX_PASSES=1;
#define CK(x) do {if((x)!=0)throw std::runtime_error("mock CUDA error");} while(0)
#define NTT_PROBE_NAME "cpu_arena_fixture"
namespace stage2_log {enum {debug,phases};template<class... T>static void print(T...) {}}
static double now_s(){return 0;}
static Word fuse_env_ull(const char *key,Word fallback) {const auto e=std::getenv(key);return e?std::strtoull(e,nullptr,10):fallback;}
static bool ntt_carry_check_requested(){return fuse_env_ull("NTT_CARRY_CHECK_FUSED",0)!=0;}
static bool fuse_compact_scratch(){return true;}
static void ntt_gl_reduce_configure(){}
static int fuse_outer_max(int,int,bool &coop){coop=false;return 1;}
static void ntt_tile_attributes(bool,int){}
struct FuseCtx {
    Word n=0;int t=0,m_max=1;
    bool tables_cached=false,compact_scratch=true,coop_outer=false,warp_tail=false,arena_borrowed=false;
    Word *passF[1]{},*radF[1]{},*passI[1]{},*radI[1]{},*base=nullptr;
};
static Word fuse_planned_base_words(const FuseCtx &f){return 2*f.n;}
static Word fuse_planned_table_words(const FuseCtx &f){return f.n;}
static Word fuse_base_words(const FuseCtx &f){return f.base?2*f.n:0;}
static void fuse_describe(FuseCtx &f,Word n,int k,Word,Word){f=FuseCtx{};f.n=n;f.t=std::min(k,12);}
static void fuse_init(FuseCtx &f,Word n,int k,Word a,Word b) {
    fuse_describe(f,n,k,a,b);CK(cudaMalloc(reinterpret_cast<void **>(&f.base),16*n));
}
static void ntt_fuse_cache_tables(FuseCtx &f,Word,Word) {
    CK(cudaMalloc(reinterpret_cast<void **>(&f.passF[0]),8*f.n));f.tables_cached=true;
}
static void fuse_release(FuseCtx &f) {
    cudaFree(f.base);for(int i=0;i<FUSE_MAX_PASSES;++i) {
        cudaFree(f.passF[i]);cudaFree(f.radF[i]);cudaFree(f.passI[i]);cudaFree(f.radI[i]);
    }
    f=FuseCtx{};
}
struct {Word live_bytes=0,allocations=0,frees=0,peak_bytes=0;} g_fuse_base;

// @NATIVE_ARENA@

static void flag(const char *key,bool on){_putenv_s(key,on?"1":"0");}
static auto describe=[](Word n,Word &table,Word &base){table=8*n;base=16*n;return true;};
static auto query=[](Word m,int,Word *n,Word *slots){*n=8;while(*n<4*m)*n*=2;*slots=2*m-1;return true;};
static void compare(const NttMemoryState &s,const NttArena &a,bool compare_peak=true) {
    const auto p=a.payload();
    check(s.live.big==p.big && s.live.digits==p.digits && s.live.table==p.table && s.live.base==p.base && s.live.total==p.total);
    check(a.bytes==p.total && heap_live==p.total);
    if(compare_peak)check(s.peak==a.peak_full_bytes && s.peak==heap_peak);
    check(s.counters.fuse_builds==a.fuse_builds && s.counters.fuse_hits==a.fuse_hits);
    check(s.counters.workspace_grows==a.workspace_grows && s.counters.cap_evictions==a.tbl_evictions && s.counters.cap_evicted_bytes==a.tbl_words_freed*8);
    check(s.counters.cold_evictions==a.phase_fuse_evictions && s.counters.cold_evicted_bytes==a.phase_fuse_evicted_bytes);
    check(s.counters.carry_grows==a.carry_grows && s.counters.carry_refusals==a.carry_refusals);
}
static bool native_call(NttArena &a,Word n,Word slots,Word slices) {
    int k=0;for(auto v=n;v>1;v/=2)++k;
    FuseCtx f;ntt_arena_fuse(&a,n,k,0,0,f);
    if(!f.arena_borrowed){fuse_release(f);return false;}
    if(!ntt_arena_bufs(&a,n,slots,slices))return false;
    for(Word offset=0;offset<slices;) {
        const Word batch=std::min(Word(65535),slices-offset);
        a.carry_workspace(n,batch);offset+=batch;
    }
    return true;
}
static void policy(NttArena &a,const NttMemoryPolicy &p) {
    a.workspace_pool=p.pool;a.workspace_reuse_bq=p.reuse_bq;a.cap_bytes=p.cap_bytes;
    flag("NTT_CARRY_CHECK_FUSED",p.carry_check);
}
static void reset_heap(){check(heap.empty() && !heap_live);heap_peak=0;}
static Word random_state=0x953db71217ull;
static Word random_word(){random_state^=random_state<<13;random_state^=random_state>>7;random_state^=random_state<<17;return random_state;}
int main() {
    try {
        // Growth evicts a cold table and digits, retaining its base. A later
        // hit MUST NOT recache the table. Full cold trim permits reconstruction.
        context="cap eviction and reconstruction";reset_heap();
        {
            NttMemoryPolicy p{12500,true,true,false};NttMemoryState s(p);NttArena a;policy(a,p);
            for(const auto &request:std::vector<std::vector<Word>>{{64,100,1},{128,1,4},{64,1,1}}) {
                check(s.call(request[0],request[1],request[2],describe));
                check(native_call(a,request[0],request[1],request[2]));compare(s,a);
            }
            check(s.counters.cap_evictions==1 && s.counters.cap_evicted_bytes==1328 && s.live.table==1024 && s.live.base==3072);
            Word available=0;check(s.cold_trim(64,3072,available) && available==3072);
            check(a.drop_cold_fuse(64)==3072);compare(s,a);
            check(s.call(128,1,1,describe) && native_call(a,128,1,1));compare(s,a);
            check(s.counters.fuse_builds==3);a.release();
        }
        context="carry scratch cap refusal";reset_heap();
        {
            const Word n=Word(1)<<20;NttMemoryPolicy p{40*n+100,true,true,true};
            NttMemoryState s(p);NttArena a;policy(a,p);
            check(s.call(n,1,1,describe) && native_call(a,n,1,1));compare(s,a);
            check(s.counters.carry_refusals==1 && s.counters.carry_grows==0);a.release();
        }
        context="gridDim.y split carry scratch";
        for(Word n:{Word(16),Word(32)})for(Word slices:{Word(65535),Word(65536),Word(131071)}) {
            reset_heap();NttMemoryPolicy p{0,true,true,true};NttMemoryState s(p);NttArena a;policy(a,p);
            check(s.call(n,1,slices,describe) && native_call(a,n,1,slices));compare(s,a);
            if(n==16)check(s.counters.carry_grows==0);
            else check(s.counters.carry_grows==1 && a.carry_scratch_bytes==8*65535);
            a.release();
        }
        context="buffer size overflow before allocation";
        for(bool pool:{false,true})for(bool reuse:{false,true})for(unsigned kind=0;kind<2;++kind) {
            reset_heap();NttMemoryPolicy p{0,pool,reuse,false};NttMemoryState s(p);NttArena a;policy(a,p);
            const Word slots=kind?std::numeric_limits<size_t>::max()/8-1:1;
            const Word slices=kind?1:std::numeric_limits<size_t>::max()/192+1;
            check(!s.call(8,slots,slices,describe) && !native_call(a,8,slots,slices));compare(s,a);
            check(!std::strcmp(s.reason,"invalid_buffer_size"));a.release();
        }
        // Direct transitions use the ACTUAL arena/cap/eviction implementation.
        for(bool pool:{false,true})for(bool reuse:{false,true})for(bool carry:{false,true})
            for(unsigned trial=0;trial<100;++trial) {
                reset_heap();NttMemoryPolicy p{trial%3?Word(0):Word(100000+trial*70000),pool,reuse,carry};
                NttMemoryState s(p);NttArena a;policy(a,p);
                context="random pool="+std::to_string(pool)+" reuse="+std::to_string(reuse)+" carry="+std::to_string(carry)+" trial="+std::to_string(trial);
                for(unsigned i=0;i<100;++i) {
                    const Word n=Word(1)<<(3+random_word()%18),slots=1+random_word()%std::min(Word(10000),n),slices=1+random_word()%23;
                    const bool predicted=s.call(n,slots,slices,describe),actual=native_call(a,n,slots,slices);
                    check(predicted==actual);compare(s,a,actual);
                    if(!actual)break;
                    // Compare logical cold headroom to the native first-largest
                    // eviction, including ties and full context reconstruction.
                    if(i%11==10) {
                        Word available=0,required=1000+random_word()%100000;
                        check(s.cold_trim(n,required,available));
                        Word real_available=0;while(real_available<required) {
                            const Word freed=a.drop_cold_fuse(n);if(!freed)break;real_available+=freed;
                        }
                        check(available==real_available);compare(s,a);
                    }
                }
                a.release();check(!heap_live && heap.empty());
            }
        // Full eager arena execution checks compressed program/chunk traversal.
        for(Word p:{Word(1),Word(7),Word(48),Word(129),Word(257)})
            for(bool pool:{false,true})for(bool reuse:{false,true})for(bool physical:{false,true})
            for(Word cap:{Word(0),Word(20000),Word(50000),Word(100000)}) {
                reset_heap();NttMemoryPolicy config{cap,pool,reuse,false};
                RequestProgram program;check(request_program(p,20*p+std::max(Word(1),p/3),program));
                NttMemoryPlan plan;check(ntt_memory_plan(program,129,query,describe,16384,physical,0,config,plan));
                NttArena actual;policy(actual,config);
                context="program P="+std::to_string(p)+" pool="+std::to_string(pool);
                Word calls=0;bool success=true;
                auto eager=[&]() {
                  for(const auto &block:program.blocks)for(Word rep=0;rep<block.repeat;++rep)
                    for(const auto &r:block.requests) {
                        Word n=0,slots=0;query(std::max(r.ma,r.mb),129,&n,&slots);
                        Word c=chunk_slices(n,slots,r.pairs,16384,physical?(pool && reuse?2:3):3,physical);
                        for(Word offset=0;offset<r.pairs;offset+=c) {
                            ++calls;if(!native_call(actual,n,slots,std::min(c,r.pairs-offset)))return false;
                        }
                    }
                  return true;
                };
                success=eager();const auto payload=actual.payload();
                check(plan.valid && plan.finished==success && plan.peak_bytes==actual.peak_full_bytes);
                if(success)check(plan.peak_bytes==heap_peak);
                check(plan.final_payload.total==payload.total && plan.final_payload.big==payload.big && plan.final_payload.digits==payload.digits && plan.final_payload.table==payload.table && plan.final_payload.base==payload.base);
                check(plan.counters.fuse_builds==actual.fuse_builds && plan.counters.fuse_hits==actual.fuse_hits && plan.counters.workspace_grows==actual.workspace_grows);
                check(plan.counters.calls==calls && plan.counters.cap_evictions==actual.tbl_evictions && plan.counters.cap_evicted_bytes==actual.tbl_words_freed*8);
                if(!success)check(plan.stop.block<program.blocks.size() && plan.stop.n>0 && plan.stop.slices>0);
                actual.release();check(!heap_live);
            }
        // Huge repeated work remains bounded; native eager traversal above proves
        // the same cycle compression for smaller repetitions.
        context="huge repeat";RequestProgram huge;check(request_program(48,Word(1)<<60,huge));
        NttMemoryPlan plan;check(ntt_memory_plan(huge,129,query,describe,16384,true,0,{0,true,true,true},plan));
        check(plan.valid && plan.finished && plan.executed_blocks<20 && plan.skipped_blocks>(Word(1)<<50));
        context="overflow";RequestProgram over;over.supported=true;over.blocks={RequestBlock{~Word(0),{{RequestFold,1,1,3,0,1}}}};
        check(!ntt_memory_plan(over,129,query,describe,16384,true,1,{},plan));
        context="unsupported";RequestProgram no;check(ntt_memory_plan(no,129,query,describe,16384,true,0,{},plan) && !plan.valid && !plan.finished);
        context="empty repeat";over.blocks[0].repeat=0;check(!ntt_memory_plan(over,129,query,describe,16384,true,0,{},plan));
        std::cout<<"{\"checks\":"<<checks<<",\"bad\":0,\"gpu_calls\":0}\n";return 0;
    } catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
