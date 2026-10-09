// The runner inserts native FuseCtx/planning/allocation and arena statements.
// Kernel launches are removed. CUDA allocations are opaque CPU tokens only.
#include "../../src/core/ecm_stage2_ntt_memory.h"
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
using namespace ecm_stage2;
static Word checks=0,cases=0,event_checks=0;
static std::string context;
static void check(bool v,const char *why="state") {
    ++checks;if(!v)throw std::runtime_error(context+": "+why+" check="+std::to_string(checks));
}
struct NativeAllocation {Word bytes;NttMemorySite site;unsigned index;};
struct NativeEvent {Word bytes;NttMemorySite site;unsigned index;bool allocation;NttMemoryPayload live;};
static std::map<void *,NativeAllocation> heap;
static std::vector<NativeEvent> actual;
static std::vector<NttMemoryEvent> predicted;
static uintptr_t next_address=4096;
static NttMemoryPayload native_live;
static Word native_peak=0;
static NttMemorySite native_site=NttCarry;
static unsigned native_index=0;
static void record(NativeAllocation a,bool alloc) {
    Word *part=nullptr;
    switch(a.site) {
        case NttBaseTileF:case NttBaseTileI:case NttBaseScratch:case NttBaseRadix:part=&native_live.base;break;
        case NttTablePassF:case NttTableRadF:case NttTablePassI:case NttTableRadI:part=&native_live.table;break;
        case NttWorkspaceA:case NttWorkspaceB:case NttWorkspaceQ:
        case NttKeyedA:case NttKeyedB:case NttKeyedQ:part=&native_live.big;break;
        case NttDigitsOutput:case NttDigitsResult:case NttCarry:part=&native_live.digits;break;
        default:check(false,"native site");
    }
    if(alloc){*part+=a.bytes;native_live.total+=a.bytes;}
    else {check(*part>=a.bytes && native_live.total>=a.bytes,"native underflow");*part-=a.bytes;native_live.total-=a.bytes;}
    native_peak=std::max(native_peak,native_live.total);
    actual.push_back({a.bytes,a.site,a.index,alloc,native_live});
}
template<class T> static int cudaMalloc(T **p,size_t bytes) {
    check(bytes>0,"zero native allocation");*p=reinterpret_cast<T *>(next_address);next_address+=4096;
    NativeAllocation a{Word(bytes),native_site,native_index};check(heap.emplace(*p,a).second,"duplicate pointer");
    record(a,true);return 0;
}
static int cudaFree(void *p) {
    if(!p)return 0;auto it=heap.find(p);check(it!=heap.end(),"unknown free");record(it->second,false);heap.erase(it);return 0;
}
static int cudaGetLastError(){return 0;}
static int cudaDeviceSynchronize(){return 0;}
static int cudaMemGetInfo(size_t *free,size_t *total){*total=size_t(64)<<30;*free=*total-native_live.total;return 0;}
static constexpr int cudaSuccess=0,FUSE_MAX_M=4,FUSE_MAX_PASSES=8;
#define CK(x) do {if((x)!=0)throw std::runtime_error("mock CUDA failure");} while(0)
#define NTT_PROBE_NAME "cpu_ntt_events"
namespace stage2_log {enum {debug,phases};template<class... T>static void print(T...) {}}
static double now_s(){return 0;}
static void ntt_gl_reduce_configure(){}
static void ntt_tile_attributes(bool,int,bool=false){}
static Word gl_pow_host(Word,Word){return 1;}
static bool ntt_carry_check_requested(){const auto e=std::getenv("NTT_CARRY_CHECK_FUSED");return e && std::atoi(e)!=0;}

// @NATIVE_FUSE@
// @NATIVE_ARENA@

static void flag(const char *key,bool on){_putenv_s(key,on?"1":"0");}
static void value(const char *key,Word v){_putenv_s(key,std::to_string(v).c_str());}
static auto describe=[](Word n,NttFuseMemoryLayout &layout) {
    int k=0;for(Word v=n;v>1;v/=2)++k;
    FuseCtx fc;fuse_describe(fc,n,k,0,0);
    if(!ntt_fuse_memory_layout(fc,layout))return false;
    Word base=0,table=0;check(layout.totals(base,table));
    check(base==fuse_planned_base_words(fc)*8 && table==fuse_planned_table_words(fc)*8,"layout totals");
    return true;
};
static bool native_call(NttArena &a,Word n,Word slots,Word slices) {
    int k=0;for(Word v=n;v>1;v/=2)++k;
    FuseCtx fc;const size_t start=actual.size();ntt_arena_fuse(&a,n,k,0,0,fc);
    if(!fc.arena_borrowed) {
        // The model stops at arena refusal; per-call fallback is outside its
        // contract. Preserve the successful prefix and exclude this transient.
        fuse_release(fc);actual.resize(start);return false;
    }
    if(!ntt_arena_bufs(&a,n,slots,slices))return false;
    for(Word offset=0;offset<slices;) {const Word batch=std::min(Word(65535),slices-offset);a.carry_workspace(n,batch);offset+=batch;}
    return true;
}
static bool equal(const NttMemoryPayload &a,const NttMemoryPayload &b) {
    return a.big==b.big && a.digits==b.digits && a.table==b.table && a.base==b.base && a.total==b.total;
}
static void compare(const NttMemoryState &s,const NttArena &a) {
    check(actual.size()==predicted.size(),"event count");
    for(size_t i=event_checks;i<actual.size();++i) {
        const auto &x=actual[i];const auto &y=predicted[i];
        check(x.site==y.site && x.index==y.index && x.bytes==y.bytes && x.allocation==y.allocation &&
            !y.grouped && equal(x.live,y.live),"event_order_or_payload");
    }
    event_checks=actual.size();const auto p=a.payload();
    check(s.live.big==p.big && s.live.digits==p.digits && s.live.table==p.table && s.live.base==p.base && s.live.total==p.total,"final payload");
    check(equal(s.live,native_live) && a.bytes==native_live.total,"native accounting");
    check(s.peak==a.peak_full_bytes,"model peak");
    check(s.counters.fuse_builds==a.fuse_builds && s.counters.fuse_hits==a.fuse_hits,"fuse counters");
    check(s.counters.cap_evictions==a.tbl_evictions && s.counters.cap_evicted_bytes==a.tbl_words_freed*8,"cap counters");
    check(s.counters.cold_evictions==a.phase_fuse_evictions && s.counters.cold_evicted_bytes==a.phase_fuse_evicted_bytes,"trim counters");
    check(s.counters.carry_grows==a.carry_grows && s.counters.carry_refusals==a.carry_refusals,"carry counters");
    Word allocs=0,frees=0;for(const auto &e:actual)(e.allocation?allocs:frees)++;
    check(s.counters.allocations==allocs && s.counters.frees==frees && !s.counters.grouped_events,"event counters");
}
static void reset() {
    check(heap.empty() && !native_live.total,"leak");actual.clear();predicted.clear();native_peak=0;event_checks=0;++cases;
}
static Word total_events=0;
int main() {
    try {
        for(bool compact:{false,true})for(Word t:{Word(0),Word(5),Word(12)})for(Word coop:{Word(0),Word(1),Word(2)})
          for(bool pool:{false,true})for(bool reuse:{false,true})for(bool carry:{false,true}) {
            reset();context="layout compact="+std::to_string(compact)+" t="+std::to_string(t)+" coop="+std::to_string(coop)+" pool="+std::to_string(pool)+" reuse="+std::to_string(reuse)+" carry="+std::to_string(carry);
            flag("NTT_FUSE_COMPACT_SCRATCH",compact);value("NTT_FUSE_T",t);value("NTT_FUSE_COOP_OUTER",coop);
            value("NTT_FUSE_M",4);value("NTT_FUSE_COOP_M",8);flag("NTT_FUSE_WARP_TAIL",true);
            NttMemoryPolicy p{0,pool,reuse,carry};NttMemoryState s(p);NttArena a;
            s.observe=[](const NttMemoryEvent &e){predicted.push_back(e);};
            a.workspace_pool=pool;a.workspace_reuse_bq=reuse;flag("NTT_CARRY_CHECK_FUSED",carry);
            const std::vector<std::vector<Word>> requests={{8,1,1},{8192,7,1},{8,3,1},{65536,8,2},{8192,33,1},
                {1048576,1,1},{1048576,1,4},{32,1,65536},{Word(1)<<24,1,1},{Word(1)<<27,1,1}};
            for(const auto &r:requests) {
                check(s.call_layout(r[0],r[1],r[2],describe) && native_call(a,r[0],r[1],r[2]),"successful call");compare(s,a);
            }
            Word available=0;check(s.cold_trim(Word(1)<<27,Word(1)<<40,available));
            Word freed=0;while(const auto b=a.drop_cold_fuse(Word(1)<<27))freed+=b;
            check(available==freed,"trim freed bytes");compare(s,a);
            check(s.call_layout(65536,19,2,describe) && native_call(a,65536,19,2),"reconstruct cold shape");compare(s,a);
            check(s.close(),"close");a.release();compare(s,a);check(!s.live.total && heap.empty() && !native_live.total,"released");
            check(s.peak==native_peak,"allocation event peak");
            const Word count=predicted.size();check(s.close(),"idempotent close");check(predicted.size()==count,"idempotent event count");
            total_events+=actual.size();
        }
        // Tight caps exercise reverse-insertion keyed evictions, table-only
        // eviction with base retention, failed fuse/big/digits prefixes, and ties.
        for(bool pool:{false,true})for(bool reuse:{false,true})for(Word cap:{Word(20000),Word(70000),Word(200000),Word(1000000)}) {
            reset();context="cap="+std::to_string(cap)+" pool="+std::to_string(pool);
            flag("NTT_FUSE_COMPACT_SCRATCH",true);value("NTT_FUSE_T",5);value("NTT_FUSE_COOP_OUTER",0);
            NttMemoryPolicy p{cap,pool,reuse,true};NttMemoryState s(p);NttArena a;
            s.observe=[](const NttMemoryEvent &e){predicted.push_back(e);};
            a.workspace_pool=pool;a.workspace_reuse_bq=reuse;a.cap_bytes=cap;flag("NTT_CARRY_CHECK_FUSED",true);
            for(const auto &r:std::vector<std::vector<Word>>{{64,10,1},{128,11,1},{64,12,3},{128,1,8},{32,3,1},{1024,1,30},{1ull<<20,1,1}}) {
                const bool wanted=s.call_layout(r[0],r[1],r[2],describe),got=native_call(a,r[0],r[1],r[2]);
                check(wanted==got,"refusal agreement");compare(s,a);if(!got)break;
            }
            check(s.close());a.release();compare(s,a);total_events+=actual.size();
        }
        context="carry cap refusal";reset();
        {
            NttFuseMemoryLayout l;check(describe(Word(1)<<20,l));Word base=0,table=0;check(l.totals(base,table));
            NttMemoryPolicy p{base+table+16*(Word(1)<<20)+24,true,true,true};
            NttMemoryState s(p);NttArena a;s.observe=[](const NttMemoryEvent &e){predicted.push_back(e);};
            a.workspace_pool=true;a.workspace_reuse_bq=true;a.cap_bytes=p.cap_bytes;
            check(s.call_layout(Word(1)<<20,1,1,describe) && native_call(a,Word(1)<<20,1,1));compare(s,a);
            check(s.counters.carry_refusals==1 && !s.counters.carry_grows,"carry refusal retained payload");
            check(s.close());a.release();compare(s,a);total_events+=actual.size();
        }
        // Eager execution against the native arena independently checks the
        // compressed program's physical allocation/free counters and descriptor.
        auto query=[](Word m,int,Word *n,Word *slots){*n=8;while(*n<4*m)*n*=2;*slots=2*m-1;return true;};
        for(Word p:{Word(1),Word(48),Word(129)})for(bool pool:{false,true})for(bool reuse:{false,true}) {
            reset();context="exact program P="+std::to_string(p)+" pool="+std::to_string(pool);
            flag("NTT_FUSE_COMPACT_SCRATCH",true);value("NTT_FUSE_T",12);value("NTT_FUSE_COOP_OUTER",0);
            NttMemoryPolicy config{0,pool,reuse,false};NttArena a;a.workspace_pool=pool;a.workspace_reuse_bq=reuse;
            flag("NTT_CARRY_CHECK_FUSED",false);RequestProgram program;check(request_program(p,20*p+1,program));
            NttMemoryPlan plan;check(ntt_memory_plan(program,129,query,describe,16384,true,0,config,plan));
            Word calls=0;
            for(const auto &b:program.blocks)for(Word rep=0;rep<b.repeat;++rep)for(const auto &r:b.requests) {
                Word n=0,slots=0;query(std::max(r.ma,r.mb),129,&n,&slots);
                const Word c=chunk_slices(n,slots,r.pairs,16384,pool && reuse?2:3,true);
                for(Word offset=0;offset<r.pairs;offset+=c) {++calls;check(native_call(a,n,slots,std::min(c,r.pairs-offset)));}
            }
            Word allocs=0,frees=0;for(const auto &e:actual)(e.allocation?allocs:frees)++;
            const auto x=a.payload();
            check(plan.valid && plan.finished && plan.exact_allocation_events && !plan.counters.grouped_events,"exact plan scope");
            check(plan.counters.calls==calls && plan.counters.allocations==allocs && plan.counters.frees==frees,"compressed event counters");
            check(plan.final_payload.big==x.big && plan.final_payload.digits==x.digits && plan.final_payload.base==x.base && plan.final_payload.table==x.table && plan.final_payload.total==x.total,"compressed final payload");
            check(plan.peak_bytes==a.peak_full_bytes && plan.peak_bytes==native_peak,"compressed peak");
            a.release();check(heap.empty() && !native_live.total);total_events+=actual.size();
        }
        context="exact huge repeat";
        RequestProgram huge;check(request_program(48,Word(1)<<60,huge));NttMemoryPlan huge_plan;
        check(ntt_memory_plan(huge,129,query,describe,16384,true,0,{0,true,true,false},huge_plan));
        check(huge_plan.finished && huge_plan.exact_allocation_events && huge_plan.executed_blocks<20 && huge_plan.skipped_blocks>(Word(1)<<50),"exact repeat compression");
        context="invalid descriptor";
        FuseCtx invalid;invalid.n=8;invalid.k=3;invalid.t=2;invalid.outer_stages=1;invalid.nms=1;invalid.ms[0]=2;
        NttFuseMemoryLayout layout;check(!ntt_fuse_memory_layout(invalid,layout));
        NttMemoryState bad;check(!bad.call_layout(8,1,1,[](Word,NttFuseMemoryLayout &x){x.base={{NttBaseTileF,0,8},{NttBaseTileF,0,8}};return true;}));
        check(!bad.live.total && !std::strcmp(bad.reason,"descriptor_overflow"));
        std::cout<<"{\"checks\":"<<checks<<",\"cases\":"<<cases<<",\"events\":"<<total_events<<",\"bad\":0,\"gpu_calls\":0}\n";return 0;
    } catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
