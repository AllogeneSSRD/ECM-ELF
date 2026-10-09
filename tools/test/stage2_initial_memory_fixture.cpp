// CPU opaque allocator. Runner inserts native startup and baby allocation sites.
#include "../../src/core/ecm_stage2_initial_memory.h"
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdint>
using namespace ecm_stage2;
struct Event {bool allocation;Word bytes,total;};
static std::vector<Event> actual,predicted;
static std::map<void*,Word> heap;
static Word live=0,peak=0,checks=0,event_checks=0;static uintptr_t token=4096;
static void check(bool ok){++checks;if(!ok)throw std::runtime_error("initial check "+std::to_string(checks));}
template<class T>static int cudaMalloc(T **p,size_t bytes) {
    check(bytes>0);*p=reinterpret_cast<T*>(token);token+=4096;heap[*p]=bytes;live+=bytes;
    peak=std::max(peak,live);actual.push_back({true,(Word)bytes,live});return 0;
}
static int cudaFree(void *p) {
    if(!p)return 0;check(heap.count(p)==1);const Word bytes=heap[p];heap.erase(p);live-=bytes;
    actual.push_back({false,bytes,live});return 0;
}
#define CK(x) do{check((x)==0);}while(0)
struct BabyBuffer {
    void *ptr=nullptr;
    ~BabyBuffer(){if(ptr)CK(cudaFree(ptr));}
    bool allocate(size_t bytes){return cudaMalloc(&ptr,bytes)==0;}
};
struct FakeVector {size_t count;size_t size()const{return count;}};
static void native_mont(size_t nw) {
    const int cases=2048;FakeVector ha{cases*nw},hb{cases*nw};
    // @NATIVE_MONT_ALLOC@
    // @NATIVE_MONT_FREE@
}
static bool native_baby(size_t n,size_t w,bool mutate) {
    // Independent dense halving oracle, not the shared closed-form layout.
    size_t counts[9]={n},offset[9]={},words=0;
    for(unsigned i=1;i<9;++i) {
        for(size_t j=0;j<counts[i-1];j+=2)++counts[i];
        offset[i]=words;words+=counts[i]*w;
    }
    const size_t groups=counts[8];BabyMemoryLayout layout;
    check(baby_memory_layout(n,w,layout));
    for(unsigned i=0;i<9;++i)check(layout.counts[i]==counts[i] && layout.offset[i]==offset[i]);
    check(layout.tree_words==words && layout.bytes==8*((3*n+5)*w+n+words)+groups);
    // @NATIVE_BABY_ALLOC@
    return true;
}
int main(int argc,char**) {
    try {
        Word cases=0;
        for(Word p:{1ull,2ull,3ull,17ull,255ull,256ull,257ull,65535ull,103680ull,126720ull})
        for(Word w:{1ull,7ull,92ull,128ull,256ull}) {
            check(heap.empty() && !live);peak=0;actual.clear();predicted.clear();
            native_mont((size_t)w);check(native_baby((size_t)p,(size_t)w,argc>1));
            InitialMemoryState model({p,w,1ull<<40});
            model.observe=[&](const InitialMemoryEvent &e){predicted.push_back({e.allocation,e.bytes,e.total});};
            check(model.startup() && model.baby());check(actual.size()==predicted.size());
            for(size_t i=0;i<actual.size();++i) {
                check(actual[i].allocation==predicted[i].allocation && actual[i].bytes==predicted[i].bytes &&
                      actual[i].total==predicted[i].total);++event_checks;
            }
            check(heap.empty() && !live && !model.live && model.peak==peak);++cases;
        }
        BabyMemoryLayout invalid;
        check(!baby_memory_layout(0,1,invalid));check(!baby_memory_layout(1,0,invalid));
        check(!baby_memory_layout(1,257,invalid));check(!baby_memory_layout(~Word(0),256,invalid));
        InitialMemoryState refusal({1,1,0});check(refusal.startup() && !refusal.baby());
        check(!std::string(refusal.reason).compare("baby_budget_refusal") && !refusal.live);
        std::cout<<"{\"cases\":"<<cases<<",\"checks\":"<<checks<<",\"event_checks\":"<<event_checks
            <<",\"gpu_calls\":0}\n";
    }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
