#include "../../src/core/ecm_stage2_owner_memory.h"
#include <cstdint>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>
using namespace ecm_stage2;
struct Event {bool allocation;Word bytes,total;};
static std::vector<Event> actual,predicted;
static std::map<void*,Word> heap;
static Word live=0,peak=0,checks=0,cases=0;static uintptr_t token=4096;
static void check(bool ok){++checks;if(!ok)throw std::runtime_error("owner check "+std::to_string(checks));}
template<class T>static int cudaMalloc(T **p,size_t bytes) {
    check(bytes>0);*p=reinterpret_cast<T*>(token);token+=4096;heap[*p]=bytes;
    live+=bytes;peak=std::max(peak,live);actual.push_back({true,(Word)bytes,live});return 0;
}
static int cudaFree(void *p) {
    if(!p)return 0;check(heap.count(p)==1);const auto bytes=heap[p];heap.erase(p);live-=bytes;
    actual.push_back({false,bytes,live});return 0;
}
static int cudaMemsetAsync(void*,int,size_t){return 0;}
static int cudaGetLastError(){return 0;}
constexpr int cudaErrorMemoryAllocation=2;
#define CK(x) do {check((x)==0);}while(0)
struct Memory {
    Word words[2]={};Word *data[2]={};
    void release(){for(auto &p:data)if(p){cudaFree(p);p=nullptr;}}
};
struct NativeFold {
    Memory memory;Word *map=nullptr,*length=nullptr,*modulus=nullptr,*digest=nullptr;
    size_t P=0,W=0,t=0;bool active=false;
    struct Stats {const char *fallback="none";Word layout_bytes=0;} st;
    Stats *stats=&st;
    void release() {
        // @NATIVE_RELEASE@
    }
    bool init(const FoldOwnerLayout &layout) {
        // @NATIVE_ALLOCATE@
        active=true;st.layout_bytes=layout.bytes;return true;
    }
};
struct NativeFrontier {
    Word *metadata=nullptr;const char *fallback="none";
    Word metadata_bytes=0,metadata_words=0,owner_and_metadata_bytes=0;
    bool init(NativeFold &owner,bool mutate) {
        const size_t p=owner.P,w=owner.W;
        // @NATIVE_METADATA_SHAPE@
        if(mutate)bytes+=8;
        // @NATIVE_METADATA_ALLOCATE@
        return true;
    }
    void release(){if(metadata){cudaFree(metadata);metadata=nullptr;}}
};
int main(int argc,char**) {
    try {
        for(Word p:{1ull,2ull,3ull,17ull,64ull,65ull,103680ull})
        for(Word w:{1ull,2ull,4ull,128ull,256ull})for(unsigned reuse=0;reuse<4;++reuse) {
            actual.clear();predicted.clear();peak=0;check(heap.empty() && !live);
            NativeFold owner;owner.P=p;owner.W=w;
            FoldOwnerLayout layout;check(fold_owner_layout(p,w,reuse,layout));
            check(owner.init(layout));NativeFrontier frontier;check(frontier.init(owner,argc>1));
            frontier.release();owner.release();
            OwnerMemoryState model({p,w,1ull<<40,reuse});
            model.observe=[&](const OwnerMemoryEvent &e){predicted.push_back({e.allocation,e.bytes,e.total});};
            check(model.fold_begin() && model.frontier_begin() && model.close());
            check(actual.size()==predicted.size());
            for(size_t i=0;i<actual.size();++i) {
                check(actual[i].allocation==predicted[i].allocation && actual[i].bytes==predicted[i].bytes &&
                      actual[i].total==predicted[i].total);
            }
            check(!live && heap.empty() && !model.live && model.peak==peak);
            ++cases;
        }
        std::cout<<"{\"cases\":"<<cases<<",\"checks\":"<<checks<<",\"allocation_events\":"<<cases*14
                 <<",\"gpu_calls\":0}\n";
    } catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
