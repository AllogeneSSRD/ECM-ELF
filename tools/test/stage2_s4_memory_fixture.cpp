// CPU allocation-event proof. The runner inserts production S4Ctx, lookup and
// allocation statements. No CUDA runtime or arithmetic kernels are linked.
#include "../../src/core/ecm_stage2_s4_memory.h"
#include "../../src/core/ecm_stage2_requests.h"
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdint>
using namespace ecm_stage2;
struct Event {bool alloc;Word bytes,total;};
static std::vector<Event> actual,predicted;
static std::map<void*,Word> heap;
static Word total=0,peak=0,checks=0,event_checks=0,scenarios=0;static uintptr_t token=4096;
static void check(bool ok,const char *why="check"){
    ++checks;if(!ok)throw std::runtime_error(std::string(why)+" "+std::to_string(checks));
}
template<class T>static int cudaMalloc(T **p,size_t bytes) {
    check(bytes!=0);*p=reinterpret_cast<T*>(token);token+=4096;heap[*p]=bytes;
    total+=bytes;peak=std::max(peak,total);actual.push_back({true,(Word)bytes,total});return 0;
}
static int cudaFree(void *p) {
    if(!p)return 0;check(heap.count(p)==1);const Word bytes=heap[p];heap.erase(p);
    total-=bytes;actual.push_back({false,bytes,total});return 0;
}
static int cudaFreeHost(void *p){check(p==nullptr);return 0;}
#define CK(x) do{check((x)==0);}while(0)
enum {BC_NCAT=RequestPhaseCount};
struct S4Reduce;
struct {Word grows=0;}g_chunk_output;

// @NATIVE_CTX@

static void output_reserve(S4Ctx &C,Word words) {
    const size_t allocation_need=(size_t)std::max(1ull,words);
    // @NATIVE_OUTPUT@
}
static void pack_reserve(S4Ctx &C,Word words) {
    const size_t pack_words=(size_t)words;const bool g_s4_pack_direct=false;
    // @NATIVE_PACK@
}
struct Reducer {
    int nw=0;size_t w=0;Word *dn=nullptr,*dbad=nullptr;
    Word dbad_host=0;
    struct Shape {Word slot_bits=0,slot_words=0;int bpw=0;Word *dy=nullptr;};
    std::vector<Shape*>shapes;
    // @NATIVE_FIND@
    void init(Word words) {Reducer &R=*this;R.w=words;R.nw=(int)words;
        // @NATIVE_MODULUS@
    }
    void shape(Word bits,Word slots,int bpw) {
        if(find(bits,slots,bpw))return;
        Reducer &R=*this;Shape *S=new Shape{bits,slots,bpw};
        // @NATIVE_CONSTANT@
        shapes.push_back(S);
        const Word cases=96;std::vector<Word>dig((size_t)(cases*slots));
        Word *dd=nullptr,*dout=nullptr,*ddbg=nullptr;constexpr bool dbg=false;
        // @NATIVE_SELFTEST_ALLOC@
        // @NATIVE_SELFTEST_FREE@
    }
    void counter(){Reducer &R=*this;
        // @NATIVE_COUNTER@
    }
    ~Reducer() {
        for(Shape *s:shapes){if(s->dy)cudaFree(s->dy);delete s;}
        if(dn)cudaFree(dn);if(dbad)cudaFree(dbad);
    }
};
static void verify(const S4MemoryState &s) {
    check(total==s.live.total,"S4 live bytes differ");check(peak==s.peak,"S4 peak bytes differ");
    check(actual.size()==predicted.size(),"S4 event count differs");
    for(size_t i=0;i<actual.size();++i)check(actual[i].alloc==predicted[i].alloc &&
        actual[i].bytes==predicted[i].bytes && actual[i].total==predicted[i].total,"S4 event differs");
}
static void one(Word w,bool compact) {
    check(heap.empty());actual.clear();predicted.clear();total=peak=0;
    S4MemoryState model;
    model.observe=[](const S4MemoryEvent &e){predicted.push_back({e.allocation,e.bytes,e.payload.total});};
    {
        S4Ctx C;Reducer R;R.init(w);check(model.init(w));verify(model);
        for(int turn=0;turn<3;++turn) {
            for(Word n:{1ull,2ull,3ull,63ull,64ull,65ull,127ull,129ull}) {
                // Independent dense padded tree to determine the first parent frontier.
                Word pad=1;while(pad<n)pad*=2;
                std::vector<Word>deg((size_t)(2*pad));for(Word i=0;i<n;++i)deg[(size_t)(pad+i)]=1;
                for(Word i=pad;--i;)deg[(size_t)i]=deg[(size_t)(2*i)]+deg[(size_t)(2*i+1)];
                Word parents=0;for(Word i=pad/2;pad>1 && i<pad;++i)if(deg[(size_t)i])parents+=(deg[(size_t)i]+1)*w;
                C.raw_reserve((size_t)(2*n*w),(size_t)(compact?parents:2*n*w));
                {
                // @NATIVE_METADATA_CTX@
                if(pad>1) {
                    // @NATIVE_METADATA_ALLOC@
                }
                check(model.tree_begin(n,compact));verify(model);
                R.shape(129,5,26);check(model.shape(129,5,26));verify(model);
                // Same bit window but different pack key: must allocate a second constant.
                R.shape(129,6,22);check(model.shape(129,6,22));verify(model);
                R.shape(129,5,26);check(model.shape(129,5,26));verify(model);
                output_reserve(C,n*w);check(model.output_reserve(n*w));verify(model);
                R.counter();check(model.canonical_counter());verify(model);
                } // Exact native metadata destructor frees the temporary lease.
                check(model.tree_end());verify(model);
                // Host path uses independent padded A/B capacities; retain then shrink requests.
                C.raw_reserve((size_t)(n*w+7),(size_t)(2*n*w+3));
                check(model.raw_reserve(n*w+7,2*n*w+3));verify(model);
                pack_reserve(C,n*8);check(model.pack_reserve(n*8));verify(model);
                output_reserve(C,0);check(model.output_reserve(0));verify(model);
            }
            C.raw_release();check(model.raw_release());verify(model);
            C.output_release();check(model.output_release());verify(model);
        }
        check(model.shape_count()==2);
    }
    check(model.close());verify(model);check(heap.empty() && total==0);
    event_checks+=actual.size();++scenarios;
}
int main() {
    try {
        for(Word w:{1ull,7ull,126ull,256ull})for(bool compact:{false,true})one(w,compact);
        S4MemoryState s;check(!s.init(0));S4MemoryState t;check(!t.init(257));
        S4MemoryState o;check(o.init(1));check(!o.raw_reserve(~Word(0),1));
        S4MemoryState b;check(b.init(1));check(!b.shape(129,5,25));
        check(!b.shape(0,1,1));check(!b.tree_begin(0,true));
        check(b.tree_begin(1,true));check(!b.tree_begin(1,true));check(b.tree_end());check(b.close());
        std::cout<<"{\"checks\":"<<checks<<",\"allocation_events\":"<<event_checks
            <<",\"scenarios\":"<<scenarios<<",\"tree_leases\":"<<scenarios*24
            <<",\"bad\":0,\"gpu_calls\":0}\n";return 0;
    }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}
}
