#pragma once
#include "ecm_stage2_geometry.h"
#include <functional>

namespace ecm_stage2 {
// Native baby caller order: five constants, indices, X, Z, product tree,
// ordinary leaf output, byte mask. Destruction is reverse declaration order.
struct BabyMemoryLayout {
    Word counts[9]={},offset[9]={},tree_words=0,bytes=0,allocation_bytes[11]={};
};
inline bool baby_memory_layout(Word p,Word w,BabyMemoryLayout &layout) {
    layout=BabyMemoryLayout{};
    if(!p || !w || w>max_words)return false;
    layout.counts[0]=p;
    for(unsigned level=1;level<=8;++level) {
        layout.counts[level]=layout.counts[level-1]/2+(layout.counts[level-1]%2!=0);
        layout.offset[level]=layout.tree_words;
        Word count=0;
        if(!multiply(layout.counts[level],w,count) || !add(layout.tree_words,count,layout.tree_words))return false;
    }
    Word scalar=0,indices=0,coordinates=0,tree=0;
    if(!multiply(w,8,scalar) || !multiply(p,8,indices) ||
       !multiply(p,scalar,coordinates) || !multiply(layout.tree_words,8,tree))return false;
    const Word sizes[]={scalar,scalar,scalar,scalar,scalar,indices,coordinates,coordinates,tree,coordinates,layout.counts[8]};
    for(unsigned i=0;i<11;++i) {
        layout.allocation_bytes[i]=sizes[i];
        if(!add(layout.bytes,sizes[i],layout.bytes))return false;
    }
    return true;
}
struct InitialMemoryPolicy {
    Word p=0,words=0,baby_budget_bytes=512ull<<20;
    bool baby_requested=true,diagnostic=false;
};
struct InitialMemoryEvent {unsigned site=0;bool allocation=false;Word bytes=0,total=0;};
class InitialMemoryState {
    bool refresh(unsigned site,bool allocation,Word bytes) {
        if(allocation){if(!add(live,bytes,live)){reason="payload_overflow";return false;}}
        else {if(live<bytes){reason="invalid_initial_release";return false;}live-=bytes;}
        peak=std::max(peak,live);if(observe)observe({site,allocation,bytes,live});return true;
    }
public:
    InitialMemoryPolicy policy;
    Word live=0,peak=0,montgomery_bytes=0,baby_bytes=0;
    const char *reason="ok";
    bool startup_done=false,baby_done=false;
    std::function<void(const InitialMemoryEvent &)> observe;
    explicit InitialMemoryState(InitialMemoryPolicy p={}):policy(p){}
    bool startup() {
        if(startup_done || !policy.p || !policy.words || policy.words>max_words) {
            reason="invalid_initial_policy";return false;
        }
        if(policy.diagnostic){reason="diagnostic_initial_workspace_not_modeled";return false;}
        // mont_selftest: 2048 cases; three case arrays and one modulus.
        const Word scalar=policy.words*8,values=2048*scalar,sizes[]={values,values,scalar,values};
        for(unsigned i=0;i<4;++i)if(!refresh(i,true,sizes[i]))return false;
        montgomery_bytes=live;
        for(unsigned i=0;i<4;++i)if(!refresh(i,false,sizes[i]))return false;
        startup_done=true;return true;
    }
    bool baby() {
        BabyMemoryLayout layout;
        if(!startup_done || baby_done || !baby_memory_layout(policy.p,policy.words,layout)) {
            reason="invalid_baby_shape";return false;
        }
        baby_bytes=layout.bytes;
        if(!policy.baby_requested || baby_bytes>policy.baby_budget_bytes) {
            reason="baby_budget_refusal";return false;
        }
        for(unsigned i=0;i<11;++i)if(!refresh(4+i,true,layout.allocation_bytes[i]))return false;
        for(unsigned i=11;i-->0;)if(!refresh(4+i,false,layout.allocation_bytes[i]))return false;
        baby_done=true;return true;
    }
};
} // namespace ecm_stage2
