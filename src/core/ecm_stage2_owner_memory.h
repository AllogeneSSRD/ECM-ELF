#pragma once
#include "ecm_stage2_geometry.h"
#include <functional>

namespace ecm_stage2 {
enum OwnerMemorySite {OwnerSource,OwnerResult,OwnerMap,OwnerLength,OwnerModulus,OwnerDigest,OwnerFrontierMetadata};
struct OwnerMemoryEvent {
    OwnerMemorySite site=OwnerSource;
    bool allocation=false;
    Word bytes=0,total=0;
};
struct OwnerMemoryPolicy {
    Word p=0,words=0,budget_bytes=0;unsigned reuse=3;
    Word frontier_budget_bytes=std::numeric_limits<Word>::max();
};
class OwnerMemoryState {
    Word capacity[7]={};
    bool refresh(OwnerMemorySite site,bool allocation,Word bytes) {
        if(allocation) {if(!add(live,bytes,live)){reason="payload_overflow";return false;}}
        else {if(live<bytes){reason="invalid_release";return false;}live-=bytes;}
        peak=std::max(peak,live);
        if(observe)observe({site,allocation,bytes,live});return true;
    }
    bool allocate(OwnerMemorySite site,Word bytes) {
        if(capacity[site] || !bytes){reason="invalid_allocation";return false;}
        if(!refresh(site,true,bytes))return false;capacity[site]=bytes;return true;
    }
    bool release(OwnerMemorySite site) {
        const Word bytes=capacity[site];if(!bytes)return true;
        if(!refresh(site,false,bytes))return false;capacity[site]=0;return true;
    }
public:
    OwnerMemoryPolicy policy;
    Word live=0,peak=0,fold_bytes=0,frontier_bytes=0;
    const char *reason="ok";
    std::function<void(const OwnerMemoryEvent &)> observe;
    explicit OwnerMemoryState(OwnerMemoryPolicy p={}):policy(p){}
    bool same_allocations(const OwnerMemoryState &o)const {
        for(unsigned i=0;i<7;++i)if(capacity[i]!=o.capacity[i])return false;
        return true;
    }
    bool fold_begin() {
        FoldOwnerLayout l;Word a=0,b=0,n=0;
        if(!policy.p || !policy.words || policy.words>max_words ||
           !fold_owner_layout(policy.p,policy.words,policy.reuse,l) ||
           !multiply(l.source_words,8,a) || !multiply(l.result_words,8,b) ||
           !multiply(policy.words,8,n)){reason="invalid_owner_shape";return false;}
        fold_bytes=l.bytes;
        if(fold_bytes>policy.budget_bytes){reason="fold_budget_refusal";return false;}
        // Successful FoldDeviceState::init order, including its small buffers.
        return allocate(OwnerSource,a) && allocate(OwnerResult,b) &&
            allocate(OwnerMap,24) && allocate(OwnerLength,8) &&
            allocate(OwnerModulus,n) && allocate(OwnerDigest,16);
    }
    bool frontier_begin() {
        Word total=0;
        if(!capacity[OwnerSource] || !capacity[OwnerResult] ||
           !multiply(policy.p,24,frontier_bytes) || !add(live,frontier_bytes,total)) {
            reason="invalid_frontier_shape";return false;
        }
        if(total>std::min(policy.budget_bytes,policy.frontier_budget_bytes)) {
            reason="frontier_budget_refusal";return false;
        }
        return allocate(OwnerFrontierMetadata,frontier_bytes);
    }
    bool close() {
        // End of frontier.run: frontier metadata first, then FoldDeviceState.
        return release(OwnerFrontierMetadata) && release(OwnerSource) &&
            release(OwnerResult) && release(OwnerMap) && release(OwnerLength) &&
            release(OwnerModulus) && release(OwnerDigest);
    }
};
} // namespace ecm_stage2
