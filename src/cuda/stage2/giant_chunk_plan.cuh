#pragma once
#define ECM_STAGE2_GIANT_CHUNK_PLAN 1
#include <algorithm>
#include <cstdint>
#include <limits>

// A point chunk contains whole G-tree batches. The budget covers X/Z payloads
// only; seeds, segment products and NTT/owner lifetimes must be charged separately.
namespace stage2_giant_chunk {
struct Plan {
    bool valid=false, minimum_over_budget=false;
    uint64_t points=0, coordinate_bytes=0;
};
inline Plan plan(uint64_t P,uint64_t nw,uint64_t budget,bool floor) {
    Plan out;
    const auto maximum=std::numeric_limits<uint64_t>::max();
    if(!P || !nw || !budget || nw>maximum/16)return out;
    const uint64_t per_point=16*nw;
    const uint64_t k=budget/per_point;
    const uint64_t batches=std::max(uint64_t(1),k/P+uint64_t(!floor && k%P!=0));
    if(batches>maximum/P)return out;
    out.points=P*batches;
    if(out.points>maximum/per_point)return out;
    out.coordinate_bytes=out.points*per_point;
    out.minimum_over_budget=P>budget/per_point;
    out.valid=true;
    return out;
}
inline void fixture() {
    uint64_t checks=0,bad=0;
    auto check=[&](bool value){++checks;if(!value)++bad;};
    for(uint64_t P:{1ull,48ull,126720ull,138240ull})for(uint64_t nw:{1ull,126ull,258ull}) {
        const uint64_t batch_bytes=16*nw*P;
        for(uint64_t budget:{1ull,batch_bytes-1,batch_bytes,batch_bytes+1,2*batch_bytes-1,2*batch_bytes}) {
            const auto f=plan(P,nw,budget,true),c=plan(P,nw,budget,false);
            check(f.valid && c.valid && f.points%P==0 && c.points%P==0);
            check(f.coordinate_bytes==std::max(uint64_t(1),budget/batch_bytes)*batch_bytes);
            const uint64_t k=budget/(16*nw);
            check(c.coordinate_bytes==std::max(uint64_t(1),k/P+uint64_t(k%P!=0))*batch_bytes);
            check(f.minimum_over_budget==(batch_bytes>budget));
        }
    }
    check(plan(126720,126,256ull<<20,true).points==126720);
    check(plan(126720,126,256ull<<20,false).points==253440);
    check(plan(138240,126,256ull<<20,true).points==138240);
    check(plan(138240,126,256ull<<20,false).points==138240);
    const auto maximum=std::numeric_limits<uint64_t>::max();
    check(!plan(0,1,1,true).valid);check(!plan(1,0,1,true).valid);
    check(!plan(1,1,0,true).valid);check(!plan(1,maximum,1,true).valid);
    check(!plan(maximum,1,1,true).valid);check(!plan(maximum,1,maximum,false).valid);
    stage2_log::print(stage2_log::debug,"giant_chunk_check: checks=%llu bad=%llu\n",
        (unsigned long long)checks,(unsigned long long)bad);
    if(bad)std::exit(3);
}
}
