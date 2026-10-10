#pragma once
#include "ecm_stage2_geometry.h"

namespace ecm_stage2 {
// Exact route counts for the normal chunk loop; no timing or memory admission.
struct GiantWork {
    Word chain_points=0,ladder_points=0,chain_chunks=0,ladder_chunks=0,ladder_steps=0;
};
inline bool giant_ladder_steps(Word first,Word last,Word d,Word &steps) {
    steps=0;Word scalar=0;
    if(!first || first>last || !d || !multiply(last,d,scalar))return false;
    // Each scalar i*D performs floor(log2(i*D)) ladder iterations. Count the
    // indices at each power-of-two boundary without iterating over the points.
    for(unsigned bit=1;bit<64;++bit) {
        const Word power=Word(1)<<bit;
        const Word threshold=power/d+(power%d!=0);
        const Word low=std::max(first,threshold);
        if(low<=last && !add(steps,last-low+1,steps))return false;
    }
    return true;
}
inline bool giant_work(Word points,Word d,Word chunk,Word minimum,bool force,GiantWork &out) {
    out=GiantWork{};
    if(!points || !d || !chunk)return false;
    const Word full=points/chunk,tail=points%chunk;
    if(force || chunk<minimum) {
        out.ladder_points=points;out.ladder_chunks=full+(tail!=0);
        return giant_ladder_steps(1,points,d,out.ladder_steps);
    }
    out.chain_chunks=full;out.chain_points=full*chunk;
    if(tail) {
        if(tail>=minimum){++out.chain_chunks;out.chain_points+=tail;}
        else {
            out.ladder_points=tail;out.ladder_chunks=1;
            if(!giant_ladder_steps(points-tail+1,points,d,out.ladder_steps))return false;
        }
    }
    return true;
}
} // namespace ecm_stage2
