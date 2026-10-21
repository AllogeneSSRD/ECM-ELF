#pragma once
#include "ecm_stage2_geometry.h"

namespace ecm_stage2 {
// Preserve the complete original grid, including guard coverage. Relocation
// changes only its first index; hits/nonunits must replay first=1 before publish.
struct GiantRange {
    Word first=1,last=0,points=0,missing=0;
    bool active=false;
};
inline GiantRange giant_range(Word b2,Word d,bool enabled) {
    GiantRange r;
    if(d<6 || d%2 || b2/d>std::numeric_limits<Word>::max()-2)return r;
    r.last=b2/d+2;r.points=r.last;
    if(enabled) {
        // The least integer >1 coprime to D is necessarily prime.
        Word m=2;
        for(;;++m) {Word a=m,b=d;while(b){const auto t=a%b;a=b;b=t;}if(a==1)break;}
        r.missing=m;r.first=std::max(Word(1),r.last/m);
        r.points=r.last-r.first+1;r.active=r.first>1;
    }
    return r;
}
// Borrow whole G batches from the penultimate point chunk. Preserve capacity,
// point count and chunk count; never turn either of the two chunks into a ladder.
inline Word giant_tail_borrow(Word points,Word capacity,Word p,Word minimum,bool rebalance) {
    if(!rebalance || !capacity || !p || capacity%p || points<=capacity || !minimum)return 0;
    const Word tail=points%capacity;
    if(!tail || tail>=minimum)return 0;
    const Word deficit=minimum-tail;
    Word borrow=0;
    if(!multiply(deficit/p+(deficit%p!=0),p,borrow) || borrow>=capacity ||
       capacity-borrow<minimum || tail>capacity-borrow)return 0;
    return borrow;
}
inline Word giant_chunk_take(Word remaining,Word capacity,Word p,Word minimum,bool rebalance) {
    if(remaining<=capacity)return remaining;
    const auto borrow=giant_tail_borrow(remaining,capacity,p,minimum,rebalance);
    return borrow && remaining/capacity==1 ? capacity-borrow : capacity;
}
}
