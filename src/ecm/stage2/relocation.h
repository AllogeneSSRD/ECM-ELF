#pragma once
#include "giant_schedule.h"

namespace ecm_stage2 {
// Preserve the complete original grid, including guard coverage. Relocation
// changes only its first index; hits/nonunits replay the original grid.
struct GiantRange {
    Word first=1,last=0,points=0,missing=0;
    bool active=false;
};
inline GiantRange giant_range(Word b2,Word d,bool enabled,Word lower=0) {
    GiantRange r;
    if(d<6 || d%2 || b2/d>std::numeric_limits<Word>::max()-2)return r;
    r.first=giant_first(lower,d);
    r.last=b2/d+2;
    if(r.first>r.last)return {};
    const auto original_first=r.first;
    if(enabled) {
        // The least integer >1 coprime to D is necessarily prime.
        Word m=2;
        for(;;++m) {Word a=m,b=d;while(b){const auto t=a%b;a=b;b=t;}if(a==1)break;}
        r.missing=m;r.first=std::max(r.first,r.last/m);
        r.active=r.first>original_first;
    }
    r.points=r.last-r.first+1;
    return r;
}
} // namespace ecm_stage2
