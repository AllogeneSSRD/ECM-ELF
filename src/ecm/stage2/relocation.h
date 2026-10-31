#pragma once
#include "giant_schedule.h"

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
} // namespace ecm_stage2
