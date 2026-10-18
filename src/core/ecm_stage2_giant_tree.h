#pragma once
#include "ecm_stage2_geometry.h"
#include <vector>

namespace ecm_stage2 {
// Shape/reducer setup and extra copies need repeated wide polynomial work to
// amortize. Retain the dense layout for narrow or short Stage2 executions.
inline bool giant_tree_balanced(Word words,Word points,Word p) {
    return words>128 && p && points/p>=4;
}
// Keep dense power-of-two tiles at the bottom, distributing at most 128 tiles
// across the upper complete tree. Only G roots use this layout; the F tree and
// its ordered descent stay left packed. Empty slots represent constant one.
struct GiantTreeLayout {
    Word pad=1,tile=1,slots=1,tiles=0;
    Word leaf(Word i)const {return ((i/tile)*slots/tiles)*tile+i%tile;}
};
inline bool giant_tree_layout(Word n,GiantTreeLayout &out) {
    out=GiantTreeLayout{};
    while(out.pad<n)if(!multiply(out.pad,2,out.pad))return false;
    out.tile=std::min(out.pad,std::max(Word(16),out.pad/128));
    out.slots=out.pad/out.tile;out.tiles=n/out.tile+(n%out.tile!=0);
    return true;
}
// Compressed shape enumeration matches the resident (min,max) operand map at
// each level. Bottom tiles have the same groups as a dense prefix; only the
// bounded upper tree is materialized, independent of the full leaf count.
template<class Visit> bool giant_tree_multiply_groups(Word n,Visit visit) {
    GiantTreeLayout l;if(!giant_tree_layout(n,l))return false;
    for(Word h=1;h<l.tile;h*=2) {
        const Word pairs=n/h/2,r=n-h*(2*pairs);
        if(r>h && !visit(r-h+1,h+1,Word(1)))return false;
        if(pairs && !visit(h+1,h+1,pairs))return false;
    }
    std::vector<Word> degree(2*l.slots);
    for(Word i=0;i<l.tiles;++i)degree[l.slots+i*l.slots/l.tiles]=std::min(l.tile,n-i*l.tile);
    for(Word i=l.slots;--i;)degree[i]=degree[2*i]+degree[2*i+1];
    for(Word base=l.slots/2;base;base/=2) {
        std::map<std::pair<Word,Word>,Word> groups;
        for(Word i=base;i<2*base;++i) {
            const auto a=degree[2*i],b=degree[2*i+1];
            if(a && b)++groups[{std::min(a,b)+1,std::max(a,b)+1}];
        }
        for(const auto &g:groups)if(!visit(g.first.first,g.first.second,g.second))return false;
    }
    return true;
}
} // namespace ecm_stage2
