#pragma once
#include "geometry.h"
#include <vector>
#include <cstdlib>

namespace ecm_stage2 {
inline thread_local int groot_layout_override=-1;
struct GrootLayoutScope {
    int previous=groot_layout_override;
    explicit GrootLayoutScope(int layout){groot_layout_override=layout;}
    ~GrootLayoutScope(){groot_layout_override=previous;}
};
inline bool legacy_groot_layout(Word words,Word points,Word p) {
    return words>128 && p && points/p>=4;
}
// Override/legacy resolver. New automatic planning also evaluates NTT cliffs.
inline bool giant_tree_balanced(Word words,Word points,Word p) {
    // Explicit controls, native memory planning and execution share one layout.
    if(const char *forced=std::getenv("NTT_GROOT_BALANCED"))return forced[0]=='1';
    if(groot_layout_override>=0)return groot_layout_override!=0;
    return legacy_groot_layout(words,points,p);
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
// Preserve qualified wide defaults. Narrow additions need an actual root NTT
// cliff and >=10% less complete G-tree N*log2(N) work, with <=10% extra output
// coefficients. No arithmetic/memory admission is replaced by this cost gate.
template<class Query> bool select_groot_layout(Word words,Word points,Word p,int bits,Query query) {
    if(std::getenv("NTT_GROOT_BALANCED") || groot_layout_override>=0)
        return giant_tree_balanced(words,points,p);
    const bool legacy=legacy_groot_layout(words,points,p);
    if(legacy || !p || points/p<4)return legacy;
    GiantTreeLayout layout;if(!giant_tree_layout(p,layout) || p==layout.pad)return false;
    Word left=0;
    for(Word i=0;i<layout.tiles;++i)if(i*layout.slots/layout.tiles<layout.slots/2)
        left+=std::min(layout.tile,p-i*layout.tile);
    Word before=0,after=0,slots=0;
    if(!query(layout.pad/2+1,bits,&before,&slots) ||
       !query(std::max(left,p-left)+1,bits,&after,&slots) || after>=before)return false;
    auto cost=[&](bool balanced,double &ntt,double &coeff) {
        ntt=coeff=0;
        auto tree=[&](Word count,Word repeat) {
            auto visit=[&](Word a,Word b,Word pairs) {
                Word n=0,out=0;if(!query(std::max(a,b),bits,&n,&out) || !n)return false;
                ntt+=double(repeat)*pairs*n*std::log2(double(n));
                coeff+=double(repeat)*pairs*(a+b-1);return true;
            };
            return balanced?giant_tree_multiply_groups(count,visit):tree_multiply_groups(count,visit);
        };
        return tree(p,points/p) && (!(points%p) || tree(points%p,1));
    };
    double a=0,b=0,ca=0,cb=0;
    return cost(false,a,ca) && cost(true,b,cb) && b<=.90*a && cb<=1.10*ca;
}
} // namespace ecm_stage2
