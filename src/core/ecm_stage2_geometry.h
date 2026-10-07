#pragma once
#include <limits>

// Shared integer geometry. These are component payloads/planning estimates,
// never a sum of simultaneous process allocations or a promise of residency.
namespace ecm_stage2 {
using Word = unsigned long long;
// Points per sequential chain thread; the CUDA block itself has 64 threads.
// A disabled short policy preserves the base value, including large chunks.
inline Word giant_chain_block(Word points, Word base, Word short_block, Word short_max) {
    return short_block && points < short_max && short_block < base ? short_block : base;
}
inline bool add(Word a, Word b, Word &out) {
    if (a > std::numeric_limits<Word>::max()-b) return false;
    out=a+b; return true;
}
inline bool multiply(Word a, Word b, Word &out) {
    if (b && a > std::numeric_limits<Word>::max()/b) return false;
    out=a*b; return true;
}
inline Word owner_bytes(Word p, Word w) {
    Word v=0;
    if (!multiply(p,9,v) || !add(v,8,v) || !multiply(v,w,v) ||
        !multiply(v,8,v) || !add(v,48,v)) return std::numeric_limits<Word>::max();
    return v;
}
struct Geometry {
    Word p=0, bits=0, words=0, fold_length=0, tree_length=0;
    Word fold_big_bytes=0, arena_estimate_bytes=0, fold_owner_bytes=0;
};
// query(coefficients, bits, &length, &output_slots) is the actual NTT backend.
template<class Query> bool geometry(Word p, int bits, Query query, Geometry &g) {
    g=Geometry{};
    if (!p || p==std::numeric_limits<Word>::max() || bits<2 || bits>8192) return false;
    Word nf=0, nt=0, of=0, ot=0, wf=0, wt=0, total=0;
    if (!query(p+1,bits,&nf,&of) || !query(p/2+1,bits,&nt,&ot) ||
        !multiply(nf,3,wf) || !add(wf,of,wf) ||
        !multiply(nt,3,wt) || !add(wt,ot,wt) || !multiply(wt,2,wt) ||
        !add(wf,wt,total) || !multiply(total,8,total)) return false;
    g.p=p; g.bits=bits; g.words=(bits+63)/64; g.fold_length=nf; g.tree_length=nt;
    g.arena_estimate_bytes=total;
    if (!multiply(nf,24,g.fold_big_bytes)) return false;
    g.fold_owner_bytes=owner_bytes(p,g.words);
    return g.fold_owner_bytes!=std::numeric_limits<Word>::max();
}
struct Plan {
    Geometry geometry;
    Word d=0, b1=0, b2=0, giant_points=0, batches=0;
    Word free_bytes=0, arena_cap_bytes=0, owner_budget_bytes=0, baby_bytes=0;
    double estimated_seconds=0;
    bool calibrated=false, owner_budget_fits=false, arena_estimate_fits=false;
    const char *model="unavailable";
};
} // namespace ecm_stage2
