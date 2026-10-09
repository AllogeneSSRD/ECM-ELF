#pragma once
#include <limits>

// Shared integer geometry. These are component payloads/planning estimates,
// never a sum of simultaneous process allocations or a promise of residency.
namespace ecm_stage2 {
using Word = unsigned long long;
constexpr int max_input_bits = 16384;
constexpr int max_words = max_input_bits / 64;
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
// Per-request NTT payload only: packed A/B[/Q], digit output and two verdict
// words per slice. Caches, retained capacities and reduced S4 output are separate.
inline bool chunk_request_bytes(Word n, Word out_slots, Word slices,
                                unsigned buffers, Word &bytes) {
    Word words=0;
    return n && slices && (buffers==2 || buffers==3) &&
        multiply(n,buffers,words) && add(words,out_slots,words) &&
        add(words,2,words) && multiply(words,slices,words) && multiply(words,8,bytes);
}
// Preserve the engine's halving sequence (nbatch, nbatch/2, ...), including a
// mandatory single slice when even one request exceeds the soft chunk budget.
inline Word chunk_slices(Word n, Word out_slots, Word nbatch, Word budget,
                         unsigned buffers, bool verdicts=true) {
    for(Word c=nbatch;c;c/=2) {
        Word bytes=0, words=0;
        const bool ok=verdicts ? chunk_request_bytes(n,out_slots,c,buffers,bytes) :
            (multiply(n,buffers,words) && add(words,out_slots,words) &&
             multiply(words,c,words) && multiply(words,8,bytes));
        if(ok && bytes<=budget)return c;
    }
    return 1;
}
// bit 0 reuses q for qb; bit 1 reuses G for reverse(T)/reverse(q).
// Both transitions are ordered on the engine's default stream, after the old
// coefficients have been gathered/digested or reversed. Source/result remain
// separate allocations, so packing never reads an in-place NTT destination.
struct FoldOwnerLayout {
    Word source_words=0, result_words=0, bytes=0;
    Word f=0, inv=0, h=0, g=0, reverse=0, t=0, q=0, qb=0;
};
inline bool fold_owner_layout(Word p, Word w, unsigned reuse, FoldOwnerLayout &l) {
    l=FoldOwnerLayout{};
    if (reuse>3) return false;
    Word cell=0, v=0, total=0;
    if (!add(p,1,cell) || !multiply(cell,w,cell) ||
        !multiply(cell,2,l.h) || !multiply(cell,3,l.g) ||
        !multiply(cell,(reuse&2)?4:5,l.source_words)) return false;
    l.inv=cell;
    if (reuse&2) l.reverse=l.g;
    else if (!multiply(cell,4,l.reverse)) return false;
    if (!multiply(p,2,v) || !add(v,1,v) || !multiply(v,w,l.q)) return false;
    if (reuse&1) l.qb=l.q;
    else if (!multiply(p,3,v) || !add(v,2,v) || !multiply(v,w,l.qb)) return false;
    if (!multiply(p,(reuse&1)?3:4,v) || !add(v,2,v) ||
        !multiply(v,w,l.result_words) || !add(l.source_words,l.result_words,total) ||
        !add(total,w,total) || !multiply(total,8,total) || !add(total,48,l.bytes)) return false;
    return true;
}
inline Word owner_bytes(Word p, Word w, unsigned reuse=0) {
    FoldOwnerLayout l;
    return fold_owner_layout(p,w,reuse,l) ? l.bytes : std::numeric_limits<Word>::max();
}
struct Geometry {
    Word p=0, bits=0, words=0, fold_length=0, tree_length=0;
    Word workspace_buffers=3;
    Word fold_big_bytes=0, arena_estimate_bytes=0, fold_owner_bytes=0;
};
// query(coefficients, bits, &length, &output_slots) is the actual NTT backend.
template<class Query> bool geometry(Word p, int bits, Query query, Geometry &g, unsigned owner_reuse=0,
                                   unsigned workspace_buffers=3) {
    g=Geometry{};
    if (!p || p==std::numeric_limits<Word>::max() || bits<2 || bits>max_input_bits ||
        (workspace_buffers!=2 && workspace_buffers!=3)) return false;
    Word nf=0, nt=0, of=0, ot=0, wf=0, wt=0, total=0;
    if (!query(p+1,bits,&nf,&of) || !query(p/2+1,bits,&nt,&ot) ||
        !multiply(nf,workspace_buffers,wf) || !add(wf,of,wf) ||
        !multiply(nt,workspace_buffers,wt) || !add(wt,ot,wt) || !multiply(wt,2,wt) ||
        !add(wf,wt,total) || !multiply(total,8,total)) return false;
    g.p=p; g.bits=bits; g.words=(bits+63)/64; g.fold_length=nf; g.tree_length=nt;
    g.workspace_buffers=workspace_buffers;
    g.arena_estimate_bytes=total;
    if (!multiply(nf,8*workspace_buffers,g.fold_big_bytes)) return false;
    g.fold_owner_bytes=owner_bytes(p,g.words,owner_reuse);
    return g.fold_owner_bytes!=std::numeric_limits<Word>::max();
}
struct Plan {
    Geometry geometry;
    Word target_bits=0, carrier_exponent=0;
    Word d=0, b1=0, b2=0, giant_points=0, batches=0;
    Word free_bytes=0, arena_cap_bytes=0, owner_budget_bytes=0, baby_bytes=0;
    double estimated_seconds=0;
    bool calibrated=false, owner_budget_fits=false, arena_estimate_fits=false;
    const char *model="unavailable";
};
} // namespace ecm_stage2
