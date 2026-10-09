#pragma once
#include "ecm_stage2_geometry.h"
#include <vector>

// Predictive request topology, independent of CUDA allocation or coefficient data.
// Contract: resident batched path, non-trimmed fold remainder (P coefficients),
// cached inverse, scaled root/frontier, and no diagnostic extra multiplies.
// This is NOT a process memory/admission model. Allocation order, evictions,
// selftest temporaries and non-NTT lifetimes still require a separate simulator.
namespace ecm_stage2 {
enum RequestPhase { RequestFtree, RequestGtrees, RequestFold, RequestDescent,
                    RequestInverse, RequestPhaseCount };
inline const char *request_phase_name(unsigned phase) {
    const char *names[]={"ftree","gtrees","fold","descent","inverse"};
    return phase<RequestPhaseCount?names[phase]:"invalid";
}
struct MultiplyRequest {
    unsigned phase=0;
    Word ma=0,mb=0,pairs=0,first=0,count=0;
};
struct RequestBlock {
    Word repeat=1;
    std::vector<MultiplyRequest> requests;
};
struct RequestProgram {
    std::vector<RequestBlock> blocks;
    bool supported=false;
    const char *reason="invalid_geometry";
};
// Polynomial rolling signature in Z/(2^64). Unsigned wrap is intentional.
// It audits ordering/shapes, not cryptographic identity (the collector uses SHA).
struct RequestSignature {
    Word multiplier=1,addend=0;
    static constexpr Word radix=0x9e3779b185ebca87ull;
    void append_word(Word x) {multiplier*=radix;addend=addend*radix+x;}
    void append(const MultiplyRequest &r,Word n,Word slots,Word chunk) {
        for(Word x:{Word(r.phase),r.ma,r.mb,r.pairs,r.first,r.count,n,slots,chunk})append_word(x);
    }
    // Compose a later sequence; h -> (h * this.multiplier + this.addend).
    void then(const RequestSignature &later) {
        addend=addend*later.multiplier+later.addend;multiplier*=later.multiplier;
    }
    void repeat(Word times) {
        RequestSignature result,power=*this;
        while(times) {if(times&1)result.then(power);power.then(power);times>>=1;}
        *this=result;
    }
};
inline bool append_tree_requests(Word p,unsigned phase,RequestBlock &block) {
    return tree_multiply_groups(p,[&](Word a,Word b,Word nb) {
        Word count=0;if(!add(a,b-1,count))return false;
        block.requests.push_back({phase,a,b,nb,0,count});return true;
    });
}
inline bool append_fold_requests(Word p,Word ng,Word h,RequestBlock &block) {
    Word nt=0;if(!ng || !h || !add(ng,h-1,nt))return false;
    block.requests.push_back({RequestFold,ng,h,1,0,nt});
    if(nt>p) {
        const Word k=nt-p;
        block.requests.push_back({RequestFold,k,k,1,0,k});
        block.requests.push_back({RequestFold,k,p+1,1,0,p});
    }
    return true;
}
inline bool request_program(Word p,Word giant_points,RequestProgram &program) {
    program=RequestProgram{};
    if(!p || !giant_points || p>std::numeric_limits<Word>::max()/2)return false;
    // One full G root is monic of degree P and needs a separate root division.
    // Do not silently substitute the resident scaled-root topology for it.
    if(giant_points<=p) {program.reason="single_batch_local_inverse_or_root_division";return true;}
    RequestBlock initial;
    if(!append_tree_requests(p,RequestFtree,initial))return false;
    for(Word g=1;g<p+1;) {
        const Word next=std::min(2*g,p+1);
        initial.requests.push_back({RequestInverse,next,g,1,0,next});
        initial.requests.push_back({RequestInverse,g,next,1,0,next});
        g=next;
    }
    program.blocks.push_back(std::move(initial));
    const Word full=giant_points/p,rem=giant_points%p;
    if(full) {
        RequestBlock first;
        if(!append_tree_requests(p,RequestGtrees,first))return false;
        program.blocks.push_back(std::move(first));
    }
    if(full>=2) {
        RequestBlock second;
        if(!append_tree_requests(p,RequestGtrees,second) ||
           !append_fold_requests(p,p+1,p+1,second))return false;
        program.blocks.push_back(std::move(second));
    }
    if(full>2) {
        RequestBlock middle;middle.repeat=full-2;
        if(!append_tree_requests(p,RequestGtrees,middle) ||
           !append_fold_requests(p,p+1,p,middle))return false;
        program.blocks.push_back(std::move(middle));
    }
    if(rem) {
        RequestBlock last;
        if(!append_tree_requests(rem,RequestGtrees,last) ||
           (full && !append_fold_requests(p,rem+1,full==1?p+1:p,last)))return false;
        program.blocks.push_back(std::move(last));
    }
    RequestBlock descent;
    descent.requests.push_back({RequestDescent,p,p,1,0,p});
    // Dense left-packed padded degrees, without materializing O(P) nodes.
    // Map key/order exactly matches the scaled frontier's (child, sibling).
    Word pad=1;while(pad<p)pad*=2;
    for(Word h=pad;h>1;h/=2) {
        const Word half=h/2,full_nodes=p/h,r=p%h;
        std::map<std::pair<Word,Word>,Word> groups;
        if(full_nodes)groups[{half,half}]=2*full_nodes;
        if(r>half) {++groups[{half,r-half}];++groups[{r-half,half}];}
        for(const auto &entry:groups) {
            const Word a=entry.first.first,b=entry.first.second;
            descent.requests.push_back({RequestDescent,a+b,b+1,entry.second,b,a});
        }
    }
    program.blocks.push_back(std::move(descent));
    program.supported=true;program.reason="resident_full_fold_degree";return true;
}
struct RequestPlan {
    RequestProgram program;
    TreeWorkspacePlan phases[RequestPhaseCount],retained;
    RequestSignature signatures[RequestPhaseCount],signature;
    bool valid=false;
};
// NTT payload retention for the ENTIRE conditional request program, no eviction.
// Repeated G-tree/fold blocks are summarized in O(log repeat), not expanded.
template<class Query,class Describe> bool request_plan(Word p,Word giant_points,int bits,
        Query query,Describe describe,Word budget,unsigned buffers,bool physical,
        Word chunk_max,bool pool,RequestPlan &plan) {
    plan=RequestPlan{};
    if(bits<2 || bits>max_input_bits || (buffers!=2 && buffers!=3) ||
       !request_program(p,giant_points,plan.program))return false;
    if(!plan.program.supported)return true;
    const Word w=(bits+63)/64;
    std::map<std::pair<Word,Word>,Word> digits,bigs;
    for(const auto &block:plan.program.blocks) {
        RequestSignature signature,phase_signatures[RequestPhaseCount];
        for(const auto &r:block.requests) {
            Word n=0,slots=0,pairs=0,chunks=0,bytes=0,out=0;
            const Word operand=std::max(r.ma,r.mb);
            if(!query(operand,bits,&n,&slots) || !n || slots!=2*operand-1)return false;
            Word c=chunk_slices(n,slots,r.pairs,budget,physical?buffers:3,physical);
            if(chunk_max)c=std::min(c,chunk_max);
            if(!multiply(r.pairs,block.repeat,pairs) ||
               !multiply(r.pairs/c+(r.pairs%c!=0),block.repeat,chunks) ||
               !multiply(n,c,bytes) || !multiply(bytes,8*buffers,bytes) ||
               !multiply(r.count,c,out) || !multiply(out,8*w,out))return false;
            auto &phase=plan.phases[r.phase];
            if(!add(phase.groups,block.repeat,phase.groups) || !add(phase.pairs,pairs,phase.pairs) ||
               !add(phase.chunks,chunks,phase.chunks))return false;
            phase.big_peak_bytes=std::max(phase.big_peak_bytes,bytes);
            phase.output_peak_bytes=std::max(phase.output_peak_bytes,out);
            plan.retained.big_peak_bytes=std::max(plan.retained.big_peak_bytes,bytes);
            plan.retained.output_peak_bytes=std::max(plan.retained.output_peak_bytes,out);
            for(Word slices:{c,r.pairs%c})if(slices) {
                Word small=0,keyed=0;
                if(!add(slots,2,small) || !multiply(small,slices,small) || !multiply(small,8,small) ||
                   !multiply(n,slices,keyed) || !multiply(keyed,24,keyed))return false;
                auto &entry=digits[{n,slices}];entry=std::max(entry,small);
                bigs[{n,slices}]=keyed;
            }
            if(!plan.retained.caches.count(n)) {
                TreeWorkspacePlan::Cache cache;
                if(!describe(n,cache.table,cache.base) ||
                   !add(plan.retained.table_retained_bytes,cache.table,plan.retained.table_retained_bytes) ||
                   !add(plan.retained.base_retained_bytes,cache.base,plan.retained.base_retained_bytes))return false;
                plan.retained.caches.emplace(n,cache);
            }
            signature.append(r,n,slots,c);phase_signatures[r.phase].append(r,n,slots,c);
        }
        signature.repeat(block.repeat);plan.signature.then(signature);
        for(unsigned i=0;i<RequestPhaseCount;++i) {
            phase_signatures[i].repeat(block.repeat);plan.signatures[i].then(phase_signatures[i]);
        }
    }
    for(const auto &entry:digits)
        if(!add(plan.retained.digit_retained_bytes,entry.second,plan.retained.digit_retained_bytes))return false;
    for(const auto &entry:bigs)
        if(!add(plan.retained.keyed_big_retained_bytes,entry.second,plan.retained.keyed_big_retained_bytes))return false;
    Word total=pool?plan.retained.big_peak_bytes:plan.retained.keyed_big_retained_bytes;
    if(!add(total,plan.retained.digit_retained_bytes,total) || !add(total,plan.retained.table_retained_bytes,total) ||
       !add(total,plan.retained.base_retained_bytes,total))return false;
    plan.retained.ntt_retained_bytes=total;plan.valid=true;return true;
}
} // namespace ecm_stage2
