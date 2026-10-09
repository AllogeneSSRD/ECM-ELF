#pragma once
#include "ecm_stage2_requests.h"
#include "ecm_stage2_s4_memory.h"
#include <cstring>

// Conditional resident, direct-device-packing execution. This predicts S4 only;
// neither NTT/owner admission nor driver free memory is simulated here.
namespace ecm_stage2 {
struct S4ShapeDescriptor {
    Word n=0,slots=0,slot_bits=0,slot_words=0;
    unsigned bpw=0;
};
struct S4ProgramPolicy {
    Word batch_bytes=0,chunk_max=0;
    unsigned buffers=3;
    bool physical_chunks=false,output_window=true,chunk_output=true;
    bool compact_raw=true,trim_raw=false,trim_output=false;
};
struct S4ProgramCounters {
    Word requests=0,chunks=0,trees=0,allocations=0,frees=0;
    bool accumulate(const S4ProgramCounters &after,const S4ProgramCounters &before,Word times) {
        Word *dst[]={&requests,&chunks,&trees,&allocations,&frees};
        const Word a[]={after.requests,after.chunks,after.trees,after.allocations,after.frees};
        const Word b[]={before.requests,before.chunks,before.trees,before.allocations,before.frees};
        for(unsigned i=0;i<5;++i){Word delta=0;
            if(a[i]<b[i] || !multiply(a[i]-b[i],times,delta) || !add(*dst[i],delta,*dst[i]))return false;
        }
        return true;
    }
};
// Boundary records and request routes have a separate ordering signature. The
// older arithmetic-only RequestSignature remains byte-for-byte unchanged.
enum S4ProgramBoundary { S4InverseDone=1,S4TreeBegin,S4TreeEnd,S4FoldDone };
inline void s4_signature_boundary(RequestSignature &sig,unsigned kind,Word leaves=0) {
    sig.append_word(0x5334424f554e4400ull);sig.append_word(kind);sig.append_word(leaves);
}
inline void s4_signature_request(RequestSignature &sig,const MultiplyRequest &r,Word n,Word slots,Word chunk) {
    sig.append_word(0x5334524551554553ull);sig.append(r,n,slots,chunk);sig.append_word(r.input);
}
struct S4ProgramPlan {
    bool valid=false;
    const char *reason="unsupported_request_program";
    S4ProgramPolicy policy;
    S4MemoryPayload after_inverse,after_giant,final_payload,released_payload;
    S4ProgramCounters counters;
    RequestSignature signature;
    std::map<Word,S4ShapeDescriptor> descriptors;
    Word peak_bytes=0,shape_count=0,executed_blocks=0,skipped_blocks=0;
};
template<class Query> bool s4_program_plan(const RequestProgram &program,int bits,
        Query query,S4ProgramPolicy policy,S4ProgramPlan &plan) {
    plan=S4ProgramPlan{};plan.policy=policy;
    if(!program.supported || bits<2 || bits>max_input_bits)return true;
    if((policy.buffers!=2 && policy.buffers!=3) || !policy.batch_bytes) {
        plan.reason="invalid_policy";return false;
    }
    S4MemoryState state;
    state.observe=[&](const S4MemoryEvent &e) {
        Word &v=e.allocation?plan.counters.allocations:plan.counters.frees;
        if(!add(v,1,v))plan.reason="counter_overflow";
    };
    if(!state.init((bits+63)/64)){plan.reason=state.reason;return false;}
    bool giant_started=false,descent_started=false;
    const Word w=(bits+63)/64;
    auto fail=[&](const char *why){plan.reason=why;return false;};
    auto trim=[&]() {return (!policy.trim_raw || state.raw_release()) &&
        (!policy.trim_output || state.output_release());};
    for(const auto &block:program.blocks) {
        if(!block.repeat)return fail("zero_repeat");
        if(block.tree_leaves && !giant_started) {
            giant_started=true;plan.after_inverse=state.live;
            s4_signature_boundary(plan.signature,S4InverseDone);
            if(!trim())return fail(state.reason);
        }
        const bool descent=!block.requests.empty() && block.requests.front().phase==RequestDescent;
        if(descent && !descent_started) {
            descent_started=true;plan.after_giant=state.live;
            s4_signature_boundary(plan.signature,S4FoldDone);
            if(!trim())return fail(state.reason);
        }
        Word remaining=block.repeat;
        while(remaining) {
            const auto before=state;const auto before_counters=plan.counters;
            RequestSignature signature;
            bool tree=block.tree_leaves!=0;
            if(tree) {
                if(!state.tree_begin(block.tree_leaves,policy.compact_raw))return fail(state.reason);
                if(!add(plan.counters.trees,1,plan.counters.trees))return fail("counter_overflow");
                s4_signature_boundary(signature,S4TreeBegin,block.tree_leaves);
            }
            for(const auto &r:block.requests) {
                if(tree && r.phase!=RequestGtrees) {
                    if(!state.tree_end())return fail(state.reason);
                    tree=false;s4_signature_boundary(signature,S4TreeEnd,block.tree_leaves);
                }
                const bool route_ok=(r.input==RequestHost &&
                    (r.phase==RequestFtree || r.phase==RequestInverse)) ||
                    (r.input==RequestTreeRaw && tree && r.phase==RequestGtrees) ||
                    (r.input==RequestFoldOwner && (r.phase==RequestFold || r.phase==RequestDescent)) ||
                    (r.input==RequestFrontierOwner && r.phase==RequestDescent);
                const Word operand=std::max(r.ma,r.mb);
                S4ShapeDescriptor d;
                if(!route_ok || !r.ma || !r.mb || !r.pairs || !r.count ||
                   !query(operand,bits,d) || !d.n || (d.n&(d.n-1)) ||
                   d.slots!=2*operand-1 || r.first>d.slots || r.count>d.slots-r.first)
                    return fail("invalid_request_route_or_shape");
                plan.descriptors.emplace(operand,d);
                Word c=chunk_slices(d.n,d.slots,r.pairs,policy.batch_bytes,
                    policy.physical_chunks?policy.buffers:3,policy.physical_chunks);
                if(policy.chunk_max)c=std::min(c,policy.chunk_max);
                Word output=policy.output_window?r.count:d.slots,need=0,raw=0;
                if(!multiply(policy.chunk_output?c:r.pairs,output,need) || !multiply(need,w,need) ||
                   !multiply(c,operand,raw) || !multiply(raw,w,raw))return fail("payload_overflow");
                // Native: shape/selftest -> output -> host raw -> NTT -> first
                // canonical counter. Resident operands borrow their own owner.
                if(!state.shape(d.slot_bits,d.slot_words,d.bpw) || !state.output_reserve(need) ||
                   (r.input==RequestHost && !state.raw_reserve(raw,raw)) ||
                   !state.canonical_counter())return fail(state.reason);
                if(!add(plan.counters.requests,1,plan.counters.requests) ||
                   !add(plan.counters.chunks,r.pairs/c+(r.pairs%c!=0),plan.counters.chunks))
                    return fail("counter_overflow");
                s4_signature_request(signature,r,d.n,d.slots,c);
            }
            if(tree){if(!state.tree_end())return fail(state.reason);
                s4_signature_boundary(signature,S4TreeEnd,block.tree_leaves);}
            if(std::strcmp(plan.reason,"unsupported_request_program"))return false;
            --remaining;++plan.executed_blocks;plan.signature.then(signature);
            if(remaining && state.same_allocations(before)) {
                if(!plan.counters.accumulate(plan.counters,before_counters,remaining) ||
                   !add(plan.skipped_blocks,remaining,plan.skipped_blocks))return fail("counter_overflow");
                signature.repeat(remaining);plan.signature.then(signature);break;
            }
        }
    }
    if(!giant_started || !descent_started)return fail("missing_phase_boundary");
    plan.final_payload=state.live;plan.shape_count=state.shape_count();
    if(!state.close())return fail(state.reason);
    plan.released_payload=state.live;plan.peak_bytes=state.peak;
    if(std::strcmp(plan.reason,"unsupported_request_program"))return false;
    plan.valid=true;plan.reason="ok";return true;
}
} // namespace ecm_stage2
