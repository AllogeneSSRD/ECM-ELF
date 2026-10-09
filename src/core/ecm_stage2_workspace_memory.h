#pragma once
#include "ecm_stage2_ntt_memory.h"
#include "ecm_stage2_s4_program.h"

namespace ecm_stage2 {
// Interleave both allocators in native request order. This is the NTT+S4
// component, not a complete curve admission proof: point/fold/frontier owners,
// device context and nonresident fallback remain the caller's responsibility.
struct WorkspaceMemoryEvent {
    bool ntt=false,allocation=false;
    Word bytes=0,ntt_live=0,s4_live=0,total=0;
};
struct WorkspaceMemoryPlan {
    bool valid=false,finished=false;
    const char *reason="unsupported_request_program";
    Word peak_bytes=0,ntt_peak_bytes=0,s4_peak_bytes=0;
    Word ntt_at_peak=0,s4_at_peak=0,final_bytes=0,released_bytes=0;
    Word executed_blocks=0,skipped_blocks=0,simulated_events=0;
    NttMemoryPayload ntt_final;
    S4MemoryPayload s4_final;
};
template<class Query,class Describe> bool workspace_memory_plan(
        const RequestProgram &program,int bits,Query query,Describe describe,
        NttMemoryPolicy ntt_policy,S4ProgramPolicy s4_policy,WorkspaceMemoryPlan &plan,
        bool compress=true,std::function<void(const WorkspaceMemoryEvent &)> observe={}) {
    plan=WorkspaceMemoryPlan{};
    if(!program.supported || bits<2 || bits>max_input_bits)return true;
    const unsigned buffers=ntt_policy.pool && ntt_policy.reuse_bq?2:3;
    if(!s4_policy.batch_bytes || s4_policy.buffers!=buffers) {
        plan.reason="inconsistent_workspace_policy";return false;
    }
    NttMemoryState ntt(ntt_policy);S4MemoryState s4;
    bool overflow=false;
    auto event=[&](bool from_ntt,bool allocation,Word bytes) {
        Word total=0;
        if(!add(ntt.live.total,s4.live.total,total) ||
           !add(plan.simulated_events,1,plan.simulated_events)){overflow=true;return;}
        if(total>plan.peak_bytes) {
            plan.peak_bytes=total;plan.ntt_at_peak=ntt.live.total;plan.s4_at_peak=s4.live.total;
        }
        if(observe)observe({from_ntt,allocation,bytes,ntt.live.total,s4.live.total,total});
    };
    ntt.observe=[&](const NttMemoryEvent &e){event(true,e.allocation,e.bytes);};
    s4.observe=[&](const S4MemoryEvent &e){event(false,e.allocation,e.bytes);};
    auto save=[&]() {
        plan.ntt_final=ntt.live;plan.s4_final=s4.live;
        plan.ntt_peak_bytes=ntt.peak;plan.s4_peak_bytes=s4.peak;
        return add(ntt.live.total,s4.live.total,plan.final_bytes);
    };
    auto fail=[&](const char *why){plan.reason=why;save();return false;};
    if(!s4.init((bits+63)/64))return fail(s4.reason);
    bool giant_started=false,descent_started=false;
    auto trim=[&]() {return (!s4_policy.trim_raw || s4.raw_release()) &&
        (!s4_policy.trim_output || s4.output_release());};
    for(const auto &block:program.blocks) {
        if(!block.repeat)return fail("zero_repeat");
        if(block.tree_leaves && !giant_started) {
            giant_started=true;if(!trim())return fail(s4.reason);
        }
        if(!block.requests.empty() && block.requests.front().phase==RequestDescent && !descent_started) {
            descent_started=true;if(!trim())return fail(s4.reason);
        }
        Word remaining=block.repeat;
        while(remaining) {
            const auto before_ntt=ntt;const auto before_s4=s4;
            bool tree=block.tree_leaves!=0;
            if(tree && !s4.tree_begin(block.tree_leaves,s4_policy.compact_raw))return fail(s4.reason);
            for(const auto &r:block.requests) {
                if(tree && r.phase!=RequestGtrees) {
                    if(!s4.tree_end())return fail(s4.reason);tree=false;
                }
                const bool route=(r.input==RequestHost && (r.phase==RequestFtree || r.phase==RequestInverse)) ||
                    (r.input==RequestTreeRaw && tree && r.phase==RequestGtrees) ||
                    (r.input==RequestFoldOwner && (r.phase==RequestFold || r.phase==RequestDescent)) ||
                    (r.input==RequestFrontierOwner && r.phase==RequestDescent);
                const Word operand=std::max(r.ma,r.mb);S4ShapeDescriptor d;
                if(!route || !r.ma || !r.mb || !r.pairs || !r.count ||
                   !query(operand,bits,d) || !d.n || (d.n&(d.n-1)) ||
                   d.slots!=2*operand-1 || r.first>d.slots || r.count>d.slots-r.first)
                    return fail("invalid_request_route_or_shape");
                Word c=chunk_slices(d.n,d.slots,r.pairs,s4_policy.batch_bytes,
                    s4_policy.physical_chunks?buffers:3,s4_policy.physical_chunks);
                if(s4_policy.chunk_max)c=std::min(c,s4_policy.chunk_max);
                Word output=s4_policy.output_window?r.count:d.slots,need=0,raw=0;
                if(!multiply(s4_policy.chunk_output?c:r.pairs,output,need) ||
                   !multiply(need,(bits+63)/64,need) || !multiply(c,operand,raw) ||
                   !multiply(raw,(bits+63)/64,raw))return fail("payload_overflow");
                // Mandatory shape test precedes output/raw growth. NTT work
                // then precedes the first canonical counter allocation.
                if(!s4.shape(d.slot_bits,d.slot_words,d.bpw) || !s4.output_reserve(need) ||
                   (r.input==RequestHost && !s4.raw_reserve(raw,raw)))return fail(s4.reason);
                Word chunks=r.pairs/c+(r.pairs%c!=0);
                for(Word index=0;index<chunks;++index) {
                    const Word slices=std::min(c,r.pairs-index*c);
                    const auto before=ntt;
                    if(!ntt.call_layout(d.n,d.slots,slices,describe)) {
                        plan.reason=ntt.reason;save();
                        // A valid refusal prefix is useful evidence, but never
                        // licenses admission of the unmodeled malloc fallback.
                        plan.valid=!std::strcmp(ntt.reason,"fuse_cap_refusal") ||
                            !std::strcmp(ntt.reason,"big_cap_refusal") || !std::strcmp(ntt.reason,"digits_cap_refusal");
                        return plan.valid;
                    }
                    if(!s4.canonical_counter())return fail(s4.reason);
                    // Only full chunks can repeat unchanged. Preserve the
                    // distinct final tail and test the combined block later.
                    if(compress && slices==c && ntt.same_allocations(before) && chunks-index>2)
                        index=chunks-2;
                }
            }
            if(tree && !s4.tree_end())return fail(s4.reason);
            if(overflow)return fail("payload_overflow");
            if(!add(plan.executed_blocks,1,plan.executed_blocks))return fail("counter_overflow");
            --remaining;
            if(compress && remaining && ntt.same_allocations(before_ntt) && s4.same_allocations(before_s4)) {
                if(!add(plan.skipped_blocks,remaining,plan.skipped_blocks))return fail("counter_overflow");
                break;
            }
        }
    }
    if(!giant_started || !descent_started)return fail("missing_phase_boundary");
    if(!save())return fail("payload_overflow");
    // run_real destroys reducer/context before NttArena; release in that order.
    if(!s4.close() || !ntt.close())return fail("release_failed");
    if(overflow || !add(ntt.live.total,s4.live.total,plan.released_bytes))return fail("payload_overflow");
    plan.valid=plan.finished=true;plan.reason="ok";return true;
}
} // namespace ecm_stage2
