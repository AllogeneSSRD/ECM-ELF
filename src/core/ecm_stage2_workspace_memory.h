#pragma once
#include "ecm_stage2_ntt_memory.h"
#include "ecm_stage2_s4_program.h"
#include "ecm_stage2_owner_memory.h"
#include "ecm_stage2_giant_state.h"
#include <optional>

namespace ecm_stage2 {
// Interleave both allocators in native request order. This is the NTT+S4
// component, optionally including fold/frontier and giant events. This is not
// a complete admission proof: earlier initialization and fallback remain unmodeled.
struct WorkspaceMemoryEvent {
    bool ntt=false,allocation=false;
    Word bytes=0,ntt_live=0,s4_live=0,total=0;
    Word owner_live=0;
    bool owner_event=false;
    Word giant_live=0;
    bool giant_event=false;
};
struct WorkspaceMemoryPlan {
    bool valid=false,finished=false;
    const char *reason="unsupported_request_program";
    Word peak_bytes=0,ntt_peak_bytes=0,s4_peak_bytes=0;
    Word ntt_at_peak=0,s4_at_peak=0,final_bytes=0,released_bytes=0;
    Word owner_at_peak=0,owner_peak_bytes=0,fold_bytes=0,frontier_bytes=0;
    Word giant_at_peak=0,giant_peak_bytes=0,giant_final_bytes=0,point_chunks=0,points_consumed=0;
    Word executed_blocks=0,skipped_blocks=0,simulated_events=0;
    NttMemoryPayload ntt_final;
    S4MemoryPayload s4_final;
};
template<class Query,class Describe> bool workspace_memory_plan(
        const RequestProgram &program,int bits,Query query,Describe describe,
        NttMemoryPolicy ntt_policy,S4ProgramPolicy s4_policy,WorkspaceMemoryPlan &plan,
        bool compress=true,std::function<void(const WorkspaceMemoryEvent &)> observe={},
        const OwnerMemoryPolicy *owner_policy=nullptr,const GiantTimelinePolicy *giant_policy=nullptr) {
    plan=WorkspaceMemoryPlan{};
    if(!program.supported || bits<2 || bits>max_input_bits)return true;
    const unsigned buffers=ntt_policy.pool && ntt_policy.reuse_bq?2:3;
    if(!s4_policy.batch_bytes || s4_policy.buffers!=buffers) {
        plan.reason="inconsistent_workspace_policy";return false;
    }
    NttMemoryState ntt(ntt_policy);S4MemoryState s4;
    OwnerMemoryState owner(owner_policy?*owner_policy:OwnerMemoryPolicy{});
    GiantMemoryState giant(giant_policy?*giant_policy:GiantTimelinePolicy{});
    if(owner_policy && (!owner_policy->p || owner_policy->words!=(Word)((bits+63)/64))) {
        plan.reason="inconsistent_owner_policy";return false;
    }
    if(giant_policy && (giant_policy->words!=(Word)((bits+63)/64) ||
       (owner_policy && giant_policy->p!=owner_policy->p))) {
        plan.reason="inconsistent_giant_policy";return false;
    }
    bool overflow=false;
    auto event=[&](bool from_ntt,bool allocation,Word bytes,bool from_owner=false,bool from_giant=false) {
        Word total=0;
        if(!add(ntt.live.total,s4.live.total,total) || !add(total,owner.live,total) || !add(total,giant.live,total) ||
           !add(plan.simulated_events,1,plan.simulated_events)){overflow=true;return;}
        if(total>plan.peak_bytes) {
            plan.peak_bytes=total;plan.ntt_at_peak=ntt.live.total;plan.s4_at_peak=s4.live.total;
            plan.owner_at_peak=owner.live;
            plan.giant_at_peak=giant.live;
        }
        if(observe)observe({from_ntt,allocation,bytes,ntt.live.total,s4.live.total,total,owner.live,from_owner,giant.live,from_giant});
    };
    ntt.observe=[&](const NttMemoryEvent &e){event(true,e.allocation,e.bytes);};
    s4.observe=[&](const S4MemoryEvent &e){event(false,e.allocation,e.bytes);};
    owner.observe=[&](const OwnerMemoryEvent &e){event(false,e.allocation,e.bytes,true);};
    giant.observe=[&](const GiantMemoryEvent &e){event(false,e.allocation,e.bytes,false,true);};
    auto save=[&]() {
        plan.ntt_final=ntt.live;plan.s4_final=s4.live;
        plan.ntt_peak_bytes=ntt.peak;plan.s4_peak_bytes=s4.peak;
        plan.owner_peak_bytes=owner.peak;plan.fold_bytes=owner.fold_bytes;plan.frontier_bytes=owner.frontier_bytes;
        plan.giant_peak_bytes=giant.peak;plan.giant_final_bytes=giant.live;
        plan.point_chunks=giant.chunks;plan.points_consumed=giant.consumed_points;
        return add(ntt.live.total,s4.live.total,plan.final_bytes) && add(plan.final_bytes,owner.live,plan.final_bytes) &&
            add(plan.final_bytes,giant.live,plan.final_bytes);
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
            if(owner_policy && !owner.fold_begin()) {
                plan.reason=owner.reason;save();plan.valid=!std::strcmp(owner.reason,"fold_budget_refusal");return plan.valid;
            }
        }
        if(!block.requests.empty() && block.requests.front().phase==RequestDescent && !descent_started) {
            descent_started=true;if(!trim())return fail(s4.reason);
            if(owner_policy && !owner.frontier_begin()) {
                plan.reason=owner.reason;save();plan.valid=!std::strcmp(owner.reason,"frontier_budget_refusal");return plan.valid;
            }
        }
        Word remaining=block.repeat;
        struct Cycle {NttMemoryState ntt;S4MemoryState s4;OwnerMemoryState owner;GiantMemoryState giant;Word blocks=0;};
        std::optional<Cycle> cycle;
        while(remaining) {
            const auto before_ntt=ntt;const auto before_s4=s4;
            const auto before_owner=owner;
            bool tree=block.tree_leaves!=0;
            if(tree && giant_policy) {
                // Compare an entire point chunk, not individual G trees. The
                // coordinates can span several repeated trees and folds.
                if(compress && !giant.chunk_remaining && block.tree_leaves==giant_policy->p &&
                   remaining>=giant_policy->chunk_points/giant_policy->p &&
                   giant_policy->points-giant.consumed_points>=giant_policy->chunk_points)
                    cycle=Cycle{ntt,s4,owner,giant,0};
                if(!giant.chunk_remaining && !giant.chunk_begin())return fail(giant.reason);
            }
            if(tree && !s4.tree_begin(block.tree_leaves,s4_policy.compact_raw))return fail(s4.reason);
            for(const auto &r:block.requests) {
                // S3Workspace is constructed after F-tree and before inverse;
                // small-prime ladders may already have reserved seed arrays.
                if(giant_policy && r.phase==RequestInverse && !giant.started && !giant.init())return fail(giant.reason);
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
            if(block.tree_leaves && giant_policy && !giant.tree_done(block.tree_leaves))return fail(giant.reason);
            if(overflow)return fail("payload_overflow");
            if(!add(plan.executed_blocks,1,plan.executed_blocks))return fail("counter_overflow");
            --remaining;
            if(cycle) {
                ++cycle->blocks;
                if(!giant.chunk_remaining) {
                    if(ntt.same_allocations(cycle->ntt) && s4.same_allocations(cycle->s4) &&
                       owner.same_allocations(cycle->owner) && giant.same_allocations(cycle->giant)) {
                        const Word count=std::min(remaining/cycle->blocks,
                            (giant_policy->points-giant.consumed_points)/giant_policy->chunk_points);
                        Word skipped=0;
                        if(!multiply(count,cycle->blocks,skipped) || !add(plan.skipped_blocks,skipped,plan.skipped_blocks) ||
                           !giant.skip_full_chunks(count))return fail("invalid_joint_cycle_skip");
                        remaining-=skipped;
                    }
                    cycle.reset();
                }
            }
            if(compress && remaining && !(giant_policy && block.tree_leaves) &&
               ntt.same_allocations(before_ntt) && s4.same_allocations(before_s4) &&
               owner.same_allocations(before_owner)) {
                if(!add(plan.skipped_blocks,remaining,plan.skipped_blocks))return fail("counter_overflow");
                break;
            }
        }
    }
    if(!giant_started || !descent_started)return fail("missing_phase_boundary");
    if(owner_policy && !owner.close())return fail(owner.reason);
    if(giant_policy && !giant.accumulate())return fail(giant.reason);
    if(!save())return fail("payload_overflow");
    if(giant_policy && !giant.close())return fail(giant.reason);
    // run_real destroys reducer/context before NttArena; release in that order.
    if(!s4.close() || !ntt.close())return fail("release_failed");
    if(overflow || !add(ntt.live.total,s4.live.total,plan.released_bytes))return fail("payload_overflow");
    plan.valid=plan.finished=true;plan.reason="ok";return true;
}
} // namespace ecm_stage2
