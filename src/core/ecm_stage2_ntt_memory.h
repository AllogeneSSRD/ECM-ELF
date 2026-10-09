#pragma once
#include "ecm_stage2_requests.h"
#include <cstring>

// Shadow the arena's successful allocation path, with no CUDA calls. This is
// owned NTT payload, NOT device free memory or a complete Stage2 admission plan.
// Descriptors must use the same fixed fuse policy as the executing device.
namespace ecm_stage2 {
struct NttMemoryPolicy {
    Word cap_bytes=0; // zero means unlimited, as in NttArena
    bool pool=true,reuse_bq=false,carry_check=false;
};
struct NttMemoryPayload {
    Word big=0,digits=0,table=0,base=0,total=0;
};
struct NttMemoryCounters {
    Word calls=0,fuse_builds=0,fuse_hits=0,workspace_grows=0;
    Word cap_evictions=0,cap_evicted_bytes=0,cold_evictions=0,cold_evicted_bytes=0;
    Word carry_grows=0,carry_refusals=0;
    bool accumulate(const NttMemoryCounters &after,const NttMemoryCounters &before,Word times) {
        Word *dst[]={&calls,&fuse_builds,&fuse_hits,&workspace_grows,&cap_evictions,
            &cap_evicted_bytes,&cold_evictions,&cold_evicted_bytes,&carry_grows,&carry_refusals};
        const Word a[]={after.calls,after.fuse_builds,after.fuse_hits,after.workspace_grows,
            after.cap_evictions,after.cap_evicted_bytes,after.cold_evictions,after.cold_evicted_bytes,
            after.carry_grows,after.carry_refusals};
        const Word b[]={before.calls,before.fuse_builds,before.fuse_hits,before.workspace_grows,
            before.cap_evictions,before.cap_evicted_bytes,before.cold_evictions,before.cold_evicted_bytes,
            before.carry_grows,before.carry_refusals};
        for(unsigned i=0;i<10;++i) {Word delta=0;
            if(a[i]<b[i] || !multiply(a[i]-b[i],times,delta) || !add(*dst[i],delta,*dst[i]))return false;
        }
        return true;
    }
};
class NttMemoryState {
    struct Fuse {
        Word n=0,table=0,base=0;
        bool operator==(const Fuse &o)const{return n==o.n && table==o.table && base==o.base;}
    };
    // Fuse order matters: cold trim chooses the first largest context on ties.
    std::vector<Fuse> fuses;
    std::map<std::pair<Word,Word>,Word> bigs,smalls; // bytes and out_cap respectively
    Word workspace_words=0,carry_bytes=0;
    bool fits(Word extra)const {
        return !policy.cap_bytes || (extra<=policy.cap_bytes && live.total<=policy.cap_bytes-extra);
    }
    bool bump(Word &counter,Word value=1) {
        if(!add(counter,value,counter)){reason="counter_overflow";return false;}return true;
    }
    bool refresh() {
        NttMemoryPayload p;Word value=0;
        if(!multiply(workspace_words,policy.reuse_bq?16:24,p.big))return fail("payload_overflow");
        for(const auto &e:bigs)if(!add(p.big,e.second,p.big))return fail("payload_overflow");
        p.digits=carry_bytes;
        for(const auto &e:smalls) {
            if(!add(e.second,2,value) || !multiply(value,e.first.second,value) ||
               !multiply(value,8,value) || !add(p.digits,value,p.digits))return fail("payload_overflow");
        }
        for(const auto &f:fuses)
            if(!add(p.table,f.table,p.table) || !add(p.base,f.base,p.base))return fail("payload_overflow");
        if(!add(p.big,p.digits,p.total) || !add(p.total,p.table,p.total) ||
           !add(p.total,p.base,p.total))return fail("payload_overflow");
        live=p;peak=std::max(peak,p.total);return true;
    }
    bool fail(const char *why){reason=why;return false;}
    bool evict(Word n,Word slices) {
        // Runtime always drops carry scratch, but excludes it from tbl_words_freed.
        carry_bytes=0;Word freed=0;
        for(auto &f:fuses)if(f.n!=n) {
            if(!add(freed,f.table,freed))return fail("payload_overflow");f.table=0;
        }
        const auto keep=std::make_pair(n,slices);
        for(auto it=bigs.begin();it!=bigs.end();) {
            if(it->first==keep){++it;continue;}
            if(!add(freed,it->second,freed))return fail("payload_overflow");it=bigs.erase(it);
        }
        for(auto it=smalls.begin();it!=smalls.end();) {
            if(it->first==keep){++it;continue;}
            Word bytes=0;
            if(!add(it->second,2,bytes) || !multiply(bytes,it->first.second,bytes) ||
               !multiply(bytes,8,bytes) || !add(freed,bytes,freed))return fail("payload_overflow");
            it=smalls.erase(it);
        }
        if(freed && (!bump(counters.cap_evictions) || !bump(counters.cap_evicted_bytes,freed)))return false;
        return refresh();
    }
public:
    NttMemoryPolicy policy;
    NttMemoryPayload live;
    NttMemoryCounters counters;
    Word peak=0;
    const char *reason="ok";
    explicit NttMemoryState(NttMemoryPolicy p={}):policy(p){}
    // Equality excludes counters/peaks; a repeated block with identical live
    // allocation state has identical future transitions and counter deltas.
    bool same_allocations(const NttMemoryState &o)const {
        return workspace_words==o.workspace_words && carry_bytes==o.carry_bytes &&
            bigs==o.bigs && smalls==o.smalls && fuses==o.fuses;
    }
    Word shared_workspace_bytes()const{return workspace_words*(policy.reuse_bq?16:24);}
    template<class Describe> bool call(Word n,Word slots,Word slices,Describe describe) {
        if(!n || !slices || (n&(n-1)))return fail("invalid_shape");
        if(!bump(counters.calls))return false;
        // Shape planning allocates fuse tables/base BEFORE buffer lookup. The
        // cap check here does not evict; refusal invokes a per-call context.
        auto found=std::find_if(fuses.begin(),fuses.end(),[&](const Fuse &f){return f.n==n;});
        if(found==fuses.end()) {
            Word table=0,base=0,need=0;
            if(!describe(n,table,base) || !add(table,base,need))return fail("descriptor_overflow");
            if(!fits(need))return fail("fuse_cap_refusal");
            fuses.push_back({n,table,base});if(!bump(counters.fuse_builds) || !refresh())return false;
        } else if(!bump(counters.fuse_hits))return false;
        // A hit with evicted outer tables keeps them evicted. Runtime uses its
        // existing scratch to generate twiddles; it does not rebuild the cache.
        // Buffer lookup conservatively validates THREE buffers even for B/Q
        // reuse. Match its size_t bounds before simulating any buffer mutation.
        const Word allocation_max=std::numeric_limits<size_t>::max();
        if(n>allocation_max/24/slices || slots>allocation_max/8/slices-2)return fail("invalid_buffer_size");
        Word required=0,big_bytes=0;
        if(!multiply(n,slices,required) || !multiply(required,policy.pool && policy.reuse_bq?16:24,big_bytes))
            return fail("payload_overflow");
        const auto key=std::make_pair(n,slices);
        if(policy.pool) {
            if(workspace_words<required) {
                workspace_words=0;if(!refresh())return false; // free BEFORE growth
                if(!fits(big_bytes) && !evict(n,slices))return false;
                if(!fits(big_bytes))return fail("big_cap_refusal");
                workspace_words=required;
                if(!bump(counters.workspace_grows) || !refresh())return false;
            }
        } else if(!bigs.count(key)) {
            if(!fits(big_bytes) && !evict(n,slices))return false;
            if(!fits(big_bytes))return fail("big_cap_refusal");
            bigs[key]=big_bytes;if(!refresh())return false;
        }
        const auto old=smalls.find(key);
        Word need=0;
        if(old==smalls.end()) {
            if(!add(slots,2,need) || !multiply(need,slices,need) || !multiply(need,8,need))return fail("payload_overflow");
            if(!fits(need))return fail("digits_cap_refusal");
            smalls[key]=slots;
        } else if(old->second<slots) {
            // The real allocator checks incremental bytes before freeing the
            // old dOut; dRes remains live throughout replacement.
            if(!multiply(slots-old->second,slices,need) || !multiply(need,8,need))return fail("payload_overflow");
            if(!fits(need))return fail("digits_cap_refusal");
            old->second=slots;
        }
        if(!refresh())return false;
        // Buffer lookup uses the entire NTT batch, but the pass runner splits
        // gridDim.y at 65535. Carry scratch is requested per INNER pass batch.
        auto carry=[&](Word batch,Word times) {
            if(!policy.carry_check || !times || n*batch<(Word(1)<<20))return true;
            Word blocks=n/256+(n%256!=0),scratch=0;
            if(!multiply(blocks,batch,scratch) || !multiply(scratch,8,scratch))return fail("payload_overflow");
            if(scratch<=carry_bytes)return true;
            const Word growth=scratch-carry_bytes;
            if(!fits(growth))return bump(counters.carry_refusals,times);
            carry_bytes=scratch;return bump(counters.carry_grows) && refresh();
        };
        if(!carry(65535,slices/65535) || !carry(slices%65535,slices%65535?1:0))return false;
        return true;
    }
    // Boundary hook for a future whole-process lifetime plan. `available` is
    // a logical payload budget supplied by that plan, NEVER measured device free.
    bool cold_trim(Word keep_n,Word required,Word &available) {
        while(available<required) {
            auto chosen=fuses.end();Word largest=0;
            for(auto it=fuses.begin();it!=fuses.end();++it)if(it->n!=keep_n) {
                Word bytes=0;if(!add(it->table,it->base,bytes))return fail("payload_overflow");
                if(bytes>largest){largest=bytes;chosen=it;}
            }
            if(chosen==fuses.end())break;
            fuses.erase(chosen);
            if(!add(available,largest,available) || !bump(counters.cold_evictions) ||
               !bump(counters.cold_evicted_bytes,largest) || !refresh())return false;
        }
        return true;
    }
};
struct NttMemoryCheckpoint {
    Word repeat=1;
    NttMemoryPayload live;
    NttMemoryCounters counters;
    Word peak_bytes=0;
};
struct NttMemoryPlan {
    bool valid=false,finished=false;
    const char *reason="unsupported_request_program";
    NttMemoryPolicy policy;
    NttMemoryPayload final_payload;
    NttMemoryCounters counters;
    Word peak_bytes=0,executed_blocks=0,skipped_blocks=0;
    struct Stop {
        Word block=0,repeat_index=0,request_index=0,n=0,slots=0,slices=0;
        MultiplyRequest request;
    } stop;
    std::vector<NttMemoryCheckpoint> checkpoints;
};
// Compress identical chunk calls and steady repeated G-tree/fold blocks. Cap
// evictions retain bases, so even tight keyed caches eventually reach a fixed
// allocation state. Refusal stops at the exact successful prefix; the fallback
// allocation/transient footprint is deliberately NOT presented as complete.
template<class Query,class Describe> bool ntt_memory_plan(const RequestProgram &program,
        int bits,Query query,Describe describe,Word budget,bool physical,Word chunk_max,
        NttMemoryPolicy policy,NttMemoryPlan &plan) {
    plan=NttMemoryPlan{};plan.policy=policy;
    if(!program.supported || bits<2 || bits>max_input_bits)return true;
    NttMemoryState state(policy);
    auto save=[&]() {plan.final_payload=state.live;plan.counters=state.counters;
        plan.peak_bytes=state.peak;plan.reason=state.reason;};
    auto calls=[&](Word n,Word slots,Word slices,Word times) {
        while(times) {
            const auto before=state;
            if(!state.call(n,slots,slices,describe)){plan.stop.slices=slices;return false;}
            --times;
            if(times && state.same_allocations(before)) {
                const auto after=state.counters;
                if(!state.counters.accumulate(after,before.counters,times)) {state.reason="counter_overflow";return false;}
                break;
            }
        }
        return true;
    };
    for(size_t bi=0;bi<program.blocks.size();++bi) {
        const auto &block=program.blocks[bi];
        if(!block.repeat){plan.reason="zero_repeat";return false;}
        Word remaining=block.repeat;
        while(remaining) {
            const auto before=state;
            for(size_t ri=0;ri<block.requests.size();++ri) {
                const auto &r=block.requests[ri];
                Word n=0,slots=0;
                if(!query(std::max(r.ma,r.mb),bits,&n,&slots) || !n ||
                   slots!=2*std::max(r.ma,r.mb)-1 || !r.pairs) {plan.reason="invalid_shape";return false;}
                const Word buffers=policy.pool && policy.reuse_bq?2:3;
                Word c=chunk_slices(n,slots,r.pairs,budget,physical?(unsigned)buffers:3,physical);
                if(chunk_max)c=std::min(c,chunk_max);
                if(!calls(n,slots,c,r.pairs/c) || (r.pairs%c && !calls(n,slots,r.pairs%c,1))) {
                    plan.stop.block=bi;plan.stop.repeat_index=block.repeat-remaining;
                    plan.stop.request_index=ri;plan.stop.n=n;plan.stop.slots=slots;plan.stop.request=r;
                    save();plan.valid=!std::strcmp(state.reason,"fuse_cap_refusal") ||
                        !std::strcmp(state.reason,"big_cap_refusal") || !std::strcmp(state.reason,"digits_cap_refusal");
                    return plan.valid;
                }
            }
            if(!add(plan.executed_blocks,1,plan.executed_blocks)){plan.reason="counter_overflow";return false;}
            --remaining;
            if(remaining && state.same_allocations(before)) {
                const auto after=state.counters;
                if(!state.counters.accumulate(after,before.counters,remaining) ||
                   !add(plan.skipped_blocks,remaining,plan.skipped_blocks)){plan.reason="counter_overflow";return false;}
                break;
            }
        }
        plan.checkpoints.push_back({block.repeat,state.live,state.counters,state.peak});
    }
    save();plan.valid=plan.finished=true;return true;
}
} // namespace ecm_stage2
