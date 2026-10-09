#pragma once
#include "ecm_stage2_requests.h"
#include <cstring>
#include <functional>
#include <set>
#include <type_traits>

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
// Sites describe individual successful allocations. The legacy total-only
// descriptor uses explicitly marked groups; never count those as cudaMallocs.
enum NttMemorySite {
    NttBaseTileF,NttBaseTileI,NttBaseScratch,NttBaseRadix,
    NttTablePassF,NttTableRadF,NttTablePassI,NttTableRadI,
    NttWorkspaceA,NttWorkspaceB,NttWorkspaceQ,
    NttKeyedA,NttKeyedB,NttKeyedQ,NttDigitsOutput,NttDigitsResult,NttCarry,
    NttBaseGroup,NttTableGroup
};
struct NttFuseAllocation {
    NttMemorySite site=NttBaseTileF;
    unsigned index=0;
    Word bytes=0;
    bool operator==(const NttFuseAllocation &o)const {
        return site==o.site && index==o.index && bytes==o.bytes;
    }
};
struct NttFuseMemoryLayout {
    std::vector<NttFuseAllocation> base,table; // allocation order, not release order
    bool grouped=false;
    bool operator==(const NttFuseMemoryLayout &o)const {
        return base==o.base && table==o.table && grouped==o.grouped;
    }
    bool totals(Word &base_bytes,Word &table_bytes)const {
        base_bytes=table_bytes=0;std::set<std::pair<unsigned,unsigned>> seen;
        for(unsigned kind=0;kind<2;++kind)for(const auto &a:kind?table:base) {
            const bool site_ok=kind?(a.site>=NttTablePassF && a.site<=NttTableRadI):
                (a.site>=NttBaseTileF && a.site<=NttBaseRadix);
            const bool group_ok=grouped && a.site==(kind?NttTableGroup:NttBaseGroup);
            Word &total=kind?table_bytes:base_bytes;
            if(!a.bytes || (!site_ok && !group_ok) ||
               !seen.emplace(unsigned(a.site),a.index).second || !add(total,a.bytes,total))return false;
        }
        return base_bytes!=0;
    }
};
// Accept the allocation-free native FuseCtx description (or a CPU fixture with
// the same fields). Device/policy selection stays in fuse_describe; no guessed
// default transform plan is embedded here.
template<class Descriptor> bool ntt_fuse_memory_layout(const Descriptor &c,NttFuseMemoryLayout &out) {
    out=NttFuseMemoryLayout{};
    if(!c.n || (c.n&(c.n-1)) || c.k<0 || c.k>=64 || c.n!=(Word(1)<<c.k) ||
       c.t<0 || c.t>c.k || c.nms<0 || c.nms>8 || c.outer_stages!=c.k-c.t)return false;
    auto push=[&](std::vector<NttFuseAllocation> &v,NttMemorySite site,unsigned p,Word words) {
        Word bytes=0;if(!multiply(words,8,bytes))return false;
        if(bytes)v.push_back({site,p,bytes});return true;
    };
    if(!push(out.base,NttBaseTileF,0,Word(1)<<c.t) || !push(out.base,NttBaseTileI,0,Word(1)<<c.t) ||
       !push(out.base,NttBaseScratch,0,c.scrWords) || !push(out.base,NttBaseRadix,0,c.scr2Words))return false;
    int L=0;
    for(int p=0;p<c.nms;++p) {
        const int m=c.ms[p];if(m<=0 || m>=31 || m>c.outer_stages-L)return false;
        if(!push(out.table,NttTablePassF,p,c.n>>(L+m)) ||
           !push(out.table,NttTableRadF,p,Word(1)<<m))return false;L+=m;
    }
    if(L!=c.outer_stages)return false;
    for(int p=c.nms-1;p>=0;--p) {
        const int m=c.ms[p];L-=m;const int s0=c.k-L-m;
        if(s0<0 || s0>=64 || !push(out.table,NttTablePassI,p,Word(1)<<s0) ||
           !push(out.table,NttTableRadI,p,Word(1)<<m))return false;
    }
    Word base=0,table=0;return out.totals(base,table);
}
struct NttMemoryEvent {
    NttMemorySite site;
    unsigned index=0;
    Word n=0,slices=0,bytes=0;
    bool allocation=false,grouped=false;
    NttMemoryPayload live;
};
struct NttMemoryCounters {
    Word calls=0,fuse_builds=0,fuse_hits=0,workspace_grows=0;
    Word cap_evictions=0,cap_evicted_bytes=0,cold_evictions=0,cold_evicted_bytes=0;
    Word carry_grows=0,carry_refusals=0;
    Word allocations=0,frees=0,grouped_events=0;
    bool accumulate(const NttMemoryCounters &after,const NttMemoryCounters &before,Word times) {
        Word *dst[]={&calls,&fuse_builds,&fuse_hits,&workspace_grows,&cap_evictions,
            &cap_evicted_bytes,&cold_evictions,&cold_evicted_bytes,&carry_grows,&carry_refusals,
            &allocations,&frees,&grouped_events};
        const Word a[]={after.calls,after.fuse_builds,after.fuse_hits,after.workspace_grows,
            after.cap_evictions,after.cap_evicted_bytes,after.cold_evictions,after.cold_evicted_bytes,
            after.carry_grows,after.carry_refusals,after.allocations,after.frees,after.grouped_events};
        const Word b[]={before.calls,before.fuse_builds,before.fuse_hits,before.workspace_grows,
            before.cap_evictions,before.cap_evicted_bytes,before.cold_evictions,before.cold_evicted_bytes,
            before.carry_grows,before.carry_refusals,before.allocations,before.frees,before.grouped_events};
        for(unsigned i=0;i<13;++i) {Word delta=0;
            if(a[i]<b[i] || !multiply(a[i]-b[i],times,delta) || !add(*dst[i],delta,*dst[i]))return false;
        }
        return true;
    }
};
class NttMemoryState {
    struct Fuse {
        Word n=0,table=0,base=0;
        NttFuseMemoryLayout layout;
        bool operator==(const Fuse &o)const{return n==o.n && table==o.table && base==o.base && layout==o.layout;}
    };
    // Fuse order matters: cold trim chooses the first largest context on ties.
    std::vector<Fuse> fuses;
    std::map<std::pair<Word,Word>,Word> bigs,smalls; // bytes and out_cap respectively
    using Key=std::pair<Word,Word>;
    std::vector<Key> big_order,small_order; // native vectors evict in reverse insertion order
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
    bool event(NttMemorySite site,Word n,Word slices,unsigned index,Word bytes,bool allocation,bool grouped=false) {
        if(!bytes)return true;
        Word &part=(site<=NttBaseRadix || site==NttBaseGroup)?live.base:
            (site<=NttTableRadI || site==NttTableGroup)?live.table:
            site<=NttKeyedQ?live.big:live.digits;
        if(allocation) {
            if(!add(part,bytes,part) || !add(live.total,bytes,live.total))return fail("payload_overflow");
        } else {
            if(bytes>part || bytes>live.total)return fail("event_underflow");
            part-=bytes;live.total-=bytes;
        }
        peak=std::max(peak,live.total);
        if(!bump(allocation?counters.allocations:counters.frees) || (grouped && !bump(counters.grouped_events)))return false;
        if(observe)observe({site,index,n,slices,bytes,allocation,grouped,live});
        return true;
    }
    bool big_events(Word n,Word slices,Word bytes,bool pool,bool allocation) {
        const unsigned count=pool && policy.reuse_bq?2:3;
        if(bytes%count)return fail("invalid_big_payload");
        for(unsigned i=0;i<count;++i)
            if(!event(NttMemorySite((pool?NttWorkspaceA:NttKeyedA)+i),n,slices,0,bytes/count,allocation))return false;
        return true;
    }
    bool fuse_release_events(const Fuse &f,bool base) {
        auto table=f.layout.table;
        std::sort(table.begin(),table.end(),[](const NttFuseAllocation &a,const NttFuseAllocation &b) {
            return std::make_pair(a.index,a.site)<std::make_pair(b.index,b.site);
        });
        if(f.table)for(const auto &a:table)
            if(!event(a.site,f.n,0,a.index,a.bytes,false,f.layout.grouped))return false;
        if(base)for(const auto &a:f.layout.base)
            if(!event(a.site,f.n,0,a.index,a.bytes,false,f.layout.grouped))return false;
        return true;
    }
    bool evict(Word n,Word slices) {
        // Runtime always drops carry scratch, but excludes it from tbl_words_freed.
        if(!event(NttCarry,0,0,0,carry_bytes,false))return false;
        carry_bytes=0;Word freed=0;
        for(auto &f:fuses)if(f.n!=n) {
            if(!add(freed,f.table,freed) || !fuse_release_events(f,false))return fail("payload_overflow");f.table=0;
        }
        const auto keep=std::make_pair(n,slices);
        for(size_t i=big_order.size();i-- >0;) {
            const auto key=big_order[i];if(key==keep)continue;
            const Word bytes=bigs.at(key);
            if(!add(freed,bytes,freed) || !big_events(key.first,key.second,bytes,false,false))return fail("payload_overflow");
            bigs.erase(key);big_order.erase(big_order.begin()+i);
        }
        for(size_t i=small_order.size();i-- >0;) {
            const auto key=small_order[i];if(key==keep)continue;
            Word bytes=0;
            const Word slots=smalls.at(key);
            if(!add(slots,2,bytes) || !multiply(bytes,key.second,bytes) ||
               !multiply(bytes,8,bytes) || !add(freed,bytes,freed))return fail("payload_overflow");
            if(!event(NttDigitsOutput,key.first,key.second,0,slots*key.second*8,false) ||
               !event(NttDigitsResult,key.first,key.second,0,key.second*16,false))return false;
            smalls.erase(key);small_order.erase(small_order.begin()+i);
        }
        if(freed && (!bump(counters.cap_evictions) || !bump(counters.cap_evicted_bytes,freed)))return false;
        return refresh();
    }
public:
    std::function<void(const NttMemoryEvent &)> observe;
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
            bigs==o.bigs && smalls==o.smalls && fuses==o.fuses &&
            big_order==o.big_order && small_order==o.small_order;
    }
    Word shared_workspace_bytes()const{return workspace_words*(policy.reuse_bq?16:24);}
    template<class Describe> bool call(Word n,Word slots,Word slices,Describe describe) {
        return call_layout(n,slots,slices,[&](Word size,NttFuseMemoryLayout &layout) {
            Word table=0,base=0;if(!describe(size,table,base))return false;
            layout.grouped=true;
            if(base)layout.base.push_back({NttBaseGroup,0,base});
            if(table)layout.table.push_back({NttTableGroup,0,table});
            return true;
        });
    }
    template<class Describe> bool call_layout(Word n,Word slots,Word slices,Describe describe) {
        if(!n || !slices || (n&(n-1)))return fail("invalid_shape");
        if(!bump(counters.calls))return false;
        // Shape planning allocates fuse tables/base BEFORE buffer lookup. The
        // cap check here does not evict; refusal invokes a per-call context.
        auto found=std::find_if(fuses.begin(),fuses.end(),[&](const Fuse &f){return f.n==n;});
        if(found==fuses.end()) {
            Word table=0,base=0,need=0;
            NttFuseMemoryLayout layout;
            if(!describe(n,layout) || !layout.totals(base,table) || !add(table,base,need))return fail("descriptor_overflow");
            if(!fits(need))return fail("fuse_cap_refusal");
            for(const auto &a:layout.base)if(!event(a.site,n,0,a.index,a.bytes,true,layout.grouped))return false;
            for(const auto &a:layout.table)if(!event(a.site,n,0,a.index,a.bytes,true,layout.grouped))return false;
            fuses.push_back({n,table,base,layout});if(!bump(counters.fuse_builds) || !refresh())return false;
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
                if(!big_events(0,0,shared_workspace_bytes(),true,false))return false;
                workspace_words=0;if(!refresh())return false; // free BEFORE growth
                if(!fits(big_bytes) && !evict(n,slices))return false;
                if(!fits(big_bytes))return fail("big_cap_refusal");
                workspace_words=required;
                if(!big_events(n,slices,big_bytes,true,true))return false;
                if(!bump(counters.workspace_grows) || !refresh())return false;
            }
        } else if(!bigs.count(key)) {
            if(!fits(big_bytes) && !evict(n,slices))return false;
            if(!fits(big_bytes))return fail("big_cap_refusal");
            if(!big_events(n,slices,big_bytes,false,true))return false;
            bigs[key]=big_bytes;big_order.push_back(key);if(!refresh())return false;
        }
        const auto old=smalls.find(key);
        Word need=0;
        if(old==smalls.end()) {
            if(!add(slots,2,need) || !multiply(need,slices,need) || !multiply(need,8,need))return fail("payload_overflow");
            if(!fits(need))return fail("digits_cap_refusal");
            if(!event(NttDigitsOutput,n,slices,0,slots*slices*8,true) ||
               !event(NttDigitsResult,n,slices,0,slices*16,true))return false;
            smalls[key]=slots;small_order.push_back(key);
        } else if(old->second<slots) {
            // The real allocator checks incremental bytes before freeing the
            // old dOut; dRes remains live throughout replacement.
            if(!multiply(slots-old->second,slices,need) || !multiply(need,8,need))return fail("payload_overflow");
            if(!fits(need))return fail("digits_cap_refusal");
            if(!event(NttDigitsOutput,n,slices,0,old->second*slices*8,false) ||
               !event(NttDigitsOutput,n,slices,0,slots*slices*8,true))return false;
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
            if(!event(NttCarry,0,0,0,carry_bytes,false) ||
               !event(NttCarry,n,batch,0,scratch,true))return false;
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
            if(!fuse_release_events(*chosen,true))return false;
            fuses.erase(chosen);
            if(!add(available,largest,available) || !bump(counters.cold_evictions) ||
               !bump(counters.cold_evicted_bytes,largest) || !refresh())return false;
        }
        return true;
    }
    // Explicit close mirrors NttArena::release. Copies used for compression
    // have no destructor side effects and never free another state's payload.
    bool close() {
        if(!event(NttCarry,0,0,0,carry_bytes,false))return false;carry_bytes=0;
        if(!big_events(0,0,shared_workspace_bytes(),true,false))return false;workspace_words=0;
        for(const auto &f:fuses)if(!fuse_release_events(f,true))return false;
        fuses.clear();
        for(const auto &key:big_order)if(!big_events(key.first,key.second,bigs.at(key),false,false))return false;
        bigs.clear();big_order.clear();
        for(const auto &key:small_order) {
            if(!event(NttDigitsOutput,key.first,key.second,0,smalls.at(key)*key.second*8,false) ||
               !event(NttDigitsResult,key.first,key.second,0,key.second*16,false))return false;
        }
        smalls.clear();small_order.clear();return refresh();
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
    bool exact_allocation_events=false;
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
    std::map<Word,NttFuseMemoryLayout> fuse_layouts;
};
// Compress identical chunk calls and steady repeated G-tree/fold blocks. Cap
// evictions retain bases, so even tight keyed caches eventually reach a fixed
// allocation state. Refusal stops at the exact successful prefix; the fallback
// allocation/transient footprint is deliberately NOT presented as complete.
template<class Query,class Describe> bool ntt_memory_plan(const RequestProgram &program,
        int bits,Query query,Describe describe,Word budget,bool physical,Word chunk_max,
        NttMemoryPolicy policy,NttMemoryPlan &plan) {
    plan=NttMemoryPlan{};plan.policy=policy;
    plan.exact_allocation_events=std::is_invocable_r_v<bool,Describe,Word,NttFuseMemoryLayout &>;
    if(!program.supported || bits<2 || bits>max_input_bits)return true;
    NttMemoryState state(policy);
    auto save=[&]() {plan.final_payload=state.live;plan.counters=state.counters;
        plan.peak_bytes=state.peak;plan.reason=state.reason;};
    auto calls=[&](Word n,Word slots,Word slices,Word times) {
        while(times) {
            const auto before=state;
            bool success=false;
            if constexpr(std::is_invocable_r_v<bool,Describe,Word,NttFuseMemoryLayout &>) {
                success=state.call_layout(n,slots,slices,[&](Word size,NttFuseMemoryLayout &layout) {
                    if(!describe(size,layout) || layout.grouped)return false;
                    plan.fuse_layouts.emplace(size,layout);return true;
                });
            } else success=state.call(n,slots,slices,describe);
            if(!success){plan.stop.slices=slices;return false;}
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
