#pragma once
#include "ecm_stage2_geometry.h"
#include <vector>

// Giant component lifetimes only. No NTT, S4 raw/output, fold owner, physical
// free-memory admission, driver overhead or optional diagnostic workspaces.
namespace ecm_stage2 {
inline Word ceil_ratio(Word n,Word d) {return n/d+(n%d!=0);}
inline bool giant_workspace_bytes(Word w,Word points,Word values,Word segfix,
                                  bool base,Word &bytes) {
    Word a=0,b=0,c=0,total=0;
    return w && multiply(5,w,total) && multiply(points,w,a) && multiply(a,2,a) &&
        add(a,points,a) && add(total,a,total) && multiply(values,w,b) &&
        multiply(b,2,b) && add(total,b,total) &&
        (!segfix || (add(segfix,1,c) && multiply(c,w,c) && add(total,c,total))) &&
        (!base || (multiply(w,2,c) && add(total,c,total))) && multiply(total,8,bytes);
}
struct GiantMemoryPolicy {
    Word chain_block=64,short_block=0,short_max=8192,chain_min=32768;
    Word segment=16,group=64,resident_limit_bytes=512ull<<20,initial_points=0;
    bool force_ladder=false,seed_device=true,seed_pair=true;
    bool exact_segments=true,resident_requested=true,resident_eligible=true;
};
struct GiantMemoryChunk {
    Word repeat=1,points=0,block=0,seeds=0,segments=0,groups=0;
    Word point_capacity=0,workspace_bytes=0,coordinate_bytes=0,segment_bytes=0;
    Word group_bytes=0,legacy_seed_bytes=0,prepare_bytes=0,tree_bytes=0;
    bool chain=false,resident=false;
};
struct GiantMemoryPlan {
    bool valid=false;
    const char *reason="invalid_geometry";
    GiantMemoryPolicy policy;
    Word p=0,giant_points=0,words=0,chunk_points=0,point_chunks=0;
    Word initial_bytes=0,after_giant_bytes=0,accumulation_bytes=0,peak_bytes=0;
    Word final_point_capacity=0;
    std::vector<GiantMemoryChunk> chunks;
};
// All full chunks have the same retained capacities. Only one full and one
// tail transition are needed even for enormous B2. The tail may use the ladder
// and GROW the seed workspace; borrowed ladder X/Z are never counted twice.
inline bool giant_memory_plan(Word p,Word points,Word w,Word chunk_points,
                              GiantMemoryPolicy policy,GiantMemoryPlan &plan) {
    plan=GiantMemoryPlan{};plan.policy=policy;plan.p=p;plan.giant_points=points;
    plan.words=w;plan.chunk_points=chunk_points;
    if(!p || !points || !w || w>max_words || !chunk_points || chunk_points%p ||
       !policy.chain_block || !policy.segment || !policy.group)return false;
    Word capacity=policy.initial_points,segfix=0;bool base=false;
    if(!giant_workspace_bytes(w,capacity,0,0,false,plan.initial_bytes))return false;
    plan.peak_bytes=plan.initial_bytes;
    auto chunk=[&](Word n,Word repeat) {
        GiantMemoryChunk c;c.points=n;c.repeat=repeat;
        c.chain=!policy.force_ladder && n>=policy.chain_min;
        c.block=c.chain?giant_chain_block(n,policy.chain_block,policy.short_block,policy.short_max):0;
        Word chains=0,v=0;
        if(c.chain) {
            chains=ceil_ratio(n,c.block);
            if(!multiply(chains,2,c.seeds) || !add(c.seeds,1,c.seeds))return false;
            base=base || (policy.seed_device && policy.seed_pair);
        } else c.seeds=n;
        capacity=std::max(capacity,c.seeds);c.point_capacity=capacity;
        if(!multiply(n,w,v) || !multiply(v,16,c.coordinate_bytes))return false;
        c.resident=policy.resident_requested && policy.resident_eligible &&
            policy.exact_segments && c.coordinate_bytes<=policy.resident_limit_bytes;
        c.segments=ceil_ratio(n,policy.segment);
        if(c.chain || c.resident) {
            if(!multiply(c.segments,w,v) || !multiply(v,8,c.segment_bytes))return false;
            if(policy.exact_segments)segfix=std::max(segfix,policy.segment);
        }
        if(c.resident) {
            c.groups=ceil_ratio(c.segments,policy.group);
            if(!add(c.groups,policy.group,v) || !add(v,1,v) ||
               !multiply(v,w,v) || !multiply(v,8,c.group_bytes))return false;
        }
        if(c.chain && !policy.seed_device) {
            if(!multiply(chains,4,v) || !add(v,2,v) ||
               !multiply(v,w,v) || !multiply(v,8,c.legacy_seed_bytes))return false;
        }
        if(!giant_workspace_bytes(w,capacity,0,segfix,base,c.workspace_bytes))return false;
        // During chain production legacy seeds overlap X/Z and segment outputs;
        // they are released BEFORE resident inversion groups are allocated.
        c.prepare_bytes=c.workspace_bytes;
        if(c.chain && !add(c.prepare_bytes,c.coordinate_bytes,c.prepare_bytes))return false;
        if(!add(c.prepare_bytes,c.segment_bytes,c.prepare_bytes) ||
           !add(c.prepare_bytes,c.legacy_seed_bytes,c.prepare_bytes))return false;
        c.tree_bytes=c.workspace_bytes;
        if(c.resident && ((c.chain && !add(c.tree_bytes,c.coordinate_bytes,c.tree_bytes)) ||
           !add(c.tree_bytes,c.segment_bytes,c.tree_bytes) ||
           !add(c.tree_bytes,c.group_bytes,c.tree_bytes)))return false;
        plan.peak_bytes=std::max(plan.peak_bytes,std::max(c.prepare_bytes,c.tree_bytes));
        plan.chunks.push_back(c);return true;
    };
    const Word full=points/chunk_points,tail=points%chunk_points;
    if((full && !chunk(chunk_points,full)) || (tail && !chunk(tail,1))) {
        plan.reason="payload_overflow";return false;
    }
    plan.point_chunks=full+(tail!=0);plan.final_point_capacity=capacity;
    if(!giant_workspace_bytes(w,capacity,0,segfix,base,plan.after_giant_bytes) ||
       !giant_workspace_bytes(w,capacity,p,segfix,base,plan.accumulation_bytes)) {
        plan.reason="payload_overflow";return false;
    }
    plan.peak_bytes=std::max(plan.peak_bytes,plan.accumulation_bytes);
    plan.valid=true;plan.reason="ok";return true;
}
} // namespace ecm_stage2
