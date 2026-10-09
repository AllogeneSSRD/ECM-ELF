#pragma once
#include "ecm_stage2_giant_memory.h"
#include <functional>

namespace ecm_stage2 {
enum GiantMemorySite {
    GiantN,GiantQx,GiantQz,GiantA24,GiantOne,GiantIndices,GiantSeedX,GiantSeedZ,
    GiantBase,GiantSegfix,GiantValues,GiantProducts,
    GiantLegacyDx,GiantLegacyDz,GiantLegacyEx,GiantLegacyEz,GiantLegacyDiffX,GiantLegacyDiffZ,
    GiantCoordinatesX,GiantCoordinatesZ,GiantSegments,GiantGroups,GiantGroupFix,GiantSiteCount
};
struct GiantMemoryEvent {GiantMemorySite site;bool allocation;Word bytes,total;};
struct GiantTimelinePolicy {Word p=0,points=0,words=0,chunk_points=0;GiantMemoryPolicy component;};
// Successful allocations only. A shared ladder coordinate is the seed buffer,
// not a second allocation. Chunk coordinates/segments/groups die after the last
// G tree using that point chunk. Counters exclude allocator failure/fallback.
class GiantMemoryState {
    Word slots[GiantSiteCount]={},point_capacity=0,segfix_capacity=0,value_capacity=0,product_capacity=0;
    bool refresh(GiantMemorySite site,bool allocation,Word bytes) {
        if(allocation){if(!add(live,bytes,live)){reason="payload_overflow";return false;}}
        else {if(live<bytes){reason="invalid_release";return false;}live-=bytes;}
        peak=std::max(peak,live);if(observe)observe({site,allocation,bytes,live});return true;
    }
    bool release(GiantMemorySite site) {
        const Word bytes=slots[site];if(!bytes)return true;
        if(!refresh(site,false,bytes))return false;slots[site]=0;return true;
    }
    bool allocate(GiantMemorySite site,Word bytes) {
        if(!bytes || slots[site]){reason="invalid_allocation";return false;}
        if(!refresh(site,true,bytes))return false;slots[site]=bytes;return true;
    }
    bool words(GiantMemorySite site,Word count) {
        Word bytes=0;if(!multiply(count,8,bytes)){reason="payload_overflow";return false;}
        return allocate(site,bytes);
    }
    bool points_reserve(Word n) {
        if(n<=point_capacity)return true;Word coordinates=0;
        if(!multiply(n,policy.words,coordinates)){reason="payload_overflow";return false;}
        // S3Workspace::need_pts releases all three old arrays first.
        if(!release(GiantIndices) || !release(GiantSeedX) || !release(GiantSeedZ) ||
           !words(GiantIndices,n) || !words(GiantSeedX,coordinates) || !words(GiantSeedZ,coordinates))return false;
        point_capacity=n;return true;
    }
    bool segfix_reserve(Word n) {
        if(n<=segfix_capacity)return true;Word count=0;
        if(!add(n,1,count) || !multiply(count,policy.words,count)){reason="payload_overflow";return false;}
        if(!release(GiantSegfix) || !words(GiantSegfix,count))return false;segfix_capacity=n;return true;
    }
public:
    GiantTimelinePolicy policy;
    Word live=0,peak=0,consumed_points=0,chunk_remaining=0,chunks=0;
    bool started=false;
    const char *reason="ok";
    std::function<void(const GiantMemoryEvent &)> observe;
    explicit GiantMemoryState(GiantTimelinePolicy p={}):policy(p){}
    bool init() {
        const auto &c=policy.component;
        if(started || !policy.p || !policy.points || !policy.words || policy.words>max_words ||
           !policy.chunk_points || policy.chunk_points%policy.p || !c.chain_block ||
           !c.segment || !c.group || !c.accumulation_block) {reason="invalid_giant_policy";return false;}
        for(auto site:{GiantN,GiantQx,GiantQz,GiantA24,GiantOne})if(!words(site,policy.words))return false;
        if(!points_reserve(c.initial_points))return false;started=true;return true;
    }
    bool chunk_begin() {
        if(!started || chunk_remaining || consumed_points>=policy.points){reason="invalid_chunk_boundary";return false;}
        const auto &c=policy.component;
        const Word n=std::min(policy.chunk_points,policy.points-consumed_points);
        const bool chain=!c.force_ladder && n>=c.chain_min;
        const Word block=chain?giant_chain_block(n,c.chain_block,c.short_block,c.short_max):0;
        const Word chains=chain?ceil_ratio(n,block):0;
        Word seeds=n,coord=0,coord_bytes=0,segments=ceil_ratio(n,c.segment),seg_words=0;
        if((chain && (!multiply(chains,2,seeds) || !add(seeds,1,seeds))) ||
           !multiply(n,policy.words,coord) || !multiply(coord,16,coord_bytes) ||
           !multiply(segments,policy.words,seg_words)) {reason="payload_overflow";return false;}
        const bool resident=c.resident_requested && c.resident_eligible && c.exact_segments &&
            coord_bytes<=c.resident_limit_bytes;
        if(!points_reserve(seeds))return false;
        if(chain && c.seed_device && c.seed_pair && !slots[GiantBase]) {
            Word count=0;if(!multiply(policy.words,2,count) || !words(GiantBase,count))return false;
        }
        if(chain && !c.seed_device) {
            Word count=0;if(!multiply(chains,policy.words,count)){reason="payload_overflow";return false;}
            for(auto site:{GiantLegacyDx,GiantLegacyDz,GiantLegacyEx,GiantLegacyEz})if(!words(site,count))return false;
            if(!words(GiantLegacyDiffX,policy.words) || !words(GiantLegacyDiffZ,policy.words))return false;
        }
        if(chain && (!words(GiantCoordinatesX,coord) || !words(GiantCoordinatesZ,coord)))return false;
        if(chain || resident) {
            if(!words(GiantSegments,seg_words) || (c.exact_segments && !segfix_reserve(c.segment)))return false;
            if(!resident && !release(GiantSegments))return false;
        }
        for(auto site:{GiantLegacyDx,GiantLegacyDz,GiantLegacyEx,GiantLegacyEz,GiantLegacyDiffX,GiantLegacyDiffZ})
            if(!release(site))return false;
        if(chain && !resident && (!release(GiantCoordinatesX) || !release(GiantCoordinatesZ)))return false;
        if(resident) {
            Word group_words=0,fix_words=0;
            if(!multiply(ceil_ratio(segments,c.group),policy.words,group_words) ||
               !add(c.group,1,fix_words) || !multiply(fix_words,policy.words,fix_words) ||
               !words(GiantGroups,group_words) || !words(GiantGroupFix,fix_words))return false;
        }
        chunk_remaining=n;++chunks;return true;
    }
    bool tree_done(Word leaves) {
        if(!leaves || leaves>chunk_remaining || leaves>policy.points-consumed_points) {
            reason="invalid_giant_tree_range";return false;
        }
        chunk_remaining-=leaves;consumed_points+=leaves;
        if(!chunk_remaining)for(auto site:{GiantCoordinatesX,GiantCoordinatesZ,GiantSegments,GiantGroups,GiantGroupFix})
            if(!release(site))return false;
        return true;
    }
    bool same_allocations(const GiantMemoryState &o)const {
        if(started!=o.started || chunk_remaining!=o.chunk_remaining || point_capacity!=o.point_capacity ||
           segfix_capacity!=o.segfix_capacity || value_capacity!=o.value_capacity || product_capacity!=o.product_capacity)return false;
        for(unsigned i=0;i<GiantSiteCount;++i)if(slots[i]!=o.slots[i])return false;
        return true; // Absolute progress is checked separately before skipping.
    }
    bool skip_full_chunks(Word count) {
        Word points=0;
        if(chunk_remaining || !multiply(count,policy.chunk_points,points) ||
           points>policy.points-consumed_points || !add(chunks,count,chunks)) {
            reason="invalid_giant_skip";return false;
        }
        consumed_points+=points;return true;
    }
    bool accumulate() {
        if(!started || chunk_remaining || consumed_points!=policy.points){reason="incomplete_giant_range";return false;}
        const auto &c=policy.component;const Word n=policy.p;
        if(n>value_capacity) {
            if(!release(GiantValues) || (!c.compact_products && !release(GiantProducts)))return false;
            Word count=0;if(!multiply(n,policy.words,count) || !words(GiantValues,count))return false;
            if(!c.compact_products && !words(GiantProducts,count))return false;
            value_capacity=n;if(!c.compact_products)product_capacity=n;
        }
        const Word products=c.compact_products?ceil_ratio(n,c.accumulation_block):n;
        if(products>product_capacity) {
            Word count=0;if(!multiply(products,policy.words,count) || !release(GiantProducts) ||
                !words(GiantProducts,count))return false;product_capacity=products;
        }
        return true;
    }
    bool close() {
        for(auto site:{GiantCoordinatesX,GiantCoordinatesZ,GiantSegments,GiantGroups,GiantGroupFix})if(!release(site))return false;
        for(auto site:{GiantN,GiantQx,GiantQz,GiantA24,GiantOne,GiantIndices,GiantSeedX,GiantSeedZ,
            GiantBase,GiantValues,GiantProducts,GiantSegfix})if(!release(site))return false;
        return true;
    }
};
} // namespace ecm_stage2
