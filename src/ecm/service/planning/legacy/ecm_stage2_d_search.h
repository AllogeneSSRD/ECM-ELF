#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <set>
#include <vector>
#include "../../../core/stage2/ecm_stage2_geometry.h"

namespace ecm_stage2 { namespace tune { namespace portable {
inline bool d_search_improves(double a,double sa,double b,double sb) {
    return a>0 && b>0 && a+2*sa < .98*b-2*sb;
}
using DSearchRegion=std::array<Word,8>;
struct DSearchNode {
    Word d=0,p=0;DSearchRegion region{};
    double rank=0,seconds=0,stddev=0;Word samples=0;
    bool rejected=false;
};
struct DSearchStep {
    size_t index=0;bool available=false;
    const char *reason="complete";double scale=0;
};
// A reconstructed, bounded search, shared by production trials and formal tune.
// No assumption of unimodality, no persistent per-curve search journal, and no
// authority to adopt a route: production adoption is a separate policy.
class DSearch {
    const std::vector<DSearchNode> &nodes;
    Word baseline;unsigned repeats,limit,regions;
    size_t missing()const{return nodes.size();}
    template<class Predicate> size_t best(Predicate accept)const {
        size_t out=missing();
        for(size_t i=0;i<nodes.size();++i) {
            const auto &n=nodes[i];
            if(n.rejected || !n.d || !n.p || !(n.rank>0) || !std::isfinite(n.rank) || !accept(n))continue;
            if(out==missing() || n.rank<nodes[out].rank || (n.rank==nodes[out].rank && n.d<nodes[out].d))out=i;
        }
        return out;
    }
    DSearchStep step(size_t i,const char *reason,double scale=0)const {
        return {i,i!=missing(),reason,scale};
    }
public:
    DSearch(const std::vector<DSearchNode> &pool,Word reference,unsigned observations=2,
            unsigned new_candidates=4,unsigned region_limit=3):nodes(pool),baseline(reference),
        repeats(observations),limit(new_candidates),regions(region_limit) {}
    size_t center()const {
        size_t out=missing();
        // Centre on the fastest repeated observation, not the model minimum.
        for(size_t i=0;i<nodes.size();++i) {
            const auto &n=nodes[i];if(n.rejected || n.samples<repeats || !(n.seconds>0))continue;
            if(out==missing() || n.seconds<nodes[out].seconds ||
                (n.seconds==nodes[out].seconds && n.d<nodes[out].d))out=i;
        }
        if(out==missing())for(size_t i=0;i<nodes.size();++i)if(nodes[i].d==baseline && !nodes[i].rejected)return i;
        return out==missing()?best([](const DSearchNode &){return true;}):out;
    }
    unsigned explored()const {
        unsigned out=0;for(const auto &n:nodes)out+=n.d!=baseline && n.samples!=0;return out;
    }
    DSearchStep next()const {
        for(size_t i=0;i<nodes.size();++i)if(nodes[i].d==baseline && !nodes[i].rejected && nodes[i].samples<repeats)
            return step(i,"reference");
        // Finish one independent repeat before exposing another unmeasured D.
        const auto partial=best([&](const DSearchNode &n){return n.samples && n.samples<repeats;});
        if(partial!=missing())return step(partial,"repeat");
        const unsigned used=explored();if(used>=limit)return step(missing(),"candidate_budget");
        const auto c=center();if(c==missing())return step(missing(),"no_reference");
        std::set<DSearchRegion> seen;
        for(const auto &n:nodes)if(!n.rejected && n.samples>=repeats)seen.insert(n.region);
        auto new_region_allowed=[&](const DSearchNode &n){return seen.count(n.region) || seen.size()<regions;};
        // An improving edge supplies a measured direction. Extend its last
        // meaningful P step before considering a new model seed or backfill.
        // A change in packing/NTT region is admissible here, never a hard wall.
        // Reconstruct from the measured bracket; no timestamps/journal needed.
        size_t left=missing(),right=missing();bool d_left=false,d_right=false;
        for(size_t i=0;i<nodes.size();++i) {
            const auto &n=nodes[i];
            if(n.rejected || n.samples<repeats || !(n.seconds>0) || !n.p)continue;
            // Use the best prior contender, not a closer but slower backfill:
            // filling a 2% P gap must not erase the last successful 12% step.
            if(n.p<nodes[c].p && (left==missing() || n.seconds<nodes[left].seconds ||
                (n.seconds==nodes[left].seconds && n.p>nodes[left].p)))left=i;
            if(n.p>nodes[c].p && (right==missing() || n.seconds<nodes[right].seconds ||
                (n.seconds==nodes[right].seconds && n.p<nodes[right].p)))right=i;
            d_left|=n.d<nodes[c].d;d_right|=n.d>nodes[c].d;
        }
        const int direction=left!=missing() && right==missing()?1:
                            right!=missing() && left==missing()?-1:0;
        const auto previous=direction>0?left:right;
        const int d_direction=d_left && !d_right?1:d_right && !d_left?-1:0;
        if(direction && nodes[c].seconds<nodes[previous].seconds) {
            const double distance=std::clamp(std::abs(std::log(double(nodes[c].p)/nodes[previous].p)),
                                            std::log(1.25)/8,std::log(1.25));
            for(unsigned level=0;level<4;++level) {
                const double span=distance/double(1u<<level);
                const auto i=best([&](const DSearchNode &n) {
                    const double delta=std::log(double(n.p)/nodes[c].p)*direction;
                    // P is not monotonic in D. When both have an observed edge,
                    // do not call a reversal of D a forward P continuation.
                    if((d_direction>0 && n.d<=nodes[c].d) || (d_direction<0 && n.d>=nodes[c].d))return false;
                    // Old losing regions must not exhaust all forward options.
                    // Permit one extra region for a measured edge; candidate
                    // count and engine-time caps still bound the whole search.
                    return (seen.count(n.region) || seen.size()<regions+1) &&
                        delta>=span*.65 && delta<=span*1.35;
                });
                if(i!=missing() && !nodes[i].samples)
                    return step(i,"improving_edge",std::exp(direction*span));
            }
        }
        const auto seed=best([&](const DSearchNode &n){return new_region_allowed(n);});
        if(seed!=missing() && !nodes[seed].samples)return step(seed,"model_seed");
        auto jump=[&]() {
            const auto i=best([&](const DSearchNode &n){return !seen.count(n.region) && seen.size()<regions;});
            return i!=missing() && !nodes[i].samples?i:missing();
        };
        // Reserve the last candidate for an unvisited execution region.
        if(used+1>=limit) {const auto i=jump();if(i!=missing())return step(i,"region_restart");}
        for(unsigned level=0;level<4;++level) {
            const double distance=std::log(1.25)/double(1u<<level);
            size_t selected=missing();int selected_direction=0;
            for(int sign:{-1,1}) {
                const auto i=best([&](const DSearchNode &n) {
                    if(n.region!=nodes[c].region || n.d==nodes[c].d)return false;
                    const double delta=std::log(double(n.p)/nodes[c].p)*sign;
                    return delta>=distance*.65 && delta<=distance*1.35;
                });
                // Pick the representative before checking its sample count:
                // otherwise every restart would crawl through the same band.
                if(i!=missing() && !nodes[i].samples && (selected==missing() ||
                    nodes[i].rank<nodes[selected].rank ||
                    (nodes[i].rank==nodes[selected].rank && nodes[i].d<nodes[selected].d))) {
                    selected=i;selected_direction=sign;
                }
            }
            if(selected!=missing())return step(selected,level?"refine":"scale",std::exp(selected_direction*distance));
        }
        if(const auto i=jump();i!=missing())return step(i,"region_restart");
        const auto far_candidate=best([&](const DSearchNode &n){return n.region==nodes[c].region &&
            std::abs(std::log(double(n.p)/nodes[c].p))>2*std::log(1.25);});
        if(far_candidate!=missing() && !nodes[far_candidate].samples)return step(far_candidate,"distant_restart");
        return step(missing(),"neighborhood_complete");
    }
};
} } }
