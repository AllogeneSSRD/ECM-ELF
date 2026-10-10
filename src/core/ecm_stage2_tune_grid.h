#pragma once
#include "ecm_stage2_giant_work.h"
#include <set>
#include <vector>

namespace ecm_stage2 { namespace tune {
constexpr const char *tail_sampling_model="giant_tail_grid_v1";
struct TailGridPoint {Word b2=0;const char *source="base";};
struct TailGrid {
    bool valid=false;
    const char *reason="invalid_policy";
    Word low=0,high=0,existing_chain_anchors=0,existing_ladder_anchors=0;
    std::vector<TailGridPoint> points;
};
// Integer fractions without overflowing even at large interval endpoints.
inline Word grid_fraction(Word value,Word numerator,Word denominator) {
    return value/denominator*numerator+(value%denominator)*numerator/denominator;
}
inline bool tune_grid_bounds(const std::vector<Word> &bounds,Word b1,Word d,Word p,Word batches,
                             Word &low,Word &high) {
    if(bounds.empty() || !d || !p)return false;
    low=*std::min_element(bounds.begin(),bounds.end());high=*std::max_element(bounds.begin(),bounds.end());
    Word first=0;
    if(!multiply(p-1,d,first) || b1==std::numeric_limits<Word>::max())return false;
    low=std::max(low,std::max(b1+1,first));
    Word cap=0;
    Word last=0;
    if(!add(high/d,2,last))return false;
    if(batches && multiply(p,batches,cap) && last>cap) {
        if(cap<2 || !multiply(cap-1,d,high) || !high)return false;--high;
    }
    return low<=high;
}
// All adaptive points stay inside the original B2 interval and its G-tree cap.
// Point chunk capacity comes from the current native planner, never a guessed
// width/device formula. Complexity is bounded by the input and requested count.
inline TailGrid tune_tail_grid(const std::vector<Word> &bounds,Word b1,Word d,Word p,
                               Word chunk,Word minimum,bool force,Word batches,unsigned requested) {
    TailGrid out;
    if(!d || !p || !chunk || chunk%p || requested>16)return out;
    out.valid=true;
    if(!requested){out.reason="disabled";return out;}
    if(!tune_grid_bounds(bounds,b1,d,p,batches,out.low,out.high)) {
        out.reason="no_admissible_range";return out;
    }
    if(out.low==out.high){out.reason="single_bound";return out;}
    if(force || chunk<minimum){out.reason="all_ladder";return out;}
    const Word lo=out.low/d+2,hi=out.high/d+2;
    std::set<Word> occupied;
    for(Word b2:bounds) {
        Word points=0;
        if(!add(b2/d,2,points)){out.valid=false;out.reason="work_overflow";return out;}
        const bool unique=occupied.insert(points).second;
        if(b2<out.low || b2>out.high)continue;
        GiantWork work;if(!giant_work(points,d,chunk,minimum,force,work)) {
            out.valid=false;out.reason="work_overflow";return out;
        }
        if(unique){if(!work.ladder_steps)++out.existing_chain_anchors;else ++out.existing_ladder_anchors;}
    }
    auto emit=[&](Word point,const char *source) {
        Word b2=0;if(point<2 || !multiply(point-2,d,b2))return false;
        b2=std::max(out.low,b2);
        GiantWork work;
        if(b2>out.high || !giant_work(point,d,chunk,minimum,force,work) || !occupied.insert(point).second)return false;
        out.points.push_back({b2,source});return true;
    };
    auto next_chain=[&](Word point,Word &result) {
        const Word remainder=point%chunk;
        if(!remainder || remainder>=minimum){result=point;return true;}
        return add(point-remainder,minimum,result);
    };
    // Mixed fitting needs at least seven anchors, including three per route.
    // Three base chain anchors plus three tails need one further chain point.
    // At most four extra chain points, independent of enormous B2 spans.
    const Word ladder=out.existing_ladder_anchors+requested;
    const Word required_chain=ladder>=3?std::max(Word(3),ladder>=7?Word(0):7-ladder):Word(3);
    const Word need=out.existing_chain_anchors>=required_chain?0:required_chain-out.existing_chain_anchors;
    Word first_chain=0;
    if(next_chain(lo,first_chain) && first_chain<=hi)for(Word j=0;j<need;++j) {
        Word desired=lo+grid_fraction(hi-lo,j+1,need+1),point=0;
        if(!next_chain(desired,point) || point>hi)point=first_chain;
        const auto attempts=occupied.size()+1;
        for(size_t attempt=0;attempt<attempts;++attempt) {
            if(emit(point,"chain_anchor"))break;
            if(point==std::numeric_limits<Word>::max() || !next_chain(point+1,point) || point>hi)point=first_chain;
        }
    }
    if(minimum<2){out.reason="no_ladder_tail";return out;}
    auto next_tail=[&](Word point,Word &result) {
        const Word remainder=point%chunk;
        if(remainder && remainder<minimum){result=point;return true;}
        if(!remainder)return add(point,1,result);
        Word next=0;return add(point-remainder,chunk,next) && add(next,1,result);
    };
    auto previous_tail=[&](Word point,Word &result) {
        const Word remainder=point%chunk,base=point-remainder;
        if(remainder){if(remainder<minimum){result=point;return true;}return add(base,minimum-1,result);}
        if(base<chunk)return false;return add(base-chunk,minimum-1,result);
    };
    Word first=0,last=0;
    if(!next_tail(lo,first) || !previous_tail(hi,last) || first>hi || last<lo || first>last) {
        out.reason="no_ladder_tail";return out;
    }
    const Word first_chunk=first/chunk,last_chunk=last/chunk;
    for(unsigned j=0;j<requested;++j) {
        Word q=first_chunk+(requested==1?(last_chunk-first_chunk)/2:
            grid_fraction(last_chunk-first_chunk,j,requested-1));
        const Word tail=requested==1?(minimum-1)/2:
            grid_fraction(minimum-1,(requested-1)+14*j,16*(requested-1));
        const Word wanted=std::max(Word(1),tail);
        // Preserve the requested residual when another chunk can contain it.
        // Clamping only the point can turn several distinct fractions into
        // adjacent copies of the same lower endpoint, despite available tails.
        bool exact_fraction=false;
        if(hi>=wanted) {
            const Word difference=lo>wanted?lo-wanted:0;
            const Word qlow=difference/chunk+(difference%chunk!=0),qhigh=(hi-wanted)/chunk;
            if(qlow<=qhigh){q=std::max(qlow,std::min(qhigh,q));exact_fraction=true;}
        }
        const auto attempts=occupied.size()+1;
        // A clipped chunk may contain only an already measured endpoint. Try
        // another eligible chunk rather than losing an available tail sample.
        bool emitted=false;
        for(size_t chunk_attempt=0;chunk_attempt<attempts && !emitted;++chunk_attempt) {
            Word base=0;if(!multiply(q,chunk,base)){out.valid=false;out.reason="work_overflow";return out;}
            Word start=0,end=0;
            if(!add(base,1,start))break;
            if(!add(base,minimum-1,end))end=std::numeric_limits<Word>::max();
            const Word left=std::max(lo,start),right=std::min(hi,end);
            if(left<=right) {
                Word point=0;
                if(exact_fraction) {
                    if(!add(base,wanted,point))point=right;
                    point=std::max(left,std::min(right,point));
                } else point=left+(requested==1?(right-left)/2:grid_fraction(right-left,j,requested-1));
                for(size_t attempt=0;attempt<attempts;++attempt) {
                    if(emit(point,"ladder_tail")){emitted=true;break;}
                    point=point==right?left:point+1;
                }
            }
            q=q==last_chunk?first_chunk:q+1;
        }
    }
    out.reason="generated";return out;
}
} } // namespace ecm_stage2::tune
