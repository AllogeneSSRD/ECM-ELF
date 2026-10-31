#pragma once
#include "geometry.h"
#include <vector>

namespace ecm_stage2 {
// A compact schedule has at most three runs, even for very large B2. A run
// repeats equal point chunks; only the final two can borrow whole G batches.
struct GiantChunkRun {Word points=0,repeat=0;};
struct GiantSchedule {
    Word points=0,capacity=0,p=0,minimum=0,borrow=0,count=0;
    bool valid=false;
    std::vector<GiantChunkRun> runs;
    Word take(Word remaining)const {
        if(remaining<=capacity)return remaining;
        return borrow && remaining/capacity==1?capacity-borrow:capacity;
    }
    Word repeatable(Word remaining)const {
        const Word full=remaining/capacity;
        return borrow && full?full-1:full;
    }
};
inline Word giant_tail_borrow(Word points,Word capacity,Word p,Word minimum,bool rebalance) {
    if(!rebalance || !capacity || !p || capacity%p || points<=capacity || !minimum)return 0;
    const Word tail=points%capacity;
    if(!tail || tail>=minimum)return 0;
    const Word deficit=minimum-tail;
    Word borrow=0;
    if(!multiply(deficit/p+(deficit%p!=0),p,borrow) || borrow>=capacity ||
       capacity-borrow<minimum || tail>capacity-borrow)return 0;
    return borrow;
}
inline GiantSchedule giant_schedule(Word points,Word capacity,Word p,Word minimum,bool balance) {
    GiantSchedule s;s.points=points;s.capacity=capacity;s.p=p;s.minimum=minimum;
    if(!points || !capacity || !p || capacity%p)return s;
    const Word full=points/capacity,tail=points%capacity;
    s.borrow=giant_tail_borrow(points,capacity,p,minimum,balance);
    s.count=full+(tail!=0);
    if(s.borrow) {
        if(full>1)s.runs.push_back({capacity,full-1});
        s.runs.push_back({capacity-s.borrow,1});s.runs.push_back({tail+s.borrow,1});
    } else {
        if(full)s.runs.push_back({capacity,full});
        if(tail)s.runs.push_back({tail,1});
    }
    s.valid=true;return s;
}
inline Word giant_chunk_take(Word remaining,Word capacity,Word p,Word minimum,bool balance) {
    if(remaining<=capacity)return remaining;
    const auto borrow=giant_tail_borrow(remaining,capacity,p,minimum,balance);
    return borrow && remaining/capacity==1?capacity-borrow:capacity;
}
// Smallest upward bound covering an integer number of P-point G batches,
// with no sub-threshold point chunk. mode=1 enables rounding; 0 preserves B2.
// Failure preserves the input bound; callers report why rounding was skipped.
inline bool giant_round_b2(Word b2,Word d,Word p,Word capacity,Word minimum,unsigned mode,Word &actual) {
    actual=b2;if(!mode)return true;
    if(mode>1 || !d || !p || !capacity || capacity%p || capacity<minimum)return false;
    Word points=0;if(!add(b2/d,2,points))return false;
    const Word unit=p;
    Word rounded=0;if(!multiply(points/unit+(points%unit!=0),unit,rounded))return false;
    const Word tail=rounded%capacity;
    if(tail && tail<minimum) {
        Word extra=0;
        if(!multiply((minimum-tail)/unit+((minimum-tail)%unit!=0),unit,extra) || !add(rounded,extra,rounded))return false;
    }
    Word bound=0;if(rounded<2 || !multiply(rounded-2,d,bound))return false;
    actual=std::max(b2,bound);return true;
}
} // namespace ecm_stage2
