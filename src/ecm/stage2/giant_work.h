#pragma once
#include "giant_schedule.h"
#include <cmath>

namespace ecm_stage2 {
// Exact route counts for the normal chunk loop; no timing or memory admission.
struct GiantWork {
    Word chain_points=0,ladder_points=0,chain_chunks=0,ladder_chunks=0,ladder_steps=0;
};
inline bool giant_ladder_steps(Word first,Word last,Word d,Word &steps) {
    steps=0;Word scalar=0;
    if(!first || first>last || !d || !multiply(last,d,scalar))return false;
    // Each scalar i*D performs floor(log2(i*D)) ladder iterations. Count the
    // indices at each power-of-two boundary without iterating over the points.
    for(unsigned bit=1;bit<64;++bit) {
        const Word power=Word(1)<<bit;
        const Word threshold=power/d+(power%d!=0);
        const Word low=std::max(first,threshold);
        if(low<=last && !add(steps,last-low+1,steps))return false;
    }
    return true;
}
inline bool giant_work(Word points,Word d,Word chunk,Word minimum,bool force,GiantWork &out,
                       Word p=1,bool balance=false) {
    out=GiantWork{};
    if(!points || !d || !chunk)return false;
    const auto schedule=giant_schedule(points,chunk,p,minimum,balance && !force);
    if(!schedule.valid)return false;
    Word first=1;
    for(const auto &run:schedule.runs) {
        Word n=0;if(!multiply(run.points,run.repeat,n))return false;
        if(!force && run.points>=minimum) {
            if(!add(out.chain_points,n,out.chain_points) || !add(out.chain_chunks,run.repeat,out.chain_chunks))return false;
        } else {
            Word last=0,steps=0;
            if(!add(first,n-1,last) || !giant_ladder_steps(first,last,d,steps) ||
               !add(out.ladder_points,n,out.ladder_points) || !add(out.ladder_chunks,run.repeat,out.ladder_chunks) ||
               !add(out.ladder_steps,steps,out.ladder_steps))return false;
        }
        if(n<points-first+1)first+=n;
    }
    return true;
}

struct GiantPointWork {
    Word seed_ladders=0,seed_steps=0,xadds=0,ladder_points=0,ladder_steps=0;
    Word max_chain_steps=0;
    double modmuls()const {return 5.*seed_ladders+11.*seed_steps+6.*xadds+5.*ladder_points+11.*ladder_steps;}
};
// Paired seeds ladder on [D]Q using i, rather than i*D. Count their bit
// lengths over repeated chunks without expanding either points or seeds.
// This counts dependent modular products, not GPU cycles. It excludes the
// shared [D]Q construction, normalization, segments and polynomial arithmetic.
inline bool giant_point_work(Word points,Word d,Word chunk,Word minimum,bool force,
        Word length,Word p,bool balance,GiantPointWork &out) {
    out={};if(length<2)return false;
    GiantWork routes;if(!giant_work(points,d,chunk,minimum,force,routes,p,balance))return false;
    out.ladder_points=routes.ladder_points;out.ladder_steps=routes.ladder_steps;
    const auto schedule=giant_schedule(points,chunk,p,minimum,balance && !force);
    Word first=1;
    for(const auto &run:schedule.runs) {
        Word n=0;if(!multiply(run.points,run.repeat,n))return false;
        if(!force && run.points>=minimum) {
            const Word groups=run.points/length+(run.points%length!=0);
            Word seeds=0,adds=0;
            const Word last=run.points-(groups-1)*length;
            if(!multiply(groups,run.repeat,seeds) || !add(out.seed_ladders,seeds,out.seed_ladders) ||
               !multiply(run.points-2*(groups-1)-std::min(last,Word(2)),run.repeat,adds) ||
               !add(out.xadds,adds,out.xadds))return false;
            out.max_chain_steps=std::max(out.max_chain_steps,std::max(Word(2),std::min(length,run.points))-2);
            for(unsigned bit=1;bit<64;++bit) {
                const Word threshold=Word(1)<<bit;
                Word above=seeds;
                if(threshold>first) {
                    const Word j=(threshold-first)/run.points;
                    if(j>=run.repeat)above=0;
                    else {
                        const Word offset=threshold-(first+j*run.points);
                        const Word below=std::min(groups,offset/length+(offset%length!=0));
                        above=(run.repeat-j-1)*groups+groups-below;
                    }
                }
                if(!add(out.seed_steps,above,out.seed_steps))return false;
            }
        }
        if(n<points-first+1)first+=n;
    }
    return std::isfinite(out.modmuls());
}
} // namespace ecm_stage2
