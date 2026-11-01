#include "../../src/ecm/stage2/giant_work.h"
#include "../../src/ecm/stage2/planning/giant_state.h"
#include "../../src/ecm/stage2/giant_policy.h"
#include <iostream>
#include <string>
#include <stdexcept>
using namespace ecm_stage2;
static Word checks=0;
static void check(bool v){++checks;if(!v)throw std::runtime_error("schedule check "+std::to_string(checks));}
int main(){try{
    auto s=giant_schedule(149940,138240,17280,32768,true);
    check(s.valid && s.borrow==34560 && s.runs.size()==2 && s.runs[0].points==103680 && s.runs[1].points==46260);
    GiantWork a,b;
    check(giant_work(149940,180180,138240,32768,false,a,17280,false));
    check(giant_work(149940,180180,138240,32768,false,b,17280,true));
    check(a.ladder_points==11700 && b.ladder_points==0 && b.chain_points==149940 && b.chain_chunks==2);
    Word bound=0;check(giant_round_b2(27015978990ull,180180,17280,138240,32768,1,bound));
    check(bound==Word(172800-2)*180180);
    check(!giant_round_b2(~Word(0),180180,17280,138240,32768,1,bound));
    check(!giant_round_b2(10000,210,24,24,32768,1,bound));
    for(Word p:{24ull,240ull,17280ull})for(Word batches:{1ull,2ull,4ull,8ull})
    for(Word minimum:{0ull,127ull,8192ull,32768ull})for(Word n:{1ull,p,p+1,2*p,3*p+17,9*p+1,17*p+13})
    for(bool balance:{false,true})for(bool force:{false,true}){
        const Word cap=p*batches;
        s=giant_schedule(n,cap,p,minimum,balance && !force);check(s.valid);
        Word count=0,total=0,chain=0,ladder=0,steps=0;
        for(const auto &run:s.runs)for(Word i=0;i<run.repeat;++i){
            check(run.points && run.points<=cap && (total+run.points==n || run.points%p==0));
            check(run.points==s.take(n-total));
            if(!force && run.points>=minimum)chain+=run.points;
            else {ladder+=run.points;for(Word j=1;j<=run.points;++j){Word v=(total+j)*210;while(v>1){++steps;v>>=1;}}}
            total+=run.points;++count;
        }
        check(total==n && count==s.count && count==n/cap+(n%cap!=0));
        GiantWork work;check(giant_work(n,210,cap,minimum,force,work,p,balance));
        check(work.chain_points==chain && work.ladder_points==ladder && work.ladder_steps==steps);
        for(Word length:{32ull,64ull}) {
            GiantPointWork point;check(giant_point_work(n,210,cap,minimum,force,length,p,balance,point));
            Word seeds=0,seed_steps=0,adds=0,first=1;
            for(const auto &run:s.runs)for(Word i=0;i<run.repeat;++i) {
                if(!force && run.points>=minimum)for(Word k=0;k<run.points;k+=length) {
                    ++seeds;Word index=first+k;while(index>1){++seed_steps;index>>=1;}
                    const auto count=std::min(length,run.points-k);if(count>2)adds+=count-2;
                }
                first+=run.points;
            }
            check(point.seed_ladders==seeds && point.seed_steps==seed_steps && point.xadds==adds &&
                  point.ladder_points==ladder && point.ladder_steps==steps);
        }
        GiantMemoryPolicy policy;policy.chain_min=minimum;policy.force_ladder=force;policy.tail_balance=balance;
        GiantMemoryPlan memory;check(giant_memory_plan(p,n,126,cap,policy,memory));
        GiantMemoryState state({p,n,126,cap,policy});check(state.init());
        while(state.consumed_points<n){check(state.chunk_begin());while(state.chunk_remaining)check(state.tree_done(std::min(p,state.chunk_remaining)));}
        check(state.accumulate());check(memory.peak_bytes==state.peak && memory.after_giant_bytes<=state.live);
        check(state.close() && state.live==0);
        if(cap>=minimum){
            Word actual=0;check(giant_round_b2(n*210,210,p,cap,minimum,1,actual));
            const Word points=actual/210+2;check(actual>=n*210 && points%p==0);
            check(giant_work(points,210,cap,minimum,false,work,p,false) && work.ladder_points==0);
        }
    }
    s=giant_schedule(1000000000000ull+17,138240,17280,32768,true);
    check(s.valid && s.runs.size()<=3);
    {GiantPolicyScope x(1,32);check(giant_tail_balance_requested() && giant_chain_points()==32);
     {GiantPolicyScope y(0,64);check(!giant_tail_balance_requested() && giant_chain_points()==64);}
     check(giant_tail_balance_requested() && giant_chain_points()==32);}
    // The policy recorded in a plan is effective, not just permission to borrow.
    GiantMemoryPolicy permission;permission.tail_balance=true;
    GiantMemoryPlan identity;
    check(giant_memory_plan(20736,130213,126,145152,permission,identity));
    check(!identity.policy.tail_balance);
    check(giant_memory_plan(17280,149940,126,138240,permission,identity));
    check(identity.policy.tail_balance);
    permission.force_ladder=true;
    check(giant_memory_plan(17280,149940,126,138240,permission,identity));
    check(!identity.policy.tail_balance);
    std::cout<<"giant_schedule: checks="<<checks<<" bad=0\n";return 0;
}catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 2;}}
