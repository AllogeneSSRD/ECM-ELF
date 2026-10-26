#define NOMINMAX
#ifdef _WIN32
#include <windows.h>
#endif
#include "../../src/ecm/stage2/planning/runtime_tune.h"
#include <iostream>
namespace p=ecm_stage2::tune::portable;
static unsigned checks=0;
static void require(bool ok,const char *message){++checks;if(!ok)throw std::runtime_error(message);}
static p::DSearchNode node(unsigned d,unsigned degree,double rank,double seconds=0,unsigned samples=0,unsigned region=0) {
    p::DSearchNode n;n.d=d;n.p=degree;n.rank=rank;n.seconds=seconds;n.samples=samples;n.region[0]=region;return n;
}
int main() {
    try {
        // Production M5351/cofactor4916: an observed 1.90% win must be used,
        // even when the incumbent has ten samples and narrower dispersion.
        std::vector<p::RuntimeCandidate> production(2);
        production[0].shape.d=1141140;production[0].priority=0;
        production[0].statistics={10,1792332823133745200ull,39.59522408000001,.17348454780529188};
        production[1].shape.d=1231230;production[1].priority=2;
        production[1].statistics={2,0,38.841368599999996,.3099226394523714};
        require(p::runtime_choice(production,false).index==1,"production fastest mean still blocked by adoption hysteresis");
        // The improved edge must try the next larger P region before filling
        // smaller-P gaps, rather than treating the NTT boundary as a wall.
        std::vector<p::DSearchNode> edge{
            node(1141140,103680,1,39.59522408000001,10),
            node(1231230,115200,1.01,38.841368599999996,2),
            node(1089270,108864,1.1),node(1381380,126720,1.2,0,0,1)};
        auto continuation=p::DSearch(edge,1141140,2,8).next();
        require(continuation.available && edge[continuation.index].d==1381380,
            "improving edge turned back instead of crossing the next P/NTT boundary");
        require(std::string(continuation.reason)=="improving_edge" && continuation.scale>1,
            "directed continuation has no identifiable reason/direction");
        auto misleading=edge;misleading.push_back(node(1185030,116640,.9));
        continuation=p::DSearch(misleading,1141140,2,8).next();
        require(continuation.available && misleading[continuation.index].d==1381380,
            "higher P but smaller D was mistaken for continuing the measured D trend");
        auto backfilled=misleading;backfilled.push_back(node(1108380,112896,1.1,41.02137495,2));
        continuation=p::DSearch(backfilled,1141140,2,8).next();
        require(continuation.available && backfilled[continuation.index].d==1381380,
            "slower backfill shrank the successful forward step to an empty narrow band");
        auto legacy_regions=edge;
        legacy_regions.push_back(node(930930,86400,2,45.29672205,2,1));
        legacy_regions.push_back(node(1018710,105840,2,43.9585362,2,2));
        legacy_regions[3].region[0]=3;
        continuation=p::DSearch(legacy_regions,1141140,2,8,3).next();
        require(continuation.available && legacy_regions[continuation.index].d==1381380,
            "old losing regions exhausted the single measured-edge boundary allowance");
        auto bracket=edge;bracket[3].samples=2;bracket[3].seconds=42;
        continuation=p::DSearch(bracket,1141140,2,8).next();
        require(continuation.available && bracket[continuation.index].d==1089270 &&
            std::string(continuation.reason)!="improving_edge","measured uphill edge did not reopen refinement");
        auto descending=edge;descending[0].seconds=38;descending[2].region[0]=1;
        descending.push_back(node(1021020,92160,1.3,0,0,1));
        continuation=p::DSearch(descending,1141140,2,8).next();
        require(continuation.available && descending[continuation.index].d==1021020 && continuation.scale<1,
            "measured decreasing-P direction was replaced by a larger-D bias");
        auto full_regions=legacy_regions;full_regions[4].region[0]=3;
        full_regions[3].region[0]=4;
        full_regions.push_back(node(746130,69120,2,53.1207831,2,1));
        continuation=p::DSearch(full_regions,1141140,2,8,3).next();
        require(!continuation.available || full_regions[continuation.index].region[0]!=4,
            "directed continuation exceeded its single additional-region allowance");
        require(!p::DSearch(edge,1141140,2,1).next().available,"directed continuation ignored the D budget");
        std::vector<unsigned> production_order{0,1};
        do {
            std::vector<p::RuntimeCandidate> permuted;
            for(auto i:production_order)permuted.push_back(production[i]);
            const auto chosen=p::runtime_choice(permuted,false);
            require(chosen.available && permuted[chosen.index].shape.d==1231230,
                "fastest production route depends on iteration order");
        }while(std::next_permutation(production_order.begin(),production_order.end()));
        production[1].statistics.seconds=production[0].statistics.seconds;
        require(p::runtime_choice(production,false).index==0,"an equal mean displaced the incumbent");
        production[1].statistics.count=1;production[1].statistics.seconds=1;
        require(p::runtime_choice(production,false).index==0,"one observation was promoted to a stable route");
        production[1].statistics.count=2;production[1].statistics.seconds=std::numeric_limits<double>::quiet_NaN();
        require(p::runtime_choice(production,false).index==0,"nonfinite timing replaced a valid route");
        p::Shape large;large.target_bits=4916;large.arithmetic_bits=5351;large.carrier=5351;
        large.mersenne=true;large.b1=260000000;large.b2=1300000000000;large.d=1381380;
        auto large_query=[](ecm_stage2::Word m,int,ecm_stage2::Word *n,ecm_stage2::Word *slots) {
            *n=1ull<<28;*slots=2*m-1;return true;
        };
        p::Work large_work;
        require(p::work(large,large_query,[](ecm_stage2::Word,ecm_stage2::Word){return 1.;},large_work) &&
            large_work.largest_ntt==(1ull<<28),"model rejected a larger NTT already accepted by native packing");
        ecm_stage2::tune::NttReferences large_reference;
        require(ecm_stage2::tune::ntt_references(large_work.p,large_work.points,large.arithmetic_bits,
            large_query,large.batch_bytes,3,false,0,{},large_reference) && large_reference.valid &&
            !large_reference.complete,"unmeasured long NTT was rejected or falsely marked as measured");
        std::vector<p::DSearchNode> nodes{node(100,1000,100,100,2),node(95,950,90),node(80,800,91)};
        auto step=p::DSearch(nodes,100).next();require(step.available && nodes[step.index].d==95,"model seed not scheduled");
        nodes[1].samples=1;nodes[1].seconds=99;
        require(p::DSearch(nodes,100).next().index==1,"independent repeat not completed first");
        nodes[1].samples=2;
        require(!p::d_search_improves(99,0,100,0),"hysteresis was weakened");
        require(p::DSearch(nodes,100).center()==1,"geometric search stopped at the 2 percent adoption gate");
        step=p::DSearch(nodes,100).next();require(step.available && nodes[step.index].d==80,"farther improvement hidden behind small step");
        nodes[2].samples=2;nodes[2].seconds=96;
        require(p::d_search_improves(96,0,100,0),"real improvement cannot be adopted");
        require(!p::DSearch(nodes,100).next().available,"completed representatives retried");
        // Tight catalogue neighbors have no special status: pick a meaningful
        // change in P even when there are hundreds of closer entries.
        nodes={node(100,1000,1,100,2)};
        for(unsigned degree=700;degree<=1400;++degree)if(degree!=1000)
            nodes.push_back(node(10000+degree,degree,100+std::abs(double(degree)-800)*.001));
        step=p::DSearch(nodes,100).next();
        require(step.available && nodes[step.index].p>=740 && nodes[step.index].p<=865,"scaled step became a catalogue-neighbor walk");
        const auto selected=step.index;nodes[selected].samples=2;nodes[selected].seconds=110;
        const auto reconstructed=p::DSearch(nodes,100).next();
        require(!reconstructed.available || reconstructed.index!=selected,"reconstruction repeats a completed scale representative");
        auto snapshot=nodes;
        const auto again=p::DSearch(snapshot,100).next();
        require(reconstructed.available==again.available && reconstructed.index==again.index,"search needs a hidden per-process journal");
        // An uphill local region does not close another execution region.
        nodes={node(100,1000,1,100,2),node(80,800,105,105,2),node(125,1250,106,106,2),
            node(90,900,107,107,2),node(200,2000,120,0,0,1)};
        step=p::DSearch(nodes,100,2,4).next();
        require(step.available && nodes[step.index].d==200 && std::string(step.reason)=="region_restart","jump reservation lost");
        nodes[4].samples=2;nodes[4].seconds=90;
        require(!p::DSearch(nodes,100,2,4).next().available,"candidate count cap ignored");
        auto budget=p::runtime_search_budget(nodes,100,10);
        require(budget.candidates==4,"initial exploration cap wrong");
        nodes[4].samples=8;budget=p::runtime_search_budget(nodes,100,10);
        require(budget.candidates==8,"stable winner did not unlock the refinement round");
        nodes={node(100,1000,1,100,10),node(95,950,.99,99,2)};
        require(p::runtime_search_budget(nodes,100,10).candidates==8,
            "slightly faster unadopted candidate blocked the next exploration round");
        nodes={node(100,1000,1,100,2),node(80,800,120,170,2)};
        budget=p::runtime_search_budget(nodes,100,10);
        require(budget.remaining==0 && !budget.permits(90,0),"observed extra-time budget ignored");
        nodes={node(100,1000,1,100,2)};budget=p::runtime_search_budget(nodes,100,10);
        require(!budget.permits(110,0,3) && budget.permits(110,0,20),"short queue did not restrict extra exploration");
        require(!budget.permits(170,0,20) && budget.permits(150,0,20),"two-observation predicted cost ignored");
        require(p::DSearch(std::vector<p::DSearchNode>{},100).next().available==false,"empty pool not handled");
        nodes={node(100,1000,1,100,2),node(80,800,.1)};nodes[1].rejected=true;
        require(!p::DSearch(nodes,100).next().available,"rejected native shape selected");
        nodes={node(100,1000,1),node(80,800,2)};nodes[0].rejected=true;
        require(p::DSearch(nodes,100).next().available && p::DSearch(nodes,100).next().index==1,
            "one rejected model seed incorrectly rejected the whole region");
        std::vector<p::RuntimeCandidate> routes(2);
        routes[0].priority=1;routes[0].statistics={2,0,100,0};routes[1].priority=3;routes[1].probe_allowed=false;
        auto choice=p::runtime_choice(routes);
        require(choice.available && choice.index==0 && !choice.probe,"deferred trial replaced reference");
        routes[1].probe_allowed=true;
        require(p::runtime_choice(routes).probe && !p::runtime_choice(routes,false).probe,"runtime disable no longer suppresses trials");
        routes[1].statistics={2,100,99,1};routes[0].statistics.preferred_order=200;
        require(p::runtime_choice(routes).index==1,"dispersion vetoed the fastest repeated mean");
        routes[1].statistics={2,100,90,.1};
        require(p::runtime_choice(routes).index==1,"significant measured win was rejected");
        // Replayed production pattern: a newly preferred ordinary route must
        // not hide a faster known carrier behind an intermediate candidate.
        routes.resize(3);
        routes[0].shape.d=180180;routes[0].statistics={5,200,2.07930226,.0206580101};routes[0].priority=1;
        routes[1].shape.d=174720;routes[1].statistics={2,0,1.8664843,.0205118949};routes[1].priority=2;
        routes[2].shape.d=210210;routes[2].statistics={10,100,1.75911053,.0250890914};routes[2].priority=2;
        require(p::LocalSearch::improves(routes[2].statistics.timing(),routes[0].statistics.timing()) &&
            !p::LocalSearch::improves(routes[2].statistics.timing(),routes[1].statistics.timing()),
            "production ordering regression fixture no longer represents overlapping intermediate bands");
        std::vector<unsigned> order{0,1,2};
        do {
            std::vector<p::RuntimeCandidate> permuted;for(auto i:order)permuted.push_back(routes[i]);
            const auto selected=p::runtime_choice(permuted,false);
            require(selected.available && !selected.probe && permuted[selected.index].shape.d==210210,
                "candidate ordering hid a qualified faster route behind an unexecuted intermediate");
        }while(std::next_permutation(order.begin(),order.end()));
        routes[2].statistics.preferred_order=300;
        require(p::runtime_choice(routes,false).index==2,"overlapping alternatives displaced the successful incumbent");
        p::RuntimePauseHints hints;const std::string a(64,'a'),b(64,'b'),c(64,'c');
        require(hints.assign(a+","+b) && hints.contains(a) && hints.contains(b),"parent pause roundtrip failed");
        hints.add(c);require(hints.keys.size()==2,"parent pause hints are unbounded");
        require(!hints.assign(a+",broken") && hints.keys.empty(),"bad hint did not fail open");
        require(!hints.assign(a+","+b+","+c) && hints.keys.empty(),"oversized hint accepted");
        require(hints.assign(a+","+a) && hints.keys.size()==1,"duplicate pause hints not deduplicated");
        require(!hints.assign(a+",") && hints.keys.empty(),"torn parent hint accepted");
        p::Shape reference;reference.d=180180;
        const auto old_scope=p::runtime_pause_scope("profileA/conditionA/N/B1/B2/version",reference,routes);
        require(old_scope==p::runtime_pause_scope("profileA/conditionA/N/B1/B2/version",reference,routes),"unchanged pause scope is not deterministic");
        require(old_scope!=p::runtime_pause_scope("profileB/conditionA/N/B1/B2/version",reference,routes),"new observations did not invalidate pause");
        reference.arithmetic_bits=2003;
        require(old_scope!=p::runtime_pause_scope("profileA/conditionA/N/B1/B2/version",reference,routes),"width/path change did not invalidate pause");
        reference.arithmetic_bits=0;routes[2].shape.frontier=false;
        require(old_scope!=p::runtime_pause_scope("profileA/conditionA/N/B1/B2/version",reference,routes),"actual candidate residency did not invalidate pause");
        routes[2].shape.frontier=true;++routes[2].statistics.count;
        require(old_scope!=p::runtime_pause_scope("profileA/conditionA/N/B1/B2/version",reference,routes),"changed candidate statistics did not invalidate pause");
        p::RuntimePlanMemo<unsigned,std::string> memo;const p::RuntimeMemorySnapshot memory{100,200};
        memo.scope("N/sigma/B1/B2/device/environment");
        require(memo.put(1,"approved",memory,100,memory,true) && memo.find(1,memory) && *memo.find(1,memory)=="approved","approved local plan not reused");
        require(!memo.find(1,{99,200}) && !memo.find(1,memory),"changed free memory reused a stale plan");
        require(!memo.put(1,"failed",memory,100,memory,false) && !memo.find(1,memory),"failed admission cached");
        require(!memo.put(1,"wrong snapshot",memory,99,memory,true),"reported native free snapshot ignored");
        require(!memo.put(1,"changing",memory,100,{101,200},true),"memory changed during planning but plan cached");
        require(!memo.put(1,"wrong total",memory,100,{100,201},true),"device total-memory change ignored");
        require(!memo.put(1,"invalid",{},0,{},true),"failed live memory query accepted");
        memo.put(1,"approved",memory,100,memory,true);memo.scope("new environment");
        require(!memo.find(1,memory),"changed planner input/environment did not invalidate memo");
        for(unsigned i=0;i<40;++i)memo.put(i,"approved",memory,100,memory,true);
        require(memo.size()<=32 && memo.find(39,memory),"worker plan memo is unbounded");
        p::RuntimeSearchAttempts attempts(100);bool advanced=true;
        for(unsigned i=0;i<25;++i)advanced&=attempts.next();
        require(advanced && attempts.admissions==0 && attempts.next(),"cheap model rejection consumed native-admission budget");
        for(unsigned i=0;i<16;++i)++attempts.admissions;
        require(!attempts.next(),"native-admission budget not enforced");
        p::RuntimeSearchAttempts bounded(3);unsigned steps=0;while(bounded.next())++steps;
        require(steps==4,"pure search rejection loop is unbounded");
        p::RuntimeRouteHints route_hints;route_hints.scope(a);
        p::RuntimeRouteHint route{180180,2003,100,200,1,1,34560,32768,0},restored;
        require(route_hints.put(route,true,memory) && route_hints.find(180180,2003,memory,restored) && restored.values()==route.values(),
            "approved route hint not restored");
        const auto encoded=route_hints.text();p::RuntimeRouteHints decoded;
        require(decoded.assign(encoded) && decoded.find(180180,2003,memory,restored),"route hint roundtrip failed");
        require(!decoded.find(180180,2003,{101,200},restored) && !decoded.find(180180,2003,memory,restored),"changed live free memory reused a parent route");
        decoded.assign(encoded);decoded.scope(b);
        require(!decoded.find(180180,2003,memory,restored),"changed route scope retained a previous parent plan");
        decoded.scope(a);require(!decoded.put(route,false,memory) && decoded.entries.empty(),"failed admission retained a route hint");
        require(!decoded.put(route,true,{99,200}) && decoded.entries.empty(),"free memory changed while approving a route");
        require(!decoded.assign(encoded+';') && decoded.entries.empty(),"torn route record accepted");
        require(!decoded.assign(a+"|180180,2003,18446744073709551616,200,1,1,34560,32768,0"),"overflowing route field accepted");
        require(!decoded.assign(a+"|180180,2003,100,200,0,1,34560,32768,0"),"frontier without fold accepted");
        require(!decoded.assign(a+"|180181,2003,100,200,1,1,34560,32768,0"),"invalid odd route D accepted");
        const auto row=encoded.substr(65);
        require(!decoded.assign(encoded+';'+row),"duplicate route entries accepted");
        decoded.scope(a);for(unsigned i=0;i<40;++i){route.d=180180+2*i;decoded.put(route,true,memory);}
        require(decoded.entries.size()==32 && decoded.text().size()<8192,"parent route cache is unbounded");
        p::RuntimeRouteHints unscoped;
        require(!unscoped.put(route,true,memory),"unscoped route proof accepted");
        std::cout<<"{\"complete\":true,\"checks\":"<<checks<<",\"scaled_steps\":true,\"threshold_escape\":true,"
            "\"region_restarts\":true,\"reconstruction\":true,\"candidate_budget\":true,\"extra_time_budget\":true,"
            "\"runtime_fastest_mean\":true,\"improving_edge\":true,\"CUDA_calls\":0}\n";
    }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
