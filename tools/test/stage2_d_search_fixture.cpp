#define NOMINMAX
#ifdef _WIN32
#include <windows.h>
#endif
#include "../../src/core/ecm_stage2_runtime_tune.h"
#include <iostream>
namespace p=ecm_stage2::tune::portable;
static unsigned checks=0;
static void require(bool ok,const char *message){++checks;if(!ok)throw std::runtime_error(message);}
static p::DSearchNode node(unsigned d,unsigned degree,double rank,double seconds=0,unsigned samples=0,unsigned region=0) {
    p::DSearchNode n;n.d=d;n.p=degree;n.rank=rank;n.seconds=seconds;n.samples=samples;n.region[0]=region;return n;
}
int main() {
    try {
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
        require(p::runtime_choice(routes).index==0,"noisy observation weakened default performance protection");
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
        std::cout<<"{\"complete\":true,\"checks\":"<<checks<<",\"scaled_steps\":true,\"threshold_escape\":true,"
            "\"region_restarts\":true,\"reconstruction\":true,\"candidate_budget\":true,\"extra_time_budget\":true,"
            "\"unchanged_hysteresis\":true,\"CUDA_calls\":0}\n";
    }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
