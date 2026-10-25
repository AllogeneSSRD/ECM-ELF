#include "../../src/ecm/backend/cuda/stage2/host_wait_policy.h"
#include <stdexcept>
#include <iostream>
#include <vector>

struct Backend {
    const int success=0,not_ready=34;
    std::vector<int> statuses,delays;
    unsigned queries=0,syncs=0;
    bool timer_ok=true;
    int sync_status=0;
    int query(){if(queries>=statuses.size())throw std::runtime_error("unexpected query");return statuses[queries++];}
    int synchronize(){++syncs;return sync_status;}
    bool pause(unsigned us){delays.push_back(us);return timer_ok;}
};
void require(bool value,const char *message){if(!value)throw std::runtime_error(message);}
int main(){try{
    using stage2_wait::complete_event;
    stage2_wait::Counters stats;
    Backend legacy;
    require(complete_event(legacy,false,stats)==0 && legacy.syncs==1 && legacy.queries==0,"legacy must call native sync only");
    Backend ready;ready.statuses={0};
    require(complete_event(ready,true,stats)==0 && ready.syncs==1 && ready.delays.empty(),"ready path must retain completion boundary without sleep");
    Backend pending;pending.statuses={34,34,34,34,34,34,0};
    require(complete_event(pending,true,stats)==0 && pending.syncs==1 &&
        pending.delays==std::vector<int>({32,64,128,256,256,256}),"sleep backoff/cap or final sync changed");
    Backend error;error.statuses={719};
    require(complete_event(error,true,stats)==719 && error.syncs==0 && error.delays.empty(),"CUDA query failure must propagate");
    Backend late_error;late_error.statuses={34,719};
    require(complete_event(late_error,true,stats)==719 && late_error.syncs==0 && late_error.delays.size()==1,"late CUDA failure must stop polling");
    Backend failed_timer;failed_timer.statuses={34};failed_timer.timer_ok=false;
    require(complete_event(failed_timer,true,stats)==0 && failed_timer.syncs==1 && failed_timer.queries==1,"unavailable timer must fall back to real sync");
    Backend sync_error;sync_error.statuses={0};sync_error.sync_status=719;
    require(complete_event(sync_error,true,stats)==719,"completion sync error must propagate");
    require(stats.calls==7 && stats.fallbacks==1 && stats.sleeps==7,"diagnostic accounting changed");
    std::cout<<"event wait policy: legacy, ready, pending, bounded backoff, errors, timer fallback, visibility passed\n";
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
