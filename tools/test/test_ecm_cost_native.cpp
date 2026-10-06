// CPU-only solver/loader regression. Synthetic packing is not a CUDA gate.
#include "../../src/core/ecm_stage2_cost_profile.h"
#include <cstring>
#include <iostream>
bool ecm_cuda_stage2_shape_query(uint64_t p,int bits,uint64_t *n,uint64_t *out) {
    uint64_t size=1;while(size<2*p*(uint64_t)bits)size*=2;
    if(n)*n=size;if(out)*out=2*p-1;return true;
}
int main() {
    namespace c=ecm_stage2::cost;using ecm_stage2::Word;
    auto check=[](bool ok,const char *name){if(!ok)throw std::runtime_error(name);};
    c::Work work{2203};
    for(Word n=0;n<260;++n){auto t=work.stats(n);check(t.pairs==(n?n-1:0),"exact pairs");}
    c::Profile p;p.features=7;p.chain_min=8192;p.uuid=std::string(32,'1');p.major=8;p.minor=9;
    p.runtime=p.driver=13030;p.fixed=3;p.stage1.push_back({2203,1000,1,1,0.1});
    c::Scope low;low.bits=2203;low.b1=1000;low.arena_mb=32768;low.resident=0;low.d={30030};
    low.pmin=low.pmax=c::phi(30030)/2;low.gmin=low.gmax=1;low.b2min=3000000;
    low.b2max=30030*(low.pmax-2);low.rate.fill(1e-9);low.covered.fill(1);low.cold=0.25;
    auto high=low;high.d={60060};high.pmin=high.pmax=c::phi(60060)/2;
    high.gmin=2;high.gmax=1000;high.b2min=3000000000;high.b2max=6000000000;p.scopes={low,high};
    EcmStage2DeviceInfo dev;std::strcpy(dev.uuid_hex,p.uuid.c_str());dev.major=8;dev.minor=9;
    dev.runtime=dev.driver=13030;dev.fixed_mode=3;dev.free_bytes=64ull<<30;
    c::Request r;r.bits=2203;r.b1=1000;r.owner_mb=0;
    const auto joint=c::choose(p,r,dev);
    check((joint.b2>=low.b2min&&joint.b2<=low.b2max)||(joint.b2>=high.b2min&&joint.b2<=high.b2max),"union/gap admission");
    r.b2min=r.b2max=4000000;const auto g1=c::choose(p,r,dev);
    check(g1.g==1&&!g1.resident&&g1.local_inverse>0&&g1.phases[6]==0,"G1 local inverse");
    check(g1.gt_pairs==g1.i-1,"G1 tree count");
    r.b2min=r.b2max=low.b2max;const auto boundary=c::choose(p,r,dev);
    check(boundary.i==boundary.p&&boundary.root_reduction>0,"G1 root boundary");
    r.b2min=r.b2max=1000000000;bool rejected=false;
    try{c::choose(p,r,dev);}catch(const std::runtime_error&){rejected=true;}
    check(rejected,"unmeasured gap rejected");
    p.scopes[1].covered.fill(0);r.b2min=r.b2max=3000000000;rejected=false;
    try{c::choose(p,r,dev);}catch(const std::runtime_error&){rejected=true;}
    check(rejected,"unmeasured giant route rejected");
    std::cout<<"cost native CPU: exact pairs, union, G1 inverse/root, gap and route coverage passed\n";
}
