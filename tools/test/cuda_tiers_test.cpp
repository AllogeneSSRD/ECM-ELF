#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <sstream>
#include <stdexcept>
struct Tier {int bits,tpi,group,mont,dev,mont_tpb,edwards_tpb;};
#define ECM_CUDA_ALL_TIERS
#define ECM_CUDA_TIER(W,T,G,M,D,A,MT,ET) {W,T,G,M,D,MT,ET},
static const Tier all[]={
#include "../../src/ecm/backend/cuda/dispatch/tiers.def"
};
#undef ECM_CUDA_TIER
#undef ECM_CUDA_ALL_TIERS
#define ECM_CUDA_TIER(W,T,G,M,D,A,MT,ET) W,
static const int selected[]={0,
#include "../../src/ecm/backend/cuda/dispatch/tiers.def"
};
#undef ECM_CUDA_TIER
#define CAPACITY(NW,FN,...) NW,
static const int capacities[]={ECM_CUDA_STAGE2_CAPACITIES(CAPACITY,unused)};
#undef CAPACITY
static void require(bool value){if(!value)throw std::runtime_error("tier contract mismatch");}
int main(int argc,char **argv) {
    try {
        require((argc==1 || argc==2) && sizeof(all)/sizeof(*all)==32);
        const int golden[]={128,192,256,384,512,768,1024,1280,1536,1792,2048,
            2560,3072,3584,4096,4608,5120,5632,6144,6656,7168,7680,8192,
            9216,10240,11264,12288,13312,14336,15360,16384,17408};
        for(unsigned i=0;i<32;++i){require(all[i].bits==golden[i]);
            require(all[i].tpi==(i<5?4:i<11?8:i<23?16:32));
            require(all[i].mont==(i<31) && all[i].dev==(i<7));
            require(all[i].mont_tpb==ECM_CUDA_MONT_TPB && all[i].edwards_tpb==128);}
        std::vector<int> expected;std::istringstream input(argc==2?argv[1]:"");std::string item;
        while(std::getline(input,item,','))if(!item.empty())expected.push_back(std::stoi(item));
        require(expected.size()+1==sizeof(selected)/sizeof(*selected));
        for(unsigned i=0;i<expected.size();++i)require(selected[i+1]==expected[i]);
        auto pick=[](unsigned bits,bool mont){for(const auto &t:all)
            if((!mont || t.mont) && (mont?t.bits>=bits+6:t.bits>bits))return t.bits;return 0;};
        require(pick(8191,true)==9216 && pick(8191,false)==8192);
        require(pick(16378,true)==16384 && pick(16379,true)==0);
        require(pick(17407,false)==17408 && pick(17408,false)==0);
        require(sizeof(capacities)/sizeof(*capacities)==7);
        for(int words=1;words<=257;++words){int actual=0;for(int n:capacities)if(words<=n){actual=n;break;}
            int expected_capacity=4;while(expected_capacity<words)expected_capacity*=2;
            require(actual==(words<=256?expected_capacity:0));}
        std::printf("PASS selected=%zu boundaries=8 capacities=257\n",expected.size());return 0;
    }catch(const std::exception &e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
