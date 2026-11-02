#include "../../src/ecm/stage2/cuda/snapshot_pinned.h"
#include <stdexcept>
#include <vector>
#include <iostream>
#include <set>
#include <map>
#include <algorithm>
using Word=unsigned long long;
static void require(bool ok,const char *why){if(!ok)throw std::runtime_error(why);}
static void snapshot_tests() {
    struct Slot : stage2_pinned::SnapshotBuffers {int event_metadata=73;} b;
    std::map<Word*,size_t> live;std::vector<size_t> attempts;
    size_t limit=16384,used=0,peak=0,releases=0;bool fail_red=false;
    auto alloc=[&](Word *&out,size_t n) {
        attempts.push_back(n);if(n>limit || (fail_red && n==1500))return false;
        out=new Word[n];live[out]=n;used+=n;peak=std::max(peak,used);return true;
    };
    auto free=[&](Word *p) {
        auto it=live.find(p);require(it!=live.end(),"snapshot double/free-view");
        used-=it->second;live.erase(it);++releases;delete[] p;
    };
    auto reserve=[&](size_t d,size_t r,bool pool,size_t q=0){return stage2_pinned::reserve_snapshot(b,d,r,pool,q,alloc,free);};
    auto r=reserve(128,64,true);require(r.ready && r.grows==1 && b.pinned_words()==192 && live.size()==1,"pool ownership");
    Word *first=b.storage;const auto count=attempts.size();r=reserve(64,128,true);
    require(r.ready && !r.grows && b.storage==first && b.reduced==first+64 && attempts.size()==count,"pool repack views");
    for(size_t i=0;i<64;++i)b.digits[i]=17;for(size_t i=0;i<128;++i)b.reduced[i]=29;
    for(size_t i=0;i<64;++i)require(b.digits[i]==17,"snapshot views overlap");
    r=reserve(4096,128,true,8192);require(r.ready && peak==8192 && b.capacity==8192,"pool growth doubled peak");
    stage2_pinned::release_snapshot(b,free);require(live.empty(),"pool cleanup");
    limit=4096;r=reserve(3500,1500,true,8192);
    require(r.ready && r.pool_fallbacks==1 && b.separate_fallback && !b.storage && b.pinned_words()==5000 && live.size()==2,"separate exact fallback");
    const auto tried=attempts.size();r=reserve(1000,1000,true,8192);
    require(r.ready && !r.pool_fallbacks && attempts.size()==tried,"pool failure retried");
    stage2_pinned::release_snapshot(b,free);require(live.empty(),"fallback cleanup");
    fail_red=true;r=reserve(3500,1500,false);
    require(!r.ready && b.digits && !b.reduced,"partial failure ownership");
    fail_red=false;r=reserve(500,500,true);require(r.ready && b.storage && live.size()==1,"separate to pool transition");
    Word *old=b.storage;r=reserve(std::numeric_limits<size_t>::max()/8,1,true);
    require(!r.ready && b.storage==old,"snapshot sum overflow");
    stage2_pinned::release_snapshot(b,free);require(live.empty() && !used && releases==6,"snapshot leaked ownership");
    require(b.event_metadata==73,"buffer cleanup lost slot event metadata");
}
int main() {
    try {
        Word *data=nullptr;size_t capacity=0;std::set<Word*> live;std::vector<size_t> sizes;
        bool fail_rounded=false,fail_all=false;size_t releases=0;
        auto alloc=[&](Word *&out,size_t n) {
            sizes.push_back(n);
            // All growth must free the previous allocation first.
            require(live.empty(),"overlapping pinned capacities");
            if(fail_all || (fail_rounded && n%8192==0))return false;
            out=new Word[n];live.insert(out);return true;
        };
        auto free=[&](Word *p){require(live.erase(p)==1,"invalid release");++releases;delete[] p;};
        auto reserve=[&](size_t n,size_t q){return stage2_pinned::reserve(data,capacity,n,q,alloc,free);};
        require(reserve(123,8192) && capacity==8192,"rounded capacity");
        Word *first=data;for(size_t i=0;i<123;++i)data[i]=i+17;
        require(reserve(8192,8192)==first && sizes.size()==1,"reuse allocated tail");
        for(size_t i=0;i<123;++i)require(data[i]==i+17,"reuse changed prefix");
        fail_rounded=true;
        require(reserve(8201,8192) && capacity==8201,"exact allocation fallback");
        require(sizes[sizes.size()-2]==16384 && sizes.back()==8201,"fallback sizes");
        fail_all=true;require(!reserve(20000,8192) && !data && !capacity && live.empty(),"total failure ownership");
        fail_all=fail_rounded=false;require(reserve(33,0) && capacity==33,"exact control");
        const auto allocations=sizes.size();Word *old=data;
        require(!reserve(std::numeric_limits<size_t>::max(),8192) && data==old && capacity==33 && sizes.size()==allocations,
                "byte overflow changed owner");
        require(reserve(34,std::numeric_limits<size_t>::max()) && capacity==34,"quantum overflow");
        free(data);data=nullptr;
        require(live.empty() && releases==4,"leaked pinned allocation");
        snapshot_tests();
        std::cout<<"{\"complete\":true,\"reuse\":true,\"exact_fallback\":true,\"no_double_peak\":true,\"overflow\":true,\"snapshot_views\":true,\"pool_fallback\":true,\"CUDA_calls\":0}\n";
    }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
