#pragma once
#include "pinned_reserve.h"

namespace stage2_pinned {
struct SnapshotBuffers {
    unsigned long long *storage=nullptr,*digits=nullptr,*reduced=nullptr;
    std::size_t capacity=0,digcap=0,redcap=0;
    bool separate_fallback=false;
    std::size_t pinned_words()const {return storage?capacity:digcap+redcap;}
};
struct SnapshotReserve {bool ready=false;unsigned grows=0,pool_fallbacks=0;};
template<class Release> void release_snapshot(SnapshotBuffers &b,Release release) {
    if(b.storage)release(b.storage);
    else {if(b.digits)release(b.digits);if(b.reduced)release(b.reduced);}
    b=SnapshotBuffers{};
}
// The ring owner must consume the previous snapshot before changing its views.
template<class Allocate,class Release>
SnapshotReserve reserve_snapshot(SnapshotBuffers &b,std::size_t digits,std::size_t reduced,
    bool pool,std::size_t quantum,Allocate allocate,Release release) {
    SnapshotReserve result;
    const auto maximum=std::numeric_limits<std::size_t>::max()/sizeof(unsigned long long);
    if(digits>maximum || reduced>maximum-digits)return result;
    if(pool && !b.separate_fallback) {
        if(!b.storage && (b.digits || b.reduced))release_snapshot(b,release);
        const auto old=b.capacity;
        auto *p=reserve(b.storage,b.capacity,digits+reduced,quantum,allocate,release);
        b.digits=b.reduced=nullptr;b.digcap=b.redcap=0;
        if(p) {
            b.digits=p;b.reduced=p+digits;
            b.digcap=digits;b.redcap=reduced;
            result.ready=true;result.grows=b.capacity>old;return result;
        }
        // A larger contiguous allocation may fail while two smaller ones fit.
        // Stop retrying the pool for this slot and preserve the exact control.
        b.separate_fallback=true;result.pool_fallbacks=1;
    } else if(b.storage)release_snapshot(b,release);
    const auto old_dig=b.digcap,old_red=b.redcap;
    const auto q=b.separate_fallback?1:quantum;
    auto *d=reserve(b.digits,b.digcap,digits,q,allocate,release);
    auto *r=reserve(b.reduced,b.redcap,reduced,q,allocate,release);
    result.ready=d && r;result.grows=(b.digcap>old_dig)+(b.redcap>old_red);return result;
}
}
