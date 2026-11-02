#pragma once
#include <cstddef>
#include <limits>

namespace stage2_pinned {
// Host capacity policy only. Consumers always copy/check the requested prefix.
// Free before growth keeps the old+new allocations from doubling the peak.
template<class Allocate,class Release>
unsigned long long *reserve(unsigned long long *&data,std::size_t &capacity,
    std::size_t words,std::size_t quantum,Allocate allocate,Release release) {
    const auto maximum=std::numeric_limits<std::size_t>::max()/sizeof(unsigned long long);
    if(words>maximum)return nullptr;
    if(data && capacity>=words)return data;
    auto rounded=words;
    if(quantum>1 && quantum-1<=maximum && words<=maximum-(quantum-1))
        rounded=(words+quantum-1)/quantum*quantum;
    if(data){release(data);data=nullptr;}
    capacity=0;
    unsigned long long *next=nullptr;
    if(!allocate(next,rounded)) {
        // Rounding is optional: scarce pinned RAM must retain the exact-size
        // allocation opportunity of the original path.
        next=nullptr;
        if(rounded==words || !allocate(next,words))return nullptr;
        rounded=words;
    }
    data=next;capacity=rounded;return data;
}
}
