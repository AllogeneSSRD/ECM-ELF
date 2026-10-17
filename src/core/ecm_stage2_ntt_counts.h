#pragma once
#include <array>
#include <cstdint>

namespace ecm_stage2 {
// One host thread owns a curve. Counts are mathematical convolutions, not
// launches: each batch member performs two forwards and one inverse.
struct NttLengthCounts {
    struct Entry { uint64_t convolutions=0,chunks=0; };
    bool enabled=false;
    std::array<Entry,33> entries{};
    void add(uint64_t length,uint64_t convolutions,uint64_t chunks) {
        if(!enabled)return;
        unsigned k=0;for(auto n=length;n>1;n>>=1)++k;
        if(k<entries.size()){entries[k].convolutions+=convolutions;entries[k].chunks+=chunks;}
    }
};
}
