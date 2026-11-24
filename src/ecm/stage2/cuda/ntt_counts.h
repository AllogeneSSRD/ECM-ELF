#pragma once
#include <array>
#include <cstdint>

namespace ecm_stage2 {
// One host thread owns a curve. Count logical transforms and omit reused
// forward spectra. Several batch members can share the same launches.
struct NttLengthCounts {
    struct Entry { uint64_t convolutions=0,chunks=0,reused_forwards=0; };
    bool enabled=false;
    std::array<Entry,33> entries{};
    void add(uint64_t length,uint64_t convolutions,uint64_t chunks,uint64_t reused_forwards) {
        if(!enabled)return;
        unsigned k=0;for(auto n=length;n>1;n>>=1)++k;
        if(k<entries.size()) {
            entries[k].convolutions+=convolutions;
            entries[k].chunks+=chunks;
            entries[k].reused_forwards+=reused_forwards;
        }
    }
};
}
