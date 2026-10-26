#pragma once
#define ECM_STAGE2_GIANT_CHUNK_PLAN 1
#include <algorithm>
#include <cstdint>
#include <limits>

// A point chunk contains whole G-tree batches. The budget covers X/Z payloads
// only; seeds, segment products and NTT/owner lifetimes must be charged separately.
namespace stage2_giant_chunk {
struct Plan {
    bool valid=false, minimum_over_budget=false;
    uint64_t points=0, coordinate_bytes=0;
};
inline Plan plan(uint64_t P,uint64_t nw,uint64_t budget,bool floor) {
    Plan out;
    const auto maximum=std::numeric_limits<uint64_t>::max();
    if(!P || !nw || !budget || nw>maximum/16)return out;
    const uint64_t per_point=16*nw;
    const uint64_t k=budget/per_point;
    const uint64_t batches=std::max(uint64_t(1),k/P+uint64_t(!floor && k%P!=0));
    if(batches>maximum/P)return out;
    out.points=P*batches;
    if(out.points>maximum/per_point)return out;
    out.coordinate_bytes=out.points*per_point;
    out.minimum_over_budget=P>budget/per_point;
    out.valid=true;
    return out;
}
}
