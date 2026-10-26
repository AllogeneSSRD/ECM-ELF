#pragma once
#include "../../../curve/edwards.h"
#include <vector>
namespace ed_gpu_driver {
using namespace ecm_edwards;
struct Batch {
    uint32_t count,done=0;
    unsigned window=6;
    int doubling=4;
    Array state,factors;
    std::vector<uint64_t> sigmas;
    std::vector<int> hit;
    explicit Batch(uint32_t c):count(c),state(size_t(c)*9),factors(c),sigmas(c),hit(c,0){}
};
}
