#pragma once
#include "launch.h"

namespace ecm_stage2::baby_wheel {
using Word = unsigned long long;
struct Segment {
    unsigned walk_begin, seed, span;
};
inline constexpr unsigned skip = ~0u;
}

template<int NW>
void ecm_launch_s2g_baby_wheel_scalar(EcmCudaLaunch cfg, int nw,
    const unsigned long long* n, unsigned long long ninv, const unsigned long long* one,
    const ecm_stage2::baby_wheel::Segment* segments, unsigned count, const unsigned* slots,
    const unsigned long long* sx, const unsigned long long* sz,
    unsigned long long* out_x, unsigned long long* out_z, unsigned long long* guard);
void ecm_launch_s2g_baby_wheel_coop(EcmCudaLaunch cfg, int nw,
    const unsigned long long* n, unsigned long long ninv, const unsigned long long* one,
    const ecm_stage2::baby_wheel::Segment* segments, unsigned count, const unsigned* slots,
    const unsigned long long* sx, const unsigned long long* sz, int mersenne_bits,
    unsigned long long* out_x, unsigned long long* out_z, unsigned long long* guard);
void ecm_launch_s2g_baby_wheel_scatter(EcmCudaLaunch cfg, int nw, const unsigned* slots, size_t words,
    const unsigned long long* x, const unsigned long long* z,
    unsigned long long* out_x, unsigned long long* out_z);
