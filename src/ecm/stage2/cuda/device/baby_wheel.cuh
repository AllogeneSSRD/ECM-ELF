#pragma once

// Arithmetic modules have already included their own point constants.
#define ECM_STAGE2_BABY_WHEEL_SCALAR_ONLY
#include "../../../../../kernels/cuda/stage2_baby_wheel.cuh"
#undef ECM_STAGE2_BABY_WHEEL_SCALAR_ONLY

template<int NW>
void ecm_launch_s2g_baby_wheel_scalar(EcmCudaLaunch cfg, int nw,
    const unsigned long long* n, unsigned long long ninv, const unsigned long long* one,
    const ecm_stage2::baby_wheel::Segment* segments, unsigned count, const unsigned* slots,
    const unsigned long long* sx, const unsigned long long* sz,
    unsigned long long* out_x, unsigned long long* out_z, unsigned long long* guard) {
    using ecm_stage2::baby_wheel::scalar_chain;
    scalar_chain<NW><<<cfg.grid, cfg.block, cfg.shared, cfg.stream>>>(
        n, ninv, nw, one, segments, count, slots, sx, sz, out_x, out_z, guard);
}

void ecm_launch_s2g_baby_wheel_scatter(EcmCudaLaunch cfg, int nw, const unsigned* slots, size_t words,
    const unsigned long long* x, const unsigned long long* z,
    unsigned long long* out_x, unsigned long long* out_z) {
    using ecm_stage2::baby_wheel::scatter_replayed;
    scatter_replayed<unsigned><<<cfg.grid, cfg.block, cfg.shared, cfg.stream>>>(slots, words, nw, x, z, out_x, out_z);
}
