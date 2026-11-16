#pragma once
#include "../../src/ecm/stage2/cuda/stage2_baby_wheel_launch.h"
#if !defined(ECM_STAGE2_BABY_WHEEL_COOP_ONLY) && !defined(ECM_STAGE2_BABY_WHEEL_SCALAR_ONLY)
#include "../../src/ecm/stage2/cuda/device/point.cuh"
#endif
#ifndef ECM_STAGE2_BABY_WHEEL_SCALAR_ONLY
#include "../../src/ecm/stage2/cuda/device/point_coop_field.cuh"
#endif

namespace ecm_stage2::baby_wheel {

#ifndef ECM_STAGE2_BABY_WHEEL_COOP_ONLY
// Each walked position is inspected, including holes in the coprime indices.
// Only selected positions scatter to the regular sorted baby-point slots.
template<int NW>
__global__ void scalar_chain(const Word* n, Word ninv, int nw, const Word* one,
    const Segment* segments, unsigned count, const unsigned* slots, const Word* sx, const Word* sz,
    Word* out_x, Word* out_z, Word* guard) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const auto seg = segments[i];
    Word x0[NW], z0[NW], x1[NW], z1[NW], product[NW];
    Word *prev_x = x0, *prev_z = z0, *cur_x = x1, *cur_z = z1;
    for (int k = 0; k < nw; ++k) {
        x1[k] = sx[size_t(seg.seed) * nw + k];
        z1[k] = sz[size_t(seg.seed) * nw + k];
        product[k] = one[k];
    }
    for (unsigned j = 0; j < seg.span; ++j) {
        if (j == 1) {
            for (int k = 0; k < nw; ++k) {
                prev_x[k] = sx[size_t(seg.seed + 1) * nw + k];
                prev_z[k] = sz[size_t(seg.seed + 1) * nw + k];
            }
        } else if (j >= 2) {
            s2g_xadd<NW, true>(prev_x, prev_z, cur_x, cur_z, sx, sz, prev_x, prev_z, n, ninv, nw);
        }
        if (j) {
            auto* tmp = cur_x; cur_x = prev_x; prev_x = tmp;
            tmp = cur_z; cur_z = prev_z; prev_z = tmp;
        }
        s2g_mont_mul<NW>(product, product, cur_z, n, ninv, nw);
        const unsigned slot = slots[seg.walk_begin + j];
        if (slot != skip) {
            for (int k = 0; k < nw; ++k) {
                out_x[size_t(slot) * nw + k] = cur_x[k];
                out_z[size_t(slot) * nw + k] = cur_z[k];
            }
        }
    }
    for (int k = 0; k < nw; ++k) guard[size_t(i) * nw + k] = product[k];
}

template<class Slot>
__global__ void scatter_replayed(const Slot* slots, size_t words, int nw,
    const Word* x, const Word* z, Word* out_x, Word* out_z) {
    const size_t word = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (word >= words) return;
    const size_t dst = size_t(slots[word / nw]) * nw + word % nw;
    out_x[dst] = x[word];
    out_z[dst] = z[word];
}
#endif

#ifndef ECM_STAGE2_BABY_WHEEL_SCALAR_ONLY
template<int NW, int TPI>
__global__ void coop_chain(const Word* n, Word ninv, int nw, const Word* one,
    const Segment* segments, unsigned count, const unsigned* slots, const Word* sx, const Word* sz,
    int mersenne_bits, Word* out_x, Word* out_z, Word* guard) {
    const unsigned i = (blockIdx.x * blockDim.x + threadIdx.x) / TPI;
    if (i >= count) return;
    const auto seg = segments[i];
    using Ctx = cgbn_context_t<TPI>;
    using Env = cgbn_env_t<Ctx, 64 * NW>;
    using Bn = typename Env::cgbn_t;
    Ctx ctx(cgbn_no_checks);
    Env e(ctx);
    Bn modulus, dx, dz, prev_x, prev_z, cur_x, cur_z, product;
    stage2_coop_load<Env>(modulus, n, nw);
    stage2_coop_load<Env>(dx, sx, nw);
    stage2_coop_load<Env>(dz, sz, nw);
    stage2_coop_load<Env>(cur_x, sx + size_t(seg.seed) * nw, nw);
    stage2_coop_load<Env>(cur_z, sz + size_t(seg.seed) * nw, nw);
    stage2_coop_load<Env>(product, one, nw);
    const Stage2PaddedCoopField<Env> f{e, modulus, uint32_t(ninv), mersenne_bits, 64 * nw};
    for (unsigned j = 0; j < seg.span; ++j) {
        if (j == 1) {
            stage2_coop_load<Env>(prev_x, sx + size_t(seg.seed + 1) * nw, nw);
            stage2_coop_load<Env>(prev_z, sz + size_t(seg.seed + 1) * nw, nw);
        } else if (j >= 2) {
            f.xadd(prev_x, prev_z, cur_x, cur_z, dx, dz, prev_x, prev_z);
        }
        if (j) {
            cgbn_swap(e, cur_x, prev_x);
            cgbn_swap(e, cur_z, prev_z);
        }
        f.mul(product, product, cur_z);
        const unsigned slot = slots[seg.walk_begin + j];
        if (slot != skip) {
            stage2_coop_store<Env>(out_x + size_t(slot) * nw, cur_x, nw);
            stage2_coop_store<Env>(out_z + size_t(slot) * nw, cur_z, nw);
        }
    }
    stage2_coop_store<Env>(guard + size_t(i) * nw, product, nw);
}
#endif

} // namespace ecm_stage2::baby_wheel
