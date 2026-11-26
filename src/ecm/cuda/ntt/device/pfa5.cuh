#pragma once

#include "../launch.h"
#include "radix5.cuh"
#include "tile_pass.cuh"

namespace ecm_cuda_ntt {

// CRT index n = (1 + 4*256)*column + 256*plane mod 1280.
// After both operands use the same CRT order, their spectra can be
// multiplied in that order. The inverse restores ordinary digit indices.
__global__ __launch_bounds__(128)
void pfa5_convolution(unsigned long long* a, const unsigned long long* b,
    const unsigned long long* fwd, const unsigned long long* inv,
    Radix5Constants fwd5, Radix5Constants inv5, unsigned long long scale) {
    constexpr unsigned width=256;
    constexpr unsigned stride=2048;
    __shared__ unsigned long long left[5][width], right[5][width];
    a+=size_t(blockIdx.x)*stride;
    b+=size_t(blockIdx.x)*stride;
    for(unsigned column=threadIdx.x;column<width;column+=blockDim.x) {
        unsigned long long x[5],y[5];
        const unsigned rotation=(4*column)%5;
#pragma unroll
        for(unsigned plane=0;plane<5;++plane) {
            const unsigned i=column+width*radix5_plane(plane,rotation);
            x[plane]=a[i];
            y[plane]=b[i];
        }
        radix5_dft(x,fwd5);
        radix5_dft(y,fwd5);
#pragma unroll
        for(unsigned plane=0;plane<5;++plane) {
            left[plane][column]=x[plane];
            right[plane][column]=y[plane];
        }
    }
    __syncthreads();
    for(unsigned plane=0;plane<5;++plane) {
        tile_radix4_block(left[plane],width,7,fwd);
        tile_radix4_block(right[plane],width,7,fwd);
    }
    __syncthreads();
    const unsigned i=(threadIdx.x>>5)*64+(threadIdx.x&31);
    for(unsigned plane=0;plane<5;++plane) {
        auto lo=left[plane][i], hi=left[plane][i+32];
        auto rlo=right[plane][i], rhi=right[plane][i+32];
        tile_warp6_pair<false>(lo,hi,fwd);
        tile_warp6_pair<false>(rlo,rhi,fwd);
        lo=gl_mul(gl_mul(lo,rlo),scale);
        hi=gl_mul(gl_mul(hi,rhi),scale);
        tile_warp6_pair<true>(lo,hi,inv);
        left[plane][i]=lo;
        left[plane][i+32]=hi;
    }
    __syncthreads();
    for(unsigned plane=0;plane<5;++plane)
        tile_radix4_block_inv(left[plane],width,6,inv);
    __syncthreads();
    for(unsigned column=threadIdx.x;column<width;column+=blockDim.x) {
        unsigned long long x[5];
#pragma unroll
        for(unsigned plane=0;plane<5;++plane)x[plane]=left[plane][column];
        radix5_dft(x,inv5);
        const unsigned rotation=(4*column)%5;
#pragma unroll
        for(unsigned plane=0;plane<5;++plane)
            a[column+width*radix5_plane(plane,rotation)]=x[plane];
    }
    // Keep the existing 2048-word stride and carry/S4 contracts.
    for(unsigned j=ntt_pfa5_length+threadIdx.x;j<stride;j+=blockDim.x)a[j]=0;
}

} // namespace ecm_cuda_ntt
