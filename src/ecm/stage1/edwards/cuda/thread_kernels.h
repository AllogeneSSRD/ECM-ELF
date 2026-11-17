#pragma once
#include <cuda_runtime_api.h>
#include <cstdint>

namespace ecm_cuda_edwards_thread {
struct KernelPair {
    void (*dictionary)(const uint32_t*, const uint32_t*, uint32_t*, uint32_t*,
                       uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, unsigned, cudaStream_t);
    void (*slice)(const uint32_t*, uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*,
                  uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, unsigned, cudaStream_t);
    cudaError_t (*attributes)(cudaFuncAttributes*);
};

KernelPair small_kernels(unsigned limbs, bool two_xy);
KernelPair medium_kernels(unsigned limbs, bool two_xy);
KernelPair wide_kernels(unsigned limbs, bool two_xy);
}
