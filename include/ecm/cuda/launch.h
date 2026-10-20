#pragma once
#include <cuda_runtime.h>

// Launch geometry and borrowed stream only. The caller retains error checks,
// synchronization, timings and ownership of every argument buffer.
struct EcmCudaLaunch {
    dim3 grid, block;
    size_t shared = 0;
    cudaStream_t stream = nullptr;
    EcmCudaLaunch(dim3 grid_, dim3 block_, size_t shared_ = 0,
                  cudaStream_t stream_ = nullptr)
        : grid(grid_), block(block_), shared(shared_), stream(stream_) {}
};
