#pragma once

// Older NVCC uses a per-isolated-TU register limit supplied by the builder.
#if defined(__CUDACC__) && (__CUDACC_VER_MAJOR__ < 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ < 6))
#define __maxnreg__(count)
#endif
#include <cstdint>
#include <cuda_runtime.h>
// Upstream CGBN's host parser rejects its non-GMP fallback. This is a compiler
// compatibility include; project kernels and launch arguments contain no GMP API.
#if !defined(__CUDA_ARCH__)
#include <gmp.h>
#endif
#include <cgbn/cgbn.h>
