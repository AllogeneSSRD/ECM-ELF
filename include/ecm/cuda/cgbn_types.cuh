#pragma once
#include <cstdint>
#include <cuda_runtime.h>
// Upstream CGBN's host parser rejects its non-GMP fallback. This is a compiler
// compatibility include; project kernels and launch arguments contain no GMP API.
#if !defined(__CUDA_ARCH__)
#include <gmp.h>
#endif
#include <cgbn/cgbn.h>
