#include "../../src/ecm/stage1/edwards/cuda/thread_kernels.h"

namespace ecm_cuda_edwards_thread {
namespace bits768 {
#define LIMBS 24
#include "stage1_edwards_thread_variant.cuh"
#undef LIMBS
}
namespace bits1024 {
#define LIMBS 32
#include "stage1_edwards_thread_variant.cuh"
#undef LIMBS
}

KernelPair wide_kernels(unsigned limbs, bool two_xy) {
    switch (limbs) {
    case 24: return bits768::kernels(two_xy);
    case 32: return bits1024::kernels(two_xy);
    default: return {};
    }
}
}
