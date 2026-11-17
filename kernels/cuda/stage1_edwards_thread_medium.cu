#include "../../src/ecm/stage1/edwards/cuda/thread_kernels.h"

namespace ecm_cuda_edwards_thread {
namespace bits384 {
#define LIMBS 12
#include "stage1_edwards_thread_variant.cuh"
#undef LIMBS
}
namespace bits512 {
#define LIMBS 16
#include "stage1_edwards_thread_variant.cuh"
#undef LIMBS
}

KernelPair medium_kernels(unsigned limbs, bool two_xy) {
    switch (limbs) {
    case 12: return bits384::kernels(two_xy);
    case 16: return bits512::kernels(two_xy);
    default: return {};
    }
}
}
