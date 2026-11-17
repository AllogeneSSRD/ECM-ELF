#include "../../src/ecm/stage1/edwards/cuda/thread_kernels.h"

namespace ecm_cuda_edwards_thread {
namespace bits128 {
#define LIMBS 4
#include "stage1_edwards_thread_variant.cuh"
#undef LIMBS
}
namespace bits192 {
#define LIMBS 6
#include "stage1_edwards_thread_variant.cuh"
#undef LIMBS
}
namespace bits256 {
#define LIMBS 8
#include "stage1_edwards_thread_variant.cuh"
#undef LIMBS
}

KernelPair small_kernels(unsigned limbs, bool two_xy) {
    switch (limbs) {
    case 4: return bits128::kernels(two_xy);
    case 6: return bits192::kernels(two_xy);
    case 8: return bits256::kernels(two_xy);
    default: return {};
    }
}
}
