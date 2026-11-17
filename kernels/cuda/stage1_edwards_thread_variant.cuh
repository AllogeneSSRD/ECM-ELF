// Included in a distinct namespace for each limb tier. Both drivers compile
// the same point, field and SoA kernel bodies using their native frontends.
#define __global
#define EDWARDS_INLINE static __device__ __forceinline__
#define EDWARDS_ASM asm
#define EDWARDS_KERNEL template<bool two_xy> __global__
#define EDWARDS_LANE (size_t(blockIdx.x) * blockDim.x + threadIdx.x)
#define EDWARDS_TWO_XY two_xy
#define EDWARDS_UNROLL 1
#define EDWARDS_FIPS (LIMBS <= 16)
#define EDWARDS_FIPS_SQUARE EDWARDS_FIPS
#define EDWARDS_FIPS_PTX EDWARDS_FIPS

using ulong = unsigned long long;
using uint = uint32_t;
EDWARDS_INLINE uint32_t mul_hi(uint32_t a, uint32_t b) { return __umulhi(a, b); }

#include "../../src/ecm/opencl/resources/edwards/field.clh"
#include "../../src/ecm/opencl/resources/edwards/point.clh"
#include "../../src/ecm/opencl/resources/edwards/stage1.cl"

#undef EDWARDS_FIPS_PTX
#undef EDWARDS_FIPS_SQUARE
#undef EDWARDS_FIPS
#undef EDWARDS_UNROLL
#undef EDWARDS_TWO_XY
#undef EDWARDS_LANE
#undef EDWARDS_KERNEL
#undef EDWARDS_ASM
#undef EDWARDS_INLINE
#undef __global

template<bool two_xy>
static void launch_dictionary(const uint32_t* params, const uint32_t* seeds,
        uint32_t* table, uint32_t* scratch, uint32_t count, uint32_t np0,
        uint32_t first, uint32_t stop, uint32_t weights, unsigned block, cudaStream_t stream) {
    const unsigned grid = (count - 1) / block + 1;
    edwards_dictionary<two_xy><<<grid, block, 0, stream>>>(
        params, seeds, table, scratch, count, np0, first, stop, weights);
}

template<bool two_xy>
static void launch_slice(const uint32_t* params, uint32_t* points, const uint32_t* table,
        const uint32_t* nonzero, const uint32_t* codes, uint32_t count, uint32_t np0,
        uint32_t bits, uint32_t first, uint32_t stop, uint32_t weights, unsigned block, cudaStream_t stream) {
    const unsigned grid = (count - 1) / block + 1;
    edwards_slice<two_xy><<<grid, block, 0, stream>>>(
        params, points, table, nonzero, codes, count, np0, bits, first, stop, weights);
}

template<bool two_xy>
static cudaError_t attributes(cudaFuncAttributes* info) {
    const auto error = cudaFuncGetAttributes(info, edwards_slice<two_xy>);
    if (error != cudaSuccess)
        return error;
    cudaFuncAttributes dictionary{};
    const auto result = cudaFuncGetAttributes(&dictionary, edwards_dictionary<two_xy>);
    if (dictionary.maxThreadsPerBlock < info->maxThreadsPerBlock)
        info->maxThreadsPerBlock = dictionary.maxThreadsPerBlock;
    return result;
}

static KernelPair kernels(bool two_xy) {
    if (two_xy)
        return {launch_dictionary<true>, launch_slice<true>, attributes<true>};
    return {launch_dictionary<false>, launch_slice<false>, attributes<false>};
}
