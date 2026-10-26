#include "launch.h"
#ifdef ECM_CUDA_ISOLATED_DEVICE
#include <cstring>
#include <map>
#include <mutex>
namespace {
struct Constants {S2GDeviceConstants value{}; unsigned long long revision=1;};
std::mutex constants_mutex;
std::map<int,Constants> constants;
}
cudaError_t ecm_stage2_copy_constants(const void *source,size_t bytes,size_t offset,cudaMemcpyKind kind) {
    if (!source || offset>sizeof(S2GDeviceConstants) || bytes>sizeof(S2GDeviceConstants)-offset)
        return cudaErrorInvalidValue;
    if (kind!=cudaMemcpyHostToDevice) return cudaErrorInvalidMemcpyDirection;
    int device=0;const auto error=cudaGetDevice(&device);if(error!=cudaSuccess)return error;
    std::lock_guard<std::mutex> guard(constants_mutex);
    auto &state=constants[device];
    std::memcpy(reinterpret_cast<unsigned char*>(&state.value)+offset,source,bytes);
    ++state.revision;return cudaSuccess;
}
cudaError_t ecm_stage2_constant_snapshot(S2GDeviceConstants *out,unsigned long long *revision,int *device) {
    const auto error=cudaGetDevice(device);if(error!=cudaSuccess)return error;
    std::lock_guard<std::mutex> guard(constants_mutex);
    const auto &state=constants[*device];*out=state.value;*revision=state.revision;
    return cudaSuccess;
}
#endif
