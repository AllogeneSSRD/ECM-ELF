#include "edwards_runtime.inl"

size_t ecm_edwards_cuda_detail::prepare_device(int device) {
    check(cudaSetDevice(device));
    unsigned flags=0;check(cudaGetDeviceFlags(&flags));
    check(cudaSetDeviceFlags((flags&~unsigned(cudaDeviceScheduleMask))|cudaDeviceScheduleBlockingSync));
    size_t free = 0, total = 0;
    check(cudaMemGetInfo(&free, &total));
    return free;
}
