#include "edwards_runtime.inl"

void ecm_edwards_cuda_detail::prepare_device(int device) {
    check(cudaSetDevice(device));
    unsigned flags=0;check(cudaGetDeviceFlags(&flags));
    check(cudaSetDeviceFlags((flags&~unsigned(cudaDeviceScheduleMask))|cudaDeviceScheduleBlockingSync));
}
