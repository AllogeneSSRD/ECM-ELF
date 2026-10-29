#pragma once
// The save/queue driver is shared. Engine implementations are selected at link
// time, so the native OpenCL worker has no CUDA runtime dependency.
namespace stage2_backend {
#ifdef ECM_STAGE2_OPENCL
inline constexpr bool opencl=true;
inline constexpr const char *name="opencl-ntt";
inline constexpr const char *executable="ecm_opencl_stage2";
#else
inline constexpr bool opencl=false;
inline constexpr const char *name="cuda-ntt";
inline constexpr const char *executable="ecm_cuda_stage2";
#endif
}
