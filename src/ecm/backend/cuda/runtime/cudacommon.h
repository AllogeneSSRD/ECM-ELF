/* cudacommon.h — slim CUDA helper header for the MPA-OpenCl CUDA/CGBN backend.

   This is a trimmed replacement for GMP-ECM's cudacommon.h. It provides only
   what src/cuda/stage1/montgomery.cu actually needs:
     - CUDA_CHECK / cuda_check          (error-checking wrapper, used pervasively)
     - ecm_cuda_print_ptx_version       (runtime PTX/SM version report; CUDA glue)

   Device enumeration / selection (GMP-ECM's select_and_init_GPU / get_device_prop)
   is intentionally omitted: the ecm_cuda driver performs device selection through
   the backend prepare hook (src/cuda/backend.cpp).
*/

#ifndef _CUDACOMMON_SLIM_H
#define _CUDACOMMON_SLIM_H 1

#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#ifdef __cplusplus
/* C++ / CUDA only */

#define CUDA_CHECK(action) cuda_check(action, #action, __FILE__, __LINE__)

inline void cuda_check(cudaError_t status, const char *action = NULL,
                       const char *file = NULL, int32_t line = 0) {
  if (status != cudaSuccess) {
    fprintf(stderr, "CUDA error (%d) occurred: %s\n", status,
            cudaGetErrorString(status));
    if (action != NULL)
      fprintf(stderr, "While running %s   (file %s, line %d)\n", action, file,
              line);
    exit(EXIT_FAILURE);
  }
}

/* Defined in src/cuda/backend.cpp. Queries the given compiled kernel and
   prints, at runtime, the PTX ISA version it was compiled to (and the SM/cubin
   binary version). `func` must be an instantiated __global__ kernel pointer. */
void ecm_cuda_print_ptx_version(const void *func);

#endif /* __cplusplus */

#endif /* _CUDACOMMON_SLIM_H */
