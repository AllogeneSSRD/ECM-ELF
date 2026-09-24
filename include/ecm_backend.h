/* ecm_backend.h — GPU backend seam for the ECM driver.

   The single ecm driver (src/core/ecm_driver.cpp) talks to a GPU backend only
   through these three hooks. Each backend provides its own implementation in a
   small glue translation unit:

     - OpenCL: src/opencl_backend_glue.cpp   (linked into the `ecm` target)
     - CUDA:   src/cuda/ecm_cuda_backend.cu  (linked into the `ecm_cuda` target)

   This lets both executables share the exact same driver, argument parsing,
   checkpoint/save logic and logging, while swapping the GPU implementation at
   link time.
*/

#ifndef ECM_BACKEND_H
#define ECM_BACKEND_H 1

#include <stdint.h>
#include <stdio.h>
#include <gmp.h>

#ifdef __cplusplus
extern "C" {
#endif

/* --showkernel: print the operators/kernels the active backend supports. */
void ecm_backend_print_kernels(FILE *out);

/* Name of the GPU backend linked into THIS executable: "OpenCL" (ecm.exe) or
   "CUDA/CGBN" (ecm_cuda.exe).  Never NULL.

   The driver is shared by both executables, so any banner/help text that names the
   GPU implementation must ask here instead of hardcoding "OpenCL" -- the CUDA build
   used to announce itself as "gpu (OpenCL)" and list "OpenCL kernel paths". */
const char *ecm_backend_name(void);

/* Select the GPU device and prepare the backend for an N of n_log2 bits.
     device_index          : user -d value (0-based); default 0.
     gpu_*_path             : OpenCL operator overrides (ignored by CUDA).
   Returns 0 on success, non-zero on failure. */
int ecm_backend_prepare(size_t n_log2, int verbose, int device_index,
                        const char *gpu_mul_path, const char *gpu_sqr_path,
                        const char *gpu_add_path, const char *gpu_sub_path,
                        const char *gpu_special_mult_path);

/* Run ECM stage 1 on the prepared device. The CUDA backend ignores the gpu_*_path
   arguments.
     sigma     : in/out, the first curve's sigma.  64-bit because Suyama param0
                 carries a full sigma (the CPU path uses the same 53-bit random
                 generator); the batch case (gpu_param = 3) requires sigma + curves
                 <= 2^32 and rejects larger values.
     gpu_param : curve parametrization, 3 = gmp-ecm batch (historical GPU path) or
                 0 = Suyama param0 (Prime95 sigma_type=1 / gmp-ecm -param 0, the
                 same curves the CPU path runs).  0 is CUDA-only for now: the
                 OpenCL backend rejects it instead of silently using param3. */
int ecm_backend_stage1(mpz_t *factors, int *array_found,
                       const mpz_t N, const mpz_t s,
                       uint32_t curves, uint64_t *sigma,
                       unsigned long checkpoint_interval_ms,
                       float *gputime, int verbose, int gpu_param,
                       const char *gpu_mul_path, const char *gpu_sqr_path,
                       const char *gpu_add_path, const char *gpu_sub_path,
                       const char *gpu_special_mult_path);

#ifdef __cplusplus
}
#endif

#endif /* ECM_BACKEND_H */
