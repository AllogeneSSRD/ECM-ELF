/* cgbn_stage1_cuda.h — entry declaration for the native CUDA/CGBN ECM stage 1.

   MPA-OpenCl port. This is the *original* 9-argument cgbn_ecm_stage1 interface
   (Seth Troisi's GMP-ECM CGBN implementation). It intentionally uses a distinct
   include guard from the project-wide include/cgbn_stage1.h (which declares the
   14-argument OpenCL variant) so that the two never collide.

   The CUDA backend glue (src/cuda/ecm_cuda_backend.cu) adapts the driver's
   14-argument backend hook down to this 9-argument entry.
*/

#ifndef _CGBN_STAGE1_CUDA_H
#define _CGBN_STAGE1_CUDA_H 1

#include <stdint.h>
#include <gmp.h>

#ifdef __cplusplus
extern "C" {
#endif

int cgbn_ecm_stage1(mpz_t *factors, int *array_found,
             const mpz_t N, const mpz_t s,
             uint32_t curves, uint64_t *sigma,
             unsigned long checkpoint_interval_ms,
             float *gputime, int verbose, int gpu_param);

#ifdef __cplusplus
}
#endif

#endif /* _CGBN_STAGE1_CUDA_H */
