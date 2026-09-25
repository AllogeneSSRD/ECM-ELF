/* opencl_backend_glue.cpp — OpenCL implementation of the ECM backend seam.

   Linked only into the `ecm` executable. Each hook forwards to the pre-existing
   OpenCL host functions, so the shared driver behaves exactly as before.
*/

#include "ecm_backend.h"
#include "ecm.h"                    /* ECM_ERROR */

#include "cgbn_stage1.h"            /* gpu_prepare_opencl */
#include "cl_probe.h"              /* configureOpenclDeviceIndex */
#include "opencl_ecm_entry.h"       /* opencl_ecm_stage1 */
#include "opencl_ecm_path_registry.h" /* opencl_ecm_print_available_kernels */
#include "opencl_ecm_log.h"         /* ecm_ts_fprintf */

extern "C" const char *ecm_backend_name(void) {
    return "OpenCL";
}

extern "C" void ecm_backend_print_kernels(FILE *out) {
    opencl_ecm_print_available_kernels(out);
}

extern "C" int ecm_backend_prepare(size_t n_log2, int verbose, int device_index,
                                   const char *gpu_mul_path, const char *gpu_sqr_path,
                                   const char *gpu_add_path, const char *gpu_sub_path,
                                   const char *gpu_special_mult_path) {
    if (!configureOpenclDeviceIndex(device_index, true)) {
        return 1;
    }
    return gpu_prepare_opencl(n_log2, verbose, gpu_mul_path, gpu_sqr_path,
                              gpu_add_path, gpu_sub_path, gpu_special_mult_path);
}

extern "C" int ecm_backend_stage1(mpz_t *factors, int *array_found,
                                  const mpz_t N, const mpz_t s,
                                  uint32_t curves, uint64_t *sigma,
                                  unsigned long checkpoint_interval_ms,
                                  float *gputime, int verbose, int gpu_param,
                                  const char *gpu_mul_path, const char *gpu_sqr_path,
                                  const char *gpu_add_path, const char *gpu_sub_path,
                                  const char *gpu_special_mult_path) {
    /* gpu_param = 0 (Suyama param0) and 2 (param2 batch-2 / 6-torsion) exist for the
       CUDA/CGBN kernels only: the OpenCL .cl kernels still bake in the batch
       parametrization's fixed shape (P = (2:1), difference x = 2, a24 = the 32-bit d).
       Refuse loudly instead of silently running a different curve family than the user
       asked for -- docs §20.5. */
    if (gpu_param == 0 || gpu_param == 2) {
        ecm_ts_fprintf(stderr,
                       "ERROR: gpu_param = %d (%s) is not implemented for the OpenCL backend.\n"
                       "       Use gpu_param = 3 here, or the ecm_cuda build for param0/param2.\n",
                       gpu_param, gpu_param == 0 ? "Suyama param0" : "param2 batch-2");
        return ECM_ERROR;
    }
    /* The batch parametrization carries d = sigma/2^32 as a 32-bit kernel value. */
    if (sigma == nullptr || *sigma + (uint64_t)curves > 0x100000000ull) {
        ecm_ts_fprintf(stderr,
                       "ERROR: the OpenCL batch path needs sigma + curves <= 2^32.\n");
        return ECM_ERROR;
    }
    uint32_t sigma32 = (uint32_t)*sigma;
    const int rc = opencl_ecm_stage1(factors, array_found, N, s, curves, &sigma32,
                                     checkpoint_interval_ms, gputime, verbose,
                                     gpu_mul_path, gpu_sqr_path, gpu_add_path,
                                     gpu_sub_path, gpu_special_mult_path);
    *sigma = sigma32;               /* the backend may override it from a checkpoint */
    return rc;
}
