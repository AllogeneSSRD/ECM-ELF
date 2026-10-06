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
     B1/torsion: integer lcm bound and multiplier (1 or 12), used by CUDA PRAC;
                 OpenCL ignores these and continues to consume s.
     gpu_param : curve parametrization, 3 = gmp-ecm batch (historical GPU path) or
                 0 = Suyama param0 (Prime95 sigma_type=1 / gmp-ecm -param 0, the
                 same curves the CPU path runs).  0 is CUDA-only for now: the
                 OpenCL backend rejects it instead of silently using param3. */
int ecm_backend_stage1(mpz_t *factors, int *array_found,
                       const mpz_t N, const mpz_t s,
                       uint32_t curves, uint64_t *sigma,
                       unsigned long checkpoint_interval_ms,
                       float *gputime, int verbose, int gpu_param,
                       uint64_t B1, uint32_t torsion,
                       const char *gpu_mul_path, const char *gpu_sqr_path,
                       const char *gpu_add_path, const char *gpu_sub_path,
                       const char *gpu_special_mult_path);

/* --gpu-info: report what the linked backend would actually do, without running a
   single curve and without writing a file. The driver prints the result as plain
   `key=value` lines (see ecm_driver.cpp print_gpu_info). */

/* Largest number of tiers any backend may report; the array below is sized with it
   so the glue never allocates. */
#define ECM_BACKEND_MAX_TIERS 64

typedef struct {
    uint32_t bits;          /* CGBN container size this tier computes in */
    uint32_t tpb;           /* threads per block launched with */
    uint32_t tpi;           /* threads per curve */
    uint32_t ipb;           /* tpb / tpi: curves per block */
    uint32_t blocks_per_sm; /* register-allowed resident blocks per SM (0 = unknown) */
    uint32_t blocks_min;    /* one block per SM */
    uint64_t curves_min;    /* blocks_min * ipb: the "some SMs idle" threshold */
    uint32_t blocks_wave;   /* register-allowed block slots; a multiple avoids a partial wave */
    uint64_t curves_wave;   /* blocks_wave * ipb: the gpucurves the kernel recommends */
    uint32_t fold_blocks_min; /* >= 2 resident blocks/SM when the fold kernels are used */
    uint64_t fold_curves_min;
} ecm_backend_tier;

typedef struct {
    int      device_index;
    int      sm_count;
    int      cc_major;
    int      cc_minor;
    int      fold;          /* 1 = this binary carries the Mersenne-fold kernels */
    int      gpu_param;     /* the parametrization the tiers were resolved for */
    uint32_t carry_bits;    /* container overhead the kernel adds to the bit size */
    int      picked;        /* 1 = only the tier the kernel would pick is reported */
    char     name[128];
    int      tier_count;
    ecm_backend_tier tiers[ECM_BACKEND_MAX_TIERS];
} ecm_backend_gpu_info;

/* Fill `*out` with the device the backend would select and the tiers it carries.
     device_index : 0-based, as given by -d (default 0)
     gpu_param    : 0 = Suyama param0, 2 = param2, 3 = gmp-ecm batch (param3)
     want_bits    : 0 = every tier; otherwise only the tier the kernel would pick
                    for an N of that many bits
   Returns ECM_BACKEND_QUERY_OK on success, or ECM_BACKEND_QUERY_FAILED with a
   one-line reason in `*err` (set on every path, never left NULL, so the driver can
   always print something useful). A backend that has no such concept (OpenCL)
   returns ECM_BACKEND_QUERY_NOT_APPLICABLE and says so in *err -- the driver prints
   that and still exits 0, because asking is not itself an error. */
int ecm_backend_query_gpu(int device_index, int gpu_param, uint32_t want_bits,
                          ecm_backend_gpu_info *out, const char **err);

#define ECM_BACKEND_QUERY_OK           0
#define ECM_BACKEND_QUERY_FAILED       1
#define ECM_BACKEND_QUERY_NOT_APPLICABLE 2

#ifdef __cplusplus
}
#endif

#endif /* ECM_BACKEND_H */
