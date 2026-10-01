/* cgbn_stage2_cuda.h — entry declaration for the native CUDA/CGBN ECM stage 2.

   M1 of docs/DEV_STAGE2_GPU_PLAN.md.  The interface is deliberately the same SHAPE as
   the stage-1 entry (kernels/cuda/cgbn_stage1_cuda.h), so that the driver can later
   reach stage 2 through the backend seam (include/ecm_backend.h) exactly the way it
   already reaches stage 1:

     - what to factor (N) and how many curves,
     - where the stage-1 result comes from (a save file's affine x per curve, or the
       stage-1 ladder computed on the device when no x is supplied),
     - B1/B2/D and how many pairing instances per curve,
     - factors out, GPU time out.

   The algorithm is the CPU reference's pairing/BSGS stage 2
   (tools/bench/stage2_ref.cpp, --algorithm pairing), which is the correctness oracle:
   identical candidate set, identical hit test.  See cgbn_stage2_kernel.h for the
   algorithm, the domain conventions and why the tables are built with chains.
*/

#ifndef _CGBN_STAGE2_CUDA_H
#define _CGBN_STAGE2_CUDA_H 1

#include <stdint.h>
#include <gmp.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint64_t b1;            /* stage-1 bound; primes <= b1 are the stage-1 primes */
    uint64_t b2;            /* stage-2 bound; the candidates are (b1, b2] */
    uint64_t d;             /* baby-step / giant-step span (D), D >= 2 */
    uint32_t segs;          /* pairing instances per curve (0 -> 1): more ILP, same tables */
    int      device_index;  /* < 0 -> device 0 */
    uint32_t torsion;       /* 0 -> 1 (gmp-ecm -param 0 convention, same as the reference) */
} ecm_stage2_opts;

/* Runs stage 2 on `curves` curves and fills factors[]/array_found[] with up to
   max_factors distinct factors (array_found[i] = 1 for each one written).

   x_hex: curves entries; x_hex[i] is the affine x of the stage-1 result in hex (the
   `X=0x...` field of a save file), or NULL/empty to run the stage-1 ladder on the
   device instead.  May itself be NULL, which means "compute stage 1 for every curve".

   Returns 0 on success, a nonzero ECM_* style code on failure.  *gputime (optional)
   receives the GPU time in seconds; *stage1_done (optional) the number of curves whose
   stage-1 Z was itself divisible by a factor of N (the reference skips those curves).

   Caller must have selected the device (cudaSetDevice) or leave opts->device_index >= 0
   and let this function do it. */
int cgbn_ecm_stage2(mpz_t *factors, int *array_found, int max_factors,
                    const mpz_t N, uint32_t curves, const uint64_t *sigma,
                    const char *const *x_hex, const ecm_stage2_opts *opts,
                    float *gputime, uint32_t *stage1_done, int verbose);

/* Which tier would this build use for an N of `nbits` bits?  Returns 1 and fills the
   outputs, or 0 when no instantiated tier covers it.  Pure query: no context, no
   launch.  (Same idea as ecm_cuda_tier_list, but for the stage-2 kernel family.) */
int cgbn_stage2_tier_for(uint32_t nbits, uint32_t *bits, uint32_t *tpi, uint32_t *tpb);

#ifdef __cplusplus
}
#endif

#endif /* _CGBN_STAGE2_CUDA_H */
