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

/* ── D4: kernel-tier query (ecm_cuda.exe --gpu-info) ─────────────────────────────

   `ecm_gui`'s worktodo generator has to recommend `gpucurves`, and doing that from a
   copied table would drift from the kernel. These two hooks report what THIS build
   actually carries, straight from the same tier list and the same per-TPI dispatch the
   run path uses (see ecm_build_tier_list / the dispatch calls in cgbn_stage1.cu).

   Pure query: no kernel launch, no context warmup, no file access. */
typedef struct {
    uint32_t bits;   /* CGBN container size this tier computes in */
    uint32_t tpb;    /* threads per block launched with */
    uint32_t tpi;    /* threads per curve; instances/block IPB = tpb / tpi */
} ecm_cuda_tier;

/* Fills `out` with the tiers of this build in ascending `bits` order and returns the
   number written (capped at max_out), or -1 when no tier satisfies the request.
     param0    : 1 = Suyama param0 (`--gpu-param 0`), 0 = param3, param2 selects param2
     want_bits : 0 = every tier; otherwise the tier the kernel would pick for an N of
                 that many bits, i.e. the smallest tier with `bits >= want_bits + carry`
   `carry_bits` (optional) receives CARRY_BITS, the container overhead the kernel adds. */
int ecm_cuda_tier_list(int param0, int param2, uint32_t want_bits,
                       ecm_cuda_tier *out, int max_out, uint32_t *carry_bits);

/* 1 when this binary was built with the Mersenne-fold kernels (ECM_MERS_FOLD), which
   need at least two resident blocks per SM -- the generator recommends accordingly. */
int ecm_cuda_fold_build(void);

/* Register-allowed resident blocks per SM for one tier, asked through the read-only
   occupancy API the run path itself uses (cudaOccupancyMaxActiveBlocksPerMultiprocessor
   with TPB_DEFAULT: no kernel launch, no context warmup, no file access). This is the
   `blocks_per_sm` in the kernel's own gpucurves recommendation
   `curves = blocks_per_sm * sm_count * (tpb / tpi)`.
   Returns 1 and writes *blocks_per_sm when the tier exists in this build, 0 otherwise
   (leaving *blocks_per_sm at 0). */
int ecm_cuda_tier_occupancy(int param0, int param2, uint32_t bits,
                            uint32_t *blocks_per_sm);

#ifdef __cplusplus
}
#endif

#endif /* _CGBN_STAGE1_CUDA_H */
