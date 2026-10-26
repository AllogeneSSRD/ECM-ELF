/* cuda_ecm_shim.h — minimal GMP-ECM compatibility shim for the CUDA/CGBN port.

   src/cuda/stage1/montgomery.cu was lifted almost verbatim from GMP-ECM, where it
   included "ecm.h" and "ecm-gpu.h". Those headers drag in the entire GMP-ECM
   internals. In MPA-OpenCl we only need a tiny surface:

     - OUTPUT_* verbosity levels
     - outputf() / test_verbose()  (logging; implemented in the CUDA glue and
                                     routed to the project's timestamped logger)
     - ECM_GPU_* constants
     - ECM_* return codes           (reused from the project's include/ecm.h)

   The logging shims live in src/cuda/backend.cpp.
*/

#ifndef _CUDA_ECM_SHIM_H
#define _CUDA_ECM_SHIM_H 1

/* Project return codes: ECM_NO_FACTOR_FOUND, ECM_FACTOR_FOUND_STEP1,
   ECM_ERROR, ECM_PARAM_BATCH_32BITS_D, ... */
#include "../../stage1/libecm_compat.h"

/* ── GMP-ECM verbosity levels (values must match GMP-ECM semantics) ────────
   outputf(level, ...) prints when the active verbose threshold >= level.
   OUTPUT_ERROR is special: always emitted (to stderr). */
#define OUTPUT_ERROR      (-1)
#define OUTPUT_ALWAYS       0
#define OUTPUT_NORMAL       1
#define OUTPUT_VERBOSE      2
#define OUTPUT_RESVERBOSE   3
#define OUTPUT_DEVVERBOSE   4
#define OUTPUT_TRACE        5

/* ── CGBN GPU limits (from GMP-ECM ecm-gpu.h) ───────────────────────────── */
#ifndef ECM_GPU_CGBN_MAX_BITS
#define ECM_GPU_CGBN_MAX_BITS (32 * 1024)
#endif
#ifndef ECM_GPU_CURVES_BY_BLOCK
#define ECM_GPU_CURVES_BY_BLOCK 32
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Logging shims (defined in src/cuda/backend.cpp). */
void outputf(int verbosity, const char *format, ...);
int  test_verbose(int level);

/* Sets the active verbose threshold used by outputf/test_verbose. Called by the
   CUDA backend before entering cgbn_ecm_stage1. */
void ecm_cuda_set_verbose(int level);

#ifdef __cplusplus
}
#endif

#endif /* _CUDA_ECM_SHIM_H */
