/* ecm_cuda_backend.cu — CUDA implementation of the ECM backend seam.

   Linked only into the `ecm_cuda` executable. Provides:
     - the GMP-ECM logging shims (outputf / test_verbose) that
       kernels/cuda/cgbn_stage1.cu expects, routed to the project's timestamped
       logger (ecm_ts_vfprintf);
     - ecm_cuda_print_ptx_version (runtime PTX/SM version report);
     - the three backend hooks (print_kernels / prepare / stage1), with device
       enumeration + selection done here (the .cu never calls cudaSetDevice).
*/

#include "ecm_backend.h"
#include "cgbn_stage1_cuda.h"   /* native cgbn_ecm_stage1 */
#include "cuda_ecm_shim.h"      /* OUTPUT_*, outputf/test_verbose/ecm_cuda_set_verbose */
#include "cudacommon.h"         /* kernel_info declaration */
#include "opencl_ecm_log.h"     /* ecm_ts_vfprintf / ecm_ts_fprintf */

#include <cuda_runtime.h>

#include <cstdarg>
#include <cstdio>
#include <cstring>

/* ── Verbose threshold + logging shims (used by cgbn_stage1.cu) ──────────── */

/* Default to OUTPUT_NORMAL so informative "GPU:" lines show; ecm_backend_stage1
   overrides this from the driver's -v flag before the run. */
static int g_cuda_verbose = OUTPUT_NORMAL;

extern "C" void ecm_cuda_set_verbose(int level) { g_cuda_verbose = level; }

extern "C" int test_verbose(int level) { return g_cuda_verbose >= level; }

extern "C" void outputf(int verbosity, const char *format, ...) {
    /* OUTPUT_ERROR is always emitted (to stderr); everything else is gated. */
    if (verbosity != OUTPUT_ERROR && g_cuda_verbose < verbosity)
        return;
    FILE *stream = (verbosity == OUTPUT_ERROR) ? stderr : stdout;
    va_list ap;
    va_start(ap, format);
    ecm_ts_vfprintf(stream, format, ap);
    va_end(ap);
    fflush(stream);
}

/* ── PTX version report (used once by cgbn_stage1.cu) ────────────────────── */

/* Prints, at runtime, the PTX ISA version the kernel was compiled to and the
   SM/cubin binary version it targets. cudaFuncAttributes encodes both as
   major*10 + minor (e.g. ptxVersion=85 -> PTX ISA 8.5, binaryVersion=80 -> SM 8.0).
   Also reports the CUDA runtime version for context. */
void ecm_cuda_print_ptx_version(const void *func) {
    struct cudaFuncAttributes attr;
    cudaError_t err = cudaFuncGetAttributes(&attr, func);
    if (err != cudaSuccess) {
        /* cudaErrorInvalidDeviceFunction here means the binary has no device code
           for this GPU's compute capability (no matching cubin, and no embedded
           PTX to JIT). The actual kernel launch will fail the same way. */
        int dev = 0;
        cudaGetDevice(&dev);
        struct cudaDeviceProp p;
        if (cudaGetDeviceProperties(&p, dev) == cudaSuccess) {
            outputf(OUTPUT_ERROR,
                    "GPU: cannot query kernel (%s). This build has no device code for "
                    "your GPU (sm_%d%d). Rebuild with -DECM_CUDA_ARCHITECTURES=%d%d "
                    "(or a list, e.g. \"61;80;89\").\n",
                    cudaGetErrorString(err), p.major, p.minor, p.major, p.minor);
        } else {
            outputf(OUTPUT_ERROR, "GPU: cannot query kernel: %s\n",
                    cudaGetErrorString(err));
        }
        return;
    }

    int rt = 0;
    cudaRuntimeGetVersion(&rt); /* e.g. 12060 -> CUDA 12.6 */

    outputf(OUTPUT_NORMAL,
            "GPU: kernel PTX ISA %d.%d, SM binary sm_%d%d (CUDA runtime %d.%d)\n",
            attr.ptxVersion / 10, attr.ptxVersion % 10,
            attr.binaryVersion / 10, attr.binaryVersion % 10,
            rt / 1000, (rt % 1000) / 10);

    outputf(OUTPUT_NORMAL,
            "GPU: maxThreadsPerBlock = %d GPU: numRegsPerThread = %d sharedMemPerBlock = %zu bytes\n",
            attr.maxThreadsPerBlock, attr.numRegs,
            attr.sharedSizeBytes);
}

/* ── Backend hooks ──────────────────────────────────────────────────────── */

extern "C" const char *ecm_backend_name(void) {
    return "CUDA/CGBN";
}

extern "C" void ecm_backend_print_kernels(FILE *out) {
    fprintf(out, "CUDA/CGBN backend: compiled kernel sizes (bits):\n");
#ifdef ECM_CUDA_FULL_BUILD
    fprintf(out, "  128 192 256 384 512 768 1024 1280 1536 ... 32768 (full build)\n");
#else
    fprintf(out, "  128 192 256 384 512 768 1024 (dev build)\n");
    fprintf(out, "  define ECM_CUDA_FULL_BUILD to build the full kernel set.\n");
#endif
    fprintf(out, "  note: --mul/--sqr/--add/--sub/--special_mult are OpenCL-only "
                 "and ignored by the CUDA backend.\n");
    fflush(out);
}

extern "C" int ecm_backend_prepare(size_t n_log2, int verbose, int device_index,
                                   const char *gpu_mul_path, const char *gpu_sqr_path,
                                   const char *gpu_add_path, const char *gpu_sub_path,
                                   const char *gpu_special_mult_path) {
    (void)n_log2;
    (void)gpu_mul_path; (void)gpu_sqr_path; (void)gpu_add_path;
    (void)gpu_sub_path; (void)gpu_special_mult_path;

    ecm_cuda_set_verbose(verbose);

    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    if (err != cudaSuccess || count == 0) {
        ecm_ts_fprintf(stderr, "GPU: no CUDA devices available: %s\n",
                       cudaGetErrorString(err));
        return 1;
    }

    ecm_ts_fprintf(stdout, "Available CUDA devices:\n");
    for (int i = 0; i < count; ++i) {
        cudaDeviceProp p;
        if (cudaGetDeviceProperties(&p, i) == cudaSuccess) {
            ecm_ts_fprintf(stdout,
                           "  [%d] %s | CC %d.%d | %d SMs | %.0f MB\n",
                           i, p.name, p.major, p.minor, p.multiProcessorCount,
                           (double)p.totalGlobalMem / (1024.0 * 1024.0));
        }
    }

    int dev = (device_index < 0) ? 0 : device_index;
    if (dev >= count) {
        ecm_ts_fprintf(stderr, "GPU: requested device %d out of range (%d present)\n",
                       dev, count);
        return 1;
    }

    err = cudaSetDevice(dev);
    if (err != cudaSuccess) {
        ecm_ts_fprintf(stderr, "GPU: cudaSetDevice(%d) failed: %s\n", dev,
                       cudaGetErrorString(err));
        return 1;
    }

    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, dev) == cudaSuccess) {
        ecm_ts_fprintf(stdout,
                       "GPU: will use device %d: %s, compute capability %d.%d, %d MPs.\n",
                       dev, p.name, p.major, p.minor, p.multiProcessorCount);
        ecm_ts_fprintf(stdout,
                       "GPU: maxSharedPerBlock = %zu maxThreadsPerBlock = %d "
                       "maxRegsPerBlock = %d\n",
                       p.sharedMemPerBlock, p.maxThreadsPerBlock, p.regsPerBlock);
    }

    /* Light context warmup (blocking sync scheduling + establish context). */
    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    err = cudaFree(0);
    if (err != cudaSuccess) {
        ecm_ts_fprintf(stderr, "GPU: context init failed: %s\n",
                       cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

/* ── --gpu-info query ───────────────────────────────────────────────────────
   Answers "which kernel would you run for this N, and how many curves does it
   take to fill the GPU" without running a curve, writing a checkpoint or
   touching the save directories.

   The device IS selected here (cudaSetDevice), because the register-allowed
   occupancy is per compute capability: asking device 0's context about device
   1's kernel would report numbers for the wrong GPU. Nothing is launched and
   nothing is written, so -d 1 --gpu-info remains safe to run at any time.

   The tier list and the occupancy query live in cgbn_stage1.cu, next to the
   kernels they describe, so this report cannot drift from the run path. */
extern "C" int ecm_backend_query_gpu(int device_index, int gpu_param, uint32_t want_bits,
                                     ecm_backend_gpu_info *out, const char **err) {
    static char reason[256];
    reason[0] = '\0';
    if (err != nullptr) *err = reason;
    if (out == nullptr) {
        snprintf(reason, sizeof(reason), "no output buffer");
        return ECM_BACKEND_QUERY_FAILED;
    }
    memset(out, 0, sizeof(*out));

    const int fold = ecm_cuda_fold_build();
    out->fold = fold;
    out->gpu_param = gpu_param;

    if (gpu_param != 0 && gpu_param != 2 && gpu_param != 3) {
        snprintf(reason, sizeof(reason),
                 "unknown gpu_param %d (expected 0, 2 or 3)", gpu_param);
        return ECM_BACKEND_QUERY_FAILED;
    }
    if (fold && gpu_param == 3) {
        /* Exactly what the run path refuses: see the fold checks in cgbn_stage1.cu. */
        snprintf(reason, sizeof(reason),
                 "this binary was built with the Mersenne fold (-DECM_MERS_FOLD=1) and "
                 "only carries fold kernels for gpu_param 0 and 2; gpu_param=3 needs the "
                 "Montgomery kernels");
        return ECM_BACKEND_QUERY_FAILED;
    }

    int count = 0;
    cudaError_t e = cudaGetDeviceCount(&count);
    if (e != cudaSuccess || count <= 0) {
        snprintf(reason, sizeof(reason), "no CUDA devices available: %s",
                 cudaGetErrorString(e));
        return ECM_BACKEND_QUERY_FAILED;
    }
    const int dev = (device_index < 0) ? 0 : device_index;
    if (dev >= count) {
        snprintf(reason, sizeof(reason), "requested device %d out of range (%d present)",
                 dev, count);
        return ECM_BACKEND_QUERY_FAILED;
    }

    struct cudaDeviceProp prop;
    e = cudaGetDeviceProperties(&prop, dev);
    if (e != cudaSuccess) {
        snprintf(reason, sizeof(reason), "cudaGetDeviceProperties(%d) failed: %s",
                 dev, cudaGetErrorString(e));
        return ECM_BACKEND_QUERY_FAILED;
    }
    /* Needed for correct occupancy numbers on -d 1 / -d 2 (see the note above). */
    e = cudaSetDevice(dev);
    if (e != cudaSuccess) {
        snprintf(reason, sizeof(reason), "cudaSetDevice(%d) failed: %s", dev,
                 cudaGetErrorString(e));
        return ECM_BACKEND_QUERY_FAILED;
    }

    out->device_index = dev;
    out->sm_count = prop.multiProcessorCount;
    out->cc_major = prop.major;
    out->cc_minor = prop.minor;
    snprintf(out->name, sizeof(out->name), "%s", prop.name);

    const int param0 = (gpu_param == 0) ? 1 : 0;
    const int param2 = (gpu_param == 2) ? 1 : 0;

    ecm_cuda_tier raw[ECM_BACKEND_MAX_TIERS];
    const int n = ecm_cuda_tier_list(param0, param2, want_bits, raw,
                                     ECM_BACKEND_MAX_TIERS, &out->carry_bits);
    if (n < 0 || n == 0) {
        snprintf(reason, sizeof(reason),
                 "no tier of this build covers an N of %u bits with gpu_param=%d",
                 want_bits, gpu_param);
        return ECM_BACKEND_QUERY_FAILED;
    }
    out->picked = (want_bits != 0u) ? 1 : 0;

    const uint32_t sm = (uint32_t)prop.multiProcessorCount;
    for (int i = 0; i < n && i < ECM_BACKEND_MAX_TIERS; ++i) {
        ecm_backend_tier &t = out->tiers[i];
        t.bits = raw[i].bits;
        t.tpb = raw[i].tpb;
        t.tpi = raw[i].tpi;
        t.ipb = (raw[i].tpi != 0u) ? (raw[i].tpb / raw[i].tpi) : 0u;

        uint32_t bps = 0;
        ecm_cuda_tier_occupancy(param0, param2, t.bits, &bps);
        t.blocks_per_sm = bps;

        t.blocks_min = sm;
        t.curves_min = (uint64_t)t.blocks_min * (uint64_t)t.ipb;
        /* The register-allowed block slots: this is what the run path's own note calls
           "a whole multiple avoids a partial last wave", and curves_wave is exactly the
           `curves = blocks_per_sm * sm_count * (tpb / tpi)` it recommends. */
        t.blocks_wave = sm * ((bps != 0u) ? bps : 1u);
        t.curves_wave = (uint64_t)t.blocks_wave * (uint64_t)t.ipb;

        t.fold_blocks_min = 2u * sm;
        t.fold_curves_min = (uint64_t)t.fold_blocks_min * (uint64_t)t.ipb;
    }
    out->tier_count = (n < ECM_BACKEND_MAX_TIERS) ? n : ECM_BACKEND_MAX_TIERS;
    return ECM_BACKEND_QUERY_OK;
}

extern "C" int ecm_backend_stage1(mpz_t *factors, int *array_found,
                                  const mpz_t N, const mpz_t s,
                                  uint32_t curves, uint64_t *sigma,
                                  unsigned long checkpoint_interval_ms,
                                  float *gputime, int verbose, int gpu_param,
                                  uint64_t B1, uint32_t torsion,
                                  const char *gpu_mul_path, const char *gpu_sqr_path,
                                  const char *gpu_add_path, const char *gpu_sub_path,
                                  const char *gpu_special_mult_path) {
    (void)gpu_mul_path; (void)gpu_sqr_path; (void)gpu_add_path;
    (void)gpu_sub_path; (void)gpu_special_mult_path;

    ecm_cuda_set_verbose(verbose);
    return cgbn_ecm_stage1(factors, array_found, N, s, curves, sigma,
                           checkpoint_interval_ms, gputime, verbose, gpu_param, B1, torsion);
}
