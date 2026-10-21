/* cgbn_stage1.h: header for CGBN (GPU) based ecm stage 1.

Copyright 2021 Seth Troisi

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 3 of the License, or (at your
option) any later version.

This program is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
more details.

You should have received a copy of the GNU General Public License
along with this program; see the file COPYING.  If not, see
http://www.gnu.org/licenses/ or write to the Free Software Foundation, Inc.,
51 Franklin St, Fifth Floor, Boston, MA 02110-1301, USA.
*/

#ifndef _CGBN_STAGE1_CU
#define _CGBN_STAGE1_CU 1

#ifndef __CUDACC__
#error "This file should only be compiled with nvcc"
#endif

#include "api.h"
#include "host_math.h"
#include "host_io.h"
using namespace ecm_cuda_mont_host;

#include <cassert>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

// GMP import must proceed cgbn.h
#include <gmp.h>
#include <cgbn.h>
#include <cuda.h>

#include "launch.h"  // device templates + per-TPI dispatch seam

#include "../../runtime/cudacommon.h"

// progress-bar colour helpers (ANSI code / reset)
#include "ecm/platform/logging.h"

// MPA-OpenCl port: replaces GMP-ECM's "ecm.h"/"ecm-gpu.h". Provides OUTPUT_*,
// outputf/test_verbose shims, ECM_GPU_* constants, and pulls in the project's
// ECM_* return codes (include/ecm.h).
#include "../../runtime/cuda_ecm_shim.h"
#include "../../../../storage/checkpoint/ecm_stage1_checkpoint_runtime.h"
#include "ecm/platform/stage1_progress.h"

#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <algorithm>
#include <filesystem>

// For %I64u on windows or %llu on linux
#include <cinttypes>

#ifdef _WIN32
#include <io.h>      // _isatty, _fileno
#else
#include <unistd.h>  // isatty, fileno
#endif


// ASCII progress bar. Deliberately no Unicode: nvcc's EDG frontend misparses
// UTF-8 string literals under the GBK code page on Windows, and -Xcompiler
// flags only reach the host cl.exe stage, not nvcc's frontend.
// How often to emit a full (newline-terminated) progress line when stdout is
// redirected to a file/pipe. work_manager.ps1 tails the child's stdout file by
// complete lines, so a `\r` in-place update would never be flushed to the log.
// Redirected (pipe / file, e.g. under ecm_gui) progress: keep COMPLETE lines, but at the
// same ~200 ms cadence the interactive branch gets for free from the batch loop (batch_size
// is auto-tuned so one batch is ~100 ms -- see the pacing code at the end of the loop).
// Gating on the batch number alone is wrong: that counter is RESTORED from the checkpoint
// (`batches_complete = ckpt_header.batches_complete`), so a resumed run starts at, say,
// 73900 and then matches only `n % 10000 == 0` -- with ~1 s per batch that is one line every
// ~1.7 hours. That is exactly why the GUI sat on "waiting for the first progress line"
// while the same exe in a console updated every batch (measured 2026-09-29 on a resumed
// B1=2.6e8 task). The legacy milestones stay active so an old log tailer still sees them.
static double g_last_progress_line_ms = -1.0e9;   // *gputime (ms) of the last line


// ── Kernel tier list (D4) ──────────────────────────────────────────────────────
// The tiers this build carries, in ASCENDING bits order. The run path used to build this
// list inline twice (once for the default param2/param3 path, once for param0); both
// copies are reproduced here verbatim so the run path and `--gpu-info`
// (ecm_cuda_tier_list below) can never disagree about which tiers exist. The selection
// loop additionally skips a tier whose dispatch returns no kernel, which is how a dev
// build copes with the larger tiers.

static void ecm_build_tier_list_default(std::vector<uint32_t> &out) {
  out.clear();
#define ECM_CUDA_ALL_TIERS
#ifdef IS_DEV_BUILD
#define ECM_CUDA_TIER(W,T,G,M,D,A,MT,ET) if constexpr (M && D) out.push_back(W);
#else
#define ECM_CUDA_TIER(W,T,G,M,D,A,MT,ET) if constexpr (M) out.push_back(W);
#endif
#include "../../dispatch/tiers.def"
#undef ECM_CUDA_TIER
#undef ECM_CUDA_ALL_TIERS
#ifdef IS_DEV_BUILD
    outputf(OUTPUT_ALWAYS, "Warning: Dev buils, only support N<1024.\n");
    outputf(OUTPUT_ALWAYS, "Warning: Using dev build with only 2 kernels. Consider adding more kernels for better performance on large inputs.\n");
#endif
}

// param0 uses the same container grid as param3, but the list is built without the
// dev-build guards: the selection loop skips the tiers this build has no kernel for.
static void ecm_build_tier_list_param0(std::vector<uint32_t> &out) {
  out.clear();
#define ECM_CUDA_ALL_TIERS
#define ECM_CUDA_TIER(W,T,G,M,D,A,MT,ET) if constexpr (M) out.push_back(W);
#include "../../dispatch/tiers.def"
#undef ECM_CUDA_TIER
#undef ECM_CUDA_ALL_TIERS
}


// Checkpoint configuration
// 小端格式，magic number 0x45555047 在内存中表示为 "GPUE"，用于验证 checkpoint 文件的正确性

// support routine copied from  "CGBN/samples/utility/support.h"
void cgbn_check(cgbn_error_report_t *report, const char *file=NULL, int32_t line=0) {
  // check for cgbn errors

  if(cgbn_error_report_check(report)) {
    fprintf (stderr, "\n");
    fprintf (stderr, "CGBN error occurred: %s\n", cgbn_error_string(report));

    if(report->_instance!=0xFFFFFFFF) {
      fprintf (stderr, "Error reported by instance %d", report->_instance);
      if(report->_blockIdx.x!=0xFFFFFFFF)
        fprintf (stderr, ", blockIdx=(%d, %d, %d)", report->_blockIdx.x, report->_blockIdx.y, report->_blockIdx.z);
      if(report->_threadIdx.x!=0xFFFFFFFF)
        fprintf (stderr, ", threadIdx=(%d, %d, %d)", report->_threadIdx.x, report->_threadIdx.y, report->_threadIdx.z);
      fprintf (stderr, "\n");
    }
    else {
      fprintf (stderr, "Error reported by blockIdx=(%d %d %d)", report->_blockIdx.x, report->_blockIdx.y, report->_blockIdx.z);
      fprintf (stderr, "threadIdx=(%d %d %d)\n", report->_threadIdx.x, report->_threadIdx.y, report->_threadIdx.z);
    }
    if(file!=NULL)
      fprintf (stderr, "file %s, line %d\n", file, line);
    exit(1);
  }
}

#define CGBN_CHECK(report) cgbn_check(report, __FILE__, __LINE__)

static
int verify_size_of_n(const mpz_t N, size_t max_bits) {
  size_t n_log2 = mpz_sizeinbase(N, 2);

  /* Using check_gpuecm.sage it looks like 4 bits would suffice. */
  size_t max_usable_bits = max_bits - CARRY_BITS;

  if (n_log2 <= max_usable_bits)
    return ECM_NO_FACTOR_FOUND;

  outputf (OUTPUT_ERROR, "GPU: N(%d bits) + carry(%d bits) > BITS(%d)\n",
      n_log2, CARRY_BITS, max_bits);
  outputf (OUTPUT_ERROR, "GPU: Error, input number should be stricly lower than 2^%d\n",
      max_usable_bits);
  return ECM_ERROR;
}


// Resolve a kernel BITS to its instantiated __global__ function pointer (and
// TPI) by consulting the per-TPI dispatch TUs in order.
static cgbn_stage1_kernel_fn cgbn_stage1_kernel_dispatch(uint32_t BITS, uint32_t *TPI_out) {
    cgbn_stage1_kernel_fn k = cgbn_stage1_kernel_tpi4(BITS, TPI_out);
    if (k != nullptr) return k;
    k = cgbn_stage1_kernel_tpi8(BITS, TPI_out);
    if (k != nullptr) return k;
    k = cgbn_stage1_kernel_tpi16(BITS, TPI_out);
    if (k != nullptr) return k;
    k = cgbn_stage1_kernel_tpi32(BITS, TPI_out);
    return k;
}

/* Suyama param0 kernel lookup (see cgbn_stage1_kernels_suyama.cu). */
static cgbn_stage1_kernel_fn cgbn_stage1_kernel_suyama_dispatch(uint32_t BITS, uint32_t *TPI_out) {
    cgbn_stage1_kernel_fn k = cgbn_stage1_kernel_suyama_tpi4(BITS, TPI_out);
    if (k != nullptr) return k;
    k = cgbn_stage1_kernel_suyama_tpi8(BITS, TPI_out);
    if (k != nullptr) return k;
    k = cgbn_stage1_kernel_suyama_tpi16(BITS, TPI_out);
    if (k != nullptr) return k;
    return cgbn_stage1_kernel_suyama_tpi32(BITS, TPI_out);
}

/* param2 kernel lookup (see cgbn_stage1_kernels_param2.cu). */
static cgbn_stage1_kernel_fn cgbn_stage1_kernel_param2_dispatch(uint32_t BITS, uint32_t *TPI_out) {
    cgbn_stage1_kernel_fn k = cgbn_stage1_kernel_param2_tpi4(BITS, TPI_out);
    if (k != nullptr) return k;
    k = cgbn_stage1_kernel_param2_tpi8(BITS, TPI_out);
    if (k != nullptr) return k;
    k = cgbn_stage1_kernel_param2_tpi16(BITS, TPI_out);
    if (k != nullptr) return k;
    return cgbn_stage1_kernel_param2_tpi32(BITS, TPI_out);
}

/* ── D4: kernel-tier query (ecm_cuda.exe --gpu-info) ─────────────────────────────
   Reports the tiers this build carries plus the TPB/TPI each one launches with, using the
   SAME tier lists and the SAME per-TPI dispatch as the run path. That is what makes the
   `gpucurves` recommendation (`curves = blocks_per_sm * sm_count * (tpb / tpi)`)
   trustworthy: the numbers cannot drift from what the kernel would actually pick.

   Pure query: no kernel launch, no device call, no file access. `want_bits` is the bit
   length of the number the task will factor (CARRY_BITS is added here, so callers pass the
   plain bit size); 0 means "every tier". Returns the number of entries written, or -1 when
   no tier is large enough. */
int ecm_cuda_tier_list(int param0, int param2, uint32_t want_bits,
                       ecm_cuda_tier *out, int max_out, uint32_t *carry_bits) {
    if (carry_bits != nullptr) *carry_bits = (uint32_t)CARRY_BITS;
    if (out == nullptr || max_out <= 0) return -1;

    std::vector<uint32_t> tiers;
    if (param0) {
        ecm_build_tier_list_param0(tiers);
    } else {
        ecm_build_tier_list_default(tiers);
    }

    int written = 0;
    for (std::size_t i = 0; i < tiers.size(); ++i) {
        const uint32_t bits = tiers[i];
        if (want_bits != 0u && bits < want_bits + (uint32_t)CARRY_BITS) continue;
        uint32_t tpi_u32 = 0;
        cgbn_stage1_kernel_fn k = param0 ? cgbn_stage1_kernel_suyama_dispatch(bits, &tpi_u32)
                                 : param2 ? cgbn_stage1_kernel_param2_dispatch(bits, &tpi_u32)
                                          : cgbn_stage1_kernel_dispatch(bits, &tpi_u32);
        if (k == nullptr) continue;          /* tier not instantiated in this build */
        if (written < max_out) {
            out[written].bits = bits;
            out[written].tpb = (uint32_t)TPB_DEFAULT;
            out[written].tpi = tpi_u32;
        }
        ++written;
        if (want_bits != 0u) break;          /* the kernel would stop at the first fit */
    }
    if (want_bits != 0u && written == 0) return -1;
    return written > max_out ? max_out : written;
}

int ecm_cuda_fold_build(void) {
#if ECM_MERS_FOLD
    return 1;
#else
    return 0;
#endif
}

int ecm_cuda_tier_occupancy(int param0, int param2, uint32_t bits,
                            uint32_t *blocks_per_sm) {
    if (blocks_per_sm == nullptr) return 0;
    *blocks_per_sm = 0;

    uint32_t tpi_u32 = 0;
    cgbn_stage1_kernel_fn k = param0 ? cgbn_stage1_kernel_suyama_dispatch(bits, &tpi_u32)
                             : param2 ? cgbn_stage1_kernel_param2_dispatch(bits, &tpi_u32)
                                      : cgbn_stage1_kernel_dispatch(bits, &tpi_u32);
    if (k == nullptr) return 0;   /* tier not instantiated in this build */

    int occ = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, k, (int)TPB_DEFAULT, 0)
            != cudaSuccess || occ <= 0) {
        return 0;
    }
    *blocks_per_sm = (uint32_t)occ;
    return 1;
}

#ifndef ECM_CUDA_ENABLE_PRAC
#define ECM_CUDA_ENABLE_PRAC 0
#endif
#if ECM_CUDA_ENABLE_PRAC
#include "prac/prac_runtime.inl"
#endif

int cgbn_ecm_stage1(mpz_t *factors, int *array_found,
             const mpz_t N, const mpz_t s,
             uint32_t curves, uint64_t *sigma_ptr,
             unsigned long checkpoint_interval_ms,
             float *gputime, int verbose, int gpu_param, uint64_t B1, uint32_t torsion)
{
  const char *algorithm = getenv("ECM_GPU_STAGE1_ALGO");
  const char *window = getenv("ECM_PRAC_WINDOW");
  const char *variant = getenv("ECM_PRAC_VARIANT");
  if (((window && *window) || (variant && *variant && strcmp(variant, "baseline") != 0)) &&
      (!algorithm || strcmp(algorithm, "prac") != 0)) {
    outputf(OUTPUT_ERROR, "GPU: PRAC window/variant requires ECM_GPU_STAGE1_ALGO=prac\n");
    return ECM_ERROR;
  }
  if (algorithm && *algorithm && strcmp(algorithm, "ladder") != 0) {
    if (strcmp(algorithm, "prac") != 0 && strcmp(algorithm, "resident") != 0) {
      outputf(OUTPUT_ERROR, "GPU: ECM_GPU_STAGE1_ALGO must be ladder, resident or prac\n");
      return ECM_ERROR;
    }
#if ECM_CUDA_ENABLE_PRAC
    if (gpu_param != 0 || ECM_MERS_FOLD) {
      outputf(OUTPUT_ERROR, "GPU: resident/PRAC requires param0 and a Montgomery build\n");
      return ECM_ERROR;
    }
    return cgbn_stage1_resident(factors, array_found, N, s, curves, sigma_ptr,
                               checkpoint_interval_ms, gputime, B1, torsion,
                               strcmp(algorithm, "prac") == 0);
#else
    outputf(OUTPUT_ERROR, "GPU: resident/PRAC kernels are not built; rebuild with "
                         "-EnablePrac or -DECM_CUDA_ENABLE_PRAC=ON\n");
    return ECM_ERROR;
#endif
  }
  uint64_t sigma64 = (sigma_ptr != NULL) ? *sigma_ptr : 0;

  /* -------------------------------------------------------------------------
   * Parametrization selection.
   *
   * 3 (default) = gmp-ecm batch parametrization, the historical GPU path:
   *               P = (2:1), 2P = (9, 64d+8), d = sigma/2^32 doubling constant,
   *               difference x = 2 folded into the formulas  => 4M+4S + cheap
   *               32-bit multiply, save file carries PARAM=3.
   * 0          = Suyama (Prime95 sigma_type=1 / gmp-ecm -param 0): P = (u^3:v^3),
   *               full-width a24, difference x = xdiff  => 6M+4S, same curves as
   *               the CPU path, save file carries no PARAM (param0 form).
   * 2          = param2, gmp-ecm's "batch 2" 6-torsion family (-param 2): x0 = 2 like
   *               the batch family (so the difference needs no multiply) but a
   *               FULL-WIDTH a24  => 5M+4S, save file carries PARAM=2.
   *               Success rate matches Suyama's (D_eff ~20) at ~11% less time per
   *               curve than param0; the catch is that Prime95 cannot read its saves
   *               (sigma_type only accepts 0/1/3).  See
   *               docs/performance/STAGE1.md
   *
   * NOTE: the selector arrives from the driver (ini gpu_param / CLI --gpu-param);
   * see docs/architecture/STAGE1.md.
   * ------------------------------------------------------------------------- */
  if (gpu_param != 0 && gpu_param != 2 && gpu_param != 3) {
      outputf(OUTPUT_ERROR, "GPU: gpu_param=%d is not 0, 2 or 3; using 3\n", gpu_param);
      gpu_param = 3;
  }
  const bool param0 = (gpu_param == 0);
  const bool param2 = (gpu_param == 2);
  /* param0 and param2 share the 7-word Suyama buffer layout and kernel family shape
     (they differ only in the step variant the family instantiates). */
  const bool suyama_layout = param0 || param2;
  /* sigma64 is the authoritative curve index (Suyama param0 uses a full 64-bit
     sigma, like the CPU path); the batch parametrizations carry a 32-bit parameter
     (param3: d = sigma/2^32; param2: the scalar multiplier sigma itself). */
  if (gpu_param == 3 && sigma64 + (uint64_t)curves > 0x100000000ull) {
      outputf(OUTPUT_ERROR, "GPU: param3 needs sigma + curves <= 2^32\n");
      return ECM_ERROR;
  }
  uint32_t sigma32 = (uint32_t)(sigma64 & 0xFFFFFFFFull);
  if (param0) {
      /* OUTPUT_ALWAYS: which curve family is being run is as important to see as
         "Using B1=..." -- it decides what the save file can be handed to. */
      outputf(OUTPUT_ALWAYS, "GPU: parametrization = Suyama param0 (gmp-ecm -param 0 / Prime95 sigma_type=1)\n");
  } else if (param2) {
      outputf(OUTPUT_ALWAYS, "GPU: parametrization = param2 batch-2 / 6-torsion (gmp-ecm -param 2; "
                             "Prime95 CANNOT read these saves)\n");
      outputf(OUTPUT_ALWAYS,
              "GPU: param2 stage-1 x verified against gmp-ecm (identical X for the same\n"
              "     sigma and B1; tools/test/test_cuda_param2.ps1).  gmp-ecm can read these\n"
              "     PARAM=2 saves, Prime95 cannot (sigma_type 0/1/3 only).\n");
  }
      

  uint64_t s_num_bits;
  uint32_t *s_bits = allocate_and_set_s_bits(s, &s_num_bits);
  if (s_num_bits >= 4000000000)
      outputf (OUTPUT_ALWAYS, "GPU: Very Large B1! Check magnitute of B1.\n");

  if (s_num_bits >= 100000000)
      outputf (OUTPUT_NORMAL, "GPU: Large B1, S = %lu bits = %d MB\n",
               s_num_bits, s_num_bits >> 23);
  assert( s_bits != NULL );

  cudaEvent_t global_start, batch_start, stop;
  CUDA_CHECK(cudaEventCreate (&global_start));
  CUDA_CHECK(cudaEventCreate (&batch_start));
  CUDA_CHECK(cudaEventCreate (&stop));
  CUDA_CHECK(cudaEventRecord (global_start));

  // Copy s_bits
  uint32_t *gpu_s_bits;
  uint32_t s_words = (s_num_bits + 31) / 32;
  CUDA_CHECK(cudaMalloc((void **)&gpu_s_bits, sizeof(uint32_t) * s_words));
  CUDA_CHECK(cudaMemcpy(gpu_s_bits, s_bits, sizeof(uint32_t) * s_words, cudaMemcpyHostToDevice));

  cgbn_error_report_t *report;
  // create a cgbn_error_report for CGBN to report back errors
  CUDA_CHECK(cgbn_error_report_alloc(&report));

  size_t    data_size;
  uint32_t *data, *gpu_data;

  uint32_t  BITS = 0;        // kernel bits
  int32_t   TPB=TPB_DEFAULT; // Always the same default
  int32_t   TPI;
  int32_t   IPB;             // IPB = TPB / TPI, instances per block
  size_t    BLOCK_COUNT;     // How many blocks to cover all curves
  
  /* Progress variables (may be loaded from checkpoint) */
  uint64_t s_partial = 0;    // current bit progress
  int batches_complete = 0;  // number of completed batches
  
  /* Result / status variable used throughout the function */
  int youpi = ECM_NO_FACTOR_FOUND;

  const char *dump_env = getenv("ECM_GPU_DUMP");
  uint32_t dump_enable = (dump_env != NULL && atoi(dump_env) != 0) ? 1u : 0u;
  FILE *dump_file = NULL;
  uint32_t *dump_host = NULL;
  const char *dump_filename = "dump.csv";

  /**
   * Smaller TPI is faster, Larger TPI is needed for large inputs.
   * N > 512 TPI=8 | N > 2048 TPI=16 | N > 8192 TPI=32
   *
   * Larger takes longer to compile (and increases binary size)
   * No GPU, No CGBN | ecm 3.4M, 2 seconds to compile
   * GPU, No CGBN    | ecm 3.5M, 3 seconds
   * (8, 1024)       | ecm 3.8M, 12 seconds
   * (16,8192)       | ecm 4.2M, 1 minute
   * (32,16384)      | ecm 4.2M, 1 minute
   * (32,32768)      | ecm 5.2M, 4.7 minutes
   */
  /* NOTE: Custom kernel changes here
   * For "Compiling custom kernel for %d bits should be XX% faster"
   * Change the 512 in cgbn_params_t<4, 512> cgbn_params_small;
   * to the suggested value (a multiple of 32 >= bits + 6).
   * You may need to change the 4 to an 8 (or 16) if bits >512, >2048
   */
  /** TODO: try with const vector for BITs/TPI, see if compiler is happy */
  std::vector<uint32_t> available_kernels;
  // The tier list lives in ecm_build_tier_list_default() (see the D4 note near the top)
  // so this run path and --gpu-info can never disagree about which tiers exist.
  ecm_build_tier_list_default(available_kernels);

  /* Pointer to CUDA kernel. */
  void(*kernel)(cgbn_error_report_t *, uint64_t, uint64_t, uint64_t,
        uint32_t*, uint32_t*, uint32_t, uint32_t, uint32_t) = NULL;

  size_t n_log2 = mpz_sizeinbase(N, 2);
  
  // ========== CHECKPOINT LOADING ==========
  char *ckpt_filename = get_checkpoint_filename(N);
  checkpoint_header_t ckpt_header;
  uint32_t *ckpt_data = NULL;
  size_t ckpt_data_size = 0;
  int ckpt_loaded = 0;
  
  // The primary may be torn or corrupted; try each older generation in order.
  for (const std::string &candidate : ecm_stage1_checkpoint_candidates(ckpt_filename)) {
    if (load_checkpoint(candidate.c_str(), &ckpt_header, &ckpt_data, &ckpt_data_size) !=
        ECM_NO_FACTOR_FOUND) continue;
    const uint32_t wpc_want = suyama_layout ? 7u : 5u;
    const uint32_t param_want = param0 ? 0u : (param2 ? 2u : 3u);
    const uint64_t stride = (uint64_t)curves * (uint64_t)(ckpt_header.BITS / 32u) *
                            sizeof(uint32_t);
    if (ckpt_header.curves != curves || ckpt_header.s_num_bits != s_num_bits ||
        ckpt_header.s_partial > s_num_bits || ckpt_header.BITS < 32u ||
        ckpt_header.BITS % 32u || stride == 0 ||
        ckpt_header.data_size != stride * wpc_want ||
        ckpt_header.gpu_param != param_want || ckpt_header.TPI == 0u) {
      outputf(OUTPUT_NORMAL, "Checkpoint parameters/layout mismatch: %s\n", candidate.c_str());
      free(ckpt_data);
      ckpt_data = NULL;
      continue;
    }
    outputf(OUTPUT_NORMAL, "Resuming from checkpoint: %.1f%% complete (s_partial=%lu/%lu)%s\n",
            s_num_bits ? 100.0 * ckpt_header.s_partial / s_num_bits : 0.0,
            ckpt_header.s_partial, s_num_bits,
            candidate == ckpt_filename ? "" : " [backup]");
    ckpt_loaded = 1;
    if (candidate != ckpt_filename) ecm_stage1_mark_bad_primary(ckpt_filename);
    sigma64 = ckpt_header.sigma;
    sigma32 = static_cast<uint32_t>(sigma64);
    BITS = ckpt_header.BITS;
    TPI = ckpt_header.TPI;
    data_size = ckpt_data_size;
    data = ckpt_data;
    s_partial = ckpt_header.s_partial;
    batches_complete = ckpt_header.batches_complete;
    break;
  }
  
  // If no checkpoint, proceed with normal initialization
  if (!ckpt_loaded) {
  std::string existing_checkpoint;
  if (!ecm_stage1_checkpoint_path().empty() && ecm_stage1_read_blob(ckpt_filename,existing_checkpoint)) {
    outputf(OUTPUT_ERROR,"GPU: invalid or incompatible batch checkpoint payload\n");
    return ECM_ERROR;
  }
  /* param0 uses the same container grid as param3 (see cgbn_stage1_kernels_suyama.cu):
     TPI=4 for <=512, TPI=8 up to 2048, TPI=16 2560..8192, TPI=32 9216..16384, all on
     the 512-bit grid above 2560.  In a dev build only the small tiers exist, and the
     loop below skips the ones without a kernel instead of failing. */
  if (param0) {
    ecm_build_tier_list_param0(available_kernels);
  }
  for (int k_i = 0; k_i < available_kernels.size(); k_i++) {
    uint32_t kernel_bits = available_kernels[k_i];
    if (kernel_bits >= n_log2 + CARRY_BITS) {
      BITS = kernel_bits;
      assert( BITS % 32 == 0 );

      /* Resolve the kernel function pointer via the per-TPI dispatch TUs. */
      uint32_t tpi_u32 = 0;
      kernel = param2 ? cgbn_stage1_kernel_param2_dispatch(BITS, &tpi_u32)
               : param0 ? cgbn_stage1_kernel_suyama_dispatch(BITS, &tpi_u32)
                        : cgbn_stage1_kernel_dispatch(BITS, &tpi_u32);
      if (kernel == nullptr) {
        if (param0) {
          /* not instantiated in this build (dev builds only carry 768/1024):
             try the next tier instead of failing outright */
          BITS = 0;
          continue;
        }
        outputf (OUTPUT_ERROR, "CGBN kernel not found for %d bits\n", BITS);
        return ECM_ERROR;
      }
      TPI = (int32_t)tpi_u32;

      IPB = TPB / TPI;
      BLOCK_COUNT = (curves + IPB - 1) / IPB;

      break;
    }
  }
  if (BITS == 0 || kernel == NULL)
    {
#if ECM_NO_PARAM2
      if (param2) {
          outputf (OUTPUT_ERROR, "GPU: --gpu-param 2 was requested but the param2 kernels are "
                   "NOT compiled into this binary (configured with -DECM_NO_PARAM2=1).\n");
          outputf (OUTPUT_ERROR, "     Re-configure with -DECM_NO_PARAM2=0 and rebuild ecm_cuda.\n");
          return ECM_ERROR;
      }
#endif
      outputf (OUTPUT_ERROR, "No available CGBN Kernel large enough to process N(%d bits)%s\n",
               n_log2,
               param0 ? " (param0 kernels follow the param3 grid; a dev build only has <=1024)" : "");
      return ECM_ERROR;
    }

  ecm_cuda_print_ptx_version((const void*)kernel);

  /* Alert that recompiling with a smaller kernel would likely improve speed */
  {
    size_t optimized_bits = ((n_log2 + CARRY_BITS + 127)/128) * 128;
    /* Assume speed is roughly O(N) but slightly slower for not being a power of two */
    float pct_faster = 90 * BITS / optimized_bits;

    if (pct_faster > 110) {
      outputf (OUTPUT_VERBOSE, "Compiling custom kernel for %d bits should be ~%.0f%% faster see README.gpu\n",
              optimized_bits, pct_faster);
    }
  }

  youpi = verify_size_of_n(N, BITS);
  if (youpi != ECM_NO_FACTOR_FOUND) {
    return youpi;
  }

  /* Consistency check that struct cgbn_mem_t is byte aligned without extra fields. */
  assert( sizeof(cgbn_mem_t<512>) == 512/8 );
  assert( sizeof(cgbn_mem_t<1024>) == 1024/8 );
  
  if (!ckpt_loaded) {
    data = param2 ? set_p_2p_param2(N, curves, sigma32, BITS, &data_size)
                  : param0 ? set_p_2p_suyama(N, curves, sigma64, BITS, &data_size)
                           : set_p_2p(N, curves, sigma32, BITS, &data_size);
    s_partial = 1;      // First bit (doubling) is handled in set_p_2p[_suyama]
    batches_complete = 0;
  }
  } // Close the "if (!ckpt_loaded)" block from checkpoint loading
  else {
    // If checkpoint loaded, still need to resolve the kernel from its BITS.
    uint32_t tpi_u32 = 0;
    kernel = param2 ? cgbn_stage1_kernel_param2_dispatch(BITS, &tpi_u32)
             : param0 ? cgbn_stage1_kernel_suyama_dispatch(BITS, &tpi_u32)
                      : cgbn_stage1_kernel_dispatch(BITS, &tpi_u32);
    if (kernel == nullptr) {
      outputf(OUTPUT_ERROR, "CGBN kernel not found for BITS=%d TPI=%d from checkpoint\n", BITS, TPI);
      return ECM_ERROR;
    }
    TPI = (int32_t)tpi_u32;
    
    IPB = TPB / TPI;
    BLOCK_COUNT = (curves + IPB - 1) / IPB;
    outputf(OUTPUT_VERBOSE, "Checkpoint: restored BITS=%d, TPI=%d, BLOCK_COUNT=%lu\n", BITS, TPI, BLOCK_COUNT);
  }

  /* Buffer layout of this run: 5 words/curve for param3 (N, aX, aZ, bX, bZ) and
     7 for param0 (N, a24, xdiff, aX, aZ, bX, bZ).  Derived, not hard-coded, so a
     checkpoint resume is validated against it in both directions. */
  const uint32_t words_per_curve = suyama_layout ? 7u : 5u;
  /* param0 AND param2 use the 7-word layout (N, a24, xdiff, aX, aZ, bX, bZ);
     using the param0 ternary here made param2 decode its result from word 1
     (= a24) instead of word 3 (= aX), which is what broke the gmp-ecm comparison. */
  const int      p1_word = suyama_layout ? 3 : 1;      /* X of P_a */
  const int      p2_word = suyama_layout ? 5 : 3;      /* X of P_b */
  if (data_size != (size_t)words_per_curve * curves * (size_t)(BITS / 32) * sizeof(uint32_t)) {
    outputf(OUTPUT_ERROR,
            "GPU: internal error: curve buffer size %zu does not match the %u-word/curve layout\n",
            data_size, words_per_curve);
    return ECM_ERROR;
  }

  // Print the *actual* sigma now that any checkpoint resume has been applied
  // (the checkpoint may override the freshly-computed sigma from the driver).
  outputf(OUTPUT_NORMAL, "GPU: sigma=%llu (param %d, %u curves)%s\n",
          (unsigned long long)(param0 ? sigma64 : (uint64_t)sigma32),
          param0 ? 0 : (int)ECM_PARAM_BATCH_32BITS_D, curves,
          ckpt_loaded ? " [restored from checkpoint]" : " [computed]");

  /* np0 is -(N^-1 mod 2**32), used for montgomery representation */
  uint32_t np0 = find_np0(N);

#if ECM_MERS_FOLD
  /* ---------------------------------------------------------------------------
   * MERSENNE FOLD DOMAIN (kernels/cuda/stage1/montgomery/arithmetic.cuh, ECM_MERS_FOLD).
   *
   * This build's suyama family (param0 + param2) runs the fold domain, which is
   * only defined for N = 2^k - 1.  Anything else must be refused LOUDLY: the fold
   * would silently drop the top bits of the product (CGBN's shift_left inside
   * fold_mul is a fixed-width shift) and every curve would be wrong.
   *
   * The fold needs t = BITS - k, which is passed through the kernel's unused
   * `sigma_0` argument (the shared kernel-pointer typedef is unchanged).  t >= 1
   * is required (t == 0 would mean a shift by the whole word width); the tier
   * search guarantees t >= CARRY_BITS for a Mersenne N because it picks the
   * smallest kernel with BITS >= n_log2 + CARRY_BITS = k + CARRY_BITS.
   * ------------------------------------------------------------------------- */
  uint32_t fold_shift = 0;
  {
    if (!suyama_layout) {
      outputf(OUTPUT_ERROR,
              "GPU: this binary was built with the Mersenne fold (-DECM_MERS_FOLD=1) and only\n"
              "     carries fold kernels for --gpu-param 0 and 2; gpu_param=%d needs the\n"
              "     Montgomery kernels.  Rebuild without -DECM_MERS_FOLD for that path.\n",
              gpu_param);
      return ECM_ERROR;
    }
    mpz_t n_plus_1;
    mpz_init(n_plus_1);
    mpz_add_ui(n_plus_1, N, 1);
    const bool is_mersenne = (mpz_popcount(n_plus_1) == 1);
    mpz_clear(n_plus_1);
    if (!is_mersenne) {
      outputf(OUTPUT_ERROR,
              "GPU: this binary was built with the Mersenne fold (-DECM_MERS_FOLD=1), which is\n"
              "     only valid for N = 2^k - 1; this N is %zu bits but not of that form.\n"
              "     Rebuild without -DECM_MERS_FOLD for a general modulus.\n", n_log2);
      return ECM_ERROR;
    }
    if (BITS <= (int)n_log2) {
      outputf(OUTPUT_ERROR, "GPU: internal error: fold has no headroom (BITS=%d, k=%zu)\n",
              BITS, n_log2);
      return ECM_ERROR;
    }
    fold_shift = (uint32_t)(BITS - (int)n_log2);
    /* OUTPUT_ALWAYS: which DOMAIN the kernel runs matters as much as which
       parametrization -- a fold build behaves differently from a stock build, and
       OUTPUT_NORMAL is invisible at the default verbosity. */
    outputf(OUTPUT_ALWAYS,
            "GPU: Mersenne fold domain: N = 2^%zu - 1, k=%zu, t=%u (BITS=%d, TPI=%d)\n",
            n_log2, n_log2, fold_shift, BITS, TPI);
  }
#endif

  /* The suyama family does not use its `sigma_0` slot: a fold build puts t there. */
#if ECM_MERS_FOLD
  const uint32_t sigma_arg = fold_shift;
#else
  const uint32_t sigma_arg = sigma32;
#endif

  // Copy data
  outputf (OUTPUT_VERBOSE, "Copying %'lu bytes of curves data to GPU\n", data_size);
  CUDA_CHECK(cudaMalloc((void **)&gpu_data, data_size));
  CUDA_CHECK(cudaMemcpy(gpu_data, data, data_size, cudaMemcpyHostToDevice));

  outputf (OUTPUT_NORMAL,
          "GPU: CGBN<%d, %d> kernel, N is %zu bits (%d blocks x %d threads)\n",
          TPI, BITS, n_log2, BLOCK_COUNT, TPB);

  /* ---------------------------------------------------------------------------
   * Occupancy advisory (docs/performance/STAGE1.md 5.5/5.7/8.11).
   *
   * BLOCK_COUNT = ceil(curves / (TPB/TPI)), so the batch size - not the GPU - sets how many
   * blocks are in flight.  Two DIFFERENT things live here and only the first one is a real
   * problem:
   *
   *   (a) BLOCK_COUNT < number of SMs  =>  some SMs have no work at all.  That is a real
   *       underuse, and it is what this warning now reports.  Measured on the 511-bit tier
   *       at TPB=256/TPI=4: 4096 vs 8192 curves was +7.6% curve-bits/s.
   *
   *   (b) BLOCK_COUNT < sm_count * blocks_per_sm  =>  the register-allowed block SLOTS are
   *       not all filled.  This is NOT a throughput deficit for the big tiers: the kernel is
   *       issue bound (ncu: Compute SOL ~83%, DRAM ~0.5%), so extra resident warps buy no
   *       throughput -- and on a power-limited card they can cost a little.  The user's
   *       B1=260e6 / 5120-bit param0 runs (60-SM 4070 Ti) measured 120 blocks (2 blocks/SM,
   *       8 warps) at 72.88-72.90 s/curve vs 240 blocks (4 blocks/SM, 16 warps) at
   *       73.41 s/curve.  So filling the slots is reported as INFORMATION, without the old
   *       "raise -gpucurves" claim: what actually helps is the batch fitting a whole number
   *       of resident blocks (avoid a partial last wave), and that is what the text says.
   * ------------------------------------------------------------------------- */
  {
    int dev = 0, sm_count = 0, blocks_per_sm = 0;
    if (cudaGetDevice(&dev) == cudaSuccess &&
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev) == cudaSuccess &&
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, kernel, (int)TPB, 0) == cudaSuccess &&
        sm_count > 0 && blocks_per_sm > 0) {
      const long capacity = (long)sm_count * (long)blocks_per_sm;
      /* The Mersenne fold family is LATENCY bound (it deletes mont_mul's Q*N chains, so
         there is less independent work to fill the issue slots): it only reaches parity/
         lead when at least 2 blocks per SM stay resident.  Measured (4070 Ti, param0,
         M5003/M4441, docs/performance/STAGE1.md 9.9):
           TPB=256, 120 blocks = 2 blocks/SM (16 warps): fold 47.1 s/curve  vs basic 49.1
           TPB=256,  60 blocks = 1 block/SM  ( 8 warps): fold 55.7 s/curve  vs basic 50.2
           TPB=128, 240 blocks = 4 blocks/SM (16 warps): fold 51.5 s/curve  vs basic 50.3
         i.e. below 2 blocks/SM the fold is 10-25% SLOWER than the Montgomery kernels.  The
         check comes FIRST because a fold batch with less than one block per SM is entirely
         in its bad regime.  ECM_MERS_FOLD is 0 in production builds: nothing runs there. */
#if ECM_MERS_FOLD
      if ((long)BLOCK_COUNT < 2L * (long)sm_count) {
        outputf(OUTPUT_ALWAYS,
                "GPU: warning: the Mersenne fold is latency bound and needs >= 2 resident "
                "blocks/SM; %d blocks for %d SMs is below that (measured 10-25%% SLOWER than a "
                "non-fold build there).  Raise -gpucurves to about %ld (%d curves/block) or "
                "rebuild with -DECM_TPB=128.\n",
                (int)BLOCK_COUNT, sm_count, 2L * (long)sm_count * (long)IPB, (int)IPB);
      } else
#endif
      if ((long)BLOCK_COUNT < (long)sm_count) {
        /* The suggestion is CURVES, not blocks: `curves = blocks * IPB`.  The old text
           printed sm_count here, which is a block count -- feeding it into `gpucurves`
           left the GPU under-occupied (docs/usage/STAGE1.md 5.4). */
        outputf(OUTPUT_NORMAL,
                "GPU: warning: only %d blocks for %d SMs - some SMs idle; raise -gpucurves to "
                "about %ld (%d curves/block; a multiple of %ld keeps whole waves)\n",
                (int)BLOCK_COUNT, sm_count, (long)sm_count * (long)IPB, (int)IPB,
                (long)sm_count * (long)IPB);
      } else if ((long)BLOCK_COUNT < capacity) {
        outputf(OUTPUT_VERBOSE,
                "GPU: note: %d blocks = %ld%% of the %ld register-allowed block slots "
                "(%d SMs x %d blocks/SM).  That is not a throughput deficit (the kernel is "
                "issue bound), but a batch that is a whole multiple of %ld blocks avoids a "
                "partial last wave.\n",
                (int)BLOCK_COUNT, (long)BLOCK_COUNT * 100 / capacity, capacity,
                sm_count, blocks_per_sm, capacity);
      }
    }
  }

  /* Start with small batches and increase till timing is ~100ms */
  uint64_t batch_size = 200;
  
  /* gputime and batch_time are measured in ms */
  float batch_time = 0;
  
  /* Track time for checkpoint saving */
  float last_checkpoint_time = 0;

  // if (checkpoint_interval_ms > 86400000) { // > 1 day, treat as disabled
  //   checkpoint_interval_ms = 86400000; // 1 day max
  // }

  if (checkpoint_interval_ms > 0) {
    outputf(OUTPUT_NORMAL, "Checkpoint autosave interval: %lu ms (%.2f min)\n",
            checkpoint_interval_ms, checkpoint_interval_ms / 1000.0 / 60.0);
  } else {
    outputf(OUTPUT_NORMAL, "Checkpoint autosave disabled\n");
  }

  if (dump_enable != 0) {
    dump_file = open_dump_csv(dump_filename);
    if (dump_file != NULL) {
      dump_host = (uint32_t*) malloc(data_size);
      if (dump_host == NULL) {
        outputf(OUTPUT_ALWAYS, "Warning: Could not allocate host buffer for dump.csv\n");
        fclose(dump_file);
        dump_file = NULL;
      } else {
        outputf(OUTPUT_VERBOSE, "GPU dump enabled: writing kernel call states to %s\n", dump_filename);
      }
    }
  }

  // ── Progress bar (ASCII, only shown on an interactive terminal) ───────
  const bool show_progress = stdout_is_tty();
  // ──────────────────────────────────────────────────────────────────────

  // Sliding window of recent per-batch speeds (bits/ms) for a stable ETA.
  const size_t SPEED_WINDOW = 50;
  double speed_ring[SPEED_WINDOW];
  size_t speed_count = 0;
  size_t speed_idx = 0;
  double speed_sum = 0.0;

  while (s_partial < s_num_bits) {
    /* decrease batch_size for final batch if needed */
    uint64_t this_batch = std::min(s_num_bits - s_partial, batch_size);

    CUDA_CHECK(cudaEventRecord (batch_start));

    if (dump_file != NULL && dump_host != NULL) {
      CUDA_CHECK(cudaMemcpy(dump_host, gpu_data, data_size, cudaMemcpyDeviceToHost));
      dump_curve_state_csv(dump_file, "begin", batches_complete, s_partial, this_batch,
                           sigma32, BITS, TPI, dump_host, curves, BITS / 32);
    }

    /* Call CUDA Kernel. */
    assert (kernel != NULL);
    cgbn_stage1_launch(kernel,BLOCK_COUNT,TPB,report, s_num_bits, s_partial, this_batch, gpu_s_bits, gpu_data, curves, sigma_arg, np0);

    s_partial += this_batch;
    batches_complete++;

    /* error report uses managed memory, sync the device and check for cgbn errors */
    CUDA_CHECK(cudaDeviceSynchronize());
    if (report->_error)
      outputf (OUTPUT_ERROR, "\n\nerror: %d\n", report->_error);
    CGBN_CHECK(report);

    if (dump_file != NULL && dump_host != NULL) {
      CUDA_CHECK(cudaMemcpy(dump_host, gpu_data, data_size, cudaMemcpyDeviceToHost));
      dump_curve_state_csv(dump_file, "end", batches_complete, s_partial, this_batch,
                           sigma32, BITS, TPI, dump_host, curves, BITS / 32);
    }

    CUDA_CHECK(cudaEventRecord (stop));
    CUDA_CHECK(cudaEventSynchronize (stop));
    cudaEventElapsedTime (&batch_time, batch_start, stop);
    cudaEventElapsedTime (gputime, global_start, stop);

    // ── Update recent-speed window + ETA ──────────────────────────────────
    if (batch_time > 0.0f && this_batch > 0u) {
        const double speed = static_cast<double>(this_batch) / static_cast<double>(batch_time);
        if (speed_count < SPEED_WINDOW) {
            speed_ring[speed_count++] = speed;
            speed_sum += speed;
        } else {
            speed_sum -= speed_ring[speed_idx];
            speed_ring[speed_idx] = speed;
            speed_sum += speed;
            speed_idx = (speed_idx + 1) % SPEED_WINDOW;
        }
    }
    double remaining_s = 0.0;
    double per_curve_s = 0.0;
    if (speed_count > 0) {
        const double avg_speed = speed_sum / static_cast<double>(speed_count);
        if (avg_speed > 0.0) {
            // Whole-task estimate (全程): s_num_bits at the current average speed.
            const double total_ms = static_cast<double>(s_num_bits) / avg_speed;
            per_curve_s = (curves > 0u)
                              ? (total_ms / static_cast<double>(curves) / 1000.0)
                              : 0.0;
            if (s_num_bits > s_partial) {
                remaining_s = static_cast<double>(s_num_bits - s_partial) / avg_speed / 1000.0;
            }
        }
    }

    // ── Update progress bar ───────────────────────────────────────────────
    {
        double pct =
            (s_num_bits > 0u) ? (100.0 * (double)s_partial / (double)s_num_bits) : 0.0;
        if (pct > 100.0) pct = 100.0;
        const bool final_batch = (s_partial >= s_num_bits);

        if (show_progress) {
            // Interactive terminal: live in-place update every batch.
            print_progress(pct, s_partial, this_batch, per_curve_s,
                           (double)*gputime, remaining_s, false);
        } else if (should_emit_progress_line(batches_complete, (double)*gputime, g_last_progress_line_ms) || final_batch) {
            // Redirected (pipe/file, e.g. under ecm_gui): complete lines, emitted at the
            // ~200 ms batch cadence (or on the legacy batch milestones), plus always the
            // final batch so the log shows 100%.
            print_progress(pct, s_partial, this_batch, per_curve_s,
                           (double)*gputime, remaining_s, true);
            g_last_progress_line_ms = (double)*gputime;
        }
    }

    /* Adjust batch_size to aim for 100ms */
    if (batch_time < 80) {
      batch_size = 11*batch_size/10;
    } else if (batch_time > 120) {
      batch_size = max((uint64_t)100, 9*batch_size / 10);  // MPA-OpenCl: uint64_t literal for Win64 overload resolution
    }
    
    // ========== CHECKPOINT SAVING ==========
    const bool stop_now = ecm_stage1_stop_requested();
    if (stop_now || s_partial>=s_num_bits || (checkpoint_interval_ms > 0 &&
      (*gputime - last_checkpoint_time) >= checkpoint_interval_ms)) {
      // Copy data back from GPU temporarily for checkpoint
      CUDA_CHECK(cudaMemcpy(data, gpu_data, data_size, cudaMemcpyDeviceToHost));
      
      checkpoint_header_t header;
      header.magic = CHECKPOINT_MAGIC;
      header.version = CHECKPOINT_VERSION;
      header.s_partial = s_partial;
      header.s_num_bits = s_num_bits;
      header.batches_complete = batches_complete;
      header.curves = curves;
      /* v4: full 64-bit sigma (param0 needs it; the batch path's sigma fits in 32
         bits by construction) plus the parametrization itself. */
      header.sigma = param0 ? sigma64 : (uint64_t)sigma32;
      header.BITS = BITS;
      header.TPI = TPI;  // Save TPI for kernel selection on reload
      header.gpu_param = param0 ? 0u : (param2 ? 2u : 3u);
      header.reserved = 0;
      header.data_size = data_size;
      header.timestamp = time(NULL);
      
      const int saved = save_checkpoint(ckpt_filename, &header, data, data_size);
      if (saved == ECM_NO_FACTOR_FOUND) last_checkpoint_time = *gputime;
      if (stop_now) {
        outputf(saved == ECM_NO_FACTOR_FOUND ? OUTPUT_NORMAL : OUTPUT_ERROR,
                saved == ECM_NO_FACTOR_FOUND ? "Stage1 stopped: checkpoint saved\n" :
                                               "Stage1 stop requested but checkpoint write failed\n");
        break;
      }
    }
  }

  // ── Mark progress bar complete ──────────────────────────────────────
  if (show_progress) {
      fprintf(stdout, "\n");
      fflush(stdout);
  }

  if (ecm_stage1_stop_requested() && s_partial < s_num_bits) {
    // A partial point is never a completed Stage1 result.
    youpi = ECM_ERROR;
  } else {
    outputf (OUTPUT_VERBOSE, "Copying results back to CPU ...\n");
    CUDA_CHECK(cudaMemcpy(data, gpu_data, data_size, cudaMemcpyDeviceToHost));
    cudaEventElapsedTime (gputime, global_start, stop);
    youpi = process_results(factors, array_found, N, data, BITS, curves, sigma64,
                            words_per_curve, p1_word, p2_word, gpu_param);
  }

  // clean up
  CUDA_CHECK(cudaFree(gpu_s_bits));
  CUDA_CHECK(cudaFree(gpu_data));
  CUDA_CHECK(cgbn_error_report_free(report));
  CUDA_CHECK(cudaEventDestroy (global_start));
  CUDA_CHECK(cudaEventDestroy (batch_start));
  CUDA_CHECK(cudaEventDestroy (stop));

  free(s_bits);
  free(data);
  
  // ========== CHECKPOINT CLEANUP ==========
  // Remove checkpoint file on successful completion
  if (youpi != ECM_ERROR) {
    ecm_stage1_remove_checkpoints(ckpt_filename);
    outputf (OUTPUT_VERBOSE, "Checkpoint file removed\n");
  }

  if (dump_file != NULL) {
    fclose(dump_file);
  }
  if (dump_host != NULL) {
    free(dump_host);
  }

  /* Write back possibly-updated sigma to caller */
  *sigma_ptr = sigma64;

  return youpi;
}

#ifdef __CUDA_ARCH__
  #if __CUDA_ARCH__ < 350
    #error "Unsupported architecture"
  #endif
#endif

#endif  /* _CGBN_STAGE1_CU */
