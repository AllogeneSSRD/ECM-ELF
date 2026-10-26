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
#include "runtime.h"
#include "checkpoint.h"
#include "construction.h"
#include "output.h"
#include "host_policy.h"
using namespace ecm_cuda_mont_host;

#include <cassert>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <memory>

// GMP import must proceed cgbn.h
#include <gmp.h>
#include "../../cuda/arithmetic/cgbn/types.cuh"
#include <cuda.h>

#include "launch.h"  // device templates + per-TPI dispatch seam

#include "../../cuda/runtime/cudacommon.h"
#include "resources.cuh"
#undef CUDA_CHECK
#define CUDA_CHECK(action) ecm_cuda_stage1::check_cuda(action,#action)

// progress-bar colour helpers (ANSI code / reset)
#include "ecm/platform/logging.h"

// MPA-OpenCl port: replaces GMP-ECM's "ecm.h"/"ecm-gpu.h". Provides OUTPUT_*,
// outputf/test_verbose shims, ECM_GPU_* constants, and pulls in the project's
// ECM_* return codes (include/ecm.h).
#include "../../cuda/runtime/cuda_ecm_shim.h"
#include "../../../io/stage1_checkpoint.h"
#include "ecm/stage1/progress.h"

#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <algorithm>
#include <filesystem>

// For %I64u on windows or %llu on linux
#include <cinttypes>
#include <chrono>

#ifdef _WIN32
#include <io.h>      // _isatty, _fileno
#else
#include <unistd.h>  // isatty, fileno
#endif


#include "registry.cuh"
#define CGBN_CHECK(report) ecm_cuda_stage1::check_cgbn(report,__FILE__,__LINE__)

#ifndef ECM_CUDA_ENABLE_PRAC
#define ECM_CUDA_ENABLE_PRAC 0
#endif
#if ECM_CUDA_ENABLE_PRAC
#include "prac/runtime.h"
#endif

static int run_mont_runtime(ecm_stage1_checkpoint::Context& checkpoint, mpz_t *factors, int *array_found,
             const mpz_t N, const mpz_t s,
             uint32_t curves, uint64_t *sigma_ptr,
             unsigned long checkpoint_interval_ms,
             float *gputime, int verbose, int gpu_param, uint64_t B1, uint32_t torsion,
             ecm_cuda_stage1::RunResult &outcome,ecm_cuda_stage1::FactorSink sink,void *context,
             ecm_cuda_stage1::SigmaGenerator generate_sigma,const MontRequestedSpec &requested)
{
  using namespace ecm_cuda_stage1;
  if(!factors || !array_found || !N || !s || !curves || !sigma_ptr || !gputime ||
     mpz_cmp_ui(N,3)<=0 || !mpz_odd_p(N) || mpz_sgn(s)<=0 ||
     *sigma_ptr>UINT64_MAX-(curves-1ull) ||
     (gpu_param!=0 && gpu_param!=2 && gpu_param!=3))
      throw RunError(Failure::Preparation,"invalid CUDA Montgomery Stage1 request");
  if(gpu_param!=0 && (*sigma_ptr>UINT32_MAX || curves>0x100000000ull-*sigma_ptr))
      throw RunError(Failure::Preparation,"CUDA batch parameter range exceeds uint32");
  MontResources resources;
  const ConstructionResults construction{factors,array_found,sink,context,&outcome.degenerate_curve};
  const char *algorithm = stage1_algorithm(requested);
  if (!algorithm) return ECM_ERROR;
  if (strcmp(algorithm, "ladder") != 0) {
#if ECM_CUDA_ENABLE_PRAC
    if (gpu_param != 0 || ECM_MERS_FOLD) {
      outputf(OUTPUT_ERROR, "GPU: resident/PRAC requires param0 and a Montgomery build\n");
      return ECM_ERROR;
    }
    return run_mont_resident(checkpoint, factors, array_found, N, s, curves, sigma_ptr,
                               checkpoint_interval_ms, gputime, B1, torsion,
                               strcmp(algorithm, "prac") == 0,outcome,construction,generate_sigma,requested);
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
  const auto layout=ladder_construction(uint32_t(gpu_param));
  const bool param0 = layout.full_sigma;
  const bool param2 = (gpu_param == 2);
  /* param0 and param2 share the 7-word Suyama buffer layout and kernel family shape
     (they differ only in the step variant the family instantiates). */
  const bool suyama_layout = layout.words==7;
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
  auto *&s_bits=resources.scalar;
  s_bits = allocate_and_set_s_bits(s, &s_num_bits);
  if (s_num_bits >= 4000000000)
      outputf (OUTPUT_ALWAYS, "GPU: Very Large B1! Check magnitute of B1.\n");

  if (s_num_bits >= 100000000)
      outputf (OUTPUT_NORMAL, "GPU: Large B1, S = %lu bits = %d MB\n",
               s_num_bits, s_num_bits >> 23);
  assert( s_bits != NULL );

  auto &global_start=resources.start,&batch_start=resources.batch,&stop=resources.stop;
  auto *&gpu_s_bits=resources.device_scalar;
  auto *&report=resources.report;
  outcome.failure=Failure::Execution;
  CUDA_CHECK(cudaEventCreate (&global_start));
  CUDA_CHECK(cudaEventCreate (&batch_start));
  CUDA_CHECK(cudaEventCreate (&stop));
  CUDA_CHECK(cudaEventRecord (global_start));

  // Copy s_bits
  const size_t s_words=size_t(s_num_bits/32+(s_num_bits%32!=0));
  const auto scalar_bytes=checked_size({sizeof(uint32_t),s_words});
  CUDA_CHECK(cudaMalloc((void **)&gpu_s_bits, scalar_bytes));
  CUDA_CHECK(cudaMemcpy(gpu_s_bits, s_bits, scalar_bytes, cudaMemcpyHostToDevice));

  // create a cgbn_error_report for CGBN to report back errors
  CUDA_CHECK(cgbn_error_report_alloc(&report));


  outcome.failure=Failure::Preparation;
  size_t data_size=0;
  auto *&data=resources.host;
  auto *&gpu_data=resources.device_data;

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

  uint32_t dump_enable = gpu_dump_enabled(requested) ? 1u : 0u;
  auto *&dump_file=resources.dump_file;
  auto *&dump_host=resources.dump;
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
  const auto ckpt_path=checkpoint_record(checkpoint);
  const auto *ckpt_filename=ckpt_path.c_str();
  checkpoint_header_t ckpt_header{};
  std::vector<uint32_t> checkpoint_data;
  bool ckpt_loaded=false;
  auto decode=[&](const std::string &bytes) {
    const auto result=decode_checkpoint(bytes,N,s_num_bits,curves,uint32_t(gpu_param),ckpt_header,checkpoint_data);
    if(result.state==ecm_stage1_checkpoint::RestoreState::Incompatible)
        throw ecm_stage1_checkpoint::Error(result.state,result.error);
    return result.restored();
  };
  checkpoint.validate_pending([&](const std::vector<std::string> &records,std::vector<uint64_t> &sigmas) {
    if(records.size()!=curves || sigmas.size()!=curves)return false;
    for(size_t i=1;i<records.size();++i)if(!records[i].empty())return false;
    if(records[0].empty())return true;
    if(!decode(records[0]))return false;
    uint32_t resolved_tpi=0;
    const auto available=param2?cgbn_stage1_kernel_param2_dispatch(ckpt_header.BITS,&resolved_tpi):
        param0?cgbn_stage1_kernel_suyama_dispatch(ckpt_header.BITS,&resolved_tpi):
               cgbn_stage1_kernel_dispatch(ckpt_header.BITS,&resolved_tpi);
    if(!available)throw ecm_stage1_checkpoint::Error(ecm_stage1_checkpoint::RestoreState::Incompatible,
        "checkpoint kernel tier is unavailable in this build");
    for(uint32_t i=0;i<curves;++i)if(sigmas[i] && sigmas[i]!=ckpt_header.sigma+i)return false;
    for(uint32_t i=0;i<curves;++i)sigmas[i]=ckpt_header.sigma+i;
    ckpt_loaded=true;return true;
  });
  if(!ckpt_loaded && !ckpt_path.empty())for(const auto &candidate:checkpoint.candidates(ckpt_path)) {
    std::string bytes;
    const auto read=checkpoint.read_checkpoint_blob(candidate,bytes);
    if(read==ecm_stage1_checkpoint::BlobRead::IoFailure)
        throw ecm_stage1_checkpoint::Error(ecm_stage1_checkpoint::RestoreState::IoFailure,"cannot read Montgomery checkpoint");
    if(read==ecm_stage1_checkpoint::BlobRead::Missing || !decode(bytes))continue;
    if(candidate!=ckpt_path)checkpoint.mark_bad_primary(ckpt_path);
    ckpt_loaded=true;break;
  }
  if(ckpt_loaded) {
    outputf(OUTPUT_NORMAL,"Resuming from checkpoint: %.1f%% complete (s_partial=%lu/%lu)\n",
        100.0*ckpt_header.s_partial/s_num_bits,ckpt_header.s_partial,s_num_bits);
    sigma64=ckpt_header.sigma;BITS=ckpt_header.BITS;TPI=ckpt_header.TPI;
    data_size=ckpt_header.data_size;
    data=static_cast<uint32_t *>(malloc(data_size));if(!data)throw std::bad_alloc();
    memcpy(data,checkpoint_data.data(),data_size);std::vector<uint32_t>().swap(checkpoint_data);
    s_partial=ckpt_header.s_partial;batches_complete=ckpt_header.batches_complete;
  } else {
    uint64_t saved=0;
    if(checkpoint.saved_sigma(0,saved))sigma64=saved;
    else if(generate_sigma)sigma64=generate_sigma(curves,gpu_param);
  }
  if(sigma64>UINT64_MAX-(curves-1ull) ||
     (gpu_param!=0 && (sigma64>UINT32_MAX || curves>0x100000000ull-sigma64)))
      throw RunError(Failure::Preparation,"CUDA sigma range overflow");
  *sigma_ptr=sigma64;sigma32=uint32_t(sigma64);
  for(uint32_t i=0;i<curves;++i)
      if(checkpoint.sigma(i,sigma64+i)!=sigma64+i)
          throw RunError(Failure::Preparation,"nonconsecutive Montgomery checkpoint sigmas");

  // If no checkpoint, proceed with normal initialization
  if (!ckpt_loaded) {
  std::string existing_checkpoint;
  if (!checkpoint.path().empty() && checkpoint.read_blob(ckpt_filename,existing_checkpoint)) {
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
      BLOCK_COUNT = (size_t(curves) + IPB - 1) / IPB;

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

  youpi = verify_size_of_n(N, BITS,CARRY_BITS);
  if (youpi != ECM_NO_FACTOR_FOUND) {
    return youpi;
  }

  /* Consistency check that struct cgbn_mem_t is byte aligned without extra fields. */
  assert( sizeof(cgbn_mem_t<512>) == 512/8 );
  assert( sizeof(cgbn_mem_t<1024>) == 1024/8 );
  
  if (!ckpt_loaded) {
    data = param2 ? set_p_2p_param2(N, curves, sigma32, BITS, &data_size,&construction)
                  : param0 ? set_p_2p_suyama(N, curves, sigma64, BITS, &data_size,&construction)
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
    BLOCK_COUNT = (size_t(curves) + IPB - 1) / IPB;
    outputf(OUTPUT_VERBOSE, "Checkpoint: restored BITS=%d, TPI=%d, BLOCK_COUNT=%lu\n", BITS, TPI, BLOCK_COUNT);
  }

  if(ckpt_loaded && (param0 || param2)) {
    size_t initial_size=0;
    std::unique_ptr<uint32_t,decltype(&free)> initial(param0?
        set_p_2p_suyama(N,curves,sigma64,BITS,&initial_size,&construction):
        set_p_2p_param2(N,curves,sigma32,BITS,&initial_size,&construction),&free);
  }

  /* Buffer layout of this run: 5 words/curve for param3 (N, aX, aZ, bX, bZ) and
     7 for param0 (N, a24, xdiff, aX, aZ, bX, bZ).  Derived, not hard-coded, so a
     checkpoint resume is validated against it in both directions. */
  const uint32_t words_per_curve = layout.words;
  /* param0 AND param2 use the 7-word layout (N, a24, xdiff, aX, aZ, bX, bZ);
     using the param0 ternary here made param2 decode its result from word 1
     (= a24) instead of word 3 (= aX), which is what broke the gmp-ecm comparison. */
  const int      p1_word = int(layout.first_x);      /* X of P_a */
  const int      p2_word = int(layout.second_x);      /* X of P_b */
  if (data_size != checked_size({size_t(words_per_curve),size_t(curves),size_t(BITS/8)})) {
    outputf(OUTPUT_ERROR,
            "GPU: internal error: curve buffer size %zu does not match the %u-word/curve layout\n",
            data_size, words_per_curve);
    return ECM_ERROR;
  }

  const MontResolvedKernel resolved{kernel,BITS,uint32_t(TPI),uint32_t(TPB),BLOCK_COUNT,data_size};

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

  outcome.failure=Failure::Execution;
  // Copy data
  outputf (OUTPUT_VERBOSE, "Copying %'lu bytes of curves data to GPU\n", data_size);
  CUDA_CHECK(cudaMalloc((void **)&gpu_data,resolved.state_bytes));
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
  
  double last_progress_line_ms=-1.0e9;
  /* Track time for checkpoint saving */
  auto last_checkpoint_time=std::chrono::steady_clock::now();
  outcome.total=s_num_bits;outcome.cursor=s_partial;
  outcome.snapshot_cursor=s_partial;outcome.durable_cursor=ckpt_loaded?s_partial:0;

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

  ecm_stage1_gpu_progress::SpeedWindow speed;

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
    cgbn_stage1_launch(resolved.entry,resolved.blocks,resolved.tpb,report, s_num_bits, s_partial, this_batch, gpu_s_bits, gpu_data, curves, sigma_arg, np0);
    CUDA_CHECK(cudaGetLastError());

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
    CUDA_CHECK(cudaEventElapsedTime (&batch_time, batch_start, stop));
    CUDA_CHECK(cudaEventElapsedTime (gputime, global_start, stop));

    // Only synchronized successful work enters the live speed window.
    outcome.cursor=s_partial;
    speed.add(this_batch,batch_time);
    const double remaining_s=speed.remaining(s_num_bits,s_partial);
    const double per_curve_s=speed.per_curve(s_num_bits,curves);

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
        } else if (should_emit_progress_line(batches_complete, (double)*gputime, last_progress_line_ms) || final_batch) {
            // Redirected (pipe/file, e.g. under ecm_gui): complete lines, emitted at the
            // ~200 ms batch cadence (or on the legacy batch milestones), plus always the
            // final batch so the log shows 100%.
            print_progress(pct, s_partial, this_batch, per_curve_s,
                           (double)*gputime, remaining_s, true);
            last_progress_line_ms = (double)*gputime;
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
      std::chrono::steady_clock::now()-last_checkpoint_time >= std::chrono::milliseconds(checkpoint_interval_ms))) {
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
      
      outcome.snapshot_cursor=s_partial;
      const bool saved=ckpt_path.empty() || save_checkpoint(checkpoint, ckpt_filename,&header,data,data_size);
      if(!saved)throw RunError(Failure::Persistence,"Montgomery checkpoint write failed");
      if(!ckpt_path.empty())outcome.durable_cursor=s_partial;
      last_checkpoint_time=std::chrono::steady_clock::now();
      if (stop_now) {
        if(s_partial<s_num_bits)outcome.termination=Termination::Stopped;
        outputf(saved ? OUTPUT_NORMAL : OUTPUT_ERROR,
                saved ? (ckpt_path.empty()?"Stage1 stopped: no checkpoint configured, progress is not recoverable\n":"Stage1 stopped: checkpoint saved\n") :
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
    outcome.snapshot_cursor=s_partial;
    CUDA_CHECK(cudaEventElapsedTime (gputime, global_start, stop));
    youpi = process_results(factors, array_found, N, data, BITS, curves, sigma64,
                            words_per_curve, p1_word, p2_word, gpu_param,true,&outcome.degenerate_curve);
  }

  // Task finalization owns checkpoint cleanup after all result publication.
  outcome.cursor=s_partial;

  /* Write back possibly-updated sigma to caller */
  *sigma_ptr = sigma64;

  return youpi;
}

ecm_cuda_stage1::RunResult ecm_cuda_stage1::execute_mont_batch(ecm_stage1_checkpoint::Context& checkpoint, mpz_t *factors,int *found,
    mpz_srcptr n,mpz_srcptr scalar,uint32_t curves,uint64_t *sigma,unsigned long checkpoint_ms,
    float *milliseconds,int verbose,int param,uint64_t b1,uint32_t torsion,FactorSink sink,void *context,SigmaGenerator generate_sigma,const ecm_cuda_mont_host::MontRequestedSpec *requested) {
    RunResult result;
    try {
        MontRequestedSpec fallback;if(!requested){fallback=capture_mont_request();requested=&fallback;}
        result.code=run_mont_runtime(checkpoint, factors,found,n,scalar,curves,sigma,checkpoint_ms,
                                    milliseconds,verbose,param,b1,torsion,result,sink,context,generate_sigma,*requested);
        if(result.code!=-1){result.termination=Termination::Completed;result.failure=Failure::None;}
        else if(result.termination==Termination::Stopped || result.termination==Termination::SampleLimit)
            result.failure=Failure::None;
    } catch(const RunError &e) {
        result.code=-1;result.termination=Termination::Failed;result.failure=e.failure;result.error=e.what();
    } catch(const ecm_stage1_checkpoint::Error &e) {
        result.code=-1;result.termination=Termination::Failed;result.error=e.what();
        result.failure=e.state==ecm_stage1_checkpoint::RestoreState::IoFailure?Failure::Persistence:Failure::Preparation;
    } catch(const std::exception &e) {
        result.code=-1;result.termination=Termination::Failed;result.error=e.what();
    }
    if(!result.error.empty())outputf(OUTPUT_ERROR,"GPU: Stage1 failed: %s\n",result.error.c_str());
    return result;
}

// ABI compatibility for existing standalone probes and backend hooks. Production
// workflows use the typed runtime result directly.
int cgbn_ecm_stage1(mpz_t *factors,int *found,mpz_srcptr n,mpz_srcptr scalar,
    uint32_t curves,uint64_t *sigma,unsigned long checkpoint_ms,float *milliseconds,
    int verbose,int param,uint64_t b1,uint32_t torsion) {
    ecm_stage1_checkpoint::Context checkpoint;
    return ecm_cuda_stage1::execute_mont_batch(checkpoint, factors,found,n,scalar,curves,sigma,
        checkpoint_ms,milliseconds,verbose,param,b1,torsion).code;
}

#ifdef __CUDA_ARCH__
  #if __CUDA_ARCH__ < 350
    #error "Unsupported architecture"
  #endif
#endif

#endif  /* _CGBN_STAGE1_CU */
