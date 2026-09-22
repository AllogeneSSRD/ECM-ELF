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

#include "cgbn_stage1_cuda.h"

#include <cassert>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

// GMP import must proceed cgbn.h
#include <gmp.h>
#include <cgbn.h>
#include <cuda.h>

#include "cgbn_stage1_kernel.h"  // device templates + per-TPI dispatch seam

#include "cudacommon.h"

// progress-bar colour helpers (ANSI code / reset)
#include "opencl_ecm_log.h"

// MPA-OpenCl port: replaces GMP-ECM's "ecm.h"/"ecm-gpu.h". Provides OUTPUT_*,
// outputf/test_verbose shims, ECM_GPU_* constants, and pulls in the project's
// ECM_* return codes (include/ecm.h).
#include "cuda_ecm_shim.h"

#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <algorithm>

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
static bool stdout_is_tty() {
#ifdef _WIN32
    return _isatty(_fileno(stdout)) != 0;
#else
    return isatty(fileno(stdout)) != 0;
#endif
}

static void print_progress(double pct, uint64_t s_partial, uint64_t this_batch,
                           double per_curve_s, double elapsed_ms, double remaining_s,
                           bool newline) {
    const int bar_width = 40;
    int filled = (int)(bar_width * (pct / 100.0));
    if (filled < 0) filled = 0;
    if (filled > bar_width) filled = bar_width;

    char bar[bar_width + 1];
    int i;
    for (i = 0; i < bar_width; i++) {
        bar[i] = (i < filled) ? '=' : ' ';
    }
    bar[bar_width] = '\0';
    if (filled > 0 && filled < bar_width) {
        bar[filled - 1] = '>';
    }

    const double elapsed_s = elapsed_ms / 1000.0;

    if (newline) {
        // Redirected / log mode: a full timestamped line, mirrored to screen.log
        // via the outputf → ecm_ts_vfprintf path. No ANSI colour in the log.
        outputf(OUTPUT_ALWAYS,
                "GPU: [%s] %.1f%%  %llu, +%llu bits (~%.2f s/curve)  elapsed %.1fs  remaining %.1fs\n",
                bar, pct,
                (unsigned long long)s_partial, (unsigned long long)this_batch,
                per_curve_s, elapsed_s, remaining_s);
    } else {
        // Interactive terminal: in-place update, coloured (ANSI; colour is
        // configurable via ecm.ini progress_color).
        fprintf(stdout,
                "\r%sGPU: [%s] %.1f%%  %llu, +%llu bits (~%.2f s/curve)  elapsed %.1fs  remaining %.1fs%s",
                ecm_log_progress_color_code(),
                bar, pct,
                (unsigned long long)s_partial, (unsigned long long)this_batch,
                per_curve_s, elapsed_s, remaining_s,
                ecm_log_progress_color_reset());
        fflush(stdout);
    }
}

// How often to emit a full (newline-terminated) progress line when stdout is
// redirected to a file/pipe. work_manager.ps1 tails the child's stdout file by
// complete lines, so a `\r` in-place update would never be flushed to the log.
static bool emit_progress_line(int n) {
    return ((n < 3) ||
            (n < 30 && n % 10 == 0) ||
            (n < 500 && n % 100 == 0) ||
            (n < 5000 && n % 1000 == 0) ||
            (n % 10000 == 0));
}


// Checkpoint configuration
#define CHECKPOINT_MAGIC 0x45555047  // EPUG -> "GPUE" in hex (GPU ECM)
// 小端格式，magic number 0x45555047 在内存中表示为 "GPUE"，用于验证 checkpoint 文件的正确性
#define CHECKPOINT_VERSION 3         // Incremented to invalidate old checkpoint files

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
void to_mpz(mpz_t r, const uint32_t *x, uint32_t count) {
  mpz_import (r, count, -1, sizeof(uint32_t), 0, 0, x);
}

static
void from_mpz(const mpz_t s, uint32_t *x, uint32_t count) {
  size_t words;

  if(mpz_sizeinbase (s, 2) > count * 32) {
    fprintf (stderr, "from_mpz failed -- result does not fit\n");
    exit(EXIT_FAILURE);
  }

  mpz_export (x, &words, -1, sizeof(uint32_t), 0, 0, s);
  while(words<count)
    x[words++]=0;
}



static
int findfactor(mpz_t factor, const mpz_t N, const mpz_t x_final, const mpz_t z_final) {
    // XXX: combine / refactor logic with cudawrapper.c findfactor

    /* Check if factor found */

    bool inverted = mpz_invert(factor, z_final, N);    // aZ ^ (N-2) % N
    if (inverted) {
        mpz_mul(factor, x_final, factor);         // aX * aZ^-1
        mpz_mod(factor, factor, N);             // "Residual"
        return ECM_NO_FACTOR_FOUND;
    }

    mpz_gcd(factor, z_final, N);
    return ECM_FACTOR_FOUND_STEP1;
}


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


static
uint32_t find_np0(const mpz_t N) {
  uint32_t np0;
  mpz_t temp;
  mpz_init(temp);
  mpz_ui_pow_ui(temp, 2, 32);
  // NB: must not put mpz_invert() inside assert() — NDEBUG (Release) would drop
  // the call and leave temp = 2^32, giving np0 = 0 on this platform.
  int inv_ok = mpz_invert(temp, N, temp);
  assert(inv_ok);
  (void)inv_ok;
  np0 = -mpz_get_ui(temp);
  mpz_clear(temp);
  return np0;
}


static
uint32_t* allocate_and_set_s_bits(const mpz_t s, uint64_t *nbits) {
  uint64_t num_bits = *nbits = mpz_sizeinbase (s, 2);

  uint64_t allocated = (num_bits + 31) / 32;
  uint32_t *s_bits = (uint32_t*) malloc (sizeof(uint32_t) * allocated);

  uint64_t countp;
  mpz_export (s_bits, &countp, -1, sizeof(uint32_t), 0, 0, s);
  assert (countp == allocated);

  return s_bits;
}


static
uint32_t* set_p_2p(const mpz_t N,
                   uint32_t curves, uint32_t sigma,
                   uint32_t BITS, size_t *data_size) {
  /**
   * Store 5 numbers per curve:
   * N, P_a (x, z), P_b (x, z)
   *
   * P_a is initialized with (2, 1)
   * P_b (for the doubled terms) is initialized with (9, 64 * d + 8)
   */

  const size_t limbs_per = BITS/32;
  *data_size = 5 * curves * limbs_per * sizeof(uint32_t);
  uint32_t *data = (uint32_t*) malloc(*data_size);
  uint32_t *datum = data;

  mpz_t x;
  mpz_init(x);
  for(int index = 0; index < curves; index++) {
      // d = (sigma / 2^32) mod N BUT 2^32 handled by special_mul_ui32
      uint32_t d = sigma + index;

      // Modulo (N)
      from_mpz(N, datum + 0 * limbs_per, BITS/32);

      // P1 (X, Z)
      mpz_set_ui(x, 2);
      from_mpz(x, datum + 1 * limbs_per, BITS/32);
      mpz_set_ui(x, 1);
      from_mpz(x, datum + 2 * limbs_per, BITS/32);

      // 2P = P2 (X, Z)
      // P2_x = 9
      mpz_set_ui(x, 9);
      from_mpz(x, datum + 3 * limbs_per, BITS/32);

      // d = sigma * mod_inverse(2 ** 32, N)
      mpz_ui_pow_ui(x, 2, 32);
      mpz_invert(x, x, N);
      mpz_mul_ui(x, x, d);
      // P2_x = 64 * d + 8;
      mpz_mul_ui(x, x, 64);
      mpz_add_ui(x, x, 8);
      mpz_mod(x, x, N);

      outputf (OUTPUT_TRACE, "sigma %d => P2_y: %Zd\n", d, x);
      from_mpz(x, datum + 4 * limbs_per, BITS/32);
      datum += 5 * limbs_per;
  }
  mpz_clear(x);
  return data;
}


static
int process_results(mpz_t *factors, int *array_found,
                    const mpz_t N,
                    const uint32_t *data, uint32_t cgbn_bits,
                    int curves, uint32_t sigma) {
  mpz_t x_final, z_final, modulo;
  mpz_init(modulo);
  mpz_init(x_final);
  mpz_init(z_final);

  const uint32_t limbs_per = cgbn_bits / 32;

  int youpi = ECM_NO_FACTOR_FOUND;
  int errors = 0;
  for(size_t i = 0; i < curves; i++) {
    const uint32_t *datum = data + (5 * i * limbs_per);;

    if (test_verbose (OUTPUT_TRACE) && i == 0) {
      to_mpz(modulo, datum + 0 * limbs_per, limbs_per);
      outputf (OUTPUT_TRACE, "index: 0 modulo: %Zd\n", modulo);

      to_mpz(x_final, datum + 1 * limbs_per, limbs_per);
      to_mpz(z_final, datum + 2 * limbs_per, limbs_per);
      outputf (OUTPUT_TRACE, "index: 0 pA: (%Zd, %Zd)\n", x_final, z_final);

      to_mpz(x_final, datum + 3 * limbs_per, limbs_per);
      to_mpz(z_final, datum + 4 * limbs_per, limbs_per);
      outputf (OUTPUT_TRACE, "index: 0 pB: (%Zd, %Zd)\n", x_final, z_final);
    }

    // Make sure we were testing the right number.
    to_mpz(modulo, datum + 0 * limbs_per, limbs_per);
    assert(mpz_cmp(modulo, N) == 0);

    to_mpz(x_final, datum + 1 * limbs_per, limbs_per);
    to_mpz(z_final, datum + 2 * limbs_per, limbs_per);

    /* Very suspicious for (x_final, z_final) to match (x_0, z_0) == (2, 1)
     * Can happen when
     * 1. block calculation performed incorrectly (and some blocks not run)
     * 2. Kernel didn't run because not enough register
     * 3. nvcc links old version of kernel when something changed
     */
    if (mpz_cmp_ui (x_final, 2) == 0 && mpz_cmp_ui (z_final, 1) == 0) {
      errors += 1;
      if (errors < 10 || errors % 100 == 1)
        outputf (OUTPUT_ERROR, "GPU: curve %d didn't compute?\n", i);
    }

    array_found[i] = findfactor(factors[i], N, x_final, z_final);
    if (array_found[i] != ECM_NO_FACTOR_FOUND) {
      youpi = array_found[i];
      outputf (OUTPUT_NORMAL, "GPU: factor %Zd found in Step 1 with curve %ld (-sigma %d:%lu)\n",
          factors[i], i, ECM_PARAM_BATCH_32BITS_D, sigma + i);
    }
  }

  mpz_clear(modulo);
  mpz_clear(x_final);
  mpz_clear(z_final);

#ifdef IS_DEV_BUILD
  if (errors)
        outputf (OUTPUT_ERROR, "Had %d errors. Try `make clean; make` or reducing TPB_DEFAULT\n",
            errors);
#endif

  if (errors > 2)
      return ECM_ERROR;

  return youpi;
}


/**
 * Checkpoint structure containing state information
 */
typedef struct {
  uint32_t magic;            // Magic number for validation
  uint32_t version;          // Checkpoint format version
  uint64_t s_partial;        // Current bit progress
  uint64_t s_num_bits;       // Total bits to process
  int32_t batches_complete;  // Number of completed batches
  uint32_t curves;           // Number of curves
  uint32_t sigma;            // Starting sigma value
  uint32_t BITS;             // Kernel bit size
  uint32_t TPI;              // Threads per instance (needed for kernel selection)
  size_t data_size;          // Size of GPU data
  time_t timestamp;          // When checkpoint was created
} checkpoint_header_t;

/**
 * Get checkpoint filename for given N (more readable format)
 * Uses bit length and first/last few hex chars as identifier
 */
static
char* get_checkpoint_filename(const mpz_t N) {
  static char filename[512];
  
  size_t nbits = mpz_sizeinbase(N, 2);
  
  // Get first and last few chars of N in hex for unique identifier.
  // Use GMP-allocated buffer to avoid stack overflow for large N.
  char *N_str = mpz_get_str(NULL, 16, N);
  if (N_str == NULL) {
    snprintf(filename, sizeof(filename), ".ecm_ckpt_%zu_alloc_fail.dat", nbits);
    return filename;
  }

  size_t len = strlen(N_str);
  char first_hex[16] = {0};
  char last_hex[16] = {0};
  
  // First 8 chars (or less if N is small)
  strncpy(first_hex, N_str, (len >= 8) ? 8 : len);
  // Last 8 chars
  if (len > 8) {
    strncpy(last_hex, N_str + len - 8, 8);
  }
  
  // Format: .ecm_ckpt_<nbits>_<first8>_<last8>.dat
  // Example: .ecm_ckpt_127_7fffffff_ffffffff.dat (much more readable than full hex)
  if (len > 8 && last_hex[0] != '\0') {
    snprintf(filename, sizeof(filename), ".ecm_ckpt_%zu_%s_%s.dat", nbits, first_hex, last_hex);
  } else {
    snprintf(filename, sizeof(filename), ".ecm_ckpt_%zu_%s.dat", nbits, first_hex);
  }

  free(N_str);
  
  return filename;
}

/**
 * Save checkpoint to file
 */
static
int save_checkpoint(const char *filename, 
                    const checkpoint_header_t *header,
                    const uint32_t *data, 
                    size_t data_size) {
  FILE *f = fopen(filename, "wb");
  if (!f) {
    outputf(OUTPUT_ALWAYS, "Warning: Could not open checkpoint file '%s' for writing\n", filename);
    return ECM_ERROR;
  }
  
  // Write header
  if (fwrite(header, sizeof(checkpoint_header_t), 1, f) != 1) {
    outputf(OUTPUT_ERROR, "Error writing checkpoint header\n");
    fclose(f);
    return ECM_ERROR;
  }
  
  // Write data
  if (fwrite(data, 1, data_size, f) != data_size) {
    outputf(OUTPUT_ERROR, "Error writing checkpoint data\n");
    fclose(f);
    return ECM_ERROR;
  }
  
  fclose(f);
  outputf(OUTPUT_VERBOSE, "Checkpoint saved: s_partial=%lu/%lu (%.1f%%)\n", 
          header->s_partial, header->s_num_bits, 
          100.0 * header->s_partial / header->s_num_bits);
  return ECM_NO_FACTOR_FOUND;
}

/**
 * Load checkpoint from file if it exists
 * Returns 0 if successful, -1 if file doesn't exist or is invalid
 */
static
int load_checkpoint(const char *filename,
                    checkpoint_header_t *header,
                    uint32_t **data_ptr,
                    size_t *data_size_ptr) {
  FILE *f = fopen(filename, "rb");
  if (!f) {
    // File doesn't exist - this is normal on first run
    return -1;
  }
  
  // Read and validate header
  checkpoint_header_t temp_header;
  if (fread(&temp_header, sizeof(checkpoint_header_t), 1, f) != 1) {
    outputf(OUTPUT_ALWAYS, "Warning: Could not read checkpoint header\n");
    fclose(f);
    return -1;
  }
  
  if (temp_header.magic != CHECKPOINT_MAGIC) {
    outputf(OUTPUT_ALWAYS, "Warning: Checkpoint file has invalid magic number\n");
    fclose(f);
    return -1;
  }
  
  if (temp_header.version != CHECKPOINT_VERSION) {
    outputf(OUTPUT_ALWAYS, "Warning: Checkpoint version mismatch (expected %d, got %d)\n", 
            CHECKPOINT_VERSION, temp_header.version);
    fclose(f);
    return -1;
  }
  
  // Allocate memory for data
  uint32_t *data = (uint32_t*) malloc(temp_header.data_size);
  if (!data) {
    outputf(OUTPUT_ERROR, "Error: Could not allocate memory for checkpoint data\n");
    fclose(f);
    return -1;
  }
  
  // Read data
  if (fread(data, 1, temp_header.data_size, f) != temp_header.data_size) {
    outputf(OUTPUT_ALWAYS, "Warning: Could not read checkpoint data completely\n");
    free(data);
    fclose(f);
    return -1;
  }
  
  fclose(f);
  
  *header = temp_header;
  *data_ptr = data;
  *data_size_ptr = temp_header.data_size;
  
  time_t now = time(NULL);
  time_t age = now - temp_header.timestamp;
  outputf(OUTPUT_NORMAL, "Checkpoint loaded: s_partial=%lu/%lu (%.1f%%), age=%ld seconds\n", 
          temp_header.s_partial, temp_header.s_num_bits,
          100.0 * temp_header.s_partial / temp_header.s_num_bits, age);
  
  return ECM_NO_FACTOR_FOUND;
}

static
FILE* open_dump_csv(const char *filename) {
  FILE *f = fopen(filename, "w");
  if (!f) {
    outputf(OUTPUT_ALWAYS, "Warning: Could not open dump file '%s' for writing\n", filename);
    return NULL;
  }

  fprintf(f,
          "stage,batch_index,s_partial,batch_size,sigma,curve_index,BITS,TPI,word0,word1,word2,word3,word4\n");
  return f;
}

static
void dump_curve_state_csv(FILE *f,
                          const char *stage,
                          int batch_index,
                          uint64_t s_partial,
                          uint64_t batch_size,
                          uint32_t sigma,
                          uint32_t BITS,
                          uint32_t TPI,
                          const uint32_t *data,
                          uint32_t curves,
                          size_t limbs_per) {
  if (!f) {
    return;
  }

  for (uint32_t i = 0; i < curves; i++) {
    const uint32_t *datum = data + (5 * i * limbs_per);
    fprintf(f, "%s,%d," "%" PRIu64 "," "%" PRIu64 ",%u,%u,%u,%u,",
            stage, batch_index, s_partial, batch_size, sigma, i, BITS, TPI);
    for (uint32_t word = 0; word < 5; word++) {
      if (word != 0) {
        fputc(',', f);
      }
      fputs("0x", f);
      const uint32_t *field = datum + (word * limbs_per);
      for (size_t limb = limbs_per; limb > 0; limb--) {
        fprintf(f, "%08x", field[limb - 1]);
      }
    }
    fputc('\n', f);
  }

  fflush(f);
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

int cgbn_ecm_stage1(mpz_t *factors, int *array_found,
             const mpz_t N, const mpz_t s,
             uint32_t curves, uint32_t *sigma_ptr,
             unsigned long checkpoint_interval_ms,
             float *gputime, int verbose)
{
  uint32_t sigma = *sigma_ptr;
  assert( sigma > 0 );
  assert( ((uint64_t) sigma + curves) <= 0xFFFFFFFF ); // no overflow

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
  available_kernels.push_back((uint32_t)cgbn_params_128::BITS);
  available_kernels.push_back((uint32_t)cgbn_params_192::BITS);
  available_kernels.push_back((uint32_t)cgbn_params_256::BITS);
  available_kernels.push_back((uint32_t)cgbn_params_384::BITS);
  available_kernels.push_back((uint32_t)cgbn_params_small::BITS);
  available_kernels.push_back((uint32_t)cgbn_params_768::BITS);
  available_kernels.push_back((uint32_t)cgbn_params_medium::BITS);

  #ifdef IS_DEV_BUILD
    outputf(OUTPUT_ALWAYS, "Warning: Dev buils, only support N<1024.\n");
    outputf(OUTPUT_ALWAYS, "Warning: Using dev build with only 2 kernels. Consider adding more kernels for better performance on large inputs.\n");
  #endif

  #ifndef IS_DEV_BUILD
  /**
   * TPI and BITS have to be set at compile time. Adding multiple cgbn_params
   * (and their associated kernels) allows for better dynamic selection based
   * on the size of N (e.g. N < 1024, N < 2048, N < 4096) but increase compile
   * time and binary size. A few reasonable sizes are included and a verbose
   * warning is printed when a particular N might benefit from a custom sized
   * kernel.
   *
   * BITS规则: 必须是32的倍数，建议间隔256~512
   * TPI规则: N>512用8, N>2048用16, N>8192用32
   */

  // TPI=8 kernels (for 512-2048 bits)
  available_kernels.push_back((uint32_t)cgbn_params_1280::BITS);
  available_kernels.push_back((uint32_t)cgbn_params_1536::BITS);
  available_kernels.push_back((uint32_t)cgbn_params_1792::BITS);
  available_kernels.push_back((uint32_t)cgbn_params_2048::BITS);

  // TPI=16 kernels (for 2560-8192 bits, 256 interval)
  available_kernels.push_back((uint32_t)cgbn_params_2560::BITS);
  // available_kernels.push_back((uint32_t)cgbn_params_2816::BITS);
  available_kernels.push_back((uint32_t)cgbn_params_3072::BITS);
  // available_kernels.push_back((uint32_t)cgbn_params_3328::BITS);
  available_kernels.push_back((uint32_t)cgbn_params_3584::BITS);
  // available_kernels.push_back((uint32_t)cgbn_params_3840::BITS);
  available_kernels.push_back((uint32_t)cgbn_params_4096::BITS);
  // available_kernels.push_back((uint32_t)cgbn_params_4352::BITS);
  available_kernels.push_back((uint32_t)cgbn_params_4608::BITS);
  // available_kernels.push_back((uint32_t)cgbn_params_4864::BITS);
  available_kernels.push_back((uint32_t)cgbn_params_5120::BITS);
  // available_kernels.push_back((uint32_t)cgbn_params_5376::BITS);
  available_kernels.push_back((uint32_t)cgbn_params_5632::BITS);
  // available_kernels.push_back((uint32_t)cgbn_params_5888::BITS);
  available_kernels.push_back((uint32_t)cgbn_params_6144::BITS);
  // available_kernels.push_back((uint32_t)cgbn_params_6400::BITS);
  available_kernels.push_back((uint32_t)cgbn_params_6656::BITS);
  // available_kernels.push_back((uint32_t)cgbn_params_6912::BITS);
  available_kernels.push_back((uint32_t)cgbn_params_7168::BITS);
  // available_kernels.push_back((uint32_t)cgbn_params_7424::BITS);
  available_kernels.push_back((uint32_t)cgbn_params_7680::BITS);
  // available_kernels.push_back((uint32_t)cgbn_params_7936::BITS);
  available_kernels.push_back((uint32_t)cgbn_params_8192::BITS);

  // TPI=32 kernels (for 10240+ bits, 512 interval for better optimization)
  // 12288-16384 gap is critical: 9820s->16950s (73% increase!)
  available_kernels.push_back((uint32_t)cgbn_params_9216::BITS);
  available_kernels.push_back((uint32_t)cgbn_params_10240::BITS);
  available_kernels.push_back((uint32_t)cgbn_params_11264::BITS);
  available_kernels.push_back((uint32_t)cgbn_params_12288::BITS);
  available_kernels.push_back((uint32_t)cgbn_params_13312::BITS);
  available_kernels.push_back((uint32_t)cgbn_params_14336::BITS);
  available_kernels.push_back((uint32_t)cgbn_params_15360::BITS);
  available_kernels.push_back((uint32_t)cgbn_params_16384::BITS);

#endif

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
  
  // Try to load checkpoint
  if (load_checkpoint(ckpt_filename, &ckpt_header, &ckpt_data, &ckpt_data_size) == ECM_NO_FACTOR_FOUND) {
    // Validate checkpoint compatibility
    // Only check curves and s_num_bits (sigma is a range in real usage)
    outputf(OUTPUT_TRACE, "Checkpoint validation: curves=%u(expect %u), s_num_bits=%lu(expect %lu)\n",
            ckpt_header.curves, curves, ckpt_header.s_num_bits, s_num_bits);
    
    if (ckpt_header.curves == curves && 
        ckpt_header.s_num_bits == s_num_bits) {
      
      outputf(OUTPUT_NORMAL, "Resuming from checkpoint: %.1f%% complete (s_partial=%lu/%lu)\n",
              100.0 * ckpt_header.s_partial / ckpt_header.s_num_bits,
              ckpt_header.s_partial, ckpt_header.s_num_bits);

      if (sigma != ckpt_header.sigma) {
        outputf(OUTPUT_VERBOSE, "Checkpoint sigma overrides current sigma: %u -> %u\n",
                sigma, ckpt_header.sigma);
      }
      
      ckpt_loaded = 1;
      sigma = ckpt_header.sigma;
      BITS = ckpt_header.BITS;
      TPI = ckpt_header.TPI;  // Restore TPI from checkpoint
      data_size = ckpt_header.data_size;
      data = ckpt_data;
      s_partial = ckpt_header.s_partial;
      batches_complete = ckpt_header.batches_complete;
    } else {
      outputf(OUTPUT_NORMAL, "Checkpoint parameters mismatch (curves or s_num_bits differ), starting fresh\n");
      if (ckpt_data) free(ckpt_data);
      ckpt_loaded = 0;
    }
  }
  
  // If no checkpoint, proceed with normal initialization
  if (!ckpt_loaded) {
  for (int k_i = 0; k_i < available_kernels.size(); k_i++) {
    uint32_t kernel_bits = available_kernels[k_i];
    if (kernel_bits >= n_log2 + CARRY_BITS) {
      BITS = kernel_bits;
      assert( BITS % 32 == 0 );

      /* Resolve the kernel function pointer via the per-TPI dispatch TUs. */
      uint32_t tpi_u32 = 0;
      kernel = cgbn_stage1_kernel_dispatch(BITS, &tpi_u32);
      if (kernel == nullptr) {
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
      outputf (OUTPUT_ERROR, "No available CGBN Kernel large enough to process N(%d bits)\n", n_log2);
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
  assert( sizeof(curve_t<cgbn_params_small>::mem_t) == cgbn_params_small::BITS/8 );
  assert( sizeof(curve_t<cgbn_params_medium>::mem_t) == cgbn_params_medium::BITS/8 );
  
  if (!ckpt_loaded) {
    data = set_p_2p(N, curves, sigma, BITS, &data_size);
    s_partial = 1;      // First bit (doubling) is handled in set_p_2p
    batches_complete = 0;
  }
  } // Close the "if (!ckpt_loaded)" block from checkpoint loading
  else {
    // If checkpoint loaded, still need to resolve the kernel from its BITS.
    uint32_t tpi_u32 = 0;
    kernel = cgbn_stage1_kernel_dispatch(BITS, &tpi_u32);
    if (kernel == nullptr) {
      outputf(OUTPUT_ERROR, "CGBN kernel not found for BITS=%d TPI=%d from checkpoint\n", BITS, TPI);
      return ECM_ERROR;
    }
    TPI = (int32_t)tpi_u32;
    
    IPB = TPB / TPI;
    BLOCK_COUNT = (curves + IPB - 1) / IPB;
    outputf(OUTPUT_VERBOSE, "Checkpoint: restored BITS=%d, TPI=%d, BLOCK_COUNT=%lu\n", BITS, TPI, BLOCK_COUNT);
  }

  // Print the *actual* sigma now that any checkpoint resume has been applied
  // (the checkpoint may override the freshly-computed sigma from the driver).
  outputf(OUTPUT_NORMAL, "GPU: sigma=%u (param %d, %u curves)%s\n",
          sigma, ECM_PARAM_BATCH_32BITS_D, curves,
          ckpt_loaded ? " [restored from checkpoint]" : " [computed]");

  /* np0 is -(N^-1 mod 2**32), used for montgomery representation */
  uint32_t np0 = find_np0(N);

  // Copy data
  outputf (OUTPUT_VERBOSE, "Copying %'lu bytes of curves data to GPU\n", data_size);
  CUDA_CHECK(cudaMalloc((void **)&gpu_data, data_size));
  CUDA_CHECK(cudaMemcpy(gpu_data, data, data_size, cudaMemcpyHostToDevice));

  outputf (OUTPUT_NORMAL,
          "GPU: CGBN<%d, %d> kernel, N is %zu bits (%d blocks x %d threads)\n",
          TPI, BITS, n_log2, BLOCK_COUNT, TPB);

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
                           sigma, BITS, TPI, dump_host, curves, BITS / 32);
    }

    /* Call CUDA Kernel. */
    assert (kernel != NULL);
    (*kernel)<<<BLOCK_COUNT, TPB>>>(report, s_num_bits, s_partial, this_batch, gpu_s_bits, gpu_data, curves, sigma, np0);

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
                           sigma, BITS, TPI, dump_host, curves, BITS / 32);
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
        } else if (emit_progress_line(batches_complete) || final_batch) {
            // Redirected (work_manager log tailing): periodic full lines, and
            // always the final batch so the log shows 100%.
            print_progress(pct, s_partial, this_batch, per_curve_s,
                           (double)*gputime, remaining_s, true);
        }
    }

    /* Adjust batch_size to aim for 100ms */
    if (batch_time < 80) {
      batch_size = 11*batch_size/10;
    } else if (batch_time > 120) {
      batch_size = max((uint64_t)100, 9*batch_size / 10);  // MPA-OpenCl: uint64_t literal for Win64 overload resolution
    }
    
    // ========== CHECKPOINT SAVING ==========
    if (checkpoint_interval_ms > 0 &&
      (*gputime - last_checkpoint_time) >= checkpoint_interval_ms) {
      // Copy data back from GPU temporarily for checkpoint
      CUDA_CHECK(cudaMemcpy(data, gpu_data, data_size, cudaMemcpyDeviceToHost));
      
      checkpoint_header_t header;
      header.magic = CHECKPOINT_MAGIC;
      header.version = CHECKPOINT_VERSION;
      header.s_partial = s_partial;
      header.s_num_bits = s_num_bits;
      header.batches_complete = batches_complete;
      header.curves = curves;
      header.sigma = sigma;
      header.BITS = BITS;
      header.TPI = TPI;  // Save TPI for kernel selection on reload
      header.data_size = data_size;
      header.timestamp = time(NULL);
      
      save_checkpoint(ckpt_filename, &header, data, data_size);
      last_checkpoint_time = *gputime;
    }
  }

  // ── Mark progress bar complete ──────────────────────────────────────
  if (show_progress) {
      fprintf(stdout, "\n");
      fflush(stdout);
  }

  // Copy data back from GPU memory
  outputf (OUTPUT_VERBOSE, "Copying results back to CPU ...\n");
  CUDA_CHECK(cudaMemcpy(data, gpu_data, data_size, cudaMemcpyDeviceToHost));

  cudaEventElapsedTime (gputime, global_start, stop);

  youpi = process_results(factors, array_found, N, data, BITS, curves, sigma);

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
  if (youpi != ECM_ERROR && remove(ckpt_filename) == 0) {
    outputf (OUTPUT_VERBOSE, "Checkpoint file removed\n");
  }

  if (dump_file != NULL) {
    fclose(dump_file);
  }
  if (dump_host != NULL) {
    free(dump_host);
  }

  /* Write back possibly-updated sigma to caller */
  *sigma_ptr = sigma;

  return youpi;
}

#ifdef __CUDA_ARCH__
  #if __CUDA_ARCH__ < 350
    #error "Unsupported architecture"
  #endif
#endif

#endif  /* _CGBN_STAGE1_CU */
