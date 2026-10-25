/* cgbn_stage1.h: header for CGBN (GPU) based ecm stage 1.

  Copyright 2021 Seth Troisi

  This program is free software; you can redistribute it and/or modify it
  under the terms of the GNU General Public License as published by the
  Free Software Foundation; either version 3 of the License, or (at your
  option) any later version.

  This program is distributed in the hope that it will be useful, but WITHOUT
  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
  more details.

  You should have received a copy of the GNU General Public License along
  with this program; see the file COPYING.  If not, write to the Free
  Software Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA
  02111-1307, USA.
*/

#ifndef _CGBN_STAGE1_H
#define _CGBN_STAGE1_H 1

#include <stdint.h>

#include <gmp.h>
#include "../stage1/gpu_common.h"

#ifdef __cplusplus
extern "C" {
#endif

int cgbn_ecm_stage1(void* checkpoint_context, mpz_t *factors, int *array_found,
             const mpz_t N, const mpz_t s,
             uint32_t curves, uint32_t *sigma,
             unsigned long checkpoint_interval_ms,
             float *gputime, int verbose,
             const char *gpu_mul_path, const char *gpu_sqr_path,
             const char *gpu_add_path, const char *gpu_sub_path,
             const char *gpu_special_mult_path);

/* Create OpenCL context, compile kernel for N, print GPU device info (once). */
int gpu_prepare_opencl(size_t n_log2, int verbose, const char *gpu_mul_path,
                       const char *gpu_sqr_path, const char *gpu_add_path,
                       const char *gpu_sub_path,
                       const char *gpu_special_mult_path);

#ifdef __cplusplus
}
#endif


#endif /* _CGBN_STAGE1_H */
