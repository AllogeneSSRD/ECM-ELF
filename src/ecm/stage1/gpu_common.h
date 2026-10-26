#pragma once
#include <cstdint>
#include <gmp.h>

// Backend-neutral GMP helpers; deliberately separate from the OpenCL entry ABI.
extern "C" uint32_t gpu_pick_random_sigma(uint32_t curves);
extern "C" void gpu_compute_batch_d(mpz_t d_out,uint32_t sigma,mpz_srcptr n);
