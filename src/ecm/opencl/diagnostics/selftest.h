#pragma once

#include "../runtime/cgbn_opencl.h"

#include <gmp.h>

#include <cstdint>

int opencl_ecm_selftest_montgomery(const mpz_t N, uint32_t bits);
int opencl_ecm_selftest_mont_mul(cgbn::opencl::context_t &ctx, const mpz_t N, uint32_t bits,
                                 uint32_t np0);
