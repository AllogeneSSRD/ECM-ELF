#pragma once

#include "ecm_mont_cpu.h"

// Null on success. The optional DLL is loaded only on the first GWNUM request.
const char* mont_stage1_gwnum_load_error();

// GWNUM implementation of the PARAM0 binary ladder. Checkpoints use the
// same plain-domain coordinates and bit offsets as mont_stage1_curve_bits_ex.
int mont_stage1_curve_gwnum_bits_ex(mpz_t factor, mpz_t Qx, mpz_t Qz, const mpz_t N,
                                   uint64_t sigma, const uint8_t* bits, size_t nbits,
                                   size_t start_bit, mont_ladder_state_t* st,
                                   mont_progress_fn cb, void* cb_ctx, size_t chunk_bits);
