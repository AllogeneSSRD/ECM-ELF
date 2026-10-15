#pragma once
#include <gmp.h>
#include <cstdint>
#include <string>

// Per curve: d, Px, Py, Pz, Pt, Rx, Ry, Rz, Rt (ordinary residues).
// done counts binary scalar bits from the MSB, NOT CPU/Prime95 NAF positions.
// A callback sees canonical ordinary residues at a completed slice boundary.
// Return 0=completed, 2=callback stopped, -1=error (message populated).
using EcmEdwardsProgress = bool (*)(uint32_t done, void *context);
int ecm_cuda_edwards_stage1(const mpz_t n,const mpz_t scalar,mpz_t *states,
                            uint32_t count,uint32_t &done,uint32_t chunk_bits,
                            int device,EcmEdwardsProgress progress,void *context,
                            std::string &error);
