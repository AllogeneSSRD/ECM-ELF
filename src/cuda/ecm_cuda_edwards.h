#pragma once
#include <gmp.h>
#include <cstdint>
#include <cstddef>
#include <string>

// Per curve: d, Px, Py, Pz, Pt, Rx, Ry, Rz, Rt (ordinary residues).
// done counts binary scalar bits from the MSB, NOT CPU/Prime95 NAF positions.
// A callback sees canonical ordinary residues at a completed slice boundary.
// Return 0=completed, 2=callback stopped, -1=error (message populated).
// chunk_bits=0 adapts slices toward 100 ms; positive values stay fixed.
using EcmEdwardsProgress = bool (*)(uint32_t done, void *context);
// Optional one-shot notification after allocations/uploads/dictionary preparation,
// before any scalar slice. Uses the same context; does not imply completed work.
using EcmEdwardsPrepared = void (*)(void *context);
int ecm_cuda_edwards_stage1(const mpz_t n,const mpz_t scalar,mpz_t *states,
                            uint32_t count,uint32_t &done,uint32_t chunk_bits,
                            int device,EcmEdwardsProgress progress,void *context,
                            std::string &error,EcmEdwardsPrepared prepared=nullptr);

// Query the smallest compiled container with one spare bit; no GPU initialization.
// Returns false (and zero outputs) when the build has no suitable instance.
bool ecm_cuda_edwards_tier(size_t bits,uint32_t &container,uint32_t &tpi);
