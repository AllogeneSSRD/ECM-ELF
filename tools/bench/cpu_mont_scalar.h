#ifndef CPU_MONT_SCALAR_H
#define CPU_MONT_SCALAR_H

#include <cstdint>
#include <cstddef>

#ifndef CPU_MONT_MAX_LIMBS
#define CPU_MONT_MAX_LIMBS 128
#endif

/* Scalar CIOS Montgomery multiplication (reference implementation) */
void cpu_mont_scalar_cios(uint32_t *out, const uint32_t *a, const uint32_t *b,
                          const uint32_t *N, uint32_t np0, uint32_t limbs);

/* Compute Montgomery constant np0 = -N^{-1} mod 2^32 */
void cpu_mont_np0_compute(uint32_t *np0_out, const uint32_t *N, uint32_t limbs);

#endif // CPU_MONT_SCALAR_H
