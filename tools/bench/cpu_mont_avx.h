#ifndef CPU_MONT_AVX_H
#define CPU_MONT_AVX_H

#include <cstdint>
#include <cstddef>

/* CPU feature detection */
bool cpu_has_avx512f();
bool cpu_has_avx2();

/* Batched Montgomery multiplication — SoA layout.
 * Each array is [kInstances × max_limbs] uint32_t, row-major:
 *   arr[inst * max_limbs + limb]
 * where max_limbs is the padded limb count (e.g. 16 for 512-bit, 32 for 1024-bit).
 *
 * kInstances: 16 for AVX512, 8 for AVX2.
 *
 * out[inst*max_limbs + limb] = mont_mul(a[inst*max_limbs + limb], b[inst*max_limbs + limb])
 *    = a*b*R^{-1} mod N (R = 2^{32*limbs})
 *
 * np0 and N are the same for all instances (same modulus).
 */

void avx512_mont_cios_batch(uint32_t *out, const uint32_t *a, const uint32_t *b,
                             const uint32_t *N, uint32_t np0, uint32_t limbs,
                             uint32_t max_limbs, uint32_t kInstances);

void avx2_mont_cios_batch(uint32_t *out, const uint32_t *a, const uint32_t *b,
                          const uint32_t *N, uint32_t np0, uint32_t limbs,
                          uint32_t max_limbs, uint32_t kInstances);

#endif // CPU_MONT_AVX_H