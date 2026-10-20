/* cgbn_stage2_kernels_tpi4.cu — TPI=4 instantiations of the stage-2 kernels.

   The tier list is deliberately short: stage 2 is an independent algorithm from stage
   1 and every extra tier costs compile time, so this covers the sizes that are actually
   exercised today (a 129-bit N -- the frozen test vector 2^128+1 -- up to 512 bits) and
   the list is extended on demand.  Add a line and the matching tier appears on the next
   configure; cgbn_stage2.cu picks the smallest tier >= N_bits + S2_CARRY_BITS.
*/

#include "cgbn_stage2_kernel.h"

cgbn_s2_kernels_t cgbn_stage2_kernels_tpi4(uint32_t want_bits) {
  cgbn_s2_kernels_t r;
  r.tables = nullptr;
  r.pair = nullptr;
  r.bits = 0;
  r.tpi = 0;
  r.tpb = 0;

#define S2_TIER(B)                                                                     \
  if (want_bits <= (B)) {                                                              \
    r.tables = kernel_s2_tables<cgbn_s2_params_t<4, (B)>>;                             \
    r.pair = kernel_s2_pair<cgbn_s2_params_t<4, (B)>>;                                 \
    r.bits = (B);                                                                      \
    r.tpi = 4;                                                                         \
    r.tpb = cgbn_s2_params_t<4, (B)>::TPB;                                             \
    return r;                                                                          \
  }

  S2_TIER(192)
  S2_TIER(256)
  S2_TIER(384)
  S2_TIER(512)

#undef S2_TIER
  return r;
}
