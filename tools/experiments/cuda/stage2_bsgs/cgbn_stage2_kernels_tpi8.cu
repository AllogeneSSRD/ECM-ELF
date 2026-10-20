/* cgbn_stage2_kernels_tpi8.cu — TPI=8 instantiations of the stage-2 kernels.

   Only the mid sizes: 768/1024/2048 bits are what a 256..2048-bit N needs, which is the
   range the correctness tests actually walk (the frozen vector is 129 bits, the
   Prime95-parameter cross-check is a few hundred).  See the tpi4 file for the rationale
   of the deliberately short list.
*/

#include "cgbn_stage2_kernel.h"

cgbn_s2_kernels_t cgbn_stage2_kernels_tpi8(uint32_t want_bits) {
  cgbn_s2_kernels_t r;
  r.tables = nullptr;
  r.pair = nullptr;
  r.bits = 0;
  r.tpi = 0;
  r.tpb = 0;

#define S2_TIER(B)                                                                     \
  if (want_bits <= (B)) {                                                              \
    r.tables = kernel_s2_tables<cgbn_s2_params_t<8, (B)>>;                             \
    r.pair = kernel_s2_pair<cgbn_s2_params_t<8, (B)>>;                                 \
    r.bits = (B);                                                                      \
    r.tpi = 8;                                                                         \
    r.tpb = cgbn_s2_params_t<8, (B)>::TPB;                                             \
    return r;                                                                          \
  }

  S2_TIER(768)
  S2_TIER(1024)
  S2_TIER(2048)

#undef S2_TIER
  return r;
}
