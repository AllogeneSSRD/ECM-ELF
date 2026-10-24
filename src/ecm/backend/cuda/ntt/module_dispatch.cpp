#include "launch.h"
#ifdef ECM_CUDA_ISOLATED_DEVICE
const void *ecm_ntt_tile_true_false();
const void *ecm_ntt_tile_true_true();
const void *ecm_ntt_tile_false_false();
const void *ecm_ntt_tile_false_true();
const void *ecm_ntt_tile_handle(bool inverse,bool warp) {
    if(inverse)return warp?ecm_ntt_tile_true_true():ecm_ntt_tile_true_false();
    return warp?ecm_ntt_tile_false_true():ecm_ntt_tile_false_false();
}
cudaError_t ecm_ntt_copy_reduce_mode(unsigned int enabled) {
    static_assert(NTT_GL_FIXED_MODE==3,"Isolated production modules require fixed PTX arithmetic");
    (void)enabled;return cudaSuccess;
}
#endif
