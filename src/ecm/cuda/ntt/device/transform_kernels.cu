#include "../launch.h"
#include "transform.cuh"
#include "carry.cuh"
#include "tile.cuh"
#include "outer_coop.cuh"
#include "carry_partial.cuh"
#include "tune_checks.cuh"

const void *ecm_ntt_tile_handle(bool inverse,bool warp) {
    if(inverse)return warp ? (const void*)tile_kernel<true,true> : (const void*)tile_kernel<true>;
    return warp ? (const void*)tile_kernel<false,true> : (const void*)tile_kernel<false>;
}
cudaError_t ecm_ntt_copy_reduce_mode(unsigned int enabled) {
#if NTT_GL_FIXED_MODE == -1
    return cudaMemcpyToSymbol(ntt_gl_short_reduce_device,&enabled,sizeof(enabled));
#else
    (void)enabled;return cudaSuccess;
#endif
}

template void ecm_launch_carry_cone_kernel<10>(EcmCudaLaunch, const unsigned long long *c, unsigned long long *cout, unsigned long long n, int bpw, unsigned long long stride);
template void ecm_launch_carry_cone_kernel<9>(EcmCudaLaunch, const unsigned long long *c, unsigned long long *cout, unsigned long long n, int bpw, unsigned long long stride);
template void ecm_launch_carry_cone_kernel<8>(EcmCudaLaunch, const unsigned long long *c, unsigned long long *cout, unsigned long long n, int bpw, unsigned long long stride);
template void ecm_launch_carry_cone_kernel<7>(EcmCudaLaunch, const unsigned long long *c, unsigned long long *cout, unsigned long long n, int bpw, unsigned long long stride);
template void ecm_launch_carry_cone_kernel<6>(EcmCudaLaunch, const unsigned long long *c, unsigned long long *cout, unsigned long long n, int bpw, unsigned long long stride);
template void ecm_launch_carry_cone_kernel<5>(EcmCudaLaunch, const unsigned long long *c, unsigned long long *cout, unsigned long long n, int bpw, unsigned long long stride);
template void ecm_launch_carry_cone_kernel<4>(EcmCudaLaunch, const unsigned long long *c, unsigned long long *cout, unsigned long long n, int bpw, unsigned long long stride);
template void ecm_launch_carry_cone_kernel<3>(EcmCudaLaunch, const unsigned long long *c, unsigned long long *cout, unsigned long long n, int bpw, unsigned long long stride);
template void ecm_launch_carry_cone_kernel<2>(EcmCudaLaunch, const unsigned long long *c, unsigned long long *cout, unsigned long long n, int bpw, unsigned long long stride);
template void ecm_launch_carry_cone_kernel<1>(EcmCudaLaunch, const unsigned long long *c, unsigned long long *cout, unsigned long long n, int bpw, unsigned long long stride);

template void ecm_launch_outer_fwd_kernel<6>(EcmCudaLaunch, unsigned long long *a, unsigned long long n, int L, const unsigned long long *tbl, const unsigned long long *D, unsigned long long grp_per_blk, unsigned long long stride);
template void ecm_launch_outer_fwd_kernel<5>(EcmCudaLaunch, unsigned long long *a, unsigned long long n, int L, const unsigned long long *tbl, const unsigned long long *D, unsigned long long grp_per_blk, unsigned long long stride);
template void ecm_launch_outer_fwd_kernel<4>(EcmCudaLaunch, unsigned long long *a, unsigned long long n, int L, const unsigned long long *tbl, const unsigned long long *D, unsigned long long grp_per_blk, unsigned long long stride);
template void ecm_launch_outer_fwd_kernel<3>(EcmCudaLaunch, unsigned long long *a, unsigned long long n, int L, const unsigned long long *tbl, const unsigned long long *D, unsigned long long grp_per_blk, unsigned long long stride);
template void ecm_launch_outer_fwd_kernel<2>(EcmCudaLaunch, unsigned long long *a, unsigned long long n, int L, const unsigned long long *tbl, const unsigned long long *D, unsigned long long grp_per_blk, unsigned long long stride);
template void ecm_launch_outer_fwd_kernel<1>(EcmCudaLaunch, unsigned long long *a, unsigned long long n, int L, const unsigned long long *tbl, const unsigned long long *D, unsigned long long grp_per_blk, unsigned long long stride);

template void ecm_launch_outer_inv_kernel<6>(EcmCudaLaunch, unsigned long long *a, unsigned long long n, int s0, const unsigned long long *tq, const unsigned long long *dw, unsigned long long grp_per_blk, unsigned long long stride);
template void ecm_launch_outer_inv_kernel<5>(EcmCudaLaunch, unsigned long long *a, unsigned long long n, int s0, const unsigned long long *tq, const unsigned long long *dw, unsigned long long grp_per_blk, unsigned long long stride);
template void ecm_launch_outer_inv_kernel<4>(EcmCudaLaunch, unsigned long long *a, unsigned long long n, int s0, const unsigned long long *tq, const unsigned long long *dw, unsigned long long grp_per_blk, unsigned long long stride);
template void ecm_launch_outer_inv_kernel<3>(EcmCudaLaunch, unsigned long long *a, unsigned long long n, int s0, const unsigned long long *tq, const unsigned long long *dw, unsigned long long grp_per_blk, unsigned long long stride);
template void ecm_launch_outer_inv_kernel<2>(EcmCudaLaunch, unsigned long long *a, unsigned long long n, int s0, const unsigned long long *tq, const unsigned long long *dw, unsigned long long grp_per_blk, unsigned long long stride);
template void ecm_launch_outer_inv_kernel<1>(EcmCudaLaunch, unsigned long long *a, unsigned long long n, int s0, const unsigned long long *tq, const unsigned long long *dw, unsigned long long grp_per_blk, unsigned long long stride);

template void ecm_launch_tile_kernel<true,false>(EcmCudaLaunch, unsigned long long *a, const unsigned long long *b, unsigned long long n, int k, int t, const unsigned long long *tbl, unsigned long long n_scale, unsigned long long stride, bool shift_scale);
template void ecm_launch_tile_kernel<true,true>(EcmCudaLaunch, unsigned long long *a, const unsigned long long *b, unsigned long long n, int k, int t, const unsigned long long *tbl, unsigned long long n_scale, unsigned long long stride, bool shift_scale);
template void ecm_launch_tile_kernel<false,false>(EcmCudaLaunch, unsigned long long *a, const unsigned long long *b, unsigned long long n, int k, int t, const unsigned long long *tbl, unsigned long long n_scale, unsigned long long stride, bool shift_scale);
template void ecm_launch_tile_kernel<false,true>(EcmCudaLaunch, unsigned long long *a, const unsigned long long *b, unsigned long long n, int k, int t, const unsigned long long *tbl, unsigned long long n_scale, unsigned long long stride, bool shift_scale);

template void ecm_launch_outer_coop_kernel<8,true>(EcmCudaLaunch, unsigned long long *a, unsigned long long n, int stage, const unsigned long long *coarse, const unsigned long long *radix, unsigned long long stride);
template void ecm_launch_outer_coop_kernel<7,true>(EcmCudaLaunch, unsigned long long *a, unsigned long long n, int stage, const unsigned long long *coarse, const unsigned long long *radix, unsigned long long stride);
template void ecm_launch_outer_coop_kernel<6,true>(EcmCudaLaunch, unsigned long long *a, unsigned long long n, int stage, const unsigned long long *coarse, const unsigned long long *radix, unsigned long long stride);
template void ecm_launch_outer_coop_kernel<5,true>(EcmCudaLaunch, unsigned long long *a, unsigned long long n, int stage, const unsigned long long *coarse, const unsigned long long *radix, unsigned long long stride);
template void ecm_launch_outer_coop_kernel<8,false>(EcmCudaLaunch, unsigned long long *a, unsigned long long n, int stage, const unsigned long long *coarse, const unsigned long long *radix, unsigned long long stride);
template void ecm_launch_outer_coop_kernel<7,false>(EcmCudaLaunch, unsigned long long *a, unsigned long long n, int stage, const unsigned long long *coarse, const unsigned long long *radix, unsigned long long stride);
template void ecm_launch_outer_coop_kernel<6,false>(EcmCudaLaunch, unsigned long long *a, unsigned long long n, int stage, const unsigned long long *coarse, const unsigned long long *radix, unsigned long long stride);
template void ecm_launch_outer_coop_kernel<5,false>(EcmCudaLaunch, unsigned long long *a, unsigned long long n, int stage, const unsigned long long *coarse, const unsigned long long *radix, unsigned long long stride);

template void ecm_launch_carry_cone_check_kernel<10>(EcmCudaLaunch, const unsigned long long *c, unsigned long long *cout, unsigned long long n, int bpw, unsigned long long stride, unsigned int *partial);
template void ecm_launch_carry_cone_check_kernel<9>(EcmCudaLaunch, const unsigned long long *c, unsigned long long *cout, unsigned long long n, int bpw, unsigned long long stride, unsigned int *partial);
template void ecm_launch_carry_cone_check_kernel<8>(EcmCudaLaunch, const unsigned long long *c, unsigned long long *cout, unsigned long long n, int bpw, unsigned long long stride, unsigned int *partial);
template void ecm_launch_carry_cone_check_kernel<7>(EcmCudaLaunch, const unsigned long long *c, unsigned long long *cout, unsigned long long n, int bpw, unsigned long long stride, unsigned int *partial);
template void ecm_launch_carry_cone_check_kernel<6>(EcmCudaLaunch, const unsigned long long *c, unsigned long long *cout, unsigned long long n, int bpw, unsigned long long stride, unsigned int *partial);
template void ecm_launch_carry_cone_check_kernel<5>(EcmCudaLaunch, const unsigned long long *c, unsigned long long *cout, unsigned long long n, int bpw, unsigned long long stride, unsigned int *partial);
template void ecm_launch_carry_cone_check_kernel<4>(EcmCudaLaunch, const unsigned long long *c, unsigned long long *cout, unsigned long long n, int bpw, unsigned long long stride, unsigned int *partial);
template void ecm_launch_carry_cone_check_kernel<3>(EcmCudaLaunch, const unsigned long long *c, unsigned long long *cout, unsigned long long n, int bpw, unsigned long long stride, unsigned int *partial);
template void ecm_launch_carry_cone_check_kernel<2>(EcmCudaLaunch, const unsigned long long *c, unsigned long long *cout, unsigned long long n, int bpw, unsigned long long stride, unsigned int *partial);
template void ecm_launch_carry_cone_check_kernel<1>(EcmCudaLaunch, const unsigned long long *c, unsigned long long *cout, unsigned long long n, int bpw, unsigned long long stride, unsigned int *partial);
