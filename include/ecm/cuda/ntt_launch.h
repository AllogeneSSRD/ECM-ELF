#pragma once
#include "launch.h"
#include "ntt_parameters.h"
const void *ecm_ntt_tile_handle(bool inverse,bool warp);
cudaError_t ecm_ntt_copy_reduce_mode(unsigned int enabled);
void ecm_launch_pointwise_twiddle_kernel(EcmCudaLaunch cfg, unsigned long long *a, const unsigned long long *b, unsigned long long n, unsigned long long omega, unsigned long long exp_scale);
void ecm_launch_dit_stage_kernel(EcmCudaLaunch cfg, unsigned long long *a, unsigned long long half, unsigned long long exp_step, unsigned long long omega, unsigned long long n);
void ecm_launch_ntt_stage_kernel(EcmCudaLaunch cfg, unsigned long long *a, unsigned long long half, unsigned long long exp_step, unsigned long long omega, unsigned long long n);
void ecm_launch_bitrev_kernel(EcmCudaLaunch cfg, const unsigned long long *src, unsigned long long *dst, unsigned long long logn, unsigned long long n);
void ecm_launch_bitrev_swap_kernel(EcmCudaLaunch cfg, unsigned long long *a, unsigned long long logn, unsigned long long n);
void ecm_launch_carry_extract_kernel(EcmCudaLaunch cfg, unsigned long long *c, unsigned long long *q, unsigned long long n, int bpw);
void ecm_launch_carry_add_kernel(EcmCudaLaunch cfg, unsigned long long *c, const unsigned long long *q, unsigned long long n);
void ecm_launch_carry_residual_kernel(EcmCudaLaunch cfg, const unsigned long long *c, unsigned long long n, int bpw, unsigned long long *out, unsigned long long stride);
void ecm_launch_slot_assemble_kernel(EcmCudaLaunch cfg, const unsigned long long *c, unsigned long long slots, unsigned long long slot_bits, unsigned long long slot_stride, int bpw, unsigned long long *out, unsigned long long stride);
void ecm_launch_scale_kernel(EcmCudaLaunch cfg, unsigned long long *d, unsigned long long n, unsigned long long n_scale);
template<int ROUNDS>
void ecm_launch_carry_cone_kernel(EcmCudaLaunch cfg, const unsigned long long *c, unsigned long long *cout, unsigned long long n, int bpw, unsigned long long stride);
void ecm_launch_build_tile_table_kernel(EcmCudaLaunch cfg, unsigned long long *tbl, int k, int t, unsigned long long root);
void ecm_launch_build_pass_table_kernel(EcmCudaLaunch cfg, unsigned long long *tbl, unsigned long long sz, int shift, unsigned long long root);
void ecm_launch_build_radix_table_kernel(EcmCudaLaunch cfg, unsigned long long *tbl, int m, unsigned long long root, int rev);
template<int M>
void ecm_launch_outer_fwd_kernel(EcmCudaLaunch cfg, unsigned long long *a, unsigned long long n, int L, const unsigned long long *tbl, const unsigned long long *D, unsigned long long grp_per_blk, unsigned long long stride);
template<int M>
void ecm_launch_outer_inv_kernel(EcmCudaLaunch cfg, unsigned long long *a, unsigned long long n, int s0, const unsigned long long *tq, const unsigned long long *dw, unsigned long long grp_per_blk, unsigned long long stride);
template<bool INVERSE,bool WARP_TAIL=false>
void ecm_launch_tile_kernel(EcmCudaLaunch cfg, unsigned long long *a, const unsigned long long *b, unsigned long long n, int k, int t, const unsigned long long *tbl, unsigned long long n_scale, unsigned long long stride, bool shift_scale=false);
void ecm_launch_gl_mul_test_kernel(EcmCudaLaunch cfg, const unsigned long long *x, const unsigned long long *y, unsigned long long *out, int n);
void ecm_launch_gl_twiddle_test_kernel(EcmCudaLaunch cfg, unsigned long long *out, unsigned long long n, unsigned long long omega);
void ecm_launch_dft_direct_kernel(EcmCudaLaunch cfg, const unsigned long long *x, unsigned long long *out, unsigned long long n, unsigned long long omega);
template<int M,bool INVERSE>
void ecm_launch_outer_coop_kernel(EcmCudaLaunch cfg, unsigned long long *a, unsigned long long n, int stage, const unsigned long long *coarse, const unsigned long long *radix, unsigned long long stride);
void ecm_launch_carry_partial_kernel(EcmCudaLaunch cfg, const unsigned long long *c, unsigned long long n, int bpw, unsigned int *partial, unsigned long long stride);
template<int ROUNDS>
void ecm_launch_carry_cone_check_kernel(EcmCudaLaunch cfg, const unsigned long long *c, unsigned long long *cout, unsigned long long n, int bpw, unsigned long long stride, unsigned int *partial);
void ecm_launch_carry_partial_finish_kernel(EcmCudaLaunch cfg, const unsigned int *partial, unsigned long long blocks, unsigned long long *out);

void ecm_ntt_tune_seed(EcmCudaLaunch,unsigned long long *,unsigned long long *,unsigned long long,unsigned long long);
void ecm_ntt_tune_verdict(EcmCudaLaunch,const unsigned long long *,unsigned long long,unsigned long long,const unsigned long long *,unsigned long long *);
