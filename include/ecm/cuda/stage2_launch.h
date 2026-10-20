#pragma once
#include "launch.h"
#include <cstddef>
#define ECM_STAGE2_CAP_ROW(NW,FN,...) NW,
#include "tiers.def"
inline constexpr int ecm_stage2_capacities[]={ECM_CUDA_STAGE2_CAPACITIES(ECM_STAGE2_CAP_ROW,unused)};
#undef ECM_STAGE2_CAP_ROW
inline constexpr int ecm_stage2_capacity_max=ecm_stage2_capacities[sizeof(ecm_stage2_capacities)/sizeof(int)-1];
struct S2GDeviceConstants {
    unsigned long long divisor[ecm_stage2_capacity_max];
    int point_mersenne_bits;
};
static_assert(offsetof(S2GDeviceConstants,divisor)==0,"divisor must lead constant storage");
cudaError_t ecm_stage2_copy_constants(const void *source,size_t bytes,size_t offset=0,cudaMemcpyKind kind=cudaMemcpyHostToDevice);
template<int NW,bool XADD6=false>
void ecm_launch_s2g_ladder_kernel(EcmCudaLaunch cfg, const unsigned long long *n, unsigned long long ninv, int nw, const unsigned long long *qx, const unsigned long long *qz, const unsigned long long *a24, const unsigned long long *mone, const unsigned long long *js, int npts, unsigned long long *out_x, unsigned long long *out_z, bool normal_output);
template<int NW>
void ecm_launch_s2g_mont_test_kernel(EcmCudaLaunch cfg, const unsigned long long *a, const unsigned long long *b, const unsigned long long *n, unsigned long long ninv, int nw, int cases, unsigned long long *out);
template<int NW,bool XADD6>
void ecm_launch_s2g_xadd_test_kernel(EcmCudaLaunch cfg, const unsigned long long *input, const unsigned long long *n, unsigned long long ninv, int nw, int cases, unsigned long long *output);
template<int NW,bool XADD6>
void ecm_launch_s2g_seed_pair_kernel(EcmCudaLaunch cfg, const unsigned long long *n, unsigned long long ninv, int nw, const unsigned long long *hx, const unsigned long long *hz, const unsigned long long *a24, const unsigned long long *mone, unsigned long long clo, unsigned long long npts, unsigned long long per_block, unsigned long long blocks, unsigned long long offset, unsigned long long count, unsigned long long *ox, unsigned long long *oz);
template<int NW>
void ecm_launch_s2g_segprod_kernel(EcmCudaLaunch cfg, const unsigned long long *dn, unsigned long long ninv, int nw, const unsigned long long *dz, unsigned long long npts, unsigned long long seg, unsigned long long nseg, unsigned long long *out, const unsigned long long *fix);
void ecm_launch_s2g_projective_leaf_kernel(EcmCudaLaunch cfg, const unsigned long long *n, int nw, const unsigned long long *x, const unsigned long long *z, size_t first, size_t count, unsigned long long *out);
template<int NW>
void ecm_launch_s2g_block_prod_kernel(EcmCudaLaunch cfg, const unsigned long long *vals, int nvals, int nw, int per_block, const unsigned long long *n, unsigned long long ninv, unsigned long long *out);
template<int NW>
void ecm_launch_s4_mersenne_fixture_kernel(EcmCudaLaunch cfg, const unsigned long long *input, int stride, const int *lengths, int cases, int nw, int bits, unsigned long long *out, int *counts);
template<int NW,bool MERSENNE=false>
void ecm_launch_s4_reduce_kernel(EcmCudaLaunch cfg, const unsigned long long *digits, unsigned long long n, int bpw, unsigned long long slot_words, unsigned long long out_slots, unsigned long long nbatch, const unsigned long long *dn, unsigned long long ninv, int nw, int L, const unsigned long long *dy, unsigned long long w, unsigned long long *out, unsigned long long slot_bits, unsigned long long *bad, unsigned long long *s4_dbg, int tail_mont, int div_shift, unsigned long long div_recip, unsigned long long first, unsigned long long count, int mersenne_bits);
void ecm_launch_s4_scatter_result_kernel(EcmCudaLaunch cfg, const unsigned long long *src, size_t stride, size_t first, size_t count, size_t m, int W, const unsigned long long *offsets, unsigned long long *dst);
void ecm_launch_groot_input_digest_kernel(EcmCudaLaunch cfg, const unsigned long long *input, size_t count, unsigned long long base, unsigned long long *digest);
void ecm_launch_scaled_root_reverse_kernel(EcmCudaLaunch cfg, const unsigned long long *src, size_t count, size_t degree, int W, unsigned long long *dst);
void ecm_launch_fold_reverse_kernel(EcmCudaLaunch cfg, const unsigned long long *src, size_t top, size_t count, int W, unsigned long long *out);
void ecm_launch_fold_subtract_kernel(EcmCudaLaunch cfg, const unsigned long long *a, const unsigned long long *b, const unsigned long long *n, int W, size_t count, unsigned long long *out, unsigned long long *length);
namespace stage2_pack {
template<bool GATHER,bool REVERSE>
void ecm_launch_pack_zero_kernel(EcmCudaLaunch cfg, const unsigned long long *src, const unsigned long long *offsets, unsigned nc, unsigned W, unsigned ndig, unsigned bpw, unsigned sw, unsigned reciprocal, unsigned N, unsigned long long first_slice, unsigned long long *dst);
}
void ecm_launch_s4_pack_batch_kernel(EcmCudaLaunch cfg, const unsigned long long *src, int S, int bpw, unsigned long long slot_words, unsigned long long N, unsigned long long ds, unsigned long long ma, int W, unsigned long long *dst);
void ecm_launch_s4_pack_gather_kernel(EcmCudaLaunch cfg, const unsigned long long *src, const unsigned long long *offsets, size_t m, size_t nc, int W, int bits, int bpw, unsigned long long sw, unsigned long long N, unsigned long long *dst);
void ecm_launch_s4_pack_gather_reverse_kernel(EcmCudaLaunch cfg, const unsigned long long *src, const unsigned long long *offsets, size_t m, size_t nc, int W, int bits, int bpw, unsigned long long sw, unsigned long long N, unsigned long long *dst);
template<int NW>
void ecm_launch_s2g_baby_product_kernel(EcmCudaLaunch cfg, const unsigned long long *in, size_t count, unsigned long long *out, const unsigned long long *n, unsigned long long ninv, int nw);
template<int NW>
void ecm_launch_s2g_baby_inverse_kernel(EcmCudaLaunch cfg, const unsigned long long *parent, unsigned long long *children, size_t count, const unsigned char *good, size_t parents_per_group, const unsigned long long *n, unsigned long long ninv, int nw);
template<int NW>
void ecm_launch_s2g_baby_leaf_kernel(EcmCudaLaunch cfg, const unsigned long long *parent, const unsigned long long *x, const unsigned long long *z, size_t count, unsigned long long *out, const unsigned char *good, const unsigned long long *n, unsigned long long ninv, int nw);
template<int NW>
void ecm_launch_s2g_plain_scale_kernel(EcmCudaLaunch cfg, unsigned long long *h, size_t count, const unsigned long long *factor_mont, const unsigned long long *n, unsigned long long ninv, int nw);
template<int NW,bool XADD6=false>
void ecm_launch_s2g_chain_kernel(EcmCudaLaunch cfg, const unsigned long long *dn, unsigned long long ninv, int nw, const unsigned long long *ddx, const unsigned long long *ddz, const unsigned long long *dsx, const unsigned long long *dsz, const unsigned long long *esx, const unsigned long long *esz, unsigned long long npts, unsigned long long per_block, unsigned long long blocks, unsigned long long *out_x, unsigned long long *out_z, size_t seed_stride);
