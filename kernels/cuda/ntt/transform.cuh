#pragma once
#include "ecm/cuda/ntt_launch.h"
#include "../shared/goldilocks.cuh"
namespace {
__global__ void pointwise_twiddle_kernel(unsigned long long *a, const unsigned long long *b,
                                         unsigned long long n, unsigned long long omega,
                                         unsigned long long exp_scale)
{
    const unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned long long e = gl_mul_mod_n(i, exp_scale, n);
    a[i] = gl_mul(gl_mul(a[i], b[i]), gl_twiddle(omega, e));
}
}

void ecm_launch_pointwise_twiddle_kernel(EcmCudaLaunch cfg, unsigned long long *a, const unsigned long long *b, unsigned long long n, unsigned long long omega, unsigned long long exp_scale) {
    pointwise_twiddle_kernel<<<cfg.grid,cfg.block,cfg.shared,cfg.stream>>>(a,b,n,omega,exp_scale);
}

namespace {
__global__ void dit_stage_kernel(unsigned long long *a, unsigned long long half,
                                 unsigned long long exp_step, unsigned long long omega,
                                 unsigned long long n)
{
    dit_stage(a, half, exp_step, omega, n);
}
}

void ecm_launch_dit_stage_kernel(EcmCudaLaunch cfg, unsigned long long *a, unsigned long long half, unsigned long long exp_step, unsigned long long omega, unsigned long long n) {
    dit_stage_kernel<<<cfg.grid,cfg.block,cfg.shared,cfg.stream>>>(a,half,exp_step,omega,n);
}

namespace {
__global__ void ntt_stage_kernel(unsigned long long *a, unsigned long long half,
                                 unsigned long long exp_step, unsigned long long omega,
                                 unsigned long long n)
{
    ntt_stage(a, half, exp_step, omega, n);
}
}

void ecm_launch_ntt_stage_kernel(EcmCudaLaunch cfg, unsigned long long *a, unsigned long long half, unsigned long long exp_step, unsigned long long omega, unsigned long long n) {
    ntt_stage_kernel<<<cfg.grid,cfg.block,cfg.shared,cfg.stream>>>(a,half,exp_step,omega,n);
}

namespace {
__global__ void bitrev_kernel(const unsigned long long *src, unsigned long long *dst,
                              unsigned long long logn, unsigned long long n)
{
    const unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned long long r = 0, v = i;
    for (unsigned long long k = 0; k < logn; ++k) { r = (r << 1) | (v & 1ull); v >>= 1; }
    dst[i] = src[r];
}
}

void ecm_launch_bitrev_kernel(EcmCudaLaunch cfg, const unsigned long long *src, unsigned long long *dst, unsigned long long logn, unsigned long long n) {
    bitrev_kernel<<<cfg.grid,cfg.block,cfg.shared,cfg.stream>>>(src,dst,logn,n);
}

namespace {
__global__ void bitrev_swap_kernel(unsigned long long *a, unsigned long long logn,
                                   unsigned long long n)
{
    const unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned long long r = 0, v = i;
    for (unsigned long long k = 0; k < logn; ++k) { r = (r << 1) | (v & 1ull); v >>= 1; }
    if (i < r) {
        const unsigned long long t = a[i];
        a[i] = a[r];
        a[r] = t;
    }
}
}

void ecm_launch_bitrev_swap_kernel(EcmCudaLaunch cfg, unsigned long long *a, unsigned long long logn, unsigned long long n) {
    bitrev_swap_kernel<<<cfg.grid,cfg.block,cfg.shared,cfg.stream>>>(a,logn,n);
}

__global__ void scale_kernel(unsigned long long *d, unsigned long long n,
                             unsigned long long n_scale)
{
    const unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n) d[i] = gl_mul(d[i], n_scale);
}

void ecm_launch_scale_kernel(EcmCudaLaunch cfg, unsigned long long *d, unsigned long long n, unsigned long long n_scale) {
    scale_kernel<<<cfg.grid,cfg.block,cfg.shared,cfg.stream>>>(d,n,n_scale);
}

__global__ void gl_mul_test_kernel(const unsigned long long *x, const unsigned long long *y,
                                   unsigned long long *out, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = gl_mul(x[i], y[i]);
}

void ecm_launch_gl_mul_test_kernel(EcmCudaLaunch cfg, const unsigned long long *x, const unsigned long long *y, unsigned long long *out, int n) {
    gl_mul_test_kernel<<<cfg.grid,cfg.block,cfg.shared,cfg.stream>>>(x,y,out,n);
}

__global__ void gl_twiddle_test_kernel(unsigned long long *out, unsigned long long n,
                                       unsigned long long omega)
{
    const unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n) out[i] = gl_twiddle(omega, i);
}

void ecm_launch_gl_twiddle_test_kernel(EcmCudaLaunch cfg, unsigned long long *out, unsigned long long n, unsigned long long omega) {
    gl_twiddle_test_kernel<<<cfg.grid,cfg.block,cfg.shared,cfg.stream>>>(out,n,omega);
}

__global__ void dft_direct_kernel(const unsigned long long *x, unsigned long long *out,
                                  unsigned long long n, unsigned long long omega)
{
    const unsigned long long k = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (k >= n) return;
    unsigned long long acc = 0;
    for (unsigned long long j = 0; j < n; ++j) {
        acc = gl_add_dev(acc, gl_mul(x[j], gl_twiddle(omega, (j * k) % n)));
    }
    out[k] = acc;
}

void ecm_launch_dft_direct_kernel(EcmCudaLaunch cfg, const unsigned long long *x, unsigned long long *out, unsigned long long n, unsigned long long omega) {
    dft_direct_kernel<<<cfg.grid,cfg.block,cfg.shared,cfg.stream>>>(x,out,n,omega);
}
