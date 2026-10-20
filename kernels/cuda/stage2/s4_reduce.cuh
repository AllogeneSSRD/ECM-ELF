#pragma once
#include "ecm/cuda/stage2_launch.h"
#include "../shared/stage2_scalar.cuh"
template <int NW>
__global__ void s4_mersenne_fixture_kernel(const unsigned long long *input,int stride,
    const int *lengths,int cases,int nw,int bits,unsigned long long *out,int *counts)
{
    const int k=blockIdx.x*blockDim.x+threadIdx.x;if(k>=cases)return;
    unsigned long long t[2*NW+4]={},u[NW];
    for(int i=0;i<lengths[k];++i)t[i]=input[(size_t)k*stride+i];
    counts[k]=s4_mersenne_rem<NW>(t,lengths[k],nw,bits,u);
    for(int i=0;i<nw;++i)out[(size_t)k*nw+i]=u[i];
}

template <int NW, bool MERSENNE=false>
__global__ void s4_reduce_kernel(const unsigned long long *digits, unsigned long long n,
                                 int bpw, unsigned long long slot_words,
                                 unsigned long long out_slots, unsigned long long nbatch,
                                 const unsigned long long *dn, unsigned long long ninv,
                                 int nw, int L, const unsigned long long *dy,
                                 unsigned long long w, unsigned long long *out,
                                 unsigned long long slot_bits, unsigned long long *bad,
                                 unsigned long long *s4_dbg, int tail_mont,
                                 int div_shift, unsigned long long div_recip,
                                 unsigned long long first, unsigned long long count, int mersenne_bits)
{
    const unsigned long long total = out_slots * nbatch;
    const unsigned long long gid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (gid >= total) return;
    const unsigned long long s = gid / out_slots;
    const unsigned long long k = gid - s * out_slots;
    const unsigned long long *d = digits + s * n + k * slot_words;
    /* SAME source-slot bound assertion, even for omitted coefficients. No NTT truncation.
       Separate the source index from compact output slice*count+(k-first). */
    const unsigned long long top_bits=slot_bits-(slot_words-1)*(unsigned long long)bpw;
    if(top_bits<64 && bad && (d[slot_words-1]>>top_bits)!=0) atomicAdd(bad,1ull);
    if(k<first || k-first>=count) return;
    const unsigned long long out_gid=s*count+k-first;

    /* ---- (1) bpw-bit digits -> base-2^64 limbs, least significant first.  acc holds the
       bits not yet flushed; v << nacc may spill past bit 63, and the spill is recovered as
       v >> (64-nacc) (nacc is never 0 when the spill happens, because bpw < 64). ------- */
    unsigned long long t[2 * NW + 4];
#pragma unroll 1
    for (int i = 0; i < 2 * NW + 4; ++i) t[i] = 0;
    {
        unsigned long long acc = 0;
        int nacc = 0, limb = 0;
        for (unsigned long long j = 0; j < slot_words; ++j) {
            const unsigned long long v = d[j];
            acc |= v << nacc;
            if (nacc + bpw >= 64) {
                t[limb++] = acc;
                acc = (nacc == 0) ? 0ull : (v >> (64 - nacc));
                nacc = nacc + bpw - 64;
            } else {
                nacc += bpw;
            }
        }
        if (nacc > 0) t[limb++] = acc;
    }
    unsigned long long u[NW];
    if constexpr(MERSENNE) {
        const int limbs=(int)((slot_words*(unsigned long long)bpw+63)/64);
        s4_mersenne_rem<NW>(t,limbs,nw,mersenne_bits,u);
        for(unsigned long long i=0;i<w;++i)out[out_gid*w+i]=i<(unsigned long long)nw ? u[i] : 0ull;
        return;
    }
        const int limbs = (int)((slot_words * (unsigned long long)bpw + 63) / 64);
        const int rc = s4_div_rem<NW>(t, limbs, g_s2g_device_constants.divisor, nw, div_shift, div_recip, u);
        if (rc < 0 && bad != nullptr) atomicAdd(bad, 1ull);
        for (unsigned long long i = 0; i < w; ++i)
            out[out_gid * w + i] = (i < (unsigned long long)nw) ? u[i] : 0ull;
        return;

}

template<int NW>
void ecm_launch_s4_mersenne_fixture_kernel(EcmCudaLaunch cfg, const unsigned long long *input, int stride, const int *lengths, int cases, int nw, int bits, unsigned long long *out, int *counts) {
    s4_mersenne_fixture_kernel<NW><<<cfg.grid,cfg.block,cfg.shared,cfg.stream>>>(input,stride,lengths,cases,nw,bits,out,counts);
}

template<int NW,bool MERSENNE>
void ecm_launch_s4_reduce_kernel(EcmCudaLaunch cfg, const unsigned long long *digits, unsigned long long n, int bpw, unsigned long long slot_words, unsigned long long out_slots, unsigned long long nbatch, const unsigned long long *dn, unsigned long long ninv, int nw, int L, const unsigned long long *dy, unsigned long long w, unsigned long long *out, unsigned long long slot_bits, unsigned long long *bad, unsigned long long *s4_dbg, int tail_mont, int div_shift, unsigned long long div_recip, unsigned long long first, unsigned long long count, int mersenne_bits) {
    s4_reduce_kernel<NW,MERSENNE><<<cfg.grid,cfg.block,cfg.shared,cfg.stream>>>(digits,n,bpw,slot_words,out_slots,nbatch,dn,ninv,nw,L,dy,w,out,slot_bits,bad,s4_dbg,tail_mont,div_shift,div_recip,first,count,mersenne_bits);
}
