#include <gmp.h>
#include <ecm/cuda/cgbn.h>
#include "../launch.h"
#include "point_coop_chain.cuh"

void ecm_launch_s2g_coop_chain_kernel(EcmCudaLaunch cfg,const unsigned long long *n,
    unsigned long long ninv,int nw,const unsigned long long *dx,const unsigned long long *dz,
    const unsigned long long *sx,const unsigned long long *sz,const unsigned long long *ex,
    const unsigned long long *ez,unsigned long long chains,unsigned long long points,
    unsigned long long length,size_t seed_stride,int mersenne_bits,
    unsigned long long *out_x,unsigned long long *out_z)
{
    stage2_coop_chain<256,16><<<cfg.grid,cfg.block,cfg.shared,cfg.stream>>>(n,ninv,nw,dx,dz,sx,sz,
        ex,ez,chains,points,length,seed_stride,mersenne_bits,out_x,out_z);
}

void ecm_launch_s2g_coop_mont_test_kernel(EcmCudaLaunch cfg,const unsigned long long *a,
    const unsigned long long *b,const unsigned long long *n,unsigned long long ninv,int nw,
    int cases,int mersenne_bits,bool square,unsigned long long *out)
{
    stage2_coop_mont_test<256,16><<<cfg.grid,cfg.block,cfg.shared,cfg.stream>>>(
        a,b,n,ninv,nw,cases,mersenne_bits,square,out);
}
