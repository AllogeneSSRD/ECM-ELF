#pragma once
#include "launch.h"

// Products are Montgomery images. Inverses deliberately stay ordinary:
// Mont(1/(ab), bR) = 1/a, and Mont(xR, 1/z) = x/z.
// Each 256-leaf root is checked/inverted on the host; nonunit groups are skipped.

template<int NW>
static void s2g_launch_baby_product(int nw,size_t count,const unsigned long long *in,
    unsigned long long *out,const unsigned long long *n,unsigned long long ninv)
{ecm_launch_s2g_baby_product_kernel<NW>(EcmCudaLaunch{(unsigned)((count+127)/128),64}, in,count,out,n,ninv,nw);}
template<int NW>
static void s2g_launch_baby_inverse(int nw,size_t count,const unsigned long long *parent,
    unsigned long long *children,const unsigned char *good,size_t ppg,
    const unsigned long long *n,unsigned long long ninv)
{ecm_launch_s2g_baby_inverse_kernel<NW>(EcmCudaLaunch{(unsigned)((count+127)/128),64}, parent,children,count,good,ppg,n,ninv,nw);}
template<int NW>
static void s2g_launch_baby_leaf(int nw,size_t count,const unsigned long long *parent,
    const unsigned long long *x,const unsigned long long *z,unsigned long long *out,
    const unsigned char *good,const unsigned long long *n,unsigned long long ninv)
{ecm_launch_s2g_baby_leaf_kernel<NW>(EcmCudaLaunch{(unsigned)((count+127)/128),64}, parent,x,z,count,out,good,n,ninv,nw);}
