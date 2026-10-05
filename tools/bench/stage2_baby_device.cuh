#pragma once

// Products are Montgomery images. Inverses deliberately stay ordinary:
// Mont(1/(ab), bR) = 1/a, and Mont(xR, 1/z) = x/z.
// Each 256-leaf root is checked/inverted on the host; nonunit groups are skipped.
template<int NW>
__global__ void s2g_baby_product_kernel(const unsigned long long *in,size_t count,
    unsigned long long *out,const unsigned long long *n,unsigned long long ninv,int nw)
{
    const size_t k=blockIdx.x*(size_t)blockDim.x+threadIdx.x;
    if(k>=(count+1)/2)return;
    const auto *a=in+2*k*nw;
    if(2*k+1==count) {for(int j=0;j<nw;++j)out[k*nw+j]=a[j];return;}
    unsigned long long v[NW];
    s2g_mont_mul<NW>(v,a,a+nw,n,ninv,nw);
    for(int j=0;j<nw;++j)out[k*nw+j]=v[j];
}

template<int NW>
__global__ void s2g_baby_inverse_kernel(const unsigned long long *parent,
    unsigned long long *children,size_t count,const unsigned char *good,size_t parents_per_group,
    const unsigned long long *n,unsigned long long ninv,int nw)
{
    const size_t k=blockIdx.x*(size_t)blockDim.x+threadIdx.x;
    if(k>=(count+1)/2 || !good[k/parents_per_group])return;
    auto *a=children+2*k*nw;const auto *p=parent+k*nw;
    if(2*k+1==count){for(int j=0;j<nw;++j)a[j]=p[j];return;}
    // Read both sibling products before overwriting either child.
    unsigned long long left[NW],right[NW];
    s2g_mont_mul<NW>(left,p,a+nw,n,ninv,nw);
    s2g_mont_mul<NW>(right,p,a,n,ninv,nw);
    for(int j=0;j<nw;++j){a[j]=left[j];a[nw+j]=right[j];}
}

template<int NW>
__device__ __forceinline__ void s2g_baby_negative(unsigned long long *out,
    const unsigned long long *x,const unsigned long long *inv,
    const unsigned long long *n,unsigned long long ninv,int nw)
{
    unsigned long long v[NW];s2g_mont_mul<NW>(v,x,inv,n,ninv,nw);
    bool zero=true;for(int j=0;j<nw;++j)if(v[j])zero=false;
    unsigned long long borrow=0;
    for(int j=0;j<nw;++j){const unsigned long long t=n[j]-v[j],b1=n[j]<v[j],u=t-borrow,b2=t<borrow;
        borrow=b1|b2;out[j]=zero?0:u;}
}

template<int NW>
__global__ void s2g_baby_leaf_kernel(const unsigned long long *parent,
    const unsigned long long *x,const unsigned long long *z,size_t count,
    unsigned long long *out,const unsigned char *good,const unsigned long long *n,
    unsigned long long ninv,int nw)
{
    const size_t k=blockIdx.x*(size_t)blockDim.x+threadIdx.x;
    if(k>=(count+1)/2)return;
    const size_t first=2*k;
    if(!good[first/256]) {
        for(size_t i=first;i<count && i<first+2;++i)for(int j=0;j<nw;++j)out[i*nw+j]=0;
        return;
    }
    const auto *p=parent+k*nw;
    if(first+1==count){s2g_baby_negative<NW>(out+first*nw,x+first*nw,p,n,ninv,nw);return;}
    unsigned long long left[NW],right[NW];
    s2g_mont_mul<NW>(left,p,z+(first+1)*nw,n,ninv,nw);
    s2g_mont_mul<NW>(right,p,z+first*nw,n,ninv,nw);
    s2g_baby_negative<NW>(out+first*nw,x+first*nw,left,n,ninv,nw);
    s2g_baby_negative<NW>(out+(first+1)*nw,x+(first+1)*nw,right,n,ninv,nw);
}

template<int NW>
static void s2g_launch_baby_product(int nw,size_t count,const unsigned long long *in,
    unsigned long long *out,const unsigned long long *n,unsigned long long ninv)
{s2g_baby_product_kernel<NW><<<(unsigned)((count+127)/128),64>>>(in,count,out,n,ninv,nw);}
template<int NW>
static void s2g_launch_baby_inverse(int nw,size_t count,const unsigned long long *parent,
    unsigned long long *children,const unsigned char *good,size_t ppg,
    const unsigned long long *n,unsigned long long ninv)
{s2g_baby_inverse_kernel<NW><<<(unsigned)((count+127)/128),64>>>(parent,children,count,good,ppg,n,ninv,nw);}
template<int NW>
static void s2g_launch_baby_leaf(int nw,size_t count,const unsigned long long *parent,
    const unsigned long long *x,const unsigned long long *z,unsigned long long *out,
    const unsigned char *good,const unsigned long long *n,unsigned long long ninv)
{s2g_baby_leaf_kernel<NW><<<(unsigned)((count+127)/128),64>>>(parent,x,z,count,out,good,n,ninv,nw);}
