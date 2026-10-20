#pragma once
#include "ecm/cuda/stage2_launch.h"
// Host packing policy, accounting and launch geometry. Device kernels live in
// kernels/cuda/stage2/pack.cuh.
namespace stage2_pack {

static bool supported(unsigned long long N,unsigned long long sw,unsigned long long nc,
    int bits,int bpw,int W)
{
    return N && N<=0xffffffffull && sw && sw<=0xffffffffull && nc<=N/sw &&
        bits>0 && bpw>0 && bpw<64 && W==(bits+63)/64 &&
        (unsigned long long)((bits+bpw-1)/bpw)<=sw;
}
template<bool GATHER,bool REVERSE=false>
static void launch(const unsigned long long *src,const unsigned long long *offsets,
    unsigned long long m,unsigned long long nc,int W,int bits,int bpw,
    unsigned long long sw,unsigned long long N,unsigned long long *dst)
{
    const unsigned reciprocal=sw==1 ? 0 : (unsigned)((1ull<<32)/sw);
    for(unsigned long long first=0;first<m;first+=65535) {
        const unsigned y=(unsigned)std::min(65535ull,m-first);
        const dim3 grid((unsigned)((N+255)/256),y);
        ecm_launch_pack_zero_kernel<GATHER,REVERSE>(EcmCudaLaunch{grid,256}, src,offsets,(unsigned)nc,(unsigned)W,
            (unsigned)((bits+bpw-1)/bpw),(unsigned)bpw,(unsigned)sw,reciprocal,(unsigned)N,first,dst);
        CK(cudaGetLastError());
    }
}
struct Stats {
    unsigned long long contiguous=0,gather=0,reverse=0,launches=0,words=0,avoided_memsets=0;
};
static Stats stats;
static bool enabled(){return fuse_env_ull("NTT_S4_PACK_ZERO",0)!=0;}
static void account(unsigned long long m,unsigned long long N,bool gather,bool reverse=false) {
    if(gather)++stats.gather;else ++stats.contiguous;
    if(reverse)++stats.reverse;
    stats.launches+=(m+65534)/65535;stats.words+=m*N;++stats.avoided_memsets;
}
}



static void s4_launch_pack_batch(const unsigned long long *src, int S, int bpw,
                                 unsigned long long slot_words, unsigned long long N,
                                 unsigned long long ds, unsigned long long ma, int W,
                                 unsigned long long *dst)
{
    const unsigned long long total = ds * ma;
    if (total == 0) return;
    const unsigned int th = 256;
    const unsigned int bl = (unsigned int)((total + th - 1) / th);
    ecm_launch_s4_pack_batch_kernel(EcmCudaLaunch{bl, th}, src, S, bpw, slot_words, N, ds, ma, W, dst);
    CK(cudaGetLastError());
}
