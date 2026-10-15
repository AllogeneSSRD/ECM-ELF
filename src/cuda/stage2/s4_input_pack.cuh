#pragma once
// Production pack kernels shared verbatim with the independent pack/zero probe.
namespace stage2_pack {
// sw is positive. For sw>1, floor(2^32/sw) gives a quotient at most one
// below floor(j/sw); a multiply-high and one correction are exact for uint32 j.
__device__ __forceinline__ unsigned quotient(unsigned j,unsigned sw,unsigned reciprocal,unsigned &digit) {
    if(sw==1){digit=0;return j;}
    unsigned q=__umulhi(j,reciprocal),r=j-q*sw;
    const unsigned correction=r>=sw;
    digit=r-correction*sw;return q+correction;
}
template<bool GATHER,bool REVERSE>
__global__ void pack_zero_kernel(const unsigned long long *src,const unsigned long long *offsets,
    unsigned nc,unsigned W,unsigned ndig,unsigned bpw,unsigned sw,unsigned reciprocal,
    unsigned N,unsigned long long first_slice,unsigned long long *dst)
{
    const unsigned j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j>=N)return;
    const unsigned long long slice=first_slice+blockIdx.y;
    unsigned long long value=0;
    if((unsigned long long)j<(unsigned long long)nc*sw) {
        unsigned digit=0;const unsigned coef=quotient(j,sw,reciprocal,digit);
        if(digit<ndig) {
            const unsigned source_coef=REVERSE ? nc-1-coef : coef;
            const auto *c=src+(GATHER ? offsets[slice] : slice*(unsigned long long)nc*W)+
                (unsigned long long)source_coef*W;
            const unsigned bit=digit*bpw,w=bit>>6,shift=bit&63;
            value=c[w]>>shift;
            if(shift+bpw>64 && w+1<W)value|=c[w+1]<<(64-shift);
            value&=(1ull<<bpw)-1;
        }
    }
    dst[slice*(unsigned long long)N+j]=value;
}
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
        pack_zero_kernel<GATHER,REVERSE><<<grid,256>>>(src,offsets,(unsigned)nc,(unsigned)W,
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

__global__ void s4_pack_batch_kernel(const unsigned long long *src, int S, int bpw,
                                     unsigned long long slot_words, unsigned long long N,
                                     unsigned long long ds, unsigned long long ma, int W,
                                     unsigned long long *dst)
{
    const unsigned long long gid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (gid >= ds * ma) return;
    const unsigned long long s = gid / ma, i = gid - s * ma;
    const unsigned long long *c = src + gid * (unsigned long long)W;  /* slice stride = ma*W */
    unsigned long long *d = dst + s * N + i * slot_words;
    const unsigned long long mask = (bpw >= 64) ? ~0ull : ((1ull << bpw) - 1ull);
    const int ndig = (S + bpw - 1) / bpw;
    for (int k = 0; k < ndig; ++k) {
        const int bit = k * bpw;
        const int w = bit >> 6, sh = bit & 63;
        unsigned long long v = c[w] >> sh;
        if (sh + bpw > 64 && (w + 1) < W) v |= c[w + 1] << (64 - sh);   /* bits >= S stay zero */
        d[k] = v & mask;
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
    s4_pack_batch_kernel<<<bl, th>>>(src, S, bpw, slot_words, N, ds, ma, W, dst);
    CK(cudaGetLastError());
}

__global__ void s4_pack_gather_kernel(const unsigned long long *src,
    const unsigned long long *offsets, size_t m, size_t nc, int W, int bits,
    int bpw, unsigned long long sw, unsigned long long N, unsigned long long *dst)
{
    const size_t gid=blockIdx.x*(size_t)blockDim.x+threadIdx.x;
    if(gid>=m*nc)return;
    const size_t slice=gid/nc,coef=gid%nc;
    const auto *c=src+offsets[slice]+coef*W;
    auto *d=dst+slice*N+coef*sw;
    const unsigned long long mask=(1ull<<bpw)-1;
    for(int k=0;k<(bits+bpw-1)/bpw;++k) {
        const int bit=k*bpw,w=bit>>6,shift=bit&63;
        unsigned long long v=c[w]>>shift;
        if(shift+bpw>64 && w+1<W)v|=c[w+1]<<(64-shift);
        d[k]=v&mask;
    }
}
// Separate reverse specialization leaves the original forward gather unchanged.
__global__ void s4_pack_gather_reverse_kernel(const unsigned long long *src,
    const unsigned long long *offsets,size_t m,size_t nc,int W,int bits,
    int bpw,unsigned long long sw,unsigned long long N,unsigned long long *dst)
{
    const size_t gid=blockIdx.x*(size_t)blockDim.x+threadIdx.x;
    if(gid>=m*nc)return;
    const size_t slice=gid/nc,coef=gid%nc;
    const auto *c=src+offsets[slice]+(nc-1-coef)*W;
    auto *d=dst+slice*N+coef*sw;
    const unsigned long long mask=(1ull<<bpw)-1;
    for(int k=0;k<(bits+bpw-1)/bpw;++k) {
        const int bit=k*bpw,w=bit>>6,shift=bit&63;
        unsigned long long v=c[w]>>shift;
        if(shift+bpw>64 && w+1<W)v|=c[w+1]<<(64-shift);
        d[k]=v&mask;
    }
}
