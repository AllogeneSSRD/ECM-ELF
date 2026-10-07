#pragma once
// Exact Montgomery product for N=2^bits-1, R=2^(64*nw).
// The caller proves the modulus shape and canonical inputs. s2g_mac/ge_n/sub_n
// are the production helpers. The output may alias either input.
template<int NW>
__device__ __noinline__ void s2g_mersenne_mont_reduce(
    unsigned long long *r, const unsigned long long *t,
    const unsigned long long *n, int nw, int bits, unsigned long long *out)
{
    // ab < N^2. The first fold is <2N, hence one subtraction suffices.
    const int word=bits/64,shift=bits%64;
    unsigned long long carry=0;
    for(int i=0;i<nw;++i){
        const unsigned long long low=(i==nw-1 && shift) ? t[i]&n[i] : t[i];
        const unsigned long long high=shift ? (t[i+word]>>shift)|(t[i+word+1]<<(64-shift)) : t[i+word];
        const unsigned long long sum=low+high;
        const unsigned long long c1=sum<low;
        out[i]=sum+carry;
        carry=c1+(out[i]<carry);
    }
    if(carry || s2g_ge_n(out,n,nw))s2g_sub_n(out,n,nw);
    // 2^bits = 1 (mod N): R^-1 is a cyclic right rotation by
    // d=(64*nw-bits) mod bits. Rotation preserves canonicality except
    // the all-one representation, already removed by the subtraction.
    const int d=(64*nw-bits)%bits;
    if(!d){for(int i=0;i<nw;++i)r[i]=out[i];return;}
    const int offset=bits-d,ow=offset/64,os=offset%64;
    const unsigned long long tail=out[0]&((1ull<<d)-1);
    for(int i=0;i<nw;++i){
        unsigned long long v=(out[i]>>d)|(out[i+1<nw?i+1:i]<<(64-d));
        if(i==nw-1)v=out[i]>>d;
        if(i==ow)v|=tail<<os;
        if(os && i==ow+1)v|=tail>>(64-os);
        r[i]=v;
    }
}

// Probe adapter. Production reuses its existing SOS product directly and calls
// only the linear reducer, avoiding a second inlined quadratic loop.
template<int NW>
__device__ __forceinline__ void s2g_mersenne_mont_mul(
    unsigned long long *r,const unsigned long long *a,
    const unsigned long long *b,const unsigned long long *n,int nw,int bits)
{
    unsigned long long t[2*NW+2];
    for(int i=0;i<2*nw+2;++i)t[i]=0;
    for(int i=0;i<nw;++i){
        unsigned long long carry=0,bi=b[i];
        for(int j=0;j<nw;++j){
            unsigned long long lo,hi;
            s2g_mac(a[j],bi,t[i+j],carry,lo,hi);
            t[i+j]=lo;carry=hi;
        }
        t[i+nw]=carry;
    }
    unsigned long long out[NW];
    s2g_mersenne_mont_reduce<NW>(r,t,n,nw,bits,out);
}
