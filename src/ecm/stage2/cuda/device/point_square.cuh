#pragma once
// Included after point.cuh's common Montgomery reducer. Canonical a<n and odd n
// are the same preconditions as s2g_mont_mul. Finish the product before touching
// r, including r==a. W*(W+1)/2 partial products replace the W*W full product;
// the ordinary REDC or exact-Mersenne fold remains unchanged.
template<int NW>
__device__ __forceinline__ void s2g_mont_sqr(unsigned long long *r,
    const unsigned long long *a,const unsigned long long *n,
    unsigned long long ninv,int nw)
{
    unsigned long long t[2*NW+2];
    for(int i=0;i<2*nw+2;++i)t[i]=0;
    // Strict upper triangle: carry at i+nw is untouched before this row.
    for(int i=0;i<nw;++i) {
        unsigned long long carry=0;
        const auto ai=a[i];
        for(int j=i+1;j<nw;++j) {
            unsigned long long lo,hi;
            s2g_mac(ai,a[j],t[i+j],carry,lo,hi);
            t[i+j]=lo;carry=hi;
        }
        t[i+nw]=carry;
    }
    // Double cross terms as a single multiword shift, without doubling each
    // 128-bit product separately or losing its 129th bit.
    unsigned long long carry=0;
    for(int i=0;i<2*nw;++i) {
        const auto v=t[i];t[i]=(v<<1)|carry;carry=v>>63;
    }
    t[2*nw]=carry;
    // Diagonals occupy adjacent, nonoverlapping 128-bit positions. Propagate
    // one addition carry between pairs, rather than an unbounded ripple loop.
    carry=0;
    for(int i=0;i<nw;++i) {
        const auto ai=a[i],lo=ai*ai,hi=__umul64hi(ai,ai);
        const auto x=t[2*i]+lo;
        auto next=(x<t[2*i])?1ull:0ull;
        const auto y=x+carry;next+=(y<x)?1ull:0ull;
        const auto z=t[2*i+1]+hi;
        carry=(z<t[2*i+1])?1ull:0ull;
        const auto v=z+next;carry+=(v<z)?1ull:0ull;
        t[2*i]=y;t[2*i+1]=v;
    }
    t[2*nw]+=carry;
    s2g_mont_reduce<NW>(r,t,n,ninv,nw);
}
