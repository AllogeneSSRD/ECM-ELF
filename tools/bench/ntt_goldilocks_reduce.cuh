#pragma once

// For canonical x and 1<=k<=32, write x=h*2^k+r. Since
// 2^-k = 2^(32-k)-2^(64-k) mod GL_P, x/2^k is
// (h+r*2^(32-k)) - r*2^(64-k). Both operands are canonical:
// the first <2^(64-k)+2^32, the second <=2^64-2^(64-k)<GL_P.
// A borrowed subtraction is corrected by subtracting epsilon modulo 2^64.
__host__ __device__ __forceinline__ unsigned long long gl_scale_inverse_pow2(
    unsigned long long x, int k)
{
    const auto r=x&((1ull<<k)-1);
    const auto a=(x>>k)+(r<<(32-k)), b=r<<(64-k);
    const auto difference=a-b;
    return difference-((a<b) ? 0xffffffffull : 0ull);
}

__host__ __device__ __forceinline__ bool gl_scale_is_inverse_pow2(
    unsigned long long scale, int k)
{
    // Short circuit before shifting: arbitrary/custom scales retain gl_mul.
    return k>=1 && k<=32 && scale==GL_P-(1ull<<(64-k))+(1ull<<(32-k));
}

// Exact canonical remainder of any unsigned128-bit value. GL_P is supplied by
// the includer. Write hi=hi_high*2^32+hi_low; 2^64=epsilon, 2^96=-1 modq.
__host__ __device__ __forceinline__ unsigned long long gl_reduce128_short(
    unsigned long long lo,unsigned long long hi)
{
    constexpr unsigned long long epsilon=0xffffffffull;
    const auto minus=lo-(hi>>32);
    const auto a=minus-((minus>lo) ? epsilon : 0ull);
    const auto b=((hi&epsilon)<<32)-(hi&epsilon);
    const auto sum=a+b;
    // If sum carried, sum<=2^64-epsilon-2, so adding epsilon cannot carry again.
    const auto value=sum+((sum<a) ? epsilon : 0ull);
    return value>=GL_P ? value-GL_P : value;
}
