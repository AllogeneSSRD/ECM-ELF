#pragma once

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
