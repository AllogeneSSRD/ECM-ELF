#pragma once

// Canonical inputs only: 0 <= a,b < q = 2^64-(2^32-1).
// Each complete CC chain is contained in one asm block.
__device__ __forceinline__ unsigned long long gl_sub_canonical_ptx(
    unsigned long long a,unsigned long long b)
{
    unsigned long long out;
    asm("{\n\t"
        ".reg .u32 a0,a1,b0,b1,l,h,c;\n\t"
        "mov.b64 {a0,a1},%1;\n\t"
        "mov.b64 {b0,b1},%2;\n\t"
        "sub.cc.u32 l,a0,b0;\n\t"
        "subc.cc.u32 h,a1,b1;\n\t"
        "subc.u32 c,0,0;\n\t"
        // A borrow means d=a-b+2^64. Subtract epsilon to get a-b+q.
        "sub.cc.u32 l,l,c;\n\t"
        "subc.u32 h,h,0;\n\t"
        "mov.b64 %0,{l,h};\n\t"
        "}" : "=l"(out) : "l"(a),"l"(b));
    return out;
}
