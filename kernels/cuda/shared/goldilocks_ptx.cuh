#pragma once

// Independently derived canonical Goldilocks remainder, for arbitrary lo/hi.
// Same identity as gl_reduce128_short: lo-hi_high+hi_low*(2^32-1).
// Keep every CC chain in one asm block; CC is not preserved across calls.
__device__ __forceinline__ unsigned long long gl_reduce128_ptx(
    unsigned long long lo,unsigned long long hi)
{
    unsigned long long out;
    asm("{\n\t"
        ".reg .u32 l,h,z,hh,bl,bh,c;\n\t"
        ".reg .pred p,p2;\n\t"
        "mov.b64 {l,h},%1;\n\t"
        "mov.b64 {z,hh},%2;\n\t"
        "sub.cc.u32 l,l,hh;\n\t"
        "subc.cc.u32 h,h,0;\n\t"
        "subc.u32 c,0,0;\n\t"       // borrow -> 0 or -1
        "sub.cc.u32 l,l,c;\n\t"     // subtract epsilon if borrowed
        "subc.u32 h,h,0;\n\t"
        "sub.cc.u32 bl,0,z;\n\t"
        "subc.u32 bh,z,0;\n\t"     // b=z*(2^32-1)
        "add.cc.u32 l,l,bl;\n\t"
        "addc.cc.u32 h,h,bh;\n\t"
        "addc.u32 c,0,0;\n\t"
        "sub.u32 c,0,c;\n\t"       // carry -> 0 or -1 (epsilon low word)
        "add.cc.u32 l,l,c;\n\t"
        "addc.u32 h,h,0;\n\t"      // canonical correction cannot overflow again
        "setp.eq.u32 p,h,0xffffffff;\n\t"
        "setp.ne.u32 p2,l,0;\n\t"
        "and.pred p,p,p2;\n\t"
        "@p sub.u32 l,l,1;\n\t"
        "@p mov.u32 h,0;\n\t"
        "mov.b64 %0,{l,h};\n\t"
        "}" : "=l"(out) : "l"(lo),"l"(hi));
    return out;
}

// Explicit four-limb 64x64 product. No field-range assumption. The two middle
// accumulations retain carry into limb3 before adding the final high product.
__device__ __forceinline__ unsigned long long gl_mul_ptx(
    unsigned long long a,unsigned long long b)
{
    unsigned long long lo,hi;
    asm("{\n\t"
        ".reg .u32 a0,a1,b0,b1,r0,r1,r2,r3;\n\t"
        "mov.b64 {a0,a1},%2;\n\t"
        "mov.b64 {b0,b1},%3;\n\t"
        "mul.lo.u32 r0,a0,b0;\n\t"
        "mul.hi.u32 r1,a0,b0;\n\t"
        "mad.lo.cc.u32 r1,a1,b0,r1;\n\t"
        "madc.hi.u32 r2,a1,b0,0;\n\t"
        "mad.lo.cc.u32 r1,a0,b1,r1;\n\t"
        "madc.hi.cc.u32 r2,a0,b1,r2;\n\t"
        "addc.u32 r3,0,0;\n\t"
        "mad.lo.cc.u32 r2,a1,b1,r2;\n\t"
        "madc.hi.u32 r3,a1,b1,r3;\n\t"
        "mov.b64 %0,{r0,r1};\n\t"
        "mov.b64 %1,{r2,r3};\n\t"
        "}" : "=l"(lo),"=l"(hi) : "l"(a),"l"(b));
    return gl_reduce128_ptx(lo,hi);
}
