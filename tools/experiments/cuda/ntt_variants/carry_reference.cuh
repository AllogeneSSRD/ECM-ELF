#pragma once
__global__ void carry_residual_kernel(const unsigned long long *c, unsigned long long n,
                                      int bpw, unsigned long long *out,
                                      unsigned long long stride)
{
    c += (size_t)blockIdx.y * stride;
    out += 2 * (size_t)blockIdx.y;
    const unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    const unsigned long long lim = 1ull << bpw;
    unsigned long long bad = 0, mx = 0;
    if (i < n) {
        const unsigned long long v = c[i];
        if (v >= lim) bad = 1;
        /* max digit height, in a second slot of `out`: 64 - clz, and 0 for the digit 0 --
           exactly the value of the old `while (t) { ++h; t >>= 1; }` loop */
        mx = v ? (unsigned long long)(64 - __clzll((long long)v)) : 0ull;
    }
    /* THE REDUCTION IS PER WARP, THEN PER BLOCK -- NOT PER THREAD (objective 4, section 41).
       The old kernel had EVERY thread in the grid doing an atomicAdd AND an atomicMax on the
       SAME two globals: for a slice of N words that is 2*N/32 serialised global atomics for a
       value that is a single 64-bit count, and it was 60.6 s at the production shape (24% of
       ntt_seconds).  A first attempt folded per BLOCK with *shared* 64-bit atomics and came out
       2.6x SLOWER at B2=1e11 (measured: t_check 1.864 -> 4.858 s), because 64-bit shared-memory
       atomics are emulated and serialise ~32 deep per warp -- so the fold starts in the warp
       (shuffles, no memory traffic) and only lane 0 of each warp touches shared memory, leaving
       one global atomic pair per block.  The results are bit-identical to the old kernel's. */
    const unsigned mask = 0xffffffffu;
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        bad += __shfl_down_sync(mask, bad, off);
        const unsigned long long o = __shfl_down_sync(mask, mx, off);
        if (o > mx) mx = o;
    }
    __shared__ unsigned long long sh_bad, sh_mx;
    if (threadIdx.x == 0) { sh_bad = 0ull; sh_mx = 0ull; }
    __syncthreads();
    if ((threadIdx.x & 31u) == 0u) {
        if (bad) atomicAdd(&sh_bad, bad);
        atomicMax(&sh_mx, mx);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        if (sh_bad) atomicAdd(out, sh_bad);
        atomicMax((unsigned long long *)&out[1], sh_mx);
    }
}
