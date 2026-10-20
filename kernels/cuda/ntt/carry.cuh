#pragma once
#include "ecm/cuda/ntt_launch.h"
#include "../shared/goldilocks.cuh"
namespace {
__global__ void carry_extract_kernel(unsigned long long *c, unsigned long long *q,
                                     unsigned long long n, int bpw)
{
    const unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned long long v = c[i];
    q[i] = v >> bpw;
    c[i] = v & ((1ull << bpw) - 1ull);
}
}

void ecm_launch_carry_extract_kernel(EcmCudaLaunch cfg, unsigned long long *c, unsigned long long *q, unsigned long long n, int bpw) {
    carry_extract_kernel<<<cfg.grid,cfg.block,cfg.shared,cfg.stream>>>(c,q,n,bpw);
}

namespace {
__global__ void carry_add_kernel(unsigned long long *c, const unsigned long long *q,
                                 unsigned long long n)
{
    const unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i == 0 || i >= n) return;
    c[i] += q[i - 1];
}
}

void ecm_launch_carry_add_kernel(EcmCudaLaunch cfg, unsigned long long *c, const unsigned long long *q, unsigned long long n) {
    carry_add_kernel<<<cfg.grid,cfg.block,cfg.shared,cfg.stream>>>(c,q,n);
}

namespace {
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
}

void ecm_launch_carry_residual_kernel(EcmCudaLaunch cfg, const unsigned long long *c, unsigned long long n, int bpw, unsigned long long *out, unsigned long long stride) {
    carry_residual_kernel<<<cfg.grid,cfg.block,cfg.shared,cfg.stream>>>(c,n,bpw,out,stride);
}

namespace {
__global__ void slot_assemble_kernel(const unsigned long long *c, unsigned long long slots,
                                     unsigned long long slot_bits, unsigned long long slot_stride,
                                     int bpw, unsigned long long *out, unsigned long long stride)
{
    c += (size_t)blockIdx.y * stride;
    out += (size_t)blockIdx.y * slots;
    const unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i >= slots) return;
    /* NOTE the argument is slot_bits (BITS per coefficient), not the word count: the caller
       passed the word count here once, which silently made the window ~26x too short and the
       assembler returned mostly zeros. */
    const unsigned long long words_per_slot = (slot_bits + (unsigned long long)bpw - 1) /
                                              (unsigned long long)bpw;
    const unsigned long long base_word = (i * slot_stride) / (unsigned long long)bpw;
    const unsigned long long shift = (i * slot_stride) % (unsigned long long)bpw;
    /* The highest word of the window carries only the top `top_bits` bits of THIS slot; its
       remaining bits belong to the NEXT slot and must not be counted.  (The earlier version
       looped one word too far AND masked every word to bpw bits, which fed the next slot's
       low bits into this slot's high bits.  With the word-aligned stride the tail of the
       window is the zero padding of the packed operand.) */
    const unsigned long long top_bits =
        slot_bits - (words_per_slot - 1) * (unsigned long long)bpw;
    unsigned long long v = 0;
    /* Horner from the most significant word down; each word contributes its low bpw bits */
    for (unsigned long long j = words_per_slot; j-- > 0;) {
        unsigned long long x = c[base_word + j] >> shift;
        if (shift != 0) {
            /* c[] holds bpw-BIT DIGITS in 64-bit containers, NOT 64-bit words: the next
               digit's bits belong (bpw - shift) above this digit's low part, so the shift
               is (bpw - shift).  With (64 - shift) every slot except slot 0 (shift == 0)
               came out wrong.  Isolated in tools/bench/slot_extract_model.py, which runs
               this extraction against the EXACT product digits and reports bad=0 for
               P=4/64/128/512 after the change. */
            const unsigned long long nxt = c[base_word + j + 1];
            x |= nxt << (bpw - shift);
        }
        const unsigned long long keep =
            (j == words_per_slot - 1) ? top_bits : (unsigned long long)bpw;
        if (keep < 64) x &= ((1ull << keep) - 1ull);
        /* v * 2^bpw + x, EXACTLY.  v < SLOT_MOD < 2^32 and bpw <= 31, so `v << bpw` cannot
           overflow 64 bits.  (The earlier `((v << (64-bpw)) - v + x)` equals
           v*(2^(64-bpw)-1), which is not v*2^bpw at all, and its `v << (64-bpw)` overflowed
           for most bpw -- which is why the assembler returned garbage.) */
        v = (((v << bpw) % SLOT_MOD) + (x % SLOT_MOD)) % SLOT_MOD;
    }
    out[i] = v;
}
}

void ecm_launch_slot_assemble_kernel(EcmCudaLaunch cfg, const unsigned long long *c, unsigned long long slots, unsigned long long slot_bits, unsigned long long slot_stride, int bpw, unsigned long long *out, unsigned long long stride) {
    slot_assemble_kernel<<<cfg.grid,cfg.block,cfg.shared,cfg.stream>>>(c,slots,slot_bits,slot_stride,bpw,out,stride);
}

template <int ROUNDS>
__device__ __forceinline__ unsigned long long carry_cone_value(
    const unsigned long long *c, unsigned long long i, int bpw)
{
    const unsigned long long mask = (1ull << bpw) - 1ull;
    unsigned long long v[ROUNDS + 1];
    v[0] = c[i];
#pragma unroll
    for (int j = 1; j <= ROUNDS; ++j) v[j] = (i >= (unsigned long long)j) ? c[i - j] : 0ull;
#pragma unroll
    for (int r = 1; r <= ROUNDS; ++r) {
#pragma unroll
        for (int j = 0; j + r <= ROUNDS; ++j) v[j] = (v[j] & mask) + (v[j + 1] >> bpw);
    }
    return v[0];
}

template <int ROUNDS>
__global__ void carry_cone_kernel(const unsigned long long *c, unsigned long long *cout,
                                  unsigned long long n, int bpw, unsigned long long stride)
{
    const size_t sl = (size_t)blockIdx.y * stride;
    const unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    c += sl; cout += sl;
    const unsigned long long mask = (1ull << bpw) - 1ull;
    const unsigned long long x = carry_cone_value<ROUNDS>(c, i, bpw);
    const unsigned int live = __activemask(), lane = threadIdx.x & 31;
    const unsigned int stop = __ballot_sync(live, x != mask);
    const unsigned int generate = __ballot_sync(live, x > mask);
    unsigned int incoming = 0;
    if (lane == 0) {
        unsigned long long j = i;
        while (j) {
            const unsigned long long prev = carry_cone_value<ROUNDS>(c, --j, bpw);
            if (prev != mask) { incoming = prev > mask; break; }
        }
    }
    incoming = __shfl_sync(live, incoming, 0);
    const unsigned int preceding = stop & ((1u << lane) - 1u);
    if (preceding) incoming = (generate >> (31 - __clz(preceding))) & 1u;
    cout[i] = (x + incoming) & mask;
}

template<int ROUNDS>
void ecm_launch_carry_cone_kernel(EcmCudaLaunch cfg, const unsigned long long *c, unsigned long long *cout, unsigned long long n, int bpw, unsigned long long stride) {
    carry_cone_kernel<ROUNDS><<<cfg.grid,cfg.block,cfg.shared,cfg.stream>>>(c,cout,n,bpw,stride);
}
