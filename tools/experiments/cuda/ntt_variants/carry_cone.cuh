#pragma once
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

/* After the height-reduction cone x <= radix.  Thus its remaining carry is binary:
   x==radix generates, x==radix-1 propagates, all smaller x kill.  A warp ballot finds
   the nearest preceding non-propagating digit in the warp.  Lane zero determines the
   incoming carry by reading immutable cone values backwards across warp/block boundaries.
   Only one lane does that lookback; no global scratch, host drain or extra launch is needed.
   The worst case is a long all-mask run (lookback is linear per warp), but the result is
   exact for any length, rather than depending on an empirical number of ripple rounds. */
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
