/* ---------------------------------------------------------------------------
 * ntt_poly_probe.cu -- M0 follow-up of docs/DEV_STAGE2_GPU_PLAN.md sections 8.1/8.3/8.4:
 * the SAME figure of merit as tools/bench/cufft_kron_probe.cu poly mode (ns per
 * operand-bit of a Kronecker polynomial multiply), but the convolution is done by an
 * INTEGER NTT over the Goldilocks prime instead of an fp64 cuFFT.
 *
 * Why: on consumer Ada fp64 runs at 1/64 of fp32, and cuFFT's double-precision path
 * eats exactly that.  An integer NTT has no such penalty.  This probe is deliberately a
 * CORRECTNESS-FIRST, SLOW-BUT-CORRECT first cut: one kernel launch per stage, no stage
 * fusion, no shared-memory tiling of several stages.  The pass count can be read off the
 * printed `stages=` field, and getting that number down is the next optimisation round.
 *
 * Transform:
 *   p = 2^64 - 2^32 + 1 = 0xFFFFFFFF00000001 ("Goldilocks"), p - 1 = 2^32 * (2^32 - 1),
 *   so power-of-two transform lengths exist up to 2^32.  Primitive root 7, thus
 *   omega = 7^((p-1)/N) mod p is a primitive N-th root of unity.
 *   Reduction: 2^64 == 2^32 - 1 (mod p), so for x = hi*2^64 + lo
 *      x == lo + hi*(2^32 - 1) = (lo - hi) + (hi << 32)  (mod p),
 *   folded once more, then one conditional subtract (sometimes two -- verified against
 *   GMP for 200k random and boundary pairs at startup, and mpz_powm agrees).
 *
 * Exactness rule (the whole point):
 *   every convolution coefficient is a sum of at most L products of two bpw-bit words,
 *   where L = P * slot_words is the number of NONZERO digits per operand (the arrays are
 *   zero beyond the payload, so the array length N is NOT the term count), so the NTT is
 *   exact iff  L * (2^bpw - 1)^2 < p.  bpw is chosen from that bound and the bound is
 *   ASSERTED with exact integer arithmetic before anything runs, from L computed out of
 *   P and slot_words; in addition, whenever the full convolution fits in host memory the
 *   TRUE maximum coefficient is recomputed with GMP from the packed digits and asserted
 *   < p, and its ratio to (2^bpw-1)^2 is printed next to L.  The result is verified
 *   coefficient by coefficient against a GMP schoolbook product, projected modulo the
 *   32-bit prime 4294967291 (identical to the cuFFT probe, so the two probes are directly
 *   comparable).  The tighter bound is worth 2x on the whole probe: P=8192/S=5153 went from
 *   bpw=18/N=2^24 to bpw=21/N=2^23, halving every one of the 13 array passes.
 *
 * Usage (identical CLI to the cuFFT probe):
 *   ntt_poly_probe.exe poly <P> <S> [device] [verify]
 *   ntt_poly_probe.exe bench <bits> [bpw] [device]
 *
 * Machine-readable result line (same field names as the cuFFT probe's poly line, with
 * bpw/nwords/ns_per_coeff/ns_per_operand_bit added):
 *   poly: mode=poly P=8192 S=5153 slot_bits=10319 bpw=21 nwords=8388608 fft=8388608
 *         mem_mb=256 ok=1 t_total=0.014 ns_per_coeff=1700.0 ns_per_operand_bit=0.084
 * ------------------------------------------------------------------------- */
#ifndef NTT_PROBE_NAME
#define NTT_PROBE_NAME "ntt_poly_probe"
#endif

#include <cuda_runtime.h>

#include <gmp.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

/* Every CUDA call is checked, and a failure says WHICH call, WHICH line, WHAT the driver
   answered, HOW MANY bytes it was asked for (when the call carries a size) and how much device
   memory was free at that moment.  The real-shape deaths of section 18.3 arrived as a bare
   "exit=1" with an empty stderr, so "the CUDA call failed" has to be excluded loudly rather
   than silently.  An OOM here is fatal on purpose: continuing past a failed allocation reads
   freed memory. */
#define CK(x)                                                                        \
    do {                                                                             \
        const cudaError_t e_ = (x);                                                  \
        if (e_ != cudaSuccess) {                                                      \
            size_t ck_free_ = 0, ck_total_ = 0;                                       \
            cudaMemGetInfo(&ck_free_, &ck_total_);                                    \
            std::fprintf(stderr,                                                      \
                         "CUDA error %s (%d) at %s:%d\n  call: %s\n  device free=%.1f " \
                         "MB of %.1f MB (%.1f MB in use)\n",                         \
                         cudaGetErrorString(e_), (int)e_, __FILE__, __LINE__, #x,     \
                         (double)ck_free_ / 1048576.0, (double)ck_total_ / 1048576.0, \
                         (double)(ck_total_ - ck_free_) / 1048576.0);                 \
            std::fflush(stderr);                                                      \
            std::exit(2);                                                            \
        }                                                                            \
    } while (0)

namespace {

/* The one prime.  Everything below is normalised into [0, p). */
#define GL_P 0xFFFFFFFF00000001ull

/* Modular reduction, device and host (this is the whole arithmetic core, so it is shared
   and unit-tested against GMP before any kernel runs).
 *
 * One fold step uses 2^64 == 2^32 - 1 (mod p):
 *     x = hi*2^64 + lo  ==  lo + hi*(2^32 - 1)  (mod p)          (*)
 * The right-hand side is NOT a 64-bit quantity (hi*(2^32-1) reaches ~2^96 when
 * hi is near 2^64), so the exact sum is kept as its own (lo', hi') pair and folded again --
 * this is the "fold BOTH parts" trap the parent flagged for Goldilocks (the same trap as
 * 2^64 == 8 mod M61).  The exact sum is
 *     lo' = lo + hi*0xFFFFFFFF        (mod 2^64)
 *     hi' = carries of the same sum, plus hi's contribution shifted out of lo'  ...
 * which is simplest to write as the two exact partial products below:
 *     lo + hi*0xFFFFFFFF = lo + (hi << 32) - hi
 * so track lo' = lo + (hi<<32) - hi with an explicit carry/borrow pair.
 *
 * Provably at most THREE folds are needed for any 128-bit input: hi <= 2^64-1 gives
 * hi' <= 2^33, then hi'' <= 2, then hi''' == 0 (checked exhaustively against GMP over
 * 200k random products plus every boundary pair at startup -- the earlier versions of this
 * function were wrong in three different ways and the GMP selftest caught each one).
 * No conditional subtract is needed at the end: the converged value is already < p.
 *
 * BRANCH-FREE (only the THREE final conditional subtracts remain): the four folds are run
 * unconditionally.  A fold with hi == 0 is a PROVABLE no-op -- hi = 0 gives hl = hh = 0,
 * c1 = 0 and b = 0, hence lo' = lo and hi' = 0 -- and hi reaches 0 after the third fold
 * (hi <= 2^33 after one, <= 2 after two, == 0 after three), so the iteration that follows
 * cannot change anything and `while (hi != 0)` was only skipping work, never changing the
 * answer.  The loop test was a data-dependent branch (it depends on the reduced value, so
 * it is divergent across a warp and unpredictable); dropping it measured ~3% faster.  The
 * equivalence is not assumed: the startup selftest still compares this function against GMP
 * on 3024 host + 200000 device cases and 100064 (lo,hi) folds, boundary pairs included.
 */
__host__ __device__ inline unsigned long long gl_reduce(unsigned long long lo,
                                                        unsigned long long hi)
{
    for (int iter = 0; iter < 4; ++iter) {
        /* exact: lo + hi*(2^32-1) = lo + (hi<<32) - hi, split into (lo', hi') */
        const unsigned long long hl = hi << 32;
        const unsigned long long hh = hi >> 32;
        /* s1 = lo + hl, carry c1 */
        const unsigned long long s1 = lo + hl;
        const unsigned long long c1 = (s1 < lo) ? 1ull : 0ull;
        /* s2 = s1 - hi, borrow b */
        const unsigned long long s2 = s1 - hi;
        const unsigned long long b = (s1 < hi) ? 1ull : 0ull;
        /* hi' = hh (from hl's high half) + c1 - b ; hh < 2^32, so this fits and stays
           in [0, 2^33] */
        lo = s2;
        hi = hh + c1 - b;
    }
    if (lo >= GL_P) lo -= GL_P;
    if (lo >= GL_P) lo -= GL_P;
    if (lo >= GL_P) lo -= GL_P;
    return lo;
}

__host__ __device__ inline unsigned long long gl_mod(unsigned long long x)
{
    return gl_reduce(x, 0);
}

__device__ __forceinline__ unsigned long long gl_mul(unsigned long long a, unsigned long long b)
{
    return gl_reduce(a * b, __umul64hi(a, b));
}

/* 128-bit product as two 64-bit halves -- the same decomposition __umul64hi compiles to,
   written so that every step is a fact about the LOW word:
       low1  = al*bl + ((al*bh) << 32)
       low2  = low1 + ((ah*bl) << 32)
       lo    = low2
       carry = [low1 < (al*bh)<<32] + [low2 < (ah*bl)<<32]
       hi    = ah*bh + (al*bh >> 32) + (ah*bl >> 32) + carry
   (Each (x<<32) term is the low word of that partial product, its high 32 bits going into
   hi; the two overflow bits are the carries into bit 64.)  Four earlier versions of this
   function were wrong in different ways and the GMP selftest caught each one, so this is
   now cross-checked against GMP both as a split against an exact 128-bit reference and as
   a full multiply, host and device. */
static inline void mul_64x64_hi_lo(unsigned long long a, unsigned long long b,
                                   unsigned long long &lo, unsigned long long &hi)
{
    const unsigned long long al = a & 0xFFFFFFFFull, ah = a >> 32;
    const unsigned long long bl = b & 0xFFFFFFFFull, bh = b >> 32;
    const unsigned long long p0 = al * bl;
    const unsigned long long p1 = al * bh;         /* high word p1>>32 */
    const unsigned long long p2 = ah * bl;         /* high word p2>>32 */
    const unsigned long long s1 = p0 + (p1 << 32);
    const unsigned long long c1 = (s1 < (p1 << 32)) ? 1ull : 0ull;
    const unsigned long long s2 = s1 + (p2 << 32);
    const unsigned long long c2 = (s2 < (p2 << 32)) ? 1ull : 0ull;
    lo = s2;
    hi = ah * bh + (p1 >> 32) + (p2 >> 32) + c1 + c2;
}

/* host copy of gl_mul: __umul64hi is device-only, and nvcc's HOST pass of a .cu does not
   accept unsigned __int128 at all (verified: "expected a ;" on the typedef), so the host
   takes the 128-bit product apart with the portable split above.  The reduction is the SAME
   gl_reduce the device uses, so the two paths differ only in how the product is formed, and
   the selftest compares BOTH against GMP. */
static inline unsigned long long gl_mul_host(unsigned long long a, unsigned long long b)
{
    unsigned long long lo = 0, hi = 0;
    mul_64x64_hi_lo(a, b, lo, hi);
    return gl_reduce(lo, hi);
}

__host__ __device__ inline unsigned long long gl_add(unsigned long long a, unsigned long long b)
{
    unsigned long long s = a + b;
    if (s < a || s >= GL_P) s -= GL_P;
    return s;
}

__host__ __device__ inline unsigned long long gl_sub(unsigned long long a, unsigned long long b)
{
    return (a >= b) ? (a - b) : (a + (GL_P - b));
}

/* device alias: reduce an already-64-bit value (x < 2^64, so this is x - p when x >= p) */
__device__ __forceinline__ unsigned long long gl_mod_dev(unsigned long long x)
{
    return gl_reduce(x, 0);
}

/* device-only aliases used inside kernels */
__device__ __forceinline__ unsigned long long gl_sub_dev(unsigned long long a, unsigned long long b)
{
    return gl_sub(a, b);
}

__device__ __forceinline__ unsigned long long gl_add_dev(unsigned long long a, unsigned long long b)
{
    return gl_add(a, b);
}

/* host copy (see gl_mul_host) -- used to build omega, 1/N and the inverse root */
static inline unsigned long long gl_pow_host(unsigned long long a, unsigned long long e)
{
    unsigned long long r = 1 % GL_P;
    a %= GL_P;
    while (e) {
        if (e & 1ull) r = gl_mul_host(r, a);
        a = gl_mul_host(a, a);
        e >>= 1;
    }
    return r;
}

static inline unsigned long long gl_add_host(unsigned long long a, unsigned long long b)
{
    unsigned long long t = a + b;
    if (t < a || t >= GL_P) t -= GL_P;
    return t;
}

static inline unsigned long long gl_sub_host(unsigned long long a, unsigned long long b)
{
    return (a >= b) ? (a - b) : (a + (GL_P - b));
}

double now_s()
{
    using clock = std::chrono::steady_clock;
    static const clock::time_point t0 = clock::now();
    return std::chrono::duration<double>(clock::now() - t0).count();
}

inline int bitlen_u64(unsigned long long v)
{
    int n = 0;
    while (v) { ++n; v >>= 1; }
    return n;
}

/* ---- packing: bits [bit, bit+nbits) of a packed word array, LSB first --------------- */
inline unsigned long long bits_of(const unsigned long long *w, unsigned long long nw,
                                  unsigned long long bit, int nbits)
{
    const unsigned long long wi = bit >> 6;
    const int off = (int)(bit & 63ull);
    unsigned long long v = (wi < nw) ? w[(size_t)wi] : 0ull;
    v >>= off;
    if (off + nbits > 64) {
        const unsigned long long nxt = (wi + 1 < nw) ? w[(size_t)(wi + 1)] : 0ull;
        v |= nxt << (64 - off);
    }
    if (nbits < 64) v &= ((1ull << nbits) - 1ull);
    return v;
}

/* the same on a vector; it forwards, so there is still exactly one implementation (the
   packing loop of ntt_poly_mul_host takes raw pointers, see the extraction note there) */
inline unsigned long long bits_of(const std::vector<uint64_t> &w, unsigned long long bit,
                                  int nbits)
{
    return bits_of(w.data(), (unsigned long long)w.size(), bit, nbits);
}

/* ---- kernels ----------------------------------------------------------------------- */

/*
 * Twiddles are computed IN the kernel rather than looked up.  Reason: the table is
 * omega^0 .. omega^(N/2-1), i.e. N/2 * 8 bytes = 2.1 GB at N = 2^29 (P = 65000), and that
 * does not fit next to the working array on the 8 GB device 1 -- the first version
 * allocated one and would not have run at the largest shape.  In-kernel cost is
 * log2(N) <= 29 modular squarings per element (the first twiddle is 1, so the loop starts
 * at the second bit) and the exponent is reduced with floating point (exact: the product
 * j*step is at most N, and 2^34 * eps < 1e-5).  The two big transforms are bandwidth
 * bound, so this is the right trade at these sizes.
 */
__device__ __forceinline__ unsigned long long gl_twiddle(unsigned long long omega,
                                                         unsigned long long e)
{
    if (e == 0) return 1ull;                     /* omega^0: the first stage's j = 0 */
    unsigned long long r = omega;
    int top = 63;
    while (((e >> top) & 1ull) == 0) --top;      /* e >= 1 always here */
    for (int i = top - 1; i >= 0; --i) {
        r = gl_mul(r, r);
        if ((e >> i) & 1ull) r = gl_mul(r, omega);
    }
    return r;
}

/*
 * (a * b) mod N, reduced EXACTLY, for a, b < N = 2^k.
 *
 * The point is that a*b can be ~N^2 (up to 2^58 here), which does NOT fit in 64 bits, and
 * neither nvcc pass of this .cu accepts a 128-bit integer type (verified: both the host and
 * the device pass reject unsigned __int128).  So the product is evaluated MODULO p instead:
 * the true product a*b is < p (2^58 << 2^64), hence (a mod p)*(b mod p) mod p == a*b
 * exactly, with every intermediate staying in [0, p).  One conditional subtract then brings
 * the result into [0, N) because a*b < N^2 and N^2 < p.
 *
 * The first version reduced this exponent with single-precision float
 * (q = (float)a * (float)b * __frcp_rn((float)n)); that is only exact while a*b < 2^24,
 * whereas a*b here reaches N^2/2 -- so for N >= 2^13 the low bits of the product, and with
 * them the twiddle exponent, were WRONG for most butterflies.  The twiddle kernel itself
 * (checking omega^i against the host) passed the whole time, which is exactly why the bug
 * showed up as "every coefficient wrong" instead of "the twiddles are wrong".
 */
__device__ __forceinline__ unsigned long long gl_mul_mod_n(unsigned long long a,
                                                           unsigned long long b,
                                                           unsigned long long n)
{
    const unsigned long long ap = gl_mod_dev(a);
    const unsigned long long bp = gl_mod_dev(b);
    unsigned long long e = gl_mul(ap, bp);
    if (e >= n) e -= n;
    if (e >= n) e -= n;
    return e;
}

static inline unsigned long long gl_mul_mod_n_host(unsigned long long a, unsigned long long b,
                                                   unsigned long long n)
{
    const unsigned long long ap = gl_reduce(a, 0);
    const unsigned long long bp = gl_reduce(b, 0);
    unsigned long long e = gl_mul_host(ap, bp);
    if (e >= n) e -= n;
    if (e >= n) e -= n;
    return e;
}

/* A[i] *= w^e, in place.  Element i of the (bit-reversed) array is the slot
   bitrev(i), and omega^bitrev(i) is the point index i, so the pointwise product needs no
   permutation.  ngen == 1 means "exponent 1", used for the pointwise product itself. */
__global__ void pointwise_twiddle_kernel(unsigned long long *a, const unsigned long long *b,
                                         unsigned long long n, unsigned long long omega,
                                         unsigned long long exp_scale)
{
    const unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned long long e = gl_mul_mod_n(i, exp_scale, n);
    a[i] = gl_mul(gl_mul(a[i], b[i]), gl_twiddle(omega, e));
}

/*
 * Radix-2 butterfly stage, in place, one launch per stage -- SLOW ON PURPOSE (this is the
 * correctness-first cut; fusing stages is the next optimisation round).
 *
 * One generic Cooley-Tukey stage, parameterised by (half, exp_step):
 *     stride = 2*half,  M = 2*stride = N/exp_step,  groups = N/stride
 *     within group g (base = g*stride), pair index j in [0, half):
 *         u = A[base + j], v = A[base + j + half]
 *         A[base + j]        = u + v
 *         A[base + j + half] = (u - v) * omega^((j * N) / M)) = omega^(j * exp_step)
 * Threads are laid out so that the `half`-chunk is contiguous == coalesced.
 *
 * FORWARD (DIT, i.e. "decimation in time", used with a preceding bit-reversal, see
 * bitrev_kernel) starts at half = 1 and doubles:   half = 2^s, exp_step = N/(2*half).
 * INVERSE (DIF, decimation in frequency) starts at half = N/4 and halves:
 *   half = N/2^(s+1), exp_step = N/(2*half) = 2^s, and it is fed by a bit-reversal too.
 * A DIF forward MUST be paired with a DIT inverse and vice versa -- the earlier version
 * used "stride = N >> (stage+1)" for the forward, which is the DIF order with the
 * twiddles of DIT and silently dropped the final stage (half == 0); the device-vs-naive-DFT
 * check (dftcheck) caught it: the fused stages disagreed with an in-kernel O(n^2) DFT while
 * the twiddles themselves agreed.
 */
__device__ __forceinline__ void ntt_stage(unsigned long long *a, unsigned long long half,
                                          unsigned long long exp_step,
                                          unsigned long long omega, unsigned long long n)
{
    const unsigned long long stride = 2 * half;
    const unsigned long long groups = n / stride;
    const unsigned long long tid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (tid >= groups * half) return;
    const unsigned long long g = tid / half;
    const unsigned long long j = tid - g * half;
    const unsigned long long i0 = g * stride + j;
    const unsigned long long i1 = i0 + half;
    const unsigned long long tw = gl_twiddle(omega, gl_mul_mod_n(j, exp_step, n));
    const unsigned long long u = a[i0];
    const unsigned long long v = a[i1];
    a[i0] = gl_add_dev(u, v);
    a[i1] = gl_mul(gl_sub_dev(u, v), tw);
}

/* DIT (Cooley-Tukey) butterfly: twiddle BEFORE the add/subtract; SAME index decomposition and
   exp_step as the DIF above, only the multiplication moves.  This is the canonical inverse of
   the DIF above: DIF produces the frequency vector in BIT-REVERSED order and the DIT consumes
   it in that order, so NO bit-reversal kernel is needed anywhere between them. */
__device__ __forceinline__ void dit_stage(unsigned long long *a, unsigned long long half,
                                          unsigned long long exp_step,
                                          unsigned long long omega, unsigned long long n)
{
    const unsigned long long stride = 2 * half;
    const unsigned long long groups = n / stride;
    const unsigned long long tid = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (tid >= groups * half) return;
    const unsigned long long g = tid / half;
    const unsigned long long j = tid - g * half;
    const unsigned long long i0 = g * stride + j;
    const unsigned long long i1 = i0 + half;
    const unsigned long long tw = gl_twiddle(omega, gl_mul_mod_n(j, exp_step, n));
    const unsigned long long u = a[i0];
    const unsigned long long v = gl_mul(a[i1], tw);
    a[i0] = gl_add_dev(u, v);
    a[i1] = gl_sub_dev(u, v);
}

__global__ void dit_stage_kernel(unsigned long long *a, unsigned long long half,
                                 unsigned long long exp_step, unsigned long long omega,
                                 unsigned long long n)
{
    dit_stage(a, half, exp_step, omega, n);
}

__global__ void ntt_stage_kernel(unsigned long long *a, unsigned long long half,
                                 unsigned long long exp_step, unsigned long long omega,
                                 unsigned long long n)
{
    ntt_stage(a, half, exp_step, omega, n);
}

/* dst[i] = src[bitrev(i, logn)] -- the inverse permutation is itself (evaluated on i). */
__global__ void bitrev_kernel(const unsigned long long *src, unsigned long long *dst,
                              unsigned long long logn, unsigned long long n)
{
    const unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned long long r = 0, v = i;
    for (unsigned long long k = 0; k < logn; ++k) { r = (r << 1) | (v & 1ull); v >>= 1; }
    dst[i] = src[r];
}

/* In-place bit-reversal by SWAPS: safe when one buffer is both source and destination
   (the non-swap form `dst[i] = src[bitrev(i)]` on one buffer is a read-after-write race). */
__global__ void bitrev_swap_kernel(unsigned long long *a, unsigned long long logn,
                                   unsigned long long n)
{
    const unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned long long r = 0, v = i;
    for (unsigned long long k = 0; k < logn; ++k) { r = (r << 1) | (v & 1ull); v >>= 1; }
    if (i < r) {
        const unsigned long long t = a[i];
        a[i] = a[r];
        a[r] = t;
    }
}

/* ---- carry stage ------------------------------------------------------------------- */
/*
 * After the inverse NTT every word holds a raw convolution coefficient (up to ~2^(k+2*bpw),
 * i.e. ~2^60) and must be reduced back to bpw bits, with the excess carried into the NEXT
 * word.  The carry is EXACTLY the classic redundant-digit normalisation and it is a ripple:
 *
 *     q[i]   = c[i] >> bpw          (the raw quotient of digit i)
 *     c[i]  &= mask                 (digit i becomes canonical)
 *     c[i]  += q[i-1]               (digit i receives the quotient of its left neighbour)
 *
 * repeated until every digit is < 2^bpw.  One round is NOT enough (q[i-1] can be up to
 * 2^(k+bpw), so the digits are still taller than bpw bits afterwards), but the digit height
 * falls by ~bpw bits per round.  The height bound only ensures digits <= 2^bpw;
 * it does NOT bound convergence: a unit carry can cross an arbitrarily long run of
 * digits equal to 2^bpw-1.  The fused kernel resolves that final binary carry exactly.
 *
 * WHY THE PREVIOUS THREE-KERNEL SCHEME WAS WRONG (this was the bug): it replaced q[i-1] by an
 * exclusive PREFIX SUM of all the quotients below i, `c[i] += sum_{j<i} q[j]`.  For the exact
 * carry, a quotient q[j] must enter digit i with weight 2^(bpw*(j+1-i)), not weight 1, so the
 * prefix sum OVER-ESTIMATES the carry and does not even preserve the value: for digits
 * c = [2^bpw, 2^bpw] (value 2^bpw + 2^2bpw, canonical digits d1=1, d2=1) it produced
 * c''_1 = 1, c''_2 = q0 + q1 = 2, i.e. 2^bpw + 2*2^2bpw.  (The inter-block offsets were
 * added on top of that, and carry_scan_blocks_kernel also indexed its 256-entry shared array
 * with j up to nb-1, i.e. out of bounds for N >= 65536 -- both are gone with this scheme:
 * the neighbour quotient q[i-1] is read straight from global memory, so block boundaries
 * need no special handling at all.)
 *
 * The exact ripple in closed form: with low_i = sum_{j<i} c_j 2^(bpw*j), the carry into digit
 * i is floor(low_i / 2^(bpw*i)), and d_i = c_i + carry_in_i < 2^bpw -- this is what the
 * rounds converge to, and why digits never exceed the mask at a fixed point.
 */

/* q[i] = C[i] >> bpw ; C[i] &= mask   (each thread touches only its own word) */
__global__ void carry_extract_kernel(unsigned long long *c, unsigned long long *q,
                                     unsigned long long n, int bpw)
{
    const unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i >= n) return;
    const unsigned long long v = c[i];
    q[i] = v >> bpw;
    c[i] = v & ((1ull << bpw) - 1ull);
}

/* C[i] += q[i-1]: the ripple move, one digit per round (see above). */
__global__ void carry_add_kernel(unsigned long long *c, const unsigned long long *q,
                                 unsigned long long n)
{
    const unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i == 0 || i >= n) return;
    c[i] += q[i - 1];
}

/* diagnostics: max digit (in bits) and how many digits are still >= 2^bpw after the last
   carry round.  A non-zero count means the round count was too small -- a carry bug is
   silent (it just stops producing factors), so this is asserted rather than assumed.
   SLICE-BATCHED (S4): `stride` is the distance in words between two slices and blockIdx.y is
   the slice index (gridDim.y == 1 on the per-call path, where stride == n, so the single-slice
   behaviour is unchanged); each slice gets its OWN two counters at out[2*blockIdx.y]. */
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

/*
 * One PARALLEL thread per coefficient: fold the bpw-bit words covering this coefficient's
 * bit range back into an integer (Horner, most significant word first) reduced modulo the
 * 32-bit prime 4294967291.  A real coefficient is 2S + log2(P) bits (10323 for S = 5153),
 * so it does not fit in a uint64 -- the probe therefore verifies through this modular
 * projection, exactly like the cuFFT probe.  The slot is wide enough that no carry crosses
 * it, so a plain per-slot carry is sufficient here (the value is < 2^bpw after the global
 * pass).
 *
 * The coefficient's bits start at bit  o*slot_bits  (the SAME convention the packer uses),
 * which is NOT word-aligned: words_per_slot = ceil(slot_bits/bpw) whole words are read and
 * only the low `slot_bits` bits of that window belong to this coefficient.  Packing and
 * assembling must agree on this stride -- a mismatch here (packing at i*slot_bits bits but
 * assembling whole words per slot) was a real bug that made every coefficient wrong.
 *
 * `slot_stride` is the packing stride in BITS: coefficient i starts at bit i*slot_stride of
 * the packed product.  It MUST be a multiple of bpw (word-aligned packing), otherwise a
 * single digit of the packed operand holds bits of two different bpw-bit chunks of the
 * coefficient (value up to 2^(2*bpw-1) instead of 2^bpw), the exactness criterion
 * L*(2^bpw-1)^2 < p no longer bounds the convolution coefficients (its derivation needs
 * every digit < 2^bpw), and EVERY coefficient of
 * the product wraps mod p -- silently, because the geometric assertion still passes.  That
 * was the second half of the bug (see the packing comment in run_poly).
 */
#define SLOT_MOD 4294967291ull

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

} /* namespace */

/*
 * The two transforms, used by every mode, so there is exactly one implementation.
 *
 * FORWARD = bitrev -> DIF stages (half = N/2 ... 1, exp_step = N/2half) with omega
 *           -> bitrev.
 * INVERSE = bitrev -> DIF stages with omega^{-1} -> bitrev -> multiply by 1/N.
 *
 * i.e. both directions are the same DIF network with the two possible roots, wrapped in the
 * same pair of bit-reversals; the inverse reaches the forward's input because
 * DIT(omega) . DIT(omega^{-1}) == N on the bit-reversed ordering that a DIF network produces.
 *
 * The bit-reversal is IN PLACE BY SWAPS: written the other way (`dst[i] = src[bitrev(i)]`
 * with one buffer) it is a read-after-write race, which is what made an earlier version
 * disagree with a hand-written mirror while agreeing with itself.
 */
static void ntt_bitrev_inplace(unsigned long long *d, unsigned long long n, int k)
{
    const unsigned int threads = 256;
    const unsigned int blocks = (unsigned int)((n + threads - 1) / threads);
    bitrev_swap_kernel<<<blocks, threads>>>(d, (unsigned long long)k, n);
    CK(cudaGetLastError());
}

static void ntt_dif(unsigned long long *d, unsigned long long n, int k,
                    unsigned long long omega)
{
    const unsigned int threads = 256;
    for (int s = k - 1; s >= 0; --s) {          /* DIF: half = N/2 ... 1 */
        const unsigned long long half = 1ull << s;
        const unsigned long long exp_step = n / (2 * half);
        const unsigned int bl = (unsigned int)((n / 2 + threads - 1) / threads);
        ntt_stage_kernel<<<bl, threads>>>(d, half, exp_step, omega, n);
        CK(cudaGetLastError());
    }
}

/* inverse bit-reversal: ONLY the permutation, no copy (writes b[i] = a[bitrev(i)]).  Safe
   here because it is applied when the source is a distinct buffer (see ntt_inverse). */
static void ntt_bitrev_perm(unsigned long long *d, unsigned long long *scratch,
                            unsigned long long n, int k)
{
    const unsigned int threads = 256;
    const unsigned int blocks = (unsigned int)((n + threads - 1) / threads);
    bitrev_kernel<<<blocks, threads>>>(d, scratch, (unsigned long long)k, n);
    CK(cudaGetLastError());
    CK(cudaMemcpy(d, scratch, n * sizeof(unsigned long long), cudaMemcpyDeviceToDevice));
}

static void ntt_forward(unsigned long long *d, unsigned long long n, int k,
                        unsigned long long omega, unsigned long long *scratch)
{
    (void)scratch;
    ntt_dif(d, n, k, omega);   /* halves DESCENDING: natural in, bit-reversed spectrum out */
}

/* d[i] *= n_scale (the 1/N of the inverse) */
__global__ void scale_kernel(unsigned long long *d, unsigned long long n,
                             unsigned long long n_scale)
{
    const unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n) d[i] = gl_mul(d[i], n_scale);
}

static void ntt_inverse(unsigned long long *d, unsigned long long n, int k,
                        unsigned long long omega_inv, unsigned long long n_scale,
                        unsigned long long *scratch)
{
    (void)scratch;
    const unsigned int threads = 256;
    const unsigned int blocks = (unsigned int)((n + threads - 1) / threads);
    /* DIT, halves ASCENDING, same index decomposition: consumes the bit-reversed frequency
       vector the forward produced, and gives the natural-order result. */
    for (int s = 0; s < k; ++s) {
        const unsigned long long half = 1ull << s;
        const unsigned long long exp_step = n / (2 * half);
        const unsigned int bl = (unsigned int)((n / 2 + threads - 1) / threads);
        dit_stage_kernel<<<bl, threads>>>(d, half, exp_step, omega_inv, n);
        CK(cudaGetLastError());
    }
    scale_kernel<<<blocks, threads>>>(d, n, n_scale);
    CK(cudaGetLastError());
}

/* =====================================================================================
 * STAGE FUSION (docs/DEV_STAGE2_GPU_PLAN.md section 12) -- the M2 optimisation round.
 *
 * WHAT COUNTS AS A PASS HERE: one FULL READ *and* one FULL WRITE of the N-word array
 * (2*N*8 bytes of traffic).  The correctness-first first cut launched one kernel per
 * stage, i.e. 1 pass per stage, and the pipeline needs two forward transforms plus one
 * inverse, so P=8192 (k=24) paid 3*24 + pointwise + scale + 12 carry passes + assembly
 * ~= 88 passes.  That is the number this section removes.
 *
 * Two pass kinds, both keeping the SAME DIF/DIT network, the same twiddles and the same
 * Goldilocks arithmetic -- only where the data lives between butterflies changes:
 *
 *  (1) OUTER PASS: a fused radix-R = 2^M butterfly held entirely in REGISTERS, covering the
 *      M stages whose pairs are far apart.  After the stages k-1..k-L have been done, the
 *      array is 2^L independent blocks of 2^(k-L) words, and the next M stages act on
 *      groups of R elements at stride S = 2^(k-L-M) inside each block (one element per
 *      sub-block, same offset).  Those R elements are loaded coalesced (consecutive
 *      threads take consecutive offsets), the M stages run in registers, and the result is
 *      written back: M stages for ONE pass.
 *      The twiddle of stage i (i = 0 is the widest) at offset u inside the sub-block is
 *        omega^((v + u*S) * 2^(L+i))  =  TB_i(v) * D[i][u],
 *      with v the group offset, TB_i = tbl[v]^(2^i) (one table read per group plus i
 *      squarings) and D[i][u] = baseW^(u*2^i), baseW = omega^(S*2^L) -- D depends only on
 *      the pass, so it is built once per pass and sits in shared memory (all threads read
 *      the same entry at the same time -> broadcast).
 *      (The old code recomputed every twiddle with square-and-multiply PER ELEMENT PER
 *      STAGE: up to k/2 modular squarings, i.e. more arithmetic than the butterflies.)
 *
 *  (2) TILE PASS: the remaining t stages have pair spacing < 2^t, so their pairs never
 *      leave one aligned block of 2^t words.  The block is loaded into shared memory once,
 *      all t stages run there, and it is written back: t stages for ONE pass.  The forward
 *      does the DIF stages t-1..0; the inverse does the DIT stages 0..t-1 and ALSO absorbs
 *      the pointwise product and the 1/N scale (both elementwise), which removes two more
 *      whole passes from the pipeline.
 *
 * Twiddle table for the tile pass: 2^t entries, laid out PER STAGE (stage s occupies
 * [2^s-1, 2^(s+1)-1), entry j = root^(j*2^(k-1-s))).  The per-stage layout is deliberate:
 * within a stage consecutive threads walk consecutive j, so the shared-memory read is
 * conflict-free; the natural "e" layout would make the middle stages 32-way bank conflicts.
 *
 * The plan (t and the list of M per outer pass) is chosen at run time by a tiny DP that
 * minimises the total pass count; NTT_FUSE_T / NTT_FUSE_M override it for tuning.
 * ===================================================================================== */
#define FUSE_TILE_THREADS 512
#define FUSE_OUTER_THREADS 256
/* radix ceiling for the outer passes.  Measured on the acceptance shapes (P=8192/S=5153,
   13 passes, device 1): M=4 -> 0.174 ns/operand-bit, M=5 -> 0.181, M=6 -> 0.224 (the radix-64
   kernel needs 255 registers and spills 384 bytes/thread, which costs more than the two
   passes it saves).  Re-measured after the L-bound change, WITH the fuse_plan_stages
   tie-break below so that M=5 is used only where it removes a whole pass (rem >= 13, i.e.
   N >= 2^25): P=65536/S=5153 16 passes at M=4 -> 0.183, 13 passes at M=5 -> 0.196;
   P=92160/S=5261 0.128 -> 0.137; P=8192/S=5153 0.085 -> 0.099.  So one radix-32 pass costs
   ~1.43x a radix-16 pass (32 live words + spills), and saving one pass out of four does not
   pay for it.  M=5 is therefore NOT enabled: the pass count is not the whole cost model.
   NTT_FUSE_M (<= this ceiling) overrides the planner for experiments. */
#define FUSE_MAX_M 4
#define FUSE_MAX_CARRY_ROUNDS 10
/* fuse_plan_stages stores at most 8 peels in FuseCtx::ms; the cached-table arrays (section
   18, slice S3) are indexed by pass number and sized the same. */
#define FUSE_MAX_PASSES 8

/* ---- one-pass exact carry ------------------------------------------------------------
 * The carry is the ripple from the carry-stage comment above:
 *     c_i  <-  (c_i mod 2^bpw) + floor(c_{i-1} / 2^bpw)
 * Round r+1 needs round r at i-1, so round R at i depends on the RAW digits at i, i-1,
 * ..., i-R only.  Unfolding the cone into registers therefore reproduces the round-by-round
 * ripple EXACTLY (same recurrence, same integer arithmetic, no reordering of the values),
 * while touching the array once instead of once per round.  The digit heights stay far
 * inside 64 bits by the same argument as before (each level adds at most 2^bpw).
 * This cone reduces the height to at most the radix.  The binary carry below then
 * canonicalises long all-mask runs; the convergence assert in run_poly is unchanged.
 *
 * RACE WARNING (this bit me): the cone READS the raw digits at i-1..i-R while it WRITES digit
 * i, so it must NOT write the array it reads -- a neighbouring thread may already have
 * overwritten the raw value this thread still needs (measured: 2 wrong slots at P=512 and 8
 * at P=1024, non-deterministic, while `fusecheck` -- which only exercises the transform --
 * stayed clean).  Hence the separate output buffer `cout`: reads come from c (never modified
 * while the kernel runs), writes go to cout, and every downstream consumer (slot assembly,
 * convergence assert, pipedump) reads cout.
 */
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

/* ---- twiddle tables ---------------------------------------------------------------- */

/* tile table: stage s at [2^s - 1, 2^(s+1) - 1), entry j = root^(j * 2^(k-1-s)) */
__global__ void build_tile_table_kernel(unsigned long long *tbl, int k, int t,
                                        unsigned long long root)
{
    const unsigned long long idx = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (idx >= (1ull << t)) return;
    int s = 0;
    while ((1ull << (s + 1)) <= idx + 1ull) ++s;
    if (s >= k) { tbl[idx] = 1ull; return; }
    const unsigned long long j = idx + 1ull - (1ull << s);
    tbl[idx] = gl_twiddle(root, j << (k - 1 - s));
}

/* per-group coarse twiddle of an outer pass: tbl[v] = root^(v * 2^shift) for v < sz */
__global__ void build_pass_table_kernel(unsigned long long *tbl, unsigned long long sz,
                                        int shift, unsigned long long root)
{
    const unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i >= sz) return;
    tbl[i] = gl_twiddle(root, i << shift);
}

/* per-pass radix table: D[i][u] = root^(u * 2^(rev ? m-1-i : i)), compacted so that stage i
   occupies [R - (R>>i), R - (R>>(i+1))) in the forward layout (off = R - (R>>i), size
   2^(m-1-i)) and [2^i - 1, 2^(i+1) - 1) in the inverse layout (size 2^i).  Total R-1. */
__global__ void build_radix_table_kernel(unsigned long long *tbl, int m, unsigned long long root,
                                         int rev)
{
    const int R = 1 << m;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= R - 1) return;
    int i = 0;
    int off = 0;
    for (;;) {
        const int d = rev ? (1 << i) : (1 << (m - 1 - i));
        if (idx < off + d) break;
        off += d;
        ++i;
    }
    const int u = idx - off;
    const int sh = rev ? (m - 1 - i) : i;
    tbl[idx] = gl_twiddle(root, (unsigned long long)u << sh);
}

/* ---- (1) outer fused radix-R pass, forward (DIF: stages k-1-L down to k-L-M) --------- */
template <int M>
__global__ void outer_fwd_kernel(unsigned long long *a, unsigned long long n, int L,
                                 const unsigned long long *tbl, const unsigned long long *D,
                                 unsigned long long grp_per_blk, unsigned long long stride)
{
    a += (size_t)blockIdx.y * stride;
    constexpr int R = 1 << M;
    const unsigned long long S = n >> (L + M);      /* group offset count == stride */
    const unsigned long long MB = n >> L;           /* block size */
    const unsigned long long groups = n >> M;
    __shared__ unsigned long long sD[R - 1];
    for (int i = threadIdx.x; i < R - 1; i += blockDim.x) sD[i] = D[i];
    __syncthreads();
    const unsigned long long g0 = blockIdx.x * grp_per_blk;
    const unsigned long long g1 = (groups < g0 + grp_per_blk) ? groups : (g0 + grp_per_blk);
    for (unsigned long long g = g0 + threadIdx.x; g < g1; g += blockDim.x) {
        const unsigned long long blk = g / S;
        const unsigned long long v = g - blk * S;
        unsigned long long *p0 = a + blk * MB + v;
        unsigned long long x[R];
#pragma unroll
        for (int i = 0; i < R; ++i) x[i] = p0[(unsigned long long)i * S];
        unsigned long long tb = tbl[v];             /* omega^(v*2^L), squared per stage */
#pragma unroll
        for (int i = 0; i < M; ++i) {
            const int d = 1 << (M - 1 - i);
#pragma unroll
            for (int u = 0; u < d; ++u) {
                const unsigned long long w = gl_mul(tb, sD[(R - (R >> i)) + u]);
#pragma unroll
                for (int b = 0; b < R / (2 * d); ++b) {
                    const int i1 = b * 2 * d + u, i2 = i1 + d;
                    const unsigned long long uu = x[i1], vv = x[i2];
                    x[i1] = gl_add_dev(uu, vv);
                    x[i2] = gl_mul(gl_sub_dev(uu, vv), w);
                }
            }
            tb = gl_mul(tb, tb);
        }
#pragma unroll
        for (int i = 0; i < R; ++i) p0[(unsigned long long)i * S] = x[i];
    }
}

/* ---- (1') outer fused radix-R pass, inverse (DIT: stages s0 .. s0+M-1 ascending) ------ */
template <int M>
__global__ void outer_inv_kernel(unsigned long long *a, unsigned long long n, int s0,
                                 const unsigned long long *tq, const unsigned long long *dw,
                                 unsigned long long grp_per_blk, unsigned long long stride)
{
    a += (size_t)blockIdx.y * stride;
    constexpr int R = 1 << M;
    const unsigned long long S = 1ull << s0;
    const unsigned long long MB = S << M;
    const unsigned long long groups = n >> M;
    __shared__ unsigned long long sDW[R - 1];
    for (int i = threadIdx.x; i < R - 1; i += blockDim.x) sDW[i] = dw[i];
    __syncthreads();
    const unsigned long long g0 = blockIdx.x * grp_per_blk;
    const unsigned long long g1 = (groups < g0 + grp_per_blk) ? groups : (g0 + grp_per_blk);
    for (unsigned long long g = g0 + threadIdx.x; g < g1; g += blockDim.x) {
        const unsigned long long blk = g >> s0;
        const unsigned long long base = g - (blk << s0);
        unsigned long long *p0 = a + blk * MB + base;
        unsigned long long x[R];
#pragma unroll
        for (int i = 0; i < R; ++i) x[i] = p0[(unsigned long long)i << s0];
        /* q = omega_inv^(base*2^L); stage i needs q^(2^(m-1-i)) -> squaring chain downwards */
        unsigned long long tbv[M];
        tbv[M - 1] = tq[base];
#pragma unroll
        for (int i = M - 2; i >= 0; --i) tbv[i] = gl_mul(tbv[i + 1], tbv[i + 1]);
#pragma unroll
        for (int i = 0; i < M; ++i) {
            const int d = 1 << i;
#pragma unroll
            for (int u = 0; u < d; ++u) {
                const unsigned long long w = gl_mul(tbv[i], sDW[(d - 1) + u]);
#pragma unroll
                for (int b = 0; b < R / (2 * d); ++b) {
                    const int i1 = b * 2 * d + u, i2 = i1 + d;
                    const unsigned long long uu = x[i1];
                    const unsigned long long vv = gl_mul(x[i2], w);
                    x[i1] = gl_add_dev(uu, vv);
                    x[i2] = gl_sub_dev(uu, vv);
                }
            }
        }
#pragma unroll
        for (int i = 0; i < R; ++i) p0[(unsigned long long)i << s0] = x[i];
    }
}

/* ONE fused RADIX-4 BLOCK of two consecutive tile stages, held in registers.
 *
 * Why: the tile pass is LATENCY bound, not ALU bound -- replacing its one twiddle multiply
 * per butterfly with an add (experiment on a scratch copy) cut the forward tile pass from
 * 3.7 ms to 1.5 ms while the arithmetic per element stayed the same, i.e. the cost sits in
 * the per-stage dependency chain: shared load -> modular multiply -> shared store ->
 * __syncthreads(), twelve times over.  Fusing two stages into one block halves both the
 * shared traffic (4 reads + 4 writes per 4 elements for TWO stages instead of 8 + 8) and
 * the barrier count (6 instead of 12) at the same multiply count.
 *
 * The two stages are (st, st-1) for the DIF and (st, st+1) for the DIT; h2 = 2^(st-1)
 * (resp. 2^st) is the smaller half, so a block of 4*h2 words holds one unit.  Positions:
 * base+u, base+h2+u, base+2*h2+u, base+3*h2+u, and the twiddle of every pair is
 * tw_s[i0 mod 2^s] -- which for these four is exactly tw_{st}[u], tw_{st}[h2+u],
 * tw_{st-1}[u], tw_{st-1}[u] (DIF; see the derivation in the forward branch below).
 * `tbl` is the per-stage table (stage s at offset 2^s-1) built by build_tile_table_kernel.
 */
__device__ __forceinline__ void tile_radix4_block(unsigned long long *sm, unsigned long long TILE,
                                                 int st, const unsigned long long *tbl)
{
    const unsigned long long half = 1ull << st;      /* the LARGER half of the pair */
    const unsigned long long h2 = half >> 1;         /* the smaller one */
    const unsigned long long *twA = tbl + (half - 1);        /* larger-half stage */
    const unsigned long long *twB = tbl + (h2 - 1);          /* smaller-half stage */
    const unsigned long long units = TILE >> 2;
    for (unsigned long long m = threadIdx.x; m < units; m += blockDim.x) {
        const unsigned long long base = (m >> (st - 1)) << (st + 1);
        const unsigned long long u = m & (h2 - 1);
        const unsigned long long a = sm[base + u];
        const unsigned long long b = sm[base + u + half];
        const unsigned long long c = sm[base + h2 + u];
        const unsigned long long d = sm[base + h2 + u + half];
        /* stage st (half): (base+u, base+u+half) with j = u; (base+h2+u, base+h2+u+half)
           with j = (base+h2+u) mod half = h2+u  -- DIF multiplies AFTER the subtraction */
        const unsigned long long a1 = gl_add_dev(a, b);
        const unsigned long long b1 = gl_mul(gl_sub_dev(a, b), twA[u]);
        const unsigned long long c1 = gl_add_dev(c, d);
        const unsigned long long d1 = gl_mul(gl_sub_dev(c, d), twA[h2 + u]);
        /* stage st-1 (h2): (base+u, base+h2+u) with j = u  and  (base+2*h2+u, base+3*h2+u)
           with j = u  -- BOTH pairs of the smaller stage share the same twiddle */
        const unsigned long long tw = twB[u];
        sm[base + u] = gl_add_dev(a1, c1);
        sm[base + h2 + u] = gl_mul(gl_sub_dev(a1, c1), tw);
        sm[base + 2 * h2 + u] = gl_add_dev(b1, d1);
        sm[base + 3 * h2 + u] = gl_mul(gl_sub_dev(b1, d1), tw);
    }
}

/* the same fusion for the INVERSE tile pass, where the pair is (st, st+1) and the DIT
   multiplies BEFORE the add/subtract (half = 2^(st+1) is the larger half here) */
__device__ __forceinline__ void tile_radix4_block_inv(unsigned long long *sm,
                                                      unsigned long long TILE, int st,
                                                      const unsigned long long *tbl)
{
    const unsigned long long h2 = 1ull << st;        /* the smaller half (stage st) */
    const unsigned long long half = h2 << 1;         /* the larger half (stage st+1) */
    const unsigned long long *twA = tbl + (h2 - 1);          /* stage st */
    const unsigned long long *twB = tbl + (half - 1);        /* stage st+1 */
    const unsigned long long units = TILE >> 2;
    for (unsigned long long m = threadIdx.x; m < units; m += blockDim.x) {
        const unsigned long long base = (m >> st) << (st + 2);
        const unsigned long long u = m & (h2 - 1);
        const unsigned long long a = sm[base + u];
        const unsigned long long b = sm[base + 2 * h2 + u];
        const unsigned long long c = sm[base + h2 + u];
        const unsigned long long d = sm[base + 3 * h2 + u];
        /* stage st: pairs (base+u, base+h2+u) and (base+2h2+u, base+3h2+u), both j = u */
        const unsigned long long w1 = twA[u];
        const unsigned long long cv = gl_mul(c, w1);
        const unsigned long long a1 = gl_add_dev(a, cv);
        const unsigned long long c1 = gl_sub_dev(a, cv);
        const unsigned long long dv = gl_mul(d, w1);
        const unsigned long long b1 = gl_add_dev(b, dv);
        const unsigned long long d1 = gl_sub_dev(b, dv);
        /* stage st+1: (base+u, base+2h2+u) with j = u, (base+h2+u, base+3h2+u) with j = h2+u */
        const unsigned long long bv = gl_mul(b1, twB[u]);
        const unsigned long long dv2 = gl_mul(d1, twB[h2 + u]);
        sm[base + u] = gl_add_dev(a1, bv);
        sm[base + 2 * h2 + u] = gl_sub_dev(a1, bv);
        sm[base + h2 + u] = gl_add_dev(c1, dv2);
        sm[base + 3 * h2 + u] = gl_sub_dev(c1, dv2);
    }
}

/* one plain radix-2 tile stage (used for a leftover odd stage) */
template <bool INVERSE>
__device__ __forceinline__ void tile_radix2_stage(unsigned long long *sm, unsigned long long TILE,
                                                  int st, const unsigned long long *tbl)
{
    const unsigned long long half = 1ull << st;
    const unsigned long long *tw = tbl + (half - 1);
    for (unsigned long long m = threadIdx.x; m < TILE / 2; m += blockDim.x) {
        const unsigned long long g = m >> st;
        const unsigned long long j = m - (g << st);
        const unsigned long long i0 = (g << (st + 1)) + j;
        const unsigned long long i1 = i0 + half;
        const unsigned long long uu = sm[i0], vv = sm[i1];
        const unsigned long long w0 = __ldg(tw + j);
        if (INVERSE) {
            const unsigned long long w = gl_mul(vv, w0);
            sm[i0] = gl_add_dev(uu, w);
            sm[i1] = gl_sub_dev(uu, w);
        } else {
            sm[i0] = gl_add_dev(uu, vv);
            sm[i1] = gl_mul(gl_sub_dev(uu, vv), w0);
        }
    }
    __syncthreads();
}

/* ---- (2) tile pass: t stages inside shared memory, one read + one write ---------------
 * OCCUPANCY NOTE: only the TILE lives in shared memory (2^t words = 32 KB at t=12), the
 * twiddle table is read straight from global memory.  Keeping the table in shared too would
 * need 64 KB per block, which on this part allows ONE block (512 threads) per SM: the twelve
 * __syncthreads() barriers then have nothing to hide their latency behind and the pass ran
 * 3.4x slower than the 1.34 ms/pass memory floor (measured, NTT_FUSE_TRACE=1).  With 32 KB
 * the SM holds 3 blocks = 1536 threads = 48 warps, and the table is L2-resident (32 KB, read
 * by every block).
 * The stages themselves run as fused radix-4 blocks (see tile_radix4_block): six barriers
 * instead of twelve. */
template <bool INVERSE>
__global__ void tile_kernel(unsigned long long *a, const unsigned long long *b,
                            unsigned long long n, int k, int t,
                            const unsigned long long *tbl, unsigned long long n_scale,
                            unsigned long long stride)
{
    const size_t sl = (size_t)blockIdx.y * stride;
    a += sl; if (b) b += sl;
    extern __shared__ unsigned long long sm[];
    const unsigned long long TILE = 1ull << t;
    const unsigned long long base = blockIdx.x * TILE;
    for (unsigned long long i = threadIdx.x; i < TILE; i += blockDim.x)
        sm[i] = a[base + i];
    __syncthreads();
    if (INVERSE) {
        /* pointwise product + 1/N, both elementwise: fusing them here saves two passes */
        for (unsigned long long i = threadIdx.x; i < TILE; i += blockDim.x)
            sm[i] = gl_mul(gl_mul(sm[i], b[base + i]), n_scale);
        __syncthreads();
    }
    int q = 0;
    if (!INVERSE) {
        /* DIF runs half = t-1 down to 0: pair (t-1,t-2), (t-3,t-4), ... */
        for (; q + 1 < t; q += 2) {
            tile_radix4_block(sm, TILE, t - 1 - q, tbl);
            __syncthreads();
        }
        if (q < t) tile_radix2_stage<false>(sm, TILE, 0, tbl);
    } else {
        /* DIT runs half = 0 up to t-1: pair (0,1), (2,3), ... */
        for (; q + 1 < t; q += 2) {
            tile_radix4_block_inv(sm, TILE, q, tbl);
            __syncthreads();
        }
        if (q < t) tile_radix2_stage<true>(sm, TILE, q, tbl);
    }
    for (unsigned long long i = threadIdx.x; i < TILE; i += blockDim.x)
        a[base + i] = sm[i];
}

/* ---- fusion plan: which t, and which M per outer pass -------------------------------- */
struct FuseCtx {
    unsigned long long n = 0;
    int k = 0, t = 0;
    int ms[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    int nms = 0;
    int outer_stages = 0;                                /* stages covered by outer passes */
    unsigned long long *tblF = nullptr, *tblI = nullptr; /* 2^t words each */
    unsigned long long *scr = nullptr;                   /* per-pass coarse tables */
    unsigned long long *scr2 = nullptr;                  /* per-pass radix tables */
    size_t scrWords = 0;
    size_t scr2Words = 64;
    int m_max = FUSE_MAX_M;
    bool ready = false;
    bool arena_borrowed = false;                         /* this copy must not release arena storage */
    int passes_fwd = 0;
    /* ---- OPTIONAL cache of the per-pass twiddle tables (section 18, slice S3) ------------
       build_pass_table_kernel / build_radix_table_kernel are pure functions of (shape, pass),
       yet the per-call path rebuilds them with two extra launches on EVERY multiply.  A
       caller that multiplies the same shape thousands of times (the batched stage-2 tree)
       fills these once with ntt_fuse_cache_tables() and every later multiply reuses them.
       When tables_cached is false nothing here is touched and the path is bit-identical to
       the original one. */
    bool tables_cached = false;
    unsigned long long *passF[FUSE_MAX_PASSES] = {nullptr};
    unsigned long long *radF[FUSE_MAX_PASSES] = {nullptr};
    unsigned long long *passI[FUSE_MAX_PASSES] = {nullptr};
    unsigned long long *radI[FUSE_MAX_PASSES] = {nullptr};
};

/* Requested payload of the mandatory per-FuseCtx allocations, including temporary plans.
   Single host-thread accounting, like the rest of this probe's arena and CUDA context state. */
struct FuseBaseStats {
    unsigned long long allocations=0, frees=0, live_bytes=0, peak_bytes=0;
};
static FuseBaseStats g_fuse_base;
static void fuse_base_allocate(unsigned long long **p, size_t words)
{
    if (!words) { *p=nullptr; return; }
    CK(cudaMalloc((void **)p,words*8));
    ++g_fuse_base.allocations; g_fuse_base.live_bytes+=words*8;
    g_fuse_base.peak_bytes=std::max(g_fuse_base.peak_bytes,g_fuse_base.live_bytes);
}
static void fuse_base_free(unsigned long long *&p, size_t words)
{
    if (!p) return;
    CK(cudaFree(p)); p=nullptr;
    ++g_fuse_base.frees; g_fuse_base.live_bytes-=words*8;
}

/* minimise the number of outer passes for `rem` stages with m <= m_max.  PRIMARY key: the
   pass count (every pass moves the whole array).  SECONDARY key: the WIDEST FINAL PEEL, so
   that a plan is never made to end in a leftover radix-2 pass when an equally short plan
   exists that spreads the width better.  (The total butterfly work is rem*n/2 for EVERY plan,
   so the pass count plus per-pass efficiency is the whole cost model; this tie-break was
   added while testing m_max=5 and is kept because it costs nothing and removes that class of
   degenerate plan.  Raising m_max to 5 was measured and rejected -- see FUSE_MAX_M.) */
static void fuse_plan_stages(FuseCtx &c, int rem)
{
    static int best[64], bestm[64], bestlast[64];
    for (int r = 0; r <= rem; ++r) { best[r] = 1 << 20; bestm[r] = 0; bestlast[r] = 0; }
    best[0] = 0;
    for (int r = 1; r <= rem; ++r) {
        const int mx = (r < c.m_max) ? r : c.m_max;
        for (int m = mx; m >= 1; --m) {
            const int cand = best[r - m] + 1;
            const int last = (r - m == 0) ? m : bestlast[r - m];
            if (cand < best[r] || (cand == best[r] && last > bestlast[r])) {
                best[r] = cand;
                bestm[r] = m;
                bestlast[r] = last;
            }
        }
    }
    int tmp[8], n = 0, r = rem;
    while (r > 0 && n < 8) { tmp[n++] = bestm[r]; r -= bestm[r]; }
    /* the peel is already widest-first, which is the order the stages must be done in */
    for (int i = 0; i < n; ++i) c.ms[i] = tmp[i];
    int sum = 0;
    for (int i = 0; i < n; ++i) sum += c.ms[i];
    if (sum != rem) {                       /* cannot happen; loud if the DP is ever changed */
        std::fprintf(stderr, NTT_PROBE_NAME ": fuse plan does not cover %d stages (sum=%d)\n",
                     rem, sum);
        std::exit(3);
    }
    c.nms = n;
    c.outer_stages = rem;
}

static unsigned long long fuse_env_ull(const char *name, unsigned long long dflt)
{
    const char *s = std::getenv(name);
    if (!s || !*s) return dflt;
    return std::strtoull(s, nullptr, 10);
}

static bool fuse_compact_scratch()
{
    const char *e=std::getenv("NTT_FUSE_COMPACT_SCRATCH");
    return !e || !*e || std::atoi(e)!=0;
}
static void fuse_init(FuseCtx &c, unsigned long long n, int k, unsigned long long omega,
                      unsigned long long omega_inv, bool compact=fuse_compact_scratch())
{
    c.n = n;
    c.k = k;
    int t = (int)fuse_env_ull("NTT_FUSE_T", 12);
    if (t > k) t = k;
    if (t < 0) t = 0;
    c.t = t;
    c.m_max = (int)fuse_env_ull("NTT_FUSE_M", FUSE_MAX_M);
    if (c.m_max > FUSE_MAX_M) c.m_max = FUSE_MAX_M;
    if (c.m_max < 1) c.m_max = 1;
    fuse_plan_stages(c, k - t);

    fuse_base_allocate(&c.tblF,(size_t)(1ull << t));
    fuse_base_allocate(&c.tblI,(size_t)(1ull << t));
    if (t > 0) {
        const unsigned int tb = (unsigned int)(((1ull << t) + 255) / 256);
        build_tile_table_kernel<<<tb, 256>>>(c.tblF, k, t, omega);
        CK(cudaGetLastError());
        build_tile_table_kernel<<<tb, 256>>>(c.tblI, k, t, omega_inv);
        CK(cudaGetLastError());
    }
    /* Size from BOTH actual pass sequences, not an assumed radix. The old N/4 guess broke
       radix-2; a tile-only plan has no coarse/radix scratch readers at all. */
    c.scrWords = compact ? 0 : (size_t)(n >> 1) + 64;
    c.scr2Words = compact ? 0 : 64;
    if (compact) {
        int L=0;
        for (int p=0;p<c.nms;++p) {
            const int M=c.ms[p];
            c.scrWords=std::max(c.scrWords,(size_t)(n>>(L+M)));
            c.scr2Words=std::max(c.scr2Words,(size_t)(1ull<<M)); L+=M;
        }
        L=c.outer_stages;
        for (int p=c.nms-1;p>=0;--p) {
            const int M=c.ms[p]; L-=M;
            c.scrWords=std::max(c.scrWords,(size_t)(1ull<<(c.k-L-M)));
        }
    }
    fuse_base_allocate(&c.scr,c.scrWords);
    fuse_base_allocate(&c.scr2,c.scr2Words);
    /* 2^t words of dynamic shared memory for the tile pass (the twiddle table is in global).
       The carveout hint matters: without it the driver may leave the L1/shared split too
       small for more than one block per SM, and the tile pass is latency bound (its modular
       multiply chain), so the extra resident blocks are what hides that latency. */
    const int smem = (int)(1ull << t) * (int)sizeof(unsigned long long);
    cudaFuncSetAttribute(tile_kernel<false>, cudaFuncAttributePreferredSharedMemoryCarveout,
                         100);
    cudaFuncSetAttribute(tile_kernel<true>, cudaFuncAttributePreferredSharedMemoryCarveout,
                         100);
    CK(cudaFuncSetAttribute(tile_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                            smem));
    CK(cudaFuncSetAttribute(tile_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                            smem));
    c.passes_fwd = c.nms + (t > 0 ? 1 : 0);
    c.ready = true;
}

static void fuse_release(FuseCtx &c)
{
    for (int p = 0; p < FUSE_MAX_PASSES; ++p) {
        if (c.passF[p]) cudaFree(c.passF[p]);
        if (c.radF[p]) cudaFree(c.radF[p]);
        if (c.passI[p]) cudaFree(c.passI[p]);
        if (c.radI[p]) cudaFree(c.radI[p]);
        c.passF[p] = c.radF[p] = c.passI[p] = c.radI[p] = nullptr;
    }
    c.tables_cached = false;
    fuse_base_free(c.tblF,(size_t)(1ull << c.t));
    fuse_base_free(c.tblI,(size_t)(1ull << c.t));
    fuse_base_free(c.scr,c.scrWords);
    fuse_base_free(c.scr2,c.scr2Words);
    c.tblF = c.tblI = c.scr = c.scr2 = nullptr;
    c.ready = false;
    c.arena_borrowed = false;
}

/* A non-null arena can refuse a plan. Cleanup follows the actual owner, on EVERY return. */
struct FuseCallGuard {
    FuseCtx &fc;
    explicit FuseCallGuard(FuseCtx &context):fc(context) {}
    ~FuseCallGuard() { if (!fc.arena_borrowed) fuse_release(fc); }
    FuseCallGuard(const FuseCallGuard &)=delete;
    FuseCallGuard &operator=(const FuseCallGuard &)=delete;
};
static size_t fuse_base_words(const FuseCtx &c)
{
    return (c.tblF ? (size_t)(1ull<<c.t) : 0) + (c.tblI ? (size_t)(1ull<<c.t) : 0)
         + (c.scr ? c.scrWords : 0) + (c.scr2 ? c.scr2Words : 0);
}

/* Build EVERY per-pass twiddle table of this shape once, into its own buffer, and mark the
   FuseCtx as cached.  The kernels and the index arithmetic are the very same ones the
   per-call path runs -- only the timing changes (once per shape instead of once per pass of
   every multiply), so a cached transform is bit-identical to an uncached one. */
static void ntt_fuse_cache_tables(FuseCtx &c, unsigned long long omega,
                                  unsigned long long omega_inv)
{
    if (c.tables_cached) return;
    {   /* forward: outer passes over stages k-1..t, ascending L; same for() as
           ntt_forward_fused so the (S, L, baseW) of pass p are identical. */
        int L = 0;
        for (int p = 0; p < c.nms; ++p) {
            const int M = c.ms[p];
            const unsigned long long S = c.n >> (L + M);
            CK(cudaMalloc(&c.passF[p], (size_t)S * sizeof(unsigned long long)));
            CK(cudaMalloc(&c.radF[p], (size_t)(1 << M) * sizeof(unsigned long long)));
            const unsigned int tb = (unsigned int)((S + 255) / 256);
            build_pass_table_kernel<<<tb, 256>>>(c.passF[p], S, L, omega);
            CK(cudaGetLastError());
            const unsigned long long baseW = gl_pow_host(omega, S << L);
            build_radix_table_kernel<<<1, (1 << M)>>>(c.radF[p], M, baseW, 0);
            CK(cudaGetLastError());
            L += M;
        }
    }
    {   /* inverse: same bookkeeping as ntt_inverse_fused (L walks back DOWN from k-t) */
        int L = c.outer_stages;
        for (int p = c.nms - 1; p >= 0; --p) {
            const int M = c.ms[p];
            L -= M;
            const int s0 = c.k - L - M;
            const unsigned long long S = 1ull << s0;
            CK(cudaMalloc(&c.passI[p], (size_t)S * sizeof(unsigned long long)));
            CK(cudaMalloc(&c.radI[p], (size_t)(1 << M) * sizeof(unsigned long long)));
            const unsigned int tb = (unsigned int)((S + 255) / 256);
            build_pass_table_kernel<<<tb, 256>>>(c.passI[p], S, L, omega_inv);
            CK(cudaGetLastError());
            const unsigned long long Wbase = gl_pow_host(omega_inv, 1ull << (c.k - M));
            build_radix_table_kernel<<<1, (1 << M)>>>(c.radI[p], M, Wbase, 1);
            CK(cudaGetLastError());
        }
    }
    CK(cudaDeviceSynchronize());
    c.tables_cached = true;
}

static unsigned long long fuse_grid(unsigned long long groups, unsigned long long grp_per_blk)
{
    return (groups + grp_per_blk - 1) / grp_per_blk;
}

/* NTT_FUSE_TRACE=1 prints the wall time of every fused kernel launch (diagnostic only: it
   synchronises between launches, so the numbers are the kernels' own times). */
static bool fuse_trace_on()
{
    static const bool on = (std::getenv("NTT_FUSE_TRACE") != nullptr);
    return on;
}

static void fuse_mark(const char *tag, double &t0)
{
    if (!fuse_trace_on()) return;
    CK(cudaDeviceSynchronize());
    const double t1 = now_s();
    std::printf("    [trace] %-28s %.3f ms\n", tag, (t1 - t0) * 1e3);
    t0 = t1;
}

/* forward: outer passes over stages k-1..t, then the tile pass over stages t-1..0.
   `nbatch` slices of `n` words each, laid out consecutively: blockIdx.y selects the slice, so
   one launch transforms the whole batch.  nbatch == 1 (the per-call path) is bit-identical to
   the original single-slice code. */
static void ntt_forward_fused(unsigned long long *d, const FuseCtx &c,
                              unsigned long long omega, unsigned long long nbatch = 1)
{
    unsigned long long n = c.n;
    const unsigned long long stride = n;
    double ft0 = now_s();
    int L = 0;
    for (int p = 0; p < c.nms; ++p) {
        const int M = c.ms[p];
        const unsigned long long S = n >> (L + M);
        const unsigned long long groups = n >> M;
        const unsigned long long gpb = 1024;
        /* cached tables (S3): the same kernel wrote them once per SHAPE instead of once per
           pass of every multiply; when tables_cached is false this is exactly the old path */
        unsigned long long *scr = c.scr, *scr2 = c.scr2;
        if (c.tables_cached) { scr = c.passF[p]; scr2 = c.radF[p]; }
        else {
            build_pass_table_kernel<<<(unsigned int)fuse_grid(S, 256), 256>>>(scr, S, L, omega);
            CK(cudaGetLastError());
            const unsigned long long baseW = gl_pow_host(omega, S << L);
            build_radix_table_kernel<<<1, (1 << M)>>>(scr2, M, baseW, 0);
            CK(cudaGetLastError());
        }
        const unsigned int g = (unsigned int)fuse_grid(groups, gpb);
        dim3 gr(g, (unsigned int)nbatch);
        switch (M) {
            case 1: outer_fwd_kernel<1><<<gr, FUSE_OUTER_THREADS>>>(d, n, L, scr, scr2, gpb, stride); break;
            case 2: outer_fwd_kernel<2><<<gr, FUSE_OUTER_THREADS>>>(d, n, L, scr, scr2, gpb, stride); break;
            case 3: outer_fwd_kernel<3><<<gr, FUSE_OUTER_THREADS>>>(d, n, L, scr, scr2, gpb, stride); break;
            case 4: outer_fwd_kernel<4><<<gr, FUSE_OUTER_THREADS>>>(d, n, L, scr, scr2, gpb, stride); break;
            case 5: outer_fwd_kernel<5><<<gr, FUSE_OUTER_THREADS>>>(d, n, L, scr, scr2, gpb, stride); break;
            case 6: outer_fwd_kernel<6><<<gr, FUSE_OUTER_THREADS>>>(d, n, L, scr, scr2, gpb, stride); break;
            default: std::fprintf(stderr, NTT_PROBE_NAME ": bad outer radix M=%d\n", M); std::exit(3);
        }
        CK(cudaGetLastError());
        fuse_mark("fwd outer pass", ft0);
        L += M;
    }
    if (c.t > 0) {
        const unsigned int blocks = (unsigned int)(n >> c.t);
        const int smem = (int)(1ull << c.t) * (int)sizeof(unsigned long long);
        dim3 gr(blocks, (unsigned int)nbatch);
        tile_kernel<false><<<gr, FUSE_TILE_THREADS, smem>>>(d, nullptr, n, c.k, c.t,
                                                            c.tblF, 0ull, stride);
        CK(cudaGetLastError());
        fuse_mark("fwd tile pass", ft0);
    }
}

/* inverse: tile pass (stages 0..t-1 + pointwise + 1/N), then the outer passes in reverse */
static void ntt_inverse_fused(unsigned long long *d, const unsigned long long *b,
                              const FuseCtx &c, unsigned long long omega_inv,
                              unsigned long long n_scale, unsigned long long nbatch = 1)
{
    unsigned long long n = c.n;
    const unsigned long long stride = n;
    double ft0 = now_s();
    if (c.t > 0) {
        const unsigned int blocks = (unsigned int)(n >> c.t);
        const int smem = (int)(1ull << c.t) * (int)sizeof(unsigned long long);
        dim3 gr(blocks, (unsigned int)nbatch);
        tile_kernel<true><<<gr, FUSE_TILE_THREADS, smem>>>(d, b, n, c.k, c.t, c.tblI,
                                                           n_scale, stride);
        CK(cudaGetLastError());
        fuse_mark("inv tile pass (+pw+scale)", ft0);
    }
    int L = c.outer_stages;                 /* stages already done walking DOWN from k-1 */
    for (int p = c.nms - 1; p >= 0; --p) {
        const int M = c.ms[p];
        L -= M;                             /* prefix sum of the passes before p */
        const int s0 = c.k - L - M;         /* lowest stage of this pass (ascending now) */
        const unsigned long long S = 1ull << s0;
        const unsigned long long groups = n >> M;
        const unsigned long long gpb = 1024;
        unsigned long long *scr = c.scr, *scr2 = c.scr2;
        if (c.tables_cached) { scr = c.passI[p]; scr2 = c.radI[p]; }
        else {
            build_pass_table_kernel<<<(unsigned int)fuse_grid(S, 256), 256>>>(scr, S, L,
                                                                             omega_inv);
            CK(cudaGetLastError());
            const unsigned long long Wbase = gl_pow_host(omega_inv, 1ull << (c.k - M));
            build_radix_table_kernel<<<1, (1 << M)>>>(scr2, M, Wbase, 1);
            CK(cudaGetLastError());
        }
        const unsigned int g = (unsigned int)fuse_grid(groups, gpb);
        dim3 gr(g, (unsigned int)nbatch);
        switch (M) {
            case 1: outer_inv_kernel<1><<<gr, FUSE_OUTER_THREADS>>>(d, n, s0, scr, scr2, gpb, stride); break;
            case 2: outer_inv_kernel<2><<<gr, FUSE_OUTER_THREADS>>>(d, n, s0, scr, scr2, gpb, stride); break;
            case 3: outer_inv_kernel<3><<<gr, FUSE_OUTER_THREADS>>>(d, n, s0, scr, scr2, gpb, stride); break;
            case 4: outer_inv_kernel<4><<<gr, FUSE_OUTER_THREADS>>>(d, n, s0, scr, scr2, gpb, stride); break;
            case 5: outer_inv_kernel<5><<<gr, FUSE_OUTER_THREADS>>>(d, n, s0, scr, scr2, gpb, stride); break;
            case 6: outer_inv_kernel<6><<<gr, FUSE_OUTER_THREADS>>>(d, n, s0, scr, scr2, gpb, stride); break;
            default: std::fprintf(stderr, NTT_PROBE_NAME ": bad outer radix M=%d\n", M); std::exit(3);
        }
        CK(cudaGetLastError());
        fuse_mark("inv outer pass", ft0);
    }
}

/* ===================================================================================== *
 *  THE MULTIPLY ARENA (docs/DEV_STAGE2_GPU_PLAN.md section 18, slice S3)
 *
 *  ntt_poly_mul_host() was written for a probe: one call, one shape, everything allocated,
 *  built and released again.  Measured cost of that discipline when the tree engine calls it
 *  ~4e4 times at the frozen shape (P=24, S=129): 52.4 s for the whole stage-2 tail, of which
 *  45.2 s is INSIDE this function while the transform itself moves a few hundred words --
 *  i.e. almost all of it is ten cudaMallocs, six cudaFrees, four cudaFuncSetAttribute calls,
 *  a fuse_init and two twiddle-table launches PER CALL.
 *
 *  An arena is the same multiply with those per-call costs paid once per SHAPE:
 *    * the six data buffers are cached by (nwords, out_slots),
 *    * the FuseCtx (fusion plan + tile tables + per-PASS twiddle tables) is cached by
 *      (nwords, k, omega) and its per-pass tables are prebuilt with the SAME kernels and the
 *      SAME index arithmetic the per-call path uses, so a cached multiply is bit-identical.
 *  Everything else -- packing, the exactness assertions, the transforms, the carry, the
 *  exact-coefficient extraction and its two-extraction cross-check -- is the very same code,
 *  because the arena is threaded through ntt_poly_mul_host as an extra argument rather than
 *  reimplemented.  Passing nullptr (the default) reproduces the probe's original behaviour
 *  exactly, byte for byte.
 * ===================================================================================== */

struct NttArena {
    int device = -1;
    size_t cap_bytes = 0;                 /* 0 = unlimited */
    size_t bytes = 0;
    size_t attr_words = 0;                /* the shared-memory ceiling pinned on the tile
                                             kernels (see ntt_arena_pin_smem) */
    unsigned long long fuse_hits = 0, fuse_builds = 0, buf_hits = 0, buf_builds = 0;

    struct FuseEntry {
        unsigned long long n = 0;
        int k = 0;
        unsigned long long omega = 0;
        FuseCtx fc;
    };
    struct BufEntry {
        unsigned long long n = 0, out_slots = 0;
        /* slice S4: a batched multiply of `nbatch` slices needs nbatch*N words per buffer,
           so the batch count is part of the key (0 and 1 mean the same thing: one slice) */
        unsigned long long nbatch = 1;
        unsigned long long *dA = nullptr, *dB = nullptr, *dC = nullptr, *dQ = nullptr,
                           *dOut = nullptr, *dRes = nullptr;
    };
    std::vector<FuseEntry> fuses;
    struct BigEntry {                       /* the shape-independent buffers: 3*N words each */
        unsigned long long n = 0, nbatch = 1;
        unsigned long long *dA = nullptr, *dB = nullptr, *dQ = nullptr;
        size_t words = 0;                   /* what this entry costs, for eviction accounting */
    };
    struct SmallEntry {                     /* only these depend on out_slots */
        unsigned long long n = 0, nbatch = 1;
        unsigned long long out_cap = 0;     /* GROWN, never shrunk: reused by every smaller
                                               out_slots at the same N */
        unsigned long long *dOut = nullptr, *dRes = nullptr;
        size_t words = 0;
    };
    std::vector<BigEntry> bigs;
    /* A/B/Q are temporary default-stream scratch. Preserve shape-local dRes below, and
       retain keyed buffers for calls that export a borrowed digit pointer. */
    bool workspace_pool = [] {
        const char *e = std::getenv("NTT_ARENA_WORKSPACE_POOL");
        return !e || !*e || std::atoi(e) != 0;
    }();
    BigEntry workspace;
    unsigned long long workspace_hits=0, workspace_grows=0, workspace_mallocs=0, workspace_frees=0;
    unsigned long long legacy_mallocs=0, legacy_frees=0;
    unsigned long long alias_snapshots=0, alias_snapshot_bytes=0;
    size_t peak_big_bytes=0, peak_small_bytes=0, peak_table_bytes=0, peak_owned_bytes=0;
    size_t peak_fuse_base_bytes=0, peak_full_bytes=0;
    /* Deterministic allocation-failure fixture; never set by the production engine. */
    int workspace_fail_alloc=0;
    std::vector<SmallEntry> smalls;
    BufEntry cur;                           /* the composite the caller is handed */
    unsigned long long overflow = 0;      /* shapes that did NOT fit the cap (per-call path) */
    /* ---- THE TABLE CACHES ARE EVICTABLE, AND THAT IS THE CHEAPEST MEMORY WE HAVE ------------
       ntt_fuse_cache_tables() stores one table pair per outer pass, summing to about N + N/2
       words for a cached shape -- 35% of what one transform length costs, ~1.2 GB of the 4.6 GB
       the real shape holds.  Every byte of it is a PURE CACHE: the per-call path rebuilds the same
       tables with two extra launches, and the kernels that read them check `tables_cached` first,
       so dropping them can only cost speed, never correctness.  When an allocation would exceed
       the cap we therefore evict the caches of every shape EXCEPT the one being allocated (the hot
       shape keeps its caching) and retry before falling back to the per-call path. */
    unsigned long long tbl_evictions = 0, tbl_words_freed = 0;

    /* A cudaMalloc THAT CAN FAIL WITHOUT KILLING THE RUN.  The cap accounting is in words the
       arena believes it owns, but the device's REAL free memory can be smaller -- the stage-2
       engine holds its own pools (the ladder, the frontier, the S5 forests) and the driver
       reserves some -- so an allocation that passes the cap check can still come back "out of
       memory".  Measured, not theorised: at NTT_S4_BATCH_MB=96 the arena's own check passed and
       the very next cudaMalloc failed, and because that site used CK() the whole run died
       (ntt_poly_probe.cu:1634 cudaMalloc dA, exit 2).  Every arena allocation is a CACHE, so
       failing one is recoverable: the caller falls back to the per-call path, which is exactly
       the `overflow` path that already exists for cap refusals. */
    static bool try_malloc(unsigned long long **p, size_t bytes)
    {
        if (cudaMalloc((void **)p, bytes) == cudaSuccess) return true;
        (void)cudaGetLastError();             /* clear the sticky error before anything else runs */
        *p = nullptr;
        return false;
    }

    void drop_workspace()
    {
        /* cudaFree waits for queued default-stream readers, including oracle D2H copies.
           Drop before growth so old/new triples never overlap in device memory. */
        if (workspace.dA) { CK(cudaFree(workspace.dA)); ++workspace_frees; }
        if (workspace.dB) { CK(cudaFree(workspace.dB)); ++workspace_frees; }
        if (workspace.dQ) { CK(cudaFree(workspace.dQ)); ++workspace_frees; }
        if (workspace.words) bytes -= workspace.words*8 + 16;
        workspace=BigEntry{};
    }

    /* Returns the remaining allocation span for an arena-owned device input. Snapshot such
       inputs before lookup: it can overwrite scratch or evict an older exported keyed buffer. */
    bool input_span(const unsigned long long *p, size_t &remaining) const
    {
        if (!p) return false;
        const uintptr_t address=(uintptr_t)p;
        auto contains=[&](const unsigned long long *base, size_t words) {
            if (!base) return false;
            const uintptr_t start=(uintptr_t)base;
            if (address<start || address-start>=words*8) return false;
            if ((address-start)%8) { remaining=0; return true; }
            remaining=words-(address-start)/8; return true;
        };
        if (contains(workspace.dA,workspace.words/3) || contains(workspace.dB,workspace.words/3) ||
            contains(workspace.dQ,workspace.words/3)) return true;
        for (const auto &b : bigs)
            if (contains(b.dA,b.words/3) || contains(b.dB,b.words/3) || contains(b.dQ,b.words/3)) return true;
        for (const auto &b : smalls)
            if (contains(b.dOut,(size_t)(b.out_cap*b.nbatch)) || contains(b.dRes,(size_t)(2*b.nbatch))) return true;
        return false;
    }

    void update_peaks()
    {
        size_t big=workspace.words*8, small=0, table=0, base=0;
        for (const auto &b : bigs) big+=b.words*8;
        for (const auto &b : smalls) small+=b.words*8;
        for (const auto &f : fuses) { table+=(size_t)fuse_table_words(f.fc)*8; base+=fuse_base_words(f.fc)*8; }
        peak_big_bytes=std::max(peak_big_bytes,big); peak_small_bytes=std::max(peak_small_bytes,small);
        peak_table_bytes=std::max(peak_table_bytes,table);
        peak_owned_bytes=std::max(peak_owned_bytes,big+small+table);
        peak_fuse_base_bytes=std::max(peak_fuse_base_bytes,base);
        peak_full_bytes=std::max(peak_full_bytes,big+small+table+base);
    }

    void print_workspace_stats() const
    {
        size_t big=workspace.words*8, small=0, table=0, base=0;
        for (const auto &b : bigs) big+=b.words*8;
        for (const auto &b : smalls) small+=b.words*8;
        for (const auto &f : fuses) { table+=(size_t)fuse_table_words(f.fc)*8; base+=fuse_base_words(f.fc)*8; }
        std::printf("ntt_workspace_stats: pool=%d hits=%llu grows=%llu mallocs=%llu frees=%llu "
                    "workspace_bytes=%llu big_bytes=%llu small_bytes=%llu table_bytes=%llu "
                    "owned_bytes=%llu big_peak_bytes=%llu small_peak_bytes=%llu table_peak_bytes=%llu "
                    "owned_peak_bytes=%llu aliases=%llu alias_bytes=%llu evictions=%llu evicted_words=%llu "
                    "legacy_mallocs=%llu legacy_frees=%llu "
                    "fuse_base_bytes=%llu fuse_base_peak_bytes=%llu full_bytes=%llu full_peak_bytes=%llu\n",
                    (int)workspace_pool, workspace_hits, workspace_grows, workspace_mallocs, workspace_frees,
                    (unsigned long long)(workspace.words*8), (unsigned long long)big, (unsigned long long)small,
                    (unsigned long long)table, (unsigned long long)(big+small+table),
                    (unsigned long long)peak_big_bytes, (unsigned long long)peak_small_bytes,
                    (unsigned long long)peak_table_bytes, (unsigned long long)peak_owned_bytes,
                    alias_snapshots, alias_snapshot_bytes, tbl_evictions, tbl_words_freed, legacy_mallocs, legacy_frees,
                    (unsigned long long)base,(unsigned long long)peak_fuse_base_bytes,
                    (unsigned long long)(big+small+table+base),(unsigned long long)peak_full_bytes);
        std::printf("ntt_fuse_base_stats: compact=%d allocations=%llu frees=%llu live_bytes=%llu peak_bytes=%llu\n",
                    (int)fuse_compact_scratch(),g_fuse_base.allocations,g_fuse_base.frees,
                    g_fuse_base.live_bytes,g_fuse_base.peak_bytes);
    }

    /* the words a cached shape's tables occupy: sum over outer passes of (S + 2^M) for the forward
       and again for the inverse, exactly as ntt_fuse_cache_tables allocates them */
    static unsigned long long fuse_table_words(const FuseCtx &c)
    {
        if (!c.tables_cached) return 0;
        unsigned long long w = 0;
        int L = 0;
        for (int p = 0; p < c.nms; ++p) { const int M = c.ms[p]; w += (c.n >> (L + M)) + (1ull << M); L += M; }
        L = c.outer_stages;
        for (int p = c.nms - 1; p >= 0; --p) { const int M = c.ms[p]; L -= M; w += (1ull << (c.k - L - M)) + (1ull << M); }
        return w;
    }

    static void fuse_drop_tables(FuseCtx &c)
    {
        for (int p = 0; p < FUSE_MAX_PASSES; ++p) {
            if (c.passF[p]) cudaFree(c.passF[p]);
            if (c.radF[p]) cudaFree(c.radF[p]);
            if (c.passI[p]) cudaFree(c.passI[p]);
            if (c.radI[p]) cudaFree(c.radI[p]);
            c.passF[p] = c.radF[p] = c.passI[p] = c.radI[p] = nullptr;
        }
        c.tables_cached = false;             /* the per-call path rebuilds them on demand */
    }

    /* free the table caches of every shape except `keep_n`, AND the big/small buffers of every
       other (n, nbatch) -- see the long comment above: all of it is a cache, the multiply that is
       being set up now is the only live user, and without evicting the BIGS the arena still held
       5627 MB of earlier shapes when the fold's own 3072 MB shape asked for room at P=115200 and
       the run fell back to per-call cudaMalloc and died with "out of memory" (measured).  Returns
       the words freed. */
    unsigned long long evict_other_shapes(unsigned long long keep_n,
                                          unsigned long long keep_nbatch)
    {
        unsigned long long freed = 0;
        size_t entry_charges = 0;
        for (FuseEntry &e : fuses) {
            if (e.n == keep_n || !e.fc.tables_cached) continue;
            freed += fuse_table_words(e.fc);
            fuse_drop_tables(e.fc);
        }
        for (size_t i = bigs.size(); i-- > 0;) {
            BigEntry &b = bigs[i];
            if (b.n == keep_n && b.nbatch == keep_nbatch) continue;
            if (b.dA) { cudaFree(b.dA); ++legacy_frees; }
            if (b.dB) { cudaFree(b.dB); ++legacy_frees; }
            if (b.dQ) { cudaFree(b.dQ); ++legacy_frees; }
            freed += b.words;
            entry_charges += 16;
            bigs.erase(bigs.begin() + (long)i);
        }
        for (size_t i = smalls.size(); i-- > 0;) {
            SmallEntry &s = smalls[i];
            if (s.n == keep_n && s.nbatch == keep_nbatch) continue;
            if (s.dOut) cudaFree(s.dOut);
            if (s.dRes) cudaFree(s.dRes);
            freed += s.words;
            entry_charges += 16;
            smalls.erase(smalls.begin() + (long)i);
        }
        if (freed) { ++tbl_evictions; tbl_words_freed += freed; bytes -= freed * 8 + entry_charges; }
        return freed;
    }

    /* a fused shape becomes unbounded work per call, so it must not be built while another
       shape's buffers are still live on a small card: the caller sets cap_bytes and the two
       out-of-memory paths below simply fall back to the per-call behaviour. */
    ~NttArena() { release(); }

    void release()
    {
        drop_workspace();
        for (FuseEntry &e : fuses) fuse_release(e.fc);
        fuses.clear();
        for (BigEntry &b : bigs) {
            if (b.dA) { cudaFree(b.dA); ++legacy_frees; }
            if (b.dB) { cudaFree(b.dB); ++legacy_frees; }
            if (b.dQ) { cudaFree(b.dQ); ++legacy_frees; }
        }
        bigs.clear();
        for (SmallEntry &b : smalls) {
            if (b.dOut) cudaFree(b.dOut);
            if (b.dRes) cudaFree(b.dRes);
        }
        smalls.clear();
        bytes = 0;
    }
    double mb() const { return (double)bytes / 1048576.0; }
};

static NttArena::BufEntry *ntt_arena_bufs(NttArena *ar, unsigned long long n,
                                          unsigned long long out_slots,
                                          unsigned long long nbatch = 1, bool allow_pool = true)
{
    if (!ar) return nullptr;
    if (nbatch == 0) nbatch = 1;
    /* The three BUFFER-SIZE-INDEPENDENT buffers are cached by (N, nbatch) alone.  Keying them by
       (N, out_slots) -- as the first S4 version did -- multiplies the arena by the number of
       distinct out_slots that share one N, and at the real shape the fold's several multiplies
       at N = 2^27 with different out_slots would each have wanted their own 3*N words (3 GB).
       Only dOut (out_slots words) and dRes really depend on out_slots. */
    NttArena::BigEntry *big = nullptr;
    if (ar->workspace_pool && allow_pool) {
        if (!nbatch || n>((size_t)-1)/nbatch || n*nbatch>((size_t)-1-16)/24) { ++ar->overflow; return nullptr; }
        const size_t required=(size_t)(n*nbatch);
        if (ar->workspace.words/3>=required) { big=&ar->workspace; ++ar->workspace_hits; ++ar->buf_hits; }
        else {
            ar->drop_workspace();
            const size_t need=required*24+16;
            if (ar->cap_bytes && (need>ar->cap_bytes || ar->bytes>ar->cap_bytes-need))
                ar->evict_other_shapes(n,nbatch);
            if (ar->cap_bytes && (need>ar->cap_bytes || ar->bytes>ar->cap_bytes-need)) {
                ++ar->overflow; return nullptr;
            }
            NttArena::BigEntry e;
            auto allocate=[&](unsigned long long **p, int index) {
                if (ar->workspace_fail_alloc==index) { *p=nullptr; return false; }
                ++ar->workspace_mallocs; return NttArena::try_malloc(p,required*8);
            };
            if (!allocate(&e.dA,1) || !allocate(&e.dB,2) || !allocate(&e.dQ,3)) {
                if (e.dA) { CK(cudaFree(e.dA)); ++ar->workspace_frees; }
                if (e.dB) { CK(cudaFree(e.dB)); ++ar->workspace_frees; }
                if (e.dQ) { CK(cudaFree(e.dQ)); ++ar->workspace_frees; }
                ++ar->overflow; return nullptr;
            }
            e.n=n; e.nbatch=nbatch; e.words=required*3;
            ar->workspace=e; ar->bytes+=need; big=&ar->workspace;
            ++ar->workspace_grows; ++ar->buf_builds; ar->update_peaks();
        }
    } else {
        /* Exported digits retain the keyed-cache lifetime. No shared scratch reader remains
           after host input snapshots and prior default-stream work have completed. */
        if (ar->workspace_pool) ar->drop_workspace();
    }
    for (NttArena::BigEntry &b : ar->bigs)
        if (!big)
        if (b.n == n && b.nbatch == nbatch) { big = &b; ++ar->buf_hits; break; }
    if (!big) {
        const size_t need = (size_t)(3 * n * nbatch) * sizeof(unsigned long long) + 16;
        /* BEFORE REFUSING, EVICT EVERYTHING THE ARENA CACHES FOR OTHER SHAPES (see the NttArena
           comment).  The table caches alone were not enough: at P=115200 the fold's shape needed
           3072 MB while the arena still held 5627 MB of earlier shapes' BIG buffers, so the run
           fell back to per-call cudaMalloc and died with "out of memory" (measured).  The hot
           shape keeps its own entry, and every evicted shape is rebuilt on demand. */
        if (ar->cap_bytes && ar->bytes + need > ar->cap_bytes)
            ar->evict_other_shapes(n, nbatch);
        if (ar->cap_bytes && ar->bytes + need > ar->cap_bytes) {
            ++ar->overflow;
            if (ar->overflow <= 4)
                std::fprintf(stderr, "%s: arena refuses N=%llu nbatch=%llu (needs %.0f MB, has "
                                     "%.0f of %.0f MB): falling back to per-call cudaMalloc\n",
                             NTT_PROBE_NAME, n, nbatch, need / 1048576.0, ar->bytes / 1048576.0,
                             ar->cap_bytes / 1048576.0);
            return nullptr;
        }
        NttArena::BigEntry e;
        e.n = n;
        e.nbatch = nbatch;
        /* the three big buffers are the whole cost of a shape, so this is where a real out-of-
           memory shows up first -- and it is a cache, so it degrades instead of aborting */
        auto allocate_legacy=[&](unsigned long long **p) {
            ++ar->legacy_mallocs; return NttArena::try_malloc(p,n*nbatch*sizeof(unsigned long long));
        };
        if (!allocate_legacy(&e.dA) || !allocate_legacy(&e.dB) || !allocate_legacy(&e.dQ)) {
            if (e.dA) { cudaFree(e.dA); ++ar->legacy_frees; }
            if (e.dB) { cudaFree(e.dB); ++ar->legacy_frees; }
            if (e.dQ) { cudaFree(e.dQ); ++ar->legacy_frees; }
            ++ar->overflow;
            if (ar->overflow <= 4)
                std::fprintf(stderr, "%s: arena could NOT allocate N=%llu nbatch=%llu (%.0f MB): "
                                     "the device is out of memory (the cap says %.0f of %.0f MB "
                                     "free) -- falling back to per-call cudaMalloc\n",
                             NTT_PROBE_NAME, n, nbatch, need / 1048576.0, ar->bytes / 1048576.0,
                             ar->cap_bytes / 1048576.0);
            return nullptr;
        }
        e.words = (size_t)(3 * n * nbatch);
        ar->bytes += need;
        ar->bigs.push_back(e);
        ++ar->buf_builds;
        big = &ar->bigs.back();
        ar->update_peaks();
    }
    /* The dOut buffer is cached by (N, nbatch) and merely GROWN to the largest out_slots at that
       N.  Keying it by (N, out_slots) -- the second version -- was catastrophic at the real
       shape: every P in a range maps to the SAME power-of-two N but a DIFFERENT out_slots, so
       one N accumulated thousands of small entries whose sum (GBs) blew the cap and pushed the
       largest shape onto the per-call path, where the allocation then failed. */
    NttArena::SmallEntry *small = nullptr;
    for (NttArena::SmallEntry &b : ar->smalls)
        if (b.n == n && b.nbatch == nbatch) { small = &b; ++ar->buf_hits; break; }
    if (small && out_slots > small->out_cap) {
        /* grow: the extra words are charged to the cap before anything is allocated */
        const size_t extra = (size_t)((out_slots - small->out_cap) * nbatch) *
                             sizeof(unsigned long long);
        if (ar->cap_bytes && ar->bytes + extra > ar->cap_bytes) {
            ++ar->overflow;
            return nullptr;
        }
        const size_t old_bytes=(size_t)(small->out_cap*nbatch)*8;
        if (small->dOut) cudaFree(small->dOut);
        small->dOut = nullptr;
        small->out_cap = 0;
        small->words=(size_t)(2*nbatch);
        ar->bytes-=old_bytes;
        if (!NttArena::try_malloc(&small->dOut, out_slots * nbatch * sizeof(unsigned long long))) {
            /* out_cap is left at 0, so the next call at this (N, nbatch) simply retries the
               growth -- the entry stays valid, only unusable until an allocation succeeds */
            ++ar->overflow;
            return nullptr;
        }
        small->out_cap = out_slots;
        small->words = (size_t)(out_slots * nbatch + 2 * nbatch);
        ar->bytes += (size_t)(out_slots*nbatch)*8;
    } else if (!small) {
        const size_t need = (size_t)(out_slots * nbatch + 2 * nbatch) *
                            sizeof(unsigned long long) + 16;
        if (ar->cap_bytes && ar->bytes + need > ar->cap_bytes) {
            ++ar->overflow;
            return nullptr;
        }
        NttArena::SmallEntry e;
        e.n = n;
        e.nbatch = nbatch;
        e.out_cap = out_slots;
        if (!NttArena::try_malloc(&e.dOut, out_slots * nbatch * sizeof(unsigned long long)) ||
            !NttArena::try_malloc(&e.dRes, 2 * nbatch * sizeof(unsigned long long))) {
            if (e.dOut) cudaFree(e.dOut);
            if (e.dRes) cudaFree(e.dRes);
            ++ar->overflow;
            if (ar->overflow <= 4)
                std::fprintf(stderr, "%s: arena could NOT allocate the small buffers for N=%llu "
                                     "nbatch=%llu (%.0f MB): the device is out of memory -- "
                                     "falling back to per-call cudaMalloc\n",
                             NTT_PROBE_NAME, n, nbatch, need / 1048576.0);
            return nullptr;
        }
        e.words = (size_t)(out_slots * nbatch + 2 * nbatch);
        ar->bytes += need;
        ar->smalls.push_back(e);
        ++ar->buf_builds;
        small = &ar->smalls.back();
    }
    ar->cur.n = n;
    ar->cur.out_slots = out_slots;
    ar->cur.nbatch = nbatch;
    ar->cur.dA = big->dA; ar->cur.dB = big->dB; ar->cur.dQ = big->dQ; ar->cur.dC = nullptr;
    ar->cur.dOut = small->dOut; ar->cur.dRes = small->dRes;
    ar->update_peaks();
    return &ar->cur;
}

/* `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` is a property of the FUNCTION, not of a
   FuseCtx: fuse_init() sets it to ITS OWN 2^t words, so building a later shape with a SMALLER
   t would leave the tile kernel unable to launch for a cached shape with a larger t
   (measured, not theorised: that is CUDA "invalid argument" at the tile launch, i.e. exactly
   the failure mode of a per-call design that has been made persistent).  The arena therefore
   keeps a running MAXIMUM and pins the attribute to it after every build, so every cached
   shape's launch stays legal. */
static void ntt_arena_pin_smem(NttArena *ar, int t)
{
    if (!ar) return;
    const size_t need = (size_t)1 << ((t < 0) ? 0 : t);
    if (need > ar->attr_words) ar->attr_words = need;
    const int smem = (int)(ar->attr_words * sizeof(unsigned long long));
    CK(cudaFuncSetAttribute(tile_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                            smem));
    CK(cudaFuncSetAttribute(tile_kernel<true>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                            smem));
}

/* the FuseCtx of this shape, tables included, built once and shared by every later call */
static void ntt_arena_fuse(NttArena *ar, unsigned long long n, int k, unsigned long long omega,
                           unsigned long long omega_inv, FuseCtx &out)
{
    if (!ar) { fuse_init(out, n, k, omega, omega_inv); return; }
    for (NttArena::FuseEntry &e : ar->fuses) {
        if (e.n == n && e.k == k && e.omega == omega) {
            out = e.fc;                                  /* pointers are owned by the arena */
            out.arena_borrowed = true;
            ntt_arena_pin_smem(ar, e.fc.t);
            ++ar->fuse_hits;
            return;
        }
    }
    NttArena::FuseEntry e;
    e.n = n;
    e.k = k;
    e.omega = omega;
    const size_t need = (size_t)(n + n / 2 + 4 * FUSE_MAX_PASSES * 64) *
                        sizeof(unsigned long long);
    if (ar->cap_bytes && ar->bytes + need > ar->cap_bytes) {
        ++ar->overflow;
        fuse_init(out, n, k, omega, omega_inv);      /* over budget: per-call tables */
        ntt_arena_pin_smem(ar,out.t);
        return;
    }
    fuse_init(e.fc, n, k, omega, omega_inv);
    ntt_arena_pin_smem(ar, e.fc.t);                      /* undo fuse_init's own (smaller) set */
    ntt_fuse_cache_tables(e.fc, omega, omega_inv);
    ar->bytes += need;
    ar->fuses.push_back(e);
    ar->update_peaks();
    ++ar->fuse_builds;
    out = ar->fuses.back().fc;
    out.arena_borrowed = true;
}

/* device arithmetic selftest kernel: out[i] = gl_mul(x[i], y[i]) */
__global__ void gl_mul_test_kernel(const unsigned long long *x, const unsigned long long *y,
                                   unsigned long long *out, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = gl_mul(x[i], y[i]);
}

/* device twiddle check: out[i] = omega^i for i in [0, n) (compared against the host) */
__global__ void gl_twiddle_test_kernel(unsigned long long *out, unsigned long long n,
                                       unsigned long long omega)
{
    const unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (i < n) out[i] = gl_twiddle(omega, i);
}

/* naive in-kernel DFT (O(n^2), n <= a few thousand): the reference the fast butterfly
   stages are checked against, with no index algebra in common. */
__global__ void dft_direct_kernel(const unsigned long long *x, unsigned long long *out,
                                  unsigned long long n, unsigned long long omega)
{
    const unsigned long long k = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x;
    if (k >= n) return;
    unsigned long long acc = 0;
    for (unsigned long long j = 0; j < n; ++j) {
        acc = gl_add_dev(acc, gl_mul(x[j], gl_twiddle(omega, (j * k) % n)));
    }
    out[k] = acc;
}

/* --------------------------------------------------------------------------------- */
/* unit test of the reduction against GMP (host side, cheap)                          */
/* --------------------------------------------------------------------------------- */
static int gl_selftest()
{
    mpz_t p, a, b, z, want, got;
    mpz_inits(p, a, b, z, want, got, nullptr);
    const unsigned long long glp = GL_P;
    mpz_import(p, 1, -1, 8, 0, 0, &glp);

    unsigned long long s = 0x243F6A8885A308D3ull;
    const unsigned long long edges[] = {0ull, 1ull, 2ull, GL_P - 1, GL_P - 2, GL_P >> 1,
                                        0xFFFFFFFFull, 0xFFFFFFFF00000000ull};
    const int nedges = (int)(sizeof(edges) / sizeof(edges[0]));
    int tested = 0;
    int reduce_cases = 0;
    for (int t = 0; t < 3000; ++t) {
        unsigned long long x, y;
        if (t < nedges * nedges) {
            x = edges[t / nedges];
            y = edges[t % nedges];
        } else {
            s = s * 6364136223846793005ull + 1442695040888963407ull;
            x = s;
            s = s * 6364136223846793005ull + 1442695040888963407ull;
            y = s;
        }
        x %= GL_P; y %= GL_P;
        const unsigned long long got_d = gl_mul_host(x, y);
        /* GMP: exact product then one mod */
        mpz_import(a, 1, -1, 8, 0, 0, &x);
        mpz_import(b, 1, -1, 8, 0, 0, &y);
        mpz_mul(z, a, b);
        mpz_mod(want, z, p);
        mpz_import(got, 1, -1, 8, 0, 0, &got_d);
        ++tested;
        if (mpz_cmp(want, got) != 0) {
            std::fprintf(stderr, "gl_selftest: FAILED at x=%llu y=%llu (got %llu, want %s)\n",
                         x, y, got_d, mpz_get_str(nullptr, 10, want));
            mpz_clears(p, a, b, z, want, got, nullptr);
            return 1;
        }
    }
    /* a few modular exponentiations too: gl_pow is what builds the omega / 1/N constants */
    for (int t = 0; t < 24; ++t) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        const unsigned long long base = s % GL_P;
        const unsigned long long e = (s >> 7) % (1ull << 40);
        const unsigned long long got_d = gl_pow_host(base, e);
        mpz_import(a, 1, -1, 8, 0, 0, &base);
        mpz_import(b, 1, -1, 8, 0, 0, &e);
        mpz_powm(want, a, b, p);
        mpz_import(got, 1, -1, 8, 0, 0, &got_d);
        ++tested;
        if (mpz_cmp(want, got) != 0) {
            std::fprintf(stderr, "gl_selftest: gl_pow FAILED base=%llu e=%llu\n", base, e);
            mpz_clears(p, a, b, z, want, got, nullptr);
            return 1;
        }
    }
    mpz_clears(p, a, b, z, want, got, nullptr);

    /*
     * Independent test of gl_reduce on ARBITRARY (lo, hi) pairs, including the full
     * hi < 2^64 range: this is the core that the two-step fold got wrong, and it is not
     * reachable through gl_mul alone (gl_mul always produces a genuine 64x64 product).
     */
    {
        mpz_t p2, lo2, hi2, got2, w2;
        mpz_inits(p2, lo2, hi2, got2, w2, nullptr);
        mpz_import(p2, 1, -1, 8, 0, 0, &glp);
        unsigned long long r = 0xC0FFEE123456789ull;
        int bad = 0;
        std::vector<unsigned long long> los, his;
        /* hand-picked boundaries plus random pairs */
        const unsigned long long los0[] = {0, 1, 2, GL_P - 1, GL_P - 2, 0xFFFFFFFFull,
                                           0xFFFFFFFF00000000ull, ~0ull};
        const unsigned long long his0[] = {0, 1, 2, 3, 0xFFFFFFFFull, 0x1FFFFFFFFull,
                                           0xFFFFFFFF00000000ull, ~0ull};
        for (unsigned long long x : los0) {
            for (unsigned long long y : his0) { los.push_back(x); his.push_back(y); }
        }
        for (int t = 0; t < 100000; ++t) {
            r = r * 6364136223846793005ull + 1442695040888963407ull;
            los.push_back(r);
            r = r * 6364136223846793005ull + 1442695040888963407ull;
            /* mix: sometimes a near-2^64 hi (what a product of two values < p produces) */
            his.push_back((t & 1) ? r : (0xFFFFFFFFFFFFFFFFull - (r & 0xFFFFFFFFull)));
        }
        for (size_t i = 0; i < los.size(); ++i) {
            const unsigned long long got_d = gl_reduce(los[i], his[i]);
            mpz_import(lo2, 1, -1, 8, 0, 0, &los[i]);
            mpz_import(hi2, 1, -1, 8, 0, 0, &his[i]);
            mpz_mul_2exp(hi2, hi2, 64);
            mpz_add(hi2, hi2, lo2);
            mpz_mod(w2, hi2, p2);   /* NOT `want`: it was mpz_clear()ed at the end of the block above, and writing through a cleared mpz_t is a use-after-free -- it corrupted the heap (0xC0000374 / 0xC0000005) in ~5 % of runs, before the first output line */
            mpz_import(got2, 1, -1, 8, 0, 0, &got_d);
            if (mpz_cmp(w2, got2) != 0) {
                if (bad == 0) {
                    std::fprintf(stderr,
                                 "gl_selftest: gl_reduce FAILED lo=%llu hi=%llu got=%llu\n",
                                 los[i], his[i], got_d);
                }
                ++bad;
            }
        }
        mpz_clears(p2, lo2, hi2, got2, w2, nullptr);
        if (bad) return 1;
        reduce_cases = (int)los.size();
        std::printf(NTT_PROBE_NAME ": gl_reduce selftest vs GMP on %d (lo,hi) pairs OK\n",
                    reduce_cases);
    }
    mpz_inits(p, a, b, z, want, got, nullptr);
    mpz_import(p, 1, -1, 8, 0, 0, &glp);

    /*
     * The kernels use the DEVICE gl_mul (the two-64-bit-halves version), which is a
     * different code path from the host's split version, so run the same comparison on the
     * device.  200k pairs, ~0.2 s.
     */
    const int nt = 200000;
    std::vector<unsigned long long> hx(nt), hy(nt);
    {
        unsigned long long r = 0x9E3779B97F4A7C15ull;
        for (int i = 0; i < nt; ++i) {
            if (i < nedges * nedges) {
                hx[(size_t)i] = edges[i / nedges];
                hy[(size_t)i] = edges[i % nedges];
            } else {
                r = r * 6364136223846793005ull + 1442695040888963407ull; hx[(size_t)i] = r;
                r = r * 6364136223846793005ull + 1442695040888963407ull; hy[(size_t)i] = r;
            }
            hx[(size_t)i] %= GL_P;
            hy[(size_t)i] %= GL_P;
        }
    }
    unsigned long long *dx = nullptr, *dy = nullptr, *dz = nullptr;
    CK(cudaMalloc(&dx, (size_t)nt * 8));
    CK(cudaMalloc(&dy, (size_t)nt * 8));
    CK(cudaMalloc(&dz, (size_t)nt * 8));
    CK(cudaMemcpy(dx, hx.data(), (size_t)nt * 8, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dy, hy.data(), (size_t)nt * 8, cudaMemcpyHostToDevice));
    gl_mul_test_kernel<<<(nt + 255) / 256, 256>>>(dx, dy, dz, nt);
    CK(cudaGetLastError());
    std::vector<unsigned long long> hz(nt, 0);
    CK(cudaMemcpy(hz.data(), dz, (size_t)nt * 8, cudaMemcpyDeviceToHost));

    int dtested = 0, dbad = 0;
    for (int i = 0; i < nt; ++i) {
        mpz_import(a, 1, -1, 8, 0, 0, &hx[(size_t)i]);
        mpz_import(b, 1, -1, 8, 0, 0, &hy[(size_t)i]);
        mpz_mul(z, a, b);
        mpz_mod(want, z, p);
        mpz_import(got, 1, -1, 8, 0, 0, &hz[(size_t)i]);
        ++dtested;
        if (mpz_cmp(want, got) != 0) {
            if (dbad == 0) {
                std::fprintf(stderr, "gl_selftest(device): FAILED at x=%llu y=%llu got=%llu\n",
                             hx[(size_t)i], hy[(size_t)i], hz[(size_t)i]);
            }
            ++dbad;
        }
    }
    mpz_clears(p, a, b, z, want, got, nullptr);
    cudaFree(dx); cudaFree(dy); cudaFree(dz);
    if (dbad) return 1;
    std::printf(NTT_PROBE_NAME ": reduction selftest vs GMP: %d host + %d device cases and %d "
                "(lo,hi) folds OK (p=0x%llX)\n", tested, dtested, reduce_cases,
                (unsigned long long)GL_P);
    return 0;
}

/* --------------------------------------------------------------------------------- */
/* parameter search: smallest power-of-two N and the matching bpw                     */
/* --------------------------------------------------------------------------------- */
struct NttCfg {
    unsigned long long nwords;      /* N = 2^k transform length */
    unsigned long long slot_words;  /* words occupied by one coefficient (padded) */
    unsigned long long L;           /* NONZERO digits per operand = the true term count */
    int k, bpw;
    bool ok;
    std::string why;
};

/* =====================================================================================
 * THE EXACTNESS BOUND: L TERMS, NOT N.
 *
 * The inverse NTT returns the CYCLIC convolution of the two packed operands:
 *      c_k = sum_{i+j == k (mod N)} hA[i] * hB[j]     (mod p),
 * and the answer is exact iff c_k < p for every k (then the mod-p value equals the integer
 * value).  The counting argument the old code used bounded the NUMBER OF TERMS by the array
 * length N, which is only tight when the payload fills the whole array.  The packed operands
 * are zero beyond the payload, so the real term count is the number of NONZERO digits:
 *
 *   [P1]  coefficient i of the operand occupies the whole `slot_words` digits starting at
 *         digit i*slot_words (word-aligned stride, slot_stride = slot_words*bpw), and every
 *         bit of that window at or past slot_bits is zero (the packer writes at most
 *         slot_bits bits per coefficient and the slots do not overlap), so the nonzero digits
 *         of one operand are contained in [0, P*slot_words) -- at most L = P*slot_words.
 *   [P2]  for a FIXED k the pairs (i,j) with i+j == k are in bijection with i, so their
 *         number is at most |{i : hA[i] != 0}| <= L (and symmetrically at most L from B).
 *   [P3]  every digit is < 2^bpw, asserted at run time from the ACTUAL array contents (the
 *         convergence assert below), so each product is <= (2^bpw - 1)^2 and
 *              c_k <= L * (2^bpw - 1)^2 .
 * Therefore  L * (2^bpw - 1)^2 < p  is sufficient for exactness.  It is also much weaker
 * than the old N*(2^bpw)^2 < p, and that is the whole point: for P=8192/S=5153 the old form
 * allowed only bpw=18 (L was irrelevant, N=2^24 was the binding term) while L allows
 * bpw=21 with L=8192*492=4030464 -- the same payload now fits in N=2^23 instead of 2^24, so
 * every one of the 13 passes moves half the bytes.
 *
 * The bound is PROVED above, not assumed: L is computed exactly from P and slot_words, the
 * comparison is exact GMP integer arithmetic, and for every shape whose full convolution fits
 * in host memory the TRUE maximum c_k is additionally computed with GMP on the very digits
 * the device was given (max_conv_coeff_gmp) and asserted < p, with the ratio
 * max_c_k / (2^bpw - 1)^2 printed next to L.  The generic `bench` mode, which has no P, uses
 * the same argument with L = ceil(payload_bits/bpw): all nonzero digits of one operand lie in
 * the first payload_bits bits.  The old N-based test is still available through
 * exact_ok_nterms and is printed for comparison.
 * ===================================================================================== */

/* L * (2^bpw - 1)^2 < p, in exact integer arithmetic */
static bool exact_ok_terms(unsigned long long L, int bpw)
{
    if (bpw <= 0 || bpw > 62) return false;
    if (L == 0) return true;
    /* L*(2^bpw-1)^2 < L*2^(2bpw) < 2^(bitlen(L)+2bpw) and >= 2^(bitlen(L)+2bpw-3), so the
       bit length settles everything outside [2^62, 2^68).  Inside it, do it with GMP. */
    const int bl = bitlen_u64(L) + 2 * bpw;
    if (bl <= 62) return true;                            /* < 2^62 < p */
    if (bl >= 68) return false;                           /* >= 2^65 > p */
    mpz_t Lz, bz, wz, one, prod, p;
    mpz_inits(Lz, bz, wz, one, prod, p, nullptr);
    mpz_import(Lz, 1, -1, 8, 0, 0, &L);
    mpz_set_ui(one, 1);
    mpz_mul_2exp(bz, one, (unsigned long)bpw);
    mpz_sub_ui(bz, bz, 1);
    mpz_mul(wz, bz, bz);
    mpz_mul(prod, Lz, wz);
    const unsigned long long glp = GL_P;
    mpz_import(p, 1, -1, 8, 0, 0, &glp);
    const bool r = (mpz_cmp(prod, p) < 0);
    mpz_clears(Lz, bz, wz, one, prod, p, nullptr);
    return r;
}

/* the OLD, conservative test N*(2^bpw)^2 < p -- kept so the probe can print both numbers
   side by side (the old one is what test_ntt_poly.ps1 asserts on, see the note in run_poly) */
static bool exact_ok_nterms(unsigned long long n, int bpw)
{
    return exact_ok_terms(n, bpw);
}

/* log2 of an mpz, as a double (mpz_sizeinbase rounds UP, so near a power of two it says
   "64 bits" for a number that is 2^63.9999 -- which is exactly the case here, p ~ 2^64, so
   the printed "2^x" must not be derived from the bit count alone, and the VERDICT never is) */
static double mpz_log2d(const mpz_t x)
{
    if (mpz_sgn(x) <= 0) return 0.0;
    long e = 0;
    const double d = mpz_get_d_2exp(&e, x);           /* d in [0.5, 1) */
    return (double)e + std::log2(d);
}

/* the same for p, printed for contrast */
static double mpz_log2d_p()
{
    mpz_t p;
    mpz_init(p);
    const unsigned long long glp = GL_P;
    mpz_import(p, 1, -1, 8, 0, 0, &glp);
    const double r = mpz_log2d(p);
    mpz_clear(p);
    return r;
}

/* L*(2^bpw-1)^2 as a bit count (the printed bound) and as a double ratio against p */
static double coeff_bound_bits_terms(unsigned long long L, int bpw)
{
    mpz_t Lz, bz, wz, one, prod;
    mpz_inits(Lz, bz, wz, one, prod, nullptr);
    mpz_import(Lz, 1, -1, 8, 0, 0, &L);
    mpz_set_ui(one, 1);
    mpz_mul_2exp(bz, one, (unsigned long)bpw);
    mpz_sub_ui(bz, bz, 1);
    mpz_mul(wz, bz, bz);
    mpz_mul(prod, Lz, wz);
    const double r = mpz_log2d(prod);
    mpz_clears(Lz, bz, wz, one, prod, nullptr);
    return r;
}

/* bound / p as a double, computed exactly then converted (the margin, printed) */
static double bound_over_p_terms(unsigned long long L, int bpw)
{
    mpz_t Lz, bz, wz, one, prod, p;
    mpz_inits(Lz, bz, wz, one, prod, p, nullptr);
    mpz_import(Lz, 1, -1, 8, 0, 0, &L);
    mpz_set_ui(one, 1);
    mpz_mul_2exp(bz, one, (unsigned long)bpw);
    mpz_sub_ui(bz, bz, 1);
    mpz_mul(wz, bz, bz);
    mpz_mul(prod, Lz, wz);
    const unsigned long long glp = GL_P;
    mpz_import(p, 1, -1, 8, 0, 0, &glp);
    const double r = mpz_get_d(prod) / mpz_get_d(p);
    mpz_clears(Lz, bz, wz, one, prod, p, nullptr);
    return r;
}

/* the OLD, conservative bound printed for contrast: log2 of N*(2^bpw - 1)^2 */
static double coeff_bound_bits(unsigned long long n, int bpw)
{
    mpz_t nz, bz, wz, one, prod;
    mpz_inits(nz, bz, wz, one, prod, nullptr);
    mpz_import(nz, 1, -1, 8, 0, 0, &n);
    mpz_set_ui(one, 1);
    mpz_mul_2exp(bz, one, (unsigned long)bpw);
    mpz_sub_ui(bz, bz, 1);
    mpz_mul(wz, bz, bz);
    mpz_mul(prod, nz, wz);
    const double r = mpz_log2d(prod);
    mpz_clears(nz, bz, wz, one, prod, nullptr);
    return r;
}

/*
 * Choose the smallest power-of-two transform length N = 2^k and its bpw.
 *
 * Two constraints:
 *   (1) EXACTNESS -- a convolution coefficient is a sum of at most L = P*slot_words products
 *       of two bpw-bit words (L = the number of NONZERO digits per operand), NOT of N
 *       products, so the criterion is L * (2^bpw - 1)^2 < p, checked with exact integer
 *       arithmetic (see the proof above exact_ok_terms).  This is only valid while every
 *       DIGIT of the packed operands is < 2^bpw, which is what the word-aligned slot stride
 *       (below) guarantees and what run_poly asserts from the actual array contents.
 *       NOTE the consequence: L does NOT depend on N, so the largest usable bpw is a property
 *       of the SHAPE (P, slot_bits) alone, and N is then the smallest power of two that the
 *       capacity test below accepts.  bpw and slot_words are therefore chosen by a direct
 *       downward scan of b (larger b -> fewer slot_words -> smaller L), not by a fixed point.
 *   (2) CAPACITY -- one coefficient occupies slot_words = ceil(slot_bits/bpw) whole digits
 *       (word-aligned packing), so the two P-coefficient operands need 2*P*slot_words digits
 *       and their linear product needs 2*P*slot_words - 1.  The slot assembler also peeks one
 *       digit past the window, hence the extra word.
 *       (Packing compactly at exactly slot_bits bits per coefficient -- i.e. only
 *       ceil(2*P*slot_bits/bpw) digits -- is NOT usable: it breaks (1), see run_poly.)
 *
 *   (3) `force_bpw` (default 0 = auto) makes the CHOICE of bpw the caller's, and the caller is
 *       the only thing that can legitimately make it: the DEVICE-descent caller (S5 in
 *       stage2_tree_gpu.cu) packs the operands itself, so whether slot_stride == slot_bits is
 *       visible to it and to nobody else.  It exists because slot_stride = ceil(slot_bits/bpw)*bpw
 *       is slot_bits only when bpw divides slot_bits, and the S5 reducer reads ONE WINDOW PER
 *       COEFFICIENT, so a stride wider than the window would put each coefficient's block one
 *       word outside the window the reducer asserts against.  Everything else here is unchanged:
 *       the exactness bound L*(2^bpw-1)^2 < p is still enforced for the forced bpw, and it must
 *       pass, otherwise the choice is rejected exactly as an auto choice would be.
 */
static NttCfg choose_cfg(unsigned long long payload_bits, unsigned long long slot_bits,
                         unsigned long long P = 0, int force_bpw = 0)
{
    NttCfg c{};
    c.ok = false;
    c.L = 0;
    if (force_bpw != 0) {
        if (force_bpw < 1 || force_bpw > 62) {
            c.why = "the forced bpw is outside 1..62";
            return c;
        }
        const unsigned long long s =
            (slot_bits + (unsigned long long)force_bpw - 1) / (unsigned long long)force_bpw;
        if (P > 0 && P > (~0ull) / s) {
            c.why = "the forced bpw overflows L = P * ceil(slot_bits/bpw)";
            return c;
        }
        const unsigned long long l =
            (P > 0) ? (P * s)
                    : ((payload_bits + (unsigned long long)force_bpw - 1) /
                       (unsigned long long)force_bpw);
        if (!exact_ok_terms(l, force_bpw)) {
            c.why = "the forced bpw fails the exactness bound L*(2^bpw-1)^2 < p";
            return c;
        }
        c.slot_words = s;
        c.L = l;
        c.bpw = force_bpw;
        for (int k = 1; k <= 32; ++k) {
            const unsigned long long n = 1ull << k;
            if (n > (1ull << 32) / 8) break;
            const unsigned long long need_words =
                (P > 0) ? (2 * P * c.slot_words + 1)
                        : ((2 * payload_bits + (unsigned long long)force_bpw - 1) /
                           (unsigned long long)force_bpw + 1);
            if (n >= need_words) {
                c.nwords = n;
                c.k = k;
                c.ok = true;
                return c;
            }
        }
        c.bpw = 0;
        c.slot_words = 0;
        c.L = 0;
        c.why = "no power-of-two transform length holds the forced bpw's slot words";
        return c;
    }
    for (int k = 1; k <= 32; ++k) {
        const unsigned long long n = 1ull << k;
        if (n > (1ull << 32) / 8) break;
        /* largest bpw with L*(2^bpw - 1)^2 < p.  (The old code used N*(2^bpw)^2 < p and then
           subtracted one bit "for safety"; that is what forced bpw=18/N=2^24 on the P=8192
           shape.  With the term count L the comparison is exact and no fudge is needed.) */
        int bpw = 0;
        unsigned long long sw = 0, L = 0;
        for (int b = 62; b >= 1; --b) {
            const unsigned long long s =
                (slot_bits + (unsigned long long)b - 1) / (unsigned long long)b;
            if (P > 0 && P > (~0ull) / s) continue;          /* L would overflow: skip b */
            const unsigned long long l =
                (P > 0) ? (P * s)
                        : ((payload_bits + (unsigned long long)b - 1) / (unsigned long long)b);
            if (!exact_ok_terms(l, b)) continue;
            bpw = b;
            sw = s;
            L = l;
            break;
        }
        if (bpw <= 0) continue;
        c.slot_words = sw;
        c.L = L;
        /* word-aligned footprint: 2*P*slot_words digits, plus the top digit the assembler
           peeks at.  (With P unknown -- the generic `bench` mode -- fall back to the
           bit-based estimate.) */
        const unsigned long long need_words =
            (P > 0) ? (2 * P * c.slot_words + 1)
                    : ((2 * payload_bits + (unsigned long long)bpw - 1) /
                       (unsigned long long)bpw + 1);
        if (n >= need_words) {
            c.nwords = n;
            c.k = k;
            c.bpw = bpw;
            c.ok = true;
            return c;
        }
    }
    c.why = "no power-of-two transform length up to 2^32 fits this payload: the exactness "
            "bound L*(2^bpw-1)^2 < p cannot be met for ANY bpw, or no transform length "
            "reaches 2*P*ceil(slot_bits/bpw)+1 digits";
    return c;
}

/* --------------------------------------------------------------------------------- */
/* INDEPENDENT check of the exactness bound: the TRUE maximum convolution coefficient  */
/* --------------------------------------------------------------------------------- */
/*
 * The counting bound c_k <= L*(2^bpw - 1)^2 is a proof, but a proof of a bound is only as
 * good as the L it uses, so the probe ALSO measures the real thing wherever that is
 * affordable, on the very digits the device was fed.
 *
 * Trick: build  A = sum_j hA[j]*2^(64*j)  and  B = sum_j hB[j]*2^(64*j)  as GMP integers
 * (mpz_import of the first L limbs -- the nonzero region is exactly the first L limbs).  Then
 * limb k of C = A*B is
 *      limb_k(C) = sum_{i+j=k} hA[i]*hB[j] + carry from the limb below,
 * and the carry is ZERO as long as every c_k < 2^64 -- which is exactly what the assertion
 * c_k < p < 2^64 is about.  So the LIMBS of C ARE the convolution coefficients c_k, exactly,
 * with no schoolbook loop and no digit extraction: one subquadratic GMP multiply plus a linear
 * scan give the true maximum over ALL 2L-1 coefficients (not a sample).  If any c_k were
 * >= 2^64 the carry would corrupt limb k+1 and the maximum would come out >= 2^64, so the
 * check cannot silently pass on a violation -- it either reports the true max or a value
 * >= 2^64, and both are caught below.
 *
 * Cost is O(L log L) time and ~3*8*L bytes of host memory, so it runs for every shape with
 * L <= NTT_MAXCOEFF_LIMBS (default 8M limbs = 64 MB per operand, i.e. up to P=8192/S=5153
 * at L=4.03e6).  Larger shapes print the bound instead and say that it is the counting
 * argument, validated empirically by the smaller shape in the same run.  Always returns true
 * for now -- the caller decides whether to run it at all, and checks the value against p.
 */
static bool max_conv_coeff_gmp(const std::vector<unsigned long long> &hA,
                               const std::vector<unsigned long long> &hB, unsigned long long L,
                               unsigned long long *out_max, unsigned long long *out_k)
{
    *out_max = 0;
    *out_k = 0;
    mpz_t A, B, C;
    mpz_inits(A, B, C, nullptr);
    mpz_import(A, (size_t)L, -1, 8, 0, 0, hA.data());
    mpz_import(B, (size_t)L, -1, 8, 0, 0, hB.data());
    mpz_mul(C, A, B);
    const size_t nc = mpz_size(C);
    std::vector<unsigned long long> limbs(nc + 1, 0);
    size_t cnt = 0;
    mpz_export(limbs.data(), &cnt, -1, 8, 0, 0, C);
    unsigned long long best = 0, bk = 0;
    for (size_t i = 0; i < cnt; ++i) {
        if (limbs[i] > best) { best = limbs[i]; bk = (unsigned long long)i; }
    }
    mpz_clears(A, B, C, nullptr);
    *out_max = best;
    *out_k = bk;
    return true;
}

/* spill-free 64-bit product of (2^bpw - 1)^2 for the printed ratio */
static double digit_sq_double(int bpw)
{
    const long double d = (long double)((1ull << bpw) - 1ull);
    return (double)(d * d);
}

/* --------------------------------------------------------------------------------- */
/* poly mode                                                                          */
/* --------------------------------------------------------------------------------- */
/* ---- debug: compare the device digit array with the EXACT product's digit array ----- */
/* ---- debug: compare the device digit array with the EXACT product's digit array ----- */
/*
 * `pipedump <P> <S> <dev>` (run_poly with dump != 0) prints, for a small shape:
 *   (a) the exact product's DIGIT array, computed on the host with GMP from the very same
 *       packed operands the device was given (digit j = bits [j*bpw, (j+1)*bpw) of A*B);
 *   (b) the device digit array right after the inverse NTT (before the carry stage);
 *   (c) the device digit array after the carry stage.
 * and, for both (b) and (c), the index of the first differing digit, the number of differing
 * digits, and whether the INTEGER the array represents equals A*B at all.  That last
 * distinction is the decisive one: "same integer, different digits" points at the carry stage
 * (a representation problem), while "different integer" points at the transform/pointwise
 * path (a value problem).
 */
/* add a 64-bit value to an mpz.  NOT mpz_add_ui: on Windows `unsigned long` is 32 bits, so
   mpz_add_ui SILENTLY TRUNCATES every digit to its low 32 bits -- which made the first
   version of this dump "prove" that a correct transform output was wrong. */
static void mpz_add_u64(mpz_t acc, unsigned long long v)
{
    if (v == 0) return;
    mpz_t t;
    mpz_init(t);
    mpz_import(t, 1, -1, 8, 0, 0, &v);
    mpz_add(acc, acc, t);
    mpz_clear(t);
}

static void dump_cmp(const char *tag, const std::vector<unsigned long long> &d, const mpz_t C,
                     unsigned long long n, int bpw, int show)
{
    mpz_t x, t;
    mpz_inits(x, t, nullptr);
    mpz_set_ui(x, 0);
    for (unsigned long long j = n; j-- > 0;) {
        mpz_mul_2exp(x, x, (unsigned long)bpw);
        mpz_add_u64(x, d[(size_t)j]);
    }
    const bool value_ok = (mpz_cmp(x, C) == 0);
    /* difference as an integer: x - C (small if only the low digits are off) */
    mpz_sub(t, x, C);
    const long dbl = (mpz_sgn(t) == 0) ? 0 : (long)mpz_sizeinbase(t, 2);
    unsigned long long first = 0, bad = 0;
    mpz_t mask, cj;
    mpz_inits(mask, cj, nullptr);
    mpz_set_ui(mask, 1);
    mpz_mul_2exp(mask, mask, (unsigned long)bpw);
    mpz_sub_ui(mask, mask, 1);
    mpz_t cur;
    mpz_init_set(cur, C);
    for (unsigned long long j = 0; j < n; ++j) {
        mpz_and(cj, cur, mask);
        if (mpz_get_ui(cj) != d[(size_t)j]) {
            if (bad == 0) first = j;
            ++bad;
        }
        mpz_fdiv_q_2exp(cur, cur, (unsigned long)bpw);
    }
    std::printf("  %s: digits_bad=%llu/%llu first_bad_digit=%llu value_eq_exact=%d "
                "|value_diff|_bits=%ld\n", tag, bad, n, first, value_ok ? 1 : 0, dbl);
    std::printf("    exact");
    {
        mpz_t cur2, m2;
        mpz_init_set(cur2, C);
        mpz_init(m2);
        for (unsigned long long j = 0; j < (unsigned long long)show && j < n; ++j) {
            mpz_and(m2, cur2, mask);
            std::printf(" [%llu]=%llu", j, (unsigned long long)mpz_get_ui(m2));
            mpz_fdiv_q_2exp(cur2, cur2, (unsigned long)bpw);
        }
        std::printf("\n");
        mpz_clears(cur2, m2, nullptr);
    }
    std::printf("    %s   ", tag);
    for (unsigned long long j = 0; j < (unsigned long long)show && j < n; ++j)
        std::printf(" [%llu]=%llu", j, d[(size_t)j]);
    std::printf("\n");
    mpz_clears(x, t, mask, cj, cur, nullptr);
}

/* =====================================================================================
 * THE REUSABLE MULTIPLY: ONE implementation, TWO callers.
 *
 * `ntt_poly_mul_host` is this probe's Kronecker / integer-NTT polynomial multiply, extracted
 * so that the CLI's `poly` mode (run_poly, below) and the GPU tree engine
 * (tools/bench/stage2_tree_gpu.cu, M3 slice S1 of docs/DEV_STAGE2_GPU_PLAN.md section 18)
 * call THE SAME CODE.  Nothing is copy-pasted into the tree: the packing convention, the
 * exactness assertions, the fusion plan, the transform, the carry and the slot assembly below
 * are the single implementation of each, and the tree reaches them by including this file
 * (with NTT_POLY_PROBE_NO_MAIN defined) and calling this function.  That is the whole point of
 * the extraction: section 14.8's lesson is that a mirror and an implementation drift apart,
 * and the packing convention in particular (word-aligned slot stride, section 14.10) is
 * exactly where a silent wrong-everything bug lived.
 *
 * CONTRACT (identical to `poly <P> <S> [device]`):
 *   in   wordsA, wordsB   P coefficients of S bits each; coefficient i occupies the W =
 *                         ceil(S/64) words at [i*W, (i+1)*W), little-endian, with every bit
 *                         at or above S zero.  Operands shorter than P must be ZERO-PADDED by
 *                         the caller (a product tree multiplies unbalanced nodes; padding is
 *                         safe because the exactness bound L = P*slot_words only grows with P
 *                         while the true nonzero digit count does not -- see the proof above
 *                         exact_ok_terms).
 *   out  out_slots_u64    2P-1 values: coefficient k of the exact product, projected modulo
 *                         SLOT_MOD by the device slot assembler.  This is the probe's own
 *                         verification path and it is unchanged.
 *        out_exact        optional.  Coefficient k as the EXACT integer, cw =
 *                         ceil(slot_bits/64) words, little-endian, NOT reduced -- the tree
 *                         needs the full value so that it can reduce it modulo its own
 *                         composite N (which is far wider than the slots).  Extracted on the
 *                         host with GMP from the post-carry canonical digit array, and then
 *                         CROSS-CHECKED against out_slots_u64 for every k: two independent
 *                         extractions of the same coefficient, so a mistake in either one is
 *                         fatal here instead of silent downstream.
 *   in   verbose          print the shape / exactness / fusion diagnostics (run_poly: true,
 *                         the tree: false -- it multiplies hundreds of times per F tree).
 *        dump             the `pipedump` digit dumps (run_poly: dump != 0).
 *   ret  0 on success; 2..8 on a refused shape or a failed assertion -- never garbage.
 * ===================================================================================== */

/* what the caller gets back; run_poly prints its result line and its diagnostics from here */
struct NttMulStats {
    unsigned long long P = 0, N = 0, out_slots = 0;      /* coefficients, transform length */
    unsigned long long slot_bits = 0, slot_stride = 0, slot_words = 0, L_terms = 0;
    unsigned long long cw = 0;                           /* words per exact coefficient */
    int S = 0, k = 0, bpw = 0;
    int passes_fwd = 0, passes_total = 0, carry_rounds = 0;
    double mem_mb = 0.0, t_fwd = 0.0, t_inv = 0.0, t_slot = 0.0;
    /* OBJECTIVE 4 (docs/DEV_STAGE2_GPU_PLAN.md section 32): the batched entry point packs its
       operands ON THE HOST, scans them for the max coefficient, and ships them to the device --
       none of which any of the timers above covers, because they were written for the other
       entry point.  Measured separately so the "111 us per call that nobody measured" can be
       attributed instead of guessed. */
    double t_hpack = 0.0, t_scan = 0.0, t_h2d_batch = 0.0;
    /* the extra pass the carry-convergence assert costs (section 34), and its three parts
       (section 41: the reset launch, the kernel, the blocking readback) */
    double t_check = 0.0;
    double t_check_reset = 0.0, t_check_kernel = 0.0, t_check_d2h = 0.0;
    /* where the REST of the device entry point's time goes: the shape plan (choose_cfg, which
       proves the exactness bound) and the device-to-device copy of the two operands into the
       arena's scratch (section 34) */
    double t_plan = 0.0, t_opcopy = 0.0;
    double t_hout = 0.0;
    /* the WHOLE call's wall time (section 27): the tree's per-level floor is whatever is left once
       the parts below are subtracted, so the total has to be reported next to them */
    double t_total_call = 0.0;
    unsigned long long carry_residual = 0, carry_max_bits = 0;
    bool carry_deferred = false;            /* actual path, including arena-refused fallback */
    int fuse_t = 0, fuse_nms = 0, fuse_ms[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    bool exact_valid = false;
    /* slice S4: where the per-CALL FIXED cost goes.  t_fwd/t_inv/t_slot already isolate the
       device passes; these seven split the rest (the "host_side" of the S3 breakdown) so the
       question "is the fixed cost launch overhead or host GMP?" is answered by measurement
       instead of by argument.  Zero unless NTT_HOST_BREAK=1 (the timers are pure clock reads,
       but keeping the default path free of them keeps the probe's timings comparable with the
       ones already recorded in the plan). */
    double t_setup = 0, t_pack = 0, t_maxc = 0, t_h2d = 0, t_carry = 0, t_d2h = 0,
           t_ext = 0, t_xchk = 0;
};

static bool ntt_host_break_on()
{
    static const bool on = (std::getenv("NTT_HOST_BREAK") != nullptr);
    return on;
}
/* NTT_HBS(name, t0): accumulate now_s()-t0 into the local `hb_<name>` of ntt_poly_mul_host.
   Empty when NTT_HOST_BREAK is unset -- the default path is byte-identical to before. */
#define NTT_HBS(name, t0) do { if (ntt_host_break_on()) hb_##name += now_s() - (t0); } while (0)

/* Coefficient k of the exact product, read out of the post-carry canonical digit array.
 *
 * The carry stage's fixed point is the canonical base-2^bpw representation of the exact
 * product A*B of the two packed operands (the probe's own `pipedump` checks exactly that:
 * value_eq_exact=1), so `mpz_import` with nails = 64-bpw reconstructs that integer exactly.
 * Coefficient k of the polynomial product then sits at bits [k*slot_stride, k*slot_stride +
 * slot_bits) of it, because the packer placed coefficient k there and slot_stride >= slot_bits
 * means no two slots overlap.  (Same convention as slot_assemble_kernel, which does the same
 * extraction reduced modulo the 32-bit SLOT_MOD.) */
static bool extract_exact_coeffs(const unsigned long long *digits, unsigned long long n,
                                 int bpw, unsigned long long slot_bits,
                                 unsigned long long slot_stride, unsigned long long out_slots,
                                 std::vector<std::vector<unsigned long long>> *out)
{
    if (bpw <= 0 || bpw >= 64 || slot_bits == 0) return false;
    mpz_t C, t;
    mpz_inits(C, t, nullptr);
    mpz_import(C, (size_t)n, -1, 8, 0, (size_t)(64 - bpw), digits);
    const size_t cw = (size_t)((slot_bits + 63) / 64);
    out->assign((size_t)out_slots, std::vector<unsigned long long>(cw, 0));
    std::vector<unsigned long long> tmp(cw + 1, 0);
    for (unsigned long long k = 0; k < out_slots; ++k) {
        mpz_tdiv_q_2exp(t, C, (mp_bitcnt_t)(k * slot_stride));
        mpz_fdiv_r_2exp(t, t, (mp_bitcnt_t)slot_bits);
        std::fill(tmp.begin(), tmp.end(), 0ull);
        size_t cnt = 0;
        mpz_export(tmp.data(), &cnt, -1, 8, 0, 0, t);
        for (size_t j = 0; j < cw && j < cnt; ++j) (*out)[(size_t)k][j] = tmp[j];
    }
    mpz_clears(C, t, nullptr);
    return true;
}

/* ===================================================================================== *
 *  SLICE S4 -- THE SHAPE, THE PACKING AND THE PASS RUNNER, SHARED BY BOTH ENTRY POINTS
 *
 *  S3 measured the per-CALL fixed cost of this multiply as 195-209 us while the butterfly
 *  arithmetic of a small shape is ~1 us, and 97% of the stage-2 wall clock was inside this
 *  function.  The fixed cost is ~15 CUDA API round trips per call (device properties, two
 *  H2D copies, six kernel launches, a D2H copy, a synchronise) plus the host-side GMP work.
 *  S4 attacks it two ways, and both need the shape work to be separable from the slice work:
 *
 *    * ntt_poly_mul_batch_host() multiplies MANY same-shape slices in ONE launch per pass
 *      (blockIdx.y = slice), so a product tree level costs one launch instead of 2^l of them;
 *    * an optional device-side post-reduction hook (NttReduceHook) lets the caller reduce the
 *      exact coefficients modulo its own modulus on the device, so the D2H copy of the whole
 *      digit array, the host extraction and the host GMP reduction all disappear.
 *
 *  For nbatch == 1 and no hook, the batched entry point runs EXACTLY the code the per-call
 *  entry point runs -- same shape plan, same packing, same pass runner -- so the two cannot
 *  drift; the per-call entry point keeps its own tails (host exact extraction, the two-
 *  extraction cross-check) that the probe's oracle is built on.
 * ===================================================================================== */

/* (P,S) -> the shape, WITHOUT touching the device.  `choose_cfg` is a pure function, so a
   caller that must prepare something per shape (the tree's device reduction) can ask for the
   shape before it calls the multiply.  Returns false if the shape is refused. */
static bool ntt_shape_query(unsigned long long P, int S, unsigned long long *N_out, int *bpw_out,
                            unsigned long long *slot_bits_out, unsigned long long *slot_words_out,
                            unsigned long long *slot_stride_out, unsigned long long *out_slots_out,
                            int force_bpw = 0)
{
    unsigned long long log2P = 1;
    while ((1ull << log2P) < P) ++log2P;
    const unsigned long long slot_bits = 2ull * (unsigned long long)S + log2P;
    const NttCfg cfg = choose_cfg(P * slot_bits, slot_bits, P, force_bpw);
    if (!cfg.ok) return false;
    if (N_out) *N_out = cfg.nwords;
    if (bpw_out) *bpw_out = cfg.bpw;
    if (slot_bits_out) *slot_bits_out = slot_bits;
    if (slot_words_out) *slot_words_out = cfg.slot_words;
    if (slot_stride_out) *slot_stride_out = cfg.slot_words * (unsigned long long)cfg.bpw;
    if (out_slots_out) *out_slots_out = 2 * P - 1;
    return true;
}

struct NttShape {
    int device = 0;
    char dev_name[256] = {0};
    unsigned long long P = 0, N = 0, out_slots = 0, slot_bits = 0, slot_stride = 0,
                       slot_words = 0, L_terms = 0, W = 0, payload_bits = 0;
    int S = 0, k = 0, bpw = 0;
    unsigned long long omega = 0, omega_inv = 0, n_scale = 0;
    double mem_mb = 0, bound_bits = 0, bound_ratio = 0, old_bound_bits = 0, p_bits = 0;
    bool bound_ok = false, old_bound_ok = false;
    int passes_fwd = 0, passes_total = 0, carry_rounds = 0;
};

/* (P,S) -> (N, bpw, k, slot_words, omega, ...) with every assertion the multiply needs.
   Returns 0, or the same non-zero code the per-call multiply has always returned.

   MEMOISED (slice S4).  Every term here is a pure function of (P,S,device) -- choose_cfg, the
   omega powers and the bound bit counts -- and the S4 engine calls this once per batched
   launch, i.e. ~1.8e3 times per curve at the frozen shape.  `cudaGetDeviceProperties` alone is
   an expensive driver round trip on Windows (it is the same call the S3 breakdown charged 36 us
   per CALL to).  The memo returns a copy of the cached plan; the FuseCtx is NOT part of it (the
   arena owns that) and is still resolved per call. */
struct NttShapeCacheEntry {
    unsigned long long P = 0;
    int S = 0, device = -1, force_bpw = 0;
    NttShape sh;
};

static int ntt_shape_plan_uncached(unsigned long long P, int S, int device, NttArena *arena,
                                   FuseCtx &fc, NttShape &sh, int force_bpw);

static int ntt_shape_plan(unsigned long long P, int S, int device, NttArena *arena,
                          FuseCtx &fc, NttShape &sh, int force_bpw = 0)
{
    static std::vector<NttShapeCacheEntry> memo;
    for (const NttShapeCacheEntry &e : memo) {
        if (e.P == P && e.S == S && e.device == device && e.force_bpw == force_bpw) {
            sh = e.sh;
            ntt_arena_fuse(arena, sh.N, sh.k, sh.omega, sh.omega_inv, fc);
            return 0;
        }
    }
    const int rc = ntt_shape_plan_uncached(P, S, device, arena, fc, sh, force_bpw);
    if (rc == 0) {
        NttShapeCacheEntry e;
        e.P = P;
        e.S = S;
        e.device = device;
        e.force_bpw = force_bpw;
        e.sh = sh;
        memo.push_back(e);
    }
    return rc;
}

static int ntt_shape_plan_uncached(unsigned long long P, int S, int device, NttArena *arena,
                                   FuseCtx &fc, NttShape &sh, int force_bpw)
{
    if (P == 0 || S <= 0) {
        std::fprintf(stderr, NTT_PROBE_NAME ": bad shape P=%llu S=%d\n",
                     (unsigned long long)P, S);
        return 2;
    }
    CK(cudaSetDevice(device));
    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, device));
    CK(cudaFree(0));
    std::snprintf(sh.dev_name, sizeof(sh.dev_name), "%s", prop.name);
    sh.device = device;
    sh.P = P;
    sh.S = S;
    unsigned long long log2P = 1;
    while ((1ull << log2P) < P) ++log2P;
    sh.slot_bits = 2ull * (unsigned long long)S + log2P;
    sh.payload_bits = P * sh.slot_bits;

    const NttCfg cfg = choose_cfg(sh.payload_bits, sh.slot_bits, P, force_bpw);
    if (!cfg.ok) {
        std::fprintf(stderr, NTT_PROBE_NAME ": %s\n", cfg.why.c_str());
        return 2;
    }
    sh.N = cfg.nwords;
    sh.bpw = cfg.bpw;
    sh.k = cfg.k;
    /* WORD-ALIGNED slot stride: coefficient i is packed at bit i*slot_stride, and
       slot_stride = ceil(slot_bits/bpw)*bpw keeps every operand digit < 2^bpw. */
    sh.slot_words = cfg.slot_words;
    sh.slot_stride = sh.slot_words * (unsigned long long)sh.bpw;
    sh.out_slots = 2 * P - 1;

    if (sh.slot_bits > sh.slot_stride || sh.slot_stride > (unsigned long long)sh.bpw * sh.N ||
        (2 * P - 1) * sh.slot_words + sh.slot_words > sh.N) {
        std::fprintf(stderr,
                     NTT_PROBE_NAME ": ASSERT FAILED slot_bits=%llu slot_stride=%llu needs "
                     "%llu words of N=%llu\n", (unsigned long long)sh.slot_bits,
                     (unsigned long long)sh.slot_stride,
                     (unsigned long long)((2 * P - 1) * sh.slot_words + sh.slot_words),
                     (unsigned long long)sh.N);
        return 3;
    }
    sh.W = (unsigned long long)(S + 63) / 64;

    /* twiddle base: omega = 7^((p-1)/N).  Checked: omega^N == 1 and omega^(N/2) == p-1
       (i.e. omega really is a primitive N-th root), and N^{-1} mod p really inverts N. */
    const unsigned long long root = 7ull;
    sh.omega = gl_pow_host(root, (GL_P - 1ull) / sh.N);
    sh.omega_inv = gl_pow_host(sh.omega, GL_P - 2ull);            /* inverse root */
    if (gl_pow_host(sh.omega, sh.N) != 1ull || gl_pow_host(sh.omega, sh.N / 2) != GL_P - 1ull) {
        std::fprintf(stderr, NTT_PROBE_NAME ": omega is not a primitive %llu-th root\n",
                     (unsigned long long)sh.N);
        return 3;
    }
    if (gl_mul_host(sh.omega, sh.omega_inv) != 1ull) {
        std::fprintf(stderr, NTT_PROBE_NAME ": omega * omega_inv != 1\n");
        return 3;
    }
    sh.n_scale = gl_pow_host(sh.N % GL_P, GL_P - 2ull);           /* N^-1 mod p */
    if (gl_mul_host(sh.N % GL_P, sh.n_scale) != 1ull) {
        std::fprintf(stderr, NTT_PROBE_NAME ": 1/N failed\n");
        return 3;
    }

    sh.mem_mb = (double)(4 * sh.N * sizeof(unsigned long long) +
                         sh.out_slots * sizeof(unsigned long long)) / 1048576.0;
    if (P > (~0ull) / sh.slot_words) {
        std::fprintf(stderr, NTT_PROBE_NAME ": L = P*slot_words overflows 64 bits\n");
        return 3;
    }
    sh.L_terms = P * sh.slot_words;
    if (!exact_ok_terms(sh.L_terms, sh.bpw)) {
        std::fprintf(stderr, NTT_PROBE_NAME ": EXACTNESS VIOLATED: L=%llu (P=%llu * "
                             "slot_words=%llu) * (2^%d - 1)^2 >= p -- the convolution would "
                             "wrap mod p silently\n", (unsigned long long)sh.L_terms,
                     (unsigned long long)sh.P, (unsigned long long)sh.slot_words, sh.bpw);
        return 6;
    }
    sh.bound_bits = coeff_bound_bits_terms(sh.L_terms, sh.bpw);
    sh.bound_ratio = bound_over_p_terms(sh.L_terms, sh.bpw);
    sh.old_bound_bits = coeff_bound_bits(sh.N, sh.bpw);           /* the old N-based form */
    sh.bound_ok = exact_ok_terms(sh.L_terms, sh.bpw);
    sh.old_bound_ok = exact_ok_nterms(sh.N, sh.bpw);
    sh.p_bits = mpz_log2d_p();

    /* fusion plan (section 12): t stages in the shared-memory tile pass, the rest fused into
       radix-2^M register passes.  PASS COUNT = number of FULL ARRAY PASSES. */
    ntt_arena_fuse(arena, sh.N, sh.k, sh.omega, sh.omega_inv, fc);
    sh.passes_fwd = fc.passes_fwd;
    sh.passes_total = 2 * fc.passes_fwd + fc.passes_fwd + 1;      /* +1 = the carry cone */
    sh.carry_rounds = 2 + (sh.k + 2 * sh.bpw + sh.bpw - 1) / sh.bpw;
    /* NTT_CARRY_ROUNDS raises the height-reduction rounds for diagnostic comparisons.
       Long binary carry chains are resolved exactly, independently of this override. */
    {
        const char *e = std::getenv("NTT_CARRY_ROUNDS");
        if (e && *e) {
            const int v = std::atoi(e);
            if (v > sh.carry_rounds) sh.carry_rounds = v;
        }
    }
    return 0;
}

/* the EXACTNESS bound, re-derived for this shape: L = P*slot_words is the number of NONZERO
   digits per operand, and the criterion is L*(2^bpw-1)^2 < p, compared in exact integer
   arithmetic.  Called by both entry points; never inherited from another shape. */
static int ntt_shape_exactness(const NttShape &sh)
{
    if (sh.P > (~0ull) / sh.slot_words) {
        std::fprintf(stderr, NTT_PROBE_NAME ": L = P*slot_words overflows 64 bits\n");
        return 3;
    }
    if (!exact_ok_terms(sh.L_terms, sh.bpw)) {
        std::fprintf(stderr, NTT_PROBE_NAME ": EXACTNESS VIOLATED: L=%llu (P=%llu * "
                             "slot_words=%llu) * (2^%d - 1)^2 >= p -- the convolution would "
                             "wrap mod p silently\n", (unsigned long long)sh.L_terms,
                     (unsigned long long)sh.P, (unsigned long long)sh.slot_words, sh.bpw);
        return 6;
    }
    return 0;
}

/* pack ONE slice into N bpw-bit digits, and assert that every digit of BOTH operands is
   < 2^bpw (the exactness criterion's derivation needs it; the compact-packing bug was exactly
   a violation of it that the geometric check could not see).  Returns false after printing. */
static bool ntt_pack_operand(const NttShape &sh, const unsigned long long *wordsA,
                             const unsigned long long *wordsB, unsigned long long *hA,
                             unsigned long long *hB, unsigned long long *maxdigit_out)
{
    const unsigned long long P = sh.P, N = sh.N, W = sh.W;
    const int S = sh.S, bpw = sh.bpw;
    const unsigned long long slot_stride = sh.slot_stride;
    std::fill(hA, hA + N, 0ull);
    std::fill(hB, hB + N, 0ull);
    for (unsigned long long i = 0; i < P; ++i) {
        for (int b = 0; b < S; b += bpw) {
            const int nb = (b + bpw <= S) ? bpw : (S - b);
            if (nb <= 0) continue;
            const unsigned long long bit = i * slot_stride + (unsigned long long)b;
            const unsigned long long v = bits_of(wordsA, P * W, i * W * 64ull + (unsigned long long)b, nb);
            const unsigned long long v2 = bits_of(wordsB, P * W, i * W * 64ull + (unsigned long long)b, nb);
            hA[(size_t)(bit / bpw)] |= v << (int)(bit % bpw);
            hB[(size_t)(bit / bpw)] |= v2 << (int)(bit % bpw);
        }
    }
    const unsigned long long lim = 1ull << bpw;
    unsigned long long over = 0, maxd = 0;
    for (unsigned long long j = 0; j < N; ++j) {
        const unsigned long long m = (hA[(size_t)j] > hB[(size_t)j]) ? hA[(size_t)j]
                                                                     : hB[(size_t)j];
        if (m >= lim) ++over;
        if (m > maxd) maxd = m;
    }
    if (maxdigit_out) *maxdigit_out = maxd;
    if (over) {
        std::fprintf(stderr, NTT_PROBE_NAME ": PACKING NOT CANONICAL: %llu of %llu digits "
                             ">= 2^%d (max digit %llu) -> the exactness criterion "
                             "L*(2^bpw-1)^2 < p does not bound the convolution and every "
                             "coefficient would wrap mod p; slot_stride=%llu bpw=%d\n",
                     (unsigned long long)over, (unsigned long long)N, bpw,
                     (unsigned long long)maxd, (unsigned long long)slot_stride, bpw);
        return false;
    }
    return true;
}

/* ---- INDEPENDENT empirical check of the same bound -------------------------------------
   The TRUE maximum convolution coefficient, from the digits that were just packed, with GMP
   -- not a sample and not the bound itself (see max_conv_coeff_gmp).  Returns 0 on success. */
static int ntt_shape_maxcoeff(const NttShape &sh, const std::vector<uint64_t> &hA,
                              const std::vector<uint64_t> &hB, unsigned long long *maxcoeff,
                              unsigned long long *maxcoeff_k, bool *ran,
                              unsigned long long *limbs)
{
    /* the empirical maximum is a check on the BOUND DERIVATION (are the digits really < 2^bpw,
       is the convolution really below p), so it is run once per SHAPE: at the frozen shape the
       S4 engine launches ~1.8e3 batches and running a GMP multiply in every one of them is pure
       cost.  The counting bound itself is still asserted on EVERY call, and the canonical-digit
       property is checked on every call by the reduction kernel. */
    static std::vector<unsigned long long> done;
    const unsigned long long maxcoeff_limbs = fuse_env_ull("NTT_MAXCOEFF_LIMBS", 8000000ull);
    if (limbs) *limbs = maxcoeff_limbs;
    *maxcoeff = 0;
    *maxcoeff_k = 0;
    for (unsigned long long key : done)
        if (key == sh.L_terms * 4096ull + (unsigned long long)sh.bpw) { *ran = false; return 0; }
    *ran = (maxcoeff_limbs > 0 && sh.L_terms <= maxcoeff_limbs);
    if (!*ran) return 0;
    done.push_back(sh.L_terms * 4096ull + (unsigned long long)sh.bpw);
    max_conv_coeff_gmp(hA, hB, sh.L_terms, maxcoeff, maxcoeff_k);
    if (*maxcoeff >= GL_P) {
        std::fprintf(stderr, NTT_PROBE_NAME ": EXACTNESS VIOLATED (measured): the true "
                             "maximum convolution coefficient is %llu >= p = %llu "
                             "(k=%llu, L=%llu, bpw=%d) -> the NTT would wrap silently\n",
                     (unsigned long long)*maxcoeff, (unsigned long long)GL_P,
                     (unsigned long long)*maxcoeff_k, (unsigned long long)sh.L_terms, sh.bpw);
        return 7;
    }
    return 0;
}

static void ntt_shape_print(const NttShape &sh, const FuseCtx &fc, bool verbose,
                            unsigned long long maxcoeff, unsigned long long maxcoeff_k,
                            bool maxcoeff_ran, unsigned long long maxcoeff_limbs)
{
    if (!verbose) return;
    std::printf(NTT_PROBE_NAME ": mode=poly device=%d (%s) P=%llu S=%d slot_bits=%llu "
                "bpw=%d nwords=%llu (log2=%d)\n",
                sh.device, sh.dev_name, (unsigned long long)sh.P, sh.S,
                (unsigned long long)sh.slot_bits, sh.bpw, (unsigned long long)sh.N, sh.k);
    std::printf("  exactness: L*(2^bpw-1)^2 = 2^%.3f < p = 2^%.3f -> %s ; L=P*slot_words="
                "%llu*%llu=%llu terms (N=%llu would be the old count) ; bound/p=%.6f "
                "(headroom %.4fx) ; the old N-form N*(2^bpw)^2 = 2^%.3f would be %s ; "
                "payload=%llu bits capacity=%llu bits (%.2fx) ; mem=%.0f MB\n",
                sh.bound_bits, sh.p_bits, sh.bound_ok ? "OK" : "VIOLATED",
                (unsigned long long)sh.P, (unsigned long long)sh.slot_words,
                (unsigned long long)sh.L_terms, (unsigned long long)sh.N, sh.bound_ratio,
                (sh.bound_ratio > 0) ? 1.0 / sh.bound_ratio : 0.0, sh.old_bound_bits,
                sh.old_bound_ok ? "OK" : "VIOLATED (that form is what the L bound replaces: "
                                          "it is the reason bpw and N were smaller/larger "
                                          "before)",
                (unsigned long long)sh.payload_bits, (unsigned long long)sh.bpw * sh.N,
                (double)(sh.bpw * sh.N) / (double)sh.payload_bits, sh.mem_mb);
    if (maxcoeff_ran) {
        const double dsq = digit_sq_double(sh.bpw);
        std::printf("  maxcoeff: EXACT GMP maximum over all %llu convolution coefficients of "
                    "this shape (the full sum, not a sample) = %llu at k=%llu = %.4f x "
                    "(2^bpw-1)^2, vs the proven bound L=%llu (max is %.4f of the bound) ; "
                    "max/p=%.4f -> < p OK\n",
                    (unsigned long long)(2 * (unsigned long long)sh.L_terms - 1),
                    (unsigned long long)maxcoeff, (unsigned long long)maxcoeff_k,
                    (double)maxcoeff / dsq, (unsigned long long)sh.L_terms,
                    (double)maxcoeff / (dsq * (double)sh.L_terms),
                    sh.bound_ratio * ((double)maxcoeff / (dsq * (double)sh.L_terms)));
    } else {
        std::printf("  maxcoeff: NOT recomputed for this shape (L=%llu terms > "
                    "NTT_MAXCOEFF_LIMBS=%llu, the GMP product would need %.0f MB of host "
                    "memory) -- the coefficient bound above is the COUNTING bound "
                    "c_k <= L*(2^bpw-1)^2 (proof above exact_ok_terms), which the smaller "
                    "shapes of this run verify exactly against GMP\n",
                    (unsigned long long)sh.L_terms, (unsigned long long)maxcoeff_limbs,
                    3.0 * 8.0 * (double)sh.L_terms / 1048576.0);
    }
    std::printf("  fusion: PASSES=%d total (each pass = one full read AND one full write of the "
                "N-word array) = %d forward + %d forward + %d inverse (+%d carry) ; "
                "transform_stages=%d ; plan: tile t=%d stages, then outer",
                sh.passes_total, sh.passes_fwd, sh.passes_fwd, sh.passes_fwd, 1, sh.k, fc.t);
    for (int p = 0; p < fc.nms; ++p) std::printf(" radix-%d", 1 << fc.ms[p]);
    std::printf(" ; slot assembly reads %.2f of a pass\n",
                (double)(sh.out_slots * sh.slot_words) / (double)sh.N);
    std::printf("  packing: word-aligned slot stride=%llu bits = %llu digits of %d bits "
                "(coefficients every %llu bits), max operand digit < 2^%d asserted\n",
                (unsigned long long)sh.slot_stride, (unsigned long long)sh.slot_words, sh.bpw,
                (unsigned long long)sh.slot_stride, sh.bpw);
}

/* what one run of the device passes produced */
struct NttPassResult {
    double t_fwd = 0.0, t_inv = 0.0, t_slot = 0.0;
    /* the carry-convergence assert below is a WHOLE EXTRA PASS over the digit array (a memset, a
       kernel over N*nbatch digits and a device-to-host copy) and it was untimed, which is why
       more than half of ntt_seconds had no owner (docs/DEV_STAGE2_GPU_PLAN.md section 34). */
    double t_check = 0.0;
    /* ... and THAT total is itself split, because its three parts have three different fixes
       (section 41): the reset memset launch, the kernel itself, and the blocking 16-byte D2H
       that drains the whole pipeline before the host can look at the two counters. */
    double t_check_reset = 0.0, t_check_kernel = 0.0, t_check_d2h = 0.0;
    /* the slot assembly + its D2H: skipped entirely when the caller has a reduction hook, which
       never reads them (section 35) */
    double t_hout = 0.0;
    unsigned long long *digits = nullptr;      /* the buffer holding the canonical digits */
    std::vector<unsigned long long> hOut;      /* nbatch * out_slots slot projections */
    std::vector<unsigned long long> hRes;      /* 2 * nbatch carry diagnostics */
    /* DEFERRED IN-STREAM MARKS (docs section 41).  Reading an event's elapsed time is only
       possible once the event has completed, so a call that has no other reason to drain the
       pipeline must either block here (which is exactly the cost being removed) or hand the
       marks back untouched.  The batched device entry point does the latter: it owns the one
       blocking copy at the end of its chunk, and reads these five marks right afterwards.
       ev[0..1] = forward, ev[1..2] = inverse, ev[3..4] = carry-residual kernel. */
    cudaEvent_t dev_ev[5] = {nullptr, nullptr, nullptr, nullptr, nullptr};
    /* the carry-residual counters, read in the same deferred step (empty until then) */
    bool res_deferred = false;
};

/* forward, pointwise product, inverse, carry, slot assembly and the carry-convergence
   diagnostics, over `nbatch` consecutive slices of N words in ONE launch per pass.
   The kernels are the very same ones the single-slice path always used (blockIdx.y = slice,
   stride = N); with nbatch == 1 this is bit-identical to the original code. */
static NttPassResult ntt_run_passes(const NttShape &sh, const FuseCtx &fc,
                                    unsigned long long *dA, unsigned long long *dB,
                                    unsigned long long *dQ, unsigned long long *dOut,
                                    unsigned long long *dRes, unsigned long long nbatch,
                                    int dump, std::vector<unsigned long long> *hfa,
                                    std::vector<unsigned long long> *hfb,
                                    std::vector<unsigned long long> *hPre,
                                    std::vector<unsigned long long> *hPost,
                                    bool need_hout = true, bool defer_res = false)
{
    NttPassResult r;
    const unsigned long long N = sh.N, out_slots = sh.out_slots;
    const unsigned int threads = 256;
    const unsigned int blocks = (unsigned int)((N + threads - 1) / threads);
    const unsigned long long inv_n = ((~0ull) / N) + 1;   /* 2^64 / N: exact exponent mod */

    /* ---- forward, pointwise product, inverse ----
       FORWARD = DIF (Gentleman-Sande), halves DESCENDING: natural in, bit-reversed spectrum
       out.  INVERSE = DIT (Cooley-Tukey), halves ASCENDING, same index decomposition, root
       omega^{-1}: consumes that bit-reversed spectrum and yields natural order.  The pointwise
       product sits between them and needs NO permutation and NO twiddle factor (exp_scale 0).
       This pair is verified by `nttcheck` (roundtrip bit-exact and forward == direct DFT at
       every length 2..4096) -- there is no bit-reversal kernel anywhere in the data path.
       The FUSED implementation (section 12 above) runs the very same stages in the very same
       order; `fusecheck` compares it element by element against this per-stage code. */
    /* ---- THE TIMERS MUST NOT SERIALISE THE PIPELINE (objective 4, section 39) ---------------
       These two phases used to be followed by `cudaDeviceSynchronize()` purely so that t_fwd and
       t_inv could be measured.  The GPU-Z sensor log says what that costs: over a B2=1e11 run the
       GPU was at >=80% load only 33% of the samples and BELOW 20% for 26% of them (mean 58.7%,
       mean board power 19.8 W of a ~60 W budget) -- i.e. the engine leaves the device idle about
       two fifths of the time, and one cause is exactly this: the host drained the pipeline twice
       per multiply for no reason but the clock.
       The dependency the sync provided is already provided by the STREAM (the inverse reads what
       the forward wrote, the carry reads what the inverse wrote), so the timers move to CUDA
       events: recorded in-stream, read once at the end of the call, never blocking.  The numbers
       stay honest (events measure GPU time, not host time) and the host can queue the next
       phases while the device is still executing. */
    cudaEvent_t ev_f0, ev_f1, ev_i1;
    CK(cudaEventCreate(&ev_f0));
    CK(cudaEventCreate(&ev_f1));
    CK(cudaEventCreate(&ev_i1));
    CK(cudaEventRecord(ev_f0));
    ntt_forward_fused(dA, fc, sh.omega, nbatch);
    ntt_forward_fused(dB, fc, sh.omega, nbatch);
    if (dump && nbatch == 1 && hfa && hfb) {
        hfa->assign(N, 0);
        hfb->assign(N, 0);
        CK(cudaMemcpy(hfa->data(), dA, N * sizeof(unsigned long long),
                      cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hfb->data(), dB, N * sizeof(unsigned long long),
                      cudaMemcpyDeviceToHost));
    }
    /* the pointwise product AND the 1/N scale are elementwise, so they ride along inside the
       inverse's tile pass (which touches the array once anyway) -- two whole passes saved */
    CK(cudaEventRecord(ev_f1));

    ntt_inverse_fused(dA, dB, fc, sh.omega_inv, sh.n_scale, nbatch);
    CK(cudaEventRecord(ev_i1));

    if (dump && nbatch == 1 && hPre) {
        hPre->assign(N, 0);
        CK(cudaMemcpy(hPre->data(), dA, N * sizeof(unsigned long long),
                      cudaMemcpyDeviceToHost));
    }

    /* ---- carry stage: reduce every digit back to bpw bits and ripple the quotient into the
       next digit.  ONE PASS: the cone of the same recurrence is unfolded into registers
       (carry_cone_kernel), followed inside that kernel by exact binary propagation.
       If the round count exceeds the register cone, height reduction uses the old kernels
       and a final cone kernel resolves the remaining binary carry. */
    const double t2 = now_s();
    const int carry_rounds = sh.carry_rounds;
    unsigned long long *dDig = dA;
    if (carry_rounds <= FUSE_MAX_CARRY_ROUNDS) {
        const dim3 gr(blocks, (unsigned int)nbatch);
        switch (carry_rounds) {
            case 1: carry_cone_kernel<1><<<gr, threads>>>(dA, dQ, N, sh.bpw, N); break;
            case 2: carry_cone_kernel<2><<<gr, threads>>>(dA, dQ, N, sh.bpw, N); break;
            case 3: carry_cone_kernel<3><<<gr, threads>>>(dA, dQ, N, sh.bpw, N); break;
            case 4: carry_cone_kernel<4><<<gr, threads>>>(dA, dQ, N, sh.bpw, N); break;
            case 5: carry_cone_kernel<5><<<gr, threads>>>(dA, dQ, N, sh.bpw, N); break;
            case 6: carry_cone_kernel<6><<<gr, threads>>>(dA, dQ, N, sh.bpw, N); break;
            case 7: carry_cone_kernel<7><<<gr, threads>>>(dA, dQ, N, sh.bpw, N); break;
            case 8: carry_cone_kernel<8><<<gr, threads>>>(dA, dQ, N, sh.bpw, N); break;
            case 9: carry_cone_kernel<9><<<gr, threads>>>(dA, dQ, N, sh.bpw, N); break;
            default: carry_cone_kernel<10><<<gr, threads>>>(dA, dQ, N, sh.bpw, N); break;
        }
        CK(cudaGetLastError());
        dDig = dQ;
    } else {
        for (unsigned long long s = 0; s < nbatch; ++s) {
            for (int rr = 0; rr < carry_rounds; ++rr) {
                carry_extract_kernel<<<blocks, threads>>>(dA + s * N, dQ + s * N, N, sh.bpw);
                CK(cudaGetLastError());
                carry_add_kernel<<<blocks, threads>>>(dA + s * N, dQ + s * N, N);
                CK(cudaGetLastError());
            }
        }
        const dim3 gr(blocks, (unsigned int)nbatch);
        carry_cone_kernel<1><<<gr, threads>>>(dA, dQ, N, sh.bpw, N);
        CK(cudaGetLastError());
        dDig = dQ;
    }
    r.digits = dDig;
    if (dump && nbatch == 1 && hPost) {
        hPost->assign(N, 0);
        CK(cudaMemcpy(hPost->data(), dDig, N * sizeof(unsigned long long),
                      cudaMemcpyDeviceToHost));
    }
    r.t_slot = now_s() - t2;
    /* THE SLOT PROJECTION IS ONLY FOR A CALLER WITHOUT A HOOK (objective 4, section 35): the
       caller that installed one reduces the coefficients on the device out of `digits` itself and
       never looks at `dOut`/`hOut`, so the assembly kernel and the D2H of out_slots*nbatch words
       are pure waste for it.  Measured and switchable (NTT_S4_KEEPHOUT=1 restores the old
       behaviour) rather than assumed. */
    const double t4 = now_s();
    if (need_hout) {
        const unsigned int blSlots = (unsigned int)((out_slots + threads - 1) / threads);
        {
            const dim3 gr(blSlots, (unsigned int)nbatch);
            slot_assemble_kernel<<<gr, threads>>>(dDig, out_slots, sh.slot_bits, sh.slot_stride,
                                                  sh.bpw, dOut, N);
        }
        CK(cudaGetLastError());
        r.hOut.assign((size_t)(out_slots * nbatch), 0);
        CK(cudaMemcpy(r.hOut.data(), dOut, r.hOut.size() * sizeof(unsigned long long),
                      cudaMemcpyDeviceToHost));
    } else {
        r.hOut.clear();
    }
    r.t_hout = now_s() - t4;

    /* Carry convergence assert (UNTIMED until now, and always run -- a carry that stops early is
       silent: the digits are simply wrong above some index).  ONE PASS over the whole digit array
       plus a D2H per call, which is why it is now measured. */
    const double t3 = now_s();
    /* the kernel's OWN GPU time, from events recorded in-stream around it.  With defer_res the
       marks are handed back (r.dev_ev[3..4]) and read by the caller after its own drain. */
    cudaEvent_t ev_c0, ev_c1;
    CK(cudaEventCreate(&ev_c0));
    CK(cudaEventCreate(&ev_c1));
    if (!defer_res) {
        r.hRes.assign((size_t)(2 * nbatch), 0);
        CK(cudaMemset(dRes, 0, 2 * nbatch * sizeof(unsigned long long)));
    }
    const double t3a = now_s();
    r.t_check_reset = t3a - t3;
    CK(cudaEventRecord(ev_c0));
    {
        const dim3 gr(blocks, (unsigned int)nbatch);
        carry_residual_kernel<<<gr, threads>>>(dDig, N, sh.bpw, dRes, N);
    }
    CK(cudaGetLastError());
    CK(cudaEventRecord(ev_c1));
    if (defer_res) {
        /* NOTHING HERE DRAINS THE PIPELINE (section 41).  `dRes` holds 2 words per slice and the
           kernel indexes them by slice, so the counters of every call in a chunk simply
           ACCUMULATE in place: one memset before the caller's chunk loop and one readback after
           it reproduce exactly the per-call values, at one drain per chunk instead of one per
           call.  Measured at the production shape: the whole carry check was 60.6 s (24% of
           ntt_seconds) of which the kernel itself is 0.2 us per call -- the rest was the 16-byte
           blocking readback draining the pipeline. */
        r.res_deferred = true;
        r.dev_ev[3] = ev_c0;
        r.dev_ev[4] = ev_c1;
        r.t_check = now_s() - t3;
        r.dev_ev[0] = ev_f0;
        r.dev_ev[1] = ev_f1;
        r.dev_ev[2] = ev_i1;
        return r;
    }
    const double tc0 = now_s();
    CK(cudaMemcpy(r.hRes.data(), dRes, r.hRes.size() * sizeof(unsigned long long),
                  cudaMemcpyDeviceToHost));
    r.t_check_d2h = now_s() - tc0;
    r.t_check_kernel = 0.0;
    {
        float ms = 0.0f;
        if (cudaEventElapsedTime(&ms, ev_c0, ev_c1) == cudaSuccess)
            r.t_check_kernel = (double)ms * 1e-3;       /* the GPU's own time */
    }
    cudaEventDestroy(ev_c0);
    cudaEventDestroy(ev_c1);
    r.t_check = now_s() - t3;
    /* the in-stream event timers, read once the stream has drained (the copies above guarantee
       that) -- see the note at ev_f0 */
    {
        float ms = 0.0f;
        CK(cudaEventElapsedTime(&ms, ev_f0, ev_f1));
        r.t_fwd = (double)ms * 1e-3;
        CK(cudaEventElapsedTime(&ms, ev_f1, ev_i1));
        r.t_inv = (double)ms * 1e-3;
        cudaEventDestroy(ev_f0);
        cudaEventDestroy(ev_f1);
        cudaEventDestroy(ev_i1);
    }
    return r;
}

/* Read (and release) the DEFERRED in-stream marks of one call (section 41).  Only valid once
   the stream has been drained past them, which the caller's own blocking copy guarantees. */
static void ntt_pass_read_marks(NttPassResult &r, double *t_fwd, double *t_inv, double *t_chk)
{
    float ms = 0.0f;
    if (r.dev_ev[0] && r.dev_ev[1] &&
        cudaEventElapsedTime(&ms, r.dev_ev[0], r.dev_ev[1]) == cudaSuccess)
        *t_fwd += (double)ms * 1e-3;
    if (r.dev_ev[1] && r.dev_ev[2] &&
        cudaEventElapsedTime(&ms, r.dev_ev[1], r.dev_ev[2]) == cudaSuccess)
        *t_inv += (double)ms * 1e-3;
    if (r.dev_ev[3] && r.dev_ev[4] &&
        cudaEventElapsedTime(&ms, r.dev_ev[3], r.dev_ev[4]) == cudaSuccess)
        *t_chk += (double)ms * 1e-3;
    for (int i = 0; i < 5; ++i)
        if (r.dev_ev[i]) { cudaEventDestroy(r.dev_ev[i]); r.dev_ev[i] = nullptr; }
}

/* ---- the optional device-side post-reduction hook (slice S4) ---------------------------
   The per-call multiply hands the caller EXACT integer coefficients on the host and leaves
   the mod-N reduction to GMP.  At the real shape (S = 5261, P ~ 9e4) that is one mpz_mod per
   product coefficient -- 1.8e5 of them per multiply, three per outer loop -- and it also
   forces the WHOLE canonical digit array (N words) back to the host on every call.
   A caller that has its own verified device arithmetic can instead reduce the coefficients
   ON THE DEVICE, straight out of the digit buffer the carry left behind, with no host round
   trip at all.  It supplies this hook; the multiply knows nothing about the modulus. */
struct NttReduceHook {
    void *ctx = nullptr;
    /* digits: nbatch*N u64 of bpw-bit digits, laid out slice after slice.  Coefficient k of
       slice s occupies exactly the slot_words digits at [s*N + k*slot_words, +slot_words)
       (the packing stride is word-aligned, so a coefficient never straddles a slot boundary),
       and its value is < 2^slot_bits.  Write nbatch*out_slots coefficients of `w` u64 words
       each into `out`, slice-major.  Runs on the device, one launch (or a few). */
    void (*run)(void *ctx, const unsigned long long *digits, unsigned long long n, int bpw,
                unsigned long long slot_words, unsigned long long slot_bits,
                unsigned long long out_slots, unsigned long long nbatch,
                unsigned long long *out, unsigned long long w) = nullptr;
    unsigned long long *out = nullptr;     /* device buffer: nbatch*out_slots*w words */
    unsigned long long w = 0;              /* u64 words per reduced coefficient */
    /* how many (slice, coefficient) pairs to verify against the host on EVERY batch: a full
       check is cheap for small shapes, and a sample is the honest option for big ones.  0 =
       none (the caller then says so in its own report). */
    unsigned long long sample = 0;
};

/* Populate the engine's FINAL scratch after shape planning and buffer lookup. The callback
   queues writes on the same default stream as the passes, including ALL padding digits.
   It must not allocate/evict arena entries or retain these pointers after the call. Works
   with owned fallback scratch too; the caller never needs to cache an arena pointer. */
struct NttInputHook {
    void *ctx = nullptr;
    int (*run)(void *ctx, const NttShape &shape, unsigned long long nbatch,
               unsigned long long *dA, unsigned long long *dB) = nullptr;
};

/* the batched multiply.  Same shape plan, same packing, same passes as ntt_poly_mul_host --
   the differences are (a) nbatch slices per launch, (b) an optional device post-reduction,
   (c) no host exact extraction (the hook replaces it).  Returns 0 or a refusal code. */
int ntt_poly_mul_batch_host(unsigned long long P, int S, int device, unsigned long long nbatch,
                            const unsigned long long *wordsA, const unsigned long long *wordsB,
                            std::vector<unsigned long long> *out_slots_u64,
                            NttMulStats *st, NttArena *arena,
                            const NttReduceHook *hook,
                            unsigned long long **digits_out = nullptr)
{
    if (nbatch == 0) {
        std::fprintf(stderr, NTT_PROBE_NAME ": batch of zero slices\n");
        return 2;
    }
    NttShape sh;
    FuseCtx fc;
    FuseCallGuard fuse_guard(fc);
    {
        const int rc = ntt_shape_plan(P, S, device, arena, fc, sh);
        if (rc) return rc;
    }
    sh.L_terms = sh.P * sh.slot_words;
    {
        const int rc = ntt_shape_exactness(sh);
        if (rc) return rc;
    }
    const unsigned long long N = sh.N, out_slots = sh.out_slots, W = sh.W;
    const size_t slice_in = (size_t)P * W;               /* u64 words per operand slice */
    std::vector<uint64_t> hA((size_t)(N * nbatch), 0), hB((size_t)(N * nbatch), 0);
    unsigned long long maxdigit = 0;
    /* with a reduction hook the caller never reads the slot projections, so the assembly kernel
       and its D2H are skipped (NTT_S4_KEEPHOUT=1 restores the old behaviour for the A/B) */
    const bool want_hout = (hook == nullptr || hook->run == nullptr) ? true : [] {
        const char *e = std::getenv("NTT_S4_KEEPHOUT");
        return e && *e && std::atoi(e) != 0;
    }();
    const double thp0 = now_s();
    for (unsigned long long s = 0; s < nbatch; ++s) {
        if (!ntt_pack_operand(sh, wordsA + s * slice_in, wordsB + s * slice_in,
                              hA.data() + s * N, hB.data() + s * N, &maxdigit))
            return 5;
    }
    const double t_hpack = now_s() - thp0;
    unsigned long long maxcoeff = 0, maxcoeff_k = 0, maxcoeff_limbs = 0;
    bool maxcoeff_ran = false;
    const double tsc0 = now_s();
    {
        const int rc = ntt_shape_maxcoeff(sh, hA, hB, &maxcoeff, &maxcoeff_k, &maxcoeff_ran,
                                          &maxcoeff_limbs);
        if (rc) return rc;
    }
    const double t_scan = now_s() - tsc0;
    /* Borrowed digits use the original keyed-cache lifetime, never shared scratch. */
    NttArena::BufEntry *ab = ntt_arena_bufs(arena, N, out_slots, nbatch, digits_out == nullptr);
    if (!ab && digits_out) {
        *digits_out=nullptr;

        return 3;                         /* an owned fallback would return a freed pointer */
    }
    unsigned long long *dA = nullptr, *dB = nullptr, *dQ = nullptr, *dOut = nullptr,
                       *dRes = nullptr;
    bool own = false;
    if (ab) {
        dA = ab->dA; dB = ab->dB; dQ = ab->dQ; dOut = ab->dOut; dRes = ab->dRes;
    } else {
        own = true;
        CK(cudaMalloc(&dA, (size_t)(N * nbatch) * sizeof(unsigned long long)));
        CK(cudaMalloc(&dB, (size_t)(N * nbatch) * sizeof(unsigned long long)));
        CK(cudaMalloc(&dQ, (size_t)(N * nbatch) * sizeof(unsigned long long)));
        CK(cudaMalloc(&dOut, (size_t)(out_slots * nbatch) * sizeof(unsigned long long)));
        CK(cudaMalloc(&dRes, 2 * (size_t)nbatch * sizeof(unsigned long long)));
    }
    const double h2d0 = now_s();
    CK(cudaMemcpy(dA, hA.data(), hA.size() * sizeof(unsigned long long),
                  cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hB.data(), hB.size() * sizeof(unsigned long long),
                  cudaMemcpyHostToDevice));
    const double t_h2d_batch = now_s() - h2d0;
    /* nbatch is a gridDim.y, and 65535 is the whole range of gridDim.y: a caller that asks for
       more is SPLIT here (chunk by chunk, the hook included) rather than silently truncated. */
    const unsigned long long max_y = 65535;
    const unsigned long long nchunk = (nbatch + max_y - 1) / max_y;
    NttPassResult r;
    r.hOut.assign((size_t)(out_slots * nbatch), 0);
    r.hRes.assign((size_t)(2 * nbatch), 0);
    for (unsigned long long ci = 0; ci < nchunk; ++ci) {
        const unsigned long long s0 = ci * max_y;
        const unsigned long long m = ((nbatch - s0) < max_y) ? (nbatch - s0) : max_y;
        NttPassResult rr = ntt_run_passes(sh, fc, dA + s0 * N, dB + s0 * N, dQ + s0 * N,
                                          dOut + s0 * out_slots, dRes + 2 * s0, m, 0,
                                          nullptr, nullptr, nullptr, nullptr, want_hout);
        if (want_hout)
            std::copy(rr.hOut.begin(), rr.hOut.end(), r.hOut.begin() + (long)(s0 * out_slots));
        std::copy(rr.hRes.begin(), rr.hRes.end(), r.hRes.begin() + (long)(2 * s0));
        r.t_fwd += rr.t_fwd; r.t_inv += rr.t_inv; r.t_slot += rr.t_slot;
        r.t_check += rr.t_check;
        r.t_check_reset += rr.t_check_reset;
        r.t_check_kernel += rr.t_check_kernel;
        r.t_check_d2h += rr.t_check_d2h;
        if (nchunk == 1) r.digits = rr.digits;
        if (hook && hook->run && hook->out && hook->w) {
            hook->run(hook->ctx, rr.digits, N, sh.bpw, sh.slot_words, sh.slot_bits, out_slots,
                      m, hook->out + (size_t)(s0 * out_slots) * hook->w, hook->w);
            CK(cudaGetLastError());
            CK(cudaDeviceSynchronize());
        }
    }
    if (st) {
        st->P = P; st->S = S; st->N = N; st->k = sh.k; st->bpw = sh.bpw;
        st->slot_bits = sh.slot_bits; st->slot_stride = sh.slot_stride;
        st->slot_words = sh.slot_words; st->out_slots = out_slots; st->L_terms = sh.L_terms;
        st->cw = (unsigned long long)((sh.slot_bits + 63) / 64);
        st->passes_fwd = sh.passes_fwd; st->passes_total = sh.passes_total;
        st->carry_rounds = sh.carry_rounds;
        st->carry_residual = 0; st->carry_max_bits = 0;
        for (unsigned long long s = 0; s < nbatch; ++s) {
            st->carry_residual += r.hRes[(size_t)(2 * s)];
            if (r.hRes[(size_t)(2 * s + 1)] > st->carry_max_bits)
                st->carry_max_bits = r.hRes[(size_t)(2 * s + 1)];
        }
        st->mem_mb = sh.mem_mb * (double)nbatch;
        st->t_fwd = r.t_fwd; st->t_inv = r.t_inv; st->t_slot = r.t_slot;
        st->t_check += r.t_check;
        st->t_check_reset += r.t_check_reset;
        st->t_check_kernel += r.t_check_kernel;
        st->t_check_d2h += r.t_check_d2h;
        st->t_hpack += t_hpack;
        st->t_scan += t_scan;
        st->t_h2d_batch += t_h2d_batch;
        st->fuse_t = fc.t; st->fuse_nms = fc.nms;
        for (int q = 0; q < fc.nms && q < 8; ++q) st->fuse_ms[q] = fc.ms[q];
        st->exact_valid = true;
    }
    unsigned long long bad_res = 0;
    for (unsigned long long s = 0; s < nbatch; ++s)
        if (r.hRes[(size_t)(2 * s)] != 0) ++bad_res;
    if (bad_res) {
        std::fprintf(stderr, NTT_PROBE_NAME ": CARRY DID NOT CONVERGE in %llu of %llu slices "
                             "after %d rounds\n", bad_res, nbatch, sh.carry_rounds);
        if (own) { cudaFree(dA); cudaFree(dB); cudaFree(dQ); cudaFree(dOut); cudaFree(dRes); }
        return 4;
    }
    if (out_slots_u64) *out_slots_u64 = r.hOut;
    if (digits_out) {
        if (nchunk != 1) {
            std::fprintf(stderr, NTT_PROBE_NAME ": the digit buffer of a split batch is not "
                                 "one contiguous array\n");
            if (own) { cudaFree(dA); cudaFree(dB); cudaFree(dQ); cudaFree(dOut); cudaFree(dRes); }
            return 3;
        }
        *digits_out = r.digits;
    }
    int rc = 0;

    if (own) { cudaFree(dA); cudaFree(dB); cudaFree(dQ); cudaFree(dOut); cudaFree(dRes); }
    return rc;
}

/* ---- the DEVICE-to-DEVICE batched multiply (M3 slice S5) ------------------------------
   The tree's descent keeps every operand on the device, so the host packing of
   ntt_poly_mul_batch_host (one host loop over P*W words per slice, plus a full H2D copy) is
   pure cost there.  This entry point takes the operands ALREADY PACKED as bpw-bit digits --
   the caller owns its layout and packs on the device -- and runs the very same shape plan,
   the very same passes, the very same carry and the very same exactness assertions.  It only
   writes the reduced coefficients where the hook says, or the raw digit buffer.
   `dAin`/`dBin` are nbatch*N digits each and copied into engine scratch. Alternatively,
   `input` populates that scratch after planning/lookup; then both input pointers may be null.

   `force_bpw` (default 0 = auto) is passed straight through to the shape plan.  The caller packs
   the operands itself, so when it needs slot_stride == slot_bits (one reduction window per
   coefficient, which is what the S5 reducer asserts) it must make the SAME bpw choice the
   multiply makes; see choose_cfg rule (3).  When the forced bpw cannot meet
   L*(2^bpw-1)^2 < p or the capacity test, the plan fails and this returns non-zero. */
int ntt_poly_mul_batch_dev(unsigned long long P, int S, int device, unsigned long long nbatch,
                           const unsigned long long *dAin, const unsigned long long *dBin,
                           NttMulStats *st, NttArena *arena, const NttReduceHook *hook,
                           unsigned long long **digits_out = nullptr, int force_bpw = 0,
                           bool defer_carry = false, const NttInputHook *input = nullptr)
{
    if (nbatch == 0) {
        std::fprintf(stderr, NTT_PROBE_NAME ": batch of zero slices\n");
        return 2;
    }
    NttShape sh;
    FuseCtx fc;
    FuseCallGuard fuse_guard(fc);
    const double tplan0 = now_s();
    {
        const int rc = ntt_shape_plan(P, S, device, arena, fc, sh, force_bpw);
        if (rc) return rc;
    }
    const double t_plan = now_s() - tplan0;
    sh.L_terms = sh.P * sh.slot_words;
    {
        const int rc = ntt_shape_exactness(sh);
        if (rc) return rc;
    }
    const unsigned long long N = sh.N, out_slots = sh.out_slots, W = sh.W;
    std::vector<unsigned long long> alias_a, alias_b;
    auto preserve=[&](const unsigned long long *p, std::vector<unsigned long long> &saved) {
        size_t remaining=0;
        if (!arena || !arena->input_span(p,remaining)) return true;
        if ((size_t)(N*nbatch)>remaining) return false;
        saved.resize((size_t)(N*nbatch));
        CK(cudaMemcpy(saved.data(),p,saved.size()*8,cudaMemcpyDeviceToHost));
        ++arena->alias_snapshots; arena->alias_snapshot_bytes+=saved.size()*8;
        return true;
    };
    if (!input && (!preserve(dAin,alias_a) || !preserve(dBin,alias_b))) {
        std::fprintf(stderr, NTT_PROBE_NAME ": device input exceeds its arena allocation\n");

        return 3;
    }
    NttArena::BufEntry *ab = ntt_arena_bufs(arena, N, out_slots, nbatch, digits_out == nullptr);
    if (!ab && digits_out) {
        *digits_out=nullptr;

        return 3;
    }
    unsigned long long *dA = nullptr, *dB = nullptr, *dQ = nullptr, *dOut = nullptr,
                       *dRes = nullptr;
    bool own = false;
    if (ab) {
        dA = ab->dA; dB = ab->dB; dQ = ab->dQ; dOut = ab->dOut; dRes = ab->dRes;
    } else {
        own = true;
        CK(cudaMalloc(&dA, (size_t)(N * nbatch) * sizeof(unsigned long long)));
        CK(cudaMalloc(&dB, (size_t)(N * nbatch) * sizeof(unsigned long long)));
        CK(cudaMalloc(&dQ, (size_t)(N * nbatch) * sizeof(unsigned long long)));
        CK(cudaMalloc(&dOut, (size_t)(out_slots * nbatch) * sizeof(unsigned long long)));
        CK(cudaMalloc(&dRes, 2 * (size_t)nbatch * sizeof(unsigned long long)));
    }
    const double tcopy0 = now_s();
    if (input) {
        const int rc = input->run ? input->run(input->ctx, sh, nbatch, dA, dB) : 2;
        if (rc) {

            if (own) { cudaFree(dA); cudaFree(dB); cudaFree(dQ); cudaFree(dOut); cudaFree(dRes); }
            return rc;
        }
        CK(cudaGetLastError());
    } else {
        CK(cudaMemcpy(dA, alias_a.empty() ? dAin : alias_a.data(), (size_t)N * nbatch * sizeof(unsigned long long),
                      alias_a.empty() ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dB, alias_b.empty() ? dBin : alias_b.data(), (size_t)N * nbatch * sizeof(unsigned long long),
                      alias_b.empty() ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice));
    }
    const double t_copy = input ? 0.0 : now_s() - tcopy0;
    const unsigned long long max_y = 65535;
    const unsigned long long nchunk = (nbatch + max_y - 1) / max_y;
    /* with a reduction hook the caller never reads the slot projections, so the assembly kernel
       and its D2H are skipped here too (NTT_S4_KEEPHOUT=1 restores the old behaviour for the A/B) */
    const bool want_hout = (hook == nullptr || hook->run == nullptr) ? true : [] {
        const char *e = std::getenv("NTT_S4_KEEPHOUT");
        return e && *e && std::atoi(e) != 0;
    }();
    NttPassResult r;
    if (want_hout) r.hOut.assign((size_t)(out_slots * nbatch), 0);
    /* ONE reset and ONE readback for the whole chunk (section 41): the counters live at
       dRes[2*slice] and the kernel only ever touches its own slice's two words, so accumulating
       them across the chunk and reading once is exactly equivalent to reading after each
       call -- see the note where the call defers. */
    /* ---- DEFERRED CARRY CHECK (section 29) -------------------------------------------------
       A chunked caller pays this blocking readback ONCE PER CHUNK, and at the production shape
       that is 22471 chunks x 1.64 ms = 36.75 s of the 253.53 s run (measured: `carrysplt d2h`),
       even though the copy itself is 2*nbatch words -- the cost is the pipeline DRAIN, not the
       bytes.  With defer_carry the caller asks this call to leave the counters in place (no
       memset, no readback) and to read them once after the last deferred chunk, through
       ntt_batch_carry_finish.  The memset must be skipped for the same reason: it would wipe the
       interior chunks' counters before they are read.  Only legal with an arena, because the
       dRes buffer has to outlive this call. */
    const bool defer = defer_carry && (arena != nullptr) && (ab != nullptr) && !own;
    if (st) st->carry_deferred = defer;
    if (!defer)
        CK(cudaMemset(dRes, 0, 2 * (size_t)nbatch * sizeof(unsigned long long)));
    std::vector<NttPassResult> sub((size_t)nchunk);
    for (unsigned long long ci = 0; ci < nchunk; ++ci) {
        const unsigned long long s0 = ci * max_y;
        const unsigned long long m = ((nbatch - s0) < max_y) ? (nbatch - s0) : max_y;
        NttPassResult rr = ntt_run_passes(sh, fc, dA + s0 * N, dB + s0 * N, dQ + s0 * N,
                                          dOut + s0 * out_slots, dRes + 2 * s0, m, 0,
                                          nullptr, nullptr, nullptr, nullptr, want_hout, true);
        sub[(size_t)ci] = rr;
        if (!rr.hOut.empty())
            std::copy(rr.hOut.begin(), rr.hOut.end(), r.hOut.begin() + (long)(s0 * out_slots));
        r.t_slot += rr.t_slot;
        r.t_check += rr.t_check;
        r.t_check_reset += rr.t_check_reset;
        if (nchunk == 1) r.digits = rr.digits;
        if (hook && hook->run && hook->out && hook->w) {
            hook->run(hook->ctx, rr.digits, N, sh.bpw, sh.slot_words, sh.slot_bits, out_slots,
                      m, hook->out + (size_t)(s0 * out_slots) * hook->w, hook->w);
            CK(cudaGetLastError());
        }
    }
    /* THE ONE DRAIN OF THE WHOLE CHUNK (section 41): the carry counters and the deferred
       in-stream marks are both read here, after a single blocking copy, instead of once per
       call.  This is also where an asynchronous kernel failure surfaces. */
    {
        const double tr0 = now_s();
        if (!defer) {
            r.hRes.assign((size_t)(2 * nbatch), 0);
            CK(cudaMemcpy(r.hRes.data(), dRes, r.hRes.size() * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
        }
        for (unsigned long long ci = 0; ci < nchunk; ++ci)
            ntt_pass_read_marks(sub[(size_t)ci], &r.t_fwd, &r.t_inv, &r.t_check_kernel);
        if (!defer) r.t_check_d2h += now_s() - tr0;
    }
    if (st) {
        st->P = P; st->S = S; st->N = N; st->k = sh.k; st->bpw = sh.bpw;
        st->slot_bits = sh.slot_bits; st->slot_stride = sh.slot_stride;
        st->slot_words = sh.slot_words; st->out_slots = out_slots; st->L_terms = sh.L_terms;
        st->cw = (unsigned long long)((sh.slot_bits + 63) / 64);
        st->passes_fwd = sh.passes_fwd; st->passes_total = sh.passes_total;
        st->carry_rounds = sh.carry_rounds;
        st->carry_residual = 0; st->carry_max_bits = 0;
        /* with a deferred carry check hRes is deliberately EMPTY here (the caller reads the whole
           accumulation once, through ntt_batch_carry_finish), so the per-slice scan must not run:
           it would index an empty vector */
        if (!defer)
            for (unsigned long long s = 0; s < nbatch; ++s) {
                st->carry_residual += r.hRes[(size_t)(2 * s)];
                if (r.hRes[(size_t)(2 * s + 1)] > st->carry_max_bits)
                    st->carry_max_bits = r.hRes[(size_t)(2 * s + 1)];
            }
        st->mem_mb = sh.mem_mb * (double)nbatch;
        st->t_fwd = r.t_fwd; st->t_inv = r.t_inv; st->t_slot = r.t_slot;
        st->t_check += r.t_check;
        st->t_check_reset += r.t_check_reset;
        st->t_check_kernel += r.t_check_kernel;
        st->t_check_d2h += r.t_check_d2h;
        st->t_plan += t_plan;
        st->t_opcopy += t_copy;
        st->t_hout += r.t_hout;
        /* the DEVICE entry point packs on the device: no host packing, no host scan and no
           operand upload to attribute, so those three fields stay 0 on this path */
        st->fuse_t = fc.t; st->fuse_nms = fc.nms;
        for (int q = 0; q < fc.nms && q < 8; ++q) st->fuse_ms[q] = fc.ms[q];
        st->exact_valid = true;
    }
    unsigned long long bad_res = 0;
    /* a deferred check has no hRes to scan: the caller's ntt_batch_carry_finish reads the whole
       accumulation and rejects it there, so scanning an empty vector here is both wrong (OOB)
       and redundant */
    if (!defer)
        for (unsigned long long s = 0; s < nbatch; ++s)
            if (r.hRes[(size_t)(2 * s)] != 0) ++bad_res;
    if (bad_res) {
        std::fprintf(stderr, NTT_PROBE_NAME ": CARRY DID NOT CONVERGE in %llu of %llu slices "
                             "after %d rounds\n", bad_res, nbatch, sh.carry_rounds);
        if (own) { cudaFree(dA); cudaFree(dB); cudaFree(dQ); cudaFree(dOut); cudaFree(dRes); }
        return 4;
    }
    if (digits_out) *digits_out = (nchunk == 1) ? r.digits : nullptr;

    if (own) { cudaFree(dA); cudaFree(dB); cudaFree(dQ); cudaFree(dOut); cudaFree(dRes); }
    return 0;
}

/* =====================================================================================
 * THE ONE READBACK OF A CHUNKED CALL (section 29).
 *
 * A chunked caller pays the carry check ONCE PER CHUNK, and the check is a pageable D2H of
 * 2*nbatch words -- which cannot return until the device has finished everything queued before
 * it, so its cost is a full pipeline DRAIN rather than the bytes.  Measured at the production
 * shape: 22471 chunks x 1.64 ms = 36.75 s of a 253.53 s run, the largest single host-side item.
 *
 * With `defer_carry` the multiply leaves the counters in the arena's dRes buffer (no memset, no
 * readback), so this function can read them once for the whole batch and give the same verdict.
 * It must be called with the SAME (N, nbatch) the deferred chunks used, and BEFORE any later
 * multiply reuses that arena entry -- a non-deferred multiply of the same shape memsets it.
 * ===================================================================================== */
int ntt_batch_carry_finish(NttArena *arena, unsigned long long N, unsigned long long nbatch,
                           NttMulStats *st)
{
    if (!arena) {
        std::fprintf(stderr, NTT_PROBE_NAME ": ntt_batch_carry_finish without an arena\n");
        return 2;
    }
    /* the composite the last multiply left behind -- NOT a fresh lookup, because a fresh entry
       would be uninitialised memory and would pass a check it never earned */
    if (!arena->cur.dRes || arena->cur.n != N || arena->cur.nbatch != nbatch) {
        std::fprintf(stderr, NTT_PROBE_NAME ": ntt_batch_carry_finish: the arena's current buffers "
                             "are not the deferred shape (want N=%llu nbatch=%llu, have N=%llu "
                             "nbatch=%llu)\n", N, nbatch, arena->cur.n, arena->cur.nbatch);
        return 3;
    }
    const double t0 = now_s();
    std::vector<unsigned long long> hRes((size_t)(2 * nbatch), 0);
    if (cudaMemcpy(hRes.data(), arena->cur.dRes, hRes.size() * sizeof(unsigned long long),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::fprintf(stderr, NTT_PROBE_NAME ": ntt_batch_carry_finish: the deferred readback "
                             "failed\n");
        (void)cudaGetLastError();
        return 5;
    }
    const double td = now_s() - t0;
    unsigned long long bad = 0;
    if (st) {
        st->t_check_d2h += td;                 /* charged to the one call that really paid it */
        st->carry_residual = 0;
        st->carry_max_bits = 0;
        for (unsigned long long s = 0; s < nbatch; ++s) {
            st->carry_residual += hRes[(size_t)(2 * s)];
            if (hRes[(size_t)(2 * s + 1)] > st->carry_max_bits)
                st->carry_max_bits = hRes[(size_t)(2 * s + 1)];
        }
    }
    for (unsigned long long s = 0; s < nbatch; ++s)
        if (hRes[(size_t)(2 * s)] != 0) ++bad;
    if (bad) {
        std::fprintf(stderr, NTT_PROBE_NAME ": CARRY DID NOT CONVERGE in %llu of %llu slices "
                             "(deferred check)\n", bad, nbatch);
        return 4;
    }
    return 0;
}

static void ntt_workspace_check(int device)
{
    unsigned long long checks=0, bad=0, words=0;
    auto check=[&](bool ok) { ++checks; if (!ok) ++bad; };
    {
        NttArena ar; ar.device=device; ar.workspace_pool=true;
        auto *first=ntt_arena_bufs(&ar,128,10,4);
        check(first!=nullptr); if (!first) std::exit(3);
        auto *a=first->dA; auto *res=first->dRes;
        unsigned long long poison=77, value=0;
        CK(cudaMemcpy(res,&poison,8,cudaMemcpyHostToDevice));
        auto *second=ntt_arena_bufs(&ar,256,10,2);
        check(second && second->dA==a && second->dRes!=res && ar.workspace_grows==1);
        auto *again=ntt_arena_bufs(&ar,128,10,4);
        check(again && again->dA==a && again->dRes==res);
        CK(cudaMemcpy(&value,res,8,cudaMemcpyDeviceToHost)); check(value==77);
        size_t span=0; check(ar.input_span(a+3,span) && span==509);
        const size_t small_charge=ar.bytes-ar.workspace.words*8-16;
        for (int index=1; index<=3; ++index) {
            ar.workspace_fail_alloc=index;
            check(ntt_arena_bufs(&ar,512,20,2)==nullptr);
            check(ar.workspace.words==0 && ar.bytes==small_charge);
        }
        ar.workspace_fail_alloc=0;
        check(ntt_arena_bufs(&ar,512,20,2)!=nullptr && ar.workspace.words==3072);
        CK(cudaMemcpy(&value,res,8,cudaMemcpyDeviceToHost)); check(value==77);
        ar.cap_bytes=ar.bytes;
        check(ntt_arena_bufs(&ar,4096,20,2)==nullptr && ar.workspace.words==0);
        ar.release(); check(ar.bytes==0 && ar.bigs.empty() && ar.smalls.empty());
        ar.release(); check(ar.bytes==0);
    }
    const unsigned long long P=3, nb=4;
    const int S=129;
    const size_t W=3;
    std::vector<unsigned long long> a(nb*P*W,0), b(a.size());
    for (size_t s=0; s<nb; ++s) for (size_t i=0; i<P; ++i) {
        a[(s*P+i)*W]=i+s+1; a[(s*P+i)*W+2]=1;
        b[(s*P+i)*W]=2*i+s+4; b[(s*P+i)*W+1]=s+1;
    }
    for (bool pool : {false,true}) {
        NttArena ar; ar.device=device; ar.workspace_pool=pool;
        std::vector<unsigned long long> output;
        unsigned long long *exported=nullptr;
        NttMulStats st;
        int rc=ntt_poly_mul_batch_host(P,S,device,nb,a.data(),b.data(),&output,&st,&ar,nullptr,&exported);
        check(rc==0 && exported && ar.workspace.words==0 && !ar.bigs.empty());
        if (rc || !exported) std::exit(3);
        std::vector<unsigned long long> expected(nb*st.out_slots,0);
        mpz_t x,y,sum; mpz_inits(x,y,sum,nullptr);
        for (size_t s=0; s<nb; ++s) for (size_t k=0; k<2*P-1; ++k) {
            mpz_set_ui(sum,0);
            for (size_t i=0; i<P; ++i) if (k>=i && k-i<P) {
                mpz_import(x,W,-1,8,0,0,&a[(s*P+i)*W]);
                mpz_import(y,W,-1,8,0,0,&b[(s*P+k-i)*W]);
                mpz_mul(x,x,y); mpz_add(sum,sum,x);
            }
            expected[s*st.out_slots+k]=mpz_fdiv_ui(sum,4294967291ul);
        }
        mpz_clears(x,y,sum,nullptr);
        check(output==expected); words+=expected.size();
        NttShape sh; FuseCtx fc;
        check(ntt_shape_plan(P,S,device,&ar,fc,sh)==0 && sh.N==st.N);
        std::vector<unsigned long long> pa(nb*sh.N,0), pb(pa.size());
        unsigned long long maxdigit=0;
        for (size_t s=0; s<nb; ++s)
            check(ntt_pack_operand(sh,&a[s*P*W],&b[s*P*W],&pa[s*sh.N],&pb[s*sh.N],&maxdigit));
        /* Crossed inputs test overwrite hazards. Snapshot before either destination write. */
        auto *src_a=ar.cur.dB, *src_b=ar.cur.dA;
        CK(cudaMemcpy(src_a,pa.data(),pa.size()*8,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(src_b,pb.data(),pb.size()*8,cudaMemcpyHostToDevice));
        const auto aliases_before=ar.alias_snapshots;
        /* A shorter batch forces eviction of the nb=4 source when the pool grows to nb=3. */
        ar.cap_bytes=ar.bytes;
        rc=ntt_poly_mul_batch_dev(P,S,device,3,src_a,src_b,&st,&ar,nullptr,nullptr,sh.bpw);
        check(rc==0 && ar.alias_snapshots==aliases_before+2);
        check(pool ? ar.bigs.empty() : ar.bigs.size()==1 && ar.bigs[0].nbatch==3);
        if (rc) std::exit(3);
        output.resize(3*st.out_slots);
        CK(cudaMemcpy(output.data(),ar.cur.dOut,output.size()*8,cudaMemcpyDeviceToHost));
        expected.resize(output.size()); check(output==expected); words+=output.size();
        check(st.carry_deferred==false);
        /* Force owned fallback. Its carry has already been checked, so never defer it. */
        ar.cap_bytes=1;
        unsigned long long *di_a=nullptr, *di_b=nullptr;
        CK(cudaMalloc(&di_a,3*sh.N*8)); CK(cudaMalloc(&di_b,3*sh.N*8));
        CK(cudaMemcpy(di_a,pa.data(),3*sh.N*8,cudaMemcpyHostToDevice));
        CK(cudaMemcpy(di_b,pb.data(),3*sh.N*8,cudaMemcpyHostToDevice));
        rc=ntt_poly_mul_batch_dev(P,S,device,3,di_a,di_b,&st,&ar,nullptr,nullptr,sh.bpw,true);
        if (st.carry_deferred) { NttMulStats finish; check(ntt_batch_carry_finish(&ar,st.N,3,&finish)==0); }
        /* Existing cached buffers can still be reused under a reduced cap; evict first. */
        if (!ar.workspace.words && ar.bigs.empty()) check(rc==0 && !st.carry_deferred);
        ar.release();
        rc=ntt_poly_mul_batch_dev(P,S,device,3,di_a,di_b,&st,&ar,nullptr,nullptr,sh.bpw,true);
        check(rc==0 && !st.carry_deferred && ar.overflow>0);
        exported=(unsigned long long *)1;
        rc=ntt_poly_mul_batch_dev(P,S,device,3,di_a,di_b,&st,&ar,nullptr,&exported,sh.bpw);
        check(rc==3 && exported==nullptr);
        CK(cudaFree(di_a)); CK(cudaFree(di_b));
    }
    std::printf("ntt_workspace_check: checks=%llu words=%llu bad=%llu "
                "(GMP, capacity reuse, dRes isolation, failure rollback, aliases/exports/fallback)\n",checks,words,bad);
    if (bad) std::exit(3);
}

int ntt_poly_mul_host(unsigned long long P,int S,int device,bool verbose,int dump,
    const unsigned long long *a,const unsigned long long *b,
    std::vector<unsigned long long> *out,std::vector<std::vector<unsigned long long>> *exact,
    NttMulStats *stats,NttArena *arena);
static void ntt_fuse_lifetime_check(int device)
{
    const unsigned long long P=17;
    const int S=129;
    std::vector<unsigned long long> a(P*3,0), b(P*3,0), output;
    for (size_t i=0;i<P;++i) { a[i*3]=i+1; b[i*3]=2*i+3; }
    NttArena ar; ar.device=device; ar.cap_bytes=1;
    const auto live_before=g_fuse_base.live_bytes;
    const auto allocations_before=g_fuse_base.allocations;
    const auto frees_before=g_fuse_base.frees;
    unsigned long long bad=0,calls=0;
    std::vector<unsigned long long> expected(2*P-1,0);
    for (size_t k=0;k<expected.size();++k)
        for (size_t i=0;i<P;++i) if(k>=i && k-i<P) expected[k]+=a[i*3]*b[(k-i)*3];
    for (int i=0;i<8;++i) {
        NttMulStats st;
        const int rc=ntt_poly_mul_batch_host(P,S,device,1,a.data(),b.data(),&output,&st,&ar,nullptr);
        ++calls;
        if (rc || output!=expected || g_fuse_base.live_bytes!=live_before || !ar.fuses.empty()) ++bad;
    }
    NttMulStats st;
    for (int i=0;i<2;++i) {
        const int rc=ntt_poly_mul_host(P,S,device,false,0,a.data(),b.data(),&output,nullptr,&st,&ar);
        ++calls; if(rc || output!=expected || g_fuse_base.live_bytes!=live_before) ++bad;
    }
    unsigned long long *exported=(unsigned long long *)1;
    int rc=ntt_poly_mul_batch_host(P,S,device,1,a.data(),b.data(),&output,&st,&ar,nullptr,&exported);
    ++calls; if(rc!=3 || exported || g_fuse_base.live_bytes!=live_before) ++bad;
    NttInputHook input;
    input.run=[](void *,const NttShape &,unsigned long long,unsigned long long *,unsigned long long *) { return 7; };
    rc=ntt_poly_mul_batch_dev(P,S,device,1,nullptr,nullptr,&st,&ar,nullptr,nullptr,0,false,&input);
    ++calls; if(rc!=7 || g_fuse_base.live_bytes!=live_before) ++bad;
    ar.cap_bytes=0;
    rc=ntt_poly_mul_batch_host(P,S,device,1,a.data(),b.data(),&output,&st,&ar,nullptr);
    ++calls; if(rc || output!=expected || ar.fuses.size()!=1 || g_fuse_base.live_bytes<=live_before) ++bad;
    const auto cached_allocs=g_fuse_base.allocations, cached_bytes=g_fuse_base.live_bytes;
    rc=ntt_poly_mul_batch_host(P,S,device,1,a.data(),b.data(),&output,&st,&ar,nullptr);
    ++calls; if(rc || output!=expected || g_fuse_base.allocations!=cached_allocs || g_fuse_base.live_bytes!=cached_bytes) ++bad;
    ar.release(); if(g_fuse_base.live_bytes!=live_before) ++bad;
    std::printf("ntt_fuse_lifetime_check: calls=%llu bad=%llu allocations=%llu frees=%llu leaked_bytes=%llu\n",
                calls,bad,g_fuse_base.allocations-allocations_before,g_fuse_base.frees-frees_before,
                g_fuse_base.live_bytes-live_before);
    if (bad) std::exit(3);
}

static void fuse_fixture_env(const char *key,const char *value)
{
#if defined(_WIN32)
    _putenv_s(key,value ? value : "");
#else
    if(value && *value) setenv(key,value,1); else unsetenv(key);
#endif
}
static void ntt_fuse_capacity_check(int device)
{
    CK(cudaSetDevice(device));
    const char *et=std::getenv("NTT_FUSE_T"), *em=std::getenv("NTT_FUSE_M");
    const std::string saved_t=et ? et : "", saved_m=em ? em : "";
    const auto live_before=g_fuse_base.live_bytes;
    unsigned long long cases=0,words=0,bad=0;
    const int ks[]={7,15,7},ts[]={4,8,8};
    for (int shape=0;shape<3;++shape) {
        const int k=ks[shape]; const unsigned long long n=1ull<<k;
        const auto om=gl_pow_host(7ull,(GL_P-1)/n), omi=gl_pow_host(om,GL_P-2);
        const auto nsc=gl_pow_host(n,GL_P-2);
        std::vector<unsigned long long> original(n,0),ones(n,1),spectrum(n,0),got(n);
        original[0]=GL_P-1; original[1]=0x8000000000000000ull; original[n-1]=GL_P-2;
        /* Independent sparse DFT in GMP, stored in DIF bit-reversed order. */
        mpz_t p,w,wi,u,v,sum,tmp,ca,cb,cc;
        mpz_inits(p,w,wi,u,v,sum,tmp,ca,cb,cc,nullptr);
        const unsigned long long prime=GL_P;
        mpz_import(p,1,-1,8,0,0,&prime); mpz_import(w,1,-1,8,0,0,&om);
        mpz_import(wi,1,-1,8,0,0,&omi);
        mpz_import(ca,1,-1,8,0,0,&original[0]); mpz_import(cb,1,-1,8,0,0,&original[1]);
        mpz_import(cc,1,-1,8,0,0,&original[n-1]); mpz_set_ui(u,1); mpz_set_ui(v,1);
        for (unsigned long long j=0;j<n;++j) {
            mpz_mul(sum,cb,u); mpz_add(sum,sum,ca); mpz_mul(tmp,cc,v); mpz_add(sum,sum,tmp); mpz_mod(sum,sum,p);
            unsigned long long r=0,x=j; for(int bit=0;bit<k;++bit){r=(r<<1)|(x&1);x>>=1;}
            size_t count=0; mpz_export(&spectrum[r],&count,-1,8,0,0,sum);
            mpz_mul(u,u,w);mpz_mod(u,u,p);mpz_mul(v,v,wi);mpz_mod(v,v,p);
        }
        mpz_clears(p,w,wi,u,v,sum,tmp,ca,cb,cc,nullptr);
        unsigned long long *data=nullptr,*dones=nullptr;
        CK(cudaMalloc(&data,n*8));CK(cudaMalloc(&dones,n*8));
        CK(cudaMemcpy(dones,ones.data(),n*8,cudaMemcpyHostToDevice));
        fuse_fixture_env("NTT_FUSE_T",std::to_string(ts[shape]).c_str());
        for (int m=1;m<=FUSE_MAX_M;++m) for(bool compact:{false,true}) {
            fuse_fixture_env("NTT_FUSE_M",std::to_string(m).c_str());
            FuseCtx fc; fuse_init(fc,n,k,om,omi,compact);
            for (int state=0;state<3;++state) {
                if(state==1) ntt_fuse_cache_tables(fc,om,omi);
                if(state==2) NttArena::fuse_drop_tables(fc);
                CK(cudaMemcpy(data,original.data(),n*8,cudaMemcpyHostToDevice));
                ntt_forward_fused(data,fc,om);
                CK(cudaMemcpy(got.data(),data,n*8,cudaMemcpyDeviceToHost));
                for(size_t j=0;j<n;++j){if(got[j]!=spectrum[j]) ++bad;} words+=n;
                ntt_inverse_fused(data,dones,fc,omi,nsc);
                CK(cudaMemcpy(got.data(),data,n*8,cudaMemcpyDeviceToHost));
                for(size_t j=0;j<n;++j){if(got[j]!=original[j]) ++bad;} words+=n;
                ++cases;
            }
            if(compact && fc.nms==0 && (fc.scr || fc.scr2 || fc.scrWords || fc.scr2Words)) ++bad;
            fuse_release(fc);
        }
        CK(cudaFree(data));CK(cudaFree(dones));
    }
    fuse_fixture_env("NTT_FUSE_T",saved_t.c_str());fuse_fixture_env("NTT_FUSE_M",saved_m.c_str());
    if(g_fuse_base.live_bytes!=live_before) ++bad;
    std::printf("ntt_fuse_capacity_check: cases=%llu words=%llu bad=%llu (GMP DFT, inverse, radix 1..%d, cache/evict/tile-only)\n",
                cases,words,bad,FUSE_MAX_M);
    if(bad) std::exit(3);
}

int ntt_poly_mul_host(unsigned long long P, int S, int device, bool verbose, int dump,
                      const unsigned long long *wordsA, const unsigned long long *wordsB,
                      std::vector<unsigned long long> *out_slots_u64,
                      std::vector<std::vector<unsigned long long>> *out_exact,
                      NttMulStats *st, NttArena *arena = nullptr)
{
    const double thb_setup = now_s();
    double hb_t_setup = 0, hb_t_pack = 0, hb_t_maxc = 0, hb_t_h2d = 0, hb_t_d2h = 0,
           hb_t_ext = 0, hb_t_xchk = 0;
    /* THE SHAPE PLAN (shared with the batched entry point, so the two cannot drift) */
    NttShape sh;
    FuseCtx fc;
    FuseCallGuard fuse_guard(fc);
    {
        const int rc = ntt_shape_plan(P, S, device, arena, fc, sh);
        if (rc) return rc;
    }
    const unsigned long long N = sh.N, out_slots = sh.out_slots, W = sh.W;
    const int bpw = sh.bpw;
    const unsigned long long L_terms = sh.L_terms, slot_bits = sh.slot_bits;
    const unsigned long long slot_stride = sh.slot_stride, slot_words = sh.slot_words;
    const int k = sh.k;
    const unsigned long long omega = sh.omega;
    NTT_HBS(t_setup, thb_setup);

    /* ---- THE PACKING (shared): coefficient i at bit i*slot_stride, bpw bits per digit ----
       `slot_stride` MUST be a multiple of bpw.  Packing compactly at exactly i*slot_bits
       (slot_bits = 2S + log2 P is not a multiple of bpw) makes one DIGIT of the packed
       operand hold the tail of one bpw-bit chunk of the coefficient plus the head of the
       next chunk (the write `h[bit/bpw] |= v << (bit%bpw)` spills up to bpw-1 bits past the
       digit).  A digit then reaches 2^(2*bpw-1) instead of 2^bpw, the convolution
       coefficient bound becomes L*(2^(2bpw-1))^2 -- e.g. 2^66 for P=4/S=64 and 2^67 for
       P=1024/S=5153, both far ABOVE p -- so the exactness test L*(2^bpw-1)^2 < p keeps passing
       while EVERY coefficient of the product wraps mod p.  Symptom: every slot wrong
       (bad_slots = 2P-1, slot 0 included) with the geometry and the timings unchanged. */
    const double thb_pack = now_s();
    std::vector<uint64_t> hA(N, 0), hB(N, 0);
    if (!ntt_pack_operand(sh, wordsA, wordsB, hA.data(), hB.data(), nullptr)) return 5;
    NTT_HBS(t_pack, thb_pack);

    /* ---- INDEPENDENT empirical check of the bound, on the digits just packed ------------ */
    unsigned long long maxcoeff = 0, maxcoeff_k = 0, maxcoeff_limbs = 0;
    bool maxcoeff_ran = false;
    const double thb_maxc = now_s();
    {
        const int rc = ntt_shape_maxcoeff(sh, hA, hB, &maxcoeff, &maxcoeff_k, &maxcoeff_ran,
                                          &maxcoeff_limbs);
        if (rc) return rc;
    }
    NTT_HBS(t_maxc, thb_maxc);
    ntt_shape_print(sh, fc, verbose, maxcoeff, maxcoeff_k, maxcoeff_ran, maxcoeff_limbs);

    unsigned long long *dA = nullptr, *dB = nullptr, *dC = nullptr, *dQ = nullptr,
                       *dOut = nullptr, *dRes = nullptr;
    NttArena::BufEntry *ab = ntt_arena_bufs(arena, N, out_slots);   /* nullptr: allocate here */
    if (ab) {
        dA = ab->dA; dB = ab->dB; dC = ab->dC; dQ = ab->dQ; dOut = ab->dOut; dRes = ab->dRes;
    } else {
        CK(cudaMalloc(&dA, N * sizeof(unsigned long long)));
        CK(cudaMalloc(&dB, N * sizeof(unsigned long long)));
        CK(cudaMalloc(&dC, N * sizeof(unsigned long long)));
        CK(cudaMalloc(&dQ, N * sizeof(unsigned long long)));
        CK(cudaMalloc(&dOut, out_slots * sizeof(unsigned long long)));
        CK(cudaMalloc(&dRes, 2 * sizeof(unsigned long long)));
    }
    const double thb_h2d = now_s();
    CK(cudaMemcpy(dA, hA.data(), N * sizeof(unsigned long long), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hB.data(), N * sizeof(unsigned long long), cudaMemcpyHostToDevice));
    NTT_HBS(t_h2d, thb_h2d);

    /* ---- the device passes (shared runner; nbatch == 1) --------------------------------- */
    std::vector<unsigned long long> hfa, hfb, hPre, hPost;
    const NttPassResult pr = ntt_run_passes(sh, fc, dA, dB, dQ, dOut, dRes, 1,
                                           dump, &hfa, &hfb, &hPre, &hPost);
    const double t_fwd = pr.t_fwd, t_inv = pr.t_inv, t_slot = pr.t_slot;
    const std::vector<unsigned long long> &hOut = pr.hOut;
    const unsigned long long *dDig = pr.digits;
    const unsigned long long hRes[2] = {pr.hRes[0], pr.hRes[1]};
    if (dump) {
        /* host direct DFT of the same inputs, in the bit-reversed order the DIF produces */
        auto brev = [&](unsigned long long i) {
            unsigned long long r = 0, v = i;
            for (int q = 0; q < k; ++q) { r = (r << 1) | (v & 1ull); v >>= 1; }
            return r;
        };
        int b1 = 0, b2 = 0;
        for (unsigned long long i = 0; i < N; ++i) {
            unsigned long long acc = 0, acc2 = 0;
            for (unsigned long long j = 0; j < N; ++j) {
                const unsigned long long w = gl_pow_host(omega, (i * j) % N);
                acc = gl_add_host(acc, gl_mul_host(hA[(size_t)j], w));
                acc2 = gl_add_host(acc2, gl_mul_host(hB[(size_t)j], w));
            }
            if (acc != hfa[(size_t)brev(i)]) ++b1;
            if (acc2 != hfb[(size_t)brev(i)]) ++b2;
        }
        std::printf("  dump: forward-vs-directDFT  A bad=%d/%llu  B bad=%d/%llu\n", b1,
                    (unsigned long long)N, b2, (unsigned long long)N);
    }
    if (dump && verbose) {
    std::printf("  carry: %d ripple rounds computed in 1 pass, residual digits >= 2^%d: %llu "
                "(max digit height %llu bits)\n", sh.carry_rounds, bpw,
                (unsigned long long)hRes[0], (unsigned long long)hRes[1]);
    }
    if (hRes[0] != 0) {
        std::fprintf(stderr, NTT_PROBE_NAME ": CARRY DID NOT CONVERGE: %llu digits still >= "
                             "2^%d after %d rounds\n",
                     (unsigned long long)hRes[0], bpw, sh.carry_rounds);
        return 4;
    }
    /* ---- hand the result back, then release everything this call allocated -------------- */
    if (out_slots_u64) *out_slots_u64 = hOut;
    bool exact_ok = true;
    if (dump) {
        /* (a) the exact product's digit array, from the SAME packed operands, via GMP:
           A = sum_j hA[j]*2^(bpw*j), C = A*B, digit j = bits [j*bpw, (j+1)*bpw) of C. */
        mpz_t A, B, C;
        mpz_inits(A, B, C, nullptr);
        mpz_set_ui(A, 0);
        for (unsigned long long j = N; j-- > 0;) {
            mpz_mul_2exp(A, A, (unsigned)bpw);
            mpz_add_u64(A, hA[(size_t)j]);
        }
        mpz_set_ui(B, 0);
        for (unsigned long long j = N; j-- > 0;) {
            mpz_mul_2exp(B, B, (unsigned)bpw);
            mpz_add_u64(B, hB[(size_t)j]);
        }
        mpz_mul(C, A, B);
        std::printf("  dump: P=%llu S=%d bpw=%d N=%llu slot_bits=%llu payload_digits=%llu "
                    "needed(2*payload-1)=%llu exact_product_bits=%zu\n",
                    (unsigned long long)P, S, bpw, (unsigned long long)N,
                    (unsigned long long)slot_bits,
                    (unsigned long long)((sh.payload_bits + (unsigned)bpw - 1) / (unsigned)bpw),
                    (unsigned long long)((2 * sh.payload_bits - 1) / (unsigned)bpw + 1),
                    mpz_sizeinbase(C, 2));
        std::printf("  dump: input digits hA[0..3]=%llu %llu %llu %llu   hB[0..3]=%llu %llu "
                    "%llu %llu\n", hA[0], hA[1], hA[2], hA[3], hB[0], hB[1], hB[2], hB[3]);
        std::printf("  dump: input coefficients (S-bit, %llu word(s) each):", (unsigned long long)W);
        for (unsigned long long i = 0; i < P && i < 6; ++i)
            std::printf(" A[%llu]=%llu B[%llu]=%llu", i, wordsA[(size_t)(i * W)], i,
                        wordsB[(size_t)(i * W)]);
        std::printf("\n");
        dump_cmp("pre-carry (after inverse NTT)", hPre, C, N, bpw, 12);
        dump_cmp("post-carry", hPost, C, N, bpw, 12);
        /* What the inverse NTT MUST return: the cyclic convolution of the two input arrays,
           reduced mod p.  Compared element by element, and also up to a constant factor (a
           missing/incorrect 1/N shows up as a single global ratio). */
        std::vector<unsigned long long> conv(N, 0);
        for (unsigned long long i = 0; i < N; ++i)
            for (unsigned long long j = 0; j < N; ++j)
                conv[(size_t)((i + j) % N)] = gl_add_host(conv[(size_t)((i + j) % N)],
                                                          gl_mul_host(hA[(size_t)i],
                                                                      hB[(size_t)j]));
        unsigned long long cbad = 0, cfirst = 0;
        for (unsigned long long i = 0; i < N; ++i)
            if (conv[(size_t)i] != hPre[(size_t)i]) { if (!cbad) cfirst = i; ++cbad; }
        std::printf("  dump: inverse-vs-host-cyclic-conv bad=%llu/%llu first=%llu\n", cbad,
                    (unsigned long long)N, cfirst);
        if (conv[0] != 0) {
            const unsigned long long ratio =
                gl_mul_host(hPre[0], gl_pow_host(conv[0], GL_P - 2ull));
            unsigned long long rbad = 0;
            for (unsigned long long i = 0; i < N; ++i)
                if (gl_mul_host(conv[(size_t)i], ratio) != hPre[(size_t)i]) ++rbad;
            std::printf("  dump: pre-carry / conv ratio=%llu (uniform for %llu/%llu entries); "
                        "N^-1 mod p=%llu, N mod p=%llu\n", ratio, N - rbad,
                        (unsigned long long)N, sh.n_scale, N % GL_P);
        }
        std::printf("  dump: conv[0..5]=%llu %llu %llu %llu %llu %llu  pre-carry[0..5]=%llu "
                    "%llu %llu %llu %llu %llu\n", conv[0], conv[1], conv[2], conv[3], conv[4],
                    conv[5], hPre[0], hPre[1], hPre[2], hPre[3], hPre[4], hPre[5]);
        mpz_clears(A, B, C, nullptr);
    }
    if (out_exact) {
        std::vector<unsigned long long> hDig(N, 0);
        double thb = now_s();
        CK(cudaMemcpy(hDig.data(), dDig, N * sizeof(unsigned long long),
                      cudaMemcpyDeviceToHost));
        NTT_HBS(t_d2h, thb);
        thb = now_s();
        exact_ok = extract_exact_coeffs(hDig.data(), N, bpw, slot_bits, slot_stride, out_slots,
                                        out_exact);
        NTT_HBS(t_ext, thb);
        thb = now_s();
        /* TWO INDEPENDENT EXTRACTIONS of the same coefficient must agree: the device's
           mod-SLOT_MOD projection (checked against GMP by run_poly's `verify` block) and this
           host extraction of the full integer.  A disagreement means one of them is wrong, so
           it is reported and turned into a non-zero return instead of being passed on. */
        for (unsigned long long kx = 0; kx < out_slots && exact_ok; ++kx) {
            mpz_t tx;
            mpz_init(tx);
            mpz_import(tx, (*out_exact)[(size_t)kx].size(), -1, 8, 0, 0,
                       (*out_exact)[(size_t)kx].data());
            mpz_mod_ui(tx, tx, (unsigned long)SLOT_MOD);
            const unsigned long long got = (unsigned long long)mpz_get_ui(tx);
            mpz_clear(tx);
            if (got != hOut[(size_t)kx]) {
                std::fprintf(stderr, "%s: EXACT SLOT MISMATCH at k=%llu: host extraction "
                                     "%llu != device slot assembly %llu\n",
                             NTT_PROBE_NAME, (unsigned long long)kx, got,
                             (unsigned long long)hOut[(size_t)kx]);
                exact_ok = false;
            }
        }
        NTT_HBS(t_xchk, thb);
    }
    if (st) {
        st->P = P;
        st->S = S;
        st->N = N;
        st->k = k;
        st->bpw = bpw;
        st->slot_bits = slot_bits;
        st->slot_stride = slot_stride;
        st->slot_words = slot_words;
        st->out_slots = out_slots;
        st->L_terms = L_terms;
        st->cw = (unsigned long long)((slot_bits + 63) / 64);
        st->passes_fwd = sh.passes_fwd;
        st->passes_total = sh.passes_total;
        st->carry_rounds = sh.carry_rounds;
        st->carry_residual = hRes[0];
        st->carry_max_bits = hRes[1];
        st->mem_mb = sh.mem_mb;
        st->t_fwd = t_fwd;
        st->t_inv = t_inv;
        st->t_slot = t_slot;
        st->fuse_t = fc.t;
        st->fuse_nms = fc.nms;
        for (int q = 0; q < fc.nms && q < 8; ++q) st->fuse_ms[q] = fc.ms[q];
        st->exact_valid = exact_ok;
        st->t_setup = hb_t_setup;
        st->t_pack = hb_t_pack;
        st->t_maxc = hb_t_maxc;
        st->t_h2d = hb_t_h2d;
        st->t_carry = 0.0;
        st->t_d2h = hb_t_d2h;
        st->t_ext = hb_t_ext;
        st->t_xchk = hb_t_xchk;
    }

    if (!ab) { cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dQ); cudaFree(dOut);
               cudaFree(dRes); }
    return exact_ok ? 0 : 8;
}

static int run_poly(unsigned long long P, int S, int device, bool verify, int dump = 0)
{
    /* Coefficients are S bits each: W = ceil(S/64) words, top word masked. */
    const unsigned long long W = (unsigned long long)(S + 63) / 64;
    std::vector<uint64_t> wordsA(P * W, 0), wordsB(P * W, 0);
    {
        uint64_t s = 0x1234567ull;
        for (size_t i = 0; i < wordsA.size(); ++i) {
            s = s * 6364136223846793005ull + 1442695040888963407ull;
            wordsA[i] = s;
        }
        s = 0x89abcdefull;
        for (size_t i = 0; i < wordsB.size(); ++i) {
            s = s * 6364136223846793005ull + 1442695040888963407ull;
            wordsB[i] = s;
        }
        const int top = S - (int)(W - 1) * 64;
        const uint64_t top_mask = (top >= 64) ? ~0ull : ((1ull << top) - 1ull);
        for (unsigned long long i = 0; i < P; ++i) {
            wordsA[(size_t)(i * W + W - 1)] &= top_mask;
            wordsB[(size_t)(i * W + W - 1)] &= top_mask;
        }
    }
    /* debug hook: with NTT_PROBE_DEBUG=1 every coefficient is a power of two so the exact
       product slot is known by hand (A = x^0, B = x^1 -> product 2 at coefficient 1). */
    const bool dbg = std::getenv("NTT_PROBE_DEBUG") != nullptr;
    if (dbg) {
        for (unsigned long long i = 0; i < P; ++i) {
            for (unsigned long long w = 0; w < W; ++w) {
                wordsA[(size_t)(i * W + w)] = 0;
                wordsB[(size_t)(i * W + w)] = 0;
            }
        }
        wordsA[0] = 1;                       /* A = 1 + 0*x + ... */
        wordsB[0] = 0;                       /* B = x */
        if (W > 0) wordsB[1] = 1;
    }

    /* ---- THE multiply: one call into the shared implementation (above) ------------------
       Everything the pipeline needs (packing, exactness assertions, fused transform, carry,
       slot assembly) lives in ntt_poly_mul_host, which the GPU tree engine calls too. */
    std::vector<unsigned long long> hOut;
    NttMulStats st{};
    const int rc = ntt_poly_mul_host(P, S, device, true, dump, wordsA.data(), wordsB.data(),
                                     &hOut, nullptr, &st);
    if (rc != 0) return rc;
    /* the names the code below (and the printed result line) already used */
    const unsigned long long out_slots = st.out_slots;
    const unsigned long long N = st.N;
    const int bpw = st.bpw;
    const int passes_total = st.passes_total;
    const double mem_mb = st.mem_mb;
    const double t_fwd = st.t_fwd, t_inv = st.t_inv, t_slot = st.t_slot;
    const unsigned long long slot_bits = st.slot_bits;

    /* ---- verification: GMP schoolbook, coefficient by coefficient, mod 4294967291 ---- */
    bool ok = true;
    std::string detail;
    /* For P > 4096 the full P^2 schoolbook is too slow (same as the cuFFT probe), but a
       deterministic SAMPLE of coefficients is cheap and EXACT: coefficient k is
       sum_{i+j=k} a_i b_j, which costs O(P) GMP multiply-adds.  This runs even with
       verify=0, so that `ok=1` on a timing run means "the printed timing comes from a shape
       whose slots were checked against GMP", not merely "no check was requested". */
    if (P > 4096) {
        verify = false;
        const unsigned long long samples[] = {
            0, 1, 2, 3, P / 4, P / 2, P - 1, P, P + 1, 2 * P - 3, 2 * P - 2
        };
        mpz_t *ca = new mpz_t[P];
        mpz_t *cb = new mpz_t[P];
        for (unsigned long long i = 0; i < P; ++i) {
            mpz_init(ca[i]);
            mpz_init(cb[i]);
            mpz_import(ca[i], W, -1, 8, 0, 0, &wordsA[(size_t)(i * W)]);
            mpz_import(cb[i], W, -1, 8, 0, 0, &wordsB[(size_t)(i * W)]);
        }
        mpz_t t, acc, mod, lo;
        mpz_inits(t, acc, mod, lo, nullptr);
        mpz_set_ui(mod, 4294967291ull);
        unsigned long long bad = 0, checked = 0;
        for (size_t si = 0; si < sizeof(samples) / sizeof(samples[0]); ++si) {
            const unsigned long long kk = samples[si];
            if (kk >= out_slots) continue;
            mpz_set_ui(acc, 0);
            const unsigned long long i0 = (kk + 1 > P) ? (kk + 1 - P) : 0ull;
            const unsigned long long i1 = (kk < P - 1) ? kk : (P - 1);
            for (unsigned long long i = i0; i <= i1; ++i) {
                mpz_mul(t, ca[i], cb[kk - i]);
                mpz_add(acc, acc, t);
            }
            mpz_mod(lo, acc, mod);
            const unsigned long long want = mpz_get_ui(lo);
            ++checked;
            if (want != hOut[(size_t)kk]) {
                if (bad == 0) {
                    char buf[192];
                    std::snprintf(buf, sizeof(buf), " first_bad_slot=%llu got=%llu want=%llu",
                                  kk, hOut[(size_t)kk], want);
                    detail += buf;
                }
                ++bad;
            }
        }
        if (bad) detail += " bad_sampled_slots=" + std::to_string(bad);
        ok = (bad == 0);
        std::printf("  verify: P=%llu -> FULL P^2 schoolbook skipped; %llu sample coefficients "
                    "checked exactly against GMP (slots 0,1,2,3,P/4,P/2,P-1,P,P+1,2P-3,2P-2): "
                    "bad=%llu\n", (unsigned long long)P, (unsigned long long)checked,
                    (unsigned long long)bad);
        mpz_clears(t, acc, mod, lo, nullptr);
        for (unsigned long long i = 0; i < P; ++i) { mpz_clear(ca[i]); mpz_clear(cb[i]); }
        delete[] ca;
        delete[] cb;
    }
    if (verify) {
        mpz_t *ca = new mpz_t[P];
        mpz_t *cb = new mpz_t[P];
        for (unsigned long long i = 0; i < P; ++i) {
            mpz_init(ca[i]);
            mpz_init(cb[i]);
            mpz_import(ca[i], W, -1, 8, 0, 0, &wordsA[(size_t)(i * W)]);
            mpz_import(cb[i], W, -1, 8, 0, 0, &wordsB[(size_t)(i * W)]);
        }
        /* GMP's mpz_t is an array type, so std::vector<mpz_t> will not compile (MSVC's
           allocator needs a new-initializer for it) -- use new[]/delete[] like the cuFFT
           probe does. */
        mpz_t *acc = new mpz_t[out_slots];
        for (unsigned long long i = 0; i < out_slots; ++i) mpz_init(acc[i]);
        mpz_t t;
        mpz_init(t);
        for (unsigned long long i = 0; i < P; ++i) {
            for (unsigned long long j = 0; j < P; ++j) {
                mpz_mul(t, ca[i], cb[j]);
                mpz_add(acc[i + j], acc[i + j], t);
            }
        }
        mpz_clear(t);
        unsigned long long bad = 0;
        mpz_t mod, lo;
        mpz_init_set_ui(mod, 4294967291ull);
        mpz_init(lo);
        for (unsigned long long i = 0; i < out_slots; ++i) {
            mpz_mod(lo, acc[i], mod);
            const unsigned long long want = mpz_get_ui(lo);
            if (want != hOut[(size_t)i]) {
                if (bad == 0) {
                    char buf[192];
                    std::snprintf(buf, sizeof(buf), " first_bad_slot=%llu got=%llu want=%llu",
                                  (unsigned long long)i, hOut[(size_t)i], want);
                    detail += buf;
                }
                ++bad;
            }
        }
        if (bad) detail += " bad_slots=" + std::to_string(bad);
        ok = (bad == 0);
        /* printed for ALL of the out_slots = 2P-1 coefficients (the gate "P=1024/S=5153 must
           compare ALL 2047 of them" is only auditable if the count is visible) */
        std::printf("  verify: FULL P^2 schoolbook (every one of the %llu = 2P-1 product "
                    "coefficients vs GMP, projected mod 4294967291): bad=%llu\n",
                    (unsigned long long)out_slots, (unsigned long long)bad);
        for (unsigned long long i = 0; i < out_slots; ++i) mpz_clear(acc[i]);
        delete[] acc;
        for (unsigned long long i = 0; i < P; ++i) { mpz_clear(ca[i]); mpz_clear(cb[i]); }
        delete[] ca;
        delete[] cb;
        mpz_clear(mod);
        mpz_clear(lo);
    }

    const double total = t_fwd + t_inv + t_slot;
    const double ns_per_coeff = total * 1e9 / (double)P;
    /* FIGURE OF MERIT -- must match the fp64 cuFFT probe exactly, or the two probes cannot
       be compared: PACKED slot bits of BOTH operands (2*P*slot_bits), which is what the
       published cuFFT baselines 0.284 / 0.278 / 0.340 ns/operand-bit were computed with.
       (A payload-based P*S denominator is ~1.066x smaller for S = 5153, so using it would
       flatter this probe by ~7%.) */
    const double ns_per_bit = total * 1e9 / (2.0 * (double)P * (double)slot_bits);
    const double ns_per_payload_bit = total * 1e9 / ((double)P * (double)S);
    std::printf("poly: mode=poly P=%llu S=%d slot_bits=%llu bpw=%d nwords=%llu fft=%llu "
                "mem_mb=%.0f ok=%d t_fwd=%.3f t_inv=%.3f t_slot=%.3f t_total=%.3f "
                "ns_per_coeff=%.1f ns_per_operand_bit=%.4f ns_per_payload_bit=%.4f "
                "passes=%d stages=%d%s%s\n",
                (unsigned long long)P, S, (unsigned long long)slot_bits, bpw,
                (unsigned long long)N, (unsigned long long)N, mem_mb, ok ? 1 : 0,
                t_fwd, t_inv, t_slot, total, ns_per_coeff, ns_per_bit, ns_per_payload_bit,
                /* `passes=` and `stages=` are the SAME number on purpose: the old field carried
                   a stage count, and what is being optimised now is the number of FULL ARRAY
                   PASSES (definition in the stage-fusion comment near the top).  The transform
                   stage count is printed as transform_stages= on the fusion line above. */
                passes_total, passes_total, detail.empty() ? "" : " detail=", detail.c_str());

    return ok ? 0 : 1;
}

/* --------------------------------------------------------------------------------- */
/* --------------------------------------------------------------------------------- */
/* generic big-integer multiply bench (optional): ns per operand-bit                   */
/* --------------------------------------------------------------------------------- */
static int run_bench(unsigned long long bits, int bpw_in, int device)
{
    CK(cudaSetDevice(device));
    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, device));
    CK(cudaFree(0));

    const NttCfg cfg = choose_cfg(bits, 64ull);
    if (!cfg.ok) {
        std::fprintf(stderr, NTT_PROBE_NAME ": %s\n", cfg.why.c_str());
        return 2;
    }
    const unsigned long long N = cfg.nwords;
    const int k = cfg.k;
    const int bpw = bpw_in > 0 ? bpw_in : cfg.bpw;
    /* the SAME term-count criterion as the poly path, with L = ceil(bits/bpw): an explicit
       bpw on the command line must not be able to break exactness silently */
    const unsigned long long L_terms = (bits + (unsigned long long)bpw - 1) /
                                       (unsigned long long)bpw;
    if (!exact_ok_terms(L_terms, bpw)) {
        std::fprintf(stderr, NTT_PROBE_NAME ": bench: bpw=%d violates L*(2^bpw-1)^2 < p for "
                             "L=%llu (bits=%llu); the old N-form check N*(2^bpw)^2 < p for "
                             "N=%llu is %s\n", bpw, (unsigned long long)L_terms,
                     (unsigned long long)bits, (unsigned long long)N,
                     exact_ok_nterms(N, bpw) ? "satisfied" : "violated too");
        return 3;
    }
    if ((unsigned long long)bpw * N < bits) {
        std::fprintf(stderr, NTT_PROBE_NAME ": bench: %llu bits do not fit in %llu words of %d "
                     "bits\n", (unsigned long long)bits, (unsigned long long)N, bpw);
        return 3;
    }
    const unsigned long long n = bits / 8 / sizeof(unsigned long long);
    const double mem_mb = (double)(4 * N * sizeof(unsigned long long)) / 1048576.0;
    std::printf(NTT_PROBE_NAME ": mode=bench device=%d (%s) bits=%llu bpw=%d nwords=%llu "
                "(log2=%d) n=%llu\n", device, prop.name, (unsigned long long)bits, bpw,
                (unsigned long long)N, k, (unsigned long long)n);

    std::vector<unsigned long long> hA(N, 0), hB(N, 0);
    {
        uint64_t s = 0x1234567ull;
        for (unsigned long long i = 0; i < n; ++i) {
            s = s * 6364136223846793005ull + 1442695040888963407ull;
            hA[(size_t)i] = s & ((1ull << bpw) - 1ull);
        }
        s = 0x89abcdefull;
        for (unsigned long long i = 0; i < n; ++i) {
            s = s * 6364136223846793005ull + 1442695040888963407ull;
            hB[(size_t)i] = s & ((1ull << bpw) - 1ull);
        }
    }

    const unsigned long long omega = gl_pow_host(7ull, (GL_P - 1ull) / N);
    const unsigned long long omega_inv = gl_pow_host(omega, GL_P - 2ull);
    const unsigned long long n_scale = gl_pow_host(N % GL_P, GL_P - 2ull);

    unsigned long long *dA = nullptr, *dB = nullptr, *dC = nullptr, *dQ = nullptr, *dBlk = nullptr;
    CK(cudaMalloc(&dA, N * sizeof(unsigned long long)));
    CK(cudaMalloc(&dB, N * sizeof(unsigned long long)));
    CK(cudaMalloc(&dC, N * sizeof(unsigned long long)));
    CK(cudaMalloc(&dQ, N * sizeof(unsigned long long)));
    CK(cudaMemcpy(dA, hA.data(), N * sizeof(unsigned long long), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hB.data(), N * sizeof(unsigned long long), cudaMemcpyHostToDevice));

    const unsigned int threads = 256;
    const unsigned int blocks = (unsigned int)((N + threads - 1) / threads);
    const unsigned int smem = (unsigned int)(threads * sizeof(unsigned long long));
    const unsigned long long inv_n = ((~0ull) / N) + 1;

    FuseCtx fc;
    fuse_init(fc, N, k, omega, omega_inv);

    const double t0 = now_s();
    ntt_forward_fused(dA, fc, omega);
    ntt_forward_fused(dB, fc, omega);
    CK(cudaDeviceSynchronize());
    const double t_fwd = now_s() - t0;

    const double t1 = now_s();
    ntt_inverse_fused(dA, dB, fc, omega_inv, n_scale);
    CK(cudaGetLastError());
    /* same one-pass ripple carry as the poly path (carry_cone_kernel writes to a separate
       buffer because it reads the raw neighbours it would otherwise race with) */
    const int carry_rounds = 2 + (k + 2 * bpw + bpw - 1) / bpw;
    unsigned long long *dDig = dA;
    switch (carry_rounds) {
        case 1: carry_cone_kernel<1><<<blocks, threads>>>(dA, dQ, N, bpw, N); break;
        case 2: carry_cone_kernel<2><<<blocks, threads>>>(dA, dQ, N, bpw, N); break;
        case 3: carry_cone_kernel<3><<<blocks, threads>>>(dA, dQ, N, bpw, N); break;
        case 4: carry_cone_kernel<4><<<blocks, threads>>>(dA, dQ, N, bpw, N); break;
        case 5: carry_cone_kernel<5><<<blocks, threads>>>(dA, dQ, N, bpw, N); break;
        case 6: carry_cone_kernel<6><<<blocks, threads>>>(dA, dQ, N, bpw, N); break;
        default: std::fprintf(stderr, NTT_PROBE_NAME ": bench: carry rounds %d out of range\n",
                              carry_rounds); return 3;
    }
    CK(cudaGetLastError());
    dDig = dQ;
    CK(cudaDeviceSynchronize());
    const double t_inv = now_s() - t1;

    /* compare against GMP: unpack both operands and the result from the packed bpw-bit
       words (bit by bit -- independent of the packing code path) and compare big ints */
    const double t2 = now_s();
    bool ok = false;
    std::string detail;
    {
        auto unpack = [&](const unsigned long long *w, unsigned long long nw, mpz_t out) {
            std::vector<uint8_t> bytes;
            for (unsigned long long i = 0; i < nw * (unsigned long long)bpw; i += 8) {
                unsigned long long byte = 0;
                for (int t = 0; t < 8; ++t) {
                    const unsigned long long bit = i + (unsigned long long)t;
                    const unsigned long long wi = bit / (unsigned long long)bpw;
                    const int off = (int)(bit % (unsigned long long)bpw);
                    unsigned long long v = (wi < nw) ? w[(size_t)wi] : 0ull;
                    v >>= off;
                    if (off + 1 > bpw && wi + 1 < nw) v |= w[(size_t)(wi + 1)] << (bpw - off);
                    byte |= (v & 1ull) << t;
                }
                bytes.push_back((uint8_t)byte);
            }
            while (!bytes.empty() && bytes.back() == 0) bytes.pop_back();
            mpz_set_ui(out, 0);
            if (!bytes.empty()) mpz_import(out, bytes.size(), -1, 1, 0, 0, bytes.data());
        };
        std::vector<unsigned long long> hA1(n, 0), hB1(n, 0), hC(N, 0);
        CK(cudaMemcpy(hA1.data(), dA, n * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hB1.data(), dB, n * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hC.data(), dDig, N * sizeof(unsigned long long),
                      cudaMemcpyDeviceToHost));

        mpz_t a, b, prod, mine, diff;
        mpz_inits(a, b, prod, mine, diff, nullptr);
        unpack(hA1.data(), n, a);
        unpack(hB1.data(), n, b);
        unpack(hC.data(), 2 * n, mine);
        mpz_mul(prod, a, b);
        ok = (mpz_cmp(prod, mine) == 0);
        if (!ok) {
            mpz_sub(diff, prod, mine);
            mpz_abs(diff, diff);
            detail += " abs diff bits=" + std::to_string(mpz_sizeinbase(diff, 2));
            bool top_nonzero = false;
            for (unsigned long long i = 2 * n; i < N; ++i) {
                if (hC[(size_t)i] != 0) { top_nonzero = true; break; }
            }
            if (top_nonzero) detail += " spill_above_2n=1";
        }
        mpz_clears(a, b, prod, mine, diff, nullptr);
    }
    const double t_carry = now_s() - t2;

    const double total = t_fwd + t_inv + t_carry;
    std::printf("kron: mode=bench bits=%llu bpw=%d n=%llu nwords=%llu fft=%llu mem_mb=%.0f "
                "ok=%d t_fwd=%.3f t_inv=%.3f t_carry=%.3f t_total=%.3f "
                "ns_per_operand_bit=%.4f%s%s\n",
                (unsigned long long)bits, bpw, (unsigned long long)n,
                (unsigned long long)N, (unsigned long long)N, mem_mb, ok ? 1 : 0,
                t_fwd, t_inv, t_carry, total, total * 1e9 / (double)(n * (unsigned long long)bpw),
                detail.empty() ? "" : " detail=", detail.c_str());

    fuse_release(fc);
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dQ);
    return ok ? 0 : 1;
}

#ifndef NTT_POLY_PROBE_NO_MAIN
int main(int argc, char **argv)
{
    /* Unbuffered stdout: this probe allocates GBs and is run under scripts, and a run that
       dies (or is killed) must not lose the lines that say how far it got.  Same lesson as
       the stage2 tree reference (docs section 15.4). */
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (gl_selftest() != 0) return 2;

    const std::string mode = (argc > 1) ? argv[1] : "poly";
    /*
     * nttcheck [log2 N] [device] -- the transform's own selftest, using the O(n^2) direct
     * DFT kernel as the oracle (it shares no index algebra with the butterfly stages, so
     * the two cannot agree by accident):
     *   (1) forward NTT of a fixed pseudo-random vector == the direct DFT sum, accounting
     *       for the bit-reversed output order;
     *   (2) inverse(forward(x)) == x, bit-exact;
     *   (3) the twiddle powers gl_twiddle(omega, i) == omega^i.
     * The earlier butterfly stages failed (1) while passing (3), which is how the bug in the
     * per-stage index decomposition was localised.
     */
    if (mode == "nttcheck") {
        const int logn = (argc > 2) ? std::atoi(argv[2]) : 8;
        const int dev = (argc > 3) ? std::atoi(argv[3]) : 0;
        CK(cudaSetDevice(dev));
        const unsigned long long n = 1ull << logn;
        const unsigned long long om = gl_pow_host(7ull, (GL_P - 1ull) / n);
        const unsigned long long omi = gl_pow_host(om, GL_P - 2ull);
        const unsigned long long nsc = gl_pow_host(n % GL_P, GL_P - 2ull);
        /* the FUSED transform (section 12) is what is checked here: same stages, same order,
           same twiddles -- the pass structure is the only thing that changed */
        FuseCtx fc;
        fuse_init(fc, n, logn, om, omi);
        std::vector<unsigned long long> x(n, 0);
        unsigned long long s = 0x1234567ull;
        for (unsigned long long i = 0; i < n; ++i) {
            s = s * 6364136223846793005ull + 1442695040888963407ull;
            x[(size_t)i] = s % GL_P;
        }
        unsigned long long *dx = nullptr, *dsp = nullptr, *dref = nullptr, *dtw = nullptr,
                           *dback = nullptr, *dsave = nullptr;
        CK(cudaMalloc(&dx, n * 8));
        CK(cudaMalloc(&dsp, n * 8));
        CK(cudaMalloc(&dref, n * 8));
        CK(cudaMalloc(&dtw, n * 8));
        CK(cudaMalloc(&dback, n * 8));
        CK(cudaMalloc(&dsave, n * 8));
        CK(cudaMemcpy(dx, x.data(), n * 8, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dsp, x.data(), n * 8, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dsave, x.data(), n * 8, cudaMemcpyHostToDevice));
        const unsigned int threads = 256;
        const unsigned int blocks = (unsigned int)((n + threads - 1) / threads);
        /* bit-reversal kernel test: dst[i] must equal src[bitrev(i, logn)] */
        {
            std::vector<unsigned long long> bi(n, 0);
            for (unsigned long long i = 0; i < n; ++i) bi[(size_t)i] = i;
            CK(cudaMemcpy(dref, bi.data(), n * 8, cudaMemcpyHostToDevice));
            bitrev_kernel<<<blocks, threads>>>(dref, dsave, (unsigned long long)logn, n);
            CK(cudaGetLastError());
            CK(cudaDeviceSynchronize());
            std::vector<unsigned long long> bo(n, 0), want(n, 0);
            CK(cudaMemcpy(bo.data(), dsave, n * 8, cudaMemcpyDeviceToHost));
            for (unsigned long long i = 0; i < n; ++i) {
                unsigned long long r = 0, v = i;
                for (int q = 0; q < logn; ++q) { r = (r << 1) | (v & 1ull); v >>= 1; }
                want[(size_t)i] = r;
            }
            int bb = 0;
            for (unsigned long long i = 0; i < n; ++i) if (bo[(size_t)i] != want[(size_t)i]) ++bb;
            std::printf("nttcheck: bitrev kernel test bad=%d/%llu  (got", bb,
                        (unsigned long long)n);
            for (int i = 0; i < (int)n && i < 8; ++i) std::printf(" %llu", bo[(size_t)i]);
            std::printf("  want");
            for (int i = 0; i < (int)n && i < 8; ++i) std::printf(" %llu", want[(size_t)i]);
            std::printf(")\n");
            CK(cudaMemcpy(dsave, x.data(), n * 8, cudaMemcpyHostToDevice));
        }
        ntt_forward_fused(dsp, fc, om);
        dft_direct_kernel<<<(unsigned int)n, 1>>>(dsave, dref, n, om);
        gl_twiddle_test_kernel<<<blocks, threads>>>(dtw, n, om);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        std::vector<unsigned long long> sp(n, 0), ref(n, 0), tw(n, 0);
        CK(cudaMemcpy(sp.data(), dsp, n * 8, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(ref.data(), dref, n * 8, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(tw.data(), dtw, n * 8, cudaMemcpyDeviceToHost));
        /* host model of the EXACT steps the device performs (bitrev, DIF desc, bitrev) */
        {
            auto bitrev_idx = [&](unsigned long long i) {
                unsigned long long r = 0, v = i;
                for (int q = 0; q < logn; ++q) { r = (r << 1) | (v & 1ull); v >>= 1; }
                return r;
            };
            std::vector<unsigned long long> h1(n, 0), h2(n, 0);
            for (unsigned long long i = 0; i < n; ++i) h1[(size_t)i] = x[(size_t)bitrev_idx(i)];
            h2 = h1;
            for (int s = logn - 1; s >= 0; --s) {
                const unsigned long long half = 1ull << s;
                const unsigned long long stride = 2 * half;
                for (unsigned long long g = 0; g < n / stride; ++g) {
                    for (unsigned long long j = 0; j < half; ++j) {
                        const unsigned long long i0 = g * stride + j, i1 = i0 + half;
                        const unsigned long long twv =
                            gl_pow_host(om, (j * (n / stride)) % n);
                        const unsigned long long u = h2[(size_t)i0], v2 = h2[(size_t)i1];
                        h2[(size_t)i0] = gl_add_host(u, v2);
                        h2[(size_t)i1] = gl_mul_host(gl_sub_host(u, v2), twv);
                    }
                }
            }
            std::vector<unsigned long long> h3(n, 0);
            for (unsigned long long i = 0; i < n; ++i) h3[(size_t)i] = h2[(size_t)bitrev_idx(i)];
            int hb = 0;
            for (unsigned long long i = 0; i < n; ++i) {
                if (sp[(size_t)i] != h3[(size_t)i]) {
                    if (hb < 3) {
                        std::printf("  model mismatch i=%llu: device=%llu model=%llu\n",
                                    (unsigned long long)i, sp[(size_t)i], h3[(size_t)i]);
                    }
                    ++hb;
                }
            }
            std::printf("nttcheck: device vs host-model(bitrev,DIF,bitrev): bad=%d/%llu\n", hb,
                        (unsigned long long)n);
            {
                std::vector<unsigned long long> sq(n, 0);
                CK(cudaMemcpy(sq.data(), dx, n * 8, cudaMemcpyDeviceToHost));
                std::printf("  dbg input buffer after fwd: %llu %llu   out buffer: %llu %llu %llu\n",
                            sq[0], sq[1], sp[0], sp[1], sp[2]);
                std::vector<unsigned long long> sd(n, 0);
                CK(cudaMemcpy(sd.data(), dsp, n * 8, cudaMemcpyDeviceToHost));
                std::printf("  dbg dsp direct: %llu %llu %llu\n", sd[0], sd[1], sd[2]);
                std::vector<unsigned long long> sm(n, 0);
                CK(cudaMemcpy(sm.data(), dsave, n * 8, cudaMemcpyDeviceToHost));
                std::printf("  dbg dsave (input copy): %llu %llu\n", sm[0], sm[1]);
            }
            int hd = 0;
            for (unsigned long long i = 0; i < n; ++i) {
                unsigned long long acc = 0;
                for (unsigned long long j = 0; j < n; ++j) {
                    acc = gl_add_host(acc, gl_mul_host(x[(size_t)j], gl_pow_host(om, (j * i) % n)));
                }
                if (ref[(size_t)i] != acc) ++hd;
            }
            std::printf("nttcheck: oracle check (again) bad=%d\n", hd);
        }
        int bad1 = 0, bad3 = 0;
        /* FIRST: is the on-device DFT oracle itself right?  Compare ref[k] against a host
           sum computed with the (GMP-verified) host arithmetic. */
        {
            int oracle_bad = 0;
            for (unsigned long long kk = 0; kk < n; ++kk) {
                unsigned long long acc = 0;
                for (unsigned long long j = 0; j < n; ++j) {
                    acc = gl_add_host(acc, gl_mul_host(x[(size_t)j],
                                                       gl_pow_host(om, (j * kk) % n)));
                }
                if (acc != ref[(size_t)kk]) {
                    if (oracle_bad < 3) {
                        std::printf("  ORACLE mismatch k=%llu: device=%llu host=%llu\n",
                                    (unsigned long long)kk, ref[(size_t)kk], acc);
                    }
                    ++oracle_bad;
                }
            }
            std::printf("nttcheck: direct-DFT oracle vs host sums: bad=%d/%llu\n", oracle_bad,
                        (unsigned long long)n);
        }
        /* which convention did the forward actually produce?  Count matches under both
           hypotheses: sp[i] == X_bitrev(i) (bit-reversed output) and sp[i] == X_i (natural). */
        {
            int as_bitrev = 0, as_natural = 0;
            for (unsigned long long i = 0; i < n; ++i) {
                unsigned long long r = 0, v = i;
                for (int q = 0; q < logn; ++q) { r = (r << 1) | (v & 1ull); v >>= 1; }
                if (sp[(size_t)i] == ref[(size_t)r]) ++as_bitrev;
                if (sp[(size_t)i] == ref[(size_t)i]) ++as_natural;
            }
            std::printf("  dbg convention: matches-as-bitrev=%d/%llu matches-as-natural=%d/%llu\n",
                        as_bitrev, (unsigned long long)n, as_natural, (unsigned long long)n);
            if (n <= 16) {
                std::printf("  dbg sp:");
                for (unsigned long long i = 0; i < n; ++i) std::printf(" %llu", sp[(size_t)i]);
                std::printf("\n  dbg ref:");
                for (unsigned long long i = 0; i < n; ++i) std::printf(" %llu", ref[(size_t)i]);
                std::printf("\n  dbg x:");
                for (unsigned long long i = 0; i < n; ++i) std::printf(" %llu", x[(size_t)i]);
                std::printf("\n");
            }
        }
        for (unsigned long long i = 0; i < n; ++i) {
            unsigned long long r = 0, v = i;
            for (int q = 0; q < logn; ++q) { r = (r << 1) | (v & 1ull); v >>= 1; }
            if (sp[(size_t)i] != ref[(size_t)r]) {
                if (bad1 < 3) {
                    std::printf("  fwd mismatch i=%llu (point %llu): got %llu want %llu\n",
                                (unsigned long long)i, r, sp[(size_t)i], ref[(size_t)r]);
                }
                ++bad1;
            }
            if (tw[(size_t)i] != gl_pow_host(om, i)) ++bad3;
        }
        /* (2) inverse(forward(x)) == x, up to the 1/N factor which the pointwise kernel
           applies (omega^(i*nsc mod N) = omega^i / N) */
        /* stage-by-stage host mirror: find the first stage where the device diverges */
        {
            std::vector<unsigned long long> hs(n, 0);
            for (unsigned long long i = 0; i < n; ++i) {
                unsigned long long r = 0, v = i;
                for (int q = 0; q < logn; ++q) { r = (r << 1) | (v & 1ull); v >>= 1; }
                hs[(size_t)i] = x[(size_t)r];
            }
            for (int st = 0; st < logn; ++st) {
                const unsigned long long half = 1ull << st;
                const unsigned long long stride = 2 * half;
                const unsigned long long exp_step = n / (2 * half);
                for (unsigned long long g = 0; g < n / stride; ++g) {
                    for (unsigned long long j = 0; j < half; ++j) {
                        const unsigned long long i0 = g * stride + j;
                        const unsigned long long i1 = i0 + half;
                        const unsigned long long tw = gl_pow_host(om, (j * exp_step) % n);
                        const unsigned long long u = hs[(size_t)i0], v2 = hs[(size_t)i1];
                        hs[(size_t)i0] = gl_add_host(u, v2);
                        hs[(size_t)i1] = gl_mul_host(gl_sub_host(u, v2), tw);
                    }
                }
            }
            int mirror_bad = 0;
            for (unsigned long long i = 0; i < n; ++i) {
                if (sp[(size_t)i] != hs[(size_t)i]) {
                    if (mirror_bad < 3) {
                        std::printf("  mirror mismatch i=%llu: device=%llu hostmirror=%llu\n",
                                    (unsigned long long)i, sp[(size_t)i], hs[(size_t)i]);
                    }
                    ++mirror_bad;
                }
            }
            std::printf("nttcheck: host DIT mirror mismatches=%d/%llu\n", mirror_bad,
                        (unsigned long long)n);
        }
        CK(cudaMemcpy(dx, sp.data(), n * 8, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dback, sp.data(), n * 8, cudaMemcpyHostToDevice));
        {
            std::vector<unsigned long long> ones(n, 1ull);
            unsigned long long *dones = nullptr;
            CK(cudaMalloc(&dones, n * 8));
            CK(cudaMemcpy(dones, ones.data(), n * 8, cudaMemcpyHostToDevice));
            ntt_inverse_fused(dback, dones, fc, omi, nsc);   /* fused (pointwise by ones) */
            (void)dones;
            CK(cudaGetLastError());
            cudaFree(dones);
        }
        CK(cudaDeviceSynchronize());
        std::vector<unsigned long long> back(n, 0);
        CK(cudaMemcpy(back.data(), dback, n * 8, cudaMemcpyDeviceToHost));
        int bad2 = 0;
        for (unsigned long long i = 0; i < n; ++i) {
            if (back[(size_t)i] != x[(size_t)i]) {
                if (bad2 < 3) {
                    std::printf("  roundtrip mismatch i=%llu: got %llu want %llu\n",
                                (unsigned long long)i, back[(size_t)i], x[(size_t)i]);
                }
                ++bad2;
            }
        }
        std::printf("nttcheck: logn=%d  forward-vs-directDFT bad=%d/%llu  roundtrip bad=%d/%llu"
                    "  twiddle bad=%d/%llu\n", logn, bad1, (unsigned long long)n, bad2,
                    (unsigned long long)n, bad3, (unsigned long long)n);
        fuse_release(fc);
        cudaFree(dx); cudaFree(dsp); cudaFree(dref); cudaFree(dtw); cudaFree(dback);
        cudaFree(dsave);
        return (bad1 || bad2 || bad3) ? 1 : 0;
    }

    /*
     * fusecheck <log2 N> [device] -- does the STAGE-FUSED transform produce exactly the same
     * numbers as the per-stage (one kernel per stage) implementation it replaced?  This is
     * the cheapest way to localise a fusion bug: the transform's meaning (DIF descending, DIT
     * ascending, bit-reversed spectrum in between, omega vs omega^-1) is unchanged, so any
     * difference is an index/twiddle error in one of the fused passes.
     *   fwd: per-stage forward  vs fused forward                (must be bit-identical)
     *   inv: per-stage inverse  vs fused inverse with B = all ones, which reproduces the
     *        pointwise product and the 1/N scale the fused inverse absorbs  (bit-identical)
     */
    if (mode == "fusecheck") {
        const int logn = (argc > 2) ? std::atoi(argv[2]) : 12;
        const int dev = (argc > 3) ? std::atoi(argv[3]) : 1;
        CK(cudaSetDevice(dev));
        const unsigned long long n = 1ull << logn;
        const unsigned long long om = gl_pow_host(7ull, (GL_P - 1ull) / n);
        const unsigned long long omi = gl_pow_host(om, GL_P - 2ull);
        const unsigned long long nsc = gl_pow_host(n % GL_P, GL_P - 2ull);
        std::vector<unsigned long long> x(n, 0);
        unsigned long long s = 0xC0FFEEull;
        for (unsigned long long i = 0; i < n; ++i) {
            s = s * 6364136223846793005ull + 1442695040888963407ull;
            x[(size_t)i] = s % GL_P;
        }
        unsigned long long *dOld = nullptr, *dNew = nullptr, *dOnes = nullptr, *dScr = nullptr;
        CK(cudaMalloc(&dOld, n * 8));
        CK(cudaMalloc(&dNew, n * 8));
        CK(cudaMalloc(&dOnes, n * 8));
        CK(cudaMalloc(&dScr, n * 8));
        std::vector<unsigned long long> ones(n, 1ull);
        CK(cudaMemcpy(dOnes, ones.data(), n * 8, cudaMemcpyHostToDevice));
        FuseCtx fc;
        fuse_init(fc, n, logn, om, omi);
        std::printf("fusecheck: logn=%d n=%llu tile_t=%d outer=", logn,
                    (unsigned long long)n, fc.t);
        for (int p = 0; p < fc.nms; ++p) std::printf(" radix-%d", 1 << fc.ms[p]);
        std::printf(" -> %d passes per transform, %d total\n", fc.passes_fwd,
                    3 * fc.passes_fwd + 1);
        std::vector<unsigned long long> a(n, 0), b(n, 0);
        int badF = 0, badI = 0;
        unsigned long long firstF = 0, firstI = 0;
        CK(cudaMemcpy(dOld, x.data(), n * 8, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dNew, x.data(), n * 8, cudaMemcpyHostToDevice));
        ntt_forward(dOld, n, logn, om, dScr);          /* per-stage reference */
        ntt_forward_fused(dNew, fc, om);               /* fused */
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(a.data(), dOld, n * 8, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(b.data(), dNew, n * 8, cudaMemcpyDeviceToHost));
        for (unsigned long long i = 0; i < n; ++i) {
            if (a[(size_t)i] != b[(size_t)i]) { if (!badF) firstF = i; ++badF; }
        }
        /* inverse: write the forward result back into BOTH buffers, then compare */
        CK(cudaMemcpy(dOld, b.data(), n * 8, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dNew, b.data(), n * 8, cudaMemcpyHostToDevice));
        ntt_inverse(dOld, n, logn, omi, nsc, dScr);    /* per-stage reference */
        ntt_inverse_fused(dNew, dOnes, fc, omi, nsc);  /* fused, pointwise by ones */
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(a.data(), dOld, n * 8, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(b.data(), dNew, n * 8, cudaMemcpyDeviceToHost));
        for (unsigned long long i = 0; i < n; ++i) {
            if (a[(size_t)i] != b[(size_t)i]) { if (!badI) firstI = i; ++badI; }
        }
        std::printf("fusecheck: forward per-stage-vs-fused bad=%d/%llu (first=%llu)  "
                    "inverse bad=%d/%llu (first=%llu)\n", badF, (unsigned long long)n,
                    firstF, badI, (unsigned long long)n, firstI);
        fuse_release(fc);
        cudaFree(dOld); cudaFree(dNew); cudaFree(dOnes); cudaFree(dScr);
        return (badF || badI) ? 1 : 0;
    }

    if ((mode == "poly" || mode == "pipedump") && argc > 2) {
        const unsigned long long P = std::strtoull(argv[2], nullptr, 10);
        const int S = (argc > 3) ? std::atoi(argv[3]) : 5153;
        const int dev = (argc > 4) ? std::atoi(argv[4]) : 0;
        const bool verify = (argc > 5) ? (std::atoi(argv[5]) != 0) : true;
        return run_poly(P, S, dev, verify, (mode == "pipedump") ? 1 : 0);
    }
    if (mode == "bench" && argc > 2) {
        const unsigned long long bits = std::strtoull(argv[2], nullptr, 10);
        /* bench: an optional bpw override, then the device (same CLI shape as the cuFFT
           probe's `bench <bits> [chunk_bits] [device]`) */
        int bpw = (argc > 3) ? std::atoi(argv[3]) : 0;
        int dev = (argc > 4) ? std::atoi(argv[4]) : 0;
        return run_bench(bits, bpw, dev);
    }
    std::fprintf(stderr,
                 "usage: ntt_poly_probe poly <P> <S> [device] [verify]\n"
                 "       ntt_poly_probe pipedump <P> <S> [device] [verify]  (digit dump)\n"
                 "       ntt_poly_probe nttcheck <log2N> [device]           (DFT oracle)\n"
                 "       ntt_poly_probe fusecheck <log2N> [device]          (fused vs per-stage)\n"
                 "       ntt_poly_probe bench <bits> [bpw] [device]\n");
    return 2;
}
#endif /* NTT_POLY_PROBE_NO_MAIN */
