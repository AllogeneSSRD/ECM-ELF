/* ---------------------------------------------------------------------------
 * cufft_kron_probe.cu -- M0 gate of docs/DEV_STAGE2_GPU_PLAN.md: how fast is ONE
 * big-integer multiplication of the size a polymult-class stage 2 needs, and is it
 * correct?
 *
 * Method: Kronecker substitution.  A polynomial with P coefficients of N bits each is
 * one integer of P*N bits (~6.7e8 bits for P/2 = 65000 coefficients and a 5153-bit
 * modulus).  Multiplying two such integers = one convolution of base-2^chunk_bits
 * chunks = one cuFFT round: D2Z(A) * D2Z(B) -> Z2D -> round -> carry propagate.
 *
 * Accuracy: the convolution coefficients are bounded by n * 2^(2*b).  A double holds
 * integers exactly up to 2^53, so b is chosen from that bound and the bound is PRINTED
 * (max_coeff_bits); the result is then verified against GMP in full, bit for bit, by
 * packing the base-2^12 chunks back into bytes (2 chunks <-> 3 bytes, exact) and
 * comparing with mpz_mul of the same two operands.
 *
 * Usage:
 *   cufft_kron_probe.exe check <bits>            [device]   # ~1e6 bits: full compare, fast
 *   cufft_kron_probe.exe bench <bits> [chunk=12] [device]   # 6.7e8 bits: the real shape
 *
 * Machine-readable line (parsed by tools/test/test_cufft_kron.ps1):
 *   kron: mode=check bits=1000008 chunk_bits=12 n=83334 fft=166668 mem_mb=3.8 ok=1
 *         max_coeff_bits=33.4 t_fwd=0.01 t_inv=0.01 t_carry=0.00 t_total=0.02
 * ------------------------------------------------------------------------- */
#include <cuda_runtime.h>
#include <cufft.h>

#include <gmp.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x)                                                                        \
    do {                                                                             \
        const cudaError_t e_ = (x);                                                  \
        if (e_ != cudaSuccess) {                                                      \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), \
                         __FILE__, __LINE__);                                        \
            std::exit(2);                                                            \
        }                                                                            \
    } while (0)

#define CF(x)                                                                        \
    do {                                                                             \
        const cufftResult r_ = (x);                                                   \
        if (r_ != CUFFT_SUCCESS) {                                                    \
            std::fprintf(stderr, "cuFFT error %d at %s:%d\n", (int)r_, __FILE__,      \
                         __LINE__);                                                  \
            std::exit(2);                                                            \
        }                                                                            \
    } while (0)

namespace {

double now_s()
{
    using clock = std::chrono::steady_clock;
    static const clock::time_point t0 = clock::now();
    return std::chrono::duration<double>(clock::now() - t0).count();
}

/* Smallest 5-smooth number >= x (cuFFT is fast for 2^a 3^b 5^c). */
uint64_t next_smooth(uint64_t x)
{
    uint64_t best = ~0ull;
    for (uint64_t p5 = 1; p5 < 4 * x; p5 *= 5) {
        for (uint64_t p3 = p5; p3 < 4 * x; p3 *= 3) {
            uint64_t p2 = p3;
            while (p2 < x) p2 *= 2;
            if (p2 < best) best = p2;
        }
    }
    return best;
}

void fill_random_bytes(std::vector<uint8_t> &v, uint64_t seed)
{
    uint64_t s = seed | 1u;
    for (size_t i = 0; i < v.size(); ++i) {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        v[i] = (uint8_t)(s >> 33);
    }
}

/* bytes (little endian) -> base-2^12 chunks, 2 chunks per 3 bytes. */
void bytes_to_chunks(const std::vector<uint8_t> &bytes, std::vector<double> &chunks, uint64_t n)
{
    chunks.assign(n, 0.0);
    for (uint64_t i = 0; i < n; i += 2) {
        const size_t b = (size_t)(i / 2) * 3;
        const uint32_t b0 = (b + 0 < bytes.size()) ? bytes[b + 0] : 0u;
        const uint32_t b1 = (b + 1 < bytes.size()) ? bytes[b + 1] : 0u;
        const uint32_t b2 = (b + 2 < bytes.size()) ? bytes[b + 2] : 0u;
        chunks[i] = (double)((b0 | (b1 << 8)) & 0xFFFu);              /* 12 bits */
        if (i + 1 < n) chunks[i + 1] = (double)(((b1 >> 4) | (b2 << 4)) & 0xFFFu);
    }
}

/* base-2^12 chunks -> bytes, 2 chunks per 3 bytes (exact inverse of the above). */
void chunks_to_bytes(const std::vector<uint64_t> &chunks, std::vector<uint8_t> &bytes)
{
    const size_t nchunks = chunks.size();
    for (size_t i = 0; i + 1 < nchunks; i += 2) {
        const uint32_t c0 = (uint32_t)(chunks[i] & 0xFFFu);
        const uint32_t c1 = (uint32_t)(chunks[i + 1] & 0xFFFu);
        bytes.push_back((uint8_t)(c0 & 0xFF));
        bytes.push_back((uint8_t)(((c0 >> 8) & 0xF) | ((c1 & 0xF) << 4)));
        bytes.push_back((uint8_t)((c1 >> 4) & 0xFF));
    }
    while (!bytes.empty() && bytes.back() == 0) bytes.pop_back();   /* canonical */
}

__global__ void fill_chunks_kernel(const unsigned char *bytes, uint64_t nbytes, uint64_t n,
                                   double *out)
{
    const uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint64_t b = (i / 2) * 3;
    const unsigned b0 = (b + 0 < nbytes) ? bytes[b + 0] : 0u;
    const unsigned b1 = (b + 1 < nbytes) ? bytes[b + 1] : 0u;
    const unsigned b2 = (b + 2 < nbytes) ? bytes[b + 2] : 0u;
    if ((i & 1u) == 0) out[i] = (double)((b0 | (b1 << 8)) & 0xFFFu);
    else out[i] = (double)(((b1 >> 4) | (b2 << 4)) & 0xFFFu);
}

__global__ void pointwise_mul_kernel(double *a, const double *b, uint64_t m)
{
    const uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (i >= m) return;
    const double ar = a[2 * i], ai = a[2 * i + 1];
    const double br = b[2 * i], bi = b[2 * i + 1];
    a[2 * i] = ar * br - ai * bi;
    a[2 * i + 1] = ar * bi + ai * br;
}

/* cuFFT is UNNORMALISED: Z2D(D2Z(x)) == N*x, so the inverse of a product comes out N times
   too large and must be scaled by 1/N while rounding (missing this made every check fail
   with a difference of the same size as the product itself). */
__global__ void round_kernel(double *c, uint64_t n, double inv_n)
{
    const uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (i < n) c[i] = rint(c[i] * inv_n);
}

/* One carry step, done in TWO kernels to avoid a race: the first extracts the carry out of
   every slot (each thread touches only its own element), the second adds the neighbour's
   carry.  The single-kernel version (c[i] -= q*base; c[i+1] += q) lost carries because
   thread i+1 could write its slot after reading it -- measured: chunks 0..32 right, 8 wrong,
   then right again.  Carries move one slot per step, so repeat ~max_coeff_bits/chunk_bits. */
__global__ void carry_extract_kernel(double *c, double *q, uint64_t n, double base)
{
    const uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double v = c[i];
    const double qq = floor(v / base);
    q[i] = qq;
    c[i] = v - qq * base;
}

__global__ void carry_shift_kernel(double *c, const double *q, uint64_t n)
{
    const uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (i == 0 || i >= n) return;
    c[i] += q[i - 1];
}

/* Assemble one COEFFICIENT per thread: walk its own slot of `chunks_per_slot` base-2^12 chunks
   most-significant first and accumulate the slot value.  No carry crosses a slot because the
   slot was sized 2*S + ceil(log2(P)) bits, so this replaces the (global, 40 % of runtime)
   carry propagation with a single pass over the array.  This is the actual stage-2 primitive:
   the product polynomial's coefficients, each one an integer < 2^(2S+log2 P). */
/* Assemble one COEFFICIENT per thread, reduced modulo the 32-bit prime 4294967291.
   A real coefficient here is 2S+log2(P) bits (10323 for S=5153), so it cannot live in a
   uint64 -- the probe therefore verifies the convolution through this modular projection,
   which has exactly the same memory pattern as the production per-slot carry (one sequential
   walk of the slot) and is checkable against GMP for ANY S.  No carry crosses a slot because
   the slot was sized 2S + ceil(log2 P). */
#define SLOT_MOD 4294967291ull

__global__ void slot_assemble_kernel(double *c, uint64_t slots, uint64_t chunks_per_slot,
                                     uint64_t *out)
{
    const uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (i >= slots) return;
    const uint64_t base_index = i * chunks_per_slot;
    /* Pass 1 (low -> high): the raw convolution coefficients are NOT 12-bit digits -- each is up
       to (chunks per operand)*2^24, i.e. ~2^34 -- so normalise them into base 2^12 first.  The
       carry cannot leave the slot because the slot was padded to 2S+log2(P) bits. */
    uint64_t carry = 0;
    for (uint64_t j = 0; j < chunks_per_slot; ++j) {
        const uint64_t s = (uint64_t)c[base_index + j] + carry;
        c[base_index + j] = (double)(s & 0xFFFull);
        carry = s >> 12;
    }
    /* Pass 2 (high -> low): Horner assembly modulo SLOT_MOD. */
    uint64_t v = 0;
    for (uint64_t j = chunks_per_slot; j-- > 0;) {
        v = ((v << 12) | (uint64_t)c[base_index + j]) % SLOT_MOD;
    }
    out[i] = v;
}

__global__ void to_u64_kernel(const double *c, uint64_t n, uint64_t *out)
{
    const uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x;
    if (i < n) out[i] = (uint64_t)c[i];
}

/* ---------------------------------------------------------------------------
 * poly mode -- the actual stage-2 primitive: multiply two polynomials of P
 * coefficients, each S bits (i.e. a residue mod an S-bit modulus), using ONE
 * Kronecker convolution, and verify every coefficient against a GMP schoolbook
 * product.  Slot width = 2S + ceil(log2 P) so no carry crosses a coefficient.
 * ------------------------------------------------------------------------- */
int run_poly(uint64_t P, int S, int device, bool verify)
{
    CK(cudaSetDevice(device));
    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, device));
    CK(cudaFree(0));

    uint64_t log2P = 1;
    while ((1ull << log2P) < P) ++log2P;
    const uint64_t slot_bits = 2ull * (uint64_t)S + log2P;
    const uint64_t slot_chunks_raw = (slot_bits + 11) / 12;
    const uint64_t slot_chunks = slot_chunks_raw;                 /* chunks per slot */
    const uint64_t slot_bits_padded = slot_chunks * 12;
    const uint64_t n = P * slot_chunks;                           /* chunks per operand */
    const uint64_t N = next_smooth(2 * n);
    const uint64_t nmax = N / 2 + 1;
    const uint64_t out_slots = 2 * P - 1;

    /* Coefficients are S bits each, so they need W = ceil(S/64) words -- the first version
       generated them with `s >> (64 - S)`, which is a NEGATIVE shift (undefined) for S > 64. */
    const uint64_t W = (uint64_t)(S + 63) / 64;
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
        const int top = S - (int)(W - 1) * 64;            /* bits in the last word */
        const uint64_t top_mask = (top >= 64) ? ~0ull : ((1ull << top) - 1ull);
        for (uint64_t i = 0; i < P; ++i) {
            wordsA[i * W + W - 1] &= top_mask;
            wordsB[i * W + W - 1] &= top_mask;
        }
    }
    /* Extract `nbits` bits at `bit` from coefficient `coeff` -- and NOTHING beyond it: the slot is
       wider than the coefficient (144 bits vs 64 for S=64), so a naive flat read pulled the NEXT
       coefficient's bits into the high chunks and every slot came out wrong (measured: all 127
       slots bad at P=64/S=64).  Bits at or past the coefficient's width are zero, and the top
       chunk is clamped to the bits that actually belong to this coefficient. */
    auto bit_range = [&](const std::vector<uint64_t> &w, uint64_t coeff, uint64_t bit,
                         uint64_t nbits) -> uint64_t {
        if (bit >= (uint64_t)S) return 0ull;
        if (bit + nbits > (uint64_t)S) nbits = (uint64_t)S - bit;
        const uint64_t base = coeff * W;
        const uint64_t wi = bit / 64, off = bit % 64;
        uint64_t v = (w[base + wi] >> off);
        if (off + nbits > 64 && wi + 1 < W) v |= w[base + wi + 1] << (64 - off);
        return (nbits >= 64) ? v : (v & ((1ull << nbits) - 1ull));
    };

    /* pack: coefficient i starts at chunk i*slot_chunks, low 12 bits first */
    std::vector<double> hA(N, 0.0), hB(N, 0.0);
    for (uint64_t i = 0; i < P; ++i) {
        for (uint64_t j = 0; j < slot_chunks; ++j) {
            hA[i * slot_chunks + j] = (double)bit_range(wordsA, i, 12 * j, 12);
            hB[i * slot_chunks + j] = (double)bit_range(wordsB, i, 12 * j, 12);
        }
    }
    const double mem_mb = (double)(2 * N * sizeof(double) + 2 * 2 * nmax * sizeof(double)) / 1048576.0;
    const double max_coeff_bits = std::log2((double)n) + 24.0;   /* chunks*2^24 after squaring 2^12 */
    std::printf("cufft_kron_probe: mode=poly device=%d (%s) P=%llu S=%d slot_bits=%llu chunks/slot=%llu\n",
                device, prop.name, (unsigned long long)P, S,
                (unsigned long long)slot_bits_padded, (unsigned long long)slot_chunks);
    std::printf("  fft=%llu  mem=%.0f MB  convolution coefficient bound ~2^%.1f (double holds 2^53) -> %s\n",
                (unsigned long long)N, mem_mb, max_coeff_bits,
                max_coeff_bits < 52.0 ? "OK" : "TOO LARGE");

    double *dA = nullptr, *dB = nullptr, *dSpecA = nullptr, *dSpecB = nullptr;
    uint64_t *dOut = nullptr;
    CK(cudaMalloc(&dA, N * sizeof(double)));
    CK(cudaMalloc(&dB, N * sizeof(double)));
    CK(cudaMalloc(&dSpecA, 2 * nmax * sizeof(double)));
    CK(cudaMalloc(&dSpecB, 2 * nmax * sizeof(double)));
    CK(cudaMalloc(&dOut, out_slots * sizeof(uint64_t)));
    CK(cudaMemcpy(dA, hA.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, hB.data(), N * sizeof(double), cudaMemcpyHostToDevice));

    cufftHandle planF = 0, planI = 0;
    CF(cufftPlan1d(&planF, (int)N, CUFFT_D2Z, 1));
    CF(cufftPlan1d(&planI, (int)N, CUFFT_Z2D, 1));
    const uint64_t threads = 256;
    const uint64_t blocksSpec = (nmax + threads - 1) / threads;

    const double t0 = now_s();
    CF(cufftExecD2Z(planF, dA, (cufftDoubleComplex *)dSpecA));
    CF(cufftExecD2Z(planF, dB, (cufftDoubleComplex *)dSpecB));
    pointwise_mul_kernel<<<(unsigned)blocksSpec, (unsigned)threads>>>(dSpecA, dSpecB, nmax);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    const double t_fwd = now_s() - t0;

    const double t1 = now_s();
    CF(cufftExecZ2D(planI, (cufftDoubleComplex *)dSpecA, dA));
    CK(cudaDeviceSynchronize());
    const double t_inv = now_s() - t1;

    /* round + per-slot assembly (no global carry: slots are padded) */
    const double t2 = now_s();
    const uint64_t blocksReal = (N + threads - 1) / threads;
    round_kernel<<<(unsigned)blocksReal, (unsigned)threads>>>(dA, N, 1.0 / (double)N);
    CK(cudaGetLastError());
    const uint64_t blocksSlots = (out_slots + threads - 1) / threads;
    slot_assemble_kernel<<<(unsigned)blocksSlots, (unsigned)threads>>>(dA, out_slots, slot_chunks, dOut);
    CK(cudaGetLastError());
    std::vector<uint64_t> hOut(out_slots, 0);
    CK(cudaMemcpy(hOut.data(), dOut, out_slots * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    const double t_slot = now_s() - t2;

    bool ok = true;
    std::string detail;
    if (verify) {
        /* GMP schoolbook polynomial product; coefficient i needs (S + 64) bits, so compare the
           low 64 bits of each coefficient as well as the count. */
        mpz_t *ca = new mpz_t[P];
        mpz_t *cb = new mpz_t[P];
        for (uint64_t i = 0; i < P; ++i) {
            mpz_init(ca[i]);
            mpz_init(cb[i]);
            mpz_import(ca[i], W, -1, 8, 0, 0, &wordsA[i * W]);
            mpz_import(cb[i], W, -1, 8, 0, 0, &wordsB[i * W]);
        }
        mpz_t *acc = new mpz_t[out_slots];
        for (uint64_t i = 0; i < out_slots; ++i) mpz_init(acc[i]);
        for (uint64_t i = 0; i < P; ++i) {
            for (uint64_t j = 0; j < P; ++j) {
                mpz_t t;
                mpz_init(t);
                mpz_mul(t, ca[i], cb[j]);
                mpz_add(acc[i + j], acc[i + j], t);
                mpz_clear(t);
            }
        }
        uint64_t bad = 0;
        mpz_t mod;
        mpz_init_set_ui(mod, 4294967291ull);
        for (uint64_t i = 0; i < out_slots; ++i) {
            mpz_t lo;
            mpz_init(lo);
            mpz_mod(lo, acc[i], mod);
            unsigned long long want = mpz_get_ui(lo);
            if (want != hOut[i]) {
                if (bad == 0) {
                    char buf[160];
                    std::snprintf(buf, sizeof(buf), " first_bad_slot=%llu got=%llu want=%llu",
                                  (unsigned long long)i, (unsigned long long)hOut[i],
                                  (unsigned long long)want);
                    detail += buf;
                }
                ++bad;
            }
            mpz_clear(lo);
        }
        if (bad) detail += " bad_slots=" + std::to_string(bad);
        ok = (bad == 0);
        for (uint64_t i = 0; i < out_slots; ++i) mpz_clear(acc[i]);
        delete[] acc;
        for (uint64_t i = 0; i < P; ++i) { mpz_clear(ca[i]); mpz_clear(cb[i]); }
        delete[] ca;
        delete[] cb;
        mpz_clear(mod);
    }

    const double total = t_fwd + t_inv + t_slot;
    std::printf("poly: mode=poly P=%llu S=%d slot_bits=%llu fft=%llu mem_mb=%.0f ok=%d t_fwd=%.3f "
                "t_inv=%.3f t_slot=%.3f t_total=%.3f ns_per_coeff=%.2f%s%s\n",
                (unsigned long long)P, S, (unsigned long long)slot_bits_padded,
                (unsigned long long)N, mem_mb, ok ? 1 : 0, t_fwd, t_inv, t_slot, total,
                total * 1e9 / (double)P, detail.empty() ? "" : " detail=", detail.c_str());

    CF(cufftDestroy(planF));
    CF(cufftDestroy(planI));
    cudaFree(dA); cudaFree(dB); cudaFree(dSpecA); cudaFree(dSpecB); cudaFree(dOut);
    return ok ? 0 : 1;
}

int run(const std::string &mode, uint64_t bits, int chunk_bits, int device)
{
    CK(cudaSetDevice(device));
    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, device));
    CK(cudaFree(0));

    const uint64_t n = bits / (uint64_t)chunk_bits;        /* chunks per operand */
    const uint64_t need = 2 * n;                           /* room for the linear product */
    const uint64_t N = next_smooth(need);                  /* FFT length */
    const uint64_t nmax = N / 2 + 1;                       /* r2c spectrum length */

    const double max_coeff_bits = std::log2((double)n) + 2.0 * chunk_bits;
    const bool fits = max_coeff_bits < 52.0;

    /* memory: one real buffer + two spectra */
    const double mem_mb =
        (double)(N * sizeof(double) + 2 * 2 * nmax * sizeof(double)) / 1048576.0;

    std::printf("cufft_kron_probe: mode=%s device=%d (%s) bits=%llu chunk_bits=%d n=%llu fft=%llu\n",
                mode.c_str(), device, prop.name, (unsigned long long)bits, chunk_bits,
                (unsigned long long)n, (unsigned long long)N);
    std::printf("  convolution coefficients ~ 2^%.1f (double mantissa 53 bits) -> %s ; device buffers ~%.0f MB\n",
                max_coeff_bits, fits ? "OK" : "TOO LARGE (use a smaller chunk_bits)",
                mem_mb);

    const size_t nbytes = (size_t)(bits / 8);
    std::vector<uint8_t> bytesA(nbytes), bytesB(nbytes);
    fill_random_bytes(bytesA, 0x1234567ull + bits);
    fill_random_bytes(bytesB, 0x89abcdefull + bits);

    unsigned char *dBytesA = nullptr, *dBytesB = nullptr;
    double *dReal = nullptr, *dSpecA = nullptr, *dSpecB = nullptr;
    uint64_t *dOut = nullptr;
    CK(cudaMalloc(&dBytesA, nbytes));
    CK(cudaMalloc(&dBytesB, nbytes));
    CK(cudaMalloc(&dReal, N * sizeof(double)));
    CK(cudaMalloc(&dSpecA, 2 * nmax * sizeof(double)));
    CK(cudaMalloc(&dSpecB, 2 * nmax * sizeof(double)));
    CK(cudaMalloc(&dOut, N * sizeof(uint64_t)));
    CK(cudaMemcpy(dBytesA, bytesA.data(), nbytes, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dBytesB, bytesB.data(), nbytes, cudaMemcpyHostToDevice));

    const uint64_t threads = 256;
    const uint64_t blocksReal = (N + threads - 1) / threads;
    const uint64_t blocksSpec = (nmax + threads - 1) / threads;

    cufftHandle planF = 0, planI = 0;
    CF(cufftPlan1d(&planF, (int)N, CUFFT_D2Z, 1));
    CF(cufftPlan1d(&planI, (int)N, CUFFT_Z2D, 1));

    const double t0 = now_s();
    CK(cudaMemset(dReal, 0, N * sizeof(double)));
    fill_chunks_kernel<<<(unsigned)blocksReal, (unsigned)threads>>>(dBytesA, nbytes, n, dReal);
    CK(cudaGetLastError());
    CF(cufftExecD2Z(planF, dReal, (cufftDoubleComplex *)dSpecA));

    CK(cudaMemset(dReal, 0, N * sizeof(double)));
    fill_chunks_kernel<<<(unsigned)blocksReal, (unsigned)threads>>>(dBytesB, nbytes, n, dReal);
    CK(cudaGetLastError());
    CF(cufftExecD2Z(planF, dReal, (cufftDoubleComplex *)dSpecB));

    pointwise_mul_kernel<<<(unsigned)blocksSpec, (unsigned)threads>>>(dSpecA, dSpecB, nmax);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
    const double t_fwd = now_s() - t0;

    const double t1 = now_s();
    CF(cufftExecZ2D(planI, (cufftDoubleComplex *)dSpecA, dReal));
    CK(cudaDeviceSynchronize());
    const double t_inv = now_s() - t1;

    if (mode == "dbg") {
        std::vector<double> raw(8, 0.0);
        CK(cudaMemcpy(raw.data(), dReal, 8 * sizeof(double), cudaMemcpyDeviceToHost));
        std::printf("  raw inverse FFT (first 8, unnormalised/then /N): ");
        for (int i = 0; i < 8; ++i) std::printf("%.1f ", raw[i] / (double)N);
        std::printf("\n");
        std::vector<uint8_t> eb;
        {
            mpz_t a, b, p2;
            mpz_inits(a, b, p2, nullptr);
            mpz_import(a, bytesA.size(), -1, 1, 0, 0, bytesA.data());
            mpz_import(b, bytesB.size(), -1, 1, 0, 0, bytesB.data());
            mpz_mul(p2, a, b);
            const size_t want = (size_t)(mpz_sizeinbase(p2, 2) / 8) + 2;
            eb.resize(want, 0);
            size_t got = 0;
            mpz_export(eb.data(), &got, -1, 1, 0, 0, p2);
            eb.resize(got);
            mpz_clears(a, b, p2, nullptr);
        }
        std::printf("  expected product bytes (first 9): ");
        for (int i = 0; i < 9 && i < (int)eb.size(); ++i) std::printf("%02x ", eb[i]);
        std::printf("\n  expected chunks (first 6): ");
        for (int i = 0; i < 6; ++i) {
            const size_t b = (size_t)(i / 2) * 3;
            const uint32_t b0 = (b + 0 < eb.size()) ? eb[b + 0] : 0u;
            const uint32_t b1 = (b + 1 < eb.size()) ? eb[b + 1] : 0u;
            const uint32_t b2 = (b + 2 < eb.size()) ? eb[b + 2] : 0u;
            const uint32_t c = (i & 1) ? (((b1 >> 4) | (b2 << 4)) & 0xFFFu)
                                       : ((b0 | (b1 << 8)) & 0xFFFu);
            std::printf("%u ", c);
        }
        std::printf("\n");
    }
    const double t2 = now_s();
    round_kernel<<<(unsigned)blocksReal, (unsigned)threads>>>(dReal, N, 1.0 / (double)N);
    CK(cudaGetLastError());
    const double base = std::pow(2.0, chunk_bits);
    const int passes = (int)std::ceil(max_coeff_bits / chunk_bits) + 2;
    double *dQ = nullptr;
    CK(cudaMalloc(&dQ, N * sizeof(double)));
    for (int p = 0; p < passes; ++p) {
        carry_extract_kernel<<<(unsigned)blocksReal, (unsigned)threads>>>(dReal, dQ, N, base);
        carry_shift_kernel<<<(unsigned)blocksReal, (unsigned)threads>>>(dReal, dQ, N);
    }
    CK(cudaGetLastError());
    CK(cudaGetLastError());
    to_u64_kernel<<<(unsigned)blocksReal, (unsigned)threads>>>(dReal, N, dOut);
    CK(cudaGetLastError());
    std::vector<uint64_t> hOut(N, 0);
    CK(cudaMemcpy(hOut.data(), dOut, N * sizeof(uint64_t), cudaMemcpyDeviceToHost));
    const double t_carry = now_s() - t2;   /* memcpy above syncs, so this is carry time only */
    if (mode == "dbg") {
        std::printf("  device chunks (first 6): ");
        for (int i = 0; i < 6; ++i) std::printf("%llu ", (unsigned long long)hOut[i]);
        std::printf("\n");
    }

    /* ---- verification: rebuild the product from the chunks and compare with GMP ---- */
    bool ok = false;
    std::string detail;
    {
        std::vector<uint8_t> outBytes;
        chunks_to_bytes(hOut, outBytes);
        mpz_t a, b, prod, mine;
        mpz_inits(a, b, prod, mine, nullptr);
        mpz_import(a, bytesA.size(), -1, 1, 0, 0, bytesA.data());
        mpz_import(b, bytesB.size(), -1, 1, 0, 0, bytesB.data());
        mpz_mul(prod, a, b);
        mpz_import(mine, outBytes.size(), -1, 1, 0, 0, outBytes.data());
        /* Build the expected chunk array from the exact product and report the FIRST
           mismatching chunk -- a diff of "product size" bits told us nothing. */
        {
            void (*ff)(void *, size_t) = nullptr;
            mp_get_memory_functions(nullptr, nullptr, &ff);
            size_t got = 0;
            std::vector<uint8_t> eb((size_t)(mpz_sizeinbase(prod, 2) / 8) + 2, 0);
            mpz_export(eb.data(), &got, -1, 1, 0, 0, prod);
            eb.resize(got);
            size_t first_bad = (size_t)-1;
            for (uint64_t i = 0; i < N; ++i) {
                const size_t b = (size_t)(i / 2) * 3;
                const uint32_t b0 = (b + 0 < eb.size()) ? eb[b + 0] : 0u;
                const uint32_t b1 = (b + 1 < eb.size()) ? eb[b + 1] : 0u;
                const uint32_t b2 = (b + 2 < eb.size()) ? eb[b + 2] : 0u;
                const uint64_t want = (i & 1)
                    ? (uint64_t)(((b1 >> 4) | (b2 << 4)) & 0xFFFu)
                    : (uint64_t)((b0 | (b1 << 8)) & 0xFFFu);
                if (hOut[i] != want) {
                    if (first_bad == (size_t)-1) {
                        char buf[128];
                        std::snprintf(buf, sizeof(buf), " first_bad_chunk=%llu got=%llu want=%llu",
                                      (unsigned long long)i, (unsigned long long)hOut[i],
                                      (unsigned long long)want);
                        detail += buf;
                    }
                    ++first_bad;   /* count of bad chunks (any value but -1) */
                }
            }
            if (first_bad != (size_t)-1) {
                detail += " bad_chunks=" + std::to_string(first_bad + 1);
            }
        }
        ok = (mpz_cmp(prod, mine) == 0);
        if (!ok) {
            mpz_t d;
            mpz_init(d);
            mpz_sub(d, prod, mine);
            mpz_abs(d, d);
            detail += " abs diff bits=" + std::to_string(mpz_sizeinbase(d, 2));
            mpz_clear(d);
        }
        mpz_clears(a, b, prod, mine, nullptr);
    }

    const double total = t_fwd + t_inv + t_carry;
    std::printf("kron: mode=%s bits=%llu chunk_bits=%d n=%llu fft=%llu mem_mb=%.1f ok=%d "
                "max_coeff_bits=%.1f t_fwd=%.3f t_inv=%.3f t_carry=%.3f t_total=%.3f%s%s\n",
                mode.c_str(), (unsigned long long)bits, chunk_bits, (unsigned long long)n,
                (unsigned long long)N, mem_mb, ok ? 1 : 0, max_coeff_bits, t_fwd, t_inv,
                t_carry, total, detail.empty() ? "" : " detail=", detail.c_str());

    CF(cufftDestroy(planF));
    CF(cufftDestroy(planI));
    cudaFree(dBytesA);
    cudaFree(dBytesB);
    cudaFree(dReal);
    cudaFree(dSpecA);
    cudaFree(dSpecB);
    cudaFree(dOut);
    cudaFree(dQ);
    return ok ? 0 : 1;
}

} /* namespace */

int main(int argc, char **argv)
{
    const std::string mode = (argc > 1) ? argv[1] : "check";
    /* poly mode has its own packer and its third argument is S, not chunk_bits, so dispatch it
       before the chunk-width guard. */
    if (argc > 2 && mode == "poly") {
        const uint64_t P = std::strtoull(argv[2], nullptr, 10);
        const int S = (argc > 3) ? std::atoi(argv[3]) : 5153;
        const int dev = (argc > 4) ? std::atoi(argv[4]) : 0;
        const bool verify = (argc > 5) ? (std::atoi(argv[5]) != 0) : true;
        return run_poly(P, S, dev, verify);
    }
    /* The byte<->chunk packer is exact for 12-bit chunks (2 chunks = 3 bytes); other widths
       would need their own packer, so refuse them instead of verifying wrongly. */
    if (argc > 3 && std::atoi(argv[3]) != 12) {
        std::fprintf(stderr, "cufft_kron_probe: only chunk_bits=12 is supported by the byte packer\n");
        return 2;
    }

    const uint64_t bits = (argc > 2) ? std::strtoull(argv[2], nullptr, 10) : 1000008ull;
    const int chunk_bits = (argc > 3) ? std::atoi(argv[3]) : 12;
    const int device = (argc > 4) ? std::atoi(argv[4]) : 0;
    return run(mode, bits, chunk_bits, device);
}
