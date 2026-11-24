#include "../two_axis.h"
#include "../../../cuda/ntt/runtime.h"
#include "two_axis.cuh"

#include <array>
#include <memory>

namespace stage2_two_axis {

static constexpr size_t memory_reserve = 1ull << 30;

struct FoldWorkspace {
    FuseCtx y{}, x{};
    Word* data = nullptr;
    Word *a = nullptr, *b = nullptr, *input = nullptr;
    Word* staging = nullptr;
    Word *twiddle = nullptr, *inverse_weights = nullptr;
    const mersenne::WeightedDigit* digits = nullptr;
    const mersenne::WeightedLimb* spans = nullptr;
    unsigned p = 0, words = 0, limbs = 0, ly = 0, k = 0, lx = 0;
    unsigned limb_shared = 0;
    size_t allocated = 0;
    std::array<Word*,2> retained{};
    Word cube = 0, cube2 = 0, x_root = 0, x_inv = 0, x_scale = 0;

    ~FoldWorkspace() {
        fuse_release(y);
        fuse_release(x);
        for (auto* spectrum : retained) if (spectrum) CK(cudaFree(spectrum));
        if (data) CK(cudaFree(data));
    }
};

static Word inv(Word x) { return gl_pow_host(x, GL_P - 2); }

__global__ __launch_bounds__(FUSE_TILE_THREADS)
static void inverse_rows(Word* data, unsigned length, int bits, const Word* table) {
    extern __shared__ Word shared[];
    auto* row = data + size_t(blockIdx.x) * length;
    for (unsigned i = threadIdx.x; i < length; i += blockDim.x) shared[i] = row[i];
    __syncthreads();
    ntt_tile_pass<true, true>(row, shared, length, bits, table);
}

__global__ static void multiply_spectra(Word* out, const Word* a, const Word* b, size_t words, Word scale) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < words) out[i] = gl_mul(gl_mul(a[i], b[i]), scale);
}

static FoldWorkspace* create_shape(unsigned p, unsigned ma, unsigned mb, unsigned first, unsigned output_rows,
                                   bool host, size_t limit, const char*& fallback) {
    fallback = "shape";
    if (!p || !ma || !mb || ma > 32768 || mb > 32768) return nullptr;
    const unsigned overlap = std::min(ma, mb);
    const auto required = output_rows ? window_span(ma,mb,first,output_rows) : ma + mb - 1;
    auto state = std::make_unique<FoldWorkspace>();
    auto& s = *state;
    s.p = p;
    s.words = (p + 63) / 64;
    s.limbs = (p + 31) / 32;
    unsigned ybits = 4;
    for (; ybits <= 12; ++ybits) {
        const unsigned ly = 1u << ybits;
        if (ly > p || (p + ly - 1) / ly > 31) continue;
        const auto bound = mersenne::weighted_unsigned_bound(p, ly);
        if (bound && *bound <= (GL_P - 1) / overlap) {
            s.ly = ly;
            break;
        }
    }
    if (!s.ly) return nullptr;
    unsigned xbits = 3;
    for (; (1u << xbits) < required; ++xbits) {}
    s.lx = 1u << xbits;
    const bool radix3 = xbits > 3 && (s.lx / 4) * 3 >= required;
    if (radix3) xbits -= 2;
    s.k = 1u << xbits;
    if (radix3) s.lx = 3 * s.k;
    const size_t matrix = size_t(s.lx) * s.ly, input = size_t(mb) * s.ly;
    const size_t staging_words = host ? size_t(ma + mb + output_rows) * s.words : 0;
    const size_t twiddles = radix3 ? 4 * size_t(s.k) : 0;
    const size_t digit_words = s.ly * sizeof(mersenne::WeightedDigit) / sizeof(Word);
    const size_t span_words = s.limbs * sizeof(mersenne::WeightedLimb) / sizeof(Word);
    const auto yroot = gl_pow_host(7, (GL_P - 1) / s.ly), yinv = inv(yroot);
    s.x_root = gl_pow_host(7, (GL_P - 1) / s.k);
    s.x_inv = inv(s.x_root);
    s.x_scale = inv(s.k);
    const FuseOverrides yopts{int(ybits), 4, true};
    FuseCtx yplan{}, xplan{};
    fuse_describe(yplan, s.ly, ybits, yroot, yinv, true, &yopts);
    fuse_describe(xplan, s.k, xbits, s.x_root, s.x_inv, true);
    const size_t data_words = 2 * matrix + input + twiddles + s.ly + digit_words + span_words + staging_words;
    s.allocated = (data_words + fuse_planned_base_words(yplan) + fuse_planned_table_words(yplan) +
                   fuse_planned_base_words(xplan) + fuse_planned_table_words(xplan)) * sizeof(Word);
    fallback = "budget";
    if (s.allocated > limit) return nullptr;
    size_t free_bytes, total_bytes;
    CK(cudaMemGetInfo(&free_bytes, &total_bytes));
    fallback = "headroom";
    if (free_bytes < memory_reserve || s.allocated > free_bytes - memory_reserve) return nullptr;
    const auto status = cudaMalloc(&s.data, data_words * sizeof(Word));
    if (status == cudaErrorMemoryAllocation) {
        cudaGetLastError();
        fallback = "allocation";
        return nullptr;
    }
    CK(status);
    s.a = s.data;
    s.b = s.a + matrix;
    s.input = s.b + matrix;
    s.twiddle = s.input + input;
    s.inverse_weights = s.twiddle + twiddles;
    auto* digit_data = s.inverse_weights + s.ly;
    auto* span_data = digit_data + digit_words;
    if (host) s.staging = span_data + span_words;
    s.digits = reinterpret_cast<const mersenne::WeightedDigit*>(digit_data);
    s.spans = reinterpret_cast<const mersenne::WeightedLimb*>(span_data);
    const auto weights = mersenne::weighted_tables(p, s.ly, inv(s.ly * (radix3 ? 3ull : 1ull)));
    s.limb_shared = mersenne::weighted_limb_smem(weights.limbs);
    CK(cudaMemcpy(s.inverse_weights, weights.inverse.data(), s.ly * sizeof(Word), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(digit_data, weights.digits.data(), digit_words * sizeof(Word), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(span_data, weights.limbs.data(), span_words * sizeof(Word), cudaMemcpyHostToDevice));
    if (radix3) {
        const Word root = gl_pow_host(7, (GL_P - 1) / s.lx), root2 = gl_mul_host(root, root);
        s.cube = gl_pow_host(root, s.k);
        s.cube2 = gl_mul_host(s.cube, s.cube);
        const Word steps[] = {root, root2, inv(root), inv(root2)};
        std::vector<Word> table(twiddles);
        for (unsigned i = 0; i < 4; ++i) {
            auto* plane = table.data() + size_t(i) * s.k;
            plane[0] = 1;
            for (unsigned j = 1; j < s.k; ++j) plane[j] = gl_mul_host(plane[j - 1], steps[i]);
        }
        CK(cudaMemcpy(s.twiddle, table.data(), twiddles * sizeof(Word), cudaMemcpyHostToDevice));
    }
    fuse_init(s.y, s.ly, ybits, yroot, yinv, true, &yopts);
    fuse_init(s.x, s.k, xbits, s.x_root, s.x_inv, true);
    ntt_fuse_cache_tables(s.x, s.x_root, s.x_inv);
    fallback = "none";
    return state.release();
}

FoldWorkspace* create(unsigned p, unsigned max_coefficients, size_t limit, const char*& fallback) {
    return create_shape(p, max_coefficients, max_coefficients, 0, 0, false, limit, fallback);
}

FoldWorkspace* create_product(unsigned p, unsigned ma, unsigned mb, unsigned first, unsigned count,
                              bool host, size_t limit, const char*& fallback) {
    return create_shape(p, std::min(ma, first + count), std::min(mb, first + count), first, count, host, limit, fallback);
}

void release(FoldWorkspace*& workspace) {
    delete workspace;
    workspace = nullptr;
}

size_t bytes(const FoldWorkspace& workspace) { return workspace.allocated; }
unsigned coefficient_length(const FoldWorkspace& workspace) { return workspace.ly; }
unsigned polynomial_length(const FoldWorkspace& workspace) { return workspace.lx; }
Word* host_staging(FoldWorkspace& workspace) { return workspace.staging; }
size_t spectrum_words(const FoldWorkspace& workspace) { return size_t(workspace.ly) * workspace.lx; }

static void forward_attributes(const FoldWorkspace& s) {
    CK(cudaFuncSetAttribute(pack_forward, cudaFuncAttributeMaxDynamicSharedMemorySize, int(s.ly * sizeof(Word))));
    ntt_tile_attributes(s.x.warp_tail && s.x.t >= 5, int(sizeof(Word)) << s.x.t);
}

static Spectrum pretransform_ready(FoldWorkspace& s, Operand operand, const Word* coefficients, unsigned rows,
                                   unsigned reverse_rows, Word* retained) {
    const bool left = operand == Operand::left;
    auto* packed = left ? s.b : s.input;
    auto* spectrum = retained ? retained : (left ? s.a : s.b);
    const bool radix3 = s.lx != s.k;
    const dim3 grid((s.ly + 31) / 32, (s.k + 31) / 32), threads(32, 8);
    const size_t shared = s.ly * sizeof(Word);
    pack_forward<<<rows, FUSE_TILE_THREADS, shared>>>(coefficients, packed, s.words, s.ly, s.y.k,
                                                     s.digits, s.y.tblF, reverse_rows);
    if (radix3)
        transpose_forward<true><<<grid, threads>>>(packed, spectrum, s.ly, s.k, rows, s.ly, s.twiddle, s.cube);
    else
        transpose_forward<false><<<grid, threads>>>(packed, spectrum, s.ly, s.k, rows, s.ly, nullptr, 0);
    CK(cudaGetLastError());
    ntt_forward_fused(spectrum, s.x, s.x_root, s.ly * (s.lx / s.k));
    return {spectrum};
}

Spectrum pretransform(FoldWorkspace& s, Operand operand, const Word* coefficients, unsigned rows,
                      unsigned reverse_rows, Word* retained) {
    forward_attributes(s);
    return pretransform_ready(s, operand, coefficients, rows, reverse_rows, retained);
}

Spectrum retain(FoldWorkspace& s, unsigned slot, const Word* coefficients, unsigned rows, size_t limit) {
    auto& storage = s.retained.at(slot);
    if (!storage) {
        const size_t needed = spectrum_words(s) * sizeof(Word);
        if (s.allocated > limit || needed > limit - s.allocated) return {};
        size_t free_bytes, total_bytes;
        CK(cudaMemGetInfo(&free_bytes, &total_bytes));
        if (free_bytes < memory_reserve || needed > free_bytes - memory_reserve) return {};
        const auto status = cudaMalloc(&storage, needed);
        if (status == cudaErrorMemoryAllocation) {
            cudaGetLastError();
            storage = nullptr;
            return {};
        }
        CK(status);
        s.allocated += needed;
    }
    return pretransform(s, Operand::left, coefficients, rows, 0, storage);
}

void frequency_product(FoldWorkspace& s, Spectrum left, Spectrum right) {
    const size_t words = spectrum_words(s);
    constexpr unsigned threads = 256;
    multiply_spectra<<<unsigned((words + threads - 1) / threads), threads>>>(
        s.a, left.data, right.data, words, s.x_scale);
    CK(cudaGetLastError());
}

static void finish_window(FoldWorkspace& s, Word* polynomial, Word* coefficients,
                          Word* out, unsigned first, unsigned count) {
    const bool radix3 = s.lx != s.k;
    const dim3 threads(32, 8);
    const unsigned inverse_columns = first ? s.k : std::min(s.k, count);
    const dim3 inverse_grid((inverse_columns + 31) / 32, (s.ly + 31) / 32);
    const size_t shared = s.ly * sizeof(Word);
    CK(cudaFuncSetAttribute(inverse_rows, cudaFuncAttributeMaxDynamicSharedMemorySize, int(shared)));
    if (radix3)
        transpose_inverse<true><<<inverse_grid, threads>>>(polynomial, coefficients, s.ly, s.k, count, s.ly, s.twiddle + 2 * s.k, s.cube2, first);
    else
        transpose_inverse<false><<<inverse_grid, threads>>>(polynomial, coefficients, s.ly, s.k, count, s.ly, nullptr, 0, first);
    inverse_rows<<<count, FUSE_TILE_THREADS, shared>>>(coefficients, s.ly, s.y.k, s.y.tblI);
    const unsigned limb_blocks = (s.limbs + mersenne::weighted_limb_threads - 1) / mersenne::weighted_limb_threads;
    mersenne::weighted_limb_sums<<<dim3(limb_blocks, count), mersenne::weighted_limb_threads, s.limb_shared>>>(
        polynomial, coefficients, s.inverse_weights, s.digits, s.spans, s.p, s.limbs, s.ly);
    normalize<<<count, carry_threads>>>(reinterpret_cast<uint32_t*>(out), polynomial, s.limbs, s.p, 2 * s.words);
    CK(cudaGetLastError());
}

void inverse_window(FoldWorkspace& s, Word* out, unsigned first, unsigned count) {
    // frequency_product has already applied the power-of-two axis scale.
    if (s.x.t) {
        const unsigned tile = 1u << s.x.t;
        const size_t shared = tile * sizeof(Word);
        CK(cudaFuncSetAttribute(inverse_rows, cudaFuncAttributeMaxDynamicSharedMemorySize, int(shared)));
        inverse_rows<<<unsigned(spectrum_words(s) / tile), FUSE_TILE_THREADS, shared>>>(s.a, tile, s.x.t, s.x.tblI);
        CK(cudaGetLastError());
    }
    ntt_inverse_outer_fused(s.a, s.x, s.x_inv, s.ly * (s.lx / s.k), 0);
    finish_window(s, s.a, s.b, out, first, count);
}

void multiply_retained(FoldWorkspace& s, const Word* coefficients, unsigned rows, Spectrum fixed,
                       Word* out, unsigned first, unsigned count, unsigned reverse_rows) {
    pretransform(s, Operand::left, coefficients, std::min(rows, first + count), reverse_rows);
    ntt_inverse_fused(s.a, fixed.data, s.x, s.x_inv, s.x_scale, s.ly * (s.lx / s.k));
    finish_window(s, s.a, s.b, out, first, count);
}

void multiply_preserving_left(FoldWorkspace& s, const Word* coefficients, unsigned rows,
                              Word* out, unsigned first, unsigned count, unsigned reverse_rows) {
    pretransform(s, Operand::right, coefficients, std::min(rows, first + count), reverse_rows);
    ntt_inverse_fused(s.b, s.a, s.x, s.x_inv, s.x_scale, s.ly * (s.lx / s.k));
    finish_window(s, s.b, s.input, out, first, count);
}

void multiply(FoldWorkspace& s, const Word* a, unsigned ma, const Word* b, unsigned mb, Word* out, unsigned count,
              unsigned first, bool reverse_b) {
    // Terms at or above the window end cannot contribute.
    const unsigned end = first + count;
    forward_attributes(s);
    pretransform_ready(s, Operand::left, a, std::min(ma, end), 0, nullptr);
    const auto right = pretransform_ready(s, Operand::right, b, std::min(mb, end), reverse_b ? mb : 0, nullptr);
    // Fuse the spectral product with the first inverse pass for one-shot calls.
    ntt_inverse_fused(s.a, right.data, s.x, s.x_inv, s.x_scale, s.ly * (s.lx / s.k));
    finish_window(s, s.a, s.b, out, first, count);
}

}
