#include "../two_axis.h"
#include "../../../cuda/ntt/runtime.h"
#include "two_axis.cuh"

#include <memory>

namespace stage2_two_axis {

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
    Word cube = 0, cube2 = 0, x_root = 0, x_inv = 0, x_scale = 0;

    ~FoldWorkspace() {
        fuse_release(y);
        fuse_release(x);
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
    constexpr size_t reserve = 1ull << 30;
    fallback = "headroom";
    if (free_bytes < reserve || s.allocated > free_bytes - reserve) return nullptr;
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

void multiply(FoldWorkspace& s, const Word* a, unsigned ma, const Word* b, unsigned mb, Word* out, unsigned count,
              unsigned first, bool reverse_b) {
    // Input indices at or above the end of the output window cannot contribute.
    const unsigned b_rows = mb;
    ma = std::min(ma, first + count);
    mb = std::min(mb, first + count);
    const bool radix3 = s.lx != s.k;
    const dim3 threads(32, 8), forward_grid((s.ly + 31) / 32, (s.k + 31) / 32);
    const unsigned inverse_columns = first ? s.k : std::min(s.k, count);
    const dim3 inverse_grid((inverse_columns + 31) / 32, (s.ly + 31) / 32);
    const size_t shared = s.ly * sizeof(Word);
    CK(cudaFuncSetAttribute(pack_forward, cudaFuncAttributeMaxDynamicSharedMemorySize, int(shared)));
    CK(cudaFuncSetAttribute(inverse_rows, cudaFuncAttributeMaxDynamicSharedMemorySize, int(shared)));
    pack_forward<<<ma, FUSE_TILE_THREADS, shared>>>(a, s.b, s.words, s.ly, s.y.k, s.digits, s.y.tblF);
    pack_forward<<<mb, FUSE_TILE_THREADS, shared>>>(b, s.input, s.words, s.ly, s.y.k, s.digits, s.y.tblF,
                                                  reverse_b ? b_rows : 0);
    auto forward = [&](const Word* from, Word* to, unsigned rows) {
        if (radix3)
            transpose_forward<true><<<forward_grid, threads>>>(from, to, s.ly, s.k, rows, s.ly, s.twiddle, s.cube);
        else
            transpose_forward<false><<<forward_grid, threads>>>(from, to, s.ly, s.k, rows, s.ly, nullptr, 0);
        CK(cudaGetLastError());
    };
    forward(s.b, s.a, ma);
    forward(s.input, s.b, mb);
    ntt_tile_attributes(s.x.warp_tail && s.x.t >= 5, int(sizeof(Word)) << s.x.t);
    const unsigned batch = s.ly * (radix3 ? 3 : 1);
    ntt_forward_fused(s.a, s.x, s.x_root, batch);
    ntt_forward_fused(s.b, s.x, s.x_root, batch);
    ntt_inverse_fused(s.a, s.b, s.x, s.x_inv, s.x_scale, batch);
    if (radix3)
        transpose_inverse<true><<<inverse_grid, threads>>>(s.a, s.b, s.ly, s.k, count, s.ly, s.twiddle + 2 * s.k, s.cube2, first);
    else
        transpose_inverse<false><<<inverse_grid, threads>>>(s.a, s.b, s.ly, s.k, count, s.ly, nullptr, 0, first);
    inverse_rows<<<count, FUSE_TILE_THREADS, shared>>>(s.b, s.ly, s.y.k, s.y.tblI);
    const unsigned limb_blocks = (s.limbs + mersenne::weighted_limb_threads - 1) / mersenne::weighted_limb_threads;
    mersenne::weighted_limb_sums<<<dim3(limb_blocks, count), mersenne::weighted_limb_threads, s.limb_shared>>>(
        s.a, s.b, s.inverse_weights, s.digits, s.spans, s.p, s.limbs, s.ly);
    normalize<<<count, carry_threads>>>(reinterpret_cast<uint32_t*>(out), s.a, s.limbs, s.p, 2 * s.words);
    CK(cudaGetLastError());
}

}
