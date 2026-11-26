#include "ecm/cuda/ntt/runtime.h"
#include "ecm/stage2/cuda/two_axis.h"
#include "ecm/stage2/cuda/device/two_axis.cuh"

#include <array>
#include <charconv>
#include <optional>
#include <random>
#include <string_view>

using Word = unsigned long long;
namespace mersenne = ecm_mersenne_ntt;
namespace two_axis = stage2_two_axis;

static Word inverse(Word a) { return gl_pow_host(a, GL_P - 2); }

struct Buffer {
    Word* data = nullptr;
    explicit Buffer(size_t words) { if (words) CK(cudaMalloc(&data, words * sizeof(Word))); }
    ~Buffer() { if (data) CK(cudaFree(data)); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};

struct Integer {
    mpz_t value;
    Integer() { mpz_init(value); }
    ~Integer() { mpz_clear(value); }
    Integer(const Integer&) = delete;
    Integer& operator=(const Integer&) = delete;
};

struct Axis {
    FuseCtx ctx{};
    Word length, root, inv_root, scale;

    Axis(unsigned bits, bool single_tile = false)
        : length(Word{1} << bits), root(gl_pow_host(7, (GL_P - 1) / length)),
          inv_root(inverse(root)), scale(inverse(length)) {
        const FuseOverrides options{int(bits), 4, true};
        fuse_init(ctx, length, bits, root, inv_root, true, single_tile ? &options : nullptr);
        ntt_fuse_cache_tables(ctx, root, inv_root);
    }

    ~Axis() { fuse_release(ctx); }
    Axis(const Axis&) = delete;
    Axis& operator=(const Axis&) = delete;

    void forward(Word* a, unsigned rows) const { ntt_forward_fused(a, ctx, root, rows); }
    void multiply_inverse(Word* a, const Word* b, unsigned rows) const {
        ntt_inverse_fused(a, b, ctx, inv_root, scale, rows);
    }
};

__global__ void transpose(const Word* in, Word* out, unsigned columns, unsigned rows) {
    __shared__ Word tile[32][33];
    const unsigned x = blockIdx.x * 32 + threadIdx.x;
    const unsigned y = blockIdx.y * 32 + threadIdx.y;
    for (unsigned i = 0; i < 32; i += 8)
        if (x < columns && y + i < rows) tile[threadIdx.y + i][threadIdx.x] = in[size_t(y + i) * columns + x];
    __syncthreads();
    const unsigned tx = blockIdx.y * 32 + threadIdx.x;
    const unsigned ty = blockIdx.x * 32 + threadIdx.y;
    for (unsigned i = 0; i < 32; i += 8)
        if (tx < rows && ty + i < columns) out[size_t(ty + i) * rows + tx] = tile[threadIdx.x][threadIdx.y + i];
}

static void transpose_matrix(const Word* in, Word* out, unsigned columns, unsigned rows) {
    transpose<<<dim3((columns + 31) / 32, (rows + 31) / 32), dim3(32, 8)>>>(in, out, columns, rows);
    CK(cudaGetLastError());
}

template<bool Inverse>
__global__ void outer_radix3(Word* rows, unsigned length, const Word* twiddle,
                            Word cube_root, Word cube_root2, Word scale) {
    const unsigned j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= length) return;
    auto* row = rows + size_t(blockIdx.y) * 3 * length;
    const auto w = twiddle[j], w2 = twiddle[length + j];
    const auto a = row[j];
    auto b = row[length + j], c = row[2 * length + j];
    if constexpr (Inverse) {
        b = gl_mul(b, w);
        c = gl_mul(c, w2);
    }
    auto u = gl_add(a, gl_add(b, c));
    auto v = gl_add(a, gl_add(gl_mul(b, cube_root), gl_mul(c, cube_root2)));
    auto z = gl_add(a, gl_add(gl_mul(b, cube_root2), gl_mul(c, cube_root)));
    if constexpr (Inverse) {
        u = gl_mul(u, scale);
        v = gl_mul(v, scale);
        z = gl_mul(z, scale);
    } else {
        v = gl_mul(v, w);
        z = gl_mul(z, w2);
    }
    row[j] = u;
    row[length + j] = v;
    row[2 * length + j] = z;
}

__global__ void coefficient_inverse(Word* out, unsigned length, unsigned bits, const Word* table) {
    extern __shared__ Word shared[];
    auto* row = out + size_t(blockIdx.x) * length;
    for (unsigned i = threadIdx.x; i < length; i += blockDim.x) shared[i] = row[i];
    __syncthreads();
    ntt_tile_pass<true, true>(row, shared, length, bits, table);
}

static std::vector<Word> fill(std::vector<Integer>& coefficients, unsigned p, Word seed) {
    std::mt19937_64 rng(seed);
    const unsigned words = (p + 63) / 64;
    const Word top = ~Word{0} >> ((64 - p % 64) % 64);
    std::vector<Word> raw(coefficients.size() * words);
    for (unsigned i = 0; i < coefficients.size(); ++i) {
        auto* row = raw.data() + size_t(i) * words;
        for (unsigned j = 0; j < words; ++j) row[j] = rng();
        row[words - 1] &= top;
        if (i == 0) {
            std::fill_n(row, words, ~Word{0});
            row[words - 1] = top;
            --row[0]; // M-1 is the largest canonical input.
        } else if (i == 1) {
            std::fill_n(row, words, Word{0});
        } else if (row[words - 1] == top && std::all_of(row, row + words - 1, [](Word limb) { return limb == ~Word{0}; })) {
            std::fill_n(row, words, Word{0});
        }
        mpz_import(coefficients[i].value, words, -1, sizeof(Word), 0, 0, row);
    }
    return raw;
}

static std::vector<Word> pack_weighted(const std::vector<Word>& raw, unsigned words,
                                       const mersenne::WeightedTables& weights) {
    const unsigned length = unsigned(weights.digits.size());
    const size_t rows = raw.size() / words;
    std::vector<Word> packed(rows * length);
    for (size_t i = 0; i < rows; ++i) {
        const auto* row = raw.data() + i * words;
        for (unsigned j = 0; j < length; ++j) {
            const auto [weight, bit, width] = weights.digits[j];
            const unsigned limb = bit / 64, offset = bit % 64;
            auto value = row[limb] >> offset;
            if (offset + width > 64) value |= row[limb + 1] << (64 - offset);
            packed[i * length + j] = gl_mul_host(value & ((Word{1} << width) - 1), weight);
        }
    }
    return packed;
}

static void convolution_at(Integer& expected, unsigned i, const std::vector<Integer>& a,
                           const std::vector<Integer>& b, const Integer& modulus) {
    mpz_set_ui(expected.value, 0);
    const unsigned first = i >= b.size() ? i - unsigned(b.size()) + 1 : 0;
    const unsigned last = std::min(i, unsigned(a.size() - 1));
    for (unsigned j = first; j <= last; ++j)
        mpz_addmul(expected.value, a[j].value, b[i - j].value);
    mpz_mod(expected.value, expected.value, modulus.value);
}

struct ProductOracle {
    const unsigned count;
    Integer modulus;
    std::vector<Integer> samples;

    ProductOracle(const std::vector<Integer>& a, const std::vector<Integer>& b, unsigned p)
        : count(unsigned(a.size() + b.size() - 1)), samples(count <= 64 ? count : 9) {
        mpz_setbit(modulus.value, p);
        mpz_sub_ui(modulus.value, modulus.value, 1);
        for (unsigned sample = 0; sample < samples.size(); ++sample)
            convolution_at(samples[sample], index(sample), a, b, modulus);
    }

    unsigned index(unsigned sample) const {
        return samples.size() == 1 ? 0 : unsigned(size_t(sample) * (count - 1) / (samples.size() - 1));
    }
};

static bool verify(const Word* data, const ProductOracle& reference, unsigned p,
                   const mersenne::WeightedTables& weights, bool canonical) {
    const unsigned length = unsigned(weights.digits.size());
    std::vector<Word> row(length);
    std::vector<uint32_t> limbs((p + 31) / 32);
    Integer actual, digit;
    for (unsigned sample = 0; sample < reference.samples.size(); ++sample) {
        const unsigned i = reference.index(sample);
        if (canonical) {
            CK(cudaMemcpy(limbs.data(), reinterpret_cast<const uint32_t*>(data) + size_t(i) * limbs.size(),
                          limbs.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
            mpz_import(actual.value, limbs.size(), -1, sizeof(uint32_t), 0, 0, limbs.data());
        } else {
            CK(cudaMemcpy(row.data(), data + size_t(i) * length, length * sizeof(Word), cudaMemcpyDeviceToHost));
            mpz_set_ui(actual.value, 0);
            for (unsigned j = 0; j < length; ++j) {
                const auto value = gl_mul_host(row[j], weights.inverse[j]);
                mpz_import(digit.value, 1, -1, sizeof(value), 0, 0, &value);
                mpz_mul_2exp(digit.value, digit.value, weights.digits[j].bit);
                mpz_add(actual.value, actual.value, digit.value);
            }
            mpz_mod(actual.value, actual.value, reference.modulus.value);
        }
        if (mpz_cmp(actual.value, reference.samples[sample].value) != 0) {
            std::fprintf(stderr, "GMP polynomial mismatch at coefficient %u\n", i);
            return false;
        }
    }
    return true;
}

static void pack_flat(std::vector<Word>& input, const std::vector<Integer>& coefficients,
                      unsigned p, int bpw, Word stride) {
    const auto mask = (Word{1} << bpw) - 1;
    std::vector<Word> limbs((p + 63) / 64 + 1);
    for (size_t i = 0; i < coefficients.size(); ++i) {
        std::fill(limbs.begin(), limbs.end(), 0);
        mpz_export(limbs.data(), nullptr, -1, sizeof(Word), 0, 0, coefficients[i].value);
        auto* slot = input.data() + i * stride;
        for (unsigned bit = 0; bit < p; bit += bpw) {
            const unsigned limb = bit / 64, offset = bit % 64;
            auto value = limbs[limb] >> offset;
            if (offset + bpw > 64) value |= limbs[limb + 1] << (64 - offset);
            *slot++ = value & mask;
        }
    }
}

static int benchmark_flat(const std::vector<Integer>& a, const std::vector<Integer>& b,
                          const ProductOracle& reference, unsigned p, unsigned repeats) {
    Word n, slot_bits, stride, slot_stride, out_slots;
    int bpw;
    if (!ntt_shape_query(std::max(a.size(), b.size()), p, &n, &bpw, &slot_bits, &stride,
                         &slot_stride, &out_slots)) return 2;
    unsigned bits = 0;
    for (Word length = n; length > 1; length >>= 1) ++bits;
    const Axis axis(bits);
    Buffer dev_a(n), dev_b(n);
    std::vector<Word> input_a(a.size() * stride), input_b(b.size() * stride), row(stride);
    pack_flat(input_a, a, p, bpw, stride);
    pack_flat(input_b, b, p, bpw, stride);
    Integer actual, digit;
    cudaEvent_t start, end;
    CK(cudaEventCreate(&start)); CK(cudaEventCreate(&end));
    const unsigned sample_count = unsigned(reference.samples.size());
    std::printf("flat p=%u length=%llu bpw=%d ma=%zu mb=%zu array_bytes=%llu working_array_bytes=%llu\n",
                p, n, bpw, a.size(), b.size(), n * sizeof(Word), 2 * n * sizeof(Word));
    for (unsigned repeat = 0; repeat < repeats; ++repeat) {
        CK(cudaMemset(dev_a.data, 0, n * sizeof(Word)));
        CK(cudaMemset(dev_b.data, 0, n * sizeof(Word)));
        CK(cudaMemcpy(dev_a.data, input_a.data(), input_a.size() * sizeof(Word), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dev_b.data, input_b.data(), input_b.size() * sizeof(Word), cudaMemcpyHostToDevice));
        CK(cudaEventRecord(start));
        axis.forward(dev_a.data, 1);
        axis.forward(dev_b.data, 1);
        axis.multiply_inverse(dev_a.data, dev_b.data, 1);
        CK(cudaEventRecord(end));
        CK(cudaEventSynchronize(end));
        float ms;
        CK(cudaEventElapsedTime(&ms, start, end));
        for (unsigned sample = 0; sample < sample_count; ++sample) {
            const unsigned i = reference.index(sample);
            CK(cudaMemcpy(row.data(), dev_a.data + size_t(i) * stride, stride * sizeof(Word), cudaMemcpyDeviceToHost));
            mpz_set_ui(actual.value, 0);
            for (auto it = row.rbegin(); it != row.rend(); ++it) {
                mpz_mul_2exp(actual.value, actual.value, bpw);
                mpz_import(digit.value, 1, -1, sizeof(Word), 0, 0, &*it);
                mpz_add(actual.value, actual.value, digit.value);
            }
            mpz_mod(actual.value, actual.value, reference.modulus.value);
            if (mpz_cmp(actual.value, reference.samples[sample].value)) {
                std::fprintf(stderr, "GMP flat polynomial mismatch at coefficient %u\n", i);
                return 1;
            }
        }
        std::printf("sample=%u total_ms=%.3f gmp_checked=%u\n", repeat, ms, sample_count);
    }
    CK(cudaEventDestroy(start)); CK(cudaEventDestroy(end));
    return 0;
}

static unsigned read_unsigned(std::string_view text) {
    unsigned value = 0;
    const auto last = text.data() + text.size();
    const auto parsed = std::from_chars(text.data(), last, value);
    if (parsed.ec != std::errc{} || parsed.ptr != last) std::exit(2);
    return value;
}

static bool workspace_available(size_t words) {
    size_t free_bytes, total_bytes;
    CK(cudaMemGetInfo(&free_bytes, &total_bytes));
    // Leave space for runtime tables, the CUDA context and display allocations.
    constexpr size_t reserve = 384ull << 20;
    if (free_bytes >= reserve && words <= (free_bytes - reserve) / sizeof(Word)) return true;
    std::fprintf(stderr, "Insufficient device memory: arrays=%zu free=%zu reserve=%zu\n",
                 words * sizeof(Word), free_bytes, reserve);
    return false;
}

static bool check_staged_product(unsigned p, unsigned ma, unsigned mb, unsigned first, unsigned count, bool reverse) {
    const auto width = (p + 63) / 64;
    const char* reason = nullptr;
    auto* workspace = two_axis::create_product(p, ma, mb, first, count, false, 512ull << 20, reason);
    if (!workspace) {
        std::fprintf(stderr, "staged workspace: %s\n", reason);
        return false;
    }
    const auto x = two_axis::polynomial_length(*workspace);
    Buffer da(size_t(ma) * width), db(size_t(mb) * width), retained(two_axis::spectrum_words(*workspace));
    Buffer out(size_t(count) * width);
    std::vector<Integer> b(mb);
    auto raw_b = fill(b, p, 73);
    CK(cudaMemcpy(db.data, raw_b.data(), raw_b.size() * sizeof(Word), cudaMemcpyHostToDevice));
    const auto rhs = two_axis::pretransform(*workspace, two_axis::Operand::right, db.data,
        std::min(mb, first + count), reverse ? mb : 0, retained.data);
    if (reverse)
        for (unsigned i = 0; i < mb / 2; ++i) mpz_swap(b[i].value, b[mb - i - 1].value);
    Integer expected, actual, modulus;
    mpz_setbit(modulus.value, p);
    mpz_sub_ui(modulus.value, modulus.value, 1);
    std::vector<Word> row(size_t(count) * width), fused(row.size());
    bool valid = true;
    for (unsigned iteration = 0; iteration < 2; ++iteration) {
        std::vector<Integer> a(ma);
        auto raw_a = fill(a, p, 41 + iteration);
        CK(cudaMemcpy(da.data, raw_a.data(), raw_a.size() * sizeof(Word), cudaMemcpyHostToDevice));
        const auto lhs = two_axis::pretransform(*workspace, two_axis::Operand::left, da.data, std::min(ma, first + count));
        two_axis::frequency_product(*workspace, lhs, rhs);
        two_axis::inverse_window(*workspace, out.data, first, count);
        CK(cudaMemcpy(row.data(), out.data, row.size() * sizeof(Word), cudaMemcpyDeviceToHost));
        two_axis::multiply(*workspace, da.data, ma, db.data, mb, out.data, count, first, reverse);
        CK(cudaMemcpy(fused.data(), out.data, fused.size() * sizeof(Word), cudaMemcpyDeviceToHost));
        valid &= row == fused;
        for (unsigned i = 0; i < count; ++i) {
            convolution_at(expected, first + i, a, b, modulus);
            mpz_import(actual.value, width, -1, sizeof(Word), 0, 0, row.data() + size_t(i) * width);
            valid &= mpz_cmp(expected.value, actual.value) == 0;
        }
    }
    two_axis::release(workspace);
    std::printf("staged_api p=%u x=%u first=%u count=%u reverse=%u gmp_checked=%u %s\n",
        p, x, first, count, unsigned(reverse), 2 * count, valid ? "PASS" : "FAIL");
    return valid;
}

static bool check_tree_batch(unsigned p, unsigned ma, unsigned mb) {
    constexpr unsigned products = 4, batch = 3;
    const unsigned count = ma + mb - 1, width = (p + 63) / 64;
    const size_t input_stride = size_t(ma + mb) * width;
    const size_t output_stride = size_t(count) * width + 1;
    std::array<Word, products * 3> offsets{};
    std::vector<std::vector<Integer>> a, b;
    Buffer src(products * input_stride), dst(products * output_stride), map(offsets.size());
    for (unsigned i = 0; i < products; ++i) {
        a.emplace_back(ma);
        b.emplace_back(mb);
        const auto left = fill(a.back(), p, 41 + i);
        const auto right = fill(b.back(), p, 71 + i);
        CK(cudaMemcpy(src.data + i * input_stride, left.data(), left.size() * sizeof(Word), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(src.data + i * input_stride + left.size(), right.data(), right.size() * sizeof(Word), cudaMemcpyHostToDevice));
        offsets[i] = i * input_stride;
        offsets[products + i] = ((i + 1) % products) * input_stride + size_t(ma) * width;
        offsets[2 * products + i] = (products - 1 - i) * output_stride;
    }
    CK(cudaMemcpy(map.data, offsets.data(), sizeof(offsets), cudaMemcpyHostToDevice));
    CK(cudaMemset(dst.data, 0xff, products * output_stride * sizeof(Word)));
    const char* reason = nullptr;
    auto* workspace = two_axis::create_product(p, ma, mb, 0, count, false, 16ull << 20, reason, batch);
    if (!workspace || two_axis::batch_capacity(*workspace) != batch) {
        std::fprintf(stderr, "batch fixture workspace: %s\n", reason);
        two_axis::release(workspace);
        return false;
    }
    const auto x = two_axis::polynomial_length(*workspace);
    for (unsigned i = 0; i < products; i += batch)
        two_axis::multiply_tree_batch(*workspace, src.data, map.data + i, map.data + products + i,
            ma, mb, dst.data, map.data + 2 * products + i, std::min(batch, products - i));
    std::vector<Word> row(output_stride);
    Integer actual;
    bool valid = true;
    for (unsigned i = 0; i < products; ++i) {
        const ProductOracle expected(a[i], b[(i + 1) % products], p);
        CK(cudaMemcpy(row.data(), dst.data + offsets[2 * products + i], row.size() * sizeof(Word), cudaMemcpyDeviceToHost));
        valid &= row.back() == ~Word{0};
        for (unsigned j = 0; j < count; ++j) {
            mpz_import(actual.value, width, -1, sizeof(Word), 0, 0, row.data() + size_t(j) * width);
            valid &= mpz_cmp(actual.value, expected.samples[j].value) == 0;
        }
    }
    two_axis::release(workspace);
    std::printf("tree_batch_api p=%u x=%u products=%u batch=%u gmp_checked=%u %s\n",
        p, x, products, batch, count * products, valid ? "PASS" : "FAIL");
    return valid;
}

int main(int argc, char** argv) {
    if (argc == 3 && std::string_view(argv[1]) == "--tree-batch") {
        CK(cudaSetDevice(read_unsigned(argv[2])));
        const bool mixed = check_tree_batch(521, 9, 9);
        const bool radix2 = check_tree_batch(521, 16, 15);
        return mixed && radix2 ? 0 : 1;
    }
    if (argc == 3 && std::string_view(argv[1]) == "--staged-api") {
        CK(cudaSetDevice(read_unsigned(argv[2])));
        const bool low = check_staged_product(127, 37, 27, 0, 11, true);
        const bool middle = check_staged_product(521, 17, 20, 13, 11, false);
        const bool multi_tile = check_staged_product(2203, 6150, 6190, 6150, 19, true);
        return low && middle && multi_tile ? 0 : 1;
    }
    if (argc < 9 || argc > 11) {
        std::fprintf(stderr, "usage: %s DEVICE P COEFF_BITS POLY_BITS RADIX3 MA MB REPEATS [legacy|fused|gpu|stripe|flat|plan] [STRIPE_ROWS]\n", argv[0]);
        return 2;
    }
    const auto mode = argc >= 10 ? std::string_view(argv[9]) : "fused";
    const bool flat = mode == "flat", legacy = mode == "legacy", striped = mode == "stripe", plan = mode == "plan";
    const bool gpu = mode == "gpu" || striped;
    if ((!flat && !legacy && !gpu && !plan && mode != "fused") || (argc == 11 && !striped && !plan)) return 2;
    const unsigned dev = read_unsigned(argv[1]), p = read_unsigned(argv[2]);
    const unsigned coeff_bits = read_unsigned(argv[3]), poly_bits = read_unsigned(argv[4]);
    const unsigned radix3 = read_unsigned(argv[5]), ma = read_unsigned(argv[6]), mb = read_unsigned(argv[7]);
    const unsigned repeats = read_unsigned(argv[8]);
    if (!p || p > 262144 || coeff_bits < 4 || coeff_bits > 13 || poly_bits < 3 || poly_bits > 17 || radix3 > 1 ||
        !ma || !mb || ma > 65535 || mb > 65535 || !repeats) return 2;
    const unsigned ly = 1u << coeff_bits, k = 1u << poly_bits, lx = (radix3 ? 3 : 1) * k;
    const auto longest = std::min(ma, mb);
    const unsigned width = unsigned((Word(p) + ly - 1) / ly);
    const unsigned output_count = ma + mb - 1;
    const unsigned stripe_rows = argc == 11 ? read_unsigned(argv[10]) : std::min(1024u, ly);
    if (!stripe_rows || stripe_rows > ly) return 2;
    const bool weighted_shape = p >= ly && width <= 31 && output_count <= lx;
    const auto bound = weighted_shape ? mersenne::weighted_unsigned_bound(p, ly) : std::nullopt;
    const bool exact = bound && *bound <= (GL_P - 1) / longest;
    const unsigned input_words = (p + 63) / 64, output_words = (p + 31) / 32;
    const size_t words = size_t(ly) * lx, compact = size_t(mb) * ly;
    const size_t matrix = striped ? size_t(stripe_rows) * lx : words;
    const size_t scratch_words = striped ? size_t(ma + mb) * ly : legacy ? words : compact;
    const size_t raw_words = size_t(ma + mb) * input_words;
    const size_t workspace = 2 * matrix + scratch_words + (gpu ? raw_words : 0);
    Word flat_n = 0;
    int flat_bpw = 0;
    if (flat || plan)
        ntt_shape_query(std::max(ma, mb), p, &flat_n, &flat_bpw, nullptr, nullptr, nullptr, nullptr);
    if (plan) {
        std::printf("p=%u coeff_axis=%u poly_axis=%u digit_bits=%u exact=%u max_terms=%llu flat_length=%llu flat_bpw=%d "
                    "legacy_bytes=%zu fused_bytes=%zu gpu_bytes=%zu stripe_bytes=%zu flat_bytes=%llu\n",
                    p, ly, lx, width, unsigned(exact), bound && *bound ? (GL_P - 1) / *bound : 0,
                    flat_n, flat_bpw, 3 * words * sizeof(Word), (2 * words + compact) * sizeof(Word),
                    (2 * words + compact + raw_words) * sizeof(Word),
                    (2 * size_t(stripe_rows) * lx + size_t(ma + mb) * ly + raw_words) * sizeof(Word),
                    2 * flat_n * sizeof(Word));
        return 0;
    }
    if (!flat && !exact) {
        std::fprintf(stderr, "Goldilocks coefficient bound exceeded or output exceeds the supported axis shape\n");
        return 2;
    }
    CK(cudaSetDevice(dev));
    if (flat ? (!flat_n || !workspace_available(2 * flat_n)) : !workspace_available(workspace)) return 2;
    std::vector<Integer> reference_a(ma), reference_b(mb);
    auto raw_a = fill(reference_a, p, 17), raw_b = fill(reference_b, p, 117);
    const ProductOracle reference(reference_a, reference_b, p);
    const unsigned sample_count = unsigned(reference.samples.size());
    if (flat) {
        std::vector<Word>().swap(raw_a);
        std::vector<Word>().swap(raw_b);
        return benchmark_flat(reference_a, reference_b, reference, p, repeats);
    }
    const Word third = inverse(3);
    const auto weights = mersenne::weighted_tables(p, ly, !legacy && radix3 ? gl_mul_host(inverse(ly), third) : inverse(ly));
    const auto input_a = gpu ? std::vector<Word>() : pack_weighted(raw_a, input_words, weights);
    const auto input_b = gpu ? std::vector<Word>() : pack_weighted(raw_b, input_words, weights);
    const size_t raw_b_offset = raw_a.size();
    const Axis y(coeff_bits, true), x(poly_bits);
    if (y.ctx.nms) return 2;
    // The same kernel variant can be used by both axes.
    const int shared_bytes = int(sizeof(Word)) << std::max(y.ctx.t, x.ctx.t);
    ntt_tile_attributes(false, shared_bytes, true);
    ntt_tile_attributes(true, shared_bytes, true);
    CK(cudaFuncSetAttribute(coefficient_inverse, cudaFuncAttributeMaxDynamicSharedMemorySize, ly * sizeof(Word)));
    if (gpu) CK(cudaFuncSetAttribute(two_axis::pack_forward<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, ly * sizeof(Word)));
    Buffer a(matrix), b(matrix), scratch(scratch_words), twiddle(radix3 ? 4ull * k : 0);
    Buffer raw(gpu ? raw_words : 0), inverse_weights(gpu ? ly : 0);
    Buffer digits(gpu ? ly * sizeof(mersenne::WeightedDigit) / sizeof(Word) : 0);
    Buffer spans(gpu ? output_words * sizeof(mersenne::WeightedLimb) / sizeof(Word) : 0);
    auto* device_digits = reinterpret_cast<const mersenne::WeightedDigit*>(digits.data);
    if (gpu) {
        CK(cudaMemcpy(raw.data, raw_a.data(), raw_a.size() * sizeof(Word), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(raw.data + raw_b_offset, raw_b.data(), raw_b.size() * sizeof(Word), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(inverse_weights.data, weights.inverse.data(), ly * sizeof(Word), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(digits.data, weights.digits.data(), ly * sizeof(mersenne::WeightedDigit), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(spans.data, weights.limbs.data(), output_words * sizeof(mersenne::WeightedLimb), cudaMemcpyHostToDevice));
    }
    std::vector<Word>().swap(raw_a);
    std::vector<Word>().swap(raw_b);
    const Word root = gl_pow_host(7, (GL_P - 1) / lx);
    const Word cube = gl_pow_host(root, k), cube2 = gl_mul_host(cube, cube);
    if (radix3) {
        std::vector<Word> table(4ull * k);
        const Word step[] = {root, gl_mul_host(root, root), inverse(root), inverse(gl_mul_host(root, root))};
        for (unsigned i = 0; i < 4; ++i) {
            auto* plane = table.data() + size_t(i) * k;
            plane[0] = 1;
            for (unsigned j = 1; j < k; ++j) plane[j] = gl_mul_host(plane[j - 1], step[i]);
        }
        CK(cudaMemcpy(twiddle.data, table.data(), table.size() * sizeof(Word), cudaMemcpyHostToDevice));
    }
    cudaEvent_t start, coeff_done, poly_done, inverse_done, end;
    CK(cudaEventCreate(&start)); CK(cudaEventCreate(&coeff_done));
    CK(cudaEventCreate(&poly_done)); CK(cudaEventCreate(&inverse_done)); CK(cudaEventCreate(&end));
    const dim3 radix_grid((k + 255) / 256, ly);
    const dim3 transpose_threads(32, 8);
    auto forward_transpose = [&](const Word* from, Word* to, unsigned rows, unsigned first, unsigned frequencies) {
        const dim3 grid((frequencies + 31) / 32, (k + 31) / 32);
        if (radix3)
            two_axis::transpose_forward<true><<<grid, transpose_threads>>>(
                from + first, to, ly, k, rows, frequencies, twiddle.data, cube);
        else
            two_axis::transpose_forward<false><<<grid, transpose_threads>>>(
                from + first, to, ly, k, rows, frequencies, nullptr, 0);
        CK(cudaGetLastError());
    };
    auto inverse_transpose = [&](const Word* from, Word* to, unsigned first, unsigned frequencies) {
        const dim3 grid((std::min(k, output_count) + 31) / 32, (frequencies + 31) / 32);
        if (radix3)
            two_axis::transpose_inverse<true><<<grid, transpose_threads>>>(
                from, to + first, ly, k, output_count, frequencies, twiddle.data + 2 * k, cube2);
        else
            two_axis::transpose_inverse<false><<<grid, transpose_threads>>>(
                from, to + first, ly, k, output_count, frequencies, nullptr, 0);
        CK(cudaGetLastError());
    };
    const unsigned limb_blocks = (output_words + mersenne::weighted_limb_threads - 1) / mersenne::weighted_limb_threads;
    const unsigned limb_shared = mersenne::weighted_limb_smem(weights.limbs);
    const unsigned normalize_chunk = unsigned(std::min<size_t>(65535, matrix / output_words));
    if (gpu && !normalize_chunk) return 2;
    auto* coefficient_rows = striped ? scratch.data : b.data;
    auto* weighted_a = striped ? scratch.data : b.data;
    auto* weighted_b = striped ? scratch.data + size_t(ma) * ly : scratch.data;
    std::printf("mode=%s p=%u coeff_axis=%u poly_axis=%u stripe_rows=%u ma=%u mb=%u bound=%llu array_bytes=%zu working_array_bytes=%zu\n",
        mode.data(), p, ly, lx, striped ? stripe_rows : ly, ma, mb, *bound * longest,
        matrix * sizeof(Word), workspace * sizeof(Word));
    for (unsigned sample = 0; sample < repeats; ++sample) {
        if (legacy) {
            CK(cudaMemset(a.data, 0, words * sizeof(Word)));
            CK(cudaMemset(b.data, 0, words * sizeof(Word)));
            CK(cudaMemcpy(a.data, input_a.data(), input_a.size() * sizeof(Word), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(b.data, input_b.data(), input_b.size() * sizeof(Word), cudaMemcpyHostToDevice));
        } else if (!gpu) {
            CK(cudaMemcpy(b.data, input_a.data(), input_a.size() * sizeof(Word), cudaMemcpyHostToDevice));
            CK(cudaMemcpy(scratch.data, input_b.data(), input_b.size() * sizeof(Word), cudaMemcpyHostToDevice));
        }
        CK(cudaEventRecord(start));
        if (legacy) {
            y.forward(a.data, ma);
            y.forward(b.data, mb);
            transpose_matrix(a.data, scratch.data, ly, lx);
            transpose_matrix(b.data, a.data, ly, lx);
        } else {
            if (gpu) {
                two_axis::pack_forward<false><<<ma, FUSE_TILE_THREADS, ly * sizeof(Word)>>>(
                    raw.data, weighted_a, input_words, ly, coeff_bits, device_digits, y.ctx.tblF, 0, nullptr);
                two_axis::pack_forward<false><<<mb, FUSE_TILE_THREADS, ly * sizeof(Word)>>>(
                    raw.data + raw_b_offset, weighted_b, input_words, ly, coeff_bits, device_digits, y.ctx.tblF, 0, nullptr);
                CK(cudaGetLastError());
            } else {
                y.forward(b.data, ma);
                y.forward(scratch.data, mb);
            }
            if (!striped) {
                forward_transpose(weighted_a, a.data, ma, 0, ly);
                forward_transpose(weighted_b, b.data, mb, 0, ly);
            }
        }
        CK(cudaEventRecord(coeff_done));
        Word* left = legacy ? scratch.data : a.data;
        Word* right = legacy ? a.data : b.data;
        if (legacy && radix3) {
            outer_radix3<false><<<radix_grid, 256>>>(left, k, twiddle.data, cube, cube2, third);
            outer_radix3<false><<<radix_grid, 256>>>(right, k, twiddle.data, cube, cube2, third);
            CK(cudaGetLastError());
        }
        const unsigned batch_factor = radix3 ? 3 : 1;
        for (unsigned first = 0; first < ly; first += striped ? stripe_rows : ly) {
            const unsigned frequencies = striped ? std::min(stripe_rows, ly - first) : ly;
            if (striped) {
                forward_transpose(weighted_a, left, ma, first, frequencies);
                forward_transpose(weighted_b, right, mb, first, frequencies);
            }
            const unsigned batch = frequencies * batch_factor;
            x.forward(left, batch);
            x.forward(right, batch);
            x.multiply_inverse(left, right, batch);
            // Every input at these frequencies has been consumed, so the
            // product can overwrite those columns of the original rows.
            if (striped) inverse_transpose(left, coefficient_rows, first, frequencies);
        }
        if (legacy && radix3) {
            outer_radix3<true><<<radix_grid, 256>>>(left, k, twiddle.data + 2 * k, cube2, cube, third);
            CK(cudaGetLastError());
        }
        CK(cudaEventRecord(poly_done));
        if (legacy) transpose_matrix(left, coefficient_rows, lx, ly);
        else if (!striped) inverse_transpose(left, coefficient_rows, 0, ly);
        coefficient_inverse<<<output_count, FUSE_TILE_THREADS, ly * sizeof(Word)>>>(
            coefficient_rows, ly, coeff_bits, y.ctx.tblI);
        CK(cudaGetLastError());
        CK(cudaEventRecord(inverse_done));
        if (gpu) {
            // Limb sums borrow one frequency buffer. Canonical rows are
            // shorter than the coefficient rows, so the output can compact
            // over the inverse coefficients in increasing row order.
            for (unsigned i = 0; i < output_count; i += normalize_chunk) {
                const unsigned rows = std::min(output_count - i, normalize_chunk);
                mersenne::weighted_limb_sums<<<dim3(limb_blocks, rows), mersenne::weighted_limb_threads, limb_shared>>>(
                    a.data, coefficient_rows + size_t(i) * ly, inverse_weights.data,
                    device_digits, reinterpret_cast<const mersenne::WeightedLimb*>(spans.data), p, output_words, ly);
                CK(cudaGetLastError());
                two_axis::normalize<false><<<rows, two_axis::carry_threads>>>(
                    reinterpret_cast<uint32_t*>(coefficient_rows) + size_t(i) * output_words, a.data, output_words, p, 0, nullptr);
                CK(cudaGetLastError());
            }
        }
        CK(cudaEventRecord(end));
        CK(cudaEventSynchronize(end));
        float forward_ms, polynomial_ms, inverse_ms, normalize_ms, total_ms;
        CK(cudaEventElapsedTime(&forward_ms, start, coeff_done));
        CK(cudaEventElapsedTime(&polynomial_ms, coeff_done, poly_done));
        CK(cudaEventElapsedTime(&inverse_ms, poly_done, inverse_done));
        CK(cudaEventElapsedTime(&normalize_ms, inverse_done, end));
        CK(cudaEventElapsedTime(&total_ms, start, end));
        if (!verify(coefficient_rows, reference, p, weights, gpu)) return 1;
        std::printf("sample=%u coefficient_forward_ms=%.3f polynomial_ms=%.3f coefficient_inverse_ms=%.3f normalize_ms=%.3f total_ms=%.3f gmp_checked=%u\n",
            sample, forward_ms, polynomial_ms, inverse_ms, normalize_ms, total_ms, sample_count);
    }
    CK(cudaEventDestroy(start)); CK(cudaEventDestroy(coeff_done)); CK(cudaEventDestroy(poly_done));
    CK(cudaEventDestroy(inverse_done)); CK(cudaEventDestroy(end));
    return 0;
}
