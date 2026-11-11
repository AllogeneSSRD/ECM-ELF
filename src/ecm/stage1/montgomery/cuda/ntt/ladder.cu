#include "ladder.h"
#include "profiles.h"
#include "weighted.cuh"
#include "ecm/stage1/ntt_length.h"
#include "../../../../../../kernels/cuda/stage1_ntt_radix5.cuh"
#include "../../../../cuda/ntt/runtime.h"
#include "../../../../../../kernels/cuda/stage1_ntt_dataflow.cuh"
#include "../../../../../../kernels/cuda/stage1_ntt_layout.cuh"
#include "../../../../../../kernels/cuda/stage1_ntt_segments.cuh"
#include <stdexcept>
#include <type_traits>

namespace ecm_mersenne_ntt {
namespace {
using namespace stage1_layout;
using U = unsigned long long;
constexpr unsigned threads = 256;
template<class T> struct Buffer {
    T* data = nullptr;
    explicit Buffer(size_t count) { if (count) CK(cudaMalloc(&data, count * sizeof(T))); }
    ~Buffer() { if (data) cudaFree(data); }
    Buffer(const Buffer&) = delete;
};
struct Stream {
    cudaStream_t value;
    Stream() { CK(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(value); }
};
struct Event {
    cudaEvent_t value;
    Event() { CK(cudaEventCreateWithFlags(&value, cudaEventBlockingSync)); }
    ~Event() { cudaEventDestroy(value); }
};
struct Graph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    ~Graph() { if (exec) cudaGraphExecDestroy(exec); if (graph) cudaGraphDestroy(graph); }
};
struct Tables {
    FuseCtx ctx{};
    ~Tables() { fuse_release(ctx); }
};

template<bool Weighted, bool Mixed = false, bool Modulus = false, bool Deferred = false,
         bool GoodThomas = false, bool Coalesced = false>
__global__ void pack(const uint32_t* state, U* out, Shape s, Indices in,
                     const WeightedDigit* weighted, const U* twiddles = nullptr,
                     Radix5Constants radix5 = {}, const unsigned* headers = nullptr,
                     unsigned deferred_mask = 0, unsigned rotation_step = 0) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned width = Mixed ? s.length / 5 : s.length;
    const unsigned op = blockIdx.y / s.curves;
    const size_t residue = size_t(in.x[op]) * s.curves + blockIdx.y % s.curves;
    const uint32_t* source = Modulus ? nullptr : state + residue * s.words;
    __shared__ unsigned limits[Deferred ? stage1_max_segments : 1];
    const bool pending = Deferred && ((deferred_mask >> op) & 1);
    if (pending) {
        const unsigned segments = stage1_segment_count(s.words);
        stage1_segment_corrections(limits, headers + residue * segments, segments);
    }
    if (i >= width) return;
    const auto read_word = [&](unsigned word) {
        return pending ? stage1_segment_word(source, limits, word, limb_mask(s,word)) : source[word];
    };
    const auto read = [&](unsigned j) { return stage1_ntt_pack_digit_from<Weighted,Modulus>(read_word,s,weighted,j); };
    auto* row = out + size_t(blockIdx.y) * s.length + i;
    if constexpr(Mixed) {
        U x[5];
        const unsigned rotation = GoodThomas ? (i * rotation_step) % 5 : 0;
        if constexpr (Coalesced) {
            extern __shared__ U local[];
            auto* column = local + threadIdx.x;
#pragma unroll
            for (unsigned j = 0; j < 5; ++j)
                column[j * blockDim.x] = read(i + j * width);
#pragma unroll
            for (unsigned j = 0; j < 5; ++j)
                x[j] = column[radix5_plane(j, rotation) * blockDim.x];
        } else {
#pragma unroll
            for (unsigned j = 0; j < 5; ++j)
                x[j] = read(i + (GoodThomas ? radix5_plane(j, rotation) : j) * width);
        }
        radix5_dft(x, radix5);
        row[0] = x[0];
#pragma unroll
        for (unsigned j = 1; j < 5; ++j)
            row[j * width] = GoodThomas ? x[j] : gl_mul(x[j], twiddles[(j - 1) * width + i]);
    } else {
        *row = read(i);
    }
}

struct ProductInputs {
    const U* left[4];
    const U* right[4];
    const U* aa;
    const U* bb;
    const U* modulus;
    U* cached_e;
};

template<bool WarpTail, bool MultiplyAdd = false, bool Loose = false>
__global__ __launch_bounds__(FUSE_TILE_THREADS, WarpTail ? 3 : 1)
void inverse_product_tile(U* out,
                          // Keep dynamically indexed pointers in grid parameter memory.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
                          const __grid_constant__ ProductInputs products,
#else
                          const ProductInputs products,
#endif
                          unsigned length, int tile,
                          const U* twiddles, U scale) {
    const unsigned op = blockIdx.z;
    const size_t curve = size_t(blockIdx.y) * length;
    const unsigned width = 1u << tile;
    const unsigned tile_start = blockIdx.x * width;
    const size_t start = curve + tile_start;
    const U* left = products.left[op] + start;
    const U* right = products.right[op] + start;
    const bool square = left == right;
    const bool last = op + 1 == gridDim.z;
    extern __shared__ U sm[];
    for (unsigned i = threadIdx.x; i < width; i += blockDim.x) {
        U a, bb = 0;
        if constexpr (Loose) {
            if (last) {
                bb = products.bb[start+i];
                a = stage1_ntt_loose_difference(products.aa[start+i], bb, products.modulus[tile_start+i]);
                products.cached_e[start+i] = a;
            } else a = left[i];
        } else a = left[i];
        U product = gl_mul(a, square ? a : right[i]);
        if constexpr (MultiplyAdd)
            if (last) product = gl_add_dev(product, Loose ? bb : products.bb[start+i]);
        sm[i] = gl_mul(product, scale);
    }
    __syncthreads();
    ntt_tile_pass<true, WarpTail>(out + size_t(op) * gridDim.y * length + start,
                                sm, width, tile, twiddles);
}

// A limb's incoming carry is mapped to: 0=kill, 1=propagate, 2=generate.
// Concatenation is associative and preserves low-to-high order.
struct Join {
    __device__ unsigned operator()(unsigned lo, unsigned hi) const { return hi == 1 ? lo : hi; }
};
__device__ uint32_t extract(const U* digits, Shape s, unsigned bit) {
    const unsigned k = bit/s.bpw, off = bit%s.bpw;
    U value = digits[k] >> off;
    for (unsigned shift = s.bpw-off, j = k+1; shift < 32 && j < s.length; shift += s.bpw, ++j)
        value |= digits[j] << shift;
    return uint32_t(value);
}

// One block per output residue. A thread first summarizes a contiguous limb
// interval; warp ballots compute its prefix. Feed the high carry back into the
// low limb (2^p=1). An all-propagate sum is exactly M, represented as zero.
template<bool Fold, bool Select, bool Weighted = false, bool Routed = false, bool Cached = false>
__global__ void combine(uint32_t* state, const U* products, Shape s, Sums ops, Cursor cursor) {
    __shared__ unsigned warp_tags[threads/32];
    extern __shared__ uint32_t saved_raw[];
    const unsigned op = blockIdx.x/s.curves, curve = blockIdx.x%s.curves;
    const unsigned out = output_slot<Routed>(ops.out.x[op],cursor);
    unsigned a = ops.left.x[op], b = ops.right.x[op];
    if constexpr(Select) {
        const unsigned swap = 2u * scalar_bit(cursor);
        a ^= swap; b ^= swap;
    }
    const unsigned stride = (s.words+threads-1)/threads;
    const unsigned begin = min(threadIdx.x * stride, s.words), end = min(begin+stride, s.words);
    const auto raw = [=](unsigned i, uint32_t mask) -> U {
        if constexpr(Weighted) {
            const auto* sums = products + size_t(blockIdx.x) * s.words;
            const unsigned tail_bits = s.p - 32 * (s.words - 1);
            return weighted_limb(sums, s.words, tail_bits, i, mask);
        } else if constexpr(Fold) {
            const auto* digits = products + size_t(blockIdx.x)*s.length;
            return U(extract(digits,s,i*32) & mask) + (extract(digits,s,s.p+i*32) & mask);
        } else {
            return add_limb(state,s,a,b,curve,i,mask,(ops.subtract >> op) & 1);
        }
    };
    unsigned tag = 1, limb_generate = 0, limb_propagate = 0;
    for (unsigned i=begin; i<end; ++i) {
        const uint32_t mask = limb_mask(s,i);
        const U v = raw(i,mask);
        if constexpr (Cached) {
            // The supported exponent limit gives at most 32 limbs per thread.
            const unsigned j = i-begin;
            saved_raw[j*threads + threadIdx.x] = uint32_t(v);
            limb_generate |= unsigned(v > mask) << j;
            limb_propagate |= unsigned(v == mask) << j;
        }
        tag = Join{}(tag, v > mask ? 2u : v == mask ? 1u : 0u);
    }
    const unsigned warp=threadIdx.x/32, lane=threadIdx.x%32;
    const unsigned stop=__ballot_sync(0xffffffffu,tag!=1), generate=__ballot_sync(0xffffffffu,tag==2);
    const auto last=[=](unsigned mask) { return mask ? ((generate >> (31-__clz(mask)))&1u)*2 : 1u; };
    if (lane==31) warp_tags[warp]=last(stop);
    __syncthreads();
    unsigned prefix=1, whole=1;
#pragma unroll
    for (unsigned w=0;w<threads/32;++w) {
        const unsigned t=warp_tags[w];
        if (w<warp) prefix=Join{}(prefix,t);
        whole=Join{}(whole,t);
    }
    prefix=Join{}(prefix,last(stop&((1u<<lane)-1)));
    unsigned carry = prefix == 2 || (prefix == 1 && whole == 2);
    for (unsigned i=begin; i<end; ++i) {
        const uint32_t mask = limb_mask(s,i);
        if constexpr (Cached) {
            const unsigned j = i-begin;
            const uint32_t v = saved_raw[j*threads + threadIdx.x] + carry;
            state[limb(s,out,curve,i)] = whole == 1 ? 0 : v & mask;
            carry = ((limb_generate | (limb_propagate & (0u - carry))) >> j) & 1;
        } else {
            const U v = raw(i,mask) + carry;
            state[limb(s,out,curve,i)] = whole == 1 ? 0 : uint32_t(v) & mask;
            carry = v > mask;
        }
    }
}

__global__ void route(uint32_t* state, Shape s, Cursor cursor) {
    const unsigned j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= s.words) return;
    const unsigned curve = blockIdx.y, bit = scalar_bit(cursor);
    state[limb(s,2*bit,curve,j)] = state[limb(s,XD,curve,j)];
    state[limb(s,2*bit+1,curve,j)] = state[limb(s,ZD,curve,j)];
    state[limb(s,2*(1-bit),curve,j)] = state[limb(s,XA,curve,j)];
    state[limb(s,2*(1-bit)+1,curve,j)] = state[limb(s,ZA,curve,j)];
}
__global__ void advance(unsigned* start, unsigned window) { *start += window; }
} // namespace

size_t memory_budget(size_t free_bytes, unsigned limit_mib) {
    const size_t available = free_bytes - free_bytes / 5;
    return limit_mib ? std::min(available, size_t(limit_mib) * 1024 * 1024) : available;
}

Packing resolve_packing(unsigned p, const ecm_stage1::Stage1NttOptions& options, bool prefer_radix5) {
    if (p < 32 || p > 262144) throw std::invalid_argument("invalid Stage1 NTT exponent");
    Packing s;
    if (options.weighted) {
        const auto usable = [p, &options](unsigned length) {
            const unsigned bpw = (p + length - 1) / length;
            return length <= p && length <= 65536 && bpw <= 31 &&
                (!options.bpw || bpw <= options.bpw) && exact_ok_terms(2ull * length, bpw);
        };
        s.length = options.length ? options.length : 2;
        if (!options.length) {
            while (s.length < 65536 && !usable(s.length)) s.length *= 2;
            // Only the 5/8-length shape between this and the previous power
            // of two can be the shortest safe mixed transform.
            const unsigned shorter = (s.length / 8) * 5;
            if (prefer_radix5 && ecm_stage1::stage1_ntt_supported_radix5(shorter) && usable(shorter) &&
                options.tile <= ecm_stage1::stage1_ntt_length_log(shorter / 5))
                s.length = shorter;
        }
        if (s.length < 2 || (!ecm_stage1::stage1_ntt_power_of_two(s.length) &&
                            !ecm_stage1::stage1_ntt_supported_radix5(s.length)) || !usable(s.length))
            throw std::invalid_argument("unsafe weighted NTT length or digit width");
        s.digits = s.length;
        s.bpw = (p + s.length - 1) / s.length;
    } else {
        const unsigned first_bpw = options.bpw ? options.bpw : 31;
        const unsigned last_bpw = options.bpw ? options.bpw : 16;
        for (unsigned b = first_bpw; b >= last_bpw && b >= 16 && b <= 31; --b) {
            const unsigned digits = (p + b - 1) / b;
            if (exact_ok_terms(digits, b)) {
                s.bpw = b;
                s.digits = digits;
                break;
            }
        }
        if (!s.bpw) throw std::invalid_argument("digit width exceeds the exact Goldilocks coefficient bound");
        s.length = options.length ? options.length : 1;
        if (!options.length)
            while (s.length < 2 * s.digits + 1) s.length *= 2;
        if ((s.length & (s.length - 1)) || s.length < 2 * s.digits + 1 || s.length > 65536)
            throw std::invalid_argument("NTT length must be a power of two covering 2*digits+1, at most 65536");
    }
    return s;
}

static Run run_ladder(int device, unsigned p, unsigned curves, const std::vector<uint32_t>& initial,
                      const std::vector<uint32_t>& scalar_words, unsigned count,
                      const ecm_stage1::Stage1NttOptions& options, const Control* control,
                      unsigned samples) {
    if (p < 32 || p > 262144 || !curves || curves > ecm_stage1::stage1_ntt_max_curves || options.window < -1 || options.window > 64 ||
        !samples || samples > 20 || count > UINT32_MAX - 64 ||
        scalar_words.size() < (size_t(count) + 31) / 32 ||
        (control && control->resume > count))
        throw std::invalid_argument("invalid Mersenne NTT Stage1 shape or scalar");
    Shape s{p, (p+31)/32, curves, 0, 0, 0};
    const size_t stride = size_t(s.words)*curves;
    if (initial.size() != 6*stride) throw std::invalid_argument("invalid Stage1 initial points");
    for (size_t n=0; n<6*curves; ++n) {
        const unsigned top = p%32 ? (1u << (p%32))-1 : UINT32_MAX;
        const auto* v = initial.data()+n*s.words;
        if (v[s.words-1] > top || (v[s.words-1] == top && std::all_of(v,v+s.words-1,
                [](auto x) { return x == UINT32_MAX; })))
            throw std::invalid_argument("noncanonical Stage1 initial residue");
    }
    const auto packing = resolve_packing(p, options);
    s.length=packing.length; s.bpw=packing.bpw; s.digits=packing.digits;
    const bool mixed = ecm_stage1::stage1_ntt_radix5(s.length);
    const unsigned planes = mixed ? 5 : 1;
    const unsigned width = s.length / planes;
    const int log = int(ecm_stage1::stage1_ntt_length_log(width));
    if (options.tile > unsigned(log) || options.tile > 12 ||
        options.outer_max > 4 || options.profile > 2)
        throw std::invalid_argument("NTT tile, outer width or profile is out of range for this length");
    const auto prep = now_s();
    CK(cudaSetDevice(device));
    CK(cudaFree(nullptr)); // Establish the CUDA context before measuring the worker's delta.
    size_t free_before = 0, total = 0;
    CK(cudaMemGetInfo(&free_before, &total));
    Run result;
    unsigned defaults = FusedMultiplyAdd | LooseDifference | DirectRoute | PackedForward |
                        GoodThomasRadix5 | CoalescedRadix5;
    // Wider weighted reductions showed a batch-dependent cache tradeoff.
    if (p <= 65536) defaults |= CachedCarry;
    // Enable segmentation where enough limbs amortize its extra launch.
    if (p > 14 * stage1_carry_segment * 32) {
        defaults |= SegmentedCarry;
        if (mixed || curves == 1) defaults |= DeferredPack;
    }
    const unsigned requested = control ? control->experimental_dataflow.value_or(defaults) : 0;
    const bool good_thomas = mixed && (requested & GoodThomasRadix5);
    const bool coalesced = good_thomas && (requested & CoalescedRadix5);
    const bool square_probe = control && control->modular_square_probe;
    // n = (1 + rotation_step * width) * i + width * j (mod length).
    // The first multiplier is divisible by 5 and congruent to 1 modulo width.
    // omega^5 still gives a primitive width root, with permuted frequencies.
    constexpr unsigned crt_rotation[] = {0, 4, 2, 3, 1};
    const unsigned rotation_step = good_thomas ? crt_rotation[width % 5] : 0;
    if (good_thomas) result.dataflow |= GoodThomasRadix5;
    if (coalesced) result.dataflow |= CoalescedRadix5;
    result.budget_bytes = memory_budget(free_before, options.memory_mib);
    auto observe_memory = [&] {
        size_t free_now = 0;
        CK(cudaMemGetInfo(&free_now, &total));
        if (free_now < free_before)
            result.observed_bytes = std::max(result.observed_bytes, free_before - free_now);
        if (std::max(result.owned_bytes, result.observed_bytes) > result.budget_bytes)
            throw BudgetError("Stage1 NTT exceeds the total VRAM budget");
    };
    auto profile = default_profile(log, options.profile);
    if (control && !options.profile && !options.weighted) {
        cudaDeviceProp prop{};
        CK(cudaGetDeviceProperties(&prop, device));
        if (prop.major == 8 && prop.minor == 9 && prop.multiProcessorCount == 24 &&
            std::string_view(prop.name).find("RTX 4060 Laptop") != std::string_view::npos)
            profile = laptop_4060_profile(log, curves);
    }
    const unsigned window = options.window >= 0 ? unsigned(options.window) : profile.window;
    const FuseOverrides overrides{options.tile ? int(options.tile) : profile.tile,
        options.outer_max ? int(options.outer_max) : profile.outer,
        options.warp_tail >= 0 ? options.warp_tail != 0 : profile.warp_tail};
    Stream stream; const auto q=stream.value;
    const U omega=gl_pow_host(7,(GL_P-1)/s.length), inv=gl_pow_host(omega,GL_P-2);
    const U omega2 = gl_pow_host(omega, planes), inv2 = gl_pow_host(inv, planes);
    const U scale=gl_pow_host(s.length,GL_P-2);
    Tables tables;
    auto& ctx=tables.ctx;
    const auto* resolved = control ? &overrides : nullptr;
    fuse_describe(ctx,width,log,omega2,inv2,true,resolved);
    result.owned_bytes = 11 * sizeof(U) * curves * s.length + slots * stride * sizeof(uint32_t) +
        (std::max(size_t(1), scalar_words.size()) + 1) * sizeof(uint32_t) +
        (fuse_planned_base_words(ctx) + fuse_planned_table_words(ctx)) * sizeof(U);
    if (mixed && !good_thomas) result.owned_bytes += size_t(8) * width * sizeof(U);
    if (options.weighted)
        result.owned_bytes += size_t(s.length) * (sizeof(U) + sizeof(WeightedDigit)) +
                              size_t(s.words) * sizeof(WeightedLimb);
    if (result.owned_bytes > result.budget_bytes)
        throw BudgetError("Stage1 NTT buffers exceed the total VRAM budget");
    const unsigned segment_count = stage1_segment_count(s.words);
    const size_t header_words = size_t(slots) * curves * segment_count;
    const bool segmented = (requested & SegmentedCarry) && options.weighted &&
        header_words * sizeof(unsigned) <= result.budget_bytes - result.owned_bytes;
    if (segmented) {
        result.owned_bytes += header_words * sizeof(unsigned);
        result.dataflow |= SegmentedCarry;
    }
    const U terms = options.weighted ? 2ull * s.length : s.digits;
    const size_t modulus_bytes = size_t(s.length) * sizeof(U);
    // A loose E digit can be twice the canonical maximum. The weighted
    // cyclic reducer handles the full coefficient range without a 2p cutoff.
    const bool loose = (requested & LooseDifference) && options.weighted &&
        exact_ok_terms(2 * terms, s.bpw) && modulus_bytes <= result.budget_bytes - result.owned_bytes;
    if (loose) {
        result.owned_bytes += modulus_bytes;
        result.dataflow |= LooseDifference;
    }
    const U digit_max = (1ull << s.bpw) - 1;
    // Require terms * digit_max^2 + digit_max < Q, without overflow.
    const bool fma = (requested & FusedMultiplyAdd) &&
        (loose ? 2 : 1) * terms <= (GL_P - 1 - digit_max) / digit_max / digit_max;
    if (fma) result.dataflow |= FusedMultiplyAdd;
    fuse_init(ctx,width,log,omega2,inv2,true,resolved);
    if (!ctx.t) throw std::invalid_argument("Stage1 NTT requires an inverse tile");
    const bool warp_tile = ctx.warp_tail && ctx.t >= 5;
    const size_t tile_smem = sizeof(U) << ctx.t;
    const bool routed = requested & DirectRoute;
    const bool packed = (requested & PackedForward) && !mixed && ctx.nms == 0;
    const bool cached = (requested & CachedCarry) && options.weighted && !segmented;
    const bool deferred = segmented && !packed && (requested & DeferredPack);
    if (routed) result.dataflow |= DirectRoute;
    if (packed) result.dataflow |= PackedForward;
    if (cached) result.dataflow |= CachedCarry;
    if (deferred) result.dataflow |= DeferredPack;
    if (control && control->describe) {
        const auto join = [](std::string& stages, int width) {
            if (!stages.empty()) stages += '+';
            stages += std::to_string(width);
        };
        Plan plan{s.length, s.bpw, s.digits, unsigned(ctx.t), unsigned(ctx.m_max), window,
                  FUSE_TILE_THREADS, ctx.nms ? FUSE_OUTER_THREADS : 0u,
                  unsigned(ctx.passes_fwd) + unsigned(mixed),
                  warp_tile, profile.source, options.weighted, {}, {}, result.dataflow};
        if (mixed) plan.forward = "r5";
        for (int i = 0; i < ctx.nms; ++i) join(plan.forward, ctx.ms[i]);
        join(plan.forward, ctx.t);
        join(plan.inverse, ctx.t);
        for (int i = ctx.nms; i > 0; --i) join(plan.inverse, ctx.ms[i-1]);
        if (mixed) plan.inverse += plan.inverse.empty() ? "r5" : "+r5";
        control->describe(plan);
    }
    const size_t carry_smem = size_t((s.words + threads - 1) / threads) * threads * sizeof(uint32_t);
    const auto select_packed_kernel = [warp_tile](auto weight) {
        constexpr bool weighted = decltype(weight)::value;
        return warp_tile ? stage1_ntt_packed_forward<weighted,true,Shape,Indices,WeightedDigit>
                         : stage1_ntt_packed_forward<weighted,false,Shape,Indices,WeightedDigit>;
    };
    const auto packed_kernel = options.weighted ? select_packed_kernel(std::true_type{})
                                                : select_packed_kernel(std::false_type{});
    if (packed) {
        cudaFuncSetAttribute(packed_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        CK(cudaFuncSetAttribute(packed_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(tile_smem)));
    }
    const auto configure_tile = [warp_tile, tile_smem](auto add, auto difference) {
        constexpr bool add_bb = decltype(add)::value, loose_e = decltype(difference)::value;
        const void* kernel = warp_tile ? (const void*)inverse_product_tile<true,add_bb,loose_e>
                                       : (const void*)inverse_product_tile<false,add_bb,loose_e>;
        cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        CK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(tile_smem)));
    };
    configure_tile(std::false_type{},std::false_type{});
    if (loose && fma) configure_tile(std::true_type{},std::true_type{});
    else if (loose) configure_tile(std::false_type{},std::true_type{});
    else if (fma) configure_tile(std::true_type{},std::false_type{});
    ntt_fuse_cache_tables(ctx,omega2,inv2);
    CK(cudaDeviceSynchronize()); // table creation uses the legacy stream
    Buffer<uint32_t> state(slots*stride);
    Buffer<unsigned> carry_headers(segmented ? header_words : 0);
    Buffer<unsigned> start(1);
    Buffer<uint32_t> scalar(std::max(size_t(1), scalar_words.size()));
    Buffer<U> freq(size_t(5)*curves*s.length), fixed(size_t(2)*curves*s.length);
    Buffer<U> modulus(loose ? s.length : 0);
    Buffer<U> inverse(size_t(4)*curves*s.length);
    const size_t mixed_table_words = mixed && !good_thomas ? size_t(4) * width : 0;
    Buffer<U> mixed_fwd(mixed_table_words), mixed_inv(mixed_table_words);
    Radix5Constants radix5_fwd{}, radix5_inv{};
    std::vector<U> fwd_twiddles, inv_twiddles;
    if (mixed) {
        const auto fill = [width, q](std::vector<U>& host, U* device, U root) {
            host.resize(size_t(4) * width);
            for (unsigned plane = 1; plane < 5; ++plane) {
                const U step = gl_pow_host(root, plane);
                U w = 1;
                for (unsigned j = 0; j < width; ++j) {
                    host[(plane - 1) * width + j] = w;
                    w = gl_mul_host(w, step);
                }
            }
            CK(cudaMemcpyAsync(device, host.data(), host.size() * sizeof(U), cudaMemcpyHostToDevice, q));
        };
        if (!good_thomas) {
            fill(fwd_twiddles, mixed_fwd.data, omega);
            fill(inv_twiddles, mixed_inv.data, inv);
        }
        radix5_fwd = radix5_constants(gl_pow_host(omega, width));
        radix5_inv = radix5_constants(gl_pow_host(inv, width));
    }
    const auto weights = options.weighted ? weighted_tables(p, s.length) : WeightedTables{};
    const unsigned limb_smem = weighted_limb_smem(weights.limbs);
    Buffer<WeightedDigit> digit_table(weights.digits.size());
    Buffer<U> inverse_weights(weights.inverse.size());
    Buffer<WeightedLimb> limb_table(weights.limbs.size());
    if (options.weighted) {
        CK(cudaMemcpyAsync(digit_table.data, weights.digits.data(), weights.digits.size()*sizeof(WeightedDigit),
                           cudaMemcpyHostToDevice, q));
        CK(cudaMemcpyAsync(inverse_weights.data, weights.inverse.data(), weights.inverse.size()*sizeof(U),
                           cudaMemcpyHostToDevice, q));
        CK(cudaMemcpyAsync(limb_table.data, weights.limbs.data(), weights.limbs.size()*sizeof(WeightedLimb),
                           cudaMemcpyHostToDevice, q));
    }
    const auto grid=[&](unsigned ops) { return dim3((s.length+threads-1)/threads, ops*curves); };
    const auto mixed_grid=[&](unsigned ops) { return dim3((width+threads-1)/threads, ops*curves); };
    constexpr size_t radix5_smem = 5 * threads * sizeof(U);
    const auto segment_grid=[&](unsigned ops) { return dim3(segment_count, ops*curves); };
    CK(cudaMemcpyAsync(state.data,initial.data(),initial.size()*4,cudaMemcpyHostToDevice,q));
    if (!scalar_words.empty())
        CK(cudaMemcpyAsync(scalar.data,scalar_words.data(),scalar_words.size()*4,cudaMemcpyHostToDevice,q));
    const auto forward_inputs = [&](U* dest, unsigned count, Indices in, unsigned pending = 0) {
        if (packed) {
            packed_kernel<<<dim3(1,count*curves),FUSE_TILE_THREADS,tile_smem,q>>>(
                state.data,dest,s,in,digit_table.data,ctx.t,ctx.tblF);
            CK(cudaGetLastError());
            return;
        }
        if (good_thomas) {
            const auto launch_pfa = [&](auto use_shared) {
                constexpr bool shared = decltype(use_shared)::value;
                if (pending)
                    pack<true,true,false,true,true,shared><<<mixed_grid(count),threads,shared ? radix5_smem : 0,q>>>(
                        state.data,dest,s,in,digit_table.data,nullptr,radix5_fwd,carry_headers.data,pending,rotation_step);
                else
                    pack<true,true,false,false,true,shared><<<mixed_grid(count),threads,shared ? radix5_smem : 0,q>>>(
                        state.data,dest,s,in,digit_table.data,nullptr,radix5_fwd,nullptr,0,rotation_step);
            };
            if (coalesced) launch_pfa(std::true_type{});
            else launch_pfa(std::false_type{});
        } else if (pending && mixed)
            pack<true,true,false,true><<<mixed_grid(count),threads,0,q>>>(
                state.data,dest,s,in,digit_table.data,mixed_fwd.data,radix5_fwd,carry_headers.data,pending);
        else if (pending)
            pack<true,false,false,true><<<grid(count),threads,0,q>>>(
                state.data,dest,s,in,digit_table.data,nullptr,{},carry_headers.data,pending);
        else if (mixed)
            pack<true,true><<<mixed_grid(count),threads,0,q>>>(state.data,dest,s,in,digit_table.data,
                                                               mixed_fwd.data,radix5_fwd);
        else if (options.weighted)
            pack<true><<<grid(count),threads,0,q>>>(state.data,dest,s,in,digit_table.data);
        else
            pack<false><<<grid(count),threads,0,q>>>(state.data,dest,s,in,nullptr);
        ntt_forward_fused(dest,ctx,omega2,count*curves*planes,q);
    };
    forward_inputs(fixed.data, 2, {{A24,Base}});
    if (loose) {
        Shape one = s;
        one.curves = 1;
        if (good_thomas)
            pack<true,true,true,false,true><<<(width+threads-1)/threads,threads,0,q>>>(
                nullptr,modulus.data,one,{},digit_table.data,nullptr,radix5_fwd,nullptr,0,rotation_step);
        else if (mixed)
            pack<true,true,true><<<(width+threads-1)/threads,threads,0,q>>>(nullptr,modulus.data,one,{},
                digit_table.data,mixed_fwd.data,radix5_fwd);
        else
            pack<true,false,true><<<(width+threads-1)/threads,threads,0,q>>>(nullptr,modulus.data,one,{},digit_table.data);
        CK(cudaGetLastError());
        ntt_forward_fused(modulus.data,ctx,omega2,planes,q);
    }
    CK(cudaStreamSynchronize(q));
    const auto finish_carries = [&](unsigned count, Indices out, Cursor cursor,
                                    bool route, unsigned pending) {
        if (route)
            finish_segment_carries<true><<<segment_grid(count),stage1_carry_segment,0,q>>>(
                state.data,carry_headers.data,s,out,cursor,pending);
        else
            finish_segment_carries<false><<<segment_grid(count),stage1_carry_segment,0,q>>>(
                state.data,carry_headers.data,s,out,cursor,pending);
    };
    const auto add_inputs = [&](auto select, unsigned count, Sums ops, Cursor cursor) {
        constexpr bool select_coords = decltype(select)::value;
        if (!segmented) {
            combine<false,select_coords><<<count*curves,threads,0,q>>>(state.data,nullptr,s,ops,cursor);
            return;
        }
        segment_limb_sums<false,select_coords,false><<<segment_grid(count),stage1_carry_segment,0,q>>>(
            state.data,nullptr,carry_headers.data,s,ops,cursor);
        if (!deferred) finish_carries(count,ops.out,{},false,0);
    };
    auto mul=[&](unsigned inputs, unsigned outputs, Indices in, Indices a, Indices b, Indices out,
                 bool build_e = false, bool add_bb = false, Cursor cursor = {},
                 unsigned pending_inputs = 0, unsigned pending_outputs = 0) {
        forward_inputs(freq.data, inputs, in, pending_inputs);
        const size_t plane = size_t(curves) * s.length;
        const auto spectrum = [&](unsigned id) {
            return id < 5 ? freq.data + id * plane : fixed.data + (id - 5) * plane;
        };
        ProductInputs products{};
        for (unsigned op = 0; op < outputs; ++op) {
            products.left[op] = spectrum(a.x[op]);
            products.right[op] = spectrum(b.x[op]);
        }
        if (add_bb || build_e) products.bb = spectrum(3);
        if (build_e) {
            products.aa = spectrum(2);
            products.modulus = modulus.data;
            products.cached_e = spectrum(4);
        }
        const dim3 tiles(s.length >> ctx.t, curves, outputs);
        const auto invert = [&](auto add, auto difference) {
            constexpr bool add_bb = decltype(add)::value, loose_e = decltype(difference)::value;
            if (warp_tile)
                inverse_product_tile<true,add_bb,loose_e><<<tiles,FUSE_TILE_THREADS,tile_smem,q>>>(
                    inverse.data,products,s.length,ctx.t,ctx.tblI,scale);
            else
                inverse_product_tile<false,add_bb,loose_e><<<tiles,FUSE_TILE_THREADS,tile_smem,q>>>(
                    inverse.data,products,s.length,ctx.t,ctx.tblI,scale);
        };
        if (add_bb && build_e) invert(std::true_type{},std::true_type{});
        else if (build_e) invert(std::false_type{},std::true_type{});
        else if (add_bb) invert(std::true_type{},std::false_type{});
        else invert(std::false_type{},std::false_type{});
        CK(cudaGetLastError());
        ntt_inverse_outer_fused(inverse.data,ctx,inv2,outputs*curves*planes,q);
        if (coalesced)
            radix5_inverse<true,true><<<mixed_grid(outputs),threads,radix5_smem,q>>>(
                inverse.data,width,nullptr,radix5_inv,rotation_step);
        else if (good_thomas)
            radix5_inverse<true><<<mixed_grid(outputs),threads,0,q>>>(inverse.data,width,nullptr,radix5_inv,rotation_step);
        else if (mixed)
            radix5_inverse<false><<<mixed_grid(outputs),threads,0,q>>>(inverse.data,width,mixed_inv.data,radix5_inv,0);
        const unsigned batch = outputs * curves;
        const bool route_output = routed && cursor.bits;
        const Sums reduction{{},{},out,0};
        const auto reduce = [&](auto weight, auto direct, auto cache) {
            constexpr bool weighted = decltype(weight)::value;
            constexpr bool route = decltype(direct)::value, cached = decltype(cache)::value;
            combine<true,false,weighted,route,cached><<<batch,threads,cached ? carry_smem : 0,q>>>(
                state.data,freq.data,s,reduction,cursor);
        };
        constexpr std::true_type yes{};
        constexpr std::false_type no{};
        if (options.weighted) {
            const dim3 limb_grid((s.words + weighted_limb_threads - 1) / weighted_limb_threads, batch);
            weighted_limb_sums<<<limb_grid,weighted_limb_threads,limb_smem,q>>>(
                freq.data, inverse.data, inverse_weights.data, digit_table.data,
                limb_table.data, p, s.words, s.length);
            if (segmented) {
                if (route_output)
                    segment_limb_sums<true,false,true><<<segment_grid(outputs),stage1_carry_segment,0,q>>>(
                        state.data,freq.data,carry_headers.data,s,reduction,cursor);
                else
                    segment_limb_sums<true,false,false><<<segment_grid(outputs),stage1_carry_segment,0,q>>>(
                        state.data,freq.data,carry_headers.data,s,reduction,cursor);
                finish_carries(outputs,out,cursor,route_output,pending_outputs);
            } else if (route_output && cached) reduce(yes,yes,yes);
            else if (route_output) reduce(yes,yes,no);
            else if (cached) reduce(yes,no,yes);
            else reduce(yes,no,no);
        } else {
            ecm_launch_carry_cone_kernel<6>(EcmCudaLaunch(grid(outputs),threads,0,q),inverse.data,freq.data,s.length,s.bpw,s.length);
            if (route_output) reduce(no,yes,no);
            else reduce(no,no,no);
        }
    };
    auto step=[&](unsigned i) {
        if (square_probe) {
            mul(1,1,{X0},{0},{0},{X0});
            CK(cudaGetLastError());
            return;
        }
        const Cursor cursor{scalar.data,start.data,i,count};
        add_inputs(std::true_type{},4,
            {{X0,X0,X1,X1},{Z0,Z0,Z1,Z1},{A,B,C,D},0b1010},cursor);
        mul(4,4,{A,B,C,D},{0,1,3,2},{0,1,0,1},{AA,BB,DA,CB},false,false,{},deferred ? 0b1111 : 0);
        add_inputs(std::false_type{},loose ? 2 : 3,
            {{DA,DA,AA},{CB,CB,BB},{Sum,Diff,E},0b110},{});
        // The old X/Z values were consumed by the first combine. Both
        // multiplication batches can now write the final X/Z slots.
        // The second pack resolves the fresh sums. DiffSq and the fused Sum
        // can then stay tentative until pack three. X/Z are always finalized.
        mul(loose ? 4 : 5,4,{Sum,Diff,AA,BB,E},{0,1,2,4},{0,1,3,5},{XA,DiffSq,XD,fma ? Sum : AE},
            loose,fma,cursor,deferred ? (loose ? 0b11 : 0b10011) : 0,
            deferred ? (fma ? 0b1010 : 0b10) : 0);
        if (!fma)
            add_inputs(std::false_type{},1,{{BB},{AE},{Sum},0},{});
        // E is still in frequency plane 4 from the second batch (packed
        // normally or produced by the last product tile). Both
        // reducers only overwrite planes 0–3, and this pack uses planes 0–1.
        constexpr unsigned cached_e = 4;
        mul(2,2,{DiffSq,Sum},{0,cached_e},{6,1},{ZA,ZD},false,false,cursor,deferred ? 0b11 : 0);
        if (!routed)
            route<<<dim3((s.words+threads-1)/threads,curves),threads,0,q>>>(state.data,s,cursor);
        CK(cudaGetLastError());
    };
    Graph full, tail, warm_tail;
    auto capture=[&](Graph& graph, unsigned count) {
        CK(cudaStreamBeginCapture(q,cudaStreamCaptureModeThreadLocal));
        for (unsigned i=0;i<count;++i) step(i);
        advance<<<1,1,0,q>>>(start.data,count);
        CK(cudaStreamEndCapture(q,&graph.graph));
        CK(cudaGraphInstantiateWithFlags(&graph.exec,graph.graph,0));
        CK(cudaGraphUpload(graph.exec,q));
    };
    const unsigned resume = control ? control->resume : 0;
    const unsigned remaining = count - resume;
    const unsigned warmup = control ? control->warmup_iterations : 0;
    if (warmup > remaining) throw std::invalid_argument("Stage1 NTT warmup exceeds scalar");
    if (window && remaining >= window) capture(full,window);
    if (window && remaining % window) capture(tail,remaining % window);
    if (window && warmup % window && warmup % window != remaining % window)
        capture(warm_tail,warmup % window);
    Event begin,end,boundary;
    CK(cudaStreamSynchronize(q));
    observe_memory();
    result.length=s.length; result.digit_bits=s.bpw; result.passes=ctx.passes_fwd+unsigned(mixed); result.tile=ctx.t;
    result.setup_seconds=now_s()-prep;
    result.points.resize(4*stride);
    auto reset = [&] {
        CK(cudaMemcpyAsync(state.data,initial.data(),4*stride*4,cudaMemcpyHostToDevice,q));
        CK(cudaMemcpyAsync(start.data,&resume,sizeof(unsigned),cudaMemcpyHostToDevice,q));
    };
    if (warmup) {
        reset();
        if (window) {
            for (unsigned i=0;i<warmup/window;++i) CK(cudaGraphLaunch(full.exec,q));
            if (warmup % window) CK(cudaGraphLaunch(warm_tail.exec ? warm_tail.exec : tail.exec,q));
        } else for (unsigned i=0;i<warmup;++i) step(i);
        CK(cudaStreamSynchronize(q));
    }
    for (unsigned sample=0; sample<samples; ++sample) {
        reset();
        CK(cudaStreamSynchronize(q));
        const double wall=now_s();
        CK(cudaEventRecord(begin.value,q));
        unsigned cursor = resume;
        unsigned next_poll = resume;
        while (cursor < count) {
            if (control && control->poll && cursor >= next_poll) {
                CK(cudaEventRecord(boundary.value,q));
                CK(cudaEventSynchronize(boundary.value));
                const auto action = control->poll(cursor);
                if (action.save && control->checkpoint) {
                    CK(cudaMemcpy(result.points.data(),state.data,result.points.size()*4,cudaMemcpyDeviceToHost));
                    control->checkpoint(cursor, result.points);
                }
                if (action.stop) {
                    result.stopped = true;
                    break;
                }
                next_poll = cursor + (window ? std::max(window, 128u) : 128u);
            }
            if (!window) {
                step(cursor - resume);
                ++cursor;
            } else if (count - cursor >= window) {
                CK(cudaGraphLaunch(full.exec,q));
                cursor += window;
            } else {
                CK(cudaGraphLaunch(tail.exec,q));
                cursor = count;
            }
        }
        result.completed_bits = cursor;
        CK(cudaEventRecord(end.value,q));
        CK(cudaEventSynchronize(end.value));
        result.wall_ms.push_back((now_s()-wall)*1000);
        float ms=0; CK(cudaEventElapsedTime(&ms,begin.value,end.value));
        result.device_ms.push_back(ms);
        const double read=now_s();
        std::vector<uint32_t> got(4*stride);
        CK(cudaMemcpy(got.data(),state.data,got.size()*4,cudaMemcpyDeviceToHost));
        result.readback_seconds+=now_s()-read;
        if (sample && got!=result.points) throw std::runtime_error("nondeterministic NTT Stage1 endpoint");
        result.points=std::move(got);
    }
    observe_memory();
    return result;
}
Run ladder(int device, unsigned p, unsigned curves, const std::vector<uint32_t>& initial,
           const std::vector<uint32_t>& scalar, unsigned count,
           const ecm_stage1::Stage1NttOptions& options, const Control& control, unsigned samples) {
    return run_ladder(device, p, curves, initial, scalar, count, options, &control, samples);
}

Run ladder(int device, unsigned p, unsigned curves, const std::vector<uint32_t>& initial,
           const std::vector<uint8_t>& bits, unsigned window, unsigned samples, bool weighted) {
    if (bits.empty() || bits.size() > UINT32_MAX - 64)
        throw std::invalid_argument("invalid probe scalar size");
    std::vector<uint32_t> packed((bits.size()+31)/32);
    for (size_t i = 0; i < bits.size(); ++i) {
        if (bits[i] > 1) throw std::invalid_argument("invalid probe scalar bit");
        const size_t j = bits.size() - 1 - i;
        packed[j / 32] |= uint32_t(bits[i]) << (j % 32);
    }
    ecm_stage1::Stage1NttOptions options;
    options.window = window;
    options.weighted = weighted;
    return run_ladder(device, p, curves, initial, packed, unsigned(bits.size()), options, nullptr, samples);
}
} // namespace ecm_mersenne_ntt
