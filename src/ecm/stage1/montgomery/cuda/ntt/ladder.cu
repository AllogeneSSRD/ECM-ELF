#include "ladder.h"
#include "profiles.h"
#include "weighted.cuh"
#include "../../../../cuda/ntt/runtime.h"
#include "../../../../cuda/ntt/device/tile_pass.cuh"
#include <stdexcept>

namespace ecm_mersenne_ntt {
namespace {
using U = unsigned long long;
constexpr unsigned threads = 256;
constexpr unsigned slots = 23;
enum Slot : unsigned {
    X0, Z0, X1, Z1, A24, Base, A, B, C, D, AA, BB, DA, CB,
    Sum, Diff, E, XA, DiffSq, XD, AE, ZA, ZD
};
struct Indices { unsigned x[5]; };
struct Sums { Indices left, right, out; unsigned subtract; };
struct Shape { unsigned p, words, curves, length, bpw, digits; };
struct Cursor { const uint32_t* bits; const unsigned* start; unsigned offset, count; };

__device__ unsigned scalar_bit(Cursor cursor) {
    const unsigned i = cursor.count - 1 - (*cursor.start + cursor.offset);
    return (cursor.bits[i / 32] >> (i % 32)) & 1u;
}

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

__device__ size_t limb(Shape s, unsigned slot, unsigned curve, unsigned word) {
    return (size_t(slot) * s.curves + curve) * s.words + word;
}
__device__ uint32_t limb_mask(Shape s, unsigned i) {
    const unsigned rest = s.p % 32;
    return i + 1 == s.words && rest ? (1u << rest) - 1 : 0xffffffffu;
}

template<bool Weighted>
__global__ void pack(const uint32_t* state, U* out, Shape s, Indices in,
                     const WeightedDigit* weighted) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= s.length) return;
    U v = 0;
    if (i < s.digits) {
        const auto digit = Weighted ? weighted[i] : WeightedDigit{1, i*s.bpw, s.bpw};
        const unsigned word = digit.bit / 32, offset = digit.bit % 32;
        const auto pos = limb(s, in.x[blockIdx.y/s.curves], blockIdx.y%s.curves, word);
        v = state[pos];
        if (word+1 < s.words) v |= U(state[pos+1]) << 32;
        v = (v >> offset) & ((1ull << digit.width)-1);
        if constexpr(Weighted) v = gl_mul(v, digit.weight);
    }
    out[size_t(blockIdx.y)*s.length+i] = v;
}

struct ProductInputs {
    const U* left[4];
    const U* right[4];
};

template<bool WarpTail>
__global__ __launch_bounds__(FUSE_TILE_THREADS, WarpTail ? 3 : 1)
void inverse_product_tile(U* out, ProductInputs products, unsigned length, int tile,
                          const U* twiddles, U scale) {
    const unsigned op = blockIdx.z;
    const size_t curve = size_t(blockIdx.y) * length;
    const unsigned width = 1u << tile;
    const size_t start = curve + blockIdx.x * width;
    const U* left = products.left[op] + start;
    const U* right = products.right[op] + start;
    const bool square = left == right;
    extern __shared__ U sm[];
    for (unsigned i = threadIdx.x; i < width; i += blockDim.x) {
        const U a = left[i];
        sm[i] = gl_mul(gl_mul(a, square ? a : right[i]), scale);
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
template<bool Fold, bool Select, bool Weighted = false>
__global__ void combine(uint32_t* state, const U* products, Shape s, Sums ops, Cursor cursor) {
    __shared__ unsigned warp_tags[threads/32];
    const unsigned op = blockIdx.x/s.curves, curve = blockIdx.x%s.curves;
    const unsigned out = ops.out.x[op];
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
            const auto rhs = state[limb(s,b,curve,i)];
            return U(state[limb(s,a,curve,i)]) + (((ops.subtract >> op) & 1) ? (rhs ^ mask) : rhs);
        }
    };
    unsigned tag = 1;
    for (unsigned i=begin; i<end; ++i) {
        const uint32_t mask = limb_mask(s,i);
        const U v = raw(i,mask);
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
        const U v = raw(i,mask) + carry;
        state[limb(s,out,curve,i)] = whole == 1 ? 0 : uint32_t(v) & mask;
        carry = v > mask;
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
    if (options.weighted) {
        const auto usable = [p, &options](unsigned length) {
            const unsigned bpw = (p + length - 1) / length;
            return length <= p && length <= 65536 && bpw <= 31 &&
                (!options.bpw || bpw <= options.bpw) && exact_ok_terms(2ull * length, bpw);
        };
        s.length = options.length ? options.length : 2;
        if (!options.length)
            while (s.length < 65536 && !usable(s.length)) s.length *= 2;
        if (s.length < 2 || (s.length & (s.length - 1)) || !usable(s.length))
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
    int log = 0;
    for (unsigned n = s.length; n > 1; n >>= 1) ++log;
    if (options.tile > unsigned(log) || options.tile > 12 ||
        options.outer_max > 4 || options.profile > 2)
        throw std::invalid_argument("NTT tile, outer width or profile is out of range for this length");
    const auto prep = now_s();
    CK(cudaSetDevice(device));
    CK(cudaFree(nullptr)); // Establish the CUDA context before measuring the worker's delta.
    size_t free_before = 0, total = 0;
    CK(cudaMemGetInfo(&free_before, &total));
    Run result;
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
    const U scale=gl_pow_host(s.length,GL_P-2);
    Tables tables;
    auto& ctx=tables.ctx;
    const auto* resolved = control ? &overrides : nullptr;
    fuse_describe(ctx,s.length,log,omega,inv,true,resolved);
    result.owned_bytes = 11 * sizeof(U) * curves * s.length + slots * stride * sizeof(uint32_t) +
        (std::max(size_t(1), scalar_words.size()) + 1) * sizeof(uint32_t) +
        (fuse_planned_base_words(ctx) + fuse_planned_table_words(ctx)) * sizeof(U);
    if (options.weighted)
        result.owned_bytes += size_t(s.length) * (sizeof(U) + sizeof(WeightedDigit)) +
                              size_t(s.words) * sizeof(WeightedLimb);
    if (result.owned_bytes > result.budget_bytes)
        throw BudgetError("Stage1 NTT buffers exceed the total VRAM budget");
    fuse_init(ctx,s.length,log,omega,inv,true,resolved);
    if (control && control->describe) {
        const auto join = [](std::string& stages, int width) {
            if (!stages.empty()) stages += '+';
            stages += std::to_string(width);
        };
        Plan plan{s.length, s.bpw, s.digits, unsigned(ctx.t), unsigned(ctx.m_max), window,
                  FUSE_TILE_THREADS, FUSE_OUTER_THREADS, unsigned(ctx.passes_fwd),
                  ctx.warp_tail && ctx.t >= 5, profile.source, options.weighted, {}, {}};
        for (int i = 0; i < ctx.nms; ++i) join(plan.forward, ctx.ms[i]);
        join(plan.forward, ctx.t);
        join(plan.inverse, ctx.t);
        for (int i = ctx.nms; i > 0; --i) join(plan.inverse, ctx.ms[i-1]);
        control->describe(plan);
    }
    if (!ctx.t) throw std::invalid_argument("Stage1 NTT requires an inverse tile");
    const bool warp_tile = ctx.warp_tail && ctx.t >= 5;
    const size_t tile_smem = sizeof(U) << ctx.t;
    const void* tile_kernel = warp_tile ? (const void*)inverse_product_tile<true>
                                       : (const void*)inverse_product_tile<false>;
    cudaFuncSetAttribute(tile_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
    CK(cudaFuncSetAttribute(tile_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(tile_smem)));
    ntt_fuse_cache_tables(ctx,omega,inv);
    CK(cudaDeviceSynchronize()); // table creation uses the legacy stream
    Buffer<uint32_t> state(slots*stride);
    Buffer<unsigned> start(1);
    Buffer<uint32_t> scalar(std::max(size_t(1), scalar_words.size()));
    Buffer<U> freq(size_t(5)*curves*s.length), fixed(size_t(2)*curves*s.length);
    Buffer<U> inverse(size_t(4)*curves*s.length);
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
    CK(cudaMemcpyAsync(state.data,initial.data(),initial.size()*4,cudaMemcpyHostToDevice,q));
    if (!scalar_words.empty())
        CK(cudaMemcpyAsync(scalar.data,scalar_words.data(),scalar_words.size()*4,cudaMemcpyHostToDevice,q));
    const auto pack_inputs = [&](U* dest, unsigned count, Indices in) {
        if (options.weighted)
            pack<true><<<grid(count),threads,0,q>>>(state.data,dest,s,in,digit_table.data);
        else
            pack<false><<<grid(count),threads,0,q>>>(state.data,dest,s,in,nullptr);
    };
    pack_inputs(fixed.data, 2, {{A24,Base}});
    ntt_forward_fused(fixed.data,ctx,omega,2*curves,q);
    CK(cudaStreamSynchronize(q));
    auto mul=[&](unsigned inputs, unsigned outputs, Indices in, Indices a, Indices b, Indices out) {
        pack_inputs(freq.data, inputs, in);
        ntt_forward_fused(freq.data,ctx,omega,inputs*curves,q);
        const size_t plane = size_t(curves) * s.length;
        const auto spectrum = [&](unsigned id) {
            return id < 5 ? freq.data + id * plane : fixed.data + (id - 5) * plane;
        };
        ProductInputs products{};
        for (unsigned op = 0; op < outputs; ++op) {
            products.left[op] = spectrum(a.x[op]);
            products.right[op] = spectrum(b.x[op]);
        }
        const dim3 tiles(s.length >> ctx.t, curves, outputs);
        if (warp_tile)
            inverse_product_tile<true><<<tiles,FUSE_TILE_THREADS,tile_smem,q>>>(
                inverse.data,products,s.length,ctx.t,ctx.tblI,scale);
        else
            inverse_product_tile<false><<<tiles,FUSE_TILE_THREADS,tile_smem,q>>>(
                inverse.data,products,s.length,ctx.t,ctx.tblI,scale);
        CK(cudaGetLastError());
        ntt_inverse_outer_fused(inverse.data,ctx,inv,outputs*curves,q);
        if (options.weighted) {
            const unsigned batch = outputs * curves;
            const dim3 limb_grid((s.words + weighted_limb_threads - 1) / weighted_limb_threads, batch);
            weighted_limb_sums<<<limb_grid,weighted_limb_threads,limb_smem,q>>>(
                freq.data, inverse.data, inverse_weights.data, digit_table.data,
                limb_table.data, p, s.words, s.length);
            combine<true,false,true><<<batch,threads,0,q>>>(state.data,freq.data,s,{{},{},out,0},{});
        } else {
            ecm_launch_carry_cone_kernel<6>(EcmCudaLaunch(grid(outputs),threads,0,q),inverse.data,freq.data,s.length,s.bpw,s.length);
            combine<true,false><<<outputs*curves,threads,0,q>>>(state.data,freq.data,s,{{},{},out,0},{});
        }
    };
    auto step=[&](unsigned i) {
        const Cursor cursor{scalar.data,start.data,i,count};
        combine<false,true><<<4*curves,threads,0,q>>>(state.data,nullptr,s,
            {{X0,X0,X1,X1},{Z0,Z0,Z1,Z1},{A,B,C,D},0b1010},cursor);
        mul(4,4,{A,B,C,D},{0,1,3,2},{0,1,0,1},{AA,BB,DA,CB});
        combine<false,false><<<3*curves,threads,0,q>>>(state.data,nullptr,s,
            {{DA,DA,AA},{CB,CB,BB},{Sum,Diff,E},0b110},{});
        mul(5,4,{Sum,Diff,AA,BB,E},{0,1,2,4},{0,1,3,5},{XA,DiffSq,XD,AE});
        combine<false,false><<<curves,threads,0,q>>>(state.data,nullptr,s,{{BB},{AE},{Sum},0},{});
        // E is still in frequency plane 4 from the second batch. Both
        // reducers only overwrite planes 0–3, and this pack uses planes 0–1.
        constexpr unsigned cached_e = 4;
        mul(2,2,{DiffSq,Sum},{0,cached_e},{6,1},{ZA,ZD});
        route<<<dim3((s.words+threads-1)/threads,curves),threads,0,q>>>(state.data,s,cursor);
        CK(cudaGetLastError());
    };
    Graph full, tail;
    auto capture=[&](Graph& graph, unsigned count) {
        CK(cudaStreamBeginCapture(q,cudaStreamCaptureModeThreadLocal));
        for (unsigned i=0;i<count;++i) step(i);
        advance<<<1,1,0,q>>>(start.data,count);
        CK(cudaStreamEndCapture(q,&graph.graph));
        CK(cudaGraphInstantiate(&graph.exec,graph.graph,0));
        CK(cudaGraphUpload(graph.exec,q));
    };
    const unsigned resume = control ? control->resume : 0;
    const unsigned remaining = count - resume;
    if (window && remaining >= window) capture(full,window);
    if (window && remaining % window) capture(tail,remaining % window);
    Event begin,end,boundary;
    CK(cudaStreamSynchronize(q));
    observe_memory();
    result.length=s.length; result.digit_bits=s.bpw; result.passes=ctx.passes_fwd; result.tile=ctx.t;
    result.setup_seconds=now_s()-prep;
    result.points.resize(4*stride);
    for (unsigned sample=0; sample<samples; ++sample) {
        CK(cudaMemcpyAsync(state.data,initial.data(),4*stride*4,cudaMemcpyHostToDevice,q));
        CK(cudaMemcpyAsync(start.data,&resume,sizeof(unsigned),cudaMemcpyHostToDevice,q));
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
