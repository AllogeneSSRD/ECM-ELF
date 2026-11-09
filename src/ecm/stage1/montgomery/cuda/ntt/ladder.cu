#include "ladder.h"
#include "profiles.h"
#include "../../../../cuda/ntt/runtime.h"
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
    explicit Buffer(size_t count) { CK(cudaMalloc(&data, count * sizeof(T))); }
    ~Buffer() { cudaFree(data); }
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

__global__ void pack(const uint32_t* state, U* out, Shape s, Indices in) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= s.length) return;
    U v = 0;
    if (i < s.digits) {
        const unsigned bit = i*s.bpw, word = bit/32, offset = bit%32;
        const auto pos = limb(s, in.x[blockIdx.y/s.curves], blockIdx.y%s.curves, word);
        v = state[pos];
        if (word+1 < s.words) v |= U(state[pos+1]) << 32;
        v = (v >> offset) & ((1ull << s.bpw)-1);
    }
    out[size_t(blockIdx.y)*s.length+i] = v;
}

// Forward each unique operand once. The existing inverse kernel fuses the
// pointwise product and scaling; gather keeps the shared spectra immutable.
__global__ void gather(const U* input, const U* fixed, U* x, U* y,
                       Shape s, Indices left, Indices right) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= s.length) return;
    const unsigned op = blockIdx.y/s.curves, curve = blockIdx.y%s.curves;
    const unsigned a = left.x[op], b = right.x[op];
    const auto at = [=](unsigned id) {
        return (id < 5 ? input : fixed)[(size_t(id < 5 ? id : id-5)*s.curves+curve)*s.length+i];
    };
    x[size_t(blockIdx.y)*s.length+i] = at(a);
    y[size_t(blockIdx.y)*s.length+i] = at(b);
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
template<bool Fold, bool Select>
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
        if constexpr(Fold) {
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

static Run run_ladder(int device, unsigned p, unsigned curves, const std::vector<uint32_t>& initial,
                      const std::vector<uint32_t>& scalar_words, unsigned count,
                      const ecm_stage1::Stage1NttOptions& options, const Control* control,
                      unsigned samples) {
    if (p < 32 || p > 262144 || !curves || curves > 8 || options.window < -1 || options.window > 64 ||
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
    int log = 0;
    for (unsigned n = s.length; n > 1; n >>= 1) ++log;
    if (options.tile > unsigned(log) || options.tile > 12 ||
        options.outer_max > 4 || options.profile > 2)
        throw std::invalid_argument("NTT tile, outer width or profile is out of range for this length");
    const auto prep = now_s();
    CK(cudaSetDevice(device));
    auto profile = default_profile(log, options.profile);
    if (control && !options.profile) {
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
    fuse_init(ctx,s.length,log,omega,inv,true,control ? &overrides : nullptr);
    if (control && control->describe) {
        const auto join = [](std::string& stages, int width) {
            if (!stages.empty()) stages += '+';
            stages += std::to_string(width);
        };
        Plan plan{s.length, s.bpw, s.digits, unsigned(ctx.t), unsigned(ctx.m_max), window,
                  FUSE_TILE_THREADS, FUSE_OUTER_THREADS, unsigned(ctx.passes_fwd),
                  ctx.warp_tail && ctx.t >= 5, profile.source, {}, {}};
        for (int i = 0; i < ctx.nms; ++i) join(plan.forward, ctx.ms[i]);
        join(plan.forward, ctx.t);
        join(plan.inverse, ctx.t);
        for (int i = ctx.nms; i > 0; --i) join(plan.inverse, ctx.ms[i-1]);
        control->describe(plan);
    }
    ntt_fuse_cache_tables(ctx,omega,inv);
    CK(cudaDeviceSynchronize()); // table creation uses the legacy stream
    Buffer<uint32_t> state(slots*stride);
    Buffer<unsigned> start(1);
    Buffer<uint32_t> scalar(std::max(size_t(1), scalar_words.size()));
    Buffer<U> freq(size_t(5)*curves*s.length), fixed(size_t(2)*curves*s.length);
    Buffer<U> left(size_t(4)*curves*s.length), right(size_t(4)*curves*s.length);
    const auto grid=[&](unsigned ops) { return dim3((s.length+threads-1)/threads, ops*curves); };
    CK(cudaMemcpyAsync(state.data,initial.data(),initial.size()*4,cudaMemcpyHostToDevice,q));
    if (!scalar_words.empty())
        CK(cudaMemcpyAsync(scalar.data,scalar_words.data(),scalar_words.size()*4,cudaMemcpyHostToDevice,q));
    pack<<<grid(2),threads,0,q>>>(state.data,fixed.data,s,{{A24,Base}});
    ntt_forward_fused(fixed.data,ctx,omega,2*curves,q);
    CK(cudaStreamSynchronize(q));
    auto mul=[&](unsigned inputs, unsigned outputs, Indices in, Indices a, Indices b, Indices out) {
        pack<<<grid(inputs),threads,0,q>>>(state.data,freq.data,s,in);
        ntt_forward_fused(freq.data,ctx,omega,inputs*curves,q);
        gather<<<grid(outputs),threads,0,q>>>(freq.data,fixed.data,left.data,right.data,s,a,b);
        ntt_inverse_fused(left.data,right.data,ctx,inv,scale,outputs*curves,q);
        ecm_launch_carry_cone_kernel<6>(EcmCudaLaunch(grid(outputs),threads,0,q),left.data,freq.data,s.length,s.bpw,s.length);
        combine<true,false><<<outputs*curves,threads,0,q>>>(state.data,freq.data,s,{{},{},out,0},{});
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
        mul(3,2,{DiffSq,E,Sum},{0,1},{6,2},{ZA,ZD});
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
    Run result;
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
    return result;
}
Run ladder(int device, unsigned p, unsigned curves, const std::vector<uint32_t>& initial,
           const std::vector<uint32_t>& scalar, unsigned count,
           const ecm_stage1::Stage1NttOptions& options, const Control& control) {
    return run_ladder(device, p, curves, initial, scalar, count, options, &control, 1);
}

Run ladder(int device, unsigned p, unsigned curves, const std::vector<uint32_t>& initial,
           const std::vector<uint8_t>& bits, unsigned window, unsigned samples) {
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
    return run_ladder(device, p, curves, initial, packed, unsigned(bits.size()), options, nullptr, samples);
}
} // namespace ecm_mersenne_ntt
