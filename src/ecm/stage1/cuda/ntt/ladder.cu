#include "ladder.h"
#include "pm1_replay.h"
#include "pm1_probe.h"
#include <mutex>
#include "host_lanes.h"
#include "profiles.h"
#include "weighted.cuh"
#include "ecm/stage1/ntt_sizes.h"
#include "ecm/stage1/ntt_length.h"
#include "ecm/integer.h"
#include "../../../../../kernels/cuda/stage1_ntt_radix5.cuh"
#include "../../../cuda/ntt/runtime.h"
#include "../../../../../kernels/cuda/stage1_ntt_dataflow.cuh"
#include "../../../../../kernels/cuda/stage1_ntt_layout.cuh"
#include "../../../../../kernels/cuda/stage1_ntt_segments.cuh"
#include <array>
#include <cmath>
#include <map>
#include <stdexcept>
#include <type_traits>

namespace ecm_mersenne_ntt {
namespace {
using namespace stage1_layout;
using U = unsigned long long;
constexpr unsigned threads = 256;
__device__ __forceinline__ unsigned cubic_digit(unsigned plane, unsigned column, unsigned width) {
    // Digit i belongs to (i mod 3, i mod Width). Width is its own inverse modulo 3.
    return column + width * ((plane + 3 - column % 3) * (width % 3) % 3);
}

template<int Tile, bool Transform, bool WarpTail, bool Modulus = false>
__global__ void pack_cubic(const uint32_t* state, U* out, Shape s, Indices in,
                           const WeightedDigit* weights, const U* twiddles = nullptr) {
    const unsigned width = Tile ? 1u << Tile : s.length / 3;
    const unsigned plane = blockIdx.y % 3, row = blockIdx.y / 3;
    const auto* source = Modulus ? nullptr : state + limb(s,in.x[row / s.curves],row % s.curves,0);
    auto* dest = out + size_t(blockIdx.y) * width;
    const auto load = [=](unsigned column) {
        const auto word = [=](unsigned index) { return source[index]; };
        return stage1_ntt_pack_digit_from<true,Modulus>(word,s,weights,cubic_digit(plane,column,width));
    };
    if constexpr (Transform) {
        extern __shared__ U scratch[];
        stage1_ntt_forward_from<WarpTail,false,Tile>(load,dest,scratch,Tile,twiddles);
    } else {
        for (unsigned column = blockIdx.x * blockDim.x + threadIdx.x; column < width;
             column += gridDim.x * blockDim.x)
            dest[column] = load(column);
    }
}

struct Cubic { U x, y, z; };

__device__ __forceinline__ Cubic cubic_load(const U* p, unsigned width) { return {p[0],p[width],p[2*width]}; }
__device__ __forceinline__ void cubic_store(U* p, unsigned width, Cubic a) {
    p[0]=a.x; p[width]=a.y; p[2*width]=a.z;
}
__device__ __forceinline__ U cubic_scale(U x, unsigned shift) {
    if (shift) x=gl_add_dev(x,x);
    if (shift==2) x=gl_add_dev(x,x);
    return x;
}
__device__ __forceinline__ Cubic cubic_add(Cubic a,Cubic b) {
    return {gl_add_dev(a.x,b.x),gl_add_dev(a.y,b.y),gl_add_dev(a.z,b.z)};
}
__device__ __forceinline__ Cubic cubic_sub(Cubic a,Cubic b) {
    return {gl_sub_dev(a.x,b.x),gl_sub_dev(a.y,b.y),gl_sub_dev(a.z,b.z)};
}
__device__ __forceinline__ Cubic cubic_square(Cubic a,unsigned shift) {
    const U xx=gl_mul(a.x,a.x), zz=gl_mul(a.z,a.z);
    const U xy=gl_mul(a.x,gl_add_dev(a.y,a.y)), yz=gl_mul(a.z,gl_add_dev(a.y,a.y));
    const auto diff=gl_add_dev(gl_sub_dev(a.x,a.y),a.z);
    const U mid=gl_sub_dev(gl_add_dev(gl_add_dev(xy,yz),gl_mul(diff,diff)),gl_add_dev(xx,zz));
    return {gl_add_dev(xx,cubic_scale(yz,shift)),gl_add_dev(xy,cubic_scale(zz,shift)),mid};
}
__device__ __forceinline__ Cubic cubic_product(Cubic a,Cubic b,unsigned shift) {
    const U xx=gl_mul(a.x,b.x), yy=gl_mul(a.y,b.y), zz=gl_mul(a.z,b.z);
    const auto cross = [](U a0,U a1,U b0,U b1,U m0,U m1) {
        return gl_sub_dev(gl_mul(gl_add_dev(a0,a1),gl_add_dev(b0,b1)), gl_add_dev(m0,m1));
    };
    const U xy=cross(a.x,a.y,b.x,b.y,xx,yy), xz=cross(a.x,a.z,b.x,b.z,xx,zz);
    const U yz=cross(a.y,a.z,b.y,b.z,yy,zz);
    return {gl_add_dev(xx,cubic_scale(yz,shift)),gl_add_dev(xy,cubic_scale(zz,shift)),gl_add_dev(xz,yy)};
}

__global__ void cubic_square_bias(U* values, unsigned width, unsigned shift) {
    auto* p=values+blockIdx.x*blockDim.x+threadIdx.x;
    cubic_store(p,width,cubic_square(cubic_load(p,width),shift));
}

__global__ void cubic_double(const U* spectra,U* out,const U* bias,unsigned width,unsigned shift) {
    const unsigned j=blockIdx.y*blockDim.x+threadIdx.x, length=3*width;
    const auto offset=size_t(blockIdx.x)*length+j;
    const auto stride=size_t(gridDim.x)*length;
    const auto x=cubic_load(spectra+offset,width), y=cubic_load(spectra+stride+offset,width);
    const auto xx=cubic_square(x,shift), yy=cubic_square(y,shift);
    const auto zz=cubic_square(cubic_load(spectra+2*stride+offset,width),shift);
    const auto xy=cubic_product(x,y,shift), sum=cubic_add(xx,yy), mod=cubic_load(bias+j,width);
    const auto minus_z=cubic_sub(mod,zz);
    cubic_store(out+offset,width,cubic_add(xy,xy));
    cubic_store(out+stride+offset,width,cubic_add(sum,cubic_add(minus_z,minus_z)));
    cubic_store(out+2*stride+offset,width,sum);
    cubic_store(out+3*stride+offset,width,cubic_add(xx,cubic_sub(mod,yy)));
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

template<bool Weighted, bool Mixed = false, bool Modulus = false, bool Deferred = false,
         bool Pfa = false, bool Coalesced = false>
__global__ void pack(const uint32_t* state, U* out, Shape s, Indices in,
                     const WeightedDigit* weighted, const U* twiddles = nullptr,
                     Radix5Constants radix5 = {}, const unsigned* headers = nullptr,
                     unsigned deferred_mask = 0, unsigned rotation_step = 0) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned width = Mixed ? s.length / 5 : s.length;
    const unsigned op = blockIdx.y / s.curves;
    const size_t residue = size_t(in.x[op]) * s.curves + blockIdx.y % s.curves;
    const uint32_t* source = Modulus ? nullptr : state + residue * s.words;
    __shared__ unsigned local[Deferred ? inline_carry_segments : 1];
    const unsigned* limits = local;
    const bool pending = Deferred && ((deferred_mask >> op) & 1);
    if (pending) {
        const unsigned segments = stage1_segment_count(s.words);
        limits = headers + residue * segments;
        if (segments <= inline_carry_segments) {
            stage1_segment_corrections(local, limits, segments);
            limits = local;
        }
    }
    if (i >= width) return;
    const auto read_word = [&](unsigned word) {
        return pending ? stage1_segment_word(source, limits, word, limb_mask(s,word)) : source[word];
    };
    const auto read = [&](unsigned j) { return stage1_ntt_pack_digit_from<Weighted,Modulus>(read_word,s,weighted,j); };
    auto* row = out + size_t(blockIdx.y) * s.length + i;
    if constexpr(Mixed) {
        U x[5];
        const unsigned rotation = Pfa ? (i * rotation_step) % 5 : 0;
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
                x[j] = read(i + (Pfa ? radix5_plane(j, rotation) : j) * width);
        }
        radix5_dft(x, radix5);
        row[0] = x[0];
#pragma unroll
        for (unsigned j = 1; j < 5; ++j)
            row[j * width] = Pfa ? x[j] : gl_mul(x[j], twiddles[(j - 1) * width + i]);
    } else {
        *row = read(i);
    }
}

template<int Layers>
__device__ __forceinline__ void packed_outer_dif(U (&x)[1u << Layers], U root, const U* radix) {
    constexpr unsigned count = 1u << Layers;
    if constexpr (Layers == 1 || Layers == 3) {
#pragma unroll
        for (unsigned i = 0; i < count / 2; ++i) {
            const auto a = x[i], b = x[i + count / 2];
            x[i] = gl_add_dev(a, b);
            x[i + count / 2] = gl_mul(gl_sub_dev(a, b), i ? gl_mul(root, radix[i]) : root);
        }
        if constexpr (Layers == 3) root = gl_mul(root, root);
    }
    if constexpr (Layers >= 2) {
        // The radix-4 row starts after the larger radix rows (count - 4 entries).
        const U rotated = gl_mul(root, radix[(count - 4) + 1]);
        const U squared = gl_mul(root, root);
#pragma unroll
        for (unsigned i = 0; i < count; i += 4) {
            const auto q = tile_radix4_dif(x[i], x[i+1], x[i+2], x[i+3], root, rotated, squared);
            x[i] = q.x0;
            x[i+1] = q.x1;
            x[i+2] = q.x2;
            x[i+3] = q.x3;
        }
    }
}

// Column j contains one digit from each arm of the first DIF pass.
// Later outer passes consume the usual in-place spectrum layout.
template<int Layers>
__global__ void pack_weighted_outer(const uint32_t* state, U* dest, Shape s, Indices in,
                                    const WeightedDigit* digits, const U* roots, const U* radix) {
    constexpr unsigned count = 1u << Layers;
    const unsigned width = s.length / count;
    const unsigned j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= width) return;
    const unsigned curve = blockIdx.y % s.curves;
    const auto* row = state + limb(s, in.x[blockIdx.y / s.curves], curve, 0);
    U x[count];
#pragma unroll
    for (unsigned i = 0; i < count; ++i)
        x[i] = stage1_ntt_pack_digit<true>(row, s, digits, j + i * width);
    packed_outer_dif<Layers>(x, roots[j], radix);
    auto* out = dest + size_t(blockIdx.y) * s.length + j;
#pragma unroll
    for (unsigned i = 0; i < count; ++i) out[i * width] = x[i];
}

template<int Layers>
__global__ __launch_bounds__(threads)
void pack_radix5_outer(const uint32_t* state, U* dest, Shape s, Indices in,
                       const WeightedDigit* digits, const U* roots, const U* radix,
                       Radix5Constants constants, unsigned rotation_step,
                       const unsigned* headers, unsigned deferred_mask) {
    constexpr unsigned arms = 1u << Layers;
    constexpr unsigned columns = threads / arms;
    static_assert(columns >= 32); // A warp stays in one contiguous column run.
    const unsigned width = s.length / 5;
    const unsigned stride = width / arms;
    const unsigned block_column = blockIdx.x * columns;
    const unsigned lane_column = threadIdx.x % columns;
    const unsigned col = block_column + lane_column + (threadIdx.x / columns) * stride;
    const unsigned op = blockIdx.y / s.curves;
    const size_t residue = size_t(in.x[op]) * s.curves + blockIdx.y % s.curves;
    const auto* source = state + residue * s.words;
    __shared__ unsigned local_limits[inline_carry_segments];
    const bool pending = (deferred_mask >> op) & 1;
    const unsigned* limits = nullptr;
    if (pending) {
        const unsigned segments = stage1_segment_count(s.words);
        limits = headers + residue * segments;
        if (segments <= inline_carry_segments) {
            stage1_segment_corrections(local_limits, limits, segments);
            limits = local_limits;
        }
    }
    const auto read_word = [&](unsigned word) {
        return pending ? stage1_segment_word(source, limits, word, limb_mask(s,word)) : source[word];
    };
    // Pack with coalesced reads before the per-column Good–Thomas rotation.
    // Each thread owns its five shared entries until all radix-5 outputs are ready.
    __shared__ U tile[5][threads];
#pragma unroll
    for (unsigned i = 0; i < 5; ++i)
        tile[i][threadIdx.x] = stage1_ntt_pack_digit_from<true,false>(read_word,s,digits,col + i * width);
    const unsigned rotation = (col * rotation_step) % 5;
    U row[5];
#pragma unroll
    for (unsigned i = 0; i < 5; ++i) row[i] = tile[radix5_plane(i, rotation)][threadIdx.x];
    radix5_dft(row, constants);
#pragma unroll
    for (unsigned i = 0; i < 5; ++i) tile[i][threadIdx.x] = row[i];
    __syncthreads();
    // The next pass groups the columns across DIF arms within each plane.
    auto* out = dest + size_t(blockIdx.y) * s.length + block_column + lane_column;
    for (unsigned group = threadIdx.x / columns; group < 5; group += arms) {
        U x[arms];
#pragma unroll
        for (unsigned i = 0; i < arms; ++i) x[i] = tile[group][i * columns + lane_column];
        packed_outer_dif<Layers>(x, roots[block_column + lane_column], radix);
#pragma unroll
        for (unsigned i = 0; i < arms; ++i) out[group * width + i * stride] = x[i];
    }
}

// A variable spectrum and its prepared dictionary operand have matching
// tile layouts. Each warp owns both sides of the DIF/product/DIT boundary.
__global__ __launch_bounds__(FUSE_TILE_THREADS, 2)
void dictionary_product_tile(const U* input, const U* dictionary, U* output,
                             unsigned length, const U* forward, const U* inverse) {
    constexpr unsigned width = 4096;
    const size_t start = size_t(blockIdx.y) * length + blockIdx.x * width;
    extern __shared__ U sm[];
    for (unsigned i = threadIdx.x; i < width; i += blockDim.x)
        sm[i] = input[start + i];
    __syncthreads();
    for (int stage = 11; stage > 6; stage -= 2) {
        tile_radix4_block(sm, width, stage, forward);
        __syncthreads();
    }
    const unsigned lane = threadIdx.x & 31;
    for (unsigned i = (threadIdx.x >> 5) * 64 + lane; i < width; i += 2 * blockDim.x) {
        U lo = sm[i], hi = sm[i + 32];
        tile_warp6_pair<false>(lo, hi, forward);
        const auto rhs = *reinterpret_cast<const ulonglong2*>(dictionary + start + i + lane);
        lo = gl_mul(lo, rhs.x);
        hi = gl_mul(hi, rhs.y);
        tile_warp6_pair<true>(lo, hi, inverse);
        sm[i] = lo;
        sm[i + 32] = hi;
    }
    __syncthreads();
    ntt_tile_finish_inverse_4096(output + start, sm, inverse);
}

struct ProductInputs {
    const U* left[4];
    const U* right[4];
    const U* aa;
    const U* bb;
    const U* modulus;
    U* cached_e;
};

__global__ void cubic_products(U* out, ProductInputs products, unsigned width, unsigned shift) {
    const unsigned j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= width) return;
    const size_t offset = size_t(blockIdx.y) * (3 * width) + j;
    const auto* left = products.left[blockIdx.z] + offset;
    const auto* right = products.right[blockIdx.z] + offset;
    const auto a = cubic_load(left,width);
    const auto value = left == right ? cubic_square(a,shift) : cubic_product(a,cubic_load(right,width),shift);
    cubic_store(out + size_t(blockIdx.z) * gridDim.y * (3 * width) + offset,width,value);
}

template<bool WarpTail>
__global__ __launch_bounds__(threads, 3)
void cubic_inverse_products(U* out,
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
                            const __grid_constant__ ProductInputs products,
#else
                            const ProductInputs products,
#endif
                            unsigned width,int tile,const U* twiddles,unsigned shift) {
    extern __shared__ U values[];
    const unsigned count=1u<<tile, op=blockIdx.z;
    const unsigned length=3*width;
    const size_t offset=size_t(blockIdx.y)*length+blockIdx.x*count;
    const auto* left=products.left[op]+offset;
    const auto* right=products.right[op]+offset;
    const auto product=[=](unsigned j) {
        const auto a=cubic_load(left+j,width);
        return left==right?cubic_square(a,shift):cubic_product(a,cubic_load(right+j,width),shift);
    };
    auto* dest=out+size_t(op)*gridDim.y*length+offset;
    if constexpr (WarpTail) {
        if(tile>=6) {
            const unsigned lane=threadIdx.x&31;
            for(unsigned i=(threadIdx.x>>5)*64+lane;i<count;i+=2*blockDim.x) {
                auto lo=product(i+lane), hi=product(i+lane+1);
                tile_warp6_pair<true>(lo.x,hi.x,twiddles);
                tile_warp6_pair<true>(lo.y,hi.y,twiddles);
                tile_warp6_pair<true>(lo.z,hi.z,twiddles);
                values[i]=lo.x; values[i+32]=hi.x;
                values[count+i]=lo.y; values[count+i+32]=hi.y;
                values[2*count+i]=lo.z; values[2*count+i+32]=hi.z;
            }
            __syncthreads();
            for(unsigned plane=0;plane<3;++plane)
                ntt_tile_finish_inverse(dest+plane*width,values+plane*count,count,6,tile,twiddles,
                                        tile >= 8 && tile % 2 == 0);
            return;
        }
    }
    for(unsigned j=threadIdx.x;j<count;j+=blockDim.x) {
        const auto c=product(j);
        values[j]=c.x;values[count+j]=c.y;values[2*count+j]=c.z;
    }
    __syncthreads();
    for(unsigned plane=0;plane<3;++plane)
        ntt_tile_pass<true,WarpTail>(dest+plane*width,values+plane*count,count,tile,twiddles);
}

template<bool WarpTail>
__global__ void cubic_inverse_spectra(U* data,unsigned width,int tile,const U* twiddles) {
    extern __shared__ U values[];
    const unsigned count=1u<<tile;
    auto* dest=data+size_t(blockIdx.y)*width+blockIdx.x*count;
    if constexpr (WarpTail) {
        if (tile >= 6) {
            const unsigned lane = threadIdx.x & 31;
            for (unsigned i = (threadIdx.x >> 5) * 64 + lane; i < count; i += 2 * blockDim.x) {
                U lo = dest[i + lane], hi = dest[i + lane + 1];
                tile_warp6_pair<true>(lo, hi, twiddles);
                values[i] = lo;
                values[i + 32] = hi;
            }
            __syncthreads();
            ntt_tile_finish_inverse(dest,values,count,6,tile,twiddles,tile >= 8 && tile % 2 == 0);
            return;
        }
    }
    for(unsigned j=threadIdx.x;j<count;j+=blockDim.x)values[j]=dest[j];
    __syncthreads();
    ntt_tile_pass<true,WarpTail>(dest,values,count,tile,twiddles);
}

enum class InverseInput { Products, CombinedSpectrum, Edwards2XY };

template<bool WarpTail, bool MultiplyAdd = false, bool Loose = false,
         bool WarpInput = false, bool StoreLastRadix4 = false, InverseInput Input = InverseInput::Products,
         bool Fixed12 = false>
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
    const unsigned width = Fixed12 ? 4096 : 1u << tile;
    const unsigned tile_start = blockIdx.x * width;
    const size_t start = curve + tile_start;
    const U* left = products.left[op] + start;
    const U* right = products.right[op] + start;
    const bool last = op + 1 == gridDim.z;
    extern __shared__ U sm[];
    auto* output = out + size_t(op) * gridDim.y * length + start;
    if constexpr (WarpInput) {
        const unsigned lane = threadIdx.x & 31;
        auto* destination = !Fixed12 && tile == 6 ? output : sm;
        for (unsigned i = (threadIdx.x >> 5) * 64 + lane; i < width; i += 2 * blockDim.x) {
            const unsigned pair = i + lane;
            ulonglong2 values;
            if constexpr (Input == InverseInput::Edwards2XY) {
                const auto offset = start + pair;
                const auto bias_index = tile_start + pair;
                switch (op) {
                    case 0: values = stage1_ntt_edwards_double_pair<0>(products, offset, bias_index); break;
                    case 1: values = stage1_ntt_edwards_double_pair<1>(products, offset, bias_index); break;
                    case 2: values = stage1_ntt_edwards_double_pair<2>(products, offset, bias_index); break;
                    default: values = stage1_ntt_edwards_double_pair<3>(products, offset, bias_index); break;
                }
            } else if constexpr (Input == InverseInput::CombinedSpectrum) {
                values = *reinterpret_cast<const ulonglong2*>(left + pair);
            } else {
                values = stage1_ntt_product_pair<MultiplyAdd, Loose>(products, left + pair, right + pair,
                    start + pair, tile_start + pair, last, scale);
            }
            tile_warp6_pair<true>(values.x, values.y, twiddles);
            destination[i] = values.x;
            destination[i + 32] = values.y;
        }
        if constexpr (Fixed12) {
            static_assert(StoreLastRadix4);
            __syncthreads();
            ntt_tile_finish_inverse_4096(output, sm, twiddles);
        } else if (tile > 6) {
            __syncthreads();
            ntt_tile_finish_inverse(output, sm, width, 6, tile, twiddles, StoreLastRadix4);
        }
    } else {
        const bool square = left == right;
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
            sm[i] = scale == 1 ? product : gl_mul(product, scale);
        }
        __syncthreads();
        ntt_tile_pass<true, WarpTail>(output, sm, width, tile, twiddles);
    }
}

// A limb's incoming carry is mapped to: 0=kill, 1=propagate, 2=generate.
// Concatenation is associative and preserves low-to-high order.
struct Join {
    __device__ unsigned operator()(unsigned lo, unsigned hi) const { return hi == 1 ? lo : hi; }
};
// One block per output residue. A thread first summarizes a contiguous limb
// interval; warp ballots compute its prefix. Feed the high carry back into the
// low limb (2^p=1). An all-propagate sum is exactly M, represented as zero.
template<bool Cached, class Raw>
__device__ __forceinline__ void combine_limbs(uint32_t* output, Shape s, Raw raw, uint32_t* saved_raw) {
    __shared__ unsigned warp_tags[threads/32];
    const unsigned stride = (s.words+threads-1)/threads;
    const unsigned begin = min(threadIdx.x * stride, s.words), end = min(begin+stride, s.words);
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
            output[i] = whole == 1 ? 0 : v & mask;
            carry = ((limb_generate | (limb_propagate & (0u - carry))) >> j) & 1;
        } else {
            const U v = raw(i,mask) + carry;
            output[i] = whole == 1 ? 0 : uint32_t(v) & mask;
            carry = v > mask;
        }
    }
}

template<bool Fold, bool Select, bool Weighted = false, bool Routed = false, bool Cached = false>
__global__ void combine(uint32_t* state, const U* products, Shape s, Sums ops, Cursor cursor) {
    extern __shared__ uint32_t saved_raw[];
    const unsigned op = blockIdx.x/s.curves, curve = blockIdx.x%s.curves;
    const unsigned out = output_slot<Routed>(ops.out.x[op],cursor);
    unsigned a = ops.left.x[op], b = ops.right.x[op];
    if constexpr(Select) {
        const unsigned swap = 2u * scalar_bit(cursor);
        a ^= swap; b ^= swap;
    }
    const auto raw = [=](unsigned i, uint32_t mask) -> U {
        if constexpr(Weighted) {
            const auto* sums = products + size_t(blockIdx.x) * s.words;
            const unsigned tail_bits = s.p - 32 * (s.words - 1);
            return weighted_limb(sums, s.words, tail_bits, i, mask);
        } else if constexpr(Fold) {
            const auto* digits = products + size_t(blockIdx.x)*s.length;
            return linear_fold_limb(digits,s,i,mask);
        } else {
            return add_limb(state,s,a,b,curve,i,mask,(ops.subtract >> op) & 1);
        }
    };
    combine_limbs<Cached>(state + limb(s,out,curve,0),s,raw,saved_raw);
}

// Keep the deweighted spectrum and the limb sums in one CTA. Each block
// owns a complete output residue, so its carry can wrap without a global pass.
constexpr unsigned block_weighted_length = 512;
template<unsigned Length = block_weighted_length, bool CubicLayout = false>
__global__ __launch_bounds__(threads, 3)
void weighted_block_reduce(uint32_t* state, WeightedLimbSource source, Shape s, Indices out) {
    extern __shared__ U scratch[];
    auto* coefficients = scratch;
    auto* sums = scratch + Length;
    const unsigned curve = blockIdx.x, op = blockIdx.y;
    const auto* input = source.coefficients + (size_t(op) * s.curves + curve) * Length;
    for (unsigned j = threadIdx.x; j < Length; j += threads) {
        constexpr unsigned width = Length / 3;
        const unsigned index=CubicLayout?(j%3)*width+j%width:j;
        coefficients[j] = gl_mul(input[index], source.inverse_weights[j]);
    }
    __syncthreads();
    for (unsigned i = threadIdx.x; i < s.words; i += threads)
        sums[i] = gather_carry_limb<Length>(source,s.p,i,
            [coefficients](unsigned j) { return coefficients[j]; });
    __syncthreads();
    const unsigned tail_bits = s.p - 32 * (s.words - 1);
    const auto raw = [=](unsigned i, uint32_t mask) { return weighted_limb(sums,s.words,tail_bits,i,mask); };
    // The coefficient phase is complete. The carry scan can reuse its shared storage.
    combine_limbs<true>(state + limb(s,out.x[op],curve,0),s,raw,reinterpret_cast<uint32_t*>(coefficients));
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
__global__ void advance(unsigned* start, unsigned window, unsigned wrap) {
    const unsigned next = *start + window;
    *start = next == wrap ? 0 : next;
}
} // namespace

size_t memory_budget(size_t free_bytes, unsigned limit_mib) {
    const size_t available = free_bytes - free_bytes / 5;
    return limit_mib ? std::min(available, size_t(limit_mib) * 1024 * 1024) : available;
}

Packing resolve_packing(unsigned p, const ecm_stage1::Stage1NttOptions& options, bool prefer_radix5,
                        WeightedBound weighted_bound) {
    if (p < ecm_stage1::stage1_ntt_min_exponent || p > ecm_stage1::stage1_ntt_max_exponent)
        throw std::invalid_argument("invalid Stage1 NTT exponent");
    Packing s;
    if (options.weighted) {
        const bool allow_unsigned_bound = weighted_bound == WeightedBound::UnsignedProducts &&
                                          options.tight_weighted_bound;
        const auto usable = [p, &options, allow_unsigned_bound](unsigned length) {
            const unsigned bpw = (p + length - 1) / length;
            return length <= p && p <= ecm_stage1::ntt_max_exp(length,true,allow_unsigned_bound) && bpw <= 31 &&
                (!options.bpw || bpw <= options.bpw) &&
                options.tile <= ecm_stage1::stage1_ntt_length_log(ecm_stage1::stage1_ntt_radix2_length(length)) &&
                (exact_ok_terms(2ull * length, bpw) ||
                 (allow_unsigned_bound && weighted_unsigned_bound(p, length).has_value()));
        };
        s.length = options.length;
        for (unsigned binary = 2; !s.length && binary <= ecm_stage1::stage1_ntt_max_length; binary *= 2) {
            // The supported mixed length between binary/2 and binary, if any,
            // must pass its own bound before trying the next power of two.
            const unsigned mixed = (binary / 8) * 5;
            if (prefer_radix5 && ecm_stage1::stage1_ntt_supported_radix5(mixed) && usable(mixed))
                s.length = mixed;
            else if (usable(binary))
                s.length = binary;
        }
        if (s.length < 2 || (!ecm_stage1::stage1_ntt_power_of_two(s.length) &&
                            !ecm_stage1::stage1_ntt_radix3(s.length) && !ecm_stage1::stage1_ntt_supported_radix5(s.length)) || !usable(s.length))
            throw std::invalid_argument("unsafe weighted NTT length or digit width");
        s.digits = s.length;
        s.bpw = (p + s.length - 1) / s.length;
        s.needs_unsigned_bound = !exact_ok_terms(2ull * s.length, s.bpw);
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
        if ((s.length & (s.length - 1)) || s.length < 2 * s.digits + 1 || s.length > ecm_stage1::stage1_ntt_max_length)
            throw std::invalid_argument("NTT length must be a supported power of two covering 2*digits+1");
    }
    return s;
}

TransformWorkspace::~TransformWorkspace() { reset(); }
void TransformWorkspace::reset() {
    if (data) cudaFree(data);
    data = nullptr;
    size = 0;
}
unsigned long long* TransformWorkspace::acquire(size_t words) {
    if (size != words) {
        reset();
        CK(cudaMalloc(&data,words*sizeof(*data)));
        size = words;
    }
    return data;
}

// Discard speculative adds for zero bits. Lanes whose leading 1 has not
// yet been consumed start the next iteration at Q.
__global__ void select_auxiliary_bit(uint32_t* state, Shape s, const uint64_t* sigmas,
                                     unsigned bit, bool added) {
    const unsigned word = blockIdx.x*blockDim.x+threadIdx.x;
    const unsigned lane = blockIdx.y, coord = blockIdx.z;
    if (word >= s.words) return;
    const auto prefix = sigmas[lane] >> bit;
    const size_t index = (size_t(coord)*s.curves+lane)*s.words+word;
    if (prefix <= 1) state[index] = word ? 0 : coord==0 ? 12 : coord==1 ? 40 : 1;
    else if (added && !(prefix & 1))
        state[index] = state[index + size_t(XA)*s.curves*s.words];
}

struct Pm1Request {
    const std::vector<uint32_t>& exponent;
    Pm1Progress& progress;
    const std::function<bool(const Pm1Progress&)>& verified;
    const Pm1Probe* probe = nullptr;
    const std::vector<uint32_t>* base = nullptr;
    const std::function<void(const Plan&)>* ready = nullptr;
};

static Run run_ladder(int device, unsigned p, unsigned curves, const std::vector<uint32_t>& initial,
                      const std::vector<uint32_t>& scalar_words, unsigned count,
                      const ecm_stage1::Stage1NttOptions& options, const Control* control,
                      unsigned samples, const EdwardsNaf* naf = nullptr,
                      const std::vector<uint64_t>* auxiliary_sigmas = nullptr,
                      const Pm1Request* pm1 = nullptr) {
    const auto prep = now_s();
    if (p < ecm_stage1::stage1_ntt_min_exponent || p > ecm_stage1::stage1_ntt_max_exponent ||
        !curves || curves > ecm_stage1::stage1_ntt_max_curves || options.window < -1 || options.window > 64 ||
        !samples || samples > ecm_stage1::ntt_max_repeats || count > UINT32_MAX - 64 ||
        (!naf && !(control && control->describe_only) && scalar_words.size() < (size_t(count) + 31) / 32) ||
        (control && control->resume > count))
        throw std::invalid_argument("invalid Mersenne NTT Stage1 shape or scalar");
    Shape s{p, (p+31)/32, curves, 0, 0, 0};
    const size_t stride = size_t(s.words)*curves;
    const unsigned initial_slots = pm1 ? 1 : naf ? 3 : 6, checkpoint_slots = naf ? 3 : 4;
    const unsigned state_slots = pm1 ? unsigned(pm1_detail::SlotCount) : slots;
    const bool describe_only = control && control->describe_only;
    if (!describe_only && initial.size() != initial_slots*stride) throw std::invalid_argument("invalid Stage1 initial points");
    auto check_residues = [&](const std::vector<uint32_t>& values) {
        const unsigned top = p%32 ? (1u << (p%32))-1 : UINT32_MAX;
        for (size_t offset = 0; offset < values.size(); offset += s.words) {
            const auto* v = values.data() + offset;
            if (v[s.words-1] > top || (v[s.words-1] == top && std::all_of(v,v+s.words-1,
                    [](auto x) { return x == UINT32_MAX; })))
                throw std::invalid_argument("noncanonical Stage1 NTT residue");
        }
    };
    check_residues(initial);
    unsigned entries = 0;
    if (naf) {
        if (naf->window < 2 || naf->window > 12 || naf->digits.size() != count ||
            control->warmup_iterations || control->modular_square_probe)
            throw std::invalid_argument("invalid Edwards NTT NAF request");
        entries = 1u << (naf->window - 2);
        if (naf->affine_output && (!naf->build_dictionary || naf->host_fallback ||
            naf->affine_output->size() != 4 * entries * stride))
            throw std::invalid_argument("invalid affine dictionary output");
        if (!describe_only && naf->dictionary.size() != (naf->build_dictionary ? 3 : 4 * entries) * stride)
            throw std::invalid_argument("invalid Edwards NTT NAF dictionary size");
        for (int digit : naf->digits)
            if (digit && (digit % 2 == 0 || unsigned(std::abs(digit)) >= 2 * entries))
                throw std::invalid_argument("invalid Edwards NTT NAF digit");
        check_residues(naf->dictionary);
    }
    const bool square_probe = control && control->modular_square_probe;
    const auto bound = (naf || square_probe || auxiliary_sigmas || pm1) ? WeightedBound::UnsignedProducts : WeightedBound::Conservative;
    const auto packing = resolve_packing(p, options, true, bound);
    s.length=packing.length; s.bpw=packing.bpw; s.digits=packing.digits;
    const bool mixed = ecm_stage1::stage1_ntt_radix5(s.length);
    const bool cubic = ecm_stage1::stage1_ntt_radix3(s.length);
    const bool cubic_wide = cubic && s.length > ecm_stage1::stage1_ntt_cubic_block_limit;
    const unsigned planes = mixed ? 5 : cubic ? 3 : 1;
    const unsigned width = s.length / planes;
    const unsigned cubic_shift = cubic ? ecm_stage1::stage1_ntt_cubic_twist(p, width) : 0;
    const auto cubic_width_dispatch = [width](auto launch) {
        switch (width) {
        case 256: launch(std::integral_constant<int, 8>{}); break;
        case 512: launch(std::integral_constant<int, 9>{}); break;
        case 1024: launch(std::integral_constant<int, 10>{}); break;
        }
    };
    const int log = int(ecm_stage1::stage1_ntt_length_log(width));
    if (options.tile > unsigned(log) || options.tile > 12 ||
        options.outer_max > 4 || options.profile > 2)
        throw std::invalid_argument("NTT tile, outer width or profile is out of range for this length");
    CK(cudaSetDevice(device));
    CK(cudaFree(nullptr)); // Establish the CUDA context before measuring the worker's delta.
    TransformWorkspace local_workspace;
    auto& workspace = control && control->workspace ? *control->workspace : local_workspace;
    const size_t workspace_words = size_t(9)*curves*s.length;
    if (workspace.size != workspace_words) workspace.reset();
    size_t free_before = 0, total = 0;
    CK(cudaMemGetInfo(&free_before, &total));
    free_before += workspace.size*sizeof(U);
    Run result;
    unsigned defaults = FusedMultiplyAdd | LooseDifference | DirectRoute | PackedForward |
                        PfaRadix5 | CoalescedRadix5 | InverseWarpInput | ShortInverseTile | WarpWeightedCarry |
                        PackedWeightedOuter | FixedCarryOffsets | BlockWeightedCarry;
    if (curves >= 8) defaults |= DictionaryProductTile;
    // Wider weighted reductions showed a batch-dependent cache tradeoff.
    if (p <= 65536) defaults |= CachedCarry;
    // The mixed pack can consume the segment carries directly. The 5120
    // shape amortizes segmentation even with multiple live curves.
    const bool early_segments = (curves == 1 && p >= 53249 && (s.length == 2560 || s.length == 5120)) ||
                                (s.length == 5120 && p >= 106497);
    const auto segment_count = stage1_segment_count(s.words);
    // Wide Edwards batches amortize a separate pass to resolve segment carries.
    const bool wide_segments = segment_count > 14 || (naf && options.weighted && curves >= 32 &&
        ((s.length == 8192 && segment_count >= 13) ||
         (s.length == 4096 && s.words >= 10 * stage1_carry_segment)));
    if (options.weighted ? wide_segments || early_segments : segment_count > 32) {
        defaults |= SegmentedCarry;
        if (mixed || curves == 1) defaults |= DeferredPack;
    }
    if (curves == 1 && ((s.length == 2560 && p >= 53249) || (s.length == 5120 && p >= 106497)))
        defaults |= FusedWeightedCarry;
    // On 5120-point plans the single-curve reducer already fuses this work.
    const bool cache_edwards_carry = s.length == 5120 ? p >= 106497 && curves >= 8 : wide_segments;
    if (naf && cache_edwards_carry)
        defaults |= CachedFusedWeightedCarry;
    if (naf) defaults |= Edwards2XYFrequency | Edwards2XYInverseInput | TightCarrySpans |
                         FixedWideInverse | FixedWideForward;
    unsigned requested = control ? control->experimental_dataflow.value_or(defaults) : 0;
    using Layout = ecm_stage1::Stage1NttOptions::MixedLayout;
    if (mixed && options.mixed_layout != Layout::Automatic) {
        requested &= ~(PfaRadix5 | CoalescedRadix5);
        if (options.mixed_layout != Layout::Twiddle) requested |= PfaRadix5;
        if (options.mixed_layout == Layout::Coalesced) requested |= CoalescedRadix5;
    }
    if (naf || pm1) requested &= ~(FusedMultiplyAdd | LooseDifference | DirectRoute);
    if (cubic) requested &= ~(FusedMultiplyAdd | LooseDifference | DirectRoute | DeferredPack);
    if (cubic && !cubic_wide) requested &= ~SegmentedCarry;
    const bool pfa = mixed && (requested & PfaRadix5);
    const bool coalesced = pfa && (requested & CoalescedRadix5);
    // n = (1 + rotation_step * width) * i + width * j (mod length).
    // The first multiplier is divisible by 5 and congruent to 1 modulo width.
    // omega^5 still gives a primitive width root, with permuted frequencies.
    constexpr unsigned crt_rotation[] = {0, 4, 2, 3, 1};
    const unsigned rotation_step = pfa ? crt_rotation[width % 5] : 0;
    if (pfa) result.dataflow |= PfaRadix5;
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
    const unsigned window = pm1
        ? (options.window >= 0 ? std::min(unsigned(options.window),pm1_max_graph_bits)
                               : s.length >= pm1_grouped_graph_min_length ? pm1_max_graph_bits : 1u)
        : (options.window >= 0 ? unsigned(options.window) : profile.window);
    const FuseOverrides overrides{options.tile ? int(options.tile) : profile.tile,
        options.outer_max ? int(options.outer_max) : profile.outer,
        options.warp_tail >= 0 ? options.warp_tail != 0 : profile.warp_tail};
    const U omega=gl_pow_host(7,(GL_P-1)/s.length), inv=gl_pow_host(omega,GL_P-2);
    const U omega2 = gl_pow_host(omega, planes), inv2 = gl_pow_host(inv, planes);
    const U inverse_scale = gl_pow_host(cubic ? width : s.length, GL_P - 2);
    // Weighted carry already multiplies each coefficient by its inverse weight.
    const U product_scale = options.weighted ? 1 : inverse_scale;
    Tables tables;
    auto& ctx=tables.ctx;
    const auto* resolved = control ? &overrides : nullptr;
    fuse_describe(ctx,width,log,omega2,inv2,true,resolved);
    // A table export needs only the five spectra for 2P and one for d.
    unsigned fixed_planes = pm1 ? 0 : naf && naf->affine_output ? 6 : naf ? 4 * entries : 2;
    const size_t dictionary_bytes = naf ? size_t(fixed_planes) * sizeof(U) * curves * s.length : 0;
    bool gpu_dictionary = naf && naf->build_dictionary;
    const size_t dictionary_workspace = gpu_dictionary ? 4ull * entries * stride * sizeof(uint32_t) : 0;
    result.owned_bytes = (naf || pm1 ? 9 : 11) * sizeof(U) * curves * s.length + dictionary_bytes +
        state_slots * stride * sizeof(uint32_t) +
        (std::max(size_t(1), scalar_words.size()) + 1) * sizeof(uint32_t) +
        (fuse_planned_base_words(ctx) + fuse_planned_table_words(ctx)) * sizeof(U);
    if (auxiliary_sigmas) result.owned_bytes += curves*sizeof(uint64_t);
    if (mixed && !pfa) result.owned_bytes += size_t(8) * width * sizeof(U);
    if (options.weighted)
        result.owned_bytes += size_t(s.length) * (sizeof(U) + sizeof(WeightedDigit)) +
                              size_t(s.words) * sizeof(WeightedLimb);
    if (gpu_dictionary && naf->host_fallback &&
        ((naf->dictionary_budget && dictionary_bytes + dictionary_workspace > naf->dictionary_budget) ||
         result.owned_bytes + dictionary_workspace > result.budget_bytes)) {
        gpu_dictionary = false;
    }
    result.dictionary_built_on_gpu = gpu_dictionary;
    result.dictionary_spectra_bytes = dictionary_bytes;
    result.dictionary_workspace_bytes = gpu_dictionary ? dictionary_workspace : 0;
    if (naf && naf->dictionary_budget && dictionary_bytes + result.dictionary_workspace_bytes > naf->dictionary_budget)
        throw BudgetError("Edwards NTT dictionary exceeds its VRAM budget");
    result.owned_bytes += result.dictionary_workspace_bytes;
    if (result.owned_bytes > result.budget_bytes)
        throw BudgetError("Stage1 NTT buffers exceed the total VRAM budget");
    const size_t header_words = size_t(state_slots) * curves * segment_count;
    const bool segmented = (cubic_wide || (requested & SegmentedCarry)) &&
        header_words * sizeof(unsigned) <= result.budget_bytes - result.owned_bytes;
    if (cubic_wide && !segmented) throw BudgetError("radix-3 carry headers exceed the VRAM budget");
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
    if (!ctx.t) throw std::invalid_argument("Stage1 NTT requires an inverse tile");
    const bool warp_tile = ctx.warp_tail && ctx.t >= 5;
    const bool inverse_warp_input = warp_tile && ctx.t >= 6 && (requested & InverseWarpInput);
    const bool store_last_inverse = ctx.t == 12 || (ctx.t == 10 && curves >= 24);
    const bool fixed_wide_inverse = naf && options.weighted && inverse_warp_input &&
        ctx.t == 12 && s.length >= 8192 && curves >= 24 && (requested & FixedWideInverse);
    if (fixed_wide_inverse) result.dataflow |= FixedWideInverse;
    const auto unsigned_bound = options.weighted && bound == WeightedBound::UnsignedProducts &&
                                options.tight_weighted_bound ? weighted_unsigned_bound(p, s.length) : std::nullopt;
    const bool frequency_double = naf && inverse_warp_input && (requested & Edwards2XYFrequency) &&
        unsigned_bound && *unsigned_bound <= (GL_P - 1) / 4 &&
        modulus_bytes <= result.budget_bytes - result.owned_bytes;
    const bool direct_frequency = !cubic && frequency_double && (requested & Edwards2XYInverseInput) &&
        ((control && control->experimental_dataflow.has_value()) ||
         (!mixed && ctx.nms == 0 && s.length >= 128 && s.length <= 512));
    if (frequency_double) {
        result.owned_bytes += modulus_bytes;
        result.dataflow |= Edwards2XYFrequency;
    }
    if (direct_frequency) result.dataflow |= Edwards2XYInverseInput;
    // The shared tile still wins for isolated 16-point and linear 32-point work.
    const bool short_shape = curves > 1 || s.length <= 8 ||
                             (s.length == 32 && options.weighted) ||
                             (control && control->experimental_dataflow.has_value());
    const bool short_inverse = ctx.warp_tail && (requested & ShortInverseTile) &&
                               !mixed && !cubic && ctx.nms == 0 && ctx.t <= 5 && short_shape;
    const unsigned inverse_rows = curves >= 64 ? 2 : 1;
    // Each lane loads two coefficients. Small tiles need only those warps.
    const unsigned inverse_threads = cubic ? (ctx.t >= 10 ? 256 : 128) : short_inverse ? inverse_rows * (s.length / 2) : inverse_warp_input
        ? std::min(unsigned(FUSE_TILE_THREADS), 1u << (ctx.t - 1)) : FUSE_TILE_THREADS;
    if (inverse_warp_input) result.dataflow |= InverseWarpInput;
    if (short_inverse) result.dataflow |= ShortInverseTile;
    const size_t tile_smem = sizeof(U) << ctx.t;
    const bool cubic_fused_product = cubic && ctx.t <= 10;
    const size_t inverse_smem = cubic ? (cubic_fused_product ? 3*tile_smem : tile_smem)
        : short_inverse || (inverse_warp_input && ctx.t == 6) ? 0 : tile_smem;
    const bool cubic_packed = cubic && !cubic_wide && ctx.nms == 0;
    const bool packed = (requested & PackedForward) && !mixed && !cubic && ctx.nms == 0;
    const bool default_forward = control && !control->experimental_dataflow.has_value();
    const bool packed_batch = packed && curves >= 24;
    const bool forward_input = (requested & ForwardTileInput) || (default_forward &&
        ((!packed && ctx.t >= 9) || (packed_batch && (ctx.t == 1 || (warp_tile && ctx.t <= 7)))));
    const bool fixed_wide_forward = naf && options.weighted && !packed && forward_input && warp_tile &&
        ctx.t == 10 && s.length >= 5120 && curves >= 24 && (requested & FixedWideForward);
    if (fixed_wide_forward) result.dataflow |= FixedWideForward;
    const bool compact_pack = packed_batch && ((ctx.t >= 2 && ctx.t <= 4) || (ctx.t >= 8 && ctx.t <= 10));
    const bool compact_forward = ((forward_input || packed) && (requested & ForwardTileCompact)) ||
        (default_forward && compact_pack);
    const unsigned forward_limit = compact_forward ? 256 : FUSE_TILE_THREADS;
    const unsigned forward_threads = cubic_packed ? std::max(128u, width / 4) : (forward_input || compact_forward)
        ? std::max(32u, std::min(forward_limit, 1u << (ctx.t - 1))) : FUSE_TILE_THREADS;
    const size_t forward_smem = forward_input && (ctx.t <= 2 || (warp_tile && ctx.t <= 6)) ? 0 : tile_smem;
    if (forward_input) result.dataflow |= ForwardTileInput;
    if (compact_forward && (!forward_input || ctx.t >= 10))
        result.dataflow |= ForwardTileCompact;
    const bool routed = requested & DirectRoute;
    const bool warp_reduce = (requested & WarpWeightedCarry) && options.weighted && !segmented &&
                             s.words <= 32 && s.length <= 64;
    if (warp_reduce) result.dataflow |= WarpWeightedCarry;
    const bool block_reduce = (cubic && !cubic_wide) || ((requested & BlockWeightedCarry) && naf && options.weighted && !segmented &&
                              s.length == block_weighted_length && s.words <= block_weighted_length && curves >= 32);
    if (block_reduce) result.dataflow |= BlockWeightedCarry;
    const bool cached = (requested & CachedCarry) && options.weighted && !segmented &&
                        !warp_reduce && !block_reduce && s.words <= 32 * threads;
    const bool deferred = segmented && !packed && (requested & DeferredPack);
    const bool wide_outer = s.length == 262144 && segment_count > inline_carry_segments && ctx.nms > 0;
    const bool single_outer = ctx.nms == 1 && !deferred &&
        (s.length == 4096 || s.length == 8192 || s.length == 32768);
    const bool packed_outer = (requested & PackedWeightedOuter) && options.weighted && !mixed &&
        forward_input && (single_outer || wide_outer) && ctx.ms[0] >= 1 && ctx.ms[0] <= 3;
    const bool radix5_forward_outer = naf && coalesced && forward_input && ctx.nms > 0 &&
        ctx.ms[0] >= 1 && ctx.ms[0] <= 3 &&
        (options.fused_radix5_forward || (requested & Radix5ForwardOuter));
    const bool dictionary_tile = naf && packed_outer && inverse_warp_input &&
                                 (s.length == 32768 || s.length == 262144) &&
                                 ctx.t == 12 && (requested & DictionaryProductTile);
    if (dictionary_tile) result.dataflow |= DictionaryProductTile;
    if (packed_outer) result.dataflow |= PackedWeightedOuter;
    if (radix5_forward_outer) result.dataflow |= Radix5ForwardOuter;
    if (routed) result.dataflow |= DirectRoute;
    if (packed || cubic_packed) result.dataflow |= PackedForward;
    if (cached) result.dataflow |= CachedCarry;
    if (deferred) result.dataflow |= DeferredPack;
    unsigned coefficient_bits = 64;
    if (naf && (cubic || s.length >= 2560) && unsigned_bound && (requested & TightCarrySpans)) {
        // Canonical Edwards products fit U, including dictionary products.
        // The biased 2XY combination is the only wider consumer of the table.
        const U upper = (frequency_double ? 4 : 1) * *unsigned_bound;
        coefficient_bits = 1;
        for (U high = upper >> 1; high; high >>= 1) ++coefficient_bits;
        if (coefficient_bits < 64) result.dataflow |= TightCarrySpans;
    }
    const auto weights = options.weighted ? weighted_tables(p, s.length, inverse_scale, coefficient_bits)
                                          : WeightedTables{};
    static_assert(radix5_limb_halo * radix5_limb_min_digit_bits >= 64 + 32);
    const bool radix5_limb_sums = mixed && (options.fused_radix5_inverse || (requested & Radix5InverseLimbSums)) &&
        p / s.length >= radix5_limb_min_digit_bits;
    if (radix5_limb_sums) result.dataflow |= Radix5InverseLimbSums;
    const unsigned carry_cache_smem = !cubic && !radix5_limb_sums && options.weighted && segmented &&
        (requested & CachedFusedWeightedCarry)
        ? weighted_segment_cache_bytes(weights.limbs) : 0;
    // Keep the shared coefficient tile small enough to allow several CTAs per SM.
    const bool cached_fused = carry_cache_smem && carry_cache_smem <= 16 * 1024;
    const bool fixed_carry_offsets = (requested & FixedCarryOffsets) && s.length == 5120 && cached_fused;
    if (fixed_carry_offsets) result.dataflow |= FixedCarryOffsets;
    const bool fused_weighted = !cubic && !radix5_limb_sums && options.weighted && segmented &&
        (cached_fused || (requested & FusedWeightedCarry));
    if (fused_weighted) result.dataflow |= FusedWeightedCarry;
    if (cached_fused) result.dataflow |= CachedFusedWeightedCarry;
    // With >=16 bits between coefficient starts, a limb sees at most 12
    // fragments across the two folds. Even after scaling by 3^15 the 64-bit
    // sum fits, and the split multibit carry still leaves a one-bit ripple.
    const bool pm1_scale = pm1 && !pm1->base && options.weighted && segmented &&
        !fused_weighted && !radix5_limb_sums && !block_reduce && p / s.length >= 16;
    if (pm1_scale) result.dataflow |= Pm1WindowScale;
    if (pm1 && pm1->base) {
        // Budget the table alongside the ordinary NTT buffers for every lane.
        const size_t lane_budget = result.budget_bytes / std::max(1u, options.max_batch);
        const size_t spectrum_bytes = size_t(curves) * s.length * sizeof(U);
        const size_t available = lane_budget > result.owned_bytes ? lane_budget - result.owned_bytes : 0;
        const auto powers = (1u << std::max(1u, window)) - 1;
        fixed_planes = powers * spectrum_bytes <= available ? powers : unsigned(available >= spectrum_bytes);
        result.owned_bytes += fixed_planes * spectrum_bytes;
    }
    Plan pm1_plan{};
    if ((control && control->describe) || (pm1 && pm1->ready)) {
        const auto join = [](std::string& stages, int width) {
            if (!stages.empty()) stages += '+';
            stages += std::to_string(width);
        };
        Plan plan{s.length, s.bpw, s.digits, unsigned(ctx.t), unsigned(ctx.m_max), window,
                  forward_threads, unsigned(forward_smem), inverse_threads, unsigned(inverse_smem),
                  ctx.nms ? FUSE_OUTER_THREADS : 0u,
                  unsigned(ctx.passes_fwd) + unsigned(mixed),
                  warp_tile || short_inverse, profile.source, options.weighted, {}, {}, result.dataflow};
        plan.max_exp = ecm_stage1::ntt_max_exp(s.length,options.weighted,unsigned_bound.has_value());
        plan.field_modulus = GL_P;
        plan.carry_bits = coefficient_bits;
        if (unsigned_bound)
            plan.coefficient_bound = (frequency_double ? 4 : 1) * *unsigned_bound;
        else
            plan.coefficient_bound = (loose ? 2 : 1) * terms * digit_max * digit_max + (fma ? digit_max : 0);
        if (mixed) plan.forward = "r5";
        for (int i = 0; i < ctx.nms; ++i) join(plan.forward, ctx.ms[i]);
        join(plan.forward, ctx.t);
        join(plan.inverse, ctx.t);
        for (int i = ctx.nms; i > 0; --i) join(plan.inverse, ctx.ms[i-1]);
        if (mixed) plan.inverse += plan.inverse.empty() ? "r5" : "+r5";
        if (cubic) {
            plan.forward = "3x(" + plan.forward + ')';
            plan.inverse = "3x(" + plan.inverse + ')';
        }
        if (pm1) plan.pm1_cached_powers = fixed_planes;
        if (control && control->describe) control->describe(plan);
        if (pm1 && pm1->ready) pm1_plan=std::move(plan);
    }
    if (describe_only) return result;
    // Plan setup uses legacy-stream table generation and stream capture. Keep
    // setup serialized for independent P-1 streams, then release before replay.
    static std::mutex pm1_setup;
    std::unique_lock<std::mutex> setup_guard(pm1_setup,std::defer_lock);
    if (pm1) setup_guard.lock();
    Stream stream; const auto q=stream.value;
    fuse_init(ctx,width,log,omega2,inv2,true,resolved);
    const size_t carry_smem = size_t((s.words + threads - 1) / threads) * threads * sizeof(uint32_t);
    const auto select_packed_kernel = [warp_tile, forward_input](auto weight) {
        constexpr bool weighted = decltype(weight)::value;
        if (forward_input) return warp_tile ? stage1_ntt_packed_forward_registers<weighted,true,Shape,Indices,WeightedDigit>
                                            : stage1_ntt_packed_forward_registers<weighted,false,Shape,Indices,WeightedDigit>;
        return warp_tile ? stage1_ntt_packed_forward<weighted,true,Shape,Indices,WeightedDigit>
                         : stage1_ntt_packed_forward<weighted,false,Shape,Indices,WeightedDigit>;
    };
    const auto packed_kernel = options.weighted ? select_packed_kernel(std::true_type{})
                                                : select_packed_kernel(std::false_type{});
    const auto forward_kernel = fixed_wide_forward ? stage1_ntt_forward_tile<true,10>
        : warp_tile ? stage1_ntt_forward_tile<true> : stage1_ntt_forward_tile<false>;
    if (packed) {
        cudaFuncSetAttribute(packed_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        CK(cudaFuncSetAttribute(packed_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(forward_smem)));
    }
    if (forward_input) {
        cudaFuncSetAttribute(forward_kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        CK(cudaFuncSetAttribute(forward_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(forward_smem)));
    }
    const auto configure_inverse = [&](const void* kernel) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, 100);
        CK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(inverse_smem)));
    };
    const auto fixed_inverse_product =
        inverse_product_tile<true,false,false,true,true,InverseInput::Products,true>;
    const auto configure_tile = [&](auto add, auto difference) {
        constexpr bool add_bb = decltype(add)::value, loose_e = decltype(difference)::value;
        configure_inverse(fixed_wide_inverse ? (const void*)fixed_inverse_product
            : inverse_warp_input ? (const void*)inverse_product_tile<true,add_bb,loose_e,true>
            : warp_tile ? (const void*)inverse_product_tile<true,add_bb,loose_e>
                        : (const void*)inverse_product_tile<false,add_bb,loose_e>);
    };
    if (!short_inverse && !cubic) {
        configure_tile(std::false_type{},std::false_type{});
        if (loose && fma) configure_tile(std::true_type{},std::true_type{});
        else if (loose) configure_tile(std::false_type{},std::true_type{});
        else if (fma) configure_tile(std::true_type{},std::false_type{});
    }
    const auto frequency_kernel = [&](auto input) {
        constexpr auto kind = decltype(input)::value;
        if constexpr (kind == InverseInput::CombinedSpectrum) {
            if (fixed_wide_inverse) return inverse_product_tile<true,false,false,true,true,kind,true>;
        }
        return store_last_inverse ? inverse_product_tile<true,false,false,true,true,kind>
                                  : inverse_product_tile<true,false,false,true,false,kind>;
    };
    const auto frequency_inverse = direct_frequency
        ? frequency_kernel(std::integral_constant<InverseInput, InverseInput::Edwards2XY>{})
        : frequency_kernel(std::integral_constant<InverseInput, InverseInput::CombinedSpectrum>{});
    if (frequency_double && !cubic) configure_inverse((const void*)frequency_inverse);
    ntt_fuse_cache_tables(ctx,omega2,inv2);
    CK(cudaDeviceSynchronize()); // table creation uses the legacy stream
    Buffer<uint32_t> state(state_slots*stride);
    Buffer<unsigned> carry_headers(segmented ? header_words : 0);
    Buffer<unsigned> start(1);
    Buffer<uint32_t> scalar(std::max(size_t(1), scalar_words.size()));
    U* freq = workspace.acquire(workspace_words);
    U* inverse = freq + size_t(5)*curves*s.length;
    Buffer<U> fixed(size_t(fixed_planes)*curves*s.length);
    Buffer<U> modulus(loose || frequency_double ? s.length : 0);
    const size_t mixed_table_words = mixed && !pfa ? size_t(4) * width : 0;
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
        if (!pfa) {
            fill(fwd_twiddles, mixed_fwd.data, omega);
            fill(inv_twiddles, mixed_inv.data, inv);
        }
        radix5_fwd = radix5_constants(gl_pow_host(omega, width));
        radix5_inv = radix5_constants(gl_pow_host(inv, width));
    }
    const unsigned limb_smem = fused_weighted || warp_reduce || block_reduce || radix5_limb_sums
        ? 0 : weighted_limb_smem(weights.limbs);
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
    const auto forward_tiles = [&](U* dest, unsigned batch) {
        forward_kernel<<<dim3(width >> ctx.t, batch),forward_threads,forward_smem,q>>>(dest,width,ctx.t,ctx.tblF);
        CK(cudaGetLastError());
    };
    const auto forward_spectra = [&](U* dest, unsigned batch) {
        if (!forward_input) {
            ntt_forward_fused(dest,ctx,omega2,batch,q);
            return;
        }
        ntt_forward_outer_fused(dest,ctx,omega2,batch,q);
        forward_tiles(dest,batch);
    };
    const auto pack_outer_inputs = [&](U* dest, unsigned count, Indices in) {
        const std::array kernels{pack_weighted_outer<1>, pack_weighted_outer<2>, pack_weighted_outer<3>};
        const unsigned columns = s.length >> ctx.ms[0];
        const dim3 blocks((columns + threads - 1) / threads, count * curves);
        const auto kernel = kernels[ctx.ms[0] - 1];
        kernel<<<blocks,threads,0,q>>>(state.data,dest,s,in,digit_table.data,ctx.passF[0],ctx.radF[0]);
        CK(cudaGetLastError());
        if (ctx.nms > 1)
            ntt_forward_outer_fused(dest,ctx,omega2,count*curves,q,ForwardOuterInput::FirstPassComplete);
    };
    const auto forward_inputs = [&](U* dest, unsigned count, Indices in, unsigned pending = 0) {
        if (cubic) {
            const unsigned batch=count*curves*planes;
            if (cubic_wide) {
                pack_cubic<0,false,false><<<dim3((width + threads - 1) / threads,batch),threads,0,q>>>(
                    state.data,dest,s,in,digit_table.data);
                forward_spectra(dest,batch);
            } else cubic_width_dispatch([&](auto plane_layers) {
                constexpr int tile = decltype(plane_layers)::value;
                if (cubic_packed) {
                    const auto kernel=warp_tile?pack_cubic<tile,true,true>:pack_cubic<tile,true,false>;
                    kernel<<<dim3(1,batch),forward_threads,tile_smem,q>>>(state.data,dest,s,in,digit_table.data,ctx.tblF);
                } else {
                    pack_cubic<tile,false,false><<<dim3(1,batch),threads,0,q>>>(state.data,dest,s,in,digit_table.data);
                    forward_spectra(dest,batch);
                }
            });
            CK(cudaGetLastError());
            return;
        }
        if (packed) {
            packed_kernel<<<dim3(1,count*curves),forward_threads,forward_smem,q>>>(
                state.data,dest,s,in,digit_table.data,ctx.t,ctx.tblF);
            CK(cudaGetLastError());
            return;
        }
        if (packed_outer && !pending) {
            pack_outer_inputs(dest,count,in);
            forward_tiles(dest,count*curves);
            return;
        }
        if (radix5_forward_outer) {
            const std::array kernels{pack_radix5_outer<1>, pack_radix5_outer<2>, pack_radix5_outer<3>};
            // The power-of-two width is a multiple of the CTA's 256 input columns.
            const auto kernel = kernels[ctx.ms[0] - 1];
            kernel<<<dim3(width / threads,count * curves),threads,0,q>>>(
                state.data,dest,s,in,digit_table.data,ctx.passF[0],ctx.radF[0],
                radix5_fwd,rotation_step,carry_headers.data,pending);
            CK(cudaGetLastError());
            const unsigned batch = count * curves * planes;
            ntt_forward_outer_fused(dest,ctx,omega2,batch,q,ForwardOuterInput::FirstPassComplete);
            forward_tiles(dest,batch);
            return;
        }
        if (pfa) {
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
        else if (pending && options.weighted)
            pack<true,false,false,true><<<grid(count),threads,0,q>>>(
                state.data,dest,s,in,digit_table.data,nullptr,{},carry_headers.data,pending);
        else if (pending)
            pack<false,false,false,true><<<grid(count),threads,0,q>>>(
                state.data,dest,s,in,nullptr,nullptr,{},carry_headers.data,pending);
        else if (mixed)
            pack<true,true><<<mixed_grid(count),threads,0,q>>>(state.data,dest,s,in,digit_table.data,
                                                               mixed_fwd.data,radix5_fwd);
        else if (options.weighted)
            pack<true><<<grid(count),threads,0,q>>>(state.data,dest,s,in,digit_table.data);
        else
            pack<false><<<grid(count),threads,0,q>>>(state.data,dest,s,in,nullptr);
        forward_spectra(dest,count*curves*planes);
    };
    if (!naf && !pm1) {
        forward_inputs(fixed.data, 2, {{A24,Base}});
    }
    if (loose || frequency_double) {
        Shape one = s;
        one.curves = 1;
        if (cubic)
            pack_cubic<0,false,false,true><<<dim3((width + threads - 1) / threads,3),threads,0,q>>>(
                nullptr,modulus.data,one,{},digit_table.data);
        else if (pfa)
            pack<true,true,true,false,true><<<(width+threads-1)/threads,threads,0,q>>>(
                nullptr,modulus.data,one,{},digit_table.data,nullptr,radix5_fwd,nullptr,0,rotation_step);
        else if (mixed)
            pack<true,true,true><<<(width+threads-1)/threads,threads,0,q>>>(nullptr,modulus.data,one,{},
                digit_table.data,mixed_fwd.data,radix5_fwd);
        else
            pack<true,false,true><<<(width+threads-1)/threads,threads,0,q>>>(nullptr,modulus.data,one,{},digit_table.data);
        CK(cudaGetLastError());
        forward_spectra(modulus.data,planes);
        if (frequency_double) {
            if (cubic) cubic_square_bias<<<width/threads,threads,0,q>>>(modulus.data,width,cubic_shift);
            else stage1_ntt_square_spectrum<<<(s.length+threads-1)/threads,threads,0,q>>>(modulus.data,s.length);
            CK(cudaGetLastError());
        }
    }
    CK(cudaStreamSynchronize(q));
    const auto finish_carries = [&](unsigned count, Indices out, Cursor cursor,
                                    bool route, unsigned pending) {
        if (segment_count > inline_carry_segments) {
            const unsigned scan_threads = ((segment_count + 31) / 32) * 32;
            if (route)
                resolve_segment_headers<true><<<count*curves,scan_threads,0,q>>>(carry_headers.data,s,out,cursor);
            else
                resolve_segment_headers<false><<<count*curves,scan_threads,0,q>>>(carry_headers.data,s,out,cursor);
            CK(cudaGetLastError());
        }
        if (pending == (1u << count) - 1) return;
        if (route)
            finish_segment_carries<true><<<segment_grid(count),stage1_carry_segment,0,q>>>(
                state.data,carry_headers.data,s,out,cursor,pending);
        else
            finish_segment_carries<false><<<segment_grid(count),stage1_carry_segment,0,q>>>(
                state.data,carry_headers.data,s,out,cursor,pending);
    };
    const auto add_inputs = [&](auto select, unsigned count, Sums ops, Cursor cursor, unsigned pending = 0) {
        constexpr bool select_coords = decltype(select)::value;
        if (!segmented) {
            combine<false,select_coords><<<count*curves,threads,0,q>>>(state.data,nullptr,s,ops,cursor);
            return;
        }
        segment_limb_sums<SegmentInput::Add,select_coords,false><<<segment_grid(count),stage1_carry_segment,0,q>>>(
            state.data,nullptr,carry_headers.data,s,ops,cursor,1);
        finish_carries(count,ops.out,{},false,pending);
    };
    enum class ProductKind { Ordinary, EdwardsDouble, Dictionary };
    auto mul=[&](unsigned inputs, unsigned outputs, Indices in, Indices a, Indices b, Indices out,
                 bool build_e = false, bool add_bb = false, Cursor cursor = {},
                 unsigned pending_inputs = 0, unsigned pending_outputs = 0, unsigned fixed_offset = 0,
                 ProductKind kind = ProductKind::Ordinary, uint32_t output_scale = 1) {
        const bool combine_double = kind == ProductKind::EdwardsDouble;
        const bool fuse_dictionary = dictionary_tile && !pending_inputs && kind == ProductKind::Dictionary;
        if (fuse_dictionary) pack_outer_inputs(freq, inputs, in);
        else forward_inputs(freq, inputs, in, pending_inputs);
        if (combine_double && !direct_frequency && !cubic)
            stage1_ntt_edwards_double_spectra<<<grid(1),threads,0,q>>>(freq,modulus.data,s.length);
        const size_t plane = size_t(curves) * s.length;
        const auto spectrum = [&](unsigned id) {
            return id < 5 ? freq + id * plane : fixed.data + (fixed_offset + id - 5) * plane;
        };
        ProductInputs products{};
        for (unsigned op = 0; op < outputs; ++op) {
            products.left[op] = spectrum(a.x[op]);
            products.right[op] = spectrum(b.x[op]);
        }
        if (combine_double && direct_frequency) products.modulus = modulus.data;
        if (add_bb || build_e) products.bb = spectrum(3);
        if (build_e) {
            products.aa = spectrum(2);
            products.modulus = modulus.data;
            products.cached_e = spectrum(4);
        }
        const dim3 tiles(s.length >> ctx.t, curves, outputs);
        const auto invert = [&](auto add, auto difference) {
            constexpr bool add_bb = decltype(add)::value, loose_e = decltype(difference)::value;
            if (short_inverse) {
                const dim3 groups((curves + inverse_rows - 1) / inverse_rows, outputs);
                const std::array kernels{
                    stage1_ntt_short_inverse<1,add_bb,loose_e,ProductInputs>,
                    stage1_ntt_short_inverse<2,add_bb,loose_e,ProductInputs>,
                    stage1_ntt_short_inverse<3,add_bb,loose_e,ProductInputs>,
                    stage1_ntt_short_inverse<4,add_bb,loose_e,ProductInputs>,
                    stage1_ntt_short_inverse<5,add_bb,loose_e,ProductInputs>,
                };
                auto kernel = kernels[ctx.t - 1];
                kernel<<<groups,inverse_threads,0,q>>>(inverse,products,curves,ctx.tblI,product_scale);
            } else if (inverse_warp_input) {
                const auto kernel = fixed_wide_inverse ? fixed_inverse_product
                    : store_last_inverse ? inverse_product_tile<true,add_bb,loose_e,true,true>
                                                       : inverse_product_tile<true,add_bb,loose_e,true>;
                kernel<<<tiles,inverse_threads,inverse_smem,q>>>(
                    inverse,products,s.length,ctx.t,ctx.tblI,product_scale);
            } else if (warp_tile)
                inverse_product_tile<true,add_bb,loose_e><<<tiles,FUSE_TILE_THREADS,tile_smem,q>>>(
                    inverse,products,s.length,ctx.t,ctx.tblI,product_scale);
            else
                inverse_product_tile<false,add_bb,loose_e><<<tiles,FUSE_TILE_THREADS,tile_smem,q>>>(
                    inverse,products,s.length,ctx.t,ctx.tblI,product_scale);
        };
        if (cubic && (combine_double || !cubic_fused_product)) {
            if (combine_double)
                cubic_double<<<dim3(curves,width/threads),threads,0,q>>>(freq,inverse,modulus.data,width,cubic_shift);
            else
                cubic_products<<<dim3((width + threads - 1) / threads,curves,outputs),threads,0,q>>>(
                    inverse,products,width,cubic_shift);
            const auto kernel=warp_tile?cubic_inverse_spectra<true>:cubic_inverse_spectra<false>;
            kernel<<<dim3(width >> ctx.t,outputs*curves*planes),inverse_threads,tile_smem,q>>>(inverse,width,ctx.t,ctx.tblI);
        } else if (cubic) {
            const dim3 blocks(width >> ctx.t,curves,outputs);
            const auto kernel=warp_tile?cubic_inverse_products<true>:cubic_inverse_products<false>;
            kernel<<<blocks,inverse_threads,inverse_smem,q>>>(inverse,products,width,ctx.t,ctx.tblI,cubic_shift);
        } else if (fuse_dictionary) {
            dictionary_product_tile<<<dim3(width >> ctx.t, inputs * curves),forward_threads,tile_smem,q>>>(
                freq, fixed.data + size_t(fixed_offset) * plane, inverse, s.length, ctx.tblF, ctx.tblI);
        } else if (combine_double)
            frequency_inverse<<<tiles,inverse_threads,inverse_smem,q>>>(
                inverse,products,s.length,ctx.t,ctx.tblI,product_scale);
        else if (add_bb && build_e) invert(std::true_type{},std::true_type{});
        else if (build_e) invert(std::false_type{},std::true_type{});
        else if (add_bb) invert(std::true_type{},std::false_type{});
        else invert(std::false_type{},std::false_type{});
        CK(cudaGetLastError());
        ntt_inverse_outer_fused(inverse,ctx,inv2,outputs*curves*planes,q);
        if (!radix5_limb_sums && coalesced)
            radix5_inverse<true,true><<<mixed_grid(outputs),threads,radix5_smem,q>>>(
                inverse,width,nullptr,radix5_inv,rotation_step);
        else if (!radix5_limb_sums && pfa)
            radix5_inverse<true><<<mixed_grid(outputs),threads,0,q>>>(inverse,width,nullptr,radix5_inv,rotation_step);
        else if (!radix5_limb_sums && mixed)
            radix5_inverse<false><<<mixed_grid(outputs),threads,0,q>>>(inverse,width,mixed_inv.data,radix5_inv,0);
        const unsigned batch = outputs * curves;
        const bool route_output = routed && cursor.bits;
        const Sums reduction{{},{},out,0};
        const auto reduce = [&](auto weight, auto direct, auto cache) {
            constexpr bool weighted = decltype(weight)::value;
            constexpr bool route = decltype(direct)::value, cached = decltype(cache)::value;
            combine<true,false,weighted,route,cached><<<batch,threads,cached ? carry_smem : 0,q>>>(
                state.data,freq,s,reduction,cursor);
        };
        const auto summarize = [&](auto input) {
            constexpr auto mode = decltype(input)::value;
            if (route_output)
                segment_limb_sums<mode,false,true><<<segment_grid(outputs),stage1_carry_segment,0,q>>>(
                    state.data,freq,carry_headers.data,s,reduction,cursor,output_scale);
            else
                segment_limb_sums<mode,false,false><<<segment_grid(outputs),stage1_carry_segment,0,q>>>(
                    state.data,freq,carry_headers.data,s,reduction,cursor,output_scale);
        };
        constexpr std::true_type yes{};
        constexpr std::false_type no{};
        if (options.weighted) {
            const WeightedLimbSource source{inverse,inverse_weights.data,digit_table.data,limb_table.data};
            if (warp_reduce) {
                const dim3 groups((curves + weighted_carry_warps - 1) / weighted_carry_warps, outputs);
                const auto kernel = route_output ? weighted_warp_reduce<true> : weighted_warp_reduce<false>;
                kernel<<<groups,32 * weighted_carry_warps,weighted_carry_warps * s.length * sizeof(U),q>>>(
                    state.data,source,s,out,cursor);
            } else if (block_reduce) {
                const size_t bytes = sizeof(U) * (s.length + s.words);
                if (cubic) {
                    cubic_width_dispatch([&](auto plane_layers) {
                        constexpr unsigned length = 3u << decltype(plane_layers)::value;
                        weighted_block_reduce<length,true><<<dim3(curves,outputs),threads,bytes,q>>>(
                            state.data,source,s,out);
                    });
                } else {
                    weighted_block_reduce<><<<dim3(curves,outputs),threads,bytes,q>>>(state.data,source,s,out);
                }
            } else if (fused_weighted) {
                const auto launch = [&](auto route) {
                    constexpr bool routed = decltype(route)::value;
                    auto kernel = fixed_carry_offsets ? weighted_segment_limb_sums<routed,true,5120>
                                  : cached_fused ? weighted_segment_limb_sums<routed,true>
                                               : weighted_segment_limb_sums<routed,false>;
                    kernel<<<segment_grid(outputs),stage1_carry_segment,cached_fused ? carry_cache_smem : 0,q>>>(
                        state.data,source,carry_headers.data,s,out,cursor);
                };
                if (route_output) launch(yes);
                else launch(no);
            } else {
                if (radix5_limb_sums) {
                    const dim3 columns(width / radix5_limb_columns, batch);
                    const auto kernel = pfa ? radix5_inverse_limb_sums<true> : radix5_inverse_limb_sums<false>;
                    kernel<<<columns,radix5_limb_threads,0,q>>>(
                        freq,source,s,log,mixed_inv.data,radix5_inv,rotation_step);
                } else {
                    const dim3 limb_grid((s.words + weighted_limb_threads - 1) / weighted_limb_threads, batch);
                    const auto kernel = cubic_wide ? weighted_limb_sums<true> : weighted_limb_sums<false>;
                    kernel<<<limb_grid,weighted_limb_threads,limb_smem,q>>>(
                        freq,inverse,inverse_weights.data,digit_table.data,limb_table.data,p,s.words,s.length);
                }
                if (output_scale != 1)
                    summarize(std::integral_constant<SegmentInput,SegmentInput::ScaledLimbSums>{});
                else if (segmented) summarize(std::integral_constant<SegmentInput,SegmentInput::LimbSums>{});
                else if (route_output && cached) reduce(yes,yes,yes);
                else if (route_output) reduce(yes,yes,no);
                else if (cached) reduce(yes,no,yes);
                else reduce(yes,no,no);
            }
        } else if (segment_count > inline_carry_segments) {
            linear_limb_sums<<<segment_grid(outputs),stage1_carry_segment,0,q>>>(freq,inverse,s);
            if (segmented) summarize(std::integral_constant<SegmentInput,SegmentInput::LimbSums>{});
            else if (route_output) reduce(yes,yes,no);
            else reduce(yes,no,no);
        } else {
            ecm_launch_carry_cone_kernel<6>(EcmCudaLaunch(grid(outputs),threads,0,q),inverse,freq,s.length,s.bpw,s.length);
            if (segmented) summarize(std::integral_constant<SegmentInput,SegmentInput::Linear>{});
            else if (route_output) reduce(no,yes,no);
            else reduce(no,no,no);
        }
        if (segmented) finish_carries(outputs,out,cursor,route_output,pending_outputs);
    };
    if (pm1) {
        using namespace pm1_detail;
        const size_t bytes = stride * sizeof(uint32_t);
        const auto slot = [&](pm1_detail::Slot id) { return state.data + size_t(id) * stride; };
        const auto copy = [&](auto dst, auto src) {
            CK(cudaMemcpyAsync(slot(dst),slot(src),bytes,cudaMemcpyDeviceToDevice,q));
        };
        const auto product = [&](auto dst, auto a, auto b) {
            if (a == b) mul(1,1,{a},{0},{0},{dst});
            else mul(2,1,{a,b},{0},{1},{dst});
            CK(cudaGetLastError());
        };
        const auto read = [&](auto src) {
            std::vector<uint32_t> host(stride);
            CK(cudaMemcpyAsync(host.data(),slot(src),bytes,cudaMemcpyDeviceToHost,q));
            CK(cudaStreamSynchronize(q));
            return host;
        };
        const auto restore = [&](const std::vector<uint32_t>& host) {
            CK(cudaMemcpyAsync(slot(Current),host.data(),bytes,cudaMemcpyHostToDevice,q));
        };
        if (pm1->base) {
            CK(cudaMemcpyAsync(slot(PowerBase),pm1->base->data(),bytes,cudaMemcpyHostToDevice,q));
        } else {
            CK(cudaMemsetAsync(slot(PowerBase),0,bytes,q));
            CK(cudaMemsetAsync(slot(PowerBase),3,1,q));
        }
        const auto square = [&](uint32_t scale = 1) {
            mul(1,1,{Current},{0},{0},{Current},false,false,{},0,0,0,ProductKind::Ordinary,scale);
            CK(cudaGetLastError());
        };
        const auto multiply_fixed_power = [&](pm1_detail::Slot dst, unsigned power) {
            mul(1,1,{dst},{0},{5},{dst},false,false,{},0,0,power-1);
            CK(cudaGetLastError());
        };
        if (fixed_planes) {
            const size_t plane = size_t(curves) * s.length;
            forward_inputs(fixed.data,1,{PowerBase});
            if (fixed_planes > 1) {
                copy(Scratch,PowerBase);
                for (unsigned power = 2; power <= fixed_planes; ++power) {
                    multiply_fixed_power(Scratch,1);
                    forward_inputs(fixed.data + (power-1)*plane,1,{Scratch});
                }
            }
        }
        const auto step = [&](bool multiply_base) {
            square(pm1_scale && multiply_base ? 3 : 1);
            if (multiply_base && fixed_planes) multiply_fixed_power(Current,1);
            else if (multiply_base && pm1->base) product(Current, Current, PowerBase);
            else if (multiply_base && !pm1_scale) {
                // Work on canonical limbs so the squared coefficient bound stays valid.
                add_inputs(std::false_type{},1,{{Current},{Current},{Scratch},0},{});
                add_inputs(std::false_type{},1,{{Current},{Scratch},{Current},0},{});
                CK(cudaGetLastError());
            }
        };
        const auto emit = [&](unsigned pattern, unsigned width) {
            if (pm1_scale) {
                // x -> x^(2^width) * 3^pattern. Only the last square scales
                // its recovered limb sums, after the field arithmetic.
                uint32_t scale = 1;
                for (unsigned i = 0; i < pattern; ++i) scale *= 3;
                for (unsigned i = 1; i < width; ++i) square();
                square(scale);
            } else if (fixed_planes >= (1u << width) - 1) {
                for (unsigned i = 0; i < width; ++i) square();
                if (pattern) multiply_fixed_power(Current,pattern);
            } else {
                while (width) step((pattern >> --width) & 1u);
            }
        };
        Graph steps[2], groups[1u << pm1_max_graph_bits];
        if (window) {
            const auto capture = [&](Graph& graph, unsigned pattern, unsigned width) {
                CK(cudaStreamBeginCapture(q,cudaStreamCaptureModeThreadLocal));
                emit(pattern,width);
                CK(cudaStreamEndCapture(q,&graph.graph));
                CK(cudaGraphInstantiateWithFlags(&graph.exec,graph.graph,0));
                CK(cudaGraphUpload(graph.exec,q));
            };
            for (unsigned bit = 0; bit < 2; ++bit) capture(steps[bit],bit,1);
            if (window > 1)
                for (unsigned pattern = 0; pattern < (1u << window); ++pattern)
                    capture(groups[pattern],pattern,window);
            CK(cudaStreamSynchronize(q));
        }
        observe_memory();
        const auto advance = [&](unsigned pattern,unsigned width) {
            if (window > 1 && width == window)
                CK(cudaGraphLaunch(groups[pattern].exec,q));
            else if (window)
                while (width) CK(cudaGraphLaunch(steps[(pattern >> --width) & 1u].exec,q));
            else emit(pattern,width);
        };
        setup_guard.unlock();
        if (pm1->ready) (*pm1->ready)(pm1_plan);
        if (pm1->probe) {
            // Check the maximum window multiplier against GMP before timing.
            const auto width = std::max(window,1u);
            const auto power = 1u << width;
            advance(power-1,width);
            ecm_edwards::Integer expected,modulus,actual;
            mpz_import(expected.v,initial.size(),-1,4,0,0,initial.data());
            mpz_setbit(modulus.v,p);
            mpz_sub_ui(modulus.v,modulus.v,1);
            mpz_powm_ui(expected.v,expected.v,power,modulus.v);
            for (unsigned i = 1; i < power; ++i) mpz_mul_ui(expected.v,expected.v,3);
            mpz_mod(expected.v,expected.v,modulus.v);
            const auto check=read(Current);
            mpz_import(actual.v,check.size(),-1,4,0,0,check.data());
            if (mpz_cmp(actual.v,expected.v)!=0) throw std::runtime_error("P-1 tune residue mismatch");
            (*pm1->probe)(q,advance);
        } else replay(pm1->exponent,pm1->progress,pm1->verified,std::max(window,1u),
                      advance,copy,product,read,restore);
        CK(cudaStreamSynchronize(q));
        return result;
    }
    if (auxiliary_sigmas) {
        // Each lane starts at Q and follows its own scalar.
        // The fixed spectra initialized above contain Q.x=12 and Q.y=40.
        constexpr auto X = X0, Y = Z0, Z = X1;
        const auto sum = [&](unsigned a, unsigned b, unsigned out, bool subtract = false) {
            add_inputs(std::false_type{},1,{{a},{b},{out},unsigned(subtract)},{});
        };
        const auto twice = [&] {
            // AA=X^2, BB=Y^2, C=Z^2, D=YZ. S=4 X BB, M=3 AA-8 C^2.
            mul(3,4,{X,Y,Z},{0,1,2,1},{0,1,2,2},{AA,BB,C,D});
            mul(3,3,{BB,X,C},{0,0,2},{0,1,2},{B,Sum,E});
            sum(D,D,Z);
            sum(Sum,Sum,C); sum(C,C,Sum);
            sum(E,E,C); sum(C,C,D); sum(D,D,E);
            sum(AA,AA,D); sum(AA,D,A); sum(A,E,DA,true);
            sum(B,B,C); sum(C,C,D); sum(D,D,B);
            mul(1,1,{DA},{0},{0},{C});
            sum(Sum,Sum,D); sum(C,D,X,true);
            sum(Sum,X,D,true);
            mul(2,1,{DA,D},{0},{1},{C});
            sum(C,B,Y,true);
        };
        const auto add_base = [&] {
            // H=12 Z^2-X, J=40 Z^3-Y. K=X H^2 and B=H^3.
            mul(1,1,{Z},{0},{0},{AA});
            mul(2,2,{AA,Z},{0,0},{1,5},{BB,C});
            mul(1,1,{BB},{0},{6},{D});
            sum(C,X,DA,true); sum(D,Y,CB,true);
            mul(3,3,{DA,CB,Z},{0,1,2},{0,1,0},{AA,D,Z});
            mul(3,2,{AA,DA,X},{0,0},{1,2},{B,Sum});
            mul(2,1,{Y,B},{0},{1},{C});
            sum(D,B,E,true); sum(Sum,Sum,A); sum(E,A,X,true);
            sum(Sum,X,A,true);
            mul(2,1,{CB,A},{0},{1},{D});
            sum(D,C,Y,true);
        };
        Buffer<uint64_t> scalars(curves);
        CK(cudaMemcpyAsync(scalars.data,auxiliary_sigmas->data(),curves*sizeof(uint64_t),cudaMemcpyHostToDevice,q));
        const auto maximum = *std::max_element(auxiliary_sigmas->begin(),auxiliary_sigmas->end());
        int bit = 63;
        while (!(maximum >> bit)) --bit;
        while (--bit >= 0) {
            twice();
            const bool added = std::any_of(auxiliary_sigmas->begin(),auxiliary_sigmas->end(),
                [bit](uint64_t k){ return (k >> bit) > 1 && ((k >> bit) & 1); });
            if (added) {
                CK(cudaMemcpyAsync(state.data+size_t(XA)*stride,state.data,3*stride*sizeof(uint32_t),
                                   cudaMemcpyDeviceToDevice,q));
                add_base();
            }
            select_auxiliary_bit<<<dim3((s.words+threads-1)/threads,curves,3),threads,0,q>>>(
                state.data,s,scalars.data,bit,added);
            CK(cudaGetLastError());
        }
        result.points.resize(3*stride);
        CK(cudaMemcpyAsync(result.points.data(),state.data,3*stride*sizeof(uint32_t),cudaMemcpyDeviceToHost,q));
        CK(cudaStreamSynchronize(q));
        return result;
    }
    auto step=[&](unsigned i) {
        if (square_probe) {
            mul(1,1,{X0},{0},{0},{X0});
            CK(cudaGetLastError());
            return;
        }
        const Cursor cursor{scalar.data,start.data,i,count};
        add_inputs(std::true_type{},4,
            {{X0,X0,X1,X1},{Z0,Z0,Z1,Z1},{A,B,C,D},0b1010},cursor,deferred ? 0b1111 : 0);
        mul(4,4,{A,B,C,D},{0,1,3,2},{0,1,0,1},{AA,BB,DA,CB},false,false,{},deferred ? 0b1111 : 0);
        add_inputs(std::false_type{},loose ? 2 : 3,
            {{DA,DA,AA},{CB,CB,BB},{Sum,Diff,E},0b110},{},deferred ? (loose ? 0b11 : 0b111) : 0);
        // The old X/Z values were consumed by the first combine. Both
        // multiplication batches can now write the final X/Z slots.
        // The second pack resolves the fresh sums. DiffSq and the fused Sum
        // can then stay tentative until pack three. X/Z are always finalized.
        mul(loose ? 4 : 5,4,{Sum,Diff,AA,BB,E},{0,1,2,4},{0,1,3,5},{XA,DiffSq,XD,fma ? Sum : AE},
            loose,fma,cursor,deferred ? (loose ? 0b11 : 0b10011) : 0,
            deferred ? (fma ? 0b1010 : 0b10) : 0);
        if (!fma)
            add_inputs(std::false_type{},1,{{BB},{AE},{Sum},0},{},deferred ? 1 : 0);
        // E is still in frequency plane 4 from the second batch (packed
        // normally or produced by the last product tile). Both
        // reducers only overwrite planes 0–3, and this pack uses planes 0–1.
        constexpr unsigned cached_e = 4;
        mul(2,2,{DiffSq,Sum},{0,cached_e},{6,1},{ZA,ZD},false,false,cursor,deferred ? 0b11 : 0);
        if (!routed)
            route<<<dim3((s.words+threads-1)/threads,curves),threads,0,q>>>(state.data,s,cursor);
        CK(cudaGetLastError());
    };
    constexpr auto x = X0, y = Z0, z = X1, t = Z1;
    // Both the 2P setup entry and the NAF dictionary use X, Y, dT, X+Y
    // spectra in consecutive planes, one fixed operand per variable input.
    const auto multiply_dictionary = [&](unsigned first_plane, unsigned pending) {
        mul(4,4,{x,y,t,Sum},{0,1,2,3},{5,6,7,8},{AA,BB,C,D},
            false,false,{},pending,0,first_plane,ProductKind::Dictionary);
    };
    const auto edwards_double = [&](bool extended) {
        if (frequency_double) {
            mul(3,4,{x,y,z},{0,1,2,3},{0,1,2,3},{Sum,Diff,A,B},false,false,{},0,
                deferred ? 0b1111 : 0,0,ProductKind::EdwardsDouble);
        } else {
            // Reduce each product before forming the sums and differences.
            mul(3,4,{x,y,z},{0,1,2,0},{0,1,2,1},{AA,BB,C,E});
            // Only A and D are read by the subtraction. Pack can resolve the
            // pending carries for Sum, Diff and B together.
            add_inputs(std::false_type{},4,{{C,E,AA,AA},{C,E,BB,BB},{D,Sum,A,B},0b1000},{},deferred ? 0b1010 : 0);
            add_inputs(std::false_type{},1,{{A},{D},{Diff},1},{},deferred ? 1 : 0);
        }
        mul(4,extended ? 4 : 3,{Sum,Diff,A,B},{0,2,1,0},{1,3,2,3},{x,y,z,t},
            false,false,{},deferred ? (frequency_double ? 0b1111 : 0b1011) : 0);
    };
    const auto edwards_step = [&](int digit) {
        edwards_double(digit != 0);
        if (!digit) return;

        const bool negative = digit < 0;
        add_inputs(std::false_type{},1,{{y},{x},{Sum},unsigned(negative)},{},deferred ? 1 : 0);
        multiply_dictionary(4 * (unsigned(std::abs(digit)) / 2), deferred ? 0b1000 : 0);
        // Sum is consumed by the next subtraction. All other coordinates
        // are consumed directly by the second multiply's pack.
        if (negative) {
            add_inputs(std::false_type{},4,{{BB,z,z,BB},{AA,C,C,AA},{Sum,Diff,A,B},0b0101},{},deferred ? 0b1110 : 0);
            add_inputs(std::false_type{},1,{{Sum},{D},{E},1},{},deferred ? 1 : 0);
        } else {
            add_inputs(std::false_type{},4,{{D,z,z,BB},{AA,C,C,AA},{Sum,Diff,A,B},0b1011},{},deferred ? 0b1110 : 0);
            add_inputs(std::false_type{},1,{{Sum},{BB},{E},1},{},deferred ? 1 : 0);
        }
        mul(4,3,{E,Diff,A,B},{0,2,1},{1,3,2},{x,y,z},false,false,{},deferred ? 0b1111 : 0);
        CK(cudaGetLastError());
    };
    if (naf) {
        std::vector<uint32_t> fallback_table;
        if (naf->build_dictionary && !gpu_dictionary) {
            const auto start = now_s();
            fallback_table = naf->host_fallback(result);
            result.dictionary_host_ms = 1000 * (now_s() - start);
            result.dictionary_fallback_host_bytes = fallback_table.size() * sizeof(uint32_t);
        }
        const auto device_dictionary_start = now_s();
        Event dictionary_begin, dictionary_end;
        CK(cudaEventRecord(dictionary_begin.value, q));
        if (gpu_dictionary) {
            // Extra storage is transient: projective X/Y and the complete Z
            // product tree. Each level overwrites its children with inverses.
            Buffer<uint32_t> work(4ull * entries * stride);
            const size_t plane = size_t(curves) * s.length;
            const unsigned curve_d = fixed_planes - 1;
            const auto xy = [&](unsigned entry, unsigned coord) { return work.data + (2ull * entry + coord) * stride; };
            const auto tree = [&](unsigned node) { return work.data + (2ull * entries + node) * stride; };
            const auto slot = [&](unsigned i) { return state.data + i * stride; };
            const size_t bytes = stride * sizeof(uint32_t);
            const auto copy = [&](uint32_t* dst, const uint32_t* src) {
                CK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, q));
            };
            const auto copy_planes = [&](const auto& planes) {
                stage1_ntt_copy_dictionary_planes<<<unsigned((stride + 127) / 128),128,0,q>>>(planes,stride);
                CK(cudaGetLastError());
            };
            const auto save_point = [&](unsigned entry) {
                copy_planes(Stage1DictionaryCopies<3>{{xy(entry,0),xy(entry,1),tree(entries+entry)},
                                                      {slot(x),slot(y),slot(z)}});
            };
            std::vector<uint32_t> one(stride);
            for (unsigned lane=0; lane<curves; ++lane) one[lane*s.words] = 1;
            CK(cudaMemcpyAsync(slot(x), naf->dictionary.data(), 2 * bytes, cudaMemcpyHostToDevice, q));
            CK(cudaMemcpyAsync(slot(z), one.data(), bytes, cudaMemcpyHostToDevice, q));
            CK(cudaMemcpyAsync(slot(A24), naf->dictionary.data()+2*stride, bytes, cudaMemcpyHostToDevice, q));
            forward_inputs(fixed.data + curve_d * plane, 1, {{A24}});
            save_point(0);
            if (entries > 1) {
                edwards_double(true);
                mul(1,1,{t},{0},{5},{D},false,false,{},0,0,curve_d);
                add_inputs(std::false_type{},1,{{x},{y},{Sum},0},{});
                // Fixed 2P spectra: X, Y, dT, X+Y, Z. They are used only
                // until the projective sequence has been generated.
                forward_inputs(fixed.data, 5, {{x,y,D,Sum,z}});
                copy_planes(Stage1DictionaryCopies<3>{{slot(x),slot(y),slot(z)},{xy(0,0),xy(0,1),tree(entries)}});
                mul(2,1,{x,y},{0},{1},{t});
                for (unsigned entry=1; entry<entries; ++entry) {
                    add_inputs(std::false_type{},1,{{x},{y},{Sum},0},{});
                    multiply_dictionary(0, 0);
                    mul(1,1,{z},{0},{9},{DA});
                    add_inputs(std::false_type{},4,{{D,DA,DA,BB},{AA,C,C,AA},{Sum,Diff,A,B},0b1011},{});
                    add_inputs(std::false_type{},1,{{Sum},{BB},{E},1},{});
                    mul(4,4,{E,Diff,A,B},{0,2,1,0},{1,3,2,3},{x,y,z,t});
                    save_point(entry);
                }
            }
            // Two sibling pairs per call. Ascend with Z_left * Z_right.
            for (unsigned end=entries; end>1; end/=2) {
                for (unsigned node=end/2; node<end; node+=2) {
                    const bool pair = node+1<end;
                    if (pair) {
                        copy_planes(Stage1DictionaryCopies<4>{{slot(A),slot(B),slot(C),slot(D)},
                            {tree(2*node),tree(2*node+1),tree(2*node+2),tree(2*node+3)}});
                    } else {
                        copy_planes(Stage1DictionaryCopies<2>{{slot(A),slot(B)},{tree(2*node),tree(2*node+1)}});
                    }
                    mul(pair ? 4 : 2,pair ? 2 : 1,{A,B,C,D},{0,2},{1,3},{AA,BB});
                    if (pair)
                        copy_planes(Stage1DictionaryCopies<2>{{tree(node),tree(node+1)},{slot(AA),slot(BB)}});
                    else copy(tree(node),slot(AA));
                }
            }
            std::vector<uint32_t> roots(stride), factors(stride);
            result.dictionary_status.resize(curves);
            CK(cudaMemcpyAsync(roots.data(), tree(1), bytes, cudaMemcpyDeviceToHost,q));
            CK(cudaStreamSynchronize(q));
            const auto inverse_start = now_s();
            ecm_edwards::Integer modulus, value, gcd;
            mpz_setbit(modulus.v,p);
            mpz_sub_ui(modulus.v,modulus.v,1);
            if (naf->check_modulus) {
                if (mpz_cmp_ui(naf->check_modulus, 1) <= 0 ||
                    !mpz_divisible_p(modulus.v, naf->check_modulus))
                    throw std::invalid_argument("Edwards dictionary check modulus must divide M_p");
                mpz_set(modulus.v, naf->check_modulus);
            }
            std::vector<unsigned char> invertible(curves);
            for_host_lanes(p,curves,[&](size_t lane) {
                ecm_edwards::Integer value, inverse;
                auto* root = roots.data()+lane*s.words;
                mpz_import(value.v,s.words,-1,sizeof(uint32_t),0,0,root);
                if (!mpz_invert(inverse.v,value.v,modulus.v)) return;
                std::fill_n(root,s.words,0u);
                mpz_export(root,nullptr,-1,sizeof(uint32_t),0,0,inverse.v);
                invertible[lane] = 1;
            });
            for (unsigned lane=0; lane<curves; ++lane) {
                if (invertible[lane]) continue;
                auto* root = roots.data()+lane*s.words;
                mpz_import(value.v,s.words,-1,sizeof(uint32_t),0,0,root);
                std::fill_n(root,s.words,0u);
                mpz_gcd(gcd.v,value.v,modulus.v);
                if (mpz_cmp(gcd.v,modulus.v)==0) {
                    // Read individual denominators only for an exceptional lane.
                    std::vector<uint32_t> leaves(size_t(entries)*s.words);
                    CK(cudaMemcpy2DAsync(leaves.data(),s.words*sizeof(uint32_t),
                        tree(entries)+lane*s.words,bytes,s.words*sizeof(uint32_t),entries,cudaMemcpyDeviceToHost,q));
                    CK(cudaStreamSynchronize(q));
                    for (unsigned entry=0; entry<entries; ++entry) {
                        mpz_import(value.v,s.words,-1,sizeof(uint32_t),0,0,leaves.data()+entry*s.words);
                        mpz_gcd(gcd.v,value.v,modulus.v);
                        if (mpz_cmp_ui(gcd.v,1)>0 && mpz_cmp(gcd.v,modulus.v)<0) break;
                    }
                }
                if (mpz_cmp_ui(gcd.v,1)>0 && mpz_cmp(gcd.v,modulus.v)<0) {
                    result.dictionary_status[lane]=1;
                    mpz_export(factors.data()+lane*s.words,nullptr,-1,sizeof(uint32_t),0,0,gcd.v);
                } else {
                    result.dictionary_status[lane]=2;
                }
            }
            result.dictionary_inverse_ms = 1000 * (now_s()-inverse_start);
            if (std::any_of(result.dictionary_status.begin(),result.dictionary_status.end(),[](auto flag){return flag==1;}))
                result.dictionary_factors = std::move(factors);
            if (naf->affine_output && std::any_of(result.dictionary_status.begin(),result.dictionary_status.end(),
                    [](auto status){ return status != 0; })) return result;
            CK(cudaMemcpyAsync(tree(1),roots.data(),bytes,cudaMemcpyHostToDevice,q));
            // Descend with inverse_parent * sibling. Both products consume
            // the old children before either child is overwritten.
            for (unsigned node=1; node<entries; ++node) {
                copy_planes(Stage1DictionaryCopies<3>{{slot(A),slot(B),slot(C)},
                                                      {tree(node),tree(2*node),tree(2*node+1)}});
                mul(3,2,{A,B,C},{0,0},{2,1},{AA,BB});
                copy_planes(Stage1DictionaryCopies<2>{{tree(2*node),tree(2*node+1)},{slot(AA),slot(BB)}});
            }
            for (unsigned entry=0; entry<entries; ++entry) {
                copy_planes(Stage1DictionaryCopies<3>{{slot(x),slot(y),slot(z)},
                                                      {xy(entry,0),xy(entry,1),tree(entries+entry)}});
                mul(3,2,{x,y,z},{0,1},{2,2},{x,y});
                mul(2,1,{x,y},{0},{1},{t});
                mul(1,1,{t},{0},{5},{D},false,false,{},0,0,curve_d);
                add_inputs(std::false_type{},1,{{x},{y},{Sum},0},{});
                if (naf->affine_output) {
                    auto* out = naf->affine_output->data() + size_t(4) * entry * stride;
                    for (auto coord : {x,y,D,Sum}) {
                        CK(cudaMemcpyAsync(out,slot(coord),bytes,cudaMemcpyDeviceToHost,q));
                        out += stride;
                    }
                } else forward_inputs(fixed.data + size_t(4) * entry * plane, 4, {{x,y,D,Sum}});
            }
            CK(cudaStreamSynchronize(q));
            observe_memory();
            if (naf->affine_output) return result;
        } else {
            const auto& table = naf->build_dictionary ? fallback_table : naf->dictionary;
            for (unsigned entry=0; entry<entries; ++entry) {
                CK(cudaMemcpyAsync(state.data + A24 * stride, table.data() + 4 * entry * stride,
                                   4 * stride * sizeof(uint32_t), cudaMemcpyHostToDevice, q));
                forward_inputs(fixed.data + size_t(4) * entry * curves * s.length, 4, {{A24,Base,A,B}});
            }
        }
        CK(cudaEventRecord(dictionary_end.value, q));
        CK(cudaEventSynchronize(dictionary_end.value));
        float elapsed_ms = 0;
        CK(cudaEventElapsedTime(&elapsed_ms, dictionary_begin.value, dictionary_end.value));
        result.dictionary_gpu_ms = elapsed_ms;
        result.dictionary_device_wall_ms = 1000 * (now_s() - device_dictionary_start);
    }
    Graph full, tail, warm_tail;
    std::array<Graph, 7> zero_graphs;
    const unsigned target_ms = control ? control->sample_target_ms : 0;
    const auto* blocks_by_size = target_ms ? control->sample_blocks : nullptr;
    auto capture=[&](Graph& graph, unsigned count, unsigned wrap = 0) {
        CK(cudaStreamBeginCapture(q,cudaStreamCaptureModeThreadLocal));
        for (unsigned i=0;i<count;++i) {
            if (naf) edwards_step(0);
            else step(i);
        }
        if (!naf) advance<<<1,1,0,q>>>(start.data,count,wrap);
        CK(cudaStreamEndCapture(q,&graph.graph));
        CK(cudaGraphInstantiateWithFlags(&graph.exec,graph.graph,0));
        CK(cudaGraphUpload(graph.exec,q));
    };
    const unsigned resume = control ? control->resume : 0;
    const unsigned remaining = count - resume;
    const unsigned warmup = control ? control->warmup_iterations : 0;
    if (warmup > remaining) throw std::invalid_argument("Stage1 NTT warmup exceeds scalar");
    if (naf) {
        for (unsigned i=0; (1u << i) <= std::min(window, remaining); ++i)
            capture(zero_graphs[i], 1u << i);
    } else if (!blocks_by_size && window && remaining >= window) capture(full,window,target_ms ? count : 0);
    if (!naf && !blocks_by_size && window && remaining % window) capture(tail,remaining % window,target_ms ? count : 0);
    if (window && warmup % window && warmup % window != remaining % window)
        capture(warm_tail,warmup % window);
    Event begin,end,boundary;
    CK(cudaStreamSynchronize(q));
    observe_memory();
    result.length=s.length; result.digit_bits=s.bpw; result.passes=ctx.passes_fwd+unsigned(mixed); result.tile=ctx.t;
    result.setup_seconds=now_s()-prep;
    result.points.resize(checkpoint_slots*stride);
    auto reset = [&] {
        CK(cudaMemcpyAsync(state.data,initial.data(),checkpoint_slots*stride*4,cudaMemcpyHostToDevice,q));
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
    if (control && control->iterations_started) {
        const auto used = std::max(result.owned_bytes, result.observed_bytes);
        control->iterations_started(result.budget_bytes > used ? result.budget_bytes - used : 0);
    }
    const int groups = int(samples);
    std::vector<uint32_t> pilot, group_endpoint;
    std::map<std::pair<unsigned,unsigned>, Graph> binary_graphs;
    const bool probe = blocks_by_size && blocks_by_size->size() > 1;
    for (int sample = probe ? -1 : 0; sample < groups + (target_ms ? 2 : 0); ++sample) {
        if (target_ms && control->poll && control->poll(0).stop) {
            result.stopped = true;
            break;
        }
        const TuneBlock* block = blocks_by_size ? &(*blocks_by_size)[result.sample_block_index] : nullptr;
        const unsigned stop = block ? block->steps : count;
        const auto* digits = naf ? &(block && !block->digits.empty() ? block->digits : naf->digits) : nullptr;
        auto full_exec = full.exec, tail_exec = tail.exec;
        if (block && !naf && window) {
            // Each captured cursor wraps at this block's end, including short tails.
            const auto graph = [&](unsigned steps) {
                auto& g = binary_graphs[{stop, steps}];
                if (!g.exec) capture(g, steps, stop);
                return g.exec;
            };
            if (stop >= window) full_exec = graph(window);
            if (stop % window) tail_exec = graph(stop % window);
        }
        reset();
        CK(cudaStreamSynchronize(q));
        const double wall=now_s();
        CK(cudaEventRecord(begin.value,q));
        const bool timed_group = target_ms && sample > 0 && sample <= groups;
        const unsigned blocks = timed_group ? result.blocks_per_sample : 1;
        auto replay = [&] {
            unsigned cursor = resume;
            unsigned next_poll = resume;
            unsigned zero_end = resume;
            while (cursor < stop) {
                if (!target_ms && control && control->poll && cursor >= next_poll) {
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
                if (naf) {
                    const int digit = (*digits)[cursor];
                    if (!digit && window) {
                        if (zero_end <= cursor) {
                            unsigned limit = std::min(window, stop - cursor);
                            if (block && block->span) {
                                const unsigned span = block->span;
                                limit = std::min(limit, span - cursor % span);
                            }
                            zero_end = cursor + 1;
                            while (zero_end - cursor < limit && !(*digits)[zero_end])
                                ++zero_end;
                        }
                        unsigned i = 0;
                        while ((2u << i) <= zero_end - cursor) ++i;
                        CK(cudaGraphLaunch(zero_graphs[i].exec,q));
                        cursor += 1u << i;
                    } else {
                        edwards_step(digit);
                        ++cursor;
                    }
                } else if (!window) {
                    step(cursor - resume);
                    ++cursor;
                } else if (stop - cursor >= window) {
                    CK(cudaGraphLaunch(full_exec,q));
                    cursor += window;
                } else {
                    CK(cudaGraphLaunch(tail_exec,q));
                    cursor = stop;
                }
            }
            result.completed_bits = cursor;
        };
        for (unsigned block = 0; block < blocks; ++block) {
            replay();
            if (result.stopped) break;
        }
        CK(cudaEventRecord(end.value,q));
        CK(cudaEventSynchronize(end.value));
        result.wall_ms.push_back((now_s()-wall)*1000);
        float ms=0; CK(cudaEventElapsedTime(&ms,begin.value,end.value));
        if (sample < 0) {
            const double fit = target_ms * double(stop) / std::max(0.001, double(ms));
            while (result.sample_block_index+1 < blocks_by_size->size() &&
                   (*blocks_by_size)[result.sample_block_index+1].steps <= fit)
                ++result.sample_block_index;
            if (result.sample_block_index) continue; // Warm the selected block to calibrate its actual cost.
            sample = 0; // The short probe also serves as the warmup.
        }
        if (!target_ms || timed_group) result.device_ms.push_back(ms);
        else if (!sample) {
            const double wanted = std::round(target_ms / std::max(0.001, double(ms)));
            result.blocks_per_sample = unsigned(std::clamp(wanted, 1., double(UINT32_MAX / std::max(1u, stop))));
        }
        const double read=now_s();
        std::vector<uint32_t> got(checkpoint_slots*stride);
        CK(cudaMemcpy(got.data(),state.data,got.size()*4,cudaMemcpyDeviceToHost));
        result.readback_seconds+=now_s()-read;
        if (target_ms && !sample) pilot = got;
        if (timed_group && sample == 1) group_endpoint = got;
        const auto& expected = !target_ms ? result.points : timed_group ? group_endpoint : pilot;
        if (sample && !result.stopped && got != expected)
            throw std::runtime_error("nondeterministic NTT Stage1 endpoint");
        result.points=std::move(got);
    }
    observe_memory();
    return result;
}
Run pm1_probe_lane(int device, unsigned p, const ecm_stage1::Stage1NttOptions& options,
                   Plan& plan, const Pm1Probe& probe) {
    Pm1Progress initial;
    Control control;
    control.describe_only = !probe;
    control.describe = [&](const Plan& value) { plan = value; };
    if (probe) {
        initial.residue.resize((p+31)/32);
        uint32_t word=0x85ebca6bu;
        for (auto& limb:initial.residue) {
            word^=word<<13; word^=word>>17; word^=word<<5;
            limb=word;
        }
        if (p%32) initial.residue.back()&=(1u<<(p%32))-1;
        initial.residue.front()&=~2u;
    }
    const std::vector<uint32_t> exponent;
    const std::function<bool(const Pm1Progress&)> verified;
    const Pm1Request request{exponent,initial,verified,&probe};
    return run_ladder(device,p,1,initial.residue,{},0,options,&control,1,nullptr,nullptr,&request);
}

Pm1Progress pm1_stage1(int device, unsigned p, const std::vector<uint32_t>& exponent,
                       Pm1Progress start, const ecm_stage1::Stage1NttOptions& options,
                       const std::function<bool(const Pm1Progress&)>& verified,
                       const std::vector<uint32_t>* extension_base,
                       const std::function<void(const Plan&)>& ready) {
    const uint64_t bits = pm1_detail::exponent_bits(exponent);
    if (p < ecm_stage1::stage1_ntt_min_exponent || p > ecm_stage1::stage1_ntt_max_exponent ||
        !start.k || start.k > bits || start.residue.size() != (p + 31) / 32 ||
        (extension_base && extension_base->size()!=start.residue.size()) ||
        std::none_of(start.residue.begin(),start.residue.end(),[](uint32_t word){ return word != 0; }))
        throw std::invalid_argument("invalid P-1 Stage1 checkpoint or modulus");
    auto weighted = options;
    weighted.weighted = true;
    Control control;
    const Pm1Request request{exponent,start,verified,nullptr,extension_base,ready ? &ready : nullptr};
    run_ladder(device,p,1,start.residue,{},0,weighted,&control,1,nullptr,nullptr,&request);
    return start;
}

std::vector<uint32_t> auxiliary_points(int device, unsigned p, const std::vector<uint64_t>& sigmas,
                                       const ecm_stage1::Stage1NttOptions& options, TransformWorkspace* workspace, size_t* owned_bytes) {
    if (p < ecm_stage1::stage1_ntt_min_exponent || p > ecm_stage1::stage1_ntt_max_exponent)
        throw std::invalid_argument("invalid auxiliary modulus width");
    if (sigmas.empty() || sigmas.size() > ecm_stage1::stage1_ntt_max_curves || std::find(sigmas.begin(),sigmas.end(),0) != sigmas.end())
        throw std::invalid_argument("invalid auxiliary curve scalar");
    const size_t words = (p+31)/32;
    const unsigned curves = unsigned(sigmas.size());
    const size_t stride = curves*words;
    std::vector<uint32_t> initial(6*stride);
    for (unsigned lane=0; lane<curves; ++lane) {
        auto* base = initial.data()+lane*words;
        base[0]=12; base[stride]=40; base[2*stride]=1;
        base[4*stride]=12; base[5*stride]=40;
    }
    Control control;
    control.workspace = workspace;
    auto run = run_ladder(device,p,curves,initial,{0},1,options,&control,1,nullptr,&sigmas);
    if (owned_bytes) *owned_bytes = run.owned_bytes;
    return std::move(run.points);
}
Run edwards_naf(int device, unsigned p, unsigned curves, const std::vector<uint32_t>& initial,
                const EdwardsNaf& naf, const ecm_stage1::Stage1NttOptions& options,
                const Control& control, unsigned samples) {
    if (naf.digits.size() > UINT32_MAX - 64)
        throw std::invalid_argument("Edwards NTT NAF exceeds the cursor range");
    return run_ladder(device, p, curves, initial, {}, unsigned(naf.digits.size()), options, &control, samples, &naf);
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
