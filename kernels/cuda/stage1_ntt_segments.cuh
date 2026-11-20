#pragma once

#include "ecm/stage1/ntt_length.h"
#include "stage1_ntt_layout.cuh"
#include "../../src/ecm/stage1/cuda/ntt/weighted.cuh"

namespace ecm_mersenne_ntt::stage1_layout {

constexpr unsigned stage1_carry_segment = 256;
constexpr unsigned stage1_max_segments = 128;
static_assert(stage1_max_segments * stage1_carry_segment * 32 == ecm_stage1::stage1_ntt_max_exponent);

__host__ __device__ constexpr unsigned stage1_segment_count(unsigned words)
{
    return (words + stage1_carry_segment - 1) / stage1_carry_segment;
}

// A carry transfer is kill (0), propagate (1) or generate (2).
__device__ __forceinline__ unsigned stage1_last_carry(unsigned stop, unsigned generate)
{
    return stop ? ((generate >> (31 - __clz(stop))) & 1u) * 2 : 1;
}

// Each lane owns one limb. Store the result for an incoming zero carry.
// A segment header stores its transfer and the number of low limbs affected
// by an incoming one, including the first non-propagating limb.
__device__ __forceinline__ uint32_t stage1_summarize_segment(
    unsigned long long raw, uint32_t mask, unsigned* headers)
{
    constexpr unsigned warps = stage1_carry_segment / 32;
    __shared__ unsigned stops[warps], generates[warps];
    const unsigned lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    const unsigned stop = __ballot_sync(0xffffffffu, raw != mask);
    const unsigned gen = __ballot_sync(0xffffffffu, raw > mask);
    if (!lane) {
        stops[warp] = stop;
        generates[warp] = gen;
    }
    __syncthreads();
    unsigned prefix = stage1_last_carry(stop & ((1u << lane) - 1), gen);
    for (int w = int(warp) - 1; prefix == 1 && w >= 0; --w)
        prefix = stage1_last_carry(stops[w], generates[w]);
    if (threadIdx.x == 0) {
        unsigned first = stage1_carry_segment, whole = 1;
#pragma unroll
        for (unsigned w = 0; w < warps; ++w) {
            if (stops[w]) {
                first = min(first, 32 * w + unsigned(__ffs(stops[w])));
                whole = stage1_last_carry(stops[w], generates[w]);
            }
        }
        headers[blockIdx.x] = (first << 2) | whole;
    }
    return (uint32_t(raw) + unsigned(prefix == 2)) & mask;
}

// Each wide group scans its segment transfers in parallel. The first
// non-propagating predecessor in cyclic order determines its carry-in.
__device__ __forceinline__ void stage1_segment_corrections(
    unsigned* limits, const unsigned* headers, unsigned segments)
{
    if (segments > 64) {
        constexpr unsigned max_groups = stage1_max_segments / 32;
        __shared__ unsigned transfers[max_groups];
        const unsigned group = threadIdx.x / 32, lane = threadIdx.x % 32;
        const unsigned groups = (segments + 31) / 32;
        const unsigned header = threadIdx.x < segments ? headers[threadIdx.x] : 1;
        const unsigned tag = header & 3;
        const unsigned stop = __ballot_sync(0xffffffffu, tag != 1);
        const unsigned gen = __ballot_sync(0xffffffffu, tag == 2);
        if (!lane && group < groups) transfers[group] = stage1_last_carry(stop, gen);
        __syncthreads();
        if (threadIdx.x < segments) {
            unsigned prefix = stage1_last_carry(stop & ((1u << lane) - 1), gen);
            unsigned previous = group;
            for (unsigned i = 0; prefix == 1 && i < groups; ++i) {
                previous = previous ? previous - 1 : groups - 1;
                prefix = transfers[previous];
            }
            // An all-propagate ring encodes M, canonicalized to zero.
            limits[threadIdx.x] = prefix != 0 ? header >> 2 : 0;
        }
        __syncthreads();
        return;
    }
    if (threadIdx.x < 32) {
        const unsigned lane = threadIdx.x;
        const unsigned header = lane < segments ? headers[lane] : 1;
        const unsigned tag = header & 3;
        const unsigned stop = __ballot_sync(0xffffffffu, tag != 1);
        const unsigned gen = __ballot_sync(0xffffffffu, tag == 2);
        const unsigned prefix = stage1_last_carry(stop & ((1u << lane) - 1), gen);
        unsigned whole = stage1_last_carry(stop, gen);
        if (segments > 32) {
            const unsigned upper = lane + 32 < segments ? headers[lane + 32] : 1;
            const unsigned upper_tag = upper & 3;
            const unsigned upper_stop = __ballot_sync(0xffffffffu, upper_tag != 1);
            const unsigned upper_gen = __ballot_sync(0xffffffffu, upper_tag == 2);
            unsigned upper_prefix = stage1_last_carry(upper_stop & ((1u << lane) - 1), upper_gen);
            if (upper_prefix == 1) upper_prefix = whole;
            if (upper_stop) whole = stage1_last_carry(upper_stop, upper_gen);
            const bool upper_carry = upper_prefix == 2 || (upper_prefix == 1 && whole != 0);
            limits[lane + 32] = upper_carry ? upper >> 2 : 0;
        }
        const bool carry = prefix == 2 || (prefix == 1 && whole != 0);
        limits[lane] = carry ? header >> 2 : 0;
    }
    __syncthreads();
}

__device__ __forceinline__ uint32_t stage1_segment_word(
    const uint32_t* row, const unsigned* limits, unsigned word, uint32_t mask)
{
    const unsigned limit = limits[word / stage1_carry_segment];
    return (row[word] + unsigned(word % stage1_carry_segment < limit)) & mask;
}

enum class SegmentInput { Add, Linear, Weighted };

template<SegmentInput Input, bool Select, bool Routed>
__global__ void segment_limb_sums(uint32_t* state, const unsigned long long* products, unsigned* headers,
                                 Shape s, Sums ops, Cursor cursor) {
    const unsigned op = blockIdx.y / s.curves, curve = blockIdx.y % s.curves;
    const unsigned slot = output_slot<Routed>(ops.out.x[op],cursor);
    const size_t row = size_t(slot) * s.curves + curve;
    const unsigned i = blockIdx.x * stage1_carry_segment + threadIdx.x;
    const uint32_t mask = i < s.words ? limb_mask(s,i) : 0;
    unsigned long long value = 0; // Inactive lanes propagate.
    if (i < s.words) {
        if constexpr (Input == SegmentInput::Weighted) {
            const auto* sums = products + size_t(blockIdx.y) * s.words;
            value = weighted_limb(sums,s.words,s.p - 32 * (s.words - 1),i,mask);
        } else if constexpr (Input == SegmentInput::Linear) {
            value = linear_fold_limb(products + size_t(blockIdx.y) * s.length, s, i, mask);
        } else {
            const unsigned swap = Select ? 2 * scalar_bit(cursor) : 0;
            value = add_limb(state,s,ops.left.x[op] ^ swap,ops.right.x[op] ^ swap,
                             curve,i,mask,(ops.subtract >> op) & 1);
        }
    }
    const auto result = stage1_summarize_segment(value,mask,headers + row * gridDim.x);
    if (i < s.words) state[row * s.words + i] = result;
}

template<bool Routed>
__global__ void finish_segment_carries(uint32_t* state, const unsigned* headers, Shape s,
                                      Indices out, Cursor cursor, unsigned deferred_mask) {
    const unsigned op = blockIdx.y / s.curves;
    if ((deferred_mask >> op) & 1) return;
    const unsigned slot = output_slot<Routed>(out.x[op],cursor);
    const size_t row = size_t(slot) * s.curves + blockIdx.y % s.curves;
    __shared__ unsigned limits[stage1_max_segments];
    stage1_segment_corrections(limits,headers + row * gridDim.x,gridDim.x);
    const unsigned i = blockIdx.x * stage1_carry_segment + threadIdx.x;
    if (i < s.words) {
        auto* limbs = state + row * s.words;
        limbs[i] = stage1_segment_word(limbs,limits,i,limb_mask(s,i));
    }
}

struct WeightedLimbSource {
    const WeightedWord *coefficients, *inverse_weights;
    const WeightedDigit* digits;
    const WeightedLimb* spans;
};

template<class Coefficient>
__device__ __forceinline__ WeightedWord gather_carry_limb(const WeightedLimbSource& source,
                                                        unsigned p, unsigned word, Coefficient coefficient) {
    const unsigned bit = 32 * word;
    const auto mask = (WeightedWord(1) << min(32u, p - bit)) - 1;
    const auto span = source.spans[word];
    // A coefficient starts below p and spans at most 64 bits. Only the
    // first two limbs can receive fragments after wrapping by p.
    const unsigned wraps = word >= 2 ? 1 : p < 64 ? 3 : 2;
    WeightedWord sum = 0;
    for (unsigned wrap = 0; wrap < wraps; ++wrap) {
        const unsigned lo = bit + wrap * p;
        for (unsigned j = span.first[wrap]; j < span.end[wrap]; ++j) {
            sum += weighted_limb_fragment(coefficient(j), source.digits[j].bit, lo, mask);
        }
    }
    return sum;
}

inline unsigned weighted_segment_cache_bytes(const std::vector<WeightedLimb>& spans) {
    unsigned capacity = 0;
    for (size_t start = 0; start < spans.size(); start += stage1_carry_segment) {
        const auto& first = spans[start ? start - 1 : 0]; // Include the incoming carry's limb.
        const auto& last = spans[std::min(spans.size(), start + stage1_carry_segment) - 1];
        capacity = std::max(capacity, last.end[0] - first.first[0]);
    }
    return capacity * sizeof(WeightedWord);
}

template<bool Routed, bool Cached = false>
__global__ __launch_bounds__(stage1_carry_segment, 2)
void weighted_segment_limb_sums(uint32_t* state, WeightedLimbSource source, unsigned* headers,
                                Shape s, Indices out, Cursor cursor) {
    source.coefficients += size_t(blockIdx.y) * s.length;
    const unsigned start = blockIdx.x * stage1_carry_segment;
    const unsigned i = start + threadIdx.x;
    const unsigned lane = threadIdx.x % 32;
    const auto load = [&](unsigned digit) { return gl_mul(source.coefficients[digit], source.inverse_weights[digit]); };
    extern __shared__ WeightedWord coefficients[];
    unsigned first = 0;
    if constexpr (Cached) {
        first = source.spans[start ? start - 1 : 0].first[0];
        const unsigned end = source.spans[min(start + stage1_carry_segment, s.words) - 1].end[0];
        for (unsigned j = first + threadIdx.x; j < end; j += blockDim.x)
            coefficients[j - first] = load(j);
        __syncthreads();
    }
    const auto gather = [&](unsigned word) {
        if constexpr (Cached) {
            if (word >= 2)
                return gather_carry_limb(source, s.p, word, [&](unsigned digit) { return coefficients[digit - first]; });
        }
        return gather_carry_limb(source, s.p, word, load);
    };
    const auto sum = i < s.words ? gather(i) : 0;
    auto previous = __shfl_up_sync(0xffffffffu, uint32_t(sum >> 32), 1);
    const uint32_t mask = i < s.words ? limb_mask(s,i) : 0;
    WeightedWord value = 0;
    if (i < s.words) {
        const unsigned last = s.words - 1;
        if (i == 0) {
            // This is the only limb that needs the opposite end of the residue.
            const unsigned tail_bits = s.p - 32 * last;
            const auto tail_mask = (WeightedWord(1) << tail_bits) - 1;
            const auto last_sum = last ? gather_carry_limb(source,s.p,last,load) : sum;
            const auto tail = (last_sum & tail_mask) + (last ?
                gather_carry_limb(source,s.p,last - 1,load) >> 32 : last_sum >> tail_bits);
            value = (last ? (sum & mask) + (last_sum >> tail_bits) : tail & tail_mask) + (tail >> tail_bits);
        } else {
            if (lane == 0) previous = uint32_t(gather(i - 1) >> 32);
            value = (sum & mask) + previous;
            if (i == last) value &= mask;
        }
    }
    const unsigned slot = output_slot<Routed>(out.x[blockIdx.y / s.curves],cursor);
    const size_t row = size_t(slot) * s.curves + blockIdx.y % s.curves;
    const auto result = stage1_summarize_segment(value,mask,headers + row * gridDim.x);
    if (i < s.words) state[row * s.words + i] = result;
}

} // namespace ecm_mersenne_ntt::stage1_layout
