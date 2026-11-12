#pragma once
#include "ecm/stage1/task.h"
#include <functional>
#include <optional>
#include <stdexcept>

namespace ecm_mersenne_ntt {

struct Plan {
    unsigned length, bpw, digits, tile, outer_max, window;
    unsigned tile_threads, tile_smem_bytes, inverse_tile_threads, inverse_tile_smem_bytes, outer_threads, passes;
    bool warp_tail;
    const char* source;
    bool weighted;
    std::string forward, inverse;
    unsigned dataflow = 0;
};

struct BudgetError : std::runtime_error { using std::runtime_error::runtime_error; };
size_t memory_budget(size_t free_bytes, unsigned limit_mib);
struct Packing { unsigned length=0, bpw=0, digits=0; };
Packing resolve_packing(unsigned p, const ecm_stage1::Stage1NttOptions&, bool prefer_radix5 = true);

enum Dataflow : unsigned {
    FusedMultiplyAdd = 1, LooseDifference = 2, DirectRoute = 4, PackedForward = 8, CachedCarry = 16,
    SegmentedCarry = 32, DeferredPack = 64, PfaRadix5 = 128, CoalescedRadix5 = 256,
    InverseWarpInput = 512, ForwardTileInput = 1024, ForwardTileCompact = 2048,
    FusedWeightedCarry = 4096
};
inline constexpr unsigned experimental_dataflow_limit = (FusedWeightedCarry << 1) - 1;

struct Control {
    unsigned resume = 0;
    unsigned warmup_iterations = 0;
    std::function<void(const Plan&)> describe;
    // Poll only when the stream is idle. Read back only when requested.
    struct Action { bool save = false, stop = false; };
    std::function<Action(unsigned)> poll;
    std::function<void(unsigned, const std::vector<uint32_t>&)> checkpoint;
    std::optional<unsigned> experimental_dataflow;
    bool modular_square_probe = false;
};

// Initial slots (slot-major, then curve, then little-endian uint32 limbs):
// X(P), Z(P), X(2P), Z(2P), a24, affine x(P). All canonical mod 2^p-1.
// For resume the first four slots contain the saved ladder pair.
struct Run {
    std::vector<uint32_t> points;
    std::vector<double> device_ms, wall_ms;
    double setup_seconds = 0, readback_seconds = 0;
    unsigned length = 0, digit_bits = 0, passes = 0, tile = 0, completed_bits = 0;
    unsigned dataflow = 0;
    bool stopped = false;
    size_t owned_bytes = 0, observed_bytes = 0, budget_bytes = 0;
};

// Scalar words are little-endian. Process the low bit_count bits from high
// to low, omitting the implicit leading 1.
Run ladder(int device, unsigned p, unsigned curves, const std::vector<uint32_t>& initial,
           const std::vector<uint32_t>& scalar, unsigned bit_count,
           const ecm_stage1::Stage1NttOptions&, const Control&, unsigned samples = 1);

// Standalone probe retains its byte-per-bit input and environment overrides.
Run ladder(int device, unsigned p, unsigned curves, const std::vector<uint32_t>& initial,
           const std::vector<uint8_t>& bits, unsigned window, unsigned samples, bool weighted = false);
} // namespace ecm_mersenne_ntt
