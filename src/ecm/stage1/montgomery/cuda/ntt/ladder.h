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
    unsigned long long coefficient_bound = 0, field_modulus = 0;
};

struct BudgetError : std::runtime_error { using std::runtime_error::runtime_error; };
size_t memory_budget(size_t free_bytes, unsigned limit_mib);
enum class WeightedBound { Conservative, UnsignedProducts };
struct Packing {
    unsigned length = 0, bpw = 0, digits = 0;
    bool needs_unsigned_bound = false;
};
Packing resolve_packing(unsigned p, const ecm_stage1::Stage1NttOptions&, bool prefer_radix5 = true,
                        WeightedBound = WeightedBound::Conservative);

enum Dataflow : unsigned {
    FusedMultiplyAdd = 1, LooseDifference = 2, DirectRoute = 4, PackedForward = 8, CachedCarry = 16,
    SegmentedCarry = 32, DeferredPack = 64, PfaRadix5 = 128, CoalescedRadix5 = 256,
    InverseWarpInput = 512, ForwardTileInput = 1024, ForwardTileCompact = 2048,
    FusedWeightedCarry = 4096, ShortInverseTile = 8192, CachedFusedWeightedCarry = 16384,
    Edwards2XYFrequency = 32768, Edwards2XYInverseInput = 65536, TightCarrySpans = 131072,
    FixedWideInverse = 262144, FixedWideForward = 524288
};
inline constexpr unsigned experimental_dataflow_limit = (FixedWideForward << 1) - 1;

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
    double dictionary_gpu_ms = 0;
    double dictionary_inverse_ms = 0;
    double dictionary_host_ms = 0, dictionary_device_wall_ms = 0;
    bool dictionary_built_on_gpu = false;
    size_t dictionary_spectra_bytes = 0, dictionary_workspace_bytes = 0;
    size_t dictionary_fallback_host_bytes = 0;
    // Empty for the host dictionary. Otherwise 0=ready, 1=factor, 2=replay.
    std::vector<uint8_t> dictionary_status;
    std::vector<uint32_t> dictionary_factors;
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

// Edwards prototype. State is X/Y/Z in slot-major order. Digits are
// MSB-first. The table holds affine x, y, d*x*y, x+y per entry for host
// preparation or just the affine seed x, y, d for device preparation.
struct EdwardsNaf {
    unsigned window;
    const std::vector<int16_t>& digits;
    const std::vector<uint32_t>& dictionary;
    size_t dictionary_budget = 0;
    bool build_dictionary = false;
    // Convert the seed to a host table only if the automatic GPU builder's
    // temporary storage exceeds a limit. Keep the same lanes for checkpoints.
    std::function<std::vector<uint32_t>(Run&)> host_fallback;
    // Optional divisor of M_p for dictionary inverses and factor classification.
    mpz_srcptr check_modulus = nullptr;
};
Run edwards_naf(int device, unsigned p, unsigned curves, const std::vector<uint32_t>& initial,
                const EdwardsNaf&, const ecm_stage1::Stage1NttOptions&, const Control&, unsigned samples = 1);

// Standalone probe retains its byte-per-bit input and environment overrides.
Run ladder(int device, unsigned p, unsigned curves, const std::vector<uint32_t>& initial,
           const std::vector<uint8_t>& bits, unsigned window, unsigned samples, bool weighted = false);
} // namespace ecm_mersenne_ntt
