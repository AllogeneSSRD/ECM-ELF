#pragma once

#include <cstdint>

namespace ecm_stage1 {

// Validate the complete sequence first + [0, curves), without unsigned wrap.
bool sigma_batch_fits(uint64_t first, uint32_t curves, uint64_t minimum = 0,
                      uint64_t maximum = UINT64_MAX) noexcept;
bool montgomery_sigma_batch_fits(uint64_t first, uint32_t curves, int param) noexcept;

// Historical independent sampler: 6 <= sigma < 2^53. Shares process rand state;
// callers must generate before launching worker threads. Not a full-u64 sampler.
uint64_t random_sigma53();

// Choose a starting value whose last lane still fits uint32_t. PARAM2 uses
// minimum=2; PARAM3 uses minimum=1. Requires curves > 0 and a nonempty range.
uint32_t random_sigma32(uint32_t curves, uint32_t minimum = 1);

// A nonzero saved value must already have passed the caller's checkpoint
// validation. These functions neither read checkpoints nor resolve conflicts.
// Independent curves and contiguous Montgomery batches retain separate policies.
uint64_t select_curve_sigma(uint64_t saved, bool fixed, uint64_t first, uint32_t index);
uint64_t select_batch_sigma(uint64_t saved, bool fixed, uint64_t first,
                          uint32_t curves, int param);

} // namespace ecm_stage1

// Historical exported C ABI, including its curves=0/UINT32_MAX return value.
extern "C" uint32_t gpu_pick_random_sigma(uint32_t curves);
