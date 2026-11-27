#pragma once
#include "ladder.h"
#include <cuda_runtime_api.h>

namespace ecm_mersenne_ntt {
using Pm1Advance = std::function<void(unsigned pattern, unsigned bits)>;
using Pm1Probe = std::function<void(cudaStream_t, const Pm1Advance&)>;
Run pm1_probe_lane(int device, unsigned p, const ecm_stage1::Stage1NttOptions& options,
                   Plan& plan, const Pm1Probe& probe);
}
