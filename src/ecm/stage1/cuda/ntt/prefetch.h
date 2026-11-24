#pragma once
#include "../../execution.h"
#include "ecm/integer.h"
#include <future>
#include <memory>

namespace ecm_cuda_stage1 {
struct PreparedCurves {
    PreparedCurves(size_t count, unsigned words)
        : factors(count), kinds(count), initial(6 * count * words), seeds(3 * count * words) {}
    ecm_edwards::Array factors;
    std::vector<ecm_stage1::CurveKind> kinds;
    std::vector<uint32_t> initial, seeds;
};

// Scoped to one immutable queue task. Never persisted in a checkpoint.
struct BatchPlanCache {
    unsigned p = 0, batch = 0, window = 0;
    ecm_stage1::Stage1NttOptions options;
    bool reused = false;
};

// Owned by one task. No speculative data is serialized or published.
struct CurvePrefetch {
    std::vector<uint64_t> predicted, next;
    std::future<std::unique_ptr<PreparedCurves>> pending;
};
}
