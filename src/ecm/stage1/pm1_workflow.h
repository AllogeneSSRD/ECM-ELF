#pragma once

#include "ecm/stage1/task.h"
#include "io/worktodo.h"
#include <filesystem>

namespace ecm_pm1 {
struct Outcome {
    bool completed = false;
    std::string factor;  // A proper divisor of the target cofactor, in decimal.
    bool whole_target = false;
    uint64_t save_hash = 0;
};
// Read only the contiguous Pminus1 prefix starting at the selected queue row.
std::vector<Pm1Task> batch_prefix(const std::string& path, size_t source_line, Pm1Task first,
                                  const ecm_stage1::Stage1NttOptions&, bool shared_prime95_dir);

Outcome run(const Pm1Task&, uint64_t b2, const ecm_stage1::Stage1Options&,
            const std::filesystem::path& owl_root, const std::filesystem::path& prime95_dir);
}
