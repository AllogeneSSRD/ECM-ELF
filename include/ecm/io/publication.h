#pragma once
#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>

namespace ecm_results {
// The owning task persists this cursor and the exact batch before writing output.
struct AppendCheckpoint {
    bool pending = false;
    uint64_t offset = 0;
    std::string anchor;
};

void append_checkpointed(const std::filesystem::path&, const std::string& batch,
                         AppendCheckpoint&, const std::function<void()>& save,
                         const std::function<void()>& acknowledge, bool adopt = false);
} // namespace ecm_results
