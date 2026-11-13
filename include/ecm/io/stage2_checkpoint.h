#pragma once
#include <cstdint>
#include <cstddef>
#include <filesystem>
#include <functional>
#include <set>
#include <string>
#include <vector>

namespace stage2_queue {
namespace fs = std::filesystem;
struct Task;

struct Reservation {
    uint64_t position = 0;
    bool completed = false;
    std::string receipt, results;
};

struct State {
    std::string identity, run, pending, pending_results;
    uint64_t done = 0, total = 0;
    unsigned version = 5;
    std::string publication_task;
    uint64_t b2 = 0;
    bool auto_b2 = false;
    std::vector<Reservation> reserved;
    std::string submission_path, latest_nf, final_nf;
    uint64_t latest_nf_position = 0;
    std::set<std::string> nf_keys;
    std::vector<std::string> factors;
    bool load(const fs::path& path);
    void save(const fs::path& path) const;
    void reconfigure(const Task& task);
    void advance_completed();
    void complete(Reservation& entry, const std::string& outcome);
    void record_outcome(uint64_t position, const std::string& report, std::string key);
};

struct Task {
    std::string aid, n, save_sha256;
    uint64_t skip = 0, total = 0, b2 = 0;
    bool auto_b2 = false;
    uint64_t b1 = 0;
    std::string id() const;
    std::string legacy_id() const;
    fs::path path(const fs::path& directory, size_t bits, unsigned suffix = 0) const;
    fs::path legacy_path(const fs::path& directory, size_t bits) const;
};

// Return the source checkpoint, or an empty path for a new task. Legacy B2
// candidates are loaded lazily only when a V2 identity cannot match directly.
fs::path load_task(const fs::path&, size_t, const Task&, State&,
                   const std::function<std::vector<uint64_t>()>& legacy_bounds = {},
                   fs::path* destination = nullptr);
std::string token();
bool result_written(const fs::path& path, const std::string& receipt);
bool finished_written(const fs::path& path, const std::string& run, const std::string& task);
} // namespace stage2_queue

namespace stage2_queue {
bool import_legacy(const fs::path&, const Task&, const std::function<bool(const std::string&)>&,
                   State&, const std::function<std::vector<uint64_t>()>& legacy_bounds = {});
}
