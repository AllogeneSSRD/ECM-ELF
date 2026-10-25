#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

namespace ecm_sync {
// A short cross-process publication lock. Never hold it across a GPU launch.
class FileLock {
    struct State;
    State* state_ = nullptr;

public:
    explicit FileLock(const std::string& path);
    ~FileLock();
    FileLock(const FileLock&) = delete;
    FileLock& operator=(const FileLock&) = delete;
};

std::string read_file(const std::filesystem::path& path);

// Flush the temporary file before replacing the destination. Concurrent writers
// must hold FileLock for this path so they cannot share its temporary filename.
void write_atomic(const std::filesystem::path& path, const std::string& bytes,
                  const std::string& temporary_suffix = ".publish.tmp");
uint64_t fingerprint(const std::string& bytes);

// Existing backup destinations are independent from P95 handoff readiness.
void backup_saves(const std::string& dir, const std::string& first, const std::string& second,
                  bool full, long long since_epoch_seconds);
} // namespace ecm_sync
