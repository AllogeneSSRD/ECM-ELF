#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

namespace ecm_io {
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

} // namespace ecm_io

// Existing checkpoint/publication callers use the same bounded replacement.
bool ecm_replace_file(const std::filesystem::path&, const std::filesystem::path&);

namespace ecm_io {
void set_work_dir(const char*);
const char* work_dir();
std::string resolve_data_path(const char*);
bool ensure_parent_dir(const char*);
} // namespace ecm_io

namespace ecm_queue {
inline constexpr const char* owner_protocol = "ECM_QUEUE_OWNER=1\n";

// A process-lifetime worker lease, distinct from FileLock's short publication lock.
class OwnerLease {
    struct State;
    State* state_ = nullptr;

public:
    OwnerLease() = default;
    OwnerLease(const std::string&, int worker, const char* suffix = ".owner.lock");
    ~OwnerLease();
    OwnerLease(const OwnerLease&) = delete;
    OwnerLease& operator=(const OwnerLease&) = delete;
    void acquire(const std::string&, int worker, const char* suffix = ".owner.lock");
    void release() noexcept;
};
} // namespace ecm_queue
