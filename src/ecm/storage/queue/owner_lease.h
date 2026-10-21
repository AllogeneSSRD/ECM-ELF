#pragma once
// Process-lifetime queue ownership, separate from the short queue edit lock.
#include <filesystem>
#include <stdexcept>
#include <string>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>
#endif

namespace ecm_queue {
inline constexpr const char* owner_protocol = "ECM_QUEUE_OWNER=1\n";
class OwnerLease {
public:
    OwnerLease() = default;
    OwnerLease(const std::string& queue, int worker, const char* suffix = ".owner.lock") { acquire(queue, worker, suffix); }
    OwnerLease(const OwnerLease&) = delete;
    OwnerLease& operator=(const OwnerLease&) = delete;
    ~OwnerLease() { release(); }
    void acquire(const std::string& queue, int worker, const char* suffix = ".owner.lock") {
        if (held_) throw std::runtime_error("queue lease already held");
        if (worker < 0 || worker > 1000000) throw std::runtime_error("invalid queue worker");
        auto path = std::filesystem::weakly_canonical(std::filesystem::absolute(std::filesystem::u8path(queue)));
        if (path.empty() || !std::filesystem::is_directory(path.parent_path())) throw std::runtime_error("queue directory does not exist");
#ifdef _WIN32
        // Hard links do not provide a stable name across atomic queue rewrites.
        HANDLE input = CreateFileW(path.c_str(), FILE_READ_ATTRIBUTES, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING, 0, nullptr);
        if (input != INVALID_HANDLE_VALUE) {
            BY_HANDLE_FILE_INFORMATION info{};
            const bool ambiguous = !GetFileInformationByHandle(input, &info) || info.nNumberOfLinks > 1;
            CloseHandle(input);
            if (ambiguous) throw std::runtime_error("cannot establish unique queue identity (hard link or inaccessible attributes)");
        }
        path += suffix;
        handle_ = CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (handle_ != INVALID_HANDLE_VALUE) {
            OVERLAPPED gate{};
            const DWORD flags = LOCKFILE_FAIL_IMMEDIATELY | (worker == 0 ? LOCKFILE_EXCLUSIVE_LOCK : 0);
            if (LockFileEx(handle_, flags, 0, 1, 0, &gate)) {
                if (worker == 0) { held_ = true; return; }
                OVERLAPPED section{}; section.Offset = static_cast<DWORD>(worker);
                if (LockFileEx(handle_, LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY, 0, 1, 0, &section)) { held_ = true; return; }
            }
            CloseHandle(handle_); handle_ = INVALID_HANDLE_VALUE;
        }
#else
        path += suffix;
        fd_ = ::open(path.c_str(), O_CREAT | O_RDWR, 0600);
        if (fd_ >= 0) {
            struct flock gate{}; gate.l_type = worker ? F_RDLCK : F_WRLCK; gate.l_whence = SEEK_SET; gate.l_len = 1;
            if (::fcntl(fd_, F_SETLK, &gate) == 0) {
                if (!worker) { held_ = true; return; }
                gate.l_type = F_WRLCK; gate.l_start = worker;
                if (::fcntl(fd_, F_SETLK, &gate) == 0) { held_ = true; return; }
            }
            ::close(fd_); fd_ = -1;
        }
#endif
        throw std::runtime_error("queue worker already owned or ownership lock inaccessible: " + queue + " worker=" + std::to_string(worker));
    }
    void release() noexcept {
#ifdef _WIN32
        if (handle_ != INVALID_HANDLE_VALUE) CloseHandle(handle_);
        handle_ = INVALID_HANDLE_VALUE;
#else
        if (fd_ >= 0) ::close(fd_);
        fd_ = -1;
#endif
        held_ = false;
    }
private:
    bool held_ = false;
#ifdef _WIN32
    HANDLE handle_ = INVALID_HANDLE_VALUE;
#else
    int fd_ = -1;
#endif
};
}
