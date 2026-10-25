#include "files.h"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <thread>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace ecm_sync {
uint64_t fingerprint(const std::string& bytes) {
    uint64_t value = 14695981039346656037ull;
    for (unsigned char c : bytes) {
        value ^= c;
        value *= 1099511628211ull;
    }
    return value;
}

struct FileLock::State {
#ifdef _WIN32
    HANDLE handle = INVALID_HANDLE_VALUE;
#else
    int fd = -1;
#endif
};

FileLock::FileLock(const std::string& name) {
    auto state = new State;
    const auto path = std::filesystem::u8path(name + ".publish.lock");
    for (int attempt = 0; attempt < 100; ++attempt) {
#ifdef _WIN32
        state->handle = CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr,
                                    OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
        if (state->handle != INVALID_HANDLE_VALUE) {
            state_ = state;
            return;
        }
#else
        state->fd = ::open(path.c_str(), O_CREAT | O_RDWR, 0600);
        if (state->fd >= 0 && ::flock(state->fd, LOCK_EX | LOCK_NB) == 0) {
            state_ = state;
            return;
        }
        if (state->fd >= 0)
            ::close(state->fd);
#endif
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    delete state;
    throw std::runtime_error("cannot acquire publication lock: " + name);
}

FileLock::~FileLock() {
    if (!state_)
        return;
#ifdef _WIN32
    CloseHandle(state_->handle);
#else
    ::flock(state_->fd, LOCK_UN);
    ::close(state_->fd);
#endif
    delete state_;
}

std::string read_file(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in)
        throw std::runtime_error("cannot read file: " + path.u8string());
    std::ostringstream out;
    out << in.rdbuf();
    if (in.bad())
        throw std::runtime_error("incomplete read: " + path.u8string());
    return out.str();
}

void write_atomic(const std::filesystem::path& path, const std::string& bytes,
                  const std::string& suffix) {
    if (!path.parent_path().empty())
        std::filesystem::create_directories(path.parent_path());
    auto temp = path;
    temp += suffix;
#ifdef _WIN32
    HANDLE f = CreateFileW(temp.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                           FILE_ATTRIBUTE_NORMAL, nullptr);
    if (f == INVALID_HANDLE_VALUE)
        throw std::runtime_error("cannot create temporary file: " + temp.u8string());
    bool ok = true;
    size_t at = 0;
    while (at < bytes.size()) {
        DWORD written = 0;
        const DWORD size =
            static_cast<DWORD>((std::min)(bytes.size() - at, size_t(16 * 1024 * 1024)));
        if (!WriteFile(f, bytes.data() + at, size, &written, nullptr) || written != size) {
            ok = false;
            break;
        }
        at += written;
    }
    // A visible destination must already contain a complete, flushed snapshot.
    ok = ok && FlushFileBuffers(f);
    CloseHandle(f);
    if (!ok || !MoveFileExW(temp.c_str(), path.c_str(),
                            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        throw std::runtime_error("cannot flush/publish file: " + path.u8string());
#else
    const int fd = ::open(temp.c_str(), O_CREAT | O_TRUNC | O_WRONLY, 0600);
    if (fd < 0)
        throw std::runtime_error("cannot create temporary file: " + temp.string());
    size_t at = 0;
    bool ok = true;
    while (at < bytes.size()) {
        const auto n = ::write(fd, bytes.data() + at, bytes.size() - at);
        if (n <= 0) {
            ok = false;
            break;
        }
        at += size_t(n);
    }
    ok = ok && ::fsync(fd) == 0;
    ::close(fd);
    if (!ok || ::rename(temp.c_str(), path.c_str()))
        throw std::runtime_error("cannot flush/publish file: " + path.string());
    // POSIX durability also requires persisting the directory entry after rename.
    const auto parent =
        path.parent_path().empty() ? std::filesystem::path(".") : path.parent_path();
    const int dir = ::open(parent.c_str(), O_RDONLY | O_DIRECTORY);
    if (dir < 0)
        throw std::runtime_error("cannot open publication directory: " + parent.string());
    const bool durable = ::fsync(dir) == 0;
    ::close(dir);
    if (!durable)
        throw std::runtime_error("cannot flush publication directory: " + parent.string());
#endif
}
} // namespace ecm_sync
