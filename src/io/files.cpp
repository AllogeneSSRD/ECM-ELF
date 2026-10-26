#include <cstring>
#include <memory>
#ifdef _WIN32
#include <direct.h>
#include <io.h>
#else
#include <sys/stat.h>
#endif
#include "io/files.h"
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

namespace ecm_io {
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

std::string file_identity(const std::filesystem::path& name) {
    std::error_code error;
    // Resolve existing directory aliases before removing '..': a symlink's
    // parent may differ from the lexical parent of the spelling supplied.
    const auto absolute = std::filesystem::absolute(name);
    const auto status = std::filesystem::symlink_status(absolute, error);
    if (error && error != std::errc::no_such_file_or_directory)
        throw std::runtime_error("cannot inspect writer identity: " + name.u8string() + ": " +
                                 error.message());
    if (std::filesystem::is_symlink(status))
        throw std::runtime_error("writer target is a symbolic link; use its actual path: " +
                                 name.u8string());
    if (std::filesystem::is_regular_file(status)) {
        const auto links = std::filesystem::hard_link_count(absolute, error);
        if (error || links > 1)
            throw std::runtime_error("cannot establish unique writer identity (hard link or "
                                     "inaccessible attributes): " +
                                     name.u8string());
    }
    return std::filesystem::weakly_canonical(absolute).u8string();
}

bool file_identity_less(const std::string& first, const std::string& second) {
#ifdef _WIN32
    // A missing leaf retains its original spelling in weakly_canonical.
    // Use the same case-insensitive order for fresh and existing resources.
    const auto a = std::filesystem::u8path(first).wstring();
    const auto b = std::filesystem::u8path(second).wstring();
    const auto compared =
        CompareStringOrdinal(a.c_str(), int(a.size()), b.c_str(), int(b.size()), TRUE);
    if (!compared)
        throw std::runtime_error("cannot compare writer identities");
    return compared == CSTR_LESS_THAN;
#else
    return first < second;
#endif
}

bool same_file_resource(const std::filesystem::path& first, const std::filesystem::path& second) {
    if (first.empty() || second.empty())
        return false;
    const auto a = file_identity(first), b = file_identity(second);
    return !file_identity_less(a, b) && !file_identity_less(b, a);
}

FileLock::FileLock(const std::string& name, FileLockDomain domain, uint32_t wait_ms) {
    auto state = std::make_unique<State>();
    const auto resource = file_identity(std::filesystem::u8path(name));
    const auto path = std::filesystem::u8path(
        resource + (domain == FileLockDomain::QueueEdit ? ".edit.lock" : ".publish.lock"));
    const auto begin = std::chrono::steady_clock::now();
    const auto deadline = begin + std::chrono::milliseconds(wait_ms);
    unsigned long system_error = 0;
    for (;;) {
#ifdef _WIN32
        state->handle =
            CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_ALWAYS,
                        FILE_ATTRIBUTE_NORMAL | FILE_FLAG_DELETE_ON_CLOSE, nullptr);
        if (state->handle != INVALID_HANDLE_VALUE) {
            state_ = state.release();
            return;
        }
        system_error = GetLastError();
        const bool busy =
            system_error == ERROR_SHARING_VIOLATION || system_error == ERROR_LOCK_VIOLATION;
        // A zero-access metadata observer can keep a DELETE_ON_CLOSE marker
        // delete-pending after its writer closes. Windows reports ACCESS_DENIED
        // until that observer closes; retry within the same bounded budget.
        const bool retryable = busy || system_error == ERROR_ACCESS_DENIED;
#else
        state->fd = ::open(path.c_str(), O_CREAT | O_RDWR, 0600);
        if (state->fd >= 0 && ::flock(state->fd, LOCK_EX | LOCK_NB) == 0) {
            state_ = state.release();
            return;
        }
        system_error = errno;
        const bool busy =
            system_error == EWOULDBLOCK || system_error == EAGAIN || system_error == EINTR;
        const bool retryable = busy;
        if (state->fd >= 0)
            ::close(state->fd);
#endif
        const auto now = std::chrono::steady_clock::now();
        if (!retryable || now >= deadline) {
            const auto waited =
                std::chrono::duration_cast<std::chrono::milliseconds>(now - begin).count();
            throw std::runtime_error(
                std::string(busy ? "writer lock busy: " : "cannot open writer lock: ") + resource +
                "; waited_ms=" + std::to_string(waited) +
                "; system_error=" + std::to_string(system_error));
        }
        std::this_thread::sleep_for(
            (std::min)(std::chrono::milliseconds(50),
                       std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now)));
    }
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
    if (!ok || !ecm_replace_file(temp, path))
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
    if (!ok || !ecm_replace_file(temp, path))
        throw std::runtime_error("cannot flush/publish file: " + path.string());
    // POSIX durability also requires persisting the directory entry after rename.
    const auto parent =
        path.parent_path().empty() ? std::filesystem::path(".") : path.parent_path();
    const int dir = ::open(parent.c_str(), O_RDONLY | O_DIRECTORY);
    if (dir < 0)
        throw PublicationUncertain("cannot open publication directory after replacement: " +
                                   parent.string());
    const bool durable = ::fsync(dir) == 0;
    ::close(dir);
    if (!durable)
        throw PublicationUncertain("cannot flush publication directory after replacement: " +
                                   parent.string());
#endif
}
} // namespace ecm_io

// A scanner or reader without delete sharing can briefly block atomic rename.
// Bound retries; permanent access errors must still preserve the pending file.
bool ecm_replace_file(const std::filesystem::path& source, const std::filesystem::path& target) {
#ifdef _WIN32
    const auto deadline = GetTickCount64() + 500;
    for (;;) {
        if (MoveFileExW(source.c_str(), target.c_str(),
                        MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
            return true;
        const auto error = GetLastError();
        if ((error != ERROR_ACCESS_DENIED && error != ERROR_SHARING_VIOLATION &&
             error != ERROR_LOCK_VIOLATION) ||
            GetTickCount64() >= deadline) {
            SetLastError(error);
            return false;
        }
        Sleep(5);
    }
#else
    std::error_code error;
    std::filesystem::rename(source, target, error);
    return !error;
#endif
}

namespace {

std::string g_work_dir;

bool is_absolute_path(const char* path) {
    if (path == nullptr || path[0] == '\0') {
        return false;
    }
#ifdef _WIN32
    if ((path[0] >= 'A' && path[0] <= 'Z') || (path[0] >= 'a' && path[0] <= 'z')) {
        return path[1] == ':' && (path[2] == '\\' || path[2] == '/');
    }
#endif
    return path[0] == '/';
}

void trim_trailing_slashes(std::string& s) {
    while (!s.empty() && (s.back() == '/' || s.back() == '\\')) {
        s.pop_back();
    }
}

} // namespace

void ecm_io::set_work_dir(const char* dir) {
    if (dir == nullptr || dir[0] == '\0') {
        g_work_dir.clear();
        return;
    }
    g_work_dir = dir;
    trim_trailing_slashes(g_work_dir);
}

const char* ecm_io::work_dir() { return g_work_dir.c_str(); }

std::string ecm_io::resolve_data_path(const char* path) {
    if (path == nullptr || path[0] == '\0') {
        return {};
    }
    if (is_absolute_path(path)) {
        return path;
    }
    if (g_work_dir.empty()) {
        return path;
    }
    return g_work_dir + "/" + path;
}

bool ecm_io::ensure_parent_dir(const char* filepath) {
    if (filepath == nullptr || filepath[0] == '\0') {
        return false;
    }
    std::string parent;
    const char* slash = std::strrchr(filepath, '/');
#ifdef _WIN32
    const char* bslash = std::strrchr(filepath, '\\');
    if (bslash != nullptr && (slash == nullptr || bslash > slash)) {
        slash = bslash;
    }
#endif
    if (slash == nullptr || slash == filepath) {
        return true;
    }
    parent.assign(filepath, slash);
    if (parent.empty()) {
        return true;
    }

    std::string built;
    size_t i = 0;
    if (parent.size() >= 2 && parent[1] == ':') {
        built = parent.substr(0, 2);
        i = 2;
    } else if (!parent.empty() && parent[0] == '/') {
        built = "/";
        i = 1;
    }

    while (i < parent.size()) {
        while (i < parent.size() && (parent[i] == '/' || parent[i] == '\\')) {
            ++i;
        }
        if (i >= parent.size()) {
            break;
        }
        size_t j = i;
        while (j < parent.size() && parent[j] != '/' && parent[j] != '\\') {
            ++j;
        }
        if (!built.empty() && built.back() != '/') {
            built.push_back('/');
        }
        built.append(parent, i, j - i);
#ifdef _WIN32
        _mkdir(built.c_str());
#else
        mkdir(built.c_str(), 0755);
#endif
        i = j;
    }
    return true;
}

namespace ecm_queue {
struct OwnerLease::State {
#ifdef _WIN32
    HANDLE handle = INVALID_HANDLE_VALUE;
#else
    int fd = -1;
#endif
};

OwnerLease::OwnerLease(const std::string& queue, int worker, const char* suffix) {
    acquire(queue, worker, suffix);
}

OwnerLease::~OwnerLease() { release(); }

void OwnerLease::acquire(const std::string& queue, int worker, const char* suffix) {
    auto incoming = std::make_unique<State>();
    auto* state = incoming.get();
    if (state_)
        throw std::runtime_error("queue lease already held");
    if (worker < 0 || worker > 1000000)
        throw std::runtime_error("invalid queue worker");
    auto path = std::filesystem::weakly_canonical(
        std::filesystem::absolute(std::filesystem::u8path(queue)));
    if (path.empty() || !std::filesystem::is_directory(path.parent_path()))
        throw std::runtime_error("queue directory does not exist");
#ifdef _WIN32
    // Hard links do not provide a stable name across atomic queue rewrites.
    HANDLE input = CreateFileW(path.c_str(), FILE_READ_ATTRIBUTES,
                               FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                               OPEN_EXISTING, 0, nullptr);
    if (input != INVALID_HANDLE_VALUE) {
        BY_HANDLE_FILE_INFORMATION info{};
        const bool ambiguous = !GetFileInformationByHandle(input, &info) || info.nNumberOfLinks > 1;
        CloseHandle(input);
        if (ambiguous)
            throw std::runtime_error("cannot establish unique queue identity (hard "
                                     "link or inaccessible attributes)");
    }
    path += suffix;
    state->handle =
        CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE, FILE_SHARE_READ | FILE_SHARE_WRITE,
                    nullptr, OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (state->handle != INVALID_HANDLE_VALUE) {
        OVERLAPPED gate{};
        const DWORD flags = LOCKFILE_FAIL_IMMEDIATELY | (worker == 0 ? LOCKFILE_EXCLUSIVE_LOCK : 0);
        if (LockFileEx(state->handle, flags, 0, 1, 0, &gate)) {
            if (worker == 0) {
                state_ = incoming.release();
                return;
            }
            OVERLAPPED section{};
            section.Offset = static_cast<DWORD>(worker);
            if (LockFileEx(state->handle, LOCKFILE_EXCLUSIVE_LOCK | LOCKFILE_FAIL_IMMEDIATELY, 0, 1,
                           0, &section)) {
                state_ = incoming.release();
                return;
            }
        }
        CloseHandle(state->handle);
        state->handle = INVALID_HANDLE_VALUE;
    }
#else
    path += suffix;
    state->fd = ::open(path.c_str(), O_CREAT | O_RDWR, 0600);
    if (state->fd >= 0) {
        struct flock gate{};
        gate.l_type = worker ? F_RDLCK : F_WRLCK;
        gate.l_whence = SEEK_SET;
        gate.l_len = 1;
        if (::fcntl(state->fd, F_SETLK, &gate) == 0) {
            if (!worker) {
                state_ = incoming.release();
                return;
            }
            gate.l_type = F_WRLCK;
            gate.l_start = worker;
            if (::fcntl(state->fd, F_SETLK, &gate) == 0) {
                state_ = incoming.release();
                return;
            }
        }
        ::close(state->fd);
        state->fd = -1;
    }
#endif
    throw std::runtime_error("queue worker already owned or ownership lock inaccessible: " + queue +
                             " worker=" + std::to_string(worker));
}

void OwnerLease::release() noexcept {
    if (!state_)
        return;
#ifdef _WIN32
    if (state_->handle != INVALID_HANDLE_VALUE)
        CloseHandle(state_->handle);
#else
    if (state_->fd >= 0)
        ::close(state_->fd);
#endif
    delete state_;
    state_ = nullptr;
}
} // namespace ecm_queue

namespace ecm_io {
bool ensure_dir(const std::string &dir) {
    if (dir.empty()) return true;
#ifdef _WIN32
    if (_mkdir(dir.c_str()) == 0) return true;
#else
    if (mkdir(dir.c_str(), 0777) == 0) return true;
#endif
    // Preserve the legacy existence-only fallback, including existing non-directories.
#ifdef _WIN32
    return _access(dir.c_str(), 0) == 0;
#else
    return access(dir.c_str(), 0) == 0;
#endif
}

}
