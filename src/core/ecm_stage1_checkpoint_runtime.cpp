#include "ecm_stage1_checkpoint_runtime.h"

#include <atomic>
#include <csignal>
#include <filesystem>
#include <mutex>
#include <system_error>
#include <unordered_set>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#endif

namespace {
std::atomic<bool> stop_requested{false};
std::atomic<unsigned> backup_limit{3};
std::mutex bad_primary_mutex;
std::unordered_set<std::string> bad_primaries;
#ifdef _WIN32
std::atomic<unsigned> console_stop_count{0};
HANDLE stop_event = nullptr; // borrowed inherited handle; GUI owns its lifetime
BOOL WINAPI console_control(DWORD event) {
    if (event != CTRL_C_EVENT && event != CTRL_BREAK_EVENT) return FALSE;
    if (console_stop_count.fetch_add(1, std::memory_order_relaxed) > 0) return FALSE;
    stop_requested.store(true, std::memory_order_relaxed);
    return TRUE;
}
#else
void signal_control(int) { stop_requested.store(true, std::memory_order_relaxed); }
#endif

bool replace_file(const std::string &source, const std::string &target) {
#ifdef _WIN32
    return MoveFileExA(source.c_str(), target.c_str(),
                       MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
    std::error_code ec;
    std::filesystem::rename(source, target, ec);
    return !ec;
#endif
}
} // namespace

void ecm_stage1_install_stop_handler() {
#ifdef _WIN32
    SetConsoleCtrlHandler(console_control, TRUE);
#else
    std::signal(SIGINT, signal_control);
    std::signal(SIGTERM, signal_control);
#endif
}

bool ecm_stage1_set_stop_handle(uintptr_t value) {
#ifdef _WIN32
    HANDLE candidate = reinterpret_cast<HANDLE>(value);
    DWORD flags = 0;
    if (!candidate || !GetHandleInformation(candidate, &flags)) return false;
    stop_event = candidate;
    return true;
#else
    (void)value;
    return false;
#endif
}

bool ecm_stage1_stop_requested() {
    if (stop_requested.load(std::memory_order_relaxed)) return true;
#ifdef _WIN32
    if (stop_event && WaitForSingleObject(stop_event, 0) == WAIT_OBJECT_0) {
        stop_requested.store(true, std::memory_order_relaxed);
        return true;
    }
#endif
    return false;
}

void ecm_stage1_set_backup_limit(unsigned count) {
    backup_limit.store(count <= ECM_STAGE1_MAX_BACKUPS ? count : ECM_STAGE1_MAX_BACKUPS,
                       std::memory_order_relaxed);
}
unsigned ecm_stage1_backup_limit() { return backup_limit.load(std::memory_order_relaxed); }

std::string ecm_stage1_backup_path(const std::string &path, unsigned generation) {
    const auto slash = path.find_last_of("/\\");
    const auto dot = path.find_last_of('.');
    const auto at = dot != std::string::npos && (slash == std::string::npos || dot > slash)
                        ? dot : path.size();
    return path.substr(0, at) + ".bak" + (generation == 1 ? "" : std::to_string(generation)) +
           path.substr(at);
}

std::vector<std::string> ecm_stage1_checkpoint_candidates(const std::string &path) {
    std::vector<std::string> result{path};
    for (unsigned i = 1; i <= ecm_stage1_backup_limit(); ++i)
        result.push_back(ecm_stage1_backup_path(path, i));
    return result;
}

std::string ecm_stage1_checkpoint_temp_path(const std::string &path) { return path + ".new"; }

void ecm_stage1_mark_bad_primary(const std::string &path) {
    std::lock_guard<std::mutex> lock(bad_primary_mutex);
    bad_primaries.insert(path);
}

bool ecm_stage1_commit_checkpoint(const std::string &path, const std::string &temp) {
    namespace fs = std::filesystem;
    std::error_code ec;
    const unsigned limit = ecm_stage1_backup_limit();
    bool bad_primary = false;
    {
        std::lock_guard<std::mutex> lock(bad_primary_mutex);
        bad_primary = bad_primaries.count(path) != 0;
    }
    if (limit && !bad_primary) {
        for (unsigned i = limit; i > 1; --i) {
            const auto from = ecm_stage1_backup_path(path, i - 1);
            if (fs::exists(from, ec) && !replace_file(from, ecm_stage1_backup_path(path, i)))
                return false;
            if (ec) return false;
        }
        if (fs::exists(path, ec)) {
            fs::copy_file(path, ecm_stage1_backup_path(path, 1),
                          fs::copy_options::overwrite_existing, ec);
            if (ec) return false;
        } else if (ec) return false;
    }
    if (!replace_file(temp, path)) return false;
    {
        std::lock_guard<std::mutex> lock(bad_primary_mutex);
        bad_primaries.erase(path);
    }
    // A reduced limit takes effect only after the new primary is safely installed.
    for (unsigned i = limit + 1; i <= ECM_STAGE1_MAX_BACKUPS; ++i)
        fs::remove(ecm_stage1_backup_path(path, i), ec);
    return true;
}

void ecm_stage1_remove_checkpoints(const std::string &path) {
    {
        std::lock_guard<std::mutex> lock(bad_primary_mutex);
        bad_primaries.erase(path);
    }
    std::error_code ec;
    std::filesystem::remove(path, ec);
    std::filesystem::remove(ecm_stage1_checkpoint_temp_path(path), ec);
    for (unsigned i = 1; i <= ECM_STAGE1_MAX_BACKUPS; ++i)
        std::filesystem::remove(ecm_stage1_backup_path(path, i), ec);
}

uint64_t ecm_stage1_checksum(uint64_t sum, const void *data, size_t size) {
    const auto *p = static_cast<const unsigned char *>(data);
    for (size_t i = 0; i < size; ++i) {
        sum ^= p[i];
        sum *= 1099511628211ull;
    }
    return sum;
}
