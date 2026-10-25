#include "files.h"
#include "ecm/platform/logging.h"
#include <chrono>

namespace ecm_sync {
void backup_saves(const std::string& dir, const std::string& first, const std::string& second,
                  bool full, long long since) {
    if (first.empty() && second.empty())
        return;
    try {
        for (const auto& entry :
             std::filesystem::directory_iterator(std::filesystem::u8path(dir))) {
            if (!entry.is_regular_file() || entry.path().extension() != ".save")
                continue;
            // C++17 filesystem and system clocks can have different epochs.
            const auto modified = std::chrono::time_point_cast<std::chrono::system_clock::duration>(
                entry.last_write_time() - std::filesystem::file_time_type::clock::now() +
                std::chrono::system_clock::now());
            if (!full && std::chrono::system_clock::to_time_t(modified) < since)
                continue;
            const auto bytes = read_file(entry.path());
            for (const auto& destination : {first, second}) {
                if (destination.empty())
                    continue;

                try {
                    const auto target =
                        std::filesystem::u8path(destination) / entry.path().filename();
                    if (std::filesystem::absolute(target).lexically_normal() ==
                        std::filesystem::absolute(entry.path()).lexically_normal())
                        continue;

                    std::filesystem::create_directories(target.parent_path());
                    FileLock lock(target.u8string());
                    write_atomic(target, bytes);
                } catch (const std::exception& e) {
                    ecm_ts_fprintf(stderr, "WARNING: save backup: %s\n", e.what());
                }
            }
        }
    } catch (const std::exception& e) {
        ecm_ts_fprintf(stderr, "WARNING: save backup: %s\n", e.what());
    }
}
} // namespace ecm_sync
