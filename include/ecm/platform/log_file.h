#pragma once
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <string_view>

// Append complete worker log records under the shared log's process lock.
void ecm_append_worker_log(const std::filesystem::path &path, int worker,
                           uint64_t max_kb, std::string_view message);

// Owned application mirror; logging.h remains the small backend logging API.
class EcmLogMirror {
    FILE *file_=nullptr;
    std::filesystem::path shared_path_;
public:
    EcmLogMirror()=default;
    ~EcmLogMirror();
    bool open(const std::string &path, int worker, uint64_t max_kb = 0);
    EcmLogMirror(const EcmLogMirror &)=delete;
    EcmLogMirror &operator=(const EcmLogMirror &)=delete;
};
