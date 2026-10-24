#pragma once
#include <cstdint>
#include <cstddef>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>
namespace stage2_queue {
namespace fs=std::filesystem;
struct Task;
struct State {
    std::string identity, run, pending, pending_results;
    uint64_t done=0,total=0;
    unsigned version=3;
    std::string publication_task;
    uint64_t b2=0;
    bool auto_b2=false;
    bool load(const fs::path &path);
    void save(const fs::path &path) const;
    void reconfigure(const Task &task);
};
struct Task {
    std::string aid,n,save_sha256;
    uint64_t skip=0,total=0,b2=0;
    bool auto_b2=false;
    std::string id()const;
    std::string legacy_id()const;
    fs::path path(const fs::path &directory,size_t bits)const;
};
// Return the source checkpoint, or an empty path for a new task. Legacy B2
// candidates are loaded lazily only when a V2 identity cannot match directly.
fs::path load_task(const fs::path &,size_t,const Task &,State &,
                  const std::function<std::vector<uint64_t>()> &legacy_bounds={});
std::string token();
bool result_written(const fs::path &path,const std::string &receipt);
bool finished_written(const fs::path &path,const std::string &run,const std::string &task);
}
