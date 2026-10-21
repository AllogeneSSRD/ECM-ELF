#pragma once
#include <cstdint>
#include <cstddef>
#include <filesystem>
#include <string>
namespace stage2_queue {
namespace fs=std::filesystem;
struct State {
    std::string identity, run, pending, pending_results;
    uint64_t done=0,total=0;
    unsigned version=2;
    bool load(const fs::path &path);
    void save(const fs::path &path) const;
};
struct Task {
    std::string aid,n,save_sha256;
    uint64_t skip=0,total=0,b2=0;
    bool auto_b2=false;
    std::string id()const;
    fs::path path(const fs::path &directory,size_t bits)const;
};
std::string token();
bool result_written(const fs::path &path,const std::string &receipt);
bool finished_written(const fs::path &path,const std::string &run,const std::string &task);
}
