#pragma once

#include "io/config_format.h"
#include <charconv>
#include <filesystem>

namespace ecm_stage2 {

struct LaunchConfig {
    std::filesystem::path path;
    int worker = 1;
    std::vector<ecm_config::TomlLine> lines;
};

inline int launch_worker(const std::string& value) {
    int worker = 0;
    const auto end = value.data() + value.size();
    const auto parsed = std::from_chars(value.data(), end, worker);
    if (parsed.ec != std::errc{} || parsed.ptr != end || worker < 1)
        throw std::runtime_error("invalid --worker");
    return worker;
}

inline LaunchConfig read_launch_config(const std::string& config, int worker,
                                       const std::filesystem::path& executable) {
    namespace fs = std::filesystem;
    auto path = config.empty() ? executable.parent_path() / "ecm.toml"
                               : fs::absolute(fs::path(config));
    path = path.lexically_normal();
    LaunchConfig launch{path, worker, {}};
    std::ifstream input(path, std::ios::binary);
    if (!ecm_config::read_toml(input, path.u8string(), launch.lines, ecm_config::TomlReadMode::Stage2,
                               nullptr, nullptr, worker) && (!config.empty() || fs::exists(path)))
        throw std::runtime_error("cannot read Stage2 TOML: " + path.u8string());
    return launch;
}

// The combined launcher passes the same config snapshot to the selected parser.
int run_cpu_stage2(int argc, char** argv, const LaunchConfig* launch = nullptr);

using GpuEntry = int (*)(int argc, char** argv, const LaunchConfig* launch);
int run_combined_stage2(int argc, char** argv, GpuEntry gpu);

} // namespace ecm_stage2
