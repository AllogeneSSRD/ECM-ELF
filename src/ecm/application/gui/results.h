#pragma once

// Read-only view of worker-owned submission JSON lines. No result generation,
// merging, checksum rewriting or ledger access belongs in the GUI.
#include "../configuration/ecm_toml.h"
#include "../configuration/generated/ecm_config_generated.h"
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace ecmgui {
std::vector<std::string> result_paths(const std::vector<ecm_config::TomlLine> &lines,
                                     const std::string &config_path,int workers);
struct ResultRecord {
    std::string id,raw,status,target,b1,b2,curves,model,timestamp,factors;
    std::size_t source=0,line=0;
    int stage=0;
};
struct ResultFilter {
    std::string status,search;
    int stage=0,source=-1;
    bool matches(const ResultRecord &row) const;
};
struct ResultFile {
    std::string path,error;
    bool exists=false;
    std::size_t invalid_lines=0,pending_lines=0;
};
class ResultStore {
public:
    void set_paths(const std::vector<std::string> &paths);
    void add_path(const std::string &path);
    // Only changed files are read. Missing files are normal before first output.
    bool refresh(bool force=false);
    const std::vector<ResultRecord> &rows() const {return rows_;}
    const std::vector<ResultFile> &files() const {return files_;}
    std::vector<std::size_t> filtered(const ResultFilter &filter) const;
    // Original JSON bytes, one row per newline, in the supplied display order.
    std::string copy_text(const std::vector<std::size_t> &indices) const;
private:
    struct Cache {
        bool loaded=false,exists=false;
        std::uintmax_t size=0;
        std::filesystem::file_time_type modified{};
        std::vector<ResultRecord> rows;
    };
    std::vector<ResultFile> files_;
    std::vector<Cache> cache_;
    std::vector<ResultRecord> rows_;
};
} // namespace ecmgui
