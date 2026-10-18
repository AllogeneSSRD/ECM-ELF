#pragma once
#include <filesystem>

namespace ecm_results {
inline std::filesystem::path legacy_submission_ledger(const std::filesystem::path &path){return std::filesystem::path(path.string()+".ecm-ledger");}
inline std::filesystem::path submission_ledger(const std::filesystem::path &path){return path.parent_path()/"data"/(path.filename().string()+".ecm-ledger");}
inline bool has_result_ledger(const std::filesystem::path &path){return std::filesystem::exists(submission_ledger(path)/"format")||std::filesystem::exists(legacy_submission_ledger(path)/"format");}
inline bool has_result_ledger_directory(const std::filesystem::path &path){return std::filesystem::exists(submission_ledger(path))||std::filesystem::exists(legacy_submission_ledger(path));}
}
