#pragma once

#include "io/config_format.h"

#include <cstddef>
#include <istream>
#include <string>
#include <string_view>

enum class WorktodoReadResult { Task, Empty, Error };
enum class WorktodoStage { Any = 0, Stage1 = 1, Stage2 = 2 };

struct WorktodoSelection {
    int priority = 0; // 0=file order, 1=Stage1 first, 2=Stage2 first
    int skip = 0;     // 0=none, 1=Stage1, 2=Stage2
};

namespace ecm_queue {
enum class LineKind { Task, Comment, Header, ForeignBracket };

inline std::string task_text(const std::string& raw) {
    const std::size_t offset = raw.compare(0, 3, "\xef\xbb\xbf") == 0 ? 3 : 0;
    return ecm_config::trim(raw.substr(offset));
}

// A foreign bracket line leaves the preceding worker section in effect.
inline LineKind classify_line(const std::string& raw, int* section, int* header) {
    const auto text = task_text(raw);
    bool bracket = false;
    const int worker = ecm_config::worker_header(text, &bracket);
    if (worker > 0) {
        if (header)
            *header = worker;
        if (section)
            *section = worker;
        return LineKind::Header;
    }
    if (bracket)
        return LineKind::ForeignBracket;
    return text.empty() || text.front() == '#' ? LineKind::Comment : LineKind::Task;
}

inline bool worker_matches(int section, int worker) {
    return worker <= 0 || section == worker;
}

inline constexpr std::string_view stage2_prefix = "ECMSTAGE2=";
inline constexpr std::string_view stage2_n_prefix = "ECMSTAGE2N=";

inline bool stage2_from_save(std::string_view text) {
    return text.substr(0, stage2_n_prefix.size()) == stage2_n_prefix;
}

inline WorktodoStage task_stage(std::string_view text) {
    // Prime95 and unknown rows go to Stage1 for parsing and diagnostics.
    return text.substr(0, stage2_prefix.size()) == stage2_prefix || stage2_from_save(text)
               ? WorktodoStage::Stage2 : WorktodoStage::Stage1;
}

inline WorktodoReadResult read_next_task(std::istream& input, int worker, WorktodoSelection selection,
                                         WorktodoStage consumer, std::string& line,
                                         std::size_t* source_line = nullptr) {
    line.clear();
    int section = 1;
    std::string first, preferred, raw;
    std::size_t number = 0, first_number = 0, preferred_number = 0;
    while (std::getline(input, raw)) {
        ++number;
        if (classify_line(raw, &section, nullptr) != LineKind::Task || !worker_matches(section, worker))
            continue;
        auto task = task_text(raw);
        const auto stage = static_cast<int>(task_stage(task));
        if (selection.skip == stage)
            continue;
        if (first.empty()) {
            first = task;
            first_number = number;
        }
        if (selection.priority == stage && preferred.empty()) {
            preferred = task;
            preferred_number = number;
        }
    }
    if (input.bad())
        return WorktodoReadResult::Error;
    const auto& selected = preferred.empty() ? first : preferred;
    if (selected.empty() || (consumer != WorktodoStage::Any && task_stage(selected) != consumer))
        return WorktodoReadResult::Empty;
    line = selected;
    if (source_line)
        *source_line = preferred.empty() ? first_number : preferred_number;
    return WorktodoReadResult::Task;
}
} // namespace ecm_queue
