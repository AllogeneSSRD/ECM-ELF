#include "pause_schedule.h"
#include "io/worker_list.h"
#include <array>
#include <charconv>
#include <cctype>
#include <cwctype>
#include <windows.h>
#include <tlhelp32.h>

namespace ecmgui_v2 {
namespace {
constexpr int day_minutes = 24 * 60;
constexpr unsigned every_day = 0x7f;

bool wildcard(const std::string& text) { return text.empty() || text == "*"; }

std::optional<std::vector<std::string>> split_rule(std::string text) {
    text.erase(std::remove_if(text.begin(), text.end(), [](unsigned char c) { return std::isspace(c) != 0; }), text.end());
    std::vector<std::string> groups;
    for (std::size_t start = 0;;) {
        auto end = text.find(',', start);
        if (start < text.size() && text[start] == '[') {
            const auto close = text.find(']', start);
            if (close == text.npos || (close + 1 < text.size() && text[close + 1] != ',')) return std::nullopt;
            end = text.find(',', close + 1);
        }
        groups.push_back(text.substr(start, end - start));
        if (end == text.npos) return groups;
        start = end + 1;
    }
}

std::wstring match_text(std::wstring text) {
    text.erase(std::remove_if(text.begin(), text.end(), [](wchar_t c) { return std::iswspace(c) != 0; }), text.end());
    std::replace(text.begin(), text.end(), L'/', L'\\');
    if (text.empty()) return text;
    const int length = static_cast<int>(text.size());
    const int count = LCMapStringEx(LOCALE_NAME_INVARIANT, LCMAP_LOWERCASE, text.data(), length, nullptr, 0, nullptr, nullptr, 0);
    std::wstring lower(count, L'\0');
    if (count && LCMapStringEx(LOCALE_NAME_INVARIANT, LCMAP_LOWERCASE, text.data(), length,
                              lower.data(), count, nullptr, nullptr, 0)) return lower;
    return text;
}

std::wstring application_pattern(const std::string& text) {
    const auto length = static_cast<int>(text.size());
    const int count = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(), length, nullptr, 0);
    std::wstring result(count, L'\0');
    if (count && MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(), length, result.data(), count))
        return match_text(std::move(result));
    return {};
}

std::vector<std::string> split(const std::string& text, char delimiter) {
    std::vector<std::string> parts;
    for (std::size_t start = 0;;) {
        const auto end = text.find(delimiter, start);
        parts.push_back(ecm_workers::trim(text.substr(start, end - start)));
        if (end == text.npos) return parts;
        start = end + 1;
    }
}

int decimal(const std::string& text) {
    int value = 0;
    const auto end = text.data() + text.size();
    const auto parsed = std::from_chars(text.data(), end, value);
    return parsed.ec == std::errc{} && parsed.ptr == end ? value : -1;
}

int weekday(std::string text) {
    if (text.size() == 1 && text.front() >= '1' && text.front() <= '7') return text.front() - '1';
    for (char& ch : text) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    static constexpr std::array names{"mon", "tue", "wed", "thu", "fri", "sat", "sun"};
    const auto found = std::find(names.begin(), names.end(), text);
    return found == names.end() ? -1 : static_cast<int>(found - names.begin());
}

unsigned weekdays(const std::string& text) {
    if (wildcard(text)) return every_day;
    unsigned result = 0;
    for (const auto& group : split(text, '|')) {
        const auto range = split(group, '-');
        const int first = weekday(range.front()), last = weekday(range.back());
        if (range.size() > 2 || first < 0 || last < first) return 0;
        for (int day = first; day <= last; ++day) result |= 1u << day;
    }
    return result;
}

int clock_minute(const std::string& text) {
    const auto parts = split(text, ':');
    if (parts.size() != 2 || parts[0].size() > 2 || parts[1].size() != 2) return -1;
    const int hour = decimal(parts[0]), minute = decimal(parts[1]);
    if (hour < 0 || hour > 24 || minute < 0 || minute >= 60 || (hour == 24 && minute != 0)) return -1;
    return hour * 60 + minute;
}
} // namespace

bool PauseSchedule::parse(const std::vector<std::string>& rows, PauseScheduleError& error) {
    std::vector<Rule> parsed;
    for (const auto& row : rows) {
        error = {PauseScheduleError::Worker, parsed.size() + 1};
        Rule rule;
        auto groups = split_rule(row);
        if (!groups) return false;
        if (groups->size() > 4) { error.group = PauseScheduleError::Extra; return false; }
        // Short rows use the legacy Worker, weekday, time format.
        if (groups->size() < 4) {
            groups->resize(3);
            groups->insert(groups->begin(), "*");
        }
        if (!wildcard((*groups)[0])) {
            rule.application = application_pattern((*groups)[0]);
            if (rule.application.empty()) { error.group = PauseScheduleError::Application; return false; }
        }

        constexpr int max_worker_id = 64; // Maximum number of GUI-managed workers.
        const auto& selector = (*groups)[1];
        if (!wildcard(selector)) {
            if (selector.front() == '[' || selector.front() == '#') {
                int ignored = 0;
                const bool bracketed = selector.front() == '[';
                const auto ids = selector.substr(1, selector.size() - (bracketed ? 2 : 1));
                if ((!bracketed && decimal(ids) < 1) ||
                    !ecm_workers::parse_list(ids, max_worker_id, rule.worker_ids, ignored) ||
                    ignored || rule.worker_ids.empty()) return false;
            } else {
                rule.count = decimal(selector);
                if (rule.count < 1 || rule.count > max_worker_id) return false;
            }
        }

        rule.weekdays = weekdays((*groups)[2]);
        if (!rule.weekdays) { error.group = PauseScheduleError::Weekday; return false; }
        error.group = PauseScheduleError::Time;
        if (wildcard((*groups)[3])) rule.windows.push_back({0, day_minutes});
        else for (const auto& item : split((*groups)[3], '|')) {
            const auto range = split(item, '-');
            if (range.size() != 2) return false;
            const int start = clock_minute(range[0]), end = clock_minute(range[1]);
            if (start < 0 || start >= day_minutes || end < 0 || end == start) return false;
            rule.windows.push_back({start, end < start ? end + day_minutes : end});
        }
        parsed.push_back(std::move(rule));
    }
    rules_ = std::move(parsed);
    error = {};
    return true;
}

bool PauseSchedule::Rule::matches(int weekday, int minute) const {
    for (const auto& window : windows) {
        if ((weekdays & (1u << weekday)) && minute >= window.start && minute < window.end) return true;
        if ((weekdays & (1u << ((weekday + 6) % 7))) && minute + day_minutes < window.end) return true;
    }
    return false;
}

void PauseSchedule::reset_selection() {
    for (auto& rule : rules_) {
        rule.active = false;
        rule.selected.clear();
    }
}

bool PauseSchedule::needs_applications(int weekday, int minute) const {
    return std::any_of(rules_.begin(), rules_.end(), [&](const auto& rule) {
        return !rule.application.empty() && rule.matches(weekday, minute);
    });
}

std::set<int> PauseSchedule::paused_workers(int weekday, int minute, const std::vector<PauseWorker>& workers,
                                           const std::vector<std::wstring>& application_paths) {
    std::vector<std::wstring> paths;
    paths.reserve(application_paths.size());
    for (const auto& path : application_paths) paths.push_back(match_text(path));
    std::set<int> paused;
    for (auto& rule : rules_) {
        const bool running = rule.application.empty() || std::any_of(paths.begin(), paths.end(), [&](const auto& path) {
            return path.find(rule.application) != path.npos;
        });
        if (!rule.matches(weekday, minute) || !running) {
            rule.active = false;
            rule.selected.clear();
            continue;
        }
        if (!rule.count) {
            for (const auto& worker : workers)
                if (rule.worker_ids.empty() || std::find(rule.worker_ids.begin(), rule.worker_ids.end(), worker.index) != rule.worker_ids.end())
                    paused.insert(worker.index);
            continue;
        }
        if (!rule.active) {
            auto candidates = workers;
            std::sort(candidates.begin(), candidates.end(), [&](const auto& a, const auto& b) {
                if (paused.count(a.index) != paused.count(b.index)) return paused.count(a.index) < paused.count(b.index);
                if (a.active != b.active) return a.active;
                return a.index > b.index;
            });
            const auto limit = std::min(candidates.size(), static_cast<std::size_t>(rule.count));
            for (std::size_t i = 0; i < limit; ++i) rule.selected.push_back(candidates[i].index);
            rule.active = true;
        }
        paused.insert(rule.selected.begin(), rule.selected.end());
    }
    return paused;
}

std::optional<std::vector<std::wstring>> running_application_paths() {
    const auto snapshot = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (snapshot == INVALID_HANDLE_VALUE) return std::nullopt;
    PROCESSENTRY32W entry{};
    entry.dwSize = sizeof(entry);
    std::vector<std::wstring> paths;
    std::wstring path(32768, L'\0');
    bool next = Process32FirstW(snapshot, &entry) != FALSE;
    for (; next; next = Process32NextW(snapshot, &entry) != FALSE) {
        const auto process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, entry.th32ProcessID);
        auto length = static_cast<DWORD>(path.size());
        const bool full_path = process && QueryFullProcessImageNameW(process, 0, path.data(), &length);
        if (process) CloseHandle(process);
        if (full_path) paths.emplace_back(path.data(), length);
        else paths.emplace_back(entry.szExeFile);
    }
    const auto error = GetLastError();
    CloseHandle(snapshot);
    if (error != ERROR_NO_MORE_FILES) return std::nullopt;
    return paths;
}
} // namespace ecmgui_v2
