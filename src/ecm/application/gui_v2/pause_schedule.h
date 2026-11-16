#pragma once

#include <cstddef>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace ecmgui_v2 {
struct PauseScheduleError {
    enum Group { None, Worker, Weekday, Time, Extra, Application } group = None;
    std::size_t item = 0;
};

struct PauseWorker {
    int index;
    bool active;
};

class PauseSchedule {
public:
    bool parse(const std::vector<std::string>& rows, PauseScheduleError& error);
    bool needs_applications(int weekday, int minute) const;
    std::set<int> paused_workers(int weekday, int minute, const std::vector<PauseWorker>& workers,
                                 const std::vector<std::wstring>& application_paths = {});
    void reset_selection();

private:
    struct Window { int start, end; };
    struct Rule {
        std::wstring application;
        int count = 0;
        std::vector<int> worker_ids;
        unsigned weekdays = 0;
        std::vector<Window> windows;
        std::vector<int> selected;
        bool active = false;
        bool matches(int weekday, int minute) const;
    };
    std::vector<Rule> rules_;
};

std::optional<std::vector<std::wstring>> running_application_paths();
} // namespace ecmgui_v2
