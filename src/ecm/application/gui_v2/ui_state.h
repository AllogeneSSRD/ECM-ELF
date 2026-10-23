#pragma once
#include "generated/ui_values.h"
#include <filesystem>
#include <array>

namespace ecmgui_v2 {
struct UiState : UiValues {
    std::filesystem::path path;
    std::string notice;
    bool writable = true;
    std::array<bool,3> dock_tab_saved{}; // Legacy layouts can supply the first preference.
    static std::filesystem::path default_path(const std::string& config);
    bool load();
    bool save();
    std::string layout() const;
    void layout(const std::string& text);
};
}
