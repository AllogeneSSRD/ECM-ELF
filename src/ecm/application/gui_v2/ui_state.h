#pragma once
#include "generated/ui_values.h"
#include <array>

namespace ecmgui { class TomlFile; }

namespace ecmgui_v2 {
struct UiState : UiValues {
    std::string notice;
    bool writable = true;
    std::array<bool,3> dock_tab_saved{};
    bool load(const ecmgui::TomlFile& config);
    void apply_changes(ecmgui::TomlFile& config) const;
    void mark_saved();
    std::string layout() const;
    void layout(const std::string& text);
private:
    UiValues baseline_;
};
}
