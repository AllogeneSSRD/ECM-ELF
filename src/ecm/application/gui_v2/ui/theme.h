#pragma once
#include "imgui.h"

namespace ecmgui_v2::ui {
struct Settings {
    bool light = false;
    bool chinese = false;
    float scale = 1;
};
struct Palette {
    ImVec4 bg, surface, sunken, overlay, hover, active, border, strong;
    ImVec4 fg, muted, disabled, accent, accent_hover, on_accent;
    ImVec4 focus, selection, success, warning, danger, info;
    ImVec4 success_bg, warning_bg, danger_bg, info_bg, stage1, stage2;
};
Palette palette(bool light);
// Always rebuild from logical sizes; never rescale the current style.
void apply_theme(const Settings& settings);
}
