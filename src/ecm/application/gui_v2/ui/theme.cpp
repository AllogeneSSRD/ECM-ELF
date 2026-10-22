#include "theme.h"

namespace ecmgui_v2::ui {
namespace {
int primary_offset=UiValues{}.ui_primary_font_offset,secondary_offset=UiValues{}.ui_secondary_font_offset;
ImVec4 color(unsigned hex, float alpha = 1) {
    return {((hex >> 16) & 255) / 255.f, ((hex >> 8) & 255) / 255.f, (hex & 255) / 255.f, alpha};
}
}
float primary_font_size(float body_size){return std::max(8.f,body_size+primary_offset);}
float secondary_font_size(float body_size){return std::max(8.f,body_size+secondary_offset);}
Palette palette(bool light) {
    if (light) return {
        color(0xfcfaf7), color(0xffffff), color(0xf6f4f1), color(0xffffff),
        color(0xf2f0ed), color(0xe9e7e5), color(0xe1dfdd), color(0xcccac8),
        color(0x181613), color(0x605d5a), color(0xacaaa8), color(0x181613), color(0x302d2b), color(0xfcfaf7),
        color(0x3772bb), color(0x3772bb, .2f), color(0x267b4c), color(0xa25f12), color(0xba3e38), color(0x2e69b2),
        color(0xe7f9ec), color(0xfef2dd), color(0xffefec), color(0xebf4ff), color(0x007e74), color(0x7656ad)};
    return {
        color(0x252525), color(0x2c2c2c), color(0x202020), color(0x323232),
        color(0x383838), color(0x3f3f3f), color(0x3f3f3f), color(0x555555),
        color(0xf3f1f0), color(0xc4c2c0), color(0x5c5a58), color(0xf3f1f0), color(0xd9d7d5), color(0x252525),
        color(0x69a1e8), color(0x69a1e8, .3f), color(0x6bbc89), color(0xe8b45e), color(0xe5756e), color(0x77abec),
        color(0x182c1f), color(0x342611), color(0x3a1d1b), color(0x19273a), color(0x4eb9ad), color(0xa998dd)};
}
void apply_theme(const Settings& settings) {
    primary_offset=settings.primary_offset;secondary_offset=settings.secondary_offset;
    ImGuiStyle style{};
    if (settings.light) ImGui::StyleColorsLight(&style); else ImGui::StyleColorsDark(&style);
    const auto p = palette(settings.light);
    style.WindowPadding = {20, 18}; style.FramePadding = {10, 6};
    style.ItemSpacing = {8, 10}; style.ItemInnerSpacing = {8, 6};
    style.WindowRounding = 8; style.ChildRounding = 8; style.PopupRounding = 8;
    style.FrameRounding = 6; style.GrabRounding = 4; style.ScrollbarRounding = 6; style.TabRounding = 4;
    style.WindowBorderSize = 0; style.ChildBorderSize = 1; style.FrameBorderSize = 1;
    style.PopupBorderSize = 1; style.ScrollbarSize = 12; style.GrabMinSize = 10;
    style.DisabledAlpha = .45f;
    auto& c = style.Colors;
    c[ImGuiCol_Text] = p.fg; c[ImGuiCol_TextDisabled] = p.muted;
    c[ImGuiCol_WindowBg] = p.bg; c[ImGuiCol_ChildBg] = p.surface; c[ImGuiCol_PopupBg] = p.overlay;
    c[ImGuiCol_Border] = p.border; c[ImGuiCol_BorderShadow] = {0, 0, 0, 0};
    c[ImGuiCol_FrameBg] = p.sunken; c[ImGuiCol_FrameBgHovered] = p.hover; c[ImGuiCol_FrameBgActive] = p.active;
    c[ImGuiCol_Button] = p.surface; c[ImGuiCol_ButtonHovered] = p.hover; c[ImGuiCol_ButtonActive] = p.active;
    c[ImGuiCol_Header] = p.selection; c[ImGuiCol_HeaderHovered] = p.hover; c[ImGuiCol_HeaderActive] = p.active;
    c[ImGuiCol_CheckMark] = p.fg; c[ImGuiCol_SliderGrab] = p.muted; c[ImGuiCol_SliderGrabActive] = p.fg;
    c[ImGuiCol_Separator] = p.border; c[ImGuiCol_SeparatorHovered] = p.focus; c[ImGuiCol_SeparatorActive] = p.focus;
    c[ImGuiCol_TextSelectedBg] = p.selection; c[ImGuiCol_NavCursor] = p.focus;
    c[ImGuiCol_Tab] = p.surface; c[ImGuiCol_TabHovered] = p.hover; c[ImGuiCol_TabSelected] = p.active;
    c[ImGuiCol_TabSelectedOverline] = p.focus; c[ImGuiCol_DockingPreview] = p.selection;
    c[ImGuiCol_DockingEmptyBg] = p.sunken;
    c[ImGuiCol_TitleBg] = p.sunken; c[ImGuiCol_TitleBgActive] = p.surface;
    c[ImGuiCol_ScrollbarBg] = p.bg; c[ImGuiCol_ScrollbarGrab] = p.strong;
    c[ImGuiCol_ScrollbarGrabHovered] = p.muted; c[ImGuiCol_ScrollbarGrabActive] = p.fg;
    c[ImGuiCol_TableHeaderBg] = p.sunken; c[ImGuiCol_TableBorderStrong] = p.border;
    c[ImGuiCol_TableBorderLight] = p.border; c[ImGuiCol_TableRowBgAlt] = p.hover;
    c[ImGuiCol_PlotLines] = p.stage1; c[ImGuiCol_PlotLinesHovered] = p.stage2;
    c[ImGuiCol_PlotHistogram] = p.stage1; c[ImGuiCol_PlotHistogramHovered] = p.stage2;
    style.ScaleAllSizes(settings.scale);
    style.FontSizeBase = settings.font_size; style.FontScaleDpi = settings.scale;
    ImGui::GetStyle() = style;
}
}
