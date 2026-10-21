#pragma once
#include "fonts.h"

namespace ecmgui_v2::ui {
enum class ButtonKind { Primary, Secondary, Ghost, Danger };
enum class Status { Success, Warning, Danger, Info };
enum class Icon { Grid, Sliders, Check, Folder };
bool button(const char* id, const char* label, ButtonKind kind, const Palette& p, float scale, bool disabled = false);
void status_badge(const char* label, Status status, const Palette& p, float scale);
void section_header(const char* title, const char* detail, const Fonts& fonts, const Palette& p);
void form_row(const char* label, const char* id, char* value, size_t capacity, float scale);
void icon(Icon shape, const Palette& p, float scale);
// Component specimen, including the six interaction states with explicit labels.
void state_strip(const Palette& p, float scale, bool chinese);
struct Showcase {
    int page = 0;
    bool selected = true;
    unsigned clicks = 0;
    ImVec2 primary_center{}, field_center{};
    char path[256] = "workflows/sample.toml";
    float body_pixels = 0, caption_pixels = 0, heading_pixels = 0, mono_pixels = 0;
    float mono_i = 0, mono_w = 0;
    void draw(Settings& pending, const Settings& active, const Fonts& fonts, const char* gpu);
};
}
