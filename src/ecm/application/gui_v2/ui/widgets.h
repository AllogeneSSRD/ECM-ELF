#pragma once
#include "fonts.h"
#include "icons.h"

namespace ecmgui_v2::ui {
enum class ButtonKind { Primary, Secondary, Ghost, Danger, Selected };
enum class Status { Success, Warning, Danger, Info };
bool navigation_button(const char* id,const char* label,Icon icon,bool selected,const Palette& p,float scale,float height);
bool icon_button(const char* id,Icon icon,const Palette& p,float scale);
bool theme_card(const char* id,const char* label,bool light,bool selected,const Palette& p,float scale,float width);
bool button(const char* id, const char* label, ButtonKind kind, const Palette& p, float scale, bool disabled = false,float* rendered_size=nullptr,bool compact=false,float width=0,float height=0,Icon icon=Icon::None);
std::string fitted_label(const char* text,float width);
void secondary_text(const char* text,float* rendered_size=nullptr);
void status_badge(const char* label, Status status, const Palette& p, float scale,float max_width=0);
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
