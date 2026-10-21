#pragma once
#include "theme.h"
#include <cstdio>
#include <string>

namespace ecmgui_v2::ui {
enum class FontRole { Body, Caption, Heading, Mono };
struct Fonts {
    ImFont* body = nullptr;
    ImFont* mono = nullptr;
    bool cjk = false;
    bool chinese = false;
    std::string notice;
    // Call between frames, after renderer shutdown has released font textures.
    void reload(bool request_chinese, bool simulate_missing_cjk, FILE* log);
    void push(FontRole role) const;
};
}
