#pragma once
#include "theme.h"
#include <cstdio>
#include <string>
#include <vector>

namespace ecmgui_v2::ui {
enum class FontRole { Body, Caption, Heading, Mono };
struct Fonts {
    ~Fonts();
    ImFont* body = nullptr;
    ImFont* mono = nullptr;
    bool cjk = false;
    bool chinese = false;
    float base_size = 15;
    std::string notice;
    ImFont* english = nullptr;
    ImFont* chinese_font = nullptr;
    std::vector<std::vector<unsigned char>> files;
    // Input files are shared by font roles; cached fonts keep viewports alive.
    void reload(bool request_chinese, bool simulate_missing_cjk, FILE* log);
    void push(FontRole role) const;
};
}
