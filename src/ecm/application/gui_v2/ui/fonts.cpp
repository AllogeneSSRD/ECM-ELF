#include "fonts.h"
#include <windows.h>
#include <filesystem>
#include <initializer_list>
#include <fstream>

namespace ecmgui_v2::ui {
namespace {
std::filesystem::path find_font(std::initializer_list<const wchar_t*> names) {
    wchar_t windows[MAX_PATH]{};
    if (!GetWindowsDirectoryW(windows, MAX_PATH)) return {};
    for (const auto name : names) {
        auto file = std::filesystem::path(windows) / L"Fonts" / name;
        if (std::filesystem::is_regular_file(file)) return file;
    }
    return {};
}
ImFont* add(const std::filesystem::path& path, std::vector<std::vector<unsigned char>>& files, bool merge = false, unsigned char* shared = nullptr, int shared_size = 0, ImFont* destination = nullptr) {
    if (path.empty()) return nullptr;
    ImFontConfig config; config.MergeMode = merge; config.FontDataOwnedByAtlas = false;
    if(destination)config.DstFont=destination;
    if (!shared) {
        std::ifstream input(path, std::ios::binary);
        files.emplace_back(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
        if (files.back().empty()) return nullptr;
        shared = files.back().data(); shared_size = static_cast<int>(files.back().size());
    }
    return ImGui::GetIO().Fonts->AddFontFromMemoryTTF(shared, shared_size, 15, &config);
}
}
void Fonts::reload(bool request_chinese, bool simulate_missing_cjk, FILE* log) {
    auto& io = ImGui::GetIO();
    const auto cjk_file = simulate_missing_cjk ? std::filesystem::path{} : find_font({L"msyh.ttc", L"simhei.ttf", L"simsun.ttc"});
    const auto ui_file = find_font({L"segoeui.ttf", L"arial.ttf"});
    const auto mono_file = find_font({L"consola.ttf", L"cour.ttf"});
    notice.clear();
    const auto primary_file = request_chinese && !cjk_file.empty() ? cjk_file : ui_file;
    if (!english) {
        english = add(ui_file, files);
        if (!english) english = io.Fonts->AddFontDefault();
        unsigned char* cjk_data = nullptr; int cjk_size = 0;
        if (!cjk_file.empty()) {
            add(cjk_file, files, true);
            cjk_data = files.back().data(); cjk_size = static_cast<int>(files.back().size());
            chinese_font = add(cjk_file, files, false, cjk_data, cjk_size);
        }
        mono = add(mono_file, files);
        if (mono && cjk_data) add(cjk_file, files, true, cjk_data, cjk_size);
        if (!mono) mono = english;
        // Preserve the battlesuit names' musical note and heart, including in English UI.
        const auto symbol_file=find_font({L"seguisym.ttf"});
        if(add(symbol_file,files,true,nullptr,0,english)){
            auto* symbol_data=files.back().data();const int symbol_size=static_cast<int>(files.back().size());
            if(chinese_font)add(symbol_file,files,true,symbol_data,symbol_size,chinese_font);
            if(mono!=english&&mono!=chinese_font)add(symbol_file,files,true,symbol_data,symbol_size,mono);
        }
    }
    body = request_chinese && chinese_font ? chinese_font : english;
    cjk = body->IsGlyphInFont(0x4e2d) && body->IsGlyphInFont(0x6587);
    chinese = request_chinese && cjk;
    if (request_chinese && !chinese) notice = "Chinese font unavailable. Using English; install Microsoft YaHei, SimHei or SimSun.";
    io.FontDefault = body;
    std::fprintf(log, "fonts: language=%s cjk=%d ui=%s mono=%s notice=%s\n", chinese ? "zh" : "en", cjk,
        primary_file.u8string().c_str(), mono_file.u8string().c_str(), notice.c_str());
    std::fflush(log);
}
Fonts::~Fonts() {
    if (ImGui::GetCurrentContext()) { ImGui::GetIO().FontDefault = nullptr; ImGui::GetIO().Fonts->Clear(); }
}
void Fonts::push(FontRole role) const {
    const float size = (role == FontRole::Caption ? 13.f : role == FontRole::Heading ? 22.f : 15.f)*base_size/15.f;
    ImGui::PushFont(role == FontRole::Mono ? mono : body, size);
}
}
