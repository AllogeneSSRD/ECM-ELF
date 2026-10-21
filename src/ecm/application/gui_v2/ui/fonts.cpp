#include "fonts.h"
#include <windows.h>
#include <filesystem>
#include <initializer_list>

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
ImFont* add(const std::filesystem::path& path, bool merge = false) {
    if (path.empty()) return nullptr;
    ImFontConfig config; config.MergeMode = merge;
    return ImGui::GetIO().Fonts->AddFontFromFileTTF(path.u8string().c_str(), 15, &config);
}
}
void Fonts::reload(bool request_chinese, bool simulate_missing_cjk, FILE* log) {
    auto& io = ImGui::GetIO();
    io.FontDefault = nullptr;
    io.Fonts->Clear(); body = mono = nullptr;
    const auto cjk_file = simulate_missing_cjk ? std::filesystem::path{} : find_font({L"msyh.ttc", L"simhei.ttf", L"simsun.ttc"});
    const auto ui_file = find_font({L"segoeui.ttf", L"arial.ttf"});
    const auto mono_file = find_font({L"consola.ttf", L"cour.ttf"});
    notice.clear();
    const auto primary_file = request_chinese && !cjk_file.empty() ? cjk_file : ui_file;
    body = add(primary_file);
    if (!body) body = io.Fonts->AddFontDefault();
    if (!request_chinese && !cjk_file.empty()) add(cjk_file, true);
    cjk = body->IsGlyphInFont(0x4e2d) && body->IsGlyphInFont(0x6587);
    chinese = request_chinese && cjk;
    if (request_chinese && !chinese) notice = "Chinese font unavailable. Using English; install Microsoft YaHei, SimHei or SimSun.";
    mono = add(mono_file);
    if (mono && !cjk_file.empty()) add(cjk_file, true);
    if (!mono) mono = body;
    io.FontDefault = body;
    std::fprintf(log, "fonts: language=%s cjk=%d ui=%s mono=%s notice=%s\n", chinese ? "zh" : "en", cjk,
        primary_file.u8string().c_str(), mono_file.u8string().c_str(), notice.c_str());
    std::fflush(log);
}
void Fonts::push(FontRole role) const {
    const float size = role == FontRole::Caption ? 13.f : role == FontRole::Heading ? 22.f : 15.f;
    ImGui::PushFont(role == FontRole::Mono ? mono : body, size);
}
}
