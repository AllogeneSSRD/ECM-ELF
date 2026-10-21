#include "version.h"
#include "render/vulkan_window.h"
#include "render/memory_audit.h"
#include "ui/widgets.h"
#include "imgui.h"
#include "imgui_impl_win32.h"
#include "imgui_impl_vulkan.h"
#include <shellapi.h>
#include <share.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND, UINT, WPARAM, LPARAM);
namespace {
bool quit = false;
bool resized = false;
ecmgui_v2::ui::Settings* audit_settings = nullptr;
unsigned audit_sequence = 0;
bool audit_pointer = false;
ImVec2 audit_position{};
int audit_button = -1;
struct Options {
    std::filesystem::path diagnostics = L"ecm_gui_v2.log";
    int frames = 0;
    bool validation = false;
    bool no_loader = false;
    bool audit_ui = false;
    bool missing_cjk = false;
    bool explicit_scale = false;
    ecmgui_v2::ui::Settings settings;
    std::string failure;
};
std::string utf8(const wchar_t* text) {
    const int size = WideCharToMultiByte(CP_UTF8, 0, text, -1, nullptr, 0, nullptr, nullptr);
    std::string result(static_cast<size_t>(size), '\0');
    WideCharToMultiByte(CP_UTF8, 0, text, -1, result.data(), size, nullptr, nullptr);
    result.pop_back();
    return result;
}
void console(const std::string& text) {
    DWORD count = 0;
    WriteFile(GetStdHandle(STD_OUTPUT_HANDLE), text.data(), static_cast<DWORD>(text.size()), &count, nullptr);
}
LRESULT CALLBACK procedure(HWND hwnd, UINT message, WPARAM w, LPARAM l) {
    if (ImGui::GetCurrentContext() && ImGui::GetIO().BackendPlatformUserData &&
        ImGui_ImplWin32_WndProcHandler(hwnd, message, w, l)) return 1;
    switch (message) {
    case WM_APP + 33:
        // Deterministic ImGui input seam for a background acceptance window.
        // Ordinary operation continues to use the unmodified Win32 backend.
        if (audit_settings && w <= 3) {
            audit_pointer = w != 3;
            audit_position = {static_cast<float>(static_cast<short>(LOWORD(l))), static_cast<float>(static_cast<short>(HIWORD(l)))};
            if (w == 1 || w == 2) audit_button = w == 1 ? 1 : 0;
        }
        return 0;
    case WM_APP + 32:
        // Test-only, enabled explicitly by --audit-ui. No workflow data is read.
        if (audit_settings && w <= 15 && l > 0) {
            const float scales[]{1, 1.25f, 1.5f, 2};
            audit_settings->scale = scales[w & 3];
            audit_settings->chinese = (w & 4) != 0;
            audit_settings->light = (w & 8) != 0;
            audit_sequence = static_cast<unsigned>(l);
        }
        return 0;
    case WM_SIZE: resized = true; return 0;
    case WM_DPICHANGED: {
        auto rect = reinterpret_cast<RECT*>(l);
        SetWindowPos(hwnd, nullptr, rect->left, rect->top, rect->right - rect->left, rect->bottom - rect->top,
            SWP_NOZORDER | SWP_NOACTIVATE);
        return 0;
    }
    case WM_CLOSE: quit = true; return 0;
    case WM_DESTROY: quit = true; PostQuitMessage(0); return 0;
    case WM_SYSCOMMAND: if ((w & 0xfff0) == SC_KEYMENU) return 0; break;
    }
    return DefWindowProcW(hwnd, message, w, l);
}
struct NativeWindow {
    HWND hwnd = nullptr;
    HINSTANCE instance = GetModuleHandleW(nullptr);
    ~NativeWindow() {
        if (hwnd) DestroyWindow(hwnd);
        UnregisterClassW(L"ecm_gui_v2", instance);
    }
    void create() {
        WNDCLASSEXW wc{sizeof(wc)};
        wc.lpfnWndProc = procedure; wc.hInstance = instance;
        wc.lpszClassName = L"ecm_gui_v2"; wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
        if (!RegisterClassExW(&wc)) throw std::runtime_error("RegisterClassExW failed");
        std::wstring title = L"ECM-ELF GUI v2 ";
        for (char value : std::string(ecmgui_v2::version)) title += static_cast<wchar_t>(value);
        hwnd = CreateWindowExW(0, wc.lpszClassName, title.c_str(), WS_OVERLAPPEDWINDOW,
            CW_USEDEFAULT, CW_USEDEFAULT, 1100, 760, nullptr, nullptr, instance, nullptr);
        if (!hwnd) throw std::runtime_error("CreateWindowExW failed");
    }
};
struct ImGuiSession {
    explicit ImGuiSession(HWND hwnd) {
        IMGUI_CHECKVERSION(); ImGui::CreateContext();
        ImGui::GetIO().IniFilename = nullptr;
        ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_DockingEnable | ImGuiConfigFlags_NavEnableKeyboard;
        if (!ImGui_ImplWin32_Init(hwnd)) {
            ImGui::DestroyContext();
            throw std::runtime_error("ImGui Win32 initialization failed");
        }
    }
    ~ImGuiSession() {
        if (ImGui::GetIO().BackendPlatformUserData) ImGui_ImplWin32_Shutdown();
        ImGui::DestroyContext();
    }
};
int run(const Options& options, int show, FILE* log) {
    ImGui_ImplWin32_EnableDpiAwareness();
    NativeWindow native; native.create();
    ImGuiSession imgui(native.hwnd);
    ecmgui_v2::VulkanContext context;
    context.initialize(native.hwnd, options.validation, options.no_loader, log);
    ecmgui_v2::MemoryAudit memory(options.audit_ui);
    ecmgui_v2::VulkanWindow renderer(context, log);
    auto pending = options.settings;
    UINT dpi = GetDpiForWindow(native.hwnd);
    if (!options.explicit_scale) pending.scale = dpi / 96.f;
    auto active = pending;
    ecmgui_v2::ui::Fonts fonts;
    fonts.reload(active.chinese, options.missing_cjk, log);
    active.chinese = pending.chinese = fonts.chinese;
    ecmgui_v2::ui::apply_theme(active);
    ecmgui_v2::ui::Showcase showcase;
    if (options.audit_ui) audit_settings = &pending;
    unsigned sampled_sequence = 0, sample_frames = 0;
    ShowWindow(native.hwnd, show);
    unsigned frames = 0;
    while (!quit) {
        const auto started = GetTickCount64();
        MSG message{};
        while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE)) {
            if (message.message == WM_QUIT) quit = true;
            TranslateMessage(&message); DispatchMessageW(&message);
        }
        if (quit) break;
        const auto new_dpi = GetDpiForWindow(native.hwnd);
        if (new_dpi && new_dpi != dpi) { dpi = new_dpi; pending.scale = dpi / 96.f; }
        RECT client{}; GetClientRect(native.hwnd, &client);
        if (IsIconic(native.hwnd) || client.right <= 0 || client.bottom <= 0) {
            MsgWaitForMultipleObjectsEx(0, nullptr, 100, QS_ALLINPUT, MWMO_INPUTAVAILABLE);
            continue;
        }
        const bool font_change = active.chinese != pending.chinese || active.scale != pending.scale;
        if (resized || renderer.needs_rebuild || font_change) {
            // Vulkan Shutdown also destroys the main viewport's Win32 data.
            // Reinitialize both backends while preserving the ImGui context.
            if (renderer.rebuilds) {
                renderer.shutdown();
                ImGui_ImplWin32_Shutdown();
                if (!ImGui_ImplWin32_Init(native.hwnd)) throw std::runtime_error("Win32 backend reinitialization failed");
            }
            if (font_change) {
                fonts.reload(pending.chinese, options.missing_cjk, log);
                pending.chinese = fonts.chinese;
            }
            renderer.rebuild(static_cast<uint32_t>(client.right), static_cast<uint32_t>(client.bottom));
            resized = false;
        }
        if (font_change || active.light != pending.light) {
            active = pending;
            ecmgui_v2::ui::apply_theme(active);
        }
        const auto background = ecmgui_v2::ui::palette(active.light).bg;
        renderer.clear_color[0] = background.x; renderer.clear_color[1] = background.y; renderer.clear_color[2] = background.z;
        if (renderer.needs_rebuild) {
            MsgWaitForMultipleObjectsEx(0, nullptr, 100, QS_ALLINPUT, MWMO_INPUTAVAILABLE);
            continue;
        }
        ImGui_ImplVulkan_NewFrame(); ImGui_ImplWin32_NewFrame();
        if (options.audit_ui && audit_pointer) {
            auto& io = ImGui::GetIO();
            io.AddFocusEvent(true);
            io.AddMouseViewportEvent(ImGui::GetMainViewport()->ID);
            io.AddMousePosEvent(audit_position.x, audit_position.y);
            if (audit_button >= 0) { io.AddMouseButtonEvent(0, audit_button != 0); audit_button = -1; }
        }
        ImGui::NewFrame();
        showcase.draw(pending, active, fonts, context.properties.deviceName);
        ImGui::Render();
        if (renderer.render(ImGui::GetDrawData())) {
            ++frames;
            if (frames == 1 || frames % 20 == 0) { std::fprintf(log, "frame: %u\n", frames); std::fflush(log); }
        }
        if (options.audit_ui && audit_sequence != sampled_sequence) {
            if (++sample_frames == 4) {
                unsigned textures = 0;
                for (auto texture : ImGui::GetPlatformIO().Textures) if (texture->BackendUserData) ++textures;
                POINT primary{static_cast<LONG>(showcase.primary_center.x), static_cast<LONG>(showcase.primary_center.y)};
                POINT field{static_cast<LONG>(showcase.field_center.x), static_cast<LONG>(showcase.field_center.y)};
                // With viewports disabled, ImGui coordinates already use the client origin.
                memory.describe(log);
                std::fprintf(log, "ui_sample: seq=%u light=%d zh=%d scale=%.2f body=%.2f caption=%.2f heading=%.2f mono=%.2f mono_i=%.2f mono_w=%.2f padding=%.2f cjk=%d mono_cjk=%d gpu_bytes=%llu allocations=%zu textures=%u fonts=%d\n",
                    audit_sequence, active.light, fonts.chinese, active.scale, showcase.body_pixels, showcase.caption_pixels,
                    showcase.heading_pixels, showcase.mono_pixels, showcase.mono_i, showcase.mono_w,
                    ImGui::GetStyle().WindowPadding.x, fonts.cjk, fonts.mono->IsGlyphInFont(0x6587),
                    static_cast<unsigned long long>(memory.bytes()), memory.allocations(), textures, ImGui::GetIO().Fonts->Fonts.Size);
                std::fflush(log); sampled_sequence = audit_sequence; sample_frames = 0;
                std::fprintf(log, "ui_controls: seq=%u clicks=%u path_length=%zu button_x=%ld button_y=%ld input_x=%ld input_y=%ld nav_visible=%d\n",
                    audit_sequence, showcase.clicks, std::strlen(showcase.path), primary.x, primary.y, field.x, field.y, ImGui::GetIO().NavVisible);
                std::fflush(log);
            }
        }
        if (frames == 5 && !options.failure.empty())
            ecmgui_v2::check_vk(options.failure == "surface-lost" ? VK_ERROR_SURFACE_LOST_KHR : VK_ERROR_DEVICE_LOST, "injected renderer failure");
        if (options.frames && frames >= static_cast<unsigned>(options.frames)) break;
        const auto elapsed = GetTickCount64() - started;
        if (elapsed < 100) MsgWaitForMultipleObjectsEx(0, nullptr, static_cast<DWORD>(100 - elapsed), QS_ALLINPUT, MWMO_INPUTAVAILABLE);
    }
    renderer.shutdown();
    audit_settings = nullptr;
    if (options.audit_ui) {
        std::fprintf(log, "memory_audit_end: bytes=%llu allocations=%zu\n", static_cast<unsigned long long>(memory.bytes()), memory.allocations());
        if (memory.allocations()) throw std::runtime_error("Vulkan allocation audit found unreleased GUI memory");
    }
    std::fprintf(log, "exit: frames=%u rebuilds=%u worker_processes=0\n", frames, renderer.rebuilds);
    std::fflush(log);
    return context.validation_errors.load() ? 3 : 0;
}
}
int WINAPI wWinMain(HINSTANCE, HINSTANCE, PWSTR, int show) {
    Options options;
    FILE* log = nullptr;
    try {
        int count = 0;
        auto argv = CommandLineToArgvW(GetCommandLineW(), &count);
        if (!argv) throw std::runtime_error("CommandLineToArgvW failed");
        std::vector<std::wstring> args(argv + 1, argv + count); LocalFree(argv);
        for (size_t i = 0; i < args.size(); ++i) {
            const auto argument = args[i];
            if (argument == L"--version") { console(std::string("ECM-ELF GUI v2 ") + ecmgui_v2::version + " (Vulkan)\n"); return 0; }
            if (argument == L"--validation") options.validation = true;
            else if (argument == L"--no-loader") options.no_loader = true;
            else if (argument == L"--audit-ui") options.audit_ui = true;
            else if (argument == L"--simulate-missing-cjk") options.missing_cjk = true;
            else if ((argument == L"--diagnostics" || argument == L"--frames" || argument == L"--inject-failure" ||
                      argument == L"--theme" || argument == L"--language" || argument == L"--ui-scale") && i + 1 < args.size()) {
                const auto value = args[++i];
                if (argument == L"--diagnostics") options.diagnostics = value;
                else if (argument == L"--theme") {
                    if (value != L"dark" && value != L"light") throw std::runtime_error("--theme requires dark or light");
                    options.settings.light = value == L"light";
                } else if (argument == L"--language") {
                    if (value != L"en" && value != L"zh") throw std::runtime_error("--language requires en or zh");
                    options.settings.chinese = value == L"zh";
                } else if (argument == L"--ui-scale") {
                    if (value != L"1" && value != L"1.25" && value != L"1.5" && value != L"2") throw std::runtime_error("--ui-scale requires 1, 1.25, 1.5 or 2");
                    options.settings.scale = std::stof(value); options.explicit_scale = true;
                }
                else if (argument == L"--frames") {
                    size_t used = 0; options.frames = std::stoi(value, &used);
                    if (used != value.size() || options.frames < 1) throw std::runtime_error("--frames requires a positive integer");
                } else {
                    options.failure = utf8(value.c_str());
                    if (options.failure != "device-lost" && options.failure != "surface-lost") throw std::runtime_error("Invalid --inject-failure mode");
                }
            } else throw std::runtime_error("Unknown or incomplete option: " + utf8(argument.c_str()));
        }
        log = _wfsopen(options.diagnostics.c_str(), L"w", _SH_DENYWR);
        if (!log) throw std::runtime_error("Cannot open GUI v2 diagnostics file");
        std::fprintf(log, "ECM-ELF GUI v2 %s backend=Vulkan\n", ecmgui_v2::version);
        const int result = run(options, show, log);
        std::fclose(log); return result;
    } catch (const std::exception& error) {
        if (log) { std::fprintf(log, "error: %s\n", error.what()); std::fclose(log); }
        console(std::string("GUI v2: ") + error.what() + "\n");
        return 1;
    }
}
