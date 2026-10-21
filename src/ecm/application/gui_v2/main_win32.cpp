#include "version.h"
#include "render/vulkan_window.h"
#include "render/memory_audit.h"
#include "ui/widgets.h"
#include "app.h"
#include "pet/window.h"
#include "ui/scroll.h"
#include <dwmapi.h>
#include <commctrl.h>
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
bool close_requested = false;
bool audit_app = false;
ecmgui_v2::App* current_app = nullptr;
ecmgui_v2::ui::Settings* audit_settings = nullptr;
unsigned audit_sequence = 0;
bool audit_pointer = false;
ImVec2 audit_position{};
int audit_button = -1;
ecmgui_v2::ui::SmoothScroll smooth_scroll;
unsigned long long scroll_tick=0;
void (*platform_create_window)(ImGuiViewport*)=nullptr;
bool collect_wheel(UINT message,WPARAM w){
    if(!ImGui::GetCurrentContext()||(message!=WM_MOUSEWHEEL&&message!=WM_MOUSEHWHEEL))return false;
    const double delta=static_cast<short>(HIWORD(w))/double(WHEEL_DELTA);
    smooth_scroll.add(message==WM_MOUSEHWHEEL?-delta:0,message==WM_MOUSEWHEEL?delta:0);return true;
}
LRESULT CALLBACK smooth_platform(HWND hwnd,UINT message,WPARAM w,LPARAM l,UINT_PTR id,DWORD_PTR){
    if(collect_wheel(message,w))return 0;
    if(message==WM_NCDESTROY)RemoveWindowSubclass(hwnd,smooth_platform,id);
    return DefSubclassProc(hwnd,message,w,l);
}
void create_smooth_platform(ImGuiViewport* viewport){
    platform_create_window(viewport);
    const auto hwnd=static_cast<HWND>(viewport->PlatformHandleRaw?viewport->PlatformHandleRaw:viewport->PlatformHandle);
    if(!SetWindowSubclass(hwnd,smooth_platform,1,0))throw std::runtime_error("Cannot attach viewport scroll handler");
}
struct Options {
    std::filesystem::path diagnostics = L"ecm_gui_v2.log";
    int frames = 0;
    bool validation = false;
    bool no_loader = false;
    bool audit_ui = false;
    bool missing_cjk = false;
    bool explicit_scale = false;
    bool explicit_theme = false, explicit_language = false;
    bool showcase = false, no_nvml = false, detach_probe = false, audit_app = false;
    std::string config;
    std::filesystem::path ui_state;
    int start_worker = 0;
    int pet = -1;
    bool pet_static = false;
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
    if(collect_wheel(message,w))return 0;
    if (ImGui::GetCurrentContext() && ImGui::GetIO().BackendPlatformUserData &&
        ImGui_ImplWin32_WndProcHandler(hwnd, message, w, l)) return 1;
    switch (message) {
    case WM_APP + 34:
        if (current_app && audit_app) {
            if (w == 1) current_app->start(static_cast<int>(l));
            if (w == 2) current_app->controller.request_stop(static_cast<int>(l));
            if (w == 3) current_app->begin_exit();
            if (w == 4) current_app->request_close();
            if (w == 5) current_app->controller.cancel_close();
            if (w == 6) current_app->test_float = l != 0;
            if (w == 7 && l >= 0 && l <= 6) current_app->state.ui_page = static_cast<int>(l);
            if (w == 8 && l >= -1 && l <= 6) current_app->pet_test_mood = static_cast<int>(l);
            if (w == 9) current_app->state.pet_enabled = l != 0;
        }
        return 0;
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
    case WM_CLOSE: close_requested = true; return 0;
    case WM_GETMINMAXINFO: {
        auto info = reinterpret_cast<MINMAXINFO*>(l); info->ptMinTrackSize = {480, 360}; return 0;
    }
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
    void create(const ecmgui_v2::UiState& state) {
        WNDCLASSEXW wc{sizeof(wc)};
        wc.lpfnWndProc = procedure; wc.hInstance = instance;
        wc.lpszClassName = L"ecm_gui_v2"; wc.hCursor = LoadCursorW(nullptr, IDC_ARROW);
        if (!RegisterClassExW(&wc)) throw std::runtime_error("RegisterClassExW failed");
        std::wstring title = L"ECM-ELF GUI v2 ";
        for (char value : std::string(ecmgui_v2::version)) title += static_cast<wchar_t>(value);
        RECT rect{state.window_x, state.window_y, state.window_x + state.window_width, state.window_y + state.window_height};
        const bool visible = state.window_x != INT_MIN && state.window_y != INT_MIN && MonitorFromRect(&rect, MONITOR_DEFAULTTONULL);
        hwnd = CreateWindowExW(0, wc.lpszClassName, title.c_str(), WS_OVERLAPPEDWINDOW,
            visible ? state.window_x : CW_USEDEFAULT, visible ? state.window_y : CW_USEDEFAULT, state.window_width, state.window_height, nullptr, nullptr, instance, nullptr);
        if (!hwnd) throw std::runtime_error("CreateWindowExW failed");
    }
};
struct ImGuiSession {
    explicit ImGuiSession(HWND hwnd) {
        IMGUI_CHECKVERSION(); ImGui::CreateContext();
        ImGui::GetIO().IniFilename = nullptr;
        ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_DockingEnable | ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_ViewportsEnable;
        ImGui::GetIO().ConfigDpiScaleViewports = true;
        if (!ImGui_ImplWin32_Init(hwnd)) {
            ImGui::DestroyContext();
            throw std::runtime_error("ImGui Win32 initialization failed");
        }
        platform_create_window=ImGui::GetPlatformIO().Platform_CreateWindow;
        ImGui::GetPlatformIO().Platform_CreateWindow=create_smooth_platform;
    }
    ~ImGuiSession() {
        if (ImGui::GetIO().BackendPlatformUserData) ImGui_ImplWin32_Shutdown();
        ImGui::DestroyContext();
    }
};
int run(const Options& options, int show, FILE* log) {
    ImGui_ImplWin32_EnableDpiAwareness();
    ecmgui_v2::UiState state;
    state.path = options.ui_state.empty() ? ecmgui_v2::UiState::default_path(options.config) : std::filesystem::absolute(options.ui_state);
    if (!options.config.empty()) {
        std::error_code error;
        const auto business = std::filesystem::weakly_canonical(std::filesystem::u8path(options.config));
        if (std::filesystem::equivalent(business, state.path, error) || _wcsicmp(business.c_str(), std::filesystem::weakly_canonical(state.path).c_str()) == 0)
            throw std::runtime_error("Business TOML and UI sidecar must be distinct");
    }
    state.load();
    if (options.explicit_theme) state.ui_theme = options.settings.light ? "light" : "dark";
    if (options.explicit_language) state.ui_language = options.settings.chinese ? "zh" : "en";
    if (options.explicit_scale) state.ui_scale = options.settings.scale;
    if (options.pet >= 0) state.pet_enabled = options.pet != 0;
    if (options.pet_static) state.pet_animate = false;
    NativeWindow native; native.create(state);
    ImGuiSession imgui(native.hwnd);
    ecmgui_v2::ui::Fonts fonts;
    ecmgui_v2::VulkanContext context;
    context.initialize(native.hwnd, options.validation, options.no_loader, log);
    ecmgui_v2::MemoryAudit memory(options.audit_ui);
    ecmgui_v2::VulkanWindow renderer(context, log);
    auto pending = options.settings;
    if (!options.explicit_theme) pending.light = state.ui_theme == "light";
    if (!options.explicit_language) pending.chinese = state.ui_language == "zh";
    if (!options.explicit_scale && state.ui_scale > 0) pending.scale = state.ui_scale;
    UINT dpi = GetDpiForWindow(native.hwnd);
    if (!options.explicit_scale && state.ui_scale == 0) pending.scale = dpi / 96.f;
    auto active = pending;
    fonts.reload(active.chinese, options.missing_cjk, log);
    active.chinese = pending.chinese = fonts.chinese;
    ecmgui_v2::ui::apply_theme(active);
    ecmgui_v2::ui::Showcase showcase;
    const bool show_components = options.showcase || options.audit_ui;
    ecmgui_v2::App app(state, log); current_app = &app; audit_app = options.audit_app;
    struct ResetPointers { ~ResetPointers() { current_app = nullptr; audit_app = false; audit_settings = nullptr; } } reset_pointers;
    ecmgui_v2::pet::Window companion(native.hwnd, state, log);
    app.initialize(options.config, options.no_nvml || show_components); app.test_float = options.detach_probe;
    if (!state.notice.empty()) app.trace(state.notice);
    try { const auto layout = state.layout(); if (!layout.empty() && !show_components) ImGui::LoadIniSettingsFromMemory(layout.data(), layout.size()); }
    catch (const std::exception& error) { app.trace(error.what()); app.reset_layout = true; }
    if (options.start_worker > 0) app.start(options.start_worker);
    if (options.audit_ui) audit_settings = &pending;
    unsigned sampled_sequence = 0, sample_frames = 0;
    ShowWindow(native.hwnd, state.window_maximized && show != SW_HIDE ? SW_SHOWMAXIMIZED : show);
    unsigned frames = 0;
    bool limit_exit = false;
    while (!quit) {
        const auto started = GetTickCount64();
        MSG message{};
        while (PeekMessageW(&message, nullptr, 0, 0, PM_REMOVE)) {
            if (message.message == WM_QUIT) quit = true;
            TranslateMessage(&message); DispatchMessageW(&message);
        }
        if (quit) break;
        app.tick();
        auto mood = ecmgui_v2::pet::Mood::Calm;
        for (const auto& worker : app.controller.workers()) {
            if (worker.state() == ecmgui::WorkerRunState::Starting || worker.state() == ecmgui::WorkerRunState::Restarting) { mood = ecmgui_v2::pet::Mood::Waiting; break; }
            if (worker.state() == ecmgui::WorkerRunState::Error) mood = ecmgui_v2::pet::Mood::Error;
            else if (worker.running() && mood != ecmgui_v2::pet::Mood::Error) mood = ecmgui_v2::pet::Mood::Working;
            else if (worker.state() == ecmgui::WorkerRunState::QueueEmpty && mood == ecmgui_v2::pet::Mood::Calm) mood = ecmgui_v2::pet::Mood::Done;
        }
        if (options.audit_app && app.pet_test_mood >= 0) mood = static_cast<ecmgui_v2::pet::Mood>(app.pet_test_mood);
        companion.tick(mood);
        if (!companion.error.empty()) app.status = companion.error;
        if (close_requested) { close_requested = false; if (app.request_close()) quit = true; }
        if (app.done()) quit = true;
        if (quit) break;
        const auto new_dpi = GetDpiForWindow(native.hwnd);
        if (new_dpi) dpi = new_dpi;
        if (!show_components) pending.scale = state.ui_scale > 0 ? state.ui_scale : dpi / 96.f;
        ImGui::GetIO().ConfigDpiScaleFonts = !show_components && state.ui_scale == 0;
        RECT client{}; GetClientRect(native.hwnd, &client);
        if (IsIconic(native.hwnd) || client.right <= 0 || client.bottom <= 0) {
            smooth_scroll.cancel();
            MsgWaitForMultipleObjectsEx(0, nullptr, 100, QS_ALLINPUT, MWMO_INPUTAVAILABLE);
            continue;
        }
        const bool font_change = active.chinese != pending.chinese || active.scale != pending.scale;
        if (font_change) { fonts.reload(pending.chinese, options.missing_cjk, log); pending.chinese = fonts.chinese; }
        if (resized || renderer.needs_rebuild) {
            renderer.rebuild(static_cast<uint32_t>(client.right), static_cast<uint32_t>(client.bottom));
            resized = false;
        }
        if (font_change || active.light != pending.light) {
            active = pending;
            ecmgui_v2::ui::apply_theme(active);
            const BOOL dark = !active.light; DwmSetWindowAttribute(native.hwnd, 20, &dark, sizeof(dark));
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
            POINT point{static_cast<LONG>(audit_position.x), static_cast<LONG>(audit_position.y)}; ClientToScreen(native.hwnd, &point);
            io.AddMousePosEvent(static_cast<float>(point.x), static_cast<float>(point.y));
            if (audit_button >= 0) { io.AddMouseButtonEvent(0, audit_button != 0); audit_button = -1; }
        }
        const auto scroll_now=GetTickCount64();double wheel_x=0,wheel_y=0;
        smooth_scroll.step(scroll_tick?(scroll_now-scroll_tick)/1000.:.016,wheel_x,wheel_y);scroll_tick=scroll_now;
        if(wheel_x||wheel_y)ImGui::GetIO().AddMouseWheelEvent(static_cast<float>(wheel_x),static_cast<float>(wheel_y));
        ImGui::NewFrame();
        if (show_components) showcase.draw(pending, active, fonts, context.properties.deviceName);
        else app.draw(pending, active, fonts);
        ImGui::Render();
        if (renderer.render(ImGui::GetDrawData())) {
            ++frames;
            if (frames == 1 || frames % 20 == 0) { std::fprintf(log, "frame: %u\n", frames); std::fflush(log); }
        }
        ImGui::UpdatePlatformWindows(); ImGui::RenderPlatformWindowsDefault();
        if (frames && frames % 20 == 0 && options.audit_app) {
            std::fprintf(log, "app_snapshot: frames=%u running=%d page=%d viewports=%d close=%d config_dirty=%d probe_dpi=%.2f probe_font=%.2f probe_padding=%.2f\n", frames, app.running_count(), state.ui_page, ImGui::GetPlatformIO().Viewports.Size, static_cast<int>(app.controller.exit_phase()), app.config.dirty(),app.probe_dpi,app.probe_font,app.probe_padding);
            std::fflush(log);
        }
        if (options.audit_ui && audit_sequence != sampled_sequence) {
            if (++sample_frames == 4) {
                unsigned textures = 0;
                for (auto texture : ImGui::GetPlatformIO().Textures) if (texture->BackendUserData) ++textures;
                POINT primary{static_cast<LONG>(showcase.primary_center.x), static_cast<LONG>(showcase.primary_center.y)};
                POINT field{static_cast<LONG>(showcase.field_center.x), static_cast<LONG>(showcase.field_center.y)};
                ScreenToClient(native.hwnd, &primary); ScreenToClient(native.hwnd, &field);
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
        if (options.frames && frames >= static_cast<unsigned>(options.frames) && !limit_exit) { limit_exit = true; app.begin_exit(); }
        const auto elapsed = GetTickCount64() - started;
        const unsigned period=smooth_scroll.active()?16:100;
        if (elapsed < period) MsgWaitForMultipleObjectsEx(0, nullptr, static_cast<DWORD>(period - elapsed), QS_ALLINPUT, MWMO_INPUTAVAILABLE);
    }
    state.ui_theme = active.light ? "light" : "dark"; state.ui_language = fonts.chinese ? "zh" : "en";
    if (show_components) state.ui_scale = active.scale;
    if (!show_components) { size_t size = 0; const auto layout = ImGui::SaveIniSettingsToMemory(&size); state.layout(std::string(layout, size)); }
    if (!IsIconic(native.hwnd)) {
        state.window_maximized = IsZoomed(native.hwnd) != FALSE;
        if (!state.window_maximized) { RECT rect{}; GetWindowRect(native.hwnd, &rect); state.window_x = rect.left; state.window_y = rect.top; state.window_width = rect.right - rect.left; state.window_height = rect.bottom - rect.top; }
    }
    if (!state.save()) app.trace(state.notice);
    current_app = nullptr; audit_app = false;
    renderer.shutdown();
    audit_settings = nullptr;
    if (options.audit_ui) {
        std::fprintf(log, "memory_audit_end: bytes=%llu allocations=%zu\n", static_cast<unsigned long long>(memory.bytes()), memory.allocations());
        if (memory.allocations()) throw std::runtime_error("Vulkan allocation audit found unreleased GUI memory");
    }
    std::fprintf(log, "exit: frames=%u rebuilds=%u worker_processes=%d\n", frames, renderer.rebuilds, app.running_count());
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
            else if (argument == L"--showcase") options.showcase = true;
            else if (argument == L"--no-nvml") options.no_nvml = true;
            else if (argument == L"--detach-probe") options.detach_probe = true;
            else if (argument == L"--audit-app") options.audit_app = true;
            else if (argument == L"--pet") options.pet = 1;
            else if (argument == L"--no-pet") options.pet = 0;
            else if (argument == L"--pet-static") options.pet_static = true;
            else if ((argument == L"--diagnostics" || argument == L"--frames" || argument == L"--inject-failure" ||
                      argument == L"--theme" || argument == L"--language" || argument == L"--ui-scale" || argument == L"--config" || argument == L"--ui-state" || argument == L"--start-worker") && i + 1 < args.size()) {
                const auto value = args[++i];
                if (argument == L"--diagnostics") options.diagnostics = value;
                else if (argument == L"--config") options.config = std::filesystem::absolute(value).u8string();
                else if (argument == L"--ui-state") options.ui_state = value;
                else if (argument == L"--start-worker") {
                    size_t used = 0; options.start_worker = std::stoi(value, &used);
                    if (used != value.size() || options.start_worker < 1 || options.start_worker > 1024) throw std::runtime_error("Invalid --start-worker");
                }
                else if (argument == L"--theme") {
                    if (value != L"dark" && value != L"light") throw std::runtime_error("--theme requires dark or light");
                    options.settings.light = value == L"light";
                    options.explicit_theme = true;
                } else if (argument == L"--language") {
                    if (value != L"en" && value != L"zh") throw std::runtime_error("--language requires en or zh");
                    options.settings.chinese = value == L"zh";
                    options.explicit_language = true;
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
