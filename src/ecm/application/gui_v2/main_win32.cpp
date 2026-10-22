#include "version.h"
#include "render/render_host.h"
#include "ui/widgets.h"
#include "app.h"
#include "pet/window.h"
#include "generated/pet_skins.h"
#include "ui/scroll.h"
#include "ui/worker_presentation.h"
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
ecmgui_v2::pet::Window* current_pet = nullptr;
ecmgui_v2::ui::Settings* audit_settings = nullptr;
unsigned audit_sequence = 0;
bool audit_pointer = false;
ImVec2 audit_position{};
int audit_button = -1;
ecmgui_v2::ui::SmoothScroll smooth_scroll;
unsigned long long scroll_tick=0;
unsigned performance_sequence=0;
void (*platform_create_window)(ImGuiViewport*)=nullptr;
bool collect_wheel(UINT message,WPARAM w){
    if(!ImGui::GetCurrentContext()||(message!=WM_MOUSEWHEEL&&message!=WM_MOUSEHWHEEL))return false;
    if(current_app&&!current_app->state.ui_smooth_scroll)return false;
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
    bool performance = false;
    bool trace_render=false;
    bool vulkan=false;
    std::string pet_skin;
    int external_windows=-1;
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
    case WM_NCHITTEST: {
        const auto hit=DefWindowProcW(hwnd,message,w,l);
        if(hit==HTCLIENT&&current_app&&!current_app->state.window_title_bar){
            POINT p{static_cast<short>(LOWORD(l)),static_cast<short>(HIWORD(l))};
            if(!(ImGui::GetIO().ConfigFlags&ImGuiConfigFlags_ViewportsEnable))ScreenToClient(hwnd,&p);
            const auto lo=current_app->title_drag_min,hi=current_app->title_drag_max;
            if(p.x>=lo.x&&p.x<hi.x&&p.y>=lo.y&&p.y<hi.y)return HTCAPTION;
        }
        return hit;
    }
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
            if (w == 10) performance_sequence=static_cast<unsigned>(l);
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
    case WM_WINDOWPOSCHANGED: if(current_pet)current_pet->follow_owner(); break;
    case WM_SIZE: resized = true;if(current_pet)current_pet->follow_owner();return 0;
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
        hwnd = CreateWindowExW(0, wc.lpszClassName, title.c_str(),state.window_title_bar?WS_OVERLAPPEDWINDOW:WS_OVERLAPPEDWINDOW&~WS_CAPTION,
            visible ? state.window_x : CW_USEDEFAULT, visible ? state.window_y : CW_USEDEFAULT, state.window_width, state.window_height, nullptr, nullptr, instance, nullptr);
        if (!hwnd) throw std::runtime_error("CreateWindowExW failed");
    }
    void title_bar(bool visible) {
        const auto style=GetWindowLongPtrW(hwnd,GWL_STYLE);
        if(visible!=!!(style&WS_CAPTION)){
            SetWindowLongPtrW(hwnd,GWL_STYLE,visible?style|WS_CAPTION:style&~WS_CAPTION);
            SetWindowPos(hwnd,nullptr,0,0,0,0,SWP_NOMOVE|SWP_NOSIZE|SWP_NOZORDER|SWP_NOACTIVATE|SWP_FRAMECHANGED);
        }
    }
};
struct ImGuiSession {
    explicit ImGuiSession(HWND hwnd,bool single_viewport) {
        IMGUI_CHECKVERSION(); ImGui::CreateContext();
        ImGui::GetIO().IniFilename = nullptr;
        ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_DockingEnable | ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_ViewportsEnable;
        if(single_viewport)ImGui::GetIO().ConfigFlags&=~ImGuiConfigFlags_ViewportsEnable;
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
    if(!options.pet_skin.empty())state.pet_skin=options.pet_skin;
    if(options.external_windows>=0)state.ui_external_windows=options.external_windows!=0;
    SYSTEMTIME utc{};GetSystemTime(&utc);
    std::fprintf(log,"render_session: utc=%04u-%02u-%02uT%02u:%02u:%02u.%03uZ ticks=%llu external_windows=%d trace=%d\n",utc.wYear,utc.wMonth,utc.wDay,utc.wHour,utc.wMinute,utc.wSecond,utc.wMilliseconds,GetTickCount64(),state.ui_external_windows,options.trace_render);std::fflush(log);
    NativeWindow native; native.create(state);
    ImGuiSession imgui(native.hwnd,!state.ui_external_windows);
    ecmgui_v2::ui::Fonts fonts;
    if(!options.vulkan&&(options.validation||options.trace_render||options.no_loader||!options.failure.empty()))
        throw std::runtime_error("Vulkan diagnostics require --renderer vulkan");
    ecmgui_v2::RenderHost renderer(native.hwnd,options.vulkan,options.validation,options.no_loader,options.trace_render,options.audit_ui||options.performance,log);
    auto& memory=renderer; // Explicit Vulkan allocations only; DX11 reports that scope as unavailable.
    auto pending = options.settings;
    pending.font_size=static_cast<float>(state.ui_font_size);
    if (!options.explicit_theme) pending.light = state.ui_theme == "light";
    if (!options.explicit_language) pending.chinese = state.ui_language == "zh";
    if (!options.explicit_scale && state.ui_scale > 0) pending.scale = state.ui_scale;
    UINT dpi = GetDpiForWindow(native.hwnd);
    if (!options.explicit_scale && state.ui_scale == 0) pending.scale = dpi / 96.f;
    auto active = pending;
    fonts.base_size=active.font_size;
    fonts.reload(active.chinese, options.missing_cjk, log);
    state.ui_language=fonts.chinese?"zh":"en";
    active.chinese = pending.chinese = fonts.chinese;
    ecmgui_v2::ui::apply_theme(active);
    ecmgui_v2::ui::Showcase showcase;
    const bool show_components = options.showcase || options.audit_ui;
    ecmgui_v2::App app(state, log); current_app = &app; audit_app = options.audit_app;
    struct ResetPointers { ~ResetPointers() { current_app = nullptr; audit_app = false; audit_settings = nullptr; } } reset_pointers;
    ecmgui_v2::pet::Window companion(native.hwnd, state, log);
    current_pet=&companion;
    struct ResetPetPointer{~ResetPetPointer(){current_pet=nullptr;}} reset_pet_pointer;
    app.initialize(options.config, options.no_nvml || show_components); app.test_float = options.detach_probe;
    if (!state.notice.empty()) app.trace(state.notice);
    if(!fonts.notice.empty()){app.status=fonts.notice;app.trace(fonts.notice);}
    try { const auto layout = state.layout(); if (!layout.empty() && !show_components) ImGui::LoadIniSettingsFromMemory(layout.data(), layout.size()); }
    catch (const std::exception& error) { app.trace(error.what()); app.reset_layout = true; }
    if (options.start_worker > 0) app.start(options.start_worker);
    if (options.audit_ui||options.audit_app) audit_settings = &pending;
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
        if (close_requested) { close_requested = false; if (app.request_close()) quit = true;
            else if(IsIconic(native.hwnd))ShowWindow(native.hwnd,SW_RESTORE);
        }
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
        if (font_change) { fonts.reload(pending.chinese, options.missing_cjk, log); pending.chinese = fonts.chinese;
            if(!show_components)state.ui_language=fonts.chinese?"zh":"en";
            if(!fonts.notice.empty()){app.toast(fonts.notice,true);app.trace(fonts.notice);}
        }
        if (resized || renderer.needs_rebuild()) {
            renderer.rebuild(static_cast<uint32_t>(client.right), static_cast<uint32_t>(client.bottom));
            resized = false;
        }
        if (font_change || active.light != pending.light || active.font_size != pending.font_size) {
            active = pending;
            fonts.base_size=active.font_size;
            ecmgui_v2::ui::apply_theme(active);
            const BOOL dark = !active.light; DwmSetWindowAttribute(native.hwnd, 20, &dark, sizeof(dark));
        }
        const auto background = ecmgui_v2::ui::palette(active.light).bg;
        const float clear_color[]{background.x,background.y,background.z,1};
        if (renderer.needs_rebuild()) {
            MsgWaitForMultipleObjectsEx(0, nullptr, 100, QS_ALLINPUT, MWMO_INPUTAVAILABLE);
            continue;
        }
        LARGE_INTEGER draw_begin{},frequency{};if(options.performance){QueryPerformanceCounter(&draw_begin);QueryPerformanceFrequency(&frequency);}
        renderer.new_frame(); ImGui_ImplWin32_NewFrame();
        if ((options.audit_ui||options.audit_app) && audit_pointer) {
            auto& io = ImGui::GetIO();
            io.AddFocusEvent(true);
            io.AddMouseViewportEvent(ImGui::GetMainViewport()->ID);
            POINT point{static_cast<LONG>(audit_position.x), static_cast<LONG>(audit_position.y)};
            if(io.ConfigFlags&ImGuiConfigFlags_ViewportsEnable)ClientToScreen(native.hwnd, &point);
            io.AddMousePosEvent(static_cast<float>(point.x), static_cast<float>(point.y));
            if (audit_button >= 0) { io.AddMouseButtonEvent(0, audit_button != 0); audit_button = -1; }
        }
        const auto scroll_now=GetTickCount64();double wheel_x=0,wheel_y=0;
        smooth_scroll.step(scroll_tick?(scroll_now-scroll_tick)/1000.:.016,wheel_x,wheel_y);scroll_tick=scroll_now;
        if(wheel_x||wheel_y)ImGui::GetIO().AddMouseWheelEvent(static_cast<float>(wheel_x),static_cast<float>(wheel_y));
        ImGui::NewFrame();
        if (show_components) showcase.draw(pending, active, fonts, renderer.device_name());
        else app.draw(pending, active, fonts);
        native.title_bar(state.window_title_bar);
        ImGui::Render();
        LARGE_INTEGER ui_end{};if(options.performance)QueryPerformanceCounter(&ui_end);
        const bool submitted=renderer.render(ImGui::GetDrawData(),clear_color);
        if (submitted) {
            ++frames;
            if (frames == 1 || frames % 20 == 0) { std::fprintf(log, "frame: %u\n", frames); std::fflush(log); }
        }
        ImGui::UpdatePlatformWindows(); ImGui::RenderPlatformWindowsDefault();
        if(options.audit_app&&!options.performance){
            const auto point=[&](ImVec2 v){POINT p{static_cast<LONG>(v.x),static_cast<LONG>(v.y)};if(ImGui::GetIO().ConfigFlags&ImGuiConfigFlags_ViewportsEnable)ScreenToClient(native.hwnd,&p);return p;};
            const auto f=point(app.result_chips[1]),nf=point(app.result_chips[2]),wheel=point(app.result_wheel),start=point(app.start_control),stop=point(app.stop_control);
            std::fprintf(log,"interaction_sample: page=%d visible=%zu scroll=%.3f toasts=%zu f=%ld,%ld nf=%ld,%ld wheel=%ld,%ld start=%ld,%ld stop=%ld,%ld\n",state.ui_page,app.result_indices.size(),app.result_scroll,app.toasts.size(),f.x,f.y,nf.x,nf.y,wheel.x,wheel.y,start.x,start.y,stop.x,stop.y);std::fflush(log);
            const auto drop=point(app.nav_drop),back=point(app.nav_return),english=point(app.language_items[0]),chinese=point(app.language_items[1]),save=point(app.save_ui_control),calculate=point(app.calculate_control);
            std::fprintf(log,"usability_sample: page=%d mask=%d right=%d shown=%d zh=%d charts=%d drop=%ld,%ld back=%ld,%ld en=%ld,%ld cn=%ld,%ld save=%ld,%ld calculate=%ld,%ld",state.ui_page,state.ui_right_pages,state.ui_right_page,state.ui_right_visible,fonts.chinese,app.probability_ready,drop.x,drop.y,back.x,back.y,english.x,english.y,chinese.x,chinese.y,save.x,save.y,calculate.x,calculate.y);
            for(int i=0;i<7;++i){const auto nav=point(app.nav_items[i]),right=point(app.right_items[i]);std::fprintf(log," nav%d=%ld,%ld tool%d=%ld,%ld",i,nav.x,nav.y,i,right.x,right.y);}
            const auto search=point(app.settings_search_control);std::fprintf(log," matches=%d search=%ld,%ld\n",app.settings_matches,search.x,search.y);std::fflush(log);
            std::fprintf(log,"navigation_sample: order=%s dragged=%d rounding=%.1f height=%.1f gap=%.1f tools=%d right_count=%zu",state.ui_nav_order.c_str(),app.nav_drag_page,app.nav_rounding,app.nav_item_height,app.nav_item_gap,app.right_container_present,app.right_docks.size());
            for(const auto& tab:app.right_close_controls){const auto pos=point(tab.second);std::fprintf(log," close%d=%ld,%ld",tab.first,pos.x,pos.y);}
            for(const auto& tab:app.right_docks)std::fprintf(log," dock%d=%u",tab.first,tab.second);std::fprintf(log,"\n");
            std::fprintf(log,"pet_skin_sample: selected=%s symbols=%d",state.pet_skin.c_str(),fonts.body->IsGlyphInFont(0x266a)&&fonts.body->IsGlyphInFont(0x2665));
            for(int i=0;i<4;++i){const auto skin=point(app.pet_skin_controls[i]);std::fprintf(log," skin%d=%ld,%ld",i,skin.x,skin.y);}std::fprintf(log,"\n");std::fflush(log);
            const auto raw=point(app.log_raw_control),follow=point(app.log_follow_control),drag=point({(app.title_drag_min.x+app.title_drag_max.x)*.5f,(app.title_drag_min.y+app.title_drag_max.y)*.5f});
            std::fprintf(log,"presentation_sample: selected=%d log_worker=%d raw=%d follow=%d scroll=%.1f title=%d raw_button=%ld,%ld follow_button=%ld,%ld drag=%ld,%ld",state.ui_worker,app.log_active_worker,app.log_current_raw,app.log_current_follow,app.log_scroll,state.window_title_bar,raw.x,raw.y,follow.x,follow.y,drag.x,drag.y);
            for(const auto& tab:app.log_tab_controls){const auto pos=point(tab.second);std::fprintf(log," log%d=%ld,%ld",tab.first,pos.x,pos.y);}
            for(const auto& row:app.worker_select_controls){if(row.first>4)break;const auto pos=point(row.second);std::fprintf(log," worker%d=%ld,%ld",row.first,pos.x,pos.y);}
            for(const auto& row:app.worker_action_controls){if(row.first>4)break;const auto pos=point(row.second);std::fprintf(log," action%d=%ld,%ld",row.first,pos.x,pos.y);}
            for(size_t i=0;i<app.worker_summary_rects.size()&&i<3;++i){const auto a=point(app.worker_summary_rects[i].first),b=point(app.worker_summary_rects[i].second);std::fprintf(log," summary%zu=%ld,%ld,%ld,%ld",i,a.x,a.y,b.x,b.y);}
            std::fprintf(log," speed_samples=%zu line_pitch=%.2f font_height=%.2f log_height=%.2f",app.log_speed_samples,app.log_line_pitch,app.log_font_height,app.log_view_height);
            std::fprintf(log," table_scroll_y=%.2f table_height=%.2f table_content=%.2f body_font=%.2f body_glyph=%.2f log_glyph=%.2f log_body=%d",app.worker_table_scroll_max_y,app.worker_table_height,app.worker_table_content_height,app.body_font_height,app.body_glyph_height,app.log_glyph_height,app.log_uses_body);
            std::fprintf(log," table_required=%.2f table_data_height=%.2f table_rows=%zu",app.worker_table_required_height,app.worker_table_data_height,app.worker_table_rows);
            const auto font_control=point(app.font_size_control),speed_min=point(app.worker_speed_min),speed_max=point(app.worker_speed_max);
            std::fprintf(log," workspace_speed=%zu workspace_series=%zu speed_min=%ld,%ld speed_max=%ld,%ld font_size=%d font_base=%.1f font_control=%ld,%ld exit_phase=%d deadline=%llu",app.worker_speed_samples,app.workspace_chart_count,speed_min.x,speed_min.y,speed_max.x,speed_max.y,state.ui_font_size,ImGui::GetStyle().FontSizeBase,font_control.x,font_control.y,static_cast<int>(app.controller.exit_phase()),app.controller.exit_deadline());
            for(int i=0;i<3;++i){const auto pos=point(app.exit_controls[i]);std::fprintf(log," exit%d=%ld,%ld",i,pos.x,pos.y);}
            for(int i=0;i<3;++i){const auto pos=point(app.title_controls[i]);std::fprintf(log," chrome%d=%ld,%ld",i,pos.x,pos.y);}std::fprintf(log,"\n");std::fflush(log);
            if(const auto* worker=app.selected()){const auto display=ecmgui_v2::ui::worker_progress(*worker,false);const auto batch=ecmgui_v2::ui::stage2_batch(worker->stage2_status,false);std::fprintf(log,"log_detail: worker=%d table_eta=%s | log_eta=%s | shape=%s | batch=%s | remaining=%s\n",worker->index,display.eta.c_str(),display.eta_detail.c_str(),ecmgui_v2::ui::worker_shape(*worker).c_str(),batch.label.c_str(),batch.remaining.c_str());}
            std::fprintf(log,"gpu_settings: seconds=%d",state.ui_gpu_history_seconds);
            for(int i=0;i<4;++i){const auto pos=point(app.gpu_range_controls[i]);std::fprintf(log," range%d=%ld,%ld",i,pos.x,pos.y);}std::fprintf(log,"\n");
            for(const auto& device:app.gpu.devices()){
                const auto found=app.gpu_presentations.find(device.index);if(found==app.gpu_presentations.end())continue;
                const auto& display=found->second;const auto details=app.gpu_details.get(device.index);const auto pos=point(display.header);
                std::fprintf(log,"gpu_detail: index=%d ready=%d driver=%s cuda=%d major=%d minor=%d sms=%d cores=%d charts=%zu hovered=%d header=%ld,%ld\n",device.index,app.gpu_details.ready(),details.driver.c_str(),details.cuda_version,details.major,details.minor,details.sm_count,device.cores,display.charts,display.hovered,pos.x,pos.y);
                const auto one_line=[](std::string text){std::replace(text.begin(),text.end(),'\n','\t');return text;};
                std::fprintf(log,"gpu_text: index=%d tooltip=%s\n",device.index,one_line(display.tooltip).c_str());
                std::fprintf(log,"gpu_text: index=%d basic=%s\n",device.index,one_line(display.basic).c_str());
                std::fprintf(log,"gpu_text: index=%d usage=%s | power=%s | clocks=%s\n",device.index,display.usage.c_str(),display.power.c_str(),display.clocks.c_str());
            }
        }
        if(options.performance){
            LARGE_INTEGER draw_end{};QueryPerformanceCounter(&draw_end);
            std::fprintf(log,"layout_sample: page=%d calculated=%d docked=%d gpu=%d external=%d\n",state.ui_page,app.probability_ready,app.workspace_docked,app.gpu.available(),!!(ImGui::GetIO().ConfigFlags&ImGuiConfigFlags_ViewportsEnable));
            std::fprintf(log,"frame_sample: ms=%.4f gpu_bytes=%llu allocations=%zu gdi=%lu ui_ms=%.4f render_ms=%.4f submitted=%d\n",1000.*(draw_end.QuadPart-draw_begin.QuadPart)/frequency.QuadPart,static_cast<unsigned long long>(memory.bytes()),memory.allocations(),GetGuiResources(GetCurrentProcess(),GR_GDIOBJECTS),1000.*(ui_end.QuadPart-draw_begin.QuadPart)/frequency.QuadPart,1000.*(draw_end.QuadPart-ui_end.QuadPart)/frequency.QuadPart,submitted);
            if(performance_sequence&&submitted){std::fprintf(log,"response_sample: seq=%u ticks=%llu\n",performance_sequence,GetTickCount64());performance_sequence=0;}
            std::fflush(log);
        }
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
                if(ImGui::GetIO().ConfigFlags&ImGuiConfigFlags_ViewportsEnable){ScreenToClient(native.hwnd, &primary); ScreenToClient(native.hwnd, &field);}
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
    if (options.audit_ui||options.performance) {
        std::fprintf(log, "memory_audit_end: bytes=%llu allocations=%zu\n", static_cast<unsigned long long>(memory.bytes()), memory.allocations());
        if (memory.allocations()) throw std::runtime_error("Vulkan allocation audit found unreleased GUI memory");
    }
    std::fprintf(log, "exit: frames=%u rebuilds=%u worker_processes=%d\n", frames, renderer.rebuilds(), app.running_count());
    std::fflush(log);
    return renderer.validation_errors() ? 3 : 0;
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
            if (argument == L"--version") { console(std::string("ECM-ELF GUI v2 ") + ecmgui_v2::version + " (DX11 default; Vulkan optional)\n"); return 0; }
            if (argument == L"--validation") options.validation = true;
            else if (argument == L"--no-loader") options.no_loader = true;
            else if (argument == L"--audit-ui") options.audit_ui = true;
            else if (argument == L"--simulate-missing-cjk") options.missing_cjk = true;
            else if (argument == L"--showcase") options.showcase = true;
            else if (argument == L"--no-nvml") options.no_nvml = true;
            else if (argument == L"--detach-probe") options.detach_probe = true;
            else if (argument == L"--audit-app") options.audit_app = true;
            else if (argument == L"--audit-performance") {options.performance=true;options.audit_app=true;}
            else if (argument == L"--trace-render") options.trace_render=true;
            else if (argument == L"--single-viewport") options.external_windows=0;
            else if (argument == L"--multi-viewport") options.external_windows=1;
            else if (argument == L"--renderer" && i+1<args.size()) {
                const auto value=args[++i];
                if(value==L"dx11")options.vulkan=false;
                else if(value==L"vulkan")options.vulkan=true;
                else throw std::runtime_error("--renderer expects dx11 or vulkan");
            }
            else if (argument == L"--pet") options.pet = 1;
            else if (argument == L"--no-pet") options.pet = 0;
            else if (argument == L"--pet-static") options.pet_static = true;
            else if(argument==L"--pet-skin"&&i+1<args.size()){
                options.pet_skin=utf8(args[++i].c_str());
                if(!ecmgui_v2::pet::find_skin(options.pet_skin))throw std::runtime_error("Unknown ELY skin: "+options.pet_skin);
            }
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
        if(options.config.empty()) {
            const auto adjacent=std::filesystem::u8path(ecmgui::default_toml_path());
            std::error_code error;
            if(std::filesystem::is_regular_file(adjacent,error))options.config=std::filesystem::weakly_canonical(adjacent).u8string();
        }
        log = _wfsopen(options.diagnostics.c_str(), L"w", _SH_DENYWR);
        if (!log) throw std::runtime_error("Cannot open GUI v2 diagnostics file");
        std::fprintf(log, "ECM-ELF GUI v2 %s backend=%s\n", ecmgui_v2::version,options.vulkan?"Vulkan":"DX11");
        const int result = run(options, show, log);
        std::fclose(log); return result;
    } catch (const std::exception& error) {
        if (log) { std::fprintf(log, "error: %s\n", error.what()); std::fclose(log); }
        console(std::string("GUI v2: ") + error.what() + "\n");
        return 1;
    }
}
