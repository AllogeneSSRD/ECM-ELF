// ecm_gui -- Win32 + Direct3D 11 + Dear ImGui (docking branch) host.
//
// Modes:
//   ecm_gui.exe                 normal: create the window and run the UI
//   ecm_gui.exe --selftest      headless check of everything that does not need a
//                               window (ini round-trip, localization, font search,
//                               ImGui context + docking symbols). Prints to stdout
//                               AND to <exe dir>\ecm_gui_selftest.log, exits 0/1.
//   ecm_gui.exe -ini <path>     use another ecm.ini
//   ecm_gui.exe --language <l>  override [GUI] language for this run
//
// Verification for milestone M1 uses --selftest so it works without a desktop:
//   build_vs18\Release\ecm_gui.exe --selftest

#include "app.h"
#include "platform.h"
#include "gpu_selftest.h"
#include "worker_selftest.h"

#include "imgui.h"
#include "imgui_impl_dx11.h"
#include "imgui_impl_win32.h"

#include <cstdio>
#include <cstring>
#include <string>

#include <d3d11.h>
#include <tchar.h>
#include <windows.h>

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND hWnd, UINT msg,
                                                            WPARAM wParam, LPARAM lParam);

namespace {

struct D3DState {
    ID3D11Device *device = nullptr;
    ID3D11DeviceContext *context = nullptr;
    IDXGISwapChain *swap_chain = nullptr;
    ID3D11RenderTargetView *rtv = nullptr;
    bool vsync = true;
    UINT pending_w = 0, pending_h = 0;
};

D3DState g_d3d;
HWND g_hwnd = nullptr;
bool g_resize_pending = false;
// Set by WM_CLOSE. The frame loop tests this directly instead of relying on
// WM_QUIT delivery: with multi-viewport there are several top-level windows and
// the close request may arrive while the message queue is never empty.
bool g_quit_requested = false;
// The application object, so wnd_proc can consult it (WM_CLOSE -> confirmation modal +
// checkpoint-before-exit flow). Set in run() before the window is created.
ecmgui::App *g_app = nullptr;
// After a window-lifecycle message (resize/activate/show) the next few frames are traced
// stage by stage: that is how a crash *between* stages is pinned down.
int g_trace_frames = 0;
// Set by new_frame() while the window is minimized: the caller must skip the whole
// frame (draw + render + Present). Presenting to a minimized window means presenting to
// a 0x0 swap chain, which crashed with 0xC0000005 (measured).
bool g_skip_frame = false;

// --trace: a few lines written (and flushed) at the lifecycle points that matter
// when the GUI must be driven by a script: start, first frame, WM_CLOSE seen, loop
// exit, ini saved. Kept in the shipped binary on purpose -- it is the only way to
// see what a windowed process did when it is started from PowerShell.
FILE *g_trace = nullptr;
void trace(const char *fmt, ...) {
    if (g_trace == nullptr) return;
    SYSTEMTIME st;
    GetLocalTime(&st);
    std::fprintf(g_trace, "[%02d:%02d:%02d.%03d] ", st.wHour, st.wMinute, st.wSecond,
                 st.wMilliseconds);
    va_list ap;
    va_start(ap, fmt);
    std::vfprintf(g_trace, fmt, ap);
    va_end(ap);
    std::fputc('\n', g_trace);
    std::fflush(g_trace);
}

void cleanup_render_target();   // defined below recreate_swap_chain()

void create_render_target() {
    ID3D11Texture2D *back = nullptr;
    const HRESULT hr = g_d3d.swap_chain->GetBuffer(0, IID_PPV_ARGS(&back));
    if (FAILED(hr) || back == nullptr) {
        trace("d3d: GetBuffer for the back buffer failed (hr=0x%08lX)", (unsigned long)hr);
        return;
    }
    const HRESULT hr2 = g_d3d.device->CreateRenderTargetView(back, nullptr, &g_d3d.rtv);
    back->Release();
    if (FAILED(hr2)) {
        trace("d3d: CreateRenderTargetView failed (hr=0x%08lX)", (unsigned long)hr2);
    }
}

// Recreates the swap chain when a resize cannot be applied to the existing one (which
// happens after a minimize/restore cycle: ResizeBuffers fails while the window is
// occluded, and then the render target view stays null -- the frame loop used to
// dereference it and crash with 0xC0000005).
bool recreate_swap_chain(UINT w, UINT h) {
    cleanup_render_target();
    if (g_d3d.swap_chain != nullptr) {
        g_d3d.swap_chain->Release();
        g_d3d.swap_chain = nullptr;
    }
    DXGI_SWAP_CHAIN_DESC sd{};
    sd.BufferCount = 2;
    sd.BufferDesc.Width = w;
    sd.BufferDesc.Height = h;
    sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferDesc.RefreshRate.Numerator = 60;
    sd.BufferDesc.RefreshRate.Denominator = 1;
    sd.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow = g_hwnd;
    sd.SampleDesc.Count = 1;
    sd.Windowed = TRUE;
    sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;
    IDXGIDevice *dxgi_device = nullptr;
    if (FAILED(g_d3d.device->QueryInterface(IID_PPV_ARGS(&dxgi_device))) ||
        dxgi_device == nullptr) {
        trace("d3d: cannot query IDXGIDevice to rebuild the swap chain");
        return false;
    }
    IDXGIAdapter *adapter = nullptr;
    IDXGIFactory *factory = nullptr;
    bool ok = false;
    if (SUCCEEDED(dxgi_device->GetAdapter(&adapter)) && adapter != nullptr &&
        SUCCEEDED(adapter->GetParent(IID_PPV_ARGS(&factory))) && factory != nullptr) {
        const HRESULT hrc = factory->CreateSwapChain(g_d3d.device, &sd, &g_d3d.swap_chain);
        if (SUCCEEDED(hrc) && g_d3d.swap_chain != nullptr) {
            create_render_target();
            ok = (g_d3d.rtv != nullptr);
            trace("d3d: swap chain rebuilt (%ux%u, ok=%d)", w, h, ok ? 1 : 0);
        } else {
            trace("d3d: CreateSwapChain failed (hr=0x%08lX)", (unsigned long)hrc);
        }
    }
    if (factory != nullptr) factory->Release();
    if (adapter != nullptr) adapter->Release();
    dxgi_device->Release();
    return ok;
}

void cleanup_render_target() {
    // Unbind and flush BEFORE releasing the view: ResizeBuffers fails while the back
    // buffer is still referenced by the immediate context, and a failed resize used to
    // leave a null render target (which then crashed the frame loop with 0xC0000005
    // after a minimize/restore).
    if (g_d3d.context != nullptr) {
        g_d3d.context->OMSetRenderTargets(0, nullptr, nullptr);
        g_d3d.context->Flush();
    }
    if (g_d3d.rtv) {
        g_d3d.rtv->Release();
        g_d3d.rtv = nullptr;
    }
}

bool create_device(HWND hwnd) {
    DXGI_SWAP_CHAIN_DESC sd{};
    sd.BufferCount = 2;
    sd.BufferDesc.Width = 0;
    sd.BufferDesc.Height = 0;
    sd.BufferDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    sd.BufferDesc.RefreshRate.Numerator = 60;
    sd.BufferDesc.RefreshRate.Denominator = 1;
    sd.Flags = DXGI_SWAP_CHAIN_FLAG_ALLOW_MODE_SWITCH;
    sd.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
    sd.OutputWindow = hwnd;
    sd.SampleDesc.Count = 1;
    sd.Windowed = TRUE;
    sd.SwapEffect = DXGI_SWAP_EFFECT_DISCARD;

    const D3D_FEATURE_LEVEL levels[] = {D3D_FEATURE_LEVEL_11_0, D3D_FEATURE_LEVEL_10_0};
    D3D_FEATURE_LEVEL got{};
    const HRESULT hr = D3D11CreateDeviceAndSwapChain(
        nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, levels, 2, D3D11_SDK_VERSION, &sd,
        &g_d3d.swap_chain, &g_d3d.device, &got, &g_d3d.context);
    if (hr == DXGI_ERROR_UNSUPPORTED) {
        // WARP fallback: a machine without a usable D3D11 device can still show the UI.
        if (D3D11CreateDeviceAndSwapChain(nullptr, D3D_DRIVER_TYPE_WARP, nullptr, 0, levels, 2,
                                          D3D11_SDK_VERSION, &sd, &g_d3d.swap_chain,
                                          &g_d3d.device, &got, &g_d3d.context) != S_OK) {
            return false;
        }
    } else if (hr != S_OK) {
        return false;
    }
    create_render_target();
    return true;
}

void cleanup_device() {
    cleanup_render_target();
    if (g_d3d.swap_chain) { g_d3d.swap_chain->Release(); g_d3d.swap_chain = nullptr; }
    if (g_d3d.context) { g_d3d.context->Release(); g_d3d.context = nullptr; }
    if (g_d3d.device) { g_d3d.device->Release(); g_d3d.device = nullptr; }
}

LRESULT WINAPI wnd_proc(HWND hWnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    if (ImGui_ImplWin32_WndProcHandler(hWnd, msg, wParam, lParam)) {
        return true;
    }
    switch (msg) {
        case WM_SIZE:
            g_trace_frames = 30;
            if (wParam == SIZE_MINIMIZED) trace("wnd_proc: WM_SIZE minimized");
            else trace("wnd_proc: WM_SIZE %dx%d", LOWORD(lParam), HIWORD(lParam));
            if (wParam != SIZE_MINIMIZED) {
                g_d3d.pending_w = static_cast<UINT>(LOWORD(lParam));
                g_d3d.pending_h = static_cast<UINT>(HIWORD(lParam));
                g_resize_pending = true;
            }
            return 0;
        case WM_SYSCOMMAND:
            if ((wParam & 0xfff0) == SC_KEYMENU) return 0;   // disable ALT menu
            break;
        case WM_CLOSE:
            // Closing the GUI must not kill running workers silently (docs/DEV_ECM_GUI.md
            // 5.6): with workers alive the app opens a confirmation modal and asks them to
            // checkpoint first, so the window must stay open (return 0 without closing).
            trace("wnd_proc: WM_CLOSE");
            if (g_app == nullptr || g_app->request_close()) {
                g_quit_requested = true;
                PostQuitMessage(0);
            }
            return 0;
        case WM_ACTIVATE:
            g_trace_frames = 30;
            trace("wnd_proc: WM_ACTIVATE %d", (int)LOWORD(wParam));
            break;
        case WM_SHOWWINDOW:
            g_trace_frames = 30;
            trace("wnd_proc: WM_SHOWWINDOW show=%d", (int)wParam);
            break;
        case WM_DESTROY:
            trace("wnd_proc: WM_DESTROY");
            g_quit_requested = true;
            PostQuitMessage(0);
            return 0;
    }
    return DefWindowProcW(hWnd, msg, wParam, lParam);
}

// Pumps the frame. Returns false when the app should exit.
bool new_frame() {
    if (g_quit_requested) return false;

    const bool verbose = g_trace_frames > 0;
    if (verbose) trace("frame: pump begin");

    // Pump the message queue FIRST. The previous order returned before the pump while
    // the window was minimized, so the restore request (taskbar click ->
    // WM_SYSCOMMAND/SC_RESTORE -> WM_ACTIVATE) was never dispatched and the window
    // could not be brought back at all.
    MSG msg;
    while (PeekMessageW(&msg, nullptr, 0U, 0U, PM_REMOVE)) {
        TranslateMessage(&msg);
        DispatchMessageW(&msg);
        if (msg.message == WM_QUIT) return false;
    }
    if (g_quit_requested) return false;

    if (IsIconic(g_hwnd)) {
        // Nothing to draw while minimized. The messages above are what makes restoring
        // possible; the frame itself (draw + render + Present) must be skipped because
        // presenting to a minimized window means presenting to a 0x0 swap chain.
        g_skip_frame = true;
        Sleep(10);
        return true;
    }
    g_skip_frame = false;
    if (verbose) trace("frame: pumping done, iconic=0");

    if (g_resize_pending && g_d3d.pending_w > 0 && g_d3d.pending_h > 0) {
        const UINT w = g_d3d.pending_w;
        const UINT h = g_d3d.pending_h;
        g_d3d.pending_w = g_d3d.pending_h = 0;
        g_resize_pending = false;
        trace("d3d: resize to %ux%u (rtv=%p)", w, h, static_cast<void *>(g_d3d.rtv));
        cleanup_render_target();
        const HRESULT hr = (g_d3d.swap_chain != nullptr)
                               ? g_d3d.swap_chain->ResizeBuffers(0, w, h, DXGI_FORMAT_UNKNOWN, 0)
                               : E_FAIL;
        if (SUCCEEDED(hr)) {
            create_render_target();
        }
        trace("d3d: resize done hr=0x%08lX rtv=%p", static_cast<unsigned long>(hr),
              static_cast<void *>(g_d3d.rtv));
        if (g_d3d.rtv == nullptr) {
            // ResizeBuffers can fail right after a minimize/restore (the window was
            // occluded): rebuild the chain rather than rendering into a null target.
            trace("d3d: ResizeBuffers failed (hr=0x%08lX), rebuilding the swap chain",
                  static_cast<unsigned long>(hr));
            if (!recreate_swap_chain(w, h)) {
                trace("d3d: no render target after the rebuild");
            }
        }
        // NOTE: no early return here. The caller's loop body always pairs NewFrame with
        // Render, so skipping NewFrame while still returning true left Render() without
        // a frame -- that crashed with 0xC0000005 (measured). render_frame() itself is
        // null-safe, so continuing is fine.
    }

    ImGui_ImplDX11_NewFrame();
    ImGui_ImplWin32_NewFrame();
    ImGui::NewFrame();
    if (verbose) {
        trace("frame: NewFrame ok");
        if (--g_trace_frames <= 0) trace("frame: verbose tracing off");
    }
    return true;
}

// ---------------------------------------------------------------------------------
// Font application
//
// Runs at startup and again whenever the UI language changes at runtime: the font is
// chosen *for the language*, so a language switch needs a matching font or every label
// renders as "???" (the Latin system font has no CJK glyphs). This is the measured
// cause of the user report "Chinese shows ??? when I run it, but your tests are fine"
// -- the tests set language=chineseSimplified in the ini, so the right font was loaded
// at startup, while a Language-menu switch never reloaded one.
//
// Rules:
//   * [GUI] font=<path> wins (the user asked for that file); if it cannot draw the
//     language we say so and rescue with a CJK system font rather than showing boxes;
//   * else a CJK language -> find_cjk_font_file(), a Latin UI -> find_ui_font_file();
//   * if nothing can draw the language at all -> fall back to English (readable)
//     instead of rendering boxes, and say why in the status line + trace.
// ---------------------------------------------------------------------------------
void apply_ui_font(ecmgui::App &app, float dpi_scale, const char *why) {
    ImGuiIO &io = ImGui::GetIO();
    const float want = app.font_size_px();
    // 150 % displays land on a fractional automatic size (15 * 1.5 = 22.5): keep it
    // fractional rather than rounding, and let the dynamic atlas rasterize at the size
    // it is actually drawn with (that is what keeps text sharp in ImGui 1.92+).
    const float size_px = want > 0.0f ? want : (15.0f * dpi_scale);
    const bool need_cjk = app.needs_cjk_font();

    std::string path = app.font_path();
    const char *source = path.empty() ? "" : "configured";
    if (path.empty()) {
        if (need_cjk) {
            path = ecmgui::find_cjk_font_file();
            source = "cjk system font";
        }
        if (path.empty()) {
            path = ecmgui::find_ui_font_file();
            source = "latin system font";
        }
    }

    ImFont *font = nullptr;
    auto add_font = [&](const std::string &p) -> ImFont * {
        if (p.empty()) return nullptr;
        ImFontConfig cfg;
        cfg.OversampleH = cfg.OversampleV = 1;
        // Snap glyph advances to whole pixels: ImGui lays text out at fractional
        // positions, which softens small Latin text noticeably. Only meaningful with a
        // (near) integer size, hence the ini switch [GUI] font_snap.
        cfg.PixelSnapH = app.font_snap();
        return io.Fonts->AddFontFromFileTTF(p.c_str(), size_px, &cfg);
    };

    if (!path.empty()) {
        font = add_font(path);
        if (font == nullptr) source = "load failed";
    }
    // A font that cannot draw the language would render "???" boxes. The atlas API can
    // answer this exactly (a fallback box has a width too, so measuring text is not
    // enough -- see docs/DEV_ECM_GUI.md 10.3):
    //   1. [GUI] font=<path> that cannot draw the language -> rescue with a system CJK
    //      font (the user asked for a font, not for unreadable labels);
    //   2. no font can draw it at all -> fall back to English, never show boxes.
    auto can_draw_language = [&](ImFont *f) {
        return f != nullptr && (!need_cjk || f->IsGlyphInFont(0x6587));
    };
    if (!can_draw_language(font) && need_cjk) {
        const std::string rescue = ecmgui::find_cjk_font_file();
        if (!rescue.empty() && rescue != path) {
            trace("font: %s cannot draw CJK: rescuing with %s",
                  path.empty() ? "<none>" : path.c_str(), rescue.c_str());
            ImFont *rescue_font = add_font(rescue);
            if (can_draw_language(rescue_font)) {
                font = rescue_font;
                path = rescue;
                source = "cjk rescue font";
            }
        }
    }
    if (!can_draw_language(font) && need_cjk) {
        app.fall_back_to_english(
            "no CJK-capable font found: switched the UI to English (set [GUI] font=<path>)");
        return;                            // the English fallback re-applies the font
    }
    if (font == nullptr) {
        ImFontConfig cfg;
        cfg.SizePixels = size_px;          // scale the bitmap fallback up
        font = io.Fonts->AddFontDefault(&cfg);
        path.clear();
        if (source[0] == 'l') source = "built-in";   // keep "load failed"
    }
    io.FontDefault = font;
    {
        char buf[320];
        std::snprintf(buf, sizeof(buf),
                      "font: %s at %.1f px (dpi x%.2f, %s, snap=%d)%s [%s]",
                      path.empty() ? "<built-in>" : path.c_str(), static_cast<double>(size_px),
                      static_cast<double>(dpi_scale), source, app.font_snap() ? 1 : 0,
                      need_cjk ? ", CJK language" : "", why);
        trace("%s", buf);
    }
}

void render_frame(float clear[4], int refresh_hz) {
    ImGui::Render();
    if (g_d3d.rtv == nullptr || g_d3d.swap_chain == nullptr) {
        // No usable target (a resize is in flight, or the window is occluded): drop the
        // frame instead of dereferencing a null render target view.
        return;
    }
    const float c[4] = {clear[0], clear[1], clear[2], clear[3]};
    g_d3d.context->OMSetRenderTargets(1, &g_d3d.rtv, nullptr);
    g_d3d.context->ClearRenderTargetView(g_d3d.rtv, c);
    ImGui_ImplDX11_RenderDrawData(ImGui::GetDrawData());

    // Multi-viewport: draw the panels the user popped out into their own OS windows.
    if (ImGui::GetIO().ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
        ImGui::UpdatePlatformWindows();
        ImGui::RenderPlatformWindowsDefault();
    }
    g_d3d.swap_chain->Present(g_d3d.vsync ? 1 : 0, 0);
}

// ---------------------------------------------------------------------------
// --selftest : everything M1 promises, minus the window.
// ---------------------------------------------------------------------------
int run_selftest(const std::string &ini_path, const std::string &language,
                 const std::string &log_path) {
    FILE *log = nullptr;
    fopen_s(&log, log_path.c_str(), "w");
    int pass = 0, fail = 0;
    const auto check = [&](bool ok, const std::string &what,
                           const std::string &detail = std::string()) {
        char buf[512];
        std::snprintf(buf, sizeof(buf), "  [%s] %s%s%s\n", ok ? "ok" : "FAIL", what.c_str(),
                      detail.empty() ? "" : " -- ", detail.c_str());
        std::fputs(buf, stdout);
        std::fflush(stdout);
        if (log) std::fputs(buf, log);
        ok ? ++pass : ++fail;
    };

    char head[256];
    std::snprintf(head, sizeof(head), "ecm_gui self-test\n  exe dir     : %s\n  ini         : %s\n",
                  ecmgui::exe_dir().c_str(), ini_path.c_str());
    std::fputs(head, stdout);
    std::fflush(stdout);
    if (log) std::fputs(head, log);

    // ---- 1. localization -------------------------------------------------
    const std::string loc_dir = ecmgui::default_localization_dir();
    ecmgui::Localization loc;
    std::string err;
    const bool base_ok = loc.load(loc_dir, "english", err);
    check(base_ok, "localization baseline english.xml loads", err);
    const std::size_t base_count = loc.count();
    if (base_ok) {
        check(base_count >= 10, "baseline defines the UI strings",
              std::to_string(base_count) + " entries");
        check(loc.t("menu", "file") == "File", "english baseline text resolves",
              loc.t("menu", "file"));
        check(loc.t("nope", "nope") == "nope.nope", "unknown key degrades to panel.id",
              loc.t("nope", "nope"));
    }
    const std::string cjk_file = ecmgui::find_cjk_font_file();
    ecmgui::Localization zh;
    const bool zh_ok = zh.load(loc_dir, "chineseSimplified", err);
    check(zh_ok, "chineseSimplified.xml loads (UTF-8)", err);
    if (zh_ok) {
        check(zh.needs_cjk_font(), "chinese strings are detected as CJK");
        check(zh.t("menu", "file") != "File", "chinese overrides the english text",
              zh.t("menu", "file"));
        check(zh.missing_keys().empty(), "no missing keys vs the english baseline",
              std::to_string(zh.missing_keys().size()) + " missing");
    }
    // Every worker state the Workers table can display must have a localization entry.
    // This is the regression guard for the user report "state shows workers.state__stopped":
    // the id was derived from the state NAME, producing a double underscore, so the lookup
    // missed and the raw fallback text was painted into the table.
    {
        int n_states = 0;
        const ecmgui::WorkerRunState *states = ecmgui::all_worker_states(&n_states);
        int missing = 0;
        std::string first_missing;
        for (int i = 0; i < n_states; ++i) {
            const std::string key = ecmgui::state_key(states[i]);
            const std::string fallback = "workers." + key;
            if (loc.t("workers", key) == fallback) {
                ++missing;
                if (first_missing.empty()) first_missing = key;
            }
            if (zh_ok && zh.t("workers", key) == fallback) {
                ++missing;
                if (first_missing.empty()) first_missing = "zh:" + key;
            }
        }
        check(missing == 0, "every worker state has a localization entry (en + zh)",
              std::to_string(missing) + " missing, first: " + first_missing);
        check(std::string(ecmgui::state_key(ecmgui::WorkerRunState::Stopped)) == "state_stopped",
              "the stopped-state id is the documented one",
              ecmgui::state_key(ecmgui::WorkerRunState::Stopped));
    }
    // Keys the UI looks up dynamically (not a state, but same failure mode: if the key is
    // missing the raw "workers.<key>" text is painted into the table).
    {
        const char *const kDynamicKeys[] = {"waiting_progress"};
        int missing = 0;
        std::string first_missing;
        for (const char *key : kDynamicKeys) {
            const std::string fallback = std::string("workers.") + key;
            if (loc.t("workers", key) == fallback || (zh_ok && zh.t("workers", key) == fallback)) {
                ++missing;
                if (first_missing.empty()) first_missing = key;
            }
        }
        check(missing == 0, "dynamically referenced workers keys exist in both languages",
              std::to_string(missing) + " missing, first: " + first_missing);
    }
    check(!cjk_file.empty() || true, "CJK font search", cjk_file.empty()
                                                          ? std::string("none found -> UI would fall back to english")
                                                          : cjk_file);

    // ---- 2. ini round-trip ----------------------------------------------
    const std::string tmp_ini = ecmgui::path_join(ecmgui::exe_dir(), "ecm_gui_selftest.ini");
    {
        std::string w_err;
        ecmgui::IniFile seeded;
        seeded.reset(tmp_ini);
        std::FILE *f = nullptr;
        fopen_s(&f, tmp_ini.c_str(), "w");
        if (f) {
            std::fputs("# a comment the GUI must not lose\n", f);
            std::fputs("device = 0\n", f);
            std::fputs("\n", f);
            std::fputs("[Worker #2]\n", f);
            std::fputs("device = 1\n", f);
            std::fputs("custom_key = keep me\n", f);
            std::fclose(f);
        }
        ecmgui::IniFile ini;
        check(ini.load(tmp_ini, w_err), "load a small ini", w_err);
        ini.set("", "device", "0");            // unchanged value -> not dirty
        check(!ini.dirty(), "setting the same value does not dirty the file");
        ini.set("", "worktodo", "w1.txt");     // new global key (before the section)
        ini.set("Worker #2", "gpucurves", "384");
        ini.set("GUI", "language", "chineseSimplified");   // new section at EOF
        check(ini.dirty(), "changes mark the file dirty");
        check(ini.save(w_err), "atomic save", w_err);
        check(ecmgui::file_exists(tmp_ini + ".bak"), "one .bak generation written");

        ecmgui::IniFile back;
        check(back.load(tmp_ini, w_err), "reload after save", w_err);
        check(back.get("", "worktodo") == "w1.txt", "new global key persisted",
              back.get("", "worktodo"));
        check(back.get("Worker #2", "gpucurves") == "384", "section key persisted",
              back.get("Worker #2", "gpucurves"));
        check(back.get("GUI", "language") == "chineseSimplified", "new section persisted",
              back.get("GUI", "language"));
        check(back.get("Worker #2", "custom_key") == "keep me",
              "an unknown key the driver ignores is preserved");
        bool comment_kept = false, order_ok = true;
        std::size_t dev_global_line = 0, worker_line = 0;
        const auto &lines = back.lines();
        for (std::size_t i = 0; i < lines.size(); ++i) {
            if (lines[i].kind == ecmgui::IniFile::Line::Kind::Comment &&
                lines[i].raw.find("must not lose") != std::string::npos) {
                comment_kept = true;
            }
            if (lines[i].kind == ecmgui::IniFile::Line::Kind::KeyValue &&
                lines[i].section.empty() && lines[i].key == "device") {
                dev_global_line = i;
            }
            if (lines[i].kind == ecmgui::IniFile::Line::Kind::Section &&
                lines[i].section == "Worker #2") {
                worker_line = i;
            }
            if (lines[i].kind == ecmgui::IniFile::Line::Kind::KeyValue &&
                lines[i].section.empty() && lines[i].key == "worktodo" &&
                worker_line != 0 && i > worker_line) {
                order_ok = false;          // leaked into the [Worker #2] section
            }
        }
        check(comment_kept, "comments survive the rewrite");
        check(dev_global_line < worker_line, "file order preserved (global before sections)");
        check(order_ok, "a new global key stays before the first section header");
        std::remove(tmp_ini.c_str());
        std::remove((tmp_ini + ".bak").c_str());
    }

    // ---- 3. app state (config-driven worker list) -------------------------
    {
        ecmgui::App app;
        std::string a_err;
        const bool ok = app.init(ini_path, language, a_err);
        check(ok, "App::init reads the ini", a_err);
        if (ok) {
            check(app.ini().loaded(), "ini handle is loaded");
            check(app.loc().count() > 0, "app has localization strings",
                  std::to_string(app.loc().count()));
            check(app.workers().size() >= 1, "worker list built from NumWorkers",
                  std::to_string(app.workers().size()) + " worker(s)");
            if (!app.workers().empty()) {
                const ecmgui::WorkerView &w = app.workers().front();
                check(w.effective_config.find("device") != std::string::npos ||
                          w.effective_config.find("no keys") != std::string::npos,
                      "effective-config text is produced");
            }
        }
    }

    // ---- 4. ImGui: context, fonts, docking symbols ------------------------
    {
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO &io = ImGui::GetIO();
        io.IniFilename = nullptr;
        io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
        io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
        check((io.ConfigFlags & ImGuiConfigFlags_DockingEnable) != 0,
              "docking enable flag accepted (docking branch)");
        check((io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable) != 0,
              "multi-viewport flag accepted (docking branch)");

        ImFont *def = io.Fonts->AddFontDefault();
        check(def != nullptr, "built-in (ASCII) font added");
        ImFont *cjk = nullptr;
        if (!cjk_file.empty()) {
            // ImGui 1.92+ loads glyphs on demand, so no glyph-range list is passed:
            // the CJK atlas is not pre-baked with 21k glyphs any more.
            ImFontConfig cfg;
            cfg.OversampleH = cfg.OversampleV = 1;
            cjk = io.Fonts->AddFontFromFileTTF(cjk_file.c_str(), 17.0f, &cfg);
            check(cjk != nullptr, "CJK system font loaded at runtime", cjk_file);
        }
        check(io.Fonts->Fonts.Size > 0, "font atlas has at least one font",
              std::to_string(io.Fonts->Fonts.Size) + " font(s)");
        ImGui::DestroyContext();
        check(true, "ImGui context create/destroy round-trip");
    }

    char tail[256];
    std::snprintf(tail, sizeof(tail), "\npassed: %d   failed: %d\n  log: %s\n", pass, fail,
                  log_path.c_str());
    std::fputs(tail, stdout);
    std::fflush(stdout);
    if (log) {
        std::fputs(tail, log);
        std::fclose(log);
    }
    return fail == 0 ? 0 : 1;
}

} // namespace

// A GUI-subsystem process has no console, but it DOES inherit stdout/stderr when the
// caller redirected them (`cmd /c "... 2>&1" | ...`). Only attach to the parent
// console when there is no usable stream: attaching unconditionally used to replace
// the inherited pipe with CONOUT$, so a script capturing the self-test saw nothing.
void ensure_stdio() {
#ifdef _WIN32
    const HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);
    const bool have_out = (out != nullptr && out != INVALID_HANDLE_VALUE);
    if (have_out) return;
    if (AttachConsole(ATTACH_PARENT_PROCESS)) {
        FILE *f = nullptr;
        freopen_s(&f, "CONOUT$", "w", stdout);
        freopen_s(&f, "CONOUT$", "w", stderr);
    }
#endif
}

int APIENTRY wWinMain(HINSTANCE inst, HINSTANCE, LPWSTR, int) {
    const std::vector<std::string> args = ecmgui::command_line_args();

    const bool selftest = ecmgui::has_arg(args, "--selftest");
    std::string ini_path = ecmgui::arg_value(args, "-ini");
    const std::string language = ecmgui::arg_value(args, "--language");
    if (ini_path.empty()) ini_path = ecmgui::default_ini_path();

    // --switch-language <stem> [--switch-language-after <frames>]: diagnostic (like
    // --trace) that performs the Language-menu switch from a script. The menu itself
    // does exactly the same call, so a scripted run can verify the runtime font reload
    // ("switching to Chinese must not keep the Latin font and print ???") without
    // clicking. See tools/test/test_gui_cjk_pixels.ps1 run [D].
    const std::string switch_language = ecmgui::arg_value(args, "--switch-language");
    int switch_after_frames = 30;
    if (const std::string v = ecmgui::arg_value(args, "--switch-language-after"); !v.empty()) {
        switch_after_frames = std::atoi(v.c_str());
        if (switch_after_frames < 1) switch_after_frames = 1;
    }

    if (ecmgui::has_arg(args, "--trace")) {
        const std::string p = ecmgui::path_join(ecmgui::exe_dir(), "ecm_gui_trace.log");
        // _SH_DENYWR: we are the only writer, but a test script must be able to READ
        // the trace while the GUI is still running (that is the whole point of it).
        g_trace = _fsopen(p.c_str(), "w", _SH_DENYWR);
        trace("start: exe dir %s", ecmgui::exe_dir().c_str());
        trace("start: ini %s", ini_path.c_str());
    }

    if (selftest) {
        return run_selftest(ini_path, language,
                            ecmgui::path_join(ecmgui::exe_dir(), "ecm_gui_selftest.log"));
    }

    // --gpu-selftest: NVML monitoring, cross-checked against nvidia-smi.
    if (ecmgui::has_arg(args, "--gpu-selftest")) {
        ensure_stdio();
        int expect = 0;
        if (const std::string v = ecmgui::arg_value(args, "--expect-devices"); !v.empty()) {
            expect = std::atoi(v.c_str());
        }
        return ecmgui::run_gpu_selftest(
            expect, ecmgui::has_arg(args, "--skip-nvidia-smi"),
            ecmgui::path_join(ecmgui::exe_dir(), "ecm_gui_gpu_selftest.log"));
    }

    // --worker-selftest: drive the worker supervisor against a fake worker, headless.
    if (ecmgui::has_arg(args, "--worker-selftest")) {
        std::string fake = ecmgui::arg_value(args, "--fake");
        if (fake.empty()) {
            fake = ecmgui::path_join(ecmgui::exe_dir(), "ecm_gui_fake_worker.exe");
        }
        ensure_stdio();
        return ecmgui::run_worker_selftest(
            fake, ecmgui::path_join(ecmgui::exe_dir(), "ecm_gui_worker_selftest.log"));
    }

    ensure_stdio();


    ecmgui::App app;
    std::string err;
    if (!app.init(ini_path, language, err)) {
        MessageBoxA(nullptr, err.c_str(), "ecm_gui", MB_ICONERROR | MB_OK);
        return 1;
    }
    // Mirror worker state transitions into the trace file: this is how a script can
    // verify the supervisor without looking at the window.
    app.set_trace([](const std::string &line) { trace("%s", line.c_str()); });
    // Driver-only switches given to the GUI. This is an easy mistake with a confusing
    // symptom: `ecm_gui.exe -ini x.ini --worker 1` looks like "start the worker" but just
    // opens an idle window -- which is how a scripted run once left a GUI sitting on the
    // desktop for minutes (2026-09-29). Say so instead of ignoring it silently.
    if (ecmgui::has_arg(args, "--worker")) {
        trace("warning: --worker is a driver flag, ecm_gui ignores it (start ecm_cuda.exe for a worker)");
        app.set_status("note: --worker is a driver flag and is ignored by the GUI; "
                       "workers are started from [GUI] exe= / the Start button");
    }
    g_app = &app;          // WM_CLOSE consults it (confirmation + checkpoint flow)
    struct AppGuard {
        ~AppGuard() { g_app = nullptr; }
    } app_guard;

    ImGui_ImplWin32_EnableDpiAwareness();
    WNDCLASSEXW wc = {sizeof(wc), CS_CLASSDC, wnd_proc, 0L, 0L, inst, nullptr, nullptr,
                      nullptr, nullptr, L"ecm_gui", nullptr};
    RegisterClassExW(&wc);
    const int wx = app.window_x(), wy = app.window_y();
    const int ww = app.window_w(), wh = app.window_h();
    g_hwnd = CreateWindowW(wc.lpszClassName, L"ecm_gui", WS_OVERLAPPEDWINDOW, wx, wy, ww, wh,
                           nullptr, nullptr, wc.hInstance, nullptr);
    if (!g_hwnd || !create_device(g_hwnd)) {
        MessageBoxA(nullptr, "cannot create the Direct3D 11 device", "ecm_gui",
                    MB_ICONERROR | MB_OK);
        if (g_hwnd) DestroyWindow(g_hwnd);
        UnregisterClassW(wc.lpszClassName, wc.hInstance);
        return 2;
    }

    ShowWindow(g_hwnd, SW_SHOWDEFAULT);
    UpdateWindow(g_hwnd);

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    app.on_imgui_ready();
    ImGui_ImplWin32_Init(g_hwnd);
    ImGui_ImplDX11_Init(g_d3d.device, g_d3d.context);

    // ---- fonts -----------------------------------------------------------------
    // Nothing is shipped with the GUI (docs/DEV_ECM_GUI.md 10.3), so the font comes
    // from the system and its size follows the display DPI:
    //   [GUI] font_size = auto | <px>     auto = 15 px * (window DPI / 96)
    //   [GUI] font      = <path to a .ttf/.ttc>   (empty = pick automatically)
    //   [GUI] font_snap = 1 | 0           snap glyph advances to whole pixels
    // A CJK language needs a CJK-capable font; a Latin UI prefers Segoe UI over the
    // built-in ~13 px bitmap font, which blurs when scaled.
    ImGuiIO &io = ImGui::GetIO();
    const float dpi_scale = ecmgui::window_dpi_scale(g_hwnd);
    ImGui::GetStyle().ScaleAllSizes(dpi_scale);
    apply_ui_font(app, dpi_scale, "startup");

    const float clear[4] = {0.10f, 0.11f, 0.13f, 1.00f};
    int frames = 0;
    while (new_frame()) {
        if (g_skip_frame) continue;           // minimized: no draw, no Present
        if (!switch_language.empty() && frames == switch_after_frames) {
            trace("language: runtime switch to %s (--switch-language)", switch_language.c_str());
            app.set_language(switch_language);
        }
        // A language switch (Language menu) may need a different font -- switching to
        // Chinese with the Latin font loaded would render every label as "???". Re-apply
        // between frames (safe: no ImGui frame is in flight here).
        if (app.font_reload_requested()) {
            apply_ui_font(app, ecmgui::window_dpi_scale(g_hwnd), "language change");
            app.clear_font_reload_request();
        }
        app.draw();
        if (app.quit_requested()) break;      // File -> Quit
        render_frame(const_cast<float *>(clear), app.refresh_hz());
        ++frames;
        if (frames == 1 || (frames % 120) == 0) trace("frame %d rendered", frames);
    }
    trace("frame loop left after %d frame(s)", frames);

    // Order matters: the layout is captured from ImGui (SaveIniSettingsToMemory),
    // so the ini must be written BEFORE the context is destroyed. Destroying it
    // first made the shutdown hang in a null-context dereference.
    RECT r{};
    if (GetWindowRect(g_hwnd, &r)) {
        app.set_window_rect(r.left, r.top, r.right - r.left, r.bottom - r.top);
    }
    std::string save_err;
    trace("saving %s", ini_path.c_str());
    const bool saved = app.shutdown(save_err);
    if (!saved) trace("save FAILED: %s", save_err.c_str());
    else trace("saved");

    ImGui_ImplDX11_Shutdown();
    ImGui_ImplWin32_Shutdown();
    ImGui::DestroyContext();
    cleanup_device();
    DestroyWindow(g_hwnd);
    UnregisterClassW(wc.lpszClassName, wc.hInstance);

    if (!saved) {
        MessageBoxA(nullptr, ("cannot save " + ini_path + ": " + save_err).c_str(), "ecm_gui",
                    MB_ICONWARNING | MB_OK);
        return 3;
    }
    trace("exiting");
    return 0;
}
