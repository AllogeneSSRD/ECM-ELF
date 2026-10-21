#include "version.h"
#include "render/vulkan_window.h"
#include "imgui.h"
#include "imgui_impl_win32.h"
#include "imgui_impl_vulkan.h"
#include <shellapi.h>
#include <share.h>
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

extern IMGUI_IMPL_API LRESULT ImGui_ImplWin32_WndProcHandler(HWND, UINT, WPARAM, LPARAM);
namespace {
bool quit = false;
bool resized = false;
struct Options {
    std::filesystem::path diagnostics = L"ecm_gui_v2.log";
    int frames = 0;
    bool validation = false;
    bool no_loader = false;
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
        if (!ImGui_ImplWin32_Init(hwnd)) throw std::runtime_error("ImGui Win32 initialization failed");
        ImGui::GetIO().Fonts->AddFontDefault();
    }
    ~ImGuiSession() {
        if (ImGui::GetIO().BackendPlatformUserData) ImGui_ImplWin32_Shutdown();
        ImGui::DestroyContext();
    }
};
void draw_host(const ecmgui_v2::VulkanContext& context) {
    auto viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->WorkPos); ImGui::SetNextWindowSize(viewport->WorkSize);
    ImGui::Begin("GUI v2 Vulkan host", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoSavedSettings);
    ImGui::Text("ECM-ELF GUI v2 %s", ecmgui_v2::version);
    ImGui::Separator();
    ImGui::TextUnformatted(context.properties.deviceName);
    ImGui::TextUnformatted("Independent Win32 / Vulkan host. No computation workers are connected.");
    ImGui::TextUnformatted("P1: resize, minimize/restore and close to exercise the renderer lifecycle.");
    static char input[128] = "Vulkan / ImGui";
    ImGui::InputText("Input", input, sizeof(input));
    ImGui::Button("Host control");
    ImGui::End();
}
int run(const Options& options, int show, FILE* log) {
    ImGui_ImplWin32_EnableDpiAwareness();
    NativeWindow native; native.create();
    ImGuiSession imgui(native.hwnd);
    ecmgui_v2::VulkanContext context;
    context.initialize(native.hwnd, options.validation, options.no_loader, log);
    ecmgui_v2::VulkanWindow renderer(context, log);
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
        RECT client{}; GetClientRect(native.hwnd, &client);
        if (IsIconic(native.hwnd) || client.right <= 0 || client.bottom <= 0) {
            MsgWaitForMultipleObjectsEx(0, nullptr, 100, QS_ALLINPUT, MWMO_INPUTAVAILABLE);
            continue;
        }
        if (resized || renderer.needs_rebuild) {
            // Vulkan Shutdown also destroys the main viewport's Win32 data.
            // Reinitialize both backends while preserving the ImGui context.
            if (renderer.rebuilds) {
                renderer.shutdown();
                ImGui_ImplWin32_Shutdown();
                if (!ImGui_ImplWin32_Init(native.hwnd)) throw std::runtime_error("Win32 backend reinitialization failed");
            }
            renderer.rebuild(static_cast<uint32_t>(client.right), static_cast<uint32_t>(client.bottom));
            resized = false;
        }
        if (renderer.needs_rebuild) {
            MsgWaitForMultipleObjectsEx(0, nullptr, 100, QS_ALLINPUT, MWMO_INPUTAVAILABLE);
            continue;
        }
        ImGui_ImplVulkan_NewFrame(); ImGui_ImplWin32_NewFrame(); ImGui::NewFrame();
        draw_host(context);
        ImGui::Render();
        if (renderer.render(ImGui::GetDrawData())) {
            ++frames;
            if (frames == 1 || frames % 20 == 0) { std::fprintf(log, "frame: %u\n", frames); std::fflush(log); }
        }
        if (frames == 5 && !options.failure.empty())
            ecmgui_v2::check_vk(options.failure == "surface-lost" ? VK_ERROR_SURFACE_LOST_KHR : VK_ERROR_DEVICE_LOST, "injected renderer failure");
        if (options.frames && frames >= static_cast<unsigned>(options.frames)) break;
        const auto elapsed = GetTickCount64() - started;
        if (elapsed < 100) MsgWaitForMultipleObjectsEx(0, nullptr, static_cast<DWORD>(100 - elapsed), QS_ALLINPUT, MWMO_INPUTAVAILABLE);
    }
    renderer.shutdown();
    std::fprintf(log, "exit: frames=%u rebuilds=%u worker_processes=0\n", frames, renderer.rebuilds);
    std::fflush(log);
    return context.validation_errors ? 3 : 0;
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
            else if ((argument == L"--diagnostics" || argument == L"--frames" || argument == L"--inject-failure") && i + 1 < args.size()) {
                const auto value = args[++i];
                if (argument == L"--diagnostics") options.diagnostics = value;
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
