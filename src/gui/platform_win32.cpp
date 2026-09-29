#include "platform.h"

#include <cstring>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <commdlg.h>            /* GetOpenFileNameA (comdlg32) */
#include <shellapi.h>
#endif

namespace ecmgui {
namespace {

#ifdef _WIN32
std::string get_env(const char *name) {
    char buf[MAX_PATH * 2];
    const DWORD n = GetEnvironmentVariableA(name, buf, sizeof(buf));
    if (n == 0 || n >= sizeof(buf)) return std::string();
    return std::string(buf, n);
}
#endif

} // namespace

std::vector<std::string> command_line_args() {
    std::vector<std::string> out;
#ifdef _WIN32
    int argc = 0;
    LPWSTR *wargv = CommandLineToArgvW(GetCommandLineW(), &argc);
    if (wargv == nullptr) return out;
    for (int i = 1; i < argc; ++i) {
        // Arguments are paths / switches in practice; convert from UTF-16.
        const int need = WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, nullptr, 0, nullptr,
                                             nullptr);
        if (need <= 1) {
            out.push_back(std::string());
            continue;
        }
        std::string s(static_cast<std::size_t>(need - 1), '\0');
        WideCharToMultiByte(CP_UTF8, 0, wargv[i], -1, &s[0], need, nullptr, nullptr);
        out.push_back(s);
    }
    LocalFree(wargv);
#endif
    return out;
}

bool has_arg(const std::vector<std::string> &args, const std::string &name) {
    for (const std::string &a : args) {
        if (a == name) return true;
    }
    return false;
}

std::string arg_value(const std::vector<std::string> &args, const std::string &name) {
    for (std::size_t i = 0; i + 1 < args.size(); ++i) {
        if (args[i] == name) return args[i + 1];
    }
    return std::string();
}


std::string exe_dir() {
#ifdef _WIN32
    char buf[MAX_PATH];
    const DWORD n = GetModuleFileNameA(nullptr, buf, MAX_PATH);
    if (n == 0 || n >= MAX_PATH) return std::string();
    std::string p(buf, n);
    const std::size_t slash = p.find_last_of("\\/");
    if (slash == std::string::npos) return std::string();
    return p.substr(0, slash);
#else
    return std::string();
#endif
}

std::string path_join(const std::string &a, const std::string &b) {
    if (a.empty()) return b;
    if (b.empty()) return a;
    std::string out = a;
    const char last = out.back();
    if (last != '/' && last != '\\') {
#ifdef _WIN32
        out.push_back('\\');
#else
        out.push_back('/');
#endif
    }
    return out + b;
}

bool file_exists(const std::string &path) {
#ifdef _WIN32
    const DWORD attrs = GetFileAttributesA(path.c_str());
    return attrs != INVALID_FILE_ATTRIBUTES && !(attrs & FILE_ATTRIBUTE_DIRECTORY);
#else
    return false;
#endif
}

std::string file_name(const std::string &path) {
    const std::size_t slash = path.find_last_of("\\/");
    return (slash == std::string::npos) ? path : path.substr(slash + 1);
}

long long newest_checkpoint_mtime(const std::string &dir, std::string *name_out) {
    if (name_out != nullptr) name_out->clear();
    if (dir.empty()) return 0;
#ifdef _WIN32
    // The driver writes "<dir>\.ecm_ckpt_<bits>_<hash>_<hash>.dat" and REMOVES it when the
    // task completes, so the presence of a fresh one is exactly "this task can be resumed
    // from here" (kernels/cuda/cgbn_stage1.cu:775,1676,1709).
    const std::string pattern = path_join(dir, ".ecm_ckpt_*.dat");
    WIN32_FIND_DATAA fd;
    HANDLE h = FindFirstFileA(pattern.c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE) return 0;
    long long newest = 0;
    std::string newest_name;
    do {
        if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY) continue;
        const long long t = (static_cast<long long>(fd.ftLastWriteTime.dwHighDateTime) << 32) |
                            static_cast<long long>(fd.ftLastWriteTime.dwLowDateTime);
        if (t > newest) {
            newest = t;
            newest_name = fd.cFileName;
        }
    } while (FindNextFileA(h, &fd));
    FindClose(h);
    if (name_out != nullptr) *name_out = newest_name;
    return newest;
#else
    return 0;
#endif
}

std::string default_ini_path() {
    const std::string dir = exe_dir();
    return dir.empty() ? std::string("ecm.ini") : path_join(dir, "ecm.ini");
}

std::string default_localization_dir() {
    const std::string dir = exe_dir();
    return dir.empty() ? std::string("localization") : path_join(dir, "localization");
}

std::vector<std::string> cjk_font_candidates() {
    std::vector<std::string> out;
#ifdef _WIN32
    // Microsoft YaHei first (the UI font of every Chinese Windows), then the
    // other fonts that ship with Windows and cover Simplified/Traditional CJK.
    const std::string windir = get_env("WINDIR");
    const std::string fonts = windir.empty() ? std::string("C:\\Windows\\Fonts")
                                             : path_join(windir, "Fonts");
    for (const char *name : {"msyh.ttc", "msyh.ttf", "msyhbd.ttc", "simhei.ttf",
                             "msjh.ttc", "simsun.ttc", "Deng.ttf"}) {
        out.push_back(path_join(fonts, name));
    }
#else
    // Linux (M7): fontconfig will replace this list.
    for (const char *p : {"/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                          "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"}) {
        out.push_back(p);
    }
#endif
    return out;
}

std::string find_cjk_font_file() {
    // ECM_GUI_CJK_FONT overrides the search (tests use it to prove the "no CJK font at
    // all -> fall back to English" path; `ECM_GUI_CJK_FONT=none` forces "not found").
    const std::string override_path = get_env("ECM_GUI_CJK_FONT");
    if (!override_path.empty()) {
        if (override_path == "none") return std::string();
        if (file_exists(override_path)) return override_path;
    }
    for (const std::string &p : cjk_font_candidates()) {
        if (file_exists(p)) return p;
    }
    return std::string();
}

std::vector<std::string> ui_font_candidates() {
    std::vector<std::string> out;
#ifdef _WIN32
    const std::string windir = get_env("WINDIR");
    const std::string fonts = windir.empty() ? std::string("C:\\Windows\\Fonts")
                                             : path_join(windir, "Fonts");
    // Segoe UI ships with Vista+; Tahoma/Arial are the older fallbacks. The built-in
    // ImGui bitmap font is only ~13 px and blurs when scaled, so a real outline font
    // is preferred whenever one is present.
    for (const char *name : {"segoeui.ttf", "tahoma.ttf", "arial.ttf", "verdana.ttf"}) {
        out.push_back(path_join(fonts, name));
    }
#endif
    return out;
}

std::string find_ui_font_file() {
    for (const std::string &p : ui_font_candidates()) {
        if (file_exists(p)) return p;
    }
    return std::string();
}

float window_dpi_scale(void *hwnd) {
#ifdef _WIN32
    if (hwnd == nullptr) return 1.0f;
    // GetDpiForWindow is Windows 10 1607+; fall back to the window DC's LOGPIXELSY.
    using get_dpi_for_window_t = UINT(WINAPI *)(HWND);
    static get_dpi_for_window_t fn = nullptr;
    static bool resolved = false;
    if (!resolved) {
        resolved = true;
        if (HMODULE user32 = GetModuleHandleA("user32.dll")) {
            fn = reinterpret_cast<get_dpi_for_window_t>(
                reinterpret_cast<void *>(GetProcAddress(user32, "GetDpiForWindow")));
        }
    }
    UINT dpi = 96;
    if (fn != nullptr) {
        dpi = fn(static_cast<HWND>(hwnd));
    } else {
        HDC dc = GetDC(static_cast<HWND>(hwnd));
        if (dc != nullptr) {
            dpi = static_cast<UINT>(GetDeviceCaps(dc, LOGPIXELSY));
            ReleaseDC(static_cast<HWND>(hwnd), dc);
        }
    }
    if (dpi < 48 || dpi > 480) dpi = 96;
    return static_cast<float>(dpi) / 96.0f;
#else
    (void)hwnd;
    return 1.0f;
#endif
}

bool open_in_explorer(const std::string &path) {
#ifdef _WIN32
    // explorer.exe returns immediately and may report a non-zero exit code even on
    // success, so the shell call result is what we report.
    const HINSTANCE h = ShellExecuteA(nullptr, "open", path.c_str(), nullptr, nullptr,
                                      SW_SHOWNORMAL);
    return reinterpret_cast<std::intptr_t>(h) > 32;
#else
    (void)path;
    return false;
#endif
}

bool run_capture(const std::string &command_line, std::string &out, int &exit_code,
                 int timeout_ms) {
    out.clear();
    exit_code = -1;
#ifdef _WIN32
    // One inheritable pipe for the child's stdout, with the write end in the child only.
    SECURITY_ATTRIBUTES sa{};
    sa.nLength = sizeof(sa);
    sa.bInheritHandle = TRUE;
    HANDLE rd = nullptr, wr = nullptr;
    if (!CreatePipe(&rd, &wr, &sa, 0)) return false;
    SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);

    // The child must also SEE the inherited stdout: STARTF_USESTDHANDLES with the pipe as
    // hStdOutput is what makes `fprintf(stdout, ...)` land in the pipe. No console window
    // appears, which a `_popen` would flash on the user's desktop.
    STARTUPINFOA si{};
    si.cb = sizeof(si);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    si.hStdOutput = wr;
    si.hStdError = GetStdHandle(STD_ERROR_HANDLE);

    std::string cmd = command_line;             // CreateProcessA may modify the buffer
    PROCESS_INFORMATION pi{};
    const DWORD flags = CREATE_NO_WINDOW;
    if (!CreateProcessA(nullptr, cmd.data(), nullptr, nullptr, TRUE, flags, nullptr, nullptr,
                        &si, &pi)) {
        CloseHandle(rd);
        CloseHandle(wr);
        return false;
    }
    CloseHandle(wr);                            // our copy: the child owns the write end

    const DWORD deadline = GetTickCount() + static_cast<DWORD>(timeout_ms < 0 ? 0 : timeout_ms);
    bool timed_out = false;
    char buf[4096];
    for (;;) {
        DWORD avail = 0;
        if (PeekNamedPipe(rd, nullptr, 0, nullptr, &avail, nullptr) && avail > 0) {
            DWORD got = 0;
            const DWORD want = (avail < sizeof(buf)) ? avail : static_cast<DWORD>(sizeof(buf));
            if (ReadFile(rd, buf, want, &got, nullptr) && got > 0) {
                out.append(buf, got);
                continue;
            }
        }
        if (WaitForSingleObject(pi.hProcess, 20) == WAIT_OBJECT_0) break;
        if (timeout_ms > 0 && GetTickCount() > deadline) { timed_out = true; break; }
    }
    // Drain whatever is still buffered after the process ended.
    for (;;) {
        DWORD avail = 0;
        if (!PeekNamedPipe(rd, nullptr, 0, nullptr, &avail, nullptr) || avail == 0) break;
        DWORD got = 0;
        const DWORD want = (avail < sizeof(buf)) ? avail : static_cast<DWORD>(sizeof(buf));
        if (!ReadFile(rd, buf, want, &got, nullptr) || got == 0) break;
        out.append(buf, got);
    }

    DWORD code = 0;
    if (timed_out) {
        TerminateProcess(pi.hProcess, 1);
        WaitForSingleObject(pi.hProcess, 2000);
    }
    GetExitCodeProcess(pi.hProcess, &code);
    exit_code = static_cast<int>(code);
    CloseHandle(rd);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    return !timed_out;
#else
    (void)command_line;
    (void)timeout_ms;
    return false;
#endif
}

bool browse_for_file(std::string &path, const std::string &title, const std::string &filter) {
#ifdef _WIN32
    char name[MAX_PATH] = {0};
    if (!path.empty() && path.size() < MAX_PATH) {
        std::memcpy(name, path.c_str(), path.size());
    }
    OPENFILENAMEA ofn{};
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = GetActiveWindow();
    ofn.lpstrFilter = filter.empty() ? "All files\0*.*\0\0" : filter.c_str();
    ofn.lpstrFile = name;
    ofn.nMaxFile = MAX_PATH;
    ofn.lpstrTitle = title.empty() ? nullptr : title.c_str();
    ofn.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_EXPLORER;
    if (!GetOpenFileNameA(&ofn)) return false;
    path.assign(name);
    return true;
#else
    (void)path; (void)title; (void)filter;
    return false;
#endif
}

} // namespace ecmgui
