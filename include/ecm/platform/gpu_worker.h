#pragma once
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace ecm_gpu {
#ifdef _WIN32
// Driver enumeration only. No context allocation on either adapter.
inline bool cuda_visible(int index) {
    const auto dll = LoadLibraryExW(L"nvcuda.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    if (!dll) return false;
    const auto init = reinterpret_cast<int (WINAPI*)(unsigned)>(GetProcAddress(dll, "cuInit"));
    const auto get = reinterpret_cast<int (WINAPI*)(int*, int)>(GetProcAddress(dll, "cuDeviceGet"));
    int device;
    const bool ok = init && get && init(0) == 0 && get(&device, index) == 0;
    FreeLibrary(dll);
    return ok;
}

inline BOOL WINAPI awaiting_worker(DWORD event) {
    return event == CTRL_C_EVENT || event == CTRL_BREAK_EVENT;
}
#endif

// The selected binary owns the queue, results, stop event and all curve children.
// Pin the handoff so a config edit or driver change cannot bounce it back.
inline std::optional<int> route(const std::filesystem::path& directory, bool stage2,
                                bool current_cuda, const std::string& mode, int index) {
#ifdef _WIN32
    const auto peer = directory / (current_cuda
        ? (stage2 ? L"ecm_opencl_stage2.exe" : L"ecm.exe")
        : (stage2 ? L"ecm_cuda_stage2.exe" : L"ecm_cuda.exe"));
    if (mode == "auto" && !std::filesystem::is_regular_file(peer)) return {};
    const bool cuda = mode == "auto" ? cuda_visible(index) : mode == "cuda";
    if (cuda == current_cuda) return {};
    if (!std::filesystem::is_regular_file(peer))
        throw std::runtime_error("GPU worker is not installed: " + peer.u8string());
    std::wstring command = GetCommandLineW();
    command += cuda ? L" --gpu-backend cuda" : L" --gpu-backend opencl";
    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    startup.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
    startup.hStdError = GetStdHandle(STD_ERROR_HANDLE);
    PROCESS_INFORMATION process{};
    // Same console/job, original quoting, standard pipes and inherited GUI stop handle.
    // Intercept console breaks in the waiting parent only; the worker handles both.
    SetConsoleCtrlHandler(awaiting_worker, TRUE);
    if (!CreateProcessW(peer.c_str(), command.data(), nullptr, nullptr, TRUE, 0, nullptr,
                        nullptr, &startup, &process)) {
        const auto error = GetLastError();
        SetConsoleCtrlHandler(awaiting_worker, FALSE);
        throw std::runtime_error("Cannot start GPU worker " + peer.u8string() + ": " + std::to_string(error));
    }
    CloseHandle(process.hThread);
    DWORD code = 2;
    const bool finished = WaitForSingleObject(process.hProcess, INFINITE) == WAIT_OBJECT_0 &&
                          GetExitCodeProcess(process.hProcess, &code);
    CloseHandle(process.hProcess);
    SetConsoleCtrlHandler(awaiting_worker, FALSE);
    if (!finished) throw std::runtime_error("Cannot obtain GPU worker exit status");
    return static_cast<int>(code);
#else
    if (mode != "auto" && (mode == "cuda") != current_cuda)
        throw std::runtime_error("Requested GPU backend is not linked into this worker");
    return {};
#endif
}
} // namespace ecm_gpu
