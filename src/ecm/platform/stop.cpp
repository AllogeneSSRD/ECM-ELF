#include "ecm/platform/stop.h"
#include <atomic>
#include <csignal>
#ifdef _WIN32
#include <windows.h>
#endif
namespace {
std::atomic<bool> stop_requested{false};
#ifdef _WIN32
std::atomic<unsigned> console_stop_count{0};
HANDLE stop_event = nullptr; // borrowed; its creator owns the lifetime

BOOL WINAPI console_control(DWORD event) {
    if (event != CTRL_C_EVENT && event != CTRL_BREAK_EVENT)
        return FALSE;
    if (console_stop_count.fetch_add(1, std::memory_order_relaxed) > 0)
        return FALSE;
    stop_requested.store(true, std::memory_order_relaxed);
    return TRUE;
}
#else
void signal_control(int) {
    stop_requested.store(true, std::memory_order_relaxed);
}
#endif
} // namespace

void ecm_stage1_install_stop_handler() {
#ifdef _WIN32
    SetConsoleCtrlHandler(console_control, TRUE);
#else
    std::signal(SIGINT, signal_control);
    std::signal(SIGTERM, signal_control);
#endif
}

bool ecm_stage1_set_stop_handle(uintptr_t value) {
#ifdef _WIN32
    HANDLE candidate = reinterpret_cast<HANDLE>(value);
    DWORD flags = 0;
    if (!candidate || !GetHandleInformation(candidate, &flags))
        return false;
    stop_event = candidate;
    return true;
#else
    (void)value;
    return false;
#endif
}

bool ecm_stage1_stop_requested() {
    if (stop_requested.load(std::memory_order_relaxed))
        return true;
#ifdef _WIN32
    if (stop_event && WaitForSingleObject(stop_event, 0) == WAIT_OBJECT_0) {
        stop_requested.store(true, std::memory_order_relaxed);
        return true;
    }
#endif
    return false;
}
