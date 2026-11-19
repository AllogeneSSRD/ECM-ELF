#include "process.h"
#include <stdexcept>
#include <vector>

namespace s2prod {
namespace fs=std::filesystem;
#ifdef _WIN32
volatile LONG stop_requests = 0;
HANDLE inherited_stop_event=nullptr;
bool set_inherited_stop(uint64_t value) {
    const auto event=reinterpret_cast<HANDLE>(static_cast<uintptr_t>(value));
    if(WaitForSingleObject(event,0)==WAIT_FAILED)return false;
    inherited_stop_event=event;
    return true;
}
fs::path executable() {
    std::vector<wchar_t> buf(32768);
    const DWORD n = GetModuleFileNameW(nullptr, buf.data(), static_cast<DWORD>(buf.size()));
    if (!n || n == buf.size()) throw std::runtime_error("cannot locate executable");
    return fs::path(std::wstring(buf.data(), n));
}
void poll_inherited_stop() {
    if(inherited_stop_event && WaitForSingleObject(inherited_stop_event,0)==WAIT_OBJECT_0)
        InterlockedCompareExchange(&stop_requests,1,0);
}
BOOL WINAPI console_control(DWORD event) {
    if (event != CTRL_C_EVENT && event != CTRL_BREAK_EVENT) return FALSE;
    InterlockedIncrement(&stop_requests);
    return TRUE;
}
#else
volatile std::sig_atomic_t stop_requests = 0;
bool set_inherited_stop(uint64_t) { return false; }
fs::path executable() { return fs::read_symlink("/proc/self/exe"); }
void poll_inherited_stop() {}
void console_control(int) {
    if (stop_requests < 2) ++stop_requests;
}
#endif
}
