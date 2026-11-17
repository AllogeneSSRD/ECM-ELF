#include "process.h"
#include <stdexcept>
#include <vector>

namespace s2prod {
namespace fs=std::filesystem;
volatile LONG stop_requests = 0;
HANDLE inherited_stop_event=nullptr;
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
}
