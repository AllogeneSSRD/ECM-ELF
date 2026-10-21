#pragma once
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include "ecm/platform/process/curve_process.h"
namespace s2prod {
struct Handle {
    HANDLE value = INVALID_HANDLE_VALUE;
    Handle()=default;
    Handle(const Handle&)=delete;
    Handle &operator=(const Handle&)=delete;
    Handle(Handle &&other)noexcept:value(other.value){other.value=INVALID_HANDLE_VALUE;}
    ~Handle() { if (value != INVALID_HANDLE_VALUE && value) CloseHandle(value); }
};
extern volatile LONG stop_requests;
extern HANDLE inherited_stop_event;
std::filesystem::path executable();
void poll_inherited_stop();
BOOL WINAPI console_control(DWORD event);
}
