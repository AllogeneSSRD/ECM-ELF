#include "host_wait.h"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <thread>
#endif

namespace stage2_wait {
Runtime::Runtime() {
#ifdef _WIN32
    if(adaptive)timer=CreateWaitableTimerExW(nullptr,nullptr,0x2,TIMER_MODIFY_STATE|SYNCHRONIZE);
#endif
}
Runtime::~Runtime() {
    if(readback_event)cudaEventDestroy(readback_event);
#ifdef _WIN32
    if(timer)CloseHandle(static_cast<HANDLE>(timer));
#endif
}
bool Runtime::pause(unsigned us) {
#ifdef _WIN32
    if(!timer)return false;
    LARGE_INTEGER due;due.QuadPart=-static_cast<LONGLONG>(us)*10;
    return SetWaitableTimer(static_cast<HANDLE>(timer),&due,0,nullptr,nullptr,FALSE) &&
           WaitForSingleObject(static_cast<HANDLE>(timer),INFINITE)==WAIT_OBJECT_0;
#else
    std::this_thread::sleep_for(std::chrono::microseconds(us));return true;
#endif
}
}
