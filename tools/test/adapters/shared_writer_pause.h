#pragma once
#include "io/files.h"
#include <filesystem>
#include <stdexcept>
#include <string>
#include <windows.h>

// Only scheduling changes: all reads, locks and writes remain production calls.
inline void shared_writer_before_publish(const std::filesystem::path& target) {
    static bool paused = false;
    char expected[32768]{}, prefix[256]{};
    if (paused || !GetEnvironmentVariableA("ECM_TEST_WRITE_TARGET", expected, sizeof(expected)) ||
        !GetEnvironmentVariableA("ECM_TEST_WRITE_EVENTS", prefix, sizeof(prefix)) ||
        std::filesystem::absolute(target).lexically_normal() !=
            std::filesystem::absolute(expected).lexically_normal())
        return;
    paused = true;
    HANDLE ready = OpenEventA(EVENT_MODIFY_STATE, FALSE, (std::string(prefix) + ".ready").c_str());
    HANDLE go = OpenEventA(SYNCHRONIZE, FALSE, (std::string(prefix) + ".go").c_str());
    if (!ready || !go)
        throw std::runtime_error("cannot open writer scheduling events");
    SetEvent(ready);
    const auto status = WaitForSingleObject(go, 10000);
    CloseHandle(ready);
    CloseHandle(go);
    if (status != WAIT_OBJECT_0)
        throw std::runtime_error("writer scheduling timed out");
}
