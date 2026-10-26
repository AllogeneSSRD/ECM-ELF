#include "io/files.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <windows.h>

static void require(bool ok, const char* message) {
    if (!ok)
        throw std::runtime_error(message);
}

int main(int argc, char** argv) try {
    require(argc == 2, "provide fresh evidence directory");
    const auto root = std::filesystem::absolute(argv[1]);
    require(!std::filesystem::exists(root), "evidence already exists");
    std::filesystem::create_directories(root);
    const auto target = root / "results.txt";
    const auto marker = root / "results.txt.publish.lock";
    std::ofstream(marker).close();
    // A metadata observer can coexist with the no-share writer because its
    // desired access is zero and it shares the writer's read/write/delete access.
    HANDLE observer = CreateFileW(marker.c_str(), 0,
        FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
        nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    require(observer != INVALID_HANDLE_VALUE, "cannot open metadata observer");
    { ecm_io::FileLock first(target.string()); }
    FILE_STANDARD_INFO info{};
    require(GetFileInformationByHandleEx(observer, FileStandardInfo, &info, sizeof(info)) &&
            info.DeletePending, "observer did not retain a delete-pending marker");
    HANDLE probe = CreateFileW(marker.c_str(), GENERIC_READ | GENERIC_WRITE, 0, nullptr,
        OPEN_ALWAYS, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_DELETE_ON_CLOSE, nullptr);
    const DWORD error = GetLastError();
    if (probe != INVALID_HANDLE_VALUE)
        CloseHandle(probe);
    require(probe == INVALID_HANDLE_VALUE && error == ERROR_ACCESS_DENIED,
            "control did not reproduce delete-pending ERROR_ACCESS_DENIED");
    std::thread release([observer] {
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        CloseHandle(observer);
    });
    const auto start = std::chrono::steady_clock::now();
    bool acquired = false;
    std::string failure;
    try {
        ecm_io::FileLock next(target.string(), ecm_io::FileLockDomain::Publication, 1000);
        acquired = true;
    } catch (const std::exception& e) {
        failure = e.what();
    }
    release.join();
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start).count();
    std::cout << "delete_pending=1 raw_error=" << error << " acquired=" << acquired
              << " elapsed_ms=" << elapsed << " detail=" << failure << '\n';
    require(acquired && elapsed >= 100 && elapsed < 1500,
            "a transient delete-pending marker prevented the next writer");

    // A genuinely unwritable marker still fails within the caller's budget.
    std::ofstream(marker).close();
    require(SetFileAttributesW(marker.c_str(), FILE_ATTRIBUTE_READONLY), "readonly setup");
    const auto denied_start = std::chrono::steady_clock::now();
    bool denied = false;
    try {
        ecm_io::FileLock invalid(target.string(), ecm_io::FileLockDomain::Publication, 200);
    } catch (const std::exception& e) {
        denied = std::string(e.what()).find("system_error=5") != std::string::npos;
    }
    SetFileAttributesW(marker.c_str(), FILE_ATTRIBUTE_NORMAL);
    const auto denied_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - denied_start).count();
    require(denied && denied_ms < 1500, "permission failure did not remain bounded/actionable");
    std::cout << "permanent_denial=1 elapsed_ms=" << denied_ms << " passed=1\n";
    return 0;
} catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
}
