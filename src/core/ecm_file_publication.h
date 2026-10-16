#pragma once
#include <filesystem>
#include <string>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

// A scanner or reader without delete sharing can briefly block atomic rename.
// Bound retries; permanent access errors must still preserve the pending file.
inline bool ecm_replace_file(const std::filesystem::path &source,const std::filesystem::path &target) {
#ifdef _WIN32
    const auto deadline=GetTickCount64()+500;
    for(;;) {
        if(MoveFileExW(source.c_str(),target.c_str(),MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH))return true;
        const auto error=GetLastError();
        if((error!=ERROR_ACCESS_DENIED && error!=ERROR_SHARING_VIOLATION && error!=ERROR_LOCK_VIOLATION) ||
           GetTickCount64()>=deadline){SetLastError(error);return false;}
        Sleep(5);
    }
#else
    std::error_code error;std::filesystem::rename(source,target,error);return !error;
#endif
}
