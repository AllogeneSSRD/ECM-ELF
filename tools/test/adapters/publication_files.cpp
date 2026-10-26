#include "io/files.h"
#include <windows.h>
#include <algorithm>
#include <cwctype>
#include <map>
#include <set>

namespace {
std::map<HANDLE, std::wstring> handles;
std::set<std::wstring> flushed;
std::wstring target;
int failure = 0;
bool unflushed = false;

std::wstring key(const wchar_t* path) {
    auto value = std::filesystem::absolute(path).lexically_normal().wstring();
    std::transform(value.begin(), value.end(), value.begin(), std::towlower);
    return value;
}
}

void publication_observe(const std::filesystem::path& path, int mode) {
    handles.clear();
    flushed.clear();
    unflushed = false;
    target = key(path.c_str());
    failure = mode;
}

bool publication_unflushed() { return unflushed; }

static HANDLE observed_create(LPCWSTR path, DWORD access, DWORD share,
                              LPSECURITY_ATTRIBUTES security, DWORD disposition,
                              DWORD flags, HANDLE template_file) {
    const auto handle = CreateFileW(path, access, share, security, disposition, flags, template_file);
    if (handle != INVALID_HANDLE_VALUE) {
        handles[handle] = key(path);
        if (disposition == CREATE_ALWAYS)
            flushed.erase(handles[handle]);
    }
    return handle;
}

static BOOL observed_flush(HANDLE handle) {
    const auto found = handles.find(handle);
    if (failure == 1 && found != handles.end() && found->second.rfind(target, 0) == 0) {
        SetLastError(ERROR_WRITE_FAULT);
        return FALSE;
    }
    const auto ok = FlushFileBuffers(handle);
    if (ok && found != handles.end())
        flushed.insert(found->second);
    return ok;
}

static BOOL observed_move(LPCWSTR from, LPCWSTR to, DWORD flags) {
    const bool watched = key(to) == target;
    if (watched && !flushed.count(key(from)))
        unflushed = true;
    if (watched && failure == 2) {
        SetLastError(ERROR_ACCESS_DENIED);
        return FALSE;
    }
    const auto ok = MoveFileExW(from, to, flags);
    if (ok && watched && failure == 3)
        throw ecm_io::PublicationUncertain("injected acknowledgement failure after real replacement");
    return ok;
}

// Full production implementation, with only Windows call observation/failure
// injection. Files and locks are real; no publication logic is reimplemented.
#define CreateFileW observed_create
#define FlushFileBuffers observed_flush
#define MoveFileExW observed_move
#include "../../../src/io/files.cpp"
#undef CreateFileW
#undef FlushFileBuffers
#undef MoveFileExW
