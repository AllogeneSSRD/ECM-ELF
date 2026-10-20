#pragma once
#include <cstdlib>
#include <cstring>
#ifdef _WIN32
// NVTX's Windows loader includes windows.h. Keep optional RPC/dialog macros
// (notably small and scr2) out of the arithmetic/planning headers.
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#endif
#if __has_include(<nvtx3/nvToolsExt.h>)
#include <nvtx3/nvToolsExt.h>
#define ECM_STAGE2_NVTX_AVAILABLE 1
#else
#define ECM_STAGE2_NVTX_AVAILABLE 0
#endif

namespace stage2_trace {
inline bool enabled() {
    static const bool value=[] {
        const char *s=std::getenv("NTT_HOST_TRACE");return s && !std::strcmp(s,"1");
    }();
    return value;
}
struct Range {
    const bool active=enabled();
    explicit Range(const char *name) {
#if ECM_STAGE2_NVTX_AVAILABLE
        if(active)nvtxRangePushA(name);
#endif
    }
    ~Range() {
#if ECM_STAGE2_NVTX_AVAILABLE
        if(active)nvtxRangePop();
#endif
    }
    Range(const Range&)=delete;
    Range& operator=(const Range&)=delete;
};
}
