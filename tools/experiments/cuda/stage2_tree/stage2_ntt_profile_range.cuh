#pragma once
#include <cuda_profiler_api.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Host-only instrumentation, copied into an isolated frozen-source project.
// There is no production include or device code in this header.
static bool s2_ncu_range_begin(int kind, bool inverse, unsigned long long n,
                               unsigned long long nbatch, int parameter)
{
    const char *target = std::getenv("S2_NCU_TARGET");
    if (!target || !*target) return false;
    static bool fired = false;
    if (fired) return false;
    char name[96] = {};
    const char *direction = inverse ? "inverse" : "forward";
    if (n == (1ull << 27) && nbatch == 1) {
        if (kind == 0 && parameter == 12)
            std::snprintf(name,sizeof(name),"n27_tile_%s",direction);
        if (kind == 1 && (parameter == 7 || parameter == 8))
            std::snprintf(name,sizeof(name),"n27_outer_m%d_%s",parameter,direction);
    }
    if (n == (1ull << 11) && nbatch == 990 && kind == 0 && parameter == 11)
        std::snprintf(name,sizeof(name),"n11_b990_tile_%s",direction);
    if (std::strcmp(name,target)) return false;
    const cudaError_t status = cudaProfilerStart();
    if (status != cudaSuccess) {
        std::fprintf(stderr,"FATAL: diagnostic profiler start: %s\n",cudaGetErrorString(status));
        std::exit(3);
    }
    fired = true;
    std::printf("s2_ncu_range: target=%s n=%llu nbatch=%llu parameter=%d begin=1\n",
        name,n,nbatch,parameter);
    std::fflush(stdout);
    return true;
}

static void s2_ncu_range_end(bool active)
{
    if (!active) return;
    const cudaError_t status = cudaProfilerStop();
    if (status != cudaSuccess) {
        std::fprintf(stderr,"FATAL: diagnostic profiler stop: %s\n",cudaGetErrorString(status));
        std::exit(3);
    }
    std::printf("s2_ncu_range_end: end=1\n");
    std::fflush(stdout);
}
