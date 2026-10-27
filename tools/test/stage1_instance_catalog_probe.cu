// Exercise actual linked factories and register attributes without running math.
#include "../../src/ecm/cuda/arithmetic/cgbn/types.cuh"
#include "../../src/ecm/stage1/montgomery/cuda/launch.h"
#include "../../src/ecm/stage1/montgomery/cuda/prac/launch.h"
#include <cstdio>
#include <cstdlib>
#include <map>

int main(int argc, char **argv) {
    const int device = argc > 1 ? std::atoi(argv[1]) : 0;
    if (cudaSetDevice(device) != cudaSuccess)
        return 1;
    const unsigned widths[] = {128,192,256,384,512,768,1024,1280,1536,1792,2048,
        2560,3072,3584,4096,4608,5120,5632,6144,6656,7168,7680,8192,9216,
        10240,11264,12288,13312,14336,15360,16384,17408,999};
    const unsigned requested[] = {0,4,8,16,32,64};
    struct Kernel { unsigned id; cudaFuncAttributes attributes; };
    std::map<cgbn_stage1_kernel_fn, Kernel> kernels;
    std::puts("bits,mode,requested_tpi,available,tpi,kernel_id,registers,local_bytes");
    for (unsigned bits : widths) {
        for (int mode = -1; mode <= 18; ++mode) {
            for (unsigned want : requested) {
                unsigned tpi = 123456;
                const auto fn = cgbn_stage1_domain_dispatch(bits, &tpi, mode, want);
                Kernel value{};
                if (fn) {
                    const auto found = kernels.find(fn);
                    if (found != kernels.end()) value = found->second;
                    else {
                        value.id = static_cast<unsigned>(kernels.size()) + 1;
                        const auto status = cudaFuncGetAttributes(&value.attributes, fn);
                        if (status != cudaSuccess) {
                            std::fprintf(stderr, "attributes: bits=%u mode=%d: %s\n",
                                         bits, mode, cudaGetErrorString(status));
                            return 2;
                        }
                        kernels.emplace(fn, value);
                    }
                }
                std::printf("%u,%d,%u,%u,%u,%u,%d,%llu\n", bits, mode, want,
                    fn ? 1u : 0u, tpi, value.id, value.attributes.numRegs,
                    static_cast<unsigned long long>(value.attributes.localSizeBytes));
            }
        }
    }
    return 0;
}
