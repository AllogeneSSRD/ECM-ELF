#include "../../../../cuda/arithmetic/cgbn/types.cuh"
#include "../launch.h"
#include "launch.h"

#define ECM_STAGE1_PRAC_INSTANCE(F,W,T,FN) cgbn_stage1_kernel_fn FN(uint32_t, uint32_t *, int);
#include <stage1_prac_instances.inc>
#undef ECM_STAGE1_PRAC_INSTANCE

namespace {
enum class Family { standard, alternate, single, constants };
using Factory = cgbn_stage1_kernel_fn (*)(uint32_t, uint32_t *, int);
struct Instance {
    uint32_t bits;
    uint32_t tpi;
    Family family;
    Factory factory;
};
const Instance instances[] = {
#define ECM_STAGE1_PRAC_INSTANCE(F,W,T,FN) {W, T, Family::F, FN},
#include <stage1_prac_instances.inc>
#undef ECM_STAGE1_PRAC_INSTANCE
    {0, 0, Family::standard, nullptr}
};

cgbn_stage1_kernel_fn find(Family family, uint32_t bits, uint32_t wanted_tpi,
                          uint32_t *tpi, int mode) {
    for (const auto &instance : instances) {
        if (instance.factory && instance.family == family && instance.bits == bits &&
            (!wanted_tpi || instance.tpi == wanted_tpi))
            return instance.factory(bits, tpi, mode);
    }
    *tpi = 0;
    return nullptr;
}
}

cgbn_stage1_kernel_fn cgbn_stage1_domain_dispatch(uint32_t bits, uint32_t *tpi,
                                                int mode, uint32_t requested_tpi) {
    if (mode == ECM_DOMAIN_PRAC_CONSTANT_RUNTIME || mode == ECM_DOMAIN_PRAC_CONSTANT_NP0 ||
        mode == ECM_DOMAIN_PRAC_CONSTANT_M4423) {
        if (!requested_tpi || requested_tpi == 16)
            return find(Family::constants, bits, 16, tpi, mode);
        *tpi = 0;
        return nullptr;
    }
    if (mode >= ECM_DOMAIN_PRAC_SINGLE_ADD && mode <= ECM_DOMAIN_PRAC_SINGLE_COMPACT_128) {
        if (!requested_tpi || requested_tpi == 16)
            return find(Family::single, bits, 16, tpi, mode);
        if (requested_tpi == 32 && (mode == ECM_DOMAIN_PRAC_SINGLE_COMPACT ||
                                    mode == ECM_DOMAIN_PRAC_SINGLE_COMPACT_168))
            return find(Family::single, bits, 32, tpi, mode);
        *tpi = 0;
        return nullptr;
    }
    auto kernel = find(Family::standard, bits, 0, tpi, mode);
    if (kernel && (!requested_tpi || requested_tpi == *tpi))
        return kernel;
    if (requested_tpi == 32)
        return find(Family::alternate, bits, 32, tpi, mode);
    *tpi = 0;
    return nullptr;
}
