#pragma once
#include "prac_descriptor.h"
enum { ECM_DOMAIN_INIT = 0, ECM_DOMAIN_EXPORT = 1, ECM_DOMAIN_LADDER = 2,
       ECM_DOMAIN_PRAC = 3, ECM_DOMAIN_PRAC_NATURAL = 4, ECM_DOMAIN_PRAC_168 = 5,
       ECM_DOMAIN_PRAC_COMPACT = 6, ECM_DOMAIN_PRAC_COMPACT_168 = 7,
       ECM_DOMAIN_PRAC_OUTLINE_ADD = 8, ECM_DOMAIN_PRAC_OUTLINE_ADD_168 = 9,
       ECM_DOMAIN_PRAC_SINGLE_ADD = 10, ECM_DOMAIN_PRAC_SINGLE_ADD_168 = 11,
       ECM_DOMAIN_PRAC_SINGLE_COMPACT = 12, ECM_DOMAIN_PRAC_SINGLE_COMPACT_168 = 13,
       ECM_DOMAIN_PRAC_SINGLE_COMPACT_128 = 14 };
cgbn_stage1_kernel_fn cgbn_stage1_domain_dispatch(uint32_t bits, uint32_t *tpi, int mode,
                                                uint32_t requested_tpi = 0);
