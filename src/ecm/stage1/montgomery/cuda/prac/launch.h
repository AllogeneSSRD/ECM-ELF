#pragma once
#include "descriptor.h"
#include "domain.h"
cgbn_stage1_kernel_fn cgbn_stage1_domain_dispatch(uint32_t bits, uint32_t *tpi, int mode,
                                                uint32_t requested_tpi = 0);
