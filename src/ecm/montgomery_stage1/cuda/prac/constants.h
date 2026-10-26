#ifndef CGBN_STAGE1_PRAC_CONSTANTS_H
#define CGBN_STAGE1_PRAC_CONSTANTS_H

// Private factory seam; include after cgbn_stage1_kernel.h. These policies use
// the existing Montgomery domain/data ABI and require explicit host validation.
#include "domain.h"
cgbn_stage1_kernel_fn cgbn_stage1_domain_constants(uint32_t bits, uint32_t *tpi, int mode);

#endif
