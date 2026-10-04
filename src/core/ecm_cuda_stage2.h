#pragma once
#include <cstdint>

// One normalized param0 Stage1 point per process. The callback is invoked only
// after Stage2 and its arithmetic checks have completed successfully.
int ecm_cuda_stage2_run(const char *n_hex, const char *x_hex, uint64_t sigma,
                       uint64_t b1, uint64_t b2, uint64_t d, int device,
                       void (*report)(const char *, void *), void *context);
