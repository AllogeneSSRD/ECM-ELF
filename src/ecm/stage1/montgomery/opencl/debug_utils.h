#pragma once

#include <cstdint>
#include <fstream>

struct opencl_dump_ctx_t {
    bool enabled = false;
    std::ofstream out;
};

void ocl_log_verbose(int verbose, const char *fmt, ...);

void opencl_dump_begin(opencl_dump_ctx_t &ctx, int verbose);
void opencl_dump_end(opencl_dump_ctx_t &ctx);
void dump_opencl_state_rows(opencl_dump_ctx_t &ctx, const char *stage, int batch_index,
                            uint64_t s_partial, uint64_t batch_size, uint32_t sigma,
                            uint32_t curves, uint32_t bits, uint32_t tpi,
                            const uint32_t *data, uint32_t limbs);
