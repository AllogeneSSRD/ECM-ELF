#include "debug_utils.h"
#include "../../../../../../include/ecm/platform/logging.h"
#include "../../../../../../include/ecm/application/configuration/runtime.h"

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <gmp.h>

void ocl_log_verbose(int verbose, const char *fmt, ...) {
    if (verbose < 1) return;
    va_list ap;
    va_start(ap, fmt);
    ecm_ts_vfprintf(stderr, fmt, ap);
    va_end(ap);
}

void opencl_dump_begin(opencl_dump_ctx_t &ctx, int verbose) {
    ctx.enabled = ecm_runtime_config().gpu_dump;
    if (!ctx.enabled) return;

    const char *dump_path = ecm_runtime_config().gpu_dump_file.c_str();
    ctx.out.open(dump_path, std::ios::out | std::ios::trunc);
    if (!ctx.out.is_open()) {
        ctx.enabled = false;
        ecm_ts_fprintf(stderr, "OpenCL: failed to open dump output file\n");
        return;
    }
    ctx.out << "stage,batch_index,s_partial,batch_size,sigma,curve_index,BITS,TPI,word0,word1,word2,word3,word4\n";
    ocl_log_verbose(verbose, "OpenCL dump enabled: writing kernel call states to %s\n", dump_path);
}

void opencl_dump_end(opencl_dump_ctx_t &ctx) {
    if (ctx.out.is_open()) {
        ctx.out.close();
    }
    ctx.enabled = false;
}

void dump_opencl_state_rows(opencl_dump_ctx_t &ctx, const char *stage, int batch_index,
                            uint64_t s_partial, uint64_t batch_size, uint32_t sigma,
                            uint32_t curves, uint32_t bits, uint32_t tpi,
                            const uint32_t *data, uint32_t limbs) {
    if (!ctx.enabled || !ctx.out.is_open()) return;

    auto to_mpz_local = [](mpz_t r, const uint32_t *x, uint32_t count) {
        mpz_import(r, count, -1, sizeof(uint32_t), 0, 0, x);
    };

    mpz_t v;
    mpz_init(v);
    const uint32_t stride = 5u * limbs;
    for (uint32_t curve = 0; curve < curves; ++curve) {
        const uint32_t *datum = data + curve * stride;
        ctx.out << stage << "," << batch_index << "," << s_partial << "," << batch_size << ","
                << sigma << "," << curve << "," << bits << "," << tpi;
        for (uint32_t slot = 0; slot < 5; ++slot) {
            to_mpz_local(v, datum + slot * limbs, limbs);
            char *hex = mpz_get_str(nullptr, 16, v);
            ctx.out << ",0x" << hex;
            free(hex);
        }
        ctx.out << "\n";
    }
    mpz_clear(v);
    ctx.out.flush();
}
