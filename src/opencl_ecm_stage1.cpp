// OpenCL ECM Stage 1 host (mirrors test/cgbn_stage1.cu)

#include "cgbn_stage1.h"
#include "ecm.h"
#include "cgbn_opencl.h"
#include "ecm_checkpoint.h"
#include "opencl_ecm_debug_utils.h"
#include "opencl_ecm_log.h"
#include "opencl_ecm_mont.h"
#include "opencl_ecm_path_registry.h"
#include "opencl_ecm_runtime_config.h"
#include "opencl_ecm_selftest.h"

#include <CL/cl.h>
#include <gmp.h>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#include <io.h>
#include <process.h>
#ifndef getpid
#define getpid _getpid
#endif
#else
#include <unistd.h>
#endif
#include <ctime>

#include <atomic>
#include <csignal>

// Progress bar (indicators header-only library)
#include "indicators/indicators.hpp"

// ── Interrupt handling ──────────────────────────────────────────────────
static std::atomic<bool> g_interrupted{false};

#ifdef _WIN32
#include <windows.h>
static BOOL WINAPI console_ctrl_handler(DWORD ctrl_type) {
    if (ctrl_type == CTRL_C_EVENT || ctrl_type == CTRL_BREAK_EVENT) {
        static int count = 0;
        if (++count > 1) return FALSE;          // second Ctrl+C → let OS kill
        g_interrupted.store(true, std::memory_order_relaxed);
        // Reset terminal attributes immediately.
        const char *reset = "\033[0m\r\n";
        _write(2, reset, (unsigned int)strlen(reset));
        return TRUE;   // TRUE = we handled it, do NOT terminate the process
    }
    return FALSE;
}
#else
static void signal_handler(int) {
    g_interrupted.store(true, std::memory_order_relaxed);
    const char *reset = "\033[0m\n";
    write(STDERR_FILENO, reset, strlen(reset));
}
#endif

#define CARRY_BITS ECM_STAGE1_MONT_CARRY_BITS

static cgbn::opencl::context_t g_ctx;
static bool g_ctx_ready = false;
static cl_program g_ecm_program = nullptr;
static cl_kernel g_ecm_kernel = nullptr;
static cl_kernel g_ecm_kernel_local = nullptr;
static cl_kernel g_ecm_kernel_sliced = nullptr;
static cl_kernel g_ecm_kernel_sliced_t16 = nullptr;
static uint32_t g_kernel_limbs = 0;
static uint32_t g_kernel_tpi = 0;
static int g_kernel_coop_wg = 1;
static bool g_kernel_use_coop_wg = false;
static uint32_t g_kernel_local_wg = 0;
static EcmStage1KernelBuildPlan g_kernel_build_plan{};
static bool g_device_info_printed = false;

static int selected_device_index_from_env() {
    return ecm_runtime_config().device_index;
}

static uint32_t requested_tpi_from_env() {
    uint32_t parsed = ecm_runtime_config().tpi;
    if (parsed == 0u || parsed > 32u) {
        ecm_ts_fprintf(stderr, "OpenCL: invalid --tpi=%u, fallback to 8\n", parsed);
        return 8u;
    }
    return parsed;
}

static bool is_power_of_two_u32(uint32_t x) {
    return x != 0u && (x & (x - 1u)) == 0u;
}

static int ensure_opencl_context_ready() {
    if (!g_ctx_ready) {
        cl_int err = cgbn::opencl::create_context(g_ctx);
        if (err != CL_SUCCESS) {
            ecm_ts_fprintf(stderr, "OpenCL: failed to create context (%d)\n", err);
            return -1;
        }
        g_ctx_ready = true;
    }
    return 0;
}

static EcmPathContext make_path_context(uint32_t limbs, cl_device_id dev) {
    EcmPathContext ctx{};
    ctx.limbs = limbs;
    ctx.n_bit_size = 0;
    ctx.container_limbs = limbs;
    ctx.os_mask = ecm_path_host_os_mask();
    ctx.gpu_vendor_mask = 0;
    if (dev != nullptr) {
        char vendor[256] = {};
        clGetDeviceInfo(dev, CL_DEVICE_VENDOR, sizeof(vendor), vendor, nullptr);
        for (char *p = vendor; *p; ++p) {
            *p = (char)std::tolower((unsigned char)*p);
        }
        ctx.gpu_vendor_mask = ecm_path_gpu_vendor_from_cl_vendor_string(vendor);
    }
    return ctx;
}

static uint32_t stage1_container_limbs(size_t n_bit_size, uint32_t limbs32,
                                       const char *gpu_add_path, const char *gpu_sub_path) {
    uint32_t limbs = limbs32;
    // Resolve add/sub at limbs32 to see if the selected path needs a larger container.
    EcmPathContext ctx = make_path_context(limbs32, g_ctx.device);
    ctx.n_bit_size = n_bit_size;
    const EcmAddSubPathDescriptor *add = opencl_ecm_resolve_addmod_path(gpu_add_path, ctx);
    if (add != nullptr && add->max_container_limbs > limbs) {
        limbs = add->max_container_limbs;
    }
    const EcmAddSubPathDescriptor *sub = opencl_ecm_resolve_submod_path(gpu_sub_path, ctx);
    if (sub != nullptr && sub->max_container_limbs > limbs) {
        limbs = sub->max_container_limbs;
    }
    return limbs;
}

static int resolve_addsub_paths(const char *gpu_add_path, const char *gpu_sub_path, size_t n_bit_size,
                                uint32_t limbs, const EcmAddSubPathDescriptor **add_out,
                                const EcmAddSubPathDescriptor **sub_out) {
    if (ensure_opencl_context_ready() != 0) {
        return -1;
    }
    EcmPathContext ctx = make_path_context(limbs, g_ctx.device);
    ctx.n_bit_size = n_bit_size;
    const EcmAddSubPathDescriptor *add = opencl_ecm_resolve_addmod_path(gpu_add_path, ctx);
    if (add == nullptr) {
        ecm_ts_fprintf(stderr,
                       "OpenCL: unknown --add path '%s' (fused, fused_unroll, default,\n"
                       "  unroll_128b, unroll_192b, unroll_256b, unroll_384b, unroll_512b, unroll_4096b,\n"
                       "  asm_128b, asm_192b, asm_256b, asm_384b, asm_512b, asm_4096b,\n"
                       "  asm_b16, asm_b32, fused_unroll_b16, fused_unroll_b32, fused_unroll_auto,\n"
                       "  or with add_mod_ prefix, e.g. add_mod_unroll_256b)\n",
                       gpu_add_path ? gpu_add_path : "");
        return -1;
    }
    const EcmAddSubPathDescriptor *sub = opencl_ecm_resolve_submod_path(gpu_sub_path, ctx);
    if (sub == nullptr) {
        ecm_ts_fprintf(stderr,
                       "OpenCL: unknown --sub path '%s' (fused, fused_unroll, default,\n"
                       "  unroll_128b, unroll_192b, unroll_256b, unroll_384b, unroll_512b, unroll_4096b,\n"
                       "  asm_128b, asm_192b, asm_256b, asm_384b, asm_512b, asm_4096b,\n"
                       "  asm_b16, asm_b32, fused_unroll_b16, fused_unroll_b32, fused_unroll_auto,\n"
                       "  or with sub_mod_ prefix, e.g. sub_mod_unroll_256b)\n",
                       gpu_sub_path ? gpu_sub_path : "");
        return -1;
    }
    if (add->max_container_limbs > 0u && limbs < add->max_container_limbs) {
        ecm_ts_fprintf(stderr, "OpenCL: add path '%s' requires >=%u limbs, got %u limbs\n",
                       add->id, add->max_container_limbs, limbs);
        return -1;
    }
    if (sub->max_container_limbs > 0u && limbs < sub->max_container_limbs) {
        ecm_ts_fprintf(stderr, "OpenCL: sub path '%s' requires >=%u limbs, got %u limbs\n",
                       sub->id, sub->max_container_limbs, limbs);
        return -1;
    }
    *add_out = add;
    *sub_out = sub;
    return 0;
}

static uint32_t choose_effective_tpi(uint32_t limbs) {
    static const uint32_t kChoices[] = {32u, 16u, 8u, 4u, 2u, 1u};
    uint32_t requested = requested_tpi_from_env();
    if (!is_power_of_two_u32(requested)) {
        ecm_ts_fprintf(stderr, "OpenCL: ECM_OPENCL_TPI=%u is not power-of-two, fallback to 8\n",
                       requested);
        requested = 8u;
    }

    uint32_t chosen = 0u;
    for (uint32_t d : kChoices) {
        if (d <= requested && (limbs % d) == 0u) {
            chosen = d;
            break;
        }
    }
    if (chosen == 0u) {
        for (uint32_t d : kChoices) {
            if ((limbs % d) == 0u) {
                chosen = d;
                break;
            }
        }
    }
    if (chosen == 0u) {
        chosen = 1u;
    }
    if (chosen != requested) {
        ecm_ts_fprintf(stderr,
                       "OpenCL: requested TPI=%u incompatible with limbs=%u, using TPI=%u\n",
                       requested, limbs, chosen);
    }
    return chosen;
}

static opencl_dump_ctx_t g_dump_ctx;

struct ecm_ops_profile_counts_t {
    uint64_t kernel_bits_processed = 0;   // bits processed by kernel loop only
    uint64_t double_add_calls = 0;        // per curve, per processed bit
    uint64_t mp_add_mod = 0;
    uint64_t mp_sub_mod = 0;
    uint64_t mont_mul_priv = 0;
    uint64_t mont_sqr_priv = 0;
    uint64_t mont_normalize = 0;
    uint64_t special_mult_ui32 = 0;
    uint64_t mp_shift_left_1_mod = 0;
};

static ecm_ops_profile_counts_t compute_ops_profile_counts(uint64_t kernel_bits_processed,
                                                           uint32_t curves) {
    const uint64_t calls = kernel_bits_processed * (uint64_t)curves;
    ecm_ops_profile_counts_t c;
    c.kernel_bits_processed = kernel_bits_processed;
    c.double_add_calls = calls;
    c.mp_add_mod = calls * 4ull;
    c.mp_sub_mod = calls * 4ull;
    c.mont_mul_priv = calls * 4ull;
    c.mont_sqr_priv = calls * 4ull;
    c.mont_normalize = calls * 8ull;
    c.special_mult_ui32 = calls;
    c.mp_shift_left_1_mod = calls;
    return c;
}

static void emit_ops_profile(const ecm_ops_profile_counts_t &c, uint32_t curves, uint64_t s_num_bits,
                             int batches_complete, float gputime_ms, int verbose) {
    (void)verbose;
    ecm_ts_fprintf(stdout,
            "ECM_PROFILE_OPS: curves=%u s_bits=%llu kernel_bits=%llu batches=%d gputime_ms=%.3f\n",
            curves, (unsigned long long)s_num_bits, (unsigned long long)c.kernel_bits_processed,
            batches_complete, gputime_ms);
    ecm_ts_fprintf(stdout,
            "ECM_PROFILE_OPS: double_add_v2=%llu, mp_add_mod=%llu, mp_sub_mod=%llu, "
            "mont_mul_priv=%llu, mont_sqr_priv=%llu, mont_normalize=%llu, "
            "special_mult_ui32=%llu, mp_shift_left_1_mod=%llu\n",
            (unsigned long long)c.double_add_calls, (unsigned long long)c.mp_add_mod,
            (unsigned long long)c.mp_sub_mod, (unsigned long long)c.mont_mul_priv,
            (unsigned long long)c.mont_sqr_priv, (unsigned long long)c.mont_normalize,
            (unsigned long long)c.special_mult_ui32,
            (unsigned long long)c.mp_shift_left_1_mod);
    fflush(stdout);

    if (!ecm_runtime_config().profile_ops) {
        return;
    }
    const char *csv_path = ecm_runtime_config().profile_ops_file.c_str();
    std::ofstream out(csv_path, std::ios::out | std::ios::app);
    if (!out.is_open()) {
        ecm_ts_fprintf(stderr, "ECM_PROFILE_OPS: failed to open %s for append\n", csv_path);
        return;
    }
    if (out.tellp() == std::streampos(0)) {
        out << "curves,s_num_bits,kernel_bits_processed,batches,gputime_ms,double_add_calls,"
               "mp_add_mod,mp_sub_mod,mont_mul_priv,mont_sqr_priv,mont_normalize,"
               "special_mult_ui32,mp_shift_left_1_mod\n";
    }
    out << curves << "," << s_num_bits << "," << c.kernel_bits_processed << "," << batches_complete
        << "," << gputime_ms << "," << c.double_add_calls << "," << c.mp_add_mod << ","
        << c.mp_sub_mod << "," << c.mont_mul_priv << "," << c.mont_sqr_priv << ","
        << c.mont_normalize << "," << c.special_mult_ui32 << "," << c.mp_shift_left_1_mod
        << "\n";
}


static void print_opencl_device_info(int device_index, double init_ms) {
    if (!g_ctx_ready) {
        return;
    }

    char dev_name[512] = {0};
    char dev_version[256] = {0};
    char driver_version[256] = {0};
    cl_uint compute_units = 0;
    cl_ulong local_mem = 0;
    size_t max_wg = 0;
    cl_ulong max_mem_alloc = 0;

    clGetDeviceInfo(g_ctx.device, CL_DEVICE_NAME, sizeof(dev_name) - 1, dev_name, nullptr);
    clGetDeviceInfo(g_ctx.device, CL_DEVICE_VERSION, sizeof(dev_version) - 1, dev_version,
                    nullptr);
    clGetDeviceInfo(g_ctx.device, CL_DRIVER_VERSION, sizeof(driver_version) - 1,
                    driver_version, nullptr);
    clGetDeviceInfo(g_ctx.device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(compute_units),
                    &compute_units, nullptr);
    clGetDeviceInfo(g_ctx.device, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(local_mem), &local_mem,
                    nullptr);
    clGetDeviceInfo(g_ctx.device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(max_wg), &max_wg,
                    nullptr);
    clGetDeviceInfo(g_ctx.device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(max_mem_alloc),
                    &max_mem_alloc, nullptr);

    ecm_ts_fprintf(stdout, "GPU: will use device %d: %s, %s, %u compute units.\n", device_index,
            dev_name, dev_version, compute_units);
    ecm_ts_fprintf(stdout, "GPU: driver %s\n", driver_version);
    ecm_ts_fprintf(stdout,
            "GPU: maxSharedPerBlock = %lu maxThreadsPerBlock = %zu maxMemAllocPerBuffer = %lu\n",
            (unsigned long)local_mem, max_wg, (unsigned long)max_mem_alloc);
    ecm_ts_fprintf(stdout, "GPU: Selection and initialization of the device took %.0fms\n", init_ms);
    fflush(stdout);
}

// gpu_pick_random_sigma() and gpu_compute_batch_d() moved to
// src/core/ecm_gpu_common.cpp (backend-neutral, shared by ecm and ecm_cuda).

static uint32_t *allocate_and_set_s_bits(const mpz_t s, uint64_t *nbits) {
    uint64_t num_bits = *nbits = mpz_sizeinbase(s, 2);
    uint64_t allocated = (num_bits + 31) / 32;
    uint32_t *s_bits = (uint32_t *)malloc(sizeof(uint32_t) * allocated);
    uint64_t countp;
    mpz_export(s_bits, &countp, -1, sizeof(uint32_t), 0, 0, s);
    for (uint64_t i = countp; i < allocated; ++i) {
        s_bits[i] = 0;
    }
    return s_bits;
}

static void stage1_curves_to_montgomery(uint32_t *data, uint32_t curves, uint32_t limbs,
                                        const mpz_t N, uint32_t bits) {
    const uint32_t stride = 5u * limbs;
    mpz_t t;
    mpz_init(t);
    for (uint32_t c = 0; c < curves; ++c) {
        uint32_t *datum = data + c * stride;
        for (uint32_t slot = 1; slot <= 4; ++slot) {
            ecm_to_mpz(t, datum + slot * limbs, limbs);
            ecm_to_montgomery(datum + slot * limbs, t, N, bits, limbs);
        }
    }
    mpz_clear(t);
}

// CUDA set_p_2p: N, P=(2,1), 2P=(9, 64*d+8) in standard form; bn2mont before each GPU batch.
static uint32_t *set_p_2p(const mpz_t N, uint32_t curves, uint32_t sigma, uint32_t limbs,
                          size_t *data_size) {
    *data_size = 5 * curves * limbs * sizeof(uint32_t);
    uint32_t *data = (uint32_t *)malloc(*data_size);
    uint32_t *datum = data;

    mpz_t x, t;
    mpz_init(x);
    mpz_init(t);

    for (uint32_t index = 0; index < curves; index++) {
        uint32_t d = sigma + index;

        ecm_from_mpz(N, datum + 0 * limbs, limbs);

        mpz_set_ui(x, 2);
        ecm_from_mpz(x, datum + 1 * limbs, limbs);
        mpz_set_ui(x, 1);
        ecm_from_mpz(x, datum + 2 * limbs, limbs);

        mpz_set_ui(x, 9);
        ecm_from_mpz(x, datum + 3 * limbs, limbs);

        mpz_ui_pow_ui(t, 2, 32);
        mpz_invert(t, t, N);
        mpz_mul_ui(t, t, d);
        mpz_mul_ui(t, t, 64);
        mpz_add_ui(t, t, 8);
        mpz_mod(t, t, N);
        ecm_from_mpz(t, datum + 4 * limbs, limbs);

        datum += 5 * limbs;
    }

    mpz_clear(x);
    mpz_clear(t);
    return data;
}

static int findfactor(mpz_t factor, const mpz_t N, const mpz_t x_final, const mpz_t z_final) {
    if (mpz_invert(factor, z_final, N)) {
        mpz_mul(factor, x_final, factor);
        mpz_mod(factor, factor, N);
        return ECM_NO_FACTOR_FOUND;
    }
    mpz_gcd(factor, z_final, N);
    return ECM_FACTOR_FOUND_STEP1;
}

static int process_results(mpz_t *factors, int *array_found, const mpz_t N,
                           const uint32_t *data, uint32_t limbs, int curves, uint32_t sigma,
                           int verbose) {
    mpz_t x_std, z_std;
    mpz_init(x_std);
    mpz_init(z_std);

    const bool verify_results = ecm_runtime_config().verify_gpu_results;
    const bool verify_strict = ecm_runtime_config().verify_gpu_strict;

    int youpi = ECM_NO_FACTOR_FOUND;
    int errors = 0;
    int verify_errors = 0;

    for (int i = 0; i < curves; i++) {
        const uint32_t *datum = data + (5 * i * limbs);

        mpz_t modulo;
        mpz_init(modulo);
        ecm_to_mpz(modulo, datum + 0 * limbs, limbs);
        if (mpz_cmp(modulo, N) != 0) {
            ecm_ts_fprintf(stderr, "GPU: curve %d modulus mismatch\n", i);
        }
        mpz_clear(modulo);

        mpz_t x_mont, z_mont;
        mpz_init(x_mont);
        mpz_init(z_mont);
        ecm_to_mpz(x_mont, datum + 1 * limbs, limbs);
        ecm_to_mpz(z_mont, datum + 2 * limbs, limbs);
        mpz_set(x_std, x_mont);
        mpz_set(z_std, z_mont);

        mpz_clear(x_mont);
        mpz_clear(z_mont);

        if (mpz_cmp_ui(x_std, 2) == 0 && mpz_cmp_ui(z_std, 1) == 0) {
            errors++;
            if (errors < 10) {
                ecm_ts_fprintf(stderr, "GPU: curve %d may not have computed (still at initial point)\n", i);
            }
        }

        array_found[i] = findfactor(factors[i], N, x_std, z_std);
        if (array_found[i] != ECM_NO_FACTOR_FOUND) {
            if (verify_results) {
                mpz_t rem, gcdz;
                mpz_init(rem);
                mpz_init(gcdz);

                mpz_mod(rem, N, factors[i]);
                if (mpz_cmp_ui(rem, 0) != 0) {
                    verify_errors++;
                    ecm_ts_fprintf(stderr,
                                   "GPU verify: curve %d reported invalid factor (N %% factor != 0), sigma=%u\n",
                                   i, sigma + (uint32_t)i);
                    if (verify_strict) {
                        mpz_clear(rem);
                        mpz_clear(gcdz);
                        mpz_clear(x_std);
                        mpz_clear(z_std);
                        return ECM_ERROR;
                    }
                }

                mpz_gcd(gcdz, z_std, N);
                if (mpz_cmp(gcdz, factors[i]) != 0) {
                    verify_errors++;
                    ecm_ts_fprintf(stderr,
                                   "GPU verify: curve %d gcd(z,N) mismatch reported factor, sigma=%u\n",
                                   i, sigma + (uint32_t)i);
                    if (verify_strict) {
                        mpz_clear(rem);
                        mpz_clear(gcdz);
                        mpz_clear(x_std);
                        mpz_clear(z_std);
                        return ECM_ERROR;
                    }
                }
                mpz_clear(rem);
                mpz_clear(gcdz);
            }
            youpi = array_found[i];
            ecm_ts_fprintf(stderr, "GPU: factor found in Step 1 with curve %d (-sigma %d:%u)\n",
                    i, ECM_PARAM_BATCH_32BITS_D, sigma + (uint32_t)i);
        }
    }

    mpz_clear(x_std);
    mpz_clear(z_std);

    if (errors > 2) {
        return ECM_ERROR;
    }
    if (verify_results && verify_errors > 0) {
        ecm_ts_fprintf(stderr, "GPU verify: detected %d invalid result(s)\n", verify_errors);
        return ECM_ERROR;
    }
    return youpi;
}

static void stage1_clamp_mont_desc(const EcmMontPathDescriptor *&mul,
                                 const EcmMontPathDescriptor *&sqr, size_t n_bit_size,
                                 uint32_t limbs) {
    EcmPathContext ctx = make_path_context(limbs, g_ctx.device);
    ctx.n_bit_size = n_bit_size;
    const uint32_t runtime_mask = ctx.os_mask | ctx.gpu_vendor_mask;
    if (mul != nullptr && !ecm_mont_path_fits(mul, ctx.limbs, runtime_mask)) {
        const EcmMontPathDescriptor *fb = mont_auto_fallback(
            opencl_ecm_mont_mul_registry_entry(0),
            opencl_ecm_mont_mul_registry_count(), limbs, runtime_mask);
        const char *new_name = fb ? fb->cl_name : "?";
        if (!ecm_path_limbs_fits(mul->min_limbs, mul->max_limbs, limbs)) {
            ecm_ts_fprintf(stderr,
                           "Warning: mul %s requires limbs [%u,%u], got %u; using %s\n", mul->id,
                           mul->min_limbs, mul->max_limbs, limbs, new_name);
        } else {
            ecm_ts_fprintf(stderr,
                           "Warning: mul %s does not fit %u-limb container; using %s\n",
                           mul->id, limbs, new_name);
        }
        mul = fb;
    }
    if (sqr != nullptr && !ecm_mont_path_fits(sqr, ctx.limbs, runtime_mask)) {
        const EcmMontPathDescriptor *fb = mont_auto_fallback(
            opencl_ecm_mont_sqr_registry_entry(0),
            opencl_ecm_mont_sqr_registry_count(), limbs, runtime_mask);
        const char *new_name = fb ? fb->cl_name : "?";
        if (!ecm_path_limbs_fits(sqr->min_limbs, sqr->max_limbs, limbs)) {
            ecm_ts_fprintf(stderr,
                           "Warning: sqr %s requires limbs [%u,%u], got %u; using %s\n", sqr->id,
                           sqr->min_limbs, sqr->max_limbs, limbs, new_name);
        } else {
            ecm_ts_fprintf(stderr,
                           "Warning: sqr %s does not fit %u-limb container; using %s\n",
                           sqr->id, limbs, new_name);
        }
        sqr = fb;
    }
}

static uint32_t select_bits(size_t n_log2) {
    static const uint32_t candidates[] = {
        192, 256, 384, 512, 768, 1024, 1280, 1536, 1792, 2048, 2560, 3072, 3584, 4096,
        4608, 5120, 5632, 6144, 6656, 7168, 7680, 8192, 8704, 9216};
    for (uint32_t b : candidates) {
        if (n_log2 + CARRY_BITS <= b) {
            return b;
        }
    }
    return 0;
}

static int ensure_ecm_kernel(const EcmStage1KernelBuildPlan &plan, int verbose,
                             double *device_init_ms) {
    const uint32_t limbs = plan.limbs;
    const uint32_t tpi = plan.tpi;
    int coop_wg = 1;
    if (plan.mul != nullptr) {
        coop_wg = std::max(coop_wg, static_cast<int>(plan.mul->coop_work_group_size));
    }
    if (plan.sqr != nullptr) {
        coop_wg = std::max(coop_wg, static_cast<int>(plan.sqr->coop_work_group_size));
    }
    const bool use_coop_wg = (coop_wg > 1);
    if (!g_ctx_ready) {
        cl_int err = cgbn::opencl::create_context(g_ctx);
        if (err != CL_SUCCESS) {
            ecm_ts_fprintf(stderr, "OpenCL: failed to create context (%d)\n", err);
            return -1;
        }
        g_ctx_ready = true;
    }
    if (g_ecm_kernel && opencl_ecm_stage1_build_plan_equal(g_kernel_build_plan, plan)) {
        if (device_init_ms) {
            *device_init_ms = 0.0;
        }
        return 0;
    }

    auto t_init0 = std::chrono::high_resolution_clock::now();
    if (g_ecm_kernel) {
        clReleaseKernel(g_ecm_kernel);
        g_ecm_kernel = nullptr;
    }
    if (g_ecm_kernel_local) {
        clReleaseKernel(g_ecm_kernel_local);
        g_ecm_kernel_local = nullptr;
    }
    if (g_ecm_kernel_sliced) {
        clReleaseKernel(g_ecm_kernel_sliced);
        g_ecm_kernel_sliced = nullptr;
    }
    if (g_ecm_kernel_sliced_t16) {
        clReleaseKernel(g_ecm_kernel_sliced_t16);
        g_ecm_kernel_sliced_t16 = nullptr;
    }
    if (g_ecm_program) {
        clReleaseProgram(g_ecm_program);
        g_ecm_program = nullptr;
    }
    g_device_info_printed = false;

    const std::vector<const char *> kernel_paths = opencl_ecm_stage1_kernel_source_paths(plan);
    std::string src = opencl_ecm_stage1_assemble_kernel_source(
        plan, [](const char *rel_path) { return cgbn::opencl::load_ecm_stage1_kernel_file(rel_path); });
    if (src.empty()) {
        for (const char *rel_path : kernel_paths) {
            if (cgbn::opencl::load_ecm_stage1_kernel_file(rel_path).empty()) {
                ecm_ts_fprintf(stderr,
                               "OpenCL: failed to load %s (set ECM_KERNEL_ROOT or run from project root)\n",
                               rel_path);
                return -1;
            }
        }
        ecm_ts_fprintf(stderr, "OpenCL: failed to assemble stage1 kernel source\n");
        return -1;
    }
    EcmStage1KernelBuildPlan build_plan = plan;
    build_plan.stage1_force_normalize = ecm_runtime_config().stage1_force_normalize;
    {
        int fused = ecm_runtime_config().add_mod_fused_unroll;
        build_plan.add_mod_fused_unroll = (fused == 1 || fused == 2) ? fused : 2;
    }
    const std::string opts = opencl_ecm_stage1_generate_build_options(build_plan);
    if (opts.empty()) {
        ecm_ts_fprintf(stderr, "OpenCL: failed to generate build options\n");
        return -1;
    }

    cl_int buildErr = CL_SUCCESS;
    g_ecm_program = cgbn::opencl::build_program_from_source(g_ctx, src.c_str(), opts.c_str(), buildErr);
    if (g_ecm_program == nullptr || buildErr != CL_SUCCESS) {
        if (g_ecm_program != nullptr) {
            size_t log_size = 0;
            clGetProgramBuildInfo(g_ecm_program, g_ctx.device, CL_PROGRAM_BUILD_LOG, 0, nullptr,
                                  &log_size);
            if (log_size > 1) {
                std::string log(log_size, '\0');
                clGetProgramBuildInfo(g_ecm_program, g_ctx.device, CL_PROGRAM_BUILD_LOG, log_size,
                                      &log[0], nullptr);
                ecm_ts_fprintf(stderr, "OpenCL build log:\n%s\n", log.c_str());
            }
            clReleaseProgram(g_ecm_program);
            g_ecm_program = nullptr;
        }
        ecm_ts_fprintf(stderr, "OpenCL: failed to build ecm_stage1.cl\n");
        return -1;
    }

    cl_int err;
    if (plan.local) {
        g_ecm_kernel_local = clCreateKernel(g_ecm_program, "kernel_double_add_local", &err);
        if (err != CL_SUCCESS) {
            ecm_ts_fprintf(stderr, "OpenCL: kernel_double_add_local not found (%d)\n", err);
            return -1;
        }
        g_kernel_local_wg = (plan.wg_size > 0) ? (uint32_t)plan.wg_size
                            : (plan.limbs <= 128u) ? 16u : 8u;
    } else {
        g_ecm_kernel = clCreateKernel(g_ecm_program, "kernel_double_add", &err);
        if (err != CL_SUCCESS) {
            ecm_ts_fprintf(stderr, "OpenCL: kernel_double_add not found (%d)\n", err);
            return -1;
        }
        g_kernel_local_wg = (plan.wg_size > 0) ? (uint32_t)plan.wg_size : 0u;
    }
    // Always build the sliced kernels too (ignored if --sliced not set)
    g_ecm_kernel_sliced = clCreateKernel(g_ecm_program, "kernel_double_add_sliced", &err);
    if (err != CL_SUCCESS) {
        if (g_ecm_kernel_sliced) { clReleaseKernel(g_ecm_kernel_sliced); g_ecm_kernel_sliced = nullptr; }
    }
    g_ecm_kernel_sliced_t16 = clCreateKernel(g_ecm_program, "kernel_double_add_sliced_t16x2", &err);
    if (err != CL_SUCCESS) {
        if (g_ecm_kernel_sliced_t16) { clReleaseKernel(g_ecm_kernel_sliced_t16); g_ecm_kernel_sliced_t16 = nullptr; }
    }
    g_kernel_limbs = limbs;
    g_kernel_tpi = tpi;
    g_kernel_coop_wg = coop_wg;
    g_kernel_use_coop_wg = use_coop_wg;
    g_kernel_build_plan = build_plan;
    auto t_init1 = std::chrono::high_resolution_clock::now();
    double init_ms =
        std::chrono::duration<double, std::milli>(t_init1 - t_init0).count();
    if (device_init_ms) {
        *device_init_ms = init_ms;
    }
    if (!g_device_info_printed) {
        print_opencl_device_info(selected_device_index_from_env(), init_ms);
        g_device_info_printed = true;
    }
    ocl_log_verbose(verbose,
                    "OpenCL: built kernel MAX_LIMBS=%u TPI=%u ADDMOD_UNROLL=%d NORM=%d "
                    "mul=%s sqr=%s add=%s sub=%s special_mult=%s asm_u32=%d asm_u16=%d coop_wg=%d (%.0fms)\n",
                    limbs, tpi, build_plan.add_mod_fused_unroll, build_plan.stage1_force_normalize,
                    plan.mul != nullptr ? plan.mul->id : "unknown",
                    plan.sqr != nullptr ? plan.sqr->id : "unknown",
                    plan.add != nullptr ? plan.add->id : "unknown",
                    plan.sub != nullptr ? plan.sub->id : "unknown",
                    plan.special_mult != nullptr ? plan.special_mult->id : "unknown",
                    ((plan.add != nullptr && plan.add->kernel_path != nullptr &&
                      strcmp(plan.add->kernel_path, "add_mod/asm_4096b.cl") == 0) ||
                     (plan.sub != nullptr && plan.sub->kernel_path != nullptr &&
                      strcmp(plan.sub->kernel_path, "sub_mod/asm_4096b.cl") == 0))
                        ? 1
                        : 0,
                    ((plan.add != nullptr && plan.add->kernel_path != nullptr &&
                      strcmp(plan.add->kernel_path, "add_mod/asm_512b.cl") == 0) ||
                     (plan.sub != nullptr && plan.sub->kernel_path != nullptr &&
                      strcmp(plan.sub->kernel_path, "sub_mod/asm_512b.cl") == 0))
                        ? 1
                        : 0,
                    use_coop_wg ? coop_wg : 1, init_ms);
    return 0;
}

static int resolve_mont_paths(const char *gpu_mul_path, const char *gpu_sqr_path, size_t n_bit_size,
                            uint32_t container_limbs, const EcmMontPathDescriptor **mul_out,
                            const EcmMontPathDescriptor **sqr_out) {
    EcmPathContext ctx = make_path_context(container_limbs, g_ctx.device);
    ctx.n_bit_size = n_bit_size;
    bool unknown = false;
    const EcmMontPathDescriptor *mul = opencl_ecm_resolve_mont_mul(gpu_mul_path, ctx, &unknown);
    if (unknown) {
        ecm_ts_fprintf(stderr, "OpenCL: unknown --mul path '%s'\n",
                       gpu_mul_path ? gpu_mul_path : "");
        return -1;
    }
    unknown = false;
    const EcmMontPathDescriptor *sqr = opencl_ecm_resolve_mont_sqr(gpu_sqr_path, ctx, &unknown);
    if (unknown) {
        ecm_ts_fprintf(stderr, "OpenCL: unknown --sqr path '%s'\n",
                       gpu_sqr_path ? gpu_sqr_path : "");
        return -1;
    }
    *mul_out = mul;
    *sqr_out = sqr;
    return 0;
}

static uint32_t stage1_ckpt_limbs(size_t data_size, uint32_t curves) {
    return static_cast<uint32_t>(data_size / (5ull * curves * sizeof(uint32_t)));
}

extern "C" int gpu_prepare_opencl(size_t n_log2, int verbose, const char *gpu_mul_path,
                                  const char *gpu_sqr_path, const char *gpu_add_path,
                                  const char *gpu_sub_path,
                                  const char *gpu_special_mult_path) {
    if (ensure_opencl_context_ready() != 0) {
        return ECM_ERROR;
    }
    const uint32_t bits32 = select_bits(n_log2);
    if (bits32 == 0u) {
        return ECM_ERROR;
    }
    const uint32_t limbs32 = bits32 / 32u;

    const uint32_t limbs =
        stage1_container_limbs(n_log2, limbs32, gpu_add_path, gpu_sub_path);

    const EcmMontPathDescriptor *mul = nullptr;
    const EcmMontPathDescriptor *sqr = nullptr;
    if (resolve_mont_paths(gpu_mul_path, gpu_sqr_path, n_log2, limbs, &mul, &sqr) != 0) {
        return ECM_ERROR;
    }
    const EcmAddSubPathDescriptor *add = nullptr;
    const EcmAddSubPathDescriptor *sub = nullptr;
    if (resolve_addsub_paths(gpu_add_path, gpu_sub_path, n_log2, limbs, &add, &sub) != 0) {
        return ECM_ERROR;
    }
    double init_ms = 0.0;
    const uint32_t tpi = choose_effective_tpi(limbs);
    stage1_clamp_mont_desc(mul, sqr, n_log2, limbs);
    EcmPathContext ctx_sm = make_path_context(limbs, g_ctx.device);
    ctx_sm.n_bit_size = n_log2;
    const EcmSpecialMultPathDescriptor *special_mult =
        opencl_ecm_resolve_special_mult(gpu_special_mult_path, ctx_sm);
    const EcmStage1KernelBuildPlan plan =
        opencl_ecm_stage1_make_build_plan(limbs, tpi, mul, sqr, add, sub, special_mult, 1, 2,
                                          ecm_runtime_config().gpu_local,
                                          ecm_runtime_config().wg_size);
    return ensure_ecm_kernel(plan, verbose, &init_ms);
}

extern "C" int cgbn_ecm_stage1(mpz_t *factors, int *array_found, const mpz_t N, const mpz_t s,
                    uint32_t curves, uint32_t *sigma_ptr,
                    unsigned long checkpoint_interval_ms, float *gputime, int verbose,
                    const char *gpu_mul_path, const char *gpu_sqr_path, const char *gpu_add_path,
                    const char *gpu_sub_path, const char *gpu_special_mult_path) {
    uint32_t sigma = *sigma_ptr;
    if (sigma == 0 || (uint64_t)sigma + curves > 0xFFFFFFFFull) {
        ecm_ts_fprintf(stderr, "Invalid sigma/curves range\n");
        return ECM_ERROR;
    }

    const size_t n_log2 = mpz_sizeinbase(N, 2);

    uint64_t s_num_bits;
    uint32_t *s_bits = allocate_and_set_s_bits(s, &s_num_bits);
    if (!s_bits) {
        return ECM_ERROR;
    }

    const char *ckpt_filename = opencl_ecm_checkpoint_filename(N);
    opencl_ecm_checkpoint_header_t ckpt_header{};
    uint32_t *ckpt_data = nullptr;
    size_t ckpt_data_size = 0;
    int ckpt_loaded = 0;

    if (opencl_ecm_checkpoint_load(ckpt_filename, &ckpt_header, &ckpt_data, &ckpt_data_size) ==
        ECM_NO_FACTOR_FOUND) {
        if (ckpt_header.curves == curves && ckpt_header.s_num_bits == s_num_bits) {
            ecm_ts_fprintf(stderr,
                           "Resuming from checkpoint: %.1f%% complete (s_partial=%llu/%llu)\n",
                           ckpt_header.s_num_bits
                               ? (100.0 * ckpt_header.s_partial / ckpt_header.s_num_bits)
                               : 0.0,
                           (unsigned long long)ckpt_header.s_partial,
                           (unsigned long long)ckpt_header.s_num_bits);
            if (sigma != ckpt_header.sigma) {
                ocl_log_verbose(verbose, "Checkpoint sigma overrides current sigma: %u -> %u\n",
                                sigma, ckpt_header.sigma);
            }
            ckpt_loaded = 1;
            sigma = ckpt_header.sigma;
        } else {
            ecm_ts_fprintf(stderr,
                           "Checkpoint parameters mismatch (curves or s_num_bits differ), "
                           "starting fresh\n");
            free(ckpt_data);
            ckpt_data = nullptr;
        }
    }

    if (ensure_opencl_context_ready() != 0) {
        free(s_bits);
        return ECM_ERROR;
    }
    const uint32_t bits32_plan = select_bits(n_log2);
    if (bits32_plan == 0u && !ckpt_loaded) {
        ecm_ts_fprintf(stderr, "No OpenCL kernel large enough for N (%zu bits)\n", n_log2);
        free(s_bits);
        return ECM_ERROR;
    }
    const uint32_t limbs32_plan = bits32_plan / 32u;
    const uint32_t expect_limbs =
        stage1_container_limbs(n_log2, limbs32_plan, gpu_add_path, gpu_sub_path);
    const uint32_t expect_bits = expect_limbs * 32u;

    uint32_t limbs = 0;
    uint32_t BITS = 0;
    if (ckpt_loaded) {
        limbs = stage1_ckpt_limbs(ckpt_data_size, curves);
        BITS = ckpt_header.BITS;
        if (limbs != expect_limbs || BITS != expect_bits) {
            ecm_ts_fprintf(stderr,
                           "Checkpoint container mismatch (limbs=%u bits=%u, expected limbs=%u "
                           "bits=%u), starting fresh\n",
                           limbs, BITS, expect_limbs, expect_bits);
            ckpt_loaded = 0;
            free(ckpt_data);
            ckpt_data = nullptr;
        }
    }
    if (!ckpt_loaded) {
        limbs = expect_limbs;
        BITS = expect_bits;
    }

    const uint32_t tpi = ckpt_loaded ? ckpt_header.TPI : choose_effective_tpi(limbs);

    // Print the *actual* sigma now that any checkpoint resume has been applied.
    ecm_ts_fprintf(stdout, "GPU: sigma=%u (param %d, %u curves)%s\n",
                   sigma, ECM_PARAM_BATCH_32BITS_D, curves,
                   ckpt_loaded ? " [restored from checkpoint]" : " [computed]");

    const EcmAddSubPathDescriptor *add = nullptr;
    const EcmAddSubPathDescriptor *sub = nullptr;
    if (resolve_addsub_paths(gpu_add_path, gpu_sub_path, n_log2, limbs, &add, &sub) != 0) {
        free(s_bits);
        if (ckpt_data) {
            free(ckpt_data);
        }
        return ECM_ERROR;
    }

    const EcmMontPathDescriptor *mul = nullptr;
    const EcmMontPathDescriptor *sqr = nullptr;
    if (resolve_mont_paths(gpu_mul_path, gpu_sqr_path, n_log2, limbs, &mul, &sqr) != 0) {
        free(s_bits);
        if (ckpt_data) {
            free(ckpt_data);
        }
        return ECM_ERROR;
    }

    stage1_clamp_mont_desc(mul, sqr, n_log2, limbs);
    EcmPathContext ctx_sm2 = make_path_context(limbs, g_ctx.device);
    ctx_sm2.n_bit_size = n_log2;
    const EcmSpecialMultPathDescriptor *special_mult2 =
        opencl_ecm_resolve_special_mult(gpu_special_mult_path, ctx_sm2);
    const EcmStage1KernelBuildPlan build_plan =
        opencl_ecm_stage1_make_build_plan(limbs, tpi, mul, sqr, add, sub, special_mult2, 1, 2,
                                          ecm_runtime_config().gpu_local,
                                          ecm_runtime_config().wg_size);

    double device_init_ms = 0.0;
    if (ensure_ecm_kernel(build_plan, verbose, &device_init_ms) != 0) {
        free(s_bits);
        if (ckpt_data) {
            free(ckpt_data);
        }
        return ECM_ERROR;
    }

    const uint32_t np0 = ecm_find_np0(N);
    if (!ckpt_loaded && opencl_ecm_selftest_montgomery(N, BITS) != 0) {
        free(s_bits);
        return ECM_ERROR;
    }
    if (!ckpt_loaded && opencl_ecm_selftest_mont_mul(g_ctx, N, BITS, np0) != 0) {
        ecm_ts_fprintf(stderr, "GPU: warning: mont.cl mul self-test failed\n");
    }

    size_t data_size = 0;
    uint32_t *data = nullptr;
    uint64_t s_partial = 1;
    int batches_complete = 0;
    if (ckpt_loaded) {
        data = ckpt_data;
        data_size = ckpt_data_size;
        s_partial = ckpt_header.s_partial;
        batches_complete = ckpt_header.batches_complete;
        ckpt_data = nullptr;
        ocl_log_verbose(verbose, "Checkpoint: restored BITS=%u, TPI=%u, limbs=%u\n", BITS, tpi,
                        limbs);
    } else {
        data = set_p_2p(N, curves, sigma, limbs, &data_size);
        s_partial = 1;
        batches_complete = 0;
    }

    uint64_t batch_size = 10;

    cl_int err;
    uint32_t s_words = (uint32_t)((s_num_bits + 31) / 32);
    cl_mem gpu_s_bits = clCreateBuffer(g_ctx.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       sizeof(uint32_t) * s_words, s_bits, &err);
    cl_mem gpu_data = clCreateBuffer(g_ctx.ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                     data_size, data, &err);

    auto t_global_start = std::chrono::high_resolution_clock::now();

    const uint32_t container_limbs = limbs;
    const uint32_t wg_threads = (
        (mul != nullptr && mul->coop_work_group_size > 1u) ||
        (sqr != nullptr && sqr->coop_work_group_size > 1u))
        ? std::max<uint32_t>(
              mul != nullptr ? mul->coop_work_group_size : 1u,
              sqr != nullptr ? sqr->coop_work_group_size : 1u)
        : 1u;

    const bool is_local = build_plan.local;
    const char *local_tag = is_local ? " local" : "";
    uint32_t wg_actual = (is_local || g_kernel_local_wg > 0u) ? g_kernel_local_wg : wg_threads;
    uint32_t wi_total = curves;
    if (is_local || g_kernel_local_wg > 0u || g_kernel_use_coop_wg) {
        ecm_ts_fprintf(stdout,
                "GPU: OpenCL<%u limbs, %u wg * %u size%s> kernel, %zu-bit N, s=%llu bits, np0=0x%08x\n",
                container_limbs, std::max(wi_total/wg_actual, 1u), wg_actual, local_tag,
                n_log2, (unsigned long long)s_num_bits, np0);
    } else {
        ecm_ts_fprintf(stdout,
                "GPU: OpenCL<%u limbs, %u WI%s> kernel, %zu-bit N, s=%llu bits, np0=0x%08x\n",
                container_limbs, wi_total, local_tag,
                n_log2, (unsigned long long)s_num_bits, np0);
    }
    const char *mont_mul_op = (mul != nullptr && mul->id) ? mul->id : "?";
    const char *mont_sqr_op = (sqr != nullptr && sqr->id) ? sqr->id : "?";
    const char *add_op = add != nullptr ? add->cl_name : "unknown";
    const char *sub_op = sub != nullptr ? sub->cl_name : "unknown";
    const char *special_op = special_mult2 != nullptr ? special_mult2->cl_name : "special_mult_ui32_generic";
    if (wg_threads > 1u) {
        ecm_ts_fprintf(stdout, "GPU: stage1 operators: mul=%s, sqr=%s, add=%s, sub=%s, special_mult=%s, coop_wg=%u\n",
                       mont_mul_op, mont_sqr_op, add_op, sub_op, special_op, wg_threads);
    } else {
        ecm_ts_fprintf(stdout, "GPU: stage1 operators: mul=%s, sqr=%s, add=%s, sub=%s, special_mult=%s\n",
                       mont_mul_op, mont_sqr_op, add_op, sub_op, special_op);
    }
    if (device_init_ms > 0.0) {
        ecm_ts_fprintf(stdout, "GPU: kernel compile/build for this limb size took %.0fms\n",
                device_init_ms);
    }
    if (checkpoint_interval_ms > 0) {
        ecm_ts_fprintf(stderr, "Checkpoint autosave interval: %.0f seconds\n",
                       checkpoint_interval_ms / 1000.0);
    } else {
        ecm_ts_fprintf(stderr, "Checkpoint autosave disabled\n");
    }
    fflush(stdout);

    if (!ckpt_loaded) {
        stage1_curves_to_montgomery(data, curves, limbs, N, BITS);
        err = clEnqueueWriteBuffer(g_ctx.queue, gpu_data, CL_TRUE, 0, data_size, data, 0, nullptr,
                                   nullptr);
        if (err != CL_SUCCESS) {
            ecm_ts_fprintf(stderr, "initial GPU upload failed\n");
            free(s_bits);
            free(data);
            clReleaseMemObject(gpu_s_bits);
            clReleaseMemObject(gpu_data);
            return ECM_ERROR;
        }
    } else {
        ocl_log_verbose(verbose, "Checkpoint: GPU curve data restored (%zu bytes, Montgomery form)\n",
                        data_size);
    }

    opencl_dump_begin(g_dump_ctx, verbose);

    const bool sync_each_batch = g_dump_ctx.enabled || ecm_runtime_config().sync_each_batch;

    using steady_clock = std::chrono::steady_clock;
    const auto checkpoint_epoch = steady_clock::now();
    auto last_checkpoint_wall = checkpoint_epoch;

    // ── Progress bar ──────────────────────────────────────────────────────
    indicators::ProgressBar bar{
        indicators::option::BarWidth{40},
        indicators::option::Start{"["},
        indicators::option::Fill{"="},
        indicators::option::Lead{">"},
        indicators::option::Remainder{" "},
        indicators::option::End{"]"},
        indicators::option::PrefixText{"GPU: "},
        indicators::option::PostfixText{""},
        indicators::option::ShowElapsedTime{true},
        indicators::option::ShowRemainingTime{true},
        indicators::option::ForegroundColor{indicators::Color::cyan},
        indicators::option::FontStyles{
            std::vector<indicators::FontStyle>{indicators::FontStyle::bold}}
    };
    // ──────────────────────────────────────────────────────────────────────
#ifdef _WIN32
    SetConsoleCtrlHandler(console_ctrl_handler, TRUE);
#else
    std::signal(SIGINT, signal_handler);
#endif
    while (s_partial < s_num_bits) {
        uint64_t this_batch = std::min(batch_size, s_num_bits - s_partial);
        if (g_dump_ctx.enabled) {
            size_t words_total = data_size / sizeof(uint32_t);
            std::vector<uint32_t> dump_rows(data, data + words_total);
            ecm_curves_from_montgomery(dump_rows.data(), curves, limbs, N, np0);
            dump_opencl_state_rows(g_dump_ctx, "begin", batches_complete, s_partial, this_batch,
                                   sigma, curves, BITS, tpi, dump_rows.data(), limbs);
        }

        cl_ulong s_num_bits_arg = (cl_ulong)s_num_bits;
        cl_ulong s_start_arg = (cl_ulong)s_partial;
        cl_ulong s_interval_arg = (cl_ulong)this_batch;
        cl_uint count_arg = curves;
        cl_uint sigma_arg = sigma;
        cl_uint np0_arg = np0;
        cl_uint limbs_arg = limbs;

        bool use_sliced     = ecm_runtime_config().gpu_sliced && (g_ecm_kernel_sliced != nullptr);
        bool use_sliced_t16 = ecm_runtime_config().gpu_sliced_t16 && (g_ecm_kernel_sliced_t16 != nullptr);
        bool use_local      = g_ecm_kernel_local != nullptr;
        bool use_explicit_wg = !use_local && !g_kernel_use_coop_wg && (g_kernel_local_wg > 0u);
        uint32_t sliced_wg  = use_sliced_t16 ? 16u : 32u;
        cl_kernel active_kernel = use_sliced_t16 ? g_ecm_kernel_sliced_t16
                                : use_sliced     ? g_ecm_kernel_sliced
                                : use_local      ? g_ecm_kernel_local
                                : g_ecm_kernel;
        err = clSetKernelArg(active_kernel, 0, sizeof(cl_mem), &gpu_s_bits);
        err |= clSetKernelArg(active_kernel, 1, sizeof(cl_ulong), &s_num_bits_arg);
        err |= clSetKernelArg(active_kernel, 2, sizeof(cl_ulong), &s_start_arg);
        err |= clSetKernelArg(active_kernel, 3, sizeof(cl_ulong), &s_interval_arg);
        err |= clSetKernelArg(active_kernel, 4, sizeof(cl_mem), &gpu_data);
        err |= clSetKernelArg(active_kernel, 5, sizeof(cl_uint), &count_arg);
        err |= clSetKernelArg(active_kernel, 6, sizeof(cl_uint), &sigma_arg);
        err |= clSetKernelArg(active_kernel, 7, sizeof(cl_uint), &np0_arg);
        err |= clSetKernelArg(active_kernel, 8, sizeof(cl_uint), &limbs_arg);
        if (err != CL_SUCCESS) {
            ecm_ts_fprintf(stderr, "clSetKernelArg failed\n");
            break;
        }

        bool is_sliced = use_sliced || use_sliced_t16;
        size_t global = use_local
            ? (size_t)curves * (size_t)g_kernel_local_wg
            : (is_sliced
                ? (size_t)curves * (size_t)sliced_wg
                : (use_explicit_wg
                    ? (size_t)curves * (size_t)g_kernel_local_wg
                    : (g_kernel_use_coop_wg ? (size_t)curves * (size_t)g_kernel_coop_wg : (size_t)curves)));
        size_t local_wg = use_local       ? (size_t)g_kernel_local_wg
                        : use_explicit_wg ? (size_t)g_kernel_local_wg
                        : is_sliced       ? (size_t)sliced_wg
                        : (size_t)g_kernel_coop_wg;
        const size_t *local_ptr = (use_local || use_explicit_wg || is_sliced || g_kernel_use_coop_wg) ? &local_wg : nullptr;
        auto t0 = std::chrono::high_resolution_clock::now();
        err = clEnqueueNDRangeKernel(g_ctx.queue, active_kernel, 1, nullptr, &global, local_ptr, 0,
                                     nullptr, nullptr);
        if (err != CL_SUCCESS) {
            ecm_ts_fprintf(stderr, "kernel enqueue failed (%d)\n", err);
            break;
        }
        err = clFinish(g_ctx.queue);
        if (err != CL_SUCCESS) {
            ecm_ts_fprintf(stderr, "kernel wait failed (%d)\n", err);
            break;
        }

        if (sync_each_batch) {
            err = clEnqueueReadBuffer(g_ctx.queue, gpu_data, CL_TRUE, 0, data_size, data, 0,
                                      nullptr, nullptr);
            if (err == CL_SUCCESS) {
                ecm_curves_from_montgomery(data, curves, limbs, N, np0);
                dump_opencl_state_rows(g_dump_ctx, "end", batches_complete + 1,
                                       s_partial + this_batch, this_batch, sigma, curves, BITS,
                                       tpi, data, limbs);
                if (s_partial + this_batch < s_num_bits) {
                    stage1_curves_to_montgomery(data, curves, limbs, N, BITS);
                    err = clEnqueueWriteBuffer(g_ctx.queue, gpu_data, CL_TRUE, 0, data_size, data,
                                               0, nullptr, nullptr);
                }
            }
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        const double batch_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        const float current_gputime =
            (float)std::chrono::duration<double, std::milli>(t1 - t_global_start).count();
        if (gputime) {
            *gputime = current_gputime;
        }

        if (batch_ms < 80.0) {
            batch_size = 11 * batch_size / 10;
        } else if (batch_ms > 120.0) {
            batch_size = std::max<uint64_t>(10, 9 * batch_size / 10);
        }

        s_partial += this_batch;
        batches_complete++;

        if (checkpoint_interval_ms > 0) {
            const auto now_wall = steady_clock::now();
            const auto since_last_ckpt_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                                now_wall - last_checkpoint_wall)
                                                .count();
            if (since_last_ckpt_ms >= (long long)checkpoint_interval_ms) {
                std::vector<uint32_t> ckpt_buf(data_size / sizeof(uint32_t));
                cl_int ckpt_err = clEnqueueReadBuffer(g_ctx.queue, gpu_data, CL_TRUE, 0, data_size,
                                                      ckpt_buf.data(), 0, nullptr, nullptr);
                if (ckpt_err == CL_SUCCESS) {
                    opencl_ecm_checkpoint_header_t header{};
                    header.magic = OPENCL_ECM_CHECKPOINT_MAGIC;
                    header.version = OPENCL_ECM_CHECKPOINT_VERSION;
                    header.s_partial = s_partial;
                    header.s_num_bits = s_num_bits;
                    header.batches_complete = batches_complete;
                    header.curves = curves;
                    header.sigma = sigma;
                    header.BITS = BITS;
                    header.TPI = tpi;
                    header.data_size = (uint64_t)data_size;
                    header.timestamp = (int64_t)time(nullptr);
                    opencl_ecm_checkpoint_save(ckpt_filename, &header, ckpt_buf.data(), data_size);
                    last_checkpoint_wall = now_wall;
                } else {
                    ecm_ts_fprintf(stderr, "Warning: checkpoint GPU readback failed (%d)\n",
                                   ckpt_err);
                }
            }
        }

        // ── Update progress bar (every batch) ────────────────────────────────
        {
            double pct =
                (s_num_bits > 0u) ? (100.0 * (double)s_partial / (double)s_num_bits) : 0.0;
            if (pct > 100.0) pct = 100.0;
            bar.set_progress(static_cast<float>(pct));

            double per_curve_ms = (curves > 0u) ? (batch_ms / (double)curves) : 0.0;
            char postfix[160];
            snprintf(postfix, sizeof(postfix), "%.1f%%  %llu, +%llu bits (~%.1f ms/curve)",
                     pct,
                     (unsigned long long)s_partial, (unsigned long long)this_batch,
                     per_curve_ms);
            bar.set_option(indicators::option::PostfixText{postfix});
        }

        // ── Ctrl+C → break early ─────────────────────────────────────
        if (g_interrupted.load(std::memory_order_relaxed)) break;
    }

    // ── Mark progress bar complete ──────────────────────────────────────
    bar.mark_as_completed();

    // ── Ctrl+C: save checkpoint before normal exit ─────────────────────
    if (g_interrupted.load(std::memory_order_relaxed)) {
        ecm_ts_fprintf(stderr, "\nInterrupted — saving checkpoint before exit...\n");
        fflush(stderr);
        std::vector<uint32_t> ckpt_buf(data_size / sizeof(uint32_t));
        clFinish(g_ctx.queue);
        cl_int ckpt_err = clEnqueueReadBuffer(g_ctx.queue, gpu_data, CL_FALSE, 0, data_size,
                                              ckpt_buf.data(), 0, nullptr, nullptr);
        if (ckpt_err == CL_SUCCESS)
            ckpt_err = clFinish(g_ctx.queue);
        if (ckpt_err == CL_SUCCESS) {
            opencl_ecm_checkpoint_header_t hdr{};
            hdr.magic    = OPENCL_ECM_CHECKPOINT_MAGIC;
            hdr.version  = OPENCL_ECM_CHECKPOINT_VERSION;
            hdr.s_partial = s_partial;
            hdr.s_num_bits = s_num_bits;
            hdr.batches_complete = batches_complete;
            hdr.curves   = curves;
            hdr.sigma    = sigma;
            hdr.BITS     = BITS;
            hdr.TPI      = tpi;
            hdr.data_size = (uint64_t)data_size;
            hdr.timestamp = (int64_t)time(nullptr);
            opencl_ecm_checkpoint_save(ckpt_filename, &hdr, ckpt_buf.data(), data_size);
            ecm_ts_fprintf(stderr, "Checkpoint saved (%llu/%llu bits, %.1f%%)\n",
                           (unsigned long long)s_partial, (unsigned long long)s_num_bits,
                           s_num_bits > 0u ? (100.0*s_partial/s_num_bits) : 0.0);
        } else {
            ecm_ts_fprintf(stderr, "Warning: checkpoint GPU readback failed (%d)\n", ckpt_err);
        }
        free(s_bits);  s_bits  = nullptr;
        free(data);    data    = nullptr;
        clReleaseMemObject(gpu_s_bits);  gpu_s_bits = nullptr;
        clReleaseMemObject(gpu_data);    gpu_data   = nullptr;
        *sigma_ptr = sigma;
        opencl_dump_end(g_dump_ctx);
        return ECM_ERROR;
    }

    if (err == CL_SUCCESS && !sync_each_batch) {
        err = clEnqueueReadBuffer(g_ctx.queue, gpu_data, CL_TRUE, 0, data_size, data, 0,
                                  nullptr, nullptr);
        if (err == CL_SUCCESS) {
            ecm_curves_from_montgomery(data, curves, limbs, N, np0);
        } else {
            ecm_ts_fprintf(stderr, "final GPU readback failed (%d)\n", err);
        }
    }

    auto t_global_end = std::chrono::high_resolution_clock::now();
    if (gputime) {
        *gputime = (float)std::chrono::duration<double, std::milli>(t_global_end - t_global_start)
                       .count();
    }
    const uint64_t kernel_bits_processed = (s_partial > 0u) ? (s_partial - 1u) : 0u;
    if (ecm_runtime_config().profile_ops) {
        const float gputime_local = gputime ? *gputime : 0.0f;
        const ecm_ops_profile_counts_t counts =
            compute_ops_profile_counts(kernel_bits_processed, curves);
        emit_ops_profile(counts, curves, s_num_bits, batches_complete, gputime_local, verbose);
    }

    int youpi = ECM_NO_FACTOR_FOUND;
    if (err == CL_SUCCESS) {
        youpi = process_results(factors, array_found, N, data, limbs, (int)curves, sigma, verbose);
    } else {
        youpi = ECM_ERROR;
    }

cleanup:
    clReleaseMemObject(gpu_s_bits);
    clReleaseMemObject(gpu_data);
    opencl_dump_end(g_dump_ctx);
    free(s_bits);
    free(data);

    if (youpi != ECM_ERROR && opencl_ecm_checkpoint_remove(ckpt_filename) == 0) {
        ocl_log_verbose(verbose, "Checkpoint file removed\n");
    }

    *sigma_ptr = sigma;
    return youpi;
}
