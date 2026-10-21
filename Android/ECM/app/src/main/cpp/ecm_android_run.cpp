#include "ecm_android_run.h"

#include "ecm_log_android.h"
#include "opencl_android_shim.h"
#include "opencl_loader.h"
#include "ecm/platform/logging.h"
#include "ecm/application/configuration/runtime.h"
#include "opencl_program_cache.h"

#include <sstream>
#include <string>

#ifndef ECM_HAVE_GMP

std::string run_ecm_android(const EcmAndroidRunRequest& req) {
    std::ostringstream out;
    out << "=== ECM GPU Stage-1 (Android) ===\n";
    out << "N expr: " << req.n_expr << "\n";
    out << "B1=" << req.b1 << " B2=" << req.b2 << " gpucurves=" << req.gpu_curves
        << " device=" << req.device_index << "\n\n";
    out << "ECM factorization is not linked in this APK build.\n";
    out << "See Android/ECM/docs/DEV_GMP_SETUP.md\n";
    out << "  vcpkg install gmp:arm64-android\n";
    out << "  local.properties: ecm.android.gmp.root=<vcpkg>/installed/arm64-android\n\n";
    out << "Desktop equivalent:\n";
    out << "  echo '" << req.n_expr << "' | ecm.exe";
    if (req.verbose) {
        out << " -v";
    }
    out << " -d " << req.device_index << " --curve mont --target gpu -gpucurves " << req.gpu_curves << " "
        << static_cast<long long>(req.b1) << " " << static_cast<long long>(req.b2) << "\n";
    return out.str();
}

#else

#include "../../../../../../src/ecm/application/configuration/ecm_expr.h"
#include "../../../../../../src/ecm/storage/checkpoint/ecm_checkpoint.h"
#include "../../../../../../src/ecm/backend/opencl/stage1/montgomery/entry.h"
#include "../../../../../../src/ecm/storage/save/ecm_save.h"

#include "../../../../../../src/ecm/service/execution/ecm.h"
#include "../../../../../../src/ecm/service/execution/cgbn_stage1.h"

#include <cstdlib>
#include <cstring>
#include <sstream>
#include <vector>

extern "C" uint32_t gpu_pick_random_sigma(uint32_t curves);

namespace {

bool configure_device_index(int device_index, std::string& log) {
    if (device_index < 0) {
        log += "Invalid device index\n";
        return false;
    }
    // Android 无命令行：直接写入运行时配置（替代旧的 setenv 环境变量）。
    ecm_runtime_config().device_index = device_index;
    ecm_runtime_config().cache_dir = get_opencl_cache_dir();
    return true;
}

} // namespace

std::string run_ecm_android(const EcmAndroidRunRequest& req) {
    std::ostringstream header;
    header << "=== ECM GPU Stage-1 (Android) ===\n";
    header << "equiv: echo '" << req.n_expr << "' | ecm.exe";
    if (req.verbose) {
        header << " -v";
    }
    header << " -d " << req.device_index << " --curve mont --target gpu -gpucurves " << req.gpu_curves << " "
           << static_cast<long long>(req.b1) << " " << static_cast<long long>(req.b2);
    if (!req.save_file.empty()) {
        header << (req.save_append ? " -savea " : " -save ") << req.save_file;
    }
    header << "\n\n";

    std::string body;
    android_ecm_log_begin(&body);
    if (android_ecm_log_listener_active()) {
        ecm_ts_fprintf(stdout, "%s", header.str().c_str());
    }

    OpenCLApi api{};
    bool own_lib = false;
    std::ostringstream load_log;
    if (!load_opencl_api(api, own_lib, load_log)) {
        android_ecm_log_end();
        return header.str() + load_log.str() + "\nFAIL: cannot load OpenCL\n";
    }
    android_ecm_bind_opencl_api(&api);

    std::string err;
    mpz_t N;
    mpz_init(N);
    if (!ecm_parse_expression(req.n_expr, N, &err)) {
        android_ecm_unbind_opencl_api();
        unload_opencl_api(api, own_lib);
        android_ecm_log_end();
        return header.str() + "Failed to parse N: " + err + "\n";
    }

    ecm_params params;
    ecm_init(params);
    params->gpu = 1;
    params->gpu_number_of_curves = req.gpu_curves;
    params->verbose = req.verbose ? 1 : 0;
    params->param = ECM_PARAM_BATCH_32BITS_D;
    if (req.gpu_ckpt_sec <= 0.0) {
        params->gpu_checkpoint_interval_ms = 0;
    } else {
        params->gpu_checkpoint_interval_ms =
            static_cast<unsigned long>(req.gpu_ckpt_sec * 1000.0 + 0.5);
    }
    if (!req.mul_path.empty()) {
        std::strncpy(params->gpu_mul_path, req.mul_path.c_str(), sizeof(params->gpu_mul_path) - 1);
    }
    if (!req.sqr_path.empty()) {
        std::strncpy(params->gpu_sqr_path, req.sqr_path.c_str(), sizeof(params->gpu_sqr_path) - 1);
    }
    if (!req.add_path.empty()) {
        std::strncpy(params->gpu_add_path, req.add_path.c_str(), sizeof(params->gpu_add_path) - 1);
    }
    if (!req.sub_path.empty()) {
        std::strncpy(params->gpu_sub_path, req.sub_path.c_str(), sizeof(params->gpu_sub_path) - 1);
    }

    mpz_t batch_s;
    mpz_init(batch_s);
    if (!ecm_compute_batch_s(batch_s, req.b1)) {
        mpz_clear(N);
        mpz_clear(batch_s);
        ecm_clear(params);
        android_ecm_unbind_opencl_api();
        unload_opencl_api(api, own_lib);
        android_ecm_log_end();
        return header.str() + "Failed to compute batch_s\n";
    }
    mpz_set(params->batch_s, batch_s);
    params->batch_last_B1_used = req.b1;

    const uint32_t curves = req.gpu_curves;
    if (curves == 0) {
        mpz_clear(N);
        mpz_clear(batch_s);
        ecm_clear(params);
        android_ecm_unbind_opencl_api();
        unload_opencl_api(api, own_lib);
        android_ecm_log_end();
        return header.str() + "gpucurves must be > 0\n";
    }

    std::string dev_err;
    if (!configure_device_index(req.device_index, dev_err)) {
        mpz_clear(N);
        mpz_clear(batch_s);
        ecm_clear(params);
        android_ecm_unbind_opencl_api();
        unload_opencl_api(api, own_lib);
        android_ecm_log_end();
        return header.str() + dev_err;
    }

    ecm_ts_fprintf(stdout, "Parsed N bit-size: %zu\n", mpz_sizeinbase(N, 2));
    ecm_ts_fprintf(stdout, "batch_s bit-size: %zu\n", mpz_sizeinbase(batch_s, 2));
    if (const char *work_dir = opencl_ecm_get_work_dir(); work_dir != nullptr && work_dir[0] != '\0') {
        ecm_ts_fprintf(stdout, "Data work dir: %s\n", work_dir);
    }
    ecm_ts_fprintf(stdout, "Checkpoint file: %s\n", opencl_ecm_checkpoint_filename(N));

    std::string resolved_save;
    if (!req.save_file.empty()) {
        resolved_save = opencl_ecm_resolve_data_path(req.save_file.c_str());
        if (!opencl_ecm_check_save_file_writable(resolved_save, req.save_append)) {
            mpz_clear(N);
            mpz_clear(batch_s);
            ecm_clear(params);
            android_ecm_unbind_opencl_api();
            unload_opencl_api(api, own_lib);
            android_ecm_log_end();
            return header.str() + "Save file is not writable: " + resolved_save + "\n";
        }
        ecm_ts_fprintf(stdout, "Save file: %s (%s)\n", resolved_save.c_str(),
                       req.save_append ? "append" : "create");
    }

    const int prep = gpu_prepare_opencl(
        static_cast<size_t>(mpz_sizeinbase(N, 2)), params->verbose,
        params->gpu_mul_path[0] ? params->gpu_mul_path : nullptr,
        params->gpu_sqr_path[0] ? params->gpu_sqr_path : nullptr,
        params->gpu_add_path[0] ? params->gpu_add_path : nullptr,
        params->gpu_sub_path[0] ? params->gpu_sub_path : nullptr,
        req.special_mult_path.empty() ? nullptr : req.special_mult_path.c_str());
    if (prep != 0) {
        mpz_clear(N);
        mpz_clear(batch_s);
        ecm_clear(params);
        android_ecm_unbind_opencl_api();
        unload_opencl_api(api, own_lib);
        android_ecm_log_end();
        return header.str() + body + "\nGPU: OpenCL prepare failed\n";
    }

    mpz_t* factors = static_cast<mpz_t*>(std::malloc(sizeof(mpz_t) * curves));
    int* array_found = static_cast<int*>(std::malloc(sizeof(int) * curves));
    if (factors == nullptr || array_found == nullptr) {
        mpz_clear(N);
        mpz_clear(batch_s);
        ecm_clear(params);
        android_ecm_unbind_opencl_api();
        unload_opencl_api(api, own_lib);
        android_ecm_log_end();
        return header.str() + "Out of memory\n";
    }
    for (uint32_t i = 0; i < curves; ++i) {
        mpz_init(factors[i]);
        array_found[i] = ECM_NO_FACTOR_FOUND;
    }

    uint32_t firstsigma = req.sigma_fixed ? req.sigma : gpu_pick_random_sigma(curves);
    if (static_cast<uint64_t>(firstsigma) + curves > 0x100000000ull) {
        for (uint32_t i = 0; i < curves; ++i) {
            mpz_clear(factors[i]);
        }
        std::free(factors);
        std::free(array_found);
        mpz_clear(N);
        mpz_clear(batch_s);
        ecm_clear(params);
        android_ecm_unbind_opencl_api();
        unload_opencl_api(api, own_lib);
        android_ecm_log_end();
        return header.str() + "sigma range overflows uint32\n";
    }

    ecm_ts_fprintf(stdout, "Using B1=%.0f, B2=%.0f, sigma=%d:%u-%u (%u curves)\n", req.b1, req.b2,
                   ECM_PARAM_BATCH_32BITS_D, firstsigma, firstsigma + curves - 1, curves);

    float gputime = 0.0f;
    const int ret = opencl_ecm_stage1(
        factors, array_found, N, params->batch_s, curves, &firstsigma,
        params->gpu_checkpoint_interval_ms, &gputime, params->verbose,
        params->gpu_mul_path[0] ? params->gpu_mul_path : nullptr,
        params->gpu_sqr_path[0] ? params->gpu_sqr_path : nullptr,
        params->gpu_add_path[0] ? params->gpu_add_path : nullptr,
        params->gpu_sub_path[0] ? params->gpu_sub_path : nullptr,
        params->gpu_special_mult_path[0] ? params->gpu_special_mult_path : nullptr);

    ecm_ts_fprintf(stdout, "opencl_ecm_stage1 returned: %d gputime=%.3f ms\n", ret, gputime);

    int found_count = 0;
    for (uint32_t i = 0; i < curves; ++i) {
        if (array_found[i] != ECM_NO_FACTOR_FOUND) {
            ++found_count;
            char* dec = mpz_get_str(nullptr, 10, factors[i]);
            ecm_ts_fprintf(stdout, "factor[%u]=%s\n", i, dec != nullptr ? dec : "?");
            if (dec != nullptr) {
                std::free(dec);
            }
        }
    }
    if (found_count == 0) {
        ecm_ts_fprintf(stdout, "No factor found in this batch.\n");
    }

    if (ret != ECM_ERROR && !resolved_save.empty()) {
        const std::string n_expr_save =
            opencl_ecm_build_saved_n_expr(req.n_expr, N, curves, factors, array_found);
        if (!opencl_ecm_append_save_lines(resolved_save, N, req.b1, firstsigma, curves, factors,
                                          n_expr_save)) {
            ecm_ts_fprintf(stderr, "Failed to append save lines into %s\n", resolved_save.c_str());
        }
    }

    for (uint32_t i = 0; i < curves; ++i) {
        mpz_clear(factors[i]);
    }
    std::free(factors);
    std::free(array_found);
    mpz_clear(N);
    mpz_clear(batch_s);
    ecm_clear(params);

    android_ecm_unbind_opencl_api();
    unload_opencl_api(api, own_lib);
    android_ecm_log_end();

    std::ostringstream out;
    if (android_ecm_log_listener_active()) {
        out << "\nRESULT: " << (ret == ECM_ERROR ? "ERROR" : "OK") << "\n";
    } else {
        out << header.str() << body;
        out << "\nRESULT: " << (ret == ECM_ERROR ? "ERROR" : "OK") << "\n";
    }
    return out.str();
}

#endif
