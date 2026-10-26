#include "io/files.h"
#include <jni.h>

#include <android/asset_manager_jni.h>

#include <string>

#include "ecm_android_run.h"
#include "ecm_log_android.h"
#include "kernel_assets.h"
#include "jni_utf8.h"
#include "opencl_program_cache.h"
#include "opencl_runtime.h"
#include "../../../../../../src/ecm/backend/opencl/arithmetic/path_registry.h"

#include "../../../../../../src/ecm/montgomery_stage1/opencl/checkpoint.h"

#include <cstdint>

static std::string jstring_to_utf8(JNIEnv* env, jstring value) {
    if (env == nullptr || value == nullptr) {
        return {};
    }
    const char* chars = env->GetStringUTFChars(value, nullptr);
    if (chars == nullptr) {
        return {};
    }
    std::string out(chars);
    env->ReleaseStringUTFChars(value, chars);
    return out;
}

static std::string prepend_opencl_error(JNIEnv* env, jstring j_err, const std::string& body) {
    std::string report;
    if (j_err != nullptr) {
        const char* err = env->GetStringUTFChars(j_err, nullptr);
        if (err != nullptr && err[0] != '\0') {
            report += "System.loadLibrary(OpenCL) failed:\n";
            report += err;
            report += "\n\n";
            env->ReleaseStringUTFChars(j_err, err);
        }
    }
    report += body;
    return report;
}

namespace {

std::string build_path_list(const char *const *aliases, std::string &result) {
    for (; *aliases != nullptr; ++aliases) {
        result += *aliases;
        result += '\n';
    }
    return result;
}

std::string build_mont_list(const EcmMontPathDescriptor *registry, size_t count) {
    std::string result = "auto\n";
    for (size_t i = 0; i < count; ++i) {
        if (registry[i].os_mask & OS_ANDROID) {
            if (registry[i].id != nullptr) {
                result += registry[i].id;
                result += '\n';
            }
        }
    }
    return result;
}

std::string build_addsub_list(const EcmAddSubPathDescriptor *registry, size_t count) {
    std::string result = "auto\n";
    for (size_t i = 0; i < count; ++i) {
        if (registry[i].os_mask & OS_ANDROID) {
            if (registry[i].aliases != nullptr && registry[i].aliases[0] != nullptr) {
                result += registry[i].aliases[0];
                result += '\n';
            }
        }
    }
    return result;
}

std::string build_special_mult_list() {
    const size_t count = opencl_ecm_special_mult_registry_count();
    std::string result = "auto\n";
    for (size_t i = 0; i < count; ++i) {
        const EcmSpecialMultPathDescriptor *desc = opencl_ecm_special_mult_registry_entry(i);
        if (desc != nullptr && (desc->os_mask & OS_ANDROID)) {
            if (desc->aliases != nullptr && desc->aliases[0] != nullptr) {
                result += desc->aliases[0];
                result += '\n';
            }
        }
    }
    return result;
}

} // namespace

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_ecm_MainActivity_nativeListMulPaths(JNIEnv* env, jobject /* this */) {
    return new_jstring_utf8(env,
        build_mont_list(opencl_ecm_mont_mul_registry_entry(0),
                        opencl_ecm_mont_mul_registry_count()));
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_ecm_MainActivity_nativeListSqrPaths(JNIEnv* env, jobject /* this */) {
    return new_jstring_utf8(env,
        build_mont_list(opencl_ecm_mont_sqr_registry_entry(0),
                        opencl_ecm_mont_sqr_registry_count()));
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_ecm_MainActivity_nativeListAddPaths(JNIEnv* env, jobject /* this */) {
    return new_jstring_utf8(env,
        build_addsub_list(opencl_ecm_addmod_registry_entry(0),
                          opencl_ecm_addmod_registry_count()));
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_ecm_MainActivity_nativeListSubPaths(JNIEnv* env, jobject /* this */) {
    return new_jstring_utf8(env,
        build_addsub_list(opencl_ecm_submod_registry_entry(0),
                          opencl_ecm_submod_registry_count()));
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_ecm_MainActivity_nativeListSpecialMultPaths(JNIEnv* env, jobject /* this */) {
    return new_jstring_utf8(env, build_special_mult_list());
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_ecm_MainActivity_nativeProbe(JNIEnv* env, jobject /* this */, jstring j_opencl_load_error) {
    std::string body = get_opencl_cache_status();
    body += "\n";
    body += probe_opencl();
    return new_jstring_utf8(env, prepend_opencl_error(env, j_opencl_load_error, body));
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_ecm_MainActivity_nativeShortTest(JNIEnv* env, jobject /* this */) {
    return new_jstring_utf8(env, run_short_test());
}

extern "C" void ecm_read_gpu_stats_ex(int* out_busy_percent, long long* out_freq_hz, int* out_busy_src,
                                      int* out_freq_src, long long* out_max_freq_hz);

extern "C" JNIEXPORT void JNICALL
Java_com_example_ecm_MainActivity_nativeInitAssets(
        JNIEnv* env,
        jobject /* this */,
        jobject asset_manager,
        jstring cache_dir) {
    set_kernel_asset_manager(AAssetManager_fromJava(env, asset_manager));
    if (cache_dir != nullptr) {
        const char* path = env->GetStringUTFChars(cache_dir, nullptr);
        if (path != nullptr) {
            set_opencl_cache_dir(path);
            ecm_io::set_work_dir(path);
            env->ReleaseStringUTFChars(cache_dir, path);
        }
    }
}

extern "C" JNIEXPORT jlongArray JNICALL
Java_com_example_ecm_DevicePerfMonitor_nativeGpuStats(JNIEnv* env, jclass /* clazz */) {
    int busy = -1;
    long long freq_hz = -1;
    int busy_src = 0;
    int freq_src = 0;
    long long max_freq_hz = -1;
    ecm_read_gpu_stats_ex(&busy, &freq_hz, &busy_src, &freq_src, &max_freq_hz);
    jlong out[5] = {static_cast<jlong>(busy), static_cast<jlong>(freq_hz),
                    static_cast<jlong>(busy_src), static_cast<jlong>(freq_src),
                    static_cast<jlong>(max_freq_hz)};
    jlongArray arr = env->NewLongArray(5);
    if (arr != nullptr) {
        env->SetLongArrayRegion(arr, 0, 5, out);
    }
    return arr;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_ecm_MainActivity_nativeAddSubBench(
        JNIEnv* env,
        jobject /* this */,
        jint bits,
        jint kernel_iters,
        jint instances,
        jint launch_repeats,
        jint limb_bits) {
    return new_jstring_utf8(
        env, run_addsub_bench(bits, kernel_iters, instances, launch_repeats, limb_bits));
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_ecm_MainActivity_nativeMontSqrBench(
        JNIEnv* env,
        jobject /* this */,
        jint bits,
        jint kernel_iters,
        jint instances,
        jint launch_repeats,
        jboolean use_wg,
        jint tpi,
        jint limb_bits) {
    return new_jstring_utf8(
        env,
        run_montsqr_bench(bits, kernel_iters, instances, launch_repeats, use_wg == JNI_TRUE, tpi,
                          limb_bits));
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_ecm_MainActivity_nativeRunEcm(
        JNIEnv* env,
        jobject /* this */,
        jstring j_n_expr,
        jdouble b1,
        jdouble b2,
        jint gpu_curves,
        jint device_index,
        jboolean verbose,
        jdouble gpu_ckpt_sec,
        jstring j_sigma,
        jstring j_mul_path,
        jstring j_sqr_path,
        jstring j_add_path,
        jstring j_sub_path,
        jstring j_special_mult_path,
        jstring j_save_file,
        jboolean j_save_append,
        jobject j_log_callback) {
    EcmAndroidRunRequest req;
    req.n_expr = jstring_to_utf8(env, j_n_expr);
    req.b1 = b1;
    req.b2 = b2;
    req.gpu_curves = static_cast<uint32_t>(gpu_curves);
    req.device_index = device_index;
    req.verbose = verbose == JNI_TRUE;
    req.gpu_ckpt_sec = gpu_ckpt_sec;

    const std::string sigma = jstring_to_utf8(env, j_sigma);
    if (!sigma.empty()) {
        try {
            req.sigma_fixed = true;
            // Desktop sigma format: "3:N" (param:value). Extract only the value.
            auto colon = sigma.find(':');
            std::string value_str = (colon != std::string::npos) ? sigma.substr(colon + 1) : sigma;
            req.sigma = static_cast<uint32_t>(std::stoul(value_str));
        } catch (...) {
            return new_jstring_utf8(env, "Invalid sigma value");
        }
    }

    req.mul_path = jstring_to_utf8(env, j_mul_path);
    req.sqr_path = jstring_to_utf8(env, j_sqr_path);
    req.add_path = jstring_to_utf8(env, j_add_path);
    req.sub_path = jstring_to_utf8(env, j_sub_path);
    req.special_mult_path = jstring_to_utf8(env, j_special_mult_path);
    req.save_file = jstring_to_utf8(env, j_save_file);
    req.save_append = j_save_append == JNI_TRUE;
    android_ecm_log_set_listener(env, j_log_callback);
    const std::string result = run_ecm_android(req);
    android_ecm_log_clear_listener(env);
    return new_jstring_utf8(env, result);
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_example_ecm_MainActivity_nativeBitBench(
        JNIEnv* env,
        jobject /* this */,
        jint limb_bits,
        jint elements,
        jint kernel_iters,
        jint launch_repeats) {
    return new_jstring_utf8(
        env, run_bit_bench(limb_bits, elements, kernel_iters, launch_repeats));
}
