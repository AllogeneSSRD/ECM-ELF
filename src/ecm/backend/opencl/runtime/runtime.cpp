#include "cgbn_opencl.h"
#include "../../../../../include/ecm/application/configuration/runtime.h"
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cstdint>
#include <cerrno>
#include <sys/stat.h>

#ifdef _WIN32
#include <direct.h>
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

namespace cgbn {
namespace opencl {

static std::string query_device_string(cl_device_id dev, cl_device_info key) {
    size_t sz = 0;
    if (clGetDeviceInfo(dev, key, 0, nullptr, &sz) != CL_SUCCESS || sz == 0) {
        return std::string();
    }
    std::string out(sz, '\0');
    if (clGetDeviceInfo(dev, key, sz, &out[0], nullptr) != CL_SUCCESS) {
        return std::string();
    }
    while (!out.empty() && out.back() == '\0') {
        out.pop_back();
    }
    return out;
}

static uint64_t fnv1a64_update(uint64_t h, const void *data, size_t n) {
    const unsigned char *p = static_cast<const unsigned char *>(data);
    for (size_t i = 0; i < n; ++i) {
        h ^= (uint64_t)p[i];
        h *= 1099511628211ull;
    }
    return h;
}

static uint64_t fnv1a64_string(uint64_t h, const std::string &s) {
    h = fnv1a64_update(h, s.data(), s.size());
    const char sep = '\n';
    h = fnv1a64_update(h, &sep, 1);
    return h;
}

static std::string cache_root_dir() {
    const std::string &d = ecm_runtime_config().cache_dir;
    return d.empty() ? std::string(".opencl_cache") : d;
}

static bool cache_enabled() {
    return !ecm_runtime_config().cache_disable;
}

static bool cache_verbose() {
    return ecm_runtime_config().cache_verbose;
}

static bool compile_verbose() {
    return ecm_runtime_config().compile_verbose;
}

static bool ensure_cache_dir_exists(const std::string &dir) {
#ifdef _WIN32
    if (_mkdir(dir.c_str()) == 0) return true;
    if (errno == EEXIST) return true;
#else
    if (mkdir(dir.c_str(), 0755) == 0) return true;
    if (errno == EEXIST) return true;
#endif
    return false;
}

static std::string make_cache_path(context_t &ctx, const char *source, const char *options) {
    const std::string dev_name = query_device_string(ctx.device, CL_DEVICE_NAME);
    const std::string dev_vendor = query_device_string(ctx.device, CL_DEVICE_VENDOR);
    const std::string dev_ver = query_device_string(ctx.device, CL_DEVICE_VERSION);
    const std::string drv_ver = query_device_string(ctx.device, CL_DRIVER_VERSION);
    const std::string opts = options ? options : "";
    const std::string src = source ? source : "";

    uint64_t h = 1469598103934665603ull;
    h = fnv1a64_string(h, dev_name);
    h = fnv1a64_string(h, dev_vendor);
    h = fnv1a64_string(h, dev_ver);
    h = fnv1a64_string(h, drv_ver);
    h = fnv1a64_string(h, opts);
    h = fnv1a64_string(h, src);

    std::ostringstream oss;
    oss << std::hex << std::setfill('0') << std::setw(16) << h;
    std::string dir = cache_root_dir();
    if (!dir.empty() && dir.back() != '/' && dir.back() != '\\') {
        dir += "/";
    }
    return dir + "opencl_" + oss.str() + ".bin";
}

static bool load_program_binary(const std::string &path, std::vector<unsigned char> &bytes) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) return false;
    in.seekg(0, std::ios::end);
    std::streampos end = in.tellg();
    if (end <= 0) return false;
    bytes.resize((size_t)end);
    in.seekg(0, std::ios::beg);
    in.read(reinterpret_cast<char *>(bytes.data()), bytes.size());
    return in.good();
}

static bool save_program_binary(context_t &ctx, cl_program program, const std::string &path) {
    size_t sz = 0;
    if (clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &sz, nullptr) != CL_SUCCESS ||
        sz == 0) {
        return false;
    }
    std::vector<unsigned char> bytes(sz);
    unsigned char *ptr = bytes.data();
    if (clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(unsigned char *), &ptr, nullptr) != CL_SUCCESS) {
        return false;
    }
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) return false;
    out.write(reinterpret_cast<const char *>(bytes.data()), bytes.size());
    return out.good();
}

static cl_int create_context_impl(context_t &out, int device_index) {
    cl_int err;
    cl_uint numPlatforms = 0;
    err = clGetPlatformIDs(0, NULL, &numPlatforms);
    if (err != CL_SUCCESS || numPlatforms == 0) return err ? err : -1;

    std::vector<cl_platform_id> platforms(numPlatforms);
    err = clGetPlatformIDs(numPlatforms, platforms.data(), NULL);
    if (err != CL_SUCCESS) return err;

    std::vector<std::pair<cl_platform_id, cl_device_id>> all_devices;
    for (cl_uint p = 0; p < numPlatforms; ++p) {
        cl_uint numDevices = 0;
        err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 0, NULL, &numDevices);
        if (err == CL_DEVICE_NOT_FOUND || numDevices == 0) continue;
        if (err != CL_SUCCESS) return err;
        std::vector<cl_device_id> devices(numDevices);
        err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, numDevices, devices.data(), NULL);
        if (err != CL_SUCCESS) return err;
        for (auto d : devices) {
            all_devices.push_back({platforms[p], d});
        }
    }
    if (all_devices.empty()) return -2;
    if (device_index < 0 || device_index >= (int)all_devices.size()) return -3;

    out.platform = all_devices[(size_t)device_index].first;
    out.device = all_devices[(size_t)device_index].second;
    out.ctx = clCreateContext(NULL, 1, &out.device, NULL, NULL, &err);
    if (err != CL_SUCCESS) return err;

    // create a command queue (non-deprecated compatible API)
#if CL_TARGET_OPENCL_VERSION >= 200
    cl_queue_properties props[] = { 0 };
    out.queue = clCreateCommandQueueWithProperties(out.ctx, out.device, props, &err);
#else
    out.queue = clCreateCommandQueue(out.ctx, out.device, 0, &err);
#endif
    if (err != CL_SUCCESS) {
        clReleaseContext(out.ctx);
        out.ctx = nullptr;
        return err;
    }

    return CL_SUCCESS;
}

cl_int create_context_with_device_index(context_t &out, int device_index) {
    return create_context_impl(out, device_index);
}

cl_int create_context(context_t &out) {
    return create_context_impl(out, ecm_runtime_config().device_index);
}

cl_int destroy_context(context_t &c) {
    cl_int err = CL_SUCCESS;
    if (c.queue) { err = clReleaseCommandQueue(c.queue); c.queue = nullptr; }
    if (c.ctx) { err = clReleaseContext(c.ctx); c.ctx = nullptr; }
    c.device = nullptr;
    c.platform = nullptr;
    return err;
}

cl_program build_program_from_source(context_t &ctx, const char *source, const char *options, cl_int &errcode) {
    if (source == nullptr) {
        errcode = CL_INVALID_VALUE;
        return nullptr;
    }
    const std::string cache_path = make_cache_path(ctx, source, options);
    const std::string cache_dir = cache_root_dir();
    const bool use_cache = cache_enabled();
    const bool verbose = cache_verbose();
    const bool compile_log = compile_verbose();

    if (use_cache) {
        const auto cache_t0 = std::chrono::steady_clock::now();
        ensure_cache_dir_exists(cache_dir);
        std::vector<unsigned char> bin;
        if (load_program_binary(cache_path, bin) && !bin.empty()) {
            size_t bin_size = bin.size();
            const unsigned char *bin_ptr = bin.data();
            cl_int err = CL_SUCCESS;
            cl_int bin_status = CL_SUCCESS;
            cl_program program = clCreateProgramWithBinary(
                ctx.ctx, 1, &ctx.device, &bin_size, &bin_ptr, &bin_status, &err);
            if (program != nullptr && err == CL_SUCCESS && bin_status == CL_SUCCESS) {
                // Some drivers still require clBuildProgram for binaries.
                err = clBuildProgram(program, 1, &ctx.device, options, NULL, NULL);
                if (err == CL_SUCCESS) {
                    if (verbose) {
                        std::cerr << "OpenCL cache hit: " << cache_path << std::endl;
                    }
                    if (compile_log) {
                        const double cache_ms = std::chrono::duration<double, std::milli>(
                                                    std::chrono::steady_clock::now() - cache_t0)
                                                    .count();
                        std::cerr << "OpenCL cache load+build: " << cache_ms << " ms" << std::endl;
                    }
                    errcode = CL_SUCCESS;
                    return program;
                }
            }
            if (verbose) {
                std::cerr << "OpenCL cache stale, rebuilding: " << cache_path << std::endl;
            }
            if (program != nullptr) {
                clReleaseProgram(program);
            }
        }
    }

    size_t src_len = strlen(source);
    const char *src = source;
    cl_int err = CL_SUCCESS;
    cl_program program = clCreateProgramWithSource(ctx.ctx, 1, &src, &src_len, &err);
    if (err != CL_SUCCESS) {
        errcode = err;
        return nullptr;
    }

    const auto build_t0 = std::chrono::steady_clock::now();
    err = clBuildProgram(program, 1, &ctx.device, options, NULL, NULL);
    if (compile_log) {
        const double build_ms = std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - build_t0)
                                    .count();
        std::cerr << "OpenCL clBuildProgram: " << build_ms << " ms"
                  << " (src=" << src_len / 1024u << " KiB)" << std::endl;
    }
    if (err != CL_SUCCESS) {
        size_t log_size = 0;
        clGetProgramBuildInfo(program, ctx.device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
        std::string log(log_size, '\0');
        clGetProgramBuildInfo(program, ctx.device, CL_PROGRAM_BUILD_LOG, log_size, &log[0], NULL);
        std::cerr << "OpenCL build error:\n" << log << std::endl;
        errcode = err;
        return program;
    }

    if (use_cache) {
        if (!save_program_binary(ctx, program, cache_path) && verbose) {
            std::cerr << "OpenCL cache save failed: " << cache_path << std::endl;
        } else if (verbose) {
            std::cerr << "OpenCL cache save: " << cache_path << std::endl;
        }
    }
    errcode = CL_SUCCESS;
    return program;
}

std::string load_text_file(const char *path) {
    std::ifstream ifs(path, std::ios::in | std::ios::binary);
    if (!ifs) {
        return std::string();
    }
    ifs.seekg(0, std::ios::end);
    std::streampos end = ifs.tellg();
    if (end <= 0) {
        return std::string();
    }
    std::string content((size_t)end, '\0');
    ifs.seekg(0, std::ios::beg);
    ifs.read(&content[0], content.size());
    return content;
}

#if defined(__ANDROID__)
std::string android_load_kernel_asset(const char *rel_path);
#endif

static std::string kernel_root_with_slash(const char *root) {
    std::string path(root);
    if (!path.empty() && path.back() != '/' && path.back() != '\\') {
        path += '/';
    }
    return path;
}

static std::string ecm_stage1_kernel_root() {
    const std::string &root = ecm_runtime_config().kernel_root;
    if (!root.empty()) {
        return kernel_root_with_slash(root.c_str());
    }
#ifdef ECM_KERNEL_ROOT_DEFAULT
    return kernel_root_with_slash(ECM_KERNEL_ROOT_DEFAULT);
#endif
    return "src/ecm/backend/opencl/resources/";
}

static std::string load_kernel_at_roots(const char *rel_path) {
    if (rel_path == nullptr || *rel_path == '\0') {
        return std::string();
    }
    const char *prefixes[] = {"", "../", "../../", "../../../", "../../../../"};
    for (const char *pfx : prefixes) {
        std::string path = std::string(pfx) + rel_path;
        std::string content = load_text_file(path.c_str());
        if (!content.empty()) {
            return content;
        }
    }
    if (!ecm_runtime_config().kernel_root.empty()) {
        std::string content = load_text_file(
            (kernel_root_with_slash(ecm_runtime_config().kernel_root.c_str()) + rel_path).c_str());
        if (!content.empty()) {
            return content;
        }
    }
#ifdef ECM_KERNEL_ROOT_DEFAULT
    {
        std::string content =
            load_text_file((kernel_root_with_slash(ECM_KERNEL_ROOT_DEFAULT) + rel_path).c_str());
        if (!content.empty()) {
            return content;
        }
    }
#endif
    return std::string();
}

std::string load_kernel_file(const char *rel_path) {
    if (rel_path == nullptr || *rel_path == '\0') {
        return std::string();
    }
#if defined(__ANDROID__)
    {
        std::string asset = android_load_kernel_asset(rel_path);
        if (!asset.empty()) {
            return asset;
        }
    }
#endif
    std::string content = load_kernel_at_roots(rel_path);
    // Preserve the existing diagnostic logical name, including custom roots
    // with the old layout, while the production source lives in diagnostics/.
    if (content.empty() && std::strcmp(rel_path, "bench/mont.cl") == 0) {
        return load_kernel_file("diagnostics/mont.cl");
    }
    return content;
}

std::string load_ecm_stage1_kernel_file(const char *rel_path) {
    if (rel_path == nullptr || *rel_path == '\0') {
        return std::string();
    }
#if defined(__ANDROID__)
    {
        std::string asset = android_load_kernel_asset(rel_path);
        if (!asset.empty()) {
            return asset;
        }
        asset = android_load_kernel_asset((std::string("kernels/opencl/") + rel_path).c_str());
        if (!asset.empty()) {
            return asset;
        }
    }
#endif
    const std::string ecm_root = ecm_stage1_kernel_root();
    const char *prefixes[] = {"", "../", "../../", "../../../", "../../../../"};
    for (const char *pfx : prefixes) {
        std::string path = std::string(pfx) + ecm_root + rel_path;
        std::string content = load_text_file(path.c_str());
        if (!content.empty()) {
            return content;
        }
        path = std::string(pfx) + "src/ecm/backend/opencl/resources/" + rel_path;
        content = load_text_file(path.c_str());
        if (!content.empty()) {
            return content;
        }
    }
    return load_kernel_at_roots((ecm_root + rel_path).c_str());
}

} // namespace opencl
} // namespace cgbn
