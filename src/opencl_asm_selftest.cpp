#include "ecm/backend/opencl/runtime/device.h"
#include "ecm/backend/opencl/runtime/cgbn_opencl.h"

#include <CL/cl.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

struct InstrCase {
    int what;
    const char *name;
    int latency_ops_per_iter;
    int throughput_ops_per_iter;
};

const InstrCase kCases[] = {
    {0, "V_NOP", 1, 4},
    {1, "V_ADD_I32", 1, 4},
    {2, "V_FMA_F32", 1, 4},
    {3, "V_ADD_F64", 1, 4},
    {4, "V_FMA_F64", 1, 4},
    {5, "V_MUL_F64", 1, 4},
    {6, "V_MAD_U64_U32", 1, 4},
};

bool has_amd_gpu(const cgbn::opencl::context_t &ctx) {
    char pname[1024] = {0};
    char dname[1024] = {0};
    clGetPlatformInfo(ctx.platform, CL_PLATFORM_NAME, sizeof(pname), pname, nullptr);
    clGetDeviceInfo(ctx.device, CL_DEVICE_NAME, sizeof(dname), dname, nullptr);
    std::string p = pname;
    std::string d = dname;
    for (char &c : p) c = (char)tolower((unsigned char)c);
    for (char &c : d) c = (char)tolower((unsigned char)c);
    return p.find("amd") != std::string::npos || d.find("amd") != std::string::npos ||
           d.find("gfx") != std::string::npos || d.find("radeon") != std::string::npos;
}

bool has_valid_sink(const std::vector<int64_t> &sink, size_t active_elems) {
    active_elems = std::min(active_elems, sink.size());
    for (size_t i = 0; i < active_elems; ++i) {
        if (sink[i] >= 0) return true;
    }
    return false;
}

bool has_strict_invalid_sink(const std::vector<int64_t> &sink, size_t active_elems) {
    active_elems = std::min(active_elems, sink.size());
    if (active_elems == 0) return true;
    for (size_t i = 0; i < active_elems; ++i) {
        if (sink[i] >= 0) return false;
    }
    return true;
}

double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    size_t n = v.size();
    if (n & 1u) return v[n / 2];
    return 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

} // namespace

static double run_kernel_host_timed_ns(cl_command_queue q, cl_kernel k,
                                       size_t global, size_t local) {
    auto t0 = std::chrono::high_resolution_clock::now();
    cl_int err = clEnqueueNDRangeKernel(q, k, 1, nullptr, &global, &local, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) return -1.0;
    err = clFinish(q);
    if (err != CL_SUCCESS) return -1.0;
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::nano>(t1 - t0).count();
}

int main(int argc, char **argv) {
    bool verbose_groups = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "-v") {
            verbose_groups = true;
            continue;
        }
        if (a == "-h" || a == "--help") {
            std::cout << "Usage: opencl_asm_selftest [-v]\n"
                         "  -v    Print throughput for every tested groups value\n";
            return EXIT_SUCCESS;
        }
        std::cerr << "Unknown argument: " << a << std::endl;
        std::cerr << "Use -h for help." << std::endl;
        return EXIT_FAILURE;
    }
    if (!configureFirstAmdGpuDevice(true)) {
        return EXIT_FAILURE;
    }

    cgbn::opencl::context_t ctx;
    cl_int err = cgbn::opencl::create_context(ctx);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create OpenCL context: " << err << std::endl;
        return EXIT_FAILURE;
    }
    if (!has_amd_gpu(ctx)) {
        std::cerr << "Selected device is not AMD GPU; aborting selftest." << std::endl;
        cgbn::opencl::destroy_context(ctx);
        return EXIT_FAILURE;
    }
    cl_int qerr = CL_SUCCESS;
#if CL_TARGET_OPENCL_VERSION >= 200
    cl_queue_properties props[] = {0};
    cl_command_queue prof_queue = clCreateCommandQueueWithProperties(ctx.ctx, ctx.device, props, &qerr);
#else
    cl_command_queue prof_queue = clCreateCommandQueue(ctx.ctx, ctx.device, 0, &qerr);
#endif
    if (qerr != CL_SUCCESS || prof_queue == nullptr) {
        std::cerr << "Failed to create profiling queue: " << qerr << std::endl;
        cgbn::opencl::destroy_context(ctx);
        return EXIT_FAILURE;
    }

    std::string src = cgbn::opencl::load_kernel_file("bench/selftest.cl");
    if (src.empty()) {
        std::cerr << "Failed to load selftest.cl" << std::endl;
        cgbn::opencl::destroy_context(ctx);
        return EXIT_FAILURE;
    }
    cl_int buildErr = CL_SUCCESS;
    const char *opts = "-DSELFTEST_DISABLE_MAD_U64=1 -DSELFTEST_ENABLE_MAD_U64=0";
    cl_program prog = cgbn::opencl::build_program_from_source(ctx, src.c_str(), opts, buildErr);
    if (prog == nullptr || buildErr != CL_SUCCESS) {
        std::cerr << "Failed to build selftest.cl: " << buildErr << std::endl;
        cgbn::opencl::destroy_context(ctx);
        return EXIT_FAILURE;
    }

    cl_kernel k_lat = clCreateKernel(prog, "testLatency", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create kernel testLatency: " << err << std::endl;
        clReleaseProgram(prog);
        cgbn::opencl::destroy_context(ctx);
        return EXIT_FAILURE;
    }
    cl_kernel k_thr = clCreateKernel(prog, "testThroughput", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create kernel testThroughput: " << err << std::endl;
        clReleaseKernel(k_lat);
        clReleaseProgram(prog);
        cgbn::opencl::destroy_context(ctx);
        return EXIT_FAILURE;
    }

    cl_kernel k_base_lat = clCreateKernel(prog, "testBaselineLatency", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create kernel testBaselineLatency: " << err << std::endl;
        clReleaseKernel(k_lat);
        clReleaseKernel(k_thr);
        clReleaseProgram(prog);
        clReleaseCommandQueue(prof_queue);
        cgbn::opencl::destroy_context(ctx);
        return EXIT_FAILURE;
    }
    cl_kernel k_base_thr = clCreateKernel(prog, "testBaselineThroughput", &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create kernel testBaselineThroughput: " << err << std::endl;
        clReleaseKernel(k_base_lat);
        clReleaseKernel(k_lat);
        clReleaseKernel(k_thr);
        clReleaseProgram(prog);
        clReleaseCommandQueue(prof_queue);
        cgbn::opencl::destroy_context(ctx);
        return EXIT_FAILURE;
    }

    const size_t groups = 2048;
    const size_t local = 64;
    const int it_latency = 1 << 15;
    const int it_throughput = 1 << 18;
    const int repeats = 4;
    const std::vector<size_t> group_candidates = {32, 64, 96, 128, 256, 384, 512, 768, 1024, 1536, 2048, 4096, 8192};
    std::vector<int64_t> host(groups, 0);
    cl_mem buf = clCreateBuffer(ctx.ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                sizeof(int64_t) * host.size(), host.data(), &err);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create timing buffer: " << err << std::endl;
        clReleaseKernel(k_lat);
        clReleaseKernel(k_thr);
        clReleaseProgram(prog);
        cgbn::opencl::destroy_context(ctx);
        return EXIT_FAILURE;
    }

    std::cout << "Running AMD ISA selftest on first AMD GPU..." << std::endl;
    std::cout << "Build options: " << opts << std::endl;
    std::cout << "Local=" << local << " it_latency=" << it_latency
              << " it_throughput=" << it_throughput
              << " repeats=" << repeats << std::endl;
    for (const auto &tc : kCases) {
        int what = tc.what;
        double best_gops = -1.0;
        size_t best_groups = 0;
        double best_lat_ns = 0.0;

        for (size_t g : group_candidates) {
            size_t global = g * local;
            if (g > groups) break;

            std::vector<double> lat_samples;
            std::vector<double> lat_base_samples;
            std::vector<double> thr_samples;
            std::vector<double> thr_base_samples;
            bool unsupported = false;

            for (int r = 0; r < repeats; ++r) {
                int iters = it_latency;
                err = clSetKernelArg(k_lat, 0, sizeof(int), &what);
                err |= clSetKernelArg(k_lat, 1, sizeof(int), &iters);
                err |= clSetKernelArg(k_lat, 2, sizeof(cl_mem), &buf);
                err |= clSetKernelArg(k_base_lat, 0, sizeof(int), &iters);
                err |= clSetKernelArg(k_base_lat, 1, sizeof(cl_mem), &buf);
                if (err != CL_SUCCESS) {
                    unsupported = true;
                    break;
                }
                double lat_one = run_kernel_host_timed_ns(prof_queue, k_lat, global, local);
                double lat_base_one = run_kernel_host_timed_ns(prof_queue, k_base_lat, global, local);
                if (lat_one <= 0.0 || lat_base_one <= 0.0) {
                    unsupported = true;
                    break;
                }
                lat_samples.push_back(lat_one);
                lat_base_samples.push_back(lat_base_one);

                // quick sink check once at first candidate/repeat
                if (g == group_candidates.front() && r == 0) {
                    std::fill(host.begin(), host.end(), 0);
                    clEnqueueReadBuffer(prof_queue, buf, CL_TRUE, 0, sizeof(int64_t) * host.size(),
                                        host.data(), 0, nullptr, nullptr);
                    if (!has_valid_sink(host, global)) {
                        unsupported = true;
                        break;
                    }
                    if (what == 6 && has_strict_invalid_sink(host, global)) {
                        unsupported = true;
                        break;
                    }
                }

                iters = it_throughput;
                err = clSetKernelArg(k_thr, 0, sizeof(int), &what);
                err |= clSetKernelArg(k_thr, 1, sizeof(int), &iters);
                err |= clSetKernelArg(k_thr, 2, sizeof(cl_mem), &buf);
                err |= clSetKernelArg(k_base_thr, 0, sizeof(int), &iters);
                err |= clSetKernelArg(k_base_thr, 1, sizeof(cl_mem), &buf);
                if (err != CL_SUCCESS) {
                    unsupported = true;
                    break;
                }
                double thr_one = run_kernel_host_timed_ns(prof_queue, k_thr, global, local);
                double thr_base_one = run_kernel_host_timed_ns(prof_queue, k_base_thr, global, local);
                if (thr_one <= 0.0 || thr_base_one <= 0.0) {
                    unsupported = true;
                    break;
                }
                thr_samples.push_back(thr_one);
                thr_base_samples.push_back(thr_base_one);
            }

            if (unsupported || lat_samples.empty() || thr_samples.empty()) {
                continue;
            }

            const double lat_med = median(lat_samples);
            const double lat_base_med = median(lat_base_samples);
            const double thr_med = median(thr_samples);
            const double thr_base_med = median(thr_base_samples);

            // Prefer baseline-diff, but if diff collapses due timer granularity/noise,
            // fall back to absolute kernel time to keep throughput ranking meaningful.
            double lat_ns = lat_med - lat_base_med;
            if (lat_ns <= 0.0 || lat_ns < 0.05 * lat_med) {
                lat_ns = lat_med;
            }
            double thr_ns = thr_med - thr_base_med;
            if (thr_ns <= 0.0 || thr_ns < 0.05 * thr_med) {
                thr_ns = thr_med;
            }
            lat_ns = std::max(1.0, lat_ns);
            thr_ns = std::max(1.0, thr_ns);
            double lat_ns_per_op = lat_ns / (double(global) * double(it_latency) * double(tc.latency_ops_per_iter));
            double thr_total_ops = double(global) * double(it_throughput) * double(tc.throughput_ops_per_iter);
            double gops = thr_total_ops / thr_ns;

            if (verbose_groups) {
                std::cout << "  " << tc.name
                          << " groups=" << g
                          << " global=" << global
                          << " ns=" << thr_ns
                          << " throughput=" << gops
                          << " Gops/s" << std::endl;
            }

            if (gops > best_gops) {
                best_gops = gops;
                best_groups = g;
                best_lat_ns = lat_ns_per_op;
            }
        }

        if (best_gops <= 0.0) {
            std::cout << tc.name << " : unsupported on this compiler/device (skipped)" << std::endl;
            continue;
        }

        cl_uint mhz = 0;
        clGetDeviceInfo(ctx.device, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(mhz), &mhz, nullptr);
        double cycles_per_op_est = best_lat_ns * (double(mhz) * 1e-3);

        std::cout << tc.name
                  << " : latency " << best_lat_ns << " ns/op"
                  << " (~" << cycles_per_op_est << " cycles/op @ " << mhz << "MHz)"
                  << "; best throughput " << best_gops << " Gops/s"
                  << " at groups=" << best_groups
                  << " (global=" << (best_groups * local) << ")" << std::endl;
    }

    clReleaseMemObject(buf);
    clReleaseKernel(k_base_lat);
    clReleaseKernel(k_base_thr);
    clReleaseKernel(k_lat);
    clReleaseKernel(k_thr);
    clReleaseProgram(prog);
    clReleaseCommandQueue(prof_queue);
    cgbn::opencl::destroy_context(ctx);
    return EXIT_SUCCESS;
}
