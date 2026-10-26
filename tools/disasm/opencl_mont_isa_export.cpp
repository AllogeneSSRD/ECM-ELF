#include "ecm/opencl/runtime/device.h"
#include "ecm/opencl/runtime/cgbn_opencl.h"

#include <CL/cl.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

std::string strip_include_line(const std::string &src, const std::string &line) {
    std::string out = src;
    size_t pos = out.find(line);
    if (pos != std::string::npos) {
        out.erase(pos, line.size());
    }
    return out;
}

bool write_binary(const std::string &path, const unsigned char *data, size_t size) {
    std::ofstream ofs(path, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!ofs.is_open()) return false;
    ofs.write(reinterpret_cast<const char *>(data), (std::streamsize)size);
    return ofs.good();
}

void print_kernel_resource_row(const char *kname, cl_kernel k, cl_device_id dev) {
    size_t private_bytes = 0, local_bytes = 0, pref = 0, wg = 0;
    clGetKernelWorkGroupInfo(k, dev, CL_KERNEL_PRIVATE_MEM_SIZE, sizeof(size_t), &private_bytes, nullptr);
    clGetKernelWorkGroupInfo(k, dev, CL_KERNEL_LOCAL_MEM_SIZE, sizeof(size_t), &local_bytes, nullptr);
    clGetKernelWorkGroupInfo(k, dev, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &pref, nullptr);
    clGetKernelWorkGroupInfo(k, dev, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &wg, nullptr);
    std::cout << "  " << kname
              << " | private=" << private_bytes
              << " B | local=" << local_bytes
              << " B | prefWG=" << pref
              << " | maxWG=" << wg << std::endl;
}

} // namespace

int main(int argc, char **argv) {
    const int bits = 4096;
    const int tpi = 8;
    bool list_devices = true;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--no-list") {
            list_devices = false;
            continue;
        }
        if (a == "-h" || a == "--help") {
            std::cout << "Usage: opencl_mont_isa_export [--no-list]\n";
            return 0;
        }
    }

    if (!configureFirstAmdGpuDevice(list_devices)) {
        return 1;
    }

    cgbn::opencl::context_t ctx;
    cl_int err = cgbn::opencl::create_context(ctx);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create OpenCL context: " << err << std::endl;
        return 1;
    }

    const uint32_t limbs = bits / 32;
    const std::string mont_priv =
        cgbn::opencl::load_kernel_file("bench/mont_priv.cl");
    const std::string mont_priv_opt =
        cgbn::opencl::load_kernel_file("bench/mont_priv_opt.cl");
    const std::string mont_mul_manual_src = cgbn::opencl::load_kernel_file(
        "bench/mont_mul_unroll_only_512_manual_generated.cl");
    std::string mont_priv_bench_src =
        cgbn::opencl::load_kernel_file("bench/mont_priv_bench.cl");
    std::string mont_priv_opt_bench_src =
        cgbn::opencl::load_kernel_file("bench/mont_priv_opt_bench.cl");
    const std::string bench_src =
        cgbn::opencl::load_kernel_file("bench/ecm_addsub_bench.cl");
    const std::string mont_wg_src =
        cgbn::opencl::load_kernel_file("bench/mont_wg.cl");
    std::string mont_wg_bench_src =
        cgbn::opencl::load_kernel_file("bench/mont_wg_bench.cl");
    if (mont_priv.empty() || mont_priv_opt.empty() || mont_priv_bench_src.empty() ||
        mont_priv_opt_bench_src.empty() || bench_src.empty() || mont_wg_src.empty() ||
        mont_wg_bench_src.empty() ||
        (limbs == 16u && mont_mul_manual_src.empty())) {
        std::cerr << "Failed to load kernel sources." << std::endl;
        cgbn::opencl::destroy_context(ctx);
        return 1;
    }

    mont_wg_bench_src = strip_include_line(mont_wg_bench_src, "#include \"mont_wg.cl\"");
    mont_priv_bench_src = strip_include_line(mont_priv_bench_src, "#include \"mont_priv.cl\"");
    mont_priv_opt_bench_src = strip_include_line(mont_priv_opt_bench_src, "#include \"mont_priv_opt.cl\"");
    const std::string src = mont_wg_src + "\n" + mont_priv + "\n" + mont_priv_opt + "\n" +
                            mont_mul_manual_src + "\n" + mont_wg_bench_src + "\n" +
                            mont_priv_bench_src + "\n" + mont_priv_opt_bench_src + "\n" + bench_src;

    char build_opts[128];
    snprintf(build_opts, sizeof(build_opts), "-DMAX_LIMBS=%u -DTPI=%d", limbs, tpi);
    cl_int buildErr = CL_SUCCESS;
    cl_program program = cgbn::opencl::build_program_from_source(ctx, src.c_str(), build_opts, buildErr);
    if (program == nullptr || buildErr != CL_SUCCESS) {
        std::cerr << "Failed to build mont ISA export program: " << buildErr << std::endl;
        cgbn::opencl::destroy_context(ctx);
        return 1;
    }

    // Export OpenCL binary (often contains AMD code object / ISA container).
    size_t bin_size = 0;
    clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &bin_size, nullptr);
    if (bin_size > 0) {
        std::vector<unsigned char> bin(bin_size);
        unsigned char *ptr = bin.data();
        clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(unsigned char *), &ptr, nullptr);
        const std::string out_bin = "bench/mont_isa_4096_amd.bin";
        if (write_binary(out_bin, bin.data(), bin.size())) {
            std::cout << "Exported binary: " << out_bin << " (" << bin.size() << " bytes)" << std::endl;
        } else {
            std::cerr << "Failed to write binary export file." << std::endl;
        }
    } else {
        std::cerr << "No program binary available from runtime." << std::endl;
    }

    std::cout << "Kernel resource summary (4096-bit, TPI=8):" << std::endl;
    const char *kernels[] = {
        "ecm_mont_mul_priv_bench",
        "ecm_mont_sqr_priv_bench",
        "ecm_mont_mul_priv_opt_bench",
        "ecm_mont_sqr_priv_opt_bench",
        "cgbn_mont_mul_wg_bench",
        "cgbn_mont_sqr_wg_bench",
    };
    for (const char *kname : kernels) {
        cl_int kerr = CL_SUCCESS;
        cl_kernel k = clCreateKernel(program, kname, &kerr);
        if (kerr != CL_SUCCESS) {
            std::cerr << "  " << kname << " | create failed: " << kerr << std::endl;
            continue;
        }
        print_kernel_resource_row(kname, k, ctx.device);
        clReleaseKernel(k);
    }

    clReleaseProgram(program);
    cgbn::opencl::destroy_context(ctx);

    std::cout << "Note: install RGA/llvm-objdump to disassemble exported binary into ISA text." << std::endl;
    return 0;
}
