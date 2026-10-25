#include "ecm/backend/opencl/runtime/device.h"
#include "ecm/backend/opencl/runtime/cgbn_opencl.h"
#include "opencl_ecm_addsub_manifest.h"

#include <CL/cl.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

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
    int bits = 4096;
    bool list_devices = true;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--no-list") {
            list_devices = false;
            continue;
        }
        if (a == "--bits" && i + 1 < argc) {
            bits = std::stoi(argv[++i]);
            continue;
        }
        if (a == "-h" || a == "--help") {
            std::cout << "Usage: opencl_addsub_isa_export [--bits <bits>] [--no-list]\n";
            return 0;
        }
    }

    if (bits <= 0 || (bits % 32) != 0) {
        std::cerr << "bits must be a positive multiple of 32." << std::endl;
        return 1;
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

    const uint32_t limbs = (uint32_t)bits / 32u;
    std::string bench_src =
        cgbn::opencl::load_kernel_file("bench/ecm_addsub_bench.cl");
    if (bench_src.empty()) {
        std::cerr << "Failed to load ecm_addsub_bench.cl" << std::endl;
        cgbn::opencl::destroy_context(ctx);
        return 1;
    }
    const bool asm_enabled = (limbs == 128u || limbs == 8u || limbs == 16u);
    const bool asm_b64 = (limbs == 128u);
    const EcmAddSubBuildManifest manifest =
        opencl_ecm_addsub_build_manifest((uint32_t)limbs, asm_enabled, asm_b64);
    for (const std::string &rel : manifest.source_paths) {
        if (rel == "bench/ecm_addsub_bench.cl") {
            continue;
        }
        const std::string extra = cgbn::opencl::load_kernel_file(rel.c_str());
        if (!extra.empty()) {
            bench_src += "\n" + extra;
        }
    }

    char build_opts[96];
    if (limbs == 128u || limbs == 16u || limbs == 8u) {
        snprintf(build_opts, sizeof(build_opts), "-DMAX_LIMBS=%u -DMP_ADDMOD_ASM_ENABLE=1", limbs);
    } else {
        snprintf(build_opts, sizeof(build_opts), "-DMAX_LIMBS=%u", limbs);
    }
    cl_int buildErr = CL_SUCCESS;
    cl_program program = cgbn::opencl::build_program_from_source(ctx, bench_src.c_str(), build_opts, buildErr);
    if (program == nullptr || buildErr != CL_SUCCESS) {
        std::cerr << "Failed to build addsub ISA export program: " << buildErr << std::endl;
        cgbn::opencl::destroy_context(ctx);
        return 1;
    }

    size_t bin_size = 0;
    clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &bin_size, nullptr);
    if (bin_size > 0) {
        std::vector<unsigned char> bin(bin_size);
        unsigned char *ptr = bin.data();
        clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(unsigned char *), &ptr, nullptr);
        const std::string out_bin = "bench/addsub_isa_" + std::to_string(bits) + "_amd.bin";
        if (write_binary(out_bin, bin.data(), bin.size())) {
            std::cout << "Exported binary: " << out_bin << " (" << bin.size() << " bytes)" << std::endl;
        } else {
            std::cerr << "Failed to write binary export file." << std::endl;
        }
    } else {
        std::cerr << "No program binary available from runtime." << std::endl;
    }

    std::cout << "Kernel resource summary (" << bits << "-bit pure add/sub/mod kernels):" << std::endl;
    const char *kernels[] = {
        "ecm_mp_add_n",
        "ecm_mp_sub_n",
        "ecm_mp_add_mod_fused",
        "ecm_mp_add_mod_legacy",
        "ecm_mp_add_mod_mask",
        "ecm_mp_add_mod_fused_unroll",
        "ecm_mp_add_mod_fused_unroll_asm",
        "ecm_mp_sub_mod",
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
    return 0;
}
