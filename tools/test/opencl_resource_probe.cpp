// Link the actual release runtime object; no OpenCL context or GPU is created.
#include "ecm/application/configuration/runtime.h"
#include "ecm/backend/opencl/runtime/cgbn_opencl.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>

EcmRuntimeConfig &ecm_runtime_config() { static EcmRuntimeConfig config; return config; }
static std::string text(const std::filesystem::path &path) {
    std::ifstream stream(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}
int main(int argc, char **argv) {
    if (argc != 3) return 2;
    const std::filesystem::path reference(argv[1]);
    unsigned count = 0;
    for (const auto &entry : std::filesystem::recursive_directory_iterator(reference)) {
        if (!entry.is_regular_file()) continue;
        const auto relative = entry.path().lexically_relative(reference).generic_string();
        if (cgbn::opencl::load_ecm_stage1_kernel_file(relative.c_str()) != text(entry.path())) {
            std::cerr << "Resource mismatch: " << relative << '\n'; return 3;
        }
        ++count;
    }
    if (cgbn::opencl::load_kernel_file("bench/mont.cl") != text(reference / "diagnostics/mont.cl")) return 4;
    ecm_runtime_config().kernel_root = argv[2];
    if (cgbn::opencl::load_ecm_stage1_kernel_file("override.cl") != "explicit override\n") return 5;
    if (cgbn::opencl::load_ecm_stage1_kernel_file("diagnostics/mont.cl") != "explicit override\n") return 6;
    std::cout << "PASS relocated resources=" << count << ", diagnostic alias and explicit root\n";
}
