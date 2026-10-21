#ifndef CGBN_OPENCL_H
#define CGBN_OPENCL_H

#include <CL/cl.h>
#include <string>

namespace cgbn {
namespace opencl {

struct context_t {
    cl_platform_id platform = nullptr;
    cl_device_id device = nullptr;
    cl_context ctx = nullptr;
    cl_command_queue queue = nullptr;
};

// Create a simple OpenCL context for the given device (or the first available)
// Returns 0 on success, non-zero on error (cl_int)
cl_int create_context(context_t &out);
cl_int create_context_with_device_index(context_t &out, int device_index);
cl_int destroy_context(context_t &ctx);

// Build a program from source in-memory. The returned program must be released
// with clReleaseProgram().
cl_program build_program_from_source(context_t &ctx, const char *source, const char *options, cl_int &errcode);

// Helper: load a text resource from disk (returns empty string on failure)
std::string load_text_file(const char *path);

// Load kernel source: tries rel_path from cwd, then ../.. prefixes (for build/Debug runs).
std::string load_kernel_file(const char *rel_path);

// Load ECM stage1 kernel relative to kernels/opencl/ (ECM_KERNEL_ROOT or ECM_KERNEL_ROOT_DEFAULT).
std::string load_ecm_stage1_kernel_file(const char *rel_path);

} // namespace opencl
} // namespace cgbn

#endif // CGBN_OPENCL_H
