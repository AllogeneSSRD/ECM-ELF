#pragma once
// Internal to native OpenCL operators. Do not expose these handles to task/planner APIs.
#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 120
#endif
#include <CL/cl.h>
#include "runtime.h"
namespace ecm_opencl::ntt {
struct NativeAccess {
    static cl_context context(const Runtime &);
    static cl_command_queue queue(const Runtime &);
    static cl_device_id device(const Runtime &);
    static cl_mem memory(const Buffer &);
    static bool same_owner(const Buffer &, const Buffer &);
    static void span(const Runtime &, const Buffer &, std::size_t words, std::size_t offset);
    static std::string resource_root(const Runtime &);
    // Returns a retained program. Each caller still creates private mutable
    // kernels; a cache hit must never share clSetKernelArg state between plans.
    static cl_program program(const Runtime &,const std::string &source,const std::string &options);
};
} // namespace ecm_opencl::ntt
