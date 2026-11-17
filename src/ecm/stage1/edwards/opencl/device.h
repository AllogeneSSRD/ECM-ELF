#pragma once
#include "../../../opencl/runtime/cgbn_opencl.h"
#include <cstdint>
#include <stdexcept>
#include <string>

namespace ecm_opencl_stage1 {
inline void check(cl_int status, const char* operation) {
    if (status != CL_SUCCESS)
        throw std::runtime_error(std::string(operation) + " (OpenCL " + std::to_string(status) + ")");
}

template<class T, auto Release> class Handle {
    T value_;
public:
    explicit Handle(T value) : value_(value) {}
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
    ~Handle() { if (value_) Release(value_); }
    operator T() const { return value_; }
};
using Buffer = Handle<cl_mem, clReleaseMemObject>;
using Program = Handle<cl_program, clReleaseProgram>;
using Kernel = Handle<cl_kernel, clReleaseKernel>;

class Device {
    cgbn::opencl::context_t context_;
public:
    explicit Device(int index) {
        check(cgbn::opencl::create_context_with_device_index(context_, index, true), "create Edwards context");
    }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    ~Device() { cgbn::opencl::destroy_context(context_); }

    template<class T> T info(cl_device_info key) const {
        T value{};
        check(clGetDeviceInfo(context_.device, key, sizeof(value), &value, nullptr), "query Edwards device");
        return value;
    }
    Program program(unsigned limbs, bool two_xy, bool unroll) {
        std::string source;
        for (const char* path : {"edwards/field.clh", "edwards/point.clh", "edwards/stage1.cl"}) {
            const auto part = cgbn::opencl::load_ecm_stage1_kernel_file(path);
            if (part.empty())
                throw std::runtime_error(std::string("missing OpenCL kernel resource ") + path);
            source += part + '\n';
        }
        const auto options = "-cl-std=CL1.2 -DLIMBS=" + std::to_string(limbs) +
            " -DTWO_XY=" + std::to_string(two_xy) + " -DEDWARDS_UNROLL=" + std::to_string(unroll);
        cl_int status = 0;
        const auto program = cgbn::opencl::build_program_from_source(context_, source.c_str(),
                                                                    options.c_str(), status);
        check(status, "build Edwards kernels");
        return Program(program);
    }
    Buffer buffer(size_t bytes, cl_mem_flags flags) const {
        cl_int status = 0;
        const auto memory = clCreateBuffer(context_.ctx, flags, bytes, nullptr, &status);
        check(status, "allocate Edwards buffer");
        return Buffer(memory);
    }
    void upload(cl_mem buffer, const void* data, size_t bytes) const {
        check(clEnqueueWriteBuffer(context_.queue, buffer, CL_TRUE, 0, bytes, data, 0, nullptr, nullptr),
              "upload Edwards buffer");
    }
    void download(cl_mem buffer, void* data, size_t bytes) const {
        check(clEnqueueReadBuffer(context_.queue, buffer, CL_TRUE, 0, bytes, data, 0, nullptr, nullptr),
              "download Edwards buffer");
    }
    double run(cl_kernel kernel, size_t count, size_t group) const {
        const size_t global = group ? ((count + group - 1) / group) * group : count;
        cl_event raw = nullptr;
        check(clEnqueueNDRangeKernel(context_.queue, kernel, 1, nullptr, &global, group ? &group : nullptr,
                                     0, nullptr, &raw), "submit Edwards kernel");
        const Handle<cl_event, clReleaseEvent> event(raw);
        check(clWaitForEvents(1, &raw), "wait for Edwards kernel");
        cl_ulong start = 0, stop = 0;
        check(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, nullptr),
              "profile Edwards kernel start");
        check(clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(stop), &stop, nullptr),
              "profile Edwards kernel end");
        return double(stop - start) / 1e6;
    }
    void validate_group(cl_kernel kernel, size_t group) const {
        size_t limit = 0;
        check(clGetKernelWorkGroupInfo(kernel, context_.device, CL_KERNEL_WORK_GROUP_SIZE,
                                       sizeof(limit), &limit, nullptr), "query Edwards work-group limit");
        if (group > limit)
            throw std::invalid_argument("OpenCL Edwards work-group exceeds the kernel's device limit");
    }
};

inline Kernel make_kernel(cl_program program, const char* name) {
    cl_int status = 0;
    const auto kernel = clCreateKernel(program, name, &status);
    check(status, name);
    return Kernel(kernel);
}

template<class... T> void kernel_args(cl_kernel kernel, const T&... values) {
    cl_uint index = 0;
    (check(clSetKernelArg(kernel, index++, sizeof(values), &values), "set Edwards kernel argument"), ...);
}
}
