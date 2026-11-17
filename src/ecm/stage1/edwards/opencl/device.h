#pragma once
#include "../../../opencl/runtime/cgbn_opencl.h"
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include "../../execution.h"
#include "ecm/application/configuration/runtime.h"

namespace ecm_opencl_stage1 {
inline void check(cl_int status, const char* operation) {
    if (status != CL_SUCCESS)
        throw std::runtime_error(std::string(operation) + " (OpenCL " + std::to_string(status) + ")");
}

template<class T, auto Release> class Handle {
    T value_;
public:
    explicit Handle(T value) : value_(value) {}
    Handle(Handle&& rhs) noexcept : value_(std::exchange(rhs.value_, nullptr)) {}
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
    Handle& operator=(Handle&& rhs) noexcept {
        std::swap(value_, rhs.value_);
        return *this;
    }
    ~Handle() { if (value_) Release(value_); }
    operator T() const { return value_; }
};
using Buffer = Handle<cl_mem, clReleaseMemObject>;
using Program = Handle<cl_program, clReleaseProgram>;
using Kernel = Handle<cl_kernel, clReleaseKernel>;

enum class Square { Multiply, Cios, Fips };
struct FieldPlan {
    bool fips_mul = false;
    Square square = Square::Multiply;
    bool ptx = false;
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

inline FieldPlan field_plan(const ecm_stage1::ExecutionOptions& opt, unsigned limbs) {
    const auto& mul = opt.execution.gpu_mul_path;
    const auto& sqr = opt.execution.gpu_sqr_path;
    FieldPlan plan;
    plan.fips_mul = mul == "fips" || mul == "fips_ptx";
    plan.ptx = mul == "fips_ptx" || sqr == "fips_ptx";
    if (sqr == "cios")
        plan.square = Square::Cios;
    else if (sqr == "fips" || sqr == "fips_ptx")
        plan.square = Square::Fips;
    if (plan.ptx && (mul == "fips" || sqr == "fips"))
        throw std::invalid_argument("OpenCL Edwards cannot mix portable FIPS with fips_ptx");
    if (limbs > 16 && (plan.fips_mul || plan.square == Square::Fips))
        throw std::invalid_argument("OpenCL Edwards FIPS supports up to 512-bit moduli");
    return plan;
}

class Device {
    FieldPlan field_;
    const unsigned limbs_;
    const size_t group_;
    cgbn::opencl::context_t context_;
    bool unroll_ = true;
    Kernel dictionary_{nullptr};
    Kernel slice_{nullptr};
public:
    static constexpr const char* name = "opencl";
    static constexpr const char* label = "OpenCL";
    static constexpr const char* checkpoint_format = "OCL_ED_NAF_V1";

    Device(const ecm_stage1::ExecutionOptions& opt, unsigned limbs)
        : field_(field_plan(opt, limbs)), limbs_(limbs), group_(ecm_runtime_config().wg_size) {
        check(cgbn::opencl::create_context_with_device_index(context_, opt.execution.device_index, true),
              "create Edwards context");
        try {
            const auto vendor = info<cl_uint>(CL_DEVICE_VENDOR_ID);
            if (field_.ptx && vendor != 0x10de)
                throw std::invalid_argument("OpenCL Edwards fips_ptx requires an NVIDIA device");
            if (!info<cl_bool>(CL_DEVICE_ENDIAN_LITTLE))
                throw std::runtime_error("OpenCL Edwards requires a little-endian device");
            unroll_ = limbs_ < 16 || vendor != 0x1002;
        } catch (...) {
            cgbn::opencl::destroy_context(context_);
            throw;
        }
    }
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    ~Device() { cgbn::opencl::destroy_context(context_); }

    template<class T> T info(cl_device_info key) const {
        T value{};
        check(clGetDeviceInfo(context_.device, key, sizeof(value), &value, nullptr), "query Edwards device");
        return value;
    }
    uint64_t max_buffer() const { return info<cl_ulong>(CL_DEVICE_MAX_MEM_ALLOC_SIZE); }
    uint64_t memory_budget() const { return info<cl_ulong>(CL_DEVICE_GLOBAL_MEM_SIZE) / 2; }

    std::string details() const {
        const char* mul = field_.fips_mul ? (field_.ptx ? "fips_ptx" : "fips") : "cios";
        const char* sqr = field_.square == Square::Fips ? (field_.ptx ? "fips_ptx_square" : "fips_square") :
                          field_.square == Square::Cios ? "cios" : mul;
        return "workgroup=" + (group_ ? std::to_string(group_) : "auto") +
               " field_mul=" + mul + " field_sqr=" + sqr + " limb_unroll=" + std::to_string(unroll_);
    }

    void prepare(bool two_xy) {
        const auto p = program(two_xy);
        dictionary_ = make_kernel(p, "edwards_dictionary");
        slice_ = make_kernel(p, "edwards_slice");
        validate_group(dictionary_, group_);
        validate_group(slice_, group_);
    }

    double dictionary(cl_mem params, cl_mem seeds, cl_mem table, cl_mem scratch, uint32_t count,
                      uint32_t np0, uint32_t first, uint32_t stop, uint32_t weights) {
        kernel_args(dictionary_, params, seeds, table, scratch, count, np0, first, stop, weights);
        return run(dictionary_, count, group_);
    }
    double slice(cl_mem params, cl_mem points, cl_mem table, cl_mem nonzero, cl_mem codes,
                 uint32_t count, uint32_t np0, uint32_t bits, uint32_t first, uint32_t stop, uint32_t weights) {
        kernel_args(slice_, params, points, table, nonzero, codes, count, np0, bits, first, stop, weights);
        return run(slice_, count, group_);
    }

    Program program(bool two_xy) {
        std::string source;
        for (const char* path : {"edwards/field.clh", "edwards/point.clh", "edwards/stage1.cl"}) {
            const auto part = cgbn::opencl::load_ecm_stage1_kernel_file(path);
            if (part.empty())
                throw std::runtime_error(std::string("missing OpenCL kernel resource ") + path);
            source += part + '\n';
        }
        const auto options = "-cl-std=CL1.2 -DLIMBS=" + std::to_string(limbs_) +
            " -DTWO_XY=" + std::to_string(two_xy) + " -DEDWARDS_UNROLL=" + std::to_string(unroll_) +
            " -DEDWARDS_FIPS=" + std::to_string(field_.fips_mul) +
            " -DEDWARDS_FIPS_SQUARE=" + std::to_string(field_.square == Square::Fips) +
            " -DEDWARDS_CIOS_SQUARE=" + std::to_string(field_.square == Square::Cios) +
            " -DEDWARDS_FIPS_PTX=" + std::to_string(field_.ptx);
        cl_int status = 0;
        const auto program = cgbn::opencl::build_program_from_source(context_, source.c_str(),
                                                                    options.c_str(), status);
        check(status, "build Edwards kernels");
        return Program(program);
    }
    Buffer buffer(size_t bytes, bool read_only) const {
        const auto flags = read_only ? CL_MEM_READ_ONLY : CL_MEM_READ_WRITE;
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


}
