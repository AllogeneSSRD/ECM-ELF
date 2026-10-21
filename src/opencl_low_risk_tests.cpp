#include "opencl_low_risk_tests.h"

#include "ecm/backend/opencl/runtime/cgbn_opencl.h"

#include <CL/cl.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

using LimbVec = std::vector<cl_uint>;
constexpr size_t kLimbs = 4;

void checkCl(cl_int err, const char *message) {
    if (err != CL_SUCCESS) {
        std::cerr << message << " Error: " << err << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

LimbVec cpuSetUi32(cl_uint value) {
    LimbVec result(kLimbs, 0u);
    result[0] = value;
    return result;
}

LimbVec cpuSetOne() {
    LimbVec result(kLimbs, 0u);
    result[0] = 1u;
    return result;
}

LimbVec cpuBitwiseAnd(const LimbVec &a, const LimbVec &b) {
    LimbVec result(kLimbs, 0u);
    for (size_t i = 0; i < kLimbs; ++i) {
        result[i] = a[i] & b[i];
    }
    return result;
}

LimbVec cpuBitwiseIor(const LimbVec &a, const LimbVec &b) {
    LimbVec result(kLimbs, 0u);
    for (size_t i = 0; i < kLimbs; ++i) {
        result[i] = a[i] | b[i];
    }
    return result;
}

LimbVec cpuBitwiseXor(const LimbVec &a, const LimbVec &b) {
    LimbVec result(kLimbs, 0u);
    for (size_t i = 0; i < kLimbs; ++i) {
        result[i] = a[i] ^ b[i];
    }
    return result;
}

LimbVec cpuBitwiseNot(const LimbVec &a) {
    LimbVec result(kLimbs, 0u);
    for (size_t i = 0; i < kLimbs; ++i) {
        result[i] = ~a[i];
    }
    return result;
}

LimbVec cpuAddUi32(const LimbVec &a, cl_uint value) {
    LimbVec result(kLimbs, 0u);
    std::uint64_t carry = value;
    for (size_t i = 0; i < kLimbs; ++i) {
        std::uint64_t sum = static_cast<std::uint64_t>(a[i]) + carry;
        result[i] = static_cast<cl_uint>(sum);
        carry = sum >> 32;
    }
    return result;
}

LimbVec cpuSubUi32(const LimbVec &a, cl_uint value) {
    LimbVec result(kLimbs, 0u);
    std::uint64_t borrow = value;
    for (size_t i = 0; i < kLimbs; ++i) {
        std::uint64_t minuend = static_cast<std::uint64_t>(a[i]);
        std::uint64_t wide = minuend - borrow;
        result[i] = static_cast<cl_uint>(wide);
        borrow = minuend < borrow ? 1u : 0u;
    }
    return result;
}

LimbVec cpuShiftLeft(const LimbVec &a, cl_uint shift) {
    LimbVec result(kLimbs, 0u);
    const cl_uint limbShift = shift / 32u;
    const cl_uint bitShift = shift % 32u;
    for (size_t i = 0; i < kLimbs; ++i) {
        if (i < limbShift) {
            result[i] = 0u;
            continue;
        }
        size_t src = i - limbShift;
        cl_uint lo = a[src] << bitShift;
        cl_uint hi = 0u;
        if (bitShift != 0u && src > 0u) {
            hi = a[src - 1] >> (32u - bitShift);
        }
        result[i] = lo | hi;
    }
    return result;
}

LimbVec cpuShiftRight(const LimbVec &a, cl_uint shift) {
    LimbVec result(kLimbs, 0u);
    const cl_uint limbShift = shift / 32u;
    const cl_uint bitShift = shift % 32u;
    for (size_t i = 0; i < kLimbs; ++i) {
        size_t src = i + limbShift;
        if (src >= kLimbs) {
            result[i] = 0u;
            continue;
        }
        cl_uint lo = a[src] >> bitShift;
        cl_uint hi = 0u;
        if (bitShift != 0u && (src + 1) < kLimbs) {
            hi = a[src + 1] << (32u - bitShift);
        }
        result[i] = lo | hi;
    }
    return result;
}

bool sameLimbs(const LimbVec &lhs, const LimbVec &rhs) {
    if (lhs.size() != rhs.size()) {
        return false;
    }

    for (size_t i = 0; i < lhs.size(); ++i) {
        if (lhs[i] != rhs[i]) {
            return false;
        }
    }

    return true;
}

int cpuCompare(const LimbVec &a, const LimbVec &b) {
    for (size_t i = kLimbs; i-- > 0;) {
        if (a[i] > b[i]) return 1;
        if (a[i] < b[i]) return -1;
    }
    return 0;
}

int cpuIsZero(const LimbVec &a) {
    for (size_t i = 0; i < kLimbs; ++i) {
        if (a[i] != 0u) return 0;
    }
    return 1;
}

int cpuEquals(const LimbVec &a, const LimbVec &b) {
    return sameLimbs(a, b) ? 1 : 0;
}

bool runUnarySetUi32(cl_context ctx, cl_command_queue queue, cl_program program, cl_uint value) {
    cl_int err = CL_SUCCESS;
    cl_kernel kernel = clCreateKernel(program, "cgbn_set_ui32", &err);
    checkCl(err, "Failed to create cgbn_set_ui32 kernel");

    LimbVec output(kLimbs, 0xDEADBEEFu);
    cl_mem outBuffer = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                      sizeof(cl_uint) * kLimbs, output.data(), &err);
    checkCl(err, "Failed to create output buffer");

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &outBuffer);
    checkCl(err, "Failed to set output arg");
    err = clSetKernelArg(kernel, 1, sizeof(cl_uint), &value);
    checkCl(err, "Failed to set value arg");
    cl_uint limbs = static_cast<cl_uint>(kLimbs);
    err = clSetKernelArg(kernel, 2, sizeof(cl_uint), &limbs);
    checkCl(err, "Failed to set limbs arg");

    size_t globalWorkSize = 1;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize, nullptr, 0, nullptr, nullptr);
    checkCl(err, "Failed to enqueue cgbn_set_ui32");

    err = clFinish(queue);
    checkCl(err, "Failed to finish queue");

    err = clEnqueueReadBuffer(queue, outBuffer, CL_TRUE, 0, sizeof(cl_uint) * kLimbs, output.data(), 0, nullptr, nullptr);
    checkCl(err, "Failed to read cgbn_set_ui32 output");

    const LimbVec expected = cpuSetUi32(value);
    bool ok = sameLimbs(output, expected);

    clReleaseMemObject(outBuffer);
    clReleaseKernel(kernel);
    return ok;
}

bool runSetOne(cl_context ctx, cl_command_queue queue, cl_program program) {
    cl_int err = CL_SUCCESS;
    cl_kernel kernel = clCreateKernel(program, "cgbn_set_one", &err);
    checkCl(err, "Failed to create cgbn_set_one kernel");

    LimbVec output(kLimbs, 0xDEADBEEFu);
    cl_mem outBuffer = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                      sizeof(cl_uint) * kLimbs, output.data(), &err);
    checkCl(err, "Failed to create output buffer");

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &outBuffer);
    checkCl(err, "Failed to set output arg");
    cl_uint limbs = static_cast<cl_uint>(kLimbs);
    err = clSetKernelArg(kernel, 1, sizeof(cl_uint), &limbs);
    checkCl(err, "Failed to set limbs arg");

    size_t globalWorkSize = 1;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize, nullptr, 0, nullptr, nullptr);
    checkCl(err, "Failed to enqueue cgbn_set_one");

    err = clFinish(queue);
    checkCl(err, "Failed to finish queue");

    err = clEnqueueReadBuffer(queue, outBuffer, CL_TRUE, 0, sizeof(cl_uint) * kLimbs, output.data(), 0, nullptr, nullptr);
    checkCl(err, "Failed to read cgbn_set_one output");

    const LimbVec expected = cpuSetOne();
    bool ok = sameLimbs(output, expected);

    clReleaseMemObject(outBuffer);
    clReleaseKernel(kernel);
    return ok;
}

bool runBinaryBitwise(cl_context ctx, cl_command_queue queue, cl_program program,
                      const char *kernelName, const LimbVec &lhs, const LimbVec &rhs,
                      const LimbVec &expected) {
    cl_int err = CL_SUCCESS;
    cl_kernel kernel = clCreateKernel(program, kernelName, &err);
    checkCl(err, kernelName);

    LimbVec output(kLimbs, 0u);
    cl_mem aBuffer = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(cl_uint) * kLimbs, const_cast<cl_uint *>(lhs.data()), &err);
    checkCl(err, "Failed to create lhs buffer");
    cl_mem bBuffer = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(cl_uint) * kLimbs, const_cast<cl_uint *>(rhs.data()), &err);
    checkCl(err, "Failed to create rhs buffer");
    cl_mem outBuffer = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                      sizeof(cl_uint) * kLimbs, output.data(), &err);
    checkCl(err, "Failed to create output buffer");

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &aBuffer);
    checkCl(err, "Failed to set arg 0");
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bBuffer);
    checkCl(err, "Failed to set arg 1");
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &outBuffer);
    checkCl(err, "Failed to set arg 2");
    cl_uint limbs = static_cast<cl_uint>(kLimbs);
    err = clSetKernelArg(kernel, 3, sizeof(cl_uint), &limbs);
    checkCl(err, "Failed to set limbs arg");

    size_t globalWorkSize = 1;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize, nullptr, 0, nullptr, nullptr);
    checkCl(err, "Failed to enqueue bitwise kernel");

    err = clFinish(queue);
    checkCl(err, "Failed to finish queue");

    err = clEnqueueReadBuffer(queue, outBuffer, CL_TRUE, 0, sizeof(cl_uint) * kLimbs, output.data(), 0, nullptr, nullptr);
    checkCl(err, "Failed to read bitwise output");

    bool ok = sameLimbs(output, expected);

    clReleaseMemObject(aBuffer);
    clReleaseMemObject(bBuffer);
    clReleaseMemObject(outBuffer);
    clReleaseKernel(kernel);
    return ok;
}

bool runUnaryAddSub(cl_context ctx, cl_command_queue queue, cl_program program,
                    const char *kernelName, const LimbVec &lhs, cl_uint value,
                    const LimbVec &expected) {
    cl_int err = CL_SUCCESS;
    cl_kernel kernel = clCreateKernel(program, kernelName, &err);
    checkCl(err, kernelName);

    LimbVec output(kLimbs, 0u);
    cl_mem aBuffer = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(cl_uint) * kLimbs, const_cast<cl_uint *>(lhs.data()), &err);
    checkCl(err, "Failed to create lhs buffer");
    cl_mem outBuffer = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                      sizeof(cl_uint) * kLimbs, output.data(), &err);
    checkCl(err, "Failed to create output buffer");

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &aBuffer);
    checkCl(err, "Failed to set arg 0");
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &outBuffer);
    checkCl(err, "Failed to set arg 1");
    err = clSetKernelArg(kernel, 2, sizeof(cl_uint), &value);
    checkCl(err, "Failed to set value arg");
    cl_uint limbs = static_cast<cl_uint>(kLimbs);
    err = clSetKernelArg(kernel, 3, sizeof(cl_uint), &limbs);
    checkCl(err, "Failed to set limbs arg");

    size_t globalWorkSize = 1;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize, nullptr, 0, nullptr, nullptr);
    checkCl(err, "Failed to enqueue add/sub kernel");

    err = clFinish(queue);
    checkCl(err, "Failed to finish queue");

    err = clEnqueueReadBuffer(queue, outBuffer, CL_TRUE, 0, sizeof(cl_uint) * kLimbs, output.data(), 0, nullptr, nullptr);
    checkCl(err, "Failed to read add/sub output");

    bool ok = sameLimbs(output, expected);

    clReleaseMemObject(aBuffer);
    clReleaseMemObject(outBuffer);
    clReleaseKernel(kernel);
    return ok;
}

bool runUnaryShift(cl_context ctx, cl_command_queue queue, cl_program program,
                   const char *kernelName, const LimbVec &lhs, cl_uint shift,
                   const LimbVec &expected) {
    cl_int err = CL_SUCCESS;
    cl_kernel kernel = clCreateKernel(program, kernelName, &err);
    checkCl(err, kernelName);

    LimbVec output(kLimbs, 0u);
    cl_mem aBuffer = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(cl_uint) * kLimbs, const_cast<cl_uint *>(lhs.data()), &err);
    checkCl(err, "Failed to create lhs buffer");
    cl_mem outBuffer = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                      sizeof(cl_uint) * kLimbs, output.data(), &err);
    checkCl(err, "Failed to create output buffer");

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &aBuffer);
    checkCl(err, "Failed to set arg 0");
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &outBuffer);
    checkCl(err, "Failed to set arg 1");
    err = clSetKernelArg(kernel, 2, sizeof(cl_uint), &shift);
    checkCl(err, "Failed to set shift arg");
    cl_uint limbs = static_cast<cl_uint>(kLimbs);
    err = clSetKernelArg(kernel, 3, sizeof(cl_uint), &limbs);
    checkCl(err, "Failed to set limbs arg");

    size_t globalWorkSize = 1;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize, nullptr, 0, nullptr, nullptr);
    checkCl(err, "Failed to enqueue shift kernel");

    err = clFinish(queue);
    checkCl(err, "Failed to finish queue");

    err = clEnqueueReadBuffer(queue, outBuffer, CL_TRUE, 0, sizeof(cl_uint) * kLimbs, output.data(), 0, nullptr, nullptr);
    checkCl(err, "Failed to read shift output");

    bool ok = sameLimbs(output, expected);

    clReleaseMemObject(aBuffer);
    clReleaseMemObject(outBuffer);
    clReleaseKernel(kernel);
    return ok;
}

bool runUnaryBitwise(cl_context ctx, cl_command_queue queue, cl_program program,
                     const char *kernelName, const LimbVec &lhs,
                     const LimbVec &expected) {
    cl_int err = CL_SUCCESS;
    cl_kernel kernel = clCreateKernel(program, kernelName, &err);
    checkCl(err, kernelName);

    LimbVec output(kLimbs, 0u);
    cl_mem aBuffer = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(cl_uint) * kLimbs, const_cast<cl_uint *>(lhs.data()), &err);
    checkCl(err, "Failed to create lhs buffer");
    cl_mem outBuffer = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                      sizeof(cl_uint) * kLimbs, output.data(), &err);
    checkCl(err, "Failed to create output buffer");

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &aBuffer);
    checkCl(err, "Failed to set arg 0");
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &outBuffer);
    checkCl(err, "Failed to set arg 1");
    cl_uint limbs = static_cast<cl_uint>(kLimbs);
    err = clSetKernelArg(kernel, 2, sizeof(cl_uint), &limbs);
    checkCl(err, "Failed to set limbs arg");

    size_t globalWorkSize = 1;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize, nullptr, 0, nullptr, nullptr);
    checkCl(err, "Failed to enqueue unary bitwise kernel");

    err = clFinish(queue);
    checkCl(err, "Failed to finish queue");

    err = clEnqueueReadBuffer(queue, outBuffer, CL_TRUE, 0, sizeof(cl_uint) * kLimbs, output.data(), 0, nullptr, nullptr);
    checkCl(err, "Failed to read unary bitwise output");

    bool ok = sameLimbs(output, expected);

    clReleaseMemObject(aBuffer);
    clReleaseMemObject(outBuffer);
    clReleaseKernel(kernel);
    return ok;
}

bool runUnaryPredicate(cl_context ctx, cl_command_queue queue, cl_program program,
                       const char *kernelName, const LimbVec &lhs, int expected) {
    cl_int err = CL_SUCCESS;
    cl_kernel kernel = clCreateKernel(program, kernelName, &err);
    checkCl(err, kernelName);

    cl_int outValue = -7;
    cl_mem aBuffer = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(cl_uint) * kLimbs, const_cast<cl_uint *>(lhs.data()), &err);
    checkCl(err, "Failed to create lhs buffer");
    cl_mem outBuffer = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                      sizeof(cl_int), &outValue, &err);
    checkCl(err, "Failed to create predicate output buffer");

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &aBuffer);
    checkCl(err, "Failed to set arg 0");
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &outBuffer);
    checkCl(err, "Failed to set arg 1");
    cl_uint limbs = static_cast<cl_uint>(kLimbs);
    err = clSetKernelArg(kernel, 2, sizeof(cl_uint), &limbs);
    checkCl(err, "Failed to set limbs arg");

    size_t globalWorkSize = 1;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize, nullptr, 0, nullptr, nullptr);
    checkCl(err, "Failed to enqueue predicate kernel");

    err = clFinish(queue);
    checkCl(err, "Failed to finish queue");

    err = clEnqueueReadBuffer(queue, outBuffer, CL_TRUE, 0, sizeof(cl_int), &outValue, 0, nullptr, nullptr);
    checkCl(err, "Failed to read predicate output");

    bool ok = (outValue == expected);

    clReleaseMemObject(aBuffer);
    clReleaseMemObject(outBuffer);
    clReleaseKernel(kernel);
    return ok;
}

bool runBinaryPredicate(cl_context ctx, cl_command_queue queue, cl_program program,
                        const char *kernelName, const LimbVec &lhs, const LimbVec &rhs,
                        int expected) {
    cl_int err = CL_SUCCESS;
    cl_kernel kernel = clCreateKernel(program, kernelName, &err);
    checkCl(err, kernelName);

    cl_int outValue = -7;
    cl_mem aBuffer = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(cl_uint) * kLimbs, const_cast<cl_uint *>(lhs.data()), &err);
    checkCl(err, "Failed to create lhs buffer");
    cl_mem bBuffer = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    sizeof(cl_uint) * kLimbs, const_cast<cl_uint *>(rhs.data()), &err);
    checkCl(err, "Failed to create rhs buffer");
    cl_mem outBuffer = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                                      sizeof(cl_int), &outValue, &err);
    checkCl(err, "Failed to create predicate output buffer");

    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &aBuffer);
    checkCl(err, "Failed to set arg 0");
    err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &bBuffer);
    checkCl(err, "Failed to set arg 1");
    err = clSetKernelArg(kernel, 2, sizeof(cl_mem), &outBuffer);
    checkCl(err, "Failed to set arg 2");
    cl_uint limbs = static_cast<cl_uint>(kLimbs);
    err = clSetKernelArg(kernel, 3, sizeof(cl_uint), &limbs);
    checkCl(err, "Failed to set limbs arg");

    size_t globalWorkSize = 1;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize, nullptr, 0, nullptr, nullptr);
    checkCl(err, "Failed to enqueue binary predicate kernel");

    err = clFinish(queue);
    checkCl(err, "Failed to finish queue");

    err = clEnqueueReadBuffer(queue, outBuffer, CL_TRUE, 0, sizeof(cl_int), &outValue, 0, nullptr, nullptr);
    checkCl(err, "Failed to read binary predicate output");

    bool ok = (outValue == expected);

    clReleaseMemObject(aBuffer);
    clReleaseMemObject(bBuffer);
    clReleaseMemObject(outBuffer);
    clReleaseKernel(kernel);
    return ok;
}

std::string loadKernelSource() {
    std::string source = cgbn::opencl::load_kernel_file("bench/base.cl");
    if (!source.empty()) {
        return source;
    }
    return std::string();

    return R"CLC(
__kernel void cgbn_set_ui32(__global uint *out, uint value, uint limbs) {
    for (uint i = 0; i < limbs; ++i) {
        out[i] = 0u;
    }
    if (limbs > 0u) {
        out[0] = value;
    }
}

__kernel void cgbn_set_one(__global uint *out, uint limbs) {
    for (uint i = 0; i < limbs; ++i) {
        out[i] = 0u;
    }
    if (limbs > 0u) {
        out[0] = 1u;
    }
}

__kernel void cgbn_bitwise_and(__global const uint *a, __global const uint *b, __global uint *out, uint limbs) {
    for (uint i = 0; i < limbs; ++i) {
        out[i] = a[i] & b[i];
    }
}

__kernel void cgbn_bitwise_ior(__global const uint *a, __global const uint *b, __global uint *out, uint limbs) {
    for (uint i = 0; i < limbs; ++i) {
        out[i] = a[i] | b[i];
    }
}

__kernel void cgbn_bitwise_xor(__global const uint *a, __global const uint *b, __global uint *out, uint limbs) {
    for (uint i = 0; i < limbs; ++i) {
        out[i] = a[i] ^ b[i];
    }
}

__kernel void cgbn_bitwise_complement(__global const uint *a, __global uint *out, uint limbs) {
    for (uint i = 0; i < limbs; ++i) {
        out[i] = ~a[i];
    }
}

__kernel void cgbn_add_ui32(__global const uint *a, __global uint *out, uint value, uint limbs) {
    ulong carry = (ulong)value;
    for (uint i = 0; i < limbs; ++i) {
        ulong sum = (ulong)a[i] + carry;
        out[i] = (uint)sum;
        carry = sum >> 32;
    }
}

__kernel void cgbn_sub_ui32(__global const uint *a, __global uint *out, uint value, uint limbs) {
    ulong borrow = (ulong)value;
    for (uint i = 0; i < limbs; ++i) {
        ulong minuend = (ulong)a[i];
        ulong wide = minuend - borrow;
        out[i] = (uint)wide;
        borrow = minuend < borrow ? 1u : 0u;
    }
}

__kernel void cgbn_is_zero(__global const uint *a, __global int *out, uint limbs) {
    int allZero = 1;
    for (uint i = 0; i < limbs; ++i) {
        if (a[i] != 0u) {
            allZero = 0;
            break;
        }
    }
    out[0] = allZero;
}

__kernel void cgbn_equals(__global const uint *a, __global const uint *b, __global int *out, uint limbs) {
    int eq = 1;
    for (uint i = 0; i < limbs; ++i) {
        if (a[i] != b[i]) {
            eq = 0;
            break;
        }
    }
    out[0] = eq;
}

__kernel void cgbn_compare(__global const uint *a, __global const uint *b, __global int *out, uint limbs) {
    int cmp = 0;
    for (int i = (int)limbs - 1; i >= 0; --i) {
        uint av = a[(uint)i];
        uint bv = b[(uint)i];
        if (av > bv) {
            cmp = 1;
            break;
        }
        if (av < bv) {
            cmp = -1;
            break;
        }
    }
    out[0] = cmp;
}

__kernel void cgbn_shift_left(__global const uint *a, __global uint *out, uint shift, uint limbs) {
    uint limbShift = shift / 32u;
    uint bitShift = shift % 32u;
    for (uint i = 0; i < limbs; ++i) {
        if (i < limbShift) {
            out[i] = 0u;
            continue;
        }
        uint src = i - limbShift;
        uint lo = a[src] << bitShift;
        uint hi = 0u;
        if (bitShift != 0u && src > 0u) {
            hi = a[src - 1u] >> (32u - bitShift);
        }
        out[i] = lo | hi;
    }
}

__kernel void cgbn_shift_right(__global const uint *a, __global uint *out, uint shift, uint limbs) {
    uint limbShift = shift / 32u;
    uint bitShift = shift % 32u;
    for (uint i = 0; i < limbs; ++i) {
        uint src = i + limbShift;
        if (src >= limbs) {
            out[i] = 0u;
            continue;
        }
        uint lo = a[src] >> bitShift;
        uint hi = 0u;
        if (bitShift != 0u && (src + 1u) < limbs) {
            hi = a[src + 1u] << (32u - bitShift);
        }
        out[i] = lo | hi;
    }
}
)CLC";
}

} // namespace

bool runOpenClLowRiskOperatorTests() {
    std::cout << "\n[OpenCL low-risk tests] starting" << std::endl;

    cgbn::opencl::context_t ctx;
    cl_int err = cgbn::opencl::create_context(ctx);
    if (err != CL_SUCCESS) {
        std::cerr << "Failed to create OpenCL context: " << err << std::endl;
        return false;
    }

    std::string source = loadKernelSource();
    cl_int buildErr = CL_SUCCESS;
    cl_program program = cgbn::opencl::build_program_from_source(ctx, source.c_str(), "", buildErr);
    if (program == nullptr || buildErr != CL_SUCCESS) {
        std::cerr << "Failed to build OpenCL kernel program: " << buildErr << std::endl;
        cgbn::opencl::destroy_context(ctx);
        return false;
    }

    const LimbVec lhs = {0xFFFF0000u, 0x12345678u, 0xAAAAAAAAu, 0x00000001u};
    const LimbVec rhs = {0x0000FFFFu, 0x87654321u, 0x55555555u, 0x00000002u};

    bool ok = true;
    ok = runUnarySetUi32(ctx.ctx, ctx.queue, program, 0xDEADBEEFu) && ok;
    ok = runSetOne(ctx.ctx, ctx.queue, program) && ok;
    ok = runBinaryBitwise(ctx.ctx, ctx.queue, program, "cgbn_bitwise_and", lhs, rhs, cpuBitwiseAnd(lhs, rhs)) && ok;
    ok = runBinaryBitwise(ctx.ctx, ctx.queue, program, "cgbn_bitwise_ior", lhs, rhs, cpuBitwiseIor(lhs, rhs)) && ok;
    ok = runBinaryBitwise(ctx.ctx, ctx.queue, program, "cgbn_bitwise_xor", lhs, rhs, cpuBitwiseXor(lhs, rhs)) && ok;
    ok = runUnaryBitwise(ctx.ctx, ctx.queue, program, "cgbn_bitwise_complement", lhs, cpuBitwiseNot(lhs)) && ok;
    ok = runUnaryAddSub(ctx.ctx, ctx.queue, program, "cgbn_add_ui32", lhs, 0x11111111u, cpuAddUi32(lhs, 0x11111111u)) && ok;
    ok = runUnaryAddSub(ctx.ctx, ctx.queue, program, "cgbn_sub_ui32", lhs, 0x11111111u, cpuSubUi32(lhs, 0x11111111u)) && ok;
    ok = runUnaryPredicate(ctx.ctx, ctx.queue, program, "cgbn_is_zero", LimbVec(kLimbs, 0u), 1) && ok;
    ok = runUnaryPredicate(ctx.ctx, ctx.queue, program, "cgbn_is_zero", lhs, cpuIsZero(lhs)) && ok;
    ok = runBinaryPredicate(ctx.ctx, ctx.queue, program, "cgbn_equals", lhs, lhs, 1) && ok;
    ok = runBinaryPredicate(ctx.ctx, ctx.queue, program, "cgbn_equals", lhs, rhs, cpuEquals(lhs, rhs)) && ok;
    ok = runBinaryPredicate(ctx.ctx, ctx.queue, program, "cgbn_compare", lhs, rhs, cpuCompare(lhs, rhs)) && ok;
    ok = runBinaryPredicate(ctx.ctx, ctx.queue, program, "cgbn_compare", rhs, lhs, cpuCompare(rhs, lhs)) && ok;
    ok = runBinaryPredicate(ctx.ctx, ctx.queue, program, "cgbn_compare", lhs, lhs, cpuCompare(lhs, lhs)) && ok;
    ok = runUnaryShift(ctx.ctx, ctx.queue, program, "cgbn_shift_left", lhs, 1u, cpuShiftLeft(lhs, 1u)) && ok;
    ok = runUnaryShift(ctx.ctx, ctx.queue, program, "cgbn_shift_left", lhs, 37u, cpuShiftLeft(lhs, 37u)) && ok;
    ok = runUnaryShift(ctx.ctx, ctx.queue, program, "cgbn_shift_right", lhs, 1u, cpuShiftRight(lhs, 1u)) && ok;
    ok = runUnaryShift(ctx.ctx, ctx.queue, program, "cgbn_shift_right", lhs, 37u, cpuShiftRight(lhs, 37u)) && ok;

    clReleaseProgram(program);
    cgbn::opencl::destroy_context(ctx);

    if (ok) {
        std::cout << "[OpenCL low-risk tests] all tests passed" << std::endl;
    } else {
        std::cerr << "[OpenCL low-risk tests] at least one test failed" << std::endl;
    }

    return ok;
}
