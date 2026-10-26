#include "selftest.h"

#include "../../../../include/ecm/platform/logging.h"
#include "../../stage1/montgomery/opencl/montgomery.h"

#include <CL/cl.h>

#include <cstdio>
#include <string>

int opencl_ecm_selftest_montgomery(const mpz_t N, uint32_t bits) {
    const uint32_t limbs = bits / 32;
    uint32_t buf[OPENCL_ECM_MAX_LIMBS] = {0};
    mpz_t two, mont_mpz, back;
    mpz_init(two);
    mpz_init(mont_mpz);
    mpz_init(back);
    mpz_set_ui(two, 2);
    ecm_to_montgomery(buf, two, N, bits, limbs);
    ecm_to_mpz(mont_mpz, buf, limbs);
    ecm_from_montgomery(back, mont_mpz, N, ecm_find_np0(N), limbs);
    int ok = (mpz_cmp(back, two) == 0);
    if (!ok) {
        ecm_ts_fprintf(stderr, "GPU: Montgomery self-test failed (2 -> mont -> 2)\n");
    }
    mpz_clear(two);
    mpz_clear(mont_mpz);
    mpz_clear(back);
    return ok ? 0 : -1;
}

int opencl_ecm_selftest_mont_mul(cgbn::opencl::context_t &ctx, const mpz_t N, uint32_t bits,
                                 uint32_t np0) {
    const uint32_t limbs = bits / 32;
    std::string mont_src =
        cgbn::opencl::load_kernel_file("bench/mont.cl");
    if (mont_src.empty()) {
        ecm_ts_fprintf(stderr,
                       "GPU: mont.cl not found (add mont.cl to APK assets for self-test)\n");
        return -1;
    }
    char opts[64];
    snprintf(opts, sizeof(opts), "-DMAX_LIMBS=%u", limbs);
    cl_int buildErr = CL_SUCCESS;
    cl_program prog =
        cgbn::opencl::build_program_from_source(ctx, mont_src.c_str(), opts, buildErr);
    if (prog == nullptr || buildErr != CL_SUCCESS) {
        ecm_ts_fprintf(stderr, "GPU: mont.cl build failed (err=%d)\n", (int)buildErr);
        return -1;
    }
    cl_int err;
    cl_kernel kMul = clCreateKernel(prog, "cgbn_mont_mul", &err);
    if (err != CL_SUCCESS) {
        clReleaseProgram(prog);
        return -1;
    }

    uint32_t a[OPENCL_ECM_MAX_LIMBS] = {0}, b[OPENCL_ECM_MAX_LIMBS] = {0},
             n[OPENCL_ECM_MAX_LIMBS] = {0}, out[OPENCL_ECM_MAX_LIMBS] = {0};
    mpz_t two, three, six, r;
    mpz_init(two);
    mpz_init(three);
    mpz_init(six);
    mpz_init(r);
    mpz_set_ui(two, 2);
    mpz_set_ui(three, 3);
    mpz_mul_ui(six, two, 3);
    ecm_to_montgomery(a, two, N, bits, limbs);
    ecm_to_montgomery(b, three, N, bits, limbs);
    ecm_from_mpz(N, n, limbs);

    cl_mem bufA = clCreateBuffer(ctx.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 limbs * sizeof(uint32_t), a, &err);
    cl_mem bufB = clCreateBuffer(ctx.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 limbs * sizeof(uint32_t), b, &err);
    cl_mem bufN = clCreateBuffer(ctx.ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                 limbs * sizeof(uint32_t), n, &err);
    cl_mem bufOut = clCreateBuffer(ctx.ctx, CL_MEM_WRITE_ONLY, limbs * sizeof(uint32_t),
                                   nullptr, &err);
    clSetKernelArg(kMul, 0, sizeof(cl_mem), &bufA);
    clSetKernelArg(kMul, 1, sizeof(cl_mem), &bufB);
    clSetKernelArg(kMul, 2, sizeof(cl_mem), &bufN);
    clSetKernelArg(kMul, 3, sizeof(cl_mem), &bufOut);
    clSetKernelArg(kMul, 4, sizeof(cl_uint), &np0);
    cl_uint limbs_arg = limbs;
    clSetKernelArg(kMul, 5, sizeof(cl_uint), &limbs_arg);
    size_t g = 1;
    err = clEnqueueNDRangeKernel(ctx.queue, kMul, 1, nullptr, &g, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        ecm_ts_fprintf(stderr, "GPU: mont_mul self-test enqueue failed (err=%d)\n", err);
        clReleaseMemObject(bufA);
        clReleaseMemObject(bufB);
        clReleaseMemObject(bufN);
        clReleaseMemObject(bufOut);
        clReleaseKernel(kMul);
        clReleaseProgram(prog);
        mpz_clear(two);
        mpz_clear(three);
        mpz_clear(six);
        mpz_clear(r);
        return -1;
    }
    err = clFinish(ctx.queue);
    if (err != CL_SUCCESS) {
        ecm_ts_fprintf(stderr, "GPU: mont_mul self-test clFinish failed (err=%d)\n", err);
    }
    err = clEnqueueReadBuffer(ctx.queue, bufOut, CL_TRUE, 0, limbs * sizeof(uint32_t), out, 0,
                              nullptr, nullptr);
    if (err != CL_SUCCESS) {
        ecm_ts_fprintf(stderr, "GPU: mont_mul self-test readback failed (err=%d)\n", err);
        clReleaseMemObject(bufA);
        clReleaseMemObject(bufB);
        clReleaseMemObject(bufN);
        clReleaseMemObject(bufOut);
        clReleaseKernel(kMul);
        clReleaseProgram(prog);
        mpz_clear(two);
        mpz_clear(three);
        mpz_clear(six);
        mpz_clear(r);
        return -1;
    }
    ecm_to_mpz(r, out, limbs);
    ecm_from_montgomery(r, r, N, np0, limbs);
    int ok = (mpz_cmp(r, six) == 0);
    if (!ok) {
        ecm_ts_fprintf(stderr, "GPU: mont_mul self-test failed (2*3 mod N)\n");
    }
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufN);
    clReleaseMemObject(bufOut);
    clReleaseKernel(kMul);
    clReleaseProgram(prog);
    mpz_clear(two);
    mpz_clear(three);
    mpz_clear(six);
    mpz_clear(r);
    return ok ? 0 : -1;
}
