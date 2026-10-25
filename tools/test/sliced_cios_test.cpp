// Standalone Sliced CIOS verification.
// Compares sliced_cios_mul with GMP Montgomery multiplication.
// 1024-bit (32 limbs), 1 WG × 32 lanes, 1 barrier per CIOS outer iteration.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <chrono>
#include <gmp.h>
#include <CL/cl.h>

#define SLICED_LIMBS 32

static void gmp_to_limbs(const mpz_t v, uint32_t *out) {
    mpz_t mod; mpz_init(mod); mpz_ui_pow_ui(mod, 2, 32*SLICED_LIMBS);
    mpz_t tmp; mpz_init(tmp); mpz_mod(tmp, v, mod);
    size_t cnt = 0;
    mpz_export(out, &cnt, -1, sizeof(uint32_t), 0, 0, tmp);
    for (size_t i = cnt; i < SLICED_LIMBS; i++) out[i] = 0;
    mpz_clear(tmp); mpz_clear(mod);
}

static uint32_t inv_np0(int n_bits) {
    // For Mersenne primes 2^p-1: np0 = 1
    // For general: Newton iteration inverse of N[0] mod 2^32
    // We use a fixed test with (2^991-1) which has N[0] = 0xFFFFFFFF → inv = 1
    (void)n_bits;
    return 1u;
}

static std::string load_file(const char *path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "Failed to open %s\n", path); return {}; }
    std::ostringstream ss; ss << f.rdbuf();
    return ss.str();
}

int main(int argc, char **argv) {
    int device_index = 0, iterations = 1;
    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "-d") && i+1 < argc) device_index = std::atoi(argv[++i]);
        if (!std::strcmp(argv[i], "-n") && i+1 < argc) iterations = std::atoi(argv[++i]);
    }

    // ── GMP setup: Mersenne prime 2^991-1 ────────────────────────────
    mpz_t N_gmp, A_gmp, B_gmp, R_gmp;
    mpz_init(N_gmp); mpz_ui_pow_ui(N_gmp, 2, 991); mpz_sub_ui(N_gmp, N_gmp, 1);
    mpz_init(A_gmp); mpz_init(B_gmp); mpz_init(R_gmp);

    gmp_randstate_t rng; gmp_randinit_default(rng); gmp_randseed_ui(rng, 42);
    mpz_urandomm(A_gmp, rng, N_gmp);
    mpz_urandomm(B_gmp, rng, N_gmp);
    gmp_randclear(rng);

    uint32_t n_limbs[SLICED_LIMBS], a_limbs[SLICED_LIMBS], b_limbs[SLICED_LIMBS];
    gmp_to_limbs(N_gmp, n_limbs);
    gmp_to_limbs(A_gmp, a_limbs);
    gmp_to_limbs(B_gmp, b_limbs);
    uint32_t np0_val = inv_np0(991);

    // Montgomery encode
    mpz_t R_val, R_inv;
    mpz_init(R_val); mpz_ui_pow_ui(R_val, 2, 32*SLICED_LIMBS);
    mpz_init(R_inv); mpz_invert(R_inv, R_val, N_gmp);
    mpz_mul(A_gmp, A_gmp, R_val); mpz_mod(A_gmp, A_gmp, N_gmp);
    mpz_mul(B_gmp, B_gmp, R_val); mpz_mod(B_gmp, B_gmp, N_gmp);
    gmp_to_limbs(A_gmp, a_limbs);
    gmp_to_limbs(B_gmp, b_limbs);

    // ── GMP reference: N iterations of mont_mul ──────────────────────
    uint32_t ref_limbs[SLICED_LIMBS];
    mpz_set(R_gmp, A_gmp);
    auto t0_gmp = std::chrono::high_resolution_clock::now();
    for (int k = 0; k < iterations; k++) {
        mpz_mul(R_gmp, R_gmp, B_gmp);
        mpz_mul(R_gmp, R_gmp, R_inv);
        mpz_mod(R_gmp, R_gmp, N_gmp);
    }
    auto t1_gmp = std::chrono::high_resolution_clock::now();
    double gmp_ms = std::chrono::duration<double, std::milli>(t1_gmp - t0_gmp).count();
    gmp_to_limbs(R_gmp, ref_limbs);

    // ── OpenCL ───────────────────────────────────────────────────────
    cl_int err;
    cl_uint np;
    clGetPlatformIDs(0, nullptr, &np);
    std::vector<cl_platform_id> plats(np);
    clGetPlatformIDs(np, plats.data(), nullptr);
    cl_device_id dev = nullptr;
    { int dc = 0;
      for (uint32_t p = 0; p < np && !dev; p++) {
        cl_uint nd; clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_GPU, 0, nullptr, &nd);
        if (!nd) continue;
        std::vector<cl_device_id> ds(nd);
        clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_GPU, nd, ds.data(), nullptr);
        for (uint32_t d = 0; d < nd; d++) { if (dc == device_index) { dev = ds[d]; break; } dc++; }
      }
    }
    if (!dev) { std::fprintf(stderr, "GPU device %d not found\n", device_index); return 1; }

    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    cl_command_queue q = clCreateCommandQueue(ctx, dev, 0, &err);

    std::string src = load_file("kernels/opencl/bench/sliced_cios_test.cl");
    if (src.empty()) return 1;
    const char *csrc = src.c_str(); size_t srclen = src.size();
    cl_program prog = clCreateProgramWithSource(ctx, 1, &csrc, &srclen, &err);
    std::string opts = "-cl-std=CL2.0";
    err = clBuildProgram(prog, 1, &dev, opts.c_str(), nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t logsz;
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logsz);
        std::vector<char> log(logsz+1);
        clGetProgramBuildInfo(prog, dev, CL_PROGRAM_BUILD_LOG, logsz, log.data(), nullptr);
        std::fprintf(stderr, "Build error:\n%s\n", log.data());
        return 1;
    }
    cl_kernel k = clCreateKernel(prog, "sliced_cios_mul", &err);
    if (!k) { std::fprintf(stderr, "kernel not found\n"); return 1; }

    cl_mem dA = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 4*SLICED_LIMBS, a_limbs, &err);
    cl_mem dB = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 4*SLICED_LIMBS, b_limbs, &err);
    cl_mem dN = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 4*SLICED_LIMBS, n_limbs, &err);
    cl_mem dR = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 128, nullptr, &err);

    clSetKernelArg(k, 0, sizeof(cl_mem), &dA);
    clSetKernelArg(k, 1, sizeof(cl_mem), &dB);
    clSetKernelArg(k, 2, sizeof(cl_mem), &dN);
    clSetKernelArg(k, 3, sizeof(cl_mem), &dR);
    clSetKernelArg(k, 4, sizeof(cl_uint), &np0_val);

    // ── Launch N iterations, feeding result back as next input ───────
    size_t global = 32, local = 32;
    uint32_t cur_a[SLICED_LIMBS];
    std::memcpy(cur_a, a_limbs, sizeof(cur_a));

    auto t0_gpu = std::chrono::high_resolution_clock::now();
    for (int iter = 0; iter < iterations; iter++) {
        clEnqueueWriteBuffer(q, dA, CL_TRUE, 0, sizeof(cur_a), cur_a, 0, nullptr, nullptr);
        clEnqueueNDRangeKernel(q, k, 1, nullptr, &global, &local, 0, nullptr, nullptr);
        clFinish(q);
        if (iter < iterations - 1)
            clEnqueueReadBuffer(q, dR, CL_TRUE, 0, sizeof(cur_a), cur_a, 0, nullptr, nullptr);
    }
    auto t1_gpu = std::chrono::high_resolution_clock::now();
    double gpu_ms = std::chrono::duration<double, std::milli>(t1_gpu - t0_gpu).count();

    uint32_t gpu_limbs[SLICED_LIMBS];
    clEnqueueReadBuffer(q, dR, CL_TRUE, 0, sizeof(gpu_limbs), gpu_limbs, 0, nullptr, nullptr);

    // ── Compare ──────────────────────────────────────────────────────
    int mismatches = 0;
    for (uint32_t i = 0; i < SLICED_LIMBS; i++) {
        if (gpu_limbs[i] != ref_limbs[i]) {
            if (mismatches < 33)
                std::fprintf(stdout, "  limb[%u]: GPU=0x%08x GMP=0x%08x\n", i, gpu_limbs[i], ref_limbs[i]);
            mismatches++;
        }
    }
    std::fprintf(stdout, "sliced_cios: %s (%d/31 match), %d iterations, GPU=%.1fms (%.0f/s), GMP=%.1fms\n",
        mismatches == 0 ? "PASS" : "FAIL", 31 - mismatches, iterations,
        gpu_ms, (double)iterations / gpu_ms * 1000.0, gmp_ms);

    // ── Cleanup ───────────────────────────────────────────────────────
    clReleaseMemObject(dA); clReleaseMemObject(dB); clReleaseMemObject(dN); clReleaseMemObject(dR);
    clReleaseKernel(k); clReleaseProgram(prog);
    clReleaseCommandQueue(q); clReleaseContext(ctx);
    mpz_clear(N_gmp); mpz_clear(A_gmp); mpz_clear(B_gmp); mpz_clear(R_gmp);
    mpz_clear(R_val); mpz_clear(R_inv);
    return mismatches ? 1 : 0;
}
