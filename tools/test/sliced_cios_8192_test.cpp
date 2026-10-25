#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <gmp.h>
#include <CL/cl.h>

static void gmp_to_limbs(const mpz_t v, uint32_t *out, int n) {
    mpz_t mod; mpz_init(mod); mpz_ui_pow_ui(mod, 2, 32*n);
    mpz_t tmp; mpz_init(tmp); mpz_mod(tmp, v, mod);
    size_t cnt = 0; mpz_export(out, &cnt, -1, sizeof(uint32_t), 0, 0, tmp);
    for (size_t i = cnt; i < (size_t)n; i++) out[i] = 0;
    mpz_clear(tmp); mpz_clear(mod);
}
static uint32_t inv_np0(const mpz_t N) {
    uint32_t x = (uint32_t)mpz_get_ui(N); uint32_t inv = x;
    for (int i = 0; i < 5; i++) inv = inv * (2u - x * inv);
    return (uint32_t)(-inv);
}
static std::string load_file(const char *path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "Failed to open %s\n", path); return {}; }
    std::ostringstream ss; ss << f.rdbuf(); return ss.str();
}

int main(int argc, char **argv) {
    int device_index = 1, iterations = 1, prime_bits = 991;
    for (int i = 1; i < argc; i++) {
        if (!std::strcmp(argv[i], "-d") && i+1 < argc) device_index = std::atoi(argv[++i]);
        if (!std::strcmp(argv[i], "-n") && i+1 < argc) iterations = std::atoi(argv[++i]);
        if (!std::strcmp(argv[i], "-p") && i+1 < argc) prime_bits = std::atoi(argv[++i]);
    }

    mpz_t Ng, Ag, Bg, Rg;
    mpz_init(Ng); mpz_ui_pow_ui(Ng, 2, prime_bits); mpz_sub_ui(Ng, Ng, 1);
    mpz_init(Ag); mpz_init(Bg); mpz_init(Rg);
    gmp_randstate_t rng; gmp_randinit_default(rng); gmp_randseed_ui(rng, 42);
    mpz_urandomm(Ag, rng, Ng); mpz_urandomm(Bg, rng, Ng); gmp_randclear(rng);

    int al = (prime_bits + 31) / 32;
    int total = (al + 31) / 8 * 8; // round up to 8*N
    std::vector<uint32_t> nL(al), aL(al), bL(al);
    gmp_to_limbs(Ng, nL.data(), al); gmp_to_limbs(Ag, aL.data(), al); gmp_to_limbs(Bg, bL.data(), al);
    uint32_t np0v = inv_np0(Ng);

    mpz_t Rv, Ri; mpz_init(Rv); mpz_ui_pow_ui(Rv, 2, 32*al); mpz_init(Ri); mpz_invert(Ri, Rv, Ng);
    mpz_mul(Ag, Ag, Rv); mpz_mod(Ag, Ag, Ng); mpz_mul(Bg, Bg, Rv); mpz_mod(Bg, Bg, Ng);
    gmp_to_limbs(Ag, aL.data(), al); gmp_to_limbs(Bg, bL.data(), al);

    std::vector<uint32_t> ref(al);
    mpz_set(Rg, Ag);
    for (int k = 0; k < iterations; k++) { mpz_mul(Rg, Rg, Bg); mpz_mul(Rg, Rg, Ri); mpz_mod(Rg, Rg, Ng); }
    gmp_to_limbs(Rg, ref.data(), al);

    // Pad to 256 limbs (32 lanes × 8)
    std::vector<uint32_t> Ap(256,0), Bp(256,0), Np(256,0);
    memcpy(Ap.data(), aL.data(), al*4); memcpy(Bp.data(), bL.data(), al*4); memcpy(Np.data(), nL.data(), al*4);

    cl_int e; cl_uint np;
    clGetPlatformIDs(0, nullptr, &np);
    std::vector<cl_platform_id> ps(np); clGetPlatformIDs(np, ps.data(), nullptr);
    cl_device_id dv = nullptr;
    { int dc = 0;
      for (uint32_t p = 0; p < np && !dv; p++) {
        cl_uint nd; clGetDeviceIDs(ps[p], CL_DEVICE_TYPE_GPU, 0, nullptr, &nd);
        if (!nd) continue;
        std::vector<cl_device_id> ds(nd); clGetDeviceIDs(ps[p], CL_DEVICE_TYPE_GPU, nd, ds.data(), nullptr);
        for (uint32_t d = 0; d < nd; d++) { if (dc == device_index) { dv = ds[d]; break; } dc++; }
      }
    }
    if (!dv) { std::fprintf(stderr, "GPU device %d not found\n", device_index); return 1; }

    cl_context cx = clCreateContext(nullptr, 1, &dv, nullptr, nullptr, &e);
    cl_command_queue q = clCreateCommandQueue(cx, dv, 0, &e);
    std::string src = load_file("kernels/opencl/bench/sliced_cios_8192.cl");
    if (src.empty()) return 1;
    const char *cs = src.c_str(); size_t sl = src.size();
    cl_program pg = clCreateProgramWithSource(cx, 1, &cs, &sl, &e);
    e = clBuildProgram(pg, 1, &dv, "-cl-std=CL2.0", nullptr, nullptr);
    if (e != CL_SUCCESS) {
        size_t lz; clGetProgramBuildInfo(pg, dv, CL_PROGRAM_BUILD_LOG, 0, nullptr, &lz);
        std::vector<char> log(lz+1); clGetProgramBuildInfo(pg, dv, CL_PROGRAM_BUILD_LOG, lz, log.data(), nullptr);
        std::fprintf(stderr, "Build error:\n%s\n", log.data()); return 1;
    }
    cl_kernel k = clCreateKernel(pg, "sliced_cios_mul_8192", &e);
    if (!k) { std::fprintf(stderr, "kernel not found\n"); return 1; }

    cl_mem dA=clCreateBuffer(cx,CL_MEM_READ_WRITE|CL_MEM_COPY_HOST_PTR,1024,Ap.data(),&e);
    cl_mem dB=clCreateBuffer(cx,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,1024,Bp.data(),&e);
    cl_mem dN=clCreateBuffer(cx,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,1024,Np.data(),&e);
    cl_mem dR=clCreateBuffer(cx,CL_MEM_READ_WRITE,1024,nullptr,&e);
    clSetKernelArg(k,0,sizeof(cl_mem),&dA);clSetKernelArg(k,1,sizeof(cl_mem),&dB);
    clSetKernelArg(k,2,sizeof(cl_mem),&dN);clSetKernelArg(k,3,sizeof(cl_mem),&dR);
    clSetKernelArg(k,4,sizeof(cl_uint),&np0v);

    size_t gl=32, lc=32; uint32_t ca[256]; memcpy(ca,Ap.data(),sizeof(ca));

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
        clEnqueueWriteBuffer(q, dA, CL_TRUE, 0, sizeof(ca), ca, 0, nullptr, nullptr);
        clEnqueueNDRangeKernel(q, k, 1, nullptr, &gl, &lc, 0, nullptr, nullptr);
        clFinish(q);
        if (i < iterations-1) clEnqueueReadBuffer(q, dR, CL_TRUE, 0, sizeof(ca), ca, 0, nullptr, nullptr);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    uint32_t gr[256]; clEnqueueReadBuffer(q, dR, CL_TRUE, 0, sizeof(gr), gr, 0, nullptr, nullptr);

    int mm = 0;
    for (int i = 0; i < al; i++) {
        if (gr[i] != ref[i]) {
            if (mm < 8) std::printf("  limb[%d]: GPU=0x%08x GMP=0x%08x\n", i, gr[i], ref[i]);
            mm++;
        }
    }
    std::printf("sliced_cios_8192: %s (%d/%d match), %d iters, %d-bit, GPU=%.1fms (%.0f/s)\n",
        mm==0?"PASS":"FAIL", al-mm, al, iterations, prime_bits, ms, (double)iterations/ms*1000.0);

    clReleaseMemObject(dA);clReleaseMemObject(dB);clReleaseMemObject(dN);clReleaseMemObject(dR);
    clReleaseKernel(k);clReleaseProgram(pg);clReleaseCommandQueue(q);clReleaseContext(cx);
    mpz_clear(Ng);mpz_clear(Ag);mpz_clear(Bg);mpz_clear(Rg);mpz_clear(Rv);mpz_clear(Ri);
    return mm?1:0;
}
