// cgbn_op_probe.cu -- per-operator cost of the CGBN ops our ECM stage-1 kernel uses per bit,
// plus an A/B of CGBN's multiply chains (XMP_IMAD / XMP_XMAD / XMP_WMAD).  CGBN picks WMAD
// automatically for __CUDA_ARCH__ >= 700 (Volta), and that choice has never been re-tuned for
// Ada / Blackwell, so forcing the other two variants is a cheap experiment.
//
// Measured ops:
//   mont_mul      a = a*b            (dependency chain)
//   mont_sqr      a = a*a            (CGBN implements this as mont_mul(a,a) -- alpha == 1.0 today)
//   norm          cgbn_add + (compare >= 0 ? sub : -)   == our normalize_addition
//   add_norm      cgbn_add + normalize
//   cmp           cgbn_compare only
//   sub_cond      cgbn_sub with conditional add-back (the differential pattern in our kernel)
//   shift_left    cgbn_shift_left(a, 1)                  (param3's v <<= 1)
//
// GMP is required by CGBN's host headers and MUST be included before cgbn.h.
//
// NOTE: keep this file ASCII-only.  nvcc reads a BOM-less UTF-8 file as ANSI (GBK on this box),
// and a trailing multi-byte CJK character can swallow the newline, which comments out the NEXT
// line of code.  That cost an hour once -- do not reintroduce non-ASCII here.
//
// usage: cgbn_op_probe.exe [tier 0..5] [instances] [iterations] [device]
//   tier 0..5 = (4,512) (8,1024) (8,2048) (16,3072) (16,4096) (16,8192)

#include <gmp.h>          // must precede cgbn.h (CGBN host side requires GMP)
#include <cgbn/cgbn.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

#define CUDA_CHECK(call)                                                            \
  do {                                                                              \
    cudaError_t e_ = (call);                                                        \
    if (e_ != cudaSuccess) {                                                        \
      fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
      exit(2);                                                                      \
    }                                                                               \
  } while (0)

// Per-tier params, mirroring kernels/cuda/cgbn_stage1_kernel.h::cgbn_params_t.
template <uint32_t tpi, uint32_t bits>
struct params_t {
  static const uint32_t TPB = 128;
  static const uint32_t MAX_ROTATION = 4;
  static const uint32_t SHM_LIMIT = 0;
  static const bool CONSTANT_TIME = false;
  static const uint32_t TPI = tpi;
  static const uint32_t BITS = bits;
};

enum { OP_MONT_MUL, OP_MONT_SQR, OP_NORM, OP_ADD_NORM, OP_CMP, OP_SUB_COND, OP_SHIFT, OP_COUNT };

template <class params, int OP>
__global__ void k_probe(cgbn_error_report_t *report,
                        cgbn_mem_t<params::BITS> *data,   // 3 slots per instance: a, b, n
                        uint32_t np0, int iters,
                        uint64_t *sink) {
  typedef cgbn_context_t<params::TPI, params> context_t;
  typedef cgbn_env_t<context_t, params::BITS> env_t;
  typedef typename env_t::cgbn_t bn_t;

  context_t ctx(cgbn_no_checks, report, (uint32_t)blockIdx.x);
  env_t env(ctx);

  const int32_t ipb = blockDim.x / params::TPI;
  const int32_t instance = blockIdx.x * ipb + (threadIdx.x / params::TPI);

  cgbn_mem_t<params::BITS> *slot = &data[3 * instance];      // cgbn_load wants a non-const pointer
  cgbn_mem_t<params::BITS> *slot_n = &data[3 * instance + 2];

  bn_t a, b, n, t;
  cgbn_load(env, a, &slot[0]);
  cgbn_load(env, b, &slot[1]);
  cgbn_load(env, n, slot_n);

  uint64_t acc = 0;
  if (OP == OP_MONT_MUL) {
    for (int i = 0; i < iters; i++) env.mont_mul(a, a, b, n, np0);
  } else if (OP == OP_MONT_SQR) {
    for (int i = 0; i < iters; i++) env.mont_sqr(a, a, n, np0);
  } else if (OP == OP_NORM) {
    for (int i = 0; i < iters; i++) {
      cgbn_add(env, t, a, n);
      if (cgbn_compare(env, t, n) >= 0) cgbn_sub(env, t, t, n);
    }
    a = t;
  } else if (OP == OP_ADD_NORM) {
    for (int i = 0; i < iters; i++) {
      cgbn_add(env, a, a, b);
      if (cgbn_compare(env, a, n) >= 0) cgbn_sub(env, a, a, n);
    }
  } else if (OP == OP_CMP) {
    for (int i = 0; i < iters; i++) acc += (uint64_t)(cgbn_compare(env, a, n) >= 0);
  } else if (OP == OP_SUB_COND) {
    for (int i = 0; i < iters; i++) {
      if (cgbn_sub(env, a, a, b)) cgbn_add(env, a, a, n);
    }
  } else if (OP == OP_SHIFT) {
    for (int i = 0; i < iters; i++) cgbn_shift_left(env, a, a, 1);
  }

  acc += cgbn_get_ui32(env, a);
  if (acc == 0xdeadbeefull) sink[instance] = acc;   // defeat DCE
}

template <class params>
static void launch(int op, int blocks, int iters, cgbn_error_report_t *report,
                   cgbn_mem_t<params::BITS> *dev, uint32_t np0, uint64_t *sink) {
  switch (op) {
    case OP_MONT_MUL: k_probe<params, OP_MONT_MUL><<<blocks, params::TPB>>>(report, dev, np0, iters, sink); break;
    case OP_MONT_SQR: k_probe<params, OP_MONT_SQR><<<blocks, params::TPB>>>(report, dev, np0, iters, sink); break;
    case OP_NORM:     k_probe<params, OP_NORM><<<blocks, params::TPB>>>(report, dev, np0, iters, sink); break;
    case OP_ADD_NORM: k_probe<params, OP_ADD_NORM><<<blocks, params::TPB>>>(report, dev, np0, iters, sink); break;
    case OP_CMP:      k_probe<params, OP_CMP><<<blocks, params::TPB>>>(report, dev, np0, iters, sink); break;
    case OP_SUB_COND: k_probe<params, OP_SUB_COND><<<blocks, params::TPB>>>(report, dev, np0, iters, sink); break;
    case OP_SHIFT:    k_probe<params, OP_SHIFT><<<blocks, params::TPB>>>(report, dev, np0, iters, sink); break;
  }
}

static uint32_t np0_for(uint32_t n0) {
  uint32_t inv = 1;
  for (int i = 0; i < 5; i++) inv *= 2u - n0 * inv;   // n0^{-1} mod 2^32 (Newton)
  return (uint32_t)(0u - inv);
}

template <class params>
static void run_tier(int instances, int iters, const char *tag) {
  typedef cgbn_mem_t<params::BITS> mem_t;
  const int32_t ipb = params::TPB / params::TPI;
  const int32_t blocks = (instances + ipb - 1) / ipb;
  const int32_t total = blocks * ipb;

  mem_t *host = (mem_t *)malloc((size_t)total * 3 * sizeof(mem_t));
  memset(host, 0, (size_t)total * 3 * sizeof(mem_t));

  const int limbs_per_mem = params::BITS / 32;
  for (int i = 0; i < total; i++) {
    uint32_t *n = host[3 * i + 2]._limbs;
    for (int l = 0; l < limbs_per_mem; l++) n[l] = 0xFFFFFFFFu;
    n[0] |= 1u;                                     // odd modulus
    uint32_t *a = host[3 * i + 0]._limbs;
    uint32_t *b = host[3 * i + 1]._limbs;
    for (int l = 0; l < limbs_per_mem; l++) {
      a[l] = 0x9E3779B9u ^ (uint32_t)(l * 2654435761u + i * 40503u);
      b[l] = 0x85EBCA6Bu ^ (uint32_t)(l * 2246822519u + i * 668265263u);
    }
    a[0] |= 1u;
    b[0] |= 1u;
    a[limbs_per_mem - 1] &= 0x7FFFFFFFu;            // keep a, b < n
    b[limbs_per_mem - 1] &= 0x7FFFFFFFu;
  }
  const uint32_t np0 = np0_for(host[2]._limbs[0]);

  mem_t *dev = nullptr;
  uint64_t *sink = nullptr;
  cgbn_error_report_t *report = nullptr;
  CUDA_CHECK(cudaMalloc(&dev, (size_t)total * 3 * sizeof(mem_t)));
  CUDA_CHECK(cudaMalloc(&sink, (size_t)total * sizeof(uint64_t)));
  CUDA_CHECK(cudaMemcpy(dev, host, (size_t)total * 3 * sizeof(mem_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(sink, 0, (size_t)total * sizeof(uint64_t)));
  CUDA_CHECK(cgbn_error_report_alloc(&report));

  cudaEvent_t e0, e1;
  CUDA_CHECK(cudaEventCreate(&e0));
  CUDA_CHECK(cudaEventCreate(&e1));

  const char *names[OP_COUNT] = {"mont_mul", "mont_sqr", "norm(cmp+sub)", "add+norm",
                                 "cmp", "sub+condadd", "shift_left"};
  for (int op = 0; op < OP_COUNT; op++) {
    launch<params>(op, blocks, 8, report, dev, np0, sink);    // warmup
    CUDA_CHECK(cudaDeviceSynchronize());
    if (cgbn_error_report_check(report)) {
      fprintf(stderr, "CGBN error (op=%d)\n", op);
      cgbn_error_report_reset(report);
    }
    CUDA_CHECK(cudaEventRecord(e0));
    launch<params>(op, blocks, iters, report, dev, np0, sink);
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    const double ns_per_op = (double)ms * 1e6 / ((double)total * (double)iters);
    printf("  %-16s %8.2f ns/op   %9.2f Mops/s\n", names[op], ns_per_op, 1e3 / ns_per_op);
  }
  printf("  [%s, instances=%d]\n", tag, total);

  cgbn_error_report_free(report);
  cudaFree(dev);
  cudaFree(sink);
  free(host);
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
}

int main(int argc, char **argv) {
  int tier = (argc > 1) ? atoi(argv[1]) : 3;
  int instances = (argc > 2) ? atoi(argv[2]) : 4096;
  int iters = (argc > 3) ? atoi(argv[3]) : 2000;
  int device = (argc > 4) ? atoi(argv[4]) : 0;

  CUDA_CHECK(cudaSetDevice(device));
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

#if defined(XMP_IMAD)
  const char *variant = "XMP_IMAD";
#elif defined(XMP_XMAD)
  const char *variant = "XMP_XMAD";
#elif defined(XMP_WMAD)
  const char *variant = "XMP_WMAD";
#else
  const char *variant = "(arch default)";
#endif
  printf("CGBN op probe -- variant=%s, device=%d (%s), instances=%d, iters=%d\n",
         variant, device, prop.name, instances, iters);

  switch (tier) {
    case 0: run_tier<params_t<4, 512>>(instances, iters, "TPI=4 BITS=512"); break;
    case 1: run_tier<params_t<8, 1024>>(instances, iters, "TPI=8 BITS=1024"); break;
    case 2: run_tier<params_t<8, 2048>>(instances, iters, "TPI=8 BITS=2048"); break;
    case 3: run_tier<params_t<16, 3072>>(instances, iters, "TPI=16 BITS=3072"); break;
    case 4: run_tier<params_t<16, 4096>>(instances, iters, "TPI=16 BITS=4096"); break;
    case 5: run_tier<params_t<16, 8192>>(instances, iters, "TPI=16 BITS=8192"); break;
    default: fprintf(stderr, "tier 0..5\n"); return 1;
  }
  return 0;
}
