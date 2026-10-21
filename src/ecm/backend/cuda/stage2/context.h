#pragma once
#include "../../../../core/ecm_stage2_logging.h"
#include "../../../../core/ecm_cuda_stage2.h"
#include "../../../../core/ecm_stage2_ntt_counts.h"
#include "../../../../core/ecm_stage2_d_catalog.h"
#include "../../../../core/ecm_stage2_relocation.h"
#include "../../../../core/ecm_stage2_geometry.h"
#include "../../../../core/ecm_stage2_requests.h"
#include "../../../../core/ecm_stage2_ntt_memory.h"
#include "../../../../core/ecm_stage2_giant_memory.h"
#include "../../../../core/ecm_stage2_s4_program.h"
#include "../../../../core/ecm_stage2_workspace_memory.h"
#include "../../../../core/ecm_stage2_modulus.h"
#include "../ntt/runtime.h"
#include "launch.h"
#include "stage2_scalar.cuh"
#include <utility>
#include <initializer_list>
#include <set>
#include <array>
#include <functional>
static_assert(ecm_stage2_capacity_max==ecm_stage2::max_words,"Stage2 capacity boundary");
#if !defined(NTT_GL_FIXED_MODE) || NTT_GL_FIXED_MODE != 3
#error "Production Stage2 requires fixed PTX3 Goldilocks"
#endif
#if !defined(NTT_OUTER_UNROLL_U) || NTT_OUTER_UNROLL_U != 0
#error "Production Stage2 requires the verified outer schedule"
#endif
#if !defined(NTT_GL_ADD_SUB_MASK) || NTT_GL_ADD_SUB_MASK != 1
#error "Production Stage2 requires canonical PTX subtraction"
#endif

enum { BC_FTREE = 0, BC_GTREE = 1, BC_FOLD = 2, BC_DESCENT = 3, BC_FINV = 4, BC_NCAT = 5 };

struct Stage2Params {
    unsigned long long D = 0, B1 = 0, B2 = 0;
    std::vector<unsigned long long> baby_j;
};

struct LadderCtx {
    std::vector<unsigned long long> hn, hqx, hqz, ha24, hmone;
    std::vector<unsigned long long> target_n; // host identity only, never uploaded as arithmetic modulus
    unsigned long long ninv = 0;
    size_t nw = 0;
};

using CPoly = std::vector<std::vector<unsigned long long>>;

inline constexpr int s4_tail_mont_mode() { return 0; }
inline const size_t S2G_GFINV_SEG = 16;
inline constexpr bool g_giant_seed_pair = true;
inline constexpr bool g_giant_base_cpu = false;
inline constexpr bool g_xadd6 = true;
inline const char *const kBatchedCat[BC_NCAT] = { "f_tree", "g_tree", "fold", "descent", "inv" };

inline const unsigned long long g_s4_batch_budget_mb = 64;
inline constexpr unsigned kFoldOwnerReuse=3;

bool mpz_to_words(std::vector<unsigned long long> &w, size_t W, const mpz_t v);
void words_to_mpz(mpz_t out, const unsigned long long *w, size_t W);
int ceil_log2_u64(unsigned long long m);
void set_u64_mpz(mpz_t r, unsigned long long v);

unsigned long long mulmod_u64(unsigned long long a, unsigned long long b,
                                            unsigned long long m);

unsigned long long powmod_u64(unsigned long long a, unsigned long long e,
                                     unsigned long long m);

bool is_prime_u64(unsigned long long p);
