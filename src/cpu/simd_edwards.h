/* ---------------------------------------------------------------------------
 * simd_edwards.h — batched Edwards stage-1 point layer, 8 curves in one zmm.
 *
 * Sits on top of simd_mont_ifma (lane = curve, SoA over 52-bit limbs) and
 * mirrors src/cpu/ecm_edwards_cpu.cpp's scalar formulas exactly:
 *   dbl          = 4 sqr + 4 mul
 *   add          = 9 mul (incl. d)
 *   add_affine   = 7 mul (affine inputs carry (x, y, dxy = d*x*y))
 *   result       = Qx = z + y, Qz = z - y, factor = gcd(Qz, N)
 *
 * Why the batch works at all: the ladder's w-NAF digits are digits of
 * s = 48*lcm(1..B1), which is the SAME for every curve, so at each step all
 * lanes use the same dictionary index.  Only the dictionary *contents* differ
 * per curve, hence the dictionary lives in one SoA block covering all lanes.
 *
 * Window size w is a batch-level decision, not just a speed knob: the
 * dictionary is 3 * 2^(w-2) * 8n words, i.e. 236 KB per batch at w=8/n=154
 * versus 3.8 MB at w=12, and w=12 would drag every ladder step out of L3.
 * Extra adds from the smaller window cost only ~3% of the modmuls
 * (per bit: 8 + 7/(w+1)).
 *
 * !!! AVX512-IFMA TU: compile with /arch:AVX512 and only call after CPUID.
 * Not thread safe per context (each context owns its scratch arena).
 * ------------------------------------------------------------------------- */
#ifndef SIMD_EDWARDS_H
#define SIMD_EDWARDS_H

#include <stddef.h>
#include <stdint.h>
#include <gmp.h>

#include "simd_mont_ifma.h"

#ifdef __cplusplus
extern "C" {
#endif

#define ED_SOA_ARENA 16   /* 8n-word scratch buffers handed out by the ops */

typedef struct {
    ifma_ctx_t mc;        /* modulus, R, np0, modmul scratch */
    size_t     n;         /* 52-bit limbs */
    int        w;         /* batch w-NAF window */
    size_t     m;         /* dictionary entries: 2^(w-2) */
    uint64_t  *dict;      /* 3*m*8n: entry j -> x, y, dxy (each 8n, SoA) */
    uint64_t  *d;         /* 8n: per-lane curve parameter d (Montgomery) */
    uint64_t  *arena;     /* ED_SOA_ARENA * 8n scratch */
    int        set;       /* curves/dictionary built? */
    int        bad_inv;   /* 诊断: 上一次 set_curves 里 Z 与 N 不互素(z 不可逆)的字典项数 */
    /* --- checkpoint / resume: 进度回调携带当前点, 恢复时从 (digit, 点) 继续 --- */
    size_t     resume_digit;   /* 从第几个 digit 开始 (0 = 从恒等点从头跑) */
    uint64_t  *rx0, *ry0, *rz0, *rt0;   /* 8n each: 恢复用初始点 (Montgomery), 可空 */
    int      (*progress)(void *ctx, size_t bits_done, size_t bits_total,
                         const uint64_t *Rx, const uint64_t *Ry, const uint64_t *Rz);
    void      *progress_ctx;
} ed_soa_ctx_t;

/* w in [3,12].  Returns 0 on success.
   field_mode: IFMA_FIELD_AUTO | IFMA_FIELD_MONT | IFMA_FIELD_MERS (see
   simd_mont_ifma.h).  AUTO uses the Mersenne fold kernel when N = 2^k - 1,
   which halves the madds per field mul; everything above the field layer
   (point ops, dictionary, checkpoint, gcd) is domain agnostic. */
int  ed_soa_init(ed_soa_ctx_t *c, const mpz_t N, int w);
int  ed_soa_init_ex(ed_soa_ctx_t *c, const mpz_t N, int w, int field_mode);
/* "montgomery (R=2^52n)" / "mersenne (2^k=1, k=3001, n52=58, fold<<15)" */
const char *ed_soa_field_name(const ed_soa_ctx_t *c);
void ed_soa_clear(ed_soa_ctx_t *c);

/* Per-lane curve setup: sigma[k] -> (d_k, P_k) via Atkin-Morain, then the
   dictionary (2j+1)P in affine form for j < m.  Lanes >= lanes are untouched.
   Builds the whole batch at once, so pass all 8 (pad sigma with anything). */
int  ed_soa_set_curves(ed_soa_ctx_t *c, const uint64_t *sigma, int lanes);

/* [s]P over the batch.  s must be the same for all lanes (48*lcm(1..B1)).
   Fills Qx/Qz (reduced, Montgomery-exited) and factor = gcd(Qz, N) (1 if
   none) for lanes [0, lanes).  Returns 0 on success, 1 if the progress
   callback asked to stop (in which case NO output is written). */
int  ed_soa_stage1(ed_soa_ctx_t *c, const mpz_t s, int lanes,
                   mpz_t *Qx, mpz_t *Qz, mpz_t *factor);

/* 批内进度回调: 每 ED_SOA_PROGRESS_BITS 个 digit 调一次, 并给出当前点 (SoA, Montgomery),
   上层可据此落盘 checkpoint。返回 0 = 继续, 非 0 = 请求中止 (ed_soa_stage1 立刻返回 1)。 */
#define ED_SOA_PROGRESS_BITS 16384
typedef int (*ed_soa_progress_fn)(void *ctx, size_t bits_done, size_t bits_total,
                                  const uint64_t *Rx, const uint64_t *Ry, const uint64_t *Rz);
void ed_soa_set_progress(ed_soa_ctx_t *c, ed_soa_progress_fn fn, void *ctx);

/* 从 checkpoint 恢复: start_digit 个 digit 已完成, 每 lane 给出普通域的 (Rx,Ry,Rz)
   (Z 必须可逆; 与标量存档同一语义)。内部重算 T = X*Y/Z, 每 lane 一次求逆。
   传 start_digit = 0 或 Rx = NULL 表示从恒等点从头跑。
   必须在 ed_soa_set_curves 之后调用 (需要 modulus/one)。 */
int  ed_soa_set_resume(ed_soa_ctx_t *c, size_t start_digit, int lanes,
                       const mpz_t *Rx, const mpz_t *Ry, const mpz_t *Rz);

/* Diagnostics for the bench/verification tooling. */
size_t ed_soa_dict_words(const ed_soa_ctx_t *c);
size_t ed_soa_dict_bytes(const ed_soa_ctx_t *c);

/* Field-op self test: random a,b < N per lane, checks that the SoA
   add/sub/neg helpers agree with mpz.  Returns the number of failures. */
int ed_soa_field_selftest(ed_soa_ctx_t *c, int trials);

/* Point-op self test: dbl / add / add_affine each compared against the same
   formulas evaluated in mpz.  Returns the number of failures. */
int ed_soa_point_selftest(ed_soa_ctx_t *c, int trials);

/* Debug hooks: run one point op on the whole batch with caller-provided buffers
   (8n words each); afterwards c->arena holds the op's internal slots
   (0..6 = A,B,C,E,F,G,H, 7 = X+Y scratch, 8/9 = add/sub scratch) so a failing
   step can be compared against mpz. */
void ed_soa_debug_dbl(ed_soa_ctx_t *c, uint64_t *ox, uint64_t *oy, uint64_t *oz, uint64_t *ot,
                      const uint64_t *ix, const uint64_t *iy, const uint64_t *iz,
                      const uint64_t *it);
void ed_soa_debug_add_affine(ed_soa_ctx_t *c, uint64_t *ox, uint64_t *oy, uint64_t *oz,
                             uint64_t *ot, const uint64_t *ix, const uint64_t *iy,
                             const uint64_t *iz, const uint64_t *it,
                             const uint64_t *qx, const uint64_t *qy, const uint64_t *qdxy);
/* field-level hooks (one 8n-word buffer per element) */
void ed_soa_debug_add_field(ed_soa_ctx_t *c, uint64_t *r, const uint64_t *a, const uint64_t *b);
void ed_soa_debug_sub_field(ed_soa_ctx_t *c, uint64_t *r, const uint64_t *a, const uint64_t *b,
                            uint64_t *scratch);
void ed_soa_debug_neg_field(ed_soa_ctx_t *c, uint64_t *r, const uint64_t *a);

#ifdef __cplusplus
}
#endif

#endif /* SIMD_EDWARDS_H */
