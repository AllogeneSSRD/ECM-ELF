#pragma once
// Edwards (Atkin-Morain, a=1, Z/2xZ/8) scalar Stage1 backend.
//
// Window NAF uses shared GMP limb arithmetic. Larger moduli fall back to
// mpz double-and-add. The stage-1 result is the
// Montgomery point (Qx:Qz) = (z+y : z-y) of [s]P, plus a gcd factor check.

#include <stdint.h>
#include <gmp.h>
#include <cstddef>
#include <memory>
#include <vector>

class EdwardsNafPlan {
public:
    EdwardsNafPlan(mpz_srcptr scalar, int window, bool packed = false);

    size_t size() const { return size_; }
    int window() const { return window_; }
    bool packed() const { return !nonzero_.empty(); }
    size_t bytes() const;

    // CPU checkpoints count columns from the MSB. The plan stores the exact
    // existing CPU digits in LSB-first order in either representation.
    int operator[](size_t bit) const {
        if (!dense_.empty()) return dense_[bit];
        if (!wide_.empty()) return wide_[bit];
        const auto word = bit / 32;
        const auto shift = bit % 32;
        if ((nonzero_[word] & (uint32_t{1} << shift)) == 0) return 0;
        const auto bits = uint64_t{codes_[word]} | (uint64_t{codes_[word + 1]} << 32);
        const auto sign = uint32_t{1} << (window_ - 2);
        const auto code = static_cast<uint32_t>(bits >> shift);
        const int value = static_cast<int>(2 * (code & (sign - 1)) + 1);
        return (code & sign) ? -value : value;
    }

private:
    size_t size_ = 0;
    int window_;
    std::vector<int16_t> dense_;
    std::vector<int> wide_;
    std::vector<uint32_t> nonzero_, codes_;
};

// byte estimate for the production CPU windows (3..12), before allocating.
size_t edwards_naf_bytes(size_t scalar_bits, bool packed = false);

// Nonzero B1 identifies the canonical CPU scalar 48*lcm(1..B1). Use B1=0 for
// a caller-supplied scalar with no cache identity.
std::shared_ptr<const EdwardsNafPlan> edwards_cached_naf(uint64_t b1, int window);
std::shared_ptr<const EdwardsNafPlan> edwards_prepare_naf(mpz_srcptr scalar, uint64_t b1,
                                                         int window, bool packed = false);
void edwards_drop_cached_naf();

// Run one Atkin-Morain Edwards stage-1 curve.
//
//   factor : receives the non-trivial factor when found (may be NULL).
//   Qx, Qz : receive the Montgomery point (z+y, z-y) mod N when non-NULL
//            (for later save/checkpoint; may be NULL).
//   N      : composite to factor.
//   sigma  : 64-bit curve parameter (same value Prime95 uses for sigma_type=0).
//   s      : stage-1 exponent (must already be 48 * lcm(1..B1)).
//
// Returns:
//   1  : non-trivial factor found (1 < gcd(Qz,N) < N), stored in `factor`.
//   0  : no factor (gcd(Qz,N) == 1, or gcd == N meaning [s]P is identity mod N).
//   -1 : internal error.
int edwards_stage1_curve(mpz_t factor, mpz_t Qx, mpz_t Qz,
                         const mpz_t N, uint64_t sigma, const mpz_t s);

// Set the w-NAF window size (default 12; must be >= 2). Larger w -> fewer
// additions in the scalar multiply, but a 2^(w-2)-entry dictionary; measured
// optimum is 12 for 347..4003-bit operands (see ecm_edwards_cpu.cpp).
// Provided for tuning/benchmarking.
void edwards_set_naf_w(int w);

// Atkin-Morain 曲线构造: 输出 d 与基点 (Px, Py). 供 checkpoint 写 dict_start.
void edwards_atkin_morain(mpz_t d, mpz_t Px, mpz_t Py, uint64_t sigma, const mpz_t N);

// 当前 NAF 窗口 w (字典大小 = 2^(w-2)).
int edwards_get_naf_w(void);

// 批处理 SIMD (AVX512-IFMA, 8 曲线/批) 是否可用。1 = 可用。
// 在基线 TU 里用 CPUID 探测; 只有返回 1 才能调用 simd_edwards 的接口。
int edwards_simd_available(void);

// ---------------------------------------------------------------------------
// 分块可恢复标量乘 (self-checkpoint 支持)
// ---------------------------------------------------------------------------

// 标量乘 checkpoint 状态: 累加点 (plain 标准投影 X:Y:Z) + 已处理位数 (从 MSB).
typedef struct {
    uint32_t bitnum;       // 已处理位数 (从 MSB); 0 = 从头 (恒等点)
    mpz_t Rx, Ry, Rz;      // 累加点 (plain, 仅 bitnum>0 时有效)
} edwards_checkpoint_t;

void edwards_checkpoint_init(edwards_checkpoint_t *c);
void edwards_checkpoint_clear(edwards_checkpoint_t *c);

// 每处理完一个 chunk 调用. cur 为当前累加点 (plain) 与已处理位数.
// 返回 0 中止标量乘 (调用方可用 cur 写 checkpoint), 非 0 继续.
typedef int (*edwards_progress_fn)(void *ctx, const edwards_checkpoint_t *cur);

// 分块 stage-1 曲线 (支持 checkpoint/resume).
//   chunk_bits: 每块位数 (>0); 0 = 一次性跑完 (等价 edwards_stage1_curve).
//   resume: 非 NULL 且 resume->bitnum>0 时从该累加点继续.
//   progress: 非 NULL 时每 chunk 回调.
// 返回: 1=因子, 0=无因子(完成), -1=错误, 2=被 progress 中止 (未完成).
int edwards_stage1_curve_progress(
    mpz_t factor, mpz_t Qx, mpz_t Qz,
    const mpz_t N, uint64_t sigma, const mpz_t s,
    uint32_t chunk_bits,
    const edwards_checkpoint_t *resume,
    edwards_progress_fn progress, void *ctx,
    const EdwardsNafPlan *naf = nullptr, int window = 0);
