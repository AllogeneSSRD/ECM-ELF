// ecm_edwards_cpu.cpp — Atkin-Morain Edwards (a=1, Z/2xZ/8) Stage-1, CPU / mpz.
//
// 交叉验证入口: 给定 N、sigma、B1, 计算
//   1. Atkin-Morain 曲线 d 与基点 P = (x1, y1)
//   2. s = 48 * lcm(1..B1)  (48 = lcm(12,16), 对齐 Prime95 ecm_calc_exp)
//   3. [s]P  (扩展 Edwards, 先 double-and-add)
//   4. ed_to_Montgomery: Qx = z + y, Qz = z - y  (对齐 Prime95 ed_to_Montgomery)
// 输出 d, P, s_bits, Qx, Qz, y_affine, u=Qx/Qz (投影无关不变量), gcd(Qz,N).
// 与 Prime95 存档 (e0000347 等) 交叉验证: u 与 gcd(Qz,N) 应一致.
//
// 编译: 见 build_edwards_test.bat (MSVC + vcpkg GMP)。

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string>
#include <gmp.h>

#include "ecm_edwards_cpu.h"
#include "ecm_edwards_mont.h"

// 标量乘算法开关: 1 = w-NAF + 仿射字典 (默认), 0 = double-and-add (基准).
#define ECM_EDWARDS_USE_NAF 1

// w-NAF 窗口 (默认 12; 调优/基准用 edwards_set_naf_w, driver 用 --edwards-naf-w).
// 交替 A/B 实测 (zen3/GMP 内核, B1=1e6/2e5, 3 轮取 min):
//   M347   w=8 0.622 -> w=12 0.609  (-2.1%)
//   M677   w=8 1.501 -> w=12 1.467  (-2.3%)
//   M991   w=8 2.785 -> w=12 2.705  (-2.9%)   [w=14 2.707, w=16 2.823, w=18 3.34]
//   M2203  w=8 2.120 -> w=12 2.076  (-2.1%)   [w=14 2.130]
//   M4003  w=8 6.963 -> w=12 6.729  (-3.4%)
// 并行 (24 曲线/24 线程): 7.31s -> 6.96s (-4.8%)。
// w>=14 收益转负: 字典 (2^(w-2) 项 × 3840 B) 超出缓存常驻容量。
static int g_edwards_naf_w = 12;
void edwards_set_naf_w(int w) {
    if (w >= 2) {
        g_edwards_naf_w = w;
    }
}

// ---------------------------------------------------------------------------
// 域运算 (mod N)
// ---------------------------------------------------------------------------
static void mod_mul(mpz_t r, const mpz_t a, const mpz_t b, const mpz_t N) {
    mpz_t t; mpz_init(t);
    mpz_mul(t, a, b); mpz_mod(r, t, N); mpz_clear(t);
}
static void mod_sqr(mpz_t r, const mpz_t a, const mpz_t N) {
    mpz_t t; mpz_init(t);
    mpz_mul(t, a, a); mpz_mod(r, t, N); mpz_clear(t);
}
static void mod_add(mpz_t r, const mpz_t a, const mpz_t b, const mpz_t N) {
    mpz_add(r, a, b);
    if (mpz_cmp(r, N) >= 0) mpz_sub(r, r, N);
}
static void mod_sub(mpz_t r, const mpz_t a, const mpz_t b, const mpz_t N) {
    mpz_sub(r, a, b);
    if (mpz_sgn(r) < 0) mpz_add(r, r, N);
}
static bool mod_inv(mpz_t r, const mpz_t a, const mpz_t N) {
    // 逆存在当且仅当 gcd(a,N)=1; 失败时返回 false (会揭示 N 的因子)
    return mpz_invert(r, a, N) != 0;
}

// ---------------------------------------------------------------------------
// Weierstrass 曲线 T^2 = S^3 - 8S - 32 上的标量乘 (Atkin-Morain 构造用)
// ---------------------------------------------------------------------------
struct ws_point { mpz_t s, t; };
static void ws_init(ws_point &p) { mpz_init(p.s); mpz_init(p.t); }
static void ws_clear(ws_point &p) { mpz_clear(p.s); mpz_clear(p.t); }

static void ws_dbl(ws_point &r, const ws_point &p, const mpz_t N) {
    // λ = (3s^2 - 8) / (2t); s3 = λ^2 - 2s; t3 = λ(s - s3) - t
    // 用局部副本 s,t 避免 r 与 p 别名时的覆写 bug
    mpz_t lam, s2, num, den, s, t;
    mpz_inits(lam, s2, num, den, s, t, NULL);
    mpz_set(s, p.s); mpz_set(t, p.t);
    mod_sqr(s2, s, N);
    mpz_mul_ui(num, s2, 3);               // 3s^2
    mpz_sub_ui(num, num, 8);              // 3s^2 - 8
    while (mpz_sgn(num) < 0) mpz_add(num, num, N);
    if (mpz_cmp(num, N) >= 0) mpz_mod(num, num, N);
    mpz_mul_2exp(den, t, 1);              // 2t
    if (mpz_cmp(den, N) >= 0) mpz_sub(den, den, N);
    mod_inv(den, den, N);
    mod_mul(lam, num, den, N);
    mod_sqr(s2, lam, N);
    mpz_sub(r.s, s2, s); mpz_sub(r.s, r.s, s);   // λ^2 - 2s
    mpz_mod(r.s, r.s, N);
    mpz_sub(num, s, r.s); mod_mul(num, lam, num, N);
    mpz_sub(r.t, num, t); mpz_mod(r.t, r.t, N);
    mpz_clears(lam, s2, num, den, s, t, NULL);
}

static void ws_add(ws_point &r, const ws_point &p, const ws_point &q, const mpz_t N) {
    mpz_t lam, num, den, s1, t1, s2, t2;
    mpz_inits(lam, num, den, s1, t1, s2, t2, NULL);
    mpz_set(s1, p.s); mpz_set(t1, p.t);
    mpz_set(s2, q.s); mpz_set(t2, q.t);
    mpz_sub(num, t2, t1);
    mpz_sub(den, s2, s1);
    mod_inv(den, den, N);
    mod_mul(lam, num, den, N);
    mod_sqr(num, lam, N);
    mpz_sub(r.s, num, s1); mpz_sub(r.s, r.s, s2);  // λ^2 - s1 - s2
    mpz_mod(r.s, r.s, N);
    mpz_sub(num, s1, r.s); mod_mul(num, lam, num, N);
    mpz_sub(r.t, num, t1); mpz_mod(r.t, r.t, N);
    mpz_clears(lam, num, den, s1, t1, s2, t2, NULL);
}

static void ws_mul(ws_point &r, const ws_point &p, uint64_t k, const mpz_t N) {
    // 简单 double-and-add
    ws_point base, acc;
    ws_init(base); ws_init(acc);
    mpz_set(base.s, p.s); mpz_set(base.t, p.t);
    mpz_set_ui(acc.s, 0); mpz_set_ui(acc.t, 0);  // acc = 无穷远 (用 (0,0) 哨兵)
    bool acc_zero = true;
    while (k) {
        if (k & 1) {
            if (acc_zero) { mpz_set(acc.s, base.s); mpz_set(acc.t, base.t); acc_zero = false; }
            else ws_add(acc, acc, base, N);
        }
        ws_dbl(base, base, N);
        k >>= 1;
    }
    mpz_set(r.s, acc.s); mpz_set(r.t, acc.t);
    ws_clear(base); ws_clear(acc);
}

// ---------------------------------------------------------------------------
// 扩展 Edwards 点 (X:Y:Z:T)
// ---------------------------------------------------------------------------
struct ed_point { mpz_t x, y, z, t; };
static void ed_init(ed_point &p) { mpz_inits(p.x, p.y, p.z, p.t, NULL); }
static void ed_clear(ed_point &p) { mpz_clears(p.x, p.y, p.z, p.t, NULL); }

// 统一加法 (标准 Edwards 加法律, a=1), 9M
//   曲线 x^2 + y^2 = 1 + d*x^2*y^2, 恒等点 (0,1)
//   x3 = (x1*y2 + x2*y1) / (1 + d*x1*x2*y1*y2)
//   y3 = (y1*y2 - x1*x2) / (1 - d*x1*x2*y1*y2)
static void ed_add(ed_point &r, const ed_point &p, const ed_point &q,
                   const mpz_t d, const mpz_t N) {
    mpz_t A, B, C, D, E, F, G, H, t;
    mpz_inits(A, B, C, D, E, F, G, H, t, NULL);
    mod_mul(A, p.x, q.x, N);                       // A = X1*X2
    mod_mul(B, p.y, q.y, N);                       // B = Y1*Y2
    mod_mul(C, p.t, q.t, N); mod_mul(C, C, d, N);  // C = d*T1*T2
    mod_mul(D, p.z, q.z, N);                       // D = Z1*Z2
    // E = (X1+Y1)(X2+Y2) - A - B
    mpz_add(t, p.x, p.y); mpz_add(E, q.x, q.y); mod_mul(E, t, E, N);
    mpz_sub(E, E, A); mpz_sub(E, E, B);
    while (mpz_sgn(E) < 0) mpz_add(E, E, N);
    mpz_sub(F, D, C); while (mpz_sgn(F) < 0) mpz_add(F, F, N);   // F = D - C
    mpz_add(G, D, C); if (mpz_cmp(G, N) >= 0) mpz_sub(G, G, N);  // G = D + C
    mpz_sub(H, B, A); while (mpz_sgn(H) < 0) mpz_add(H, H, N);   // H = B - A  (a=1)
    mod_mul(r.x, E, F, N);
    mod_mul(r.y, G, H, N);
    mod_mul(r.t, E, H, N);
    mod_mul(r.z, F, G, N);
    mpz_clears(A, B, C, D, E, F, G, H, t, NULL);
}

// 倍点 dbl-2008-hwcd (a=1), 4M+4S
static void ed_dbl(ed_point &r, const ed_point &p, const mpz_t N) {
    mpz_t A, B, C, D, E, F, G, H, t;
    mpz_inits(A, B, C, D, E, F, G, H, t, NULL);
    mod_sqr(A, p.x, N);
    mod_sqr(B, p.y, N);
    mod_sqr(C, p.z, N); mpz_mul_2exp(C, C, 1); if (mpz_cmp(C, N) >= 0) mpz_mod(C, C, N);
    mpz_set(D, A);  // a=1
    // E = (X1+Y1)^2 - A - B
    mpz_add(t, p.x, p.y); mod_sqr(E, t, N); mpz_sub(E, E, A); mpz_sub(E, E, B);
    if (mpz_sgn(E) < 0) { mpz_add(E, E, N); if (mpz_sgn(E) < 0) mpz_add(E, E, N); }
    mpz_add(G, D, B); if (mpz_cmp(G, N) >= 0) mpz_sub(G, G, N);
    mpz_sub(F, G, C); if (mpz_sgn(F) < 0) mpz_add(F, F, N);
    mpz_sub(H, D, B); if (mpz_sgn(H) < 0) mpz_add(H, H, N);
    mod_mul(r.x, E, F, N);
    mod_mul(r.y, G, H, N);
    mod_mul(r.t, E, H, N);
    mod_mul(r.z, F, G, N);
    mpz_clears(A, B, C, D, E, F, G, H, t, NULL);
}

// w-NAF: 有符号数字, 每个非零位后至少 w-1 个零位. 返回 LSB-first.
// 移植 Prime95 ecm.cpp:4824-4856 的 O(bits) 单遍算法 (tstbit + carry, 不做逐位右移).
static void naf_digits(const mpz_t k, int w, std::vector<int> &digits) {
    digits.clear();
    const size_t nbits = mpz_sizeinbase(k, 2);
    if (nbits == 0) return;
    const int max_val = (1 << (w - 1)) - 1;      // 2^(w-1)-1 (w=4 → 7)

    std::vector<int> out(nbits + 1, 0);
    int value = 0;        // 正在构造的 NAF 值 (奇数)
    int addin = 1;        // 当前位权
    int carry = 0;        // 负 NAF 码的借位
    size_t start = 0;     // 当前 NAF 值起始位

    for (size_t bitnum = 0; bitnum < nbits; bitnum++) {
        int this_bit = carry + (mpz_tstbit(k, bitnum) ? 1 : 0);
        carry = this_bit >> 1;                   // 0 或 1
        this_bit &= 1;
        if (this_bit) {
            if (value == 0) start = bitnum;      // 新 NAF 值起始
            value += addin;
        }
        if (value == 0) continue;                // addin 仅在构造中翻倍
        addin <<= 1;

        bool complete;
        if (bitnum == nbits - 1) {
            complete = true;
        } else if (addin < max_val) {
            complete = false;
        } else if (value <= max_val && addin - value <= max_val) {
            complete = false;
        } else {
            complete = true;
        }
        if (!complete) continue;

        if (value <= max_val) {
            out[start] = value;                  // 正 NAF 码
        } else {
            out[start] = value - addin;          // 负 NAF 码
            carry = 1;
        }
        value = 0;
        addin = 1;
    }
    if (carry) {
        out[nbits] = 1;                          // 末尾借位补 +1
    }
    size_t sz = out.size();
    while (sz > 1 && out[sz - 1] == 0) sz--;
    out.resize(sz);
    digits = std::move(out);
}

// ---------------------------------------------------------------------------
// Montgomery 域 Edwards 点运算 (mpn, 标量乘性能路径)
// ---------------------------------------------------------------------------
struct ed_point_mont { mont_t x, y, z, t; };
struct ed_affine_mont { mont_t x, y, dxy; };

static void ed_dbl_mont(ed_point_mont &r, const ed_point_mont &p, const mont_ctx_t *ctx) {
    mont_t A, B, C, E, F, G, H, t;
    mont_sqr(&A, &p.x, ctx);
    mont_sqr(&B, &p.y, ctx);
    mont_sqr(&C, &p.z, ctx);
    mont_add(&C, &C, &C, ctx);              // C = 2Z^2
    mont_add(&t, &p.x, &p.y, ctx);
    mont_sqr(&E, &t, ctx);
    mont_sub(&E, &E, &A, ctx);
    mont_sub(&E, &E, &B, ctx);              // E = (X+Y)^2 - A - B
    mont_add(&G, &A, &B, ctx);              // G = A + B  (=D+B, a=1)
    mont_sub(&F, &G, &C, ctx);              // F = G - C
    mont_sub(&H, &A, &B, ctx);              // H = A - B  (=D-B)
    mont_mul(&r.x, &E, &F, ctx);
    mont_mul(&r.y, &G, &H, ctx);
    mont_mul(&r.t, &E, &H, ctx);
    mont_mul(&r.z, &F, &G, ctx);
}

static void ed_add_mont(ed_point_mont &r, const ed_point_mont &p, const ed_point_mont &q,
                        const mont_t *d, const mont_ctx_t *ctx) {
    mont_t A, B, C, D, E, F, G, H, t;
    mont_mul(&A, &p.x, &q.x, ctx);
    mont_mul(&B, &p.y, &q.y, ctx);
    mont_mul(&C, &p.t, &q.t, ctx);
    mont_mul(&C, &C, d, ctx);               // C = d*T1*T2
    mont_mul(&D, &p.z, &q.z, ctx);
    mont_add(&t, &p.x, &p.y, ctx);
    mont_add(&E, &q.x, &q.y, ctx);
    mont_mul(&E, &t, &E, ctx);
    mont_sub(&E, &E, &A, ctx);
    mont_sub(&E, &E, &B, ctx);
    mont_sub(&F, &D, &C, ctx);
    mont_add(&G, &D, &C, ctx);
    mont_sub(&H, &B, &A, ctx);
    mont_mul(&r.x, &E, &F, ctx);
    mont_mul(&r.y, &G, &H, ctx);
    mont_mul(&r.t, &E, &H, ctx);
    mont_mul(&r.z, &F, &G, ctx);
}

static void ed_add_affine_mont(ed_point_mont &r, const ed_point_mont &p, const ed_affine_mont &q,
                               const mont_ctx_t *ctx) {
    mont_t A, B, C, E, F, G, H, t;
    mont_mul(&A, &p.x, &q.x, ctx);
    mont_mul(&B, &p.y, &q.y, ctx);
    mont_mul(&C, &p.t, &q.dxy, ctx);
    mont_add(&t, &p.x, &p.y, ctx);
    mont_add(&E, &q.x, &q.y, ctx);
    mont_mul(&E, &t, &E, ctx);
    mont_sub(&E, &E, &A, ctx);
    mont_sub(&E, &E, &B, ctx);
    mont_sub(&F, &p.z, &C, ctx);
    mont_add(&G, &p.z, &C, ctx);
    mont_sub(&H, &B, &A, ctx);
    mont_mul(&r.x, &E, &F, ctx);
    mont_mul(&r.y, &G, &H, ctx);
    mont_mul(&r.t, &E, &H, ctx);
    mont_mul(&r.z, &F, &G, ctx);
}

// 批量求逆 (mpz): iz[i] = z[i]^{-1} mod N, 一次 mpz_invert + O(n) 乘
static void mpz_batch_invert(mpz_t *iz, mpz_t *z, size_t n, const mpz_t N) {
    if (n == 0) return;
    std::vector<mpz_t> prefix(n);
    for (size_t i = 0; i < n; i++) mpz_init(prefix[i]);
    mpz_set(prefix[0], z[0]);
    for (size_t i = 1; i < n; i++) { mpz_mul(prefix[i], prefix[i - 1], z[i]); mpz_mod(prefix[i], prefix[i], N); }
    mpz_t inv;
    mpz_init(inv);
    mpz_invert(inv, prefix[n - 1], N);
    for (size_t i = n; i-- > 0;) {
        if (i > 0) {
            mpz_mul(iz[i], inv, prefix[i - 1]); mpz_mod(iz[i], iz[i], N);
            mpz_mul(inv, inv, z[i]); mpz_mod(inv, inv, N);
        } else {
            mpz_set(iz[i], inv);
        }
    }
    mpz_clear(inv);
    for (size_t i = 0; i < n; i++) mpz_clear(prefix[i]);
}

// 归一化字典到仿射 (Z=1): 用 mpz 做那 1 次逆, 其余 mont 运算
static void ed_to_affine_batch_mont(ed_affine_mont *aff, ed_point_mont *pts, size_t n,
                                    const mont_t *d, const mont_ctx_t *ctx) {
    if (n == 0) return;
    std::vector<mpz_t> z(n), iz(n);
    for (size_t i = 0; i < n; i++) { mpz_init(z[i]); mpz_init(iz[i]); mont_from(z[i], &pts[i].z, ctx); }
    mpz_batch_invert(iz.data(), z.data(), n, ctx->Nz);

    mont_t izm, xy;
    for (size_t i = 0; i < n; i++) {
        mont_to(&izm, iz[i], ctx);               // iz[i] 进 Montgomery 域
        mont_mul(&aff[i].x, &pts[i].x, &izm, ctx);
        mont_mul(&aff[i].y, &pts[i].y, &izm, ctx);
        mont_mul(&xy, &aff[i].x, &aff[i].y, ctx);
        mont_mul(&aff[i].dxy, &xy, d, ctx);
    }
    for (size_t i = 0; i < n; i++) { mpz_clear(z[i]); mpz_clear(iz[i]); }
}

// 分块 Montgomery NAF 标量乘 (支持 checkpoint/resume).
// 返回 true=完成 (r 有效), false=被 progress 中止 (cur 已填充当前累加点).
static bool ed_mul_chunked(ed_point &r, const mpz_t k, const ed_point &P, const mpz_t d,
                           const mpz_t N, uint32_t chunk_bits,
                           const edwards_checkpoint_t *resume,
                           edwards_progress_fn progress, void *pctx,
                           edwards_checkpoint_t *cur) {
    mont_ctx_t mc;
    if (mont_init(&mc, N) != 0) {
        // N 超固定尺寸上限 (不该发生, <10000 bit); 回退 mpz double-and-add, 无 checkpoint.
        ed_point acc, base;
        ed_init(acc); ed_init(base);
        mpz_set_ui(acc.x, 0); mpz_set_ui(acc.y, 1); mpz_set_ui(acc.z, 1); mpz_set_ui(acc.t, 0);
        mpz_set(base.x, P.x); mpz_set(base.y, P.y); mpz_set(base.z, P.z); mpz_set(base.t, P.t);
        size_t bits = mpz_sizeinbase(k, 2);
        for (size_t i = 0; i < bits; ++i) {
            if (mpz_tstbit(k, i)) ed_add(acc, acc, base, d, N);
            ed_dbl(base, base, N);
        }
        mpz_set(r.x, acc.x); mpz_set(r.y, acc.y); mpz_set(r.z, acc.z); mpz_set(r.t, acc.t);
        ed_clear(acc); ed_clear(base);
        return true;
    }

    const int w = g_edwards_naf_w;
    const size_t m = (size_t)1 << (w - 2);

    ed_point_mont Pm;
    mont_t dm;
    mont_to(&Pm.x, P.x, &mc);
    mont_to(&Pm.y, P.y, &mc);
    mont_to(&Pm.z, P.z, &mc);
    mont_to(&Pm.t, P.t, &mc);
    mont_to(&dm, d, &mc);

    std::vector<ed_point_mont> dict(m);
    mont_set(&dict[0].x, &Pm.x, &mc); mont_set(&dict[0].y, &Pm.y, &mc);
    mont_set(&dict[0].z, &Pm.z, &mc); mont_set(&dict[0].t, &Pm.t, &mc);
    ed_point_mont dblP, curpt;
    ed_dbl_mont(dblP, Pm, &mc);
    mont_set(&curpt.x, &Pm.x, &mc); mont_set(&curpt.y, &Pm.y, &mc);
    mont_set(&curpt.z, &Pm.z, &mc); mont_set(&curpt.t, &Pm.t, &mc);
    for (size_t j = 1; j < m; j++) {
        ed_add_mont(curpt, curpt, dblP, &dm, &mc);
        mont_set(&dict[j].x, &curpt.x, &mc); mont_set(&dict[j].y, &curpt.y, &mc);
        mont_set(&dict[j].z, &curpt.z, &mc); mont_set(&dict[j].t, &curpt.t, &mc);
    }

    std::vector<ed_affine_mont> aff(m);
    ed_to_affine_batch_mont(aff.data(), dict.data(), m, &dm, &mc);

    std::vector<int> digits;
    naf_digits(k, w, digits);
    const size_t total = digits.size();

    // 累加点初始化: 恒等点 或 resume 点
    ed_point_mont Rm;
    size_t done = 0;
    if (resume && resume->bitnum > 0) {
        mont_to(&Rm.x, resume->Rx, &mc);
        mont_to(&Rm.y, resume->Ry, &mc);
        mont_to(&Rm.z, resume->Rz, &mc);
        // T = X*Y/Z (仅 resume 时 1 次逆)
        mpz_t Zinv;
        mpz_init(Zinv);
        mpz_invert(Zinv, resume->Rz, mc.Nz);
        mont_t zinvm, xy;
        mont_to(&zinvm, Zinv, &mc);
        mont_mul(&xy, &Rm.x, &Rm.y, &mc);
        mont_mul(&Rm.t, &xy, &zinvm, &mc);
        mpz_clear(Zinv);
        done = resume->bitnum;
    } else {
        mont_set_ui(&Rm.x, 0, &mc);
        mont_set_ui(&Rm.y, 1, &mc);
        mont_set_ui(&Rm.z, 1, &mc);
        mont_set_ui(&Rm.t, 0, &mc);
    }

    // 主循环 (分块; done 从 MSB 已处理位数)
    ed_affine_mont neg;
    bool aborted = false;
    while (done < total) {
        size_t chunk_end = total;
        if (chunk_bits != 0 && done + chunk_bits < total) chunk_end = done + chunk_bits;
        for (; done < chunk_end; done++) {
            ed_dbl_mont(Rm, Rm, &mc);
            const int dgt = digits[total - 1 - done];
            if (dgt != 0) {
                const int a = dgt > 0 ? dgt : -dgt;
                const size_t idx = (size_t)(a - 1) / 2;
                if (dgt > 0) {
                    ed_add_affine_mont(Rm, Rm, aff[idx], &mc);
                } else {
                    mont_neg(&neg.x, &aff[idx].x, &mc);
                    mont_set(&neg.y, &aff[idx].y, &mc);
                    mont_neg(&neg.dxy, &aff[idx].dxy, &mc);
                    ed_add_affine_mont(Rm, Rm, neg, &mc);
                }
            }
        }
        if (done < total && progress && cur) {
            mont_from(cur->Rx, &Rm.x, &mc);
            mont_from(cur->Ry, &Rm.y, &mc);
            mont_from(cur->Rz, &Rm.z, &mc);
            cur->bitnum = (uint32_t)done;
            if (!progress(pctx, cur)) { aborted = true; break; }
        }
    }

    if (!aborted) {
        mont_from(r.x, &Rm.x, &mc);
        mont_from(r.y, &Rm.y, &mc);
        mont_from(r.z, &Rm.z, &mc);
        mont_from(r.t, &Rm.t, &mc);
    }

    mont_clear(&mc);
    return !aborted;
}

// 标量乘 [k]P (一次性, 无 checkpoint).
static void ed_mul(ed_point &r, const mpz_t k, const ed_point &P, const mpz_t d, const mpz_t N) {
#if ECM_EDWARDS_USE_NAF
    ed_mul_chunked(r, k, P, d, N, 0, nullptr, nullptr, nullptr, nullptr);
#else
    // double-and-add 基准路径
    ed_point acc, base;
    ed_init(acc); ed_init(base);
    mpz_set_ui(acc.x, 0); mpz_set_ui(acc.y, 1); mpz_set_ui(acc.z, 1); mpz_set_ui(acc.t, 0);
    mpz_set(base.x, P.x); mpz_set(base.y, P.y); mpz_set(base.z, P.z); mpz_set(base.t, P.t);
    size_t bits = mpz_sizeinbase(k, 2);
    for (size_t i = 0; i < bits; ++i) {
        if (mpz_tstbit(k, i)) {
            ed_add(acc, acc, base, d, N);
        }
        ed_dbl(base, base, N);
    }
    mpz_set(r.x, acc.x); mpz_set(r.y, acc.y); mpz_set(r.z, acc.z); mpz_set(r.t, acc.t);
    ed_clear(acc); ed_clear(base);
#endif
}

// ---------------------------------------------------------------------------
// Atkin-Morain 曲线构造
// ---------------------------------------------------------------------------
// 输出: d (曲线参数), P (基点, 扩展坐标: x, y, z=1, t=x*y)
static void atkin_morain(mpz_t d, ed_point &P, uint64_t sigma, const mpz_t N) {
    // (s,t) = sigma * (12,40) on T^2 = S^3 - 8S - 32
    ws_point w, r;
    ws_init(w); ws_init(r);
    mpz_set_ui(w.s, 12); mpz_set_ui(w.t, 40);
    ws_mul(r, w, sigma, N);

    mpz_t s, t, a, b, b2m1, num, den, t1, t2;
    mpz_inits(s, t, a, b, b2m1, num, den, t1, t2, NULL);
    mpz_set(s, r.s); mpz_set(t, r.t);
    mpz_mod(s, s, N); mpz_mod(t, t, N);  // 防御性: 确保 s,t ∈ [0,N)

    // α = (s-9) / (t+s+16)
    mpz_sub_ui(num, s, 9); if (mpz_sgn(num) < 0) mpz_add(num, num, N);
    mpz_add(den, t, s); mpz_add_ui(den, den, 16); if (mpz_cmp(den, N) >= 0) mpz_sub(den, den, N);
    mod_inv(den, den, N); mod_mul(a, num, den, N);

    // β = 2α(4α+1) / (8α²-1)
    mpz_mul_ui(t1, a, 4); mpz_add_ui(t1, t1, 1); if (mpz_cmp(t1, N) >= 0) mpz_sub(t1, t1, N); // 4α+1
    mpz_mul_2exp(t2, a, 1); mod_mul(num, t2, t1, N);  // 2α(4α+1)
    mod_sqr(t1, a, N); mpz_mul_2exp(t2, t1, 3); if (mpz_cmp(t2, N) >= 0) mpz_mod(t2, t2, N); mpz_sub_ui(t2, t2, 1); if (mpz_sgn(t2) < 0) mpz_add(t2, t2, N); // 8α²-1
    mod_inv(t2, t2, N); mod_mul(b, num, t2, N);

    // d = (2(2β-1)² - 1) / (2β-1)⁴
    mpz_mul_2exp(t1, b, 1); mpz_sub_ui(t1, t1, 1); if (mpz_sgn(t1) < 0) mpz_add(t1, t1, N); // 2β-1
    mpz_set(b2m1, t1);
    mod_sqr(t2, t1, N);  // (2β-1)²
    mpz_mul_2exp(num, t2, 1); mpz_sub_ui(num, num, 1); if (mpz_sgn(num) < 0) mpz_add(num, num, N); // 2(2β-1)²-1
    mod_sqr(den, t2, N);  // (2β-1)⁴
    mod_inv(den, den, N); mod_mul(d, num, den, N);

    // x = (2β-1)(4β-3) / (6β-5)
    mpz_mul_ui(t1, b, 4); mpz_sub_ui(t1, t1, 3); if (mpz_sgn(t1) < 0) mpz_add(t1, t1, N); // 4β-3
    mod_mul(num, b2m1, t1, N);
    mpz_mul_ui(t1, b, 6); mpz_sub_ui(t1, t1, 5); if (mpz_sgn(t1) < 0) mpz_add(t1, t1, N); // 6β-5
    mod_inv(t1, t1, N); mod_mul(P.x, num, t1, N);

    // y = (2β-1)(t²+50t-2s³+27s²-104) / ((t+3s-2)(t+s+16))
    mod_sqr(t1, t, N);  // t²
    mpz_mul_ui(t2, t, 50); mod_add(num, t1, t2, N);  // t²+50t
    mod_sqr(t1, s, N); mod_mul(t1, t1, s, N); mpz_mul_2exp(t1, t1, 1); if (mpz_cmp(t1, N) >= 0) mpz_mod(t1, t1, N); // 2s³
    mod_sub(num, num, t1, N);
    mod_sqr(t1, s, N); mpz_mul_ui(t1, t1, 27); if (mpz_cmp(t1, N) >= 0) mpz_mod(t1, t1, N); // 27s²
    mod_add(num, num, t1, N);
    mpz_sub_ui(num, num, 104); if (mpz_sgn(num) < 0) { mpz_add(num, num, N); if (mpz_sgn(num) < 0) mpz_add(num, num, N); }
    mod_mul(num, b2m1, num, N);

    mpz_mul_ui(t1, s, 3); mpz_add(t1, t1, t); mpz_sub_ui(t1, t1, 2); if (mpz_sgn(t1) < 0) mpz_add(t1, t1, N); // t+3s-2
    mpz_add(t2, t, s); mpz_add_ui(t2, t2, 16); if (mpz_cmp(t2, N) >= 0) mpz_sub(t2, t2, N); // t+s+16
    mod_mul(den, t1, t2, N);
    mod_inv(den, den, N); mod_mul(P.y, num, den, N);

    mpz_set_ui(P.z, 1);
    mod_mul(P.t, P.x, P.y, N);

    ws_clear(w); ws_clear(r);
    mpz_clears(s, t, a, b, b2m1, num, den, t1, t2, NULL);
}

// ---------------------------------------------------------------------------
// 公共入口: 单条 Edwards stage-1 曲线 (分块可恢复)
// ---------------------------------------------------------------------------
void edwards_atkin_morain(mpz_t d, mpz_t Px, mpz_t Py, uint64_t sigma, const mpz_t N) {
    ed_point P;
    ed_init(P);
    atkin_morain(d, P, sigma, N);
    if (Px) mpz_set(Px, P.x);
    if (Py) mpz_set(Py, P.y);
    ed_clear(P);
}
int edwards_get_naf_w(void) { return g_edwards_naf_w; }

void edwards_checkpoint_init(edwards_checkpoint_t *c) {
    c->bitnum = 0;
    mpz_inits(c->Rx, c->Ry, c->Rz, NULL);
}
void edwards_checkpoint_clear(edwards_checkpoint_t *c) {
    mpz_clears(c->Rx, c->Ry, c->Rz, NULL);
}

int edwards_stage1_curve_progress(mpz_t factor, mpz_t Qx, mpz_t Qz,
                                  const mpz_t N, uint64_t sigma, const mpz_t s,
                                  uint32_t chunk_bits,
                                  const edwards_checkpoint_t *resume,
                                  edwards_progress_fn progress, void *ctx) {
    mpz_t d, g;
    mpz_inits(d, g, NULL);
    ed_point P, R;
    ed_init(P); ed_init(R);

    atkin_morain(d, P, sigma, N);

    edwards_checkpoint_t cur;
    edwards_checkpoint_init(&cur);
    const bool completed = ed_mul_chunked(R, s, P, d, N, chunk_bits, resume, progress, ctx, &cur);
    edwards_checkpoint_clear(&cur);

    if (!completed) {
        ed_clear(P); ed_clear(R);
        mpz_clears(d, g, NULL);
        return 2;   // 被 progress 中止 (checkpoint 由调用方处理)
    }

    // ed_to_Montgomery: Qx = z + y, Qz = z - y  (对齐 Prime95 ed_to_Montgomery)
    mpz_t zq, zz;
    mpz_inits(zq, zz, NULL);
    mpz_add(zq, R.z, R.y); if (mpz_cmp(zq, N) >= 0) mpz_sub(zq, zq, N);
    mpz_sub(zz, R.z, R.y); if (mpz_sgn(zz) < 0) mpz_add(zz, zz, N);
    if (Qx) mpz_set(Qx, zq);
    if (Qz) mpz_set(Qz, zz);

    // 因子判定: gcd(Qz, N)
    mpz_gcd(g, zz, N);
    int rc = 0;
    if (mpz_cmp_ui(g, 1) > 0 && mpz_cmp(g, N) < 0) {
        if (factor) mpz_set(factor, g);
        rc = 1;
    } else if (mpz_cmp(g, N) == 0) {
        rc = 0;
    }

    mpz_clears(zq, zz, g, NULL);
    ed_clear(P); ed_clear(R);
    mpz_clear(d);
    return rc;
}

int edwards_stage1_curve(mpz_t factor, mpz_t Qx, mpz_t Qz,
                         const mpz_t N, uint64_t sigma, const mpz_t s) {
    return edwards_stage1_curve_progress(factor, Qx, Qz, N, sigma, s,
                                         0, nullptr, nullptr, nullptr);
}

#ifdef BUILD_ECM_EDWARDS_STANDALONE
// ---------------------------------------------------------------------------
// 批处理 SIMD (AVX512-IFMA) 是否可用。必须在**基线 TU** 里做 CPUID 探测：
// simd_*.cpp 是用 /arch:AVX512 编的，在它内部执行任何代码（哪怕只探测）都可能
// 让编译器在探测前就发出 AVX512 指令。
#if defined(_MSC_VER)
#include <intrin.h>
int edwards_simd_available(void) {
    int regs[4] = {0,0,0,0};
    __cpuid(regs, 0);
    if (regs[0] < 7) return 0;
    __cpuidex(regs, 1, 0);
    const int osxsave = (regs[2] >> 27) & 1;      // OSXSAVE
    const int avx     = (regs[2] >> 28) & 1;      // AVX
    if (!osxsave || !avx) return 0;
    const unsigned long long xcr0 = _xgetbv(0);
    // 需要 opmask(5) + ZMM_Hi256(6) + Hi16_ZMM(7) 才能安全用 zmm
    if ((xcr0 & 0xE6ULL) != 0xE6ULL) return 0;
    __cpuidex(regs, 7, 0);
    const int f  = (regs[1] >> 16) & 1;           // AVX512F
    const int dq = (regs[1] >> 17) & 1;           // AVX512DQ (movepi64_mask 要它)
    const int ifma = (regs[1] >> 21) & 1;         // AVX512_IFMA
    return (f && dq && ifma) ? 1 : 0;
}
#else
int edwards_simd_available(void) {
#if defined(__AVX512F__) && defined(__AVX512IFMA__) && defined(__AVX512DQ__)
    return 1;   // 这个 TU 若用 -mavx512* 编过，则整机假定支持
#else
    return 0;
#endif
}
#endif

// 主程序: 交叉验证 (对比 Prime95 存档)
// 用法: ecm_edwards_cpu <N> <sigma> <B1>
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <N> <sigma> <B1>\n", argv[0]); return 1; }
    mpz_t N, d, s, Qx, Qz, tmp;
    mpz_inits(N, d, s, Qx, Qz, tmp, NULL);
    mpz_set_str(N, argv[1], 10);
    uint64_t sigma = strtoull(argv[2], nullptr, 10);
    uint64_t B1 = strtoull(argv[3], nullptr, 10);

    ed_point P, R;
    ed_init(P); ed_init(R);
    atkin_morain(d, P, sigma, N);

    // s = 48 * lcm(1..B1)
    mpz_set_ui(s, 48);
    for (uint64_t p = 2; p <= B1; ++p) {
        // 简单判素
        bool prime = true;
        for (uint64_t q = 2; q * q <= p; ++q) if (p % q == 0) { prime = false; break; }
        if (!prime) continue;
        uint64_t v = p;
        while (v <= B1 / p) v *= p;  // 最大幂
        mpz_mul_ui(s, s, (unsigned long)v);
    }

    ed_mul(R, s, P, d, N);

    // ed_to_Montgomery: Qx = z + y, Qz = z - y
    mpz_add(Qx, R.z, R.y); if (mpz_cmp(Qx, N) >= 0) mpz_sub(Qx, Qx, N);
    mpz_sub(Qz, R.z, R.y); if (mpz_sgn(Qz) < 0) mpz_add(Qz, Qz, N);

    // 归一化: y_affine = Y / Z, u = Qx / Qz = (1+y)/(1-y)  (投影无关的不变量)
    mpz_t yaff, u, g;
    mpz_inits(yaff, u, g, NULL);
    if (mpz_invert(tmp, R.z, N)) {
        mod_mul(yaff, R.y, tmp, N);
        if (mpz_invert(tmp, Qz, N)) {
            mod_mul(u, Qx, tmp, N);
        }
    }
    mpz_gcd(g, Qz, N);   // 因子判定: gcd(Qz, N) > 1 即命中因子

    gmp_printf("N       = %Zd\n", N);
    gmp_printf("d       = %Zd\n", d);
    gmp_printf("P.x     = %Zd\n", P.x);
    gmp_printf("P.y     = %Zd\n", P.y);
    gmp_printf("s_bits  = %zu\n", mpz_sizeinbase(s, 2));
    gmp_printf("Qx      = %Zd\n", Qx);
    gmp_printf("Qz      = %Zd\n", Qz);
    gmp_printf("y_affine= %Zd\n", yaff);
    gmp_printf("u       = %Zd\n", u);
    gmp_printf("gcd(Qz,N)= %Zd\n", g);

    mpz_clears(N, d, s, Qx, Qz, tmp, yaff, u, g, NULL);
    ed_clear(P); ed_clear(R);
    return 0;
}
#endif /* BUILD_ECM_EDWARDS_STANDALONE */
