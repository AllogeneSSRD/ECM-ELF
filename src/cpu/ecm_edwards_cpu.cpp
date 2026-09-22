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

// 标量乘 [k]P, double-and-add (交叉验证里程碑; NAF 后续)
static void ed_mul(ed_point &r, const mpz_t k, const ed_point &P, const mpz_t d, const mpz_t N) {
    ed_point acc, base;
    ed_init(acc); ed_init(base);
    // acc = 恒等点 (0,1,1,0)
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
