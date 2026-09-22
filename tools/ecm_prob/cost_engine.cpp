// cost_engine.cpp — ECM stage-1 PRAC 成本引擎.
//
// 输入 B1, 输出 JSON: {s_bits, n_primes, n_powers, dbl_2, add_3, prac_dbl, prac_add}.
//   其中 2^k -> k 次倍点, 3^k -> k 次倍点 + k 次加法, p>=5 的每个素数幂 p^k -> PRAC Lucas 链.
//
// 编译 (MSVC): cl /O2 /EHsc cost_engine.cpp
//   移植自 GMP-ECM ecm.c 的 lucas_cost (改写为返回 (倍点, 差分加法) 独立计数).
//   仅用 uint64_t, 无 GMP 依赖.

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <vector>
#include <algorithm>

using namespace std;

// 简单 Eratosthenes 筛, 返回 [2, n] 内素数. 对 B1 <= ~1e9 可行 (bit-packed, ~125MB).
static vector<uint32_t> sieve_primes(uint64_t n) {
    vector<bool> isp(n + 1, true);
    isp[0] = isp[1] = false;
    for (uint64_t i = 2; i * i <= n; ++i) {
        if (isp[i]) {
            for (uint64_t j = i * i; j <= n; j += i) isp[j] = false;
        }
    }
    vector<uint32_t> primes;
    for (uint64_t i = 2; i <= n; ++i) if (isp[i]) primes.push_back((uint32_t)i);
    return primes;
}

// Lucas 链成本 -> (倍点数, 差分加法数). 移植自 GMP-ECM ecm.c lucas_cost, 但拆开计数.
// 前提: n >= 5 (2,3 由调用方单独处理), v 是黄金比变体 (0.618..., 等).
static void lucas_chain_counts(uint64_t n, double v, uint64_t &dbl, uint64_t &add) {
    uint64_t d = n, e, r;
    r = (uint64_t)((double)d * v + 0.5);
    if (r >= n) { dbl = 0; add = n; return; }   // 退化, 不回环
    d = n - r;
    e = 2 * r - n;
    dbl = 1; add = 1;                            // 初始 duplicate + 最终 addition
    while (d != e) {
        if (d < e) swap(d, e);
        if (d - e <= e / 4 && (d + e) % 3 == 0) {          // cond 1: 3 add
            d = (2 * d - e) / 3;
            e = (e - d) / 2;                              // 用新 d
            add += 3;
        } else if (d - e <= e / 4 && (d - e) % 6 == 0) {   // cond 2: 1 add + 1 dbl
            d = (d - e) / 2;
            add += 1; dbl += 1;
        } else if ((d + 3) / 4 <= e) {                    // cond 3: 1 add
            d -= e;
            add += 1;
        } else if ((d + e) % 2 == 0) {                    // cond 4: 1 add + 1 dbl
            d = (d - e) / 2;
            add += 1; dbl += 1;
        } else if (d % 2 == 0) {                          // cond 5: 1 add + 1 dbl
            d /= 2;
            add += 1; dbl += 1;
        } else if (d % 3 == 0) {                          // cond 6: 3 add + 1 dbl
            d = d / 3 - e;
            add += 3; dbl += 1;
        } else if ((d + e) % 3 == 0) {                    // cond 7: 3 add + 1 dbl
            d = (d - 2 * e) / 3;
            add += 3; dbl += 1;
        } else if ((d - e) % 3 == 0) {                    // cond 8: 3 add + 1 dbl
            d = (d - e) / 3;
            add += 3; dbl += 1;
        } else {                                          // cond 9: 1 add + 1 dbl
            e /= 2;
            add += 1; dbl += 1;
        }
    }
}

// 黄金比变体 (GMP-ECM val[0..9]).
static const double GOLDEN[10] = {
    0.61803398874989485, 0.72360679774997897, 0.58017872829546410,
    0.63283980608870629, 0.61242994950949500, 0.62018198080741576,
    0.61721461653440386, 0.61834711965622806, 0.61791440652881789,
    0.61807966846989581};

// PRAC: 试 10 个黄金比变体, 按加权成本 6*add+5*dbl 取最省, 返回其 (dbl, add).
static void prac_counts(uint64_t n, uint64_t &dbl, uint64_t &add) {
    double best = 1e18;
    uint64_t bd = 0, ba = 0;
    for (int i = 0; i < 10; ++i) {
        uint64_t d = 0, a = 0;
        lucas_chain_counts(n, GOLDEN[i], d, a);
        double w = 6.0 * a + 5.0 * d;
        if (w < best) { best = w; bd = d; ba = a; }
    }
    dbl = bd; add = ba;
}

int main(int argc, char **argv) {
    uint64_t B1 = (argc > 1) ? strtoull(argv[1], nullptr, 10) : 1000000;
    if (B1 < 5) { fprintf(stderr, "B1 too small\n"); return 1; }

    vector<uint32_t> primes = sieve_primes(B1);
    uint64_t n_primes = primes.size();

    // s = lcm(1..B1) 的 bit 数 = ceil(sum over prime powers of log2(p)).
    double s_bits_f = 0.0;
    uint64_t n_powers = 0;
    uint64_t dbl_2 = 0, add_3 = 0;
    uint64_t prac_dbl = 0, prac_add = 0;

    for (uint32_t p : primes) {
        uint64_t v = p;
        while (v <= B1) {
            n_powers++;
            if (p == 2) { dbl_2++; }                 // 2^k: 1 doubling each
            else if (p == 3) { add_3++; }            // 3^k: 1 dbl + 1 add each (dbl 计入下方)
            else {
                uint64_t d = 0, a = 0;
                prac_counts(p, d, a);
                prac_dbl += d; prac_add += a;
            }
            if (v <= B1 / p) v *= p; else break;
        }
    }
    // 3^k 的倍点 = add_3 (每个 3^k 一次 duplicate)
    uint64_t total_dbl = dbl_2 + add_3 + prac_dbl;
    uint64_t total_add = add_3 + prac_add;
    uint64_t s_bits = 0;
    for (uint32_t p : primes) {
        uint64_t v = p;
        while (v <= B1 / p) v *= p;    // 最大幂 p^e <= B1
        s_bits_f += log2((double)v);
    }
    s_bits = (uint64_t)ceil(s_bits_f);

    printf("{\"B1\":%llu,\"s_bits\":%llu,\"n_primes\":%llu,\"n_powers\":%llu,"
           "\"dbl_2\":%llu,\"add_3\":%llu,\"prac_dbl\":%llu,\"prac_add\":%llu,"
           "\"total_dbl\":%llu,\"total_add\":%llu}\n",
           (unsigned long long)B1, (unsigned long long)s_bits,
           (unsigned long long)n_primes, (unsigned long long)n_powers,
           (unsigned long long)dbl_2, (unsigned long long)add_3,
           (unsigned long long)prac_dbl, (unsigned long long)prac_add,
           (unsigned long long)total_dbl, (unsigned long long)total_add);
    return 0;
}
