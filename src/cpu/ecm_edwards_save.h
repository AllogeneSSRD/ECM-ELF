#pragma once
// ecm_edwards_save.h — Prime95 ECM 二进制存档 (ECM_VERSION=6) 读写.
//
// 布局与 checksum 公式已与 Prime95 ecm_save/ecm_restore 及实测 e0000347 双向核对
// (checksum=0x16d2bff7 精确匹配)。

#include <cstdint>
#include <string>
#include <gmp.h>

// 公共头/数据字段 (k*b^n+c, B1/B2/sigma/curve).
struct ecm_save_common {
    double k = 1.0;
    uint32_t b = 2;
    uint32_t n = 0;
    int32_t c = -1;
    char stage[10] = {0};         // "C1S1" / "C1S2" (null-terminated, <=9 chars)
    double pct_complete = 0.0;
    uint32_t curve = 1;
    uint64_t B1 = 0;
    uint64_t B2 = 0;
    uint64_t sigma = 0;
};

// MIDSTAGE (stage-1 完成, Montgomery 点 Qx/Qz). state=2, montg_stage1=1.
bool ecm_edwards_write_midstage(const std::string &path, const ecm_save_common &cm,
                                const mpz_t Qx, const mpz_t Qz);
bool ecm_edwards_read_midstage(const std::string &path, ecm_save_common &cm,
                               mpz_t Qx, mpz_t Qz);

// 只读文件头 + 数据区首部字段 (magic/version/k/b/n/c + curve/state/sigma/B1/B2),
// 不解析 giant。用于"目标文件已存在时是否允许覆盖"的冲突判定。
// 成功返回 true 并填 cm.curve/cm.B1/cm.B2/cm.sigma 与 *state; 文件缺失/损坏返回 false。
bool ecm_save_read_header(const std::string &path, ecm_save_common &cm, uint32_t *state);

// STAGE1 (Edwards stage-1 中途 checkpoint). state=1, montg_stage1=0.
//   dict_start=(x,y) 为基点, e=(x,y,z) 为当前累加点 (标准投影).
bool ecm_edwards_write_stage1(const std::string &path, const ecm_save_common &cm,
                              uint64_t stage1_start_prime, uint32_t exp_buffer_size,
                              uint32_t bitnum, uint32_t dict_size,
                              const mpz_t dict_x, const mpz_t dict_y,
                              const mpz_t e_x, const mpz_t e_y, const mpz_t e_z);
bool ecm_edwards_read_stage1(const std::string &path, ecm_save_common &cm,
                             uint64_t *stage1_start_prime, uint32_t *exp_buffer_size,
                             uint32_t *bitnum, uint32_t *dict_size,
                             mpz_t dict_x, mpz_t dict_y,
                             mpz_t e_x, mpz_t e_y, mpz_t e_z);
