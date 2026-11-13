#pragma once
#include "io/prime95_save.h"

// ecm_edwards_save.h — Prime95 ECM 二进制存档 (ECM_VERSION=6) 读写.
//
// 布局与 checksum 公式已与 Prime95 ecm_save/ecm_restore 及实测 e0000347 双向核对
// (checksum=0x16d2bff7 精确匹配)。

#include <cstdint>
#include <string>
#include <gmp.h>

// MIDSTAGE (stage-1 完成, Montgomery 点 Qx/Qz). state=2, montg_stage1=1.
bool ecm_edwards_write_midstage(const std::string& path, const ecm_save_common& cm, const mpz_t Qx,
                                const mpz_t Qz);
bool ecm_edwards_read_midstage(const std::string& path, ecm_save_common& cm, mpz_t Qx, mpz_t Qz);

// 只读文件头 + 数据区首部字段 (magic/version/k/b/n/c + curve/state/sigma/B1/B2),
// 不解析 giant。用于"目标文件已存在时是否允许覆盖"的冲突判定。
// 成功返回 true 并填 cm.curve/cm.B1/cm.B2/cm.sigma 与 *state; 文件缺失/损坏返回 false。
bool ecm_save_read_header(const std::string& path, ecm_save_common& cm, uint32_t* state);
