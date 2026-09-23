# ECM Stage1 __local (LDS) Kernel

## 概述

`--local` 将 `mont_mul` / `mont_sqr` 内部两个最大的临时数组 (`t[limbs+2]` 和 `B[limbs]`) 移至 `__local` 内存 (LDS)，
避免编译器将私有数组溢出到慢速 Scratch Memory。

## 使用

```powershell
echo '(2^8059-1)' | ecm.exe --local -gpu -d 0 -gpucurves 32 100 0
```

- `--local` 启用 LDS 版 kernel（`ecm_stage1_local.cl`）
- `--wg N` 可覆盖默认 work-group 大小

## 默认 Work-Group 大小

| MAX_LIMBS | WG Size | LDS 消耗 (per WG) |
|-----------|---------|-------------------|
| ≤128 (≤4096b) | 16 | 2×(128+2)×4×16 = 16KB |
| >128 (>4096b) | 8  | 2×(256+2)×4×8 = 16KB |

## LDS 布局

每个 work-item 分配 `ECM_STAGE1_WG_SIZE × 2 × (MAX_LIMBS+2)` uints 的 LDS scratch，
按 `local_id` 分区：

```
scratch[lid * 2 * (MAX_LIMBS+2) + 0]              → t_local
scratch[lid * 2 * (MAX_LIMBS+2) + (MAX_LIMBS+2)]  → b_local
```

## 性能

### AMD gfx1150 (8059-bit)

| 模式 | 耗时 |
|------|------|
| 默认（私有内存） | ~10.0s |
| `--local` (WG=8)  | ~4.5s |

### NVIDIA RTX 4060 Laptop (8059-bit)

| 模式 | 耗时 |
|------|------|
| 默认（私有内存） | ~10.0s |
| `--local` (WG=8)  | ~2.5s |

## 编译器兼容性

### NVIDIA `#pragma unroll` 崩溃 (0xC0000005)

**现象：** NVIDIA OpenCL 编译器在编译 ≥1536-bit (48 limbs) 的 `_local` mont_mul 函数时，
`#pragma unroll`（完整展开）会导致编译器进程崩溃（exit code 0xC0000005，"2 warnings generated" 后静默退出）。

**根因：** 完整展开产生 ~9200+ 条内层循环指令（48×192 次迭代 × 2 个内层循环），
NVIDIA 编译器内部 IR 膨胀超出处理能力。

**解决：** `gen_mont_unroll.py` 的 `body_local()` 对 A≥48 的 `_local` 变体使用平台守卫：

```c
#if defined(__AMDGCN__)
#define LOCAL_UNROLL _Pragma("unroll")       // AMD: 完整展开，最大 ILP
#else
#define LOCAL_UNROLL _Pragma("unroll 32")    // NVIDIA: 部分展开，编译器安全
#endif
```

AMD (`__AMDGCN__`) 使用完整 `#pragma unroll` 以保持最大指令级并行度；
NVIDIA 及其他平台使用 `#pragma unroll 32` 将每轮迭代体控制在 32 条，编译器可处理。

A<48（<1536b）的 `_local` 变体无条件使用完整展开（所有平台安全）。

### 相关文件

| 文件 | 说明 |
|------|------|
| `kernels/opencl/ecm_stage1_local.cl` | LDS 版 ECM stage1 kernel |
| `kernels/opencl/mont_mul/mont_mul_unroll_*_local.cl` | LDS 版 mont_mul/sqr（生成器产出） |
| `tools/gen/gen_mont_unroll.py` | 生成器，`body_local()` 含平台守卫逻辑 |
| `src/opencl_ecm_path_registry.cpp` | Build plan 选择 `_local` 路径、注入 `ECM_STAGE1_WG_SIZE` |
| `src/opencl_ecm_stage1.cpp` | Host 端 kernel 创建与 WG 启动 |
| `include/opencl_ecm_runtime_config.h` | `gpu_local`、`wg_size` 字段 |
