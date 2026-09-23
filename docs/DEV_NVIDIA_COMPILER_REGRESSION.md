# NVIDIA OpenCL 编译器代码膨胀与性能退化

## 现象

升级 `gen_mp_addsub_bits_stage1.py` 生成 ≥1536-bit 的 add/sub kernel 后，
NVIDIA (RTX 4060 Laptop) 性能退化 50%：

| 提交 | 4096b add/sub 模式 | 耗时 |
|------|-------------------|------|
| 4e4641d (手工) | 块融合 `unroll 32` | ~1714ms |
| 生成器扁平展开 | 逐 limb 展开 `#pragma unroll` | ~3005ms |

AMD 不受影响。

## 根因

### 问题 1：add/sub 扁平展开导致 VGPR 压力

生成器为每个 limb 生成一个独立的 `{ ... }` 代码块：

```c
// 生成器扁平展开（退化版，128 个独立 block）
static inline void add_mod_unroll_4096b_body(...) {
    { ulong sum = a[0]+b[0]+ca; ca=sum>>32; ... }
    { ulong sum = a[1]+b[1]+ca; ca=sum>>32; ... }
    // ... 126 个类似 block
    for (i=0;i<128;i++) { r[i] += N[i]; }  // #pragma unroll 32
}
```

NVIDIA 编译器将此视为 128 个独立的基本块，分配大量 VGPR 来
承载 `a[i]`、`b[i]`、`N[i]` 的加载结果，导致寄存器溢出到 Scratch Memory。

### 问题 2：mont_mul 完整展开导致编译器崩溃

`#pragma unroll`（不加因子）在 256 limbs 时产生 ~500 条内层指令。
NVIDIA 编译器内部 IR 膨胀超出处理能力 → 进程崩溃 (0xC0000005)。

## 修复

### 修复 1：add/sub 块融合（`gen_mp_addsub_bits_stage1.py`）

对 `limbs >= 32` 使用**块融合**模式：外层 `#pragma unroll` 遍历 32-element 块，
内层 `#pragma unroll 32` 处理当前块。

```c
// 块融合（修复版，4 × 32 块）
#pragma unroll
for (uint blk = 0u; blk < 4u; ++blk) {
    uint off = blk * 32u;
    #pragma unroll 32
    for (uint j = 0u; j < 32u; ++j) {
        uint i = off + j;
        // ... 标准 fused add/sub 逻辑
    }
}
// 全局 fixup: #pragma unroll 32 for i=0..127
```

关键思路：**将数据依赖隔离在 32-element 窗口内**，编译器将每个块
视为独立的寄存器生命周期，块之间通过 `carry_add`/`carry_sub` 传递。
VGPR 压力从 O(limbs) 降至 O(32)。

对 `limbs > 128` (>4096b) 的全局 fixup 循环不展开（`// no unroll`），
因为此时代码体积已接近编译器处理上限。

### 修复 2：mont_mul 平台守卫展开（`gen_mont_unroll.py`）

对 `A >= 48`（≥1536b）的 mont_mul，使用平台守卫：

```c
#if defined(__AMDGCN__)
#define MONT_UNROLL _Pragma("unroll")       // AMD: 完整展开，最大 ILP
#else
#define MONT_UNROLL _Pragma("unroll 32")    // NVIDIA: 部分展开，编译器安全
#endif
```

AMD 保留完整展开以利用更多指令级并行度（已验证性能更优）；
NVIDIA 限制为 32-way 展开，避免编译器崩溃或 VGPR 溢出。

此模式同样应用于 `_local` 变体的 `t_local[]`/`B_local[]` 循环。

## 调试方法论

### 1. 构建快速反馈循环

用户报告 "NVIDIA 性能退化 50%"，我迅速建立复现循环：
```powershell
echo '(2^4027-1)' | .\build_rel\Release\ecm.exe -v -d 0 -gpu -gpucurves 6144 1e2 0 --special_mult generic
```
20 秒内得到 pass/fail 信号，随后通过 git bisect 定位退化引入点。

### 2. 逐层 bisect，定位精确退化源

| 步骤 | 操作 | 耗时 | 结论 |
|------|------|------|------|
| 1 | 恢复全部 `.cl` 到 4e4641d | ~1958ms | 退化在 CL 代码 |
| 2 | 仅恢复 `ecm_stage1.cl` | ~3463ms | 部分恢复，不是唯一原因 |
| 3 | 删除 sliced 内核 | ~3402ms | 无关 |
| 4 | 仅恢复 `add/unroll_4096b.cl` + `sub/unroll_4096b.cl` | ~1884ms | **根因定位** |
| 5 | 同时恢复 mont_mul 的展开守卫 | ~1780ms | 完全恢复 |

### 3. 启发：编译器友好的代码生成原则

**原则 1：控制展开因子。** `#pragma unroll`（无参数）= "全展"。
NVIDIA 编译器在大展开时表现极差。始终使用明确的展开因子：
`#pragma unroll 32` 或 `#pragma unroll 16`。

**原则 2：数据局部性 > 完全展开。** 块融合模式将数据依赖窗口缩小到
32 elements，编译器可将窗口内的数据保持在寄存器中，块间通过
标量 carry 传递。这比完全展开 128 elements 更编译器友好。

**原则 3：平台差异需要守卫。** AMD 和 NVIDIA 对代码膨胀的容忍度
截然不同。用 `#if defined(__AMDGCN__)` 区分展开策略，让两个
平台都获得最优性能。

**原则 4：生成器要可回退。** `SKIP_AUTO = {4096}` 模式允许手工
优化的内核文件不被生成器覆盖。每个生成器都应有此保护机制。

## 相关文件

| 文件 | 变更 |
|------|------|
| `tools/gen/gen_mp_addsub_bits_stage1.py` | `emit_add_unroll`/`emit_sub_unroll`: limbs≥32 → 块融合模式 |
| `tools/gen/gen_mont_unroll.py` | `body()`/`body_local()`: A≥48 → 平台守卫展开 |
| `kernels/opencl/add_mod/add_mod_unroll_*.cl` | 生成器产出，≥32 limb 使用块融合 |
| `kernels/opencl/sub_mod/sub_mod_unroll_*.cl` | 同上 |
| `kernels/opencl/mont_mul/mont_mul_unroll_*{,_local}.cl` | 生成器产出，含 MONT_UNROLL 宏 |
| `docs/DEV_ECM_STAGE1_LOCAL.md` | `__local` kernel 文档 |
