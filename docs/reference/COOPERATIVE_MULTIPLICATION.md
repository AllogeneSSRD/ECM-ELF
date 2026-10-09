# 协作 Karatsuba 与切片 CIOS 的算术分析

固定参考资料。来源：`DEV_COOP_KARATSUBA_2048.md Architecture、Single-thread、Multi-thread 数学部分`。保留数学推导、第三方源码分析及其条件；源码行号针对原文研究的本地副本，第三方上游可能已有变化。本文不声明本仓库当前支持、默认值或性能。涉及实验数字时只按原文条件理解；当前实现见 [架构](../architecture/OVERVIEW.md)，当前计时口径见 [性能](../performance/STAGE2.md)。

## Architecture

### Descriptor registration

```
注册表 (ECM_MONT_OPERATORS):
  X(karatsuba_2048b,    karatsuba_2048b, -1, 48, 64, 64, OS_ANY, GPU_ANY, 0, true, 1, 0)
  X(karatsuba_2048b_mt4, karatsuba_2048b, -1, 48, 64, 64, OS_ANY, GPU_ANY, 0, true, 4, 320)
                                                                                     ^  ^
                                                                    coop_work_group_size  local_scratch_u32
```

关键字段：
- `cl_name` 相同 → 注入同一个 `#define ECM_STAGE1_MUL_IMPL mont_mul_karatsuba_2048b`
- `coop_work_group_size` 不同 → 控制是否生成 `ECM_STAGE1_COOP_WG` 及 `reqd_work_group_size(N)`
- `local_scratch_u32` 不同 → mt4 需要 256+ u32 LDS 存储 4×64 子积 + 1 carry

### enum dispatch 链

```c
// include/opencl_ecm_path_registry.h
enum { EcmCoopKernelPath_K2048_MT4 = 5 };  // 新增路径值

// ecm_coop_kernel_path_from_desc()  → strstr("karatsuba_2048b") → 5
// mont_kernel_path_for_plan()       → 5 注入为 -DECM_STAGE1_MUL_PATH=5
```

```c
// ecm_stage1_coop.cl (PATH dispatch)
#if ECM_STAGE1_MUL_PATH == 5 || ECM_STAGE1_MUL_PATH == (5 + 63)
    mont_mul_stage1_karatsuba_2048_coop(out, a, b, N, mont_scratch, lid);
#endif
```

PATH=5+63 处理 sqr 路径（高位置 1 复用同一函数，传入 `a=N`）。

### 条件编译结构

```
ecm_stage1_coop.cl:
  #if ECM_STAGE1_USE_COOP_WG         ← 只在 coop_wg>1 时有效
    mont_mul/sqr_stage1_coop() dispatch
  #endif

ecm_stage1.cl:
  #if ECM_STAGE1_COOP_WG > 1         ← coop 时使用 reqd_work_group_size
  kernel_double_add()                 ← 单文件唯一定义（coop.cl 中已移除重复定义）
```

---

## Single-thread implementation

### 算法：CIOS 交织 (完全等价于 `mont_mul_unroll_2048b`)

```
mont_mul_karatsuba_2048b(out, a, b, N, np0, limbs):
    t[66] = 0
    B[64] = b
    for i in 0..63:
        // 1) 加载一行乘积 ← t + a[i] * B
        carry = 0
        for j in 0..63:
            uv = t[j] + a[i] * B[j] + carry
            t[j] = uv.lo; carry = uv.hi
        t[64] = carry.lo; t[65] = carry.hi
        // 2) CIOS 归约
        m = t[0] * np0  (ECM 素域 np0=1, 即 m = t[0])
        carry = 0
        for j in 0..63:
            uv = t[j] + m * N[j] + carry
            if j>0: t[j-1] = uv.lo
            carry = uv.hi
        t[63] = t[64] + carry; t[64] = t[65] + (carry >> 32)
    3) 条件减法
```

### 与 `mont_mul_unroll_2048b` 的区别

| 维度 | unroll | karatsuba |
|------|--------|-----------|
| 外层循环 | `for i<64` (无 `#pragma unroll`) | 相同 |
| 内层循环 | `#pragma unroll` | 无 unroll hint |
| 逻辑 | CIOS 交织，t[66] | 完全相同 |
| 结果 | 数学等价 | 数学等价 |

**为什么不用 Karatsuba 公式** (a0*b0, a0*b1, a1*b0, a1*b1 三分法)：
- 拆分后在单线程内并不能减少乘法次数——三分法节约的是 1 次"大乘法"但引入了 carry 修正项 `carry_a * b0p1 * B + carry_b * a0p1 * B + carry_a * carry_b * B²`
- 单线程中这些修正项需要逐 limb 加法传播，复杂度和正确性风险远超收益
- 当前 CIOS 版本 64 迭代 × 64 乘加 = 4096 次 `mul_hi:mul_lo`，已经在 VGPR 预算内

**Karatsuba 的"节约"体现在多线程版本**——见下一节。

---

## Multi-thread (MT4) implementation

### 并行化策略

64×64 limb 全积 = 4 个 32×32 limb 子积：

```
A = [a0 | a1]    B = [b0 | b1]     (每半 32 limbs, 1024 bits)

P_lo  = a0 * b0    ← Thread 0    (32×32→64 limbs)    位移 0 limbs
P_ma  = a0 * b1    ← Thread 1    (32×32→64 limbs)    位移 32 limbs
P_mb  = a1 * b0    ← Thread 2    (32×32→64 limbs)    位移 32 limbs
P_hi  = a1 * b1    ← Thread 3    (32×32→64 limbs)    位移 64 limbs

全积 T[128] = P_lo + (P_ma + P_mb) << 1024 + P_hi << 2048
```

### LDS layout

```
offset  0:    p_lo  (64 u32)  — Thread 0 写入
offset  64:   p_ma  (64 u32)  — Thread 1 写入
offset 128:   p_mb  (64 u32)  — Thread 2 写入
offset 192:   p_hi  (64 u32)  — Thread 3 写入
        ────
total        256 u32 = registry.local_scratch_u32 = 320 (余量 64)
```

每个线程在自己的私有 `buf[64]` 中计算 32×32 卷积 → barrier → 写入 LDS。

### Master 线程 (lid==0) 组装

1. 从 LDS 复制 4 个子积到私有数组（避免 `__local`→`__private` 地址空间冲突）
2. 组装 `T[128]` = `P_lo[0..63]` + `(P_ma+P_mb)[0..63]@offset32` + `P_hi[0..63]@offset64`（每次加法含进位传播）
3. Montgomery reduce（64 轮 CIOS，np0=1 利用为 `m = T[0]`）
4. 条件减法 + 输出

### VGPR / LDS 预算

- 每线程 VGPR ~66（32×32 卷积，比 64×64 减半）
- LDS 总计 256 u32 = 1 KB（远低于 64 KB 上限）

---

## Multi-thread 为什么节约了 1 次乘法

### 传统 Karatsuba 公式

Karatsuba 用 3 次半长乘法替代 4 次：

```
P_lo  = a0 * b0                         ← 乘法 1
P_hi  = a1 * b1                         ← 乘法 2
C_sum = (a0+a1) * (b0+b1)               ← 乘法 3
Mid   = C_sum - P_lo - P_hi             ← 减法（无乘法开销）
全积  = P_lo + Mid * B + P_hi * B²
```

**但加法的进位** (`a0+a1` 可能溢出 32 limbs) 会**破坏公式**：
- `(a0+a1) mod 2^1024 ≠ true a0+a1`，差值为 `carry_a * 2^1024`
- `(a0+a1)(b0+b1)` 展开后需要修正 `carry_a * b0p1 * 2^1024 + carry_b * a0p1 * 2^1024 + carry_a * carry_b * 2^2048`
- 这些修正本身包含 32-limb 的乘法和多 limb 进位传播 → **在 GPU 上并不比直接做第 4 次乘法快**

### 当前选择：4-subproduct 方案

保留 4 次 32×32 乘法 → 每个线程独立完成一次，无需跨线程 carry 修正项。

| 方案 | 乘法次数 | 数据依赖 | carry 修正 | 适用 |
|------|---------|---------|-----------|------|
| 原始 unroll_2048b | 64 次逐行 CIOS | 串行 64 轮 | 自动吸收 | 单线程 |
| 3 乘法 Karatsuba | 3 次 32×32 | c_sum 依赖 lo+hi | 需 carry 修正 | GPU 不合适 |
| **4 乘法 subproduct** | 4 次 32×32 | **完全并行** | 无 | **当前方案** |

"节约"体现在**并行化**而非减少乘法次数：
- 单线程 CIOS：4096 次 64×64 `muladd`（512cycle 量级）
- MT4：每线程 1024 次 32×32 `muladd`（~128cycle 量级）+ 1 线程组装 + 64 轮 CIOS reduce

---
