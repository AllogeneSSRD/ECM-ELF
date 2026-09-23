# mp_add_mod 全展开（full unroll）试验

## 思路

将 fused v2 主循环与修正循环按固定 limb 数完全展开，并把 `a/b/N/r` 放在标量寄存器（`uint a0..`、`uint r0..`），避免：

1. 循环分支与 induction 变量开销；
2. 对 `private uint x[MAX_LIMBS]` 的反复索引（部分驱动会 spill 到 scratch）。

试验核 `ecm_mp_add_mod_fused_unroll` 进一步从 **global 直接加载** 到标量，不写 `x/y/m/r[MAX_LIMBS]` 四块 private 缓冲（相对 `ecm_mp_add_mod_fused` 省约 `4×limbs×4` 字节 private）。

**未并入** `ecm_stage1.cl`；仅 `--addsub-only` bench 拼接 `mp_addmod_unroll_generated.cl`。

## 生成

```powershell
python tools/gen/gen_mp_add_mod_unroll.py
cmake --build build --config Debug --target opencl_ecm_addsub
```

支持 limb：`64`（2048-bit）、`128`（4096-bit）、`256`（8192-bit），由 `-DMAX_LIMBS=` 与 `--bits` 对齐。

## Bench

```powershell
.\build\Debug\opencl_ecm_addsub.exe -d 1 --addsub-only --bits 4096 1000 128 50
```

输出含 `mp_add_mod_fused_unroll` 与 `fused` / `legacy` 对比倍数。

## 预期利弊

| 可能收益 | 可能代价 |
|----------|----------|
| 去掉循环控制，利于 ILP / 指令调度 | ISA 体积暴增（8192-bit 约 270KB 源） |
| 标量寄存器持有 limb，减 private 索引 | VGPR 激增 → wave 占用下降 |
| global 直载减 private 流量 | 编译变慢；ICache 压力 |
| 修正分支保留（仅 S&lt;N） | 与 v2 相同 warp 分歧 |

## 实测（AMD 890M，`-d 1`，`1000 128 50`）

| bits | legacy | fused v2 | unroll (global) | unroll (private 缓冲) |
|------|--------|----------|-----------------|------------------------|
| 2048 | 2.59M | 2.55M | **23.0M** (~9×) | — |
| 4096 | 2.49M | 2.56M | **17.5M** (~6.8×) | **17.1M** (~6.5×) |
| 8192 | 0.67M | 0.70M | **1.86M** (~2.8×) | — |

GMP：legacy / mask / fused / `fused_unroll` 均 **PASS**。

### 解读

1. **全展开本身**是主要收益：`fused_unroll_priv`（仍走 `x/y/m/r[MAX_LIMBS]` 加载）与 `fused_unroll`（global 直载）在 4096-bit 上几乎相同（~0.98×），说明不是单纯省掉 private 四块数组，而是编译器对 **完全展开的标量链** 调度远好于 2-limb 循环版 `mp_add_mod`。
2. **位宽越大，加速比越低**（8192 约 2.8×）：符合 VGPR / ICache / 占用率上升的预期。
3. 你给的 **8-limb 样例对应 256-bit**，本 bench 的 2048/4096/8192 对应 **64 / 128 / 256 limb**；思路一致，但 limb 数需与 `--bits` 对齐。
4. **生产谨慎**：`ecm_stage1` 内还有 Montgomery / barrier 等，单独把 `mp_add_mod` 展开 256 limb 会显著增大编译体积；且仅适用于编译期固定位宽。

**结论**：在 addsub 微基准上，**固定 2048/4096/8192 的全展开值得继续试验**；在并入 stage1 前建议再测 NVIDIA，并看 RGA 的 VGPR/ISA 与占用率。

---

## 分而治之（limbs per thread / lpt）

### 参数

- 核名：`ecm_mp_add_mod_fused_lpt{16|32|64}`（`lpt` = limbs per thread）
- `work-group` 大小 = `MAX_LIMBS / lpt`（须整除；`lpt=64` 且总 limb=64 时退化为单 thread，不生成）
- **`lpt=48`**：128 / 256 limb **不能整除 48**，4096/8192 bench 会显示 `n/a`

### 动机（RGA 资源）

| 位宽 | 单 thread 全展开 | 问题 |
|------|------------------|------|
| 2048 / 64 limb | VGPR ~163 | 无 spill |
| 4096 / 128 limb | VGPR 256 + 少量 spill | 顶满 |
| 8192 / 256 limb | 大量 spill + ICache 超 32KB | 严重 |

每 thread 只展开 `lpt` 个 limb，经 LDS 传 `carry_add` / `carry_sub`（修正传 `c`）。

### 实测（890M，`1000×128×50`）

**4096-bit（128 limb）**

| lpt | threads | ops/s | vs unroll |
|-----|---------|-------|-----------|
| 16 | 8 | 17.5M | 0.88× |
| 32 | 4 | 19.0M | 0.96× |
| 64 | 2 | **20.5M** | **~1.03×** |
| 48 | — | n/a | — |

**8192-bit（256 limb）**

| lpt | threads | ops/s | vs unroll |
|-----|---------|-------|-----------|
| 16 | 16 | 9.0M | 2.6× |
| 32 | 8 | 9.0M | 2.6× |
| 64 | 4 | **10.7M** | **~3.1×** |
| 48 | — | n/a | — |

**2048-bit（64 limb）**：单 thread unroll 仍最快；`lpt16/32` 略慢于 unroll（多 barrier 开销）。

趋势：**位宽越大，优先更大 lpt（64）**；过小 lpt（16）增加 barrier 次数，8192 上与 lpt32 几乎持平但仍慢于 lpt64。

GMP：已生成的各 `lpt*` 核 **PASS**。
