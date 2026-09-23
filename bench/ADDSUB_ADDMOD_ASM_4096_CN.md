# mp_add_mod 行内汇编版（AMDGCN，fused v2）

## 文件

| 文件 | 说明 |
|------|------|
| `cgbn/backends/opencl/kernels/mp_addmod_asm_fused.cl` | `asm_fused_block8`：8-limb `v_add_co_ci` / `v_cndmask` VCC 链 |
| `cgbn/backends/opencl/kernels/mp_addmod_asm_fused_generated.cl` | 核：`ecm_mp_add_mod_fused_asm8`（8 limb）、`ecm_mp_add_mod_fused_unroll_asm`（128=16×block8） |
| `tools/gen/gen_mp_addmod_asm_fused.py` | 生成器 |

仅在 **`__AMDGCN__`** 且 **`--bits 256`（8 limb）或 `4096`（128 limb）** 时拼入 bench；`-DMP_ADDMOD_ASM_ENABLE=1`。

## 算法（与你给的 fused VCC 思路一致）

每个 **8-limb 块** 内一条 asm 链：

1. `v_cmp_eq` 把 `ca_in` / `cs_in`（0/1）装入 `vcc_lo`
2. `v_add_co_ci` 链算 `S = a+b`（跨 limb 不断裂）
3. `v_not` + `v_add_co_ci` 链算 `T = S + ~N + cs`
4. `v_cndmask` 把 `vcc_lo` 存回 `ca` / `cs`
5. 若 `ca==0 && cs==0`，C+asm 修正 `r += N`

**4096-bit**：`asm_fused_block8` 调用 **16 次**（每段 8 limb），段间用 `ca/cs` 传递进位（等价于分治，但保持 VCC 链语义）。

> 你示例里的 **legacy 掩码版**（`S-N` + `v_cndmask` 选 `S`/`D`）未实现；当前为 **fused 推测减** 与 C `mp_add_mod` 一致。

## 生成与 bench

```powershell
python tools/gen/gen_mp_addmod_asm_fused.py
cmake --build build --config Debug --target opencl_ecm_addsub
.\build\Debug\opencl_ecm_addsub.exe -d 1 --addsub-only --bits 256 500 64 20
.\build\Debug\opencl_ecm_addsub.exe -d 1 --addsub-only --bits 4096 1000 128 50
```

## 消除 Scratch（private 大数组）

**原因**：`uint a[128], b[128], n[128], r[128]`（约 2KB private）易被 spill 到 scratch。

**做法**：

1. 内联函数改为 **`__global` 指针**，标量 `a0..a7` 从 global 直接加载（与 `fused_unroll` 一致）。
2. **4096**：16×8-limb **流式**块（`asm_fused_block8*` 写 `out+off`），无全宽 private 数组。
3. 修正段 **`asm_fix_add_n8`**：完全展开标量，避免 `r[i]`/`ri[8]` 索引。

## 实测（890M，约）

| 位宽 | C unroll | asm | private_mem | 备注 |
|------|----------|-----|-------------|------|
| 256（8 limb） | — | ~40M | **0B** | `asm8` / `asm8_vccsoft` |
| 4096 | ~20.0M | **~22.5M（~1.12×）** | **0B** | 流式 `unroll_asm`；GMP **PASS** |
| 4096（旧） | ~22.5M | ~15.7M | 2080B | 全 private[128] 版 |

要点：**必须用 `vcc_lo` + `v_add_co_ci`**；单 limb 拆 asm 会破坏 VCC 链。`v_addc_co` 在 gfx1150 OpenCL 内联 asm 中不可用。

## ISA 导出

```powershell
.\build\Debug\opencl_addsub_isa_export.exe --bits 4096
```
