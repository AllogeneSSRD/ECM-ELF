# Android Montgomery mul/sqr 性能分析（Adreno 642 / 830）

完整 512-bit 路径跑通后，结论与早期（仅 `priv_opt` / `wg` 有效时）完全不同。本文汇总 **32-bit limb 全路径 bench** 与 **24-bit limb `mont_mul_unroll_i24`（mul24）** 实测，并给出选型与优化建议。

> **构建说明**：`fips512_mt*` 与 `wg` @512 已从 manifest / bench 中剔除，不再编译或跑分。

配置：`instances=128`，`kernel_iterations=1000`，`launch_repeats=1`，WG 模式 `tpi=4`。

> **数据说明**：24b 与 384@32 对照段曾将两台 GPU 对调；已按 **Adreno 830 绝对性能高于 642** 校正（与同文档 §1 历史 512@32 unroll 数据一致：830 为 M ops/s 档，642 为百 K～1.4M 档）。

---

## 1. 总览：512-bit @ 32-bit limb（历史完整 bench）

| 路径 | Adreno 642 mul | Adreno 830 mul | 830/642 |
|------|----------------|----------------|---------|
| **unroll_only_512_manual** | 323K | **8.47M** | 26× |
| **unroll_only_512** (auto) | **1.43M** | 7.73M | 5.4× |
| local_only_512 | 265K | 2.74M | 10× |
| opt2_512_local | 265K | 2.41M | 9× |
| fips512 | 406K | 2.07M | 5.1× |
| priv_opt | 282K | 1.60M | 5.7× |
| priv (legacy) | 273K | 1.05M | 3.8× |
| unroll32/64 | 241–285K | 857–897K | ~3× |
| wg | 201K | 1.05M | 5.2× |
| fips512_mt* / cs | 174–218K | 831K–1.85M | — |

**结论**：512-bit 必须用 **512 专用路径 `unroll_only_512*`**，不能用 generic `priv_opt` / `wg` / `unroll32`。与桌面 stage1 默认一致。

### sqr @ 512-bit / 32-bit limb

| 路径 | 642 sqr | 830 sqr |
|------|---------|---------|
| unroll_only_512 | 355K | 3.57M |
| fips512 | 407K | 2.19M |
| priv_opt | 279K | 1.58M |
| wg | 198K | 1.07M |

---

## 2. 24-bit limb（`mont_mul_unroll_i24`）实测

内核：`mont_mul_unroll_i24.cl`，CIOS + `#pragma unroll`，内乘 `mont_i24_mul_full`（12-bit 分解 + `mul24`/`mad24`）；支持任意 `bits % 24 == 0`（512 固定 22 limbs）。

> **Level 1 优化（mad24 融合）**：详见 [`docs/architecture/OPENCL.md`](../docs/architecture/OPENCL.md)。下表 **「mad24 后」** 为 `kernel_iterations=10000`；**「优化前」** 为同路径初版实现、`kernel_iterations=1000`。

### 2.1 Level 1 mad24（ulong CIOS）

| 会话 | 参数 | 512@24 830 mul | 384@24 830 mul | 512@24 642 mul | 384@24 642 mul |
|------|------|----------------|----------------|----------------|----------------|
| 首轮 | `10000×1`，src 5KiB | 2.38M | 5.77M | 767K | 1.21M |
| Level2 同会话基线 | `1000×10`，src 11KiB | **2.76M** | **5.95M** | 735K | **1.20M** |

> 跨会话差异分析见 [`docs/architecture/OPENCL.md`](../docs/architecture/OPENCL.md)「跨次跑分」。**同会话内**对比 Level 1/2 时以第二列 ulong 为准。

### 2.2 Level 1–2 同会话对比（`1000×10`，8 内核）

| 配置 | GPU | L1 ulong | L2 u32 | **选用** |
|------|-----|----------|--------|----------|
| **384@24** | 830 | 5.84M | **6.58M** | **L2 u32** |
| **512@24** | 830 | **2.75M** | 2.70M | **L1 ulong** |
| **384@24** | 642 | **1.17M** | 1.15M | **L1 ulong** |
| **512@24** | 642 | 741K | **779K** | **L2 u32** |

> Level 3（nocopy）已废弃（830 @512 **−21%**）。Level 4（`blsub`）见 §2.4。

详见 [`docs/architecture/OPENCL.md`](../docs/architecture/OPENCL.md)。

### 2.4 Level 4 全位宽 bench（`1000×10`，8 内核）

**mul 最优 ops/s**（同档位内粗体为选用路径）：

| bits | **830 最优** | 830 vs L1 | **642 最优** | 642 vs L1 |
|------|-------------|-----------|-------------|-----------|
| **192** | **u32_blsub 32.3M** | +33% | blsub 2.62M | +1.5% |
| **384** | **manual 7.11M** | +23% vs L1 | blsub 1.19M | ≈0% |
| **512** | blsub 2.82M | +5% | **u32 777K** | +8% |
| **768** | **u32 991K** | +3% | **u32 364K** | **+37%** |

**Level 4 要点**：

- **blsub 单独**：642 全档位、830@512 仅 **0～3%**，可忽略。
- **u32_blsub**：830 **@192/@384 显著胜出**（+20～33%），需 correctness VERIFY 后再作默认。
- **830 @768**：blsub **−38%**（596K vs 963K），**必须禁用**；仅用 u32。
- **642 @768**：u32 相对 ulong **+37%**，与 @512 趋势一致。
- **生产 @512**：仍用 32b `unroll_only_512*`，i24 仅 bench 参考。

### 2.5 384-bit manual 展开（`u32_blsub`，`1000×10`，10 内核）

| GPU | auto 最优 mul | manual mul | 比值 | 结论 |
|-----|---------------|------------|------|------|
| **830** | u32_blsub 6.92M | **7.11M** | **1.027×** | manual 略胜，可作 830@384 首选（VERIFY 后） |
| **642** | blsub 1.19M | 540K | **0.45×** | manual **暴跌**，与 512 manual@642 同型；**禁用** |

- 编译体积：`src_kib≈86`（含 1785 行生成体）；仅 384@24 bench / 830 生产候选。
- **642 @384**：继续 **ulong/blsub auto**（~1.18M），勿链 manual。

### 2.3 优化前基线（历史，mad24 之前）

| 配置 | limbs | Adreno **830** mul | Adreno **642** mul |
|------|-------|--------------------|--------------------|
| **512-bit @24** | 22 | 1.57M | 541K |
| **384-bit @24** | 16 | 3.56M | 881K |

**提升（mul）**：830 @384 **~1.62×**，@512 **~1.52×**；642 @384 **~1.37×**，@512 **~1.42×**。

对照：**384-bit @32**（同次 bench，通用路径，12 limbs）：

| 路径 | Adreno **830** mul | Adreno **642** mul |
|------|--------------------|--------------------|
| priv_opt | **2.61M** | 482K |
| unroll32 | 1.51M | 485K |
| priv | 1.83M | 476K |
| wg | 1.83M | 340K |

---

## 3. 24b vs 32b：核心结论

### 3.1 不能只看 bit 宽度，要看 limb 数与路径类型

| 对比 | 说明 |
|------|------|
| 512@32 **unroll_only** | 16 limbs，全展开专用体 → **两台机各自最快档** |
| 512@24 **unroll_i24** | 22 limbs（⌈512/24⌉），CIOS + mul24；limb 多 **37.5%**，`mont_i24_mul_full` 开销大 |
| 384@32 **priv_opt** | 12 limbs，运行时 `limbs` 循环 |
| 384@24 **unroll_i24** | 16 limbs → 比 384@32 多 **33%** limb，但无 runtime 循环 |

Montgomery CIOS 工作量近似 **∝ limbs²**：

- 512：22² / 16² ≈ **1.89×** 更多内层乘加（相对 32b unroll_only）
- 384：16² / 12² ≈ **1.78×**

### 3.2 与各自「同位宽、同 GPU 最优」对比（mul）

| GPU | 位宽 | 24b 推荐路径 | ops/s (mul) | 32b 最优 | 比值 |
|-----|------|-------------|-------------|----------|------|
| **830** | 512 | ulong priv（**禁用 nocopy**） | **2.75M** | unroll_only **7.73–8.47M** | **~32%** |
| **830** | 384 | **u32 priv** | **6.58M** | priv_opt **2.61M** | **2.52×** |
| **642** | 512 | **u32** | **779K** | unroll_only **1.43M** | **~55%** |
| **642** | 384 | **ulong** | **1.17M** | priv_opt 482K | **2.43×** |

### 3.3 修正后的判断（两台机共性 + 差异）

**共性（生产 @512-bit）**

- **两台机 @512-bit 均应使用 32b `unroll_only_512*`**，**不要**用 unroll_i24 作 Mont 生产路径。
- unroll_i24 @512（ulong，同会话）约为 unroll_only 最优的 **32%（830）～54%（642）**。

**830（绝对性能高）**

- @512：32b unroll_only 终极；i24 ulong priv **2.75M**。
- @384：**u32 6.58M** >> priv_opt。

**642（绝对性能低）**

- @512：unroll_only **1.43M** >> i24；若必须用 i24：**u32 779K**。
- @384：**ulong 1.17M**；勿 u32。
- 绝对值约为 830 @384 最优的 **18%**。
- **manual unroll** 在 642 上仍劣于 auto（§1），与 unroll_i24 选型无关。

### 3.4 为何 unroll_i24 @512 无法替代 32b unroll_only（两台机）

| 因素 | 说明 |
|------|------|
| Limb 数 | 22 vs 16 → CIOS 约 **1.9×** 工作量 |
| 乘法语义 | `u24_mul_full` 每步 **4× mul24** + 组合；32b unroll 为单条 32×32→64 |
| 专用展开 | `unroll_only_512_body` 为 16-limb 深度优化；unroll_i24 为通用 22-limb CIOS |
| 绝对算力 | 830 把 32b unroll 推到 **8M+** ops/s；mad24 后 unroll_i24 **2.38M**，相对 32b 最优仍差 **~3.6×**（结构差距为主） |

### 3.5 为何 unroll_i24 @384 能赢 generic 32b（两台机，830 赢得更多）

- `priv_opt` / `unroll32` 无固定位宽展开，带 `limbs` 循环。
- unroll_i24 全展开 + mul24，在 **384@24（16 limb）** 上弥补 limb 数增加的开销。
- **830** 上绝对 **5.77M vs 2.61M**（**2.2×**）；**642** 上 **1.21M vs 482K**（**2.5×**）— mad24 后优势进一步扩大。

### 3.6 sqr

两台机 **sqr ≈ mul**（unroll_i24 用 `mul(a,a)`）。32b 侧仍有 `mont_sqr_priv_unroll_only_512_body` 可挖。

---

## 4. 为什么 `unroll_only_512`（32b）远快于 `priv_opt`

`mont_mul_priv_unroll_only_512_body`（`mont_priv_opt.cl`）：

- 固定 **16 limb**，无运行时 `limbs` 分支
- `#pragma unroll` 内外层全展开 → 纯 ALU 链
- N/B 私有缓存，modulus 走 `__constant`
- 无 local memory、无 WG 协作

`priv_opt` 为通用 `limbs` 循环 + 部分展开，Adreno 上难以达到同等常量折叠与指令调度。

---

## 5. 路径分类（工程建议）

| 类别 | 路径 | 建议 |
|------|------|------|
| **两台机生产默认 @512** | `unroll_only_512` 32b | 830 mul→**manual**；642→**auto**；**均不用 unroll_i24** |
| **830 @384 及以下** | **unroll_i24（mad24）首选** | **5.77M** vs priv_opt **2.61M**（**2.2×**） |
| **642 @384 及以下** | unroll_i24 优于 priv_opt | **1.21M** vs 482K；绝对性能仍低 |
| **禁止 @512 生产** | `mont_mul_unroll_i24` | 两台机均显著慢于 32b unroll |
| bench 已剔除 | `fips512_mt*`、`wg` @512 | 手机上全面落后 |
| 642 慎用 | `unroll_only_512_manual` | 实测比 auto 慢 4.4× |

---

## 6. 公平对比尚未完成项

| 对比 | 目的 |
|------|------|
| **288-bit @24**（12 limbs）vs **384-bit @32**（12 limbs） | 与 add/sub 相同「公平 limb 数」口径 |
| **504-bit @24**（21 limbs）vs **512-bit @32** unroll_only | ECM 常用 504@24 |
| unroll_i24 **hot 内核**（单次 enqueue，`inner=kernel_iterations`） | 剥离 global 重载 |
| unroll_i24 vs unroll_only @512 + **VERIFY** | 确认两台机均不切换生产 |

---

## 7. 优化方向（按优先级，含 unroll_i24）

### P0 — 算子选型

1. **@512-bit（830 / 642）**：`mont_mul_priv_unroll_only_512` + `mont_sqr_priv_unroll_only_512`（830 mul 可试 manual）
2. **勿将 unroll_i24 作 @512 生产路径**（两台机均已证实落后 32b unroll）
3. **@384 及以下**：可评估 unroll_i24；830 收益更大（绝对 3.56M 档）
4. micro-probe 仅保留：`unroll_only_512` vs `_manual`（按 GPU），**不必** probe unroll_i24@512

### P1 — unroll_i24 内核（research / 非 512 生产）

5. ~~**`mont_i24_mul_full` mad24 融合**~~ — **已完成**（Level 1）
6. ~~**32-bit CIOS MAC**~~ — 830@384 **u32 priv**；642@512 **u32** 或 **u32_nocopy**
7. ~~**nocopy B/N（Level 3）**~~ — 已移除（830 @512 −21%）
8. **Level 4 branchless final sub（`blsub`）** — 已实现，待 bench
9. **按上表 dispatch 接 stage1**（bench 后可将 blsub 并入选定 body）
10. **22-limb manual 展开**
10. **专用 sqr body**（24-bit `sqr_basecase` + REDC）
11. **hot 内核**（对齐 addsub `fused_hot`）

### P2 — 编译与产物

9. 生产 APK：512@32 只链 `unroll_only_512*`；unroll_i24 仅 bench / 非 512 实验
10. 642 manual 退化：RGA/ISA 分析
11. 修正 bench 文案：384@24 勿显示「22 limbs for 512-bit」

### P3 — 端到端

12. Mont 选型后瓶颈或在 add/sub、global 带宽、curve 调度
13. 830 sqr：专用 `mont_sqr_priv_unroll_only_512_body`

### P4 — bench 工具

14. 输出「相对最优」列；按 GPU 标注 `recommended`

---

## 8. 数据速查表（校正后）

### mul ops/s

| 配置 | Adreno **830** | Adreno **642** | 830/642 |
|------|----------------|----------------|---------|
| 512 @32 unroll_only_manual | **8.47M** | 323K | 26× |
| 512 @32 unroll_only auto | 7.73M | **1.43M** | 5.4× |
| 512 @24 最优 i24 | **2.75M** (830 ulong) | **779K** (642 u32) | 3.5× |
| 384 @24 最优 i24 | **6.58M** (830 u32) | **1.17M** (642 ulong) | 5.6× |
| 384 @32 priv_opt | **2.61M** | 482K | 5.4× |
| 384 @32 unroll32 | 1.51M | 485K | 3.1× |
| 512 @32 priv_opt | 1.60M | 282K | 5.7× |
| 512 @32 wg | 1.05M | 201K | 5.2× |

### 相对各自 @512 最优 mul（830→manual，642→auto unroll）

| 配置 | 830 | 642 |
|------|-----|-----|
| 512 @24 ulong vs 512 最优 | **0.32×** | **0.54×** |
| 384 @24 推荐路径 vs 512 最优 | 0.78× (u32 priv) | 0.56× (nocopy) |
| 384 @24 推荐路径 vs 384 priv_opt | **2.52×** (u32 priv) | **2.47×** (nocopy) |

---

## 9. 一句话策略（校正后）

| GPU | 512-bit mul（生产） | 512-bit sqr | 384-bit 及以下 | 避免 @512 |
|-----|---------------------|-------------|----------------|-----------|
| **Adreno 830** | `unroll_only_512_manual` | `unroll_only_512` | **i24 manual**（**7.11M** @384） | i24@512 |
| **Adreno 642** | `unroll_only_512` auto | `unroll_only_512` | **ulong/blsub**（**1.19M** @384） | u32@384、**i24 manual** |

**两台机 @512-bit：Mont 一律 32b `unroll_only_512*`，不用 unroll_i24。**

桌面 stage1 已默认 `unroll_only_512`（32b）。Android port 与 §1 历史结论一致；unroll_i24 仅作 **<512 bit 或无专用 unroll 时** 的研究/备选，且 **830 上绝对收益远高于 642**。

---

## 10. 与 add/sub limb24 的联动

add/sub 在 Adreno 上可从 limb24 + `fused_hot` 获益。Montgomery CIOS **以内层乘为主**：

- **@512**：add/sub 可用 24b；**Mont 必须用 32b unroll_only**（两台机 `mont_mul_unroll_i24` 均远慢于 unroll）
- **@384**：均优于 priv_opt；**830 manual（7.11M）**；**642 ulong/blsub（1.19M）**，禁 manual

stage1 应对 **add/sub 与 mont 分开配置**；勿假设 unroll_i24 对 Mont @512 有效。
