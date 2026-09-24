# ECM Montgomery Stage-1（Suyama sigma，SIMD 优先）开发文档

> 状态：**设计阶段（grilling 进行中）**。本文档按"事实 → 决策 → 实现 → 存档格式"顺序累积记录；
> 每一条决策都会写清**理由**与**被否掉的选项**，因为这类曲线族/sigma 语义的坑一旦踩错，
> 后面所有对齐工作都会白做。
>
> 姊妹文档：`docs/ECM_EDWARDS_STAGE1.md`（Atkin-Morain Edwards，sigma_type=0，已落地）。

---

## 1. 目标与非目标

**目标**
1. 实现 **Suyama sigma（Prime95 `sigma_type=1`）的 Montgomery 曲线 stage-1**，数学与
   gmp-ecm `param0`、Prime95 `choose12`、PrMers `ECM_TE` 一致（Z/12 挠子群，有效除子 D≈22.97）。
2. **SIMD 批处理版本为主要交付物**：AVX-512 IFMA，8 曲线/lane 批，复用 `src/cpu/simd_mont_ifma.*`
   的域层（Montgomery CIOS + `N = 2^k−1` 折叠域）。
3. 存档**统一为可读文本**（不再是本仓库自定义二进制），字段/语义对齐 gmp-ecm / Prime95 / PrMers 家族。

**非目标（本轮）**
- stage-2（B2）实现与交接本体；本任务只保证 stage-1 产物**可被** stage-2 消费。
- 不改动现有 Edwards(sigma_type=0) 与 GPU/OpenCL param3 路径的行为（可共存）。
- PRAC/windowed 链作为**第二阶段**候选（见 §6 决策 Q4），不在第一里程碑内。

---

## 2. 背景知识：Montgomery 曲线、x-only 算术与乘法链

> 本节是给"不熟 Montgomery 曲线算术"的读者（也包括未来的我）写的入门 + 决策依据。
> 所有公式都写成**我们实现里真正会用的形式**（射影 (X:Z)、差分加法需要差分点、a24=(A+2)/4），
> 不是教科书里的 (x,y) 形式。

### 2.1 ECM stage-1 到底在算什么

ECM 的想法：随机选一条曲线 E，它在模素数 p 下的群阶 `#E(F_p)` 是个"随机"的数；若
`#E(F_p)` 是 **B1-光滑**的（所有素因子 ≤ B1），那么让点 P 乘上

```
s = lcm(1..B1) = ∏_{q ≤ B1} q^⌊log_q B1⌋
```

就必然把 P 送到**单位元 O**（因为 `#E(F_p) | s`）。此时用射影坐标算出来的点满足 `Z ≡ 0 (mod p)`，
于是 `gcd(Z, N)` 露出 p 这个因子 ✓。

- 多条曲线 = 多次尝试（`#E(F_p)` 光滑是概率事件）；
- 曲线带**大挠子群**（torsion）时，`#E(F_p)` 总是那个挠子群的倍数，等价于"白送"一个小的光滑因子，
  于是有效除子 `D_eff` 更大、成功率更高。Suyama σ 参数化给 **Z/12**（D≈22.97），
  而 GMP-ECM **param3** 只有 Z/4（D≈7.6）⇒ **有效除子 D 大约 3×**。这就是本次要做 Suyama 的原因。
  ⚠ 注意这个 ≈3× 是 **D 的比值，不是成功率比值**：成功率 = ρ(log(p/D)/log B1)，D 只以 "log D / log B1"
  的形式进入，所以比值会小很多，且随位宽/B1 变化。B1=256 实测（bit 15–40，见 §19.5 表）
  单曲线成功率只有 **1.30×–1.8×**（bit20：30.47% vs 21.64% = 1.41×），模型在 bit130/B1=44e6 给 1.18×。
- stage-2（B2）是另一套机制，本任务不做，只要求 stage-1 的产物能被它消费。

### 2.2 Montgomery 曲线的形状与 x-only 的好处

标准形式（B=1）：

```
E_A :  y² = x³ + A·x² + x        （A ≠ ±2，A 是曲线参数）
```

**为什么 ECM 用这种曲线**：它的**倍点与加法只需要 x 坐标**。因为 (x,y) 与 (x,−y) 在群律里
"作用相同"（P 与 −P 有相同的 x），把 x 射影化 `x = X/Z` 之后，整个标量乘可以在
`(X:Z)` 上做完，**完全不需要 y**、不需要求平方根、不需要一般域逆（只在最后归一化时来一次）。

对我们（SIMD 批量）的额外好处：
- 每个曲线状态只有 **2 个域元素**（Edwards 需要 4 个：X,Y,Z,T）⇒ 寄存器压力小；
- 每步的公式**形状固定、无分支**（见 2.5）⇒ 8 条 lane 可以走同一条指令流。

### 2.3 x-only 算术：射影坐标、单位元、以及"差分加法"这个坑

- 点写成 `P = (X : Z)`，代表仿射 `x = X/Z`。**没有 y**。
- **单位元 O**（无穷远点）在 x-only 里没有 x 坐标，代码里用 `Z = 0` 表示 ✓ 这就是命中判定
  `gcd(Z, N) > 1` 的来源。
- **倍点**：`[2]P` 只需要 P 自己 ⇒ 有纯公式（2.4）。
- **加法 `P + Q` 就不行了**：只知道 `x_P, x_Q` 无法算出 `x_{P+Q}`，必须额外知道
  **`P − Q` 的 x**。这就是"**差分加法**（differential addition）"：
  `xADD(P, Q, P−Q) → x_{P+Q}`，而 `{P, Q, P−Q}` 三个 x 一旦给定，答案唯一。

> **这是我本轮踩过的坑，写进文档免得重犯**：`xADD` 里那个"差分点"必须用**仿射 x**
> （`X_diff / Z_diff`）。我一开始把 `u³`（起点射影 X，而它的 Z 是 `v³`）直接当差分 x 用，
> 结果 200 多个候选约定全部"不匹配"；归一化之后同一个测试立刻 28/28 通过。
> 教训：x-only 代码里凡是"当作 x 用"的量，先问一句"它的 Z 是 1 吗？"

### 2.4 两个公式与成本（我们实现里真正会写的形式）

记 `a24 = (A+2)/4`（避免除法/大常数乘法），`k = a24`：

```
倍点 xDBL:   t0 = X+Z ; t1 = X−Z
             t0 = t0² ; t1 = t1²
             t2 = t0 − t1
             X2 = t0 · t1
             Z2 = t2 · (t1 + a24·t2)          成本 3M + 2S

差分加法 xADD(P,Q,xdiff):
             t0 = X_P+Z_P ; t1 = X_P−Z_P
             t2 = X_Q+Z_Q ; t3 = X_Q−Z_Q
             t4 = t0·t3   ; t5 = t1·t2
             X3 = (t4+t5)²
             Z3 = (t4−t5)² · xdiff            成本 3M + 2S
```

- `a24` 的两种写法等价：`Z2 = t2·(t1 + ((A+2)/4)·t2)` ⇔ `t2·(t0 + ((A−2)/4)·t2)`（因为 `t0 = t1 + t2`）。
  **我们的存储/对拍口径**：存档写的是 **Montgomery `A`**（已用 29 条 PrMers 样本 + gmp-ecm 验证），
  内部用 `a24`；参考实现 gmp-ecm 也用 `(A+2)/4` 这一支。
- 一次 **ladder 步**（一次倍点 + 一次差分加法）= **6M + 4S**。

### 2.5 Montgomery ladder：为什么它天然适合 SIMD

经典二进 ladder 维护不变式

```
(R0, R1) = (k·P, (k+1)·P)        差分 R1 − R0 = P（固定！所以 xdiff 只需算一次）
```

从 MSB 往 LSB 走，每一位做**同样的一件"事"**：

```
if bit == 1:  R0 ← R0 + R1 ;  R1 ← 2·R1
else:         R1 ← R0 + R1 ;  R0 ← 2·R0
```

标准实现用 **cswap**（条件交换）写成**无分支**形式：

```
if bit != swapped: swap(R0, R1); swapped ^= 1
R0, R1 = xDBLADD(R0, R1, xdiff)      // 一步同时算 R0+R1 与 2R
```

要点：
1. 每位恒定成本（6M+4S），无分支、无内存访问模式变化 ⇒ **分支预测/指令流对 SIMD 友好**；
2. 8 条曲线共用**同一个 `s`**（同一批的 B1 相同）⇒ 连"哪一位是 1"都一样，
   所以 8 个 lane 连 cswap 掩码都相同，**完全无发散**；
3. 中途存档很自然：存 `bitnum` + `(R0.X:R0.Z)` 即可续跑（与本仓库 Edwards 的 `bitnum` 方案同构）。

### 2.6 指数 `s` 的构造与"两家的常数差"

```
s = ∏_{q ≤ B1} q^⌊log_q B1⌋ = lcm(1..B1)
```

- **gmp-ecm param0**：`s = lcm(1..B1)`，**没有**额外乘子（本轮用 B1=2 暴力枚举得唯一 `s=2`、
  再验证 B1=3/5/7/11 的比值链、最后 `x(lcm(1..5000)·P₀)` 与它的 X 完全相符而钉死）。
- **Prime95 `choose12`（Montgomery stage-1）**：`s = 12·lcm(1..B1)`，源码注释写着
  "choose12 means we should include 2 extra twos and 1 extra 3"（`ecm.cpp:7461-7463`）。
  这不是错，而是"把挠子群 12 显式乘进去"的约定；对 B1 ≥ 4 来说 `12 | lcm(1..B1)`，
  所以**不影响命中概率**，但**改变 `[s]P` 的具体点**。
- 因此："逐点对齐哪家"必须先选定常数 —— 本项目的决定（Q1）是**对齐 gmp-ecm ⇒ 用 `lcm(1..B1)`**；
  Prime95 的 `12·` 作为可选参数保留（`tools/stat/suyama_mont_ref.py --torsion 12` 就能切）。

### 2.7 "乘法链"与"plan 搜索"到底是什么

要算 `[s]P`，不同的"加法链"成本不同。概念上分三层：

1. **加法链（addition chain）**：用最少的"加倍 + 一般加法"把一个整数 s 造出来。
   纯 2 进 ladder 是"每位一次加倍 + 视 bit 一次加法"的最朴素做法。
2. **Lucas 链**：在 x-only 世界里，一般加法必须先知道差分点，所以主流的链形式是
   **Lucas 链**——始终保持若干"相邻"的倍数（如 `a·P, (a+1)·P, (a+2)·P` 或更一般的
   `{base·P, addin·P, diff·P}` 三元组），这样每一步的差分都是已知的。
3. **PRAC**（Peter Montgomery 的 *PRactical Addition Chain*）：一种**实用的启发式搜索**，
   把 s 拆成"某个 D 的倍数 + 若干已知倍数的 Lucas 加法"，用代价模型挑最省的那条链。

**"搜索"在搜什么**：给定目标指数 s，找一组 `(D, 一串 Lucas 加法)`，使得
"先乘/加到 `D·P`（用重复加倍），再用已知的 `base/addin/diff` 做若干次 Lucas 加法"能把 `s·P` 造出来，
且**总乘法次数最少**。Prime95 的实现：

- `lucas_cost(n, d)`（`ecm.cpp:2645`）：估计用参数 d 造出 n 的代价；
- `lucas_cost_several(n, &d)`（`ecm.cpp:2847`）：在 d 附近**开一个小窗口扫描**
  （`PRAC_SEARCH = 7`，即 ±3），取最省的 d —— **这就是"plan 搜索"的具体形态**；
- 代价单位是 **FFT 次数**（Prime95 的 gwnum 里一次大数乘法 = 若干 FFT），
  对应到我们就是 **madd 条数**；
- 早期 PRAC 里的一些特殊规则被 Prime95 **删掉**了（`ecm.cpp:2756/2790` 的注释），
  理由是"实测更少的 FFT / 最优链几乎不用这些规则"——说明这种搜索是**经验 + 测量驱动**的。

**关键性质（决定了我们能否自由选链）**：`[s]P` 是**唯一确定**的点，链只影响
**成本**与**中途状态**，不影响最终结果。所以：
- 用哪条链，**不影响**与参考实现的"结果对齐"验收（§4.3 的 28/28 就是这么来的）；
- 但它决定：每 bit/每步的成本（性能）、以及中途存档的记账方式（`bitnum` vs `prime index`）。

### 2.8 为什么第一阶段选二进 ladder（Q4 的结论）

| 链 | 相对成本 | 实现量 | 说明 |
|---|---|---|---|
| **二进 ladder（选定）** | 1.00 | 小 | 与本仓库 GPU 路径同算法（`kernels/opencl/ecm_stage1.cl:78-95`），SIMD 脚手架/测试可直接沿用；无分支、无发散 |
| 固定窗口 / Lucas 链 | ~0.80 | 中高 | 需要"已知差分点"的簿记（x-only 的硬约束） |
| PRAC（Prime95/gmp-ecm） | ~0.80–0.85 | 高 | 还要一个 plan 搜索器；好处是逐条指令与参考实现同源、中途存档可用 `prime index` |

结论：**先把 ladder 做对、拿到稳定基线与验收，再评估 windowing/PRAC**（列入 M4）。
注意一个容易误解的点：PRAC 的链**不造成 SIMD 发散**（整批 s 相同 ⇒ 计划相同），
推迟它的理由纯粹是**实现复杂度**，不是批处理不友好。

### 2.9 成本账（落到本仓库的 IFMA 域上）

记 radix 2^52、limb 数 `n = ⌈bits/52⌉`，则一次模乘

- Montgomery 域（CIOS）：`n(4n+3)` 条 `vpmadd52`；
- **Mersenne 折叠域**（`N = 2^k−1` 时自动启用，`src/cpu/simd_mont_ifma.cpp`）：模乘 `2n²`、
  模平方 `n(n−1)+2n`（对称平方 C2）。

于是"每 bit 成本"可以统一折算成 madd 条数来比较链的好坏：

```
ladder 每 bit = 6M + 4S
Montgomery 域 (n=58):  6·2n² + 4·2n²            = 10·6728 ≈ 67.3k madd/bit
折叠域     (n=58):     6·2n² + 4·(n(n−1)+2n)    ≈ 6·6728 + 4·3422 ≈ 54.0k madd/bit
```

对照：本仓库 Edwards（a=1，扩展坐标）在 w=12 字典下每 bit ≈ 4S+4M ≈ 8 个域运算，
这也是"为什么 Edwards 目前比裸 Montgomery ladder 快"的算账来源 ——
**Montgomery 侧的性能要在 M4 的链优化（或更好的坐标/公式）里找回来**。

### 2.10 名词表

| 名词 | 含义 |
|---|---|
| σ (sigma) | Suyama 参数化的随机参数（32-bit；本轮样本里是 unsigned 32-bit），由它唯一决定曲线与起点 |
| A / a24 | Montgomery 曲线系数 `A`；`a24=(A+2)/4` 是倍点公式里那个常数 |
| torsion（挠子群） | 曲线群里的固定小阶子群；Suyama = Z/12，param3 = Z/4 |
| `s` / 指数 / K | stage-1 的标量 `lcm(1..B1)`（PrMers 日志里叫 K） |
| xDBLADD | 把"倍点 + 差分加法"合成一步的例程，二进 ladder 每次循环调一次 |
| cswap | 条件交换（用掩码实现的无分支交换），让 ladder 不需要真分支 |
| Lucas 链 / PRAC | 适配 x-only 算术的加法链形式 / Peter Montgomery 的实用加法链搜索 |
| plan（计划） | 搜索结果：一条具体的"加倍 + 已知差分 Lucas 加法"序列 |
| D-multiple / base / diff | PRAC/阶段2 里"某个倍数 + 已知差分集合"的记账对象 |
| 归一化 | 把 `(X:Z)` 变成 `x = X/Z`（一次域逆），写进存档的 `X` 就是它 |

---

## 3. 参考实现（本机路径与已确认事实）

| 参考 | 本机路径 | 已确认事实 |
|---|---|---|
| Prime95 | `D:\code\GIMPS\p95v3106b01.source\ecm.cpp`（另有 `p95v3104b02.source`） | `ecm.cpp:1490` `sigma_type`：`1`=old-style Suyama（Montgomery curves `choose12`）、`0`=Atkin-Morain Edwards、`3`=GMP-ECM-param3；乘法链为 **PRAC/Lucas**（`lucas_cost` 2645、`lucas_mul` 2711、`lucas_cost_several` 2847，plan 表在 529 起） |
| gmp-ecm | `D:\code\GIMPS\gmp-ecm` | `param0` = Suyama σ 参数化（Z/12）；`param3` = batch 32-bits-D（Z/4），本仓库 GPU 路径用的就是 param3 |
| PrMers | `D:\code\GIMPS\prmers\prmers-windows_v4.18.08` | 含**真实存档样本**：`resume_p1277_ECM_TE_B1_10000.save`、`..._c00000N.p95`、`ecm2_te_m_1277_c0.ckpt`、`prmers.log`、`results.txt` |

### 3.1 PrMers 文本存档样本（原文照录，注意这是格式事实，不是猜测）

**单曲线结果 `.p95`**（475 B，一行）：

```
METHOD=ECM; SIGMA=628577301; B1=10000; N=2^1277-1; X=0x<hex>; CHECKSUM=4035705219;
PROGRAM=PrMers 4.18.08-alpha; X0=0x0; Y0=0x0; TIME=Fri Mar 27 00:03:58 2026;
```
（实际文件是一行，这里为可读性折行）

**多曲线 resume `.save`**（每曲线一行，**用 `A=` 代替 `SIGMA=`**）：

```
METHOD=ECM; B1=10000; N=2^1277-1; X=0x<hex>; A=<十进制大整数>; CHECKSUM=4233710978;
PROGRAM=PrMers 4.18.08-alpha; X0=0x0; Y0=0x0; TIME=Thu Mar 26 23:42:28 2026;
```

要点：
- `N=2^1277-1` 是**表达式**而非十进制展开；`X` 是 **hex**（`0x` 前缀），`A` 是**十进制**。
- `X0=0x0; Y0=0x0;` 恒为 0（这两位是 gmp-ecm 家族的历史遗留位，保留但不使用）。
- `CHECKSUM` 是 32-bit 校验（gmp-ecm/Prime95 家族用 `B1 · sigma · N · factor · param` 之类的
  模乘组合；本仓库已有同族实现，见 §7）。
- `PARAM=` 键在 PrMers 的 Suyama 行里**不出现**（而本仓库 param3 行会写 `PARAM=3`）——
  这是 sigma_type/param 语义在文本里的表达方式，需要按参考实现逐字段确认（§7 待办）。

### 3.2 本仓库已有的、可直接复用的部分

| 资产 | 位置 | 复用方式 |
|---|---|---|
| IFMA 域层（CIOS + Mersenne 折叠 + 对称平方） | `src/cpu/simd_mont_ifma.{h,cpp}` | **原样复用**（对 `N = 2^k−1` 自动启用折叠域，等价于省一次 `m*n` 归约行） |
| 批内 SoA 布局 / 字典批求逆 / checkpoint / resume | `src/cpu/simd_edwards.*` | 结构照搬，点运算换成 XZ |
| 文本存档 writer（同族格式） | `src/core/ecm_save.cpp:189` | 扩展为 Suyama 行（`PARAM=`/`SIGMA=`/`A=` 语义按 §7 定稿） |
| 批 ladder 先例 | `kernels/opencl/ecm_stage1.cl:78-95` | 现有 GPU 路径就是**二进 Montgomery ladder**（`double_add_v2` + cswap），是 SIMD 版最直接的算法先例 |
| 三后端互认回归 | `tools/test/test_agreement.ps1` | 直接扩成"标量 Suyama / SIMD-Mont / SIMD-折叠域"三后端逐字节互认 |

---

## 4. 数学与参数化（**已用参考实现的数据/源码钉死**）

### 4.1 已确认的常量（方法：读 Prime95 源码 + 用 gmp-ecm 二进制做实验 + PrMers 存档对拍）

| 事实 | 值 | 证据 |
|---|---|---|
| σ → 曲线 | `u = σ²−5`，`v = 4σ` | `ecm.cpp:4141-4144`（Prime95 `choose12`） |
| Montgomery A | `A = (v−u)³(3u+v) / (4u³v) − 2 (mod N)`，即 `A+2 = An/Ad`，`An=(v−u)³(3u+v)`、`Ad=4u³v` | 同上 4180-4190；**与 29 条 PrMers 存档的 `A=` 全部逐字节相同**；与 gmp-ecm 在 7 个 σ 上一致 |
| 起始点 | `(X : Z) = (u³ : v³)` | `ecm.cpp:4175/4177` 注释直接写明 "u^3 (also Montgomery starting x)"、"v^3 (also Montgomery starting z)"；并用 gmp-ecm 复现 |
| **gmp-ecm param0 的指数** | **`s = lcm(1..B1)`（无挠子群乘子）** | B1=2 时暴力枚举得唯一 `s=2`；再验证 `X_3=[3]X_2`、`X_5=[10]X_3`、`X_7=[7]X_5`、`X_11=[66]X_7`；最后 `x(lcm(1..5000)·P₀) == gmp-ecm 的 X` 精确成立 |
| **Prime95 Montgomery/choose12 的指数** | **`s = 12·lcm(1..B1)`**（2 多两次、3 多一次） | `ecm.cpp:7461-7463`：`/* Adjust the count of prime powers. I'm not sure, but I think choose12 means we should include 2 extra twos and 1 extra 3. */ if (stage1_prime <= 3) count += (prime==2) ? 2 : 1;` |
| 存档里的 `X` | **归一化后的 Montgomery x（Z=1）** | 用 x-only 差分加法验证 `[m]·X_5000 == X_10000`（`m=lcm(1..1e4)/lcm(1..5e3)`）成立 |
| PrMers 实际跑的 stage-1 | **twisted Edwards（不是 XZ）** | `prmers.log`：`Compute in Twisted Edwards Mode`、`curves ... twisted_edwards`，并把 `.p95` 交给 Prime95 stage 2 |

**⇒ 三个参考实现的 σ→曲线数学相同**（同一公式，已双向对拍）；**差异只在指数的常数**：
gmp-ecm 用 `lcm(1..B1)`，Prime95 的 choose12 路径额外乘 12。这个差异不影响命中概率的量级，
但**决定 `[s]P` 的具体点**，因此"逐点对齐"时必须显式选定采用哪一家的常数（见 §6 Q3）。

### 4.2 已确认的 Torsion 与概率（用户已核对）

- Suyama σ 参数化 ⇒ 曲线带 **Z/12** 有理点，有效除子 **D ≈ 22.97**；
- 与 PrMers `ECM_TE`（Prime95 `choose12` 生成的 twisted Edwards）通过
  **`A = 2(a+d)/(a−d)`** 双有理等价 ⇒ 群阶与挠子群相同；
- 对比本仓库 GPU 路径的 param3（Z/4，D≈7.6）：**有效除子 D 约 3×**（成功率比值远小于此，
  随位宽/B1 变化：B1=256 实测 1.30×–1.8×，详见 §19.5 表与 §2.1 的说明）。

### 4.3 已建立的验证仪器（本轮产物，可复用）

**M0/M1 已达成**：参考配方落地为 `tools/stat/suyama_mont_ref.py`（纯 Python、无依赖），并与
gmp-ecm 二进制做**逐点对拍：28/28 全部一致**（B1 ∈ {2,3,5,7,11,97,1000} × σ ∈ {12345,999,20260922,31415926}，
N = 2^1277−1）。**M1 标量路径**（`src/cpu/ecm_mont_cpu.{h,cpp}`）经
`tools/test/test_mont_gmp_oracle.ps1` **27 用例 PASS**。**M2 SIMD 内核**
（`src/cpu/simd_mont_curve.{h,cpp}`）经 `tools/bench/mont_simd_verify.cpp` 与标量路径逐 lane 一致。

```powershell
python tools\stat\suyama_mont_ref.py --check-gmp-ecm                    # 28/28
powershell -File tools\test\test_mont_gmp_oracle.ps1                     # 27 用例 PASS
tools\build_tool.bat tools\bench\mont_simd_verify.cpp src\cpu\simd_mont_curve.cpp src\cpu\simd_mont_ifma.cpp src\cpu\ecm_mont_cpu.cpp
build_vs18\tools\mont_simd_verify.exe <N> <B1> <sigma0> 8 1             # SIMD vs 标量，含计时
```

| 仪器 | 作用 | 位置 |
|---|---|---|
| 参考配方 + oracle harness | σ→A、起点、指数、XZ ladder、命中判定、文本存档行；对 gmp-ecm 28/28 | `tools/stat/suyama_mont_ref.py` |
| M1 验收（标量 vs gmp-ecm） | 27 用例：归一化 x 逐字节 + 命中判定（口径见上） | `tools/test/test_mont_gmp_oracle.ps1` |
| M2 验收（SIMD vs 标量） | 逐 lane 判定 + x 逐字节 + 批/标量计时 | `tools/bench/mont_simd_verify.cpp` |
| 单曲线 dump | 打印 A / s_bits / X0,Z0 / x / Z / gcd，便于与参考实现 diff | `tools/bench/mont_standalone.cpp` |
| gmp-ecm 三向对拍 | `ecm.exe -param 0 -sigma σ -save f <B1> <B1>`；`-A/-x0` 可注入**我们自己的**曲线与起点，用来单独钉指数约定 | `D:\code\GIMPS\gmp-ecm\ecm-7.0.5-znver3\ecm.exe` |
| σ→A 对拍 | 与 PrMers 29 条存档 + gmp-ecm 7 个 σ 一致 | `.bench_tmp/prmers_forensics.py`、`sigma_map_check.py` |
| 指数约定 | B1=2 暴力枚举 + 跨 B1 比值链（B1=11→5000） | `.bench_tmp/exponent_structure.py`、`exponent_ratio.py` |
| ladder 自检 | 素数模数下与朴素全坐标 double-and-add 对拍（19/19） | `.bench_tmp/ladder_selftest.py` |

> **踩过的坑（写下来免得重犯）**：差分加法 `xADD` 里那个"已知差分点"必须用**仿射 x**
> （`X/Z`），不能直接把 `u³` 当差分 x 用（差分点的 Z=v³）。我最初把所有约 200 个组合都
> 判成"不匹配"，就是因为这一个错误 —— 归一化之后同一个测试立刻全部通过。
> 这也解释了为什么不变量/统计类测试没抓到：错的是**差分点的尺度**，曲线本身仍然对。
>
> **第二个坑（对拍方法学，务必照做）**：**参考实现必须只跑 stage-1**。
> gmp-ecm 默认还会跑 stage-2，一旦它在 step 2 命中，写进存档的 `X` 就**不再是 stage-1 的点**
> （实测是一个退化的极小值，如 `0x…a3`），此时任何"逐点对拍"都会假失败 —— 我在 M3001
> 上就被这个坑骗了一轮。正确调用是**显式给 `B2 = B1`**：
>
> ```
> ecm.exe -param 0 -sigma σ -c 1 -save f <B1> <B1>      # B2=B1 => 只跑 stage 1
> ```
>
> 本轮的两个 harness（`tools/test/test_mont_gmp_oracle.ps1`、
> `tools/stat/suyama_mont_ref.py --check-gmp-ecm`）都已经按这个方式调用。
>
> **第三个坑（参考实现的指数策略不可完全控制）**：gmp-ecm 在**小模数**上会用比 `lcm(1..B1)`
> 更大的指数。实测 `p20×(2^101−1)`、σ=12345、B1=1000：它在 step 1 命中，而该命中实际需要
> **41² = 1681**，即它的有效指数 ⊇ `lcm(1..1681)`。`-I 0` 被 gmp-ecm 拒绝（"requires f > 0"），
> 所以这一点关不掉。因此对拍口径定为三条：
>
> 1. **我们命中 ⇒ 参考必须也命中**（我们不允许漏掉真实的 stage-1 命中）；
> 2. **两边都不命中 ⇒ 归一化 x 逐字节相同**（26/27 个用例的内容就是这条）；
> 3. **只有参考命中 ⇒ 归类为"指数策略差异"，不算失败**（不是我们的 bug）。
>
> **第四个坑（别忘了它的存在）**：`mpz_invert` 在命中时必然失败（Z 不可逆），所以"归一化 x"
> 只在没命中时才有意义 —— 两边都要检查这一点再决定是否比较。

---

## 5. 里程碑（草案，随决策更新）

| # | 里程碑 | 交付 | 验收 |
|---|---|---|---|
| **M1** | 标量 Suyama-Montgomery XZ stage-1（mpz + MPN 域层） | ✅ **已完成**：`src/cpu/ecm_mont_cpu.{h,cpp}`（`mont_build_s` / `mont_suyama_curve` / `mont_stage1_curve[_x]`，域运算复用通用 MPN 层 `ecm_edwards_mont.h`）；工具 `tools/bench/mont_standalone.cpp` | ✅ `tools/test/test_mont_gmp_oracle.ps1`：**27 用例 PASS**（M1277 四个 B1、M3001、M4003、`p20×q101`）× 3 个 σ；✅ `tools/stat/suyama_mont_ref.py --check-gmp-ecm`：**28/28**；✅ 与独立 Python 参考在 M3001 上逐位相同 |
| M2 | SIMD 批（8 曲线，AVX-512 IFMA） | ✅ **内核已完成**：`src/cpu/simd_mont_curve.{h,cpp}`（SoA `(X:Z)` ladder，`xz_dbl`/`xz_add` 各 3M+2S；因整批共享 `s`，位分支对所有 lane 相同 ⇒ **无发散、连 cswap 都不需要**）；验证工具 `tools/bench/mont_simd_verify.cpp` | ✅ 与标量路径**逐 lane 一致**：M1277 B1=2/97/1000、M3001 B1=1000/10000、M4003 B1=1000/10000 全部 `PASS (0/8 lanes disagree)`；批 vs 8 次标量 **5.1–5.5×**（M3001/M4003，B1=1e3~1e4） |
| M2b | 驱动接线（Q7=(a)） | ✅ **已打通**：`--mont` / `--mont-backend auto\|simd\|gmp` / `--mont-torsion 1\|12`；`run_mont_stage1()` 走 8-lane SIMD 批（`gmp` 或曲线数 < 2 时回退标量）；σ **64-bit**（`-sigma` 给定则 σ+i，否则随机）；**一个任务一个共享存档** `e%07<bits>_B<B1>.save` | ✅ 实测 `--mont --mont-backend simd -gpucurves 8 -sigma 12345 --tmp-dir … 1000 0` → 8 行共享存档、`wall=0.01s`，第 1 行 X 与 gmp-ecm param0 **逐字节相同** |
| M2c | `--mont-threads`（多批并行）与 ini `mont*`（队列模式） | ✅ **已完成**（2026-09-24，见 §10）：`--mont-threads <n>`（0=auto、1=串行）+ ini `mont` / `mont_backend` / `mont_torsion` / `mont_threads` / `mont_save_pattern`；原子任务计数分发（任务 = 1 个 8 曲线 SIMD 批 或 1 条标量曲线），每线程独立 `mont_soa_ctx_t` 与 scratch；命中行改为**join 后按曲线号顺序打印**（线程数不再影响输出） | ✅ 64 曲线(M1277,B1=1e5)：1/4/8/16 worker = 8.61/1.97/1.30/1.30 s ⇒ **4.37× / 6.62×**，16 worker 与 8 相同（任务数 8 是上界，符合设计）；✅ 1 worker 与 8 worker 的共享存档**除 `TIME=`（人类可读时间戳）外逐字节相同**；✅ oracle 27 用例复跑 PASS |
| M3 | 文本存档统一 | ✅ **writer 已落地且互通**（§9.2/§9.3）；reader（本仓库 resume）⏳ 待做 | ✅ `ecm -resume` 接受我们的行（exit 0，无 CHECKSUM 报错）；reader 待补 |
| M4 | 性能与链优化/缓存 | **M4a ✅**（`s` 位数组化，§10）；**M4b ✗ 已关闭**（EFD `ladd-1987-m-3` = `dbl`+`dadd` 之和，无子表达式共享，§7.1）；**M4e ✅**（finish 优化，端到端 −7~8%，§11.6）；**M4f ✅**（辅助 SoA pass 30→18 趟/bit，ladder −8%，合计 −14.5%，§11.3b）；**M4c ✗ 已关闭**（PRAC/Lucas 链实测 +3~4%，§12）；**M4d ⏳ 低优先**（`s`/链表缓存；实测任务固定开销仅 2.1%，§11.4） | M4a：oracle 27/27 + 逐 lane PASS；M4e/M4f：交替 A/B −6.9~−14.5%、存档逐字节一致 |

---

## 6. 决策记录（grilling 议程）

> 每条格式：**问题 → 候选 → 结论/推荐 → 理由**。已被代码/数据回答的直接标注来源。

- **Q1（已定，用户 2026-09-23）验收口径 = `(a)` 结果对齐**：同 `(σ, B1, N)` 下我们的
  **归一化 Montgomery x** 与 **gmp-ecm param0** 的 `X` 逐字节一致；PrMers 样本用于核对 σ→A 与 `A=` 字段；
  twisted Edwards 侧（PrMers `ECM_TE` 实际跑的 stage-1）**本轮不做**。
  ⇒ 由此**推出 Q3 的取值**：必须跟随 **gmp-ecm 的 `s = lcm(1..B1)`**（而不是 Prime95 的 `12·lcm`），
  否则 `[s]P` 不同、逐点对齐无意义。Prime95 的常数保留为**可选模式**（将来若要对齐 Prime95/PrMers 的
  `X` 再加开关），实现上做成一个常量参数即可。
- **Q2（已定，数据）** σ→曲线的精确公式与起点：`u=σ²−5`、`v=4σ`、
  `A=(v−u)³(3u+v)/(4u³v)−2`、起点 `(u³:v³)`。29 条 PrMers 存档 + gmp-ecm 7 个 σ 双向一致。
- **Q3（已定，数据+源码）** stage-1 指数：**两个参考实现不一致** ——
  gmp-ecm param0 = `lcm(1..B1)`（已用 B1=2 暴力枚举 + 跨 B1 比值链 + `x(lcm(1..5000)P₀)==X` 钉死）；
  Prime95 choose12 = `12·lcm(1..B1)`（源码注释 + 计数调整逻辑）。**需要用户选定我们要跟随哪一家**
  （我推荐 **Prime95 的 12·lcm**：它是 GIMPS 侧的事实标准、且与 PrMers 的 K 构造同源，
  逐点对齐 PrMers/Prime95 时才有意义；gmp-ecm 的 `lcm` 只作为工具链里的对拍参考）。
- **Q4（已定，用户 2026-09-23）** 乘法链：**先做二进 Montgomery ladder**（xDBLADD 一步/bit + cswap），
  与 `kernels/opencl/ecm_stage1.cl` 的现有批处理先例同算法；checkpoint 用 `bitnum` + `(X:Z)`。
  windowed / PRAC 推迟到 **M4**（可选优化，相对成本 ~0.80–0.85，但需要 plan 搜索器与"已知差分点"簿记）。
  背景与理由见 §2.7/§2.8。注意：PRAC 的链**不会**造成 SIMD 发散（整批 s 相同 ⇒ 计划相同），
  推迟的理由是**实现复杂度**，不是批处理不友好。
- **Q5（待定）** 批宽与 bit 段：沿用 8 曲线/lane（与 Edwards 一致）。
- **Q6（已定，数据）** 存档 `X` 的语义：归一化 Montgomery x（Z=1）✓ 已用 gmp-ecm 自身两个
  存档验证 `[m]X_5000 == X_10000`。
- **Q7（待定）** 驱动接线：`--method`/ini/worktodo 开关与 `sigma_type=1` 的表达（文本行是否写 `PARAM=`）。
- **Q8（已定，用户 2026-09-23）** 存档格式：**照 gmp-ecm 7.0.6 param0 的真实输出定稿**，见 §9。
  要点：param 0 省略 `PARAM=` 键；`SIGMA=` 是 **64-bit**；`X=` 写归一化 Montgomery x（hex + `0x`）；
  `X0=0x0; Y0=0x0;` 保留；`WHO=`/`TIME=` 与本仓库 param3 writer 同款；CHECKSUM 复用同族实现。
  命中曲线写因子而不是半截 x。`.ckpt` 暂保持二进制。
- **Q9（已定，用户 2026-09-23）** 范围 = **两条路径一起做**：M1 同时落地 **mpz 标量 + SIMD 批**，
  两边用**同一条二进 ladder 与同一套公式**，目标是 `.tmp` 存档**逐字节互认**（仓库内最强回归，
  不依赖外部二进制）。标量路径**尽量复用 mpn**：`src/cpu/ecm_edwards_mont.h` 其实是一个**通用**
  mpn Montgomery 域层（`mont_ctx_t` + `mont_mul/mont_sqr` 走 GMP 的 `redc_1`/`redc_n`，
  且今天刚修掉它的非规范 bug），直接复用它即可拿到 mpn 级性能，无需自研。

### 6.1 Q1 重写：验收对标哪一条语义？

本轮发现把原来的问题变得具体了，因为**三家的"stage-1 输出"并不是同一种东西**：

| 语义 | 产出 | 现成 oracle | 我们的实现代价 |
|---|---|---|---|
| **A. Montgomery XZ**（用户要求的"Montgomery curves"） | `(X:Z)` 归一化后的 Montgomery x | **gmp-ecm param0**（可无限生成：`ecm -param 0 -sigma σ -save f B1`）；约定已全部钉死（§4.1） | 中等：XZ ladder + 我们已有 IFMA 域层 |
| **B. twisted Edwards**（PrMers `ECM_TE` 实际跑的） | Edwards 侧点 → `ed_to_Montgomery` 的 `Qx`（Prime95 MIDSTAGE 语义） | **PrMers 存档样本**（29 条 `(σ,A,X)`，M1277） | 更大：需要 choose12 的 twisted Edwards 曲线与算术（a≠1，非我们现有 a=1 内核） |

**候选**
- **(a) 只做 A**：以 gmp-ecm param0 为 oracle 做逐点对齐；PrMers 样本只用来核对 σ→A 与 `A=` 字段。
- **(b) 只做 B**：直接复刻 PrMers/Prime95 的 twisted Edwards stage-1，逐点对齐 `.p95`。
- **(c) A 先做、B 后做**（推荐）：先把 Montgomery XZ 的 SIMD 版本落地并用 gmp-ecm 逐点验收（约定已钉死，风险最低），
  之后若确实要做 ECMSTAGE2 交接/与 PrMers 逐点对齐，再补 twisted Edwards 侧（或证明两者通过
  `u=(z+y)/(z−y)` 映射后一致 —— 目前证据显示**不一定一致**，因为 stage-1 的 `s` 常数不同）。

**我的推荐：Q1 = (c)，并且把"A 的对齐口径"写成硬指标**：
同 `(σ, B1, N)` 下我们的归一化 Montgomery x 与 gmp-ecm param0 的 `X` **逐字节一致**；
同时用 PrMers 存档核对 `A=` 字段与 σ 解析；命中率与 `tools/ecm_prob` 模型一致作为统计层。
理由：(i) 约定已全部钉死，这是唯一"能立刻逐点验收"的目标；(ii) 它正是用户要的 Montgomery curves；
(iii) B 的工作量更大且会牵动曲线族抽象，等 A 落地后再评估更划算。

---

## 7. 补充问答：EFD 成本口径差异 & Lucas 链的构造与复用

### 7.1 为什么 EFD 的公式/成本与本文 §2.4 略有不同？

参考：<https://hyperelliptic.org/EFD/g1p/auto-montgom-xz.html>（Bernstein–Lange 的显式公式数据库）。
结论：**两者算的是同一件事，差异来自"计什么、怎么计"，其中一小部分是真能拿到的优化**。分四类：

**(1) 计数约定（工程口径，不是数学差异）**

- EFD 明确区分 `M`（一般乘法）、`S`（平方）、以及**乘曲线常数**（如 `(A+2)/4`）。有些条目把
  "乘一个小常数"算作 `1M`，有些在特定曲线（例如 Curve25519 的 `a24=121665`）下把它当作
  "少量移位加法"而近似免费 —— **这是曲线/实现特定的，不是通用的**。
- EFD 的"总成本"有时写成 **M-等价**（例如 `5M` 表示 `3M+2S`，其中 `S≈0.8M`），而本文 §2.4 是
  **分项列 M 与 S**。看起来不同，换算后一致。
- 两边都假设 ×2/×4 是移位（免费）、`B=1`、`(A+2)/4` 已预计算 ✓（我们也是这么做的）。

**(2) 我们的 §2.4 是"两次独立调用"的成本 —— 但 EFD 的最佳 ladder 就是这两次之和（2026-09-24 更正）**

§2.4 的 3M+2S + 3M+2S = 6M+4S 看起来像"先调一次 `xDBL`、再调一次 `xADD`"的**朴素**口径。曾经据此推断
EFD 的 `mladd`（combined ladder step）能把两者**共享子表达式**、省掉约 **1M/bit（≈15%）**。
**这个推断是错的**，已按 EFD 原文（上引页面）逐条核对：

- EFD 的最优 ladder 条目 `ladder-ladd-1987-m-3` 成本 = **6M + 4S + 1·a24**；
- 它恰好等于该页 `doubling-dbl-1987-m-3`（**2M + 2S + 1·a24**）+ `diffadd-dadd-1987-m-3`（**4M + 2S**）之和
  ⇒ **ladder 条目本身就是"dbl 与 diffadd 的拼接"，不存在跨步骤的公共子表达式**（若有共享，EFD 的
  ladder 成本会低于两项之和，而它并没有）。
- 我们每 bit = `dbl 3M+2S` + `add 3M+2S`，其中 `dbl` 的 `3M` 含 `a24·E` 一次乘法（= EFD 记作 `1·a24` 的那次），
  `add` 走 **Z1=1（仿射差分）** 分支 `mdadd-1987-m` = 3M+2S（EFD 的通用 `dadd` 要 4M）。
  ⇒ 我们实际是 **6M + 4S**（若把 `a24` 记成独立项则是 `5M+4S+1·a24`），**与 EFD 最优条目同价或更省**。
- 因此 **M4b（合并 `xz_dbladd`）收益为 0，已关闭**；per-bit 的域运算条数没有再压缩的空间。

**(4) 有限域层面：`S` 的钱我们已经在赚（核对代码后确认）**

- `ifma_mont_mul` / `ifma_mont_sqr` 会按域分派：`N = 2^k−1` 时走 `ifma_mersenne_mul` / **`ifma_mersenne_sqr`**
  （`src/cpu/simd_mont_ifma.cpp:644,729`）。后者只算严格上三角：`n(n−1)+2n` madds，**≈ 一般乘法的一半**
  ⇒ 每 bit 的 4 次平方在折叠域里真的按 ≈0.5M 计价。
- 非折叠域（普通 N）走 `ifma_cios`，此时 `sqr` 退化为 `mul(a,a)`，`S = M` ⇒ 换域才有收益，
  不是 `xz_dbladd` 能解决的问题。

**M4 结论（2026-09-24）**：per-bit 已到 EFD 最优，`S` 折扣已在折叠域拿到。剩下的真实杠杆只有
**(a) 步数（链，§7.2/§8）** 与 **(b) 并行度/批（§10）**。

### 7.2 Lucas/PRAC 链的构造取决于什么？占比多少？能否复用？

**取决于**：只有**指数 `s`**（因而只与 `B1` 和挠子群常数有关）。

- **与 N 无关**：链是"用加法/加倍把整数 s 造出来"的调度，不含任何模数信息；
- **与 σ 无关**：σ 只决定曲线与起点，不决定调度；
- ⇒ **同一次运行的 8 条 lane、以及所有批次，用的是同一条链**；换 N（不同曲线）也**完全一样**。
  这也是我们在 §2.8 说的"PRAC 不会造成 SIMD 发散"的根据：整批共享 s ⇒ 共享计划。

**占比**：**计划构造 ≈ 0**，链执行 ≈ 100%。

- 构造 = `lucas_cost_several`（Prime95）在 `d` 附近开 `PRAC_SEARCH=7` 的窗口、对每个候选算一次
  代价函数，量级 `O(7·log₂ s)` 次**整数**运算；
- 以 B1=1e5（`s≈144k bit`，阶梯数秒级）为例，构造是**微秒级**，占比 **<0.1%**；
- 真正的时间全在"按计划执行的那几百万次域运算"上。我们自己的实现里，`mont_build_s`（lcm 筛）
  同理是微秒级，可以忽略。

**复用可能（分三层，价值递减）**：

1. **批内/批间复用（必做，已天然成立）**：链（或我们现在的二进 ladder 的位序列）由 s 决定 ⇒
   一次构造、所有 lane 与所有批次共用 ✓。我们当前实现里 `mpz_tstbit(s, i)` 每位都在查 s —— 
   可优化成"先把 s 展开成 bit 数组/`s_bits`"（像本仓库 GPU 路径 `ecm_stage1.cl` 的做法）✓，
   省掉热循环里的 `mpz` 调用（M4 小项）。
2. **跨运行复用（可缓存，价值中等）**：计划/位序列是 `(B1, torsion)` 的纯函数 ⇒ 可落盘缓存。
   但对一次真实运行（秒级以上）意义不大，除非是极短任务。
3. **预计算倍数（不可跨曲线复用）**：PRAC 的 base/diff 集、以及我们的 `xdiff`，都依赖**具体点 P**
   （即 σ）⇒ 每曲线一份，不能跨曲线共享；能共享的只有**调度**本身。

## 8. 多年期分布式项目：链的缓存与优化计划

**工作负载画像**（用户说明）：**同一个 B1**，海量不同的 N 与 σ，持续数年、跨大量机器。
这把 §7.2 的结论变成工程重点：链只依赖 `s`（即只依赖 `B1` 与 torsion 常数）⇒ **整个项目共用
一份链工件**，一次离线生成、长期复用。

### 8.1 参考实现的现成机制（`.refactor/ecm`）

- `README:653-658`：仅 `-param 0`，stage 1 对每个素数 `p ∈ [11, B1]`，用**从文件读出的 64-bit code**
  重建**近最优 Lucas 链**，替代 `prac` 现场生成的链；
- `LucasChainGenerator/README`：`LucasChainGen -B1 <B1> -nT <threads>` 离线生成 `Lchain_codes.dat`；
  **B1=1e6 ≈ 19 s、11e6 ≈ 7.5 min、43e6 ≈ 47 min、260e6 ≈ 9.7 h、850e6 ≈ 45 h**；
  运行时同目录自动加载；`--enable-assert` + `make check` 校验。
- 执行模型注意：gmp-ecm 是**逐素数**乘（每个 p 一条链），最终点仍是 `[s]P`（我们已逐字节对拍证明，
  见 §4.1/§4.3）⇒ **链只影响成本，不影响结果**，因此执行引擎可以自由替换 ✓。

### 8.2 可缓存的三样东西（按性价比排序）

| # | 缓存物 | 键 | 生成成本 | 收益 | 结论 |
|---|---|---|---|---|---|
| 1 | **`s` 的位数组**（`s = torsion·lcm(1..B1)` 展开成 bit 数组/文件） | (B1, torsion) | 毫秒（lcm 筛） | 去掉热循环里的 `mpz_tstbit`（每 bit 一次 mpz 调用） | **必做**，小改动大收益（阶梯越短越明显） |
| 2 | **逐素数链表**（`p → chain code`，即 `Lchain_codes.dat` 思路） | (B1, torsion, 链族/版本) | 秒~天（离线） | 相对 `prac` 现场搜索的近最优链，省约 10~20% 域运算；**多年期项目一次摊销** | **值得做**，M4 第二阶段 |
| 3 | 完整"直程序"（把 1+2 编成算子级调度表） | 同上 | 同 2 | 再省分支/装载 | 观察项 |

**不可缓存**：`xdiff` 与 PRAC 的 base/diff 倍数集 —— 都依赖**具体点 P**（即 σ），每曲线一份 ✓（§7.2）。

### 8.3 与 SIMD 批的关系（为什么缓存对我们更划算）

- 整批 8 条 lane 共享同一个 `s` ⇒ **共享同一张链表/位数组**；逐素数链表同样 lane-uniform ✓
  ⇒ 缓存带来的调度规则完全不破坏 SoA（无发散问题）；
- 每 bit 的域运算是 IFMA 批量 ⇒ "省一次乘法"是 **8 条曲线同时省**，比单曲线实现更值钱；
- 所以**链优化（M4）与缓存应一起做**：先定链族 → 离线生成表 → 让 `--mont` 直接吃这个表。

### 8.4 计划（M4 细化）

1. ✅ **M4a（已完成，2026-09-24）**：`s` 位数组化 —— `mont_expand_bits()` 把 `s` 一次展开成
   **MSB-first 的 `uint8_t[]`**（`bits[0]` = 最高置位，与 ladder 起点 `R1=2R0` 对应），
   经由新入口 `mont_stage1_curve_bits[_x]()` / `mont_soa_stage1_bits()` 进入热循环；
   旧的 mpz 入口保留为薄包装（先展开再调用），所以工具与测试无需改动。
   位数组**只依赖 (B1, torsion)**，因此每个任务只建一次、被所有 curve/batch/线程共享（见 §8.2 第 1 条）。
   实测：这一步本身在 SIMD 路径里微不足道（每 bit 一次 `mpz_tstbit` ≈ 几 ns，B1=1e5 共 14.4 万 bit ≈ 亚毫秒），
   价值主要在于**把"指数只依赖 (B1,torsion)"这件事变成代码事实**，为 M4c/M4d 的缓存铺路。
2. ✗ **M4b 已关闭**：`xz_dbladd` 合并**无收益** —— EFD 最优 ladder 条目就等于 `dbl` + `dadd` 之和，
   不存在共享子表达式；我们当前 per-bit = **6M+4S**（`add` 走仿射差分分支），已在最优档。证据见 §7.1(2)。
3. ✗ **M4c 已关闭（2026-09-24）**：逐素数 Lucas 链 / PRAC 移植**不划算** —— 链把每 bit 的原语数从 2.0
   降到 1.70，但每个原语平均贵 18%（链的差是射影点，加法 5.26M vs 我们 ladder 的 4.26M），
   总账 **+3~4%**（照抄参考实现的加法则 +32%）。证据与"何时会翻转"见 §12；原型 `tools/stat/prac_cost.py`。
   M4d（缓存文件格式）随之降为**低优先**：实测任务固定开销仅 2.1%（§11.4），只有 B1 ≳ 1e7
   或"每任务曲线数极少"时才值得做。
5. 每一步都跑：`mont_simd_verify`（SIMD vs 标量逐 lane）+ `test_mont_gmp_oracle`（对 gmp-ecm 逐点）。

---

## 9. 存档格式（Suyama/Montgomery，**已定稿**）

### 9.1 判据：gmp-ecm 7.0.6 的真实输出（用户提供，M3001/B1=1e5，10 条曲线）

`D:\code\GIMPS\gmp-ecm\ecm-2025.10.28-win.multiarch\3001_B1e5.save`，每曲线一行：

```
METHOD=ECM; SIGMA=2565338038635275335; B1=100000; N=(2^3001-1)/17177616768358031;
X=0x48c3ab17821579f85eca6192...; CHECKSUM=670349301; PROGRAM=GMP-ECM 7.0.6;
X0=0x0; Y0=0x0; WHO=Elysia@LAPTOP-4DQID519; TIME=Wed Sep 23 23:17:10 2026;
```

要点（与 PrMers 文本样本一致，**同一个家族**）：

| 字段 | 语义 / 我们的取值 |
|---|---|
| `METHOD=ECM` | 固定 |
| `PARAM=` | **param 0（Suyama σ）时参考实现省略该键**（本仓库 param3 writer 会写 `PARAM=3`）。决定：**我们跟随参考——省略**；reader 两种都接受 |
| `SIGMA=` | **十进制 64-bit**（★注意：gmp-ecm 自动生成的 σ 超过 32 位，样本是 2.5e18 ≈ 2^61）。我们的实现按 `uint64_t` 处理 ✓ |
| `B1=` | stage-1 界（整数） |
| `N=` | 数字/表达式；参考实现会把已找到的因子除出去写成 `(2^3001-1)/17177616768358031`。我们写调用者给的表达式，保持可解析 |
| `X=` | **归一化 Montgomery x**（hex，带 `0x` 前缀），即 `X/Z mod N`，与 gmp-ecm/PrMers 同义 ✓ |
| `CHECKSUM=` | 32-bit；同族公式（本仓库 param3 writer 已有实现与 Prime95 真档逐字节对拍的先例，直接复用同一实现） |
| `PROGRAM=` | 写我们自己（例：`PROGRAM=ECM-ELY`）；reference 写 `GMP-ECM 7.0.6` |
| `X0=0x0; Y0=0x0;` | 家族历史遗留位，保留恒 0 |
| `WHO=` | 可选 `user@host`（本仓库已有 `build_who_field()`） |
| `TIME=` | `%a %b %d %H:%M:%S %Y`（ctime 风格，与本仓库 param3 writer 相同） |

**命中曲线怎么落盘**：命中时 `Z ≡ 0 (mod p)`，归一化 x 无意义；gmp-ecm 的样例正好展示了它的做法——
把已找到的因子从 `N=` 里除出去继续跑。我们的处理：**命中曲线的 `.tmp` 写因子**（沿用本仓库既有约定），
并在日志里标明命中曲线号；不写半截的 x。

### 9.2 多曲线共用存档（用户 2026-09-23 的要求）

**要求**：同一次运行里 `N`、`B1` 相同的多条曲线**共用一个存档**，不再一条曲线一个文件。

**结论**：参考实现就是这个形态 —— gmp-ecm 的 `3001_B1e5.save` 是**一个文件 10 行**（每曲线一行）。
因此我们采用：

- **一任务一文件**，命名对齐参考：`<tmp_dir>/<n_stem>_B<B1>.save`（例：`3001_B1e5.save`）；
- **每曲线一行，且行内自包含**（每行都重复 `METHOD/B1/N/SIGMA/X/CHECKSUM/PROGRAM/X0/Y0/WHO/TIME`）；
- **为什么不把公共字段提到文件头**：参考实现的 reader 是**逐行解析**的，`-resume` 会把 `B1` 当作参数、
  从行里读 `N/SIGMA/X/CHECKSUM`。抽走公共字段会直接破坏 `ecm -resume` 的兼容性 —— 而这一条
  我们已经用实测验证过（§9.3），不能为了少写几个字节把它丢掉。

**已落地验证**（`tools/bench/mont_simd_verify.cpp` 新增可选存档参数，一次 SIMD 批写 8 行）：

```
mont_simd_verify.exe <N> 1000 12345 8 1 shared.save
  -> TIMING: simd batch(8) = 0.0088 s   scalar x8 = 0.0364 s   speedup = 4.16x
  -> shared save file: written (8 curve lines)          RESULT: PASS (0/8 lanes disagree)

shared.save:  lines=8
  SIGMA=12345  N=260198304866...  X=0x1475666af371...
  SIGMA=12346  N=260198304866...  X=0x1c94eaaf957c...
  ...                                    (同一 N、同一 B1、σ 递增，命中曲线该行写因子)

# 共用文件里任意一行都能被参考实现续跑：
ecm.exe -resume line2.save 1000 50000   -> exit 0, "sigma=0:12346", Step 1/2 正常
```

⇒ "多曲线共用存档"与"可被参考实现续跑"**同时成立** ✓。

### 9.3 与既有实现的关系（不重复造轮子）

本仓库 `src/core/ecm_save.cpp` 已经有一个**同族**的文本 writer（param3）+ `ecm_edwards_save.cpp`
的 Prime95 二进制读写（`.ckpt`）。M3 的落地方式与**已完成部分**：

1. ✅ **已落地**：`src/core/ecm_save.{h,cpp}` 新增 `ecm_append_save_lines_mont(...)` —— 沿用 param3
   writer 的字段顺序与 CHECKSUM 实现，三处按 §9.1 差异化：**64-bit SIGMA**（两半拼装，避开
   Windows `unsigned long` 只有 32 位的坑）、**param 0 不写 `PARAM=`**、命中曲线写**因子**而非半截 x。
2. ✅ **互通验证通过**：`tools/bench/mont_standalone.exe <N> <σ> <B1> 1 <file>` 生成一行，交给 gmp-ecm：

   ```
   ecm.exe -resume .bench_tmp\our_mont.save 1000 50000
     -> Resuming ECM residue saved by ... with ECM-ELY on ...
     -> Using B1=1000-1000, B2=50000, polynomial x^1, sigma=0:12345
     -> Step 1 took 0ms / Step 2 took 0ms        (exit 0，无 CHECKSUM 报错)
   ```

   ⇒ 字段集与 CHECKSUM **被参考实现接受**，可直接作为 ECMSTAGE2 输入 ✓。
   （注意 `ecm -resume` 必须跟 B1/B2 位置参数，否则报 "Invalid arguments"。）
3. ⏳ 待做：`.ckpt`（本仓库自有的中途存档）**先保持二进制不动**；`--mont` 的 resume 语义按
   "从 `.ckpt` 恢复 (bitnum, X:Z)"设计（与 Edwards 的 `bitnum` 方案同构）。

### 9.4 性能参考（用户实测，gmp-ecm 7.0.6 zen5，单线程，B2=0）

M3001、B1=1e5、10 条曲线：**Step 1 每条 ~718–735 ms（≈0.72 s/curve）**，其中 2 条在 step 1 命中
（17 位合因子 17177616768358031、15 位素因子 445373542756127）。

对照我们本轮的数据（M3001，B1=1e4，8 曲线批）：SIMD 批 0.32 s、标量 8 条 1.64 s ⇒
**每曲线等效 0.040 s（SIMD）/ 0.205 s（标量）**；按 B1 线性外推到 B1=1e5 约为
**0.40 s/curve（SIMD）/ 2.1 s/curve（标量）** ⇒ SIMD 大致是同口径 gmp-ecm 的 **~1.8× 快**、
标量约慢 3×。（这是粗外推；正式对比列入 M4，需同 B1、同 B2、交替 A/B。）

上面这条外推**已在 §10.4 用同机实测替换**（M1277/B1=1e5，1e5~1e6 是用户指定的测试区间）。

---

## 10. 并行度：`--mont-threads`（M2c）与实测基线（2026-09-24）

### 10.1 为什么并行度的单位是"任务"而不是"曲线"

SIMD 路径一次算 **8 条曲线**（SoA 8 lane，共享同一个 `s`），所以：
**1 个任务 = 1 个 8 曲线批**，标量路径则 **1 个任务 = 1 条曲线**。
⇒ 并行度上界是 `ceil(curves/8)`（SIMD）或 `curves`（标量）：**8 条曲线的 simd 任务无论多少核都只有 1 个线程在干活**。
驱动因此显式打印并自我约束：

```
stage1 threads  : 8 worker(s) x 8 task(s) of 8 curves      # 64 曲线，--mont-threads 8
```

`mont_default_threads(curves, use_simd)`（`src/core/ecm_driver.cpp`）取 `min(核数, 任务数)`，
用户给的 `--mont-threads n` 同样被 `任务数` 夹住（否则多出来的线程只会抢带宽）。

### 10.2 实现要点（为什么这次改动很小）

- **任务分发**：`std::atomic<uint32_t> next_task` + `fetch_add`，worker 干完一个批就领下一个；
  批成本均匀 ⇒ 与静态划分等价，但尾批/异常时更稳。
- **每线程独立状态**：一个 `mont_soa_ctx_t`（含自己的 22 元素 scratch pool + 折叠域预计算）
  `+` 自己的 `bx/bg` 结果缓冲。共享的只有**只读**的 `bits[]`、`sigmas[]`、`N` 与本线程独占写的
  `xs[]/hit[]/factors[]` 下标 ⇒ 无锁、无 false sharing（结果元素是 `mpz_t` 结构，不是热数据）。
- **命中输出改为 join 后按曲线号打印**：此前在循环里打印，多线程下会交错；现在**输出与线程数无关**，
  任何解析 stdout 的脚本/自检都稳定。
- **失败传播**：worker 里 `mont_soa_init` 失败 → `atomic<int> init_failed`，join 后统一清理并返回 `ECM_ERROR`。
- **亲核性**：沿用既有 `apply_thread_affinity(opt.affinity_cpus, t)`（与 Edwards 路径同一套语义）。
- **ini/队列模式接线**（此前 mont 在队列模式下完全不可用）：新增 `mont`(0/1)、`mont_backend`(auto|simd|gmp)、
  `mont_torsion`(1|12)、`mont_threads`(0=auto)、`mont_save_pattern`，并写入默认 INI 模板；
  `mont = 1` 时强制 `use_edwards = false`（方法互斥）。
  **2026-09-24 键名整理**：上面这组键与其 Edwards 对应键已合并为**单一字段**——
  `method = gpu|edwards|mont`、`backend`、`field = auto|mersenne|montgomery`、
  `stage1_threads`、`naf_w`、`exponent = lcm|choose12`、`save_name_pattern`；
  旧键仍可读（每次运行提示一次），模板与迁移表见 `src/core/ecm_queue_config.cpp`。

### 10.3 实测：任务级线性扩展（M1277，B1=1e5，64 曲线，σ=900..963，B2=0）

| workers | wall (s) | s/curve | speedup |
|---|---|---|---|
| 1 | 8.61 | 0.1346 | 1.00× |
| 4 | 1.97 | 0.0308 | **4.37×** |
| 8 | 1.30 | 0.0203 | **6.62×** |
| 16 | 1.30 | 0.0204 | 6.62×（= 任务数上界，符合设计）|

- 8 任务 8 线程 = 6.62×（理想 8×；差距来自尾批、8 个 IFMA 批共用 L2/L3 带宽与频率回落）。- **确定性**：`--mont-threads 1` 与 `--mont-threads 8` 的共享存档，除 `TIME=`
  （人类可读时间戳，gmp-ecm 同字段）外 **逐字节相同** ⇒ 并行不改变任何数学结果，可作为回归判据。

### 10.4 同机硬指标对照（gmp-ecm 7.0.6 `ecm-zen3.exe`，B2=B1，单线程）

用户指定测试区间 **B1 = 1e5 ~ 1e6**，本轮把两端都测了（1 线程 = 1 核；SIMD 一比 8 条曲线）：

| 用例 | 本实现 1 线程 | 本实现 8 线程 | gmp-ecm 1 线程 | 每核倍数 |
|---|---|---|---|---|
| M1277，B1=1e5，64 曲线 | 0.135 s/curve | 0.0203 s/curve（6.62×）| 0.234 s/curve | **1.74×** |
| M1277，B1=1e6，16 曲线 | 1.418 s/curve | —（仅 2 任务）| 3.551 s/curve | **2.50×** |
| M3001，B1=1e5，64 曲线 | 0.526 s/curve | 0.0785 s/curve（6.71×）| 1.148 s/curve | **2.18×** |

- 复测（无后台负载）：M1277 = 0.1402 s/curve、M3001 = 0.5872 s/curve ⇒ 上表 ±5% 内可复现。
- ⚠ **本表（含下方 §10.3 的并行加速比）测于用户钉钟之前**，那时持续 AVX-512 会热降频；
  钉钟后（0-7 → 3990 MHz）单线程持续快约 30%，但 8 线程并行效率降到 **5.16×**。
  以 **§13.3 的钉钟后数据为准**，本表只作历史对照。
- gmp-ecm 的 = `Step 1 took` 合计 ÷ 曲线数（同机、同 B1、B2=B1 关闭 stage 2）。
- M1277 是**素数** ⇒ hits=0 属正确结果（stage 1 不可能命中）。
- 标量 gmp 后端（1 曲线/任务）≈5.6 s/curve（M1277,B1=1e5 口径）⇒ 只作对拍，不用于生产。

### 10.5 正确性回归（本轮改动后全部复跑）

- `mont_simd_verify.exe <M1277> 100000 12345 8 1`：**PASS (0/8 lanes disagree)**，
  批 0.854 s vs 标量 ×8 4.592 s ⇒ **5.38×**。
- `tools/test/test_mont_gmp_oracle.ps1`：**27 用例 PASS**（其中 11 个命中用例只比判定；
  1 个 `reference-only hit` 是 gmp-ecm 的指数策略差异，脚本已按策略跳过 x 比较）。
- 位数组重构未改任何对外签名（`mont_stage1_curve_x` / `mont_soa_stage1` 仍是薄包装）⇒ 既有工具链零改动即可复跑。

### 10.6 下一步（按性价比）

1. **M4c 逐素数链表**（§8.4 第 3 条）：唯一剩下的**算法级**收益（预计 −15~25% 步数）。
2. **M3 reader**（本仓库 resume + 与 gmp-ecm 的 `-resume` 双向往返）。
3. 把 §10.4 的对照扩到 **B1=1e6**（用户指定区间上端）与 M3001/M4003，做成 `tools/bench` 里的固定 A/B 脚本。

---

## 11. 字段层体检：per-bit 预算闭合，钱在哪、哪没钱（2026-09-24）

本节全部数字来自本轮新工具 `tools/bench/mers_sqr_phases.cpp`（把折叠域平方拆成 5 个相位，
并支持"稀疏 / 满宽随机"两种操作数），以及 `tools/bench/simd_mont_gate.cpp`。

### 11.1 每 bit 预算（**2026-09-24 修正：改为"相对本机屋顶"的口径**）

> **修正声明**：本节初版写的是"乘积相位 1.67 madd/cycle、Zen5 顶 ≈2/cycle ⇒ 已贴顶"。
> 这是**错的**，错法与 `docs/ECM_EDWARDS_STAGE1.md` §13.8 已经撤回过的那个结论**完全一样**：
> 用"采样到的时钟"去除，把利用率算高了。正确做法是**时钟无关**的：
> 用**同一进程内**测出的机器屋顶做分母（Gmadd/s，per-lane）。

**本机（Ryzen AI 9 HX 370，Strix Point）的屋顶只有桌面的一半**：Strix Point 的 Zen 5/Zen 5c 是
**256-bit FPU 数据通路**（[TechPowerUp: …Have 256-bit FPU Datapaths](https://www-techpowerup-com.analytics-portals.com/324873/amd-strix-point-soc-zen-5-and-zen-5c-cpu-cores-have-256-bit-fpu-datapaths)），
512-bit AVX-512 指令按 2×256-bit 执行 ⇒ zmm 吞吐约为全宽核心的一半。
本仓库 `tools/bench/mers_loop_bench.cpp` 早就在这台机器上实测过：**4/8/12/16 条独立 `vpmadd52` 链全部饱和在
4.00 Gmadd/s（= 1 madd/cycle @4 GHz，端口极限而非延迟极限）**，即 **32 Gmadd/s per-lane**。
（桌面 Zen 5 的对应数字是 zmm `vpmadd52luq/huq` ≈ 2 条/周期、跑在两条 512-bit FMA 流水线 FP0/FP1 上 ——
这一条是用户给出的实测/预期值，**本机无法验证**，故按"待桌面机确认"记录。）

`tools/bench/mers_sqr_phases.cpp` 现在在同一进程内同时测屋顶、内核与**真实 ladder**：

| 项目（n52=58，M3001） | 值 | 占本机屋顶 |
|---|---|---|
| 机器屋顶（8 条独立链，本进程实测） | 31.8 Gmadd/s per-lane | 100% |
| `ifma_mont_mul` | 26.6 Gmadd/s | **84%** |
| `ifma_mont_sqr` | 21.3 Gmadd/s | **67%** |
| 真实 ladder（`mont_soa_stage1_bits`，每 bit） | 20.5 Gmadd/s，**21.1 µs/bit** | 65% |
| 模型 6 mul + 4 sqr（同进程 best-of） | 17.3 µs/bit | — |
| **⇒ 非 madd 的 SoA 辅助开销**（add/sub 的多次 pass，见 §11.3b） | **3.8 µs/bit** | **18%** |

| 项目（n52=25，M1277） | 值 | 占本机屋顶 |
|---|---|---|
| `ifma_mont_mul` | 21.5 Gmadd/s | 67% |
| `ifma_mont_sqr` | 14.3 Gmadd/s | 45% |
| 真实 ladder | 14.6 Gmadd/s，5.5 µs/bit | 46% |
| 辅助 SoA 开销 | 1.3 µs/bit（23%） | — |

> ⚠ **不要把"sqr 67% vs mul 84%"读成"sqr 的乘积循环还有 17 个点"**：那是把 tail 算进分母的结果。
> 按**相位**看（§11.3）：sqr 的乘积相位 = 3248 条 / 887.8 ns = **91%**，mul 的乘积相位
> = 6612 条 / 1716 ns = **96%** ⇒ 两者都已贴顶，sqr 只剩约 5 个点（≈ ladder 的 1.3%）。
> 真正可动的是**辅助 SoA pass**（下表 18~26%），已在 §11.3b 落地。

**工作集不敏感**：把操作数缓冲从 1 个加到 16 个（≈60 KB，覆盖 ladder 的池规模），mul/sqr 的时间变化
**< 1%**（2020.7 → 2026.7 ns @ n52=58）⇒ **内核不是访存受限**，缺口不是缓存问题。

**修正后的结论（与初版相反）**：

1. **不是"已贴顶"**。mul 在 84%、sqr 只有 67%、ladder 整体 65% ⇒ 仍有实打实的余量：
   - `sqr` 比 `mul` 低 17 个百分点 ⇒ 若把 sqr 提到 mul 的水平，每 bit 省 4×0.26 µs ≈ **−5%**；
   - **非 madd 的辅助 SoA 开销占 18~23%** —— 初版凭"pass 数×n"的纸面估计把它当成个位数百分比，
     **实测把它排到了下一个最大的结构性靶子**（`soa_sub` 目前是 `neg`+`add` = 5 遍；`xz_add` 末尾还有一次 memcpy）。
2. **桌面机上的含义**：桌面 Zen 5 屋顶是本机的 2 倍，同一份代码在桌面上的"相对利用率"会**减半**
   （若其 ILP 不足以喂满两条全宽流水线）。所以"乘积相位 ILP"这类工作**在桌面上更值得做**，
   但在**本机**最多只能拿到 mul 那 16% 的余量。用户要求"乘积相位的 ILP 值得测试"——
   本机可测的判据就是上面这张利用率表；结论：**可测的余量是"sqr 追 mul"与"辅助 pass 削减"，
   而不是继续加 ILP**（本机端口已饱和，加 ILP 不会超过 100%）。

### 11.2 折叠域平方确实在赚钱（`sqr/mul = 0.66–0.69`）

| 对照 | 时间 | 说明 |
|---|---|---|
| `ifma_mont_sqr` | 1472 ns | 只算严格上三角 `n(n−1)+2n` |
| `ifma_mont_mul(a,a)`（同一操作数） | 2138 ns | 通用乘法内核 |
| 比值 | **0.69** | 对称平方省下 ~31% |

⇒ 每 bit 实际 = `6M + 4×0.69M ≈ 8.8M`（而不是朴素 10M），即**已经吃到 ~12%**；
即使平方做到理论 0.5M，也只再得 ~8.7%。

**⚠ 一个容易误读的陷阱（已记录）**：`simd_mont_gate` 打印的 `sqr ≈ mul`（3750 vs 3611 ns @ n52=58）
**不是折叠域**——gate 用 `ifma_ctx_init`(AUTO) + **随机 N** ⇒ 走 **Montgomery/CIOS** 分支，
那里 `sqr` 就是 `mul(a,a)`、madd 数 `n(4n+3)=13630`（gate 打印的正是这个 CIOS 公式），所以两者同价是应该的。

### 11.3 五个相位的成本（折叠域平方，n52 = 58）

| 相位 | 稀疏操作数 | 满宽随机操作数 |
|---|---|---|
| init（清零 2n+2 列） | 39.7 ns | 39.7 ns |
| **prod（上三角乘积）** | **891.8 ns** | **887.8 ns** |
| double（上三角 ×2 的串行进位） | 115.9 ns | 108.5 ns |
| diag（`a_i²` 对角） | 48.9 ns | 45.1 ns |
| **finish（归一/折叠/drain/规范化）** | **368.4 ns** | **387.3 ns** |
| 合计 | 1464.7 ns | 1468.5 ns |

- 乘积相位 = 3248 条 madd 指令 / 887.8 ns = **3.66 Gmadd/s per-lane... 不，按 §11.1 的正确口径：
  乘积相位 887.8 ns 内 3248 条 zmm madd 指令 ⇒ 3.66 条指令/ns = 与"屋顶 4.0 条指令/ns"相比 ~91%**
  （注意单位：`mers_loop_bench` 与本节屋顶都用 **zmm 指令数/秒**，不是 per-lane；per-lane 要 ×8）
- **操作数幅度完全不影响**（稀疏 vs 随机差 <2%）⇒ `drain` 循环里的 6 次上限**实际不级联**，
  我先前"finish 数据相关"的猜测**被自己的测量证伪**（记录在此以免复犯）。
- `finish` = 每个域运算的 **~26%**，且实测 ~0.7 ops/cycle ⇒ **串行 carry 链受限**（\(2n\) 列归一 + 折叠回写 + ≥1 次 \(n\) 列 drain + 2 遍规范化，共 ~4n 列迭代）。
  ⇒ 这是**单个域运算内**唯一还剩的靶子，上限约 10%（把 finish 砍半 ⇒ per-bit −13%）。

### 11.3b 辅助 SoA pass 优化（已落地，2026-09-24）

**靶子来源**：§11.1 实测 ladder 有 **18~26%** 的时间花在"非 madd"的 SoA 辅助运算上。
按 pass 数一查就明白：每个域元素的加/减要**多趟**遍历 n 列，而 ladder 每 bit 有 4 加 + 4 减。

| 助手 | 旧实现 | pass 数 | 新实现 | pass 数 |
|---|---|---|---|---|
| `soa_add` | add 一趟 + `soa_cond_sub` 两趟（borrow、blend）| 3 | 一趟同时算 `r` 与候选 `r−N`，再一趟 select | **2** |
| `soa_sub` | `soa_neg`（2 趟）+ `soa_add`（3 趟）| **5** | 一趟带借位相减，再一趟只在借位 lane 上加回 N | **2** |
| `xz_add` 末尾 | `sqr` 到临时 + `memcpy` 到 `r.X` | 1 | `sqr` 直接写进 `r.X`（读全部在前，无别名问题）| **0** |

每 bit 的 pass 总数 **≈30 → 18**（`xz_dbl` 13→6，`xz_add` 16→8）。

**`soa_add` 的判定规则**（这里最容易写错，故留推导）：令 `cy` 为第 n−1 列的进位、
`borrow` 为逐 limb 做 `limbs − N` 的借位，则

```
value >= N   <=>   cy != 0  或  borrow == 0
```

- `cy == 0`：value 就是那 n 个 limb，`borrow == 0` 恰好表示 value ≥ N ✓；
- `cy != 0`：此时**必有 masked_limbs < N**（否则 value = masked + 2^(52n) > 2N，与 `value < 2N` 矛盾），
  于是逐 limb 相减的结果正好等于 `value − N`（丢掉的 2^(52n) 进位与借位相消）✓。

`soa_sub` 同理：借位为 1 的 lane（a < b）真值是 `a − b + N`，第二趟加 N 时最高进位被丢弃即正确；
借位为 0 的 lane 保持 `a − b < N` ✓。两条规则在 `mont_simd_verify` 的逐 lane 对拍里被覆盖。

**实测（同进程、同时钟条件，工具 `mers_sqr_phases`）**

| 口径 | 改前 | 改后 | 变化 |
|---|---|---|---|
| 辅助 SoA 开销（突发 B1=3000） | 3.8 µs/bit（18%） | **2.1 µs/bit（11%）** | **−45%** |
| ladder 每 bit（突发） | 21.1 µs | **19.4 µs** | **−8.1%** |
| 辅助 SoA 开销（持续 B1=1e5） | 6.0 µs/bit（26%） | **4.0 µs/bit（19%）** | −33% |
| ladder 每 bit（持续） | 23.2 µs | **21.3 µs** | **−8.2%** |
| 整程序交替 A/B（M3001 B1=3e4，32 曲线，1 线程；对手 = finish 前 + pass 前的二进制） | 3.96 s | **3.39 s** | **−14.5%**（含 §11.6 的 finish −7%）|

**验证（全绿）**：`mont_simd_verify` 逐 lane PASS（M1277/B1=1e5：批 0.7127 s vs 标量 ×8 3.845 s；
M3001/B1=1e4：0.2804 vs 1.658）；oracle **27/27 PASS**；**存档 32 行与最初基线逐字节一致**
（辅助 pass 改动纯属表示层，数值不变）。

> 残余观察：**持续负载下辅助开销占比反而更高**（19% vs 突发 11%），说明这些 pass 是**延迟/访存**
> 主导而非 madd 吞吐主导 —— 降低有效时钟时它们不按比例变便宜。若还要继续压，方向是
> **减少 pass 的趟数**（例如把 select 与下一趟运算融合，或允许"冗余表示"跨一次乘法），
> 而不是加 ILP。

### 11.4 每个任务的固定开销（对"多 N、小任务"场景的关键数字）

M1277，B1=1e6，用 `t(8) 与 t(16)` 差分（`s` 的构造对两者相同）：

```
t(8 curves)  = 9.86 s     8 曲线批 = 9.66 s
t(16 curves) = 19.52 s    固定开销 = build_s + ctx init = 0.21 s  (8 曲线任务的 2.1%)
```

⇒ **`s`/位数组的磁盘缓存（§8.2 第 1 条）不紧急**：只在 B1 ≳ 1e7 或"每任务曲线数极少"时才值得做；
真正的分布式收益来自 §8.2 第 2/3 条（逐素数链表 / 直线程序）。

### 11.5 本轮结论

| 层面 | 状态 |
|---|---|
| 每 bit 域运算条数 | ✅ 已到 EFD 最优（6M+4S，且 add 走仿射差分便宜支）|
| 平方折扣（`S < M`） | ✅ 折叠域已拿到（`sqr/mul = 0.60~0.77`，随 n 增大）|
| madd 吞吐 | ✅ 乘积相位 91%（sqr）/ 96%（mul）的本机屋顶；"sqr 67% vs mul 84%"是含 tail 的分母，勿误读 |
| `finish` | ✅ 已优化，见 §11.6（端到端 −7~8%）|
| **辅助 SoA pass** | ✅ **已优化，见 §11.3b（18~26% → 11~19%，ladder −8%，端到端合计 −14.5%）** |
| 步数（链） | ✗ **已关闭**：PRAC/Lucas 链实测 +3~4%（更慢），见 §12 |
| 并行度 | ✅ 任务级（§13.3：不绑定 16/24 线程 = 8.46×/9.01×）|
| 任务固定开销 | ✅ 2.1%（B1=1e6、8 曲线）|

### 11.6 finish 优化落地（2026-09-24，"低风险优先"的取舍）

两处改动，**都是保值融合，不动任何数学**（`src/cpu/simd_mont_ifma.cpp`）：

**(1) canonical 从"减 N + 选择"变为"等值检测 + 稀有修正"。**
Mersenne 下 `N = 2^k−1` 的 52-bit limb 模式正好是"全 1"（低位 limb = `2^52−1`，
顶 limb = `2^(52−sh)−1`），而 `nb` 里存的就是 N 的原样 limb ⇒ 规范化判据可以写成
**逐 limb 与 `nb` 比较**：所有 limb 都相等的 lane 持有的就是 N，置 0 即可；修正只在真的出现该
lane 时执行（实践上几乎不发生）。旧写法是"整数组减 N、记录 borrow、再 blend"，两趟 + 串行 borrow 链
（n52=58 时 133.5 ns = 整个 tail 的 34%）⇒ 现在约 1 趟无依赖的 compare+store。
- 依据：drain 已保证值 < 2^k，而 `[N, 2^k)` 里只有 `2^k−1` 这一个值 ✓（§3 注释同结论）。

**(2) 平方尾部三趟合一。** 原来 `double(上三角×2)` → `diag(a_i²)` → `normalize` 三趟各走一遍
列；现在**一趟**同时完成"加倍 + 加对角 + 归一化"：
- 对角必须不被加倍 ⇒ 不是单独一趟加，而是**逐列加**：第 `2q` 列加 `lo(a_q²)`、第 `2q+1` 列加 `hi(a_q²)`，
  两者的来源 `a_q`（下标 `q ≤ k/2`）在到达该列前已知；
- 每列上界 `2·t[k] + carry + a_q² 项 < 2^58`（`t[k]` 是 ≤ n/2 个 lo 项之和）⇒ 64-bit 加法不会回绕；
  循环按两列展开以把 lo/hi 选择移出循环体，2n 为偶数无需收尾；
- 这趟本身就归一化了全部 2n 列，且链尾的进位正是 tail 需要的 `top` ⇒ 新增
  `ifma_mersenne_fold_top(out, t, c, top)`（tail 的步骤 3–5），**平方不再重复跑 normalize**。

**顺带修掉一个潜伏错位**：旧代码把加倍的进位 `init_carry` 作为"进入 column 0 的进位"传给
finish，而它其实是 **position 2n** 的数字。实际恒为 0（列和 < 2^58 ⇒ 进位 < 2^6，最后一列再掩码后为 0）
才一直没有暴露；新实现把它当作 `top` 交给 `fold_top`（语义正确，非零时也对）。

**验证（全部通过）**

| 检查 | 结果 |
|---|---|
| `mont_simd_verify`（折叠域 SIMD vs 标量 MPN/GMP，逐 lane） | PASS：M1277/B1=1e5（批 0.792 s vs 标量 ×8 4.667 s，**5.89×**）、M3001/B1=1e4（0.301 s vs 1.647 s，**5.47×**）|
| `test_mont_gmp_oracle.ps1`（对 gmp-ecm 逐点） | **27/27 PASS** |
| 存档逐字节回归（同 σ/B1/N，改动前后） | **32 行完全一致**（`SIGMA/B1/N/X/CHECKSUM`），只有 `TIME=` 不同 |
| 交替 A/B（同进程、best-of-7） | 平方 −13.5%（n52=25）/ −12.1%（58）/ −10.7%（77）|
| 交替 A/B（整程序，旧二进制 = 改内核前且已含线程+位数组） | M1277/B1=1e5/32 曲线 3.50→3.23 s（**−7.9%**）；M3001/B1=3e4/32 曲线 3.98→3.71 s（**−6.9%**）|

> 说明：这台笔记本频率漂移可达 20%（同一改动前的 64 曲线 M1277 曾从 8.61 s 变到 8.97 s），
> 所以 §11.3 那种跨时段的"改动前 vs 改动后"对比不可用；上表一律用**同进程交替**或**同机交替二进制**。
> **2026-09-24 补充**：用户已把时钟钉死（0-7 → 3990 MHz、8-23 → 2995 MHz），
> 上述"漂移"不再存在；钉死后的完整机器画像与并行效率见 **§13**。

**未做（评估后放弃）**：去掉 fold 循环里对高位列的防御性清零（约 1%，但会引入"scratch 残留"隐式依赖，性价比不合算）；
把 2n 列的 normalize 串行链拆成两条独立链做 ILP（约 4%，但要动 $2^k=1$ 折叠的边界，属"高风险区"，见 §3 注释里
那个曾经产出非规范值的 bug）。

---

## 12. Lucas 链（PRAC）移植评估：**结论是不做**（2026-09-24，有原型与对拍支撑）

用户要求"finish 之后进行 lucas 链开发移植"。本轮先把**盈亏算清楚再写代码**，结果是：**在我们这套内核里，
逐素数 Lucas 链（PRAC）不划算，1~4% 更慢**。以下是证据链与"什么情况下结论会翻转"。

### 12.1 参考实现到底做了什么（已读源码，非猜测）

- **gmp-ecm**：每个素数 p 一个位打包的 chain code（`Lchain_codes.dat`，由 156 KB 的 `LucasChainGen` 离线生成），
  运行时 `generate_Lucas_chain(p, code, Lchain[])`（`ecm.c:549`）解出
  `chain_element{value, comp_offset_1, comp_offset_2, dif_offset}` —— 两个加数的**窗口内相对下标**，
  以及它们**差的下标**（`dif_offset = 0` 表示自身加倍）。执行原语是 `duplicate` 与 `add3`；
  `add3` 的注释写明 **6 muls（4M+2S）**（`ecm.c:148-157`）。
- **Prime95**：不查表，运行时用 **PRAC**（Montgomery 的 Euclid/连分数式链）现算：`lucas_mul()`（`ecm.cpp:2711`）
  维护 (A,B,C) 三点，`lucas_cost()`（`ecm.cpp:2645`）是它的**精确成本模型**（12 = 一次加法、22 = 加法+加倍、
  10 = 一次外层循环的账目边界），`lucas_cost_several` 在 `PRAC_SEARCH=7` 个候选 d（从 $\phi n$ 附近）里取最省。
  两个原语：`ell_dbl_xz_scr` = 10-11 FFT / 4 adds，`ell_add_xz_scr` = 12 FFT / 6 adds（`ecm.cpp:2388,2494`）。
- 关键结构事实：Lucas 链的每次加法都要求**差是链上已算出的点**（这正是 `dif_offset` 存在的原因）
  ⇒ 差是**射影点**，而不是我们 ladder 里那个恒定的**仿射**起点。

### 12.2 成本必须换算到我们的单位（这才是决定性的一步）

折叠域里 `S = 0.628M`（§11.2 实测），于是：

| 原语 | 公式 | 我们的单位 |
|---|---|---|
| 我们的 ladder `xz_dbl` | 2S + 3M（含 `a24·E`） | **4.26M** |
| 我们的 ladder `xz_add` | 2 交叉积 + 2S + `xdiff` 1M（**差分仿射**） | **4.26M** |
| gmp-ecm/P95 的 `add3`/`ell_add_xz_scr` | 4 个内积 + 2S + `·zdiff` + `·xdiff` | 7.26M |
| EFD `dadd-1987-m-3`（**和差形式**，射影差分的最优写法） | 2 个乘积 + 2S + `·Z_D`/`·X_D` | **5.26M** |
| P95 `ell_dbl_xz_scr` | 1S + 4M | 4.63M |

⇒ ladder 每 bit = 4.26 + 4.26 = **8.51M**（每 bit 恰 2 个原语）。
链法每 bit 的**原语数**确实更少（PRAC ≈ **1.70**，见 §12.3），但它的加法贵 23.5%（5.26 vs 4.26）——
两者相乘基本抵消。

### 12.3 原型与对拍（`tools/stat/prac_cost.py`）

逐字转写 Prime95 的 `lucas_mul` 状态机（含"外层首步三种分支"与"结尾一次加法"），逐素数统计 dbl/add，
并按上表定价。**校验**：把我的 op 计数换算成 Prime95 自己的单位（12/22/10）与 `lucas_cost()` 聚合比较：

```
per-prime mismatches: 9588 of 9592; aggregate units mine 2.7408e+06 vs ref 2.8367e+06 (-3.38%)
```

（逐素数不一致集中在 `d` 退化的特殊素数 5/13/17 一类的边界；**聚合只差 −3.4%，方向是"我少算"**
⇒ 下面的经济性对 PRAC **偏有利**。）

| B1 | 素数 | dbls | adds | ops/bit(s) | 用参考加法 6M+2S | 用最优加法 4M+2S | 修正后 |
|---|---|---|---|---|---|---|---|
| 1e5 | 9592 | 57948 | 189777 | 1.707 | 1.645e6 M | 1.244e6 M | **1.042×** ladder |
| 1e6 | 78498 | 534072 | 1898508 | 1.684 | 1.625e7 M | 1.225e7 M | **1.031×** ladder |

（ladder 基线：B1=1e5 为 1.236e6 M、B1=1e6 为 1.229e7 M。）

**读法**：链法把每 bit 的原语数从 2.0 降到 1.70（−15%），但每个原语平均贵 18%（4.26→5.02M），
于是总账是 **+3~4%**；如果照抄参考实现的加法（6M+2S），则是 **+32%**。

### 12.4 为什么 ladder 这么难被超过（结构性原因）

1. 我们的**差分点是恒定且仿射的**（起点 P，`xdiff = X0/Z0` 预先算一次）⇒ 每次加法只花
   3M+2S，其中 `xdiff` 只付 1M。链法里差是任意射影点 ⇒ 至少 4M+2S（EFD 最优），实际参考实现是 6M+2S。
2. 链法要便宜下去，只有让差也是仿射 —— 那就需要**每次加法一次求逆**（或每 k 次批量求逆）：
   一次 3000-bit 求逆 ≈ 数百次乘法，而该素数整条链也就 ~20 个原语 ⇒ 立刻亏。
3. 链法省的是**原语条数**（1.70 vs 2.00，且理论上限是 $1 + 1/\log_2\log_2 p \approx 1.25$），
   它省不掉"每个原语都要做一次完整的折叠归约（§11.6 的 finish）"这件事 —— 而我们的 ladder
   每 bit 只做 2 次归约，本来就已经接近下界。

### 12.5 什么情况下结论会翻转（留给未来的判据）

- 如果加法能保持仿射差分（需要求逆便宜到可忽略，例如硬件加速逆元、或 $N$ 很小以致求逆便宜）⇒ 链法 −20% 左右；
- 如果我们的 ladder **不是**共享一个指数、而是每个 lane 有不同指数（发散），则"链的 op 数优势"会重新有意义；
- 如果目标是**性能对齐 gmp-ecm 的行为**（例如复现它的 `Lchain_codes.dat` 完全一致的中间点），
  那是"兼容性"而不是"性能"需求 —— 我们的存档/结果对齐（Q1）已经用 `--mont-torsion 1` 满足了。
- 反之，如果未来在**别的方法**（比如 Montgomery SOS / 对称平方那条线，用户已"ab 暂缓"）上，
  加法的相对成本变了，这份原型脚本可直接改成本表重算。

### 12.6 结论与后续

**M4c 关闭（不移植）。** 这一轮的资源改投到仍有实测余量的地方（口径见 §11.1 的修正）：
① **`sqr` 追平 `mul`**（67% vs 84% 的本机屋顶，每 bit 约 −5%）；
② **削减非 madd 的辅助 SoA 开销**（实测占 ladder 的 **18~23%**：`soa_sub` 现在是 `neg`+`add` 五遍、
`xz_add` 末尾还有 memcpy）—— 这是本轮之后**最大的单一结构性靶子**；
③ 乘积相位 ILP：本机端口已饱和（mul 84%），**加 ILP 在本机最多再拿 16%**，但在桌面 Zen 5
（屋顶 2 条/周期）上更值得做，需要一台桌面机才能判定其 ILP 是否够喂满两条全宽流水线。

> 复现：`python tools\stat\prac_cost.py 100000 1000000`
> （脚本内注释给出 `--sqr` 覆盖、`--check` 符号检查为何"不可作为证据"。）

---

## 13. 本机画像：钉死时钟、混合核、并行效率（2026-09-24，用户提供 + 实测）

用户把频率钉死以形成**稳定测试环境**（避免过热降频）：

| 逻辑 CPU | 类型 | 钉定上限 | 默认睿频 |
|---|---|---|---|
| 0–7 | Zen 5（大核，4 个物理核 × SMT） | **3990 MHz** | 5100 MHz |
| 8–23 | Zen 5c（小核，8 个物理核 × SMT） | **2995 MHz** | 3300 MHz |

> 平台：Ryzen AI 9 HX 370（Strix Point），**256-bit FPU 数据通路** ⇒ 512-bit AVX-512 按 2×256-bit 执行，
> 吞吐约为全宽桌面核的一半（[TechPowerUp](https://www-techpowerup-com.analytics-portals.com/324873/amd-strix-point-soc-zen-5-and-zen-5c-cpu-cores-have-256-bit-fpu-datapaths)）。

### 13.1 钉死后屋顶可读成"指令级"数字

同进程实测屋顶 = **31.8 – 32.2 Gmadd/s per-lane**（8 条独立 `vpmadd52` 链）÷ 8 lane
= **3.98 – 4.02 G 条 zmm 指令/s**，在 3990 MHz 上正好是 **1.0 条 zmm madd / 周期**。
⇒ 这台机器的 madd 屋顶 = 1 指令/周期（桌面 Zen 5 为 2/周期，用户给出的实测/预期值）。
也正因如此，`mers_loop_bench.cpp` 里"1.0 madd/cycle = 硬件屋顶"那句**在本机是对的**，
而 §11.1 初版"顶 ≈ 2/cycle"是错的（已修正）。

### 13.2 单线程时大/小核**没有**速度差；并发时才显现

逐逻辑 CPU 单线程实测（M3001，B1=5e4，8 曲线 = 1 批，90 W）：

```
cpu 0..23: 1.531 s … 1.584 s      最快/最慢 = 1.03x
0-7 均值 = 1.546s   8-23 均值 = 1.554s   比值 = 1.00x
```

⇒ **单独跑时 24 个逻辑核一样快**（大核被功耗压到接近小核的水平）。但**并发时两个区间表现完全不同**。
按**每物理核吞吐**（= 线程数/墙钟 ÷ 物理核数，单核独占 = 0.326 b/s）看：

| 小核（Zen5c）配置 | 线程 | 墙钟 | 每物理核 | 相对独占 |
|---|---|---|---|---|
| 1 个物理核 `9`（独占） | 1 | 3.07 s | **0.326 b/s** | 100% |
| 2 个物理核 `8,10` | 2 | 4.35 s | 0.230 | 71% |
| 4 个物理核 `8,10,12,14` | 4 | 4.54 s | 0.220 | 67% |
| 8 个物理核 `8,10,…,22` | 8 | 5.56 s | 0.180 | 55% |
| 大核 `0,2`（2 物理核，对照） | 2 | 3.07 s | 0.326 | **100%** |
| 大核 `1,3,5,7`（4 物理核，对照） | 4 | 3.08 s | 0.326 | **100%** |

⇒ 本机不是"4 快 + 8 慢"，而是 **"4 个可线性扩展的大核 + 一个一旦有并发就整体掉速的小核簇（Zen5c）"**：
小核**单独跑和大核一样快**（0.326 b/s，§13.2 的逐核表），但**只要有 ≥2 个线程并发，整簇每核效率就掉到 71%→55%**
—— 这是**簇级**的功耗/频率效应，与 SMT 无关（SMT 的净效应见 §13.4，其实是小的正收益）。

### 13.3 并行效率：钉钟 + 90 W 后的完整表（**推翻上一版的"功耗墙"解释**）

M3001，B1=1e5（每批 8 曲线 = 144344 bit，每线程恰好 1 批），`--mont-backend simd`，
吞吐倍数 = `线程数 × 单大核每批时间 / 墙钟`（单大核每批 = 3.07 s；本表测于 §11.3b 的 pass 优化**之前**，
优化后绝对时间各降约 8%，**比值不变**——同一份内核跑所有配置）：

| 配置 | 线程 | 墙钟 | 吞吐倍数 | 每核效率 |
|---|---|---|---|---|
| 单核 `3`（用户指定） | 1 | 3.07 s | 1.00× | 100% |
| SMT 对 `2,3`（用户指定） | 2 | 5.55 s | **1.11×** | 55% |
| 大核 `0,2`（不同物理核） | 2 | 3.07 s | **2.00×** | 100% |
| 4 大核 `1,3,5,7`（用户指定） | 4 | 3.08 s | **3.99×** | 100% |
| 4 小核 `9,11,13,15`（用户指定） | 4 | 4.93 s | 2.49× | 62% |
| 大核满 SMT `0-7` | 8 | 5.58 s | 4.40× | 55% |
| 8 小核 `9,11,…,23` | 8 | 5.33 s | 4.61× | 58% |
| 全部 12 物理核（4 大 + 8 小） | 12 | 6.88 s | 5.35× | 45% |
| **不绑定** | 8 | 4.67 s | **5.26×** | 66% |
| **不绑定** | 12 | 5.31 s | **6.94×** | 58% |
| **不绑定** | 16 | 5.81 s | **8.46×** | 53% |
| **不绑定** | 24 | 8.18 s | **9.01×** | 38% |
| 绑定 `0-7` + 8 个小核物理核（16 线程） | 16 | 6.08 s | 8.07× | 50% |
| 绑定**全部 24 逻辑核**（含小核 SMT） | 24 | **12.60 s** | **5.85×** ❌ | 24% |

**结论**：

1. **`affinity` 一律不要设**。不绑定在任何线程数下都不劣于手工绑定，24 线程时甚至**快 54%**
   （9.01× vs 5.85×）—— 原因见 13.4：Windows 调度器**自己知道小核 SMT 的惩罚并规避它**，
   手工绑定把这个信息丢掉了。
2. **SMT 是"最后资源"**：大核 SMT 稳定 +10~11%；小核 SMT 在簇未满时 **+6~8%**、
   簇占满（16 线程）时 **−19%**（详见 13.4 的同簇同占用对照）。
   **先填满物理核，再考虑 SMT 兄弟**。
3. **吞吐天花板 ≈ 9.0× 单大核**（不绑定 24 线程，每核 38%）；**16 线程是效率甜点**（8.46×，每核 53%）。

**推翻上一版的解释**：此前把 8 线程只有 5.16× 归因为"整机功耗墙"，并按用户要求把功耗从 40 W 提到
90 W 后复测 —— **8 线程只从 5.16× 变成 5.26×，单核 3.06→3.07 s 完全没变** ⇒ **功耗墙不是主因**。
真正的限制是：**小核簇并发掉速**（13.2）+ **SMT 不提供浮点吞吐**（13.4）+ 核间速度不均。
（40 W 时代的旧数字 2/4/8 线程 = 1.94×/3.46×/5.16× 已作废：那次 4 线程是**不绑定**跑的，
混进了小核与 SMT，故低于 4 个不同大核的 3.99×。）

### 13.4 亲核性：SMT 兄弟配对 + SMT 的真实效率（**同簇同占用对比**）

**配对规则（两个区间都验证过）：相邻编号 = 同一物理核的 SMT 兄弟。** 2 线程 / 2 批：

| 配对 | 墙钟 | 吞吐 |
|---|---|---|
| 大核 `(0,1)` **兄弟** | 5.54 s | 2 批 / 5.54 s = 0.361 b/s（1 物理核）|
| 大核 `(0,2)` 不同核 | 3.07 s | 2 批 / 3.07 s = 0.651 b/s（2 物理核 ⇒ 0.326/核）|
| 小核 `(8,9)` **兄弟** | 8.18 s | 2 批 / 8.18 s = 0.244 b/s（1 物理核）|
| 小核 `(8,10)` 不同核 | 4.35 s | 2 批 / 4.35 s = 0.460 b/s（2 物理核 ⇒ 0.230/核）|

**SMT 效率必须与"同一簇、同一占用"对比**（这是 2026-09-24 的一次修正）：
拿 SMT 对去比**单核独占**会把簇级掉速算到 SMT 头上，得到错误的"负收益"结论。正确算法：

| 小核簇占用 | 无 SMT（每物理核 b/s） | 有 SMT（每物理核 b/s） | SMT 净效应 |
|---|---|---|---|
| 1 个物理核 / 2 线程 | 0.230（用 2 个物理核）| **0.244**（用 1 个核 + SMT）| **+6%** |
| 4 个物理核 / 8 线程 | 0.220 | **0.237** | **+8%** |
| 8 个物理核 / 16 线程 | 0.180 | **0.146** | **−19%** |

| 大核簇占用 | 无 SMT | 有 SMT | SMT 净效应 |
|---|---|---|---|
| 2 个物理核 / 2 线程 | 0.326 | 0.361（1 核 + SMT）| **+11%** |
| 4 个物理核 / 8 线程 | 0.326 | 0.359（3.99→4.40 倍 / 4）| **+10%** |

⇒ **小核 SMT 是小的正收益（+6~8%），不是负收益**；但**当整个 Zen5c 簇被占满（16 线程）时转为 −19%**
（簇的功耗/频率预算已经用尽，SMT 只是在抢同一份预算）。大核 SMT 稳定在 +10~11%。

**这张表解释了"不绑定最快"**：调度器在混合核上会先填**物理核**、把 SMT 当**最后资源**；
手工钉满 24 逻辑核则强制小核簇同时开 SMT（−19%）并压满大核，吞吐从 9.01× 掉到 5.85×（−35%）。
手工钉 12 个物理核（5.35×）也劣于不绑定 12（6.94×），因为 OS 会优先用大核 + 少量 SMT 空位，
而不是把 8 个小核全部压上（小核簇越满越慢：0.230→0.180）。

**实践规则**：
- 生产环境**留空 `affinity`**（让调度器按"物理核优先、SMT 殿后"排布）；
- 真要绑定：**先一个 worker 一个物理核**（`1,3,5,7` 或 `8,10,12,14`），**SMT 兄弟只作为最后资源**；
- **不要把 16 个线程全塞进小核簇**（8 物理核全满 + SMT ⇒ 每核 −19%）。

### 13.5 用户指定配置的逐条结果（M3001，B1=1e5，SIMD，每线程 1 批）

| 用户指定 | 线程/曲线 | 墙钟 | 吞吐倍数 | 判读 |
|---|---|---|---|---|
| 单独核心 `3` | 1 / 8 | 3.07 s | 1.00× | 单核基准（21.3 µs/bit）|
| 核心 `2,3`（SMT） | 2 / 16 | 5.55 s | 1.11× | SMT 同核，几乎没收益 |
| `1,3,5,7` | 4 / 32 | 3.08 s | **3.99×** | 4 个不同大核，**线性 100%** ✓ 最优效率 |
| `9,11,13,15` | 4 / 32 | 4.93 s | 2.49× | 4 个不同小核，62% |

**给生产排产的建议**（0-7 = 大核、两组都有 SMT 已确认）：
- **不绑定**，`--mont-threads 16`（8.46×，每核 53%）作为效率甜点；
  `--mont-threads 24` 拿到吞吐上限 9.01×（每核 38%）；
- 若必须绑定：`1,3,5,7`（4 线程，100% 效率）优先；SMT 兄弟只作最后资源
  （大核 +10~11%，小核簇未满时 +6~8%、占满时 −19%，见 13.4）；
- **不要**把 16 个线程全塞进小核簇。

### 13.6 顺带修掉的 `affinity` 解析 bug（真实 5.5× 事故）

`parse_affinity_spec()` 原先用 `std::stol()` 解析每个逗号分隔项，于是：
- `--affinity 0-7` 被 `stol("0-7")` **静默解析为 0**（只读前导数字，不报错）⇒ 所有 worker 钉到 CPU 0
  ⇒ 8 线程 64 曲线从 4.7 s 掉到 **26.5 s**，看起来像硬件故障；
- 该函数**只接受逗号列表**，文档里写的范围写法其实不支持。

已修（`src/core/ecm_driver.cpp`）：支持 `<cpu>`、`<lo>-<hi>`、逗号混合列表（`0-3,8,10-11`），
并**严格校验整段消费**（`std::stol` 的 `used == size`），非法项打印告警而不是静默变成 0；
超长列表（>1024）整体忽略。同时**新增命令行 `--affinity <list>`**（此前只有 ini 键 `affinity`），
并在启动日志打印实际绑定：

```
affinity        : 0,1,2,3,4,5,6,7 (worker t -> cpu[0,1,2,3,4,5,6,7][t % 8])
```

复现：`ecm.exe --affinity 0-7 --mont --mont-threads 8 … ` 与不带 `--affinity` 对比（§13.4 表）。

---

## 14. 三方单线程对比：本实现 SIMD vs GMP-ECM vs Prime95（2026-09-24）

**同一 N = M4001 = 2^4001−1**（与用户正在跑的任务相同），**同为 Suyama/Montgomery 曲线族**，
**stage 1 only**，**都是单线程（1 物理核，无 SMT 共享）**。

### 14.1 测量条件与命令

| 实现 | 版本/配置 | 命令 | 口径 |
|---|---|---|---|
| **本实现** | `build_vs18\Release\ecm.exe`，AVX512-IFMA 8 lane 批 | `echo (2^4001-1) \| ecm.exe --mont --mont-backend simd --mont-threads 1 -gpucurves 8 --sigma 4242 <B1> 0` | 一批 8 条曲线同时算 ⇒ **每曲线 = 批时间/8** |
| **GMP-ECM** | 7.0.6 `ecm-zen3.exe`（`--enable-asm-redc`）| `<N> \| ecm-zen3.exe -param 0 -c 8 <B1> <B1>`（B2=B1 关闭 stage 2）| 日志 `Step 1 took …ms` 平均 |
| **Prime95** | v31.04b05，`NumWorkers=1`，用户正在跑 M4001/B1=1e7 | 用户 `screen.log`：`Stage 1 complete … Total time: 67.1 s` | 每曲线 stage 1 总时间（含建表）|

**指数口径**：本实现与 GMP-ECM 用 `lcm(1..B1)`（`-param 0` 对齐）；Prime95 用 `12·lcm(1..B1)`（choose12）
⇒ 只多 `log2 12 ≈ 3.59` bit（相对 1e7 时约 1.44e7 bit 是 **+0.000025%**）⇒ 计时上**可忽略**。

### 14.2 实测（每曲线，秒）

| B1 | **本实现（单线程）** | GMP-ECM 7.0.6（单线程）| Prime95（1 worker）|
|---|---|---|---|
| 1e5 | **0.596**（批 4.77 s）| 1.787 | 0.672（由 1e7 线性缩放）|
| 1e6 | **6.264**（批 50.1 s）| 15.008 | 6.72（同上）|
| 2e6 | **13.455**（批 107.6 s）| ~30（缩放）| 13.44（同上）|
| 1e7 | **62.6 ~ 67.3**（外推：1e6×10 / 2e6×5）| ~150（1e6×10；注意 GMP-ECM 自身是**次线性**的：1e5→1e6 只用了 8.4×，故真值可能略低）| **67.2（实测）** |

**比值**：

| 对照 | B1=1e5 | B1=1e6 | B1=2e6 | B1=1e7 |
|---|---|---|---|---|
| 本实现 vs **GMP-ECM** | **3.00×** 快 | **2.40×** 快 | ~2.2× | ~2.2× |
| 本实现 vs **Prime95** | 1.13× 快 | 1.07× 快 | **1.00×（持平，差 0.1%）** | **~1.0×（持平）** |

### 14.3 读法与结论

1. **单线程：与 Prime95 基本持平**。最能说明问题的是 **B1=2e6** 这个点：我们的批 107.6 s 与
   Prime95 的 67.2 s 曲线处于**同一量级的热/频率状态**，此时 13.455 vs 13.44 s/曲线 ⇒ **差 0.1%**。
   B1=1e7 的 62.6~67.3（外推）对上 67.2（实测）也是同一个结论。
2. **比 GMP-ECM 快 2.2~3.0×**（单线程同机、同 N、同 B1、同为 param 0 曲线族）。
   这个差距的来源已在本轮逐项量化：**8 lane SIMD 批（同一条曲线成本摊到 8 条）**
   + **折叠域（$2^k=1$）把模乘降到 $2n^2$ madds、平方再减半** + **ladder 的差分点仿射**（加法 3M+2S）。
3. **Prime95 为什么能追平**：它用 **GWNUM FFT**（4001 bit 用 257,501,901 transforms/曲线）+
   **PRAC/Lucas 链**（§12 测过：链本身在我们这套内核里不划算，但在 FFT 成本模型下是划算的）。
   换句话说：**我们赢在算术表示与批量，它赢在乘法算法**，两者在 4000 bit 这个规模上打平。
4. **注意 B1 缩放并非严格线性**：我们 1e5→1e6（10× B1）实测为 10.5× 时间，1e6→2e6（2×）为 2.15×
   ——**越长的运行越趋于稳态低频**（与 §13.3 的"持续 vs 突发"一致）。因此跨 B1 外推时应取**最接近目标
   B1 的实测点**（本表 1e7 用 2e6×5，而不是 1e6×10）。
5. **多线程另算**：本实现不绑定 16/24 线程可达 **8.46×/9.01×**（§13.3）；Prime95 若要同等吞吐需多开
   worker（其 `NumWorkers=1` 时只吃 1 个物理核）。⇒ **同样的装配下我们的总吞吐更高**，
   但**单核算术效率两家同级**。

### 14.4 复现清单

```powershell
# 本实现（B1=1e6，单线程，约 50 s）
echo "(2^4001-1)" | build_vs18\Release\ecm.exe --mont --mont-backend simd --mont-threads 1 `
    -gpucurves 8 --sigma 4242 --tmp-dir .bench_tmp\cmp 1000000 0

# GMP-ECM 7.0.6（单线程，stage 1 only）
python -c "print(2**4001-1)" | D:\code\GIMPS\gmp-ecm\ecm-2025.10.28-win.multiarch\ecm-zen3.exe `
    -param 0 -c 2 1000000 1000000

# Prime95：读 D:\code\GIMPS\p95v3104b05.win64\screen.log 的
#   "M4001 curve N stage 1 ... Stage 1 complete. ... Total time: 67.1xx sec."（NumWorkers=1）
```

> 测量期间的干扰：Prime95 当时正在运行（1 worker）；我们与 GMP-ECM 的测量都在其之外的空闲核上，
> 且频率被钉在 3990/2995 MHz（§13），因此单线程数字可复现（±3%）。

### 14.5 M3001 加入 + 固定 B1 的尺寸扫描（找交叉点）

**Prime95 的 M3001 实测**（用户 2026-09-24 01:36，`NumWorkers=1` 且绑定到 CPU core #1，
`B1=1000000, B2=1000000` 即 stage 1 only）：

```
Using AVX-512 FFT length 256 ; 10.758 bits-per-word below FFT limit
Stage 1 complete. 25612647 transforms ... Total time: 5.667 sec.   (7 条曲线: 5.265 ~ 5.722)
```

⇒ **Prime95 在 M3001/B1=1e6 上 = 5.65 s/曲线（中位）**，25.6M transforms/曲线，**FFT length 256**。

**固定 B1=1e6、单线程、每曲线秒**（本实现 8 曲线/批；GMP-ECM `-c 2`；Prime95 = 14.6 的档位模型，
M3001 处为实测锚点）：

| N（bit） | FFT 档 | **本实现** | GMP-ECM 7.0.6 | Prime95 v31 | 本实现/GMP-ECM | 本实现/Prime95 |
|---|---|---|---|---|---|---|
| M127（127） | 128 | **0.118** | — | 3.86 | — | **32.7× 快** |
| M521（521） | 128 | **0.371** | — | 3.86 | — | **10.4× 快** |
| M1277 | 128 | **0.979** | 2.484 | 3.86 | 2.54× 快 | **3.94× 快** |
| M2203 | 128 | **2.285** | 5.804 | 3.86 | 2.54× 快 | 1.69× 快 |
| M3001 | 256 | **4.137** | 9.586 | **5.65（实测）** | 2.32× 快 | 1.37× 快 |
| M3500 | 256 | **5.164** | 11.656 | 5.65 | 2.26× 快 | 1.09× 快 |
| M4001 | 256 | **6.290** | 14.624 | 5.65 | 2.32× 快 | 0.90× |
| M5755（拟合） | 384 | 12.05 | ~21 | 8.06 | ~1.7× 快 | 0.67× |
| M8527（拟合） | 512 | 24.9 | ~43 | 10.06 | ~1.7× 快 | 0.40× |
| M19701（拟合） | 1024 | ~123 | ~215 | 20.6 | ~1.75× 快 | 0.17× |

（本实现在 1277~4001 bit 的实测点用 §13 的拟合 `≈ 2.98e-7·k² + 3.79e-4·k` 秒外推；
GMP-ECM 同区间比值恒为 2.26~2.55×，故按 2.3~2.4× 外推。）

### 14.6 交叉点：FFT 档位模型给出**两个交叉点**

**Prime95 的成本是"分档阶梯 + 档内恒定"**（用户提供的 FFT 档位表 + iters/s 表，附实测锚点）：

| FFT length | 档位起点（exponent） | iters/s | ns/transform | 25.61M transforms（B1=1e6）预测 | 标定后（×0.913）|
|---|---|---|---|---|---|
| 128 | >2 | 6 061 627 | 164.97 | 4.23 s | **3.86 s** |
| 256 | 2 905 | 4 140 418 | 241.52 | 6.19 s | **5.65 s**（实测锚点）|
| 384 | 5 755 | 2 901 499 | 344.65 | 8.83 s | **8.06 s** |
| 512 | 8 527 | 2 325 936 | 429.94 | 11.01 s | **10.06 s** |
| 640 | 11 309 | 1 801 421 | 555.12 | 14.22 s | 12.99 s |
| 768 | 14 119 | 1 580 507 | 632.71 | 16.21 s | 14.80 s |
| 896 | 16 839 | 1 256 356 | 795.95 | 20.39 s | 18.62 s |
| 1024 | 19 701 | 1 136 615 | 879.81 | 22.54 s | 20.58 s |

- **标定**：表中 iters/s 与 ECM 日志里的 `transforms` 口径略有差别（预测 6.19 s vs 实测 5.65 s，差 9%），
  故绝对量按 M3001 实测锚点缩放（×0.913）；**档间的相对比例**直接用表。
- **档内恒定**的两条依据：① 同一 FFT 长度下"用了多少 bit"不影响变换成本；② 每曲线
  `transforms ≈ 25.7 × B1`，**与尺寸无关**（M3001@1e6 = 25.61M、M4001@1e7 = 257.5M）。

**两条成本曲线的形状**：

| 实现 | 每曲线成本随位宽 | 依据 |
|---|---|---|
| **本实现** | ≈ **n^1.66**（实测 1277→4001；极限为 n²，因固定 tail 开销而偏低）| 折叠域 schoolbook `2n²` madd，n = k/52 |
| **Prime95** | 档内 **n⁰**（常数），**档间 ≈ n^0.80** | iters/s 从 128→1024（8×）降 5.33× ⇒ 每 transform 成本 ∝ n^0.80 |

⇒ 我们**单调上升**、Prime95 是**阶梯**，因此交叉点：

| 交叉 | 位置 | 依据 |
|---|---|---|
| **① ≈ 2 880 bit** | 本实现 3.86 s = FFT-128 档的 3.86 s（该档到 2 904 结束）| 实测 2203→3001 插值 |
| **② ≈ 3 760 bit** | 本实现 5.65 s = FFT-256 档的 5.65 s | 实测 3500→4001 插值 |
| 之后 | **Prime95 永久领先**（2 905 起的 256 档我们只赢到 3 760；5 755 起跳到 384 档后我们已是 12.05 vs 8.06）| — |

**结论（B1=1e6，单线程，每曲线）**：

| 位宽区间 | 谁快 | 倍数 |
|---|---|---|
| ≤ 2 880 bit | **本实现** | M127 **32.7×**、M521 **10.4×**、M1277 3.94×、M2203 1.69× |
| 2 880 ~ 2 904 | Prime95（我们恰好越过 FFT-128 档尾）| ~1.05× |
| 2 905 ~ 3 760 | **本实现**（P95 跳到 256 档）| M3001 1.37×、M3500 1.09× |
| 3 760 ~ 5 754 | Prime95 | M4001 1.11× |
| ≥ 5 755（384 档起）| **Prime95，且优势随档位递增** | 5 755 处 ~1.5×、8 527 处 ~2.5×、19 701 处 ~6× |

**对 GMP-ECM 没有交叉点**：1277~4001 bit 全区间慢我们 **2.26~2.55×**（两者同为多字 schoolbook 类算术；
我们的优势来自 8 lane 批 + 折叠域 + 仿射差分 ladder，与尺寸无关）。

**机制**：交叉点完全由**乘法算法**决定 —— 我们 `2n²` 次 IFMA madd 随 k² 增长；Prime95 的 FFT 在整档内
成本恒定、只在档边界阶梯上升，且**档间每 transform 成本只按 n^0.80 增长** ⇒ 位宽越大越偏向 FFT。
**实用判据**：**≲3700 bit 的合数用本实现划算（尤其 2000 bit 以下优势成倍~数十倍）；
≳6000 bit 交给 Prime95（GWNUM FFT）**；3700~6000 之间两家互有胜负、按档位具体算。

> M4001@1e7 的中位数 60.1 s（8 条，46.8~67.3）对应每 B1 单位 6.01e-6 s，比 M3001@1e6 的 5.65e-6 高 6%
> （若同档应为 0）⇒ 那批运行受当时功耗/频率设置变动影响，故本节以 **M3001@1e6 为锚**。
> 若你能补跑 **M4001/B1=1e6**（预测 5.65 s）与 **M5200 或 M6000/B1=1e6**（逼出 384 档边界），
> 两个交叉点即可从"模型 + 插值"升级为"直接实测"。

---

## 15. 可行性研究：亚二次乘法（Karatsuba / Toom-Cook）与底层优化（2026-09-24）

背景：我们的模乘是**折叠域 schoolbook**，`2n²` 条 madd（n = 52-bit limb 数），复杂度 O(n²)；
Prime95 的 GWNUM 是 FFT，O(n log n log log n)。两条候选路径都被实测/建模评估如下。

### 15.1 路径二：Karatsuba / Toom-Cook —— **实测更慢，不采纳**

**先例**：仓库已有 GPU 版（`kernels/opencl/mont_mul/mont_mul_karatsuba_2048b.cl`），
`docs/DEV_COOP_KARATSUBA_2048.md` 明确**放弃三分法**，理由是
"修正项需要逐 limb 传播，复杂度与正确性风险远超收益"，改用 4 子积（只为 4 线程并行，乘法次数不减）。
**但那个论证针对 32-bit limb 的 CIOS 结构**；我们的折叠域把所有中间量累加进 64-bit 列，
理论上可以把三分法的"修正项"降成几次加/减 pass。于是写了原型实测（`tools/bench/mers_karatsuba.cpp`）：

```
a = a0 + a1*X,  b = b0 + b1*X,  X = 2^(52h),  h = ceil(n/2),  m = n - h
a*b = m0 + (m1 - m0 - m2)*X + m2*X^2     m0 = a0b0, m2 = a1b1, m1 = (a0+a1)(b0+b1)
```

**实测（同进程、同一份折叠域 tail，与生产 `ifma_mont_mul` 逐字节对拍）**：

| n52（N） | 生产 schoolbook | Karatsuba levels=1 | levels=2 |
|---|---|---|---|
| 25（M1277） | 467 ns | 1101 ns（**+136%**）| 1862 ns（+299%）|
| 58（M3001） | 2018 ns | 3185 ns（**+58%**）| 4659 ns（+131%）|
| 77（M4001） | 3446 ns | 4883 ns（**+42%**）| 6616 ns（+92%）|
| 154（M8000） | 12840 ns | 14751 ns（**+15%**）| 16724 ns（+30%）|

⇒ **每一个尺寸都更慢**，且惩罚随 n 减小而放大；按 (77,+42%)→(154,+15%) 线性外推，
**盈亏平衡点约 n ≈ 197 limb ≈ N ≈ 10 250 bit** —— 而那时 Prime95 已经比我们快 6 倍以上（§14.6）。
**结论：在我们能赢的尺寸（≲4 000 bit）它纯亏；在我们落后的尺寸（≳6 000 bit）它也救不了。**

**为什么理论上的 −25% 从未出现**（这是本次研究最有价值的部分）：

1. **我们的 schoolbook 太密**：乘积相位 91~96% 的 madd 屋顶，1 条 madd 只摊 ~2.75 条其他指令；
   而 Karatsuba 的修正 pass 是**每列 ~4 条指令、零 madd**（load/add/shr/and/store）。
   省下 `0.5n²` 条 madd，却要付出 ~O(n) 条×4 的修正 ⇒ 在 n≈50 附近正好抵消。
2. **减法必须先归一化**：带借位传播的减法只在**规范 digit** 上有意义；
   未归一化累加器的列值可达 `n·2^52`，其"借位"不是 0/1 ⇒ 相减前必须多跑一次 carry pass
   （`ifma_carry_pass`），每个子积还要各自归一化 ⇒ 修正开销再翻倍。
   **这正是 GPU 文档所说的"修正项要逐 limb 传播"**，只是在我们这里表现为 O(n) 的 pass 而不是逐 limb。
3. **三个实测踩到的坑**（记录以免复犯）：① 累加器宽度 `2n+2` 对奇数 n 差 2 列，**越界写**污染了
   相邻 scratch 里的 m0；② 子积未清零就累加；③ **奇数 n 时 `a1 = floor(a/X)` 只有 `n−h` 个 limb**，
   按 h 个 limb 处理会多读一列并把 `a0+a1` 算错（表现为 n=25/77 失配而 n=58/116 通过）。

**Toom-Cook**：Toom-3 的乘法次数是 `5(n/3)² ≈ 0.556n²`（比 Karatsuba-1 的 0.75n² 更好），
但插值需要除以 2/3、求值点更多 ⇒ 修正 pass 比 Karatsuba 更多，按上面的结论只会更亏；
且它同样不改变"我们赢在 ≤4 000 bit、输在 ≥6 000 bit"的格局。**未做实测即放弃，理由已充分。**

### 15.2 路径一：底层/访存/汇编优化 —— **实测只剩 ≤5%，且不在访存也不在批宽**

已实测的边界（§11、§13）：

| 事实 | 数值 | 含义 |
|---|---|---|
| 乘积相位 madd 利用率 | 91%（sqr）/ 96%（mul）的本机屋顶（4.0 条 zmm madd/ns）| **madd 端口已饱和** |
| 工作集敏感性 | 1→16 个操作数缓冲 < 1% | **不是访存受限**，L1 常驻 |
| 辅助 SoA pass | 已从 30 趟/bit 降到 18 趟（§11.3b，−8%） | 已吃掉 |
| tail（normalize/fold/drain/canonical） | 每个域运算 15~25%，**串行 carry 链**、~0.7 ops/cycle | 唯一还剩的靶子 |

#### 15.2.1 批宽 8→16 lane：**实测 0 收益（原判断错误，已更正）**

我原先在 §15.2 写过"加宽批可摊薄 load/store 固定开销，估 −10~15%"——**这个推理是错的**：
把列从 1 条 zmm 变成 2 条 zmm 时，**访存体积随 lane 一起翻倍**，比例不变：

```
8  lane: 每迭代 4 madd + (1 b-load + 2 t-load + 2 t-store + 2 add)  = 4 : 7
16 lane: 每迭代 8 madd + (2 b-load + 4 t-load + 4 t-store + 4 add)  = 8 : 14
```

唯一不随 lane 变的是循环索引/分支（每 ~11 条指令 1 条）⇒ 理论收益 <1%。
`tools/bench/lane_width.cpp` 在同一进程里把两种循环形状都跑了一遍（同 n、同 reps、取 best-of-7）：

| 变体 | n52=58 | n52=77 | n52=154 |
|---|---|---|---|
| 8 lane/列（生产形状） | 30.80 per-lane madd/ns | 30.54 | 31.46 |
| **16 lane/列（2×zmm）** | 30.89（**+0.3%**）| 30.72（**+0.6%**）| 31.41（**−0.1%**）|
| 8 lane + 寄存器分块（8 列累加器常驻，朴素实现） | 11.43（**−63%**）| 11.36（−63%）| 11.80（−63%）|

- **16 lane = 0**：批宽不是杠杆。
- **寄存器分块朴素实现 −63%**：把输出列搬进寄存器确实省掉了 t 列的载入/存储，但**代价是按块重复读取 `b` 列**
  （每块 8n 次 b 载入 × 2n/8 块 ≈ 2n² 次），比原来的 n²/2 次更多。要真正做成需要"两个操作数都分块"的
  寄存器/缓存复用，而 32 个 zmm 寄存器装不下 n=58~77 所需的累加器+操作数瓦片 ⇒ **此路不通**。
- 生产版 3.85 条 zmm 指令/ns ÷ 屋顶 4.0 条/ns = **96%** ⇒ 乘积相位确实做完了。
  （注：`lane_width` 打印的"参考屋顶"行不可信——编译器会把常量操作数的 `madd52lo` 强度削减成加法；
  屋顶一律以 `mers_sqr_phases` 的 8 条独立链测量为准：31.8~32.2 per-lane = 4.0 条指令/ns。）

⇒ 路径一**剩余空间 ≤5%**，只剩 tail 的串行 carry 链（把 2n 列归一化拆成两条独立链做 ILP，约 4%；
或把 canonical 与 drain 融合，约 3%）——两者都要动 $2^k=1$ 折叠的边界，属"高风险区"（§3 注释里的非规范值 bug）。

#### 15.2.2 OpenCL 的经验为什么帮不上忙

仓库的 OpenCL kernel 是 GPU 目标（256 线程/工作组、LDS 暂存、合并访存、占用率调优），
其文档（`ECM_OPERATOR_ANALYSIS.md`、`DEV_CPU_MONT_AVX_PLAN.md`、`MONT_UNROLL_I24_MAD24_OPTIMIZATION_CN.md`）
讨论的是 GPU 占用率与访存合并，而我们的瓶颈是**端口饱和 + 串行 carry 链**，两者没有交集。
唯一可迁移的是"寄存器分块"思想，而它已在 15.2.1 被实测否决。

### 15.3 结论与建议

| 路径 | 预期收益 | 成本/风险 | 建议 |
|---|---|---|---|
| **Karatsuba / Toom-Cook** | **−15%（n=154）→ +136%（n=25），均值纯亏**；盈亏平衡 ≈10 250 bit | 高（本次原型踩了 3 个坑） | **不做** |
| **自研 FFT/NTT** | 在 ≳10 000 bit 才有意义；但那正是 Prime95 的强区（已快我们 6×）| 极高（误差控制/旋转因子表/尺寸分档） | **不做**（战略上交给 Prime95）|
| ~~16-lane 加宽批~~ | **实测 +0.3%/−0.1% ⇒ 0**（原 −10~15% 的判断已更正，见 15.2.1）| 中 | **不做** |
| ~~寄存器分块~~ | 朴素实现 **−63%**；要真做需操作数也分块，寄存器不够 | 高 | **不做** |
| **tail carry 链 ILP/融合** | **≤5%**（−4% + −3%，两者不完全叠加）| 中高（$2^k=1$ 折叠边界，历史上有非规范值 bug）| 唯一剩余项，**风险/收益比不划算，建议搁置** |

**一句话**：复杂度换不来收益（我们的 schoolbook 太密、修正开销是 O(n) 的 4 指令/列）；
**批宽与访存也不是杠杆**（16 lane 实测 0、分块实测 −63%、工作集不敏感）；
乘积相位已在**本机 madd 屋顶的 96%**。⇒ **字段层性能优化到此为止**，
剩余只有 tail 的串行 carry 链（≤5%、高风险），更大的收益只能来自算法层面（链/FFT，均已论证不划算）
或换硬件（桌面 Zen 5 的 2 条 zmm madd/周期会让同一份代码再快一截，那属于"换机器"而非"改代码"）。

> 复现：`tools\build_tool.bat tools\bench\lane_width.cpp` 后
> `build_vs18\tools\lane_width.exe 58 2000`（打印三种循环形状的 per-lane 吞吐）；
> Karatsuba 原型见 15.1 末尾的复现命令。

---

## 16. 新项目可行性：Prime95 高效多项式 stage 2 的移植（2026-09-24）

### 16.1 现状：本仓库没有自己的 stage 2，靠"交接"完成

- CPU 两条路径（Edwards / Montgomery）**只有 stage 1**（命令行接受 `B2` 但只用于显示，`B2=0` 即关闭）。
- 队列能解析 `ECMSTAGE2=` 行（`ecm_worktodo.cpp:133`，注释写明是 "CUDA-oriented, unchanged" 格式），
  驱动只做参数搬运（`ecm_driver.cpp:2694`）；但 **GPU/CPU 侧都没有 stage-2 内核**——
  全仓库 grep `stage2|continuation|pairing` 只命中 driver/config/worktodo 三个文件，无任何 kernel。
- 实际生产做法（已互通验证，§9.2）：stage 1 写共享存档 `m{n}_{b1}.save` → 交给
  Prime95（`ecm_p95feeder`）或 gmp-ecm `-resume` 做 stage 2。

### 16.2 Prime95 的 stage 2 到底是什么（源码事实，非猜测）

**两套算法共存，按内存选择**：

| 常量 | 值/位置 | 说明 |
|---|---|---|
| `ECM_STAGE2_PAIRING` | 0，`ecm.cpp:1470` | "Old fashioned prime pairing stage 2"（经典 BSGS 素数配对）|
| `ECM_STAGE2_POLYMULT` | 1，`ecm.cpp:1471` | "FFT/polymult stage 2"（多项式乘法版，主力）|
| 选择器 | `ecm.cpp:5954` | 对两种 type 分别估价取优；**`:6013`：只有能放进 ≥200 个 gwnum 临时量时才选 polymult**，否则配对 |
| 低内存退路 | `ecm.cpp:1069` | windowed pairing，窗口 100 ⇒ 额外 ~65 MB |
| D 值表 | `poly_D_data[]`，`POLY_MAX_D = 164,894,730`、`POLY_MAX_RELPRIMES = 14,100,480` | 预置的一组 (D, first_missing_prime) 参数 |
| Ftree | `:1579` | 多项式的乘/余树，可按内存决定常驻、落盘或重建 |
| pairmap | `:874`，`MaximumBitArraySize` 默认 250 MB | 配对位图，超限则分块（会增加 setup 成本）|
| 压缩与归一 | `polymult_preprocess`、`normalize_pool`（`:717`、`:5342`）| 多项式压缩、按内存约束减少乘法次数 |
| **成本口径** | `est_stage2_transforms`、`est_stage2_polymult`、`est_stage2_stage1_ratio`（`:747-752`）| **全部以 gwnum transform 为单位** |

⇒ 这套东西的**设计前提就是 FFT**：它的算法（产品树 + 多点求值）、内存规划（pairmap/Ftree 分块）
与调参（D、多项式的正常化池）都是围绕"乘法是 FFT"来做的。

### 16.3 实测成本（用户日志，M4001，B1=1e7，B2=124,155,521,490 = 12415×B1，单 worker）

```
Stage 2 init complete.   3,139,541 transforms   Time: 2.65 s
Stage 2 complete.       14,640,228 transforms   Total time: 13.8 s
Optimal B2 is 12415*B1 ...  Curve is worth 2.69 B2=100*B1 curves
Estimated stage 2 vs. stage 1 runtime ratio: 0.312
```

⇒ **stage 2 ≈ 16.5 s = 0.31 × stage 1（67.2 s）**，而且这是把 B2 抬到 **B1 的 1.24 万倍**之后的成本
（收益面：这条曲线"值 2.69 条 B2=100·B1 的曲线"）。**这才是"高效"二字的含义。**

### 16.4 我们的算术能不能做？—— 两条路分别判定

**(a) polymult 路：不可行（等于重写 GWNUM FFT）。**
多项式乘法在这里必须"快"：D 个相对素数、系数 4000 bit，Kronecker 代入后是
`D × 4000 bit` 量级的**单次大整数乘法**（D=10^6 时约 **4×10^9 bit ≈ 7.7×10^7 个 52-bit limb**）：

| 方法 | 该规模下的 madd/运算量 | 判定 |
|---|---|---|
| 我们的 IFMA schoolbook（n²）| (7.7e7)² ≈ **6×10^15 madd** | ✗ 天量 |
| Karatsuba（n^1.585，§15.1 已实测在小尺寸就亏）| ≈ **10^12** | ✗ |
| Toom-3（n^1.465）| ≈ **10^11** | ✗ |
| FFT（n log n log log n）| ≈ 10^9 量级 | ✅ 但这正是 GWNUM |

⇒ 移植 polymult = 自研一个 GWNUM 级 FFT **外加** Ftree（可落盘）、pairmap 分块、polymult_preprocess
压缩、normalize_pool 调参。与 §15.3 "自研 FFT 不做" 同一结论，且工程量更大。

**(b) pairing 路：可行，但只在 B2/B1 ≲ 100 时有意义。**
算法简单（经典 BSGS：预存 D 个 baby 点的 x，逐个 giant step 做乘积 `∏(x_jD − x_r)`），我们的原语够用。
成本模型（每次配对 ≈ 1 次模乘 + 1 次减法；配对次数 ≈ 区间内候选素数数 × 0.5~1）：

| 量 | 值 | 来源 |
|---|---|---|
| 我们 batched-8 的一次模乘 | 2020 ns/8 曲线 ⇒ **0.25 µs 核心时间/次** | §11 实测（n52=58）|
| 候选素数数（B1→B2） | ≈ (B2−B1)/ln B2 | 素数定理 |

以 B1=1e6（我们 stage 1 = **4.14 s/曲线**）为基准：

| B2/B1 | B2 | 候选素数 | pairing stage 2 | 相对 stage 1 | 相对 Prime95（同 B2）|
|---|---|---|---|---|---|
| 100 | 1e8 | 5.2e6 | **1.3 s** | **0.32×** ✅ | — |
| 1000 | 1e9 | 4.8e7 | 12 s | 2.9× | — |
| 1e4 | 1e10 | 4.5e8 | 113 s | 27× | — |
| 12415 | 1.24e11 | 4.9e9 | **600~1200 s** | **150~300×** | **36~73× 更慢**（Prime95 16.5 s）|

外加内存与 D 的取舍（**不是硬约束**）：pairing 要常驻 D 个 baby 点 —— 单曲线 D=1e5、n=77 时约 61 MB，
8 lane 批则约 490 MB。但**配对总数与 D 几乎无关**：

```
配对次数 ≈ |Rs| × #giants ≈ (0.2·D) × (B2−B1)/(D·ln B2) ≈ 0.2·(B2−B1)/ln B2      ← D 被消掉
```

⇒ **D 只是"内存 ↔ setup/giant-step 开销"的旋钮**：取 D=1e4 时 8 lane 批的 baby 表仅约 **49 MB**，
而配对总数不变。所以对 pairing 路而言**内存可控，真正的成本就是那 ~1 次模乘/候选素数**。

### 16.5 判定与建议

| 方案 | 结论 |
|---|---|
| 移植 **polymult**（主力算法）| ✗ **不可行**：等于重写 GWNUM FFT + Ftree/pairmap/压缩全栈 |
| 移植 **pairing**（低内存退路）| ⚠ 可行，但只在 **B2/B1 ≲ 100** 划算（+0.3× stage 1 换 stage-2 收益）；到 1e4 就是 27× stage 1 |
| **维持现状：我们算 stage 1，stage 2 交给 Prime95 / gmp-ecm** | ✅ **推荐**（三条通路已互通：`.save` 交接、`ECMSTAGE2=` 队列行、`-resume`）|
| 自研 pairing + 小 B2 作为"无 Prime95 环境兜底" | 可选的小项目，**不要指望替代 polymult** |

### 16.6 真正还有收益的地方：交接本身（附一个关键兼容性结论）

- **成本结构**：我们 stage 1（B1=1e6）4.14 s/曲线，Prime95 stage 2（B1=1e7/B2=1.24e11）16.5 s/曲线
  ⇒ 在整条流水线里 **stage 2 不再是零头**；随着我们的 stage 1 变快，瓶颈会移到 stage 2。
- **指数语义必须匹配**（重要，但差异是"窗口"而非系统性）：Prime95 的 stage 1 用 `12·lcm(1..B1)`
  （choose12），我们默认 `lcm(1..B1)`（`--mont-torsion 1`，与 gmp-ecm `-param 0` 对齐）。
  两者给出的点**不同**：`12·lcm` 只是把 2-adic 指数 +2、3-adic 指数 +1
  （`v2(lcm) = ⌊log2 B1⌋`，`v3` 同理），因此**只在群阶的 2、3 幂次恰好比 lcm 多出 ≤2 / ≤1 的窗口内**
  才会有差异（概率小但非零，不是"系统性漏掉一半曲线"）。不过既然 stage 2 会把 stage-1 的点继续用下去，
  **严格对齐时仍应匹配语义**：
  - **要把 stage-1 结果交给 Prime95 做 stage 2，用 `--mont-torsion 12`**，这样整条流水线与
    Prime95 标准运行同构；
  - **要与 gmp-ecm `-param 0` 逐点对齐（本项目的验收口径 Q1），用 `torsion 1`，stage 2 交给 gmp-ecm**
    （gmp-ecm 的 stage 1 同样是 `lcm`，语义自洽）；
  - 两个目标冲突，**不要混用**：给定 B1，要么"我们 torsion=1 + gmp-ecm stage 2"，
    要么"我们 torsion=12 + Prime95 stage 2"。
- **速度比（§14 实测，用于规划分工）**：M3001 档我们 stage 1 快 **1.37×**（4.14 vs 5.65 s @B1=1e6），
  M4001 档基本**持平**（~67 vs 67 s @B1=1e7）；只有小尺寸才大幅领先（M1277 约 3.9×）。
  ⇒ 在 4000 bit 级、大 B1 的 GIMPS 场景里，"我们算 stage 1 + 它算 stage 2" 的收益主要来自
  **并行/调度与流水线**，而不是单核算术优势。
- 可做的交接优化：worker 持续产出 `.save` 而不空转、按曲线粒度批量交接、
  让 `ecm_p95feeder` 的队列深度与我们的产出速率匹配（避免 Prime95 侧饥饿或积压）。

> 结论一句话：**Prime95 的 stage 2 不是"一段可以搬过来的代码"，而是一整套以 FFT 为前提的
> 多项式算法 + 内存规划系统；我们的 IFMA 算术做不了它的主力路径（polymult），做它的退路（pairing）
> 只在 B2/B1 ≲ 100 时划算。正确的分工仍是"我们算 stage 1 + 它算 stage 2"，并保证指数语义一致；
> 在 4000 bit 级大 B1 场景下我们 stage 1 的优势只有 ~1.0~1.4×，所以交接/流水线的优化比继续压
> 单核算术更值得做。**

### 16.7 Prime95 究竟怎么消化 gmp-ecm 的 param0 / param3 存档（源码答案）

**它把 gmp-ecm 的 resume.c 直接搬了进来**：`ecm.cpp:32-34` → `#include "resume_gmp.c"`（12 KB，
即 gmp-ecm 的 `resume.c`）。所以字段级解析是原生的：`METHOD / X / Y / Z / N / SIGMA / A / B1 /
PARAM / PROGRAM / CHECKSUM`，缺 `PARAM=` 时**默认 param 0**（`resume_gmp.c:161`：
`*param = 0;`，注释 "For compatibility reason, param = ECM_PARAM_SUYAMA by default"）——
这正是我们 writer 对 param 0 省略 `PARAM=` 的依据 ✓。必填字段是
`METHOD && X && SIGMA && B1`（`resume_gmp.c:284`），我们的行都满足 ✓；CHECKSUM 也按
`B1 · σ · (param+1) mod CHKSUMMOD` 一致计算（`:297-303`）✓。

**拿到这些字段之后的全部处理**（`ecm.cpp:7348-7380`，即 `w->gmp_ecm_file != NULL` 分支）：

```c
read_resumefile_line (..., x, n, a, sigma, &param, &b1, stage1_program);
if (param == 0)      ecmdata.sigma_type = 1;   // gmp-ecm param0 → Suyama 曲线族
else if (param == 3) ecmdata.sigma_type = 3;   // gmp-ecm param3
else { "Unsupported GMP-ECM param=%d"; 报错退出 }   // 只支持 0 与 3
ecmdata.sigma = atoll (sigma 的十进制串);
ecmdata.B     = (uint64_t) b1;                 // B1 **取自文件**
ecmdata.state = ECM_STATE_MIDSTAGE;            // ← 直接进入 stage 2
mpztog (x, ecmdata.Qx_binary);                 // ← 文件的 X 就是 stage-2 输入点
goto restart3;                                 // 从不重算 stage 1
```

⇒ **关键答案：`PARAM=` 的唯一作用是"用哪条公式把 σ 变成曲线系数 A"**（param0=Suyama、
param3=gmp-ecm 的第三种参数化，其他值直接报错）。**指数差异它根本不处理——因为它从不重算
stage 1**：`X` 被当作既有事实消费，`B1` 也取自文件，后续只是在这个点上再乘 (B1,B2] 的素数。
`lcm` 与 `12·lcm` 的区别早已烘焙进 `X`，所以整条链路自洽；差别只体现在**覆盖集**：
`lcm(1..B1) ⊗ primes(B1,B2]` 而不是 Prime95 原生的 `12·lcm ⊗ primes(B1,B2]`。
⇒ **要用 Prime95 做 stage 2 时，我们 stage 1 用 `--mont-torsion 12` 即可对齐覆盖；存档格式无需改动**
（仍是"省略 PARAM="的 param0 形态，Prime95 会正确映射成 `sigma_type=1` 的同一曲线族）。

#### 16.7.1 顺着这段代码发现的两个互操作坑（其一已修）

**① σ ≥ 2^63 会被截断（已修）**：Prime95 读 σ 是 `mpz_get_str()` → **`atoll()`**（`ecm.cpp:7375`）。
σ ≥ 2^63 溢出 ⇒ Prime95 按**被截断的 σ 重建出另一条曲线**，而它加载的 `x` 来自我们的曲线
⇒ stage 2 在"错误的曲线"上白跑（只损失效率，不会产生错因子：ECM 的 gcd 只会吐出 N 的真因子）。
Prime95 自己只生成 σ < 2^53（`ecm.cpp:7416`：`(rand()&0x1F)<<48 + (rand()&0xFFFF)<<32 + rdtsc 位`）。

- 我们的 **Edwards 路径**（`random_sigma_u64()`）本来就是照抄这个构造 ⇒ σ < 2^53 ✓ 安全；
- 但**我写的 Montgomery 路径**用了 `std::mt19937_64` 的**全 64 位** ⇒ 约一半曲线的 σ ≥ 2^63 ✗
  ⇒ **已改为同一生成器**（`src/core/ecm_driver.cpp`，`run_mont_stage1`），并在
  `--edwards --sigma <64 位>` 且 `σ+curves ≥ 2^63` 时打印警告、建议交给 gmp-ecm 做 stage 2
  （gmp-ecm 的 reader 是 mpz 的，不受 `atoll` 限制）。
- 验证：随机 64 条曲线存档 σ 最大 1.52e15（< 2^53）✓；`--edwards --sigma 1.8447e19` 触发警告 ✓。

**② `N=` 字段只对 `ECMSTAGE2N=` 形式重要**：逐行读取时 `N=` 被**跳过**（`resume_gmp.c:225`），
N 来自 worktodo 的 `k,b,n,c`；只有"行里没有 N"的 `ECMSTAGE2N=` 变体才会让 Prime95
去文件里找 `N=`（`ecm.cpp:6890`，找不到就报 "Could not find line containing N= in file"），
而那条路用 `mpz_set_str(buf, 0)` 解析 ⇒ **只认纯整数**（`0x` 前缀可），
表达式形如 `N=(2^1277-1)/f` 解析不了。⇒ 若将来要支持 `ECMSTAGE2N=`，
存档里的 `N=` 应写**纯十进制整数**；`ECMSTAGE2=`（带 k,b,n,c）则无所谓。

> 附：仓库现有的交接是**另一条通路**——`ecm_p95feeder` 把我们的二进制 stage-1 存档
> （`e{n:07d}_c{k}.tmp`）拷成 Prime95 自己的 `e{n:07d}`，再往 `worktodo.add` 追加一条
> `ECM=k,b,n,c,B1,B2,1,<σ>`（`ecm_p95feeder.cpp:298-313`）⇒ Prime95 用**它自己的 resume 文件**
> 直接进 stage 2（`ecm.cpp:7321`："We've finished stage 1, resume stage 2. The save file contained
> normalized Q"）。两条通路殊途同归：**Prime95 只做 stage 2，用我们产出的点**。
> 新的 Montgomery 文本存档（§9）走的是 `ECMSTAGE2=` + gmp-ecm 文本格式这条通路（本节 16.7 描述的就是它）。

## 17. 中途检查点与进度条（2026-09-24，用户要求：「参考 CUDA param3，顺便做 Edwards 那套进度界面」）

### 17.1 需求与边界（用户拍板）

| 项 | 结论 |
|---|---|
| 检查点格式 | **只需内部自洽**（同一程序写得进、读得出），不要求与 gmp-ecm / Prime95 互通 |
| 存档（`.save`） | **必须与别的软件互通** ⇒ 本次一行没动它的语义（只修了一个既存 bug，见 §17.6） |
| 参考对象 | GPU/OpenCL 路径的 `opencl_ecm_checkpoint_*`（整块曲线缓冲 + 一个全局指数位偏移，按 `ckpt_seconds` 定期落盘、启动时校验 `curves`/`s_num_bits` 后恢复）与 Edwards 路径（每曲线 `.ckpt` + 进度条 + 速率窗口） |
| 键名 | ini `ckpt_seconds`（旧 `gpuckpt_seconds` 仍接受，警告一次）；CLI `--ckpt <秒>`（旧 `-gpuckpt` 仍接受） |

`ckpt_seconds = 0` = 不做定时保存；**Ctrl+C 仍然保存一次**（这一条几乎不花钱，却把「被打断」从
「白跑」变成「续跑」）。

### 17.2 为什么这条 ladder 的检查点特别便宜（数学事实，不是工程技巧）

ladder 迭代不变式（§2.5）：

```
每次迭代开始时：p0 = [k]P ，p1 = [k+1]P      （k = 已消耗的指数位数）
```

而曲线常数 `a24`、差分加法的 `xdiff` **都是 σ 的函数**（`xdiff` 是起点 P 的仿射 x），不是随 k
变化的状态。⇒ 一个完整的续跑点就是

```
(bitnum = k, p0, p1) + 识别这批工作的参数 (N, B1, torsion, σ)
```

一条曲线只要 **4 个域元素**（X0,Z0,X1,Z1）；对照 Edwards：它的 ladder 是 NAF 分块推进的，恢复
还要窗口字典/分块相位 ⇒ 这条路径天生更适合做检查点。已完成的曲线另存结果（miss 存 x，hit 存
因子），否则被打断的一次运行会重算所有算完的曲线。

### 17.3 文件格式（内部，`src/core/ecm_mont_ckpt.{h,cpp}`）

每条曲线一个文件，与 .save 同目录：

```
<tmp_dir>/<save stem>_c%07u.ckpt        例：saves/m3001_1e6_c0000017.ckpt
```

正文（明文，理由见下）：

```
MPA-MONT-CKPT 1
N=<hex>              ← 完整 N，同一性的最强判据
B1=1000000
TORSION=12           ← 1 = lcm（gmp-ecm param 0），12 = Prime95 choose12
SBITS=1442099        ← 指数位数（B1 的另一种指纹）
CURVE=17
SIGMA=1234567890123
LIMBS=0
FIELD=ifma | mpn
STATUS=INFLIGHT | DONE
BITNUM=655360
X0=.. Z0=.. X1=.. Z1=..              （INFLIGHT 且 BITNUM>0 时）
HIT=0|1 ; XOUT=.. | FACTOR=..        （DONE 时）
CHECKSUM=<fnv1a-64 of every byte before this line>
END
```

* 三种记录：`INFLIGHT`（BITNUM>0，带 ladder 中间态）、`INFLIGHT`+`BITNUM=0`（**种子**：只钉住该曲线
  的 σ）、`DONE`（带结果）。
* **明文而不是二进制**：它在不可预测的时刻被多个 worker 线程写（定时器 / Ctrl+C），崩溃后是人要
  盯着看的东西；`CHECKSUM`+`END` 让「写了一半」的文件被明确拒绝，而不是当成有效状态续跑。
* 状态里的 `X0/Z0/X1/Z1` 是**普通域**（mod N）的整数，不是 IFMA 内部表示：写出时走
  `ifma_to_mpz_lane`，读回时走 `ifma_from_mpz_lane`。这样同一份检查点可以由标量后端接着跑，
  也可以换 `field` 层接着跑（§17.6 T3 就是用这条性质做的验证）。
* 拒绝条件（读失败一律当「没有检查点」，不是当「空状态」）：版本不符 / N 不同 / B1 不同 /
  torsion 不同 / s_bits 不同 / 缺 END / 校验和不符 / `INFLIGHT` 缺状态 / `DONE` 缺结果。

### 17.4 运行时行为（`run_mont_stage1`）

1. **种子写盘**：若启用检查点，每条曲线开工前先写一条 `BITNUM=0` 记录。理由：随机 σ 每次运行都
   不同，若不钉住，被打断后的续跑会对「还没轮到」的曲线抽一批**新** σ，曲线集合悄悄变了。
2. **预扫描**：逐曲线读检查点 ⇒ 采纳其中的 σ（`-sigma` 固定值时要求一致）、恢复 `DONE` 曲线的结
   果（x / 因子 / hits）、记下 `bit_off`（已消耗位数）。
3. **组任务**：按 `bit_off` 降序分组再切批。一个 SIMD 批的所有 lane 共享一条指数前缀 ⇒
   **同批必须同起点**；恢复后分组只会让每个「被打断的旧批」浪费至多一个不满的批（1000 条曲线、
   24 线程量级下 <0.5%），全新运行时所有 `bit_off` 都是 0，退化成原来的顺序切批。
4. **定时自动保存 = 暂停 → 写盘 → 继续**；**SIGINT = 写盘 → 结束整个运行**（打印「rerun the same
   command line to resume」并返回 `ECM_ERROR`）。两者由同一个回调返回 1 触发，必须靠
   `g_stage1_stop` 区分 —— 第一版没有区分，结果「1 秒的自动保存间隔」直接变成了「运行 1 秒就退出」，
   被 E1 端到端测试抓出来（§17.6）。
5. **成功后删掉检查点**：`.save` 已经是持久产物（也是互通产物），删掉才能让「再跑一次同样的命令」
   是**一次全新的运行**，而不是把上次结果原样重放。
6. **断点粒度** `chunk = nbits/512` 向上取到 2 的幂、下限 4096（B1=1e6 时 4096 bit，约 0.3% 的
   进度更新粒度）。热循环里用的是**倒计数比较**而不是 `i % chunk`：后者每 bit 一次整数除法，而
   每 bit 总共只有约 10 次域乘法，实测能看出 1% 量级。

### 17.5 进度显示（与 Edwards 共用一套）

`stage1: [========>          ]  61.2%  9.8/16 (~0.12 s/curve)  elapsed 1.2s  ETA 0.8s`

* 复用 Edwards 的 `g_stage1_bar` + `Stage1SpeedMeter`（**速度样本**窗口而不是时间平均，回推
  per-curve 与 ETA，避免并行时跳变）+ 非 TTY 时的衰减整行输出（`emit_progress_line`）。
* 工作量的单位是「曲线」：已完成曲线数 + Σ(每条在跑曲线已消耗位数 / s_bits)，每曲线一个原子计数
  器，回调里求和 ⇒ 多线程下单调、不需要锁。
* 中断时进度条不完整（`mark_as_completed` 不会被调用），与「被暂停」的语义一致。

### 17.6 验证（两个测试，全绿）

**单元验证** `tools/test/mont_ckpt_verify.cpp`（N = 2^1277−1，B1=5000，chunk=512，24 项断言）

| 组 | 内容 |
|---|---|
| T1 | 标量 ladder：**每个 chunk 都暂停**并从写下的状态续跑 14 次 ⇒ x 与 gcd 与一次跑完逐位相同 |
| T2 | SIMD 批：8 条 lane 同时暂停/续跑 14 次 ⇒ 8 条 lane 的 x 与 gcd 全部相同 |
| T3 | **跨路径交接**：SIMD 在 512 位处暂停 → 状态转成普通 mpz → **标量 ladder** 从 512 续跑 ⇒ 与不中断的 x/gcd 相同（这一项同时钉死了 to_mpz/from_mpz 的往返精度） |
| T4 | 文件往返 + 拒绝：B1/torsion/s_bits/N 不符、截断（缺 END）、改一个字节（校验和）、`DONE` 缺结果，全部被拒；改用例作对照仍被接受 |

**端到端** `tools/test/test_mont_checkpoint.ps1`（M1277，B1=1e6）

| 组 | 内容 | 结果 |
|---|---|---|
| E1 | 固定 σ：不中断跑一遍；另起进程跑到 3 s **硬杀**（不给收尾机会）→ 32 个 `.ckpt`、其中 32 个含 ladder 中间态 → 重跑同一命令行 ⇒ 32 行曲线内容（SIGMA/B1/N/X/CHECKSUM）与不中断那次完全一致，跑完检查点被删 | PASS |
| E2 | 随机 σ：被杀的那次把 32 条曲线的 σ 钉住了 → 续跑必须**采纳**它们（存档里 32/32 条 σ 都与杀进程前的 `.ckpt` 一致） | PASS |
| E3 | 标量后端（`--backend gmp`，8 曲线 / 2 线程）：同样杀 + 续跑，8 行内容一致 | PASS |

> 说明：E1 比较的是**去除 `WHO=`/`TIME=` 后的曲线内容**，不是逐字节——存档每行都带用户名与
> 时间戳，两次运行本来就不可能逐字节相同。

**顺带修掉的既存 bug**：`ecm_append_save_lines_mont()` 原先只拿到 `firstsigma`，按
`firstsigma + i` 推第 i 条曲线的 σ。固定 σ 模式下这是对的，**随机 σ 模式下写出来的 σ 与同一行
的 X 不是同一条曲线** —— 交给 Prime95 做 stage 2 时它会按错的 σ 重建另一条曲线，然后静默地在
一条自己从没有过点的曲线上搜索。现在签名收 `const uint64_t *sigmas`（完整数组）。

### 17.7 性能：检查点几乎不花钱（同进程交替 A/B）

第一版用 `i % chunk` 判断点，先在**同一进程内交替**比较三种变体（`tools/bench/mont_ckpt_ab.cpp`，
旧版 ladder 从 git HEAD 取出、重命名符号后链进同一个二进制；N=2^3001−1，B1=1e6，3 轮）：

```
  old ladder          : 24014.95 ns/bit  (34.632 s/batch)
  new, cb = NULL      : 24197.51 ns/bit  (+0.76%)
  new, cb = progress  : 23877.60 ns/bit  (-0.57%)   ← 驱动实际安装的那种回调
```

结论：**在噪声内（±1%），不需要为检查点牺牲 ladder 速度**。同机驱动复测 `4.318 s/curve`
（记录基线 §14.2 是 4.14 s/curve，同一台机器不同时刻的 4% 抖动）。

> 教训记一笔：中间有一次单独测到 42.45 s（= +28%），差点当成回归；把旧实现链进同一个进程交替测
> 才发现那是机器状态（刚跑完长时间构建/测试后的频率与功耗状态）造成的。**跨进程、跨时间点的
> 单次测量不能用来判定几个百分点的回归。**

### 17.8 复现命令

```powershell
# 单元验证（24 项）
tools\build_tool.bat tools\test\mont_ckpt_verify.cpp src\cpu\ecm_mont_cpu.cpp `
    src\cpu\simd_mont_curve.cpp src\cpu\simd_mont_ifma.cpp src\core\ecm_mont_ckpt.cpp `
    src\core\ecm_stage1_exp.cpp
.\build_vs18\tools\mont_ckpt_verify.exe 5000 512

# 端到端（杀进程 + 续跑，约 1 分钟）
powershell -NoProfile -File tools\test\test_mont_checkpoint.ps1

# 检查点开销 A/B（先把 git HEAD 的旧 ladder 抽出来并改名，再链进同一个二进制）
powershell -NoProfile -File tools\test\make_old_ladder.ps1
tools\build_tool.bat tools\bench\mont_ckpt_ab.cpp src\cpu\simd_mont_curve.cpp `
    src\cpu\simd_mont_ifma.cpp src\cpu\ecm_mont_cpu.cpp src\core\ecm_stage1_exp.cpp `
    .bench_tmp\ab_old\old_ladder.cpp
.\build_vs18\tools\mont_ckpt_ab.exe 3001 1e6 3

# 手工：1 秒自动保存 + 中途 Ctrl+C，然后重跑同一命令行
echo (2^3001-1) | .\ecm.exe --method mont --tmp-dir saves --ckpt 1 -gpucurves 64 1e6
```

## 18. 启动阶段的 29 秒：定位与修复（2026-09-24，用户实测报障）

### 18.1 症状

用户的队列任务（`ECM2=1,2,3001,-1,10000000,0,8`，B1=1e7）日志：

```
[13:59:22] START: ECM2=1,2,3001,-1,10000000,0,8
[13:59:51] method          : montgomery (Suyama sigma, AVX512-IFMA 8-lane batch, torsion=1)
```

⇒ **29 秒**卡在"开始算曲线"之前，而 GPU/CUDA 路径跑 B1=1.1e8 只要 5 秒。用户问：是不是
stage-1 指数（`lcm(1..B1)`）算得太慢？实现是不是不一样？

**答案：是，而且确实不一样——两条路径建指数的方法不同。**

### 18.2 根因：`mont_build_s` 是逐个素数的累加乘法

旧实现（`src/cpu/ecm_mont_cpu.cpp`）：

```cpp
for (p = 2; p <= B1; ++p) { ...; mpz_mul_ui(s, s, p^e); }   // s 一直在变长
```

第 i 次乘法的代价是 O(len(s))，而 len(s) 随 i 线性增长 ⇒ 总代价 O(π(B1) × len(s)) —— **素数个数
× 结果长度**：

| B1 | π(B1) | len(s) | 量级估算 |
|---|---|---|---|
| 1e7 | 620,000 | ~225k limb | 620k × 112k ≈ **7e10 limb 操作** ⇒ 实测 **~29 s** ✓ |

而 GPU 路径与 Edwards 路径用的是**乘积树**（`ecm_driver.cpp` 里的 `compute_batch_s()`：二进制计数
器，把素数幂两两合并），每次 `mpz_mul` 的两个操作数长度相当，GMP 的 Karatsuba/Toom/FFT 全部生效
⇒ 复杂度降到 O(M(n)·log n)。

> 所以"CUDAC 110e6 只要 5 秒"并不代表 GPU 有特殊算法：**那 5 秒本身也主要是这个乘积树**（见 18.5
> 实测：B1=1.1e8 需要 ~5.2 s，其中筛法 0.4 s、其余是 GMP 的大数乘法）。两条路径的差距不是 GPU 快，
> 而是 CPU Montgomery 那条路当时**没有**用乘积树。

### 18.3 修复：三条路径共用一份实现

新增 `src/core/ecm_stage1_exp.{h,cpp}`：

```c
bool ecm_build_lcm_exponent(mpz_t s, uint64_t B1, uint64_t torsion);
```

* 奇数/素数筛（从 `p*p` 开始标记，`char` 数组）；
* 每个素数幂 `p^floor(log_p B1)` 用 `set_u64()` 装进 mpz（**注意 Windows 上 `mpz_set_ui` 只吃
  32 位 `unsigned long`**，B1 可能超过 2^32——这是 `mont_set_sigma` 记录过的同一个坑）；
* **二进制计数器**合并：槽 j 保存 2^j 个素数幂的乘积，满了就带着乘积往上进位；最后从最大槽往下乘
  进 `s`（torsion 先放进去）；
* 只有 B1 > 5e9 或内存不足才返回 `false`，此时 `s` 保持 `torsion`（绝不留下错的值）。

调用方：
* `mont_build_s()`（`src/cpu/ecm_mont_cpu.cpp`）→ 薄封装，失败返回 **0**（调用方必须检查）；
* `compute_batch_s()`（`src/core/ecm_driver.cpp`，GPU 与 Edwards 共用）→ 薄封装，保留原有的
  B1 范围/取整保护；
* 于是三个方法（GPU 批量、CPU Edwards、CPU Montgomery）**只有一份指数构造实现**。

### 18.4 正确性验证（`tools/test/stage1_exp_check.cpp`）

| 项 | 方法 | 结果 |
|---|---|---|
| 与旧实现逐位相同 | 把旧的二次循环原样搬进测试当参照，扫 12 个 B1（含 0/1/2/3/4 边界）× torsion ∈ {1,12} | PASS |
| 与实现无关的判据 | 对每个素数 p ≤ 20000：`v_p(s) == floor(log_p B1) + v_p(torsion)` | PASS |
| 除尽后无残余 | 把 s 里所有素数幂除干净，余数必须正好是 1 | PASS |
| **换算法的交叉验证** | `lcm(1..B1) = Π_j primorial(⌊B1^(1/j)⌋)`，用 GMP 自己的 `mpz_primorial_ui` 重算（5 个 B1） | PASS |
| 端到端不回归 | `mont_simd_verify`（8 lane 与标量逐 lane 对拍，M1277/B1=1e5） | PASS 0/8 不一致 |
| 检查点不回归 | `mont_ckpt_verify` 24 项 + `test_mont_checkpoint.ps1` 三组 | 全绿 |

### 18.5 实测（Ryzen AI 9 HX 370，单线程）

```
B1 = 1000000       0.016 s   s_bits = 1442099
B1 = 10000000      0.231 s   s_bits = 14424844      ← 旧实现 ~29 s，用户报的那个 30 秒
B1 = 110000000     5.257 s   s_bits = 158705536     ← 与 GPU 路径同一份实现，同一个数量级
mont_expand_bits(1e7)   0.021 s   14424844 bits     ← 位数组展开，可忽略
B1 = 1.1e8: 筛法 0.414 s（6,303,309 个素数，104 MB），其余 ~4.8 s = 乘积树（GMP FFT）
```

⇒ 用户场景（B1=1e7）：**29 s → 0.23 s**；端到端复测，`START` 到 `stage1 exponent` 行约 0.23 s，
到第一条进度条也在 1 s 以内。

> 为什么不换成 primorial 链：同一台机器上 `primorial` 链只快 ~8%（4.23 s vs 4.59 s @1.1e8），
> 而当前实现既能直接处理素数幂、又少一处整数开方代码；B1 ≥ 1e7 时真正的下界是"把 1.6e8 bit 的
> 乘积算出来"这一件事本身的 O(M(n)·log n)，没有便宜的捷径。（primorial 恒等式仍然很有用——它被
> 留作 18.4 里的独立交叉验证。）

### 18.6 复现

```powershell
tools\build_tool.bat tools\test\stage1_exp_check.cpp src\core\ecm_stage1_exp.cpp src\cpu\ecm_mont_cpu.cpp
.\build_vs18\tools\stage1_exp_check.exe

# 端到端（队列模式，B1=1e7 的那条任务）
echo (2^3001-1) | .\ecm.exe --method mont --tmp-dir saves -gpucurves 8 1e7
```

## 19. 可行性：把 Suyama param0 移植到 CUDA/CGBN（2026-09-24，用户提问）

### 19.1 结论

| 维度 | 判定 |
|---|---|
| 技术可行性 | **可行，且改动很小**：host 侧换曲线/起点设置（~150 行 GMP），device 侧两处算子替换（~20 行）。**不需要新算法、不需要 GPU 端求逆**（所有除法都在 host 上做）。 |
| 性能 | **值得**：同一台机器上，现有 CGBN kernel（param3）在 3000–3200 bit、B1=1e5、8192 条曲线时达 **7.35M（Mersenne）/ 12.1M（随机 N）curve-bits/s**；CPU param0 全 24 线程是 **2.69M / 1.56M** ⇒ 移植后（算子 +22%，见 19.4）预计仍为 **2.2×（Mersenne）～6.4×（随机 N）**。 |
| 前提 | GPU 必须"上千条曲线同时在飞"：256 条只有 3.8M，8192 条 15.6M。像 `ECM2=1,2,3001,-1,10000000,0,8`（8 条曲线）这种任务在 GPU 上等于空转，应留在 CPU。 |
| 是否非做不可 | **不一定**。现有 param3 GPU 路径的存档**已经**能被 gmp-ecm（`-param 3` 是 gmp-ecm 自己的 batch 参数化）和 Prime95（`sigma_type=3`，`choose_gmp_ecm_param3()` 注释即 "A = 4*s/2^32-2, x0 = 2"）消费。移植买到的是**曲线族统一**（CPU/GPU 曲线可互换、可与 gmp-ecm param0 逐点对拍、一条 save/resume 通路），不是"从不能用到能用"。见 19.7 的决策问题。 |

### 19.2 现状：GPU 路径的目标形状（源码事实）

`set_p_2p()`（`kernels/cuda/cgbn_stage1.cu:254`，OpenCL 侧同形 `src/opencl_ecm_stage1.cpp:367`）：

```
每曲线 5 个字：N, P_a(x,z), P_b(x,z)
P_a = (2 : 1)  固定！        P_b = 2P = (9 : 64·d + 8),  d = σ/2^32
```

上游 gmp-ecm `batch.c:167` 把前提写死在注释里：`assume (x2:z2) - (x1:z1) = (2:1)` —— **阶梯对的差分点 x 坐标恒为 2**。kernel 里两处"便宜"全部来自这个前提：

1. 加法步（`cgbn_stage1_kernel.h:252-263`）：`bX = (DA+CB)²`、`bZ = 2(DA−CB)²`
   —— `z_D=1`、`x_D=2` 直接内联，**不需要乘差分坐标**；
2. 加倍步（同文件 `:219-238`）：曲线常数用 `special_mult_ui32(K, d)`，**d 就是 a24 且只有 32 位**。

指数 `s = lcm(1..B1)`、32 位字数组、`s_bits_start/interval` 分片推进 —— 与 param0 的 τ=1 完全一致，**这部分不用改**。

### 19.3 移植要改什么

Suyama param0 的定义（与本仓库 CPU 路径、gmp-ecm 完全一致）：

```
u = σ²−5,  v = 4σ,  A = (v−u)³(3u+v)/(4u³v) − 2,  a24 = (A+2)/4
起点 P = (u³ : v³)         差分点的仿射 x： xdiff = u³/v³
```

* **Host（GMP，~150 行）**：`set_p_2p` → Suyama 版：算 u,v,A,a24,X0,Z0,xdiff，再用一次 xDBL 得 2P；
  每曲线 **7 个字**（多 a24 与 xdiff）。σ 可 64 位（数组按 32 位字存）。三次 `mpz_invert` 都在 host。
* **Device（kernel，~20 行）**，替换 `double_add_v2()`（`cgbn_stage1_kernel.h:162`）里的两处：
  1. `special_mult_ui32(dK, d, …)`（:226）→ `cgbn_mont_mul(dK, K, a24, …)`
  2. `cgbn_shift_left(v, v, 1)`（:261）→ `cgbn_mont_mul(v, (DA−CB)², xdiff, …)`
  3. 从数据数组多载入两个常量。
* **算子账**：现 kernel 4M+4S+便宜 special_mult → 移植后 **6M+4S**，与 CPU 侧 param0 的算子数**完全相同**
  （param3 省下的正是这两项）。CGBN 没有专用平方：`impl_cuda.cu` 里
  `mont_sqr(r,a,n,np0) { …mont_mul(r._limbs, a._limbs, a._limbs, …); }` ⇒ 每个 S 就是一次满宽乘。
  以 mul 等价计：**8.1 → 10.0 ⇒ 单 bit 代价 +22%**。
* **顺带的好处**：kernel 变成与 σ 无关（只吃 a24/xdiff），σ→曲线的语义完全回到 host，
  和 CPU 路径、gmp-ecm 的定义对齐。

### 19.4 实测基线（RTX 4070 Ti / 60 SM，`ecm_cuda` 全量 kernel 构建 sm_89；CPU = Ryzen AI 9 HX 370 24 逻辑核）

单位用 **curve-bits/s**（= 曲线数 × s_bits ÷ 墙钟），因为 GPU 需要上千条曲线、CPU 不需要，用"s/curve"没法比。
B1=1e5 ⇒ s_bits = 144344。为防"命中即提前退出"污染计时，随机 N 用 3000 位半素数、Mersenne 侧用**素数** M3217。

**GPU（CGBN，param3 = 现状）**

| N | kernel 档 | 曲线数 | 墙钟 | curve-bits/s |
|---|---|---|---|---|
| 2999-bit 半素数 | CGBN<16,3072> | 256 | 9.68 s | 3.82 M |
| 〃 | 〃 | 1024 | 16.5 s | 8.94 M |
| 〃 | 〃 | 4096 | 40.7 s | **14.52 M** |
| 〃 | 〃 | 8192 | 75.9 s | **15.58 M** |
| 〃 | 〃 | 16384 | 156 s | 15.16 M（饱和）|
| 3999-bit 半素数 | CGBN<16,4096> | 8192 | 115 s | 10.29 M |
| 3071-bit 半素数 | CGBN<16,3584> | 8192 | 97 s | 12.15 M |
| 3215-bit 半素数 | CGBN<16,3584> | 8192 | 97 s | 12.13 M |
| **M3217 = 2^3217−1** | CGBN<16,3584> | 2048 | 80.0 s | 3.69 M |
| 〃 | 〃 | 4096 | 141 s | 4.19 M |
| 〃 | 〃 | 8192 | 161 s | **7.35 M** |

**CPU（本仓库 param0，8-lane AVX512-IFMA）**

| N | 归约域 | 线程 | 曲线数 | 墙钟 | curve-bits/s |
|---|---|---|---|---|---|
| 2999-bit 半素数 | Montgomery CIOS | 1 | 32 | 27.1 s | 0.17 M |
| 〃 | CIOS | 8 | 64 | 10.8 s | 0.85 M |
| 〃 | CIOS | 24 | 192 | 17.8 s | **1.56 M** |
| M3217 | Mersenne 折叠 | 1 | 8 | 3.73 s | 0.31 M |
| 〃 | 折叠 | 24 | 192 | 10.3 s | **2.69 M** |

**两个不显然的发现（都是实测，不是推断）**

1. ~~**CGBN 在 `N = 2^k−1` 上慢 1.65×**~~ **—— 2026-09-24 复测推翻**：
   干净复测（同一构建、同容器 3584、同曲线数、前后相邻运行）显示 Mersenne 与随机 N 吞吐**基本相同**：

   | N（3584 容器，param3） | 4096 曲线 | 8192 曲线 |
   |---|---|---|
   | M3217 = 2^3217−1 | 11.26 M | 12.38 M |
   | 3215-bit 随机半素数 | 10.99 M | 12.57 M |

   3072 容器同样一致（M3001 14.31 M vs 3001-bit 半素数 14.32 M）。
   原先那组"M3217 只有 3.69/4.19/7.35 M"的数据是在**后台正在跑全量 CUDA 编译**时测的，
   CPU 侧主机循环（launch/回读/批调度）被拖慢，GPU 于是饿着跑 ⇒ 数字不可用。
   **教训（本项目第二次踩同一个坑，见 §17.7）**：GPU/H2D 混合负载的计时必须在机器空闲时做，
   且对照点要前后相邻、同参数。
2. **GPU 必须喂上千条曲线**，而且寄存器压力决定上限：kernel 每线程 88（3072 容器）/98（3584）/105（4096）个寄存器，
   每 block 512–640 线程 ⇒ 占用率只有一半左右。256 条曲线时只有 3.8M（60 个 SM 绝大多数闲着）。


### 19.5 移植后预计性能（实测 × 算子比 0.82）

**2026-09-24 已在真机上实测到 param0 kernel 本体**（全量构建，RTX 4070 Ti，B1=1e5，4096 曲线）：

| 场景 | param3（旧族） | **param0（新族，实测）** | param0/param3 |
|---|---|---|---|
| M3001（3072 容器）| 14.31 M | **11.71 M** | 0.82× |
| 3001-bit 随机半素数 | 14.32 M | **11.57 M** | 0.81× |
| M3217（3584 容器）| 11.38 M | **8.85 M** | 0.78× |

⇒ 算子比预测的 0.82 与实测 0.78–0.82 吻合（CGBN 无专用平方，多出的两次满宽乘就是全部代价）。

**同一口径下的命中率实测**（`tools/ecm_prob/out/measure_<bit>_256.json`，B1=256，bit 15–40 全覆盖，
每点 65536 个采样素数；bit ≤ 20 为穷举）：param0 与 param3 的单曲线成功率比值随位宽缓慢上升 ——

| bit | 15 | 20 | 25 | 30 | 33 | 35 | 36 | 38 | 40 |
|---|---|---|---|---|---|---|---|---|---|
| param0 / param3 | 1.30× | 1.41× | 1.49× | 1.62× | 1.53× | 1.58× | 1.72× | 2.12× | 1.52× |
| param0 绝对值 | 68.80% | 30.47% | 10.26% | 2.841% | 1.169% | 0.568% | 0.444% | 0.278% | 0.102% |

bit ≥ 35 时每点命中数只剩几十个（bit40：67/65536），±0.1–0.3 个百分点是采样噪声，**别把单个点的
比值当趋势读**；能读出的只有"比值在 1.3–1.8 之间、且比 D 的 3× 小得多"。

与 CPU（同一台机器，干净复测，24 线程 = 全机箱）对比：

| N | GPU param0（4096 曲线） | CPU param0（24 线程） | CPU param0（1 线程） | GPU / 整机 | GPU / 单核 |
|---|---|---|---|---|---|
| M3001 | 11.71 M | 3.09 M（折叠域） | ~0.33 M | **3.8×** | ~35× |
| M3217 | 8.85 M | 2.72 M（折叠域） | 0.31 M | **3.3×** | ~29× |
| 3001-bit 随机 N | 11.57 M | 1.58 M（CIOS） | ~0.17 M | **7.3×** | ~68× |

⇒ 换成"每因子期望代价"：
* **GPU param0 vs CPU param0**（同曲线族，成功率项相消）＝ 吞吐比 ＝ 整机 **3.3–7.3×**，单核 ~29–68×；
* **GPU param0 vs GPU param3** ＝ `(1/0.81) ÷ (1.3–1.8)` ≈ **0.7–0.9×**（省 10–30%）。
  （早期版本这里写的是 0.27× / 1/11–1/22，那是把"**D 的 3×**"错当成"**成功率的 3×**"叠进去的结果；
  按实测 1.3–1.8× 修正后就是这个量级。param0 相对 param3 的确定优势更多在于
  **存档能直接被 gmp-ecm `-param 0` / Prime95 续做 stage 2**、以及与 CPU 路径同曲线同 σ。）

即：**移植（CPU param0 → GPU param0）的收益 ≈ 整个 24 线程 CPU 机箱的 3.3–7.3 倍**，
且 B1 越大、曲线越多收益越稳（每 bit 成本两边都固定，GPU 的分片启动开销被摊薄）。

### 19.6 移植的风险 / 待测量项（按优先级）

1. **两个常驻满宽常数（a24、xdiff）会吃寄存器**：现在 kernel 只有一个 32 位标量 `d`；
   改成两个满宽常数 ⇒ 每线程 +约 14 个寄存器（7 limb × 2）⇒ 可能把 512 线程/block 压到 448。
   缓解：把两者放 **shared memory**（CGBN 自己 `SHM_LIMIT=0`，但我们可用自留区；每实例每 bit 只读 14 次
   shared load，相对每 bit 数百周期可忽略）。**移植时必须实测这一项**，它可能把 19.5 的 0.82 系数再拉低一点。
2. **容器档位跳变**：3001 bit 落在 3072（+2% 填充），3217 落在 3584（+11%）；tier 之间是 n² 关系
   （实测 3072→3584：15.58M → 12.15M，与 (3584/3072)² = 1.36 吻合）。选任务尺寸时值得看一眼档位。
3. **驱动/命名**：现在 `method` 是互斥三选一（`gpu|edwards|mont`），"mont on CUDA" 需要新的表达方式
   （见 19.7）。
4. **OpenCL/AMD 路径不会自动获得 param0**：.cl kernel 是另一份实现，要同步才有同样语义（iGPU 实测只有
   ~0.2M curve-bits/s 量级 ⇒ 不建议为它做）。
5. **checkpoint / 存档**：结构不变，但要动格式版本 —— GPU 的 ckpt 存整块"5 字/曲线 + s_partial"，
   param0 需要 7 字（或只存 (σ, 位偏移) 在 host 侧重算 a24/xdiff，更省）；存档改为
   `ecm_append_save_lines_mont()`（省略 `PARAM=`，即 param0 文本形态）。

### 19.7 需要你决策的那一个问题

移植买到的是"**曲线族统一**"，而不是"从不能跑到能跑"：

* 现有 param3 存档**已经**能交 Prime95 做 stage 2（`sigma_type=3`）和 gmp-ecm（`-param 3`）；
* 但它与 gmp-ecm `-param 0`（本项目 CPU 路径与验收口径 Q1）**不是同一条曲线族**。

⇒ 决策点是：**你的 GPU 任务是否必须是 param0 曲线族**（要和 CPU 结果逐点互校、要让同一条 worktodo/存档在
CPU 与 GPU 之间互换、要交 gmp-ecm 用 `-param 0` 续跑），还是 **param3 曲线族对 GPU 任务可以接受**？

* 若"必须 param0" ⇒ 建议做，工作量约 1-2 天（host 设置 + kernel 两处 + ckpt/存档接线 + 一条对拍验证）。
* 若"param3 可以" ⇒ **不必移植**：直接用 `ecm_cuda` 跑 GPU 任务，stage 2 交给 Prime95（`param=3`）。

### 19.8 复现命令

```powershell
# 基准数（半素数：stage 1 不会命中，计时才有效；Mersenne 侧用素数 M3217）
tools\build_tool.bat tools\bench\gen_semiprime.cpp
.\build_vs18\tools\gen_semiprime.exe 3001 1 > n3001.txt

# GPU（现成 CUDA 全量构建，NMake 树，见 README「为何单独构建」）
.\build_cuda_cmake\ecm_cuda.exe -v -gpu -d 0 -sigma 3:12345678 -gpucurves 8192 1e5 0 < n3001.txt

# CPU（同 B1；折叠域只在 N = 2^k-1 时启用，随机 N 走 CIOS，两者相差 1.8×）
.\build_vs18\Release\ecm.exe --method mont --tmp-dir . -gpucurves 192 --stage1-threads 24 1e5 0 < n3217.txt
```

## 20. 实施清单：param0 on CUDA/CGBN（2026-09-24 决策后，逐处改动）

决策（用户 2026-09-24）：**必须实现 param0**；选择方式 `method = gpu` + 新增 **`gpu_param = 0|3`**；
**param3 保留可选**（代码默认 3 = 保持旧行为，ini 模板写 0 并注明推荐）。

### 20.1 现状（已读源码，改动面就在这几处）

`kernels/cuda/cgbn_stage1.cu`：

| 行 | 现状 |
|---|---|
| `:254` | `set_p_2p(N, curves, sigma, BITS, &data_size)`：**5 字/曲线** = `N, aX, aZ, bX, bZ`，`P_a=(2:1)`、`P_b=(9, 64d+8)` |
| `:309` | `process_results(..., const uint32_t *data, cgbn_bits, curves, sigma)`：按 `limbs_per = BITS/32`、5 字步长读回 |
| `:340` `:1664` | ckpt 头/数据：`header{curves,sigma,BITS,TPI,data_size}` + **整块 data**；恢复时按 `data_size` 反推 strides |
| `:606` | `cgbn_ecm_stage1(factors, array_found, N, s, curves, sigma_ptr, ckpt_ms, gputime, verbose)` |
| `:937` | 批循环：`this_batch` 自适应到 ~100 ms，`(*kernel)<<<BLOCK_COUNT,TPB>>>(report, s_num_bits, s_partial, this_batch, gpu_s_bits, gpu_data, curves, sigma, np0)` |
| `:951` | kernel 签名 9 参（见下），曲线常数靠 `sigma` 标量传进 kernel |

`kernels/cuda/cgbn_stage1_kernel.h`：`:162` `double_add_v2(q,u,w,v,uint32_t d,modulus,np0)`；
`:272` `kernel_double_add<params>(report, s_bits, s_bits_start, s_bits_interval, gpu_s_bits, data, count, sigma_0, np0)`；
`:423` 每 TPI 一个 dispatch 函数；4 个 `cgbn_stage1_kernels_tpi*.cu` 负责实例化。

### 20.2 改动清单

**(1) kernel：新增 param0 变体（不动原函数）**

`cgbn_stage1_kernel.h` 里加 `double_add_v2_suyama(q,u,w,v, const bn_t &a24, const bn_t &xdiff, modulus, np0)`：
与 `double_add_v2` 只差两处（其余逐行照抄，保证与 param3 版本同样的调度/归一化模式）：

```
- special_mult_ui32(dK, d, modulus, np0);            // d 是 32 位 a24
+ cgbn_mont_mul(_env, dK, K, a24, modulus, np0);     // 满宽 a24（Montgomery 域）
...
- cgbn_shift_left(_env, v, v, 1);                    // ×2 = 差分坐标 x_D=2
+ cgbn_mont_mul(_env, v, v, xdiff, modulus, np0);    // ×xdiff（差分点仿射 x）
```

新增 `kernel_double_add_suyama<params>`：与 `kernel_double_add` 同结构，但 Setup 段多载入两个常量：

```
data 布局（param0，7 字/曲线）：N, a24, xdiff, aX, aZ, bX, bZ
  a24, xdiff 一次性载入并 bn2mont（与现有 aX/aZ/bX/bZ 同样处理）
  循环里调用 double_add_v2_suyama(...)
  （首版把两常数放寄存器；若 regs 涨到掉占用率，改放自留 shared memory —— 见 20.4）
```

**为什么不用 `special_mult_ui32`**：它只能乘 32 位常数（`(K·R·σ)>>32`），而 Suyama 的 a24 是满宽值。
**为什么 `cgbn_mont_mul` 是对的**：`a24` 在 Setup 里已 `bn2mont`，与被乘量同域 ✓。

**(2) dispatch：只实例化需要的档位（编译时间是稀缺资源）**

新增 `kernels/cuda/cgbn_stage1_kernels_suyama_tpi16.cu`（先只做 TPI=16 的 3072/3584/4096 三档，
覆盖 M3001/M3217/4001 目标尺寸），导出
`cgbn_stage1_kernel_suyama_tpi16(BITS, *TPI_out)`；`cgbn_stage1.cu` 里加
`cgbn_stage1_kernel_suyama_dispatch(BITS, TPI_out)`（先查 TPI16 表，再按需加其它 TPI）。
CMake：把新 TU 加进 `ecm_cuda` 的源列表（与 4 个 tpi*.cu 并列）。

**(3) host：Suyama 曲线/起点设置**

`cgbn_stage1.cu` 新增 `set_p_2p_suyama(const mpz_t N, uint32_t curves, uint64_t sigma0, uint32_t BITS, size_t *data_size, const char *field)`：

```
每曲线 i：σ_i = sigma0 + i（uint64，与 CPU 路径/batch 语义一致）
  u = σ²−5, v = 4σ
  A  = (v−u)³(3u+v)/(4u³v) − 2      (mod N)
  a24 = (A+2)/4                      (mod N)      → 写入 datum[1]
  X0 = u³, Z0 = v³                   (mod N)
  xdiff = X0·Z0⁻¹                    (mod N)      → 写入 datum[2]
  2P ← 一次 xDBL(a24)  → datum[5], datum[6]
  N → datum[0]; X0,Z0 → datum[3], datum[4]
```

要点：
* **所有除法都在 host**（`mpz_invert`），GPU 端不需要求逆；
* σ 用 `mont_set_sigma()` 同款的 64 位装配（Windows 上 `mpz_set_ui` 只吃 32 位）；
* 与 `src/cpu/ecm_mont_cpu.cpp:mont_suyama_curve()` 用**同一套公式**（可直接复用/复制该函数以保证逐位一致）；
* `process_results` 的步长改为「7 字 + 由 `data_size` 反推」而不是硬编码 5。

**(4) 驱动 / 配置：`gpu_param = 0|3`**

* `src/core/ecm_queue_config.{h,cpp}`：新增 `int gpu_param = 3;`（`[gpu]` 组，取值 0|3；非法值警告并回落 3），
  ini 模板双语注释：`0 = Suyama param0（Z/12，成功率≈3×，推荐）/ 3 = gmp-ecm batch param3（现状）`。
* `src/core/ecm_driver.cpp`：CLI `--gpu-param 0|3`；经 `Stage1RunOptions` 传给后端 seam。
* `include/ecm_backend.h` + `src/cuda/ecm_cuda_backend.cu` + `src/opencl_backend_glue.cpp`：
  seam 增加一个 `gpu_param` 入参；CUDA 侧转给 `cgbn_ecm_stage1(..., param)`；
  **OpenCL 侧先只接受 3，收到 0 时明确报错**（.cl kernel 未移植，见 20.5）。

**(5) 存档**

* `gpu_param = 0` ⇒ 走 `ecm_append_save_lines_mont()`（`src/core/ecm_save.cpp:219`，param0 文本，**省略 PARAM=**）
  ⇒ gmp-ecm `-param 0` / Prime95 `sigma_type=1` 都能续 stage 2；
* `gpu_param = 3` ⇒ 保持现有 `opencl_ecm_append_save_lines()`（写 `PARAM=3`，Prime95 `sigma_type=3`）。

**(6) 检查点**

* 头里加 `param` 字段 + 版本号 +1；`data_size = 7 × curves × BITS/32 × 4`。
* 恢复时校验 `param` 一致，否则拒绝（不同参数化的缓冲区不可互读）。
* σ 与位偏移的语义与 CPU 侧一致（σ 决定曲线与起点，`s_partial` 决定阶梯位置）⇒ 续跑行为与 CPU 路径同构。

### 20.3 验收（必须全部通过才算完成）

1. **逐点对拍（前哨，dev/小 kernel 即可迭代）**：同一组 σ、同一 B1，
   `gpu_param = 0` 的 CUDA 结果与 `--method mont`（CPU 标量 `mont_stage1_curve_bits_x`）**逐曲线 x 相同、hits 相同**。
   先用 M991（1024 容器，dev build 编译快），再 M1277/M3001。
2. **吞吐**：M3001 / M3217 上 `gpu_param = 0` vs `gpu_param = 3` vs CPU 24 线程，报 curve-bits/s
   （预期：param0 比 param3 慢 ~22%，但每因子期望代价 ≈0.41×；CPU 侧见 §19.4 的表）。
3. **存档互通**：`gpu_param = 0` 的 `.save` 交给 gmp-ecm `-param 0` 续 stage 2（用已有的
   `tools/test/test_mont_gmp_oracle.ps1` 同款做法）；并与 CPU param0 的 `.save` **内容一致**（同 σ 集合）。
4. **续跑**：`gpu_param = 0` 跑到一半杀进程 → 重跑同一命令行 → 曲线内容与不中断那次一致（复用
   `tools/test/test_mont_checkpoint.ps1` 的判据）。

### 20.4 已知风险与先测项

1. **寄存器**：kernel 现在每线程 88/98/105 regs（3072/3584/4096 容器），常驻两个满宽常数额外 +14。
   首版放寄存器并实测 `numRegsPerThread`；若 block 占用率下降（512→448）则改放 shared memory。
   **这是移植里唯一可能需要调结构的地方，建议第一步就量化。**
2. **编译时间**：全量 TPI×BITS 实例化很贵（`cgbn_stage1.cu:664-675` 的注释记录过成本）
   ⇒ param0 变体只实例化需要的档位，按需扩表。
3. **容器档位**：3001→3072、3217→3584，tier 之间是 n² 关系（§19.4）⇒ 选任务尺寸时看一眼档位。

### 20.5 明确不做的部分

* **OpenCL (.cl) 路径的 param0**：AMD iGPU 实测 ~0.2M curve-bits/s 量级，投入产出不成立；
  `gpu_param = 0` 在 OpenCL 后端下明确报错而不是静默降级。
* param3 路径的任何语义改动（用户要求保留可选）。

### 20.6 进展（2026-09-24：**四项验收全部通过**）

GPU 侧实现已落地，`method = gpu` + `gpu_param = 0` 可用（param3 保持不变）：

| 改动 | 位置 |
|---|---|
| param0 阶梯变体 | `kernels/cuda/cgbn_stage1_kernel.h`：`double_add_v2_suyama()` + `kernel_double_add_suyama<params>`（7 字/曲线：`N, a24, xdiff, aX, aZ, bX, bZ`）|
| 分发 | 新 TU `kernels/cuda/cgbn_stage1_kernels_suyama.cu`：tpi8 的 768/1024（始终编译，供 dev build 做正确性迭代）+ tpi16 的 3072/3584/4096（仅 full build）|
| host 设置 | `kernels/cuda/cgbn_stage1.cu`：`set_p_2p_suyama()`（u,v,A,a24,X0,Z0,xdiff,2P；三次求逆全在 host）；`process_results()` 改为按 `words_per_curve`/`p1_word`/`p2_word` 读取，不再硬编码 5 字 |
| 64 位 σ | `cgbn_ecm_stage1(..., uint64_t *sigma, ..., int gpu_param)`；驱动 `firstsigma64`（param0 用同一个 53 位随机生成器）贯通到 seam 的两个后端；param3 仍要求 σ+curves ≤ 2^32（那条路把 d=σ/2^32 当 32 位核参数） |
| 选择器 | ini `gpu_param`（`[gpu]` 组）+ CLI `--gpu-param 0|3`；代码默认 3（旧 ini 行为不变），模板写 0 并注明推荐 |
| 存档 | param0 走 `ecm_append_save_lines_mont()`（省略 `PARAM=`）且**必须携带原始 N**；param3 保持 `PARAM=3` + 原有的"N 除以已找到因子"写法 |
| 检查点 | **格式升到 v4**（两条路径各自的结构体都改了）：`sigma` 变成 **64 位**、新增 `gpu_param` 字段；仍用 `data_size` 反推"每曲线几个字"（5 = param3、7 = param0）双向校验，参数化或布局不符直接拒绝重来。旧 v3 检查点按设计作废（头布局变了）|

**验收结果**

| # | 判据 | 结果 |
|---|---|---|
| 1 | GPU param0 与 CPU param0 逐曲线一致 | **PASS**：M991 64/64、991-bit 半素数 64/64、64 位 σ 32/32（含 hit 的因子值；8218291649 与 `docs/ECM_EDWARDS_STAGE1.md` 的 M991 不变量一致）|
| 2 | 吞吐（M3001/M3217，B1=1e5，4096 曲线）| **PASS**：param0/param3 = 0.78–0.82（与 +22% 算子预测吻合）；vs CPU 24 线程 = 3.3–7.3×（§19.5 表）|
| 3 | 存档互通 | **PASS**：`ecm-zen3.exe -resume <gpu param0 save> 1e5 1e6` 接受全部 32 行（无 bad checksum）、读到 `N=(2^991-1)`、`sigma=0:777`，并在 **stage 2 找到因子 41473350001** |
| 4 | 杀进程续跑 | **PASS**：M3217 / 4096 曲线跑到 25 s 硬杀（留下 12.85 MB = 4096×7×(3584/32)×4 B 的 ckpt）→ 重跑同一命令行 ⇒ 4096/4096 行与不中断那次完全一致，成功后 ckpt 自动删除 |

回归：param3 未受影响 —— `ecm_cuda --gpu-param 3` 与 OpenCL `ecm --gpu-param 3` 在同一 N/σ/B1 下 **32/32 行一致**；
OpenCL 后端收到 `--gpu-param 0` 会**明确报错**（不静默降级）。整套判据固化为
`tools/test/test_cuda_param0.ps1`（A 对拍 / B 64 位 σ / C gmp-ecm 互通 / D 杀进程续跑 / E param3 回归，
共 17 项检查，`-SkipSlow` 可跳过 D），2026-09-24 全绿：

```
=== A. GPU param0 vs CPU param0 (M991, B1=1e5, 64 curves) ===
  [PASS] GPU reports the Suyama param0 parametrization
  [PASS] both saves have 64 curve lines (GPU 64, CPU 64)
  [PASS] GPU and CPU agree on all 64 curves (sigma AND x)
=== B. 64-bit sigma (> 2^32) ===
  [PASS] GPU reports the full 64-bit sigma (9007199254740881), not a truncated one
  [PASS] GPU and CPU agree on all 32 curves at sigma > 2^32
  [PASS] the save carries the untruncated sigma
=== C. gmp-ecm accepts the param0 save (resume + stage 2) ===
  [PASS] no "bad checksum" complaint (the N field is the real N)
  [PASS] gmp-ecm read N = (2^991-1)
  [PASS] gmp-ecm read the line as param 0 (Suyama) with our sigma
=== D. hard kill + resume (M3217, 4096 curves, B1=1e5) ===
  [PASS] reference run completed
  [PASS] the run was still going after 25 s (so the kill is a real mid-run kill)
  [PASS] the killed run left a checkpoint behind
  [PASS] the checkpoint holds the whole 7-word/curve buffer (12845120 bytes)
  [PASS] the resumed run completed
  [PASS] killed+resumed save is identical to the uninterrupted one (4096/4096 lines)
  [PASS] the checkpoint is removed once the save is written
=== E. param3 regression: CUDA vs OpenCL ===
  [PASS] param3 results unchanged across the two backends (32/32 lines)
  [PASS] the OpenCL backend refuses gpu_param = 0 loudly
ALL OK
```

> 迭代速度提示：`build_cuda_dev`（NMake，dev kernel 集）只带 768/1024 的 param0 kernel，
> 编译快，适合做 M991 这种小 N 的正确性迭代；吞吐/大 N 必须用全量的 `build_cuda_cmake`。


**过程中发现并修掉的两个真问题**

1. **param0 存档的 `N=` 不能沿用 GPU 老路径的写法**：`opencl_ecm_build_saved_n_expr()` 会把 N 改写成
   "N ÷ 已找到的因子"，而 gmp-ecm 用 N 校验每行 checksum ⇒ 32 行全被判 `bad checksum`。param0 分支改用
   原始 `n_expr`（与 CPU 路径一致）后 gmp-ecm 立刻接受并成功做 stage 2。
2. **64 位 σ 在 seam 上被截断**：原 seam 是 `uint32_t *sigma`，param0 的 53 位随机 σ 会被截断（曲线变另一条）。
   现已改为 `uint64_t *sigma` 贯通两端；param3 侧补了显式的 32 位窗口校验。

**顺带（用户要求）：TPI=16 档位从 256 间隔回退到 512 间隔**
`kernels/cuda/cgbn_stage1_kernels_tpi16.cu` 之前把 2560..8192 按 **256** 间隔全部实例化（23 个），
选择列表里其实早就只留了 512 间隔（那些 256 档是死实例化），等于每次全量构建白编译 11 个模板。
现在两份列表统一为 **512 间隔**（2560/3072/3584/4096/4608/5120/5632/6144/6656/7168/7680/8192），
`available_kernels` 里的注释掉的残留行也清掉了。落在两档之间的 N（如 3300 bit）会用上一档容器
（3584，多约 8% 每乘代价）——这是这个网格有意接受的取舍（实测 256 网格没有吞吐收益）。

**已知限制（下一步可做）**

* ~~检查点的 `sigma` 字段只有 32 位~~ **已修（2026-09-24）**：两条路径的 ckpt 头都升到 **v4**
  （`sigma` 64 位 + `gpu_param` 字段，CUDA 侧与 OpenCL 侧的共享结构都是 72 字节），param0 的 64 位 σ
  现在完整落盘并在恢复时用于校验（参数化不一致会被拒绝，而不是按错的步长/错 σ 读取）。
* ~~param0 kernel 只实例化 3072/3584/4096~~ **已补全**：现在与 param3 同一张档位表
  （TPI=4 的 128–512、TPI=8 的 768–2048、TPI=16 的 2560–8192、TPI=32 的 9216–16384），
  代价是全量构建的实例化数量翻倍（见 §21）。
* 命中率工具 `tools/stat/ecm_hitrate.ps1` 已重写：可选 `-Engine edwards|mont|gpu`、`-GpuParam 0|3`、
  `-BitsFrom/-BitsTo` 位宽范围、`-Count N` 采样或 `-All` 全体素数；命中按驱动权威行 `factor[i]=` 计数。
  顺带修掉一个既有显示 bug：项目日志 shim 就是 `vfprintf`，**不支持 gmp-ecm 的 `%Zd`**
  （CUDA 的命中行过去打成 `factor d found`，因子值丢失），现在用 `mpz_get_str` 显式渲染。
* 同工具的两个**测量口径 bug**（第二轮自查发现并修复，都会让"命中率"这一列失真）：
  1. **多行输出被当成一行**：脚本把 `cmd /c ... | Out-String` 得到的**单个多行字符串**直接喂
     给 `Select-String`，而 Select-String 对一个字符串只返回**一条**匹配 ⇒ 每个素数的多次命中
     只算一次，速率被压低成 6.25%（bit20/B1=256 的真值是 30.47%，差 4.9 倍）。
     修法：先 `-split "`r?`n"` 再匹配（或 `[regex]::Matches`）。教训记在脚本头里。
  2. **`simd-auto` 不是合法后端名**：驱动器只认 `auto|simd|gmp`，脚本却把 `simd-auto` 原样
     当 `--backend` 传过去 ⇒ 每个素数都被拒（`Invalid --backend`），1000 个素数全 0 命中而
     脚本不吭声。现在脚本把 `simd-auto/simd-mers/simd-mont` 翻译成
     `--backend simd --field auto|mersenne|montgomery`，并对"有输出但没有结果行"的运行报警
     （summary 里新增 `failedRuns` 列）。
* 修好后与**独立 Python 模型**（`tools/ecm_prob`，逐曲线独立判 `Z == 0`，bit20 / B1=256）对拍：

  | 引擎 / 曲线族 | 本工具实测 | Python 模型 |
  |---|---|---|
  | CPU mont(simd) param0（64 primes × 32 curves）| 625/2048 = **30.518%** | 30.472% |
  | GPU param0（同上，同一批 σ）| 625/2048 = **30.518%**（与 CPU 逐曲线完全一致）| 30.472% |
  | GPU param3（同上）| 458/2048 = **22.363%** | 21.641% |
  | Edwards Z/2xZ/8（1000 primes × 8 curves）| 2611/8000 = **32.6375%** | 32.665% |

  ⇒ param0 的 CUDA 实现不只与 CPU 逐曲线一致，也与完全独立的 Python 模型在同一 σ 族上吻合到
  0.05 个百分点；param3 的 0.7pp 差异来自 σ 取值族不同（工具用 32 位 σ 序列，模型固定 σ=10）。
* `tools/ecm_prob/ecm_sweep.py` 的位置参数是**位宽列表**而不是区间（`sweep 31 40` = 只算 31 和 40，
  实测踩到过），现在显式支持区间写法 `sweep 31-40`（`31..40` 亦可），非法区间直接报错。
* **横幅在 CUDA 版里谎报 OpenCL（2026-09-24 用户指出并修复）**：driver 是两个 exe 共用的，但
  "gpu 实现"是**链接期**决定的，于是写死的字符串在 `ecm_cuda.exe` 里全是错的 —— 队列管理器打印
  `method : gpu (OpenCL)`，帮助首行是 `OpenCL ECM stage-1 driver`，`-d` 写成 "OpenCL device
  index"，`--showkernel` 写成 "OpenCL kernel paths"，运行行 `Using B1=... (N curves)` 也不带后端。
  修法：给 backend 接缝加 `const char *ecm_backend_name(void)`（OpenCL glue 返回 `"OpenCL"`，
  CUDA glue 返回 `"CUDA/CGBN"`，见 `include/ecm_backend.h`），driver 里所有会暴露实现名的位置
  都改成问它；队列管理器另加一行 `gpu backend : <名字>, param<N>, device <d>`，运行行改成
  `Using B1=100000, B2=0 (N curves, CUDA/CGBN)`。OpenCL 专用开关（`--mul/--sqr/--add/--sub/
  --special-mult`、内核缓存、`tpi`/`wg_size`）在帮助与 ini 注释里都显式标了 "(OpenCL only)"，
  ini 模板里 `method = gpu`、`ckpt_seconds`、`device` 也不再自称 OpenCL（跑哪个实现由 exe 决定，
  ini 不区分）。两个本地 `ecm.ini`（`build_vs18/Release`、`build_cuda_cmake`）已按新模板刷新，
  刷新前核对过 key=value 与模板默认值完全一致（没有丢用户设置）。
* 顺带修掉一个**编码**坑：`tools/stat/ecm_hitrate.ps1` 里有中文注释/字符串但没有 UTF-8 BOM，
  Windows PowerShell 5.1 于是按系统 ANSI（GBK）解码，UTF-8 的中文会把后面的引号当作 GBK 尾字节
  **吃掉**（`"（速率不可信）："` 的 `：` 吞了收尾的 `"`），报出一堆看不懂的语法错误。
  现在该脚本（连同类风险的两个测试脚本）都改成 UTF-8 **with BOM**；`pwsh` 默认按 UTF-8 读，
  所以这类问题只在用 `powershell`（5.1）跑时暴露 —— 仓库脚本是按 5.1 写的，务必带 BOM。

## 21. param0 与 param3 的 kernel 能否复用？（2026-09-24 用户提问）

**结论：不能复用同一批实例化；两者是两组独立的 `__global__` 函数。** 但"编译两份"这件事可以有不同的
取舍，下面是完整账。

### 21.1 为什么不能复用

两者的差别全部在**热循环里的每 bit 算术**：

| | param3（batch）| param0（Suyama）|
|---|---|---|
| 曲线常数乘法 | `special_mult_ui32(K, d)`：32 位常数 + 单字约简 | `cgbn_mont_mul(K, a24)`：满宽常数 |
| 差分坐标 | 内联常量 `2`（`shift_left(v,1)`）| `cgbn_mont_mul(v, xdiff)`：满宽常数 |
| 数据布局 | 5 字/曲线 | 7 字/曲线 |
| σ | 32 位（`d = σ/2^32`）| 64 位（不进入 kernel，只进 host 设置）|

这几处都是**编译期决定的操作序列**（操作数宽度、常量来源），不是运行期能"传参数解决"的东西 ⇒
在 CUDA 里它们就是两个不同的 kernel 函数，各自按 (TPI, BITS) 实例化一次。

> （顺带说明为什么这让 param0 略微变慢：CGBN 没有专用平方，`mont_sqr` 就是 `mont_mul`，
> 所以每 bit 从 8 个满宽乘变成 10 个 ⇒ 实测吞吐 0.78–0.82×，与预测的 +22% 一致。）

### 21.2 如果一定要"只编译一份"，有三条路（都不免费）

| 方案 | 编译时间 | 运行时代价 | 适用 |
|---|---|---|---|
| (a) 现状：两个 kernel | 2× | 0 | **当前选择**：两个族都要长期可用时最干净 |
| (b) 一个 kernel + 运行期 `bool param0` 参数 | **1×** | 热循环里多一次 uniform 分支；两条路径的临时变量都要活着 ⇒ 寄存器上涨（3072 档目前 88 → 可能 >100），占用率可能下降；param3 分支还要额外携带 a24/xdiff 常数 | 只有编译时间/二进制体积是硬约束时才考虑；**必须先实测寄存器与吞吐再决定** |
| (c) 模板加一个 `bool PARAM0` 参数 | 仍是 2×（模板实例化数不变），只是源码合一 | 0 | 想要"单一实现来源"（避免两份公式漂移）时用它 |
| (d) 只保留 param0，删掉 param3 | **0.5×**（相对现状）| 0 | **如果确认不再需要 batch 族**，这是最划算的减半方式：老 GPU 存档（`PARAM=3`）将无法续跑 |

**我的建议**：(a) 保持现状。原因是 param3 的实例化本来就是既有资产（编译一次可长期复用），而 (b)
把"每 bit 热循环"变成带分支的代码，风险/收益不成比例。(d) 是真正的省时间路径，但它是一个**产品决策**
（是否放弃 batch 族），不该由 kernel 结构决定——如果决定放弃，删掉 param3 的实例化即可，param0 侧的
成本完全不变。

### 21.3 已落地的档位表（2026-09-24：param0 补全到与 param3 相同）

`kernels/cuda/cgbn_stage1_kernels_suyama.cu` 现在实例化与 param3 完全相同的网格：

```
TPI=4  :  128, 192, 256, 384, 512                       （始终编译）
TPI=8  :  768, 1024                                     （始终编译）
          1280, 1536, 1792, 2048                        （仅 full build）
TPI=16 :  2560, 3072, 3584, 4096, 4608, 5120, 5632,
          6144, 6656, 7168, 7680, 8192                  （仅 full build，512 间隔）
TPI=32 :  9216, 10240, 11264, 12288, 13312, 14336,
          15360, 16384                                  （仅 full build，512 间隔）
```

⇒ `gpu_param = 0` 现在能处理与 batch 路径相同范围的所有 N（≤16384 bit），包括
`docs/ECM_Edwards...` 里那类 M8237 规模的输入（8192 容器）与 hit-rate 工具用的小 N（768 容器）。

代价：全量 CUDA 构建的模板实例化数量翻倍（param3 31 个 + param0 31 个）。开发用的小集合
（`build_cuda_dev`，`IS_DEV_BUILD`）仍然只编译 TPI=4/8 的小档，所以"改一行、编一次、在 M991 上验证"
的迭代速度不变。
#### 20.6.1 补充验证（2026-09-24 晚，ckpt v4 + 档位补全之后）

| 项 | 结果 |
|---|---|
| 检查点 v4（64 位 σ） | **PASS**：`test_cuda_param0.ps1` 用例 D 改用 σ = 9007199254740847（> 2³²）跑到 25 s 硬杀 → 续跑 ⇒ 4096/4096 行与不中断一致，且断言"存档仍带完整 64 位 σ"通过；检查点文件 12,845,128 字节 = 4096×7×112×4 + 72 字节 v4 头 ✓ |
| 新实例化的档位 | **PASS**：M1277 现在走 **CGBN&lt;8, 1536&gt;**（本次新增档）、M4001 走 CGBN&lt;16, 4096&gt;，两者与 CPU param0 各 16/16 行一致 |
| OpenCL 侧 v4 头 | **PASS**：`ecm.exe -gpu`（param3）写出的 `.dat` 头为 `magic=0x45555047 version=4`、`gpu_param=3`、`BITS=1024`、`TPI=8`、`data_size=327680`（= 512×5×32×4 ✓）；续跑打印 `Resuming from checkpoint: 4.0% complete` 并按存档 σ 恢复 |
| 全套回归 | `tools/test/test_cuda_param0.ps1` **18/18 PASS**（A 对拍 / B 64 位 σ / C gmp-ecm 互通 / D 硬杀续跑+64 位 σ / E param3 跨后端回归）|
