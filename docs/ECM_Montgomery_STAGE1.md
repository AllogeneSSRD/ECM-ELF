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
  而 GMP-ECM **param3** 只有 Z/4（D≈7.6）⇒ 成功率大约 3×。这就是本次要做 Suyama 的原因。
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
- 对比本仓库 GPU 路径的 param3（Z/4，D≈7.6）：成功率约 **3×**。

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

### 15.2 路径一：底层/访存/汇编优化 —— 只剩 5~15%，且不在访存

已实测的三条边界（§11、§13）：

| 事实 | 数值 | 含义 |
|---|---|---|
| 乘积相位 madd 利用率 | 91%（sqr）/ 96%（mul）的本机屋顶 | **madd 端口已饱和**，加 ILP 在本机无空间 |
| 工作集敏感性 | 1→16 个操作数缓冲 < 1% | **不是访存受限**，L1 常驻；"访存优化"无处下手 |
| 辅助 SoA pass | 已从 30 趟/bit 降到 18 趟（§11.3b，−8%） | 已吃掉 |
| tail（normalize/fold/drain/canonical） | 每个域运算 15~25%，**串行 carry 链**、~0.7 ops/cycle | 唯一还剩的靶子 |

⇒ 路径一**剩余空间 5~15%**，集中在两处：
1. **tail 的串行 carry 链**：把 2n 列的归一化拆成两条独立链做 ILP（约 4%），
   或把 canonical 与 drain 融合（约 3%）——两者都要动 $2^k=1$ 折叠的边界，属"高风险区"（§3 注释里的非规范值 bug）。
2. **乘积循环里的非 madd 指令**（每 4 条 madd 摊 ~7 条 load/store/add）：
   最有效的手段不是汇编重排，而是**加宽批**——把 8 lane 提到 **16 lane（每个元素 2 条 zmm）**，
   因为 load/store 是"每向量一次"的固定开销，批宽翻倍即把这部分**按比例摊薄**。
   估算：非 madd 部分约占乘积相位的 25~30%，加宽批可削掉一半 ⇒ **−10~15%**，代价是池内存翻倍
   （n=77 时 8 lane 元素 3.7 KB → 16 lane 7.4 KB；22 个元素的池约 90 KB → 180 KB，仍可接受）
   与寄存器压力上升。
3. OpenCL 侧的"优化经验"**可迁移的很少**：那些 kernel 是 GPU 目标（256 线程/工作组、LDS 暂存、
   合并访存、占用率调优），其文档（`ECM_OPERATOR_ANALYSIS.md`、`DEV_CPU_MONT_AVX_PLAN.md`）
   讨论的是 GPU 占用率与合并访存，和 SIMD-IFMA 的瓶颈（端口饱和 + 串行 carry）不是同一件事。

### 15.3 结论与建议

| 路径 | 预期收益 | 成本/风险 | 建议 |
|---|---|---|---|
| **Karatsuba / Toom-Cook** | **−15%（n=154）→ +136%（n=25），均值纯亏**；盈亏平衡 ≈10 250 bit | 高（本次原型踩了 3 个坑） | **不做** |
| **自研 FFT/NTT** | 在 ≳10 000 bit 才有意义；但那正是 Prime95 的强区（已快我们 6×）| 极高（误差控制/旋转因子表/尺寸分档） | **不做**（战略上交给 Prime95）|
| **16-lane 加宽批** | **−10~15%**（摊薄非 madd 固定开销）| 中（内核与池改造，ASI 不变） | **推荐，下一步首选** |
| **tail carry 链 ILP/融合** | −5~8% | 中高（$2^k=1$ 折叠边界，历史上有非规范值 bug）| 可做，排在后面 |

**一句话**：复杂度换不来收益（我们的 schoolbook 太密、修正开销是 O(n) 的 4 指令/列），
真正还有空间的是**把批从 8 lane 加宽到 16 lane**——它摊薄的是 load/store/add 这些"每向量一次"的固定成本，
而这正是我们与 FFT 之间差的那部分。

> 复现：`tools\build_tool.bat tools\bench\mers_karatsuba.cpp` 后
> `build_vs18\tools\mers_karatsuba.exe <k> 2 1200`（`levels=0` 行是纯 schoolbook 基线，
> 与生产 mul 逐字节相同才算通过；levels=1 已在 n=25/58/77/116/154 验证 identical）。
