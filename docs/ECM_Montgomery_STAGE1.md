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
| M4 | 性能与链优化/缓存 | **M4a ✅**（`s` 位数组化，见 §10）；**M4b ✗ 已关闭**（EFD `ladd-1987-m-3` = `dbl`+`dadd` 之和，无子表达式共享；我们已在 6M+4S，§7.1）；**M4c/M4d ⏳**（逐素素数链表 + 缓存，§8.4） | M4a：oracle 27/27 + SIMD/标量逐 lane PASS；M4c：交替 A/B 相对基线增益 |

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
3. **M4c**：逐素数链表 —— 先用 gmp-ecm 的 `Lchain_codes.dat`（同 B1 时）作为**性能上界与正确性对拍**，
   再决定复刻其生成器还是自研：我们的目标函数是 **madd 条数**，而它们按 FFT 计数，且折叠域里
   `S ≈ M/2` ⇒ 最优链可能与它们不同，自研有正当理由。
4. **M4d**：定稿缓存文件格式（键 (B1,torsion) + 版本 + sha256 校验）并与 `--mont-chain <file>` 接线；
   分布式场景下随客户端分发。
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

- 8 任务 8 线程 = 6.62×（理想 8×；差距来自尾批、8 个 IFMA 批共用 L2/L3 带宽与频率回落）。
- **确定性**：`--mont-threads 1` 与 `--mont-threads 8` 的共享存档，除 `TIME=`
  （人类可读时间戳，gmp-ecm 同字段）外 **逐字节相同** ⇒ 并行不改变任何数学结果，可作为回归判据。

### 10.4 同机硬指标对照（gmp-ecm 7.0.6 `ecm-zen3.exe`，B2=B1，单线程）

用户指定测试区间 **B1 = 1e5 ~ 1e6**，本轮把两端都测了（1 线程 = 1 核；SIMD 一比 8 条曲线）：

| 用例 | 本实现 1 线程 | 本实现 8 线程 | gmp-ecm 1 线程 | 每核倍数 |
|---|---|---|---|---|
| M1277，B1=1e5，64 曲线 | 0.135 s/curve | 0.0203 s/curve（6.62×）| 0.234 s/curve | **1.74×** |
| M1277，B1=1e6，16 曲线 | 1.418 s/curve | —（仅 2 任务）| 3.551 s/curve | **2.50×** |
| M3001，B1=1e5，64 曲线 | 0.526 s/curve | 0.0785 s/curve（6.71×）| 1.148 s/curve | **2.18×** |

- 复测（无后台负载）：M1277 = 0.1402 s/curve、M3001 = 0.5872 s/curve ⇒ 上表 ±5% 内可复现。
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

### 11.1 每 bit 预算闭合（M3001，n52 = 58，B1=1e5，1 线程，64 曲线）

| 项 | 值 | 来源 |
|---|---|---|
| `ifma_mont_mul`（折叠域，8 lane 批） | 2138 ns | mers_sqr_phases（短测，boost 时钟）|
| `ifma_mont_sqr`（折叠域，8 lane 批） | 1472 ns | 同上 |
| 模型：每 bit = 6M + 4S | 6×2138 + 4×1472 = **18.7 µs** | §2.4 公式 + 上面两项 |
| 实测：0.5872 s/curve ÷ 8 ⇒ 4.70 s/batch ÷ 144344 bit | **32.6 µs/bit** | ecm.exe 实测（持续负载时钟）|
| 比值 | 1.74× | ≈ 2.1 GHz（持续）→ 3.5 GHz（短测 boost）的**时钟差** |

⇒ **ladder 里没有"神秘开销"**：per-bit 时间 = 6M+4S 的域运算成本 × 时钟因子。
（此前担心的 add/sub/memcpy 辅助 pass 只占个位数百分比。）

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

- 乘积相位 = 3248 条 madd 指令 / 887.8 ns ≈ **1.67 madd/cycle**（Zen5 的 VPMADD52 zmm 顶 ≈ 2/cycle）⇒ 乘积相位已贴顶。
- **操作数幅度完全不影响**（稀疏 vs 随机差 <2%）⇒ `drain` 循环里的 6 次上限**实际不级联**，
  我先前"finish 数据相关"的猜测**被自己的测量证伪**（记录在此以免复犯）。
- `finish` = 每个域运算的 **~26%**，且实测 ~0.7 ops/cycle ⇒ **串行 carry 链受限**（\(2n\) 列归一 + 折叠回写 + ≥1 次 \(n\) 列 drain + 2 遍规范化，共 ~4n 列迭代）。
  ⇒ 这是**单个域运算内**唯一还剩的靶子，上限约 10%（把 finish 砍半 ⇒ per-bit −13%）。

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
| 平方折扣（`S < M`） | ✅ 折叠域已拿到 0.69M（理论 0.5M，余量 ~8.7%）|
| madd 吞吐 | ✅ 乘积相位 ~1.67/2.0 per cycle |
| `finish` | ⚠️ 每个域运算 26%，串行 carry；微优化上限 ~10% |
| 步数（链） | ❌ **未动** —— M4c，预计 −15~25%，当前最大杠杆 |
| 并行度 | ✅ 任务级（§10.3：4.37×/6.62×），受批数上界约束 |
| 任务固定开销 | ✅ 2.1%（B1=1e6、8 曲线）|
