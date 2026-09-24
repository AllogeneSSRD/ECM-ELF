# ECM 参数化综合分析报告

> 依据：`.refactor/ecm/README`、`.refactor/ecm/rho.c`、`.refactor/ecm/parametrizations.c`、`.refactor/ecm/ecm.c`、
> 论文 *ECM using Edwards curves* (Bernstein–Birkner–Lange–Peters, ePrint 2008/016)、
> *Revisiting ECM on GPUs* (Wloka et al., ePrint 2020/1265)、
> *Parametrizations for Families of ECM-friendly curves* (Gélin–Kleinjung–Lenstra, 2016/1092)、
> [Explicit-Formulas Database (EFD)](https://hyperelliptic.org/EFD/)。

---

## 0. 先讲结论：参数化到底在"调"什么

ECM 对素数因子 `p` 成功当且仅当所选椭圆曲线 `E` 的群阶 `#E(F_p)` 是 **B1-光滑**（stage 1）或 "B1-光滑直到一个 B2 内的因子"（stage 2）。`#E(F_p)` 落在 Hasse 区间 `[p+1−2√p, p+1+2√p]` 内、近似随机。

不同参数化（parameterization）**唯一改变的量**，是 `#E(F_p)` 的**平均小素数估值结构**——即 `#E` 被哪些小素数 `ℓ` 以多大的指数整除。它不影响单条曲线上的加减乘除的正确性，只影响：

1. 群阶被某个已知整数 `T`（扭子群/挠子群）整除 → 未知余因子变小 `#E/T` → 更易光滑；
2. 群阶的"额外光滑性"（Galois 结构，见 §8）；
3. 曲线**生成**成本与**点运算**成本（工程侧）。

因此比较参数化就是比较：**(扭子群 T, 有效除子 D, 生成开销, 每比特运算开销, σ 的语义)**。

---

## 1. 参数化全景表（GMP-ECM `-param` 0–8 + 论文曲线）

GMP-ECM 的 `param` 枚举定义在 `ecm.h:179-189`。所有 Montgomery 形式曲线方程均为
`B·y² = x³ + A·x² + x`，stage 1 起始点取 `(x0 :: 1)`（只用 XZ 坐标，无需 y0）。

| 编号 | 名称 | 曲线形式 | 已知扭子群 T | σ 的语义 | A、x0 公式 | 生成开销 |
|---|---|---|---|---|---|---|
| 0 | Suyama | Montgomery | **Z/12**（12=2²·3） | 自由随机种子 | `u=σ²−5, v=4σ`；`A=(v−u)³(3u+v)/(4u³v)−2`；`x0=u³/v³` | **高**（含模逆） |
| 1 | batch square | Montgomery | Z/4（2²），d 为平方带来 2-adic 增强 | 32-bit 随机数 | `d=σ²/2⁶⁴`；`A=4d−2`；`x0=2` | 低 |
| 2 | batch 2（6-挠） | Montgomery | **Z/6**（2·3） | 标量乘数 | `P=σ·(−3:3:1)` 在 `y²=x³+36`；`x3=(3x+y+6)/(2(y−3))`；`A=−(3x3⁴+6x3²−1)/(4x3³)`；`x0=2` | **极高**（完整标量乘 + 2 次逆） |
| 3 | batch 32-bit d | Montgomery | Z/4（2²） | 32-bit 随机数 | `d=σ/2³²`；`A=4d−2`；`x0=2` | **最低** |
| 5 | Weierstrass | `y²=x³+Ax+b` | 无保证（用户自选） | 无（A,x0,y0 直接给定） | `b=y0²−x0³−A·x0 mod N` | 用户提供 |
| 6 | Hessian | `X³+Y³+Z³=D·X·Y·Z` | **Z/3×Z/3**（需 p≡1 mod 3） | 无（D,x0,y0 给定） | — | 用户提供 |
| 7 | twisted Hessian | `a·X³+Y³+Z³=d·X·Y·Z` | **Z/3×Z/3** | 无（D=a³/d, x0,y0 给定） | — | 用户提供 |
| 8 | `-torsion` | 依构造 | Z5/Z7/Z9/Z10/**Z/2×Z/8**/Z/3×Z/3/Z/4×Z/4 | σ 生成扭子群 + 无穷阶点 | Atkin–Morain 等构造 | 依构造 |
| — | Edwards（论文） | `x²+y²=1+d·x²y²` | 恒含 Z/4；可达 **Z/8, Z/12, Z/2×Z/8(16)** | 无（直接由 d 参数化） | §5 定理 4.5/4.9 | 低（小 d） |
| — | twisted Edwards（论文） | `a·x²+y²=1+d·x²y²` | 可达 Z/2×Z/6, Z/2×Z/8 | 无（由 (a,d) 参数化） | GKL17 族 | 低 |

> 注：`param 4` 保留未用。默认参数化由 `get_default_param()`（`parametrizations.c:463`）选择：64-bit 机器→**param 1**，32-bit 机器→**param 3**（即"batch"族）。

---

## 2. σ 的含义（逐参数化精确语义）

**这是最容易混淆的地方**：σ 在不同参数化里是完全不同的东西，只有 Suyama（param 0）里才是经典的 "Suyama 参数"。

- **param 0 (Suyama)**：σ 是**自由随机种子**。通过 `u=σ²−5, v=4σ` 把 σ 映射成一条带 12 阶有理挠点的曲线。非法 σ 值（模 N 的 `{0,1,3,5,5/3,−1,−3,−5,−5/3}`）会导致退化曲线/退化起点，代码里显式拒绝 `{0,1,3,5}` 并留 TODO 补齐其余（`parametrizations.c:161,177-183`）。σ 无位宽约束。

- **param 1 (batch square)**：σ 是 **32-bit 随机数**（`get_curve_from_random_parameter` 中 `bitsize=32`）。`d = σ²/2⁶⁴ mod N` 是"按 2⁶⁴ 缩放的随机**平方**"（σ² 与 2⁶⁴ 都是平方）。要求 σ<2³² 使 σ² 落入 64-bit。

- **param 2 (batch 2)**：σ≥2 是**标量乘数**——在固定曲线 `y²=x³+36`（有 6 阶挠点）上计算 `P = σ·(−3:3:1)`，再由 P 的坐标导出新 Montgomery 曲线的 A。σ 越大生成越贵。

- **param 3 (batch 32-bit d)**：σ 是 **32-bit 随机数**，`d = σ/2³² mod N` 是"随机值"（**不保证是平方**）。非法值 `d ∈ {0,1,−1/8}` 被拒绝：`d=0 ⇔ A=−2`、`d=1 ⇔ A=2`（Montgomery 奇异条件 A²=4）、`d=−1/8 ⇔ A=−5/2`（使起点 x=2 退化）。

- **Edwards / GPUs 论文**：**没有 σ**。曲线由 Edwards 系数 `d`（twisted 情形由 `(a,d)`）直接定义；"Revisiting ECM on GPUs" 采用 [GKL17] 的 ECM-friendly 曲线族构造，全文中 σ 一词不出现。

---

## 3. 单条曲线生成开销

从 `parametrizations.c` 逐函数看成本（M=模乘，S=模方，I=模逆）：

| 参数化 | 生成成本 | 说明 |
|---|---|---|
| **param 3** | ≈ 1 次常数逆(2³²，可预计算) + 1 M | `A=4·(σ/2³²)−2`。**x0=2 固定**，故 P、2P 可**符号预计算** |
| **param 1** | ≈ 1 次常数逆(2⁶⁴) + 2 M | `A=4·(σ²/2⁶⁴)−2`，x0=2 固定 |
| **param 0 (Suyama)** | ≈ 10 M/S + **1 I** + gcd 检查 | 需对 `b·z` 求逆做归一化（`parametrizations.c:204-220`），最贵 |
| **param 2** | **完整标量乘** σ·(−3:3:1)（O(log σ) 次倍点+加点）+ **2 I** | `addchain_param` 递归加法链 + 两次 `mpres_invert` |
| **Edwards (论文)** | 低 | 直接用小整数 (a,d) 与小时的非挠基点；乘法退化为小常数乘 |
| **GKL17 族** | 论文未量化 | 曲线由参数族公式直接构造 |

### 关键工程洞察：param 3 为何"几乎零成本"建曲线

对 `A=4d−2`，Montgomery 倍点公式里的常数 `a24=(A+2)/4 = d`（`EFD dbl-1987-m-3`）。于是：

1. **起点固定** `x0=2` ⇒ `P=(2:1)` 与 `2P` 可符号算出（见 `cgbn_stage1.cu` 的 `set_p_2p`）：
   `X_P=2, Z_P=1, X_2P=(x²−1)²=9, Z_2P=4x(x²+Ax+1)=64d+8`。
   曲线建立**零曲线级算术**。
2. **"乘以 a24"退化为"乘以 32-bit 标量"**（`special_mult_ui32`），比满模乘便宜一个量级。这正是 workspace `ECM_OPERATOR_ANALYSIS.md` 里 `double_add_v2` 出现 `special_mult_ui32×1` 的原因。

生成开销排序：**param 3 < param 1 < Edwards(小 d) < Suyama(param 0) < param 2**。

---

## 4. 单点运算效率（EFD 成本，S 约 0.67–0.8 M）

### 4.1 Montgomery XZ（差分梯形，GMP-ECM GPU / workspace 所用）

来源 [EFD montgom-xz](https://hyperelliptic.org/EFD/g1p/auto-montgom-xz.html)：

| 运算 | 条件 | 成本 |
|---|---|---|
| 倍点 DBL | `a24=(a+2)/4` 预计算 | **2M+2S** |
| 倍点 DBL | 一般 | 3M+5S |
| 差分加 diffADD | — | **4M+2S**（Z1=1 时 3M+2S） |
| **融合梯形（每比特）** | `a24` 预计算 | **6M+4S**（Z1=1 时 5M+4S） |

- 每比特融合梯形 ≈ **6M+4S ≈ 8.7–9.2 M**（含 a24 常数乘）。
- 只能做**差分加法**（需要 P、Q、P−Q 三者），故只能用"加法链"（如 PRAC）而非任意窗口法。
- workspace 实测（`ECM_OPERATOR_ANALYSIS.md`）：`double_add_v2` = 4 `mont_mul` + 4 `mont_sqr` + 4 add + 4 sub + 1 `special_mult_ui32` + 1 shift，与 6M+4S 结构一致（常数乘单列）。

### 4.2 Twisted Edwards a=−1，扩展坐标 (X:Y:T:Z)（"Revisiting ECM on GPUs" 所用）

来源 [EFD twisted-extended-1](https://hyperelliptic.org/EFD/g1p/auto-twisted-extended-1.html)：

| 运算 | 条件 | 成本 |
|---|---|---|
| 统一加法 ADD | 一般 | **8M**（或 8M+1k，k=2d） |
| 混合加法 madd | Z2=1 | **7M** |
| 双混合加法 mmadd | Z1=Z2=1 | **6M** |
| 倍点 DBL | 扩展 | **4M+4S**（Z1=1 时 3M+4S） |
| 三倍点 TPL | — | 11M+3S |

论文实际数值：通用加法 **9M**（其 Algorithm 1 计数）、混合加法 **7M**、倍点 **3M+4S / 4M+4S**、三倍点 **9M+3S**（[HWCD08]/[BCL17]）。

### 4.3 Edwards (a=1) 坐标（EECM-MPFQ 早期版）

| 坐标 | 加法 | 倍点 |
|---|---|---|
| 标准 projective | 10M+1S+1D | 3M+4S |
| inverted | 9M+1S+1D | 3M+4S+1D |

加法律**强统一**（同式可倍点），且当 `d` 非平方时**完备**（对所有输入无例外，天然抗侧信道）。

### 4.4 横向结论（Edwards 论文 §3.5 的计数）

- GMP-ECM（Montgomery/PRAC）：实测 **≈ 9 M/bit**（B1=10⁶ 时 12 982 280 次模乘 / 2 196 070 次加法）。
- EECM-MPFQ：倍点 **7 M**（3M+4S）+ 加法 **12 M**（仅占 ε 比例，ε→0 随窗口增大）；总 **7+12ε < 9 当 ε<1/6**。用**有符号滑动窗口 + 批量素数**把 ε 压到很小，故乘法总数更少。

---

## 5. 成功概率与扭子群

### 5.1 扭子群对成功率的直接作用

`#E(F_p)` 被 T 整除 ⇒ 未知余因子约 `#E/T`。若 `#E≈p`，则成功率 ≈ 一个规模 `p/T` 的随机整数为 B1 光滑的概率。**T 越大，成功率越高**（单条曲线）。

Montgomery 形式的群阶恒被 **4** 整除（README §2 长篇 NOTE）。各参数化把 T 提升到不同水平：

| 参数化 | T（整除性） | 说明 |
|---|---|---|
| param 3 | 4 | 丢 3-挠，只保留 2-挠 |
| param 1 | 4（+2^(1/3) 的 2-adic 平均增强） | d 为平方略微增强 2-adic |
| param 2 | 6 | 6-挠点 |
| param 0 Suyama | **12** | 2²·3 |
| Edwards Z/12 | **12** | 恒有 4 阶点 + 3 阶点 |
| Edwards Z/2×Z/8 | **16** | 论文 §4–6 构造 |
| Hessian Z/3×Z/3 | 9（p≡1 mod 3） | 3² |

### 5.2 论文给的具体曲线（Edwards 论文 §6）

- **Z/2×Z/8（16 整除）**：`(a,b,e,f)=(3,1,19,33)` ⇒ `d = 161²/17⁴ = 161²/289²`，8 阶点 `(17/7,17/7)`，非挠点 `(17/19,17/33)`。曲线 `x²+y²=1+(161²/289²)x²y²`（论文 §3.6 数值示例即用此曲线）。
- **Z/12**：`(a,b,e,f)=(3,2,23,7)` ⇒ `d = −11·13³/5²`，3 阶点 `(5/13,−1/13)`，非挠点 `(5/23,−1/7)`。
- 参数化（通用）：Z/2×Z/8 由 `x8=(u²+2u+2)/(u²−2)`, `d=(2x8²−1)/x8⁴`（定理 4.5）；Z/12 由 `x3=(u²−1)/(u²+1)`, `d=(u²+1)³(u²−4u+1)/((u−1)⁶(u+1)²)`（定理 4.9）。

---

## 6. 成功概率差异的数学根源（核心）

分三层，从浅到深：

### 6.1 层一：扭子群（Mazur 定理）

Mazur 定理：Q 上椭圆曲线的挠群只有 15 种。Edwards 形式**恒有 4 阶点**（`(±1,0)`），故其挠群 ⊆ {Z/4, Z/8, Z/12, Z/2×Z/4, Z/2×Z/8}。构造给定挠群的曲线即固定了 `#E mod ℓ` 中 ℓ=2,3（对应 4、12、16…）的估值。这是**"免费"的整除性**，是成功率差异的第一来源。

### 6.2 层二：Galois 额外光滑性（Barbulescu–Bos–Bouvier–Kleinjung–Montgomery 2013）

关键发现：**#E 比"同规模随机整数除以 T"还要光滑**。原因在于 `#E` 的除法多项式（division polynomials）的 Galois 结构使群阶在**所有**小素数 ℓ 上的平均估值都高于随机整数，而不仅是挠群固定的那几项。

GMP-ECM 把这一整块编码进 `rho.c` 的 `ECM_EXTRA_SMOOTHNESS = 3.134`：`#E` 的平均光滑性等价于随机整数 `#E/exp(3.134) ≈ #E/22.97`。注释（`rho.c:46-53`）给出 Suyama 曲线的实测 2-adic/3-adic 估值 `2^3.323·3^1.687 ≈ 63.9`，远超朴素挠群贡献 `2^2.5·3^1.333 ≈ 24.5`。

### 6.3 层三：定量——"有效除子 D"

把 6.1+6.2 合并成一个量：**群阶的光滑性等价于随机整数 `#E/D`**。由 `ecm.c:46-56` 与 `print_expcurves`（`ecm.c:1552-1596`）的实际用法：

| 参数化 | 校正常数（源码） | 有效除子 D = 常数 × exp(3.134) | 相对 Suyama 需曲线数 |
|---|---|---|---|
| Suyama / param 2 | 1.0 | **≈ 22.97** | 1.0× |
| param 1 (d 平方) | `EXTRA_SMOOTHNESS_SQUARE = 0.41638…` = 2^(1/3)/(3·3^(1/128)) | **≈ 9.57** | ≈ 2.40× |
| param 3 (d 随机) | `EXTRA_SMOOTHNESS_32BITS_D = 0.33048…` = 1/(3·3^(1/128)) | **≈ 7.59** | ≈ 3.03× |

**成功率（单条曲线）随 D 单调上升**：`Pr ≈ ρ( ln(p/D) / ln B1 )`（Dickman ρ）。因此：

- param 3 单条曲线成功率 ≈ Suyama 的 **1/3**（7.59/22.97）；param 1 ≈ **1/2.4**。
- 常数 `0.33048 ≈ 1/3`、`0.41638 ≈ 2^(1/3)/3` 的代数结构清晰说明：**batch 族把 Suyama 的 3-挠"除以 3"丢掉（只留 Z/4），param 1 额外靠 d 是平方拿回 2^(1/3) 的 2-adic 增益**。`3^(1/128)` 是 Galois 修正的一阶项。

### 6.4 层四：为什么"16 比 12"没想象中那么强（Edwards 论文的修正）

朴素启发：Z/2×Z/8 使 `#E` 被 16 整除 vs Suyama 被 12 整除 ⇒ 16/12 = 1.33 倍优势。**这是错的**。Edwards 论文（AMS 版）明确指出：计入 Galois 额外光滑性后，16 相对 12 的实际优势只有 **1.051–1.093 倍（约 5–9%）**，而非 33%。

原因：额外光滑性与挠群**不独立**。挠群越大，"已知被整除"的部分越大，Galois 额外光滑性贡献的相对增量越小；两者近似相乘的朴素模型高估了大挠群带来的边际收益。GMP-ECM 正是用 `ECM_EXTRA_SMOOTHNESS`（而非朴素 `ρ(p/T)`）来计算期望曲线数，才能与实验吻合（README §2 NOTE）。

> **经验实测的 D_eff 归一化**：§6.3 的 D 是源码常数（GMP-ECM 标定）。本仓库 `tools/ecm_prob/` 工具用穷举素数集实测 stage-1 命中率，再按 `ρ_local((log p − log D)/log B1) = f` 反解出**经验 D_eff**——把位宽从成功率里扣除后剩下的纯曲线量。实测 D_eff 跨位宽(15–25) 近乎恒定（Suyama≈20.8、Edwards Z/12≈25.8、param3≈6.4），且 Edwards Z/12 稳定压过 Suyama、Z/2×Z/8 仅比 Z/12 高 ~1.06×，均与本文结论一致。口径与归一化细节见 `tools/ecm_prob/README.md` §「D_eff 归一化方法」。

---

## 7. 综合对比与工程取舍

| 维度 | Suyama (0) | batch 1/3 | Edwards Z/12 | Edwards Z/2×Z/8 | Hessian Z/3×Z/3 |
|---|---|---|---|---|---|
| 扭子群 | Z/12 | Z/4 | Z/12 | 16（Z/2×Z/8） | Z/3×Z/3 |
| 有效除子 D | ≈22.97 | ≈9.6 / 7.6 | ≈22.97 | >22.97（≈5–9% 更优） | 9（需 p≡1 mod 3） |
| 生成开销 | 高（模逆） | 极低 | 低（小 d） | 低（小 d） | 低 |
| 每比特运算 | 6M+4S（差分梯形） | 同左（a24=32-bit） | 7+12ε M（窗口） | 同左 | 依坐标 |
| 加法律 | 差分（非完备） | 差分（非完备） | **完备**（d 非平方） | **完备** | 统一 |

**结论式的判断：**

1. **追求单条曲线成功率**：Suyama（D≈22.97）或 Edwards Z/2×Z/8（16 整除，再赢 5–9%）。GMP-ECM 默认（单核、非 batch）就选它们。
2. **追求吞吐（GPU/批量）**：param 1/3。牺牲 **2.4–3× 的有效除子 D**，换取：极低生成成本、固定 x0=2 的符号预计算、a24 退化为 32-bit 标量乘、可并行批量处理。净收益由吞吐碾压（workspace 与 GMP-ECM `-gpu` 均用 param 3）。
   ⚠ 这个 2.4–3× 是 **D 的比值**，不是成功率比值。成功率 = ρ(log(p/D)/log B1)，D 只以 log D / log B1 进入，
   所以实际成功率差距小得多且随位宽/B1 变化：B1=256 实测（bit 15–40，本仓库 `tools/ecm_prob`）Suyama/param3 =
   **1.30×–1.8×**（bit20：30.47% vs 21.64% = 1.41×），模型在 bit130/B1=44e6 给 1.18×。
   定量对照见 `docs/ECM_Montgomery_STAGE1.md` §19.5。
3. **现代 GPU 前沿**（Wloka et al. 2020/1265）：转向 **a=−1 twisted Edwards + GKL17 族 + 扩展坐标 (X:Y:T:Z)**，通用加法 9M/混合 7M/倍点 3M+4S，单曲线单线程、w=4 NAF 链；RTX 2080 Ti 上 192-bit B1=8192 达 **214 k trials/s**、448-bit B1=50k B2=5M 达 **2.78 k trials/s**。该论文**明确不评估成功率**，只做吞吐对比。
4. **成功率差异的数学根源**一句话：不同参数化改变了 `#E(F_p)` 的**已知整除因子（扭子群）+ 平均 Galois 光滑性（有效除子 D）**，从而改变 `#E/D` 的 Dickman-ρ 光滑概率；差异大小由 `D` 的比值（≈3×、≈2.4×、≈1.05–1.09×）定量刻画。

---

## 8. 参考

- GMP-ECM 源码：`README`（§2 光滑性 NOTE、§6 参数化）、`rho.c`（`ECM_EXTRA_SMOOTHNESS=3.134`、Dickman ρ 实现）、`parametrizations.c`（param 0–3 公式）、`ecm.c`（`EXTRA_SMOOTHNESS_*` 与 `print_expcurves`）。
- [EFD: Montgomery XZ](https://hyperelliptic.org/EFD/g1p/auto-montgom-xz.html)、[EFD: twisted Edwards extended (a=−1)](https://hyperelliptic.org/EFD/g1p/auto-twisted-extended-1.html)、[EFD: twisted Edwards extended](https://hyperelliptic.org/EFD/g1p/auto-twisted-extended.html)。
- Bernstein, Birkner, Lange, Peters, *ECM using Edwards curves*, [ePrint 2008/016](https://eprint.iacr.org/2008/016)（Math. Comp. 82 (2013)）。
- Wloka, Richter-Brockmann, Stahlke, Kleinjung, Priplata, Güneysu, *Revisiting ECM on GPUs*, [ePrint 2020/1265](https://eprint.iacr.org/2020/1265)（CANS 2020）。
- Gélin, Kleinjung, Lenstra, *Parametrizations for Families of ECM-friendly curves*, [ePrint 2016/1092](https://eprint.iacr.org/2016/1092)（ISSAC 2017）。
- Barbulescu, Bos, Bouvier, Kleinjung, Montgomery, *Finding ECM-Friendly Curves through a Study of Galois Properties*, [ePrint 2012/070](https://eprint.iacr.org/2012/070)。
- Kruppa, *Optimising the Elliptic Curve Method of Factoring*, PhD thesis (ρ 函数的 smoothness 模型来源)。
