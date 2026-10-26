# ECM 参数化、挠子群与成功率模型

本文汇集参数化的数学与第三方实现依据。来源：`ECM_PARAMETERIZATION_ANALYSIS.md §§0–6、8`，另据 GKL17 原论文及 [PrMers](https://github.com/cherubrock-seb/PrMers) 源码补充 §7。第三方源码行号以所引提交为准；论文结果不声明本仓库已经实现相应曲线族或获得同样性能。当前可选构造、sigma 约束与跨阶段身份以 [Stage 1](STAGE1.md) 和 [Stage 2](STAGE2.md) 为准，当前计时口径见 [性能](../performance/STAGE2.md)；论文译述与条件见 [论文概述](../paper/ECM%20友好曲线与%20CANS%202020%20论文概述.md)。

## 0. 先讲结论：参数化到底在"调"什么

在模素因子p的非奇异曲线上，Stage1将基点P消去的充要条件是 `ord(P)` 整除所用标量；若标量为 `c·lcm(1,…,B1)`，应核对 `ord(P)/gcd(ord(P),c)` 的素数幂约束。单素数Stage2还可消去一个B2内剩余素数的一次幂。完整群阶 `#E(F_p)` 落在Hasse区间 `[p+1−2√p,p+1+2√p]`，其光滑性用于成功率模型，但不能代替具体点阶，也不能仅凭“所有素因子≤B1”忽略幂次。对复合目标实际取出proper factor还要求最终GCD既非1也非整个目标。

不同参数化（parameterization）改变曲线族、基点及其阶，并影响 `#E(F_p)` 的**平均小素数估值结构**——即 `#E` 被哪些小素数 `ℓ` 以多大的指数整除。它不改变正确群运算的数学规则，主要影响：

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
| — | ECM-ELF Edwards / Atkin–Morain | `x²+y²=1+d·x²y²`（`a=1`） | 设计为 Z/2×Z/8，须满足非退化条件 | 64 位正整数，作为固定辅助曲线上的倍点标量 | `[σ](12,40)` → `d` 与起点 `P`，详见 §2 | 辅助曲线标量乘、模逆和检查 |
| — | Edwards（论文） | `x²+y²=1+d·x²y²` | 恒含 Z/4；可达 **Z/8, Z/12, Z/2×Z/8(16)** | 无（直接由 d 参数化） | §5 定理 4.5/4.9 | 低（小 d） |
| — | GKL17 family (iv) | `−x²+y²=1−e⁴x²y²` | 有理挠群 Z/2×Z/4；另有模 p 的平均 2 进光滑性优势 | 论文参数是辅助曲线上的点，不是本仓库 sigma | 辅助曲线、表 1 第 4 行及表 2，详见 §7 | 取决于选点和模 N 转换 |

> 注：`param 4` 保留未用。默认参数化由 `get_default_param()`（`parametrizations.c:463`）选择：64-bit 机器→**param 1**，32-bit 机器→**param 3**（即"batch"族）。

---


## 2. σ 的含义（逐参数化精确语义）

**这是最容易混淆的地方**：σ 在不同参数化里是完全不同的东西，只有 Suyama（param 0）里才是经典的 "Suyama 参数"。

- **param 0 (Suyama)**：σ 是**自由随机种子**。通过 `u=σ²−5, v=4σ` 把 σ 映射成一条带 12 阶有理挠点的曲线。非法 σ 值（模 N 的 `{0,1,3,5,5/3,−1,−3,−5,−5/3}`）会导致退化曲线/退化起点，代码里显式拒绝 `{0,1,3,5}` 并留 TODO 补齐其余（`parametrizations.c:161,177-183`）。σ 无位宽约束。

- **param 1 (batch square)**：σ 是 **32-bit 随机数**（`get_curve_from_random_parameter` 中 `bitsize=32`）。`d = σ²/2⁶⁴ mod N` 是"按 2⁶⁴ 缩放的随机**平方**"（σ² 与 2⁶⁴ 都是平方）。要求 σ<2³² 使 σ² 落入 64-bit。

- **param 2 (batch 2)**：σ≥2 是**标量乘数**——在固定曲线 `y²=x³+36`（有 6 阶挠点）上计算 `P = σ·(−3:3:1)`，再由 P 的坐标导出新 Montgomery 曲线的 A。σ 越大生成越贵。

- **param 3 (batch 32-bit d)**：σ 是 **32-bit 随机数**，`d = σ/2³² mod N` 是"随机值"（**不保证是平方**）。非法值 `d ∈ {0,1,−1/8}` 被拒绝：`d=0 ⇔ A=−2`、`d=1 ⇔ A=2`（Montgomery 奇异条件 A²=4）、`d=−1/8 ⇔ A=−5/2`（使起点 x=2 退化）。

### 本仓库 Edwards / Atkin–Morain（`curve=ed`，没有 `param`）

本仓库的 Edwards σ 是**固定辅助 Weierstrass 曲线上的倍点标量**，不是 Edwards 方程里的系数 `d`，也不是 Suyama 的参数。对实际待分解目标 `N`，在 `C: t²=s³−8s−32` 上取 `Q=(12,40)`，计算 `(s,t)=[σ]Q`，再令

`α=(s−9)/(t+s+16)`，`β=2α(4α+1)/(8α²−1)`，`h=2β−1`，`d=(2h²−1)/h⁴`。

目标曲线是 `E: x²+y²=1+d·x²y²`（`a=1`），起点也是 σ 的确定函数：

`Pₓ=h(4β−3)/(6β−5)`，`Pᵧ=h(t²+50t−2s³+27s²−104)/[(t+3s−2)(t+s+16)]`。

以上除法均指模 `N` 的可逆元；[公共构造入口 `construct`](../../src/ecm/curve/edwards.h#L37)以 Jacobian 坐标求辅助倍点、归一化一次，并检查各分母、`d(1−d)` 的单位性和 `P` 在曲线上。逆元不存在时求 GCD：真因子与退化构造分开报告；`σ=0` 直接视为退化。构造还输出 `a24=(A+2)/4=1/(1−d)`，供转成 Montgomery x-only 后使用；推导和 Prime95 `sigma_type=0` 的坐标约定见[Edwards 数学说明](../reference/ECM_EDWARDS_MATH.md#1-曲线构造atkin-morainsigma_type0)。

调用方接受固定的正 `uint64` σ；固定批次的第 `i` 条使用 `σ+i`，先检查 64 位回绕。未指定 σ 时，每条曲线分别取随机 σ（生成器排除 `σ≤5`）；保存/恢复记录每条曲线的完整 64 位值，不能用批次首值替代各条随机 σ。[CPU 批次](../../src/ecm/stage1/edwards/cpu/workflow.cpp)、[CUDA 批次](../../src/ecm/stage1/edwards/cuda/workflow.cpp#L197)和[输入校验](../../src/io/normalize.cpp)采用这套身份。固定 σ 可以取生成器不会随机选到的小正整数，但仍须通过构造与单位性检查。

**现有 Edwards 执行路径共享上述同一构造，区别在点算术和标量执行：**

| 路径 | 现有算法与约束 | 构造入口 |
| --- | --- | --- |
| CPU GMP 标量 | 扩展 Edwards 坐标、窗口 NAF；单曲线执行和恢复 | [CPU Stage 1](../../src/ecm/stage1/edwards/cpu/ecm_edwards_cpu.cpp#L597)调用 `construct` |
| CPU AVX512-IFMA | 可用 ISA/构建条件下按最多 8 条曲线并行，尾部和异常曲线可走标量 | [SIMD 字典准备](../../src/ecm/stage1/edwards/cpu/simd_edwards.cpp#L889)逐 lane 使用相同构造 |
| CUDA CGBN | 默认窗口 NAF（默认窗口 6），可显式改用二进制标量路径；两者共用曲线批次 | [CUDA 初始化](../../src/ecm/stage1/edwards/cuda/host_math.cpp#L3)调用 `construct` |
| OpenCL | 当前不提供 Edwards Stage 1 | [Stage 1 能力说明](STAGE1.md#方法选择与输入校验) |

三条可运行路径均使用 `48·lcm(1,…,B1)` 作为 Edwards Stage 1 标量；NAF 窗口、点表示、CPU SIMD 归约或 CUDA 倍点策略只改变执行方法，不改变固定 `(N, curve=ed, σ)` 的 `d` 与 `P`。CPU 与 CUDA 将完成点的 Edwards 投影坐标转为 Montgomery x-only `x_M=(Z+Y)/(Z−Y) mod N`，写入 `METHOD=ECM-EDWARDS-AM; VERSION=1; SCALAR=48LCM` 的[文本 save](SAVE.md#5-edwards-save-version1)；[CUDA Stage 2](STAGE2.md#输入与表示)与 [CPU Stage 2](../usage/EDWARDS_CPU_STAGE2.md)按 σ 重构曲线。只比较设备域字数组或未归一化投影坐标，不能证明两条路径的数学点相同。

### 其他 Edwards 论文与第三方构造

GKL17 family (iv) 的数学参数是另一条辅助曲线上的点 `mQ`，经有理函数产生 `e`、`d=−e⁴` 和目标起点；`m` 不是本仓库 σ。PrMers `-iv163` 又从 `curve_seed` 选 `m`；Wloka 等的 GPU 论文只说明使用 GKL17 构造，未给出与本仓库 σ 或 PrMers 种子的逐曲线映射。三者均不能仅凭相同整数值判作同一曲线，详见 §7。

---


## 3. 单条曲线生成开销

GMP-ECM 的 param 0–3 按 `parametrizations.c` 比较；本仓库 Edwards 按公共构造入口另计（M=模乘，S=模方，I=模逆）：

| 参数化 | 生成成本 | 说明 |
|---|---|---|
| **param 3** | ≈ 1 次常数逆(2³²，可预计算) + 1 M | `A=4·(σ/2³²)−2`。**x0=2 固定**，故 P、2P 可**符号预计算** |
| **param 1** | ≈ 1 次常数逆(2⁶⁴) + 2 M | `A=4·(σ²/2⁶⁴)−2`，x0=2 固定 |
| **param 0 (Suyama)** | ≈ 10 M/S + **1 I** + gcd 检查 | 需对 `b·z` 求逆做归一化（`parametrizations.c:204-220`）；比 batch param 1/3 贵 |
| **param 2** | **完整标量乘** σ·(−3:3:1)（O(log σ) 次倍点+加点）+ **2 I** | `addchain_param` 递归加法链 + 两次 `mpres_invert` |
| **Edwards (论文中的固定小参数曲线)** | 低 | 直接用小整数 (a,d) 与小坐标的非挠基点；不代表本仓库的 σ 构造成本 |
| **ECM-ELF Edwards / Atkin–Morain** | 一次 64 位 σ 的辅助曲线标量乘；成功构造路径调用 7 次模逆，并作单位性及点校验 | 在主机上逐条求 `[σ](12,40)`，再计算 `d`、`P`、`a24`；模逆失败可在构造时发现因子，不能按“直接给小 d”计价 |
| **GKL17 family (iv)** | 论文未给出可与上述 Montgomery 构造直接比较的统一成本 | 从辅助曲线 `mQ`、有理函数与非挠起点构造；具体实现还须处理分母不可逆、退化和取样 |

### 关键工程洞察：param 3 为何"几乎零成本"建曲线

对 `A=4d−2`，Montgomery 倍点公式里的常数 `a24=(A+2)/4 = d`（`EFD dbl-1987-m-3`）。于是：

1. **起点固定** `x0=2` ⇒ `P=(2:1)` 与 `2P` 可符号算出（见 `cgbn_stage1.cu` 的 `set_p_2p`）：
   `X_P=2, Z_P=1, X_2P=(x²−1)²=9, Z_2P=4x(x²+Ax+1)=64d+8`。
   曲线建立**零曲线级算术**。
2. **"乘以 a24"退化为"乘以 32-bit 标量"**（`special_mult_ui32`），比满模乘便宜一个量级。这正是 workspace `ECM_OPERATOR_ANALYSIS.md` 里 `double_add_v2` 出现 `special_mult_ui32×1` 的原因。

上述 Montgomery 构造可按其公式讨论生成开销；Edwards 族的选点、模逆与退化处理不同，不能据“小 `d`”把 GKL17 family (iv) 插入同一个总成本排序。实际比较应同时计入构造、点运算和成功率。

本仓库的 7 次模逆是[完整成功分支](../../src/ecm/curve/edwards.h#L168)中辅助点归一化、`α`、`β`、`d`、`a24`、`Pₓ`、`Pᵧ` 的源码调用数；提前遇到因子或退化时会少于此数。辅助倍点使用 Jacobian 坐标，运算量随 `σ` 的有效位数及置位数增长，故不能把“64 位”误读成 64 次模逆。完成 Stage 1 后转换为 save 的 Montgomery `x_M` 还要另行处理 `(Z−Y)` 的逆元/GCD；这属于末点输出，不在上述构造成本内。CPU 标量、CPU IFMA 和 CUDA 均调用公共构造，批量 GPU 计时还包括字典、上传、核函数及回读。[Stage 1 性能页](../performance/STAGE1.md#gpu-edwards-细分实例三曲线族对照)提供同一构建下的完整批次成本，不能用单点公式代替。

---


## 4. 单点运算效率（抽象域运算计数；论文估算 S 约 0.67–0.8 M）

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

### 4.3 Edwards (a=1)：论文坐标与本仓库扩展坐标

下表是 EECM-MPFQ 早期论文的 projective / inverted 公式，**不是**本仓库 CPU/CUDA 的扩展 `(X:Y:Z:T)` 实现：

| 坐标 | 加法 | 倍点 |
|---|---|---|
| 标准 projective | 10M+1S+1D | 3M+4S |
| inverted | 9M+1S+1D | 3M+4S+1D |

论文中的加法律**强统一**；当 `d` 非平方时可满足相应完备条件。本仓库的 Atkin–Morain σ 映射没有逐素因子保证 `d` 非平方，不能据此把当前实现宣称为完备公式或推断侧信道性质。

本仓库扩展坐标的[CPU 模域实现](../../src/ecm/stage1/edwards/cpu/ecm_edwards_cpu.cpp#L248)与[CUDA CGBN 实现](../../src/ecm/stage1/edwards/cuda/point.cuh#L38)使用 `a=1` 的 HWCD 倍点：完整 `(X,Y,Z,T)` 输出为 **4M+4S**。CUDA NAF 在后续只倍点的列可省暂不需要的 `T`，该列为 **3M+4S**，加法或片边界前恢复 `T`；可选算法 5 将 `(X+Y)²−X²−Y²` 改写为 `2XY`，完整/省 `T` 分别是 **5M+3S / 4M+3S**。[NAF 倍点分支](../../src/ecm/stage1/edwards/cuda/naf.cuh#L31)给出实际选择。通用扩展加法为 **10M**（9 次普通乘法和 1 次乘 `d`）；CPU 标量用预先归一化的仿射字典时，混合加法为 **8M**，另有字典生成及批量求逆成本。CPU IFMA 的 lane 约减与调度仍应按其实现和实测计时，不能将单条公式数直接乘以 8 得出耗时。

### 4.4 论文模型与本仓库实现的比较

- GMP-ECM（Montgomery/PRAC）：实测 **≈ 9 M/bit**（B1=10⁶ 时 12 982 280 次模乘 / 2 196 070 次加法）。
- EECM-MPFQ：倍点 **7 M**（3M+4S）+ 加法 **12 M**（仅占 ε 比例，ε→0 随窗口增大）；总 **7+12ε < 9 当 ε<1/6**。用**有符号滑动窗口 + 批量素数**把 ε 压到很小，故乘法总数更少。
- 本仓库 CUDA NAF 的主循环每列做一次倍点，仅非零列加点；宽度 6 的典型非零密度约 `1/7`，因此公式层的粗估是“每列 3M+4S 或 4M+4S，加上约 `10M/7` 的加法”，另计字典、`T` 恢复、模约减和访存。实际 NAF 列数与非零列数由编码决定，不能由此直接推出 GPU 秒数。[同一构建的完整进程矩阵](../performance/STAGE1.md#完整进程成本)使用 GPU1 RTX 4060 Laptop（24 SM、驱动 610.88；利用率≥50% 的整卡遥测功耗 8.90–26.51 W，非进程功耗）、exe SHA256=`fa8b903e9186dada7055e61176fff237411b575c4707a04541f5cc50e15ad3eb`、完整梅森数 `M_p=2^p−1` 的 p=347/677/991/2203/4003、`B1=10000`、每批 768 条、每条件 3 次正式进程；完整进程 wall 包含启动、构造、GPU、GCD/归一化及 save。该条件下 Edwards NAF/PARAM3 的中位成本比为 **1.423–1.719**。它是实现成本比，不是纯点公式代价或成功率比；原始日志与审计位置见性能页。
- Stage 2 接收 Edwards save 后使用 Montgomery x-only `x_M` 与 `a24=1/(1−d)`，沿[共用点与 NTT 多项式管线](STAGE2.md#输入与表示)计算。Edwards 加法/倍点公式的成本优势或劣势主要发生在 Stage 1；Stage 2 的曲线身份通过起点、末点和 `a24` 体现，不能把本节 `M/S` 计数套到 NTT 多项式运算上。

---


## 5. 成功概率与扭子群

### 5.1 扭子群对成功率的直接作用

`#E(F_p)` 被 T 整除 ⇒ 未知余因子约 `#E/T`。若 `#E≈p`，可把规模 `p/T` 的随机整数光滑率当作第一近似；具体成功率还取决于点阶、额外 Galois 光滑性、`B1/B2` 和所用标量。**其他条件相同**时较大的已知整除因子通常有利，不能仅以 T 给跨族成功率排序。

Montgomery 形式的群阶恒被 **4** 整除（README §2 长篇 NOTE）。各参数化把 T 提升到不同水平：

| 参数化 | T（整除性） | 说明 |
|---|---|---|
| param 3 | 4 | 丢 3-挠，只保留 2-挠 |
| param 1 | 4（+2^(1/3) 的 2-adic 平均增强） | d 为平方略微增强 2-adic |
| param 2 | 6 | 6-挠点 |
| param 0 Suyama | **12** | 2²·3 |
| Edwards Z/12 | **12** | 恒有 4 阶点 + 3 阶点 |
| Edwards Z/2×Z/8 | **16** | 论文 §4–6 的固定曲线与本仓库 Atkin–Morain 族都属于这一挠群类型；构造和起点各不相同 |
| Hessian Z/3×Z/3 | 9（p≡1 mod 3） | 3² |

### 5.2 论文给的具体曲线（Edwards 论文 §6）

- **Z/2×Z/8（16 整除）**：`(a,b,e,f)=(3,1,19,33)` ⇒ `d = 161²/17⁴ = 161²/289²`，8 阶点 `(17/7,17/7)`，非挠点 `(17/19,17/33)`。曲线 `x²+y²=1+(161²/289²)x²y²`（论文 §3.6 数值示例即用此曲线）。
- **Z/12**：`(a,b,e,f)=(3,2,23,7)` ⇒ `d = −11·13³/5²`，3 阶点 `(5/13,−1/13)`，非挠点 `(5/23,−1/7)`。
- 参数化（通用）：Z/2×Z/8 由 `x8=(u²+2u+2)/(u²−2)`, `d=(2x8²−1)/x8⁴`（定理 4.5）；Z/12 由 `x3=(u²−1)/(u²+1)`, `d=(u²+1)³(u²−4u+1)/((u−1)⁶(u+1)²)`（定理 4.9）。

本仓库实际运行的 Z/2×Z/8 曲线按 §2 的 Atkin–Morain `σ→(d,P)` 生成，**不是**上述论文固定样本。对目标的奇素因子 `p`，在分母可逆、曲线良好约化时，16 阶有理挠结构给 `#E(F_p)` 提供整除性；Stage 1 的精确命中仍是 `ord(P mod p) | 48·lcm(1,…,B1)`，等价于 `ord(P)/gcd(ord(P),48) | lcm(1,…,B1)`。其中 48 是实际标量约定，不能把论文用 `lcm` 的命中率直接代入本仓库。若 Stage 1 未命中，Stage 2 的收益还取决于剩余点阶因子、`B2` 和实际覆盖的素数集合；NTT 只改变计算成本，不改变相同曲线和覆盖集合的数学命中条件。

---


## 6. 成功概率差异的数学根源（核心）

分三层，从浅到深：

### 6.1 层一：扭子群（Mazur 定理）

Mazur 定理：Q 上椭圆曲线的挠群只有 15 种。Edwards 形式**恒有 4 阶点**（`(±1,0)`），故其挠群 ⊆ {Z/4, Z/8, Z/12, Z/2×Z/4, Z/2×Z/8}。构造给定挠群的曲线即固定了 `#E mod ℓ` 中 ℓ=2,3（对应 4、12、16…）的估值。这是**"免费"的整除性**，是成功率差异的第一来源。

本仓库的 Atkin–Morain 构造选择 Z/2×Z/8 族；此性质约束群阶，不会单独决定所生成**起点**的阶。因而固定 `(N, σ)` 的 CPU/GPU 实现应得到相同数学曲线、起点和命中界限，而更换 σ 后必须重新讨论 `ord(P mod p)`。构造阶段遇到不可逆分母可能直接给出因子，这类命中须与 Stage 1 标量乘、Stage 2 的命中分别统计。

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

此表是 GMP-ECM 对相应 Montgomery 参数化的模型常数；**没有本仓库 Atkin–Morain σ 族的标定行**。不能因为其已知挠群阶为 16，就设 `D_AM=16`、`22.97×16/12` 或套用论文固定 Edwards 曲线的经验 `D_eff`。

**D 决定成功率，但不是线性地决定**：`Pr ≈ ρ_local( ln(p/D) / ln B1 )`（Dickman ρ）。因此：

- 上表的"相对曲线数"一列是 **D 的比值**（22.97/7.59 = 3.03、22.97/9.57 = 2.40），**不是成功率比值**：
  D 只以 `log D / log B1` 进入 u，实际成功率差距远小于此，且随位宽/B1 变化（见 **§6.5–§6.6**）。
- 常数 `0.33048 ≈ 1/3`、`0.41638 ≈ 2^(1/3)/3` 的代数结构清晰说明：**batch 族把 Suyama 的 3-挠"除以 3"丢掉（只留 Z/4），param 1 额外靠 d 是平方拿回 2^(1/3) 的 2-adic 增益**。`3^(1/128)` 是 Galois 修正的一阶项。

### 6.4 层四：为什么"16 比 12"没想象中那么强（Edwards 论文的修正）

朴素启发：Z/2×Z/8 使 `#E` 被 16 整除 vs Suyama 被 12 整除 ⇒ 16/12 = 1.33 倍优势。**这是错的**。Edwards 论文（AMS 版）在其曲线和成功率比较条件下给出：计入 Galois 额外光滑性后，16 相对 12 的优势只有 **1.051–1.093 倍（约 5–9%）**，而非 33%；这不是本仓库 Atkin–Morain σ 族在任意 `B1/B2` 上的实测提升。

原因：额外光滑性与挠群**不独立**。挠群越大，"已知被整除"的部分越大，Galois 额外光滑性贡献的相对增量越小；两者近似相乘的朴素模型高估了大挠群带来的边际收益。GMP-ECM 正是用 `ECM_EXTRA_SMOOTHNESS`（而非朴素 `ρ(p/T)`）来计算期望曲线数，才能与实验吻合（README §2 NOTE）。

> **经验实测的 D_eff 归一化**：§6.3 的 D 是源码常数（GMP-ECM 标定）。本仓库 [`tools/ecm_prob/`](../../tools/ecm_prob/README.md#d_eff-归一化方法) 工具用穷举素数集实测 stage-1 命中率，再按 `ρ_local((log p − log D)/log B1) = f` 反解出**经验 D_eff**。现有 [曲线清单](../../tools/ecm_prob/curves.py#L82)的 Edwards 四条都是论文给定的固定有理曲线，标量为 `lcm(1,…,B1)`；**没有**通过本仓库 Atkin–Morain σ 构造及 `48·lcm` 抽样。已测固定曲线的 D_eff 跨位宽(15–25)近乎恒定（Suyama≈20.8、Edwards Z/12≈25.8、param3≈6.4），论文固定 Z/2×Z/8 的估值与固定 Z/12 接近；这些值仅作为族选择的研究参考，不能标作本仓库当前 Edwards 实现的成功率。

### 6.5 D 的数值含义：三种等价读法（2026-09-24 补）

D 不是"曲线数"，也不只是排序用的抽象分数。它只有一种进入方式 —— `effN = p/D`、`u = (ln p − ln D)/ln B1`
（`tools/ecm_prob/rho.py` 的 `stage1_prob`）—— 沿三个轴看它就有三种等价读法：

**读法 A：D 等价于"把因子位宽减掉 log2 D 个 bit"**（最直观）。因为 D 只以 `ln D` 出现在 `ln p` 旁边：

$$u=\frac{\ln p-\ln D}{\ln B_1}=\frac{\ln 2\cdot(\text{bit}-\log_2 D)}{\ln B_1}$$

| D | 2 | 6.41（实测 param3） | 7.59（理论 param3） | 20.90（实测 param0/Suyama） | 22.97（GMP-ECM 常数） | 41 |
|---|---|---|---|---|---|---|
| log2 D（白送的 bit 数） | 1.00 | **2.68** | 2.92 | **4.39** | 4.52 | 5.36 |

⇒ param0 相对 param3 值 **1.71 bit**（4.39 − 2.68）；论文固定 Edwards Z/2×Z/8 样本（D≈26.4）值 4.72 bit；p−1（D≈2.7）值 1.43 bit。这些是各自模型条件下的换算，不是本仓库 Atkin–Morain 族的位宽收益。

**读法 B：给定 (bit, B1) 换成成功率与曲线数** `N = 1/f`（N 条时 miss = 36.8%）。D 不能单独给出 N，必须代进 ρ。
下表是 **stage-1 only** 口径（`rho.stage1_prob`）。注意 CLI 的默认口径不是这个：`ecm_prob.py predict/solve`
不给 `--B2` 时是 `B2 = 100·B1` 的 stage1+2，要复现下表需显式 `--B2 0`（或任何 ≤ B1 的值）；stage 2 的影响见 §6.8。

| (bit, B1) | D=6.41(param3)：f / N | D=20.90(param0)：f / N | N 比 |
|---|---|---|---|
| 20 / 256 | 21.463% / 4.7 | 30.870% / 3.2 | 1.439× |
| 40 / 256 | 0.0755% / 1324.8 | 0.1335% / 748.9 | 1.768× |
| 130 / 4.4e7 | 0.0327% / 3055.4 | 0.0394% / 2536.8 | 1.204× |

（D 用报告里显示的两位小数 6.41 / 20.90；代表值文件 `out/d_eff_representative.json` 里的完整精度是
6.4059 / 20.9021，换用完整精度对上表/N 比的影响 < 0.05%。）

**读法 C：D 也等于"省下多少 B1"**（固定 bit 与曲线预算，反解 B1；口径 **stage1+stage2，B2 = 100·B1**，
即 `ecm_prob.py solve` 的默认约定）：

| bit | N=100 条(param3→param0 所需 B1) | 省 |
|---|---|---|
| 20 | 8 → 6 | 1.30× |
| 40 | 192 → 145 | 1.32× |
| 130 | 1.117e9 → 8.23e8 | 1.36× |

**所以"D=20 表示 20 条曲线在某 B1 达到 xx%"是错位的读法**：曲线数与 D 是两个方向的解，正确用法是
给定 N 反解 B1（`ecm_prob.py solve --bit B --curves N --curve C`）。

### 6.6 不同参数化的 D 不能线性比较 —— 但存在"局部幂律"（2026-09-24 补）

f 对 D 的响应是**幂律**而非比例：`f ∝ D^e`，局部指数

$$e=\frac{\partial\ln f}{\partial\ln D}\approx\frac{1+\ln u}{\ln B_1},\qquad u=\frac{\ln p-\ln D}{\ln B_1}$$

**e 同时依赖位宽和 B1**，所以同一对 D 在不同工作点换来的曲线数完全不同。取 D 比 = 20.90/6.41 = **3.26×**
（本仓库实测口径，`tools/ecm_prob`；下表为 **stage-1 only**，即 `rho.stage1_prob` 口径）：

| 工作点 | u | N 比（D=6.41 ÷ D=20.9 所需曲线） | e 实测 | e 公式 |
|---|---|---|---|---|
| bit20 / B1=256 | 1.95 | **1.439×** | 0.307 | 0.301 |
| bit30 / B1=256 | 3.20 | **1.613×** | 0.404 | 0.390 |
| bit40 / B1=256 | 4.45 | **1.768×** | 0.482 | 0.450 |
| bit60 / B1=1e4 | 4.19 | **1.390×** | 0.278 | 0.264 |
| bit130 / B1=1e6 | 6.30 | **1.302×** | 0.223 | 0.206 |
| bit130 / B1=4.4e7 | 4.95 | **1.204×** | 0.157 | 0.148 |

（比值按 D=20.90/6.41 计算；换成代表值文件的完整精度 20.9021/6.4059，比值变化 < 0.05%。）

⇒ **"D=6 需要 3 倍曲线数"不成立**：3.26× 的 D 比最多换到 ~1.8× 曲线（bit40/B1=256 这种 e 最大处），
在 production 的 B1（1e6–1e8）下只有 ~1.2×。

同口径的**实测**旁证：B1=256、bit 15–40（`tools/stat/ecm_hitrate.ps1` 与
`tools/ecm_prob/out/measure_<bit>_256.json`，每点 65536 个采样素数），param0/param3 的单曲线成功率比
从 1.30×（bit15）升到 ~1.6–1.8×（bit30–40，大位宽处命中数只剩几十个、噪声 ±0.1–0.3pp），
与上表"e 随 u 增大"的趋势一致，且始终**远小于** D 的 3.26×。

反向（要"少 k 倍曲线"需要多大 D 比，= k^(1/e)）：

| 工作点 | e | 2× | 3.26× | 10× |
|---|---|---|---|---|
| bit20 / B1=256 | 0.301 | 10.0× | 50.7× | 2103× |
| bit40 / B1=256 | 0.450 | 4.67× | 13.9× | 167× |
| bit130 / B1=4.4e7 | 0.148 | 109× | 2990× | 5.9e6× |

**唯一可"线性"用的形式**是小范围线性化 `Δf/f ≈ e·ΔD/D`：在 ±20% 内误差 <1%（bit40/B1=256：
D/D0=0.8 时精确 0.8989 vs 幂律 0.9045，+0.63%），**但斜率 e 仍依赖 (bit, B1)**，跨工作点直接比 ΔD 无效。

### 6.7 跨来源与当前 Edwards 的口径警告

1. **D 的绝对值带口径，跨来源比较前先统一**：本仓库经验 `D_eff` 是 *stage-1-only + 点口径
   `p_ref = 2^(bit−0.5)`*（实测 Suyama ≈20.9；改用"素数集平均"口径为 20.7）；GMP-ECM 的
   `exp(3.134) = 22.97` 是 *stage-1+stage-2* 口径。差 ~10% 是口径差，不是矛盾。
2. **D 混了两部分**：`D = T × (Galois 额外光滑性)`，T 是已知挠子群（Z/12 vs Z/4 本身就差 3 倍）。
   本仓库报告里的 `extra = D/T` 列才是剥掉 T 后的可比量：param0 = 20.9/12 = **1.74**，
   param3 = 6.41/4 = **1.60**。谈"哪个族的 Galois 性质更好"要比 `extra`（或同口径的 D/T），
   直接比 D 会把"整数挠点的白送红利"和"拟合出的 Galois 因子"混在一起。
3. **当前 Edwards 的 48 倍标量改变了事件定义**：`tools/ecm_prob/` 对论文固定 Edwards 曲线使用 `lcm` 命中判定；实际 Atkin–Morain Stage 1 用 `48·lcm`，并在构造、末点转换和 Stage 2 有另外的因子发现机会。比较时应先统一目标素数分布、σ 取样、标量、`B1/B2` 与命中阶段；再分别报告单曲线命中率及完整单曲线成本。仅按挠群 T 或现有固定曲线 D 推出的成功率，不能作为当前实现的测量结果。

### 6.8 B2 怎么影响这一切（2026-09-24 补，B2 = 10000·B1 的统一估算）

`ecm_prob.py predict/solve` 现在与 `ecm_plot.py` 同名支持 `--b2-factor`（默认 100 = `model.B2_FACTOR`），
可直接算"统一 B2 = 10000·B1"的口径。bit130 / B1=1.1e8（B2=1.1e12）：

| | f_s1 | f_s12 | 阶段2贡献 |
|---|---|---|---|
| param0（D=20.90） | 0.0770% | **0.8046%** | 90.4% |
| param3（D=6.41） | 0.0648% | **0.6938%** | 90.7% |

三个结论：

1. **在表中 param0/param3 工作点，大 B2 几乎不稀释 D 的优势**：p 比从 `B2=100B1` 的 1.167× 只降到 `B2=10000B1` 的 **1.160×**
   （bit140: 1.176→1.170；bit150: 1.184→1.178，稀释 0.994–0.996×）。原因：`effN = p/D` 对
   stage 1 与 stage 2 **同时**生效，D 把两段的 u 一起平移。⇒ 在这组模型输入下，**param0/param3 的比值**对所取两档 B2 不敏感；不能推广为任意曲线族或任意 B2 都如此。
2. **B2 本身是更大的旋钮，且两种参数化同等受益**：B2 从 1.1e10 提到 1.1e12，概率 ×2.15
   （10% miss 所需曲线：param0 614→285、param3 717→331，比值仍是 1.16×）。
   ⇒ 这是"调 B2"的收益，不是"选参数化"的收益。代价：stage-2 工作量主项是 (B1,B2] 内素数个数，
   B2 涨 100× 时，**逐素数或直接点对模型**的候选素数数目约涨 **80×**（`≈100·(lnB1+4.6)/(lnB1+9.2)`，B1≈1e8）。本仓库的 NTT 多项式 Stage 2 还受 D、树形状、批次及内存预算影响，不能把这 80× 当作其时间预测。
3. **"固定曲线数下的 miss 比"会骗人**：B2=10000·B1 时 N=960 已远超 1/p（≈124），
   miss 比看似 2.922×，但两个 miss 都趋零（0.043% vs 0.125%），**Δpp 只剩 0.082pp**；
   同一对参数化在 `B2=100B1` 下是 1.674× / 1.845pp。⇒ 比较参数化请固定**目标 miss**
   （或直接读 1/p 比），不要用固定曲线数下的 miss 比。

> **工程换算（含 kernel 代价）**：把实测吞吐差算进来（param0 kernel 多两次满宽乘 ⇒ 吞吐 =
> param3 的 0.78–0.82，即每曲线成本 c ≈ 1.22–1.28），并让两者各自把 B1 调到最优（最小化 `c·B1/p`）：

| bit | param3 B1opt | param0 B1opt | 每成功耗时 param0/param3 |
|---|---|---|---|
| 130 | 3.27e6 | 2.82e6 | 1.007× |
| 140 | 6.85e6 | 5.91e6 | 1.014× |
| 150 | 1.39e7 | 1.24e7 | 1.020× |

⇒ **只看 stage-1 时间，param0 与 param3 打平略亏**（1.16–1.24× 的成功率优势打不过 1.25× 的 kernel 代价）；
把 stage-2 时间摊进来后 `((1.25+k)/(1+k))·(p3/p0)`：k=0.5 → param0 省 5.9%，k=1 → 省 9.3%。
param0 的确定收益在**存档能被 gmp-ecm `-param 0`/Prime95 直接续做 stage 2**、以及与 CPU 路径同曲线同 σ。

**复现命令**（均在 `tools/ecm_prob/`）：

```powershell
python ecm_prob.py predict --all --bit 130 --B1 1.1e8 --b2-factor 10000   # 统一 B2=10000*B1 全曲线表
python ecm_prob.py solve --bit 130 --B1 110e6 --curves 960 --curve suyama_s10 --b2-factor 10000
python ecm_prob.py solve --bit 130 --B1 110e6 --curves 960 --curve param3_s10 --b2-factor 10000
python ecm_prob.py solve --bit 130 --curves 100 --curve suyama_s10        # 读法 C：反解 B1
```

**当前有限域验证：** [GKL IV / AM 实验](../performance/ECM_FRIENDLY_IV.md)已用固定 sigma=1…100、15/20/25/30-bit 素数集合独立计算本仓库 AM 原始点阶，分别按 LCM 和 48LCM 统计 Stage1、理想单大素数 Stage2 及条件 D_eff。30-bit、B1=256、48LCM 下 AM 的 Stage1 拟合 D_eff 为 27.73。该数据有明确的有限种子和素数取样范围；真实复合模数上的构造因子、GCD=N 恢复、实际覆盖集合及大位宽生产 B1/B2 的完整单位时间收益仍需验证。固定论文曲线的旧 D_eff 与本次 AM 取样结果应分别引用。

---


## 7. GKL17 family (iv)、PrMers `-iv163` 与 GPU 论文的关系

### 7.1 家族性质与成功率证据

Gélin、Kleinjung、Lenstra（ISSAC 2017）专门研究 `a=−1` twisted Edwards 曲线
`−x²+y²=1−e⁴x²y²`。family (iv) 可写成
`e=(g−1/g)/2=(g²−1)/(2g)`（`g` 为非零有理参数）；等价的判别是 `e²+1` 为有理数平方。
该族在有理数域上的挠群仍是 `Z/2×Z/4`，阶为 8。令 `k_p=v₂(#E(F_p))`，即 `2^k_p` 是模 `p` **整个群阶**中的 2 次幂部分；论文 §2.3 的 `16/3` 是固定此族曲线、让良好约化素数 `p` 变化时 `k_p` 的平均值，不是对种子或起点阶取平均。一般 `a=−1, Z/2×Z/4` 族的相应值为 `14/3`，故 IV 增加 `2/3`。[Barbulescu 等的原始分析表 3](https://msp.org/obs/2013/1-1/obs-v1-n1-p04-p.pdf)进一步将 IV 分成 `p≡1 (mod 4)` 时 `17/3`、`p≡3 (mod 4)` 时 `5`，两类素数合起来为 `16/3`。对非负整数估值，还有等价的整除概率表达 `E[k_p]=Σ_{j≥1} Pr(2^j | #E(F_p))`。

`2^(16/3)≈40.32` 只等于 `2^k_p` 的**几何平均值**；相对一般族的比值为 `2^(2/3)≈1.59`。它不是群阶 2 部分的算术平均值，也不是“平均有 40 个有理挠点”或“每条曲线存在 32 阶点”。事实上，IV 的阶为 8 的有理挠群在好约化时只直接保证 `8 | #E(F_p)`；该族额外的 Galois 性质使好奇素数 `p>3` 下总有 `16 | #E(F_p)`，而 `p≡1 (mod 4)` 且 `g(g−1)(g+1)` 为模 `p` 二次剩余时有 `32 | #E(F_p)`（同文推论 3.5–3.6）。论文指出 IV 的 `16/3` 与一种有理挠群为 `Z/2×Z/8` 的 Edwards 族相同；这只说明**平均群阶 2 进估值相同**，不说明两族的挠群、起点阶或 ECM 命中率相同。

`ord(P mod p) | #E(F_p)`，其 2 进估值可以严格小于 `k_p`；ECM Stage 1 命中还须由所用标量消去这个**起点**。本仓库的经验 `D_eff` 是根据给定 `B1`、起点、标量和取样素数的 Stage 1 命中率 `f`，用 `f≈ρ_local((ln p−ln D_eff)/ln B1)` 反解的模型参数，综合了点阶的 2 部分、其他素因子及取样影响（见[拟合口径](../../tools/ecm_prob/README.md#d_eff-归一化方法)）。因此 `D_eff≠2^(16/3)`，`2^(2/3)` 也不是 `D_eff` 或成功率的已知比值；比较 IV 与 AM 应使用相同 `B1`、标量和起点取样，再看实测命中率及单位时间成本。

论文摘要中的约 **1%–2% 改进**来自其固定 ECM 软件、参数及 **15–63 位目标因子**上的曲线检出实验，不是点公式、GPU 内核或本仓库完整 Stage 1+2 的已测提速。论文 §4 还指出，选用的非挠起点在约半数测试素数下可表现为另一点的二倍，说明只看有理挠群阶不足以解释全部检出差异。详细实验条件与图表见[原论文](../paper/Parametrizations%20for%20Families%20of%20ECM-Friendly%20Curves.pdf)及[中文概述](../paper/ECM%20友好曲线与%20CANS%202020%20论文概述.md)。

AM 的 `Z/2×Z/8` 含 16 个有理挠点，精确阶 1、2、4、8 的数量分别为 1、3、4、8，最大点阶为 8；IV 的 `Z/2×Z/4` 对应数量为 1、3、4，最大点阶为 4。前者没有 16 或 32 阶有理点。AM 的有理挠群在好奇素数下保证 `16 | #E(F_p)`，不保证 3 整除群阶；生产标量中的额外乘 3 不能改写为有理三挠点。群阶整除性、给定起点的阶及标量是否消去该点是三个不同问题。

当前[有限种子实验](../performance/ECM_FRIENDLY_IV.md)在 30-bit 素数上测得 IV、AM、通用 Z2×Z4 的平均群阶 v2 分别为 5.3336、5.3363、4.6653；IV 对通用族的优势不能直接转写成对现有 AM 的优势。成功率及 D_eff 应连同 B1、标量、起点和取样口径引用。

### 7.2 `-iv163` 对应论文的具体构造

选项名中的 `iv` 指 GKL17 的 **family (iv)**。结合 PrMers [CLI 帮助](https://github.com/cherubrock-seb/PrMers/blob/ffd3399163736eaf423fd712e0c935b0210e43f1/src/io/CliParser.cpp#L146)中的 “family_iv_163” 与 “16/3 average v2”，`163` 可读作将论文 §2.3 的平均 `v₂(#E(F_p))=16/3` 省去斜杠的简写；源码没有另行定义 163 这个整数参数。`16/3` 是跨素数的平均 2 进估值，不是 163 位模数、163 条曲线、挠群阶 163，也不表示每条曲线都含有 32 阶有理点。

GKL17 表 2 的 family (iv) 行从辅助曲线 `C: y²=x³−x²−9x+9` 和 `Q=(5,8)` 开始。取 `mQ=(s,r)`，令 `t=(4s+4)/(r−4)`；表 1 第 4 行给出 `e=(t²+4t)/(t²−4)`、目标曲线 `d=−e⁴` 及其非挠起点。`m=1` 时 `t=6`、`e=15/8`、起点 `(X,Y)=(28/195,3152/3495)`；代入 `−X²+Y²=1−e⁴X²Y²` 成立，可用作公式核对样本。

[PrMers 辅助曲线点运算](https://github.com/cherubrock-seb/PrMers/blob/ffd3399163736eaf423fd712e0c935b0210e43f1/src/modes/RunEcmTwistedEdwards.cpp#L2291)与[构造分支](https://github.com/cherubrock-seb/PrMers/blob/ffd3399163736eaf423fd712e0c935b0210e43f1/src/modes/RunEcmTwistedEdwards.cpp#L2461)逐式对应这组公式，故可确认 **PrMers 的 `-iv163` 使用 GKL17 family (iv) 的这一参数化**。但 PrMers 从 `curve_seed` 映射到 `m∈[1,100]`，最多重试 128 次以避开不可用参数；这是有限取样和工程退化处理，不能说它逐条复现论文的 `C[1]…C[100]` 测试序列。[模式选择](https://github.com/cherubrock-seb/PrMers/blob/ffd3399163736eaf423fd712e0c935b0210e43f1/src/modes/RunEcmTwistedEdwards.cpp#L853)还受 `-torsion16`、`-notorsion` 与强制 sigma 的优先级约束。这里的 `curve_seed`、`m` 与本仓库 Edwards sigma 是三种不同身份；只比较其整数值不足以确定同一曲线和起点。

### 7.3 与当前 ECM-ELF 的实现边界

本仓库 [Edwards 构造](../../src/ecm/curve/edwards.h#L6)采用 Atkin–Morain 的 `a=1` 模型，由 sigma 推导 `d` 与起点；[Stage 1](STAGE1.md)和[Stage 2](STAGE2.md)按该构造重建同一曲线。生产 CLI 当前未提供 GKL17 family (iv) / PrMers `-iv163` 的构造入口。引入该族意味着定义新的外部构造身份、`seed→m→(a,d,P)` 映射及存档重建规则，并验证 `a=−1` 点运算与 Stage 2 接口；不能只把 `d` 或一个现有 sigma 改成新值。

当前对外构造身份按 `(curve, param, sigma)` 区分，`param` 仅用于 Montgomery；本仓库 Edwards 的逐 σ 映射和后端矩阵见 §2。[Stage 1 的能力与限制](STAGE1.md)及[Stage 2 输入合同](STAGE2.md#输入与表示)另列 Montgomery 各参数化的可运行范围。独立研究入口现已实现正 uint64 参数 `m` 直接取辅助点倍数，并具备 GMP a=−1 Stage1 与显式 a24 的 CUDA Stage2 接续，见[实验报告](../performance/ECM_FRIENDLY_IV.md)。该入口没有接入生产 sigma、save 或结果身份；生产 GPU Stage1 的 a=−1 内核尚未接入。

Wloka 等的 [GPU ECM 论文](../paper/Revisiting%20ECM%20on%20GPUs_2020-1265.pdf)引用 GKL17 来构造随机 `a=−1` 曲线，但没有披露与 PrMers `-iv163` 逐种子相同的映射；其 §5 测的是 128–448 位模数的**每秒曲线试验数**，明确未评估实际因子检出率。它的吞吐表不能验证 GKL17 所报的 1%–2% 选曲收益，也不能据此推定本仓库 NTT Stage 2 的收益。算法层面对照见[论文分析](../paper/Revisiting%20ECM%20on%20GPUs_与本仓库算法对照.md)。

**待验证：** 独立实验已验证 `m=1` 黄金值、模合数构造、不可逆分母、GMP Stage1 末点及显式 a24 的 GPU Stage2 接续。若接入生产，仍需验证独立构造身份下的 CPU/GPU Stage1、save/ckpt 重建及跨阶段点一致性；更大目标因子与生产 `B1/B2` 下的完整单位时间因子产出尚未标定。

---

## 8. 参考

- GMP-ECM 源码：`README`（§2 光滑性 NOTE、§6 参数化）、`rho.c`（`ECM_EXTRA_SMOOTHNESS=3.134`、Dickman ρ 实现）、`parametrizations.c`（param 0–3 公式）、`ecm.c`（`EXTRA_SMOOTHNESS_*` 与 `print_expcurves`）。
- [EFD: Montgomery XZ](https://hyperelliptic.org/EFD/g1p/auto-montgom-xz.html)、[EFD: twisted Edwards extended (a=−1)](https://hyperelliptic.org/EFD/g1p/auto-twisted-extended-1.html)、[EFD: twisted Edwards extended](https://hyperelliptic.org/EFD/g1p/auto-twisted-extended.html)。
- Bernstein, Birkner, Lange, Peters, *ECM using Edwards curves*, [ePrint 2008/016](https://eprint.iacr.org/2008/016)（Math. Comp. 82 (2013)）。
- Wloka, Richter-Brockmann, Stahlke, Kleinjung, Priplata, Güneysu, *Revisiting ECM on GPUs*, [ePrint 2020/1265](https://eprint.iacr.org/2020/1265)（CANS 2020）。
- Gélin, Kleinjung, Lenstra, *Parametrizations for Families of ECM-friendly curves*, [ePrint 2016/1092](https://eprint.iacr.org/2016/1092)（ISSAC 2017）。
- [PrMers](https://github.com/cherubrock-seb/PrMers)：本节引用固定提交 [`ffd3399` 的 `RunEcmTwistedEdwards.cpp`](https://github.com/cherubrock-seb/PrMers/blob/ffd3399163736eaf423fd712e0c935b0210e43f1/src/modes/RunEcmTwistedEdwards.cpp) 与 [CLI 帮助](https://github.com/cherubrock-seb/PrMers/blob/ffd3399163736eaf423fd712e0c935b0210e43f1/src/io/CliParser.cpp#L146)；与 Prime95 31.06b01 是不同参考对象。
- Barbulescu, Bos, Bouvier, Kleinjung, Montgomery, *Finding ECM-Friendly Curves through a Study of Galois Properties*, [ePrint 2012/070](https://eprint.iacr.org/2012/070)。
- Kruppa, *Optimising the Elliptic Curve Method of Factoring*, PhD thesis (ρ 函数的 smoothness 模型来源)。
