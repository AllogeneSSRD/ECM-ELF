# Edwards ECM：构造、坐标公式、NAF 与 Prime95 格式

固定参考资料。来源：`ECM_EDWARDS_STAGE1.md §§1–5、15.2 的数学部分`。保留数学推导、第三方源码分析及其条件；源码行号针对原文研究的本地副本，第三方上游可能已有变化。本文不声明本仓库当前支持、默认值或性能。涉及实验数字时只按原文条件理解；当前实现见 [架构](../architecture/OVERVIEW.md)，当前计时口径见 [性能](../performance/STAGE2.md)。

## 1. 曲线构造（Atkin-Morain，sigma_type=0）

Prime95 `choose_atkin_morain`（`ecm.cpp:4256`），参考论文 §7（ePrint 2008/016）：

1. 在 Weierstrass 曲线 `T² = S³ − 8S − 32` 上取点 `(12, 40)`。
2. 标量乘 `(s, t) = σ · (12, 40)`（σ 为 uint64）。
3. 归一化 `s = sN/sD`、`t = tN/tD`（tD = sD）。
4. 推导 α、β、d（**等价域运算，避免中间归一化**）：
   - `α = (s−9) / (t+s+16)`
   - `β = 2α(4α+1) / (8α²−1)`
   - `d = (2(2β−1)² − 1) / (2β−1)⁴`
5. 基点 P = (x1, y1)：
   - `x = (2β−1)(4β−3) / (6β−5) = (2β−1)(2(2β−1)−1) / (3(2β−1)−1)`
   - `y = (2β−1)(t²+50t−2s³+27s²−104) / ((t+3s−2)(t+s+16))`

结果：plain Edwards 曲线 `x²+y² = 1 + d·x²y²`（a=1），Z/2×Z/8 扭子群（16 整除 #E）。

> Prime95 为省逆采用 `sN/sD`、`dN/dD` 形式（`ecm.cpp:4356-4433` 的 `#else` 分支），只做 1 次模逆（`inv(sD·dD)`）。用 `mpz` 实现时可直接按上式归一化（多次 `mpz_invert`），正确性优先。


## 2. 扩展 Edwards 域公式（a=1）

扩展坐标 `(X:Y:Z:T)`，`x=X/Z, y=Y/Z, T=XY/Z`。mod N（合数）。

标准 Edwards 加法律（曲线 `x²+y²=1+dx²y²`，恒等点 `(0,1)`）：
`x3 = (x1y2+x2y1)/(1+dx1x2y1y2)`，`y3 = (y1y2−x1x2)/(1−dx1x2y1y2)`。

- **统一加法**（HWCD08 通用 a 公式，a=1，**9M+1·d**；若乘 d 与一般乘法同价则为 10M）：
  ```
  A = X1·X2;  B = Y1·Y2;  C = d·T1·T2;  D = Z1·Z2
  E = (X1+Y1)(X2+Y2) − A − B
  F = D − C;  G = D + C;  H = B − A        # a=1
  X3 = E·F;  Y3 = G·H;  T3 = E·H;  Z3 = F·G
  ```
- **倍点**（`dbl-2008-hwcd`，a=1，**4M+4S**）：
  ```
  A = X1²; B = Y1²; C = 2Z1²; D = A (=aA, a=1)
  E = (X1+Y1)² − A − B;  G = D+B;  F = G−C;  H = D−B
  X3 = E·F;  Y3 = G·H;  T3 = E·H;  Z3 = F·G
  ```
  （倍点不含 d：用曲线方程 `x²+y²=1+dx²y²` 把 `1+dx²y²` 代成 `x²+y²`，故无需 d。）

> ⚠️ 不要用 `add-2008-hwcd-4`（`C=2T1Z2, D=2T2Z1` 那组）：它给出 `x3=(x1y1+x2y2)/(y1y2−x1x2)`，是**错误**的加法律（会把 `[2]P` 算成曲线外点）。Prime95 的 `ed_add`（`ecm.cpp:2016`）实际用的是 Atnashev eprint 2021/1061 的替代公式（替代坐标 `(XZ,YZ,XY,Z²)`，`x3=(x1y1+x2y2)/(y1y2+x1x2)`），与标准加法律等价（同恒等点 `(0,1)`、同曲线），故 `[s]P` 一致，仅投影缩放不同。limb 实现用标准 9M 公式即可。


### 2.1 Prime95 倍点算法编号与 FFT 条件

研究版本为 `.refactor/p95v3106b01.source`。`undoc.txt:367` 列出 `EdDblAlgo5`、`EdDblAlgo6`，但同版 `ecm.cpp:2143` 明确说算法 6 已废弃；`ed_dbl` 实际按安全余量选择 5、5a、7 或 4。旧配置说明不能证明源码还执行算法 6。

源码中的算法 4 和 5 都用 `E=2XY`；5 通过 `gwmulmuladd5` 合并计算 `X²+Y²`，依赖 FFT 复合运算及额外安全位。将多项式展开为独立模运算时，可取 `A=X², B=Y², C=2Z², E=2XY, G=A+B, F=G−C, H=A−B`，再按上面的四个输出乘积计算；这与 HWCD 的 `E=(X+Y)²−A−B` 相等，但完整扩展倍点由 4M+4S 变成 5M+3S。两者投影尺度可以选成相同，不能从 FFT 的变换次数直接推出 limb/CGBN 的速度。

当下一步只读取 X/Y/Z 做倍点时，可以不计算当前输出的 T；加法前和完整扩展坐标边界仍须满足 XY=ZT。这分别减少一次乘法，不能省略恢复或外部交接所需的坐标不变量。

本文只保留第三方版本与代数依据，不声明当前 GPU 采用哪个窗口或倍点实现。

## 3. NAF 标量乘法

二进制与窗口 NAF 都将标量写成 `S = Σ dᵢ·2ⁱ`。二进制数字为 0 或 1；宽度 w 的 NAF 数字为 0 或正负奇数，满足 `|dᵢ| < 2^(w−1)`，相邻非零位置至少相隔 w 列。一列是一个幂次位置，含零列；列数与非零数字个数是两个不同量。高位带符号重编码及进位会改变总列数，不能把它固定等同于二进制位数，也不能用二进制位数除以 w。

例如 `15 = 8+4+2+1 = 16−1`：二进制有 4 个位置，宽度 2 的 NAF 有 5 列，其中只有 2 列非零。对从恒等点开始、逐列倍点再按数字加点的扫描，NAF 可以用更多列换取更少的加点。更宽窗口还可用最高位的较大奇数数字缩短表示，例如宽度 6 允许数字 15，此标量只需一列，并从字典取 `[15]P`。字典准备有独立成本，列数本身不能代表两种算法的总运算量。

- 指数 `s = 48 · lcm(1..B1)`（`48 = lcm(12,16)`，见 `ecm_calc_exp` 初始 `g=48`；screen.log 实测 B1=1e6 指数长 1442105 bit = ceil(log2(48·lcm(1..1e6)))，与 48·lcm 精确一致）。
- 转 **w-NAF**（有符号、无相邻非零位），标准宽度 w 的 w-NAF 非零密度约 1/(w+1)；字典参数若采用不同窗口编号，须先换算 w。
- **字典**：预计算奇数倍 `3P, 5P, …, (2^(w−1)−1)P`，批量模逆归一化（`dict_normalize`，1 次逆归一化整表）。
- 逐 bit：1 次倍点 +（非零位）1 次加法。
- 命中判定：在非奇异模素数 p 曲线上，`[s]P = 恒等点 (0,1)` ⇔ `ord(P)` 整除 s；s 含额外挠子群乘子时，不可把此条件直接写成 lcm 的 powersmooth 条件。

> 窗口 w 只影响性能不影响结果，交叉验证用任意 w 均可；Prime95 默认 DictionaryMemory=256MB → w≈14（4675 字典项，见 screen.log）。


## 4. 曲线构造 + 标量乘的 stage-1 结果（交叉验证口径）

stage-1 完成后，Edwards 点 `[s]P=(e.x:e.y:e.z)` 转 Montgomery（`ed_to_Montgomery`），得 `(Qx:Qz)`：

- Montgomery ↔ Edwards 双有理映射：`u=(1+y)/(1−y), v=u/x`；Montgomery `A = 2(1+d)/(1−d) = 2(a+d)/(a−d)`（a=1），`B=4/(1−d)`。
- Prime95 存 `Ad4 = (dD−dN)/dD = 4/(A+2)`（见 `choose_atkin_morain` 与 `ed_to_Montgomery`）。

交叉验证：同 σ（如 20260922）、同 N（M347=2³⁴⁷−1）、同 B1=1000000，计算 `[s]P` → `(Qx,Qz)`，与存档 `e0000347` 中的 `Qx_binary`/`Qz_binary` 比对。

> **关键**：`(Qx,Qz)` 的**投影缩放不唯一**。不同加法律（标准 9M vs Atnashev 替代式）产生同仿射点但不同 Z，故原始 `Qx/Qz` 字节不逐位相等；**投影无关的不变量是 `u = Qx/Qz = (1+y)/(1−y)`**（Montgomery x 坐标），以及 `gcd(Qz,N)`（因子判定）。交叉验证以 `u` 一致 + `gcd(Qz,N)` 一致为准。


### 4.1 在有限素域计算群阶与点阶

设 `p>3` 为已证明素数，`d≠0,1`，采用曲线的光滑射影模型。上述映射是同一素域上的双有理等价，因而保持完整曲线群阶和非例外基点的阶。保留 `B=4/(1-d)` 后，令 `X=B·u`、`Y=B²·v`，则

`Y² = X³ + (A·B)X² + B²X`，即PARI模型 `[a1,a2,a3,a4,a6]=[0,A·B,0,B²,0]`。

对一般Edwards曲线，将 `Bv²` 直接改为 `v²` 可能改变二次扭转身份，群阶变为 `2p+2−#E`。本节Atkin-Morain族还有额外性质：令 `h=2β−1`，则 `1-d=(h²-1)²/h⁴`，故 `B=(2h²/(h²-1))²` 在非退化素域上是平方；该族可通过相应平方根缩放v而令B=1。保留B的Weierstrass映射无需另选平方根，也直接保持给定基点身份。同sigma的Suyama PARAM0、PARAM3和Atkin-Morain不定义同一曲线，不能共用其群阶。

PARI的 `ellinit` 接受Weierstrass系数及有限素域；`ellcard` 计算 `#E(F_p)`，`ellorder` 计算基点P的阶。群阶需满足精确Hasse条件 `(#E−p−1)²≤4p`。若候选点阶为m，验证 `[m]P=O`，且对每个不同素因子q验证 `[m/q]P≠O`，可确认其最小性；同时核对 `m | #E` 与分解乘积。模数或曲线退化时不能套用这些结论。

`(0,1)` 对应Weierstrass无穷远点，`(0,-1)` 对应 `(0,0)`；其余映射必须检查分母。直接枚举仿射Edwards方程可能漏掉光滑模型上位于无穷远处的点，不能把这样的枚举数直接当作完整群阶。

Stage1若使用 `48·lcm(1,…,B1)`，应对 `ord(P)/gcd(ord(P),48)` 推导幂光滑界限。群阶整除所用标量是充分条件，具体基点被消去的充要条件是点阶整除该标量；单素数Stage2只额外消去一个剩余素数的一次幂，不能忽略重复素因子。

参考：[PARI有限域椭圆曲线接口](https://pari.math.u-bordeaux.fr/dochtml/html-stable/Elliptic_curves.html)、[Edwards/Montgomery映射](https://www.hyperelliptic.org/tanja/teaching/cryptoI11/exercises/exercise-12.pdf)。本节采用 `v=u/x`、`B=4/(1-d)`；文献中 `v=2u/x`、`B=1/(1-d)` 是等价的坐标缩放。

## 5. Prime95 存档字节格式（ECM_VERSION=6）

全部 **小端**。实测 `e0000347`（204 字节）如下：

```
偏移    字段            类型        实测(M347) 说明
0x00    magic          u32          0x1725bcd9
0x04    version        u32          6
0x08    k              double       1.0
0x10    b              u32          2
0x14    n              u32          347          (M347 = 2^347 - 1)
0x18    c              i32          -1
0x1c    stage          char[10]     "C1S2"+6×0
0x26    2×pad          byte         0,0
0x28    pct_complete   double       0.0
0x30    checksum       u32          (write_checksum 回填到 offset 48)
0x34    curve          u32          1
0x38    average_B2     u64          0           (不计入 checksum)
0x40    state          u32          2           (ECM_STATE_MIDSTAGE)
0x44    sigma          u64          20260922    (不计入 checksum)
0x4c    B              u64          1000000     (B1)
0x54    C              u64          100000000   (B2)
0x5c    sigma_type     u32          0           (0=Atkin-Morain Edwards)
0x60    montg_stage1   u32          1           (见下方 ⚠️)
-- MIDSTAGE 状态数据 --
0x64    Qx_binary      giant        len=11 + 44B
0x94    Qz flag        i32          1
0x98    Qz_binary      giant        len=11 + 44B
0xc8    gg flag        i32          0
```

**字段读取顺序**（`ecm_restore`，`ecm.cpp:6445-6465` 与 `ecm_save` 一致）：`curve, average_B2, state, sigma, B, C, sigma_type, montg_stage1`。

**giant 编码**（`write_giant`/`commonc.c:4548`）：4 字节 limb 数（u32）+ `len×4` 字节 limb 数组（**32-bit 小端**，即数值 mod N 的标准二进制）。`gwnum` 经 `gwtogiant` 转此格式。

**checksum**（`write_checksum`）：把各 `write_uint32/64`（sum≠NULL 的）与 giant（len+Σlimb）累加（u32 溢出回绕），回填到 **offset 48**（`CHECKSUM_OFFSET`）。`average_B2`、`sigma` 以 `sum=NULL` 写入，不计入。

**footer**（可选）：`"MOREINFOJSONDATA"`(16B) + chunk_size + version=1 + crc32 + JSON，仅当 `morejsoninfo` 非空。

**state 枚举**：`0=STAGE1_INIT, 1=STAGE1, 2=MIDSTAGE, 3=STAGE2, 4=GCD`。
- `STAGE1`（Edwards, montg_stage1=0）：`stage1_start_prime, stage1_exp_buffer_size, stage1_bitnum, NAF_dictionary_size, dict_start.x/y, e.x/e.y/e.z`。
- `MIDSTAGE`（stage1 完成）：`Qx_binary` + 可选 `Qz_binary` + 可选 `gg_binary`。


## radix 2^52 的梅森折叠与对称平方

`N = 2^k−1 ⇒ 2^k ≡ 1`。radix 2^52、`n = ceil(k/52)`、`sh = 52n − k ∈ [0,52)`，于是

```
B^n = 2^(52n) = 2^(k+sh) = 2^sh        2^(52c) = 2^(52(c−n)+sh)  (c ≥ n)
```

即**第 c 列（c ≥ n）折回第 c−n 列、左移 sh 位**：一个移位加一次加法，替代整行 `m_i·N` 归约。
每模乘 madds 从 `n(4n+3)` 降到 `2n²`，且**不需要 Montgomery 常数**（域就是普通域，`one = 1`）。

折叠后必须继续排空超出 k 位的进位，并规范化到 `[0,N)`。仅将读回值再次 mod N 后比较，不能检验原始 limb 的规范性。对称平方只计算上三角和对角项：一般乘积约 `2n²` 个 IFMA low/high 部分积，平方约 `n(n−1)+2n`；这些是该 radix 拆分的算术条数，不是硬件周期数。
