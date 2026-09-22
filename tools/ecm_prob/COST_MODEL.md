# ECM Stage-1 成本模型

给定 B1 与曲线，计算 stage-1（曲线初始化 + 快速幂）的运算量，按**点运算层**（倍点/加法次数）与**域运算层**（M/S/A/Sub/I/D）两级输出，覆盖四种实现方案。

## 0. 四种方案的本质差异

| 方案 | 曲线形式 | 加法链 | 算术单位 | 代码位置 |
|---|---|---|---|---|
| 论文理论 | Montgomery / Edwards | 梯形 / 滑动窗口 | 字段乘法 M/S | EFD 显式公式 |
| Python（本项目 `ecmath.py`） | Montgomery / Edwards | 梯形 / double-and-add | Python 大整数 | `tools/ecm_prob/ecmath.py` |
| gmp-ecm（CPU，param0/1/2/3） | Montgomery | **逐素数 PRAC** | GMP `mpn` limb（ADD=6, DUP=5 模乘） | `.refactor/ecm/ecm.c` |
| gmp-ecm（GPU=本项目） | Montgomery | 单标量梯形 | 32-bit limb CIOS | `.refactor/ecm/cgbn_stage1.cu`、`kernels/opencl/ecm_stage1.cl` |
| prime95 | Montgomery(suyama) / Edwards | PRAC / NAF | **gwnum FFT**（dbl=10/add=12 变换） | `D:\code\GIMPS\p95v3106b01.source\ecm.cpp` |

**三个关键差异**：① 加法链（PRAC vs 梯形 vs NAF）；② 曲线形式（Montgomery vs Edwards）；③ 算术单位（limb 乘法 vs FFT 变换）。

## 1. 加法链算法原理

### 1.1 PRAC（Lucas 链，Montgomery 1983）

目标是给每个素数 p 找一条**加法链**（从 1 通过"倍点"与"差分加法"到达 p），使链上运算次数最小。因为 Montgomery 只有 XZ 坐标、只能做**差分加法**（需已知两点之差），所以用 Lucas 链：若已知 `V_r` 与 `V_{n-r}`，则 `V_n = V_r·V_{n-r} − V_{n-2r}`。

GMP-ECM `lucas_cost(n, v)` 用 9 条规则（`ecm.c:309-365`）递归收缩 `(d, e)` 对，累计**倍点（duplicate）**与**差分加法（add3）**次数：

| 条件 | 收缩 | 成本 |
|---|---|---|
| 1: `d−e ≤ e/4` 且 `d+e ≡ 0 mod 3` | `d=(2d−e)/3, e=(e−d)/2` | +3 加 |
| 2: `d−e ≤ e/4` 且 `d−e ≡ 0 mod 6` | `d=(d−e)/2` | +1 加 +1 倍 |
| 3: `(d+3)/4 ≤ e` | `d=d−e` | +1 加 |
| 4: `d+e` 偶 | `d=(d−e)/2` | +1 加 +1 倍 |
| 5: `d` 偶 | `d=d/2` | +1 加 +1 倍 |
| 6: `d ≡ 0 mod 3` | `d=d/3−e` | +3 加 +1 倍 |
| 7: `d+e ≡ 0 mod 3` | `d=(d−2e)/3` | +3 加 +1 倍 |
| 8: `d−e ≡ 0 mod 3` | `d=(d−e)/3` | +3 加 +1 倍 |
| 9: 其余（e 偶） | `e=e/2` | +1 加 +1 倍 |

初始 `v` 取黄金比倒数 `1/φ≈0.618`（及其连分数变体 `val[0..9]`），`prac` 试多个 v 取最省。**平均 ~1.32 次差分加法/bit**。素数 2、3 单独处理（2^k → k 次倍点；3^k → k 次倍点 + k 次加法）。

### 1.2 梯形（Montgomery ladder，融合差分）

对单个标量 s（= lcm(1..B1)），从左到右扫 s 的每个 bit，每 bit 做**一次融合的差分加+倍**（`ladd-1987-m-3`）。本项目 `double_add_v2`（`kernels/opencl/ecm_stage1.cl`）每 bit：

```
AA = (u+q)², BB = (u-q)²            # 2S，同时服务倍点与加法
X(2P) = AA·BB                        # 1M
K = AA−BB                            # 1Sub
Z(2P) = K·(BB + d·K)                 # 1D(×d) + 1A + 1M
…（差分加法：2M + 2S + 4A + 4Sub + 1D(×2)）
```

共 **4M + 4S + 4A + 4Sub + 2D**/bit。**省 M 的关键**：`AA`、`BB` 两个平方被倍点（X=AA·BB）和加法（差分）**复用**，比"分开的 duplicate(3M+2S) + add3(4M+2S) = 7M+4S"少 ~3M。

### 1.3 NAF（Non-Adjacent Form，Prime95 Edwards）

把指数 s 转成**有符号二进制 NAF**（无相邻非零位），非零密度 ≈ **1/3**（vs 普通二进制的 1/2）。预先用**字典**存奇数倍 `P, 3P, 5P, …, (2^{w-1}−1)P`，逐 bit 只做 1 次倍点 +（遇到非零位时）1 次加法。Prime95 把 s 按缓冲分块（`ecm_calc_exp`），NAF 字典用批量模逆归一化（`dict_normalize`）。

## 2. 域运算层（每次点运算的 M/S/A/Sub/I/D）

| 方案 | 倍点 | 差分加法 |
|---|---|---|
| 论文理论（EFD） | 2M+2S+1D（×a24） | 4M+2S |
| gmp-ecm（param0，`duplicate`/`add3`） | **3M**+2S+2A+2Sub（×a24 是满 M） | 4M+2S+3A+3Sub |
| 本项目 param3（融合/bit） | — | 4M+4S+4A+4Sub+2D |
| prime95（FFT） | 10 变换 | 12 变换 |

Edwards（a=1 射影）：倍点 3M+4S+1A，统一加法 8M（a=−1 扩展）；prime95 用 eprint 2021/1061 的 FFT 融合算法（algorithm 5/5a/7），`gwmulmuladd5` 等融合指令进一步减少变换数。

## 3. 曲线初始化（每条一次，M/S/A/Sub/I/D）

| 参数化 | 成本 | 说明 |
|---|---|---|
| Suyama（param0） | 10M+4S+**1I** | `u=s²−5,v=4s; A=(v−u)³(3u+v)/(4u³v)−2; x0=u³/v³`，含一次模逆 |
| param1（batch square） | 2M+1S | `d=s²/2⁶⁴; A=4d−2; x0=2`（2⁶⁴ 逆预计算一次） |
| param2（6-挠） | **2I** + 完整标量乘 | `P=s·(−3:3:1)` on `y²=x³+36`，再推导 A、x3 |
| param3（batch 32-bit d） | 1M | `d=s/2³²; A=4d−2; x0=2`（最省） |
| Edwards（论文/本项目） | 0 | `(a,d)` 小整数直接给定，基点小，无逐曲线逆 |
| prime95 Suyama（choose12） | 1I 量级 | 同 Suyama，gwnum 域 |
| prime95 Edwards（Atkin-Morain） | 标量乘 | `T²=S³−8S−32` 上算 `(s,t)` 倍数 + 曲线推导 |

**要点**：param3 初始化最省（1M、x0=2 固定、a24=d 为 32-bit），这是 GPU/本项目选它的核心理由；Suyama 需 1 次模逆；param2 需完整标量乘。

## 4. 成本合计示例（B1=1e6）

`python ecm_cost.py --B1 1000000 --curve suyama_s10`（s=1.44M bits，78498 素数，78734 素数幂；PRAC 273718 倍 + 2082156 加）：

| 方案 | 加法链 | 总模乘 M | 说明 |
|---|---|---|---|
| gmp-ecm（param0，PRAC） | 0.27M 倍 + 2.08M 加 | **9.15M** | 分开 duplicate+add3 |
| 本项目 param3（梯形） | 1.44M 融合 | **5.77M** | 复用平方，省 ~37% M |
| prime95（PRAC） | 同 gmp-ecm | 27.7M **FFT 变换** | 单位不同，不可直接比 |

**结论**：本项目 param3 的融合梯形在"域模乘数"上比 gmp-ecm 的逐素数 PRAC 省 ~37%，尽管总点运算次数更多（2.88M vs 2.36M）——因融合消除了重复平方；prime95 用 FFT，成本以变换数计，与 limb 乘法不可直接换算，但对 Mersenne/FFT 数（2ⁿ±1）而言单次变换吞吐极高。

## 5. 工具与实现

- **C++ 引擎** `cost_engine.cpp`（uint64 + Eratosthenes 筛，无 GMP）：输入 B1 → 输出 `{s_bits, n_primes, n_powers, prac_dbl, prac_add, ...}`。PRAC 的 `lucas_chain_counts` 改写为返回 `(倍点, 差分加法)` 独立计数（修掉了早期 Python 版的死循环 bug——`e//=2` 对 e=0 不回环；2/3 单独处理）。
- **Python CLI** `ecm_cost.py`：subprocess 调引擎拿点运算计数，结合静态域公式表格式化输出。
- 编译：`cl /O2 /EHsc cost_engine.cpp`（MSVC，VS 18 Community）。

## 参考

- Montgomery, *Speeding the Pollard and elliptic curve methods of factorization* (1987)：PRAC/Lucas 链、XZ 梯形。
- [EFD](https://hyperelliptic.org/EFD/g1p/auto-montgom-xz.html)：`dbl-1987-m-3`、`dadd-1987-m-3`、`ladd-1987-m-3`。
- GMP-ECM `.refactor/ecm/ecm.c`（`lucas_cost`/`prac`/`duplicate`/`add3`）、`parametrizations.c`。
- Prime95 `D:\code\GIMPS\p95v3106b01.source\ecm.cpp`（`lucas_cost`/`ell_mul`/`ed_dbl`/`ed_add`/`init_curve`）；Atnashev et al., *Edwards curves and FFT-based multiplication* (eprint 2021/1061)。
- 本项目 `kernels/opencl/ecm_stage1.cl`（`double_add_v2`）、`kernels/opencl/mont_mul/mont_mul_priv_opt.cl`（CIOS）。
