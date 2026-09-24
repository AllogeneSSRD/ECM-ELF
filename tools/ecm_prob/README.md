# ecm_prob — ECM 参数化概率定量工具

Windows 可用的 ECM 参数化成功概率定量分析工具。分两阶段：

- **阶段 A（预测）**：`rho.py` 精确移植 GMP-ECM `rho.c`（Dickman-ρ + `ECM_EXTRA_SMOOTHNESS`），
  `model.py` 在其上叠加"曲线 → 有效除子 D → 成功率/期望曲线数"的模型层。
- **阶段 B（经验）**：`ecmath.py` 纯 Python 实现 Edwards(a=1) / Montgomery(XZ) / Weierstrass 群律，
  对穷举素数集做 stage-1 命中测量，`model.py` 反推有效除子 D。

代码分三层：**库（6 个，可导入，无 CLI）** / **用户 CLI（3 个，`ecm_` 前缀）** / **校验（`tests/`）**。

## 目录结构

```
ecm_prob/
  # ---- 库（可导入，无 CLI）----
  rho.py            Dickman-ρ + stage1/stage2 概率数学核心（唯一底层模型来源）
  model.py          模型层：有效除子 D 三口径 + 预测 + 四元组反解 + D_eff 拟合 + GMP 推荐表
  curves.py         曲线清单：Montgomery param0/1/2/3 + Edwards 4 条 + p-1/p+1
  ecmath.py         纯 Python 点运算（Edwards/Montgomery/Weierstrass）+ [s]P 标量乘
  estimates.py      论文 §9.3 五类朴素估计
  data.py           素数集生成/缓存/加载 + 经验命中测量
  # ---- 用户 CLI ----
  ecm_prob.py       预测 / 四元组反解 / GMP 推荐表（subcommand: predict | solve | gmp-table）
  ecm_sweep.py      素数生成 / 经验扫掠 / 汇总报告（subcommand: primes | sweep | report）
  ecm_plot.py       绘图（subcommand: empirical | predict | list）；曲线登记表 + 暖/冷配色见文件头
  check_plot_colors.py  配色自检：Montgomery 暖 / Edwards 冷（HSV 色相）+ 同族两两 RGB 距离 >=60
  ecm_cost.py       stage-1 成本（点运算层 + 域运算层，调 cost_engine.exe）
  # ---- 成本模型 ----
  cost_engine.cpp   C++ PRAC 引擎（uint64 + 筛，无 GMP；cl /O2 编译 -> cost_engine.exe）
  COST_MODEL.md     成本模型文档（PRAC/梯形/NAF 算法 + 四方案对照）
  # ---- 校验 ----
  tests/test_golden.py   论文 §9.1 黄金数字对拍
  tests/cross_check.py   与 PARI/GP 的点阶交叉校验
  tests/gp_check.gp      gp 交叉校验脚本
  tests/bench.py         每素数耗时基准
  data/primes/       bits{b}.bin + manifest.json
  out/               measure*.json / report.md / summary.csv / plots/
```

## 快速开始

```powershell
cd D:\code\MPA-OpenCl\tools\ecm_prob

# 校验
python tests/test_golden.py        # 对拍论文 §9.1 四条 Edwards 曲线（20-bit/B1=256）
python estimates.py                # 对拍论文 §9.4 五类估计
python check_plot_colors.py        # 绘图配色约定自检（退出码 0=合规 / 1=违规）
#   已知偏差：test_golden 四条曲线一致比论文低 0.03–0.18pp（Z/12: 12404 vs 12467）。
#   在改动前的提交 edb717a 上复核结果相同 ⇒ pre-existing，不是回归；精确计数比对故意保留，
#   因为真回归的表现是"偏差突然变大"或"只剩一条曲线掉队"。

# 经验阶段（生成素数 -> 扫掠 -> 汇总/绘图）
#   素数集规模上限：bit <= 30 穷举；**bit >= 31 只生成 65536 个**（区间内 64 个均匀窗口各取
#   1024 个，秒级完成；穷举 bit31 会是 ~420 MB）。--count/--exhaustive-max-bit/--windows 可调。
#   位置参数是**位宽列表**，区间要写 `31-40`（`sweep 31 40` 只算这两个 bit，不是区间！）。
python ecm_sweep.py primes         # 生成/复用 15–30（幂等，sha256 校验）
python ecm_sweep.py primes 31-40 --count 65536      # 31+ 自动走采样（每个 512 KB，秒级）
python ecm_sweep.py sweep          # 跨位宽测量（默认 bit 15–30，B1=256）
python ecm_sweep.py sweep --B1 1e5                   # 换 B1；结果存 out/measure_<bit>_<B1>.json
python ecm_sweep.py sweep 20 22 --B1 1e4 --force     # 离散 bit 列表；--force 强制重算
python ecm_sweep.py sweep 32-39 --B1 256             # 区间写法（约 9 分钟/bit，单线程 Python）
python ecm_sweep.py report 20 256  # 生成 out/report.md + summary.csv
python ecm_plot.py empirical       # 经验图（bit 范围自动发现，当前数据 15–40 全覆盖）
python ecm_plot.py list            # 打印曲线登记表 + 各 B1 的可用 bit
python ecm_plot.py empirical --bits 15-30 --series suyama_s10,edwards_Z2xZ8

# 预测阶段
#   B2 口径三个 CLI 一致：默认 B2 = 100*B1（= model.B2_FACTOR）；--b2-factor 改比例
#   （例如 10000 ⇒ B2 = 10000*B1），--B2 固定绝对值（优先，同时给 --b2-factor 会提示）。
python ecm_prob.py predict --all --bit 25 --B1 256            # 全部曲线表（stage1 vs stage1+stage2）
python ecm_prob.py predict --curve suyama_s10 --bit 30 --B1 1024 --B2 25600
python ecm_prob.py predict --all --bit 130 --B1 1.1e8 --b2-factor 10000   # 统一 B2=10000*B1 对比
python ecm_prob.py solve --bit 130 --B1 44e6 --curves 960 --curve param3_s10
python ecm_prob.py solve --bit 130 --B1 110e6 --curves 960 --curve suyama_s10 --b2-factor 10000
python ecm_prob.py solve --bit 130 --curves 960              # 反解 B1
python ecm_prob.py gmp-table 30 35 40                         # GMP-ECM 推荐表
python ecm_plot.py predict --B1 256 --bit 40                  # 预测图（stage1 vs stage1+stage2）
#   （ecm_plot.py 同一个 --b2-factor 名字，语义相同）
python ecm_plot.py predict --B1 256 --mix-empirical           # 叠加 measure_*.json 的实测点
python ecm_plot.py predict --B1 1e6 --bit 60 --series suyama_s10,param3_s10

# 成本模型（stage-1 运算量）
python ecm_cost.py --B1 1000000 --curve suyama_s10   # 点运算层 + 域运算层(M/S/A/Sub/I/D)
```

依赖：仅 Python 标准库 + `numpy`/`matplotlib`（绘图）。
外部工具：`primesieve.exe`（`.refactor/primesieve-12.15-win-x64`）、`gp.exe`（`D:\AppData\Pari64-2-17-3`，可选交叉校验）。

## 方法论

**命中判定**：对素数 p，stage 1 命中 ⇔ `[s]P = ∞`（Edwards 恒等点 (0:1:1)，Montgomery 的 Z=0），
其中 `s = lcm(1..B1) = ∏_{p≤B1} p^{⌊log_p B1⌋}`。等价于 `ord(P mod p)` 为 B1-powersmooth
（论文 §9 口径，仅 stage 1）。

**有效除子 D_eff**：`model.py` 反解唯一的 `D = exp(δ)`，使 GMP-ECM 的 stage-1 模型
`stage1_prob(B1, p_ref, δ) = ρ_local(log(p_ref/D)/log B1)` 等于经验成功率
（`p_ref = 2^(bit−0.5)`），并输出 `extra = D/T`（Galois 额外光滑性因子）。
归一化方法见下节。

## D_eff 归一化方法

不同 bit 测得的 fraction 差异很大（15 bit ≈70%、25 bit ≈5%），这是**模型里 `log p`
项的预期结果**，不是曲线差异。归一化的任务就是把位宽从 fraction 里扣除，反解出
纯曲线量 D_eff。

**模型**（GMP-ECM local-Dickman-ρ）：

$$f \approx \rho_{\text{local}}(u),\qquad u=\frac{\log(p/D)}{\log B_1}=\frac{\log p-\log D}{\log B_1}$$

- `log p`：因子规模（随 bit 变，已知）；`log B1`：光滑界（已知）；`log D`：曲线有效除子（反解）。

**反解（归一化）**：对每个 bit b 解

$$D_{\text{eff}}=\exp(\delta)\ \text{s.t.}\ \rho_{\text{local}}\!\Big(\tfrac{\log p_{\text{ref}}-\log D_{\text{eff}}}{\log B_1}\Big)=f_b,\qquad p_{\text{ref}}=2^{\,b-0.5}$$

即"固定 B1、把 `log p` 从光滑度参数 u 里扣除、反解剩余的唯一自由参数 `log D`"。

**两种口径**：

| 口径 | 反解目标 | suyama@bit20 |
|---|---|---|
| 点口径（当前实现） | `ρ_local(log p_ref − log D / log B1) = f` | D=20.00 |
| 素数集平均（更严格） | `(1/N)Σ_{p∈primes} ρ_local(log p − log D / log B1) = f` | D=20.71 |

点口径用单点 `p_ref=2^(b−0.5)` 近似，引入 ~1–2% 系统误差；素数集平均对同一批素数逐点
求模型再平均，更接近 GMP-ECM 的 `exp(3.134)=22.97`。

**有效性判据**：D_eff 跨位宽应近乎恒定。实测（B1=256）：

| 曲线 | T | bit15 | bit17 | bit19 | bit21 | bit23 | bit25 |
|---|---|---|---|---|---|---|---|
| suyama_s10 | 12 | 21.68 | 22.17 | 19.50 | 21.05 | 21.40 | 20.85 |
| edwards_Z12 | 12 | 25.89 | 27.97 | 25.18 | 25.34 | 24.90 | 25.27 |
| edwards_Z2×Z8 | 16 | 26.13 | 27.29 | 25.32 | 25.81 | 26.03 | 25.30 |
| param3_s10 | 4 | 5.96 | 6.51 | 6.14 | 6.81 | 6.43 | 6.78 |
| pm1 | 2 | 2.63 | 2.64 | 2.71 | 2.72 | 2.67 | 2.59 |

D_eff 波动仅 ~5–8%（而 fraction 波动 ~10×），证明归一化正确、D_eff 确为曲线固有属性。

**D_eff ≈ 21 vs GMP-ECM 22.97 的口径差异**：本工具 D_eff 是"stage-1-only、单一 bit、
local-ρ 模型"口径，用于**跨曲线比较**；GMP-ECM 的 `3.134` 按"stage1+stage2 + 数位区间
期望曲线数"整体标定。两者都是有效除子但基线不同；跨曲线的**相对值**（Edwards Z/12≈25.8、
Suyama≈20.8、param3≈6.4）在两种口径下都稳健。

## ecm_prob.py solve 参数反解语义

`ecm_prob.py solve` 把 `{bit, B1, N(曲线数), p(单曲线概率)}` 按 **正向 / 反向** 两类语义处理：

- **正向**（`--bit` 与 `--B1` 都给）：`p = f(bit, B1)` 由 rho 模型决定；`--curves` 是"跑多少条"的输入，直接算该配置的 miss。
- **反向**（`--bit` 或 `--B1` 缺其一）：用 `--prob`，或 `--curves` 经 `p = 1/N`（固定标准 miss = e⁻¹ = 36.8%），反解缺的 `B1`/`bit`。

| 输入 | 语义 | 输出 |
|---|---|---|
| `--bit --B1` | 正向，标准 | `p=f(bit,B1)`、`N=1/p`、`miss=36.8%` |
| `--bit --B1 --curves` | 正向，实际 | `p` + `miss=(1-p)^N` + 36.8% 参考行 |
| `--bit --B1 --prob` | 正向 | `p`（模型）；若 `prob≠p` 则注一行 |
| `--bit --curves` | 反向 | `B1`（`p=1/N`，标准 miss=36.8%） |
| `--B1 --curves` | 反向 | `bit`（`p=1/N`，标准 miss=36.8%） |
| `--bit --prob` | 反向 | `B1`（`p=prob`，标准 miss=36.8%） |
| `--B1 --prob` | 反向 | `bit`（`p=prob`，标准 miss=36.8%） |

示例（正向实际 miss，含 stage2）：

```powershell
python ecm_prob.py solve --curve param3_s10 --bit 130 --B1 44000000 --curves 960
D = 6.41, B2=100*B1
p = f(bit=130, B1=4.4e+07) = 0.1824%
N = 960 curves -> miss = (1-p)^N = 17.3287%
   (ref: miss = e^-1 = 36.8%  needs N = 1/p = 548.2 curves)
```

示例（反向求 bit，标准 miss）：

```powershell
python ecm_prob.py solve --B1 44000000 --curves 960
D = 20.90, B2=100*B1
bit = 120.90
   (standard miss = e^-1 = 36.8%,  p = 0.1042%,  N = 960.0)
```

核心一句话：**正向时 `p` 由模型决定、`curves` 是输入（算 miss）；反向时 `curves`/`prob` 是目标（`p=1/N` 固定标准 miss=36.8%），反解 `B1`/`bit`。**

## 数据来源（provenance）

- 素数集：**本地生成**（`primesieve 12.15`），每 bit 一组穷举 `[2^(b−1), 2^b−1]`，
  二进制 uint64 LE 存于 `data/primes/bits{b}.bin`，`manifest.json` 记录
  `{范围, 数量, sha256, 生成命令, 时间}`。
- 曲线参数：来自论文 §9.1（四条 Edwards）与 GMP-ECM `parametrizations.c`（Montgomery param 0/1/2/3）。
- 与论文的差异：论文 §9.2 的 30-bit 样本未公开种子，本工具 30-bit 以上用固定种子新样本；
  论文 §9.1 数字比本工具严格判定高约 0.16%（见下）。

## 验证结果

| 项 | 结果 |
|---|---|
| `rho(u)=28.1894%`, `u^-u=22.8824%`（§9.4） | ✅ 精确一致 |
| 5 类估计 × 4 扭子群（§9.4 共 20 项） | ✅ 精确一致 |
| Edwards 4 曲线点阶 vs gp（`tests/cross_check.py`） | ✅ 4/4 一致 |
| Edwards §9.1 命中数 | 差 0.16%（见下） |
| Montgomery 2P vs 符号公式 `(9 : 64d+8)` | ✅ 一致 |

### 关于 0.16% 差异（重要）

本工具测得 20-bit/B1=256 的严格 B1-powersmooth 命中数为 **Z/12→12404**、Z/2×Z/8→12620、
Z/2×Z/4→10608、Z/4→9054；论文 §9.1 为 12467 / ~12689 / ~10619 / ~9068。

两者相差 ~0.16%，且系统性地本工具偏低。经 `tests/cross_check.py` 与 gp 点阶逐例核对，
本工具的 `[s]P=identity` 判定是**数学上严格正确**的（ord(P) 为 B1-powersmooth）。
论文的 EECM-MPFQ 实测数字略高，是因为其**窗口化加法链在中间倍数撞上 2-挠点时
会"提前命中"**（inverted/扩展坐标中除零即报因子），这部分素数满足
`ord(P) | k`（k 为链的某个中间倍数）但不满足 `ord(P) | s`。

对研究用途而言，本工具的严格判定是更可复现的基准；论文数字作为 EECM-MPFQ 实现的
实测参照并列呈现。

## 参考

- GMP-ECM：`.refactor/ecm/rho.c`、`parametrizations.c`、`ecm.c`、`README`
- Bernstein–Birkner–Lange–Peters, *ECM using Edwards curves*（Math. Comp. 82, 2013；本地 `docs/ECM USING EDWARDS CURVES.pdf`）
- [EFD: Edwards/Montgomery 显式公式](https://hyperelliptic.org/EFD/)
- 综合分析见 `docs/ECM_PARAMETERIZATION_ANALYSIS.md`
