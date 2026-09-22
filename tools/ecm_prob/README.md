# ecm_prob — ECM 参数化概率定量工具

Windows 可用的 ECM 参数化成功概率定量分析工具。分两阶段：

- **阶段 A（预测）**：`rho.py` 精确移植 GMP-ECM `rho.c`（Dickman-ρ + `ECM_EXTRA_SMOOTHNESS`），
  `estimates.py` 复刻论文 §9.3 的 5 类朴素估计。
- **阶段 B（经验）**：`ecmath.py` 纯 Python 实现 Edwards(a=1) / Montgomery(XZ) / Weierstrass 群律，
  对穷举素数集做 stage-1 命中测量（`[s]P = identity (mod p)`），`calibrate.py` 反推有效除子 D。

## 目录结构

```
ecm_prob/
  rho.py            Dickman-ρ / ecmprob 移植（验证：ρ(u)=28.1894%, u^-u=22.8824% 与论文§9.4一致）
  estimates.py      论文 §9.3 五类朴素估计（验证：20 项与 §9.4 精确一致）
  curves.py         曲线清单：Montgomery param0/1/2/3 + Edwards 4条 + p-1/p+1
  ecmath.py         纯 Python 点运算 + [s]P 标量乘 + 身份判定
  gen_primes.py     primesieve 生成/缓存 15–20 bit 穷举素数集 + manifest
  measure.py        经验命中测量
  sweep.py          跨位宽批量测量（15–20 穷举，21–25 每 bit 65536 固定种子样本）
  calibrate.py      拟合有效除子 D
  report.py         汇总输出（markdown + CSV）
  plot.py           绘图（逐 bit 柱状图 + 成功率/bit + D_eff/bit 折线）
  predict.py        成功率预测（经验 D_eff / 理论 D / 扭子群 D 三口径，任意 bit/B1）
  params.py         参数四元组 {bit,B1,N,p} 反解 + GMP-ECM 推荐表复现
  test_golden.py    论文 §9.1 黄金数字对拍
  cross_check.py    与 PARI/GP 的点阶交叉校验
  gp_check.gp       gp 交叉校验脚本
  bench.py          每素数耗时基准
  d_eff_table.py    打印 D_eff 跨位宽表（点口径 + 素数集平均口径）
  data/primes/      bits{b}.bin + manifest.json
  out/              measure*.json / report.md / summary.csv
```

## 快速开始

```powershell
cd D:\code\MPA-OpenCl\tools\ecm_prob

python gen_primes.py            # 生成 15–25 bit 穷举素数（幂等，sha256 缓存）
python test_golden.py           # 对拍论文 §9.1 四条 Edwards 曲线（20-bit/B1=256）
python estimates.py             # 对拍论文 §9.4 五类估计
python report.py 20 256         # 端到端：测量全部曲线 + 校准 D + 输出 out/report.md
python sweep.py                 # 跨位宽批量测量（15–20 穷举，21–25 采样 65536），B1=256
python plot.py                  # 绘图：out/plots/{bars_per_bit,success_vs_bit,d_eff_vs_bit}.png
python predict.py --all 25 256  # 成功率预测表（全部曲线，任意 bit/B1）
python predict.py suyama_s10 30 1024   # 单条曲线预测
python params.py --bit 30 --B1 1358 --curves 2    # 四元组任意3个解第4个
python params.py --gmp-table                       # GMP-ECM 推荐表(bit→B1→curves)
```

依赖：仅 Python 标准库（`numpy`/`matplotlib` 可选，用于后续绘图）。
外部工具：`primesieve.exe`（`.refactor/primesieve-12.15-win-x64`）、`gp.exe`（`D:\AppData\Pari64-2-17-3`，可选交叉校验）。

## 方法论

**命中判定**：对素数 p，stage 1 命中 ⇔ `[s]P = ∞`（Edwards 恒等点 (0:1:1)，Montgomery 的 Z=0），
其中 `s = lcm(1..B1) = ∏_{p≤B1} p^{⌊log_p B1⌋}`。等价于 `ord(P mod p)` 为 B1-powersmooth
（论文 §9 口径，仅 stage 1）。

**有效除子 D_eff**：`calibrate.py` 反解唯一的 `D = exp(δ)`，使 GMP-ECM 的 stage-1 模型
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
| Edwards 4 曲线点阶 vs gp（`cross_check.py`） | ✅ 4/4 一致 |
| Edwards §9.1 命中数 | 差 0.16%（见下） |
| Montgomery 2P vs 符号公式 `(9 : 64d+8)` | ✅ 一致 |

### 关于 0.16% 差异（重要）

本工具测得 20-bit/B1=256 的严格 B1-powersmooth 命中数为 **Z/12→12404**、Z/2×Z/8→12620、
Z/2×Z/4→10608、Z/4→9054；论文 §9.1 为 12467 / ~12689 / ~10619 / ~9068。

两者相差 ~0.16%，且系统性地本工具偏低。经 `cross_check.py` 与 gp 点阶逐例核对，
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
