# GKL family (iv) 实验草稿

## 范围

按 Gélin–Kleinjung–Lenstra，ISSAC 2017，表 1 第 4 行及表 2 最后一行，实现 family (iv) 的确定性构造。种子 m 表示辅助曲线 `y²=x³−x²−9x+9` 上的 `[m](5,8)`，不冒充 Prime95 Atkin–Morain sigma。实验入口与当前生产曲线身份、保存格式和默认值分离。

本轮验证构造、群阶和点阶、Stage1/单大素数 Stage2 数学命中、D_eff，以及同算术路径的构造和标量乘性能。论文给出的 1–2% 不作为本仓库预设结果。原始证据与报告数据置于 `data/experiments/ecm_friendly_iv_20261024/`、`data/benchmarks/ecm_friendly_iv_20261024/`。

## 已核对的数学边界

- 论文第 2.3 节式 (2)：`e=(g−1/g)/2`，不是 `(g−1)/g²`；等价条件为 `e²+1` 是有理数平方。
- family (iv) 目标是 `−x²+y²=1−e⁴x²y²`，Q 上挠子群为 Z/2×Z/4，群阶为 8、指数为 4；原文仍保证好素数下 16 整除有限域群阶，并给出部分素数下 32 整除的条件。
- 16/3 和 14/3 是对好素数平均的群阶 2 进估值，不是有理挠群阶，也不能直接换成普适 D_eff。
- 当前 Atkin–Morain 使用 a=1，须区分 Z/2×Z/8 的 16 个有理挠点与最大挠点阶 8。当前 48LCM 是计算标量，不能据此声称有理 3 挠点或群阶必含因子 3。
- 新旧构造的原始点阶与乘 48 后点阶分别记录，后者为 `r/gcd(r,48)`。

## 验证计划

1. m=1 黄金值：t=6、e=15/8、P=(28/195,3152/3495)。核对曲线方程、平方条件、映射及有理挠群。
2. 以 PARI/GP 的 ellcard/ellorder 为独立判据，覆盖多个种子和素数，检查好约化、Hasse 界、点阶最小性及有限域整除性。
3. 在相同素数和种子网格比较 family (iv)、现有 Atkin–Morain 及通用 Z/2×Z/4 对照；坏约化不混作正常 NF，报告所有分母。
4. 分别报告原始 LCM 与 48LCM 的实际点命中，及与之匹配的 D_eff；Stage2 单大素数条件单独列出。
5. 性能记录实现版本、输入、样本数、硬件和计时边界；构造、标量乘、完整处理分别测量，不从概率工具耗时推断 GPU 内核吞吐。

## 状态

实验版本 1.0.0 已完成，宿主版本 0.52.3；生产默认与身份未改，主程序版本不因独立实验再次变化。正式结论见 [GKL family (iv) 实验](../performance/ECM_FRIENDLY_IV.md)。

- 实现：`src/ecm/core/curve/ecm_gkl_iv_curve.h` 使用 Jacobian 辅助倍点、检查分母与非奇异性；`tools/test/ecm_friendly_probe.cpp` 支持 GMP w6 Stage1 和显式 a24 的 GPU Stage2。Stage2 复用已有 0.52.2 引擎，只编译宿主入口并链接，零 CUDA 源文件编译。
- 独立参考：Python Fraction、PARI elliptic arithmetic、原生 GMP 三路核对。110 个构造样例（1 个退化）、60 个精确有理余数核对、96 个末点、160 个复合构造样例通过；7 项数学与分析测试通过。
- 异常点：小素数 1009 下旧 AM 的两条 NAF 路径出现无效投影点；原型检验扩展坐标后从原始点重放 Montgomery X/Z ladder，96 条末点全部与 PARI 相同。性能批次仍要求所有 Edwards 末点直接通过，不把重放隐藏到计时中。
- 群阶扫描：固定 100 条 IV、100 条 AM、100 条通用族及 2 条固定论文对照，共 4,197,800 输入、493 不可用构造。素数按 15/20/25/30 bits 取样；15 bits 全量，其余各 4096。成功率按实际点阶和标量判定，不把点阶同群阶混用。
- 概率：30-bit、B1=256、48LCM，IV/AM Stage1 为 3.1570%/3.1367%，条件 D_eff 为 28.19/27.73；配对置信区间不能确认生产标量下的正收益。群阶平均 v2 为 5.3336/5.3363，通用族 4.6653。
- 实际 GPU 验证：GPU1，N=537033121×M127，B1=256、B2=25600；两族都找回 537033121，各 672 selftest、183 checked、bad=0。PARI 给出的剩余素数分别为 593、281。
- 性能：CPU 32 曲线×5 批、GPU1 Stage2 3 种子×3 轮；521/2203/4423 bits。线性及对数图已复制到正式报告同目录 attachments。实测波动保留，未证明新族有稳定净收益。

原始证据：`sweep_01/`、`native_validation_01/02/`、`native_validation_final/`（01 保留失败以呈现异常，final 为最终宿主构建复核）、`gpu_validation_02/`、`cpu_performance_01/02/`、`gpu_performance_01/`；汇总与图片分别在 `data/benchmarks/ecm_friendly_iv_20261024/` 和 `data/figures/ecm_friendly_iv_20261024/`。

待后续生产接入时，先定义独立曲线族与存档身份，再实现并测量 a=−1 GPU Stage1；概率模型需要更大目标因子与生产 B1/B2 的验证。当前 CPU/GPU Stage2 性能不支持宣称稳定提速。

本草稿保留，实验完成后由用户决定是否清理；正式数学依据及实测结论另行整理。
