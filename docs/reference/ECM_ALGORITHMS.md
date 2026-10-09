# 算法参考与适配边界

本页说明当前实现所依赖的算法和外部代码阅读入口。外部项目的能力不等于本仓库已实现或已完成性能验证。

完整数学、第三方源码分析、公式和伪代码见[固定资料目录](README.md)。

## ECM 与参数化

ECM 的发现条件取决于点阶在 B1/B2 下的可消去部分；曲线群阶和初始点阶相关，但不相同。不同参数化改变挠子群、平均光滑性、构造及点算术成本，sigma 的含义也不同。

本仓库生产 CUDA Stage1 支持相应 Suyama/batch 曲线族，CPU 还有 Edwards。PARAM0 由自由 sigma 构造；batch32 的 sigma 表示曲线常数种子；不能将一种参数化的保存点按另一种解释。选择12倍指数的界限必须与 lcm 分开。

参考本地 GMP-ECM 的 parametrizations、ecm、rho 源；论文 *ECM using Edwards curves*、*Parametrizations for Families of ECM-friendly Curves*、*Revisiting ECM on GPUs* 在 `docs/paper/`。它们用于数论/算术依据，不作为当前工程功能规范。

## 多项式 Stage2

Montgomery 的 FFT 扩展使用 baby/giant 结构，将候选相遇转为多项式乘积和多点求值。Bernstein 的 scaled remainder/middle-product 表达用于减少下降中的除法与存储压力。本仓库当前采用 F 积树、分批 G、H mod F 和一次 scaled 下降。

本地论文入口：`docs/paper/Stage2/`，包括 *An FFT Extension of ECM*、*FFT Extension for Algebraic-Group Factorization Algorithms* 和 `scaledmod-20040820.pdf`。算法量级不能直接变成 GPU 速度；实际取决于系数算术、截断、NTT 长度、准备和生命周期。

Prime95 阅读入口为 `.refactor/p95v3106b01.source/ecm.cpp`：poly Stage2、F/G/倒数与 scaled 下降、B2/内存规划，以及目标余因子和原梅森形式的分离。其 gwnum/polymult 使用浮点 FFT 与安全余量，本仓库使用精确 Goldilocks NTT+carry+S4，不能直接复制它的内存系数或 FFT 周期成本。

Prime95 的 Lucas/PRAC 链与 GMP-ECM 的 `lucas`/`prac` 为 Stage1 链规划参考。CPU 点算术平方成本、GPU CGBN 的活跃点/寄存器成本不同，少点运算不保证 GPU 吞吐更高。

## PrMers ECM

比较对象为 `.refactor/PrMers-main/src/modes/RunGaussianMersenneEcmOptimized.cpp` 的 differential BSGS ECM；P−1 的 V-trace/Pair95 不是椭圆曲线 Stage2，不能混入 ECM 对照。

其 ECM 路线按素数计划挑选实际需要的 baby residue，giant 流保持相邻点，累乘 XgZb−ZgXb 并分批 GCD。小 D/少 baby 的寄存器驻留减少 global memory，但工作仍按候选项增长；当前多项式路线以较大树/NTT 和存储换取大 B2 的批处理能力。

工程可借鉴 residue 去重、差分 chain、乘积/GCD 的边界恢复；不能用 Gaussian-Mersenne 的字段布局直接替代本仓库普通/承载模数。源码、设备和数学目标不同，未经同输入测量不报告速度比。

## PRPLL / gpuowl 与开源 NTT

PRPLL 的 tune 通过实际变换吞吐规划尺寸拐点，本仓库对应 field-convolution tune 与完整 ECM 成本标定分离。PRPLL/gpuowl 的 NTT/IBDWT 处理整数/梅森变换，布局、权重和 carry 合同需单独适配多项式槽。

开源 CUDA Core 参考包括 GPU-NTT、ICICLE、sppark；Tensor Core 参考包括 tensor-core-ntt、TensorFHE 和 cuda-zkp-ntt。小字长模数/多模数 FHE 与 Goldilocks64 精确重建的成本不同；可参考 butterfly fusion、tile/outer 划分、根表布局、batch 和传输管理，而非直接替换数论表示。

Tensor Core 精确拆分的重建、MMA 数量、数据布局及 carry 可能超过收益。本仓库独立整数实验未成为生产默认。当前实际实现和可用性见 [NTT](../architecture/NTT.md)、[性能](../performance/STAGE2.md)。

## 当前代码联系

- [Stage1](../architecture/STAGE1.md)：点乘和 PRAC。
- [Stage2](../architecture/STAGE2.md)：树/fold/scaled 下降。
- [目标/承载上下文](../../src/core/ecm_stage2_modulus.h)：N∣M 与域语义。
- [Auto B2](../architecture/AUTO_B2.md)：相对收益、标定资格和搜索。

短/截断乘积、固定操作数频域缓存及并发租约仍需独立实现与测量，统一列在 [TODO](../TODO.md)，不写为当前算法规则。
