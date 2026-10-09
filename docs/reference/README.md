# 固定数学与第三方源码资料

这里保留数学原理、算法推导和第三方实现研究，允许公式、引用代码、伪代码和数据流图。它们按研究的版本与条件阅读，不随本仓库实现状态删改；错误仍需核对修正。使用方式、当前实现、默认参数与性能资格见[项目文档](../README.md)。

## ECM 数学

- [GPU Stage1 数学流程](ECM_STAGE1_GPU_MATH.md)：输入、ladder、因子提取、并行和成本；来自 `ECM_GPU_FLOW.md`。
- [Montgomery / Suyama / Lucas / PRAC](ECM_MONTGOMERY_MATH.md)：差分公式、标量、链复用与 Prime95/GMP-ECM 入口。
- [Edwards 构造与公式](ECM_EDWARDS_MATH.md)：Atkin–Morain、扩展坐标、NAF、坐标转换及 Prime95 v6 格式；来自 `ECM_EDWARDS_STAGE1.md`。
- [参数化与成功率](ECM_PARAMETERIZATIONS.md)：sigma、挠子群、有效除子、Dickman 模型；来自 `ECM_PARAMETERIZATION_ANALYSIS.md`。
- [X-only 标量乘研究](ECM_XONLY_SCALAR.md)：差分信息限制、成本枚举、伪代码与不确定性。
- [多项式 Stage2 数学](ECM_STAGE2_MATH.md)：Newton 逆、fold、齐次尺度、scaled 下降及工作量。

## 第三方实现

- [GPUOWL NTT](GPUOWL_NTT.md)：保留 `DEV_GPUOWL_NTT_NOTES.md` §§1–10 的域运算、IBDWT、twiddle、CRT、carry 与源码索引。
- [GPUOWL CUDA FFT](GPUOWL_FFT.md)：共享内核、浮点/整数/混合路径和权重布局。
- [Prime95 Poly Stage2](PRIME95_STAGE2.md)：D/B2/内存、F/G/倒数、截断 fold、共享变换、scaled 下降与源码行号。
- [PrMers ECM](PRMERS_ECM.md)：prime plan、所需 baby residue、giant 流、交叉乘积、GCD/恢复及访存。
- [Prime95 Auto B2 / PRPLL tune](PRIME95_AUTO_B2_TUNE.md)：收益公式、成本与内存选择、FFT tune 的实际搜索。
- [gwnum / polymult](GWNUM_POLYMULT.md)：FFT 表示、接口与第三方源码事实。
- [开源 GPU NTT](GPU_NTT_LIBRARIES.md)：GPU-NTT、ICICLE、sppark、TensorFHE 等库的算法与适配条件。

## 整数算术与映射

- [Goldilocks 与梅森算术](GOLDILOCKS_ARITHMETIC.md)：任意128-bit归约证明、PTX进位链、逆长度缩放和 Montgomery 旋转。
- [Tensor NTT 数学](TENSOR_NTT_MATH.md)：byte MMA、132-bit重构、算术和容量公式。
- [协作乘法](COOPERATIVE_MULTIPLICATION.md)：Karatsuba 分解、协作布局与算术条数。

来源章节在各页开头列出。第三方代码路径及行号来自原文研究的本地副本；外部库的能力不等于本仓库已有相应实现。引用中的数学模型、样本和推断按各自条件解释，不作为当前默认或通用性能承诺。

原论文与译述保留在 `docs/paper/`，其中 [ECM Stage2 文献综述与关键章节译述](<../paper/Stage2/ECM Stage 2 文献综述与关键章节译述.md>) 提供文献入口。
