# avx-ecm 的 AVX-512 实现

研究对象为 Ben Buhrow 的独立版 `118e015`（2021-03-09），本地源码在 `.refactor/avx-ecm/`。YAFU 内置的 ECM 后端有独立演进，本页以独立版为准。实测条件、分派对照和性能见 [Stage1 CPU 性能](../performance/STAGE1.md#avx-ecm-单核对照)。[上游仓库](https://github.com/bbuhrow/avx-ecm)

## 曲线与 Stage1

每个线程负责一个满 SIMD 批次。模数、素数序列和点运算的控制流在线程内共享，sigma、曲线参数和点坐标各占一条 SIMD lane。命令行可指定起始 sigma，单线程首批依次使用 `sigma, sigma+1, …`。曲线使用 Suyama PARAM0。令 \(u=\sigma^2-5\)、\(v=4\sigma\)，初始 \(X:Z=u^3:v^3\)，曲线常数为 \(A_{24}=(v-u)^3(3u+v)/(16u^3v)\)。构造用 GMP，将初始 \(Z\) 归一化为 1。该版本没有检查构造中的 `mpz_invert` 返回值，涉及不可逆分母的输入需单独验证。[批次与 sigma](https://github.com/bbuhrow/avx-ecm/blob/118e015caba54747c6ac9f0e293bfd9f8d58b3a0/main.c#L757-L762) · [曲线构造](https://github.com/bbuhrow/avx-ecm/blob/118e015caba54747c6ac9f0e293bfd9f8d58b3a0/ecm.c#L1587-L1778)

Stage1 先处理 2 的幂，再逐个处理奇素数。每当 \(p^j<B_1\)，就对当前点执行一次乘以 \(p\) 的 PRAC 链。链用 10 个比例候选估算成本，默认将模平方按 0.75 次模乘计价。链中一般差分加点用 4 次模乘和 2 次平方，倍点用 3 次模乘和 2 次平方。程序直接执行这些小标量链，省去完整 LCM 标量的展开。Stage2 采用 baby/giant 点与素数 pairing，本轮对照用 `B2=B1` 禁用它。[点公式与 PRAC](https://github.com/bbuhrow/avx-ecm/blob/118e015caba54747c6ac9f0e293bfd9f8d58b3a0/ecm.c#L407-L613) · [Stage1 循环](https://github.com/bbuhrow/avx-ecm/blob/118e015caba54747c6ac9f0e293bfd9f8d58b3a0/ecm.c#L1806-L1849)

## SIMD 算术与归约

| 构建 | 每线程曲线数 | 容器步长 | 算术 |
| --- | ---: | ---: | --- |
| 52 位 IFMA | 8 | 208 位 | IFMA 分别累加 52 位乘积的高、低半部。通用 REDC 内仍有显式 FP64 FMA。 |
| 52 位 FMA | 8 | 208 位 | 用带偏置的 FP64 FMA 拆分整数乘积。 |
| 32 位整数 | 16 | 128 位 | `vpmuludq` 分别处理偶、奇 lane，乘积按 64 位累加。梅森平方调用乘法入口。 |

每个 limb 在内存中并排存放各曲线的值。容器按四个 limb 为一块，选取严格大于输入位宽的最小整块大小。内核使用二次复杂度的乘法。52 位梅森内核按四列累积，将正规化的双宽乘积在指数 \(p\) 处分开，再利用 \(2^p\equiv1\pmod{2^p-1}\) 折回。一般奇数走 Montgomery REDC。独立版的载体探测只枚举 \(p<2048\)。识别到的载体也可能因实际模数容器宽度不足载体的 70% 而回到 REDC。完整 \(2^p-1\) 在未剥除代数因子时，这个比值为 1。[宽度与归约选型](https://github.com/bbuhrow/avx-ecm/blob/118e015caba54747c6ac9f0e293bfd9f8d58b3a0/main.c#L405-L526) · [52 位内核](https://github.com/bbuhrow/avx-ecm/blob/118e015caba54747c6ac9f0e293bfd9f8d58b3a0/vecarith52.c) · [32 位内核](https://github.com/bbuhrow/avx-ecm/blob/118e015caba54747c6ac9f0e293bfd9f8d58b3a0/vecarith.c)

本机构建发现两处影响结果的路径。32 位梅森加减将最高 limb 的 32 位掩码按 64 位广播，导致奇数索引 lane 的最高 limb 被清零。52 位 IFMA 构建的通用 REDC 在本轮样本中产生错误余数。该路径有直接调用带偏置 FP64 FMA 的片段，而 `IFMA` 构建禁用了相应的 `SUB_BIAS_*` 宏。[32 位加减掩码](https://github.com/bbuhrow/avx-ecm/blob/118e015caba54747c6ac9f0e293bfd9f8d58b3a0/vecarith.c#L2593-L2645) · [52 位偏置宏](https://github.com/bbuhrow/avx-ecm/blob/118e015caba54747c6ac9f0e293bfd9f8d58b3a0/vecarith52.c#L146-L245) · [通用 REDC 的 FMA 调用](https://github.com/bbuhrow/avx-ecm/blob/118e015caba54747c6ac9f0e293bfd9f8d58b3a0/vecarith52.c#L2521-L2577)

ECM-ELF 的 [CPU IFMA 算术](../../src/ecm/cpu/arithmetic/ifma/modular_ifma.cpp) 按实际需要的 52 位 limb 数工作，在梅森乘法中直接累积折叠结果。Montgomery CPU 默认使用 [固定差点的二进制 ladder](../../src/ecm/stage1/montgomery/cpu/simd_mont_curve.cpp)，差点的 \(Z=1\) 可以省去一次模乘。PRAC 的执行成本还包括一般差点的额外模乘、不同位宽的平方成本、链的计划开销和保存恢复方式。ECM-ELF 的 [PRAC 主机计划器](../../src/ecm/stage1/montgomery/prac/ecm_prac_plan.cpp) 由 CPU 和 CUDA 共用。

以本次 \(B_1=10^5\) 为例，avx-ecm 日志记录 195,448 次差分加点和 23,269 次倍点。按上式折算为 851,599 次模乘、437,434 次平方。ECM-ELF 的 144,344 位标量以固定差点梯形法计算，对应 866,061 次模乘、577,374 次平方。两者的主要差别是平方数约少 24%，而模乘约少 1.7%。按平方为 0.75 次模乘估算，总乘方成本约少 9.2%，还需加上实际内核、加减、调度和链搜索的成本。
