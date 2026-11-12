# GPUOWL NTT：域运算、布局、IBDWT 与重构

固定参考资料。来源：`DEV_GPUOWL_NTT_NOTES.md §§1–10`。保留数学推导、第三方源码分析及其条件；源码行号针对原文研究的本地副本，第三方上游可能已有变化。本文不声明本仓库当前支持、默认值或性能。涉及实验数字时只按原文条件理解；当前实现见 [架构](../architecture/OVERVIEW.md)，当前计时口径见 [性能](../performance/STAGE2.md)。



| `FFT_TYPES`   | 计算内容             | 元素（域）                     | B/元素 | `WordSize` | I/O B/word | 最低 bpw | 最低指数 \(p\) |
| ------------- | -------------------- | ------------------------------ | ------ | ---------- | ---------- | -------: | --------------: |
| `FFT64` 0     | FFT                  | `double2`                      | 16     | 4          | 8          | 3 | 393,216 |
| `FFT3161` 1   | 两个 NTT + CRT       | `GF(M31²)`+`GF(M61²)`          | 8+16   | 8          | 12         | 3 | 393,216 |
| `FFT3261` 2   | FFT + NTT            | `float2`+`GF(M61²)`            | 8+16   | 8          | 12         | 3 | 393,216 |
| `FFT61` 3     | NTT                  | `GF(M61²)` `ulong2`            | 16     | 4          | 8          | 3 | 393,216 |
| `FFT323161` 4 | FFT + 两个 NTT + CRT | `float2`+`GF(M31²)`+`GF(M61²)` | 8+8+16 | 8          | 16         | 3 | 393,216 |
| `FFT3231` 50  | FFT + NTT            | `float2`+`GF(M31²)`            | 8+8    | 4          | 8          | 3 | 393,216 |
| `FFT6431` 51  | FFT + NTT            | `double2`+`GF(M31²)`           | 16+8   | 8          | 12         | 3 | 393,216 |
| `FFT31` 52    | NTT                  | `GF(M31²)` `uint2`             | 8      | 4          | 4          | 3 | 393,216 |
| `FFT32` 53    | FFT                  | `float2`                       | 8      | 4          | 4          | 1 | 131,072 |

`32`、`64` 表示 FP32、FP64。`31`、`61` 表示以 \(2^{31}-1\)、\(2^{61}-1\) 为特征的二次扩域。最低指数按本地 `481d11ae` 的最小形状 `256:1:256` 计算，共 131,072 个整数 word。表列出指定类型时的形状准入下限，指数再增大时也须满足所选形状的 bpw 上限。[类型和 `minBpw()`](https://github.com/gwoltman/gpuowl/blob/481d11ae/src/FFTConfig.h#L26-L57)、[形状约束](https://github.com/gwoltman/gpuowl/blob/481d11ae/src/FFTConfig.cpp#L138-L148)

用户提供的消费级 GPU 测试来自 GIMPS 波前，指数约 8,000 万～1.5 亿位。优先参考类型 **1（3161）、2（3261）、4（323161）**。类型 1 是双域 NTT + CRT，类型 2 和 4 含 FP32。在这组经验里，类型 0（64）和 51（6431）只在 FP64 吞吐强的 GPU 上显示优势，纯 31、纯 32 等在该范围的对比中落后。ECM 关注的几千至几十万位需要在相同指数、设备、进位边界和批量下重新比较。按现有 GPUowl 形状，类型 1／2／4 的最低指数已高于 ECM-ELF 当前的 262,144 位 Stage1 NTT 上限。

下表取 `481d11ae` 的 `fftbpw.h` 原始容量值，位／字节等于容量 bpw 除以上表的 I/O B/word。运行时还会按变体和 carry 配置调整上限。[GPUowl `fftbpw.h`](https://github.com/gwoltman/gpuowl/blob/481d11ae/src/fftbpw.h)

| 类型 | `256:2:256` bpw 上限 | 位／字节 | `4K:16:1K` bpw 上限 | 位／字节 |
| --- | ---: | ---: | ---: | ---: |
| FFT64 | 19.288–19.702 | 2.41–2.46 | 16.851–17.079 | 2.11–2.13 |
| FFT3161 | 40.54 | 3.38 | 37.12 | 3.09 |
| FFT3261 | 35.65 | 2.97 | 33.01 | 2.75 |
| FFT61 | 25.02 | 3.13 | 22.42 | 2.80 |
| FFT323161 | 51.11 | 3.19 | 48.38 | 3.02 |
| FFT3231 | 20.65 | 2.58 | 18.03 | 2.25 |
| FFT6431 | 35.27 | 2.94 | 32.25 | 2.69 |
| FFT31 | 10.01 | 2.50 | 7.56 | 1.89 |
| FFT32 | 5.21 | 1.30 | 2.60 | 0.65 |

### 最短原生形状复测

使用 CUDA PRPLL `v8.0-683-g481d11ae`，GPU1 RTX 4060 Laptop（8 GiB，SM 目标 1500 MHz）。1500 MHz 且利用率至少 50% 的遥测点整卡功耗中位数为 11.2 W。各类型固定为 `256:1:256:202`，逐类型调用原生 tune，每格交替测 3 次。下表为含变换、归约和 PRP 块间校验的 `us/iter` 中位数，寄存器等 kernel 参数由 PRPLL 原生 tune 选择。四种类型均使用 long carry。

| 指数 \(p\) | 3161 | 3261 | 323161 | FP32 |
|---:|---:|---:|---:|---:|
| 393,241 | 163.3 | 155.1 | 214.4 | 87.4 |
| 524,287 | 158.2 | 152.4 | 214.0 | 84.5 |

在这个固定形状上，纯 FP32 用同样的 word 数，只计算一个算术通道。混合路线可利用的高 bpw 在这里未用满。这个排序属于最低形状附近的低 bpw，实际评估 ECM 目标位宽仍需更短的形状。日志、调优行和整卡遥测见 `data/experiments/fft_ntt_20261112/native_short/`。



## 1. 实现边界

项目没有独立的 `ntt*.cl`。NTT 与浮点 FFT 共用一套 OpenCL 内核，通过主机端
宏定义选择路径：

- `src/Gpu.cpp` 的 `kernelDefines()` 设置 `FFT_TYPE`、`WordSize`、`WIDTH`、
  `SMALL_HEIGHT`、`MIDDLE`、`CARRY_LEN`、`NW`、`NH` 等参数；
- `NTT_GF31`、`NTT_GF61`、`FFT_FP32`、`FFT_FP64` 决定每个内核的算术分支；
- `src/cl/fft4.cl`、`src/cl/fft8.cl` 中的 radix butterfly 由不同类型重载共用；
- `src/FFTConfig.cpp` 将 FFT 类型映射为数据类型和字宽。

CUDA 后端也不是另一套 NTT 算法。它通过 `src/cuda/opencl_compat.cuh` 提供的
兼容层，以及 `src/cuda/clwrap_cuda.cpp` 中的 NVRTC 编译流程，复用 `.cl` 内核。
只有在脱离现有兼容层、实现纯 CUDA 内核时，才需要重新实现本文列出的算术和布局。

源码转换、设备兼容头、主机调用适配，以及本地 PRPLL 与 PrMers/Aevum 副本的差异，见 [OpenCL 内核编译为 CUDA](OPENCL_KERNEL_CUDA_COMPILATION.html)。该资料另列 2026-10-27 所查副本的关键文件哈希，不将本文原始研究的源码行号用于新副本。


## 2. 域、类型与数据组合

核心模数和类型定义在 `src/cl/base.cl`、`src/cl/math.cl`：

```c
M61 = 2^61 - 1;   Z61 = ulong;   GF61 = ulong2;
M31 = 2^31 - 1;   Z31 = uint;    GF31 = uint2;
```

`Z31`/`Z61` 是素域元素；`GF31`/`GF61` 表示二次扩域中的一对元素，代码按
`a + i b` 处理，满足 `i² = -1`。长度为 2 的幂的根在扩域中生成：

- GF61 使用 `src/TrigBufCache.cpp` 中阶为 `2^62` 的根；
- GF31 使用阶为 `2^32` 的根。

主机端支持的组合包括 `FFT61`、`FFT3161`、`FFT3261`、`FFT323161`，以及
FP32/FP64 与 GF31 的混合类型。GF31 与 GF61 并非各自输出一个独立结果：
进位阶段通过 CRT 合并，基本形式为 `n61 * M31 + n31`；`FFT323161` 还利用
FP32 数据确定 `M31·M61` 的倍数。GF61 数据位于 GF31 数据之后，偏移由
`DISTGF61` 等主机参数传入内核。


## 3. 域运算约定

### 3.1 GF61

M61 约减利用：

```text
2^64 ≡ 8 (mod M61)
```

128 位乘积先折叠低 61 位和高位；乘积可能达到完整 128 位时，再折叠高端的
6 位。`weakModM61()` 和 `modM61()` 都是惰性约减，结果不保证落在严格的
`[0, M61)` 内。因此移植时必须同时保留：

1. `weakMul` 操作数为非负值的约定；
2. 调用者传入的折叠次数/上界；
3. butterfly 中每个 `modM61q(..., k)` 的人工推导上界。

加减法通过补加若干个 M61 保持中间值为正，最终再惰性约减。扩域乘法使用
Karatsuba 形式的 3 次宽乘法；扩域平方使用 2 次宽乘法。相关实现集中在
`src/cl/math.cl` 的 GF61 分支。

因为 `2^61 ≡ 1 (mod M61)`，乘以 `2^k` 实现为 61 位循环移位，而不是普通
整数乘法。`shl30`、`shl31` 是常用的专用路径。

### 3.2 GF31

GF31 使用相同的扩域结构；`modM31()` 将 64 位乘积按 31 位切分折叠。
当前代码包含多种实现，由 `MODM31` 选择。不要把 GF31 的严格范围假设套用
到 GF61：两者的惰性约减细节和可接受中间范围不同。


## 4. 变换结构与内存布局

主机端形状为 `WIDTH : MIDDLE : HEIGHT`，并定义：

```text
NWORDS = WIDTH * MIDDLE * HEIGHT * 2
ND     = NWORDS / 2
BIG_H  = SMALL_HEIGHT * MIDDLE
```

一个复数或扩域元素承载两个连续的大整数 word；因此长度为 `ND` 的成对数据对应
`NWORDS` 个整数 word。`NW` 和 `NH` 通常为 4 或 8，分别控制 width/height
方向的 radix 和每个线程暂存的元素数。本地 `481d11ae` 副本的 `FFTShape` 允许
`MIDDLE=1…16`，目录会按类型继续筛选；某些奇因子可在整数或混合路线中用
PFA 处理，并非只接受 2 的幂。见 `FFTConfig.h::pfaMiddle` 与
`src/cl/fftmiddlein.cl` 的 PFA 分支。

这不是单一的一维 Stockham 循环，而是带转置的分解变换：

1. `fftP`：读取整数 word，应用 IBDWT 权重并执行 width 方向变换；
2. `fftMidIn`：middle 方向 butterfly、乘 twiddle、转置；
3. `tailSquare` 或 `tailMul`：height 方向变换及中间乘法；
4. `fftMidOut`：逆向 middle 变换并转置回去；
5. `fftW` 与 carry，或融合后的 `carryFused`；`Gpu::square()` 允许把结果保留在 `LEAD_WIDTH`，省掉下一轮单独的 `fftP`。普通 `Gpu::mul()` 使用分离的 `fftW` 与 carry，并能复用已做到 `LEAD_MIDDLE` 的 B 操作数。

平方和乘法的主机调用链分别在 `Gpu::square()` 与乘法流程中记录。`INPLACE`
决定是否使用原地的 swizzled 布局；关闭原地模式时会使用额外 scratch buffer。
`src/cl/middle.cl` 中的 `SWIZ`、`SIZEBLK`、`SIZEW`、`SIZEM` 必须与主机端
分配和 kernel 参数保持一致。

每个线程在 width、height、middle 阶段分别持有 `NW`、`NH`、`MIDDLE` 个元素。
OpenCL 向量类型是 `ulong2`、`uint2`、`double2`、`float2`；不要在移植时擅自
改为 `float4`，因为这会改变布局和对齐。


## 5. Twiddle 表与 butterfly

twiddle 表由 `src/TrigBufCache.cpp` 生成并按 width、middle、height 区域拼接，
再由 `DISTWTRIGGF61`、`DISTMTRIGGF61`、`DISTHTRIGGF61` 绑定到 kernel。
普通 radix butterfly 的索引为：

```text
p = me & ~(f - 1)
trig[(i - 1) * WG + p]       (i = 1 .. RADIX-1)
```

因此第 `i` 个元素使用的 twiddle 指数为 `i * p`。`TABMUL_CHAIN61` 开启时，
只读取一个基准 twiddle，再通过连续乘法得到其幂；关闭时直接读取整行表项。
radix-8/radix-4 的压缩表由 `tabMul8_4a/b` 处理，不能与普通表布局混用。


## 6. IBDWT 权重与平方

权重变换用于处理 Mersenne 指数对应的非整数 bits-per-word。第 `j` 个 word
的权重为：

```text
2^(ceil(EXP*j/NWORDS) - EXP*j/NWORDS)
```

在 GF61 路径中，权重通过模 61 的循环移位实现。大小端 word 标志、分数位和
移位量由同一组 `fracBits`/`comboFracBits` 状态推进；逆变换还要应用
`log2(NWORDS) + 1` 修正，因为 NTT 输出带有 `2*NWORDS` 的比例因子。

中间阶段的平方不是逐点复数平方。`src/cl/tailsquare.cl` 的 `pairSq()` 和
`onePairSq()` 会先处理共轭配对，再按 `t_squared_type` 组合 `×1`、`×i`、
`×-1`、`×-i` 的结果。第 0 行和第 `H/2` 行是自配对行，需额外处理
`TAILTGF61`；完成中间运算后还会执行 `SWAP_XY`。这些特殊情况缺失时，结果
通常不是崩溃，而是静默错误。


## 7. bits-per-word 与精确性

`src/fftbpw.h` 提供按形状和类型索引的经验表。这里的 bpw 是每个大整数
word 的 bit 数，不是每个复数元素的 bit 数。主机端用：

```text
bitsPerWord = EXP / NWORDS
```

表项保存最大允许平均 bpw，而非本次真实字宽。表中 6 个值以 `000`、`101`、
`202`、`010`、`111`、`212` 为基准，部分较大 width 形状使用变体替代；
`FFTConfig::maxBpw()` 按 W/M/H 变体选值或插值，并根据 carry 模式限幅。
缺表时 `FFTShape` 负责从邻近形状估计。容量约为 `NWORDS·maxBpw`，还需检查
`minBpw`；`FFTConfig::bestFits()` 优先选 `tune.txt` 中成本最低的合格配置。
表是按数值和进位安全余量标定的配置容量；含浮点路径才涉及浮点舍入误差。

### 为什么也有最低 bpw

同一个形状固定了整数 word 数 \(N\)。指数太小时，平均字宽 \(p/N\) 下降，进位每经过一个字只能消去约 \(p/N\) 位。GPUowl 的融合进位先求局部 carry-out，再吸收前一组的 carry-in，可能把未完全归一化的高字留给下一次平方。上游记录表明，当字的基数 \(2^{p/N}\) 接近 \(\sqrt N\) 时，这些余量可能在重复平方中放大。因此 `Gpu` 对低于 \(\max(10,\frac12\log_2 N+0.5)\) bpw 的配置启用 long carry。分离的 carry 也可能跨组传播，低 bpw 的 FP32 路径在比较残值时会补做归一化。[长进位切换](https://github.com/gwoltman/gpuowl/blob/481d11ae/src/Gpu.cpp#L1402-L1414)、[跨组归一化](https://github.com/gwoltman/gpuowl/blob/481d11ae/src/Gpu.cpp#L2696-L2711)

`minBpw()` 是实现额外施加的准入线。给定形状，\(p_{\min}=N\cdot\mathrm{minBpw}\)，\(p_{\max}\) 则取决于变体、精度与 carry 容量。通用 **3 bpw** 的选值依据在该版本源码中未说明。纯 FP32 曾由 3 放宽到 **1 bpw**。低于 1 bpw 会出现 0 位 word，超出现有动态取位函数的约定。[FP32 下限变更](https://github.com/gwoltman/gpuowl/commit/bd5e35dd)、[取位约定](https://github.com/gwoltman/gpuowl/blob/481d11ae/src/cl/carryutil.cl#L30-L63)


## 8. 进位与 CRT 重构

每个 word 的有效位数为：

```text
nBits = EXP / NWORDS + (isBigWord ? 1 : 0)
```

carry 从一个 word 传播到下一个 word，大小端标志必须按同一序列推进。通常路径
使用 `carryFused`：逆权重、整数重构和 carry 在一个 kernel 链中完成，并通过
`carryShuttle` 在 workgroup 之间传递进位。long-carry 路径则拆成 `carry`、
`carryB` 和后续 `fftP` 等 kernel。

移植时必须保持以下语义：

- GF31/GF61 的 CRT 合并顺序为 `n61 * M31 + n31`；
- LL 路径的初始 carry 为 `-2`；
- `WordSize` 与 `Word` 的 4/8 字节选择必须一致；
- 一个复数元素对应两个连续 word；
- carry 中使用的 64 位路径不能被 32 位临时变量截断。


## 9. CUDA 移植清单

优先按以下顺序验证，性能优化放在功能正确之后：

1. 先复现 Z31/Z61 的惰性约减、宽乘法和扩域乘/平方；
2. 复现根、twiddle 表拼接顺序和 butterfly 索引；
3. 复现 IBDWT 权重、大小端序列以及逆变换比例修正；
4. 复现 `pairSq` 的配对规则、特殊行和 `SWAP_XY`；
5. 复现数据 swizzle、padding、`WordSize` 与 buffer 偏移；
6. 最后接入 carry、CRT 和跨 workgroup 的 carry hand-off。

以下宏主要影响性能，不应成为第一阶段的移植阻塞项：`INPLACE`、
`SHUFL_BYTES_*`、`LDSPAD_*`、`LDSMUL_*`、`WMUL`、`MULTI_Q`、`L2_STRIPING`、
`ZEROHACK_*`、`UNROLL_*`、寄存器上限，以及 `TABMUL_CHAIN*` 等表读取策略。
但它们改变工作组大小、共享内存或寄存器压力时，仍必须重新检查设备限制。


## 10. 源码索引与已知边界

- 主机参数、kernel 宏和调用链：`src/Gpu.cpp`、`src/FFTConfig.cpp`、`src/FFTConfig.h`；
- 域运算：`src/cl/math.cl`、`src/cl/base.cl`；
- butterfly 与 twiddle：`src/cl/fft4.cl`、`src/cl/fft8.cl`、`src/cl/fftbase.cl`、
  `src/TrigBufCache.cpp`；
- middle/tail/layout：`src/cl/fftmiddlein.cl`、`src/cl/fftmiddleout.cl`、
  `src/cl/tailsquare.cl`、`src/cl/middle.cl`；
- 权重与 carry：`src/cl/weight.cl`、`src/cl/carryutil.cl`、
  `src/cl/carryfused.cl`、`src/cl/carry.cl`、`src/cl/carryb.cl`；
- CUDA 兼容编译：`src/cuda/opencl_compat.cuh`、`src/cuda/clwrap_cuda.cpp`。

源码没有给出完整的 GF(p²) 理论推导，也没有固定的 FP64、GF61、GF31+GF61
性能对照表；实际选择由运行时调优和设备测量决定。`fftbpw.h` 未覆盖所有
类型/形状，缺失项应以 `FFTConfig::maxBpw()` 的实际回退行为为准，不要从邻近
表项外推未经验证的容量或性能结论。
