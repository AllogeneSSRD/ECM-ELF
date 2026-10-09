# GPUOWL NTT：域运算、布局、IBDWT 与重构

固定参考资料。来源：`DEV_GPUOWL_NTT_NOTES.md §§1–10`。保留数学推导、第三方源码分析及其条件；源码行号针对原文研究的本地副本，第三方上游可能已有变化。本文不声明本仓库当前支持、默认值或性能。涉及实验数字时只按原文条件理解；当前实现见 [架构](../architecture/OVERVIEW.md)，当前计时口径见 [性能](../performance/STAGE2.md)。

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

一个复数元素承载两个连续的大整数 word；因此长度为 `ND` 的复数 NTT 对应
`NWORDS` 个整数 word。`NW` 和 `NH` 通常为 4 或 8，分别控制 width/height
方向的 radix 和每个线程暂存的元素数。`MIDDLE` 为 2、4、8 或 16。

这不是单一的一维 Stockham 循环，而是带转置的分解变换：

1. `fftP`：读取整数 word，应用 IBDWT 权重并执行 width 方向变换；
2. `fftMidIn`：middle 方向 butterfly、乘 twiddle、转置；
3. `tailSquare` 或 `tailMul`：height 方向变换及中间乘法；
4. `fftMidOut`：逆向 middle 变换并转置回去；
5. `fftW` 与 carry，或融合后的 `carryFused`。

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

表中 6 个值对应不同的 middle/height 变体；形状不在表中时，
`FFTConfig::maxBpw()` 会尝试映射到已知形状并扣减余量，仍找不到时使用默认值
并记录错误。bpw 表是按长期 ROE/溢出经验标定的安全裕量，不是数学上的硬上界。

它影响三件事：是否超过形状可支持的最大指数、是否启用 long-carry，以及
kernel 中 `MAXBPW` 和 carry 快捷路径的选择。`CARRY32` 对 word 位宽有严格
限制；NTT 路径不能在超过其允许范围后强行使用 32 位 carry。


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
