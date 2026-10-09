# GPUOWL CUDA FFT 与混合 NTT 路径

固定参考资料。来源：`DEV_GPUOWL_CUDA_FFT_PATHS.md`。保留数学推导、第三方源码分析及其条件；源码行号针对原文研究的本地副本，第三方上游可能已有变化。本文不声明本仓库当前支持、默认值或性能。涉及实验数字时只按原文条件理解；当前实现见 [架构](../architecture/OVERVIEW.md)，当前计时口径见 [性能](../performance/STAGE2.md)。

## 1. CUDA 路径本身：`.cl` 内核*就是* CUDA 内核

* **没有独立的 CUDA 实现。** `src/cl/*.cl` 是唯一的内核源码；在 CUDA 上它由 **NVRTC** 在运行时编译，并由在 CUDA *驱动* API 之上重新实现的 OpenCL API 函数驱动 —— `src/cuda/clwrap_cuda.cpp:1-3`：*"This replaces clwrap.cpp when building with the native CUDA backend, mapping all cl* calls to cu* equivalents"*。
  `.cl` 文本在运行时并不从磁盘读取：`genbundle.sh:8-38`（由 `Makefile:113-114` 驱动）把每个 `src/cuda/*.cuh` 与 `src/cl/*.cl` 作为原始字符串字面量嵌入 `src/bundle.cpp`，随后 `src/KernelCompiler.cpp:181-192,302-317` 把它们作为 OpenCL "headers" 交给 `clCompileProgram`。
* **这套转换是文本预处理器加宏/类型 shim，而不是编译器前端。** `cudawrap.cpp:221` 的 `preprocessOpenCL()` 剥掉 `#pragma OPENCL`（`:226-234`），
  把 `#define KERNEL(x)` 改写为 `extern "C" __global__ void __launch_bounds__(x)`（`:239-250`），删除 base.cl 的 OpenCL `typedef`（`:255-273`），把
  `__attribute__((reqd_work_group_size(N,1,1)))` 变成 `__launch_bounds__(N)` 并丢弃 `overloadable`（`:289-314`），把 `(ulong2)(a,b)` 改写为 `make_ulong2(a,b)`（`:316-358`），
  把 `local TYPE NAME[` 转换成 `__shared__ TYPE NAME[`，同时删除参数列表里的 `local`（`:360-401`），把 asm 约束 `"n"` 改成 `"r"`（`:403-415`，NVRTC 要求
  真正的常量），并把 `.a[0]`/`.a[1]` 映射为 `.a.x`/`.a.y`（`:275-287`）。该 shim 即 `src/cuda/opencl_compat.cuh`，被注入每个内核（`clwrap_cuda.cpp:415-418`），
  它提供：`#define __kernel extern "C" __global__`（`:16`）、`__local`/`local` → 空（`:23-24`）、`__constant` → `const`（`:28`，CUDA 的 `__constant__` 不能作为内核
  参数）、`restrict` → `__restrict__`（`:32`）、`get_global_id(d) ((unsigned)(blockIdx.x*blockDim.x+threadIdx.x))`、`get_local_id(d) threadIdx.x`（`:35-41`）、`barrier(flags)`
  → 无条件的 `__syncthreads()`（`:47-52`）、`as_uint2`/`as_double`/`as_ulong2` 重解释（`:183-247`）、向量 `+ - * fma mul_hi`（`:115-274`）、`atomic_max/add/cmpxchg`
  （`:276-279`）、`__asm` → `asm`（`:307`）、`sub_group_broadcast` → `__shfl_sync`（`:310`），以及强制的 `NVIDIAGPU 1` + `HAS_PTX 1200`（`:326-331`）。
* **编译标志与缓存**（`clwrap_cuda.cpp:293-393`）：`--gpu-architecture` 取设备自身的 `sm_XY`（CUBIN，无 JIT），而对这张 NVRTC 已不再认识的 GPU，则取低于它的最新
  `compute_XY`，由驱动 JIT（`:296-303`）。始终带 `-default-device -std=c++17 -w --fmad=true`；`-cl-std`/`-cl-finite-math-only` 被丢弃，而 `--fmad` 被称为安全子集 ——
  *"no flush-to-zero, no reduced-precision division/sqrt"* —— 因为 *"every butterfly is multiply-add pairs"*（`:365-370`）。`--restrict` 被故意**关闭**：
  *"tested but causes GPU read errors — some PRPLL kernels use in-place operations where in/out buffers alias"*（`:372-374`）。请求 `--maxrregcount` 会强制走 PTX
  路径，因为 NVRTC 自带的 ptxas 会忽略它生成的 CUBIN 中的这个上限（`:460-478`）。缓存的二进制是 PTX + CUBIN（`:216-234`、`:236-250`），保留 PTX 是因为 `clCreateKernel`
  要从它读取 work-group 大小与 PDL 等待（`:664-696`）。
* **shim 未模拟的部分**（是硬限制，不是慢路径）：
1. **只有维度 0。** `get_global_id(d)`/`get_local_id(d)` 忽略 `d`（`opencl_compat.cuh:35-41`），尽管 `workDim == 2` 确实会构造 2-D grid（`clwrap_cuda.cpp:872-884`、
  `src/clwrap.cpp:451-462`）。没有任何*活跃*内核读取维度 1 —— 仅有的 `get_group_id(1)` 用法都被注释掉了（`tailmul.cl:24`、`tailsquare.cl:24`）—— 尽管 host 说 Y 坐标会把行号抬高 `WIDTH`（`src/Gpu.cpp:1397-1398`）。
2. **没有动态 local memory。** 启动时始终 `sharedMemBytes = 0`（`:862,869`），而 `local` 参数（NULL `cl_mem`）会变成*空 global 指针*（`:760-765`）：只有静态的 `local TYPE NAME[` 改写有效。
3. **没有 FP64 能力查询。** `clGetDeviceInfo` 没有 `CL_DEVICE_DOUBLE_FP_CONFIG` 分支（`:1116-1252`；未知键返回 `CL_INVALID_VALUE`），于是 `hasFP64()` 吞掉该
  错误并返回 true —— *"every CUDA device has FP64"*（`src/clwrap.cpp:187-194`）。因此 CUDA 上从不定义 `NO_FP64`（`src/Gpu.cpp:496`），`T/T2` 保持为 `double/double2`
  （`src/cl/base.cl:266-272`），类型选择器也从不排除 FP64（`src/FFTConfig.cpp:342-343`）：FFT64 总能编译，即使在 FP64 只有 1/32–1/64 速率的卡上亦然，只能靠计时来避开（`src/tune.cpp:454-484`）。
4. **栅栏丢失其标志，原子操作丢失其顺序/作用域。** `barrier(0)` 与 `barrier(CLK_LOCAL_MEM_FENCE)` 是同一个 `__syncthreads()`（`:47-52`），使 FAST_BARRIER
  （`base.cl:901-908`）失去意义；`memory_order_*`/`memory_scope_device` 被 `#define` 为 0，`atomic_load_explicit(p,order,scope)` 两者都忽略（`:285-303`）；`atomic_max` 是 32 位*无符号*比较（`:277`）。

5. **OpenCL 库的其余部分一概没有**：没有 images/samplers/half/generic address space/`vload`，`clSVMAlloc` 是一个闲置未用的 `cuMemAlloc` 包装（`:1332-1350`），`globalOffset` 被
  忽略（`:873`），而 global size 不是 group size 整数倍时会启动未加掩码的越界项（`:879-884`；仅由一个 `assert` 守着，`src/Kernel.cpp:38`）。

* **`.cl` 这棵树是与该后端共同设计的**，这也是为何这么少的模拟就已足够。`HAS_PTX 1200` 在名义上可移植的 OpenCL 里选择*内联 PTX*：
  `bar.warp.sync`（`base.cl:895-898`）、用于 sub-group 栅栏的 `bar.sync N, count`（`:930-931`），以及用于 programmatic dependent launch 的 `griddepcontrol.launch_dependents/.wait`（`:1005-1015`；sm_90+ 上的 `-use PDL=1`，在 `src/Gpu.cpp:264-272` 中标为 CUDA-only，而 shim 会在 PTX 中检测 `griddepcontrol.wait`，以便用
  `CU_LAUNCH_ATTRIBUTE_PROGRAMMATIC_STREAM_SERIALIZATION` 启动，`clwrap_cuda.cpp:687-696,840-870`）。`#if CUDA_BACKEND` 还会禁用 NVIDIA `__constant` 缓存的权重变通方案
  （`carryfused.cl:166-172`）以及寄存器 `bar.sync` 路径（`base.cl:883-890`）。
* **后果。** 源码在首次运行时于用户机器上编译，需要匹配的 toolkit/驱动（CUDA 13 的 NVRTC 去掉了 `sm_50..sm_72`，
  `clwrap_cuda.cpp:334-350`）；预处理器不认识的东西会以 NVRTC 错误失败，并把源码 dump 到 `prpll_fail_N.cu`（`:438-457`）；group size 就是内核声明的工作组大小，grid 是
  `ceil(global/local)`；而 "does this card do doubles well?" 只能靠实测回答，永远不是查询。

**许可（一段）。** `LICENSE:1-3` 是 *"GNU GENERAL PUBLIC LICENSE Version 3"*；`README.md:47` 声明本项目 *"licensed under the **GNU General Public License
  v3.0**"*。对本文件前提的一处更正：**本仓库自己的根 `LICENSE` 同样是 GPL-3.0**，与 gpuowl 的差异仅在于 FSF URL 和 `<>` 样板占位符。本项目所倚重的 BSD 风格文本属于 **gwnum**
  （`gwnum/readme.txt:86-110`；参阅 [gwnum 分析](GWNUM_POLYMULT.md)），那是一个独立组件。
  因此把 gpuowl/PRPLL 源码拷进 MPA-OpenCl 属于 GPLv3 → GPLv3 兼容：只要保留署名与 GPL 声明、且组合作品仍为 GPLv3，逐字的内核*可以*拷贝。**不可以**做的，是把这些行按宽松许可重新授权，
  或把它们装进一个以 BSD 风格 gwnum 条款分发的组件里 —— 如果本仓库哪天真被重新授权为宽松许可，每一行源自 gpuowl 的代码都必须删除或重写（算法与公式不受版权保护；内核源码受）。

---


## 2. IBDWT：在这里它是什么，以及为何存在

* **代码自己的定义**：输入字在变换前按 2 的分数次幂缩放，变换后再乘以其逆 —— *"Weight is 2^[ceil(qj / n) - qj/n]
  where j is the word index, q is the Mersenne exponent, and n is the number of words."*（`src/cl/fftp.cl:446-448`、`carryfused.cl:861-863`）。`N = NWORDS = ND*2`，其中
  `ND = WIDTH*BIG_HEIGHT`（`base.cl:239-242`）；每字位长 `bitlen(N,E,k) = E/N + isBigWord(N,E,k)`，其中 `extra(N,E,k) = step(N,E)*k % N`、`step(N,E) = N - E%N`
  （`src/state.h:14-17`），因此各字在 `⌊E/N⌋` 与 `⌈E/N⌉` 位之间交替。
* **它为何存在：零填充变得不必要。** 变换长度恰好是 `ND = N/2` 个复数元素、恰好覆盖 `N` 个字 —— 就是这个指数的字数，没有保护段 —— 因为循环回绕*就是* Mersenne
  检验所需的 mod `2^E − 1` 归约（`2^E ≡ 1`），而分数权重使一个并不整除 `E` 的字网格保持精确。这种回绕体现在 host 的 pack/unpack 中：`compactBits()` 跨字边界进位（`src/state.cpp:22-48`），
  `expandBits()` 以 `data[0] += u32(bucket.bits); // carry wrap-around.`（`:105`）结尾 —— 那就是高位的折叠。
* **权重如何施加。** FP64/FP32：实数乘法，表以 `weight − 1` 存储，从而一次 FMA 即可完成 —— `fancyMul(a,b) = fma(a, b, a)`（`math.cl:426-427`），用法如
  `T base = optionalHalve(fancyMul(THREAD_WEIGHTS[me].y, THREAD_WEIGHTS[G_W + g].y))`（`fftp.cl:24-30`）。减半就是翻转一个指数位：*"we use inverse weights between 1.0 and
  2.0 because it allows us to implement this routine with a single OR instruction on the exponent"*（`weight.cl:77-100`）；组内 8 步是 `2^(k/8) − 1` /
  `2^(−k/8) − 1`（`:35-63`），由 `weightStepIndex(i) = i*STEP % NW*(8/NW)` 选取（`:29-30`）。NTT：因为 `2` 是 `Z61`（`Z31`）中的 61 次（31 次）单位根，权重变成
  **循环移位** —— `shr(a,k) = (a>>k) + ((a<<(61-k)) & M61)`、`shl(a,k) = shr(a,61-k)`（`math.cl:1083-1088`）、`adjust_m61_weight_shift(w) = optional_mod(w, 61)`
  （`weight.cl:172-180`）。初始化：`m61_log2_root_two = ((1ULL << 60)/NWORDS) % 61` 与 `m61_bigword_weight_shift = (NWORDS - EXP % NWORDS) * m61_log2_root_two % 61`
  （`fftp.cl:449-454`）；逐元素的移位*与*大/小字标志一起前进，共用一个打包的 64 位计数器（`combo_step`/`combo_bigstep`，`:465-494`）。把它称作 "the 60th root" 的
  注释（`fftp.cl:447`、`carryfused.cl:862`）是笔误 —— 每一次归约都是 mod 61。
* **表在哪里构建 —— 对预期答案的更正。** FP 权重表由 `Gpu::genWeights()` 构建（`src/Gpu.cpp:91-168`，公式在 `:56-70`），作为 `bufWeights`/`bufConstWeights` 上传
  （`:1091-1092`）并绑定为固定内核参数（`:1184-1190`）；`Gpu.cpp:589-595` 把 `FRAC_BPW_HI/LO = (E % N)/N·2^64` 编进每个内核（注意 `bpw--; // bpw must not be an exact value`）。
  它们**不在** `TrigBufCache.cpp` 里，后者唯一相关的内容是 NTT 的根常量（`_h_0`/`_h_1`/`_h_order`）。`src/cl/weight.cl` 只保存那 8 项的步长表与移位调整。使用者：`fftP`（进入时的权重，`src/cl/fftp.cl`）、`fftw`
  （最后的宽度 pass，`src/cl/fftw.cl`），以及 `carry`/`carryFused`（逆权重，`carry.cl:29-42`、`carryfused.cl:890-905`）。对 NTT 而言，变换自身的比例因子被折进同一个移位：
  `weight_shift += log2_NWORDS + 1` *"for the fact that NTT returns results multiplied by 2*NWORDS"*（`carryfused.cl:881-886`）。
* **代价与前提。** 代价是每字几次移位（NTT），或每字进入两次乘法、出去两次乘法（FP）—— 便宜，并且相对于它所取代的零填充是可摊薄的。
  前提：(i) 模数必须是 `2^E − 1`，因为回绕就是归约，且 `EXP` 进入权重的指数（`weight.cl:3`、`fftp.cl:453`）；(ii) 小数部分必须跨字边界进位，故有 `isBigWord`/`bitlen` 与 `state.cpp:105`；(iii) 为配合 twiddle 机制，长度必须是 2 的幂
  （`root_one(n) = h^(2^62/n)`，`TrigBufCache.cpp:824`），对 NTT 由 `// Reject non-power-of-two NTTs` 强制（`src/FFTConfig.cpp:96`）。
* **IBDWT 对非 Mersenne 模数（例如 stage-2 的多项式乘积）有用吗？没有。** 它的全部收益在于回绕*就是*你本来就想要的归约；多项式乘积要的是线性（或刻意 negacyclic）卷积，
  所以你要零填充到 ≥ `2L−1`（或施加 twist），而分数权重的记账毫无所得 —— 它还要求 `mod 2^E − 1`，一般性的 stage-2 模数没有可喂给 `weight.cl:3` 的 `E`。可迁移的不是权重，而是这套纪律：逐字位长
  是变化的、一个整数计数器同时产生移位/twiddle 流与大/小字标志，以及把逆权重折进 carry 内核而不是单独一趟。

---


## 3. FFT64（浮点）

* **元素类型与布局**：`typedef double T; typedef double2 T2;`（`base.cl:267-268`）；一个 `T2` 装两个字，因此缓冲区是 `N/2` 个复数元素 = `8N` 字节
  （`src/Gpu.h:390`：`FP64_DATA_SIZE = PAD_ADJUST(W*M*H*2,…)`，单位 `sizeof(double)`）。Host I/O 的字是 4 字节（`WordSize == 4` → `Word2 = int2`，`base.cl:315-324`；
  类型→`WordSize` 见 `src/FFTConfig.cpp:297-305`）。
* **radix / middle / width 结构**：对长度 `ND` 的复数变换做 3-D 分解，`ND = WIDTH*MIDDLE*SMALL_HEIGHT`，`N = 2ND`。宽度与高度 pass 使用 `NW`/`NH`
  ∈ {4,8}（`src/FFTConfig.h:44-46`），一个由 `G_W = WIDTH/NW` 个线程组成的工作组各持 `NW` 个元素，循环
  `for (u32 s = 1; s < WG; s *= RADIX) { fft_RADIX(u); tabMul(...); shufl(...); }` 外加最后一次 `fft_RADIX(u)`（`base.cl:1054-1060`；`WG`/`RADIX` 即 `G_W`/`NW`，
  `src/cl/fftwidth.cl:4-14`）。`MIDDLE` 在 middle 内核中就是每线程一个大小 2/4/8/16 的蝶形（`fft-middle.cl`、`fftmiddlein/out`）；twiddle 的索引为
  `trig[(i-1)*WG + (me & ~(f-1))]`（`base.cl:1915-1933`）。
* **trig 来自两套彼此独立的机制。** (a) 在 `TrigBufCache.cpp` 中按 shape 生成的设备端表（`genSmallTrigFP64:141`、combo/tail `:242`、middle `:283`、分派
  `:931-1011`）。(b) 在内核内求值的 8 项多项式 `T2 reducedCosSin(int k, double cosBase)`（`src/cl/trig.cl:7-41`），由从 `trigCoefs(fft.shape.size()/4)`（`src/Gpu.cpp:514-517`）编入的 `TRIG_SCALE/TRIG_SIN/TRIG_COS`
  驱动，minimax 表 `COS[7]`/`SIN[7]` 在 `src/Trig.cpp:31-49`；`trigCoefs` 断言该 shape 可分解为
  `mid·2^twos` 且 `mid ≤ 15 || mid = 625k`（`:77-91`），而 `slowTrig_N` 把 `k` 折进 `[0, n/8]` 并做符号/交换修正（`trig.cl:44-71`）。用哪一种由 `-use TAIL_TRIGS` 决定，
  默认 2 = 计算，无访存（`TrigBufCache.cpp:1065`）。
* **舍入误差控制。** 一个被进位的元素只有在量测过它到舍入边界的距离之后才成为整数：
  `float roundoff = fabs((float) fma(u, invWeight, RNDVALCarry - d)); *maxROE = max(*maxROE, roundoff);`（`carryutil.cl:208-227`）；0.5 意味着整数取错。这棵树里**没有
  `gw_passes_safety_margin`**；与之等价的是 `fftbpw.h` 的表、`fft.maxExp() < E` 警告（`src/Gpu.cpp:1130-1132`）、`bitsPerWord < minBpw()`
  抛出的 *"FFT size too large"*（`:1135-1138`；`minBpw() = 3.0`，`src/FFTConfig.h:48`），以及 sloppy-carry 的截断 `MAXBPW = maxBpw()*100`（`src/Gpu.cpp:505`）配上
  `#define SLOPPY_MAXBPW (MAXBPW - 110)` —— *"We only allow sloppy results when not near the maximum bits-per-word"*（`carryutil.cl:695-702`）。低于 10.0
  bpw 时强制 `useLongCarry`（`src/Gpu.cpp:1140`），且 `EXP/NWORDS >= 19` 时 32 位进位是非法的（`carryutil.cl:807-819`）。
* **可达 bpw**（`src/fftbpw.h`；每个 key 的 6 个值是变体 000,101,202,010,111,212）：

| key | 引用的 bpw | 每字节位数（8 B/word） |
|---|---|---|
| `256:2:256` (`:2`) | `19.204 19.547 19.636 19.204 19.547 19.636` | 2.40 – 2.45 |
| `512:8:512` (`:38`) | `18.256 18.280 18.314 18.319 18.369 18.444` | 2.28 – 2.31 |
| `4K:16:1K` (`:94`) | `16.744 16.887 16.966 16.921 17.048 17.208` | 2.09 – 2.15 |

在固定 shape 下，一个变体值约 0.35 bpw（≈1.8 %）。蝶形开销按仓库的记法 `2·FMA + ADD` 列表：radix-4 `0 FMAs + 16 ADDs`（`fft4.cl:15`），radix-8
  `4 MUL + 52 ADD`（`fft8.cl:25`；开销在 `FFT.md:12,16` 中为 16 vs 60）。

---


## 4. NTT61、NTT31 与混合型

它们与 FFT64 共用内核，由 `FFT_TYPE`、`NTT_GF31`、`NTT_GF61`、`FFT_FP32`、`FFT_FP64`、`WordSize` 切换（`src/FFTConfig.cpp:297-305`、`src/Gpu.cpp:931-961`）。类型
  （`base.cl:273-278`）：`Z31=uint`、`GF31=uint2`（8 B）、`Z61=ulong`、`GF61=ulong2`（16 B）、`F2=float2`（8 B）、`T2=double2`（16 B）；一个元素装**两个**字。

| `FFT_TYPES` | 计算内容 | 元素（域） | B/元素 | `WordSize` | I/O B/word |
|---|---|---|---|---|---|
| `FFT64` 0 | FFT | `double2` | 16 | 4 | 8 |
| `FFT32` 53 | FFT | `float2` | 8 | 4 | 4 |
| `FFT31` 52 | NTT | `GF(M31²)` `uint2` | 8 | 4 | 4 |
| `FFT61` 3 | NTT | `GF(M61²)` `ulong2` | 16 | 4 | 8 |
| `FFT3161` 1 | 两个 NTT + CRT | `GF(M31²)`+`GF(M61²)` | 8+16 | 8 | 12 |
| `FFT3261` 2 | FFT + NTT | `float2`+`GF(M61²)` | 8+16 | 8 | 12 |
| `FFT6431` 51 | FFT + NTT | `double2`+`GF(M31²)` | 16+8 | 8 | 12 |
| `FFT3231` 50 | FFT + NTT | `float2`+`GF(M31²)` | 8+8 | 4 | 8 |
| `FFT323161` 4 | FFT + 两个 NTT + CRT | `float2`+`GF(M31²)`+`GF(M61²)` | 8+8+16 | 8 | 16 |

字节/字由 `src/Gpu.h:390-395` 推出（单位 `sizeof(double)`；每个分量贡献 `PAD_ADJUST(W*M*H*2,…)`，再按 `sizeof(element)/sizeof(double)` 缩放：FP64 `8N`、FP32
  `4N`、GF31 `4N`、GF61 `8N`）。`WordSize` 是*host I/O* 宽度，不是元素宽度 —— FFT61 每 2 个字存 16 B，而读写的字是 4 字节的 `int2`（`FFTConfig.cpp:300`、
  `base.cl:316-324`）。

* **M31 与 M61 在 carry 阶段、同一趟里用 CRT 重新组合。** `weightAndCarryOne(Z31,Z61,…)` 先把两个逆权重作为移位施加，然后（`carryutil.cl:446-478`）：
```c
u32 n31 = get_Z31(u31);
u61 += make_u64(hi32(M61), lo32(M61) - n31);   // u61 - u31
u61 += shl(u61, 31);                           // u61 + (u61 << 31)
i64 n61 = get_balanced_Z61(modM61(u61));
i96 value = make_i96(n61 >> 1, ((u32)n61 << 31) | n31);   // n61*M31 + n31
```
一个 92 位的 `n61·M31 + n31`，其乘法被折成移位（注释 `:452-459` 归功于 Gallot 的 `mersenne2`）。`FFT323161` 加入 FP32 分量来选择
  `M31·M61` 的倍数：*"Use FP32 data to calculate how many multiples of M31*M61 need to be added to n3161"*（`:487-524`，128 位重组）。两个 NTT 跑在同一个缓冲区上，GF61 数据位于
  偏移 `DISTGF61`（`src/Gpu.cpp:565-575`），各有自己的 weight-shift 流。
* **长度上限 —— `v2(p−1)` 论证。** 对 `p = M61 = 2^61−1`，`p−1 = 2·(2^60−1)` 的 `v2 = 1`，因此 `F(M61)` 中除 `−1` 外不含 2 的幂次单位根：单靠 `F_p`
  无法承载长度 `2^k` 的变换。代码工作在二次扩域中 —— `TrigBufCache.cpp:791-800`：*"GF((2^61 - 1)^2): the prime field of order p^2"* —— 并有
  `static const uint64_t _h_order = uint64_t(1) << 62;` 与 `static GF61 root_one(const size_t n) { return GF61(Z61(_h_0), Z61(_h_1)).pow(_h_order / n); }`，
  `_h_0 = 264036120304204`、`_h_1 = 4677669021635377`（`:796-800,824`），与 `v2(p²−1) = v2(2^62·(2^60−1)) = 62` 吻合。M31 的对应物是 `_h_order = 1 << 32`、`_h_0 = 7735`、
  `_h_1 = 748621`（`:618-620,644`），对应 `v2(31²−1) = 32`。这棵树陈述了这些*阶*，却从未写下这一推导（这是我自己做的；常量佐证了它）。从操作层面看，
  每种 NTT 类型都要求 `MIDDLE` 为 2 的幂（`FFTConfig.cpp:96`），且 `root_one(n)` 用 `n` 去除 `2^62`；可达长度是 `ND ≤ 2^26`（`WIDTH ∈ {256,512,1024,4096}`，
  `fftwidth.cl:20-22`；`HEIGHT ∈ {256,512,1024}`，`MIDDLE ∈ 2..16`，`FFTConfig.cpp:92-104`），在 `2^62`（M61）之内，也在 `2^32`（M31）之内。
* **实测 bpw 与每字节载荷**（前缀 = FFT 类型编号，`FFTConfig.h:51`）：

| 类型 | `…:256:2:256` | bpw | 位/字节 | `…:4K:16:1K` | bpw | 位/字节 |
|---|---|---|---|---|---|---|
| FFT64 | `256:2:256` (`:2`) | 19.204 – 19.636 | **2.40 – 2.45** | `4K:16:1K` (`:94`) | 16.744 – 17.208 | 2.09 – 2.15 |
| FFT3161 | `1:256:2:256` (`:96`) | 40.54 | **3.38** | `1:4K:16:1K` (`:113`) | 37.12 | 3.09 |
| FFT3261 | `2:256:2:256` (`:115`) | 34.53 | 2.88 | `2:4K:16:1K` (`:132`) | 28.54 | 2.38 |
| FFT61 | `3:256:2:256` (`:134`) | 25.02 | 3.13 | `3:4K:16:1K` (`:151`) | 22.42 | 2.80 |
| FFT323161 | `4:256:2:256` (`:153`) | 50.01 | 3.13 | `4:4K:16:1K` (`:170`) | 44.12 | 2.76 |
| FFT3231 | `50:256:2:256` (`:172`) | 19.57 | 2.45 | `50:4K:16:1K` (`:189`) | 7.05 | 0.88 |
| FFT6431 | `51:256:2:256` (`:191`) | 35.27 | 2.94 | `51:4K:16:1K` (`:208`) | 32.25 | 2.69 |
| FFT31, FFT32 | — | **无条目** | — | — | **无条目** | — |

每字节最密的是：**FFT3161（M31+M61）≈3.38 位/字节**，其后是 FFT61 与 FFT323161 的 ≈3.13；FFT64 最差（2.40），却常常在 wall-clock 上最快，因为这个指标不计
  算术。FFT31/FFT32 未经标定：既不在表里*也*不在 `allShapes()` 里（`FFTConfig.cpp:92`），而在未映射的 shape 上构造函数会回退到
  `bpw = {18.1f,…}` 并给出 *"ERROR: BPW info for %s not found, using default of 18.1"*（`:165-169`）—— 一个源自 FP64 的默认值，若它成立，对 FFT31 的 4 B/字就意味着 4.5 位/字节，
  是所有选项里最好的。没有任何东西测量过它。
* **这些表是经验性的，不是证明。** M61 那一节记录了唯一有据可查的硬失败：*"LL of 100028317 failed (ROEmax=0.294, ROEavg=0.247). Lowering bpw from
  23.94 to 23.84."*（`fftbpw.h:141`，行 `3:1K:8:256`）。对 NTT 类型而言，"ROE" 不是浮点误差，而是重建出的系数与 `M61/2`（相应地
  `M31/2`）回绕边界的接近程度 —— `u32 roundoff = (u32) abs((i32) hi32(value));`，并配以 *"calculate roundoff error as proximity to M61/2. 28 bits of accuracy should be sufficient"*
  （`carryutil.cl:322-326`），之后由 `(float) roundMax / (float)(M61 >> 32)` 归一化（`carryfused.cl:958`；M31 为 `/(float)M31`，`:723`）。因此一个字只有在系数和连同该表所编码的
  统计余量都留在模数窗口内时，才精确承载 `bpw` 位；当情况不再如此，该条目就被调低。

---


## 5. 真正重要的差异

| | FFT64 | FFT61 | FFT3161 (M31+M61) | 混合型 (3261/6431/3231/323161) |
|---|---|---|---|---|
| 正确性 | 近似：FP64 + ROE 判据（`carryutil.cl:208-227`） | 在其窗口内 **mod M61 精确** | **精确**，CRT `n61·M31+n31`（`:446-478`） | 精确，窗口被 FP 部分拓宽（`:487-524`） |
| 进位 | < 19 bpw 时允许 32 位（`FFTConfig.cpp:329`） | 必须 64 位 | 必须 64 位 | 必须 64 位（`carryutil.cl:807-819`） |
| 长度 | 仅由 shape 空间限定为 `2^26` | `2^62`（`TrigBufCache.cpp:800`） | `2^32`（`:620`） | 与其 NTT 成员相同 |
| radix-8 蝶形 | `4 MUL + 52 ADD` FP64（`fft8.cl:25`） | GF61 `cmul` = 3 次宽乘法（`math.cl:1241-1246`），惰性归约（`fft8.cl:144-151`） | 两者皆有，同一个内核 | FP + NTT 混合 |
| 字节/字 | 8 | 8 | 12 | 8–16 |
| 位/字节 @ `256:2:256` | 2.40 | 3.13 | **3.38** | 2.45 – 3.13 |
| 最佳场景 | FP64 快的 GPU（`-tune` 会把它与 M31+M61 比较，阈值 0.80/1.20，`tune.cpp:454-484`） | 最密的*单一*精确路径，LDS 最少 | 每字节最密，不需要 FP64 | FP32 便宜时的 3231/3261；指数最大时用 323161 |

* **长度正是单一素数失手之处。** 随着变换变大，`FFT31` 的表崩塌（`50:4K:16:1K` → 7.05 bpw，`fftbpw.h:189`）：起约束作用的是系数窗口，而不是
  算术。`GF(M61²)` 同时拥有最深的 2-adic 阶（62）与最宽的单模数（61 位），这就是为什么它是 `-tune` 拿来与 FP64 对比的默认 NTT。
* **访存流量由内核结构决定，而不是元素类型。** 一次平方是 `fftP → fftMidIn → tailSquare → fftMidOut → carryFused`（`src/Gpu.cpp:2256-2283`）：对变换缓冲区约 4
  趟完整的读+写，相比之下朴素 1-D Stockham 循环需要 `log2(ND) ≈ 17-26` 趟。逐点乘积位于 tail 内核*内部*，夹在两个半高 pass 之间（`tailsquare.cl:1072-1127`）。
* **tail 配对是平方的折扣，而它不能迁移到一般乘积。** `pairSq` 为一条 Hermitian 线对计算 `csqq(a)`、`csq(b)` 与交叉项 `ab`
  并用 `t_squared` twiddle 把它们折起来（`tailsquare.cl:892-991`）；第 0 行与第 H/2 行与自身配对，多一个因子 `TAILTGF61`（`:1099-1110`；配对关系
  `line2 = line1 ? H - line1 : H/2`，`:1063-1064`）。一般乘积的孪生版本是存在的 —— `pairMul`/`tailMul`（`src/cl/tailmul.cl:50-86`，由 `Gpu::mul` 驱动，
  `src/Gpu.cpp:2003-2016`）—— 而且是*更难*的内核：`onePairMul` 每个输出都需要真正的 `cfma`/`cmul`，而 `pairSq` 复用 `a²`、`b²` 与 `ab`。
* **carry 与重建是最不可复用的部分**：一条贯穿整个数的进位链，`nBits = EXP/NWORDS + (isBigWord?1:0)`（`carryutil.cl:547-579`、`state.h:16-17`），
  进位通过一个带 `write_mem_fence` + `atomic_store(ready)` + 自旋的全局 shuttle 在工作组之间传递（`carryfused.cl:915-1040,1726-1755`），以及 LL 的初始 `−2`
  （`:904-907`）—— 全都专为 `X² mod (2^p−1)` 定制。

### stage-2 引擎应从中吸取什么

* **(a) 用于 Kronecker 打包乘积、数字宽约 20-30 位的域：不能只用 M61。** 在实际 shape 下 `FFT61` 给出约 22–25 位/字（`fftbpw.h:134,151`），而那些数字是
  统计性的，且在 23.94 bpw 有一次记录在案的失败（`:141`）。两个长度 `L`、`d` 位数字的多项式的精确乘积必须覆盖最坏情况 `log2 P > log2 L + 2d`
  （此处由模数推导，非引用）。取 `P = M61`（约 61 位）可得 `d < (61 − log2 L)/2`：`L = 2^12` 时 `d ≈ 24`，`L = 2^16` 时 `d ≈ 22`，没有 30 位的余地。
  取 `P = M31·M61`（约 92 位，在 `carryutil.cl:446-478` 做 CRT）则任何 `d ≤ 30` 都能用到 `L = 2^32`。所以：**对 20-30 位数字，用 M31+M61 对（即 FFT3161 的域对）**；
  只有短长度且 ≤ ~24 位数字时才单用 M61；**绝不要单用 M31**（它在 `4K:16:1K` 下的 7.05 bpw 就是警告）。
* **(b) 效仿其分解方式，而不是其启动拓扑**：width×middle×height 的划分，其中只有转置触碰 global memory（`fftMiddleIn`/`fftMiddleOut`）；一组
  多 radix 蝶形，其中 `×i`/`×√½` 以交换/取负/延迟缩放实现（`fft4.cl:16-61`、`fft8.cl:16-23`）；twiddle 或查表、或来自 8 项多项式
  （`trig.cl:7-41`）；以及对于一般乘积，采用 `tailMul` 的形状（`tailmul.cl:50-86`），使逐点乘法发生在变换内部。不要把 `pairSq` 的平方
  捷径抄到双多项式乘积上。
* **(c) 不要采用**：IBDWT（需要 `mod 2^E−1` 以及一个你并不想要的回绕 —— `weight.cl:3`、`state.cpp:105`）；`n61·M31+n31` 的 CRT 进位（`carryutil.cl:446-478`，其
  乘改移位与惰性 `[0, 2M61+ε]` 区间是针对一对模数和一条进位链手工推导的，`state.h:16-17`）；LL 的初始进位 `−2`
  （`carryfused.cl:904-907`）；以及 sloppy-carry 截断（`carryutil.cl:695-702`），它以最坏情况正确性换速度。

---


## 6. 未解问题（未能从源码判定）

* **为何 NTT 的 bpw 表高到这种程度。** 长度 `ND` 的循环卷积的最坏情况系数约为 `~ND·2^(2·bpw)`；在 `3:256:2:256` 下那是
  `2·25.02 + 17 ≈ 67` 位，而 `M61` 窗口只有 61 位，然而表却声称 ROE ≈ 0.35。经验性的框定加上那次记录在案的失败（`fftbpw.h:141`）暗示存在一个统计
  判据，但没有任何地方写下推导、分布模型或安全系数依据（已搜索 `src/`、`src/cl/*.cl`、`FFT.md`、`README.md`、`z.txt`）。因此
  "how many bits `M61` can carry at length `L`" 这条规则**仍未确立**；§5(a) 是我的保守重建，不是这棵树的规则。
* **2-D tail 启动中的 Y 维度。** `Gpu.cpp:1398` 说 Y 坐标会把行号抬高 `WIDTH`，而 host 传入 `kernelsToExecuteY = (MIDDLE+1)/2`
  （`:1399-1411`），但内核侧的使用都被注释掉了（`tailmul.cl:24`、`tailsquare.cl:24`），而活跃代码只从 `get_group_id(0)` 推导行号（`tailmul.cl:13-43`）。
  这些 Y 块究竟是重复了工作（它们写入相同的值，所以不是正确性 bug）还是在别处被消费，我无法判定；该路径需要
  `-use L2_STRIPING=n`（`base.cl:232-234` 默认把它设为 0）。
* **host 如何解释 NTT 的 ROE 采样**：`bufROE` 是 `Buffer<float>`（`src/Gpu.h:216`），而 NTT 内核写入的是 `u32` 接近度计数，有时被归一化
  （`carryfused.cl:723,958`），有时则通过隐式 `u32 → float` 转换原样写入（`carryfused.cl:259,1764`、`carry.cl:51`）。哪些位置是活跃的，我无法确立。
* **这棵树里不存在实测的速度对比**：`z.txt` 没有 NTT 行，也没有内建的 FP64-vs-M31+M61 计时 —— 它在运行时测到 `tune.txt`
  （`tune.cpp:454-484`）。任何 "which path is faster" 的说法都必须实测。
* **这里的 `bits/byte` 是载荷，不是分配量**：`PAD_ADJUST`（`src/Gpu.h:376-389`）可能依 `INPLACE`/`PAD` 把真实缓冲区膨胀最多 1.6×。而且没有任何东西被
  构建或运行，所以上面每一个常量都是引用来的，而非复现的。
