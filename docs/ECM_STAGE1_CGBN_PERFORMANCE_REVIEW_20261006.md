# ECM Stage1 / CGBN 性能审查与优化路线

日期：2026-10-06。审查基点：`cd5cdee98a6320465c12fd617cbe9d95f0f60449` 加当时工作区；本文涉及的 Stage1 源码未修改，关键文件摘要见附录。

本文以 **CUDA Suyama param0 → 有效 Stage1 save → 当前 CUDA Stage2** 为主要应用场景，兼顾 param2、param3、CPU 与 OpenCL。方法是源码审查、既有实验复核及算术成本建模。本轮没有重新编译、运行 GPU、执行测试或更改实现；下文的“建议验收”是后续实施方案。

后续实测：[PRAC 实施](D:/code/MPA-OpenCl/docs/ECM_STAGE1_PRAC_IMPLEMENTATION_20261006.md:1)、[TPI/寄存器与 Nsight](D:/code/MPA-OpenCl/docs/ECM_STAGE1_TPI_REGISTER_TUNING_20261006.md:1)。本文下述“尚未实现”描述的是审查时的状态。

## 1. 结论与实施顺序

**可以继续优化，但需要按 B1、位宽和曲线批量分别选择方向。** 当前 Stage1 是曲线间并行的 Montgomery ladder；大 B1 时，主要成本随指数位数增长，普通分片没有反复进行大规模 CPU/GPU 数据传输。Stage2 的传输瓶颈不能直接作为 Stage1 的诊断结果。

建议顺序如下：

1. **建立正确的计时口径，按 param0 重新校准批量、TPB、TPI 和寄存器限制。** 复用现有旋钮，实施成本最低；小批量欠填充已有数据支持，旧生产形状收益需要在归一化修复后的基线上复测。
2. **针对短任务优化准备和收尾。** 提取循环不变量、缓存同 B1 指数、批量求逆、只回传最终 X/Z。对大 B1，这些固定成本通常占比更小。
3. **保留分片间的 Montgomery 域状态，并整理同步与计时。** 当前每个分片重复执行六次入域、四次出域；可明确消除重复工作，但收益必须先测转换占比，不能直接承诺大幅提速。
4. **专用平方是可复用的算术优化候选。** CGBN 当前把平方当一般乘法。param0 的理想算子成本改善约 10%，实际收益会被归约、进位、shuffle 和寄存器开销压缩。
5. **重新研究适用于 param0 的 PRAC / 差分链。** 旧文档按较廉价 param3 梯子作出的全面否定不适用于 `6M+4S` 的 param0。条件成立时可减少算术量；目前还没有正确的现行 GPU 实现或有效生产 A/B。
6. **梅森折叠域、细化大位宽档位、Karatsuba 属于后续方向。** 折叠域曾有正负两类结果，且收益依赖批量形状；应在正确基线上验证，并设计通用模数回退。

不建议重新把“删除归一化”“错误链探针”“只缓存指数位”“强切 XMAD”当作近期提速方案。

## 2. 当前实现的分工

### 2.1 公共驱动与后端

公共驱动先生成标量、准备后端，再运行整批曲线。主要入口是 [ecm_driver.cpp:2770](D:/code/MPA-OpenCl/src/core/ecm_driver.cpp:2770)，GPU 调用是 [ecm_driver.cpp:2904](D:/code/MPA-OpenCl/src/core/ecm_driver.cpp:2904)。

- CPU Montgomery：GMP 参考路径，以及 IFMA 的 SoA 批量路径。倍点见 [ecm_mont_cpu.cpp:103](D:/code/MPA-OpenCl/src/cpu/ecm_mont_cpu.cpp:103)，有状态 Stage1 见 [ecm_mont_cpu.cpp:178](D:/code/MPA-OpenCl/src/cpu/ecm_mont_cpu.cpp:178)。SIMD 倍点、差分加和批量 ladder 分别见 [simd_mont_curve.cpp:115](D:/code/MPA-OpenCl/src/cpu/simd_mont_curve.cpp:115)、[simd_mont_curve.cpp:133](D:/code/MPA-OpenCl/src/cpu/simd_mont_curve.cpp:133)、[simd_mont_curve.cpp:187](D:/code/MPA-OpenCl/src/cpu/simd_mont_curve.cpp:187)。
- CUDA：驱动接到 CGBN Stage1，见 [ecm_cuda_backend.cu:291](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_backend.cu:291)。该适配层在 299–302 行忽略 OpenCL 的算子路径参数，因此更改 `--mul` 等 OpenCL 配置不会替换 CUDA 模乘。
- OpenCL：另有独立的算子注册、源码拼装及 private/local/cooperative/sliced 内核，见 [opencl_ecm_stage1.cpp:620](D:/code/MPA-OpenCl/src/opencl_ecm_stage1.cpp:620)、[opencl_ecm_stage1.cpp:801](D:/code/MPA-OpenCl/src/opencl_ecm_stage1.cpp:801)、[opencl_ecm_stage1.cpp:1097](D:/code/MPA-OpenCl/src/opencl_ecm_stage1.cpp:1097)。函数虽沿用 `cgbn_ecm_stage1` 名称，这条路径没有调用 NVIDIA CUDA 的 CGBN WMAD core。

当前 OpenCL 基础步使用 `d` 的特殊乘法及固定差值乘 2，见 [ecm_stage1.cl:3](D:/code/MPA-OpenCl/kernels/opencl/ecm_stage1.cl:3)。不能默认它与 CUDA param0 是同一曲线族。CPU Edwards 也应作为另一种曲线实现独立评估。

### 2.2 CUDA 的三种曲线族

| 曲线族 | 每个普通标量位的主要算子 | 常数条件 | 当前输出与适用性 |
|---|---|---|---|
| param0 / Suyama | `6M + 4S` | 完整位宽 `a24`、完整位宽仿射差值 `xdiff` | 与 CPU Suyama 对应；64 位 sigma；本报告主要优化对象 |
| param2 | `5M + 4S` | 完整 `a24`；`xdiff=2` 用移位 | 另一曲线族，save 标记 `PARAM=2`；不能作为相同曲线的 param0 A/B |
| param3 | `4M + 4S + U` | `a24` 来自 32 位参数的特殊乘法；`xdiff=2` | 驱动默认值仍是 3；较便宜的步不能直接替代要求 param0 的流程 |

`M` 表示完整位宽模乘，`S` 表示模平方，`U` 表示 param3 的特殊常数乘法；表内不计加减、规范化和交换。这是算子数，不是测得的周期数。

源码依据：[param3 步:419](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_kernel.h:419)、[Suyama 步:816](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_kernel.h:816)、[param2 常数差值:884](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_kernel.h:884)。默认 `gpu_param=3` 见 [ecm_driver.cpp:910](D:/code/MPA-OpenCl/src/core/ecm_driver.cpp:910)；param0 / param2 save 分支见 [ecm_driver.cpp:2932](D:/code/MPA-OpenCl/src/core/ecm_driver.cpp:2932) 和 [ecm_driver.cpp:2945](D:/code/MPA-OpenCl/src/core/ecm_driver.cpp:2945)。当前 native Stage2 只接受 param0，见 [ecm_cuda_stage2_main.cpp:132](D:/code/MPA-OpenCl/src/core/ecm_cuda_stage2_main.cpp:132)。

## 3. CUDA Stage1 的完整步骤

### 3.1 构造公共标量

对整数界限 `B = floor(B1)`，构造

\[
s=t\operatorname{lcm}(1,\ldots,B)
 =t\prod_{\ell\le B,\ \ell\text{ 为素数}}\ell^{\lfloor\log_\ell B\rfloor},
\qquad t\in\{1,12\}.
\]

入口和范围检查见 [ecm_driver.cpp:263](D:/code/MPA-OpenCl/src/core/ecm_driver.cpp:263)，缓存调用见 [ecm_driver.cpp:275](D:/code/MPA-OpenCl/src/core/ecm_driver.cpp:275)，GPU 的 `choose12` 接线见 [ecm_driver.cpp:2828](D:/code/MPA-OpenCl/src/core/ecm_driver.cpp:2828)。

指数构造已经使用分段奇数筛、小累加器和两两合并的乘积树，并有磁盘缓存。开发机既有记录为：`B1=260e6`、指数约 375.1 Mbit，冷构造约 10.7 s，验证后缓存读取约 0.26 s；见 [ecm_stage1_exp.h:37](D:/code/MPA-OpenCl/src/core/ecm_stage1_exp.h:37)。这是该机器历史结果，不是本轮测量。

同一批曲线只使用一个指数；当前仍把它复制到 `params->batch_s`，见 [ecm_driver.cpp:2834](D:/code/MPA-OpenCl/src/core/ecm_driver.cpp:2834)，随后导出为 32 位字数组，见 [cgbn_stage1.cu:345](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1.cu:345)。磁盘缓存已经解决重复构造，进程内值缓存及设备端指数复用仍可研究。

`choose12` 将标量乘 12，只多约 3–4 个指数位，**不是运行十二次 Stage1**。必须在对照实验中使用相同 `t`，因为最终点会变化。B2 不进入这个 Stage1 标量或 GPU bit loop。

### 3.2 CPU 构造各条曲线和初始点

param0 在 CPU 上逐曲线用 GMP 计算：

- `u=σ²−5`，`v=4σ`；
- 曲线参数的分母 `4u³v` 及其逆；
- `a24=(A+2)/4`；
- `P=(u³:v³)`、仿射差值 `xdiff=u³/v³`；
- 一个倍点得到 `2P`。

随后每曲线打包七个 **K 位大整数槽位**：`N,a24,xdiff,aX,aZ,bX,bZ`，不是七个 32 位字。见 [cgbn_stage1.cu:435](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1.cu:435)、[cgbn_stage1.cu:456](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1.cu:456)、[cgbn_stage1.cu:517](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1.cu:517)。

可直接辨认的重复计算：每条曲线都求 `4^-1 mod N`；`u³` 先用于分母，后又计算一次用于 X0。见 [cgbn_stage1.cu:486](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1.cu:486)、[cgbn_stage1.cu:492](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1.cu:492)。param3 还在循环内重复计算公共 `2^32` 的逆，见 [cgbn_stage1.cu:396](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1.cu:396)。

当前遇到不可逆分母会记录警告并构造退化状态，见 [cgbn_stage1.cu:472](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1.cu:472)、[cgbn_stage1.cu:495](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1.cu:495)。批量求逆必须保留或明确改进这些情况的因子处理，不能假定输入都是可逆元。

### 3.3 选择编译档位并一次性上传

选择第一个存在内核且满足 `K ≥ bitlength(N)+6` 的档位，见 [cgbn_stage1.cu:1355](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1.cu:1355)。

完整构建中的主要档位是：

- TPI=4：128、192、256、384、512；
- TPI=8：768、1024、1280、1536、1792、2048；
- TPI=16：2560 至 8192，每 512 位一档；
- TPI=32：9216 至 16384，每 1024 位一档。

实际列表见 [cgbn_stage1.cu:156](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1.cu:156) 和 [cgbn_stage1.cu:201](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1.cu:201)。头文件中的额外 typedef 不代表 dispatch 已编译该档位；开发构建也不能代表完整构建。

当前 TPB 默认为 128。一条曲线由 TPI 个线程协作，一个 block 处理 `TPB/TPI` 条曲线。指数先上传一次，曲线数组也上传一次，见 [cgbn_stage1.cu:1214](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1.cu:1214)、[cgbn_stage1.cu:1535](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1.cu:1535)。

### 3.4 分片执行 ladder

初始 `(P,2P)` 已处理最高位；此后每位执行一次差分加和一次倍点。所有曲线读取同一指数位，分支序列一致。点数值不同仍会影响规范化等局部操作，不能由此声称整个内核严格等周期。

param0 每个分片：

1. 读七个大整数槽位，见 [kernel:1066](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_kernel.h:1066)。
2. 对四个点坐标、`a24`、`xdiff` 作六次 `bn2mont`，见 [kernel:1084](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_kernel.h:1084)。
3. 在寄存器中的分布式大整数上运行本分片所有指数位，见 [kernel:1107](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_kernel.h:1107)。
4. 对四个点坐标作 `mont2bn`，存回四个槽位，见 [kernel:1148](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_kernel.h:1148)。

主机初始分片为 200 bits，随后按时间动态调整至约 100 ms，见 [cgbn_stage1.cu:1612](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1.cu:1612)、[cgbn_stage1.cu:1742](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1.cu:1742)。每片设备同步、读取 CGBN 错误报告，再记录 stop event 并再次等待，见 [cgbn_stage1.cu:1670](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1.cu:1670)。

**普通分片只在 GPU 显存与寄存器之间读取/写回状态。** 整批 D2H 出现在 checkpoint、启用逐片 dump 或最终收尾，不能把每片七个槽位的设备读取算作 PCIe 传输。

### 3.5 checkpoint 与最终输出

checkpoint 复制整批设备状态，并保存位偏移等信息，见 [cgbn_stage1.cu:1753](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1.cu:1753)。当前版本是 v5，见 [cgbn_stage1.cu:241](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1.cu:241)，v4 被拒绝；不能在改变状态表示后继续沿用相同版本解释数据。

最终 D2H 仍回传全部五/七槽位，见 [cgbn_stage1.cu:1785](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1.cu:1785)。CPU 逐曲线导入 X/Z、求逆得到仿射 x；Z 不可逆时取 gcd。见 [cgbn_stage1.cu:292](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1.cu:292)、[cgbn_stage1.cu:764](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1.cu:764)。

param0 的 save 使用原始 N、原始 64 位 sigma 和归一化 x；因子命中行按相应规则处理。见 [ecm_driver.cpp:2932](D:/code/MPA-OpenCl/src/core/ecm_driver.cpp:2932)。优化必须保持这些输出语义。

## 4. CGBN 内部成本与瓶颈

### 4.1 分布式整数与 padding

定义存储字数 `w=K/32`，TPI 为 `T`。CGBN 每线程分配

\[
L=\left\lceil\frac{w}{T}\right\rceil,
\qquad W=TL,
\qquad K_{\mathrm{eff}}=32W.
\]

见 [cgbn_cuda.h:107](D:/code/MPA-OpenCl/cgbn/include/cgbn/cgbn_cuda.h:107)。`K` 决定存储字节数，`W` 更接近乘法循环的规模，两者不能混用。CGBN 支持线程组协作的固定宽度整数，官方概述见 [NVlabs/CGBN](https://github.com/NVlabs/CGBN)。

当前 load/store 按每个 lane 连续 L 个 limbs 分配，见 [impl_cuda.cu:1376](D:/code/MPA-OpenCl/cgbn/include/cgbn/impl_cuda.cu:1376)。单条 load 指令跨 lane 的地址不一定连续，但状态只在片首尾访问；没有 DRAM profiler 数据时，不应把这认定为首要瓶颈。

### 4.2 WMAD 模乘

在 sm_70 及以上选择 WMAD，见 [cgbn.h:67](D:/code/MPA-OpenCl/cgbn/include/cgbn/cgbn.h:67)。主体见 [core_mont_wmad.cu:29](D:/code/MPA-OpenCl/cgbn/include/cgbn/core/core_mont_wmad.cu:29)：局部累加数组、跨 lane shuffle、多条 carry chain 交织执行乘积与 `qN` 归约。

可用于规模估计的学校式模型为

\[
\text{32×32 位 limb 乘积数 / 模乘}\approx 2W^2,
\qquad
\text{每协作线程平均}\approx\frac{2W^2}{T}.
\]

前一半来自 `ab`，后一半来自 Montgomery 归约。若分别计 lo/hi 乘加，指令计数又不同；还有加法、进位与 shuffle。因此 **这些不是 GPU 周期数**，也不能用“乘积数减半”推出墙钟减半。实际周期模型应由 SASS 指令、依赖延迟、调度器吞吐和 spill 共同校准。

旧完整 kernel 的 Nsight 记录称 Compute SOL 约 83%、DRAM 约 0.5%，见 [kernel 参数注释:289](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_kernel.h:289)。它支持当时生产形状中发射/依赖比 DRAM 更重要；不证明现行所有位宽和批量都具有相同比例。

### 4.3 CGBN 平方没有专用算法

`cgbn_mont_sqr` 直接调用 `cgbn_mont_mul(a,a)`，见 [impl_cuda.cu:1027](D:/code/MPA-OpenCl/cgbn/include/cgbn/impl_cuda.cu:1027)。因此当前 `α=cost(S)/cost(M)` 接近 1，源码并没有利用平方的对称乘积。

CPU IFMA 已有专用平方与梅森折叠，见 [simd_mont_ifma.cpp:486](D:/code/MPA-OpenCl/src/cpu/simd_mont_ifma.cpp:486)、[simd_mont_ifma.cpp:683](D:/code/MPA-OpenCl/src/cpu/simd_mont_ifma.cpp:683)。其布局与 GPU CGBN 不同，不能移用 CPU 的平方/乘法耗时比作为 GPU 预期。

### 4.4 入域不是廉价乘常数

当前 `bn2mont` 构造宽数并执行 `rem_wide`，见 [impl_cuda.cu:981](D:/code/MPA-OpenCl/cgbn/include/cgbn/impl_cuda.cu:981)。对 padding 档位，还要按 CGBN 实际 radix 解释 Montgomery 域。`mont2bn` 见 [impl_cuda.cu:1013](D:/code/MPA-OpenCl/cgbn/include/cgbn/impl_cuda.cu:1013)。

因此分片间保留 Montgomery 表示可以消除真实的多精度转换。另一个较小改动是预计算正确 radix 的 `R² mod N`，用经过验证的 Montgomery 乘法完成入域；这会增加一份常数的寄存器/载入压力，仍需比较。

### 4.5 规范化是正确性前提

CGBN WMAD 尾部根据 radix 进位修正结果，没有保证每次都比较 `r≥N` 后减 N，见 [core_mont_wmad.cu:178](D:/code/MPA-OpenCl/cgbn/include/cgbn/core/core_mont_wmad.cu:178)。现行项目封装在模乘/平方后条件减 N，见 [kernel:352](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_kernel.h:352)。

2026-10-03 的完整 Q 对照发现旧二进制 31 个 Q 中 11 个错误；问题在删除归一化后破坏后续一次加回/减去 N 的范围前提。详见 [开发日志 §45:3092](D:/code/MPA-OpenCl/docs/DEV_GPUOWL_NTT_NOTES.md:3092)。

未来若研究 lazy reduction，必须重新证明每个中间变量的范围及加减、移位、乘法接受的范围。仅增加 carry bits，或最终再做 `%N`，不能恢复已经丢失的同余信息。

## 5. 计算量、内存与传输模型

### 5.1 符号与算术量

| 符号 | 定义 |
|---|---|
| n | `bitlength(N)`；对完整梅森数 N=2^p−1 才有 n=p |
| b | `floor(log2(s))+1`，公共标量的精确位数 |
| C | 本次 GPU batch 的曲线数 |
| K、T、H | 编译容器位数、TPI、TPB |
| V | 一个存储槽位字节数 `K/8` |
| h | 每曲线槽位数；param0/2 为 7，param3 为 5 |
| J | 实际分片数量；固定每片 m 位时约 `ceil((b−1)/m)` |
| α、ε | 模平方/模乘成本比，特殊常数乘法/模乘成本比 |

有 `b≈B1/ln 2 + log2(t)` 的渐近关系；这是大 B1 近似，实际计算使用精确 b。

每条 param0 曲线的主要算子量：

\[
N_M=6(b-1),\quad N_S=4(b-1),
\quad \mathcal W_0=(b-1)(6+4\alpha)\text{ 个模乘等价单位}.
\]

当前 `α≈1`，故约 `10(b−1)` 个单位。按上节 limb 模型，全批主要乘积工作约

\[
20C(b-1)W^2
\]

个 32×32 位 limb 乘积，另加规范化、点加减、分片转换和初始化/收尾。

param2 与 param3 分别约 `(b−1)(5+4α)`、`(b−1)(4+4α+ε)`。CPU 使用专用平方时 α 不同。不能用相同位数下的 param3 `bits/s` 直接预测 param0。

param0 分片转换为 `6CJ` 次入域和 `4CJ` 次出域。完整时间应建模为

\[
T_{\rm total}=T_{\rm exponent}+T_{\rm context}+T_{\rm prepare}
+T_{\rm transfer}+T_{\rm kernel}+T_{\rm host\ gaps}
+T_{\rm finish}+T_{\rm save}.
\]

这是一种测量分解；有异步重叠后，不能简单把各 CUDA event 之和当作进程墙钟。

### 5.2 RAM 与显存有效内容容量

令 `E=4*ceil(b/32)` 为导出指数数组字节数，则当前显式 GPU 有效数据容量近似

\[
M_{\rm GPU,payload}=E+hCV+M_{\rm error\ report}.
\]

不包含 CUDA context、模块、分配粒度、managed 页、编译器 local spill backing 等实际开销。寄存器容量也不能再作为同等持久 cudaMalloc 相加。

进入 GPU 后端时，公共驱动标量、`params->batch_s`、导出数组约有三份指数有效内容；加曲线 host buffer，RAM 内容规模约 `3E+hCV`。这是已知主要副本的近似，**不是完整 RSS 峰值**；GMP 分配容量、临时大整数、缓存加载瞬态、结果数组和冷构造乘积树须另外按生命周期测量。

示例：param0，`C=1920`，完整编译默认档位。容量单位 MiB，保留三位小数。

| N 位数 | K / TPI / 每线程 L | 七槽状态 `7CV` | 每片状态读写请求 `11CV` | 仅最终 X/Z `2CV` |
|---|---|---:|---:|---:|
| 2203 | 2560 / 16 / 5 | 4.102 | 6.445 | 1.172 |
| 4423 | 4608 / 16 / 9 | 7.383 | 11.602 | 2.109 |
| 8191 | **9216 / 32 / 9** | 14.766 | 23.203 | 4.219 |

8191 位加六个 carry bits 超过 8192，因此当前进入 9216 档。不要据 `cgbn_params_8192` 的存在误报实际档位。

`B1=260e6` 时，按历史 `b≈375.1e6`，指数 E 约 **44.72 MiB**。上表三种宽度的显式指数+状态约 **48.82 / 52.10 / 59.48 MiB**。相应主要 host 内容约 **138.25 / 141.53 / 148.91 MiB**；实际峰值仍需测量。

这些量明显不同于 Stage2 的数 GiB NTT 工作区。Stage1 的显存优化更应关注溢出访问、联合运行的生命周期和吞吐，而非仅减少几 MiB 状态。

### 5.3 CPU/GPU 传输与设备内访问

正常一次完整调用、无恢复或 dump 时：

\[
\mathrm{H2D}\approx E+hCV,
\qquad \mathrm{D2H}_{\rm final}\approx hCV.
\]

另有 k 次 checkpoint 时，D2H 增加约 `k*hCV`；逐片 begin/end dump 则增加 `2J*hCV`，见 [cgbn_stage1.cu:1662](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1.cu:1662)、[cgbn_stage1.cu:1681](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1.cu:1681)。恢复时还要考虑 checkpoint 初始载入。

设备内，每片读 h 个、写 4 个状态槽位，名义请求量 `(h+4)CV`；param0/2 是 `11CV`，param3 是 `9CV`。总计约 `J(h+4)CV`，但缓存命中、内存事务合并及 spill 会改变真实 DRAM 字节数。

指数按 bit loop 读取 32 位字；所有曲线共享该数据，缓存/广播后实际 DRAM 量远小于按每线程每位简单相加。当前显式每位读取位置见 [kernel:1115](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_kernel.h:1115)。

### 5.4 吞吐和批量形状

每 block 曲线数 `q=H/T`，block 数 `G=ceil(C/q)`。设备有 S 个 SM、实际每 SM 可驻留 R 个同类 block 时，可用 `ceil(G/(SR))` 估计批次波数；它不是精确调度模拟。

至少每 SM 有一个 block 的曲线下限约 `C=S*q`。以 GPU1 的 24 SM、TPB=128 为例：TPI16 需要约 192 曲线，TPI32 需要约 96 曲线。C=1 或 12 均不能覆盖全卡。

occupancy 查询已经存在，见 [cgbn_stage1.cu:1565](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1.cu:1565)。增加 occupancy 是否提速仍取决于依赖、发射和 spill。应同时记录：

- `curve_bits/s = C*(b−1)/Σ纯 kernel 时间`；
- `curves/s = C/进程墙钟`；
- 整批延迟、最后一波填充率、实际寄存器和 spill。

对有限 save 的延迟目标与持续扫描吞吐目标应分别选择 C，不能机械地把所有任务都扩成上万条曲线。

## 6. 已有数据能支持什么

### 6.1 2026-10-05 有效 param0 小 B1 校准

既有本地 Auto B2 校准：GPU1，RTX 4060 Laptop；`B1=1000`、`t=1`、sigma 从 26 连续递增、关闭指数缓存、每形状两个非 warmup 样本。脚本在计时外逐行核对全部曲线的独立 CPU x 与 checksum，见 [measure_ecm_costs.py:189](D:/code/MPA-OpenCl/tools/bench/measure_ecm_costs.py:189)、[measure_ecm_costs.py:195](D:/code/MPA-OpenCl/tools/bench/measure_ecm_costs.py:195)。

| N bits | C | 进程时间 / 曲线（s） | 日志 `gputime` / 曲线（s） |
|---|---:|---:|---:|
| 2203 | 1 | 0.787125 | 0.085276 |
| 2203 | 12 | 0.028797 | 0.008079 |
| 4423 | 1 | 0.969824 | 0.175130 |
| 4423 | 12 | 0.060602 | 0.016189 |
| 8191 | 1 | 0.827374 | 0.305033 |
| 8191 | 12 | 0.116145 | 0.026141 |

**用途**：说明小批量无法充分摊销固定开销，增批量显著改善每曲线成本。**限制**：两个样本不是稳定生产统计；2203/C1 进程时间范围 0.464–1.110 s，4423/C12 的每曲线范围 0.0326–0.0886 s，进程启动等噪声很大。不能用这些均值外推 `B1=260e6`，也不能认定所有差值都是 PCIe 成本。

从均值看，2203/C12 事件区间/进程约 28%，4423/C12 约 27%，8191/C12 约 23%。这只是两个计时边界的比例，**不是 kernel 占比**。

数据源：[本地 profile](D:/code/MPA-OpenCl/docs/data/ecm_auto_b2_native_20261005_profile.json:1)，其上游记录是 [measurements.json](D:/code/MPA-OpenCl/build_cuda_cmake/_auto_b2_native_20261005/study/measurements.json:1)。两个路径均属本地实验产物，不随仓库提交；本文保留必要摘要及摘要值以供追溯。

### 6.2 `gputime` 的边界

`global_start` 记录在指数 cudaMalloc/H2D 和 CPU 曲线构造之前，见 [cgbn_stage1.cu:1209](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1.cu:1209)。stop 记录在每片设备同步、CPU 检查错误之后，见 [cgbn_stage1.cu:1687](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1.cu:1687)。最终 elapsed 还使用这个 stop，见 [cgbn_stage1.cu:1787](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1.cu:1787)。

因此：

- 该事件区间包含上传、准备期间设备空隙、片间主机空隙及 GPU 运算；
- 不含公共驱动先前的指数构造、后端首次初始化等时间；
- 不含最后的整批 D2H、CPU 求逆/gcd 和写 save。

后续应在 kernel launch 后立即记录该 stream 的结束 event，分别累计 kernel 区间、host wall、传输和收尾。不要把现有字段重命名后当作新测量。

### 6.3 历史优化的证据等级

历史记录主要见 [ECM_CGBN_OPTIMIZATION.md](D:/code/MPA-OpenCl/docs/ECM_CGBN_OPTIMIZATION.md:1)。

| 历史方向 | 记录结果 | 现状解释 |
|---|---|---|
| WMAD / IMAD / XMAD 探针，§1 | 某 TPI16/3072 形状约 17.56 / 25.87 / 40.59 ns/op | 支持 WMAD 优先；不是所有宽度的单曲线延迟 |
| 删除归一化，§4 | 曾报 +5–7% | 正确性已推翻，收益作废 |
| 小位宽曲线数与寄存器，§5.5 | 4096→8192 曲线约 +7.6%；56 regs 曾 +4.7% | 旧 param3 / 旧基线结果，需重测 param0 与正确基线 |
| 增加寄存器限制，§5.7 | 4608/5120 部分形状约 +1.4–3.2% | 旋钮已存在；收益依赖最后一波和 spill，不是 occupancy 越大越好 |
| SSA 改写，§5.5 | 某 param3 形状约 +0.02% | 源级临时变量减少没有明显收益；若重排 param0，须检查实际活跃区间和 SASS |
| runtime 常数差值判断，§5.6 | 寄存器增加、吞吐约 −34% | 当前已有编译期 family 特化，不能重复计算为新收益 |
| exponent 字缓存，§9.9 | 约 ±0.1% / ±0.01% | 已试验，当前无高优先级收益证据 |
| 梅森 fold+TPB256，§9.9 | 某充分填充形状 +6–8%；欠填充慢 18–25% | 旧基线 A/B，归一化修复后需复测，不能宣布普遍提速 |
| 融合 fold core，§9.10 | 正确但完整 kernel 无收益 | 相同寄存器/spill 不足以证明 SASS 完全相同；实际无收益仍成立 |
| 轮转 fold core，§9.10 | 边界样例错误 | 不能采用 |

探针本身也需校准：当前 [cgbn_op_probe.cu:41](D:/code/MPA-OpenCl/tools/bench/cgbn_op_probe.cu:41) 使用 `MAX_ROTATION=4`，生产默认为 1，见 [kernel:113](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_kernel.h:113)。探针显示 `ns/op=elapsed/(instances*iterations)`，见 [cgbn_op_probe.cu:199](D:/code/MPA-OpenCl/tools/bench/cgbn_op_probe.cu:199)，这是多实例摊销吞吐口径；需补充现行 normalized 模乘和完整曲线配置才能用于生产预测。

## 7. 候选一：批量、TPI、TPB 与寄存器配置

### 7.1 适用性及当前旋钮

适用于所有曲线族；优先 param0、2203/4423/8191 位及真实生产批量。编译参数见 [CMakeLists.txt:419](D:/code/MPA-OpenCl/CMakeLists.txt:419)、[CMakeLists.txt:503](D:/code/MPA-OpenCl/CMakeLists.txt:503)。

现行默认寄存器上限是：K≤2048 为 56；2560–5120 为 128；5632 及以上为 255，相当于 sm_89 上不强制低上限。见 [kernel:301](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_kernel.h:301)。它被多个曲线族共用，而三者的活跃大整数和步成本不同。

应构建有限组候选，例如 TPB=128/256，默认上限/几个邻近上限；大位宽先检查自然寄存器和 spill，再决定是否扫更低限制。按 `GPU UUID + arch + family + K + TPI + TPB + binary/config hash` 保存结果，不能只按输入 n 建一个全局最优值。

### 7.2 档位细化的真实收益边界

新增 2816/TPI16 虽比 3072 的存储少，但二者 `L=6,W=96` 相同；未必减少 CGBN 乘法工作。新增 8320/TPI32 仍有 `L=9,W=288`，也不能按 K 比例宣称提速。

8191 位可研究 8704/TPI16：`W=272`，相对现行 9216/TPI32 的 W=288，学校式乘积量比约 `(272/288)²=0.892`，减少约 **10.8%**。但每线程 L 从 9 增到 17，寄存器和 spill 可能抵消甚至反转收益。这是适合单档原型的假设，不是推荐立即替换默认。

预期收益应通过有效 A/B 确定；已有记录支持“批量形状有用”，不支持某一 TPB 对所有 N 一律更好。CUDA 对 occupancy 与资源限制的说明见 [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html)。

## 8. 候选二：准备、指数复用与批量求逆

### 8.1 小改动先做

- 每个 N 提前求一次 `4^-1`；param3 提前求一次 `2^-32`。
- param0 复用已算的 `u³`；结果处理避免每条曲线重新导入同一个 N。
- 也可直接按 `a24=num/(16u³v)` 构造，避免先求 A 再转 a24；需保持合数模数下的异常处理与相同曲线参数。
- 同一进程、同 `(B1,t)` 重用指数；GPU 端重用指数 allocation / upload。
- 后端资源缓存必须以设备、上下文、指数身份为键，设置容量上限并支持释放。

源码位置分别见 §3.1、§3.2、§3.5。大 B1 的冷指数构造已有磁盘缓存，继续把筛素数搬到 GPU 的优先级低于避免既有数据重复构造/加载。

### 8.2 曲线构造批量求逆

设 I 为一次求逆成本，M 为一次 CPU 模乘成本。对 C 个可逆元，标准前缀/后缀批量求逆约为

\[
I+3(C-1)M
\]

代替 `CI`。param0 的主要一般求逆是分母和 Z0；若联合处理 2C 个值，名义成本约 `I+3(2C−1)M`，另加各参数计算。可先实现两个各 C 元素的批次，以降低改动复杂度。

N 是待分解的合数：乘积不可逆时，应以 gcd 和分治隔离不单位元素，对正常曲线继续批量求逆，对异常曲线保留因子检测/回退。不能用 `x^(N−2)` 充当通用逆，也不能让单条异常曲线污染全批输出。

多线程 CPU 构造是另一候选；每 worker 必须有独立 GMP 临时量，并控制与其他工作线程的资源竞争。GPU 参数生成只有在准备已占明显比例时才值得；它还涉及批量前缀、异常元素和逆的实现，不能仅凭“GPU 快”判断。

### 8.3 最终归一化批量求逆与紧凑输出

无因子的一批最终 Z 同样可以批量求逆，之后 C 次模乘得到 x。总成本约 `I+3(C−1)M+CM`，替代 C 次独立求逆和 C 次 x 乘法。异常 Z 需保留当前 gcd 与命中状态语义。

最终收尾只需要目标点的 X/Z，可增加导出 kernel，将正常 param0 的 D2H 从 `7CV` 减到 `2CV`，减少 **71.4% 最终状态传输**；param3 为 60%。这不等于相同比例的总时间收益；C=1920 时只节省数 MiB，且不包含公共指数 H2D。

这组改动优先服务短 B1、大 C 或频繁调用的队列。大 B1 下是否值得，使用测得的准备/收尾占比决定。

## 9. 候选三：Montgomery 域驻留与同步整理

### 9.1 分片间保留域表示

初始一次把点与常数转换为 Montgomery 表示，普通分片只运行 ladder 并写回该表示。最终或 checkpoint 时通过专用导出 kernel 转换普通整数。

理想可消除 param0

\[
6C(J-1)\text{ 次 bn2mont}+4C(J-1)\text{ 次 mont2bn}.
\]

仅改变数值表示并不会自动减少七槽状态容量或 `11CV` 每片读写。若另行移除重复 N、把公共常数分离，需要独立评估 load 合并及寄存器变化。

实现要求：

- 记录真实 CGBN radix、K/TPI 及域表示，避免 padding 引起 R 不一致。
- checkpoint 使用新版本或保持明确的普通整数导出契约；完整保存两个 ladder 点和位偏移。
- 导出到 scratch，避免 checkpoint 后把热状态来回转域；恢复时按相同规则转换一次。
- 规范化封装仍维持 `[0,N)`；“Montgomery 域驻留”不意味着允许冗余范围传播。

如转换时间占总时间 f，完全消除该部分的极限提速为 `1/(1−f)`；不能在未测 f 前给出 10% 或 20% 承诺。当前约 100 ms 分片每片执行许多位，大 B1 下转换未必是大项。

### 9.2 一次等待完成 kernel 与错误检查

在 launch 后立即记录结束 event，然后等待该 event，完成后再读 managed 错误报告。这样可以把当前 device-wide synchronize 和随后 event synchronize 的组合整理为 stream 范围的一次完成等待，同时得到更准确的 kernel 时间。

仍需保留 launch 错误检测、异步执行错误传播、CGBN report 检查、停止与 checkpoint 响应。managed report 是否引起可观迁移应由 profiler 证明；源码只能确认它是 managed 分配，见 [cgbn.cu:25](D:/code/MPA-OpenCl/cgbn/include/cgbn/cgbn.cu:25)。

若约 100 ms kernel 之间只有数十微秒主机间隔，CUDA Graphs 和更大分片的上限收益很小；若间隙占比显著，再研究 50–250 ms 分片或 graph。保持 Windows watchdog 与停止响应的可控分片，不应直接改成持续数小时的单 kernel。异步执行仍受 stream 与资源条件约束，参见 [CUDA asynchronous execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html)。

## 10. 候选四：专用 Montgomery 平方

普通乘积约 W² 个 limb 乘积；理想平方的对称乘积部分约 `W(W+1)/2`，Montgomery 归约部分仍约 W²。由此得到仅按乘积量计算的比值

\[
\alpha_{\rm products}\approx
\frac{W(W+1)/2+W^2}{2W^2}
=\frac34+\frac{1}{4W}.
\]

这不是硬件耗时下界或可保证收益：对角项加倍、跨 lane 合并、进位、保存局部和及归约调度都要付出成本。

对 param0，名义每位成本由 10 降为约 9；理想减少约 **10% 算术时间**，对应约 **11.1% 算术吞吐提升**。param2 理想 9→8；param3 约 `(8+ε)→(7+ε)`。旧文档的 12–14% 表述不能不加曲线族区别地用于 param0。

建议先做一个较大档位 normalized-square 原型，比较 α、寄存器、spill、输出范围和完整 param0 时间。只利用同一 lane 内部的对称项会遗漏大量跨 lane 项，可能远达不到上述算术比例。若微观 α 无明显改善或完整 kernel 收益被寄存器抵消，及时停止。

相比 Karatsuba，平方的输入对称性更直接；但 CGBN 的交织 WMAD core 没有可直接替换的独立乘积模块，仍是中等以上实施难度。

## 11. 候选五：重新评估 param0 PRAC / 差分链

> **2026-10-06 后续**：[PRAC / Lucas 链可行性分析](D:/code/MPA-OpenCl/docs/ECM_STAGE1_PRAC_FEASIBILITY_20261006.md:1) 已实现精确素数重复计数、Prime95 十种子邻域搜索及链合法性检查。完整 B1=10³～10⁶ 名义算术节省约 11.2～13.6%；这些是离线成本比例，尚非 GPU 实测收益。以下历史密度代入由新统计补充。

> **2026-10-06 后续**：[PRAC / Lucas 链可行性分析](D:/code/MPA-OpenCl/docs/ECM_STAGE1_PRAC_FEASIBILITY_20261006.md:1) 已实现精确素数重复计数、Prime95 十种子邻域搜索及链合法性检查。完整 B1=10³～10⁶ 名义算术节省约 11.2～13.6%；这些是离线成本比例，尚非 GPU 实测收益。以下历史密度代入由新统计补充。

### 11.1 旧否定结论的问题

旧文档主要将链与约 `8+ε` 的 param3 ladder 比较，而当前 Suyama param0 是 10 个模乘等价单位/位。普通窗口法缺少所需差值点，这个问题真实存在；它不能推出所有合法差分链都输给 param0。

EFD 中一般 XZ 倍点是 `2M+2S+1*a24`，一般 projective-difference DADD 是 `4M+2S`；把完整 a24 计作一次 M，得到 DBL=`3M+2S`。当前固定仿射差值的 ladder 是 `6M+4S`。公式条件可见 [EFD Montgomery XZ](https://www.hyperelliptic.org/EFD/g1p/auto-montgom-xz.html)，当前代码算子计数见 §2.2。

另一个历史误读是把约 1.44042/bit 的一维差分链下界当作 **DADD 数量下界**。它约束链步数，包含倍点，不能直接与 DADD 门槛比较。原始讨论见 [Bernstein differential chains](https://cr.yp.to/ecdh/diffchain-20060219.pdf)；较新的 [Searching for differential addition chains](https://pure.tue.nl/ws/portalfiles/portal/361923991/s40993-024-00604-8.pdf) 同样区分链长与不同点运算的代价。

### 11.2 当前应使用的胜出条件

令 d、a 为每 `(b−1)` 个标量位平均 DBL、DADD 数，采用完整 projective 差值和完整 a24：

\[
\mathcal W_{\rm chain}/(b-1)
=d(3+2\alpha)+a(4+2\alpha).
\]

当前 `α≈1` 时，仅算术胜出要求

\[
5d+6a<10.
\]

即使满足，也要支付更多活跃点、交换、链解释和不同调度的成本，不能把名义算术收益直接当作实测提速。

d、a 必须从目标 B1 的完整计划统计，包含各素数幂乘法的初始化/收尾、额外 t 以及特殊倍点等操作。若为了使用仿射差值在每个素数幂之间求逆，也须计入该成本；不能把单个孤立标量的最优密度直接移作整个 Stage1 的密度。

例如旧仓库文档引用的 `d≈0.135,a≈1.388`，若该密度在目标 B1、实际链生成器和正确算子计数下成立，则约 `9.003` 单位/位，比 param0 10 少约 10%。**这里只是代入历史假设；本轮没有独立复现该密度。** 另一种 `d≈0.40,a≈1.31` 只有约 1.4% 名义改善，很容易被点状态开销及计数误差吞没。

因此结论是“值得重新验证条件”，不是“PRAC 已确定快 10%”。

### 11.3 本地 Prime95 的实际实现

当前参考源是文件 `.refactor/p95v3106b01.source/ecm.cpp`，不是一个缺失的 `ecm/` 目录：

- [ecm.cpp:2645](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:2645)：`lucas_cost`；
- [ecm.cpp:2711](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:2711)：`lucas_mul`；
- [ecm.cpp:2847](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:2847)：尝试邻近 d 的 `lucas_cost_several`；
- [ecm.cpp:2857](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:2857)：`ell_mul`，试十组比例种子并选择成本较低者，黄金比例初值使用 ceil；
- [ecm.cpp:6904](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:6904)：读取 `PracSearch`，范围 1–50；
- [ecm.cpp:7461](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:7461)：额外乘 12 的操作。

GMP-ECM 也有实际 PRAC，见 [gmp-ecm/ecm.c:353](D:/code/MPA-OpenCl/.refactor/gmp-ecm/ecm.c:353)。另一个本地链实现见 [ecm/ecm.c:549](D:/code/MPA-OpenCl/.refactor/ecm/ecm.c:549)、[ecm/ecm.c:1086](D:/code/MPA-OpenCl/.refactor/ecm/ecm.c:1086)。其多槽点环不宜直接移植成 GPU 动态索引寄存器数组，否则容易产生大量 local memory。

旧研究文档“本地 Prime95 的 ecm.cpp 没有 lucas_mul”的断言已修正。任何只实现一个近似比例、使用 floor 或省略邻域搜索的计数脚本，不能被称为这份 Prime95 的完整算法。

### 11.4 GPU 实现方式与内存代价

Stage1 素数幂及标量与 sigma 无关，可以让同批曲线共享一个正确链计划，减少控制流发散。计划应按 GPU 实际 DBL/DADD 成本选择，而非直接使用 Prime95 FFT 的成本权重。

优先研究少量静态命名点、紧凑规则编码或分块重放。不要先展开数亿条带寄存器编号的指令：若操作数约 `1.5b`，在 `B1=260e6` 时，一字节/操作已约 **537 MiB**，四字节/操作约 **2.1 GiB**，远大于约 45 MiB 的当前指数数组。

新 checkpoint 至少要识别素数/链位置、点状态、计划身份与版本，不再只靠 ladder bit offset。必须验证最终完整 Q、过程中差值关系、不可逆/退化情形和恢复一致性；当前不合法的 `PROBE_ADD_DENSITY` / `PROBE_CHAIN_W` 仅供计时，见 [kernel:49](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_kernel.h:49)，不能作为实现基础直接启用。

## 12. 后续候选与不建议优先投入的方向

### 12.1 梅森折叠域

当前源码有 `ECM_MERS_FOLD`，默认关闭，见 [kernel:172](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_kernel.h:172)。折叠模乘见 [kernel:606](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_kernel.h:606)。

它消除一般 Montgomery `qN` 链，但也减少可交织的独立工作，所以更依赖延迟隐藏；历史上既出现整 kernel 变慢，也出现 TPB256 充分填充时变快。当前“至少两个 block/SM”的主机代码只是警告，不是自动切回 basic，见 [cgbn_stage1.cu:1580](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1.cu:1580)。该分支只比较 grid block 数与 `2*SM数`，并不保证实际可驻留 R≥2；高寄存器档位应结合 occupancy 查询和 profiler 验证。

现行开关是 whole-build 选择，见 [CMakeLists.txt:534](D:/code/MPA-OpenCl/CMakeLists.txt:534)。应研究同时保留 fold/basic、按纯梅森模数与有效批量调度；梅森余因子通常不再等于 `2^p−1`，不能套用相同折叠。

### 12.2 Karatsuba / 更大位宽算法

对一般 Montgomery 模乘，只优化 `ab` 乘积为单层 Karatsuba、把该半工作从 1 降到 3/4，而 `qN` 不变，理想总乘积量只从 2 降到 1.75，即减少约 12.5%。这不是一般模乘直接减少 25%。折叠域的模型不同，但额外加减、进位、shuffle 与寄存器也必须计入。

NTT/FFT 不宜直接替换当前 2K–16K 位、每条曲线逐位串行的小粒度模乘：需要转换、padding、归约及设备缓冲，Stage2 多项式批量卷积的优势不能直接移用。更大 n 或全新 Stage1 算法可以另立专题，当前没有该替换的生产性能证据。

### 12.3 同 GPU 并发 Stage1 / Stage2

可以重叠 CPU 的下一批准备与当前 GPU 执行，但同 GPU 的两个繁重 kernel 是否能受益取决于 SM 资源剩余和调度；已有 issue-bound 历史数据不支持预设二者会无代价重叠。需分别测总流程 curves/s、每阶段延迟和实际同时存活的显存峰值。

### 12.4 低收益或当前不合法的方向

- 删除 normalized wrapper：已有明确错误证据。
- 普通 w-NAF 直接套入 XZ：缺少任意差值点。
- 指数位寄存器缓存：已有接近零收益的试验；优先级低于大整数算术。
- 强制 XMAD/IMAD：旧探针明显更慢；没有当前收益依据。
- 再加入 `xdiff==2` runtime 大分支：已经因寄存器成本用编译期特化解决。
- 仅去掉每曲线的 N 副本或调整 state SoA：可减常数级内存，但不能减少逐位模乘，通常不是大 B1 主项。
- GPU GCD/求逆：难度与异常处理成本较高，先做 CPU 批量求逆及阶段占比测量。

## 13. 建议的后续测量和验收方案

本节未在本轮执行。实施时先固定现行正确二进制的 SHA、源码摘要、编译选项和所有输入，记录 GPU UUID、功耗/频率状态；相同 N、sigma、B1、t、曲线族才是算术优化 A/B。

### 13.1 测量分层

1. 进程启动至有效 save 的墙钟，包括冷/热指数缓存两种情况。
2. CPU 曲线准备、指数导出/上传、状态上传、纯 kernel 合计、片间 host 空隙、checkpoint、最终 D2H、求逆/gcd、写文件分别计时。
3. 记录每档 `K/TPI/TPB/C/J`、寄存器、spill、设备读写量；Nsight 工具开销单列，不能用 profiler 墙钟代替正常运行时间。
4. 以正常完整曲线的 `curve_bits/s` 和总流程 `curves/s` 作决策，探针只解释变化原因。

首轮覆盖 2203/4423/8191 bits、小 B1 短任务和至少一个较大 B1；C 包含 1/12、每 SM 一块附近、充分填充和真实队列批量。生产大 B1 A/B 再确认长期收益。每组有 warmup、多个重复和配对顺序，保留范围或置信区间。

### 13.2 正确性与兼容性

现有工具可以复用，实施优化时再按用户授权执行：

- [test_gpu_exponent.py](D:/code/MPA-OpenCl/tools/test/test_gpu_exponent.py:1)：归一化、不同指数约定及完整 Q 对照。
- [test_cuda_param0.ps1](D:/code/MPA-OpenCl/tools/test/test_cuda_param0.ps1:1)：CPU/GPU 同曲线、64 位 sigma、save 互操作及恢复。
- [test_cuda_param2.ps1](D:/code/MPA-OpenCl/tools/test/test_cuda_param2.ps1:1)：family 回归。
- [test_cuda_mers_fold.ps1](D:/code/MPA-OpenCl/tools/test/test_cuda_mers_fold.ps1:1)：fold 边界与通用模数限制。

覆盖 `K−6` 附近的选档边界、非梅森模数/余因子、极短指数、64 位 sigma、因子命中、不可逆 Z、checkpoint 导出/恢复。checksum、进程返回 0 或“找到了某个因子”均不能替代独立完整 Q 检查。

### 13.3 按测得占比选择下一步

| 测量结果 | 优先动作 | 可以报告的收益边界 |
|---|---|---|
| 小 C 欠填充 | 批量与波形调优 | 以曲线吞吐与总延迟分别验收 |
| prepare / finish 明显 | 循环不变量、批量求逆、紧凑导出 | 消除该阶段的理想上限由其占比决定 |
| 入/出域显著 | Montgomery 状态驻留或预计算 R² | 明确减少转换次数；总收益由转换占比决定 |
| 片间空隙显著 | 一次 stream/event 等待、分片参数 | 可用空隙占比给 Amdahl 上限 |
| 模乘/平方占主项 | 专用平方，随后合法链 | 按 α、实际链密度及完整 kernel 验收 |
| 大位宽 spill 高 | TPI、寄存器限制与档位 | 先减少实际 local traffic，再看总时间 |

性能优化在正确性通过后，完整路径收益应超过测量噪声，并满足实际队列的总流程吞吐目标。记录负结果也有价值，避免把已否定的探针方向反复列为可用收益。

## 14. 源码与证据索引

审查文件的 SHA256（大小写不影响比较）：

| 文件 | SHA256 |
|---|---|
| `src/core/ecm_driver.cpp` | `7bf7fa4c3cc4dfcfa3febce31fcce5a8fa276cf1e2acb52e774480f5bd442e61` |
| `kernels/cuda/cgbn_stage1.cu` | `0fe51802f1b4901957a4f69461879de93e45905850f4e1d0e2bc78302d89777d` |
| `kernels/cuda/cgbn_stage1_kernel.h` | `4d21b0d9460ec481c32d76577087cffa7b277637889bea0492633a429804ace0` |
| `cgbn/include/cgbn/impl_cuda.cu` | `21094ef6170489ec25c7f43fe7ac3376994c8a5ec5c49ef831a803cbe6d2ed0a` |
| `cgbn/include/cgbn/core/core_mont_wmad.cu` | `f11bd15a1d4d43fbb0bdcde30249cb2acd3cbf5ec409f6c496bc87827627e38a` |
| `src/cpu/simd_mont_curve.cpp` | `f28197a3520885e9992670307fe8ee2b29c2af91c1b7f44bc27186439e2774bb` |
| `.refactor/p95v3106b01.source/ecm.cpp` | `7e6be864ff0c76420150d36f5a72f182cdfaa21ad07f93f98132c3ae0f133964` |

本地 `cgbn/` 不是独立 Git checkout；不能将父仓库 HEAD 当作 NVIDIA upstream 版本。这里以实际文件摘要标识所分析实现。

2026-10-05 profile SHA256：`91cd7dbf3f455de27e93456aa177f2c6628772d547f6cab2e51f54af60d91dfc`。其 Stage1 二进制 SHA256：`5ff1f58a3a072fb37b7ef6e35d3ac2de5488304d6acc5d4f32b6555cb2c96e6e`。这些数据仅证明对应旧实验的来源，不说明审查源码已在本轮运行。

关联开发文档：[ECM_CGBN_OPTIMIZATION.md](D:/code/MPA-OpenCl/docs/ECM_CGBN_OPTIMIZATION.md:1)、[ECM_Montgomery_STAGE1.md](D:/code/MPA-OpenCl/docs/ECM_Montgomery_STAGE1.md:1)、[ECM_XONLY_SCALARMUL_RESEARCH.md](D:/code/MPA-OpenCl/docs/ECM_XONLY_SCALARMUL_RESEARCH.md:1)、[DEV_GPUOWL_NTT_NOTES.md §45](D:/code/MPA-OpenCl/docs/DEV_GPUOWL_NTT_NOTES.md:3092)。

外部资料用于公式条件及硬件约束；优化优先级、容量计算和对本仓库的判断来自上述本地源码与已注明边界的实验记录。
