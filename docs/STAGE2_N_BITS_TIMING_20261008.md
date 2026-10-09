# Stage2 耗时与 N 位宽：RTX 4060 Laptop，2026-10-08

## 实验口径

本报告区分三个量：原梅森指数 `p`、实际输入整数 `N`、实际位宽 `S=bit_length(N)`。图中横轴 **N bits** 指 S，每1000 bits一格；它不是剥离因子前的 p。公式中的位宽增长也使用 S，而不是把数值 N 本身当作线性变量。

输入从 `tools/ecm_dataset/ecm_stage2_dataset.sqlite` 只读提取。主实验对 `2^p−1` 剥离数据库中全部已知因子及其重数，测余因子；完整梅森数另作单次对照。保留数据库提取快照、实际 `N_hex`、已剥离因子、命令、保存点、程序/DLL/源码哈希与逐条原始结果。

- 原指数：503、1009、2003、3001、4001、5003、6011、7001、8011。
- B2：`260e8=26,000,000,000`、`260e9=260,000,000,000`、`260e10=2,600,000,000,000`。
- Stage1：统一 **B1=20 / PARAM0 / lcm**；M2003两种输入均用sigma27，其余均用sigma26。较大的B1会提前检出完整梅森数中的小因子，无法保留完整输入对照，因此使用小B1。
- 保存点在计时前由Python参考与独立GMP-ECM分别计算，要求N及归一化X完全一致；随后写入本程序的checksum。GMP-ECM明确指定B2=0，不执行附带Stage2。准备阶段发现B2=B1仍可能经区间取整检出80111并把M8011存档改为余因子；这次失败准备保留，但没有进入GPU计时数据。
- GPU1：NVIDIA GeForce RTX 4060 Laptop GPU / sm89 / 8188 MiB；GPU0原有任务未停止。本轮没有把其他系统负载完全隔离。
- 当前生产Stage2：CUDA13.3，固定Goldilocks PTX后端、canonical减法、驻留scaled根/下降frontier；显式arena6300、fold640、batch256 MiB。D=0采用现有自动选择，逐条记录实际D/P。
- `--factor-only`：保留Stage2计算、叶乘积/GCD及必要算术检查，不做命中素数命名和GP因子拆解。完整梅森对照可能检出已有小因子；它不是无因子输入。
- 主实验每个 `(N,B2)` 三遍，共81条；完整梅森每格一次，共27条。另有每个位宽一次最小B2预热，共9条，不进入均值。正式轮次正序/逆序/正序；对照插入第二轮。

### 时间边界

主指标为 **`stage2_full_wall.total=init+main`**，包含baby/F树、必需自检、inverse、giant/G树、fold、下降、叶乘积/GCD及异步算术检查收尾，**不含Stage1和自动D扫描**。计时边界见 [ecm_cuda_stage2.cu:8915](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:8915) 与 [完整计时输出:8938](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:8938)。另记shape秒数、curve result秒数及整个外层进程墙钟；这些数不互相替代。

每遍使用新的进程/上下文，包含该曲线的初始化和必需检查，属于独立单曲线口径。它不是同一进程连续多曲线、复用部分缓存/检查后的稳态吞吐率；评估后者应分别使用记录中的 init/main，并另做同进程测量。

日志 `shape` 字段是 [baby索引构造:8530](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:8530)，已经通过 [初始化计时:8897](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:8897) 加入init，不能再加到total上。真正的自动D扫描另读 `d_scan_wall`，明确排除在full_wall之外，见 [扫描计时:8508](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:8508)。

各遍均启用相同必要检查。`s4_multiply_stats.t_reduce` 为设备归约事件时间，嵌套在G树/fold/下降等阶段中；不能再与这些阶段相加。各模块的容量峰也不是同一时刻分配清单，不能相加为进程峰。NVML每约250ms采样整张GPU，利用率表示采样窗口内忙碌比例，不是SM occupancy；显存观测包含设备背景占用，也可能漏掉短暂峰值。

`batched_progress.t` 是三个模块计时的合计，不是完整循环墙钟；未覆盖的准备/回退开销仍计入 `stage2_full_wall.total`。

### 功耗条件

本轮驱动实际执行 **55 W** 限制，不能作为 90 W/100 W 满功率数据。G-Helper 在实验期间有参数写入；已复现其 0.283 版对双 NVIDIA 显卡功耗上限解析失败，产生 150 W 的错误滑块范围。完整证据、时间线及待执行对照见 [功耗诊断](D:/code/MPA-OpenCl/docs/GPU_POWER_LIMIT_4060LP_20261008.md)。本轮记录保留，不与后续不同功耗条件混合。

117次全部完成后，按用户授权退出G-Helper，以管理员权限仅对GPU1尝试`-pl 90`。驱动返回`not supported`，实际enforced limit仍为55W；没有产生90W数据。SMI把该warning处理成退出码0，判定必须同时查看原始输出与实际回读。

## 完整测量结果

**117/117次完成：81条余因子正式测量、27条完整梅森单次对照、9条预热；失败0。** 所有接受记录均通过保存点/设备身份、必要GMP算术门禁及因子整除验证；三遍之间D、工作量、叶指纹和因子结果一致，分析警告0。各格样本CV为0.06%–5.61%。测量起止UTC：`2026-10-08T21:49:14.123209+00:00` → `2026-10-08T22:54:13.952908+00:00`。

以下为三遍**均值 ± 样本标准差**，不是误差上限。单位秒，S/W使用剥离后的实际输入。剥离因子的值与重数见分析JSON的inputs。

| 原指数 p | 实际 S bits | W | 剥离因子条数 | B2=2.6×10¹⁰ / s | B2=2.6×10¹¹ / s | B2=2.6×10¹² / s |
| --- | --- | --- | --- | --- | --- | --- |
| 503 | 318 | 5 | 3 | 0.361 ± 0.017 | 0.895 ± 0.050 | 3.277 ± 0.065 |
| 1009 | 383 | 6 | 7 | 0.413 ± 0.003 | 1.088 ± 0.017 | 3.902 ± 0.036 |
| 2003 | 1939 | 31 | 2 | 1.785 ± 0.008 | 5.634 ± 0.034 | 18.532 ± 0.383 |
| 3001 | 2726 | 43 | 5 | 2.592 ± 0.024 | 7.703 ± 0.065 | 28.406 ± 0.472 |
| 4001 | 3600 | 57 | 5 | 4.029 ± 0.062 | 12.803 ± 0.173 | 42.274 ± 0.582 |
| 5003 | 4667 | 73 | 5 | 5.108 ± 0.019 | 16.148 ± 0.358 | 54.826 ± 0.456 |
| 6011 | 5872 | 92 | 3 | 7.992 ± 0.014 | 23.758 ± 0.287 | 90.436 ± 0.506 |
| 7001 | 6797 | 107 | 3 | 10.416 ± 0.245 | 32.501 ± 0.318 | 121.515 ± 0.656 |
| 8011 | 7995 | 125 | 1 | 14.553 ± 0.107 | 38.895 ± 0.273 | 182.948 ± 0.106 |

![余因子 Stage2 时间曲线](D:/code/MPA-OpenCl/docs/figures/stage2_n_scaling_20261008_time.png)

### 不同B2、位宽的阶段占比

![余因子均值：三档B2的100%阶段堆积图](D:/code/MPA-OpenCl/docs/figures/stage2_n_scaling_20261008_phases_percent.png)

以7995 bits余因子为例，图中精确口径对应以下占比；各项显示到0.1%，取整后可能不恰好100%。

| 7995 bits 余因子阶段 | B2=2.6×10¹⁰ | B2=2.6×10¹¹ | B2=2.6×10¹² |
| --- | --- | --- | --- |
| Baby / 归一化 | 19.3% | 20.8% | 6.6% |
| F树 / 初始化 / 检查 | 6.4% | 6.1% | 2.1% |
| Inverse | 3.8% | 3.6% | 1.3% |
| Giant点 | 11.3% | 12.6% | 19.0% |
| G树 | 24.6% | 30.4% | 49.3% |
| Fold | 8.7% | 12.8% | 15.9% |
| G叶准备 / 回退 | 15.2% | 1.2% | 1.7% |
| F树下降 | 9.2% | 11.4% | 3.7% |
| 叶乘积 / GCD | 0.9% | 0.5% | 0.1% |
| 其余 | 0.5% | 0.7% | 0.3% |

最大B2、7995 bits时，G树 **49.3%**、fold **15.9%**、giant点 **19.0%**，三者合计 **84.2%**。随着B2增大、循环批数增加，一次性的初始化/下降占比减少。

同一输入的S4设备归约由最小B2的2.242s / 15.4%增至最大B2的 **46.295s / 25.3%**。这是上述阶段中的嵌套工作，不能再加入堆积图。最小B2下G叶准备占15.2%，说明准备和CPU路径仍应按具体输入检查，不能用大B2图中的比例外推所有任务。

![完整梅森单次对照：三档B2的100%阶段堆积图](D:/code/MPA-OpenCl/docs/figures/stage2_n_scaling_20261008_phases_percent_controls.png)

### 自动选D与显存台阶

最大B2的实际选形及模块容量如下。NTT列为三遍`full_peak_bytes`的最大值，owner列为模块实际启用峰值。**这些不是同时分配清单，不能相加为进程显存峰。**

| S bits | D | P | G | fold NTT L | NTT full峰 / MiB | fold owner / MiB |
| --- | --- | --- | --- | --- | --- | --- |
| 318 | 1711710 | 155520 | 10 | 2^24 | 488.95 | 41.53 |
| 383 | 1711710 | 155520 | 10 | 2^24 | 480.98 | 49.83 |
| 1939 | 1711710 | 155520 | 10 | 2^26 | 1634.20 | 257.48 |
| 2726 | 1711710 | 155520 | 10 | 2^27 | 3186.81 | 357.15 |
| 3600 | 1711710 | 155520 | 10 | 2^27 | 3190.50 | 473.43 |
| 4667 | 1381380 | 126720 | 15 | 2^27 | 3189.08 | 494.04 |
| 5872 | 1141140 | 103680 | 22 | 2^27 | 3186.92 | 509.42 |
| 6797 | 1021020 | 92160 | 28 | 2^27 | 3186.11 | 526.65 |
| 7995 | 810810 | 77760 | 42 | 2^27 | 3186.11 | 519.11 |

本轮全部正式记录均启用驻留fold owner，`fallback=none`，arena溢出0。最大B2从3600→4667 bits开始，D/P下降，G从10逐渐增加至42；fold NTT长度停留在2²⁷。因此时间加速增长不能解释成“NTT长度继续增长”，还包含更多批次以及更宽的点算术/长除法。当前自动D使用未覆盖本轮作用域的legacy排序，表中D也不构成最优D证明。

### 完整梅森数单次对照

| p = S bits | B2=2.6×10¹⁰ / s | B2=2.6×10¹¹ / s | B2=2.6×10¹² / s |
| --- | --- | --- | --- |
| 503 | 0.512 | 1.356 | 4.413 |
| 1009 | 1.546 | 4.797 | 15.131 |
| 2003 | 6.563 | 20.539 | 71.799 |
| 3001 | 3.456 | 10.352 | 37.272 |
| 4001 | 12.350 | 41.060 | 130.955 |
| 5003 | 7.443 | 21.740 | 75.033 |
| 6011 | 7.804 | 22.944 | 85.462 |
| 7001 | 7.943 | 26.538 | 109.272 |
| 8011 | 34.648 | 117.400 | 692.507 |

![完整梅森与余因子对照](D:/code/MPA-OpenCl/docs/figures/stage2_n_scaling_20261008_controls.png)

完整梅森数的S4折叠通常较快，但**保留因子引发的非单位点可以使整体更慢**。M8011、最大B2实测 **692.507s**，其中G叶准备/回退 **544.441s（78.62%）**；全部3135组为bad，good segments为0，非单位segment逆元计数200417。回退bad-point D2H为 **6,464,648,736 bytes（6.02 GiB）**，patch上传为6,464,648,736 bytes。其S4仅8.578s；整张GPU平均利用率约20.5%。这与CPU逐点归一化/求逆及传输回退相符，见 [非单位回退:7527](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:7527) 和 [patch上传:7586](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:7586)。

该对照确实检出80111；`--factor-only`下`hits=0`不能解释为没有因子，应读取结果factors。M1009的叶乘积/GCD在三档B2分别占41.1%、46.5%、44.7%，也体现了命中输入与无命中余因子的不同。每格仅一次，不能据此比较微小性能差异。

### 耗时经验公式

对 **1939–7995 bits余因子**、三档B2共21个三遍均值，在log空间拟合：

```text
T_seconds ≈ 0.5852 * (S/1000)^1.4806 * (B2/2.6e10)^0.5234
```

拟合内log R²=0.99272，MAPE=8.02%，最大绝对相对误差=22.63%；按原exponent整组留出时MAPE=10.38%，最大误差=28.10%。后者检验在当前区间内预测一个未参与拟合的位宽，不是外推验证。

按B2分别拟合`T=C*(S/1000)^α`：

| B2 | C / s | 位宽指数 α | 拟合内 MAPE |
| --- | --- | --- | --- |
| 26000000000 | 0.6184 | 1.4651 | 6.28% |
| 260000000000 | 2.0535 | 1.4022 | 6.20% |
| 2600000000000 | 5.8666 | 1.5747 | 8.24% |

这一范围平均更接近 **S¹·⁵**，不是统一的线性或平方关系。全位宽含318/383 bits的拟合α=1.1403，MAPE=15.62%，固定成本使该指数偏低，不建议用于生产位宽预测。

联合模型的β约0.52隐藏了局部变化：7995 bits的相邻两档B2指数分别为 **0.4269、0.6724**。最大档NTT长度/预算、D/P和G取整变化后，不能保证继续按sqrt(B2)增长。公式适用于本轮55W观测条件、版本、预算、小B1和模数类型，不能外推为90W性能或完整梅森含因子性能。

![嵌套S4归约设备事件时间](D:/code/MPA-OpenCl/docs/figures/stage2_n_scaling_20261008_reduction.png)

### 对下一轮优化的含义

1. **大余因子、大B2：优先联看G树NTT与通用S4归约。** G树约占半数墙钟，而S4在多个模块中累计约四分之一。需要以NTT/S4内部计时或profiler区分二者，不应把G树50%等同于NTT50%。giant点在最高位宽也已接近19%，应纳入收益估算。
2. **完整梅森含小因子：先优化非单位回退或采用因子剥离后的输入。** 本例CPU/G叶路径占79%，单独加速NTT难以改变总时间。是否检出因子后提前停止需保留现有“收集全部命中”语义，不能直接删掉后续工作。
3. **小B2/小位宽：减少每进程初始化和检查重复、检查G叶准备路径。** 当前数据是每遍新进程，评估同进程缓存复用要单独测量；不能从full-wall均值直接宣布稳态吞吐率。
4. **校准Auto D/Auto B2时带入W、NTT长度、D/P/G和回退类别。** 单一S/B2幂函数适合估量级，不能替代预算台阶和模数路径特征。

### 数据与程序身份

- [逐遍CSV（含9条预热）](D:/code/MPA-OpenCl/docs/benchmarks/stage2_n_scaling_20261008_runs.csv)
- [54格分组CSV](D:/code/MPA-OpenCl/docs/benchmarks/stage2_n_scaling_20261008_summary.csv)
- [输入、拟合、审计哈希JSON](D:/code/MPA-OpenCl/docs/benchmarks/stage2_n_scaling_20261008_analysis.json)
- 图片在`docs/figures/`，每张保留PNG与SVG；原始命令、save、传感器和完整日志在排除提交的`data/stage2_n_scaling_20261008/study_v2/`。

冻结exe SHA256：`29a69175bb2ebf234c2f9a4346dbbb9ecde4a9a0e82854bbca0108a4bb04f486`。数据库没有写入；本轮没有修改CUDA生产代码。

## 图表与阶段百分比口径

时间曲线使用实际N bits作横轴，每1000 bits一格；阴影为三遍最小/最大观测范围，不是置信区间。完整梅森对照每格只有一次，不计算样本标准差。

100%堆积柱图分别展示余因子均值与完整梅森单次对照，每图包含三个B2面板。柱间距是类别间距，标签为实际N位宽，便于同时看清318/383 bits；不能据此把两个相邻柱的距离解释为位宽差。

百分比定义为 `100 × mean(阶段秒数) / mean(stage2_full_wall.total)`。分区如下：

| 图例 | 记录/计算 | 口径 |
| --- | --- | --- |
| Baby points + normalization | `real_baby.ladder+affine` | baby生成与归一化 |
| F tree + init setup/checks | `init−baby` | F树及其余初始化、索引构造、必需检查；不是纯F树kernel |
| Polynomial inverse | `real_batched_split.inv` | 多项式inverse准备 |
| Giant points | `.giant` | giant点生成 |
| G trees | `.gtrees` | G树构建，含其NTT/归约 |
| Fold | `.fold` | G多项式累积/模F归约 |
| G-leaf preparation / fallback | `real_batched_wall.gleaves` | G叶准备、设备处理、CPU归一化/非单位回退与相关传输 |
| F-tree descent | `.descent` | 根准备后的scaled下降 |
| Leaf product + GCD | `.accum+.name` | 叶乘积/GCD；本轮name为0 |
| Other / accounting remainder | `total−以上各项` | 未归入模块计时的准备、收尾、检查等残差 |

这些是程序模块的墙钟记账分区，**不是纯GPU kernel工作量或CPU/GPU忙碌比例**。`ntt_seconds`和`s4.t_reduce`嵌套在多个阶段内，不能另加到100%上；S4仍保留独立时间曲线。模块日志按毫秒取整，极短阶段的百分比精度有限。

计时分区由 [real_batched_wall:8995](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:8995) 的闭合记账支持。G叶驻留准备在 [7388](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:7388) 计入`gleaves`，CPU回退在 [7566](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:7566) 结束计时；而设备叶填充及patch上传发生在`build_groot_device`回调内，属于G树时间，见 [7586](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:7586) 和 [7616](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:7616)。因此图例表示日志的模块边界，不能把所有语义上的“叶准备”都理解为同一个独立计时器。

## 算法给出的位宽关系

令：

```text
S = bit_length(N)
W = ceil(S/64)
P = phi(D)/2
I = floor(B2/D) + 2
G = ceil(I/P)
fold_count = G - 1
```

I、G和fold次数来自当前 [批处理引擎:7129](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:7129)、[G批次数:7228](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:7228)。B1远小于D时还有低素数桥接/命中处理成本；本轮计入完整时间，没有人为扣除。

### 1. NTT 随位宽近似线性增长，但长度取2的幂

对一次最大操作数含m个系数的乘法，当前packing满足：

```text
slot_bits = 2S + max(1, ceil(log2(m)))
q = 2^64 - 2^32 + 1
b = 最大可用整数，使 m * ceil(slot_bits/b) * (2^b-1)^2 < q
d = ceil(slot_bits/b)
L = 最小的2的幂，满足 L >= 2*m*d + 1
```

见 [choose_cfg:2690](D:/code/MPA-OpenCl/src/cuda/stage2/ntt_runtime.cuh:2690)、[ntt_shape_query:3068](D:/code/MPA-OpenCl/src/cuda/stage2/ntt_runtime.cuh:3068)。一次长度L的radix2变换约有 `(L/2)log2(L)` 个蝶形。设实际变换批次为j，可用 `U_NTT=Σ batch_j·L_j·log2(L_j)` 表示变换工作量级。

固定D/P时，增大S会增加digits和L；总体近似 `S log S`，但L跃迁会形成台阶，b也不是常数。最大的fold规划使用m=P+1；实际F树使用补齐到2幂的堆布局，树顶最大子操作数可能是 `2^floor(log2(P−1))+1`，不能把规划中的P/2+1估算当作所有真实树顶操作数。

### 2. 余因子的通用归约包含二次工作量

余因子通常不再是精确 `2^S−1`。S4执行长除法：约O(W)个消去位置，每个位置对W个limb执行乘减，单个系数主导约 **O(W²)**。见 [s4_div_rem:1985](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:1985)、[s4_reduce_kernel:2145](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:2145)。点乘法的学校乘法部分也含W²个limb MAC。

因此一个解释成本来源的模型为：

```text
T ≈ A
  + a * U_NTT(S,D,P,G)
  + b * C_reduced(D,P,G) * W^2
  + c * [P*log2(D) + I] * W^2
  + CPU准备、传输、GCD和检查成本
```

`C_reduced`可直接读取本轮 `s4_multiply_stats.coeffs_reduced`，不是把逻辑多项式乘法数当作设备launch数。系数总数在树层中还含logP因素。这里给出操作数量级，a/b/c由硬件、并行度、缓存和具体路径决定；本轮没有采集周期或stall，不能转换成固定周期数。

### 3. 完整梅森数有不同的归约路径

精确梅森模数由 `popcount(N+1)==1` 识别，见 [模数判定:2389](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:2389)。其S4通过反复 `low_S(x)+high_S(x)` 折叠余数，不执行通用长除法，见 [s4_mersenne_rem:2045](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:2045)。在本轮双宽乘积范围内，折叠次数有界，归约主体接近O(W)；点算术也使用 [折叠/旋转Montgomery归约](D:/code/MPA-OpenCl/src/cuda/stage2/stage2_point_mersenne.cuh:6)。

**完整梅森数不等于所有计算都变为O(W)**：点乘积仍含学校乘法的W²工作，NTT仍需要宽系数packing。因而完整梅森单次对照与余因子主曲线必须分别解释。对同一个p，两者的实际位宽、因子命中、非单位点及可能的D也不同，不能将时间比完全归因于某一个归约内核。

### 4. 显存预算会改变位宽增长规律

本轮owner复用策略为3，当前布局精确容量为：

```text
M_owner = 8*W*(7*P+7) + 48 bytes
```

见 [fold_owner_layout:31](D:/code/MPA-OpenCl/src/core/ecm_stage2_geometry.h:31)。旧文档中的 `8W(9P+8)+48` 对应未复用布局，不能直接套用本轮。

NTT三大数组的payload为 `24*L*batch bytes`，见 [ntt_arena_bufs](D:/code/MPA-OpenCl/src/cuda/stage2/ntt_runtime.cuh:2129)，其他表、fuse base和小缓冲另计。预算与实际剩余显存会限制P；owner还有独立budget/headroom检查，见 [FoldDeviceState:5246](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:5246)。

当NTT容量限制近似固定 `P*S/b` 时，P可随S增大而缩小，D也可能下降，于是I和G增大。此时除了每次宽算术更慢，批次数也增加，不能继续按固定D的位宽指数预测。忽略b、phi(D)/D和取整变化的局部近似为：

```text
P ~ 1/S, D ~ 1/S, I ~ B2*S, G ~ B2*S^2
NTT每批规模接近预算上限，循环变换成本可接近 B2*S^2
通用归约/点算术可能另含更高的S幂次；实际是混合成本
```

这是解释预算区间的近似，不是全范围渐近定律。是否owner回退、是否arena溢出以及实际D/P，都以每条日志为准。

## 与既有 B2 报告的关系

[B2/显存预算实验](D:/code/MPA-OpenCl/docs/STAGE2_B2_MEMORY_BUDGET_SCALING.md) 已分析：固定D时B2增加主要增加I/G，趋向线性；允许D随B2重选且未受预算限制时，平衡 `aD+bB2/D` 会得到近似平方根关系。本轮横向改变S，且使用不同余因子、较新的生产版本及B1=20，不能把旧M2203/M4423/M8191的秒数直接接到新曲线上，也不能用新旧时间差宣称版本加速。

自动D仍有校准作用域：本轮需读取 `d_model_scope`。如果因当前缓存payload/作用域尚未校准而使用legacy排序，其选D是当前生产行为，不表示全局最优D；也不能把排序器内部的estimated seconds当作本轮实测时间。见 [校准条件:8302](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:8302) 与 [legacy成本说明:8275](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:8275)。

## 复现工具

使用 [bench_stage2_n_scaling.py](D:/code/MPA-OpenCl/tools/bench/bench_stage2_n_scaling.py)、[analyze_stage2_n_scaling.py](D:/code/MPA-OpenCl/tools/bench/analyze_stage2_n_scaling.py)、[plot_stage2_n_scaling.py](D:/code/MPA-OpenCl/tools/bench/plot_stage2_n_scaling.py)。完整参数与执行顺序见 [工具说明](D:/code/MPA-OpenCl/tools/bench/README_STAGE2_N_SCALING.md)。原始文件位于 `data/stage2_n_scaling_20261008/study_v2/`，根data目录沿用仓库排除规则。
