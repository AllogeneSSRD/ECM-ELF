# ECM Stage2：Prime95 CPU 与 CUDA GPU，2026-10-09

## 1. 结论

本次 Prime95 队列已结束：27 个 `(原梅森指数,B2档位)` 任务区段，134 次曲线尝试，
**100 次完整 Stage2、34 次命中因子提前结束**。队列中没有剩余活动任务。
100 次完整 Stage2 的打印计时累计 **3254.786 s**；首条曲线至末次 GCD 的日志时间
为 `2026-10-09 01:26:47–02:22:40`，约55分53秒，时区 Europe/London。
前者是各曲线阶段计时之和，后者还包含 Stage1、任务切换等，不能互相替代。

主要发现：

- **1939 bits 同一余因子**：GPU 三档 B2 分别快 **1.38、2.07、2.30 倍**。
- **7995 bits 同一余因子**：GPU 速度比仅 **0.42、0.76、0.60**，即 GPU
  分别比 CPU 耗时长 **2.38、1.31、1.66 倍**。
- GPU 使用此前完整测量的 **55 W 数据**。功耗修复后仅有7995 bits、中档 B2 的
  三遍补测：GPU **34.920 s**，CPU **29.620 s**，仍比 CPU 耗时长约17.9%。
  不能把这一个点的改善比例应用于整张表。
- Prime95 参考源码把“待分解余因子 N”和“快速算术使用的原梅森形式”分开；
  当前 GPU 余因子路径使用通用模归约。这是比单独调整 NTT 内核更值得优先
  研究的算法差异。**本轮没有独立测量该差异贡献，不能给出其确定加速比。**

可切换 B2 的[交互看板](C:/Users/Elysia/.cursor/projects/d-code-MPA-OpenCl/canvases/ECM-stage2-CPU-GPU-20261009.canvas.tsx)
包含精确配对、选形、样本数和所有 CPU 位宽分组。下面四类图同时提供 PNG 和 SVG。

## 2. 数据与配对规则

### 2.1 来源

- CPU：`D:/code/GIMPS/p95v3104b05.win64/screen.log`、同目录 `results.json.txt`、
  结束后的 `worktodo.txt`，只读提取。程序版本31.4b05。
- CPU：本机 AMD Ryzen AI 9 HX 370，12核24线程。日志主线程绑定逻辑CPU1，
  polymult helper为3、5、7；这不是使用全部12核的CPU极限吞吐率。
- GPU：RTX4060 Laptop / 8188 MiB；2026-10-08完整117次记录中的81条余因子
  正式测量，另有27条完整梅森对照与9条预热。主表不计预热。
  使用[完整 GPU 分析](D:/code/MPA-OpenCl/docs/benchmarks/stage2_n_scaling_20261008_analysis.json)，
  **没有使用 `final_preview` 的115条中间快照**。
- GPU预算：arena6300、fold640、batch256 MiB；CUDA13.3 / sm89；独立单曲线进程，
  必需算术检查开启，factor-only，不做可选因子拆解。

### 2.2 为什么不能直接按 p 配对

CPU各任务起始时可含已知小因子，命中后会缩小N并重新从curve #1计数。
GPU主实验则预先剥离数据库全部已知因子。相同 `M<p>` 标签并不表示相同输入。

本轮从52条因子结果中，以 `(exponent,sigma,B1,发现的因子)` 唯一关联结果JSON，
使用其 **发现该因子之前** 的 `known-factors` 恢复实际整数：

```text
M = 2^p - 1
N_before_curve = M / product(known-factors_before_curve)
N_next_curve  = N_before_curve / factor_found
```

每次除法均检查整除。结果JSON先输出旧known-factors，再追加新因子，依据参考源码
[结果JSON输出](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:10095)
与[更新known-factors](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:10142)。
此外，用26条NF完成结果的校验码、时间和因子列表独立核对各任务最终N；
另一个任务以找到因子、余因子为PRP结束，没有NF完成行。

全部134次实际N均恢复；100条完整记录形成44个 `(精确整数N,B2档位)` 分组。
与GPU比较时使用**完整整数相等**，而非位宽接近：得到6组余因子配对、3组
完整梅森单次对照。其余35组CPU数据只作为上下文，不计算GPU速度比。

日志与结果JSON时间相差3600秒，由52条sigma匹配记录一致推导；关联NF记录时
同时核对校验码和这个偏移，避免混入同目录更早实验的结果。

### 2.3 计时边界与限制

```text
T_CPU = Stage2_init + Stage2_main + Stage2_GCD
T_GPU = stage2_full_wall.total = init + main
speedup_GPU = mean(T_CPU) / mean(T_GPU)
```

比值大于1表示GPU更快。均排除Stage1；CPU规划/PRP和GPU自动D扫描、进程启动
不在主指标中。GPU init/main与CPU init/main的内部边界不同，不能把这两个名称
当作语义完全相同的操作集合。

CPU B1=100000，GPU B1=20，sigma不同；CPU实际B2比档位高约3.1%–7.3%。
因此这是**同一待分解整数、相同请求B2档位下的实现对比**，不是相同曲线、
相同实际边界的严格A/B。本轮没有按B2比例或功耗比例修正实测时间。

CPU日志出现432条主窗口PRP完成消息，部分与ECM并行；1939 bits两个较大B2的
CPU样本CV分别23.5%、15.3%。不删除慢样本，保留标准差与中位数。
GPU实验也未隔离GPU0既有负载。实验不能用来断言所有CPU/GPU平台的性能。

日志最后一条曲线GCD完成后，停机时又打印 `Resuming.`。原分析器将此曲线误判为
恢复执行，本轮修正为“只在本曲线终结前识别恢复消息”，完整样本从99条恢复为100条。

## 3. 精确相同余因子的时间对比

以下为均值±样本标准差，单位秒；每格GPU均为3遍。

| 实际N bits | B2档位 | CPU n | CPU完整Stage2 / s | GPU完整Stage2 / s | GPU速度比 |
| --- | --- | --- | --- | --- | --- |
| 1939 | 2.6e10 | 2 | 2.472 ± 0.011 | 1.785 ± 0.008 | 1.38× |
| 1939 | 2.6e11 | 3 | 11.649 ± 2.742 | 5.634 ± 0.034 | 2.07× |
| 1939 | 2.6e12 | 3 | 42.537 ± 6.521 | 18.532 ± 0.383 | 2.30× |
| 7995 | 2.6e10 | 4 | 6.110 ± 0.116 | 14.553 ± 0.107 | 0.42× |
| 7995 | 2.6e11 | 4 | 29.620 ± 1.005 | 38.895 ± 0.273 | 0.76× |
| 7995 | 2.6e12 | 4 | 109.987 ± 4.481 | 182.948 ± 0.106 | 0.60× |

1939 bits = `M2003 / (4007 × 6588622714946609)`；7995 bits = `M8011 / 80111`。
若改用两侧中位数，1939 bits速度比分别1.39、1.79、2.50；7995 bits为0.42、0.77、0.60。
方向不变，但1939 bits中档B2的均值明显受慢样本影响。

![相同N的CPU/GPU完整Stage2时间](D:/code/MPA-OpenCl/docs/figures/prime95_gpu_stage2_20261009_exact_times.png)

若把GPU整个外层进程墙钟作为分母，1939 bits比值为1.11、1.91、2.22；7995 bits
为0.41、0.75、0.60。该口径对短曲线影响更大，但CPU没有对应的独立进程启动计时，
因此只是GPU启动成本敏感性参考。

![GPU速度比，1为等速](D:/code/MPA-OpenCl/docs/figures/prime95_gpu_stage2_20261009_speedup.png)

完整梅森单次对照另有3组：M503前两档CPU/GPU为1.891/0.512s、5.012/1.356s；
M7001最大档为140.226/109.272s。两侧都只有1条，输入仍包含可发现的因子，
不能混进稳定余因子均值。

## 4. 全位宽趋势与B2关系

![实际N位宽与耗时](D:/code/MPA-OpenCl/docs/figures/prime95_gpu_stage2_20261009_N_scaling.png)

横轴是恢复后的N位宽，1000 bits一格；纵轴取对数。CPU空心点仅用于展示全部记录，
实心点才有精确相同余因子的GPU数据。CPU在同一p下也可能有多个位宽，不能
将这些点按p平均后与GPU的单一余因子比较。

相同N的三个B2档位相距10倍，利用首末两档给出描述性端点斜率：

```text
beta = log(T(B2_high)/T(B2_low)) / log(100)

1939 bits: CPU beta=0.618; GPU beta=0.508
7995 bits: CPU beta=0.628; GPU beta=0.550
```

只有三个档位，CPU有波动、B2取整、FFT/D/P选择变化；这些数不是经过验证的预测公式。
CPU也不能依据当前44个异质输入直接拟合统一 `T(N bits,B2)`：模数变化、FFT类型
与选形均会影响结果。GPU原实验1939–7995 bits的局部经验式见
[已有GPU报告](D:/code/MPA-OpenCl/docs/STAGE2_N_BITS_TIMING_20261008.md)，不将它用于
修复后功耗或未配对CPU输入的“校正”。

## 5. 时间主要花在哪里

![CPU/GPU阶段百分比](D:/code/MPA-OpenCl/docs/figures/prime95_gpu_stage2_20261009_phases_percent.png)

CPU图使用实测父计时与 `PolyG`、`PolyH` 计时；其余主阶段为父计时残差。
CPU `PolyG` 包含GPU单独列出的点生成工作，见
[PolyG计时边界](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:9451)。
`PolyF up/down` 存在首切片乘切片数的外推，不能堆叠为实测阶段；其实际执行
时间已经包含在主阶段残差中。GPU S4/NTT事件计时嵌套在多个阶段内，也不重复相加。

### 5.1 7995 bits、B2=2.6e12

- CPU完整109.987s：init8.156s（7.4%）、PolyG63.129s（57.4%）、
  PolyH27.449s（25.0%）、其余主阶段/GCD11.253s（10.2%）。
- GPU完整182.948s：init16.021s（8.8%）、giant34.798s（19.0%）、
  G树90.224s（49.3%）、fold29.001s（15.9%）、下降6.766s（3.7%）、
  inverse2.328s、G叶准备3.026s，其余约0.785s。
- GPU giant、G树、fold合计154.023s，约84.2%；这些是原生墙钟模块，
  不能把它们全部叫作NTT时间。
- GPU嵌套S4归约事件46.295s，占总墙钟比值25.3%。仅按串行Amdahl近似完全
  扣除这一事件成本，仍约136.7s，大于CPU的110.0s。这个预算近似不计流水重叠、
  等待连锁效应；它说明优化应同时覆盖点算术、选形和多项式批次，不能只盯归约内核。

### 5.2 7995 bits、中档B2

CPU原生init2.571s、main27.050s；GPU原生init10.465s、main28.430s。
GPU baby点生成/归一化单项8.101s，是该档明显的优化对象。由于inverse等阶段归属
不同，不能直接把7.894s的init差额全部归因于同一个操作，但主阶段总时长已接近。

功耗修复后GPU中档B2的三遍均值从38.895→34.920s，减少10.2%。这缩小了差距，
但没有消除差距；低B2的初始化与最大B2的批次问题仍需各自分析。

## 6. 两个关键算法差异

### 6.1 待分解N与算术承载模数的分离

Prime95参考源码先对待分解整数剥离known-factors：
[构造并除去已知因子](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:1162)。
但ECM的 `gwsetup(k,b,n,c)` 仍使用原始特殊形式，而非把余因子交给通用模数入口：
[初始gwsetup](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:7208)、
[Stage2选FFT时gwsetup](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:7990)。
求逆则使用实际待分解的 `ecmdata->N`：
[ecm_modinv](D:/code/MPA-OpenCl/.refactor/p95v3106b01.source/ecm.cpp:2992)。

当前GPU只有输入本身严格为 `2^p−1` 才启用S4梅森归约：
[exact_mersenne判定](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:2389)。
本轮余因子记录的 `point_mersenne=0`、`reduction=division`，所以GPU与CPU即使
目标N相同，也没有使用相同的归约策略。参考源码为31.6b01，实测程序为31.4b05；
日志能确认实际FFT类型/长度，但尚未通过31.4二进制插桩逐项确认该内部策略。

候选方案：令 `M=2^p−1`、`N|M`，设备上的加乘/多项式计算承载于M，求逆、
非单位判定及最终GCD针对N。数学依据是自然映射 `Z/MZ → Z/NZ` 保持加乘。
这不要求N与 `M/N` 互素。若某个分母在N中可逆，则可在N中求逆并提升到M中
参与后续运算，投影回N仍正确。

**这只是可行性方向，不是已经实现或测得收益。** 工程上需要分别维护目标N和
算术M；所有单位/零判定、oracle比对、存档身份、输出规范化必须采用正确的模数。
7995→8011 bits仅多16bits，但64-bit limb数从125→126；仍可能触发打包/NTT长度
台阶，不能仅按位数比例推算净收益。

直接把输入换成完整M8011并不能实现这个方案：原GPU完整梅森对照在最大B2耗时
692.507s，其中544.441s来自非单位G叶准备/回退。那次的求逆/GCD目标也变成了M，
重新遇到80111；必须与“仍分解N，只在M中承载加乘”区分。

### 6.2 内存规划导致多项式批次数不同

7995 bits、最大B2：

```text
CPU: D=1531530, P=138240, stage2 FFT=AVX-512:512, Using=3917 MB
     日志每条曲线12次 PolyG、11次 PolyH
GPU: D=810810,  P=77760,  G=42, fold=41次
     最大fold NTT L=2^27, NTT full峰≈3186 MiB, fold owner≈519 MiB
```

CPU使用更大的D/P，并采用自己的切片组织；GPU受到NTT工作区与驻留owner预算
影响而使用更小P，因此巨点/G树/fold批次更多。两侧G计数、切片与覆盖边界不完全
相同，不能用42/12直接预测加速比，但这个差异能解释为何仅优化单次NTT还不够。

固定NW按P线性估计，GPU P若从77760升到138240，fold owner约需
`519 × 138240/77760 ≈ 923 MiB`，超过当前640MiB；NTT也可能从2^27升到2^28，
大缓冲约从3GiB变6GiB。需要按**同时存活的分配生命周期**确认8GiB设备是否能容纳，
不能把各模块峰值直接相加，也不能只增加fold预算就宣称可用。

## 7. 建议的优化与下一次测量顺序

1. **先验证特殊模数承载路径。** 冻结 `N=M8011/80111`、同一Stage1点，分别选择
   通用N归约与M承载/N求逆-GCD。先核对投影后的中间结果与最终因子，再比较
   点生成、S4、G树和总墙钟；避免完整M的非单位回退污染结论。
2. **联合规划D/P、arena与fold owner。** 记录NTT长度台阶和真实峰值生命周期，
   对候选D进行小范围实测排名；最大B2优先减少G树/fold批次。不能用CPU的4GiB
   Using直接对应GPU显存预算。
3. **中小B2单独优化baby点和初始化。** 7995 bits中档baby8.1s、低档G叶准备2.2s
   有明确证据；继续看点生成的复用/批处理、准备与传输。
4. **在固定选形后再比较NTT与通用S4微优化。** S4事件占比6.9%–25.3%随位宽变化，
   G树计时不能直接替代NTT内核计时；需要拆开归约、变换和host等待。
5. **下一轮公平复测冻结输入。** 对9个指数使用相同已剥离N、B1、sigma和实际B2，
   CPU固定亲和性并停用实验外PRP，GPU固定修复后的功耗条件，至少3遍。
   同时报告单曲线初始化成本和连续曲线稳态吞吐率。当前未配对位宽不作推测补齐。

## 8. 工具和复现

工具只读日志与历史测量，不启动Prime95或GPU计算：

```powershell
python tools/log_parser/analyze_prime95_ecm.py `
  --screen D:/code/GIMPS/p95v3104b05.win64/screen.log `
  --worktodo D:/code/GIMPS/p95v3104b05.win64/worktodo.txt `
  --output data/prime95_ecm_20261009/completed `
  --b2-targets 26000000000 260000000000 2600000000000

python tools/log_parser/compare_prime95_gpu.py `
  --cpu-analysis data/prime95_ecm_20261009/completed/analysis.json `
  --results D:/code/GIMPS/p95v3104b05.win64/results.json.txt `
  --gpu-analysis docs/benchmarks/stage2_n_scaling_20261008_analysis.json `
  --output data/prime95_ecm_20261009/comparison

python tools/log_parser/plot_prime95_gpu.py `
  --input data/prime95_ecm_20261009/comparison/comparison.json `
  --output-prefix docs/figures/prime95_gpu_stage2_20261009
```

绘图可选 `--canvas <绝对路径.canvas.tsx>`，输出一个自包含文件，不联网。
比较器拒绝未完成的GPU中间快照；不能唯一恢复的N保留unknown，不猜测。
解析/比较使用标准库，绘图需要NumPy和Matplotlib。

- [精确配对CSV](D:/code/MPA-OpenCl/data/prime95_ecm_20261009/comparison/exact_pairs.csv)：
  均值、标准差、样本数、实际B2范围、D/P/FFT、原始日志行号、N哈希。
- [逐曲线N恢复记录](D:/code/MPA-OpenCl/data/prime95_ecm_20261009/comparison/modulus_runs.csv)：
  known-factors、因子结果JSON行号和传播依据。
- [完整比较JSON](D:/code/MPA-OpenCl/data/prime95_ecm_20261009/comparison/comparison.json)：
  源文件SHA256、134次曲线、任务核对及全部CPU/GPU分组。

原始数据与图片保存在已有忽略目录内；报告及通用分析脚本可纳入版本控制。

## 后续研究

[梅森承载与D/P、显存联合规划](D:/code/MPA-OpenCl/docs/STAGE2_MERSENNE_CARRIER_MEMORY_PLAN_20261009.md)
补充了target/carrier分离证明、GMP-ECM参考、Auto B2布局口径差异和NTT台阶的
并存显存下界，并给出实施顺序；未新增GPU性能实测。
