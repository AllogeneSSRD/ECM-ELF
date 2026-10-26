# Tensor NTT 长度、batch 与提交开销

## 当前结论

**精确Tensor40候选已实现并通过GMP门禁，但当前测量不支持接入生产。** 在RTX4060 Laptop默认55W、1800MHz上限条件下，batch=1的14个长度点均未超过生产CUDA完整卷积；扩大短batch、调整CTA或改用Graph提交也未得到可复现的Tensor优势。生产仍使用原CUDA内核和自动outer策略。

单卡CUDA/Tensor混合并发已实测，确有GPU内核重叠；19个Graph形状均未超过相同工作量的最优纯CUDA控制。96-bit重构与寄存器约束候选N96r64减少约5–6%的函数静态指令，局部相对Tensor40有小幅改善，仍没有部署资格。具体条件与证据见[单卡并发](#单卡cuda--tensor并发与96-bit候选)。

较小长度更值得继续减少提交开销：相同一次CUDA卷积的Graph诊断，在2⁸/2¹⁴/2¹⁸长度观测到约29.2%/14.7%/8.0%的局部时间下降；2²²以上不足1%。这些是分开采集的算子诊断，没有完整Stage2曲线或交错native/Graph验收，不能作为生产提速承诺。

生产CUDA路径的Graph缓存已有独立的[完整Stage2验证](NTT_GRAPH.md)：固定D、普通/承载、不同batch及P=155520没有显示稳定收益。它不将Tensor候选接入生产，也不改变本页独立算子的计时边界。

## 实现、输入与统计条件

- 日期2026-10-15；设备GPU1 RTX4060 Laptop，24SM、8GiB，UUID`GPU-8a67b1f8-ef1c-3177-a822-813a7ac2224d`，驱动610.88。GPU0的4070Ti保持既有工作，CPU/驱动竞争未隔离。程序未修改显卡设置。
- 用户约束是1800MHz上限与默认55W，**不是持续固定1800MHz**。六组正式扫描共170个500ms遥测样本；利用率≥50%的108个样本，SM频率1455–1800MHz、中位1665MHz，功耗14.06–55.39W、中位54.34W，温度56–67°C。原始空闲样本含590.01W异常读数，保留但不作为功耗结论；不计算能效或硬件周期。
- 同一个sm89/CUDA13.3.73/MSVC14.51.36231/GMP Zen3探针，EXE SHA256=`ef669d8fb7558752e81b342a73012fa93464061fb5a1a9fa2bac3c736e2f6264`，固定PTX3与生产规范减法，per-thread默认流。直接包含当前生产NTT源码，没有使用旧四fold算术或naive CUDA作为对照。
- q=2⁶⁴−2³²+1。L表示单条NTT长度，b表示batch，二者均不等于待分解N的位数。输入为确定性生成的canonical域元素；inverse使用独立乘数，卷积为两个forward加pointwise/1/L融合inverse，输出仍为域卷积，不运行Kronecker carry或S4模N归约。
- 默认tile为2^min(12,k)，生产CUDA CTA512，Tensor主扫描CTA256。所有路线共用同一outer/radix计划与缓存根表，候选只替换tile。k21–23为四趟forward；k24启用coop/M6，k25–26为coop/M8，均为三趟；长度曲线的变化不只来自L增加。
- native主扫描14长度×3操作；另有单操作Graph11格、短batch16格、CTA12格、16次串联Graph27格、短batch串联Graph14格，合计122组，每组3实现、每实现11个交错样本，共4026个正式样本。报告中位数与MAD，没有置信区间；每个样本的全部A/B输出在计时外与同进程生产CUDA结果比较，累计60,953,519,616个word、bad=0。
- CUDA事件覆盖完整操作的kernel执行及提交供给造成的设备空隙，host墙钟另记；分配、根生成、数据填充、参考构造、校验和Graph捕获不计入。没有丢弃的warmup样本，但CUDA参考先执行，不能声称完全相同的冷缓存条件。串联Graph统计除以16；图中主要Graph对照使用一次操作，不混合两种边界。

上述排除项适用于CUDA事件主时间。辅助`host_seconds`从start事件提交返回后量到end同步返回，可能包含此前排队的填充尾部等待；它不等于纯CPU开销，不能与CUDA事件时间简单作差归因。

## 完整卷积随长度变化

batch=1、Tensor CTA256、native提交，各11次中位数，单位ms。“增加”是Tensor40时间相对CUDA增加比例，正值表示变慢。

| L | CUDA | Tensor64 | Tensor40 | Tensor40时间增加 |
| --- | ---: | ---: | ---: | ---: |
| 2⁸ | 0.02458 | 0.02662 | 0.02560 | 4.2% |
| 2¹⁰ | 0.02765 | 0.03277 | 0.03072 | 11.1% |
| 2¹² | 0.05427 | 0.07680 | 0.07565 | 39.4% |
| 2¹⁴ | 0.07680 | 0.09626 | 0.09603 | 25.0% |
| 2¹⁶ | 0.11366 | 0.13616 | 0.13619 | 19.8% |
| 2¹⁸ | 0.20378 | 0.23040 | 0.23245 | 14.1% |
| 2¹⁹ | 0.35942 | 0.40653 | 0.40758 | 13.4% |
| 2²⁰ | 0.63283 | 0.71270 | 0.71680 | 13.3% |
| 2²¹ | 1.35171 | 1.51146 | 1.51142 | 11.8% |
| 2²² | 3.13344 | 3.45907 | 3.45190 | 10.2% |
| 2²³ | 7.86330 | 8.51763 | 8.51661 | 8.3% |
| 2²⁴ | 12.31565 | 13.70522 | 13.69907 | 11.2% |
| 2²⁵ | 27.72378 | 30.69645 | 30.35546 | 9.5% |
| 2²⁶ | 56.13773 | 61.03757 | 61.85472 | 10.2% |

![完整NTT卷积的长度对照](../../data/figures/ntt_tensor_length_20261015_report.png)

2²⁶的CUDA/Tensor40 MAD分别约0.410/1.149ms，受功耗与频率变化影响；该点不能解释为固定频率下的内核效率差。native的少数单方向短NTT出现小幅正收益，但Graph诊断未显示Tensor胜出，不能由此部署按长度切换。

## batch、CTA 与资源

短batch覆盖k=8/10/12/14/16/18/19/20，b=8/32。native的k8/b8曾显示Tensor快约12.5%；同形状16次串联Graph中，Tensor64/40分别慢约27.6%/27.6%，局部正点未复现。更有工作量的k18/b32，CUDA/64/40为6.202/6.854/6.844ms；k20/b32为24.771/27.525/27.330ms。增大并行数量没有改变当前候选的整体方向。

![短NTT批处理对照](../../data/figures/ntt_tensor_length_20261015_report_batch.png)

CTA复测覆盖k12/18/24/26，Tensor CTA128/256/512。CTA512改善单tile小长度，但仍未超过生产CUDA；k24的Tensor40时间比同组CUDA分别多约26.1%/11.9%/14.2%。这些复测按各自同组CUDA比较，不用不同轮次的绝对值拼接阈值。

当前资源API的t12结果：CUDA正/逆均40 registers/thread，Tensor64为61/62，Tensor40为62/59，均LOCAL0。32KiB shared下，CUDA CTA512容量上限3块/SM；Tensor CTA256为3块，CTA512为2块。这是容量查询，不是实测occupancy、stall或周期计数。

## 64→40 MMA 为什么没有转化为加速

Tensor64以64次u8×u8 MMA完成DFT16。Tensor40将其分为偶/奇两个DFT8，利用标准8次根的有符号稀疏byte平面，将MMA减到40次，再增加根乘与合并蝶形；精确表示和负和修正见[数学资料](../reference/TENSOR_NTT_MATH.md#有符号稀疏根与-dft8-分解)。

最终SASS确实为64/40条IMMA，减少37.5%，不是只改变命名。但整个tile函数的静态SASS指令数，forward由1440增至1504，inverse由1304增至1360；寄存器占用也没有同步下降。静态代码包含循环体/控制路径，不等于动态执行指令数。数据拆分、符号carry、输出合并与通用根乘抵消MMA节约，是源码与现象支持的解释，尚无硬件计数器归因。outer不变也会稀释tile的潜在收益。

不能将Tensor峰值MAC/s、减少的MMA数量或naive dense DFT对照的倍数，直接换算为Goldilocks NTT或ECM收益。当前结果不证明其他矩阵布局、radix、GPU或32/52-bit域一定无效，只排除本次已测路线的自动部署资格。

## 单卡CUDA / Tensor并发与96-bit候选

### 问题、合同与设备

Ada支持并发kernel，但是否得到收益由资源和依赖决定。本探针的设备属性 `concurrentKernels=1`；使用同一context中的两条nonblocking流，分别计算两个独立的完整域卷积，控制流事件fork/join。CUDA/Tensor kernel区间重叠已由Systems确认；该证据不证明同一SM同一周期发射。并发资格与资源约束参见[NVIDIA CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#concurrent-kernel-execution)，Ada容量见[NVIDIA Ada Tuning Guide](https://docs.nvidia.com/cuda/archive/13.0.0/ada-tuning-guide/index.html)。

两个任务各有独立A/B，只读共用已缓存计划和根表；初始化上传同步完成后才允许非阻塞流读取。不复用生产arena的可写scratch，不将此探针解释为已支持多曲线Stage2并发。每任务为两个forward和一个融合pointwise/1/L的inverse；不包括pack、carry、S4、树、GCD或Stage1。

2026-10-15，GPU1 RTX4060 Laptop，24SM，UUID与本页长度扫描相同，驱动610.88，保持默认55W与1800MHz上限，GPU0仍运行原任务。3432个正式样本的90个利用率≥50%的500ms遥测，频率1500–1800MHz、中位1755MHz；功耗10.35–55.47W、中位54.755W，温度55–70°C。空闲异常读数仍保留，不用于功耗结论；不声称固定频率或能效收益。

最终二进制SHA256=`ebc499d64d4433e1390782e360d36ffe207f65d6f68c99d18813723f35df45ac`，sm89、CUDA13.3.73、MSVC14.51.36231、GMP Zen3，PTX3/canonical减法、per-thread默认流。EXE、DLL、源码闭包和构建身份冻结在各实验的build目录。原CUDA及Tensor40的正逆tile指令序列在窄重构约束前后逐函数一致，避免用被改变的控制组宣称收益。

### 计时与完整对照

C为生产CUDA（CTA512），T40为原Tensor40，N40为日志中的N96r64别名。每形状有13个scene：三个单任务，以及CC、CT40、TT40、CN40、NN40的串行／并行。两任务工作量相同；以CC串行和并行中较快的中位时间作为纯CUDA控制，不只与较慢的混合串行比较。

- Graph正式19形状×13场景×11次=2717个样本，覆盖k=8/11/12/18/20/21/22/23/24/25，batch=1/8/16/720/2880以及Tensor CTA128/256/512；实际组合见原始cases。
- native提交补充5形状×13场景×11次=715个样本。各scene交错测量，没有丢弃warmup，CUDA参考在计时前执行，不声称完全相同的冷缓存。
- 主时间是整个迭代序列的GPU makespan，包含fork/join、kernel及提交供给造成的空隙；分配、填充、参考、捕获和验证均在计时外。填充已同步，host墙钟另记。
- 两任务的Conv/s=2·batch·iterations/makespan；单任务分子为batch·iterations。不能把每个kernel时间相加冒充并发墙钟，也不把该吞吐称为ECM curves/s。

下表为Graph、Tensor CTA256的部分结果。并发倍数=CT40串行/CT40并行；“增加”是混合并行时间相对最佳纯CUDA增加，正值表示更慢。

| L | batch/任务 | CT40并发倍数 | CT40时间增加 | CN96r64时间增加 |
| --- | ---: | ---: | ---: | ---: |
| 2¹² | 1 | 1.615 | 45.49% | 44.72% |
| 2¹¹ | 720 | 1.023 | 9.38% | 8.64% |
| 2¹¹ | 2880 | 1.015 | 7.47% | 6.63% |
| 2¹⁸ | 1 | 1.221 | 8.54% | 6.89% |
| 2¹⁸ | 8 | 0.956 | 11.24% | 11.19% |
| 2²¹ | 1 | 0.915 | 15.38% | 15.82% |
| 2²² | 1 | 0.802 | 31.43% | 31.31% |
| 2²³ | 1 | 0.988 | 5.28% | 4.63% |
| 2²⁵ | 1 | 1.020 | 4.34% | 4.73% |

小任务相对混合串行可以明显提速，纯CUDA同样可并发且单任务更快。大batch的提交成本已摊薄；扩大并行任务还可能增加资源与缓存压力。native的k8/b1曾比最优纯CUDA快0.59%，同形状Graph控制慢27.59%，不将这一小正点作为Tensor优势。

![混合并发与窄重构](../../data/figures/ntt_tensor_concurrency_20261015.png)

![不同提交方式的并发效率](../../data/figures/ntt_tensor_concurrency_20261015_efficiency.png)

### Systems与并发范围计数器

管理员Systems2026.1.3的4个单独范围保留完整输出检查，各重复32次操作。下列占比为Tensor kernel区间与常规CUDA tile/outer区间重叠时间，除Tensor区间并集：

| 形状／调度 | Tensor区间并集 | CUDA/Tensor重叠 | 重叠占比 |
| --- | ---: | ---: | ---: |
| L4096×1，混合并行 | 2.063ms | 1.227ms | 59.47% |
| L262144×1，混合串行 | 4.061ms | 0 | 0% |
| L262144×1，混合并行 | 4.422ms | 4.036ms | 91.26% |
| L2048×720，混合并行 | 18.817ms | 3.229ms | 17.16% |

“Tensor kernel”本身仍执行大量整数CUDA指令；区间重叠不是Tensor和普通算术资源完全独立的证明。中等任务有大量重叠仍慢于纯CUDA，说明单看overlap不足以选型。大batch重叠较少，与单任务已大量占用资源的现象一致；不能仅凭时间线确定同SM驻留分布。

![CUDA/Tensor内核时间线](../../data/figures/ntt_tensor_concurrency_20261015_timeline.png)

管理员NCU2026.2.1使用application-range重放，在ProfilerStart/Stop之间采集完整并发范围，保留fork/join关系；三组为CC、CT40、CN96r64，各9次应用重放。明确none时钟/缓存控制，核对报告设备为4060 Laptop/24SM，各次输出检查通过。[NVIDIA范围重放](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#application-range-replay)说明其范围指标与逐kernel重放的区别。

L=2¹⁸、batch=1、32次迭代的CT40范围，时钟约1792MHz；整数ALU峰值百分比53.61%、Tensor管线活跃2.97%、DRAM约0.026%。CN96r64对应约54.82%/3.08%/0.054%。这些是整个范围、设备平均的各自峰值口径，不能相加，也不证明每个SM的瞬时分布。结果说明当前MMA实现没有把主要工作转为Tensor计算，低Tensor利用率并不自动提供可利用的免费吞吐。

纯CUDA范围的Tensor计数为0，作为计数器控制。该形状工作集可驻留L2，低DRAM只适用于该范围；不外推大NTT访存。插桩秒数排除在正式性能表之外。NCU中设备筛选和range/graph参数组合产生的空报告保留为失败证据，不参与这些指标。

### N96r64优化与资源边界

候选将signed和拆为三个低32-bit word和一个signed高项，利用2⁹⁶≡−1直接归约，保留40次MMA。范围证明见[96-bit重构](../reference/TENSOR_NTT_MATH.md#96-bit有符号高项重构)。独立GPU入口限定512线程、至少2块/SM，不改变CUDA和原Tensor40的资源合同。

| 实现 | forward / inverse静态指令 | forward / inverse registers | CTA512块/SM上限 | LOCAL |
| --- | ---: | ---: | ---: | ---: |
| Tensor40 | 1504 / 1360 | 62 / 59 | 2 / 2 | 0 |
| N96r64 | 1416 / 1288 | 64 / 64 | 2 / 2 | 0 |

静态指令分别少5.85%/5.29%，IMMA仍为40条；静态条目不等于动态指令或周期。64-register约束消除了未约束候选forward=66时CTA512只能1块/SM的资源拐点，未引入stack或spill。该边界按本机65536 registers/SM计算，并用资源API核对。

19个Graph形状中，N96r64单任务相对Tensor40时间变化从慢0.68%到快1.97%；其中17个正点、2个负点，多数差异不足2%，不能承诺稳定改善。所有形状仍慢于相同CUDA单任务；混合调度同样未超过最佳纯CUDA。

资源API的t12条件下，Tensor CTA256最多3块，即768线程；CUDA CTA512最多3块，即1536线程。加上共享寄存器、shared与整数管线，不能按“CUDA核+Tensor核”简单叠加峰值。该API是容量上限，不冒充混合运行实测occupancy。

### 正确性、容量与证据

四实现的独立GMP forward/inverse/卷积门禁检查14407680 word；串行／并发小形状再查1113600 word，bad=0。故意翻转一个N96r64输出时恰检出bad=1并失败退出。3432个正式样本累计比较53758735360个最终A/B word，bad=0；全部探针持有缓冲与FuseBase释放后live=0。大形状比较对象是同进程生产CUDA，独立GMP覆盖较小形状，不将它描述为每个大输出都有独立GMP证明。

两任务的A/B与参考共64·batch·L bytes，根包6656B、错误数8B，计划/根表另计。最大batch×L=2²⁵，主数组2GiB；Graph/驱动/context不在该payload内。计时内H2D/D2H=0，计时外完整输出比较仅回传8B错误数；不能将该探针容量和其他模块峰相加。

- 原始证据：`data/experiments/ntt_tensor_concurrency_20261015/`。`graph_r64`、`native_r64`为最终同二进制正式矩阵；`initial`、`n96_initial`、`graph`保留其他候选与失败／探索证据，不拼接最快结果。
- Systems为`nsys_*`；最终NCU为`ncu_app_all_cc_checked`、`ncu_app_all_ct`、`ncu_app_all_cn`。各报告保留命令、重放输出和CSV；空报告排除。
- 汇总和图见`data/benchmarks/ntt_tensor_concurrency_20261015_final/`及本节图片；SASS逐函数对照为原始目录的`sass_comparison.json`。
- [复现工具](../../tools/bench/README_NTT_TENSOR_CONCURRENCY.md)、[提交与检查](../../tools/bench/ntt_tensor_concurrency_probe.cu#L8)、[窄重构](../../tools/experiments/cuda/ntt_tensor/goldilocks.cuh)、[约束入口](../../tools/experiments/cuda/ntt_tensor/goldilocks.cuh)。

## 设备、主机及传输容量

本探针的四个数据数组A/B/参考A/参考B同时存在，共32bL bytes；缓存计划/根表另计，Tensor双向packed roots共6656B，错误计数8B。最大测试bL=2²⁶，主数组恰为2GiB；该形状同时存活的计划基础与缓存表为25,304,064B。Graph对象的driver分配不在该请求账本内。原始计数器分别报告自身范围，不相加冒充进程峰；NVML观测整卡最大2157MiB，采样可能漏掉瞬时峰且含上下文。

性能输入由GPU生成；计时内H2D/D2H为0，单次全输出比较只回传8B错误数。根表/计划建立及GMP小规模oracle在计时外。Tensor64/40不增加完整L级工作数组；原始大矩阵源码并未导入生产。

## 正确性与复现

独立GMP门禁覆盖k6/8/10/12/13/14、batch1/3、零值/q−1/边界交错/dense/近q输入、forward/inverse/卷积与B语义；三实现合计10,805,760 word、bad=0。独立故意翻转一个Tensor40输出，恰检出bad=1并返回失败。GMP门禁CTA256；其他CTA在性能扫描中与生产CUDA逐字比较。所有调用结束时探针持有缓冲及FuseBase live=0；该检查不包含driver/context/Graph内部的未计账分配。

- 原始证据、冻结EXE/GMP/manifest/源码、SASS及各扫描目录：`data/experiments/ntt_tensor_length_20261015/`。`initial/`为开发检查，不混入正式122组。
- 正式记录：`{native,graph_single,batch,cta,graph,small_graph}/{measurements.json,summary.csv,telemetry.csv}`；汇总为`data/benchmarks/ntt_tensor_length_20261015_report_analysis.json`。
- [构建脚本](../../tools/build/test/build_ntt_tensor_length_probe.ps1)、[基准入口](../../tools/bench/benchmark_ntt_tensor_lengths.py)、[图表工具](../../tools/bench/plot_ntt_tensor_lengths.py)、[复现说明](../../tools/bench/README_NTT_TENSOR_LENGTHS.md)。
- [完整算子计划](../../tools/bench/ntt_tensor_length_probe.cu#L43)、[GMP门禁](../../tools/bench/ntt_tensor_length_probe.cu#L90)、[计时与Graph](../../tools/bench/ntt_tensor_length_probe.cu#L120)；[signed精确累加](../../tools/experiments/cuda/ntt_tensor/goldilocks.cuh)、[真实tile](../../tools/experiments/cuda/ntt_tensor/goldilocks.cuh)。生产共用[forward](../../src/ecm/cuda/ntt/runtime.cpp)及[inverse](../../src/ecm/cuda/ntt/runtime.cpp)的outer实现。

## 待推进

1. 继续Graph时评价包含pack/carry/S4的稳定完整请求；现有两个NTT Graph的完整曲线结果见[Graph性能](NTT_GRAPH.md)。需保留缓存租约、分块尾部、检查排空及预算边界，不能用本页单算子收益代替验收。
2. 独立比较k21–23的outer跨pass组织，减少完整数组遍历；当前四趟到三趟的潜力尚未实测，不直接扩大生产coop阈值。
3. Tensor的96-bit重构、64-register入口及同卡混合并发已测，尚无CUDA优势。继续时优先减少拆字节/重构和DFT8合并的整数工作；更大radix或warp分工需独立精确门禁与同工作量纯CUDA控制。不能按空闲Tensor峰值或更多流决定部署。
4. Tensor候选没有接入生产或tune；本页算子与并发范围结果不替代固定D完整Stage2及其他GPU验收。当前优先验证k21–23 outer跨pass组织，以及正常请求的准备/同步边界，通过既有性能保护后才改变默认。
