# Tensor NTT 长度、batch 与提交开销

## 当前结论

**精确Tensor40候选已实现并通过GMP门禁，但当前测量不支持接入生产。** 在RTX4060 Laptop默认55W、1800MHz上限条件下，batch=1的14个长度点均未超过生产CUDA完整卷积；扩大短batch、调整CTA或改用Graph提交也未得到可复现的Tensor优势。生产仍使用原CUDA内核和自动outer策略。

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

## 设备、主机及传输容量

本探针的四个数据数组A/B/参考A/参考B同时存在，共32bL bytes；缓存计划/根表另计，Tensor双向packed roots共6656B，错误计数8B。最大测试bL=2²⁶，主数组恰为2GiB；该形状同时存活的计划基础与缓存表为25,304,064B。Graph对象的driver分配不在该请求账本内。原始计数器分别报告自身范围，不相加冒充进程峰；NVML观测整卡最大2157MiB，采样可能漏掉瞬时峰且含上下文。

性能输入由GPU生成；计时内H2D/D2H为0，单次全输出比较只回传8B错误数。根表/计划建立及GMP小规模oracle在计时外。Tensor64/40不增加完整L级工作数组；原始大矩阵源码并未导入生产。

## 正确性与复现

独立GMP门禁覆盖k6/8/10/12/13/14、batch1/3、零值/q−1/边界交错/dense/近q输入、forward/inverse/卷积与B语义；三实现合计10,805,760 word、bad=0。独立故意翻转一个Tensor40输出，恰检出bad=1并返回失败。GMP门禁CTA256；其他CTA在性能扫描中与生产CUDA逐字比较。所有调用结束时探针持有缓冲及FuseBase live=0；该检查不包含driver/context/Graph内部的未计账分配。

- 原始证据、冻结EXE/GMP/manifest/源码、SASS及各扫描目录：`data/experiments/ntt_tensor_length_20261015/`。`initial/`为开发检查，不混入正式122组。
- 正式记录：`{native,graph_single,batch,cta,graph,small_graph}/{measurements.json,summary.csv,telemetry.csv}`；汇总为`data/benchmarks/ntt_tensor_length_20261015_report_analysis.json`。
- [构建脚本](../../tools/build/test/build_ntt_tensor_length_probe.ps1)、[基准入口](../../tools/bench/benchmark_ntt_tensor_lengths.py)、[图表工具](../../tools/bench/plot_ntt_tensor_lengths.py)、[复现说明](../../tools/bench/README_NTT_TENSOR_LENGTHS.md)。
- [完整算子计划](../../tools/bench/ntt_tensor_length_probe.cu#L43)、[GMP门禁](../../tools/bench/ntt_tensor_length_probe.cu#L90)、[计时与Graph](../../tools/bench/ntt_tensor_length_probe.cu#L120)；[signed精确累加](../../tools/bench/ntt_tensor_goldilocks.cuh#L117)、[真实tile](../../tools/bench/ntt_tensor_goldilocks.cuh#L211)。生产共用[forward](../../src/cuda/stage2/ntt_runtime.cuh#L1678)及[inverse](../../src/cuda/stage2/ntt_runtime.cuh#L1734)的outer实现。

## 待推进

1. 继续Graph时评价包含pack/carry/S4的稳定完整请求；现有两个NTT Graph的完整曲线结果见[Graph性能](NTT_GRAPH.md)。需保留缓存租约、分块尾部、检查排空及预算边界，不能用本页单算子收益代替验收。
2. 独立比较k21–23的outer跨pass组织，减少完整数组遍历；当前四趟到三趟的潜力尚未实测，不直接扩大生产coop阈值。
3. 若继续Tensor路线，应先降低总标量指令/寄存器和重构成本；可评估利用2⁹⁶≡−1的窄重构，或更大radix/outer映射。只减少MMA条数不再作为部署依据。
4. 本轮未运行NCU、完整Stage2或其他GPU，未更新生产内核、tune表或D模型。候选通过固定D全曲线及既有性能保护后，才考虑更改默认。
