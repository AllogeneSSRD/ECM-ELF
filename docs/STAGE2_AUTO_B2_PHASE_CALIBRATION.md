# Auto B2：完整阶段标定、摊销成本与第一轮离线规划

日期：2026-10-05。延续 [设计](D:/code/MPA-OpenCl/docs/STAGE2_AUTO_B2_TUNE_DESIGN.md) 的P2/P3。当前交付包括分阶段成本模型、有限实测范围内的离线B2/D/path联合搜索，以及可选 `--factor-only` 优化。原生 `--auto-b2`、INI/worktodo自动值接入、大B1标定、总显存租约和并行曲线吞吐量策略仍待推进。

## 1. 实现入口

- [measure_ecm_costs.py](D:/code/MPA-OpenCl/tools/bench/measure_ecm_costs.py)：串行采集GPU1的Stage1批次摊销、Stage2各阶段、驻留/回退和原始日志。独立参考检查在计时外；中断后仅允许相同身份/配置继续。
- [ecm_cost_model.py](D:/code/MPA-OpenCl/tools/bench/ecm_cost_model.py)：Kruppa相对价值、非负阶段拟合、真实giant分块及chain/ladder成本。
- [fit_ecm_costs.py](D:/code/MPA-OpenCl/tools/bench/fit_ecm_costs.py)：按位宽和owner路径分别拟合、诊断及发布带指纹的profile。
- [validate_ecm_costs.py](D:/code/MPA-OpenCl/tools/bench/validate_ecm_costs.py)：先冻结预测文件，再运行未用于模型修订的新B2。
- [audit_ecm_costs.py](D:/code/MPA-OpenCl/tools/bench/audit_ecm_costs.py)：核对原始日志、实际giant路径、必需检查及跨owner/重复曲线的叶子哈希，比较候选排名。
- [plan_auto_b2.py](D:/code/MPA-OpenCl/tools/bench/plan_auto_b2.py)：从实测scope搜索B2/D/owner路径，以原生plan-only复核选中方案的整数几何及当前组件预算。只规划，不执行曲线、不改队列。

独立候选：`build_cuda_cmake/_auto_b2_phase_20261005/native/ecm_cuda_stage2.exe`，SHA256 `d77f571de08947cd4cbe57bdf47177f8f1118023d50843479c329871759daa50`。sm89/PTX3/outer0，24个原始依赖冻结。HostOnly复用经过架构、toolkit、CUDA依赖/object SHA检查的对象；本轮没有改CUDA算术内核。原生产893目录保持。

## 2. 数据合同

GPU1 RTX4060 Laptop，UUID `8a67b1f8ef1c3177a822813a7ac2224d`。外部GPU0任务保持运行。输入为此前CPU/GMP-ECM核验的M2203、M4423、M8191保存点，sigma26、B1=1000、lcm；本轮再次核对CPU坐标及checksum。

Stage1：每种位宽测batch1/12，每组一次预热及两次测量。实际批次使用sigma26+i，**18批次/117个保存点**全部与独立Python参考一致。计时包含原生进程与其保存工作，参考重算在计时外。

Stage2：D=30030/60060/120120，arena4096MiB，owner640MiB或0强制回退。B2=30亿的36条用于拟合，60亿的12条用于诊断。初次模型修订参考过60亿误差，故这12条不再称为独立最终留出。随后冻结模型及预测，新增B2=45亿的**36条独立验证**。共84条算术检查干净的曲线，472,264个抽样GMP系数检查，21组跨路径/重复输出叶子哈希一致。

M8191不是无因子的校准输入：sigma26/B1=1000/B2=60亿找到真实因子338193759479，独立验证 `2^8191 mod factor=1`。首个采集版本错误假定三个输入都不会找到因子，触发的是采集条件失败；原生bad_factors和算术检查为0。后续采集改为验证proper divisor，不丢弃合法因子。这批成本数据测的是明确的factor-only配置，不用于拟合成功概率。

## 3. Stage1每曲线摊销

两次正式样本中位数，单位ms/curve：

| bits | batch1 GPU | batch12 GPU | batch1进程 | batch12进程摊销 |
| --- | --- | --- | --- | --- |
| 2203 | 86.214 | 8.010 | 765.956 | 64.956 |
| 4423 | 175.652 | 16.507 | 1217.229 | 102.939 |
| 8191 | 304.965 | 25.452 | 989.811 | 75.643 |

`gpu_seconds/batch`与`process_seconds/batch`分开保存。GPU时间约降低10.8/10.6/12.0倍是本批B1=1000的批次摊销结果，不是整条ECM加速。进程样本存在明显启动波动，profile保留min/max；两样本没有置信区间。总流程收益使用进程摊销，显式用户秒数可覆盖T1。没有从单个save的TIME字段推断Stage1耗时。

当前T1只支持实测B1=1000、lcm及上述位宽/batch。没有将Prime95的FFT系数或这张表外推到B1=2.6亿；需要补大B1及指数bit长度/缓存成本标定。

## 4. 分阶段模型与真实路径

模型保留baby、affine、F树/初始化、giant、G树、fold、descent、Newton逆元、accum及glue。GCD、oracle drain和命名关闭后仍必要的因子回收处于完整wall计时内。`init−baby−affine`归入ftree项，它还含其他初始化与必需自检费用，不声称是纯F树kernel事件。

基础几何：`P=phi(D)/2`、`I=floor(B2/D)+2`、`G=ceil(I/P)`。多项式工作量按实际整数packing形状计算，沿用此前校核的树/partial tree/Newton计数。时间率是经验秒数，不是GPU周期。

源码 [giant分块](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:9728) 使用256MiB的两坐标预算：

```text
W=ceil(bits/64)
k=max(P, floor(256MiB/(16W)))
chunk=P·ceil(k/P)
每块 n>=32768：chain；否则ladder
```

[chain门槛](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:9782) 原先在统一giant率里遗漏。M8191/D120120/B2=30亿仅24977点，走ladder；D60060/60亿99902点走chain。混合拟合会高估后者约18%。模型现在分别计chain/ladder点工作，并给chain一个实测的每chunk固定项：

```text
T_giant = a_chain·F_chain + b_chain·chain_chunks + a_ladder·F_ladder
F_path = path_points·[6+22log2(B2)/64]
```

括号仍是经验计数特征，不是精确MAC数；分块/路径选择是源代码整数规则。非负二变量最小二乘拟合chain项，ladder独立拟合；其余阶段沿用按各自特征的非负率。blind中D30030/M8191出现一块chain加短尾ladder，也已通过运行日志的chain_chunks核对。

本批shape/G范围有限。不能只使用总体B2区间当成所有D/G组合都已验证；planner还限制P/G范围，拒绝范围外及不匹配的B1、arena、位宽、binary。G=1、cache压力回退及泛型余因子尚未标定。

## 5. 独立验证

[审计摘要](D:/code/MPA-OpenCl/docs/data/ecm_auto_b2_phase_20261005_audit.json)：36条blind曲线完整engine wall误差为 **−7.479%～+5.991%**。三个bits的模型均选D60060/owner640，恰好匹配当前六种实测候选的最快中位数：

| bits | 预测选择 | 最快engine wall中位数 | 所选距最快 |
| --- | --- | --- | --- |
| 2203 | D60060 / resident | 0.895724s | 0% |
| 4423 | D60060 / resident | 2.114318s | 0% |
| 8191 | D60060 / resident | 5.558121s | 0% |

只证明这批候选/预算/位宽的engine成本和排名。没有证明所有47-smooth D的最优性、整个搜索区间每个形状的误差或全进程VRAM可分配。process cold overhead另取训练中位数，保留波动范围；上面的10%验证不覆盖冷启动总时长。

[profile](D:/code/MPA-OpenCl/docs/data/ecm_auto_b2_phase_20261005_profile.json) 和 [便携原始日志/数据](D:/code/MPA-OpenCl/docs/data/ecm_auto_b2_phase_20261005_evidence.json) 保存配置、源码/二进制SHA、实际阶段及预测冻结信息。文件中的本机来源路径用于追溯；嵌入文本换行为LF，source SHA对应原字节。

## 6. 离线Auto B2收益规划

目标遵守此前确认的总流程收益：

```text
K = 0.11343 + 0.88657·[log10(B2/B1)/2]^(1.96617−0.06781log10(B1))
T2 = ratio_adjust·(完整阶段预测 + 实测cold overhead中位数)
score = K/(T1+T2)
```

K是Prime95经验相对价值，不是绝对因子成功概率。当前搜索实测D集合、owner驻留与显式强制回退，对B2做对数网格及G/chain阈值整数邻域比较。所选方案再通过同一原生plan-only核对D/P/I/G/NTT fold_length与当前组件预算。

```powershell
python tools/bench/plan_auto_b2.py --profile docs/data/ecm_auto_b2_phase_20261005_profile.json `
  --save build_cuda_cmake/_auto_b2_phase_20261005/study_factor_only/m4423.save `
  --stage2 build_cuda_cmake/_auto_b2_phase_20261005/native/ecm_cuda_stage2.exe `
  --stage1-batch 12 --output run/auto_b2_plan.json
```

M4423例子选择B2=30亿/D60060/resident，T1≈0.10294s，T2≈2.25773s（engine预测1.63125s加cold中位数0.62648s）。结果 `range_limited=true`：最佳在当前标定区间下边界，**不是已找到通用最佳B2**。下一步必须扩展低端B2/较小D和G=1，而非将下边界作为默认生产倍率。

此工具还没有接入原生`--auto-b2`、INI/worktodo；不改变显式B2语义、不执行曲线。当前只有首条保存点/三个精确Mersenne宽度的受限原型。`process_peak_guaranteed=false`明确组件准入不等于总显存租约。

## 7. factor-only优化

已有引擎 `NTT_NAME_HITS=0` 可以跳过可选prime-witness命名。本轮接入 [CLI](D:/code/MPA-OpenCl/src/core/ecm_cuda_stage2_main.cpp:246) `--factor-only` 与INI `stage2_factor_only=1`，在独立curve worker中设置开关；result附 `requested_factor_only`。

命名保留原默认策略，未自动启用新模式。raw因子可能是复合数，可组合 `--factorize-hits --gp PATH` 获取素数分解；不能用hits=0推断无因子。

同一旧b9af二进制、同save/B2=60亿/D60060、owner640的ABBA诊断：

- 命名开启：16.148278、16.122666s；
- 命名关闭：6.236735、6.231613s；
- 关闭后engine wall降低约61.36%，约2.59倍；raw因子均338193759479，leaf hash均8604703069727795349，必需检查通过。

这是一个真实因子长尾案例，不是所有Stage2曲线的普遍加速。新d77f独立构建的84条标定/验证曲线使用显式factor-only，分开记录名称配置，未把旧命名耗时混入新模型。

## 8. Nsight诊断及优化候选

Nsight Systems 2026.1.3采集GPU1进程树，M4423/B2=45亿/D60060/factor-only。GPU活动首尾间2.171539s，kernel+copy+memset时间区间的并集1.714453s，间隙0.457086s，忙碌比例78.95%。这不包含首个GPU活动之前的启动，也不是全运行NVML利用率。

CUDA API `cudaMemcpy` 634次、合计957.286ms；实际H2D/D2H/D2D DMA合计约32.246ms。API时间含同步/等待，不能将其全部当PCIe传输成本。`cudaDeviceSynchronize`合计620.533ms，kernel launch7326次；API和GPU时间可能重叠，不可相加。

GPU kernel总时间中，两个s2g_ladder调用约469.088ms（28.8%），正/逆tile合计约364.373ms，outer forward/inverse约233.328ms；该几何中种子/点生成和NTT均值得优化。S4归约在原生日志约70ms，不能把此前大生产形状的归约占比直接套用。

Nsight Compute 2026.2.1已尝试采样ladder，驱动返回 `ERR_NVGPUCTRPERM`（设备1性能计数器权限不足）。应用自身正常结束；没有生成计数器报告，没有修改驱动设置。后续可以继续依据Systems时间线与同二进制开关验证，或在计数器权限可用后补occupancy/issue分析。

优先候选：重新验证GPU seed启用后的chain阈值；减少小形状的逐曲线进程/上下文启动；按实测VRAM活跃集检查双曲线并发吞吐；针对NTT小层大量launch/同步考虑批量或graph；扩大B1和B2范围后接入原生收益规划。尚未把这些候选宣称为已完成优化。
