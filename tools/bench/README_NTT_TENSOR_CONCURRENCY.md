# 单卡 CUDA / Tensor NTT 并发

工具测试同一GPU、同一context的两个独立卷积任务。每任务为两个forward和一个融合pointwise/scale的inverse，使用当前生产CUDA outer与Goldilocks算术，候选只替换tile。当前结果见[Tensor性能](../../docs/performance/NTT_TENSOR.md#单卡cuda--tensor并发与96-bit候选)。

## 构建与门禁

入口为 `tools/build/test/build_ntt_tensor_concurrency_probe.ps1 -Build <独立目录> -Arch sm_89`。要求sm80+；默认sm89，per-thread默认流，固定PTX3/canonical减法，记录EXE、GMP与源码身份。构建不自动运行测试。

`ntt_tensor_concurrency_probe.exe <device> --check` 核对CUDA、Tensor64、Tensor40、N96r64的完整GMP forward/inverse/卷积，再核对串行和并发的完整小形状输出。`--fault` 故意翻转一个N96r64输出，预期bad=1且退出码3。

N96r64保留40次MMA，使用96-bit低项与signed高项重构，独立入口限定512线程且至少2块/SM，控制寄存器拐点；数学见[推导](../../docs/reference/TENSOR_NTT_MATH.md#96-bit有符号高项重构)。该限制是实验入口，不是全工程寄存器参数。

## 完整并发算子矩阵

`benchmark_ntt_tensor_concurrency.py --exe <exe> --cases <cases.json> --output <新目录> --device 1 --repeats 11 --execution graph`。

cases为对象数组，每项提供 `name`、`k`、`batch`、`iterations`，可选 `threads=256`。name限字母数字下划线；单条L=2ᵏ，batch为每任务的卷积条数。k范围6–26，batch范围1–4096，CTA为128/256/512；batch×L≤2²⁵。

每个形状有13种scene：

- 单任务C、T40、N40（N40是日志中的N96r64别名）。
- 双任务CC、CT40、TT40、CN40、NN40，各测serial和parallel。
- C固定使用生产CTA512；输入threads只作用于Tensor候选。

两工作流由控制流事件fork/join；serial两任务在同一流，parallel分属两流。初始根表及上传显式同步，再由两任务只读；A/B数组各自私有。每样本交错改变scene顺序，没有丢弃warmup。`iterations` 次运算串联，最终全部A/B与相同串联次数的生产CUDA结果比较。

主时间为整个fork/join范围的GPU makespan，包含提交供给的空隙；分配、填充、根生成、CUDA参考、Graph捕获与验证在计时外。填充流已同步，host_seconds也不再包含填充尾部。Conv/s=jobs×batch×iterations/makespan；每iteration包含3×jobs×batch个数学NTT，不是ECM curves/s。

独立保存 `--execution native` 对照，用于识别CPU提交造成的小任务假象；已核对同一EXE门禁后可显式 `--skip-gate`。每次输出目录必须新建；输入、EXE/GMP、源码闭包、原日志、CSV/JSON及500ms遥测保留。工具不调整功耗或频率。

两任务加四份参考数组共64·batch·L bytes，根包6656B、错误数8B；只读计划与表另计。单任务scene仍保留相同分配，避免改预算条件。计时内H2D/D2H=0，计时外完整GPU输出比较只回传8B错误数。驱动、context、Graph内存不包含在该请求账本中。

## 管理员剖析

`profile_ntt_tensor_concurrency.py --exe <exe> --shape <shape.json> --scene <0..12> --kind systems --output <新目录> --run-admin`。

shape包含k/batch/iterations，可选threads。常用scene为4=CC_parallel、5=CT40_serial、6=CT40_parallel、10=CN40_parallel。只测该scene一次，保留完整输出检查，ProfilerStart/Stop仅包围计算和完成等待。Systems使用Graph node trace，导出内核区间并集及CUDA/Tensor重叠；它不证明同一SM同一周期发射。

Compute使用 `--kind compute --replay app-range`，默认application-range。它按整个范围采集，保留fork/join并发关系；不使用逐kernel重放证明并发。显式none时钟/缓存控制。各次应用重放输出追加保留，所有输出检查须通过；核对报告GPU名称与SM数。`--collect-only --output <目录>` 只读导出既有报告。

本机NCU2026.2.1将 `--devices 1` 加到范围模式时曾得到空报告；当前由目标程序选择GPU1，工具按名称和SM数核对物理设备。`--graph-profiling graph` 与 `--profile-from-start off` 不与range模式合用；range默认由程序的ProfilerStart/Stop定义。原始失败记录保留，空报告不进入分析。参考[NVIDIA范围重放说明](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#range-replay)。

## 汇总

`analyze_ntt_tensor_concurrency.py --experiment <graph目录> --experiment <native目录> [--profile <已核验目录>] --output <新汇总目录> --figure-prefix <图片前缀>`。

输出完整CSV/JSON、混合对最优纯CUDA的时间变化、N96r64对Tensor40的局部变化、串行/并发效率及Systems时间线。图中的数值为中位数；MAD保存在CSV。正的时间节约表示更快，serial/parallel>1仅说明并发超过相同混合串行，不能证明超过纯CUDA。

所有产物写入忽略的data/experiments、data/benchmarks及data/figures；候选没有接入生产或tune选型。
