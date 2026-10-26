# Stage2 主机等待：CPU、吞吐与验证

## 当前策略

默认采用按提交顺序回收的 S4 计时 FIFO：最旧项未完成就停止查询后续项，满环只等最旧一槽，收尾完整排空。默认仍使用 CUDA 自动调度与原生事件等待。下述性能数据来自0.52.4测试二进制，身份和计时边界见下一节。

可选配置 `stage2_low_cpu_wait=true` 使用 **CUDA 设备阻塞调度4 + 事件退避等待**；同步读回保留原生路径。CLI 为 `--low-cpu-wait 0|1`，默认 false。配置/Worker/CLI 优先级、专家开关、资源与错误合同以 [Stage2架构](../architecture/STAGE2.md#主机等待与提交回收) 和 [配置参考](../ECM_TOML_REFERENCE.md#stage2_low_cpu_wait) 为准。

`NTT_HOST_READBACK_WAIT=1` 是另一个实验控制：在既有阻塞读回前记录并等待完成事件，然后执行原拷贝。实测它能减少 CPU 消耗，但唤醒代价较大，用户配置开关不启用这一项。

## 条件、身份与计时

日期2026-10-25；GPU1 RTX4060 Laptop、24 SM、8188 MiB，UUID `8a67b1f8ef1c3177a822813a7ac2224d`；驱动610.88、CUDA13.3.73、MSVC14.51.36231、本机GMP。用户已固定SM1500MHz；本轮没有修改显卡设置。最终六位宽矩阵及策略对照中，利用率≥80%的936个GPU样本均为1500MHz。GPU0有外部任务；功耗读数异常，不用于推导功率或能效。

- 计时二进制SHA256：`6a3e9d9ed50389b4235d20a52e37f61a18e080419a3c3db084c7397de7631713`。
- 配置接线完成后的本机二进制SHA256：`10f550d8e789684ca3af9633a4a71619050ebeb36c4c0dd15a7b6cf853ed4c33`。
- 两者的 S4、engine、NTT runtime、host_wait 以及两个CUDA编译单元的对象哈希相同；后者另做真实父/子进程配置与曲线验证。CUDA对象与0.52.3基线也相同，修改位于主机等待和编排。
- 原始证据：`data/experiments/stage2_host_wait_20261025/`，冻结二进制及构建身份分别在 `head_bin/`、`ship_bin/`。数据目录由Git忽略，分发报告时须另外分发原始证据。

引擎时间采用 `stage2_full_wall.total=init+main`，包含原有初始化和必需检查，排除Stage1、进程启动及计时外规划。CPU采用该worker完整生命周期的 `GetProcessTimes` 用户态+内核态时间，除以对应生命周期墙钟，得到平均逻辑核当量；它不是整机CPU百分比，也不只覆盖引擎total。进程总墙钟另存于JSON/CSV。

使用直接曲线worker、固定save/D/B2，关闭探索；每轮反转策略次序，所有样本保留，不丢弃预热。内存预算为batch64、arena4096、fold640 MiB，Graph默认关闭；其他预算和Graph对照单列。所有位宽均为实际余因子位数。

## 默认 FIFO 的完整曲线对照

以下6格，每策略2次，共36条曲线。B1=50000，Edwards有效保存点，sigma=26；M2003使用27。`legacy`恢复满环全排空，`fifo`采用当前默认；同一二进制其余参数相同。

| 原指数 / 实际bits | B2 | D | legacy中位数/s | FIFO中位数/s |
|---|---:|---:|---:|---:|
| M503 / 318 | 260000000 | 30030 | 0.204102 | 0.207797 |
| M2003 / 1939 | 260000000 | 30030 | 0.616913 | 0.617259 |
| M5003 / 4667 | 26000000000 | 120120 | 6.967526 | 6.983728 |
| M8011 / 7995 | 26000000000 | 120120 | 21.333230 | 21.376478 |
| M12511 / 12285 | 260000000 | 30030 | 10.717374 | 10.728840 |
| M16319 / 16272 | 260000000 | 30030 | 19.785727 | 19.735060 |

各格中位数之和59.624870→59.649160秒，+0.041%。这是矩阵指标，不能解释为某条生产曲线的耗时。两次重复不足以分辨这种小差额；本轮没有测出明确的完整曲线加速，也没有明显回退。逐格的范围与CPU计数保留在汇总中。

默认策略也没有测出明确的 CPU 降幅。六格中每策略共12次运行，累计 CPU 时间除以累计进程墙钟，legacy为0.9756核，FIFO为0.9717核，表面下降约0.4%，不足以区分测量波动；累计 CPU 时间实际为123.359375→123.484375秒，并未减少。默认状态仍约占用一个逻辑核。这里的legacy是同一二进制恢复原等待策略的控制组，不是未修改的旧版本二进制；下节76.6%～86.1%的CPU降幅仅适用于显式开启低CPU配置。

另用真实生产save：M5351的4916-bit余因子、sigma2035538945928400、B1=260000000、B2=260000000000、D510510、P46080；同一二进制交替3次/侧。legacy中位数20.915641秒，FIFO20.901611秒，差−0.067%，同样接近持平。计时环等待墙钟约3.2→2.0秒，与GPU工作重叠，不能据此声称完整曲线节省1.2秒。满环次数在只回收一槽后可能上升，不能单独用次数判定性能。

证据为 `final_long/`、`final_wide/`、`production_head/`；汇总为 `data/benchmarks/stage2_host_wait_final_20261025/`。

## 可选低 CPU 策略

独立对照使用M5003/M8011的上述保存点，B1=50000、B2=26000000000、D120120，预算同上；每策略2次，共12条曲线。

| 实际bits | 策略 | 引擎中位数/s | 平均CPU核当量 |
|---:|---|---:|---:|
| 4667 | 默认FIFO | 7.009926 | 0.9722 |
| 4667 | 仅设备阻塞调度4 | 7.086009 | 0.2244 |
| 4667 | 低CPU配置：阻塞4 + 事件退避 | 7.104047 | 0.2278 |
| 7995 | 默认FIFO | 21.399206 | 0.9687 |
| 7995 | 仅设备阻塞调度4 | 21.495900 | 0.1723 |
| 7995 | 低CPU配置：阻塞4 + 事件退避 | 21.584039 | 0.1346 |

配置开启后，CPU核当量分别减少76.6%/86.1%，引擎耗时增加1.34%/0.86%。完整进程墙钟分别为7.649849→7.716583秒、22.218589→22.238478秒；进程计时包含计时外活动，小差额应结合原始样本范围阅读。

![低CPU配置的CPU与完整耗时取舍](../../data/figures/stage2_host_wait_policy_20261025.png)

上图CPU覆盖完整进程，引擎耗时采用init+main；两个量的时间边界不同。该模式适合愿意用少量GPU吞吐换取CPU余量的场景。其他设备/驱动的验证尚不完整，因此保持默认关闭。不能由这两档推导任意位宽、B2或GPU上的固定收益。

最终本机版另做每策略一次的宽位功能对照：318/1939/12285/16272 bits均通过；12285/16272-bit样本的CPU约为0.113/0.084核。生产4916-bit保存点的单次对照为20.907850→21.130163秒，CPU0.987→0.181核。这些是功能/取舍补充，不加入上表的两次中位数，也不作为稳定加速证明；证据分别在`policy_wide_smoke/`与`production_policy_ship/`。

证据：`hybrid_check/`；汇总及绘图数据：`data/benchmarks/stage2_host_wait_policy_20261025/`。额外的 `native_blocking/` 为独立交叉检查，不混入本表。

### 读回前休眠实验

同一最终执行对象的6格矩阵另测 `NTT_HOST_EVENT_WAIT=1`、`NTT_HOST_READBACK_WAIT=1`、设备调度0。4667/7995-bit的CPU约降至0.454/0.445核，但引擎约慢4.87%/1.35%；318-bit短输入约慢34%。原生阻塞调度在这台机器的取舍更好。

多数强制读回边界后没有足够独立在途工作遮蔽唤醒延迟。减少CPU忙等不直接等于减少GPU空隙；单独改等待原语也不能消除真实GMP和主机准备工作。

## 正确性与覆盖

- 六格对照逐次核对目标N、B1/B2、sigma、叶子hash、因子、oracle选择signature及selected=compared，GMP自检/抽查均无错误。
- 汇总器另外核对相同格的S4 launches、poly_muls、coeffs_reduced、必需自检数、GMP检查数及full_checks；NTT模块的full_peak_bytes逐格相同。这不代表测得了全进程显存峰。
- CPU夹具运行实际S4主机控制方法，覆盖满环最旧项、槽索引环绕、已完成回收、全排空对照、部分创建后的清理、CUDA错误、最终/重复排空；可注入时钟的等待策略覆盖退避封顶、查询失败及定时器失败回退。
- 配置检查使用真实父/子进程，核对默认、TOML、Worker、CLI、专家环境优先级及实际CUDA调度flags；非法值拒绝，oracle待校验快照篡改触发失败，未发布成功结果。故障发生在排空入口，不把它当作完成整条曲线的性能样本。
- 另覆盖真实梅森承载、驻留/预算回退、短NTT Graph启动及宽位功能。回退预算为arena1/fold0 MiB；驻留Graph输入使用M2003的1939-bit余因子承载到2003 bits、D30030。最终本机版的驻留对照两种等待策略均实际启动710次Graph，builds=13、failures=0；arena1/fold0的回退对照跳过Graph，fold回退原因为budget。证据在`graph_ship_resident/`与`graph_ship_fallback/`。不能把requested=1当作实际启动证明。

尚未覆盖：其他设备/驱动、pinned申请失败、所有取消边界和长期稳定性。先期诊断的一次legacy侧CUDA700非法访问未确认原因；原始失败保留，冻结0.52.3相同7995-bit输入独立3次及后续串行矩阵未复现，不能据此宣称该问题已经修复。早期重叠样本、工具转换限制导致的中断及重跑记录见原始目录的 `exclusions.json`；它们不进入上述性能表。

## 源码与复现入口

- [S4 FIFO回收，s4.cpp:41](../../src/ecm/stage2/cuda/s4.cpp#L41)：`dt_reap`、`dt_oldest`、`dt_slot`、`dt_flush`。
- [退避及错误处理，host_wait_policy.h:13](../../src/ecm/stage2/cuda/host_wait_policy.h#L13)：`complete_event`。
- [事件及读回前等待，host_wait.h:37](../../src/ecm/stage2/cuda/host_wait.h#L37)：`event_sync`、`before_readback`；[平台定时器](../../src/ecm/stage2/cuda/host_wait.cpp)。
- [配置优先级，host_policy.cpp:9](../../src/ecm/stage2/cuda/host_policy.cpp#L9)：`configure_host_wait`；[NTT回调接口，runtime.h:37](../../src/ecm/cuda/ntt/runtime.h#L37)。
- [基准工具](../../tools/bench/bench_stage2_event_wait.py)：输入既有matrix plan、二进制、指数列表、固定D/B2及新输出目录；`fifo`为默认控制，`hybrid`对应当前低CPU配置，`readback`为实验控制。工具记录自身副本、二进制/save哈希、完整CPU计数及GPU采样；同卡运行用命名互斥串行化。
- [汇总/绘图工具](../../tools/bench/report_stage2_event_wait.py)：输入完整measurements.json，以 `--compare hybrid` 绘制用户模式；先复核日志哈希与工作量。输出目录应为空。
- [CPU回归](../../tools/test/test_stage2_timing_ring.py)、[真实配置验证](../../tools/test/check_stage2_host_wait_config.py)、[失败不发布验证](../../tools/test/check_stage2_host_wait.py)。CPU编译使用正式构建的MSVC身份，产物写入新实验目录。

事件语义依据 [NVIDIA CUDA事件管理](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__EVENT.html)；高精度定时器依据 [Microsoft CreateWaitableTimerExW](https://learn.microsoft.com/en-us/windows/win32/api/synchapi/nf-synchapi-createwaitabletimerexw)，均于2026-10-25核对。GPUOWL的有序队列及两窗口思路见 [固定参考资料](../reference/GPUOWL_OPENCL_QUEUE.md)。

## TODO

下一步优先研究carry诊断读回的有界延后窗口，并证明计数器清零/复用前的快照、工作区寿命、错误传播和发布前排空。按GPU时间组织两个提交窗口需要独立成本与资源证明；当前尚未接入。后台GMP和多stream也不包含在本次默认策略中。研究过程保留在 [开发草稿](../development/STAGE2_HOST_WAIT_DRAFT.md)。
