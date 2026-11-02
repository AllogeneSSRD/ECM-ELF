# 普通高位宽 S4 归约性能

当前CUDA提供`stage2_reduce_backend=scalar|coop32`，默认scalar。coop32对8193…16384实际算术bits的普通系数归约使用一个warp处理一个系数，余数放在共享内存；小位宽和精确梅森归约保留原内核。CLI为`--reduce-backend`，worker覆盖和环境回退见[配置参考](../ECM_TOML_REFERENCE.md#stage2_reduce_backend)。它与seed／baby／giant的auto独立，当前不自动搜索S4后端。

RTX4060 Laptop固定D的小档B2四个普通输入，完整引擎合计113.389→94.091s（减少17.02%），完整曲线墙钟119.029→98.686s（减少17.09%）。最高位S4设备事件累计15.681→7.939s，约减半；不是整条Stage2快一倍。保持默认scalar，跨设备、大B2及更多D尚需独立资格。

## 输入、设备和计时

2026-11-02，GPU1 NVIDIA GeForce RTX4060 Laptop，SM89／CUDA13.3／驱动610.88，Goldilocks fixed3/add_sub1；GPU Ceiling Power Limit的Current／Requested／Default均为55W。未修改时钟、功耗或缓存控制。主矩阵1466条利用率≥50%的NVML样本全部为1500MHz，忙碌样本功耗9.090…47.808W；低负载和启动时可降至210MHz。未采集CPU有效频率，本轮未更改CPU控制。GPU0有外部任务，不参与本轮；GPU1逐次门禁只放行已授权并绑定路径的Codex界面PID16148。

四个Stage1输入均为Edwards、sigma26、B1=50000的已验证余因子保存点。每格请求／实际B2均为27015978990，layout1、chain32、baby32／giant32／seed32；batch／arena／fold预算为256／6300／640MiB。关闭学习、短校准、短NTT批调优、relocation、B2凑整和尾块平衡。两个后端的NTT分布、S4次数／系数数、D／P及NF结果逐格相同。

首轮每格每后端n=1、按格交替顺序，无warmup。承载最低两档出现约6%的外层波动，再各补两次反序复测，最终这两格用全部三次中位数；其余n=1。原始首轮和复测全部保留，不挑最小值。图中横轴为目标余因子bits，承载实际算术分别为p=10267／12511／14401／16319，不能把相邻点连线当成固定几何的位宽复杂度证明。

完整引擎为`stage2_full_wall.total`，含baby／F-tree及强制S4自检、主循环／GCD和oracle排空，排除启动点检查、读档／选型及发布。完整曲线为`curve_done.wall`，包含上述外层成本；两者分别汇总。`t_reduce`是S4设备事件范围，嵌套于树／fold／下降，不能再与这些阶段相加。诊断调用和NCU重放时间均不进入性能表。

## 完整Stage2结果

| 目标bits | W | D／P | S4事件 scalar→warp，s | 引擎 scalar→warp，s | 引擎减少 | 完整曲线 scalar→warp，s |
|---:|---:|---|---:|---:|---:|---:|
| 10232 | 160 | 240240／23040 | 4.460→3.281 | 16.413→13.993 | 14.74% | 17.492→14.896 |
| 12285 | 192 | 240240／23040 | 8.189→4.790 | 24.378→20.786 | 14.74% | 25.418→22.066 |
| 14328 | 224 | 210210／20160 | 10.906→6.241 | 30.948→24.776 | 19.94% | 32.671→25.861 |
| 16272 | 255 | 210210／20160 | 15.681→7.939 | 41.650→34.536 | 17.08% | 43.448→35.864 |

承载是回退控制：请求coop32时`enabled=0`，S4仍执行原来的梅森折叠。外层差异不归因于协作归约。

| 目标bits／算术p | 每后端n | 引擎 scalar请求→coop32请求，s | S4事件，s |
|---|---:|---:|---:|
| 10232／10267 | 3 | 10.652→10.749 | 0.505→0.505 |
| 12285／12511 | 3 | 14.872→14.785 | 0.630→0.630 |
| 14328／14401 | 1 | 17.816→17.870 | 0.789→0.790 |
| 16272／16319 | 1 | 21.660→21.488 | 0.907→0.908 |

八格按上述n和中位数口径的引擎合计178.389→158.982s（减少10.88%），完整曲线188.492→168.043s（减少10.85%）。首轮八格原始合计180.310→159.277s另保留，不覆盖或冒充复测后汇总。

![固定D的S4完整曲线对照](attachments/stage2_s4_coop_times.png)

最高位普通路线的G树／fold／下降由12.168／6.150／7.486s降到9.180／3.955／4.644s；合计占引擎61.95%→51.48%。S4嵌套占比37.65%→22.99%，点生成及其他阶段的占比随之上升。F树准备、多项式求逆等范围仍包含CPU工作、检查及等待，不能把其变化全部解释为纯NTT或S4设备时间。

![最高位互斥阶段分组](attachments/stage2_s4_coop_phases.png)

## NCU：显式local工作集和依赖链

管理员Nsight Compute2026.2.1分别采0.57.11标量和0.57.13协作路径，均固定最高位普通输入、D210210/P20160、相同逻辑窗口，seed scalar、baby／giant32。以NW256普通kernel过滤，跳过200次后取3次真实调用；两次完整NF、NTT／归约工作及GMP覆盖一致。clock/cache control均为none；重放显著拖慢应用，不作为普通计时成绩。

| 对应调用 | grid scalar→warp | kernel时间，ms | issue，% | eligible warps | long scoreboard／issue |
|---:|---|---:|---:|---:|---:|
| 1 | (81, 1, 1)→(2561, 1, 1) | 35.215→25.292 | 22.77→67.71 | 0.266→1.306 | 10.798→0.057 |
| 2 | (161, 1, 1)→(5041, 1, 1) | 159.308→49.468 | 9.21→68.08 | 0.115→1.318 | 44.857→0.056 |
| 3 | (30, 1, 1)→(945, 1, 1) | 13.093→9.611 | 20.40→66.50 | 0.216→1.271 | 3.282→0.050 |

block均为128线程。旧grid按源系数分配线程，新grid按有效输出分配warp，另外仍检查所有被省略源槽；grid倍数不是工作量倍数。最大调用的local读／写为665968490／667472271 sectors，协作后均为0；两侧寄存器均40/thread、寄存器spilling指令均0。显式可寻址数组也产生local流量，不能把这些访问全部称作spill。

long scoreboard表示等待L1TEX相关操作，不等于已经证明DRAM带宽耗尽；本表是每issue的ratio，不是等待百分比，sectors也不是DRAM字节量。指标定义见[NVIDIA Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)。旧最大调用L1／L2命中率36.90%／73.24%，新路径87.32%／17.52%；请求集合已从大量local重读改成输入流读，不能因新L2命中率低而判定性能回退。

最大调用活跃warp占比52.78%→40.79%，但发射率显著提高，说明不能只追求occupancy。协作的short scoreboard／issue约1.175，固定延迟wait／issue约2.714；当前仍有共享存储和指令依赖成本，不能从stall名称直接断言shared带宽饱和。

## 缓存窗口对照与复杂度

独立探针使用同一真实16272-bit N、W255、26-bit digits、单batch、0／全1／交替／高位／随机混合模式。源系数两端省略，全部输出及哨兵由GMP逐词核对。每后端两进程、相反顺序，每进程3个CUDA事件，无warmup；下表为两个进程事件中位数的中位数。8192窗口只存在于探针，不是生产选项。

| 源／有效系数 | 原标量，ms | 标量8192窗口，ms | warp，ms |
|---:|---:|---:|---:|
| 8192／8190 | 25.703 | 25.734 | 18.985 |
| 20608／20606 | 148.548 | 70.837 | 46.907 |
| 32768／32766 | 245.034 | 103.757 | 74.271 |

同样的乘减工作，仅限制标量活跃系数窗口，就能大幅改善大批量耗时；因此当前瓶颈不能概括为“只有减少MAC才有效”。warp在这三档均更快，吞吐随批量增长接近线性；这不等于任意输入、所有NTT长度或完整Stage2都有同倍率收益。

![精确S4原语批量与吞吐](attachments/stage2_s4_probe_16272.png)

设归一化被除数为U词、除数为W词，商位数J=max(0,U−W+1)。每商位仍计算W个完整64×64乘积，普通S4工作为O(JW)，常见J≈W时仍为O(W²)。两遍低／高乘积借位前缀缩短整数内部依赖，未减少NTT、商位数或乘积数；商估计、最多两次加回、反归一化均保留。推导、填充词传播及共享内存公式见[数学资料](../reference/ECM_STAGE2_BIT_COMPLEXITY.md#11-精确长除法的-warp-协作组织)。

旧NW256内核的t／u可寻址私有数组静态local为6176B/thread，新内核local为0；新动态shared为8[W+(T/32)(2W+4)]B/block，T128、W255时18488B。没有新增显式设备数组或H2D／D2H，也不增加NTT arena条目；shared是SM片上资源。上述静态容量不能乘上总系数数，或与arena／fold峰值直接相加，当作进程显存节省。

## 资格、默认保护与tune

生产头和分派入口通过216次独立GMP对照：9个位宽8193／10240／10241／12288／12289／14336／14337／16319／16384，四类模数、digit1／26／64、多batch，4523400个输出／哨兵词；另16次空窗口和16次padding／输出破坏拒绝全部通过。空窗口仍检查所有源槽。最高位完整诊断的20160个目标叶、5140800词摘要一致，GMP抽查、设备叶和seed投影检查均通过；诊断模式约翻倍的墙钟不作成绩。

7995-bit普通／8011-bit承载回退控制在两种请求下均`enabled=0`，完整引擎分别6.851→6.857s、3.924→3.935s，未观察到明显开销。相邻旧版0.57.12／新版0.57.13的标量S4控制，10232-bit为15.422→15.156s、16272-bit为41.185→40.325s，NTT／S4工作相同；这是两格n=1保护，不能声明完整26格或所有设备均已验证。

真实Worker #2验证worker覆盖、CLI优先、环境回退和非法coop16拒绝；协作S4条件下另有28条NF验证seed auto的重复门槛、只读胜者、未配对保护及baby／giant／seed／32与64链长联合学习。CPU运行时、边界、后端资格以及Python聚合／生成物检查通过。

ECM格式12强制区分`env_s4_coop_reduce=0|32`，不因忽略environment合并；旧标量条件缺该键仍按0兼容。NTT表继续共用。协作宽度模型以ceil((W+1)/32)隔离额外高词组台阶，不跨W160／192／224／256直接平滑复用；无协作实测时仍是保守先验。S4默认scalar，seed默认auto；不能把固定后端A/B当成新D或新S4后端已自动选优。

## 证据、源码与后续工作

完整曲线EXE为0.57.13，SHA256 `1f14235e6e25f7003d99b1268f954e8a831272382720cab8ba155b12c31f5c6e`，19624448B；257项构建输入和46个目标文件核验一致。GMP DLL为`969f0e96ac587dedfecf611ca3442fc7211f9218f1a553214c83548dcdbcfd6e`。旧标量保护EXE0.57.12为`3f01cc3da82a87cb26fdd3112659f26534a064ed2c1e0c20e30c45c7fd0b68cc`；NCU旧0.57.11为`0137ba813e1305e57b38947b48e2e53714132e4a574a7860091b20ba0d724157`。探针EXE为`9391d6f62d3aeb07669d774732052848b00d7a8511e4dda132970be8b8f4bcb9`，使用同一生产S4头。

原始证据根`data/experiments/stage2_s4_20261102/`：`production_correct01`／`production_empty01`／`production_fault1/2`、`production_perf01`、`curves01`／`carrier_retest01`、`native_checks01`／`s4_config01`／`seed_auto_s4_01`、`low_fallback01`／`default_guard01`、`ncu_baseline_200`／`ncu_warp_200`和`s4_release`／`s4_final_audit.json`。功耗查询保留于`power_identity.txt`；初期探针输入padding错误和编译类型错误均保留，修正测试生成器／启动转换，未放宽生产guard。

汇总、CSV和图位于`data/benchmarks/stage2_s4_20261102/{curves_final,probe01,ncu01}`及同名figures目录。`curves01`原始汇总和图也保留；`curves_final`明确纳入两档承载复测。复现入口见[测量协议](../../tools/bench/README_STAGE2_VERSION.md#s4普通归约验证)；`analyze_stage2_s4_ncu.py`绑定输入、二进制、GMP覆盖、NTT工作及CSV单位，不运行GPU。

源码依据：

- [s4_warp_reduce.cuh:6](../../src/ecm/stage2/cuda/device/s4_warp_reduce.cuh#L6)借位前缀，12／31行乘减与加回，62行kernel及全源检查；[s4_reduce.cuh:88](../../src/ecm/stage2/cuda/device/s4_reduce.cuh#L88)生产分派和合法空窗口。
- [s4.cpp:195](../../src/ecm/stage2/cuda/s4.cpp#L195)实际启用／共享字节诊断；[host_policy.cpp:63](../../src/ecm/stage2/cuda/host_policy.cpp#L63)配置优先级，`process.cpp::run_curve_process`显式转发子进程。
- [tune_portable.h:339](../../src/ecm/stage2/planning/tune_portable.h#L339)硬成本身份，361行NTT复用；[portable_model.h:213](../../src/ecm/stage2/planning/portable_model.h#L213)位宽台阶，`width_prediction.h::region`隔离模型区域。

TODO：

- 补其他GPU、大B2和W129／159／160／255／256附近的完整曲线资格，再评估S4的auto／默认策略；保留当前默认保护。
- 依据新NCU比较64／96／128线程块与两遍乘减的共享读写合并，保留精确借位及商校正；不凭occupancy或stall名称直接选型。
- 给普通／承载树阶段分离纯NTT、S4及host等待，优化新暴露的NTT／准备关键路径；新S4成本下的D/P需独立校准与持出，不沿用标量收益承诺。
