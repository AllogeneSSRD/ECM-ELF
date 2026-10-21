# 短 NTT Graph 提交与完整 Stage2 性能

## 当前结论

**已实现可复用的短 NTT Graph 缓存，并完成固定 D 的完整曲线对照；当前证据支持继续默认关闭。** Graph 减少主机提交，没有减少蝶形、归约或传输。小样本局部收益没有扩展为生产规模的稳定收益。

- 9 组、每侧 4 次的矩阵，中位数之和为 37.0860 → 36.9663 秒，时间减少约 0.32%；其中两组没有真正执行 Graph，不能将其差异归因于 Graph。
- 扩大准入的 3 组、每侧 4 次矩阵，总中位耗时 10.606886 → 10.606849 秒，基本相同；7995-bit 普通／承载均已复用 Graph。
- 生产规模 D=1711710、P=155520、B2=2.6×10¹²，每侧 3 次，19.613472 → 19.613286 秒，基本相同。Graph 复用 2742 次、节省 10314 次理论主机提交，没有形成完整曲线收益。

因此不自动启用、不修改 D/P 模型，也不以独立 NTT 的 Conv/s 决定部署。Tensor 候选的独立结果见 [Tensor NTT](NTT_TENSOR.md)；本页只评价生产 CUDA 内核的 Graph 提交。

## 条件、版本与计时

2026-10-15，GPU1 RTX4060 Laptop，24 SM、8 GiB，UUID `GPU-8a67b1f8-ef1c-3177-a822-813a7ac2224d`，驱动610.88。保持用户默认55W与1800MHz上限，没有调整显卡设置。GPU0的4070Ti仍运行原有负载；主机与驱动竞争没有隔离。

102 次正式完整曲线的500ms遥测中，利用率≥50%的912个样本，SM频率1545–1800MHz、中位1800MHz；有效功耗13.20–54.58W、中位38.585W；温度56–70°C。1800MHz是上限，并非每时刻固定频率。保留一个超过150W的异常功耗读数，汇总时排除，不据此计算能效。

所有开关对照使用同一个sm89程序：CUDA13.3.73、MSVC14.51.36231、GMP Zen3、PTX3/canonical subtraction、legacy默认流。EXE SHA256为 `d47873056543eeb6f396101e84684f39b2928c27ae27f73da726118345ecce27`。二进制、DLL、编译清单及源码闭包冻结在 `data/experiments/stage2_short_graph_20261015/build/`。

输入复用已验证的Stage1 save，B1=20、sigma=26；完整M521/M2203/M4423及N=M8011/80111（7995 bits），承载项使用M8011。较小B1用于控制准备时间，本次不验证生产B1下的点生成模型。每项锁定save、B2、D、batch、arena、fold及承载方式，关闭短校准和短NTT batch tune。

主时间为 `stage2_full_wall.total=init+main`：包含强制自检、F树、inverse、巨步/G树/fold、下降、GCD与检查排空，排除Stage1、进程启动和前置选形。Graph建立成本计入。进程墙钟另记；不与引擎时间混合。每轮off/on与on/off交替，没有丢弃的warmup，统计中位数及MAD，MAD不是置信区间。

Nsight采集与额外叶子检查均另行执行，不进入这102次计时。

## 实现与准入

每个缓存项持有两个可执行Graph：前向Graph顺序执行A、B两个forward；逆向Graph执行融合pointwise与1/L的inverse。使用私有nonblocking流捕获，重放仍在原默认流。forward/inverse的现有计时事件仍在两个Graph之间，pack、carry、S4及检查保持原路径。

准入要求数组位于稳定arena workspace、根表已缓存且由arena持有；dump/trace、临时数组和导出路径不使用此缓存。身份包含device、L、batch、tile/outer计划、算术选项、根与scale、A/B地址及全部根表地址。工作区增长或释放、表驱逐、arena释放前使Graph失效，避免复用失效地址。

环境变量只提供实验控制，不是自动tune策略：

| 控制 | 默认值 | 含义 |
| --- | ---: | --- |
| NTT_SHORT_GRAPH | 0 | 显式开启Graph缓存 |
| NTT_SHORT_GRAPH_MAX_K | 19 | 单条L≤2ᵏ |
| NTT_SHORT_GRAPH_MAX_WORDS | 4194304 | 单次batch×L上限 |
| NTT_SHORT_GRAPH_MIN_CALLS | 4 | 同身份调用到此次数后建立 |
| NTT_SHORT_GRAPH_ENTRIES | 64 | LRU条目数，钳制1–256 |

建立前核对实时free≥64MiB。可恢复的分配／捕获不支持错误关闭该arena的Graph并回退原路径，算术与其他CUDA错误不吞掉。当前测量failures=0，尚不能宣称覆盖所有驱动失败条件。

设单方向需要p趟kernel，每次卷积原本提交3p个kernel，两个Graph改为2次GraphLaunch，因此热调用理论减少3p−2次提交。`saved_launches`累计这一数量；首次捕获仍会产生3p次记录调用与实例化成本，不能当作净API节约。数学变换数与GPU执行的kernel数均保持不变。

Graph不新增完整L级数组，不新增数据H2D/D2H；缓存主机元数据为O(条目数)，最多两个执行Graph/项。驱动Graph对象的内部内存不在NTT payload账本中。相同输入的 `full_peak_bytes` 一致，只说明该模块计账的数组峰相同，不能证明进程或驱动峰完全相同，更不能与fold等组件峰相加。

## 固定 D 完整曲线结果

arena上限6300MiB；owner默认640MiB，host项为0。batch列为MiB配置，不是NTT条数。普通默认准入为L≤2¹⁹、batch×L≤2²²、同身份4次建立。时间单位秒，正的节约百分比表示更快。

| 任务 | B2 | D | batch | off中位数 | on中位数 | 节约 | Graph复用 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| M521 | 2.6×10⁸ | 30030 | 64 | 0.145678 | 0.140080 | 3.84% | 11 |
| M2203 | 2.6×10⁹ | 60060 | 64 | 0.597093 | 0.585996 | 1.86% | 90 |
| M4423 | 2.6×10⁹ | 60060 | 64 | 1.400105 | 1.383397 | 1.19% | 196 |
| M4423 | 2.6×10⁹ | 60060 | 256 | 1.380146 | 1.385876 | −0.42% | 33 |
| N7995 | 2.6×10⁹ | 60060 | 256 | 5.432925 | 5.418908 | 0.26% | 0 |
| M4423，host fold | 2.6×10⁹ | 60060 | 256 | 1.589525 | 1.568785 | 1.30% | 33 |
| M4423 | 2.6×10¹⁰ | 60060 | 256 | 6.732672 | 6.725948 | 0.10% | 713 |
| N7995 | 2.6×10¹⁰ | 120120 | 256 | 16.034854 | 16.016203 | 0.12% | 15 |
| N7995，M8011承载 | 2.6×10⁹ | 60060 | 256 | 3.773032 | 3.741107 | 0.85% | 0 |

普通N7995及M8011承载的两项小B2任务没有命中Graph，是准入上限与重复次数的控制组。其差异也说明不能把约1%的任意正点直接认作Graph收益。4423/256项首轮略慢，扩大准入复测后略快，符号不稳定，未发现需要修改正确实现的明显回退。

扩大准入到batch×L≤2²⁶、同身份2次建立，其他输入不变，仍每侧4次：

| 任务 | off中位数 | on中位数 | 节约 | Graph复用 |
| --- | ---: | ---: | ---: | ---: |
| M4423，batch256 | 1.419559 | 1.400421 | 1.35% | 59 |
| N7995，普通 | 5.432971 | 5.444071 | −0.20% | 87 |
| N7995，M8011承载 | 3.754355 | 3.762356 | −0.21% | 87 |

扩大准入解决了“没有执行Graph”，没有改善整体性能。首轮与复测属于独立测量组，不能交叉挑选最快的off/on组成结果。

### 生产规模 P

M2203、B2=2.6×10¹²、D=1711710、P=155520、1518950个giant点、10棵G树、9次fold，batch64，默认Graph准入，每侧3次：

- 完整时间：off 19.613472±0.038060s；on 19.613286±0.014240s，约0.00%节约。
- Graph：39个缓存项建立、2742次复用、5562次GraphLaunch、10314次理论提交节约、setup约0.004s，failures=0。
- NTT组件 `full_peak_bytes` 两侧均3345419832B，约3190.44MiB；最大数学长度L=2²⁷。该数字不是进程峰。
- off组件中位数：F准备2.989s、inverse1.657s、giant0.512s、G树6.709s、fold4.277s、下降2.902s、累积0.057s。S4归约0.831s已嵌套在多项式阶段内，不再次相加。

![固定D完整曲线对照](../../data/figures/stage2_short_graph_20261015_final.png)

![各阶段占比](../../data/figures/stage2_short_graph_20261015_final_phases.png)

阶段堆积图以各组件独立中位数除完整中位数；四舍五入、检查与未列入准备开销会留下残差，不是精确的同一样本时间分割。

## Systems：提交减少与剩余空隙

管理员Nsight Systems2026.1.3，Graph node trace；每个路径单独采一条完整M4423曲线，B2=2.6×10⁹、D=60060。窗口为本进程在GPU1的首个至末个kernel/copy/memset事件，包含自检和尾部，不等于引擎计时边界。

64MiB batch的off/on：

| 指标 | off | on |
| --- | ---: | ---: |
| GPU执行kernel总数 | 5103 | 5103 |
| cudaLaunchKernel API数 | 5103 | 3960 |
| cudaGraphLaunch API数 | 0 | 418 |
| 两类launch API合计 | 5103 | 4378 |
| GPU事件并集 | 1.107823s | 1.109816s |
| 无本进程GPU事件空隙 | 0.377872s | 0.309933s |
| 窗口内空隙占比 | 25.43% | 21.83% |

净少725次launch API=809次理论节约−84个捕获节点。GPU工作时长基本不变。较少插桩调用也可能放大采集中的Graph收益，不能将此窗口缩短直接替换正式A/B的1.19%。256MiB batch的off/on空隙为20.26%/20.02%，窗口1.410432/1.410525s，基本不变。

64MiB off的H2D为146490323B/360次，D2H为81929976B/591次，设备copy分别约11.41/6.94ms；Graph两侧字节和次数相同。同步cudaMemcpy API累计0.674s远大于实际copy时间，因为它包含前序工作等待；不能把0.674s全部归因为PCIe传输。

该窗口还有339次cudaMalloc、371次cudaFree、58次pinned分配和52次pinned释放；三类操作中的多项会同步或等待。仅凭次数不能声称全部可删除，但比继续只压缩两段NTT提交更值得定位准备与等待边界。事件创建/销毁共约5312次，API累计约2.4ms，不能将数量大直接等同主要瓶颈。

无本进程事件不等于整卡空闲；时间线可能受其他进程、CPU准备、驱动、分配与同步影响。API累计时间和GPU并集重叠，不相加计算总时间。

## batch 与实际吞吐

真实曲线中的短NTT已经是大batch。对L=2048的forward tile，Systems同组kernel总时间/次数给出的均值：

| batch条数 | 来源 | off单次kernel | on单次kernel | off条数/秒 |
| --- | --- | ---: | ---: | ---: |
| 720 | 64MiB batch | 137.393µs | 137.489µs | 5.240×10⁶ |
| 1440 | 256MiB batch中的半批 | 281.954µs | 282.098µs | 5.107×10⁶ |
| 2880 | 256MiB batch中的整批 | 557.432µs | 557.301µs | 5.167×10⁶ |

这里的条数/秒只描述一个forward tile kernel，排除提交空隙、另一个forward、inverse、pack/carry/S4；不是Conv/s，也不能与PRPLL迭代率比较。batch增加4倍时kernel时间约增加4.06倍，吞吐没有继续增长。720个CTA在24SM上已提供足够工作；孤立小batch中“加batch而时间不变”的区间，不适用于此实际任务。

![实际短NTT吞吐与本进程GPU事件窗口](../../data/figures/stage2_short_graph_20261015_final_systems.png)

## Compute：内核内的限制

管理员Nsight Compute2026.2.1，分别捕获Systems确定的实际kernel；逐项核对名称、调用编号、grid/block、shared与registers，完整曲线结束后再次核对结果。每项16次左右重放，`clock-control=none`、`cache-control=none`，不调节设备。

| 实际形状 | block / 动态shared / registers | occupancy | SM吞吐 | DRAM吞吐 |
| --- | --- | ---: | ---: | ---: |
| forward L2048×720，off | 512 / 16KiB / 40 | 95.60% | 80.03% | 1.12% |
| 同形状，Graph | 512 / 16KiB / 40 | 95.65% | 80.41% | 1.22% |
| forward L2048×2880，off | 512 / 16KiB / 40 | 97.71% | 76.20% | 76.36% |
| 同形状，Graph | 512 / 16KiB / 40 | 97.70% | 76.19% | 76.45% |
| forward L=2²³×1，off | 512 / 32KiB / 40 | 97.99% | 76.43% | 74.97% |
| S4归约，M4423折叠 | 128 / 0 / 40 | 17.75% | 6.35% | 37.29% |

SM与DRAM百分比是各自峰值口径，不能相加。短tile的math-pipe-throttle/issue-active约4.96，off/on近似；长tile约4.64，另有long-scoreboard约3.97；所选S4约31.43。这些ratio不是占墙钟的百分比。S4条目采自精确M4423的折叠特化，源码模板MERSENNE=true；通用余因子的硬件归约计数器需要另行采集。短tile有充足occupancy且执行管线压力较高，长tile同时具有明显访存压力，Graph不会改变两者的内核工作。

短tile重放测得时钟分别约510/360MHz，长tile与S4约1795/1798MHz。重放、输入恢复和间隔会改变时钟与缓存；短tile的478/673µs不用于宣称Graph变慢，普通负载中的137µs也不与之直接拼接。缓存命中与吞吐结论仅限此次重放条件，不把低DRAM读数推广为所有NTT都不受访存限制。

batch2880的off/on重放时钟约1795/1799MHz，时间约568/565µs、long-scoreboard ratio约3.92/3.92，Graph并未改变资源或吞吐。该卡L2为32MiB；L2048×720的单个A数组11.25MiB，×2880时为45MiB。扩大batch增加缓存压力，与DRAM吞吐显著增加相符；这属于计数器和容量支持的解释，尚未通过独立缓存控制实验隔离因果。生产不会为了Graph刻意扩大batch。

## 正确性、证据与复现

- 独立生产pack→NTT→carry门禁：368次完整GMP整数乘积对照、14779392个输出word范围检查，bad=0；另有workspace、FuseCtx生命周期及容量检查通过。故意翻转一个digit时恰好bad=1、退出非零。
- 正式102次曲线共189312个强制自检case、549678个GMP检查窗口，bad均为0；相同任务的因素结果、rootfold指纹、NTT直方图及S4分块数一致。
- 额外4条诊断曲线分别核对4423/7995-bit的全部下降叶子投影指纹，off/on一致；这不是逐叶GMP独立数学证明，也不计入正式性能。
- 原始证据：`data/experiments/stage2_short_graph_20261015/`，含`ab`、`broad_ab`、`production_ab`、`diagnostic`、`nsys_*`、`ncu_*`、门禁与故障注入日志、冻结构建和命令。
- 汇总：`data/benchmarks/stage2_short_graph_20261015_report/{summary.json,summary.csv}`，JSON包含4份Systems与6份Compute的核验摘要；图片在`data/figures/`。全部原始样本保留，不用复测覆盖首轮。
- 使用与复现：[Graph验证工具](../../tools/bench/README_STAGE2_GRAPH.md)。实现入口：[缓存身份及生命周期](../../src/ecm/backend/cuda/ntt/graph.cpp)、[生产调用](../../src/ecm/backend/cuda/ntt/runtime.cpp)、[独立门禁](../../tools/test/ntt_graph_probe.cu#L20)。

## 后续优化条件

1. 先定位正常完整曲线中的准备/同步、pinned及临时分配边界；仅在确认没有活跃借用和异步检查读者后复用缓冲。不能按CUDA API累计等待时间直接估算收益。
2. 若继续Graph，评价稳定的一整个请求，包括pack/carry/S4及其依赖；目前两个NTT Graph仍留有事件、归约、检查与host准备边界。设计须保留参数身份、常量更新、尾批、导出与检查排空。
3. 长NTT结合tile的数学执行压力、DRAM流量与outer组织优化；较低提交开销不能代替内核改进。
4. 任何默认改变仍要求同二进制固定D完整矩阵持平或更优、明显回退独立复测，并扩展生产B1、显存边界与其他GPU。缓存上限驱逐、分配失败和不同驱动的Graph回退仍需扩大验证。
