# 原生 OpenCL Stage2：完整流程与 CUDA A/B

## 当前结论

0.54.0提供独立 `ecm_opencl_stage2.exe`，已接入现有生产文本save、ECMSTAGE2队列、曲线进度与结果发布流程。OpenCL和CUDA在同一正命中样例中均提取到因子257；NVIDIA与AMD各通过13项生产流程检查，AMD另通过1939/7995-bit真实Edwards save的CPU点回退与OpenCL多项式闭环。

**当前OpenCL实现明显慢于CUDA，保持显式选择入口。** 固定D的8格矩阵、每侧各3次，共48条完整NF曲线全部通过；引擎中位数逐格比值为5.35～27.42，完整父进程墙钟比值为1.19～18.10。这里的比值均为 `OpenCL/CUDA`，大于1表示OpenCL更慢，不是加速倍数。

矩阵引擎中位数之和为CUDA **7.454387 s**、OpenCL **119.649975 s**，比值16.05；完整进程中位数之和为19.032660/121.984139 s，比值6.41。它们是8格统计量之和，不代表一条生产曲线或常驻多曲线的时间。

![完整 OpenCL/CUDA 固定D对照](../../data/figures/opencl_cuda_stage2_20261029.png)

## 条件与计时边界

- 时间：2026-10-29；两后端公开版本均为0.54.0。Windows11，MSVC14.51.36231，CPU Ryzen AI9 HX370，使用同一个本机GMP DLL，CPU未定频。
- 设备：RTX4060 Laptop，CUDA编号1、OpenCL编号1；驱动610.88，CUDA13.3.73。OpenCL枚举中另有重复NVIDIA平台，原始设备清单保留，不把两个编号当作通用映射规则。
- 本轮未修改GPU设置。500ms遥测共864条，全部记录1200MHz；利用率≥80%的640条亦为1200MHz。采样不是逐核时钟证明。GPU0 RTX4070 Ti有外部负载；没有在其上运行本轮算术，宿主环境仍非完全隔离。功耗读数不用于能效结论。
- 4个已有余因子save：M503/318 bits、M2003/1939 bits、M5003/4667 bits、M8011/7995 bits。均为Edwards、B1=50000；M2003 sigma=27，其余sigma=26。实际N位宽用于横轴，不能用原指数代替。
- B2=260000与2600000；两后端固定D2310、P=φ(D)/2=240，F树补齐256叶，giant数分别114/1127，G批数1/5。显式普通模数、关闭自动承载与运行时学习，不做D搜索。
- batch64、arena512、fold640 MiB。OpenCL arena是owned-buffer硬预算，CUDA arena是NTT工作区策略，两者口径不同。本矩阵均未接近预算边界，不因预算改变D/P。OpenCL owned-buffer峰记录不能与CUDA模块峰相加。
- 每格每后端3次，反转后端先后次序。无额外warmup，不剔除首轮；未清理驱动程序缓存，因此“首轮”不等于严格冷JIT。每次记录完整范围，图中误差棒为min/max，折线为中位数。
- CUDA采用正式脚本的isolated内核构建，固定Goldilocks PTX后端3、add/sub1、outer unroll0、split-compile8。OpenCL采用原生CL1.2整数算子与默认tile计划，没有内核翻译或CUDA链接依赖。

引擎时间取各程序的 `stage2_full_wall.total`，包含其内部规划、初始化、自检和主计算，排除Stage1。OpenCL包含设备枚举、运行时/程序准备及GMP叶子哨兵检查；CUDA也计入既有引擎内检查，但部分设备规划发生在引擎边界外。两者的init/main组成不同，不能把main或init直接当作同一算子的公平微基准。

完整进程时间覆盖父驱动、保存点校验、规划、子进程、引擎、结果发布和退出观察；不含生成Stage1 save的成本。CUDA在本矩阵中有较大的计时外固定开销，故不能用引擎倍率描述用户实际看到的总等待倍率。两后端检查策略不同；本报告比较完整实现，不将全部差距归因于NTT。

## 固定D完整曲线结果

单位s，各自为n=3中位数；两条时间边界分别计算，不能相加。

| 实际bits | B2 | CUDA引擎 | OpenCL引擎 | 引擎比值 | CUDA进程 | OpenCL进程 | 进程比值 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 318 | 260000 | 0.091763 | 1.536170 | 16.74 | 1.518607 | 1.811642 | 1.19 |
| 318 | 2600000 | 0.115804 | 1.685133 | 14.55 | 1.575858 | 1.947485 | 1.24 |
| 1939 | 260000 | 0.216326 | 2.346756 | 10.85 | 1.625386 | 2.556713 | 1.57 |
| 1939 | 2600000 | 0.262183 | 2.951494 | 11.26 | 1.707403 | 3.214623 | 1.88 |
| 4667 | 260000 | 0.694303 | 3.717712 | 5.35 | 2.149926 | 4.019331 | 1.87 |
| 4667 | 2600000 | 0.848672 | 12.868189 | 15.16 | 2.302088 | 13.326757 | 5.79 |
| 7995 | 260000 | 2.349792 | 15.709009 | 6.69 | 3.783258 | 16.011810 | 4.23 |
| 7995 | 2600000 | 2.875544 | 78.835512 | 27.42 | 4.370134 | 79.095778 | 18.10 |

OpenCL低B2下，318/1939/4667/7995 bits首条引擎时间分别为5.482/5.809/8.640/22.094 s。其余重复明显较短，符合程序准备/驱动缓存成本的影响；未做逐编译事件剖析，不能把首尾差额全部当作JIT时间。7995-bit、高B2的三条则为78.836/78.343/79.788 s，差距在重复执行后仍然存在。

## 阶段证据与优化方向

### 初始化不是全部瓶颈

OpenCL的“giant、G树和fold”占一个合并区间。以下为日志中的代表调用，不把它误称为纯点生成或纯NTT时间。

| 输入 | baby生成/归一化 | F树/逆准备 | giant/G/fold合并 | 下降 | 完整引擎 |
|---|---:|---:|---:|---:|---:|
| 4667 bits、B2=2600000、重复2 | 0.99 s | 0.54 s | 10.17 s | 0.20 s | 12.868 s |
| 7995 bits、B2=260000、重复2 | 6.72 s | 0.97 s | 6.74 s | 0.28 s | 15.709 s |

阶段边界由同一调用的 `stage2_phase.previous` 提取，仅两位小数；这些代表值不是三次中位数。7995-bit、高B2的init中位数8.604 s、main中位数70.303 s，也显示主要差距不能只靠缓存初始化解决；两个分项中位数不要求相加等于总时间中位数。

### 点生成与组织优先于直接归罪NTT

当前OpenCL点算子使用独立标量ladder、普通模数SOS乘法，尚无生产CUDA的giant chain。为限制显示设备上的单次命令规模，宽位数点生成分小块提交；它限制了并行度，也增加了分配、提交、等待和坐标回读。此处的分块不是已校准的系统超时保证。

7995-bit低B2样本的baby点准备已达6.72 s，显著高于同次F树/逆准备0.97 s。giant合并区间还包含G树和fold，现有记录不足以把其中各项精确分开。下一轮应先补设备事件与主机边界，再比较CPU点生成、设备giant chain和更合适的ladder分块，而不是假定全部27倍差距来自NTT。

归一化、最终块乘积/GCD已在CPU/GMP执行。其选择依据见 [CPU尾部成本](STAGE2_CPU_TAIL.md) 及 [OpenCL点算子对照](../web/opencl-stage2-port-plan-20261027.html#points-closure)；不应为了“全GPU”而重新加入昂贵的设备GCD。

### NTT与提交仍需改进

当前原生NTT以全局radix2与local tile组成，尚无CUDA的outer联合组织、短NTT Graph和调优表。整数多项式每次乘法在verdict边界同步，虽然同一次乘法内部不逐pass等待，独立小请求仍存在较多主机/设备边界。

本矩阵最大NTT仅2^19，不能据此评价常见大P生产任务的2^23及以上吞吐。OpenCL计数包含其启动检查，CUDA曲线计数明确排除算术自检；不能直接把两份计数相减当作算法额外工作量。

## 正确性与生产行为

最终OpenCL二进制在RTX4060 Laptop和AMD gfx1150各通过13项检查：

- PARAM0 Stage2真实正命中；N=257000771、sigma6、B1=20、B2=300、D30，得到因子257。当前CUDA二进制另跑同一save，得到相同因子。
- 普通NF、梅森NF、余因子普通/显式承载；承载先验证整除关系，最终因子属于target N。
- 错误checksum、非法承载、owned-payload预算不足、显式不支持的自动承载，均未发布成功结果。
- 基本D模型、plan-only；plan-only不生成曲线结果。
- 队列预算失败不推进完成状态；第一次运行在曲线开始后请求安全停止，恰好提交1条收据。
- 同一任务请求5条、save只有3条，夹限后继续；改变worktodo顺序后先完成另一任务，再续算原任务的记录2/3。最终原任务记录1/2/3各一次，没有重跑记录1。

AMD另用M2003/1939 bits及M8011/7995 bits、Edwards B1=50000、B2=260000、D2310跑通完整NF。其日志明确记录 `point_backend=cpu_gmp_amd_wide_guard`：这证明CPU点回退与原生OpenCL多项式接线，不代表修复或验收了宽位AMD设备点内核。这两次是功能验证，不加入RTX4060 A/B，也不构成AMD性能对比。

13项检查与上述真实保存点只是有限资格；Intel、其他驱动、长期队列、全部取消/设备丢失边界和大B2生产规模尚未完整覆盖。没有将原型切换为默认CUDA替代品，也没有将其加入CUDA自动release拆包。

## 身份、数据与复现

计时二进制：

- OpenCL SHA256：`842e89e94b9675666aeab9669803a27b4e16bcbce7b80642888738a577a765f5`。
- CUDA SHA256：`e24de9c753d50ad8642bc69984374e768da60b6f4235f955358fcee7bf28b5e0`。
- GMP DLL SHA256：`969f0e96ac587dedfecf611ca3442fc7211f9218f1a553214c83548dcdbcfd6e`。

原始证据位于 `data/experiments/opencl_production_20261029/`：

- `ab01/`：两套冻结二进制、OpenCL资源、原save、命令、TOML、每次日志与JSON结果、GPU遥测、全部48条记录。汇总逐条核对日志/结果与实际执行的二进制身份。
- `verification_final_gpu1/`、`verification_final_amd/`：最终13项验收及队列文件。
- `extra_qualification/`：CUDA因子257对照、AMD真实宽位NF。
- `build02/`、`cuda_build/`：工具链与源码身份、构建记录；配置/中间验证失败独立保留，不进入性能样本。

派生数据为 `data/benchmarks/opencl_production_20261029/{cells.csv,pairs.csv,summary.json}`，图为 `data/figures/opencl_cuda_stage2_20261029.png`。数据由Git忽略，分发报告时须另附冻结证据。

复现工具：

- [构建](../../tools/build/build_stage2_opencl_local.ps1)：原生worker；[完整探针/worker构建](../../tools/build/test/build_opencl_stage2_ntt.ps1)。
- [生产门禁](../../tools/test/check_opencl_stage2_worker.py)：`--build <dir> --out <new-dir> --device <OpenCL-index>`，只在指定隔离目录写队列/结果。
- [完整A/B](../../tools/bench/bench_opencl_cuda_stage2.py)：`--opencl-build <dir> --cuda-build <dir> --out <new-dir>`；默认4个既有指数、两档B2、D2310、每侧3次。可指定输入目录、两套设备编号与范围。要求两程序公开版本一致。
- [汇总/绘图](../../tools/bench/report_opencl_cuda_stage2.py)：`--run <ab-dir> --out <new-summary-dir> --figure <new.png>`。

运行语义、预算和目前不支持的功能以 [OpenCL架构](../architecture/OPENCL.md#生产-stage2-入口) 和 [Stage2使用](../usage/STAGE2.md#原生-opencl-入口) 为准；临时实施记录保留于 [草稿](../development/OPENCL_STAGE2_PRODUCTION_DRAFT.md)。

## TODO

1. 分离device ladder、CPU归一化、G树、fold、verdict等待的事件与主机计时，做固定D单项对照。
2. 优先比较CPU点生成分流、OpenCL giant chain与宽位ladder组织；验证后再建立分流模型。
3. 减少重复设备枚举与程序准备，设计有设备/驱动/源码/选项身份的程序缓存，保持每曲线进程隔离与错误传播。
4. 对独立短NTT请求设计有界提交/检查窗口，证明workspace复用与verdict寿命；再推进tile/outer联合组织。
5. 扩展大P、长NTT、16384-bit、真实高B2及更多设备；校准D/预算模型，接入独立OpenCL tune后再讨论自动承载和Auto B2。
