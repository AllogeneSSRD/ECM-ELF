# 原生 OpenCL Stage2：性能诊断与 CUDA 对照

## 当前结论

0.54.1提供独立 `ecm_opencl_stage2.exe`，支持现有文本save、ECMSTAGE2队列、曲线级进度与结果发布。默认保留设备点生成；AMD目标超过512 bits保留CPU点资格回退。OpenCL仍为显式入口，不替换默认CUDA。

**已确认的大差距首先来自点生成的计算与提交组织，不能全部归因于NTT。** 当前默认启用NVIDIA显式32线程工作组、宽位256点提交、giant基点复用和同context程序缓存；独立GMP检查、因子验证、队列行为和多项式算法保持。

固定D的8格矩阵，OpenCL引擎中位数之和由0.54.0的 **119.649975 s** 降至0.54.1的 **17.418295 s**，减少 **85.4%**，8格均改善。同期重跑同一CUDA二进制为 **7.460040 s**，与冻结基准7.454387 s接近。当前OpenCL/CUDA引擎中位数之和的比值为2.33，不能称为单条生产曲线加速比。

完整父进程中位数之和由121.984139降至19.376465 s；同期CUDA为18.315387 s。该口径OpenCL约为CUDA的1.06倍。引擎与父进程是两个边界，不能相加，也不能用引擎倍率代替用户实际等待倍率。

![冻结OpenCL、当前OpenCL与重跑CUDA](../../data/figures/opencl_stage2_evolution_final_20261029.png)

上排为n=3中位数及min/max，纵轴对数；下排蓝线“旧OpenCL/新OpenCL”越高表示改善越大，橙线“新OpenCL/CUDA”越高表示OpenCL越慢。它们的方向不同，须按图例读取。

## 条件、版本与计时边界

- 2026-10-29；Windows11、MSVC14.51.36231、CPU Ryzen AI9 HX370，CPU未定频。三套二进制使用相同本机GMP DLL。
- RTX4060 Laptop，CUDA1/OpenCL1；驱动610.88、CUDA13.3.73。设备编号仅为本机ICD枚举结果，不是通用CUDA/OpenCL映射。GPU0 RTX4070 Ti有外部高负载，本轮未在其上计算。
- 未修改GPU频率、功耗、驱动。新A/B的500ms遥测222条全部1200MHz，其中利用率≥80%的72条亦为1200MHz；冻结基线864条也全部1200MHz。采样不能证明每个指令周期的频率；功耗读数不用于能效推断。
- 真实Edwards save：M503/实际318 bits、M2003/1939 bits、M5003/4667 bits、M8011/7995 bits；B1=50000，M2003 sigma27，其余sigma26。横轴使用剥除已知因子后的实际N位数。
- B2=260000、2600000，固定D2310、P=φ(D)/2=240，F树补齐256叶，giant114/1127，G批1/5。普通模数，关闭自动承载、运行时学习与短校准，无D搜索。
- batch64、arena512、fold640 MiB。OpenCL arena是全部owned buffers硬预算；CUDA arena是NTT工作区策略，口径不同；本矩阵均未接近边界。程序、驱动和context不计入owned-buffer账本。
- 冻结OpenCL0.54.0基准与本轮OpenCL0.54.1/CUDA0.54.0各覆盖8格、每格3次。本轮共48次完整NF，后端次序反转，无额外warmup、不丢首轮，驱动缓存未清空。新A/B继承的NTT/OpenCL专家环境覆盖为空。
- CUDA使用冻结isolated正式构建：Goldilocks PTX后端3、add/sub1、outer unroll0、split-compile8。OpenCL仍为CL1.2整数与原有tile计划。本轮没有修改NTT蝶形、S4公式或D/P选型。

引擎时间取 `stage2_full_wall.total`，包含各自内部规划、初始化、自检、主计算，不含Stage1。OpenCL包含设备枚举、context/程序准备及GMP叶哨兵；CUDA部分设备规划发生在引擎边界外。init/main组成不同，不能直接当作同一算子的微基准。

父进程墙钟覆盖驱动、保存点校验、规划、曲线子进程、引擎、发布和退出观察。CUDA有较大的边界外固定成本。小任务中固定开销与主机波动显著，不能从少数小N总时间的非单调性推导算术复杂度。

## 完整曲线比较

单位s，全部为n=3中位数。旧OpenCL来自冻结0.54.0矩阵，新OpenCL和CUDA为本轮重跑；不把首轮编译成本删去。最后一列是完整父进程的比值。

| 实际bits | B2 | 旧OpenCL | 新OpenCL | CUDA | OpenCL减少 | 新OpenCL/CUDA引擎 | 父进程比值 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 318 | 260000 | 1.536170 | 1.235314 | 0.097063 | 19.6% | 12.73 | 0.90 |
| 318 | 2600000 | 1.685133 | 1.385780 | 0.115394 | 17.8% | 12.01 | 1.00 |
| 1939 | 260000 | 2.346756 | 1.023878 | 0.215294 | 56.4% | 4.76 | 0.84 |
| 1939 | 2600000 | 2.951494 | 0.947209 | 0.265803 | 67.9% | 3.56 | 0.99 |
| 4667 | 260000 | 3.717712 | 1.507060 | 0.694712 | 59.5% | 2.17 | 0.87 |
| 4667 | 2600000 | 12.868189 | 2.624594 | 0.849731 | 79.6% | 3.09 | 1.35 |
| 7995 | 260000 | 15.709009 | 2.735037 | 2.350845 | 82.6% | 1.16 | 0.77 |
| 7995 | 2600000 | 78.835512 | 5.959423 | 2.871198 | 92.4% | 2.08 | 1.42 |

7995-bit、高B2的新OpenCL为5.946468/5.959423/6.113412 s，旧版78.342615～79.788052 s。低B2新OpenCL为1.972181～2.739943 s；其较大的主机/准备波动全部保留。表中中位数不构成逐项置信区间或跨机器保证。

## 主要差距如何定位

### 点内核而非坐标传输占主导

固定4667 bits、B2=2600000的诊断构建，原64点提交完整引擎12.187472 s，点kernel事件累计10.813123 s，占88.7%。coordinates为10.826610 s，包含分配、上传、提交、等待、坐标回读和GMP导入。两者仅差约0.0135 s；这排除了“这条样本主要卡在坐标传输或CPU等待”的解释，没有证明所有输入均如此。

同一诊断输入的逐项对照如下，每行n=1；这些是不同配置，不能当作多次重复。全部仍执行真实完整曲线和独立校验。

| 点提交chunk | 工作组 | giant基点 | 点kernel s | 完整引擎 s |
|---:|---:|---|---:|---:|
| 64 | 运行时 | 原Q、完整iD | 10.813123 | 12.187472 |
| 240 | 运行时 | 原Q、完整iD | 7.738507 | 9.886708 |
| 240 | 32 | 原Q、完整iD | 2.780082 | 5.047441 |
| 240 | 8 | 原Q、完整iD | 2.864427 | 5.113268 |
| 240 | 32 | 复用[D]Q、标量i | 1.382963 | 3.502137 |

第一行与其余行分别见 `diagnostic4667/`、`groups4667/`，属于两次诊断构建，完整源码/二进制身份各自冻结。组内group0/32/8和基点对照使用同二进制开关。工作组实验表明运行时默认组织不适合此宽位短批量，尚未测量实际SM占用率，不能把所有收益精确归于某个occupancy数值。

giant重标定采用 iD·Q=i·(D·Q)。每点ladder由约log₂(iD)位缩为log₂i位，省去约log₂D位的xADD/xDBL迭代；只额外构造一次[D]Q。它不是“一次差分xADD生成下一个giant”的chain。种子分母在target N上检查，不能绕过复合模数的非单位情形。

4667-bit点上传从原64点分块的51232 bytes降至240点分块的21448 bytes；点坐标下载均为1596656 bytes。该小量传输变化不能解释数量级收益。它仅统计点模块，不是完整Stage2传输量。

### 重复程序准备是独立的可消除成本

相同4667-bit、高B2、group32/chunk256/giant基点的同二进制缓存开关对照，各n=3：

| 程序缓存 | 每曲线build | cache hits | build API中位数 s | 完整引擎中位数 s |
|---|---:|---:|---:|---:|
| 关闭 | 44 | 0 | 0.670503 | 3.496535 |
| 开启 | 4 | 40 | 0.064630 | 2.717717 |

program按完整源码/编译选项在同一Runtime/context复用，kernel参数、根表和计划仍独立。主机规划也存在波动，所以不能把全部0.779 s总差额等同于build API差额。代码中另有进程内设备快照复用，本轮没有单独开关实验，不能单独报告其收益。

### 当前剩余差距仍集中于宽位点算术

本轮7995-bit、高B2的代表调用为重复1，完整引擎5.959423 s：

| 字段 | 秒 | 边界 |
|---|---:|---|
| 点kernel | 4.038105 | 设备事件；嵌套于点coordinates，约67.8%引擎总时间 |
| 点coordinates | 4.043051 | 含kernel、点分配/上传/回读/导入 |
| 点GMP取样 / CPU归一化 / seed | 0.052252 / 0.091778 / 0.001668 | 独立于coordinates，包含于baby/giant |
| F树与逆多项式 | 0.163155 | 主机墙钟，含相应提交和检查 |
| G树与fold | 0.418294 | 不含giant生成与叶oracle |
| scaled下降 | 0.181674 | 含叶子回读 |
| 叶oracle / 最终积与GCD | 0.058635 / 0.004221 | CPU墙钟 |

这些字段存在明确嵌套，不能逐行相加。F/G/下降也不是纯NTT设备时间，包含分配、S4、同步与检查。最长点kernel为0.765507 s，只证明此调用，不能保证更大P、完整64位标量或显示设备的系统超时安全。

同D/P/B2下，4667/7995-bit点kernel的n=3中位数为1.383064/4.037534 s，比例2.919，与位宽平方比2.935接近；局部可用 T_point≈6.3×10⁻⁸·N_bits² s 描述此固定1367点、当前标量分布和1200MHz条件，不能外推成所有B2/P/设备的成本模型。318-bit任务的点kernel约0.009 s，引擎约1 s，优先处理初始化固定成本。

CUDA同属单线程多精度SOS ladder，当前矩阵的giant seed统计为0；不能把这8格差距解释成“CUDA用了chain、OpenCL没用”。生产大批量下CUDA另有chain/paired seed，入口为 [batch.cpp](../../src/ecm/stage2/cuda/batch.cpp#L271)、[points.cpp](../../src/ecm/stage2/cuda/points.cpp#L202)。OpenCL的动态索引私有数组、noinline乘法与循环控制仍可能影响机器码效率，但尚无设备占用率/机器码对照，列为后续剖析问题，不当作已证明原因。

8格逐一核对，旧/新OpenCL的各长度NTT次数完全相同；同一7995-bit样本owned-buffer峰均为23159696 bytes。当前收益来自点计算/组织与程序准备；不宣称减少了多项式运算量、NTT蝶形成本或进程显存峰。

### 可选CPU点分流的完整曲线对照

最终同一0.54.1二进制，保持基点复用和程序缓存，只切换 `ECM_OPENCL_POINTS=device|cpu`；两位宽、B2=2600000、每侧各3次，共12条完整NF。点计时是coordinates主机边界；CPU模式继续使用OpenCL多项式和原始完整标量oracle。

| 实际bits | 设备点 s | CPU点 s | 设备点模式完整引擎 s | CPU点模式完整引擎 s |
|---:|---:|---:|---:|---:|
| 4667 | 1.386450 | 0.648431 | 2.723581 | 1.969520 |
| 7995 | 4.040140 | 1.517323 | 5.897047 | 3.328895 |

均为各字段n=3中位数，不要求分项中位数加和等于总时间中位数。CPU未定频、宿主仍有外部负载，准备成本波动保留。7995-bit本小P条件下CPU模式进一步缩短约43.5%，证明分流值得推进；该表不是大P或多worker下的吞吐模型，尚不自动改默认。证据为 `hybrid_wide/`，不混入前述48次默认A/B。

## 正确性与适用范围

- 最终0.54.1二进制在RTX4060 Laptop、AMD gfx1150各通过14项生产门禁：正因子257、普通NF、梅森/余因子承载、非法承载、预算、checksum、基本D、plan-only、队列停止与重排续跑等。
- RTX4060 Laptop显式CPU点模式另通过相同14项生产门禁，环境覆盖与证据保存在 `verification_cpu_gpu1/`。
- 新增种子非单位回归使用N=257000771、sigma6、B1=20、B2=300、D46。测试要求到达giant阶段、得到257且尚未下降，确实覆盖[D]Q种子分母路径，不能以baby阶段命中替代。
- 两卡NTT检查各包含84810个域算术值、8982512个roundtrip word与同量卷积word、235个门禁；覆盖程序键的源码/选项、context隔离与缓存关闭。GPU1另通过普通/梅森S4至16384-bit、2820门禁，树代数90642 words/7小曲线，以及8191-bit显式group32点运算16522 words。
- AMD1939/7995-bit真实Edwards save通过CPU点回退加OpenCL多项式NF，并使用原始[iD]Q校验重标定后的CPU点；没有取消宽位AMD设备点资格限制。
- 本轮48条NF与有限正命中不代表全部曲线、Intel、其他驱动、设备丢失、长队列或大B2资格。最大NTT为2^19；不能据此评价常见大P的2^23及以上。CUDA默认路径未修改，OpenCL尚未接入Auto B2/tune、自动承载比较或release自动拆包。

## 代码与复现

主要入口：

- [点选择、种子、计时与生产闭环](../../src/ecm/stage2/opencl/engine.cpp#L94)。
- [点工作组检查与提交](../../src/ecm/stage2/opencl/points.cpp#L37)、[设备ladder](../../src/ecm/opencl/resources/stage2/points.cl#L63)。
- [程序缓存](../../src/ecm/opencl/ntt/runtime.cpp#L248)、[设备快照](../../src/ecm/opencl/ntt/runtime.cpp#L78)、[算子桥接](../../src/ecm/opencl/ntt/native.h)。

计时二进制SHA256：OpenCL0.54.1 `61c1bbde5ea2c55b986a270f7b1faa3489656cc9896a3f62bb96941d087ee3a6`；冻结OpenCL0.54.0 `842e89e94b9675666aeab9669803a27b4e16bcbce7b80642888738a577a765f5`；CUDA0.54.0 `e24de9c753d50ad8642bc69984374e768da60b6f4235f955358fcee7bf28b5e0`；共同GMP DLL `969f0e96ac587dedfecf611ca3442fc7211f9218f1a553214c83548dcdbcfd6e`。

原始证据：`data/experiments/opencl_optimization_20261029/`，包含 `diagnostic4667/`、`groups4667/`、`cache4667/`、`cache7995/`、`ab_final/`、`verification_final_gpu1/`、`verification_final_amd/`、组件JSON/日志、`amd_wide/`、`ntt_count_comparison.json`与202项输入的源码/工具快照。冻结旧证据保持在 `data/experiments/opencl_production_20261029/`，未覆盖。失败或初步夹具不计入最终门禁或性能矩阵。

派生数据：`data/benchmarks/opencl_optimization_20261029/{current,evolution_final}/`，图在 `data/figures/`。data由Git忽略，分发报告时须另附证据。

- [构建](../../tools/build/build_stage2_opencl_local.ps1)、[完整探针构建](../../tools/build/test/build_opencl_stage2_ntt.ps1)。
- [生产门禁](../../tools/test/check_opencl_stage2_worker.py)：`--build <dir> --out <new-dir> --device <OpenCL-index>`。
- [完整A/B](../../tools/bench/bench_opencl_cuda_stage2.py)：`--opencl-build <dir> --cuda-build <dir> --out <new-dir>`；默认要求同公开版本，本轮用显式 `--allow-version-difference`比较有身份的冻结CUDA与新版OpenCL。
- [命名变体对照](../../tools/bench/bench_opencl_stage2_variants.py)：`--variants <json> --out <new-dir>`；JSON数组各项指定name、build和environment，局部设置子进程环境。
- [A/B统计](../../tools/bench/report_opencl_cuda_stage2.py)及[三版本比较图](../../tools/bench/report_opencl_stage2_evolution.py)：核对原日志、结果、二进制与输入身份，再生成汇总。

运行语义和专家开关见 [OpenCL架构](../architecture/OPENCL.md#生产-stage2-入口)、[Stage2使用](../usage/STAGE2.md#opencl-性能对照开关)。[实施草稿](../development/OPENCL_STAGE2_PERFORMANCE_DRAFT.md)保留待用户处理。

## TODO

1. 宽位点运算优先：在当前标量基点重用上，研究差分chain、分段退化恢复、减少SOS依赖链与私有数组访存；以原始完整标量oracle验收。
2. 比较CPU点分流与设备批量；分流模型需同时考虑位宽、点数、标量长度及CPU负载，不能只按N阈值或单个样本切换默认。
3. 为短NTT的verdict设计有界提交窗口，证明workspace与结果寿命，再推进tile/outer联合组织；保持检查和异常传播。
4. 小任务进一步拆分ICD/context/build固定成本；缓存须保留设备/驱动/完整源码与选项身份，不能共享可变kernel参数。
5. 扩展大P、2^23以上NTT、16384-bit真实生产save与更大B2；验证点chunk/命令时长和显存边界后再校准OpenCL D模型。
