# 原生 OpenCL Stage2：性能诊断与 CUDA 对照

## 当前结论

0.54.3提供独立 `ecm_opencl_stage2.exe`，支持现有文本save、ECMSTAGE2队列、曲线级进度与结果发布。NVIDIA的513～8192-bit设备点默认采用反馈式chunk／窗口计划；CPU点、其他厂商与更宽NVIDIA保持原资格和分块。AMD目标超过512 bits保留CPU点资格回退。OpenCL仍为显式入口，不替换默认CUDA。

**已确认的大差距首先来自点生成的计算与提交组织，不能全部归因于NTT。** 当前使用NVIDIA显式32线程工作组、保守初始chunk和反馈升档、giant基点复用及同context程序缓存；独立GMP检查、因子验证、队列行为和多项式算法保持。当前默认计划的条件及同二进制结果见[自动点执行计划](#自动点执行计划)。

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

### 冻结基准中的宽位点成本

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

## 点提交组织与多曲线吞吐

本节为独立的1500MHz实验，不能与上文1200MHz矩阵直接混算。主要结论是：**小P的CPU点分流适合并行多曲线；重复开启GPU点进程收益很小或倒退；优先改善单曲线点提交组织，通常比增加GPU worker更有效。** 更大的P已有2²³ NTT对照，但生产大B2、不同厂商的最佳并发度仍未确定。

### 输入与证据口径

- 设备仍为4060 Laptop，驱动610.88、同CPU与GMP。未修改GPU设置；GPU0有外部负载，不参与计算。小P重复矩阵1377条250ms遥测中1375条1500MHz，另2条1890MHz均为0%利用率／0MiB；所有非零利用率样本为1500MHz。大P筛选412条均为1500MHz。采样不等于逐周期频率保证。
- 完整多曲线采用实际N=4667/7995 bits，从原save只取N，独立生成有效PARAM0、sigma26…29、B1=20的小存档；不同sigma共享N，后端间使用完全相同save。此处不是重复执行一条曲线假装多条曲线。
- 小P：D2310、P240、B2=2600000、giant1127，每组固定4条曲线，最大并发C=1或4，各组n=3。共144条完整NF。C=2的初筛另存，不与这组重复合并。
- 大P：D30030、P2880、B2=26000000、giant867，最大NTT=2²³，每组固定2条曲线，C=1或2，**各组仅n=1筛选**，共12条NF。arena1536 MiB；小P为512 MiB，batch64、fold640 MiB。
- 多曲线采用冻结OpenCL0.54.1与CUDA0.54.0二进制，SHA见上文；程序每曲线仍创建隔离进程。吞吐定义为完成条数／整组墙钟，覆盖进程启动、JIT/准备、强制检查、结果发布与子进程排空；不含预先生成Stage1 save。无额外warmup，不丢首轮，驱动磁盘缓存未清空。
- Windows Job统计整组进程树CPU时间与同时存在的commit峰值。CPU核当量=Job CPU秒／整组墙钟；commit既非物理内存也非显存。NVML设备used采用250ms样本最大值，包含驱动等设备占用，可能漏过短峰，不等于owned-buffer账本或进程精确峰值。各字段均分别取组间中位数，不相加。

### 多开进程能否提高吞吐

单位curves/s；小P每格n=3个固定4曲线组，中位数。倍率为同模式C4/C1。

| N bits | 点生成／多项式 | C1 | C4 | 吞吐倍率 |
|---:|---|---:|---:|---:|
| 4667 | OpenCL GPU／OpenCL | 0.4683 | 0.5022 | 1.07× |
| 4667 | CPU GMP／OpenCL | 0.6002 | 1.5689 | 2.61× |
| 4667 | CUDA | 0.6238 | 0.8936 | 1.43× |
| 7995 | OpenCL GPU／OpenCL | 0.2094 | 0.1900 | 0.91× |
| 7995 | CPU GMP／OpenCL | 0.3374 | 0.7829 | 2.32× |
| 7995 | CUDA | 0.3156 | 0.3121 | 0.99× |

7995-bit的GPU点C4单曲线进程中位延迟约21.00 s，而C1约4.74 s；不能把多曲线吞吐与单曲线加速混为一谈。CPU点C4延迟约4.99 s，亦高于C1约2.97 s，但整组结束更早。该结果支持CPU点生成、准备与GPU多项式的重叠，不能证明不同context的GPU内核在SM上同时执行。

资源代价以7995 bits为例，表内C1→C4，各为n=3组间中位数：

| 模式 | CPU核当量 | Job commit MiB | NVML采样设备used峰值 MiB |
|---|---:|---:|---:|
| OpenCL GPU点 | 0.92→3.90 | 1029→3749 | 741→2949 |
| OpenCL CPU点 | 0.90→3.48 | 387→1541 | 197→755 |
| CUDA | 0.92→3.77 | 1116→4003 | 746→2981 |

OpenCL两种点路径的owned-buffer峰都仅约22.1MiB。它与GPU点模式741MiB的NVML采样值差距很大：工作区账本不包含program、context及驱动私有资源。CPU点单worker少约544MiB设备采样占用，但尚未把这部分逐项归因于stack或驱动堆。不能用owned-buffer预算乘worker数预测进程显存，也不能将表中两类内存相加。

![小P多曲线吞吐与资源](../../data/figures/opencl_curve_concurrency_small_p_resources_20261029.png)

各列分别为两种位宽；从上到下是吞吐、CPU核当量、同时Job commit、NVML采样设备占用。上排误差棒为3组min/max；连线只连接已测C1和C4，没有测量中间点的曲线形状。

大P的n=1筛选：GPU点C1→C2为0.0732→0.0593 curves/s；CPU点为0.1158→0.1734，约1.50×；CUDA为0.2295→0.2175。CPU点仍可改善吞吐，但收益缩小，且未追上该输入的单worker CUDA。不能据此默认4 workers或承诺生产B2下收益。

![大P并发筛选](../../data/figures/opencl_curve_concurrency_large_p_resources_20261029.png)

OpenCL规范允许多队列独立提交，但不保证work-group物理并行；不同队列的依赖也需显式同步。[Khronos OpenCL API规范](https://registry.khronos.org/OpenCL/specs/unified/html/OpenCL_API.html)。NVIDIA对CUDA明确区分同context并发与多个context的时间片；这是CUDA的规则，不能直接当作所有OpenCL实现的保证。[NVIDIA CUDA Best Practices：Concurrent Kernel Execution / Multiple Contexts](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#concurrent-kernel-execution)。本实验只证明这些进程组合的实际吞吐，不据事件重叠推断SM并发。

### 单曲线点批量为何更有效

CL查询本卡compute units=24。group32下，256点命令只有8个work-groups；P240时一个G批也只有8组。仅增大点chunk会被G批截断，不能增加单次有效点数。新的坐标窗口让多批点先生成，再按原G边界进入G树；G批、NTT工作和每批原始标量校验均保持。

同二进制比较：0.54.2默认与 `POINT_WINDOW=4080, POINT_CHUNK=768`；四位宽、两B2、每格n=3，并另跑冻结0.54.1，共72条NF。B1=50000、原始Edwards save与上文矩阵相同；本节1500MHz。八格均核对NTT长度计数、点总数、坐标下载量、save身份及GMP checked_words相同。较高B2的结果如下：

| N bits | 默认点kernel s | 窗口点kernel s | 默认引擎 s | 窗口引擎 s | 引擎减少 |
|---:|---:|---:|---:|---:|---:|
| 318 | 0.007360 | 0.003970 | 0.588833 | 0.543267 | 7.7% |
| 1939 | 0.206541 | 0.114919 | 0.876159 | 0.740925 | 15.4% |
| 4667 | 1.106824 | 0.605121 | 1.887931 | 1.348631 | 28.6% |
| 7995 | 3.228669 | 1.757417 | 4.463823 | 3.058093 | 31.5% |

各字段是n=3中位数，分项不要求相加等于总时间中位数。7995-bit点提交由6次减至3次，坐标下载仍2734000 bytes；收益不是少算点或跳过校验。窗口+chunk的8格引擎中位数之和9.532801 s，默认11.681403 s，约减少18.4%。低B2只需一批giant，点内核基本不变；318/1939-bit完整引擎分别有约1.9%/3.7%的反向波动，不能声称逐格收益。冻结0.54.1之和12.462553 s，但默认新旧点kernel基本相同，初始化波动不能归为算法优化。

![单曲线点窗口完整矩阵](../../data/figures/opencl_point_windows_20261029.png)

更大的P使用大P筛选中的sigma26有效Mont save，B1=20，同D/B2/arena，n=3：

| 点提交设置 | 点kernel s | 完整引擎 s | 最长点命令 s | owned-buffer峰 MiB |
|---|---:|---:|---:|---:|
| 默认chunk256、窗口0 | 10.440042 | 13.375935 | 0.764792 | 262.75 |
| 仅chunk768、窗口0 | 3.967499 | 6.899059 | 0.765206 | 262.75 |
| chunk768、窗口4080 | 3.967338 | 6.903580 | 0.765233 | 262.75 |

这里giant867<P2880，窗口没有合并多批G的机会；收益由chunk填充带来，不能都归为跨批窗口。点提交16→6次，引擎约减少48.4%，NTT和检查相同。最长命令没有按点数线性变长，符合更好利用SM的解释，尚未测实际occupancy。窗口不是异步双缓冲，也没有实现点／NTT同时执行。

768也不是通用最优值：固定窗口的7995-bit小P初筛，chunk512/768/1024/1536引擎约3.459/2.945/3.698/3.176 s，各n=1。更大的命令还取决于私有内存、波数与标量长度；这些0.54.2手动对照不能证明跨设备最优性。当前自动计划的资格、反馈门限和回退见下节。

### 自动点执行计划

默认`ECM_OPENCL_POINT_AUTO=1`。资格限定NVIDIA、真实目标N为9…128个64-bit limbs、既定group32，且没有显式point chunk/group覆盖；CPU点、更宽NVIDIA与其他厂商维持原分流／分块。该计划不选择CPU点，也不读取CUDA tune资料。

首条命令使用原初始化chunk。后续仅从实际完成的点命令学习；没有额外warmup或重复校准点。令b为命令最大标量位数、q为有效点数、U为设备compute units，估算提交轮数为 `v=ceil(ceil(q/32)/U)`；保留最大的 `c=T_kernel/(b×v)`。下一命令预测`1.15×c×b_next≤1s`才允许升档，上限为`max(seed,min(4096,32U))`；设备规模不足以超过seed时不升档。小于一个完整工作组的样本不参与训练。该轮数是规划模型，不是实测occupancy；1秒仅为升档门限，首命令及回退命令仍使用原批量，不保证硬时限。

自动窗口优先取`lcm(G_batch,点上限)`，超过坐标额度则取额度内的整G批，至多65536点；随后按原G边界构树与检查。每条设备命令升档前还按当前owned-buffer余额检查索引及X/Z向量payload，保留原seed能够容纳的请求。没有提高预算，也不把该余额当作设备free memory或全卡多worker预算。

**同二进制A/B**：0.54.3，SHA256=`bc09f7b833879615c9f5b9ccd3e361bf62c172fe625b03686a97744f17e6d147`；legacy为AUTO=0，default不覆盖环境，manual768为AUTO=0、chunk768、window4080。设备／CPU／GMP同前述1500MHz研究，未调整显卡设置。各格n=3完整NF、无额外warmup、不丢首轮，父进程时间也保留。主矩阵223条500ms遥测中219条1500MHz、4条1890MHz均为0%利用率；所有非零利用率样本均1500MHz。大P169条、8191-bit70条与16384-bit99条均为1500MHz。

318/1939/4667/7995-bit仍使用原Edwards save、B1=50000；D2310、P240、两B2和预算同前。主矩阵共72条NF，全部逐格核对save身份、NTT计数、点数、坐标下载量与原始GMP checked_words相同。八格引擎中位数之和为legacy **11.662182 s**、default **9.734287 s**、manual768 **9.710380 s**；默认减少约16.5%，与手动设置的矩阵总和差约0.25%。该和不代表任意生产曲线的统一加速比。

| 输入 | B2 | legacy引擎 s | 默认引擎 s | 手动768引擎 s | 默认减少 |
|---|---:|---:|---:|---:|---:|
| 1939 bits，P240 | 2600000 | 0.824579 | 0.764361 | 0.745767 | 7.3% |
| 4667 bits，P240 | 2600000 | 1.870508 | 1.402659 | 1.372543 | 25.0% |
| 7995 bits，P240 | 2600000 | 4.471287 | 3.015644 | 2.993788 | 32.6% |
| 8191 bits，P240 | 2600000 | 4.764276 | 3.127920 | 3.147123 | 34.3% |
| 7995 bits，P2880 | 26000000 | 13.371597 | 7.588982 | 6.877481 | 43.2% |
| 1939 bits，P2880 | 2600000000 | 21.170231 | 9.590188 | 未测 | 54.7% |

各字段分别取n=3中位数。8191-bit使用既有Mont save首记录，B1=1000、sigma20260922；实际参数来自记录，文件名不控制sigma。7995-bit大P为前述有效Mont save、B1=20、sigma26，arena1536 MiB。1939-bit长giant对照使用原Edwards save，B1=50000、sigma27、arena512 MiB，最大NTT2²⁰；先测一遍，再补两遍，全部保留。表中不同输入不能合并拟合纯位宽规律。

小P宽位默认实际选择768点、3840点窗口，点提交6→3次，7995-bit点kernel中位数3.228998→1.759094 s。大P默认16→7次、点kernel10.439977→4.628997 s；手动768为6次／3.968405 s。首条保守命令使剩余baby点多出一条尾命令，默认完整引擎仍比手动设置慢约10.3%，这项差距保留，不把自动计划称为全局最优。大P当前owned-buffer峰仍为275517488 bytes；没有据此推断进程总显存不变。

1939-bit长giant范围共89462点，其中giant86582、G批31。默认自动选择768点／11520点窗口，设备点命令373→118次，点kernel17.251998→5.790642 s。两侧NTT长度计数、原始GMP覆盖和44373152 bytes坐标下载一致；没有以减少运算或检查换取54.7%的完整时间收益。两批遥测合计185条均1500MHz。该B2仍低于常见生产上界，不能替代生产全矩阵。

![默认点计划与关闭／手动设置](../../data/figures/opencl_point_plan_audited_20261029.png)

图上排为完整引擎，下排为其中嵌套的点kernel；误差棒为min/max。8191-bit仅测高B2，不能填补为低B2数据。低B2单G批通常不改变点命令，完整时间仍有准备波动；不承诺每格都加速。

318-bit高B2初始n=3的完整引擎中位数为0.564385→0.629049 s；该输入自动计划未启用，点kernel约0.00736 s、6条命令均相同。追加n=5对照为0.601790→0.631737 s，约5%差距；两侧范围0.462158…0.670488／0.563040…0.675239 s重叠，主要差异在初始化，主计算中位数0.131860／0.132607 s。准备时间仍待更空闲宿主条件核查，原始两批均保留，未用复测替换主矩阵。

**更宽位限制**：N=2¹⁶³⁸⁴−15、有效GMP-ECM PARAM0 save、B1=20、sigma26、B2=300、D30/P4、共16点，n=3。AUTO=0／默认分别8.046733／8.097050 s，均保持chunk16、4条点命令，点kernel约7.445 s；本输入最长命令仍约2.253 s，默认策略不构成显示设备时限保证。显式CPU点模式完整中位数0.577957 s、点坐标计算约0.016 s，均得到NF。它是极小P样例，不能据此固定全范围CPU分流阈值；CPU路径的点来源和检查结构不同，该对照未混入要求checked_words完全相同的自动计划矩阵。更宽位private资源及CPU/GPU分流继续优先研究。

0.54.3在GPU1设备点、显式CPU点、AMD各通过23项生产门禁，共69项，新增自动范围、手动锁定、非法开关及N607→M1214承载。点探针验证8191 bits／16522 words及6组计划边界检查；NTT快速回归84810域值、593904 roundtrip words、593904卷积words、215门禁，bad均0。它们与原有更大组件资格分别记录，不混报计数。

依据：[PointPlan](../../src/ecm/stage2/opencl/point_plan.h#L13)、[真实点反馈及账本准入](../../src/ecm/stage2/opencl/engine.cpp#L146)、[窗口接入](../../src/ecm/stage2/opencl/engine.cpp#L239)、[设备计算单元查询](../../src/ecm/opencl/ntt/runtime.cpp#L61)。原始证据在`data/experiments/opencl_point_plan_20261029/`的`matrix/`、`large_p/`、`edge8191/`、`fallback16384/`、`cpu16384/`、`recheck318/`、`longer_giant_{screen,repeat}/`和三份`verification_*/`；原型初筛另存`screen7995/`。派生统计为`data/benchmarks/opencl_point_plan_20261029/audited/`。前序快照与本版本构建／资源身份各自保留，不改写前序证据。

[同输入变体测量](../../tools/bench/bench_opencl_stage2_variants.py)使用本节三种环境配置；[点计划统计与图](../../tools/bench/report_opencl_point_plan.py)接受多次`--run DIR --out NEW --figure NEW.png`，先校验二进制、结果、输入／工作量及重复数。其他NVIDIA设备、长标量命令时限、Intel与全卡并发预算尚未完成性能资格验证。

### 私有算术、等待与NTT还剩什么

独立点探针使用相同7995-bit N、240个11-bit标量，所有输出逐点对照Python ladder，OpenCL/CUDA XADD6投影坐标逐word一致，每项n=3，无丢弃预热：

| 实现 | group | kernel中位数 s |
|---|---:|---:|
| OpenCL原SOS循环 | 64 | 1.258895 |
| CUDA现有SOS | 64 | 0.444175 |
| OpenCL原SOS循环 | 32 | 0.471022 |
| OpenCL内层展开4 | 32 | 0.378311 |

相同group64时OpenCL约2.83倍；改group32后大幅缩小差距。微基准的group32与CUDA group64不能当作编译器单变量比较。内层展开4在该group32输入约减少19.7%，但未接入生产默认。首次本进程build API约1.25→2.23 s，always_inline+展开2约7.27 s；驱动缓存会显著改变build成本，group64后续编译只有约0.01 s，这些不是冷编译保证。微基准处于1500MHz会话，未单独连续记录频率。

OpenCL报告private bytes=19024；CUDA NW128的cuobjdump为64 registers、stack16400 bytes。它们不是同一指标，不能直接计算spill差值；PTX虚拟寄存器亦不是硬件寄存器。已确认宽位private资源与组织方式敏感，实际cache miss、spill流量和占用率仍待硬件剖析。

同一group32点探针只改变等待方式：直接clWait的进程CPU中位数0.468750 s；flush后每1ms查询事件并休眠为0.031250 s，约减少93.3%，kernel均约0.471 s，调用墙钟约0.47165→0.47281 s。此处CPU计时有15.625ms量化，且仅代表这次长点命令。**阻塞API不等于宿主一定休眠；降低等待CPU也不等于减少设备计算。** 实验未把1ms等待推广到全部短NTT边界，未修改生产等待方式。

多项式侧仍在每次乘法后回读verdict/carry并等待完成；NTT是两次正向加一次融合乘积逆向，每层树已按shape批处理。令长度L=2ᵏ、tile=2ᵗ，当前一次卷积批调用的NTT kernel数为3(k−t+1)，不是3×batch；逻辑变换数才是3×batch。默认t=min(8,k)，L=2²³时48次NTT提交，另有pack/carry/S4。NTT主数组每slice的逻辑读写量约 `[48(k−t+1)+8]×L` bytes，不含根表、pack/S4且不等同实际DRAM流量。大NTT有充分计算时，继续加曲线主要增加资源竞争；短NTT／CPU准备间隙才更值得合并提交。

点成本也有算法差距。令w=ceil(N_bits/64)、I=floor(B2/D)+2，当前每个非零设备ladder标量k约需 `10+11 floor(log₂k)` 次Montgomery乘法（含输入/输出转换），每次SOS主项2w²次64-bit MAC。重标定后的giant总成本因此为O(w² I log I)。CUDA已有分段差分chain，段内新giant一次xADD，主体降至O(w²I)，另加种子与异常恢复。此处小P测试的CUDA实际未启用chain，不能把已测差额全部归于chain；生产大I时才需单独比较。OpenCL承载当前只改变多项式算术，点仍在真实N上计算，不能期待承载自动消除这项成本。

### 后续实现优先级

1. **扩展单曲线点执行计划**：当前已有NVIDIA 513～8192-bit反馈式chunk／窗口。继续校准group及最长命令、跨曲线复用合适的观测以减少保守首命令／尾批代价；扩大设备、长标量和更宽位资格。模型不能代替硬时限与全卡资源准入。
2. **推进分段giant chain**：复用现有六模乘xADD和原始[iD]Q oracle，先完成种子非单位、段内无穷远／退化恢复及全叶哨兵，再比较生产I范围。必要时研究按真实N投影的梅森快速点算术，不能借用多项式承载资格代替验证。
3. **CPU点分流配合有界多曲线**：按点数、位宽和CPU负载短校准，分别估算单曲线延迟与整组吞吐；小P优先试2～4曲线，大P先1～2。先建立全卡总预算、独立曲线进度／错误隔离，不能让每条曲线都独占同一份“可用显存”。
4. **共享context的跨曲线批处理**：相同N/算术形状可共享program、根表和只读模数；F/G/逆、workspace与结果按曲线隔离。把相同长度就绪卷积合入现有batch维，或使用独立queue/kernel及显式事件。不能直接并行调用当前全局ABI；共享不可变program与共享可变kernel参数是不同问题。吞吐须超过已测CPU分流和单曲线优化基线，才值得引入调度复杂度。
5. **有界NTT verdict窗口与tile/outer组织**：先让检查、工作区租约、借用所有权与失败排空跨少量操作保持有效，再批量检查；不能删除verdict换速度。长点命令低CPU等待可单独推进，短NTT采用自适应等待，避免逐次增加1ms延迟。SOS小展开保留为按设备校准候选。

### 代码与复现

- 点窗口／原始取样：[engine.cpp](../../src/ecm/stage2/opencl/engine.cpp#L115)、[G批调度](../../src/ecm/stage2/opencl/engine.cpp#L220)。SOS及ladder：[points.cl](../../src/ecm/opencl/resources/stage2/points.cl#L21)；同步等待：[points.cpp](../../src/ecm/stage2/opencl/points.cpp#L37)。
- NTT pass组织：[runtime.cpp](../../src/ecm/opencl/ntt/runtime.cpp#L342)；强制乘法检查：[polynomial.cpp](../../src/ecm/stage2/opencl/polynomial.cpp#L151)；树内shape批处理：[arithmetic.cpp](../../src/ecm/stage2/opencl/arithmetic.cpp#L128)。CUDA chain：[giant_chain.cuh](../../src/ecm/stage2/cuda/device/giant_chain.cuh#L6)、[调度入口](../../src/ecm/stage2/cuda/points.cpp#L338)。
- [多曲线测量脚本](../../tools/bench/bench_opencl_curve_concurrency.py)：`--opencl-build DIR --cuda-build DIR --out NEW --exponents 5003,8011 --parallel 1,4 --curves 4 --repeats 3`。大P另设`--d 30030 --b2 26000000 --arena-mb 1536 --exponents 8011 --parallel 1,2 --curves 2`。脚本隔离任务，未实现生产队列自动并发。
- [统计和图](../../tools/bench/report_opencl_curve_concurrency.py)：`--run DIR --out NEW --figure NEW.png`，先核对冻结二进制、输入、原日志及结果摘要。Job计账：[stage2_windows_job.py](../../tools/bench/stage2_windows_job.py)。
- [点编译／等待探针](../../tools/experiments/opencl/point_codegen.py)：`--save FILE --out NEW --device 1 --group 32 --repeats 3`；可选`--variants baseline,unroll4`、`--wait sleep`、`--cuda-dll FILE`。仅修改实验源副本，不改生产内核；PTX中local字段为程序范围静态信息，不是动态访存计数。
- 原始证据：`data/experiments/opencl_concurrency_20261029/`。主要目录为`small_p_repeat/`、`large_p_screen/`、`window_matrix/`、`large_window_matrix/`、`large_chunk_only/`、`codegen7995/`、`codegen7995_group64/`、`quiet7995/`。初筛及测试工具失败记录另存，未覆盖或并入最终样本。派生表／图为`data/benchmarks/opencl_concurrency_20261029/`与本节图。
- 窗口计时二进制0.54.2 SHA256=`b6c24966b1bab1925f22b4e48421269b44a3df739df461b0d735174fe47c02f2`，资源与GMP身份均随每组manifest冻结。CUDA点DLL SHA256=`d95874567be55ec901ca862d32346f5ab9de010050aaa635bbd741759487e27c`。0.54.1/CUDA0.54.0身份见上文，未替换生产目录。

## 正确性与适用范围

- 0.54.3默认设备点、显式CPU点、AMD各23项生产门禁通过；包含自动计划的实调用、N607→M1214提升和手动锁定。输入、D/P、强制校验与错误传播未放宽，具体计数及范围见[自动点执行计划](#自动点执行计划)。
- 0.54.2在RTX4060 Laptop默认设备点、显式CPU点、AMD gfx1150各通过18项生产门禁，共54项。新增窗口末尾短批、非整G批请求对齐、batch额度裁剪、非法窗口拒绝；对照要求NTT计数和原始点GMP覆盖完全相同。正式证据在`opencl_concurrency_20261029/verification_window_{gpu1_final,cpu_gpu1,amd}/`。AMD仍保留宽位CPU点资格限制。
- 最终0.54.1二进制在RTX4060 Laptop、AMD gfx1150各通过14项生产门禁：正因子257、普通NF、梅森/余因子承载、非法承载、预算、checksum、基本D、plan-only、队列停止与重排续跑等。
- RTX4060 Laptop显式CPU点模式另通过相同14项生产门禁，环境覆盖与证据保存在 `verification_cpu_gpu1/`。
- 新增种子非单位回归使用N=257000771、sigma6、B1=20、B2=300、D46。测试要求到达giant阶段、得到257且尚未下降，确实覆盖[D]Q种子分母路径，不能以baby阶段命中替代。
- 两卡NTT检查各包含84810个域算术值、8982512个roundtrip word与同量卷积word、235个门禁；覆盖程序键的源码/选项、context隔离与缓存关闭。GPU1另通过普通/梅森S4至16384-bit、2820门禁，树代数90642 words/7小曲线，以及8191-bit显式group32点运算16522 words。
- AMD1939/7995-bit真实Edwards save通过CPU点回退加OpenCL多项式NF，并使用原始[iD]Q校验重标定后的CPU点；没有取消宽位AMD设备点资格限制。
- 上文1200MHz的48条NF与有限正命中不代表全部曲线、Intel、其他驱动、设备丢失、长队列或大B2资格；其最大NTT为2^19。本节1500MHz研究另覆盖P2880／2²³，但仍未形成生产大B2性能矩阵。CUDA默认路径未修改，OpenCL尚未接入Auto B2/tune、自动承载比较或release自动拆包。

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

按[后续实现优先级](#后续实现优先级)推进点执行计划、giant chain、CPU分流及跨曲线批处理、NTT提交窗口。生产大B2、16384-bit长标量、更多厂商及设备丢失恢复仍需验证；不能从已测小任务外推默认并发度或全卡显存边界。[研究草稿](../development/OPENCL_STAGE2_CONCURRENCY_DRAFT.md)保留待用户处理。
