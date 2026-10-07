# Stage1 PRAC Montgomery 常量实验（2026-10-07）

状态：正确性门禁、96次生产配对计时、四份管理员NCU及独立审计全部完成。本阶段按用户要求结题；候选保持显式开启。

## 1. 范围与基线

基线提交 `f194874`。GPU1 为 RTX4060 Laptop、24SM、sm89；N=`2^4423-1`，容器4608、TPI16、TPB128、sigma26、param0。原 `single-compact/cap168` 的共享 DBL、私有 disjoint xADD 和归一化 Montgomery 运算不变。本轮检验常量传播能否减少依赖乘法及寄存器使用。

N4423默认使用TPI16；此前TPI32是 `ECM_STAGE1_TPI=32` 的显式对照，不是生产默认。TPB当前128，代码记录2026-09-25由256调整：[默认定义](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_kernel.h:104)。寄存器上限不自动改变TPB；16/32使用相应曲线数匹配提交grid，也不等于实测驻留blocks/SM一致。

exe SHA256：`6d67460aa291e8bf4aca5a440312e6de8719860a1ababfac7b9dc48a602a2854`；新增常量object：`8ecb5cd49654cec66beb12776ab358df2ec7003dbe18467748d6a0e989efd798`。46项Stage1源码、工具和CGBN核心SHA冻结在本地 `docs/data/stage1_prac_constants_timing_manifest_20261007.json`。根目录data和docs/data继续被排除，不提交原始实验数据。

## 2. 同二进制策略与原文件

设置 `ECM_PRAC_CONSTANTS=none|runtime|np0|m4423`，后三种只允许所选容器4608/TPI16、`ECM_PRAC_VARIANT=single-compact`、`ECM_PRAC_REG_TARGET=168`。

| 策略 | MODE | N | np0 | 启用条件 |
| --- | --- | --- | --- | --- |
| none | 13 | 运行时加载 | 运行时参数 | 原single-compact/cap168 |
| runtime | 15 | 运行时加载 | 运行时参数 | 独立TU中的同主体对照 |
| np0 | 16 | 运行时加载 | 编译期1 | `N mod 2^32 = 0xffffffff` |
| m4423 | 17 | 按lane构造固定M4423 | 编译期1 | `N = 2^4423-1` |

代码入口：

- [常量策略与分派声明](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_constants.h:1)。
- [独立CUDA主体](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_constants.cu:12)，[N的word构造](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_constants.cu:23)，[np0固定](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_constants.cu:33)，[工厂入口](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_constants.cu:48)。
- [host解析与严格输入检查](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_host.cuh:110)，[dispatcher拒绝不支持的TPI](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_kernels.cu:14)。
- [原Montgomery常数计算](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1.cu:330)保留，日志报告实际np0；不绕过原INIT/EXPORT。
- [窗口边界量统计](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_window.cuh:116)接收5或6个bn，反映固定N后少一次N加载。

新TU与host、dispatcher共三TU并行编译：13.0/17.8/131.2秒，关键路径131.2秒，最后仅链接。实际目标为89-real、无嵌入PTX、压缩fatbin；缓存中的通用CMAKE_CUDA_ARCHITECTURES=75不代表实际CUDA编译目标。原normal TPI16对象、single-add对象、公共PRAC头和私有数学头SHA均不变。

## 3. 数学与布局约束

对奇数N，`np0 = -N^-1 mod 2^32`。低word全1时np0恰为1；此条件也适用于部分非梅森数，因而np0策略不要求纯梅森数。m4423必须另外满足4423位且N+1只有一个置位，避免对同位宽的其他数使用错误N。

实际sm89路径是CGBN XMP_WMAD软件算术：[架构选择](D:/code/MPA-OpenCl/cgbn/include/cgbn/cgbn.h:68)，[核心选择](D:/code/MPA-OpenCl/cgbn/include/cgbn/core/core.cu:321)。两个q位置含 `shuffle(accumulator)*np0`：[WMAD第一次q](D:/code/MPA-OpenCl/cgbn/include/cgbn/core/core_mont_wmad.cu:75)、[第二次q](D:/code/MPA-OpenCl/cgbn/include/cgbn/core/core_mont_wmad.cu:114)。固定1可消除该标量乘法，但不能由源码推出所有Q*N的MAC都会消失。

CGBN load映射是 `word = lane*LIMBS + limb`：[原load实现](D:/code/MPA-OpenCl/cgbn/include/cgbn/impl_cuda.cu:1376)。TPI16/4608每线程9word，M4423 word0..137全1、word138为0x7f、其余零。lane0..14全1；lane15为`FFFFFFFF,FFFFFFFF,FFFFFFFF,7F,0,0,0,0,0`。只能在初始化时按lane设值，后续CGBN shuffle仍由整个线程组一致参与。

Montgomery R仍为`2^4608`，原7bn曲线数组、曲线构造、PRAC计划、INIT/EXPORT和checkpoint ABI不变。xADD仍4M+2S，DBL仍3M+2S，计划成本仍 `W=6A+5D`。不同策略导出的完整X/Z也逐字节相同。

## 4. 可量化的计算量、内存与传输量

令`B=4608`、`T=16`、`L=B/(32T)=9`、`S=T*L=144`、C为曲线数、P为计划prime记录数、K为kernel launch数。W由B1和torsion决定，所有策略同一输入的W相同。

- WMAD主MAC源码求值代理：每次模乘每curve `4S²=82944`，每批 `82944*C*W`。不是退休SASS、实测周期或包含归一化的总工作量。
- 原q中的np0标量乘法源码求值：每线程每次模乘S项；每curve `T*S=2304`项；每批 `2304*C*W`。固定np0为1消除此源码乘法，不能用2304/82944直接预测时间收益。
- data显存及host向量各 `7*C*B/8` bytes，C768/1536/2304分别2.953125/5.90625/8.859375MiB；本轮均不缩小这些分配。
- GPU计划 `16P` bytes，磁盘cache为`64+16P`。B1=10m时P=664,579、W=128,750,918，GPU计划10.140671MiB；B1=260m时P=14,195,860、W=3,369,476,895，GPU计划216.611633MiB。固定N不改变计划生成或传输。
- 初始H2D至少为data与control，即 `7*C*B/8 + 16P`；正常完成或一次完整checkpoint的data D2H为 `7*C*B/8`。临时curve构造、exponent缓冲等不包含在此小计。
- kernel边界逻辑读写：none/runtime/np0为 `6*C*B/8*K`；m4423为`5*C*B/8*K`，该项减少1/6。这里是按源码数组访问计算的逻辑量，不是DRAM或PCIe流量。
- 固定窗口额外seed显存 `7*C*B/8`，初始D2D同量；每轮restore同量。普通生产前缀没有此窗口seed。m4423并未减少restore和初始H2D。

模块自己的容量、临时峰值和累计流量不能直接相加为进程峰值。静态少一次N加载也不能直接代表消除了数据准备瓶颈。

## 5. 静态机器码

| 策略 | 实际寄存器/线程 | STACK bytes | LOCAL资源bytes | 静态指令数 | text bytes | LDG条数 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none / 原MODE13 | 162 | 48 | 0 | 27784 | 444544 | 39 |
| runtime / MODE15 | 162 | 48 | 0 | 27784 | 444544 | 39 |
| np0 / MODE16 | 162 | 48 | 0 | 27248 | 435968 | 39 |
| m4423 / MODE17 | 156 | 48 | 0 | 27960 | 447360 | 30 |

新runtime与原MODE13的55,571行完整SASS，在仅归一化入口Function名称后完全一致，包含调度编码和所有指令行；资源文本也相同。归一化SHA为`765f05eede819641846d32be2e61790e7e8109f0b582aa92e1cf457a9bfa2bf7`。这排除了“移入独立TU使runtime主体变化”的解释，但仍保留同机计时对照。

np0相对runtime减少536条静态指令（1.929%），IMAD从12452降为12182。m4423相对np0增加712条，寄存器少6，LDG少9。三新策略均含静态LDL7/STL5，与原runtime相同；它们不能单独证明动态spill流量或时间。没有单独导出的callee。

计数包括entry内嵌的本地callee，不是动态退休指令，也不是指令cache工作集。硬件分配粒度和最终blocks/SM须由native occupancy及NCU核对；门禁中三新策略与none的容量均为3。

## 6. 正确性证据

独立[word REDC模型](D:/code/MPA-OpenCl/tools/test/test_prac_constants.py:10)通过414组正例，含非梅森低word全1；144word整数重建M4423准确。强行对同位宽N−2使用np0=1的64个反例全部错误，证明严格输入检查必要。此为独立模型，不替代原生CUDA证明。

[完整Q门禁](D:/code/MPA-OpenCl/tools/test/test_cuda_prac.py:127)分别测试runtime/np0/m4423，每个27类、3568个Q，共10704；其中实际候选主案例及恢复共3552Q，其余为ladder/resident及边界对照。覆盖lcm/choose12、大sigma、恢复/损坏恢复、结果和target限制；不能称10704个全由新候选执行。

新增[原生常量门禁](D:/code/MPA-OpenCl/tools/test/test_cuda_prac_constants.py:52)：

- 72普通窗口覆盖B1=1000的lcm/choose12及生产B1=10m/260m三个位置、count32/chunk7和32；加4个checkpoint窗口，共76结果/608Q，均与独立整数ladder比较。
- 完整CSV跨策略、切片共63次逐字节比较；其中60次在门禁内计数，外部统一CSV审计另含3次checkpoint窗口比较。它们与Q比较重叠，不累加为独立案例。
- checkpoint策略轮换none→runtime→np0→m4423→none：窗口保持checkpoint字节不变，随后恢复产生32个正确完整Q。
- 非梅森 `N=2^4423-1-5*2^32`、B1=2：none/runtime/np0共24个原生Q与CPU一致，证明np0策略支持严格条件内的非梅森数。
- 14个拒绝用例：非法策略、TPI32、其他容器、错误寄存器/variant、resident搭配、错误np0与同位宽非梅森固定N；均在PRAC计划与曲线执行前拒绝，不写save/checkpoint。

首轮门禁在最后一个resident拒绝用例上因预期报错文字与更早入口的实际文字不同而终止；修正断言后完整重跑成功。未因此修改CUDA算术。原始失败和成功日志均保留。

## 7. 生产计时与NCU

[配对脚本](D:/code/MPA-OpenCl/tools/bench/bench_stage1_prac_constants.py:17)使用同一exe、四策略、C768/1536/2304、两个反序重复：

- 固定B1=260m、tail32、chunk4/32、6秒：48样本，event与含restore的窗口wall分别统计。
- 普通前缀B1=10m/260m、target50ms、15秒、排除初始5秒：48样本，读取精确s/curve，仅checkpoint退出。
- 计时期间不编译、不导出大SASS、不运行profiler。GPU1每500ms采样；这些memory.used采样不是完整进程峰值。
- 四份管理员NCU，C2304/tail32/chunk32，计时结束后采集。[profile入口](D:/code/MPA-OpenCl/tools/bench/profile_cuda_stage1.py:38)支持显式常量策略。

### 已完成的固定窗口

48样本的独立矩阵、冻结SHA、缓存、正常退出、实际几何及原始日志审计通过；event与wall投影按W、work、measured、C重新计算。下表为两个反序重复的wall投影中位数，单位s/curve；收益为曲线吞吐相对同C/chunk的none，公式`T_none/T_candidate-1`。

| C | chunk | none | runtime | np0 | m4423 | np0吞吐收益 | m4423吞吐收益 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 768 | 4 | 146.129469 | 146.298784 | 142.326078 | 140.299041 | +2.672% | +4.156% |
| 768 | 32 | 147.817061 | 147.752132 | 143.280993 | 144.615871 | +3.166% | +2.214% |
| 1536 | 4 | 131.520207 | 131.390317 | 128.054287 | 127.998894 | +2.707% | +2.751% |
| 1536 | 32 | 134.994748 | 134.948845 | 130.635257 | 134.210650 | +3.337% | +0.584% |
| 2304 | 4 | 130.205156 | 130.215603 | 126.974845 | 128.397637 | +2.544% | +1.408% |
| 2304 | 32 | 134.582336 | 134.469077 | 129.965919 | 134.116520 | +3.552% | +0.347% |

runtime与none墙钟吞吐差为−0.116%..+0.099%，远小于np0收益。np0各组两个单独配对重复均为正。m4423在小批量短chunk有收益，但C2304两种chunk均弱于np0，不能由少6寄存器/少1次N加载推出最高吞吐。

固定矩阵GPU1忙采样SM时钟均1800MHz，温度55..70°C；device memory.used采样最大331MiB。该容量包含设备其他分配，不是本进程峰值。

### 普通生产前缀

48样本同样完成矩阵、SHA、缓存、几何及原始日志审计。均正常以checkpoint-only退出，无最终save、无强制终止；精确进度字段重建每个样本的末尾5秒中位数，再对两个反序重复取中位数。以下是kernel工作率投影，不是完整运行墙钟。

| B1 | C | none s/curve | runtime | np0 | m4423 | np0吞吐收益 | m4423吞吐收益 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10m | 768 | 5.542133 | 5.541235 | 5.386029 | 5.326108 | +2.898% | +4.056% |
| 10m | 1536 | 5.006798 | 5.007014 | 4.876304 | 4.866585 | +2.676% | +2.881% |
| 10m | 2304 | 4.962103 | 4.962795 | 4.831269 | 4.823346 | +2.708% | +2.877% |
| 260m | 768 | 144.945335 | 144.964380 | 140.925509 | 139.523503 | +2.852% | +3.886% |
| 260m | 1536 | 131.017150 | 131.022937 | 127.636582 | 127.406503 | +2.649% | +2.834% |
| 260m | 2304 | 129.847945 | 129.874732 | 126.492224 | 126.287030 | +2.653% | +2.820% |

runtime相对none为−0.021%..+0.016%，与完整机器码一致的对照结论吻合。np0的12个单独配对重复均为正，C2304两B1分别2.700%/2.717%及2.672%/2.633%。m4423在普通前缀下比np0快0.16%..1.13%；最佳C2304只有0.164%/0.162%的额外吞吐，而固定tail/chunk32反而比np0低3.095%。两类采样反映不同prime链和切片，不能混合成同一墙钟收益。

C2304当前普通前缀最小投影为m4423的4.823346/126.287030 s/curve；稳健的np0为4.831269/126.492224。相对同轮none，np0耗时投影下降2.637%/2.584%，m4423下降2.796%/2.742%；吞吐百分比和耗时百分比不可混写。

普通前缀忙采样SM时钟均1800MHz，温度62..73°C，device memory.used采样最大321MiB。完整曲线、保存点生成总墙钟和Auto B2的完整T1未在本轮认证。

### 管理员NCU

四份采集和CSV导出均exit0，实际均20passes。独立核对exe SHA、device1、C2304/grid288/TPB128、tail32/chunk32、输入prime范围及真实被采集的MODE13/15/16/17；profile时没有生产计时。表中issue单位为peak百分点，stall为`per_issue_active.ratio`。

| 策略 | 实际/分配寄存器 | 容量blocks/SM | 活跃warps | eligible warps | issue% | wait | no_instruction | short_scoreboard |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 162/168 | 3 | 11.341029 | 0.490965 | 33.872792 | 4.186272 | 0.554378 | 0.326104 |
| runtime | 162/168 | 3 | 11.383417 | 0.493293 | 34.013534 | 4.192645 | 0.540634 | 0.326050 |
| np0 | 162/168 | 3 | 11.316423 | 0.487126 | 33.794765 | 4.242949 | 0.479975 | 0.320969 |
| m4423 | 156/160 | 3 | 11.409693 | 0.510118 | 34.741718 | 3.851981 | 0.926520 | 0.066077 |

np0与runtime的活跃warp和issue相近，没有通过增加驻留容量或warp数量实现收益；结合完整SASS删去乘法与反序计时，支持减少算术工作量的解释。wait比值没有下降，不能把本轮收益称为已测出的wait stall时间缩短。

m4423分配168→160仍只容纳3blocks/SM，eligible、issue和short_scoreboard改善，但no_instruction比np0显著增大；它不是直接的指令cache miss指标，也不能由这些比值推出唯一因果。固定tail/chunk32实际慢于np0，说明较高issue和较少寄存器并不自动意味着每curve更快。

四份local load sectors均0，local store sectors为31900/32072/32640/32700。累计`sector*32`代理0.973511/0.978760/0.996094/0.997925MiB，未出现cap128实验那种大量local读写；它不是显存容量、DRAM或PCIe流量，不乘20passes。DRAM throughput为0.000049%..0.000112% of peak，L1TEX约25.22%..26.03%，该窗口没有DRAM带宽饱和证据。不能据此排除完整流程中的CPU构造/计划加载瓶颈。所有stall比值均非墙钟时长占比。

## 8. 复现

```powershell
$env:ECM_GPU_STAGE1_ALGO='prac'
$env:ECM_PRAC_VARIANT='single-compact'
$env:ECM_PRAC_REG_TARGET='168'
$env:ECM_STAGE1_TPI='16'
$env:ECM_PRAC_CONSTANTS='np0' # none/runtime/np0/m4423
$env:ECM_PRAC_TARGET_MS='50'

python tools/test/test_prac_constants.py --output docs/data/constants_model.json
python tools/test/test_cuda_prac_constants.py --exe build_cuda_cmake/prac/ecm_cuda.exe `
  --cache build_cuda_cmake/prac --device 1 --output docs/data/constants_gate
python tools/bench/bench_stage1_prac_constants.py --exe build_cuda_cmake/prac/ecm_cuda.exe `
  --mode prefix --b1 10000000 260000000 --curves 768 1536 2304 `
  --seconds 15 --warmup 5 --target-ms 50 --repeats 2 --device 1 `
  --exp-cache build_cuda_cmake/prac --output docs/data/constants_prefix
```

仅本机实验条件内使用；默认常量策略仍none。按用户要求，本轮配对结果、NCU、文档和提交完成后，结束本阶段Stage1优化；不再启动下一轮。

## 9. 结题与采用建议

本轮优先推荐GPU1/N4423的显式 `TPI16/TPB128/C2304/single-compact/cap168/target50ms/np0`：两B1普通前缀吞吐提高2.65%..2.71%，固定窗口所有配对组也稳定为正，且np0输入检查比固定M4423更通用。其他GPU、位宽或容器没有本轮性能证据，不自动推广。

固定m4423保留为针对精确N的可选实验。普通前缀的当前最小投影为4.823346/126.287030 s/curve，比np0只提高约0.16%吞吐；tail32/chunk32则回退约3.1%。不把它作为普遍最优策略，也不修改生产默认。

本阶段已有证据支持的选择：

- PRAC继续使用经过完整Q/窗口/恢复门禁的Prime95链及归一化算术。
- [批量实验](D:/code/MPA-OpenCl/docs/ECM_STAGE1_PRAC_ALIGNED_BATCH_20261007.md:1)支持本机TPI16/C2304；相对C1536约1%提升。
- [显式TPI32对照](D:/code/MPA-OpenCl/docs/ECM_STAGE1_PRAC_SHARED_DBL_TPI32_20261007.md:1)在相同提交grid下未胜过TPI16，不据此修改大位宽的默认TPI32。
- [cap128实验](D:/code/MPA-OpenCl/docs/ECM_STAGE1_PRAC_SHARED_DBL_CAP128_20261007.md:1)在C768有局部改善，但最佳吞吐仍cap168；更多驻留warp不能抵消spill成本。
- [输出Z复用](D:/code/MPA-OpenCl/docs/ECM_STAGE1_PRAC_ADD_OUTPUTZ_20261007.md:1)让源码更明确，但完整SASS未变，不算性能收益。
- 本轮np0常量传播提供约2.7%的可重复新增收益，保留严格输入检查、同二进制对照及复现工具。

按用户要求在此收口。已验证的是当前实现、硬件和采样条件下的上述选择；没有证明所有Stage1算法已达到理论性能上限。

本地原始证据索引：`docs/data/stage1_prac_constants_{model,static_audit,timing_manifest}_20261007.json`，三个`*_full_q_gate_20261007`目录，`*_windows_gate_v2_20261007`及完整CSV审计，两个`*_pairs_{fixed,prefix}_20261007`目录的summary/audit/telemetry，四个`*_ncu_{none,runtime,np0,m4423}_20261007`目录及`*_ncu_analysis_20261007.json`。源码、工具、结论与开发日志进入Git；原始数据保留本机。
