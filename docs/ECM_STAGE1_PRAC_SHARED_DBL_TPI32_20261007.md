# Stage1 param0：共享 DBL 的 TPI16/TPI32 配对实验（2026-10-07）

## 1. 范围与版本

基线`9f5d485`，即上一轮v4共享seed/loop DBL。本轮只新增4608-bit容器的显式TPI32 single-compact MODE12/13实例及分派，使用同一共享DBL、单点ADD、normalized Montgomery算术。没有改变默认4423-bit→4608/TPI16、TPB128、算法或切片目标。

- before冻结exe SHA256：`fe2b74c7f6b17a401c256b351e55004382bc67ab4923e59356c3ea5dbda90ef5`。
- after exe SHA256：`2a7db11e451bcda16640a3d30d9582edce814aed7a7da93bcd9e3d2a28989232`。
- 公共算术头仍为`67dc71128495378991dd750d73f7d8226ebedd255e267abd435dc6331a03e0a6`。
- 私有共享helper头为`a0f3b9ae91474384d03b8ac644c509fc716236f205798b5a4e806d62a592bafc`；本轮仅澄清seed定义B、普通ADD定义T的注释，数学主体不变。
- before快照在本地`build_cuda_cmake/prac/after_shared_dbl_20261007/`；原始证据位于忽略的`docs/data/`，报告保留可复现实验口径。

[TPI32实例](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_single_add.cu:28)仅支持4608/single-compact/register255或168；[分派](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_kernels.cu:17)必须显式`ECM_STAGE1_TPI=32`。默认requested_tpi=0仍选旧TPI16函数。旧single-add及compact/outline-add不扩大支持范围；不支持的容器仍拒绝，不静默垫高容器。

## 2. 假设与反馈

事先按以下顺序提出可证伪假设：

1. TPI32降低单线程大整数负担，若寄存器/指令代价下降足够，则相同提交grid吞吐提高。
2. 4608/TPI32存在部分limb分组，填充和shuffle可能抵消收益；若资源下降而吞吐不升，不能仅据寄存器推荐TPI32。
3. 两内核驻留容量可能不同；若门槛/尾波重要，收益随三个批量档改变，同grid不代表实际blocks/SM相同。

before原生CLI在N4423/TPI32/single-compact下确实报requested policy unavailable。本轮先编译反馈和CPU模型，再原生Q/窗口门禁，最后串行计时/管理员NCU；不拿门禁的小批量投影当作生产吞吐。

## 3. 编译与静态证据

只重编专用候选和dispatcher两个TU：分别194.4/9.5秒，两个并发critical path194.4秒；最终CMake仅链接。host与normal TPI16对象SHA和before快照一致。

- 原TPI16 MODE10/11/12/13完整SASS（含调度编码）及资源记录与v4全部相同。
- 新TPI32 natural MODE12：14,528条静态指令、232,448bytes text、111寄存器、stack64、spill0/0。
- 新TPI32 cap168 MODE13：14,072条静态指令、225,152bytes text、109寄存器、stack64、spill0/0。
- TPI16共享候选分别27,752/27,784条、444,032/444,544bytes、实际161/162寄存器、stack48、spill0/0。

NCU确认TPI32硬件按112寄存器/线程分配，TPB128的寄存器限制容量为`floor(65536/(128*112))=4blocks/SM`；TPI16按168分配、容量3。无新增outlined callee/设备调用ABI。静态text/指令不是动态执行量或instruction cache占用，不能据此宣称运行时间减半。

证据为`stage1_shared_dbl_tpi32_sass{16,32}_20261007/`，16目录的完整comparison确认四函数不变。

## 4. 数学与正确性

共享数学模型17,066输入链×六模型=102,396求值；264,538角色步骤、中间/终结XZ、运算计数和独立ladder一致。normalized Montgomery模型804正例、4,824公共别名对照通过；违规差分点别名仍786错误，归一化反例保留。CPU模型不证明TPI32物理实现。

原生GPU1完整Q/save：natural总184对照、cap168/C192/50ms总1,840对照，各27类；其中强制TPI32候选的主案例/候选checkpoint恢复分别56/608 Q。其余对照包含ladder/resident与公共边界门禁，不能将所有数字都算作候选独立覆盖。lcm/choose12、64-bit sigma、checkpoint恢复/损坏重算均通过。

窗口1,648 Q、16恢复Q、22拒绝、206条结果全部通过；CPU oracle184miss/1,464hit，每份GPU输出仍检查。158项同TPI完整XZ逐字节一致，含60跨切片；另73项跨TPI16/32的同标量/同C/同chunk输出也逐字节一致，不与158重复归类。

跨TPI审计初次假定72项，实际还有一项独立checkpoint窗口，共73；读取实际分组后修正数量并逐项核对字节，未重跑GPU。完整门禁的数字只表示本批覆盖，不等于不同输入的独立数量。

[门禁新增选项](D:/code/MPA-OpenCl/tools/test/test_cuda_prac_windows.py:129)明确`--single-compact-tpi32`依赖`--single-compact`，含N4423小B1及10m/260m prefix/middle/tail，chunk7/32，以及TPI32候选checkpoint隔离后baseline32恢复。不支持的宽度/其他variant继续拒绝。本地原始证据为`stage1_shared_dbl_tpi32_{natural,cap168,windows}_q_gate_20261007/`及`stage1_shared_dbl_tpi32_gates_audit_20261007.json`。

## 5. 性能实验

GPU1 RTX4060 Laptop/24SM/sm89；N4423/容器4608/TPB128/sigma26/lcm。TPI16 C384/768/1536配TPI32 C192/384/768，提交grid均48/96/192。TPI是每曲线协作线程数，TPB是每block线程数，寄存器限制不会自动修改TPB。

[配对工具](D:/code/MPA-OpenCl/tools/bench/bench_stage1_prac_tpi_pairs.py:31)比较shared16、shared16cap168、shared32、shared32cap168和baseline32/natural五配置；逐样本检查exe前后SHA、缓存命中、容器/TPI/C/grid。两重复反序，GPU任务串行，计时不编译、不导出大SASS、不运行NCU。

- 固定窗口：B1=260m tail32，五配置×三grid×chunk4/32×两反序=60样本；6秒采样、两个排除的warmup轮。
- 普通前缀：B1=10m/260m，五配置×三grid×两反序=60样本；目标50ms、15秒采样、warmup5秒、最后5秒精确s/curve中位。

窗口事件/墙钟投影为`T_curve=W_full*t/(W_window*rounds*C)`，统一时间单位秒；吞吐`curves/s=1/T_curve`。墙钟含D2D恢复、launch/synchronize，事件只计内核。不同C不比较整个batch完成时间；不将两种时间口径混合。

如果TPI16容量3/TPI32容量4，在TPB128下满驻留曲线数分别3×8=24、4×4=16每SM。warp容量却分别12/16；“更多驻留warp”和“更多并发曲线”不同。实际尾波和动态驻留须用计时/NCU解释，不能从容量公式直接算时间。

全部120计时样本及四份管理员NCU已完成并审计。结论仅限短时投影，不是完整生产批次墙钟或Auto B2 T1。

### 5.1 固定窗口：60样本完成

以下为两次反序的墙钟投影中位，s/curve；每行顺序shared16 / shared16cap168 / shared32 / shared32cap168 / baseline32。C为TPI16曲线数，TPI32为C/2。

- C384、chunk4：146.473070 / 147.522002 / 236.941311 / 228.775448 / 209.584645。
- C384、chunk32：145.090585 / 146.292805 / 234.330600 / 225.630765 / 207.051153。
- C768、chunk4：144.978926 / 145.631137 / 193.214395 / 194.155142 / 187.690211。
- C768、chunk32：147.260082 / 147.775164 / 191.765841 / 192.685076 / 186.676533。
- C1536、chunk4：132.418366 / **131.310077** / 186.250489 / 186.309109 / 179.056244。
- C1536、chunk32：135.804109 / **134.979617** / 185.480340 / 185.532660 / 179.285357。

cap168匹配比较TPI32/TPI16：C384短/长吞吐减少35.517%/35.163%；C768减少24.992%/23.307%；C1536减少**29.520%/27.248%**。TPI32自然候选同样全面回退，且32-baseline比32-shared更快；本轮新候选未改善最佳吞吐。静态text/寄存器减少不能替代吞吐验证。

严格审计60/60矩阵，无缺项/重复，SHA/几何/缓存全部通过；每条seed和边界逻辑bytes符合实际C及chunk，checkpoint不改/无最终save。原始summary保留事件口径及重复范围；本地`stage1_shared_dbl_tpi32_pairs_fixed_20261007/audit.json`。

### 5.2 CGBN 分组与填充

[CGBN定义](D:/code/MPA-OpenCl/cgbn/include/cgbn/cgbn_cuda.h:107)为`LIMBS=(bits/32+TPI−1)/TPI`。4608-bit含144个有效32-bit limb；TPI16每线程9limb，分组槽位16×9=144；TPI32每线程5limb，分组槽位32×5=160，有16个填充槽位，槽位数多11.11%。这不是N变成5120-bit或每curve显式数组变大，也不能把11.11%直接当作MAC/时间增加。

[bn2mont转换](D:/code/MPA-OpenCl/cgbn/include/cgbn/impl_cuda.cu:1001)明确对填充值使用更大的R：TPI16的R为`2^4608`，TPI32为`2^5120`。N和显式容器仍为原值，init/export分别做相应转换；跨TPI导出的完整XZ相同，不表示内核中的Montgomery表示字节相同。

[架构选择](D:/code/MPA-OpenCl/cgbn/include/cgbn/cgbn.h:67)在本机sm89使用XMP_WMAD；编译命令未覆盖该宏，[core包含分派](D:/code/MPA-OpenCl/cgbn/include/cgbn/core/core.cu:321)进入core_mont_wmad，而不是因SASS有IMAD指令就进入core_mont_imad。WMAD是CGBN的软件算术核心名，不是Tensor Core。

[WMAD Montgomery主循环](D:/code/MPA-OpenCl/cgbn/include/cgbn/core/core_mont_wmad.cu:40)以`thread+=2`、`word<2LIMBS; word+=2`处理两limb。每对包含八条偶/奇madlo/madhi链，合计8L项（L为5/9时也包含显式尾项）。令`t=TPI`、`L=ceil(B/(32t))`、`S=tL`：主MAC求值数每线程为`(t/2)*L*8L=4tL²`，每curve线程合计为`4(tL)²=4S²`；另有q乘np0、加法/进位、shuffle及最后修正。这是源码运算计数，不能直接当作退休SASS条数或周期。

B4608时TPI16的`t=16,L=9,S=144`，主MAC每线程5,184、每curve82,944；TPI32的`t=32,L=5,S=160`，每线程3,200、每curve102,400。单线程下降，但每curve合计增加**23.4568%**。这比只比较“槽位多11.11%”更能说明潜在代价；动态效果须以计时/SASS/NCU核对，不将公式等同运行时间。初次读取IMAD源码后进一步核对宏选项，确认本机实际WMAD；报告采用实际核心的两limb推导。

### 5.3 普通前缀：60样本完成

以下是运行级投影中位，s/curve；每行仍按shared16 / shared16cap168 / shared32 / shared32cap168 / baseline32，目标50ms：

- B1=10m、C384：5.540462 / 5.586675 / 8.943841 / 8.614751 / 7.902748。
- B1=10m、C768：5.509636 / 5.537555 / 7.327128 / 7.363201 / 7.121714。
- B1=10m、C1536：5.042995 / **5.004845** / 7.092385 / 7.093370 / 6.820933。
- B1=260m、C384：145.069629 / 146.264026 / 234.101353 / 225.533937 / 206.871293。
- B1=260m、C768：144.224895 / 144.929993 / 191.847348 / 192.759390 / 186.466939。
- B1=260m、C1536：131.979774 / **130.985920** / 185.596363 / 185.657955 / 178.489207。

cap168匹配grid：10m三档TPI32吞吐分别减少35.150%/24.794%/**29.443%**；260m减少35.148%/24.813%/**29.448%**。自然策略也全部回退；32-baseline比32共享候选快，但仍比本批TPI16最佳慢。最佳TPI16数值与上一轮约0.02%内相近，本轮没有新的最佳吞吐提升。

全部120样本矩阵重建无遗漏/重复；核对每条SHA、缓存、真实TPI/容器/C/grid和原始日志。普通前缀全部正常exit1、sample limit/checkpoint-only，无强制终止/最终save；固定窗口checkpoint不改。35项Stage1源/工具SHA及五份实际CGBN核心文件SHA留档，合计40项；计时期间没有修改源码/二进制。

固定GPU采样875点/714忙点，前缀1,874点/1,758忙点（利用率≥70%）；忙时SM均1800MHz。温度分别49..68°C/59..67°C，设备memory.used采样最大323/317MiB，非进程完整峰值。原始审计为`stage1_shared_dbl_tpi32_timings_audit_20261007.json`。

### 5.4 管理员NCU：四份采集完成

四份全部管理员exit0、CSV导出exit0；同tail32/grid192/TPB128/Device1/4608、TPI16 C1536和TPI32 C768，逐项核对exe SHA、输入、实际demangled MODE、三维尺寸与完整first/prime/work/warmup。TPI16重放19passes，三个TPI32均18passes；只读审计初次沿用19的假定拒绝了实际日志，核对真实次数后更正，未重新采集。重放时间不用作吞吐，计数也不乘passes。

下面依次为shared16cap168 / shared32 / shared32cap168 / baseline32：

- 实际寄存器162 / 111 / 109 / 109；硬件分配168 / 112 / 112 / 112；寄存器限制容量3 / 4 / 4 / 4blocks/SM。
- 平均活跃warp每SM：11.034719 / 14.202071 / 14.128952 / 14.073566。
- eligible warps/scheduler/cycle：0.483320 / 0.708116 / 0.711901 / 0.777841。
- issue active：33.842780 / 39.430806 / 38.706905 / 40.592904%。
- wait/issue-active：4.159638 / 3.691673 / 3.595715 / 3.468028。
- no_instruction：0.438963 / 0.066133 / 0.070625 / 0.170232。
- short_scoreboard：0.326032 / 1.200019 / 1.069984 / 0.402002；math_pipe_throttle：0.893846 / 1.266283 / 1.458045 / 1.504833。
- local load sectors全部0，store21,676 / 21,308 / 21,252 / 21,488；累计字节代理0.661499 / 0.650269 / 0.648560 / 0.655762MiB，量级接近，无新增spill流量证据。
- DRAM throughput为0.000044%..0.000210% of peak；没有该恢复窗口的DRAM吞吐饱和证据。

TPI32的更多活跃warp、更高issue和更低wait/no_instruction没有转化为更高曲线吞吐。TPI16每warp两个curve槽位、TPI32一个；平均活跃warp对应的curve槽位代理约22.07与14.13，不能把warp数直接当作并行曲线数，也不表示每个槽位每周期都执行。

源码主MAC求值增加23.46%、curve槽位减少、short_scoreboard增加，均提供回退线索；未采集足够的动态算术指令归因数据，不能宣称某一项解释全部29%吞吐下降。short_scoreboard包括短延迟依赖，未证明全来自shuffle；no_instruction也不能直接当作I-cache miss。stall比值不是墙钟占比，local代理不是VRAM/DRAM/PCIe。

TPI32 baseline和cap候选都是109实际寄存器/112分配/容量4，但baseline仍更快；说明本位宽/TPI下的点角色/DBL复用与编译调度效应不能照搬TPI16的收益。本地证据为`stage1_shared_dbl_tpi32_ncu_{shared16cap168,shared32,shared32cap168,baseline32}_20261007/`和`stage1_shared_dbl_tpi32_ncu_analysis_20261007.json`。

### 5.5 采用范围与下一项

保留显式TPI32候选及配对工具用于复现，生产推荐仍为已测TPI16/cap168/50ms；本轮没有提高最佳吞吐，默认配置不改。该结论限GPU1/N4423/本矩阵，不推广至默认TPI32的9216及以上容器。

下一项优先在同一TPI16共享DBL算法下测试128寄存器上限：硬件分配128时TPB128有4blocks/SM，比当前3多一档，可将更多warp用于隐藏依赖延迟。预期会增加spill，须以编译、原生门禁、同二进制A/B和NCU判断是否抵消收益，不能仅按驻留容量采纳。更长期再评估normalized PRAC专用紧凑容器和Montgomery核心，先证明边界，再减少limb/MAC。

## 6. 计算量、存储与传输

计划ADD=A、DBL=D，normalized ADD=4M+2S、DBL=3M+2S，仍为模乘等价`W=6A+5D`；同标量的A/D/W不变。每条curve容器仍4608，显式data/seed各`7*C*4608/8` bytes。配对C减半时数组容量及边界逻辑总读写量减半，是批量调整的结果，不能当作每条曲线访存减半。

依本机WMAD核心，每条curve的主循环源码limb-MAD求值量代理为`4S²*(6A+5D)`；固定tail32的W=8,053，完整B1=260m计划W=3,369,476,895。这不含q乘np0、进位/修正/shuffle和init/export，WMAD编译也可能将多个源操作融合到一条机器指令，不能直接换成周期或峰值INT吞吐。

计划仍`16P` bytes、不因TPI变。边界逻辑量`6*C*4608/8*launches`；相同窗口/chunk时每curve边界量相同，单轮总量随C减半。初始host/device数组传输也随C变化，没有新传输通道或CPU准备算法。

stack64/48是每线程静态记录，不等于spill大小或进程显存峰；两TPI候选spill均0。NCU累计local sectors×32若使用，只是cache访问字节代理，非容量/DRAM/PCIe，不乘重放pass。

## 7. 复现

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File tools/build/internal/parallel_nvcc.ps1 `
  -BuildDir build_cuda_cmake/prac -Only 'cgbn_stage1_prac_(single_add|kernels)\.cu$' -Jobs 6
python tools/test/test_cuda_prac.py --exe build_cuda_cmake/prac/ecm_cuda.exe `
  --bits 4423 --tpi 32 --registers 168 --variant single-compact --curves 192 `
  --target-ms 50 --device 1 --output docs/data/tpi32_fullq
python tools/test/test_cuda_prac_windows.py --exe build_cuda_cmake/prac/ecm_cuda.exe `
  --single-add --single-compact --single-compact-tpi32 --production `
  --production-counts 32 --production-chunks 7 32 --device 1 --output docs/data/tpi32_windowsq
python tools/bench/bench_stage1_prac_tpi_pairs.py --exe build_cuda_cmake/prac/ecm_cuda.exe `
  --mode windows --b1 260000000 --seconds 6 --exp-cache build_cuda_cmake/prac `
  --output docs/data/tpi_pairs_fixed
python tools/bench/bench_stage1_prac_tpi_pairs.py --exe build_cuda_cmake/prac/ecm_cuda.exe `
  --mode prefix --b1 10000000 260000000 --target-ms 50 --seconds 15 --warmup 5 `
  --exp-cache build_cuda_cmake/prac --output docs/data/tpi_pairs_prefix
```

默认仍TPI16；TPI32必须显式指定且记录源码/二进制SHA。下一轮覆盖exe前保留本轮快照；不恢复生产checkpoint或发布采样前缀为完整Stage1 save。
