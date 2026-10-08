# ECM Stage2：按 NTT 尺寸选择 outer 与第二次 D 标定

2026-10-04，接续 `6dd0724` 的 xADD6、驻留 D 模型和 cooperative outer v2。此阶段在 GPU1 RTX4060 Laptop 8GiB/sm89 上串行执行；测试、编译与剖析分别安排，GPU0 的外部生产任务继续运行。没有使用 Tensor Core。

## 1. 确定性的尺寸策略

`NTT_FUSE_COOP_OUTER` 提供三个模式：0为原 outer；1强制 cooperative，用 `NTT_FUSE_COOP_M` 选择5–8层；2按已测尺寸选择。模式2不使用强制模式的M参数。

模式2只在RTX4060 Laptop/sm89、t12、warp tile开启、compact scratch、原M4配置下启用表：N=2²⁴用M6，N=2²⁵..2²⁷用M8。较小或未测尺寸、不同设备/t/warp/scratch/M配置回原planner。它不在线搜索、不额外执行试算卷积，不修改数学变换或检查抽样。

该表来自同二进制纯NTT交叉测量：k24的M6优于M8，k25..27的M8优于M6，k27的M7反而慢。复测自动模式覆盖k23回退和k24..27完整稀疏卷积，全部N输出在计时外与GMP参考比较。另有88项规划边界/配置覆盖检查，以及原cooperative前向/逆向/缓存/故障门禁。

arena键包含N/k/root/t/M/cooperative/compact；warp选择复用相同表，在命中缓存时重新设置kernel属性。模式2按每个实际NTT长度选择，F/G的树层、fold和Newton可能采用不同宽度。它不能仅按ECM的模数位数或P选择一个全局M。

## 2. 工作量和容量

每个NTT pass主数组读写16N B/变换；两forward+inverse/pointwise卷积约48Np+8N B。k27/t12的原M4需要5个pass，M8+7需要3个，主数组payload少96N B，即N=2²⁷时12GiB/单slice卷积。额外shared访问和同步保持，流量下降不等于同比时间下降。

compact coarse表的长度依照每个pass的实际S；前向S=N/2^(累计M)，inverse使用相同长度集合。每方向cached coarse表约8·ΣS B，另有radix根和tile表；scratch复用最大S。k27时M4的ΣS包含N/16、N/256、N/4096、N/32768，M8+7仅N/256和N/32768。减少pass也降低表容量。

八次Stage2 A/B的arena完整计账峰值3656064320→3341512288 B，少314552032 B（约299.98MiB）：cached owned payload3519583952→3303591920 B，table293542784→77550752 B，mandatory FuseCtx136480368→37920368 B。A/B/Q主workspace仍3221225472 B（3072MiB），24次分配/8次增长/8890次复用保持。主机private commit峰的组均值7616.25→7318.25MiB。以上不是NVML总显存峰或物理RAM峰，本轮没有测NVML峰值；表与scratch减少能解释部分host/device容量变化。根owner公式仍为8W(9P+8)+48 B。

## 3. 重新标定 D

旧NTT与新NTT保留各自的经验系数，显式D优先。新模型的约束仍为精确M4423、B1=1000、B2=1e11..2011326186870、单曲线、已测设备、xADD6/current resident/batch64/chain64/check配置，以及G≥2和owner/arena预算。未支持scope回旧§56.1模型。

形状精确性与内存计算继续查询实际multiply backend：slot_bits=2S+max(1,ceil(log2m))，选择bpw满足m·sw·(2^bpw−1)²<q，N=nextpow2(2m·sw+1)。成本特征采用U₂(m)=Nlog2N·w(log2N)。w24=.9386473792217225、w25=.8672913453433596、w26=.885360571656581、w27=.9302319270266772，其他尺寸为1；比值在此次D拟合前从独立纯卷积测量冻结。它加权NTT特征，不是周期数，也不会改变精确性界/内存容量。

T₂(P)=Σh=1,2,..h<P floor((P+h)/(2h))·U₂(h+1)；V₂(k)为每次Newton扩张的2U₂。partial pair按真实共同operand长度补齐计费，passthrough不乘。顶部子树使用严格小于P的最大2次幂；P跨2¹⁷后实际tree N会跳到2²⁷。

阶段公式同前报告：baby与Plog2D、affine与P、F树/下降与T₂、G树与aT₂(P)+T₂(r)、fold与(G−1)U₂(P+1)、inverse与V₂、giant与I(6+22log2B2/64)、accum与P、glue与G拟合正系数。拟合工具拒绝把原NTT的xADD锚点混入模式2；新锚点必须具有相同exe SHA与模式2控制记录。

## 4. 测量、拟合与剖析

### 4.1 固定工作量 A/B

实验exe SHA256 `b91f7cab3e4917a8abbad67284546bb30409aaa6a0a19804768ac90e3fa71322`，编译508.9 s、link2.9 s。immutable源码与SHA保存在[manifest](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_shape_20261004/manifest.json)。Stage1 Q、sigma26/extra12、M4423、B1=1000、B2=2011326186870、D1231230/P115200/I1633592/G15、batch64/arena6300MiB、检查配置相同，只改outer模式0/2。测量和编译/剖析分开。

串行ABBA+BAAB八条、每模式四样本，stage2_full_wall原模式70.280350/67.388698/68.937862/68.645294 s；模式2为66.106584/66.783583/67.745277/67.759224 s。完整均值**68.813051→67.098667 s（−2.49136%）**，main55.397621→53.599529 s（−3.24579%），init13.415430→13.499138 s。两个顺序组的均值均改善，初始化没有改善；样本存在波动，未建立置信区间。

阶段均值：giant11.80525→11.77775 s，G树20.73050→20.27550，fold10.68650→9.67425（−9.47%），descent8.24850→8.10025，inverse2.28950→2.21350，accum.16825→.16225。以新full作分母，init约20.1%、giant17.6%、G树30.2%、fold14.4%、descent12.1%、inverse3.3%；剩余为其他准备/桥接工作。初始化已包含baby/affine/F树，不能重复相加。

八条均通过相同合同：403批NTT/1979251 pairs/40218760归约coeffs、2400 mandatory checks、66139 GMP samples/1126 jobs/4 full、carry8241块/252 finishes/max_group222；pending0、errors0。Q SHA `33cc6c63cec26c39900a7ed4ff551e2c2e0a533422905b538ec4f4aa809f092f`，实验root sum033a77303713f3a6/xord16047fb14d39b21，叶115200/8064000字/FNV10619321735931855904保持。S4 H2D4.71/D2H1.34GiB保持，不能将kernel访存减少说成PCIe流量减少。

本次2.49%与前报告P2的2.62%都比较原M4控制，不能相加。此前xADD的6.26%属于另一个算术A/B，也不宜直接拼接为统一实测加速比。证据：[八次量化](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_shape_20261004/quantitative.json)、[ABBA](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_shape_20261004/stage2_ab/provenance.json)、[BAAB](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_shape_20261004/stage2_baab/provenance.json)。

自动策略纯卷积复测：k24 .021626795→.020404907 s（5.65%），k25 .049990912→.043469569（13.05%），k26 .099905878→.088131926（11.79%），k27 .198000127→.184214273（6.96%）。每尺寸各4样本/模式，1warm+3计时，全N输出在计时外比较。k23两模式实际都选原M4，.66%差异属于运行波动；策略没有声称优化该长度。[纯NTT量化](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_shape_20261004/pure_quantitative.json)。

### 4.2 新 D 系数与独立 holdout

同一实验exe、模式2，六条多D曲线：D570570为100.325490/101.593124 s（均100.959307）；D1381380为61.094581/63.411770（均62.253176）；D1411410为68.304997/70.839601（均69.572299）。加入上面四条D1231230锚点，共10行拟合，冻结后再测试小B2。D改变P/G/工作量，因此该组只用于成本拟合和候选排序，不是固定几何的NTT A/B。

新正系数按baby、affine、F树、giant、G树、fold、descent、inverse、accum、glue顺序为：

```text
3.4275264220727666e-6, 2.255624486683093e-5, 1.841123950024502e-10,
3.706072049528784e-7, 7.621290442850893e-11, 2.1330650275351144e-10,
4.192166619804115e-10, 1.636559352395959e-10, 1.4263335239938989e-6,
0.03115848164450274
```

这些系数乘§3特征得到估计秒数，不是CPU/GPU周期。拟合内full误差−4.95..+1.80%，leave-D-out−4.64..+5.65%。47-smooth D≤200000000共379419候选；arena6300MiB/owner640MiB下，大界首选D1381380/P126720/I1456028/G12，预测62.191254 s；小界首选D330330/P31680，预测12.606230 s。预算变化需要重新排名，不直接照搬D。

最后的独立B2=1e11 holdout，按330330/510510/510510/330330顺序：D330330实测14.048871/13.828779 s，D510510实测15.406727/15.233812 s；后者预测13.980214 s。四条误差−10.27/−9.26/−8.23/−8.84%，预测偏低，但每条330330都快于每条510510，排序通过。更早的八行fit/首组holdout保留为历史文件，最终fit未使用holdout反向修系数。

模型仍只能在已测scope辅助选择，不能保证全局最优。原M4使用原系数；模式2支持scope使用新系数；强制cooperative模式1没有拟合模型，回原§56.1。显式D优先，CPU选D时间另报并排除在stage2_full_wall之外。

证据：[六条测量](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_shape_20261004/calibration/measurements.json)、[最终fit](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_shape_20261004/fit_final.json)、[大界排名](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_shape_20261004/plan_final_large.json)、[小界排名](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_shape_20261004/plan_final_small.json)、[最终holdout](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_shape_20261004/holdout_final_summary.json)。

### 4.3 Nsight 与下一瓶颈

Systems 2026.1.3对原模式/模式2各采集一次，所记录kernel全部在GPU1。Stage1 chain结束后kernel池：outer/tile NTT29.388810→27.812577 s（−5.36%），point19.238211→19.237337，carry3.724846→3.634941，reduce1.935248→1.932300。NTT池只包含outer/tile/ntt前缀kernel，未含pack/carry/reduce；point包括baby、giant和其他s2g kernel。剖析时间不混入未插桩A/B。

从Stage1 chain之后第一kernel至最后GPU事件的近似窗口68.264240→66.378674 s，kernel/copy/memset事件并集58.494443→56.812298 s，无本进程事件间隙9.769798→9.566375 s，占14.31%→14.41%。此窗口不是精确Stage2计时，间隙不等于整卡idle或低occupancy；新NTT缩短了GPU计算，但没有消除CPU提交/准备间隙。

全采集范围（包含Stage1与初始化）传输计数/字节两模式一致：H2D4688次/6556779952 B，D2H6472次/3037648112 B，D2D92次/937993280 B。全范围1089次同步cudaMemcpy的host API合计22.125661→20.680337 s，其相关GPU copy仅.033602→.039464 s；host API时间含等待/暂存/同步，不能当作PCIe传输速度。CPU准备和同步仍有优化价值，需继续定位具体调用与依赖。

Compute 2026.2.1尝试采集outer_coop_kernel时退出1，实际错误为ERR_NVGPUCTRPERM（GPU1性能计数器权限不足）。本轮没有有效带宽、stall、occupancy或cycle计数，不以该次probe时间作为性能证据；未修改系统计数器权限。来源：[Systems汇总](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_shape_20261004/profile_quantitative.json)、[Compute日志](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_shape_20261004/ncu.log)。

### 4.4 正确性和生产接入

独立NTT门禁10/0，包含88项自动策略边界/override检查、原96组合/27131904字GMP前向及逆向、cached模式切换4次/3145728字、故障拒绝/资源LOCAL0。旧warp0/1各216组合/6854400字、4次切换/98304字和14次生命周期/leaked0复测通过。[NTT门禁](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_shape_20261004/gate/summary.json)、[旧路径](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_shape_20261004/legacy_gate.log)。

DPhaseModel抽出为共用header；独立CPU探针直接包含生产模型和实际NTT shape backend，2 profiles×2 bounds×8 D共32/0，比较精确N、owner、tree/inverse特征及各阶段/total估计，与整数Python特征和冻结fit一致。CPU探针不执行曲线、不分配GPU。[模型32项](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_shape_20261004/model_gate/summary.json)。

最终生产sm89/CUDA13.3编译成功，CUDA593.7 s，exe4042752 bytes、SHA256 `3cf38065e5f88347063f365ac85468475a6624f6e98b52804834d9aef3219f0e`。14项源依赖哈希复核通过，NTT CU/cooperative header与计时实验相同；最终header重构/新D系数/生产wrapper另有生产验证。入口**32/0**：基础21项、CUDA失败队列保留、saved-X已有因子、实际M4423显式D、warp/xADD默认与显式回退、默认模式2自动D大小界，以及outer0恢复原resident模型/outer1强制实验回旧模型。三种outer模式的小界叶摘要一致。第一次验收因准备样本遗漏N/A assignment-id而触发finished文本断言，已按原样本在全新目录重跑32项；源码未因此修改。

生产默认NTT_XADD6=1、NTT_D_MODEL=1、NTT_FUSE_WARP_TAIL=1、NTT_FUSE_COOP_OUTER=2。尺寸策略受设备/t/warp/compact/M限制，未测形状沿用原planner。outer0使用原resident系数，outer1没有匹配经验fit而回§56.1；D模型0恢复旧D选择。显式D仍优先。每条save使用自己的sigma/B1/Q；worktodo用户示例仍选择961–970，xxx拒绝并保留队列，冻结因子59649589127497217正确。

实际save显式D1231230：init/main/full=14.112085/53.712880/67.824965 s；自动大界D1381380：init/main/full=15.391578/48.891094/64.282673 s，CPU选D另0.146617 s；自动小界D330330：init/main/full=3.290443/10.134760/13.425203 s，CPU选D另0.183744 s。叶FNV依次10619321735931855904、4244971527793015097、7549663880496122317保持；这些是单次接入验收，不作为性能A/B。save-Q入口的root摘要与实验自算Stage1入口不同，跨入口不直接比较projective摘要。

产物：[生产exe](D:/code/MPA-OpenCl/build_cuda_cmake/production_stage2/ecm_cuda_stage2.exe)、[manifest](D:/code/MPA-OpenCl/build_cuda_cmake/production_stage2/build_manifest.json)、[32项验收](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_shape_20261004/production_accept_final/summary.json)、[来源复核](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_shape_20261004/production_provenance.json)。构建/日志及旧F85快照留在ignored目录；源码、门禁、工具和报告提交Git。

## 5. 下一阶段：Tensor Core 与多曲线吞吐

后续实验已完成独立132bit整数MMA与真实tile：35/0门禁，锁定CTA256后k24..26前向慢约18.3%、逆向慢约5.6%、roundtrip慢约10.9%；CUDA+TC双流也比CUDA+CUDA慢约9.1%，Systems重叠约47μs。保留实验工具，生产3CF38065…19F0E保持。公式、源码行号、留出尺寸、容量和剖析见[Tensor实验报告](D:/code/MPA-OpenCl/docs/STAGE2_TENSOR_GOLDILOCKS_EXPERIMENT.md)。下面保留本阶段收尾时的设计依据。

Tensor Core候选先做独立精确整数probe，再决定是否接入tile/outer。固定参考 `Terminus-IMRC/tensor-core-ntt` 提交 `6f407daa8a4cef96331511ae86b922b520d7aa33`：其16×16矩阵将64bit拆成8个byte，按byte对调用整数MMA；现有归约器构造函数明确拒绝≥63bit模数。因此模板名modulus_bits=64并不代表支持本项目q=2⁶⁴−2³²+1。[矩阵实现](https://github.com/Terminus-IMRC/tensor-core-ntt/blob/6f407daa8a4cef96331511ae86b922b520d7aa33/include/polyarith/cuda/ntt.cuh#L71)、[归约器限制](https://github.com/Terminus-IMRC/tensor-core-ntt/blob/6f407daa8a4cef96331511ae86b922b520d7aa33/include/polyarith/modular.cuh#L326)。本阶段只阅读，没有导入/编译参考库。

可尝试u8×u8→s32 MMA的小矩阵NTT。若一个输出累加K=16个64bit乘积，各byte对角的最大值16·8·255²=8323200，低于INT32上界；但完整和最多132bit，不能用128bit截断。重构到low64/high64/top4bits后，可利用2⁶⁴≡2³²−1、2¹²⁸≡−2³² (mod q)精确归约。全0、q−1、最高位、carry边界、随机dense和独立GMP频谱/逆向都需覆盖。

sm89支持整数MMA；mma.sync仍要求整个warp以一致控制流执行并等待。Tensor与CUDA算术之间的依赖、寄存器/shared和调度资源会限制重叠，不能把两个峰值吞吐直接相加。先比较含拆分/重构/约减的完整probe，再探索跨warp/双缓冲的重叠；任何生产收益仍需完整Stage2验证。[PTX整数MMA与同步约束](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma)。

多曲线目标采用curves/h和固定结果/检查覆盖，允许每条曲线延迟略增。大形状不能简单复制两个约5GiB进程到8GiB设备。优先评估两条曲线独立Q/Γ/F/finv/H/oracle/carry状态与共享NTT workspace lease，在CPU准备另一曲线时保留当前GPU执行；每曲线owner、pinned staging、host提交及共享workspace都计入预算。现有全局模数、统计和默认stream尚需所有权改造，不能直接在同进程多线程调用当前入口。

Nsight Systems用于查看kernel/copy事件并集和CPU提交间隙；无本进程事件不等于整卡idle。Nsight Compute只在有效计数器采集成功时报告带宽/occupancy/周期。公平Prime95对比仍要求相同N、sigma/Q、B1/B2、曲线族和实际覆盖，明确检查强度、线程数、冷热启动及D/内存配置；旧CPU90.460 s不能当作本轮已经超越的证据。

## 6. 实现索引

- [设备scope:1421](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:1421)、[尺寸策略:1439](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:1439)、[arena cache key:2164](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:2164)。
- [D模型与两组系数:12](D:/code/MPA-OpenCl/tools/bench/stage2_d_model.cuh:12)、[weighted unit:60](D:/code/MPA-OpenCl/tools/bench/stage2_d_model.cuh:60)、[成本:93](D:/code/MPA-OpenCl/tools/bench/stage2_d_model.cuh:93)、[scope与版本:10763](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10763)、[CPU选D计时:10952](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10952)。
- [整数features:76](D:/code/MPA-OpenCl/tools/bench/calibrate_stage2_d.py:76)、[fit锚点校验:31](D:/code/MPA-OpenCl/tools/bench/fit_stage2_d.py:31)、[排名:1](D:/code/MPA-OpenCl/tools/bench/plan_stage2_d.py:1)、[同exe Stage2 A/B:1](D:/code/MPA-OpenCl/tools/bench/bench_stage2_reduce_ab.ps1:1)。
- [策略/纯卷积probe:24](D:/code/MPA-OpenCl/tools/test/ntt_coop_outer_probe.cu:24)、[策略88项:62](D:/code/MPA-OpenCl/tools/test/ntt_coop_outer_probe.cu:62)、[GMP门禁:1](D:/code/MPA-OpenCl/tools/test/test_ntt_coop_outer.py:1)、[共用模型探针:1](D:/code/MPA-OpenCl/tools/test/stage2_d_model_probe.cu:1)、[模型32项:1](D:/code/MPA-OpenCl/tools/test/test_stage2_d_features.py:1)。
- [生产默认:15](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:15)、[生产构建:1](D:/code/MPA-OpenCl/tools/build/build_stage2_local.ps1:1)、[模型探针构建:1](D:/code/MPA-OpenCl/tools/build/test/build_stage2_d_model_probe.ps1:1)。

## 2026-10-05 后续发布状态

短归约重新标定D并完成独立留出验证，生产默认已提升short1；旧归约0保留。实际生产入口33/0、S4后端选择器30/0，同exe/save固定D1381380 ABBA均值65.014116→56.834263 s（快12.58%）。本报告前面的默认关闭/旧产物数据是历史阶段记录。最新SHA、源文件行号、scope、容量和门禁边界见[短归约 D 标定与生产报告](D:/code/MPA-OpenCl/docs/STAGE2_SHORT_REDUCTION_D_CALIBRATION.md)。


## 2026-10-08 实际热形状的管理员 NCU 诊断（已完成）

驻留根生产移植已提交930dc3a；其生产整曲线收益尚不稳定，不能沿用开发版约1.20%的比例。[完整验收与最新来源](D:/code/MPA-OpenCl/docs/ECM_CUDA_STAGE2.md:667)。本阶段保持fbfc9d24基线、有效M4423/B1=1000/sigma26/lcm save、D1381380/P126720/I1456028/B2=2011326186870，先采实际热点计数，尚未改变NTT数学或发布包893。

### 目标与异常筛选开销

从同binary已完成Systems trace核对冻结launcher、mangled名字及单stream/direct launch序列：tile长度由gridX·dynamicShared/8、outer由gridX·2^M·V重建，M8取V16，其余V32。选择来自实际几何，不声称捕获kernel实参或精确树phase。

- N=2^11/batch990的tile正/逆：grid(1,990,1)、block512、dynamic shared16384B。
- N=2^27/batch1的M7/M8正/逆outer：四项grid32768/block256，dynamic shared0，static shared另核对。
- N=2^27/batch1的tile正/逆：grid32768/block512、dynamic shared32768B。

初版r0用匹配名字的launch-skip，已完成4个报告，但等待第1053次forward tile时，前序F树首层150.842秒、调用统计forward2.350秒/inverse0.002秒；与正常前序不相容。r2用全kernel skip-before-match也把baby前序约4秒拉长至38.115秒；r3用kernel-id的精确invocation仍复现首层151.027秒/forward2.363秒。三条不完整任务均核对PID/启动时间及自身进程树后终止；原始源、命令、日志、4个报告和中断证明保留，r1仅准备未启动。没有将这些前序时间解释为Stage2生产回退或发布性能数据。证据支持筛选方式引入开销，尚未定位NCU内部机制。

### 隔离 host range 构建

[生成器](D:/code/MPA-OpenCl/tools/bench/prepare_stage2_ntt_range.py:1)复制fbfc9d24冻结闭包，仅在tile/cooperative host launch前后插入[profiler API范围](D:/code/MPA-OpenCl/tools/bench/stage2_ntt_profile_range.cuh:1)，由S2_NCU_TARGET选中上述几何的第一次调用，调用cudaProfilerStart/Stop各一次。两处原header可逐字节撤销插入恢复基线；新增header仅host代码。真实生产源码和发布包未修改。

隔离44f0149b候选CUDA编译114.8秒/split6；27个依赖、5对象。该binary不同于正式计时基线，不能冒充同binary验证。须先用[完整GPU等价工具](D:/code/MPA-OpenCl/tools/bench/verify_stage2_ntt_range.py:1)对两个实际exe分别导出cuobjdump，核对全部GPU SASS和resource。首次原始字节检查拒绝：仅生成目录引起匿名TU ID从8701b53f变为9ae733aa；原失败凭据与工具保留。最终只规范化这个已识别符号ID，172个kernel的完整指令文本、指令编码、调度编码及全部资源逐字节相同；两个原始SASS文件各341,269,914B，原始SHA仍分别保存，未写成raw hash相同。资源计数相同或插入可撤销均不能单独替代完整比较。完整GPU指令等价仍不证明host代码/周期或生产耗时等价。

[采集工具](D:/code/MPA-OpenCl/tools/bench/profile_stage2_ntt_ncu.py:1)要求上述范围项目及GPU等价凭据，Compute2026.2.1管理员串行8项，仅GPU1；profile-from-start off、kernel名字过滤、launch-count1、clock/cache control none。先验证异常的大tile前序是否恢复，再补齐8项。collect严格核对实际template、device/grid/block、dynamic/static shared和REG，Start/Stop各一次，完整leaf/factor与原未插桩正式矩阵的默认NTT/S4覆盖。replay及系统RAM备份警告保留，不混入正式A/B或当作容量认证。

原始及失败证据在ignored build_cuda_cmake/_stage2_ntt_hot_20261008。8项capture/export及collect均完成，父进程正常exit0；独立审计重新核对原始报告/CSV、8个实际几何、源/对象、完整输出和默认检查覆盖，通过。原2026-10-04及10-05的默认、时间和NCU权限失败是历史快照。


### 八个实际 kernel 的硬件结果

每项16-pass kernel replay，clock/cache control none，保留系统RAM备份警告。下列时长/带宽为诊断采集，不替代未插桩A/B，也不保证实际运行缓存状态。尤其N=2^11/batch990的数据可能受L2和重复回放影响。

- N27 tile forward：12.998592ms，DRAM165.13GB/s、64.56% peak；SM throughput81.18%、active warp98.96%、issue active63.17%。
- N27 tile inverse：16.974016ms，DRAM189.75GB/s、74.19% peak；SM75.26%、active warp99.46%、issue61.51%。
- N27 M7 forward：11.058752ms，DRAM194.77GB/s、76.15% peak；SM62.28%、active warp32.94%、issue53.10%。
- N27 M7 inverse：11.190304ms，DRAM192.49GB/s、75.26% peak；SM63.05%、active warp32.95%、issue51.49%。
- N27 M8 forward：11.951232ms，DRAM179.56GB/s、70.20% peak；SM64.46%、active warp32.97%、issue54.65%。
- N27 M8 inverse：12.069536ms，DRAM177.84GB/s、69.53% peak；SM65.45%、active warp32.99%、issue53.17%。
- N11/b990 tile forward：197.888µs，DRAM60.00GB/s、26.82% peak；SM79.24%、active warp96.22%、issue64.19%。
- N11/b990 tile inverse：261.088µs，DRAM113.81GB/s、50.86% peak；SM73.21%、active warp96.87%、issue62.58%。

四个tile均REG40/allocated40，寄存器容量3CTA/SM；t12 shared容量3CTA，t11 shared容量5CTA，已实测接近满active warp。四个outer forward REG48、inverse REG46/allocated48，寄存器容量5CTA/SM、shared容量仅2CTA/SM。八项local load/store sectors均0；没有证据支持先减local数组或提高tile occupancy。

原始stall单位也保留：tile的math-pipe-throttle per-issue-active ratio约4.59..5.27 inst，wait约1.84..2.23；outer wait约1.67..1.75，long-scoreboard约1.12..1.65，barrier约0.58..1.01。它们不是时间百分比，也不能相加成阶段墙钟。SM throughput是该指标定义下的利用率，不等于所有运算单元统一的利用率。完整数据见[quantitative.json](D:/code/MPA-OpenCl/build_cuda_cmake/_stage2_ntt_hot_20261008/quantitative.json)及8份raw metrics.csv。

显式range的首个F树层0.238秒，与之前筛选造成的150.842/151.027秒形成实际回归检查；前序开销问题被修复，不解释为Stage2算法提速。完整leaf4244971527793015097、因子及默认S4/NTT覆盖逐项与原正式参考一致；diagnostic binary与基线不是同一个exe，结论依赖上述完整GPU指令比较和实际输出核验。

### 据此调整下一候选

1. **先试outer的V轴收窄**：当前M7/V32、M8/V16的主shared数组均32KiB，使CTA容量卡在2。候选M7/V16、M8/V8保持radix M和NTT pass数、DIF/DIT顺序、主数组读写与数学工作量；每CTA数据减半、grid加倍。静态公式预计M7 forward/inverse约18,688/19,328B、M8约19,584/19,968B，加1024B/CTA driver reserve，shared容量可能分别到5/4CTA。仍需实际REG/容量API验证；多一个小d同步层、更多CTA及辅助root加载可能抵消收益，不据公式承诺加速。
2. **tile减少整数指令**：当前warp已接近满载、math-pipe压力较大，优先比较canonical Goldilocks add/sub的PTX与现有C++ SASS，再考虑固定t特化。仓库冻结sppark的[gl64_t.cuh](D:/code/MPA-OpenCl/.refactor/ntt_sources_20261004/sppark-9e5c7951d4ff4992f78af26f48d3c9230b8c4136/ff/gl64_t.cuh:67)可供算法参考，但其partially-reduced合同不能直接搬入本项目canonical路径，CC链必须在单asm块内自洽。
3. 每个候选先独立GMP/全输出/缓存切换及资源门禁，再测试真实length/batch和完整Stage2固定D交叉A/B。旧全局tile11、单位根PTX重测和单纯u展开的负/不稳定结果仍保持。NTT策略变更后需重新标定D与Auto B2；本阶段没有新cprof或发布包。

本阶段完成的是可复用的实际热形状诊断及瓶颈定位，尚未实现上述V轴/算术候选。GPU frontier驻留、较大16k容量、最终chain/短尾和多曲线RAM/VRAM lease继续属于长期工作。

### 复现与证据归档

在仓库根目录执行以下PowerShell命令。`python`指可用的Python 3；`reproduce`目录必须尚不存在，避免覆盖原证据。基线exe须连同其冻结sources、manifest和对象保留。Systems/reference取自已完成的同基线采集及正式计时矩阵。

```powershell
$repoRoot = (Get-Location).Path
$rootPhase = Join-Path $repoRoot 'build_cuda_cmake/_stage2_root_prod_20261008'
$probe = Join-Path $repoRoot 'build_cuda_cmake/_stage2_ntt_reproduce'
python tools/bench/prepare_stage2_ntt_range.py --exe "$rootPhase/production_r3/ecm_cuda_stage2.exe" --output "$probe/project"
& "$probe/project/tools/build/build_stage2_local.ps1" -Build "$probe/native" -Engine production -GlBackend ptx -SplitCompile 6
Set-Location $repoRoot
python tools/bench/verify_stage2_ntt_range.py --exe "$probe/native/ecm_cuda_stage2.exe" --project "$probe/project"
$captureArgs = @('--exe', "$probe/native/ecm_cuda_stage2.exe", '--range-project', "$probe/project", '--reference', "$rootPhase/cross_timing_final_r3/measurements.json", '--systems', "$rootPhase/nsys_root_1", '--output', "$probe/capture")
python tools/bench/profile_stage2_ntt_ncu.py @captureArgs
$captureScript = (Resolve-Path "$probe/capture/capture_all.ps1").Path
$captureJob = Start-Process powershell.exe -Verb RunAs -WindowStyle Hidden -PassThru -ArgumentList @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', $captureScript)
```

等待该进程终止、检查capture/exit.txt为0后，执行`python tools/bench/profile_stage2_ntt_ncu.py @captureArgs --collect-only`。采集依赖GPU1空闲及管理员计数器权限；不能和编译/正式计时并行。工具为当前冻结模板/几何设计，改变V或launcher后须同步核验形状公式，不能直接套用旧grid推导N。

本批独立[审计脚本](D:/code/MPA-OpenCl/build_cuda_cmake/_stage2_ntt_hot_20261008/audit_phase.py)、[审计结果](D:/code/MPA-OpenCl/build_cuda_cmake/_stage2_ntt_hot_20261008/final_audit.json)和[evidence manifest](D:/code/MPA-OpenCl/build_cuda_cmake/_stage2_ntt_hot_20261008/evidence_manifest.json)共同绑定源/对象、原始SASS、8份报告/CSV、正式参考和保留的失败。归档包含文本/输入/原始剖析证据；exe、对象和动态库在原路径保留并单独登记SHA，提交身份在归档后另记commit_identity.json。

## 2026-10-08 outer V 收窄：同二进制候选

### 实现、工作量与容量

接续上述管理员诊断。[开发header](D:/code/MPA-OpenCl/tools/bench/ntt_coop_outer.cuh:14)给kernel增加编译期V参数，host launcher按`NTT_OUTER_NARROW=0..3`选择：0原布局、1仅M7/V16、2仅M8/V8、3同时收窄。没有在每个蝶形内增加运行时选择；M5/M6保持原V32。V不影响根表及变换计划，可在同一个cached arena内切换。生产header尚未修改。

记R=2^M、N为单slice NTT长度、b为batch数。两种布局的每pass主数组读写仍16Nb B，蝶形模乘MNb/2、合成根乘积(R−1)Nb/R、coarse平方(M−1)Nb/R均保持。CTA数由Nb/(RV)变为两倍；每CTA barrier从M7的11→12、M8的13→14。radix表的逻辑加载量8(R−1)Nb/(RV)也加倍，N=2^27/b1时每pass额外M7约31.75MiB、M8约63.75MiB；实际DRAM事务受缓存影响，不能将这些公式当成硬件带宽测量。

声明shared为8[RV+(R−1)+128+(inverse?M:2)V] B，编译按16B对齐。实际资源API核验：

- M7 forward/inverse：35,328/36,608→18,688/19,328B，最大active CTA容量2→5。
- M8 forward/inverse：36,096/36,864→19,584/19,968B，容量2→4。
- 四种正向均REG48、逆向REG46；LOCAL0。这是容量上限，尚无候选实测active warp或stall。

没有新增host/pinned/NTT workspace数据数组，allocator payload和接口传输的理论增量0；开发binary多了四个kernel实例，代码/模块容量不为0。本阶段没有新的进程NVML峰或实际PCIe量认证。

### 独立门禁与完整卷积

[probe](D:/code/MPA-OpenCl/tools/test/ntt_outer_v_probe.cu:1)直接包含实际开发header，[构建脚本](D:/code/MPA-OpenCl/tools/build/test/build_ntt_outer_v_probe.ps1:1)冻结8份依赖并核验编译前后SHA，固定PTX3/u0。probe_r2 SHA150f73ea…490df1，编译19.9秒。

[串行工具](D:/code/MPA-OpenCl/tools/bench/bench_ntt_outer_v.py:1)完成17个调用：四mask各96组GMP正/逆谱、27,131,904 words及4次cached切换、3,145,728 words；每mask故意损坏被拒绝。新增24组dense GMP谱/逆结果、2,951,568 words，含独立slice、17word padding、stride及同一计划内0/3/1/2/0/3切换；dense故障拒绝。四个非法mask拒绝、资源、88项policy和200000次device算术自检均通过。

两轮k23..27完整有限域卷积，每长度16run、每run1warm＋3event样本；四mask每轮各4run，初始化/GMP参考/全N结果检查在event外，全部输出bad0。这里的run均值不是独立ECM曲线，也没有置信区间。

- k25仅M8收窄：两轮均值快2.7434%／2.7237%；同时收窄快2.7253%／2.7339%。
- k26仅M8：快2.5932%／2.6257%；同时收窄快2.6738%／2.6271%。
- k27仅M7：快4.0926%／4.2150%；仅M8快2.3612%／2.4679%；同时收窄快6.3811%／6.4077%。
- k24的两个mask都不作用于M6，波动约−0.002..+0.066%；k23不执行合作outer，仍出现约±2.4%的顺序/状态差，全部保留，不作为算法收益。

来源位于ignored build_cuda_cmake/_stage2_outer_v_20261008，timing_r0/r1分别保留完整64行/长度。一次跨构建SASS比较显示8个默认outer函数正文并不逐字节相同，保留baseline_outer_sass.json；不能因REG相同或源代码等价声称旧生产baseline机器码保持。本轮局部和原生对照都使用同一个binary中的0/3模式。

### 原生接入与验证边界

开发[成本保护和配置日志](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:11139)在mask非0时禁止沿用旧D经验profile，显式D优先；没有新的Auto B2成本认证。[原生工具](D:/code/MPA-OpenCl/tools/bench/bench_stage2_outer_v.py:1)冻结28份源码/5对象，并与原生产正式矩阵的完整输入、leaf及六项默认检查覆盖核对。

开发候选SHA848ea580…94472f，CUDA262.0秒/split6。12条原生gate完成：大界M4423两模式，以及generic8193/M16381/generic16384/已知因子/非单位五个有效save各两模式；小形状每条另有150个scaled fixture和完整GMP下降节点检查。宽小形状不一定执行M7/M8收窄，不能用它们替代较大16k候选性能或容量验收。

整曲线采用同save、B1=1000/lcm/sigma26、B2=2011326186870、显式D1381380、owner640/reuse3、arena6300、GPU Gamma/驻留根、默认NTT/S4检查；先各一次预热，再ABBA＋BAAB。8条正式full依次37.690016／37.825909／37.167210／37.233030／37.786032／37.844180／37.490107／37.066242秒，mask顺序0/3/3/0/3/0/0/3。均值37.56433325→37.46134825秒（减少0.27416%），两组分别慢0.09353%／快0.63983%，尚未建立稳定整曲线收益；全部样本保留，没有置信区间。完整leaf4244971527793015097、因子及六项默认NTT/S4覆盖保持，精确main计时总账差不超过3ms。

### 实际曲线 Systems 与决策

管理员Systems2026.1.3在全部计时结束后串行捕获0/3，capture/export正常完成、单GPU1。实际demangled模板包含V，独立审计从grid·2^M·V恢复N/batch，核对全部合作outer形状的调用次数保持、M7/M8 grid加倍和shared/REG。N27/b1四类实际grid32768→65536，shared为前述四个候选值，证明候选确实用于真实热形状。

合作outer累计4.395095→4.137593秒（少0.257502秒／5.85885%）；tile7.994454→8.094780秒，仍为最大NTT单项。N27四类outer合计约2.747671→2.526593秒，其中M7 forward0.847447→0.749971、inverse0.464501→0.406768；M8 forward0.934563→0.885174、inverse0.501160→0.484680秒。这是诊断trace中的累计kernel时间，不作为另一个正式整曲线提速比例或因果分解。

自身GPU事件span36.888666→36.842774秒，并集30.852613→30.769642秒，无自身事件6.036054→6.073131秒（16.36%→16.48%）。CPU准备/同步空隙仍存在；tile/carry等也有波动，不能把收益被抵消全部归因为某一个CPU阶段。H2D4866次/7,009,115,275B、D2H6130次/3,138,157,112B、D2D40次/841,498,560B两侧相同，符合只改变kernel组织的预期；不能声称消除了PCIe传输。

**保留开发实验，不提升生产默认。** 局部卷积和真实outer累计成本有改善，但整曲线稳定收益未证实。下一项优先tile canonical add/sub的指令减少，以及GPU下降frontier减少host准备；旧u展开/全局tile11负结果仍有效。不为这个未提升候选拟合新D/cprof，生产header、发布893保持。候选较大16k容量、实际NCU active warp/stall和未来生产移植仍须独立验证。

[完整量化](D:/code/MPA-OpenCl/build_cuda_cmake/_stage2_outer_v_20261008/quantitative.json)、[独立审计](D:/code/MPA-OpenCl/build_cuda_cmake/_stage2_outer_v_20261008/final_audit.json)、[原生矩阵](D:/code/MPA-OpenCl/build_cuda_cmake/_stage2_outer_v_20261008/native_timing_r0/measurements.json)及evidence.zip/manifest绑定冻结源码、对象、输入、原始结果和trace。首次probe编译因误用完整变换的stride接口被拒绝，后改为直接调用真实strided outer/tile并检查padding；失败源码和日志保留。首次sandbox管理员启动失败没有采集，改用获准的提升权限启动后才完成两侧；原失败与实际PID/启动时间登记保留。

收尾将collector记录的环境字段精简为NTT_*和CUDA_LAUNCH_BLOCKING，保留原采集器源码及精确两处输出表达式差异证明；输入、运行控制、算术/时序字段和原始日志均未改变。没有用修改后的工具SHA冒充原采集器身份，独立审计重新绑定各自版本。

复现（新输出目录，Python 3，GPU1）：

```powershell
& tools/build/test/build_ntt_outer_v_probe.ps1 -Build build_cuda_cmake/new_outer_v_probe
python tools/bench/bench_ntt_outer_v.py --exe build_cuda_cmake/new_outer_v_probe/ntt_outer_v_probe.exe --mode gate --output run/new_outer_v_gate
python tools/bench/bench_ntt_outer_v.py --exe build_cuda_cmake/new_outer_v_probe/ntt_outer_v_probe.exe --mode timing --gate run/new_outer_v_gate/summary.json --output run/new_outer_v_timing
& tools/build/build_stage2_local.ps1 -Engine development -Build build_cuda_cmake/new_outer_v_native -GlBackend ptx -SplitCompile 6
$nativeArgs = @('--exe', 'build_cuda_cmake/new_outer_v_native/ecm_cuda_stage2.exe', '--save', 'build_cuda_cmake/_fixed_d_20261005/native_accept/m4423.save', '--reference', 'build_cuda_cmake/_stage2_root_prod_20261008/cross_timing_final_r3/measurements.json')
python tools/bench/bench_stage2_outer_v.py @nativeArgs --mode gate --fixtures build_cuda_cmake/_stage2_wide_20261007/fixtures_r2/fixtures.json --output run/new_outer_v_native_gate
python tools/bench/bench_stage2_outer_v.py @nativeArgs --mode timing --gate run/new_outer_v_native_gate/measurements.json --output run/new_outer_v_native_timing
```

Systems复现使用[现有工具](D:/code/MPA-OpenCl/tools/bench/profile_stage2_points.py:1)新增`--outer-narrow 0|3`，其他参数采用本批capture_systems.ps1；管理员串行运行，保持point1/pair1/baseCPU0/C64/min32768/Gamma1/root1/owner640/reuse3/arena6300/factor-only，最后对两个实际trace做离线分析。

## 2026-10-08 canonical Goldilocks add/sub：隔离算术与实际 batch 形状

### 范围与数学合同

接续outer V实验，先减少tile整数指令。新增[canonical PTX原语](D:/code/MPA-OpenCl/tools/bench/ntt_goldilocks_addsub.cuh:1)仅接受`0≤a,b<q`，`q=2^64−2^32+1`，输出仍在`[0,q)`。本阶段**尚未修改开发/生产NTT实际别名或发布包**；[生成器](D:/code/MPA-OpenCl/tools/bench/prepare_ntt_addsub.py:1)从冻结来源复制隔离项目，编译期mask bit0替换减法、bit1替换加法。不存在逐蝶形的运行时选择。outer V保持原布局0、PTX归约3、u展开0、warp tail1。

减法用32bit低/高字的完整CC链求`d=a−b mod 2^64`及borrow mask。无借位时输出d；借位时从d减`epsilon=2^32−1`，即`a−b+q`。加法先求64bit和及carry，再求`s−q`及borrow；当`carry−borrow≥0`时选后者，否则选s。由于`a+b<2q`，一次修正足够。所有CC生产/消费在同一个asm块内，未借用sppark partially-reduced输入合同。

实际原别名在[开发NTT](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:278)与[独立生产NTT](D:/code/MPA-OpenCl/src/cuda/stage2/ntt_runtime.cuh:278)。隔离项目的三处文件修改均有唯一匹配和逐字节逆变换证明；编译器宏值由每次实际stdout再次确认。独立项目、实际exe、sources和GMP依赖哈希绑定，不能用当前源码代替先前编译来源。

### 正确性与静态指令

[原语探针](D:/code/MPA-OpenCl/tools/test/ntt_addsub_probe.cu:1)对每个操作/实现比较1,000,210对canonical输入，包含0/q−1/最高位、carry/borrow边界与百万随机对；四种64轮依赖链另各256字对照GMP。正常exit0、末尾减法毒化exit3已重新确认，与原始日志逐字节相同。三个设备数组payload共24,005,040B、释放后live0，不是整进程显存峰。

实际SASS四个原语kernel都占40条静态指令（含NOP）；排除NOP后，加法32→31、减法28→27，各只减少**一条**。不能将五条PTX算作五条SASS，也不能按静态chain总数比较吞吐：编译器对四种chain的展开次数不同。

四种mask的独立NTT门禁共68次进程调用，每mask17次，分别覆盖96组合/27,131,904字GMP正逆、四次cached切换/3,145,728字、24组dense与17-word stride padding/2,951,568字，以及故障、非法mask、资源、策略和GL检查。dense门禁仍覆盖outer V四档；性能测量只用V0，不能混写。

默认warp tile四种mask均REG40/LOCAL0；仅加法mask2的inverse实例有STACK8，其余默认warp实例STACK0。旧warp0实例也有寄存器差异，未据此声称全部资源或整个机器码相同。减法候选的批量probe warp forward静态非NOP1687→1640、inverse1936→1908；这些包括多个控制流分支，**不是每次调用的动态退休指令或周期**。本阶段没有候选NCU硬件计数。

### 单batch完整卷积：两轮、保留慢样本

每轮4条全长预热，随后`k=23..27`各16次进程；四mask按固定Latin顺序`0,1,3,2,1,2,0,3,2,3,1,0,3,0,2,1`，每进程8组、每组1预热＋3个CUDA event测量。两轮合计160正式进程/8全长预热；每个进程均比较全部N输出，bad0。每个长度/实现每轮只有4个进程，内部24个event不能算成24条独立曲线，没有置信区间。

仅减法mask1的耗时减少比例，依次列第一/第二轮：

- `2^23`：+1.057% / −0.684%。
- `2^24`：+0.970% / −0.991%。
- `2^25`：+2.620% / +1.698%。
- `2^26`：+2.135% / +1.838%。
- `2^27`：+2.371% / +2.481%；均值112.362→109.697ms、112.381→109.593ms（四舍五入，原始精度见quantitative）。

第二轮小长度有mask1约8.656ms/13.077ms的慢进程，全部保留，未以正常样本替换均值。仅加法mask2所有长度/两轮均变慢约0.51%..3.48%；两者同时替换mask3没有超过仅减法的大长度收益，小长度亦不稳定。因此后续只推进减法候选，不将“add/sub同时替换”作为默认。

### N=2^11/batch990：独立GMP与两轮交叉计时

[批量探针](D:/code/MPA-OpenCl/tools/test/ntt_addsub_batch_probe.cu:1)复用实际tile模板。每slice都有三个非零输入系数，slice按三种不同边界输入循环；前向参考用GMP独立计算root和bit-reversed频谱，乘积五项用GMP求值。不是把只有第一个slice非零的全局稀疏数组误用作批量输入。其余系数和padding也全部比较。

mask0/1各测试batch1/3/990、连续及stride=N+17共六种形状；每形状检查forward、inverse含pointwise/scale、完整卷积全部输出，正常进程各18行。另在最后slice的第五个系数写入毒化，两边均只检出一个错误、exit3。资源API均REG40/LOCAL0/static shared0/dynamic shared16384、容量3CTA/SM，与已采实际热点形状一致；这仍不是候选实际active-warp测量。

首次批量参考被门禁拒绝，两侧仍是原始算法：Windows的`unsigned long`只有32bit，`mpz_powm_ui`截断了root的长指数。改为`mpz_import`＋`mpz_powm`的大整数指数后，在新目录重编并完整通过；失败源码、二进制及bad=5,093,256日志保留。没有修改GPU算术来匹配错误参考。

每轮两进程预热，再固定ABBA＋BAAB八正式；每进程每phase1预热＋24测量，输入重置和完整结果比较位于event外。两轮合计16正式/4预热，没有编译或profiler与正式计时重叠：

- forward：202.379→195.232µs、202.432→195.297µs，少3.531%/3.525%。
- inverse含pointwise与scale：258.805→251.040µs、258.837→251.147µs，少3.000%/2.971%。
- 完整卷积：657.376→634.645µs、657.632→634.784µs，少3.458%/3.474%；四个ABBA/BAAB分组均同向。

这些是隔离实际形状的kernel/event成本，不含Stage2组包、carry、mod-N归约、GCD或启动成本，不能当整曲线收益。

### 工作量、容量与下一阶段

对`N_ntt=2^k`、batch=b，单次完整NTT的逻辑蝶形数为`b*N_ntt*k/2`；两次forward＋一次inverse卷积的逻辑add/sub各`3*b*N_ntt*k/2`。替换只改变实现，不减少该数学工作量、变换pass、逻辑主数组读写或PCIe接口量；原语的一条静态SASS差不能机械乘该公式当动态周期节约。

候选不新增NTT持久数组。批量probe为校验分配的两份数据及reference/error payload为`16*b*stride + 48*N_ntt + 128 B`，batch990连续32,538,752B，padding版32,808,032B；另有原NTT arena/table，不包含在这个探针payload字段中，也不当作进程峰。计时不包含reference上传。原语/NTT正确性检查增加的拷贝不能当生产传输变化。

**减法候选值得接入开发引擎做下一轮完整曲线验证。** 需编译期选择、实际mask日志、旧D/Auto B2 scope保护、默认与候选的冻结来源、宽位数/块尾/退化回退门禁，以及固定D交叉整曲线A/B。尚无开发native整曲线A/B、生产移植、新D/cprof或较大16k容量认证。GPU下降frontier仍是另一主要候选，不以这轮约2%..3%的局部收益替代减少CPU准备的目标。

证据根目录为ignored `build_cuda_cmake/_stage2_addsub_20261008`。[独立审计工具](D:/code/MPA-OpenCl/tools/bench/audit_ntt_addsub.py:1)从原始日志重新核对顺序/所有输出、完整来源、event均值、两轮全部样本、批量padding/毒化、原语及tile SASS计数；[quantitative.json](D:/code/MPA-OpenCl/build_cuda_cmake/_stage2_addsub_20261008/quantitative.json)保留分组和逐文件SHA。归档/提交凭据在该目录另存，发布893保持。

复现时使用新目录；四mask探针先运行`prepare_ntt_addsub.py --mask 0..3 --output <project>`，再在对应project根目录调用其中冻结的`tools/build/test/build_ntt_outer_v_probe.ps1 -Build <absolute-build>`。将四个build保存为study的`ntt_m0..3`，对应project保存为`project_m0..3`，运行`bench_ntt_addsub.py --study <study> --mode gate|timing --output <new-output>`，timing另指定已完成`--gate <gate/measurements.json>`。批量probe用`build_ntt_addsub_batch_probe.ps1 -SourceBuild <ntt_m0|1> -Mask 0|1 -Build <new-build>`；随后`bench_ntt_addsub_batch.py --baseline <mask0-exe> --candidate <mask1-exe> --mode gate|timing --output <new-output>`，timing同样要求gate。所有实验固定GPU1。

## 2026-10-08 canonical减法接入开发引擎

### 编译选择、成本scope与来源

接续已提交54e99b8的隔离验证。[开发NTT别名](D:/code/MPA-OpenCl/tools/bench/ntt_poly_probe.cu:287)现有编译期`NTT_GL_ADD_SUB_MASK=0..3`，默认0，bit0=sub、bit1=add；每蝶形没有运行时判断。仅非零构建包含canonical PTX头，因此其他默认0探针不增加该头的实际编译依赖。隔离生成器识别已接入的相同别名，不重复插入。生产NTT/CU未移植这个候选。

[原生builder](D:/code/MPA-OpenCl/tools/build/build_stage2_local.ps1:14)增加`-AddSubMask`并绑定source signature、编译命令、manifest和HostOnly检查；开发冻结闭包29个raw文件、5对象。production只接受0；不把实验选择加入生产算法分支。两份development/PTX3/u0/split6构建同时编译，CUDA约366.3/365.8秒，编译期间未计时曲线。

[run_real](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:11142)对非零mask关闭旧D经验scope，并输出实际编译mask；[Auto B2](D:/code/MPA-OpenCl/src/core/ecm_cuda_stage2_main.cpp:525)对非零构建在profile I/O前拒绝，避免套旧成本。cache payload v2目前本就使两边D模型回legacy；因此本轮不能声称“baseline已校准而candidate才回退”，新增算术guard保护后续标定。共享CLI默认mask0保持旧行为；发布893未替换。

两处stage2_tree_gpu.cu改动仅host guard/log，通过逐字节逆变换恢复原文件，混合换行未规范化。canonical PTX头与先前已通过百万对GMP/故障验证的编译快照逐字节相同。原生macro0/1使用同一套raw来源，差异由编译命令和实际worker日志确认；不能将相同来源误写成同二进制A/B。

### 原生正确性与资源

[原生矩阵](D:/code/MPA-OpenCl/tools/bench/bench_stage2_addsub.py:1)26次gate全部通过：M4423大界两条；三种unit位宽8193/16381/16384各2/65/66点；两种16384位不可逆回退；generic16384/D30030的66,305点（66,240点完整链＋65点ladder短尾）。小形状共22次调用，各含150组scaled GMP fixture与全部下降节点比较；unit叶hash另对独立CPU/GMP-ECM保存点的完整monic定义。大界全叶hash、因子和默认NTT/S4检查量与原生产参考一致。两条大块诊断开启全仿射/seed/段积核对，约212秒，属于正确性成本，未混入正式时间。

资源导出223个kernel。默认warp tile正逆均REG40/STACK0/LOCAL0；10个其他实例的资源字段改变，包括旧warp0 forward50→52/inverse40→38、M5 cooperative inverse46→48，以及旧outer M2/M3/M4/M5/M6的部分实例。M6旧inverse STACK512→504，不能声称全部stack零或所有资源保持；相同静态资源也不是指令/周期等价。后续实际trace需核对本轮执行的形状。

[控制门禁](D:/code/MPA-OpenCl/tools/test/check_stage2_addsub_controls.py:1)9项通过：production拒绝1/2/3、拒绝非法4、HostOnly拒绝混用mask、两种Auto B2路径及两个零曲线D规划。candidate明确拒绝未标定算术；baseline仍走原profile读取失败。所有对象/二进制前后SHA保持。首次控制采集因PowerShell5继承错误PSModulePath而无法找到Get-FileHash，保留已通过的四项与原失败日志；采集器重设为系统模块目录后，在新输出目录完整重跑。未改编译源码或放宽拒绝检查。

### 正式计时、实际诊断与决策

本阶段证据位于ignored `build_cuda_cmake/_stage2_sub_native_20261008`。两份development构建SHA256分别为baseline `9aa509480c98c566fee4b61f872b0afb7110812c7711c6ebdaae2537876e920e`、sub-only `82f50c2a26e8bf2e266984191e63287d6e3448ffe8e26f9e206766fa9b667a8b`。固定PTX3/outer策略2、V0、u0、pair1/CPUbase0/C64/min32768、owner640/reuse3/arena6300、GPU Gamma/root及默认必需检查；全部正式计时串行，未与编译、profiler或重型trace分析重叠。这里outer策略2指cooperative后端，文中旧记录的outer0/u0需按对应字段区分，不能混淆为关闭cooperative。

32正式/8预热，四种输入各先预热0/1，再按ABBA＋BAAB八条正式。所有样本保留，每版每形状仅四条独立曲线，没有置信区间：

- M4423大界，B1=1000/lcm/sigma26、D1381380/P126720/I1456028/B2=2011326186870：full均值37.6165975→36.9992900秒，少1.64105%；两组少1.55618%/1.72513%。
- generic8193，有效B1=20/sigma26、D30030/P2880/I32768：5.44383725→5.41804050秒，少0.47387%；两组少0.39256%/0.55472%。
- M16381，同宽矩阵几何：11.27921800→11.21972875秒，少0.52742%；两组少0.06652%/0.98493%。
- generic16384，同宽矩阵几何：23.72091350→23.70425600秒，少0.07022%；两组慢0.03572%/快0.17611%，未建立稳定收益。

两边完整叶指纹、factor、D/I及六项NTT/S4工作量/检查覆盖均一致，main精确账闭合误差≤0.003秒。宽矩阵使用不同模数，不能将位宽间绝对秒数当作同一N的伸缩实验。大界通过原生产参考，其他小门禁通过独立CPU/GMP定义；完整输出hash不是大规模逐系数独立GMP证明。

### 工作量、显存与后续

减法替换不改变卷积长度、batch、pass、逻辑蝶形数或PCIe接口量：单NTT逻辑蝶形为b·L·log2(L)/2，典型两forward＋一inverse的逻辑add/sub各为3b·L·log2(L)/2。候选没有新持久设备数组；owner仍为8W(7P+7)+48B，W=ceil(bits/64)、P=phi(D)/2。大界W70/P126720的owner为496746368B（473.734253MiB）；这是模块payload，不能与历史模块峰相加当进程峰。静态SASS省一条不能乘上述公式宣称动态周期节约。

本轮决策是保留sub-only开发候选，后续GPU下降frontier以它作为明确的新实验基线；默认mask0和独立生产算法保持，尚不推广add-only/both或outer V收窄。满16384位本批收益接近噪声，不能承诺普遍1.64%提速。最终生产移植要结合宽泛型和大形状复验。新D/Auto B2成本、较大16k形状容量、最终chain/短尾及多曲线总RAM/VRAM lease仍未完成。实际Systems、最终审计及归档记录见下面的收尾补充。

### 实际Systems、审计与归档收尾

GPU1管理员Systems2026.1.3顺序采集m0/m1，capture/export/collector均exit0；每侧完整leaf4244971527793015097、factor、有效save/Q及六项NTT/S4覆盖与该mask的正式矩阵一致，没有额外全量GMP读回。原始SQLite按完整demangled名字、gridXYZ/blockXYZ及dynamic shared核对：503组、102408次kernel完全一致；短名报告将不同实例合并，不能用其单条分组数量推导算法分派改变。

- tile：两边23877次，7.978363202→7.714583644秒，少3.30619%（0.263780秒）。
- cooperative outer：两边954次，4.369867188→4.335140000秒，少0.79470%。
- H2D两边4866次/7009115275B；D2H6130次/3138157112B；D2D40次/841498560B，次数与字节完全保持。
- 自身GPU事件span37.103305723→36.641967014秒，并集30.806552417→30.454416745秒，无自身事件6.296753306→6.187550269秒（16.97087%→16.88651%）。仍有约6.2秒准备/衔接窗口，不等于整卡idle，也不能把其变化全归因减法。
- 按实际malloc/free生命周期，两侧设备tracked payload峰4531652368B（4321.720474MiB），最终live0；pinned峰358886600B（342.260933MiB），末端357255360B由进程退出回收。这不是含driver/context/module的全显存峰，也不是进程RAM峰。没有新增本轮NVML采样或候选NCU计数。

Systems是单次插桩诊断，不替代前述未插桩32正式样本；约0.264秒tile改善不能解释全部full差。静态资源10/223实例变化保持披露，默认warp资源一致也不证明动态周期或完整SASS等价。

[独立审计工具](D:/code/MPA-OpenCl/tools/bench/audit_stage2_addsub.py:1)从原始日志重建门禁、所有正式/预热顺序、两组均值、完整输出/覆盖、成本保护及计时闭合；另核对29个raw依赖/5对象、两处混合换行修改逆变换、实际profile来源/命令结果、全部kernel几何/调用次数、PCIe字节和设备/pinned生命周期。最终quantitative.json为complete=true，保留首次控制环境失败和收尾前审计快照。归档evidence.zip/evidence_manifest.json保存原始来源、工具、输入、日志、矩阵和trace及逐文件SHA，exe/obj/DLL外部绑定。提交前后另核对raw源码与Git blob，根data/docs/data继续忽略；没有替换发布893。

复现先在新目录分别用builder `-Engine development -GlBackend ptx -OuterUnrollU 0 -AddSubMask 0|1 -SplitCompile 6`构建，使用`bench_stage2_addsub.py --baseline <m0-exe> --candidate <m1-exe> --reference <root-production-complete-matrix> --fixtures <wide-fixtures> --save <valid-m4423-save> --mode gate|timing|timing-wide --output <fresh-output>`，timing要求`--gate <completed-gate/measurements.json>`。输入、reference/fixtures及三个mode必须来自同一冻结组合。管理员profile工具增加`--add-sub-mask 0|1`验证实际编译选择；原生矩阵串行结束后再采集。审计命令为`audit_stage2_addsub.py --study <study> --profiles`，study目录布局及capture精确命令保存在本轮归档。
