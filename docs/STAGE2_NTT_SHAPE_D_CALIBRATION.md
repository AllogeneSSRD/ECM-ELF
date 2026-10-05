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

DPhaseModel抽出为共用header；独立CPU探针直接包含生产模型和实际NTT shape backend，2 profiles×2 bounds×8 D共64/0，比较精确N、owner、tree/inverse特征及各阶段/total估计，与整数Python特征和冻结fit一致。CPU探针不执行曲线、不分配GPU。[模型64项](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_shape_20261004/model_gate/summary.json)。

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
- [D模型与两组系数:12](D:/code/MPA-OpenCl/tools/bench/stage2_d_model.cuh:12)、[weighted unit:32](D:/code/MPA-OpenCl/tools/bench/stage2_d_model.cuh:32)、[成本:63](D:/code/MPA-OpenCl/tools/bench/stage2_d_model.cuh:63)、[scope与版本:10759](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10759)、[CPU选D计时:10938](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10938)。
- [整数features:66](D:/code/MPA-OpenCl/tools/bench/calibrate_stage2_d.py:66)、[fit锚点校验:31](D:/code/MPA-OpenCl/tools/bench/fit_stage2_d.py:31)、[排名:1](D:/code/MPA-OpenCl/tools/bench/plan_stage2_d.py:1)、[同exe Stage2 A/B:1](D:/code/MPA-OpenCl/tools/bench/bench_stage2_reduce_ab.ps1:1)。
- [策略/纯卷积probe:24](D:/code/MPA-OpenCl/tools/test/ntt_coop_outer_probe.cu:24)、[策略88项:62](D:/code/MPA-OpenCl/tools/test/ntt_coop_outer_probe.cu:62)、[GMP门禁:1](D:/code/MPA-OpenCl/tools/test/test_ntt_coop_outer.py:1)、[共用模型探针:1](D:/code/MPA-OpenCl/tools/test/stage2_d_model_probe.cu:1)、[模型64项:1](D:/code/MPA-OpenCl/tools/test/test_stage2_d_features.py:1)。
- [生产默认:15](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:15)、[生产构建:1](D:/code/MPA-OpenCl/tools/build/build_ecm_cuda_stage2.ps1:1)、[模型探针构建:1](D:/code/MPA-OpenCl/tools/build/build_stage2_d_model_probe.ps1:1)。
