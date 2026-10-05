# ECM CUDA Stage2：短归约后重新标定 D（2026-10-05）

本轮接续 [Goldilocks 短归约](D:/code/MPA-OpenCl/docs/STAGE2_GOLDILOCKS_SHORT_REDUCTION.md)，为新的 CUDA 算术成本建立 `resident_short_v1`。使用 GPU1 RTX4060 Laptop / sm89 / CUDA13.3；GPU0 的用户生产任务保持运行，未对其启动实验。编译、曲线计时和 GPU 验收串行执行。

## 1. 实现与范围

保留三个独立模型：profile0 为原 NTT、profile2 为尺寸策略加旧归约、profile3 为尺寸策略加短归约。新系数见 [stage2_d_model.cuh:27](D:/code/MPA-OpenCl/tools/bench/stage2_d_model.cuh:27)，新尺寸权重见 [32](D:/code/MPA-OpenCl/tools/bench/stage2_d_model.cuh:32)，实际 backend 查询和权重应用见 [50](D:/code/MPA-OpenCl/tools/bench/stage2_d_model.cuh:50)。模型不修改 NTT 精确性条件、打包位宽或显存形状。

实际选择器在 [stage2_tree_gpu.cu:10782](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10784) 匹配归约后端；短归约仅在已标定尺寸策略 scope 中使用新模型。旧归约仍使用对应旧系数；短归约加 outer0、强制 outer1、不同设备/N/B1/检查配置等均回 `legacy_56_1`。显式 D 优先，G≥2、owner 和 arena 过滤保持。

标定 scope：精确 N=2^4423−1、B1=1000、B2=1e11..2011326186870、单曲线、RTX4060 Laptop、t12/M4 策略入口、warp/compact、xADD6、resident roots/fold、batch64/chain64、sample96/check_every8，以及已测的 pool/oracle/carry 配置。这里的 M4 是配置入口，尺寸策略实际在 k24 选 M6、k25..27 选 M8。

## 2. 成本公式和计算量

约定 s=bit_length(N)=4423、W=ceil(s/64)=70、P=φ(D)/2、I=floor(B2/D)+2、G=ceil(I/P)，I=aP+r。每个多项式长度 m 的实际 NTT 长度记作 L(m)，避免与 ECM 模数 N 混淆。

backend 使用 slot_bits=2s+max(1,ceil(log2 m))，选择 bpw、sw=ceil(slot_bits/bpw)，满足 m·sw·(2^bpw−1)^2<q；q=2^64−2^32+1。L(m)=nextpow2(2m·sw+1)。这些仍由实际 multiply backend 决定；离线整数复现见 [calibrate_stage2_d.py:76](D:/code/MPA-OpenCl/tools/bench/calibrate_stage2_d.py:76)。

定义经验 NTT 工作特征：

```text
U3(m) = L(m)·log2 L(m)·w3(log2 L(m))
T3(P) = Σ[h=1,2,4,...; h<P] floor((P+h)/(2h))·U3(h+1)
V3(k) = Σ[Newton m=min(2m,k), 起点 m=1] 2·U3(m)
```

T3 包含 partial/unbalanced pairs；V3 计每次 Newton 扩张的两次乘法。使用特征与阶段正系数相乘得到估计秒数：baby=P·max(1,log2D−2)、affine=P、F树/下降=T3(P)、G树=aT3(P)+T3(r)、fold=(G−1)U3(P+1)、inverse=V3(P+1)、giant=I(6+22log2B2/64)、accum=P、glue=G。完整模型为上述阶段估计之和。

这些是经真实曲线拟合的工作特征，包含实际批处理、准备、同步与检查成本。它们不是 GPU 周期或精确 MAC 计数；不能用系数大小直接比较新旧内核周期。未获得新的有效 Nsight Compute 硬件计数，本轮不报实测 cycle/stall。

## 3. 在拟合前冻结尺寸权重

k16..23 新增同二进制 ABBA+BAAB；每尺寸 8 条、每模式 4 条，每条 1 warm+3 event 样本，计时外用 GMP 稀疏卷积参考比较全部 L 个输出。沿用上一阶段 k24..27 的冻结测量；probe SHA=`5bcdf7913232f1cbc6dd1c5c0cd2050ddcdf1d12adb4d76ed31c873d8577d756`。

w3(k)=w2(k)×[新短归约/同 binary 旧归约的纯卷积均值比]。k16..23 的 w2=1。k16..27 的 w3 依次为：

```text
.7929270231, .7355812422, .7235194677, .6392864988,
.6313090886, .6211608661, .6778661412, .8096173503,
.5948253971, .5687842601, .5722769388, .6134229919
```

未测 k<16 保留权重1，其小树/发射开销由阶段拟合吸收；不声称每个长度都已单独标定。新增 k16..23 的纯卷积收益约19.04%–37.88%，不直接换算完整曲线收益。权重在采多 D 曲线前冻结，没有根据 holdout 回调。

证据：[新小尺寸测量](D:/code/MPA-OpenCl/build_cuda_cmake/_short_d_20261005/pure_small/measurements.json)、[冻结权重](D:/code/MPA-OpenCl/build_cuda_cmake/_short_d_20261005/frozen_weights.json)、[原大尺寸测量](D:/code/MPA-OpenCl/build_cuda_cmake/_ntt_small_roots_20261005/volatile_full_sweep/measurements.json)。

## 4. 六条多 D 曲线与四条锚点

采集 binary 仍为冻结的 `17bf4813000e674a2046cb3765fa39e30085c816ea46fa0ac18b884fa78e58a2`，显式 D、short1/outer2；所有其他配置与上一轮短归约 A/B 相同。M4423、sigma26、extra12、B1=1000、B2=2011326186870，Stage1 Q SHA=`33cc6c63cec26c39900a7ed4ff551e2c2e0a533422905b538ec4f4aa809f092f`。

顺序570570/1381380/1411410/1411410/1381380/570570，完整 Stage2 秒数：

- D570570/P51840/I3525119/G68：86.877111、88.963678，均87.920395。
- D1381380/P126720/I1456028/G12：56.407772、56.879968，均56.643870。
- D1411410/P132480/I1425049/G11：61.097847、62.018886，均61.558367。

加入上一阶段同 binary、同控制的四条 short_fold D1231230 锚点，共10行。不同 D 改变 P/G、实际乘法和检查样本，不能将此组视为固定工作量的归约 A/B。拟合工具核对 binary SHA、完整 NTT 环境、设备、Stage1 Q 和 arithmetic/oracle 合同，拒绝旧归约锚点混入；实现见 [fit_stage2_d.py:31](D:/code/MPA-OpenCl/tools/bench/fit_stage2_d.py:31)。

十项正系数按 baby、affine、F树、giant、G树、fold、下降、inverse、accum、glue 顺序：

```text
3.426362377620879e-6, 2.6265051185028743e-5, 2.3193561077716357e-10,
3.719097111026145e-7, 7.676260404221605e-11, 2.5730236195427515e-10,
4.622033504014322e-10, 2.38168142424563e-10, 1.5229012671594509e-6,
0.03528230670537555
```

拟合内误差 −3.7765%..+0.8421%；leave-D-out −3.8804%..+6.7385%。冻结 fit SHA=`b137102de5095929f80408ecd02aa94ad11db479a162b7f171048d0a610f2484`。没有建立统计置信区间，也没有保证全局最优。

证据：[六条测量](D:/code/MPA-OpenCl/build_cuda_cmake/_short_d_20261005/calibration/measurements.json)、[冻结 fit](D:/code/MPA-OpenCl/build_cuda_cmake/_short_d_20261005/fit.json)、[阶段/容量量化](D:/code/MPA-OpenCl/build_cuda_cmake/_short_d_20261005/quantitative.json)。

## 5. 排名和独立 holdout

47-smooth D≤200000000 共379419候选。arena6300MiB、owner640MiB下，大界仍首选 D1381380/P126720，预测55.773088 s；小界首选 D330330/P31680，预测11.601560 s。owner512MiB时大界选 D1141140/P103680，预测60.352201 s；预算改变会重新排名。

拟合冻结后独立测试 B2=1e11，顺序330330/510510/510510/330330：12.555543、14.061516、14.693908、12.340467 s。330330 每条都快于510510，排序通过。预测误差分别 −7.60%、−7.79%、−11.76%、−5.99%，仍低估绝对秒数；未把 holdout 加入拟合。

证据：[大界排名](D:/code/MPA-OpenCl/build_cuda_cmake/_short_d_20261005/plan_large.json)、[小界排名](D:/code/MPA-OpenCl/build_cuda_cmake/_short_d_20261005/plan_small.json)、[512MiB排名](D:/code/MPA-OpenCl/build_cuda_cmake/_short_d_20261005/plan_budget.json)、[独立验证](D:/code/MPA-OpenCl/build_cuda_cmake/_short_d_20261005/holdout_summary.json)。

## 6. 阶段占比、显存与数据量

D1381380 六条标定中的两样本均值：init15.059702 s（26.59%），其中baby7.9745（14.08%）、CPU affine3.4555（6.10%）、剩余F树/检查/准备3.629702（6.41%）；main41.584168（73.41%）。main 内giant10.354（18.28%）、G树13.570（23.96%）、fold6.0545（10.69%）、下降7.6255（13.46%）、inverse2.0905（3.69%）、accum.1875（.33%）、残余桥接1.702168（3.01%）。括号分母均为full56.643870；init及其子项不重复相加。

根驻留 owner=8W(9P+8)+48 B。D1381380为638673328 B /609.086MiB；D1411410为636.772MiB，已接近640MiB预算；D570570为249.174MiB。这些是模型所约束的具体 payload，不是整个进程显存。

标定 D1381380 的完整 NTT arena payload 峰3341481200 B，主 A/B/Q workspace仍3221225472 B /3GiB；table和mandatory FuseCtx已包含在完整峰中。D570570完整峰1703162776 B；D1411410为3341742992 B。缓存/别名/点/owner/driver需按实际生命周期统计，不能直接把各池独立峰相加作为总VRAM。

D1381380 的 ALL S4 输出窗口逻辑returned_coeffs=36615543，包括保留在设备的输出；实际host窗口回读354851630 words，即2838813040 B，不能用逻辑系数总数×W重建回读字节。源码计数公式为8W·Σ[host_output调用] nbatch·output_slots，见 [3892](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:3892)；device/pinned窗口峰141926960/141927520 B。窗口回读字节不是整条曲线所有D2H。本轮没有为这些新D采集完整Systems copy字节或NVML进程峰。新模型本身不增加CUDA数据数组或传输；CPU扫描时间另报且排除于stage2_full_wall。

## 7. 模型接入实验版的实际入口验收

新D模型实验接入版 SHA=`085e00236f4c290252ce46cc2d0acbeacf671759265ed421bf2b0d9078857e1a`，compile580.1/link3.2 s；源快照、哈希见 [manifest](D:/code/MPA-OpenCl/build_cuda_cmake/_short_d_20261005/stage2_final/manifest.json)。与测量17bf版本的CUDA算术源相同，增加的是host模型。本节门禁和三条曲线均来自该冻结版本；后续OLDTAIL/S4_OFF守卫在生产版增补，见§8，不冒充该版本已包含增补。

- 共用 C++ 类/真实 NTT shape backend 与整数 Python 特征/冻结 fit：3 profiles×2 bounds×8 D，**48/0**。
- 实际选择器大界/小界/512MiB排名、显式D、特征、未支持配置和拒绝预算：**22/0**。
- 归约0/1、缺省0、模型关闭、outer0的匹配/回退：**36/0**。
- 最终 shared/warp 的独立 GMP 频谱、逆向、cache切换、寿命和LOCAL0：**14/0**。
- 自动 D 三条真实曲线：大界55.187779 s、小界12.259596 s、512MiB60.342262 s，全部clean1/GMP bad0/pending0；大界与小界最终叶FNV分别4244971527793015097和7549663880496122317。这是单次接入验收，不是性能 A/B。

其中512MiB D1141140没有参加拟合；单次预测60.352201与实际60.342262接近，不据此宣称所有D均能精确预测。上一轮完整188项来自冻结性能binary，此轮没有将22项planner门禁冒充重跑188项。

验收文件：[模型48项](D:/code/MPA-OpenCl/build_cuda_cmake/_short_d_20261005/model_gate/summary.json)、[planner22项](D:/code/MPA-OpenCl/build_cuda_cmake/_short_d_20261005/planner_gate/summary.json)、[scope36项](D:/code/MPA-OpenCl/build_cuda_cmake/_short_d_20261005/scope_gate/summary.json)、[NTT14项](D:/code/MPA-OpenCl/build_cuda_cmake/_short_d_20261005/ntt_gate/summary.json)、[三条自动曲线](D:/code/MPA-OpenCl/build_cuda_cmake/_short_d_20261005/actual_summary.json)。旧模型报告的64项笔误已按原始32行结果更正为32，原结果文件未改。

## 8. 生产默认提升

生产 wrapper 已加入 `NTT_GL_SHORT_REDUCE=1` 默认，实验入口无环境仍为0；显式环境0继续回旧归约。源码见 [ecm_cuda_stage2.cu:13](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:13)。生产验收与同binary A/B通过后已提升默认，原产物保留。

额外发现并复现原scope守卫只检查requested Mersenne标志：显式OLDTAIL=1仍启用拟合模型，而 [s4_launch_reduce:2262](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:2262) 实际选择Montgomery尾部。已在 [10763](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10763) 要求实际S4启用开关且OLDTAIL=0，匹配真正执行的S4路径。S4启用开关提前到 [10758](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10760)，并由后续对象构建复用。旧决策记录保留在 [before日志](D:/code/MPA-OpenCl/build_cuda_cmake/_short_d_20261005/oldtail_scope_before.log)。

初版增补误检查尚未建立的L.s4指针，生产守卫门禁28通过/2失败：两种正常归约均被错误回退，四种负配置正确回退。按初始化顺序修正为上述同一启用开关；失败候选及日志保留，未发布、未用于性能计时。候选目录 [production_wrong_pointer](D:/code/MPA-OpenCl/build_cuda_cmake/_short_d_20261005/production_wrong_pointer/build_manifest.json)，失败记录 [production_scope](D:/code/MPA-OpenCl/build_cuda_cmake/_short_d_20261005/production_scope/summary.json)。

最终生产二进制 SHA256 `e013932854d7b32c705fa0d755d444ff9efd614c318e85fc5a1d8c58fc1be331`，4127232 bytes；CUDA单元编译571.1 s，15项原始依赖SHA与构建前后源快照一致。全部生产计时使用该冻结binary；源文件在计时前后逐项核验。最终发布位置为 [ecm_cuda_stage2.exe](D:/code/MPA-OpenCl/build_cuda_cmake/production_stage2/ecm_cuda_stage2.exe)，原3CF38065…19F0E及DLL/manifest/signature保存在 [previous_production](D:/code/MPA-OpenCl/build_cuda_cmake/_short_d_20261005/previous_production/build_manifest.json)。

- save/ini/worktodo、冻结因子、队列失败保留、xADD/warp回退、自动大/小D及NTT/reducer回退：**33通过/0失败**。用户worktodo示例仍选择961–970，xxx仍拒绝。
- 修正后的生产选择器：short0/1各测正常、OLDTAIL1、S4_OFF1，**30通过/0失败**。正常分别启用resident_shape/resident_short，两类负配置均回legacy。此门禁只观察模型决策后受控停止直接worker，未完成曲线，不计作完整算术验收。
- 同production exe、同Stage1 save、固定D1381380、GPU1，四次串行ABBA（old/short/short/old）：full依次 **64.647779、56.586517、57.082008、65.380452 s**；均值 **65.014116→56.834263 s（快12.58%）**。每模式2样本，没有建立置信区间。

比较条件为N=2^4423−1、sigma26、B1=1000、B2=2011326186870、P126720；恢复存档SHA=`0fe48106563dc727c092f4baf7b4c0f2f3bd57ce3bae9fd9f8a5989f2ec324d4`，Stage1跳过。full包含init/main及mandatory检查，排除Stage1、CPU选D和父进程启动。四条最终叶值摘要 `leaves=126720 words=8870400 hash=4244971527793015097`、oracle采样/签名与因子列表完全相同，GMP bad0/pending0/clean1。显式D锁定计算量，此处量化归约开关收益；不将它与不同D或历史不同binary的收益相加。未为本次生产A/B重新采集全量PCIe字节/进程峰。

生产默认short1，PowerShell设置 `$env:NTT_GL_SHORT_REDUCE='0'` 可回旧归约和resident_shape模型；删除该环境覆盖则恢复生产默认。short1/outer0不在短模型scope内，回legacy；关闭D模型或显式D仍按既有优先级执行。实验exe默认继续short0。生产发布前后均验证exe及15项依赖，编译缓存signature也同步为已验收产物的signature。

证据：[最终33项](D:/code/MPA-OpenCl/build_cuda_cmake/_short_d_20261005/production_accept/summary.json)、[最终选择器30项](D:/code/MPA-OpenCl/build_cuda_cmake/_short_d_20261005/production_scope_fixed/summary.json)、[同exe存档A/B](D:/code/MPA-OpenCl/build_cuda_cmake/_short_d_20261005/production_ab/measurements.json)、[发布清单](D:/code/MPA-OpenCl/build_cuda_cmake/_short_d_20261005/promotion.json)、[A/B驱动源码](D:/code/MPA-OpenCl/tools/bench/bench_stage2_save_reduce.py:1)、[生产scope门禁源码](D:/code/MPA-OpenCl/tools/test/test_stage2_production_scope.py:1)。


## 9. 下一瓶颈：先处理可移到 GPU 的 baby 准备

对上一轮 short1 / D1231230 Systems SQLite 重新分解事件并集：总无本进程GPU事件间隙9.513572 s，其中copy→copy 7.625119 s /6401段；最大2.784113 s发生在两个64512000 B X/Z回读后、下一次560 B H2D之前。下一copy的runtime API仅112μs，直到空隙末尾才提交，不能将2.78s解释为其GPU copy耗时。

源码 [stage2_tree_gpu.cu:11176](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:11176) 的baby ladder之后，CPU执行256点分段prefix/product inversion、反向传播和负仿射叶物化，再构建F树。结合数据形状，最大空隙与这一准备阶段关联；原 trace 未捕获CPU栈，属于源码推断，不是已测迁移收益。[关联定位数据](D:/code/MPA-OpenCl/build_cuda_cmake/_short_d_20261005/previous_gap_locations.json)。

下一轮优先复用 giant 的设备段积/组逆元/叶生成方法处理baby，维持非可逆Z的GCD因子记录、Z=0语义、small-prime cache和检查覆盖。X/Z读回量为16PW B，D1231230约123.047MiB，D1381380约135.352MiB；目标减少这些回读和CPU大整数模乘，实际节省要以新同binary A/B确认。随后继续gl_mod直接规范化和新后端上的低层根复测。

复用边界：现有 [s2g_projective_leaf_kernel:1125](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:1125) 写的是`[-X,Z]`，giant通过Gamma补偿保持等价；baby当前写`[-X/Z,1]`，F树后续使用其monic约定。不能直接把projective叶塞进F树。候选应复用 [设备段积:1087](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:1087) 和 [group单位性/坏组处理:8999](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:8999)，补充GPU prefix/逆元传播，输出与原CPU路径逐字相同的负仿射常数和leading1；仅坏组读回做既有GCD/逐点处理。第一版可只回读P个常数再复用原F树frontend，随后再接设备叶descriptor；不能在未实现接口前声称已消除全部叶传输。

Tensor真tile和双流实验此前为负，不提升默认；多曲线仍需独立状态、共享NTT workspace lease及RAM/VRAM预算，当前单曲线arena3GiB以外仍有大量状态。相同N/Q/B1/B2/覆盖/线程的Prime95新对照尚未完成，长期目标继续；本报告没有证明已超过CPU实现。
