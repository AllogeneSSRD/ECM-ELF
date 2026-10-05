# ECM CUDA Stage2：固定 PTX 后端的 D 重标定

日期：2026-10-05；起点 `0e6b238`。接续[固定NTT后端实验](D:/code/MPA-OpenCl/docs/STAGE2_NTT_FIXED_BACKEND.md)。GPU1 RTX4060 Laptop/sm89/CUDA13.3；编译、CPU参考、GPU测量串行。GPU0用户生产未调度实验。

## 1. 后端与输入

新增 profile5 / `resident_fixed_ptx_v1`，匹配 fixed_mode3、short1、GPU baby、shape outer2、warp1、t12、compact scratch、xADD6/resident/scaled descent/检查。原 profile0/2/3/4 数值保留；运行时PTX、固定fold/short及逆移位归一化没有匹配新fit，回legacy。固定short版本不能因数学相同就沿用新PTX成本。

适用范围仍为精确 `n_ECM=2^4423−1`、B1=1000、B2=1e11..2011326186870、单曲线、已测设备/路径/预算。其他范围沿用原选择器，显式D优先。经验模型不保证全局最优。

实际采样exe `9abd6ec69286d95971467924bdc5434e09b98af5ecce760588d0a8ae6494215f`（上一阶段固定PTX9ABD），18个原始编译依赖冻结。直接native curve worker恢复同一Stage1存档，Stage1跳过；SIGMA26，Q完整输出与独立解析后的canonical X摘要 `33cc6c63cec26c39900a7ed4ff551e2c2e0a533422905b538ec4f4aa809f092f`匹配。save SHA `0fe48106563dc727c092f4baf7b4c0f2f3bd57ce3bae9fd9f8a5989f2ec324d4`。不同D的叶/样本集合会变化，各自必须通过GMP及oracle；未混入CPU baby或回退数据。

首次采样缺少内部worker必需的results路径，在GPU计算前exit2。采样器已补齐；该失败保留在calibration目录，**只有calibration_v2和anchors计入fit**。旧 `restored_X=1` 是恢复开关，不是坐标值1。

## 2. 先冻结 NTT 权重

先测 k16..27 的完整卷积（两forward、pointwise/scale、inverse），再跑任何拟合曲线。与冻结的 `2cece93` 普通短归约探针交叉，选择其shift0模式；fixed/PTX每进程8次，reference每进程取4次目标模式，各次warm+3 events，全部N输出在计时外检查。每长度2个目标进程/后端，无统计置信区间。

定义 `w5(k)=w_short(k)·t_fixed_PTX(k)/t_frozen_short(k)`，它是经验工作特征权重，不是周期数。十二个冻结值，按k16..27：

```text
0.641798297381, 0.621488905536, 0.595288343155, 0.561513562655, 0.551829905043, 0.549799329834, 0.637294350050, 0.772292659850, 0.545212739922, 0.519145540711, 0.523261500537, 0.548791338928
```

k小于16保留权重1；未对这些小尺寸另做性能证明。NTT权重JSON绑定原始测量SHA并复算全部比例，curve的四个公共NTT源码依赖必须与权重探针相同。禁止从完整曲线误差反向调整权重，禁止将其他后端/源码/设备/Q/控制的锚点混入。

本轮与冻结short参考相比，k24..27完整卷积改善8.34/8.73/8.56/10.54%；不同测量序列的绝对时间不与上一报告拼成额外生产收益。固定/short自身及GMP/cache/legacy/fault/config门禁16/0。

证据：[全部尺寸和原始交叉](D:/code/MPA-OpenCl/build_cuda_cmake/_fixed_d_20261005/ntt_weights_measurements/measurements.json)、[冻结权重](D:/code/MPA-OpenCl/build_cuda_cmake/_fixed_d_20261005/ntt_weights.json)。

## 3. 阶段公式与拟合

记s=bit_length(n_ECM)，W=ceil(s/64)，P=φ(D)/2，I=floor(B2/D)+2，G=ceil(I/P)，I=aP+r。实际backend的NTT长度L(m)来自整数exactness约束：slot_bits=2s+max(1,ceil(log2m))，sw=ceil(slot_bits/bpw)，最大bpw满足 `m·sw·(2^bpw−1)^2<q_GL`，并按实际打包长度取L(m)。C++生产通过真实shape query，不以浮点估算容量。

```text
U5(m) = L(m)·log2 L(m)·w5(log2 L(m))
T5(P) = Σ[h=1,2,4,...;h<P] floor((P+h)/(2h))·U5(h+1)
V5(k) = Σ[Newton m=min(2m,k),起点m=1] 2·U5(m)

baby    = αb·P·max(1,log2D−2)
affine  = αa·P
F树     = αF·T5(P)
giant   = αg·I·(6+22log2B2/64)
G树     = αG·[aT5(P)+T5(r)]
fold    = αf·(G−1)·U5(P+1)
下降    = αd·T5(P)
inverse = αi·V5(P+1)
accum   = αc·P
glue    = αl·G
full_est = 上述各项之和
```

这些是经验阶段特征与秒数，包含准备/分配/同步/必要检查，不是精确模乘数量或GPU周期。真实模乘计数、SOS/REDC依赖MAC与NTT流量公式沿用[步骤报告](D:/code/MPA-OpenCl/docs/STAGE2_GPU_CURRENT_PIPELINE.md)及[固定后端说明](D:/code/MPA-OpenCl/docs/STAGE2_NTT_FIXED_BACKEND.md)。本阶段改成本模型，未减少算术或传输。

六条D曲线加四条同D锚点，顺序正/反交叉：

- D1141140，P103680：54.944820/53.834027 s，均值54.389423s。
- D1231230，P115200：52.750307/51.375549 s，均值52.062928s。
- D1411410，P132480：53.850955/53.724714 s，均值53.787835s。
- D1381380，P126720：48.238670/48.266274/48.205029/48.261996 s，均值48.242992s。

P132480跨过131072的树高度边界，模型包含对应实际NTT尺寸台阶。逐阶段正系数最小二乘（零截距）拟合；fit十条的full误差约−3.70%..+1.94%，leave-D-out约−4.18%..+2.19%。B2留出不参加fit。[冻结fit](D:/code/MPA-OpenCl/build_cuda_cmake/_fixed_d_20261005/fit.json)、[复算阶段/容量](D:/code/MPA-OpenCl/build_cuda_cmake/_fixed_d_20261005/calibration_quantitative.json)。

## 4. 排名、预算与独立留出

枚举379419个47-smooth候选D，arena6300MiB/owner640MiB/baby512MiB过滤后，大界28500个、小界27300个。大界前两名D1381380/1360590，预测48.883602/49.559928s；小界前两名D390390/330330，预测10.433277/10.477641s。预算改owner512MiB时推荐D1141140，baby临时128MiB时推荐D600600。

在fit与权重冻结后，独立B2=1e11：

- D390390/330330/330330/390390：10.415344/10.409339/10.404906/10.365599s。均值10.390472/10.407123s；均值排序一致，但单条范围重叠，差约0.16%，**近似持平，未证明微小收益**。
- 另D390390/510510/510510/390390：10.322154/11.075424/11.084740/10.419581s；均值10.370868/11.080082s，前者快约6.40%，每条390390均快于每条510510。模型预测390390为10.433277s、510510为10.870066s。

两组分别保留，未用留出重拟系数，也不将它们合并为CI。小界留出的预测误差约−1.94%..+1.08%。大界最佳D仍1381380，本阶段不宣称D重标定本身创造了额外大界加速。

容量公式不变：

```text
owner_bytes = 8W(9P+8)+48
c0=P; c_l=ceil(c_(l−1)/2), l=1..8
baby_payload = 8[(3P+5)W+P+WΣ_l c_l]+c8
baby_cap = min(configured_cap, max(0,live_free−64MiB))
```

D1381380 owner638673328B、baby284592655B；D390390相对D330330有更大的P与owner/baby需求。硬预算先于模型，实际分配失败仍回退。NTT arrays、H2D/D2H/D2D、CPU数据生成的算法增量0B；新增host只读系数约176B（10个rate+12个weight double，未含对齐）。本轮没有重新采集NVML/PCIe/host-private峰，explicit payload不包含CUDA context/隐式stack，不能相加为VRAM总峰。

证据：[大界](D:/code/MPA-OpenCl/build_cuda_cmake/_fixed_d_20261005/plan_large.json)、[小界](D:/code/MPA-OpenCl/build_cuda_cmake/_fixed_d_20261005/plan_small.json)、[owner512](D:/code/MPA-OpenCl/build_cuda_cmake/_fixed_d_20261005/plan_fold512.json)、[baby128](D:/code/MPA-OpenCl/build_cuda_cmake/_fixed_d_20261005/plan_baby128.json)、[近似持平留出](D:/code/MPA-OpenCl/build_cuda_cmake/_fixed_d_20261005/holdout_summary.json)、[较大差距留出](D:/code/MPA-OpenCl/build_cuda_cmake/_fixed_d_20261005/holdout_coarse_summary.json)。

## 5. 验证边界和实现索引

CPU模型80/0：profile0/2/3/4/5 × 两个B2 × 八个D，与独立整数Python features、冻结fit及真实C++NTT shape对照。输入拒绝22/0：binary/device/Q/source/weights/evidence、CPU baby/runtime PTX/历史profile、修改phase/features和伪造backend raw log均拒绝。

实际native planner另9/0，核对大/小界autoD、两预算及五个显式D的完整features/cost；selector90/0覆盖旧tail/off、baby回退/诊断、outer/M/warp/compact/scale/baby/xADD覆盖。两者都在观察决策后早停，**不是完整曲线或算术验证**。

- [profile5系数与权重](D:/code/MPA-OpenCl/tools/bench/stage2_d_model.cuh:40)、[weighted unit](D:/code/MPA-OpenCl/tools/bench/stage2_d_model.cuh:73)。
- [实际后端guard](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10788)、[模型选择](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10805)。
- [采样与冻结权重](D:/code/MPA-OpenCl/tools/bench/calibrate_stage2_d.py:32)、[fit](D:/code/MPA-OpenCl/tools/bench/fit_stage2_d.py:1)、[排名/过滤](D:/code/MPA-OpenCl/tools/bench/plan_stage2_d.py:1)。
- [输入22项](D:/code/MPA-OpenCl/tools/test/test_stage2_fixed_d_inputs.py:1)、[C++模型80项](D:/code/MPA-OpenCl/tools/test/test_stage2_d_features.py:1)、[实际planner9项](D:/code/MPA-OpenCl/tools/test/test_stage2_save_d_plan.py:1)、[scope90项](D:/code/MPA-OpenCl/tools/test/test_stage2_production_scope.py:1)。

证据：[模型](D:/code/MPA-OpenCl/build_cuda_cmake/_fixed_d_20261005/model_gate/summary.json)、[输入](D:/code/MPA-OpenCl/build_cuda_cmake/_fixed_d_20261005/fit_input_gates/summary.json)、[实际planner](D:/code/MPA-OpenCl/build_cuda_cmake/_fixed_d_20261005/save_plan_gate/summary.json)、[selector](D:/code/MPA-OpenCl/build_cuda_cmake/_fixed_d_20261005/scope_gate/summary.json)。


## 6. 最终 native 与发布包验收

最终集成可执行文件 SHA256 `dcf70b11d51597bac2a9a1c62132d54b549ffa689df55320f031f33222c289d4`，3708416B，CUDA13.3/sm89/fixed3。CUDA编译284.9s，三组C++编译约4.0/2.5/3.2s，链接约2.9s。18项原始源码依赖与独立sources快照逐项核验；训练9ABD的host模型关闭，最终DCF改变host成本表/guard，四个NTT共同依赖一致。

完整native save/ini/worktodo/因子/CRC/不同sigma/失败队列保持/warp及xADD回退/真实M4423/自动D验收 **30/0**。这里包含真正GPU曲线，与§5的早停selector检查分别计数。自动大界D1381380完整48.527513s、小界D390390完整10.396450s；扫描分别约.140231/.140622s，未包含在stage2_full_wall内。小界实际leaf hash `1689529688547722991`。用户worktodo例子B2=26000000000/skip960/count10对应记录961..970；尾随xxx不是合法数值因子，拒绝并保留队列。

复制到production_stage2之后，重新针对**发布路径**运行入口21项和自动小界完整曲线，共 **22/0**，核对exe、DLL、18源码及leaf/oracle/GMP。发布SHA与被验收DCF相同；旧A191的exe、DLL、manifest及17份raw源码保存在previous_production，可独立恢复。

证据：[native30](D:/code/MPA-OpenCl/build_cuda_cmake/_fixed_d_20261005/native_accept/summary.json)、[发布包22](D:/code/MPA-OpenCl/build_cuda_cmake/_fixed_d_20261005/published_accept/summary.json)、[源码冻结](D:/code/MPA-OpenCl/build_cuda_cmake/_fixed_d_20261005/final_native_provenance.json)、[发布manifest](D:/code/MPA-OpenCl/build_cuda_cmake/production_stage2/build_manifest.json)。

## 7. 集成后完整曲线 A/B 与时长占比

同实际存档、D1381380、B2=2011326186870、GPU1、默认检查、shift0，顺序old/new/new/old/new/old/old/new；旧生产A191与最终DCF逐进程串行，模型关闭以固定几何。

- 旧A191：49.886689 /49.909926 /49.913762 /49.889685s，均值 **49.900015s**。
- 新DCF：48.332485 /48.310479 /48.328689 /48.372032s，均值 **48.335921s**。
- 减少 **3.1345%**；第一组ABBA3.1601%、第二组BAAB3.1088%。每后端4样本，无CI；不将上一阶段3.4852%相加，锚点48.24s也不是新算法收益。

均值init 10.614616s（21.96%）、main 37.721305s（78.04%）。init包含baby/F-tree及mandatory selftests；main包含GCD、命名和oracle drain；Stage1跳过。shape约.035s、D扫描约.11s另计。下列取2_new单条48.332485s作阶段说明：

- init10.633633s，占22.00%；giant10.339s，占21.39%；G树12.611s，占26.09%。
- fold5.764s，占11.93%；descent6.066s，占12.55%；inverse1.559s，占3.23%；accum.152s，占.31%。
- 分项可能嵌套或有未分配尾部，**不能求和当作另一完整总时间**。主循环NTT调用侧23.982s，占main63.6%；它含准备/同步，不是纯NTT kernel时长或GPU利用率。

8条结果均leaf `4244971527793015097`、oracle signature `c85031f6149bae11`、factors空、bad0。算术覆盖均397次S4 launch、1836241次poly mul、36615543个reduced coeff，2400 primitive cases、60474 GMP samples、3次full checks；oracle1010个selected/queued/compared任务，60474 samples，pending0/clean1。固定后端消除动态分派，不改变多项式模乘次数或NTT尺寸。

2_new记录workspace `full_peak_bytes=3341481200`（workspace/table/compact base的逻辑容量），arena6149MiB/overflow0，fold owner638673328B。这些分配可能共享arena或处于不同生命周期，**不能相加得到总VRAM峰**。device_gleaf group D2H798000B，fold实传H2D141928872B/D2H70963304B；root交接avoided H2D/D2H各815382400B。这是原有驻留实现的记录，本阶段未删除这些传输。

证据：[8条汇总](D:/code/MPA-OpenCl/build_cuda_cmake/_fixed_d_20261005/final_ab/measurements.json)、[单条原始记录](D:/code/MPA-OpenCl/build_cuda_cmake/_fixed_d_20261005/final_ab/2_new_engine.log)、[时长均值](D:/code/MPA-OpenCl/build_cuda_cmake/_fixed_d_20261005/final_timing_summary.json)。

## 8. 复现、回退与后续优先级

当前发布：[ecm_cuda_stage2.exe](D:/code/MPA-OpenCl/build_cuda_cmake/production_stage2/ecm_cuda_stage2.exe)。编译脚本默认仍runtime；重建这个发布后端需要明确 `-GlBackend ptx`，仅设置环境变量不等价于编译特化。

```powershell
tools/build/build_ecm_cuda_stage2.ps1 -Build build_cuda_cmake/reproduce_fixed_ptx -Arch sm_89 -GlBackend ptx -Rebuild
build_cuda_cmake/production_stage2/ecm_cuda_stage2.exe --save YOUR_STAGE1.save --b2 2011326186870 --device 1 --results results.jsonl
```

固定PTX不接受将short/PTX有效设置改为另一后端；冲突exit2。历史后端比较/回退使用previous_production中的A191，或重建runtime并选对应环境模式。逆归一化shift仍默认0。超出经验模型scope、诊断/shape覆盖/分配失败沿用保护，显式D仍优先。

六模乘xADD已在此前阶段落地，本轮native再次覆盖1280案例与显式0回退：[xADD6数学与实现索引](D:/code/MPA-OpenCl/docs/STAGE2_XADD_D_OPTIMIZATION.md)。当前point SOS+REDC仍约2W² MAC/模乘；NTT热分派减少后，下一轮先采样giant/CPU准备空隙，再按证据推进点MAC与NTT kernel。多曲线吞吐需先建立curve私有缓冲与NTT workspace lease、RAM/VRAM上限，不能直接并发拥有整套6GiB arena。Tensor整数tile此前负结果保留；公平Prime95同Q/B2/线程完整对照尚待完成，本轮不宣称超过CPU。
