# ECM CUDA Stage2：GPU baby 后重新标定 D（2026-10-05）

接续 [GPU baby 批量归一化](D:/code/MPA-OpenCl/docs/STAGE2_GPU_BABY_NORMALIZATION.md)。本轮为实际 GPU 准备路径建立 `resident_baby_v1`，保留原 CPU baby 的模型和回退。设备为 GPU1 RTX4060 Laptop / sm89 / CUDA13.3；曲线测量、编译与 GPU 验收串行执行。

## 1. 模型和计算量

新模型为 profile4，系数见 [stage2_d_model.cuh](D:/code/MPA-OpenCl/tools/bench/stage2_d_model.cuh:35)。profile0/2/3 分别继续表示原 NTT、尺寸策略、尺寸策略加短归约；profile4 沿用已冻结的短归约 NTT 权重。这里没有改 NTT 算术、打包位宽、尺寸策略或点内核。

约定 s=bit_length(N)=4423，W=ceil(s/64)=70，P=φ(D)/2，I=floor(B2/D)+2，G=ceil(I/P)，I=aP+r。每个多项式长度 m 的实际 NTT 长度记作 L(m)。backend 的精确 Goldilocks 卷积约束和 L(m) 计算见 [上一轮公式](D:/code/MPA-OpenCl/docs/STAGE2_SHORT_REDUCTION_D_CALIBRATION.md)。

```text
U4(m) = L(m)·log2 L(m)·w3(log2 L(m))
T4(P) = Σ[h=1,2,4,...; h<P] floor((P+h)/(2h))·U4(h+1)
V4(k) = Σ[Newton m=min(2m,k), 起点m=1] 2·U4(m)

baby    = αb·P·max(1,log2D−2)
affine  = αa·P
F树     = αF·T4(P)
giant   = αg·I·(6+22log2B2/64)
G树     = αG·[aT4(P)+T4(r)]
fold    = αf·(G−1)·U4(P+1)
下降    = αd·T4(P)
inverse = αi·V4(P+1)
accum   = αc·P
glue    = αl·G
full_est = 上述各项之和
```

这些是实际阶段成本的经验工作特征，计入批处理、准备、分配、同步和必需检查；不是 GPU 周期或精确指令数。GPU baby 的精确域证明、模乘数量及 SOS/REDC 主循环 MAC 公式见 [归一化报告§2–4](D:/code/MPA-OpenCl/docs/STAGE2_GPU_BABY_NORMALIZATION.md)。本轮改变的是成本模型和默认路径接入，不能将拟合系数变化说成新的内核提速。

模型适用范围仍为精确 N=2^4423−1、B1=1000、B2=1e11..2011326186870、单曲线、已测 RTX4060 Laptop/尺寸策略/short1/xADD6/resident/check 配置。未支持的 N、B1、设备或配置回 legacy；GPU baby 本身的正确性和回退适用范围与模型标定范围分别验收。

## 2. 内存约束与路径匹配

选择器源码见 [路径scope](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10765)、[实际模型匹配](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10790)、[候选过滤](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10829)。临时payload公式实现见 [d_baby_payload_bytes](D:/code/MPA-OpenCl/tools/bench/stage2_d_model.cuh:47)。

根驻留 owner 为 `8W(9P+8)+48` B。新加入 GPU baby 的显式临时 payload 过滤。记 c0=P，c_l=ceil(c_(l−1)/2)，Gb=c8，T=Σ[l=1..8]c_l，则：

```text
baby_payload = 8[(3P+5)W+P+TW]+Gb  B
baby_cap = min(NTT_BABY_DEVICE_MAX_MB·2^20, max(0,live_free−64MiB))
```

Gb 是归一化组数，与 Stage2 的 G 多项式组数不同。候选同时受 owner、arena 和 baby_cap 硬过滤；显式 D 超出 baby_cap 时拒绝使用 GPU baby 经验成本。原始 payload 公式可独立检查，不包含 driver/context 和隐式线程 stack，也不能代替完整 VRAM 生命周期账本。分配时仍再次检查 free，并在实际 OOM 时回原路径。

`NTT_BABY_DEVICE=0` 与对应 reducer/NTT 组合继续使用 profile0/2/3。GPU baby 配 short0、outer0/1、预算为零、分配失败注入或 baby 的 CHECK/TEST/BAD 诊断开关均拒绝 profile4。低预算导致的 CPU 回退不能被当作 GPU 准备测量。

## 3. 十条拟合样本和冻结系数

冻结性能 binary SHA256 `a5addcd9229cf61a6c1203142abdf2b25323904968b4ef4644506fed9b30e7f8`。M4423、sigma26、extra12、B1=1000、B2=2011326186870、GPU1；完整 Stage1 Q 的 SHA256 为 `33cc6c63cec26c39900a7ed4ff551e2c2e0a533422905b538ec4f4aa809f092f`。Stage1 排除于 full，固定显式 D。

新增六条顺序 570570/1231230/1411410/1411410/1231230/570570：

- D570570：85.304260、84.986359 秒。
- D1231230：55.975749、55.095618 秒。
- D1411410：58.108267、59.201748 秒。

加入上一轮同 binary、同控制 D1381380 的四条已验证 GPU baby 样本：52.814794、52.528683、54.949292、52.885733 秒。CPU baby 对照行未进入拟合。每曲线核对 binary 和8项编译依赖前后 SHA、完整 Q、实际 baby/reducer 路径、GMP bad0、pending0、clean1 和 oracle 覆盖。

十项正系数按 baby、affine、F树、giant、G树、fold、下降、inverse、accum、glue 顺序：

```text
3.393465300372242e-6, 2.136039841029098e-6, 2.1215649385492397e-10,
3.7066724433274104e-7, 7.596846294547282e-11, 2.576800408929496e-10,
4.5033088482375454e-10, 2.4903937851620365e-10, 1.5104078895629704e-6,
0.03580121548117153
```

拟合内误差 −5.6259%..+0.9791%，leave-D-out −5.7012%..+8.8550%。fit SHA256 `b1a1f3e176ffeb7a78dc019f095407e15ff62f642b54077ea59b815544b03ad0` 在独立验证前冻结，没有按 holdout 回调，也没有建立置信区间。

证据：[六条曲线](D:/code/MPA-OpenCl/build_cuda_cmake/_baby_d_20261005/calibration/measurements.json)、[四条 GPU 锚点](D:/code/MPA-OpenCl/build_cuda_cmake/_baby_d_20261005/anchors.json)、[冻结 fit](D:/code/MPA-OpenCl/build_cuda_cmake/_baby_d_20261005/fit.json)。拟合工具核对 raw log 与阶段数据，以及相同 binary/device/Q/controls/sources；不同 D 改变 P/G、覆盖几何和检查集合，不能把多 D 数据当作固定计算量的 A/B。

## 4. 排名和独立 B2 验证

47-smooth D≤200000000 共379419候选。在 arena6300MiB、owner640MiB、baby512MiB下，大界首选 D1381380/P126720，预测52.068767秒；小界 B2=1e11 首选 D330330/P31680，预测10.705667秒。owner512MiB时大界首选 D1141140/P103680，预测57.267041秒。baby临时预算单独降至128MiB时，大界首选D600600/P57600，预测81.127150秒；该项仅验证实际选择器预算过滤，没有完成其曲线。与原短模型相比，当前三个预算/边界的首选 D 相同；本轮没有额外的换 D 性能收益。

拟合冻结后独立小界，顺序330330/510510/510510/330330：11.349224、12.558380、12.383793、11.910535秒。每条330330都快于每条510510，排序通过；均值分别11.629880和12.471087秒。绝对预测仍低估5.67%–10.12%。模型用于排名，没有证明379419候选的全局实际最优。

证据：[大界](D:/code/MPA-OpenCl/build_cuda_cmake/_baby_d_20261005/plan_large.json)、[小界](D:/code/MPA-OpenCl/build_cuda_cmake/_baby_d_20261005/plan_small.json)、[owner512](D:/code/MPA-OpenCl/build_cuda_cmake/_baby_d_20261005/plan_budget.json)、[独立 holdout](D:/code/MPA-OpenCl/build_cuda_cmake/_baby_d_20261005/holdout_summary.json)。

## 5. 当前验收与生产发布

- C++共用模型/真实NTT shape backend，与整数Python/冻结fit，4 profiles×2 bounds×8 D：64通过/0失败。profile4还检查临时payload公式。
- 拟合输入正常重现，以及 binary/device/Q/sources、CPU controls、伪造阶段数据、CPU raw log的拒绝：8通过/0失败。
- 首次模型比较误传了历史未发布的 shape fit.json，profile2比较失败。换成文档指定的冻结 fit_final.json后64项通过；原失败目录保留，属于验收输入选择错误，未修改旧模型系数。

证据：[模型64项](D:/code/MPA-OpenCl/build_cuda_cmake/_baby_d_20261005/model_gate_final/summary.json)、[输入8项](D:/code/MPA-OpenCl/build_cuda_cmake/_baby_d_20261005/fit_input_gate/summary.json)。

最终生产 binary SHA256 `a1910cb47d4a4d567fed441a9f5002f1b0ba1675c376c095035ee01c1456ec6a`，4249600 B。CUDA单元590.2秒，整个构建驱动611.875秒；17项编译依赖和原始源快照在构建及每轮验收前后核对。NTT CU、cooperative/short-reducer header、baby device/host header与冻结性能A5版本相同；变化为host D模型、预算scope和wrapper默认。

- save/ini/worktodo、冻结因子、队列失败保留、warp/xADD/reducer回退和自动大小界：**33通过/0失败**。用户示例仍选961–970，xxx仍拒绝。
- 旧CPU baby模型scope：**30通过/0失败**；GPU baby的short0/1、OLDTAIL/S4_OFF和所有baby诊断/预算拒绝：**80通过/0失败**。这些观察模型决定后早停直接worker，不冒充完整曲线。
- 原生save入口的自动大/小界、owner512、baby128与5个显式D特征/阶段估计：**9通过/0失败**。也是早停planner门禁。
- 实际自动大界D1381380 **51.037087秒**，自动小界D330330 **10.834302秒**；已恢复save、跳过Stage1、leaf/GMP/pending/clean及因子结果符合基线。这两条为接入验收，不是A/B。
- 未参与拟合的D1141140，owner512MiB真实恢复曲线 **56.421647秒**，预测57.267041（+1.50%）；最终叶FNV12504726358590477869同原参考，GMP bad0/pending0/clean1。

同production exe、同Stage1 save、固定D1381380、GPU1、short1及同默认/检查配置，串行CPU/GPU/GPU/CPU：**54.305596、50.745340、50.295112、52.832571秒**。均值 **53.569084→50.520226秒，快5.69%**；每模式2样本，没有建立置信区间。init13.145795→10.872972，main40.423289→39.647254；baby ladder7.972000→7.902500，affine2.096000→.240500秒。main也有波动，没有修改point/NTT内核，不将其波动解释为新的内核收益。

save SHA256 `0fe48106563dc727c092f4baf7b4c0f2f3bd57ce3bae9fd9f8a5989f2ec324d4`，N/Q/B1/B2/D及检查覆盖固定；每条叶FNV4244971527793015097、oracle signature c85031f6149bae11、采样/因子列表相同，small-prime cache matched1。full排除Stage1、选D扫描与父进程启动。原实验A5八条的6.99%与本次不同入口A191四条的5.69%分别报告，不相加，也不交叉拼接均值。

发布位置：[ecm_cuda_stage2.exe](D:/code/MPA-OpenCl/build_cuda_cmake/production_stage2/ecm_cuda_stage2.exe)。生产wrapper默认 `NTT_BABY_DEVICE=1`，显式0回原CPU归一化和对应原模型；实验exe未加默认。发布前的E013 exe/DLL/manifest/signature保存在本阶段previous_production，便于恢复。GPU baby默认提升由上述完整验收和配对A/B支持；预算不足或OOM仍可回原路径。自动选择器不能保证外部显存占用变化后的GPU路径，实际enabled状态需查看日志。

本轮模型不增加新设备数组或传输。D138的GPU baby临时payload **284592655 B**，正常D2H **71240400 B**，seed/mask H2D **277695 B**；owner **638673328 B**。这些有不同生命周期，不能相加当作VRAM峰。本轮未重采完整PCIe/NVML/host-private峰，之前A5的Device分配账本4.8165→4.6736GB属于那两条profile。D114的budget曲线arena缓存账本曾到6148MiB，owner512不等于整个进程显存上限；多曲线仍须按实际时间上的所有权约束设计。

证据：[生产构建冻结清单](D:/code/MPA-OpenCl/build_cuda_cmake/_baby_d_20261005/production/frozen_manifest.json)、[33项入口](D:/code/MPA-OpenCl/build_cuda_cmake/_baby_d_20261005/production_accept/summary.json)、[旧scope30](D:/code/MPA-OpenCl/build_cuda_cmake/_baby_d_20261005/scope_0/summary.json)、[新scope80](D:/code/MPA-OpenCl/build_cuda_cmake/_baby_d_20261005/scope_1_final/summary.json)、[原生planner9](D:/code/MPA-OpenCl/build_cuda_cmake/_baby_d_20261005/planner/summary.json)、[owner512真实曲线](D:/code/MPA-OpenCl/build_cuda_cmake/_baby_d_20261005/budget512_summary.json)、[存档A/B](D:/code/MPA-OpenCl/build_cuda_cmake/_baby_d_20261005/production_ab/measurements.json)、[量化账本](D:/code/MPA-OpenCl/build_cuda_cmake/_baby_d_20261005/quantitative.json)、[发布记录](D:/code/MPA-OpenCl/build_cuda_cmake/_baby_d_20261005/promotion.json)。

源码索引：[原生入口默认](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:13)、[标定与真实路径核对](D:/code/MPA-OpenCl/tools/bench/calibrate_stage2_d.py:118)、[fit输入/raw log校验](D:/code/MPA-OpenCl/tools/bench/fit_stage2_d.py:31)、[完整存档A/B](D:/code/MPA-OpenCl/tools/bench/bench_stage2_save_reduce.py:1)、[实际planner门禁](D:/code/MPA-OpenCl/tools/test/test_stage2_save_d_plan.py:1)。

## 6. 后续优化

上述生产接入已完成。剩余最大CPU准备间隙约.654秒需要继续定位；point约18.5秒、NTT约17秒的源码/时间线证据来自上一轮配对采集，已按SQLite重新核对：Stage1的ladder_chain（7.732463秒）在池范围之前，Stage2 point精确18.542850秒，未将Stage1混入。[范围复核](D:/code/MPA-OpenCl/build_cuda_cmake/_baby_d_20261005/profile_range_check.json)。随后分别评估减少点模乘依赖MAC及NTT规范化成本，并做完整A/B。此前Tensor真tile和双流为负结果，维持回退；多曲线需要共享workspace lease和逐生命周期RAM/VRAM预算。相同N/Q/曲线族/B1/B2/覆盖/线程的Prime95新对照仍未完成，本阶段没有证明已超过Prime95。
