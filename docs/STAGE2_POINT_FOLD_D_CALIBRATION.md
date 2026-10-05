# ECM Stage2：点折叠临时空间复用与 D 重标定

日期：2026-10-05。延续 [xADD6](STAGE2_XADD_D_OPTIMIZATION.md)、[固定 PTX NTT](STAGE2_FIXED_PTX_D_CALIBRATION.md) 和 [Mersenne Montgomery 点折叠](STAGE2_POINT_MERSENNE_MONTGOMERY.md)。设备为 GPU1 RTX4060 Laptop / sm89 / CUDA13.3；测量串行进行。

## 1. 顺序与算术合同

六模乘 xADD 已落地。本轮先减少点模乘的私有临时空间，再为新点算术重新拟合 D；NTT 改进使用这个结果作为完整曲线基线。ECM 模数记为 `N_ecm=2^s−1`，`W=ceil(s/64)`，模板容量记 `C`，NTT 素数为 `q=2^64−2^32+1`，长度记 `L`。

点折叠保持 `abR⁻¹ mod N_ecm`，`R=2^(64W)`，不是普通 `ab mod N_ecm`。主导 SOS/归约 MAC 从 `2W²` 降为 `W²+O(W)`；M4423 的 W70 对应约9800→4900个依赖MAC/模乘。xADD6、xDBL5、普通梯形bit分别约6/5/11次模乘，因此主导点MAC分别从12/10/22 W²降为6/5/11 W²。这些是算法计数，不是有效硬件周期数。

## 2. caller/callee 临时空间复用

原点折叠 callee 有 `out[C]`，caller 的通用 REDC 也有 `out[C]`。两条归约路径互斥；SOS完整消费输入后才调用归约，输出可以和输入别名。现在 caller 提前声明原 `out`，callee接收它的指针，折叠、条件减模和循环旋转的数学保持。

实现：[点折叠 helper](D:/code/MPA-OpenCl/tools/bench/stage2_point_mersenne.cuh:1)、[实际 SOS/归约调用](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:681)。独立probe从实际源文件提取通用 SOS/REDC，删除新分派得到参考；没有另写一份通用乘法。[probe builder](D:/code/MPA-OpenCl/tools/build/build_stage2_point_mersenne_probe.ps1:1)。

cuobjdump 对两份真实 native exe 的 NW128 实例核验：

- baby product：STACK5136→4112 B，REG40→40。
- baby inverse：STACK6160→5136 B，REG54→54。
- baby leaf：STACK7184→6160 B，REG64→56。
- chain 两种实例：STACK14352→13328 B，REG64→56。
- ladder 两种实例：STACK17424→16400 B；旧8乘法REG64→64，xADD6 REG58→64。

每个内核stack降低 `8C=1024 B/thread`。这不表示总显存恰好少1024×启动线程数；stack backing由CUDA运行时分配，必须和显式payload区分。REG升降也不能单独证明实际occupancy或性能。新持久数组0，点状态仍4B/曲线；H2D/D2H的数据布局保持。没有新NVML峰值或有效NCU cycle计数。

资源与构建证据：[资源7对](D:/code/MPA-OpenCl/build_cuda_cmake/_point_scratch_20261005/point_resources.json)、[构建清单](D:/code/MPA-OpenCl/build_cuda_cmake/_point_scratch_20261005/native/build_manifest.json)。native SHA256 `ae5bc62972d712d75c4356064815eebe866a666e2e246177df27ed17c121b872`，CUDA编译485.1s，19份原始依赖冻结在native/sources。

## 3. scratch 实验正确性与完整曲线

原语109/0，原语内部8次generic/new交叉22.861825→12.394315ms，减少45.786%；该组仍是“通用REDC对点折叠”，不是scratch单变量收益。[原语门禁](D:/code/MPA-OpenCl/build_cuda_cmake/_point_scratch_20261005/primitive_gate/summary.json)、[原语A/B](D:/code/MPA-OpenCl/build_cuda_cmake/_point_scratch_20261005/primitive_ab/summary.json)。

真实 native 18/0：7个Mersenne位宽×两种点模式，2048 Mont/GMP及1280 xADD逐字/别名检查；两种通用奇模数请求点折叠但实际fallback；两条实际保存点模型scope/叶校验。前14条合成X=2只用于算术与分派，不宣称有效Stage1。[18组门禁](D:/code/MPA-OpenCl/build_cuda_cmake/_point_scratch_20261005/native_gate/summary.json)。

固定 M4423/B1=1000/sigma26，B2=2011326186870，D1381380，两份binary均明确point1，顺序旧/新/新/旧：39.608262 /39.046426 /39.103370 /39.149837s；均值39.379050→39.074898s，减少0.7724%。每版本2条、无CI；只确认资源节约和没有观测到明显回退，不宣称稳定的微小加速。[完整A/B与冻结旧源码](D:/code/MPA-OpenCl/build_cuda_cmake/_point_scratch_20261005/scratch_ab/measurements.json)。

四条都恢复相同Q SHA256 `33cc6c63cec26c39900a7ed4ff551e2c2e0a533422905b538ec4f4aa809f092f`；leaf4244971527793015097、oracle c85031f6149bae11、空factor集合/bad0/pending0/clean1。S4 397launches、1836241poly_muls、36615543reduced coeff、2400selftest、60474GMP、3full_checks保持。[计时工具](D:/code/MPA-OpenCl/tools/bench/bench_stage2_point_mersenne.py:1)新增baseline实际点模式参数，避免把旧实验的mode0误当point1。

## 4. 新 D 模型的特征与证据合同

新全曲线模型profile6与旧profile5分开。纯NTT仍使用已冻结的profile5固定PTX12权重，只在四个NTT共同源文件raw hash完全相同时复用：ntt_poly_probe.cu、ntt_coop_outer.cuh、ntt_goldilocks_reduce.cuh、ntt_goldilocks_ptx.cuh。权重所属组件profile5不表示整曲线仍可复用profile5速率。

令 `P=phi(D)/2`，`I=floor(B2/D)+2`，`G=ceil(I/P)`；`U(m)=L(m)log2(L(m))·weight(log2L)`。尺寸由真实NTT精确packing界决定，不由本轮点运算改变。树特征 `T(P)=Σ_h floor((P+h)/(2h))·U(h+1)`，h=1,2,4,…<P；孤立未配对节点不额外算一次乘法。Newton特征 `V(P+1)=Σ_m 2U(m)`，m沿倍增到P+1。

经验phase秒数分别按以下特征拟合非负零截距速率：baby `P·max(1,log2D−2)`、affine P、F-tree T(P)、giant `I·(6+22log2B2/64)`、G-trees `floor(I/P)T(P)+T(I mod P)`、fold `(G−1)U(P+1)`、descent T(P)、inverse V(P+1)、accum P、residual G。point倍数保留在经验速率中重新拟合；不能仅将所有旧phase速率乘0.5，因为NTT、传输和CPU工作没有减半。

显式owner容量 `8W(9P+8)+48 B`；baby临时payload `8[(3P+5)W+P+WΣ_(k=1..8)ceil(P/2^k)]+ceil(P/256) B`。arena仍使用真实三个NTT形状的保守容量过滤。owner/baby/arena预算是硬约束，成本预测只对通过约束的候选排序；运行时headroom/allocator继续决定是否可用。未来并发曲线需要私有owner和共享NTT workspace lease，不能复制全部arena到同一卡。

例如大界首选P126720/W70：owner638673328B、baby284592655B；保守三形状arena6446506008B，fold L=2^27、tree L=2^26。小界P37440：owner188702128B、baby84086867B、arena1611810840B。owner是跨阶段持有的数据，baby与NTT scratch有不同生命周期；这些容量不能简单求和当作实测VRAM峰。host树/GMP对象、pinned staging、驱动栈不包括在这些payload式中。NTT每系数canonical field word仍8B，一次完整数组读+写为16L B/pass；通常两次forward和一次inverse，而实际pass数取决于shape policy。点scratch改动不改变上述传输或NTTpass计数。

采集时显式point0/1，逐条核验实际enabled/bits/nw、固定PTX3/GPUbaby/short/shape、保存点/Q、GMP/oracle、binary及所有raw编译依赖；profile6要求实际M4423/70limbs。fit重复核验raw日志、argv对应特征和phase；拒绝旧profile混入point1、点请求落回通用REDC、来源变化、锚点环境/模型不一致。[采集与合同](D:/code/MPA-OpenCl/tools/bench/calibrate_stage2_d.py:1)、[拟合](D:/code/MPA-OpenCl/tools/bench/fit_stage2_d.py:1)、[预算排名](D:/code/MPA-OpenCl/tools/bench/plan_stage2_d.py:1)。历史profile5重新运行26/0输入门禁。[历史合同回归](D:/code/MPA-OpenCl/build_cuda_cmake/_point_scratch_20261005/historical_fit_gate_v2/summary.json)。

## 5. 标定、留出与集成

大界 B2=2011326186870，D1141140：43.870720/43.919453s；D1231230：42.090979/42.156640s；D1411410：43.763972/43.290687s。第二轮反向顺序。D1381380锚点四条39.015412/39.050954/39.087603/39.018340s，均39.043077s。各D改变P/G/检查集合，仅用于成本标定，不当成相同计算量的点内核A/B。[六条训练](D:/code/MPA-OpenCl/build_cuda_cmake/_point_scratch_20261005/calibration/measurements.json)、[四条锚点](D:/code/MPA-OpenCl/build_cuda_cmake/_point_scratch_20261005/anchors/measurements.json)。

按§4的baby/affine/ftree/giant/gtrees/fold/descent/inverse/accum/residual顺序，冻结速率为1.6015102481e−6 /1.3761475061e−6 /1.8288498317e−10 /1.8211322282e−7 /7.6003205812e−11 /2.5677359807e−10 /4.2265559208e−10 /2.1641769289e−10 /1.0426828506e−6 /0.0892007416。它们乘各自特征得到经验秒数，单位不同，不能彼此相加。fit SHA256 `f57305b1bf4e398cf479a51d1afdcfca8603e5ccc7358fb37a4750da2e1b67d8`，拟合内误差−3.464..+0.763%，leave-D-out −3.879..+0.981%。[冻结fit](D:/code/MPA-OpenCl/build_cuda_cmake/_point_scratch_20261005/fit.json)、[量化与来源审计](D:/code/MPA-OpenCl/build_cuda_cmake/_point_scratch_20261005/quantitative.json)。

47-smooth D≤200000000共379419候选；arena6300MiB/owner640MiB/baby512MiB大界首选1381380/P126720，预测39.241904s；小界1e11首选390390/P37440，预测8.479721s，次选330330预测8.510573s。owner512MiB首选1141140，baby128MiB首选600600。四个首选均与旧模型相同，不宣称本轮换D收益。[大界](D:/code/MPA-OpenCl/build_cuda_cmake/_point_scratch_20261005/plan_large.json)、[小界](D:/code/MPA-OpenCl/build_cuda_cmake/_point_scratch_20261005/plan_small.json)、[owner512](D:/code/MPA-OpenCl/build_cuda_cmake/_point_scratch_20261005/plan_fold512.json)、[baby128](D:/code/MPA-OpenCl/build_cuda_cmake/_point_scratch_20261005/plan_baby128.json)。

先冻结fit，再独立测小界1e11，顺序390390/510510/510510/390390：8.373022/8.983254/9.006851/8.322544s。均值8.347783对8.995053，390390比粗候选510510少7.1958%，每D2条/无CI；fit hash保持，留出未进入拟合。没有实测接近的330330，因此不能从预测0.36%的差别证明390390是全局最优。[独立留出](D:/code/MPA-OpenCl/build_cuda_cmake/_point_scratch_20261005/holdout/measurements.json)。

锚点phase均值和full占比：init6.526490s/16.72%；其中baby3.7235/9.54%、affine0.176/0.45%、ftree残差2.626990/6.73%；main32.516588/83.28%，分为giant4.91925/12.60%、G树12.601/32.27%、fold5.63675/14.44%、descent6.22875/15.95%、inverse1.70475/4.37%、accum0.130/0.33%、其余1.296088/3.32%。phase墙钟包括所属CPU准备、传输、GPU和等待；F-tree包络等嵌套项不可再加。不是纯kernel池占比。

profile6输入门禁27/0，与历史26/0分开；包括点控制改变、原始log实际fallback、冒用旧profile、锚点profile改变等拒绝。[新输入门禁](D:/code/MPA-OpenCl/build_cuda_cmake/_point_scratch_20261005/fit_input_gate/summary.json)。接入的C++类保留profile0/2/3/4/5全部原速率；point0继续profile5。只有精确M4423、已测fixedPTX3/GPUbaby/shape/check预算等scope允许profile6，未支持路径回legacy。fixedPTX生产wrapper缺省point1，runtime/short/fold构建保持缺省point0；显式环境覆盖有效。

最终native SHA256 `893f6e907c17803ed90b09b98ffbf6b85e08b7deb1335a9d6c6efd1f8a01f69d`，CUDA编译399.5s，19份raw源码冻结；七个点内核资源与scratch实验完全一致。C++模型96/0（六个profile×两界×八D）；原生point18/0；实际save planner9/0；selector96/0（包括point0恢复profile5）；完整save/ini/worktodo/不同sigma/已知因子/失败队列/自动D共30/0。[最终manifest](D:/code/MPA-OpenCl/build_cuda_cmake/_point_scratch_20261005/native_calibrated/build_manifest.json)、[模型96](D:/code/MPA-OpenCl/build_cuda_cmake/_point_scratch_20261005/model_gate/summary.json)、[point18](D:/code/MPA-OpenCl/build_cuda_cmake/_point_scratch_20261005/final_point_gate/summary.json)、[planner9](D:/code/MPA-OpenCl/build_cuda_cmake/_point_scratch_20261005/save_plan_gate/summary.json)、[selector96](D:/code/MPA-OpenCl/build_cuda_cmake/_point_scratch_20261005/final_scope_gate/summary.json)、[入口30](D:/code/MPA-OpenCl/build_cuda_cmake/_point_scratch_20261005/final_accept/summary.json)、[资源再核验7](D:/code/MPA-OpenCl/build_cuda_cmake/_point_scratch_20261005/calibrated_resources.json)。[新速率](D:/code/MPA-OpenCl/tools/bench/stage2_d_model.cuh:51)、[选择器](D:/code/MPA-OpenCl/tools/bench/stage2_tree_gpu.cu:10803)、[生产缺省](D:/code/MPA-OpenCl/src/cuda/ecm_cuda_stage2.cu:20)。

固定D1381380/保存点/B2/检查，DCF旧生产point0对最终新版本point1，顺序旧/新/新/旧/新/旧/旧/新。旧48.779665/48.878203/48.860879/48.813851s；新39.006332/39.083014/39.067881/39.011255s，均值**48.8331495→39.0421205s，耗时减少20.04996%**。每版本4条/无CI；不把scratch的0.77%或前轮19.49%相加。八条实际点模式在记录中逐条保存，Q/叶/oracle/因子/检查覆盖均保持§3合同。[最终8条完整对照](D:/code/MPA-OpenCl/build_cuda_cmake/_point_scratch_20261005/final_ab/measurements.json)。该记录的scope末尾仍有旧工具文案“experiment default off”；实际mode字段与原始log明确旧0/新1，最终wrapper默认1；收尾修正文案，不改计时或校验逻辑。

已发布到 [production_stage2/ecm_cuda_stage2.exe](D:/code/MPA-OpenCl/build_cuda_cmake/production_stage2/ecm_cuda_stage2.exe)，SHA同893；旧DCF与原始18份源码已备份到previous_production。发布路径再次基本入口21/0，自动D完整曲线已在同字节native完成。[发布记录](D:/code/MPA-OpenCl/build_cuda_cmake/_point_scratch_20261005/publication.json)、[发布路径21](D:/code/MPA-OpenCl/build_cuda_cmake/_point_scratch_20261005/published_accept/summary.json)。本轮只更新工作区生产编译，不修改外部GIMPS生产目录。

复现需 `tools/build/build_ecm_cuda_stage2.ps1 -Build build_cuda_cmake/reproduce_point_fold -Arch sm_89 -GlBackend ptx -Rebuild`；再用 `--save YOUR_STAGE1.save --b2 2011326186870 --device 1 --results results.jsonl`。环境 `NTT_POINT_MERSENNE=0` 回旧Montgomery路径/profile5；generic N自动fallback。没有取消任何强制GMP/oracle检查。

## 6. 随后的 NTT 优化

现有真实profile中点pool已从17.960降到8.574s，NTT pool约15.27s，来自上轮单条配对trace而非本轮无观察器A/B。点加速后NTT占比上升。优先重新评估warp内低层单位根的移位算法，以当前固定PTX作为基线；旧方法3相对四fold基线的2.2%不能叠加到当前后端。关注消除根表读和乘积指令是否抵消指数/宽移位/修正的成本，必须保持正逆频谱、pointwise/scale、tile共享边界和任意表fallback合同。下一个更大候选是继续融合pass与减少外层shared/barrier，但增加寄存器或shared会降低驻留度，需要独立资源和完整卷积证据。参考现有冻结sppark/GPU-NTT源码及调研；Tensor真实tile旧负结果保留。

CPU准备间隙、多曲线workspace lease、真正同Q/B2/线程的Prime95完整对照仍需推进。当前结果不证明达到或超过Prime95。
