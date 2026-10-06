# Auto B2：精确树调度、G=1 和 chain 策略

日期：2026-10-06。延续 [多端点标定](D:/code/MPA-OpenCl/docs/STAGE2_AUTO_B2_MULTIANCHOR.md)。本阶段目标是将G=1及较低B2纳入原生收益规划，修正树工作量，并消费实测chain阈值策略。完整目标中的生产B1、泛型模数、总显存租约、并发与后续NTT优化仍需推进。

## 1. 模型结构

Python成本特征独立为版本7；历史 `calibrate_stage2_d.py` 的经验特征保留用于追溯。原生运行profile升级为 `.cprof` v2，identity明确记录feature版本和chain最小点数，旧v1不复用。每个scope包含17个非负率及chain/chain分块/ladder的实测覆盖标记；缺少覆盖的操作路径不参与候选选择。

对 n 个线性叶子，h=1,2,4,…<n：

\[
q=\lfloor n/(2h)\rfloor,\quad r=n\bmod(2h),\quad
c=q+[r>h]=\lfloor(n+h-1)/(2h)\rfloor.
\]

每层乘法分组数为 `[q>0]+[r>h]`；`0<r<=h` 时有一次单侧复制，复制 r+1 个系数。树乘法工作特征为 `sum(c*N_h*log2(N_h))`，N_h来自真实packing所对应的h+1输入长度。复制字数乘 `ceil(bits/64)`，不把复制算成NTT乘法。

G树模型为NTT工作、分组数、复制字数的三项非负拟合；单棵树实际乘法对数n−1，全部G树I−G。Python与原生均返回这些可核验的计数。分组参数估计整组调用的开销，不声称是逐CUDA kernel launch的精确周期数。

G=1没有外层fold和预先inverse，但scaled descent会建立长度P的局部inverse。若I=P，还需先将满根H mod F：`cp_divmod`计算一个小商再乘F。模型单列 `local_inverse` 与 `root_reduction` 工作特征，后一项为 `unit(1)+unit(P+1)`。G>1使用已有长度P+1的cached inverse，局部inverse为零。

G=1 admission本来就固定D/P，相关率按位宽、D和路径标定，不从另一个P缩放启动/selftest与局部inverse的固定成本。固定D下tree/local-inverse两列可能共线，非负拟合可将其总成本归入其中一列；工作量和实际阶段仍分别记录，不声称已独立测得这两项秒数。

## 2. 小批量ladder延迟

首个15d原型的每点线性模型低估G1小批量。实际例子M8191/D30030：720点giant约0.738秒，1440点约0.763秒，纯每点模型会漏掉显著底座。源码每次ladder launch最多8192点、每block64线程；小输入不能直接套大批量吞吐率。

修正为：

\[
T_{giant}=a_c W_c+b_c C_c+a_l W_l+b_l L_l,
\]

W为经验点工作特征，C_c为chain点块数，L_l为每个ladder点块按8192点上限拆分后的设备调用数。b_l估计每次调用的典型延迟底座，包含低并行度执行成本，不能全部解释为CPU launch API耗时或已证明的occupancy波数。

active-set NNLS对列归一化，枚举可行活动列、检查秩与非负性。零工作阶段只允许其观测也接近零；未测的giant路线不会因系数为零而自动获得准入。

## 3. 原生范围和策略

新规划器取各实测scope的并集，在每个scope内部生成B2点，再按自身B2/P/G条件过滤。旧实现将所有scope区间取交集，低B2的G1范围与高B2范围不相交时无法规划。现在间隙内的固定B2仍拒绝，不做区间外插。

profile中的 `chain_min` 在实际curve worker执行前绑定，显式冲突环境会拒绝；plan-only不修改这个执行配置。仍固定block64和ladder cap8192。当前待验证策略为8192点，既有8327/24977点的chain/ladder仿射门禁已经提供算法对照依据，新一轮额外测8191/8192/8193点。

profile reader同时拒绝G1声明驻留fold owner、非有限率、错误覆盖标记、旧格式、重复scope。整数作用域键按数值规范化，不能用前导零绕过重复检查。主程序仍保留显式非零B2，队列只在成功后推进。

`plan_auto_b2.py` v2改为调用原生plan-only，使用已审计 `.cprof`，减少两份搜索算法和实时预算逻辑漂移。JSON模型与运行profile的source SHA必须匹配。

## 4. 当前验证状态

候选仅HostOnly重新构建，CUDA对象经依赖、backend、架构、toolkit与object SHA核验后复用。原4acc、生产893及原始GPU算术源码保持。17率候选SHA256为 `5f4ceca4f85a8d39d8ef02f67c4167c8e1f99cc51f3d791497a30d567718edc4`，位于 `build_cuda_cmake/_auto_b2_g1_20261006/native_latency/ecm_cuda_stage2.exe`。

CPU回归覆盖密集树独立枚举、G1 inverse/满根切换、重复特征中位数与小ladder延迟底座，共5项。17率候选头文件通过CPU mock packing下的原生并集/间隙/G1/giant覆盖检查；这些不是CUDA packing或GPU算术门禁。

17率候选标定完成：18批Stage1/117点独立核验；174条Stage2中144条用于拟合，30条holdout。新增108条冻结预测验证，90条是独立B2/边界/chain切换邻点，18条是已训练满根的重放，不能算独立盲测。共282条curve全部算术检查通过，独立GMP系数检查1,086,916个，G1曲线162条，其中36条执行满根余式，102组叶指纹一致。

**完整精度与排名门禁失败，未发布新运行profile，也没有将生产程序替换为此候选。** 15个scope中8个满足逐条误差不超过10%；失败7个仍完整保留。最大盲测误差49.710%，收益排名中M4423/batch1损失11.182%、M8191/batch1损失11.855%，均超过5%。排名按实测Stage1摊销、engine与冻结cold计算，不能解释为完整生产吞吐实测。

audit的 `integrity_passed` 表示证据身份/计数/算术检查通过，`accuracy_passed` 与 `ranking_passed` 单列；`passed` 要求两者全部通过。exporter还要求每个声明scope都通过，拒绝删除失败scope后发布剩余子集。候选JSON可以用于诊断，不能当作已审计的 `.cprof`。

首个16率原型以及每次失败拟合仍保留。原采集6个工具的文本与SHA在更改诊断工具前冻结，避免把后续工具冒充原始采集器。新脚本 `plan_auto_b2.py` 已改用原生规划，但此候选尚无合格运行profile，因此未进行新版实际auto/manual/queue验收。

## 5. 等待波动的证据与后续

M4423/D60060/B2=52.5亿/owner640，同输入两次full为3.430313与2.418996秒，G树阶段1.608与0.609秒；前者carry chunk readback 1.450748秒。运算调用/分组/叶指纹相同。这将约1秒额外成本定位到相关等待路径，但 `cudaMemcpy` 的墙钟包含之前排队执行与调度，不能直接解释为PCIe传输。

M8191/D30030/G1满根同输入，训练descent为0.356/0.363秒，重放为1.409秒；full训练2.247917/2.268699秒，重放3.304199/3.271003秒。没有改变B2、树工作量或profile来掩盖它。多G最坏案例full20.138931秒，相邻同输入10.762288秒。

两个独立Nsight Systems诊断尚未复现大等待：大B2/owner640 full9.586697秒，自己GPU活动窗口9.651478秒，其中活动区间并集8.801988秒、无自身活动0.849490秒；满根full2.249907秒，窗口无自身活动0.427874秒。大B2 memcpy API合计4.268840秒，但实际所有方向DMA合计约0.039745秒，二者不可混为传输成本。以上采集含启动检查和尾部，不能当作Stage2精确利用率或整卡空闲比例。

最小满根重复8次为2.261791～2.306736秒，GPU图形/SM采样均1800MHz；descent0.357～0.408秒，没有复现前述约1秒尾部。此重放只用于诊断，未替换失败验证，也不能证明问题消失。继续交叉重放大/小形状及采集等待时间线，区分CPU/驱动调度、GPU抢占/频率和carry路径开销；随后重新冻结预测验证全部范围。

交叉重放8条中，大形状owner0通常10.570576～10.888356秒，一条增至15.574904秒，G树4.888、fold4.730、descent0.902秒；其GPU采样在约9.9～10.1秒利用率为0，约10.35～10.55秒SM为390MHz，另在约15.35～15.55秒为420MHz，之后恢复1800MHz。这是状态异常的相关证据，尚不能确定时钟变化是原因还是等待的结果。第三次Nsight采集中owner0 full10.206569秒，仍未复现5秒长尾；不能以正常trace覆盖掉失败样本。

本阶段也未覆盖G2与低/高B2之间的全部间隙、生产B1、choose12、非精确梅森数或总内存租约；这些输入仍不能自动外插。本候选没有证明新的生产性能提升。

## 6. 源码与复现

- [原生v2 reader](D:/code/MPA-OpenCl/src/core/ecm_stage2_cost_profile.h:39)、[精确树统计](D:/code/MPA-OpenCl/src/core/ecm_stage2_cost_profile.h:94)、[scope并集/内存准入/收益选择](D:/code/MPA-OpenCl/src/core/ecm_stage2_cost_profile.h:109)。
- [执行策略绑定](D:/code/MPA-OpenCl/src/core/ecm_cuda_stage2_main.cpp:513)、[Python特征与G1成本](D:/code/MPA-OpenCl/tools/bench/ecm_cost_model.py:24)、[giant实际launch特征](D:/code/MPA-OpenCl/tools/bench/ecm_cost_model.py:112)。
- [重复阶段中位数拟合](D:/code/MPA-OpenCl/tools/bench/fit_ecm_costs.py:19)、[完整精度/排名门禁](D:/code/MPA-OpenCl/tools/bench/audit_ecm_costs.py:92)、[导出拒绝部分发布](D:/code/MPA-OpenCl/tools/bench/export_ecm_cost_profile.py:57)。

```powershell
$base = 'build_cuda_cmake/_auto_b2_g1_20261006'
$exe = "$base/native_latency/ecm_cuda_stage2.exe"
python tools/bench/measure_ecm_costs.py --stage2 $exe `
  --save-dir "$base/study_latency" --output run/new_g1_study `
  --train-b2 3000000000 6000000000 `
  --holdout-b2 3750000000 --shuffle-seed 20261008 --monitor-state `
  --g1 --chain-min 8192
python tools/bench/fit_ecm_costs.py --measurements run/new_g1_study/measurements.json `
  --estimator phase_medians --output run/new_g1_profile.json
python tools/bench/validate_ecm_costs.py --profile run/new_g1_profile.json `
  --stage2 $exe --save-dir run/new_g1_study --output run/new_g1_blind `
  --b2 5250000000 --g1
```

命令用于复现诊断；使用全新输出目录。采集会验证候选旁的冻结原始源码manifest，当前工具改动会改变工具SHA，因此新数据不能冒充本批。通过完整门禁后才可export，再进行新版原生实际曲线/INI/队列验收。

[候选模型](D:/code/MPA-OpenCl/docs/data/ecm_auto_b2_g1_20261006_profile.json)、[失败审计](D:/code/MPA-OpenCl/docs/data/ecm_auto_b2_g1_20261006_audit.json)、[原始日志/Stage1保存点/冻结源码与工具/诊断证据](D:/code/MPA-OpenCl/docs/data/ecm_auto_b2_g1_20261006_evidence.json)。该证据保留16率原型与失败拟合，不提供新版 `.cprof`。既有4acc二进制及其2026-10-06多端点v1 profile仍是上一阶段通过验收的组合，不能把旧profile交给新5f4c候选。


## 7. 分 D 标定与完整范围验证（2026-10-06）

本轮修正一个拟合假设：相同位宽、owner 路径中的不同 D 会使用不同 NTT 长度，不能仅按 N log N 共用一个秒数率。现在 `fit_ecm_costs.py` 默认 `--fit-scope per_d`，每个位宽/owner/regime/D 独立拟合；`pooled_d` 保留旧多 G 方法供追溯。运行格式、17 个率、成本特征版本 7 和收益函数不变：

\[
S(B_2,D,path)=\frac{K(B_1,B_2)}{T_{1,process}/batch+adjust\,(T_{2,engine}+T_{cold})}.
\]

它估计总流程相对收益，没有把读取 save 时的 Stage1 成本归零，也没有将 pure NTT iter/s 当作完整 Stage2 秒数。

### 7.1 数据与实现

新增 `measure_ecm_costs.py --holdout-all-d`，所有声明的 D 都测独立留出 B2；旧工具只测中间 D，拆开 scope 后其他 D 没有留出证据，不能直接发布。`--extend-study` 允许在新目录复制兼容的完整 study 并补充观测，保留原 JSON SHA、identity、controls 和原始记录。审计要求原来的 18 批 Stage1、174 条 Stage2 逐字段不变。源 study、输入和原日志不会覆盖。

同 5f4c 冻结二进制补测 24 条 B2=37.5 亿留出。联合 study 为 144 条 train、54 条 holdout，18 批 Stage1/117 个已独立核验点沿用。分 D 后声明 27 个 Stage2 scope（18 个多 G、9 个 G1），留出门禁 26/27 通过；M8191/D120120/owner640 的一条 8.751583 秒样本误差 −31.525%，保留失败。

独立验证工具不再跳过 holdout 失败的 scope；采用冻结 seed 的随机顺序，查询 GPU UUID，验证 frozen build 源码/工具身份，强制 CUDA_LAUNCH_BLOCKING=0。profile 含 G1 时必须指定 `--g1`，避免悄悄漏测。后续新增 save SHA 预检查与 `--check-inputs-only`：修改一个字节就会在设备查询前拒绝；此门禁执行 GPU 曲线数为 0。实际盲测使用新增预检查前已冻结的 collector，其原始代码和 SHA 同样保留。

### 7.2 完整结果及发布状态

新冻结验证 108 条：90 条独立 B2/边界/chain 邻点，18 条满根重放。累计成本证据 306 条全部算术检查通过、1,233,976 个独立 GMP 系数检查、108 组叶指纹一致。这里 174 条原始观测沿用，仅本轮新增 24+108=132 条成本曲线；没有把旧数据重新算成新执行。

收益排名六组全部通过：M2203、M4423、M8191 各含 Stage1 batch1/12；五组损失为 0，M8191/batch1 的选中实测收益损失为 0.178660%。计算依据仍是 engine + 冻结 cold + 实测 Stage1，尚不是完整连续生产吞吐验收，也不是对所有 B2 的全局最优证明。

**精度仍失败：27 个 scope 中 21 个通过，108 条中 8 条误差超过 10%，最大绝对误差 46.594917%。** 最大样本 M8191/D120120/B2=1,383,422,040 为 10.167326 秒，另一次同输入约 5.54 秒。附加下降、G 树等待仍会破坏秒数预测。没有删去失败范围、放宽门槛或发布部分 cprof；exporter 明确报完整精度/排名门禁失败且不创建输出。本轮没有新的原生实际 auto/manual/INI/queue 验收，没有替换生产 exe 或旧 4acc/v1 profile。

[候选模型](D:/code/MPA-OpenCl/docs/data/ecm_auto_b2_per_d_20261006_profile.json)、[完整审计](D:/code/MPA-OpenCl/docs/data/ecm_auto_b2_per_d_20261006_audit.json)、[完整原始证据及冻结工具](D:/code/MPA-OpenCl/docs/data/ecm_auto_b2_per_d_20261006_evidence.json)。

### 7.3 等待与显存诊断

外部观察只覆盖最后 29 条附近的过程窗口（584 个 200ms 样本），根据日志修改时间及 child wall 近似对齐。最大误差曲线窗口观察 SM 210..1800MHz、9 个低于 1000MHz 的样本、GPU 利用率 0..100%、clock reason 原始掩码 1/36；系统 CPU 约 11.2..28.9%。这些是相关证据，不能把低时钟直接认定为长尾原因，也不能据全机 CPU 占用排除单线程调度。

追加两个 Nsight Systems trace：上述 G1 邻点 full=5.347643 秒，多 G/B2=52.5 亿 full=6.389195 秒，均未复现长尾。自己的全部 GPU 事件窗口分别有 1.125514/1.208655 秒无自身活动（20.84%/18.78%）；包括启动检查与尾部，不等于整卡空闲比例或精确 Stage2 利用率。实际全部 H2D/D2H DMA分别约 0.0763/0.0358 和 0.0919/0.0501 秒。

重建设备 malloc 同时存活峰分别为 948,285,992 / 1,031,262,480 B；pinned 主机峰为 66,817,544 / 90,442,504 B。这些不含 driver/context/module/static，也不是整进程峰值或可相加的独立容量清单。Nsight 工具现从二进制旁原始 frozen sources 检查构建，避免工作树已改动时误核对另一版本源码。

上下文复用两版真实对照均无收益，执行路径已撤回；第二版 12 条曲线计数/输出通过但吞吐下降约 11.93%。[实验细节和冻结源码](D:/code/MPA-OpenCl/docs/STAGE2_CONTEXT_REUSE.md)。这没有改变成本模型的逐曲线进程合同。

### 7.4 复现及后续

```powershell
$old = 'build_cuda_cmake/_auto_b2_g1_20261006'
$new = 'run/new_per_d'
$exe = "$old/native_latency/ecm_cuda_stage2.exe"
python tools/bench/measure_ecm_costs.py --stage2 $exe `
  --save-dir "$old/study_latency" --output "$new/study" `
  --extend-study "$old/study_latency/measurements.json" `
  --train-b2 3000000000 6000000000 --holdout-b2 3750000000 `
  --holdout-all-d --g1 --chain-min 8192 --monitor-state --shuffle-seed 20261009
python tools/bench/fit_ecm_costs.py --measurements "$new/study/measurements.json" `
  --estimator phase_medians --fit-scope per_d --output "$new/profile.json"
python tools/bench/validate_ecm_costs.py --profile "$new/profile.json" `
  --stage2 $exe --save-dir "$new/study" --output "$new/blind" `
  --b2 5250000000 --repeats 2 --g1 --shuffle-seed 20261010
```

每次使用新目录。三个位宽和所有已声明 D/path 都须保持验证。先定位并减少大位宽下降/G 树的额外等待，再冻结完整精度与收益验证；随后才能导出 v2 cprof 并运行原生 auto/manual/INI/queue 验收。G2 桥接、生产 B1、choose12、泛型模数、总显存/RAM 准入及更广 NTT 标定仍未完成。

源码：[分 D 分组](D:/code/MPA-OpenCl/tools/bench/fit_ecm_costs.py:19)、[旧 study 扩展及所有 D 留出](D:/code/MPA-OpenCl/tools/bench/measure_ecm_costs.py:45)、[原观测保护](D:/code/MPA-OpenCl/tools/bench/audit_ecm_costs.py:47)、[输入身份及完整 scope 验证](D:/code/MPA-OpenCl/tools/bench/validate_ecm_costs.py:24)、[拒绝不合格发布](D:/code/MPA-OpenCl/tools/bench/export_ecm_cost_profile.py:35)、[原始构建的剖析核对](D:/code/MPA-OpenCl/tools/bench/profile_stage2_points.py:38)。CPU 拟合回归 4 项通过；输入正/负检查与发布负例均未执行 GPU 曲线。


## 8. CPU 等待与 Auto B2 调度边界（2026-10-06）

### 8.1 重放结果和 CPU 等待方式

继续相同输入的时长重放，长尾并非总在下降：M8191/D120120/B2=1,383,422,040 的两次 G 树 0.208→4.866 秒，下降仍约1.49秒；另一个输入下降增加时，GMP校验墙钟也由约0.05秒增至约1秒。调用计数、叶和校验选择一致。这要求区分 CPU 调度、CUDA 等待和功耗状态，不能直接将所有增加的墙钟归为 NTT 运算或 DMA。

新增 `bench_stage2_host_wait.py` 使用同binary、相同有效sigma26保存点和ABBA对照。进程优先级实验先发现本机driver升到Above Normal时，curve worker仍为Normal；失败采集和原工具保留，不能当成成功干预。改为直接启动原内部worker、校验原offset/FNV/hash后，16条真实曲线确认目标优先级实际生效，但没有消除长尾；多G形状提高优先级后中位慢16.426%，没有发布优先级策略。

新增可选 `NTT_CUDA_WAIT_MODE=0|1|2|4`：Auto / Spin / Yield / BlockingSync。未设置时执行原行为；只在真实curve worker的当前目标设备上下文中设置并核对flags，其他非调度flags保留。CUDA13返回的MapHost位是隐式位，设置时须去除，再检查返回状态。BlockingSync允许CPU线程在设备等待期间阻塞；Auto在常见条件下会忙等。[NVIDIA Runtime API](https://docs.nvidia.com/cuda/archive/13.0.2/cuda-runtime-api/group__CUDART__DEVICE.html)。它与 `CUDA_LAUNCH_BLOCKING=1` 不同，后者串行化CUDA调用。本轮只提供手动固定B2的可选模式，原Auto B2 profile要求模式0，拒绝非0的未标定配置；没有新增INI键或修改默认等待策略。

两次HostOnly构建复用受SHA/依赖/架构/toolkit核验的相同CUDA object，无新CUDA算术编译。等待版binary为22763e…359a；边界修正版为 `8ceed31fb144e37a5abe028e3a00595862fda0860c31a301cc57739f32f22774`，位于 `build_cuda_cmake/_auto_b2_host_wait_20261006/native_final`，25个原始依赖冻结。生产893不替换。

先16条M8191等待模式ABBA，再24条三位宽ABBA，G1与多G、D120120/B1=1000/arena4096/owner0均输出和校验一致。后者在进程退出后利用Popen仍持有的原句柄查询完整GetProcessTimes，不靠PID重开或将200ms样本拼成完整CPU时间。Windows以100ns单位返回，仍有调度计账粒度。

完整进程CPU中位数（同一GPU、每模式每形状两次）：

- M2203：G1 1.210938→0.843750秒（−30.32%），多G 1.468750→0.859375秒（−41.49%）。
- M4423：G1 2.320313→1.210938秒（−47.81%），多G 2.859375→1.390625秒（−51.37%）。
- M8191：G1 5.968750→2.359375秒（−60.47%），多G 7.023438→2.554688秒（−63.63%）。

这些是worker整个生命周期的user+kernel CPU时间，含启动；不是Stage2的GPU周期数。G1 B2=1,383,422,040，多G B2=5,250,000,000，三种宽度均固定相同D。CPU消耗减少已实测，但墙钟不稳定：三宽度相应完整engine中位变化约+1.40/+3.29、+1.26/−0.075、+11.26/+26.98%，另一组M8191又出现较快中位。全部长尾保留，未宣称时间加速或解决精度失败。

### 8.2 CPU 等待与串行/双进程的交叉试验

新增 `bench_stage2_concurrency.py --cuda-wait-modes 0 4`，镜像顺序为 serial0/parallel0/parallel4/serial4/serial4/parallel4/parallel0/serial0。两轮16批、32条实际曲线使用两条独立核验的sigma26/27存档，M8191/D30030/B1=1000/B2=52.5亿/arena512/owner0；输入、叶、因子及全部检查一致。

本批中位吞吐：Auto等待串行0.067395、双进程0.077301 curve/s（+14.70%）；Blocking等待串行0.074074、双进程0.078480（+5.95%）。同为双进程，Blocking只比Auto约快1.53%。较早同形状测量双进程比串行慢6.02%；这批串行出现长尾且较慢。四样本/单元不足以证明稳定生产吞吐收益，不能选择本次正结果忽略旧负结果；没有发布默认并发或内存lease。

200ms同次采样后求峰：串行整卡used最大1384.617MiB，双进程2537.234MiB；自身进程private合计最大约1605.852/3055.801MiB，working-set合计约312.500/621.238MiB。private是commit，working set包含共享页，整卡used含baseline和driver；不是唯一物理RAM或完整峰保证。仍只对已测D30030以每worker2GiB VRAM/RAM作试验防护，并外留768MiB显存/2GiB RAM，不扩充为任意输入保证。

### 8.3 Auto B2 搜索边界修复

原网格只加入起始 `chain_min` 和 `kP`。实际调度为：

\[
I=\lfloor B_2/D\rfloor+2,\quad P=\varphi(D)/2,\quad W=\lceil bits/64\rceil,
\quad C=P\left\lceil\frac{\max(P,\lfloor256MiB/(16W)\rfloor)}P\right\rceil.
\]

G在 `I=kP+1` 时增加；giant每C点重新划分尾段，尾段在 `I=kC+chain_min` 转为chain。旧搜索遗漏后续分块的这个边界。一个I对应的B2平台末端为 `D*(I-1)-1`，原来只加入平台起点及±1，也会漏掉该末端。

新搜索加入kP及kP+1、kC及kC+1、有效的kC+chain_min，并加入每个关键I平台末端，按用户范围剪裁、检查乘法/加法边界；之后仍经过每scope的B2/P/G/路径/内存准入。不扩大未测范围，不改变17率或收益函数，也不声称覆盖全部全局最优点。

CPU合成率回归先红后绿：W128/P2880/C132480/min8192，阈值I=140672；旧选择B2=4,231,858,536，新选择阈值平台末端B2=4,224,350,129。mock packing不是CUDA门禁。实际native plan-only使用真实NTT packing的独立合成率fixture也选择同一端点；该fixture没有通过成本标定，不可作为生产cprof。

再执行真实边界邻点 I=140671/140672/140673：B2分别4,224,290,070 / 4,224,320,100 / 4,224,350,130，实际giant chain chunks为1/2/2，与Python特征及真实kernel一致，树pairs/groups/copies及必需检查通过。

最终114项配置/搜索/队列检查、9条实际曲线通过，含模式0/1/2/4；未标定mode4的auto plan拒绝、非法mode3的worker失败且保留原queue/不写result。首个queue fixture错把描述文字放进known factors列，在worker之前被正确拒绝；失败目录与原工具保留，修正为空字段后重跑。较早92项/7条也是实际执行，但属于重复门禁，不新增独立覆盖。

本阶段完整批次共104条算术通过、760,628次独立GMP抽样系数检查（重复执行也计入）；未完成对照的首次priority采集另外执行两条，保留但不计为完整性能样本。所有新优化仍未替代完整Auto B2精度/排名和生产B1/choose12/泛型/内存租约验收。

### 8.4 使用与来源

仅在上述独立候选中，手动固定B2可比较：

```powershell
$env:NTT_CUDA_WAIT_MODE = '4'
PATH_TO_NATIVE_FINAL/ecm_cuda_stage2.exe --save m8191.save --b2 5250000000 `
  --d 30030 --device 1 --arena-mb 512 --factor-only
Remove-Item Env:NTT_CUDA_WAIT_MODE
```

该变量针对Stage2 curve入口，未给NTT tune附加等待模式。默认/生产旧exe不提供此新开关；模式1/2只有接口及算术检查，没有吞吐标定。CPU更低也不代表墙钟必然更短。继续定位WDDM/完成通知等等待，再进行完整新版profile和原生生产接口验收。

源码：[当前上下文等待控制](D:/code/MPA-OpenCl/src/core/ecm_cuda_stage2_main.cpp:454)、[Auto配置保护](D:/code/MPA-OpenCl/src/core/ecm_cuda_stage2_main.cpp:527)、[真实执行前绑定](D:/code/MPA-OpenCl/src/core/ecm_cuda_stage2_main.cpp:597)、[关键平台及分块边界](D:/code/MPA-OpenCl/src/core/ecm_stage2_cost_profile.h:134)、[完整CPU进程计量](D:/code/MPA-OpenCl/tools/bench/bench_stage2_host_wait.py:56)、[实际配置/边界/队列门禁](D:/code/MPA-OpenCl/tools/test/test_stage2_cuda_wait.py:41)。

[汇总](D:/code/MPA-OpenCl/docs/data/ecm_stage2_host_wait_20261006_summary.json)、[全部原始日志/存档/冻结源码及失败实验](D:/code/MPA-OpenCl/docs/data/ecm_stage2_host_wait_20261006_evidence.json)。GPU0外部生产未操作，未发布新运行profile或修改生产exe。


## 9. G2与低/高B2过渡区标定（2026-10-06）

以下计划段保留采集启动时状态；完整校准和验证现已结束，最终结论见§9.2。

本轮补齐区间生成和标定/验证工具，完整实测尚在运行；没有通过成本精度门禁或发布新的运行profile。原先G1范围截至 `D*(P-2)`，高B2从30亿起，中间有未测范围。新增 `ecm_cost_cases.py` 用整数点数生成各D的相邻范围：

```text
G1：原quarter-P起点 .. D*(P-1)-1
G2：D*(P-1) .. D*(2P-1)-1
bridge：D*(2P-1) .. min(large_training_B2)-1
multiple：现有large B2范围
```

G1的完整根现在同时采平台起点与末端；G2含首点、四分之一/半/完整第二根及末端。chain阈值位于所选区间时加入阈值−2/0/+2训练点。bridge有两端、整数几何中点及独立留出点，填到large训练最小B2之前。三个D逐整数相邻不留D−1的小间隙；这仅描述计划覆盖，尚不能作为已验证的可用范围。

`measure_ecm_costs.py --g2` 会同时启用G1，`--bridge`同时启用G1/G2和所有D留出。各regime保留独立标签，G1只有owner0，其余同时测resident640/0。fitter对g1/g2/bridge强制按D分别拟合，不合并为高G已有率，不修改特征版本7及17率模型。原生cprof按P/G/B2准入，不需要增加运行格式字段；通过完整审计之后才能导出。

独立验证新增G2内部/最后非满根点、bridge内部及G分组边界、后续giant chunk及tail-chain阈值邻点。所有声明scope都进入验证，holdout失败不跳过。与训练相同I的工作形状即使B2有少量差别也单列shape_replay，root_replay保留；它们仍进入每scope精度门禁，收益排名和独立blind计数排除重放。审计另核对多G的cached inverse复用及无根长除法。

新增 `--reuse-stage1 STUDY` 只复制完整、相同Stage1 binary和配置的原始观测，保存原study SHA、identity、controls与逐记录内容；不带入旧Stage2秒数。源保存点SHA仍检查，三条sigma26输入还在新计时前重新与CPU参考核对。审计要求原Stage1记录完全相同、GPU UUID相同、B1=1000/lcm/GPU1合同一致。此标定只覆盖lcm，未扩展choose12或生产B1。

当前8cee冻结二进制上安排666条全新Stage2：G1 126、G2 252、bridge 180、large multiple108；其中540训练、126留出，三位宽2203/4423/8191、D30030/60060/120120、arena4096、owner640/0、两重复、固定随机seed20261011、默认CUDA等待方式。18批/117点Stage1原证据复用，不宣称重新运行。预期形成63个完整scope；耗时精度或排名失败仍不得发布部分子集。

4项纯整数覆盖回归通过（各regime端点、实际G分类、旧G1兼容及chunk验证点），既有4项拟合回归通过。实测已覆盖四个regime，逐条采集保留必需GMP/oracle检查、树计数和fold路径；这属于进行中证据，不能替代完整审计。采集结束后仍须拟合、冻结新预测、完成全范围独立验证与原生auto/manual/INI/queue验收。[冻结计划及进度快照](D:/code/MPA-OpenCl/docs/data/ecm_auto_b2_bridge_20261006_plan.json)。

```powershell
$exe = 'build_cuda_cmake/_auto_b2_host_wait_20261006/native_final/ecm_cuda_stage2.exe'
python tools/bench/measure_ecm_costs.py --stage2 $exe `
  --save-dir build_cuda_cmake/_auto_b2_per_d_20261006/study `
  --output run/new_bridge/study --train-b2 3000000000 6000000000 `
  --holdout-b2 3750000000 --bridge --chain-min 8192 `
  --shuffle-seed 20261011 --monitor-state `
  --reuse-stage1 build_cuda_cmake/_auto_b2_per_d_20261006/study/measurements.json
```

新目录用于新采集。中断后用同一输出和同样controls加`--resume`，去掉首次导入用的`--reuse-stage1`；所有binary/工具/源码SHA仍须匹配，不要仅因观察超时重新开始。生产893和旧已发布profile保持，当前实验没有新的生产耗时/自动选择通过结论。


补充审计（采集进行中）：新增 `ecm_cost_coverage.py` 从controls重建完整计划，并逐case核对实际观测和存储计划。删除计划与观测中的同一项、用重复case替代缺失case（总行数不变）、重复冷Stage1观测替代另一次重复均会拒绝。盲测与冻结case也要求身份集合及次数完全一致，不只比较每scope数量。审计重新根据训练I核对replay标签，每个scope至少有一个独立工作形状样本；exporter拒绝只有重放的scope。新增7项覆盖负例/回归通过；旧174/198条完整study也通过计划核对，旧306条证据重审仍是integrity/ranking通过、accuracy失败，没有改变既有失败结论。

### 9.1 完整校准与冻结独立验证（2026-10-06）

上述采集进程已正常退出（exit=0），`complete=true`，完整计划重建确认666条Stage2和18批原Stage1证据。540条训练、126条留出全部保留。按固定工作特征的阶段中位数拟合63个scope（18 multiple、9 G1、18 G2、18 bridge），57个留出通过10%门槛。6个失败scope均为8191位：D120120的resident multiple/G2和owner0 G1/G2，以及D60060的owner0 multiple/bridge。最坏留出误差−48.931644%，本轮不能导出生产cprof。

模型与8个拟合/验证/审计依赖冻结在 `build_cuda_cmake/_auto_b2_bridge_20261006/fit_validation_sources/`。GPU1独立验证已启动，固定seed20261012，全部63个scope参与，计划468条（408条独立形状、18条root replay、42条shape replay）。留出失败scope仍参与；完整精度、收益排名和算术审计尚未结束。

新增只读诊断 [summarize_ecm_cost_variance.py](D:/code/MPA-OpenCl/tools/bench/summarize_ecm_cost_variance.py:36)，核对计划、逐日志SHA与阶段分解，按完全相同的bits/D/B2/owner/kind/regime配对。333组slow/fast中位1.010324，20组超过1.1，最大2.401893；所有666条均保留。脚本同时报告原始误差和诊断中位误差，不重拟合、不删除样本、不修改发布门槛。

重复时间差已定位到多个阶段：

- M8191/D60060/B2=691831139/owner640：full差5.217099秒，descent差5.222秒，其余小差互相抵消。
- M8191/D120120/B2=518678160/owner0留出：full 4.782074/9.514152秒，差4.732078；descent差4.772秒。
- M8191/D120120/B2=3750000000/owner640留出：full 6.012395/9.924511秒；accum差3.517秒，giant差0.260秒。
- M8191/D120120/B2=1902460560/owner640留出：full差1.902557秒，inverse差1.894秒。

G1长尾对照oracle host_total为0.148360/0.148890秒，没有随full额外增长4.7秒；不能据此把等待归因于GMP抽样。整进程NVML/CPU采样含冷启动，不能证明具体API、PCIe、WDDM或外部调度为根因。full/init/main各打印六位小数，诊断另记录约2微秒以内的分解舍入差。

复现诊断（执行零条GPU曲线）：

```powershell
python tools/bench/summarize_ecm_cost_variance.py `
  --study build_cuda_cmake/_auto_b2_bridge_20261006/study/measurements.json `
  --profile build_cuda_cmake/_auto_b2_bridge_20261006/profile.json `
  --output build_cuda_cmake/_auto_b2_bridge_20261006/variance.json
```

当前为完整校准结束、独立验证运行中，尚未发布新v2 profile。生产B1、choose12、泛型/余因子和总VRAM/RAM租约仍待实现与验证。

[校准阶段证据快照](D:/code/MPA-OpenCl/docs/data/ecm_auto_b2_bridge_20261006_calibration.json)包含666条原始阶段/工作特征/日志SHA、完整校准身份、63范围状态及20组最大重复差诊断；[冻结模型](D:/code/MPA-OpenCl/docs/data/ecm_auto_b2_bridge_20261006_profile.json)包含所有63个scope，不是可运行cprof。快照的validation_running描述生成时状态；后续完整审计应单独记录。

### 9.2 全范围验证结束，排名通过、精度失败（2026-10-06）

468条冻结验证全部完成，原进程正常exit0且complete=true。666条校准加468条验证共1134条曲线，完整计划/身份、原始日志SHA、树pairs/groups/copies、giant route、G1根除法和多G逆元复用审计通过；3,430,016次GMP系数抽样检查通过，312组相同输入的leaf指纹一致。18批/117点Stage1沿用原证据。408条验证为独立工作形状，60条重放仍参加精度检查，收益排名排除重放。

63个完整scope中42个通过耗时门槛：2203位18/21，4423位21/21，8191位3/21。全部验证误差范围−60.138171%..+13.717604%，因此accuracy_passed=false。六组完整独立网格收益排名（3位宽×Stage1 batch1/12）均通过，最大实际收益损失0.632980%，其余分别为0或约0.005927%；**这不是连续B2全局最优证明，也不是新版native auto/manual/INI/queue验收**。

全审计passed=false，exporter实际执行后按完整精度/排名要求拒绝，未生成cprof。没有仅导出4423位或通过的42个scope，生产893未替换。旧4acc/v1组合的使用范围保持；不能把本次JSON模型传给它。

#### 固定预测的可达精度下界

对同一模型输入的两次耗时 \(0<a\le b\)，最优单值预测为 \(x=2ab/(a+b)\)，其最小最坏相对误差为：

\[
\delta_{\min}=\min_x\max\{|x/a-1|,|x/b-1|\}=\frac{b-a}{b+a}.
\]

逐次±10%要求两时间之比不超过 \(1.1/0.9=1.222222\)。完整333组校准重复中11组不满足，其中5组是留出点。G1留出4.782074/9.514152秒的误差下界约33.10%，任何固定点预测都无法同时满足原10%门槛。新增诊断脚本记录该下界，不改变原审计标准。**仅增加重复或改变拟合器不能让这批已保留的矛盾时间带通过**；奇数次重复能降低一个慢样本对典型速率的污染，仍需要先处理等待波动，或另行设计长期平均成本与单曲线时间区间的验证合同。

该G1对照的归约launches/poly_muls/coeffs均相同（79/38905/453274），设备t_reduce为0.084/0.083秒、t_reduce_host同为0.013秒。其4.73秒full差不能由这些已记录的归约计时解释；这里没有记录全部NTT/GPU时间，不能推断其他内核全部无差异。整进程GPU采样均值约35.53%/60.33%，慢样本SM最低225MHz；另一个accum长尾SM最低210MHz。采样含冷启动，低时钟与长尾相关，未证明因果。

此前8191/D60060/B2=2078935270/owner640的两条独立验证均被高估约13.7%。阶段对比显示G树预测0.913545秒、实测0.581/0.578秒；训练中B2=1440657287的G树两重复为1.312/0.409秒，二重复中位等于平均，长尾污染传入预测。这与随机验证等待是两种需要分别处理的误差来源。

#### 可复查证据与后续

[完整审计](D:/code/MPA-OpenCl/docs/data/ecm_auto_b2_bridge_20261006_audit.json)、[最终证据与拒绝导出记录](D:/code/MPA-OpenCl/docs/data/ecm_auto_b2_bridge_20261006_evidence.json)、[原始数据ZIP](D:/code/MPA-OpenCl/docs/data/ecm_auto_b2_bridge_20261006_raw.zip)、[逐文件SHA清单](D:/code/MPA-OpenCl/docs/data/ecm_auto_b2_bridge_20261006_raw.json)。ZIP有3510个源文件/日志/存档/测量记录，15,250,313 bytes，逐项解压SHA核验0差异；SHA256为 `6ae459f9dc3c8898d4a7394a1e655aa9499b855170aac78b3109bfc676ea7974`。exe不内嵌，清单明确记录其SHA和本地位置；原审计记录保留绝对路径，复查时须按清单恢复原工作区结构。

新归档工具 [archive_ecm_cost_evidence.py](D:/code/MPA-OpenCl/tools/bench/archive_ecm_cost_evidence.py:25)要求完整采集及integrity通过，允许记录precision失败，核对所有输入后生成ZIP。校准的build_manifest_sha256对应build_manifest.json，验证同名字段对应frozen_sources_manifest.json；归档分别核对两文件，并核对相同binary/全部25个依赖，不能把两文件的不同SHA当成源码变化。

后续优先捕获未被profiling改变的异常时间线，区分CPU准备、CUDA API完成等待与真实GPU执行。再改进重复采样与时间不确定性处理，完成新全范围验证；生产B1/choose12/泛型、总VRAM/RAM准入和原生生产接口验收仍未完成。本轮没有更改精度门槛或宣称通用Auto B2完成。
