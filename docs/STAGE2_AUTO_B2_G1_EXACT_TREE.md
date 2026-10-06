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
