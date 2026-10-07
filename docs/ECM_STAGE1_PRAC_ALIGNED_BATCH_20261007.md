# Stage1 param0：按驻留容量选择批量的实验

## 1. 范围与版本

基线 `4389d74`，不改变 CUDA 数学内核或默认配置。exe SHA256 为 `d1a5b6476c17c95762283fd0d30205e5175b3d37ccf49198fa2a99c2476d7370`；40项 Stage1 源码、工具及实际 CGBN 核心 SHA 与该提交快照一致。本轮使用 GPU1 RTX4060 Laptop、24SM、sm89、N4423/容器4608、TPB128、sigma26、lcm。

上一轮 cap128 增加驻留容量但产生大量 spill，未超过 cap168/C1536 的最佳吞吐。本轮优先检查 cap168 的批量是否还有改进空间。

### 1.1 默认 TPI、实验 TPI 与 TPB 的澄清

N4423默认仍选择4608容器/TPI16。报告中的4423/TPI32指[显式环境变量覆盖](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_host.cuh:93) `ECM_STAGE1_TPI=32`，即32线程共同处理一条曲线；并未把该位宽的生产默认切换到TPI32。[默认类型](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_kernel.h:1201)为8192容器/TPI16、9216及以上容器/TPI32。边界按容器判断，选择要求 `BITS >= Nbits+6`，所以N8191已选择9216/TPI32。

[当前TPB默认128](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_kernel.h:104)，源码记载2026-09-25由256改为128。寄存器上限不会自动调整TPB；固定TPB时，每block寄存器资源近似 `TPB*R_alloc`，其中 `R_alloc` 是硬件分配后的每线程数量，不能直接使用寄存器上限。在GPU1的65536个32-bit寄存器/SM预算下，分配176→168、TPB128时寄存器容量为 `floor(65536/(128*176))=2` → `floor(65536/(128*168))=3`；TPB256时两者均为1。还需考虑warp、线程、共享内存和block数量限制，最终采用CUDA容量查询并用NCU测量平均活跃warp。

同理，本轮TPI16 cap168实际分配168，而TPI32 cap168实际分配112，固定TPB128下容量分别3/4。TPI32曲线数减半保证提交grid相同，不能保证实际驻留相同。

## 2. 几何与假设

[提交尺寸](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_host.cuh:168)为 `G=ceil(C*TPI/TPB)`，[容量查询](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_host.cuh:179)为 `cudaOccupancyMaxActiveBlocksPerMultiprocessor`。容量不是实测平均驻留；固定TPB128时每block曲线数为 `128/TPI`。

已测 cap168：TPI16实际162寄存器、硬件分配168、容量3blocks/SM，一轮满驻留曲线数为 `24*3*8=576`；显式TPI32同一共享布局实际109、分配112、容量4，一轮为 `24*4*4=384`。两者的寄存器上限都为168，实际分配不同。

候选 TPI16 C576/1152/1536/1728/2304，对应 TPI32 C288/576/768/864/1152；提交grid分别72/144/192/216/288。C1536是既有最佳对照。C2304/1152同时是576/384的整数倍，提交288 blocks分别相当于容量72/96 blocks的四倍/三倍，避免只让某个TPI得到整倍数批量优势。

`ceil(G/(SMs*resident_capacity))`只作为批量设计的分组数代理，CUDA不会在每个这样的组间设置全局屏障；不能用它直接推导运行时间或宣称实际blocks/SM相等。

三个可证伪假设：

1. 容量整数倍批量减少不充分利用的尾部，平均活跃warp和曲线吞吐提高。
2. 如果依赖链/指令发射占主导，几何对齐收益很小。
3. 更大批量可能加重每次切片/恢复代价；固定chunk4/32与普通50ms前缀可能表现不同。

## 3. 实验方法

- 先运行完整Q/save门禁：TPI16/C2304与TPI32/C1152，均single-compact/cap168/50ms，N4423、lcm/choose12、64-bit sigma、checkpoint及公共边界。
- 同二进制配对：`shared16cap168`与`shared32cap168`，相同共享点布局、normalized Montgomery及 `W=6ADD+5DBL`；每个TPI32配置使用一半曲线数。
- 固定tail32：B1=260m、chunk4/32、6秒采样、两个排除的warmup轮；两策略×五grid×两chunk×两反序=40样本。
- 普通前缀：B1=10m/260m、目标50ms、15秒采样、warmup5秒，最后五秒精确`s/curve`中位；两B1×两策略×五grid×两反序=40样本。
- GPU1串行，计时不编译、不导出大SASS、不运行profiler。逐样本验证exe SHA、缓存命中、实际C/TPI/容器/grid及完整窗口标量。

曲线吞吐为 `1/projected_s_per_curve`，不同C不比较整个batch完成投影。固定窗口的事件/墙钟投影分别保留；普通前缀正常sample-limit/checkpoint-only退出，不发布为完整Stage1 save或已认证Auto B2 T1。

## 4. 计算量、容量与传输

每curve标量运算计数和容器不随C变化，batch计算量为 `C*W`。显式data/seed各 `7*C*4608/8` bytes；GPU计划仍 `16P` bytes，不随C变化；每轮边界逻辑量为 `6*C*4608/8*launches`。相同TPI和chunk下每curve逻辑字节量不变。

两TPI的CGBN分组/WMAD主循环差别仍按上一轮解释：TPI16有效144槽、TPI32填充160槽，主循环源码MAC代理每curve为 `4S²W`，不等同退休指令或周期。批量选择不减少该每curve工作量。

device memory.used采样只能提供设备用量记录，不将模块容量相加作为进程峰值，也不把累计cache local sectors视为容量或PCIe传输。

## 5. 正确性与计时结果

两档大批量完整Q门禁均通过：TPI16/C2304共20848个Q、27个case，候选主运行及续跑6944个Q；TPI32/C1152共10480个Q、27个case，候选主运行及续跑3488个Q。三组完整save中，共同sigma的3456个affine Q跨TPI一致，各自同时通过独立CPU结果检查。

固定40与普通前缀40样本全部完成。独立核对预期矩阵、冻结的40项源码/工具/实际CGBN SHA、exe SHA、原始日志、实际几何、缓存命中、退出路径与投影公式；普通前缀从原始进度及精确slice日志重建所有样本和最后五秒中位数。无强制终止或完整Stage1 save；固定窗口不写生产checkpoint。

### 5.1 普通50ms前缀

以下均为两次反序运行各自最后五秒中位数的中位数，单位s/curve。不是完整曲线实际墙钟。最后一列是TPI16同轮相对C1536的曲线吞吐变化 `T1536/TC-1`。

| TPI16 C / TPI32 C | B1=10m TPI16 | B1=10m TPI32 | B1=260m TPI16 | B1=260m TPI32 | TPI16吞吐变化：10m / 260m |
| --- | ---: | ---: | ---: | ---: | ---: |
| 576 / 288 | 5.167954 | 7.474274 | 135.254169 | 195.837867 | −3.13% / −3.08% |
| 1152 / 576 | 5.071028 | 7.211944 | 132.726458 | 188.736063 | −1.28% / −1.24% |
| 1536 / 768 | 5.006023 | 7.096039 | 131.085710 | 185.658477 | 对照 |
| 1728 / 864 | 5.007428 | 7.107967 | 131.019987 | 185.997736 | −0.03% / +0.05% |
| **2304 / 1152** | **4.960886** | **6.952208** | **129.855046** | **181.895808** | **+0.91% / +0.95%** |

TPI16/C2304两次投影范围：10m为4.960216..4.961556，260m为129.834209..129.875882；同轮C1536为5.005487..5.006558和131.036174..131.135246。TPI32/C1152相对本TPI C768提高约2.07%，在相同grid288下仍比TPI16/C2304曲线吞吐低28.64%/28.61%。本轮shared32结果仅适用于该共享点布局，不能外推为所有TPI32实现的最优值。

### 5.2 固定tail32窗口

B1=260m、同一32个prime子乘积、`W=8053`，完整计划 `Wfull=3369476895`；已剔除两个warmup轮。下表以窗口墙钟投影为主要指标，同时列事件时间投影，单位s/curve。

| TPI16 C | chunk4：事件 / 墙钟 | chunk32：事件 / 墙钟 | 墙钟吞吐相对C1536：chunk4 / chunk32 |
| --- | ---: | ---: | ---: |
| 576 | 135.435138 / 136.370315 | 137.181635 / 137.365501 | −3.61% / −1.68% |
| 1152 | 132.738698 / 133.320779 | 136.140125 / 136.227357 | −1.41% / −0.86% |
| 1536 | 131.028548 / 131.446652 | 134.984056 / 135.051856 | 对照 |
| 1728 | 131.015946 / 131.370606 | 135.205671 / 135.269859 | +0.06% / −0.16% |
| **2304** | **129.981047 / 130.235439** | **134.444665 / 134.502844** | **+0.93% / +0.41%** |

TPI32/C1152墙钟为182.398521/181.741936，相对本TPI C768提高2.34%/2.10%；同grid下仍比TPI16吞吐低28.60%/25.99%。

固定窗口中chunk4的投影优于chunk32，但两者prime组间的边界、点保存/恢复和调度机会不同，不能仅据此将生产切片目标直接改为chunk4或推断数学MAC减少。普通前缀使用原有50ms反馈，本轮只调批量。

### 5.3 容量与采样环境

C1536→2304增加50%的显式curve data容量：`7*C*576`由5.90625MiB增至8.859375MiB；固定窗口seed容量同样从5.90625增至8.859375MiB。普通前缀不分配该window seed。260m GPU计划 `16*14195860=227133760` bytes，即216.611633MiB，未随C增长。每curve算术量和每curve逻辑边界字节均不变，总batch工作量与curve数组线性增长。

固定采样614条、利用率≥70%的478条；前缀1270条、其中1171条。两批忙时SM频率均1800MHz；温度范围57..72°C、64..73°C；设备memory.used采样最大331/321MiB。采样覆盖进程间空闲和准备时间，忙时比例不等同数学内核利用率；这些设备内存读数不构成进程显存峰值。

## 6. NCU驻留分析

计时结束后串行采集TPI16/C576、C1536、C2304与TPI32/C1152，均cap168/single-compact、tail32/chunk32、warmup2、launch-skip2。四份管理员采集及CSV导出均exit0，实际passes分别18/19/20/19；exe、环境、输入、窗口、实际kernel/TPI/grid、原始CSV SHA均核对。profiler重放时长不用于吞吐比较，累计sector计数不再乘passes。

| TPI / C / grid | 实际 / 分配寄存器 | 寄存器容量blocks/SM | 活跃warp/SM | eligible warp/SMSP | issue active | L1TEX throughput |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 / 576 / 72 | 162 / 168 | 3 | 10.701702 | 0.480193 | 33.300823% | 24.704725% |
| 16 / 1536 / 192 | 162 / 168 | 3 | 11.001760 | 0.482875 | 33.775159% | 25.098987% |
| 16 / 2304 / 288 | 162 / 168 | 3 | 11.343689 | 0.492164 | 33.945491% | 25.238167% |
| 32 / 1152 / 288 | 109 / 112 | 4 | 14.628848 | 0.738049 | 39.507610% | 41.444316% |

C1536→2304的TPI16平均活跃warp提高3.11%，eligible提高1.92%，issue active仅增加0.170332个百分点；寄存器与驻留容量不变。支持更大batch略改善调度利用，但不能从一次重放的平均指标把0.9%普通前缀收益唯一归因于尾部消除。

TPI32虽有更多活跃warp/eligible、更高issue active，曲线吞吐仍下降。TPB128下16每block有8条curve，32有4条；驻留容量是24curve/SM与16curve/SM。完整warp中TPI16处理两条curve、TPI32一条，不能仅以活跃warp判定曲线吞吐。另有WMAD填充与每curve主MAC代理增加，见第4节和上一轮报告；这些现象不能替代动态退休指令/周期的完整归因。

| TPI / C | wait | no_instruction | short_scoreboard | long_scoreboard | math_pipe_throttle | local LD / ST sectors |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 / 576 | 4.157818 | 0.303114 | 0.328122 | 0.000067 | 0.929774 | 0 / 8064 |
| 16 / 1536 | 4.156323 | 0.432684 | 0.326053 | 0.000049 | 0.897105 | 0 / 21676 |
| 16 / 2304 | 4.190334 | 0.543773 | 0.326109 | 0.000046 | 0.945127 | 0 / 31944 |
| 32 / 1152 | 3.624420 | 0.071972 | 1.067948 | 0.000091 | 1.508976 | 0 / 31968 |

这些stall列均为 `per_issue_active.ratio`，不是墙钟时长占比；no_instruction不是直接I-cache miss计数，short_scoreboard不能全部解释为shuffle或local load。TPI16更大batch的wait/no_instruction/math_pipe_throttle未下降，说明单一stall趋势无法代替最终吞吐。

local LD全0，累计local sector×32字节代理分别0.246094/0.661499/0.974854/0.975586MiB，未出现cap128的新增local load代价。该量不是显存占用、DRAM流量或PCIe传输。DRAM throughput四份仅0.000086%..0.000129% of peak；此窗口没有DRAM带宽饱和证据，不用于否定完整CPU准备/传输阶段的瓶颈。

## 7. 结论与下一步

在GPU1/N4423/当前共享点布局上，TPI16/cap168/C2304/50ms是本轮最佳；相对同轮C1536提升约0.9%，完整Q门禁通过。可以作为该设备该位宽的显式运行配置，不修改全局默认TPI、TPB或曲线批量。

仅匹配驻留容量整数倍并不足以产生明显提升：C1728相对1536几乎无变化，C2304也只有约1%，并非用整组填充率直接推算的约7%。NCU显示活跃warp与eligible略升、issue提高有限；批量尾部、跨block进度差异与指令发射都可能参与，当前数据不能把收益唯一归因于尾部消除。

后续优先考察私有disjoint xADD输出寄存器复用：把一个局部bn的生命周期移入已证明与输入互不别名的输出，保持4M+2S、归一化与公开别名语义。先检查完整SASS和实际分配；源码少一个变量本身不构成寄存器下降或速度提升证据。与当前冻结二进制比较时采用C1536和C2304对照。

## 8. 复现

```powershell
python tools/test/test_cuda_prac.py --exe build_cuda_cmake/prac/ecm_cuda.exe `
  --bits 4423 --tpi 16 --registers 168 --variant single-compact --curves 2304 `
  --target-ms 50 --device 1 --output docs/data/aligned_t16_fullq
python tools/test/test_cuda_prac.py --exe build_cuda_cmake/prac/ecm_cuda.exe `
  --bits 4423 --tpi 32 --registers 168 --variant single-compact --curves 1152 `
  --target-ms 50 --device 1 --output docs/data/aligned_t32_fullq
python tools/bench/bench_stage1_prac_tpi_pairs.py --exe build_cuda_cmake/prac/ecm_cuda.exe `
  --configs shared16cap168 shared32cap168 --curves16 576 1152 1536 1728 2304 `
  --mode windows --b1 260000000 --seconds 6 --exp-cache build_cuda_cmake/prac `
  --output docs/data/aligned_fixed
python tools/bench/bench_stage1_prac_tpi_pairs.py --exe build_cuda_cmake/prac/ecm_cuda.exe `
  --configs shared16cap168 shared32cap168 --curves16 576 1152 1536 1728 2304 `
  --mode prefix --b1 10000000 260000000 --target-ms 50 --seconds 15 --warmup 5 `
  --exp-cache build_cuda_cmake/prac --output docs/data/aligned_prefix
```

本輪比较相同共享布局下的TPI与批量；上轮TPI32 baseline布局曾比shared更快，因此不将本轮shared32排名外推为所有TPI32内核的最优值。

## 9. 原始证据

本轮沿用 `4389d74` 的只读二进制/对象快照 `build_cuda_cmake/prac/after_shared_dbl_cap128_20261007/`，未重编或修改Stage1源码。原始证据均位于已排除的 `docs/data/`，不提交实验数据：

- `stage1_prac_aligned_gates_audit_20261007.json`：完整Q门禁与跨TPI共同sigma核对、40项源码SHA。
- `stage1_prac_aligned_pairs_fixed_20261007/{summary,audit}.json`、`stage1_prac_aligned_pairs_prefix_20261007/{summary,audit}.json`：80样本完整矩阵、原始日志重建、逐样本SHA/几何/缓存/退出和遥测。
- `stage1_prac_aligned_ncu_t16_c{576,1536,2304}_20261007/`、`stage1_prac_aligned_ncu_t32_c1152_20261007/`：四份管理员原始报告/CSV、应用及profiler日志、环境与命令。
- `stage1_prac_aligned_ncu_analysis_20261007.json`：原始CSV指标、真实pass数量、CSV SHA、寄存器/容量核对。
- `audit_stage1_prac_aligned_20261007.py`、`audit_stage1_prac_aligned_ncu_20261007.py`：独立只读结果核对，未重新运行GPU计时。
