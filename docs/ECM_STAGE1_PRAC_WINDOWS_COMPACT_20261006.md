# Stage1 PRAC：素数窗口验证与双临时量 DBL

日期：2026-10-06。上一阶段：[TPI/寄存器与 Nsight 报告](D:/code/MPA-OpenCl/docs/ECM_STAGE1_TPI_REGISTER_TUNING_20261006.md:1)。本阶段以该轮提交 `02e7422` 为基线，继续检验生产 B1 的短时投影与 4608/TPI16 寄存器压力。

## 1. 实验身份

设备为 GPU1，RTX 4060 Laptop，24 SM，sm89；TPB128，param0，Montgomery 构建。B1 为 10,000,000 / 260,000,000，性能采样均使用 sigma=26、`exponent=lcm`。默认算法仍为原 ladder，默认 TPI/档位仍按现有规则选择。

本轮二进制：`build_cuda_cmake/prac/ecm_cuda.exe`，SHA256：

```text
6588996b1ec5f1aee6d16ed30774670b7036ed28c388daa80eb5d4bc756d65a3
```

上一版 `04dcbd6c04853f3e8344133e4c7e392d8f25d7decd1381717f7b065ed1ca1d99`、GMP DLL、编译配置和日志保存到本地 `build_cuda_cmake/prac/before_compact_20261006/`。并行编译 6 个 TU，关键路径 TPI16 471.2 s，总串行工作 784.4 s，然后完成链接。

源码仍保留原有算法与寄存器分派。新候选只支持 4608/TPI16，即当前实验档位中的 N=2^4423−1；不自动推广到其他容器。

## 2. 为什么新增窗口测量

原短采样通常只处理计划前部数百或数千个素数，再根据模乘等价工作 W 投影完整 Stage1。对于 B1=260m，这只覆盖极小比例的计划。W 能修正链长度，却不能提前证明不同 prime 大小、DBL/DADD 比例与链控制具有相同的 W/秒。

本轮使用三个位置：

```text
P = 计划描述记录数
k = min(请求窗口长度, P)
prefix: first = 0
middle: first = floor((P-k)/2)
tail: first = P-k
S_window = product(p_i ** repetitions_i), i ∈ [first, first+k)
W_window = sum(work_i * repetitions_i)
```

默认 k=16，测量间隔使用真实计划记录，没有重新合成 Lucas 链。每轮恢复初始化后的 Montgomery 域曲线，再执行同一窗口。这样不通过不断重复标量乘法使点逐轮增长或意外坍缩。

**middle/tail 是在初始 P 上执行局部子乘积，不是执行完整前缀后得到的真实中途状态。** 它隔离素数链的算术与控制形状，不能作为完整 Stage1 或精确的生产状态重放。三个短窗口也不等于全计划积分。

源码：[窗口设置与索引](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_window.cuh:23)、[恢复与计时](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_window.cuh:76)、[诊断坐标](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_window.cuh:102)。

### 2.1 时间与投影公式

先排除 2 个完整恢复轮次，至少采 3 个有效轮次；默认持续 6 s。窗口 kernel 使用 CUDA events，恢复 D2D、前置缓存/曲线准备和后置诊断导出均不计入 kernel 时间。工具同时记录窗口墙钟与进程墙钟，避免混淆。

设 R 为有效轮次数、C 为曲线批量，T 为这些 kernel 的累计秒数：

```text
单曲线算术速率 = R * W_window / T         [M-equivalent/s]
整批算术速率   = C * R * W_window / T
预计完整 Stage1 秒/曲线 = W_full * T / (C * R * W_window)
```

这个“秒/曲线”是算术速率投影，不是完成曲线。普通生产前缀采样另行进行，不能将恢复窗口的墙钟直接当作生产开销。

### 2.2 内存与传输

设 K 为容器 bits、P 为计划条数：

- 原有曲线设备缓冲：`7*C*K/8 bytes`。
- 窗口专用设备 seed：额外 `7*C*K/8 bytes`，只分配一份。
- PRAC 控制计划：`16*P bytes`，初始化上传一次。
- seed 初次建立：D2D `7*C*K/8 bytes`。
- 每轮恢复：D2D `7*C*K/8 bytes`，总量 `rounds*7*C*K/8`。
- 可选 `window_q.csv`：先出域，再 D2H 曲线缓冲一次；文本记录 X/Z，仅用于诊断，不是 Stage1 save。

4423 位、C1536、K4608 时，每份曲线缓冲 6,193,152 bytes（5.90625 MiB），窗口额外占用同样大小。N8191 用 C768/K9216，其曲线缓冲也恰好相同。B1=260m 的计划为 227,133,760 bytes（216.612 MiB）。这里列的是可计算的组件，未包含 runtime、local memory、GMP 标量与 allocator 保留量，不能当作整个进程/显卡峰值。

窗口开关未启用时不分配 seed，也不增加生产 D2D 恢复。

### 2.3 输出边界

`ECM_PRAC_WINDOW=prefix|middle|tail` 只能用于 PRAC；错误参数拒绝。窗口运行跳过生产检查点读取，完成后返回未完成状态，不调用因子/仿射 save 处理，不生成或修改 `.prac-v1` 检查点。诊断 CSV 明确标记 `partial_product`。

见 [主机隔离入口](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_host.cuh:109) 和 [CLI 前置限制](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1.cu:1153)。

## 3. compact DBL 候选

现有 xDBL：

```text
AA = (X+Z)^2
BB = (X-Z)^2
E  = AA-BB
X' = AA*BB
Z' = E*(BB+a24*E)
```

baseline 使用 AA、BB、E 三个局部域临时量。compact 先生成 X'，随后把 AA 改写为 E；输入 X/Z 已读完，输出 Z 可暂存 a24*E，BB 再复用为 BB+a24*E。最后生成 Z'。模乘/平方数量仍为 **3M+2S**；全部 normalized 操作保持不变。

输出 X/Z 可以覆盖输入 X/Z。它没有把原 DADD 的别名保障推广成假设；DADD 仍使用原三个临时量。源码：[compact DBL](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_kernel.cuh:22)、[调用选择](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_kernel.cuh:102)。

显式设置 `ECM_PRAC_VARIANT=compact`，并选择 `ECM_PRAC_REG_TARGET=255|168`。仅实例化 4608/TPI16，不支持的形状报错。baseline 保持原模式。

### 3.1 编译资源

| 4608/TPI16 PRAC | 实际寄存器 | stack bytes | spill stores / loads bytes |
| --- | ---: | ---: | ---: |
| baseline / natural 255 | 172 | 48 | 0 / 0 |
| baseline / cap168 | 168 | 80 | 36 / 28 |
| compact / natural 255 | **168** | 48 | **0 / 0** |
| compact / cap168 | 168 | 80 | 40 / 32 |

表中 spill bytes 是 ptxas 静态信息，不是整个运行的访存量。natural 与显式 cap168 即使最后显示相同寄存器数，也可能生成不同分配和调度；本轮正出现这种差异。不能由“寄存器相同”推断二者同样无 spill。

natural compact 达到 168 且无 spill，CUDA occupancy API 确认其 TPB128 驻留上限为 3 blocks/SM（baseline natural 为 2）。窗口实测仍更慢，见下节；不能按寄存器或驻留比例承诺加速。

## 4. 窗口吞吐量

72 个串行样本：4423 位 48 个，2203/8191 位各 12 个。每项两次重复，第二次反转窗口和策略顺序；每次 6 s、排除 2 个恢复轮次、窗口 16 条记录。全部指数及 PRAC 计划缓存命中。以下取两次投影中位数，单位 s/curve，**没有完成生产曲线**。

### 4.1 4423 位候选比较

| B1 | 窗口 | baseline natural | baseline cap168 | compact natural | compact cap168 |
| --- | --- | ---: | ---: | ---: | ---: |
| 10000000 | prefix | 5.573387 | 5.720325 | 5.729834 | 5.786701 |
| 10000000 | middle | 5.607161 | 5.702403 | 5.776681 | 5.711196 |
| 10000000 | tail | 5.607165 | 5.686721 | 5.838831 | 5.749182 |
| 260000000 | prefix | 145.862710 | 151.557370 | 155.484915 | 153.128908 |
| 260000000 | middle | 146.812000 | 151.310758 | 155.674481 | 153.341109 |
| 260000000 | tail | 146.898755 | 155.924654 | 158.552611 | 157.357794 |

固定窗口中 baseline natural 最快。compact natural 相对它增加约 2.8%～7.9% 投影耗时；baseline cap168 增加约 1.4%～6.1%。compact 变少的寄存器与 spill 并未转化为窗口吞吐收益。

### 4.2 不同位宽的窗口差异

下表使用 baseline，寄存器请求 255。2203/4423 位采用 natural；9216-bit 容器的 255 请求按既有分派使用 per-tier（不是放开该档位的寄存器限制）。2203/4423 位 C1536/TPI16，8191 位 C768/TPI32，均提交 grid192；资源限制的驻留 blocks/SM 分别为 4/2/3。

| N bits | B1 | prefix | middle | tail | tail/prefix−1 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2203 | 10000000 | 2.149293 | 1.904930 | 1.913420 | -10.97% |
| 2203 | 260000000 | 56.256873 | 50.057505 | 50.307240 | -10.58% |
| 4423 | 10000000 | 5.573387 | 5.607161 | 5.607165 | +0.61% |
| 4423 | 260000000 | 145.862710 | 146.812000 | 146.898755 | +0.71% |
| 8191 | 10000000 | 24.671890 | 20.442377 | 20.721297 | -16.01% |
| 8191 | 260000000 | 687.191402 | 551.766447 | 582.568205 | -15.22% |

2203/8191 位后部窗口的 W/秒高于最初 16-record 前缀，4423 natural 则较接近。8191/B1=260m 的 prefix 两次值为 674.905126～699.477679（约 3.64% 跨次差异），不能把中位数当成高精度最终耗时。

由此可排除“W 权重已经保证所有 prime 区间具有同一速率”的假设；窗口状态、启动长度和调度仍会改变实测。当前数据不足以给出完整生产积分，也不足以直接修正 Auto B2 的 T1。

### 4.3 实际测量范围

| B1 | full W | prefix W | middle W | tail W | middle first | tail first |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10000000 | 128750918 | 2753 | 3164 | 3321 | 332281 | 664563 |
| 260000000 | 3369476895 | 3319 | 3861 | 4024 | 7097922 | 14195844 |

B1=10m 的 middle p=4750919～4751167，tail p=9999749～9999991；B1=260m 的 middle p=124772083～124772321，tail p=259999687～259999991。prefix 均为 p=2～53，重复次数随 B1 改变。各窗口工作量只占全计划很小部分。

按实际记录重算的每曲线 DBL/DADD 次数：10m 的 prefix 为 175/313、middle 为 58/479、tail 为 63/501；260m 对应 209/379、75/581、86/599。DBL 占 W 的比重由前缀约 31.5%～31.8% 降到中后部约 9.2%～10.7%。这是成本组成差异的直接证据；其是否解释具体硬件速率差异仍需计数或控制变量实验，不能只由该比重判定。

原始汇总在本地 `docs/data/stage1_prac_windows_{2203,4423,8191}_20261006/summary.json`，按仓库约定不提交原始日志。

### 4.4 尾部长度复查

再加 12 个样本：N4423/C1536，B1=260m，tail 窗口 8/32 条，各测 baseline natural、baseline cap168、compact natural 两次。与已有 16 条窗口合并如下，单位 s/curve。

| tail records | baseline natural | baseline cap168 | compact natural | cap168/natural−1 |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 146.973299 | 143.576587 | 145.866349 | -2.31% |
| 16 | 146.898755 | 155.924654 | 158.552611 | +6.14% |
| 32 | 146.903516 | 172.617872 | 173.626090 | +17.50% |

baseline natural 在三种长度下接近 146.9 s/curve；cap168 在 8 条时胜出，16/32 条时变慢。compact 也有类似长度敏感性。新增窗口 prime 区间分别为 259999849～259999991、259999307～259999991；长度改变同时改变选定子乘积，不能视为完全相同算术的纯 launch-length 实验。

这把下一轮诊断收敛到“较高驻留数下的长窗口执行/调度敏感性”：需要固定相同 prime 子序列，比较切片方式和尾部计数，再判断是否应调寄存器或内核结构。当前不能全局固定 cap168，亦不能把 8 条窗口胜出当成生产最优 chunk=8。

额外汇总位于本地 `docs/data/stage1_tail_count{8,32}_{natural,cap}_20261006/summary.json`。

## 5. 普通生产前缀 A/B

[普通路径配对脚本](D:/code/MPA-OpenCl/tools/bench/bench_stage1_prac_variants.py:40) 串行运行 5 策略×2 B1×2 次重复，共 20 个样本。N4423，C1536，4608/TPI16，TPB128，grid192。15 s 采样、排除前 5 s，取末约 5 s 进度投影的中位数，再合并两次独立运行；第二次反序。与上一轮相同口径，不是完成曲线墙钟。

| B1 | resident | baseline natural | baseline cap168 | compact natural | compact cap168 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10000000 | 5.365739 | 5.612158 | 5.240895 | 5.393076 | 5.270721 |
| 260000000 | 139.511599 | 146.827318 | 138.773042 | 143.225447 | 140.303701 |

全部 20 次指数缓存命中，16 次 PRAC 计划缓存命中。全部在生产分片边界正常结束采样，返回不完整状态，只有检查点、没有完整 save，工具没有强制终止。实际批内点状态连续递进，与窗口的逐轮 seed 恢复不同。

### 5.1 策略判断

- B1=10000000：compact natural 比 baseline natural 减少 3.90% 投影耗时，但比已存在的 baseline cap168 增加 2.90%。compact cap168 相对 baseline cap168 增加 0.57%。
- 同档 baseline cap168 相对 resident 减少 2.33% 投影耗时；相对 baseline natural 减少 6.62%。该项仍只是当前前缀证据。
- B1=260000000：compact natural 比 baseline natural 减少 2.45% 投影耗时，但比已存在的 baseline cap168 增加 3.21%。compact cap168 相对 baseline cap168 增加 1.10%。
- 同档 baseline cap168 相对 resident 减少 0.53% 投影耗时；相对 baseline natural 减少 5.49%。该项仍只是当前前缀证据。

因此不将 compact 设为默认，也不以其无 spill 推断收益。原有 baseline cap168 是当前普通短采样的更好候选，但窗口中尤其 260m tail 出现相反排名，尚不具备全计划最优证据。

生产分片按 event 时间自适应到约 80～120 ms；固定窗口为 16 records，而且首 16 条含较多重复小素数。点状态、prime 区间、操作组成、启动长度同时变化。当前比较证明排名对测量范围敏感，尚不能单独归因于启动长度或状态。

原始汇总：本地 `docs/data/stage1_prac_variants_20261006/summary.json`。

### 5.2 管理员 Nsight Compute

三次串行管理员采集均成功，NCU 2026.2.1，GPU1、B1=10m、C1536、grid192、TPB128、当前同一二进制。skip=2、count=1，分别采 MODE4/5/6 的第二个 PRAC kernel；应用日志均在首片 next=16 后推进到 next=30，即相同 14-record 范围。每次 17 passes，关闭 clock/cache 控制。

| 指标 | baseline natural | baseline cap168 | compact natural |
| --- | ---: | ---: | ---: |
| 每线程寄存器 | 172 | 168 | 168 |
| 每线程实际分配寄存器 | 176.000000 | 168.000000 | 168.000000 |
| 寄存器允许 blocks/SM | 2.000000 | 3.000000 | 3.000000 |
| achieved occupancy (%) | 16.370581 | 22.157801 | 22.446961 |
| eligible warps/scheduler | 0.356952 | 0.455233 | 0.446422 |
| issue active (%) | 30.644785 | 31.608055 | 30.985519 |
| wait / issue active | 3.898981 | 4.215814 | 4.279275 |
| math pipe throttle / issue active | 0.318370 | 0.835976 | 0.873252 |
| no instruction / issue active | 0.073879 | 0.723346 | 0.811855 |
| SM throughput (%) | 72.339087 | 71.583425 | 68.784593 |
| DRAM throughput (%) | 0.000396 | 0.000178 | 0.000371 |

GPC 平均频率分别为 1.799216 / 1.799648 / 1.799560 GHz。未锁频，所以这仍不替代普通基准。

“/ issue active” 是硬件计数比值，不能当成耗时百分比，也不能直接相加得出运行时间。这里不引用被 replay 扭曲的应用 s/curve。

更高 occupancy/eligible 并未消除 wait，math pipe throttle 与 no instruction 比值还上升。compact 与 cap168 同为 168、3 blocks/SM，但 compact 的 eligible、issue 和 SM throughput 稍低，普通 A/B 也更慢。证据支持继续查指令/依赖与调度，不能宣称已定位到某条 DBL 指令。

DRAM throughput 极低，采集 kernel 缺乏外存带宽饱和证据。此指标不能排除 L1/local 的 spill 成本，也不描述计划上传或 CPU 曲线准备。下一轮重点应是算术执行与指令供给，而非仅增加 occupancy。

原始报告、CSV 与 `quantitative.json` 位于本地上述三个 `docs/data/stage1_compact_ncu_*` 目录。

## 6. 正确性

compact natural / cap168 分别通过 160 条完整 Q/因子 save 比较，另各 3 个退化拒绝检查。覆盖 4423 位、lcm/choose12、64 位 sigma、坏计划和检查点续跑/损坏回退；其他小位宽因子案例仍使用 baseline。

新增窗口门禁由 [整数 Montgomery ladder](D:/code/MPA-OpenCl/tools/test/test_cuda_prac_windows.py:20) 计算独立期望。选定记录的 p 做素性检查，repetitions 按 B1/torsion 独立重算，工作量再用 Python PRAC 计数器核对；坐标通过交叉乘法比较，不要求 projective 缩放相同。

最终窗口门禁：536 条子乘积 Q、8 条 compact 检查点恢复为 baseline 的完整 Q，及 8 个错误输入拒绝案例全部通过。包含三个主力 N、两种 TPI、两种 torsion、B1=2 边界、62 位 sigma、B1=10m/260m 的前中后 16-record 窗口。benchmark 前后有效检查点的字节摘要不变，之后可继续正式 Stage1 并与 CPU/GMP save 逐条一致。

首轮门禁在生产规模投影字段的舍入校验上停下：native 打印六位小数的累计 kernel_ms 被 `W_full/W_window` 放大，固定 1 μs 容差过紧。已使用这两个打印字段精度推导误差界，再用新目录完整重跑；没有修改 GPU 坐标比较的精度或标准。

默认 per-tier 路径在新二进制上追加通过 376 条完整 Q/因子 save 比较及 3 个退化拒绝检查，汇总 `docs/data/stage1_baseline_post_compact_gate_20261006/summary.json`。本阶段合计 1240 条 Q/save 比较、9 个退化拒绝和 8 个错误输入拒绝；重复对照也计入比较数，不能理解为 1240 个独立曲线种子。

尾部长度复查另用同一独立整数 oracle 验证 8/32-record、B1=260m 的 baseline natural/cap168/compact natural，各 8 曲线，共 48 条 Q 全通过，汇总 `docs/data/stage1_tail_length_q_gate_20261006/summary.json`。因此本阶段累计 **1288 条 Q/save 比较**。该补充门禁批量较小，证明算术与窗口边界，不证明 C1536 的吞吐或完整 B1 曲线。

最终汇总位于本地 `docs/data/stage1_prac_window_gate_20261006_v2/summary.json`；首次失败日志保留，不计为已通过验收。未测完整生产 B1 曲线完成时间。

## 7. 复现

```powershell
python tools/test/test_cuda_prac.py --bits 4423 --tpi 16 --registers 255 `
  --variant compact --device 1 --output docs/data/my_compact_q_gate
python tools/test/test_cuda_prac_windows.py --production --device 1 `
  --output docs/data/my_window_q_gate
# 扩展验收 8/16/32-record 的前中后子乘积；包含基础门禁，运行更久。
python tools/test/test_cuda_prac_windows.py --production --production-counts 8 16 32 `
  --device 1 --output docs/data/my_extended_window_q_gate

python tools/bench/bench_stage1_prac_windows.py --bits 4423 --curves 1536 `
  --b1 10000000 260000000 --variants baseline compact --registers 255 168 `
  --seconds 6 --warmup 2 --repeats 2 --device 1 `
  --exp-cache build_cuda_cmake/prac --output docs/data/my_prac_windows

python tools/bench/bench_stage1_prac_variants.py --curves 1536 `
  --b1 10000000 260000000 --seconds 15 --warmup 5 --repeats 2 --device 1 `
  --exp-cache build_cuda_cmake/prac --output docs/data/my_prac_variants
```

`--warmup` 在窗口工具中是排除的轮次，在原普通采样工具中是秒数。所有实验串行使用 GPU1；复用已有 B1/PRAC 缓存。新环境开关要在正式运行前清除窗口设置；工具自身会清除继承的窗口开关，正式 A/B 不会意外落入微基准路径。

## 8. 后续

本阶段完成 84 个恢复窗口样本、20 个普通生产前缀样本及 3 个管理员 NCU 报告。保留 compact 为显式实验候选，不改默认算法、默认 TPI 或默认寄存器策略。它只在部分对照中胜过 natural，尚未胜过已存在的普通路径 cap168，也存在长尾窗口退化。

下一阶段顺序：

1. 对同一尾部 prime 子序列固定输入，比较 8/16/32 切片与不同曲线批量；再采对应 tail 的 NCU，检查 no instruction、wait 与 math pipe throttle 的变化。先分清算术形状与分片形状的影响。
2. 沿 DADD/DBL 的 normalized 算术检查依赖与指令供给。DBL 在大素数窗口只占 W 约一成，下一候选优先检查占比更高的 DADD，或共同使用的 Montgomery 乘法/平方。
3. 若某候选获益，按 384/768/1536 批量与两档生产 B1 再测，继续完整 Q/save 门禁；TPI32 对照保持半数曲线与相等提交 grid。不得由提交 grid 宣称相等驻留 blocks/SM。
4. Auto B2 的 T1 profile 不自动导入这批子窗口或前缀投影。需覆盖实际计划成本组成与完整执行范围，才能发布生产成本参数。

本实验构建的 exe 由 2,849,792 增至 3,080,704 bytes，增加 230,912 bytes（225.5 KiB）；额外实例也增加 TPI16 编译时间。后续若长期没有生产收益，应删除 compact kernel 实例及选择开关，保留可复用窗口工具和失败证据。
