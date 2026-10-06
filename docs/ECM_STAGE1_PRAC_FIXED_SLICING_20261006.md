# Stage1 PRAC：固定子乘积切片与尾部调度

日期：2026-10-06。基线提交 `0ddb698`。接续 [窗口与 compact DBL 实验](D:/code/MPA-OpenCl/docs/ECM_STAGE1_PRAC_WINDOWS_COMPACT_20261006.md:1)。本阶段检验上一轮尾部长度敏感性；生产默认算法与 GPU 算术实现均沿用基线。

## 1. 实验身份与范围

GPU1：RTX 4060 Laptop，24 SM，sm89，TPB128。param0、Montgomery、sigma=26、lcm。固定窗口矩阵和尾部 NCU 使用的 exe SHA256：

```text
81079e19a7cd64d72985a622a13ad7aa66c0a5a0578dbd9bf5c2e9687a0e64ef
```

上一版 exe/配置及 PRAC GPU 对象摘要保留在本地 `build_cuda_cmake/prac/before_slicing_20261006/`。两次主机 TU 编译各 11.4 s；最终只链接。5 个 PRAC GPU TU 对象 SHA256 均与此前相同，没有用重新编译的算术内核混入切片对照。

恢复窗口是初始点上的局部子乘积，不是真实完成前缀后的生产点。所有“s/curve”均为工作量投影；没有等待完整生产 B1 曲线完成，不自动发布为 Auto B2 成本。

本轮 N4423 默认及性能矩阵均为 4608/TPI16。此前 N4423/TPI32 是显式对照实例，不是默认分档改变。容器需满足 Nbits+6；正常容器2560..8192使用TPI16，9216..16384使用TPI32，因此N8191会选9216/TPI32。当前TPB128与历史TPB256不同；寄存器上限不会自动改变TPB。参见 [容器选择](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_host.cuh:87)、[TPB默认](D:/code/MPA-OpenCl/CMakeLists.txt:433)、[寄存器驻留约束](D:/code/MPA-OpenCl/docs/ECM_STAGE1_TPI_REGISTER_TUNING_20261006.md:32)。

## 2. 控制变量

上一轮窗口长度改变了 prime 集合。本轮固定 `count=32`，位置相同，每轮只恢复一次 seed，连续完成同一 32-record 子序列。比较每次启动处理的 `chunk=4/8/16/32`；结果必须对应同一标量子乘积。

`ECM_PRAC_WINDOW_CHUNK=0` 或未设：一个 kernel 完成窗口。正值必须不超过实际选定记录数；最后一片可不足 chunk。默认生产路径不读取这个切片参数。

每片都像生产路径一样同步、检查 CGBN 错误；片间只保留 AX/AZ，下一记录重新建立 PRAC 链工作点。没有片间 seed 恢复、EXPORT、CPU 点运算或 D2H 坐标传输。

源码：[窗口配置](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_window.cuh:23)、[每轮切片](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_window.cuh:86)、[测量解析](D:/code/MPA-OpenCl/tools/bench/bench_stage1_prac_windows.py:16)、[批量与反序矩阵](D:/code/MPA-OpenCl/tools/bench/bench_stage1_prac_slicing.py:47)。

### 2.1 时间公式

设 C 为批量曲线数，W 为固定子序列的每曲线模乘等价工作，F 为全计划工作，R 为排除预热后的轮数，J=ceil(count/chunk)。

```text
K_ms = sum(all measured per-slice CUDA event milliseconds)
H_ms = sum(measured round wall milliseconds)
device projected s/curve = F*K_ms/(1000*C*R*W)
wall projected s/curve   = F*H_ms/(1000*C*R*W)
```

K 不包括片间主机等待间隔及 seed 恢复；H 包括每轮一次 D2D 恢复、启动/事件/同步与错误检查，排除轮后进度打印。两者都排除指数/计划读取、CPU 曲线构造、首次上传和诊断导出。完整 `window_wall_seconds` 还包括预热与输出，不能直接除以有效轮数。

### 2.2 访存与容量

设 B 为容器 bits。设备曲线数组和额外 seed 各 `7*C*B/8 bytes`；控制计划为 `16*P bytes`（P 为计划记录数），不因切片再上传。每轮恢复 D2D 一次 `7*C*B/8`。

每个 PRAC kernel 边界逻辑读取 N、a24、AX、AZ，写回 AX、AZ，合计 `6*C*B/8 bytes`。一轮 J 片为 `J*6*C*B/8`；相对不切片多 `(J−1)*6*C*B/8`。这是源码访问量，**不是实测 DRAM 流量**，可能命中缓存；不包含 prime 控制读取、spill 或内部线程通信。

4608-bit/C1536 时，每份曲线/seed 为 5.90625 MiB，单片边界逻辑量为 5.0625 MiB；32条窗口 chunk4/8/16/32 的 J=8/4/2/1，对应 40.5/20.25/10.125/5.0625 MiB 每轮。显存容量不随 J 成倍分配；两事件句柄逐片复用。

## 3. 正确性与隔离

独立整数 Montgomery ladder 验证 1160 个窗口 Q，并有 8 个完整恢复 Q；10 个错误输入拒绝通过。覆盖 lcm/choose12、三个 N 位宽、TPI16/32、compact 两策略、64位 sigma、B1边界、检查点隔离及生产 B1=10m/260m。

生产 32-record 窗口使用 chunk7/8/16/32；chunk7 的最后一片为4条。72次跨切片诊断 CSV 字节摘要对比全部一致，证明同一顺序算术分片后 AX/AZ 不变；另以独立 oracle 检查数学结果。小批量门禁不能证明 C1536 吞吐，也不是完整生产 Stage1 曲线。

汇总：本地 `docs/data/stage1_slicing_q_gate_20261006/summary.json`。窗口仍不读取/修改生产检查点、不发布 Stage1 save。

## 4. 固定范围性能

GPU1、B1=260m、N4423/4608/TPI16，固定 tail 32 条记录：prime 259999307～259999991，W=8053，F=3369476895。C384/768/1536 × 寄存器 natural/168 × chunk4/8/16/32 × 2 次，共 48 个有效样本；第二轮反序。指数与计划缓存均命中。以下为两次中位数，单位 s/curve；每格为 CUDA event / 测量墙钟投影。

| C / 策略 | chunk4 | chunk8 | chunk16 | chunk32 |
| --- | ---: | ---: | ---: | ---: |
| 384 / natural | 147.918821 / 149.009709 | 147.594871 / 148.214039 | 147.379716 / 147.756349 | 147.292162 / 147.513394 |
| 384 / cap168 | 149.941598 / 151.053245 | 149.601757 / 150.226578 | 149.403539 / 149.765242 | 149.324262 / 149.554860 |
| 768 / natural | 147.488359 / 148.071558 | 147.281717 / 147.614927 | 147.188576 / 147.377133 | 147.129201 / 147.243193 |
| 768 / cap168 | 147.871924 / 148.469482 | 147.789440 / 148.119098 | 160.211650 / 160.397433 | 182.196187 / 182.317385 |
| 1536 / natural | 147.100643 / 147.409425 | 146.986284 / 147.151184 | 146.935210 / 147.030659 | 146.895801 / 146.956639 |
| 1536 / cap168 | 132.648828 / 132.966142 | 142.146513 / 142.311205 | 156.584108 / 156.681469 | 173.073260 / 173.134901 |

C1536 cap168，chunk32→4 的墙钟投影减少 23.201%；chunk4 比 natural 的 chunk32 少 9.520%。C384 没有同类长片退化，C768 的 cap168 在 chunk16/32 退化。固定子乘积控制消除了上一轮不同 prime 区间的混杂因素，但不能外推完整生产收益。

GPU1 500ms 整卡采样共 759 行；利用率≥90%的 561 行中，SM 时钟范围 1800～1800 MHz、中位 1800 MHz，温度 50～60°C。没有锁时钟或统计置信区间。采样进程由工具主动终止，Windows exit1 是终止结果，CSV 有有效记录，不表示 GPU 任务失败。

原始证据：本地 `docs/data/stage1_slicing_matrix_20261006/summary.json`、各样本 `run.log` 与 `gpu.csv`。

整卡采样中有 4 行含不可读字段，保留为缺失，不填零、不用于上述数值统计。

## 5. Nsight Compute 与代码体积

三份管理员采集均成功：natural32 和 cap16832 为19 passes，cap1688 为17 passes；同一窗口二进制 SHA81079、GPU1/grid192/TPB128，clock-control/cache-control 均 none。前两份捕获同一32-record算术及同一初始点；第三份仅捕获同窗口的前8条，单 kernel 工作量不同，不能当作严格等工作量的计数对照。

| 指标 | natural / 32 | cap168 / 32 | cap168 / 8 |
| --- | ---: | ---: | ---: |
| 实际分配寄存器/线程 | 176.000000 | 168.000000 | 168.000000 |
| register-limited blocks/SM | 2.000000 | 3.000000 | 3.000000 |
| eligible warps/scheduler | 0.357301 | 0.355395 | 0.477470 |
| issue active % | 30.660073 | 26.810590 | 33.762671 |
| no_instruction / issue_active | 0.059075 | 2.534028 | 0.217220 |
| wait / issue_active | 3.889536 | 3.992643 | 4.276537 |
| math_pipe_throttle / issue_active | 0.319734 | 0.609766 | 0.844472 |
| DRAM throughput % | 0.000126 | 0.000058 | 0.000452 |

长片 cap168 的 no_instruction 比值上升、issue active 下降，结合固定子乘积计时，支持把指令供给/调度作为下一优先方向。该计数包含获取指令等待，不能全部定性为指令缓存 miss。wait 仍显著，DRAM 吞吐极低；不能把这些比值相加为墙钟占比，也不能使用重放的时间当作基准。

本地证据：`stage1_tail_ncu_255_chunk32_20261006`、`stage1_tail_ncu_168_chunk32_20261006`、`stage1_tail_ncu_168_chunk8_20261006` 中 `command.json`、`trace.ncu-rep`、`metrics.csv` 和 `quantitative.json`。

### 5.1 静态 SASS

对保留的 4608/TPI16 PRAC GPU 对象使用本机 CUDA13.3 cuobjdump，逐个已发现的 mangled function 导出。静态指令/文本跨度如下：

| 模式 | 指令条数 | 文本跨度 bytes |
| --- | ---: | ---: |
| baseline natural MODE4 | 56352 | 901632 |
| baseline cap168 MODE5 | 55928 | 894848 |
| compact natural MODE6 | 56344 | 901504 |
| compact cap168 MODE7 | 56000 | 896000 |

单个函数约 0.85～0.86 MiB；导出中仅一个 Function 段，包含内部分支/子程序。这里不是动态执行条数、热指令工作集或指令缓存容量。不能用总文本字节直接计算命中率，也不能由静态数量给出周期数。

Nsight 本机说明 [CPIStall.py](<C:/Program Files/NVIDIA Corporation/Nsight Compute 2026.2.1/sections/CPIStall.py:148>) 将 no_instruction 描述为指令获取等待或指令缓存 miss，也指出大汇编块间跳转可触发这种状态。结合静态体积，这使指令供给成为值得验证的方向，尚不能仅凭体积定因。

静态证据本地保存在 `docs/data/stage1_sass_20261006/`；对象 SHA 为 `2170cc6d91a36a46cf5d1c559d8a54b00051b4e81468d00809ef5843a5cd403b`。性能矩阵没有同时运行这些导出工具。

## 6. 复现

```powershell
python tools/test/test_cuda_prac_windows.py --production --production-counts 32 `
  --production-chunks 7 8 16 32 --device 1 --output docs/data/my_slicing_gate
python tools/bench/bench_stage1_prac_slicing.py --b1 260000000 `
  --curves 384 768 1536 --chunks 4 8 16 32 --count 32 `
  --seconds 6 --warmup 2 --repeats 2 --device 1 `
  --exp-cache build_cuda_cmake/prac --output docs/data/my_slicing_matrix
python tools/bench/profile_cuda_stage1.py --exe build_cuda_cmake/prac/ecm_cuda.exe `
  --exp-cache build_cuda_cmake/prac --curves 1536 --tpi 16 --registers 168 `
  --b1 260000000 --window tail --window-count 32 --window-chunk 32 `
  --seconds 6 --launch-skip 1 --output docs/data/my_tail_ncu --prepare-only
# 以管理员运行生成的 admin.ps1 后，再 --collect-only 导出。
```

## 7. 生产切片目标参数

加入 `ECM_PRAC_TARGET_MS`，有限数值范围10..500，未设为100。普通PRAC按每片CUDA event时间调整记录数：低于0.8×target增大约10%，高于1.2×target缩小约10%，至少一条记录；初始16条不变。它是反馈目标，不保证每次片长恰好达到目标，单条记录和批次调度限制最小可实现时长。窗口路径使用自己的固定chunk，resident仍采用100ms。

这将原有80/120ms门限参数化。没有新增设备分配或拷贝类型；短片会增加kernel边界坐标访问、事件及同步次数。记录数边界仍完整处理prime幂，没有在PRAC链中断点；checkpoint仍保存下一记录索引，不需要改变格式。日志新增已完成片长和event时长，避免只凭目标值推断实际切片。

实现：[参数校验](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_host.cuh:173)、[反馈控制](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_host.cuh:237)、[采样参数及日志确认](D:/code/MPA-OpenCl/tools/bench/bench_cuda_prac.py:35)、[同二进制反序对照](D:/code/MPA-OpenCl/tools/bench/bench_stage1_prac_variants.py:54)。采样器要求日志确认选定目标，旧binary不能静默忽略新参数后仍被当作有效A/B。

目标参数版只重编译host TU，11.8s，5个PRAC GPU对象SHA不变。exe SHA256：

```text
54dae87bd6ef1569c7f48a39494266120387add0a5a14c0fb7c91532d42a509b
```

默认100ms、三宽度门禁376条完整Q/save比较通过；N4423/cap168/target10ms/C384另有3544条完整比较通过，含lcm/choose12、64位sigma、正常checkpoint恢复、损坏checkpoint/plan重建和退化拒绝。10ms门禁的首片16条为16.414ms，最终片4条5.964ms，实际经过不同记录分片。每组另有5个非法目标（0/nan/501/10x/空串）拒绝，无最终save和checkpoint。Q计数包含ladder/resident/PRAC对CPU/GMP的比较，不全是PRAC运行次数。代码见 [完整门禁](D:/code/MPA-OpenCl/tools/test/test_cuda_prac.py:125)。

最终补上50/500ms合法目标的独立完整Q，以及同一checkpoint由100ms恢复为10ms；重跑两组后分别400/3568条比较全部通过（包含前述重复case，不再与首轮相加）。汇总为本地 `stage1_target_default_final_gate_20261006/summary.json` 和 `stage1_target10_cap168_final_gate_20261006/summary.json`。边界及跨目标恢复见 [target_edges](D:/code/MPA-OpenCl/tools/test/test_cuda_prac.py:87)。

同一目标参数版binary、N4423/C1536/TPI16/grid192，B1=10m/260m；resident、natural目标50/100、cap168目标50/100，各2次反序，共20个普通生产前缀样本。每次运行15s、排除前5s，再取最后5s内报告值的中位数。指数/PRAC缓存全部命中，无最终save，均以sample-limit保存checkpoint后exit1；没有强制终止。以下是两次样本中位数，s/curve。

| B1 | resident | natural 50ms | natural 100ms | cap168 50ms | cap168 100ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10000000 | 5.364109 | 5.616383 | 5.611412 | 5.060526 | 5.237004 |
| 260000000 | 139.496683 | 146.981886 | 146.815491 | 132.427626 | 138.744016 |

报告的投影仍使用既有50片速率环：speed=mean(work_i/event_ms_i)，s/curve=F/(1000*C*speed)。短片改变该环覆盖的时间长度，取稳定尾部可减轻启动影响，但它不是完整曲线的累计墙钟。两档B1中，两次同方向配对均改善；每配置只有两次，没有统计置信区间。

B1=10000000：cap168 的100→50ms投影耗时减少 3.370%，吞吐提高 3.487%；相对resident减少 5.660%。

B1=260000000：cap168 的100→50ms投影耗时减少 4.553%，吞吐提高 4.770%；相对resident减少 5.068%。

| B1 / 策略 / 目标 | 末5s打印片长范围 | event时长中位 ms |
| --- | ---: | ---: |
| 10000000 / natural / 50 | 6～6 | 47.581 |
| 10000000 / natural / 100 | 12～12 | 94.764 |
| 10000000 / cap168 / 50 | 7～7 | 51.287 |
| 10000000 / cap168 / 100 | 13～13 | 99.320 |
| 260000000 / natural / 50 | 3～4 | 45.613 |
| 260000000 / natural / 100 | 7～8 | 108.413 |
| 260000000 / cap168 / 50 | 4～4 | 54.778 |
| 260000000 / cap168 / 100 | 7～7 | 99.909 |

整卡500ms采样643行，3行有不可读字段并保留缺失。利用率≥90%的584行中，SM时钟1800～1800MHz，温度56～65°C。没有与GPU计时并行编译或NCU重放。

采用决定：保留50ms为N4423/cap168的大批量显式候选，默认仍100ms。固定tail短片和普通prefix均支持其可行性，但还缺真实中/后段生产点及其他批量的普通路径证据；不将此投影发布为Auto B2的T1。原始证据为本地 `docs/data/stage1_target_matrix_20261006/`。

### 7.1 使用

```powershell
$env:ECM_PRAC_TARGET_MS = '50'
# 同时指定已有的PRAC与寄存器策略；默认100ms可显式恢复。
python tools/bench/bench_stage1_prac_variants.py --configs resident natural cap168 `
  --target-ms 50 100 --b1 10000000 260000000 --curves 1536 `
  --seconds 15 --warmup 5 --repeats 2 --device 1 `
  --exp-cache build_cuda_cmake/prac --output docs/data/my_target_matrix
```

## 8. 后续内核方向

优先检验减少共同Montgomery/点运算代码展开，比较显式调用或有限展开的代码体积、寄存器、spill、尾部计数及真实吞吐。当前 [CGBN mont_mul](D:/code/MPA-OpenCl/cgbn/include/cgbn/core/core_mont_wmad.cu:29) 内部多处完全展开，[mont_sqr](D:/code/MPA-OpenCl/cgbn/include/cgbn/impl_cuda.cu:1032) 调用同一乘法核心。调用边界可能引入栈/参数开销，因此需保留同二进制候选和完整Q门禁，不能由体积缩小直接宣称加速。

关闭错误监控不能作为当前主要解法：[mont_mul入口](D:/code/MPA-OpenCl/cgbn/include/cgbn/impl_cuda.cu:1027) 本身直接调用核心，并未检查context错误；必须先证明某检查确实在热路径中。后续继续前/中/后固定范围，以及C384/768/1536和生产B1两端的验证；仅前缀改善不足以将策略全局设为默认。
