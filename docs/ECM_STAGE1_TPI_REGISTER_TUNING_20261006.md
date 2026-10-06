# CUDA param0 Stage1：TPI、寄存器与 Nsight 实测

日期：2026-10-06。承接 [PRAC 实施记录](D:/code/MPA-OpenCl/docs/ECM_STAGE1_PRAC_IMPLEMENTATION_20261006.md:1)。本轮使用 GPU1：RTX 4060 Laptop，24 SM，sm89。GPU0 保持其已有生产任务。

## 1. 构建与术语

本轮独立实验二进制为 `build_cuda_cmake/prac/ecm_cuda.exe`，SHA256：

```text
04dcbd6c04853f3e8344133e4c7e392d8f25d7decd1381717f7b065ed1ca1d99
```

编译档位为 512、1024、2560、4608、9216；param2 关闭；Montgomery 构建；**TPB=128**。并行编译 6 个 TU，最长 TU 为 native TPI16，372.2 s，随后完成主机编译与链接。实验档位不能替代覆盖全部位宽的生产安装版。

仓库默认分档依然是容器 2560～8192 位使用 TPI16、9216～16384 位使用 TPI32，见 [原分档](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_kernel.h:1178)。N=2^4423−1 默认使用 **4608/TPI16**。选容器须覆盖 N bits+6：N=8191 bit 会选择 9216/TPI32。此边界不能仅按 N>8192 理解。本轮另行实例化 2560/TPI32、4608/TPI32；只有显式设置 `ECM_STAGE1_TPI=32` 的新 resident/PRAC 路径才使用它们。

TPB 的仓库默认已于 2026-09-25 从 256 改为 128，见 [配置背景](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_kernel.h:100) 和 [CMake 默认值](D:/code/MPA-OpenCl/CMakeLists.txt:433)。调节寄存器预算不会改变 TPB。

设 C 为曲线数、T 为 TPB、I 为 TPI、S 为设备 SM 数：

```text
每 block 曲线数 = T / I
提交 grid blocks = ceil(C / (T/I))
平均提交 blocks/SM = grid blocks / S
寄存器允许驻留 blocks/SM ≈ floor(65536 / (T × 实际分配寄存器/线程))
```

最后一个公式须考虑分配粒度、warp/thread/block 上限及 shared memory；程序报告值由 CUDA occupancy API 给出，[调用位置](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_host.cuh:160)。**提交的 grid 不等于同时驻留的 block**。

T=128 时，TPI16 的 C=384/768/1536 与 TPI32 的 C=192/384/768 分别提交相同的 48/96/192 blocks，即平均 2/4/8 blocks/SM。比较吞吐量时使用曲线/秒，不能只比较较小批次的完成时间。

对 4608/TPI16 PRAC，自然分配 172 个寄存器，NCU 报告实际按 176 分配：`128×176×3=67584>65536`，最多驻留 2 blocks/SM。168 策略有机会达到 3 个：`128×168×3=64512`。**若 TPB=256，176 与 168 两者都只能驻留 1 个 block/SM**；本轮 168 实验的占用率收益不能直接移植到 TPB256。

## 2. 实现入口与适用范围

- [实验 TPI 实例](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_alternate32.cu:9)：相同容器、相同公式，只改变协作线程数。
- [分派](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_kernels.cu:10)：请求 0 保留正常分档；请求 32 才尝试两个新增实例。
- [主机策略](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_host.cuh:55)：`ECM_PRAC_REG_TARGET=0|168|255`；`ECM_STAGE1_TPI=0|16|32`。先选正常的最小容器，再验证请求策略，缺失实例明确报错，不悄悄改成更大容器。
- [寄存器实例](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_kernel.cuh:63)：0 沿用 per-tier；255 允许 natural 分配；168 仅提供 **4608/TPI16 PRAC**，其他档位不支持。
- [检查点兼容性](D:/code/MPA-OpenCl/kernels/cuda/cgbn_stage1_prac_host.cuh:117)：记录实际 TPI，跨 TPI 恢复拒绝；寄存器预算不改变数学状态，可在相同 TPI 间恢复。

原 ladder 不接受这些实验 TPI 覆盖。原有默认算法保持 ladder；新路径显式选择 `ECM_GPU_STAGE1_ALGO=resident|prac`。

TPI32 不意味着物理工作位宽恰好等于容器宽。CGBN 的 [LIMBS、UNPADDED_BITS、PADDING](D:/code/MPA-OpenCl/cgbn/include/cgbn/cgbn_cuda.h:107) 会对不整除的 limb 数补齐：2560/TPI32 每线程 3 limbs，内部槽宽 3072；4608/TPI32 每线程 5 limbs，内部槽宽 5120。存档/数据 stride 仍使用 2560/4608 位容器。多出的内部槽及跨 lane 通信是可能的性能代价，不能仅凭每线程 limb 数推断加速。

## 3. 测量方法与缓存

[配对矩阵脚本](D:/code/MPA-OpenCl/tools/bench/bench_stage1_tpi_matrix.py:11) 默认两次重复；第二次反转 TPI 与算法顺序。每个独立目录采样 12 s、暖机 4 s；[采样脚本](D:/code/MPA-OpenCl/tools/bench/bench_cuda_prac.py:73) 取最后约 5 s 进度投影的中位数，再取两次运行中位数。

各次运行固定 param0、sigma=26、`exponent=lcm`、B1=10,000,000。显式指定共享 `--exp-cache`，复用已有指数缓存；PRAC 同时命中已有计划缓存。两档共 48 次矩阵运行，均不重复冷生成 lcm。缓存命中记录在 JSON，见 [解析](D:/code/MPA-OpenCl/tools/bench/bench_cuda_prac.py:97)。

输出 `s/curve` 是窗口内 GPU 工作速率对整个 Stage1 的投影；PRAC 用模乘等价 W 加权，resident 用标量位数加权。它不包含冷准备、CPU 曲线初始化、最终仿射化和最终 save。此次不等待生产 B1 整批结束，不把检查点或空 save 认作 Stage1 完成。

所有吞吐量采样独立于 profiler。NCU 会重放和注入额外开销，不能将采集时的进度速率当作普通生产速率。未锁时钟；结果只对本机/本构建/本次短时窗口成立，后部素数仍需单独验证。

## 4. 配对吞吐量

### 2203 位，B1=10m

| TPI | 曲线 C | grid | resident s/curve | resident 曲线/s | PRAC natural s/curve | PRAC 曲线/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 384 | 48 | 2.335011 | 0.428264 | 2.298102 | 0.435142 |
| 32 | 192 | 48 | 4.751330 | 0.210467 | 4.176774 | 0.239419 |
| 16 | 768 | 96 | 2.094874 | 0.477356 | 2.014654 | 0.496363 |
| 32 | 384 | 96 | 3.608585 | 0.277117 | 3.151444 | 0.317315 |
| 16 | 1536 | 192 | 1.978457 | 0.505445 | 1.972263 | 0.507032 |
| 32 | 768 | 192 | 3.412408 | 0.293048 | 3.013390 | 0.331852 |

### 4423 位，B1=10m

| TPI | 曲线 C | grid | resident s/curve | resident 曲线/s | PRAC natural s/curve | PRAC 曲线/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 384 | 48 | 5.634269 | 0.177485 | 5.623518 | 0.177825 |
| 32 | 192 | 48 | 9.279464 | 0.107765 | 7.896946 | 0.126631 |
| 16 | 768 | 96 | 5.513670 | 0.181367 | 5.618975 | 0.177968 |
| 32 | 384 | 96 | 8.037979 | 0.124409 | 7.116580 | 0.140517 |
| 16 | 1536 | 192 | 5.363717 | 0.186438 | 5.611616 | 0.178202 |
| 32 | 768 | 192 | 7.577967 | 0.131962 | 6.816781 | 0.146697 |

两档 TPI32 都未胜过 TPI16。2203 位的 TPI16 从 384 增到 1536 曲线，resident 投影由 2.335011 降到 1.978457 s/curve，吞吐提高约 18.0%；PRAC natural 从 2.298102 降到 1.972263，吞吐提高约 16.5%。4423 位 resident 的相同扩批收益约 5.0%，PRAC natural 接近不变。

这与资源形状一致：CUDA API 报告 4423 位 native resident 可驻留 4 blocks/SM，natural PRAC 仅 2；2203 位 native resident/PRAC natural 分别为 5/4，C=384 仅提交平均 2 blocks/SM。增加批量可以填充更多可驻留 block，但仍需检查实际调度和尾波，不能将收益完全归因于单一因素。

原始汇总：本地 `docs/data/stage1_tpi_matrix_2203_20261006/summary.json`、`stage1_tpi_matrix_4423_20261006/summary.json`。每个单元格两次重复。

### 4423 位：168 寄存器策略

先按用户指定批量比较，两次投影中位数如下；同 TPI16、TPB128，CUDA occupancy API 确认 168 策略可驻留 3 blocks/SM。

| C | PRAC natural s/curve | PRAC 168 s/curve |
| ---: | ---: | ---: |
| 384 | 5.623518 | 5.700672 |
| 768 | 5.618975 | 5.741026 |
| 1536 | 5.611616 | 5.221469 |

补测理论上可均匀填满 3 blocks/SM 的 C=576/1152，并重复 C=1536。此处记录另一次两轮矩阵，不能当作首次样本的第三/第四次重复。

| B1 | C | PRAC 168 s/curve | 曲线/s |
| ---: | ---: | ---: | ---: |
| 10000000 | 576 | 5.416052 | 0.184636 |
| 10000000 | 1152 | 5.327206 | 0.187716 |
| 10000000 | 1536 | 5.214393 | 0.191777 |
| 260000000 | 576 | 143.159706 | 0.006985 |
| 260000000 | 1152 | 141.435109 | 0.007070 |
| 260000000 | 1536 | 139.742783 | 0.007156 |

576/1152 没有超过 1536。理想 CTA 波数模型未预测出实际赢家，不能仅按“整波”计算推荐批量。168 的少量 spill 与实际调度的计数器对照见第 6 节；时钟未锁定。

### 最终紧邻 A/B：resident 与 PRAC 168

固定 TPI16、TPB128、C=1536，每次采样 15 s、暖机 5 s，两次重复，第二轮反转算法顺序。

| B1 | resident s/curve | PRAC 168 s/curve | 168 耗时变化 | 168 吞吐变化 |
| ---: | ---: | ---: | ---: | ---: |
| 10000000 | 5.363491 | 5.255907 | -2.006% | +2.047% |
| 260000000 | 139.480187 | 138.162343 | -0.945% | +0.954% |

本机前缀采样中，168 策略在两端 B1 上均胜过 resident，但幅度为约 2.0% / 0.95% 耗时减少。它仍是显式实验选项：该收益不代表完整 Stage1 墙钟，也没有足够证据将 168 或 PRAC 全局设为默认。

该对照每次指数缓存都命中；PRAC 计划均命中。相较最初 natural PRAC 的退化，168 解决了该测试形状的一部分资源压力。较小批量的 168 仍可能更慢，必须同时记录 C。原始汇总为本地 `docs/data/stage1_cap168_final_ab_20261006/summary.json`。

## 5. 编译资源记录

单位：寄存器/线程；stack、spill stores、spill loads 为 ptxas 静态 bytes。spill bytes 不是一次运行的流量总量。

| 容器/TPI | 算法/策略 | regs | stack | spill stores / loads |
| --- | --- | ---: | ---: | ---: |
| 2560/16 | resident | 84 | — | 0 / 0 |
| 2560/16 | PRAC natural | 105 | — | 0 / 0 |
| 2560/32 | resident | 69 | 64 | 0 / 0 |
| 2560/32 | PRAC natural | 81 | 64 | 0 / 0 |
| 4608/16 | resident | 128 | 56 | 0 / 0 |
| 4608/16 | PRAC per-tier | 128 | 264 | 636 / 1520 |
| 4608/16 | PRAC natural | 172 | 48 | 0 / 0 |
| 4608/16 | PRAC 168 | 168 | 80 | 36 / 28 |
| 4608/32 | resident | 90 | 64 | 0 / 0 |
| 4608/32 | PRAC natural | 109 | 64 | 0 / 0 |

本轮编译原始证据在本地 `build_cuda_cmake/prac/before_compact_20261006/par_nvcc/`，不提交构建日志。

## 6. Nsight Compute：管理员采集

普通权限采集出现 `ERR_NVGPUCTRPERM`。按用户指示以管理员权限启动后采集成功，无需改动驱动全局权限设置。

基线：4608/TPI16、PRAC natural、C=768、grid=96、TPB=128，GPU1。使用本轮前保留的同算法二进制 `pre_tpi/ecm_cuda.exe`，SHA256 `57ef2e2dd83f7bbd69cd883eb1973c48cd73e205352e0c7d221fe0e40cd16ebc`。跳过 2 次目标 kernel，采 1 次，共 16 passes。

| 指标 | 基线值 |
| --- | ---: |
| registers / allocated registers | 172 / 176 |
| register-limited blocks/SM | 2 |
| achieved occupancy | 16.337% |
| SM throughput | 72.229% |
| issue active / peak sustained active | 30.603% |
| active warps / SMSP | 1.960 |
| eligible warps / SMSP | 0.356 |
| DRAM throughput | 0.000295% |
| stalled wait / issue active | 3.897099 |
| stalled dispatch / issue active | 0.504185 |
| stalled math pipe / issue active | 0.317382 |
| stalled short scoreboard / issue active | 0.253204 |
| stalled long scoreboard / issue active | 0.000139 |

stall 记录的是 NCU 原始 `smsp__average_warps_issue_stalled_*_per_issue_active.ratio` 比值，不是运行时间占比，不能相加当作墙钟分解。该片段的 76.766 ms 是重放采样 kernel 时长，不用于曲线吞吐量表。

据此推断：稳定 PRAC 片内并非 DRAM 带宽瓶颈；eligible warp 少、wait 高，优先调查算术依赖延迟与可用独立 warp 数。此组计数器不能单独证明具体是哪条 MAC，也不代表全部素数链。

### 管理员 NCU 配置对照

三次报告都是 GPU1、grid96、TPB128、B1=10m、同一短前缀附近的 kernel。natural/168 的 C=768；TPI32 的 C=384，维持相同提交 grid。两项实验采集使用新二进制，仍各为 16 passes。

| 指标 | TPI16 natural | TPI16 168 | TPI32 natural |
| --- | ---: | ---: | ---: |
| 寄存器 / 实际分配 | 172 / 176.000000 | 168 / 168.000000 | 109 / 112.000000 |
| register-limited blocks/SM | 2.000000 | 3.000000 | 4.000000 |
| achieved occupancy % | 16.337167 | 19.369763 | 27.122807 |
| issue active % | 30.602728 | 30.929009 | 38.926562 |
| eligible warps / SMSP | 0.356295 | 0.423954 | 0.726930 |
| SM throughput % | 72.229457 | 72.097616 | 78.034392 |
| DRAM throughput % | 0.000295 | 0.000042 | 0.001356 |
| stalled wait / issue active | 3.897099 | 4.114919 | 3.413388 |
| stalled math_pipe_throttle / issue active | 0.317382 | 0.747937 | 1.429882 |
| stalled short_scoreboard / issue active | 0.253204 | 0.341452 | 0.414888 |
| stalled dispatch_stall / issue active | 0.504185 | 0.678541 | 1.134971 |

168 将可驻留 block 上限从 2 提到 3，eligible warp 从 0.356 提到 0.424，但该片段 issue active 仅从 30.60% 到 30.93%，math-pipe stall 比值从 0.317 到 0.748。增加驻留容量有实际作用，却没有换来相同比例的发射率提升；不能按 occupancy 比例预测曲线加速。

TPI32 的 occupancy、eligible warp、issue active 都更高，但普通 A/B 的曲线吞吐量仍更差。每条曲线消耗的协作线程、内部补齐以及指令/通信代价同时变化。硬件利用率不是 ECM 曲线/秒。

这三份报告仍不是完整生产曲线，且 natural 基线二进制与新增实例二进制不同；用于确认资源和瓶颈趋势。最终性能结论以同一新二进制的非 profiler A/B 为准。局部 memory 与 shuffle 都可能影响 short scoreboard，仅凭该计数器不能把等待全部归因于 spill。

本地证据：`stage1_tpi_ncu_admin_baseline_prac_skip2`、`stage1_tpi_ncu_admin_cap168`、`stage1_tpi_ncu_admin_tpi32` 各目录的 `trace.ncu-rep`、`metrics.csv`、`quantitative.json`。

## 7. Nsight Systems：Stage1 稳态空隙

同基线算法，采样 6 s；分析首个到最后一个 PRAC kernel 的区间，排除曲线准备、缓存读取、首次初始化和最终检查点。

- 65 个 PRAC kernel；区间 6.068090 s。
- 自身 kernel 区间并集 6.054063 s，覆盖 **99.769%**。
- 区间中无自身 PRAC kernel 的时间 0.014027 s，即 **0.231%**。
- 整份 trace H2D：2 次、13,729,840 bytes、DMA 1.195 ms；D2H：1 次、3,096,576 bytes、DMA 0.371 ms（退出采样的检查点）。
- `cudaEventSynchronize` 65 次共 6.055633 s，这是主机等待 GPU，不是 CPU 计算 6 秒。
- `cudaLaunchKernel` 66 次共 8.943 ms；`cudaEventRecord` 130 次共 5.117 ms。

数据核对：曲线缓冲 `7×768×4608/8=3,096,576 bytes`；B1=10m PRAC 计划 10,633,264 bytes，两者之和等于本轮 H2D 量。稳态没有逐片上传曲线状态或计划。

因此，这条 Stage1 稳态路径几乎连续执行 GPU kernel。不能把此前 Stage2 的 GPU 空闲/传输瓶颈直接套到这里；仍需单独测量整个流程的准备和保存阶段。

Windows 上 NSYS 直接接收管道 stdin 的首次尝试使应用等待输入、超时无报告。当前 [采集脚本](D:/code/MPA-OpenCl/tools/bench/profile_cuda_stage1.py:27) 使用 `app.cmd` 将输入文件直接重定向给 ECM，已成功生成 trace。失败试验没有计入本报告性能结果。

## 8. 正确性与后续

新二进制四个门禁全部通过，均在 GPU1：

| 门禁 | 总 Q/因子 save 比较 | 退化拒绝 | 原始汇总（本地 docs/data） |
| --- | ---: | ---: | --- |
| native TPI / PRAC per-tier | 376 | 3 | stage1_new_pertier_gate_20261006/summary.json |
| native TPI / PRAC natural | 376 | 3 | stage1_new_native_gate_20261006/summary.json |
| 强制 2203/4423 位 TPI32 | 376 | 3 | stage1_tpi32_gate_20261006/summary.json |
| 4423 位 / PRAC 168 | 160 | 3 | stage1_cap168_gate_20261006/summary.json |

共 1288 条比较和 12 个退化拒绝检查。比较包括 CPU/GMP、原 ladder、resident 和 PRAC；不是每条比较都运行实验 kernel。TPI32 只强制支持的 2203/4423 案例；168 只用于 4423 PRAC 案例，小位宽因子/退化案例继续使用 per-tier。

覆盖完整仿射 Q 和 save 元数据、lcm/choose12、53/62 位 sigma、冷计划、坏计划重建、检查点续跑和损坏回退。小 B1 检验数学正确性，生产 B1 只做部分运行吞吐量采样；本轮未重新编译 OpenCL 或完成全部位宽生产门禁。

门禁脚本支持 `--bits 4423 --tpi 16 --registers 168`，见 [选择与限制](D:/code/MPA-OpenCl/tools/test/test_cuda_prac.py:87)。

下一轮优先级：

1. 对 PRAC 前、中、后部素数窗口分别测量 W/秒，检验短前缀投影是否能代表整个生产 B1。窗口实验必须明确仅执行子乘积，不能输出完整 Stage1 save。
2. 从寄存器生命周期与独立乘法调度入手，压缩 4608/TPI16 PRAC 在 168 预算下的 spill，保持 normalized 算术及逐曲线 Q 门禁。
3. TPB256 若要评估，使用独立构建重新测量；不得复用 TPB128 的 blocks/SM 数字。
4. Auto B2 的 T1 标定应纳入算法、TPB、TPI、寄存器策略、容器和批量身份；不自动导入本轮前缀投影为最终生产成本。

## 9. 复现

```powershell
python tools/bench/bench_stage1_tpi_matrix.py --device 1 --bits 2203 4423 `
  --b1 10000000 --seconds 12 --warmup 4 --repeats 2 `
  --exp-cache build_cuda_cmake/prac --output docs/data/my_stage1_tpi_matrix

python tools/bench/bench_stage1_tpi_matrix.py --device 1 --bits 4423 `
  --tpis 16 --algorithms prac --prac-registers 168 `
  --seconds 12 --warmup 4 --repeats 2 --output docs/data/my_stage1_cap168

python tools/bench/profile_cuda_stage1.py --exe build_cuda_cmake/prac/ecm_cuda.exe `
  --exp-cache build_cuda_cmake/prac --bits 4423 --curves 768 --tpi 16 `
  --registers 168 --seconds 15 --launch-skip 2 `
  --output docs/data/my_stage1_ncu --prepare-only

# 从管理员 PowerShell 执行准备好的采集包装；只运行一个采集任务。
powershell -NoProfile -ExecutionPolicy Bypass -File docs/data/my_stage1_ncu/admin.ps1

# 采集完成后导出 CSV；此步骤不执行 GPU 工作。
python tools/bench/profile_cuda_stage1.py --exe build_cuda_cmake/prac/ecm_cuda.exe `
  --exp-cache build_cuda_cmake/prac --output docs/data/my_stage1_ncu --collect-only
python tools/bench/summarize_stage1_profile.py docs/data/my_stage1_ncu/metrics.csv `
  --output docs/data/my_stage1_ncu/quantitative.json
```

`--tool nsys` 可生成 Systems 报告；使用 NSYS `export --type sqlite` 后，由 [汇总脚本](D:/code/MPA-OpenCl/tools/bench/summarize_stage1_profile.py:50) 分析指定模板 MODE=2/3/4/5。kernel 时间用区间并集计算，避免重叠重复相加。原始报告、CSV、SQLite、检查点和采样 save 均留在已排除的 `docs/data/`，仅提交方法及汇总文档。

后续窗口校准和 compact DBL 候选记录：[PRAC 窗口与算术实验](D:/code/MPA-OpenCl/docs/ECM_STAGE1_PRAC_WINDOWS_COMPACT_20261006.md:1)。本文性能数字保留原二进制与采样范围，源码链接更新到当前对应入口。
