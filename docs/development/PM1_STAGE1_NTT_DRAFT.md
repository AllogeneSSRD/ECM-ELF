# P−1 Stage 1 开发草稿

本稿记录 0.75.0 首版实现范围与验收条件。参考版本为本地 GpuOwl/PRPLL `6dd6dd5` 和 Prime95 `31.06b01`。[用户操作说明](../usage/STAGE1.md#pm1-stage1)

## 任务和上界

```text
Pminus1=[AID,]1,2,p,-1,B1,B2[,how_far_factored][,B2_start][,"factors"]
```

首版执行显式给界的梅森 P−1 任务。独立解析任务类型，共用现有 worker 队列选择。读写时保留 AID、试除深度、区间起点和带引号的因子列表。含逗号的因子列表按一个字段读取。Stage 2 由外部客户端消费同一语法的任务行。参考 [Prime95 解析器](../../.refactor/p95v3106b01.source/commonc.c#L3001) 和 [GpuOwl 解析器](../../.refactor/gpuowl/src/Worktodo.cpp#L94)。

- 输入 B2 非零时，交接沿用原值。输入 B2 为零、比例启用时，以当前 worker 的 `[stage2].b2_ratio` 乘 B1，校验乘积范围，输出明确的十进制 B2。例如 B1=500、比例为 2 时，交接行是 `Pminus1=1,2,400381,-1,500,1000`。
- 输入 B2 和比例同时为零时，交接行直接写 `B2=0`，由 Prime95 按自身配置选界。本地 GpuOwl 对零的解释是只跑 Stage 1。[Prime95 选界](../../.refactor/p95v3106b01.source/ecm.cpp#L11746) · [GpuOwl 任务解析](../../.refactor/gpuowl/src/Worktodo.cpp#L112)
- 非零 B2 的保留规则同样适用于 B2≤B1。发布交接行时沿用该值，记录本机只完成 Stage 1。接收方如何处理空区间取决于其配置。
- Stage 1 的 TOML 读取模式会推迟解析 `b2_ratio`。P−1 在任务 B2 为零时读取生效的全局或 worker 覆盖值。[配置定义](../../config/ecm_options.json#L213) · [读取分支](../../src/io/config_format.h#L617)

## 计算与断点

对支持范围内的素数指数 p，令 M=2^p−1，E=2p·lcm(1,…,B1)。共享以 B1 为键、乘数为 1 的 [LCM 缓存](../../src/io/exponent_cache.h#L43)，再乘 2p。GPU 从 x=3、已消费位数 k=1 开始，自高位向低位扫描剩余指数位 b，每次计算 x←x²·3ᵇ mod M。[参考指数](../../.refactor/gpuowl/src/Pm1.cpp#L170) · [参考循环](../../.refactor/gpuowl/src/Gpu.cpp#L4694)

从现有 CUDA [独立模平方路径](../../src/ecm/stage1/cuda/ntt/ladder.cu) 提取设备常驻的模平方和模乘，使用现有加权选长与系数界。先在规范余数上实现乘 3，以 GMP 对照；若把乘 3 融入规约，需证明中间量和进位界。radix-5 混合长度上限 163,840 点，靠近 p=5,242,880 的任务使用已有 262,144 点加权路径。测试采用上限以内的素数指数。[长度常量](../../include/ecm/stage1/ntt_length.h#L8) · [选长表](../usage/TUNE.md#ntt-and-transform-sizes)

按 GpuOwl 的 Gerbicz 分块恒等式验证余数，保存通过验证的块末状态。末尾用 CPU GMP 求 gcd(x−1,T)。完整任务 T=M，有已知因子时先验证并剥离相应因子得到 T。分别记录 gcd=1、真因子和 gcd=T。每个完成并通过校验的 Stage 1 都写 Prime95 存档，保存模 M 的规范余数 x，包括命中真因子和 gcd=T 的情况。

## 外部接手

- GpuOwl 双向中途恢复。读取 `OWL P1 1/2`，写入 `p1-<p>/` 下的原生 v2 文件。恢复时核对指数、B1、指数位长、k、余数字数及 CRC。k 等于指数位长时可直接使用最终余数。[格式读写](../../.refactor/gpuowl/src/Saver.cpp#L275)
- Prime95 接手已完成的 Stage 1。按原生文件名和 v8 格式写 `PM1_STATE_DONE`、`B_done=C_done=B1`、`have_x=1`、完整余数及校验和。给定 B2>B1，Prime95 能利用该余数进入后续工作。[写档依据](../../.refactor/gpuowl/src/Pm1.cpp#L468) · [恢复分支](../../.refactor/p95v3106b01.source/ecm.cpp#L11958)
- **worktodo 发布条件。** Stage 1 未发现真因子时发布。命中真因子时，默认只写存档并记录因子；若产生该存档的 worker 设置 `[stage2].stage2_after_factored=1`，同时发布 worktodo。沿用已有的 [worker 配置读取方式](../../src/main.cpp#L889)。
- **交接任务身份。** 输出行原样保留输入任务的已知因子字段，包括原本省略该字段的情况。本轮找到的因子仅记入本机结果，不追加到任务行。这样接收方可以再次检出该因子。Prime95 默认可能在进入 Stage 2 前的模逆检查中检出；若接收方要求执行 Stage 2 后再做 GCD，Prime95 的 `Stage1GCD=-1` 会跳过该提前检查，并以 x−1 初始化 Stage 2 累积量。[Prime95 接手检查](../../.refactor/p95v3106b01.source/ecm.cpp#L12294)
- 先发布 Prime95 存档，再按上述条件发布独立的 Stage 2 worktodo，最后确认源任务完成。断电重试须能识别同一次交接，防止丢任务或重复投递。源、目标行均为 `Pminus1=`，交接文件需与本机输入队列区分。带有效 `B2_start>B1` 时保留请求区间，目标选支持该语义的 Prime95；本地 GpuOwl 拒绝这种行，且对已知因子列表只做语法接收。

## 验收

1. 小样本 M37 的独立整数模幂参考值为 B1=2 时 x=24273464184、gcd=1，B1=3 时 x=77413221271、gcd=223。再对含已知因子 223 的任务核对余因子 GCD 和完整模数余数。
2. 覆盖加权 163,840 点及 262,144 点，两种指数位、设备进位回卷、Gerbicz 余块、中断、损坏断点回退。与 GMP 比较完整余数。
3. 在两个方向各做一次 ECM-ELF 与 GpuOwl 中途接续，与不中断运行比较完整终点。覆盖 v1 读取、v2 写入。
4. 对 M400381，B1=1000 应在第一阶段找到 671839319。取 B1=500、B2=1000，验证第一阶段无因子，Prime95 从生成的存档进入第二阶段并找到同一因子。小界测试在 Prime95 启用 `AllowLowB1=1`。[上游验收样本](../../.refactor/gpuowl/test/pm1/mprime.sh)
5. 交叉覆盖命中与未命中、`stage2_after_factored` 为 0 和 1 的四种组合。四种都让 Prime95 读取存档并核对完整余数；仅“命中且开关为 0”没有 worktodo。检查原有因子列表逐项保留、原本无列表的行仍无列表。对 M400381、B1=1000、B2=2000 强制交接时，核对输出行未追加 671839319，在接收端启用 `Stage1GCD=-1` 并核验 Stage 2 末尾 GCD 仍含此因子。
6. 覆盖显式 B2、由比例派生、两者皆零、worker 比例覆盖、溢出、带 AID 和全部尾字段的任务。双零用例核对输出行的 B2 确实为 0，并由 Prime95 选择实际界值。模拟发布存档后、确认源行前中断，核对交接行、任务身份和结果覆盖范围。

## 实现与实测

- 0.76.7 完成时在任务目录归档 Prime95 端点，配置外部交接目录时两端写入同样的字节。完成档写入后清理当前 B1 的计算断点。扩展完成旧 B1 时先归档旧端点并保存增量起点，再清理已经用完的旧断点。队列回归覆盖独立目录、阻塞交接后的续发和命中。原生 GUI 回归核对秒/曲线、ETA 和完成/命中计数，M37、B1=3 的完成值为 `1 / 1`。[使用说明](../usage/STAGE1.md#pm1-stage1) · [队列回归](../../tools/test/test_pm1_stage1_queue.py) · [GUI 回归](../../tools/test/gui_v2/test_pm1_progress.py)

- 0.76.6 将“完成旧 B1”的起始工作量写入 `extension.progress`，在两个阶段内续算时都恢复同一进度总量，ETA 继续使用每路实际剩余迭代。旧版 `extension.pm1` 直接按已保存的二进制游标恢复。生产四路存档的副本从 51,183,616 / 115,428,096 iter（44.34%）开始。端到端回归在两个阶段分别停止并续算，与独立模幂逐字节对拍。扩展的四位幂表通过 768、65536、163840、262144 点的 GMP 对拍。[续算日志](../../data/experiments/pm1_extend_0766/production_checkpoint_copy/worker.log) · [进度与 ETA](../usage/GUI_V2.md#pm1-progress) · [回归](../../tools/test/test_pm1_stage1_extend.py)

- 0.76.5 支持一次 CUDA Graph 提交推进最多四个指数位。参考 gpuowl 在逆变换后的进位中合并 `MUL3` 的做法，将底数 3 的四位常数乘法并入最后一次平方的进位。GMP 对拍覆盖 Graph 和直接提交、最高 limb 分别为 1 位和 31 位的模数，以及 163840 和 262144 点。B1 扩展的四位 Graph 结果与独立 Python 模幂一致。Gerbicz 回归通过 30 个已验证端点和两次故障回滚。[gpuowl 参考](../../.refactor/gpuowl/src/cl/carryinc.cl) · [四路吞吐与 Nsight](../performance/STAGE1_MERSENNE_NTT.md#rtx-4070-ti-四路-p1-stage1)

- 0.76.1 的 worker 详情按首任务、首任务迭代与实际 NTT 形状、整批迭代进度显示。扩展时从启动存档位置统计需执行的迭代，并在进入增量阶段时重置当前速度样本。日志摘要使用相同的计数和 ETA。[显示口径](../usage/GUI_V2.md#pm1-progress) · [原生界面回归](../../tools/test/gui_v2/test_pm1_progress.py)

- 0.76.0 支持提高 worktodo 中的 B1。增量指数按 Prime95 31.6 的 `ecm.cpp::calc_exp2` 和 PrMers 的 `RunPM1.cpp::buildE_incremental_fast` 取两界 LCM 之商，Gerbicz 校验使用旧端点作底数。原始指数与增量指数分别使用 GpuOwl 和本机扩展断点。恢复顺序、可读取的 Prime95 状态及运行中编辑约定见 [Stage1 使用说明](../usage/STAGE1.md#pm1-stage1)。[端到端回归](../../tools/test/test_pm1_stage1_extend.py)
- 使用 Prime95 接收端留下的 M400381 DONE 存档（B_done=500、C_done=89850）扩展至 B1=1000。K 长 715 位，GPU 推进 714 次迭代，GCD 得 671839319。原生完成档与同版本从头计算的 50,128 字节逐字节一致，SHA-256 为 `c41604c4f5d64296aa62a7067da072b7fbc2f085d1a99c8a582499888977bf63`。[扩展记录](../../data/experiments/pm1_extend_0760/prime95_native/worker.log) · [从头计算记录](../../data/experiments/pm1_extend_0760/queue/p95_forced/worker.log)

- 0.75.5 的 worker 按批汇总已校验指数位，区分续算位置和本次新增迭代。混合断点 GPU 回归核对总迭代数和逐条队列计数，原生 GUI 用 M400381 核对实际交接 B2、运行中 ETA 和文件日志节流。[显示口径](../usage/GUI_V2.md) · [双任务回归](../../tools/test/test_pm1_stage1_tune.py) · [原生界面回归](../../tools/test/gui_v2/test_pm1_progress.py)

- 0.75.4 按长度在最小参测批量上预选 Graph，再扫描批量和 NTT 形状。同长度报告分别列出最快配置和达到目标吞吐的最小批量。GPU 回归覆盖缓存命中后实际复测最佳配置、固定 Graph、多长度报告及旧格式 19 重复行合并。163840 和 262144 点、批量 1–2 的同参数连跑中，第二轮每个长度各实测一次，矩阵两轮均为 20 行。[调优说明](../usage/TUNE.md#pm1-stage1-tune) · [冷启动记录](../../data/experiments/pm1_tune_0754/large_final/cold.log) · [复测记录](../../data/experiments/pm1_tune_0754/large_final/hot.log)

- 0.75.2 的 `-tune pm1` 仅对推进指数位的生产内核计时。相同 NTT 长度的连续任务在独立 CUDA stream 中运行，各自保留指数、Gerbicz 校验和存档。GPU 回归核对了不同断点的任务及 2560 点双任务的完整余数；163840、262144 点双任务调优通过 GMP 校验。格式 19 的 P−1 与 ECM 矩阵经过读写往返核对。`ntt_max_batch_pm1` 覆盖上调、下调及按 worker 继承均已验证。[调优与生产选批](../usage/TUNE.md) · [GPU 回归](../../tools/test/test_pm1_stage1_tune.py) · [验证记录](../../data/experiments/pm1_tune_20261127)

- 0.75.1 增加 Pminus1 专用的 `handoff_mode_pm1`，默认 `prime95`，可按 worker 设为 `off` 仅写存档。队列回归覆盖通用模式关闭、专用模式关闭以及两个方向的 worker 覆盖。

- 0.75.0 的 CUDA worker 在正常队列执行 Pminus1。设备端在规范余数上以两次模加完成乘 3，按位执行 CUDA Graph 或直接提交。GMP 对照覆盖 p=37、400381、3500017、5242877；后两例分别使用 163840 和 262144 点。独立故障注入使两次校验失败并回滚，30 个已验证断点逐一与 GMP 相等。[CUDA 算术入口](../../src/ecm/stage1/cuda/ntt/ladder.cu) · [回滚验收](../../tools/test/pm1_gerbicz_test.cpp) · [证据目录](../../data/experiments/pm1_stage1_interop_20261127)
- Windows sm89 worker 和 GpuOwl `6dd6dd5` 在 RTX 4070 Ti 上双向续算 M400381、B1=500。ECM-ELF 从 GpuOwl 的 k=115/744 恢复，GpuOwl 从 ECM-ELF 的 k=691/744 恢复。两端与 GpuOwl 从头运行的 50,048 字节终点逐字节相同。上游 GMP oracle 独立核对 Prime95 完成档的字段、校验和及余数，res64 为 `8eb02f6cfcfe4958`，GCD=1。[双向恢复脚本](../../tools/test/test_pm1_gpuowl_resume.py) · [GMP 核对](../../data/experiments/pm1_stage1_interop_20261127/resume-wapicf73/upstream_oracle.log)
- 九个生产队列用例覆盖发布开关、已知因子、gcd=T、B2 推导和交接。Prime95 31.6 build 1 从阶段边界接手三个 M400381 任务，均在 Stage 2 GCD 得到 671839319。普通交接行给出 500/1000；强制交接给出 1000/2000 并使用 `Stage1GCD=-1`；双零交接给出 500/0，Prime95 启动日志显示 B2=TBD。这三个小界任务的接收端实际报告 B2=89850。任务行中的请求上界和接收端实际处理上界分别核验。[队列矩阵](../../tools/test/test_pm1_stage1_queue.py) · [接收端验收脚本](../../tools/test/test_pm1_prime95_handoff.ps1) · [接收端日志](../../data/experiments/pm1_stage1_interop_20261127/receiver_trace)
- 完成回执冻结任务行、目标路径和写档时的内容指纹。Windows NTFS 和 Linux 原生文件系统覆盖待交接时存档被替换、接收端已取走邮箱后恢复、重复源任务再次投递。原有 worker 分区回归 42 项通过。[宿主 I/O 验收](../../tools/test/stage1_host_io_contract.cpp)
