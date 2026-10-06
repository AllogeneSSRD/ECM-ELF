# Stage2 上下文复用实验：撤回执行路径

日期：2026-10-06。延续[等待/并发实验](D:/code/MPA-OpenCl/docs/STAGE2_AUTO_B2_WAIT_NTT.md)。本实验未证明吞吐收益，源码中的 `--reuse-context` / INI 执行路径及统计重置已撤回；生产程序和已发布 Auto B2 profile 保持原状。下面的选项仅存在于冻结实验二进制。

## 1. 实验实现

driver 串行调用原 `curve_worker`，只接收相同 N/B1、PARAM=0 的选中记录；重新读取 save 的 offset/hash、检查归一化 X 和 GCD。逐条成功写 result，整个任务成功才推进 queue。CRT / Win32 输出重定向在调用结束后恢复；结果附 PID、execution_mode、context_reused 和 context_curve_index。saved-X 因子提前返回不虚报 GPU 复用。零 B2 自动规划被拒绝，因为旧 cold 成本不适用于同批暖调用。

每条曲线仍创建和销毁 PolyLayer、S4Ctx、S4Reduce 与 NttArena；进程级 pinned 缓冲、事件和设备属性缓存保留。没有跨 sigma 共享 F 树，也没有并发调用非重入全局状态。

首版发现 Gdevice、carry 和 fuse 统计累积。第二版入口先检查 oracle owner 为空、fuse live_bytes=0，再仅清除统计字段；缓冲指针、容量和事件状态保留。日志中的 fuse live_bytes 是销毁前记录，不是调用结束时资源余额。

## 2. 构建与正确性

- 首版 `d26a6e97ef3d23033dab9ad47ff2acbf32e329535bd15bac9e3e143e2a0fa44f`：HostOnly；三条 sigma26→27→26 的实际 G1 曲线一致，两次 sigma26 的叶 hash 均为 1991729038653325525。
- 第二版 `3edcf1d5f83f8e33ea23a1f6e0e14e13a7a482a08a3e876368784642f760e674`：CUDA 504.4 秒、host/link 成功，25 个原始依赖冻结。
- 两个版本分别完成 9 次 CLI/INI/queue 协议调用、35 项检查。协议 fixture 从 saved X 直接发现因子，真实 GPU 曲线数为 0，不能代替 GPU 算术门禁。
- 第二版 ABBA 的 12 条真实曲线（sigma26..28）全部检查、叶/因子一致；每条 Gdevice pairs=2879、groups=14、copies=3，fuse allocations=48，没有跨曲线累积。包含合法的退化 baby 点和因子 338193759479。

首个六曲线采集因历史 parser 拒绝退化 baby 点而失败，但全部引擎调用成功。失败工具、日志和结果保留，不计为完整吞吐样本。修改后的 collector 使用 raw wall/split，不据此拟合 Auto B2。

## 3. 实际吞吐与内存

全部对照使用同一二进制、GPU1 RTX4060 Laptop、M8191/B1=1000/D30030/B2=86,426,340、arena512MiB、owner0、chain_min8192，顺序为 isolated/reuse/reuse/isolated。

首版 4 批、每批 6 曲线：独立进程 wall 为 19.015715/22.425193 秒；复用为 22.519055/20.512933 秒。吞吐中位数从 0.291542 降至 0.279470 curve/s（下降 4.141%）。全部 24 条实际曲线输出一致。

第二版 4 批、每批 3 曲线：独立进程 wall 为 8.719964/8.886700 秒；复用为 9.391462/10.682055 秒。吞吐中位数从 0.340811 降至 0.300142 curve/s（下降 11.933%）。整卡观察 GPU 峰均为 1116.617 MiB；自身 private commit 峰约 1319..1352 MiB。200ms 采样会漏过瞬时峰，以上不是进程完整显存或唯一物理 RAM 保证。

复用仍出现额外等待。样本只覆盖一个位宽及 G1，不能证明所有输入必然更慢；但它们不支持发布加速或将旧 cold 成本改成暖成本。故撤回该实现，保留实验快照，继续推进 Auto B2 的分 D 校准。

## 4. 保存与复现

[完整原始证据](D:/code/MPA-OpenCl/docs/data/ecm_stage2_context_20261006_evidence.json)包含 save、原始日志、失败采集、source manifest、冻结源码、collector 和协议脚本。二进制位于 `build_cuda_cmake/_stage2_reuse_context_20261006/native` 与 `native_reset`；这些是实验构建，未替换生产 exe。

使用证据中冻结的 collector 与实验 exe 可重新采集；不要将实验选项交给当前生产源码。完整 Auto B2 仍须通过全范围精度、排名、实际 auto/manual/INI/queue 验收，并补齐生产 B1、choose12、泛型 N、G2 和总内存准入。
