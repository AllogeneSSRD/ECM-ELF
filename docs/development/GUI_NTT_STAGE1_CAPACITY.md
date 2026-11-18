# Stage1 NTT 按计划估算批量容量

M15919 的生产任务使用原始指数 15919 作为算术位宽，Edwards NAF w12 和 4096 MB 字典预算。调度候选包括 2048 点线性卷积和 1024 点加权卷积。前者的字典估算为 75.27 MB／曲线，后者为 41.71 MB／曲线，因此字典预算分别允许 54 和 98 条。两个方案分别按自身长度计算容量，调度用最高可行容量，并在执行计划查表和补测时跳过容不下本批的候选。

用 `stage1_ntt_batch_selection` 查看两种卷积模式的容量，GUI 按 `Stage1 execution:` 显示当前计划的每曲线估算、配置和设备额度、NAF 预算和最大容量。容量估算包括每曲线工作区和字典，额外的固定工作区参与容量计算。设备执行仍会检查实际分配，系统可用内存变化时，尚未建立批内检查点的新批次可以缩小后重试。

对长度 \(L\)、算术位宽 \(p\) 和 NAF 窗口 \(w\)，令 \(W=\lceil p/32\rceil\)，字典项数 \(E=2^{w-2}\)（无 NAF 时为 0）。每曲线字典估算为 \(E(32L+16W)\) 字节，GPU 每曲线总估算再加 \(88L+128W\) 字节。容量取显存、字典、主机内存及 128 条软件上限中最小的一项。GUI 的设备额度是当前空闲显存的 80%，用户配置与设备额度取较小值供显存估算使用。

验证入口为 `tools/test/test_stage1_ntt_task_batches.py` 和 GUI v2 的 `ntt-progress` 专项。正式使用说明见 [Stage1](../usage/STAGE1.md#任务曲线数与设备批量) 和 [GUI v2](../usage/GUI_V2.md#导航与布局)。

0.66.2 在 RTX 4060 Laptop GPU1 上复测 M15919／1782929，Edwards w12，B1=B2=1024，95 条，sigma 从 20260922 起。设 6144 MiB 和 4096 MB 字典预算时，linear／weighted 上限为 54／98，首批 64 条运行 weighted。将显存上限收紧到 1024 MiB 后，上限为 14／25，首批 25 条运行 weighted。另通过 1000 条拆批、断点恢复、尾批补齐和因子发布回归。证据位于 `data/experiments/stage1_ntt_per_plan_capacity_gpu1_final_0662/`。中英文 GUI App 回归和 DX11 `ntt-progress` 专项通过，宽窄窗口截图位于 `data/experiments/stage1_ntt_per_plan_gui_progress_0662/ntt-progress/`。

开发草稿保留供后续核对，当前使用说明维护在上述两份正式文档。
