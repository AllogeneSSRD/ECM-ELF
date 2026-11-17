# GUI NTT Stage1 进度显示草稿

目标版本：0.64.5。

NTT Stage1 的标量列进度属于当前曲线批次。GUI 分别保存任务曲线总数、已完成曲线数、当前批次范围和标量游标，避免把当前批次的百分比当成任务百分比。

## 数据与计时

- `stage1_ntt_batch_resume` 恢复已完成批次。
- `stage1_ntt_batch` 提供当前批次起点和曲线数量；补齐尾批后，以实际范围扩展任务总数。
- `stage1_ntt_plan` 提供标量总长度和恢复游标。
- `stage1_ntt_progress` 提供标量游标；0.64.5 增加 `elapsed`，从 GPU 初始化完成后的首个安全轮询点计时。
- `stage1_ntt_timing` 用实际执行的列数或 bit 数和主循环墙钟时间结算样本，不重复计算途中进度。
- 新任务或进程重启清除上一次任务的执行信息与进度。

任务百分比为

\[
100\frac{C_{done}+C_{batch}\,j/J}{C_{total}}.
\]

`j` 为当前标量游标，`J` 为标量总长度。恢复后的速度样本只计算本次新增的 `j-j_resume`。s/curve 为各批次主循环累计秒数除以相应的等效曲线数；ETA 为未完成的等效曲线数乘以该值。初始化、字典建表和发布时间不参与这个主循环估计。

旧计算程序没有 `elapsed` 时，从首条收到的进度开始测量后续游标增量；该批次结束后使用其 `wall_ms` 结算。首个可用样本出现前显示 `--`。

## 显示

主进度条显示任务总百分比，下方子条显示 `curve 129–160 50.0%`。所有任务的摘要首行显示数、已知因子标记和 `curves = ...`；Stage1 的 Stage2 交接目标放在该行末尾。实际执行参数保留在下一行。

实现入口：`src/ecm/application/gui/log_parse.*`、`worker_proc.*`、`worker_controller.*`，以及 `gui_v2/ui/worker_presentation.h` 和 `gui_v2/app.cpp`。NTT 计时字段由 `src/ecm/stage1/montgomery/cuda/ntt/workflow.cpp` 输出。

## 验收

已完成以下验证，程序身份均为 0.64.5，证据保存在忽略目录：

- 日志解析 111 项检查通过，覆盖恢复游标、变批量、真实尾批补齐、bit/column 两种进度、旧输出以及后续任务重置。入口为 `tools/test/gui/log_parse_test.cpp`。
- GUI App 测试通过，包含真实子进程管道中的 NTT 恢复事件和省略 B2、曲线数等可选字段的任务摘要。证据：`data/experiments/gui_ntt_progress_app_final_0645.log`。
- 正式 DX11 入口的 `ntt-progress` 专项通过，核对总进度 14.4%、当前批次 129–160 的 50%、1.00 s/curve 和 856 秒 ETA，并保存宽窄窗口截图。证据：`data/experiments/gui_ntt_progress_suite_0645/`。
- 中英文完整展示测试通过，覆盖传统 Stage1、Stage2、日志页签、窗口缩放和最小化恢复。证据：`data/experiments/gui_ntt_progress_presentation_en_0645/`、`gui_ntt_progress_presentation_zh_serial_0645/`。前一轮中文测试在最小化断言停止，原失败截图和日志保留在 `gui_ntt_progress_presentation_final_0645/`；同一二进制的后续串行复测通过。
- GPU1 RTX 4060 Laptop 实测：`M16319/270811405193137`，Edwards，sigma 从 123456789 起连续 4 条，B1=B2=60000，NAF w6，加权卷积。主循环日志记录 0、2.006、4.006 秒的进度采样，最终 `wall_ms=5621.955`，4 条曲线正常完成。证据：`data/experiments/gui_ntt_progress_real_0645_long/`。本例用于验证计时字段和进度格式，不作为性能基准。

本地程序：`build_gui_v2_local/ecm_gui_v2.exe`、`build_stage1_local/ecm_cuda.exe`。使用说明已同步，开发草稿保留供用户决定是否清理。

最终使用说明维护于 [GUI_V2.md](../usage/GUI_V2.md)。本草稿完成后由用户决定是否清理。
