# Stage1 迭代速度与 worker 摘要

## NTT 文件日志与批次 ETA

NTT `bits` 和 `columns` 进度行共用 `progress_log_seconds` 的文件日志限制，控制台和 GUI 管道延续原来的更新节奏。批次完成按迭代游标判断，避免中间百分比四舍五入为 100.0% 时绕过限频。NTT 输出当前批次预计剩余秒数，worker 在小进度条左侧百分比之后显示该值。

RTX 4060 Laptop GPU 的 Montgomery 和 Edwards NTT 均通过三档配置测试。M4423、Edwards、B1=250000、4 条曲线的长任务中，5 秒设置记录 11 条文件进度，管道保留 32 条，30 条批次 ETA 与剩余列数除以实测速率一致。原始记录位于 `data/experiments/ntt_batch_eta_worker/` 和 `data/experiments/ntt_batch_eta_measured/`。

Worker 表格增加 iter/s/c 列；速度来自计算程序的显式 iter_per_s 字段。CUDA CGBN、Edwards 和 thread 路径复用现有 SpeedWindow。NTT 主循环按游标增量和轮询间隔采样，排除准备与检查点回调耗时，使用相同的时间窗口设置。字段缺失时不估造速度。

名称与设备/引擎合并为设备列，保留原选择日志的交互和阶段配色。Stage2 显式 CPU 配置优先于可执行文件名的 GPU 推断；缺少 worker 覆盖项时沿用 Stage2 配置。应用测试覆盖 CPU/GPU 覆盖、Stage2 默认回退及 Stage1 不引用 Stage2 设备。表格恢复为九列。

合并后的 GUI 正式构建、应用回归和中文暗色真实窗口测试通过，包含选择不同 worker 日志的交互。证据位于 `data/experiments/worker_device_merge_app/`、`worker_device_merge_presentation/`；中英文列单位统一为 iter/s/c。

NTT 执行详情补充 B1、iter_total，worker 紧凑摘要去掉 bpw 和内存行并合并 NTT 长度、形状及 NAF。日志的详细格式保持原有字段和内存摘要。中英 XML 增加列标题，表格继续使用弹性列。

验证证据保留在 data/experiments/stage1_iter_*。解析层新增显式速度、NTT 批量口径与跨批重置测试；应用层核对紧凑和完整摘要各自内容。

正式 Stage1 和 GUI 构建通过。解析层 116 项检查、GUI app 回归及中文暗色真实窗口展示测试通过。RTX 4060 Laptop GPU 的实际 CGBN PARAM0 M127/B1=4096/16 曲线和 Edwards NTT M4423/B1=1024/8 曲线均输出正的 iter_per_s，NTT 执行详情包含 B1 和非零 iter_total。原始报告在 `stage1_iter_worker_verify/`、`stage1_iter_gui_final_test/`、`stage1_iter_presentation/`；编译与解析日志位于同前缀文件。程序版本沿用当前工作区的 0.72.9。本轮不改计算算法、配置默认值或存档格式。
