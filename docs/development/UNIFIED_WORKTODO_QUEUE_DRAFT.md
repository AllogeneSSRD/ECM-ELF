# 共用 worktodo 队列开发草稿

## 需求与决定

2026-10-19：Stage1、Stage2 共用一个 `worktodo`。`ECMSTAGE1=` 由 Stage1 领取，`ECMSTAGE2=` 由 Stage2 领取；现有 Prime95 `ECM=`/`ECM2=` 仍由 Stage1 处理。GUI 根据当前 worker 分区的下一条可运行任务选择程序，不再使用 INI `stage`。同一 worker 不并行运行两个程序。

用户确认：`priority=1` 优先清空 Stage1，`priority=2` 优先清空 Stage2；优先阶段无任务时回退另一阶段。`skip=1` 完全忽略 Stage1，`skip=2` 完全忽略 Stage2，行保留原样。两个键使用字面名 `priority` / `skip`，可由 `[Worker #N]` 覆盖；`[GUI] priority` 仍专指进程优先级。

`auto_stage2=1` 时，Stage1 完成后照常写 Stage1 `finished`；未找到因子时将原任务行原位改写为 Stage2，找到因子时移除任务，避免 Stage2 反复尝试含因子的存档。`0` 或空值仍删除已完成行。`b2_ratio=0` 或空值保留原 B2 选择；1…1000000 的整数使使用该 INI 的 Stage2 有效 B2 等于 save 中 B1 × ratio，包括直接读档，仍要求 B2>B1，溢出报错。四个键均可按 worker 覆盖。

## 实现与接口

- 配置定义：`config/ecm_options.json`；生成器：`tools/gen/generate_ecm_config.py`。Stage2 `stage2_worktodo` 键及 GUI/worker `stage` 键移除；生成物由生成器同步。
- INI 分区读取：`src/core/ecm_ini.h` 排除 `[GUI]` 键，避免 GUI 的进程 `priority=below_normal` 被 CLI 当成数值队列优先级。
- 队列解析与原位替换：`src/core/ecm_worktodo.h`、`src/core/ecm_worktodo.cpp`。阶段、优先级和跳过规则在队列选择器处理；完成行通过同一短时编辑锁改写。
- Stage1 执行与交接：`src/core/ecm_driver.cpp`。保留 Stage1 finished 输出；Prime95 交接仍输出 `ECMSTAGE2=`，Edwards 跨 worker 交接写入共用队列。
- Stage2 读取与 B2 覆盖：`src/core/ecm_cuda_stage2_main.cpp`。Stage2 的 finished、结果、日志和进度仍独立。
- GUI 调度和生成：`src/gui/app.cpp`、`src/gui/worktodo_gen.cpp`。生成器输出 Stage1 前缀；worker 完成当前阶段的任务后自动选择下一阶段程序。
- 打包与脚本：`tools/build/internal/package_stage2.ps1`、`tools/build/internal/package_archive.ps1`、`tools/ecm_worktodo/ecm.py` 和 `ecmcuda.bat`。

正式使用说明维护在 `docs/usage/STAGE1.md`、`docs/usage/STAGE2.md`、`docs/usage/GUI.md`；本文件只记录开发验证。

## 验证记录

- `tools/test/worktodo_sections_test.cpp`：42 项通过，覆盖阶段选择、优先回退、跳过、原位替换、Stage1 行解析，以及 `[GUI] priority` 与队列 `priority` 的分区隔离。
- CPU/OpenCL `ecm.exe`、Stage1 CUDA、Stage2 CUDA（正式脚本 `-HostOnly`，复用校验过的 CUDA 对象）及 GUI 构建成功；版本为 0.41.0。
- `data/experiments/unified_worktodo_smoke/`：Stage1 CPU 单曲线运行后，finished 含原 `ECMSTAGE1=`，队列原位置为 `ECMSTAGE2=`；Stage2 队列与直接读档的 `--plan-only` 均显示 B1=1000、覆盖后 B2=20000。
- `data/experiments/gui_stage_switch_smoke/`：GUI 启动 Stage1、自动切换 Stage2，两个 finished 均记录任务，Stage2 实算 B2=20000 并完成曲线，GUI 正常退出。
- `data/experiments/gui_stage_switch_bom/`：UTF-8 BOM 队列与包含已保存 `[GUI] priority=below_normal` 的 INI 下，GUI 启动 Stage1、切换 Stage2、完成两份 finished，正常退出。
- `data/experiments/auto_stage2_factor_smoke/`：M131、B1=1000 的 Stage1 实际找到因子 263 后，原任务写入 Stage1 finished，队列行被移除，不再生成无法执行的 Stage2 任务。
- Prime95 交接 66 项、GUI 生成器 32 项、GUI 自检 37 项、双实际 worker 23 项通过。配置生成器 `--check` 通过。
- `tools/test/test_worktodo_pipeline.ps1` 在输入准备阶段停止：仓库当前缺少其既有样本 `tools/ecm_worktodo/sorted.csv` 和 `worktodo_p95_stage2.txt`，未进入断言；生成器与 GUI 的逐字节一致性另由 GUI 生成器验收覆盖。

## 后续核验

- 需要在更长的生产队列上核验多个 worker 混合阶段、频繁外部编辑队列及中断恢复；现有验证只覆盖小规模任务。
